#include <torch/extension.h>

#include <ATen/ATen.h>
#include <gcu/gcu_stream.h>
#include <gcu/gcu_utils.h>
#if __has_include(<tops/tops_runtime.h>)
#include <tops/tops_runtime.h>
#else
#include <tops_runtime.h>
#endif

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

namespace {

struct LoadedKernel {
  topsModule_t module = nullptr;
  topsFunction_t function = nullptr;
};

std::mutex module_mutex;
std::unordered_map<std::string, LoadedKernel> module_cache;

void check_tops(topsError_t status, const char* operation) {
  TORCH_CHECK(status == topsSuccess, operation,
              " failed with TopsRuntime status ",
              static_cast<int>(status));
}

LoadedKernel load_kernel(const std::string& fatbin_path) {
  std::lock_guard<std::mutex> guard(module_mutex);
  const auto cached = module_cache.find(fatbin_path);
  if (cached != module_cache.end()) {
    return cached->second;
  }

  LoadedKernel loaded;
  check_tops(topsModuleLoad(&loaded.module, fatbin_path.c_str()),
             "topsModuleLoad");
  check_tops(topsModuleGetFunction(&loaded.function, loaded.module,
                                   "paged_attention_v1_bf16"),
             "topsModuleGetFunction");
  module_cache.emplace(fatbin_path, loaded);
  return loaded;
}

void check_gcu_contiguous(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(torch_gcu::is_gcu(tensor), name, " must be a GCU tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

at::Tensor topscc_decode_bf16_forward(
    const at::Tensor& query, const at::Tensor& key_cache,
    const at::Tensor& value_cache, const at::Tensor& block_tables,
    const at::Tensor& context_lens, int64_t max_context_len,
    double softmax_scale, const std::string& fatbin_path) {
  check_gcu_contiguous(query, "query");
  check_gcu_contiguous(key_cache, "key_cache");
  check_gcu_contiguous(value_cache, "value_cache");
  check_gcu_contiguous(block_tables, "block_tables");
  check_gcu_contiguous(context_lens, "context_lens");

  TORCH_CHECK(query.scalar_type() == at::kBFloat16,
              "query must be bfloat16");
  TORCH_CHECK(key_cache.scalar_type() == at::kBFloat16 &&
                  value_cache.scalar_type() == at::kBFloat16,
              "K/V cache must be bfloat16");
  TORCH_CHECK(block_tables.scalar_type() == at::kInt &&
                  context_lens.scalar_type() == at::kInt,
              "block_tables and context_lens must be int32");
  TORCH_CHECK(query.dim() == 3,
              "query must have shape [num_seqs, num_heads, head_size]");
  TORCH_CHECK(key_cache.dim() == 4 &&
                  value_cache.sizes() == key_cache.sizes(),
              "packed K/V cache must have shape "
              "[num_blocks, num_kv_heads, block_size, head_size]");
  TORCH_CHECK(block_tables.dim() == 2 && context_lens.dim() == 1,
              "invalid sequence metadata rank");
  TORCH_CHECK(block_tables.size(0) == query.size(0) &&
                  context_lens.size(0) == query.size(0),
              "sequence metadata does not match query rows");
  TORCH_CHECK(query.size(2) == key_cache.size(3),
              "query and K/V head size differ");
  TORCH_CHECK(query.size(1) % key_cache.size(1) == 0,
              "query heads must be divisible by KV heads");
  TORCH_CHECK(query.size(2) == 128 || query.size(2) == 256,
              "the upstream S60 TopsCC kernel supports head size 128 or 256");
  TORCH_CHECK(max_context_len > 0, "max_context_len must be positive");

  auto output = at::empty_like(query);
  const LoadedKernel kernel = load_kernel(fatbin_path);
  const topsStream_t stream =
      torch_gcu::getCurrentGCUStream(query.get_device());

  void* output_ptr = output.data_ptr();
  void* query_ptr = query.data_ptr();
  void* key_ptr = key_cache.data_ptr();
  void* value_ptr = value_cache.data_ptr();
  void* block_tables_ptr = block_tables.data_ptr();
  void* context_lens_ptr = context_lens.data_ptr();
  int32_t num_kv_heads = static_cast<int32_t>(key_cache.size(1));
  float scale = static_cast<float>(softmax_scale);
  int32_t block_size = static_cast<int32_t>(key_cache.size(2));
  int32_t maximum_context = static_cast<int32_t>(max_context_len);
  int32_t num_seqs = static_cast<int32_t>(query.size(0));
  int32_t num_heads = static_cast<int32_t>(query.size(1));
  int32_t head_size = static_cast<int32_t>(query.size(2));
  int32_t max_blocks = static_cast<int32_t>(block_tables.size(1));
  int32_t q_stride = static_cast<int32_t>(query.stride(0));
  int32_t kv_block_stride = static_cast<int32_t>(key_cache.stride(0));
  int32_t kv_head_stride = static_cast<int32_t>(key_cache.stride(1));
  int32_t num_blocks = static_cast<int32_t>(key_cache.size(0));
  float softcapping = 0.0f;

  void* arguments[] = {
      &output_ptr,          &query_ptr,       &key_ptr,
      &value_ptr,           &num_kv_heads,    &scale,
      &block_tables_ptr,    &context_lens_ptr,
      &block_size,          &maximum_context, &num_seqs,
      &num_heads,           &head_size,       &max_blocks,
      &q_stride,            &kv_block_stride, &kv_head_stride,
      &num_blocks,          &softcapping,
  };

  // A GCU300 block contains 12 SIPs. The upstream kernel distributes query
  // heads across those SIPs and sequences across blocks.
  const uint32_t grid_x = static_cast<uint32_t>(num_seqs);
  check_tops(topsModuleLaunchKernel(
                 kernel.function, grid_x, 1, 1, 12, 1, 1, 0, stream,
                 arguments, nullptr),
             "topsModuleLaunchKernel");
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("forward", &topscc_decode_bf16_forward,
             "S60 TopsCC dynamic paged-attention decode (BF16)");
}
