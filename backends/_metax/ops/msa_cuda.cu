// This file is modified and supported by the Moonshot AI Team

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <limits>

namespace {

constexpr int kPageSize = 128;
constexpr int kThreads = 128;
constexpr int kMaxTopK = 64;
constexpr int kWarpSize = 32;
constexpr int kNumWarps = kThreads / kWarpSize;

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INT32(x) \
  TORCH_CHECK((x).scalar_type() == at::kInt, #x " must be int32")
#define CHECK_BF16(x) \
  TORCH_CHECK((x).scalar_type() == at::kBFloat16, #x " must be bfloat16")

inline void check_common_tensor(const torch::Tensor& tensor) {
  CHECK_CUDA(tensor);
  CHECK_CONTIGUOUS(tensor);
}

__device__ __forceinline__ float load_bf16(const at::BFloat16* ptr) {
  const auto* raw = reinterpret_cast<const __nv_bfloat16*>(ptr);
  return __bfloat162float(*raw);
}

__device__ __forceinline__ float2 load_bf16x2(const at::BFloat16* ptr) {
  const auto* raw = reinterpret_cast<const __nv_bfloat162*>(ptr);
  return __bfloat1622float2(*raw);
}

__device__ __forceinline__ void store_bf16(at::BFloat16* ptr, float value) {
  auto* raw = reinterpret_cast<__nv_bfloat16*>(ptr);
  *raw = __float2bfloat16_rn(value);
}

__device__ __forceinline__ int find_request(
    int query_id, const int32_t* __restrict__ cu_seqlens_q, int batch) {
  for (int request = 0; request < batch; ++request) {
    if (query_id < cu_seqlens_q[request + 1]) {
      return request;
    }
  }
  return batch - 1;
}

__device__ __forceinline__ float block_reduce_max(float value,
                                                   float* shared) {
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
#pragma unroll
  for (int mask = kWarpSize / 2; mask >= 1; mask >>= 1) {
    value = fmaxf(value,
                  __shfl_xor_sync(0xffffffffu, value, mask, kWarpSize));
  }
  if (lane == 0) {
    shared[warp] = value;
  }
  __syncthreads();

  value = lane < kNumWarps ? shared[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = kNumWarps / 2; mask >= 1; mask >>= 1) {
    value = fmaxf(value,
                  __shfl_xor_sync(0xffffffffu, value, mask, kWarpSize));
  }
  return __shfl_sync(0xffffffffu, value, 0, kWarpSize);
}

__device__ __forceinline__ float block_reduce_sum(float value,
                                                   float* shared) {
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
#pragma unroll
  for (int mask = kWarpSize / 2; mask >= 1; mask >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, mask, kWarpSize);
  }
  if (lane == 0) {
    shared[warp] = value;
  }
  __syncthreads();

  value = lane < kNumWarps ? shared[lane] : 0.0f;
#pragma unroll
  for (int mask = kNumWarps / 2; mask >= 1; mask >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, mask, kWarpSize);
  }
  return __shfl_sync(0xffffffffu, value, 0, kWarpSize);
}

template <bool kDecode>
__global__ void index_score_kernel(
    const at::BFloat16* __restrict__ idx_q,
    const at::BFloat16* __restrict__ index_kv_cache,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ cu_seqlens_q,
    const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ prefix_lens,
    float* __restrict__ score, int total_q, int num_idx_heads, int head_dim,
    int num_pages, int batch, int max_blocks, int table_stride,
    int decode_query_len) {
  const int64_t job = static_cast<int64_t>(blockIdx.x);
  const int block = job % max_blocks;
  const int64_t query_head_job = job / max_blocks;
  const int query_id = query_head_job % total_q;
  const int index_head = query_head_job / total_q;

  int request;
  int query_in_request;
  int query_pos;
  if constexpr (kDecode) {
    request = query_id / decode_query_len;
    query_in_request = query_id - request * decode_query_len;
    query_pos = seq_lens[request] - decode_query_len + query_in_request;
  } else {
    request = find_request(query_id, cu_seqlens_q, batch);
    query_in_request = query_id - cu_seqlens_q[request];
    query_pos = prefix_lens[request] + query_in_request;
  }

  const int seq_len = seq_lens[request];
  const int block_start = block * kPageSize;
  int valid_tokens = min(kPageSize, seq_len - block_start);
  valid_tokens = min(valid_tokens, query_pos - block_start + 1);

  float dot = -FLT_MAX;
  const int token = threadIdx.x;
  if (valid_tokens > 0 && token < valid_tokens) {
    const int page = block_table[request * table_stride + block];
    if (page >= 0 && page < num_pages) {
      const at::BFloat16* q_ptr =
          idx_q + (static_cast<int64_t>(query_id) * num_idx_heads +
                   index_head) *
                      head_dim;
      const at::BFloat16* k_ptr =
          index_kv_cache +
          (static_cast<int64_t>(page) * kPageSize + token) * head_dim;
      dot = 0.0f;
      for (int dim = 0; dim < head_dim; ++dim) {
        dot = fmaf(load_bf16(q_ptr + dim), load_bf16(k_ptr + dim), dot);
      }
    }
  }

  __shared__ float reduction[kThreads];
  const float maximum = block_reduce_max(dot, reduction);
  if (threadIdx.x == 0) {
    const int64_t output_offset =
        (static_cast<int64_t>(index_head) * total_q + query_id) * max_blocks +
        block;
    score[output_offset] = maximum;
  }
}

template <int kQueryTile>
__global__ void index_score_prefill_tiled_kernel(
    const at::BFloat16* __restrict__ idx_q,
    const at::BFloat16* __restrict__ index_kv_cache,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ cu_seqlens_q,
    const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ prefix_lens,
    float* __restrict__ score, int total_q, int num_idx_heads, int head_dim,
    int num_pages, int batch, int max_blocks, int table_stride) {
  constexpr int kThreadsPerToken = 16;
  constexpr int kTokenGroups = kThreads / kThreadsPerToken;
  const int query_tiles = (total_q + kQueryTile - 1) / kQueryTile;
  const int64_t job = static_cast<int64_t>(blockIdx.x);
  const int logical_block = job % max_blocks;
  const int64_t tile_head_job = job / max_blocks;
  const int query_tile = tile_head_job % query_tiles;
  const int index_head = tile_head_job / query_tiles;
  const int query_base = query_tile * kQueryTile;
  const int tid = threadIdx.x;
  const int token_group = tid / kThreadsPerToken;
  const int group_lane = tid % kThreadsPerToken;

  __shared__ float query_shared[kQueryTile * kThreads];
  __shared__ int valid_tokens[kQueryTile];
  __shared__ int pages[kQueryTile];
  __shared__ int common_page;
  __shared__ int pages_match;
  __shared__ float group_max[kQueryTile * kTokenGroups];

  if (tid == 0) {
    common_page = -1;
    pages_match = 1;
#pragma unroll
    for (int tile_q = 0; tile_q < kQueryTile; ++tile_q) {
      const int query_id = query_base + tile_q;
      valid_tokens[tile_q] = 0;
      pages[tile_q] = -1;
      if (query_id < total_q) {
        const int request = find_request(query_id, cu_seqlens_q, batch);
        const int query_in_request = query_id - cu_seqlens_q[request];
        const int query_pos = prefix_lens[request] + query_in_request;
        const int block_start = logical_block * kPageSize;
        int count = min(kPageSize, seq_lens[request] - block_start);
        count = min(count, query_pos - block_start + 1);
        count = max(0, count);
        valid_tokens[tile_q] = count;
        if (count > 0) {
          const int page =
              block_table[request * table_stride + logical_block];
          if (page >= 0 && page < num_pages) {
            pages[tile_q] = page;
            if (common_page < 0) {
              common_page = page;
            } else if (page != common_page) {
              pages_match = 0;
            }
          } else {
            valid_tokens[tile_q] = 0;
          }
        }
      }
    }
  }

  for (int flat = tid; flat < kQueryTile * head_dim; flat += kThreads) {
    const int tile_q = flat / head_dim;
    const int dim = flat - tile_q * head_dim;
    const int query_id = query_base + tile_q;
    float value = 0.0f;
    if (query_id < total_q) {
      const at::BFloat16* q_ptr =
          idx_q +
          (static_cast<int64_t>(query_id) * num_idx_heads + index_head) *
              head_dim;
      value = load_bf16(q_ptr + dim);
    }
    query_shared[tile_q * kThreads + dim] = value;
  }
  __syncthreads();

  float local_max[kQueryTile];
#pragma unroll
  for (int tile_q = 0; tile_q < kQueryTile; ++tile_q) {
    local_max[tile_q] = -FLT_MAX;
  }

  for (int token = token_group; token < kPageSize; token += kTokenGroups) {
    float dots[kQueryTile];
#pragma unroll
    for (int tile_q = 0; tile_q < kQueryTile; ++tile_q) {
      dots[tile_q] = 0.0f;
    }

    if (common_page >= 0 && common_page < num_pages) {
      const at::BFloat16* common_key =
          index_kv_cache +
          (static_cast<int64_t>(common_page) * kPageSize + token) * head_dim;
      for (int dim = 2 * group_lane; dim < head_dim;
           dim += 2 * kThreadsPerToken) {
        float key0;
        float key1 = 0.0f;
        if ((head_dim & 1) == 0 && dim + 1 < head_dim) {
          const float2 key_pair = load_bf16x2(common_key + dim);
          key0 = key_pair.x;
          key1 = key_pair.y;
        } else {
          key0 = load_bf16(common_key + dim);
          if (dim + 1 < head_dim) {
            key1 = load_bf16(common_key + dim + 1);
          }
        }
#pragma unroll
        for (int tile_q = 0; tile_q < kQueryTile; ++tile_q) {
          if (token < valid_tokens[tile_q]) {
            float tile_key0 = key0;
            float tile_key1 = key1;
            if (!pages_match && pages[tile_q] != common_page) {
              const at::BFloat16* key_ptr =
                  index_kv_cache +
                  (static_cast<int64_t>(pages[tile_q]) * kPageSize + token) *
                      head_dim +
                  dim;
              if ((head_dim & 1) == 0 && dim + 1 < head_dim) {
                const float2 key_pair = load_bf16x2(key_ptr);
                tile_key0 = key_pair.x;
                tile_key1 = key_pair.y;
              } else {
                tile_key0 = load_bf16(key_ptr);
                tile_key1 =
                    dim + 1 < head_dim ? load_bf16(key_ptr + 1) : 0.0f;
              }
            }
            dots[tile_q] =
                fmaf(query_shared[tile_q * kThreads + dim], tile_key0,
                     dots[tile_q]);
            if (dim + 1 < head_dim) {
              dots[tile_q] =
                  fmaf(query_shared[tile_q * kThreads + dim + 1], tile_key1,
                       dots[tile_q]);
            }
          }
        }
      }
    }

#pragma unroll
    for (int tile_q = 0; tile_q < kQueryTile; ++tile_q) {
      float dot = dots[tile_q];
#pragma unroll
      for (int offset = kThreadsPerToken / 2; offset >= 1; offset >>= 1) {
        dot += __shfl_down_sync(0xffffffffu, dot, offset,
                                kThreadsPerToken);
      }
      if (group_lane == 0 && token < valid_tokens[tile_q]) {
        local_max[tile_q] = fmaxf(local_max[tile_q], dot);
      }
    }
  }

  if (group_lane == 0) {
#pragma unroll
    for (int tile_q = 0; tile_q < kQueryTile; ++tile_q) {
      group_max[tile_q * kTokenGroups + token_group] = local_max[tile_q];
    }
  }
  __syncthreads();

  if (tid < kQueryTile) {
    float maximum = -FLT_MAX;
#pragma unroll
    for (int group = 0; group < kTokenGroups; ++group) {
      maximum = fmaxf(maximum, group_max[tid * kTokenGroups + group]);
    }
    const int query_id = query_base + tid;
    if (query_id < total_q) {
      const int64_t output_offset =
          (static_cast<int64_t>(index_head) * total_q + query_id) *
              max_blocks +
          logical_block;
      score[output_offset] = maximum;
    }
  }
}

__device__ __forceinline__ void insert_topk(float value, int index,
                                            float* values, int* indices,
                                            int topk) {
  if (topk <= 0 ||
      (indices[topk - 1] >= 0 && value <= values[topk - 1])) {
    return;
  }
  int position = topk - 1;
  while (position > 0 &&
         (indices[position - 1] < 0 || value > values[position - 1])) {
    values[position] = values[position - 1];
    indices[position] = indices[position - 1];
    --position;
  }
  values[position] = value;
  indices[position] = index;
}

template <bool kDecode>
__global__ void topk_kernel(
    const float* __restrict__ score,
    const int32_t* __restrict__ cu_seqlens_q,
    const int32_t* __restrict__ prefix_or_seq_lens,
    int32_t* __restrict__ output, int total_q, int num_idx_heads,
    int max_blocks, int batch, int decode_query_len, int topk,
    int init_blocks, int local_blocks) {
  if (threadIdx.x != 0) {
    return;
  }
  const int64_t job = static_cast<int64_t>(blockIdx.x);
  const int query_id = job % total_q;
  const int index_head = job / total_q;

  int query_pos;
  if constexpr (kDecode) {
    const int request = query_id / decode_query_len;
    const int query_in_request = query_id - request * decode_query_len;
    query_pos = prefix_or_seq_lens[request] - decode_query_len +
                query_in_request;
  } else {
    const int request = find_request(query_id, cu_seqlens_q, batch);
    const int query_in_request = query_id - cu_seqlens_q[request];
    query_pos = prefix_or_seq_lens[request] + query_in_request;
  }

  const int valid_blocks = min(max_blocks, (query_pos + kPageSize) / kPageSize);
  const int selected = min(topk, valid_blocks);
  float best_values[kMaxTopK];
  int best_indices[kMaxTopK];
  for (int slot = 0; slot < topk; ++slot) {
    best_values[slot] = -FLT_MAX;
    best_indices[slot] = -1;
  }

  const int local_start = max(0, valid_blocks - local_blocks);
  const float* row =
      score + (static_cast<int64_t>(index_head) * total_q + query_id) *
                  max_blocks;
  for (int block = 0; block < valid_blocks; ++block) {
    float value = row[block];
    if (block >= local_start) {
      value = 1.0e29f;
    }
    if (block < min(init_blocks, valid_blocks) && value < 1.0e29f) {
      value = 1.0e30f;
    }
    insert_topk(value, block, best_values, best_indices, topk);
  }

  int32_t* output_row =
      output + (static_cast<int64_t>(index_head) * total_q + query_id) * topk;
  for (int slot = 0; slot < topk; ++slot) {
    output_row[slot] = slot < selected ? best_indices[slot] : -1;
  }
}

template <bool kDecode>
__global__ void sparse_attention_kernel(
    const at::BFloat16* __restrict__ q,
    const at::BFloat16* __restrict__ kv_cache,
    const int32_t* __restrict__ topk_idx,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ cu_seqlens_q,
    const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ prefix_lens,
    at::BFloat16* __restrict__ output, int total_q, int num_heads,
    int num_kv_heads, int head_dim, int num_pages, int topk, int batch,
    int table_stride, int decode_query_len, float sm_scale) {
  const int64_t job = static_cast<int64_t>(blockIdx.x);
  const int query_id = job % total_q;
  const int query_head = job / total_q;
  const int group_size = num_heads / num_kv_heads;
  const int kv_head = query_head / group_size;

  int request;
  int query_in_request;
  int query_pos;
  if constexpr (kDecode) {
    request = query_id / decode_query_len;
    query_in_request = query_id - request * decode_query_len;
    query_pos = seq_lens[request] - decode_query_len + query_in_request;
  } else {
    request = find_request(query_id, cu_seqlens_q, batch);
    query_in_request = query_id - cu_seqlens_q[request];
    query_pos = prefix_lens[request] + query_in_request;
  }
  const int max_k = min(query_pos + 1, seq_lens[request]);
  const int tid = threadIdx.x;

  extern __shared__ float shared[];
  float* q_shared = shared;
  float* reduce_shared = q_shared + kThreads;
  float* probability = reduce_shared + kThreads;
  float* scalars = probability + kThreads;

  const at::BFloat16* q_row =
      q + (static_cast<int64_t>(query_id) * num_heads + query_head) * head_dim;
  if (tid < head_dim) {
    q_shared[tid] = load_bf16(q_row + tid);
  } else {
    q_shared[tid] = 0.0f;
  }
  __syncthreads();

  float running_max = -FLT_MAX;
  float running_sum = 0.0f;
  float accumulator = 0.0f;

  const int32_t* topk_row =
      topk_idx + (static_cast<int64_t>(kv_head) * total_q + query_id) * topk;

  for (int slot = 0; slot < topk; ++slot) {
    const int logical_block = topk_row[slot];
    if (logical_block < 0) {
      continue;
    }
    const int block_start = logical_block * kPageSize;
    const int valid_tokens = min(kPageSize, max_k - block_start);
    if (valid_tokens <= 0) {
      continue;
    }
    const int page = block_table[request * table_stride + logical_block];
    if (page < 0 || page >= num_pages) {
      continue;
    }

    float logit = -FLT_MAX;
    if (tid < valid_tokens) {
      const at::BFloat16* key =
          kv_cache +
          ((static_cast<int64_t>(page) * num_kv_heads + kv_head) * kPageSize +
           tid) *
              (2 * head_dim);
      logit = 0.0f;
      for (int dim = 0; dim < head_dim; ++dim) {
        logit = fmaf(q_shared[dim], load_bf16(key + dim), logit);
      }
      logit *= sm_scale;
    }

    const float block_max = block_reduce_max(logit, reduce_shared);
    if (tid == 0) {
      const float new_max = fmaxf(running_max, block_max);
      scalars[0] = new_max;
      scalars[1] = isinf(running_max) ? 0.0f : expf(running_max - new_max);
    }
    __syncthreads();
    const float new_max = scalars[0];
    const float alpha = scalars[1];

    const float weight = tid < valid_tokens ? expf(logit - new_max) : 0.0f;
    probability[tid] = weight;
    const float block_sum = block_reduce_sum(weight, reduce_shared);

    float block_output = 0.0f;
    if (tid < head_dim) {
      for (int token = 0; token < valid_tokens; ++token) {
        const at::BFloat16* value =
            kv_cache +
            ((static_cast<int64_t>(page) * num_kv_heads + kv_head) *
                 kPageSize +
             token) *
                (2 * head_dim) +
            head_dim;
        block_output = fmaf(probability[token], load_bf16(value + tid),
                            block_output);
      }
      accumulator = accumulator * alpha + block_output;
    }
    running_sum = running_sum * alpha + block_sum;
    running_max = new_max;
    __syncthreads();
  }

  if (tid < head_dim) {
    const float normalized = running_sum > 0.0f ? accumulator / running_sum : 0.0f;
    at::BFloat16* output_row =
        output +
        (static_cast<int64_t>(query_id) * num_heads + query_head) * head_dim;
    store_bf16(output_row + tid, normalized);
  }
}

template <bool kDecode, int kHeadTile>
__global__ void sparse_attention_gqa_tiled_kernel(
    const at::BFloat16* __restrict__ q,
    const at::BFloat16* __restrict__ kv_cache,
    const int32_t* __restrict__ topk_idx,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ cu_seqlens_q,
    const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ prefix_lens,
    at::BFloat16* __restrict__ output, int total_q, int num_heads,
    int num_kv_heads, int head_dim, int num_pages, int topk, int batch,
    int table_stride, int decode_query_len, float sm_scale) {
  const int group_size = num_heads / num_kv_heads;
  const int head_tiles = (group_size + kHeadTile - 1) / kHeadTile;
  const int64_t job = static_cast<int64_t>(blockIdx.x);
  const int query_id = job % total_q;
  const int64_t head_job = job / total_q;
  const int head_tile = head_job % head_tiles;
  const int kv_head = head_job / head_tiles;
  const int group_offset = head_tile * kHeadTile;
  const int tile_heads = min(kHeadTile, group_size - group_offset);
  const int query_head_base = kv_head * group_size + group_offset;

  int request;
  int query_in_request;
  int query_pos;
  if constexpr (kDecode) {
    request = query_id / decode_query_len;
    query_in_request = query_id - request * decode_query_len;
    query_pos = seq_lens[request] - decode_query_len + query_in_request;
  } else {
    request = find_request(query_id, cu_seqlens_q, batch);
    query_in_request = query_id - cu_seqlens_q[request];
    query_pos = prefix_lens[request] + query_in_request;
  }
  const int max_k = min(query_pos + 1, seq_lens[request]);
  const int tid = threadIdx.x;

  extern __shared__ float shared[];
  float* query_shared = shared;
  float* probability = query_shared + kHeadTile * kThreads;
  float* reduction = probability + kHeadTile * kThreads;

  float running_max[kHeadTile];
  float running_sum[kHeadTile];
  float accumulator[kHeadTile];
#pragma unroll
  for (int tile_head = 0; tile_head < kHeadTile; ++tile_head) {
    running_max[tile_head] = -FLT_MAX;
    running_sum[tile_head] = 0.0f;
    accumulator[tile_head] = 0.0f;
    if (tile_head < tile_heads && tid < head_dim) {
      const int query_head = query_head_base + tile_head;
      const at::BFloat16* q_row =
          q + (static_cast<int64_t>(query_id) * num_heads + query_head) *
                  head_dim;
      query_shared[tile_head * kThreads + tid] = load_bf16(q_row + tid);
    } else {
      query_shared[tile_head * kThreads + tid] = 0.0f;
    }
  }
  __syncthreads();

  const int32_t* topk_row =
      topk_idx + (static_cast<int64_t>(kv_head) * total_q + query_id) * topk;

  constexpr int kThreadsPerToken = 16;
  constexpr int kTokenGroups = kThreads / kThreadsPerToken;
  const int token_group = tid / kThreadsPerToken;
  const int group_lane = tid % kThreadsPerToken;

  for (int slot = 0; slot < topk; ++slot) {
    const int logical_block = topk_row[slot];
    if (logical_block < 0) {
      continue;
    }
    const int block_start = logical_block * kPageSize;
    const int valid_tokens = min(kPageSize, max_k - block_start);
    if (valid_tokens <= 0) {
      continue;
    }
    const int page = block_table[request * table_stride + logical_block];
    if (page < 0 || page >= num_pages) {
      continue;
    }

    float group_max[kHeadTile];
#pragma unroll
    for (int tile_head = 0; tile_head < kHeadTile; ++tile_head) {
      group_max[tile_head] = -FLT_MAX;
    }

    for (int token = token_group; token < valid_tokens;
         token += kTokenGroups) {
      float logits[kHeadTile];
#pragma unroll
      for (int tile_head = 0; tile_head < kHeadTile; ++tile_head) {
        logits[tile_head] = 0.0f;
      }

      const at::BFloat16* key =
          kv_cache +
          ((static_cast<int64_t>(page) * num_kv_heads + kv_head) * kPageSize +
           token) *
              (2 * head_dim);
      for (int dim = 2 * group_lane; dim < head_dim;
           dim += 2 * kThreadsPerToken) {
        float key0;
        float key1 = 0.0f;
        if (dim + 1 < head_dim) {
          const float2 key_pair = load_bf16x2(key + dim);
          key0 = key_pair.x;
          key1 = key_pair.y;
        } else {
          key0 = load_bf16(key + dim);
        }
#pragma unroll
        for (int tile_head = 0; tile_head < kHeadTile; ++tile_head) {
          if (tile_head < tile_heads) {
            logits[tile_head] =
                fmaf(query_shared[tile_head * kThreads + dim], key0,
                     logits[tile_head]);
            if (dim + 1 < head_dim) {
              logits[tile_head] =
                  fmaf(query_shared[tile_head * kThreads + dim + 1], key1,
                       logits[tile_head]);
            }
          }
        }
      }

#pragma unroll
      for (int tile_head = 0; tile_head < kHeadTile; ++tile_head) {
        if (tile_head < tile_heads) {
          float logit = logits[tile_head];
#pragma unroll
          for (int offset = kThreadsPerToken / 2; offset >= 1;
               offset >>= 1) {
            logit += __shfl_down_sync(0xffffffffu, logit, offset,
                                      kThreadsPerToken);
          }
          if (group_lane == 0) {
            logit *= sm_scale;
            probability[tile_head * kThreads + token] = logit;
            group_max[tile_head] = fmaxf(group_max[tile_head], logit);
          }
        }
      }
    }

    if (group_lane == 0) {
#pragma unroll
      for (int tile_head = 0; tile_head < kHeadTile; ++tile_head) {
        if (tile_head < tile_heads) {
          reduction[tile_head * kTokenGroups + token_group] =
              group_max[tile_head];
        }
      }
    }
    __syncthreads();

    if (tid < tile_heads) {
      float maximum = -FLT_MAX;
#pragma unroll
      for (int group = 0; group < kTokenGroups; ++group) {
        maximum = fmaxf(
            maximum, reduction[tid * kTokenGroups + group]);
      }
      reduction[kHeadTile * kTokenGroups + tid] = maximum;
    }
    __syncthreads();

    float alpha[kHeadTile];
#pragma unroll
    for (int tile_head = 0; tile_head < kHeadTile; ++tile_head) {
      alpha[tile_head] = 0.0f;
      if (tile_head < tile_heads) {
        const float new_max = fmaxf(
            running_max[tile_head],
            reduction[kHeadTile * kTokenGroups + tile_head]);
        alpha[tile_head] = running_max[tile_head] == -FLT_MAX
                               ? 0.0f
                               : __expf(running_max[tile_head] - new_max);
        const float weight =
            tid < valid_tokens
                ? __expf(probability[tile_head * kThreads + tid] - new_max)
                : 0.0f;
        probability[tile_head * kThreads + tid] = weight;
        const float block_sum =
            block_reduce_sum(weight, reduction + tile_head * kNumWarps);
        running_sum[tile_head] =
            running_sum[tile_head] * alpha[tile_head] + block_sum;
        running_max[tile_head] = new_max;
      }
    }

    if (tid < head_dim) {
      float block_output[kHeadTile];
#pragma unroll
      for (int tile_head = 0; tile_head < kHeadTile; ++tile_head) {
        block_output[tile_head] = 0.0f;
      }
      for (int token = 0; token < valid_tokens; ++token) {
        const at::BFloat16* value =
            kv_cache +
            ((static_cast<int64_t>(page) * num_kv_heads + kv_head) *
                 kPageSize +
             token) *
                (2 * head_dim) +
            head_dim;
        const float value_element = load_bf16(value + tid);
#pragma unroll
        for (int tile_head = 0; tile_head < kHeadTile; ++tile_head) {
          if (tile_head < tile_heads) {
            block_output[tile_head] =
                fmaf(probability[tile_head * kThreads + token], value_element,
                     block_output[tile_head]);
          }
        }
      }
#pragma unroll
      for (int tile_head = 0; tile_head < kHeadTile; ++tile_head) {
        if (tile_head < tile_heads) {
          accumulator[tile_head] =
              accumulator[tile_head] * alpha[tile_head] +
              block_output[tile_head];
        }
      }
    }
    __syncthreads();
  }

  if (tid < head_dim) {
#pragma unroll
    for (int tile_head = 0; tile_head < kHeadTile; ++tile_head) {
      if (tile_head < tile_heads) {
        const int query_head = query_head_base + tile_head;
        const float normalized = running_sum[tile_head] > 0.0f
                                     ? __fdividef(accumulator[tile_head],
                                                  running_sum[tile_head])
                                     : 0.0f;
        at::BFloat16* output_row =
            output +
            (static_cast<int64_t>(query_id) * num_heads + query_head) *
                head_dim;
        store_bf16(output_row + tid, normalized);
      }
    }
  }
}

void validate_index_inputs(const torch::Tensor& idx_q,
                           const torch::Tensor& index_kv_cache,
                           const torch::Tensor& block_table,
                           const torch::Tensor& seq_lens,
                           int64_t num_kv_heads) {
  check_common_tensor(idx_q);
  check_common_tensor(index_kv_cache);
  check_common_tensor(block_table);
  check_common_tensor(seq_lens);
  CHECK_BF16(idx_q);
  CHECK_BF16(index_kv_cache);
  CHECK_INT32(block_table);
  CHECK_INT32(seq_lens);
  TORCH_CHECK(idx_q.dim() == 3, "idx_q must be [total_q, heads, dim]");
  TORCH_CHECK(index_kv_cache.dim() == 3,
              "index_kv_cache must be [pages, 128, dim]");
  TORCH_CHECK(block_table.dim() == 2, "block_table must be rank 2");
  TORCH_CHECK(idx_q.size(1) == num_kv_heads,
              "index heads must equal num_kv_heads");
  TORCH_CHECK(index_kv_cache.size(1) == kPageSize,
              "index page size must be 128");
  TORCH_CHECK(idx_q.size(2) == index_kv_cache.size(2),
              "index Q/K head_dim mismatch");
  TORCH_CHECK(idx_q.size(2) <= kThreads,
              "native baseline supports head_dim <= 128");
}

void validate_attention_inputs(const torch::Tensor& q,
                               const torch::Tensor& kv_cache,
                               const torch::Tensor& topk_idx,
                               const torch::Tensor& block_table,
                               const torch::Tensor& seq_lens,
                               const torch::Tensor& output,
                               int64_t num_kv_heads) {
  check_common_tensor(q);
  check_common_tensor(kv_cache);
  check_common_tensor(topk_idx);
  check_common_tensor(block_table);
  check_common_tensor(seq_lens);
  check_common_tensor(output);
  CHECK_BF16(q);
  CHECK_BF16(kv_cache);
  CHECK_BF16(output);
  CHECK_INT32(topk_idx);
  CHECK_INT32(block_table);
  CHECK_INT32(seq_lens);
  TORCH_CHECK(q.dim() == 3, "q must be [total_q, heads, dim]");
  TORCH_CHECK(kv_cache.dim() == 4,
              "kv_cache must be [pages, kv_heads, 128, 2*dim]");
  TORCH_CHECK(topk_idx.dim() == 3,
              "topk_idx must be [kv_heads, total_q, topk]");
  TORCH_CHECK(q.sizes() == output.sizes(), "output shape must equal q shape");
  TORCH_CHECK(q.size(1) % num_kv_heads == 0,
              "query heads must divide by kv heads");
  TORCH_CHECK(kv_cache.size(1) == num_kv_heads,
              "kv_cache num_kv_heads mismatch");
  TORCH_CHECK(kv_cache.size(2) == kPageSize, "KV page size must be 128");
  TORCH_CHECK(kv_cache.size(3) == 2 * q.size(2),
              "KV last dimension must be 2*head_dim");
  TORCH_CHECK(q.size(2) <= kThreads,
              "native baseline supports head_dim <= 128");
  TORCH_CHECK(topk_idx.size(0) == num_kv_heads &&
                  topk_idx.size(1) == q.size(0),
              "topk_idx shape mismatch");
}

}  // namespace

torch::Tensor index_score_prefill_cuda(
    torch::Tensor idx_q, torch::Tensor index_kv_cache,
    torch::Tensor block_table, torch::Tensor cu_seqlens_q,
    torch::Tensor seq_lens, torch::Tensor prefix_lens, int64_t max_seq_len,
    int64_t num_kv_heads) {
  validate_index_inputs(idx_q, index_kv_cache, block_table, seq_lens,
                        num_kv_heads);
  check_common_tensor(cu_seqlens_q);
  check_common_tensor(prefix_lens);
  CHECK_INT32(cu_seqlens_q);
  CHECK_INT32(prefix_lens);
  const int total_q = idx_q.size(0);
  const int num_heads = idx_q.size(1);
  const int head_dim = idx_q.size(2);
  const int batch = seq_lens.size(0);
  const int max_blocks = (max_seq_len + kPageSize - 1) / kPageSize;
  TORCH_CHECK(block_table.size(0) == batch &&
                  block_table.size(1) >= max_blocks,
              "block_table is too small");
  auto score = torch::empty({num_heads, total_q, max_blocks},
                            idx_q.options().dtype(torch::kFloat32));
  constexpr int kQueryTile = 8;
  const int query_tiles = (total_q + kQueryTile - 1) / kQueryTile;
  const int64_t jobs =
      static_cast<int64_t>(num_heads) * query_tiles * max_blocks;
  auto stream = at::cuda::getCurrentCUDAStream();
  index_score_prefill_tiled_kernel<kQueryTile><<<jobs, kThreads, 0, stream>>>(
      idx_q.data_ptr<at::BFloat16>(),
      index_kv_cache.data_ptr<at::BFloat16>(),
      block_table.data_ptr<int32_t>(), cu_seqlens_q.data_ptr<int32_t>(),
      seq_lens.data_ptr<int32_t>(), prefix_lens.data_ptr<int32_t>(),
      score.data_ptr<float>(), total_q, num_heads, head_dim,
      index_kv_cache.size(0), batch, max_blocks, block_table.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return score;
}

torch::Tensor index_score_decode_cuda(
    torch::Tensor idx_q, torch::Tensor index_kv_cache,
    torch::Tensor block_table, torch::Tensor seq_lens, int64_t max_seq_len,
    int64_t num_kv_heads, int64_t decode_query_len) {
  validate_index_inputs(idx_q, index_kv_cache, block_table, seq_lens,
                        num_kv_heads);
  TORCH_CHECK(decode_query_len > 0, "decode_query_len must be positive");
  const int total_q = idx_q.size(0);
  const int num_heads = idx_q.size(1);
  const int head_dim = idx_q.size(2);
  const int batch = seq_lens.size(0);
  TORCH_CHECK(total_q == batch * decode_query_len,
              "decode query shape mismatch");
  const int max_blocks = (max_seq_len + kPageSize - 1) / kPageSize;
  TORCH_CHECK(block_table.size(0) == batch &&
                  block_table.size(1) >= max_blocks,
              "block_table is too small");
  auto score = torch::empty({num_heads, total_q, max_blocks},
                            idx_q.options().dtype(torch::kFloat32));
  const int64_t jobs =
      static_cast<int64_t>(num_heads) * total_q * max_blocks;
  auto stream = at::cuda::getCurrentCUDAStream();
  index_score_kernel<true><<<jobs, kThreads, 0, stream>>>(
      idx_q.data_ptr<at::BFloat16>(),
      index_kv_cache.data_ptr<at::BFloat16>(),
      block_table.data_ptr<int32_t>(), nullptr, seq_lens.data_ptr<int32_t>(),
      nullptr, score.data_ptr<float>(), total_q, num_heads, head_dim,
      index_kv_cache.size(0), batch, max_blocks, block_table.size(1),
      decode_query_len);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return score;
}

torch::Tensor topk_prefill_cuda(
    torch::Tensor score, torch::Tensor cu_seqlens_q,
    torch::Tensor prefix_lens, int64_t topk, int64_t init_blocks,
    int64_t local_blocks) {
  check_common_tensor(score);
  check_common_tensor(cu_seqlens_q);
  check_common_tensor(prefix_lens);
  TORCH_CHECK(score.scalar_type() == at::kFloat, "score must be float32");
  CHECK_INT32(cu_seqlens_q);
  CHECK_INT32(prefix_lens);
  TORCH_CHECK(score.dim() == 3, "score must be rank 3");
  TORCH_CHECK(topk > 0 && topk <= kMaxTopK, "topk must be in [1, 64]");
  const int num_heads = score.size(0);
  const int total_q = score.size(1);
  const int max_blocks = score.size(2);
  const int batch = cu_seqlens_q.size(0) - 1;
  auto output = torch::empty({num_heads, total_q, topk},
                             score.options().dtype(torch::kInt32));
  const int64_t jobs = static_cast<int64_t>(num_heads) * total_q;
  auto stream = at::cuda::getCurrentCUDAStream();
  topk_kernel<false><<<jobs, 1, 0, stream>>>(
      score.data_ptr<float>(), cu_seqlens_q.data_ptr<int32_t>(),
      prefix_lens.data_ptr<int32_t>(), output.data_ptr<int32_t>(), total_q,
      num_heads, max_blocks, batch, 1, topk, init_blocks, local_blocks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor topk_decode_cuda(
    torch::Tensor score, torch::Tensor seq_lens, int64_t decode_query_len,
    int64_t topk, int64_t init_blocks, int64_t local_blocks) {
  check_common_tensor(score);
  check_common_tensor(seq_lens);
  TORCH_CHECK(score.scalar_type() == at::kFloat, "score must be float32");
  CHECK_INT32(seq_lens);
  TORCH_CHECK(score.dim() == 3, "score must be rank 3");
  TORCH_CHECK(topk > 0 && topk <= kMaxTopK, "topk must be in [1, 64]");
  const int num_heads = score.size(0);
  const int total_q = score.size(1);
  const int max_blocks = score.size(2);
  const int batch = seq_lens.size(0);
  TORCH_CHECK(total_q == batch * decode_query_len,
              "decode query shape mismatch");
  auto output = torch::empty({num_heads, total_q, topk},
                             score.options().dtype(torch::kInt32));
  const int64_t jobs = static_cast<int64_t>(num_heads) * total_q;
  auto stream = at::cuda::getCurrentCUDAStream();
  topk_kernel<true><<<jobs, 1, 0, stream>>>(
      score.data_ptr<float>(), nullptr, seq_lens.data_ptr<int32_t>(),
      output.data_ptr<int32_t>(), total_q, num_heads, max_blocks, batch,
      decode_query_len, topk, init_blocks, local_blocks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

void sparse_attn_prefill_cuda(
    torch::Tensor q, torch::Tensor kv_cache, torch::Tensor topk_idx,
    torch::Tensor block_table, torch::Tensor cu_seqlens_q,
    torch::Tensor seq_lens, torch::Tensor prefix_lens,
    int64_t num_kv_heads, double sm_scale, torch::Tensor output) {
  validate_attention_inputs(q, kv_cache, topk_idx, block_table, seq_lens,
                            output, num_kv_heads);
  check_common_tensor(cu_seqlens_q);
  check_common_tensor(prefix_lens);
  CHECK_INT32(cu_seqlens_q);
  CHECK_INT32(prefix_lens);
  const int total_q = q.size(0);
  const int num_heads = q.size(1);
  const int head_dim = q.size(2);
  const int batch = seq_lens.size(0);
  const int topk = topk_idx.size(2);
  constexpr int kHeadTile = 8;
  const int group_size = num_heads / num_kv_heads;
  const int head_tiles = (group_size + kHeadTile - 1) / kHeadTile;
  const int64_t jobs =
      static_cast<int64_t>(num_kv_heads) * head_tiles * total_q;
  const size_t shared_bytes =
      (2 * kHeadTile * kThreads + kThreads) * sizeof(float);
  auto stream = at::cuda::getCurrentCUDAStream();
  sparse_attention_gqa_tiled_kernel<false, kHeadTile>
      <<<jobs, kThreads, shared_bytes, stream>>>(
      q.data_ptr<at::BFloat16>(), kv_cache.data_ptr<at::BFloat16>(),
      topk_idx.data_ptr<int32_t>(), block_table.data_ptr<int32_t>(),
      cu_seqlens_q.data_ptr<int32_t>(), seq_lens.data_ptr<int32_t>(),
      prefix_lens.data_ptr<int32_t>(), output.data_ptr<at::BFloat16>(),
      total_q, num_heads, num_kv_heads, head_dim, kv_cache.size(0), topk,
      batch, block_table.size(1), 1, static_cast<float>(sm_scale));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sparse_attn_decode_cuda(
    torch::Tensor q, torch::Tensor kv_cache, torch::Tensor topk_idx,
    torch::Tensor block_table, torch::Tensor seq_lens,
    int64_t num_kv_heads, double sm_scale, torch::Tensor output,
    int64_t decode_query_len) {
  validate_attention_inputs(q, kv_cache, topk_idx, block_table, seq_lens,
                            output, num_kv_heads);
  const int total_q = q.size(0);
  const int num_heads = q.size(1);
  const int head_dim = q.size(2);
  const int batch = seq_lens.size(0);
  TORCH_CHECK(total_q == batch * decode_query_len,
              "decode query shape mismatch");
  const int topk = topk_idx.size(2);
  constexpr int kHeadTile = 8;
  const int group_size = num_heads / num_kv_heads;
  const int head_tiles = (group_size + kHeadTile - 1) / kHeadTile;
  const int64_t jobs =
      static_cast<int64_t>(num_kv_heads) * head_tiles * total_q;
  const size_t shared_bytes =
      (2 * kHeadTile * kThreads + kThreads) * sizeof(float);
  auto stream = at::cuda::getCurrentCUDAStream();
  sparse_attention_gqa_tiled_kernel<true, kHeadTile>
      <<<jobs, kThreads, shared_bytes, stream>>>(
      q.data_ptr<at::BFloat16>(), kv_cache.data_ptr<at::BFloat16>(),
      topk_idx.data_ptr<int32_t>(), block_table.data_ptr<int32_t>(), nullptr,
      seq_lens.data_ptr<int32_t>(), nullptr, output.data_ptr<at::BFloat16>(),
      total_q, num_heads, num_kv_heads, head_dim, kv_cache.size(0), topk,
      batch, block_table.size(1), decode_query_len,
      static_cast<float>(sm_scale));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
