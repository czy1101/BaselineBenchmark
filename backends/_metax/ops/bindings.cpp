// This file is modified and supported by the Moonshot AI Team

#include <torch/extension.h>

torch::Tensor index_score_prefill_cuda(
    torch::Tensor idx_q, torch::Tensor index_kv_cache,
    torch::Tensor block_table, torch::Tensor cu_seqlens_q,
    torch::Tensor seq_lens, torch::Tensor prefix_lens, int64_t max_seq_len,
    int64_t num_kv_heads);

torch::Tensor index_score_decode_cuda(
    torch::Tensor idx_q, torch::Tensor index_kv_cache,
    torch::Tensor block_table, torch::Tensor seq_lens, int64_t max_seq_len,
    int64_t num_kv_heads, int64_t decode_query_len);

torch::Tensor topk_prefill_cuda(
    torch::Tensor score, torch::Tensor cu_seqlens_q,
    torch::Tensor prefix_lens, int64_t topk, int64_t init_blocks,
    int64_t local_blocks);

torch::Tensor topk_decode_cuda(
    torch::Tensor score, torch::Tensor seq_lens, int64_t decode_query_len,
    int64_t topk, int64_t init_blocks, int64_t local_blocks);

void sparse_attn_prefill_cuda(
    torch::Tensor q, torch::Tensor kv_cache, torch::Tensor topk_idx,
    torch::Tensor block_table, torch::Tensor cu_seqlens_q,
    torch::Tensor seq_lens, torch::Tensor prefix_lens,
    int64_t num_kv_heads, double sm_scale, torch::Tensor output);

void sparse_attn_decode_cuda(
    torch::Tensor q, torch::Tensor kv_cache, torch::Tensor topk_idx,
    torch::Tensor block_table, torch::Tensor seq_lens,
    int64_t num_kv_heads, double sm_scale, torch::Tensor output,
    int64_t decode_query_len);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("index_score_prefill", &index_score_prefill_cuda,
        "Native MACA MSA prefill index score");
  m.def("index_score_decode", &index_score_decode_cuda,
        "Native MACA MSA decode index score");
  m.def("topk_prefill", &topk_prefill_cuda,
        "Native MACA MSA prefill top-k");
  m.def("topk_decode", &topk_decode_cuda,
        "Native MACA MSA decode top-k");
  m.def("sparse_attn_prefill", &sparse_attn_prefill_cuda,
        "Native MACA MSA prefill sparse attention");
  m.def("sparse_attn_decode", &sparse_attn_decode_cuda,
        "Native MACA MSA decode sparse attention");
}
