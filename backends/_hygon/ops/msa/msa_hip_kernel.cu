#include <torch/extension.h>

#include <c10/hip/HIPException.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#define HIPBLAS_V2
#include <hipblas/hipblas.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

constexpr int kBlock = 128;
constexpr int kDim = 128;
constexpr int kThreads = 256;
constexpr int kWave = 64;
constexpr int kMaxTopK = 16;

#define HIPBLAS_CHECK(expr)                                                   \
  do {                                                                        \
    const hipblasStatus_t status = (expr);                                    \
    TORCH_CHECK(status == HIPBLAS_STATUS_SUCCESS, "hipBLAS failure: ",       \
                static_cast<int>(status), " at ", __FILE__, ":", __LINE__); \
  } while (0)

struct BlasHandle {
  hipblasHandle_t value{};
  BlasHandle() { HIPBLAS_CHECK(hipblasCreate(&value)); }
  ~BlasHandle() { hipblasDestroy(value); }
};

hipblasHandle_t current_handle(hipStream_t stream) {
  static thread_local BlasHandle handle;
  HIPBLAS_CHECK(hipblasSetStream(handle.value, stream));
  return handle.value;
}

__device__ __forceinline__ float bf16_to_float(at::BFloat16 x) {
  return static_cast<float>(x);
}

__device__ __forceinline__ at::BFloat16 float_to_bf16(float x) {
  return static_cast<at::BFloat16>(x);
}

__device__ __forceinline__ float wave_max(float x) {
  x = fmaxf(x, __shfl_down(x, 32, kWave));
  x = fmaxf(x, __shfl_down(x, 16, kWave));
  x = fmaxf(x, __shfl_down(x, 8, kWave));
  x = fmaxf(x, __shfl_down(x, 4, kWave));
  x = fmaxf(x, __shfl_down(x, 2, kWave));
  x = fmaxf(x, __shfl_down(x, 1, kWave));
  return x;
}

__device__ __forceinline__ float wave_sum(float x) {
  x += __shfl_down(x, 32, kWave);
  x += __shfl_down(x, 16, kWave);
  x += __shfl_down(x, 8, kWave);
  x += __shfl_down(x, 4, kWave);
  x += __shfl_down(x, 2, kWave);
  x += __shfl_down(x, 1, kWave);
  return x;
}

__device__ float block_max(float x, float* scratch) {
  const int lane = threadIdx.x & 63;
  const int wave = threadIdx.x >> 6;
  const int waves = (blockDim.x + kWave - 1) / kWave;
  x = wave_max(x);
  if (lane == 0) scratch[wave] = x;
  __syncthreads();
  // Some kernels use 128 threads (two wave64s), while the GEMM/TopK paths
  // use 256 threads (four wave64s).  Reading all four scratch slots for a
  // two-wave block consumes uninitialised LDS and intermittently corrupts a
  // page maximum.  Reduce exactly the number of waves in this CTA.
  float y = threadIdx.x < waves ? scratch[lane] : -INFINITY;
  if (wave == 0) y = wave_max(y);
  if (threadIdx.x == 0) scratch[0] = y;
  __syncthreads();
  return scratch[0];
}

__device__ float block_sum(float x, float* scratch) {
  const int lane = threadIdx.x & 63;
  const int wave = threadIdx.x >> 6;
  const int waves = (blockDim.x + kWave - 1) / kWave;
  x = wave_sum(x);
  if (lane == 0) scratch[wave] = x;
  __syncthreads();
  float y = threadIdx.x < waves ? scratch[lane] : 0.0f;
  if (wave == 0) y = wave_sum(y);
  if (threadIdx.x == 0) scratch[0] = y;
  __syncthreads();
  return scratch[0];
}

__device__ __forceinline__ bool better(float x, int id, float y, int other) {
  return x > y || (x == y && id < other);
}

__device__ void block_best(float& value, int& id, float* score_scratch,
                           int* id_scratch) {
  const int lane = threadIdx.x & 63;
  const int wave = threadIdx.x >> 6;
  for (int delta = 32; delta > 0; delta >>= 1) {
    const float other_value = __shfl_down(value, delta, kWave);
    const int other_id = __shfl_down(id, delta, kWave);
    if (better(other_value, other_id, value, id)) {
      value = other_value;
      id = other_id;
    }
  }
  if (lane == 0) {
    score_scratch[wave] = value;
    id_scratch[wave] = id;
  }
  __syncthreads();
  value = threadIdx.x < 4 ? score_scratch[lane] : -INFINITY;
  id = threadIdx.x < 4 ? id_scratch[lane] : INT_MAX;
  if (wave == 0) {
    for (int delta = 32; delta > 0; delta >>= 1) {
      const float other_value = __shfl_down(value, delta, kWave);
      const int other_id = __shfl_down(id, delta, kWave);
      if (better(other_value, other_id, value, id)) {
        value = other_value;
        id = other_id;
      }
    }
  }
  if (threadIdx.x == 0) {
    score_scratch[0] = value;
    id_scratch[0] = id;
  }
  __syncthreads();
  value = score_scratch[0];
  id = id_scratch[0];
}

__device__ int request_for_q(const int32_t* cu, int batch, int qid) {
  int req = 0;
  while (req + 1 < batch && qid >= cu[req + 1]) ++req;
  return req;
}

__global__ void pack_index_k_kernel(
    const at::BFloat16* __restrict__ cache,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ seq_lens, at::BFloat16* __restrict__ kpack,
    int B, int H, int max_seq, int max_pages) {
  const int64_t total = static_cast<int64_t>(B) * H * max_seq * kDim;
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int d = i % kDim;
    int64_t z = i / kDim;
    const int token = z % max_seq;
    z /= max_seq;
    const int h = z % H;
    const int b = z / H;
    const bool valid = token < seq_lens[b];
    const int logical_page = token / kBlock;
    const int offset = token & (kBlock - 1);
    int page = 0;
    if (valid && logical_page < max_pages)
      page = block_table[b * max_pages + logical_page];
    const int64_t src = (static_cast<int64_t>(page) * kBlock + offset) * kDim + d;
    kpack[i] = valid ? cache[src] : float_to_bf16(0.0f);
  }
}

template <bool Decode>
__global__ void pack_index_q_kernel(
    const at::BFloat16* __restrict__ q, const int32_t* __restrict__ cu,
    const int32_t* __restrict__ seq_lens, at::BFloat16* __restrict__ qpack,
    int B, int H, int total_q, int q0, int tile, int decode_len) {
  const int64_t total = static_cast<int64_t>(B) * H * tile * kDim;
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int d = i % kDim;
    int64_t z = i / kDim;
    const int tq = z % tile;
    z /= tile;
    const int h = z % H;
    const int b = z / H;
    const int local = q0 + tq;
    const int qid = Decode ? b * decode_len + local : cu[b] + local;
    const int qlen = Decode ? decode_len : cu[b + 1] - cu[b];
    const bool valid = local < qlen && qid < total_q;
    qpack[i] = valid ? q[(static_cast<int64_t>(qid) * H + h) * kDim + d]
                     : float_to_bf16(0.0f);
  }
}

template <bool Decode>
__global__ void reduce_index_score_kernel(
    const float* __restrict__ token_score, float* __restrict__ block_score,
    const int32_t* __restrict__ cu, const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ prefix_lens, int B, int H, int total_q,
    int max_seq, int score_stride, int q0, int tile, int decode_len,
    int init_blocks, int local_blocks, int64_t score_s0, int64_t score_s1,
    int64_t score_s2) {
  __shared__ float scratch[4];
  const int max_blocks = (max_seq + kBlock - 1) / kBlock;
  const int64_t work = static_cast<int64_t>(B) * H * tile * score_stride;
  for (int64_t item = blockIdx.x; item < work; item += gridDim.x) {
    int64_t z = item;
    const int block = z % score_stride;
    z /= score_stride;
    const int tq = z % tile;
    z /= tile;
    const int h = z % H;
    const int b = z / H;
    const int local = q0 + tq;
    const int qlen = Decode ? decode_len : cu[b + 1] - cu[b];
    const int qid = Decode ? b * decode_len + local : cu[b] + local;
    const int qabs = Decode ? seq_lens[b] - decode_len + local
                            : prefix_lens[b] + local;
    float value = -INFINITY;
    if (local < qlen && block < max_blocks) {
      const int token = block * kBlock + threadIdx.x;
      if (threadIdx.x < kBlock && token < seq_lens[b] && token <= qabs) {
        const int batch_index = b * H + h;
        value = token_score[
            ((static_cast<int64_t>(batch_index) * tile + tq) * max_seq) + token];
      }
      value = block_max(value, scratch);
    }
    if constexpr (Decode) {
      const int valid_blocks = (qabs + kBlock) / kBlock;
      if (block < init_blocks && block < valid_blocks) value = 1.0e30f;
      if (block >= max(0, valid_blocks - local_blocks) && block < valid_blocks)
        value = 1.0e29f;
    } else if (block >= (qabs + kBlock) / kBlock) {
      value = -INFINITY;
    }
    if (threadIdx.x == 0 && local < qlen && qid < total_q) {
      block_score[static_cast<int64_t>(h) * score_s0 +
                  static_cast<int64_t>(qid) * score_s1 + block * score_s2] = value;
    }
    __syncthreads();
  }
}

// Direct score path for decode and short ragged prefill.  The packed hipBLAS
// path is efficient for long prefill, but its large B*H workspace is wasteful
// for only a handful of query rows.  Mapping one CTA to one
// (request, head, query, logical-page) also makes the causal tail page exact.
template <bool Decode>
__global__ void direct_index_score_kernel(
    const at::BFloat16* __restrict__ q,
    const at::BFloat16* __restrict__ cache,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ cu,
    const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ prefix_lens,
    float* __restrict__ score, int B, int H, int total_q, int max_query,
    int max_blocks, int pages_per_req, int decode_len, int init_blocks,
    int local_blocks, int64_t score_s0, int64_t score_s1,
    int64_t score_s2) {
  __shared__ float scratch[4];
  const int64_t row = blockIdx.x;
  int64_t z = row;
  const int logical = z % max_blocks;
  z /= max_blocks;
  const int local = z % max_query;
  z /= max_query;
  const int h = z % H;
  const int b = z / H;

  const int qlen = Decode ? decode_len : cu[b + 1] - cu[b];
  if (local >= qlen) return;
  const int qid = Decode ? b * decode_len + local : cu[b] + local;
  if (qid >= total_q) return;
  const int qabs = Decode ? seq_lens[b] - decode_len + local
                          : prefix_lens[b] + local;
  const int valid_blocks = min(max_blocks, (qabs + kBlock) / kBlock);

  float value = -INFINITY;
  if (logical < valid_blocks) {
    const int token = logical * kBlock + threadIdx.x;
    if (threadIdx.x < kBlock && token < seq_lens[b] && token <= qabs) {
      const int page = block_table[b * pages_per_req + logical];
      const int offset = token & (kBlock - 1);
      const int64_t q_base = (static_cast<int64_t>(qid) * H + h) * kDim;
      const int64_t k_base =
          (static_cast<int64_t>(page) * kBlock + offset) * kDim;
      float dot = 0.0f;
#pragma unroll 4
      for (int d = 0; d < kDim; ++d)
        dot += bf16_to_float(q[q_base + d]) * bf16_to_float(cache[k_base + d]);
      value = dot;
    }
    value = block_max(value, scratch);
  }
  if constexpr (Decode) {
    if (logical < init_blocks && logical < valid_blocks) value = 1.0e30f;
    if (logical >= max(0, valid_blocks - local_blocks) &&
        logical < valid_blocks)
      value = 1.0e29f;
  }
  if (threadIdx.x == 0) {
    score[static_cast<int64_t>(h) * score_s0 +
          static_cast<int64_t>(qid) * score_s1 +
          static_cast<int64_t>(logical) * score_s2] = value;
  }
}

template <bool Decode>
__global__ void topk_kernel(
    const float* __restrict__ score, int32_t* __restrict__ out,
    const int32_t* __restrict__ cu, const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ prefix_lens, int B, int H, int total_q,
    int score_stride, int topk, int init_blocks, int local_blocks,
    int decode_len, int sort_size, int64_t score_s0, int64_t score_s1,
    int64_t score_s2, int64_t out_s0, int64_t out_s1, int64_t out_s2) {
  (void)sort_size;
  const int row = blockIdx.x;
  const int qid = row % total_q;
  const int h = row / total_q;
  int req, local, qabs;
  if constexpr (Decode) {
    req = qid / decode_len;
    local = qid - req * decode_len;
    qabs = seq_lens[req] - decode_len + local;
  } else {
    req = request_for_q(cu, B, qid);
    local = qid - cu[req];
    qabs = prefix_lens[req] + local;
  }
  const int valid_blocks = min(score_stride, (qabs + kBlock) / kBlock);
  if (valid_blocks <= topk) {
    for (int i = threadIdx.x; i < topk; i += blockDim.x) {
      out[static_cast<int64_t>(h) * out_s0 +
          static_cast<int64_t>(qid) * out_s1 + i * out_s2] =
          i < valid_blocks ? i : -1;
    }
    return;
  }
  __shared__ float score_scratch[4];
  __shared__ int id_scratch[4];
  __shared__ int selected[kMaxTopK];
  for (int rank = 0; rank < topk; ++rank) {
    float best_value = -INFINITY;
    int best_id = INT_MAX;
    for (int i = threadIdx.x; i < valid_blocks; i += blockDim.x) {
      bool used = false;
      for (int previous = 0; previous < rank; ++previous)
        used |= selected[previous] == i;
      if (used) continue;
      float x = score[static_cast<int64_t>(h) * score_s0 +
                      static_cast<int64_t>(qid) * score_s1 + i * score_s2];
      if (isnan(x)) x = -INFINITY;
      if (i < init_blocks) x = 1.0e30f;
      if (i >= max(0, valid_blocks - local_blocks)) x = 1.0e29f;
      if (better(x, i, best_value, best_id)) {
        best_value = x;
        best_id = i;
      }
    }
    block_best(best_value, best_id, score_scratch, id_scratch);
    if (threadIdx.x == 0) selected[rank] = best_id;
    __syncthreads();
  }
  // The attention result is invariant to selected-block order.  Returning
  // logical ids in ascending order matches FlagAttention's fast selector and
  // gives the paged gather monotonically increasing cache addresses.
  if (threadIdx.x == 0) {
    for (int i = 1; i < topk; ++i) {
      const int key = selected[i];
      int j = i - 1;
      while (j >= 0 && selected[j] > key) {
        selected[j + 1] = selected[j];
        --j;
      }
      selected[j + 1] = key;
    }
  }
  __syncthreads();
  for (int i = threadIdx.x; i < topk; i += blockDim.x) {
    out[static_cast<int64_t>(h) * out_s0 +
        static_cast<int64_t>(qid) * out_s1 + i * out_s2] =
        i < valid_blocks ? selected[i] : -1;
  }
}

template <bool Decode>
__global__ void pack_sparse_kv_kernel(
    const at::BFloat16* __restrict__ kv,
    const int32_t* __restrict__ topk_idx,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ cu, const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ prefix_lens, at::BFloat16* __restrict__ kpack,
    at::BFloat16* __restrict__ vpack, int B, int total_q, int H, int pages_per_req,
    int topk, int q0, int tile, int decode_len, int64_t topk_s0,
    int64_t topk_s1, int64_t topk_s2) {
  const int N = topk * kBlock;
  const int64_t total = static_cast<int64_t>(tile) * H * N * kDim;
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int d = i % kDim;
    int64_t z = i / kDim;
    const int n = z % N;
    z /= N;
    const int h = z % H;
    const int tq = z / H;
    const int qid = q0 + tq;
    int req, local, qabs;
    if constexpr (Decode) {
      req = qid / decode_len;
      local = qid - req * decode_len;
      qabs = seq_lens[req] - decode_len + local;
    } else {
      req = request_for_q(cu, B, qid);
      local = qid - cu[req];
      qabs = prefix_lens[req] + local;
    }
    const int slot = n / kBlock;
    const int offset = n & (kBlock - 1);
    const int logical = topk_idx[static_cast<int64_t>(h) * topk_s0 +
                                 static_cast<int64_t>(qid) * topk_s1 +
                                 static_cast<int64_t>(slot) * topk_s2];
    const int token = logical * kBlock + offset;
    const bool valid = logical >= 0 && logical < pages_per_req &&
                       token < seq_lens[req] && token <= qabs;
    int page = 0;
    if (valid) page = block_table[req * pages_per_req + logical];
    const int64_t dst = ((static_cast<int64_t>(tq) * H + h) * N + n) * kDim + d;
    const int64_t src = (((static_cast<int64_t>(page) * H + h) * kBlock + offset) *
                         (2 * kDim)) + d;
    kpack[dst] = valid ? kv[src] : float_to_bf16(0.0f);
    vpack[dst] = valid ? kv[src + kDim] : float_to_bf16(0.0f);
  }
}

template <bool Decode>
__global__ void sparse_softmax_kernel(
    const float* __restrict__ scores, at::BFloat16* __restrict__ probs,
    const int32_t* __restrict__ topk_idx, const int32_t* __restrict__ cu,
    const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ prefix_lens, int B, int total_q, int H, int G,
    int topk, int q0, int tile, int decode_len, int64_t topk_s0,
    int64_t topk_s1, int64_t topk_s2) {
  const int row = blockIdx.x;
  const int g = row % G;
  int m = row / G;
  const int h = m % H;
  const int tq = m / H;
  const int qid = q0 + tq;
  int req, local, qabs;
  if constexpr (Decode) {
    req = qid / decode_len;
    local = qid - req * decode_len;
    qabs = seq_lens[req] - decode_len + local;
  } else {
    req = request_for_q(cu, B, qid);
    local = qid - cu[req];
    qabs = prefix_lens[req] + local;
  }
  const int N = topk * kBlock;
  const float* score_row = scores + static_cast<int64_t>(row) * N;
  at::BFloat16* prob_row = probs + static_cast<int64_t>(row) * N;
  __shared__ float scratch[4];
  float local_max = -INFINITY;
  for (int n = threadIdx.x; n < N; n += blockDim.x) {
    const int slot = n / kBlock;
    const int offset = n & (kBlock - 1);
    const int logical = topk_idx[static_cast<int64_t>(h) * topk_s0 +
                                 static_cast<int64_t>(qid) * topk_s1 +
                                 static_cast<int64_t>(slot) * topk_s2];
    const int token = logical * kBlock + offset;
    const bool valid = logical >= 0 && token < seq_lens[req] && token <= qabs;
    local_max = fmaxf(local_max, valid ? score_row[n] : -INFINITY);
  }
  const float max_value = block_max(local_max, scratch);
  const bool any = isfinite(max_value);
  float local_sum = 0.0f;
  for (int n = threadIdx.x; n < N; n += blockDim.x) {
    const int slot = n / kBlock;
    const int offset = n & (kBlock - 1);
    const int logical = topk_idx[static_cast<int64_t>(h) * topk_s0 +
                                 static_cast<int64_t>(qid) * topk_s1 +
                                 static_cast<int64_t>(slot) * topk_s2];
    const int token = logical * kBlock + offset;
    const bool valid = logical >= 0 && token < seq_lens[req] && token <= qabs;
    const float p = any && valid ? __expf(score_row[n] - max_value) : 0.0f;
    prob_row[n] = float_to_bf16(p);
    local_sum += p;
  }
  const float sum = block_sum(local_sum, scratch);
  const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
  for (int n = threadIdx.x; n < N; n += blockDim.x)
    prob_row[n] = float_to_bf16(bf16_to_float(prob_row[n]) * inv);
}

void index_gemm(hipblasHandle_t handle, const at::BFloat16* k,
                const at::BFloat16* q, float* score, int max_seq, int tile,
                int batch, int q_batch_stride, int score_batch_stride) {
  const float alpha = 1.0f, beta = 0.0f;
  HIPBLAS_CHECK(hipblasGemmStridedBatchedEx(
      handle, HIPBLAS_OP_T, HIPBLAS_OP_N, max_seq, tile, kDim, &alpha, k,
      HIP_R_16BF, kDim, static_cast<hipblasStride>(max_seq) * kDim, q,
      HIP_R_16BF, kDim, static_cast<hipblasStride>(q_batch_stride) * kDim, &beta, score,
      HIP_R_32F, max_seq, static_cast<hipblasStride>(score_batch_stride) * max_seq, batch,
      HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
}

void sparse_qk_gemm(hipblasHandle_t handle, const at::BFloat16* k,
                    const at::BFloat16* q, float* score, int N, int G,
                    int batch, float scale) {
  const float beta = 0.0f;
  HIPBLAS_CHECK(hipblasGemmStridedBatchedEx(
      handle, HIPBLAS_OP_T, HIPBLAS_OP_N, N, G, kDim, &scale, k, HIP_R_16BF,
      kDim, static_cast<hipblasStride>(N) * kDim, q, HIP_R_16BF, kDim,
      static_cast<hipblasStride>(G) * kDim, &beta, score, HIP_R_32F, N,
      static_cast<hipblasStride>(G) * N, batch, HIPBLAS_COMPUTE_32F,
      HIPBLAS_GEMM_DEFAULT));
}

void sparse_pv_gemm(hipblasHandle_t handle, const at::BFloat16* v,
                    const at::BFloat16* probs, at::BFloat16* out, int N, int G,
                    int batch) {
  const float alpha = 1.0f, beta = 0.0f;
  HIPBLAS_CHECK(hipblasGemmStridedBatchedEx(
      handle, HIPBLAS_OP_N, HIPBLAS_OP_N, kDim, G, N, &alpha, v, HIP_R_16BF,
      kDim, static_cast<hipblasStride>(N) * kDim, probs, HIP_R_16BF, N,
      static_cast<hipblasStride>(G) * N, &beta, out, HIP_R_16BF, kDim,
      static_cast<hipblasStride>(G) * kDim, batch, HIPBLAS_COMPUTE_32F,
      HIPBLAS_GEMM_DEFAULT));
}

int next_pow2(int x) {
  int p = 1;
  while (p < x) p <<= 1;
  return p;
}

void validate_index(const torch::Tensor& q, const torch::Tensor& cache,
                    const torch::Tensor& table, const torch::Tensor& lens,
                    int heads) {
  TORCH_CHECK(q.is_cuda() && cache.is_cuda() && q.scalar_type() == torch::kBFloat16 &&
                  cache.scalar_type() == torch::kBFloat16,
              "index q/cache must be HIP BF16");
  TORCH_CHECK(q.is_contiguous() && cache.is_contiguous(),
              "index q/cache must be contiguous");
  TORCH_CHECK(q.dim() == 3 && q.size(1) == heads && q.size(2) == kDim,
              "invalid index q shape");
  TORCH_CHECK(cache.dim() == 3 && cache.size(1) == kBlock && cache.size(2) == kDim,
              "invalid index cache shape");
  TORCH_CHECK(table.is_cuda() && table.scalar_type() == torch::kInt &&
                  table.is_contiguous() && table.dim() == 2,
              "block_table must be contiguous HIP int32");
  TORCH_CHECK(lens.is_cuda() && lens.scalar_type() == torch::kInt &&
                  lens.is_contiguous(),
              "sequence metadata must be contiguous HIP int32");
}

template <bool Decode>
torch::Tensor index_score_impl(
    const torch::Tensor& q, const torch::Tensor& cache,
    const torch::Tensor& table, const torch::Tensor& cu,
    const torch::Tensor& seq_lens, const torch::Tensor& prefix_lens,
    int max_query, int max_seq, int H, int decode_len, int init_blocks,
    int local_blocks, c10::optional<torch::Tensor> score_out) {
  const int B = seq_lens.size(0), total_q = q.size(0);
  const int max_blocks = (max_seq + kBlock - 1) / kBlock;
  const int score_stride = score_out.has_value()
                               ? score_out.value().size(2)
                               : (max_blocks + 15) / 16 * 16;
  const int tile_max = std::min(max_query, 64);
  TORCH_CHECK(score_stride >= max_blocks, "score_out block dimension is too small");
  auto score = score_out.has_value()
                   ? score_out.value().narrow(1, 0, total_q)
                   : torch::full({H, total_q, score_stride}, -INFINITY,
                                 q.options().dtype(torch::kFloat));
  hipStream_t stream = c10::hip::getCurrentHIPStream().stream();

  // Short-query workloads (including all normal decode calls) are both more
  // accurate and cheaper without materialising B*H*max_seq index workspaces.
  // Long prefill continues to use the hipBLAS matrix path below.
  constexpr int kDirectMaxQuery = 8;
  if (max_query <= kDirectMaxQuery) {
    const int64_t direct_work =
        static_cast<int64_t>(B) * H * max_query * max_blocks;
    TORCH_CHECK(direct_work <= INT_MAX, "direct index-score grid is too large");
    if constexpr (Decode) {
      hipLaunchKernelGGL((direct_index_score_kernel<true>),
                         dim3(static_cast<unsigned int>(direct_work)),
                         dim3(kBlock), 0, stream, q.data_ptr<at::BFloat16>(),
                         cache.data_ptr<at::BFloat16>(), table.data_ptr<int32_t>(),
                         nullptr, seq_lens.data_ptr<int32_t>(), nullptr,
                         score.data_ptr<float>(), B, H, total_q, max_query,
                         max_blocks, table.size(1), decode_len, init_blocks,
                         local_blocks, score.stride(0), score.stride(1),
                         score.stride(2));
    } else {
      hipLaunchKernelGGL((direct_index_score_kernel<false>),
                         dim3(static_cast<unsigned int>(direct_work)),
                         dim3(kBlock), 0, stream, q.data_ptr<at::BFloat16>(),
                         cache.data_ptr<at::BFloat16>(), table.data_ptr<int32_t>(),
                         cu.data_ptr<int32_t>(), seq_lens.data_ptr<int32_t>(),
                         prefix_lens.data_ptr<int32_t>(), score.data_ptr<float>(),
                         B, H, total_q, max_query, max_blocks, table.size(1), 0,
                         0, 0, score.stride(0), score.stride(1),
                         score.stride(2));
    }
    C10_HIP_KERNEL_LAUNCH_CHECK();
    return score;
  }
  auto kpack = torch::empty({B * H, max_seq, kDim}, q.options());
  auto qpack = torch::empty({B * H, tile_max, kDim}, q.options());
  auto token_score =
      torch::empty({B * H, tile_max, max_seq}, q.options().dtype(torch::kFloat));
  hipblasHandle_t handle = current_handle(stream);
  const int64_t kwork = static_cast<int64_t>(B) * H * max_seq * kDim;
  const int kgrid = std::min<int64_t>(65535, (kwork + kThreads - 1) / kThreads);
  hipLaunchKernelGGL(pack_index_k_kernel, dim3(kgrid), dim3(kThreads), 0, stream,
                     cache.data_ptr<at::BFloat16>(), table.data_ptr<int32_t>(),
                     seq_lens.data_ptr<int32_t>(), kpack.data_ptr<at::BFloat16>(),
                     B, H, max_seq, table.size(1));
  C10_HIP_KERNEL_LAUNCH_CHECK();
  for (int q0 = 0; q0 < max_query; q0 += tile_max) {
    const int tile = std::min(tile_max, max_query - q0);
    const int64_t qwork = static_cast<int64_t>(B) * H * tile * kDim;
    const int qgrid = std::min<int64_t>(65535, (qwork + kThreads - 1) / kThreads);
    if constexpr (Decode) {
      hipLaunchKernelGGL((pack_index_q_kernel<true>), dim3(qgrid), dim3(kThreads),
                         0, stream, q.data_ptr<at::BFloat16>(), nullptr,
                         seq_lens.data_ptr<int32_t>(), qpack.data_ptr<at::BFloat16>(),
                         B, H, total_q, q0, tile, decode_len);
    } else {
      hipLaunchKernelGGL((pack_index_q_kernel<false>), dim3(qgrid), dim3(kThreads),
                         0, stream, q.data_ptr<at::BFloat16>(), cu.data_ptr<int32_t>(),
                         seq_lens.data_ptr<int32_t>(), qpack.data_ptr<at::BFloat16>(),
                         B, H, total_q, q0, tile, 0);
    }
    C10_HIP_KERNEL_LAUNCH_CHECK();
    // qpack/token_score are allocated with the fixed tile_max leading
    // dimension. Keep the GEMM batch strides fixed as well; using the
    // current (possibly smaller) tail tile aliases the next batch when
    // max_query is not divisible by tile_max.
    index_gemm(handle, kpack.data_ptr<at::BFloat16>(),
               qpack.data_ptr<at::BFloat16>(), token_score.data_ptr<float>(),
               max_seq, tile, B * H, tile_max, tile_max);
    const int64_t reduce_work = static_cast<int64_t>(B) * H * tile * score_stride;
    const int rgrid = std::min<int64_t>(2147483647, reduce_work);
    if constexpr (Decode) {
      hipLaunchKernelGGL((reduce_index_score_kernel<true>), dim3(rgrid), dim3(kBlock),
                         0, stream, token_score.data_ptr<float>(), score.data_ptr<float>(),
                         nullptr, seq_lens.data_ptr<int32_t>(), nullptr, B, H,
                         total_q, max_seq, score_stride, q0, tile, decode_len,
                         init_blocks, local_blocks, score.stride(0),
                         score.stride(1), score.stride(2));
    } else {
      hipLaunchKernelGGL((reduce_index_score_kernel<false>), dim3(rgrid), dim3(kBlock),
                         0, stream, token_score.data_ptr<float>(), score.data_ptr<float>(),
                         cu.data_ptr<int32_t>(), seq_lens.data_ptr<int32_t>(),
                         prefix_lens.data_ptr<int32_t>(), B, H, total_q, max_seq,
                         score_stride, q0, tile, 0, 0, 0, score.stride(0),
                         score.stride(1), score.stride(2));
    }
    C10_HIP_KERNEL_LAUNCH_CHECK();
  }
  return score;
}

template <bool Decode>
torch::Tensor topk_impl(const torch::Tensor& score, const torch::Tensor& cu,
                        const torch::Tensor& seq_lens,
                        const torch::Tensor& prefix, int topk, int init_blocks,
                        int local_blocks, int decode_len,
                        c10::optional<torch::Tensor> out_opt) {
  const int H = score.size(0), total_q = score.size(1), stride = score.size(2);
  const int B = Decode ? seq_lens.size(0) : cu.size(0) - 1;
  if (out_opt.has_value()) {
    const auto& buffer = out_opt.value();
    TORCH_CHECK(buffer.is_cuda() && buffer.scalar_type() == torch::kInt &&
                    buffer.dim() == 3 && buffer.size(0) == H &&
                    buffer.size(1) >= total_q && buffer.size(2) >= topk,
                "out must be HIP int32 [heads,>=total_q,>=topk]");
  }
  torch::Tensor out = out_opt.has_value()
                          ? out_opt.value().narrow(1, 0, total_q)
                          : torch::empty({H, total_q, topk},
                                         score.options().dtype(torch::kInt));
  const int sort_size = next_pow2(stride);
  TORCH_CHECK(sort_size <= 4096, "MSA v1 topk supports at most 4096 blocks");
  hipStream_t stream = c10::hip::getCurrentHIPStream().stream();
  if constexpr (Decode) {
    hipLaunchKernelGGL((topk_kernel<true>), dim3(H * total_q), dim3(kThreads), 0,
                       stream, score.data_ptr<float>(), out.data_ptr<int32_t>(),
                       nullptr, seq_lens.data_ptr<int32_t>(), nullptr, B, H,
                       total_q, stride, topk, init_blocks, local_blocks, decode_len,
                       sort_size, score.stride(0), score.stride(1), score.stride(2),
                       out.stride(0), out.stride(1), out.stride(2));
  } else {
    hipLaunchKernelGGL((topk_kernel<false>), dim3(H * total_q), dim3(kThreads), 0,
                       stream, score.data_ptr<float>(), out.data_ptr<int32_t>(),
                       cu.data_ptr<int32_t>(), nullptr, prefix.data_ptr<int32_t>(), B,
                       H, total_q, stride, topk, init_blocks, local_blocks, 0,
                       sort_size, score.stride(0), score.stride(1), score.stride(2),
                       out.stride(0), out.stride(1), out.stride(2));
  }
  C10_HIP_KERNEL_LAUNCH_CHECK();
  return out;
}

template <bool Decode>
void sparse_impl(const torch::Tensor& q, const torch::Tensor& kv,
                 const torch::Tensor& topk_idx, const torch::Tensor& table,
                 const torch::Tensor& cu, const torch::Tensor& seq_lens,
                 const torch::Tensor& prefix, int H, float scale,
                 torch::Tensor& out, int decode_len) {
  const int total_q = q.size(0), num_heads = q.size(1), G = num_heads / H;
  const int topk = topk_idx.size(2), N = topk * kBlock;
  const int B = Decode ? seq_lens.size(0) : cu.size(0) - 1;
  const int tile_max = std::min(total_q, std::max(1, 512 / H));
  const int max_batch = tile_max * H;
  auto kpack = torch::empty({max_batch, N, kDim}, q.options());
  auto vpack = torch::empty({max_batch, N, kDim}, q.options());
  auto score = torch::empty({max_batch, G, N}, q.options().dtype(torch::kFloat));
  auto probs = torch::empty({max_batch, G, N}, q.options());
  hipStream_t stream = c10::hip::getCurrentHIPStream().stream();
  hipblasHandle_t handle = current_handle(stream);
  for (int q0 = 0; q0 < total_q; q0 += tile_max) {
    const int tile = std::min(tile_max, total_q - q0);
    const int batch = tile * H;
    const int64_t work = static_cast<int64_t>(batch) * N * kDim;
    const int grid = std::min<int64_t>(65535, (work + kThreads - 1) / kThreads);
    if constexpr (Decode) {
      hipLaunchKernelGGL((pack_sparse_kv_kernel<true>), dim3(grid), dim3(kThreads),
                         0, stream, kv.data_ptr<at::BFloat16>(),
                         topk_idx.data_ptr<int32_t>(), table.data_ptr<int32_t>(),
                         nullptr, seq_lens.data_ptr<int32_t>(), nullptr,
                         kpack.data_ptr<at::BFloat16>(), vpack.data_ptr<at::BFloat16>(),
                         B, total_q, H, table.size(1), topk, q0, tile, decode_len,
                         topk_idx.stride(0), topk_idx.stride(1), topk_idx.stride(2));
    } else {
      hipLaunchKernelGGL((pack_sparse_kv_kernel<false>), dim3(grid), dim3(kThreads),
                         0, stream, kv.data_ptr<at::BFloat16>(),
                         topk_idx.data_ptr<int32_t>(), table.data_ptr<int32_t>(),
                         cu.data_ptr<int32_t>(), seq_lens.data_ptr<int32_t>(),
                         prefix.data_ptr<int32_t>(), kpack.data_ptr<at::BFloat16>(),
                         vpack.data_ptr<at::BFloat16>(), B, total_q, H,
                         table.size(1), topk, q0, tile, 0, topk_idx.stride(0),
                         topk_idx.stride(1), topk_idx.stride(2));
    }
    C10_HIP_KERNEL_LAUNCH_CHECK();
    const at::BFloat16* qptr = q.data_ptr<at::BFloat16>() +
                               static_cast<int64_t>(q0) * num_heads * kDim;
    sparse_qk_gemm(handle, kpack.data_ptr<at::BFloat16>(), qptr,
                   score.data_ptr<float>(), N, G, batch, scale);
    if constexpr (Decode) {
      hipLaunchKernelGGL((sparse_softmax_kernel<true>), dim3(batch * G),
                         dim3(kThreads), 0, stream, score.data_ptr<float>(),
                         probs.data_ptr<at::BFloat16>(), topk_idx.data_ptr<int32_t>(),
                         nullptr, seq_lens.data_ptr<int32_t>(), nullptr, B, total_q,
                         H, G, topk, q0, tile, decode_len, topk_idx.stride(0),
                         topk_idx.stride(1), topk_idx.stride(2));
    } else {
      hipLaunchKernelGGL((sparse_softmax_kernel<false>), dim3(batch * G),
                         dim3(kThreads), 0, stream, score.data_ptr<float>(),
                         probs.data_ptr<at::BFloat16>(), topk_idx.data_ptr<int32_t>(),
                         cu.data_ptr<int32_t>(), seq_lens.data_ptr<int32_t>(),
                         prefix.data_ptr<int32_t>(), B, total_q, H, G, topk, q0,
                         tile, 0, topk_idx.stride(0), topk_idx.stride(1),
                         topk_idx.stride(2));
    }
    C10_HIP_KERNEL_LAUNCH_CHECK();
    at::BFloat16* optr = out.data_ptr<at::BFloat16>() +
                         static_cast<int64_t>(q0) * num_heads * kDim;
    sparse_pv_gemm(handle, vpack.data_ptr<at::BFloat16>(),
                   probs.data_ptr<at::BFloat16>(), optr, N, G, batch);
  }
}

}  // namespace

torch::Tensor index_score_prefill(
    torch::Tensor q, torch::Tensor cache, torch::Tensor table, torch::Tensor cu,
    torch::Tensor seq_lens, torch::Tensor prefix, int64_t max_query,
    int64_t max_seq, int64_t heads) {
  validate_index(q, cache, table, seq_lens, heads);
  TORCH_CHECK(cu.scalar_type() == torch::kInt && prefix.scalar_type() == torch::kInt &&
                  cu.is_contiguous() && prefix.is_contiguous(),
              "prefill metadata must be contiguous int32");
  return index_score_impl<false>(q, cache, table, cu, seq_lens, prefix,
                                 max_query, max_seq, heads, 0, 0, 0,
                                 c10::nullopt);
}

torch::Tensor index_topk_prefill(
    torch::Tensor score, torch::Tensor cu, torch::Tensor prefix,
    int64_t max_query, int64_t topk, int64_t init_blocks,
    int64_t local_blocks, c10::optional<torch::Tensor> out) {
  TORCH_CHECK(score.is_cuda() && score.scalar_type() == torch::kFloat &&
                  score.is_contiguous(),
              "score must be contiguous HIP float32");
  TORCH_CHECK(topk > 0 && topk <= kMaxTopK, "topk must be in [1,16]");
  auto empty = torch::Tensor();
  return topk_impl<false>(score, cu, empty, prefix, topk, init_blocks,
                          local_blocks, 0, out);
}

torch::Tensor index_decode(
    torch::Tensor q, torch::Tensor cache, torch::Tensor table,
    torch::Tensor seq_lens, int64_t max_seq, int64_t topk,
    int64_t init_blocks, int64_t local_blocks, int64_t heads,
    int64_t decode_len, c10::optional<torch::Tensor> out,
    c10::optional<torch::Tensor> score_out) {
  validate_index(q, cache, table, seq_lens, heads);
  TORCH_CHECK(q.size(0) == seq_lens.size(0) * decode_len,
              "decode total_q mismatch");
  TORCH_CHECK(topk > 0 && topk <= kMaxTopK, "topk must be in [1,16]");
  if (score_out.has_value()) {
    const auto& score = score_out.value();
    TORCH_CHECK(score.is_cuda() && score.scalar_type() == torch::kFloat &&
                    score.dim() == 3 && score.size(0) == heads &&
                    score.size(1) >= q.size(0),
                "score_out must be HIP FP32 [heads,>=total_q,>=blocks]");
  }
  auto empty = torch::Tensor();
  auto score = index_score_impl<true>(q, cache, table, empty, seq_lens, empty,
                                      decode_len, max_seq, heads, decode_len,
                                      init_blocks, local_blocks, score_out);
  return topk_impl<true>(score, empty, seq_lens, empty, topk, init_blocks,
                         local_blocks, decode_len, out);
}

torch::Tensor index_decode_score(
    torch::Tensor q, torch::Tensor cache, torch::Tensor table,
    torch::Tensor seq_lens, int64_t max_seq, int64_t init_blocks,
    int64_t local_blocks, int64_t heads, int64_t decode_len,
    c10::optional<torch::Tensor> score_out) {
  validate_index(q, cache, table, seq_lens, heads);
  TORCH_CHECK(q.size(0) == seq_lens.size(0) * decode_len,
              "decode total_q mismatch");
  if (score_out.has_value()) {
    const auto& score = score_out.value();
    TORCH_CHECK(score.is_cuda() && score.scalar_type() == torch::kFloat &&
                    score.dim() == 3 &&
                    score.size(0) == heads && score.size(1) >= q.size(0),
                "score_out must be HIP FP32 [heads,>=total_q,>=blocks]");
  }
  auto empty = torch::Tensor();
  return index_score_impl<true>(q, cache, table, empty, seq_lens, empty,
                                decode_len, max_seq, heads, decode_len,
                                init_blocks, local_blocks, score_out);
}

void sparse_prefill(torch::Tensor q, torch::Tensor kv, torch::Tensor topk,
                    torch::Tensor table, torch::Tensor cu,
                    torch::Tensor seq_lens, torch::Tensor prefix,
                    int64_t max_query, int64_t heads, double scale,
                    torch::Tensor out) {
  TORCH_CHECK(q.scalar_type() == torch::kBFloat16 &&
                  kv.scalar_type() == torch::kBFloat16 &&
                  out.scalar_type() == torch::kBFloat16,
              "sparse tensors must be BF16");
  TORCH_CHECK(q.is_contiguous() && kv.is_contiguous() && out.is_contiguous(),
              "sparse tensors must be contiguous");
  TORCH_CHECK(q.size(2) == kDim && kv.size(2) == kBlock &&
                  kv.size(3) == 2 * kDim && q.size(1) % heads == 0,
              "unsupported sparse shape");
  sparse_impl<false>(q, kv, topk, table, cu, seq_lens, prefix, heads,
                     static_cast<float>(scale), out, 0);
}

void sparse_decode(torch::Tensor q, torch::Tensor kv, torch::Tensor topk,
                   torch::Tensor table, torch::Tensor seq_lens, int64_t heads,
                   double scale, torch::Tensor out, int64_t decode_len) {
  TORCH_CHECK(q.scalar_type() == torch::kBFloat16 &&
                  kv.scalar_type() == torch::kBFloat16 &&
                  out.scalar_type() == torch::kBFloat16,
              "sparse tensors must be BF16");
  TORCH_CHECK(q.is_contiguous() && kv.is_contiguous() && out.is_contiguous(),
              "sparse tensors must be contiguous");
  TORCH_CHECK(q.size(0) == seq_lens.size(0) * decode_len && q.size(2) == kDim &&
                  kv.size(2) == kBlock && kv.size(3) == 2 * kDim &&
                  q.size(1) % heads == 0,
              "unsupported decode sparse shape");
  auto empty = torch::Tensor();
  sparse_impl<true>(q, kv, topk, table, empty, seq_lens, empty, heads,
                    static_cast<float>(scale), out, decode_len);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("index_score_prefill", &index_score_prefill);
  m.def("index_topk_prefill", &index_topk_prefill);
  m.def("index_decode_score", &index_decode_score);
  m.def("index_decode", &index_decode);
  m.def("sparse_prefill", &sparse_prefill);
  m.def("sparse_decode", &sparse_decode);
}
