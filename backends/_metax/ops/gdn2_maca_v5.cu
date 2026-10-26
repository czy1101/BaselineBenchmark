// This file is modified and supported by the Moonshot AI Team

// Native MACA/CUDA-compatible GDN2 forward kernels.
//
// This file is an independent implementation of the Gated DeltaNet-2 forward
// equations. It intentionally contains no Triton, TLE, FlagGems, or
// FlagAttention kernel code. MetaX MACA's CUDA compatibility frontend compiles
// the same .cu source for C550.
//
// Layouts:
//   q/k/g/b: [B, T, H, K]
//   v/w/o:   [B, T, H, V]
//   state:   [N, H, K, V] or [N, H, V, K]

// V1/V2 retain the exact token-recurrent implementation as a correctness
// fallback. V3 adds a genuine BT=64 chunkwise WY factorization:
//   A = (I + tril((b*k*exp(G)) @ (k*exp(-G))^T, -1))^-1
//   W = A @ (b*k*exp(G)), U = A @ (write_gate*v)
// followed by one recurrent update per chunk rather than per token. V4 routes
// dense products through MACA BMM. V5 reduces the serial loop to three BMMs
// and two fused native kernels with reusable workspaces.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

namespace {

constexpr int kThreads = 64;  // One native C550 warp.
constexpr int kMaxK = 256;
constexpr int kChunk = 64;

template <typename scalar_t>
__device__ __forceinline__ float load_float(const scalar_t* ptr) {
  return static_cast<float>(*ptr);
}

template <typename scalar_t>
__device__ __forceinline__ void store_scalar(scalar_t* ptr, float value) {
  *ptr = static_cast<scalar_t>(value);
}

template <typename scalar_t>
__device__ __forceinline__ float round_to_scalar(float value) {
  return static_cast<float>(static_cast<scalar_t>(value));
}

__device__ __forceinline__ float stable_softplus(float x) {
  if (x > 20.0f) {
    return x;
  }
  if (x < -20.0f) {
    return expf(x);
  }
  return log1pf(expf(x));
}

__device__ __forceinline__ float stable_sigmoid(float x) {
  if (x >= 0.0f) {
    return 1.0f / (1.0f + expf(-x));
  }
  const float z = expf(x);
  return z / (1.0f + z);
}

template <typename scalar_t>
__global__ void gdn2_recurrent_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const scalar_t* __restrict__ g,
    const scalar_t* __restrict__ b,
    const scalar_t* __restrict__ w,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    const int64_t* __restrict__ cu_seqlens,
    scalar_t* __restrict__ out,
    float* __restrict__ state,
    int64_t B,
    int64_t T,
    int64_t H,
    int64_t K,
    int64_t V,
    int64_t N,
    float scale,
    float lower_bound,
    bool is_varlen,
    bool state_v_first,
    bool use_qk_l2norm,
    bool use_gate_in_kernel,
    bool has_dt_bias,
    bool has_lower_bound) {
  __shared__ float s_q[kMaxK];
  __shared__ float s_k[kMaxK];
  __shared__ float s_decay[kMaxK];
  __shared__ float s_bk[kMaxK];
  __shared__ float s_q_inv_norm;
  __shared__ float s_k_inv_norm;

  const int64_t v_tiles = (V + kThreads - 1) / kThreads;
  int64_t linear = static_cast<int64_t>(blockIdx.x);
  const int64_t v_tile = linear % v_tiles;
  linear /= v_tiles;
  const int64_t h = linear % H;
  const int64_t n = linear / H;
  if (n >= N) {
    return;
  }

  int64_t bos;
  int64_t eos;
  if (is_varlen) {
    bos = cu_seqlens[n];
    eos = cu_seqlens[n + 1];
  } else {
    bos = n * T;
    eos = bos + T;
  }

  const int64_t v_index = v_tile * kThreads + threadIdx.x;
  const bool valid_v = v_index < V;
  const int64_t state_head_base = (n * H + h) * K * V;

  for (int64_t token = bos; token < eos; ++token) {
    const int64_t qk_base = (token * H + h) * K;
    const int64_t vv_base = (token * H + h) * V;

    // Stage per-token K-axis values once per CTA.  s_bk temporarily stores b
    // and is overwritten with b*k after optional K normalization.
    for (int64_t kk = threadIdx.x; kk < K; kk += blockDim.x) {
      s_q[kk] = load_float(q + qk_base + kk);
      s_k[kk] = load_float(k + qk_base + kk);
      s_decay[kk] = load_float(g + qk_base + kk);
      s_bk[kk] = load_float(b + qk_base + kk);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      float q_ss = 0.0f;
      float k_ss = 0.0f;
      if (use_qk_l2norm) {
        for (int64_t kk = 0; kk < K; ++kk) {
          q_ss += s_q[kk] * s_q[kk];
          k_ss += s_k[kk] * s_k[kk];
        }
      }
      s_q_inv_norm = use_qk_l2norm ? rsqrtf(q_ss + 1.0e-6f) : 1.0f;
      s_k_inv_norm = use_qk_l2norm ? rsqrtf(k_ss + 1.0e-6f) : 1.0f;
    }
    __syncthreads();

    for (int64_t kk = threadIdx.x; kk < K; kk += blockDim.x) {
      const float q_value = s_q[kk] * s_q_inv_norm * scale;
      const float k_value = s_k[kk] * s_k_inv_norm;
      float log_decay = s_decay[kk];
      if (use_gate_in_kernel) {
        float gate_input = log_decay;
        if (has_dt_bias) {
          gate_input += dt_bias[h * K + kk];
        }
        const float A = expf(A_log[h]);
        if (has_lower_bound) {
          log_decay = lower_bound * stable_sigmoid(A * gate_input);
        } else {
          log_decay = -A * stable_softplus(gate_input);
        }
      }
      s_q[kk] = q_value;
      s_k[kk] = k_value;
      s_decay[kk] = expf(log_decay);
      s_bk[kk] *= k_value;
    }
    __syncthreads();

    if (valid_v) {
      float erase = 0.0f;

      // First pass: S <- diag(exp(g)) S, then calculate (b*k)^T S.
      for (int64_t kk = 0; kk < K; ++kk) {
        const int64_t state_offset =
            state_v_first ? state_head_base + v_index * K + kk
                          : state_head_base + kk * V + v_index;
        const float decayed = state[state_offset] * s_decay[kk];
        state[state_offset] = decayed;
        erase += decayed * s_bk[kk];
      }

      const float v_new = load_float(w + vv_base + v_index) *
                              load_float(v + vv_base + v_index) -
                          erase;

      // Second pass: S <- S + k outer v_new, then o <- (scale*q)^T S.
      float output_value = 0.0f;
      for (int64_t kk = 0; kk < K; ++kk) {
        const int64_t state_offset =
            state_v_first ? state_head_base + v_index * K + kk
                          : state_head_base + kk * V + v_index;
        const float updated = state[state_offset] + s_k[kk] * v_new;
        state[state_offset] = updated;
        output_value += updated * s_q[kk];
      }
      store_scalar(out + vv_base + v_index, output_value);
    }

    // No CTA may overwrite the shared K tile until every value-column has
    // finished both state passes for the current token.
    __syncthreads();
  }
}

// V2: keep one [K, BV] state tile in shared memory for the complete sequence.
// Compared with the V1 kernel above, recurrent state traffic changes from
// O(T*K*V) global loads/stores to one load and one final store per state value.
// The token recurrence remains exact and serial; this is the state-resident
// stage before introducing the BT=64 token-parallel factorization.
template <typename scalar_t>
__global__ void gdn2_persistent_tile_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const scalar_t* __restrict__ g,
    const scalar_t* __restrict__ b,
    const scalar_t* __restrict__ w,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    const int64_t* __restrict__ cu_seqlens,
    scalar_t* __restrict__ out,
    float* __restrict__ state,
    int64_t B,
    int64_t T,
    int64_t H,
    int64_t K,
    int64_t V,
    int64_t N,
    int64_t BV,
    float scale,
    float lower_bound,
    bool is_varlen,
    bool state_v_first,
    bool use_qk_l2norm,
    bool use_gate_in_kernel,
    bool has_dt_bias,
    bool has_lower_bound) {
  extern __shared__ float shared[];
  float* s_state = shared;
  float* s_q = s_state + K * BV;
  float* s_k = s_q + K;
  float* s_decay = s_k + K;
  float* s_bk = s_decay + K;
  float* s_q_inv_norm = s_bk + K;
  float* s_k_inv_norm = s_q_inv_norm + 1;

  const int64_t v_tiles = (V + BV - 1) / BV;
  int64_t linear = static_cast<int64_t>(blockIdx.x);
  const int64_t v_tile = linear % v_tiles;
  linear /= v_tiles;
  const int64_t h = linear % H;
  const int64_t n = linear / H;
  if (n >= N) {
    return;
  }

  int64_t bos;
  int64_t eos;
  if (is_varlen) {
    bos = cu_seqlens[n];
    eos = cu_seqlens[n + 1];
  } else {
    bos = n * T;
    eos = bos + T;
  }

  const int64_t state_head_base = (n * H + h) * K * V;

  // Cooperative state-tile load. Invalid tail columns are initialized to zero
  // but never written to the global state tensor.
  for (int64_t index = threadIdx.x; index < K * BV; index += blockDim.x) {
    const int64_t kk = index / BV;
    const int64_t local_v = index - kk * BV;
    const int64_t global_v = v_tile * BV + local_v;
    float value = 0.0f;
    if (global_v < V) {
      const int64_t state_offset =
          state_v_first ? state_head_base + global_v * K + kk
                        : state_head_base + kk * V + global_v;
      value = state[state_offset];
    }
    s_state[index] = value;
  }
  __syncthreads();

  for (int64_t token = bos; token < eos; ++token) {
    const int64_t qk_base = (token * H + h) * K;
    const int64_t vv_base = (token * H + h) * V;

    for (int64_t kk = threadIdx.x; kk < K; kk += blockDim.x) {
      s_q[kk] = load_float(q + qk_base + kk);
      s_k[kk] = load_float(k + qk_base + kk);
      s_decay[kk] = load_float(g + qk_base + kk);
      s_bk[kk] = load_float(b + qk_base + kk);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      float q_ss = 0.0f;
      float k_ss = 0.0f;
      if (use_qk_l2norm) {
        for (int64_t kk = 0; kk < K; ++kk) {
          q_ss += s_q[kk] * s_q[kk];
          k_ss += s_k[kk] * s_k[kk];
        }
      }
      *s_q_inv_norm = use_qk_l2norm ? rsqrtf(q_ss + 1.0e-6f) : 1.0f;
      *s_k_inv_norm = use_qk_l2norm ? rsqrtf(k_ss + 1.0e-6f) : 1.0f;
    }
    __syncthreads();

    for (int64_t kk = threadIdx.x; kk < K; kk += blockDim.x) {
      const float q_value = s_q[kk] * (*s_q_inv_norm) * scale;
      const float k_value = s_k[kk] * (*s_k_inv_norm);
      float log_decay = s_decay[kk];
      if (use_gate_in_kernel) {
        float gate_input = log_decay;
        if (has_dt_bias) {
          gate_input += dt_bias[h * K + kk];
        }
        const float A = expf(A_log[h]);
        log_decay = has_lower_bound
                        ? lower_bound * stable_sigmoid(A * gate_input)
                        : -A * stable_softplus(gate_input);
      }
      s_q[kk] = q_value;
      s_k[kk] = k_value;
      s_decay[kk] = expf(log_decay);
      s_bk[kk] *= k_value;
    }
    __syncthreads();

    // One thread owns one local V column for the complete sequence. K=256
    // selects BV=32, leaving the second half-warp available for staging while
    // keeping dynamic shared memory well below C550's 64 KiB limit.
    const int64_t local_v = threadIdx.x;
    const int64_t global_v = v_tile * BV + local_v;
    if (local_v < BV && global_v < V) {
      float erase = 0.0f;
      for (int64_t kk = 0; kk < K; ++kk) {
        const int64_t state_index = kk * BV + local_v;
        const float decayed = s_state[state_index] * s_decay[kk];
        s_state[state_index] = decayed;
        erase += decayed * s_bk[kk];
      }

      const float v_new = load_float(w + vv_base + global_v) *
                              load_float(v + vv_base + global_v) -
                          erase;
      float output_value = 0.0f;
      for (int64_t kk = 0; kk < K; ++kk) {
        const int64_t state_index = kk * BV + local_v;
        const float updated = s_state[state_index] + s_k[kk] * v_new;
        s_state[state_index] = updated;
        output_value += updated * s_q[kk];
      }
      store_scalar(out + vv_base + global_v, output_value);
    }
    __syncthreads();
  }

  // Cooperative final-state writeback.
  for (int64_t index = threadIdx.x; index < K * BV; index += blockDim.x) {
    const int64_t kk = index / BV;
    const int64_t local_v = index - kk * BV;
    const int64_t global_v = v_tile * BV + local_v;
    if (global_v < V) {
      const int64_t state_offset =
          state_v_first ? state_head_base + global_v * K + kk
                        : state_head_base + kk * V + global_v;
      state[state_offset] = s_state[index];
    }
  }
}

// ---------------------------------------------------------------------------
// V3 BT=64 chunkwise forward.
// ---------------------------------------------------------------------------
// For one chunk, define G_t = sum_{i<=t} g_i, ka_t=k_t*exp(-G_t), and
// c_t=b_t*k_t*exp(G_t). The recurrent update transformed into the ungated
// coordinate system is
//
//   Sbar_t = Sbar_{t-1} + ka_t (x_t - c_t^T Sbar_{t-1}),
//   x_t = write_gate_t * v_t.
//
// Its WY representation is U=A@X, W=A@C with
// A=(I+tril(C@Ka^T,-1))^-1. This makes all 64 tokens inside a chunk available
// to parallel matrix work; only the sequence of chunks remains recurrent.

template <typename scalar_t>
__global__ void gdn2_chunk64_cumsum_kernel(
    const scalar_t* __restrict__ g,
    scalar_t* __restrict__ gate_cumsum,
    int64_t B,
    int64_t T,
    int64_t H,
    int64_t K,
    int64_t NT) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = B * NT * H * K;
  if (index >= total) {
    return;
  }

  const int64_t kk = index % K;
  index /= K;
  const int64_t h = index % H;
  index /= H;
  const int64_t chunk = index % NT;
  const int64_t batch = index / NT;
  const int64_t begin = chunk * kChunk;
  const int64_t length =
      (T - begin) < static_cast<int64_t>(kChunk) ? (T - begin) : kChunk;

  float cumulative = 0.0f;
  for (int64_t row = 0; row < length; ++row) {
    const int64_t offset =
        (((batch * T + begin + row) * H + h) * K + kk);
    cumulative += load_float(g + offset);
    store_scalar(gate_cumsum + offset, cumulative);
  }
}

template <typename scalar_t>
__global__ void gdn2_chunk64_build_matrices_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ gate_cumsum,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ inverse,
    scalar_t* __restrict__ aqk,
    int64_t T,
    int64_t H,
    int64_t K,
    int64_t NT,
    float scale) {
  // 32 KiB total. This stays below the C550 64 KiB shared-memory limit.
  __shared__ float lower[kChunk * kChunk];
  __shared__ float inv[kChunk * kChunk];

  int64_t linear = static_cast<int64_t>(blockIdx.x);
  const int64_t h = linear % H;
  linear /= H;
  const int64_t chunk = linear % NT;
  const int64_t batch = linear / NT;
  const int64_t begin = chunk * kChunk;
  const int64_t length =
      (T - begin) < static_cast<int64_t>(kChunk) ? (T - begin) : kChunk;
  const int64_t matrix_base =
      ((batch * NT + chunk) * H + h) * kChunk * kChunk;

  for (int index = threadIdx.x; index < kChunk * kChunk;
       index += blockDim.x) {
    lower[index] = 0.0f;
    inv[index] = 0.0f;
    store_scalar(aqk + matrix_base + index, 0.0f);
  }
  __syncthreads();

  // Build the strictly-lower erase matrix L and causal output matrix Aqk.
  for (int pair = threadIdx.x; pair < kChunk * kChunk;
       pair += blockDim.x) {
    const int row = pair / kChunk;
    const int col = pair - row * kChunk;
    if (row < length && col <= row) {
      const int64_t row_base =
          (((batch * T + begin + row) * H + h) * K);
      const int64_t col_base =
          (((batch * T + begin + col) * H + h) * K);
      float qk = 0.0f;
      float erase = 0.0f;
      for (int64_t kk = 0; kk < K; ++kk) {
        const float delta = load_float(gate_cumsum + row_base + kk) -
                            load_float(gate_cumsum + col_base + kk);
        const float decayed_key = load_float(k + col_base + kk) * expf(delta);
        qk += load_float(q + row_base + kk) * decayed_key;
        if (col < row) {
          erase += load_float(b + row_base + kk) *
                   load_float(k + row_base + kk) * decayed_key;
        }
      }
      lower[pair] = col < row ? erase : 0.0f;
      store_scalar(aqk + matrix_base + pair, qk * scale);
    }
  }
  __syncthreads();

  for (int index = threadIdx.x; index < kChunk * kChunk;
       index += blockDim.x) {
    const int row = index / kChunk;
    const int col = index - row * kChunk;
    inv[index] = (row < length && row == col) ? 1.0f : 0.0f;
  }
  __syncthreads();

  // Forward substitution for (I+L)^-1. Rows are dependent, while all columns
  // in the current row are independent and are evaluated cooperatively.
  for (int row = 1; row < length; ++row) {
    for (int col = threadIdx.x; col < row; col += blockDim.x) {
      float value = 0.0f;
      for (int middle = col; middle < row; ++middle) {
        value += lower[row * kChunk + middle] *
                 inv[middle * kChunk + col];
      }
      inv[row * kChunk + col] = -value;
    }
    __syncthreads();
  }

  for (int index = threadIdx.x; index < kChunk * kChunk;
       index += blockDim.x) {
    store_scalar(inverse + matrix_base + index, inv[index]);
  }
}

template <typename scalar_t, bool kBuildKey>
__global__ void gdn2_chunk64_wy_kernel(
    const scalar_t* __restrict__ first,
    const scalar_t* __restrict__ second,
    const scalar_t* __restrict__ gate_cumsum,
    const scalar_t* __restrict__ inverse,
    scalar_t* __restrict__ output,
    int64_t B,
    int64_t T,
    int64_t H,
    int64_t D,
    int64_t NT,
    int64_t tiles_per_head) {
  int64_t linear = static_cast<int64_t>(blockIdx.x);
  const int64_t tile = linear % tiles_per_head;
  linear /= tiles_per_head;
  const int64_t h = linear % H;
  linear /= H;
  const int64_t chunk = linear % NT;
  const int64_t batch = linear / NT;
  const int64_t begin = chunk * kChunk;
  const int64_t length =
      (T - begin) < static_cast<int64_t>(kChunk) ? (T - begin) : kChunk;
  const int64_t local = tile * blockDim.x + threadIdx.x;
  const int64_t row = local / D;
  const int64_t d = local - row * D;
  if (row >= length || d >= D) {
    return;
  }

  const int64_t matrix_base =
      ((batch * NT + chunk) * H + h) * kChunk * kChunk;
  float value = 0.0f;
  for (int64_t col = 0; col <= row; ++col) {
    const int64_t tensor_offset =
        (((batch * T + begin + col) * H + h) * D + d);
    float rhs = load_float(first + tensor_offset) *
                load_float(second + tensor_offset);
    if (kBuildKey) {
      rhs *= expf(load_float(gate_cumsum + tensor_offset));
    }
    value += load_float(inverse + matrix_base + row * kChunk + col) * rhs;
  }
  const int64_t output_offset =
      (((batch * T + begin + row) * H + h) * D + d);
  store_scalar(output + output_offset, value);
}

template <typename scalar_t>
__global__ void gdn2_chunk64_state_output_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ gate_cumsum,
    const scalar_t* __restrict__ wy_key,
    const scalar_t* __restrict__ wy_value,
    const scalar_t* __restrict__ aqk,
    scalar_t* __restrict__ out,
    float* __restrict__ state,
    int64_t B,
    int64_t T,
    int64_t H,
    int64_t K,
    int64_t V,
    int64_t NT,
    int64_t BV,
    float scale) {
  extern __shared__ float shared[];
  float* s_state = shared;
  float* s_vnew = s_state + K * BV;

  const int64_t v_tiles = (V + BV - 1) / BV;
  int64_t linear = static_cast<int64_t>(blockIdx.x);
  const int64_t v_tile = linear % v_tiles;
  linear /= v_tiles;
  const int64_t h = linear % H;
  const int64_t batch = linear / H;
  if (batch >= B) {
    return;
  }

  const int64_t state_base = (batch * H + h) * K * V;
  for (int64_t index = threadIdx.x; index < K * BV;
       index += blockDim.x) {
    const int64_t kk = index / BV;
    const int64_t local_v = index - kk * BV;
    const int64_t global_v = v_tile * BV + local_v;
    s_state[index] =
        global_v < V ? state[state_base + kk * V + global_v] : 0.0f;
  }
  __syncthreads();

  const int64_t local_v = threadIdx.x;
  const int64_t global_v = v_tile * BV + local_v;
  const bool valid_v = local_v < BV && global_v < V;

  for (int64_t chunk = 0; chunk < NT; ++chunk) {
    const int64_t begin = chunk * kChunk;
    const int64_t length =
        (T - begin) < static_cast<int64_t>(kChunk) ? (T - begin) : kChunk;
    const int64_t matrix_base =
        ((batch * NT + chunk) * H + h) * kChunk * kChunk;

    if (valid_v) {
      for (int64_t row = 0; row < length; ++row) {
        const int64_t qk_base =
            (((batch * T + begin + row) * H + h) * K);
        const int64_t value_base =
            (((batch * T + begin + row) * H + h) * V);
        float correction = 0.0f;
        float inter_output = 0.0f;
        for (int64_t kk = 0; kk < K; ++kk) {
          const float state_value = s_state[kk * BV + local_v];
          correction += load_float(wy_key + qk_base + kk) * state_value;
          const float qg = round_to_scalar<scalar_t>(
              load_float(q + qk_base + kk) *
              expf(load_float(gate_cumsum + qk_base + kk)));
          inter_output += qg * state_value;
        }
        const float vnew = round_to_scalar<scalar_t>(
            load_float(wy_value + value_base + global_v) - correction);
        s_vnew[row * BV + local_v] = vnew;
        store_scalar(out + value_base + global_v, inter_output * scale);
      }
    }
    __syncthreads();

    if (valid_v) {
      for (int64_t row = 0; row < length; ++row) {
        float intra_output = 0.0f;
        for (int64_t col = 0; col <= row; ++col) {
          intra_output +=
              load_float(aqk + matrix_base + row * kChunk + col) *
              s_vnew[col * BV + local_v];
        }
        const int64_t output_offset =
            (((batch * T + begin + row) * H + h) * V + global_v);
        store_scalar(out + output_offset,
                     load_float(out + output_offset) + intra_output);
      }
    }
    __syncthreads();

    const int64_t last = begin + length - 1;
    for (int64_t index = threadIdx.x; index < K * BV;
         index += blockDim.x) {
      const int64_t kk = index / BV;
      const int64_t tile_v = index - kk * BV;
      const int64_t output_v = v_tile * BV + tile_v;
      if (output_v < V) {
        const int64_t last_offset =
            (((batch * T + last) * H + h) * K + kk);
        const float gate_last = load_float(gate_cumsum + last_offset);
        float updated = s_state[index] * expf(gate_last);
        for (int64_t row = 0; row < length; ++row) {
          const int64_t key_offset =
              (((batch * T + begin + row) * H + h) * K + kk);
          const float kg = round_to_scalar<scalar_t>(
              load_float(k + key_offset) *
              expf(gate_last - load_float(gate_cumsum + key_offset)));
          updated += kg * s_vnew[row * BV + tile_v];
        }
        s_state[index] = updated;
      }
    }
    __syncthreads();
  }

  for (int64_t index = threadIdx.x; index < K * BV;
       index += blockDim.x) {
    const int64_t kk = index / BV;
    const int64_t local_column = index - kk * BV;
    const int64_t global_column = v_tile * BV + local_column;
    if (global_column < V) {
      state[state_base + kk * V + global_column] = s_state[index];
    }
  }
}

// ---------------------------------------------------------------------------
// V4 matrix-core orchestration helpers.
// ---------------------------------------------------------------------------
// These kernels only transform layouts and solve the causal triangular system.
// All dense products are issued as batched ATen BMM operations by the host
// function below; on MACA they dispatch to the MCBLAS/MCTlass matrix backend.

template <typename scalar_t>
__global__ void gdn2_v4_pack_qkc_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ b,
    const scalar_t* __restrict__ gate_cumsum,
    scalar_t* __restrict__ qg,
    scalar_t* __restrict__ key_transposed,
    scalar_t* __restrict__ erase_key,
    float* __restrict__ chunk_decay,
    int64_t B,
    int64_t T,
    int64_t H,
    int64_t K,
    int64_t NT,
    int64_t M) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = M * kChunk * K;
  if (index >= total) {
    return;
  }
  const int64_t kk = index % K;
  index /= K;
  const int64_t row = index % kChunk;
  const int64_t matrix = index / kChunk;
  const int64_t chunk = matrix / (B * H);
  const int64_t batch_head = matrix - chunk * B * H;
  const int64_t batch = batch_head / H;
  const int64_t h = batch_head - batch * H;
  const int64_t begin = chunk * kChunk;
  const int64_t length =
      (T - begin) < static_cast<int64_t>(kChunk) ? (T - begin) : kChunk;
  const int64_t packed_row = (matrix * kChunk + row) * K + kk;
  const int64_t packed_transposed = (matrix * K + kk) * kChunk + row;

  if (row < length) {
    const int64_t source = (((batch * T + begin + row) * H + h) * K + kk);
    const float cumulative = load_float(gate_cumsum + source);
    const float exp_gate = expf(cumulative);
    const float key = load_float(k + source);
    store_scalar(qg + packed_row,
                 round_to_scalar<scalar_t>(load_float(q + source) * exp_gate));
    store_scalar(key_transposed + packed_transposed,
                 round_to_scalar<scalar_t>(key / exp_gate));
    store_scalar(erase_key + packed_row,
                 round_to_scalar<scalar_t>(
                     load_float(b + source) * key * exp_gate));
    if (row == 0) {
      const int64_t last_source =
          (((batch * T + begin + length - 1) * H + h) * K + kk);
      chunk_decay[matrix * K + kk] =
          expf(load_float(gate_cumsum + last_source));
    }
  } else {
    store_scalar(qg + packed_row, 0.0f);
    store_scalar(key_transposed + packed_transposed, 0.0f);
    store_scalar(erase_key + packed_row, 0.0f);
  }
}

template <typename scalar_t>
__global__ void gdn2_v4_pack_x_kernel(
    const scalar_t* __restrict__ v,
    const scalar_t* __restrict__ write_gate,
    scalar_t* __restrict__ x,
    int64_t B,
    int64_t T,
    int64_t H,
    int64_t V,
    int64_t M) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = M * kChunk * V;
  if (index >= total) {
    return;
  }
  const int64_t vv = index % V;
  index /= V;
  const int64_t row = index % kChunk;
  const int64_t matrix = index / kChunk;
  const int64_t chunk = matrix / (B * H);
  const int64_t batch_head = matrix - chunk * B * H;
  const int64_t batch = batch_head / H;
  const int64_t h = batch_head - batch * H;
  const int64_t begin = chunk * kChunk;
  const int64_t length =
      (T - begin) < static_cast<int64_t>(kChunk) ? (T - begin) : kChunk;
  const int64_t packed = (matrix * kChunk + row) * V + vv;
  if (row < length) {
    const int64_t source = (((batch * T + begin + row) * H + h) * V + vv);
    store_scalar(x + packed,
                 round_to_scalar<scalar_t>(load_float(v + source) *
                                           load_float(write_gate + source)));
  } else {
    store_scalar(x + packed, 0.0f);
  }
}

template <typename scalar_t>
__global__ void gdn2_v4_solve_kernel(
    const scalar_t* __restrict__ lower_product,
    scalar_t* __restrict__ aqk,
    scalar_t* __restrict__ inverse,
    int64_t B,
    int64_t T,
    int64_t H,
    int64_t M) {
  __shared__ float lower[kChunk * kChunk];
  __shared__ float inv[kChunk * kChunk];
  const int64_t matrix = static_cast<int64_t>(blockIdx.x);
  if (matrix >= M) {
    return;
  }
  const int64_t chunk = matrix / (B * H);
  const int64_t begin = chunk * kChunk;
  const int64_t length =
      (T - begin) < static_cast<int64_t>(kChunk) ? (T - begin) : kChunk;
  const int64_t base = matrix * kChunk * kChunk;

  for (int index = threadIdx.x; index < kChunk * kChunk;
       index += blockDim.x) {
    const int row = index / kChunk;
    const int col = index - row * kChunk;
    lower[index] = row < length && col < row
                       ? load_float(lower_product + base + index)
                       : 0.0f;
    inv[index] = row < length && row == col ? 1.0f : 0.0f;
    if (row >= length || col > row) {
      store_scalar(aqk + base + index, 0.0f);
    }
  }
  __syncthreads();

  for (int row = 1; row < length; ++row) {
    for (int col = threadIdx.x; col < row; col += blockDim.x) {
      float value = 0.0f;
      for (int middle = col; middle < row; ++middle) {
        value += lower[row * kChunk + middle] *
                 inv[middle * kChunk + col];
      }
      inv[row * kChunk + col] = -value;
    }
    __syncthreads();
  }
  for (int index = threadIdx.x; index < kChunk * kChunk;
       index += blockDim.x) {
    store_scalar(inverse + base + index, inv[index]);
  }
}

template <typename scalar_t>
__global__ void gdn2_v4_unpack_output_kernel(
    const scalar_t* __restrict__ packed,
    scalar_t* __restrict__ output,
    int64_t B,
    int64_t T,
    int64_t H,
    int64_t V) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = B * T * H * V;
  if (index >= total) {
    return;
  }
  const int64_t vv = index % V;
  int64_t decoded = index / V;
  const int64_t h = decoded % H;
  decoded /= H;
  const int64_t token = decoded % T;
  const int64_t batch = decoded / T;
  const int64_t chunk = token / kChunk;
  const int64_t row = token - chunk * kChunk;
  const int64_t matrix = (chunk * B + batch) * H + h;
  output[index] = packed[(matrix * kChunk + row) * V + vv];
}

// ---------------------------------------------------------------------------
// V5 fused chunk-loop helpers.
// ---------------------------------------------------------------------------
// V4 used four BMMs plus several ATen pointwise operators per chunk. V5
// pre-packs the two constant BMM operands and fuses every remaining pointwise
// stage into native kernels. The serial chunk loop therefore contains exactly
// three BMMs and two native kernels, with no per-chunk tensor allocation.

template <typename scalar_t>
__global__ void gdn2_v5_pack_state_key_kernel(
    const scalar_t* __restrict__ key_transposed,
    const float* __restrict__ chunk_decay,
    scalar_t* __restrict__ state_key_transposed,
    int64_t M,
    int64_t K) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = M * K * kChunk;
  if (index >= total) {
    return;
  }
  const int64_t decoded = index / kChunk;
  const int64_t kk = decoded % K;
  const int64_t matrix = decoded / K;
  const float decay_low =
      round_to_scalar<scalar_t>(chunk_decay[matrix * K + kk]);
  const float value = load_float(key_transposed + index) * decay_low;
  store_scalar(state_key_transposed + index,
               round_to_scalar<scalar_t>(value));
}

template <typename scalar_t>
__global__ void gdn2_v5_pack_output_lhs_kernel(
    const scalar_t* __restrict__ qg,
    const scalar_t* __restrict__ aqk,
    scalar_t* __restrict__ output_lhs,
    int64_t M,
    int64_t K) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t width = K + kChunk;
  const int64_t total = M * kChunk * width;
  if (index >= total) {
    return;
  }
  int64_t decoded = index;
  const int64_t column = decoded % width;
  decoded /= width;
  const int64_t row = decoded % kChunk;
  const int64_t matrix = decoded / kChunk;
  const int64_t qg_base = (matrix * kChunk + row) * K;
  const int64_t aqk_base = (matrix * kChunk + row) * kChunk;
  output_lhs[index] = column < K ? qg[qg_base + column]
                                 : aqk[aqk_base + column - K];
}

template <typename scalar_t>
__global__ void gdn2_v5_state_cast_kernel(
    const float* __restrict__ state,
    scalar_t* __restrict__ state_low,
    int64_t total) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < total) {
    store_scalar(state_low + index, state[index]);
  }
}

template <typename scalar_t>
__global__ void gdn2_v5_vnew_rhs_kernel(
    const scalar_t* __restrict__ state_low,
    const scalar_t* __restrict__ wy_value,
    const scalar_t* __restrict__ correction,
    scalar_t* __restrict__ vnew,
    scalar_t* __restrict__ output_rhs,
    int64_t BH,
    int64_t K,
    int64_t V) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t rows = K + kChunk;
  const int64_t total = BH * rows * V;
  if (index >= total) {
    return;
  }
  const int64_t vv = index % V;
  int64_t decoded = index / V;
  const int64_t row = decoded % rows;
  const int64_t batch_head = decoded / rows;
  if (row < K) {
    output_rhs[index] = state_low[(batch_head * K + row) * V + vv];
    return;
  }
  const int64_t token = row - K;
  const int64_t value_index = (batch_head * kChunk + token) * V + vv;
  const float value = load_float(wy_value + value_index) -
                      load_float(correction + value_index);
  const scalar_t low_value = static_cast<scalar_t>(value);
  vnew[value_index] = low_value;
  output_rhs[index] = low_value;
}

template <typename scalar_t>
__global__ void gdn2_v5_state_update_kernel(
    float* __restrict__ state,
    scalar_t* __restrict__ state_low,
    const scalar_t* __restrict__ injected,
    const float* __restrict__ chunk_decay,
    int64_t BH,
    int64_t K,
    int64_t V) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = BH * K * V;
  if (index >= total) {
    return;
  }
  const int64_t vv = index % V;
  const int64_t decoded = index / V;
  const int64_t kk = decoded % K;
  const int64_t batch_head = decoded / K;
  const float updated = state[index] * chunk_decay[batch_head * K + kk] +
                        load_float(injected + index);
  state[index] = updated;
  store_scalar(state_low + index, updated);
}

template <typename scalar_t>
__global__ void gdn2_v5_unpack_output_kernel(
    const scalar_t* __restrict__ packed,
    scalar_t* __restrict__ output,
    int64_t B,
    int64_t T,
    int64_t H,
    int64_t V,
    float scale) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = B * T * H * V;
  if (index >= total) {
    return;
  }
  const int64_t vv = index % V;
  int64_t decoded = index / V;
  const int64_t h = decoded % H;
  decoded /= H;
  const int64_t token = decoded % T;
  const int64_t batch = decoded / T;
  const int64_t chunk = token / kChunk;
  const int64_t row = token - chunk * kChunk;
  const int64_t matrix = (chunk * B + batch) * H + h;
  const float value =
      load_float(packed + (matrix * kChunk + row) * V + vv) * scale;
  store_scalar(output + index, value);
}

void check_same_device(const torch::Tensor& reference,
                       const torch::Tensor& tensor,
                       const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be on the MACA/CUDA device");
  TORCH_CHECK(tensor.device() == reference.device(), name,
              " must be on the same device as q");
}

std::vector<torch::Tensor> gdn2_maca_chunk64_bmm_v5_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor g,
    torch::Tensor b,
    torch::Tensor w,
    torch::Tensor initial_state,
    double scale) {
  TORCH_CHECK(q.is_cuda(), "q must be on the MACA/CUDA device");
  TORCH_CHECK(q.is_contiguous() && q.dim() == 4,
              "q must be contiguous [B,T,H,K]");
  TORCH_CHECK(q.scalar_type() == at::ScalarType::Half ||
                  q.scalar_type() == at::ScalarType::BFloat16,
              "chunk64_bmm_v5 supports float16 and bfloat16");
  for (const auto& named : std::vector<std::pair<const char*, torch::Tensor>>{
           {"k", k}, {"v", v}, {"g", g}, {"b", b}, {"w", w}}) {
    check_same_device(q, named.second, named.first);
    TORCH_CHECK(named.second.is_contiguous(), named.first,
                " must be contiguous");
    TORCH_CHECK(named.second.scalar_type() == q.scalar_type(), named.first,
                " must have the same dtype as q");
  }
  const int64_t B = q.size(0);
  const int64_t T = q.size(1);
  const int64_t H = q.size(2);
  const int64_t K = q.size(3);
  TORCH_CHECK(T > 0, "T must be positive");
  TORCH_CHECK(K > 0 && K <= kMaxK, "K must be in [1,256], got ", K);
  TORCH_CHECK(k.sizes() == q.sizes() && g.sizes() == q.sizes() &&
                  b.sizes() == q.sizes(),
              "k/g/b must match q shape");
  TORCH_CHECK(v.dim() == 4 && v.size(0) == B && v.size(1) == T &&
                  v.size(2) == H,
              "v must have shape [B,T,H,V]");
  TORCH_CHECK(w.sizes() == v.sizes(), "w must match v shape");
  const int64_t V = v.size(3);
  TORCH_CHECK(V > 0, "V must be positive");

  c10::cuda::CUDAGuard device_guard(q.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(q.get_device());
  const int64_t NT = (T + kChunk - 1) / kChunk;
  const int64_t BH = B * H;
  const int64_t M = NT * BH;
  const int64_t output_width = K + kChunk;
  auto state_options = q.options().dtype(at::ScalarType::Float);
  torch::Tensor state;
  const std::vector<int64_t> state_shape{B, H, K, V};
  if (initial_state.numel() == 0) {
    state = torch::zeros(state_shape, state_options);
  } else {
    check_same_device(q, initial_state, "initial_state");
    TORCH_CHECK(initial_state.is_contiguous() &&
                    initial_state.scalar_type() == at::ScalarType::Float,
                "initial_state must be contiguous float32");
    TORCH_CHECK(initial_state.sizes().vec() == state_shape,
                "chunk64_bmm_v5 initial_state must have shape [B,H,K,V]");
    state = initial_state.clone();
  }

  auto gate_cumsum = torch::empty_like(g);
  auto qg = torch::empty(std::vector<int64_t>{M, kChunk, K}, q.options());
  auto key_transposed =
      torch::empty(std::vector<int64_t>{M, K, kChunk}, q.options());
  auto erase_key = torch::empty_like(qg);
  auto x = torch::empty(std::vector<int64_t>{M, kChunk, V}, q.options());
  auto chunk_decay =
      torch::empty(std::vector<int64_t>{M, K}, state_options);

  const int64_t cumsum_items = B * NT * H * K;
  const int64_t cumsum_blocks = (cumsum_items + 255) / 256;
  const int64_t qkc_items = M * kChunk * K;
  const int64_t qkc_blocks = (qkc_items + 255) / 256;
  const int64_t x_items = M * kChunk * V;
  const int64_t x_blocks = (x_items + 255) / 256;
  for (const auto& named :
       std::vector<std::pair<const char*, int64_t>>{
           {"cumsum", cumsum_blocks}, {"pack_qkc", qkc_blocks},
           {"pack_x", x_blocks}, {"matrix", M}}) {
    TORCH_CHECK(named.second > 0 &&
                    named.second <= static_cast<int64_t>(UINT32_MAX),
                named.first, " launch grid is too large");
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, q.scalar_type(),
      "gdn2_maca_v5_pack", [&] {
        gdn2_chunk64_cumsum_kernel<scalar_t>
            <<<static_cast<unsigned int>(cumsum_blocks), 256, 0, stream>>>(
                g.data_ptr<scalar_t>(), gate_cumsum.data_ptr<scalar_t>(), B,
                T, H, K, NT);
        gdn2_v4_pack_qkc_kernel<scalar_t>
            <<<static_cast<unsigned int>(qkc_blocks), 256, 0, stream>>>(
                q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(),
                b.data_ptr<scalar_t>(), gate_cumsum.data_ptr<scalar_t>(),
                qg.data_ptr<scalar_t>(), key_transposed.data_ptr<scalar_t>(),
                erase_key.data_ptr<scalar_t>(), chunk_decay.data_ptr<float>(),
                B, T, H, K, NT, M);
        gdn2_v4_pack_x_kernel<scalar_t>
            <<<static_cast<unsigned int>(x_blocks), 256, 0, stream>>>(
                v.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(),
                x.data_ptr<scalar_t>(), B, T, H, V, M);
      });

  auto aqk = at::bmm(qg, key_transposed);
  auto lower_product = at::bmm(erase_key, key_transposed);
  auto inverse = torch::empty_like(lower_product);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, q.scalar_type(),
      "gdn2_maca_v5_solve", [&] {
        gdn2_v4_solve_kernel<scalar_t>
            <<<static_cast<unsigned int>(M), 256, 0, stream>>>(
                lower_product.data_ptr<scalar_t>(), aqk.data_ptr<scalar_t>(),
                inverse.data_ptr<scalar_t>(), B, T, H, M);
      });
  auto wy_key = at::bmm(inverse, erase_key);
  auto wy_value = at::bmm(inverse, x);

  // Pack operands that do not change across the serial chunk recurrence.
  auto state_key_transposed = torch::empty_like(key_transposed);
  auto output_lhs = torch::empty(
      std::vector<int64_t>{M, kChunk, output_width}, q.options());
  const int64_t state_key_blocks = (M * K * kChunk + 255) / 256;
  const int64_t output_lhs_blocks =
      (M * kChunk * output_width + 255) / 256;
  TORCH_CHECK(state_key_blocks <= static_cast<int64_t>(UINT32_MAX) &&
                  output_lhs_blocks <= static_cast<int64_t>(UINT32_MAX),
              "V5 static pack launch grid is too large");
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, q.scalar_type(),
      "gdn2_maca_v5_static_pack", [&] {
        gdn2_v5_pack_state_key_kernel<scalar_t>
            <<<static_cast<unsigned int>(state_key_blocks), 256, 0, stream>>>(
                key_transposed.data_ptr<scalar_t>(),
                chunk_decay.data_ptr<float>(),
                state_key_transposed.data_ptr<scalar_t>(), M, K);
        gdn2_v5_pack_output_lhs_kernel<scalar_t>
            <<<static_cast<unsigned int>(output_lhs_blocks), 256, 0, stream>>>(
                qg.data_ptr<scalar_t>(), aqk.data_ptr<scalar_t>(),
                output_lhs.data_ptr<scalar_t>(), M, K);
      });

  // Every workspace below is allocated exactly once and reused by all chunks.
  auto state_matrix = state.view(std::vector<int64_t>{BH, K, V});
  auto state_low = torch::empty(std::vector<int64_t>{BH, K, V}, q.options());
  auto correction =
      torch::empty(std::vector<int64_t>{BH, kChunk, V}, q.options());
  auto vnew = torch::empty_like(correction);
  auto output_rhs = torch::empty(
      std::vector<int64_t>{BH, output_width, V}, q.options());
  auto injected = torch::empty(std::vector<int64_t>{BH, K, V}, q.options());
  auto packed_output =
      torch::empty(std::vector<int64_t>{M, kChunk, V}, q.options());

  const int64_t state_items = BH * K * V;
  const int64_t state_blocks = (state_items + 255) / 256;
  const int64_t rhs_items = BH * output_width * V;
  const int64_t rhs_blocks = (rhs_items + 255) / 256;
  TORCH_CHECK(state_blocks > 0 && rhs_blocks > 0 &&
                  state_blocks <= static_cast<int64_t>(UINT32_MAX) &&
                  rhs_blocks <= static_cast<int64_t>(UINT32_MAX),
              "V5 recurrent launch grid is too large");
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, q.scalar_type(),
      "gdn2_maca_v5_initial_state_cast", [&] {
        gdn2_v5_state_cast_kernel<scalar_t>
            <<<static_cast<unsigned int>(state_blocks), 256, 0, stream>>>(
                state_matrix.data_ptr<float>(), state_low.data_ptr<scalar_t>(),
                state_items);
      });

  for (int64_t chunk = 0; chunk < NT; ++chunk) {
    const int64_t offset = chunk * BH;
    auto wy_key_chunk = wy_key.narrow(0, offset, BH);
    auto wy_value_chunk = wy_value.narrow(0, offset, BH);
    auto output_lhs_chunk = output_lhs.narrow(0, offset, BH);
    auto state_key_chunk = state_key_transposed.narrow(0, offset, BH);
    auto decay_chunk = chunk_decay.narrow(0, offset, BH);
    auto output_chunk = packed_output.narrow(0, offset, BH);

    // 1) correction=W@S. The fused kernel then computes v_new=U-correction
    // and packs [S;v_new], eliminating V4's casts, subtraction and concat.
    at::bmm_out(correction, wy_key_chunk, state_low);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, q.scalar_type(),
        "gdn2_maca_v5_vnew_rhs", [&] {
          gdn2_v5_vnew_rhs_kernel<scalar_t>
              <<<static_cast<unsigned int>(rhs_blocks), 256, 0, stream>>>(
                  state_low.data_ptr<scalar_t>(),
                  wy_value_chunk.data_ptr<scalar_t>(),
                  correction.data_ptr<scalar_t>(), vnew.data_ptr<scalar_t>(),
                  output_rhs.data_ptr<scalar_t>(), BH, K, V);
        });

    // 2) [QG,Aqk]@[S;v_new] combines V4's inter and intra BMMs.
    at::bmm_out(output_chunk, output_lhs_chunk, output_rhs);

    // 3) Kg^T@v_new. The update kernel fuses decay, FP32 accumulation and
    // the low-precision state materialization needed by the next chunk.
    at::bmm_out(injected, state_key_chunk, vnew);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, q.scalar_type(),
        "gdn2_maca_v5_state_update", [&] {
          gdn2_v5_state_update_kernel<scalar_t>
              <<<static_cast<unsigned int>(state_blocks), 256, 0, stream>>>(
                  state_matrix.data_ptr<float>(), state_low.data_ptr<scalar_t>(),
                  injected.data_ptr<scalar_t>(), decay_chunk.data_ptr<float>(),
                  BH, K, V);
        });
  }

  auto out = torch::empty_like(v);
  const int64_t output_items = B * T * H * V;
  const int64_t output_blocks = (output_items + 255) / 256;
  TORCH_CHECK(output_blocks <= static_cast<int64_t>(UINT32_MAX),
              "output launch grid is too large");
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, q.scalar_type(),
      "gdn2_maca_v5_unpack", [&] {
        gdn2_v5_unpack_output_kernel<scalar_t>
            <<<static_cast<unsigned int>(output_blocks), 256, 0, stream>>>(
                packed_output.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(),
                B, T, H, V, static_cast<float>(scale));
      });

  const cudaError_t error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess, "gdn2 chunk64_bmm_v5 launch failed: ",
              cudaGetErrorString(error));
  return {out, state};
}

std::vector<torch::Tensor> gdn2_maca_chunk64_bmm_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor g,
    torch::Tensor b,
    torch::Tensor w,
    torch::Tensor initial_state,
    double scale) {
  TORCH_CHECK(q.is_cuda(), "q must be on the MACA/CUDA device");
  TORCH_CHECK(q.is_contiguous() && q.dim() == 4,
              "q must be contiguous [B,T,H,K]");
  TORCH_CHECK(q.scalar_type() == at::ScalarType::Half ||
                  q.scalar_type() == at::ScalarType::BFloat16,
              "chunk64_bmm supports float16 and bfloat16");
  for (const auto& named : std::vector<std::pair<const char*, torch::Tensor>>{
           {"k", k}, {"v", v}, {"g", g}, {"b", b}, {"w", w}}) {
    check_same_device(q, named.second, named.first);
    TORCH_CHECK(named.second.is_contiguous(), named.first,
                " must be contiguous");
    TORCH_CHECK(named.second.scalar_type() == q.scalar_type(), named.first,
                " must have the same dtype as q");
  }
  const int64_t B = q.size(0);
  const int64_t T = q.size(1);
  const int64_t H = q.size(2);
  const int64_t K = q.size(3);
  TORCH_CHECK(T > 0, "T must be positive");
  TORCH_CHECK(K > 0 && K <= kMaxK, "K must be in [1,256], got ", K);
  TORCH_CHECK(k.sizes() == q.sizes() && g.sizes() == q.sizes() &&
                  b.sizes() == q.sizes(),
              "k/g/b must match q shape");
  TORCH_CHECK(v.dim() == 4 && v.size(0) == B && v.size(1) == T &&
                  v.size(2) == H,
              "v must have shape [B,T,H,V]");
  TORCH_CHECK(w.sizes() == v.sizes(), "w must match v shape");
  const int64_t V = v.size(3);
  TORCH_CHECK(V > 0, "V must be positive");

  c10::cuda::CUDAGuard device_guard(q.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(q.get_device());
  const int64_t NT = (T + kChunk - 1) / kChunk;
  const int64_t BH = B * H;
  const int64_t M = NT * BH;
  auto state_options = q.options().dtype(at::ScalarType::Float);
  torch::Tensor state;
  const std::vector<int64_t> state_shape{B, H, K, V};
  if (initial_state.numel() == 0) {
    state = torch::zeros(state_shape, state_options);
  } else {
    check_same_device(q, initial_state, "initial_state");
    TORCH_CHECK(initial_state.is_contiguous() &&
                    initial_state.scalar_type() == at::ScalarType::Float,
                "initial_state must be contiguous float32");
    TORCH_CHECK(initial_state.sizes().vec() == state_shape,
                "chunk64_bmm initial_state must have shape [B,H,K,V]");
    state = initial_state.clone();
  }

  auto gate_cumsum = torch::empty_like(g);
  auto qg = torch::empty(std::vector<int64_t>{M, kChunk, K}, q.options());
  auto key_transposed =
      torch::empty(std::vector<int64_t>{M, K, kChunk}, q.options());
  auto erase_key = torch::empty_like(qg);
  auto x = torch::empty(std::vector<int64_t>{M, kChunk, V}, q.options());
  auto chunk_decay =
      torch::empty(std::vector<int64_t>{M, K}, state_options);

  const int64_t cumsum_items = B * NT * H * K;
  const int64_t cumsum_blocks = (cumsum_items + 255) / 256;
  const int64_t qkc_items = M * kChunk * K;
  const int64_t qkc_blocks = (qkc_items + 255) / 256;
  const int64_t x_items = M * kChunk * V;
  const int64_t x_blocks = (x_items + 255) / 256;
  for (const auto& named :
       std::vector<std::pair<const char*, int64_t>>{
           {"cumsum", cumsum_blocks}, {"pack_qkc", qkc_blocks},
           {"pack_x", x_blocks}, {"matrix", M}}) {
    TORCH_CHECK(named.second > 0 &&
                    named.second <= static_cast<int64_t>(UINT32_MAX),
                named.first, " launch grid is too large");
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, q.scalar_type(),
      "gdn2_maca_v4_pack", [&] {
        gdn2_chunk64_cumsum_kernel<scalar_t>
            <<<static_cast<unsigned int>(cumsum_blocks), 256, 0, stream>>>(
                g.data_ptr<scalar_t>(), gate_cumsum.data_ptr<scalar_t>(), B,
                T, H, K, NT);
        gdn2_v4_pack_qkc_kernel<scalar_t>
            <<<static_cast<unsigned int>(qkc_blocks), 256, 0, stream>>>(
                q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(),
                b.data_ptr<scalar_t>(), gate_cumsum.data_ptr<scalar_t>(),
                qg.data_ptr<scalar_t>(), key_transposed.data_ptr<scalar_t>(),
                erase_key.data_ptr<scalar_t>(), chunk_decay.data_ptr<float>(),
                B, T, H, K, NT, M);
        gdn2_v4_pack_x_kernel<scalar_t>
            <<<static_cast<unsigned int>(x_blocks), 256, 0, stream>>>(
                v.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(),
                x.data_ptr<scalar_t>(), B, T, H, V, M);
      });

  // The following BMMs dispatch to MACA's dense matrix backend and are the V4
  // matrix-core path. Shapes are [M,64,K]@[M,K,64], [M,64,64]@[M,64,D].
  auto aqk = at::bmm(qg, key_transposed);
  auto lower_product = at::bmm(erase_key, key_transposed);
  auto inverse = torch::empty_like(lower_product);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, q.scalar_type(),
      "gdn2_maca_v4_solve", [&] {
        gdn2_v4_solve_kernel<scalar_t>
            <<<static_cast<unsigned int>(M), 256, 0, stream>>>(
                lower_product.data_ptr<scalar_t>(), aqk.data_ptr<scalar_t>(),
                inverse.data_ptr<scalar_t>(), B, T, H, M);
      });
  auto wy_key = at::bmm(inverse, erase_key);
  auto wy_value = at::bmm(inverse, x);

  auto packed_output =
      torch::empty(std::vector<int64_t>{M, kChunk, V}, q.options());
  auto state_matrix = state.view(std::vector<int64_t>{BH, K, V});
  for (int64_t chunk = 0; chunk < NT; ++chunk) {
    const int64_t offset = chunk * BH;
    auto qg_chunk = qg.narrow(0, offset, BH);
    auto key_t_chunk = key_transposed.narrow(0, offset, BH);
    auto aqk_chunk = aqk.narrow(0, offset, BH);
    auto wy_key_chunk = wy_key.narrow(0, offset, BH);
    auto wy_value_chunk = wy_value.narrow(0, offset, BH);
    auto decay_chunk = chunk_decay.narrow(0, offset, BH);

    auto state_low = state_matrix.to(q.scalar_type());
    auto vnew = wy_value_chunk - at::bmm(wy_key_chunk, state_low);
    auto inter = at::bmm(qg_chunk, state_low).to(at::ScalarType::Float);
    auto intra = at::bmm(aqk_chunk, vnew).to(at::ScalarType::Float);
    auto output_chunk = (inter + intra) * static_cast<float>(scale);
    packed_output.narrow(0, offset, BH).copy_(output_chunk);

    auto decay_low = decay_chunk.to(q.scalar_type());
    auto kg_transposed = key_t_chunk * decay_low.unsqueeze(-1);
    auto injected =
        at::bmm(kg_transposed, vnew).to(at::ScalarType::Float);
    state_matrix = state_matrix * decay_chunk.unsqueeze(-1) + injected;
  }
  state = state_matrix.view(state_shape);

  auto out = torch::empty_like(v);
  const int64_t output_items = B * T * H * V;
  const int64_t output_blocks = (output_items + 255) / 256;
  TORCH_CHECK(output_blocks <= static_cast<int64_t>(UINT32_MAX),
              "output launch grid is too large");
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, q.scalar_type(),
      "gdn2_maca_v4_unpack", [&] {
        gdn2_v4_unpack_output_kernel<scalar_t>
            <<<static_cast<unsigned int>(output_blocks), 256, 0, stream>>>(
                packed_output.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(),
                B, T, H, V);
      });

  const cudaError_t error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess, "gdn2 chunk64_bmm launch failed: ",
              cudaGetErrorString(error));
  return {out, state};
}

std::vector<torch::Tensor> gdn2_maca_chunk64_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor g,
    torch::Tensor b,
    torch::Tensor w,
    torch::Tensor initial_state,
    double scale) {
  TORCH_CHECK(q.is_cuda(), "q must be on the MACA/CUDA device");
  TORCH_CHECK(q.is_contiguous() && q.dim() == 4,
              "q must be contiguous [B,T,H,K]");
  TORCH_CHECK(q.scalar_type() == at::ScalarType::Half ||
                  q.scalar_type() == at::ScalarType::BFloat16,
              "chunk64 supports float16 and bfloat16");
  for (const auto& named : std::vector<std::pair<const char*, torch::Tensor>>{
           {"k", k}, {"v", v}, {"g", g}, {"b", b}, {"w", w}}) {
    check_same_device(q, named.second, named.first);
    TORCH_CHECK(named.second.is_contiguous(), named.first,
                " must be contiguous");
    TORCH_CHECK(named.second.scalar_type() == q.scalar_type(), named.first,
                " must have the same dtype as q");
  }

  const int64_t B = q.size(0);
  const int64_t T = q.size(1);
  const int64_t H = q.size(2);
  const int64_t K = q.size(3);
  TORCH_CHECK(T > 0, "T must be positive");
  TORCH_CHECK(K > 0 && K <= kMaxK, "K must be in [1,256], got ", K);
  TORCH_CHECK(k.sizes() == q.sizes() && g.sizes() == q.sizes() &&
                  b.sizes() == q.sizes(),
              "k/g/b must match q shape");
  TORCH_CHECK(v.dim() == 4 && v.size(0) == B && v.size(1) == T &&
                  v.size(2) == H,
              "v must have shape [B,T,H,V]");
  TORCH_CHECK(w.sizes() == v.sizes(), "w must match v shape");
  const int64_t V = v.size(3);
  TORCH_CHECK(V > 0, "V must be positive");

  c10::cuda::CUDAGuard device_guard(q.device());
  const int64_t NT = (T + kChunk - 1) / kChunk;
  auto state_options = q.options().dtype(at::ScalarType::Float);
  torch::Tensor state;
  const std::vector<int64_t> state_shape{B, H, K, V};
  if (initial_state.numel() == 0) {
    state = torch::zeros(state_shape, state_options);
  } else {
    check_same_device(q, initial_state, "initial_state");
    TORCH_CHECK(initial_state.is_contiguous() &&
                    initial_state.scalar_type() == at::ScalarType::Float,
                "initial_state must be contiguous float32");
    TORCH_CHECK(initial_state.sizes().vec() == state_shape,
                "chunk64 initial_state must have shape [B,H,K,V]");
    state = initial_state.clone();
  }

  auto gate_cumsum = torch::empty_like(g);
  auto inverse = torch::empty(
      std::vector<int64_t>{B, NT, H, kChunk, kChunk}, q.options());
  auto aqk = torch::empty_like(inverse);
  auto wy_key = torch::empty_like(k);
  auto wy_value = torch::empty_like(v);
  auto out = torch::empty_like(v);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(q.get_device());

  const int64_t cumsum_items = B * NT * H * K;
  const int64_t cumsum_blocks = (cumsum_items + 255) / 256;
  const int64_t matrix_blocks = B * NT * H;
  const int64_t key_tiles = (kChunk * K + 255) / 256;
  const int64_t value_tiles = (kChunk * V + 255) / 256;
  const int64_t key_blocks = matrix_blocks * key_tiles;
  const int64_t value_blocks = matrix_blocks * value_tiles;
  const int64_t BV = K > 128 ? 32 : 64;
  const int64_t v_tiles = (V + BV - 1) / BV;
  const int64_t state_blocks = B * H * v_tiles;
  for (const auto& named :
       std::vector<std::pair<const char*, int64_t>>{
           {"cumsum", cumsum_blocks}, {"matrix", matrix_blocks},
           {"wy_key", key_blocks}, {"wy_value", value_blocks},
           {"state", state_blocks}}) {
    TORCH_CHECK(named.second > 0 &&
                    named.second <= static_cast<int64_t>(UINT32_MAX),
                named.first, " launch grid is too large");
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, q.scalar_type(),
      "gdn2_maca_chunk64", [&] {
        gdn2_chunk64_cumsum_kernel<scalar_t>
            <<<static_cast<unsigned int>(cumsum_blocks), 256, 0, stream>>>(
                g.data_ptr<scalar_t>(), gate_cumsum.data_ptr<scalar_t>(), B,
                T, H, K, NT);
        gdn2_chunk64_build_matrices_kernel<scalar_t>
            <<<static_cast<unsigned int>(matrix_blocks), 256, 0, stream>>>(
                q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(),
                gate_cumsum.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(),
                inverse.data_ptr<scalar_t>(), aqk.data_ptr<scalar_t>(), T, H,
                K, NT, static_cast<float>(scale));
        gdn2_chunk64_wy_kernel<scalar_t, true>
            <<<static_cast<unsigned int>(key_blocks), 256, 0, stream>>>(
                k.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(),
                gate_cumsum.data_ptr<scalar_t>(),
                inverse.data_ptr<scalar_t>(), wy_key.data_ptr<scalar_t>(), B,
                T, H, K, NT, key_tiles);
        gdn2_chunk64_wy_kernel<scalar_t, false>
            <<<static_cast<unsigned int>(value_blocks), 256, 0, stream>>>(
                v.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(), nullptr,
                inverse.data_ptr<scalar_t>(), wy_value.data_ptr<scalar_t>(), B,
                T, H, V, NT, value_tiles);
        const size_t shared_bytes =
            static_cast<size_t>(K * BV + kChunk * BV) * sizeof(float);
        gdn2_chunk64_state_output_kernel<scalar_t>
            <<<static_cast<unsigned int>(state_blocks), kThreads, shared_bytes,
               stream>>>(
                q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(),
                gate_cumsum.data_ptr<scalar_t>(),
                wy_key.data_ptr<scalar_t>(), wy_value.data_ptr<scalar_t>(),
                aqk.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(),
                state.data_ptr<float>(), B, T, H, K, V, NT, BV,
                static_cast<float>(scale));
      });

  const cudaError_t error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess, "gdn2 chunk64 kernel launch failed: ",
              cudaGetErrorString(error));
  return {out, state};
}

std::vector<torch::Tensor> gdn2_maca_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor g,
    torch::Tensor b,
    torch::Tensor w,
    torch::Tensor initial_state,
    torch::Tensor cu_seqlens,
    torch::Tensor A_log,
    torch::Tensor dt_bias,
    double scale,
    bool output_final_state,
    bool state_v_first,
    bool use_qk_l2norm,
    bool use_gate_in_kernel,
    bool safe_gate,
    double lower_bound,
    bool has_lower_bound,
    bool use_persistent_state) {
  TORCH_CHECK(q.is_cuda(), "q must be on the MACA/CUDA device");
  TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
  TORCH_CHECK(q.dim() == 4, "q must have shape [B,T,H,K]");
  TORCH_CHECK(q.scalar_type() == at::ScalarType::Half ||
                  q.scalar_type() == at::ScalarType::BFloat16,
              "only float16 and bfloat16 are supported");

  for (const auto& named : std::vector<std::pair<const char*, torch::Tensor>>{
           {"k", k}, {"v", v}, {"g", g}, {"b", b}, {"w", w}}) {
    check_same_device(q, named.second, named.first);
    TORCH_CHECK(named.second.is_contiguous(), named.first,
                " must be contiguous");
    TORCH_CHECK(named.second.scalar_type() == q.scalar_type(), named.first,
                " must have the same dtype as q");
  }

  const int64_t B = q.size(0);
  const int64_t T = q.size(1);
  const int64_t H = q.size(2);
  const int64_t K = q.size(3);
  TORCH_CHECK(K > 0 && K <= kMaxK, "K must be in [1,256], got ", K);
  TORCH_CHECK(k.sizes() == q.sizes(), "k must match q shape");
  TORCH_CHECK(g.sizes() == q.sizes(), "g must match q shape");
  TORCH_CHECK(b.sizes() == q.sizes(), "b must match q shape");
  TORCH_CHECK(v.dim() == 4 && v.size(0) == B && v.size(1) == T &&
                  v.size(2) == H,
              "v must have shape [B,T,H,V]");
  TORCH_CHECK(w.sizes() == v.sizes(), "w must match v shape");
  const int64_t V = v.size(3);
  TORCH_CHECK(V > 0, "V must be positive");

  const bool is_varlen = cu_seqlens.numel() != 0;
  int64_t N = B;
  if (is_varlen) {
    check_same_device(q, cu_seqlens, "cu_seqlens");
    TORCH_CHECK(cu_seqlens.is_contiguous(), "cu_seqlens must be contiguous");
    TORCH_CHECK(cu_seqlens.scalar_type() == at::ScalarType::Long,
                "cu_seqlens must be int64");
    TORCH_CHECK(cu_seqlens.dim() == 1 && cu_seqlens.numel() >= 2,
                "cu_seqlens must have shape [N+1]");
    TORCH_CHECK(B == 1, "packed varlen input requires B=1");
    N = cu_seqlens.numel() - 1;
  }

  if (use_gate_in_kernel) {
    check_same_device(q, A_log, "A_log");
    TORCH_CHECK(A_log.is_contiguous() &&
                    A_log.scalar_type() == at::ScalarType::Float &&
                    A_log.numel() == H,
                "A_log must be contiguous float32 with H elements");
    if (dt_bias.numel() != 0) {
      check_same_device(q, dt_bias, "dt_bias");
      TORCH_CHECK(dt_bias.is_contiguous() &&
                      dt_bias.scalar_type() == at::ScalarType::Float &&
                      dt_bias.numel() == H * K,
                  "dt_bias must be contiguous float32 with H*K elements");
    }
  }
  if (safe_gate && use_gate_in_kernel) {
    TORCH_CHECK(has_lower_bound && lower_bound >= -5.0 && lower_bound < 0.0,
                "safe_gate requires lower_bound in [-5,0)");
  }

  c10::cuda::CUDAGuard device_guard(q.device());
  auto state_options = q.options().dtype(at::ScalarType::Float);
  torch::Tensor state;
  const std::vector<int64_t> state_shape =
      state_v_first ? std::vector<int64_t>{N, H, V, K}
                    : std::vector<int64_t>{N, H, K, V};
  if (initial_state.numel() == 0) {
    state = torch::zeros(state_shape, state_options);
  } else {
    check_same_device(q, initial_state, "initial_state");
    TORCH_CHECK(initial_state.is_contiguous(),
                "initial_state must be contiguous");
    TORCH_CHECK(initial_state.scalar_type() == at::ScalarType::Float,
                "initial_state must be float32");
    TORCH_CHECK(initial_state.sizes().vec() == state_shape,
                "initial_state has the wrong shape");
    state = initial_state.clone();
  }

  auto out = torch::empty_like(v);
  const int64_t BV = use_persistent_state && K > 128 ? 32 : 64;
  const int64_t v_tiles = (V + BV - 1) / BV;
  const int64_t blocks64 = N * H * v_tiles;
  TORCH_CHECK(blocks64 <= static_cast<int64_t>(UINT32_MAX),
              "launch grid is too large");
  const dim3 grid(static_cast<unsigned int>(blocks64));
  const dim3 block(kThreads);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(q.get_device());

  const int64_t* cu_ptr =
      is_varlen ? cu_seqlens.data_ptr<int64_t>() : nullptr;
  const float* A_ptr =
      use_gate_in_kernel ? A_log.data_ptr<float>() : nullptr;
  const bool has_dt_bias = use_gate_in_kernel && dt_bias.numel() != 0;
  const float* dt_ptr = has_dt_bias ? dt_bias.data_ptr<float>() : nullptr;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, q.scalar_type(),
      "gdn2_maca_recurrent", [&] {
        if (use_persistent_state) {
          const size_t shared_bytes =
              static_cast<size_t>(K * BV + 4 * K + 2) * sizeof(float);
          gdn2_persistent_tile_kernel<scalar_t>
              <<<grid, block, shared_bytes, stream>>>(
                  q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(),
                  v.data_ptr<scalar_t>(), g.data_ptr<scalar_t>(),
                  b.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(), A_ptr,
                  dt_ptr, cu_ptr, out.data_ptr<scalar_t>(),
                  state.data_ptr<float>(), B, T, H, K, V, N, BV,
                  static_cast<float>(scale), static_cast<float>(lower_bound),
                  is_varlen, state_v_first, use_qk_l2norm,
                  use_gate_in_kernel, has_dt_bias, has_lower_bound);
        } else {
          gdn2_recurrent_kernel<scalar_t><<<grid, block, 0, stream>>>(
              q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(),
              v.data_ptr<scalar_t>(), g.data_ptr<scalar_t>(),
              b.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(), A_ptr, dt_ptr,
              cu_ptr, out.data_ptr<scalar_t>(), state.data_ptr<float>(), B, T,
              H, K, V, N, static_cast<float>(scale),
              static_cast<float>(lower_bound), is_varlen, state_v_first,
              use_qk_l2norm, use_gate_in_kernel, has_dt_bias,
              has_lower_bound);
        }
      });

  const cudaError_t error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess, "gdn2_maca kernel launch failed: ",
              cudaGetErrorString(error));

  // The state is always materialized because the recurrence needs it.  The
  // Python wrapper discards it when output_final_state is false.
  (void)output_final_state;
  return {out, state};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("forward", &gdn2_maca_forward,
             "Native MACA GDN2 recurrent forward");
  module.def("forward_chunk64", &gdn2_maca_chunk64_forward,
             "Native MACA GDN2 BT=64 chunkwise forward");
  module.def("forward_chunk64_bmm", &gdn2_maca_chunk64_bmm_forward,
             "Native MACA GDN2 BT=64 matrix-core BMM forward");
  module.def("forward_chunk64_bmm_v5", &gdn2_maca_chunk64_bmm_v5_forward,
             "Native MACA GDN2 V5 fused three-BMM forward");
}
