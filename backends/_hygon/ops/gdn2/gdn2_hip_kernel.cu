#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPException.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <tuple>

namespace {

constexpr int MAX_K = 256;
constexpr int THREADS = 256;
constexpr int WAVE_SIZE = 64;
constexpr int WAVES_PER_BLOCK = 4;

__device__ __forceinline__ float wave_sum(float x) {
  #pragma unroll
  for (int offset = WAVE_SIZE / 2; offset > 0; offset >>= 1) {
    x += __shfl_down(x, offset, WAVE_SIZE);
  }
  return x;
}

template <int GROUP_SIZE>
__device__ __forceinline__ float subgroup_sum(float x) {
  #pragma unroll
  for (int offset = GROUP_SIZE / 2; offset > 0; offset >>= 1) {
    x += __shfl_down(x, offset, GROUP_SIZE);
  }
  return x;
}

template <typename scalar_t>
__global__ __launch_bounds__(THREADS)
void gdn2_decay_precompute_kernel(
    const scalar_t* __restrict__ g,
    float* __restrict__ decay,
    int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
  if (i < n) decay[i] = expf(static_cast<float>(g[i]));
}

// Stage 1 of the BT=64 Chunk/WY path.  Each lane owns one K coordinate and
// scans only the 64 tokens inside its chunk.  Chunks, heads and batches are
// independent, exposing T/64-way parallelism while preserving FP32 cumsum
// semantics.  Output stays in natural-log base; later kernels may use expf
// directly or multiply by log2(e) for exp2-based matrix kernels.
template <typename scalar_t, int KSIZE, int BT>
__global__ __launch_bounds__(THREADS)
void gdn2_chunk_cumsum_kernel(
    const scalar_t* __restrict__ g,
    float* __restrict__ g_cumsum,
    int B, int T, int H, int chunks) {
  const int chunk = blockIdx.x;
  const int h = blockIdx.y;
  const int bidx = blockIdx.z;
  const int ki = threadIdx.x;
  if (ki >= KSIZE || chunk >= chunks) return;

  const int begin = chunk * BT;
  const int end = min(begin + BT, T);
  float acc = 0.0f;
  #pragma unroll
  for (int ti = 0; ti < BT; ++ti) {
    const int t = begin + ti;
    if (t < end) {
      const int64_t index =
          ((static_cast<int64_t>(bidx) * T + t) * H + h) * KSIZE + ki;
      acc += static_cast<float>(g[index]);
      g_cumsum[index] = acc;
    }
  }
}

// Stage 2: fuse all gate-scaled K-side factors needed by the intra-chunk
// matrices.  FP32 outputs are intentional for the initial correctness path;
// after Aqk/Akk is validated this can be specialized to packed FP16/BF16.
template <typename scalar_t>
__global__ __launch_bounds__(THREADS)
void gdn2_chunk_factors_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ erase,
    const float* __restrict__ g_cumsum,
    float* __restrict__ qg,
    float* __restrict__ kn,
    float* __restrict__ ke,
    int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
  if (i >= n) return;
  const float gv = g_cumsum[i];
  const float ep = expf(gv);
  const float em = 1.0f / ep;
  const float qv = static_cast<float>(q[i]);
  const float kv = static_cast<float>(k[i]);
  const float bv = static_cast<float>(erase[i]);
  qg[i] = qv * ep;
  kn[i] = kv * em;
  ke[i] = kv * bv * ep;
}

template <int KSIZE, int BT>
__global__ __launch_bounds__(THREADS)
void gdn2_chunk_scores_kernel(
    const float* __restrict__ qg,
    const float* __restrict__ kn,
    const float* __restrict__ ke,
    float* __restrict__ Aqk,
    float* __restrict__ Akk,
    int B, int T, int H, int chunks,
    float scale) {
  const int64_t matrix = blockIdx.x;
  const int pair = blockIdx.y * blockDim.x + threadIdx.x;
  if (pair >= BT * BT) return;
  const int row = pair / BT;
  const int col = pair - row * BT;
  const int chunk = matrix % chunks;
  const int64_t bh = matrix / chunks;
  const int h = bh % H;
  const int bidx = bh / H;
  const int tr = chunk * BT + row;
  const int tc = chunk * BT + col;
  const int64_t out_index = matrix * BT * BT + pair;

  if (tr >= T || tc >= T || col > row) {
    Aqk[out_index] = 0.0f;
    Akk[out_index] = 0.0f;
    return;
  }

  const int64_t qbase =
      ((static_cast<int64_t>(bidx) * T + tr) * H + h) * KSIZE;
  const int64_t kbase =
      ((static_cast<int64_t>(bidx) * T + tc) * H + h) * KSIZE;
  float aq = 0.0f;
  float ak = 0.0f;
  #pragma unroll 4
  for (int ki = 0; ki < KSIZE; ++ki) {
    const float nk = kn[kbase + ki];
    aq = fmaf(qg[qbase + ki], nk, aq);
    ak = fmaf(ke[qbase + ki], nk, ak);
  }
  Aqk[out_index] = aq * scale;
  Akk[out_index] = (col < row) ? ak : 0.0f;
}

template <int KSIZE, int BT, int TILE>
__global__ __launch_bounds__(TILE * TILE)
void gdn2_chunk_scores_tiled_kernel(
    const float* __restrict__ qg,
    const float* __restrict__ kn,
    const float* __restrict__ ke,
    float* __restrict__ Aqk,
    float* __restrict__ Akk,
    int B, int T, int H, int chunks,
    float scale) {
  const int64_t matrix = blockIdx.x;
  constexpr int TILES = BT / TILE;
  const int tile_row = blockIdx.y / TILES;
  const int tile_col = blockIdx.y - tile_row * TILES;
  const int row = tile_row * TILE + threadIdx.y;
  const int col = tile_col * TILE + threadIdx.x;
  const int chunk = matrix % chunks;
  const int64_t bh = matrix / chunks;
  const int h = bh % H;
  const int bidx = bh / H;
  const int tr = chunk * BT + row;
  const int tc = chunk * BT + col;
  const bool valid = tr < T && tc < T && col <= row;

  __shared__ float sq[TILE][TILE];
  __shared__ float se[TILE][TILE];
  __shared__ float sk[TILE][TILE];
  float aq = 0.0f;
  float ak = 0.0f;

  #pragma unroll 1
  for (int kb = 0; kb < KSIZE; kb += TILE) {
    const int kval = kb + threadIdx.x;
    const int64_t qindex =
        ((static_cast<int64_t>(bidx) * T + min(tr, T - 1)) * H + h) *
        KSIZE + kval;
    const int kn_k = kb + threadIdx.x;
    const int64_t kindex =
        ((static_cast<int64_t>(bidx) * T + min(chunk * BT +
          tile_col * TILE + static_cast<int>(threadIdx.y), T - 1)) * H + h) *
        KSIZE + kn_k;
    sq[threadIdx.y][threadIdx.x] = tr < T ? qg[qindex] : 0.0f;
    se[threadIdx.y][threadIdx.x] = tr < T ? ke[qindex] : 0.0f;
    sk[threadIdx.y][threadIdx.x] =
        (chunk * BT + tile_col * TILE + threadIdx.y < T) ? kn[kindex] : 0.0f;
    __syncthreads();

    if (valid) {
      #pragma unroll
      for (int kk = 0; kk < TILE; ++kk) {
        const float nk = sk[threadIdx.x][kk];
        aq = fmaf(sq[threadIdx.y][kk], nk, aq);
        ak = fmaf(se[threadIdx.y][kk], nk, ak);
      }
    }
    __syncthreads();
  }

  const int64_t out_index = matrix * BT * BT + row * BT + col;
  if (valid) {
    Aqk[out_index] = aq * scale;
    Akk[out_index] = (col < row) ? ak : 0.0f;
  } else {
    Aqk[out_index] = 0.0f;
    Akk[out_index] = 0.0f;
  }
}

template <int BT>
__global__ __launch_bounds__(THREADS)
void gdn2_chunk_solve_kernel(
    const float* __restrict__ Akk,
    const float* __restrict__ rhs,
    float* __restrict__ out,
    int B, int T, int H, int D, int chunks) {
  const int64_t matrix = blockIdx.x;
  const int d = blockIdx.y * blockDim.x + threadIdx.x;
  if (d >= D) return;
  const int chunk = matrix % chunks;
  const int64_t bh = matrix / chunks;
  const int h = bh % H;
  const int bidx = bh / H;
  const int begin = chunk * BT;
  const int valid = min(BT, T - begin);
  const int64_t abase = matrix * BT * BT;

  // Akk is strictly lower triangular, so diag(I+Akk)=1 and forward
  // substitution needs no division.  Global output is used as the per-D
  // recurrence buffer in this correctness-first kernel; the optimized
  // version will keep 8-row panels in registers/LDS.
  #pragma unroll 1
  for (int row = 0; row < valid; ++row) {
    const int trow = begin + row;
    const int64_t rindex =
        ((static_cast<int64_t>(bidx) * T + trow) * H + h) * D + d;
    float value = rhs[rindex];
    #pragma unroll 4
    for (int col = 0; col < row; ++col) {
      const int tcol = begin + col;
      const int64_t xindex =
          ((static_cast<int64_t>(bidx) * T + tcol) * H + h) * D + d;
      value = fmaf(-Akk[abase + row * BT + col], out[xindex], value);
    }
    out[rindex] = value;
  }
}

// Precompute the chunk-end decay and the K-side state-update factor.  Keeping
// these values in a compact FP32 tensor prevents the V-parallel output kernel
// from evaluating exp() once per value channel.
template <typename scalar_t, int KSIZE, int BT>
__global__ __launch_bounds__(THREADS)
void gdn2_chunk_kg_kernel(
    const scalar_t* __restrict__ k,
    const float* __restrict__ G,
    float* __restrict__ kg,
    float* __restrict__ chunk_decay,
    int B, int T, int H, int chunks) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
  const int64_t n = static_cast<int64_t>(B) * T * H * KSIZE;
  if (i >= n) return;
  const int ki = i % KSIZE;
  int64_t x = i / KSIZE;
  const int h = x % H;
  x /= H;
  const int t = x % T;
  const int bidx = x / T;
  const int chunk = t / BT;
  const int last = min((chunk + 1) * BT, T) - 1;
  const int64_t last_index =
      ((static_cast<int64_t>(bidx) * T + last) * H + h) * KSIZE + ki;
  const float glast = G[last_index];
  kg[i] = static_cast<float>(k[i]) * expf(glast - G[i]);
  if ((t % BT) == 0) {
    const int64_t di =
        ((static_cast<int64_t>(bidx) * H + h) * chunks + chunk) * KSIZE + ki;
    chunk_decay[di] = expf(glast);
  }
}

// Device-side Chunk/WY state propagation for K=64.  A block owns 64 value
// channels of one (batch, head), keeps its KxV state columns live across every
// chunk, and replaces the former Python loop plus five matmuls per chunk.
template <int KSIZE, int BT>
__global__ __launch_bounds__(WAVE_SIZE)
void gdn2_chunk_state_output_kernel(
    const float* __restrict__ qg,
    const float* __restrict__ kg,
    const float* __restrict__ u,
    const float* __restrict__ wy,
    const float* __restrict__ Aqk,
    const float* __restrict__ chunk_decay,
    const float* __restrict__ initial_state,
    float* __restrict__ output,
    float* __restrict__ final_state,
    int B, int T, int H, int V, int chunks,
    float scale, bool has_initial) {
  const int vi = blockIdx.x * WAVE_SIZE + threadIdx.x;
  const int h = blockIdx.y;
  const int bidx = blockIdx.z;
  const bool active = vi < V;
  __shared__ float sq[KSIZE];
  __shared__ float swy[KSIZE];
  // Pad the V stride to avoid a power-of-two LDS bank pattern while every
  // lane walks the K dimension.  The two large recurrence buffers live in
  // LDS instead of per-thread scratch (the register-array version spills
  // heavily on gfx936).
  __shared__ float sstate[KSIZE][WAVE_SIZE + 1];
  __shared__ float svnew[BT][WAVE_SIZE];

  #pragma unroll
  for (int ki = 0; ki < KSIZE; ++ki) {
    const int64_t si =
        ((static_cast<int64_t>(bidx) * H + h) * KSIZE + ki) * V + vi;
    sstate[ki][threadIdx.x] =
        (active && has_initial) ? initial_state[si] : 0.0f;
  }
  __syncthreads();

  #pragma unroll 1
  for (int chunk = 0; chunk < chunks; ++chunk) {
    const int begin = chunk * BT;
    const int valid = min(BT, T - begin);
    #pragma unroll 1
    for (int row = 0; row < valid; ++row) {
      const int t = begin + row;
      const int64_t kbase =
          ((static_cast<int64_t>(bidx) * T + t) * H + h) * KSIZE;
      sq[threadIdx.x] = qg[kbase + threadIdx.x];
      swy[threadIdx.x] = wy[kbase + threadIdx.x];
      __syncthreads();
      if (active) {
        float qstate = 0.0f;
        float wstate = 0.0f;
        #pragma unroll
        for (int ki = 0; ki < KSIZE; ++ki) {
          const float hs = sstate[ki][threadIdx.x];
          qstate = fmaf(sq[ki], hs, qstate);
          wstate = fmaf(swy[ki], hs, wstate);
        }
        const int64_t vindex =
            ((static_cast<int64_t>(bidx) * T + t) * H + h) * V + vi;
        const float nv = u[vindex] - wstate;
        svnew[row][threadIdx.x] = nv;
        float local = 0.0f;
        const int64_t abase =
            (((static_cast<int64_t>(bidx) * H + h) * chunks + chunk) * BT + row) * BT;
        #pragma unroll 4
        for (int col = 0; col <= row; ++col) {
          local = fmaf(Aqk[abase + col], svnew[col][threadIdx.x], local);
        }
        output[vindex] = scale * qstate + local;
      }
      __syncthreads();
    }

    if (active) {
      #pragma unroll
      for (int ki = 0; ki < KSIZE; ++ki) {
        const int64_t di =
            ((static_cast<int64_t>(bidx) * H + h) * chunks + chunk) * KSIZE + ki;
        float next = sstate[ki][threadIdx.x] * chunk_decay[di];
        #pragma unroll 4
        for (int row = 0; row < valid; ++row) {
          const int t = begin + row;
          const int64_t kindex =
              ((static_cast<int64_t>(bidx) * T + t) * H + h) * KSIZE + ki;
          next = fmaf(kg[kindex], svnew[row][threadIdx.x], next);
        }
        sstate[ki][threadIdx.x] = next;
      }
    }
    __syncthreads();
  }

  if (active) {
    #pragma unroll
    for (int ki = 0; ki < KSIZE; ++ki) {
      const int64_t si =
          ((static_cast<int64_t>(bidx) * H + h) * KSIZE + ki) * V + vi;
      final_state[si] = sstate[ki][threadIdx.x];
    }
  }
}

// Throughput path for K/V-heavy shapes.  Sixteen 16-lane subgroups own
// sixteen V channels, while the whole block cooperatively loads q/k/g/erase
// exactly once per token.  Compared with the one-wave-per-V latency kernel,
// this removes the V-fold duplication of the K-side global-memory traffic
// without putting a K-vector in one thread's private memory.
template <typename scalar_t, int KSIZE, int GROUP_SIZE>
__global__ __launch_bounds__(THREADS)
void gdn2_forward_shared_k_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const scalar_t* __restrict__ g,
    const scalar_t* __restrict__ erase,
    const scalar_t* __restrict__ write,
    const float* __restrict__ initial_state,
    scalar_t* __restrict__ output,
    float* __restrict__ final_state,
    int B, int T, int H, int V,
    float scale, bool has_initial) {
  static_assert(KSIZE % GROUP_SIZE == 0, "K must divide subgroup mapping");
  constexpr int ITEMS = KSIZE / GROUP_SIZE;
  constexpr int GROUPS_PER_BLOCK = THREADS / GROUP_SIZE;
  const int tid = threadIdx.x;
  const int subgroup = tid / GROUP_SIZE;
  const int lane = tid & (GROUP_SIZE - 1);
  const int vi = blockIdx.x * GROUPS_PER_BLOCK + subgroup;
  const int h = blockIdx.y;
  const int bidx = blockIdx.z;
  const bool active = vi < V;

  __shared__ float sq[KSIZE];
  __shared__ float sk[KSIZE];
  __shared__ float sd[KSIZE];
  __shared__ float sek[KSIZE];

  float state[ITEMS];
  #pragma unroll
  for (int item = 0; item < ITEMS; ++item) {
    const int ki = lane + item * GROUP_SIZE;
    const int si = ((bidx * H + h) * KSIZE + ki) * V + vi;
    state[item] = (active && has_initial) ? initial_state[si] : 0.0f;
  }

  #pragma unroll 1
  for (int ti = 0; ti < T; ++ti) {
    const int base_k = ((bidx * T + ti) * H + h) * KSIZE;
    for (int ki = tid; ki < KSIZE; ki += THREADS) {
      sq[ki] = static_cast<float>(q[base_k + ki]);
      const float kval = static_cast<float>(k[base_k + ki]);
      sk[ki] = kval;
      sd[ki] = expf(static_cast<float>(g[base_k + ki]));
      // This product is independent of V.  Compute it once per K coordinate
      // instead of once in every subgroup (32 times for GROUP_SIZE=8).
      sek[ki] = static_cast<float>(erase[base_k + ki]) * kval;
    }
    __syncthreads();

    float erase_dot = 0.0f;
    #pragma unroll
    for (int item = 0; item < ITEMS; ++item) {
      const int ki = lane + item * GROUP_SIZE;
      state[item] *= sd[ki];
      erase_dot = fmaf(sek[ki], state[item], erase_dot);
    }
    erase_dot = subgroup_sum<GROUP_SIZE>(erase_dot);
    erase_dot = __shfl(erase_dot, 0, GROUP_SIZE);

    const int base_v = ((bidx * T + ti) * H + h) * V;
    float correction = 0.0f;
    if (active) {
      correction = static_cast<float>(write[base_v + vi]) *
                   static_cast<float>(v[base_v + vi]) - erase_dot;
    }

    float out = 0.0f;
    #pragma unroll
    for (int item = 0; item < ITEMS; ++item) {
      const int ki = lane + item * GROUP_SIZE;
      state[item] = fmaf(sk[ki], correction, state[item]);
      out = fmaf(sq[ki], state[item], out);
    }
    out = subgroup_sum<GROUP_SIZE>(out);
    if (active && lane == 0) {
      output[base_v + vi] = static_cast<scalar_t>(out * scale);
    }
    __syncthreads();
  }

  if (active) {
    #pragma unroll
    for (int item = 0; item < ITEMS; ++item) {
      const int ki = lane + item * GROUP_SIZE;
      const int si = ((bidx * H + h) * KSIZE + ki) * V + vi;
      final_state[si] = state[item];
    }
  }
}

// BW-optimized mapping: one 64-lane wavefront owns one V channel. Each lane
// keeps K/64 state scalars in registers, avoiding the large private array and
// scratch-memory spills of the generic v1 kernel.
template <typename scalar_t, int KSIZE>
__global__ __launch_bounds__(WAVE_SIZE * WAVES_PER_BLOCK)
void gdn2_forward_wave_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const float* __restrict__ decay,
    const scalar_t* __restrict__ erase,
    const scalar_t* __restrict__ write,
    const float* __restrict__ initial_state,
    scalar_t* __restrict__ output,
    float* __restrict__ final_state,
    int B, int T, int H, int V,
    float scale, bool has_initial) {
  constexpr int ITEMS = KSIZE / WAVE_SIZE;
  const int lane = threadIdx.x & (WAVE_SIZE - 1);
  const int wave = threadIdx.x / WAVE_SIZE;
  const int vi = blockIdx.x * WAVES_PER_BLOCK + wave;
  const int h = blockIdx.y;
  const int bidx = blockIdx.z;
  const bool active = vi < V;

  float state[ITEMS];
  #pragma unroll
  for (int item = 0; item < ITEMS; ++item) {
    const int ki = lane + item * WAVE_SIZE;
    const int si = ((bidx * H + h) * KSIZE + ki) * V + vi;
    state[item] = (active && has_initial) ? initial_state[si] : 0.0f;
  }

  #pragma unroll 1
  for (int ti = 0; ti < T; ++ti) {
    const int base_k = ((bidx * T + ti) * H + h) * KSIZE;
    float kval[ITEMS], qval[ITEMS];
    float erase_dot = 0.0f;
    #pragma unroll
    for (int item = 0; item < ITEMS; ++item) {
      const int ki = lane + item * WAVE_SIZE;
      kval[item] = static_cast<float>(k[base_k + ki]);
      qval[item] = static_cast<float>(q[base_k + ki]);
      state[item] *= decay[base_k + ki];
      erase_dot = fmaf(static_cast<float>(erase[base_k + ki]) * kval[item],
                       state[item], erase_dot);
    }
    erase_dot = wave_sum(erase_dot);
    erase_dot = __shfl(erase_dot, 0, WAVE_SIZE);

    const int base_v = ((bidx * T + ti) * H + h) * V;
    float correction = 0.0f;
    if (active) {
      correction = static_cast<float>(write[base_v + vi]) *
                       static_cast<float>(v[base_v + vi]) - erase_dot;
    }

    float out = 0.0f;
    #pragma unroll
    for (int item = 0; item < ITEMS; ++item) {
      state[item] = fmaf(kval[item], correction, state[item]);
      out = fmaf(qval[item], state[item], out);
    }
    out = wave_sum(out);
    if (active && lane == 0) {
      output[base_v + vi] = static_cast<scalar_t>(out * scale);
    }
  }

  if (active) {
    #pragma unroll
    for (int item = 0; item < ITEMS; ++item) {
      const int ki = lane + item * WAVE_SIZE;
      const int si = ((bidx * H + h) * KSIZE + ki) * V + vi;
      final_state[si] = state[item];
    }
  }
}

template <typename scalar_t>
__global__ void gdn2_forward_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const scalar_t* __restrict__ g,
    const scalar_t* __restrict__ erase,
    const scalar_t* __restrict__ write,
    const float* __restrict__ initial_state,
    scalar_t* __restrict__ output,
    float* __restrict__ final_state,
    int B, int T, int H, int K, int V,
    float scale,
    bool has_initial) {
  const int tile = blockIdx.x;
  const int h = blockIdx.y;
  const int bidx = blockIdx.z;
  const int vi = tile * blockDim.x + threadIdx.x;
  const bool active = vi < V;

  __shared__ float sq[MAX_K];
  __shared__ float sk[MAX_K];
  __shared__ float sd[MAX_K];
  __shared__ float se[MAX_K];

  // One V channel per thread. The K-vector remains live for the entire
  // sequence, eliminating all per-token state traffic and kernel launches.
  float state[MAX_K];
  #pragma unroll 1
  for (int ki = 0; ki < K; ++ki) {
    const int si = ((bidx * H + h) * K + ki) * V + vi;
    state[ki] = (active && has_initial) ? initial_state[si] : 0.0f;
  }

  #pragma unroll 1
  for (int ti = 0; ti < T; ++ti) {
    const int base_k = ((bidx * T + ti) * H + h) * K;
    for (int ki = threadIdx.x; ki < K; ki += blockDim.x) {
      sq[ki] = static_cast<float>(q[base_k + ki]);
      sk[ki] = static_cast<float>(k[base_k + ki]);
      sd[ki] = expf(static_cast<float>(g[base_k + ki]));
      se[ki] = static_cast<float>(erase[base_k + ki]);
    }
    __syncthreads();

    if (active) {
      float correction = 0.0f;
      #pragma unroll 1
      for (int ki = 0; ki < K; ++ki) {
        state[ki] *= sd[ki];
        correction = fmaf(se[ki] * sk[ki], state[ki], correction);
      }
      const int base_v = ((bidx * T + ti) * H + h) * V;
      correction = static_cast<float>(write[base_v + vi]) *
                       static_cast<float>(v[base_v + vi]) -
                   correction;

      float out = 0.0f;
      #pragma unroll 1
      for (int ki = 0; ki < K; ++ki) {
        state[ki] = fmaf(sk[ki], correction, state[ki]);
        out = fmaf(sq[ki], state[ki], out);
      }
      output[base_v + vi] = static_cast<scalar_t>(out * scale);
    }
    __syncthreads();
  }

  if (active) {
    #pragma unroll 1
    for (int ki = 0; ki < K; ++ki) {
      const int si = ((bidx * H + h) * K + ki) * V + vi;
      final_state[si] = state[ki];
    }
  }
}

void check_tensor(const torch::Tensor& x, const char* name) {
  TORCH_CHECK(x.is_cuda(), name, " must be on a Hygon HIP device");
  TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> gdn2_forward_hip(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor g, torch::Tensor erase, torch::Tensor write,
    c10::optional<torch::Tensor> initial_state,
    double scale) {
  check_tensor(q, "q"); check_tensor(k, "k"); check_tensor(v, "v");
  check_tensor(g, "g"); check_tensor(erase, "erase"); check_tensor(write, "write");
  TORCH_CHECK(q.dim() == 4 && v.dim() == 4, "inputs must be BTHK/BTHV");
  TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == g.sizes() &&
              q.sizes() == erase.sizes(), "q/k/g/erase shapes must match");
  TORCH_CHECK(v.sizes() == write.sizes(), "v/write shapes must match");
  TORCH_CHECK(q.size(0) == v.size(0) && q.size(1) == v.size(1) &&
              q.size(2) == v.size(2), "B/T/H dimensions must match");
  TORCH_CHECK(q.scalar_type() == v.scalar_type() &&
              q.scalar_type() == k.scalar_type() &&
              q.scalar_type() == g.scalar_type() &&
              q.scalar_type() == erase.scalar_type() &&
              q.scalar_type() == write.scalar_type(), "all inputs must share dtype");
  TORCH_CHECK(q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16,
              "only float16 and bfloat16 are supported");

  const int B = q.size(0), T = q.size(1), H = q.size(2), K = q.size(3);
  const int V = v.size(3);
  TORCH_CHECK(K > 0 && K <= MAX_K, "K must be in [1,256]");
  const float* h0 = nullptr;
  bool has_initial = initial_state.has_value() && initial_state->defined();
  if (has_initial) {
    check_tensor(*initial_state, "initial_state");
    TORCH_CHECK(initial_state->scalar_type() == at::kFloat, "initial_state must be float32");
    TORCH_CHECK(initial_state->dim() == 4 && initial_state->size(0) == B &&
                initial_state->size(1) == H && initial_state->size(2) == K &&
                initial_state->size(3) == V, "initial_state must be [B,H,K,V]");
    h0 = initial_state->data_ptr<float>();
  }

  auto output = torch::empty_like(v);
  auto final_state = torch::empty({B,H,K,V}, q.options().dtype(torch::kFloat));
  // The latency path wins for the short K64/V64 case.  Larger K or V uses
  // the shared-K throughput mapping to reuse the K-side input across 16 V
  // channels.  Both paths use 256 threads and wavefront-64 native shuffles.
  // K64/V64 has two measured regimes on gfx936.  Shared-K wins when B*H
  // supplies at least 32 independent heads (the B4H8/B8H8 T1024 cases), but
  // its per-token block barriers lose for low-parallelism B1H8/B1H16 long
  // sequences.  Keep those on the barrier-free wave latency path.
  const bool use_shared_k =
      (K >= 128 || V >= 128 ||
       (K == 64 && V == 64 && T >= 1024 && B * H >= 32));
  // Eight lanes is the measured gfx936 sweet spot for both K128 and K256.
  // Four lanes increases live state to 32 scalars/lane for K128 and loses
  // performance to register pressure despite the additional K-side reuse.
  // K256 also keeps group8: group16 reduces available block parallelism on
  // gfx936 and regresses the measured K256 workloads by 12--26 percent.
  const int shared_group_size = 8;
  const int shared_groups_per_block = THREADS / shared_group_size;
  const dim3 block(THREADS);
  const dim3 grid(
      use_shared_k
          ? (V + shared_groups_per_block - 1) / shared_groups_per_block
          : (V + WAVES_PER_BLOCK - 1) / WAVES_PER_BLOCK,
      H, B);
  hipStream_t stream = c10::hip::getCurrentHIPStream().stream();
  torch::Tensor decay;
  if (!use_shared_k) {
    decay = torch::empty(q.sizes(), q.options().dtype(torch::kFloat));
  }

  AT_DISPATCH_SWITCH(q.scalar_type(), "gdn2_forward_hip", AT_DISPATCH_CASE(
      at::ScalarType::Half, [&] {
        if (!use_shared_k) {
          const int64_t n = q.numel();
          const dim3 decay_grid((n + THREADS - 1) / THREADS);
          hipLaunchKernelGGL((gdn2_decay_precompute_kernel<scalar_t>),
              decay_grid, block, 0, stream, g.data_ptr<scalar_t>(),
              decay.data_ptr<float>(), n);
        }
        if (use_shared_k && K == 64) hipLaunchKernelGGL((gdn2_forward_shared_k_kernel<scalar_t,64,8>), grid, block, 0, stream, q.data_ptr<scalar_t>(),k.data_ptr<scalar_t>(),v.data_ptr<scalar_t>(),g.data_ptr<scalar_t>(),erase.data_ptr<scalar_t>(),write.data_ptr<scalar_t>(),h0,output.data_ptr<scalar_t>(),final_state.data_ptr<float>(),B,T,H,V,static_cast<float>(scale),has_initial);
        else if (use_shared_k && K == 128) hipLaunchKernelGGL((gdn2_forward_shared_k_kernel<scalar_t,128,8>), grid, block, 0, stream, q.data_ptr<scalar_t>(),k.data_ptr<scalar_t>(),v.data_ptr<scalar_t>(),g.data_ptr<scalar_t>(),erase.data_ptr<scalar_t>(),write.data_ptr<scalar_t>(),h0,output.data_ptr<scalar_t>(),final_state.data_ptr<float>(),B,T,H,V,static_cast<float>(scale),has_initial);
        else if (use_shared_k && K == 256) hipLaunchKernelGGL((gdn2_forward_shared_k_kernel<scalar_t,256,8>), grid, block, 0, stream, q.data_ptr<scalar_t>(),k.data_ptr<scalar_t>(),v.data_ptr<scalar_t>(),g.data_ptr<scalar_t>(),erase.data_ptr<scalar_t>(),write.data_ptr<scalar_t>(),h0,output.data_ptr<scalar_t>(),final_state.data_ptr<float>(),B,T,H,V,static_cast<float>(scale),has_initial);
        else if (K == 64) hipLaunchKernelGGL((gdn2_forward_wave_kernel<scalar_t,64>), grid, block, 0, stream, q.data_ptr<scalar_t>(),k.data_ptr<scalar_t>(),v.data_ptr<scalar_t>(),decay.data_ptr<float>(),erase.data_ptr<scalar_t>(),write.data_ptr<scalar_t>(),h0,output.data_ptr<scalar_t>(),final_state.data_ptr<float>(),B,T,H,V,static_cast<float>(scale),has_initial);
        else if (K == 128) hipLaunchKernelGGL((gdn2_forward_wave_kernel<scalar_t,128>), grid, block, 0, stream, q.data_ptr<scalar_t>(),k.data_ptr<scalar_t>(),v.data_ptr<scalar_t>(),decay.data_ptr<float>(),erase.data_ptr<scalar_t>(),write.data_ptr<scalar_t>(),h0,output.data_ptr<scalar_t>(),final_state.data_ptr<float>(),B,T,H,V,static_cast<float>(scale),has_initial);
        else if (K == 256) hipLaunchKernelGGL((gdn2_forward_wave_kernel<scalar_t,256>), grid, block, 0, stream, q.data_ptr<scalar_t>(),k.data_ptr<scalar_t>(),v.data_ptr<scalar_t>(),decay.data_ptr<float>(),erase.data_ptr<scalar_t>(),write.data_ptr<scalar_t>(),h0,output.data_ptr<scalar_t>(),final_state.data_ptr<float>(),B,T,H,V,static_cast<float>(scale),has_initial);
        else TORCH_CHECK(false, "HIP wave kernel supports K=64,128,256");
      }) AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&] {
        if (!use_shared_k) {
          const int64_t n = q.numel();
          const dim3 decay_grid((n + THREADS - 1) / THREADS);
          hipLaunchKernelGGL((gdn2_decay_precompute_kernel<scalar_t>),
              decay_grid, block, 0, stream, g.data_ptr<scalar_t>(),
              decay.data_ptr<float>(), n);
        }
        if (use_shared_k && K == 64) hipLaunchKernelGGL((gdn2_forward_shared_k_kernel<scalar_t,64,8>), grid, block, 0, stream, q.data_ptr<scalar_t>(),k.data_ptr<scalar_t>(),v.data_ptr<scalar_t>(),g.data_ptr<scalar_t>(),erase.data_ptr<scalar_t>(),write.data_ptr<scalar_t>(),h0,output.data_ptr<scalar_t>(),final_state.data_ptr<float>(),B,T,H,V,static_cast<float>(scale),has_initial);
        else if (use_shared_k && K == 128) hipLaunchKernelGGL((gdn2_forward_shared_k_kernel<scalar_t,128,8>), grid, block, 0, stream, q.data_ptr<scalar_t>(),k.data_ptr<scalar_t>(),v.data_ptr<scalar_t>(),g.data_ptr<scalar_t>(),erase.data_ptr<scalar_t>(),write.data_ptr<scalar_t>(),h0,output.data_ptr<scalar_t>(),final_state.data_ptr<float>(),B,T,H,V,static_cast<float>(scale),has_initial);
        else if (use_shared_k && K == 256) hipLaunchKernelGGL((gdn2_forward_shared_k_kernel<scalar_t,256,8>), grid, block, 0, stream, q.data_ptr<scalar_t>(),k.data_ptr<scalar_t>(),v.data_ptr<scalar_t>(),g.data_ptr<scalar_t>(),erase.data_ptr<scalar_t>(),write.data_ptr<scalar_t>(),h0,output.data_ptr<scalar_t>(),final_state.data_ptr<float>(),B,T,H,V,static_cast<float>(scale),has_initial);
        else if (K == 64) hipLaunchKernelGGL((gdn2_forward_wave_kernel<scalar_t,64>), grid, block, 0, stream, q.data_ptr<scalar_t>(),k.data_ptr<scalar_t>(),v.data_ptr<scalar_t>(),decay.data_ptr<float>(),erase.data_ptr<scalar_t>(),write.data_ptr<scalar_t>(),h0,output.data_ptr<scalar_t>(),final_state.data_ptr<float>(),B,T,H,V,static_cast<float>(scale),has_initial);
        else if (K == 128) hipLaunchKernelGGL((gdn2_forward_wave_kernel<scalar_t,128>), grid, block, 0, stream, q.data_ptr<scalar_t>(),k.data_ptr<scalar_t>(),v.data_ptr<scalar_t>(),decay.data_ptr<float>(),erase.data_ptr<scalar_t>(),write.data_ptr<scalar_t>(),h0,output.data_ptr<scalar_t>(),final_state.data_ptr<float>(),B,T,H,V,static_cast<float>(scale),has_initial);
        else if (K == 256) hipLaunchKernelGGL((gdn2_forward_wave_kernel<scalar_t,256>), grid, block, 0, stream, q.data_ptr<scalar_t>(),k.data_ptr<scalar_t>(),v.data_ptr<scalar_t>(),decay.data_ptr<float>(),erase.data_ptr<scalar_t>(),write.data_ptr<scalar_t>(),h0,output.data_ptr<scalar_t>(),final_state.data_ptr<float>(),B,T,H,V,static_cast<float>(scale),has_initial);
        else TORCH_CHECK(false, "HIP wave kernel supports K=64,128,256");
      }));
  C10_HIP_KERNEL_LAUNCH_CHECK();
  return {output, final_state};
}

torch::Tensor gdn2_chunk_cumsum_hip(torch::Tensor g, int64_t chunk_size) {
  check_tensor(g, "g");
  TORCH_CHECK(g.dim() == 4, "g must be [B,T,H,K]");
  TORCH_CHECK(g.scalar_type() == at::kHalf ||
              g.scalar_type() == at::kBFloat16,
              "g must be float16 or bfloat16");
  TORCH_CHECK(chunk_size == 64,
              "initial gfx936 Chunk/WY path requires chunk_size=64");
  const int B = g.size(0);
  const int T = g.size(1);
  const int H = g.size(2);
  const int K = g.size(3);
  TORCH_CHECK(K == 64 || K == 128 || K == 256,
              "K must be 64, 128 or 256");
  const int chunks = (T + 63) / 64;
  auto out = torch::empty(g.sizes(), g.options().dtype(torch::kFloat));
  const dim3 block(THREADS);
  const dim3 grid(chunks, H, B);
  hipStream_t stream = c10::hip::getCurrentHIPStream().stream();

  AT_DISPATCH_SWITCH(g.scalar_type(), "gdn2_chunk_cumsum_hip",
    AT_DISPATCH_CASE(at::ScalarType::Half, [&] {
      if (K == 64) hipLaunchKernelGGL((gdn2_chunk_cumsum_kernel<scalar_t,64,64>), grid, block, 0, stream, g.data_ptr<scalar_t>(), out.data_ptr<float>(), B,T,H,chunks);
      else if (K == 128) hipLaunchKernelGGL((gdn2_chunk_cumsum_kernel<scalar_t,128,64>), grid, block, 0, stream, g.data_ptr<scalar_t>(), out.data_ptr<float>(), B,T,H,chunks);
      else hipLaunchKernelGGL((gdn2_chunk_cumsum_kernel<scalar_t,256,64>), grid, block, 0, stream, g.data_ptr<scalar_t>(), out.data_ptr<float>(), B,T,H,chunks);
    })
    AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&] {
      if (K == 64) hipLaunchKernelGGL((gdn2_chunk_cumsum_kernel<scalar_t,64,64>), grid, block, 0, stream, g.data_ptr<scalar_t>(), out.data_ptr<float>(), B,T,H,chunks);
      else if (K == 128) hipLaunchKernelGGL((gdn2_chunk_cumsum_kernel<scalar_t,128,64>), grid, block, 0, stream, g.data_ptr<scalar_t>(), out.data_ptr<float>(), B,T,H,chunks);
      else hipLaunchKernelGGL((gdn2_chunk_cumsum_kernel<scalar_t,256,64>), grid, block, 0, stream, g.data_ptr<scalar_t>(), out.data_ptr<float>(), B,T,H,chunks);
    }));
  C10_HIP_KERNEL_LAUNCH_CHECK();
  return out;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
gdn2_chunk_factors_hip(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor erase,
    torch::Tensor g_cumsum) {
  check_tensor(q, "q");
  check_tensor(k, "k");
  check_tensor(erase, "erase");
  TORCH_CHECK(g_cumsum.is_cuda() && g_cumsum.is_contiguous(),
              "g_cumsum must be contiguous on the Hygon HIP device");
  TORCH_CHECK(q.dim() == 4 && q.sizes() == k.sizes() &&
              q.sizes() == erase.sizes() && q.sizes() == g_cumsum.sizes(),
              "q/k/erase/g_cumsum must share [B,T,H,K] shape");
  TORCH_CHECK(q.scalar_type() == k.scalar_type() &&
              q.scalar_type() == erase.scalar_type(),
              "q/k/erase dtypes must match");
  TORCH_CHECK(g_cumsum.scalar_type() == at::kFloat,
              "g_cumsum must be float32");

  auto options = q.options().dtype(torch::kFloat);
  auto qg = torch::empty(q.sizes(), options);
  auto kn = torch::empty(q.sizes(), options);
  auto ke = torch::empty(q.sizes(), options);
  const int64_t n = q.numel();
  const dim3 block(THREADS);
  const dim3 grid((n + THREADS - 1) / THREADS);
  hipStream_t stream = c10::hip::getCurrentHIPStream().stream();

  AT_DISPATCH_SWITCH(q.scalar_type(), "gdn2_chunk_factors_hip",
    AT_DISPATCH_CASE(at::ScalarType::Half, [&] {
      hipLaunchKernelGGL((gdn2_chunk_factors_kernel<scalar_t>), grid, block,
          0, stream, q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(),
          erase.data_ptr<scalar_t>(), g_cumsum.data_ptr<float>(),
          qg.data_ptr<float>(), kn.data_ptr<float>(), ke.data_ptr<float>(), n);
    })
    AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&] {
      hipLaunchKernelGGL((gdn2_chunk_factors_kernel<scalar_t>), grid, block,
          0, stream, q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(),
          erase.data_ptr<scalar_t>(), g_cumsum.data_ptr<float>(),
          qg.data_ptr<float>(), kn.data_ptr<float>(), ke.data_ptr<float>(), n);
    }));
  C10_HIP_KERNEL_LAUNCH_CHECK();
  return {qg, kn, ke};
}

std::tuple<torch::Tensor, torch::Tensor> gdn2_chunk_scores_hip(
    torch::Tensor qg,
    torch::Tensor kn,
    torch::Tensor ke,
    double scale,
    int64_t chunk_size) {
  TORCH_CHECK(qg.is_cuda() && kn.is_cuda() && ke.is_cuda(),
              "qg/kn/ke must be on the Hygon HIP device");
  TORCH_CHECK(qg.is_contiguous() && kn.is_contiguous() && ke.is_contiguous(),
              "qg/kn/ke must be contiguous");
  TORCH_CHECK(qg.scalar_type() == at::kFloat &&
              kn.scalar_type() == at::kFloat &&
              ke.scalar_type() == at::kFloat,
              "qg/kn/ke must be float32");
  TORCH_CHECK(qg.dim() == 4 && qg.sizes() == kn.sizes() &&
              qg.sizes() == ke.sizes(),
              "qg/kn/ke must share [B,T,H,K] shape");
  TORCH_CHECK(chunk_size == 64, "chunk_size must be 64");
  const int B = qg.size(0);
  const int T = qg.size(1);
  const int H = qg.size(2);
  const int K = qg.size(3);
  TORCH_CHECK(K == 64 || K == 128 || K == 256,
              "K must be 64, 128 or 256");
  const int chunks = (T + 63) / 64;
  auto options = qg.options();
  auto Aqk = torch::empty({B,H,chunks,64,64}, options);
  auto Akk = torch::empty({B,H,chunks,64,64}, options);
  constexpr int SCORE_TILE = 16;
  const dim3 block(SCORE_TILE, SCORE_TILE);
  const dim3 grid(B * H * chunks,
                  (64 / SCORE_TILE) * (64 / SCORE_TILE));
  hipStream_t stream = c10::hip::getCurrentHIPStream().stream();
  if (K == 64) {
    hipLaunchKernelGGL((gdn2_chunk_scores_tiled_kernel<64,64,SCORE_TILE>), grid, block, 0,
        stream, qg.data_ptr<float>(), kn.data_ptr<float>(),
        ke.data_ptr<float>(), Aqk.data_ptr<float>(), Akk.data_ptr<float>(),
        B,T,H,chunks,static_cast<float>(scale));
  } else if (K == 128) {
    hipLaunchKernelGGL((gdn2_chunk_scores_tiled_kernel<128,64,SCORE_TILE>), grid, block, 0,
        stream, qg.data_ptr<float>(), kn.data_ptr<float>(),
        ke.data_ptr<float>(), Aqk.data_ptr<float>(), Akk.data_ptr<float>(),
        B,T,H,chunks,static_cast<float>(scale));
  } else {
    hipLaunchKernelGGL((gdn2_chunk_scores_tiled_kernel<256,64,SCORE_TILE>), grid, block, 0,
        stream, qg.data_ptr<float>(), kn.data_ptr<float>(),
        ke.data_ptr<float>(), Aqk.data_ptr<float>(), Akk.data_ptr<float>(),
        B,T,H,chunks,static_cast<float>(scale));
  }
  C10_HIP_KERNEL_LAUNCH_CHECK();
  return {Aqk, Akk};
}

torch::Tensor gdn2_chunk_solve_hip(
    torch::Tensor Akk,
    torch::Tensor rhs,
    int64_t chunk_size) {
  TORCH_CHECK(Akk.is_cuda() && rhs.is_cuda(),
              "Akk/rhs must be on the Hygon HIP device");
  TORCH_CHECK(Akk.is_contiguous() && rhs.is_contiguous(),
              "Akk/rhs must be contiguous");
  TORCH_CHECK(Akk.scalar_type() == at::kFloat &&
              rhs.scalar_type() == at::kFloat,
              "Akk/rhs must be float32");
  TORCH_CHECK(Akk.dim() == 5 && rhs.dim() == 4,
              "Akk must be [B,H,NT,64,64], rhs must be [B,T,H,D]");
  TORCH_CHECK(chunk_size == 64 && Akk.size(3) == 64 && Akk.size(4) == 64,
              "chunk_size and Akk tile must be 64");
  const int B = rhs.size(0);
  const int T = rhs.size(1);
  const int H = rhs.size(2);
  const int D = rhs.size(3);
  const int chunks = (T + 63) / 64;
  TORCH_CHECK(Akk.size(0) == B && Akk.size(1) == H &&
              Akk.size(2) == chunks,
              "Akk B/H/NT dimensions do not match rhs");
  auto out = torch::empty_like(rhs);
  const dim3 block(THREADS);
  const dim3 grid(B * H * chunks, (D + THREADS - 1) / THREADS);
  hipStream_t stream = c10::hip::getCurrentHIPStream().stream();
  hipLaunchKernelGGL((gdn2_chunk_solve_kernel<64>), grid, block, 0, stream,
      Akk.data_ptr<float>(), rhs.data_ptr<float>(), out.data_ptr<float>(),
      B,T,H,D,chunks);
  C10_HIP_KERNEL_LAUNCH_CHECK();
  return out;
}

std::tuple<torch::Tensor, torch::Tensor> gdn2_chunk_kg_hip(
    torch::Tensor k, torch::Tensor G, int64_t chunk_size) {
  check_tensor(k, "k");
  check_tensor(G, "G");
  TORCH_CHECK(k.dim() == 4 && k.sizes() == G.sizes(),
              "k/G must share [B,T,H,K] shape");
  TORCH_CHECK(G.scalar_type() == at::kFloat, "G must be float32");
  TORCH_CHECK(chunk_size == 64, "chunk_size must be 64");
  const int B = k.size(0), T = k.size(1), H = k.size(2), K = k.size(3);
  TORCH_CHECK(K == 64 || K == 128 || K == 256,
              "chunk_kg requires K=64, 128 or 256");
  const int chunks = (T + 63) / 64;
  auto kg = torch::empty({B,T,H,K}, G.options());
  auto decay = torch::empty({B,H,chunks,K}, G.options());
  const int64_t n = static_cast<int64_t>(B) * T * H * K;
  const dim3 block(THREADS);
  const dim3 grid((n + THREADS - 1) / THREADS);
  hipStream_t stream = c10::hip::getCurrentHIPStream().stream();
  AT_DISPATCH_SWITCH(k.scalar_type(), "gdn2_chunk_kg_hip", AT_DISPATCH_CASE(
      at::ScalarType::Half, [&] {
        if (K == 64)
          hipLaunchKernelGGL((gdn2_chunk_kg_kernel<scalar_t,64,64>), grid, block,
              0, stream, k.data_ptr<scalar_t>(), G.data_ptr<float>(),
              kg.data_ptr<float>(), decay.data_ptr<float>(), B,T,H,chunks);
        else if (K == 128)
          hipLaunchKernelGGL((gdn2_chunk_kg_kernel<scalar_t,128,64>), grid, block,
              0, stream, k.data_ptr<scalar_t>(), G.data_ptr<float>(),
              kg.data_ptr<float>(), decay.data_ptr<float>(), B,T,H,chunks);
        else
          hipLaunchKernelGGL((gdn2_chunk_kg_kernel<scalar_t,256,64>), grid, block,
              0, stream, k.data_ptr<scalar_t>(), G.data_ptr<float>(),
              kg.data_ptr<float>(), decay.data_ptr<float>(), B,T,H,chunks);
      }) AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&] {
        if (K == 64)
          hipLaunchKernelGGL((gdn2_chunk_kg_kernel<scalar_t,64,64>), grid, block,
              0, stream, k.data_ptr<scalar_t>(), G.data_ptr<float>(),
              kg.data_ptr<float>(), decay.data_ptr<float>(), B,T,H,chunks);
        else if (K == 128)
          hipLaunchKernelGGL((gdn2_chunk_kg_kernel<scalar_t,128,64>), grid, block,
              0, stream, k.data_ptr<scalar_t>(), G.data_ptr<float>(),
              kg.data_ptr<float>(), decay.data_ptr<float>(), B,T,H,chunks);
        else
          hipLaunchKernelGGL((gdn2_chunk_kg_kernel<scalar_t,256,64>), grid, block,
              0, stream, k.data_ptr<scalar_t>(), G.data_ptr<float>(),
              kg.data_ptr<float>(), decay.data_ptr<float>(), B,T,H,chunks);
      }));
  C10_HIP_KERNEL_LAUNCH_CHECK();
  return {kg, decay};
}

std::tuple<torch::Tensor, torch::Tensor> gdn2_chunk_state_output_hip(
    torch::Tensor qg, torch::Tensor kg, torch::Tensor u, torch::Tensor wy,
    torch::Tensor Aqk, torch::Tensor chunk_decay,
    c10::optional<torch::Tensor> initial_state, double scale,
    int64_t chunk_size) {
  check_tensor(qg, "qg"); check_tensor(kg, "kg");
  check_tensor(u, "u"); check_tensor(wy, "wy");
  check_tensor(Aqk, "Aqk"); check_tensor(chunk_decay, "chunk_decay");
  TORCH_CHECK(qg.scalar_type() == at::kFloat && kg.scalar_type() == at::kFloat &&
              u.scalar_type() == at::kFloat && wy.scalar_type() == at::kFloat &&
              Aqk.scalar_type() == at::kFloat &&
              chunk_decay.scalar_type() == at::kFloat,
              "device chunk propagation tensors must be float32");
  TORCH_CHECK(qg.dim() == 4 && qg.sizes() == kg.sizes() &&
              qg.sizes() == wy.sizes(), "qg/kg/wy shape mismatch");
  const int B = qg.size(0), T = qg.size(1), H = qg.size(2), K = qg.size(3);
  const int V = u.size(3), chunks = (T + 63) / 64;
  TORCH_CHECK(K == 64 && chunk_size == 64, "candidate requires K=64 and BT=64");
  TORCH_CHECK(u.dim() == 4 && u.size(0) == B && u.size(1) == T &&
              u.size(2) == H, "u must be [B,T,H,V]");
  TORCH_CHECK(Aqk.dim() == 5 && Aqk.size(0) == B && Aqk.size(1) == H &&
              Aqk.size(2) == chunks && Aqk.size(3) == 64 && Aqk.size(4) == 64,
              "Aqk must be [B,H,NT,64,64]");
  TORCH_CHECK(chunk_decay.dim() == 4 && chunk_decay.size(0) == B &&
              chunk_decay.size(1) == H && chunk_decay.size(2) == chunks &&
              chunk_decay.size(3) == K,
              "chunk_decay must be [B,H,NT,K]");
  const float* h0 = nullptr;
  const bool has_initial = initial_state.has_value() && initial_state->defined();
  if (has_initial) {
    check_tensor(*initial_state, "initial_state");
    TORCH_CHECK(initial_state->scalar_type() == at::kFloat &&
                initial_state->dim() == 4 && initial_state->size(0) == B &&
                initial_state->size(1) == H && initial_state->size(2) == K &&
                initial_state->size(3) == V,
                "initial_state must be float32 [B,H,K,V]");
    h0 = initial_state->data_ptr<float>();
  }
  auto output = torch::empty({B,T,H,V}, qg.options());
  auto final_state = torch::empty({B,H,K,V}, qg.options());
  const dim3 block(WAVE_SIZE);
  const dim3 grid((V + WAVE_SIZE - 1) / WAVE_SIZE, H, B);
  hipStream_t stream = c10::hip::getCurrentHIPStream().stream();
  hipLaunchKernelGGL((gdn2_chunk_state_output_kernel<64,64>), grid, block, 0,
      stream, qg.data_ptr<float>(), kg.data_ptr<float>(), u.data_ptr<float>(),
      wy.data_ptr<float>(), Aqk.data_ptr<float>(), chunk_decay.data_ptr<float>(),
      h0, output.data_ptr<float>(), final_state.data_ptr<float>(),
      B,T,H,V,chunks,static_cast<float>(scale),has_initial);
  C10_HIP_KERNEL_LAUNCH_CHECK();
  return {output, final_state};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &gdn2_forward_hip, "GDN2 fused HIP forward");
  m.def("chunk_cumsum", &gdn2_chunk_cumsum_hip,
        "GDN2 BT64 local cumsum for Hygon gfx936");
  m.def("chunk_factors", &gdn2_chunk_factors_hip,
        "GDN2 fused qg/kn/ke factors for Hygon gfx936");
  m.def("chunk_scores", &gdn2_chunk_scores_hip,
        "GDN2 causal Aqk/Akk scores for Hygon gfx936");
  m.def("chunk_solve", &gdn2_chunk_solve_hip,
        "GDN2 BT64 unit-lower forward solve for Hygon gfx936");
  m.def("chunk_kg", &gdn2_chunk_kg_hip,
        "GDN2 K-side chunk state factors for Hygon gfx936");
  m.def("chunk_state_output", &gdn2_chunk_state_output_hip,
        "GDN2 fused device chunk state/output propagation for Hygon gfx936");
}
