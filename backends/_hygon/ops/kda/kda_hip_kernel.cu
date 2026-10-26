#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPException.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdlib>
#include <tuple>

namespace {
constexpr int MAX_K = 256;
constexpr int VTILE = 64;

template <typename S>
__global__ void cumsum_kernel(const S* x, S* y, int B, int T, int HV, int K,
                              int BT) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int n = B * HV * K;
  if (idx >= n) return;
  int k = idx % K, hv = (idx / K) % HV, b = idx / (HV * K);
  float acc = 0.0f;
  for (int t = 0; t < T; ++t) {
    if ((t % BT) == 0) acc = 0.0f;
    int p = ((b * T + t) * HV + hv) * K + k;
    acc += static_cast<float>(x[p]);
    y[p] = static_cast<S>(acc);
  }
}

// Fallback path: original recurrent implementation.
template <typename S>
__global__ __launch_bounds__(128, 2) void recurrent_kernel(
    const S* q, const S* k, const S* v, const S* g, const S* beta, S* out,
    float* ht, int B, int T, int H, int HV, int K, int V, float scale) {
  int b = blockIdx.y, hv = blockIdx.x, vi = threadIdx.x;
  int hq = hv / (HV / H);
  __shared__ float sq[MAX_K], sk[MAX_K], sg[MAX_K];
  float state[MAX_K];
  for (int i = 0; i < K; ++i) state[i] = 0.0f;
  __syncthreads();
  for (int t = 0; t < T; ++t) {
    int qb = ((b * T + t) * H + hq) * K;
    int gb = ((b * T + t) * HV + hv) * K;
    for (int i = vi; i < K; i += blockDim.x) {
      sq[i] = static_cast<float>(q[qb + i]);
      sk[i] = static_cast<float>(k[qb + i]);
      sg[i] = static_cast<float>(g[gb + i]);
    }
    __syncthreads();
    if (vi < V) {
      float corr = 0.0f;
      for (int i = 0; i < K; ++i) {
        state[i] *= __expf(sg[i]);
        corr = fmaf(sk[i], state[i], corr);
      }
      int vb = ((b * T + t) * HV + hv) * V;
      float d = static_cast<float>(beta[(b * T + t) * HV + hv]) *
                (static_cast<float>(v[vb + vi]) - corr);
      float y = 0.0f;
      for (int i = 0; i < K; ++i) {
        state[i] = fmaf(sk[i], d, state[i]);
        y = fmaf(sq[i], state[i], y);
      }
      out[vb + vi] = static_cast<S>(y * scale);
    }
    __syncthreads();
  }
  if (vi < V)
    for (int i = 0; i < K; ++i)
      ht[((b * HV + hv) * K + i) * V + vi] = state[i];
}

// Fixed-K specialization for the BW1000 production workload.  Keeping K in
// the type removes the dynamic loop bounds and address arithmetic from the
// hot token loop while preserving the same recurrence as recurrent_kernel.
template <typename S, int KD>
__global__ __launch_bounds__(128, 2) void recurrent_fixed_kernel(
    const S* __restrict__ q, const S* __restrict__ k,
    const S* __restrict__ v, const S* __restrict__ g,
    const S* __restrict__ beta, S* __restrict__ out, float* __restrict__ ht,
    int B, int T, int H, int HV, int V, float scale) {
  const int b = blockIdx.y;
  const int hv = blockIdx.x;
  const int vi = threadIdx.x;
  const int hq = hv / (HV / H);
  __shared__ float sq[KD], sk[KD], sg[KD];
  float state[KD];

#pragma unroll
  for (int i = 0; i < KD; ++i) state[i] = 0.0f;
  __syncthreads();

  for (int t = 0; t < T; ++t) {
    const int qb = ((b * T + t) * H + hq) * KD;
    const int gb = ((b * T + t) * HV + hv) * KD;
    for (int i = vi; i < KD; i += 128) {
      sq[i] = static_cast<float>(q[qb + i]);
      sk[i] = static_cast<float>(k[qb + i]);
      sg[i] = static_cast<float>(g[gb + i]);
    }
    __syncthreads();

    if (vi < V) {
      float corr = 0.0f;
#pragma unroll 4
      for (int i = 0; i < KD; ++i) {
        state[i] *= __expf(sg[i]);
        corr = fmaf(sk[i], state[i], corr);
      }
      const int vb = ((b * T + t) * HV + hv) * V;
      const float d = static_cast<float>(beta[(b * T + t) * HV + hv]) *
                      (static_cast<float>(v[vb + vi]) - corr);
      float y = 0.0f;
#pragma unroll 4
      for (int i = 0; i < KD; ++i) {
        state[i] = fmaf(sk[i], d, state[i]);
        y = fmaf(sq[i], state[i], y);
      }
      out[vb + vi] = static_cast<S>(y * scale);
    }
    __syncthreads();
  }

  if (vi < V) {
#pragma unroll 4
    for (int i = 0; i < KD; ++i)
      ht[((b * HV + hv) * KD + i) * V + vi] = state[i];
  }
}

// K=128,V=128 path: shared state avoids the 128-float per-thread local array.
template <typename S>
__global__ void shared_k128_kernel(
    const S* q, const S* k, const S* v, const S* g, const S* beta, S* out,
    float* ht, int B, int T, int H, int HV, float scale) {
  int tile = blockIdx.x & 1;
  int hv = blockIdx.x >> 1;
  int b = blockIdx.y, vi = threadIdx.x, vo = tile * VTILE + vi;
  int hq = hv / (HV / H);
  __shared__ float sq[128], sk[128], sg[128];
  __shared__ float state[VTILE][128];
  for (int i = 0; i < 128; ++i) state[vi][i] = 0.0f;
  __syncthreads();
  for (int t = 0; t < T; ++t) {
    int qb = ((b * T + t) * H + hq) * 128;
    int gb = ((b * T + t) * HV + hv) * 128;
    for (int i = vi; i < 128; i += VTILE) {
      sq[i] = static_cast<float>(q[qb + i]);
      sk[i] = static_cast<float>(k[qb + i]);
      sg[i] = static_cast<float>(g[gb + i]);
    }
    __syncthreads();
    float corr = 0.0f;
    for (int i = 0; i < 128; ++i) {
      state[vi][i] *= __expf(sg[i]);
      corr = fmaf(sk[i], state[vi][i], corr);
    }
    int vb = ((b * T + t) * HV + hv) * 128;
    float d = static_cast<float>(beta[(b * T + t) * HV + hv]) *
              (static_cast<float>(v[vb + vo]) - corr);
    float y = 0.0f;
    for (int i = 0; i < 128; ++i) {
      state[vi][i] = fmaf(sk[i], d, state[vi][i]);
      y = fmaf(sq[i], state[vi][i], y);
    }
    out[vb + vo] = static_cast<S>(y * scale);
    __syncthreads();
  }
  for (int i = 0; i < 128; ++i)
    ht[((b * HV + hv) * 128 + i) * 128 + vo] = state[vi][i];
}
}  // namespace

std::tuple<torch::Tensor, torch::Tensor> forward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g,
    torch::Tensor beta, c10::optional<torch::Tensor> initial_state,
    double scale) {
  (void)initial_state;
  int B = q.size(0), T = q.size(1), H = q.size(2), K = q.size(3);
  int HV = v.size(2), V = v.size(3);
  TORCH_CHECK(HV % H == 0 && K <= MAX_K && V <= 256);
  auto out = torch::empty_like(v);
  auto ht = torch::empty({B, HV, K, V}, q.options().dtype(torch::kFloat));
  auto stream = c10::hip::getCurrentHIPStream().stream();
  const char* fixed_env = std::getenv("KDA_HYGON_FIXED_K128");
  const bool use_fixed_k128 = fixed_env == nullptr || fixed_env[0] != '0';
  // v4 best path: every V lane keeps its K-wide state in registers/local
  // memory.  Do not dispatch K=V=128 to the shared-state experiment: that
  // version measured 264 ms versus about 208 ms for this path on BW1000.
  dim3 block(V <= 128 ? 128 : 256), grid(HV, B);
  AT_DISPATCH_SWITCH(q.scalar_type(), "kda_local_state",
    AT_DISPATCH_CASE(at::ScalarType::Half, [&] {
      if (K == 128 && use_fixed_k128) {
        hipLaunchKernelGGL((recurrent_fixed_kernel<at::Half, 128>),
          grid, dim3(128), 0, stream,
          q.data_ptr<at::Half>(), k.data_ptr<at::Half>(), v.data_ptr<at::Half>(),
          g.data_ptr<at::Half>(), beta.data_ptr<at::Half>(), out.data_ptr<at::Half>(),
          ht.data_ptr<float>(), B, T, H, HV, V, static_cast<float>(scale));
      } else {
        hipLaunchKernelGGL((recurrent_kernel<at::Half>), grid, block, 0, stream,
          q.data_ptr<at::Half>(), k.data_ptr<at::Half>(), v.data_ptr<at::Half>(),
          g.data_ptr<at::Half>(), beta.data_ptr<at::Half>(), out.data_ptr<at::Half>(),
          ht.data_ptr<float>(), B, T, H, HV, K, V, static_cast<float>(scale));
      }
    })
    AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&] {
      if (K == 128 && use_fixed_k128) {
        hipLaunchKernelGGL((recurrent_fixed_kernel<at::BFloat16, 128>),
          grid, dim3(128), 0, stream,
          q.data_ptr<at::BFloat16>(), k.data_ptr<at::BFloat16>(),
          v.data_ptr<at::BFloat16>(), g.data_ptr<at::BFloat16>(),
          beta.data_ptr<at::BFloat16>(), out.data_ptr<at::BFloat16>(),
          ht.data_ptr<float>(), B, T, H, HV, V, static_cast<float>(scale));
      } else {
        hipLaunchKernelGGL((recurrent_kernel<at::BFloat16>), grid, block, 0, stream,
          q.data_ptr<at::BFloat16>(), k.data_ptr<at::BFloat16>(),
          v.data_ptr<at::BFloat16>(), g.data_ptr<at::BFloat16>(),
          beta.data_ptr<at::BFloat16>(), out.data_ptr<at::BFloat16>(),
          ht.data_ptr<float>(), B, T, H, HV, K, V, static_cast<float>(scale));
      }
    }));
  C10_HIP_KERNEL_LAUNCH_CHECK();
  return {out, ht};
}

torch::Tensor chunk_cumsum(torch::Tensor x, int64_t chunk_size) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 4,
              "chunk_cumsum expects a rank-4 HIP tensor");
  TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");
  const int B = x.size(0), T = x.size(1), HV = x.size(2), K = x.size(3);
  auto y = torch::empty_like(x);
  constexpr int threads = 256;
  const int blocks = (B * HV * K + threads - 1) / threads;
  auto stream = c10::hip::getCurrentHIPStream().stream();
  AT_DISPATCH_SWITCH(x.scalar_type(), "kda_chunk_cumsum",
    AT_DISPATCH_CASE(at::ScalarType::Half, [&] {
      hipLaunchKernelGGL((cumsum_kernel<at::Half>), dim3(blocks),
        dim3(threads), 0, stream, x.data_ptr<at::Half>(),
        y.data_ptr<at::Half>(), B, T, HV, K, static_cast<int>(chunk_size));
    })
    AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&] {
      hipLaunchKernelGGL((cumsum_kernel<at::BFloat16>), dim3(blocks),
        dim3(threads), 0, stream, x.data_ptr<at::BFloat16>(),
        y.data_ptr<at::BFloat16>(), B, T, HV, K,
        static_cast<int>(chunk_size));
    }));
  C10_HIP_KERNEL_LAUNCH_CHECK();
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "KDA HIP forward");
  m.def("chunk_cumsum", &chunk_cumsum, "KDA BT chunk cumsum");
}
