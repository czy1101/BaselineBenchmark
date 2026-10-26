#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPException.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <vector>

// NSA production kernels for Hygon/BW1000.  One wave (64 lanes) owns one
// query head.  The recurrence keeps the softmax normalization in registers,
// avoiding the large [T, topk, block_size] temporary used by a PyTorch path.
namespace {
constexpr int WAVE = 64;

__device__ __forceinline__ float wave_sum(float x) {
  x += __shfl_down(x, 32, WAVE);
  x += __shfl_down(x, 16, WAVE);
  x += __shfl_down(x, 8, WAVE);
  x += __shfl_down(x, 4, WAVE);
  x += __shfl_down(x, 2, WAVE);
  x += __shfl_down(x, 1, WAVE);
  return x;
}

__device__ __forceinline__ float wave_bcast(float x) {
  return __shfl(x, 0, WAVE);
}

template <typename scalar_t>
__global__ __launch_bounds__(WAVE, 2) void selected_kernel(
    const scalar_t* __restrict__ q, const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v, const int32_t* __restrict__ blocks,
    scalar_t* __restrict__ out, int B, int T, int HQ, int H, int D, int DV,
    int S, int block_size, float scale, int block_count) {
  const int lane = static_cast<int>(threadIdx.x);
  const int work = static_cast<int>(blockIdx.x);
  const int total = B * T * HQ;
  if (work >= total) return;
  const int hq = work % HQ;
  const int tmp = work / HQ;
  const int t = tmp % T;
  const int b = tmp / T;
  const int group = HQ / H;
  const int hv = hq / group;
  const int qbase = ((b * T + t) * HQ + hq) * D;
  const int obase = ((b * T + t) * HQ + hq) * DV;

  // Up to 256 value columns are covered by four register accumulators.
  float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
  float m = -INFINITY;
  float l = 0.f;
  const int count = block_count < S ? block_count : S;
  for (int si = 0; si < count; ++si) {
    const int ib = ((b * T + t) * H + hv) * S + si;
    const int block = blocks[ib];
    for (int j = 0; j < block_size; ++j) {
      const int token = block * block_size + j;
      if (token > t || token < 0 || token >= T) continue;
      const int kbase = ((b * T + token) * H + hv) * D;
      float dot = 0.f;
      for (int d = lane; d < D; d += WAVE)
        dot = fmaf(static_cast<float>(q[qbase + d]),
                   static_cast<float>(k[kbase + d]), dot);
      dot = wave_bcast(wave_sum(dot)) * scale;
      const float nm = fmaxf(m, dot);
      const float old_a = (l == 0.f) ? 0.f : expf(m - nm);
      const float w = expf(dot - nm);
      l = l * old_a + w;
      if (lane < DV) {
        const int vbase = ((b * T + token) * H + hv) * DV + lane;
        const float x = static_cast<float>(v[vbase]);
        if (lane < 64) acc0 = acc0 * old_a + w * x;
      }
      if (lane + 64 < DV) {
        const int vbase = ((b * T + token) * H + hv) * DV + lane + 64;
        acc1 = acc1 * old_a + w * static_cast<float>(v[vbase]);
      }
      if (lane + 128 < DV) {
        const int vbase = ((b * T + token) * H + hv) * DV + lane + 128;
        acc2 = acc2 * old_a + w * static_cast<float>(v[vbase]);
      }
      if (lane + 192 < DV) {
        const int vbase = ((b * T + token) * H + hv) * DV + lane + 192;
        acc3 = acc3 * old_a + w * static_cast<float>(v[vbase]);
      }
      m = nm;
    }
  }
  const float inv = l > 0.f ? 1.f / l : 0.f;
  if (lane < DV) out[obase + lane] = static_cast<scalar_t>(acc0 * inv);
  if (lane + 64 < DV) out[obase + lane + 64] = static_cast<scalar_t>(acc1 * inv);
  if (lane + 128 < DV) out[obase + lane + 128] = static_cast<scalar_t>(acc2 * inv);
  if (lane + 192 < DV) out[obase + lane + 192] = static_cast<scalar_t>(acc3 * inv);
}

template <typename scalar_t>
__global__ void pack_selected_kernel(const scalar_t* __restrict__ k,
                                     const scalar_t* __restrict__ v,
                                     const int32_t* __restrict__ blocks,
                                     scalar_t* __restrict__ kp,
                                     scalar_t* __restrict__ vp,
                                     int B, int T, int H, int D, int DV,
                                     int S, int bs, int t0, int Q, int L) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t nk = static_cast<int64_t>(B) * H * Q * L * D;
  const int64_t nv = static_cast<int64_t>(B) * H * Q * L * DV;
  if (idx < nk) {
    int64_t z = idx;
    const int d = z % D; z /= D;
    const int l = z % L; z /= L;
    const int q = z % Q; z /= Q;
    const int h = z % H; const int b = static_cast<int>(z / H);
    const int token = blocks[((b * T + t0 + q) * H + h) * S + l / bs] * bs + (l % bs);
    kp[idx] = (token >= 0 && token < T && token <= t0 + q)
        ? k[((b * T + token) * H + h) * D + d] : static_cast<scalar_t>(0);
  } else {
    const int64_t j = idx - nk;
    if (j >= nv) return;
    int64_t z = j;
    const int d = z % DV; z /= DV;
    const int l = z % L; z /= L;
    const int q = z % Q; z /= Q;
    const int h = z % H; const int b = static_cast<int>(z / H);
    const int token = blocks[((b * T + t0 + q) * H + h) * S + l / bs] * bs + (l % bs);
    vp[j] = (token >= 0 && token < T && token <= t0 + q)
        ? v[((b * T + token) * H + h) * DV + d] : static_cast<scalar_t>(0);
  }
}

// Fused selected kernel: four query waves belonging to the same KV head share
// one K vector in LDS.  This removes the repeated global K traffic that is
// particularly costly for GQA (HQ/H=16 in the target workloads).  Each wave
// keeps its own online-softmax state and PV accumulators.
template <typename scalar_t>
__global__ __launch_bounds__(256, 2) void selected_fused4_kernel(
    const scalar_t* __restrict__ q, const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v, const int32_t* __restrict__ blocks,
    scalar_t* __restrict__ out, int B, int T, int HQ, int H, int D, int DV,
    int S, int block_size, float scale, int block_count) {
  const int tid = static_cast<int>(threadIdx.x);
  const int wave = tid / WAVE;
  const int lane = tid & (WAVE - 1);
  const int work = static_cast<int>(blockIdx.x);
  const int groups = HQ / H;
  const int total = B * T * (HQ / 4);
  if (work >= total || groups < 4) return;
  const int hq4 = work % (HQ / 4);
  const int tmp = work / (HQ / 4);
  const int t = tmp % T;
  const int b = tmp / T;
  const int hq = hq4 * 4 + wave;
  const int hv = hq / groups;
  if (hq >= HQ || (hq4 * 4) / groups != hv) return;
  const int qbase = ((b * T + t) * HQ + hq) * D;
  const int obase = ((b * T + t) * HQ + hq) * DV;
  __shared__ float sk[256];
  float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
  float m = -INFINITY, norm = 0.f;
  const int count = block_count < S ? block_count : S;
  for (int si = 0; si < count; ++si) {
    const int ib = ((b * T + t) * H + hv) * S + si;
    const int block = blocks[ib];
    for (int j = 0; j < block_size; ++j) {
      const int token = block * block_size + j;
      if (token > t || token < 0 || token >= T) continue;
      const int kbase = ((b * T + token) * H + hv) * D;
      if (wave == 0 && lane < D) sk[lane] = static_cast<float>(k[kbase + lane]);
      __syncthreads();
      float dot = 0.f;
      for (int d = lane; d < D; d += WAVE)
        dot = fmaf(static_cast<float>(q[qbase + d]), sk[d], dot);
      dot = wave_bcast(wave_sum(dot)) * scale;
      const float nm = fmaxf(m, dot);
      const float old_a = (norm == 0.f) ? 0.f : expf(m - nm);
      const float w = expf(dot - nm);
      norm = norm * old_a + w;
      if (lane < DV) {
        const int vbase = ((b * T + token) * H + hv) * DV + lane;
        acc0 = acc0 * old_a + w * static_cast<float>(v[vbase]);
      }
      if (lane + 64 < DV) {
        const int vbase = ((b * T + token) * H + hv) * DV + lane + 64;
        acc1 = acc1 * old_a + w * static_cast<float>(v[vbase]);
      }
      if (lane + 128 < DV) {
        const int vbase = ((b * T + token) * H + hv) * DV + lane + 128;
        acc2 = acc2 * old_a + w * static_cast<float>(v[vbase]);
      }
      if (lane + 192 < DV) {
        const int vbase = ((b * T + token) * H + hv) * DV + lane + 192;
        acc3 = acc3 * old_a + w * static_cast<float>(v[vbase]);
      }
      m = nm;
      __syncthreads();
    }
  }
  const float inv = norm > 0.f ? 1.f / norm : 0.f;
  if (lane < DV) out[obase + lane] = static_cast<scalar_t>(acc0 * inv);
  if (lane + 64 < DV) out[obase + lane + 64] = static_cast<scalar_t>(acc1 * inv);
  if (lane + 128 < DV) out[obase + lane + 128] = static_cast<scalar_t>(acc2 * inv);
  if (lane + 192 < DV) out[obase + lane + 192] = static_cast<scalar_t>(acc3 * inv);
}

template <typename scalar_t>
__global__ __launch_bounds__(WAVE, 2) void compression_kernel(
    const scalar_t* __restrict__ q, const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v, scalar_t* __restrict__ out,
    float* __restrict__ lse, int B, int T, int HQ, int H, int D, int DV,
    int TC, int block_size, float scale) {
  const int lane = static_cast<int>(threadIdx.x);
  const int work = static_cast<int>(blockIdx.x);
  if (work >= B * T * HQ) return;
  const int hq = work % HQ;
  const int tmp = work / HQ;
  const int t = tmp % T;
  const int b = tmp / T;
  const int hv = hq / (HQ / H);
  const int qbase = ((b * T + t) * HQ + hq) * D;
  const int obase = ((b * T + t) * HQ + hq) * DV;
  const int n = (t + 1) / block_size;
  float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
  float m = -INFINITY, norm = 0.f;
  for (int j = 0; j < n && j < TC; ++j) {
    const int kbase = ((b * TC + j) * H + hv) * D;
    float dot = 0.f;
    for (int d = lane; d < D; d += WAVE)
      dot = fmaf(static_cast<float>(q[qbase + d]), static_cast<float>(k[kbase + d]), dot);
    dot = wave_bcast(wave_sum(dot)) * scale;
    const float nm = fmaxf(m, dot);
    const float old_a = (norm == 0.f) ? 0.f : expf(m - nm);
    const float w = expf(dot - nm);
    norm = norm * old_a + w;
    if (lane < DV) {
      const int vb = ((b * TC + j) * H + hv) * DV + lane;
      acc0 = acc0 * old_a + w * static_cast<float>(v[vb]);
    }
    if (lane + 64 < DV) {
      const int vb = ((b * TC + j) * H + hv) * DV + lane + 64;
      acc1 = acc1 * old_a + w * static_cast<float>(v[vb]);
    }
    if (lane + 128 < DV) {
      const int vb = ((b * TC + j) * H + hv) * DV + lane + 128;
      acc2 = acc2 * old_a + w * static_cast<float>(v[vb]);
    }
    if (lane + 192 < DV) {
      const int vb = ((b * TC + j) * H + hv) * DV + lane + 192;
      acc3 = acc3 * old_a + w * static_cast<float>(v[vb]);
    }
    m = nm;
  }
  const float inv = norm > 0.f ? 1.f / norm : 0.f;
  if (lane < DV) out[obase + lane] = static_cast<scalar_t>(acc0 * inv);
  if (lane + 64 < DV) out[obase + lane + 64] = static_cast<scalar_t>(acc1 * inv);
  if (lane + 128 < DV) out[obase + lane + 128] = static_cast<scalar_t>(acc2 * inv);
  if (lane + 192 < DV) out[obase + lane + 192] = static_cast<scalar_t>(acc3 * inv);
  if (lane == 0) lse[((b * T + t) * HQ + hq)] = norm > 0.f ? logf(norm) + m : 0.f;
}

template <typename scalar_t>
void launch_selected(const torch::Tensor& q, const torch::Tensor& k,
                     const torch::Tensor& v, const torch::Tensor& blocks,
                     torch::Tensor& out, int block_count, int block_size,
                     float scale) {
  const int B = q.size(0), T = q.size(1), HQ = q.size(2), D = q.size(3);
  const int H = k.size(2), DV = v.size(3), S = blocks.size(3);
  auto stream = c10::hip::getCurrentHIPStream().stream();
  hipLaunchKernelGGL((selected_kernel<scalar_t>), dim3(B * T * HQ), dim3(WAVE), 0,
                     stream, q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(),
                     v.data_ptr<scalar_t>(), blocks.data_ptr<int32_t>(),
                     out.data_ptr<scalar_t>(), B, T, HQ, H, D, DV, S,
                     block_size, scale, block_count);
  C10_HIP_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void launch_selected_fused4(const torch::Tensor& q, const torch::Tensor& k,
                            const torch::Tensor& v, const torch::Tensor& blocks,
                            torch::Tensor& out, int block_count, int block_size,
                            float scale) {
  const int B = q.size(0), T = q.size(1), HQ = q.size(2), D = q.size(3);
  const int H = k.size(2), DV = v.size(3), S = blocks.size(3);
  auto stream = c10::hip::getCurrentHIPStream().stream();
  const int work = B * T * (HQ / 4);
  hipLaunchKernelGGL((selected_fused4_kernel<scalar_t>),
                     dim3((work + 1) / 1), dim3(256), 0, stream,
                     q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(),
                     v.data_ptr<scalar_t>(), blocks.data_ptr<int32_t>(),
                     out.data_ptr<scalar_t>(), B, T, HQ, H, D, DV, S,
                     block_size, scale, block_count);
  C10_HIP_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
std::vector<torch::Tensor> pack_selected(const torch::Tensor& k,
                                         const torch::Tensor& v,
                                         const torch::Tensor& blocks,
                                         int t0, int qcount, int block_size) {
  const int B = k.size(0), T = k.size(1), H = k.size(2), D = k.size(3);
  const int DV = v.size(3), S = blocks.size(3), L = S * block_size;
  auto kp = torch::empty({B, H, qcount, L, D}, k.options());
  auto vp = torch::empty({B, H, qcount, L, DV}, v.options());
  const int64_t nk = static_cast<int64_t>(B) * H * qcount * L * D;
  const int64_t nv = static_cast<int64_t>(B) * H * qcount * L * DV;
  const int threads = 256;
  const int64_t total = nk + nv;
  const int blocks_n = static_cast<int>((total + threads - 1) / threads);
  auto stream = c10::hip::getCurrentHIPStream().stream();
  hipLaunchKernelGGL((pack_selected_kernel<scalar_t>), dim3(blocks_n), dim3(threads), 0,
                     stream, k.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(),
                     blocks.data_ptr<int32_t>(), kp.data_ptr<scalar_t>(),
                     vp.data_ptr<scalar_t>(), B, T, H, D, DV, S, block_size,
                     t0, qcount, L);
  C10_HIP_KERNEL_LAUNCH_CHECK();
  return {kp, vp};
}

template <typename scalar_t>
void launch_compression(const torch::Tensor& q, const torch::Tensor& k,
                        const torch::Tensor& v, torch::Tensor& out,
                        torch::Tensor& lse, int block_size, float scale) {
  const int B = q.size(0), T = q.size(1), HQ = q.size(2), D = q.size(3);
  const int H = k.size(2), DV = v.size(3), TC = k.size(1);
  auto stream = c10::hip::getCurrentHIPStream().stream();
  hipLaunchKernelGGL((compression_kernel<scalar_t>), dim3(B * T * HQ), dim3(WAVE), 0,
                     stream, q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(),
                     v.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(),
                     lse.data_ptr<float>(), B, T, HQ, H, D, DV, TC,
                     block_size, scale);
  C10_HIP_KERNEL_LAUNCH_CHECK();
}

void check_common(const torch::Tensor& q, const torch::Tensor& k,
                  const torch::Tensor& v) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "NSA tensors must be on HCU");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "NSA tensors must be rank 4");
  TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(),
              "q/k/v dtype mismatch");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
              "NSA tensors must be contiguous");
  TORCH_CHECK(q.size(0) == k.size(0) && q.size(1) == k.size(1), "shape mismatch");
  TORCH_CHECK(k.size(2) == v.size(2) && k.size(3) == q.size(3), "shape mismatch");
  TORCH_CHECK(q.size(2) % k.size(2) == 0, "HQ must be divisible by H");
  TORCH_CHECK(q.size(3) <= 256 && v.size(3) <= 256, "D/DV must be <= 256");
}

void check_compression(const torch::Tensor& q, const torch::Tensor& k,
                       const torch::Tensor& v) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "NSA tensors must be on HCU");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "NSA tensors must be rank 4");
  TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(),
              "q/k/v dtype mismatch");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
              "NSA tensors must be contiguous");
  TORCH_CHECK(q.size(0) == k.size(0) && k.size(0) == v.size(0), "batch mismatch");
  TORCH_CHECK(k.size(2) == v.size(2) && q.size(2) % k.size(2) == 0,
              "compressed head mismatch");
  TORCH_CHECK(k.size(3) == q.size(3) && v.size(3) <= 256 && q.size(3) <= 256,
              "compressed dimension mismatch");
}
}  // namespace

torch::Tensor nsa_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                          torch::Tensor block_indices, int64_t block_count,
                          int64_t block_size, double scale) {
  check_common(q, k, v);
  TORCH_CHECK(block_indices.is_cuda() && block_indices.scalar_type() == torch::kInt,
              "block_indices must be CUDA int32");
  TORCH_CHECK(block_indices.is_contiguous() && block_indices.dim() == 4,
              "block_indices must be contiguous [B,T,H,S]");
  TORCH_CHECK(block_indices.size(0) == q.size(0) && block_indices.size(1) == q.size(1) &&
              block_indices.size(2) == k.size(2), "block_indices shape mismatch");
  auto out = torch::empty({q.size(0), q.size(1), q.size(2), v.size(3)}, q.options());
  const char* fused_env = std::getenv("NSA_HYGON_FUSED4");
  const bool fused4 = fused_env == nullptr || fused_env[0] != '0';
  const bool can_fuse4 = q.size(2) % 4 == 0 && (q.size(2) / k.size(2)) >= 4;
  if (q.scalar_type() == torch::kFloat16) {
    if (fused4 && can_fuse4)
      launch_selected_fused4<at::Half>(q, k, v, block_indices, out, (int)block_count, (int)block_size, (float)scale);
    else
      launch_selected<at::Half>(q, k, v, block_indices, out, (int)block_count, (int)block_size, (float)scale);
  } else if (q.scalar_type() == torch::kBFloat16) {
    if (fused4 && can_fuse4)
      launch_selected_fused4<at::BFloat16>(q, k, v, block_indices, out, (int)block_count, (int)block_size, (float)scale);
    else
      launch_selected<at::BFloat16>(q, k, v, block_indices, out, (int)block_count, (int)block_size, (float)scale);
  }
  else TORCH_CHECK(false, "NSA supports FP16/BF16");
  return out;
}

std::vector<torch::Tensor> nsa_pack_selected(torch::Tensor k, torch::Tensor v,
                                              torch::Tensor block_indices,
                                              int64_t t0, int64_t qcount,
                                              int64_t block_size) {
  TORCH_CHECK(k.is_cuda() && v.is_cuda() && block_indices.is_cuda(), "NSA tensors must be on HCU");
  TORCH_CHECK(k.is_contiguous() && v.is_contiguous() && block_indices.is_contiguous(), "NSA tensors must be contiguous");
  TORCH_CHECK(block_indices.scalar_type() == torch::kInt && block_indices.dim() == 4, "invalid block indices");
  TORCH_CHECK(k.scalar_type() == v.scalar_type(), "K/V dtype mismatch");
  TORCH_CHECK(k.size(0) == block_indices.size(0) && k.size(1) == block_indices.size(1) &&
              k.size(2) == block_indices.size(2), "block indices shape mismatch");
  TORCH_CHECK(t0 >= 0 && qcount > 0 && t0 + qcount <= k.size(1) && block_size > 0,
              "invalid tile range");
  if (k.scalar_type() == torch::kFloat16)
    return pack_selected<at::Half>(k, v, block_indices, (int)t0, (int)qcount, (int)block_size);
  if (k.scalar_type() == torch::kBFloat16)
    return pack_selected<at::BFloat16>(k, v, block_indices, (int)t0, (int)qcount, (int)block_size);
  TORCH_CHECK(false, "NSA supports FP16/BF16");
}

std::vector<torch::Tensor> nsa_compression(torch::Tensor q, torch::Tensor k,
                                            torch::Tensor v, int64_t block_size,
                                            double scale) {
  check_compression(q, k, v);
  auto out = torch::empty({q.size(0), q.size(1), q.size(2), v.size(3)}, q.options());
  auto lse = torch::empty({q.size(0), q.size(1), q.size(2)}, q.options().dtype(torch::kFloat));
  if (q.scalar_type() == torch::kFloat16)
    launch_compression<at::Half>(q, k, v, out, lse, (int)block_size, (float)scale);
  else if (q.scalar_type() == torch::kBFloat16)
    launch_compression<at::BFloat16>(q, k, v, out, lse, (int)block_size, (float)scale);
  else TORCH_CHECK(false, "NSA supports FP16/BF16");
  return {out, lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nsa_forward", &nsa_forward, "NSA selected attention (HIP)");
  m.def("nsa_pack_selected", &nsa_pack_selected, "NSA selected K/V pack (HIP)");
  m.def("nsa_compression", &nsa_compression, "NSA compression attention (HIP)");
}
