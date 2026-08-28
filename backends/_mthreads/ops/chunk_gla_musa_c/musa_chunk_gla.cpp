// Copyright 2026 FlagOS Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <torch/extension.h>

#include <algorithm>
#include <cstdint>

#include <ATen/ATen.h>
#include <ATen/ops/empty.h>
#include <musa_runtime_api.h>

#include "torch_musa/csrc/core/MUSAStream.h"

// The optimized implementation is split into small MUSA kernels and batched
// muBLAS GEMMs. The latter turns the O(T*K*V) recurrent work into the same
// chunked matrix formulation used by the Triton implementation.
int launch_gla_forward(
    const void* q, const void* k, const void* v, const void* g, void* out,
    const float* initial_state, float* checkpoints, float* final_state,
    void* qbar, void* kbar, void* vf, float* h0, float* chunk_decay,
    float* chunk_update_scale, float* state_output_scale,
    float* chunk_updates, void* chunk_updates_low,
    void* a, void* out_low, float* out_fp32, int B, int T, int H, int K,
    int V, int chunk_size,
    int num_chunks, float scale, int dtype, musaStream_t stream,
    size_t shared_bytes);

int launch_gla_backward(
    const void* q, const void* k, const void* v, const void* g,
    const void* do_, const float* dht, const float* checkpoints,
    const float* chunk_decay, void* qbar, void* kbar, void* vf, void* dof,
    void* a, void* da, float* a_fp32, float* da_fp32, float* dqbar,
    float* dkbar, float* dv_local,
    float* dh0_local, float* dk_state, float* dv_state, float* dg_state,
    float* dh_chunks, void* gemm_scratch, void* state_scratch, void* dq,
    void* dk, void* dv, void* dg, float* dh0, int B, int T, int H, int K,
    int V, int chunk_size, int num_chunks, int state_scratch_chunks,
    float scale, int dtype, musaStream_t stream, size_t shared_bytes);

namespace {

void check_musa_tensor(const torch::Tensor& tensor, const char* name,
                       int device_index) {
  TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1,
              name, " must be a MUSA tensor");
  TORCH_CHECK(tensor.device().index() == device_index,
              name, " must be on the same MUSA device as q");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

int dtype_code(const torch::Tensor& q) {
  if (q.scalar_type() == torch::kFloat16) return 0;
  if (q.scalar_type() == torch::kBFloat16) return 1;
  TORCH_CHECK(q.scalar_type() == torch::kFloat32,
              "only fp16, bf16 and fp32 are supported");
  return 2;
}

// Use a larger native chunk than the public Triton path so the many small
// strided-batched GEMMs operate on fewer, larger matrices.  The chunked
// formulation remains unchanged; only the boundary granularity changes.
int gla_chunk_size(int T) {
  int chunk = 16;
  while (chunk < T && chunk < 128) chunk <<= 1;
  return chunk;
}

void check_same_dtype(const torch::Tensor& q, const torch::Tensor& k,
                      const torch::Tensor& v, const torch::Tensor& g) {
  TORCH_CHECK(q.scalar_type() == k.scalar_type() &&
                  q.scalar_type() == v.scalar_type() &&
                  q.scalar_type() == g.scalar_type(),
              "q, k, v and g must have the same dtype");
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> forward(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    const torch::Tensor& g, double scale, const torch::Tensor& initial_state,
    bool output_final_state) {
  TORCH_CHECK(q.dim() == 4, "q must have shape [B, T, H, K]");
  TORCH_CHECK(k.sizes() == q.sizes() && g.sizes() == q.sizes(),
              "q, k and g must have identical shapes");
  TORCH_CHECK(v.dim() == 4 && v.size(0) == q.size(0) &&
                  v.size(1) == q.size(1) && v.size(2) == q.size(2),
              "v must have shape [B, T, H, V]");

  const int device_index = q.device().index();
  check_musa_tensor(q, "q", device_index);
  check_musa_tensor(k, "k", device_index);
  check_musa_tensor(v, "v", device_index);
  check_musa_tensor(g, "g", device_index);
  check_same_dtype(q, k, v, g);

  const int B = q.size(0);
  const int T = q.size(1);
  const int H = q.size(2);
  const int K = q.size(3);
  const int V = v.size(3);
  const int chunk_size = gla_chunk_size(T);
  const int num_chunks = (T + chunk_size - 1) / chunk_size;
  const int BH = B * H;

  auto out = torch::empty_like(v);
  auto checkpoints = torch::empty({B, H, num_chunks + 1, K, V},
                                  q.options().dtype(torch::kFloat32));
  torch::Tensor final_state;
  if (output_final_state) {
    final_state = torch::empty({B, H, K, V},
                               q.options().dtype(torch::kFloat32));
  }

  auto qbar = torch::empty({BH, num_chunks, chunk_size, K}, q.options());
  auto kbar = torch::empty({BH, num_chunks, chunk_size, K}, q.options());
  auto vf = torch::empty({BH, num_chunks, chunk_size, V}, q.options());
  auto h0 = torch::empty({BH, num_chunks, K, V},
                         q.options().dtype(torch::kFloat32));
  auto chunk_decay = torch::empty({BH, num_chunks, K},
                                  q.options().dtype(torch::kFloat32));
  // The two factors are distinct when the adaptive shift is zero:
  // chunk_update_scale=exp(c_end-shift) is used by the state update, while
  // state_output_scale=exp(shift) restores the output state contribution.
  auto chunk_update_scale = torch::empty(
      {BH, num_chunks, K}, q.options().dtype(torch::kFloat32));
  auto state_output_scale = torch::empty(
      {BH, num_chunks, K}, q.options().dtype(torch::kFloat32));
  auto chunk_updates = torch::empty({BH, num_chunks, K, V},
                                     q.options().dtype(torch::kFloat32));
  torch::Tensor chunk_updates_low;
  if (q.scalar_type() == torch::kBFloat16 &&
      static_cast<int64_t>(K) * V > 8192) {
    chunk_updates_low = torch::empty({BH, num_chunks, K, V}, q.options());
  }
  auto a = torch::empty({BH, num_chunks, chunk_size, chunk_size}, q.options());
  torch::Tensor out_low;
  if (q.scalar_type() == torch::kBFloat16) {
    out_low = torch::empty({BH, num_chunks, chunk_size, V}, q.options());
  }
  auto out_fp32 = torch::empty({BH, num_chunks, chunk_size, V},
                               q.options().dtype(torch::kFloat32));

  const float* initial_ptr = nullptr;
  if (initial_state.defined() && initial_state.numel() != 0) {
    check_musa_tensor(initial_state, "initial_state", device_index);
    TORCH_CHECK(initial_state.scalar_type() == torch::kFloat32,
                "initial_state must be float32");
    TORCH_CHECK(initial_state.dim() == 4 && initial_state.size(0) == B &&
                    initial_state.size(1) == H && initial_state.size(2) == K &&
                    initial_state.size(3) == V,
                "initial_state must have shape [B, H, K, V]");
    initial_ptr = initial_state.data_ptr<float>();
  }

  auto stream = c10::musa::getCurrentMUSAStream(device_index).stream();
  const int status = launch_gla_forward(
      q.data_ptr(), k.data_ptr(), v.data_ptr(), g.data_ptr(), out.data_ptr(),
      initial_ptr, checkpoints.data_ptr<float>(),
      output_final_state ? final_state.data_ptr<float>() : nullptr,
       qbar.data_ptr(), kbar.data_ptr(), vf.data_ptr(),
       h0.data_ptr<float>(), chunk_decay.data_ptr<float>(),
       chunk_update_scale.data_ptr<float>(),
       state_output_scale.data_ptr<float>(),
       chunk_updates.data_ptr<float>(),
       chunk_updates_low.defined() ? chunk_updates_low.data_ptr() : nullptr,
       a.data_ptr(),
      out_low.defined() ? out_low.data_ptr() : nullptr,
      out_fp32.data_ptr<float>(), B, T, H, K, V,
      chunk_size, num_chunks, static_cast<float>(scale), dtype_code(q), stream,
      static_cast<size_t>(K) * V * sizeof(float) +
          static_cast<size_t>(K) * sizeof(float));
  TORCH_CHECK(status == 0, "optimized MUSA GLA forward launch failed: ", status);
  return {out, final_state, checkpoints, chunk_decay};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
backward(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
         const torch::Tensor& g, const torch::Tensor& do_,
         const torch::Tensor& dht, const torch::Tensor& initial_state,
         const torch::Tensor& checkpoints, const torch::Tensor& chunk_decay,
         double scale) {
  TORCH_CHECK(q.dim() == 4 && k.sizes() == q.sizes() &&
                  g.sizes() == q.sizes(),
              "invalid q/k/g shapes");
  TORCH_CHECK(v.dim() == 4 && v.size(0) == q.size(0) &&
                  v.size(1) == q.size(1) && v.size(2) == q.size(2),
              "invalid v shape");
  TORCH_CHECK(do_.dim() == 4 && do_.size(0) == q.size(0) &&
                  do_.size(1) == q.size(1) && do_.size(2) == q.size(2) &&
                  do_.size(3) == v.size(3),
              "do must have shape [B, T, H, V]");

  const int device_index = q.device().index();
  check_musa_tensor(q, "q", device_index);
  check_musa_tensor(k, "k", device_index);
  check_musa_tensor(v, "v", device_index);
  check_musa_tensor(g, "g", device_index);
  check_musa_tensor(do_, "do", device_index);
  check_musa_tensor(checkpoints, "checkpoints", device_index);
  check_musa_tensor(chunk_decay, "chunk_decay", device_index);
  TORCH_CHECK(dht.defined() &&
                  dht.device().type() == c10::DeviceType::PrivateUse1,
              "dht must be a MUSA tensor, including an empty tensor when unused");
  check_same_dtype(q, k, v, g);
  TORCH_CHECK(do_.scalar_type() == q.scalar_type(),
              "do must have the same dtype as q");
  TORCH_CHECK(checkpoints.scalar_type() == torch::kFloat32,
              "checkpoints must be float32");
  TORCH_CHECK(chunk_decay.scalar_type() == torch::kFloat32,
              "chunk_decay must be float32");

  const int B = q.size(0);
  const int T = q.size(1);
  const int H = q.size(2);
  const int K = q.size(3);
  const int V = v.size(3);
  const int chunk_size = gla_chunk_size(T);
  const int num_chunks = (T + chunk_size - 1) / chunk_size;
  const int BH = B * H;

  TORCH_CHECK(checkpoints.dim() == 5 && checkpoints.size(0) == B &&
                  checkpoints.size(1) == H &&
                  checkpoints.size(2) == num_chunks + 1 &&
                  checkpoints.size(3) == K && checkpoints.size(4) == V,
              "checkpoints has an invalid shape");
  TORCH_CHECK(chunk_decay.dim() == 3 && chunk_decay.size(0) == B * H &&
                  chunk_decay.size(1) == num_chunks &&
                  chunk_decay.size(2) == K,
              "chunk_decay has an invalid shape");

  auto dq = torch::empty_like(q);
  auto dk = torch::empty_like(k);
  auto dv = torch::empty_like(v);
  auto dg = torch::empty_like(g);
  auto dh0 = torch::empty({B, H, K, V}, q.options().dtype(torch::kFloat32));

  // Packed [BH, NC, BT, *] workspaces. They are intentionally private to
  // this call; autograd only needs the compact chunk checkpoints.
  auto qbar = torch::empty({BH, num_chunks, chunk_size, K}, q.options());
  auto kbar = torch::empty({BH, num_chunks, chunk_size, K}, q.options());
  auto vf = torch::empty({BH, num_chunks, chunk_size, V}, q.options());
  auto dof = torch::empty({BH, num_chunks, chunk_size, V}, q.options());
  auto a = torch::empty({BH, num_chunks, chunk_size, chunk_size}, q.options());
  auto da = torch::empty_like(a);
  torch::Tensor a_fp32;
  torch::Tensor da_fp32;
  // The scalar FP32 reconstruction is only an accuracy escape hatch for the
  // short-wide consistency cases.  Normal benchmark chunks use the batched
  // muBLAS path, so avoid allocating two O(BH*NC*BT^2) FP32 tensors there.
  if (q.scalar_type() == torch::kBFloat16 &&
      static_cast<int64_t>(K) * V > 8192 && chunk_size <= 16) {
    a_fp32 = torch::empty({BH, num_chunks, chunk_size, chunk_size},
                          q.options().dtype(torch::kFloat32));
    da_fp32 = torch::empty_like(a_fp32);
  }
  auto dqbar = torch::empty({BH, num_chunks, chunk_size, K},
                            q.options().dtype(torch::kFloat32));
  auto dkbar = torch::empty_like(dqbar);
  auto dv_local = torch::empty({BH, num_chunks, chunk_size, V},
                               q.options().dtype(torch::kFloat32));
  auto dh0_local = torch::empty({BH, num_chunks, K, V},
                                q.options().dtype(torch::kFloat32));
  auto dk_state = torch::empty_like(dqbar);
  auto dv_state = torch::empty_like(dv_local);
  auto dg_state = torch::empty_like(dqbar);
  auto dh_chunks = torch::empty({BH, num_chunks, K, V},
                                q.options().dtype(torch::kFloat32));

  // BF16 muBLAS cannot write FP32 output directly on the deployed runtime.
  // Keep a separate, compact temporary for those four GEMMs.  Its leading
  // dimension is padded to the largest matrix used by the batch so the same
  // buffer can be reused for dV_local, dQbar, dKbar and dH0_local.
  torch::Tensor gemm_scratch;
  int64_t gemm_scratch_stride = 0;
  if (q.scalar_type() == torch::kBFloat16) {
    gemm_scratch_stride = std::max(
        {static_cast<int64_t>(chunk_size) * V,
         static_cast<int64_t>(chunk_size) * K,
         static_cast<int64_t>(K) * V});
    gemm_scratch = torch::empty(
        {static_cast<int64_t>(BH) * num_chunks, gemm_scratch_stride},
        q.options());
  }

  // State-token backward only needs the h_{t-1} snapshots for the chunks
  // currently being processed.  Reuse a bounded number of chunks instead of
  // allocating [BH, NC, BT, K, V], which is O(B*T*H*K*V).
  constexpr size_t kStateScratchBudget = 1ULL << 30;  // 1 GiB
  const size_t state_bytes_per_chunk =
      static_cast<size_t>(BH) * chunk_size * K * V * q.element_size();
  int state_scratch_chunks = 1;
  if (state_bytes_per_chunk != 0) {
    const size_t budget_chunks = kStateScratchBudget / state_bytes_per_chunk;
    state_scratch_chunks = static_cast<int>(std::max<size_t>(1, budget_chunks));
  }
  state_scratch_chunks = std::min(state_scratch_chunks, num_chunks);
  auto state_scratch = torch::empty(
      {BH, state_scratch_chunks, chunk_size, K, V}, q.options());

  const float* dht_ptr = nullptr;
  if (dht.numel() != 0) {
    check_musa_tensor(dht, "dht", device_index);
    TORCH_CHECK(dht.scalar_type() == torch::kFloat32,
                "dht must be float32");
    TORCH_CHECK(dht.numel() == static_cast<int64_t>(BH) * K * V,
                "dht must have shape [B, H, K, V]");
    dht_ptr = dht.data_ptr<float>();
  }
  if (initial_state.defined() && initial_state.numel() != 0) {
    check_musa_tensor(initial_state, "initial_state", device_index);
    TORCH_CHECK(initial_state.scalar_type() == torch::kFloat32,
                "initial_state must be float32");
  }

  auto stream = c10::musa::getCurrentMUSAStream(device_index).stream();
  void* gemm_scratch_ptr = gemm_scratch.defined()
                                ? gemm_scratch.data_ptr()
                                : nullptr;
  const int status = launch_gla_backward(
      q.data_ptr(), k.data_ptr(), v.data_ptr(), g.data_ptr(), do_.data_ptr(),
      dht_ptr, checkpoints.data_ptr<float>(),
      chunk_decay.data_ptr<float>(), qbar.data_ptr(), kbar.data_ptr(),
      vf.data_ptr(), dof.data_ptr(), a.data_ptr(), da.data_ptr(),
      a_fp32.defined() ? a_fp32.data_ptr<float>() : nullptr,
      da_fp32.defined() ? da_fp32.data_ptr<float>() : nullptr,
      dqbar.data_ptr<float>(),
      dkbar.data_ptr<float>(), dv_local.data_ptr<float>(),
      dh0_local.data_ptr<float>(), dk_state.data_ptr<float>(),
      dv_state.data_ptr<float>(), dg_state.data_ptr<float>(),
      dh_chunks.data_ptr<float>(), gemm_scratch_ptr, state_scratch.data_ptr(),
      dq.data_ptr(), dk.data_ptr(), dv.data_ptr(), dg.data_ptr(),
      dh0.data_ptr<float>(), B, T, H, K, V, chunk_size, num_chunks,
      state_scratch_chunks, static_cast<float>(scale), dtype_code(q), stream,
      static_cast<size_t>(2) * K * V * sizeof(float) +
          static_cast<size_t>(K) * sizeof(float));
  TORCH_CHECK(status == 0,
              "optimized MUSA GLA backward launch failed: ", status);
  return {dq, dk, dv, dg, dh0};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "optimized native MUSA GLA forward");
  m.def("backward", &backward, "optimized native MUSA GLA backward");
}
