#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/BFloat16.h>
#include <torch/all.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>

#include "../../type_convert.cuh"

namespace {

constexpr int kHeadDim = 128;
constexpr int kValueDim = 128;
constexpr int kThreads = 64;
constexpr int kValueTile = 64;
constexpr int kV2Threads = 512;
constexpr int kV2KeyLanes = 8;
constexpr int kV2KeysPerThread = kHeadDim / kV2KeyLanes;
constexpr int kV3ValueTile = 32;
constexpr int kV3Threads = 256;
constexpr int kV3KeyLanes = 8;
constexpr int kV3KeysPerThread = kHeadDim / kV3KeyLanes;
constexpr int kMacaWarpSize = 64;
constexpr unsigned long long kFullWarpMask64 =
    0xffffffffffffffffULL;
static_assert(kMacaWarpSize % kV2KeyLanes == 0);
constexpr float kNormEpsilon = 1.0e-12f;

using BFloat16Converter = vllm::_typeConvert<c10::BFloat16>;
using DeviceBFloat16 = BFloat16Converter::hip_type;

static_assert(
    sizeof(DeviceBFloat16) == sizeof(c10::BFloat16),
    "device and PyTorch BF16 types must have identical size");

__device__ __forceinline__ float load_bf16(
    const DeviceBFloat16* pointer,
    int64_t offset) {
  return BFloat16Converter::convert(pointer[offset]);
}

__device__ __forceinline__ void store_bf16(
    DeviceBFloat16* pointer,
    int64_t offset,
    float value) {
  pointer[offset] = BFloat16Converter::convert(value);
}

__device__ __forceinline__ float kda_sigmoid(float value) {
  if (value >= 0.0f) {
    const float z = expf(-value);
    return 1.0f / (1.0f + z);
  }
  const float z = expf(value);
  return z / (1.0f + z);
}

// Correctness-first dense KDA forward.
//
// One block owns one [K=128, Vtile=64] state tile. Two blocks cover V=128.
// The external state layout is [B,H,V,K], while the shared recurrence state
// is [K,Vtile], so initial/final state copies explicitly transpose layouts.
__global__ void chunk_kda_fwd_bf16_d128_v1_kernel(
    DeviceBFloat16* __restrict__ out,
    float* __restrict__ final_state,
    const DeviceBFloat16* __restrict__ q,
    const DeviceBFloat16* __restrict__ k,
    const DeviceBFloat16* __restrict__ v,
    const DeviceBFloat16* __restrict__ g,
    const DeviceBFloat16* __restrict__ beta,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    const float* __restrict__ initial_state,
    int64_t batch_size,
    int64_t sequence_length,
    int64_t num_heads,
    float scale,
    float lower_bound) {
  __shared__ float state[kHeadDim * kValueTile];
  __shared__ float q_hat[kHeadDim];
  __shared__ float k_hat[kHeadDim];
  __shared__ float alpha[kHeadDim];
  __shared__ float q_reduce[kThreads];
  __shared__ float k_reduce[kThreads];
  __shared__ float beta_value;
  __shared__ float gate_scale;

  const int thread_id = static_cast<int>(threadIdx.x);
  const int value_tile = static_cast<int>(blockIdx.y);
  const int value_index = value_tile * kValueTile + thread_id;

  const int64_t batch_head = static_cast<int64_t>(blockIdx.x);
  const int64_t batch_index = batch_head / num_heads;
  const int64_t head_index = batch_head - batch_index * num_heads;

  if (thread_id == 0) {
    gate_scale = expf(A_log[head_index]);
  }

  // Cooperative, contiguous external [V,K] -> internal [K,Vtile] copy.
  for (int linear = thread_id;
       linear < kHeadDim * kValueTile;
       linear += kThreads) {
    const int local_value = linear / kHeadDim;
    const int key_index = linear - local_value * kHeadDim;
    const int global_value = value_tile * kValueTile + local_value;
    const int shared_offset = key_index * kValueTile + local_value;

    if (initial_state != nullptr) {
      const int64_t external_offset =
          (batch_head * kValueDim + global_value) * kHeadDim + key_index;
      state[shared_offset] = initial_state[external_offset];
    } else {
      state[shared_offset] = 0.0f;
    }
  }
  __syncthreads();

  for (int64_t token = 0; token < sequence_length; ++token) {
    const int key_index_0 = thread_id;
    const int key_index_1 = thread_id + kThreads;
    const int64_t token_base =
        ((batch_index * sequence_length + token) * num_heads + head_index) *
        kHeadDim;

    const float q0 = load_bf16(q, token_base + key_index_0);
    const float q1 = load_bf16(q, token_base + key_index_1);
    const float k0 = load_bf16(k, token_base + key_index_0);
    const float k1 = load_bf16(k, token_base + key_index_1);

    q_reduce[thread_id] = q0 * q0 + q1 * q1;
    k_reduce[thread_id] = k0 * k0 + k1 * k1;
    __syncthreads();

    for (int offset = kThreads / 2; offset > 0; offset /= 2) {
      if (thread_id < offset) {
        q_reduce[thread_id] += q_reduce[thread_id + offset];
        k_reduce[thread_id] += k_reduce[thread_id + offset];
      }
      __syncthreads();
    }

    const float q_inverse_norm =
        1.0f / fmaxf(sqrtf(q_reduce[0]), kNormEpsilon);
    const float k_inverse_norm =
        1.0f / fmaxf(sqrtf(k_reduce[0]), kNormEpsilon);

    q_hat[key_index_0] = q0 * q_inverse_norm;
    q_hat[key_index_1] = q1 * q_inverse_norm;
    k_hat[key_index_0] = k0 * k_inverse_norm;
    k_hat[key_index_1] = k1 * k_inverse_norm;

    const float gate_input_0 =
        gate_scale *
        (load_bf16(g, token_base + key_index_0) +
         dt_bias[head_index * kHeadDim + key_index_0]);
    const float gate_input_1 =
        gate_scale *
        (load_bf16(g, token_base + key_index_1) +
         dt_bias[head_index * kHeadDim + key_index_1]);

    float gate_0 = lower_bound * kda_sigmoid(gate_input_0);
    float gate_1 = lower_bound * kda_sigmoid(gate_input_1);

    // Keep exponent arguments inside the safe-gate interval.
    gate_0 = fminf(0.0f, fmaxf(lower_bound, gate_0));
    gate_1 = fminf(0.0f, fmaxf(lower_bound, gate_1));

    alpha[key_index_0] = expf(gate_0);
    alpha[key_index_1] = expf(gate_1);

    if (thread_id == 0) {
      const int64_t beta_offset =
          (batch_index * sequence_length + token) * num_heads + head_index;
      beta_value = kda_sigmoid(load_bf16(beta, beta_offset));
    }
    __syncthreads();

    // Decay the state first, then predict v from the decayed state.
    float prediction = 0.0f;
    for (int key_index = 0; key_index < kHeadDim; ++key_index) {
      const int shared_offset = key_index * kValueTile + thread_id;
      const float decayed = state[shared_offset] * alpha[key_index];
      state[shared_offset] = decayed;
      prediction = fmaf(k_hat[key_index], decayed, prediction);
    }

    const float input_value =
        load_bf16(v, token_base + value_index);
    const float residual =
        beta_value * (input_value - prediction);

    // Apply the delta update and compute output from the updated state.
    float output_value = 0.0f;
    for (int key_index = 0; key_index < kHeadDim; ++key_index) {
      const int shared_offset = key_index * kValueTile + thread_id;
      const float updated =
          fmaf(k_hat[key_index], residual, state[shared_offset]);
      state[shared_offset] = updated;
      output_value = fmaf(q_hat[key_index], updated, output_value);
    }

    store_bf16(out, token_base + value_index, scale * output_value);

    // No lane may overwrite q_hat/k_hat/alpha for the next token early.
    __syncthreads();
  }

  __syncthreads();

  // Cooperative internal [K,Vtile] -> external [V,K] transpose.
  for (int linear = thread_id;
       linear < kHeadDim * kValueTile;
       linear += kThreads) {
    const int local_value = linear / kHeadDim;
    const int key_index = linear - local_value * kHeadDim;
    const int global_value = value_tile * kValueTile + local_value;
    const int shared_offset = key_index * kValueTile + local_value;
    const int64_t external_offset =
        (batch_head * kValueDim + global_value) * kHeadDim + key_index;

    final_state[external_offset] = state[shared_offset];
  }
}


// Register-state recurrent KDA forward.
//
// One block still owns one [K=128, Vtile=64] state tile, but the state is
// distributed across 512 threads instead of being repeatedly read from and
// written to shared memory.
//
// Thread mapping:
//   local_value = thread_id / 8
//   key_lane    = thread_id % 8
//   key_index   = key_lane + item * 8, item in [0, 16)
//
// Each thread keeps 16 FP32 state elements in registers across the full T loop.
__global__ void chunk_kda_fwd_bf16_d128_v2_kernel(
    DeviceBFloat16* __restrict__ out,
    float* __restrict__ final_state,
    const DeviceBFloat16* __restrict__ q,
    const DeviceBFloat16* __restrict__ k,
    const DeviceBFloat16* __restrict__ v,
    const DeviceBFloat16* __restrict__ g,
    const DeviceBFloat16* __restrict__ beta,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    const float* __restrict__ initial_state,
    int64_t batch_size,
    int64_t sequence_length,
    int64_t num_heads,
    float scale,
    float lower_bound) {
  __shared__ float q_hat[kHeadDim];
  __shared__ float k_hat[kHeadDim];
  __shared__ float alpha[kHeadDim];
  __shared__ float q_reduce[kThreads];
  __shared__ float k_reduce[kThreads];
  __shared__ float partial[kValueTile * kV2KeyLanes];
  __shared__ float residual[kValueTile];
  __shared__ float beta_value;
  __shared__ float gate_scale;
  __shared__ float q_inverse_norm;
  __shared__ float k_inverse_norm;

  const int thread_id = static_cast<int>(threadIdx.x);
  const int local_value = thread_id / kV2KeyLanes;
  const int key_lane =
      thread_id - local_value * kV2KeyLanes;
  const int value_tile = static_cast<int>(blockIdx.y);
  const int value_index =
      value_tile * kValueTile + local_value;

  const int64_t batch_head = static_cast<int64_t>(blockIdx.x);
  const int64_t batch_index = batch_head / num_heads;
  const int64_t head_index =
      batch_head - batch_index * num_heads;

  (void)batch_size;

  float state_local[kV2KeysPerThread];

#pragma unroll
  for (int item = 0; item < kV2KeysPerThread; ++item) {
    const int key_index = key_lane + item * kV2KeyLanes;
    if (initial_state != nullptr) {
      const int64_t external_offset =
          (batch_head * kValueDim + value_index) * kHeadDim +
          key_index;
      state_local[item] = initial_state[external_offset];
    } else {
      state_local[item] = 0.0f;
    }
  }

  if (thread_id == 0) {
    gate_scale = expf(A_log[head_index]);
  }

  for (int64_t token = 0; token < sequence_length; ++token) {
    const int64_t token_base =
        ((batch_index * sequence_length + token) * num_heads +
         head_index) *
        kHeadDim;

    // The first wave loads q/k contiguously and produces 64 norm partials.
    if (thread_id < kThreads) {
      const int key_index_0 = thread_id;
      const int key_index_1 = thread_id + kThreads;

      const float q0 = load_bf16(q, token_base + key_index_0);
      const float q1 = load_bf16(q, token_base + key_index_1);
      const float k0 = load_bf16(k, token_base + key_index_0);
      const float k1 = load_bf16(k, token_base + key_index_1);

      q_hat[key_index_0] = q0;
      q_hat[key_index_1] = q1;
      k_hat[key_index_0] = k0;
      k_hat[key_index_1] = k1;

      q_reduce[thread_id] = q0 * q0 + q1 * q1;
      k_reduce[thread_id] = k0 * k0 + k1 * k1;
    }

    // One lane per V loads the input value. The buffer is reused for residual.
    if (key_lane == 0) {
      residual[local_value] =
          load_bf16(v, token_base + value_index);
    }

    if (thread_id == 0) {
      const int64_t beta_offset =
          (batch_index * sequence_length + token) * num_heads +
          head_index;
      beta_value = kda_sigmoid(load_bf16(beta, beta_offset));
    }
    __syncthreads();

    // A single lane finishes the small 64-way norm reduction. This avoids
    // six additional full-block barriers on the 512-thread block.
    if (thread_id == 0) {
      float q_sum = 0.0f;
      float k_sum = 0.0f;
#pragma unroll
      for (int lane = 0; lane < kThreads; ++lane) {
        q_sum += q_reduce[lane];
        k_sum += k_reduce[lane];
      }
      q_inverse_norm =
          1.0f / fmaxf(sqrtf(q_sum), kNormEpsilon);
      k_inverse_norm =
          1.0f / fmaxf(sqrtf(k_sum), kNormEpsilon);
    }
    __syncthreads();

    // Normalize q/k and build the per-channel decay once per token.
    if (thread_id < kHeadDim) {
      const int key_index = thread_id;

      q_hat[key_index] *= q_inverse_norm;
      k_hat[key_index] *= k_inverse_norm;

      const float gate_input =
          gate_scale *
          (load_bf16(g, token_base + key_index) +
           dt_bias[head_index * kHeadDim + key_index]);

      float gate = lower_bound * kda_sigmoid(gate_input);
      gate = fminf(0.0f, fmaxf(lower_bound, gate));
      alpha[key_index] = expf(gate);
    }
    __syncthreads();

    // Decay the private state and form one prediction partial per thread.
    float prediction_partial = 0.0f;
#pragma unroll
    for (int item = 0; item < kV2KeysPerThread; ++item) {
      const int key_index = key_lane + item * kV2KeyLanes;
      state_local[item] *= alpha[key_index];
      prediction_partial =
          fmaf(k_hat[key_index], state_local[item],
               prediction_partial);
    }

    partial[thread_id] = prediction_partial;
    __syncthreads();

    // One lane per V reduces the eight K-lane partials.
    if (key_lane == 0) {
      const int partial_base = local_value * kV2KeyLanes;
      float prediction = 0.0f;
#pragma unroll
      for (int lane = 0; lane < kV2KeyLanes; ++lane) {
        prediction += partial[partial_base + lane];
      }

      residual[local_value] =
          beta_value *
          (residual[local_value] - prediction);
    }
    __syncthreads();

    // Apply the delta update and form one output partial per thread.
    const float residual_value = residual[local_value];
    float output_partial = 0.0f;

#pragma unroll
    for (int item = 0; item < kV2KeysPerThread; ++item) {
      const int key_index = key_lane + item * kV2KeyLanes;
      state_local[item] =
          fmaf(k_hat[key_index], residual_value,
               state_local[item]);
      output_partial =
          fmaf(q_hat[key_index], state_local[item],
               output_partial);
    }

    partial[thread_id] = output_partial;
    __syncthreads();

    if (key_lane == 0) {
      const int partial_base = local_value * kV2KeyLanes;
      float output_value = 0.0f;
#pragma unroll
      for (int lane = 0; lane < kV2KeyLanes; ++lane) {
        output_value += partial[partial_base + lane];
      }

      store_bf16(
          out,
          token_base + value_index,
          scale * output_value);
    }

    // No thread may start the next token while output lanes still consume
    // the current token's partial buffer.
    __syncthreads();
  }

#pragma unroll
  for (int item = 0; item < kV2KeysPerThread; ++item) {
    const int key_index = key_lane + item * kV2KeyLanes;
    const int64_t external_offset =
        (batch_head * kValueDim + value_index) * kHeadDim +
        key_index;
    final_state[external_offset] = state_local[item];
  }
}

// V3 A/B candidate: [K=128, Vtile=32] on 256 threads.
//
// Compared with V2, each thread still owns 16 FP32 state elements. Only the
// value-tile width and block size change, doubling grid-level parallelism.
__global__ void chunk_kda_fwd_bf16_d128_v3_kernel(
    DeviceBFloat16* __restrict__ out,
    float* __restrict__ final_state,
    const DeviceBFloat16* __restrict__ q,
    const DeviceBFloat16* __restrict__ k,
    const DeviceBFloat16* __restrict__ v,
    const DeviceBFloat16* __restrict__ g,
    const DeviceBFloat16* __restrict__ beta,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    const float* __restrict__ initial_state,
    int64_t batch_size,
    int64_t sequence_length,
    int64_t num_heads,
    float scale,
    float lower_bound) {
  __shared__ float q_hat[kHeadDim];
  __shared__ float k_hat[kHeadDim];
  __shared__ float alpha[kHeadDim];
  __shared__ float q_reduce[kThreads];
  __shared__ float k_reduce[kThreads];
  __shared__ float partial[kV3ValueTile * kV3KeyLanes];
  __shared__ float residual[kV3ValueTile];
  __shared__ float beta_value;
  __shared__ float gate_scale;
  __shared__ float q_inverse_norm;
  __shared__ float k_inverse_norm;

  const int thread_id = static_cast<int>(threadIdx.x);
  const int local_value = thread_id / kV3KeyLanes;
  const int key_lane =
      thread_id - local_value * kV3KeyLanes;
  const int value_tile = static_cast<int>(blockIdx.y);
  const int value_index =
      value_tile * kV3ValueTile + local_value;

  const int64_t batch_head = static_cast<int64_t>(blockIdx.x);
  const int64_t batch_index = batch_head / num_heads;
  const int64_t head_index =
      batch_head - batch_index * num_heads;

  (void)batch_size;

  float state_local[kV3KeysPerThread];

#pragma unroll
  for (int item = 0; item < kV3KeysPerThread; ++item) {
    const int key_index = key_lane + item * kV3KeyLanes;
    if (initial_state != nullptr) {
      const int64_t external_offset =
          (batch_head * kValueDim + value_index) * kHeadDim +
          key_index;
      state_local[item] = initial_state[external_offset];
    } else {
      state_local[item] = 0.0f;
    }
  }

  if (thread_id == 0) {
    gate_scale = expf(A_log[head_index]);
  }

  for (int64_t token = 0; token < sequence_length; ++token) {
    const int64_t token_base =
        ((batch_index * sequence_length + token) * num_heads +
         head_index) *
        kHeadDim;

    // The first wave loads q/k contiguously and produces 64 norm partials.
    if (thread_id < kThreads) {
      const int key_index_0 = thread_id;
      const int key_index_1 = thread_id + kThreads;

      const float q0 = load_bf16(q, token_base + key_index_0);
      const float q1 = load_bf16(q, token_base + key_index_1);
      const float k0 = load_bf16(k, token_base + key_index_0);
      const float k1 = load_bf16(k, token_base + key_index_1);

      q_hat[key_index_0] = q0;
      q_hat[key_index_1] = q1;
      k_hat[key_index_0] = k0;
      k_hat[key_index_1] = k1;

      q_reduce[thread_id] = q0 * q0 + q1 * q1;
      k_reduce[thread_id] = k0 * k0 + k1 * k1;
    }

    // One lane per V loads the input value. The buffer is reused for residual.
    if (key_lane == 0) {
      residual[local_value] =
          load_bf16(v, token_base + value_index);
    }

    if (thread_id == 0) {
      const int64_t beta_offset =
          (batch_index * sequence_length + token) * num_heads +
          head_index;
      beta_value = kda_sigmoid(load_bf16(beta, beta_offset));
    }
    __syncthreads();

    // A single lane finishes the small 64-way norm reduction. This avoids
    // six additional full-block barriers on the 256-thread block.
    if (thread_id == 0) {
      float q_sum = 0.0f;
      float k_sum = 0.0f;
#pragma unroll
      for (int lane = 0; lane < kThreads; ++lane) {
        q_sum += q_reduce[lane];
        k_sum += k_reduce[lane];
      }
      q_inverse_norm =
          1.0f / fmaxf(sqrtf(q_sum), kNormEpsilon);
      k_inverse_norm =
          1.0f / fmaxf(sqrtf(k_sum), kNormEpsilon);
    }
    __syncthreads();

    // Normalize q/k and build the per-channel decay once per token.
    if (thread_id < kHeadDim) {
      const int key_index = thread_id;

      q_hat[key_index] *= q_inverse_norm;
      k_hat[key_index] *= k_inverse_norm;

      const float gate_input =
          gate_scale *
          (load_bf16(g, token_base + key_index) +
           dt_bias[head_index * kHeadDim + key_index]);

      float gate = lower_bound * kda_sigmoid(gate_input);
      gate = fminf(0.0f, fmaxf(lower_bound, gate));
      alpha[key_index] = expf(gate);
    }
    __syncthreads();

    // Decay the private state and form one prediction partial per thread.
    float prediction_partial = 0.0f;
#pragma unroll
    for (int item = 0; item < kV3KeysPerThread; ++item) {
      const int key_index = key_lane + item * kV3KeyLanes;
      state_local[item] *= alpha[key_index];
      prediction_partial =
          fmaf(k_hat[key_index], state_local[item],
               prediction_partial);
    }

    partial[thread_id] = prediction_partial;
    __syncthreads();

    // One lane per V reduces the eight K-lane partials.
    if (key_lane == 0) {
      const int partial_base = local_value * kV3KeyLanes;
      float prediction = 0.0f;
#pragma unroll
      for (int lane = 0; lane < kV3KeyLanes; ++lane) {
        prediction += partial[partial_base + lane];
      }

      residual[local_value] =
          beta_value *
          (residual[local_value] - prediction);
    }
    __syncthreads();

    // Apply the delta update and form one output partial per thread.
    const float residual_value = residual[local_value];
    float output_partial = 0.0f;

#pragma unroll
    for (int item = 0; item < kV3KeysPerThread; ++item) {
      const int key_index = key_lane + item * kV3KeyLanes;
      state_local[item] =
          fmaf(k_hat[key_index], residual_value,
               state_local[item]);
      output_partial =
          fmaf(q_hat[key_index], state_local[item],
               output_partial);
    }

    partial[thread_id] = output_partial;
    __syncthreads();

    if (key_lane == 0) {
      const int partial_base = local_value * kV3KeyLanes;
      float output_value = 0.0f;
#pragma unroll
      for (int lane = 0; lane < kV3KeyLanes; ++lane) {
        output_value += partial[partial_base + lane];
      }

      store_bf16(
          out,
          token_base + value_index,
          scale * output_value);
    }

    // No thread may start the next token while output lanes still consume
    // the current token's partial buffer.
    __syncthreads();
  }

#pragma unroll
  for (int item = 0; item < kV3KeysPerThread; ++item) {
    const int key_index = key_lane + item * kV3KeyLanes;
    const int64_t external_offset =
        (batch_head * kValueDim + value_index) * kHeadDim +
        key_index;
    final_state[external_offset] = state_local[item];
  }
}

// V4 A/B candidate: warp-scope synchronization for warp-local exchanges.
//
// Thread/value mapping and arithmetic are identical to V2. The prediction,
// residual and output-partial exchanges never cross a 64-lane MACA warp, so
// only those three barriers use __syncwarp(). Cross-warp producer/consumer
// barriers and the end-of-token barrier remain full-block __syncthreads().
__global__ void chunk_kda_fwd_bf16_d128_v4_kernel(
    DeviceBFloat16* __restrict__ out,
    float* __restrict__ final_state,
    const DeviceBFloat16* __restrict__ q,
    const DeviceBFloat16* __restrict__ k,
    const DeviceBFloat16* __restrict__ v,
    const DeviceBFloat16* __restrict__ g,
    const DeviceBFloat16* __restrict__ beta,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    const float* __restrict__ initial_state,
    int64_t batch_size,
    int64_t sequence_length,
    int64_t num_heads,
    float scale,
    float lower_bound) {
  __shared__ float q_hat[kHeadDim];
  __shared__ float k_hat[kHeadDim];
  __shared__ float alpha[kHeadDim];
  __shared__ float q_reduce[kThreads];
  __shared__ float k_reduce[kThreads];
  __shared__ float partial[kValueTile * kV2KeyLanes];
  __shared__ float residual[kValueTile];
  __shared__ float beta_value;
  __shared__ float gate_scale;
  __shared__ float q_inverse_norm;
  __shared__ float k_inverse_norm;

  const int thread_id = static_cast<int>(threadIdx.x);
  const int local_value = thread_id / kV2KeyLanes;
  const int key_lane =
      thread_id - local_value * kV2KeyLanes;
  const int value_tile = static_cast<int>(blockIdx.y);
  const int value_index =
      value_tile * kValueTile + local_value;

  const int64_t batch_head = static_cast<int64_t>(blockIdx.x);
  const int64_t batch_index = batch_head / num_heads;
  const int64_t head_index =
      batch_head - batch_index * num_heads;

  (void)batch_size;

  float state_local[kV2KeysPerThread];

#pragma unroll
  for (int item = 0; item < kV2KeysPerThread; ++item) {
    const int key_index = key_lane + item * kV2KeyLanes;
    if (initial_state != nullptr) {
      const int64_t external_offset =
          (batch_head * kValueDim + value_index) * kHeadDim +
          key_index;
      state_local[item] = initial_state[external_offset];
    } else {
      state_local[item] = 0.0f;
    }
  }

  if (thread_id == 0) {
    gate_scale = expf(A_log[head_index]);
  }

  for (int64_t token = 0; token < sequence_length; ++token) {
    const int64_t token_base =
        ((batch_index * sequence_length + token) * num_heads +
         head_index) *
        kHeadDim;

    // The first wave loads q/k contiguously and produces 64 norm partials.
    if (thread_id < kThreads) {
      const int key_index_0 = thread_id;
      const int key_index_1 = thread_id + kThreads;

      const float q0 = load_bf16(q, token_base + key_index_0);
      const float q1 = load_bf16(q, token_base + key_index_1);
      const float k0 = load_bf16(k, token_base + key_index_0);
      const float k1 = load_bf16(k, token_base + key_index_1);

      q_hat[key_index_0] = q0;
      q_hat[key_index_1] = q1;
      k_hat[key_index_0] = k0;
      k_hat[key_index_1] = k1;

      q_reduce[thread_id] = q0 * q0 + q1 * q1;
      k_reduce[thread_id] = k0 * k0 + k1 * k1;
    }

    // One lane per V loads the input value. The buffer is reused for residual.
    if (key_lane == 0) {
      residual[local_value] =
          load_bf16(v, token_base + value_index);
    }

    if (thread_id == 0) {
      const int64_t beta_offset =
          (batch_index * sequence_length + token) * num_heads +
          head_index;
      beta_value = kda_sigmoid(load_bf16(beta, beta_offset));
    }
    __syncthreads();

    // A single lane finishes the small 64-way norm reduction. This avoids
    // six additional full-block barriers on the 512-thread block.
    if (thread_id == 0) {
      float q_sum = 0.0f;
      float k_sum = 0.0f;
#pragma unroll
      for (int lane = 0; lane < kThreads; ++lane) {
        q_sum += q_reduce[lane];
        k_sum += k_reduce[lane];
      }
      q_inverse_norm =
          1.0f / fmaxf(sqrtf(q_sum), kNormEpsilon);
      k_inverse_norm =
          1.0f / fmaxf(sqrtf(k_sum), kNormEpsilon);
    }
    __syncthreads();

    // Normalize q/k and build the per-channel decay once per token.
    if (thread_id < kHeadDim) {
      const int key_index = thread_id;

      q_hat[key_index] *= q_inverse_norm;
      k_hat[key_index] *= k_inverse_norm;

      const float gate_input =
          gate_scale *
          (load_bf16(g, token_base + key_index) +
           dt_bias[head_index * kHeadDim + key_index]);

      float gate = lower_bound * kda_sigmoid(gate_input);
      gate = fminf(0.0f, fmaxf(lower_bound, gate));
      alpha[key_index] = expf(gate);
    }
    __syncthreads();

    // Decay the private state and form one prediction partial per thread.
    float prediction_partial = 0.0f;
#pragma unroll
    for (int item = 0; item < kV2KeysPerThread; ++item) {
      const int key_index = key_lane + item * kV2KeyLanes;
      state_local[item] *= alpha[key_index];
      prediction_partial =
          fmaf(k_hat[key_index], state_local[item],
               prediction_partial);
    }

    partial[thread_id] = prediction_partial;
    __syncwarp();

    // One lane per V reduces the eight K-lane partials.
    if (key_lane == 0) {
      const int partial_base = local_value * kV2KeyLanes;
      float prediction = 0.0f;
#pragma unroll
      for (int lane = 0; lane < kV2KeyLanes; ++lane) {
        prediction += partial[partial_base + lane];
      }

      residual[local_value] =
          beta_value *
          (residual[local_value] - prediction);
    }
    __syncwarp();

    // Apply the delta update and form one output partial per thread.
    const float residual_value = residual[local_value];
    float output_partial = 0.0f;

#pragma unroll
    for (int item = 0; item < kV2KeysPerThread; ++item) {
      const int key_index = key_lane + item * kV2KeyLanes;
      state_local[item] =
          fmaf(k_hat[key_index], residual_value,
               state_local[item]);
      output_partial =
          fmaf(q_hat[key_index], state_local[item],
               output_partial);
    }

    partial[thread_id] = output_partial;
    __syncwarp();

    if (key_lane == 0) {
      const int partial_base = local_value * kV2KeyLanes;
      float output_value = 0.0f;
#pragma unroll
      for (int lane = 0; lane < kV2KeyLanes; ++lane) {
        output_value += partial[partial_base + lane];
      }

      store_bf16(
          out,
          token_base + value_index,
          scale * output_value);
    }

    // No thread may start the next token while output lanes still consume
    // the current token's partial buffer.
    __syncthreads();
  }

#pragma unroll
  for (int item = 0; item < kV2KeysPerThread; ++item) {
    const int key_index = key_lane + item * kV2KeyLanes;
    const int64_t external_offset =
        (batch_head * kValueDim + value_index) * kHeadDim +
        key_index;
    final_state[external_offset] = state_local[item];
  }
}

// V5 is the default tile-64/warp implementation.
//
// V4 arithmetic and thread/value ownership are retained, while independent
// width-8 K-lane reductions use warp shuffle instead of shared-memory exchange.
// This changes FP32 addition order, so final-state low bits need not be bitwise
// identical to V4. Set MCOPLIB_CHUNK_KDA_REDUCTION=shared for the V4 fallback.
__global__ void chunk_kda_fwd_bf16_d128_v5_kernel(
    DeviceBFloat16* __restrict__ out,
    float* __restrict__ final_state,
    const DeviceBFloat16* __restrict__ q,
    const DeviceBFloat16* __restrict__ k,
    const DeviceBFloat16* __restrict__ v,
    const DeviceBFloat16* __restrict__ g,
    const DeviceBFloat16* __restrict__ beta,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    const float* __restrict__ initial_state,
    int64_t batch_size,
    int64_t sequence_length,
    int64_t num_heads,
    float scale,
    float lower_bound) {
  __shared__ float q_hat[kHeadDim];
  __shared__ float k_hat[kHeadDim];
  __shared__ float alpha[kHeadDim];
  __shared__ float q_reduce[kThreads];
  __shared__ float k_reduce[kThreads];
  __shared__ float beta_value;
  __shared__ float gate_scale;
  __shared__ float q_inverse_norm;
  __shared__ float k_inverse_norm;

  const int thread_id = static_cast<int>(threadIdx.x);
  const int local_value = thread_id / kV2KeyLanes;
  const int key_lane =
      thread_id - local_value * kV2KeyLanes;
  const int value_tile = static_cast<int>(blockIdx.y);
  const int value_index =
      value_tile * kValueTile + local_value;

  const int64_t batch_head = static_cast<int64_t>(blockIdx.x);
  const int64_t batch_index = batch_head / num_heads;
  const int64_t head_index =
      batch_head - batch_index * num_heads;

  (void)batch_size;

  float state_local[kV2KeysPerThread];

#pragma unroll
  for (int item = 0; item < kV2KeysPerThread; ++item) {
    const int key_index = key_lane + item * kV2KeyLanes;
    if (initial_state != nullptr) {
      const int64_t external_offset =
          (batch_head * kValueDim + value_index) * kHeadDim +
          key_index;
      state_local[item] = initial_state[external_offset];
    } else {
      state_local[item] = 0.0f;
    }
  }

  if (thread_id == 0) {
    gate_scale = expf(A_log[head_index]);
  }

  for (int64_t token = 0; token < sequence_length; ++token) {
    const int64_t token_base =
        ((batch_index * sequence_length + token) * num_heads +
         head_index) *
        kHeadDim;

    // The first wave loads q/k contiguously and produces 64 norm partials.
    if (thread_id < kThreads) {
      const int key_index_0 = thread_id;
      const int key_index_1 = thread_id + kThreads;

      const float q0 = load_bf16(q, token_base + key_index_0);
      const float q1 = load_bf16(q, token_base + key_index_1);
      const float k0 = load_bf16(k, token_base + key_index_0);
      const float k1 = load_bf16(k, token_base + key_index_1);

      q_hat[key_index_0] = q0;
      q_hat[key_index_1] = q1;
      k_hat[key_index_0] = k0;
      k_hat[key_index_1] = k1;

      q_reduce[thread_id] = q0 * q0 + q1 * q1;
      k_reduce[thread_id] = k0 * k0 + k1 * k1;
    }

    if (thread_id == 0) {
      const int64_t beta_offset =
          (batch_index * sequence_length + token) * num_heads +
          head_index;
      beta_value = kda_sigmoid(load_bf16(beta, beta_offset));
    }
    __syncthreads();

    // A single lane finishes the small 64-way norm reduction. This avoids
    // six additional full-block barriers on the 512-thread block.
    if (thread_id == 0) {
      float q_sum = 0.0f;
      float k_sum = 0.0f;
#pragma unroll
      for (int lane = 0; lane < kThreads; ++lane) {
        q_sum += q_reduce[lane];
        k_sum += k_reduce[lane];
      }
      q_inverse_norm =
          1.0f / fmaxf(sqrtf(q_sum), kNormEpsilon);
      k_inverse_norm =
          1.0f / fmaxf(sqrtf(k_sum), kNormEpsilon);
    }
    __syncthreads();

    // Normalize q/k and build the per-channel decay once per token.
    if (thread_id < kHeadDim) {
      const int key_index = thread_id;

      q_hat[key_index] *= q_inverse_norm;
      k_hat[key_index] *= k_inverse_norm;

      const float gate_input =
          gate_scale *
          (load_bf16(g, token_base + key_index) +
           dt_bias[head_index * kHeadDim + key_index]);

      float gate = lower_bound * kda_sigmoid(gate_input);
      gate = fminf(0.0f, fmaxf(lower_bound, gate));
      alpha[key_index] = expf(gate);
    }
    __syncthreads();

    // Load V after shared q/k/alpha publication so this register does not
    // span the preceding producer/consumer barriers.
    float input_value = 0.0f;
    if (key_lane == 0) {
      input_value = load_bf16(v, token_base + value_index);
    }

    // Decay the private state and form one prediction partial per thread.
    float prediction_partial = 0.0f;
#pragma unroll
    for (int item = 0; item < kV2KeysPerThread; ++item) {
      const int key_index = key_lane + item * kV2KeyLanes;
      state_local[item] *= alpha[key_index];
      prediction_partial =
          fmaf(k_hat[key_index], state_local[item],
               prediction_partial);
    }

    // Reduce each aligned width-8 K subgroup entirely in registers.
#pragma unroll
    for (int offset = kV2KeyLanes / 2;
         offset > 0;
         offset >>= 1) {
      prediction_partial += __shfl_xor_sync(
          kFullWarpMask64,
          prediction_partial,
          offset,
          kV2KeyLanes);
    }

    float residual_value = 0.0f;
    if (key_lane == 0) {
      residual_value =
          beta_value * (input_value - prediction_partial);
    }

    // All warp lanes participate; lane zero is relative to each width-8 group.
    residual_value = __shfl_sync(
        kFullWarpMask64,
        residual_value,
        0,
        kV2KeyLanes);

    // Apply the delta update and form one output partial per thread.
    float output_partial = 0.0f;

#pragma unroll
    for (int item = 0; item < kV2KeysPerThread; ++item) {
      const int key_index = key_lane + item * kV2KeyLanes;
      state_local[item] =
          fmaf(k_hat[key_index], residual_value,
               state_local[item]);
      output_partial =
          fmaf(q_hat[key_index], state_local[item],
               output_partial);
    }

    // Reduce output contributions in the same width-8 subgroup.
#pragma unroll
    for (int offset = kV2KeyLanes / 2;
         offset > 0;
         offset >>= 1) {
      output_partial += __shfl_xor_sync(
          kFullWarpMask64,
          output_partial,
          offset,
          kV2KeyLanes);
    }

    if (key_lane == 0) {
      store_bf16(
          out,
          token_base + value_index,
          scale * output_partial);
    }

    // No warp may overwrite q/k/alpha for the next token while another
    // warp is still consuming the current token's shared inputs.
    __syncthreads();
  }

#pragma unroll
  for (int item = 0; item < kV2KeysPerThread; ++item) {
    const int key_index = key_lane + item * kV2KeyLanes;
    const int64_t external_offset =
        (batch_head * kValueDim + value_index) * kHeadDim +
        key_index;
    final_state[external_offset] = state_local[item];
  }
}

// Packed-varlen V5 forward.
//
// q/k/v/g/out use the official packed layout [1,T_total,H,128].
// cu_seqlens partitions T_total into independent sequences. Each block owns
// one [K=128,Vtile=64] state tile for one sequence/head, preserving V5's
// arithmetic order and external [sequence,H,V,K] state layout.
template <typename IndexType>
__global__ void chunk_kda_varlen_fwd_bf16_d128_v5_kernel(
    DeviceBFloat16* __restrict__ out,
    float* __restrict__ final_state,
    const DeviceBFloat16* __restrict__ q,
    const DeviceBFloat16* __restrict__ k,
    const DeviceBFloat16* __restrict__ v,
    const DeviceBFloat16* __restrict__ g,
    const DeviceBFloat16* __restrict__ beta,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    const float* __restrict__ initial_state,
    const IndexType* __restrict__ cu_seqlens,
    int64_t total_tokens,
    int64_t num_heads,
    float scale,
    float lower_bound) {
  __shared__ float q_hat[kHeadDim];
  __shared__ float k_hat[kHeadDim];
  __shared__ float alpha[kHeadDim];
  __shared__ float q_reduce[kThreads];
  __shared__ float k_reduce[kThreads];
  __shared__ float beta_value;
  __shared__ float gate_scale;
  __shared__ float q_inverse_norm;
  __shared__ float k_inverse_norm;
  __shared__ int64_t sequence_start;
  __shared__ int64_t sequence_end;

  const int thread_id = static_cast<int>(threadIdx.x);
  const int local_value = thread_id / kV2KeyLanes;
  const int key_lane =
      thread_id - local_value * kV2KeyLanes;
  const int value_tile = static_cast<int>(blockIdx.y);
  const int value_index =
      value_tile * kValueTile + local_value;

  const int64_t sequence_head =
      static_cast<int64_t>(blockIdx.x);
  const int64_t sequence_index = sequence_head / num_heads;
  const int64_t head_index =
      sequence_head - sequence_index * num_heads;

  if (thread_id == 0) {
    sequence_start =
        static_cast<int64_t>(cu_seqlens[sequence_index]);
    sequence_end =
        static_cast<int64_t>(cu_seqlens[sequence_index + 1]);
  }
  __syncthreads();

  // Device-side memory-safety guard. Complete partition validity
  // (zero start, monotonicity, no gaps/overlap, final offset=T_total)
  // remains a trusted caller contract to avoid a host synchronization.
  if (sequence_start < 0 ||
      sequence_end < sequence_start ||
      sequence_end > total_tokens) {
    return;
  }

  float state_local[kV2KeysPerThread];

#pragma unroll
  for (int item = 0; item < kV2KeysPerThread; ++item) {
    const int key_index = key_lane + item * kV2KeyLanes;
    if (initial_state != nullptr) {
      const int64_t external_offset =
          (sequence_head * kValueDim + value_index) * kHeadDim +
          key_index;
      state_local[item] = initial_state[external_offset];
    } else {
      state_local[item] = 0.0f;
    }
  }

  if (thread_id == 0) {
    gate_scale = expf(A_log[head_index]);
  }

  for (int64_t token = sequence_start;
       token < sequence_end;
       ++token) {
    const int64_t token_base =
        (token * num_heads + head_index) * kHeadDim;

    // The first wave loads q/k contiguously and produces 64 norm partials.
    if (thread_id < kThreads) {
      const int key_index_0 = thread_id;
      const int key_index_1 = thread_id + kThreads;

      const float q0 = load_bf16(q, token_base + key_index_0);
      const float q1 = load_bf16(q, token_base + key_index_1);
      const float k0 = load_bf16(k, token_base + key_index_0);
      const float k1 = load_bf16(k, token_base + key_index_1);

      q_hat[key_index_0] = q0;
      q_hat[key_index_1] = q1;
      k_hat[key_index_0] = k0;
      k_hat[key_index_1] = k1;

      q_reduce[thread_id] = q0 * q0 + q1 * q1;
      k_reduce[thread_id] = k0 * k0 + k1 * k1;
    }

    if (thread_id == 0) {
      const int64_t beta_offset =
          token * num_heads + head_index;
      beta_value = kda_sigmoid(load_bf16(beta, beta_offset));
    }
    __syncthreads();

    // A single lane finishes the small 64-way norm reduction. This avoids
    // six additional full-block barriers on the 512-thread block.
    if (thread_id == 0) {
      float q_sum = 0.0f;
      float k_sum = 0.0f;
#pragma unroll
      for (int lane = 0; lane < kThreads; ++lane) {
        q_sum += q_reduce[lane];
        k_sum += k_reduce[lane];
      }
      q_inverse_norm =
          1.0f / fmaxf(sqrtf(q_sum), kNormEpsilon);
      k_inverse_norm =
          1.0f / fmaxf(sqrtf(k_sum), kNormEpsilon);
    }
    __syncthreads();

    // Normalize q/k and build the per-channel decay once per token.
    if (thread_id < kHeadDim) {
      const int key_index = thread_id;

      q_hat[key_index] *= q_inverse_norm;
      k_hat[key_index] *= k_inverse_norm;

      const float gate_input =
          gate_scale *
          (load_bf16(g, token_base + key_index) +
           dt_bias[head_index * kHeadDim + key_index]);

      float gate = lower_bound * kda_sigmoid(gate_input);
      gate = fminf(0.0f, fmaxf(lower_bound, gate));
      alpha[key_index] = expf(gate);
    }
    __syncthreads();

    // Load V after shared q/k/alpha publication so this register does not
    // span the preceding producer/consumer barriers.
    float input_value = 0.0f;
    if (key_lane == 0) {
      input_value = load_bf16(v, token_base + value_index);
    }

    // Decay the private state and form one prediction partial per thread.
    float prediction_partial = 0.0f;
#pragma unroll
    for (int item = 0; item < kV2KeysPerThread; ++item) {
      const int key_index = key_lane + item * kV2KeyLanes;
      state_local[item] *= alpha[key_index];
      prediction_partial =
          fmaf(k_hat[key_index], state_local[item],
               prediction_partial);
    }

    // Reduce each aligned width-8 K subgroup entirely in registers.
#pragma unroll
    for (int offset = kV2KeyLanes / 2;
         offset > 0;
         offset >>= 1) {
      prediction_partial += __shfl_xor_sync(
          kFullWarpMask64,
          prediction_partial,
          offset,
          kV2KeyLanes);
    }

    float residual_value = 0.0f;
    if (key_lane == 0) {
      residual_value =
          beta_value * (input_value - prediction_partial);
    }

    // All warp lanes participate; lane zero is relative to each width-8 group.
    residual_value = __shfl_sync(
        kFullWarpMask64,
        residual_value,
        0,
        kV2KeyLanes);

    // Apply the delta update and form one output partial per thread.
    float output_partial = 0.0f;

#pragma unroll
    for (int item = 0; item < kV2KeysPerThread; ++item) {
      const int key_index = key_lane + item * kV2KeyLanes;
      state_local[item] =
          fmaf(k_hat[key_index], residual_value,
               state_local[item]);
      output_partial =
          fmaf(q_hat[key_index], state_local[item],
               output_partial);
    }

    // Reduce output contributions in the same width-8 subgroup.
#pragma unroll
    for (int offset = kV2KeyLanes / 2;
         offset > 0;
         offset >>= 1) {
      output_partial += __shfl_xor_sync(
          kFullWarpMask64,
          output_partial,
          offset,
          kV2KeyLanes);
    }

    if (key_lane == 0) {
      store_bf16(
          out,
          token_base + value_index,
          scale * output_partial);
    }

    // No warp may overwrite q/k/alpha for the next token while another
    // warp is still consuming the current token's shared inputs.
    __syncthreads();
  }

#pragma unroll
  for (int item = 0; item < kV2KeysPerThread; ++item) {
    const int key_index = key_lane + item * kV2KeyLanes;
    const int64_t external_offset =
        (sequence_head * kValueDim + value_index) * kHeadDim +
        key_index;
    final_state[external_offset] = state_local[item];
  }
}

template <typename IndexType>
void launch_chunk_kda_varlen_fwd_bf16_d128_v5(
    torch::Tensor& out,
    torch::Tensor& final_state,
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& g,
    const torch::Tensor& beta,
    const torch::Tensor& A_log,
    const torch::Tensor& dt_bias,
    const torch::Tensor& cu_seqlens,
    const float* initial_state,
    int64_t sequence_heads,
    int64_t total_tokens,
    int64_t num_heads,
    float scale,
    float lower_bound,
    cudaStream_t stream) {
  const dim3 grid(
      static_cast<unsigned int>(sequence_heads),
      kValueDim / kValueTile);
  const dim3 block(kV2Threads);

  chunk_kda_varlen_fwd_bf16_d128_v5_kernel<IndexType>
      <<<grid, block, 0, stream>>>(
          reinterpret_cast<DeviceBFloat16*>(
              out.data_ptr<c10::BFloat16>()),
          final_state.data_ptr<float>(),
          reinterpret_cast<const DeviceBFloat16*>(
              q.data_ptr<c10::BFloat16>()),
          reinterpret_cast<const DeviceBFloat16*>(
              k.data_ptr<c10::BFloat16>()),
          reinterpret_cast<const DeviceBFloat16*>(
              v.data_ptr<c10::BFloat16>()),
          reinterpret_cast<const DeviceBFloat16*>(
              g.data_ptr<c10::BFloat16>()),
          reinterpret_cast<const DeviceBFloat16*>(
              beta.data_ptr<c10::BFloat16>()),
          A_log.data_ptr<float>(),
          dt_bias.data_ptr<float>(),
          initial_state,
          cu_seqlens.data_ptr<IndexType>(),
          total_tokens,
          num_heads,
          scale,
          lower_bound);
}

void check_kda_tensor(
    const torch::Tensor& tensor,
    const char* name,
    at::ScalarType expected_dtype,
    const char* expected_dtype_name) {
  TORCH_CHECK(tensor.defined(), name, " must be defined");
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA/MACA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(
      tensor.scalar_type() == expected_dtype,
      name, " must have dtype ", expected_dtype_name);
}

// Process-local environment selectors are diagnostic rollback controls.
// The default path is tile-64 + warp synchronization + V5 shuffle reduction.
int selected_chunk_kda_value_tile() {
  static const int value_tile = []() -> int {
    const char* raw =
        std::getenv("MCOPLIB_CHUNK_KDA_VALUE_TILE");

    if (raw == nullptr || std::strcmp(raw, "64") == 0) {
      return kValueTile;
    }

    TORCH_CHECK(
        std::strcmp(raw, "32") == 0,
        "MCOPLIB_CHUNK_KDA_VALUE_TILE must be 32 or 64, got ",
        raw);
    return kV3ValueTile;
  }();

  return value_tile;
}

bool selected_chunk_kda_warp_sync() {
  static const bool use_warp_sync = []() -> bool {
    const char* raw =
        std::getenv("MCOPLIB_CHUNK_KDA_SYNC_SCOPE");

    if (raw == nullptr || std::strcmp(raw, "warp") == 0) {
      return true;
    }

    TORCH_CHECK(
        std::strcmp(raw, "block") == 0,
        "MCOPLIB_CHUNK_KDA_SYNC_SCOPE must be block or warp, got ",
        raw);
    return false;
  }();

  return use_warp_sync;
}


bool selected_chunk_kda_shuffle_reduction() {
  static const bool use_shuffle = []() -> bool {
    const char* raw =
        std::getenv("MCOPLIB_CHUNK_KDA_REDUCTION");

    if (raw == nullptr || std::strcmp(raw, "shuffle") == 0) {
      return true;
    }

    TORCH_CHECK(
        std::strcmp(raw, "shared") == 0,
        "MCOPLIB_CHUNK_KDA_REDUCTION must be shared or shuffle, got ",
        raw);
    return false;
  }();

  return use_shuffle;
}

}  // namespace

void chunk_kda_fwd(
    torch::Tensor& out, torch::Tensor& final_state,
    const torch::Tensor& q, const torch::Tensor& k,
    const torch::Tensor& v, const torch::Tensor& g,
    const torch::Tensor& beta, const torch::Tensor& A_log,
    const torch::Tensor& dt_bias,
    const std::optional<torch::Tensor>& initial_state,
    double scale, double lower_bound) {
  check_kda_tensor(q, "q", at::ScalarType::BFloat16, "bfloat16");
  check_kda_tensor(k, "k", at::ScalarType::BFloat16, "bfloat16");
  check_kda_tensor(v, "v", at::ScalarType::BFloat16, "bfloat16");
  check_kda_tensor(g, "g", at::ScalarType::BFloat16, "bfloat16");
  check_kda_tensor(beta, "beta", at::ScalarType::BFloat16, "bfloat16");
  check_kda_tensor(A_log, "A_log", at::ScalarType::Float, "float32");
  check_kda_tensor(dt_bias, "dt_bias", at::ScalarType::Float, "float32");
  check_kda_tensor(out, "out", at::ScalarType::BFloat16, "bfloat16");
  check_kda_tensor(
      final_state, "final_state", at::ScalarType::Float, "float32");

  const auto device = q.device();
  auto check_same_device =
      [&](const torch::Tensor& tensor, const char* name) {
        TORCH_CHECK(
            tensor.device() == device,
            name, " must be on the same device as q");
      };

  check_same_device(k, "k");
  check_same_device(v, "v");
  check_same_device(g, "g");
  check_same_device(beta, "beta");
  check_same_device(A_log, "A_log");
  check_same_device(dt_bias, "dt_bias");
  check_same_device(out, "out");
  check_same_device(final_state, "final_state");

  TORCH_CHECK(q.dim() == 4, "q must have shape [B,T,H,128]");
  const int64_t batch_size = q.size(0);
  const int64_t sequence_length = q.size(1);
  const int64_t num_heads = q.size(2);

  TORCH_CHECK(
      batch_size > 0 && sequence_length > 0 && num_heads > 0,
      "B, T and H must all be positive");
  TORCH_CHECK(
      q.size(3) == kHeadDim,
      "q must have K=128, got ", q.size(3));
  TORCH_CHECK(
      k.sizes() == q.sizes(),
      "k must have the same [B,T,H,128] shape as q");
  TORCH_CHECK(
      v.sizes() == q.sizes(),
      "v must have shape [B,T,H,128]");
  TORCH_CHECK(
      g.sizes() == q.sizes(),
      "g must have the same [B,T,H,128] shape as q");
  TORCH_CHECK(
      out.sizes() == q.sizes(),
      "out must have shape [B,T,H,128]");

  TORCH_CHECK(
      beta.dim() == 3 &&
          beta.size(0) == batch_size &&
          beta.size(1) == sequence_length &&
          beta.size(2) == num_heads,
      "beta must have shape [B,T,H]");
  TORCH_CHECK(
      A_log.dim() == 1 && A_log.size(0) == num_heads,
      "A_log must have shape [H]");
  TORCH_CHECK(
      dt_bias.dim() == 2 &&
          dt_bias.size(0) == num_heads &&
          dt_bias.size(1) == kHeadDim,
      "dt_bias must have shape [H,128]");
  TORCH_CHECK(
      final_state.dim() == 4 &&
          final_state.size(0) == batch_size &&
          final_state.size(1) == num_heads &&
          final_state.size(2) == kValueDim &&
          final_state.size(3) == kHeadDim,
      "final_state must have shape [B,H,128,128] in [V,K] layout");

  const bool has_initial_state = initial_state.has_value();
  TORCH_CHECK(
      !has_initial_state || initial_state->defined(),
      "initial_state must be None or a defined tensor");

  const float* initial_state_pointer = nullptr;
  if (has_initial_state) {
    check_kda_tensor(
        *initial_state,
        "initial_state",
        at::ScalarType::Float,
        "float32");
    check_same_device(*initial_state, "initial_state");

    TORCH_CHECK(
        initial_state->dim() == 4 &&
            initial_state->size(0) == batch_size &&
            initial_state->size(1) == num_heads &&
            initial_state->size(2) == kValueDim &&
            initial_state->size(3) == kHeadDim,
        "initial_state must have shape [B,H,128,128] in [V,K] layout");
    TORCH_CHECK(
        final_state.data_ptr<float>() != initial_state->data_ptr<float>(),
        "final_state must not alias initial_state");

    initial_state_pointer = initial_state->data_ptr<float>();
  }

  TORCH_CHECK(
      out.data_ptr<c10::BFloat16>() != q.data_ptr<c10::BFloat16>() &&
          out.data_ptr<c10::BFloat16>() != k.data_ptr<c10::BFloat16>() &&
          out.data_ptr<c10::BFloat16>() != v.data_ptr<c10::BFloat16>() &&
          out.data_ptr<c10::BFloat16>() != g.data_ptr<c10::BFloat16>(),
      "out must not alias q, k, v or g");

  TORCH_CHECK(
      std::isfinite(scale) && scale > 0.0,
      "scale must be finite and positive");
  TORCH_CHECK(
      std::isfinite(lower_bound) &&
          lower_bound >= -5.0 &&
          lower_bound < 0.0,
      "lower_bound must satisfy -5 <= lower_bound < 0");

  const int64_t batch_heads = batch_size * num_heads;
  TORCH_CHECK(
      batch_heads <=
          static_cast<int64_t>(
              std::numeric_limits<unsigned int>::max()),
      "B*H is too large for the launch grid");

  const at::cuda::CUDAGuard device_guard{
      static_cast<c10::DeviceIndex>(q.get_device())};
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(q.get_device()).stream();

  const int selected_value_tile =
      selected_chunk_kda_value_tile();

  if (selected_value_tile == kV3ValueTile) {
    const dim3 grid(
        static_cast<unsigned int>(batch_heads),
        kValueDim / kV3ValueTile);
    const dim3 block(kV3Threads);

    chunk_kda_fwd_bf16_d128_v3_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<DeviceBFloat16*>(
            out.data_ptr<c10::BFloat16>()),
        final_state.data_ptr<float>(),
        reinterpret_cast<const DeviceBFloat16*>(
            q.data_ptr<c10::BFloat16>()),
        reinterpret_cast<const DeviceBFloat16*>(
            k.data_ptr<c10::BFloat16>()),
        reinterpret_cast<const DeviceBFloat16*>(
            v.data_ptr<c10::BFloat16>()),
        reinterpret_cast<const DeviceBFloat16*>(
            g.data_ptr<c10::BFloat16>()),
        reinterpret_cast<const DeviceBFloat16*>(
            beta.data_ptr<c10::BFloat16>()),
        A_log.data_ptr<float>(),
        dt_bias.data_ptr<float>(),
        initial_state_pointer,
        batch_size,
        sequence_length,
        num_heads,
        static_cast<float>(scale),
        static_cast<float>(lower_bound));
  } else if (selected_chunk_kda_warp_sync()) {
    const dim3 grid(
        static_cast<unsigned int>(batch_heads),
        kValueDim / kValueTile);
    const dim3 block(kV2Threads);

    if (selected_chunk_kda_shuffle_reduction()) {
      chunk_kda_fwd_bf16_d128_v5_kernel<<<grid, block, 0, stream>>>(
          reinterpret_cast<DeviceBFloat16*>(
              out.data_ptr<c10::BFloat16>()),
          final_state.data_ptr<float>(),
          reinterpret_cast<const DeviceBFloat16*>(
              q.data_ptr<c10::BFloat16>()),
          reinterpret_cast<const DeviceBFloat16*>(
              k.data_ptr<c10::BFloat16>()),
          reinterpret_cast<const DeviceBFloat16*>(
              v.data_ptr<c10::BFloat16>()),
          reinterpret_cast<const DeviceBFloat16*>(
              g.data_ptr<c10::BFloat16>()),
          reinterpret_cast<const DeviceBFloat16*>(
              beta.data_ptr<c10::BFloat16>()),
          A_log.data_ptr<float>(),
          dt_bias.data_ptr<float>(),
          initial_state_pointer,
          batch_size,
          sequence_length,
          num_heads,
          static_cast<float>(scale),
          static_cast<float>(lower_bound));
    } else {
      chunk_kda_fwd_bf16_d128_v4_kernel<<<grid, block, 0, stream>>>(
          reinterpret_cast<DeviceBFloat16*>(
              out.data_ptr<c10::BFloat16>()),
          final_state.data_ptr<float>(),
          reinterpret_cast<const DeviceBFloat16*>(
              q.data_ptr<c10::BFloat16>()),
          reinterpret_cast<const DeviceBFloat16*>(
              k.data_ptr<c10::BFloat16>()),
          reinterpret_cast<const DeviceBFloat16*>(
              v.data_ptr<c10::BFloat16>()),
          reinterpret_cast<const DeviceBFloat16*>(
              g.data_ptr<c10::BFloat16>()),
          reinterpret_cast<const DeviceBFloat16*>(
              beta.data_ptr<c10::BFloat16>()),
          A_log.data_ptr<float>(),
          dt_bias.data_ptr<float>(),
          initial_state_pointer,
          batch_size,
          sequence_length,
          num_heads,
          static_cast<float>(scale),
          static_cast<float>(lower_bound));
    }
  } else {
    const dim3 grid(
        static_cast<unsigned int>(batch_heads),
        kValueDim / kValueTile);
    const dim3 block(kV2Threads);

    chunk_kda_fwd_bf16_d128_v2_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<DeviceBFloat16*>(
            out.data_ptr<c10::BFloat16>()),
        final_state.data_ptr<float>(),
        reinterpret_cast<const DeviceBFloat16*>(
            q.data_ptr<c10::BFloat16>()),
        reinterpret_cast<const DeviceBFloat16*>(
            k.data_ptr<c10::BFloat16>()),
        reinterpret_cast<const DeviceBFloat16*>(
            v.data_ptr<c10::BFloat16>()),
        reinterpret_cast<const DeviceBFloat16*>(
            g.data_ptr<c10::BFloat16>()),
        reinterpret_cast<const DeviceBFloat16*>(
            beta.data_ptr<c10::BFloat16>()),
        A_log.data_ptr<float>(),
        dt_bias.data_ptr<float>(),
        initial_state_pointer,
        batch_size,
        sequence_length,
        num_heads,
        static_cast<float>(scale),
        static_cast<float>(lower_bound));
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Native packed-varlen BF16 KDA forward. The device cu_seqlens values
// must describe a complete, monotonic partition of T_total.
void chunk_kda_varlen_fwd(
    torch::Tensor& out, torch::Tensor& final_state,
    const torch::Tensor& q, const torch::Tensor& k,
    const torch::Tensor& v, const torch::Tensor& g,
    const torch::Tensor& beta, const torch::Tensor& A_log,
    const torch::Tensor& dt_bias,
    const torch::Tensor& cu_seqlens,
    const std::optional<torch::Tensor>& initial_state,
    double scale, double lower_bound) {
  check_kda_tensor(q, "q", at::ScalarType::BFloat16, "bfloat16");
  check_kda_tensor(k, "k", at::ScalarType::BFloat16, "bfloat16");
  check_kda_tensor(v, "v", at::ScalarType::BFloat16, "bfloat16");
  check_kda_tensor(g, "g", at::ScalarType::BFloat16, "bfloat16");
  check_kda_tensor(beta, "beta", at::ScalarType::BFloat16, "bfloat16");
  check_kda_tensor(A_log, "A_log", at::ScalarType::Float, "float32");
  check_kda_tensor(dt_bias, "dt_bias", at::ScalarType::Float, "float32");
  check_kda_tensor(out, "out", at::ScalarType::BFloat16, "bfloat16");
  check_kda_tensor(
      final_state, "final_state", at::ScalarType::Float, "float32");

  TORCH_CHECK(
      cu_seqlens.defined(),
      "cu_seqlens must be defined");
  TORCH_CHECK(
      cu_seqlens.is_cuda(),
      "cu_seqlens must be a CUDA/MACA tensor");
  TORCH_CHECK(
      cu_seqlens.is_contiguous(),
      "cu_seqlens must be contiguous");
  TORCH_CHECK(
      cu_seqlens.scalar_type() == at::ScalarType::Int ||
          cu_seqlens.scalar_type() == at::ScalarType::Long,
      "cu_seqlens must have dtype int32 or int64");

  const auto device = q.device();
  auto check_same_device =
      [&](const torch::Tensor& tensor, const char* name) {
        TORCH_CHECK(
            tensor.device() == device,
            name, " must be on the same device as q");
      };

  check_same_device(k, "k");
  check_same_device(v, "v");
  check_same_device(g, "g");
  check_same_device(beta, "beta");
  check_same_device(A_log, "A_log");
  check_same_device(dt_bias, "dt_bias");
  check_same_device(cu_seqlens, "cu_seqlens");
  check_same_device(out, "out");
  check_same_device(final_state, "final_state");

  TORCH_CHECK(
      q.dim() == 4 && q.size(0) == 1,
      "q must have packed shape [1,T_total,H,128]");
  const int64_t total_tokens = q.size(1);
  const int64_t num_heads = q.size(2);

  TORCH_CHECK(num_heads > 0, "H must be positive");
  TORCH_CHECK(
      q.size(3) == kHeadDim,
      "q must have K=128, got ", q.size(3));
  TORCH_CHECK(
      k.sizes() == q.sizes(),
      "k must have the same packed [1,T_total,H,128] shape as q");
  TORCH_CHECK(
      v.sizes() == q.sizes(),
      "v must have packed shape [1,T_total,H,128]");
  TORCH_CHECK(
      g.sizes() == q.sizes(),
      "g must have the same packed [1,T_total,H,128] shape as q");
  TORCH_CHECK(
      out.sizes() == q.sizes(),
      "out must have packed shape [1,T_total,H,128]");

  TORCH_CHECK(
      beta.dim() == 3 &&
          beta.size(0) == 1 &&
          beta.size(1) == total_tokens &&
          beta.size(2) == num_heads,
      "beta must have packed shape [1,T_total,H]");
  TORCH_CHECK(
      A_log.dim() == 1 && A_log.size(0) == num_heads,
      "A_log must have shape [H]");
  TORCH_CHECK(
      dt_bias.dim() == 2 &&
          dt_bias.size(0) == num_heads &&
          dt_bias.size(1) == kHeadDim,
      "dt_bias must have shape [H,128]");

  TORCH_CHECK(
      cu_seqlens.dim() == 1 && cu_seqlens.numel() >= 2,
      "cu_seqlens must have shape [num_sequences+1]");
  const int64_t num_sequences = cu_seqlens.numel() - 1;

  TORCH_CHECK(
      final_state.dim() == 4 &&
          final_state.size(0) == num_sequences &&
          final_state.size(1) == num_heads &&
          final_state.size(2) == kValueDim &&
          final_state.size(3) == kHeadDim,
      "final_state must have shape "
      "[num_sequences,H,128,128] in [V,K] layout");

  const bool has_initial_state = initial_state.has_value();
  TORCH_CHECK(
      !has_initial_state || initial_state->defined(),
      "initial_state must be None or a defined tensor");

  const float* initial_state_pointer = nullptr;
  if (has_initial_state) {
    check_kda_tensor(
        *initial_state,
        "initial_state",
        at::ScalarType::Float,
        "float32");
    check_same_device(*initial_state, "initial_state");

    TORCH_CHECK(
        initial_state->dim() == 4 &&
            initial_state->size(0) == num_sequences &&
            initial_state->size(1) == num_heads &&
            initial_state->size(2) == kValueDim &&
            initial_state->size(3) == kHeadDim,
        "initial_state must have shape "
        "[num_sequences,H,128,128] in [V,K] layout");
    TORCH_CHECK(
        final_state.data_ptr<float>() != initial_state->data_ptr<float>(),
        "final_state must not alias initial_state");

    initial_state_pointer = initial_state->data_ptr<float>();
  }

  TORCH_CHECK(
      final_state.data_ptr<float>() != A_log.data_ptr<float>() &&
          final_state.data_ptr<float>() != dt_bias.data_ptr<float>(),
      "final_state must not alias A_log or dt_bias");

  if (total_tokens > 0) {
    TORCH_CHECK(
        out.data_ptr<c10::BFloat16>() !=
                q.data_ptr<c10::BFloat16>() &&
            out.data_ptr<c10::BFloat16>() !=
                k.data_ptr<c10::BFloat16>() &&
            out.data_ptr<c10::BFloat16>() !=
                v.data_ptr<c10::BFloat16>() &&
            out.data_ptr<c10::BFloat16>() !=
                g.data_ptr<c10::BFloat16>() &&
            out.data_ptr<c10::BFloat16>() !=
                beta.data_ptr<c10::BFloat16>(),
        "out must not alias q, k, v, g or beta");
  }

  TORCH_CHECK(
      std::isfinite(scale) && scale > 0.0,
      "scale must be finite and positive");
  TORCH_CHECK(
      std::isfinite(lower_bound) &&
          lower_bound >= -5.0 &&
          lower_bound < 0.0,
      "lower_bound must satisfy -5 <= lower_bound < 0");

  const int64_t max_grid_x =
      static_cast<int64_t>(
          std::numeric_limits<unsigned int>::max());
  TORCH_CHECK(
      num_sequences <= max_grid_x / num_heads,
      "num_sequences*H is too large for the launch grid");
  const int64_t sequence_heads = num_sequences * num_heads;

  const at::cuda::CUDAGuard device_guard{
      static_cast<c10::DeviceIndex>(q.get_device())};
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(q.get_device()).stream();

  if (cu_seqlens.scalar_type() == at::ScalarType::Int) {
    launch_chunk_kda_varlen_fwd_bf16_d128_v5<int32_t>(
        out, final_state, q, k, v, g, beta, A_log, dt_bias,
        cu_seqlens, initial_state_pointer, sequence_heads,
        total_tokens, num_heads, static_cast<float>(scale),
        static_cast<float>(lower_bound), stream);
  } else {
    launch_chunk_kda_varlen_fwd_bf16_d128_v5<int64_t>(
        out, final_state, q, k, v, g, beta, A_log, dt_bias,
        cu_seqlens, initial_state_pointer, sequence_heads,
        total_tokens, num_heads, static_cast<float>(scale),
        static_cast<float>(lower_bound), stream);
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
