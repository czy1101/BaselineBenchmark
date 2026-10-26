#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

static constexpr int THREADS = 256;
static constexpr int WAVE_SIZE = 64;
static constexpr int MAX_WAVES = THREADS / WAVE_SIZE;
static constexpr int MAX_D = 8192;

template <typename T>
__device__ inline float to_float(T value) {
    return static_cast<float>(value);
}

template <>
__device__ inline float to_float<__half>(__half value) {
    return __half2float(value);
}

template <>
__device__ inline float to_float<hip_bfloat16>(hip_bfloat16 value) {
    return static_cast<float>(value);
}

template <typename T>
__device__ inline T from_float(float value) {
    return static_cast<T>(value);
}

template <>
__device__ inline __half from_float<__half>(float value) {
    return __float2half(value);
}

template <>
__device__ inline hip_bfloat16 from_float<hip_bfloat16>(float value) {
    return hip_bfloat16(value);
}

__device__ inline float wave_sum(float value) {
    for (int offset = 32; offset > 0; offset >>= 1) {
        value += __shfl_down(value, offset, WAVE_SIZE);
    }
    return value;
}

template <typename T, int D_CONST = 0, int NORM_CONST = 0>
__global__ void attnes_kernel(
    const T* __restrict__ query,
    const T* __restrict__ r0,
    const T* __restrict__ r1,
    const T* __restrict__ r2,
    const T* __restrict__ r3,
    const T* __restrict__ r4,
    const T* __restrict__ r5,
    const T* __restrict__ r6,
    const T* __restrict__ r7,
    const T* __restrict__ r8,
    const T* __restrict__ rms_weight,
    const T* __restrict__ output_rms_weight,
    T* __restrict__ output,
    int64_t rows,
    int D,
    int L,
    float eps,
    float scale,
    int has_output_norm
) {
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    const int tid = threadIdx.x;
    const int wave = tid / WAVE_SIZE;
    const int lane = tid % WAVE_SIZE;

    if (row >= rows || tid >= THREADS || D > MAX_D) {
        return;
    }

    if (D_CONST != 0 && D != D_CONST) {
        return;
    }

    const int kernel_D = D_CONST != 0 ? D_CONST : D;
    constexpr bool kernel_has_output_norm = NORM_CONST != 0;

    extern __shared__ float shared_memory[];

    float* wave_norm = shared_memory;
    float* wave_dot = wave_norm + MAX_WAVES;
    float* shared_score = wave_dot + MAX_WAVES;
    float* shared_running_max = shared_score + 1;
    float* shared_running_sum = shared_running_max + 1;
    float* shared_rescale = shared_running_sum + 1;
    float* shared_probability = shared_rescale + 1;
    float* shared_rstd = shared_probability + 1;

    const T* sources[9] = {
        r0, r1, r2, r3, r4, r5, r6, r7, r8
    };

    const int tiles = (kernel_D + THREADS - 1) / THREADS;
    float query_values[MAX_D / THREADS];
    float source_values[MAX_D / THREADS];
    float accum[MAX_D / THREADS];

    for (int i = 0; i < tiles; ++i) {
        const int d = tid + i * THREADS;

        if (d < kernel_D) {
            query_values[i] =
                to_float(query[d]) * to_float(rms_weight[d]);
        }

        accum[i] = 0.0f;
    }

    for (int source = 0; source < L; ++source) {
        const T* source_row = sources[source] + row * D;

        float local_norm = 0.0f;
        float local_dot = 0.0f;

        #pragma unroll 4
        for (int i = 0; i < tiles; ++i) {
            const int d = tid + i * THREADS;

            if (d < kernel_D) {
                const float value = to_float(source_row[d]);
                source_values[i] = value;
                local_norm += value * value;
                local_dot += value * query_values[i];
            }
        }

        const float wave_norm_value = wave_sum(local_norm);
        const float wave_dot_value = wave_sum(local_dot);

        if (lane == 0) {
            wave_norm[wave] = wave_norm_value;
            wave_dot[wave] = wave_dot_value;
        }

        __syncthreads();

        if (tid == 0) {
            float total_norm = 0.0f;
            float total_dot = 0.0f;

            for (int w = 0; w < MAX_WAVES; ++w) {
                total_norm += wave_norm[w];
                total_dot += wave_dot[w];
            }

            const float rstd =
                rsqrtf(total_norm / static_cast<float>(kernel_D) + eps);

            shared_score[0] = total_dot * rstd * scale;

            if (source == 0) {
                shared_running_max[0] = shared_score[0];
                shared_running_sum[0] = 1.0f;
                shared_rescale[0] = 1.0f;
                shared_probability[0] = 1.0f;
            } else {
                const float previous_max = shared_running_max[0];
                const float current_max =
                    fmaxf(previous_max, shared_score[0]);

                shared_rescale[0] =
                    __expf(previous_max - current_max);
                shared_probability[0] =
                    __expf(shared_score[0] - current_max);

                shared_running_max[0] = current_max;
                shared_running_sum[0] =
                    shared_running_sum[0] * shared_rescale[0]
                    + shared_probability[0];
            }
        }

        __syncthreads();

        const float score = shared_score[0];

        if (source == 0) {
            #pragma unroll 4
        for (int i = 0; i < tiles; ++i) {
                const int d = tid + i * THREADS;
                if (d < kernel_D) {
                    accum[i] = source_values[i];
                }
            }
        } else {
            const float rescale = shared_rescale[0];
            const float probability = shared_probability[0];

            #pragma unroll 4
        for (int i = 0; i < tiles; ++i) {
                const int d = tid + i * THREADS;
                if (d < kernel_D) {
                    accum[i] =
                        accum[i] * rescale
                        + probability * source_values[i];
                }
            }
        }

    }

    const float running_sum = shared_running_sum[0];

    for (int i = 0; i < tiles; ++i) {
        accum[i] /= running_sum;
    }

    if (kernel_has_output_norm) {
        float local_norm = 0.0f;

        #pragma unroll 4
        for (int i = 0; i < tiles; ++i) {
            const int d = tid + i * THREADS;
            if (d < kernel_D) {
                local_norm += accum[i] * accum[i];
            }
        }

        const float wave_norm_value = wave_sum(local_norm);

        if (lane == 0) {
            wave_norm[wave] = wave_norm_value;
        }

        __syncthreads();

        if (tid == 0) {
            float total_norm = 0.0f;

            for (int w = 0; w < MAX_WAVES; ++w) {
                total_norm += wave_norm[w];
            }

            shared_rstd[0] =
                rsqrtf(total_norm / static_cast<float>(kernel_D) + eps);
        }

        __syncthreads();

        const float rstd = shared_rstd[0];

        #pragma unroll 4
        for (int i = 0; i < tiles; ++i) {
            const int d = tid + i * THREADS;
            if (d < kernel_D) {
                accum[i] *=
                    rstd * to_float(output_rms_weight[d]);
            }
        }
    }

    for (int i = 0; i < tiles; ++i) {
        const int d = tid + i * THREADS;
        if (d < kernel_D) {
            output[row * D + d] =
                from_float<T>(accum[i]);
        }
    }
}

extern "C" void attnes_forward(
    const void* query,
    const void* r0,
    const void* r1,
    const void* r2,
    const void* r3,
    const void* r4,
    const void* r5,
    const void* r6,
    const void* r7,
    const void* r8,
    const void* rms_weight,
    const void* output_rms_weight,
    void* output,
    int64_t rows,
    int D,
    int L,
    float eps,
    float scale,
    int dtype,
    int has_output_norm
) {
    const int shared_bytes =
        (2 * MAX_WAVES + 6) * sizeof(float);

#define LAUNCH(TYPE, KERNEL_D, NORM_CONST)                                                        \
    hipLaunchKernelGGL(                                                     \
        (attnes_kernel<TYPE, KERNEL_D, NORM_CONST>),                                              \
        dim3(static_cast<unsigned>(rows)),                                  \
        dim3(THREADS),                                                       \
        shared_bytes,                                                        \
        0,                                                                   \
        reinterpret_cast<const TYPE*>(query),                               \
        reinterpret_cast<const TYPE*>(r0),                                  \
        reinterpret_cast<const TYPE*>(r1),                                  \
        reinterpret_cast<const TYPE*>(r2),                                  \
        reinterpret_cast<const TYPE*>(r3),                                  \
        reinterpret_cast<const TYPE*>(r4),                                  \
        reinterpret_cast<const TYPE*>(r5),                                  \
        reinterpret_cast<const TYPE*>(r6),                                  \
        reinterpret_cast<const TYPE*>(r7),                                  \
        reinterpret_cast<const TYPE*>(r8),                                  \
        reinterpret_cast<const TYPE*>(rms_weight),                          \
        reinterpret_cast<const TYPE*>(output_rms_weight),                   \
        reinterpret_cast<TYPE*>(output),                                     \
        rows, D, L, eps, scale, has_output_norm                              \
    )

    if (dtype == 0) {
        if (D == 7168 && rows <= 128) {
            if (has_output_norm) {
                LAUNCH(__half, 7168, 1);
            } else {
                LAUNCH(__half, 7168, 0);
            }
        } else {
            if (has_output_norm) {
                LAUNCH(__half, 0, 1);
            } else {
                LAUNCH(__half, 0, 0);
            }
        }
    } else {
        if (D == 7168 && rows <= 128) {
            if (has_output_norm) {
                LAUNCH(hip_bfloat16, 7168, 1);
            } else {
                LAUNCH(hip_bfloat16, 7168, 0);
            }
        } else {
            if (has_output_norm) {
                LAUNCH(hip_bfloat16, 0, 1);
            } else {
                LAUNCH(hip_bfloat16, 0, 0);
            }
        }
    }

#undef LAUNCH

    return;
}
