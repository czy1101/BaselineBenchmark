# Functionality

Note : PPU ACTLIZE requires users to use PPU1.0/PPU1.5.
Please refer to the [Prerequisites](/README.md###Prerequisites) section for more details.

- N - Column major matrix
- T - Row major matrix
- {N,T} x {N,T} - All combinations, i.e., NN, NT, TN, TT
- f - floating point
- s - signed int
- b - bit
- bf16 - bfloat16
- tf32 - tfloat32
- Simt - Use Simt vector-core MMA
- TensorOp - Use Tensor Cell MMA

## Device-level GEMM

The following tables summarize device-level GEMM kernels in ACTLIZE, organized by opcode class, data type, and layout

### ACTLIZE 1.0.0 Kernels

|**Opcode Class** | **Compute Capability** | **Toolchain** | **Data Type**                  | **Layouts**            | **Unit Test**    |
|-----------------|------------------------|------------------|--------------------------------|------------------------|------------------|
| **TensorOp**        | PPU0010                 |  PPU SDK           | `f16 * f16 + { f16, f32 } => { f16, f32 }`       | {N,T} x {N,T} => {N,T} |  [example](/test/unit/gemm_ppu/device/ppu0010_gemm_f16_f16_f32_tensor_op_f32_aiu.cu) |
| **TensorOp**        | PPU0010                 |  PPU SDK           | `tf32 * tf32 + f32 => f32`| { N,T } x { N,T } => {N} |  [example](/test/unit/gemm_ppu/device/ppu0010_gemm_tf32_tf32_f32_tensor_op_f32_aiu.cu) |
| **TensorOp**        | PPU0010                 |  PPU SDK           | `s8 * s8 + s32 => s32`   | { T } x { N } => {N} |  [example](/test/unit/gemm_ppu/device/ppu0010_gemm_s8_s8_s32_tensor_op_aiu.cu) |


### ACTLIZE 0.x Kernels

|**Opcode Class** | **Compute Capability** | **Toolchain** | **Data Type**                  | **Layouts**            | **Unit Test**    |
|-----------------|------------------------|------------------|--------------------------------|------------------------|------------------|
| **Simt**        | PPU 1.0/1.5                   |  PPU SDK            | `f32 * f32 + f32 => f32`       | {N,T} x {N,T} => {N,T} |  [example](/examples/00_basic_gemm/basic_gemm.cu)                |
| **Simt**        | PPU 1.0/1.5                    |  PPU SDK            | `f64 * f64 + f64 => f64`       | {N,T} x {N,T} => {N,T} |                  |
| **Simt**        | PPU 1.0/1.5                    |  PPU SDK            | `f16 * f16 + f16 => f16`       | {N,T} x {N,T} => {N,T} |                  |
| **Simt**        | PPU 1.0/1.5                    |  PPU SDK            | `s8 * s8 + s32 => {s32,s8}`    | {N,T} x {N,T} => {N,T} |                |



## Device-level Implicit GEMM convolution

# PPU ACTLIZE Functionality

The following tables summarize device-level GEMM and convolution kernels in PPU ACTLIZE.

## Supported PPU Architectures

| Architecture | Tag | AIU Support |
|--------------|-----|---------|-------------|
| PPU 1.0 | `cutlass::arch::PPU0010` | Yes |
| PPU 1.5 | `cutlass::arch::PPU0015` | Yes |

## Device-level GEMM (ACTLIZE 1.0.0)

|**Opcode Class** | **Architecture** | **Data Type** | **Layouts** | **Status** |
|-----------------|------------------|---------------|-------------|------------|
| **TensorOp** | PPU 1.0/1.5 | `f16 * f16 + { f16, f32 } => { f16, f32 }` | {N,T} x {N,T} => {N,T} | ✅ Full |
| **TensorOp** | PPU 1.0/1.5 | `bf16 * bf16 + { bf16, f32 } => { bf16, f32 }` | {N,T} x {N,T} => {N,T} | ✅ Full |
| **TensorOp** | PPU 1.0/1.5 | `tf32 * tf32 + f32 => f32` | { N,T } x { N,T } => {N,T} | ✅ Full |
| **TensorOp** | PPU 1.0/1.5 | `s8 * s8 + s32 => s32` | {N,T} x {N,T} => {N,T} | ✅ Full |
| **TensorOp** | PPU 1.5 | `f8 * f8 + f32 => f32` | {T} x {N} => {N,T} | ✅ Full |
| **TensorOp** | PPU 1.5 | `f4 * f4 + f32 => f32` | {T} x {N} => {N,T} | ✅ Full |


## Grouped GEMM

|**Type** | **Architecture** | **Status** |
|---------|------------------|------------|
| **Grouped GEMM** | PPU 1.0 | ✅ Full |
| **Grouped GEMM** | PPU 1.5 | ✅ Full |

## Batched GEMM

|**Type** | **Architecture** | **Status** |
|---------|------------------|------------|
| **Batched Strided** | PPU 1.0 | ✅ Full |
| **Batched Strided** | PPU 1.5 | ✅ Full |
| **Pointer Array** | PPU 1.0 | ✅ Full |
| **Pointer Array** | PPU 1.5 | ✅ Full |

## Implicit GEMM Convolution
<!-- currently supports simt -->

|**Operation** | **Architecture** | **Data Type** | **Ops** | **Status** |
|--------------|------------------|---------------|---------|------------|
| **Conv2d Fprop** | PPU 1.0/1.5 | f32 | SIMT | ✅ Full |
| **Conv2d Dgrad** | PPU 1.0/1.5 | f32 | SIMT | ✅ Full |
| **Conv2d Wgrad** | PPU 1.0/1.5 | f32 | SIMT | ✅ Full |
| **Conv3d Fprop** | PPU 1.0/1.5 | f32 | SIMT | ✅ Full |

---

Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD. All rights reserved.
Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause

```
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
