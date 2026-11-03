# Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD. All rights reserved. 
# Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cutlass_example_add_executable(
  100_ppu0015_fp8_gemm
  100_ppu0015_fp8_gemm.cu
  )

cutlass_example_add_executable(
  100_ppu0015_fp8_gemm_amax
  100_ppu0015_fp8_gemm_amax.cu
  )

cutlass_example_add_executable(
  100_ppu0015_fp8_gemm_bias_aux_gelu
  100_ppu0015_fp8_gemm_bias_aux_gelu.cu
  )

  cutlass_example_add_executable(
  100_ppu0015_fp8_gemm_aux_relu
  100_ppu0015_fp8_gemm_aux_relu.cu
  )

  cutlass_example_add_executable(
    100_ppu0015_fp8_gemm_dgelu_dbias
    100_ppu0015_fp8_gemm_dgelu_dbias.cu
  )

  cutlass_example_add_executable(
    100_ppu0015_fp8_gemm_RCR
    100_ppu0015_fp8_gemm_RCR.cu
  )

target_compile_options(100_ppu0015_fp8_gemm PRIVATE -w)
target_compile_options(100_ppu0015_fp8_gemm_amax PRIVATE -w)
target_compile_options(100_ppu0015_fp8_gemm_bias_aux_gelu PRIVATE -w)
target_compile_options(100_ppu0015_fp8_gemm_aux_relu PRIVATE -w)
target_compile_options(100_ppu0015_fp8_gemm_dgelu_dbias PRIVATE -w)
target_compile_options(100_ppu0015_fp8_gemm_RCR PRIVATE -w)
