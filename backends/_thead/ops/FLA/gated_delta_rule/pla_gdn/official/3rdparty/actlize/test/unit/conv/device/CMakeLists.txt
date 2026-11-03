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

list(SORT CUTLASS_PPU_ARCHS_ENABLED)
set(CUTLASS_PPU_ARCHS_ENABLED_REVERSED ${CUTLASS_PPU_ARCHS_ENABLED})
list(REVERSE CUTLASS_PPU_ARCHS_ENABLED_REVERSED)
list(GET CUTLASS_PPU_ARCHS_ENABLED_REVERSED 0 CUTLASS_PPU_MAX_ARCH)

add_custom_target(
  cutlass_test_unit_conv_device
  DEPENDS
  cutlass_test_unit_conv_device_simt
)

 add_custom_target(
  test_unit_conv_device
  DEPENDS
  test_unit_conv_device_simt
)

#
# OpClassSimt (Vector cores)
#

cutlass_test_unit_add_executable(
  cutlass_test_unit_conv_device_simt
)


cutlass_target_sources(
  cutlass_test_unit_conv_device_simt
  PRIVATE
  conv2d_fprop_implicit_gemm_f32nhwc_f32nhwc_f32nhwc_simt_f32.cu
  conv2d_dgrad_implicit_gemm_f32nhwc_f32nhwc_f32nhwc_simt_f32.cu
  conv2d_wgrad_implicit_gemm_f32nhwc_f32nhwc_f32nhwc_simt_f32.cu
  conv2d_fprop_implicit_gemm_cf32nhwc_cf32nhwc_cf32nhwc_simt_f32.cu
  conv2d_dgrad_implicit_gemm_cf32nhwc_cf32nhwc_cf32nhwc_simt_f32.cu
  conv2d_wgrad_implicit_gemm_cf32nhwc_cf32nhwc_cf32nhwc_simt_f32.cu
  deconv2d_implicit_gemm_f32nhwc_f32nhwc_f32nhwc_simt_f32.cu

  conv2d_fprop_with_broadcast_simt.cu
  deconv2d_with_broadcast_simt.cu

  conv3d_fprop_implicit_gemm_f32ndhwc_f32ndhwc_f32ndhwc_simt_f32.cu
  conv3d_dgrad_implicit_gemm_f32ndhwc_f32ndhwc_f32ndhwc_simt_f32.cu
  conv3d_wgrad_implicit_gemm_f32ndhwc_f32ndhwc_f32ndhwc_simt_f32.cu
  deconv3d_implicit_gemm_f32ndhwc_f32ndhwc_f32ndhwc_simt_f32.cu

  conv3d_fprop_with_broadcast_simt.cu
  deconv3d_with_broadcast_simt.cu

)


#
# OpClassTensorOp (Tensor cells)
#

# Conv - TensorOp to do

