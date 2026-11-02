/***************************************************************************************************
 * Copyright (c) 2026 Hygon Information Technology Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include "hytlass/hytlass.h"

// Native MMAC register ABI. The gfx938 instructions retain the gfx928/gfx936
// lane layout when LIT, LTS and integer saturation are all disabled.
#if defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)
namespace hytlass::arch::detail {

typedef float MmacFloat2 __attribute__((ext_vector_type(2)));
typedef float MmacFloat4 __attribute__((ext_vector_type(4)));
typedef __fp16 MmacHalf4 __attribute__((ext_vector_type(4)));
typedef short MmacBfloat4 __attribute__((ext_vector_type(4)));
typedef int MmacInt2 __attribute__((ext_vector_type(2)));
typedef int MmacInt4 __attribute__((ext_vector_type(4)));

HYTLASS_DEVICE
MmacFloat4 mmac_f32_16x16x8_f32(MmacFloat2 a, MmacFloat2 b, MmacFloat4 c) {
#if defined(__gfx938__)
  return __builtin_hcu_mmac_16x16x8_f32_lit_lts(a, b, c, false, false);
#else
  return __builtin_hcu_mmac_f32_16x16x8f32(a, b, c);
#endif
}

HYTLASS_DEVICE
MmacFloat4 mmac_f32_16x16x8_tf32(MmacFloat2 a, MmacFloat2 b, MmacFloat4 c) {
#if defined(__gfx938__)
  return __builtin_hcu_mmac_f32_16x16x8_tf32_lit_lts(a, b, c, false, false);
#else
  return __builtin_hcu_mmac_f32_16x16x8tf32(a, b, c);
#endif
}

HYTLASS_DEVICE
MmacFloat4 mmac_f32_16x16x16_f16(MmacHalf4 a, MmacHalf4 b, MmacFloat4 c) {
#if defined(__gfx938__)
  return __builtin_hcu_mmac_f32_16x16x16_f16_lit_lts(a, b, c, false, false);
#else
  return __builtin_hcu_mmac_f32_16x16x16f16(a, b, c);
#endif
}

HYTLASS_DEVICE
MmacFloat4 mmac_f32_16x16x16_bf16(MmacBfloat4 a, MmacBfloat4 b, MmacFloat4 c) {
#if defined(__gfx938__)
  return __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(a, b, c, false, false);
#else
  return __builtin_hcu_mmac_f32_16x16x16bf16(a, b, c);
#endif
}

HYTLASS_DEVICE
MmacInt4 mmac_i32_16x16x32_i8(long a, long b, MmacInt4 c) {
#if defined(__gfx938__)
  return __builtin_hcu_mmac_i32_16x16x32_i8_lit_clamp_lts(
      __builtin_bit_cast(MmacInt2, a), __builtin_bit_cast(MmacInt2, b),
      c, false, false, false);
#else
  return __builtin_hcu_mmac_i32_16x16x32i8(a, b, c);
#endif
}

HYTLASS_DEVICE
MmacInt4 mmac_i32_16x16x32_u8(long a, long b, MmacInt4 c) {
#if defined(__gfx938__)
  return __builtin_hcu_mmac_i32_16x16x32_u8_lit_clamp_lts(
      __builtin_bit_cast(MmacInt2, a), __builtin_bit_cast(MmacInt2, b),
      c, false, false, false);
#else
  return __builtin_hcu_mmac_i32_16x16x32u8(a, b, c);
#endif
}

} // namespace hytlass::arch::detail
#endif
