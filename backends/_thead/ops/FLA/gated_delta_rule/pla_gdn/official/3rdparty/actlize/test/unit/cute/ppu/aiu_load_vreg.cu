/***************************************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD. All rights reserved. 
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "cutlass_unit_test.h"

#include "../ppu/aiu_load_vreg_testbed.hpp"

using namespace cute;
using namespace cutlass::test;


template <class T, class GMEM_Layout, class SMEM_Layout, class CTA_Tile>
auto
test_aiu_load_swizzle(GMEM_Layout const& gmem_layout,
                      SMEM_Layout const& smem_layout,
                      CTA_Tile    const& cta_tile)
{
  using namespace cute;
  return test_aiu_load_vreg<T>(gmem_layout, smem_layout, cta_tile);
}

template <class T, class GMEM_Layout, class SMEM_Layout>
auto
test_aiu_load_swizzle(GMEM_Layout const& gmem_layout,
                      SMEM_Layout const& smem_layout)
{
  using namespace cute;
  return test_aiu_load_vreg<T>(gmem_layout, smem_layout, product_each(shape(smem_layout)));
}



TEST(PPU0010_CuTe_PPU, Aiu_Load_Swizzle_Vreg_FP16_Trans0)
{
  /*----------------- AiuLoad with one-cub -----------------*/
  {
    Layout gmem_layout = Layout<Shape<_32,_16>, Stride<_16,_1>>{};
    Layout smem_layout = gmem_layout;
    test_aiu_load_swizzle<half_t>(gmem_layout, smem_layout);
  }

  {
    Layout gmem_layout = Layout<Shape<_64,_16>, Stride<_16,_1>>{};
    Layout smem_layout = gmem_layout;
    test_aiu_load_swizzle<half_t>(gmem_layout, smem_layout);
  }

  {
    Layout gmem_layout = Layout<Shape<_32,_32>, Stride<_32,_1>>{};
    Layout smem_layout = gmem_layout;
    test_aiu_load_swizzle<half_t>(gmem_layout, smem_layout);
  }

  {
    Layout gmem_layout = Layout<Shape<_64,_32>, Stride<_32,_1>>{};
    Layout smem_layout = gmem_layout;
    test_aiu_load_swizzle<half_t>(gmem_layout, smem_layout);
  }

  {
    Layout gmem_layout = Layout<Shape<_32,_64>, Stride<_64,_1>>{};
    Layout smem_layout = gmem_layout;
    test_aiu_load_swizzle<half_t>(gmem_layout, smem_layout);
  }

  {
    Layout gmem_layout = Layout<Shape<_64,_64>, Stride<_64,_1>>{};
    Layout smem_layout = gmem_layout;
    test_aiu_load_swizzle<half_t>(gmem_layout, smem_layout);
  }

  /*----------------- AiuLoad with multi-cub -----------------*/
  {
    Layout gmem_layout = Layout<Shape<_32,_96>, Stride<_96,_1>>{};
    Layout smem_layout = Layout<Shape<_32,  Shape <_32, _3>>,
                                Stride<_32, Stride<_1, _1024>>>{};
    test_aiu_load_swizzle<half_t>(gmem_layout, smem_layout);
  }

  {
    Layout gmem_layout = Layout<Shape<_64,_96>, Stride<_96,_1>>{};
    Layout smem_layout = Layout<Shape<_64,  Shape <_32, _3>>,
                                Stride<_32, Stride<_1, _2048>>>{};
    test_aiu_load_swizzle<half_t>(gmem_layout, smem_layout);
  }

  {
    Layout gmem_layout = Layout<Shape<_32,_128>, Stride<_128,_1>>{};
    Layout smem_layout = Layout<Shape<_32,  Shape <_64, _2>>,
                                Stride<_64, Stride<_1, _2048>>>{};
    test_aiu_load_swizzle<half_t>(gmem_layout, smem_layout);
  }

  {
    Layout gmem_layout = Layout<Shape<_64,_128>, Stride<_128,_1>>{};
    Layout smem_layout = Layout<Shape<_64,  Shape <_64, _2>>,
                                Stride<_64, Stride<_1, _4096>>>{};
    test_aiu_load_swizzle<half_t>(gmem_layout, smem_layout);
  }

  /*----------------- AiuLoad with multi-stage -----------------*/
  {
    Layout gmem_layout = Layout<Shape<_32,_128>, Stride<_128,_1>>{};
    Layout smem_layout = Layout<Shape <_32, _64, _2>,
                                Stride<_64, _1,  _2048>>{};
    auto cta_tile = Shape<_32,_64>{};
    test_aiu_load_swizzle<half_t>(gmem_layout, smem_layout, cta_tile);
  }

  {
    Layout gmem_layout = Layout<Shape<_32,_192>, Stride<_192,_1>>{};
    Layout smem_layout = Layout<Shape <_32, _64, _3>,
                                Stride<_64, _1,  _2048>>{};
    auto cta_tile = Shape<_32,_64>{};
    test_aiu_load_swizzle<half_t>(gmem_layout, smem_layout, cta_tile);
  }

  /*---------- AiuLoad with multi-cub and multi-stage ----------*/
  {
    Layout gmem_layout = Layout<Shape<_32,_256>, Stride<_256,_1>>{};
    Layout smem_layout = Layout<Shape<_32,  Shape <_64, _2>, _2>,
                                Stride<_64, Stride<_1, _2048>, _4096>>{};
    auto cta_tile = Shape<_32,_128>{};
    test_aiu_load_swizzle<half_t>(gmem_layout, smem_layout, cta_tile);
  }

  {
    Layout gmem_layout = Layout<Shape<_32,_512>, Stride<_512,_1>>{};
    Layout smem_layout = Layout<Shape<_32,  Shape <_64, _2>, _4>,
                                Stride<_64, Stride<_1, _2048>, _4096>>{};
    auto cta_tile = Shape<_32,_128>{};
    test_aiu_load_swizzle<half_t>(gmem_layout, smem_layout, cta_tile);
  }
}

TEST(PPU0010_CuTe_PPU, Aiu_Load_Swizzle_Vreg_FP32_Trans0)
{
  /*----------------- AiuLoad with one-cub -----------------*/
  {
    Layout gmem_layout = Layout<Shape<_32,_16>, Stride<_16,_1>>{};
    Layout smem_layout = gmem_layout;
    test_aiu_load_swizzle<float>(gmem_layout, smem_layout);
  }

  {
    Layout gmem_layout = Layout<Shape<_64,_16>, Stride<_16,_1>>{};
    Layout smem_layout = gmem_layout;
    test_aiu_load_swizzle<float>(gmem_layout, smem_layout);
  }

  {
    Layout gmem_layout = Layout<Shape<_32,_32>, Stride<_32,_1>>{};
    Layout smem_layout = gmem_layout;
    test_aiu_load_swizzle<float>(gmem_layout, smem_layout);
  }

  {
    Layout gmem_layout = Layout<Shape<_64,_32>, Stride<_32,_1>>{};
    Layout smem_layout = gmem_layout;
    test_aiu_load_swizzle<float>(gmem_layout, smem_layout);
  }
}

TEST(PPU0010_CuTe_PPU, Aiu_Load_Swizzle_Vreg_INT8_Trans0)
{
  /*----------------- AiuLoad with one-cub -----------------*/
  {
    Layout gmem_layout = Layout<Shape<_32,_32>, Stride<_32,_1>>{};
    Layout smem_layout = gmem_layout;
    test_aiu_load_swizzle<int8_t>(gmem_layout, smem_layout);
  }

  {
    Layout gmem_layout = Layout<Shape<_64,_32>, Stride<_32,_1>>{};
    Layout smem_layout = gmem_layout;
    test_aiu_load_swizzle<int8_t>(gmem_layout, smem_layout);
  }

  {
    Layout gmem_layout = Layout<Shape<_32,_64>, Stride<_64,_1>>{};
    Layout smem_layout = gmem_layout;
    test_aiu_load_swizzle<int8_t>(gmem_layout, smem_layout);
  }

  {
    Layout gmem_layout = Layout<Shape<_64,_64>, Stride<_64,_1>>{};
    Layout smem_layout = gmem_layout;
    test_aiu_load_swizzle<int8_t>(gmem_layout, smem_layout);
  }

  {
    Layout gmem_layout = Layout<Shape<_32,_128>, Stride<_128,_1>>{};
    Layout smem_layout = gmem_layout;
    test_aiu_load_swizzle<int8_t>(gmem_layout, smem_layout);
  }

  {
    Layout gmem_layout = Layout<Shape<_64,_128>, Stride<_128,_1>>{};
    Layout smem_layout = gmem_layout;
    test_aiu_load_swizzle<int8_t>(gmem_layout, smem_layout);
  }
}

