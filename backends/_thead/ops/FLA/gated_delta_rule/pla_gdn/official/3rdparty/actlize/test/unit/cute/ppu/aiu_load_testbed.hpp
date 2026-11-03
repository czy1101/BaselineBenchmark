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

#pragma once
#include "cutlass_unit_test.h"

#include <iostream>
#include <vector>

// NOTE: this header used to depend on thrust::host_vector / thrust::device_vector
// to stage the test buffer between host and device. The PPU SDK ships an
// internal thrust port whose header chain is incomplete (it references several
// hggcError* enum values that are not declared in the installed driver type
// header). Use std::vector for the host staging buffer and the PPU-aware RAII
// helper cutlass::DeviceAllocation<> for the device buffer; the latter wraps
// hggcMalloc / hggcFree / hggcMemcpy directly.
#include "cutlass/util/device_memory.h"

#include <cute/tensor.hpp>

#include "ppu_include.hpp"

namespace cutlass::test {

template <
  typename Element,
  bool Trans,
  typename Block_MN,
  typename Block_K,
  bool Swap = true
> struct DefaultGemm_AIU_Operand;

template <
  typename Element,
  typename Block_MN,
  typename Block_K,
  bool Swap
> struct DefaultGemm_AIU_Operand<
  Element,
  false,
  Block_MN,
  Block_K,
  Swap
> {
  static constexpr int BlockContSize = Block_K{} * sizeof(Element);
  static_assert(BlockContSize % 32 == 0, "aiu_no_trans: block contiguous size should be multiple 32B");
  static constexpr int AiuContByteSize = BlockContSize % 128 == 0 ? 128 : (BlockContSize % 64 == 0 ? 64 : 32);
  using AiuContElemSize = Int<AiuContByteSize / sizeof(Element)>;
  static constexpr int InstNum = Block_K{} / AiuContElemSize{};

  static constexpr int bits_per_aiu = Block_MN{} * AiuContByteSize * 8;
  using CopyInst = PPU0010_AIU_LOAD<cute::C<bits_per_aiu>, Element, false>;

  using GmemTiledCopy = decltype(
    make_tiled_copy(Copy_Atom<CopyInst, Element>{},
                    Layout<Shape <_1,_1>,
                           Stride<_1,_1>>{},
                    Layout<Shape <Block_MN, AiuContElemSize>>{}));

  using SmemCopyOp = PPU0010_TSM_LD_SWZL<Element, Block_MN{}, AiuContElemSize{}, Swap, false, InstNum>;
  using SmemCopyAtom = Copy_Atom<SmemCopyOp, Element>;
  using SmemLayoutAtom = Layout<Shape<_8, AiuContElemSize>, Stride<AiuContElemSize, _1>>;
};

template <
  typename Element,
  typename Block_MN,
  typename Block_K,
  bool Swap
> struct DefaultGemm_AIU_Operand<
  Element,
  true,
  Block_MN,
  Block_K,
  Swap
> {

  static constexpr int BlockContSize = Block_MN{} * sizeof(Element);
  static_assert(BlockContSize % 64 == 0, "aiu_trans: block contiguous size should be multiple of 64B");
  static constexpr int AiuContByteSize = BlockContSize % 128 == 0 ? 128 : 64;
  using AiuContElemSize = Int<AiuContByteSize / sizeof(Element)>;
  static constexpr int InstNum = Block_MN{} / AiuContElemSize{};

  static constexpr int bits_per_aiu = AiuContByteSize * 8 * Block_K{};
  using CopyInst = PPU0010_AIU_LOAD<cute::C<bits_per_aiu>, Element, true>;

  using GmemTiledCopy = decltype(
    make_tiled_copy(Copy_Atom<CopyInst, Element>{},
                    Layout<Shape <_1,_1>,
                           Stride<_1,_1>>{},
                    Layout<Shape <AiuContElemSize,Block_K>>{}));

  using SmemCopyOp = PPU0010_TSM_LD_SWZL<Element, Block_K{}, AiuContElemSize{}, Swap, true, InstNum>;
  using SmemCopyAtom = Copy_Atom<SmemCopyOp, Element>;
  using SmemLayoutAtom = Layout<Shape<AiuContElemSize, Block_K>, Stride<_1, AiuContElemSize>>;
};

template <class ElementType, class SmemLayout>
struct SharedStorage
{
  cute::ArrayEngine<ElementType, cute::cosize_v<SmemLayout>> smem;
};


template <class T, class Operand, class CTA_Tiler, class GmemLayout, class SmemLayout>
__global__ void
aiu_test_device_cute(T const* g_in, T* g_out,
                     typename Operand::GmemTiledCopy const gmem_tiled_copy, CTA_Tiler cta_tiler,
                     GmemLayout gmem_layout, SmemLayout smem_layout)
{
  using namespace cute;
  CUTE_STATIC_ASSERT_V(product_each(shape(cta_tiler)) == product_each(shape(smem_layout)));

  // Use Shared Storage structure to allocate and distribute aligned SMEM addresses
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorage<T, SmemLayout>;
  SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(shared_memory);

  auto smem_layout_swzl = composition(Swizzle<3,3,3>{}, smem_layout);

  // Construct SMEM tensor
  Tensor sA = make_tensor(make_smem_ptr(shared_storage.smem.begin()), smem_layout);  // (CTA_TILE_M,CTA_TILE_N,...)
  Tensor sA_swzl = make_tensor(make_smem_ptr(shared_storage.smem.begin()), smem_layout_swzl);

  Tensor gA = make_tensor(make_gmem_ptr<T>(g_in), gmem_layout);
  Tensor gB = make_tensor(make_gmem_ptr<T>(g_out), gmem_layout);

  auto gmem_thr_copy = gmem_tiled_copy.get_slice(threadIdx.x);
  Tensor tAgA = gmem_thr_copy.partition_S(make_mix_tensor_like(gA));     // (TMA,TMA_M,TMA_N,REST_M,REST_N)
  Tensor tAsA = gmem_thr_copy.partition_D(sA);                           // (TMA,TMA_M,TMA_N)

#if 1
  if (thread0()) {
    print(gmem_thr_copy);
    print("TILE  :  "); print(cta_tiler); print("\n");
    print("  gA  :  "); print(  gA);   print("\n");
    print("  gB  :  "); print(  gB);   print("\n");
    print("  sA  :  "); print(  sA);   print("\n");
    print("sAswzl:  "); print(sA_swzl);print("\n");
    print("tAgA  :  "); print(tAgA);   print("\n");
    print("tAsA  :  "); print(tAsA);   print("\n");
  }
#endif

  copy(gmem_tiled_copy, tAgA, tAsA);
  cp_async_fence();
  cp_async_wait<0>();
  __syncthreads();

  //
  // Write out trivially smem -> gmem
  //

  if (thread0()) {
    printf("gA:\n"); print_tensor(gA);
    printf("sA:\n"); print_tensor(sA);
  }

  // for (int i = threadIdx.x; i < size(sA); i += blockDim.x) {
  //   tBgB(i,stage) = sA(i);
  // }

  // Subbyte elements could cause race conditions, so be even more conservative
  if (thread0()) {
    copy(sA_swzl, gB);
  }

  __syncthreads();

  if (thread0()) {
    printf("gB:\n"); print_tensor(gB);
  }

}

template <class T, class GMEM_Layout, class SMEM_Layout, class CTA_Tile>
auto
test_aiu_load(GMEM_Layout const& gmem_layout,
              SMEM_Layout const& smem_layout,
              CTA_Tile    const& cta_tile)
{
  using namespace cute;

  static_assert(rank(GMEM_Layout{}) == 2);
  int shape0 = get<0>(shape(gmem_layout));
  int shape1 = get<1>(shape(gmem_layout));
  int stride0 = get<0>(stride(gmem_layout));
  int stride1 = get<1>(stride(gmem_layout));

  constexpr bool Trans = false; //get<1>(stride(gmem_layout)) != 1;

  constexpr int SmemH = get<0>(CTA_Tile{});
  constexpr int SmemW = get<1>(CTA_Tile{});
  using Operand = DefaultGemm_AIU_Operand<T, Trans, Int<SmemH>, Int<SmemW>>;


  constexpr int CUB_H = get<0>(shape(SMEM_Layout{}));
  constexpr int CUB_W = get<1>(shape(SMEM_Layout{}));

  // Allocate and initialize host test data
  size_t N = ceil_div(cosize(gmem_layout) * sizeof_bits<T>::value, 8);
  std::vector<char> h_in(N);
  Tensor hA_in  = make_tensor(recast_ptr<T>(h_in.data()), gmem_layout);
  Tensor hA_init  = make_tensor(recast_ptr<T>(h_in.data()), shape(gmem_layout), cute::GenColMajor{});
  for (int i = 0; i < size(hA_init); ++i) { hA_init(i) = static_cast<T>(i); }

  // Allocate device input buffer and stage host data into it.
  cutlass::DeviceAllocation<char> d_in(N);
  d_in.copy_from_host(h_in.data(), N);

  // Allocate device output buffer initialized to char(-1) (matches the
  // original thrust::device_vector<char>(N, char(-1)) construction).
  std::vector<char> h_out_init(N, char(-1));
  cutlass::DeviceAllocation<char> d_out(N);
  d_out.copy_from_host(h_out_init.data(), N);

  // Create TMA for this device Tensor
  Tensor gA = make_tensor(make_gmem_ptr<T>(d_in.get()), gmem_layout);
  typename Operand::GmemTiledCopy gmem_tiled_copy;
  gmem_tiled_copy.desc_ = {nullptr, shape0, stride0, CUB_H, CUB_W};

  // Launch
  int smem_size = int(sizeof(SharedStorage<T, decltype(smem_layout)>));
  aiu_test_device_cute<T, Operand><<<1, 32, smem_size>>>(
    reinterpret_cast<T const*>(d_in.get()),
    reinterpret_cast<T*>      (d_out.get()),
    gmem_tiled_copy, cta_tile,
    gmem_layout,
    smem_layout);

  // Copy results back to host
  std::vector<char> h_out(N);
  d_out.copy_to_host(h_out.data(), N);
  Tensor hA_out = make_tensor(recast_ptr<T>(h_out.data()), gmem_layout);

  // Validate the results. Print only the first 3 errors.
  int count = 3;
  for (int i = 0; i < size(hA_out) && count > 0; ++i) {
    EXPECT_EQ(hA_in(i), hA_out(i));
    if (hA_in(i) != hA_out(i)) {
      --count;
    }
  }

  return gmem_tiled_copy;
}

} // end namespace cutlass::test
