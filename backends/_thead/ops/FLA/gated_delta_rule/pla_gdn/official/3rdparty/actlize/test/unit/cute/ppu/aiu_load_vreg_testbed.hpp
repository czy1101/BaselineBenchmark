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

using namespace cute;

template < typename Element>
auto get_ppu_mma_atom () {}

template <>
auto get_ppu_mma_atom<half_t> () {
  return MMA_Atom<PPU0010_16x16x16_F32F16F16F32_TN>{};
}

template <>
auto get_ppu_mma_atom<float> () {
  return MMA_Atom<PPU0010_16x16x8_F32TF32TF32F32_TN>{};
}

template <>
auto get_ppu_mma_atom<int8_t> () {
  return MMA_Atom<PPU0010_16x16x32_S32S8S8S32_TN>{};
}

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

  static constexpr int CUB_H = Block_MN{};
  static constexpr int CUB_W = AiuContElemSize{};

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

  static constexpr int CUB_H = Block_MN{};
  static constexpr int CUB_W = AiuContElemSize{};

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

template <int SmemRank, class TensorS, class CTA_Tiler>
__device__ auto get_gA(TensorS mA, CTA_Tiler cta_tiler,
                       typename std::enable_if<(SmemRank > 2), void *>::type dummy = nullptr)
{
  return local_tile(mA, cta_tiler, make_coord(0, _));
}

template <int SmemRank, class TensorS, class CTA_Tiler>
__device__ auto get_gA(TensorS mA, CTA_Tiler cta_tiler,
                       typename std::enable_if<(SmemRank <= 2), void *>::type dummy = nullptr)
{
  return mA;
}

template <class T, class Operand, class TiledMma, class CTA_Tiler,
          class GmemLayout, class SmemLayout>
__global__ void
aiu_vreg_test_device_cute(T const* g_in, T* g_out,
                          typename Operand::GmemTiledCopy const gmem_tiled_copy, CTA_Tiler cta_tiler,
                          GmemLayout gmem_layout, SmemLayout smem_layout)
{
  using namespace cute;
  CUTE_STATIC_ASSERT_V(product_each(shape(cta_tiler)) == product_each(take<0,2>(shape(smem_layout))));

  // Use Shared Storage structure to allocate and distribute aligned SMEM addresses
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorage<T, SmemLayout>;
  SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(shared_memory);

  // Construct SMEM tensor
  Tensor sA = make_tensor(make_smem_ptr(shared_storage.smem.begin()), smem_layout);  // (CTA_TILE_M,CTA_TILE_N,...)

  // Construct GMEM tensor
  Tensor mA = make_mix_tensor(make_gmem_ptr<T>(g_in), gmem_layout);
  Tensor gA = get_gA<rank(SmemLayout{})>(mA, cta_tiler);
  Tensor gO = make_tensor(make_gmem_ptr<T>(g_out), gmem_layout);

  auto gmem_thr_copy = gmem_tiled_copy.get_slice(threadIdx.x);
  Tensor tAgA = gmem_thr_copy.partition_S(gA);     // (TMA,TMA_M,TMA_N,REST_M,REST_N)
  Tensor tAsA = gmem_thr_copy.partition_D(sA);     // (TMA,TMA_M,TMA_N)

  // copy gmem to smem
  copy(gmem_tiled_copy, tAgA, tAsA);
  cp_async_fence();
  cp_async_wait<0>();
  __syncthreads();

  // create vreg with tiled_mma
  int warp_idx = canonical_warp_idx_sync();
  TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
  Tensor tCrA = thr_mma.partition_fragment_A(sA); // (MMA,MMA_M,MMA_K)

  // copy tsm to vreg
  auto smem_tiled_copy_A = make_tiled_copy_A(typename Operand::SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(warp_idx * 32);
  Tensor tCsA            = smem_thr_copy_A.partition_S(make_mix_tensor_like(sA));

  copy(smem_tiled_copy_A, tCsA, tCrA);
  __syncthreads();

  // copy vreg to global
  auto gmem_tiled_copy_O = make_tiled_copy_A(Copy_Atom<DefaultCopy, T>{}, tiled_mma);
  auto gmem_thr_copy_O   = gmem_tiled_copy_O.get_thread_slice(threadIdx.x);
  auto tOgO              = gmem_thr_copy_O.partition_D(gO);

  copy(gmem_tiled_copy_O, tCrA, tOgO);

#if 0
  if (thread0()) {
    print("TILE  :  "); print(cta_tiler); print("\n");
    print("gmem_thr_copy  :  ");       print(gmem_thr_copy);   print("\n");
    print("  gA  :  "); print(  gA);   print("\n");
    print("  gO  :  "); print(  gO);   print("\n");
    print("  sA  :  "); print(  sA);   print("\n");
    print("tAgA  :  "); print(tAgA);   print("\n");
    print("tAsA  :  "); print(tAsA);   print("\n");
    print("tCrA  :\n"); print(tCrA);   print("\n");
    print("tCsA  :\n"); print(tCsA);   print("\n");
    print("gmem_tiled_copy_O  :  ");   print(gmem_tiled_copy_O);   print("\n");
    print("tOgO  :\n"); print(tOgO);   print("\n");
    print("gO    :\n"); print(gO);     print("\n");
  }
#endif

}

template <class T, class GMEM_Layout, class SMEM_Layout, class CTA_Tile>
auto
test_aiu_load_vreg(GMEM_Layout const& gmem_layout,
                   SMEM_Layout const& smem_layout,
                   CTA_Tile    const& cta_tile)
{
  using namespace cute;

  static_assert(rank(GMEM_Layout{}) == 2);
  int shape0 = get<0>(shape(gmem_layout));
  int shape1 = get<1>(shape(gmem_layout));
  int stride0 = get<0>(stride(gmem_layout));
  int stride1 = get<1>(stride(gmem_layout));

  printf("------------------- Test Layout = (%d, %d) -------------------\n", shape0, shape1);

  constexpr bool Trans = false; //get<1>(stride(gmem_layout)) != 1;

  constexpr int SmemH = get<0>(CTA_Tile{});
  constexpr int SmemW = get<1>(CTA_Tile{});
  using Operand = DefaultGemm_AIU_Operand<T, Trans, Int<SmemH>, Int<SmemW>>;

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
  gmem_tiled_copy.desc_ = {nullptr, shape0, stride0, Operand::CUB_H, Operand::CUB_W};

  using TiledMma = TiledMMA<
      decltype(get_ppu_mma_atom<T>()),
      Layout<Shape<_1,_1,_1>>,  // 1x1x1 thread group
      >; // 1x1x1 value group

  // Launch
  int smem_size = int(sizeof(SharedStorage<T, decltype(smem_layout)>));
  aiu_vreg_test_device_cute<T, Operand, TiledMma><<<1, 32, smem_size>>>(
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

  if (count != 3) {
    exit(0);
  }

  return gmem_tiled_copy;
}

} // end namespace cutlass::test
