/***************************************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD. All rights reserved. 
 * Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#define SUPPORT_FP8_SCALING 1

#include <iostream>
#include <random>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/gemm/config/gemm_configs.hpp"
#include "cutlass/gemm/config/gemm_operands.hpp"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/gett.hpp"


#include "helper.h"
#include "ppu_fp4_cmdline.hpp"

using namespace cute;

#define DEVICE_ON  1
#define DATA_CTRL  0

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////

// A matrix configuration
using         ElementA    = cutlass::float4_t;                          // Element type for A matrix operand
using         LayoutA     = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 1;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

// B matrix configuration
using         ElementB    = cutlass::float4_t;                          // Element type for B matrix operand
using         LayoutB     = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
constexpr int AlignmentB  = 1;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

// C matrix configuration
using         ElementC    = float;                          // Element type for C and D matrix operands
using         LayoutC     = cutlass::layout::RowMajor;                   // Layout type for C and D matrix operands
constexpr int AlignmentC  = 1;

// D matrix configuration
using         ElementD    = ElementC;
using         LayoutD     = LayoutC;
constexpr int AlignmentD  = AlignmentC;

// Auxiliary matrix configuration
using         ElementAux   = ElementC;
using         LayoutAux    = LayoutC;

// Core kernel configurations
using ElementAccumulator  = float;                                          // Element type for internal accumulation
using ElementCompute      = float;                                          // Element type for epilogue computation
using ElementScalar    = ElementCompute;


using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<ElementD, AlignmentD, ElementAccumulator, ElementCompute>;

#if 1//DEVICE_ON
static constexpr int BlockM = 256;
static constexpr int BlockN = 256;
static constexpr int BlockK = 128;
static constexpr int WarpM = 64;
static constexpr int WarpN = 64;
static constexpr int WarpK = 128;
static constexpr int Stage = 4;


using WarpOnM = Int<BlockM / WarpM>;
using WarpOnN = Int<BlockN / WarpN>;
static constexpr int ThreadNum = WarpOnM() * WarpOnN() * 32;
static constexpr bool TransA = false;
static constexpr bool TransB = false;

using DispatchPolicy = cutlass::gemm::MainloopWithScalePPU0015Aiu<Stage>;
using GemmOperandA = cutlass::gemm::config::DefaultGemm_AIU_Operand<cutlass::arch::PPU0015, ElementA, TransA, Int<BlockM>, Int<BlockK>, false>;
using GemmOperandB = cutlass::gemm::config::DefaultGemm_AIU_Operand<cutlass::arch::PPU0015, ElementB, TransB, Int<BlockN>, Int<BlockK>, true>;

using TransformA = cute::identity;
using TransformB = cute::identity;

using ScaleCopyInst = UniversalCopy<uint32_t>;
// T0/4/8/12 is successive, T1 is next mma for current warp, below other warps
using ThrLayoutScaleA = decltype(make_layout(make_shape(WarpOnM(), _32{}), make_stride(_32{}, _1{})));
// each thread read 4 int8 in one atom, use int32 layout to fit with copyOp's register
using ValLayoutScaleA = decltype(make_layout(make_shape(_1{}, _1{})));
using GmemTiledCopyScaleA = decltype(
    make_tiled_copy(Copy_Atom<ScaleCopyInst, uint32_t>{},
                    ThrLayoutScaleA{},
                    ValLayoutScaleA{}));

using ThrLayoutScaleB = decltype(make_layout(make_shape(WarpOnN(), _32{}), make_stride(_32{}, _1{})));
// each thread read 4 int8 in one atom, use int32 layout to fit with copyOp's register
using ValLayoutScaleB = decltype(make_layout(make_shape(_1{}, _1{})));
using GmemTiledCopyScaleB = decltype(
    make_tiled_copy(Copy_Atom<ScaleCopyInst, uint32_t>{},
                    ThrLayoutScaleB{},
                    ValLayoutScaleB{}));

using MmaInst = PPU0015_16x16x64_F32F4F4F32_TN;

static constexpr int MmaInstOnN = 16 / size<1>(typename MMA_Traits<MmaInst>::Shape_MNK{});

using TiledMma = cute::TiledMMA<
    cute::MMA_Atom<MmaInst>,
    cute::Layout<Shape<WarpOnM, WarpOnN, _1>>>;

// ElemA/B and LayoutA/B is already transfered
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveMmaScale<
  cutlass::arch::PPU0015, DispatchPolicy, Shape<Int<BlockM>, Int<BlockN>, Int<BlockK>>,
  ElementA, cutlass::detail::TagToStrideA_t<LayoutA>,
  ElementB, cutlass::detail::TagToStrideB_t<LayoutB>,
  TiledMma,
  typename GemmOperandA::GmemTiledCopy, typename GemmOperandA::SmemLayoutAtom, typename GemmOperandA::SmemCopyAtom, TransformA,
  typename GemmOperandB::GmemTiledCopy, typename GemmOperandB::SmemLayoutAtom, typename GemmOperandB::SmemCopyAtom, TransformB,
  GmemTiledCopyScaleA, GmemTiledCopyScaleB
>;

using EpilogueCopyInst = AutoVectorizingCopyWithAssumedAlignment<AlignmentC * sizeof(ElementC) * 8>;
using GemmEpilogueConfiguration = cutlass::gemm::config::DefaultGemm_Epilogue_Configuration<EpilogueCopyInst, float, AlignmentC, Int<BlockM>, Int<BlockN>, WarpOnM, ThreadNum>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::Epilogue<
  cutlass::detail::TagToStrideC_t<cutlass::layout::RowMajor>,
  cutlass::detail::TagToStrideC_t<cutlass::layout::RowMajor>,
  EpilogueOutputOp,
  typename GemmEpilogueConfiguration::SmemLayoutO,
  Copy_Atom<EpilogueCopyInst,float>,
  typename GemmEpilogueConfiguration::GmemTiledCopyO,
  Copy_Atom<EpilogueCopyInst,ElementC>
>;

using GemmKernel = typename cutlass::gemm::kernel::GemmUniversal<
  Shape<int,int,int,int>,
  CollectiveMainloop,
  CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename GemmKernel::StrideA;
using StrideB = typename GemmKernel::StrideB;
using StrideC = typename GemmKernel::StrideC;
using StrideD = typename GemmKernel::StrideD;

#else

using Gemm = void;
using StrideA = cutlass::detail::TagToStrideA_t<LayoutA>;
using StrideB = cutlass::detail::TagToStrideA_t<LayoutB>;
using StrideC = cutlass::detail::TagToStrideA_t<LayoutC>;
using StrideD = cutlass::detail::TagToStrideA_t<LayoutD>;

#endif


/// Initialization
StrideA stride_A;
StrideB stride_B;
StrideA stride_meta_A;
StrideB stride_meta_B;
StrideC stride_C;
StrideD stride_D;
uint64_t seed;

cutlass::HostTensor<ElementA  , LayoutA  > tensor_A;
cutlass::HostTensor<ElementB  , LayoutB  > tensor_B;
cutlass::HostTensor<ElementC  , LayoutC  > tensor_C;
cutlass::HostTensor<ElementD  , LayoutD  > tensor_D;
cutlass::HostTensor<ElementD  , LayoutD  > tensor_ref_D;

cutlass::HostTensor<uint8_t  , LayoutA  > tensor_meta_A;
cutlass::HostTensor<uint8_t  , LayoutB  > tensor_meta_B;



/////////////////////////////////////////////////////////////////////////////////////////////////
/// Testbed utility types
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Result structure
struct Result
{
  double avg_runtime_ms;
  double gflops;
  cutlass::Status status;
  hggcError_t error;
  bool passed;

  Result(
    double avg_runtime_ms = 0,
    double gflops = 0,
    cutlass::Status status = cutlass::Status::kSuccess,
    hggcError_t error = hggcSuccess)
  :
    avg_runtime_ms(avg_runtime_ms), gflops(gflops), status(status), error(error), passed(false)
  {}

};


template <typename Element, typename Layout>
bool print_tensor(cutlass::HostTensor<Element, Layout> &tensor) {
  Element *phost = tensor.host_data();
  for (int j = 0; j < tensor.extent().at(0); ++j) {
    printf("[j=%2d] ", j);
    for (int i = 0; i < tensor.extent().at(1); ++i) {
      int idx = j * tensor.extent().at(1) + i;
      Element val = phost[idx];
      float fval = float(val);
      printf("%5.2f(0x%02x), ", fval, val.raw());
    }
    printf("\n");
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM setup and evaluation
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Initialize operands to be used in the GEMM and reference GEMM
void initialize(const Options &options) {

  stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(options.m, options.k, options.l));
  stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(options.n, options.k, options.l));
  stride_meta_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(options.m, cute::ceil_div(options.k, 16), options.l));
  stride_meta_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(options.n, cute::ceil_div(options.k, 16), options.l));
  stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(options.m, options.n, options.l));
  stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(options.m, options.n, options.l));

  auto a_coord = cutlass::make_Coord(options.m * options.l, options.k);
  auto c_coord = cutlass::make_Coord(options.m * options.l, options.n);
  auto b_coord = cutlass::make_Coord(options.k, options.n * options.l);

  int rectify_scale_k = cute::ceil_div(options.k, 64) * 4;
  auto meta_a_coord = cutlass::make_Coord(options.m * options.l, rectify_scale_k);
  auto meta_b_coord = cutlass::make_Coord(rectify_scale_k, options.n * options.l);
  int rectify_m = (options.m + BlockM - 1) / BlockM * BlockM;
  int rectify_n = (options.n + BlockN - 1) / BlockN * BlockN;

  tensor_A.resize(a_coord);
  tensor_B.resize(b_coord);
  tensor_C.resize(c_coord);
  tensor_D.resize(c_coord);
  tensor_ref_D.resize(c_coord);

  tensor_meta_A.resize(meta_a_coord);
  tensor_meta_B.resize(meta_b_coord);

  uint8_t *data_a = reinterpret_cast<uint8_t*>(tensor_A.host_data());
  uint8_t *data_b = reinterpret_cast<uint8_t*>(tensor_B.host_data());
  uint8_t *meta_data_a = reinterpret_cast<uint8_t*>(tensor_meta_A.host_data());
  uint8_t *meta_data_b = reinterpret_cast<uint8_t*>(tensor_meta_B.host_data());

  int scale_max = 128;
  int scale_min = 126;

  std::srand(0);
  uint8_t f4_array[] = {0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xB, 0xC, 0xD, 0xE, 0xF};
  for (int i = 0; i < options.m * options.k; ++i) {
    uint8_t aval = f4_array[std::rand() % 4] << 4 | f4_array[std::rand() % 4];
    data_a[i] = aval;
  }
  for (int i = 0; i < options.n * options.k; ++i) {
    uint8_t bval = f4_array[std::rand() % 5] << 4 | f4_array[std::rand() % 5];
    data_b[i] = bval;
  }
  
  for( int p = 0; p < options.m; p++) {
    for( int q = 0; q < cute::ceil_div(options.k, 16); q++) {
      auto i = p * rectify_scale_k + q;
      // meta_data_a[i] = (std::rand() % (scale_max - scale_min + 1)) + scale_min;
      meta_data_a[i] = std::rand() % 10 + 122;
    }
  }

  for( int p = 0; p < options.n; p++) {
    for( int q = 0; q < cute::ceil_div(options.k, 16); q++) {
      auto i = p * rectify_scale_k + q;
      // meta_data_a[i] = (std::rand() % (scale_max - scale_min + 1)) + scale_min;
      meta_data_b[i] = std::rand() % 7 + 124;
    }
  }

#if DEVICE_ON
  tensor_A.sync_device();
  tensor_B.sync_device();
  tensor_C.sync_device();
  tensor_D.sync_device();

  tensor_meta_A.sync_device();
  tensor_meta_B.sync_device();
#endif

  hggcFuncAttributes attr;
  hggcFuncGetAttributes(&attr, cutlass::device_kernel<GemmKernel>);
  printf("==========\n");
  printf("reg: %d, stack: %d, kernel resource\n", attr.numRegs, attr.localSizeBytes);

}

#if DEVICE_ON
/// Populates a Gemm::Arguments structure from the given commandline options
typename Gemm::Arguments args_from_options(const Options &options)
{
  // auto aiu_desc_a = cute::AiuDesc{reinterpret_cast<const uint8_t*>(tensor_A.device_data()), options.m, options.k, options.k};
  // auto aiu_desc_b = cute::AiuDesc{reinterpret_cast<const uint8_t*>(tensor_B.device_data()), options.n, options.k, options.k};
  auto meta_a = reinterpret_cast<uint32_t*>(tensor_meta_A.device_data());
  auto meta_b = reinterpret_cast<uint32_t*>(tensor_meta_B.device_data());
  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {options.m, options.n, options.k, options.l},
    {{options.m, options.n, options.k}, tensor_A.device_data(), stride_A, tensor_B.device_data(), stride_B, meta_a, stride_meta_A, meta_b, stride_meta_B},
    {
      {1.0, 0.0}, // epilogue.thread
      tensor_C.device_data(), stride_C,
      tensor_D.device_data(), stride_D
    }
  };


  return arguments;
}
#endif

bool verify(const Options &options) {
  //
  // Compute reference output
  //

  uint8_t *host_a = reinterpret_cast<uint8_t*>(tensor_A.host_data());
  uint8_t *host_b = reinterpret_cast<uint8_t*>(tensor_B.host_data());
  ElementD *host_d = tensor_ref_D.host_data();

  uint8_t *host_meta_a = reinterpret_cast<uint8_t*>(tensor_meta_A.host_data());
  uint8_t *host_meta_b = reinterpret_cast<uint8_t*>(tensor_meta_B.host_data());

  int M = options.m;
  int N = options.n;
  int K = options.k;

  int rectify_scale_k = cute::ceil_div(options.k, 64) * 4;

  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      ElementD *out = &host_d[m * N + n];
      for (int k = 0; k < K; ++k) {
        uint8_t int8_a = host_a[m * K + k];
        uint8_t int8_b = host_b[n * K + k];
        int8_t int8_meta_a = static_cast<int8_t>(static_cast<int>(host_meta_a[m * rectify_scale_k + k / 16]) - 127);
        int8_t int8_meta_b = static_cast<int8_t>(static_cast<int>(host_meta_b[n * rectify_scale_k + k / 16]) - 127);
        float fp32_meta_a = std::pow(2, int8_meta_a);
        float fp32_meta_b = std::pow(2, int8_meta_b);
        for (int i = 0; i < 2; ++i) {
          cutlass::float4_t fp4_a, fp4_b;
          int shift_bit = i == 0 ? 4 : 0;
          fp4_a.storage = (int8_a >> shift_bit) & 0x0f;
          fp4_b.storage = (int8_b >> shift_bit) & 0x0f;
          float fp32_a = fp4_a.to_float();
          float fp32_b = fp4_b.to_float();
          *out += fp32_a * fp32_meta_a * fp32_b * fp32_meta_b;
        }
      }
    }
  }

  tensor_D.sync_host();

  float *tmp_a = reinterpret_cast<float*>(tensor_D.host_view().data());
  float *tmp_b = reinterpret_cast<float*>(tensor_ref_D.host_view().data());

  printf("=====ref=====\n");
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      printf("%.3f, ", tmp_b[i * N + j]);
    }
    printf("\n");
  }
  printf("=====dev=====\n");
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      printf("%.3f, ", tmp_a[i * N + j]);
    }
    printf("\n");
  }

  bool passed = true;
  int wrong_num = 0;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float cur_ref = tmp_b[i * N + j];
      float cur_dev = tmp_a[i * N + j];
      if (abs((cur_ref - cur_dev) / cur_ref) > 0.02) {
        ++wrong_num;
        // printf("=====error=====\n");
        // printf("i: %d, j: %d, ref: %f, dev: %f\n", i, j, cur_ref, cur_dev);
        // passed = false;
        // break;
      }
    }
    // if (passed == false) {
    //   break;
    // }
  }

  float wrong_ratio = float(wrong_num) / M / N;
  if (wrong_ratio > 0.01) {
    passed = false;
  }
  printf("wrong_ratio: %f\n", wrong_ratio);

  return passed;
}

/// Execute a given example GEMM computation
int run(Options &options)
{
  initialize(options);

#if DEVICE_ON
  // Instantiate kernel depending on templates
  Gemm gemm;

  // Create a structure of gemm kernel arguments suitable for invoking an instance of Gemm
  auto arguments = args_from_options(options);

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check if the problem size is supported or not
  CUTLASS_CHECK(gemm.can_implement(arguments));

  // Initialize kernel with arguments and workspace pointer
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));

  // Correctness / Warmup iteration
  CUTLASS_CHECK(gemm.run());
#endif

  // Check if output from kernel and reference kernel are equal or not
  Result result;
  result.passed = verify(options);
  // result.passed = true;

  std::cout << "  Disposition: " << (result.passed ? "Passed" : "Failed") << std::endl;

  if (!result.passed) {
    exit(-1);
  }

  return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {
  hggcDeviceProp props;
  int current_device_id;
  CUTLASS_PPU_CHECK(hggcGetDevice(&current_device_id));
  CUTLASS_PPU_CHECK(hggcGetDeviceProperties(&props, current_device_id));
  hggcError_t error = hggcGetDeviceProperties(&props, 0);

  // should run on PPU 1.5
  if (props.major != 8 || props.minor != 9) {
    std::cerr << " This example should be run on PPU 1.5!!! " << std::endl;
    return 0;
  }

  //
  // Parse options
  //
  Options options;
  options.parse(argc, args);
  // regard as int8 gemm
  options.k /= 2;

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  //
  // Evaluate kernels
  //
  run(options);
  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
