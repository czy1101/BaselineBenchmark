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

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/gemm/config/gemm_configs.hpp"

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
#include "ppu_fp8_cmdline.hpp"

using namespace cute;

#define DEVICE_ON  1
#define DATA_CTRL  0


constexpr bool UseAIU = true;
constexpr bool UseEVT = true;
/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////

// A matrix configuration
using         ElementA    = cutlass::float_e4m3_t;                          // Element type for A matrix operand
using         LayoutA     = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = UseAIU ? 1 : 128 / cutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

// B matrix configuration
using         ElementB    = cutlass::float_e5m2_t;                          // Element type for B matrix operand
using         LayoutB     = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
constexpr int AlignmentB  = UseAIU ? 1 : 128 / cutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

// C matrix configuration
using         ElementC    = cutlass::float_e4m3_t;                          // Element type for C and D matrix operands
using         LayoutC     = cutlass::layout::RowMajor;                   // Layout type for C and D matrix operands
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)

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

static constexpr cutlass::FloatRoundStyle RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;
using cutlass::epilogue::fusion::PPUCompute;
using cutlass::epilogue::fusion::PPUEVT;
using LinearCombOutType = cute::conditional_t<std::is_same<ElementA, cutlass::float_e4m3_t>::value || std::is_same<ElementA, cutlass::float_e5m2_t>::value,
                                              ElementCompute,
                                              ElementD>;
using DefaultOperation = cutlass::epilogue::fusion::LinearCombination<ElementD, ElementCompute, ElementC, ElementScalar, RoundStyle>;

using FusionCallbacks = PPUEVT< // D = activation(Z) * scale_d, amax_d = max(abs(elements in D))
  PPUCompute<cutlass::epilogue::fusion::detail::ScaleOutOp<ElementD>::template Op, ElementD, ElementCompute, RoundStyle>, // activation(Z) * scale_d
    PPUEVT<PPUCompute<cutlass::epilogue::thread::Identity, ElementCompute, ElementCompute, RoundStyle>, // activation(Z)
      PPUEVT<PPUCompute<cutlass::homogeneous_multiply_add, ElementCompute, ElementCompute, RoundStyle>, // Z = scale_a * scale_b * alpha * A * B + beta * scale_c * C
        cutlass::epilogue::fusion::PPUScalarBroadcast<ElementScalar, Stride<_0,_0,int64_t>, 2>, // scale_c * beta
        cutlass::epilogue::fusion::PPUSrcFetch<ElementC>, // c
        PPUEVT<PPUCompute<cutlass::multiplies, ElementCompute, ElementCompute, RoundStyle>, // scale_a * scale_b * alpha * A * B
          cutlass::epilogue::fusion::PPUScalarBroadcast<ElementScalar, Stride<_0,_0,int64_t>, 3>, // scale_a * scale_b * alpha
          cutlass::epilogue::fusion::PPUAccFetch // acc
        >
      >
    >,
    cutlass::epilogue::fusion::PPUScalarBroadcast<ElementScalar> // scale_d
  >;

#if 1//DEVICE_ON
static constexpr int BlockM = 128;
static constexpr int BlockN = 128;
static constexpr int BlockK = UseAIU ? 64 : 32;
static constexpr int WarpM = 64;
static constexpr int WarpN = 64;
static constexpr int WarpK = UseAIU ? 64 : 32;
static constexpr int Stage = 2;
using GemmKernel = typename cutlass::gemm::config::GemmKernelConfig<
  cutlass::arch::PPU0015,
  ElementA, LayoutA, AlignmentA,
  ElementB, LayoutB, AlignmentB,
  ElementAccumulator, ElementD, AlignmentD,
  ElementD, AlignmentD, ElementA,
  BlockM, BlockN, BlockK,
  WarpM, WarpN, WarpK,
  Stage,
  cute::conditional_t<UseEVT, FusionCallbacks, DefaultOperation>,
  UseAIU,
  false,
  UseEVT,
  cutlass::config::KernelScheduleType::FP8_SPLIT_ACC
>::GemmKernel;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename GemmKernel::StrideA;
using StrideB = typename GemmKernel::StrideB;
using StrideC = typename GemmKernel::StrideC;
using StrideD = typename GemmKernel::StrideD;

#else

using Gemm = void;
using StrideA = cutlass::detail::TagToStrideA_t<LayoutA>;
using StrideB = cutlass::detail::TagToStrideB_t<LayoutB>;
using StrideC = cutlass::detail::TagToStrideC_t<LayoutC>;
using StrideD = cutlass::detail::TagToStrideC_t<LayoutD>;

#endif


/// Initialization
StrideA stride_A;
StrideB stride_B;
StrideC stride_C;
StrideD stride_D;
uint64_t seed;

cutlass::HostTensor<ElementA  , LayoutA  > tensor_A;
cutlass::HostTensor<ElementB  , LayoutB  > tensor_B;
cutlass::HostTensor<ElementC  , LayoutC  > tensor_C;
cutlass::HostTensor<ElementD  , LayoutD  > tensor_D;
cutlass::HostTensor<ElementD  , LayoutD  > tensor_ref_D;

using LayoutScalar = cutlass::layout::PackedVectorLayout;
cutlass::HostTensor<ElementScalar, LayoutScalar> scalar_alpha;
cutlass::HostTensor<ElementScalar, LayoutScalar> scalar_beta;
cutlass::HostTensor<ElementScalar, LayoutScalar> scale_A;
cutlass::HostTensor<ElementScalar, LayoutScalar> scale_B;
cutlass::HostTensor<ElementScalar, LayoutScalar> scale_C;
cutlass::HostTensor<ElementScalar, LayoutScalar> scale_D;


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

template <typename Element>
bool fill_data(Element *pdata, size_t m, size_t n, float base = 0.0) {
  for (auto i = 0; i < m; ++i) {
    for (auto j = 0; j < n; ++j) {
      size_t idx = i * n + j;
      pdata[idx] = (j > i) ? Element(0) : Element(0.1 * (i+1) + base);
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM setup and evaluation
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to initialize a block of device data
template <typename Element, typename Layout>
bool initialize_tensor(
  cutlass::TensorView<Element, Layout> view,
  uint64_t seed) {

  double scope_max, scope_min;
  int bits_input = cutlass::sizeof_bits<Element>::value;
  int bits_output = cutlass::sizeof_bits<Element>::value;

  if (bits_input == 1) {
    scope_max = 2;
    scope_min = 0;
  }
  else if (bits_input <= 8) {
    scope_max = 2;
    scope_min = -2;
  }
  else if (bits_output == 16) {
    scope_max = 5;
    scope_min = -5;
  }
  else {
    scope_max = 8;
    scope_min = -8;
  }
  cutlass::reference::host::TensorFillRandomUniform(
    view, seed, scope_max, scope_min, 0);

  return true;
}

/// Initialize operands to be used in the GEMM and reference GEMM
void initialize(const Options &options) {

  stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(options.m, options.k, options.l));
  stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(options.n, options.k, options.l));
  stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(options.m, options.n, options.l));
  stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(options.m, options.n, options.l));

  auto a_coord = cutlass::make_Coord(options.m * options.l, options.k);
  auto c_coord = cutlass::make_Coord(options.n * options.l, options.m);
  auto b_coord = cutlass::make_Coord(options.k, options.n * options.l);

  tensor_A.resize(a_coord);
  tensor_B.resize(b_coord);
  tensor_C.resize(c_coord);
  tensor_D.resize(c_coord);
  tensor_ref_D.resize(c_coord);

#if DATA_CTRL
  fill_data(tensor_A.host_data(), options.m, options.k, 0.0);
  fill_data(tensor_B.host_data(), options.n, options.k, 1.0);
  fill_data(tensor_C.host_data(), options.m, options.n, -1.0);
#else
  initialize_tensor(tensor_A.host_view(), seed + 2022);
  initialize_tensor(tensor_B.host_view(), seed + 2023);
  initialize_tensor(tensor_C.host_view(), seed + 2024);
#endif

#if DEVICE_ON
  tensor_A.sync_device();
  tensor_B.sync_device();
  tensor_C.sync_device();
  tensor_D.sync_device();
#endif

  if (options.device_scale) {
    scalar_alpha.resize(cutlass::make_Coord(1));
    scalar_beta.resize(cutlass::make_Coord(1));
    scale_A.resize(cutlass::make_Coord(1));
    scale_B.resize(cutlass::make_Coord(1));
    scale_C.resize(cutlass::make_Coord(1));
    scale_D.resize(cutlass::make_Coord(1));

    cutlass::reference::host::TensorFill(scalar_alpha.host_view(), options.alpha);
    cutlass::reference::host::TensorFill(scalar_beta.host_view(), options.beta);
    cutlass::reference::host::TensorFill(scale_A.host_view(), options.scale_a);
    cutlass::reference::host::TensorFill(scale_B.host_view(), options.scale_b);
    cutlass::reference::host::TensorFill(scale_C.host_view(), options.scale_c);
    cutlass::reference::host::TensorFill(scale_D.host_view(), options.scale_d);

    scalar_alpha.sync_device();
    scalar_beta.sync_device();
    scale_A.sync_device();
    scale_B.sync_device();
    scale_C.sync_device();
    scale_D.sync_device();
  }
}

#if DEVICE_ON
/// Populates a Gemm::Arguments structure from the given commandline options
typename Gemm::Arguments args_from_options(const Options &options)
{
  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {options.m, options.n, options.k, options.l},
    {tensor_A.device_data(), stride_A, tensor_B.device_data(), stride_B},
    {
      {}, // epilogue.thread
      tensor_C.device_data(), stride_C,
      tensor_D.device_data(), stride_D
    }
  };

  // leaving/setting these as nullptr disables the fusion at runtime
  // scale_args.bias_ptr = nullptr;

  return arguments;
}
#endif

bool verify(const Options &options) {
  //
  // Compute reference output
  //

  using RefStrideA = cutlass::detail::TagToStrideA_t<LayoutA>;
  using RefStrideB = cutlass::detail::TagToStrideB_t<LayoutB>;
  using RefStrideC = cutlass::detail::TagToStrideC_t<LayoutC>;
  using RefStrideD = cutlass::detail::TagToStrideC_t<LayoutD>;
  RefStrideA ref_stride_A = cutlass::make_cute_packed_stride(RefStrideA{}, cute::make_shape(options.m, options.k, options.l));
  RefStrideB ref_stride_B = cutlass::make_cute_packed_stride(RefStrideB{}, cute::make_shape(options.n, options.k, options.l));
  RefStrideC ref_stride_C = cutlass::make_cute_packed_stride(RefStrideC{}, cute::make_shape(options.m, options.n, options.l));
  RefStrideD ref_stride_D = cutlass::make_cute_packed_stride(RefStrideD{}, cute::make_shape(options.m, options.n, options.l));

  // Create instantiation for device reference gemm kernel
  auto A = cute::make_tensor(tensor_A.host_data(),
      cute::make_layout(cute::make_shape(options.m, options.k, options.l), ref_stride_A));
  auto B = cute::make_tensor(tensor_B.host_data(),
      cute::make_layout(cute::make_shape(options.n, options.k, options.l), ref_stride_B));
  auto C = cute::make_tensor(tensor_C.host_data(),
      cute::make_layout(cute::make_shape(options.m, options.n, options.l), ref_stride_C));
  auto D = cute::make_tensor(tensor_ref_D.host_data(),
      cute::make_layout(cute::make_shape(options.m, options.n, options.l), ref_stride_D));
  using unused_t = decltype(D);

  cutlass::reference::host::GettMainloopParams<ElementAccumulator, decltype(A), decltype(B)> mainloop_params{A, B};

  using ActivationFunctor = cutlass::epilogue::thread::Identity<ElementCompute>;
  cutlass::reference::host::GettEpilogueParams<
      ElementScalar,
      ElementScalar,
      ElementAccumulator,
      ElementCompute,
      decltype(C),
      decltype(D),
      unused_t, // bias
      unused_t, // aux
      unused_t, // valpha
      unused_t  // vbeta
  > epilogue_params;

  epilogue_params.C = C;
  epilogue_params.D = D;
  epilogue_params.alpha = options.alpha;
  epilogue_params.beta = options.beta;
  epilogue_params.scale_a = options.scale_a;
  epilogue_params.scale_b = options.scale_b;
  epilogue_params.scale_c = options.scale_c;
  epilogue_params.scale_d = options.scale_d;


  // get reference result
  cutlass::reference::host::Gemm3x(mainloop_params, epilogue_params);

  // compare_reference
  tensor_D.sync_host();
#if DATA_CTRL
  printf("===== tensor A ===== \n");
  print_tensor(tensor_A);
  printf("===== tensor B ===== \n");
  print_tensor(tensor_B);
  printf("===== tensor C ===== \n");
  print_tensor(tensor_C);
  printf("===== tensor D ===== \n");
  print_tensor(tensor_D);
  printf("===== tensor ref D ===== \n");
  print_tensor(tensor_ref_D);
#elif 0
  ElementD *pref = tensor_ref_D.host_data();
  ElementD *pact = tensor_D.host_data();
  for (int j = 0; j < options.n; ++j) {
    for (int i = 0; i < options.m; ++i) {
      int idx = j * options.m + i;
      ElementD ref = pref[idx];
      ElementD act = pact[idx];
      float fref = float(ref);
      float fact = float(act);
      if (fref != fact)
        printf("(i,j)=(%2d,%2d) ref=%9.6f(0x%02x), act=%9.6f(0x%02x)\n", i, j, fref, ref.raw(), fact, act.raw());
    }
  }
#endif

  bool passed = cutlass::reference::host::TensorEquals(tensor_ref_D.host_view(), tensor_D.host_view());

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

  // evt realization must construct evt params on device, can't use GemmUniversalAdapter
  typename GemmKernel::Params params = GemmKernel::to_underlying_arguments(arguments, workspace.get());

  using ActivationArguments = typename cutlass::epilogue::fusion::PPUCompute<
                                          cutlass::epilogue::thread::Identity,
                                          ElementD,
                                          ElementCompute,
                                          RoundStyle>::Arguments;
  ActivationArguments activation = ActivationArguments();
  using StrideAlpha = Stride<_0,_0,int64_t>;
  using StrideBeta  = Stride<_0,_0,int64_t>;
  StrideAlpha dAlpha = {_0{}, _0{}, 0};
  StrideBeta  dBeta  = {_0{}, _0{}, 0};

  params.epilogue.thread =
  {    // binary op : D = activation(Z) * scale_d or activation(Z)
    { // unary op : activation(Z)
      { // ternary op : (scale_c * beta) * C + ((scale_a * scale_b * alpha) * acc + bias)
        {
          {options.beta, options.scale_c},
          {scalar_beta.device_data(), scale_C.device_data()},
          {dBeta, {_0{}, _0{}, 0}}
        }, //leaf args: (scale_c * beta)
        {}, //leaf args: C
        { //binary op:  scale_a * scale_b * alpha * A * B
          {
            {options.alpha, options.scale_a, options.scale_b},
            {scalar_alpha.device_data(), scale_A.device_data(), scalar_alpha.device_data()},
            {dAlpha, {_0{}, _0{}, 0}, {_0{}, _0{}, 0}},
          },
          {}, //leaf args: acc
          {}, //binary args: multiplies
        },
        {} // ternary args : multiplies_add
      },
      {} //activation
    }, // end unary op
    {{options.scale_d}, {scale_D.device_data()}, {}}, // leaf args : scale_d
    {} // binary args: multiplies or first
  };   // end ternary op

  dim3 const block = GemmKernel::get_block_shape();
  dim3 const grid = GemmKernel::get_grid_shape(params);
  int smem_size = GemmKernel::SharedStorageSize;
  cutlass::device_kernel<GemmKernel><<<grid, block, smem_size>>>(params);

  hggcError_t kernel_result = hggcDeviceSynchronize();
  if (kernel_result != hggcSuccess) {
    std::cerr << "Error running the kernel. Last device error is: "
              << hggcGetErrorString(kernel_result) << std::endl;
    return false;
  }

#endif

  // Check if output from kernel and reference kernel are equal or not
  Result result;
  result.passed = verify(options);

  std::cout << "  Disposition: " << (result.passed ? "Passed" : "Failed") << std::endl;

  if (!result.passed) {
    exit(-1);
  }

#if DEVICE_ON
  // Run profiling loop
  if (options.iterations > 0)
  {
    PpuTimer timer;
    timer.start();
    for (int iter = 0; iter < options.iterations; ++iter) {
      cutlass::device_kernel<GemmKernel><<<grid, block, smem_size>>>(params);
    }
    timer.stop();

    // Compute average runtime and GFLOPs.
    float elapsed_ms = timer.elapsed_millis();
    result.avg_runtime_ms = double(elapsed_ms) / double(options.iterations);
    result.gflops = options.gflops(result.avg_runtime_ms / 1000.0);

    std::cout << "  Problem Size: " << options.m << 'x' << options.n << 'x' << options.k << 'x' << options.l << std::endl;
    std::cout << "  Avg runtime: " << result.avg_runtime_ms << " ms" << std::endl;
    std::cout << "  GFLOPS: " << result.gflops << std::endl;
  }
#endif

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
