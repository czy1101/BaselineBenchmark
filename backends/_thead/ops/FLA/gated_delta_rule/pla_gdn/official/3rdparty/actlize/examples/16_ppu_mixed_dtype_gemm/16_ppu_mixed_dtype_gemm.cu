/***************************************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD. All rights reserved. 
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/*! \file
    \brief PPU GEMM example with different data types for PPU architecture

    Examples:
      
      Runs the mixed input batched gemm (with batch size 2), converting B to the type of A (mode 0)
      $ ./examples/16_ppu_mixed_dtype_gemm/16_ppu_mixed_dtype_gemm --m=2048 --n=2048 --k=2048 --l=2 --mode=0

      Runs the mixed input gemm, and applies a scaling factor to B before mma (mode 1). Applies a vector of scales to the entire
      matrix (group size is the same as the gemm k dimension).
      $ ./examples/16_ppu_mixed_dtype_gemm/16_ppu_mixed_dtype_gemm --m=4096 --n=5120 --k=8192 --g=8192 --mode=1

      Runs the mixed input gemm, and applies a scaling factor and adds a zero-point to B before mma (mode 2). Uses a group size of 128.
      $ ./examples/16_ppu_mixed_dtype_gemm/16_ppu_mixed_dtype_gemm --m=2048 --n=5120 --k=8192 --g=128 --mode=2
*/

#include <iostream>

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/device/tensor_compare.h"

#include "helper.h"
#include "unfused_weight_dequantize.hpp"

#include "ppu_include.hpp"
#include "cutlass/gemm/collective/builders/ppu_mma_builder.inl"

using namespace cute;


// This is just an example, so we use a regular enum so we can compare directly to the command-line int.
enum GemmMode {
  ConvertOnly,
  ScaleOnly,
  ScaleWithZeroPoint
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////
using MmaType = cutlass::bfloat16_t;
// using MmaType = cutlass::half_t;
// using QuantType = cutlass::int8_t;
using QuantType = cutlass::int4b_t;
constexpr int TileShapeK = 128 * 8 / sizeof_bits<MmaType>::value;

// A matrix configuration
using         ElementA    = MmaType;                                        // Element type for A matrix operand
using         LayoutA     = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

// B matrix configuration
using         ElementB    = QuantType;                                      // Element type for B matrix operand
using         LayoutB     = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
using         LayoutB_opt = cutlass::layout::ColumnMajorInterleaved<256>;
// using         LayoutB_opt = LayoutB;
constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

// This example manually swaps and transposes, so keep transpose of input layouts
using LayoutA_Transpose = typename cutlass::layout::LayoutTranspose<LayoutA>::type;
using LayoutB_Transpose = typename cutlass::layout::LayoutTranspose<LayoutB>::type;

using ElementZero = MmaType;
using ElementScale = MmaType;
using LayoutScale = cutlass::layout::RowMajor;

// C/D matrix configuration
using         ElementC    = MmaType;                                        // Element type for C and D matrix operands
using         LayoutC     = cutlass::layout::RowMajor;                      // Layout type for C and D matrix operands
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)

// D matrix configuration
using         ElementD    = ElementC;
using         LayoutD     = LayoutC;
constexpr int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value;

// Core kernel configurations
using ElementAccumulator  = float;                                          // Element type for internal accumulation
using ElementCompute      = float;                                          // Element type for epilogue computation
using ArchTag             = cutlass::arch::PPU0010;                         // Tag indicating the minimum CU that supports the intended feature
using OperatorClass       = cutlass::arch::OpClassTensorOp;                 // Operator class tag
using TileShape           = Shape<_32,_32,cute::Int<TileShapeK>>;           // Threadblock-level tile size
using WarpShape           = Shape<_16,_16,cute::Int<TileShapeK>>;           // Warp Shape

using KernelSchedule      = cutlass::gemm::KernelTmaWarpSpecializedCooperativeMixedInput;  // Kernel to launch based on the default setting in the Collective Builder 
using EpilogueSchedule    = cutlass::epilogue::EpilogueSimtVectorized;
using EpilogueTileType    = cutlass::epilogue::collective::EpilogueTileAuto;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    TileShape, WarpShape,
    EpilogueTileType,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    EpilogueSchedule
  >::CollectiveOp;

// ============================================================ MIXED INPUT NO SCALES ============================================================================
// The collective will infer that the narrow type should be upcasted to the wide type.
// We swap A and B operands to the builder here
using CollectiveMainloopConvertOnly = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB_opt, AlignmentB,
    ElementAccumulator,
    cute::tuple<TileShape>, WarpShape,
    Int<3>,
    KernelSchedule
  >::CollectiveOp;

using GemmKernelConvertOnly = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>, // Indicates ProblemShape
    CollectiveMainloopConvertOnly,
    CollectiveEpilogue
>;

using GemmConvertOnly = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelConvertOnly>;

// =========================================================== MIXED INPUT WITH SCALES ===========================================================================
// The Scale information must get paired with the operand that will be scaled. In this example, B is scaled so we make a tuple of B's information and the scale information.

using CollectiveMainloopScaleOnly = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA,
    cute::tuple<ElementB, ElementScale>, LayoutB_opt, AlignmentB,
    ElementAccumulator,
    cute::tuple<TileShape>, WarpShape,
    Int<3>,
    KernelSchedule
  >::CollectiveOp;

using GemmKernelScaleOnly = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>, // Indicates ProblemShape
    CollectiveMainloopScaleOnly,
    CollectiveEpilogue
>;

using GemmScaleOnly = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelScaleOnly>;

// =========================================================== MIXED INPUT WITH SCALES AND ZEROS ==================================================================
// We specify scale + zero elements to indicate that we require both. Scales and biases have the same format.
using CollectiveMainloopScaleWithZeroPoint = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA,
    cute::tuple<ElementB, ElementScale, ElementZero>, LayoutB_opt, AlignmentB,
    ElementAccumulator,
    cute::tuple<TileShape>, WarpShape,
    Int<3>,
    KernelSchedule
  >::CollectiveOp;

using GemmKernelScaleWithZeroPoint = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>, // Indicates ProblemShape
    CollectiveMainloopScaleWithZeroPoint,
    CollectiveEpilogue
>;

using GemmScaleWithZeroPoint = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelScaleWithZeroPoint>;
// =================================================================================================================================================================

using StrideA = cutlass::detail::TagToStrideA_t<LayoutA>;
using StrideB = cutlass::detail::TagToStrideB_t<LayoutB>;
using StrideC = typename GemmKernelConvertOnly::StrideC;
using StrideD = typename GemmKernelConvertOnly::StrideD;

using StrideC_ref = cutlass::detail::TagToStrideC_t<LayoutC>;
using StrideD_ref = cutlass::detail::TagToStrideC_t<LayoutD>;

//
// Data members
//

/// Initialization
StrideA stride_A;
StrideB stride_B;
StrideC stride_C;
StrideC_ref stride_C_ref;
StrideD stride_D;
StrideD_ref stride_D_ref;
uint64_t seed;

// Scale and Zero share a stride since the layout and shapes must be the same.
using StrideS = typename CollectiveMainloopScaleWithZeroPoint::StrideScale;
using StrideS_ref = cutlass::detail::TagToStrideB_t<LayoutScale>;
StrideS stride_S;
StrideS_ref stride_S_ref;

cutlass::DeviceAllocation<ElementA> block_A;
cutlass::DeviceAllocation<ElementB> block_B;
cutlass::HostTensor<QuantType, LayoutB> tensor_B;
cutlass::HostTensor<QuantType, LayoutB> block_B_buff;
cutlass::DeviceAllocation<ElementA> block_B_dq;
cutlass::DeviceAllocation<ElementScale> block_scale;
cutlass::DeviceAllocation<ElementZero> block_zero;
cutlass::DeviceAllocation<ElementC> block_C;
cutlass::DeviceAllocation<typename GemmConvertOnly::EpilogueOutputOp::ElementOutput> block_D;
cutlass::DeviceAllocation<typename GemmConvertOnly::EpilogueOutputOp::ElementOutput> block_ref_D;


/////////////////////////////////////////////////////////////////////////////////////////////////
/// Testbed utility types
/////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help = false;

  float alpha = 1.0f;
  float beta = 0.0f;
  int iterations = 10;
  int mode = 2;
  int m = 5120, n = 4096, k = 4096;
  int g = 128;
  int l = 1;

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("m", m);
    cmd.get_cmd_line_argument("n", n);
    cmd.get_cmd_line_argument("k", k);
    cmd.get_cmd_line_argument("l", l);
    cmd.get_cmd_line_argument("g", g);
    cmd.get_cmd_line_argument("mode", mode);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("iterations", iterations);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "16_ppu_mixed_dtype_gemm\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --m=<int>                   Sets the M extent of the GEMM\n"
      << "  --n=<int>                   Sets the N extent of the GEMM\n"
      << "  --k=<int>                   Sets the K extent of the GEMM\n"
      << "  --l=<int>                   The number of independent gemm problems with mnk shape\n"
      << "  --g=<int>                   The size of each group for the scales and zeros. To broadcast a vector of scales or zeros, set the group size to K.\n"
      << "  --mode=<int>                The mode to run the gemm. 0 does (A @ B), 1 means A @ (scale * B), 2 means A @ (scale * B + zero-point).\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n\n"
      << "  --iterations=<int>          Number of profiling iterations to perform.\n\n";

    out
      << "\n\nExamples:\n\n"
      << "$ " << "16_ppu_mixed_dtype_gemm" << " --m=1024 --n=512 --k=1024 -g 0 --l=10 --alpha=2 --mode=2 --beta=0.707 \n\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const
  {
    // Two flops per multiply-add
    uint64_t flop = uint64_t(2) * m * n * k * l;
    double gflop = double(flop) / double(1.0e9);
    return gflop / runtime_s;
  }
};

/// Result structure
struct Result
{
  double avg_runtime_ms = 0.0;
  double gflops = 0.0;
  cutlass::Status status = cutlass::Status::kSuccess;
  hggcError_t error = hggcSuccess;
  bool passed = false;

};


/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM setup and evaluation
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to initialize a block of device data
template <class Element>
bool initialize_tensor(
  cutlass::DeviceAllocation<Element>& block,
  uint64_t seed=2023) {

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
  cutlass::reference::device::BlockFillRandomUniform(
      block.get(), block.size(), seed, Element(scope_max), Element(scope_min));

  return true;
}

template <typename Element>
bool initialize_quant_tensor(
  cutlass::DeviceAllocation<Element>& block,
  uint64_t seed=2023) {

  float scope_min = float(cutlass::platform::numeric_limits<Element>::lowest());
  float scope_max = float(cutlass::platform::numeric_limits<Element>::max());

  cutlass::reference::device::BlockFillRandomUniform(
    block.get(), block.size(), seed, Element(scope_max), Element(scope_min));

  return true;
}

template <class Element>
bool initialize_scale(
  cutlass::DeviceAllocation<Element>& block, 
  Options const& options) {
  
  if (options.mode == GemmMode::ConvertOnly) {
    // No scales, so just initialize with 1 so we can use the same kernel to dequantize the data.
    std::vector<Element> stage(block.size(), Element(1.0f));
    block.copy_from_host(stage.data());
  } 
  else {
    float elt_max_f = float(cutlass::platform::numeric_limits<QuantType>::max());
    const float max_dequant_val = 4.f;
    const float min_dequant_val = 0.5f;

    float scope_max(max_dequant_val / elt_max_f);
    float scope_min(min_dequant_val / elt_max_f);

    cutlass::reference::device::BlockFillRandomUniform(
      block.get(), block.size(), seed, Element(scope_max), Element(scope_min));
  }
  return true;
}

template <class Element>
bool initialize_zero(
  cutlass::DeviceAllocation<Element>& block,
  Options const& options) {
  
  if (options.mode == GemmMode::ScaleWithZeroPoint) {
    cutlass::reference::device::BlockFillRandomUniform(
      block.get(), block.size(), seed, Element(2.0f), Element(-2.0f));
  } else {
    // No bias, so just initialize with 1 so we can use the same kernel to dequantize the data.
    std::vector<Element> stage(block.size(), Element(0.0f));
    block.copy_from_host(stage.data());
  }
  return true;
}

/// Initialize operands to be used in the GEMM and reference GEMM
void initialize(Options const& options) {

  auto shape_b = cute::make_shape(options.n, options.k, options.l);
  int const scale_k = (options.k + options.g - 1) / options.g;
  stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(options.m, options.k, options.l));
  stride_B = cutlass::make_cute_packed_stride(StrideB{}, shape_b);
  // Reverse stride here due to swap and transpose
  stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(options.n, options.m, options.l));
  stride_C_ref = cutlass::make_cute_packed_stride(StrideC_ref{}, cute::make_shape(options.m, options.n, options.l));
  // Reverse stride here due to swap and transpose
  stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(options.n, options.m, options.l));
  stride_D_ref = cutlass::make_cute_packed_stride(StrideD_ref{}, cute::make_shape(options.m, options.n, options.l));

  auto a_coord = cutlass::make_Coord(options.m * options.l, options.k);
  auto b_coord = cutlass::make_Coord(options.k, options.n * options.l);
  auto c_coord = cutlass::make_Coord(options.m * options.l, options.n);

  block_A.reset(a_coord.product());
  block_B.reset(b_coord.product());
  block_B_buff.resize(b_coord);
  tensor_B.resize(b_coord);
  block_B_dq.reset(b_coord.product());
  block_C.reset(c_coord.product());
  block_D.reset(c_coord.product());
  block_ref_D.reset(c_coord.product());

  block_scale.reset(scale_k * options.l * options.n);
  block_zero.reset(scale_k * options.l * options.n);

  initialize_tensor(block_A, seed + 2022);
  initialize_quant_tensor(block_B, seed + 2021);
  initialize_tensor(block_C, seed + 2020);
  initialize_scale(block_scale, options);
  initialize_zero(block_zero, options);

  block_B.copy_to_host(tensor_B.host_data());
  
  auto layout_B = make_layout(shape_b, stride_B);

  auto shape_scale_zero = cute::make_shape(options.n, scale_k, options.l);
  stride_S = cutlass::make_cute_packed_stride(StrideS{}, cute::make_shape(options.n, scale_k, options.l));
  stride_S_ref = cutlass::make_cute_packed_stride(StrideS_ref{}, cute::make_shape(options.n, scale_k, options.l));
  auto layout_scale_zero = make_layout(shape_scale_zero);

  dequantize_weight(block_B_dq.get(), block_B.get(), layout_B, block_scale.get(), block_zero.get(), layout_scale_zero, options.g);
  
  int row = options.n;
  int col = options.k;
  int batch = options.l;

  constexpr bool is_rowmajor = std::is_same<LayoutB, cutlass::layout::RowMajor>::value;
  QuantTypeClass quant_type;
  if (sizeof_bits<QuantType>::value == 8) {
    quant_type = QuantTypeClass::INT8_WEIGHT_ONLY;
  } else if (sizeof_bits<QuantType>::value == 4) {
    quant_type = QuantTypeClass::PACKED_INT4_WEIGHT_ONLY;
  } else {
    std::cerr << "unsupported QuantType" << std::endl;
    exit(-1);
  }
  for (int b = 0; b < batch; b++) {
    int64_t batch_offset = b * row * col;
    // preprocess_weights_for_mixed_gemm<is_rowmajor, -1>((int8_t*)(&block_B_buff.host_data()[batch_offset]),
    preprocess_weights_for_mixed_gemm<is_rowmajor, 256>((int8_t*)(&block_B_buff.host_data()[batch_offset]),
        (int8_t*)(&tensor_B.host_data()[batch_offset]), {col, row}, quant_type);
  }
  block_B_buff.sync_device();
}

/// Populates a Gemm::Arguments structure from the given commandline options
template <typename Args>
Args args_from_options(Options const& options)
{
// Swap the A and B tensors, as well as problem shapes here.
  if (options.mode == GemmMode::ConvertOnly) {
    return Args {
      cutlass::gemm::GemmUniversalMode::kGemm,
      {options.m, options.n, options.k, options.l},
      {block_A.get(), stride_A, block_B_buff.device_data(), stride_B},
      {{options.alpha, options.beta}, block_C.get(), stride_C_ref, block_D.get(), stride_D_ref}
    };
  } 
  else if (options.mode == GemmMode::ScaleOnly) {
    return Args {
      cutlass::gemm::GemmUniversalMode::kGemm,
      {options.m, options.n, options.k, options.l},
      {block_A.get(), stride_A, block_B_buff.device_data(), stride_B, block_scale.get(), stride_S, options.g},
      {{options.alpha, options.beta}, block_C.get(), stride_C_ref, block_D.get(), stride_D_ref}
    };
  }
  else if (options.mode == GemmMode::ScaleWithZeroPoint) {
    return Args {
      cutlass::gemm::GemmUniversalMode::kGemm,
      {options.m, options.n, options.k, options.l},
      {block_A.get(), stride_A, block_B_buff.device_data(), stride_B, block_scale.get(), stride_S, options.g, block_zero.get()},
      {{options.alpha, options.beta}, block_C.get(), stride_C_ref, block_D.get(), stride_D_ref}
    };
  } else {
    std::cerr << "Invalid mode " << options.mode << ". Must be 0, 1 or 2." << std::endl;
    exit(-1);
  }
}

bool verify(const Options &options) {
  //
  // Compute reference output
  //

  // In this example, we use the PPU default kernels as a reference (unfused scale)
  // This avoids numerical differences due to different accumulation order.

  // Again, due to numerical differences, we must use fast acc here when the mma type is
  // FP8 as the fused implementation only supports fast acc at the moment.
  constexpr bool IsFP8Input = cute::is_same_v<MmaType, cutlass::float_e4m3_t> || cute::is_same_v<MmaType, cutlass::float_e5m2_t>;
  using FP8Sched = cute::conditional_t<size<0>(TileShape{}) == 64, cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum, cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8FastAccum>;
  using ScheduleRef = cute::conditional_t<IsFP8Input, FP8Sched, cutlass::gemm::collective::KernelScheduleAuto>;

  using CollectiveMainloopRef = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      MmaType, LayoutA, AlignmentA,
      MmaType, LayoutB, AlignmentB,
      ElementAccumulator,
      TileShape, WarpShape,
      cutlass::gemm::collective::StageCountAuto,
      ScheduleRef
    >::CollectiveOp;

  using CollectiveEpilogueRef = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, cutlass::arch::OpClassTensorOp,
      TileShape, WarpShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::NoSmemWarpSpecialized
    >::CollectiveOp;

  using GemmKernelRef = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>, // Indicates ProblemShape
      CollectiveMainloopRef,
      CollectiveEpilogueRef
  >;

  using GemmRef = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelRef>;

  typename GemmRef::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {options.m, options.n, options.k, options.l},
    {block_A.get(), stride_A, block_B_dq.get(), stride_B},
    {{options.alpha, options.beta}, block_C.get(), stride_C_ref, block_ref_D.get(), stride_D_ref}
  };

  // Run the gemm where the scaling is performed outside of the kernel.
  GemmRef gemm_ref;
  size_t workspace_size = GemmRef::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  CUTLASS_CHECK(gemm_ref.can_implement(arguments));
  CUTLASS_CHECK(gemm_ref.initialize(arguments, workspace.get()));
  CUTLASS_CHECK(gemm_ref.run());

  // compare_reference
  ElementD const epsilon(1e-2f);
  ElementD const non_zero_floor(1e-4f);
  bool passed = cutlass::reference::device::BlockCompareRelativelyEqual(block_ref_D.get(), block_D.get(), block_D.size(), epsilon, non_zero_floor);
  return passed;
}

/// Execute a given example GEMM computation
template <typename Gemm>
int run(Options &options)
{
  initialize(options);

  // Instantiate kernel depending on templates
  Gemm gemm;

  // Create a structure of gemm kernel arguments suitable for invoking an instance of Gemm
  auto arguments = args_from_options<typename Gemm::Arguments>(options);

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

  // Check if output from kernel and reference kernel are equal or not
  Result result;
  result.passed = verify(options);

  std::cout << "  Disposition: " << (result.passed ? "Passed" : "Failed") << std::endl;

  if (!result.passed) {
    exit(-1);
  }

  // Run profiling loop
  if (options.iterations > 0)
  {
    PpuTimer timer;
    timer.start();
    for (int iter = 0; iter < options.iterations; ++iter) {
      CUTLASS_CHECK(gemm.run());
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

  return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {
  hggcDeviceProp props;
  int current_device_id;
  CUTLASS_PPU_CHECK(hggcGetDevice(&current_device_id));
  CUTLASS_PPU_CHECK(hggcGetDeviceProperties(&props, current_device_id));
  hggcError_t error = hggcGetDeviceProperties(&props, 0);

  // should run on PPU 1.0
  if (props.major != 8 || props.minor != 0) {
    std::cerr << " This example should only be run on PPU 1.0!!! " << std::endl;
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

  if (options.mode == GemmMode::ConvertOnly) {
    std::cout << "PPU1.0 Running in no scale mode." << std::endl;
    run<GemmConvertOnly>(options);
  }
  else if (options.mode == GemmMode::ScaleOnly) {
    if (options.g == options.k) {
      std::cout << "PPU1.0 Running in per-column scale mode." << std::endl;
    } else {
      std::cout << "PPU1.0 Running in group scale mode." << std::endl;
    }
    run<GemmScaleOnly>(options);
  }
  else if (options.mode == GemmMode::ScaleWithZeroPoint) {
    if (options.g == options.k) {
      std::cout << "PPU1.0 Running in per-column scale and zero mode." << std::endl;
    } else {
      std::cout << "PPU1.0 Running in group scale and zero mode." << std::endl;
    }
    run<GemmScaleWithZeroPoint>(options);
  }

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
