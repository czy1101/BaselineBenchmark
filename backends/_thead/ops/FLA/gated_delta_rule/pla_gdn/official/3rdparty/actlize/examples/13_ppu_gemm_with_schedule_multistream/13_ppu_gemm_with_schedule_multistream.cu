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

/*** multistream persistent tfu & valu gemm example ***/

#include <iostream>
#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_fill.h"

#include "helper.h"

#include "cutlass/gemm/config/gemm_operands.hpp"
#include "cutlass/gemm/config/gemm_configs.hpp"

using namespace cute;

struct Options {

  bool help;
  bool error;

  int m, n, k, l;
  float alpha, beta;
  bool run_golden;

  Options():
    help(false),
    error(false),
    m(2048), n(2048), k(2048), l(1),
    alpha(1.f), beta(0.f),
    run_golden(false)
  { }

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
    cmd.get_cmd_line_argument("l", l, 1);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("golden", run_golden, false);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "13_ppu_gemm_with_schedule_multistream\n\n"
      << "./13_ppu_gemm_with_schedule_multistream --m=4096 --n=4096 --k=4096\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --m=<int>                   Sets the M extent of the GEMM\n"
      << "  --n=<int>                   Sets the N extent of the GEMM\n"
      << "  --k=<int>                   Sets the K extent of the GEMM\n"
      << "  --l=<int>                   Sets the L extent (batch count) of the GEMM\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n\n"
      << "  --golden=<bool>             Run golden or not\n\n";

    return out;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to initialize a block of device data
template <class Element>
bool initialize_block(
  cutlass::DeviceAllocation<Element>& block,
  uint64_t seed=2023) {

  Element scope_max, scope_min;
  int bits_input = cutlass::sizeof_bits<Element>::value;

  if (bits_input == 1) {
    scope_max = 2;
    scope_min = 0;
  } else if (bits_input <= 8) {
    scope_max = 2;
    scope_min = -2;
  } else {
    scope_max = 8;
    scope_min = -8;
  }

  cutlass::reference::device::BlockFillRandomUniform(
    block.get(), block.size(), seed, scope_max, scope_min, 0);

  return true;
}

using cutlass::config::KernelScheduleType;

template <
  typename Arch,
  typename ElementA,
  typename ElementB,
  typename ElementC,
  // Type of epilogue schedule to generate
  class EpilogueScheduleType = cutlass::epilogue::collective::EpilogueScheduleAuto,
  // Number of pipeline stages to use
  int Stage = 2,
  // Type of tile scheduler to use
  KernelScheduleType schedule = cutlass::config::KernelScheduleType::DEFAULT,
  // Do we use custom epilogue visitor tree (EVT) fusion
  bool UseCustomEVT = false,
  bool UseAIU = true,
  bool ParallelSplitk = false,
  int max_swizzle_size = 1,
  class OperatorClass = cutlass::arch::OpClassTensorOp,
  typename LayoutA = cutlass::layout::RowMajor,
  typename LayoutB = cutlass::layout::ColumnMajor
>
struct ExampleRunner {

  using LayoutC = cutlass::layout::ColumnMajor;
  using LayoutD = cutlass::layout::ColumnMajor;
  using ElementD = ElementC;
  using ElementAccumulator = ElementC;
  using ElementCompute = ElementC;
  using ElementScalar = ElementC;

  // 16B alignment lets us use TMA
  static constexpr int AlignmentA = UseAIU ? 1 : 16 / sizeof(ElementA);
  static constexpr int AlignmentB = UseAIU ? 1 : 16 / sizeof(ElementB);
  static constexpr int AlignmentC = 16 / sizeof(ElementC);
  static constexpr int AlignmentD = 16 / sizeof(ElementD);

  static_assert(not UseCustomEVT ||
    (cute::is_same_v<EpilogueScheduleType, cutlass::epilogue::TmaWarpSpecialized> ||
     cute::is_same_v<EpilogueScheduleType, cutlass::epilogue::TmaWarpSpecializedCooperative>),
    "Epilogue visitor trees are currently only supported by the TMA warp-specialized epilogue");
  static constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;

  // EVTs can be constructed by composing the fundamental load/store/compute visitor operations defined in include/cutlass/epilogue/fusion
  using CustomEVT =  // alpha * acc + beta * C
    cutlass::epilogue::fusion::PPUEVT<cutlass::epilogue::fusion::PPUCompute<cutlass::homogeneous_multiply_add, ElementD, ElementCompute, RoundStyle>, // beta * C + (alpha * acc)
      cutlass::epilogue::fusion::PPUScalarBroadcast<ElementScalar>, // beta
      cutlass::epilogue::fusion::PPUSrcFetch<ElementC>, // C
      cutlass::epilogue::fusion::PPUEVT<cutlass::epilogue::fusion::PPUCompute<cutlass::multiplies, ElementCompute, ElementCompute, RoundStyle>, // alpha * acc
        cutlass::epilogue::fusion::PPUScalarBroadcast<ElementScalar>, // alpha
        cutlass::epilogue::fusion::PPUAccFetch // acc
      >
    >;

  // A predefined set of fusion operations (implemented with EVT) are supported by the TMA warp-specialized epilogue.
  // Users can select one of these operations by passing one of the tags defined in include/cutlass/epilogue/fusion/operations.hpp
  // to the CollectiveBuilder. This frees the user from having to compute additional parameters such as stage counts and copy atoms/layouts.
  // These tags also provide additional metadata that can be queried at compile time.
  // using DefaultOperation = cutlass::epilogue::fusion::LinearCombination<ElementD, ElementCompute, ElementC, ElementScalar, RoundStyle>;
  using DefaultOperation = cutlass::epilogue::thread::LinearCombination<ElementD, AlignmentD, ElementAccumulator, ElementCompute>;

  static constexpr int BlockM = 128;
  static constexpr int BlockN = 128;
  static constexpr int BlockK = UseAIU ? 64 : 32;
  static constexpr int WarpM = 64;
  static constexpr int WarpN = 64;
  static constexpr int WarpK = UseAIU ? 64 : 32;
  using GemmKernel = typename cutlass::gemm::config::GemmKernelConfig<
    Arch,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator, ElementD, AlignmentD,
    ElementD, AlignmentD, ElementA,
    BlockM, BlockN, BlockK,
    WarpM, WarpN, WarpK,
    Stage,
    cute::conditional_t<UseCustomEVT, CustomEVT, DefaultOperation>,
    UseAIU,
    ParallelSplitk, // ParallelSplitk
    UseCustomEVT,
    schedule,
    OperatorClass
  >::GemmKernel;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  using LayoutTagA = cutlass::gemm::detail::StrideToLayoutTagA_t<StrideA>;
  using LayoutTagB = cutlass::gemm::detail::StrideToLayoutTagB_t<StrideB>;
  using LayoutTagC = cutlass::gemm::detail::StrideToLayoutTagC_t<StrideC>;
  using LayoutTagD = cutlass::gemm::detail::StrideToLayoutTagC_t<StrideD>;

  //
  // Data members
  //

  /// Initialization
  StrideA stride_A;
  StrideB stride_B;
  StrideC stride_C;
  StrideD stride_D;
  uint64_t seed = 0;

  cutlass::DeviceAllocation<typename Gemm::ElementA> block_A;
  cutlass::DeviceAllocation<typename Gemm::ElementB> block_B;
  cutlass::DeviceAllocation<typename Gemm::ElementC> block_C;
  cutlass::DeviceAllocation<typename Gemm::ElementD> block_D;
  cutlass::DeviceAllocation<typename Gemm::ElementD> block_ref_D;


  Gemm gemm_op;

  //
  // Methods
  //
  /// Initialize operands to be used in the GEMM and reference GEMM
  void initialize(const ProblemShapeType& problem_size, const Options& options) {
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    auto [M, N, K, L] = problem_shape_MNKL;

    stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));
    stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, L));
    stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, L));
    stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, L));

    block_A.reset(M * K * L);
    block_B.reset(K * N * L);
    block_C.reset(M * N * L);
    block_D.reset(M * N * L);
    block_ref_D.reset(M * N * L);

    if (options.run_golden) {
      initialize_block(block_A, seed + 2023);
      initialize_block(block_B, seed + 2022);
      initialize_block(block_C, seed + 2021);
    }
  }

  bool init(const Options& options, const cutlass::KernelHardwareInfo& hw_info, hggcStream_t& stream) {
    ProblemShapeType problem_size = ProblemShapeType{options.m, options.n, options.k, options.l};

    initialize(problem_size, options);

    using TileScheduler = typename GemmKernel::TileScheduler;
    auto scheduler = typename TileScheduler::Arguments{};
    scheduler.max_swizzle_size = max_swizzle_size;
    typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size,
      {block_A.get(), stride_A, block_B.get(), stride_B},
      {{}, // epilogue.thread
       block_C.get(), stride_C, block_D.get(), stride_D},
      hw_info,
      scheduler
    };

    // Custom EVT fusions will have nested unnamed args, the structure of which
    // can be deduced from the type definition of the EVT.
    // Each node's arguments has the recursive structure of
    // {first_child_args, ..., last_child_args, op_args},
    // For more complex examples of EVT initialization please refer to
    if constexpr (UseCustomEVT) {
      arguments.epilogue.thread =
        {    // ternary op : beta * C + (alpha * acc)
          {{options.beta}}, // leaf op+args : beta
          {},               // leaf op+args : C
          {                 // binary op : alpha * acc
            {{options.alpha}}, // leaf op+args : alpha
            {},                // leaf op+args : acc
            {}              // binary args : multiplies
          },                // end binary op
          {} // ternary args : multiply_add
        };   // end ternary op
    }
    // Pre-defined fusions will have flat, named args for user-friendlyness
    else {
      arguments.epilogue.thread.alpha = options.alpha;
      arguments.epilogue.thread.beta = options.beta;
    }

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "This kernel is not supported. Last device error is: "
                << hggcGetErrorString(hggcGetLastError()) << std::endl;
      return false;
    }

    status = gemm_op.initialize(arguments, workspace.get(), stream);
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Failed to initialize the kernel. Last device error is: "
                << hggcGetErrorString(hggcGetLastError()) << std::endl;
      return false;
    }

    return true;
  }

  void run(hggcStream_t& stream) {
    // Run the GEMM
    gemm_op.run(stream);
    return;
  }

  bool verify(const Options& options) {
    ProblemShapeType problem_size = ProblemShapeType{options.m, options.n, options.k, options.l};
    auto [M, N, K, L] = problem_size;

    cutlass::TensorRef ref_A(block_A.get(), Gemm::LayoutA::packed({M, K}));
    cutlass::TensorRef ref_B(block_B.get(), Gemm::LayoutB::packed({K, N}));
    cutlass::TensorRef ref_C(block_C.get(), Gemm::LayoutC::packed({M, N}));
    cutlass::TensorRef ref_D(block_ref_D.get(), Gemm::LayoutD::packed({M, N}));

    cutlass::reference::device::GemmComplex(
          {M, N, K},
          ElementScalar(options.alpha),
          ref_A,
          cutlass::ComplexTransform::kNone,
          ref_B,
          cutlass::ComplexTransform::kNone,
          ElementScalar(options.beta),
          ref_C,
          ref_D,
          ElementAccumulator(0),
          L,     // batch_count
          M * K, // batch_stride_A
          K * N, // batch_stride_B
          M * N, // batch_stride_C
          M * N  // batch_stride_D
        );

    hggcError_t result = hggcDeviceSynchronize();
    if (result != hggcSuccess) {
      std::cerr << "Reference kernel failed. Last device error: "
                << hggcGetErrorString(result) << std::endl;
      return false;
    }

    // Check if output from kernel and reference kernel are equal or not
    bool passed = cutlass::reference::device::BlockCompareEqual(block_ref_D.get(), block_D.get(), block_D.size());

    return passed;
  }

};


///////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to print a description of the example run and its result
void print_result(const std::string& description, bool passed) {
  std::cout << description << ": " << (passed ? "Passed" : "Failed") << std::endl;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {
  hggcDeviceProp props;
  int current_device_id;
  CUTLASS_PPU_CHECK(hggcGetDevice(&current_device_id));
  CUTLASS_PPU_CHECK(hggcGetDeviceProperties(&props, current_device_id));
  hggcError_t error = hggcGetDeviceProperties(&props, 0);

  // should run on PPU 1.0/1.5
  if (props.major != 8) {
    std::cerr << " This example should be run on PPU 1.0/1.5!!! " << std::endl;
    return 0;
  }


  Options tfu_options;   // tfu use default options
  Options valu_options;
  valu_options.parse(argc, args);
  if (valu_options.help) {
    valu_options.print_usage(std::cout) << std::endl;
    return 0;
  }
  if (valu_options.error) {
    std::cerr << "Aborting execution." << std::endl;
    return -1;
  }
  cutlass::KernelHardwareInfo tfu_hw_info;
  tfu_hw_info.device_id = 0;
  cutlass::KernelHardwareInfo valu_hw_info;
  valu_hw_info.device_id = 0;

  hggcStream_t tfu_stream, valu_stream;
  hggcStreamCreate(&tfu_stream);
  hggcStreamCreate(&valu_stream);

  PpuTimer timer;
  double cute_time;

  if (props.minor == 0) {
    /*** multistream persistent tfu & valu gemm example ***/
    ExampleRunner<
      cutlass::arch::PPU0010,
      cutlass::half_t,   // ElementA
      cutlass::half_t,   // ElementB
      float,             // ElementC
      cutlass::epilogue::TmaWarpSpecialized,
      2,
      KernelScheduleType::PERSISTENT,
      false,
      true,
      false,
      8> tfu_swizzle8_schedule_stage2_runner;
    tfu_hw_info.cu_count = 20;
    std::cout << "tfu_hw_info.cu_count: " << tfu_hw_info.cu_count << std::endl;

    ExampleRunner<
      cutlass::arch::PPU0010,
      cutlass::half_t,   // ElementA
      cutlass::half_t,   // ElementB
      float,             // ElementC
      cutlass::epilogue::TmaWarpSpecialized,
      3,
      KernelScheduleType::PERSISTENT,
      false,
      false,
      false,
      8,
      cutlass::arch::OpClassSimt
    > valu_swizzle8_schedule_stage3_runner;
    valu_hw_info.cu_count = 44;
    std::cout << "valu_hw_info.cu_count: " << valu_hw_info.cu_count << std::endl;

    tfu_swizzle8_schedule_stage2_runner.init(tfu_options, tfu_hw_info, tfu_stream);
    valu_swizzle8_schedule_stage3_runner.init(valu_options, valu_hw_info, valu_stream);

    // warmup
    tfu_swizzle8_schedule_stage2_runner.run(tfu_stream);
    valu_swizzle8_schedule_stage3_runner.run(valu_stream);
    hggcDeviceSynchronize();

    timer.start();
    tfu_swizzle8_schedule_stage2_runner.run(tfu_stream);
    valu_swizzle8_schedule_stage3_runner.run(valu_stream);
    hggcDeviceSynchronize();
    timer.stop();
    cute_time = double(timer.elapsed_millis());
    std::cout << "multi-stream gemm elapsed time: " << cute_time << " ms" << std::endl;


    valu_swizzle8_schedule_stage3_runner.init(valu_options, valu_hw_info, tfu_stream);
    timer.start();
    tfu_swizzle8_schedule_stage2_runner.run(tfu_stream);
    valu_swizzle8_schedule_stage3_runner.run(tfu_stream);
    hggcDeviceSynchronize();
    timer.stop();
    cute_time = double(timer.elapsed_millis());
    std::cout << "single-stream gemm elapsed time: " << cute_time << " ms" << std::endl;


    /*** persistent simt gemm function & performance test ***/
    using Layout_A = cutlass::layout::ColumnMajor;
    using Layout_B = cutlass::layout::ColumnMajor;
    ExampleRunner<
      cutlass::arch::PPU0010,
      float,             // ElementA
      float,             // ElementB
      float,             // ElementC
      cutlass::epilogue::TmaWarpSpecialized,
      3,
      KernelScheduleType::PERSISTENT,
      false,
      false,
      false,
      8,
      cutlass::arch::OpClassSimt,
      Layout_A,
      Layout_B
    > fp32_valu_swizzle8_schedule_stage3_runner;
    valu_hw_info.cu_count = 2 * cutlass::KernelHardwareInfo::query_device_multiprocessor_count(valu_hw_info.device_id);
    std::cout << "valu_hw_info.cu_count: " << valu_hw_info.cu_count << std::endl;

    fp32_valu_swizzle8_schedule_stage3_runner.init(valu_options, valu_hw_info, valu_stream);
    timer.start();
    fp32_valu_swizzle8_schedule_stage3_runner.run(valu_stream);
    hggcDeviceSynchronize();
    timer.stop();
    cute_time = double(timer.elapsed_millis());
    std::cout << "FP32 VALU TB_SWIZZLE_8 schedule with stage 3 elapsed time: " << cute_time << " ms" << std::endl;
    if (valu_options.run_golden) {
      printf("fp32 valu goldennnnnn\n");
      bool passed = fp32_valu_swizzle8_schedule_stage3_runner.verify(valu_options);
      print_result("FP32 VALU TB_SWIZZLE_8 schedule with stage 3", passed);
    }


    ExampleRunner<
      cutlass::arch::PPU0010,
      cutlass::half_t,   // ElementA
      cutlass::half_t,   // ElementB
      float,             // ElementC
      cutlass::epilogue::TmaWarpSpecialized,
      3,
      KernelScheduleType::PERSISTENT,
      false,
      false,
      false,
      8,
      cutlass::arch::OpClassSimt,
      Layout_A,
      Layout_B
    > mixed_valu_swizzle8_schedule_stage3_runner;
    valu_hw_info.cu_count = 2 * cutlass::KernelHardwareInfo::query_device_multiprocessor_count(valu_hw_info.device_id);
    std::cout << "valu_hw_info.cu_count: " << valu_hw_info.cu_count << std::endl;

    mixed_valu_swizzle8_schedule_stage3_runner.init(valu_options, valu_hw_info, valu_stream);
    timer.start();
    mixed_valu_swizzle8_schedule_stage3_runner.run(valu_stream);
    hggcDeviceSynchronize();
    timer.stop();
    cute_time = double(timer.elapsed_millis());
    std::cout << "MIXED VALU TB_SWIZZLE_8 schedule with stage 3 elapsed time: " << cute_time << " ms" << std::endl;
    if (valu_options.run_golden) {
      printf("mixed valu goldennnnnn\n");
      bool passed = mixed_valu_swizzle8_schedule_stage3_runner.verify(valu_options);
      print_result("MIXED VALU TB_SWIZZLE_8 schedule with stage 3", passed);
    }

    ExampleRunner<
      cutlass::arch::PPU0010,
      cutlass::half_t,   // ElementA
      cutlass::half_t,   // ElementB
      cutlass::half_t,   // ElementC
      cutlass::epilogue::TmaWarpSpecialized,
      3,
      KernelScheduleType::PERSISTENT,
      false,
      false,
      false,
      8,
      cutlass::arch::OpClassSimt,
      Layout_A,
      Layout_B
    > fp16_valu_swizzle8_schedule_stage3_runner;
    valu_hw_info.cu_count = 3 * cutlass::KernelHardwareInfo::query_device_multiprocessor_count(valu_hw_info.device_id);
    std::cout << "valu_hw_info.cu_count: " << valu_hw_info.cu_count << std::endl;

    fp16_valu_swizzle8_schedule_stage3_runner.init(valu_options, valu_hw_info, valu_stream);
    timer.start();
    fp16_valu_swizzle8_schedule_stage3_runner.run(valu_stream);
    hggcDeviceSynchronize();
    timer.stop();
    cute_time = double(timer.elapsed_millis());
    std::cout << "FP16 VALU TB_SWIZZLE_8 schedule with stage 3 elapsed time: " << cute_time << " ms" << std::endl;

    if (valu_options.run_golden) {
      printf("fp16 valu goldennnnnn\n");
      bool passed = fp16_valu_swizzle8_schedule_stage3_runner.verify(valu_options);
      print_result("FP16 VALU TB_SWIZZLE_8 schedule with stage 3", passed);
    }

  } else if(props.minor == 9) {
        ExampleRunner<
      cutlass::arch::PPU0015,
      cutlass::half_t,   // ElementA
      cutlass::half_t,   // ElementB
      float,             // ElementC
      cutlass::epilogue::TmaWarpSpecialized,
      2,
      KernelScheduleType::PERSISTENT,
      false,
      true,
      false,
      8> tfu_swizzle8_schedule_stage2_runner;
    tfu_hw_info.cu_count = 20;
    std::cout << "tfu_hw_info.cu_count: " << tfu_hw_info.cu_count << std::endl;

    ExampleRunner<
      cutlass::arch::PPU0015,
      cutlass::half_t,   // ElementA
      cutlass::half_t,   // ElementB
      float,             // ElementC
      cutlass::epilogue::TmaWarpSpecialized,
      3,
      KernelScheduleType::PERSISTENT,
      false,
      false,
      false,
      8,
      cutlass::arch::OpClassSimt
    > valu_swizzle8_schedule_stage3_runner;
    valu_hw_info.cu_count = 44;
    std::cout << "valu_hw_info.cu_count: " << valu_hw_info.cu_count << std::endl;

    tfu_swizzle8_schedule_stage2_runner.init(tfu_options, tfu_hw_info, tfu_stream);
    valu_swizzle8_schedule_stage3_runner.init(valu_options, valu_hw_info, valu_stream);

    // warmup
    tfu_swizzle8_schedule_stage2_runner.run(tfu_stream);
    valu_swizzle8_schedule_stage3_runner.run(valu_stream);
    hggcDeviceSynchronize();

    timer.start();
    tfu_swizzle8_schedule_stage2_runner.run(tfu_stream);
    valu_swizzle8_schedule_stage3_runner.run(valu_stream);
    hggcDeviceSynchronize();
    timer.stop();
    cute_time = double(timer.elapsed_millis());
    std::cout << "multi-stream gemm elapsed time: " << cute_time << " ms" << std::endl;


    valu_swizzle8_schedule_stage3_runner.init(valu_options, valu_hw_info, tfu_stream);
    timer.start();
    tfu_swizzle8_schedule_stage2_runner.run(tfu_stream);
    valu_swizzle8_schedule_stage3_runner.run(tfu_stream);
    hggcDeviceSynchronize();
    timer.stop();
    cute_time = double(timer.elapsed_millis());
    std::cout << "single-stream gemm elapsed time: " << cute_time << " ms" << std::endl;


    /*** persistent simt gemm function & performance test ***/
    using Layout_A = cutlass::layout::ColumnMajor;
    using Layout_B = cutlass::layout::ColumnMajor;
    ExampleRunner<
      cutlass::arch::PPU0015,
      float,             // ElementA
      float,             // ElementB
      float,             // ElementC
      cutlass::epilogue::TmaWarpSpecialized,
      3,
      KernelScheduleType::PERSISTENT,
      false,
      false,
      false,
      8,
      cutlass::arch::OpClassSimt,
      Layout_A,
      Layout_B
    > fp32_valu_swizzle8_schedule_stage3_runner;
    valu_hw_info.cu_count = 2 * cutlass::KernelHardwareInfo::query_device_multiprocessor_count(valu_hw_info.device_id);
    std::cout << "valu_hw_info.cu_count: " << valu_hw_info.cu_count << std::endl;

    fp32_valu_swizzle8_schedule_stage3_runner.init(valu_options, valu_hw_info, valu_stream);
    timer.start();
    fp32_valu_swizzle8_schedule_stage3_runner.run(valu_stream);
    hggcDeviceSynchronize();
    timer.stop();
    cute_time = double(timer.elapsed_millis());
    std::cout << "FP32 VALU TB_SWIZZLE_8 schedule with stage 3 elapsed time: " << cute_time << " ms" << std::endl;
    if (valu_options.run_golden) {
      printf("fp32 valu goldennnnnn\n");
      bool passed = fp32_valu_swizzle8_schedule_stage3_runner.verify(valu_options);
      print_result("FP32 VALU TB_SWIZZLE_8 schedule with stage 3", passed);
    }


    ExampleRunner<
      cutlass::arch::PPU0015,
      cutlass::half_t,   // ElementA
      cutlass::half_t,   // ElementB
      float,             // ElementC
      cutlass::epilogue::TmaWarpSpecialized,
      3,
      KernelScheduleType::PERSISTENT,
      false,
      false,
      false,
      8,
      cutlass::arch::OpClassSimt,
      Layout_A,
      Layout_B
    > mixed_valu_swizzle8_schedule_stage3_runner;
    valu_hw_info.cu_count = 2 * cutlass::KernelHardwareInfo::query_device_multiprocessor_count(valu_hw_info.device_id);
    std::cout << "valu_hw_info.cu_count: " << valu_hw_info.cu_count << std::endl;

    mixed_valu_swizzle8_schedule_stage3_runner.init(valu_options, valu_hw_info, valu_stream);
    timer.start();
    mixed_valu_swizzle8_schedule_stage3_runner.run(valu_stream);
    hggcDeviceSynchronize();
    timer.stop();
    cute_time = double(timer.elapsed_millis());
    std::cout << "MIXED VALU TB_SWIZZLE_8 schedule with stage 3 elapsed time: " << cute_time << " ms" << std::endl;
    if (valu_options.run_golden) {
      printf("mixed valu goldennnnnn\n");
      bool passed = mixed_valu_swizzle8_schedule_stage3_runner.verify(valu_options);
      print_result("MIXED VALU TB_SWIZZLE_8 schedule with stage 3", passed);
    }

    ExampleRunner<
      cutlass::arch::PPU0015,
      cutlass::half_t,   // ElementA
      cutlass::half_t,   // ElementB
      cutlass::half_t,   // ElementC
      cutlass::epilogue::TmaWarpSpecialized,
      3,
      KernelScheduleType::PERSISTENT,
      false,
      false,
      false,
      8,
      cutlass::arch::OpClassSimt,
      Layout_A,
      Layout_B
    > fp16_valu_swizzle8_schedule_stage3_runner;
    valu_hw_info.cu_count = 3 * cutlass::KernelHardwareInfo::query_device_multiprocessor_count(valu_hw_info.device_id);
    std::cout << "valu_hw_info.cu_count: " << valu_hw_info.cu_count << std::endl;

    fp16_valu_swizzle8_schedule_stage3_runner.init(valu_options, valu_hw_info, valu_stream);
    timer.start();
    fp16_valu_swizzle8_schedule_stage3_runner.run(valu_stream);
    hggcDeviceSynchronize();
    timer.stop();
    cute_time = double(timer.elapsed_millis());
    std::cout << "FP16 VALU TB_SWIZZLE_8 schedule with stage 3 elapsed time: " << cute_time << " ms" << std::endl;

    if (valu_options.run_golden) {
      printf("fp16 valu goldennnnnn\n");
      bool passed = fp16_valu_swizzle8_schedule_stage3_runner.verify(valu_options);
      print_result("FP16 VALU TB_SWIZZLE_8 schedule with stage 3", passed);
    }
  } else {
    std::cerr << "Unknown PPU arch !!!" << std::endl;
  }

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
