/***************************************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD. All rights reserved. 
 * Copyright (c) 2023 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_fill.h"

#include "ppu_include.hpp"
#include "cutlass/gemm/config/gemm_operands.hpp"
#include "cutlass/gemm/config/gemm_configs.hpp"
#include "helper.h"

using namespace cute;

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Command line options parsing
struct Options {

  bool help;
  bool error;

  int m, n, k, l;
  float alpha, beta;

  Options():
    help(false),
    error(false),
    m(128), n(128), k(128), l(1),
    alpha(1.f), beta(0.f)
  { }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("m", m, 128);
    cmd.get_cmd_line_argument("n", n, 128);
    cmd.get_cmd_line_argument("k", k, 128);
    cmd.get_cmd_line_argument("l", l, 1);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "102_ppu0015_gemm_evt\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --m=<int>                   Sets the M extent of the GEMM\n"
      << "  --n=<int>                   Sets the N extent of the GEMM\n"
      << "  --k=<int>                   Sets the K extent of the GEMM\n"
      << "  --l=<int>                   Sets the L extent (batch count) of the GEMM\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n\n";

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
    scope_max = Element(2);
    scope_min = Element(0);
  } else if (bits_input <= 8) {
    scope_max = Element(2);
    scope_min = Element(-2);
  } else {
    scope_max = Element(8);
    scope_min = Element(-8);
  }

  cutlass::reference::device::BlockFillRandomUniform(
    block.get(), block.size(), seed, scope_max, scope_min, 0);

  return true;
}

template <typename Arch, cutlass::config::KernelScheduleType schedule_type,
          int BLOCK_M, int BLOCK_N,
          int WARP_M, int WARP_N,
          int BLOCK_K,
          int STAGE, int ALIGN_CD>
struct ExampleRunner {

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using ElementD = cutlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute = float;
  using ElementScalar = float;

  static constexpr int AlignmentA = 1;
  static constexpr int AlignmentB = 1;
  static constexpr int AlignmentC = ALIGN_CD;
  static constexpr int AlignmentD = ALIGN_CD;


  using CustomEVT = cutlass::epilogue::fusion::PPUEVT<
    cutlass::epilogue::fusion::PPUCompute<cutlass::homogeneous_multiply_add, ElementD, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>,
    cutlass::epilogue::fusion::PPUScalarBroadcast<ElementScalar>,
    cutlass::epilogue::fusion::PPUSrcFetch<ElementC>,
    cutlass::epilogue::fusion::PPUEVT<
      cutlass::epilogue::fusion::PPUCompute<cutlass::multiplies, ElementCompute, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>,
      cutlass::epilogue::fusion::PPUScalarBroadcast<ElementScalar>,
      cutlass::epilogue::fusion::PPUAccFetch
    >
  >;

  using GemmKernel = typename cutlass::gemm::config::GemmKernelConfig<
    Arch,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator, ElementC, AlignmentC,
    ElementD, AlignmentD, ElementA,
    BLOCK_M, BLOCK_N, BLOCK_K,
    WARP_M, WARP_N, BLOCK_K,
    STAGE,
    CustomEVT,
    true,
    false,
    true,
    schedule_type,
    cutlass::arch::OpClassTensorOp
  >::GemmKernel;

  using ProblemShapeType = typename GemmKernel::ProblemShape;

  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideC = typename GemmKernel::StrideC;
  using StrideD = typename GemmKernel::StrideD;

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

  cutlass::DeviceAllocation<typename GemmKernel::ElementA> block_A;
  cutlass::DeviceAllocation<typename GemmKernel::ElementB> block_B;
  cutlass::DeviceAllocation<typename GemmKernel::ElementC> block_C;
  cutlass::DeviceAllocation<typename GemmKernel::ElementD> block_D;
  cutlass::DeviceAllocation<typename GemmKernel::ElementD> block_ref_D;

  //
  // Methods
  //

  bool verify(const ProblemShapeType& problem_size, float alpha, float beta) {
    auto [M, N, K, L] = problem_size;

    cutlass::TensorRef ref_A(block_A.get(), LayoutTagA::packed({M, K}));
    cutlass::TensorRef ref_B(block_B.get(), LayoutTagB::packed({K, N}));
    cutlass::TensorRef ref_C(block_C.get(), LayoutTagC::packed({M, N}));
    cutlass::TensorRef ref_D(block_ref_D.get(), LayoutTagD::packed({M, N}));

    cutlass::reference::device::GemmComplex(
          {M, N, K},
          ElementScalar(alpha),
          ref_A,
          cutlass::ComplexTransform::kNone,
          ref_B,
          cutlass::ComplexTransform::kNone,
          ElementScalar(beta),
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

    // ElementD *ref = new ElementD[M*N];
    // ElementD *res = new ElementD[M*N];

    // hggcMemcpy(ref, block_ref_D.get(), M*N*sizeof(ElementD), hggcMemcpyDeviceToHost);
    // hggcMemcpy(res, block_D.get(), M*N*sizeof(ElementD), hggcMemcpyDeviceToHost);

    // for (int i = 0; i < 10; ++i) {
    //   // float *ref = reinterpret_cast<float*>(block_ref_D.get());
    //   // float *res = reinterpret_cast<float*>(block_D.get());
    //   printf("ref: %f, res: %f\n", float(ref[i]), float(res[i]));
    // }

    // delete[] ref;
    // delete[] res;

    return passed;
  }

  /// Initialize operands to be used in the GEMM and reference GEMM
  void initialize(const ProblemShapeType& problem_size) {
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

    initialize_block(block_A, seed + 2023);
    initialize_block(block_B, seed + 2022);
    initialize_block(block_C, seed + 2021);
  }

  template <bool static_scheduler = false>
  bool run(const Options& options, cutlass::KernelHardwareInfo& hw_info, int blocks_per_cu = 1) {
    hw_info.cu_count *= blocks_per_cu;
    ProblemShapeType problem_size = ProblemShapeType{options.m, options.n, options.k, options.l};

    initialize(problem_size);

    typename GemmKernel::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size,
      {block_A.get(), stride_A, block_B.get(), stride_B},
      {
       {}, // epilogue.thread
       block_C.get(), stride_C, block_D.get(), stride_D
      },
      hw_info
    };
    if constexpr (static_scheduler) {
      arguments.scheduler.tb_per_cu = blocks_per_cu;
    }
    
    // evt realization must construct evt params on device, can't use GemmUniversalAdapter
    typename GemmKernel::Params params = GemmKernel::to_underlying_arguments(arguments, nullptr);

    params.epilogue.thread =
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

    size_t workspace_size = GemmKernel::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    bool status = GemmKernel::can_implement(arguments);
    if (status == false) {
      std::cerr << "This kernel is not supported. Last device error is: "
                << hggcGetErrorString(hggcGetLastError()) << std::endl;
      return false;
    }

    dim3 const block = GemmKernel::get_block_shape();
    dim3 const grid = GemmKernel::get_grid_shape(params);
    int smem_size = GemmKernel::SharedStorageSize;

    cutlass::device_kernel<GemmKernel><<<grid, block, smem_size>>>(params);

    hggcError_t result = hggcDeviceSynchronize();
    if (result != hggcSuccess) {
      std::cerr << "Error running the kernel. Last device error is: "
                << hggcGetErrorString(result) << std::endl;
      return false;
    }

    // Verify that the result is correct
    bool passed = verify(problem_size, options.alpha, options.beta);
    if (!passed) {
      std::cerr << "Reference check failed" << std::endl;
    }

    return passed;
  }

};

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to print a description of the example run and its result
void print_result(const std::string& description, bool passed) {
  std::cout << description << ": " << (passed ? "Passed" : "Failed") << std::endl;
  if (false == passed) {
    exit(-1);
  }
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

  if (options.error) {
    std::cerr << "Aborting execution." << std::endl;
    return -1;
  }

  //
  // Run examples
  //

  // The KernelHardwareInfo struct holds the number of CUs on the PPU with a given device ID. This
  // information is used by the underlying kernel.
  cutlass::KernelHardwareInfo hw_info;

  // Change device_id to another value if you are running on a machine with multiple PPUs and wish
  // to use a PPU other than that with device ID 0.
  hw_info.device_id = 0;
  hw_info.cu_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  bool passed;

  // Here, we override the fusion operation to use a customized EVT fusion, in addition to the previous schedule overrides
  static constexpr cutlass::config::KernelScheduleType gemm_with_overlap_prologue 
      = cutlass::config::KernelScheduleType(int(cutlass::config::KernelScheduleType::PERSISTENT_OVERLAP_PROEPILOGUE));
  // --m=20488 --n=20488 --k=1024 --l=1
  ExampleRunner<cutlass::arch::PPU0015, gemm_with_overlap_prologue, 256, 256, 64, 64, 64, 4, 4> overlap_runner_big;
  passed = overlap_runner_big.run<true>(options, hw_info, 1);
  print_result("aiu gemm evt with overlap for big tile", passed);

  // --m=40960 --n=40960 --k=1024 --l=1
  ExampleRunner<cutlass::arch::PPU0015, gemm_with_overlap_prologue, 128, 128, 64, 32, 32, 4, 4> overlap_runner_med;
  passed = overlap_runner_med.run<true>(options, hw_info, 4);
  print_result("aiu gemm evt with overlap for medium tile", passed);

  static constexpr cutlass::config::KernelScheduleType gemm_with_default 
      = cutlass::config::KernelScheduleType(int(cutlass::config::KernelScheduleType::DEFAULT));
  // --m=40960 --n=40960 --k=1024 --l=1
  ExampleRunner<cutlass::arch::PPU0015, gemm_with_default, 128, 128, 64, 32, 32, 4, 4> dft_runner;
  passed = dft_runner.run(options, hw_info, 4);
  print_result("aiu gemm evt", passed);

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
