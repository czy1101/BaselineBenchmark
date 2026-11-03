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
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_fill.h"

#include "helper.h"

#include "ppu_include.hpp"
#include "cutlass/gemm/config/gemm_operands.hpp"
#include "cutlass/gemm/config/gemm_configs.hpp"

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

    out << "103_ppu0015_gemm_array\n\n"
      << "  This example showcases the use of collective operation builders to easily construct\n"
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

template <typename Arch, cutlass::config::KernelScheduleType Kernel_Schedule_Type, int STAGE, int BLOCK_M, int BLOCK_N, int BLOCK_K, int WARP_M, int WARP_N, int WARP_K>
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
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<ElementD,ElementAccumulator,ElementC,ElementAccumulator>;
  using GemmKernel = typename cutlass::gemm::config::GemmKernelConfig<
    Arch,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator, ElementC, AlignmentC,
    ElementD, AlignmentD, ElementA,
    BLOCK_M, BLOCK_N, BLOCK_K,
    WARP_M, WARP_N, WARP_K,
    STAGE,
    EpilogueOp,
    true,
    false,
    true,
    Kernel_Schedule_Type,
    cutlass::arch::OpClassTensorOp
  >::GemmKernel;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using ProblemShapeType = typename GemmKernel::ProblemShape;

  // Reference device GEMM implementation type
  using DeviceGemmReference = cutlass::reference::device::Gemm<
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  ElementAccumulator>;

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

  std::vector<int64_t> offset_A;
  std::vector<int64_t> offset_B;
  std::vector<int64_t> offset_C;
  std::vector<int64_t> offset_D;

  cutlass::DeviceAllocation<typename GemmKernel::ElementA> block_A;
  cutlass::DeviceAllocation<typename GemmKernel::ElementB> block_B;
  cutlass::DeviceAllocation<typename GemmKernel::ElementC> block_C;
  cutlass::DeviceAllocation<typename GemmKernel::ElementD> block_D;
  cutlass::DeviceAllocation<typename GemmKernel::ElementD> block_ref_D;

  cutlass::DeviceAllocation<const typename Gemm::ElementA *> ptr_A;
  cutlass::DeviceAllocation<const typename Gemm::ElementB *> ptr_B;
  cutlass::DeviceAllocation<const typename Gemm::ElementC *> ptr_C;
  cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput *> ptr_D;
  cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput *> ptr_ref_D;

  //
  // Methods
  //
  bool verify(const Options &options) {
    bool passed = true;
    for (int32_t i = 0; i < options.l; ++i) {
      cutlass::TensorRef ref_A(block_A.get() + offset_A.at(i), Gemm::LayoutA::packed({options.m, options.k}));
      cutlass::TensorRef ref_B(block_B.get() + offset_B.at(i), Gemm::LayoutB::packed({options.k, options.n}));
      cutlass::TensorRef ref_C(block_C.get() + offset_C.at(i), Gemm::LayoutC::packed({options.m, options.n}));
      cutlass::TensorRef ref_D(block_ref_D.get() + offset_D.at(i), Gemm::LayoutD::packed({options.m, options.n}));
  
      //
      // Compute reference output
      //
  
      // Create instantiation for device reference gemm kernel
      DeviceGemmReference gemm_reference;
  
      // Launch device reference gemm kernel
      gemm_reference(
        {options.m, options.n, options.k},
        ElementAccumulator(options.alpha),
        ref_A,
        ref_B,
        ElementAccumulator(options.beta),
        ref_C,
        ref_D);
  
      // Wait for kernel to finish
      CUTLASS_PPU_CHECK(hggcDeviceSynchronize());
  
      // Check if output from kernel and reference kernel are equal or not
      passed &= cutlass::reference::device::BlockCompareEqual(block_ref_D.get() + offset_D.at(i), block_D.get() + offset_D.at(i), options.m * options.n);
    }
    return passed;
  }


/// Allocates device-side data
void allocate(const Options &options) {
  int64_t total_elements_A = 0;
  int64_t total_elements_B = 0;
  int64_t total_elements_C = 0;
  int64_t total_elements_D = 0;

  for (int32_t i = 0; i < options.l; ++i) {

    offset_A.push_back(total_elements_A);
    offset_B.push_back(total_elements_B);
    offset_C.push_back(total_elements_C);
    offset_D.push_back(total_elements_D);

    int64_t elements_A = options.m * options.k;
    int64_t elements_B = options.k * options.n;
    int64_t elements_C = options.m * options.n;
    int64_t elements_D = options.m * options.n;

    total_elements_A += elements_A;
    total_elements_B += elements_B;
    total_elements_C += elements_C;
    total_elements_D += elements_D;
  }

  block_A.reset(total_elements_A);
  block_B.reset(total_elements_B);
  block_C.reset(total_elements_C);
  block_D.reset(total_elements_D);
  block_ref_D.reset(total_elements_D);
}

/// Initialize operands to be used in the GEMM and reference GEMM
void initialize(const Options &options) {

  stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(options.m, options.k, options.l));
  stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(options.n, options.k, options.l));
  stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(options.m, options.n, options.l));
  stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(options.m, options.n, options.l));

  //
  // Assign pointers
  //

  std::vector<ElementA *> ptr_A_host(options.l);
  std::vector<ElementB *> ptr_B_host(options.l);
  std::vector<ElementC *> ptr_C_host(options.l);
  std::vector<ElementC *> ptr_D_host(options.l);

  for (int32_t i = 0; i < options.l; ++i) {
    ptr_A_host.at(i) = block_A.get() + offset_A.at(i);
    ptr_B_host.at(i) = block_B.get() + offset_B.at(i);
    ptr_C_host.at(i) = block_C.get() + offset_C.at(i);
    ptr_D_host.at(i) = block_D.get() + offset_D.at(i);
  }

  ptr_A.reset(options.l);
  ptr_A.copy_from_host(ptr_A_host.data());

  ptr_B.reset(options.l);
  ptr_B.copy_from_host(ptr_B_host.data());

  ptr_C.reset(options.l);
  ptr_C.copy_from_host(ptr_C_host.data());

  ptr_D.reset(options.l);
  ptr_D.copy_from_host(ptr_D_host.data());

  initialize_block(block_A, seed + 2023);
  initialize_block(block_B, seed + 2022);
  initialize_block(block_C, seed + 2021);
}

  bool run(const Options& options, const cutlass::KernelHardwareInfo& hw_info) {
    allocate(options);
    initialize(options);

    typename GemmKernel::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kArray,
      {{options.m, options.n, options.k, options.l}},
      {ptr_A.get(), stride_A, ptr_B.get(), stride_B},
      {
       {options.alpha, options.beta}, // epilogue.thread
       ptr_C.get(), stride_C, ptr_D.get(), stride_D
      },
      hw_info
    };

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
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    hggcError_t result = hggcDeviceSynchronize();
    if (result != hggcSuccess) {
      std::cerr << "Error running the kernel. Last device error is: "
                << hggcGetErrorString(result) << std::endl;
      return false;
    }

    // Verify that the result is correct
    bool passed = verify(options);
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

  ExampleRunner<cutlass::arch::PPU0015, cutlass::config::KernelScheduleType::PTR_ARRAY, 1, 32, 32, 32, 16, 16, 32> stage1_runner;
  passed = stage1_runner.run(options, hw_info);
  print_result("aiu gemm array with stage=1:", passed);
  
  static constexpr cutlass::config::KernelScheduleType gemm_array_with_overlap_prologue 
      = cutlass::config::KernelScheduleType(int(cutlass::config::KernelScheduleType::PTR_ARRAY) | 
                                            int(cutlass::config::KernelScheduleType::PERSISTENT_OVERLAP_PROEPILOGUE));
  ExampleRunner<cutlass::arch::PPU0015, gemm_array_with_overlap_prologue, 3, 512, 64, 32, 64, 64, 32> overlapPrologue_runner;
  hw_info.cu_count *= 2; // above tile has 2 CTA/CU
  passed = overlapPrologue_runner.run(options, hw_info);
  print_result("aiu gemm array with overlapping epilogue and next tile's prologue:", passed);

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
