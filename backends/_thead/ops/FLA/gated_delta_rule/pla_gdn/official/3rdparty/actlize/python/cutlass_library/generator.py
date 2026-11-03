#################################################################################################
#
# Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD. All rights reserved. 
# Copyright (c) 2024, PTG Group Holding Limited. All rights reserved.
# Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#################################################################################################

"""
Utilities for enumerating CUTLASS library kernels
"""

import argparse
import enum
from itertools import chain, product
import logging
import os.path
import shutil
import sys
import copy
from typing import Any, Dict, Optional, Sequence, Tuple

_LOGGER = logging.getLogger(__name__)

def logging_prefix(indent_level: int = 0) -> str:
  """String prefix for start of each debug log entry"""
  prefix = '*** '
  indent = '  '
  return f"{prefix}{indent_level * indent}"

def log_debug_line(line: str, indent_level: int = 0) -> None:
  """Log one line of debug output"""
  prefix = logging_prefix(indent_level)
  _LOGGER.debug(prefix + line)

# Certain usecases of cutlass_library nearly always prefer to run as scripts with
# relative imports, rather than via an installed Python package. An example of this
# is using CUTLASS's CMake system to generate a library of kernels to be profiled.
# To make it easy to use these use cases when an existing installation of cutlass_library
# exists, this global flag can be set to true (via command-line arguments) to ensure
# that package-based installations are not used.

# Create a temporary argument parser to check only for the availability of the
# --disable-cutlass-package-imports argument, which controls whether package-based
# imports are disabled.
def _add_package_disablement_flag(argparser):
  argparser.add_argument("--disable-cutlass-package-imports", action='store_true', required=False,
                     help="Disable use of cutlass_library from Python package")

_parser = argparse.ArgumentParser()
_add_package_disablement_flag(_parser)
_args, _ = _parser.parse_known_args()

# Add `CUTLASS_IGNORE_PACKAGE` to `builtins` so that it is visible for gating future
# imports without requiring importing another module. Ideally, we would just place this
# as a global variable in a module to that could be imported and checked (e.g.,
# utils.CUTLASS_IGNORE_PACKAGE). However, this raises the issue of determining
# where this module should be sourced (from the cutlass_library package or from
# a relative import), which is the problem this variable is being used to solve in the
# first place.
import builtins
builtins.CUTLASS_IGNORE_PACKAGE = _args.disable_cutlass_package_imports

try:
  if CUTLASS_IGNORE_PACKAGE:
    raise ImportError("Disabling attempt to import cutlass_library")
  from cutlass_library.library import *
  from cutlass_library.manifest import *
except ImportError:
  from library import *
  from manifest import *
###################################################################################################

###################################################################################################
###################################################################################################

#
def EpilogueAlignment(max_alignment, tile, epilogue_steps = 8):
  ''' Helper to compute the maximum alignment of the epilogue '''

  def product(X, identity = 1):
    result = identity
    for item in X:
      result *= item
    return result

  elements_per_thread = product(tile.threadblock_shape[:-1]) // product(tile.warp_count) // 32 // epilogue_steps
  return min(max_alignment, elements_per_thread)

def DefaultSwizzlingFunctor():
    return SwizzlingFunctor.Identity8
    # To use StreamK decomposition for basic GEMMs, set `swizzling_functor = SwizzlingFunctor.StreamK`

#
def CreateGemmOperator(manifest, layouts, tile_descriptions, data_type, \
  alignment_constraints, complex_transforms = None, epilogue_functor = EpilogueFunctor.LinearCombination, \
  swizzling_functor = DefaultSwizzlingFunctor()):

  if complex_transforms is None:
    complex_transforms = [(ComplexTransform.none, ComplexTransform.none),]

  element_a, element_b, element_c, element_epilogue = data_type

  operations = []

  # by default, only generate the largest tile and largest alignment
  if manifest.kernel_filter == '':
    tile_descriptions = [tile_descriptions[0],]
    alignment_constraints = [alignment_constraints[0],]

  for layout in layouts:
    for tile_description in tile_descriptions:
      for alignment in alignment_constraints:
        for complex_transform in complex_transforms:

            # If alignment is a tuple or a list, then we have different alignments for A and B
            alignment_a = alignment if isinstance(alignment, int) else alignment[0]
            alignment_b = alignment if isinstance(alignment, int) else alignment[1]
            alignment_c = min(8, alignment_a) if isinstance(alignment, int) else alignment[2]

            A = TensorDescription(element_a, layout[0], alignment_a, complex_transform[0])
            B = TensorDescription(element_b, layout[1], alignment_b, complex_transform[1])
            C = TensorDescription(element_c, layout[2], alignment_c)

            new_operation = GemmOperation(GemmKind.Universal, tile_description.minimum_compute_capability, \
              tile_description, A, B, C, element_epilogue, epilogue_functor, swizzling_functor)

            manifest.append(new_operation)
            operations.append(new_operation)

  return operations

# Generates 3.0 API based GemmUniversal API kernels. Alignment constraints are folded in with layouts
def CreateGemmUniversal3xOperator(
    manifest, layouts, tile_descriptions, data_types,
    schedules = [[KernelScheduleType.ScheduleAuto, EpilogueScheduleType.ScheduleAuto]],
    complex_transforms=None,
    epilogue_functor=EpilogueFunctor.LinearCombination,
    swizzling_functor=SwizzlingFunctor.Identity1,
    # PPU default use persistent scheduler
    tile_schedulers=[TileSchedulerType.Persistent]):

  # Per-op dispatch (matches procedural_name's max_cc-based naming):
  #   tile_desc.maximum_compute_capability == 80 -> PPU0010 path
  #     gemm_kind = GemmKind.Universal3x       (emitted by EmitGemmUniversal3xInstance, ArchTag = PPU0010)
  #     tile_scheduler = TileSchedulerType.Persistent
  #   tile_desc.maximum_compute_capability == 89 -> PPU0015 path
  #     gemm_kind = GemmKind.PPU0015Universal3x (emitted by EmitPPU0015GemmUniversal3xInstance, ArchTag = PPU0015)
  #     tile_scheduler = TileSchedulerType.PersistentPPU0015
  # Note: ppu0015 tiles declare cc range (80, 89), so minimum_compute_capability
  # cannot be used for path selection; only maximum_compute_capability is
  # unique per arch.  The two paths are independent; multi-arch builds
  # (CUTLASS_PPU_ARCHS=ppu0010;ppu0015) produce both kernel families without
  # cross-contamination.

  if type(data_types) is dict:
    data_types = [data_types]

  for s in schedules:
    assert(len(s) == 2)

  if complex_transforms is None:
    complex_transforms = [(ComplexTransform.none, ComplexTransform.none), ]

  operations = []

  # by default, only generate the largest tile and largest alignment
  if manifest.kernel_filter == '':
    tile_descriptions = [tile_descriptions[0]]

  combinations = product(layouts, tile_descriptions, data_types, complex_transforms, schedules, tile_schedulers)
  for layout, tile_description, data_type, complex_transform, schedules, tile_scheduler in combinations:
    kernel_schedule, epilogue_schedule = schedules
    A = TensorDescription(
        data_type["a_type"], layout[0][0], layout[0][1], complex_transform[0])
    B = TensorDescription(
        data_type["b_type"], layout[1][0], layout[1][1], complex_transform[1])

    C = TensorDescription(data_type["c_type"], layout[2][0], layout[2][1])
    D = TensorDescription(data_type["d_type"], layout[2][0], layout[2][1])

    gemm_op_extra_args = {}
    op_max_cc = tile_description.maximum_compute_capability
    if op_max_cc == 89:
      gemm_kind = GemmKind.PPU0015Universal3x
      # PPU0015 path uses its dedicated persistent scheduler unless caller
      # explicitly overrides (e.g. StreamK).
      if tile_scheduler == TileSchedulerType.Persistent:
        op_tile_scheduler = TileSchedulerType.PersistentPPU0015
      else:
        op_tile_scheduler = tile_scheduler
    else:
      gemm_kind = GemmKind.Universal3x
      op_tile_scheduler = tile_scheduler
    element_compute = data_type.get("epi_type", data_type["acc_type"])

    operation = GemmOperation(
        gemm_kind, tile_description.minimum_compute_capability,
        tile_description, A, B, C, element_compute, epilogue_functor, swizzling_functor, D,
        kernel_schedule, epilogue_schedule, op_tile_scheduler, **gemm_op_extra_args)

    manifest.append(operation)
    operations.append(operation)

  return operations

# Generates 3.0 API based GemmUniversal API kernels. Alignment constraints are folded in with layouts
def CreateSparseGemmUniversal3xOperator(
    manifest, layouts, tile_descriptions, data_types,
    schedules = [[KernelScheduleType.ScheduleAuto, EpilogueScheduleType.ScheduleAuto]],
    complex_transforms=None,
    epilogue_functor=EpilogueFunctor.LinearCombination,
    swizzling_functor=SwizzlingFunctor.Identity1,
    tile_schedulers=[TileSchedulerType.Default]):

  if type(data_types) is dict:
    data_types = [data_types]

  for s in schedules:
    assert(len(s) == 2)

  if complex_transforms is None:
    complex_transforms = [(ComplexTransform.none, ComplexTransform.none), ]

  operations = []

  # by default, only generate the largest tile and largest alignment
  if manifest.kernel_filter == '':
    tile_descriptions = [tile_descriptions[0]]

  combinations = product(layouts, tile_descriptions, data_types, complex_transforms, schedules, tile_schedulers)
  for layout, tile_description, data_type, complex_transform, schedules, tile_scheduler in combinations:
    kernel_schedule, epilogue_schedule = schedules
    A = TensorDescription(
        data_type["a_type"], layout[0][0], layout[0][1], complex_transform[0])
    B = TensorDescription(
        data_type["b_type"], layout[1][0], layout[1][1], complex_transform[1])

    # Currently assume tensor C/D have same layout requirement.
    C = TensorDescription(data_type["c_type"], layout[2][0], layout[2][1])
    D = TensorDescription(data_type["d_type"], layout[2][0], layout[2][1])

    element_compute = data_type.get("epi_type", data_type["acc_type"])

    operation = GemmOperation(
        GemmKind.SparseUniversal3x, tile_description.minimum_compute_capability,
        tile_description, A, B, C, element_compute, epilogue_functor, swizzling_functor, D,
        kernel_schedule, epilogue_schedule, tile_scheduler)

    manifest.append(operation)
    operations.append(operation)

  return operations

#
def CreateSparseGemmOperator(manifest, layouts, tile_descriptions, data_type, \
  alignment_constraints, complex_transforms = None, epilogue_functor = EpilogueFunctor.LinearCombination, \
  swizzling_functor = SwizzlingFunctor.Identity8):

  if complex_transforms is None:
    complex_transforms = [(ComplexTransform.none, ComplexTransform.none),]

  element_a, element_b, element_c, element_epilogue = data_type

  gemm_kinds = [GemmKind.Sparse]

  operations = []

  # by default, only generate the largest tile and largest alignment
  if manifest.kernel_filter == '':
    tile_descriptions = [tile_descriptions[0],]
    alignment_constraints = [alignment_constraints[0],]

  for layout in layouts:
    for tile_description in tile_descriptions:
      for alignment in alignment_constraints:
        for complex_transform in complex_transforms:

            alignment_c = min(8, alignment)

            A = TensorDescription(element_a, layout[0], alignment, complex_transform[0])
            B = TensorDescription(element_b, layout[1], alignment, complex_transform[1])
            C = TensorDescription(element_c, layout[2], alignment_c)

            new_operation = GemmOperation(GemmKind.Sparse, tile_description.minimum_compute_capability, \
              tile_description, A, B, C, element_epilogue, epilogue_functor, swizzling_functor)

            manifest.append(new_operation)
            operations.append(new_operation)

  return operations

#
def CreateGemmPlanarComplexOperator(manifest, layouts, tile_descriptions, data_type, \
  alignment_constraints, complex_transforms):

  if complex_transforms is None:
    complex_transforms = [(ComplexTransform.none, ComplexTransform.none),]

  element_a, element_b, element_c, element_epilogue = data_type

  gemm_kinds = [GemmKind.PlanarComplex, GemmKind.PlanarComplexArray]

  # by default, only generate the largest tile and largest alignment
  if manifest.kernel_filter == '':
    tile_descriptions = [tile_descriptions[0],]
    alignment_constraints = [alignment_constraints[0],]

  for gemm_kind in gemm_kinds:
    for layout in layouts:
      for tile_description in tile_descriptions:
        for alignment in alignment_constraints:
          for complex_transform in complex_transforms:

            alignment_c = min(8, alignment)

            A = TensorDescription(element_a, layout[0], alignment, complex_transform[0])
            B = TensorDescription(element_b, layout[1], alignment, complex_transform[1])
            C = TensorDescription(element_c, layout[2], alignment_c)

            manifest.append(GemmOperation(gemm_kind, \
              tile_description.minimum_compute_capability, \
              tile_description, A, B, C, element_epilogue))
  return

#
def CreateGemmGroupedOperator(manifest, layouts, tile_descriptions, data_type, \
  alignment_constraints, complex_transforms = None, epilogue_functor = EpilogueFunctor.LinearCombination, \
  swizzling_functor = SwizzlingFunctor.Identity8):

  if complex_transforms is None:
    complex_transforms = [(ComplexTransform.none, ComplexTransform.none),]

  element_a, element_b, element_c, element_epilogue = data_type

  operations = []

  # by default, only generate the largest tile and largest alignment
  if manifest.kernel_filter == '':
    tile_descriptions = [tile_descriptions[0],]
    alignment_constraints = [alignment_constraints[0],]

  for layout in layouts:
    for tile_description in tile_descriptions:
      for alignment in alignment_constraints:
        for complex_transform in complex_transforms:

            alignment_c = min(8, alignment)

            A = TensorDescription(element_a, layout[0], alignment, complex_transform[0])
            B = TensorDescription(element_b, layout[1], alignment, complex_transform[1])
            C = TensorDescription(element_c, layout[2], alignment_c)

            new_operation = GroupedGemmOperation(GemmKind.Grouped, tile_description.minimum_compute_capability, \
              tile_description, A, B, C, element_epilogue, epilogue_functor, swizzling_functor)

            manifest.append(new_operation)
            operations.append(new_operation)

  return operations

###################################################################################################
###################################################################################################


try:
    from .ppu_utils import (
        generate_fp16_tile_descriptions_ppu,
        get_valid_schedules,
        generate_data_types_from_math_instruction,
        fix_alignments,
    )
except ImportError:
    from ppu_utils import (
        generate_fp16_tile_descriptions_ppu,
        get_valid_schedules,
        generate_data_types_from_math_instruction,
        fix_alignments,
    )


###################################################################################################

def GeneratePPU_TensorOp_16b_MMA_gemm(manifest):

  instantiation_level = manifest.get_ppu_instantiation_level(pruned_level=100, default_level=131, exhaustive_level=9999)
  is_aligned = True

  # layouts for ABC and their alignments.
  layouts = [
    [[LayoutType.ColumnMajor, 8], [LayoutType.ColumnMajor, 8], [LayoutType.ColumnMajor, 1]],
    [[LayoutType.ColumnMajor, 8], [LayoutType.RowMajor,    8], [LayoutType.ColumnMajor, 1]],
    [[LayoutType.RowMajor,    8], [LayoutType.ColumnMajor, 8], [LayoutType.ColumnMajor, 1]],
    [[LayoutType.RowMajor,    8], [LayoutType.RowMajor,    8], [LayoutType.ColumnMajor, 1]],
  ]

  math_instructions = [
    MathInstruction(
      [16, 16, 16],
      DataType.f16, DataType.f16, DataType.f32,
      OpcodeClass.TensorOp,
      MathOperation.multiply_add),
  ]

  tile_descriptions = generate_fp16_tile_descriptions_ppu(manifest.compute_capabilities, math_instructions=math_instructions)

  for tile_desc in tile_descriptions:
    math_inst = tile_desc.math_instruction
    data_type_w_source = generate_data_types_from_math_instruction(math_inst)
    # PPU not support Ctype void
    # data_type_wo_source = generate_data_types_from_math_instruction(math_inst, element_source=DataType.void)
    data_types = [data_type_w_source]

    # for mixed precision kernels, also generate kernels that write output matrix in the A/B format
    # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
    if math_inst.element_a != math_inst.element_accumulator:
        data_type_mixed_w_source = generate_data_types_from_math_instruction(
            math_inst,
            element_source=math_inst.element_a,
            element_dest=math_inst.element_a
        )
        # data_type_mixed_wo_source = generate_data_types_from_math_instruction(
        #     math_inst,
        #     element_source=DataType.void,
        #     element_dest=math_inst.element_a
        # )
        data_types.append(data_type_mixed_w_source)
        # data_types.append(data_type_mixed_wo_source)

    for layout in layouts:
        for data_type in data_types:
            layout = fix_alignments(data_type, layout, alignment_bits=128)

            schedules, stream_k_schedules = get_valid_schedules(
              tile_description=tile_desc,
              is_aligned=is_aligned,
              data_types=data_type,
              instantiation_level=instantiation_level,
              layout=layout,
            )

            if len(schedules):
              CreateGemmUniversal3xOperator(manifest, [layout], [tile_desc], data_type, schedules)
              if len(stream_k_schedules):
                CreateGemmUniversal3xOperator(manifest, [layout], [tile_desc], data_type,
                                              stream_k_schedules,
                                              tile_schedulers=[TileSchedulerType.StreamK])



###################################################################################################


def GeneratePPU(manifest):
  # todo: support more PPU native gemm ops
  GeneratePPU_TensorOp_16b_MMA_gemm(manifest)


###################################################################################################



def numeric_log_level(log_level: str) -> int:
  """
  Converts the string identifier of the log level
  into the numeric identifier used in setting the log level.

  :param x: string representation of log level (e.g., 'INFO', 'DEBUG')
  :type x: str

  :return: numeric representation of log level
  :rtype: int
  """
  numeric_level = getattr(logging, log_level.upper(), None)
  if not isinstance(numeric_level, int):
    raise ValueError(f'Invalid log level: {log_level}')
  return numeric_level


# This function for defining the ArgumentParser is used to make it easy for the CUTLASS Python interface
# to leverage the functionality in this file without running this script via a shell prompt.
def define_parser():
  parser = argparse.ArgumentParser(description="Generates device kernel registration code for CUTLASS Kernels")
  parser.add_argument("--operations", default="all", help="Specifies the operation to generate (gemm, all)")
  parser.add_argument("--build-dir", default=".", required=False, help="CUTLASS top-level build directory")
  parser.add_argument("--curr-build-dir", default=".", help="CUTLASS current build directory. cmake files will be emitted in this directory")
  parser.add_argument("--generator-target", default='library', help="Target of CUTLASS Library Generator.")
  parser.add_argument("--architectures", default='53;60;61;70;75;80;90', help="Target compute architectures")
  parser.add_argument("--kernels", default='', help='Comma-delimited list to filter kernels by name.  ' +
                      'Specifying this as \"all\" includes ALL the kernels, ' +
                      'while not specifying this includes only the default set of kernels.')
  parser.add_argument("--ignore-kernels", default='', help='Comma-delimited list of kernels ' +
                      'to exclude from build.  For backwards compatibility reasons, ' +
                      'this option only takes effect if --kernels is set to a nonempty value.')
  parser.add_argument("--exclude-kernels", default='', help='Comma-delimited list of kernels ' +
                      'to exclude from build.  In contrast to --ignore-kernels, ' +
                      'this option always takes effect, ' +
                      'whether or not --kernels is set to a nonempty value.  ' +
                      'It also can exclude kernels from the filter file ' +
                      '(see --kernel-filter-file option below).')
  parser.add_argument("--filter-by-cc", default='True', type=str, help='If enabled, kernels whose compute capability range is not satisfied by the build target are excluded.')
  parser.add_argument('--kernel-filter-file',   type=str, default=None, required=False, help='Full path of filter file')
  parser.add_argument('--selected-kernel-list',   type=str, default=None, required=False,
                        help='Specify the output log file containing all enabled kernels in this build')
  parser.add_argument("--interface-dir", default=None, required=False, help="Interface header to kernels")
  parser.add_argument("--disable-full-archs-compilation", action="store_true", required=False, help="Disable compilation for every archs in --architectures")
  parser.add_argument("--log-level", default='info', type=numeric_log_level, required=False,
                      help='Logging level to be used by the generator script')
  parser.add_argument('--instantiation-level', type=str, default="", required=False, help="Instantiation level for PPU kernels. Set to `max` and make sure `--kernels` is not empty to generate all possible configurations.")
  _add_package_disablement_flag(parser)
  return parser


if __name__ == "__main__":
  parser = define_parser()
  args = parser.parse_args()

  # Set the logging level based on the user-provided `--log-level` command-line option
  logging.basicConfig(level=args.log_level)

  manifest = Manifest(args)

  # PPU 1.0/1.5
  GeneratePPU(manifest)
  if 'library' in args.generator_target.split(','):
    manifest.emit(GeneratorTarget.Library)

  if args.selected_kernel_list is not None:
    if len(manifest.selected_kernels) > 0:
      with open(args.selected_kernel_list, 'w') as file_writer:
        for line in manifest.selected_kernels:
          file_writer.write("%s\n" % line)

###################################################################################################
