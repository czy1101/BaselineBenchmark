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
Utilities for emitting GEMM kernels
"""

import collections
import enum
import functools
import logging
import operator
import os.path
import shutil

try:
  import builtins
  if hasattr(builtins, "CUTLASS_IGNORE_PACKAGE") and CUTLASS_IGNORE_PACKAGE == True:
    raise ImportError("Disabling attempt to import cutlass_library")
  from cutlass_library.library import *
except ImportError:
  from library import *

_LOGGER = logging.getLogger(__name__)

###################################################################################################
#
# Data structure modeling a GEMM operation
#
###################################################################################################

#
class GemmOperation:
  #
  def __init__(self, gemm_kind, arch, tile_description, A, B, C, element_epilogue, \
      epilogue_functor = EpilogueFunctor.LinearCombination, swizzling_functor = SwizzlingFunctor.Identity8, D = None,
      kernel_schedule = KernelScheduleType.ScheduleAuto, epilogue_schedule = EpilogueScheduleType.ScheduleAuto,
      tile_scheduler = TileSchedulerType.Default
    ):

    kinds_3x = {
      GemmKind.Universal3x,
      GemmKind.SparseUniversal3x,
      GemmKind.PPU0015Universal3x,
    }
    self.is_3x = gemm_kind in kinds_3x
    self.prefix = "3x" if self.is_3x else ""
    self.operation_kind = OperationKind.Gemm
    self.arch = arch
    self.tile_description = tile_description
    self.gemm_kind = gemm_kind
    self.A = A
    self.B = B
    self.C = C
    self.D = D

    if self.D == None:
      self.D = self.C

    if not self.is_3x:
      assert(kernel_schedule == KernelScheduleType.ScheduleAuto)
      assert(epilogue_schedule == EpilogueScheduleType.ScheduleAuto)
    self.kernel_schedule = kernel_schedule
    self.epilogue_schedule = epilogue_schedule
    self.element_epilogue = element_epilogue
    self.epilogue_functor = epilogue_functor

    if self.is_3x and epilogue_functor == EpilogueFunctor.LinearCombination:
      self.epilogue_functor = EpilogueFunctor3x.LinearCombination

    self.swizzling_functor = swizzling_functor
    self.tile_scheduler = tile_scheduler

  #
  def is_complex(self):
    complex_operators = [
      MathOperation.multiply_add_complex,
      MathOperation.multiply_add_complex_gaussian,
      MathOperation.multiply_add_complex_fast_f32
    ]
    return self.tile_description.math_instruction.math_operation in complex_operators

  #
  def is_mixed_input(self):
    return self.A.element != self.B.element

  #
  def is_planar_complex(self):
    return self.gemm_kind in (GemmKind.PlanarComplex, GemmKind.PlanarComplexArray)

  #
  def accumulator_type(self):
    accum = self.tile_description.math_instruction.element_accumulator

    if self.is_complex():
      return get_complex_from_real(accum)

    return accum

  #
  def short_math_name(self):
    if self.tile_description.math_instruction.math_operation == MathOperation.multiply_add_complex_gaussian:
      return "g%s" % ShortDataTypeNames[self.accumulator_type()]
    return ShortDataTypeNames[self.accumulator_type()]


  #
  def core_name(self):
    ''' The basic operation kind is prefixed with a letter indicating the accumulation type. '''

    inst_shape = ''
    inst_operation = ''
    intermediate_type = ''

    math_operations_map = {
      MathOperation.xor_popc: 'xor',
      MathOperation.and_popc: 'and',
      MathOperation.multiply_add_fast_accum: 'fastaccum',
    }

    tensor_ops = [
      OpcodeClass.TensorOp,
      OpcodeClass.SparseTensorOp,
    ]

    is_tensor_op = self.tile_description.math_instruction.opcode_class in tensor_ops

    if is_tensor_op:

      math_op = self.tile_description.math_instruction.math_operation
      math_op_string = math_operations_map[math_op] if math_op in math_operations_map.keys() else ''

      if self.is_3x:
        inst_shape = "{0}x{1}x{2}".format(*tuple(self.tile_description.math_instruction.instruction_shape))
      else:
        inst_shape = "{0}{1}{2}".format(*tuple(self.tile_description.math_instruction.instruction_shape))

      inst_shape += math_op_string

      if self.tile_description.math_instruction.element_a != self.A.element and \
        self.tile_description.math_instruction.element_a != self.tile_description.math_instruction.element_accumulator:
        intermediate_type = DataTypeNames[self.tile_description.math_instruction.element_a]

    return "%s%s%s%s" % (self.short_math_name(), inst_shape, intermediate_type, GemmKindNames[self.gemm_kind])

  # Generates a string representing the MMA instruction.
  def extended_name(self):
    ''' Append data types if they differ from compute type. '''
    if self.is_complex():
      extended_name = "${core_name}"
    else:
      if self.is_mixed_input():
        extended_name = "${core_name}_${element_a}_${element_b}"
        if self.C.element != self.tile_description.math_instruction.element_accumulator:
          extended_name = "${element_c}_" + extended_name
      else:
        extended_name = "${core_name}"
        if self.C.element != self.tile_description.math_instruction.element_accumulator:
          extended_name = "${element_c}_" + extended_name
        if self.A.element != self.tile_description.math_instruction.element_accumulator:
          extended_name += "_${element_a}"

    extended_name = SubstituteTemplate(extended_name, {
      'element_a': DataTypeNames[self.A.element],
      'element_b': DataTypeNames[self.B.element],
      'element_c': DataTypeNames[self.C.element],
      'core_name': self.core_name()
      })

    return extended_name

  def extended_name_3x(self):
    '''Generates a string representing the MMA atom. Assumes accumulator type is C type.'''
    extended_name = "{core_name}_{element_a}_{element_b}_{element_acc}_{element_c}_{element_d}".format(
      element_a = DataTypeNames[self.A.element],
      element_b = DataTypeNames[self.B.element],
      element_acc = DataTypeNames[self.accumulator_type()],
      element_c = DataTypeNames[self.C.element],
      element_d = DataTypeNames[self.D.element],
      core_name = self.core_name())
    return extended_name

  def datatype_name_3x(self):
    '''Generates a string representing the MMA atom. Assumes accumulator type is C type.'''
    datatype_name = "{element_a}_{element_b}_{element_acc}_{element_c}_{element_d}".format(
      element_a = DataTypeNames[self.A.element],
      element_b = DataTypeNames[self.B.element],
      element_acc = DataTypeNames[self.accumulator_type()],
      element_c = DataTypeNames[self.C.element],
      element_d = DataTypeNames[self.D.element])
    return datatype_name

  # Generates a short string representing the AB layout tags (e.g. nt or tn)
  def layout_name(self):
    if self.is_complex() or self.is_planar_complex():
      return "%s%s" % (
        ShortComplexLayoutNames[(self.A.layout, self.A.complex_transform)],
        ShortComplexLayoutNames[(self.B.layout, self.B.complex_transform)]
      )
    return "%s%s" % (ShortLayoutTypeNames[self.A.layout], ShortLayoutTypeNames[self.B.layout])

  # Generates a short string representing the ABC layout tags (e.g. ntn or tnn)
  def layout_name_3x(self):
    if self.is_complex() or self.is_planar_complex():
      return "{}{}{}".format(
        ShortComplexLayoutNames[(self.A.layout, self.A.complex_transform)],
        ShortComplexLayoutNames[(self.B.layout, self.B.complex_transform)],
        ShortComplexLayoutNames[(self.C.layout, self.C.complex_transform)])
    else:
      return "{}{}{}".format(
        ShortLayoutTypeNames[self.A.layout],
        ShortLayoutTypeNames[self.B.layout],
        ShortLayoutTypeNames[self.C.layout])

  # Generates a short string representing underlying kernel schedule type
  def kernel_schedule_name_3x(self):
    return KernelScheduleSuffixes[self.kernel_schedule]

  # Generates a short string representing underlying epilogue schedule type
  def epilogue_schedule_name_3x(self):
    return EpilogueScheduleSuffixes[self.epilogue_schedule]

  # Generate a short string representing the operation class
  def opcode_class_name(self):
    return OpcodeClassNames[self.tile_description.math_instruction.opcode_class]

  # Generates the full kernel function name
  def procedural_name(self):
    ''' The full procedural name indicates architecture, extended name, tile size, and layout. '''
    opcode_class_name = OpcodeClassNames[self.tile_description.math_instruction.opcode_class]

    if self.tile_description.maximum_compute_capability == 89:
      ppu_arch = '0015'
    elif self.tile_description.maximum_compute_capability == 80:
      ppu_arch = '0010'
    else:
      ppu_arch = 'xxxx'

    kernel_name_template = "cutlass{p}_ppu{ar}_{op}_{ex}{ct}_{l}_{s}_align{al}{t}{k}{e}"
    return kernel_name_template.format(
        p = self.prefix,
        ar = ppu_arch,
        op = opcode_class_name,
        ex = self.extended_name_3x(),
        ct = '_' + 'x'.join([str(i) for i in self.tile_description.tile_shape]) if self.tile_description.tile_shape[0] > 0 else "",
        l = self.tile_description.stages,
        s = self.layout_name_3x(),
        al = str(max(self.A.alignment, self.B.alignment)),
        t = TileSchedulerSuffixes[self.tile_scheduler],
        k = self.kernel_schedule_name_3x(),
        e = self.epilogue_schedule_name_3x())

  #
  def configuration_name(self):
    ''' The full procedural name indicates architecture, extended name, tile size, and layout. '''
    return self.procedural_name()

  def __hash__(self):
    return hash(self.configuration_name())

  def __eq__(self, other):
    return self.configuration_name() == other.configuration_name()

###################################################################################################
#
# Data structure modeling a grouped GEMM operation
#
###################################################################################################

#
class GroupedGemmOperation(GemmOperation):
  #
  def __init__(self, gemm_kind, arch, tile_description, A, B, C, element_epilogue, \
      epilogue_functor = EpilogueFunctor.LinearCombination, swizzling_functor = SwizzlingFunctor.Identity8, \
      scheduler_mode = GroupScheduleMode.Device):
    super().__init__(gemm_kind, arch, tile_description, A, B, C, element_epilogue, \
                     epilogue_functor, swizzling_functor)

    self.scheduler_mode = scheduler_mode

  #
  def procedural_name(self):
    ''' The full procedural name indicates architecture, extended name, tile size, and layout. '''
    base = super().procedural_name()
    return SubstituteTemplate(
      base + "_schedule${schedule}",
      {
        'schedule': ShortGroupScheduleModeNames[self.scheduler_mode]
      })


###################################################################################################
#
# Emits single instances of a CUTLASS device-wide operator
#
###################################################################################################

#
class EmitGemmInstance:
  ''' Responsible for emitting a CUTLASS template definition'''

  def __init__(self, operation_suffix = ''):
    self.operation_suffix = operation_suffix
    self.includes = []
    self.gemm_template = """
  // Gemm operator ${operation_name}
  using Operation_${operation_name} = cutlass::gemm::device::Gemm<
    ${element_a}, ${layout_a},
    ${element_b}, ${layout_b},
    ${element_c}, ${layout_c},
    ${element_accumulator},
    ${opcode_class},
    ${arch},
    cutlass::gemm::GemmShape<${threadblock_shape_m}, ${threadblock_shape_n}, ${threadblock_shape_k}>,
    cutlass::gemm::GemmShape<${warp_shape_m}, ${warp_shape_n}, ${warp_shape_k}>,
    cutlass::gemm::GemmShape<${instruction_shape_m}, ${instruction_shape_n}, ${instruction_shape_k}>,
    ${epilogue_functor}<
      ${element_c},
      ${epilogue_vector_length},
      ${element_accumulator},
      ${element_epilogue}
    >,
    ${swizzling_functor},
    ${stages},
    ${align_a},
    ${align_b},
    false,
    ${math_operation}
    ${residual}
  >;
"""
    self.gemm_complex_template = """
  // Gemm operator ${operation_name}
  using Operation_${operation_name} = cutlass::gemm::device::GemmComplex<
    ${element_a}, ${layout_a},
    ${element_b}, ${layout_b},
    ${element_c}, ${layout_c},
    ${element_accumulator},
    ${opcode_class},
    ${arch},
    cutlass::gemm::GemmShape<${threadblock_shape_m}, ${threadblock_shape_n}, ${threadblock_shape_k}>,
    cutlass::gemm::GemmShape<${warp_shape_m}, ${warp_shape_n}, ${warp_shape_k}>,
    cutlass::gemm::GemmShape<${instruction_shape_m}, ${instruction_shape_n}, ${instruction_shape_k}>,
    ${epilogue_functor}<
      ${element_c},
      ${epilogue_vector_length},
      ${element_accumulator},
      ${element_epilogue}
    >,
    ${swizzling_functor},
    ${stages},
    ${transform_a},
    ${transform_b},
    ${math_operation}
    ${residual}
  >;
"""

  #
  def instance_template(self):
    return """
${compile_guard_start}
  manifest.append(new ${gemm_kind}<Operation_${operation_name}>("${operation_name}"));
${compile_guard_end}
"""

  def get_arch_tag(self, arch: int) -> str:
    arch_map = {
        80: "cutlass::arch::PPU0010",
        89: "cutlass::arch::PPU0015"
    }
    arch_tag = arch_map.get(arch)
    if arch_tag is None:
        raise ValueError(f"Unsupported architecture: {arch}.")
        
    return arch_tag

  #
  def emit(self, operation):

    warp_shape = [operation.tile_description.threadblock_shape[idx] // operation.tile_description.warp_count[idx] for idx in range(3)]

    epilogue_vector_length = int(min(operation.C.alignment * DataTypeSize[operation.C.element], 128) / DataTypeSize[operation.C.element])

    residual = ''

    values = {
      'operation_name': operation.procedural_name(),
      'element_a': DataTypeTag[operation.A.element],
      'layout_a': LayoutTag[operation.A.layout],
      'element_b': DataTypeTag[operation.B.element],
      'layout_b': LayoutTag[operation.B.layout],
      'element_c': DataTypeTag[operation.C.element],
      'layout_c': LayoutTag[operation.C.layout],
      'element_accumulator': DataTypeTag[operation.accumulator_type()],
      'opcode_class': OpcodeClassTag[operation.tile_description.math_instruction.opcode_class],
      'arch': self.get_arch_tag(operation.arch),
      'threadblock_shape_m': str(operation.tile_description.threadblock_shape[0]),
      'threadblock_shape_n': str(operation.tile_description.threadblock_shape[1]),
      'threadblock_shape_k': str(operation.tile_description.threadblock_shape[2]),
      'warp_shape_m': str(warp_shape[0]),
      'warp_shape_n': str(warp_shape[1]),
      'warp_shape_k': str(warp_shape[2]),
      'instruction_shape_m': str(operation.tile_description.math_instruction.instruction_shape[0]),
      'instruction_shape_n': str(operation.tile_description.math_instruction.instruction_shape[1]),
      'instruction_shape_k': str(operation.tile_description.math_instruction.instruction_shape[2]),
      'epilogue_vector_length': str(epilogue_vector_length),
      'element_epilogue': str(DataTypeTag[operation.element_epilogue]),
      'epilogue_functor': EpilogueFunctorTag[operation.epilogue_functor],
      'swizzling_functor': SwizzlingFunctorTag[operation.swizzling_functor],
      'stages': str(operation.tile_description.stages),
      'align_a': str(operation.A.alignment),
      'align_b': str(operation.B.alignment),
      'transform_a': ComplexTransformTag[operation.A.complex_transform],
      'transform_b': ComplexTransformTag[operation.B.complex_transform],
      'math_operation': MathOperationTag[operation.tile_description.math_instruction.math_operation],
      'residual': residual
    }

    template = self.gemm_complex_template if operation.is_complex() else self.gemm_template

    return SubstituteTemplate(template, values)

###################################################################################################

#
class EmitGemmUniversalInstance:
  ''' Responsible for emitting a CUTLASS template definition'''

  def __init__(self, operation_suffix = ''):
    self.operation_suffix = operation_suffix
    self.includes = [
      "cutlass/cutlass.h",
      "cutlass/numeric_types.h",
      "cutlass/arch/arch.h",
      "cutlass/arch/mma.h",
      "cutlass/layout/matrix.h",
      "cutlass/gemm/device/gemm.h",
      "cutlass/gemm/device/gemm_universal_adapter.h",
      "cutlass/gemm/kernel/default_gemm_universal.h",
    ]
    self.builtin_epilogue_functor_template = """
    ${epilogue_functor}<
      ${element_c},
      ${epilogue_vector_length},
      ${element_accumulator},
      ${element_epilogue}
    >
"""
    self.gemm_template = """
// Gemm operator ${operation_name}
using ${operation_name}_base =
  typename cutlass::gemm::kernel::DefaultGemmUniversal<
    ${element_b}, ${layout_b}, ${transform_b}, ${align_b},    // transposed B operand
    ${element_a}, ${layout_a}, ${transform_a}, ${align_a},    // transposed A operand
    ${element_c}, ${layout_c},
    ${element_accumulator},
    ${opcode_class},
    ${arch},
    cutlass::gemm::GemmShape<${threadblock_shape_m}, ${threadblock_shape_n}, ${threadblock_shape_k}>,
    cutlass::gemm::GemmShape<${warp_shape_m}, ${warp_shape_n}, ${warp_shape_k}>,
    cutlass::gemm::GemmShape<${instruction_shape_m}, ${instruction_shape_n}, ${instruction_shape_k}>,
    ${epilogue_functor},
    ${swizzling_functor},
    ${stages},
    ${math_operation}
>::GemmKernel;

// Define named type
struct ${operation_name}${operation_suffix} :
  public ${operation_name}_base { };
"""
    self.gemm_template_interleaved = """
// Gemm operator ${operation_name}
using ${operation_name}_base =
  typename cutlass::gemm::kernel::DefaultGemmUniversal<
    ${element_a}, ${layout_a}, ${transform_a}, ${align_a},
    ${element_b}, ${layout_b}, ${transform_b}, ${align_b},
    ${element_c}, ${layout_c},
    ${element_accumulator},
    ${opcode_class},
    ${arch},
    cutlass::gemm::GemmShape<${threadblock_shape_m}, ${threadblock_shape_n}, ${threadblock_shape_k}>,
    cutlass::gemm::GemmShape<${warp_shape_m}, ${warp_shape_n}, ${warp_shape_k}>,
    cutlass::gemm::GemmShape<${instruction_shape_m}, ${instruction_shape_n}, ${instruction_shape_k}>,
    ${epilogue_functor},
    ${swizzling_functor},
    ${stages},
    ${math_operation}
>::GemmKernel;

// Define named type
struct ${operation_name}${operation_suffix} :
  public ${operation_name}_base { };
"""

  #
  def instance_template(self):
    return """
${compile_guard_start}
  manifest.append(new ${gemm_kind}<
      cutlass::gemm::device::GemmUniversalAdapter<${operation_name}>
    >("${operation_name}"));
${compile_guard_end}
"""

  def get_arch_tag(self, arch: int) -> str:
    arch_map = {
        80: "cutlass::arch::PPU0010",
        89: "cutlass::arch::PPU0015"
    }
    arch_tag = arch_map.get(arch)
    if arch_tag is None:
        raise ValueError(f"Unsupported architecture: {arch}.")
        
    return arch_tag

  #
  def emit(self, operation):

    threadblock_shape = operation.tile_description.threadblock_shape
    warp_count = operation.tile_description.warp_count

    warp_shape = [threadblock_shape[idx] // warp_count[idx] for idx in range(3)]

    transpose_layouts = {
      LayoutType.ColumnMajor: LayoutType.RowMajor,
      LayoutType.RowMajor: LayoutType.ColumnMajor
    }

    if operation.A.layout in transpose_layouts.keys() and \
      operation.B.layout in transpose_layouts.keys() and \
      operation.C.layout in transpose_layouts.keys():

      instance_layout_A = transpose_layouts[operation.A.layout]
      instance_layout_B = transpose_layouts[operation.B.layout]
      instance_layout_C = transpose_layouts[operation.C.layout]

      gemm_template = self.gemm_template
    else:
      instance_layout_A, instance_layout_B, instance_layout_C = \
        (operation.A.layout, operation.B.layout, operation.C.layout)

      gemm_template = self.gemm_template_interleaved
    #

    # Support built-in epilogue functors or user-defined functions
    if isinstance(operation.epilogue_functor, enum.Enum):

      epilogue_vector_length = \
        min(operation.C.alignment * DataTypeSize[operation.C.element], 128) // DataTypeSize[operation.C.element]

      values = {
        'epilogue_vector_length': str(epilogue_vector_length),
        'element_epilogue': str(DataTypeTag[operation.element_epilogue]),
        'epilogue_functor': EpilogueFunctorTag[operation.epilogue_functor],
      }
      epilogue_functor = SubstituteTemplate(self.builtin_epilogue_functor_template, values)
    else:
      epilogue_functor = self.epilogue_functor.emit_declaration()
    #

    values = {
      'operation_name': operation.procedural_name(),
      'operation_suffix': self.operation_suffix,
      'element_a': DataTypeTag[operation.A.element],
      'layout_a': LayoutTag[instance_layout_A],
      'element_b': DataTypeTag[operation.B.element],
      'layout_b': LayoutTag[instance_layout_B],
      'element_c': DataTypeTag[operation.C.element],
      'layout_c': LayoutTag[instance_layout_C],
      'element_accumulator': DataTypeTag[operation.accumulator_type()],
      'opcode_class': OpcodeClassTag[operation.tile_description.math_instruction.opcode_class],
      'arch': self.get_arch_tag(operation.arch),
      'threadblock_shape_m': str(operation.tile_description.threadblock_shape[0]),
      'threadblock_shape_n': str(operation.tile_description.threadblock_shape[1]),
      'threadblock_shape_k': str(operation.tile_description.threadblock_shape[2]),
      'warp_shape_m': str(warp_shape[0]),
      'warp_shape_n': str(warp_shape[1]),
      'warp_shape_k': str(warp_shape[2]),
      'instruction_shape_m': str(operation.tile_description.math_instruction.instruction_shape[0]),
      'instruction_shape_n': str(operation.tile_description.math_instruction.instruction_shape[1]),
      'instruction_shape_k': str(operation.tile_description.math_instruction.instruction_shape[2]),
      'epilogue_functor': epilogue_functor,
      'swizzling_functor': SwizzlingFunctorTag[operation.swizzling_functor],
      'stages': str(operation.tile_description.stages),
      'align_a': str(operation.A.alignment),
      'align_b': str(operation.B.alignment),
      'transform_a': ComplexTransformTag[operation.A.complex_transform],
      'transform_b': ComplexTransformTag[operation.B.complex_transform],
      'math_operation': MathOperationTag[operation.tile_description.math_instruction.math_operation]
    }

    return SubstituteTemplate(gemm_template, values)


###################################################################################################

class EmitGemmUniversal3xInstance:
  ''' Responsible for emitting a CUTLASS 3.x template definition'''

  def __init__(self, operation_suffix = ''):
    self.operation_suffix = operation_suffix
    self.includes = [
      "cutlass/cutlass.h",
      "cutlass/numeric_types.h",
      "cutlass/epilogue/collective/default_epilogue.hpp",
      "cutlass/epilogue/thread/linear_combination.h",
      "cutlass/epilogue/collective/collective_builder.hpp",

      "ppu_include.hpp",
      "cutlass/gemm/collective/builders/ppu_mma_builder.inl",
    ]
    self.builtin_epilogue_functor_template = \
"""${epilogue_functor}<
      ${element_d},
      ${element_epilogue},
      ${element_c},
      ${element_epilogue}
    >"""

    self.gemm_template = """

using namespace cute;

using ${operation_name}_epilogue =
  typename cutlass::epilogue::collective::CollectiveBuilder<
    ${arch}, ${opcode_class_epi},
    cute::Shape<cute::_${tile_shape_epi_m}, cute::_${tile_shape_epi_n}, cute::_${tile_shape_epi_k}>,
    cute::Shape<cute::_${warp_shape_m}, cute::_${warp_shape_n}, cute::_${warp_shape_k}>,
    ${epi_tile_mn},
    ${element_accumulator}, ${element_epilogue},
    ${element_c}, ${layout_c}, ${align_c},
    ${element_d}, ${layout_d}, ${align_d},
    ${epilogue_schedule},
    ${epilogue_functor}
  >::CollectiveOp;

using ${operation_name}_mainloop =
  typename cutlass::gemm::collective::CollectiveBuilder<
    ${arch}, ${opcode_class_main},
    ${element_a}, ${layout_a}, ${align_a},
    ${element_b}, ${layout_b}, ${align_b},
    ${element_accumulator},
    cute::Shape<cute::_${tile_shape_main_m}, cute::_${tile_shape_main_n}, cute::_${tile_shape_main_k}>,
    cute::Shape<cute::_${warp_shape_m}, cute::_${warp_shape_n}, cute::_${warp_shape_k}>,
    ${stages},
    ${kernel_schedule}
  >::CollectiveOp;

// Gemm operator ${operation_name}
using ${operation_name}_base = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int,int>,
    ${operation_name}_mainloop,
    ${operation_name}_epilogue,
    ${tile_scheduler}>;

// Define named type
struct ${operation_name} :
  public ${operation_name}_base { };

"""
  #
  def instance_template(self):
    return """
${compile_guard_start}
  {
    using GemmKernel = cutlass::gemm::device::GemmUniversalAdapter<${operation_name}>;
    manifest.append(
      new ${gemm_kind}<GemmKernel>("${operation_name}"));
  }
${compile_guard_end}
"""

  #
  def emit(self, operation):
    _LOGGER.debug("*** EmitGemmConfigurationLibrary::emit(operation)")
    _LOGGER.debug("***   operation.procedural_name(): " + operation.procedural_name())
    _LOGGER.debug("***   tile_shape: " + str(operation.tile_description.tile_shape))
    _LOGGER.debug("***   warp_count: " + str(operation.tile_description.warp_count))

    opcode_class_main = operation.tile_description.math_instruction.opcode_class
    opcode_class_epi = opcode_class_main
    tile_shape = operation.tile_description.tile_shape
    warp_count = operation.tile_description.warp_count

    warp_shape = [tile_shape[idx] // warp_count[idx] for idx in range(3)]

    instruction_shape = operation.tile_description.math_instruction.instruction_shape
    cluster_m = operation.tile_description.cluster_shape[0]
    cluster_n = operation.tile_description.cluster_shape[1]

    tile_shape_main_m, tile_shape_main_n, tile_shape_main_k = tile_shape
    tile_shape_epi_m, tile_shape_epi_n, tile_shape_epi_k = tile_shape

    # account for static/dynamic cluster shapes
    cta_m = tile_shape[0] // cluster_m if cluster_m > 0 else tile_shape[0]
    cta_n = tile_shape[1] // cluster_n if cluster_n > 0 else tile_shape[1]

    # stage count set to zero indicates builder automatic stage selection
    if operation.tile_description.stages > 0:
      stage_count_string = f"cutlass::gemm::collective::StageCount<{str(operation.tile_description.stages)}>"
    else:
      stage_count_string = f"cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename {str(operation.procedural_name())}_epilogue::SharedStorage))>"

    epi_tile_mn = "cutlass::epilogue::collective::EpilogueTileAuto"

    instance_layout_A, instance_layout_B, instance_layout_C , instance_layout_D = \
      (operation.A.layout, operation.B.layout, operation.C.layout, operation.D.layout)

    # 3.0 profiler integration only supports trivial epilogues for now
    epilogue_vector_length = 1

    # Support built-in epilogue functors or user-defined functions
    if isinstance(operation.epilogue_functor, enum.Enum):
      values = {
        'element_epilogue': str(DataTypeTag[operation.element_epilogue]),
        'epilogue_functor': EpilogueFunctor3xTag[operation.epilogue_functor],
      }
      epilogue_functor = SubstituteTemplate(self.builtin_epilogue_functor_template, values)
    else:
      epilogue_functor = self.epilogue_functor.emit_declaration()
    #
    # Cutlass3x complex kernels' ElementA(B) is a tuple in collective mainloop builder, e.g. cute::tuple<Element, Transform>, Transform : cute::identity / cute::conjugate.
    element_a = DataTypeTag[operation.A.element] if not operation.is_complex() else f"cute::tuple<{str(DataTypeTag[operation.A.element])},{str(ComplexTransformTag3x[operation.A.complex_transform])}>"
    element_b = DataTypeTag[operation.B.element] if not operation.is_complex() else f"cute::tuple<{str(DataTypeTag[operation.B.element])},{str(ComplexTransformTag3x[operation.B.complex_transform])}>"
    epilogue_schedule_type = EpilogueScheduleTag[operation.epilogue_schedule]
    is_no_smem_epilogue = operation.epilogue_schedule == EpilogueScheduleType.NoSmemWarpSpecialized
    values = {
      'operation_name': operation.procedural_name(),
      'operation_suffix': self.operation_suffix,
      'element_a': element_a,
      'layout_a': LayoutTag[instance_layout_A],
      'element_b': element_b,
      'layout_b': LayoutTag[instance_layout_B],
      'element_c': DataTypeTag[operation.C.element],
      'layout_c': LayoutTag[instance_layout_C],
      'element_d': DataTypeTag[operation.D.element],
      'layout_d': LayoutTag[instance_layout_D],
      'element_accumulator': DataTypeTag[operation.accumulator_type()],
      'opcode_class_main': OpcodeClassTag[opcode_class_main],
      'opcode_class_epi': OpcodeClassTag[opcode_class_epi],
      'arch': "cutlass::arch::PPU0010",
      'tile_shape_epi_m': str(tile_shape_epi_m),
      'tile_shape_epi_n': str(tile_shape_epi_n),
      'tile_shape_epi_k': str(tile_shape_epi_k),
      'tile_shape_main_m': str(tile_shape_main_m),
      'tile_shape_main_n': str(tile_shape_main_n),
      'tile_shape_main_k': str(tile_shape_main_k),
      'warp_shape_m': str(warp_shape[0]),
      'warp_shape_n': str(warp_shape[1]),
      'warp_shape_k': str(warp_shape[2]),
      'instruction_shape_m': str(instruction_shape[0]),
      'instruction_shape_n': str(instruction_shape[1]),
      'instruction_shape_k': str(instruction_shape[2]),
      'kernel_schedule' : str(KernelScheduleTag[operation.kernel_schedule]),
      'epilogue_schedule' : str(epilogue_schedule_type),
      'epi_tile_mn' : epi_tile_mn,
      'epilogue_functor': epilogue_functor,
      'stages': stage_count_string,
      'align_a': str(operation.A.alignment),
      'align_b': str(operation.B.alignment),
      'align_c': str(operation.C.alignment),
      'align_d': str(operation.C.alignment),
      'transform_a': ComplexTransformTag[operation.A.complex_transform],
      'transform_b': ComplexTransformTag[operation.B.complex_transform],
      'math_operation': MathOperationTag[operation.tile_description.math_instruction.math_operation],
      'epilogue_vector_length': str(epilogue_vector_length),
      'element_epilogue': str(DataTypeTag[operation.element_epilogue]),
      'tile_scheduler': str(TileSchedulerTag[operation.tile_scheduler]),
    }

    return SubstituteTemplate(self.gemm_template, values)

###################################################################################################


class EmitPPU0015GemmUniversal3xInstance:
  ''' Responsible for emitting a CUTLASS 3.x template definition'''

  def __init__(self, operation_suffix = ''):
    self.operation_suffix = operation_suffix
    self.includes = [
      "cutlass/cutlass.h",
      "cutlass/numeric_types.h",
      "cutlass/epilogue/collective/default_epilogue.hpp",
      "cutlass/epilogue/thread/linear_combination.h",
      "cutlass/epilogue/collective/collective_builder.hpp",

      "ppu_include.hpp",
      "cutlass/gemm/config/gemm_operands.hpp",
      "cutlass/gemm/config/gemm_configs.hpp",
    ]
    self.builtin_epilogue_functor_template = \
"""${epilogue_functor}<
      ${element_d},
      ${element_epilogue},
      ${element_c},
      ${element_epilogue}
    >"""

    self.gemm_template = """

using namespace cute;

static constexpr cutlass::config::KernelScheduleType schedule_type 
      = cutlass::config::KernelScheduleType(int(cutlass::config::KernelScheduleType::${kernel_schedule}));

using ${operation_name}_base = typename cutlass::gemm::config::GemmKernelConfig<
    ${arch},
    ${element_a}, ${layout_a}, ${align_a},
    ${element_b}, ${layout_b}, ${align_b},
    ${element_accumulator}, ${element_c}, ${align_c},
    ${element_d}, ${align_d}, ${element_a},
    ${tile_shape_main_m}, ${tile_shape_main_n}, ${tile_shape_main_k},
    ${warp_shape_m}, ${warp_shape_n}, ${warp_shape_k},
    ${stages},
    ${epilogue_functor},
    true,
    false,
    true,
    schedule_type,
    cutlass::arch::OpClassTensorOp
  >::GemmKernel;

// Define named type
struct ${operation_name} :
  public ${operation_name}_base { };

"""
  #
  def instance_template(self):
    return """
${compile_guard_start}
  {
    using GemmKernel = cutlass::gemm::device::GemmUniversalAdapter<${operation_name}>;
    manifest.append(
      new ${gemm_kind}<GemmKernel>("${operation_name}"));
  }
${compile_guard_end}
"""

  #
  def emit(self, operation):
    _LOGGER.debug("*** EmitGemmConfigurationLibrary::emit(operation)")
    _LOGGER.debug("***   operation.procedural_name(): " + operation.procedural_name())
    _LOGGER.debug("***   tile_shape: " + str(operation.tile_description.tile_shape))
    _LOGGER.debug("***   warp_count: " + str(operation.tile_description.warp_count))

    opcode_class_main = operation.tile_description.math_instruction.opcode_class
    opcode_class_epi = opcode_class_main
    tile_shape = operation.tile_description.tile_shape
    warp_count = operation.tile_description.warp_count

    warp_shape = [tile_shape[idx] // warp_count[idx] for idx in range(3)]

    instruction_shape = operation.tile_description.math_instruction.instruction_shape
    cluster_m = operation.tile_description.cluster_shape[0]
    cluster_n = operation.tile_description.cluster_shape[1]

    tile_shape_main_m, tile_shape_main_n, tile_shape_main_k = tile_shape
    tile_shape_epi_m, tile_shape_epi_n, tile_shape_epi_k = tile_shape

    # PPU1.5 kernel schedule
    if tile_shape_main_k == 64:
      kernel_sch_str = "PERSISTENT_OVERLAP_PROEPILOGUE"
    else:
      # K32 use normal KernelAiuMultistage
      kernel_sch_str = "DEFAULT"

    # account for static/dynamic cluster shapes
    cta_m = tile_shape[0] // cluster_m if cluster_m > 0 else tile_shape[0]
    cta_n = tile_shape[1] // cluster_n if cluster_n > 0 else tile_shape[1]

    epi_tile_mn = "cutlass::epilogue::collective::EpilogueTileAuto"

    instance_layout_A, instance_layout_B, instance_layout_C , instance_layout_D = \
      (operation.A.layout, operation.B.layout, operation.C.layout, operation.D.layout)

    # 3.0 profiler integration only supports trivial epilogues for now
    epilogue_vector_length = 1

    # Support built-in epilogue functors or user-defined functions
    if isinstance(operation.epilogue_functor, enum.Enum):
      values = {
        'element_epilogue': str(DataTypeTag[operation.element_epilogue]),
        'epilogue_functor': EpilogueFunctor3xTag[operation.epilogue_functor],
      }
      epilogue_functor = SubstituteTemplate(self.builtin_epilogue_functor_template, values)
    else:
      epilogue_functor = self.epilogue_functor.emit_declaration()
    #
    # Cutlass3x complex kernels' ElementA(B) is a tuple in collective mainloop builder, e.g. cute::tuple<Element, Transform>, Transform : cute::identity / cute::conjugate.
    element_a = DataTypeTag[operation.A.element] if not operation.is_complex() else f"cute::tuple<{str(DataTypeTag[operation.A.element])},{str(ComplexTransformTag3x[operation.A.complex_transform])}>"
    element_b = DataTypeTag[operation.B.element] if not operation.is_complex() else f"cute::tuple<{str(DataTypeTag[operation.B.element])},{str(ComplexTransformTag3x[operation.B.complex_transform])}>"
    epilogue_schedule_type = EpilogueScheduleTag[operation.epilogue_schedule]
    is_no_smem_epilogue = operation.epilogue_schedule == EpilogueScheduleType.NoSmemWarpSpecialized
    values = {
      'operation_name': operation.procedural_name(),
      'operation_suffix': self.operation_suffix,
      'element_a': element_a,
      'layout_a': LayoutTag[instance_layout_A],
      'element_b': element_b,
      'layout_b': LayoutTag[instance_layout_B],
      'element_c': DataTypeTag[operation.C.element],
      'layout_c': LayoutTag[instance_layout_C],
      'element_d': DataTypeTag[operation.D.element],
      'layout_d': LayoutTag[instance_layout_D],
      'element_accumulator': DataTypeTag[operation.accumulator_type()],
      'opcode_class_main': OpcodeClassTag[opcode_class_main],
      'opcode_class_epi': OpcodeClassTag[opcode_class_epi],
      'arch': "cutlass::arch::PPU0015",
      'tile_shape_epi_m': str(tile_shape_epi_m),
      'tile_shape_epi_n': str(tile_shape_epi_n),
      'tile_shape_epi_k': str(tile_shape_epi_k),
      'tile_shape_main_m': str(tile_shape_main_m),
      'tile_shape_main_n': str(tile_shape_main_n),
      'tile_shape_main_k': str(tile_shape_main_k),
      'warp_shape_m': str(warp_shape[0]),
      'warp_shape_n': str(warp_shape[1]),
      'warp_shape_k': str(warp_shape[2]),
      'instruction_shape_m': str(instruction_shape[0]),
      'instruction_shape_n': str(instruction_shape[1]),
      'instruction_shape_k': str(instruction_shape[2]),
      'kernel_schedule' : kernel_sch_str,
      'epilogue_schedule' : str(epilogue_schedule_type),
      'epi_tile_mn' : epi_tile_mn,
      'epilogue_functor': epilogue_functor,
      'stages': str(operation.tile_description.stages),
      # aiu ld use alignment 1
      'align_a': str(1),
      'align_b': str(1),
      'align_c': str(operation.C.alignment),
      'align_d': str(operation.C.alignment),
      'transform_a': ComplexTransformTag[operation.A.complex_transform],
      'transform_b': ComplexTransformTag[operation.B.complex_transform],
      'math_operation': MathOperationTag[operation.tile_description.math_instruction.math_operation],
      'epilogue_vector_length': str(epilogue_vector_length),
      'element_epilogue': str(DataTypeTag[operation.element_epilogue]),
      'tile_scheduler': str(TileSchedulerTag[operation.tile_scheduler]),
    }

    return SubstituteTemplate(self.gemm_template, values)


###################################################################################################

#
# Emitters functions for all targets
#
###################################################################################################

class EmitGemmConfigurationLibrary:
  def __init__(self, operation_path, configuration_name):
    self.configuration_name = configuration_name
    self.configuration_path = os.path.join(operation_path, "%s.cu" % configuration_name).replace('\\', '/')

    self.instance_emitter = {
      GemmKind.Gemm: EmitGemmInstance,
      GemmKind.Universal: EmitGemmUniversalInstance,
      GemmKind.Universal3x: EmitGemmUniversal3xInstance,
      GemmKind.PPU0015Universal3x: EmitPPU0015GemmUniversal3xInstance,
    }

    self.gemm_kind_wrappers = {
      GemmKind.Gemm: 'GemmOperation',
      GemmKind.Universal: 'GemmUniversalOperation',
      # PPU supports universal3x
      GemmKind.Universal3x: 'GemmUniversal3xOperation',
      GemmKind.PPU0015Universal3x: 'GemmUniversal3xOperation',
    }

    self.separator = """
///////////////////////////////////////////////////////////////////////////////////////////////////

"""

    self.header_template = """
/*
  Generated by gemm_operation.py - Do not edit.
*/
"""

    self.initialize_function_template = """

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace library {

///////////////////////////////////////////////////////////////////////////////////////////////////

void initialize_${configuration_name}(Manifest &manifest) {

"""
    self.epilogue_template = """

}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////

"""

  def __enter__(self):
    _LOGGER.debug("*** EmitGemmConfigurationLibrary::__enter__")
    _LOGGER.debug("***   configuration_path (file to write): " +
                  str(self.configuration_path))

    self.configuration_file = open(self.configuration_path, "w")
    self.configuration_file.write(self.header_template)
    self.configuration_file.write(self.separator)

    self.includes = collections.OrderedDict([
      ("cute/tensor.hpp", None),
      ("cutlass/cutlass.h", None),
      ("cutlass/library/library.h", None),
      ("cutlass/library/manifest.h", None),
      ("library_internal.h", None),
      # gemm_operation.h based on cutlass2
      # ("gemm_operation.h", None),
      ("gemm_operation_3x.hpp", None),
      # todo: PPU1.0 support sparse gemm for profiler
      # ("sparse_gemm_operation_3x.hpp", None),
      ("cutlass/numeric_types.h", None)
    ])
    self.instance_definitions = []
    self.instance_wrappers = []

    self.operations = []
    return self

  def emit(self, operation):
    _LOGGER.debug("*** EmitGemmConfigurationLibrary::emit(operation)")
    _LOGGER.debug("***   operation.gemm_kind: " + str(operation.gemm_kind))

    emitter = self.instance_emitter[operation.gemm_kind]()

    for incl in emitter.includes:
      self.includes[incl] = None

    self.operations.append(operation)

    self.instance_definitions.append(emitter.emit(operation))

    self.instance_wrappers.append(SubstituteTemplate(emitter.instance_template(), {
      'configuration_name': self.configuration_name,
      'operation_name': operation.procedural_name(),
      'gemm_kind': self.gemm_kind_wrappers[operation.gemm_kind],
      'compile_guard_start': "",
      'compile_guard_end': ""
      }))

  def __exit__(self, exception_type, exception_value, traceback):

    # Write includes
    for incl, _ in self.includes.items():
      include_statement = "#include \"%s\"\n" % incl
      self.configuration_file.write(include_statement)

    self.configuration_file.write(self.separator)

    # Write instance definitions in top-level namespace
    for instance_definition in self.instance_definitions:
      self.configuration_file.write(instance_definition)

    # Add wrapper objects within initialize() function
    self.configuration_file.write(SubstituteTemplate(self.initialize_function_template, {
      'configuration_name': self.configuration_name
      }))

    for instance_wrapper in self.instance_wrappers:
      self.configuration_file.write(instance_wrapper)

    self.configuration_file.write(self.epilogue_template)
    self.configuration_file.close()

###################################################################################################
###################################################################################################
