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
Utilities for enumerating CUTLASS3x library kernels
"""

import argparse
import enum
from itertools import product
import math
import logging
import os.path
import shutil
import sys
import copy
from typing import Any, Optional, Sequence, Tuple

try:
  import builtins
  if hasattr(builtins, "CUTLASS_IGNORE_PACKAGE") and CUTLASS_IGNORE_PACKAGE == True:
    raise ImportError("Disabling attempt to import cutlass_library")
  from cutlass_library.library import *
except ImportError:
  from library import *

###########

#### generate tile descriptions from math instruction shapes

def generate_fp16_tile_descriptions_ppu(compute_capabilities, math_instructions):
    tile_descriptions = set()
    math_inst = math_instructions[0]

    for cap in compute_capabilities:
        # not support cluster
        # PPU1.0 tiles
        if cap == 80:
            tile_descriptions.update ([
            # stage 2
            TileDescription([256, 128, 64],  2, [4, 2, 1], math_inst, 80, 80),
            TileDescription([128, 256, 64],  2, [2, 4, 1], math_inst, 80, 80),
            TileDescription([128, 128, 64],  2, [4, 2, 1], math_inst, 80, 80),
            # TileDescription([256,  64, 64],  2, [4, 2, 1], math_inst, 80, 80),
            # TileDescription([ 64, 256, 64],  2, [2, 4, 1], math_inst, 80, 80),
            # TileDescription([128,  64, 64],  2, [4, 2, 1], math_inst, 80, 80),
            # TileDescription([ 64, 128, 64],  2, [2, 4, 1], math_inst, 80, 80),
            # TileDescription([ 64,  64, 64],  2, [2, 2, 1], math_inst, 80, 80),

            # stage3
            # TileDescription([256,  64, 64],  3, [4, 2, 1], math_inst, 80, 80),
            # TileDescription([64,  256, 64],  3, [2, 4, 1], math_inst, 80, 80),
            TileDescription([128,  64, 64],  3, [4, 2, 1], math_inst, 80, 80),
            TileDescription([64,  128, 64],  3, [2, 4, 1], math_inst, 80, 80),
            ])
        # PPU1.5 tiles
        if cap == 89:
            tile_descriptions.update ([
            TileDescription([256, 256, 64],  4, [4, 4, 1], math_inst, 80, 89),
            TileDescription([512, 128, 64],  3, [8, 2, 1], math_inst, 80, 89),
            TileDescription([256, 128, 32],  4, [4, 2, 1], math_inst, 80, 89),
            TileDescription([128, 128, 32],  4, [4, 2, 1], math_inst, 80, 89),
            TileDescription([128, 512, 64],  3, [2, 8, 1], math_inst, 80, 89),
            TileDescription([128, 256, 32],  4, [2, 4, 1], math_inst, 80, 89),
            # TileDescription([128, 64,  64],  4, [4, 2, 1], math_inst, 80, 89),
            # TileDescription([64, 128,  64],  3, [2, 4, 1], math_inst, 80, 89),

            # memory bound
            TileDescription([128, 128, 64],  2, [4, 2, 1], math_inst, 80, 89),
            ])

    return tile_descriptions

#### Step 3: map tile description to valid schedules

def is_tile_desc_compatible_with_cooperative(tile_description):
    # Cooperative kernels require a minimum CTA-M of 128
    return tile_description.threadblock_shape[0] >= 128


def can_tile_desc_use_shmem_in_epilogue(tile_description, data_types):
    dtype_a, dtype_b, dtype_c, dtype_d, dtype_acc, dtype_epi = (
        data_types["a_type"],
        data_types["b_type"],
        data_types["c_type"],
        data_types["d_type"],
        data_types["acc_type"],
        data_types["epi_type"]
    )
    mn = tile_description.threadblock_shape[0] * tile_description.threadblock_shape[1]
    bitsize_c, bitsize_d = DataTypeSize[dtype_c], DataTypeSize[dtype_d]

    shmem_bits_c, shmem_bits_d = bitsize_c * mn, bitsize_d * mn
    shmem_bits_total = shmem_bits_c + shmem_bits_d
    # Magic number: 2^20
    # Existing logic suggested that tile shape 256x128 (or 128x256)
    # would run out of shmem if D is FP32, and source is needed.
    # That would be 256 * 128 * 32 == 2^21 (~262 KB), which is over the limit.
    # max shmem size is 228 KB, and 2^20 ~= 131 KB.
    # Since epilogue can't possibly use ALL of the shmem available
    # we can just settle on 2^20 bits (~ 131 KB) being the upper bound
    # we would allow for epilogue.
    # This can be different for non-persistent kernels where epilogue and
    # mainloop shmem is shared.
    if shmem_bits_total > 2 ** 20:
        return False

    return True


def get_valid_schedules(tile_description, is_aligned, data_types, layout,
                        instantiation_level, enable_fp8_fast_acc=True):
    schedules = []
    stream_k_schedules = []

    if tile_description.maximum_compute_capability == 80:
        # by default for PPU1.0 profiler
        schedules.append([
            KernelScheduleType.AiuMultistage,
            EpilogueScheduleType.EpilogueSimtVectorized
        ])
        schedules.append([
            KernelScheduleType.AiuMultistage,
            EpilogueScheduleType.NoSmemWarpSpecialized
        ])
        # stream_k_schedules.append([
        #     KernelScheduleType.KernelAiuMultistageStreamK,
        #     EpilogueScheduleType.EpilogueSimtVectorized
        # ])
    elif tile_description.maximum_compute_capability == 89:
        # PPU1.5 scheduler determined in EmitPPU0015GemmUniversal3xInstance
        schedules.append([
            KernelScheduleType.AiuMultistageOverlapPrologue,
            EpilogueScheduleType.EpilogueSimtVectorized
        ])
        schedules.append([
            KernelScheduleType.AiuMultistageOverlapPrologue,
            EpilogueScheduleType.NoSmemWarpSpecialized
        ])
        # stream_k_schedules.append([
        #     KernelScheduleType.KernelAiuMultistageStreamK,
        #     EpilogueScheduleType.EpilogueSimtVectorized
        # ])

    return schedules, stream_k_schedules


#### Misc: helpers

def generate_data_types_from_math_instruction(math_instruction, element_source = None, element_dest = None, element_epilogue = None):
    element_a, element_b = math_instruction.element_a, math_instruction.element_b
    element_accumulator = math_instruction.element_accumulator
    element_c = element_source or element_accumulator
    element_d = element_dest or element_accumulator
    element_epilogue = element_epilogue or element_accumulator
    data_types = {
        "a_type"   : element_a,
        "b_type"   : element_b,
        "c_type"   : element_c,
        "d_type"   : element_d,
        "acc_type" : element_accumulator,
        "epi_type" : element_epilogue
    }
    return data_types

def fix_alignments(data_types, layout, alignment_bits = 128):
    operand_keys = ["a_type", "b_type", "c_type"]
    operands_to_fix = ["c_type"]
    new_layout = []
    assert len(layout) == len(operand_keys)
    for i, k in enumerate(operand_keys):
        assert k in data_types and data_types[k] in DataTypeSize
        dtype = data_types[k]
        dtype_size_bits = DataTypeSize[dtype]

        layout_type = layout[i][0]
        layout_alignment = layout[i][1]

        # Don't modify alignment if dtype's been changed to void
        if k in operands_to_fix and dtype_size_bits >= 1:
            layout_alignment = alignment_bits // dtype_size_bits

        new_layout.append([layout_type, layout_alignment])

    return new_layout
