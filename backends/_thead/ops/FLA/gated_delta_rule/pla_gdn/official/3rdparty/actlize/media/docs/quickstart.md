# Quickstart for PPU

## Prerequisites

- **PPU Hardware:** ZW 610 / 610E / 810 / 810E / M890
- **PPU SDK:** with `bin/hgcc` (PPU device compiler) installed under `${PPU_SDK}` (default `/usr/local/PPU_SDK`)
- **CMake:** 3.19+
- **Python 3.6+**

For detailed SDK installation, driver setup, and compiler usage guide, please visit the
[PPU Developer Portal](https://developer.t-head.cn/docs_center/doc_list/index.html).

## Initial Build Steps for PPU

Construct a build directory and run CMake with PPU-specific architecture flags.

### For PPU 1.0 in a PPU SDK env:

```bash
$ mkdir build && cd build

$ cmake .. -DCUTLASS_PPU_ARCHS=ppu0010 \
           -DCUTLASS_ENABLE_LIBRARY=ON \
           -DCUTLASS_ENABLE_PROFILER=ON
```

### For PPU 1.5 in a PPU SDK env:

```bash
$ mkdir build && cd build

$ cmake .. -DCUTLASS_PPU_ARCHS=ppu0015 \
           -DCUTLASS_ENABLE_LIBRARY=ON \
           -DCUTLASS_ENABLE_PROFILER=ON
```

### For Both Architectures:

```bash
$ cmake .. -DCUTLASS_PPU_ARCHS="ppu0010;ppu0015" \
           -DCUTLASS_ENABLE_LIBRARY=ON \
           -DCUTLASS_ENABLE_PROFILER=ON
```

## Build and Run the Profiler

From the `build/` directory created above, compile the Profiler.

```bash
$ make cutlass_profiler -j12
```

### Running a Basic GEMM

Execute the Profiler to compute GEMM with PPU-specific kernels:
(ppu profiler only supports gemm currently)

```bash
$ ./tools/profiler/cutlass_profiler --operation=gemm --m=3456 --n=4096 --k=8192
```

See [documentation for the Profiler](profiler.md) for more details.


## Build and Run Unit Tests

From the `build/` directory created above, simply build the target `test_unit` to compile and run
all unit tests.

```bash
$ make test_unit -j
```

For PPU-specific tensor-op gemm tests:

```bash
$ make test_unit_gemm_ppu -j
```

The unit tests are arranged hierarchically mirroring the Template Library. This enables
parallelism in building and running tests as well as reducing compilation times when a specific
set of tests are desired.

### Running Specific PPU Unit Tests

```bash
# Run FP16 GEMM test
./test/unit/gemm_ppu/device/test_ppu0010_gemm_f16_f16_f32_tensor_op_f32

# Run INT8 GEMM test
./test/unit/gemm_ppu/device/test_ppu0010_gemm_s8_s8_s32_tensor_op
```

## Build and Run Examples

From the `build/` directory created above, simply build the target `cutlass_examples` to compile all examples.

```bash
$ make cutlass_examples -j8
```

For PPU basic tensor-op gemm tests:

```bash
$ make 08_ppu_basic_tensor_op_gemm -j
```

For PPU1.5-specific FP4 tensor-op gemm tests:

```bash
$ make 101_ppu_fp4_gemm -j
```

### Running Specific PPU Examples/Unit Tests

```bash
# PPU1.0 examples/unit tests
$ cd ..
$ ./run.sh 10000

# PPU1.5 examples/unit tests
$ cd ..
$ ./run.sh 10500
```


## Using ACTLIZE within Other Applications

Applications should list [`/include`](/include) within their include paths. They must be
compiled as C++17 or greater.

**Example:** Print the contents of a variable storing half-precision data.

```c++
#include <iostream>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/core_io.h>

int main() {
  cutlass::half_t x = 2.25_hf;

  std::cout << x << std::endl;

  return 0;
}
```

## Launching a GEMM Kernel Using ACTLIZE 1.0.0 for PPU

**Example:** See the PPU-specific examples in `examples/*_ppu_*` directories.

### Basic Tensor Operation GEMM

```c++
#include <ppu_include.hpp>

// Define problem size
int M = 512;
int N = 256;
int K = 128;

static constexpr bool UseAIU = true;
// Define data types
using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;
using ElementAccumulator = float;

using OperatorClass       = cutlass::arch::OpClassTensorOp;                 // Operator class tag
using BlockK = cute::conditional_t<UseAIU, _32, _16>;
using TileShape           = Shape<_128,_128, BlockK>;                       // Threadblock-level tile size
using WarpShape           = Shape<_64,_64, BlockK>;

using StageCountType = cutlass::gemm::collective::StageCountAuto;           // Stage count maximized based on the tile size
using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;       // Kernel to launch based on the default setting in the Collective Builder

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::PPU0010, cutlass::arch::OpClassTensorOp,
    TileShape, WarpShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, AlignmentC,
    ElementC, LayoutC, AlignmentC,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::PPU0010, OperatorClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape, WarpShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int>, // Indicates ProblemShape
    CollectiveMainloop,
    CollectiveEpilogue
>;

// Wrap with device adapter
using GemmHandle = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
```

## Building Only PPU Kernels

To reduce build times, you can build only PPU kernels:

```bash
# PPU 1.0 only
cmake .. -DCUTLASS_PPU_ARCHS=ppu0010 \
         -DCUTLASS_LIBRARY_KERNELS="ppu0010*"

# PPU 1.5 only
cmake .. -DCUTLASS_PPU_ARCHS=ppu0015 \
         -DCUTLASS_LIBRARY_KERNELS="ppu0015*"

# Both architectures
cmake .. -DCUTLASS_PPU_ARCHS="ppu0010;ppu0015" \
         -DCUTLASS_LIBRARY_KERNELS="ppu*"
```

## Building Library for PPU

To build the library with PPU kernels:

```bash
cmake .. -DCUTLASS_PPU_ARCHS=ppu0010 \
         -DCUTLASS_ENABLE_LIBRARY=ON \
         -DCUTLASS_LIBRARY_KERNELS="ppu0010*"

make cutlass_lib -j16
```

## Troubleshooting

### Common Build Issues

**Issue:** `Invalid value for CUTLASS_PPU_ARCHS`

**Solution:** Use only allowed architectures: `ppu0010`, `ppu0015`
```bash
cmake .. -DCUTLASS_PPU_ARCHS=ppu0010  # For PPU 1.0
cmake .. -DCUTLASS_PPU_ARCHS=ppu0015  # For PPU 1.5
```

**Issue:** `PPU SDK not found`

**Solution:** Source the PPU SDK env or set `PPU_SDK` to the SDK root that
contains `bin/hgcc`.

**Issue:** `Out of shared memory`

**Solution:** Reduce tile sizes or pipeline stages
```bash
# Use smaller tile size
--cta_m=128 --cta_n=128 --cta_k=32

# Use fewer stages
--stages=2
```

### Verifying Build

```bash
# Check if PPU kernels were built
grep -r "ppu0010" build/tools/library/generated/

# Run a basic test on ppu1.0
./test/unit/gemm_ppu/device/test_ppu0010_gemm_f16_f16_f32_tensor_op_f32
```

## Next Steps

- [Functionality](./functionality.md) - See supported operations
- [Examples](../../examples/) - Browse PPU-specific examples
- [Performance Profiling](./profiler.md) - Learn about profiling

# Copyright

Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD. All rights reserved.
Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause

```
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
