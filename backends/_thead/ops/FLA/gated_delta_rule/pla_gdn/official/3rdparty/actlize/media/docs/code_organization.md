# ACTLIZE Code Organization

This document describes the layout of the ACTLIZE repository. The main components are:

* **Template Library** - Device-side templates for Linear Algebra Subroutines and Solvers (PPU, header only)
* **CuTe Template Library** - Core vocabulary layout type and associated algebra (header only)
* **Utilities** - Additional templates 
* **Instance Library** - instantiations of ACTLIZE templates covering the design space
* **Profiler** - Library, Profiler, and Utilities
* **Examples** - Examples of Template Library and components
* **Media** - supporting documentation and media content
* **Tests** - test components for Template Library and tools

## Template Library

Device-side templates for Linear Algebra Subroutines and Solvers is a library of device-side C++ template classes for
performing efficient matrix computations on PPU devices.

Like upstream HGB-style libraries, the components of ACTLIZE are organized hierarchically based on the scope of cooperative
elements. For example, warp-level GEMM components perform a matrix multiply collectively by the
set of threads within a warp. The following figure illustrates each layer.

Components are designed to be usable by client applications accessing functionailty at each scope.

Templates are implemented by header files in the following directory structure:

```
include/                     # Top-level include directory. Client applications should target this path.
  cutlass/                   # Device-side templates for Linear Algebra Subroutines and Solvers (PPU) - headers only

    arch/                    # direct exposure of architecture features (including instruction-level GEMMs)
      *
    gemm/                    # code specialized for general matrix product computations
      thread/                #   thread-level operators
      warp/                  #   warp-level operators
      collective/            #   API operators for all threads a tiled mma/copy are built over
      threadblock/           #   CTA-level operators
      kernel/                #   device kernel entry points
      device/                #   launches kernel(s) over a full device
      *                      # scope-agnostic components and basic vocabulary type definitions for GEMM

    layout/                  # layout definitions for matrices, tensors, and other mathematical objects in memory
      *

    reduction/               # bandwidth-limited reduction kernels that do not fit the "gemm" models
      thread/                #   thread-level operators
      kernel/                #   device kernel entry points
      device/                #   launches kernel(s) over a full device
      *                      # scope-agnostic components and basic vocabulary type definitions

    transform/               # code specialized for layout, type, and domain transformations
      thread/                #   thread-level operators
      warp/                  #   warp-level operators
      threadblock/           #   CTA-level operators
      kernel/                #   device kernel entry points
      device/                #   launches kernel(s) over a full device
      *                      # scope-agnostic components and basic vocabulary type definitions
    *                        # core vocabulary types and fundamental arithmetic operators

  cute /                     # CuTe Layout, layout algebra, MMA/Copy atoms, tiled MMA/Copy
    algorithm/               # Definitions of core operations such as copy, gemm, and operations on cute::tuples
    arch/                    # Bare bones PTX wrapper structs for copy and math instructions
    atom/                    # Meta-information either link to or built from arch/ operators
      mma_atom.hpp           # cute::Mma_Atom and cute::TiledMma
      copy_atom.hpp          # cute::Copy_Atom and cute::TiledCopy
      *traits_*.hpp          # Arch specific meta-information for copy and math operations
    container/               # Core container types used across CuTe, namely, cute::tuple
    numeric/                 # CuTe's internal numerics implementation
    *                        # Core library types such as Shape, Stride, Layout, Tensor, and associated operations

  ppu_include.hpp            # Main PPU header (includes all PPU components)
  accutlass.hpp              # Legacy compatibility header
```

## CuTe

CuTe is a collection of C++ device-side template abstractions for defining and operating on hierarchically multidimensional layouts of threads and data. CuTe provides `Layout` and `Tensor` objects that compactly packages the type, shape, memory space, and layout of data, while performing the complicated indexing for the user. This lets programmers focus on the logical descriptions of their algorithms while CuTe does the mechanical bookkeeping for them. With these tools, we can quickly design, implement, and modify all dense linear algebra operations.

## Tools

The `tools/` directory contains clients of the Template library and includes the following.

## Instance Library

The Instance Library contains instantiations of the above templates covering supported configurations,
data types, block structure, and tile sizes. These instantiations are procedurally generated using a set of 
scripts to span the design space.

```
tools/
  library/                   # static/dynamic library containing all kernel instantiations of interest
                             # (with some build-level filter switches to compile specific subsets)

    include/
      cutlass/
        library/             # header files for Deliverables Library (in cutlass::library:: namespace)

          handle.h           # implements a host-side API for launching kernels, similar to upstream device BLAS handles
          library.h          # defines enums and structs to describe the tiled structure of operator instances          
          manifest.h         # collection of all instances

    src/

python/
    cutlass_library/       # scripts to procedurally generate template instances

      gemm_operations.py
      library.py
      generator.py            # entry point of procedural generation scripts - invoked by cmake
      manifest.py
```

When CMake is executed, the Instance Library generator scripts are executed to construct a set of
instantiations in `build/tools/library/generated/`.

### Profiler

The Profiler is designed to load the Instance Library and execute all operations contained therein.
This command-line driven application constructs an execution environment for evaluating functionality and performance. 
It is implemented in
```
tools/
  profiler/
```

and may be built as follows.
```
$ make cutlass_profiler -j
```

[Further details about the Profiler are described here.](/media/docs/profiler.md)

### Utilities

`tools/util/` defines a companion library of headers and sources that support the test programs, examples, and other client applications. Its structure is as follows:

```
tools/
  util/
    include/
      cutlass/
        util/                   # Utility companion library

          reference/            #  functional reference implementation of operators
                                #    (minimal consideration for performance)
            
            detail/
              *

            device/             #  device-side reference implementations of operators
              thread/
              kernel/
                *
            host/               #  host-side reference implementations of operators
              *
          *
```



## Examples

To demonstrate components, several examples are implemented in `examples/`. 

ACTLIZE examples apply templates to implement basic computations.

```
examples/
  00_basic_gemm/             # launches a basic GEMM with single precision inputs and outputs

  01_cutlass_utilities/      # demonstrates Utilities for allocating and initializing tensors

  02_tile_iterator/          # example demonstrating an iterator over tiles in memory

  03_batched_gemm/           # example demonstrating batched strided GEMM operation

```



## Media

This directory contains documentation and images.

## Tests

Test programs for ACTLIZE. Tests are organized hierarchically, mirroring the organization of source files.
```
test/                        # unit tests for Template Library
  unit/
    core/
    gemm_ppu/                # gemm tests for PPU platform
      *
```
Tests can be built and run at the top level scope by invoking `make test_unit` or by building
and explicitly executing each individual target, e.g. `cutlass_test_unit_gemm_device`.

Tests are configured to specify appropriate GTest filter strings to avoid running except on
architectures where they are expected to pass. Thus, no tests should fail. The actual number
of tests run may vary over time as more are added.

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
