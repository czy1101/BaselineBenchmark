# ACTLIZE Profiler

The ACTLIZE Profiler is a command-line driven test and profiling environment for ACTLIZE computations
defined in the ACTLIZE Instance Library. PPU ACTLIZE Profiler only supports GEMM kernel currently.

The ACTLIZE Profiler may be compiled with:
```bash
$ make cutlass_profiler -j
```

To limit compilation time, one may choose one or several tiles of tile_descriptions in python/cutlass_library/ppu_utils.py. To instantiate all sizes, set the following environment variable when running CMake from an
empty `build/` directory.
```bash
$ cmake .. -DCUTLASS_PPU_ARCHS=ppu0010 -DCUTLASS_LIBRARY_KERNELS=all
...
$ make cutlass_profiler -j
```


The ACTLIZE Profiler sources are stored in:

```bash
tools/
  profiler/
```

The ACTLIZE Profiler usage statement may be obtained by executing `cutlass_profiler --help` and appears as follows.
```bash
Performance Tool
usage:

    cutlass_profiler [options]

  --help

  --mode=<string>                                  Cutlass profiler execution mode.
                                                    --mode=profile    regular verification and profiling (default)
                                                    --mode=dry_run    no kernels are launched or workspaces allocated
                                                    --mode=enumerate  lists all operation kind and operations
                                                    --mode=trace      executes a single device-side computation with
                                                                       no other kernel launches

  --device-info                                    Prints information on all PPUs present in the system

  --operation=<operation_kind>                     Operation to profile.

  --kernels=<string_list>                          Filter operations by kernel names. For example, call all kernels with
                                                   ("s1688" and "nt") or ("s844" and "tn" and "align8") in their
                                                   operation name using --kernels="s1688*nt, s884*tn*align8"

  --ignore-kernels=<string_list>                   Excludes kernels whose names match anything in this list.

Device:
  --device=<int>                                   Device ID

  --compute-capability=<int>                       Override the compute capability.

  --llc-capacity=<capacity in KiB>                 Capacity of last-level cache in kilobytes. If this is non-zero,
                                                   profiling phases cycle through different input tensors to induce
                                                   capacity misses in the L2.

  --allocations=<name>:<device>,<name>:<device>    Pairs of allocation names to devices. If <device> is negative,
                                                   the execution device is used


Initialization:
  --initialization=<bool>                          Enables initialization (default: true). If false, device memory is
                                                   not initialized after allocation.

  --initialization-provider=<provider>             Selects initialization provider {host, device*}. (default: '*')

  --dist=<distribution>                            Data distribution of input tensors {uniform*, gaussian, identity, sequential}
                                                    --dist=uniform,min:<double>,max:<double>,scale:<integer>
                                                    --dist=gaussian,mean:<double>,stddev:<double>,scale:<integer>
                                                    --dist=sequential,start:<double>,delta:<double>,scale:<integer>
                                                    --dist=identity

  --seed=<int>                                     Random number generator seed. Used to enforce deterministic
                                                   initialization.


Library:
  --library-algo-mode=<mode>                       Indicates algorithm mode used to call libraries such as acBLAS and acDNN.
                                                   mode={default*,matching,best}

  --library-algos=<range-list>                     If --algorithm-mode=best, permits specifying a selection of algorithms.


Profiling:
  --workspace-count=<workspace count>              Number of discrete workspaces maintained to avoid cache-resident
                                                 If zero (default), the amount is chosen for each workload based on
                                                 capacity of the last-level cache.

  --profiling-iterations=<iterations>              Number of iterations to profile each kernel. If zero, kernels
                                                   are launched up to the profiling duration.

  --warmup-iterations=<iterations>                 Number of iterations to execute each kernel prior to profiling.

  --sleep-duration=<duration>                      Number of ms to sleep between profiling periods (ms).

  --profiling-enabled=<bool>                       If true, profiling is actually conducted.

Verification:
  --verification-enabled=<bool>                    Whether to perform verification checks.

  --epsilon=<error>                                Error threshold. Setting to zero (default) requires
                                                   bit-level equivalence.

  --nonzero-floor=<floor>                          Results whose absolute value is less than this quantity
                                                   are treated as zero for comparisons.

  --save-workspace=<string>                        Specifies when to save the GEMM inputs and results to the filesystem.
                                                    --save-workspace=never      never save workspace (default)
                                                    --save-workspace=incorrect  save workspace for incorrect results
                                                    --save-workspace=always     always save workspace

  --verification-providers=<providers>             List of providers used to verify result. (default: '*')
                                                   Gemm verification-providers {acblas*}
                                                   Conv2d verification-providers {acdnn*, device*, host}


Report:
  --append=<bool>                                  If true, result is appended to possibly existing file. Otherwise,
                                                   any existing file is overwritten.

  --output=<path>                                  Path to output file for machine readable results. Operation kind and '.csv' is appended.

  --junit-output=<path>                            Path to junit output file for result reporting. Operation kind and '.junit.xml' is appended.

  --report-not-run=<bool>                          If true, reports the status of all kernels including those that
                                                   do not satisfy the given arguments.

  --tags=<column:tag,...>                          Inserts leading columns in output table and uniform values for each
                                                   column. Useful for generating pivot tables.

  --verbose=<bool>                                 Prints human-readable text to stdout. If false, nothing is written to stdout.



# currently supports tensor-op gemm
Operations:

     gemm                                          General matrix-matrix product. D = alpha * A*B + beta * C


For details about a particular function, specify the function name with --help.

Example:

  $ cutlass_profiler --operation=Gemm --help
```

# GEMM

The ACTLIZE Profiler is capable of executing GEMM problems.

## GEMM Arguments

The complete set of arguments available to each operation may be viewed by specifying the operation name
in addition to `--help`. The argument flags and their aliases usable for GEMM appear as follows.

```bash
$ ./tools/profiler/cutlass_profiler --operation=gemm --help

GEMM

  [enum]      --gemm_kind                                       Variant of GEMM (e.g. universal, gemm, planar_complex, planar_complex_array)
  [int]       --m,--problem-size::m                             M dimension of the GEMM problem space
  [int]       --n,--problem-size::n                             N dimension of the GEMM problem space
  [int]       --k,--problem-size::k                             K dimension of the GEMM problem space
  [tensor]    --A                                               Tensor storing the A operand
  [tensor]    --B                                               Tensor storing the B operand
  [tensor]    --C                                               Tensor storing the C operand
  [scalar]    --alpha,--epilogue::alpha                         Epilogue scalar alpha
  [scalar]    --beta,--epilogue::beta                           Epilogue scalar beta
  [enum]      --split_k_mode,--split-k-mode                     Variant of split K mode(serial, parallel)
  [int]       --split_k_slices,--split-k-slices                 Number of partitions of K dimension
  [int]       --batch_count,--batch-count                       Number of GEMMs computed in one batch
  [enum]      --op_class,--opcode-class                         Class of math instruction (simt, tensorop, wmmatensorop, wmma).
  [enum]      --accum,--accumulator-type                        Math instruction accumulator data type
  [int]       --cta_m,--threadblock-shape::m                    Threadblock shape in the M dimension
  [int]       --cta_n,--threadblock-shape::n                    Threadblock shape in the N dimension
  [int]       --cta_k,--threadblock-shape::k                    Threadblock shape in the K dimension
  [int]       --stages,--threadblock-stages                     Number of stages of threadblock-scoped matrix multiply
  [int]       --warps_m,--warp-count::m                         Number of warps within threadblock along the M dimension
  [int]       --warps_n,--warp-count::n                         Number of warps within threadblock along the N dimension
  [int]       --warps_k,--warp-count::k                         Number of warps within threadblock along the K dimension
  [int]       --inst_m,--instruction-shape::m                   Math instruction shape in the M dimension
  [int]       --inst_n,--instruction-shape::n                   Math instruction shape in the N dimension
  [int]       --inst_k,--instruction-shape::k                   Math instruction shape in the K dimension
  [int]       --min_cc,--minimum-compute-capability             Minimum device compute capability
  [int]       --max_cc,--maximum-compute-capability             Maximum device compute capability
  [enum]      --raster_order={heuristic|H|along_m|M|along_n|N}  If supported by kernel, sets the tile raster direction
  [int]       --swizzle_size={1,2,4,8}                          If supported by kernel, sets the 2D tile swizzle extent
Examples:

Profile a particular problem size:
  $ cutlass_profiler --operation=Gemm --m=1024 --n=1024 --k=128

Schmoo over problem size and beta:
  $ cutlass_profiler --operation=Gemm --m=1024:4096:256 --n=1024:4096:256 --k=128:8192:128 --beta=0,1,2.5

Schmoo over accumulator types:
  $ cutlass_profiler --operation=Gemm --accumulator-type=f16,f32

Run when A is f16 with column-major and B is any datatype with row-major (For column major, use column, col, or n. For row major use, row or t):
  $ cutlass_profiler --operation=Gemm --A=f16:column --B=*:row

Using various input value distribution:
  $ cutlass_profiler --operation=Gemm --dist=uniform,min:0,max:3
  $ cutlass_profiler --operation=Gemm --dist=gaussian,mean:0,stddev:3
  $ cutlass_profiler --operation=Gemm --dist=sequential,start:0,delta:1

Using ACTLIZE 1.0.0 GEMM kernel with a tile scheduler that supports runtime tile remapping and raster mode order:
  $ cutlass_profiler --operation=Gemm --m=2048 --n=2048 --k=2048 --raster_order=M --swizzle_size=2

Run a kernel with cta tile size of 256x128x32 and save workspace if results are incorrect (note that --cta-tile::k=32 is default cta-tile size):
 $ cutlass_profiler --operation=Gemm --cta_m=256 --cta_n=128  --cta_k=32 --save-workspace=incorrect

Test your changes to gemm kernels with a quick functional test and save results in functional-test.csv:
 $ cutlass_profiler  --operation=Gemm \
   --m=8,56,120,136,256,264,512,520,1024,1032,4096,8192,16384 \
   --n=8,56,120,136,256,264,512,520,1024,1032,4096,8192,16384 \
   --k=8,16,32,64,128,256,288,384,504,512,520 \
   --beta=0,1,2 --profiling-iterations=1 \
   --providers=cutlass --output=functional-test.csv

Profile when execution is performed on device 0 and the C tensor is located on a device 1 and D on device 2:
  $ cutlass_profiler --device=0 --allocations=C:1,D:2 --operation=Gemm --m=1024 --n=1024 --k=128
```


## Example Tensor Cell GEMM Operations

To execute kernels targeting Tensor Cell operations, supply the flag `--op_class=tensorop` in the command line.
```bash
$ ./tools/profiler/cutlass_profiler --op_class=tensorop --m=3456 --n=4096 --k=8192


=============================
  Problem ID: 1

        Provider: CUTLASS
   OperationKind: gemm
       Operation: cutlass3x_ppu0010_tensorop_s16x16x16gemm_f16_f16_f32_f32_f32_128x256x64_2_nnn_align8_aiu_multistage_epi_nosmem

          Status: Success
    Verification: ON
     Disposition: Passed

reference_device: Passed
          acBLAS: Not run
           acDNN: Not run

       Arguments: --gemm_kind=universal --m=3456 --n=4096 --k=8192 --A=f16:column --B=f16:column --C=f32:column --D=f32:column  \
                  --alpha=1 --beta=0 --split_k_mode=serial --split_k_slices=1 --batch_count=1 --raster_order=heuristic  \
                  --swizzle_size=1 --op_class=tensorop --accum=f32 --cta_m=128 --cta_n=256 --cta_k=64 --cluster_m=1 --cluster_n=1  \
                  --cluster_k=1 --stages=2 --warps_m=4 --warps_n=2 --warps_k=1 --inst_m=16 --inst_n=16 --inst_k=16 --min_cc=80  \
                  --max_cc=80

           Bytes: 180355072  bytes
           FLOPs: 231956545536  flops
           FLOPs/Byte: 1286

         Runtime: 2.37604  ms
          Memory: 70.6928 GiB/s

            Math: 97623.3 GFLOP/s
```

## Covering the problem space

All arguments may have single values or comma-delimited set of values. Integers may also be specified
as an inclusive range with the following syntax `start:end:increment` or simply `start:end`.

For example, the following sweeps over the range of the GEMM K dimension from 8 to 4096 in increments
of 8 elements.

```bash
$ ./tools/profiler/cutlass_profiler --kernels=cutlass3x_ppu0010_tensorop_s16x16x16gemm_f16_f16_f32_f32_f32_128x64x64_2_nnn_align8_aiu_multistage_simt_vectorized --m=4352 --n=4096 --k=8:4096:8
```

## Output

By default, runtime and computed GFLOP/s are reported for each operation and problem size. Additionally,
a table of comma separated values are reported at the end of the execution. This may be output to a file
with the `--output=<filename.csv>` command line option as shown:

```bash
$ ./tools/profiler/cutlass_profiler --kernels=cutlass3x_ppu0010_tensorop_s16x16x16gemm_f16_f16_f32_f32_f32_128x64x64_2_nnn_align8_aiu_multistage_simt_vectorized            \
                                    --m=3456 --n=4096 --k=8:4096:8 --output=report.csv
```

To faclitate generation of pivot tables and charts, additional columns may be prepended with the
`--tags=<column>:<value>` option. One or more tags may be specified using a comma-delimited list.

```bash
$ ./tools/profiler/cutlass_profiler --kernels=cutlass3x_ppu0010_tensorop_s16x16x16gemm_f16_f16_f32_f32_f32_128x64x64_2_nnn_align8_aiu_multistage_simt_vectorized            \
                                    --m=3456 --n=4096 --k=8:4096:8 --output=report.csv \
                                    --tags=cutlass:2.2,date:2020-06-08
```

## ACTLIZE 1.0.0 GEMM procedural names.

To best illustrate this naming convention, we will walk through the meaning of each of the components
in a GEMM kernel used by the profiler:

```
cutlass3x_ppu0015_tensorop_s16x16x16gemm_f16_f16_f32_f16_f16_128x128x32_4_nnn_align8_aiu_multistage_overlap_prologue_simt_vectorized.cu
```

The components within this name are as follows:

* `PPU0015`: indicates that the kernel target platform PPU0015
* `tensorop`: indicates that the kernel makes use of Tensor Cells
(as opposed to `simt`, which indicates the use of "Vector cores")
* `s`: indicates that the Tensor Cell instruction being used accumulates in single precision
(as opposed to `h`, which indicates half precision)
* `16x16x16gemm`: indicates that the shape of the Tensor Cell instruction being used (MxNxK) is 16x16x16
* `f16_f16_f32_f16_f16`: indicates that the data types for operands A, B, Accumulator, C and D (in that order).
* `128x128x32`: indicates that the thread block shape used in the GEMM (MxNxK) is 128x128x32
* `4`: indicates the number of mainloop pipeline stages to be used.
* `nnn`: indicates that the layouts for operands A, B, and C are column major ("n"; non-transposed),
row major ("t"; transposed), and column major, respectively.
* `align8`: indicates that the maximum alignment between operands A and B is 8.
* `aiu_multistage_overlap_prologue`: Mainloop employs a persistent aiu multistage overlap_prologue mainloop and kernel schedule.
* `simt_vectorized`: Kernel epilogue employs simt based vectorization.


## Interpreting PPU Profiler Output

### Key Metrics

| Metric | Description | Optimal Value |
|--------|-------------|---------------|
| Runtime | Kernel execution time (ms) | Lower is better |
| Memory | Memory bandwidth (GiB/s) | Higher is better |
| Math | GFLOP/s | Higher is better |
| FLOPs/Byte | Computational intensity | Higher is better |

### Expected Performance Ranges

**PPU 1.0 FP16 GEMM:**
- 256x128x64 tile: 80-120 GFLOP/s

**PPU 1.5 FP16 GEMM:**
- 256x128x64 tile: 250-350 GFLOP/s

## PPU Profiling Troubleshooting

### Issue: Low Memory Bandwidth

**Solution:** Increase tile sizes to improve memory reuse:
```bash
--cta_m=256 --cta_n=256 --cta_k=64
```

### Issue: Low Occupancy

**Solution:** Reduce register usage by using fewer stages:
```bash
--stages=2
```

### Issue: No Kernels Found

**Solution:** Verify PPU architecture flag:
```bash
# For PPU 1.0
cmake .. -DCUTLASS_PPU_ARCHS=ppu0010

# For PPU 1.5
cmake .. -DCUTLASS_PPU_ARCHS=ppu0015
```

---

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

