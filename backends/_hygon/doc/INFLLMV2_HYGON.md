# InfLLM-V2 Hygon native baseline

This is a self-developed HIP/MMAC implementation for BW1000/gfx936, derived
from the frozen Native v3 baseline. It is not an official Hygon InfLLM-V2
release. All data-dependent stages use HIP; the operator imports no Triton,
FlagTree, TileLang or FlagAttention. PyTorch/HIP supplies tensors, streams and
autograd, and DTK hipcc compiles the native sources for wave64.

## Layout

```text
backends/_hygon/
├── benchmarks/benchmark_infllmv2_hygon.py
├── doc/INFLLMV2_HYGON.md
├── doc/THIRD_PARTY_NOTICES.md
├── ops/infllmv2/
│   ├── __init__.py
│   ├── operator.py
│   ├── forward.hip
│   ├── backward.hpp
│   ├── common.hpp
│   ├── generic.hip
│   └── third_party/hytlass/include/hytlass/  # three licensed headers
└── test/infllmv2/test_infllmv2_hip.py
```

`operator.py` contains configuration, plans, dispatch, components, autograd
and native loading. `forward.hip` contains selection/attention and C ABI
launches, including the kernels in `backward.hpp`; `common.hpp` contains
MMAC types/conversions. `generic.hip` is compiled separately with its original
flags to cover additional configurations. Both ctypes libraries remain
independent of DSL compilers. HYTLASS files and licenses are unmodified;
their provenance and the reference license appear in THIRD_PARTY_NOTICES.md.

Public APIs are NativeConfig, NativePlan, infllmv2_attention and
infllmv2_decode. Common BF16/FP16 shapes use MMAC; extended cases use HIP.
FP32 does not claim MMAC acceleration. Inputs use packed suffix-aligned
causal Q/K/V with Hq divisible by Hkv. First-order Q/K/V gradients are
supported; discrete selection is not differentiated. There is no paged KV
or in-place cache append. Sparse selection permits <=4096 candidate blocks
and power-of-two Top-K <=2048, subject to configuration, int32 and actual
device-resource limits. Cumulative lengths must cover the supplied packed
tensors. Untracked cu updates via .data/raw pointers are outside the cache
contract. A prepared plan excludes simultaneous metadata/config overrides.

## Environment and commands

Validated runtime: BW1000/gfx936, DTK26.04/DCC26.02, PyTorch2.9.0 with
HIP6.3.26045; Python3.10 and pytest. Install the Hygon framework separately.
The default compiler is `/opt/dtk/bin/hipcc`; override HIPCC if needed.
The caller selects the device. No physical card number or PCI address is
bound by the committed operator, test or benchmark.

From the repository root after selecting an available device:

```bash
source /opt/dtk/env.sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest -q backends/_hygon/test/infllmv2/test_infllmv2_hip.py
python -m backends._hygon.benchmarks.benchmark_infllmv2_hygon \
  --output results/infllmv2_hygon.json
```

The benchmark can also be launched as a script using its full relative path.
Its default native-only execution asserts that compiler packages were not
imported. Both native libraries are compiled on first use; compilation and
input/reference generation occur before measurement. Each timed call uses
the public API with warmed length/plan caches but fresh output/workspace
allocations; data-dependent selection/output values are not cached.

Defaults: BF16, Hq32/Hkv2, D64/D128, Top-K64, block64, dense threshold8192,
20ms warmup, >=100ms GPU event samples with >=5 calls per round, seven rounds.
Report median of round medians, all-sample mean, raw samples and p99 only
when there are >=100 samples. Backward cases time forward plus backward.
The 17 cases cover Dense1K, Full8K/32K/64K, Chunk128x16K/65K, backward128x8K,
and B1/B8 Decode. Use --case to select rows and --rounds/--warmup-ms/--rep-ms
to make explicitly labeled diagnostic runs.

Each case checks finite results/gradients and three same-index Stage2 query
rows per sequence against FP32 Torch outside timing. This sample does not
claim full-output or independent Stage1 correctness: the separate test file
checks compression, Stage1, pooling, exact Top-K, fixed-index attention and
gradients against Torch references. Tests also cover FP16/BF16/FP32,
ragged/noncontiguous inputs, GQA, nonpositive scale, decode/cu mutation and
dispatch/plan contracts. No reference kernel calls Triton.

With an installed FlagAttention Hygon/Triton runtime, optional peer timing is:

```bash
export FLAG_ATTN_BACKEND=hygon
python -m backends._hygon.benchmarks.benchmark_infllmv2_hygon \
  --compare-flagattention --output results/infllmv2_hygon_vs_flagattention.json
```

The same inputs, precision and event protocol are used, with alternating
implementation order each round. HIP/FlagAttention is a latency ratio, so
values >1 mean HIP is slower. Forward comparisons retain atol=rtol=0.04,
and gradient comparisons retain 0.03. A mismatch is saved with a nonzero
exit status; tolerances are not widened. The optional peer is the only mode
that imports FlagAttention and its compiler dependencies.

## Frozen reference and numerical boundary

The unique historical performance reference remains
`BW1000-InfLLMv2-NativeV3-BF16-17-20261028-r7`:

- Original source SHA256: c9b0ed0cb7407c7449e31b2fb91486c851f2d5a5a858988beef61e9013c2fa83.
- Frozen native binary SHA256: ff33b83bc3babfe44e41522b3d3c449e5e1ccac41eee774149e10e231f23731a.
- Semantic source: FlagAttention 54417c6e9e4f85cdca248981ed18dc5b9fa5eb5d.
- Parent Native v2 SHA256: ba059ab5c9153f59c27858017435de756a5ee0bee4aa4090d0dd73611b793de0.

The compact Python API/component definitions were audited against the frozen
source. The expanded MMAC translation unit has identical computation tokens,
the generic HIP source is identical after LF newline normalization, and
compile flags are unchanged. The three third-party headers are byte-identical.
File organization/cache source keys and source identities are distinct from
the frozen archive. New measurements are drift/reproduction results, never
replacements for the frozen source/binary/measurement contract.

Historical verification reported 19 official tests passed / 4 skipped and
17 official benchmarks passed. The directory reorganization additionally
passed 10 forward/gradient cases in a Triton-free environment. These counts
do not describe this repository's own pytest collection.

Known historical exception: 64K/D64 versus Triton had 1/134217728 outputs
outside atol=rtol=0.04 (max absolute difference 0.0409393), associated with
BF16 Stage1 rounding and Top-K sensitivity. Same-index Stage2 comparison
passed. The implementation/precision were not changed to erase this case;
full-pipeline equality on every input is not claimed. Generic HIP backward
uses FP32 atomic accumulation and may show reduction-order differences.

Generated JSON, raw measurements, build logs and libraries belong outside
the committed source. They are not part of the frozen reference definition.
