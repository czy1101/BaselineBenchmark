# S60 NSA and MSA adaptation notes

## Scope

These notes describe the validated sparse-attention paths adapted for
Enflame S60/GCU300 and the corresponding Torch reference baselines.

The production kernels are maintained in FlagAttention. This repository
contains independent Torch references, correctness tests, and controlled
same-card benchmarks.

## NSA

The submitted NSA implementation supports the explicit block-index
forward path:

- fixed-length rank-4 Q/K/V tensors;
- grouped-query attention;
- BF16 and FP16 inputs;
- key and value dimensions up to 128;
- int32 GCU index arithmetic;
- Triton and the validated NSA TLE path.

The submitted interface intentionally does not claim support for:

- backward propagation;
- compression-driven block selection;
- sliding-window composition;
- implicit generation of block indices.

The Torch reference is the same implementation used during the S60
same-card evaluation. It does not import FlagAttention, FlagGems,
Triton, or TLE.

## MSA

The submitted MSA implementation supports:

- BF16 prefill;
- BF16 decode;
- fused prefill index-score and TopK selection;
- grouped-query sparse attention;
- flattened decode launch mapping for the GCU grid limits;
- int32 device indices.

The retained prefill optimizations include grouped-eight query
processing, exact selection reuse, compact union handling, direct
candidate extraction, and fused Score+TopK.

The retained decode implementation includes the GCU-compatible flattened
launch mapping and the validated direct Triton attention paths.

### Decode score matrix orientation

The largest D4 decode improvement came from changing the orientation of
the index-score matrix multiplication. Before this change, the D4 stage
breakdown was approximately:

| Stage | Latency | Share |
|---|---:|---:|
| Index Score | 4.12 ms | 94%--96% |
| Sparse Attention | 0.108 ms | 2.5% |
| Other overhead | 0.1 ms | Remaining share |

The bottleneck was therefore score generation rather than sparse
attention or page ordering. The original decode score calculation used:

```text
K[128, D] x Q[D, HQ] -> Score[128, HQ]
```

Decode has a small `HQ`, so this formulation gives the matrix
multiplication a very narrow output-N dimension. That shape has poor
utilization of the S60 matrix-compute units.

The optimized calculation reverses the multiplication orientation:

```text
Q[HQ, D] x K[D, 128] -> Score[HQ, 128]
```

It then performs the maximum reduction along the token dimension:

```text
block_score[q, block] = max_token(dot(q, k_token))
```

The new score matrix is the transpose of the original score matrix.
Consequently, each query-key dot product and the subsequent maximum over
the 128 tokens are unchanged; only the input and output layouts of the
matrix multiplication differ.

On D4, this change reduced the measured score-kernel latency from
4.118884 ms to 0.056207 ms, a 73.3x speedup. It was the single largest
contributor to the decode optimization. After the score kernel was
accelerated, Torch-GCU `topk`, multi-operator Identity page generation,
and merging intermediate attention results became the dominant fixed
overheads.

## GCU300 TLE limitation

The GCU300 compiler in the validated environment does not register or
lower the Triton TLE dialect used by the upstream MSA implementation.
In particular, upstream barrier-based TLE constructs and
`tle.gpu.alloc`/`tle.gpu.local_ptr` cannot be compiled on this target.

Consequently:

- MSA uses the validated direct Triton implementation;
- requesting MSA TLE explicitly raises a clear runtime error;
- no non-functional pseudo-TLE implementation is included.

This limitation is specific to MSA. The validated NSA TLE path remains
available.

## Torch baselines

The baseline modules are correctness references rather than fused vendor
kernels:

- NSA uses the previously validated chunked PyTorch implementation;
- MSA prefill uses vectorized torch_gcu/TopsAten operators;
- MSA decode is extracted from the previously validated official-stack
  same-card benchmark and exposed as a reusable function.

Each implementation has an independent small-shape correctness test.

## Benchmark protocol

Torch and adapted implementations are measured in separate processes.
The benchmark protocol uses:

- identical inputs and case definitions;
- explicit device synchronization;
- configurable warmup, repeat, and sustained inner-loop counts;
- latency reported per invocation;
- median, minimum, and maximum latency;
- optional machine-readable JSON output;
- correctness testing outside timed regions.

Generated logs and results are not committed.

## Final MSA same-card results

| Stage | Case | Shape | Sequence lengths | Torch (ms) | Adapted (ms) | Speedup |
|---|---|---|---|---:|---:|---:|
| Prefill | P1 | `(1, 512, 2, 16, 128)` | `(512,)` | 4.338029 | 0.592948 | 7.316x |
| Prefill | P2 | `(1, 1024, 2, 16, 128)` | `(1024,)` | 8.193309 | 1.690364 | 4.847x |
| Prefill | P3 | `(1, 4096, 2, 16, 128)` | `(4096,)` | 33.145353 | 11.213723 | 2.956x |
| Prefill | P4 | `(1, 1024, 4, 32, 128)` | `(1024,)` | 15.821532 | 3.214786 | 4.921x |
| Decode | D1 | `(1, 1, 2, 16, 128)` | `(512,)` | 3.667791 | 1.078274 | 3.402x |
| Decode | D2 | `(1, 1, 2, 16, 128)` | `(8192,)` | 3.680157 | 2.133624 | 1.725x |
| Decode | D3 | `(8, 1, 2, 16, 128)` | `(2048,) * 8` | 3.886726 | 4.074408 | 0.954x |
| Decode | D4 | `(3, 4, 2, 16, 128)` | `(2048, 1024, 513)` | 3.897820 | 4.963325 | 0.785x |

Geometric-mean speedups:

- Prefill: 4.766x
- Decode: 1.448x
- Overall: 2.627x

## Known performance limitations

Prefill remains below the original 10x geometric-mean target, with the
longest-sequence case showing the largest remaining gap.

Decode D3 and D4 are slower than the Torch reference. These cases should
not be presented as performance wins; they are retained to make the
current limitation explicit and reproducible.
