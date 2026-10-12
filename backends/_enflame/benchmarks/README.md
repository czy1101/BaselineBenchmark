# Enflame S60 sparse-attention benchmarks

This directory contains controlled same-card benchmarks for:

- NSA explicit block-index forward;
- MSA BF16 prefill;
- MSA BF16 decode.
- HPC paged-attention BF16/FP8 dynamic decode.
- HPC block-sparse-attention FP8 dynamic prefill.

The isolated benchmark modes measure exactly one implementation per process. Run the Torch
reference and adapted implementation in separate processes with the
same arguments.

Required environment:

```bash
export PYTHONPATH=/path/to/BaselineBenchmark:/path/to/FlagAttention/src
export ENFLAME_PT_OP_DEBUG_CONFIG='op_sync_mode=false op_statistics=false'
```

Examples:

```bash
python backends/_enflame/benchmarks/bench_nsa_forward.py           --implementation torch --case CALIB_H16_S2K --device 1

python backends/_enflame/benchmarks/bench_nsa_forward.py           --implementation adapted --case CALIB_H16_S2K --device 1

python backends/_enflame/benchmarks/bench_msa_prefill.py           --implementation torch --case P2 --device 1

python backends/_enflame/benchmarks/bench_msa_prefill.py           --implementation adapted --case P2 --device 1

python backends/_enflame/benchmarks/bench_msa_decode.py           --implementation torch --case D4 --device 1

python backends/_enflame/benchmarks/bench_msa_decode.py           --implementation adapted --case D4 --device 1

python backends/_enflame/benchmarks/bench_hpc_ops_decode.py \
  --implementation torch --case uniform_512 --quant-type bf16 \
  --layout NHD --mtp 1 --device 0

python -m pytest -q -s \
  backends/_enflame/benchmarks/bench_hpc_ops_decode.py \
  --implementation topscc-compare --case uniform_512 \
  --quant-type bf16 --layout NHD --mtp 1 --device 0

python -m pytest -q -s \
  backends/_enflame/benchmarks/bench_hpc_ops_decode.py \
  --implementation compare --case uniform_512 \
  --quant-type qpertoken_perhead_kvpertensor \
  --layout NHD --mtp 1 --device 0

python backends/_enflame/benchmarks/bench_hpc_ops_decode.py \
  --implementation adapted --case uniform_512 \
  --quant-type qpertoken_perhead_kvpertensor \
  --layout NHD --mtp 1 --device 0

python backends/_enflame/benchmarks/bench_hpc_ops_prefill.py \
  --implementation torch --case q512_kv4096 \
  --quant-type qpertoken_perhead_kvpertensor \
  --layout NHD --masked --device 0

python backends/_enflame/benchmarks/bench_hpc_ops_prefill.py \
  --implementation adapted --case q512_kv4096 \
  --quant-type qpertoken_perhead_kvpertensor \
  --layout NHD --masked --device 0

python backends/_enflame/benchmarks/bench_hpc_ops_prefill.py \
  --implementation topscc-compare --case q512_kv4096 \
  --quant-type qpertoken_perhead_kvpertensor \
  --layout NHD --masked --device 0

```

Timing protocol:

- explicit warmup;
- sustained inner-loop execution;
- synchronization outside each measured group;
- median/min/max latency in milliseconds per invocation;
- optional JSON result output;
- correctness checks are kept outside timed benchmarks and live under
  `backends/_enflame/test/`.

## Parallax baseline versus FlagAttention Triton

The two Parallax benchmarks are pytest comparison tests. Parallel and decode
both use the Torch-GCU baseline. The Parallax TopsCC implementations and their
benchmarks/tests have been removed.
Both providers receive the same inputs, dtype, scale,
window, device, warmup and sampling budget. Output columns include
baseline latency, FlagAttention Triton latency and `speedup` (no MAD columns).
At the end of the pytest run, completed results are collected into fixed-width
`[Forward]` and `[Forward + Backward]` tables, each with one header for all
selected shapes. Provider provenance, timing settings and validation error
are retained below the tables. Failed rows are not added; interrupted/failed
runs can show only the completed validated measurements.

`speedup = baseline_ms / triton_ms`: greater than 1 means Triton is faster;
less than 1 means baseline is faster. This is same-card operator latency,
not a hardware peak-throughput measurement.

Keep `BaselineBenchmark` and the updated `FlagAttention` checkouts side by side,
or expose the intended FlagAttention implementation through `PYTHONPATH`.
The benchmark prints both implementation source paths. Missing dependencies,
GCU devices or decode TLE are skipped; compilation and numerical
failures are not silently treated as unavailable implementations.

```bash
cd /workspace/BaselineBenchmark

# Parallel forward and forward+backward, Torch-GCU baseline, first shape.
python -m pytest -q -s -m parallax \
  backends/_enflame/benchmarks/bench_parallax_torch.py \
  --shape-index 0 --mode both --warmup 10 --iter 12

# Decode, Torch-GCU baseline, matched GCU graph/event timing on both providers.
python -m pytest -q -s -m parallax \
  backends/_enflame/benchmarks/bench_parallax_decode_torch.py \
  --shape-index 0 --timing graph --calls-per-graph 100 --warmup 10 --iter 12

# Host-only comparison/timing checks (no GCU required).
python -m pytest -q \
  backends/_enflame/test/test_parallax_benchmark_host.py
```

Use `--dtype float16` for FP16; the default is BF16.
`--window-size` supplies the same window to both providers.

Without `--all`, only shape 0 is selected unless `--shape-index` specifies
another case. `--all` covers the existing 14 parallel or 10 decode shapes.
Torch-GCU backward on long parallel shapes retains quadratic attention state
and can require substantial memory; first validate a single case or use
`--mode forward` before a full sweep.

Parallel uses matched synchronized eager wall-clock timing including operator
host overhead; it does not support graph-mode autograd timing. Decode defaults
to graph replay timed by GCU events, with caller-owned output buffers for both
providers; `--timing eager` instead includes host overhead on both sides.
These timing modes must not be mixed in one speedup ratio.

Sampling alternates mirrored ABBA/BAAB provider order: `--iter N` yields `2*N`
samples per provider. Each graph sample contains `--calls-per-graph` calls;
eager samples contain `--inner` calls (default 1). Latency is divided by the
same call count for both providers. Compilation is outside timing.

Parallel validates forward outputs, and all four gradients when requested,
against the selected baseline before timing. Decode validates both outputs
against the Torch-GCU reference before capture and after graph replay/timing.
Incorrect or nonfinite results never produce a speedup row. Unlike the
isolated modes above, these comparison tests measure both providers in the
same process.
