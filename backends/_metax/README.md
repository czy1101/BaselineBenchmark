# MetaX C550 operators and benchmarks

This backend also contains independent comparison baselines for GLA, NSA, KDA, and SageAttention under per-operator ops/, tests/, and benchmarks/ directories.
This backend contains the five C550 operator lines adapted in FlagAttention:

- `flash_mla`
- `flash_mla_with_kvcache`
- `flash_mla_sparse_fwd`
- `chunk_gdn2`
- MiniMax M3 sparse attention (MSA)

The adapted implementations are imported from `czy1101/FlagAttention:cyc`.
Performance baselines live under `ops/`; correctness tests live under `test/`;
executable performance programs live under `benchmarks/`. Generated
logs and result files must remain
untracked and should be written under the repository-level `results/`
directory.

## Environment

- MetaX C550
- PyTorch with MetaX support
- FlagTree 3.6+ with TLE for the optimized TLE paths
- `czy1101/FlagAttention:cyc` on `PYTHONPATH` for the shared GDN2 primitives
- vLLM is optional and is used only as the MSA comparison baseline

Set the repository root and FlagAttention source on `PYTHONPATH` before
running the programs:

```bash
export PYTHONPATH=/path/to/BaselineBenchmark:/path/to/FlagAttention/src
```

Run the correctness suite before collecting performance data:

```bash
pytest -q backends/_metax/test
```

Individual operator suites can also be selected directly:

```bash
pytest -q backends/_metax/test/test_flash_mla.py
pytest -q backends/_metax/test/test_flash_mla_with_kvcache.py
pytest -q backends/_metax/test/test_flash_mla_sparse.py
pytest -q backends/_metax/test/test_chunk_gdn2.py
pytest -q backends/_metax/test/test_minimax_sparse_attention.py
```

After correctness passes, launch the benchmark programs from the repository
root:

```bash
python backends/_metax/benchmarks/flash_mla_benchmark.py
python backends/_metax/benchmarks/flash_mla_with_kvcache_benchmark.py
python backends/_metax/benchmarks/flash_mla_sparse_benchmark.py
python backends/_metax/benchmarks/chunk_gdn2_benchmark.py
python backends/_metax/benchmarks/minimax_sparse_attention_benchmark.py
```

The three FlashMLA programs report the adapted MetaX latency. The GDN2
program compares the TLE path with its native Triton fallback. The MSA
program compares the MetaX implementation with vLLM when vLLM is installed.
