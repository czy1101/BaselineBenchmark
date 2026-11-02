# T-Head Wall Attention benchmark

The reference implementation is the t-head/flash-linear-attention-for-sail
Git submodule at ops/FLA/wall_attn/flash-linear-attention-for-sail, pinned to
ea54b7d30c9015bec8c0837f4b5c3d627d1f745f. The benchmark imports it
directly as the FLA baseline; it is not the candidate implementation.

Clone both repositories:

```bash
git clone --branch thead-wallattention --recurse-submodules \
  https://github.com/Atopos0313/BaselineBenchmark.git
git clone --branch thead-wall \
  https://github.com/Atopos0313/FlagAttention.git
```

For an existing BaselineBenchmark checkout, run
`git submodule update --init --recursive` before running the benchmark.
The FLA for SAIL revision is fixed by the BaselineBenchmark gitlink, so there
is no need to select an FLA branch manually.

Run the benchmark from the BaselineBenchmark root. Set
`CUDA_VISIBLE_DEVICES` to an idle T-Head PPU index:

```bash
cd BaselineBenchmark
FLAGATTN_ROOT="$(realpath ../FlagAttention)"
CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$FLAGATTN_ROOT/src:$FLAGATTN_ROOT" \
  python -m pytest -q -s backends/_thead/benchmarks/test_parallel_wall_attn_benchmark.py
```

The candidate `flag_attn` package and its `benchmark.recording` helper come
from the separately cloned FlagAttention repository. The official FLA provider
comes from this repository's pinned submodule.

The script verifies the imported FLA source path and compares all measured
outputs with the candidate before reporting per-shape latency.
