# BaselineBenchmark

BaselineBenchmark collects reference implementations and operator benchmarks across hardware backends. Each backend owns its implementations, benchmark entry points, and validation notes; shared conventions and utilities live at the repository level.

## Intended use

These implementations are **experimental baselines for research and comparison**. They are not automatically official vendor implementations, production-ready operators, or drop-in replacements. Some combine code from several sources or use internal APIs. Check each operator's README for provenance, supported inputs, software versions, correctness limits, and fallback behavior.

Interpret performance results together with their case definitions, device model, software environment, source revision, and measurement method. Identify fallback results separately. Performance measurement does not establish numerical correctness.

## Repository layout

- `backends/`: backend-specific operators, benchmarks, and tests.
- `common/`: shared benchmark and result utilities.
- `configs/`: reusable case definitions and settings.
- `docs/`: benchmark conventions and design notes.
- `scripts/`: repository-level automation.
- `results/`: generated measurements and associated metadata.

The repository contains implementations for multiple accelerator families. `_template` is a starting point for new backends, not a runnable implementation.

## Evaluating a baseline

1. Distinguish upstream code from local adaptations and document all sources.
2. State supported inputs, dependency versions, device model, and fallback paths.
3. Validate correctness independently of timed execution.
4. Record shapes, dtypes, layouts, warmup, repetitions, latency units, and source revisions.
5. Compare compatible measurements and retain failures and regressions.

See [benchmark conventions](docs/benchmark_conventions.md) for general guidance. Operator-specific instructions belong with the corresponding backend.

## Moore Threads MoBA example

The [MoBA baseline](backends/_mthreads/ops/moba/README.md) combines project-specific routing with a kernel generated from MATE's FMHA template. It is **not an official Moore Threads MoBA implementation** and is kept for reference and performance comparison. The pure Triton operator in FlagAttention is a separate implementation.
