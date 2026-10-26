"""Shared pytest command-line options for Enflame benchmarks."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


# Prefer the sibling FlagAttention checkout over an older installed package.
_FLAG_ATTENTION_SRC = Path(__file__).resolve().parents[4] / "FlagAttention" / "src"
if _FLAG_ATTENTION_SRC.is_dir() and str(_FLAG_ATTENTION_SRC) not in sys.path:
    sys.path.insert(0, str(_FLAG_ATTENTION_SRC))


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("enflame-benchmark")

    # Options shared by the MSA, NSA, and Chunk GLA benchmarks.
    group.addoption(
        "--implementation",
        choices=("torch", "topscc", "adapted", "compare", "topscc-compare"),
    )
    group.addoption("--case")
    group.addoption("--all", action="store_true", default=None)
    group.addoption("--device", type=int)
    group.addoption("--warmup", type=int)
    group.addoption("--repeat", type=int)
    group.addoption("--inner", type=int)
    group.addoption("--topk", type=int)
    group.addoption("--json-output")
    group.addoption("--quant-type")
    group.addoption("--layout", choices=("NHD", "HND"))
    group.addoption("--masked", action="store_true", default=None)
    group.addoption("--mtp", type=int)
    group.addoption("--num-query-heads", type=int)
    group.addoption("--dtype", choices=("bfloat16", "float16"))
    group.addoption("--max-batch-query", type=int)
    group.addoption("--iter", type=int, dest="iterations")
    group.addoption("--chunk-size", type=int)
    group.addoption("--quick", action="store_true", default=None)
    group.addoption("--skip-current", action="store_true", default=None)
    group.addoption("--shape-index", type=int)

    # Parallax baseline/Triton comparisons.
    group.addoption("--mode", choices=("forward", "fwdbwd", "both"))
    group.addoption("--timing", choices=("eager", "graph"))
    group.addoption("--calls-per-graph", type=int)
    group.addoption("--window-size", type=int)
    group.addoption("--max-score-elements", type=int)

    # SageAttention-specific options.
    group.addoption(
        "--backend",
        choices=("auto", "torch-gcu", "vllm-gcu", "flash-attn", "triton"),
    )
    group.addoption("--allow-triton-fallback", action="store_true", default=None)
    group.addoption("--compare-triton", action="store_true", default=None)
    group.addoption("--custom-shapes", action="store_true", default=None)
    group.addoption("--isolate-shapes", action="store_true", default=None)
    group.addoption("--batch-size", type=int)
    group.addoption("--num-heads", type=int)
    group.addoption("--num-kv-heads", type=int)
    group.addoption("--head-dim", type=int, choices=(64, 128))
    group.addoption("--seq-lens", type=int, nargs="+")
    group.addoption("--rep", type=int)
    group.addoption(
        "--check-reference",
        action="store_true",
        dest="check_reference",
        default=None,
    )
    group.addoption(
        "--no-check-reference",
        action="store_false",
        dest="check_reference",
    )
    group.addoption("--atol", type=float)
    group.addoption("--rtol", type=float)
    group.addoption("--seed", type=int)


class BenchmarkOptions:
    """Read a pytest option while retaining benchmark-specific defaults."""

    def __init__(self, config: pytest.Config) -> None:
        self._config = config

    def get(self, name: str, default=None, *, required: bool = False):
        value = self._config.getoption(name)
        if value is None:
            if required:
                flag = name.replace("_", "-")
                raise pytest.UsageError(f"--{flag} is required for this benchmark")
            return default
        return value

    def add_parallax_result(self, record) -> None:
        records = getattr(self._config, "_parallax_benchmark_results", None)
        if records is None:
            records = []
            self._config._parallax_benchmark_results = records
        records.append(record)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    records = getattr(config, "_parallax_benchmark_results", ())
    if not records:
        return
    from backends._enflame.benchmarks._parallax_table import render_tables

    if exitstatus != 0:
        terminalreporter.write_line("Parallax: partial results from completed, validated measurements.")
    terminalreporter.write("\n" + render_tables(records) + "\n")


@pytest.fixture
def benchmark_options(pytestconfig: pytest.Config) -> BenchmarkOptions:
    return BenchmarkOptions(pytestconfig)
