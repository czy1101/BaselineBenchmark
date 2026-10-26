"""Compare optimized Torch-GCU and Enflame Triton AttnRes.

Run from the BaselineBenchmark repository root:
    pytest backends/_enflame/benchmarks/bench_attnres_torch.py -vs -ra

Both providers allocate their own outputs and run in inference mode. Timing
matches FlagAttention's test_fused_attnres_benchmark.py through its do_bench:
seven rounds, each with a 1000 ms warmup and 100 ms sampling budget, using
the median of round medians. GCU events are used when available; otherwise
synchronized wall time is used. This does not time a preallocated kernel.
Speedup is Torch latency / Triton latency; values above one favor Triton.
Both outputs are checked against FlagAttention's Torch reference before timing.

Optional overrides: --warmup is milliseconds, --rep is sampling milliseconds,
and --iter (or --repeat) is rounds. --inner must remain 1.
"""

from __future__ import annotations

import importlib
import os
import statistics
from pathlib import Path

import pytest
import torch

from backends._enflame.benchmarks._common import write_json
from backends._enflame.ops.attnres import (
    attnres_torch_opt_level,
    attnres_torch_score_mode,
    fused_attnres_torch_baseline,
    torch_gcu_attnres_available,
)


pytestmark = pytest.mark.fused_attnres

# Match FlagAttention's AttnRes benchmark, including both output RMSNorm modes.
BENCHMARK_SHAPES = ((2, 1), (5, 128), (5, 1024), (9, 128), (9, 8192))
HIDDEN_SIZE = 7168
BENCHMARK_ROUNDS = 7
BENCHMARK_WARMUP_MS = 1000
BENCHMARK_REP_MS = 100
CASES = tuple(
    (f"L{sources}_N{rows}_D{HIDDEN_SIZE}_ON{int(output_norm)}", sources, rows, output_norm)
    for sources, rows in BENCHMARK_SHAPES
    for output_norm in (False, True)
)


def pytest_generate_tests(metafunc):
    if "case" not in metafunc.fixturenames:
        return
    selected = metafunc.config.getoption("case")
    cases = [case for case in CASES if selected is None or case[0] == selected]
    if not cases:
        raise pytest.UsageError(f"Unknown --case {selected!r}; choose from {', '.join(case[0] for case in CASES)}")
    metafunc.parametrize("case", cases, ids=[case[0] for case in cases])


@pytest.fixture(scope="module")
def comparison_records(pytestconfig):
    records = []
    yield records
    if not records:
        return
    print(f"\nAttnRes: public API forward, {records[0]['timing']} (ms)")
    baseline_name = records[0]["baseline_name"]
    print(f"Speedup = {baseline_name} / Triton")
    print(f"{'Case':<29} {baseline_name:>12} {'Triton':>12} {'Speedup':>12}")
    for record in records:
        print(
            f"{record['case']:<29} {record['baseline_ms']:>12.6f} {record['triton_ms']:>12.6f} "
            f"{record['speedup']:>11.3f}x"
        )
    write_json(
        pytestconfig.getoption("json_output") or f"results/attnres_{baseline_name.lower()}_comparison.json",
        {
            "operator": "fused_attnres",
            "baseline": "Optimized Torch-GCU",
            "candidate": "Enflame Triton",
            "scope": "public_api_forward",
            "timing": records[0]["timing"],
            "timing_protocol": "flag_attn.testing.backend.do_bench",
            "aggregation": "median_of_round_medians",
            "speedup_definition": "baseline_ms / triton_ms",
            "results": records,
        },
    )


@torch.inference_mode()
def test_attnres_torch_vs_triton(case, benchmark_options, comparison_records):
    _run_attnres_comparison(
        case, benchmark_options, comparison_records,
        baseline_name="Torch", baseline_fn=fused_attnres_torch_baseline,
        baseline_available=torch_gcu_attnres_available,
    )


def _measure_comparison(backend, run_baseline, run_triton, *, warmup_ms, rep_ms, rounds):
    def bench(fn):
        return backend.do_bench(fn, warmup=warmup_ms, rep=rep_ms, return_mode="median")

    backend.device_fn.synchronize()
    baseline_rounds = []
    triton_rounds = []
    # Match FlagAttention's alternating provider order and round aggregation.
    for round_index in range(rounds):
        if round_index % 2 == 0:
            baseline_rounds.append(bench(run_baseline))
        triton_rounds.append(bench(run_triton))
        if round_index % 2 == 1:
            baseline_rounds.append(bench(run_baseline))

    baseline_ms = statistics.median(baseline_rounds)
    triton_ms = statistics.median(triton_rounds)
    return {
        "baseline_ms": baseline_ms,
        "triton_ms": triton_ms,
        "speedup": baseline_ms / triton_ms,
        "baseline_rounds_ms": baseline_rounds,
        "triton_rounds_ms": triton_rounds,
    }


@torch.inference_mode()
def _run_attnres_comparison(
    case, benchmark_options, comparison_records, *, baseline_name, baseline_fn, baseline_available,
):
    if not baseline_available():
        pytest.skip(f"{baseline_name} AttnRes environment unavailable")

    warmup_ms = benchmark_options.get("warmup", BENCHMARK_WARMUP_MS)
    rep_ms = benchmark_options.get("rep", BENCHMARK_REP_MS)
    rounds = benchmark_options.get("iterations", benchmark_options.get("repeat", BENCHMARK_ROUNDS))
    inner = benchmark_options.get("inner", 1)
    if warmup_ms < 0 or rep_ms < 1 or rounds < 1:
        raise pytest.UsageError("--warmup (ms) must be non-negative; --rep (ms) and --iter/--repeat (rounds) must be positive")
    if inner != 1:
        raise pytest.UsageError("AttnRes uses one API call per timing sample; --inner must be 1")
    if benchmark_options.get("timing", "eager") != "eager":
        raise pytest.UsageError("AttnRes comparison uses --timing eager (FlagAttention do_bench)")

    device_index = benchmark_options.get("device", 0)
    torch.gcu.set_device(device_index)
    device = torch.device(f"gcu:{device_index}")
    dtype_name = benchmark_options.get("dtype", "bfloat16")
    dtype = getattr(torch, dtype_name)
    seed = benchmark_options.get("seed", 0)
    torch.manual_seed(seed)

    # Load the concrete Enflame module, so a generic fallback is never timed.
    candidate_module = importlib.import_module("flag_attn.runtime.backend._enflame.FLA.attnres.fused")
    if not candidate_module.HAS_TLE_ATTNRES:
        pytest.fail("Enflame AttnRes requires Triton TLE; the candidate is unavailable")
    triton_attnres = candidate_module.fused_attnres
    import flag_attn
    from flag_attn.testing import backend as test_backend

    if test_backend.device != "gcu" or not test_backend.is_available():
        pytest.fail("FlagAttention's benchmark backend must be an available GCU")

    case_name, num_sources, num_rows, output_norm = case
    print(
        f"\n{case_name} device={device} dtype={dtype_name} baseline={baseline_name}: "
        "warming up and measuring performance", flush=True,
    )
    residuals = [
        torch.randn(num_rows, HIDDEN_SIZE, device=device, dtype=dtype)
        for _ in range(num_sources)
    ]
    query = torch.randn(HIDDEN_SIZE, device=device, dtype=dtype)
    rms_weight = torch.randn(HIDDEN_SIZE, device=device, dtype=dtype)
    output_rms_weight = torch.randn(HIDDEN_SIZE, device=device, dtype=dtype) if output_norm else None
    args = (query, residuals, rms_weight, output_rms_weight)
    kwargs = {"rms_eps": 1e-6, "scale": HIDDEN_SIZE**-0.5, "return_weights": False}

    def run_baseline():
        return baseline_fn(*args, **kwargs)

    def run_triton():
        return triton_attnres(*args, **kwargs)

    reference = flag_attn.testing.fused_attnres(*args, **kwargs)
    for provider in (run_baseline, run_triton):
        actual = provider()
        torch.testing.assert_close(actual.float(), reference.float(), atol=5e-3, rtol=1e-2)

    # Reuse exactly the same backend timer as the Triton benchmark.
    timings = _measure_comparison(
        test_backend, run_baseline, run_triton,
        warmup_ms=warmup_ms, rep_ms=rep_ms, rounds=rounds,
    )
    import triton

    record = {
        "case": case_name,
        "baseline_name": baseline_name,
        "num_sources": num_sources,
        "num_rows": num_rows,
        "hidden_size": HIDDEN_SIZE,
        "output_norm": output_norm,
        "return_weights": False,
        "dtype": dtype_name,
        "device": str(device),
        "device_name": torch.gcu.get_device_name(device_index),
        "torch_version": str(torch.__version__),
        "triton_version": str(triton.__version__),
        "candidate_file": str(Path(candidate_module.__file__).resolve()),
        "candidate_autotune_env": os.environ.get("FLAG_ATTN_ATTNRES_AUTOTUNE"),
        "seed": seed,
        "warmup_ms": warmup_ms,
        "rep_ms": rep_ms,
        "rounds": rounds,
        "timing": "backend_events" if getattr(test_backend.device_fn, "Event", None) is not None else "synchronized_wall_time",
        "timing_protocol": "flag_attn.testing.backend.do_bench",
        "aggregation": "median_of_round_medians",
        **timings,
    }
    if baseline_name == "Torch":
        record.update(torch_score_mode=attnres_torch_score_mode(num_rows), torch_opt_level=attnres_torch_opt_level())
    comparison_records.append(record)
    print(
        f"RESULT {case_name} {baseline_name}={timings['baseline_ms']:.6f} ms Triton={timings['triton_ms']:.6f} ms "
        f"speedup={timings['speedup']:.3f}x",
        flush=True,
    )
