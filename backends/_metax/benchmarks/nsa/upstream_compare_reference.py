"""Multi-shape C550 NSA E2E comparison."""

from __future__ import annotations

import gc
import importlib
import json
import math
import os
import statistics
import time
from collections.abc import Callable

import torch

from tileops.ops import NSAForwardVarlenOp


SHAPES = tuple(
    int(value)
    for value in os.environ.get(
        "NSA_SHAPES", "1024,2048,4096,8192,16384"
    ).split(",")
)
HQ, HKV, DIM = 32, 2, 128
BLOCK_SIZE, SELECTED_BLOCKS, WINDOW_SIZE = 32, 16, 128
SCALE = DIM**-0.5
DTYPE, ACCUM_DTYPE = torch.float16, torch.float32

WARMUP = int(os.environ.get("NSA_WARMUP", "10"))
CALLS_PER_SAMPLE = int(os.environ.get("NSA_CALLS_PER_SAMPLE", "5"))
SAMPLES = int(os.environ.get("NSA_SAMPLES", "30"))
ROUNDS = int(os.environ.get("NSA_ROUNDS", "5"))
ATOL = RTOL = 1e-2
RELATIVE_L2_LIMIT = 1e-2
PERFORMANCE_THRESHOLD = 0.90

os.environ.setdefault("FLA_NSA_TLE", "1")
_fg_module = importlib.import_module(
    "flaggems_vllm.runtime.backend._metax.ops.parallel_nsa"
)
parallel_nsa = _fg_module.parallel_nsa


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def describe(values: list[float]) -> dict[str, float]:
    mean = statistics.fmean(values)
    std = statistics.pstdev(values)
    return {
        "min_ms": min(values),
        "p20_ms": percentile(values, 0.20),
        "p50_ms": percentile(values, 0.50),
        "p80_ms": percentile(values, 0.80),
        "p95_ms": percentile(values, 0.95),
        "mean_ms": mean,
        "max_ms": max(values),
        "cv_percent": 100.0 * std / mean,
    }


def ci95(values: list[float]) -> tuple[float, float]:
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, mean
    half_width = 2.776 * statistics.stdev(values) / math.sqrt(len(values))
    return mean - half_width, mean + half_width


def make_inputs(total_tokens: int) -> tuple[torch.Tensor, ...]:
    seed = 20260826 + total_tokens
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    q = (
        torch.randn(total_tokens, HQ, DIM, device="cuda", dtype=DTYPE) * 0.1
    ).contiguous()
    k = (
        torch.randn(total_tokens, HKV, DIM, device="cuda", dtype=DTYPE) * 0.1
    ).contiguous()
    v = (
        torch.randn(total_tokens, HKV, DIM, device="cuda", dtype=DTYPE) * 0.1
    ).contiguous()
    gates = torch.softmax(
        torch.randn(
            total_tokens, HQ, 3, device="cuda", dtype=torch.float32
        ),
        dim=-1,
    ).to(DTYPE)
    g_cmp = gates[..., 0].contiguous()
    g_slc = gates[..., 1].contiguous()
    g_swa = gates[..., 2].contiguous()
    offsets = torch.tensor(
        [0, total_tokens], dtype=torch.int32, device="cuda"
    )
    return q, k, v, g_cmp, g_slc, g_swa, offsets


def make_op(total_tokens: int) -> NSAForwardVarlenOp:
    return NSAForwardVarlenOp(
        seq_num=1,
        c_seq_len=total_tokens,
        max_seqlen=total_tokens,
        heads=HQ,
        heads_kv=HKV,
        dim=DIM,
        chunk_num=total_tokens // BLOCK_SIZE,
        block_size=BLOCK_SIZE,
        selected_blocks=SELECTED_BLOCKS,
        window_size=WINDOW_SIZE,
        scale=SCALE,
        accum_dtype=ACCUM_DTYPE,
        tune=False,
    )


def make_runners(
    total_tokens: int,
    op: NSAForwardVarlenOp,
    inputs: tuple[torch.Tensor, ...],
) -> tuple[Callable[[], torch.Tensor], Callable[[], torch.Tensor]]:
    q, k, v, g_cmp, g_slc, g_swa, offsets = inputs

    @torch.no_grad()
    def tileops_run() -> torch.Tensor:
        return op(q, k, v, g_cmp, g_slc, g_swa, offsets)

    @torch.no_grad()
    def flaggems_run() -> torch.Tensor:
        core = parallel_nsa(
            q=q.unsqueeze(0),
            k=k.unsqueeze(0),
            v=v.unsqueeze(0),
            g_cmp=g_cmp.unsqueeze(0),
            g_slc=g_slc.unsqueeze(0),
            g_swa=None,
            block_indices=None,
            block_counts=SELECTED_BLOCKS,
            block_size=BLOCK_SIZE,
            window_size=0,
            scale=SCALE,
            cu_seqlens=offsets,
        ).squeeze(0)
        sliding = op._sliding_op(
            q, k, v, offsets, offsets, total_tokens
        )
        return torch.addcmul(core, sliding, g_swa.unsqueeze(-1))

    return tileops_run, flaggems_run


def correctness(
    total_tokens: int,
    tileops_run: Callable[[], torch.Tensor],
    flaggems_run: Callable[[], torch.Tensor],
    inputs: tuple[torch.Tensor, ...],
) -> dict[str, object]:
    snapshots = tuple(tensor.clone() for tensor in inputs)

    torch.cuda.synchronize()
    start = time.perf_counter()
    tileops_output = tileops_run()
    torch.cuda.synchronize()
    tileops_first_s = time.perf_counter() - start

    start = time.perf_counter()
    flaggems_output = flaggems_run()
    torch.cuda.synchronize()
    flaggems_first_s = time.perf_counter() - start

    assert tileops_output.shape == (total_tokens, HQ, DIM)
    assert flaggems_output.shape == tileops_output.shape
    assert tileops_output.dtype == flaggems_output.dtype == DTYPE
    assert bool(torch.isfinite(tileops_output).all())
    assert bool(torch.isfinite(flaggems_output).all())

    difference = (
        tileops_output.float() - flaggems_output.float()
    ).abs()
    max_abs = difference.max().item()
    relative_l2 = (
        difference.norm()
        / flaggems_output.float().norm().clamp_min(1e-12)
    ).item()

    torch.testing.assert_close(
        tileops_output,
        flaggems_output,
        atol=ATOL,
        rtol=RTOL,
    )
    assert relative_l2 <= RELATIVE_L2_LIMIT

    unchanged = all(
        torch.equal(current, saved)
        for current, saved in zip(inputs, snapshots, strict=True)
    )
    assert unchanged

    result = {
        "max_abs_err": max_abs,
        "relative_l2_err": relative_l2,
        "inputs_unchanged": unchanged,
        "tileops_first_call_seconds": tileops_first_s,
        "flaggems_first_call_seconds": flaggems_first_s,
        "tileops_checksum": tileops_output.float().sum().item(),
        "flaggems_checksum": flaggems_output.float().sum().item(),
    }
    del snapshots, difference, tileops_output, flaggems_output
    return result


def warmup(*functions: Callable[[], torch.Tensor]) -> None:
    for function in functions:
        output = None
        for _ in range(WARMUP):
            output = function()
        torch.cuda.synchronize()
        del output


def measure_sample(function: Callable[[], torch.Tensor]) -> float:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = None
    for _ in range(CALLS_PER_SAMPLE):
        output = function()
    end.record()
    end.synchronize()
    latency = start.elapsed_time(end) / CALLS_PER_SAMPLE
    del output
    return float(latency)


def benchmark(
    tileops_run: Callable[[], torch.Tensor],
    flaggems_run: Callable[[], torch.Tensor],
) -> dict[str, object]:
    tileops_rounds: list[list[float]] = []
    flaggems_rounds: list[list[float]] = []

    for round_id in range(ROUNDS):
        tile_values: list[float] = []
        flag_values: list[float] = []

        for sample_id in range(SAMPLES):
            if (round_id + sample_id) % 2 == 0:
                order = (
                    (tileops_run, tile_values),
                    (flaggems_run, flag_values),
                )
            else:
                order = (
                    (flaggems_run, flag_values),
                    (tileops_run, tile_values),
                )

            for function, destination in order:
                destination.append(measure_sample(function))

        tileops_rounds.append(tile_values)
        flaggems_rounds.append(flag_values)
        print(
            f"round={round_id + 1} "
            f"tileops_p50_ms={percentile(tile_values, 0.5):.6f} "
            f"flaggems_p50_ms={percentile(flag_values, 0.5):.6f}"
        )

    tileops_values = [
        value for values in tileops_rounds for value in values
    ]
    flaggems_values = [
        value for values in flaggems_rounds for value in values
    ]
    tileops_stats = describe(tileops_values)
    flaggems_stats = describe(flaggems_values)
    round_ratios = [
        percentile(tile, 0.5) / percentile(flag, 0.5)
        for tile, flag in zip(
            tileops_rounds, flaggems_rounds, strict=True
        )
    ]
    ratio = tileops_stats["p50_ms"] / flaggems_stats["p50_ms"]
    lower, upper = ci95(round_ratios)

    return {
        "tileops": tileops_stats,
        "flaggems": flaggems_stats,
        "speedup": ratio,
        "performance_percent": ratio * 100.0,
        "latency_reduction_percent": (
            1.0
            - flaggems_stats["p50_ms"] / tileops_stats["p50_ms"]
        ) * 100.0,
        "round_ratios": round_ratios,
        "round_ratio_mean": statistics.fmean(round_ratios),
        "round_ratio_95ci": [lower, upper],
        "pass_90_percent": (
            ratio >= PERFORMANCE_THRESHOLD
            and lower >= PERFORMANCE_THRESHOLD
        ),
    }


def memory_probe(function: Callable[[], torch.Tensor]) -> float:
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    output = function()
    torch.cuda.synchronize()
    incremental_mib = (
        torch.cuda.max_memory_allocated() - baseline
    ) / (1024**2)
    del output
    return incremental_mib


def run_shape(total_tokens: int) -> dict[str, object]:
    print(f"\n{'=' * 24} T={total_tokens} {'=' * 24}")
    inputs = make_inputs(total_tokens)
    op = make_op(total_tokens)
    tileops_run, flaggems_run = make_runners(
        total_tokens, op, inputs
    )

    check = correctness(
        total_tokens, tileops_run, flaggems_run, inputs
    )
    print("correctness:", json.dumps(check, sort_keys=True))

    tileops_memory = memory_probe(tileops_run)
    flaggems_memory = memory_probe(flaggems_run)
    warmup(tileops_run, flaggems_run)
    performance = benchmark(tileops_run, flaggems_run)

    result = {
        "T": total_tokens,
        "correctness": check,
        "tileops_incremental_peak_allocated_mib": tileops_memory,
        "flaggems_incremental_peak_allocated_mib": flaggems_memory,
        "performance": performance,
    }
    print("result_json:", json.dumps(result, sort_keys=True))

    del tileops_run, flaggems_run, op, inputs
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return result


def main() -> None:
    assert torch.cuda.is_available()
    assert "MetaX C550" in torch.cuda.get_device_name(0)
    assert "+metax" in torch.__version__
    assert _fg_module.HAS_TLE
    assert os.environ["FLA_NSA_TLE"] == "1"

    print("device:", torch.cuda.get_device_name(0))
    print("torch:", torch.__version__)
    print("FlagGems_TLE:", _fg_module.HAS_TLE)
    print("shapes:", SHAPES)
    print(
        "mode: native-boundary, three-branch E2E, "
        "shared TileOps C550 sliding fallback"
    )
    print(
        f"timing: CUDA Event, {CALLS_PER_SAMPLE} calls/sample, "
        f"{SAMPLES} samples, {ROUNDS} ABBA rounds"
    )

    results = [
        run_shape(total_tokens) for total_tokens in SHAPES
    ]
    speedups = [
        result["performance"]["speedup"] for result in results
    ]
    all_pass = all(
        result["performance"]["pass_90_percent"]
        for result in results
    )
    geometric_mean = math.exp(
        statistics.fmean(math.log(value) for value in speedups)
    )

    print(
        "\nT,TileOps_p50_ms,FlagGems_p50_ms,"
        "speedup,performance_percent,pass90"
    )
    for result in results:
        performance = result["performance"]
        print(
            f"{result['T']},"
            f"{performance['tileops']['p50_ms']:.6f},"
            f"{performance['flaggems']['p50_ms']:.6f},"
            f"{performance['speedup']:.6f},"
            f"{performance['performance_percent']:.2f},"
            f"{performance['pass_90_percent']}"
        )

    print("geometric_mean_speedup:", geometric_mean)
    print("worst_shape_speedup:", min(speedups))
    print("all_shapes_pass_90_percent:", all_pass)
    print(
        "NSA_MULTI_SHAPE_E2E_BENCHMARK:",
        "PASS" if all_pass else "FAIL",
    )
    raise SystemExit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
