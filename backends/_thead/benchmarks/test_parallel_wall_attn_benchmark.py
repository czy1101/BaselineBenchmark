# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Official-contract Wall-Attention benchmark against upstream FLA."""

from __future__ import annotations


import math
import statistics
import importlib
import inspect
import subprocess
import sys
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path


# Local T-Head FLA source checkout.
BENCHMARK_DIR = Path(__file__).resolve().parent
FLA_ROOT = (
    BENCHMARK_DIR.parent
    / "ops"
    / "FLA"
    / "wall_attn"
    / "flash-linear-attention-for-sail"
).resolve()

# Prioritize the local official FLA source.
if str(FLA_ROOT) in sys.path:
    sys.path.remove(str(FLA_ROOT))
sys.path.insert(0, str(FLA_ROOT))

import pytest
import torch
import triton

try:
    from benchmark.recording import benchmark_metric, record_benchmark_result
except ModuleNotFoundError:
    from recording import benchmark_metric, record_benchmark_result

import flag_attn
from flag_attn.testing import backend as test_backend
from flag_attn import parallel_wall_attn
from flag_attn.FLA.wall_attn import get_last_route, select_route
from flag_attn.utils import has_triton_tle

SEQUENCE_LENGTHS = (512, 1024, 2048, 4096)
SHAPE_FAMILIES = (("mha", 8), ("gqa", 2))
WARMUP_CALLS = 30
MEASUREMENT_MS = 200
REPEATS = 7


def _source_revision(directory: Path) -> str | None:
    """Record a Git revision when running from a checkout, without requiring Git."""
    try:
        result = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None



def _environment():
    import fla

    device = test_backend.device_fn.current_device()

    try:
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version",
             "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        driver_version = (
            driver.stdout.splitlines()[0]
            if driver.returncode == 0 and driver.stdout
            else None
        )
    except (OSError, subprocess.TimeoutExpired):
        driver_version = None

    return {
        "gpu_model": test_backend.get_device_name(device),
        "compute_capability": list(test_backend.cuda_capability(device)),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "triton_version": str(triton.__version__),
        "driver_version": driver_version,
        "fla_version": getattr(fla, "__version__", None),
        "fla_revision": _source_revision(FLA_ROOT),
        "fla_source": str(FLA_ROOT),
        "candidate_revision": _source_revision(
            Path(flag_attn.__file__).resolve().parents[2]
        ),
    }


@lru_cache(maxsize=1)
def _load_fla_provider():
    # Do not silently fall back to a single-provider benchmark.
    provider_path = FLA_ROOT / "fla" / "ops" / "wall_attn" / "parallel.py"

    if not provider_path.is_file():
        raise FileNotFoundError(
            f"Official FLA source not found: {provider_path}. "
            "Run 'git submodule update --init --recursive' from the "
            "BaselineBenchmark repository root."
        )

    # Detect whether another fla package was imported earlier.
    existing_fla = sys.modules.get("fla")
    if existing_fla is not None:
        existing_file = getattr(existing_fla, "__file__", None)
        if (
            existing_file is None
            or FLA_ROOT not in Path(existing_file).resolve().parents
        ):
            raise RuntimeError(
                "Another fla package was already imported. "
                "Please run this benchmark in a fresh Python process."
            )

    try:
        importlib.invalidate_caches()

        import fla
        from fla.ops.wall_attn import (
            parallel_wall_attn as fla_wall_attn,
        )

    except Exception as exc:
        raise RuntimeError(
            f"Failed to import official FLA from {FLA_ROOT}"
        ) from exc

    fla_init = Path(fla.__file__).resolve()

    if FLA_ROOT not in fla_init.parents:
        raise RuntimeError(
            f"Wrong fla package loaded: {fla_init}"
        )

    source_file = Path(
        inspect.getsourcefile(fla_wall_attn)
        or inspect.getfile(fla_wall_attn)
    ).resolve()

    if FLA_ROOT not in source_file.parents:
        raise RuntimeError(
            f"Wrong Wall Attention implementation loaded: {source_file}"
        )

    print("[Official FLA] root:", FLA_ROOT, flush=True)
    print("[Official FLA] package:", fla_init, flush=True)
    print("[Official FLA] implementation:", source_file, flush=True)
    print("[Official FLA] revision:", _source_revision(FLA_ROOT), flush=True)

    return fla_wall_attn


def _make_inputs(t: int, h: int, dtype: torch.dtype):
    torch.manual_seed(0)
    q = torch.randn(1, t, 8, 64, device=test_backend.device, dtype=dtype)
    k = torch.randn(1, t, h, 64, device=test_backend.device, dtype=dtype)
    v = torch.randn(1, t, h, 64, device=test_backend.device, dtype=dtype)
    g = (-torch.randn(1, t, 8, 64, device=test_backend.device).abs() * 0.05).to(dtype)
    return q, k, v, g


def _bench_ms(fn: Callable[[], torch.Tensor]) -> float:
    return float(
        test_backend.do_bench(
            fn,
            warmup=0,
            rep=MEASUREMENT_MS,
            return_mode="median",
        )
    )


def _summary(samples: list[float]) -> dict[str, float | list[float]]:
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "std_ms": statistics.pstdev(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


@torch.inference_mode()
def _run_benchmark(dtype_name, dtype, record_property=None):
    fla_wall_attn = _load_fla_provider()
    environment = _environment()
    metrics = []
    speedups = []

    print("dtype family T FLA_ms Final_ms speedup")
    for family, h in SHAPE_FAMILIES:
        for t in SEQUENCE_LENGTHS:
            inputs = _make_inputs(t, h, dtype)
            scale = 64**-0.5
            providers = {"optimized": lambda: parallel_wall_attn(*inputs, scale=scale)}
            if fla_wall_attn is not None:
                providers["fla"] = lambda: fla_wall_attn(*inputs, scale=scale)
            generic_hopper = (test_backend.is_nvidia() and test_backend.cuda_capability() == (9, 0)
                              and has_triton_tle() and not test_backend.has_specialization("parallel_wall_attn"))
            if generic_hopper:
                expected_route = "hopper_bf16" if dtype is torch.bfloat16 else "hopper_fp16"
                assert select_route(*inputs, scale=scale) == expected_route

            outputs = {name: fn() for name, fn in providers.items()}
            actual_route = get_last_route() if not test_backend.has_specialization("parallel_wall_attn") else test_backend.runtime.device.vendor_name
            test_backend.device_fn.synchronize()
            for name, output in outputs.items():
                assert torch.isfinite(output).all(), f"{name} produced NaN/Inf"
            if "fla" in outputs:
                torch.testing.assert_close(outputs["optimized"], outputs["fla"], atol=0.05, rtol=0.05)

            provider_names = tuple(providers)
            for warmup_index in range(WARMUP_CALLS):
                order = provider_names if warmup_index % 2 == 0 else provider_names[::-1]
                for name in order:
                    providers[name]()
            test_backend.device_fn.synchronize()

            samples = {name: [] for name in provider_names}
            for repeat in range(REPEATS):
                order = provider_names if repeat % 2 == 0 else provider_names[::-1]
                for name in order:
                    samples[name].append(_bench_ms(providers[name]))

            latency = {name: _summary(values) for name, values in samples.items()}
            fla_ms = float(latency["fla"]["median_ms"]) if "fla" in latency else None
            optimized_ms = float(latency["optimized"]["median_ms"])
            speedup = fla_ms / optimized_ms if fla_ms is not None else None
            if speedup is not None:
                speedups.append(speedup)
            print(
                dtype_name,
                family,
                t,
                "N/A" if fla_ms is None else f"{fla_ms:.6f}",
                f"{optimized_ms:.6f}",
                "N/A" if speedup is None else f"{speedup:.4f}",
                flush=True,
            )
            metrics.append(
                benchmark_metric(
                    shape_detail={
                        "B": 1,
                        "T": t,
                        "H": h,
                        "HQ": 8,
                        "D": 64,
                        "family": family,
                    },
                    latency_base=fla_ms,
                    latency=optimized_ms,
                    speedup=speedup,
                    provider_route=actual_route,
                    samples=latency,
                )
            )

    geomean_speedup = math.exp(sum(math.log(value) for value in speedups) / len(speedups)) if speedups else None
    if geomean_speedup is not None:
        print(f"geomean {dtype_name}: {geomean_speedup:.4f}x vs FLA", flush=True)
    record_benchmark_result(
        record_property,
        op_name="parallel_wall_attn",
        dtype=dtype_name,
        result=metrics,
        baseline="official_fla" if fla_wall_attn is not None else None,
        phase="forward",
        hardware=test_backend.get_device_name(),
        input_contract="gate_scale=0.05, seed=0",
        warmup_calls=WARMUP_CALLS,
        measurement_ms=MEASUREMENT_MS,
        repeats=REPEATS,
        geomean_speedup_vs_fla=geomean_speedup,
        environment=environment,
    )


@pytest.mark.parallel_wall_attn
@pytest.mark.parametrize(
    "dtype_name,dtype",
    (("bf16", torch.bfloat16), ("fp16", torch.float16)),
    ids=("bf16", "fp16"),
)
@pytest.mark.skipif(
    not test_backend.supports_operator("parallel_wall_attn"),
    reason="requires an available accelerator",
)
def test_parallel_wall_attn_benchmark(dtype_name, dtype, record_property):
    _run_benchmark(dtype_name, dtype, record_property)
