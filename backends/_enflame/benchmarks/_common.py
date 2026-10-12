"""Shared utilities for isolated S60 benchmarks."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import torch


def synchronize() -> None:
    torch.gcu.synchronize()


def measure(
    function: Callable[[], Any],
    *,
    warmup: int,
    repeat: int,
    inner: int,
    print_iterations: bool = True,
) -> tuple[list[float], Any]:
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if repeat <= 0:
        raise ValueError("repeat must be positive")
    if inner <= 0:
        raise ValueError("inner must be positive")

    result = None

    with torch.no_grad():
        for _ in range(warmup):
            for _ in range(inner):
                result = function()
            synchronize()

        samples = []

        for iteration in range(1, repeat + 1):
            synchronize()
            start = time.perf_counter()

            for _ in range(inner):
                result = function()

            synchronize()

            latency_ms = (
                time.perf_counter() - start
            ) * 1000.0 / inner

            samples.append(latency_ms)

            if print_iterations:
                print(
                    f"ITERATION index={iteration} "
                    f"inner={inner} "
                    f"latency_ms={latency_ms:.6f}",
                    flush=True,
                )

    return samples, result


def summarize(
    *,
    operator: str,
    stage: str,
    implementation: str,
    case: str,
    dtype: str,
    samples: list[float],
    finite: bool,
) -> dict[str, Any]:
    record = {
        "operator": operator,
        "stage": stage,
        "implementation": implementation,
        "case": case,
        "dtype": dtype,
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "finite": finite,
    }

    print(
        "RESULT "
        f"operator={operator} "
        f"stage={stage} "
        f"implementation={implementation} "
        f"case={case} "
        f"dtype={dtype} "
        f"median_ms={record['median_ms']:.6f} "
        f"min_ms={record['min_ms']:.6f} "
        f"max_ms={record['max_ms']:.6f} "
        f"finite={finite}",
        flush=True,
    )

    return record


def write_json(
    output_path: str | None,
    record: dict[str, Any],
) -> None:
    if output_path is None:
        return

    path = Path(output_path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        json.dumps(
            record,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"JSON_RESULT path={path}", flush=True)
