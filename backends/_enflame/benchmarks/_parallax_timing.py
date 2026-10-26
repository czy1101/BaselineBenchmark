# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import statistics
import time


def check_close(name, expected, actual, tolerance, *, metric="rms"):
    assert expected.shape == actual.shape, f"{name}: output shapes differ"
    assert expected.dtype == actual.dtype, f"{name}: output dtypes differ"
    reference = expected.detach().float()
    result = actual.detach().float()
    assert bool(reference.isfinite().all().item()), f"{name}: baseline/reference contains NaN or Inf"
    assert bool(result.isfinite().all().item()), f"{name}: Triton/result contains NaN or Inf"
    difference = result - reference
    max_abs = difference.abs().max().item()
    if metric == "max":
        numerator = max_abs
        denominator = reference.abs().max().item()
        scale_floor = 1e-6
    elif metric == "rms":
        numerator = difference.square().mean().sqrt().item()
        denominator = reference.square().mean().sqrt().item()
        scale_floor = 1e-8
    else:
        raise ValueError(f"unsupported error metric: {metric}")
    error = numerator / max(denominator, scale_floor)
    valid = error < tolerance or (metric == "rms" and max_abs <= 1e-6)
    assert valid, (
        f"{name}: relative {metric} error={error:.3e}, max_abs={max_abs:.3e}, tolerance={tolerance:.3e}"
    )
    return error


def summarize_pair(baseline_samples, triton_samples):
    for samples in (baseline_samples, triton_samples):
        if not samples or any(not math.isfinite(value) or value <= 0 for value in samples):
            raise ValueError("latency samples must be nonempty, finite and positive")
    baseline_ms = statistics.median(baseline_samples)
    triton_ms = statistics.median(triton_samples)
    return {
        "baseline_ms": baseline_ms,
        "triton_ms": triton_ms,
        "speedup": baseline_ms / triton_ms,
        "baseline_mad_percent": 100 * statistics.median(abs(v - baseline_ms) for v in baseline_samples) / baseline_ms,
        "triton_mad_percent": 100 * statistics.median(abs(v - triton_ms) for v in triton_samples) / triton_ms,
    }


def validate_timing(warmup, iterations, inner, timing):
    if warmup < 0 or iterations < 1 or inner < 1:
        raise ValueError("warmup must be non-negative; iter and inner must be positive")
    if timing not in ("eager", "graph"):
        raise ValueError("timing must be eager or graph")


def _capture_graph(gcu, function, inner):
    graph = gcu.GCUGraph()
    with gcu.graph(graph):
        for _ in range(inner):
            function()
    gcu.synchronize()
    return graph


def _sample_ms(gcu, function, inner, timing):
    if timing == "graph":
        start = gcu.Event(enable_timing=True)
        end = gcu.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / inner
    gcu.synchronize()
    start = time.perf_counter()
    for _ in range(inner):
        function()
    gcu.synchronize()
    return (time.perf_counter() - start) * 1000 / inner


def measure_pair(gcu, baseline_fn, triton_fn, *, warmup, iterations, inner, timing, check=None):
    validate_timing(warmup, iterations, inner, timing)
    functions = (baseline_fn, triton_fn)
    # Compile and complete lazy work before either provider enters timing.
    for function in functions:
        function()
    gcu.synchronize()
    for cycle in range(warmup):
        order = (0, 1) if cycle % 2 == 0 else (1, 0)
        for index in order:
            functions[index]()
        gcu.synchronize()
    graphs = ()
    if timing == "graph":
        graphs = tuple(_capture_graph(gcu, function, inner) for function in functions)
        functions = tuple(graph.replay for graph in graphs)
        for _ in range(3):
            for function in functions:
                function()
        gcu.synchronize()
    if check is not None:
        check()
    samples = ([], [])
    # Mirrored ABBA/BAAB gives each provider the same sample count and budget.
    for cycle in range(iterations):
        order = (0, 1, 1, 0) if cycle % 2 == 0 else (1, 0, 0, 1)
        for index in order:
            samples[index].append(_sample_ms(gcu, functions[index], inner, timing))
    gcu.synchronize()
    if check is not None:
        check()
    return summarize_pair(*samples)
