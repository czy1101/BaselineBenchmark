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

from dataclasses import replace
from importlib import import_module

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
_accelerator = pytest.importorskip("flag_attn.runtime.accelerator")

from backends._enflame.benchmarks._parallax_timing import check_close, measure_pair, validate_timing

try:
    ACCELERATOR = _accelerator.detect_accelerator(("gcu",))
except RuntimeError as exc:
    pytest.skip(str(exc), allow_module_level=True)

PARALLAX_MARKS = [
    pytest.mark.parallax,
    pytest.mark.skipif(ACCELERATOR is None, reason="Parallax comparisons require an Enflame GCU"),
]

# Parallel: (B, T, HKV, HQ, D). Decode: (B, L, HQ, HKV, D).
PARALLEL_CASES = (
    (1, 15, 2, 2, 64), (1, 63, 1, 1, 64), (1, 111, 2, 2, 64),
    (2, 200, 2, 8, 64), (2, 256, 2, 8, 64), (2, 512, 2, 8, 64),
    (2, 1024, 2, 2, 64), (2, 2048, 2, 8, 64),
    (2, 4096, 2, 2, 64), (2, 8192, 2, 8, 64),
    (3, 111, 2, 2, 100), (3, 1024, 2, 8, 128),
    (2, 2048, 2, 8, 128), (2, 8192, 8, 16, 128),
)
DECODE_CASES = (
    (1, 512, 8, 8, 64), (1, 512, 8, 8, 128), (1, 1024, 8, 8, 128),
    (1, 2048, 8, 8, 128), (1, 4096, 8, 8, 128), (1, 16384, 8, 8, 128),
    (2, 1000, 8, 2, 128), (2, 2048, 8, 2, 64), (2, 2048, 8, 2, 128),
    (1, 8192, 8, 2, 128),
)


def parametrize_cases(metafunc, cases):
    if "case" not in metafunc.fixturenames:
        return
    index = metafunc.config.getoption("shape_index")
    run_all = metafunc.config.getoption("all")
    if run_all and index is not None:
        raise pytest.UsageError("choose either --all or --shape-index")
    if run_all:
        selected = list(enumerate(cases))
    else:
        index = 0 if index is None else index
        if not 0 <= index < len(cases):
            raise pytest.UsageError(f"--shape-index must be in [0, {len(cases) - 1}]")
        selected = [(index, cases[index])]
    metafunc.parametrize("case", [case for _, case in selected], ids=[f"shape-{i}" for i, _ in selected])


def _settings(options, *, decode):
    if ACCELERATOR is None:
        pytest.skip("Parallax comparisons require an Enflame GCU")
    index = options.get("device", ACCELERATOR.index)
    if index < 0 or index >= torch.gcu.device_count():
        pytest.skip(f"GCU device {index} is unavailable")
    accelerator = replace(ACCELERATOR, index=index)
    _accelerator.set_device(accelerator)
    timing = options.get("timing", "graph" if decode else "eager")
    warmup = options.get("warmup", 10)
    iterations = options.get("iterations", 12)
    inner = options.get("calls_per_graph", 100) if timing == "graph" else options.get("inner", 1)
    validate_timing(warmup, iterations, inner, timing)
    if not decode and timing == "graph":
        raise pytest.UsageError("parallel forward/fwdbwd comparison uses --timing eager")
    if timing == "graph" and not all(hasattr(torch.gcu, name) for name in ("GCUGraph", "graph", "Event")):
        pytest.skip("GCU graph capture and device events are unavailable")
    dtype_name = options.get("dtype", "bfloat16")
    dtype = getattr(torch, dtype_name)
    window = options.get("window_size")
    if window is not None and window < (0 if decode else 1):
        raise pytest.UsageError("--window-size must be non-negative for decode and positive for parallel")
    return accelerator, dtype, window, {
        "warmup": warmup, "iterations": iterations, "inner": inner, "timing": timing,
    }


def _implementations(baseline, *, decode, accelerator):
    if baseline != "torch":
        raise pytest.UsageError("Parallax comparisons only support the Torch-GCU baseline")
    name = "parallax_decode" if decode else "parallax"
    module_path = "FLA.parallax.decode" if decode else "FLA.parallax.parallel"
    try:
        base = import_module(f"backends._enflame.ops.parallax.{name}_torch")
        current = import_module(f"flag_attn.runtime.backend._enflame.{module_path}")
    except (ImportError, OSError) as exc:
        pytest.skip(f"Parallax implementation/dependency is unavailable: {exc}")
    available = base.torch_gcu_parallax_decode_available if decode else base.torch_gcu_parallax_available
    if not available():
        pytest.skip(f"{baseline} Parallax baseline is unavailable")
    if decode and not current.HAS_TLE:
        pytest.skip("FlagTree TLE decode is unavailable")
    return base, current


def _report(case, baseline, phase, dtype, settings, result, error, base, current, options):
    options.add_parallax_result({
        "case": tuple(case),
        "baseline": baseline,
        "phase": phase,
        "dtype": str(dtype).removeprefix("torch."),
        "device": options.get("device", ACCELERATOR.index),
        "window_size": options.get("window_size"),
        "baseline_source": base.__file__,
        "triton_source": current.__file__,
        "error": error,
        **settings,
        **result,
    })


def run_parallel_comparison(case, baseline, options):
    accelerator, dtype, window, settings = _settings(options, decode=False)
    base, current = _implementations(baseline, decode=False, accelerator=accelerator)
    batch, length, kv_heads, query_heads, dim = case
    seed = options.get("seed", 2026) + sum((i + 1) * value for i, value in enumerate(case))
    generator = torch.Generator(device=accelerator.device).manual_seed(seed)
    q = torch.randn(batch, length, query_heads, dim, device=accelerator.device, dtype=dtype, generator=generator)
    r = torch.randn(q.shape, device=q.device, dtype=dtype, generator=generator)
    k = torch.randn(batch, length, kv_heads, dim, device=q.device, dtype=dtype, generator=generator)
    v = torch.randn(k.shape, device=q.device, dtype=dtype, generator=generator)
    mode = options.get("mode", "both")
    if mode != "forward":
        for tensor in (q, r, k, v):
            tensor.requires_grad_(True)
    grad_output = torch.randn(q.shape, device=q.device, dtype=dtype, generator=generator)
    scale = dim ** -0.5
    kwargs = {"scale": scale, "window_size": window}
    base_kwargs = dict(kwargs)
    base_kwargs["max_score_elements"] = options.get("max_score_elements", 8 * 1024 * 1024)
    baseline_op = base.parallel_parallax_torch_gcu

    def baseline_forward():
        with torch.no_grad():
            return baseline_op(q, r, k, v, **base_kwargs)

    def triton_forward():
        with torch.no_grad():
            return current.parallel_parallax(q, r, k, v, **kwargs)

    tolerance = 5e-3 if dtype == torch.float16 else 2e-2
    with torch.no_grad():
        expected = baseline_forward().detach().clone()
        actual = triton_forward()
        error = check_close("forward", expected, actual, tolerance)
    del expected, actual
    phases = ("forward", "fwdbwd") if mode == "both" else (mode,)
    for phase in phases:
        if phase == "forward":
            functions = (baseline_forward, triton_forward)
            phase_error = error
        else:
            def baseline_fwdbwd():
                return torch.autograd.grad(baseline_op(q, r, k, v, **base_kwargs), (q, r, k, v), grad_output)

            def triton_fwdbwd():
                return torch.autograd.grad(current.parallel_parallax(q, r, k, v, **kwargs), (q, r, k, v), grad_output)

            functions = (baseline_fwdbwd, triton_fwdbwd)
            # Release the baseline autograd graph before creating Triton's.
            expected_grads = tuple(grad.detach().clone() for grad in baseline_fwdbwd())
            actual_grads = triton_fwdbwd()
            phase_error = max(
                check_close(name, expected, actual, tolerance)
                for name, expected, actual in zip(("dq", "dr", "dk", "dv"), expected_grads, actual_grads)
            )
            del expected_grads, actual_grads
        result = measure_pair(torch.gcu, *functions, **settings)
        _report(case, baseline, phase, dtype, settings, result, phase_error, base, current, options)


def run_decode_comparison(case, baseline, options):
    accelerator, dtype, window, settings = _settings(options, decode=True)
    base, current = _implementations(baseline, decode=True, accelerator=accelerator)
    try:
        reference_module = import_module("backends._enflame.ops.parallax.parallax_decode_torch")
    except (ImportError, OSError) as exc:
        pytest.skip(f"Torch decode reference is unavailable: {exc}")
    batch, length, query_heads, kv_heads, dim = case
    seed = options.get("seed", 2026) + sum((i + 1) * value for i, value in enumerate(case))
    generator = torch.Generator(device=accelerator.device).manual_seed(seed)
    q = torch.randn(batch, 1, query_heads, dim, device=accelerator.device, dtype=dtype, generator=generator)
    r = torch.randn(q.shape, device=q.device, dtype=dtype, generator=generator) * 0.5
    k = torch.randn(batch, length, kv_heads, dim, device=q.device, dtype=dtype, generator=generator)
    v = torch.randn(k.shape, device=q.device, dtype=dtype, generator=generator)
    baseline_out, triton_out = torch.empty_like(q), torch.empty_like(q)
    scale = dim ** -0.5
    kwargs = {"window_size_left": -1 if window is None else window}
    baseline_op = base.parallax_decode_torch_gcu

    def baseline_forward():
        with torch.no_grad():
            return baseline_op(q, r, k, v, scale, out=baseline_out, **kwargs)

    def triton_forward():
        with torch.no_grad():
            return current.parallax_decode(q, r, k, v, scale, out=triton_out, **kwargs)

    with torch.no_grad():
        reference = reference_module.parallax_decode_torch_gcu(q, r, k, v, scale, **kwargs).detach().clone()
    baseline_forward()
    triton_forward()
    _accelerator.synchronize(accelerator)
    errors = []

    def validate_outputs():
        with torch.no_grad():
            errors.extend((
                check_close("baseline decode/reference", reference, baseline_out, 1e-2, metric="max"),
                check_close("Triton decode/reference", reference, triton_out, 1e-2, metric="max"),
            ))

    validate_outputs()
    result = measure_pair(torch.gcu, baseline_forward, triton_forward, **settings, check=validate_outputs)
    _report(case, baseline, "decode", dtype, settings, result, max(errors), base, current, options)
