# SPDX-License-Identifier: Apache-2.0 AND MIT
# Copyright 2026 FlagOS Contributors
"""Performance-only native TileOPs/Triton comparison.

From BaselineBenchmark:
    python backends/_nvidia/benchmarks/test_apply_min_p.py --warmup 100 --iter 300
    python -m pytest -s backends/_nvidia/benchmarks/test_apply_min_p.py -m apply_min_p

Requires importable flaggems_vllm, TileOPs, TileLang >= 0.1.12, cuda-bindings,
PyTorch/CUDA and the CUDA toolchain. Install the libraries or set PYTHONPATH.
"""

import argparse
import functools
import hashlib
import inspect
import json
import math
import os
import statistics
import time
import warnings
from pathlib import Path

import pytest
import torch
from packaging.version import Version

# Cold-L2 preparation helpers follow TileOPs benchmarks/timing.py.
# MIT License
#
# Copyright (c) Tile-AI.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE


def _tileops_baseline():
    import tilelang
    from tileops.backend import BUILTIN
    from tileops.sampling import MinPMaskFwdOp

    return MinPMaskFwdOp(target=BUILTIN), tilelang.__version__


@functools.lru_cache(maxsize=8)
def _get_l2_flush_cache(device):
    l2_bytes = getattr(torch.cuda.get_device_properties(device), "L2_cache_size", 0)
    if l2_bytes <= 0:
        warnings.warn("L2 size unavailable; using the upstream 512 MB flush buffer")
        l2_bytes = int(256e6)
    return torch.empty(2 * l2_bytes, dtype=torch.int8, device=device)


def _reset_persisting_l2_cache():
    from cuda.bindings import runtime

    result = runtime.cudaCtxResetPersistingL2Cache()
    if isinstance(result, tuple):
        result = result[0]
    torch.cuda.check_error(int(result))


def _measure_device_time(fn, reset, device, warmup_ms, repeat_ms, label):
    """Measure current-stream elapsed time, including inter-kernel gaps.

    Reset and cold-L2 preparation precede the start event on the same stream.
    Warmup and sampling budgets use wall time, including preparation and
    synchronization; each sampling phase collects at least five readings.
    Compilation and lazy event initialization are outside both budgets.
    """
    if not math.isfinite(warmup_ms) or warmup_ms < 0:
        raise ValueError("warmup_ms must be finite and >= 0")
    if not math.isfinite(repeat_ms) or repeat_ms <= 0:
        raise ValueError("repeat_ms must be finite and > 0")
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Min-P CUDA Event timing requires a CUDA device")

    with torch.cuda.device(device):
        stream = torch.cuda.current_stream(device)
        cache = _get_l2_flush_cache(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        end.record(stream)
        end.synchronize()

        def prepare():
            reset()
            _reset_persisting_l2_cache()
            cache.zero_()

        # Compile/autotune and finish first allocations before collecting samples.
        prepare()
        _ = fn()
        stream.synchronize()
        del _

        warmup_start = time.perf_counter()
        warmup_count = 0
        while (time.perf_counter() - warmup_start) * 1000 < warmup_ms:
            prepare()
            _ = fn()
            stream.synchronize()
            del _
            warmup_count += 1
        warmup_wall_ms = (time.perf_counter() - warmup_start) * 1000

        samples = []
        sample_start = time.perf_counter()
        while (
            len(samples) < 5 or (time.perf_counter() - sample_start) * 1000 < repeat_ms
        ):
            prepare()
            start.record(stream)
            _ = fn()
            end.record(stream)
            end.synchronize()
            samples.append({"latency_ms": start.elapsed_time(end)})
            # Keep a native out-of-place result alive until the GPU is finished.
            del _
        sample_wall_ms = (time.perf_counter() - sample_start) * 1000

    return samples, {
        "timing": "cuda-event",
        "device_metric": "complete-call current-stream elapsed time",
        "includes_inter_kernel_gaps": True,
        "preparation": "input reset and cold-L2 flush before the start event",
        "budget_basis": "wall time including preparation and synchronization",
        "warmup_ms": warmup_ms,
        "sample_ms": repeat_ms,
        "warmup_wall_ms": warmup_wall_ms,
        "sample_wall_ms": sample_wall_ms,
        "warmup_iterations": warmup_count,
        "sample_iterations": len(samples),
        "label": label,
    }


DEFAULT_SHAPES = [
    (1, 128256, "bfloat16"),
    (1, 128256, "float32"),
    (64, 128256, "bfloat16"),
    (256, 128256, "float16"),
    (256, 128256, "bfloat16"),
    (128, 151936, "bfloat16"),
    (17, 129280, "bfloat16"),
    (100, 32000, "float16"),
]
CASE_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _load_shapes(shape_file):
    if shape_file is None:
        return DEFAULT_SHAPES
    import yaml

    with Path(shape_file).open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data["apply_min_p"]["shapes"]


def _input_iter(shapes, dtype, device):
    generator = torch.Generator(device=device).manual_seed(2026)
    for shape in shapes:
        if len(shape) != 3:
            raise ValueError("Expected apply_min_p shape (batch, vocab, dtype)")
        batch, vocab, case_dtype = shape
        if (
            not isinstance(batch, int)
            or not isinstance(vocab, int)
            or batch <= 0
            or vocab <= 0
            or case_dtype not in CASE_DTYPES
        ):
            raise ValueError(
                "Expected positive integer batch/vocab and float16/bfloat16/float32"
            )
        if dtype != CASE_DTYPES[case_dtype]:
            continue
        logits = torch.randn(
            (batch, vocab), device=device, dtype=dtype, generator=generator
        )
        mapping = torch.arange(batch, device=device, dtype=torch.int32)
        min_p = torch.full((batch,), 0.1, device=device, dtype=torch.float32)
        yield logits, mapping, min_p


def _sample_stats(samples):
    values = sorted(sample["latency_ms"] for sample in samples)
    return {
        "samples": len(values),
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "p99_ms": (
            values[math.ceil(0.99 * len(values)) - 1] if len(values) >= 100 else None
        ),
    }


def run_benchmark(
    *,
    warmup_ms=100,
    repeat_ms=300,
    device_index=0,
    dtypes=None,
    shape_file=None,
    output=None,
):
    if not math.isfinite(warmup_ms) or warmup_ms < 0:
        raise ValueError("warmup must be finite and >= 0 (milliseconds)")
    if not math.isfinite(repeat_ms) or repeat_ms <= 0:
        raise ValueError("iter must be finite and > 0 (milliseconds)")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    if output is not None and Path(output).exists():
        raise FileExistsError(f"Refusing to overwrite existing result: {output}")
    selected = list(CASE_DTYPES) if dtypes is None else dtypes
    if not selected or any(name not in CASE_DTYPES for name in selected):
        raise ValueError("dtypes must select float16, bfloat16 or float32")
    shapes = _load_shapes(shape_file)
    if not shapes:
        raise ValueError("At least one shape is required")

    import tilelang
    from flaggems_vllm.ops.apply_min_p import apply_min_p
    from tileops._kernels.sampling.min_p_mask import MinPMaskFwdKernel

    if Version(tilelang.__version__) < Version("0.1.12"):
        raise RuntimeError("TileLang >= 0.1.12 required")
    tileops_op, tilelang_version = _tileops_baseline()
    kernel_path = Path(inspect.getfile(MinPMaskFwdKernel))
    triton_path = Path(inspect.getfile(apply_min_p))
    device = torch.device("cuda", device_index)
    source = {
        "tileops_commit": os.environ.get("MIN_P_TILEOPS_COMMIT", "unknown"),
        "baseline_implementation": "original TileOPs MinPMaskFwdOp API",
        "tilelang_version": tilelang_version,
        "kernel_path": str(kernel_path),
        "kernel_sha256": hashlib.sha256(kernel_path.read_bytes()).hexdigest(),
        "triton_path": str(triton_path),
        "triton_sha256": hashlib.sha256(triton_path.read_bytes()).hexdigest(),
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(device),
        "device_index": device_index,
    }
    protocol = {
        "timer": "cuda-event",
        "metric": "latency_ms",
        "measurement": "complete-call current-stream elapsed time, including kernel gaps",
        "phase_order": ["tileops", "triton", "triton", "tileops"],
        "warmup_ms_per_implementation": warmup_ms,
        "sample_ms_per_implementation": repeat_ms,
        "minimum_samples_per_phase": 5,
        "budget_basis": "wall time including preparation and synchronization",
        "l2": "persisting-L2 reset and flush before every invocation, after input reset",
        "input_reset": "private input restored outside every timed invocation",
        "min_p": 0.1,
        "seed": 2026,
        "tileops_api": "out-of-place, native output/scratch allocation",
        "triton_api": "in-place, identity request mapping",
    }
    print("Baseline: original TileOPs/TileLang API; timer: CUDA Event")
    print(json.dumps(source))
    details = []
    with torch.cuda.device(device):
        for dtype_name, dtype in CASE_DTYPES.items():
            if dtype_name not in selected:
                continue
            rows = []
            print(f"\napply_min_p dtype={dtype}; times in ms; speedup=TileOPs/Triton")
            print(
                f"{'Batch':>8} {'Vocab':>10} {'TileOPs':>12} {'Triton':>12} {'Speedup':>10}"
            )
            for logits, mapping, min_p in _input_iter(shapes, dtype, device):
                batch, vocab = logits.shape
                tileops_logits, triton_logits = torch.empty_like(
                    logits
                ), torch.empty_like(logits)
                tileops_logits.copy_(logits)
                triton_logits.copy_(logits)
                calls = {
                    "tileops": lambda: tileops_op(tileops_logits, min_p),
                    "triton": lambda: apply_min_p(triton_logits, mapping, min_p),
                }
                resets = {
                    "tileops": lambda: tileops_logits.copy_(logits),
                    "triton": lambda: triton_logits.copy_(logits),
                }
                calls["tileops"]()
                calls["triton"]()
                torch.cuda.synchronize()
                samples = {"tileops": [], "triton": []}
                phases = []
                for phase_index, tag in enumerate(protocol["phase_order"]):
                    measured, meta = _measure_device_time(
                        calls[tag],
                        reset=resets[tag],
                        device=device,
                        warmup_ms=warmup_ms / 2,
                        repeat_ms=repeat_ms / 2,
                        label=f"{phase_index}-{tag}",
                    )
                    samples[tag].extend(measured)
                    phases.append({"implementation": tag, "metadata": meta})
                baseline_ms = statistics.median(
                    s["latency_ms"] for s in samples["tileops"]
                )
                triton_ms = statistics.median(
                    s["latency_ms"] for s in samples["triton"]
                )
                row = {
                    "shape_detail": {
                        "batch": batch,
                        "vocab": vocab,
                        "inputs": [list(t.shape) for t in (logits, mapping, min_p)],
                    },
                    "latency_base": baseline_ms,
                    "latency": triton_ms,
                    "speedup": baseline_ms / triton_ms,
                    "samples": samples,
                    "phases": phases,
                    "statistics": {
                        tag: _sample_stats(values) for tag, values in samples.items()
                    },
                }
                rows.append(row)
                print(
                    f"{batch:>8} {vocab:>10} {baseline_ms:>12.6f} {triton_ms:>12.6f} {row['speedup']:>9.3f}x",
                    flush=True,
                )
            if rows:
                details.append(
                    {
                        "op_name": "apply_min_p",
                        "dtype": str(dtype),
                        "level": "core",
                        "mode": "kernel",
                        "timer": "cuda-event",
                        "baseline": "tileops-tilelang",
                        "source": source,
                        "protocol": protocol,
                        "result": rows,
                    }
                )
    if not details:
        raise ValueError("No shapes match the selected dtypes")
    result = {"apply_min_p": {"details": details}}
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
    return result


@pytest.mark.apply_min_p
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_apply_min_p(request):
    pytest.importorskip("tilelang", minversion="0.1.12")
    pytest.importorskip("tileops")
    pytest.importorskip("cuda.bindings.runtime")
    pytest.importorskip("flaggems_vllm")
    run_benchmark(
        warmup_ms=request.config.getoption("min_p_warmup_ms"),
        repeat_ms=request.config.getoption("min_p_repeat_ms"),
        device_index=request.config.getoption("min_p_device"),
        dtypes=request.config.getoption("min_p_dtypes"),
        shape_file=request.config.getoption("min_p_shape_file"),
        output=request.config.getoption("min_p_output"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--warmup", type=float, default=100, help="warmup wall-time budget in ms"
    )
    parser.add_argument(
        "--iter",
        type=float,
        default=300,
        dest="repeat_ms",
        help="sample wall-time budget in ms",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dtypes", nargs="+", choices=tuple(CASE_DTYPES))
    parser.add_argument("--shape-file")
    parser.add_argument("--output")
    args = parser.parse_args()
    run_benchmark(
        warmup_ms=args.warmup,
        repeat_ms=args.repeat_ms,
        device_index=args.device,
        dtypes=args.dtypes,
        shape_file=args.shape_file,
        output=args.output,
    )
