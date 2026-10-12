from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path


PROJECT_SRC = Path(__file__).resolve().parents[4] / "FlagAttention" / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

import torch

try:
    import triton_gcu.triton  # noqa: F401
except ModuleNotFoundError:
    import triton  # noqa: F401

import triton.language.math as tl_math

# Compatibility aliases for the Triton version bundled in the Enflame image.
tl_math.tanh = getattr(tl_math, "tanh", lambda x: x)
tl_math.exp = getattr(tl_math, "exp", lambda x: x)
tl_math.erf = getattr(tl_math, "erf", lambda x: x)
tl_math.pow = getattr(tl_math, "pow", lambda x, y: x**y)

from flag_attn import chunk_gla as current_chunk_gla

OPS_DIR = Path(__file__).resolve().parents[1] / "ops" / "chunk_gla"
sys.path.insert(0, str(OPS_DIR))
from chunk_gla_tops import chunk_gla_tops
from chunk_gla_baseline_utils import select_safe_chunk_size


SHAPES = [
    (1, 8192, 96, 128),
    (2, 16384, 16, 128),
    (4, 2048, 16, 128),
    (4, 4096, 64, 128),
    (8, 2048, 32, 256),
    (2, 2048, 16, 512),
    (4, 1024, 8, 512),
    (8, 1024, 8, 64),
]


def synchronize() -> None:
    torch.gcu.synchronize()


def measure(function, warmup: int, repetitions: int) -> tuple[float, float, float]:
    for _ in range(warmup):
        function()
        synchronize()
    samples = []
    for _ in range(repetitions):
        synchronize()
        start = time.perf_counter()
        function()
        synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples), min(samples), max(samples)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iter", type=int, default=10, dest="repetitions")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-current", action="store_true")
    parser.add_argument("--shape-index", type=int)
    return parser.parse_args()


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()
    if args.shape_index is not None:
        shapes = [SHAPES[args.shape_index]]
    else:
        shapes = SHAPES[-2:] if args.quick else SHAPES
    device = torch.device("gcu")
    dtype = torch.bfloat16

    print(
        f"device={device} vendor=enflame dtype={dtype} "
        f"baseline=tops_cpp_topsaten "
        f"warmup={args.warmup} iter={args.repetitions} "
        f"chunk={args.chunk_size}"
    )
    if args.skip_current:
        print("B T H D safe_chunk baseline_ms")
    else:
        print("B T H D safe_chunk current_ms baseline_ms baseline/current")

    for batch, seqlen, heads, dim in shapes:
        torch.manual_seed(42)
        q = torch.randn(batch, seqlen, heads, dim, device=device, dtype=dtype)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        g = torch.nn.functional.logsigmoid(torch.randn_like(q))
        kwargs = {"scale": dim**-0.5, "output_final_state": False}
        safe_chunk = select_safe_chunk_size(g, args.chunk_size)

        baseline_output, _ = chunk_gla_tops(
            q, k, v, g, chunk_size=args.chunk_size, **kwargs
        )
        synchronize()
        if not bool(torch.isfinite(baseline_output).all().cpu()):
            raise RuntimeError("baseline produced non-finite output")

        current_ms = None
        if not args.skip_current:
            current_ms = measure(
                lambda: current_chunk_gla(q, k, v, g, **kwargs),
                args.warmup,
                args.repetitions,
            )
        baseline_ms = measure(
            lambda: chunk_gla_tops(
                q, k, v, g, chunk_size=args.chunk_size, **kwargs
            ),
            args.warmup,
            args.repetitions,
        )

        if current_ms is None:
            print(
                f"{batch} {seqlen} {heads} {dim} {safe_chunk} "
                f"{baseline_ms[0]:.3f}"
            )
        else:
            print(
                f"{batch} {seqlen} {heads} {dim} {safe_chunk} "
                f"{current_ms[0]:.3f} {baseline_ms[0]:.3f} "
                f"{baseline_ms[0] / current_ms[0]:.2f}x"
            )


def test_chunk_gla_benchmark(benchmark_options) -> None:
    args = argparse.Namespace(
        warmup=benchmark_options.get("warmup", 2),
        repetitions=benchmark_options.get("iterations", 10),
        chunk_size=benchmark_options.get("chunk_size", 256),
        quick=benchmark_options.get("quick", False),
        skip_current=benchmark_options.get("skip_current", False),
        shape_index=benchmark_options.get("shape_index"),
    )
    main(args)


if __name__ == "__main__":
    main()
