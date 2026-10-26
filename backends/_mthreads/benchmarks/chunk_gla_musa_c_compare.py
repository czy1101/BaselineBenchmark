# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Compare the MUSA C chunk GLA baseline with production Triton/TLE."""

from __future__ import annotations

import argparse
from collections.abc import Callable

import torch
import torch.nn.functional as F
import triton

try:
    import torch_musa  # noqa: F401
except ImportError:
    torch_musa = None

from BaselineBenchmark.backends._mthreads.ops.chunk_gla_musa_c import (
    is_available as musa_c_available,
    musa_chunk_gla,
    torch_recurrent_chunk_gla,
)
from flag_attn.runtime.backend._mthreads.gated_linear_attention.chunk_gla import (
    ChunkGLAFunction,
)


DEFAULT_SHAPES = (
    (1, 8192, 96, 128),
    (2, 16384, 16, 128),
    (4, 2048, 16, 128),
    (4, 4096, 64, 128),
    (8, 2048, 32, 256),
    (2, 2048, 16, 512),
    (4, 1024, 8, 512),
    (8, 1024, 8, 64),
)
SMALL_SHAPES = ((1, 64, 2, 32), (1, 128, 4, 32))
DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("forward", "backward", "all"), default="all")
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="bfloat16")
    parser.add_argument(
        "--shape",
        nargs=4,
        action="append",
        type=int,
        metavar=("B", "T", "H", "D"),
        help="repeat to benchmark custom B T H D cases",
    )
    parser.add_argument("--small", action="store_true", help="use smoke-test shapes")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument(
        "--include-torch",
        action="store_true",
        help="also time the very slow timestep-by-timestep PyTorch reference",
    )
    args = parser.parse_args()
    if args.warmup < 0 or args.rep <= 0:
        parser.error("--warmup must be non-negative and --rep must be positive")
    return args


def musa_triton_chunk_gla(
    q,
    k,
    v,
    g,
    scale=None,
    initial_state=None,
    output_final_state=False,
    state_v_first=False,
    cu_seqlens=None,
    cu_seqlens_cpu=None,
):
    """Directly call the original source-tree Triton/TLE implementation."""

    if scale is None:
        scale = q.shape[-1] ** -0.5
    return ChunkGLAFunction.apply(
        q,
        k,
        v,
        g,
        scale,
        initial_state,
        output_final_state,
        state_v_first,
        cu_seqlens,
        cu_seqlens_cpu,
    )


def synchronize():
    torch.musa.synchronize()


def bench_ms(function: Callable[[], object], warmup: int, rep: int) -> float:
    return triton.testing.do_bench(
        function,
        warmup=warmup,
        rep=rep,
        return_mode="median",
    )


def build_forward_inputs(B: int, T: int, H: int, D: int, dtype: torch.dtype):
    device = "musa"
    q = torch.randn(B, T, H, D, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    g = F.logsigmoid(torch.randn_like(q))
    kwargs = {
        "scale": D**-0.5,
        "initial_state": None,
        "output_final_state": False,
        "state_v_first": False,
        "cu_seqlens": None,
        "cu_seqlens_cpu": None,
    }
    return q, k, v, g, kwargs


def build_backward_inputs(B: int, T: int, H: int, D: int, dtype: torch.dtype):
    device = "musa"
    q = torch.randn(B, T, H, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    g_logit = torch.randn_like(q, requires_grad=True)
    kwargs = {
        "scale": D**-0.5,
        "initial_state": None,
        "output_final_state": False,
        "state_v_first": False,
        "cu_seqlens": None,
        "cu_seqlens_cpu": None,
    }
    return q, k, v, g_logit, kwargs


def bench_forward(function, inputs, warmup: int, rep: int) -> float:
    q, k, v, g, kwargs = inputs

    def run():
        with torch.no_grad():
            return function(q, k, v, g, **kwargs)

    return bench_ms(run, warmup, rep)


def bench_forward_backward(function, inputs, warmup: int, rep: int) -> float:
    q, k, v, g_logit, kwargs = inputs

    def run():
        q.grad = None
        k.grad = None
        v.grad = None
        g_logit.grad = None
        g = F.logsigmoid(g_logit)
        result = function(q, k, v, g, **kwargs)
        out = result[0] if isinstance(result, tuple) else result
        out.sum().backward()

    return bench_ms(run, warmup, rep)


def format_ms(value: float | None, width: int = 13) -> str:
    return f"{value:>{width}.3f}" if value is not None else f"{'n/a':>{width}}"


def format_ratio(numerator: float | None, denominator: float | None) -> str:
    if numerator is None or denominator is None or denominator <= 0:
        return f"{'n/a':>13}"
    return f"{numerator / denominator:>12.2f}x"


def print_header(title: str, args: argparse.Namespace):
    print(f"\n[MUSA chunk GLA: {title}]")
    print(f"device=musa dtype={args.dtype} " f"warmup={args.warmup} rep={args.rep}")
    print(
        f"{'B':>3} {'T':>6} {'H':>4} {'D':>4} "
        f"{'torch(ms)':>13} {'musa_c(ms)':>13} {'triton(ms)':>13} "
        f"{'musa_c/triton':>13}"
    )


def run_benchmark(args: argparse.Namespace):
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        raise RuntimeError("an available MUSA device is required")
    if not musa_c_available():
        raise RuntimeError(
            "the MUSA C extension is not built; run "
            "BaselineBenchmark/backends/_mthreads/ops/chunk_gla_musa_c/"
            "build_musa_chunk_gla.py build_ext --inplace --force"
        )

    dtype = DTYPES[args.dtype]
    shapes = tuple(tuple(shape) for shape in args.shape) if args.shape else None
    if shapes is None:
        shapes = SMALL_SHAPES if args.small else DEFAULT_SHAPES
    implementations = (
        ("MUSA C", musa_chunk_gla),
        ("Triton/TLE", musa_triton_chunk_gla),
    )

    if args.mode in ("forward", "all"):
        print_header("FWD", args)
        for B, T, H, D in shapes:
            inputs = build_forward_inputs(B, T, H, D, dtype)
            torch_ms = None
            if args.include_torch:
                torch_ms = bench_forward(
                    torch_recurrent_chunk_gla, inputs, args.warmup, args.rep
                )
            timings = {
                name: bench_forward(function, inputs, args.warmup, args.rep)
                for name, function in implementations
            }
            print(
                f"{B:>3} {T:>6} {H:>4} {D:>4} "
                f"{format_ms(torch_ms)} {format_ms(timings['MUSA C'])} "
                f"{format_ms(timings['Triton/TLE'])} "
                f"{format_ratio(timings['MUSA C'], timings['Triton/TLE'])}"
            )

    if args.mode in ("backward", "all"):
        print_header("FWD+BWD", args)
        for B, T, H, D in shapes:
            inputs = build_backward_inputs(B, T, H, D, dtype)
            torch_ms = None
            if args.include_torch:
                torch_ms = bench_forward_backward(
                    torch_recurrent_chunk_gla, inputs, args.warmup, args.rep
                )
            timings = {
                name: bench_forward_backward(function, inputs, args.warmup, args.rep)
                for name, function in implementations
            }
            print(
                f"{B:>3} {T:>6} {H:>4} {D:>4} "
                f"{format_ms(torch_ms)} {format_ms(timings['MUSA C'])} "
                f"{format_ms(timings['Triton/TLE'])} "
                f"{format_ratio(timings['MUSA C'], timings['Triton/TLE'])}"
            )
    synchronize()


if __name__ == "__main__":
    run_benchmark(parse_args())
