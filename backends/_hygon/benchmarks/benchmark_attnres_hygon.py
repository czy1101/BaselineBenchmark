from __future__ import annotations

import os

import torch

from backends._hygon.ops.attnes import fused_attnres as hip_attnres
from flag_attn.FLA.attnres.fused import fused_attnres as flag_attnres


D = 7168
CASES = (
    (2, 1),
    (5, 128),
    (5, 1024),
    (9, 128),
    (9, 8192),
)

DTYPE = (
    torch.float16
    if os.environ.get("ATTNRES_DTYPE", "bf16") == "fp16"
    else torch.bfloat16
)


def bench(fn, warmup=10, repeats=30):
    for _ in range(warmup):
        fn()

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()

    return start.elapsed_time(end) / repeats


def run_case(sources, rows, output_norm):
    query = torch.randn((D,), device="cuda", dtype=DTYPE)
    residuals = [
        torch.randn((rows, D), device="cuda", dtype=DTYPE)
        for _ in range(sources)
    ]
    rms_weight = torch.randn((D,), device="cuda", dtype=DTYPE)

    output_rms_weight = (
        torch.randn((D,), device="cuda", dtype=DTYPE)
        if output_norm
        else None
    )

    def run_hip():
        return hip_attnres(
            query,
            residuals,
            rms_weight,
            output_rms_weight,
        )

    def run_flag():
        return flag_attnres(
            query,
            residuals,
            rms_weight,
            output_rms_weight,
        )

    hip_output = run_hip()
    flag_output = run_flag()
    torch.cuda.synchronize()

    max_error = (
        hip_output.float() - flag_output.float()
    ).abs().max().item()

    hip_ms = bench(run_hip)
    flag_ms = bench(run_flag)

    ratio = hip_ms / flag_ms

    print(
        f"L{sources}_N{rows}_D{D} "
        f"output_norm={output_norm}: "
        f"HIP={hip_ms:.6f} ms, "
        f"FlagAttention={flag_ms:.6f} ms, "
        f"HIP/FlagAttention={ratio:.3f}x, "
        f"max_abs_error={max_error:.6f}"
    )


def main():
    torch.manual_seed(2026)

    print(f"dtype: {DTYPE}")
    print("=" * 100)

    for sources, rows in CASES:
        for output_norm in (False, True):
            run_case(sources, rows, output_norm)


if __name__ == "__main__":
    main()
