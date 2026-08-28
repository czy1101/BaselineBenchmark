"""Performance benchmark for the pure PyTorch/DTK GDN2 implementation."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import time
from pathlib import Path

import torch

from gdn2_hygon_reference import chunk_gdn2_hygon


DEFAULT_SHAPES = [
    (2, 512, 8, 64, 64),
    (4, 1024, 8, 64, 64),
    (1, 2048, 8, 64, 64),
    (1, 4096, 16, 64, 64),
    (1, 8192, 96, 128, 128),
    # The source table shows D=512, while the FlagGems test contract is
    # K=256 and V=512 for these two cases.
    (2, 2048, 16, 256, 512),
    (2, 16384, 16, 128, 128),
    (4, 1024, 8, 256, 512),
    (4, 2048, 16, 128, 128),
    (4, 4096, 64, 128, 128),
    (8, 1024, 8, 64, 64),
    (8, 2048, 32, 256, 256),
]

# Existing Hygon measurements supplied for acceptance comparison.
# Key: (dtype, B, T, H, K, V), value: (HY FLA ms, HY optimized ms).
TARGETS = {
    ("float16",2,512,8,64,64):(0.899,0.424), ("bfloat16",2,512,8,64,64):(0.911,0.434),
    ("float16",4,1024,8,64,64):(0.902,0.430), ("bfloat16",4,1024,8,64,64):(0.908,0.436),
    ("float16",1,2048,8,64,64):(0.903,0.443), ("bfloat16",1,2048,8,64,64):(0.915,0.455),
    ("float16",1,4096,16,64,64):(0.922,0.744), ("bfloat16",1,4096,16,64,64):(0.954,0.825),
    ("float16",1,8192,96,128,128):(15.131,9.655), ("bfloat16",1,8192,96,128,128):(14.947,10.191),
    ("float16",2,2048,16,256,512):(3.859,6.033), ("bfloat16",2,2048,16,256,512):(4.159,6.797),
    ("float16",2,16384,16,128,128):(10.240,6.894), ("bfloat16",2,16384,16,128,128):(10.367,7.491),
    ("float16",4,1024,8,256,512):(1.943,3.039), ("bfloat16",4,1024,8,256,512):(1.994,3.431),
    ("float16",4,2048,16,128,128):(2.600,1.754), ("bfloat16",4,2048,16,128,128):(2.650,1.910),
    ("float16",4,4096,64,128,128):(20.271,12.270), ("bfloat16",4,4096,64,128,128):(20.291,13.019),
    ("float16",8,1024,8,64,64):(0.902,0.431), ("bfloat16",8,1024,8,64,64):(0.932,0.447),
    ("float16",8,2048,32,256,256):(20.751,22.395), ("bfloat16",8,2048,32,256,256):(20.800,25.030),
}


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] if lo == hi else values[lo] + (values[hi] - values[lo]) * (pos - lo)


def make_inputs(shape, dtype):
    B, T, H, K, V = shape
    kw = dict(device="cuda:0", dtype=dtype)
    q = torch.randn(B, T, H, K, **kw) / math.sqrt(K)
    k = torch.randn(B, T, H, K, **kw) / math.sqrt(K)
    v = torch.randn(B, T, H, V, **kw)
    g = (-torch.rand(B, T, H, K, device="cuda:0") * 0.1).to(dtype)
    b = torch.rand(B, T, H, K, **kw)
    w = torch.rand(B, T, H, V, **kw)
    return q, k, v, g, b, w


def benchmark_one(shape, dtype, warmup, iterations, implementation):
    args = make_inputs(shape, dtype)
    if implementation == "hip":
        from gdn2_hip import chunk_gdn2_hip
        operator = chunk_gdn2_hip
    elif implementation == "chunk14":
        from gdn2_chunk14 import chunk_gdn2_chunk14
        operator = chunk_gdn2_chunk14
    elif implementation == "chunked":
        from gdn2_chunked_torch import chunk_gdn2_chunked
        operator = chunk_gdn2_chunked
    elif implementation in ("hip_auto", "hip_chunk_staged"):
        from gdn2_chunked_hip_staged import chunk_gdn2_hip_staged
        if implementation == "hip_chunk_staged":
            operator = lambda *xs, **kw: chunk_gdn2_hip_staged(
                *xs, force_chunk_staged=True, **kw)
        else:
            operator = chunk_gdn2_hip_staged
    elif implementation == "affine_scan":
        from gdn2_affine_scan import chunk_gdn2_affine_scan
        operator = chunk_gdn2_affine_scan
    else:
        operator = chunk_gdn2_hygon
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    with torch.inference_mode():
        for _ in range(warmup):
            operator(*args, output_final_state=True, chunk_size=64)
        torch.cuda.synchronize()

        times_ms = []
        for _ in range(iterations):
            start = time.perf_counter()
            o, ht = operator(*args, output_final_state=True, chunk_size=64)
            torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - start) * 1000.0)

    B, T, H, K, V = shape
    p50 = statistics.median(times_ms)
    dtype_name = str(dtype).replace("torch.", "")
    target_fla, target_opt = TARGETS.get((dtype_name,B,T,H,K,V), (float("nan"),float("nan")))
    return {
        "implementation": ("hip_recurrent" if implementation == "hip" else
                           "chunk14_segmented_dispatch" if implementation == "chunk14" else
                           "hip_auto_dispatch" if implementation == "hip_auto" else
                           "dtk_affine_scan" if implementation == "affine_scan" else
                           "hip_chunk_staged" if implementation == "hip_chunk_staged" else
                           "torch_chunk_wy" if implementation == "chunked" else
                           "pytorch_dtk_reference"),
        "dtype": dtype_name,
        "B": B, "T": T, "H": H, "K": K, "V": V,
        "warmup": warmup,
        "iterations": iterations,
        "mean_ms": statistics.mean(times_ms),
        "p50_ms": p50,
        "p95_ms": percentile(times_ms, 0.95),
        "min_ms": min(times_ms),
        "hy_fla_ms": target_fla,
        "hy_optimized_ms": target_opt,
        "hip_over_hy_fla": p50 / target_fla,
        "hip_over_hy_optimized": p50 / target_opt,
        "tokens_per_second": B * T / (p50 / 1000.0),
        "peak_memory_mib": torch.cuda.max_memory_allocated(0) / 1024**2,
        "output_nan": bool(torch.isnan(o.float()).any().item()),
        "state_nan": bool(torch.isnan(ht).any().item()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("fp16", "bf16", "both"), default="both")
    parser.add_argument("--implementation",
                        choices=("hip", "hip_auto", "chunk14", "hip_chunk_staged", "affine_scan",
                                 "chunked", "reference"),
                        default="hip_auto")
    parser.add_argument("--shape", nargs=5, type=int, action="append", metavar=("B", "T", "H", "K", "V"))
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--csv", default="",
                        help="optional CSV path; omitted means no file output")
    parser.add_argument("--start-index", type=int, default=0,
                        help="start at this zero-based default-shape index")
    parser.add_argument("--end-index", type=int, default=None,
                        help="stop before this zero-based default-shape index")
    args = parser.parse_args()

    if not getattr(torch.version, "hip", None):
        raise RuntimeError("Hygon HIP PyTorch is required")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one Docker/mask-isolated Hygon device must be visible")
    torch.cuda.set_device(0)
    shapes = ([tuple(x) for x in args.shape] if args.shape else
              DEFAULT_SHAPES[args.start_index:args.end_index])
    dtypes = {
        "fp16": [torch.float16], "bf16": [torch.bfloat16],
        "both": [torch.float16, torch.bfloat16],
    }[args.dtype]

    rows = []
    for dtype in dtypes:
        for shape in shapes:
            row = benchmark_one(shape, dtype, args.warmup, args.iterations, args.implementation)
            rows.append(row)
            print(row, flush=True)

    if args.csv:
        path = Path(args.csv)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print("CSV:", path.resolve())


if __name__ == "__main__":
    main()
