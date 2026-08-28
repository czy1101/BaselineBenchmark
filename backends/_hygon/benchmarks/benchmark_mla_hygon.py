"""Dense MLA benchmark for the official H800/Hygon comparison shapes."""
import argparse
import math
import statistics
import time

import torch

from mla_hip import flash_mla_hygon


# (B, Sq, Hq, Hkv, Dqk, Dv, Skv)
SHAPES = [
    (128, 1, 128, 1, 576, 512, 1024),
    (128, 1, 128, 1, 576, 512, 2048),
    (128, 1, 128, 1, 576, 512, 4096),
    (128, 1, 128, 1, 576, 512, 8192),
    (128, 1, 128, 1, 576, 512, 16384),
    # Prefill shapes are single-request measurements; B=128 here would
    # require over 72 GiB just for Q at D=576.
    (1, 4096, 128, 1, 576, 512, 8192),
    (1, 4096, 64, 1, 512, 512, 8192),
    (1, 4096, 128, 1, 512, 512, 8192),
]


def make_inputs(shape, dtype):
    B, SQ, HQ, HKV, D, DV, SKV = shape
    bs = 64
    mp = ((SKV + bs - 1) // bs) * bs
    nb = B * (mp // bs)
    dev = "cuda:0"
    q = torch.randn(B, SQ, HQ, D, device=dev, dtype=dtype)
    table = torch.arange(nb, device=dev, dtype=torch.int32).view(B, mp // bs)
    cache = torch.randn(nb, bs, HKV, D, device=dev, dtype=dtype)
    lengths = torch.full((B,), SKV, device=dev, dtype=torch.int32)
    return q, table, cache, mp, bs, lengths, HQ, HKV, D, DV, True


def bench(shape, dtype, warmup, iterations):
    args = make_inputs(shape, dtype)
    torch.cuda.empty_cache()
    for _ in range(warmup):
        flash_mla_hygon(*args)
    torch.cuda.synchronize()
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        out, lse = flash_mla_hygon(*args)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    p50 = statistics.median(times)
    B, SQ, _, _, _, _, _ = shape
    return {
        "B": B, "Sq": SQ, "Hq": shape[2], "Hkv": shape[3],
        "D": shape[4], "DV": shape[5], "Skv": shape[6],
        "dtype": str(dtype).replace("torch.", ""),
        "mean_ms": statistics.mean(times), "p50_ms": p50,
        "min_ms": min(times), "tok_per_s": B * SQ / (p50 / 1000),
        "peak_mib": torch.cuda.max_memory_allocated() / 2**20,
        "nan": bool(torch.isnan(out.float()).any() or torch.isnan(lse).any()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=("fp16", "bf16", "both"), default="both")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=len(SHAPES))
    ap.add_argument(
        "--shape", nargs=7, type=int, action="append",
        metavar=("B", "SQ", "HQ", "HKV", "D", "DV", "SKV"),
        help="custom shape; may be repeated",
    )
    args = ap.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one masked Hygon device is required")
    dtypes = {"fp16": [torch.float16], "bf16": [torch.bfloat16],
              "both": [torch.float16, torch.bfloat16]}[args.dtype]
    shapes = [tuple(x) for x in args.shape] if args.shape else SHAPES[args.start:args.end]
    for dtype in dtypes:
        for shape in shapes:
            print(bench(shape, dtype, args.warmup, args.iterations), flush=True)


if __name__ == "__main__":
    main()
