"""NSA Hygon benchmark using the official workload shapes."""

import argparse
import statistics
import time

import torch

from nsa_hip import parallel_nsa


SHAPES = [
    (1, 16384, 4, 64, 64),
    (1, 8192, 16, 256, 64),
    (1, 16384, 16, 256, 64),
    (1, 65536, 16, 256, 64),
    (1, 16384, 32, 512, 64),
    (1, 16384, 16, 256, 128),
    (4, 8192, 16, 256, 64),
]

REFERENCE = {
    # (B,T,H,HQ,D,dtype): (HY native, HY TLE)
    (1, 16384, 4, 64, 64, "bfloat16"): (3.833, 3.831),
    (1, 8192, 16, 256, 64, "bfloat16"): (7.722, 7.700),
    (1, 16384, 16, 256, 64, "bfloat16"): (15.371, 15.328),
    (1, 65536, 16, 256, 64, "bfloat16"): (73.789, 73.514),
    (1, 16384, 32, 512, 64, "bfloat16"): (30.879, 30.813),
    (1, 16384, 16, 256, 128, "bfloat16"): (37.875, 37.536),
    (4, 8192, 16, 256, 64, "bfloat16"): (30.709, 30.654),
    (1, 16384, 4, 64, 64, "float16"): (3.506, 3.524),
    (1, 8192, 16, 256, 64, "float16"): (7.045, 7.085),
    (1, 16384, 16, 256, 64, "float16"): (14.051, 14.114),
    (1, 65536, 16, 256, 64, "float16"): (70.980, 71.141),
    (1, 16384, 32, 512, 64, "float16"): (28.277, 28.467),
    (1, 16384, 16, 256, 128, "float16"): (35.659, 35.381),
    (4, 8192, 16, 256, 64, "float16"): (28.102, 28.226),
}


def make_inputs(shape, dtype, block_size, topk):
    b, t, h, hq, d = shape
    nblocks = (t + block_size - 1) // block_size
    s = min(topk, nblocks)
    q = torch.randn(b, t, hq, d, device="cuda", dtype=dtype)
    k = torch.randn(b, t, h, d, device="cuda", dtype=dtype)
    v = torch.randn(b, t, h, d, device="cuda", dtype=dtype)
    indices = torch.randint(0, nblocks, (b, t, h, s), device="cuda", dtype=torch.int32)
    return q, k, v, indices, s


def bench(shape, dtype, warmup, iterations, block_size, topk):
    q, k, v, indices, s = make_inputs(shape, dtype, block_size, topk)
    scale = shape[-1] ** -0.5

    def run():
        return parallel_nsa(q, k, v, block_indices=indices,
                            block_counts=s, block_size=block_size, scale=scale)

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        run()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples), torch.cuda.max_memory_allocated() / 2**20


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dtype", choices=("fp16", "bf16", "both"), default="both")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=len(SHAPES))
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iterations", type=int, default=30)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--topk", type=int, default=16)
    p.add_argument("--shape", nargs=5, type=int, action="append",
                   metavar=("B", "T", "H", "HQ", "D"))
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("one Hygon device is required")
    shapes = a.shape or SHAPES[a.start:a.end]
    dtypes = ([torch.float16] if a.dtype == "fp16" else
              [torch.bfloat16] if a.dtype == "bf16" else
              [torch.float16, torch.bfloat16])
    print("B T H HQ D dtype current_ms native_ms tle_ms current/native current/tle peak_mib", flush=True)
    for dtype in dtypes:
        name = str(dtype).split(".")[-1]
        for shape in shapes:
            torch.cuda.reset_peak_memory_stats()
            ms, peak = bench(tuple(shape), dtype, a.warmup, a.iterations,
                             a.block_size, a.topk)
            native, tle = REFERENCE.get(tuple(shape) + (name,), (float("nan"), float("nan")))
            rn = ms / native if native == native else float("nan")
            rt = ms / tle if tle == tle else float("nan")
            b, t, h, hq, d = shape
            print(
                f"{b} {t} {h} {hq} {d} {name} {ms:.3f} "
                f"{native:.3f} {tle:.3f} {rn:.3f} {rt:.3f} {peak:.1f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
