import argparse
import statistics
import time

import torch

from kda_hip import chunk_kda_dtk


p = argparse.ArgumentParser()
p.add_argument("--dtype", choices=["fp16", "bf16", "both"], default="both")
p.add_argument("--warmup", type=int, default=10)
p.add_argument("--iterations", type=int, default=30)
# 64 is the measured best path for the production shapes; 16 remains
# available as an explicit regression/control setting.
p.add_argument("--chunk-size", type=int, choices=[16, 32, 64], default=64)
a = p.parse_args()
dtypes = ([torch.float16] if a.dtype == "fp16" else
          [torch.bfloat16] if a.dtype == "bf16" else
          [torch.float16, torch.bfloat16])
shapes = [(1, 8192, 96, 96, 128, 128), (8, 1024, 96, 96, 128, 128)]

for dtype in dtypes:
    for B, T, H, HV, K, V in shapes:
        q = torch.randn(B, T, H, K, device="cuda", dtype=dtype)
        k = torch.randn_like(q)
        v = torch.randn(B, T, HV, V, device="cuda", dtype=dtype)
        g = (-torch.rand(B, T, HV, K, device="cuda") * 0.1).to(dtype)
        beta = torch.randn(B, T, HV, device="cuda", dtype=dtype)
        def run():
            return chunk_kda_dtk(
                q, k, v, g, beta,
                chunk_size=a.chunk_size,
                gemm_dtype=dtype,
            )
        torch.cuda.reset_peak_memory_stats()
        for _ in range(a.warmup):
            run()
        torch.cuda.synchronize()
        samples = []
        for _ in range(a.iterations):
            t0 = time.perf_counter()
            run()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - t0) * 1e3)
        p50 = statistics.median(samples)
        print({
            "implementation": "dtk_bmm_chunk", "chunk_size": a.chunk_size,
            "B": B, "T": T,
            "H": H, "HV": HV, "K": K, "V": V,
            "dtype": str(dtype).split(".")[-1], "p50_ms": p50,
            "tokens_per_sec": B * T / (p50 / 1e3),
            "peak_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
        }, flush=True)
