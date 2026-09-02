import argparse
import statistics
import time

import torch

from sageattention_hip import backend_name, forward, per_block_int8


DEFAULT_SHAPES = [
    (1, 32, 32, 1024, 1024, 64),
    (1, 32, 32, 4096, 4096, 64),
    (1, 32, 32, 8192, 8192, 64),
    (1, 32, 32, 16384, 16384, 64),
    (1, 32, 32, 1024, 1024, 128),
    (1, 32, 32, 4096, 4096, 128),
    (1, 32, 32, 8192, 8192, 128),
    (1, 32, 32, 16384, 16384, 128),
]

# Supplied BW1000 optimized reference numbers. They are displayed for
# comparison only and are not used by the timing loop.
HYGON_OPT_REFERENCE_MS = {
    (1, 1024, 32, 64): 0.1798,
    (1, 4096, 32, 64): 1.8393,
    (1, 8192, 32, 64): 7.0687,
    (1, 16384, 32, 64): 27.8410,
    (1, 1024, 32, 128): 0.3224,
    (1, 4096, 32, 128): 4.2880,
    (1, 8192, 32, 128): 16.4336,
    (1, 16384, 32, 128): 61.5374,
}


def measure(fn, warmup, iterations):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples), statistics.mean(samples)


def bench(shape, output_dtype, warmup, iterations):
    b, qh, kvh, q_len, kv_len, dim = shape
    q = torch.randn((b, qh, q_len, dim), device="cuda", dtype=torch.float16)
    k = torch.randn((b, kvh, kv_len, dim), device="cuda", dtype=torch.float16)
    v = torch.randn((b, kvh, kv_len, dim), device="cuda", dtype=torch.float16)
    qi, qs, ki, ks = per_block_int8(q, k)

    def quant_run():
        return per_block_int8(q, k)

    def attention_run():
        return forward(qi, ki, v, qs, ks, output_dtype=output_dtype)

    def pipeline_run():
        qx, qxs, kx, kxs = per_block_int8(q, k)
        return forward(qx, kx, v, qxs, kxs, output_dtype=output_dtype)

    torch.cuda.reset_peak_memory_stats()
    quant_p50, _ = measure(quant_run, warmup, iterations)
    attn_p50, attn_mean = measure(attention_run, warmup, iterations)
    pipe_p50, _ = measure(pipeline_run, warmup, iterations)
    peak = torch.cuda.max_memory_allocated() / 1024**2
    flops = 4.0 * b * qh * q_len * kv_len * dim
    return quant_p50, attn_p50, attn_mean, pipe_p50, flops / attn_p50 * 1e-9, peak


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dtype", choices=("fp16", "bf16", "both"), default="both")
    p.add_argument("--shape", nargs=4, type=int, action="append",
                   metavar=("B", "T", "H", "D"))
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iterations", type=int, default=30)
    args = p.parse_args()
    shapes = ([(b, h, h, t, t, d) for b, t, h, d in args.shape]
              if args.shape else DEFAULT_SHAPES)
    shapes = shapes[args.start:args.end]
    dtypes = ([torch.float16, torch.bfloat16] if args.dtype == "both" else
              [torch.float16 if args.dtype == "fp16" else torch.bfloat16])
    print("# backend=", backend_name())
    print("B T H D dtype quant_ms attention_p50_ms attention_mean_ms pipeline_ms tflops peak_mib hygon_opt_ref_ms speedup_vs_hygon_opt")
    for dtype in dtypes:
        for shape in shapes:
            values = bench(shape, dtype, args.warmup, args.iterations)
            b, qh, kvh, q_len, kv_len, dim = shape
            ref = HYGON_OPT_REFERENCE_MS.get((b, q_len, qh, dim))
            speedup = ref / values[1] if ref is not None else None
            print(b, q_len, qh, dim, str(dtype).replace("torch.", ""),
                  *(f"{x:.4f}" for x in values),
                  f"{ref:.4f}" if ref is not None else "nan",
                  f"{speedup:.3f}x" if speedup is not None else "nan")


if __name__ == "__main__":
    main()
