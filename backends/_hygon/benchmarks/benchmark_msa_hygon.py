"""End-to-end MiniMax M3 MSA benchmark on Hygon DTK/HIP."""

from __future__ import annotations

import argparse
import math
import statistics
from dataclasses import dataclass

import torch

from msa_hip import (
    minimax_m3_index_decode,
    minimax_m3_index_score,
    minimax_m3_index_topk,
    minimax_m3_sparse_attn,
    minimax_m3_sparse_attn_decode,
    sparse_backend_name,
)


BLOCK = 128
DIM = 128
TOPK = 16
PREFILL_SHAPES = [
    (1, 8192, 16, 96),
    (2, 16384, 8, 96),
    (1, 32768, 16, 96),
    (2, 8192, 8, 96),
    (4, 4096, 16, 384),
    (4, 4096, 16, 256),
]
DECODE_SHAPES = [
    (1, 4096, 16, 96),
    (1, 16384, 16, 96),
    (1, 65536, 16, 96),
    (4, 4096, 8, 96),
    (4, 16384, 8, 96),
    (16, 4096, 8, 96),
    (32, 2048, 4, 48),
    (64, 1024, 4, 48),
]


@dataclass
class Data:
    q: torch.Tensor
    idx_q: torch.Tensor
    kv: torch.Tensor
    index_k: torch.Tensor
    table: torch.Tensor
    cu: torch.Tensor
    lens: torch.Tensor
    prefix: torch.Tensor


def make_data(shape, decode: bool) -> Data:
    batch, seq_len, kv_heads, heads = shape
    pages = math.ceil(seq_len / BLOCK)
    total_pages = batch * pages
    total_q = batch if decode else batch * seq_len
    generator = torch.Generator(device="cuda")
    generator.manual_seed(17 + int(decode))

    def randn(size):
        return torch.randn(
            size, device="cuda", dtype=torch.bfloat16, generator=generator
        ).contiguous()

    q = randn((total_q, heads, DIM))
    idx_q = randn((total_q, kv_heads, DIM))
    kv = randn((total_pages, kv_heads, BLOCK, 2 * DIM))
    index_k = randn((total_pages, BLOCK, DIM))
    table = torch.randperm(total_pages, device="cuda", generator=generator)
    table = table.view(batch, pages).to(torch.int32).contiguous()
    q_stride = 1 if decode else seq_len
    cu = torch.arange(
        0, (batch + 1) * q_stride, q_stride, device="cuda", dtype=torch.int32
    )
    lens = torch.full((batch,), seq_len, device="cuda", dtype=torch.int32)
    prefix = torch.zeros_like(lens)
    return Data(q, idx_q, kv, index_k, table, cu, lens, prefix)


def median_ms(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def run_shape(mode: str, shape, warmup: int, iterations: int):
    batch, seq_len, kv_heads, _ = shape
    decode = mode == "decode"
    data = make_data(shape, decode)
    output = torch.empty_like(data.q)

    if decode:
        def index_fn():
            return minimax_m3_index_decode(
                data.idx_q, data.index_k, data.table, data.lens, seq_len,
                TOPK, 1, 2, kv_heads, 1, 1
            )

        selected = index_fn()

        def sparse_fn():
            minimax_m3_sparse_attn_decode(
                data.q, data.kv, selected, data.table, data.lens, kv_heads,
                DIM**-0.5, output, 1
            )

        def pipeline_fn():
            current = index_fn()
            minimax_m3_sparse_attn_decode(
                data.q, data.kv, current, data.table, data.lens, kv_heads,
                DIM**-0.5, output, 1
            )
    else:
        def score_fn():
            return minimax_m3_index_score(
                data.idx_q, data.index_k, data.table, data.cu, data.lens,
                data.prefix, seq_len, seq_len, kv_heads
            )

        score = score_fn()

        selected = minimax_m3_index_topk(
            score, data.cu, data.prefix, seq_len, TOPK, 1, 2
        )

        def index_fn():
            current_score = score_fn()
            return minimax_m3_index_topk(
                current_score, data.cu, data.prefix, seq_len, TOPK, 1, 2
            )

        def sparse_fn():
            minimax_m3_sparse_attn(
                data.q, data.kv, selected, data.table, data.cu, data.lens,
                data.prefix, seq_len, kv_heads, DIM**-0.5, output
            )

        def pipeline_fn():
            current = index_fn()
            minimax_m3_sparse_attn(
                data.q, data.kv, current, data.table, data.cu, data.lens,
                data.prefix, seq_len, kv_heads, DIM**-0.5, output
            )

    index_ms = median_ms(index_fn, warmup, iterations)
    sparse_ms = median_ms(sparse_fn, warmup, iterations)
    pipeline_ms = median_ms(pipeline_fn, warmup, iterations)
    tokens = batch if decode else batch * seq_len
    return index_ms, sparse_ms, pipeline_ms, tokens * 1000.0 / pipeline_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prefill", "decode", "both"), default="both")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("A Hygon DTK/HIP device is required")
    print(f"# sparse_backend={sparse_backend_name()}")
    modes = ("prefill", "decode") if args.mode == "both" else (args.mode,)
    print("mode,index,B,T,KVH,QH,index_ms,sparse_ms,pipeline_ms,tokens_per_s")
    for mode in modes:
        shapes = PREFILL_SHAPES if mode == "prefill" else DECODE_SHAPES
        stop = len(shapes) if args.end is None else min(args.end, len(shapes))
        for index in range(args.start, stop):
            shape = shapes[index]
            try:
                values = run_shape(mode, shape, args.warmup, args.iterations)
                print(
                    f"{mode},{index},{','.join(map(str, shape))},"
                    f"{values[0]:.4f},{values[1]:.4f},{values[2]:.4f},{values[3]:.2f}"
                )
            finally:
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
