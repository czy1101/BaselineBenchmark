"""Controlled same-card benchmark for S60 NSA forward."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch
import torch_gcu  # noqa: F401
import pytest

from backends._enflame.benchmarks._common import (
    measure,
    summarize,
    write_json,
)
from backends._enflame.ops.nsa import (
    parallel_nsa_torch_baseline,
)


CASES = {
    "CALIB_H16_S2K": (1, 2048, 16, 256, 64),
    "H4_S16K": (1, 16384, 4, 64, 64),
    "H16_S8K": (1, 8192, 16, 256, 64),
    "H16_S16K": (1, 16384, 16, 256, 64),
    "H16_S64K": (1, 65536, 16, 256, 64),
    "H32_S16K": (1, 16384, 32, 512, 64),
    "H16_D128": (1, 16384, 16, 256, 128),
    "B4_H16_S8K": (4, 8192, 16, 256, 64),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument(
        "--implementation",
        required=True,
        choices=("torch", "adapted"),
    )
    parser.add_argument(
        "--case",
        required=True,
        choices=tuple(CASES),
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=9,
    )
    parser.add_argument(
        "--inner",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max-batch-query",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--json-output",
    )
    return parser.parse_args()


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()
    torch.gcu.set_device(args.device)
    torch.manual_seed(42)

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]

    batch, tokens, kv_heads, query_heads, head_dim = (
        CASES[args.case]
    )
    selected_blocks = 16
    block_size = 64
    scale = head_dim**-0.5
    block_count = (
        tokens + block_size - 1
    ) // block_size
    device = f"gcu:{args.device}"

    q = torch.randn(
        (
            batch,
            tokens,
            query_heads,
            head_dim,
        ),
        dtype=dtype,
        device=device,
    )
    k = torch.randn(
        (
            batch,
            tokens,
            kv_heads,
            head_dim,
        ),
        dtype=dtype,
        device=device,
    )
    v = torch.randn(
        (
            batch,
            tokens,
            kv_heads,
            head_dim,
        ),
        dtype=dtype,
        device=device,
    )

    first_block = torch.zeros(
        (
            batch,
            tokens,
            kv_heads,
            1,
        ),
        dtype=torch.int32,
        device=device,
    )
    random_blocks = torch.randint(
        0,
        block_count,
        (
            batch,
            tokens,
            kv_heads,
            selected_blocks - 1,
        ),
        dtype=torch.int32,
        device=device,
    )
    block_indices = torch.cat(
        (first_block, random_blocks),
        dim=-1,
    ).contiguous()

    if args.implementation == "torch":

        def run():
            return parallel_nsa_torch_baseline(
                q=q,
                k=k,
                v=v,
                block_indices=block_indices,
                block_counts=selected_blocks,
                block_size=block_size,
                scale=scale,
                max_batch_query=args.max_batch_query,
            )

    else:
        public = importlib.import_module(
            "flag_attn.runtime"
        )

        def run():
            return public.parallel_nsa(
                q=q,
                k=k,
                v=v,
                block_indices=block_indices,
                block_counts=selected_blocks,
                block_size=block_size,
                scale=scale,
            )

    print("===== S60 NSA FORWARD BENCHMARK =====")
    print("implementation:", args.implementation)
    print("device:", args.device)
    print("current_device:", torch.gcu.current_device())
    print("case:", args.case)
    print(
        "shape:",
        (
            batch,
            tokens,
            kv_heads,
            query_heads,
            head_dim,
        ),
    )
    print("dtype:", dtype)
    print("block_size:", block_size)
    print("selected_blocks:", selected_blocks)
    print("warmup:", args.warmup)
    print("repeat:", args.repeat)
    print("inner:", args.inner)

    samples, output = measure(
        run,
        warmup=args.warmup,
        repeat=args.repeat,
        inner=args.inner,
    )

    finite = bool(
        torch.isfinite(output.float()).all().item()
    )
    if not finite:
        raise AssertionError(
            "NSA output contains non-finite values"
        )

    record = summarize(
        operator="nsa",
        stage="forward",
        implementation=args.implementation,
        case=args.case,
        dtype=args.dtype,
        samples=samples,
        finite=finite,
    )
    record["shape"] = [
        batch,
        tokens,
        kv_heads,
        query_heads,
        head_dim,
    ]
    record["block_size"] = block_size
    record["selected_blocks"] = selected_blocks
    record["max_batch_query"] = (
        args.max_batch_query
    )
    write_json(args.json_output, record)

    print("S60 NSA FORWARD BENCHMARK PASS")


@pytest.mark.parallel_nsa
def test_nsa_forward_benchmark(benchmark_options) -> None:
    case = benchmark_options.get("case", required=True)
    if case not in CASES:
        raise pytest.UsageError(
            f"--case must be one of {', '.join(CASES)} for NSA forward"
        )
    args = argparse.Namespace(
        implementation=benchmark_options.get("implementation", required=True),
        case=case,
        dtype=benchmark_options.get("dtype", "bfloat16"),
        device=benchmark_options.get("device", 0),
        warmup=benchmark_options.get("warmup", 3),
        repeat=benchmark_options.get("repeat", 9),
        inner=benchmark_options.get("inner", 1),
        max_batch_query=benchmark_options.get("max_batch_query", 512),
        json_output=benchmark_options.get("json_output"),
    )
    main(args)


if __name__ == "__main__":
    main()
