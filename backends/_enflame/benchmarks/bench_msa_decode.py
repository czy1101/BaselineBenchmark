"""Controlled same-card benchmark for S60 MSA decode."""

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
from backends._enflame.benchmarks._msa_inputs import (
    HEAD_DIM,
    make_decode_inputs,
)
from backends._enflame.ops.msa import (
    torch_gcu_msa_decode,
)


CASES = {
    "D1": ((512,), 1),
    "D2": ((8192,), 1),
    "D3": ((2048,) * 8, 1),
    "D4": ((2048, 1024, 513), 4),
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
        "--device",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=7,
    )
    parser.add_argument(
        "--inner",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--json-output",
    )
    return parser.parse_args()


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()
def main() -> None:
    args = parse_args()
    torch.gcu.set_device(args.device)
    torch.manual_seed(42)

    sequence_lengths, decode_query_len = (
        CASES[args.case]
    )
    batch = len(sequence_lengths)
    num_kv_heads = 2
    group_size = 8
    num_query_heads = (
        num_kv_heads * group_size
    )
    device = f"gcu:{args.device}"
    dtype = torch.bfloat16
    init_blocks = 1
    local_blocks = 2

    data = make_decode_inputs(
        sequence_lengths=sequence_lengths,
        decode_query_len=decode_query_len,
        num_kv_heads=num_kv_heads,
        group_size=group_size,
        device=device,
        dtype=dtype,
    )

    if args.implementation == "torch":

        def run():
            _, output = torch_gcu_msa_decode(
                idx_q=data.idx_q,
                index_kv_cache=(
                    data.index_kv_cache
                ),
                q=data.q,
                kv_cache=data.kv_cache,
                block_table=data.block_table,
                seq_lens=data.seq_lens,
                decode_query_len=(
                    decode_query_len
                ),
                topk=args.topk,
                init_blocks=init_blocks,
                local_blocks=local_blocks,
                num_kv_heads=num_kv_heads,
                sm_scale=data.sm_scale,
            )
            return output

    else:
        public = importlib.import_module(
            "flag_attn.runtime"
        backend = importlib.import_module(
            "flag_attn.runtime.backend._enflame"
        )
        backend.install_msa_prefill(
            use_tle=False
        )
        public = importlib.import_module(
            "flag_attn.minimax_sparse_attention"
        )
        output = torch.empty_like(data.q)

        def run():
            topk_idx = (
                public.minimax_m3_index_decode(
                    data.idx_q,
                    data.index_kv_cache,
                    data.block_table,
                    data.seq_lens,
                    data.max_seq_len,
                    args.topk,
                    init_blocks,
                    local_blocks,
                    num_kv_heads,
                    decode_query_len,
                    decode_query_len,
                )
            )

            public.minimax_m3_sparse_attn_decode(
                data.q,
                data.kv_cache,
                topk_idx,
                data.block_table,
                data.seq_lens,
                num_kv_heads,
                data.sm_scale,
                output,
                decode_query_len,
            )
            return output

    print("===== S60 MSA DECODE BENCHMARK =====")
    print("implementation:", args.implementation)
    print("device:", args.device)
    print("current_device:", torch.gcu.current_device())
    print("case:", args.case)
    print(
        "shape:",
        (
            batch,
            decode_query_len,
            num_kv_heads,
            num_query_heads,
            HEAD_DIM,
        ),
    )
    print(
        "sequence_lengths:",
        sequence_lengths,
    )
    print("dtype:", dtype)
    print("topk:", args.topk)
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
            "MSA decode output is not finite"
        )

    record = summarize(
        operator="msa",
        stage="decode",
        implementation=args.implementation,
        case=args.case,
        dtype="bfloat16",
        samples=samples,
        finite=finite,
    )
    record["shape"] = [
        batch,
        decode_query_len,
        num_kv_heads,
        num_query_heads,
        HEAD_DIM,
    ]
    record["sequence_lengths"] = list(
        sequence_lengths
    )
    record["topk"] = args.topk
    write_json(args.json_output, record)

    print("S60 MSA DECODE BENCHMARK PASS")


@pytest.mark.minimax_sparse_attention
def test_msa_decode_benchmark(benchmark_options) -> None:
    case = benchmark_options.get("case", required=True)
    if case not in CASES:
        raise pytest.UsageError(
            f"--case must be one of {', '.join(CASES)} for MSA decode"
        )
    args = argparse.Namespace(
        implementation=benchmark_options.get("implementation", required=True),
        case=case,
        device=benchmark_options.get("device", 0),
        warmup=benchmark_options.get("warmup", 20),
        repeat=benchmark_options.get("repeat", 7),
        inner=benchmark_options.get("inner", 50),
        topk=benchmark_options.get("topk", 4),
        json_output=benchmark_options.get("json_output"),
    )
    main(args)


if __name__ == "__main__":
    main()
