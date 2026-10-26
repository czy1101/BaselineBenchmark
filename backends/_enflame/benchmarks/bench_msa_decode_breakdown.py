# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0.

"""Stage-level S60 benchmark for the adapted MSA Decode pipeline."""

from __future__ import annotations

import argparse
import importlib
import statistics
import sys
from pathlib import Path
from typing import Any, Callable

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch
import torch_gcu  # noqa: F401
import pytest

from backends._enflame.benchmarks._common import (
    measure,
    write_json,
)
from backends._enflame.benchmarks._msa_inputs import (
    HEAD_DIM,
    make_decode_inputs,
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
    parser.add_argument("--json-output")
    return parser.parse_args()


def _measure_stage(
    name: str,
    function: Callable[[], Any],
    *,
    warmup: int,
    repeat: int,
    inner: int,
) -> tuple[dict[str, Any], Any]:
    print(f"===== stage={name} =====", flush=True)
    samples, result = measure(
        function,
        warmup=warmup,
        repeat=repeat,
        inner=inner,
    )
    record = {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }
    print(
        "STAGE_RESULT "
        f"stage={name} "
        f"median_ms={record['median_ms']:.6f} "
        f"min_ms={record['min_ms']:.6f} "
        f"max_ms={record['max_ms']:.6f}",
        flush=True,
    )
    return record, result


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()
    torch.gcu.set_device(args.device)
    torch.manual_seed(42)

    sequence_lengths, decode_query_len = CASES[args.case]
    batch = len(sequence_lengths)
    num_kv_heads = 2
    group_size = 8
    num_query_heads = num_kv_heads * group_size
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

    index_module = importlib.import_module(
        "flag_attn.runtime.backend._enflame.msa.index_topk"
    )
    attention_module = importlib.import_module(
        "flag_attn.runtime.backend._enflame.msa.sparse_attn"
    )

    total_q = data.q.shape[0]
    output = torch.empty_like(data.q)
    topk_workspace = torch.empty(
        (num_kv_heads, total_q, args.topk),
        dtype=torch.int32,
        device=data.q.device,
    )

    index_args = (
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

    score_args = (
        data.idx_q,
        data.index_kv_cache,
        data.block_table,
        data.seq_lens,
        data.max_seq_len,
        init_blocks,
        local_blocks,
        num_kv_heads,
        decode_query_len,
        decode_query_len,
    )

    score_workspace = None
    if data.max_seq_len > args.topk * 128:
        score_workspace = (
            index_module.minimax_m3_index_decode_score(
                *score_args
            )
        )
        torch.gcu.synchronize()

    def run_score_reuse():
        return index_module.minimax_m3_index_decode_score(
            *score_args,
            score_out=score_workspace,
        )

    def run_index_reuse():
        return index_module.minimax_m3_index_decode(
            *index_args,
            out=topk_workspace,
            score_out=score_workspace,
        )

    def run_index_e2e():
        return index_module.minimax_m3_index_decode(
            *index_args
        )

    topk_idx = run_index_reuse()
    torch.gcu.synchronize()

    def run_attention():
        attention_module.minimax_m3_sparse_attn_decode(
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

    def run_full():
        selected = run_index_e2e()
        attention_module.minimax_m3_sparse_attn_decode(
            data.q,
            data.kv_cache,
            selected,
            data.block_table,
            data.seq_lens,
            num_kv_heads,
            data.sm_scale,
            output,
            decode_query_len,
        )
        return output

    print("===== S60 MSA DECODE BREAKDOWN =====")
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
    print("sequence_lengths:", sequence_lengths)
    print("dtype:", dtype)
    print("topk:", args.topk)
    print("warmup:", args.warmup)
    print("repeat:", args.repeat)
    print("inner:", args.inner)

    stages: dict[str, dict[str, Any]] = {}

    if score_workspace is not None:
        stages["index_score_reuse"], _ = _measure_stage(
            "index_score_reuse",
            run_score_reuse,
            warmup=args.warmup,
            repeat=args.repeat,
            inner=args.inner,
        )

    stages["index_reuse"], topk_idx = _measure_stage(
        "index_reuse",
        run_index_reuse,
        warmup=args.warmup,
        repeat=args.repeat,
        inner=args.inner,
    )
    stages["index_e2e"], _ = _measure_stage(
        "index_e2e",
        run_index_e2e,
        warmup=args.warmup,
        repeat=args.repeat,
        inner=args.inner,
    )
    stages["attention"], attention_output = _measure_stage(
        "attention",
        run_attention,
        warmup=args.warmup,
        repeat=args.repeat,
        inner=args.inner,
    )
    stages["full"], full_output = _measure_stage(
        "full",
        run_full,
        warmup=args.warmup,
        repeat=args.repeat,
        inner=args.inner,
    )

    for name, value in (
        ("attention", attention_output),
        ("full", full_output),
    ):
        finite = bool(
            torch.isfinite(value.float()).all().item()
        )
        if not finite:
            raise AssertionError(
                f"{name} output is not finite"
            )

    index_reuse_ms = stages["index_reuse"]["median_ms"]
    index_e2e_ms = stages["index_e2e"]["median_ms"]
    attention_ms = stages["attention"]["median_ms"]
    full_ms = stages["full"]["median_ms"]

    score_ms = None
    topk_or_identity_ms = index_reuse_ms
    if "index_score_reuse" in stages:
        score_ms = stages[
            "index_score_reuse"
        ]["median_ms"]
        topk_or_identity_ms = index_reuse_ms - score_ms

    derived = {
        "score_kernel_ms": score_ms,
        "topk_or_identity_estimate_ms": topk_or_identity_ms,
        "index_allocation_estimate_ms": (
            index_e2e_ms - index_reuse_ms
        ),
        "index_share_of_full_pct": (
            100.0 * index_e2e_ms / full_ms
        ),
        "attention_share_of_full_pct": (
            100.0 * attention_ms / full_ms
        ),
        "full_residual_ms": (
            full_ms - index_e2e_ms - attention_ms
        ),
    }

    score_text = (
        "n/a"
        if score_ms is None
        else f"{score_ms:.6f}"
    )
    print(
        "BREAKDOWN "
        f"case={args.case} "
        f"score_kernel_ms={score_text} "
        f"topk_or_identity_estimate_ms="
        f"{topk_or_identity_ms:.6f} "
        f"index_allocation_estimate_ms="
        f"{derived['index_allocation_estimate_ms']:.6f} "
        f"index_e2e_ms={index_e2e_ms:.6f} "
        f"attention_ms={attention_ms:.6f} "
        f"full_ms={full_ms:.6f} "
        f"index_share_pct="
        f"{derived['index_share_of_full_pct']:.2f} "
        f"attention_share_pct="
        f"{derived['attention_share_of_full_pct']:.2f} "
        f"residual_ms={derived['full_residual_ms']:.6f}",
        flush=True,
    )

    record = {
        "operator": "msa",
        "stage": "decode_breakdown",
        "implementation": "adapted",
        "case": args.case,
        "dtype": "bfloat16",
        "shape": [
            batch,
            decode_query_len,
            num_kv_heads,
            num_query_heads,
            HEAD_DIM,
        ],
        "sequence_lengths": list(sequence_lengths),
        "topk": args.topk,
        "stages": stages,
        "derived": derived,
    }
    write_json(args.json_output, record)
    print("S60 MSA DECODE BREAKDOWN PASS")


@pytest.mark.minimax_sparse_attention
def test_msa_decode_breakdown_benchmark(benchmark_options) -> None:
    case = benchmark_options.get("case", required=True)
    if case not in CASES:
        raise pytest.UsageError(
            f"--case must be one of {', '.join(CASES)} for MSA decode breakdown"
        )
    args = argparse.Namespace(
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
