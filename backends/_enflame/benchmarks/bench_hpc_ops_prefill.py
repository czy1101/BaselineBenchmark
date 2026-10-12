"""Same-card S60 benchmark for dynamic block-sparse FP8 prefill."""

from __future__ import annotations

import argparse
import gc
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import pytest
import torch
import torch_gcu  # noqa: F401

from backends._enflame.benchmarks._common import measure, write_json
from backends._enflame.ops.hpc_ops_attention import (
    PREFILL_BLOCK_SIZE,
    prepare_topscc_prefill_fp8,
)


HEAD_DIM = 128
PAGE_SIZE = 64
NUM_QUERY_HEADS = 32
NUM_KV_HEADS = 4
MASK_SKIP_RATIO = 0.75
SEED = 10086
QUANT_TYPES = (
    "qkpertoken_perhead_vperhead",
    "qpertoken_perhead_kvpertensor",
)
CASES = {
    "q127_kv127": (127, 127),
    "q512_kv4096": (512, 4096),
    "q512_kv32768": (512, 32768),
}


@dataclass
class Panel:
    q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    q_scale: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    cu_seqlens_q: torch.Tensor
    block_ids: torch.Tensor
    kv_lens: torch.Tensor
    max_q_len: int
    quant_type: str
    block_mask: torch.Tensor | None
    sparsity_bucket: int
    effective_skip_ratio: float


def _copy_to_device(tensor: torch.Tensor | None, device: torch.device):
    if tensor is None:
        return None
    if tensor.dtype == torch.float8_e4m3fn:
        return tensor.view(torch.uint8).to(device).view(torch.float8_e4m3fn)
    return tensor.to(device)


def _as_hnd_view(cache: torch.Tensor) -> torch.Tensor:
    return cache.transpose(1, 2).contiguous().transpose(1, 2)


def _sparsity_bucket(active_tiles: int, causal_tiles: int) -> int:
    active_ratio = active_tiles / causal_tiles if causal_tiles else 1.0
    if active_ratio >= 0.75:
        return 0
    if active_ratio >= 0.50:
        return 1
    if active_ratio >= 0.25:
        return 2
    return 3


def _make_block_mask(
    q_len: int,
    kv_len: int,
    masked: bool,
) -> tuple[torch.Tensor | None, int, float]:
    q_tiles = math.ceil(q_len / PREFILL_BLOCK_SIZE)
    kv_tiles = math.ceil(kv_len / PREFILL_BLOCK_SIZE)
    rows = torch.arange(q_tiles).view(q_tiles, 1)
    cols = torch.arange(kv_tiles).view(1, kv_tiles)
    causal_boundary = rows + (kv_tiles - q_tiles)
    causal = cols <= causal_boundary
    causal_tiles = int(causal.sum().item()) * NUM_QUERY_HEADS
    if not masked:
        return None, 0, 0.0

    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED + 17)
    block_mask = (
        torch.rand(
            (1, NUM_QUERY_HEADS, q_tiles, kv_tiles),
            generator=generator,
        )
        >= MASK_SKIP_RATIO
    )
    block_mask &= causal.view(1, 1, q_tiles, kv_tiles)

    # Retain the causal frontier and the first visible tile for every query
    # tile so no valid query row receives an all-masked softmax.
    frontier = torch.clamp(causal_boundary, max=kv_tiles - 1)
    block_mask |= (cols == frontier).view(1, 1, q_tiles, kv_tiles)
    first_q_positions = kv_len - q_len + rows * PREFILL_BLOCK_SIZE
    first_q_tiles = torch.div(
        first_q_positions,
        PREFILL_BLOCK_SIZE,
        rounding_mode="floor",
    ).clamp(min=0, max=kv_tiles - 1)
    block_mask |= (cols == first_q_tiles).view(1, 1, q_tiles, kv_tiles)

    active_tiles = int(block_mask.sum().item())
    return (
        block_mask.to(torch.uint8).contiguous(),
        _sparsity_bucket(active_tiles, causal_tiles),
        1.0 - active_tiles / causal_tiles,
    )


def make_inputs(
    q_len: int,
    kv_len: int,
    *,
    quant_type: str,
    masked: bool,
    layout: str,
    device: torch.device,
) -> Panel:
    """Build FP8 inputs on CPU, then preserve their bytes when staging."""

    torch.manual_seed(SEED)
    requested_pages = math.ceil(kv_len / PAGE_SIZE)
    physical_pages = max(requested_pages * 2, requested_pages + 8)
    fp8 = torch.float8_e4m3fn

    q = (
        torch.randn(q_len, NUM_QUERY_HEADS, HEAD_DIM, dtype=torch.bfloat16)
        / math.sqrt(HEAD_DIM)
    ).to(fp8)
    raw_cache = torch.randn(
        physical_pages,
        2,
        PAGE_SIZE,
        NUM_KV_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
    )
    k_cache = (raw_cache[:, 0] / math.sqrt(HEAD_DIM)).to(fp8)
    v_cache = raw_cache[:, 1].to(fp8)
    if layout == "HND":
        k_cache = _as_hnd_view(k_cache)
        v_cache = _as_hnd_view(v_cache)

    q_scale = torch.zeros(
        1,
        NUM_QUERY_HEADS,
        math.ceil(q_len / PREFILL_BLOCK_SIZE) * PREFILL_BLOCK_SIZE,
        dtype=torch.float32,
    )
    q_scale[:, :, :q_len] = (
        torch.randn(1, NUM_QUERY_HEADS, q_len).abs() / 10.0
    )
    if quant_type == "qkpertoken_perhead_vperhead":
        k_scale = (
            torch.randn(
                physical_pages,
                PAGE_SIZE // 32,
                NUM_KV_HEADS,
                HEAD_DIM // 4,
                dtype=torch.float32,
            )
            .abs()
            .clamp_min_(1e-6)
            .view(fp8)
        )
        v_scale = torch.randn(NUM_KV_HEADS).abs().clamp_min_(1e-6)
    elif quant_type == "qpertoken_perhead_kvpertensor":
        k_scale = torch.rand(1, dtype=torch.float32) + 0.5
        v_scale = torch.rand(1, dtype=torch.float32) + 0.5
    else:
        raise ValueError(f"unsupported quant_type: {quant_type}")

    cu_seqlens_q = torch.tensor([0, q_len], dtype=torch.int32)
    block_ids = torch.randperm(physical_pages)[:requested_pages]
    block_ids = block_ids.to(torch.int32).view(1, -1)
    kv_lens = torch.tensor([kv_len], dtype=torch.int32)
    block_mask, bucket, effective_skip_ratio = _make_block_mask(
        q_len, kv_len, masked
    )

    return Panel(
        q=_copy_to_device(q, device),
        k_cache=_copy_to_device(k_cache, device),
        v_cache=_copy_to_device(v_cache, device),
        q_scale=_copy_to_device(q_scale, device),
        k_scale=_copy_to_device(k_scale, device),
        v_scale=_copy_to_device(v_scale, device),
        cu_seqlens_q=_copy_to_device(cu_seqlens_q, device),
        block_ids=_copy_to_device(block_ids, device),
        kv_lens=_copy_to_device(kv_lens, device),
        max_q_len=q_len,
        quant_type=quant_type,
        block_mask=_copy_to_device(block_mask, device),
        sparsity_bucket=bucket,
        effective_skip_ratio=effective_skip_ratio,
    )


def _adapted_call(panel: Panel):
    from flag_attn.runtime.backend._enflame.hpc_ops_attention.prefill import (
        attention_with_kvcache_blocksparse_prefill_fp8,
    )

    output = torch.empty_like(panel.q, dtype=torch.bfloat16)
    quant_type = (
        0 if panel.quant_type == "qkpertoken_perhead_vperhead" else 1
    )
    return lambda: attention_with_kvcache_blocksparse_prefill_fp8(
        panel.q,
        panel.k_cache,
        panel.v_cache,
        panel.q_scale,
        panel.k_scale,
        panel.v_scale,
        panel.cu_seqlens_q,
        panel.block_ids,
        panel.kv_lens,
        panel.max_q_len,
        quant_type,
        block_mask=panel.block_mask,
        output=output,
        sparsity_bucket=panel.sparsity_bucket,
    )


def _topscc_call(panel: Panel, layout: str):
    return prepare_topscc_prefill_fp8(
        panel.q,
        panel.k_cache,
        panel.v_cache,
        panel.q_scale,
        panel.k_scale,
        panel.v_scale,
        panel.cu_seqlens_q,
        panel.block_ids,
        panel.kv_lens,
        panel.max_q_len,
        panel.quant_type,
        panel.block_mask,
        layout=layout,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--implementation",
        choices=("topscc", "adapted", "topscc-compare"),
        default="topscc-compare",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="run all Prefill FP8 cases, layouts, masks, and quant modes",
    )
    parser.add_argument("--case", choices=tuple(CASES))
    parser.add_argument(
        "--quant-type", choices=(*QUANT_TYPES, "fp8"), default=QUANT_TYPES[0]
    )
    parser.add_argument("--layout", choices=("NHD", "HND"), default="NHD")
    parser.add_argument(
        "--masked", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--inner", type=int, default=1)
    parser.add_argument("--json-output")
    return parser.parse_args()


def _run_configuration(
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    q_len, kv_len = CASES[args.case]
    panel = make_inputs(
        q_len,
        kv_len,
        quant_type=args.quant_type,
        masked=args.masked,
        layout=args.layout,
        device=device,
    )
    metadata = {
        "operator": "hpc_ops_attention",
        "stage": "prefill_dynamic",
        "case": args.case,
        "q_len": q_len,
        "kv_len": kv_len,
        "dtype": args.quant_type,
        "masked": args.masked,
        "effective_skip_ratio": panel.effective_skip_ratio,
        "layout": args.layout,
        "num_query_heads": NUM_QUERY_HEADS,
        "num_kv_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "page_size": PAGE_SIZE,
    }

    def run_provider(provider: str):
        if provider == "topscc":
            run = _topscc_call(panel, args.layout)
        else:
            run = _adapted_call(panel)
        samples, output = measure(
            run,
            warmup=args.warmup,
            repeat=args.repeat,
            inner=args.inner,
            print_iterations=False,
        )
        finite = bool(torch.isfinite(output.float()).all().item())
        if not finite:
            raise AssertionError(
                f"{provider} dynamic FP8 prefill output is not finite"
            )
        return {
            "average_ms": statistics.fmean(samples),
            "median_ms": statistics.median(samples),
            "min_ms": min(samples),
            "max_ms": max(samples),
        }, output

    if args.implementation == "topscc-compare":
        baseline = "topscc"
        baseline_stats, baseline_output = run_provider(baseline)
        triton_stats, triton_output = run_provider("adapted")
        max_abs_diff = float(
            (baseline_output.float() - triton_output.float()).abs().max().item()
        )
        record = {
            **metadata,
            "implementation": args.implementation,
            "baseline": baseline,
            "baseline_average_ms": baseline_stats["average_ms"],
            "baseline_median_ms": baseline_stats["median_ms"],
            "baseline_min_ms": baseline_stats["min_ms"],
            "baseline_max_ms": baseline_stats["max_ms"],
            "triton_average_ms": triton_stats["average_ms"],
            "triton_median_ms": triton_stats["median_ms"],
            "triton_min_ms": triton_stats["min_ms"],
            "triton_max_ms": triton_stats["max_ms"],
            "speedup": (
                baseline_stats["average_ms"] / triton_stats["average_ms"]
            ),
            "max_abs_diff": max_abs_diff,
            "finite": True,
        }
        record.update(
            {
                "topscc_average_ms": baseline_stats["average_ms"],
                "topscc_median_ms": baseline_stats["median_ms"],
                "topscc_min_ms": baseline_stats["min_ms"],
                "topscc_max_ms": baseline_stats["max_ms"],
            }
        )
        del baseline_output, triton_output
    else:
        stats, output = run_provider(args.implementation)
        record = {
            **metadata,
            "implementation": args.implementation,
            **stats,
            "finite": True,
        }
        del output

    del panel
    gc.collect()
    torch.gcu.empty_cache()
    return record


def _format_summary(
    records: list[dict[str, object]], args: argparse.Namespace
) -> None:
    comparison = bool(
        records and records[0]["implementation"] == "topscc-compare"
    )
    if comparison:
        baseline_label = "TopsCC"
        headers = (
            "Q Len",
            "KV Len",
            "Quant Type",
            "Mask",
            "Skip",
            "Layout",
            f"{baseline_label} (ms)",
            "Triton (ms)",
            "Speedup",
            "Max Abs Diff",
        )
        rows = [headers]
        rows.extend(
            (
                str(record["q_len"]),
                str(record["kv_len"]),
                str(record["dtype"]),
                str(record["masked"]),
                f'{record["effective_skip_ratio"]:.1%}',
                str(record["layout"]),
                f'{record["baseline_average_ms"]:.6f}',
                f'{record["triton_average_ms"]:.6f}',
                f'{record["speedup"]:.3f}x',
                f'{record["max_abs_diff"]:.6f}',
            )
            for record in records
        )
    else:
        headers = (
            "Q Len",
            "KV Len",
            "Quant Type",
            "Mask",
            "Skip",
            "Layout",
            "Implementation",
            "Average (ms)",
        )
        rows = [headers]
        rows.extend(
            (
                str(record["q_len"]),
                str(record["kv_len"]),
                str(record["dtype"]),
                str(record["masked"]),
                f'{record["effective_skip_ratio"]:.1%}',
                str(record["layout"]),
                str(record["implementation"]),
                f'{record["average_ms"]:.6f}',
            )
            for record in records
        )
    widths = [max(len(row[i]) for row in rows) for i in range(len(headers))]

    def format_row(row) -> str:
        return "| " + " | ".join(
            value.ljust(widths[index]) for index, value in enumerate(row)
        ) + " |"

    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    print("S60 HPC-Ops Prefill Attention FP8 performance comparison (mean)")
    print(
        f"Warmup: {args.warmup}; Timed samples: {args.repeat}; "
        f"Inner calls: {args.inner}"
    )
    print(separator)
    print(format_row(rows[0]))
    print(separator)
    for row in rows[1:]:
        print(format_row(row))
    print(separator)
    if comparison:
        baseline_label = "TopsCC"
        print(
            f"Speedup = {baseline_label} / Triton; "
            "> 1 means Triton is faster."
        )


def _matrix_configurations(args: argparse.Namespace) -> list[argparse.Namespace]:
    if not args.all:
        if args.case is None:
            raise ValueError("--case is required unless --all is specified")
        if args.quant_type == "fp8":
            raise ValueError("--quant-type fp8 is only valid with --all")
        return [args]
    if args.case is not None:
        raise ValueError("--case cannot be combined with --all")

    quant_types = QUANT_TYPES if args.quant_type == "fp8" else (args.quant_type,)
    configurations = []
    for case in CASES:
        for quant_type in quant_types:
            for masked in (False, True):
                for layout in ("NHD", "HND"):
                    values = vars(args).copy()
                    values.update(
                        case=case,
                        quant_type=quant_type,
                        masked=masked,
                        layout=layout,
                        all=False,
                    )
                    configurations.append(argparse.Namespace(**values))
    return configurations


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()
    configurations = _matrix_configurations(args)
    torch.gcu.set_device(args.device)
    device = torch.device(f"gcu:{args.device}")
    records = [
        _run_configuration(configuration, device)
        for configuration in configurations
    ]
    payload = records[0] if len(records) == 1 else {"results": records}
    write_json(args.json_output, payload)
    _format_summary(records, args)


def test_hpc_ops_dynamic_prefill_benchmark(benchmark_options) -> None:
    case = benchmark_options.get("case", required=True)
    if case not in CASES:
        raise pytest.UsageError(f"--case must be one of {', '.join(CASES)}")
    quant_type = benchmark_options.get("quant_type", QUANT_TYPES[0])
    if quant_type not in QUANT_TYPES:
        raise pytest.UsageError(
            f"--quant-type must be one of {', '.join(QUANT_TYPES)}"
        )
    args = argparse.Namespace(
        implementation=benchmark_options.get("implementation", "topscc-compare"),
        all=False,
        case=case,
        quant_type=quant_type,
        layout=benchmark_options.get("layout", "NHD"),
        masked=benchmark_options.get("masked", False),
        device=benchmark_options.get("device", 0),
        warmup=benchmark_options.get("warmup", 2),
        repeat=benchmark_options.get("repeat", 5),
        inner=benchmark_options.get("inner", 1),
        json_output=benchmark_options.get("json_output"),
    )
    main(args)


if __name__ == "__main__":
    main()
