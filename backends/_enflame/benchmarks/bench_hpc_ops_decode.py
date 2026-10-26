"""Same-card S60 benchmark for dynamic HPC paged-attention decode."""

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

import pytest  # noqa: E402
import torch  # noqa: E402
import torch_gcu  # noqa: E402, F401

from backends._enflame.benchmarks._common import measure, write_json  # noqa: E402
from backends._enflame.ops.hpc_ops_attention import (  # noqa: E402
    BLOCK_SIZE,
    prepare_topscc_decode_bf16,
    prepare_topscc_decode_fp8,
)


HEAD_DIM = 128
pytestmark = [pytest.mark.hpc_ops_attention]
QUANT_TYPES = (
    "bf16",
    "qpertoken_perhead_kvpertensor",
    "qkpertoken_perhead_vperhead",
)
FP8_QUANT_TYPES = QUANT_TYPES[1:]
CASES = {
    "uniform_512": (512,) * 64,
    "uniform_4096": (4096,) * 64,
    "skewed_mix": (128,) * 32 + (4096,) * 32,
    "skewed_extreme": (64,) * 15 + (16 * 1024,),
    "one_64k_7x4k": (64 * 1024,) + (4096,) * 7,
    "one_64k_15x4k": (64 * 1024,) + (4096,) * 15,
    "one_64k_31x4k": (64 * 1024,) + (4096,) * 31,
    "one_128k_31x4k": (128 * 1024,) + (4096,) * 31,
    "two_32k_30x4k": (32 * 1024,) * 2 + (4096,) * 30,
}


@dataclass
class Panel:
    q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    block_ids: torch.Tensor
    kv_lens: torch.Tensor
    q_scale: torch.Tensor | None = None
    k_scale: torch.Tensor | None = None
    v_scale: torch.Tensor | None = None


def _as_hnd_view(cache: torch.Tensor) -> torch.Tensor:
    return cache.permute(0, 2, 1, 3).contiguous().permute(0, 2, 1, 3)


def _copy_to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    if tensor.dtype == torch.float8_e4m3fn:
        return tensor.view(torch.uint8).to(device).view(torch.float8_e4m3fn)
    return tensor.to(device)


def _quantize_k_per_token(storage: torch.Tensor) -> torch.Tensor:
    blocks, _, heads, dim = storage.shape
    scale = storage[:, :BLOCK_SIZE].float().abs().amax(-1).clamp_min(1e-6) / 448.0
    result = torch.empty_like(storage, dtype=torch.float8_e4m3fn)
    result[:, :BLOCK_SIZE] = (storage[:, :BLOCK_SIZE] / scale[..., None]).to(
        torch.float8_e4m3fn
    )
    packed = (
        scale.permute(0, 2, 1)
        .contiguous()
        .view(torch.float8_e4m3fn)
        .reshape(blocks, heads, -1, dim)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    result[:, BLOCK_SIZE:] = packed
    return result


def _quantize_v_per_head(
    storage: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    heads = storage.shape[2]
    scale = (
        storage[:, :BLOCK_SIZE]
        .float()
        .abs()
        .permute(2, 0, 1, 3)
        .reshape(heads, -1)
        .amax(-1)
        .clamp_min(1e-6)
        / 448.0
    )
    quantized = (storage.float() / scale[None, None, :, None]).to(torch.float8_e4m3fn)
    return quantized, scale


def make_inputs(
    lengths: tuple[int, ...],
    *,
    mtp: int,
    num_kv_heads: int,
    num_query_heads: int,
    layout: str,
    quant_type: str,
    device: torch.device,
) -> Panel:
    """Build inputs on CPU, including FP8 quantization, then stage to GCU."""

    torch.manual_seed(41)
    kv_lens = torch.tensor(lengths, dtype=torch.int32)
    if min(lengths) < mtp:
        raise ValueError("every final KV length must be at least MTP")
    block_counts = (kv_lens + BLOCK_SIZE - 1) // BLOCK_SIZE
    total_blocks = int(block_counts.sum().item())
    capacity = int(total_blocks * 1.2) + len(lengths) + 8
    q_bf16 = torch.randn(
        len(lengths) * mtp,
        num_query_heads,
        HEAD_DIM,
        dtype=torch.bfloat16,
    ) / math.sqrt(HEAD_DIM)

    if quant_type == "bf16":
        q = q_bf16
        k_cache = torch.randn(
            capacity,
            BLOCK_SIZE,
            num_kv_heads,
            HEAD_DIM,
            dtype=torch.bfloat16,
        ) / math.sqrt(HEAD_DIM)
        v_cache = torch.randn_like(k_cache)
        q_scale = k_scale = v_scale = None
    else:
        q_scale = q_bf16.float().abs().amax(-1).clamp_min(1e-6)
        q = (q_bf16 / q_scale[..., None]).to(torch.float8_e4m3fn)
        if quant_type == "qkpertoken_perhead_vperhead":
            raw_k = torch.randn(
                capacity,
                BLOCK_SIZE + 2,
                num_kv_heads,
                HEAD_DIM,
                dtype=torch.bfloat16,
            )
            raw_v = torch.randn_like(raw_k)
            k_storage = _quantize_k_per_token(raw_k)
            v_storage, v_scale = _quantize_v_per_head(raw_v)
            k_cache = k_storage[:, :BLOCK_SIZE]
            v_cache = v_storage[:, :BLOCK_SIZE]
            k_scale = k_storage[:, BLOCK_SIZE:]
        else:
            k_cache = (
                torch.randn(
                    capacity,
                    BLOCK_SIZE,
                    num_kv_heads,
                    HEAD_DIM,
                    dtype=torch.bfloat16,
                )
                / math.sqrt(HEAD_DIM)
            ).to(torch.float8_e4m3fn)
            v_cache = torch.randn(
                capacity,
                BLOCK_SIZE,
                num_kv_heads,
                HEAD_DIM,
                dtype=torch.bfloat16,
            ).to(torch.float8_e4m3fn)
            k_scale = torch.rand(1, dtype=torch.float32).clamp_min(1e-6)
            v_scale = torch.rand(1, dtype=torch.float32).clamp_min(1e-6)

    if layout == "HND":
        k_cache = _as_hnd_view(k_cache)
        v_cache = _as_hnd_view(v_cache)

    packed_ids = torch.randperm(capacity)[:total_blocks].to(torch.int32)
    block_ids = torch.empty(
        len(lengths), int(block_counts.max().item()), dtype=torch.int32
    )
    offset = 0
    for batch_id, count in enumerate(block_counts.tolist()):
        block_ids[batch_id, :count] = packed_ids[offset : offset + count]
        offset += count

    panel = Panel(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_ids=block_ids,
        kv_lens=kv_lens,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    return Panel(
        **{
            name: (None if value is None else _copy_to_device(value, device))
            for name, value in panel.__dict__.items()
        }
    )


def _adapted_call(panel: Panel, layout: str, quant_type: str):
    if quant_type == "bf16":
        from flag_attn.runtime.backend._enflame.hpc_ops_attention.decode.dynamic import (
            bf16_dynamic as implementation,
        )

        inputs = implementation.DynamicBF16Inputs(
            panel.q,
            panel.k_cache,
            panel.v_cache,
            panel.block_ids,
            panel.kv_lens,
            layout,
        )
        workspace = implementation.prepare_dynamic_bf16_workspace(inputs)
        return lambda: implementation.attention_decode_bf16_dynamic(inputs, workspace)

    if quant_type == "qpertoken_perhead_kvpertensor":
        from flag_attn.runtime.backend._enflame.hpc_ops_attention.decode.dynamic import (
            fp8_qpertoken_perhead_kvpertensor_dynamic as implementation,
        )
    else:
        from flag_attn.runtime.backend._enflame.hpc_ops_attention.decode.dynamic import (
            fp8_qkpertoken_perhead_vperhead_dynamic as implementation,
        )

    inputs = implementation.FP8DecodeInputs(
        panel.q,
        panel.k_cache,
        panel.v_cache,
        panel.block_ids,
        panel.kv_lens,
        panel.q_scale,
        panel.k_scale,
        panel.v_scale,
    )
    workspace = implementation.prepare_decode_workspace(inputs)
    return lambda: implementation.attention_decode_fp8(inputs, workspace)


def _topscc_call(panel: Panel, layout: str, quant_type: str):
    if quant_type != "bf16":
        return prepare_topscc_decode_fp8(
            panel.q,
            panel.k_cache,
            panel.v_cache,
            panel.block_ids,
            panel.kv_lens,
            panel.q_scale,
            panel.k_scale,
            panel.v_scale,
            layout,
            quant_type=quant_type,
        )
    return prepare_topscc_decode_bf16(
        panel.q,
        panel.k_cache,
        panel.v_cache,
        panel.block_ids,
        panel.kv_lens,
        layout,
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
        help="run every official case, layout, and supported MTP value",
    )
    parser.add_argument("--case", choices=tuple(CASES))
    parser.add_argument("--quant-type", choices=(*QUANT_TYPES, "fp8"), default="bf16")
    parser.add_argument("--layout", choices=("NHD", "HND"), default="NHD")
    parser.add_argument("--mtp", type=int, default=1)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--inner", type=int, default=1)
    parser.add_argument("--num-kv-heads", type=int, default=1)
    parser.add_argument("--num-query-heads", type=int, default=8)
    parser.add_argument("--json-output")
    return parser.parse_args()


def _validate_configuration(args: argparse.Namespace) -> None:
    if args.quant_type == "bf16" and args.mtp not in (1, 2, 3):
        raise ValueError("BF16 MTP must be 1, 2, or 3")
    if args.quant_type != "bf16" and args.mtp not in (1, 2, 4):
        raise ValueError("FP8 MTP must be 1, 2, or 4")
    if args.num_query_heads % args.num_kv_heads:
        raise ValueError("query heads must be divisible by KV heads")


def _run_configuration(
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    _validate_configuration(args)
    lengths = CASES[args.case]
    panel = make_inputs(
        lengths,
        mtp=args.mtp,
        num_kv_heads=args.num_kv_heads,
        num_query_heads=args.num_query_heads,
        layout=args.layout,
        quant_type=args.quant_type,
        device=device,
    )
    metadata = {
        "operator": "hpc_ops_attention",
        "stage": "decode_dynamic",
        "case": args.case,
        "dtype": args.quant_type,
        "layout": args.layout,
        "mtp": args.mtp,
        "batch": len(lengths),
        "sequence_lengths": list(lengths),
        "num_query_heads": args.num_query_heads,
        "num_kv_heads": args.num_kv_heads,
        "head_dim": HEAD_DIM,
    }

    def run_provider(provider: str):
        if provider == "topscc":
            run = _topscc_call(panel, args.layout, args.quant_type)
        else:
            run = _adapted_call(panel, args.layout, args.quant_type)
        samples, output = measure(
            run,
            warmup=args.warmup,
            repeat=args.repeat,
            inner=args.inner,
            print_iterations=False,
        )
        finite = bool(torch.isfinite(output.float()).all().item())
        if not finite:
            raise AssertionError(f"{provider} dynamic decode output is not finite")
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
            "speedup": (baseline_stats["average_ms"] / triton_stats["average_ms"]),
            "max_abs_diff": max_abs_diff,
            "finite": True,
        }
        prefix = "topscc"
        record.update(
            {
                f"{prefix}_average_ms": baseline_stats["average_ms"],
                f"{prefix}_median_ms": baseline_stats["median_ms"],
                f"{prefix}_min_ms": baseline_stats["min_ms"],
                f"{prefix}_max_ms": baseline_stats["max_ms"],
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

    gc.collect()
    torch.gcu.empty_cache()
    return record


def _format_summary(records: list[dict[str, object]], args) -> None:
    comparison = bool(
        records and records[0]["implementation"] == "topscc-compare"
    )
    if comparison:
        baseline_name = str(records[0]["baseline"])
        baseline_label = "Torch Baseline" if baseline_name == "torch" else "TopsCC"
        headers = (
            "Case",
            "MTP",
            "Layout",
            "Quant Type",
            f"{baseline_label} (ms)",
            "Triton (ms)",
            "Speedup",
            "Max Abs Diff",
        )
        rows = [headers]
        rows.extend(
            (
                str(record["case"]),
                str(record["mtp"]),
                str(record["layout"]),
                str(record["dtype"]),
                f'{record["baseline_average_ms"]:.6f}',
                f'{record["triton_average_ms"]:.6f}',
                f'{record["speedup"]:.3f}x',
                f'{record["max_abs_diff"]:.6f}',
            )
            for record in records
        )
    else:
        headers = (
            "Case",
            "Schedule",
            "MTP",
            "Layout",
            "Quant Type",
            "Implementation",
            "Average (ms)",
        )
        rows = [headers]
        rows.extend(
            (
                str(record["case"]),
                "dynamic",
                str(record["mtp"]),
                str(record["layout"]),
                str(record["dtype"]),
                str(record["implementation"]),
                f'{record["average_ms"]:.6f}',
            )
            for record in records
        )
    widths = [max(len(row[index]) for row in rows) for index in range(len(headers))]

    def format_row(row) -> str:
        return (
            "| "
            + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))
            + " |"
        )

    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    print("S60 HPC-Ops Decode Attention performance comparison (mean)")
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
        print(f"Speedup = {baseline_label} / Triton; " "> 1 means Triton is faster.")


def _matrix_configurations(args: argparse.Namespace) -> list[argparse.Namespace]:
    if not getattr(args, "all", False):
        if args.case is None:
            raise ValueError("--case is required unless --all is specified")
        if args.quant_type == "fp8":
            raise ValueError("--quant-type fp8 is only valid together with --all")
        return [args]

    if args.case is not None:
        raise ValueError("--case cannot be combined with --all")
    quant_types = FP8_QUANT_TYPES if args.quant_type == "fp8" else (args.quant_type,)
    configurations = []
    for case in CASES:
        for quant_type in quant_types:
            mtp_values = (1, 2, 3) if quant_type == "bf16" else (1, 2, 4)
            for layout in ("NHD", "HND"):
                for mtp in mtp_values:
                    values = vars(args).copy()
                    values.update(
                        case=case,
                        quant_type=quant_type,
                        layout=layout,
                        mtp=mtp,
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
        _run_configuration(configuration, device) for configuration in configurations
    ]
    payload = records[0] if len(records) == 1 else {"results": records}
    write_json(args.json_output, payload)
    _format_summary(records, args)


def test_hpc_ops_dynamic_decode_benchmark(benchmark_options) -> None:
    run_all = benchmark_options.get("all", False)
    case = benchmark_options.get("case", required=not run_all)
    if case is not None and case not in CASES:
        raise pytest.UsageError(f"--case must be one of {', '.join(CASES)}")
    args = argparse.Namespace(
        implementation=benchmark_options.get("implementation", "topscc-compare"),
        all=run_all,
        case=case,
        quant_type=benchmark_options.get("quant_type", "bf16"),
        layout=benchmark_options.get("layout", "NHD"),
        mtp=benchmark_options.get("mtp", 1),
        device=benchmark_options.get("device", 0),
        warmup=benchmark_options.get("warmup", 2),
        repeat=benchmark_options.get("repeat", 5),
        inner=benchmark_options.get("inner", 1),
        num_kv_heads=benchmark_options.get("num_kv_heads", 1),
        num_query_heads=benchmark_options.get("num_query_heads", 8),
        json_output=benchmark_options.get("json_output"),
    )
    main(args)


if __name__ == "__main__":
    main()
