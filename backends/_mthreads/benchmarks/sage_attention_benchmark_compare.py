# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compare FlagAttention Triton SageAttention with Moore Threads MATE.

The two implementations do not use exactly the same quantization algorithm:

* FlagAttention: Q/K INT8 per block (Q=128, K=64), V FP16.
* MATE default: Q/K INT8 with recipe (128, 16, -1, 1), V FP8.

For that reason this script reports both accuracy against a common FP32
reference and two separate performance comparisons:

* core: input quantization is performed before timing;
* end-to-end: quantization and attention are timed together.

Results are printed to the terminal only.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Callable


def _find_flagattention_root() -> Path:
    configured_root = os.environ.get("FLAG_ATTENTION_ROOT")
    if configured_root:
        root = Path(configured_root).expanduser().resolve()
        if not (root / "src" / "flag_attn").is_dir():
            raise RuntimeError(
                "FLAG_ATTENTION_ROOT must point to the FlagAttention repository root; "
                f"got {root}"
            )
        return root

    for parent in Path(__file__).resolve().parents:
        if (parent / "src" / "flag_attn").is_dir():
            return parent
    raise RuntimeError(
        "Could not locate FlagAttention. Put BaselineBenchmark-main inside the "
        "FlagAttention repository or set FLAG_ATTENTION_ROOT=/path/to/FlagAttention-main."
    )


PROJECT_ROOT = _find_flagattention_root()
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import triton  # noqa: E402

try:  # noqa: E402
    import torch_musa  # noqa: F401
except ImportError as exc:  # pragma: no cover - only runs on the MUSA server
    raise RuntimeError("torch_musa is required for this comparison") from exc

try:  # noqa: E402
    import mate
    import sageattention
    from mate.testing import quantize_sage_attention_tensor
    from sageattention import sageattn
except ImportError as exc:  # pragma: no cover - only runs on the MUSA server
    raise RuntimeError(
        "MATE SageAttention is required. Install sageattention from the "
        "Moore Threads package index first."
    ) from exc

from flag_attn.runtime.backend._mthreads.sage_attention import (  # noqa: E402
    forward as flag_forward,
    per_block_int8 as flag_per_block_int8,
)


MATE_RECIPE = (128, 16, -1, 1)
DEFAULT_DTYPES = ("float16", "bfloat16")
DEFAULT_HEAD_DIMS = (64, 128)
DEFAULT_SEQ_LENS = (1024, 4096, 8192, 16384)
DEFAULT_CHECK_SHAPES = (
    "1x128x1",
    "1x256x8",
    "1x512x16",
    "1x1024x32",
    "2x256x8",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare FlagAttention Triton SageAttention with the official "
            "Moore Threads MATE implementation"
        )
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("benchmark", "correctness", "all"),
        default="benchmark",
        help=(
            "benchmark (default) runs performance only; correctness runs "
            "accuracy only; all runs both"
        ),
    )
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument(
        "--head-dims",
        type=int,
        nargs="+",
        choices=(64, 128),
        default=DEFAULT_HEAD_DIMS,
    )
    parser.add_argument(
        "--seq-lens", type=int, nargs="+", default=DEFAULT_SEQ_LENS
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=DEFAULT_DTYPES,
        default=DEFAULT_DTYPES,
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--rep", type=int, default=200)
    parser.add_argument(
        "--batch4-seq-len",
        type=int,
        default=1024,
        help="Also benchmark B=4 at this sequence length; use 0 to disable",
    )
    parser.add_argument(
        "--check-shapes",
        nargs="+",
        default=DEFAULT_CHECK_SHAPES,
        metavar="BxTxH",
        help=(
            "Correctness shapes in BxTxH form; D and dtype come from "
            "--head-dims and --dtypes"
        ),
    )
    parser.add_argument("--min-cosine", type=float, default=0.998)
    parser.add_argument("--max-relative-l1", type=float, default=0.03)
    parser.add_argument("--max-rmse", type=float, default=0.015)
    parser.add_argument(
        "--skip-core", action="store_true", help="Skip pre-quantized kernel timing"
    )
    parser.add_argument(
        "--skip-e2e", action="store_true", help="Skip end-to-end timing"
    )
    return parser.parse_args()


def synchronize() -> None:
    torch.musa.synchronize()


def _seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.musa.manual_seed(seed)


def _dtype(name: str) -> torch.dtype:
    return getattr(torch, name)


def _make_inputs(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = (batch_size, num_heads, seq_len, head_dim)
    q = torch.randn(shape, device="musa", dtype=dtype)
    k = torch.randn(shape, device="musa", dtype=dtype)
    v = torch.randn(shape, device="musa", dtype=dtype)
    return q, k, v


def _flag_v(v: torch.Tensor) -> torch.Tensor:
    # The current FlagAttention MUSA kernel requires V to be FP16. For BF16
    # source inputs, E2E timing includes this conversion while core timing does
    # not. Accuracy is always measured against the original source values.
    return v if v.dtype == torch.float16 else v.to(torch.float16)


def _flag_prequantize(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q_int8, q_scale, k_int8, k_scale = flag_per_block_int8(
        q, k, tensor_layout="HND"
    )
    return q_int8, q_scale, k_int8, k_scale, _flag_v(v)


def _flag_core(
    prepared: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    output_dtype: torch.dtype,
) -> torch.Tensor:
    q_int8, q_scale, k_int8, k_scale, v = prepared
    out, _ = flag_forward(
        q_int8,
        k_int8,
        v,
        q_scale,
        k_scale,
        tensor_layout="HND",
        output_dtype=output_dtype,
    )
    return out


def _flag_e2e(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    prepared = _flag_prequantize(q, k, v)
    return _flag_core(prepared, q.dtype)


def _mate_prequantize(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # MATE's low-level interface uses BNHD and its public wrapper converts
    # source tensors to BF16 before quantization.
    q_bnhd = q.transpose(1, 2).contiguous().to(torch.bfloat16)
    k_bnhd = k.transpose(1, 2).contiguous().to(torch.bfloat16)
    v_bnhd = v.transpose(1, 2).contiguous().to(torch.bfloat16)

    q_quant, q_scale = quantize_sage_attention_tensor(
        q_bnhd,
        operand="q",
        quant_recipe=MATE_RECIPE,
        quant_dtype=torch.int8,
    )
    k_quant, k_scale = quantize_sage_attention_tensor(
        k_bnhd,
        operand="k",
        quant_recipe=MATE_RECIPE,
        quant_dtype=torch.int8,
        smooth_k=True,
    )
    v_quant, v_scale = quantize_sage_attention_tensor(
        v_bnhd,
        operand="v",
        quant_recipe=MATE_RECIPE,
        quant_dtype=torch.float8_e4m3fn,
    )
    return q_quant, q_scale, k_quant, k_scale, v_quant, v_scale


def _mate_core(
    prepared: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    head_dim: int,
) -> torch.Tensor:
    q_quant, q_scale, k_quant, k_scale, v_quant, v_scale = prepared
    return mate.sage_attn_quantized(
        q=q_quant,
        k=k_quant,
        v=v_quant,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        softmax_scale=head_dim**-0.5,
        causal=False,
        quant_recipe=MATE_RECIPE,
        return_lse=False,
    )


def _mate_e2e(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    return sageattn(
        q,
        k,
        v,
        tensor_layout="HND",
        is_causal=False,
        qk_quant_dtype="int8",
        quant_recipe=MATE_RECIPE,
        smooth_k=True,
        return_lse=False,
    )


def _reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1))
    scores *= q.shape[-1] ** -0.5
    return torch.matmul(torch.softmax(scores, dim=-1), v.float())


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | bool]:
    actual = actual.float()
    expected = expected.float()
    diff = actual - expected
    return {
        "finite": bool(torch.isfinite(actual).all().item()),
        "cosine": F.cosine_similarity(
            actual.flatten(), expected.flatten(), dim=0
        ).item(),
        "relative_l1": (
            diff.abs().sum()
            / (actual.abs().sum() + expected.abs().sum() + 1e-8)
        ).item(),
        "rmse": diff.square().mean().sqrt().item(),
        "max_abs": diff.abs().max().item(),
    }


def _accuracy_passes(result: dict[str, float | bool], args: argparse.Namespace) -> bool:
    return bool(
        result["finite"]
        and result["cosine"] >= args.min_cosine
        and result["relative_l1"] <= args.max_relative_l1
        and result["rmse"] <= args.max_rmse
    )


def _parse_check_shape(spec: str) -> tuple[int, int, int]:
    try:
        batch_size, seq_len, num_heads = (
            int(value) for value in spec.lower().split("x")
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid correctness shape {spec!r}; expected BxTxH, for example 1x128x1"
        ) from exc
    if batch_size <= 0 or seq_len <= 0 or num_heads <= 0:
        raise ValueError(f"Correctness shape dimensions must be positive: {spec!r}")
    return batch_size, seq_len, num_heads


def run_correctness(args: argparse.Namespace) -> bool:
    print("\nCorrectness Results")
    print(
        f"{'B':>3} {'T':>6} {'H':>3} {'D':>4} {'dtype':>9} "
        f"{'comparison':>18} {'cosine':>10} {'rel_l1':>10} "
        f"{'rmse':>10} {'max_abs':>10} {'status':>7}"
    )
    print("-" * 112)

    all_passed = True
    check_shapes = [_parse_check_shape(spec) for spec in args.check_shapes]
    for batch_size, seq_len, num_heads in check_shapes:
        for dtype_name in args.dtypes:
            dtype = _dtype(dtype_name)
            for head_dim in args.head_dims:
                _seed(
                    123
                    + batch_size * 17
                    + seq_len * 3
                    + num_heads * 5
                    + head_dim
                )
                q, k, v = _make_inputs(
                    batch_size,
                    num_heads,
                    seq_len,
                    head_dim,
                    dtype,
                )

                flag_out = _flag_e2e(q, k, v)
                mate_out = _mate_e2e(q, k, v)
                ref_out = _reference(q, k, v)
                synchronize()

                comparisons = (
                    ("Triton vs FP32", _metrics(flag_out, ref_out), True),
                    ("MATE vs FP32", _metrics(mate_out, ref_out), True),
                    ("Triton vs MATE", _metrics(flag_out, mate_out), False),
                )

                for name, result, enforce in comparisons:
                    passed = _accuracy_passes(result, args) if enforce else None
                    if enforce:
                        all_passed = all_passed and bool(passed)
                    status = "PASS" if passed else "FAIL" if enforce else "INFO"
                    print(
                        f"{batch_size:>3d} "
                        f"{seq_len:>6d} "
                        f"{num_heads:>3d} "
                        f"{head_dim:>4d} "
                        f"{dtype_name:>9} "
                        f"{name:>18} "
                        f"{result['cosine']:>10.6f} "
                        f"{result['relative_l1']:>10.6f} "
                        f"{result['rmse']:>10.6f} "
                        f"{result['max_abs']:>10.6f} "
                        f"{status:>7}"
                    )

                del q, k, v, flag_out, mate_out, ref_out
                torch.musa.empty_cache()

    return all_passed


def _bench(fn: Callable[[], torch.Tensor], warmup: int, rep: int) -> float:
    # Use whole-device synchronization and wall-clock timing so work launched
    # through Triton and through MATE's TVM-FFI/MUBIN path is measured even if
    # the two runtimes do not record work on the same event-visible stream.
    fn()
    synchronize()
    for _ in range(warmup):
        fn()
    synchronize()

    start = time.perf_counter()
    for _ in range(rep):
        fn()
    synchronize()
    return (time.perf_counter() - start) * 1000.0 / rep


def _shape_specs(args: argparse.Namespace):
    for dtype_name in args.dtypes:
        for head_dim in args.head_dims:
            for seq_len in args.seq_lens:
                yield 1, seq_len, args.num_heads, head_dim, dtype_name
            if args.batch4_seq_len > 0:
                yield (
                    4,
                    args.batch4_seq_len,
                    args.num_heads,
                    head_dim,
                    dtype_name,
                )


def _print_performance_table(title: str, results: list[dict[str, float | int | str]]) -> None:
    print(f"\n{title}")
    print(
        f"{'B':>3} {'T':>6} {'H':>3} {'D':>4} {'dtype':>9} "
        f"{'Triton_ms':>11} {'MATE_ms':>11} {'Triton_speedup':>15} "
        f"{'Triton_TF':>11} {'MATE_TF':>9}"
    )
    print("-" * 94)
    for result in results:
        print(
            f"{result['batch_size']:>3d} "
            f"{result['seq_len']:>6d} "
            f"{result['num_heads']:>3d} "
            f"{result['head_dim']:>4d} "
            f"{result['dtype']:>9} "
            f"{result['flag_ms']:>11.4f} "
            f"{result['mate_ms']:>11.4f} "
            f"{result['triton_speedup']:>15.3f} "
            f"{result['flag_tflops']:>11.2f} "
            f"{result['mate_tflops']:>9.2f}"
        )


def run_benchmark(args: argparse.Namespace) -> None:
    core_results: list[dict[str, float | int | str]] = []
    e2e_results: list[dict[str, float | int | str]] = []

    for batch_size, seq_len, num_heads, head_dim, dtype_name in _shape_specs(args):
        dtype = _dtype(dtype_name)
        seed = 1000 + batch_size * 17 + seq_len + head_dim
        _seed(seed)
        q, k, v = _make_inputs(
            batch_size, num_heads, seq_len, head_dim, dtype
        )
        flops = 4 * batch_size * num_heads * seq_len * seq_len * head_dim

        print(
            f"Running B={batch_size}, T={seq_len}, H={num_heads}, "
            f"D={head_dim}, dtype={dtype_name}",
            flush=True,
        )

        if not args.skip_core:
            flag_prepared = _flag_prequantize(q, k, v)
            mate_prepared = _mate_prequantize(q, k, v)

            flag_ms = _bench(
                lambda: _flag_core(flag_prepared, dtype), args.warmup, args.rep
            )
            mate_ms = _bench(
                lambda: _mate_core(mate_prepared, head_dim), args.warmup, args.rep
            )
            core_results.append(
                {
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "num_heads": num_heads,
                    "head_dim": head_dim,
                    "dtype": dtype_name,
                    "flag_ms": flag_ms,
                    "mate_ms": mate_ms,
                    "triton_speedup": mate_ms / flag_ms,
                    "flag_tflops": flops / flag_ms * 1e-9,
                    "mate_tflops": flops / mate_ms * 1e-9,
                }
            )
            del flag_prepared, mate_prepared

        if not args.skip_e2e:
            flag_ms = _bench(lambda: _flag_e2e(q, k, v), args.warmup, args.rep)
            mate_ms = _bench(lambda: _mate_e2e(q, k, v), args.warmup, args.rep)
            e2e_results.append(
                {
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "num_heads": num_heads,
                    "head_dim": head_dim,
                    "dtype": dtype_name,
                    "flag_ms": flag_ms,
                    "mate_ms": mate_ms,
                    "triton_speedup": mate_ms / flag_ms,
                    "flag_tflops": flops / flag_ms * 1e-9,
                    "mate_tflops": flops / mate_ms * 1e-9,
                }
            )

        del q, k, v
        torch.musa.empty_cache()

    if core_results:
        _print_performance_table(
            "Core Performance (quantization excluded)", core_results
        )
    if e2e_results:
        _print_performance_table(
            "End-to-End Performance (quantization included)", e2e_results
        )


def print_environment(args: argparse.Namespace) -> None:
    print("SageAttention comparison: FlagAttention Triton vs Moore Threads MATE")
    print(f"device={torch.musa.get_device_name(0)}")
    print(f"capability={torch.musa.get_device_capability(0)}")
    print(f"torch={torch.__version__}")
    print(f"torch_musa={getattr(torch_musa, '__version__', 'unknown')}")
    print(f"triton={triton.__version__}")
    print(f"mate={getattr(mate, '__version__', 'unknown')}")
    print(f"sageattention={getattr(sageattention, '__version__', 'unknown')}")
    print(f"MATE quant_recipe={MATE_RECIPE}, smooth_k=True, V=FP8")
    print("FlagAttention quantization: Q block=128, K block=64, V=FP16")
    print(f"warmup={args.warmup}, rep={args.rep}")
    print("timer=synchronized wall clock (all MUSA streams completed)")
    print("Triton_speedup = MATE latency / Triton latency; >1 means Triton is faster")
    if "bfloat16" in args.dtypes:
        print(
            "Note: the current FlagAttention kernel requires V=FP16; for BF16 "
            "source inputs its E2E timing includes BF16->FP16 V conversion."
        )


def main() -> None:
    args = parse_args()
    if not torch.musa.is_available():
        raise RuntimeError("A MUSA device is required")
    if args.warmup <= 0 or args.rep <= 0:
        raise ValueError("--warmup and --rep must be positive")
    for spec in args.check_shapes:
        _parse_check_shape(spec)

    print_environment(args)

    correctness_passed = True
    if args.mode in {"correctness", "all"}:
        correctness_passed = run_correctness(args)

    if args.mode in {"benchmark", "all"} and (
        not args.skip_core or not args.skip_e2e
    ):
        run_benchmark(args)

    print()
    if args.mode == "benchmark":
        print("Benchmark comparison completed.")
    elif correctness_passed:
        print("Comparison completed: all enforced correctness checks passed.")
    else:
        raise AssertionError("One or more correctness checks failed")


if __name__ == "__main__":
    main()
