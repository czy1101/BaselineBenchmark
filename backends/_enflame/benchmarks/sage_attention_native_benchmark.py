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

"""Correctness-gated benchmark for the Enflame native SageAttention path.

The reported native latency is end to end: INT8-to-FP16 per-block
dequantization, layout conversion, fused attention, and output conversion are
all inside the timed callable.  This is the latency relevant to replacing the
existing Triton ``forward`` at its current API boundary.
"""

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import triton


OPS_DIR = Path(__file__).resolve().parents[1] / "ops" / "sage_attention"
# The Triton SageAttention implementation lives in the FlagAttention repo.
# Default to the sibling FlagAttention checkout under the workspace root; set
# FLAGATTN_SAGE_ATTN_DIR to override (e.g. for a different container layout).
FLAGATTN_OPS_DIR = Path(
    os.environ.get(
        "FLAGATTN_SAGE_ATTN_DIR",
        str(
            Path(__file__).resolve().parents[4]
            / "FlagAttention"
            / "src"
            / "flag_attn"
            / "runtime"
            / "backend"
            / "_enflame"
            / "sage_attention"
        ),
    )
)
if not FLAGATTN_OPS_DIR.is_dir():
    raise FileNotFoundError(
        f"Triton SageAttention implementation not found at {FLAGATTN_OPS_DIR}"
    )
sys.path.insert(0, str(OPS_DIR))
sys.path.insert(0, str(FLAGATTN_OPS_DIR))

from attn_qk_int8_per_block import forward as triton_forward
from attn_qk_int8_per_block_native import (
    NativeBackendUnavailable,
    forward as native_forward,
    native_backend_status,
    resolve_backend,
)


_TABLE_BATCH_SEQ_SHAPES = (
    (1, 1024),
    (4, 1024),
    (1, 4096),
    (1, 8192),
    (1, 16384),
)
_TABLE_HEAD_DIMS = (64, 128)
_TABLE_DTYPES = ("float16", "bfloat16")
_LN2 = math.log(2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the Enflame native SageAttention forward"
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "torch-gcu", "vllm-gcu", "flash-attn", "triton"),
        default="auto",
    )
    parser.add_argument(
        "--allow-triton-fallback",
        action="store_true",
        help="Allow --backend auto to benchmark Triton when no native path exists",
    )
    parser.add_argument(
        "--compare-triton",
        action="store_true",
        help=(
            "Also time the optimized operator and report its speedup over "
            "the native Torch baseline"
        ),
    )
    parser.add_argument(
        "--custom-shapes",
        action="store_true",
        help=(
            "Use the custom B/H/HKV/D/T/dtype arguments instead of the "
            "default 20-shape table"
        ),
    )
    parser.add_argument(
        "--isolate-shapes",
        action="store_true",
        help=(
            "Run every shape in a fresh Python process to avoid allocator, "
            "compiler-cache, and device-state interference between shapes"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument(
        "--num-kv-heads",
        type=int,
        help="Defaults to --num-heads; set a smaller value for GQA/MQA",
    )
    parser.add_argument("--head-dim", type=int, choices=(64, 128), default=128)
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=(1024, 4096, 8192, 16384),
    )
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16"), default="float16"
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rep", type=int, default=20)
    reference_group = parser.add_mutually_exclusive_group()
    reference_group.add_argument(
        "--check-reference",
        dest="check_reference",
        action="store_true",
        help="Run a small CPU-reference gate before timing (default)",
    )
    reference_group.add_argument(
        "--no-check-reference",
        dest="check_reference",
        action="store_false",
    )
    parser.set_defaults(check_reference=True)
    parser.add_argument("--atol", type=float, default=3e-2)
    parser.add_argument("--rtol", type=float, default=3e-2)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def _configs(args):
    if args.custom_shapes:
        num_kv_heads = args.num_kv_heads or args.num_heads
        return (
            (
                args.batch_size,
                seq_len,
                args.num_heads,
                num_kv_heads,
                args.head_dim,
                args.dtype,
            )
            for seq_len in args.seq_lens
        )
    return (
        (batch_size, seq_len, 32, 32, head_dim, dtype_name)
        for dtype_name in _TABLE_DTYPES
        for head_dim in _TABLE_HEAD_DIMS
        for batch_size, seq_len in _TABLE_BATCH_SEQ_SHAPES
    )


def _validate_args(args):
    positive = (
        args.batch_size,
        args.num_heads,
        args.head_dim,
        args.warmup,
        args.rep,
        *args.seq_lens,
    )
    if args.num_kv_heads is not None:
        positive += (args.num_kv_heads,)
    if any(value <= 0 for value in positive):
        raise ValueError("all sizes, warmup, and rep must be positive")
    num_kv_heads = args.num_kv_heads or args.num_heads
    if args.num_heads % num_kv_heads != 0:
        raise ValueError("num-kv-heads must divide num-heads")
    if args.atol < 0 or args.rtol < 0:
        raise ValueError("atol and rtol must be non-negative")


def _run_isolated_shapes(args):
    script = str(Path(__file__).resolve())
    configs = tuple(_configs(args))

    for index, (
        batch_size,
        seq_len,
        num_query_heads,
        num_kv_heads,
        head_dim,
        dtype_name,
    ) in enumerate(configs, start=1):
        print(
            f"# isolated shape {index}/{len(configs)}: "
            f"B={batch_size}, T={seq_len}, HQ={num_query_heads}, "
            f"HKV={num_kv_heads}, D={head_dim}, dtype={dtype_name}",
            flush=True,
        )
        command = [
            sys.executable,
            script,
            "--backend",
            args.backend,
            "--custom-shapes",
            "--batch-size",
            str(batch_size),
            "--num-heads",
            str(num_query_heads),
            "--num-kv-heads",
            str(num_kv_heads),
            "--head-dim",
            str(head_dim),
            "--seq-lens",
            str(seq_len),
            "--dtype",
            dtype_name,
            "--warmup",
            str(args.warmup),
            "--rep",
            str(args.rep),
            "--atol",
            str(args.atol),
            "--rtol",
            str(args.rtol),
            "--seed",
            str(args.seed),
        ]
        if args.allow_triton_fallback:
            command.append("--allow-triton-fallback")
        if args.compare_triton:
            command.append("--compare-triton")
        command.append(
            "--check-reference" if args.check_reference else "--no-check-reference"
        )
        subprocess.run(command, check=True)


def _make_inputs(batch_size, num_query_heads, num_kv_heads, seq_len, head_dim):
    q_shape = (batch_size, num_query_heads, seq_len, head_dim)
    kv_shape = (batch_size, num_kv_heads, seq_len, head_dim)
    q = torch.randint(-16, 16, q_shape, device="gcu", dtype=torch.int8)
    k = torch.randint(-16, 16, kv_shape, device="gcu", dtype=torch.int8)
    v = torch.randn(kv_shape, device="gcu", dtype=torch.float16)
    q_scale = 0.005 + 0.015 * torch.rand(
        (batch_size, num_query_heads, triton.cdiv(seq_len, 128)),
        device="gcu",
        dtype=torch.float32,
    )
    k_scale = 0.005 + 0.015 * torch.rand(
        (batch_size, num_kv_heads, triton.cdiv(seq_len, 64)),
        device="gcu",
        dtype=torch.float32,
    )
    return q, k, v, q_scale, k_scale


def _cpu_reference(q, k, v, q_scale, k_scale):
    q = q.cpu().float()
    k = k.cpu().float()
    v = v.cpu().float()
    q_rows = q_scale.cpu().float().repeat_interleave(128, dim=-1)[
        ..., : q.shape[-2], None
    ]
    k_rows = k_scale.cpu().float().repeat_interleave(64, dim=-1)[
        ..., : k.shape[-2], None
    ]
    q = q * q_rows
    k = k * k_rows
    groups = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(groups, dim=1)
    v = v.repeat_interleave(groups, dim=1)
    logits = torch.matmul(q, k.transpose(-1, -2)) * _LN2
    return torch.matmul(torch.softmax(logits, dim=-1), v)


def _reference_gate(
    backend,
    num_query_heads,
    num_kv_heads,
    head_dim,
    output_dtype,
    atol,
    rtol,
):
    groups = num_query_heads // num_kv_heads
    gate_kv_heads = min(num_kv_heads, 2)
    gate_query_heads = gate_kv_heads * groups
    qo_len, kv_len = 129, 70
    q = torch.randint(
        -8,
        8,
        (1, gate_query_heads, qo_len, head_dim),
        device="gcu",
        dtype=torch.int8,
    )
    k = torch.randint(
        -8,
        8,
        (1, gate_kv_heads, kv_len, head_dim),
        device="gcu",
        dtype=torch.int8,
    )
    v = torch.randn(
        (1, gate_kv_heads, kv_len, head_dim),
        device="gcu",
        dtype=torch.float16,
    )
    q_scale = 0.005 + 0.015 * torch.rand(
        (1, gate_query_heads, triton.cdiv(qo_len, 128)),
        device="gcu",
        dtype=torch.float32,
    )
    k_scale = 0.005 + 0.015 * torch.rand(
        (1, gate_kv_heads, triton.cdiv(kv_len, 64)),
        device="gcu",
        dtype=torch.float32,
    )
    actual, _ = native_forward(
        q,
        k,
        v,
        q_scale,
        k_scale,
        output_dtype=output_dtype,
        backend=backend,
    )
    expected = _cpu_reference(q, k, v, q_scale, k_scale)
    actual_cpu = actual.cpu().float()
    max_abs = (actual_cpu - expected).abs().max().item()
    dtype_atol = max(atol, 4e-2) if output_dtype == torch.bfloat16 else atol
    dtype_rtol = max(rtol, 4e-2) if output_dtype == torch.bfloat16 else rtol
    correct = torch.allclose(
        actual_cpu, expected, atol=dtype_atol, rtol=dtype_rtol
    )
    return correct, max_abs


def _time(function, warmup, rep):
    output, _ = function()
    if not bool(torch.isfinite(output).all().item()):
        raise RuntimeError("forward produced non-finite output")
    return triton.testing.do_bench(function, warmup=warmup, rep=rep)


def benchmark(args):
    _validate_args(args)
    if args.isolate_shapes:
        _run_isolated_shapes(args)
        return

    torch.manual_seed(args.seed)
    selected_backend = resolve_backend(
        args.backend, return_lse=False, device="gcu"
    )
    if (
        args.backend == "auto"
        and selected_backend == "triton"
        and not args.allow_triton_fallback
    ):
        raise NativeBackendUnavailable(
            "auto resolved to Triton; install/enable torch-gcu SDPA or pass "
            "--allow-triton-fallback explicitly"
        )

    print(f"# native backend status: {native_backend_status()}")
    print(f"# requested backend: {args.backend}; resolved backend: {selected_backend}")
    print("# native latency includes dequantization and layout/output conversion")
    print(
        "provider\tbackend\tB\tT\tHQ\tHKV\tD\tdtype\tlatency_ms\t"
        "tflops\tspeedup_vs_torch\tstatus\tgate_max_abs"
    )

    gate_cache = {}
    for (
        batch_size,
        seq_len,
        num_query_heads,
        num_kv_heads,
        head_dim,
        dtype_name,
    ) in _configs(args):
        output_dtype = getattr(torch, dtype_name)
        gate_key = (
            selected_backend,
            num_query_heads // num_kv_heads,
            head_dim,
            dtype_name,
        )
        if args.check_reference and gate_key not in gate_cache:
            gate_cache[gate_key] = _reference_gate(
                selected_backend,
                num_query_heads,
                num_kv_heads,
                head_dim,
                output_dtype,
                args.atol,
                args.rtol,
            )
        correct, max_abs = gate_cache.get(gate_key, (True, ""))
        if not correct:
            print(
                f"native\t{selected_backend}\t{batch_size}\t{seq_len}\t"
                f"{num_query_heads}\t{num_kv_heads}\t{head_dim}\t{dtype_name}\t"
                f"\t\t\tINCORRECT\t{max_abs:.6f}"
            )
            continue

        q, k, v, q_scale, k_scale = _make_inputs(
            batch_size,
            num_query_heads,
            num_kv_heads,
            seq_len,
            head_dim,
        )

        def run_native():
            return native_forward(
                q,
                k,
                v,
                q_scale,
                k_scale,
                output_dtype=output_dtype,
                backend=selected_backend,
            )

        native_ms = _time(run_native, args.warmup, args.rep)
        operations = (
            4
            * batch_size
            * num_query_heads
            * seq_len
            * seq_len
            * head_dim
        )
        native_tflops = operations / native_ms * 1e-9

        # Torch is the comparison baseline. Print it first and keep its
        # normalized speedup fixed at 1.000x.
        max_abs_text = "" if max_abs == "" else f"{max_abs:.6f}"
        print(
            f"native\t{selected_backend}\t{batch_size}\t{seq_len}\t"
            f"{num_query_heads}\t{num_kv_heads}\t{head_dim}\t{dtype_name}\t"
            f"{native_ms:.4f}\t{native_tflops:.2f}\t1.000x\tPASS\t"
            f"{max_abs_text}"
        )

        triton_ms = None
        if args.compare_triton and selected_backend != "triton":

            def run_triton():
                return triton_forward(
                    q,
                    k,
                    v,
                    q_scale,
                    k_scale,
                    output_dtype=output_dtype,
                )

            triton_ms = _time(run_triton, args.warmup, args.rep)
            triton_tflops = operations / triton_ms * 1e-9
            speedup_vs_torch = native_ms / triton_ms
            print(
                f"triton\ttriton\t{batch_size}\t{seq_len}\t{num_query_heads}\t"
                f"{num_kv_heads}\t{head_dim}\t{dtype_name}\t{triton_ms:.4f}\t"
                f"{triton_tflops:.2f}\t{speedup_vs_torch:.3f}x\tPASS\t"
            )


@pytest.mark.sage_attention
def test_sage_attention_native_benchmark(benchmark_options) -> None:
    args = argparse.Namespace(
        backend=benchmark_options.get("backend", "auto"),
        allow_triton_fallback=benchmark_options.get(
            "allow_triton_fallback", False
        ),
        compare_triton=benchmark_options.get("compare_triton", False),
        custom_shapes=benchmark_options.get("custom_shapes", False),
        isolate_shapes=benchmark_options.get("isolate_shapes", False),
        batch_size=benchmark_options.get("batch_size", 1),
        num_heads=benchmark_options.get("num_heads", 32),
        num_kv_heads=benchmark_options.get("num_kv_heads"),
        head_dim=benchmark_options.get("head_dim", 128),
        seq_lens=benchmark_options.get(
            "seq_lens", (1024, 4096, 8192, 16384)
        ),
        dtype=benchmark_options.get("dtype", "float16"),
        warmup=benchmark_options.get("warmup", 5),
        rep=benchmark_options.get("rep", 20),
        check_reference=benchmark_options.get("check_reference", True),
        atol=benchmark_options.get("atol", 3e-2),
        rtol=benchmark_options.get("rtol", 3e-2),
        seed=benchmark_options.get("seed", 2026),
    )
    benchmark(args)


if __name__ == "__main__":
    benchmark(parse_args())
