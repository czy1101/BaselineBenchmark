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

"""GDN comparison against native PLA, using PLA's official standard shapes.

Why the original FlagAttention shape set cannot be used with this PLA entry:
The client benchmark uses equal Q/K and V head counts, including 96/96,
16/16 and 64/64, and includes head dimensions 64 and 256. At the PLA version
pinned below, FlashQLA accepts only its listed (V heads, Q/K heads) pairs,
all with more V heads than Q/K heads, and requires K=V=128. Its Python API
rejects the client configurations; the native dispatcher has no matching
entries and aborts on unsupported pairs. Removing the Python check does not
provide a usable native baseline. This is this implementation's coverage
limit, not a restriction of the GDN algorithm.

Official source links (pinned to the PLA checkout used for this comparison):
- Supported head pairs and head-dimension checks:
  https://github.com/t-head/pla/blob/ace8581717b080aafdd3c9604f582d35da8e6355/pla/prefill/flashqla/ops.py#L60
- Native dispatch entries and unsupported-pair failure:
  https://github.com/t-head/pla/blob/ace8581717b080aafdd3c9604f582d35da8e6355/src/prefill/flashqla/kernels/fused_fwd_cute.cu#L1851

Where the selected shape set comes from:
The 13 GDN_CASES are copied verbatim, in official order, from PLA's prefill
benchmark collector. Each entry is (V heads, Q/K heads, batch, sequence length).
Both FP16 and BF16 are measured. The output table uses (B,T,H,HV,K,V), where
H is the Q/K head count and HV is the V head count.
- Official 13-case definition:
  https://github.com/t-head/pla/blob/ace8581717b080aafdd3c9604f582d35da8e6355/scripts/prefill/collect_bench_vs_fla.py#L77
- Official collector consuming these cases:
  https://github.com/t-head/pla/blob/ace8581717b080aafdd3c9604f582d35da8e6355/scripts/prefill/collect_bench_vs_fla.py#L438
- Official native-PLA versus Triton performance entry:
  https://github.com/t-head/pla/blob/ace8581717b080aafdd3c9604f582d35da8e6355/scripts/prefill/flashqla/bench_vs_fla.py#L108

Scope of this copy:
The client file benchmark/test_gated_delta_rule_benchmark.py stays unchanged.
Only its reference provider and shape set are replaced here, with the needed
separate V-head axis. Input distributions, seed=42, the 1% accuracy gate,
no incoming state, no Q/K normalization, and final-state output follow the
client benchmark. Timing also follows the client: do_bench warmup=10 ms,
rep=50 ms, median. This is NOT the complete official indexed-state-pool
benchmark recipe. PLA's packed-batch and FP32-gate conversions stay inside
the timed reference call. FP16 and BF16 are printed as separate tables.
"""


import os
import sys
from importlib import import_module
from pathlib import Path
from contextlib import contextmanager, nullcontext
from typing import Callable

import pytest
import torch
import triton

# Copied from benchmark/test_gated_delta_rule_benchmark.py.
# Use the official PLA GDN shapes; retain the client input distributions and timer.
_BASELINE_ROOT = Path(__file__).resolve().parents[3]
_PROJECT_ROOT = _BASELINE_ROOT.parent
for _import_root in (_PROJECT_ROOT / "src", _PROJECT_ROOT, _BASELINE_ROOT):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

try:
    from benchmark.recording import BenchmarkRecorder
except ModuleNotFoundError:  # Direct script execution.
    from recording import BenchmarkRecorder

from flag_attn.testing import backend as test_backend
from flag_attn import chunk_gated_delta_rule, runtime
from flag_attn.utils import has_triton_tle


ASSERT_RATIO = 0.01
WARMUP = 10
REPETITIONS = 50
FULL_TLE_ENV = "FLAG_ATTN_CHUNK_GATED_DELTA_RULE_TLE"
RECOMPUTE_TLE_ENV = "FLAG_ATTN_CHUNK_GDR_RECOMPUTE_TLE"
TWO_KERNEL_TLE_ENV = "FLAG_ATTN_CHUNK_GDR_TWO_KERNEL_TLE"
HYBRID_TLE_ENV = "FLAG_ATTN_CHUNK_GDR_HYBRID_TLE"

# Official PLA GDN standard set: (value heads, Q/K heads, batch, length).
# Source: /workspace/pla/scripts/prefill/collect_bench_vs_fla.py, GDN_CASES.
GDN_CASES = [
    (16, 8, 1, 2048),
    (16, 8, 1, 4096),
    (16, 8, 1, 8192),
    (16, 8, 4, 2048),
    (16, 8, 4, 4096),
    (16, 8, 8, 2048),
    (32, 16, 1, 2048),
    (32, 16, 1, 4096),
    (32, 16, 1, 8192),
    (32, 16, 1, 16384),
    (32, 16, 4, 2048),
    (32, 16, 4, 4096),
    (32, 16, 8, 2048),
]
# Benchmark table/input order: (B, T, Q/K heads, V heads, K, V).
SHAPES = [(B, T, H, HV, 128, 128) for HV, H, B, T in GDN_CASES]
DTYPES = (torch.float16, torch.bfloat16)


def _load_pla_reference():
    try:
        package = "backends._thead.ops.FLA.gated_delta_rule.pla_gdn"
        import_module(package + ".official.pla.prefill.flashqla._C")
        ops = import_module(package + ".official.pla.prefill.flashqla.ops")
        return ops.chunk_gated_delta_rule_fwd, ops.SUPPORTED_HEAD_CONFIGS, None
    except (ImportError, OSError) as exc:
        return None, frozenset(), f"{type(exc).__name__}: {exc}"


PLA_CHUNK_GDN, PLA_SUPPORTED_HEAD_CONFIGS, PLA_IMPORT_ERROR = _load_pla_reference()


def _pla_unavailable_reason(shape: tuple[int, int, int, int, int, int]) -> str | None:
    if runtime.device.vendor_name != "thead":
        return "PLA requires a T-Head accelerator"
    if PLA_CHUNK_GDN is None:
        return f"PLA import failed: {PLA_IMPORT_ERROR}"
    _, _, H, HV, K, V = shape
    reasons = []
    if K != 128 or V != 128:
        reasons.append(f"requires K=V=128; got K={K}, V={V}")
    if (HV, H) not in PLA_SUPPORTED_HEAD_CONFIGS:
        reasons.append(f"unsupported (value heads, Q/K heads)=({HV}, {H})")
    return "; ".join(reasons) or None


def _cuda_tle_available() -> bool:
    return test_backend.is_nvidia() and has_triton_tle(3, 6, 0)


def _require_pla_reference():
    if PLA_CHUNK_GDN is None:
        raise RuntimeError(
            f"official PLA GDN implementation is unavailable: {PLA_IMPORT_ERROR}"
        )
    return PLA_CHUNK_GDN


@contextmanager
def _use_optimized_tle():
    names = (
        FULL_TLE_ENV,
        RECOMPUTE_TLE_ENV,
        TWO_KERNEL_TLE_ENV,
        HYBRID_TLE_ENV,
    )
    old_values = {name: os.environ.get(name) for name in names}
    os.environ[FULL_TLE_ENV] = "1"
    os.environ[RECOMPUTE_TLE_ENV] = "0"
    os.environ[TWO_KERNEL_TLE_ENV] = "1"
    os.environ[HYBRID_TLE_ENV] = "1"
    try:
        yield
    finally:
        for name, value in old_values.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _make_inputs(shape: tuple[int, int, int, int, int, int], dtype: torch.dtype):
    B, T, H, HV, K, V = shape
    q = torch.randn(B, T, H, K, device=test_backend.device, dtype=dtype) / (K**0.5)
    k = torch.randn(B, T, H, K, device=test_backend.device, dtype=dtype) / (K**0.5)
    v = torch.randn(B, T, HV, V, device=test_backend.device, dtype=dtype)
    g = (-torch.rand(B, T, HV, device=test_backend.device, dtype=torch.float32) * 0.1).to(dtype)
    beta = torch.rand(B, T, HV, device=test_backend.device, dtype=dtype).sigmoid()
    return q, k, v, g, beta, K**-0.5


def _call_pla_reference(inputs):
    """Adapt PLA's representation, keeping every conversion inside timing.

    Cast the existing gate values to FP32; do not regenerate them or add
    Q/K normalization, an incoming state or a state pool.
    """
    pla_chunk_gdn = _require_pla_reference()
    q, k, v, g, beta, scale = inputs
    B, T, Hg, K = q.shape
    H, V = v.shape[2:]
    if K != 128 or V != 128 or (H, Hg) not in PLA_SUPPORTED_HEAD_CONFIGS:
        raise ValueError(
            f"PLA does not support K={K}, V={V}, value_heads={H}, qk_heads={Hg}"
        )
    # Preserve sequence boundaries when viewing dense batches as packed tokens.
    cu_seqlens = None
    if B > 1:
        cu_seqlens = torch.arange(
            0, (B + 1) * T, T, device=q.device, dtype=torch.int32,
        )
    _, _, out, _, final_state = pla_chunk_gdn(
        q=q.reshape(1, B * T, Hg, K),
        k=k.reshape(1, B * T, Hg, K),
        v=v.reshape(1, B * T, H, V),
        g=g.to(torch.float32).reshape(1, B * T, H),
        beta=beta.to(torch.float32).reshape(1, B * T, H),
        scale=scale,
        initial_state=None,
        cu_seqlens=cu_seqlens,
        output_final_state=True,
        normalize_qk=False,
    )
    return out.reshape(B, T, H, V), final_state


def _call_flag_attn(inputs):
    q, k, v, g, beta, scale = inputs
    return chunk_gated_delta_rule(
        q,
        k,
        v,
        beta,
        g,
        head_first=False,
        scale=scale,
        output_final_state=True,
    )


def _err_ratio(expected: torch.Tensor, actual: torch.Tensor) -> float:
    error = (expected.float() - actual.float()).flatten().square().mean().sqrt().item()
    baseline = expected.float().flatten().square().mean().sqrt().item()
    return error / (baseline + 1e-8)


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    actual = actual.float()
    expected = expected.float()
    absolute_error = (actual - expected).abs().max().item()
    if absolute_error <= 1e-6:
        return
    assert not torch.isnan(actual).any(), f"{name}: NaN detected in actual"
    assert not torch.isnan(expected).any(), f"{name}: NaN detected in baseline"
    ratio = _err_ratio(expected, actual)
    assert ratio < ASSERT_RATIO, (
        f"{name} diff: abs={absolute_error:.6f} ratio={ratio:.6f} "
        f"limit={ASSERT_RATIO}"
    )


@torch.inference_mode()
def _benchmark_case(dtype: torch.dtype, shape: tuple[int, int, int, int, int, int]):
    torch.manual_seed(42)
    inputs = _make_inputs(shape, dtype)
    compare_pla = _pla_unavailable_reason(shape) is None
    pla_ms = None
    if compare_pla:
        expected_o, expected_final_state = _call_pla_reference(inputs)
        pla_ms = test_backend.do_bench(lambda: _call_pla_reference(inputs), warmup=WARMUP, rep=REPETITIONS, return_mode="median")
    with _use_optimized_tle() if _cuda_tle_available() else nullcontext():
        actual_o, actual_final_state = _call_flag_attn(inputs)
        if compare_pla:
            _assert_close("o", actual_o, expected_o)
            _assert_close("final_state", actual_final_state, expected_final_state)
        else:
            assert torch.isfinite(actual_o).all()
            assert torch.isfinite(actual_final_state).all()
        flag_attn_ms = test_backend.do_bench(lambda: _call_flag_attn(inputs), warmup=WARMUP, rep=REPETITIONS, return_mode="median")
    return pla_ms, flag_attn_ms


def _shape_name(shape: tuple[int, int, int, int, int, int]) -> str:
    return f"B{shape[0]}_T{shape[1]}_H{shape[2]}_HV{shape[3]}_K{shape[4]}_V{shape[5]}"


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _speedup(pla_ms: float | None, flag_attn_ms: float):
    return pla_ms / flag_attn_ms if pla_ms is not None and flag_attn_ms > 0 else None


def _print_header(dtype: torch.dtype) -> None:
    print(f"\n{'=' * 92}")
    dtype_label = "FP16" if dtype == torch.float16 else "BF16"
    print(f"  chunk_gated_delta_rule benchmark - FWD ONLY - {dtype_label}")
    print(
        "  provider: pla_chunk_gated_delta_rule vs "
        "flag_attn_chunk_gated_delta_rule"
    )
    print(f"  warmup={WARMUP}ms  rep={REPETITIONS}ms")
    print(f"{'=' * 92}")
    print(
        f"{'B':>3} {'T':>6} {'H':>4} {'HV':>4} {'K':>4} {'V':>4} {'dtype':>8} "
        f"{'pla(ms)':>10} {'flag_attn(ms)':>14} {'pla/flag_attn':>14}"
    )


def _print_row(dtype, shape, pla_ms, flag_attn_ms) -> None:
    B, T, H, HV, K, V = shape
    baseline = "N/A" if pla_ms is None else f"{pla_ms:.3f}"
    ratio = _speedup(pla_ms, flag_attn_ms)
    speedup = "N/A" if ratio is None else f"{ratio:.2f}x"
    print(f"{B:>3} {T:>6} {H:>4} {HV:>4} {K:>4} {V:>4} {_dtype_name(dtype):>8} {baseline:>10} {flag_attn_ms:>14.3f} {speedup:>14}")


def _print_footer(dtype: torch.dtype) -> None:
    print(f"\n{'=' * 92}")
    print(f"  Completed {_dtype_name(dtype)} table.")
    print(f"{'=' * 92}\n")


def _record_result(
    record_property: Callable[[str, object], None] | None,
    dtype: torch.dtype,
    shape: tuple[int, int, int, int, int, int],
    pla_ms: float,
    flag_attn_ms: float,
) -> None:
    if record_property is None:
        return
    prefix = f"{_shape_name(shape)}_{_dtype_name(dtype)}"
    record_property(f"{prefix}.pla_unavailable_reason", _pla_unavailable_reason(shape))
    record_property(f"{prefix}.pla_ms", pla_ms)
    record_property(f"{prefix}.flagattention_ms", flag_attn_ms)
    record_property(f"{prefix}.speedup_vs_pla", _speedup(pla_ms, flag_attn_ms))


@pytest.mark.chunk_gated_delta_rule
@pytest.mark.skipif(not test_backend.supports_operator("chunk_gated_delta_rule"), reason="requires an available accelerator")
def test_chunk_gated_delta_rule_benchmark(
    record_property: Callable[[str, object], None],
) -> None:
    run_benchmark(record_property)


def run_benchmark(
    record_property: Callable[[str, object], None] | None = None,
) -> None:
    if not test_backend.is_available():
        raise RuntimeError("GDN benchmark requires an available accelerator")
    if runtime.device.vendor_name != "thead":
        raise RuntimeError("the PLA comparison benchmark requires T-Head")
    _require_pla_reference()
    compare_pla = any(_pla_unavailable_reason(shape) is None for shape in SHAPES)

    if record_property is not None:
        record_property("scope", "operator_only")
        record_property("baseline", "PLA")

    for dtype in DTYPES:
        _print_header(dtype)
        recorder = BenchmarkRecorder(
            record_property,
            op_name="chunk_gated_delta_rule",
            dtype=str(dtype),
            baseline="PLA",
            phase="forward",
            reference_available=compare_pla,
            reference_unavailable_reason=(
                None if compare_pla else "PLA does not support the selected official shapes"
            ),
        )
        for shape in SHAPES:
            pla_ms, flag_attn_ms = _benchmark_case(dtype, shape)
            _print_row(dtype, shape, pla_ms, flag_attn_ms)
            reason = _pla_unavailable_reason(shape)
            if reason is not None:
                print(f"  PLA unavailable: {reason}")
            recorder.add(
                shape_detail=shape, latency_base=pla_ms, latency=flag_attn_ms,
                reference_available=reason is None,
                reference_unavailable_reason=reason,
            )
            _record_result(
                record_property,
                dtype,
                shape,
                pla_ms,
                flag_attn_ms,
            )
            test_backend.device_fn.empty_cache()
        recorder.record()
        _print_footer(dtype)
    if not compare_pla:
        print("\nPLA comparison coverage: 0 supported selected shapes; no speedup is available.")
