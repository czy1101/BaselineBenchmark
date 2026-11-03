# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""T-Head packed GDN benchmark based on T-Head's official PLA script.

Reference repository: https://github.com/t-head/pla
Reference file: scripts/decode/test_fused_sigmoid_gating_recurrent_k_last_packed.py
Pinned commit: 3065b5a7238b5023f92bbeda534fca8bde3e3675
Exact source:
https://github.com/t-head/pla/blob/3065b5a7238b5023f92bbeda534fca8bde3e3675/scripts/decode/test_fused_sigmoid_gating_recurrent_k_last_packed.py

PLA official benchmark location in the reference file:
- bench_shape(), line 261: actual Triton packed / PLA packed calls and timing.
  https://github.com/t-head/pla/blob/3065b5a7238b5023f92bbeda534fca8bde3e3675/scripts/decode/test_fused_sigmoid_gating_recurrent_k_last_packed.py#L261
- run_benchmark(), line 592: batch-size sweep and result table.
  https://github.com/t-head/pla/blob/3065b5a7238b5023f92bbeda534fca8bde3e3675/scripts/decode/test_fused_sigmoid_gating_recurrent_k_last_packed.py#L592
- main(), line 628: --benchmark flag and model/TP configuration.
  https://github.com/t-head/pla/blob/3065b5a7238b5023f92bbeda534fca8bde3e3675/scripts/decode/test_fused_sigmoid_gating_recurrent_k_last_packed.py#L628

Reference sections:
- make_inputs(): seed=42, random packed QKV/gates, FP32 A_log, state *= 0.1.
- main()/run_benchmark(): default and --qwen35 presets, TP=4, K=V=128,
  batch limit=129 (listed batches 1 through 256), BF16 activations.
- bench_shape(): 10 alternating warmup calls; triton.testing.do_bench with
  warmup=50 ms, rep=200 ms, quantiles=[0.5, 0.2, 0.8].
- check_correctness(): activation-dtype tolerances for output and state.

Local adaptations: retain FlagAttention versus bundled PLA, independent CPU
reference, independent provider states, and the existing terminal table.
FP32 state follows the official benchmark; BF16 state adds a matching table.
The local batch sweep additionally includes B=512 (official --batch=256).
State slots start at one to match FlagAttention's vLLM null-slot convention.
"""

# Resolve the enclosing checkout so this copy runs from any working directory.
import sys
from pathlib import Path

_BASELINE_ROOT = Path(__file__).resolve().parents[3]
_PROJECT_ROOT = _BASELINE_ROOT.parent
for _import_root in (_PROJECT_ROOT / "src", _PROJECT_ROOT, _BASELINE_ROOT):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))


from functools import lru_cache
from importlib import import_module
import math
import time
import warnings

import pytest
import torch
import triton

from flag_attn import runtime
from flag_attn.testing import backend as test_backend

try:
    from benchmark.recording import BenchmarkRecorder
except ModuleNotFoundError:
    from recording import BenchmarkRecorder

pytestmark = pytest.mark.chunk_gated_delta_rule

# Full official batch list, equivalent to --batch=256, including B=512.
BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
STATE_DTYPES = (torch.float32, torch.bfloat16)
# Default and --qwen35 presets, each with the official default --tp=4.
HEAD_CONFIGS = ((4, 4, 8), (4, 4, 16))
HEAD_DIM = 128
DTYPE = torch.bfloat16
SEED = 42
INITIAL_WARMUP = 10
WARMUP_MS, REP_MS = 50, 200
QUANTILES = (0.5, 0.2, 0.8)
PLA_IMPORT_ERROR = None

def _assert_close(name, actual, expected):
    # PLA selects tolerance from activation dtype for BOTH output and state.
    atol, rtol = (1e-4, 1e-4) if DTYPE == torch.float32 else (2e-2, 1e-2)
    actual, expected = actual.detach().float().cpu(), expected.detach().float().cpu()
    torch.testing.assert_close(
        actual, expected, atol=atol, rtol=rtol,
        msg=lambda message: f"{name}: {message}",
    )


def _measure(call):
    try:
        values = triton.testing.do_bench(
            call, quantiles=list(QUANTILES), warmup=WARMUP_MS, rep=REP_MS,
        )
        timing = "triton_do_bench_device_event"
    except (RuntimeError, NotImplementedError, AttributeError) as error:
        # Match the official fallback while making its different scope visible.
        warnings.warn(f"do_bench unavailable; using official 200-call wall timing: {error}", stacklevel=2)
        test_backend.device_fn.synchronize()
        start = time.perf_counter()
        for _ in range(200):
            call()
        test_backend.device_fn.synchronize()
        values = [(time.perf_counter() - start) / 200 * 1000, None, None]
        timing = "synchronized_wall_mean_200_calls"
    assert math.isfinite(values[0]) and values[0] > 0, values
    return [value * 1000 if value is not None else None for value in values], timing

def _shape_matrix():
    return [
        (tp, batch, h, hv, state_dtype)
        for state_dtype in STATE_DTYPES
        for tp, h, hv in HEAD_CONFIGS
        for batch in BATCHES
    ]


@lru_cache(maxsize=1)
def _load_implementations():
    global PLA_IMPORT_ERROR
    if not test_backend.is_available() or runtime.device.vendor_name != "thead":
        pytest.skip("requires the T-Head accelerator")
    from flag_attn.runtime.backend._thead.FLA.gated_delta_rule.fused_recurrent_packed_decode import (
        fused_recurrent_gated_delta_rule_packed_decode,
    )

    try:
        import_module("backends._thead.ops.FLA.gated_delta_rule.pla_gdn.official.pla.decode._C")
        from backends._thead.ops.FLA.gated_delta_rule.pla_gdn import packed_decode
    except (ImportError, OSError) as exc:
        PLA_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
        return fused_recurrent_gated_delta_rule_packed_decode, None
    PLA_IMPORT_ERROR = None
    return fused_recurrent_gated_delta_rule_packed_decode, packed_decode


def _make_inputs(batch, h, hv, state_dtype):
    # Match PLA's tensor generation order, dtypes, distribution and seed.
    torch.manual_seed(SEED)
    device = test_backend.device
    mixed = torch.randn(batch, (2 * h + hv) * HEAD_DIM, device=device, dtype=DTYPE)
    b = torch.randn(batch, hv, device=device, dtype=DTYPE)
    alog = torch.randn(hv, device=device, dtype=torch.float32)
    a = torch.randn(batch, hv, device=device, dtype=DTYPE)
    bias = torch.randn(hv, device=device, dtype=DTYPE)
    initial = torch.randn(
        max(512, batch + 16), hv, HEAD_DIM, HEAD_DIM, device=device, dtype=state_dtype,
    ) * 0.1
    return dict(
        mixed_qkv=mixed, a=a, b=b, A_log=alog, dt_bias=bias,
        initial_state=initial,
        ssm_state_indices=torch.arange(1, batch + 1, device=device, dtype=torch.int32),
    )

def _reference(data, h, hv):
    """Independent CPU FP32 packed decoding, including gates and one update."""
    mixed = data["mixed_qkv"].detach().float().cpu()
    q, k, v = torch.split(mixed, (h * HEAD_DIM, h * HEAD_DIM, hv * HEAD_DIM), dim=-1)
    batch = mixed.shape[0]
    q, k = (x.reshape(batch, h, HEAD_DIM).repeat_interleave(hv // h, dim=1) for x in (q, k))
    v = v.reshape(batch, hv, HEAD_DIM)
    q = q / torch.sqrt(q.square().sum(-1, keepdim=True) + 1e-6)
    k = k / torch.sqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    a, b, alog, bias = (data[name].detach().float().cpu() for name in ("a", "b", "A_log", "dt_bias"))
    g = -alog.exp() * torch.nn.functional.softplus(a + bias, beta=1.0, threshold=20.0)
    beta = b.sigmoid()  # Keep beta FP32 through the recurrence.
    indices = data["ssm_state_indices"].long()
    initial = data["initial_state"][indices].detach().float().cpu()
    state = initial * g.exp()[:, :, None, None]
    residual = (v - (state * k[:, :, None, :]).sum(-1)) * beta[:, :, None]
    state = state + residual[:, :, :, None] * k[:, :, None, :]
    out = (state * (q * HEAD_DIM ** -0.5)[:, :, None, :]).sum(-1)
    return out.unsqueeze(1), state


@torch.inference_mode()
def run_benchmark(record_property, shape):
    tp, batch, h, hv, state_dtype = shape
    flag, pla = _load_implementations()
    data = _make_inputs(batch, h, hv, state_dtype)
    initial = data["initial_state"]
    states = (initial.clone(), initial.clone())
    outputs = tuple(torch.empty(batch, 1, hv, HEAD_DIM, device=initial.device, dtype=DTYPE) for _ in states)
    pla_bias = data["dt_bias"]
    cu = torch.arange(batch + 1, device=initial.device, dtype=torch.int32)

    def run_flag():
        return flag(
            data["mixed_qkv"],
            data["a"],
            data["b"],
            data["A_log"],
            data["dt_bias"],
            HEAD_DIM ** -0.5,
            states[0],
            outputs[0],
            data["ssm_state_indices"],
            True,
        )[0]

    def run_pla():
        # is_kda=False, is_sglang=False: slot zero is invalid on both sides.
        return pla(
            data["mixed_qkv"],
            data["a"],
            data["b"],
            data["A_log"],
            pla_bias,
            1.0,
            20.0,
            HEAD_DIM ** -0.5,
            states[1],
            data["ssm_state_indices"],
            outputs[1],
            True,
            cu,
            False,
            False,
        )

    calls = (run_flag, run_pla) if pla is not None else (run_flag,)
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        expected, expected_state = _reference(data, h, hv)
        indices = data["ssm_state_indices"].long()
        for name, call, state, output in zip(("FlagAttention", "PLA packed"), calls, states, outputs):
            actual = call()
            test_backend.device_fn.synchronize()
            _assert_close(f"{name} output", actual, expected)
            _assert_close(f"{name} state", state[indices], expected_state)
            assert torch.equal(state[0], initial[0]), f"{name}: null state slot changed"
            assert actual.data_ptr() == output.data_ptr()
    finally:
        torch.set_num_threads(threads)
    del expected, expected_state

    # Same ten alternating warmup calls as PLA, using independent provider states.
    for state in states:
        state.copy_(initial)
    for _ in range(INITIAL_WARMUP):
        for call in calls:
            call()
    test_backend.device_fn.synchronize()
    measurements = []
    methods = []
    for call, state in zip(calls, states):
        state.copy_(initial)
        values, method = _measure(call)
        measurements.append(values)
        methods.append(method)
    flag_us = measurements[0][0]
    pla_us = measurements[1][0] if pla is not None else None
    if pla is not None:
        assert methods[0] == methods[1], "providers used different timing methods"

    recorder = BenchmarkRecorder(
        record_property, op_name="fused_recurrent_gated_delta_rule_packed_decode",
        dtype=str(DTYPE), phase="forward",
        baseline="pla_official_packed" if pla is not None else None,
    )
    recorder.add(
        shape_detail=dict(
            TP=tp, B=batch, T=1, H=h, HV=hv, K=HEAD_DIM, V=HEAD_DIM,
            qkv_dtype=str(DTYPE), state_dtype=str(state_dtype),
            mixed_qkv_shape=list(data["mixed_qkv"].shape), state_shape=list(initial.shape),
            qk_normalization=True, scale=HEAD_DIM ** -0.5, seed=SEED,
        ),
        latency=flag_us, latency_base=pla_us, latency_unit="us",
        initial_warmup=INITIAL_WARMUP, warmup_ms=WARMUP_MS, rep_ms=REP_MS,
        quantiles=list(QUANTILES), quantiles_us=measurements[0],
        pla_quantiles_us=measurements[1] if pla is not None else [],
        execution_kind="eager", timing_method=methods[0],
        latency_scope="packed_qkv_gates_recurrence",
        cache_protocol="triton_driver_clear_cache" if methods[0].startswith("triton") else "repeated_no_flush",
        correctness="cpu_fp32_single_step_pla_activation_dtype_tolerances",
        pla_available=pla is not None, pla_unavailable_reason=PLA_IMPORT_ERROR,
        pla_entrypoint=f"{pla.__module__}.{pla.__name__}" if pla is not None else None,
        pla_dispatch="official_compiled_lut_or_default" if pla is not None else None,
        source="t-head/pla@3065b5a7238b5023f92bbeda534fca8bde3e3675",
        shape_selection="official_default_and_qwen35_tp4_batch256_fp32_and_bf16_states",
    )
    recorder.record()
    return (*shape, flag_us, pla_us)



@pytest.fixture(scope="module")
def benchmark_table(request):
    prefix = request.node.nodeid + "::"

    class QuietPassedCases:
        @pytest.hookimpl(tryfirst=True)
        def pytest_report_teststatus(self, report, config):
            if report.nodeid.startswith(prefix) and report.when == "call" and report.passed:
                return "passed", "", ""

    plugin = QuietPassedCases()
    manager = request.config.pluginmanager
    manager.register(plugin)
    reporter = manager.get_plugin("terminalreporter")

    def write(line):
        with manager.get_plugin("capturemanager").global_and_fixture_disabled():
            if reporter is not None:
                reporter.write(line + "\n", flush=True)
            else:
                print(line, flush=True)

    emitted_dtype = None

    def emit(row):
        tp, batch, h, hv, state_dtype, flag_us, pla_us = row
        nonlocal emitted_dtype
        if state_dtype != emitted_dtype:
            emitted_dtype = state_dtype
            write("")
            write("State: BF16" if state_dtype == torch.bfloat16 else "State: FP32")
            write(
                f"{'TP':>2} {'B':>3} {'H':>2} {'HV':>2} {'K':>3} {'V':>3} "
                f"{'QKV':>5} {'State':>5} {'flag_attn(us)':>14} {'pla(us)':>11} {'pla/flag_attn':>14}"
            )
            write("-" * 93)
        state = "BF16" if state_dtype == torch.bfloat16 else "FP32"
        pla_text = "N/A" if pla_us is None else f"{pla_us:.3f}"
        ratio_text = "N/A" if pla_us is None else f"{pla_us / flag_us:.3f}"
        write(
            f"{tp:>2} {batch:>3} {h:>2} {hv:>2} {HEAD_DIM:>3} {HEAD_DIM:>3} "
            f"{'BF16':>5} {state:>5} {flag_us:>14.3f} {pla_text:>11} {ratio_text:>14}"
        )

    try:
        yield emit
    finally:
        manager.unregister(plugin)


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "shape",
    _shape_matrix(),
    ids=[
        f"TP{tp}-B{batch}-H{h}-HV{hv}-state-{'bf16' if dtype == torch.bfloat16 else 'fp32'}"
        for tp, batch, h, hv, dtype in _shape_matrix()
    ],
)
def test_fused_recurrent_packed_decode_benchmark(record_property, shape, benchmark_table):
    benchmark_table(run_benchmark(record_property, shape))
