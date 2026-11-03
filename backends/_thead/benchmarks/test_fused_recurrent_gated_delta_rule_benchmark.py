# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""T-Head recurrent GDN versus PLA; both consume precomputed g/beta.

Shapes come from the pinned PLA and upstream/SAIL vLLM tests below: nine
single-token decode shapes and six multi-token verification shapes (T=2/4/8).
T is per request, not total packed tokens. No T=16 or stride stress workloads.
The FP16/BF16 sweep, FP32 state, precomputed gates and Graph timing are our
comparison protocol, not a claim that every upstream test uses these dtypes.
CPU FP64 reference and input helper retain their pinned PR #1012 provenance.

PLA wraps the bundled k-last kernel with precomputed gate loads on BOTH decode
and MTP paths. exp(g), the official recurrence, Q/K normalization, state transfers
and LUT dispatch remain. This is an adaptation, not the original fused-gate API.
Compilation, metadata conversion, input preparation and reset are outside timing.
Each graph contains 64 sequential calls on the same state pool. Reset before
each replay and alternate providers within each case; report event duration / 64.
Both providers receive the same int32 state-index table. CPU FP64 checks the
full replay too.
Run: pytest -s <this-file> -m chunk_gated_delta_rule [-k 'bf16 and mtp']
Optional raw records: GDN_RESULTS=/path/to/results.jsonl pytest -s <this-file>
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import statistics
import sys

_BASELINE_ROOT = Path(__file__).resolve().parents[3]
_PROJECT_ROOT = _BASELINE_ROOT.parent
for _root in (_PROJECT_ROOT / "src", _PROJECT_ROOT, _BASELINE_ROOT, Path(__file__).parent):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import pytest
import torch
import triton

from flag_attn import runtime
from flag_attn.testing import backend as test_backend
from _recurrent_gdn_reference import assert_result, build, reference

try:
    from benchmark.recording import BenchmarkRecorder
except ModuleNotFoundError:
    from recording import BenchmarkRecorder

pytestmark = [pytest.mark.benchmark, pytest.mark.chunk_gated_delta_rule]
SOURCE = "official-pla-vllm-shapes-t1-t2-t4-t8"
PLA_SOURCE = "t-head/pla@ace8581717b080aafdd3c9604f582d35da8e6355"
_PLA_URL = "https://github.com/t-head/pla/blob/ace8581717b080aafdd3c9604f582d35da8e6355/"
_VLLM_URL = "https://github.com/vllm-project/vllm/blob/fb7def9ef9fa27fb794f655fb993a41e0b9bfe0c/"
_SAIL_URL = "https://github.com/vllm-project/vllm-sail/blob/95c24b3c21a132a4987f7daba99b69fccf7d6190/"
SHAPE_SOURCES = {
    "vllm_decode": (_VLLM_URL + "tests/kernels/fla/test_fused_sigmoid_gating_delta_rule.py",
                    "test_fused_sigmoid_gating_delta_rule_update_non_spec"),
    "pla_decode": (_PLA_URL + "scripts/decode/test_fused_sigmoid_gating_recurrent_k_last.py",
                   "main: default --batch=64, --tp=1, with/without --qwen35"),
    "pla_mtp": (_PLA_URL + "scripts/decode/test_fused_sigmoid_gating_recurrent_k_last_mtp.py",
                "test_all_configurations: GDN decode / causal T sweep / batch sweep"),
    "sail_test": (_SAIL_URL + "tests/e2e/gdn/test_gdn_pla_decode.py", "test_update_cuda_matches_triton"),
    "sail_perf": (_SAIL_URL + "tests/e2e/gdn/test_gdn_pla_decode.py", "test_update_perf"),
}
WARMUP_CALLS, SAMPLE_COUNT = 5, 25
GRAPH_CALLS = 64
# (id, request count, tokens per request, key heads, value heads, shape source).
# Copy existing combinations; do not form a Cartesian product of heads/batches.
PERF_CASES = [
    ("decode_b1_h16_hv32", 1, 1, 16, 32, "vllm_decode"),
    ("decode_b2_h16_hv32", 2, 1, 16, 32, "vllm_decode"),
    ("decode_b4_h16_hv32", 4, 1, 16, 32, "vllm_decode"),
    ("decode_b64_h16_hv32", 64, 1, 16, 32, "pla_decode"),
    ("decode_b64_h16_hv64", 64, 1, 16, 64, "pla_decode"),
    ("decode_b2_h4_hv8", 2, 1, 4, 8, "pla_mtp"),
    ("decode_b4_h4_hv8", 4, 1, 4, 8, "sail_test"),
    ("decode_b32_h4_hv8", 32, 1, 4, 8, "sail_perf"),
    ("decode_b128_h4_hv8", 128, 1, 4, 8, "sail_perf"),
    ("mtp_b2_t2", 2, 2, 4, 8, "pla_mtp"),
    ("mtp_b2_t4", 2, 4, 4, 8, "pla_mtp"),
    ("mtp_b2_t8", 2, 8, 4, 8, "pla_mtp"),
    ("mtp_b1_t8", 1, 8, 4, 8, "pla_mtp"),
    ("mtp_b4_t8", 4, 8, 4, 8, "pla_mtp"),
    ("mtp_b8_t8", 8, 8, 4, 8, "pla_mtp"),
]
DTYPES = (torch.float16, torch.bfloat16)


@lru_cache(maxsize=1)
def _implementations():
    if not test_backend.is_available() or runtime.device.vendor_name != "thead":
        pytest.skip("requires the T-Head accelerator")
    from flag_attn.runtime.backend._thead.FLA.gated_delta_rule.fused_recurrent import (
        fused_recurrent_gated_delta_rule_fwd,
    )
    name = "backends._thead.ops.FLA.gated_delta_rule.pla_gdn.precomputed.recurrent.pla_recurrent_precomputed"
    try:
        pla = importlib.import_module(name)
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Build PLA first: cd BaselineBenchmark/backends/_thead/ops/FLA/"
            "gated_delta_rule/pla_gdn/precomputed/recurrent && python setup.py build_ext --inplace"
        ) from exc
    return fused_recurrent_gated_delta_rule_fwd, pla


def _clone_state(state):
    result = torch.empty_strided(state.shape, state.stride(), dtype=state.dtype, device=state.device)
    return result.copy_(state)


def _check(call, data, pristine, expected, dtype, written):
    state = data["initial_state"]
    state.copy_(pristine)
    out, final = call()
    test_backend.device_fn.synchronize()
    assert out.dtype == dtype and final.dtype == pristine.dtype
    assert final.data_ptr() == state.data_ptr(), "in-place state alias changed"
    assert_result((out, final), expected, dtype)
    untouched = torch.ones(len(pristine), dtype=torch.bool, device=state.device)
    untouched[written] = False
    assert torch.equal(final[untouched], pristine[untouched]), "untouched state slots changed"


def _capture(call, state, pristine):
    device = test_backend.device_fn
    stream = device.Stream()
    stream.wait_stream(device.current_stream())
    with device.stream(stream):
        for _ in range(WARMUP_CALLS):
            state.copy_(pristine)
            call()
    device.current_stream().wait_stream(stream)
    device.synchronize()
    state.copy_(pristine)
    device.synchronize()
    graph = device.CUDAGraph()
    with device.graph(graph, stream=stream):
        for _ in range(GRAPH_CALLS):
            result = call()
    device.synchronize()

    def replay():
        graph.replay()
        return result

    return replay


def _replay_reference(data):
    # Compact the CPU state pool to slots used by this workload. The large
    # unused pool remains covered by the exact untouched-slot check.
    cpu = {key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
           for key, value in data.items()}
    slots = cpu["ssm_state_indices"].long().unique()
    remap = torch.zeros(len(cpu["initial_state"]), dtype=torch.int64)
    remap[slots] = torch.arange(1, len(slots) + 1)
    cpu["ssm_state_indices"] = remap[cpu["ssm_state_indices"].long()]
    original = cpu["initial_state"]
    cpu["initial_state"] = torch.cat((torch.zeros_like(original[:1]), original[slots]))
    for _ in range(GRAPH_CALLS):
        expected = reference(cpu)
        cpu["initial_state"] = expected[1]
    final = original.clone()
    final[slots] = expected[1][1:]
    return expected[0], final


def _measure_pair(replays, providers, pristine, first):
    device = test_backend.device_fn
    names = (first, "pla" if first == "flag" else "flag")
    for name in names:
        for _ in range(WARMUP_CALLS):
            providers[name]["initial_state"].copy_(pristine)
            replays[name]()
    device.synchronize()
    events = {name: [(device.Event(enable_timing=True), device.Event(enable_timing=True))
                     for _ in range(SAMPLE_COUNT)] for name in names}
    samples = {name: [] for name in names}
    for sample in range(SAMPLE_COUNT):
        order = names if sample % 2 == 0 else names[::-1]
        for name in order:
            start, end = events[name][sample]
            providers[name]["initial_state"].copy_(pristine)
            device.synchronize()
            start.record()
            replays[name]()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) / GRAPH_CALLS)
    for values in samples.values():
        assert len(values) == SAMPLE_COUNT
        assert all(math.isfinite(value) and value > 0 for value in values), values
    return samples


def _save(row, record_property):
    raw_path = os.environ.get("GDN_RESULTS")
    if raw_path:
        path = Path(raw_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
    recorder = BenchmarkRecorder(record_property, op_name="fused_recurrent_gated_delta_rule_fwd",
                                 dtype=row["dtype"], phase="forward", baseline="pla_precomputed_recurrent")
    metadata = {k: v for k, v in row.items() if k not in ("flag_ms", "pla_ms", "config")}
    recorder.add(shape_detail=row["config"], latency=row.get("flag_ms"),
                 latency_base=row.get("pla_ms"), latency_unit="ms", **metadata)
    recorder.record()


@pytest.fixture(scope="module")
def benchmark_table(request):
    _implementations()
    manager = request.config.pluginmanager
    reporter = manager.get_plugin("terminalreporter")
    prefix = request.node.nodeid + "::"
    class QuietCases:
        @pytest.hookimpl(tryfirst=True)
        def pytest_report_teststatus(self, report, config):
            if report.nodeid.startswith(prefix):
                if report.when == "call" and report.passed:
                    return "passed", "", ""
                if report.skipped:
                    return "skipped", "", ""
    quiet = QuietCases()
    manager.register(quiet)
    def write(text):
        with manager.get_plugin("capturemanager").global_and_fixture_disabled():
            if reporter is None:
                print(text, flush=True)
            else:
                reporter.write(text + "\n", flush=True)
    write(f"\nT-Head {torch.cuda.get_device_name()}; torch={torch.__version__}; triton={triton.__version__}")
    write(f"Precomputed g/beta and int32 state indices on BOTH sides; FP32 state; Graph: {GRAPH_CALLS} calls/replay.")
    write("5 warmups / 25 samples per provider; alternate order within each case; reset outside timing.")
    write("Median event time per call; capture/preparation excluded.")
    write(f"{'Case':>22} {'QKV':>5} {'B':>4} {'T/req':>5} {'H':>3} {'HV':>3} {'K':>3} {'V':>3} "
          f"{'Flag(us)':>11} {'PLA(us)':>11} {'PLA/Flag':>9} {'Status':>12}")
    write("-" * 121)
    def emit(row):
        c = row["config"]
        def us(key):
            value = row.get(key)
            return "N/A" if value is None else f"{value * 1000:.3f}"
        dtype = row["dtype"].replace("torch.", "").replace("bfloat16", "bf16").replace("float16", "fp16")
        ratio = row.get("speedup")
        speedup = "N/A" if ratio is None else f"{ratio:.3f}"
        write(f"{row['case']:>22} {dtype:>5} {len(c['lengths']):>4} {max(c['lengths']):>5} "
              f"{c['H']:>3} {c['HV']:>3} {c['K']:>3} {c['V']:>3} "
              f"{us('flag_ms'):>11} {us('pla_ms'):>11} {speedup:>9} {row['status']:>12}")
        if row.get("reason"):
            write("    " + row["reason"])
    try:
        yield emit
    finally:
        manager.unregister(quiet)


@pytest.mark.parametrize("case", PERF_CASES, ids=[case[0] for case in PERF_CASES])
@pytest.mark.parametrize("dtype", DTYPES, ids=("fp16", "bf16"))
@torch.inference_mode()
def test_fused_recurrent_gated_delta_rule_benchmark(case, dtype, record_property, benchmark_table):
    flag, pla = _implementations()
    name, batch, tokens, heads, value_heads, origin = case
    lengths = [tokens] * batch
    workload = (name, lengths, heads, value_heads, 128, 128,
                "spec" if tokens > 1 else "index", True, False)
    data = build(workload, dtype, test_backend.device)
    # The reference helper makes MTP tables int64; PLA requires int32. Give
    # both kernels one shared table so neither reads a different index dtype.
    data["ssm_state_indices"] = data["ssm_state_indices"].to(torch.int32)
    if tokens > 1:
        # Legal accepted positions vary across requests and never exceed T.
        data["num_accepted_tokens"] = (
            torch.arange(batch, dtype=torch.int32, device=test_backend.device) % tokens + 1
        )
    source_file = Path(inspect.getfile(flag))
    row = dict(
        case=name, mode="mtp" if tokens > 1 else "decode", dtype=str(dtype), backend="thead", graph=True,
        unit="ms", source=SOURCE, pla_source=PLA_SOURCE, pla_kind="precomputed_gates_adaptation",
        shape_source=dict(zip(("url", "entry"), SHAPE_SOURCES[origin])),
        gate_scope="precomputed_g_beta_recurrence_with_optional_qk_normalization",
        flag_source=str(source_file), flag_sha256=hashlib.sha256(source_file.read_bytes()).hexdigest(),
        pla_extension=pla.__file__, sample_count=SAMPLE_COUNT, warmup_calls=WARMUP_CALLS,
        protocol="graph-64-sequential-calls-reset-outside-5warmup-25paired-alternating-samples-median-per-call",
        graph_calls=GRAPH_CALLS, state_reset="before_each_replay", graph_state_policy="sequential_updates",
        flag_accuracy="NOT_RUN", pla_accuracy="NOT_RUN", status="NOT_RUN",
        config=dict(lengths=lengths, H=heads, HV=value_heads, K=128, V=128,
                    seed=2701, scale=data["scale"], qk_normalization=True, state_dtype="torch.float32",
                    tensors={k: dict(shape=list(v.shape), stride=list(v.stride()), dtype=str(v.dtype))
                             for k, v in data.items() if isinstance(v, torch.Tensor)}),
    )
    pristine = _clone_state(data["initial_state"])
    providers = {"flag": dict(data, initial_state=_clone_state(pristine))}
    calls = {"flag": lambda: flag(**providers["flag"])}
    pla_data = dict(data, initial_state=_clone_state(pristine))
    providers["pla"] = pla_data
    def run_pla():
        out = pla.run(pla_data["q"], pla_data["k"], pla_data["v"], pla_data["g"],
                      pla_data["beta"], pla_data["initial_state"], pla_data["ssm_state_indices"],
                      pla_data["cu_seqlens"], pla_data["num_accepted_tokens"],
                      pla_data["scale"], pla_data["use_qk_l2norm_in_kernel"], tokens)
        return out, pla_data["initial_state"]
    calls["pla"] = run_pla
    row["pla_config"] = pla.config(batch, heads, value_heads, tokens)
    row["state_index_dtype"] = "torch.int32 for both providers"
    indices = data["ssm_state_indices"]
    written = (indices[:, :tokens] if indices.ndim == 2 else indices).long().flatten()
    replays = {}
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        expected = reference(data)
        # Correctness for every supported provider comes BEFORE all timing.
        for name, call in calls.items():
            try:
                _check(call, providers[name], pristine, expected, dtype, written)
                row[name + "_accuracy"] = "PASS"
            except Exception as exc:
                row.update(status="FAIL", reason=f"{name} correctness failed: {exc}")
                row[name + "_accuracy"] = "FAIL"
                _save(row, record_property)
                benchmark_table(row)
                raise
        del expected
        expected = _replay_reference(data)
        # Captured state updates must match the same number of CPU recurrence
        # calls, including FP32 state writeback between calls.
        for name, call in calls.items():
            try:
                replay = _capture(call, providers[name]["initial_state"], pristine)
                _check(replay, providers[name], pristine, expected, dtype, written)
                replays[name] = replay
                row[name + "_graph_accuracy"] = "PASS"
            except Exception as exc:
                row.update(status="FAIL", reason=f"{name} graph correctness failed: {exc}")
                row[name + "_graph_accuracy"] = "FAIL"
                _save(row, record_property)
                benchmark_table(row)
                raise
    finally:
        torch.set_num_threads(threads)
    del expected
    first = "pla" if PERF_CASES.index(case) % 2 else "flag"
    row["measurement_order"] = "alternating within case, starting with " + first
    paired_samples = _measure_pair(replays, providers, pristine, first)
    for name, values in paired_samples.items():
        row[name + "_samples_ms"] = values
        row[name + "_ms"] = statistics.median(values)
        row[name + "_p20_ms"] = torch.quantile(torch.tensor(values), 0.2).item()
        row[name + "_p80_ms"] = torch.quantile(torch.tensor(values), 0.8).item()
    row["speedup"] = row["pla_ms"] / row["flag_ms"]
    row["status"] = "PASS"
    _save(row, record_property)
    benchmark_table(row)
