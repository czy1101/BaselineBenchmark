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

import contextlib
import io
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from backends._enflame.benchmarks import _parallax_timing as timing
from backends._enflame.benchmarks._parallax_table import render_tables


pytestmark = pytest.mark.parallax


class FakeGCU:
    def __init__(self):
        self.synchronizations = 0

    def synchronize(self):
        self.synchronizations += 1


def test_speedup_uses_baseline_as_numerator():
    result = timing.summarize_pair([4, 8, 12], [2, 2, 2])
    assert result["baseline_ms"] == 8
    assert result["triton_ms"] == 2
    assert result["speedup"] == 4
    assert result["baseline_mad_percent"] == 50
    assert result["triton_mad_percent"] == 0
    assert timing.summarize_pair([1], [2])["speedup"] == 0.5


def test_bad_latency_never_produces_speedup():
    for bad in ([], [0], [-1], [float("nan")], [float("inf")]):
        with pytest.raises(ValueError):
            timing.summarize_pair(bad, [1])
        with pytest.raises(ValueError):
            timing.summarize_pair([1], bad)


def test_invalid_timing_budget_is_rejected_before_launch():
    for settings in ((-1, 1, 1, "eager"), (0, 0, 1, "eager"), (0, 1, 0, "eager"), (0, 1, 1, "other")):
        with pytest.raises(ValueError):
            timing.validate_timing(*settings)


def test_balanced_sampling_and_equal_budget():
    gcu = FakeGCU()
    launches, samples = [], []

    def baseline():
        launches.append("A")

    def triton():
        launches.append("B")

    def sample(gcu, function, inner, mode):
        assert inner == 3 and mode == "eager"
        samples.append("A" if function is baseline else "B")
        return 8 if function is baseline else 2

    with patch.object(timing, "_sample_ms", side_effect=sample):
        result = timing.measure_pair(gcu, baseline, triton, warmup=2, iterations=2, inner=3, timing="eager")
    assert launches == ["A", "B", "A", "B", "B", "A"]
    assert samples == ["A", "B", "B", "A", "B", "A", "A", "B"]
    assert result["speedup"] == 4


def test_validation_happens_before_and_after_timing():
    events = []
    def check():
        events.append("check")
    def sample(*args):
        events.append("sample")
        return 1
    with patch.object(timing, "_sample_ms", side_effect=sample):
        timing.measure_pair(FakeGCU(), lambda: None, lambda: None, warmup=0, iterations=1, inner=1, timing="eager", check=check)
    assert events == ["check", "sample", "sample", "sample", "sample", "check"]


def test_failed_validation_prevents_measurement():
    def check():
        raise AssertionError("incorrect output")
    with patch.object(timing, "_sample_ms") as sample:
        with pytest.raises(AssertionError):
            timing.measure_pair(FakeGCU(), lambda: None, lambda: None, warmup=0, iterations=1, inner=1, timing="eager", check=check)
    sample.assert_not_called()


def test_graph_events_divide_by_calls_per_graph():
    class Event:
        def __init__(self, enable_timing):
            assert enable_timing
        def record(self):
            pass
        def synchronize(self):
            pass
        def elapsed_time(self, end):
            return 10
    gcu = FakeGCU()
    gcu.Event = Event
    calls = []
    result = timing._sample_ms(gcu, lambda: calls.append(1), 5, "graph")
    assert result == 2
    assert calls == [1]


def test_eager_divides_by_inner_calls():
    calls = []
    with patch.object(timing.time, "perf_counter", side_effect=[1, 1.012]):
        result = timing._sample_ms(FakeGCU(), lambda: calls.append(1), 4, "eager")
    assert abs(result - 3) < 1e-9
    assert len(calls) == 4


def test_graph_capture_uses_same_call_count():
    class Graph:
        pass
    gcu = FakeGCU()
    gcu.GCUGraph = Graph
    gcu.graph = lambda graph: contextlib.nullcontext()
    calls = []
    assert isinstance(timing._capture_graph(gcu, lambda: calls.append(1), 7), Graph)
    assert len(calls) == 7


def test_graph_failures_do_not_fall_back_to_eager():
    with patch.object(timing, "_capture_graph", side_effect=RuntimeError("capture failed")):
        with pytest.raises(RuntimeError):
            timing.measure_pair(FakeGCU(), lambda: None, lambda: None, warmup=0, iterations=1, inner=1, timing="graph")


def test_close_checks_output_and_gradient_errors():
    torch = pytest.importorskip("torch")
    reference = torch.tensor([1.0, 2.0])
    assert timing.check_close("output", reference, reference.clone(), 0.01) == 0
    assert timing.check_close("dq", reference, reference.clone(), 0.01, metric="max") == 0
    with pytest.raises(AssertionError):
        timing.check_close("dk", reference, reference * 2, 0.01)
    with pytest.raises(AssertionError):
        timing.check_close("dv", reference, torch.tensor([float("nan"), 2]), 0.01)
    with pytest.raises(AssertionError):
        timing.check_close("dr", reference, reference.double(), 0.01)


def test_zero_reference_and_nonfinite_reference():
    torch = pytest.importorskip("torch")
    zeros = torch.zeros(2)
    assert timing.check_close("zero", zeros, zeros.clone(), 0.01) == 0
    with pytest.raises(AssertionError):
        timing.check_close("zero", zeros, torch.ones(2), 0.01)
    with pytest.raises(AssertionError):
        timing.check_close("zero decode", zeros, torch.full((2,), 1e-7), 0.01, metric="max")
    with pytest.raises(AssertionError):
        timing.check_close("reference", torch.tensor([float("inf"), 0]), zeros, 0.01)


def _table_record(case=(1, 15, 2, 2, 64), *, phase="forward", baseline="torch"):
    return {
        "case": case, "baseline": baseline, "phase": phase, "dtype": "bfloat16",
        "device": 0, "window_size": None, "timing": "eager", "warmup": 10,
        "iterations": 12, "inner": 1, "baseline_ms": 2.0, "triton_ms": 1.0,
        "speedup": 2.0, "error": 0.0,
        "baseline_source": "/baseline/parallax.py", "triton_source": "/current/parallax.py",
    }


def test_table_groups_shapes_by_phase():
    records = [
        _table_record(), _table_record(phase="fwdbwd"),
        _table_record((1, 63, 1, 1, 64)), _table_record((1, 63, 1, 1, 64), phase="fwdbwd"),
    ]
    output = render_tables(records)
    assert output.count("[Forward]") == 1
    assert output.count("[Forward + Backward]") == 1
    assert output.count("Torch-GCU-fwd(ms)") == 1
    assert output.count("Torch-GCU-fwdbwd(ms)") == 1
    forward, backward = output.split("[Forward + Backward]")
    assert forward.count("2.000x") == 2
    assert backward.count("2.000x") == 2


def test_table_has_aligned_columns_and_no_mad():
    output = render_tables([_table_record(), _table_record((2, 8192, 8, 16, 128))])
    lines = output.splitlines()
    separator = next(line for line in lines if line and set(line) == {"="})
    header = next(line for line in lines if "Torch-GCU-fwd(ms)" in line)
    rows = [line for line in lines if "2.000x" in line]
    assert len(header) == len(separator)
    assert len(rows) == 2
    assert all(len(row) == len(header) for row in rows)
    assert "MAD" not in output
    assert "2.000000" in output and "1.000000" in output
    assert "dtype" in header and "speedup" in header


def test_table_names_actual_baseline_and_source_once():
    output = render_tables([_table_record(), _table_record(phase="fwdbwd")])
    assert "FLA-fwd" not in output
    assert output.count("Torch-GCU source:") == 1
    assert output.count("FlagAttention Triton source:") == 1
    assert "baseline latency / FlagAttention latency" in output


def test_table_keeps_different_timing_settings_separate():
    output = render_tables([_table_record(), {**_table_record(), "timing": "graph"}])
    assert output.count("[Forward]") == 2
    assert "Torch-GCU-fwd(ms)" in output
    assert "timing=eager" in output
    assert "timing=graph" in output


def test_empty_table_does_not_fabricate_results():
    assert render_tables([]) == ""


def test_decode_table_preserves_query_and_kv_head_order():
    output = render_tables([_table_record((1, 512, 8, 2, 128), phase="decode")])
    header = next(line for line in output.splitlines() if "Torch-GCU-decode(ms)" in line)
    assert header.split()[:5] == ["B", "L", "HQ", "HKV", "D"]


def test_pytest_summary_collects_all_shapes_without_repeated_headers():
    from backends._enflame.benchmarks.conftest import BenchmarkOptions, pytest_terminal_summary

    config = SimpleNamespace()
    options = BenchmarkOptions(config)
    options.add_parallax_result(_table_record())
    options.add_parallax_result(_table_record((1, 63, 1, 1, 64)))
    stream = io.StringIO()
    reporter = SimpleNamespace(write=stream.write)
    pytest_terminal_summary(reporter, 0, config)
    assert stream.getvalue().count("[Forward]") == 1
    assert stream.getvalue().count("2.000x") == 2


def test_failed_pytest_run_labels_partial_completed_results():
    from backends._enflame.benchmarks.conftest import BenchmarkOptions, pytest_terminal_summary

    config = SimpleNamespace()
    BenchmarkOptions(config).add_parallax_result(_table_record())
    stream = io.StringIO()
    reporter = SimpleNamespace(write=stream.write, write_line=lambda line: stream.write(line + "\n"))
    pytest_terminal_summary(reporter, 1, config)
    assert "partial results" in stream.getvalue()
    assert stream.getvalue().count("2.000x") == 1
