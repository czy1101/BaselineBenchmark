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

_BASELINE_NAMES = {"torch": "Torch-GCU"}
_PHASE_NAMES = {
    "forward": ("Forward", "fwd"),
    "fwdbwd": ("Forward + Backward", "fwdbwd"),
    "decode": ("Decode", "decode"),
}
_COLUMN_WIDTHS = (3, 6, 3, 3, 4, 9, 22, 26, 9)
_GROUP_FIELDS = (
    "baseline", "phase", "timing", "warmup", "iterations", "inner",
    "device", "window_size", "baseline_source", "triton_source",
)


def _format_row(values):
    return " ".join(f"{value:>{width}}" for value, width in zip(values, _COLUMN_WIDTHS))


def render_tables(records):
    if not records:
        return ""
    groups = {}
    for record in records:
        key = tuple(record[field] for field in _GROUP_FIELDS)
        groups.setdefault(key, []).append(record)
    lines = []
    width = sum(_COLUMN_WIDTHS) + len(_COLUMN_WIDTHS) - 1
    for rows in groups.values():
        first = rows[0]
        baseline = _BASELINE_NAMES[first["baseline"]]
        title, suffix = _PHASE_NAMES[first["phase"]]
        dimensions = ("B", "L", "HQ", "HKV", "D") if first["phase"] == "decode" else ("B", "T", "H", "HQ", "D")
        headers = (
            *dimensions, "dtype", f"{baseline}-{suffix}(ms)", f"FlagAttention-{suffix}(ms)",
            "speedup",
        )
        if lines:
            lines.append("")
        lines.extend((
            "=" * width,
            f"[{title}]",
            "=" * width,
            _format_row(headers),
            "-" * width,
        ))
        for record in rows:
            lines.append(_format_row((
                *record["case"], record["dtype"],
                f"{record['baseline_ms']:.6f}", f"{record['triton_ms']:.6f}",
                f"{record['speedup']:.3f}x",
            )))
        lines.extend((
            "-" * width,
            f"baseline={baseline} device=gcu:{first['device']} timing={first['timing']} "
            f"window={first['window_size']} warmup={first['warmup']} "
            f"samples/provider={2 * first['iterations']} calls/sample={first['inner']} "
            f"max_error={max(record['error'] for record in rows):.3e}",
        ))
    lines.extend((
        "",
        "speedup = baseline latency / FlagAttention latency; >1 means FlagAttention Triton is faster.",
    ))
    sources = {}
    for record in records:
        key = (record["baseline"], record["baseline_source"], record["triton_source"])
        sources.setdefault(key, None)
    for baseline, baseline_source, triton_source in sources:
        lines.extend((
            f"{_BASELINE_NAMES[baseline]} source: {baseline_source}",
            f"FlagAttention Triton source: {triton_source}",
        ))
    return "\n".join(lines)
