"""Native HIP InfLLM-V2 benchmark: 17 frozen shapes, optional FlagAttention peer.

Run from the repository root with ``python -m
backends._hygon.benchmarks.benchmark_infllmv2_hygon --help``.
No Triton compiler is imported unless --compare-flagattention is requested.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from backends._hygon.ops.infllmv2 import infllmv2_attention, infllmv2_decode
from backends._hygon.ops.infllmv2.operator import get_plan


@dataclass(frozen=True)
class Case:
    name: str
    batch: int
    q: int
    kv: int
    d: int
    mode: str = "prefill"


CASES = (
    Case("dense-1x1024-d64", 1, 1024, 1024, 64),
    Case("full-1x8192-d64", 1, 8192, 8192, 64),
    Case("full-1x8192-d128", 1, 8192, 8192, 128),
    Case("full-1x32768-d64", 1, 32768, 32768, 64),
    Case("full-1x32768-d128", 1, 32768, 32768, 128),
    Case("full-1x65536-d64", 1, 65536, 65536, 64),
    Case("full-1x65536-d128", 1, 65536, 65536, 128),
    Case("chunk-1x128x16384-d64", 1, 128, 16384, 64),
    Case("chunk-1x128x65536-d128", 1, 128, 65536, 128),
    Case("backward-1x128x8192-d64", 1, 128, 8192, 64, "forward_backward"),
    Case("backward-1x128x8192-d128", 1, 128, 8192, 128, "forward_backward"),
    Case("decode-1x8192-d64", 1, 1, 8192, 64, "decode"),
    Case("decode-1x8192-d128", 1, 1, 8192, 128, "decode"),
    Case("decode-1x65536-d64", 1, 1, 65536, 64, "decode"),
    Case("decode-1x65536-d128", 1, 1, 65536, 128, "decode"),
    Case("decode-8x32768-d64", 8, 1, 32768, 64, "decode"),
    Case("decode-8x32768-d128", 8, 1, 32768, 128, "decode"),
)


def measure(fn, warmup_ms, rep_ms):
    start = time.perf_counter()
    while (time.perf_counter() - start) * 1000 < warmup_ms:
        fn()
        torch.cuda.synchronize()
    samples = []
    elapsed = 0.0
    while elapsed < rep_ms or len(samples) < 5:
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        output = fn()
        end.record()
        end.synchronize()
        ms = start.elapsed_time(end)
        samples.append(ms)
        elapsed += ms
        del output
    return samples


def check_sampled_attention(q, k, v, buffers, case):
    """Independent FP32 Stage2 checks for three query rows per sequence.

    Selection itself is checked independently in the test suite. This check
    uses the actual selected indices and does not claim full-pipeline equality.
    """
    out, selected = buffers["out"], buffers["selected"]
    for batch in range(case.batch):
        for qi in sorted({0, case.q // 2, case.q - 1}):
            index = batch * case.q + qi
            limit = case.kv - case.q + qi + 1
            for hk in range(2):
                if case.kv < 8192:
                    positions = torch.arange(limit, device=q.device)
                else:
                    blocks = selected[hk, index].long()
                    blocks = blocks[blocks >= 0]
                    positions = (
                        blocks[:, None] * 64 + torch.arange(64, device=q.device)
                    ).flatten()
                    positions = positions[positions < limit]
                assert positions.numel(), (
                    "Selection leaves a query without valid context"
                )
                ids = positions + batch * case.kv
                query = q[index, hk * 16 : (hk + 1) * 16].float()
                probs = torch.softmax(query @ k[ids, hk].float().T * case.d**-0.5, -1)
                expected = probs @ v[ids, hk].float()
                torch.testing.assert_close(
                    out[index, hk * 16 : (hk + 1) * 16].float(),
                    expected,
                    atol=0.04,
                    rtol=0.04,
                )


def run_case(case, args, peer):
    generator = torch.Generator(device="cuda").manual_seed(101)
    training = case.mode == "forward_backward"
    q, k, v = [
        torch.randn(
            shape, device="cuda", dtype=torch.bfloat16, generator=generator
        ).requires_grad_(training)
        for shape in (
            (case.batch * case.q, 32, case.d),
            (case.batch * case.kv, 2, case.d),
            (case.batch * case.kv, 2, case.d),
        )
    ]
    cuq = torch.arange(case.batch + 1, device="cuda", dtype=torch.int32) * case.q
    cuk = torch.arange(case.batch + 1, device="cuda", dtype=torch.int32) * case.kv
    grad = (
        torch.randn(q.shape, device=q.device, dtype=q.dtype, generator=generator)
        if training
        else None
    )

    def call(attention, decode):
        out = (
            decode(q, k, v, cuk, case.kv)
            if case.mode == "decode"
            else attention(q, k, v, cuq, cuk, case.q, case.kv)
        )
        return torch.autograd.grad(out, (q, k, v), grad) if training else out

    calls = {"hip": lambda: call(infllmv2_attention, infllmv2_decode)}
    if peer:
        calls["flagattention"] = lambda: call(
            peer.infllmv2_attention, peer.infllmv2_decode
        )
    plan = get_plan(
        (case.q,) * case.batch, (case.kv,) * case.batch, q, max_seqlen_k=case.kv
    )
    with torch.no_grad():
        buffers = plan.forward_buffers(q, k, v)
        check_sampled_attention(q, k, v, buffers, case)
    del buffers
    first = {name: fn() for name, fn in calls.items()}
    accuracy = {}
    for name, values in first.items():
        tensors = values if training else (values,)
        assert all(bool(torch.isfinite(x).all()) for x in tensors), name
    if peer:
        a = first["hip"] if training else (first["hip"],)
        b = first["flagattention"] if training else (first["flagattention"],)
        tolerance = 0.03 if training else 0.04
        accuracy = {
            label: {
                "max_abs": float((x.float() - y.float()).abs().max()),
                "allclose": bool(torch.allclose(x, y, atol=tolerance, rtol=tolerance)),
                "atol": tolerance,
                "rtol": tolerance,
            }
            for label, x, y in zip(("dq", "dk", "dv") if training else ("out",), a, b)
        }
    del first
    rounds = {name: [] for name in calls}
    for turn in range(args.rounds):
        order = list(calls) if turn % 2 == 0 else list(reversed(calls))
        for name in order:
            rounds[name].append(measure(calls[name], args.warmup_ms, args.rep_ms))
    stats = {}
    for name, rr in rounds.items():
        samples = [x for row in rr for x in row]
        stats[name] = dict(
            median_ms=statistics.median([statistics.median(row) for row in rr]),
            mean_ms=statistics.mean(samples),
            samples=len(samples),
            raw_ms=rr,
            p99_ms=sorted(samples)[math.ceil(len(samples) * 0.99) - 1]
            if len(samples) >= 100
            else None,
        )
    result = dict(
        case=case.name,
        mode=case.mode,
        B=case.batch,
        Q=case.q,
        KV=case.kv,
        D=case.d,
        dtype="bfloat16",
        Hq=32,
        Hkv=2,
        implementations=stats,
        accuracy=accuracy,
        sampled_stage2_pass=True,
        all_results_finite=True,
    )
    if peer:
        result["hip_over_flagattention"] = (
            stats["hip"]["median_ms"] / stats["flagattention"]["median_ms"]
        )
    print(
        f"{case.name:28s} HIP={stats['hip']['median_ms']:.6f} ms"
        + (
            f" FlagAttention={stats['flagattention']['median_ms']:.6f} ms"
            f" HIP/FlagAttention={result['hip_over_flagattention']:.3f}x accuracy={accuracy}"
            if peer
            else ""
        ),
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", action="append", help="Exact case name; repeat to select several"
    )
    parser.add_argument("--warmup-ms", type=int, default=20)
    parser.add_argument("--rep-ms", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--compare-flagattention", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("results/infllmv2_hygon.json")
    )
    args = parser.parse_args()
    if min(args.warmup_ms, args.rep_ms, args.rounds) <= 0:
        parser.error("warmup, measurement budgets and rounds must be positive")
    if not torch.cuda.is_available() or not torch.version.hip:
        raise SystemExit("This benchmark requires Hygon PyTorch/HIP on BW1000")
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    if not props.gcnArchName.startswith("gfx936"):
        raise SystemExit("This native implementation targets BW1000/gfx936")
    selected = [c for c in CASES if not args.case or c.name in args.case]
    if not selected or (args.case and set(args.case) - {c.name for c in selected}):
        parser.error("Unknown case selection")
    peer = importlib.import_module("flag_attn") if args.compare_flagattention else None
    repo = Path(__file__).resolve().parents[3]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    report = dict(
        operator="infllmv2",
        implementation="native HIP/MMAC",
        torch=torch.__version__,
        hip=torch.version.hip,
        device=str(props),
        commit=commit.stdout.strip() or "unknown",
        warmup_ms=args.warmup_ms,
        rep_ms=args.rep_ms,
        rounds=args.rounds,
        timing="GPU events; median of round medians; alternating peer order; public API; fresh outputs/workspaces; warmed metadata; no graph/profiler",
        correctness="independent sampled same-index Stage2 plus finite outputs/gradients; see tests for Stage1/selection/gradient references",
        cases=[],
    )
    if peer:
        report["flagattention_path"] = peer.__file__
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for case in selected:
        report["cases"].append(run_case(case, args, peer))
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    failed = any(
        not metric["allclose"]
        for row in report["cases"]
        for metric in row["accuracy"].values()
    )
    report["status"] = "PEER_ACCURACY_MISMATCH" if failed else "PASS"
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not peer:
        assert not any(
            n.split(".")[0] in {"triton", "flagtree", "tilelang", "flag_attn"}
            for n in sys.modules
        )
    if failed:
        raise SystemExit("Peer accuracy mismatch recorded; tolerances were not widened")
    print(f"INFLLMV2_HYGON_BENCHMARK_PASS {len(selected)}", flush=True)


if __name__ == "__main__":
    main()
