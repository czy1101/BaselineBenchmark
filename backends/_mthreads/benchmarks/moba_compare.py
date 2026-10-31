"""纯 Triton 与 MATE 混合 MoBA 的端到端推理性能对比；不执行精度校验。"""
import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from itertools import accumulate
from dataclasses import dataclass, asdict
import gc
import hashlib


def load_package(name, path):
    spec = importlib.util.spec_from_file_location(
        name, path / "__init__.py", submodule_search_locations=[str(path)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def revision(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def summary(samples):
    result = {"mean_ms": statistics.mean(samples), "median_ms": statistics.median(samples),
              "samples_ms": samples, "count": len(samples)}
    if len(samples) >= 100:
        result["p99_ms"] = sorted(samples)[int(0.99 * (len(samples) - 1))]
    return result


@dataclass(frozen=True)
class Case:
    name: str
    seq_lens: tuple
    q_heads: int = 16
    kv_heads: int = 16
    head_dim: int = 128
    chunk_size: int = 128
    topk: int = 8
    dtype: str = "bfloat16"


def make_cases():
    # A fixed, deduplicated suite: every benchmark run covers every workload.
    cases = [Case(f"length-{n}", (n, n)) for n in (512, 2048, 4096, 8192, 16384)]
    cases += [Case(f"batch-{b}", (4096,) * b) for b in (1, 4)]
    cases += [Case(f"dim-{d}", (4096, 4096), head_dim=d) for d in (64, 96, 256)]
    cases += [Case(f"chunk-{c}", (4096, 4096), chunk_size=c) for c in (64, 256, 512)]
    cases += [Case(f"topk-{k}", (4096, 4096), topk=k) for k in (2, 4, 16)]
    cases += [Case("gqa-16q-4kv", (4096, 4096), kv_heads=4),
              Case("mqa-16q-1kv", (4096, 4096), kv_heads=1),
              Case("heads-4", (4096, 4096), q_heads=4, kv_heads=4)]
    cases += [Case("fp16-d128", (4096, 4096), dtype="float16"),
              Case("fp16-d64", (4096, 4096), head_dim=64, dtype="float16"),
              Case("fp16-gqa", (4096, 4096), kv_heads=4, dtype="float16")]
    cases += [Case("chunk-boundaries", (127, 128, 129), q_heads=4, kv_heads=4),
              Case("one-token-tail", (4097, 4097)),
              Case("varlen-gqa", (200, 400, 648), q_heads=8, kv_heads=2),
              Case("varlen-long", (1023, 4097, 8191)),
              Case("fp16-varlen", (127, 513, 2049), q_heads=8, kv_heads=2,
                   head_dim=64, dtype="float16")]
    cases += [Case(f"long-gqa-{n}", (n,), q_heads=32, kv_heads=8)
              for n in (32768, 65536)]
    cases += [Case(f"long-c{c}-k{k}", (32768,), q_heads=32, kv_heads=8,
                   chunk_size=c, topk=k)
              for c, k in ((512, 8), (1024, 8), (512, 16))]
    # Fixed-token-count cases compare sequence packing and batch distribution.
    cases += [Case("tokens16k-b1", (16384,)),
              Case("tokens16k-b8", (2048,) * 8),
              Case("tokens16k-varlen", (1024, 2048, 4096, 9216))]
    # Preserve the original performance-test length sweep. The first three
    # overlap the main suite and are deduplicated by the full workload shape.
    cases += [Case(f"length-{n}", (n, n)) for n in
              (4096, 8192, 16384, 32768, 65536)]
    cases += [Case("topk-1", (4096, 4096), topk=1),
              Case("topk64-saturated", (512, 512), topk=64),
              Case("topk64-long-gqa", (32768,), q_heads=32, kv_heads=8, topk=64)]
    unique = {}
    for case in cases:
        key = (case.seq_lens, case.q_heads, case.kv_heads, case.head_dim,
               case.chunk_size, case.topk, case.dtype)
        unique.setdefault(key, case)
    return list(unique.values())


def run_case(case, args, funcs, native, torch):
    torch.manual_seed(42)
    dtype = getattr(torch, case.dtype)
    total = sum(case.seq_lens)
    q = torch.randn(total, case.q_heads, case.head_dim, device="musa", dtype=dtype)
    k = torch.randn(total, case.kv_heads, case.head_dim, device="musa", dtype=dtype)
    v = torch.randn_like(k)
    cu = torch.tensor([0, *accumulate(case.seq_lens)], device="musa", dtype=torch.int32)
    call_args = (q, k, v, cu, max(case.seq_lens), case.chunk_size, case.topk)
    # Initialize/JIT each implementation outside timing, even when warmup=0.
    for fn in funcs.values():
        fn(*call_args)
    torch.musa.synchronize()
    measurements = {name: [] for name in funcs}
    for round_index in range(args.rounds):
        order = list(funcs) if round_index % 2 == 0 else list(reversed(funcs))
        for name in order:
            fn = funcs[name]
            for _ in range(args.warmup):
                fn(*call_args)
            torch.musa.synchronize()
            for _ in range(args.iterations):
                start = time.perf_counter()
                result = fn(*call_args)
                torch.musa.synchronize()
                measurements[name].append((time.perf_counter() - start) * 1000)
                del result
    stats = {name: summary(values) for name, values in measurements.items()}
    supported = bool(native._supported(q))
    eligible = supported and case.topk > 1 and max(case.seq_lens) > case.chunk_size
    reason = ("native sparse path eligible" if eligible else
              "local attention only" if case.topk == 1 or max(case.seq_lens) <= case.chunk_size else
              "Triton fallback: dtype/head_dim/device/MATE version unsupported")
    return {**asdict(case), "status": "measured", "q_shape": list(q.shape), "kv_shape": list(k.shape),
            "layout": "packed contiguous [tokens, heads, dim]",
            "native_sparse_path_eligible": eligible, "path_reason": reason,
            "latencies": stats,
            "speedup": stats["mate_baseline"]["median_ms"] / stats["pure_triton"]["median_ms"]}


def format_results(results):
    """ASCII headers keep column widths predictable in both Chinese/English terminals."""
    headers = ["Case", "Seq lens", "Hq/Kv", "D", "C", "K", "Dtype",
               "FlagAttn ms", "Baseline ms", "Speedup", "Base path*"]
    rows = []
    for result in results:
        seq = result["seq_lens"]
        seq_text = f"{len(seq)}x{seq[0]}" if len(set(seq)) == 1 else "/".join(map(str, seq))
        prefix = [result["name"], seq_text, f"{result['q_heads']}/{result['kv_heads']}",
                  str(result["head_dim"]), str(result["chunk_size"]), str(result["topk"]),
                  "bf16" if result["dtype"] == "bfloat16" else "fp16"]
        if result["status"] == "measured":
            flag_ms = result["latencies"]["pure_triton"]["median_ms"]
            base_ms = result["latencies"]["mate_baseline"]["median_ms"]
            ratio = base_ms / flag_ms
            path = ("MATE" if result["native_sparse_path_eligible"] else
                    "Local-only" if result["topk"] == 1 or max(seq) <= result["chunk_size"] else
                    "Triton")
            rows.append(prefix + [f"{flag_ms:.4f}", f"{base_ms:.4f}", f"{ratio:.4f}x",
                                  path])
        else:
            rows.append(prefix + ["--", "--", "--", "--"])
    widths = [max(len(row[i]) for row in [headers, *rows]) for i in range(len(headers))]
    numeric = {3, 4, 5, 7, 8, 9}
    def line(row):
        return "| " + " | ".join(value.rjust(widths[i]) if i in numeric else value.ljust(widths[i])
                                  for i, value in enumerate(row)) + " |"
    border = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    return "\n".join([border, line(headers), border, *(line(row) for row in rows), border])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=int, choices=[6, 7], default=6)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=25, help="每轮每个实现的测量次数")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cases = make_cases()
    repository = Path(__file__).resolve().parents[3]
    workspace = repository.parent
    os.environ["MUSA_VISIBLE_DEVICES"] = str(args.card)
    os.environ["MATE_WORKSPACE_BASE"] = str(workspace)
    for variable, path in {
        "TMPDIR": workspace / ".cache/tmp",
        "XDG_CACHE_HOME": workspace / ".cache",
        "TRITON_CACHE_DIR": workspace / f".cache/triton-moba-native-route{args.card}",
    }.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
    sys.dont_write_bytecode = True
    import torch
    import torch_musa  # 注册 MUSA 设备
    import triton

    paths = {
        "pure_triton": workspace / "src/flag_attn/runtime/backend/_mthreads/FLA/moba",
        "mate_baseline": repository / "backends/_mthreads/ops/moba",
    }
    packages = {name: load_package("_bench_" + name, path) for name, path in paths.items()}
    native = sys.modules["_bench_mate_baseline.native_moba"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or repository / "results" / f"moba_card{args.card}_{stamp}.json"
    output = output.resolve()
    if not output.is_relative_to(workspace.resolve()):
        parser.error("output 必须位于 FlagAttention 工作区内")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 4,
        "case_selection": "all predefined, deduplicated cases",
        "speedup_definition": "baseline_median_ms / flagattention_median_ms; >1 means FlagAttention faster",
        "timestamp_utc": stamp, "arguments": {**vars(args), "output": str(output)},
        "environment": {"device": str(torch.musa.get_device_properties(0)),
                        "torch": torch.__version__, "torch_musa": version("torch_musa"),
                        "triton": triton.__version__, "mate": version("mate"),
                        "driver_version": "not queried",
                        "flagattention_commit": revision(workspace),
                        "baselinebenchmark_commit": revision(repository)},
        "implementation_paths": {k: str(v) for k, v in paths.items()},
        "source_sha256": {name: {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(path.glob("*.py"))}
                          for name, path in paths.items()},
        "planned_cases": [asdict(case) for case in cases],
        "timing": "perf_counter + device synchronize per call; includes Python dispatch, allocations, routing, packing, attention, merge, and final synchronization; excludes input generation, compilation and warmup; no correctness checks; alternating round order; inference only",
        "correctness_checked": False,
        "cases": [],
    }
    print(f"Running all {len(cases)} MoBA cases; results: {output}", flush=True)
    funcs = {name: package.parallel_moba for name, package in packages.items()}
    failures = 0
    with torch.inference_mode():
        for index, case in enumerate(cases, 1):
            print(f"[{index:>{len(str(len(cases)))}}/{len(cases)}] {case.name}", flush=True)
            try:
                result = run_case(case, args, funcs, native, torch)
            except Exception as exc:
                failures += 1
                result = {**asdict(case), "status": "error", "error_type": type(exc).__name__,
                          "error": str(exc)}
                print(f"  ERROR: {type(exc).__name__} (details after table)", flush=True)
            report["cases"].append(result)
            report["summary"] = {"planned": len(cases), "completed": len(report["cases"]),
                                 "measured": len(report["cases"]) - failures, "errors": failures}
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            gc.collect()
            torch.musa.empty_cache()
    print("\nMoBA inference: FlagAttention (pure Triton) vs baseline (MATE/Triton)")
    print(format_results(report["cases"]))
    print("Speedup = Baseline ms / FlagAttn ms (medians); >1: FlagAttention faster, =1: equal, <1: FlagAttention slower.")
    print("* Base path: native eligibility, not a kernel execution trace; Timings do not imply correctness; -- indicates an execution error (details below).")
    print("Speedup compares measured medians, not statistical significance.")
    for result in report["cases"]:
        if result["status"] == "error":
            print(f"\n{result['name']}: {result['error_type']}\n{result['error']}")
    print(f"Results: {output}; {len(cases) - failures} measured, {failures} errors", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
