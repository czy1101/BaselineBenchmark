"""Compare the Torch GCU KDA baseline with the Triton/TLE implementation."""

import math
import os
import statistics
import sys
import time
from pathlib import Path

import pytest
import torch


WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
FLAG_ATTN_ROOTS = (
    WORKSPACE_ROOT / "FlagAttention-main",
    WORKSPACE_ROOT / "FlagAttention",
)
FLAG_ATTN_ROOT = next(
    (
        root
        for root in FLAG_ATTN_ROOTS
        if (
            root
            / "src"
            / "flag_attn"
            / "runtime"
            / "backend"
            / "_enflame"
            / "FLA"
            / "kda"
            / "__init__.py"
        ).is_file()
    ),
    None,
)
if FLAG_ATTN_ROOT is None:
    searched = ", ".join(
        str(
            root
            / "src"
            / "flag_attn"
            / "runtime"
            / "backend"
            / "_enflame"
            / "FLA"
            / "kda"
        )
        for root in FLAG_ATTN_ROOTS
    )
    raise ImportError(
        f"Unable to locate the FlagAttention Triton KDA source; searched: {searched}"
    )

sys.path.insert(0, str(FLAG_ATTN_ROOT))
sys.path.insert(0, str(FLAG_ATTN_ROOT / "src"))
OPS_DIR = Path(__file__).resolve().parents[1] / "ops"
sys.path.insert(0, str(OPS_DIR))

from flag_attn.runtime.backend._enflame.FLA.kda import chunk_kda as chunk_kda_triton
from kda import chunk_kda_torch_gcu


FULL_CASES = (
    ("fixed-8192", [8192]),
    ("varlen-mixed", [1300, 547, 2048, 963, 271, 3063]),
    ("varlen-8x1024", [1024] * 8),
)
QUICK_CASES = (("fixed-1024", [1024]),)


def _gcu_available() -> bool:
    return hasattr(torch, "gcu") and torch.gcu.is_available()


def _synchronize() -> None:
    torch.gcu.synchronize()


def _measure(function, warmup: int, repetitions: int) -> list[float]:
    for _ in range(warmup):
        function()
        _synchronize()

    samples = []
    for _ in range(repetitions):
        _synchronize()
        start = time.perf_counter()
        function()
        _synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def _error_ratio(actual: torch.Tensor, expected: torch.Tensor) -> float:
    diff = (actual.float() - expected.float()).square().mean().sqrt().item()
    base = expected.float().square().mean().sqrt().item()
    return diff / (base + 1e-8)


@pytest.mark.skipif(
    not _gcu_available(),
    reason="Torch GCU KDA benchmark requires an available GCU device",
)
@pytest.mark.chunk_kda
@torch.inference_mode()
def test_chunk_kda_torch_gcu_vs_triton(monkeypatch):
    monkeypatch.setenv("FLAG_ATTN_CHUNK_KDA_BACKEND", "strict_tle")
    warmup = int(os.getenv("KDA_BENCH_WARMUP", "2"))
    repetitions = int(os.getenv("KDA_BENCH_ITERS", "10"))
    quick = os.getenv("KDA_BENCH_QUICK", "0") == "1"
    cases = QUICK_CASES if quick else FULL_CASES
    device = torch.device("gcu")
    dtype = torch.bfloat16
    H = 96
    D = 128

    print(
        f"device={device} dtype={dtype} H={H} D={D} "
        f"baseline=torch_gcu optimized=triton_strict_tle "
        f"warmup={warmup} iterations={repetitions}",
        flush=True,
    )

    for case_name, seq_lens in cases:
        torch.manual_seed(42)
        T = sum(seq_lens)
        N = len(seq_lens)
        q = torch.randn(1, T, H, D, device=device, dtype=dtype)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        g = torch.randn_like(q)
        beta = torch.randn(1, T, H, device=device, dtype=dtype)
        A_log = torch.rand(H, device=device, dtype=torch.float32)
        dt_bias = torch.rand(H, D, device=device, dtype=torch.float32)
        initial_state = torch.randn(
            N, H, D, D, device=device, dtype=torch.float32
        )
        cu_seqlens = None
        if N > 1:
            offsets = [0]
            for length in seq_lens:
                offsets.append(offsets[-1] + length)
            cu_seqlens = torch.tensor(
                offsets, device=device, dtype=torch.int32
            )

        kwargs = {
            "scale": 1.0 / math.sqrt(D),
            "initial_state": initial_state,
            "output_final_state": True,
            "use_qk_l2norm_in_kernel": True,
            "use_gate_in_kernel": True,
            "use_beta_sigmoid_in_kernel": True,
            "allow_neg_eigval": False,
            "safe_gate": True,
            "lower_bound": -5.0,
            "state_v_first": True,
            "cu_seqlens": cu_seqlens,
            "chunk_size": 32,
            "A_log": A_log,
            "dt_bias": dt_bias,
        }

        torch_output, torch_state = chunk_kda_torch_gcu(
            q, k, v, g, beta, **kwargs
        )
        triton_output, triton_state = chunk_kda_triton(
            q, k, v, g, beta, **kwargs
        )
        _synchronize()
        assert bool(torch.isfinite(torch_output).all().cpu())
        assert bool(torch.isfinite(torch_state).all().cpu())
        assert bool(torch.isfinite(triton_output).all().cpu())
        assert bool(torch.isfinite(triton_state).all().cpu())

        output_error = _error_ratio(torch_output, triton_output)
        state_error = _error_ratio(torch_state, triton_state)
        assert output_error < 0.005, f"output error ratio: {output_error}"
        assert state_error < 0.005, f"state error ratio: {state_error}"

        torch_samples = _measure(
            lambda: chunk_kda_torch_gcu(q, k, v, g, beta, **kwargs),
            warmup,
            repetitions,
        )
        triton_samples = _measure(
            lambda: chunk_kda_triton(q, k, v, g, beta, **kwargs),
            warmup,
            repetitions,
        )
        torch_ms = statistics.median(torch_samples)
        triton_ms = statistics.median(triton_samples)
        speedup = torch_ms / triton_ms
        saved_ms = torch_ms - triton_ms
        print(
            f"case={case_name} seq_lens={seq_lens} "
            f"torch_baseline_ms={torch_ms:.3f} "
            f"triton_ms={triton_ms:.3f} "
            f"saved_ms={saved_ms:.3f} speedup={speedup:.3f}x "
            f"output_error={output_error:.6f} state_error={state_error:.6f}",
            flush=True,
        )
