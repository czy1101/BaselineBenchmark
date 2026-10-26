"""Compare the Torch GCU GDN2 baseline with FlagAttention Triton/TLE."""

import importlib
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
            / "gdn2"
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
            / "gdn2"
        )
        for root in FLAG_ATTN_ROOTS
    )
    raise ImportError(
        f"Unable to locate the FlagAttention Triton GDN2 source; searched: {searched}"
    )

sys.path.insert(0, str(FLAG_ATTN_ROOT))
sys.path.insert(0, str(FLAG_ATTN_ROOT / "src"))
OPS_DIR = Path(__file__).resolve().parents[1] / "ops"
sys.path.insert(0, str(OPS_DIR))

from flag_attn.runtime.backend._enflame.FLA.gdn2 import (
    chunk_gdn2 as chunk_gdn2_triton,
)
from gdn2 import chunk_gdn2_torch_gcu


QUICK_CASES = ((1, 512, 8, 64, 64),)
FULL_CASES = (
    (1, 512, 8, 64, 64),
    (2, 512, 8, 64, 64),
    (4, 1024, 8, 64, 64),
    (1, 2048, 8, 64, 64),
    (1, 4096, 16, 64, 64),
    (1, 8192, 96, 128, 128),
    (2, 2048, 16, 256, 512),
    (2, 16384, 16, 128, 128),
    (4, 1024, 8, 256, 512),
    (4, 2048, 16, 128, 128),
    (4, 4096, 64, 128, 128),
    (8, 1024, 8, 64, 64),
    (8, 2048, 32, 256, 256),
)


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


def _triton_call(*args, **kwargs):
    module = importlib.import_module(
        "flag_attn.runtime.backend._enflame.FLA.gdn2.chunk_gdn2"
    )
    call_kwargs = dict(kwargs)
    if module.HAS_TLE_GDN2:
        call_kwargs["chunk_size"] = 16
    return chunk_gdn2_triton(*args, **call_kwargs)


@pytest.mark.skipif(
    not _gcu_available(),
    reason="Torch GCU GDN2 benchmark requires an Enflame GCU",
)
@pytest.mark.chunk_gdn2
@pytest.mark.gdn2
@torch.inference_mode()
def test_chunk_gdn2_torch_gcu_vs_triton():
    warmup = int(os.getenv("GDN2_BENCH_WARMUP", "2"))
    repetitions = int(os.getenv("GDN2_BENCH_ITERS", "10"))
    quick = os.getenv("GDN2_BENCH_QUICK", "0") == "1"
    cases = QUICK_CASES if quick else FULL_CASES
    device = torch.device("gcu")
    module = importlib.import_module(
        "flag_attn.runtime.backend._enflame.FLA.gdn2.chunk_gdn2"
    )
    triton_variant = "tle" if module.HAS_TLE_GDN2 else "native_triton"

    print(
        f"device={device} baseline=torch_gcu optimized={triton_variant} "
        f"warmup={warmup} iterations={repetitions}",
        flush=True,
    )

    for dtype in (torch.float16, torch.bfloat16):
        for B, T, H, K, V in cases:
            torch.manual_seed(42)
            q = torch.randn(B, T, H, K, device=device, dtype=dtype) / math.sqrt(K)
            k = torch.randn(B, T, H, K, device=device, dtype=dtype) / math.sqrt(K)
            v = torch.randn(B, T, H, V, device=device, dtype=dtype)
            g = (-torch.rand(B, T, H, K, device=device) * 0.1).to(dtype)
            b = torch.rand(B, T, H, K, device=device, dtype=dtype)
            w = torch.rand(B, T, H, V, device=device, dtype=dtype)
            initial_state = 0.01 * torch.randn(
                B, H, K, V, device=device, dtype=torch.float32
            )
            kwargs = {
                "scale": K**-0.5,
                "initial_state": initial_state,
                "output_final_state": True,
                "use_qk_l2norm_in_kernel": False,
                "use_gate_in_kernel": False,
                "safe_gate": False,
                "lower_bound": None,
                "A_log": None,
                "dt_bias": None,
                "state_v_first": False,
                "cu_seqlens": None,
                "cu_seqlens_cpu": None,
                "chunk_size": 64,
            }

            torch_output, torch_state = chunk_gdn2_torch_gcu(
                q, k, v, g, b, w, **kwargs
            )
            triton_output, triton_state = _triton_call(
                q, k, v, g, b, w, **kwargs
            )
            _synchronize()
            output_error = _error_ratio(torch_output, triton_output)
            state_error = _error_ratio(torch_state, triton_state)
            assert output_error < 0.01, f"output error ratio: {output_error}"
            assert state_error < 0.01, f"state error ratio: {state_error}"

            torch_samples = _measure(
                lambda: chunk_gdn2_torch_gcu(q, k, v, g, b, w, **kwargs),
                warmup,
                repetitions,
            )
            triton_samples = _measure(
                lambda: _triton_call(q, k, v, g, b, w, **kwargs),
                warmup,
                repetitions,
            )
            torch_ms = statistics.median(torch_samples)
            triton_ms = statistics.median(triton_samples)
            speedup = torch_ms / triton_ms
            saved_ms = torch_ms - triton_ms
            print(
                f"shape={(B, T, H, K, V)} dtype={dtype} "
                f"torch_baseline_ms={torch_ms:.3f} triton_ms={triton_ms:.3f} "
                f"saved_ms={saved_ms:.3f} speedup={speedup:.3f}x "
                f"output_error={output_error:.6f} state_error={state_error:.6f}",
                flush=True,
            )
