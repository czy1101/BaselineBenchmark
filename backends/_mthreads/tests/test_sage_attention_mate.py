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

"""Configurable SageAttention correctness checks for MATE and FlagAttention.

The default checks the MATE end-to-end path against a common FP32 PyTorch
reference with aggregate quantized-attention quality metrics. Environment
variables select the comparison target, execution path, and tolerance policy:

* SAGE_CORRECTNESS_TARGET: mate-torch, flag-torch, flag-mate, or all.
* SAGE_CORRECTNESS_PATH: e2e, core, or both.
* SAGE_CORRECTNESS_MODE: relaxed or strict.
"""

from __future__ import annotations

import inspect
import math
import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

try:
    import torch_musa  # noqa: F401
except ImportError:
    torch_musa = None

try:
    import mate
    from mate.testing import quantize_sage_attention_tensor
    from sageattention import sageattn
except ImportError:
    mate = None
    quantize_sage_attention_tensor = None
    sageattn = None


MATE_RECIPE = (128, 16, -1, 1)
MUSA_AVAILABLE = hasattr(torch, "musa") and torch.musa.is_available()

TARGET_ENV = "SAGE_CORRECTNESS_TARGET"
PATH_ENV = "SAGE_CORRECTNESS_PATH"
MODE_ENV = "SAGE_CORRECTNESS_MODE"

TARGETS = ("mate-torch", "flag-torch", "flag-mate", "all")
PATHS = ("e2e", "core", "both")
MODES = ("relaxed", "strict")


def _env_choice(name: str, default: str, choices: tuple[str, ...]) -> str:
    value = os.environ.get(name, default).strip().lower()
    if value not in choices:
        valid = ", ".join(choices)
        raise ValueError(f"{name} must be one of: {valid}; got {value!r}")
    return value


def _test_config() -> tuple[str, str, str]:
    return (
        _env_choice(TARGET_ENV, "mate-torch", TARGETS),
        _env_choice(PATH_ENV, "e2e", PATHS),
        _env_choice(MODE_ENV, "relaxed", MODES),
    )


def _uses_mate(target: str) -> bool:
    return target in {"mate-torch", "flag-mate", "all"}


def _uses_flag(target: str) -> bool:
    return target in {"flag-torch", "flag-mate", "all"}


@pytest.fixture(scope="module")
def device():
    target, _, _ = _test_config()
    if torch_musa is None or not MUSA_AVAILABLE:
        pytest.skip("SageAttention correctness tests require a MUSA device")
    if _uses_mate(target) and (
        mate is None or sageattn is None or quantize_sage_attention_tensor is None
    ):
        pytest.skip("Install the Moore Threads MATE SageAttention packages first")
    return torch.device("musa")


def _find_flagattention_root() -> Path:
    configured_root = os.environ.get("FLAG_ATTENTION_ROOT")
    if configured_root:
        root = Path(configured_root).expanduser().resolve()
        if not (root / "src" / "flag_attn").is_dir():
            raise RuntimeError(
                "FLAG_ATTENTION_ROOT must point to the FlagAttention repository "
                f"root; got {root}"
            )
        return root

    for parent in Path(__file__).resolve().parents:
        if (parent / "src" / "flag_attn").is_dir():
            return parent
    raise RuntimeError(
        "Could not locate FlagAttention; set FLAG_ATTENTION_ROOT explicitly."
    )


def _flag_api():
    src_dir = _find_flagattention_root() / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from flag_attn.runtime.backend._mthreads.sage_attention import (
        forward,
        per_block_int8,
    )

    return forward, per_block_int8


def _headwise_matmul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            torch.stack(
                [
                    left[batch, head] @ right[batch, head]
                    for head in range(left.shape[1])
                ]
            )
            for batch in range(left.shape[0])
        ]
    )


def _to_hnd(tensor: torch.Tensor, tensor_layout: str) -> torch.Tensor:
    return tensor if tensor_layout == "HND" else tensor.transpose(1, 2)


def _to_bnhd(tensor: torch.Tensor, tensor_layout: str) -> torch.Tensor:
    return tensor.transpose(1, 2) if tensor_layout == "HND" else tensor


def _reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str,
    attn_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_hnd = _to_hnd(q, tensor_layout)
    k_hnd = _to_hnd(k, tensor_layout)
    v_hnd = _to_hnd(v, tensor_layout)

    num_groups = q_hnd.shape[1] // k_hnd.shape[1]
    k_hnd = torch.repeat_interleave(k_hnd, num_groups, dim=1)
    v_hnd = torch.repeat_interleave(v_hnd, num_groups, dim=1)

    scores = _headwise_matmul(q_hnd.float(), k_hnd.float().transpose(-1, -2))
    scores *= q_hnd.shape[-1] ** -0.5
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        else:
            scores = scores + attn_mask.float()

    output_hnd = _headwise_matmul(torch.softmax(scores, dim=-1), v_hnd.float())
    lse = torch.logsumexp(scores, dim=-1)
    if tensor_layout == "NHD":
        return output_hnd.transpose(1, 2), lse
    return output_hnd, lse


def _mate_e2e(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str,
    attn_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    kwargs = {
        "tensor_layout": tensor_layout,
        "is_causal": False,
        "qk_quant_dtype": "int8",
        "quant_recipe": MATE_RECIPE,
        "smooth_k": True,
        "return_lse": True,
    }
    if attn_mask is not None:
        if "attn_mask" not in inspect.signature(sageattn).parameters:
            pytest.skip(
                "MATE sageattention does not publicly expose arbitrary attn_mask"
            )
        kwargs["attn_mask"] = attn_mask

    result = sageattn(q, k, v, **kwargs)
    assert isinstance(result, tuple) and len(result) == 2
    return result


def _mate_core(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str,
) -> tuple[torch.Tensor, None]:
    q_bnhd = _to_bnhd(q, tensor_layout).contiguous().to(torch.bfloat16)
    k_bnhd = _to_bnhd(k, tensor_layout).contiguous().to(torch.bfloat16)
    v_bnhd = _to_bnhd(v, tensor_layout).contiguous().to(torch.bfloat16)

    q_quant, q_scale = quantize_sage_attention_tensor(
        q_bnhd,
        operand="q",
        quant_recipe=MATE_RECIPE,
        quant_dtype=torch.int8,
    )
    k_quant, k_scale = quantize_sage_attention_tensor(
        k_bnhd,
        operand="k",
        quant_recipe=MATE_RECIPE,
        quant_dtype=torch.int8,
        smooth_k=True,
    )
    v_quant, v_scale = quantize_sage_attention_tensor(
        v_bnhd,
        operand="v",
        quant_recipe=MATE_RECIPE,
        quant_dtype=torch.float8_e4m3fn,
    )
    output_bnhd = mate.sage_attn_quantized(
        q=q_quant,
        k=k_quant,
        v=v_quant,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        softmax_scale=q.shape[-1] ** -0.5,
        causal=False,
        quant_recipe=MATE_RECIPE,
        return_lse=False,
    )
    output = output_bnhd.transpose(1, 2) if tensor_layout == "HND" else output_bnhd
    return output, None


def _flag_core(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    flag_forward, flag_per_block_int8 = _flag_api()
    q_int8, q_scale, k_int8, k_scale = flag_per_block_int8(
        q, k, tensor_layout=tensor_layout
    )
    kernel_v = v if v.dtype == torch.float16 else v.to(torch.float16)
    output, lse_log2 = flag_forward(
        q_int8,
        k_int8,
        kernel_v,
        q_scale,
        k_scale,
        tensor_layout=tensor_layout,
        output_dtype=q.dtype,
        return_lse=True,
    )
    return output, lse_log2 * math.log(2.0)


def _run_backend(
    backend: str,
    path: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if backend == "mate":
        if path == "e2e":
            return _mate_e2e(q, k, v, tensor_layout)
        return _mate_core(q, k, v, tensor_layout)
    if backend == "flag":
        # Functional E2E and core checks use the same kernels. Their distinction
        # is whether quantization preparation is treated as part of the selected
        # path; no timing is performed in this test.
        return _flag_core(q, k, v, tensor_layout)
    raise AssertionError(f"Unknown backend: {backend}")


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual = actual.float()
    expected = expected.float()
    diff = actual - expected
    return {
        "cosine": F.cosine_similarity(
            actual.flatten(), expected.flatten(), dim=0
        ).item(),
        "relative_l1": (
            diff.abs().sum()
            / (actual.abs().sum() + expected.abs().sum() + 1.0e-8)
        ).item(),
        "rmse": diff.square().mean().sqrt().item(),
        "max_abs": diff.abs().max().item(),
    }


def _quality_limits(comparison: str) -> tuple[float, float, float]:
    if comparison == "FlagAttention vs MATE":
        return 0.99, 0.08, 0.03
    return 0.995, 0.05, 0.02


def _assert_output(
    comparison: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    mode: str,
) -> dict[str, float]:
    assert torch.isfinite(actual).all(), f"{comparison} produced non-finite output"
    metrics = _metrics(actual, expected)
    min_cosine, max_relative_l1, max_rmse = _quality_limits(comparison)
    assert metrics["cosine"] >= min_cosine, (comparison, metrics)
    assert metrics["relative_l1"] <= max_relative_l1, (comparison, metrics)
    assert metrics["rmse"] <= max_rmse, (comparison, metrics)
    if mode == "strict":
        torch.testing.assert_close(
            actual.float(), expected.float(), atol=2.0e-2, rtol=2.0e-2
        )
    return metrics


def _assert_lse(
    comparison: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    mode: str,
) -> None:
    if mode == "strict" or comparison == "FlagAttention vs PyTorch":
        atol, rtol = 2.0e-2, 2.0e-2
    else:
        atol, rtol = 1.5e-1, 3.0e-2
    torch.testing.assert_close(
        actual.float(), expected.float(), atol=atol, rtol=rtol
    )


def _selected_paths(path: str) -> tuple[str, ...]:
    return ("e2e", "core") if path == "both" else (path,)


def _selected_pairs(target: str) -> tuple[tuple[str, str], ...]:
    pairs = {
        "mate-torch": (("mate", "torch"),),
        "flag-torch": (("flag", "torch"),),
        "flag-mate": (("flag", "mate"),),
        "all": (("mate", "torch"), ("flag", "torch"), ("flag", "mate")),
    }
    return pairs[target]


def _selected_comparison_cases():
    target, selected_path, _ = _test_config()
    return [
        pytest.param(
            path,
            actual,
            expected,
            id=f"{path}-{actual}-vs-{expected}",
        )
        for path in _selected_paths(selected_path)
        for actual, expected in _selected_pairs(target)
    ]


def _comparison_name(actual: str, expected: str) -> str:
    names = {
        "mate": "MATE",
        "flag": "FlagAttention",
        "torch": "PyTorch",
    }
    return f"{names[actual]} vs {names[expected]}"


@pytest.mark.parametrize("tensor_layout", ["HND", "NHD"])
@pytest.mark.parametrize("num_kv_heads", [1, 2])
@pytest.mark.parametrize(
    "path,actual_name,expected_name", _selected_comparison_cases()
)
def test_selected_sageattention_correctness(
    tensor_layout,
    num_kv_heads,
    path,
    actual_name,
    expected_name,
    device,
):
    target, _, mode = _test_config()
    torch.manual_seed(2026)
    batch_size, num_query_heads, seq_len, head_dim = 1, 2, 128, 64
    q = torch.randn(
        batch_size,
        num_query_heads,
        seq_len,
        head_dim,
        device=device,
        dtype=torch.float16,
    )
    k = torch.randn(
        batch_size,
        num_kv_heads,
        seq_len,
        head_dim,
        device=device,
        dtype=torch.float16,
    )
    v = torch.randn_like(k)

    if tensor_layout == "NHD":
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

    results: dict[str, tuple[torch.Tensor, torch.Tensor | None]] = {
        "torch": _reference(q, k, v, tensor_layout)
    }
    for backend in {actual_name, expected_name} - {"torch"}:
        results[backend] = _run_backend(
            backend, path, q, k, v, tensor_layout
        )
    torch.musa.synchronize()

    comparison = _comparison_name(actual_name, expected_name)
    actual_out, actual_lse = results[actual_name]
    expected_out, expected_lse = results[expected_name]
    metrics = _assert_output(comparison, actual_out, expected_out, mode)
    if actual_lse is not None and expected_lse is not None:
        _assert_lse(comparison, actual_lse, expected_lse, mode)
    print(
        f"target={target} path={path} mode={mode} "
        f"layout={tensor_layout} kv_heads={num_kv_heads} "
        f"comparison={comparison!r} "
        f"cos={metrics['cosine']:.6f} "
        f"rel_l1={metrics['relative_l1']:.6f} "
        f"rmse={metrics['rmse']:.6f} "
        f"max_abs={metrics['max_abs']:.6f}"
    )


def test_mate_has_no_maxnreg_tuning_parameter(device):
    target, _, _ = _test_config()
    if not _uses_mate(target):
        pytest.skip("Selected target does not use MATE")
    del device
    assert "maxnreg" not in inspect.signature(sageattn).parameters


@pytest.mark.parametrize("mask_kind", ["bool", "additive"])
def test_mate_e2e_masks_and_partial_blocks(mask_kind, device):
    target, selected_path, mode = _test_config()
    if target not in {"mate-torch", "all"}:
        pytest.skip("Mask check requires the MATE vs PyTorch target")
    if selected_path == "core":
        pytest.skip("The public mask check applies to the E2E wrapper")

    torch.manual_seed(7)
    q = torch.randn((1, 1, 129, 128), device=device, dtype=torch.float16)
    k = torch.randn((1, 1, 70, 128), device=device, dtype=torch.float16)
    v = torch.randn_like(k)

    if mask_kind == "bool":
        attn_mask = torch.ones((1, 1, 129, 70), device=device, dtype=torch.bool)
        attn_mask[..., ::3] = False
    else:
        attn_mask = torch.zeros(
            (1, 1, 129, 70), device=device, dtype=torch.float32
        )
        attn_mask[..., ::3] = -2.0

    actual, actual_lse = _mate_e2e(q, k, v, "HND", attn_mask=attn_mask)
    expected, expected_lse = _reference(q, k, v, "HND", attn_mask)
    _assert_output("MATE vs PyTorch", actual, expected, mode)
    _assert_lse("MATE vs PyTorch", actual_lse, expected_lse, mode)
