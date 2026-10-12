from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch


PROJECT_SRC = Path(__file__).resolve().parents[4] / "FlagAttention" / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

try:
    import triton_gcu.triton  # noqa: F401
except ModuleNotFoundError:
    import triton  # noqa: F401

import triton.language.math as tl_math

tl_math.tanh = getattr(tl_math, "tanh", lambda x: x)
tl_math.exp = getattr(tl_math, "exp", lambda x: x)
tl_math.erf = getattr(tl_math, "erf", lambda x: x)
tl_math.pow = getattr(tl_math, "pow", lambda x, y: x**y)

from flag_attn import chunk_gla


def _device() -> torch.device:
    if not hasattr(torch, "gcu") or not torch.gcu.is_available():
        pytest.skip("GCU is unavailable")
    return torch.device("gcu")


def _fp64_recurrence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent token-by-token CPU reference for the defining recurrence."""
    qd, kd, vd, gd = (tensor.double().cpu() for tensor in (q, k, v, g))
    batch, seqlen, heads, key_dim = qd.shape
    value_dim = vd.shape[-1]
    state = (
        torch.zeros(batch, heads, key_dim, value_dim, dtype=torch.float64)
        if initial_state is None
        else initial_state.double().cpu().clone()
    )
    output = torch.empty(batch, seqlen, heads, value_dim, dtype=torch.float64)
    for token in range(seqlen):
        query = qd[:, token]
        key = kd[:, token]
        value = vd[:, token]
        log_gate = gd[:, token]
        state = (
            state * torch.exp(log_gate).unsqueeze(-1)
            + key.unsqueeze(-1) * value.unsqueeze(-2)
        )
        output[:, token] = (
            torch.einsum("bhk,bhkv->bhv", query, state) * scale
        )
    return output, state


@pytest.mark.parametrize(
    "gate_mode,seqlen",
    [
        ("typical", 257),
        pytest.param(
            "negative4",
            129,
            marks=pytest.mark.xfail(
                strict=True,
                reason="BC64 midpoint factors overflow for unusually large gate spans",
            ),
        ),
    ],
)
def test_matches_independent_fp64_recurrence(gate_mode, seqlen):
    device = _device()
    torch.manual_seed(101)
    shape = (1, seqlen, 2, 16)
    q_cpu = torch.randn(shape, dtype=torch.bfloat16)
    k_cpu = torch.randn_like(q_cpu)
    v_cpu = torch.randn(1, seqlen, 2, 13, dtype=torch.bfloat16)
    if gate_mode == "typical":
        g_cpu = torch.nn.functional.logsigmoid(torch.randn_like(q_cpu))
    else:
        g_cpu = torch.full_like(q_cpu, -4.0)
    q, k, v, g = (tensor.to(device) for tensor in (q_cpu, k_cpu, v_cpu, g_cpu))
    scale = 1.0 / math.sqrt(shape[-1])

    actual, actual_state = chunk_gla(
        q, k, v, g, scale=scale, output_final_state=True
    )
    expected, expected_state = _fp64_recurrence(
        q_cpu, k_cpu, v_cpu, g_cpu, scale
    )

    assert torch.isfinite(actual).all()
    assert torch.isfinite(actual_state).all()
    torch.testing.assert_close(
        actual.cpu().float(), expected.float(), rtol=2e-2, atol=5e-2
    )
    torch.testing.assert_close(
        actual_state.cpu(), expected_state.float(), rtol=2e-2, atol=5e-3
    )


@pytest.mark.parametrize("state_v_first", [False, True])
def test_initial_and_final_state_match_fp64(state_v_first):
    device = _device()
    torch.manual_seed(131)
    batch, seqlen, heads, key_dim, value_dim = 2, 33, 2, 16, 13
    q_cpu = torch.randn(batch, seqlen, heads, key_dim, dtype=torch.bfloat16)
    k_cpu = torch.randn_like(q_cpu)
    v_cpu = torch.randn(batch, seqlen, heads, value_dim, dtype=torch.bfloat16)
    g_cpu = torch.nn.functional.logsigmoid(torch.randn_like(q_cpu))
    state_k_first = torch.randn(
        batch, heads, key_dim, value_dim, dtype=torch.float32
    )
    initial = (
        state_k_first.transpose(-1, -2).contiguous()
        if state_v_first
        else state_k_first
    )
    q, k, v, g, initial_gcu = (
        tensor.to(device) for tensor in (q_cpu, k_cpu, v_cpu, g_cpu, initial)
    )
    scale = 1.0 / math.sqrt(key_dim)

    actual, actual_state = chunk_gla(
        q, k, v, g, scale=scale, initial_state=initial_gcu,
        output_final_state=True, state_v_first=state_v_first,
    )
    expected, expected_state = _fp64_recurrence(
        q_cpu, k_cpu, v_cpu, g_cpu, scale, state_k_first
    )
    if state_v_first:
        expected_state = expected_state.transpose(-1, -2).contiguous()

    torch.testing.assert_close(
        actual.cpu().float(), expected.float(), rtol=2e-2, atol=5e-2
    )
    torch.testing.assert_close(
        actual_state.cpu(), expected_state.float(), rtol=2e-2, atol=5e-3
    )


def test_varlen_matches_independent_sequences():
    device = _device()
    torch.manual_seed(151)
    lengths = [7, 11, 5]
    seqlen = sum(lengths)
    q_cpu = torch.randn(1, seqlen, 2, 16, dtype=torch.bfloat16)
    k_cpu = torch.randn_like(q_cpu)
    v_cpu = torch.randn(1, seqlen, 2, 13, dtype=torch.bfloat16)
    g_cpu = torch.nn.functional.logsigmoid(torch.randn_like(q_cpu))
    q, k, v, g = (tensor.to(device) for tensor in (q_cpu, k_cpu, v_cpu, g_cpu))
    boundaries = [0]
    for length in lengths:
        boundaries.append(boundaries[-1] + length)
    cu_seqlens = torch.tensor(boundaries, device=device, dtype=torch.int32)
    scale = 0.25

    actual, actual_states = chunk_gla(
        q, k, v, g, scale=scale, cu_seqlens=cu_seqlens,
        output_final_state=True,
    )
    expected_outputs = []
    expected_states = []
    for begin, end in zip(boundaries, boundaries[1:]):
        output, state = _fp64_recurrence(
            q_cpu[:, begin:end], k_cpu[:, begin:end], v_cpu[:, begin:end],
            g_cpu[:, begin:end], scale,
        )
        expected_outputs.append(output)
        expected_states.append(state)
    expected = torch.cat(expected_outputs, dim=1)
    expected_states = torch.cat(expected_states, dim=0)

    torch.testing.assert_close(
        actual.cpu().float(), expected.float(), rtol=2e-2, atol=5e-2
    )
    torch.testing.assert_close(
        actual_states.cpu(), expected_states.float(), rtol=2e-2, atol=5e-3
    )


def test_wide_key_dispatch_matches_independent_fp64_recurrence():
    device = _device()
    torch.manual_seed(181)
    batch, seqlen, heads, key_dim, value_dim = 1, 17, 1, 128, 13
    q_cpu = torch.randn(
        batch, seqlen, heads, key_dim, dtype=torch.bfloat16
    )
    k_cpu = torch.randn_like(q_cpu)
    v_cpu = torch.randn(
        batch, seqlen, heads, value_dim, dtype=torch.bfloat16
    )
    g_cpu = torch.nn.functional.logsigmoid(torch.randn_like(q_cpu))
    q, k, v, g = (
        tensor.to(device) for tensor in (q_cpu, k_cpu, v_cpu, g_cpu)
    )
    scale = key_dim ** -0.5

    actual, actual_state = chunk_gla(
        q, k, v, g, scale=scale, output_final_state=True,
    )
    expected, expected_state = _fp64_recurrence(
        q_cpu, k_cpu, v_cpu, g_cpu, scale
    )

    assert torch.isfinite(actual).all()
    assert torch.isfinite(actual_state).all()
    torch.testing.assert_close(
        actual.cpu().float(), expected.float(), rtol=2e-2, atol=5e-2
    )
    torch.testing.assert_close(
        actual_state.cpu(), expected_state.float(),
        rtol=2e-2, atol=5e-3,
    )
