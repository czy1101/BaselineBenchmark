"""Correctness tests for the experimental Torch GCU KDA implementation."""

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

OPS_DIR = Path(__file__).resolve().parents[1] / "ops"
sys.path.insert(0, str(OPS_DIR))

from kda import chunk_kda_torch_gcu


def _gcu_available() -> bool:
    return hasattr(torch, "gcu") and torch.gcu.is_available()


pytestmark = pytest.mark.skipif(
    not _gcu_available(), reason="Torch GCU KDA tests require an Enflame GCU"
)


def _error_ratio(actual: torch.Tensor, expected: torch.Tensor) -> float:
    diff = (actual.float() - expected.float()).square().mean().sqrt().item()
    base = expected.float().square().mean().sqrt().item()
    return diff / (base + 1e-8)


def _run_recurrent_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float,
    initial_state: torch.Tensor | None,
    state_v_first: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent float32 recurrence for one equal-length batch."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    state = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)
    if initial_state is not None:
        state = initial_state.transpose(-1, -2) if state_v_first else initial_state
        state = state.float().contiguous()

    output = torch.empty(B, T, H, V, dtype=torch.float32, device=q.device)
    for token_id in range(T):
        q_i = q[:, token_id] * scale
        k_i = k[:, token_id]
        v_i = v[:, token_id]
        g_i = g[:, token_id]
        beta_i = beta[:, token_id]
        state = state * torch.exp(g_i).unsqueeze(-1)
        residual = v_i - (k_i.unsqueeze(-1) * state).sum(dim=-2)
        state = state + torch.einsum(
            "bhk,bhv->bhkv", beta_i.unsqueeze(-1) * k_i, residual
        )
        output[:, token_id] = torch.einsum("bhk,bhkv->bhv", q_i, state)

    final_state = state.transpose(-1, -2).contiguous() if state_v_first else state
    return output.to(v.dtype), final_state


@torch.inference_mode()
def _chunk_kda_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float,
    initial_state: torch.Tensor | None,
    state_v_first: bool,
    cu_seqlens: torch.Tensor | None,
    lower_bound: float,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    **_kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    qf = F.normalize(q.float(), p=2.0, dim=-1, eps=1e-6)
    kf = F.normalize(k.float(), p=2.0, dim=-1, eps=1e-6)
    vf = v.float()
    gf = g.float() + dt_bias.reshape(1, 1, q.shape[2], q.shape[3])
    gf = lower_bound * torch.sigmoid(
        torch.exp(A_log).reshape(1, 1, q.shape[2], 1) * gf
    )
    betaf = torch.sigmoid(beta.float())

    if cu_seqlens is None:
        return _run_recurrent_reference(
            qf,
            kf,
            vf,
            gf,
            betaf,
            scale=scale,
            initial_state=initial_state,
            state_v_first=state_v_first,
        )

    offsets = cu_seqlens.detach().cpu().tolist()
    output_parts = []
    final_state_parts = []
    for sequence_id, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        initial_state_i = (
            None
            if initial_state is None
            else initial_state[sequence_id : sequence_id + 1]
        )
        output_i, final_state_i = _run_recurrent_reference(
            qf[:, start:end],
            kf[:, start:end],
            vf[:, start:end],
            gf[:, start:end],
            betaf[:, start:end],
            scale=scale,
            initial_state=initial_state_i,
            state_v_first=state_v_first,
        )
        output_parts.append(output_i)
        final_state_parts.append(final_state_i)
    return torch.cat(output_parts, dim=1), torch.cat(final_state_parts, dim=0)


@pytest.mark.chunk_kda
@pytest.mark.parametrize("seq_lens", [[64], [29, 35, 17]])
@pytest.mark.parametrize("use_initial_state", [False, True])
@torch.inference_mode()
def test_chunk_kda_torch_gcu_matches_reference(
    seq_lens,
    use_initial_state,
):
    torch.manual_seed(42)

    device = torch.device("gcu")
    T = sum(seq_lens)
    N = len(seq_lens)
    H = 2
    D = 128
    q = torch.randn(1, T, H, D, device=device, dtype=torch.bfloat16)
    k = torch.randn(1, T, H, D, device=device, dtype=torch.bfloat16)
    v = torch.randn(1, T, H, D, device=device, dtype=torch.bfloat16)
    g = torch.randn(1, T, H, D, device=device, dtype=torch.bfloat16)
    beta = torch.randn(1, T, H, device=device, dtype=torch.bfloat16)
    A_log = torch.rand(H, device=device, dtype=torch.float32)
    dt_bias = torch.rand(H, D, device=device, dtype=torch.float32)
    initial_state = None
    if use_initial_state:
        initial_state = torch.randn(
            N, H, D, D, device=device, dtype=torch.float32
        )
    cu_seqlens = None
    if N > 1:
        offsets = [0]
        for length in seq_lens:
            offsets.append(offsets[-1] + length)
        cu_seqlens = torch.tensor(offsets, device=device, dtype=torch.int32)

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

    actual, actual_state = chunk_kda_torch_gcu(q, k, v, g, beta, **kwargs)
    expected, expected_state = _chunk_kda_reference(q, k, v, g, beta, **kwargs)

    output_ratio = _error_ratio(actual, expected)
    state_ratio = _error_ratio(actual_state, expected_state)
    assert output_ratio < 0.005, f"output error ratio: {output_ratio}"
    assert state_ratio < 0.005, f"state error ratio: {state_ratio}"
