import math

import pytest
import torch

import mcoplib._C  # noqa: F401


D = 128
DTYPE = torch.bfloat16
SCALE = D ** -0.5
LOWER_BOUND = -5.0

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="chunk_kda_fwd requires a CUDA/MACA device",
)


def _reference(q, k, v, g, beta, A_log, dt_bias, initial_state):
    q = q.float()
    k = k.float()
    v = v.float()
    g = g.float()
    beta = beta.float()

    batch, sequence_length, heads, key_dim = q.shape

    q = q / torch.linalg.vector_norm(
        q, dim=-1, keepdim=True
    ).clamp_min(1.0e-12)
    k = k / torch.linalg.vector_norm(
        k, dim=-1, keepdim=True
    ).clamp_min(1.0e-12)

    gate = LOWER_BOUND * torch.sigmoid(
        torch.exp(A_log).view(1, 1, heads, 1)
        * (
            g
            + dt_bias.view(1, 1, heads, key_dim)
        )
    )
    alpha = torch.exp(
        gate.clamp(min=LOWER_BOUND, max=0.0)
    )
    beta = torch.sigmoid(beta)

    if initial_state is None:
        state = torch.zeros(
            batch, heads, key_dim, D, dtype=torch.float32
        )
    else:
        # External state is [B,H,V,K]; recurrence state is [B,H,K,V].
        state = (
            initial_state.float()
            .transpose(-1, -2)
            .contiguous()
        )

    output = torch.empty(
        batch,
        sequence_length,
        heads,
        D,
        dtype=torch.float32,
    )

    for token in range(sequence_length):
        state = state * alpha[:, token].unsqueeze(-1)

        prediction = (
            k[:, token].unsqueeze(-1) * state
        ).sum(dim=-2)

        residual = beta[:, token].unsqueeze(-1) * (
            v[:, token] - prediction
        )

        state = state + (
            k[:, token].unsqueeze(-1)
            * residual.unsqueeze(-2)
        )

        output[:, token] = SCALE * (
            q[:, token].unsqueeze(-1) * state
        ).sum(dim=-2)

    return (
        output.to(DTYPE),
        state.transpose(-1, -2).contiguous(),
    )


def _make_cpu_inputs(batch, sequence_length, heads, has_initial, seed):
    generator = torch.Generator().manual_seed(seed)

    def random_bf16(shape, magnitude):
        return (
            torch.randn(shape, generator=generator) * magnitude
        ).to(DTYPE).contiguous()

    q = random_bf16((batch, sequence_length, heads, D), 0.20)
    k = random_bf16((batch, sequence_length, heads, D), 0.20)
    v = random_bf16((batch, sequence_length, heads, D), 0.10)
    g = random_bf16((batch, sequence_length, heads, D), 0.30)
    beta = random_bf16((batch, sequence_length, heads), 0.50)

    A_log = torch.linspace(
        -0.7, 0.2, heads, dtype=torch.float32
    ).contiguous()

    dt_bias = (
        torch.randn(
            heads,
            D,
            generator=generator,
            dtype=torch.float32,
        )
        * 0.10
    ).contiguous()

    initial_state = None
    if has_initial:
        value_axis = torch.arange(
            D, dtype=torch.float32
        ).view(1, 1, D, 1)
        key_axis = torch.arange(
            D, dtype=torch.float32
        ).view(1, 1, 1, D)
        head_axis = torch.arange(
            heads, dtype=torch.float32
        ).view(1, heads, 1, 1)

        # Deliberately non-symmetric in V and K.
        initial_state = (
            0.00020 * value_axis
            - 0.00007 * key_axis
            + 0.00300 * head_axis
        ).expand(
            batch, heads, D, D
        ).clone().contiguous()

    return q, k, v, g, beta, A_log, dt_bias, initial_state


def _to_device_contiguous(tensor):
    if tensor is None:
        return None
    result = tensor.to("cuda").contiguous()
    assert result.is_contiguous()
    return result


def _invoke_native(cpu_inputs):
    q, k, v, g, beta, A_log, dt_bias, initial_state = [
        _to_device_contiguous(tensor)
        for tensor in cpu_inputs
    ]

    batch, sequence_length, heads, _ = q.shape

    output = torch.empty(
        batch,
        sequence_length,
        heads,
        D,
        device="cuda",
        dtype=DTYPE,
    )
    final_state = torch.empty(
        batch,
        heads,
        D,
        D,
        device="cuda",
        dtype=torch.float32,
    )

    inputs = [q, k, v, g, beta, A_log, dt_bias]
    if initial_state is not None:
        inputs.append(initial_state)
    input_copies = [tensor.clone() for tensor in inputs]

    torch.ops._C.chunk_kda_fwd(
        output,
        final_state,
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        initial_state,
        SCALE,
        LOWER_BOUND,
    )
    torch.cuda.synchronize()

    for tensor, copied in zip(inputs, input_copies):
        assert torch.equal(tensor, copied)

    return (
        output,
        final_state,
        (q, k, v, g, beta, A_log, dt_bias, initial_state),
    )


@pytest.mark.parametrize(
    "batch,sequence_length,heads,has_initial,seed",
    [
        (1, 1, 1, False, 20260814),
        (1, 5, 2, True, 20260815),
        (2, 17, 4, False, 20260816),
        (1, 33, 4, True, 20260817),
    ],
)
def test_chunk_kda_fwd_matches_reference(
    batch,
    sequence_length,
    heads,
    has_initial,
    seed,
):
    cpu_inputs = _make_cpu_inputs(
        batch,
        sequence_length,
        heads,
        has_initial,
        seed,
    )
    expected_output, expected_state = _reference(*cpu_inputs)

    output, final_state, _ = _invoke_native(cpu_inputs)

    assert torch.isfinite(output).all()
    assert torch.isfinite(final_state).all()

    torch.testing.assert_close(
        output.cpu().float(),
        expected_output.float(),
        atol=2.0e-3,
        rtol=2.0e-2,
    )
    torch.testing.assert_close(
        final_state.cpu(),
        expected_state,
        atol=2.0e-4,
        rtol=5.0e-3,
    )


def test_chunk_kda_fwd_rejects_unsupported_inputs():
    cpu_inputs = _make_cpu_inputs(
        batch=1,
        sequence_length=2,
        heads=2,
        has_initial=True,
        seed=20260818,
    )
    device_inputs = [
        _to_device_contiguous(tensor)
        for tensor in cpu_inputs
    ]
    q, k, v, g, beta, A_log, dt_bias, initial_state = device_inputs

    output = torch.empty_like(q)
    final_state = torch.empty(
        1, 2, D, D, device="cuda", dtype=torch.float32
    )

    def call(
        *,
        q_arg=q,
        k_arg=k,
        v_arg=v,
        g_arg=g,
        beta_arg=beta,
        a_log_arg=A_log,
        dt_bias_arg=dt_bias,
        out_arg=output,
        final_arg=final_state,
        initial_arg=initial_state,
        scale=SCALE,
        lower_bound=LOWER_BOUND,
    ):
        torch.ops._C.chunk_kda_fwd(
            out_arg,
            final_arg,
            q_arg,
            k_arg,
            v_arg,
            g_arg,
            beta_arg,
            a_log_arg,
            dt_bias_arg,
            initial_arg,
            scale,
            lower_bound,
        )

    # T and H are both 2, so transposition preserves shape but not layout.
    noncontiguous_q = q.transpose(1, 2)
    assert noncontiguous_q.shape == q.shape
    assert not noncontiguous_q.is_contiguous()
    with pytest.raises(RuntimeError, match="q must be contiguous"):
        call(q_arg=noncontiguous_q)

    with pytest.raises(RuntimeError, match="q must have dtype bfloat16"):
        call(q_arg=q.float().contiguous())

    with pytest.raises(RuntimeError, match="beta must have dtype bfloat16"):
        call(beta_arg=beta.float().contiguous())

    with pytest.raises(
        RuntimeError,
        match="lower_bound must satisfy",
    ):
        call(lower_bound=0.0)

    with pytest.raises(
        RuntimeError,
        match="out must not alias",
    ):
        call(out_arg=v)

    with pytest.raises(
        RuntimeError,
        match="final_state must not alias",
    ):
        call(final_arg=initial_state)

    with pytest.raises(RuntimeError, match="q must have shape"):
        call(q_arg=q.reshape(1, -1))

    with pytest.raises(RuntimeError, match="q must have K=128"):
        call(q_arg=q[..., :64].contiguous())

    with pytest.raises(RuntimeError, match="k must have the same"):
        call(k_arg=k[:, :1].contiguous())

    with pytest.raises(RuntimeError, match="v must have shape"):
        call(v_arg=v[:, :1].contiguous())

    with pytest.raises(RuntimeError, match="g must have the same"):
        call(g_arg=g[:, :1].contiguous())

    with pytest.raises(RuntimeError, match="beta must have shape"):
        call(beta_arg=beta[:, :1].contiguous())

    with pytest.raises(RuntimeError, match="A_log must have dtype"):
        call(a_log_arg=A_log.to(torch.bfloat16))

    with pytest.raises(RuntimeError, match="A_log must have shape"):
        call(a_log_arg=A_log[:1].contiguous())

    with pytest.raises(RuntimeError, match="dt_bias must have dtype"):
        call(dt_bias_arg=dt_bias.to(torch.bfloat16))

    with pytest.raises(RuntimeError, match="dt_bias must have shape"):
        call(dt_bias_arg=dt_bias[:, :64].contiguous())

    with pytest.raises(RuntimeError, match="initial_state must have dtype"):
        call(initial_arg=initial_state.to(torch.bfloat16))

    with pytest.raises(RuntimeError, match="initial_state must have shape"):
        call(initial_arg=initial_state[:, :1].contiguous())

    with pytest.raises(RuntimeError, match="out must have dtype"):
        call(out_arg=output.float())

    with pytest.raises(RuntimeError, match="out must have shape"):
        call(out_arg=output[:, :1].contiguous())

    with pytest.raises(RuntimeError, match="final_state must have dtype"):
        call(final_arg=final_state.to(torch.bfloat16))

    with pytest.raises(RuntimeError, match="final_state must have shape"):
        call(final_arg=final_state[:, :1].contiguous())

    with pytest.raises(RuntimeError, match="scale must be finite"):
        call(scale=float("nan"))

    with pytest.raises(RuntimeError, match="lower_bound must satisfy"):
        call(lower_bound=float("nan"))


def test_chunk_kda_constants():
    assert D == 128
    assert math.isclose(SCALE, 1.0 / math.sqrt(128))
    assert -5.0 <= LOWER_BOUND < 0.0
