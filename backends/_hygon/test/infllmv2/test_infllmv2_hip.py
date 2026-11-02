# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Native HIP correctness against independent Torch references, without Triton.

Reference arithmetic is copied from FlagAttention commit
54417c6e9e4f85cdca248981ed18dc5b9fa5eb5d (Apache-2.0).
"""

from __future__ import annotations

import math
import pytest
import torch

from backends._hygon.ops.infllmv2 import operator as op

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.version.hip,
    reason="requires the Hygon PyTorch/HIP runtime",
)


def compressed_lengths(cu, kernel_size, stride):
    lengths = cu[1:] - cu[:-1]
    counts = torch.clamp((lengths - kernel_size) // stride + 1, min=0)
    return torch.cat((torch.zeros_like(cu[:1]), counts.cumsum(0).to(torch.int32)))


def compress_k_ref(
    k: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    kernel_size: int,
    stride: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean-pool packed K windows; output remains ``[total, Hkv, D]``."""
    cu_out = compressed_lengths(cu_seqlens_k, kernel_size, stride)
    chunks: list[torch.Tensor] = []
    cu_cpu = cu_seqlens_k.detach().cpu().tolist()
    for start, end in zip(cu_cpu[:-1], cu_cpu[1:]):
        length = end - start
        for offset in range(0, max(0, length - kernel_size + 1), stride):
            chunks.append(
                k[start + offset : start + offset + kernel_size].float().mean(dim=0)
            )
    if not chunks:
        return k.new_empty((0, k.shape[1], k.shape[2])), cu_out
    return torch.stack(chunks).to(k.dtype), cu_out


def stage1_ref(
    q: torch.Tensor,
    k1: torch.Tensor,
    k2: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k1: torch.Tensor,
    cu_seqlens_k2: torch.Tensor,
    k1_stride: int = 16,
    k2_stride: int = 64,
    softmax_scale: float | None = None,
    causal: bool = True,
    cu_seqlens_k: torch.Tensor | None = None,
) -> torch.Tensor:
    """Two-pass Stage-1 semantics, returning ``[Hkv, Tq, max_K1]``."""
    total_q, hq, d = q.shape
    hkv = k1.shape[1]
    if hq % hkv:
        raise ValueError("number of query heads must be divisible by KV heads")
    group = hq // hkv
    scale = softmax_scale if softmax_scale is not None else d**-0.5
    max_k1 = int(torch.max(cu_seqlens_k1[1:] - cu_seqlens_k1[:-1]).item())
    out = torch.zeros((hkv, total_q, max_k1), dtype=q.dtype, device=q.device)
    q_cu = cu_seqlens_q.detach().cpu().tolist()
    original_k_cu = (
        (cu_seqlens_q if cu_seqlens_k is None else cu_seqlens_k).detach().cpu().tolist()
    )
    k1_cu = cu_seqlens_k1.detach().cpu().tolist()
    k2_cu = cu_seqlens_k2.detach().cpu().tolist()

    for qs, qe, ks, ke, a1, z1, a2, z2 in zip(
        q_cu[:-1],
        q_cu[1:],
        original_k_cu[:-1],
        original_k_cu[1:],
        k1_cu[:-1],
        k1_cu[1:],
        k2_cu[:-1],
        k2_cu[1:],
    ):
        q_seq = q[qs:qe].float().view(qe - qs, hkv, group, d)
        for hk in range(hkv):
            qh = q_seq[:, hk]
            coarse = torch.einsum("qgd,kd->qgk", qh, k2[a2:z2, hk].float()) * scale
            fine = torch.einsum("qgd,kd->qgk", qh, k1[a1:z1, hk].float()) * scale
            if causal:
                qpos = torch.arange(qe - qs, device=q.device)[:, None]
                cpos = torch.arange(z2 - a2, device=q.device)[None, :]
                fpos = torch.arange(z1 - a1, device=q.device)[None, :]
                coarse_stride = k1_stride if z1 - a1 == z2 - a2 else k2_stride
                coarse_q_len = max(0, qe - qs - coarse_stride + 1) // coarse_stride
                fine_q_len = max(0, qe - qs - k1_stride + 1) // k1_stride
                coarse_valid = cpos < torch.clamp(
                    (qpos + 1) // coarse_stride - 1 + z2 - a2 - coarse_q_len,
                    min=0,
                )
                fine_valid = fpos < torch.clamp(
                    (qpos + 1) // k1_stride - 1 + z1 - a1 - fine_q_len,
                    min=0,
                )
                coarse = coarse.masked_fill(~coarse_valid[:, None, :], -torch.inf)
                fine = fine.masked_fill(~fine_valid[:, None, :], -torch.inf)
            lse = torch.logsumexp(coarse, dim=-1)
            probs = torch.exp(fine - lse[..., None])
            probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            out[hk, qs:qe, : z1 - a1] = probs.sum(dim=1).to(q.dtype).float()
    return out


def pool_scores_ref(
    score: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k1: torch.Tensor,
    max_seqlen_q: int,
    kernel_size: int = 32,
    kernel_stride: int = 16,
    block_size: int = 64,
    init_blocks: int = 1,
    local_blocks: int = 32,
    cu_seqlens_k: torch.Tensor | None = None,
    max_seqlen_k: int | None = None,
) -> torch.Tensor:
    """Map K1 scores to block scores using the operator pooling rules."""
    hkv, total_q, _ = score.shape
    max_seqlen_k = max_seqlen_q if max_seqlen_k is None else max_seqlen_k
    max_blocks = math.ceil(max_seqlen_k / block_size)
    result = torch.full(
        (hkv, total_q, max_blocks),
        -torch.inf,
        device=score.device,
        dtype=score.dtype,
    )
    window = kernel_size // kernel_stride + block_size // kernel_stride - 1
    pad = kernel_size // kernel_stride - 1
    q_cu = cu_seqlens_q.detach().cpu().tolist()
    original_k_cu = (
        (cu_seqlens_q if cu_seqlens_k is None else cu_seqlens_k).detach().cpu().tolist()
    )
    k1_cu = cu_seqlens_k1.detach().cpu().tolist()
    for qs, qe, ks, ke, k1s, k1e in zip(
        q_cu[:-1],
        q_cu[1:],
        original_k_cu[:-1],
        original_k_cu[1:],
        k1_cu[:-1],
        k1_cu[1:],
    ):
        k1_len = k1e - k1s
        q_position_offset = (ke - ks) - (qe - qs)
        for q_local in range(qe - qs):
            q_block = (q_position_offset + q_local) // block_size
            num_k_blocks = math.ceil((ke - ks) / block_size)
            for kb in range(num_k_blocks):
                if kb < init_blocks or kb <= q_block <= kb + local_blocks:
                    result[:, qs + q_local, kb] = torch.inf
                    continue
                begin = kb * (block_size // kernel_stride) - pad
                ids = torch.arange(begin, begin + window, device=score.device)
                valid = (ids >= 0) & (ids < k1_len)
                if valid.any():
                    result[:, qs + q_local, kb] = score[
                        :, qs + q_local, ids[valid]
                    ].amax(dim=-1)
    return result


def select_blocks_ref(
    block_score: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    topk: int,
    block_size: int = 64,
    cu_seqlens_k: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select causal blocks; invalid slots are ``-1`` and rows are sorted."""
    hkv, total_q, _ = block_score.shape
    result = torch.full(
        (hkv, total_q, topk),
        -1,
        dtype=torch.int32,
        device=block_score.device,
    )
    q_cu = cu_seqlens_q.detach().cpu().tolist()
    original_k_cu = (
        (cu_seqlens_q if cu_seqlens_k is None else cu_seqlens_k).detach().cpu().tolist()
    )
    for qs, qe, ks, ke in zip(
        q_cu[:-1], q_cu[1:], original_k_cu[:-1], original_k_cu[1:]
    ):
        q_position_offset = (ke - ks) - (qe - qs)
        for q_local in range(qe - qs):
            q_position = q_position_offset + q_local
            valid_blocks = min(
                math.ceil((ke - ks) / block_size),
                (q_position + block_size) // block_size,
            )
            count = min(topk, valid_blocks)
            if count:
                idx = torch.argsort(
                    block_score[:, qs + q_local, :valid_blocks],
                    dim=-1,
                    descending=True,
                    stable=True,
                )[:, :count]
                result[:, qs + q_local, :count] = idx.sort(dim=-1).values.to(
                    torch.int32
                )
    return result


def sparse_attention_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    selected_blocks: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    block_size: int = 64,
    softmax_scale: float | None = None,
    causal: bool = True,
) -> torch.Tensor:
    """Attention over the union of selected original-KV blocks."""
    _, hq, d = q.shape
    hkv = k.shape[1]
    group = hq // hkv
    scale = softmax_scale if softmax_scale is not None else d**-0.5
    q_fp32, k_fp32, v_fp32 = q.float(), k.float(), v.float()
    out = torch.zeros_like(q)
    q_cu = cu_seqlens_q.detach().cpu().tolist()
    k_cu = cu_seqlens_k.detach().cpu().tolist()
    for qs, qe, ks, ke in zip(q_cu[:-1], q_cu[1:], k_cu[:-1], k_cu[1:]):
        q_len, k_len = qe - qs, ke - ks
        for qi in range(q_len):
            causal_limit = k_len - q_len + qi + 1
            for h in range(hq):
                hk = h // group
                positions: list[int] = []
                for block in selected_blocks[hk, qs + qi].detach().cpu().tolist():
                    if block < 0:
                        continue
                    lo = block * block_size
                    positions.extend(range(lo, min(lo + block_size, k_len)))
                if causal:
                    positions = [pos for pos in positions if pos < causal_limit]
                if not positions:
                    continue
                ids = torch.tensor(positions, device=q.device, dtype=torch.long)
                logits = (k_fp32[ks + ids, hk] @ q_fp32[qs + qi, h]) * scale
                probs = torch.softmax(logits, dim=0)
                out[qs + qi, h] = (
                    (probs[:, None] * v_fp32[ks + ids, hk]).sum(dim=0).to(q.dtype)
                )
    return out


def dense_attention_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    softmax_scale: float | None = None,
    causal: bool = True,
) -> torch.Tensor:
    """Readable packed dense attention with bottom-right causal alignment."""
    _, hq, d = q.shape
    hkv = k.shape[1]
    if hq % hkv:
        raise ValueError("number of query heads must be divisible by KV heads")
    group = hq // hkv
    scale = softmax_scale if softmax_scale is not None else d**-0.5
    outputs: list[torch.Tensor] = []
    q_cu = cu_seqlens_q.detach().cpu().tolist()
    k_cu = cu_seqlens_k.detach().cpu().tolist()
    for qs, qe, ks, ke in zip(q_cu[:-1], q_cu[1:], k_cu[:-1], k_cu[1:]):
        q_seq = q[qs:qe].float().transpose(0, 1)
        k_seq = k[ks:ke].float().repeat_interleave(group, dim=1).transpose(0, 1)
        v_seq = v[ks:ke].float().repeat_interleave(group, dim=1).transpose(0, 1)
        logits = torch.matmul(q_seq, k_seq.transpose(-1, -2)) * scale
        if causal:
            q_len, k_len = qe - qs, ke - ks
            q_pos = (k_len - q_len) + torch.arange(q_len, device=q.device)[:, None]
            k_pos = torch.arange(k_len, device=q.device)[None, :]
            logits = logits.masked_fill((k_pos > q_pos)[None, :, :], -torch.inf)
        probs = torch.nan_to_num(torch.softmax(logits, dim=-1), nan=0.0)
        outputs.append(torch.matmul(probs, v_seq).transpose(0, 1).to(q.dtype))
    return torch.cat(outputs, dim=0)


def cumulative(lengths):
    values = [0]
    for length in lengths:
        values.append(values[-1] + length)
    return torch.tensor(values, device="cuda", dtype=torch.int32)


def make_inputs(
    dtype, *, d=64, hq=32, hkv=2, qlens=(3, 5), klens=(128, 192), strided=False
):
    generator = torch.Generator(device="cuda").manual_seed(101)
    tensors = []
    for length, heads in ((sum(qlens), hq), (sum(klens), hkv), (sum(klens), hkv)):
        x = (
            torch.randn(
                (length, heads, d * (2 if strided else 1)),
                device="cuda",
                dtype=dtype,
                generator=generator,
            )
            * 0.2
        )
        if strided:
            x = x[..., ::2]
        tensors.append(x.requires_grad_())
    return (*tensors, cumulative(qlens), cumulative(klens))


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("path", ("dense", "mmac_sparse", "generic_sparse"))
@pytest.mark.parametrize("scale", (None, 0.0, -0.125))
def test_attention_and_gradients(dtype, path, scale):
    q, k, v, cq, ck = make_inputs(
        dtype, strided=True, d=96 if path == "generic_sparse" else 64
    )
    config = (
        op.NativeConfig(
            dense_len=0,
            block_size=32,
            topk=8,
            k1_kernel_size=16,
            k1_stride=8,
            k2_kernel_size=64,
            k2_stride=32,
        )
        if path == "generic_sparse"
        else op.NativeConfig(dense_len=0 if path == "mmac_sparse" else 8192)
    )
    actual = op.infllmv2_attention(
        q, k, v, cq, ck, 5, 192, config=config, softmax_scale=scale
    )
    # Every valid block is selected in these small cases, so the independent
    # dense mathematical reference also checks sparse causal suffix alignment.
    expected = dense_attention_ref(q, k, v, cq, ck, scale)
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
    grad = torch.randn_like(actual)
    ga = torch.autograd.grad(actual, (q, k, v), grad)
    ge = torch.autograd.grad(expected, (q, k, v), grad)
    for a, e in zip(ga, ge):
        torch.testing.assert_close(a, e, atol=0.03, rtol=0.03)
    if scale == 0:
        assert torch.count_nonzero(ga[0]) == 0 and torch.count_nonzero(ga[1]) == 0


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float32))
def test_selection_stages(dtype):
    q, k, v, cq, ck = make_inputs(dtype, qlens=(3, 5), klens=(256, 384))
    k1, c1 = op.compress_k(k, ck, 32, 16)
    k2, c2 = op.compress_k(k, ck, 128, 64)
    r1, rc1 = compress_k_ref(k, ck, 32, 16)
    r2, rc2 = compress_k_ref(k, ck, 128, 64)
    torch.testing.assert_close(c1, rc1, atol=0, rtol=0)
    torch.testing.assert_close(c2, rc2, atol=0, rtol=0)
    torch.testing.assert_close(k1, r1, atol=0.02, rtol=0.02)
    torch.testing.assert_close(k2, r2, atol=0.02, rtol=0.02)
    score = op.stage1(q, k1, k2, cq, c1, c2, 5, cu_seqlens_k=ck)
    reference = stage1_ref(q, k1, k2, cq, c1, c2, 5, cu_seqlens_k=ck)
    torch.testing.assert_close(score, reference, atol=0.04, rtol=0.04)
    pooled = op.pool_scores(
        score,
        cq,
        c1,
        5,
        init_blocks=1,
        local_blocks=2,
        cu_seqlens_k=ck,
        max_seqlen_k=384,
    )
    pool_ref = pool_scores_ref(
        score,
        cq,
        c1,
        5,
        init_blocks=1,
        local_blocks=2,
        cu_seqlens_k=ck,
        max_seqlen_k=384,
    )
    torch.testing.assert_close(pooled, pool_ref, atol=0.04, rtol=0.04)
    ids = op.select_blocks(pooled, cq, 4, cu_seqlens_k=ck)
    torch.testing.assert_close(
        ids, select_blocks_ref(pooled, cq, 4, cu_seqlens_k=ck), atol=0, rtol=0
    )


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float32))
def test_exact_topk(dtype):
    cq, ck = cumulative((2,)), cumulative((8192,))
    score = torch.zeros((2, 2, 128), device="cuda", dtype=dtype)
    score[..., ::3] = 1
    score[..., 7] = float("nan")
    score[..., 9] = float("inf")
    score[..., 11] = -float("inf")
    actual = op.select_blocks(score, cq, 64, cu_seqlens_k=ck)
    # The documented native total order treats NaN as -inf and resolves equal
    # scores toward the lower block ID; selected indices are sorted ascending.
    sanitized = torch.where(torch.isnan(score), -float("inf"), score)
    expected = select_blocks_ref(sanitized, cq, 64, cu_seqlens_k=ck)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float32))
def test_fixed_sparse_attention_gradients(dtype):
    q, k, v, cq, ck = make_inputs(dtype, qlens=(3,), klens=(256,))
    ids = (
        torch.tensor([0, 2], device="cuda", dtype=torch.int32)
        .view(1, 1, 2)
        .expand(2, 3, 2)
        .contiguous()
    )
    actual = op.sparse_attention(q, k, v, ids, cq, ck, 3)
    expected = sparse_attention_ref(q, k, v, ids, cq, ck)
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
    grad = torch.randn_like(actual)
    ga = torch.autograd.grad(actual, (q, k, v), grad)
    ge = torch.autograd.grad(expected, (q, k, v), grad)
    for a, e in zip(ga, ge):
        torch.testing.assert_close(a, e, atol=0.03, rtol=0.03)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_decode_and_metadata_mutation(dtype):
    q, k, v, cq, ck = make_inputs(dtype, qlens=(1,), klens=(256,))
    first = op.infllmv2_decode(q, k, v, ck, 256)
    second = op.infllmv2_attention(
        q, k, v, stage="decode", cu_seqlens_k=ck, max_seqlen_k=256
    )
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    assert first.data_ptr() != second.data_ptr()
    ck[1] = 128
    actual = op.infllmv2_decode(q, k[:128], v[:128], ck, 256)
    expected = dense_attention_ref(q, k, v, cq, ck)
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)


def test_generic_gqa_and_dispatch_contract():
    q, k, v, cq, ck = make_inputs(torch.float32, d=32, hq=16, hkv=4)
    actual = op.infllmv2_attention(q, k, v, cq, ck, 5, 8192)
    expected = dense_attention_ref(q, k, v, cq, ck)
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
    plan = op.NativePlan(
        (3, 5),
        (128, 192),
        dtype=torch.float32,
        head_dim=32,
        hq=16,
        hkv=4,
        max_seqlen_k=8192,
    )
    assert not plan.dense and plan.backend == "generic_hip"
    with pytest.raises(ValueError, match="exclusive"):
        op.infllmv2_attention(q, k, v, plan=plan, softmax_scale=0.5)
    with pytest.raises(ValueError, match="finite"):
        op.NativePlan(1, 128, softmax_scale=float("nan"))
