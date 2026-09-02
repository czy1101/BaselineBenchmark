"""Correctness tests for the Hygon MiniMax M3 MSA BF16 backend."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from msa_hip import (
    load_extension,
    minimax_m3_index_decode,
    minimax_m3_index_decode_score,
    minimax_m3_index_score,
    minimax_m3_index_topk,
    minimax_m3_sparse_attn,
    minimax_m3_sparse_attn_decode,
    sparse_backend_name,
)
from msa_hygon_reference import (
    index_decode_score_reference,
    index_decode_topk_reference,
    index_score_reference,
    index_topk_reference,
    sparse_attention_decode_reference,
    sparse_attention_reference,
)


BLOCK = 128
DIM = 128


@dataclass
class Data:
    idx_q: torch.Tensor
    q: torch.Tensor
    index_cache: torch.Tensor
    kv_cache: torch.Tensor
    table: torch.Tensor
    cu_q: torch.Tensor
    seq_lens: torch.Tensor
    prefix_lens: torch.Tensor
    max_query: int
    max_seq: int


def make_data(seq_lens, prefix_lens, query_lens, kv_heads=2, group=4) -> Data:
    assert len(seq_lens) == len(prefix_lens) == len(query_lens)
    torch.manual_seed(20260826)
    device = "cuda"
    batch = len(seq_lens)
    max_seq = max(seq_lens)
    pages_per_req = math.ceil(max_seq / BLOCK)
    num_pages = batch * pages_per_req
    permutation = torch.randperm(num_pages, device=device, dtype=torch.int64)
    table = permutation.view(batch, pages_per_req).to(torch.int32).contiguous()
    index_cache = (
        torch.randn(num_pages, BLOCK, DIM, device=device) * 0.125
    ).to(torch.bfloat16)
    kv_cache = (
        torch.randn(num_pages, kv_heads, BLOCK, 2 * DIM, device=device) * 0.125
    ).to(torch.bfloat16)
    total_q = sum(query_lens)
    idx_q = (
        torch.randn(total_q, kv_heads, DIM, device=device) * 0.125
    ).to(torch.bfloat16)
    q = (
        torch.randn(total_q, kv_heads * group, DIM, device=device) * 0.125
    ).to(torch.bfloat16)
    cu = [0]
    for length in query_lens:
        cu.append(cu[-1] + length)
    return Data(
        idx_q=idx_q.contiguous(),
        q=q.contiguous(),
        index_cache=index_cache.contiguous(),
        kv_cache=kv_cache.contiguous(),
        table=table,
        cu_q=torch.tensor(cu, device=device, dtype=torch.int32),
        seq_lens=torch.tensor(seq_lens, device=device, dtype=torch.int32),
        prefix_lens=torch.tensor(prefix_lens, device=device, dtype=torch.int32),
        max_query=max(query_lens),
        max_seq=max_seq,
    )


def assert_score(actual, expected, data, decode=False, decode_len=0):
    for request in range(data.seq_lens.numel()):
        q_start = request * decode_len if decode else int(data.cu_q[request])
        q_len = decode_len if decode else int(data.cu_q[request + 1] - data.cu_q[request])
        for local in range(q_len):
            qid = q_start + local
            pos = (
                int(data.seq_lens[request]) - decode_len + local
                if decode
                else int(data.prefix_lens[request]) + local
            )
            valid = math.ceil((pos + 1) / BLOCK)
            torch.testing.assert_close(
                actual[:, qid, :valid],
                expected[:, qid, :valid],
                atol=3.0e-2,
                rtol=3.0e-2,
            )


def assert_topk_set(actual, expected):
    torch.testing.assert_close(
        actual.sort(dim=-1).values,
        expected.sort(dim=-1).values,
        atol=0,
        rtol=0,
    )


def test_prefill_pipeline():
    data = make_data((1025, 769), (1000, 512), (3, 4))
    heads = data.idx_q.shape[1]
    topk = 4
    score = minimax_m3_index_score(
        data.idx_q,
        data.index_cache,
        data.table,
        data.cu_q,
        data.seq_lens,
        data.prefix_lens,
        data.max_query,
        data.max_seq,
        heads,
    )
    ref_score = index_score_reference(
        data.idx_q,
        data.index_cache,
        data.table,
        data.cu_q,
        data.seq_lens,
        data.prefix_lens,
        data.max_seq,
    )
    assert_score(score, ref_score, data)

    sentinel = torch.full(
        (heads, data.q.shape[0] + 1, topk),
        -7,
        device="cuda",
        dtype=torch.int32,
    )
    selected = minimax_m3_index_topk(
        score, data.cu_q, data.prefix_lens, data.max_query,
        topk, 1, 2, out=sentinel
    )
    expected = index_topk_reference(score, data.cu_q, data.prefix_lens, topk, 1, 2)
    assert selected.data_ptr() == sentinel.data_ptr()
    assert torch.all(sentinel[:, -1] == -7)
    assert_topk_set(selected, expected)

    output = torch.empty_like(data.q)
    minimax_m3_sparse_attn(
        data.q,
        data.kv_cache,
        selected,
        data.table,
        data.cu_q,
        data.seq_lens,
        data.prefix_lens,
        data.max_query,
        heads,
        DIM**-0.5,
        output,
    )
    ref = sparse_attention_reference(
        data.q, data.kv_cache, expected, data.table, data.cu_q,
        data.seq_lens, data.prefix_lens, DIM**-0.5
    )
    torch.testing.assert_close(output, ref, atol=4.0e-2, rtol=4.0e-2)


def test_decode_pipeline():
    decode_len = 2
    data = make_data((1025, 769), (0, 0), (decode_len, decode_len))
    heads = data.idx_q.shape[1]
    topk = 4
    ref_score = index_decode_score_reference(
        data.idx_q, data.index_cache, data.table, data.seq_lens,
        data.max_seq, decode_len, 1, 2
    )
    expected = index_decode_topk_reference(
        ref_score, data.seq_lens, decode_len, topk, 1, 2
    )
    max_blocks = math.ceil(data.max_seq / BLOCK)
    score_storage = torch.full(
        (heads, max_blocks, data.q.shape[0] + 1),
        float("nan"), device="cuda", dtype=torch.float32
    )
    score_buffer = score_storage.transpose(1, 2)
    actual_score = minimax_m3_index_decode_score(
        data.idx_q, data.index_cache, data.table, data.seq_lens,
        data.max_seq, 1, 2, heads, decode_len, decode_len,
        score_out=score_buffer
    )
    assert actual_score.data_ptr() == score_buffer.data_ptr()
    assert_score(actual_score, ref_score, data, decode=True, decode_len=decode_len)
    assert torch.all(torch.isnan(score_buffer[:, -1]))

    topk_storage = torch.full(
        (heads, topk, data.q.shape[0] + 1),
        -9, device="cuda", dtype=torch.int32
    )
    topk_buffer = topk_storage.transpose(1, 2)
    selected = minimax_m3_index_decode(
        data.idx_q,
        data.index_cache,
        data.table,
        data.seq_lens,
        data.max_seq,
        topk,
        1,
        2,
        heads,
        decode_len,
        decode_len,
        out=topk_buffer,
        score_out=score_buffer,
    )
    assert selected.data_ptr() == topk_buffer.data_ptr()
    assert torch.all(topk_buffer[:, -1] == -9)
    assert_topk_set(selected, expected)

    output = torch.empty_like(data.q)
    minimax_m3_sparse_attn_decode(
        data.q,
        data.kv_cache,
        selected,
        data.table,
        data.seq_lens,
        heads,
        DIM**-0.5,
        output,
        decode_len,
    )
    ref = sparse_attention_decode_reference(
        data.q, data.kv_cache, expected, data.table, data.seq_lens,
        decode_len, DIM**-0.5
    )
    torch.testing.assert_close(output, ref, atol=4.0e-2, rtol=4.0e-2)


def test_decode_b1_hybrid():
    """Cover the BW1000 B=1 decode dispatch that intentionally uses HIP."""
    decode_len = 1
    data = make_data((257,), (0,), (decode_len,))
    heads = data.idx_q.shape[1]
    topk = 4
    ref_score = index_decode_score_reference(
        data.idx_q, data.index_cache, data.table, data.seq_lens,
        data.max_seq, decode_len, 1, 2
    )
    expected = index_decode_topk_reference(
        ref_score, data.seq_lens, decode_len, topk, 1, 2
    )
    selected = minimax_m3_index_decode(
        data.idx_q, data.index_cache, data.table, data.seq_lens,
        data.max_seq, topk, 1, 2, heads, decode_len, decode_len
    )
    assert_topk_set(selected, expected)
    output = torch.empty_like(data.q)
    minimax_m3_sparse_attn_decode(
        data.q, data.kv_cache, selected, data.table, data.seq_lens,
        heads, DIM**-0.5, output, decode_len
    )
    ref = sparse_attention_decode_reference(
        data.q, data.kv_cache, expected, data.table, data.seq_lens,
        decode_len, DIM**-0.5
    )
    torch.testing.assert_close(output, ref, atol=4.0e-2, rtol=4.0e-2)


def test_topk_identity_and_padding():
    score = torch.full((1, 1, 16), -float("inf"), device="cuda")
    cu = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    prefix = torch.tensor([128], device="cuda", dtype=torch.int32)
    actual = minimax_m3_index_topk(score, cu, prefix, 1, 4, 1, 2)
    expected = torch.tensor([[[0, 1, -1, -1]]], device="cuda", dtype=torch.int32)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("A Hygon DTK/HIP device is required")
    print("MSA extension:", load_extension(False).__name__)
    print("MSA sparse backend:", sparse_backend_name())
    test_topk_identity_and_padding()
    test_prefill_pipeline()
    test_decode_pipeline()
    test_decode_b1_hybrid()
    torch.cuda.synchronize()
    print("MSA Hygon BF16 correctness: PASS")
