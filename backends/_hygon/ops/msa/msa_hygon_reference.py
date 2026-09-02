"""Independent MiniMax M3 MSA reference used only by tests.

This intentionally favors readability over speed.  Production code must import
``msa_hip`` directly and never dispatch to these PyTorch loops.
"""

from __future__ import annotations

import math

import torch


BLOCK = 128
DIM = 128


def _round_up_16(value: int) -> int:
    return (value + 15) // 16 * 16


def index_score_reference(
    idx_q: torch.Tensor,
    index_kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_seq_len: int,
) -> torch.Tensor:
    """Reference prefill score: max token dot product per causal 128-page."""
    total_q, heads, dim = idx_q.shape
    assert dim == DIM
    stride = _round_up_16(math.ceil(max_seq_len / BLOCK))
    score = torch.full(
        (heads, total_q, stride),
        -float("inf"),
        dtype=torch.float32,
        device=idx_q.device,
    )
    for request in range(seq_lens.numel()):
        q_start = int(cu_seqlens_q[request])
        q_end = int(cu_seqlens_q[request + 1])
        seq_len = int(seq_lens[request])
        prefix = int(prefix_lens[request])
        for local, qid in enumerate(range(q_start, q_end)):
            visible = min(seq_len, prefix + local + 1)
            for logical in range(math.ceil(visible / BLOCK)):
                page = int(block_table[request, logical])
                count = min(BLOCK, visible - logical * BLOCK)
                keys = index_kv_cache[page, :count].float()
                score[:, qid, logical] = torch.einsum(
                    "hd,td->ht", idx_q[qid].float(), keys
                ).amax(-1)
    return score


def index_decode_score_reference(
    idx_q: torch.Tensor,
    index_kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    decode_query_len: int,
    init_blocks: int = 0,
    local_blocks: int = 0,
) -> torch.Tensor:
    """Reference decode score for flattened request-major decode queries."""
    total_q, heads, dim = idx_q.shape
    assert dim == DIM and total_q == seq_lens.numel() * decode_query_len
    stride = _round_up_16(math.ceil(max_seq_len / BLOCK))
    score = torch.full(
        (heads, total_q, stride),
        -float("inf"),
        dtype=torch.float32,
        device=idx_q.device,
    )
    for request in range(seq_lens.numel()):
        seq_len = int(seq_lens[request])
        for local in range(decode_query_len):
            qid = request * decode_query_len + local
            visible = seq_len - decode_query_len + local + 1
            for logical in range(math.ceil(visible / BLOCK)):
                page = int(block_table[request, logical])
                count = min(BLOCK, visible - logical * BLOCK)
                keys = index_kv_cache[page, :count].float()
                score[:, qid, logical] = torch.einsum(
                    "hd,td->ht", idx_q[qid].float(), keys
                ).amax(-1)
            valid = math.ceil(visible / BLOCK)
            if init_blocks:
                score[:, qid, : min(init_blocks, valid)] = 1.0e30
            if local_blocks:
                score[:, qid, max(0, valid - local_blocks) : valid] = 1.0e29
    return score


def _select_row(
    row: torch.Tensor,
    valid_blocks: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
) -> torch.Tensor:
    result = torch.full((topk,), -1, dtype=torch.int32, device=row.device)
    selected = min(topk, valid_blocks)
    if selected == 0:
        return result
    if valid_blocks <= topk:
        result[:selected] = torch.arange(selected, device=row.device, dtype=torch.int32)
        return result
    values = row[:valid_blocks].clone()
    if init_blocks:
        values[: min(init_blocks, valid_blocks)] = 1.0e30
    if local_blocks:
        values[max(0, valid_blocks - local_blocks) :] = 1.0e29
    ids = torch.topk(values, selected).indices.to(torch.int32).sort().values
    result[:selected] = ids
    return result


def index_topk_reference(
    score: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    prefix_lens: torch.Tensor,
    topk: int,
    init_blocks: int,
    local_blocks: int,
) -> torch.Tensor:
    heads, total_q, _ = score.shape
    out = torch.full(
        (heads, total_q, topk), -1, dtype=torch.int32, device=score.device
    )
    for request in range(prefix_lens.numel()):
        q_start = int(cu_seqlens_q[request])
        q_end = int(cu_seqlens_q[request + 1])
        prefix = int(prefix_lens[request])
        for local, qid in enumerate(range(q_start, q_end)):
            valid = math.ceil((prefix + local + 1) / BLOCK)
            for head in range(heads):
                out[head, qid] = _select_row(
                    score[head, qid], valid, topk, init_blocks, local_blocks
                )
    return out


def index_decode_topk_reference(
    score: torch.Tensor,
    seq_lens: torch.Tensor,
    decode_query_len: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
) -> torch.Tensor:
    heads, total_q, _ = score.shape
    out = torch.full(
        (heads, total_q, topk), -1, dtype=torch.int32, device=score.device
    )
    for request in range(seq_lens.numel()):
        seq_len = int(seq_lens[request])
        for local in range(decode_query_len):
            qid = request * decode_query_len + local
            valid = math.ceil((seq_len - decode_query_len + local + 1) / BLOCK)
            for head in range(heads):
                out[head, qid] = _select_row(
                    score[head, qid], valid, topk, init_blocks, local_blocks
                )
    return out


def sparse_attention_reference(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Reference paged, causal GQA sparse attention for prefill."""
    output = torch.zeros_like(q)
    kv_heads = kv_cache.shape[1]
    group = q.shape[1] // kv_heads
    for request in range(seq_lens.numel()):
        q_start = int(cu_seqlens_q[request])
        q_end = int(cu_seqlens_q[request + 1])
        seq_len = int(seq_lens[request])
        prefix = int(prefix_lens[request])
        for local, qid in enumerate(range(q_start, q_end)):
            visible = min(seq_len, prefix + local + 1)
            _attention_row(
                q, output, kv_cache, topk_idx, block_table, request, qid,
                visible, kv_heads, group, scale
            )
    return output


def sparse_attention_decode_reference(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    decode_query_len: int,
    scale: float,
) -> torch.Tensor:
    output = torch.zeros_like(q)
    kv_heads = kv_cache.shape[1]
    group = q.shape[1] // kv_heads
    for request in range(seq_lens.numel()):
        seq_len = int(seq_lens[request])
        for local in range(decode_query_len):
            qid = request * decode_query_len + local
            visible = seq_len - decode_query_len + local + 1
            _attention_row(
                q, output, kv_cache, topk_idx, block_table, request, qid,
                visible, kv_heads, group, scale
            )
    return output


def _attention_row(
    q, output, kv_cache, topk_idx, block_table, request, qid, visible,
    kv_heads, group, scale
):
    for kv_head in range(kv_heads):
        keys, values = [], []
        for value in topk_idx[kv_head, qid]:
            logical = int(value)
            if logical < 0:
                continue
            count = min(BLOCK, visible - logical * BLOCK)
            if count <= 0:
                continue
            page = int(block_table[request, logical])
            keys.append(kv_cache[page, kv_head, :count, :DIM].float())
            values.append(kv_cache[page, kv_head, :count, DIM:].float())
        if not keys:
            continue
        key = torch.cat(keys)
        value = torch.cat(values)
        head_slice = slice(kv_head * group, (kv_head + 1) * group)
        logits = q[qid, head_slice].float() @ key.T * scale
        output[qid, head_slice] = (torch.softmax(logits, -1) @ value).to(q.dtype)
