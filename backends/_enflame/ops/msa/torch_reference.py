"""Official-stack Torch correctness reference for S60 MSA prefill.

The implementation uses ordinary PyTorch operators dispatched through
torch_gcu/TopsAten. It is a baseline rather than a fused vendor kernel.
"""

import torch
import torch.nn.functional as F


BLOCK_SIZE = 128


@torch.no_grad()
def torch_gcu_msa_prefill_b1(
    idx_q,
    index_kv_cache,
    q,
    kv_cache,
    block_table,
    seq_len: int,
    prefix_len: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    sm_scale: float,
):
    """Vectorized single-request BF16 MSA using official torch_gcu ops."""

    total_q, num_q_heads, head_dim = q.shape
    num_kv_heads = kv_cache.shape[1]
    num_idx_heads = idx_q.shape[1]
    group_size = num_q_heads // num_kv_heads

    if num_idx_heads != num_kv_heads:
        raise ValueError(
            "index heads must match KV heads"
        )

    logical_blocks = (
        seq_len + BLOCK_SIZE - 1
    ) // BLOCK_SIZE

    page_ids = (
        block_table[0, :logical_blocks]
        .to(torch.int32)
    )

    index_keys = torch.index_select(
        index_kv_cache,
        0,
        page_ids,
    )

    index_dim = idx_q.shape[-1]

    query_by_head = (
        idx_q.float()
        .permute(1, 0, 2)
        .contiguous()
    )

    key_flat = (
        index_keys.float()
        .reshape(
            logical_blocks * BLOCK_SIZE,
            index_dim,
        )
    )

    token_scores = torch.matmul(
        query_by_head,
        key_flat.transpose(0, 1),
    ).reshape(
        num_idx_heads,
        total_q,
        logical_blocks,
        BLOCK_SIZE,
    )

    query_positions = (
        torch.arange(
            total_q,
            device=q.device,
            dtype=torch.int32,
        )
        + prefix_len
    )

    key_positions = torch.arange(
        logical_blocks * BLOCK_SIZE,
        device=q.device,
        dtype=torch.int32,
    ).reshape(
        logical_blocks,
        BLOCK_SIZE,
    )

    causal = (
        key_positions[None, None, :, :]
        <= query_positions[None, :, None, None]
    )

    valid_key = (
        key_positions[None, None, :, :]
        < seq_len
    )

    token_scores = token_scores.masked_fill(
        ~(causal & valid_key),
        float("-inf"),
    )

    scores = token_scores.amax(dim=-1)

    block_ids = torch.arange(
        logical_blocks,
        device=q.device,
        dtype=torch.int32,
    ).view(
        1,
        1,
        logical_blocks,
    )

    valid_blocks = (
        (query_positions + BLOCK_SIZE)
        // BLOCK_SIZE
    ).clamp(
        max=logical_blocks,
    )

    current = scores.masked_fill(
        block_ids
        >= valid_blocks.view(1, total_q, 1),
        float("-inf"),
    )

    local_start = (
        valid_blocks - local_blocks
    ).clamp_min(0)

    local_mask = (
        block_ids
        >= local_start.view(1, total_q, 1)
    ) & (
        block_ids
        < valid_blocks.view(1, total_q, 1)
    )

    current = torch.where(
        local_mask,
        torch.full_like(current, 1.0e29),
        current,
    )

    init_end = torch.minimum(
        valid_blocks,
        torch.full_like(
            valid_blocks,
            init_blocks,
        ),
    )

    init_mask = (
        block_ids
        < init_end.view(1, total_q, 1)
    )

    current = torch.where(
        init_mask & (current < 1.0e29),
        torch.full_like(current, 1.0e30),
        current,
    )

    select_k = min(topk, logical_blocks)

    selected = torch.topk(
        current,
        select_k,
        dim=-1,
    ).indices.to(torch.int32)

    selected_count = valid_blocks.clamp(
        max=select_k,
    )

    selected_slots = torch.arange(
        select_k,
        device=q.device,
        dtype=torch.int32,
    ).view(1, 1, select_k)

    selected = torch.where(
        selected_slots
        < selected_count.view(1, total_q, 1),
        selected,
        torch.full_like(selected, -1),
    )

    if select_k < topk:
        selected = torch.cat(
            (
                selected,
                torch.full(
                    (
                        num_idx_heads,
                        total_q,
                        topk - select_k,
                    ),
                    -1,
                    device=q.device,
                    dtype=torch.int32,
                ),
            ),
            dim=-1,
        )

    logical = (
        selected.permute(1, 0, 2)
        .contiguous()
    )

    valid_slots = logical >= 0
    safe_logical = logical.clamp_min(0)

    selected_page_ids = torch.index_select(
        block_table[0],
        0,
        safe_logical.reshape(-1),
    ).reshape(
        total_q,
        num_kv_heads,
        topk,
    )

    head_ids = torch.arange(
        num_kv_heads,
        device=q.device,
        dtype=torch.int32,
    ).view(
        1,
        num_kv_heads,
        1,
    )

    linear_page_heads = (
        selected_page_ids * num_kv_heads
        + head_ids
    ).to(torch.int32)

    packed_dim = kv_cache.shape[-1]

    if packed_dim != 2 * head_dim:
        raise ValueError(
            "BF16 KV cache must pack K and V "
            "in the final dimension"
        )

    flat_cache = kv_cache.reshape(
        kv_cache.shape[0] * num_kv_heads,
        BLOCK_SIZE,
        packed_dim,
    )

    selected_kv = torch.index_select(
        flat_cache,
        0,
        linear_page_heads.reshape(-1),
    ).reshape(
        total_q,
        num_kv_heads,
        topk,
        BLOCK_SIZE,
        packed_dim,
    )

    selected_k = selected_kv[..., :head_dim]
    selected_v = selected_kv[..., head_dim:]

    selected_k = selected_k.reshape(
        total_q,
        num_kv_heads,
        topk * BLOCK_SIZE,
        head_dim,
    )

    selected_v = selected_v.reshape(
        total_q,
        num_kv_heads,
        topk * BLOCK_SIZE,
        head_dim,
    )

    query_grouped = q.reshape(
        total_q,
        num_kv_heads,
        group_size,
        head_dim,
    )

    logits = torch.matmul(
        query_grouped,
        selected_k.transpose(-1, -2),
    ).float()

    logits *= sm_scale

    token_offsets = torch.arange(
        BLOCK_SIZE,
        device=q.device,
        dtype=torch.int32,
    )

    key_positions = (
        logical[..., None] * BLOCK_SIZE
        + token_offsets.view(
            1,
            1,
            1,
            BLOCK_SIZE,
        )
    ).reshape(
        total_q,
        num_kv_heads,
        topk * BLOCK_SIZE,
    )

    query_positions = (
        torch.arange(
            total_q,
            device=q.device,
            dtype=torch.int32,
        )
        + prefix_len
    ).view(
        total_q,
        1,
        1,
    )

    mask = (
        valid_slots[..., None]
        .expand(
            total_q,
            num_kv_heads,
            topk,
            BLOCK_SIZE,
        )
        .reshape(
            total_q,
            num_kv_heads,
            topk * BLOCK_SIZE,
        )
    )

    mask = (
        mask
        & (key_positions <= query_positions)
        & (key_positions < seq_len)
    )

    logits = logits.masked_fill(
        ~mask[:, :, None, :],
        float("-inf"),
    )

    probabilities = torch.softmax(
        logits,
        dim=-1,
    ).to(selected_v.dtype)

    output = torch.matmul(
        probabilities,
        selected_v,
    ).reshape(
        total_q,
        num_q_heads,
        head_dim,
    )

    return scores, selected, output


@torch.no_grad()
def torch_gcu_msa_decode(
    idx_q: torch.Tensor,
    index_kv_cache: torch.Tensor,
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    decode_query_len: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    num_kv_heads: int,
    sm_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized BF16 MSA decode using torch_gcu/TopsAten.

    The query rows must be packed in request-major order. For each
    request they represent the final ``decode_query_len`` positions
    in the corresponding sequence.
    """
    if decode_query_len <= 0:
        raise ValueError(
            "decode_query_len must be positive"
        )

    batch = seq_lens.numel()
    total_q, num_heads, head_dim = q.shape

    if total_q != batch * decode_query_len:
        raise ValueError(
            "q rows must equal batch * decode_query_len"
        )

    if idx_q.shape[:2] != (
        total_q,
        num_kv_heads,
    ):
        raise ValueError(
            "idx_q has an incompatible shape"
        )

    if num_heads % num_kv_heads != 0:
        raise ValueError(
            "query heads must be divisible by KV heads"
        )

    if block_table.shape[0] != batch:
        raise ValueError(
            "block_table batch dimension is incompatible"
        )

    if kv_cache.shape[1] != num_kv_heads:
        raise ValueError(
            "kv_cache has an incompatible KV-head count"
        )

    if kv_cache.shape[2] != BLOCK_SIZE:
        raise ValueError(
            f"kv_cache block size must be {BLOCK_SIZE}"
        )

    if kv_cache.shape[-1] != 2 * head_dim:
        raise ValueError(
            "BF16 KV cache must pack K and V "
            "in the final dimension"
        )

    if torch.any(seq_lens < decode_query_len).item():
        raise ValueError(
            "each sequence must contain all decode queries"
        )

    max_blocks = block_table.shape[1]

    if topk <= 0 or topk > max_blocks:
        raise ValueError(
            "topk must be in [1, max_blocks]"
        )

    group_size = num_heads // num_kv_heads
    index_dim = idx_q.shape[-1]

    query_offsets = torch.arange(
        decode_query_len,
        dtype=torch.int32,
        device=q.device,
    )

    query_positions = (
        seq_lens.to(torch.int32)[:, None]
        - decode_query_len
        + query_offsets[None, :]
    )

    valid_blocks = (
        query_positions + BLOCK_SIZE
    ) // BLOCK_SIZE

    valid_blocks = valid_blocks.clamp(
        min=0,
        max=max_blocks,
    )

    token_ids = torch.arange(
        max_blocks * BLOCK_SIZE,
        dtype=torch.int32,
        device=q.device,
    )

    block_ids = torch.arange(
        max_blocks,
        dtype=torch.int32,
        device=q.device,
    )

    physical_pages = (
        block_table[:, :max_blocks]
        .contiguous()
    )

    index_keys = torch.index_select(
        index_kv_cache,
        0,
        physical_pages.reshape(-1),
    ).reshape(
        batch,
        max_blocks,
        BLOCK_SIZE,
        index_dim,
    )

    index_queries = idx_q.reshape(
        batch,
        decode_query_len,
        num_kv_heads,
        index_dim,
    ).permute(
        0,
        2,
        1,
        3,
    ).contiguous()

    index_keys_flat = index_keys.reshape(
        batch,
        max_blocks * BLOCK_SIZE,
        index_dim,
    )

    token_scores = torch.matmul(
        index_queries,
        index_keys_flat[:, None].transpose(
            -1,
            -2,
        ),
    ).float()

    visible_tokens = (
        token_ids.view(1, 1, 1, -1)
        <= query_positions[:, None, :, None]
    )

    token_scores = token_scores.masked_fill(
        ~visible_tokens,
        float("-inf"),
    )

    scores = token_scores.reshape(
        batch,
        num_kv_heads,
        decode_query_len,
        max_blocks,
        BLOCK_SIZE,
    ).amax(dim=-1)

    visible_blocks = (
        block_ids.view(1, 1, 1, max_blocks)
        < valid_blocks[:, None, :, None]
    )

    scores = scores.masked_fill(
        ~visible_blocks,
        -1.0e30,
    )

    local_start = (
        valid_blocks - local_blocks
    ).clamp_min(0)

    local_mask = (
        visible_blocks
        & (
            block_ids.view(1, 1, 1, max_blocks)
            >= local_start[:, None, :, None]
        )
    )

    scores = torch.where(
        local_mask,
        torch.full_like(scores, 1.0e29),
        scores,
    )

    init_mask = (
        visible_blocks
        & (
            block_ids.view(1, 1, 1, max_blocks)
            < init_blocks
        )
    )

    scores = torch.where(
        init_mask & (scores < 1.0e29),
        torch.full_like(scores, 1.0e30),
        scores,
    )

    selected = torch.topk(
        scores,
        k=topk,
        dim=-1,
    ).indices.to(torch.int32)

    slot_ids = torch.arange(
        topk,
        dtype=torch.int32,
        device=q.device,
    ).view(1, 1, 1, topk)

    selected = torch.where(
        slot_ids
        < valid_blocks.clamp(
            max=topk
        )[:, None, :, None],
        selected,
        torch.full_like(selected, -1),
    )

    topk_idx = selected.permute(
        1,
        0,
        2,
        3,
    ).reshape(
        num_kv_heads,
        total_q,
        topk,
    ).contiguous()

    logical = topk_idx.permute(
        1,
        0,
        2,
    ).reshape(
        batch,
        decode_query_len,
        num_kv_heads,
        topk,
    ).contiguous()

    valid_slots = logical >= 0
    safe_logical = logical.clamp_min(0)

    row_offsets = (
        torch.arange(
            batch,
            dtype=torch.int32,
            device=q.device,
        )
        * block_table.shape[1]
    ).view(batch, 1, 1, 1)

    pages = torch.index_select(
        block_table.reshape(-1),
        0,
        (
            safe_logical + row_offsets
        ).reshape(-1),
    ).reshape(
        batch,
        decode_query_len,
        num_kv_heads,
        topk,
    )

    flat_cache = kv_cache.reshape(
        kv_cache.shape[0] * num_kv_heads,
        BLOCK_SIZE,
        2 * head_dim,
    )

    head_ids = torch.arange(
        num_kv_heads,
        dtype=torch.int32,
        device=q.device,
    ).view(1, 1, num_kv_heads, 1)

    linear_pages = (
        pages * num_kv_heads
        + head_ids
    ).to(torch.int32)

    selected_cache = torch.index_select(
        flat_cache,
        0,
        linear_pages.reshape(-1),
    ).reshape(
        batch,
        decode_query_len,
        num_kv_heads,
        topk,
        BLOCK_SIZE,
        2 * head_dim,
    )

    keys = selected_cache[..., :head_dim].reshape(
        batch,
        decode_query_len,
        num_kv_heads,
        topk * BLOCK_SIZE,
        head_dim,
    )

    values = selected_cache[..., head_dim:].reshape(
        batch,
        decode_query_len,
        num_kv_heads,
        topk * BLOCK_SIZE,
        head_dim,
    )

    grouped_queries = q.reshape(
        batch,
        decode_query_len,
        num_kv_heads,
        group_size,
        head_dim,
    )

    logits = torch.matmul(
        grouped_queries,
        keys.transpose(-1, -2),
    ).float()

    logits *= sm_scale

    token_offsets = torch.arange(
        BLOCK_SIZE,
        dtype=torch.int32,
        device=q.device,
    )

    selected_positions = (
        logical[..., None] * BLOCK_SIZE
        + token_offsets.view(
            1,
            1,
            1,
            1,
            BLOCK_SIZE,
        )
    ).reshape(
        batch,
        decode_query_len,
        num_kv_heads,
        topk * BLOCK_SIZE,
    )

    attention_mask = (
        valid_slots[..., None]
        .expand(
            batch,
            decode_query_len,
            num_kv_heads,
            topk,
            BLOCK_SIZE,
        )
        .reshape(
            batch,
            decode_query_len,
            num_kv_heads,
            topk * BLOCK_SIZE,
        )
    )

    attention_mask = (
        attention_mask
        & (
            selected_positions
            <= query_positions[:, :, None, None]
        )
        & (
            selected_positions
            < seq_lens[:, None, None, None]
        )
    )

    logits = logits.masked_fill(
        ~attention_mask[:, :, :, None, :],
        float("-inf"),
    )

    probabilities = torch.softmax(
        logits,
        dim=-1,
    ).to(values.dtype)

    output = torch.matmul(
        probabilities,
        values,
    ).reshape(
        total_q,
        num_heads,
        head_dim,
    )

    return topk_idx, output
