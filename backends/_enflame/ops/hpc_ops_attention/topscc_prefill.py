# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""S60 TopsCC compatibility baseline for dynamic FP8 prefill attention.

The native S60 kernel vendored with this repository implements paged BF16
attention.  This module lowers the HPC-Ops FP8 prefill contract to that
kernel: Q/K/V are decoded and scaled on GCU, causal query rows are represented
as independent paged-attention sequences, and a block-sparse mask is lowered
to compact per-head page tables.

Input conversion remains inside ``TopsCCPrefillPlan.__call__`` so benchmark
timings cover the complete compatibility path.  Static page-table lowering is
performed once by ``prepare_topscc_prefill_fp8`` just like ordinary operator
workspace preparation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .topscc_decode import (
    TopsCCDecodePlan,
    topscc_decode_available,
)
from .fp8_utils import _raw_fp8_bits, decode_fp8e4m3fn


_PAGE_SIZES = (32, 64)
_HEAD_DIM = 128
BLOCK_SIZE = 128


def _normalize_quant_type(quant_type) -> str:
    """Normalize the two public FP8 prefill quantization modes."""

    value = getattr(quant_type, "value", quant_type)
    if value == 0:
        return "qkpertoken_perhead_vperhead"
    if value == 1:
        return "qpertoken_perhead_kvpertensor"
    normalized = str(value).lower()
    aliases = {
        "qkpertoken_perhead_vperhead": "qkpertoken_perhead_vperhead",
        "qpertoken_perhead_kvpertensor": "qpertoken_perhead_kvpertensor",
        "qpertoken_perhead_kpertoken_perhead_vperhead": "qkpertoken_perhead_vperhead",
    }
    if normalized in aliases:
        return aliases[normalized]
    raise ValueError(f"unsupported prefill FP8 quant_type: {quant_type!r}")


def topscc_prefill_available() -> bool:
    """Return whether the native S60 TopsCC attention kernel is available."""

    return topscc_decode_available()


def _logical_cache(cache: torch.Tensor, layout: str) -> torch.Tensor:
    layout = layout.upper()
    if layout not in {"NHD", "HND"}:
        raise ValueError("layout must be 'NHD' or 'HND'")
    if cache.ndim != 4:
        raise ValueError("K/V cache must be rank 4")

    # FlagAttention's HND input keeps the public [page, token, head, dim]
    # shape and represents HND through strides.  Also accept an explicit
    # [page, head, token, dim] tensor for standalone callers.
    if cache.shape[1] in _PAGE_SIZES:
        return cache
    if layout == "HND" and cache.shape[2] in _PAGE_SIZES:
        return cache.permute(0, 2, 1, 3)
    raise ValueError("paged K/V cache must use page size 32 or 64")


def _unpack_all_k_scales(
    k_scale: torch.Tensor,
    *,
    num_pages: int,
    page_size: int,
    num_kv_heads: int,
) -> torch.Tensor:
    """Expose per-token K scales as [page, token, kv_head] FP32."""

    if (
        k_scale.dtype == torch.float32
        and k_scale.ndim == 3
        and tuple(k_scale.shape) == (num_pages, page_size, num_kv_heads)
    ):
        return k_scale

    raw = _raw_fp8_bits(k_scale)
    unpacked = raw.contiguous().view(torch.float32)
    expected = (num_pages, page_size // 32, num_kv_heads, 32)
    if tuple(unpacked.shape) != expected:
        raise ValueError(
            "packed K scales must decode to "
            f"{expected}, got {tuple(unpacked.shape)}"
        )
    return unpacked.permute(0, 1, 3, 2).reshape(
        num_pages, page_size, num_kv_heads
    )


def _selected_mask_tile(
    mask: torch.Tensor,
    batch: int,
    head: int,
    q_tile: int,
    kv_tile: int,
) -> bool:
    represented = int(mask.shape[3])
    if kv_tile < represented:
        return bool(mask[batch, head, q_tile, kv_tile].item())
    # Keep parity with the Triton/reference compatibility rule for a mask
    # shorter than the KV range: exactly one following tile remains visible.
    return kv_tile == represented


@dataclass(frozen=True)
class _LoweredMetadata:
    dense_tables: torch.Tensor | None
    dense_lens: torch.Tensor | None
    sparse_tables: tuple[torch.Tensor, ...]
    sparse_lens: tuple[torch.Tensor, ...]
    maximum_context: int


def _lower_page_tables(
    cu_seqlens_q: torch.Tensor,
    block_ids: torch.Tensor,
    kv_lens: torch.Tensor,
    block_mask: torch.Tensor | None,
    *,
    num_query_heads: int,
    page_size: int,
    device: torch.device,
) -> _LoweredMetadata:
    q_offsets = cu_seqlens_q.detach().cpu().to(torch.int64).tolist()
    lengths = kv_lens.detach().cpu().to(torch.int64).tolist()
    pages_cpu = block_ids.detach().cpu().to(torch.int32)
    mask_cpu = None if block_mask is None else block_mask.detach().cpu()

    dense_rows: list[list[int]] = []
    dense_lengths: list[int] = []
    sparse_rows: list[list[list[int]]] = [
        [] for _ in range(num_query_heads)
    ]
    sparse_lengths: list[list[int]] = [
        [] for _ in range(num_query_heads)
    ]

    maximum_context = 0
    for batch, kv_len_value in enumerate(lengths):
        q_len = int(q_offsets[batch + 1] - q_offsets[batch])
        kv_len = int(kv_len_value)
        q_start = kv_len - q_len
        for q_position in range(q_len):
            causal_length = q_start + q_position + 1
            visible_pages = math.ceil(causal_length / page_size)
            physical = [
                int(value)
                for value in pages_cpu[batch, :visible_pages].tolist()
            ]

            if mask_cpu is None:
                dense_rows.append(physical)
                dense_lengths.append(causal_length)
                maximum_context = max(maximum_context, causal_length)
                continue

            q_tile = q_position // BLOCK_SIZE
            final_page_tokens = causal_length % page_size or page_size
            for head in range(num_query_heads):
                compact: list[int] = []
                compact_length = 0
                for logical_page, physical_page in enumerate(physical):
                    if not _selected_mask_tile(
                        mask_cpu,
                        batch,
                        head,
                        q_tile,
                        (logical_page * page_size) // BLOCK_SIZE,
                    ):
                        continue
                    compact.append(physical_page)
                    compact_length += (
                        final_page_tokens
                        if logical_page == visible_pages - 1
                        else page_size
                    )
                if not compact:
                    raise ValueError(
                        "block_mask leaves a query/head row without visible KV tokens"
                    )
                sparse_rows[head].append(compact)
                sparse_lengths[head].append(compact_length)
                maximum_context = max(maximum_context, compact_length)

    def make_tables(rows: list[list[int]]) -> torch.Tensor:
        width = max(map(len, rows))
        table = torch.zeros((len(rows), width), dtype=torch.int32)
        for row_id, row in enumerate(rows):
            table[row_id, : len(row)] = torch.tensor(row, dtype=torch.int32)
        return table.to(device)

    if mask_cpu is None:
        return _LoweredMetadata(
            dense_tables=make_tables(dense_rows),
            dense_lens=torch.tensor(
                dense_lengths, dtype=torch.int32, device=device
            ),
            sparse_tables=(),
            sparse_lens=(),
            maximum_context=maximum_context,
        )

    return _LoweredMetadata(
        dense_tables=None,
        dense_lens=None,
        sparse_tables=tuple(make_tables(rows) for rows in sparse_rows),
        sparse_lens=tuple(
            torch.tensor(values, dtype=torch.int32, device=device)
            for values in sparse_lengths
        ),
        maximum_context=maximum_context,
    )


@dataclass(frozen=True)
class TopsCCPrefillPlan:
    """Prepared metadata for the full S60 TopsCC FP8 prefill baseline."""

    q: torch.Tensor
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    token_q_scale: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    quant_type: str
    metadata: _LoweredMetadata
    num_kv_heads: int
    page_size: int
    softmax_scale: float

    def _decode_inputs(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = decode_fp8e4m3fn(self.q) * self.token_q_scale[..., None]
        query = query.to(torch.bfloat16).contiguous()

        key = decode_fp8e4m3fn(self.key_cache)
        value = decode_fp8e4m3fn(self.value_cache)
        if self.quant_type == "qkpertoken_perhead_vperhead":
            scales = _unpack_all_k_scales(
                self.k_scale,
                num_pages=int(key.shape[0]),
                page_size=self.page_size,
                num_kv_heads=self.num_kv_heads,
            )
            key = key * scales[..., None]
            value = value * self.v_scale.reshape(1, 1, -1, 1)
        else:
            key = key * self.k_scale.reshape(-1)[0]
            value = value * self.v_scale.reshape(-1)[0]

        # Native paged attention consumes [page, kv_head, token, dim].
        key = key.permute(0, 2, 1, 3).to(torch.bfloat16).contiguous()
        value = value.permute(0, 2, 1, 3).to(torch.bfloat16).contiguous()
        return query, key, value

    def __call__(self) -> torch.Tensor:
        query, key, value = self._decode_inputs()
        if self.metadata.dense_tables is not None:
            plan = TopsCCDecodePlan(
                query=query,
                key_cache=key,
                value_cache=value,
                block_tables=self.metadata.dense_tables,
                context_lens=self.metadata.dense_lens,
                max_context_len=self.metadata.maximum_context,
                softmax_scale=self.softmax_scale,
            )
            return plan()

        # A block mask is specified per query head, whereas the native paged
        # kernel accepts one page table per sequence.  Lower each head to a
        # one-head launch and concatenate in the original Q-head order.
        outputs = []
        heads_per_kv = int(query.shape[1]) // self.num_kv_heads
        packed_by_kv_head = tuple(
            (
                key[:, head : head + 1].contiguous(),
                value[:, head : head + 1].contiguous(),
            )
            for head in range(self.num_kv_heads)
        )
        for head in range(int(query.shape[1])):
            kv_head = head // heads_per_kv
            head_key, head_value = packed_by_kv_head[kv_head]
            plan = TopsCCDecodePlan(
                query=query[:, head : head + 1].contiguous(),
                key_cache=head_key,
                value_cache=head_value,
                block_tables=self.metadata.sparse_tables[head],
                context_lens=self.metadata.sparse_lens[head],
                max_context_len=self.metadata.maximum_context,
                softmax_scale=self.softmax_scale,
            )
            outputs.append(plan())
        return torch.cat(outputs, dim=1)


def prepare_topscc_prefill_fp8(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    block_ids: torch.Tensor,
    kv_lens: torch.Tensor,
    max_q_len: int,
    quant_type=1,
    block_mask: torch.Tensor | None = None,
    *,
    layout: str = "NHD",
    softmax_scale: float | None = None,
) -> TopsCCPrefillPlan:
    """Validate inputs and prepare static TopsCC prefill launch metadata."""

    normalized_quant = _normalize_quant_type(quant_type)
    logical_k = _logical_cache(key_cache, layout)
    logical_v = _logical_cache(value_cache, layout)
    if logical_k.shape != logical_v.shape:
        raise ValueError("K and V cache shapes must match")
    if q.ndim != 3 or q.shape[-1] != _HEAD_DIM:
        raise ValueError("Q must have shape [total_q, num_query_heads, 128]")
    if logical_k.shape[-1] != _HEAD_DIM:
        raise ValueError("K/V head dimension must be 128")
    if q.dtype != torch.float8_e4m3fn:
        raise TypeError("Q must use torch.float8_e4m3fn")
    if (
        logical_k.dtype != torch.float8_e4m3fn
        or logical_v.dtype != torch.float8_e4m3fn
    ):
        raise TypeError("K/V cache must use torch.float8_e4m3fn")
    if q.device.type != "gcu":
        raise ValueError("the TopsCC prefill baseline requires GCU tensors")
    tensors = (
        logical_k,
        logical_v,
        q_scale,
        k_scale,
        v_scale,
        cu_seqlens_q,
        block_ids,
        kv_lens,
    )
    if any(tensor.device != q.device for tensor in tensors):
        raise ValueError("all prefill inputs must be on the same GCU device")
    if cu_seqlens_q.dtype != torch.int32:
        raise TypeError("cu_seqlens_q must be int32")
    if block_ids.dtype != torch.int32 or kv_lens.dtype != torch.int32:
        raise TypeError("block_ids and kv_lens must be int32")
    if cu_seqlens_q.ndim != 1 or block_ids.ndim != 2 or kv_lens.ndim != 1:
        raise ValueError("invalid sequence metadata rank")
    if q_scale.dtype != torch.float32 or v_scale.dtype != torch.float32:
        raise TypeError("q_scale and v_scale must be float32")
    batch = int(kv_lens.numel())
    if cu_seqlens_q.numel() != batch + 1 or block_ids.shape[0] != batch:
        raise ValueError("sequence metadata batch dimensions do not match")
    if q_scale.shape[:2] != (batch, q.shape[1]):
        raise ValueError("q_scale must start with [batch, num_query_heads]")
    if q_scale.ndim != 3 or q_scale.shape[2] < max_q_len:
        raise ValueError("q_scale must cover max_q_len tokens")
    if q.shape[1] % logical_k.shape[2]:
        raise ValueError("query heads must be divisible by KV heads")
    if normalized_quant == "qkpertoken_perhead_vperhead":
        if k_scale.element_size() not in (1, 4):
            raise TypeError("per-token K scale must use FP32 or its FP8 byte view")
        if v_scale.numel() != logical_k.shape[2]:
            raise ValueError("per-head V scale must contain num_kv_heads values")
    elif (
        k_scale.dtype != torch.float32
        or k_scale.numel() < 1
        or v_scale.numel() < 1
    ):
        raise TypeError("per-tensor K/V mode requires float32 scalar scales")
    if max_q_len <= 0:
        raise ValueError("max_q_len must be positive")

    q_offsets = cu_seqlens_q.detach().cpu().to(torch.int64).tolist()
    length_values = kv_lens.detach().cpu().to(torch.int64).tolist()
    if int(q_offsets[-1]) != q.shape[0]:
        raise ValueError("cu_seqlens_q does not cover every query token")
    for batch_id, kv_len in enumerate(length_values):
        q_len = int(q_offsets[batch_id + 1] - q_offsets[batch_id])
        if q_len <= 0 or q_len > max_q_len or kv_len < q_len:
            raise ValueError(
                "each sequence must satisfy 0 < q_len <= max_q_len and q_len <= kv_len"
            )
        if math.ceil(kv_len / logical_k.shape[1]) > block_ids.shape[1]:
            raise ValueError("block table is too short for a KV length")

    if block_mask is not None:
        expected_q_tiles = math.ceil(max_q_len / BLOCK_SIZE)
        if block_mask.device != q.device or block_mask.dtype != torch.uint8:
            raise TypeError("block_mask must be uint8 on the Q device")
        if not block_mask.is_contiguous():
            raise ValueError("block_mask must be contiguous")
        if block_mask.ndim != 4 or block_mask.shape[:3] != (
            batch,
            q.shape[1],
            expected_q_tiles,
        ):
            raise ValueError(
                "block_mask must have shape "
                "[batch, num_query_heads, ceil(max_q_len/128), Kb]"
            )

    metadata = _lower_page_tables(
        cu_seqlens_q,
        block_ids,
        kv_lens,
        block_mask,
        num_query_heads=int(q.shape[1]),
        page_size=int(logical_k.shape[1]),
        device=q.device,
    )
    token_q_scale = torch.cat(
        [
            q_scale[batch_id, :, : int(q_offsets[batch_id + 1] - q_offsets[batch_id])]
            .transpose(0, 1)
            .contiguous()
            for batch_id in range(batch)
        ],
        dim=0,
    )
    return TopsCCPrefillPlan(
        q=q,
        key_cache=logical_k,
        value_cache=logical_v,
        token_q_scale=token_q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        quant_type=normalized_quant,
        metadata=metadata,
        num_kv_heads=int(logical_k.shape[2]),
        page_size=int(logical_k.shape[1]),
        softmax_scale=(
            _HEAD_DIM**-0.5 if softmax_scale is None else float(softmax_scale)
        ),
    )


@torch.no_grad()
def topscc_attention_prefill_fp8_dynamic(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    block_ids: torch.Tensor,
    kv_lens: torch.Tensor,
    max_q_len: int,
    quant_type=1,
    block_mask: torch.Tensor | None = None,
    *,
    layout: str = "NHD",
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Run the complete S60 TopsCC compatibility prefill path."""

    return prepare_topscc_prefill_fp8(
        q,
        key_cache,
        value_cache,
        q_scale,
        k_scale,
        v_scale,
        cu_seqlens_q,
        block_ids,
        kv_lens,
        max_q_len,
        quant_type,
        block_mask,
        layout=layout,
        softmax_scale=softmax_scale,
    )()


__all__ = [
    "TopsCCPrefillPlan",
    "prepare_topscc_prefill_fp8",
    "topscc_attention_prefill_fp8_dynamic",
    "topscc_prefill_available",
]
