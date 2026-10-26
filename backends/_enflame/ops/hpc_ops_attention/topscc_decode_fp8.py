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

"""S60 TopsCC compatibility wrapper for dynamic FP8 decode attention."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .topscc_decode import (
    _compile_device_kernel,
    _load_extension,
    _pack_cache,
    prepare_topscc_decode_bf16,
    topscc_decode_available,
)
from .fp8_utils import _logical_cache, _raw_fp8_bits, decode_fp8e4m3fn


_BLOCK_SIZE = 64


def _normalize_quant_type(quant_type) -> str:
    value = getattr(quant_type, "value", quant_type)
    if value == 0:
        return "qkpertoken_perhead_vperhead"
    if value == 1:
        return "qpertoken_perhead_kvpertensor"
    normalized = str(value).lower()
    if normalized in {
        "qkpertoken_perhead_vperhead",
        "qpertoken_perhead_kpertoken_perhead_vperhead",
    }:
        return "qkpertoken_perhead_vperhead"
    if normalized == "qpertoken_perhead_kvpertensor":
        return normalized
    raise ValueError(f"unsupported FP8 decode quant_type: {quant_type!r}")


def _token_k_scales(
    k_scale: torch.Tensor,
    *,
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
) -> torch.Tensor:
    """Return per-token K scales in logical [block, token, head] order."""

    expected = (num_blocks, block_size, num_kv_heads)
    if k_scale.dtype == torch.float32 and tuple(k_scale.shape) == expected:
        return k_scale

    unpacked = _raw_fp8_bits(k_scale).contiguous().view(torch.float32)
    if unpacked.ndim != 4 or unpacked.shape[0] != num_blocks:
        raise ValueError("packed per-token K scales have an incompatible shape")
    if unpacked.shape[2] != num_kv_heads:
        raise ValueError("packed per-token K scales have an incompatible head count")
    if unpacked.shape[1] * unpacked.shape[3] != block_size:
        raise ValueError("packed per-token K scales have an incompatible block size")
    return unpacked.permute(0, 1, 3, 2).reshape(expected)


@dataclass(frozen=True)
class TopsCCFP8DecodePlan:
    """FP8 lowering plus the static metadata of a native TopsCC decode call."""

    query: torch.Tensor
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    q_scale: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    quant_type: str
    block_tables: torch.Tensor
    context_lens: torch.Tensor
    max_context_len: int
    softmax_scale: float

    def __call__(self) -> torch.Tensor:
        """Run FP8 dequantization and native BF16 paged attention on GCU."""

        query = decode_fp8e4m3fn(self.query) * self.q_scale[..., None]
        key_cache = decode_fp8e4m3fn(self.key_cache)
        value_cache = decode_fp8e4m3fn(self.value_cache)
        num_kv_heads = key_cache.shape[2]

        if self.quant_type == "qkpertoken_perhead_vperhead":
            token_scales = _token_k_scales(
                self.k_scale,
                num_blocks=key_cache.shape[0],
                block_size=key_cache.shape[1],
                num_kv_heads=num_kv_heads,
            )
            key_cache = key_cache * token_scales[..., None]
            value_cache = value_cache * self.v_scale.reshape(
                1, 1, num_kv_heads, 1
            )
        else:
            key_cache = key_cache * self.k_scale.reshape(1)
            value_cache = value_cache * self.v_scale.reshape(1)

        extension = _load_extension()
        fatbin = _compile_device_kernel()
        return extension.forward(
            query.to(torch.bfloat16).contiguous(),
            _pack_cache(key_cache.to(torch.bfloat16).contiguous(), "NHD"),
            _pack_cache(value_cache.to(torch.bfloat16).contiguous(), "NHD"),
            self.block_tables,
            self.context_lens,
            self.max_context_len,
            self.softmax_scale,
            str(fatbin),
        )


def prepare_topscc_decode_fp8(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    kv_lens: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    layout: str = "NHD",
    *,
    quant_type="qpertoken_perhead_kvpertensor",
    softmax_scale: float | None = None,
) -> TopsCCFP8DecodePlan:
    """Prepare FP8 decode by lowering quantized inputs to the TopsCC BF16 ABI.

    The native vendored TopsCC kernel accepts BF16 paged K/V cache.  This
    wrapper stores static page metadata once, but executes FP8 dequantization
    inside ``plan()``.  Thus timing the plan includes every operation needed
    for a real FP8 decode invocation.
    """

    mode = _normalize_quant_type(quant_type)
    if query.ndim != 3:
        raise ValueError("query must have shape [batch * mtp, heads, dim]")
    if q_scale.shape != query.shape[:2]:
        raise ValueError("q_scale must have shape [batch * mtp, query_heads]")
    if q_scale.dtype != torch.float32 or v_scale.dtype != torch.float32:
        raise TypeError("q_scale and v_scale must be float32")
    if any(tensor.device != query.device for tensor in (key_cache, value_cache, block_tables, kv_lens, q_scale, k_scale, v_scale)):
        raise ValueError("all FP8 decode inputs must be on the same device")

    logical_k = _logical_cache(key_cache, layout, _BLOCK_SIZE)
    logical_v = _logical_cache(value_cache, layout, _BLOCK_SIZE)
    if logical_k.shape != logical_v.shape:
        raise ValueError("K and V cache shapes must match")
    if logical_k.shape[-1] != query.shape[-1]:
        raise ValueError("query and K/V head dimensions must match")

    num_kv_heads = logical_k.shape[2]
    if mode == "qkpertoken_perhead_vperhead":
        if v_scale.numel() != num_kv_heads:
            raise ValueError("per-head V scale must contain one value per KV head")
    else:
        if k_scale.numel() != 1 or v_scale.numel() != 1:
            raise ValueError("per-tensor FP8 decode requires scalar K/V scales")

    metadata = prepare_topscc_decode_bf16(
        torch.empty_like(query, dtype=torch.bfloat16),
        torch.empty_like(logical_k, dtype=torch.bfloat16),
        torch.empty_like(logical_v, dtype=torch.bfloat16),
        block_tables,
        kv_lens,
        "NHD",
        softmax_scale=softmax_scale,
    )
    return TopsCCFP8DecodePlan(
        query=query,
        key_cache=logical_k,
        value_cache=logical_v,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        quant_type=mode,
        block_tables=metadata.block_tables,
        context_lens=metadata.context_lens,
        max_context_len=metadata.max_context_len,
        softmax_scale=metadata.softmax_scale,
    )


@torch.no_grad()
def topscc_attention_decode_fp8_dynamic(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    kv_lens: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    layout: str = "NHD",
    *,
    quant_type="qpertoken_perhead_kvpertensor",
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Run the complete S60 TopsCC compatibility FP8 decode path."""

    return prepare_topscc_decode_fp8(
        query,
        key_cache,
        value_cache,
        block_tables,
        kv_lens,
        q_scale,
        k_scale,
        v_scale,
        layout,
        quant_type=quant_type,
        softmax_scale=softmax_scale,
    )()


__all__ = [
    "TopsCCFP8DecodePlan",
    "prepare_topscc_decode_fp8",
    "topscc_attention_decode_fp8_dynamic",
    "topscc_decode_available",
]
