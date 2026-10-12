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

import os

import pytest
import torch


HEAD_DIM = 128
NUM_KV_HEADS = 2
NUM_QUERY_HEADS = 4
FP8 = getattr(torch, "float8_e4m3fn", None)
PREFILL_BLOCK_SIZE = 128


def require_gcu_device() -> torch.device:
    pytest.importorskip("torch_gcu")
    if FP8 is None or not hasattr(torch, "gcu"):
        pytest.skip("torch_gcu with float8_e4m3fn support is required")

    count = torch.gcu.device_count()
    if count == 0:
        pytest.skip("no GCU device is available")
    index = int(os.environ.get("S60_DEVICE", "0"))
    if index < 0 or index >= count:
        pytest.skip(f"requested GCU {index}, available count is {count}")
    torch.gcu.set_device(index)
    return torch.device(f"gcu:{index}")


def to_device(value, device: torch.device):
    if value is None:
        return None
    if value.dtype == FP8:
        return value.view(torch.uint8).to(device).view(FP8)
    return value.to(device)


def cache_layout(cache: torch.Tensor, layout: str) -> torch.Tensor:
    if layout == "NHD":
        return cache
    return cache.permute(0, 2, 1, 3).contiguous()


def pack_token_scales(scales: torch.Tensor, head_dim: int) -> torch.Tensor:
    blocks, _tokens, heads = scales.shape
    return (
        scales.permute(0, 2, 1)
        .contiguous()
        .view(FP8)
        .reshape(blocks, heads, -1, head_dim)
        .permute(0, 2, 1, 3)
        .contiguous()
    )


def fp8_softmax_numerator(
    scores: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = torch.exp(scores - scores.amax(dim=-1, keepdim=True))
    denominator = weights.sum(dim=-1, keepdim=True)
    return (weights * 256.0).to(FP8).float(), denominator


def make_prefill_test_inputs(quant_type: str, masked: bool):
    torch.manual_seed(2301)
    q_len = 32
    kv_len = 192
    page_size = 64
    pages = 3
    cache_shape = (pages, page_size, NUM_KV_HEADS, HEAD_DIM)
    q = (torch.randn(q_len, NUM_QUERY_HEADS, HEAD_DIM) * 0.5).to(FP8)
    k_cache = (torch.randn(cache_shape) * 0.5).to(FP8)
    v_cache = (torch.randn(cache_shape) * 0.5).to(FP8)
    q_scale = torch.zeros(1, NUM_QUERY_HEADS, PREFILL_BLOCK_SIZE)
    q_scale[:, :, :q_len] = torch.rand(1, NUM_QUERY_HEADS, q_len) * 0.08 + 0.02
    block_ids = torch.tensor([[2, 0, 1]], dtype=torch.int32)
    if quant_type == "qkpertoken_perhead_vperhead":
        token_k_scale = torch.rand(pages, page_size, NUM_KV_HEADS) * 0.08 + 0.02
        k_scale = pack_token_scales(token_k_scale, HEAD_DIM)
        v_scale = torch.rand(NUM_KV_HEADS) * 0.15 + 0.05
    else:
        token_k_scale = None
        k_scale = torch.tensor([0.08], dtype=torch.float32)
        v_scale = torch.tensor([0.12], dtype=torch.float32)
    block_mask = None
    if masked:
        block_mask = torch.zeros(1, NUM_QUERY_HEADS, 1, 2, dtype=torch.uint8)
        block_mask[..., 1] = 1
    return (
        q, k_cache, v_cache, q_scale, token_k_scale, k_scale, v_scale,
        block_ids, torch.tensor([kv_len], dtype=torch.int32),
        torch.tensor([0, q_len], dtype=torch.int32), block_mask,
    )
