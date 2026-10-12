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

import math

import pytest
import torch

from backends._enflame.ops.hpc_ops_attention import (
    BLOCK_SIZE,
    prepare_topscc_decode_fp8,
    topscc_decode_available,
)
from backends._enflame.test._hpc_ops_attention_test_utils import (
    HEAD_DIM,
    NUM_KV_HEADS,
    NUM_QUERY_HEADS,
    cache_layout,
    require_gcu_device,
    to_device,
)


pytestmark = [
    pytest.mark.hpc_ops_attention,
    pytest.mark.skipif(
        not topscc_decode_available(),
        reason="S60 TopsCC toolchain is unavailable",
    ),
]


@pytest.fixture(scope="module")
def gcu_device() -> torch.device:
    return require_gcu_device()


@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize("mtp", [1, 3])
@pytest.mark.parametrize(
    "quant_type",
    [
        "qpertoken_perhead_kvpertensor",
        "qkpertoken_perhead_vperhead",
    ],
)
def test_topscc_dynamic_decode_fp8_matches_torch_gcu(
    gcu_device: torch.device,
    quant_type: str,
    layout: str,
    mtp: int,
) -> None:
    torch.manual_seed(1401 + mtp)
    lengths = torch.tensor([67, 113], dtype=torch.int32)
    block_tables = torch.tensor([[2, 0], [3, 1]], dtype=torch.int32)
    cache_shape = (4, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
    q_bf16 = torch.randn(
        lengths.numel() * mtp,
        NUM_QUERY_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
    ) / math.sqrt(HEAD_DIM)
    q_scale = q_bf16.float().abs().amax(dim=-1).clamp_min(1e-4) / 10.0
    query = (q_bf16.float() / q_scale[..., None]).to(torch.float8_e4m3fn)
    key_cache = (
        torch.randn(cache_shape, dtype=torch.bfloat16) / math.sqrt(HEAD_DIM)
    ).to(torch.float8_e4m3fn)
    value_cache = torch.randn(cache_shape, dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    if quant_type == "qkpertoken_perhead_vperhead":
        k_scale = torch.rand(4, BLOCK_SIZE, NUM_KV_HEADS) * 0.08 + 0.02
        v_scale = torch.rand(NUM_KV_HEADS) * 0.15 + 0.05
    else:
        k_scale = torch.tensor([0.08], dtype=torch.float32)
        v_scale = torch.tensor([0.12], dtype=torch.float32)

    query_gcu = to_device(query, gcu_device)
    key_gcu = to_device(cache_layout(key_cache, layout), gcu_device)
    value_gcu = to_device(cache_layout(value_cache, layout), gcu_device)
    q_scale_gcu = to_device(q_scale, gcu_device)
    k_scale_gcu = to_device(k_scale, gcu_device)
    v_scale_gcu = to_device(v_scale, gcu_device)
    tables_gcu = to_device(block_tables, gcu_device)
    lengths_gcu = to_device(lengths, gcu_device)

    if quant_type == "qpertoken_perhead_kvpertensor":
        from flag_attn.runtime.backend._enflame.hpc_ops_attention.decode.dynamic import (
            fp8_qpertoken_perhead_kvpertensor_dynamic as adapted,
        )
    else:
        from flag_attn.runtime.backend._enflame.hpc_ops_attention.decode.dynamic import (
            fp8_qkpertoken_perhead_vperhead_dynamic as adapted,
        )
    adapted_inputs = adapted.FP8DecodeInputs(
        query_gcu,
        key_gcu,
        value_gcu,
        tables_gcu,
        lengths_gcu,
        q_scale_gcu,
        k_scale_gcu,
        v_scale_gcu,
    )
    expected = adapted.attention_decode_fp8(
        adapted_inputs,
        adapted.prepare_decode_workspace(adapted_inputs),
    )
    actual = prepare_topscc_decode_fp8(
        query_gcu,
        key_gcu,
        value_gcu,
        tables_gcu,
        lengths_gcu,
        q_scale_gcu,
        k_scale_gcu,
        v_scale_gcu,
        layout,
        quant_type=quant_type,
    )()
    torch.gcu.synchronize()

    actual_cpu = actual.cpu().float()
    expected_cpu = expected.cpu().float()
    assert torch.isfinite(actual_cpu).all()
    torch.testing.assert_close(actual_cpu, expected_cpu, atol=0.2, rtol=0.2)
