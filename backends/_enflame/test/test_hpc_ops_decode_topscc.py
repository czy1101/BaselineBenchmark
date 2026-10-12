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
    prepare_topscc_decode_bf16,
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
def test_topscc_dynamic_decode_bf16_matches_torch_gcu(
    gcu_device: torch.device,
    layout: str,
    mtp: int,
) -> None:
    torch.manual_seed(915 + mtp)
    lengths = torch.tensor([67, 113], dtype=torch.int32)
    block_tables = torch.tensor([[2, 0], [3, 1]], dtype=torch.int32)
    cache_shape = (4, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
    query = torch.randn(
        lengths.numel() * mtp,
        NUM_QUERY_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
    ) / math.sqrt(HEAD_DIM)
    key_cache = torch.randn(cache_shape, dtype=torch.bfloat16) / math.sqrt(HEAD_DIM)
    value_cache = torch.randn(cache_shape, dtype=torch.bfloat16)

    query_gcu = to_device(query, gcu_device)
    key_gcu = to_device(cache_layout(key_cache, layout), gcu_device)
    value_gcu = to_device(cache_layout(value_cache, layout), gcu_device)
    tables_gcu = to_device(block_tables, gcu_device)
    lengths_gcu = to_device(lengths, gcu_device)

    from flag_attn.runtime.backend._enflame.hpc_ops_attention.decode.dynamic import (
        bf16_dynamic as adapted,
    )

    adapted_inputs = adapted.DynamicBF16Inputs(
        query_gcu,
        key_gcu,
        value_gcu,
        tables_gcu,
        lengths_gcu,
        layout,
    )
    expected = adapted.attention_decode_bf16_dynamic(
        adapted_inputs,
        adapted.prepare_dynamic_bf16_workspace(adapted_inputs),
    )
    plan = prepare_topscc_decode_bf16(
        query_gcu,
        key_gcu,
        value_gcu,
        tables_gcu,
        lengths_gcu,
        layout,
    )
    actual = plan()
    torch.gcu.synchronize()

    actual_cpu = actual.cpu().float()
    expected_cpu = expected.cpu().float()
    assert torch.isfinite(actual_cpu).all()
    torch.testing.assert_close(
        actual_cpu,
        expected_cpu,
        atol=0.03,
        rtol=0.03,
    )
