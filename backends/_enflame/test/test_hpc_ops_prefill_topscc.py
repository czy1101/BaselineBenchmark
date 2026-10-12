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

import pytest
import torch

from backends._enflame.ops.hpc_ops_attention import (
    prepare_topscc_prefill_fp8,
    topscc_prefill_available,
)
from backends._enflame.test._hpc_ops_attention_test_utils import (
    cache_layout,
    make_prefill_test_inputs,
    require_gcu_device,
    to_device,
)
from flag_attn.runtime.backend._enflame.hpc_ops_attention.prefill import (
    attention_with_kvcache_blocksparse_prefill_fp8,
)


pytestmark = [
    pytest.mark.hpc_ops_attention,
    pytest.mark.skipif(
        not topscc_prefill_available(),
        reason="S60 TopsCC toolchain is unavailable",
    ),
]


@pytest.fixture(scope="module")
def gcu_device() -> torch.device:
    return require_gcu_device()


@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize(
    "quant_type",
    [
        "qpertoken_perhead_kvpertensor",
        "qkpertoken_perhead_vperhead",
    ],
)
def test_topscc_dynamic_prefill_fp8_matches_flagattention(
    gcu_device: torch.device,
    quant_type: str,
    masked: bool,
    layout: str,
) -> None:
    (
        q,
        k_cache,
        v_cache,
        q_scale,
        _token_k_scale,
        k_scale,
        v_scale,
        block_ids,
        kv_lens,
        cu_seqlens_q,
        block_mask,
    ) = make_prefill_test_inputs(quant_type, masked)

    q_gcu = to_device(q, gcu_device)
    k_gcu = to_device(cache_layout(k_cache, layout), gcu_device)
    v_gcu = to_device(cache_layout(v_cache, layout), gcu_device)
    q_scale_gcu = to_device(q_scale, gcu_device)
    k_scale_gcu = to_device(k_scale, gcu_device)
    v_scale_gcu = to_device(v_scale, gcu_device)
    cu_q_gcu = to_device(cu_seqlens_q, gcu_device)
    blocks_gcu = to_device(block_ids, gcu_device)
    lengths_gcu = to_device(kv_lens, gcu_device)
    mask_gcu = to_device(block_mask, gcu_device)

    expected = torch.empty_like(q_gcu, dtype=torch.bfloat16)
    attention_with_kvcache_blocksparse_prefill_fp8(
        q_gcu,
        k_gcu,
        v_gcu,
        q_scale_gcu,
        k_scale_gcu,
        v_scale_gcu,
        cu_q_gcu,
        blocks_gcu,
        lengths_gcu,
        q.shape[0],
        0 if quant_type == "qkpertoken_perhead_vperhead" else 1,
        block_mask=mask_gcu,
        output=expected,
    )
    actual = prepare_topscc_prefill_fp8(
        q_gcu,
        k_gcu,
        v_gcu,
        q_scale_gcu,
        k_scale_gcu,
        v_scale_gcu,
        cu_q_gcu,
        blocks_gcu,
        lengths_gcu,
        q.shape[0],
        quant_type,
        mask_gcu,
        layout=layout,
    )()
    torch.gcu.synchronize()

    actual_cpu = actual.cpu().float()
    expected_cpu = expected.cpu().float()
    assert torch.isfinite(actual_cpu).all()
    torch.testing.assert_close(
        actual_cpu,
        expected_cpu,
        atol=0.1,
        rtol=0.1,
    )
