"""Enflame baselines for HPC attention operators."""

from .fp8_utils import decode_fp8e4m3fn

BLOCK_SIZE = 64
from .topscc_decode import (
    TopsCCDecodePlan,
    prepare_topscc_decode_bf16,
    topscc_attention_decode_bf16_dynamic,
    topscc_decode_available,
)
from .topscc_decode_fp8 import (
    TopsCCFP8DecodePlan,
    prepare_topscc_decode_fp8,
    topscc_attention_decode_fp8_dynamic,
)
from .topscc_prefill import (
    BLOCK_SIZE as PREFILL_BLOCK_SIZE,
    TopsCCPrefillPlan,
    prepare_topscc_prefill_fp8,
    topscc_attention_prefill_fp8_dynamic,
    topscc_prefill_available,
)

__all__ = [
    "BLOCK_SIZE",
    "PREFILL_BLOCK_SIZE",
    "TopsCCDecodePlan",
    "TopsCCFP8DecodePlan",
    "TopsCCPrefillPlan",
    "decode_fp8e4m3fn",
    "prepare_topscc_decode_bf16",
    "prepare_topscc_decode_fp8",
    "prepare_topscc_prefill_fp8",
    "topscc_attention_decode_bf16_dynamic",
    "topscc_attention_decode_fp8_dynamic",
    "topscc_attention_prefill_fp8_dynamic",
    "topscc_decode_available",
    "topscc_prefill_available",
]
