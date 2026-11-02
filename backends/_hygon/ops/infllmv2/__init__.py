# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Standalone HIP/MMAC InfLLM-V2 baseline."""

from .operator import NativeConfig, NativePlan, infllmv2_attention, infllmv2_decode

__all__ = ["NativeConfig", "NativePlan", "infllmv2_attention", "infllmv2_decode"]
