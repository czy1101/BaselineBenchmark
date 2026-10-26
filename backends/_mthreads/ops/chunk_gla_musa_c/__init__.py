# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""MUSA C chunk GLA baseline used only by BaselineBenchmark."""

from .musa_chunk_gla import is_available, musa_chunk_gla
from .torch_reference import torch_recurrent_chunk_gla

__all__ = ["is_available", "musa_chunk_gla", "torch_recurrent_chunk_gla"]
