"""MetaX-MACA FlashMLA sparse-prefill performance baseline.

The baseline entry point is extracted from ``flash_mla_interface.py`` at
MetaX-MACA/FlashMLA commit b246af19465084b1a97d4de480c3ea0c1b356e4c:
https://github.com/MetaX-MACA/FlashMLA/blob/b246af19465084b1a97d4de480c3ea0c1b356e4c/flash_mla/flash_mla_interface.py#L220-L258
"""

from typing import Optional, Tuple

import torch

import flash_mla_cuda as flash_mla


def flash_mla_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    indices_all_valid_per_q: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the MetaX-MACA FlashMLA sparse-prefill baseline."""
    if indices_all_valid_per_q is None:
        indices_all_valid_per_q = torch.full(
            (q.shape[0], 1),
            False,
            dtype=torch.bool,
            device=q.device,
        )
    results = flash_mla.sparse_prefill_fwd(
        q,
        kv,
        indices,
        sm_scale,
        d_v,
        indices_all_valid_per_q,
    )
    return results


__all__ = ["flash_mla_sparse_fwd"]
