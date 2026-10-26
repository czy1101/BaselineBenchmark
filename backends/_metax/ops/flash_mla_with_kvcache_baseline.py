"""MetaX-MACA FlashMLA KV-cache performance baseline.

The baseline entry points are extracted from ``flash_mla_interface.py`` at
MetaX-MACA/FlashMLA commit b246af19465084b1a97d4de480c3ea0c1b356e4c:
https://github.com/MetaX-MACA/FlashMLA/blob/b246af19465084b1a97d4de480c3ea0c1b356e4c/flash_mla/flash_mla_interface.py#L10-L219
"""

from typing import Optional, Tuple
import warnings

import torch

import flash_mla_cuda as flash_mla


def get_mla_metadata(
    cache_seqlens: torch.Tensor,
    *args,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return scheduler metadata required by the vendor baseline."""
    if "num_heads_per_head_k" in kwargs:
        warnings.warn(
            "Parameter 'num_heads_per_head_k' is deprecated. Please use "
            "'num_q_tokens_per_head_k' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        num_heads_per_head_k = kwargs.pop("num_heads_per_head_k")

        if "num_heads_k" in kwargs:
            num_heads_k = kwargs.pop("num_heads_k")
        elif len(args) >= 1:
            num_heads_k = args[0]
            args = args[1:]
        else:
            raise TypeError(
                "Legacy call missing required 'num_heads_k' "
                "(position 2 or keyword argument)"
            )

        if "num_heads_q" in kwargs or "is_fp8_kvcache" in kwargs or "topk" in kwargs:
            raise TypeError(
                "The legacy call does not support the parameters: "
                "(num_heads_q, is_fp8_kvcache, topk). If you want to use them, "
                "please replace 'num_heads_per_head_k' with "
                "'num_q_tokens_per_head_k'."
            )

        if len(args) > 0 or len(kwargs) > 0:
            extra_args = list(args) + list(kwargs.keys())
            raise TypeError(
                "Legacy calls do not support extra parameters: "
                f"{extra_args}."
            )

        return flash_mla.get_mla_metadata(
            cache_seqlens,
            num_heads_per_head_k,
            num_heads_k,
            None,
            False,
            None,
        )

    if len(args) > 5:
        raise TypeError(
            f"get_mla_metadata() takes 6 positional arguments but {len(args) + 1} "
            "were given"
        )

    if "num_q_tokens_per_head_k" in kwargs:
        num_q_tokens_per_head_k = kwargs.pop("num_q_tokens_per_head_k")
    elif len(args) >= 1:
        num_q_tokens_per_head_k = args[0]
        args = args[1:]
    else:
        raise TypeError(
            "get_mla_metadata() missing required 'num_q_tokens_per_head_k' "
            "(position 1 or keyword argument)"
        )

    if len(args) >= 1 and "num_heads_k" not in kwargs:
        num_heads_k = args[0]
        args = args[1:]
    elif "num_heads_k" in kwargs:
        num_heads_k = kwargs.pop("num_heads_k")
    else:
        raise TypeError(
            "get_mla_metadata() missing required 'num_heads_k' "
            "(position 2 or keyword argument)"
        )

    num_heads_q: Optional[int] = None
    is_fp8_kvcache = False
    topk: Optional[int] = None

    if len(args) >= 1 and "num_heads_q" not in kwargs:
        num_heads_q = args[0]
        args = args[1:]
    elif "num_heads_q" in kwargs:
        num_heads_q = kwargs.pop("num_heads_q")

    if len(args) >= 1 and "is_fp8_kvcache" not in kwargs:
        is_fp8_kvcache = args[0]
        args = args[1:]
    elif "is_fp8_kvcache" in kwargs:
        is_fp8_kvcache = kwargs.pop("is_fp8_kvcache")

    if len(args) >= 1 and "topk" not in kwargs:
        topk = args[0]
        args = args[1:]
    elif "topk" in kwargs:
        topk = kwargs.pop("topk")

    if kwargs:
        raise TypeError(
            "Unrecognized keyword arguments: "
            f"{list(kwargs.keys())}. Supported parameters: cache_seqlens, "
            "num_q_tokens_per_head_k, num_heads_k, num_heads_q, "
            "is_fp8_kvcache, topk"
        )

    return flash_mla.get_mla_metadata(
        cache_seqlens,
        num_q_tokens_per_head_k,
        num_heads_k,
        num_heads_q,
        is_fp8_kvcache,
        topk,
    )


def flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    is_fp8_kvcache: bool = False,
    indices: Optional[torch.Tensor] = None,
    indices_all_valid_per_q: Optional[torch.Tensor] = None,
    descale_q: Optional[torch.Tensor] = None,
    descale_k: Optional[torch.Tensor] = None,
    cp_world_size=1,
    cp_rank=0,
    cp_tot_seqlen_k=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the MetaX-MACA FlashMLA KV-cache baseline."""
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    if indices is not None:
        assert causal is False, "causal must be `false` if sparse attention is enabled."
    assert (descale_q is None) == (descale_k is None), (
        "descale_q and descale_k should be both None or both not None"
    )
    if indices_all_valid_per_q is None:
        batch_size = q.shape[0]
        seqlen_q = q.shape[1]
        indices_all_valid_per_q = torch.full(
            (batch_size, seqlen_q, 1),
            False,
            dtype=torch.bool,
            device=q.device,
        )
    out, softmax_lse = flash_mla.fwd_kvcache_mla(
        q,
        k_cache,
        None,
        head_dim_v,
        cache_seqlens,
        block_table,
        softmax_scale,
        causal,
        tile_scheduler_metadata,
        num_splits,
        is_fp8_kvcache,
        indices,
        indices_all_valid_per_q,
        cp_world_size,
        cp_rank,
        cp_tot_seqlen_k,
    )
    return out, softmax_lse


__all__ = ["flash_mla_with_kvcache", "get_mla_metadata"]
