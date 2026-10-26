"""Torch-GCU/TopsAten Parallax baseline for Enflame GCU.

This module is a correctness and performance baseline.  It intentionally uses
official PyTorch operators dispatched through Enflame ``torch_gcu`` and
TopsAten. It does not depend on Triton, TLE, TopsCC, or FlagAttention kernels.
"""

from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint

try:
    import torch_gcu as _torch_gcu  # noqa: F401
except ImportError:
    _torch_gcu = None


_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)
_CHECKPOINT_SCORE_ELEMENTS = 128 * 1024 * 1024


def torch_gcu_parallax_available() -> bool:
    """Return whether the current PyTorch installation exposes an available GCU."""
    return (
        _torch_gcu is not None
        and hasattr(torch, "gcu")
        and torch.gcu.is_available()
    )


def _validate_inputs(
    q: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int | None,
    cu_seqlens: torch.Tensor | None,
    max_score_elements: int,
) -> None:
    tensors = {"q": q, "r": r, "k": k, "v": v}
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.ndim != 4:
            raise ValueError(f"{name} must have shape [B, T, H, D]")
        if tensor.device.type != "gcu":
            raise ValueError(f"{name} must be an Enflame GCU tensor")

    if r.shape != q.shape:
        raise ValueError("r must have the same shape as q")
    if v.shape != k.shape:
        raise ValueError("v must have the same shape as k")
    if any(tensor.device != q.device for tensor in (r, k, v)):
        raise ValueError("q, r, k and v must be on the same GCU device")
    if any(tensor.dtype != q.dtype for tensor in (r, k, v)):
        raise TypeError("q, r, k and v must have the same dtype")
    if q.dtype not in _SUPPORTED_DTYPES:
        raise TypeError("the GCU Torch baseline requires bfloat16 or float16")

    batch, seqlen, query_heads, dim = q.shape
    kv_heads = k.shape[2]
    if k.shape[0] != batch or k.shape[1] != seqlen or k.shape[3] != dim:
        raise ValueError("q/r and k/v must have matching B, T and D dimensions")
    if batch <= 0 or seqlen <= 0 or query_heads <= 0 or kv_heads <= 0:
        raise ValueError("B, T, HQ and H must be positive")
    if query_heads % kv_heads:
        raise ValueError("GQA requires HQ to be divisible by H")
    if not 16 <= dim <= 256:
        raise ValueError("head dimension must be in [16, 256]")
    if window_size is not None and (
        not isinstance(window_size, int) or window_size <= 0
    ):
        raise ValueError("window_size must be a positive integer or None")
    if not isinstance(max_score_elements, int) or max_score_elements <= 0:
        raise ValueError("max_score_elements must be a positive integer")

    if cu_seqlens is not None:
        if not isinstance(cu_seqlens, torch.Tensor):
            raise TypeError("cu_seqlens must be a torch.Tensor or None")
        if batch != 1:
            raise ValueError("packed cu_seqlens inputs require batch size 1")
        if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
            raise ValueError("cu_seqlens must be one-dimensional with at least two entries")
        if cu_seqlens.dtype not in (torch.int32, torch.int64):
            raise TypeError("cu_seqlens must have int32 or int64 dtype")
        if cu_seqlens.device != q.device:
            raise ValueError("cu_seqlens must be on the same GCU device as q")


def _parallax_tile(
    q: torch.Tensor,
    r: torch.Tensor,
    tile_key_t: torch.Tensor,
    tile_value: torch.Tensor,
    tile_key_positions: torch.Tensor,
    query_start: int,
    scale: float,
    window_size: int | None,
    optimized: bool,
    merge_group_rows: bool,
    combine_qr: bool,
    return_weights: bool = False,
) -> torch.Tensor:
    batch, tile_rows, query_heads, dim = q.shape
    kv_heads = tile_value.shape[1]
    groups = query_heads // kv_heads
    key_rows = tile_value.shape[-2]
    query = q.reshape(batch, tile_rows, kv_heads, groups, dim).permute(0, 2, 3, 1, 4).float()
    secondary_query = r.reshape(batch, tile_rows, kv_heads, groups, dim).permute(0, 2, 3, 1, 4).float()

    if merge_group_rows:
        # [B,H,G,Q,D] -> [B,H,Q*G,D]: share one K/V matrix per KV head.
        # This reshape may copy Q/R; it is not assumed to be a free view.
        query = query.transpose(2, 3).reshape(batch, kv_heads, tile_rows * groups, dim)
        secondary_query = secondary_query.transpose(2, 3).reshape(
            batch, kv_heads, tile_rows * groups, dim
        )
        tile_key_t = tile_key_t.squeeze(2)
        tile_value = tile_value.squeeze(2)

    if combine_qr:
        # Same K for both products: join matrix rows, not the reduction
        # dimension. FLOPs are unchanged; cat trades a copy for one fewer
        # GEMM call.
        # Keep concatenation inside checkpoint, so it is not saved per tile.
        combined_query = torch.cat((query, secondary_query), dim=-2)
        combined_logits = torch.matmul(combined_query, tile_key_t)
        logits, secondary_logits = combined_logits.split(query.shape[-2], dim=-2)
        logits = logits * scale  # R @ K.T must NOT receive Q's scale.
    else:
        logits = torch.matmul(query, tile_key_t) * scale
        secondary_logits = torch.matmul(secondary_query, tile_key_t)
    query_positions = torch.arange(
        query_start, query_start + tile_rows, dtype=torch.int32, device=q.device
    ).view(tile_rows, 1)
    valid = tile_key_positions <= query_positions
    if window_size is not None:
        valid = valid & (tile_key_positions >= query_positions - window_size + 1)
    if merge_group_rows:
        valid = valid[:, None, :].expand(tile_rows, groups, key_rows).reshape(
            1, 1, tile_rows * groups, key_rows
        )
    else:
        valid = valid.view(1, 1, 1, tile_rows, key_rows)

    probabilities = torch.softmax(logits.masked_fill(~valid, -float("inf")), dim=-1)
    weighted_secondary = probabilities * secondary_logits
    if optimized:
        # FP32 output matmul fusion; rounding order differs from original.
        mean_secondary = weighted_secondary.sum(dim=-1, keepdim=True)
        weights = probabilities * (1.0 + mean_secondary) - weighted_secondary
        if return_weights:
            return weights
        output = torch.matmul(weights, tile_value)
    else:
        mean_value = torch.matmul(probabilities, tile_value)
        mean_secondary = weighted_secondary.sum(dim=-1, keepdim=True)
        secondary_value = torch.matmul(weighted_secondary, tile_value)
        output = mean_value * (1.0 + mean_secondary) - secondary_value

    if merge_group_rows:
        output = output.reshape(batch, kv_heads, tile_rows, groups, dim).permute(0, 2, 1, 3, 4)
    else:
        output = output.permute(0, 3, 1, 2, 4)
    return output.reshape(batch, tile_rows, query_heads, dim).to(q.dtype)


class _StreamingParallax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, r, k, v, scale, window_size, max_score_elements, combine_qr):
        ctx.save_for_backward(q, r, k, v)
        ctx.settings = (scale, window_size, max_score_elements, combine_qr)
        ctx.set_materialize_grads(False)
        return _dense_parallax(
            q, r, k, v, scale, window_size, max_score_elements,
            checkpoint_tiles=False, combine_qr=combine_qr, streaming_backward=False,
        )

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return (None,) * 8
        q, r, k, v = ctx.saved_tensors
        scale, window_size, max_score_elements, combine_qr = ctx.settings
        needed = ctx.needs_input_grad[:4]
        gradients = [None] * 4

        if torch.is_grad_enabled():
            # In-place accumulation below is a first-order fast path. Preserve
            # create_graph/double backward via the differentiable V4 graph.
            with torch.enable_grad():
                output = _dense_parallax(
                    q, r, k, v, scale, window_size, max_score_elements,
                    checkpoint_tiles=True, combine_qr=combine_qr, streaming_backward=False,
                )
                indices = [i for i, required in enumerate(needed) if required]
                values = torch.autograd.grad(
                    output, tuple((q, r, k, v)[i] for i in indices),
                    grad_output, create_graph=True,
                )
            for index, value in zip(indices, values):
                gradients[index] = value
            return (*gradients, None, None, None, None)

        batch, seqlen, query_heads, dim = q.shape
        kv_heads = k.shape[2]
        groups = query_heads // kv_heads
        merge_group_rows = groups > 1
        query_tile = max(1, min(seqlen, max_score_elements // (batch * query_heads * seqlen)))
        key_t = k.float().permute(0, 2, 1, 3).unsqueeze(2).transpose(-1, -2)
        value = v.float().permute(0, 2, 1, 3).unsqueeze(2)
        key_positions = torch.arange(seqlen, dtype=torch.int32, device=q.device).view(1, seqlen)
        for index, tensor in enumerate((q, r)):
            if needed[index]:
                gradients[index] = torch.empty_like(tensor)
        # Accumulate all tiles in FP32 and cast K/V gradients only once.
        # Never accumulate BF16/FP16 contributions directly into input dtype.
        key_gradient = torch.zeros_like(key_t) if needed[2] else None
        value_gradient = torch.zeros_like(value) if needed[3] else None

        for query_start in reversed(range(0, seqlen, query_tile)):
            query_end = min(query_start + query_tile, seqlen)
            tile_rows = query_end - query_start
            key_start = max(0, query_start - window_size + 1) if window_size is not None else 0
            key_end = query_end
            # Local leaves prevent SliceBackward from expanding each tile's
            # gradients into full-sequence Q/R/K/V buffers.
            local_inputs = (
                q[:, query_start:query_end].detach().requires_grad_(needed[0]),
                r[:, query_start:query_end].detach().requires_grad_(needed[1]),
                key_t[..., key_start:key_end].detach().requires_grad_(needed[2]),
            )
            tile_value = value[..., key_start:key_end, :]
            with torch.enable_grad():
                weights = _parallax_tile(
                    *local_inputs, tile_value, key_positions[:, key_start:key_end],
                    query_start, scale, window_size, True, merge_group_rows,
                    combine_qr, return_weights=True,
                )
            dout = grad_output[:, query_start:query_end].reshape(
                batch, tile_rows, kv_heads, groups, dim
            ).permute(0, 2, 3, 1, 4).float()
            if merge_group_rows:
                dout = dout.transpose(2, 3).reshape(batch, kv_heads, tile_rows * groups, dim)
                tile_value = tile_value.squeeze(2)
            # VJP of O=W@V. Do not recompute the output matmul just to build
            # its graph; only reconstruct W and apply these two derivatives.
            if needed[3]:
                dv = torch.matmul(weights.detach().transpose(-1, -2), dout)
                if merge_group_rows:
                    dv = dv.unsqueeze(2)
                value_gradient[..., key_start:key_end, :].add_(dv)
                del dv
            indices = [i for i in range(3) if needed[i]]
            if indices:
                dweights = torch.matmul(dout, tile_value.transpose(-1, -2))
                values = torch.autograd.grad(
                    weights, tuple(local_inputs[i] for i in indices), dweights,
                )
                for index, gradient in zip(indices, values):
                    if index < 2:
                        gradients[index][:, query_start:query_end].copy_(gradient)
                    else:
                        key_gradient[..., key_start:key_end].add_(gradient)
                del dweights, values, gradient
            del weights, local_inputs, dout

        if needed[2]:
            gradients[2] = key_gradient.transpose(-1, -2).squeeze(2).permute(0, 2, 1, 3).to(k.dtype)
        if needed[3]:
            gradients[3] = value_gradient.squeeze(2).permute(0, 2, 1, 3).to(v.dtype)
        return (*gradients, None, None, None, None)


def _dense_parallax(
    q: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    window_size: int | None,
    max_score_elements: int,
    *,
    optimized: bool = True,
    merge_groups: bool = True,
    checkpoint_tiles: bool | None = None,
    combine_qr: bool = True,
    streaming_backward: bool = True,
) -> torch.Tensor:
    """Evaluate one dense sequence with memory-bounded query tiles."""
    batch, seqlen, query_heads, dim = q.shape
    kv_heads = k.shape[2]
    groups = query_heads // kv_heads
    merge_group_rows = optimized and merge_groups and groups > 1
    combine_qr = optimized and merge_groups and combine_qr

    score_elements_per_query = batch * query_heads * seqlen
    query_tile = max(
        1,
        min(seqlen, max_score_elements // max(score_elements_per_query, 1)),
    )
    # Bound the retained attention graph, not just one forward temporary.
    # This is a conservative score-size heuristic, not a device-memory cap.
    key_span = (
        min(seqlen, window_size + query_tile - 1)
        if optimized and window_size is not None else seqlen
    )
    if checkpoint_tiles is None:
        checkpoint_tiles = (
            optimized and merge_groups
            and batch * query_heads * seqlen * key_span > _CHECKPOINT_SCORE_ELEMENTS
        )
    checkpoint_tiles = (
        checkpoint_tiles and torch.is_grad_enabled()
        and any(tensor.requires_grad for tensor in (q, r, k, v))
    )
    if (
        checkpoint_tiles and streaming_backward and optimized and merge_groups
        and not torch.is_autocast_enabled(q.device.type)
    ):
        return _StreamingParallax.apply(
            q, r, k, v, scale, window_size, max_score_elements, combine_qr,
        )
    # K/V are shared by every query head in one GQA group.
    key = k.float().permute(0, 2, 1, 3).unsqueeze(2)
    value = v.float().permute(0, 2, 1, 3).unsqueeze(2)
    key_t = key.transpose(-1, -2)
    key_positions = torch.arange(
        seqlen, dtype=torch.int32, device=q.device
    ).view(1, seqlen)
    output_tiles = []

    for query_start in range(0, seqlen, query_tile):
        query_end = min(query_start + query_tile, seqlen)
        # Keep the original query partition and memory budget. Only remove
        # keys masked out for every row in this tile; every row keeps exactly
        # the same softmax domain of valid keys.
        key_start = (
            max(0, query_start - window_size + 1)
            if optimized and window_size is not None else 0
        )
        key_end = query_end if optimized else seqlen
        tile_key_t = key_t[..., key_start:key_end] if optimized else key_t
        tile_value = value[..., key_start:key_end, :] if optimized else value

        tile_key_positions = key_positions[:, key_start:key_end] if optimized else key_positions
        # Bind every tile's metadata as explicit arguments, including offsets.
        # Construct score matrices and the expanded mask INSIDE checkpoint:
        # retaining them outside would retain quadratic activation memory.
        arguments = (
            q[:, query_start:query_end], r[:, query_start:query_end],
            tile_key_t, tile_value, tile_key_positions, query_start,
            scale, window_size, optimized, merge_group_rows, combine_qr,
        )
        if checkpoint_tiles:
            output = checkpoint(
                _parallax_tile, *arguments,
                use_reentrant=False, preserve_rng_state=False,
            )
        else:
            output = _parallax_tile(*arguments)
        output_tiles.append(output)

    return torch.cat(output_tiles, dim=1)


def parallel_parallax_torch_gcu(
    q: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    window_size: int | None = None,
    cu_seqlens: torch.Tensor | None = None,
    *,
    max_score_elements: int = 8 * 1024 * 1024,
    implementation: str = "optimized",
    checkpoint_tiles: bool | None = None,
    combine_qr: bool = True,
    streaming_backward: bool = True,
) -> torch.Tensor:
    """Evaluate Parallax using official torch_gcu/TopsAten operators.

    Args:
        q, r: Query tensors with shape ``[B, T, HQ, D]``.
        k, v: Key/value tensors with shape ``[B, T, H, D]``.
        scale: Scale applied only to ``q @ k.T``. Defaults to ``D**-0.5``.
        window_size: Optional causal sliding-window length.
        cu_seqlens: Optional packed sequence offsets with shape ``[N + 1]``.
        max_score_elements: Maximum elements in one QK or RK score tile
            (at least one row). Without checkpointing, autograd
            still retains work proportional to the full attention matrix.
        implementation: "optimized" also merges GQA group/query rows to
            avoid a group-broadcast matmul batch. "optimized_v1" keeps the
            clipped key ranges and merged output matmul without changing the
            GQA layout. "original" retains the initial FP32 computation.
        checkpoint_tiles: None automatically recomputes query tiles during
            backward for large "optimized" graphs. True forces checkpointing
            when gradients are enabled; False disables checkpointing.
            Recomputing reduces saved activations but adds computation.
        combine_qr: For "optimized", join Q/R matrix rows and calculate both
            score products with one FP32 matmul. False preserves the previous
            separate products. Arithmetic count is unchanged; cat adds a copy
            and the joint score allocation has twice one score tile's size.
        streaming_backward: For checkpoint-sized "optimized" graphs, compute
            tile-local gradients and accumulate K/V in FP32. False preserves
            V4 checkpoint/autograd. Higher-order gradients and autocast
            forward calls use that V4 graph.

    Returns:
        Tensor with shape ``[B, T, HQ, D]`` and the same dtype as ``q``.
    """
    _validate_inputs(q, r, k, v, window_size, cu_seqlens, max_score_elements)
    if implementation not in ("original", "optimized_v1", "optimized"):
        raise ValueError("implementation must be 'original', 'optimized_v1' or 'optimized'")
    if checkpoint_tiles is not None and not isinstance(checkpoint_tiles, bool):
        raise TypeError("checkpoint_tiles must be a bool or None")
    if not isinstance(combine_qr, bool):
        raise TypeError("combine_qr must be a bool")
    if not isinstance(streaming_backward, bool):
        raise TypeError("streaming_backward must be a bool")
    optimized = implementation != "original"
    merge_groups = implementation == "optimized"
    actual_scale = q.shape[-1] ** -0.5 if scale is None else float(scale)

    if cu_seqlens is None:
        return _dense_parallax(
            q, r, k, v, actual_scale, window_size, max_score_elements,
            optimized=optimized,
            merge_groups=merge_groups,
            checkpoint_tiles=checkpoint_tiles,
            combine_qr=combine_qr,
            streaming_backward=streaming_backward,
        )

    # Offset extraction is control-plane work only. Attention math and all
    # differentiable tensor operations remain on GCU.
    offsets = [int(value) for value in cu_seqlens.detach().cpu().tolist()]
    if offsets[0] != 0 or offsets[-1] != q.shape[1]:
        raise ValueError("cu_seqlens must start at 0 and end at T")
    if any(end < start for start, end in zip(offsets, offsets[1:])):
        raise ValueError("cu_seqlens must be non-decreasing")

    output_segments = []
    for start, end in zip(offsets, offsets[1:]):
        if start == end:
            output_segments.append(q[:, start:end])
            continue
        output_segments.append(
            _dense_parallax(
                q[:, start:end],
                r[:, start:end],
                k[:, start:end],
                v[:, start:end],
                actual_scale,
                window_size,
                max_score_elements,
                optimized=optimized,
                merge_groups=merge_groups,
                checkpoint_tiles=checkpoint_tiles,
                combine_qr=combine_qr,
                streaming_backward=streaming_backward,
            )
        )
    return torch.cat(output_segments, dim=1)


# Compatibility aliases retained for existing baseline tests and callers.
parallel_parallax_torch = parallel_parallax_torch_gcu
parallel_parallax = parallel_parallax_torch_gcu

__all__ = [
    "parallel_parallax",
    "parallel_parallax_torch",
    "parallel_parallax_torch_gcu",
    "torch_gcu_parallax_available",
]
