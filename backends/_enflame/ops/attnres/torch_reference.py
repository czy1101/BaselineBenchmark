# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
import os

import torch
import torch.nn.functional as F

try:
    import torch_gcu as _torch_gcu  # noqa: F401
except ImportError:
    _torch_gcu = None


def torch_gcu_attnres_available() -> bool:
    return _torch_gcu is not None and hasattr(torch, "gcu") and torch.gcu.is_available()


def _validate_inputs(query, residuals, rms_weight, output_rms_weight) -> None:
    if not residuals:
        raise ValueError("residuals must contain at least one tensor")
    reference = residuals[0]
    if not isinstance(reference, torch.Tensor):
        raise TypeError("residual 0 must be a torch.Tensor")
    if reference.ndim == 0 or reference.shape[-1] == 0:
        raise ValueError("residual tensors must have a non-empty hidden dimension")
    if reference.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("residual tensors must use float16, bfloat16 or float32")

    for index, residual in enumerate(residuals):
        if not isinstance(residual, torch.Tensor):
            raise TypeError(f"residual {index} must be a torch.Tensor")
        if residual.shape != reference.shape:
            raise ValueError(f"residual {index} must have shape {tuple(reference.shape)}")
        if residual.device != reference.device or residual.dtype != reference.dtype:
            raise ValueError("all residual tensors must have the same device and dtype")

    hidden_size = reference.shape[-1]
    weights = [("query", query), ("rms_weight", rms_weight)]
    if output_rms_weight is not None:
        weights.append(("output_rms_weight", output_rms_weight))
    for name, tensor in weights:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.device != reference.device or tensor.dtype != reference.dtype:
            raise ValueError(f"{name} must have the same device and dtype as residuals")
        if tensor.numel() != hidden_size:
            raise ValueError(f"{name} must contain {hidden_size} elements, got {tensor.numel()}")


def attnres_torch_score_mode(num_rows: int) -> str:
    """Return the selected score path; the auto crossover is a heuristic."""
    mode = os.environ.get("ATTNRES_TORCH_SCORE_MODE", "auto")
    if mode not in ("auto", "normalized", "factorized"):
        raise ValueError("ATTNRES_TORCH_SCORE_MODE must be auto, normalized or factorized")
    if mode == "auto":
        # Small inputs favor fewer eager dispatches. Larger inputs favor
        # avoiding the full FP32 normalized-key allocation. Verify on GCU.
        return "normalized" if num_rows <= 128 else "factorized"
    return mode


def attnres_torch_opt_level() -> int:
    """Keep the measured path (1) by default; opt into candidate (2)."""
    level = os.environ.get("ATTNRES_TORCH_OPT_LEVEL", "1")
    if level not in ("1", "2"):
        raise ValueError("ATTNRES_TORCH_OPT_LEVEL must be 1 or 2")
    return int(level)


def _scaled_mm(matrix: torch.Tensor, vector: torch.Tensor, scale: float) -> torch.Tensor:
    # beta=0 means the fresh output's uninitialized contents are ignored.
    # Reusing it as addmm's input/output avoids an additional bias allocation.
    result = torch.empty((matrix.shape[0], 1), device=matrix.device, dtype=torch.float32)
    return torch.addmm(result, matrix, vector[:, None], beta=0, alpha=scale, out=result)


@torch.no_grad()
def fused_attnres_torch_baseline(
    query: torch.Tensor,
    residuals: Sequence[torch.Tensor],
    rms_weight: torch.Tensor,
    output_rms_weight: torch.Tensor | None = None,
    rms_eps: float = 1e-6,
    scale: float = 1.0,
    return_weights: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Inference AttnRes using FP32 matrix contractions on the input device.

    Residuals are packed as [rows, sources, hidden] so weighted aggregation
    is a BMM without a broadcast-product intermediate or a layout copy.
    Small-row scoring uses native RMSNorm plus MM. Large-row scoring uses
    dot(x, query * rms_weight * scale) / RMS(x), avoiding normalized keys.

    ATTNRES_TORCH_SCORE_MODE=normalized or factorized forces either path.
    ATTNRES_TORCH_OPT_LEVEL=1 restores the previously measured contractions;
    level 2 packs directly into FP32 and uses scaled MM/fused norm arithmetic.
    Validate level 2 on the installed torch_gcu before treating it as faster.
    The auto crossover is provisional until measured on the target GCU.
    Inputs are never modified and outputs are freshly allocated. All matrix
    operands and normalization intermediates use FP32; there is no result
    caching, implicit CPU transfer, or lower-precision matrix conversion.
    """
    _validate_inputs(query, residuals, rms_weight, output_rms_weight)
    output_shape = residuals[0].shape
    hidden_size = output_shape[-1]
    num_rows = residuals[0].numel() // hidden_size
    num_sources = len(residuals)
    mode = attnres_torch_score_mode(num_rows)
    opt_level = attnres_torch_opt_level()
    if num_rows == 0:
        output = torch.empty_like(residuals[0])
        if return_weights:
            probability = torch.empty(
                (num_sources, *output_shape[:-1]), device=output.device, dtype=torch.float32,
            )
            return output, probability
        return output

    # Stack directly in the layout consumed by BMM; no transpose/contiguous
    # of the large residual workspace is needed after score computation.
    flat_residuals = [
        residual.reshape(num_rows, hidden_size) for residual in residuals
    ]
    if opt_level == 2:
        # stack's out overload casts while packing, eliminating the separate
        # low-precision stack allocation and whole-workspace float conversion.
        values = torch.empty(
            (num_rows, num_sources, hidden_size), device=residuals[0].device, dtype=torch.float32,
        )
        torch.stack(flat_residuals, dim=1, out=values)
    else:
        values = torch.stack(flat_residuals, dim=1).float()
    flat_values = values.reshape(num_rows * num_sources, hidden_size)
    if mode == "normalized":
        keys = F.rms_norm(flat_values, (hidden_size,), rms_weight.reshape(-1).float(), rms_eps)
        if opt_level == 2:
            logits = _scaled_mm(keys, query.reshape(-1).float(), scale).reshape(num_rows, num_sources)
        else:
            scaled_query = query.reshape(-1).float() * scale
            logits = torch.mm(keys, scaled_query[:, None]).reshape(num_rows, num_sources)
            del scaled_query
        del keys
    else:
        weighted_query = query.reshape(-1).float() * rms_weight.reshape(-1).float()
        # vector_norm reduces x*x without allocating a full square tensor.
        inverse_rms = torch.linalg.vector_norm(flat_values, ord=2, dim=-1)
        if opt_level == 2:
            # Fused square * (1/D) + eps needs one pointwise operation rather
            # than separate square/mul/add; only a small row-sized eps tensor
            # is initialized. The inverse-RMS result is private to this call.
            epsilon = torch.full_like(inverse_rms, rms_eps)
            torch.addcmul(epsilon, inverse_rms, inverse_rms, value=1.0 / hidden_size, out=inverse_rms)
            inverse_rms.rsqrt_()
            del epsilon
            logits = _scaled_mm(flat_values, weighted_query, scale).reshape(num_rows, num_sources)
        else:
            inverse_rms.square_().mul_(1.0 / hidden_size).add_(rms_eps).rsqrt_()
            weighted_query.mul_(scale)
            logits = torch.mm(flat_values, weighted_query[:, None]).reshape(num_rows, num_sources)
        logits.mul_(inverse_rms.reshape(num_rows, num_sources))
        del inverse_rms, weighted_query

    probability = logits.softmax(dim=-1)
    del logits
    output = torch.bmm(probability[:, None, :], values).squeeze(1).view(output_shape)
    del flat_values, values
    if output_rms_weight is not None:
        output = F.rms_norm(output, (hidden_size,), output_rms_weight.reshape(-1).float(), rms_eps)
    output = output.to(residuals[0].dtype)
    if return_weights:
        return output, probability.transpose(0, 1).reshape(num_sources, *output_shape[:-1])
    return output


fused_attnres = fused_attnres_torch_baseline

__all__ = [
    "fused_attnres",
    "fused_attnres_torch_baseline",
    "attnres_torch_score_mode",
    "attnres_torch_opt_level",
    "torch_gcu_attnres_available",
]
