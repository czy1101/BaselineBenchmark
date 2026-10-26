"""Forward-only NSA E2E composition built from TileOps components."""

from typing import Dict, Optional

import torch

from tileops.kernels.attention.gqa_sliding_window_varlen_fwd import (
    GQASlidingWindowVarlenFwdWgmmaPipelinedKernel,
)
from tileops.kernels.attention.deepseek_nsa_fusion import NSAGatedFusionFwdKernel
from tileops.kernels.attention.deepseek_nsa_topk_paper import (
    NSATopkPaperVarlenKernel,
)
from tileops.kernels.kernel_base import Kernel

from ..op_base import Op
from ..pool import MeanPoolingForwardOp
from .deepseek_nsa import NSACmpFwdVarlenOp, NSAFwdVarlenOp
from .gqa import GroupedQueryAttentionSlidingWindowVarlenFwdOp

__all__ = [
    "NSAGatedFusionFwdOp",
    "NSAForwardVarlenOp",
    "NSATopkPaperVarlenOp",
]


class NSATopkPaperVarlenOp(Op):
    """Top-k block selection with NSA fixed-block semantics.

    The first block, previous local block, and current local block are fixed
    members of the selected set and are included in ``selected_block_num``.
    """

    def __init__(
        self,
        seq_num: int,
        c_seq_len: int,
        heads: int,
        dim: int,
        chunk_num: int,
        group: int,
        scale: float,
        selected_block_num: int,
        bc: int,
        bs: int,
        accum_dtype: torch.dtype,
        tune: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ) -> None:
        params = {k: v for k, v in locals().items() if k not in ("self", "kernel_map")}
        for key, value in params.items():
            setattr(self, key, value)
        self._kernel_params = params
        self.dispatch_kernel(kernel_map)

    def _get_kernel(self, dtype: torch.dtype) -> Kernel:
        return self.get_or_build_kernel(
            "nsa_topk_paper_varlen_kernel",
            dtype,
            lambda: self.kernel_map["nsa_topk_paper_varlen_kernel"](
                **self._kernel_params, dtype=dtype
            ),
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"nsa_topk_paper_varlen_kernel": NSATopkPaperVarlenKernel}

    def forward(
        self,
        q: torch.Tensor,
        k_cmp: torch.Tensor,
        lse_in: torch.Tensor,
        offsets: torch.Tensor,
        chunk_offsets: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> torch.Tensor:
        self.dtype = q.dtype
        return self._get_kernel(q.dtype)(
            q, k_cmp, lse_in, offsets, chunk_offsets, token_indices
        )


class NSAGatedFusionFwdOp(Op):
    """Fuse compression, selected, and sliding outputs with per-head gates."""

    def __init__(
        self,
        c_seq_len: int,
        heads: int,
        dim: int,
        accum_dtype: torch.dtype = torch.float32,
        tune: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ) -> None:
        self.c_seq_len = c_seq_len
        self.heads = heads
        self.dim = dim
        self.accum_dtype = accum_dtype
        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def _get_kernel(self, dtype: torch.dtype) -> Kernel:
        return self.get_or_build_kernel(
            "nsa_gated_fusion_fwd_kernel",
            dtype,
            lambda: self.kernel_map["nsa_gated_fusion_fwd_kernel"](
                c_seq_len=self.c_seq_len,
                heads=self.heads,
                dim=self.dim,
                dtype=dtype,
                accum_dtype=self.accum_dtype,
                tune=self.tune,
            ),
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"nsa_gated_fusion_fwd_kernel": NSAGatedFusionFwdKernel}

    def forward(
        self,
        o_cmp: torch.Tensor,
        o_slc: torch.Tensor,
        o_swa: torch.Tensor,
        g_cmp: torch.Tensor,
        g_slc: torch.Tensor,
        g_swa: torch.Tensor,
    ) -> torch.Tensor:
        output_shape = (self.c_seq_len, self.heads, self.dim)
        gate_shape = (self.c_seq_len, self.heads)
        for tensor, name in (
            (o_cmp, "o_cmp"),
            (o_slc, "o_slc"),
            (o_swa, "o_swa"),
        ):
            if tuple(tensor.shape) != output_shape:
                raise ValueError(f"{name} must have shape {output_shape}, got {tuple(tensor.shape)}")
            if tensor.dtype != o_cmp.dtype or tensor.device != o_cmp.device:
                raise ValueError(f"{name} must match o_cmp dtype and device")
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
        for tensor, name in (
            (g_cmp, "g_cmp"),
            (g_slc, "g_slc"),
            (g_swa, "g_swa"),
        ):
            if tuple(tensor.shape) != gate_shape:
                raise ValueError(f"{name} must have shape {gate_shape}, got {tuple(tensor.shape)}")
            if tensor.dtype != o_cmp.dtype or tensor.device != o_cmp.device:
                raise ValueError(f"{name} must match o_cmp dtype and device")
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
        self.dtype = o_cmp.dtype
        return self._get_kernel(o_cmp.dtype)(
            o_cmp, o_slc, o_swa, g_cmp, g_slc, g_swa
        )


class NSAForwardVarlenOp(Op):
    """Forward-only three-branch NSA composition.

    Inputs use packed THD layout. Callers may supply precomputed metadata for
    kernel-level analysis, or omit it and let the op generate metadata once
    per native E2E invocation.
    """

    def __init__(
        self,
        seq_num: int,
        c_seq_len: int,
        max_seqlen: int,
        heads: int,
        heads_kv: int,
        dim: int,
        chunk_num: int,
        block_size: int = 32,
        selected_blocks: int = 16,
        window_size: int = 128,
        scale: Optional[float] = None,
        accum_dtype: torch.dtype = torch.float32,
        tune: bool = False,
    ) -> None:
        if heads_kv <= 0 or heads % heads_kv != 0:
            raise ValueError("heads must be divisible by heads_kv")
        group = heads // heads_kv
        if group % 16 != 0:
            raise ValueError("NSA group size must be a multiple of 16")
        if selected_blocks <= 0 or 2 * selected_blocks > 32:
            raise ValueError("selected_blocks must satisfy 1 <= S <= 16")
        if window_size <= 0:
            raise ValueError("three-branch NSA requires window_size > 0")

        self.seq_num = seq_num
        self.c_seq_len = c_seq_len
        self.max_seqlen = max_seqlen
        self.heads = heads
        self.heads_kv = heads_kv
        self.dim = dim
        self.chunk_num = chunk_num
        self.block_size = block_size
        self.selected_blocks = selected_blocks
        self.window_size = window_size
        self.scale = dim**-0.5 if scale is None else scale
        self.accum_dtype = accum_dtype
        self.tune = tune
        self.group = group

        self._pool_op = MeanPoolingForwardOp(
            batch_size=1,
            seq_len=c_seq_len,
            heads=heads_kv,
            dim=dim,
            chunk_size=block_size,
            chunks_per_batch=chunk_num,
            seq_num=seq_num,
            use_offsets=1,
            accum_dtype=accum_dtype,
            tune=tune,
        )
        self._cmp_op = NSACmpFwdVarlenOp(
            seq_num=seq_num,
            c_seq_len=c_seq_len,
            heads=heads,
            dim_k=dim,
            dim_v=dim,
            chunk_num=chunk_num,
            group=group,
            scale=self.scale,
            bc=32,
            bs=block_size,
            accum_dtype=accum_dtype,
            tune=tune,
        )
        self._topk_op = NSATopkPaperVarlenOp(
            seq_num=seq_num,
            c_seq_len=c_seq_len,
            heads=heads,
            dim=dim,
            chunk_num=chunk_num,
            group=group,
            scale=self.scale,
            selected_block_num=selected_blocks,
            bc=32,
            bs=block_size,
            accum_dtype=accum_dtype,
            tune=False,
        )
        self._selected_op = NSAFwdVarlenOp(
            batch=seq_num,
            heads=heads,
            c_seq_len=c_seq_len,
            dim=dim,
            is_causal=True,
            scale=self.scale,
            block_size=block_size,
            groups=group,
            selected_blocks=selected_blocks,
            accum_dtype=accum_dtype,
            tune=tune,
        )

        sliding_config = {
            "block_m": 64,
            "block_n": 32 if dim >= 128 else 64,
            "num_stages": 1,
            "threads": 128,
        }

        def c550_sliding_kernel(**kwargs):
            kwargs["config"] = sliding_config
            kwargs["tune"] = False
            return GQASlidingWindowVarlenFwdWgmmaPipelinedKernel(**kwargs)

        self._sliding_op = GroupedQueryAttentionSlidingWindowVarlenFwdOp(
            batch=seq_num,
            heads=heads,
            heads_kv=heads_kv,
            dim=dim,
            is_causal=True,
            window_size_left=window_size - 1,
            window_size_right=0,
            accum_dtype=accum_dtype,
            kernel_map={"gqa_sliding_window_varlen_fwd_kernel": c550_sliding_kernel},
            tune=False,
        )
        self._fusion_op = NSAGatedFusionFwdOp(
            c_seq_len=c_seq_len,
            heads=heads,
            dim=dim,
            accum_dtype=accum_dtype,
            tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {}

    def kernel_delegates(self) -> tuple[Op, ...]:
        return (
            self._pool_op,
            self._cmp_op,
            self._topk_op,
            self._selected_op,
            self._sliding_op,
            self._fusion_op,
        )

    def _build_metadata(
        self,
        offsets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build packed varlen metadata from cumulative sequence lengths."""
        expected_offsets_shape = (self.seq_num + 1,)
        if tuple(offsets.shape) != expected_offsets_shape:
            raise ValueError(
                f"offsets must have shape {expected_offsets_shape}, "
                f"got {tuple(offsets.shape)}"
            )
        if offsets.dtype != torch.int32:
            raise ValueError(f"offsets must have dtype int32, got {offsets.dtype}")
        if offsets.device.type != "cuda":
            raise ValueError(f"offsets must be on a cuda device, got {offsets.device}")
        if not offsets.is_contiguous():
            raise ValueError("offsets must be contiguous")

        offset_values = [int(value) for value in offsets.tolist()]
        if offset_values[0] != 0:
            raise ValueError(f"offsets[0] must be 0, got {offset_values[0]}")
        if offset_values[-1] != self.c_seq_len:
            raise ValueError(
                f"offsets[-1] must equal c_seq_len ({self.c_seq_len}), "
                f"got {offset_values[-1]}"
            )

        lengths = [
            offset_values[index + 1] - offset_values[index]
            for index in range(self.seq_num)
        ]
        if any(length < 0 for length in lengths):
            raise ValueError("offsets must be non-decreasing")
        if max(lengths, default=0) > self.max_seqlen:
            raise ValueError(
                f"maximum sequence length exceeds max_seqlen={self.max_seqlen}"
            )

        token_parts = []
        chunk_parts = []
        block_count_parts = []
        cumulative_chunks = [0]

        for sequence_id, length in enumerate(lengths):
            positions = torch.arange(
                length, dtype=torch.int32, device=offsets.device
            )
            token_parts.append(
                torch.stack(
                    [torch.full_like(positions, sequence_id), positions],
                    dim=1,
                )
            )

            sequence_chunks = (length + self.block_size - 1) // self.block_size
            chunks = torch.arange(
                sequence_chunks, dtype=torch.int32, device=offsets.device
            )
            chunk_parts.append(
                torch.stack(
                    [torch.full_like(chunks, sequence_id), chunks],
                    dim=1,
                )
            )
            cumulative_chunks.append(cumulative_chunks[-1] + sequence_chunks)

            counts = (positions // self.block_size + 1).clamp(
                max=self.selected_blocks
            )
            block_count_parts.append(
                counts[:, None].expand(length, self.heads_kv).contiguous()
            )

        if cumulative_chunks[-1] != self.chunk_num:
            raise ValueError(
                f"generated chunk count {cumulative_chunks[-1]} does not match "
                f"constructor chunk_num={self.chunk_num}"
            )

        chunk_offsets = torch.tensor(
            cumulative_chunks, dtype=torch.int32, device=offsets.device
        )
        chunk_indices = torch.cat(chunk_parts, dim=0).contiguous()
        token_indices = torch.cat(token_parts, dim=0).contiguous()
        block_counts = torch.cat(block_count_parts, dim=0).contiguous()
        return chunk_offsets, chunk_indices, token_indices, block_counts

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g_cmp: torch.Tensor,
        g_slc: torch.Tensor,
        g_swa: torch.Tensor,
        offsets: torch.Tensor,
        chunk_offsets: Optional[torch.Tensor] = None,
        chunk_indices: Optional[torch.Tensor] = None,
        token_indices: Optional[torch.Tensor] = None,
        block_counts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        expected_q = (self.c_seq_len, self.heads, self.dim)
        expected_kv = (self.c_seq_len, self.heads_kv, self.dim)
        if tuple(q.shape) != expected_q:
            raise ValueError(f"q must have shape {expected_q}, got {tuple(q.shape)}")
        if tuple(k.shape) != expected_kv or tuple(v.shape) != expected_kv:
            raise ValueError(f"k and v must have shape {expected_kv}")
        if not all(t.is_contiguous() for t in (q, k, v, g_cmp, g_slc, g_swa)):
            raise ValueError("q, k, v, and gates must be contiguous")

        if any(
            tensor is None
            for tensor in (
                chunk_offsets,
                chunk_indices,
                token_indices,
                block_counts,
            )
        ):
            generated = self._build_metadata(offsets)
            if chunk_offsets is None:
                chunk_offsets = generated[0]
            if chunk_indices is None:
                chunk_indices = generated[1]
            if token_indices is None:
                token_indices = generated[2]
            if block_counts is None:
                block_counts = generated[3]

        metadata_specs = (
            (offsets, "offsets", (self.seq_num + 1,)),
            (chunk_offsets, "chunk_offsets", (self.seq_num + 1,)),
            (chunk_indices, "chunk_indices", (self.chunk_num, 2)),
            (token_indices, "token_indices", (self.c_seq_len, 2)),
            (
                block_counts,
                "block_counts",
                (self.c_seq_len, self.heads_kv),
            ),
        )
        for tensor, name, expected_shape in metadata_specs:
            if tensor is None:
                raise RuntimeError(f"{name} metadata was not generated")
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"{name} must have shape {expected_shape}, "
                    f"got {tuple(tensor.shape)}"
                )
            if tensor.dtype != torch.int32:
                raise ValueError(
                    f"{name} must have dtype int32, got {tensor.dtype}"
                )
            if tensor.device != q.device:
                raise ValueError(
                    f"{name} must be on {q.device}, got {tensor.device}"
                )
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous")

        k_cmp = self._pool_op(k.unsqueeze(0), offsets, chunk_indices).squeeze(0)
        v_cmp = self._pool_op(v.unsqueeze(0), offsets, chunk_indices).squeeze(0)
        o_cmp, lse_cmp = self._cmp_op(
            q, k_cmp, v_cmp, offsets, chunk_offsets, token_indices
        )
        block_indices = self._topk_op(
            q, k_cmp, lse_cmp, offsets, chunk_offsets, token_indices
        )
        o_slc = self._selected_op(
            q, k, v, block_indices, block_counts, offsets, token_indices
        )
        o_swa = self._sliding_op(q, k, v, offsets, offsets, self.max_seqlen)
        return self._fusion_op(o_cmp, o_slc, o_swa, g_cmp, g_slc, g_swa)
