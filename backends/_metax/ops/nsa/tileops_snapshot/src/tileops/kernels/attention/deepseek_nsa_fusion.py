# 2026 - NSA E2E extension for MetaX C550.

import functools
from typing import Any, Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel

__all__ = ["NSAGatedFusionFwdKernel"]


@functools.lru_cache(maxsize=32)
def _nsa_gated_fusion_kernel(
    c_seq_len: int,
    heads: int,
    dim: int,
    dtype: str,
    accum_dtype: str,
) -> Callable:
    total = c_seq_len * heads * dim

    @tilelang.jit(out_idx=[6])
    def _nsa_gated_fusion_func(threads: int, num_per_thread: int) -> None:
        block_size = threads * num_per_thread

        @T.prim_func
        def _nsa_gated_fusion_main(
            o_cmp: T.Tensor((c_seq_len, heads, dim), dtype),
            o_slc: T.Tensor((c_seq_len, heads, dim), dtype),
            o_swa: T.Tensor((c_seq_len, heads, dim), dtype),
            g_cmp: T.Tensor((c_seq_len, heads), dtype),
            g_slc: T.Tensor((c_seq_len, heads), dtype),
            g_swa: T.Tensor((c_seq_len, heads), dtype),
            output: T.Tensor((c_seq_len, heads, dim), dtype),
        ) -> None:
            with T.Kernel(T.ceildiv(total, block_size), threads=threads) as bx:
                for tx, lane in T.Parallel(threads, num_per_thread):
                    linear = bx * block_size + tx * num_per_thread + lane
                    if linear < total:
                        d = linear % dim
                        th = linear // dim
                        h = th % heads
                        t = th // heads
                        value = (
                            T.cast(o_cmp[t, h, d], accum_dtype)
                            * T.cast(g_cmp[t, h], accum_dtype)
                            + T.cast(o_slc[t, h, d], accum_dtype)
                            * T.cast(g_slc[t, h], accum_dtype)
                            + T.cast(o_swa[t, h, d], accum_dtype)
                            * T.cast(g_swa[t, h], accum_dtype)
                        )
                        output[t, h, d] = T.cast(value, dtype)

        return _nsa_gated_fusion_main

    return _nsa_gated_fusion_func


@torch.library.custom_op("top::nsa_gated_fusion_fwd_wrapped_kernel", mutates_args=())
def _nsa_gated_fusion_wrapped_kernel(
    c_seq_len: int,
    heads: int,
    dim: int,
    dtype: str,
    accum_dtype: str,
    threads: int,
    num_per_thread: int,
    o_cmp: torch.Tensor,
    o_slc: torch.Tensor,
    o_swa: torch.Tensor,
    g_cmp: torch.Tensor,
    g_slc: torch.Tensor,
    g_swa: torch.Tensor,
) -> torch.Tensor:
    return _nsa_gated_fusion_kernel(
        c_seq_len, heads, dim, dtype, accum_dtype
    )(threads, num_per_thread)(o_cmp, o_slc, o_swa, g_cmp, g_slc, g_swa)


@_nsa_gated_fusion_wrapped_kernel.register_fake
def _(
    c_seq_len: int,
    heads: int,
    dim: int,
    dtype: str,
    accum_dtype: str,
    threads: int,
    num_per_thread: int,
    *inputs: tuple[Any],
) -> torch.Tensor:
    _ = (c_seq_len, heads, dim, dtype, accum_dtype, threads, num_per_thread)
    return torch.empty_like(inputs[0])


class NSAGatedFusionFwdKernel(Kernel):
    supported_archs: list[int] = [80, 89]

    def __init__(
        self,
        c_seq_len: int,
        heads: int,
        dim: int,
        dtype: torch.dtype,
        accum_dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        self.c_seq_len = c_seq_len
        self.heads = heads
        self.dim = dim
        self.dtype = dtype
        self.accum_dtype = accum_dtype
        self.accum_dtype_str = self.dtype_to_str(accum_dtype)
        self.kernel = _nsa_gated_fusion_kernel(
            c_seq_len, heads, dim, self.dtype_str, self.accum_dtype_str
        )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {"threads": 256, "num_per_thread": 4}

    @property
    def autotune_configs(self) -> list[dict]:
        return [
            {"threads": threads, "num_per_thread": num_per_thread}
            for threads in (64, 128, 256)
            for num_per_thread in (2, 4, 8)
        ]

    def forward(
        self,
        o_cmp: torch.Tensor,
        o_slc: torch.Tensor,
        o_swa: torch.Tensor,
        g_cmp: torch.Tensor,
        g_slc: torch.Tensor,
        g_swa: torch.Tensor,
    ) -> torch.Tensor:
        return _nsa_gated_fusion_wrapped_kernel(
            self.c_seq_len,
            self.heads,
            self.dim,
            self.dtype_str,
            self.accum_dtype_str,
            self.config["threads"],
            self.config["num_per_thread"],
            o_cmp,
            o_slc,
            o_swa,
            g_cmp,
            g_slc,
            g_swa,
        )
