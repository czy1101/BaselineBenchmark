# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import shutil
import subprocess

import torch


_THIS_DIR = Path(__file__).resolve().parent
_TOPSCC_SOURCE_DIR = _THIS_DIR / "topscc"
_TOPSCC_SOURCE = _TOPSCC_SOURCE_DIR / "attention.cpp"
_EXTENSION_SOURCE = _THIS_DIR / "topscc_decode_extension.cpp"


def _topscc_binary() -> str | None:
    configured = os.environ.get("TOPSCC")
    if configured and Path(configured).is_file():
        return configured
    discovered = shutil.which("topscc")
    if discovered:
        return discovered
    installed = Path("/opt/tops/bin/topscc")
    return str(installed) if installed.is_file() else None


def topscc_decode_available() -> bool:
    """Return whether this host can build and load the S60 TopsCC kernel."""

    return (
        importlib.util.find_spec("torch_gcu") is not None
        and _topscc_binary() is not None
        and (
            Path("/opt/tops/include/tops/tops_runtime.h").is_file()
            or Path("/opt/tops/include/tops_runtime.h").is_file()
        )
    )


def _build_root() -> Path:
    configured = os.environ.get("BASELINEBENCHMARK_TOPSCC_BUILD_DIR")
    root = (
        Path(configured).expanduser()
        if configured
        else Path.home() / ".cache" / "baselinebenchmark" / "topscc_decode"
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


def _source_is_newer(output: Path) -> bool:
    if not output.is_file():
        return True
    output_time = output.stat().st_mtime
    dependencies = (
        _TOPSCC_SOURCE,
        _TOPSCC_SOURCE_DIR / "utils" / "utils.h",
        _TOPSCC_SOURCE_DIR / "third_party" / "attention_gqa_kernels.h",
    )
    return any(path.stat().st_mtime > output_time for path in dependencies)


def _acore_arguments() -> list[str]:
    library_directory = Path("/opt/tops/lib/acore")
    for name in ("libacoreop.bc", "libacore.bc"):
        if (library_directory / name).is_file():
            return [
                f"--tops-device-lib-path={library_directory}",
                f"--tops-device-lib={name}",
                "-D__ATOMIC_OP__",
                "-D__ACORE_OP__",
            ]
    return []


@lru_cache(maxsize=1)
def _compile_device_kernel() -> Path:
    if not topscc_decode_available():
        raise RuntimeError(
            "S60 TopsCC baseline requires torch_gcu and /opt/tops/bin/topscc"
        )

    build_root = _build_root()
    fatbin = build_root / "attention_gcu300.topsfb"
    if not _source_is_newer(fatbin):
        return fatbin

    temporary = build_root / "topscc"
    temporary.mkdir(parents=True, exist_ok=True)
    requested_output = temporary / "attention.topsfb"
    command = [
        _topscc_binary(),
        str(_TOPSCC_SOURCE),
        "-arch=gcu300",
        "-O3",
        "-std=c++17",
        "-fPIC",
        "-ltops",
        "-Xclang",
        "-fallow-half-arguments-and-returns",
        "--target=x86_64-unknown-linux-gnu",
        "-fno-omit-frame-pointer",
        "-DNDEBUG",
        "-D__GCU_ARCH__=300",
        "-D__KRT_ARCH__=300",
        "-DTOPS_DISABLE_FORCE_INLINE",
        f"-I{_TOPSCC_SOURCE_DIR}",
        "-I/opt/tops/include",
        "-I/opt/tops/include/acore",
        "--save-temps",
        "-o",
        str(requested_output),
        *_acore_arguments(),
    ]
    completed = subprocess.run(
        command,
        cwd=temporary,
        check=False,
        capture_output=True,
        text=True,
    )
    generated = temporary / "attention.cpp-tops-dtu-enflame-tops.topsfb"
    compiled_fatbin = generated if generated.is_file() else requested_output
    if completed.returncode != 0 and not compiled_fatbin.is_file():
        raise RuntimeError(
            "topscc failed to build the S60 paged-attention kernel:\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    if not compiled_fatbin.is_file():
        raise RuntimeError("topscc completed without producing a .topsfb file")
    shutil.copy2(compiled_fatbin, fatbin)
    return fatbin


@lru_cache(maxsize=1)
def _load_extension():
    if importlib.util.find_spec("torch_gcu") is None:
        raise RuntimeError("torch_gcu is unavailable")
    import torch_gcu  # noqa: F401
    from torch.utils.cpp_extension import load

    torch_gcu_root = Path(torch_gcu.__file__).resolve().parent
    extension_build = _build_root() / "extension"
    extension_build.mkdir(parents=True, exist_ok=True)
    return load(
        name="baseline_hpc_decode_topscc_gcu300_v1",
        sources=[str(_EXTENSION_SOURCE)],
        extra_include_paths=[
            str(torch_gcu_root / "include"),
            "/opt/tops/include",
            "/opt/tops/include/gcu",
        ],
        extra_cflags=["-O2"],
        extra_ldflags=[
            f"-L{torch_gcu_root / 'lib'}",
            "-ltorch_gcu",
            "-L/opt/tops/lib",
            "-L/lib",
            "-ltopsrt",
            f"-Wl,-rpath,{torch_gcu_root / 'lib'}",
            "-Wl,-rpath,/opt/tops/lib",
            "-Wl,-rpath,/lib",
        ],
        build_directory=str(extension_build),
        verbose=False,
    )


def _pack_cache(cache: torch.Tensor, layout: str) -> torch.Tensor:
    if cache.ndim != 4:
        raise ValueError("K/V cache must be rank 4")
    layout = layout.upper()
    if layout not in {"NHD", "HND"}:
        raise ValueError("layout must be 'NHD' or 'HND'")

    # FlagAttention represents both layouts as [block, token, head, dim]; HND
    # changes only the strides. The upstream TopsCC kernel consumes the
    # contiguous physical form [block, head, token, dim].
    if cache.shape[1] in (16, 32, 64):
        return cache.permute(0, 2, 1, 3).contiguous()
    if layout == "HND" and cache.shape[2] in (16, 32, 64):
        return cache.contiguous()
    raise ValueError("K/V cache block size must be 16, 32, or 64")


@dataclass(frozen=True)
class TopsCCDecodePlan:
    query: torch.Tensor
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    block_tables: torch.Tensor
    context_lens: torch.Tensor
    max_context_len: int
    softmax_scale: float

    def __call__(self) -> torch.Tensor:
        extension = _load_extension()
        fatbin = _compile_device_kernel()
        return extension.forward(
            self.query,
            self.key_cache,
            self.value_cache,
            self.block_tables,
            self.context_lens,
            self.max_context_len,
            self.softmax_scale,
            str(fatbin),
        )


def prepare_topscc_decode_bf16(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    kv_lens: torch.Tensor,
    layout: str = "NHD",
    *,
    softmax_scale: float | None = None,
) -> TopsCCDecodePlan:
    """Prepare dynamic Decode metadata outside the measured kernel call."""

    if query.ndim != 3:
        raise ValueError("query must have shape [batch * mtp, heads, dim]")
    if query.dtype != torch.bfloat16:
        raise TypeError("the TopsCC baseline currently supports BF16 Decode")
    if block_tables.ndim != 2 or kv_lens.ndim != 1:
        raise ValueError("block_tables and kv_lens must have rank 2 and 1")
    batch = int(kv_lens.numel())
    if batch == 0 or query.shape[0] % batch:
        raise ValueError("query rows must be divisible by batch")
    mtp = int(query.shape[0] // batch)
    if query.shape[-1] not in (128, 256):
        raise ValueError("the S60 TopsCC kernel supports head size 128 or 256")

    packed_k = _pack_cache(key_cache, layout)
    packed_v = _pack_cache(value_cache, layout)
    if packed_k.shape != packed_v.shape:
        raise ValueError("K and V cache shapes must match")
    if packed_k.shape[-1] != query.shape[-1]:
        raise ValueError("query and K/V head dimensions must match")
    if query.shape[1] % packed_k.shape[1]:
        raise ValueError("query heads must be divisible by KV heads")
    tensors = (packed_k, packed_v, block_tables, kv_lens)
    if any(tensor.device != query.device for tensor in tensors):
        raise ValueError("all inputs must be on the same GCU device")
    if block_tables.dtype != torch.int32 or kv_lens.dtype != torch.int32:
        raise TypeError("block_tables and kv_lens must be int32")

    length_values = kv_lens.detach().cpu().tolist()
    if min(length_values) < mtp:
        raise ValueError("each KV length must contain every MTP query")
    block_size = int(packed_k.shape[2])
    required_blocks = max(math.ceil(int(value) / block_size) for value in length_values)
    if required_blocks > block_tables.shape[1]:
        raise ValueError("block table is too short for a KV length")

    if mtp == 1:
        expanded_blocks = block_tables.contiguous()
        context_lens = kv_lens.contiguous()
    else:
        expanded_blocks = (
            block_tables[:, None, :]
            .expand(batch, mtp, block_tables.shape[1])
            .reshape(batch * mtp, block_tables.shape[1])
            .contiguous()
        )
        offsets = torch.arange(
            1 - mtp,
            1,
            dtype=torch.int32,
            device=kv_lens.device,
        )
        context_lens = (kv_lens[:, None] + offsets[None, :]).reshape(-1)
        context_lens = context_lens.contiguous()

    maximum = max(int(value) for value in length_values)
    scale = query.shape[-1] ** -0.5 if softmax_scale is None else float(softmax_scale)
    return TopsCCDecodePlan(
        query=query.contiguous(),
        key_cache=packed_k,
        value_cache=packed_v,
        block_tables=expanded_blocks,
        context_lens=context_lens,
        max_context_len=maximum,
        softmax_scale=scale,
    )


@torch.no_grad()
def topscc_attention_decode_bf16_dynamic(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    kv_lens: torch.Tensor,
    layout: str = "NHD",
    *,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Run the native S60 TopsCC dynamic paged-attention Decode kernel."""

    plan = prepare_topscc_decode_bf16(
        query,
        key_cache,
        value_cache,
        block_tables,
        kv_lens,
        layout,
        softmax_scale=softmax_scale,
    )
    return plan()


__all__ = [
    "TopsCCDecodePlan",
    "prepare_topscc_decode_bf16",
    "topscc_attention_decode_bf16_dynamic",
    "topscc_decode_available",
]
