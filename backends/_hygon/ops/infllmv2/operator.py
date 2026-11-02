# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Native InfLLM-V2 public APIs, execution plans, components and DTK loading."""

import ctypes
import ctypes as C
import hashlib
import json
import math
import os
import subprocess
import weakref
from collections import OrderedDict
from dataclasses import dataclass, fields
from functools import lru_cache
from pathlib import Path

import torch


def check(lib, rc):
    if rc:
        raise RuntimeError(lib.native_error(rc).decode())


@lru_cache(maxsize=1)
def _mmac_library():
    import fcntl

    root = Path(__file__).resolve().parent
    cc = os.environ.get("HIPCC", "/opt/dtk/bin/hipcc")
    version = subprocess.check_output([cc, "--version"], text=True)
    flags = [
        "-std=c++17",
        "-O3",
        "--offload-arch=gfx936",
        "-shared",
        "-fPIC",
        "-save-temps",
        "-Rpass-analysis=kernel-resource-usage",
        "-DWARP_SIZE_GPU=64",
    ]
    h = hashlib.sha256((version + os.path.realpath(cc) + repr(flags)).encode())
    for path in sorted(root.rglob("*")):
        if (
            path.is_file()
            and (
                path.name in {"forward.hip", "backward.hpp", "common.hpp"}
                or path.relative_to(root).parts[0] == "third_party"
            )
            and path.suffix in {".h", ".hpp", ".hip"}
        ):
            h.update(path.relative_to(root).as_posix().encode())
            h.update(path.read_bytes())
    folder = (
        Path(
            os.environ.get(
                "INFLLMV2_NATIVE_V3_BUILD_DIR",
                str(Path.home() / ".cache/flag_attn/infllmv2-native-base-v3"),
            )
        )
        / h.hexdigest()[:24]
    )
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / "native.so"
    with (folder / "build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not out.exists():
            cmd = [
                cc,
                *flags,
                "-I" + str(root / "third_party/hytlass/include"),
                str(root / "forward.hip"),
                "-o",
                str(folder / "native.tmp.so"),
            ]
            with (folder / "build.log").open("w") as log:
                p = subprocess.run(
                    cmd, cwd=folder, stdout=log, stderr=subprocess.STDOUT, timeout=600
                )
            if p.returncode:
                raise RuntimeError(f"DTK compilation failed: {folder / 'build.log'}")
            (folder / "native.tmp.so").replace(out)
            (folder / "build.json").write_text(
                json.dumps(
                    dict(
                        command=cmd,
                        compiler_version=version,
                        target="gfx936",
                        source_key=h.hexdigest(),
                        so_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
                    ),
                    indent=2,
                )
            )
    lib = ctypes.CDLL(str(out))
    V, I, F = ctypes.c_void_p, ctypes.c_int, ctypes.c_float
    signatures = dict(
        native_compress=[V] * 3 + [I] * 5 + [V],
        native_stage1=[V] * 5 + [I] * 7 + [F, V],
        native_pool=[V] * 3 + [I] * 8 + [V],
        native_topk=[V] * 3 + [I] * 6 + [V],
        native_attention=[V] * 8 + [I] * 9 + [F, V],
        native_backward=[V] * 13 + [I] * 9 + [F, V],
    )
    for name, signature in signatures.items():
        fn = getattr(lib, name)
        fn.argtypes = signature
        fn.restype = I
    lib.native_error.argtypes = [I]
    lib.native_error.restype = ctypes.c_char_p
    lib.build_folder = folder
    return lib


class Params(C.Structure):
    _fields_ = [
        (x, C.c_int)
        for x in [
            "d",
            "hq",
            "hkv",
            "type",
            "topk",
            "bs",
            "nb",
            "maxn1",
            "kw1",
            "st1",
            "kw2",
            "st2",
            "initial",
            "local",
            "dense",
            "batch",
            "tq",
            "tk",
            "n1",
            "n2",
        ]
    ] + [("scale", C.c_float)]


@lru_cache(maxsize=1)
def _generic_library():
    import fcntl

    root = Path(__file__).resolve().parent
    cc = os.environ.get("HIPCC", "/opt/dtk/bin/hipcc")
    version = subprocess.check_output([cc, "--version"], text=True)
    flags = ["-std=c++17", "-O3", "--offload-arch=gfx936", "-shared", "-fPIC"]
    source = root / "generic.hip"
    digest = hashlib.sha256(
        (version + os.path.realpath(cc) + repr(flags)).encode() + source.read_bytes()
    ).hexdigest()
    folder = (
        Path(
            os.environ.get(
                "INFLLMV2_NATIVE_V3_GENERIC_BUILD_DIR",
                str(Path.home() / ".cache/flag_attn/infllmv2-native-base-v3-generic"),
            )
        )
        / digest[:24]
    )
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / "generic.so"
    with (folder / "build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not out.exists():
            cmd = [cc, *flags, str(source), "-o", str(folder / "generic.tmp.so")]
            with (folder / "build.log").open("w") as log:
                p = subprocess.run(
                    cmd, cwd=folder, stdout=log, stderr=subprocess.STDOUT, timeout=600
                )
            if p.returncode:
                raise RuntimeError(
                    f"DTK generic compilation failed: {folder / 'build.log'}"
                )
            (folder / "generic.tmp.so").replace(out)
            (folder / "build.json").write_text(
                json.dumps(
                    dict(
                        command=cmd,
                        compiler_version=version,
                        target="gfx936",
                        source_key=digest,
                        so_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
                    ),
                    indent=2,
                )
            )
    lib = C.CDLL(str(out))
    V = C.c_void_p
    I = C.c_int
    for name, signature in dict(
        generic_compress=[V] * 4 + [I, V],
        generic_stage=[I] + [V] * 13,
        generic_backward=[V] * 16,
    ).items():
        fn = getattr(lib, name)
        fn.argtypes = signature
        fn.restype = I
    lib.generic_error.argtypes = [I]
    lib.generic_error.restype = C.c_char_p
    lib.native_error = lib.generic_error
    lib.build_folder = folder
    return lib


@dataclass(frozen=True)
class NativeConfig:
    k1_kernel_size: int = 32
    k1_stride: int = 16
    k2_kernel_size: int = 128
    k2_stride: int = 64
    block_size: int = 64
    topk: int = 64
    init_blocks: int = 1
    local_blocks: int = 32
    dense_len: int = 8192
    causal: bool = True


def configuration(config=None):
    if config is None:
        return NativeConfig()
    c = NativeConfig(**{f.name: getattr(config, f.name) for f in fields(NativeConfig)})
    if not c.causal:
        raise NotImplementedError("InfLLMv2 supports causal mode only")
    return c


_length_cache = OrderedDict()


def lengths(cu, allow_empty=False):
    if cu.ndim != 1 or cu.dtype != torch.int32 or not cu.is_cuda:
        raise ValueError("Cumulative lengths must be one-dimensional GPU int32 tensors")
    try:
        version = cu._version
    except RuntimeError:
        version = None
    key = (id(cu), version, allow_empty)
    entry = _length_cache.get(key) if version is not None else None
    if entry is not None and entry[0]() is cu:
        return entry[1]
    values = cu.detach().cpu().tolist()
    if (
        len(values) < 2
        or values[0] != 0
        or any(a > b if allow_empty else a >= b for a, b in zip(values, values[1:]))
    ):
        raise ValueError(
            "Cumulative lengths must start at zero and be strictly increasing"
        )
    result = tuple(b - a for a, b in zip(values, values[1:]))
    if version is not None:
        _length_cache[key] = (weakref.ref(cu), result)
        while len(_length_cache) > 64:
            _length_cache.popitem(last=False)
    return result


class NativePlan:
    """Immutable metadata. MMAC common path and native HIP general path."""

    def __init__(
        self,
        q_lengths,
        kv_lengths=None,
        *,
        head_dim=128,
        hq=32,
        hkv=2,
        dtype=torch.bfloat16,
        device="cuda:0",
        config=None,
        softmax_scale=None,
        max_seqlen_k=None,
        _compressed_lengths=None,
    ):
        if isinstance(q_lengths, int):
            q_lengths = (q_lengths,)
        if kv_lengths is None:
            kv_lengths = q_lengths
        if isinstance(kv_lengths, int):
            kv_lengths = (kv_lengths,)
        self.qlens = tuple(q_lengths)
        self.klens = tuple(kv_lengths)
        if (
            not self.qlens
            or len(self.qlens) != len(self.klens)
            or any(not 0 < a <= b for a, b in zip(self.qlens, self.klens))
        ):
            raise ValueError(
                "Expected matching nonempty batches with 0<Q<=KV per sequence"
            )
        if min(head_dim, hq, hkv) <= 0 or hq % hkv:
            raise ValueError("Expected positive head sizes and Hq divisible by Hkv")
        if dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise TypeError("Native v3 supports BF16, FP16 and FP32")
        self.d = head_dim
        self.hq = hq
        self.hkv = hkv
        self.g = hq // hkv
        self.dtype = dtype
        self.bf = int(dtype == torch.bfloat16)
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("A Hygon HIP device is required")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        p = torch.cuda.get_device_properties(self.device)
        if not torch.version.hip or not p.gcnArchName.startswith("gfx936"):
            raise NotImplementedError("Native v3 is compiled for gfx936")
        self.config = configuration(config)
        self.scale = head_dim**-0.5 if softmax_scale is None else float(softmax_scale)
        if not math.isfinite(self.scale):
            raise ValueError("softmax_scale must be finite")
        self.maxq = max(self.qlens)
        self.maxk = max(self.klens)
        self.totalq = sum(self.qlens)
        self.totalk = sum(self.klens)
        self.batch = len(self.qlens)
        self.declared_maxk = self.maxk if max_seqlen_k is None else int(max_seqlen_k)
        if self.declared_maxk < self.maxk:
            raise ValueError(
                "maximum sequence lengths do not cover the cumulative lengths"
            )
        self.dense = self.declared_maxk < self.config.dense_len
        c = self.config
        if not self.dense:
            if any(
                not isinstance(x, int) or x <= 0
                for x in (
                    c.k1_kernel_size,
                    c.k1_stride,
                    c.k2_kernel_size,
                    c.k2_stride,
                    c.block_size,
                )
            ):
                raise ValueError(
                    "Compression windows/strides and block size must be positive integers"
                )
            if c.k1_kernel_size % c.k1_stride or c.block_size % c.k1_stride:
                raise ValueError(
                    "K1 window and attention block must be divisible by K1 stride"
                )
            if (
                not 1 <= c.topk <= 2048
                or c.topk & (c.topk - 1)
                or min(c.init_blocks, c.local_blocks) < 0
            ):
                raise ValueError(
                    "Expected power-of-two TopK <=2048 and nonnegative forced block counts"
                )
        self.bs = 64 if self.dense else c.block_size
        self.topk = 1 if self.dense else c.topk
        self.nb = (self.declared_maxk + self.bs - 1) // self.bs
        if not self.dense and self.nb > 4096:
            raise NotImplementedError(
                "Sparse selection supports at most 4096 candidate blocks"
            )
        self.n1 = tuple(
            0 if self.dense else max(0, (k - c.k1_kernel_size) // c.k1_stride + 1)
            for k in self.klens
        )
        self.n2 = tuple(
            0 if self.dense else max(0, (k - c.k2_kernel_size) // c.k2_stride + 1)
            for k in self.klens
        )
        if _compressed_lengths is not None:
            self.n1, self.n2 = map(tuple, _compressed_lengths)
            if (
                len(self.n1) != self.batch
                or len(self.n2) != self.batch
                or min((*self.n1, *self.n2)) < 0
            ):
                raise ValueError("Invalid compressed lengths")
        self.maxn1 = max(self.n1)
        self.maxn2 = max(self.n2)
        common = (
            c.k1_kernel_size,
            c.k1_stride,
            c.k2_kernel_size,
            c.k2_stride,
            c.block_size,
        ) == (32, 16, 128, 64, 64)
        self.fast = (
            dtype in (torch.bfloat16, torch.float16)
            and head_dim in (64, 128)
            and hkv == 2
            and hq in (8, 32)
            and (
                self.dense
                or (
                    common
                    and c.topk <= 64
                    and self.nb <= 1024
                    and _compressed_lengths is None
                )
            )
        )
        self.backend = "mmac" if self.fast else "generic_hip"
        if (
            max(
                self.totalq,
                self.totalk,
                sum(self.n1),
                sum(self.n2),
                self.totalq * self.hq * ((self.d + 63) // 64),
                self.totalq * self.hkv * self.maxn1,
            )
            > 2147483647
        ):
            raise ValueError("Metadata/kernel grid exceeds signed int32 indexing")
        meta = []
        qs = ks = c1 = c2 = 0
        for nq, nk, n1, n2 in zip(self.qlens, self.klens, self.n1, self.n2):
            meta.append([qs, nq, ks, nk, c1, n1, c2, n2])
            qs += nq
            ks += nk
            c1 += n1
            c2 += n2
        self.total1 = c1
        self.total2 = c2
        self.splits = 1
        if self.fast and not self.dense and self.maxq == 1:
            target = min(
                64,
                self.config.topk,
                max(1, (512 + 2 * self.batch - 1) // (2 * self.batch)),
            )
            self.splits = 1 << (target.bit_length() - 1)
        if self.fast:
            self.lib = _mmac_library()
        else:
            self.lib = _generic_library()
            self.params = Params(
                self.d,
                self.hq,
                self.hkv,
                {torch.bfloat16: 0, torch.float16: 1, torch.float32: 2}[self.dtype],
                self.topk,
                self.bs,
                self.nb,
                self.maxn1,
                c.k1_kernel_size,
                c.k1_stride,
                c.k2_kernel_size,
                c.k2_stride,
                c.init_blocks,
                c.local_blocks,
                int(self.dense),
                self.batch,
                self.totalq,
                self.totalk,
                self.total1,
                self.total2,
                self.scale,
            )
        with torch.cuda.device(self.device):
            self.meta = torch.tensor(meta, device=self.device, dtype=torch.int32)

            def cu(ls):
                values = [0]
                for x in ls:
                    values.append(values[-1] + x)
                return torch.tensor(values, device=self.device, dtype=torch.int32)

            self.cuq = cu(self.qlens)
            self.cuk = cu(self.klens)
            self.cu1 = cu(self.n1)
            self.cu2 = cu(self.n2)
            self.ready = torch.cuda.Event()
            self.ready.record()

    def check_inputs(self, q, k, v):
        for x, shape in zip(
            (q, k, v),
            (
                (self.totalq, self.hq, self.d),
                (self.totalk, self.hkv, self.d),
                (self.totalk, self.hkv, self.d),
            ),
        ):
            if (
                tuple(x.shape) != shape
                or x.dtype != self.dtype
                or x.device != self.device
                or not x.is_contiguous()
                or (self.fast and x.data_ptr() % 8)
            ):
                raise ValueError(
                    f"Expected aligned contiguous {self.dtype} {shape} on {self.device}"
                )

    def empty(self, shape, dtype=None):
        return torch.empty(
            shape, device=self.device, dtype=self.dtype if dtype is None else dtype
        )

    def allocate(self):
        b = dict(
            out=self.empty((self.totalq, self.hq, self.d)),
            lse=self.empty((self.totalq, self.hq), torch.float32),
        )
        if not self.dense:
            b.update(
                k1=self.empty((self.total1, self.hkv, self.d)),
                k2=self.empty((self.total2, self.hkv, self.d)),
                scores=self.empty((self.hkv, self.totalq, self.maxn1)),
                pooled=self.empty((self.hkv, self.totalq, self.nb)),
                selected=self.empty((self.hkv, self.totalq, self.topk), torch.int32),
            )
        else:
            b["selected"] = self.empty((0,), torch.int32)
        if self.splits > 1:
            b["partial"] = self.empty(
                (self.splits * self.totalq * self.hq * (self.d + 2),), torch.float32
            )
        return b

    def _record(self, stream, *tensors):
        for t in (self.meta, self.cuq, self.cuk, self.cu1, self.cu2, *tensors):
            t.record_stream(stream)

    def launch(self, name, q, k, v, b):
        stream = torch.cuda.current_stream(self.device)
        stream.wait_event(self.ready)
        p = lambda x: x.data_ptr()
        m = p(self.meta)
        sp = stream.cuda_stream
        lib = self.lib
        if not self.fast:
            import ctypes

            stage = {"compress": 0, "stage1": 1, "pool": 2, "topk": 3, "attention": 4}[
                name
            ]
            args = [
                p(b[x]) if x in b else None
                for x in ("k1", "k2", "scores", "pooled", "selected", "out", "lse")
            ]
            rc = lib.generic_stage(
                stage, p(q), p(k), p(v), *args, m, ctypes.byref(self.params), sp
            )
        elif name == "compress":
            check(
                lib,
                lib.native_compress(
                    p(k), p(b["k1"]), m, self.maxn1, self.batch, self.d, self.bf, 1, sp
                ),
            )
            rc = lib.native_compress(
                p(k), p(b["k2"]), m, self.maxn2, self.batch, self.d, self.bf, 2, sp
            )
        elif name == "stage1":
            rc = lib.native_stage1(
                p(q),
                p(b["k1"]),
                p(b["k2"]),
                p(b["scores"]),
                m,
                self.maxq,
                self.batch,
                self.totalq,
                self.maxn1,
                self.d,
                self.bf,
                self.g,
                self.scale,
                sp,
            )
        elif name == "pool":
            rc = lib.native_pool(
                p(b["scores"]),
                p(b["pooled"]),
                m,
                self.maxq,
                self.batch,
                self.totalq,
                self.maxn1,
                self.nb,
                self.config.init_blocks,
                self.config.local_blocks,
                self.bf,
                sp,
            )
        elif name == "topk":
            rc = lib.native_topk(
                p(b["pooled"]),
                p(b["selected"]),
                m,
                self.maxq,
                self.batch,
                self.totalq,
                self.nb,
                self.topk,
                self.bf,
                sp,
            )
        elif name == "attention":
            rc = lib.native_attention(
                p(q),
                p(k),
                p(v),
                p(b["selected"]),
                p(b["out"]),
                p(b["lse"]),
                m,
                p(b["partial"]) if "partial" in b else None,
                self.maxq,
                self.batch,
                self.totalq,
                self.topk,
                self.d,
                self.bf,
                self.g,
                int(self.dense),
                self.splits,
                self.scale,
                sp,
            )
        else:
            raise ValueError(name)
        check(lib, rc)
        self._record(stream, q, k, v, *b.values())

    def forward_buffers(self, q, k, v):
        self.check_inputs(q, k, v)
        with torch.cuda.device(self.device), torch.no_grad():
            b = self.allocate()
            for name in (
                ("attention",)
                if self.dense
                else ("compress", "stage1", "pool", "topk", "attention")
            ):
                self.launch(name, q, k, v, b)
        return b

    def __call__(self, q, k, v, *, intermediates=False):
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        self.check_inputs(q, k, v)
        if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)):
            if intermediates:
                raise ValueError(
                    "Use forward_buffers for detached diagnostic intermediates"
                )
            return _Attention.apply(q, k, v, self)
        b = self.forward_buffers(q, k, v)
        return b if intermediates else b["out"]

    def backward(self, q, k, v, out, dout, lse, selected):
        dout = dout.contiguous()
        if not self.fast:
            import ctypes

            with torch.cuda.device(self.device):
                acc = [self.empty(t.shape, torch.float32) for t in (q, k, v)]
                grads = (
                    acc
                    if self.dtype == torch.float32
                    else [torch.empty_like(t) for t in (q, k, v)]
                )
                stream = torch.cuda.current_stream(self.device)
                stream.wait_event(self.ready)
                args = [
                    t.data_ptr()
                    for t in (
                        q,
                        k,
                        v,
                        out,
                        dout,
                        lse,
                        selected,
                        self.meta,
                        *acc,
                        *grads,
                    )
                ]
                check(
                    self.lib,
                    self.lib.generic_backward(
                        *args, ctypes.byref(self.params), stream.cuda_stream
                    ),
                )
                self._record(stream, q, k, v, out, dout, lse, selected, *acc, *grads)
            return tuple(grads)
        with torch.cuda.device(self.device):
            dq = torch.empty_like(q)
            dk = torch.empty_like(k)
            dv = torch.empty_like(v)
            delta = self.empty((self.totalq, self.hq), torch.float32)
            membership = self.empty(
                (1,) if self.dense else (self.batch, 2, self.nb, self.maxq), torch.uint8
            )
            stream = torch.cuda.current_stream(self.device)
            stream.wait_event(self.ready)
            p = lambda x: x.data_ptr()
            args = [
                p(t)
                for t in (
                    q,
                    k,
                    v,
                    out,
                    dout,
                    lse,
                    selected,
                    self.meta,
                    dq,
                    dk,
                    dv,
                    delta,
                    membership,
                )
            ]
            check(
                self.lib,
                self.lib.native_backward(
                    *args,
                    self.maxq,
                    self.batch,
                    self.totalq,
                    self.nb,
                    self.topk,
                    self.d,
                    self.bf,
                    self.g,
                    int(self.dense),
                    self.scale,
                    stream.cuda_stream,
                ),
            )
            self._record(
                stream, q, k, v, out, dout, lse, selected, dq, dk, dv, delta, membership
            )
        return dq, dk, dv


class _Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, plan):
        b = plan.forward_buffers(q, k, v)
        ctx.plan = plan
        ctx.save_for_backward(q, k, v, b["out"], b["lse"], b["selected"])
        return b["out"]

    @staticmethod
    def backward(ctx, dout):
        if torch.is_grad_enabled():
            raise NotImplementedError("Native v3 supports first-order gradients only")
        q, k, v, out, lse, selected = ctx.saved_tensors
        return (*ctx.plan.backward(q, k, v, out, dout, lse, selected), None)


_plans = OrderedDict()


def get_plan(ql, kl, q, config=None, softmax_scale=None, max_seqlen_k=None, hkv=2):
    config = configuration(config)
    key = (
        tuple(ql),
        tuple(kl),
        q.shape[-1],
        q.shape[1],
        q.dtype,
        q.device,
        config,
        softmax_scale,
        max_seqlen_k,
        hkv,
    )
    plan = _plans.get(key)
    if plan is None:
        plan = NativePlan(
            ql,
            kl,
            head_dim=q.shape[-1],
            hq=q.shape[1],
            hkv=hkv,
            dtype=q.dtype,
            device=q.device,
            config=config,
            softmax_scale=softmax_scale,
            max_seqlen_k=max_seqlen_k,
        )
        _plans[key] = plan
        while len(_plans) > 32:
            _plans.popitem(last=False)
    return plan


def infllmv2_attention(q, k, v, *args, stage="prefill", plan=None, **kwargs):
    if plan is not None:
        if args or kwargs:
            raise ValueError(
                "A prepared plan is exclusive with metadata/config/scale overrides"
            )
        if stage not in ("prefill", "decode"):
            raise ValueError("Unsupported stage")
        if stage == "decode" and plan.maxq != 1:
            raise ValueError("decode plan must have one query per sequence")
        return plan(q, k, v)
    if stage == "decode":
        return infllmv2_decode(q, k, v, *args, **kwargs)
    if stage != "prefill":
        raise ValueError("Unsupported stage")
    return _prefill_attention(q, k, v, *args, **kwargs)


def _prefill_attention(
    q,
    k,
    v,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    *,
    config=None,
    softmax_scale=None,
):
    if (
        max_seqlen_q is not None
        and max_seqlen_q <= 0
        or max_seqlen_k is not None
        and max_seqlen_k <= 0
    ):
        raise ValueError("maximum sequence lengths must be positive")
    if q.ndim != 3 or k.ndim != 3 or v.shape != k.shape:
        raise ValueError("Expected packed Q/K/V tensors")
    if cu_seqlens_q is None and cu_seqlens_k is None:
        ql, kl = (q.shape[0],), (k.shape[0],)
    elif cu_seqlens_q is None or cu_seqlens_k is None:
        raise ValueError("Both cumulative length arrays are required")
    else:
        if cu_seqlens_q.device != q.device or cu_seqlens_k.device != q.device:
            raise ValueError("Cumulative lengths must share the Q device")
        ql, kl = lengths(cu_seqlens_q), lengths(cu_seqlens_k)
    if (
        max_seqlen_q is not None
        and max(ql) > max_seqlen_q
        or max_seqlen_k is not None
        and max(kl) > max_seqlen_k
    ):
        raise ValueError("maximum sequence lengths do not cover the cumulative lengths")
    return get_plan(ql, kl, q, config, softmax_scale, max_seqlen_k, k.shape[1])(q, k, v)


def infllmv2_decode(
    q, k_cache, v_cache, cu_seqlens_k, max_seqlen_k, *, config=None, softmax_scale=None
):
    if q.ndim != 3 or q.shape[0] != cu_seqlens_k.numel() - 1:
        raise ValueError("decode q must have [batch, Hq, D] layout")
    if cu_seqlens_k.device != q.device:
        raise ValueError("Cumulative lengths must share the Q device")
    kl = lengths(cu_seqlens_k)
    if max_seqlen_k <= 0 or max(kl) > max_seqlen_k:
        raise ValueError("maximum sequence lengths are invalid")
    return get_plan(
        (1,) * len(kl), kl, q, config, softmax_scale, max_seqlen_k, k_cache.shape[1]
    )(q, k_cache, v_cache)


def _plan(
    ql,
    kl,
    dtype,
    device,
    *,
    d=64,
    hq=32,
    hkv=2,
    config=None,
    scale=None,
    maxk=None,
    compressed=None,
):
    return NativePlan(
        ql,
        kl,
        head_dim=d,
        hq=hq,
        hkv=hkv,
        dtype=dtype,
        device=device,
        config=config or NativeConfig(dense_len=0),
        softmax_scale=scale,
        max_seqlen_k=maxk,
        _compressed_lengths=compressed,
    )


def _lengths(cuq, cuk, maxq=None):
    ql = lengths(cuq)
    kl = ql if cuk is None else lengths(cuk)
    if maxq is not None and maxq < max(ql):
        raise ValueError("max_seqlen_q is too small")
    if len(ql) != len(kl):
        raise ValueError("Packed batches differ")
    return ql, kl


def compress_k(k, cu_seqlens_k, kernel_size, stride):
    if k.ndim != 3 or cu_seqlens_k.device != k.device:
        raise ValueError("Invalid packed key layout/device")
    kl = lengths(cu_seqlens_k)
    if sum(kl) != k.shape[0]:
        raise ValueError("Cumulative lengths do not cover K")
    k = k.contiguous()
    cfg = NativeConfig(
        k1_kernel_size=kernel_size, k1_stride=stride, block_size=stride, dense_len=0
    )
    plan = _plan(
        (1,) * len(kl),
        kl,
        k.dtype,
        k.device,
        d=k.shape[-1],
        hq=k.shape[1],
        hkv=k.shape[1],
        config=cfg,
    )
    out = plan.empty((plan.total1, plan.hkv, plan.d))
    stream = torch.cuda.current_stream(plan.device)
    stream.wait_event(plan.ready)
    check(
        plan.lib,
        plan.lib.generic_compress(
            k.data_ptr(),
            out.data_ptr(),
            plan.meta.data_ptr(),
            ctypes.byref(plan.params),
            1,
            stream.cuda_stream,
        ),
    )
    plan._record(stream, k, out)
    return out, plan.cu1


def stage1(
    q,
    k1,
    k2,
    cu_seqlens_q,
    cu_seqlens_k1,
    cu_seqlens_k2,
    max_seqlen_q,
    k1_stride=16,
    k2_stride=64,
    softmax_scale=None,
    causal=True,
    cu_seqlens_k=None,
    **kwargs,
):
    if not causal:
        raise NotImplementedError("Only causal attention is supported")
    ql, kl = _lengths(cu_seqlens_q, cu_seqlens_k, max_seqlen_q)
    n1, n2 = lengths(cu_seqlens_k1, True), lengths(cu_seqlens_k2, True)
    hkv = k1.shape[1]
    cfg = NativeConfig(
        k1_kernel_size=k1_stride,
        k1_stride=k1_stride,
        k2_kernel_size=k2_stride,
        k2_stride=k2_stride,
        block_size=k1_stride,
        dense_len=0,
    )
    plan = _plan(
        ql,
        kl,
        q.dtype,
        q.device,
        d=q.shape[-1],
        hq=q.shape[1],
        hkv=hkv,
        config=cfg,
        scale=softmax_scale,
        compressed=(n1, n2),
    )
    if q.shape != (plan.totalq, plan.hq, plan.d):
        raise ValueError("Invalid Q layout")
    for x, shape in ((k1, (sum(n1), hkv, plan.d)), (k2, (sum(n2), hkv, plan.d))):
        if tuple(x.shape) != shape or x.dtype != q.dtype or x.device != q.device:
            raise ValueError("Invalid compressed key layout")
    q, k1, k2 = q.contiguous(), k1.contiguous(), k2.contiguous()
    out = plan.empty((hkv, plan.totalq, plan.maxn1))
    lse = plan.empty((plan.totalq, plan.hq), torch.float32)
    plan.launch("stage1", q, k1, k2, dict(k1=k1, k2=k2, scores=out, lse=lse))
    return out


def pool_scores(
    score,
    cu_seqlens_q,
    cu_seqlens_k1,
    max_seqlen_q,
    kernel_size=32,
    kernel_stride=16,
    block_size=64,
    init_blocks=1,
    local_blocks=32,
    cu_seqlens_k=None,
    max_seqlen_k=None,
    **kwargs,
):
    ql, kl = _lengths(cu_seqlens_q, cu_seqlens_k, max_seqlen_q)
    n1 = lengths(cu_seqlens_k1, True)
    cfg = NativeConfig(
        k1_kernel_size=kernel_size,
        k1_stride=kernel_stride,
        block_size=block_size,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        dense_len=0,
    )
    plan = _plan(
        ql,
        kl,
        score.dtype,
        score.device,
        hq=score.shape[0],
        hkv=score.shape[0],
        config=cfg,
        maxk=max_seqlen_k,
        compressed=(n1, (0,) * len(ql)),
    )
    if tuple(score.shape) != (plan.hkv, sum(ql), max(n1)):
        raise ValueError("Score layout does not match cumulative lengths")
    out = plan.empty((plan.hkv, sum(ql), plan.nb))
    plan.launch(
        "pool", score, score, score, dict(scores=score.contiguous(), pooled=out)
    )
    return out


def select_blocks(
    block_score, cu_seqlens_q, topk, block_size=64, cu_seqlens_k=None, **kwargs
):
    ql, kl = _lengths(cu_seqlens_q, cu_seqlens_k)
    if block_score.ndim != 3 or block_score.shape[1] != sum(ql):
        raise ValueError("Invalid pooled score layout")
    cfg = NativeConfig(
        k1_kernel_size=1, k1_stride=1, block_size=block_size, topk=topk, dense_len=0
    )
    plan = _plan(
        ql,
        kl,
        block_score.dtype,
        block_score.device,
        hq=block_score.shape[0],
        hkv=block_score.shape[0],
        config=cfg,
        maxk=block_score.shape[-1] * block_size,
    )
    out = plan.empty((plan.hkv, sum(ql), topk), torch.int32)
    plan.launch(
        "topk",
        block_score,
        block_score,
        block_score,
        dict(pooled=block_score.contiguous(), selected=out),
    )
    return out


def _fixed_plan(q, k, v, selected, cuq, cuk, maxq, bs, scale, causal):
    if not causal:
        raise NotImplementedError("Only causal attention is supported")
    ql, kl = _lengths(cuq, cuk, maxq)
    cfg = NativeConfig(
        k1_kernel_size=1,
        k1_stride=1,
        block_size=bs,
        topk=selected.shape[-1],
        dense_len=0,
    )
    plan = get_plan(ql, kl, q, cfg, scale, hkv=k.shape[1])
    plan.check_inputs(q, k, v)
    if (
        selected.shape != (plan.hkv, plan.totalq, plan.topk)
        or selected.dtype != torch.int32
        or selected.device != q.device
    ):
        raise ValueError(
            "Selected blocks must be packed int32 [Hkv,Tq,TopK] on the Q device"
        )
    return plan


def _run_fixed(plan, q, k, v, selected):
    b = dict(
        out=torch.empty_like(q),
        lse=plan.empty((plan.totalq, plan.hq), torch.float32),
        selected=selected,
    )
    if plan.splits > 1:
        b["partial"] = plan.empty(
            (plan.splits * plan.totalq * plan.hq * (plan.d + 2),), torch.float32
        )
    plan.launch("attention", q, k, v, b)
    return b["out"], b["lse"]


def _sparse_attention_forward(
    q,
    k,
    v,
    selected_blocks,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    block_size=64,
    softmax_scale=None,
    causal=True,
    store_lse=True,
    **kwargs,
):
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    selected_blocks = selected_blocks.contiguous()
    plan = _fixed_plan(
        q,
        k,
        v,
        selected_blocks,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        block_size,
        softmax_scale,
        causal,
    )
    out, lse = _run_fixed(plan, q, k, v, selected_blocks)
    return out, lse if store_lse else None


class _FixedAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, selected, plan):
        out, lse = _run_fixed(plan, q, k, v, selected)
        ctx.plan = plan
        ctx.save_for_backward(q, k, v, out, lse, selected)
        return out

    @staticmethod
    def backward(ctx, dout):
        if torch.is_grad_enabled():
            raise NotImplementedError(
                "Native sparse attention supports first-order gradients only"
            )
        q, k, v, out, lse, selected = ctx.saved_tensors
        return (*ctx.plan.backward(q, k, v, out, dout, lse, selected), None, None)


def sparse_attention(
    q,
    k,
    v,
    selected_blocks,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    block_size=64,
    softmax_scale=None,
    causal=True,
    **kwargs,
):
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    selected_blocks = selected_blocks.contiguous()
    plan = _fixed_plan(
        q,
        k,
        v,
        selected_blocks,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        block_size,
        softmax_scale,
        causal,
    )
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)):
        return _FixedAttention.apply(q, k, v, selected_blocks, plan)
    return _run_fixed(plan, q, k, v, selected_blocks)[0]
