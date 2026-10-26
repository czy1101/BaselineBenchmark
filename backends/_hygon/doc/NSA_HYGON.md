# NSA_HYGON：海光 BW1000 适配与优化说明

本文记录 NSA（Native Sparse Attention）在海光 BW1000/HCU（gfx936）上的最终实现、优化过程、正确性测试和性能测试方法。

## 1. 最终提交目录

提交到：

```text
BaselineBenchmark/backends/_hygon/
├── benchmarks/
│   └── benchmark_nsa_hygon.py
├── ops/nsa/
│   ├── nsa_hip.py
│   ├── nsa_triton.py
│   └── nsa_hip_kernel.cu
└── test/nsa/
    ├── test_nsa_hip.py
    └── test_nsa_triton.py
```

文件职责：

| 文件 | 作用 |
|---|---|
| `ops/nsa/nsa_hip.py` | NSA 公共 Python 接口、压缩分支、selected 分支、滑窗分支和 Triton/原生 HIP dispatch |
| `ops/nsa/nsa_triton.py` | 海光 Triton selected attention kernel；GQA 融合、online softmax、`tl.dot` QK/PV 计算 |
| `ops/nsa/nsa_hip_kernel.cu` | 原生 HIP fallback kernel；用于不满足 Triton 条件的路径 |
| `test/nsa/test_nsa_hip.py` | 标准 NSA 正确性测试：selected、compression/LSE、packed varlen、FP16/BF16 |
| `test/nsa/test_nsa_triton.py` | 显式覆盖 Triton `block_size=64` 和 query-tile=1/2/4 的正确性测试 |
| `benchmarks/benchmark_nsa_hygon.py` | 官方 7 组 workload 的延迟、显存和相对比值测试 |

不需要提交：`__pycache__/`、本地 Triton/torch extension cache、临时 probe 脚本和旧版本备份文件。

## 2. 海光后端约束

- 设备：BW1000，`gfx936`，wavefront size 为 64。
- 运行时：PyTorch 2.9.0，HIP 6.3。
- 公开 `hipcc` 编译探测未找到 `__builtin_amdgcn_mfma_f32_16x16x16f16/bf16`，因此没有依赖手写 MFMA intrinsic。
- 环境中可以使用 FlagTree 提供的 Triton 3.6；`tl.dot` 在 HCU 上由 Triton backend lowering 到设备代码。
- Triton 路径要求 `block_size=64`、`block_indices=int32`、输入 contiguous，且 `D/DV` 属于 kernel 支持的 tile 集合。
- 不满足上述条件时自动走 `nsa_hip.py` 的原生 HIP/DTK fallback。

## 3. 优化过程

### 3.1 初始实现

最初版本使用标量 HIP C++ 循环逐 token、逐 query head、逐 selected block 计算，虽然能够复现数学语义，但大量随机 global memory 访问和串行循环使延迟达到数百毫秒，不能作为生产性能版本。

### 3.2 GQA 融合 Triton kernel

`nsa_triton.py` 将一个 program 映射为 `(batch, query_position, KV_head)`，一次处理同一个 KV head 共享的全部 `G=HQ/H` 个 query heads：

```text
QG [G,D] × K [D,block_size]  -> scores [G,block_size]
P  [G,block_size] × V [block_size,DV] -> output [G,DV]
```

这样避免了每个 query head 单独启动一个小矩阵乘，并且保留了 GQA/MQA 共享 K/V 的数据复用。

### 3.3 Online softmax 与 `exp2`

selected 分支使用 online softmax 的 `(m_i, l_i, acc)` 更新，不产生完整 attention matrix；累加在 FP32 中完成，最后转换回输入 dtype。由于 kernel 将分数乘以 `log2(e)`，指数计算使用 `tl.exp2`，避免 `exp` 路径的额外开销。

### 3.4 wave 数量调优

在 BW1000 上测试了 `num_warps=2/4/8`：

```text
warps=2：37.606 ms / 54.092 ms
warps=4：44.099 ms / 66.938 ms
warps=8：66.564 ms / 111.371 ms
```

最终保留 `NSA_HYGON_TRITON_WARPS=2`。

### 3.5 Query-tile 实验

增加了 `NSA_HYGON_TRITON_QTILE`，允许一个 program 循环处理连续 query position，以减少调度数量。该选项可以取 `1/2/4/8`。

由于不同 workload 的完整测试存在波动，最终默认恢复为：

```bash
export NSA_HYGON_TRITON_QTILE=1
```

`QTILE=2/4` 仅保留为显式 A/B 实验选项，不作为最终默认配置。

### 3.6 Block-major KV packing 实验

实现过 `NSA_HYGON_TRITON_BLOCK_PACK=1` 的 K/V block-major 预打包，但实际测试出现明显回退。因此最终配置固定关闭：

```bash
export NSA_HYGON_TRITON_BLOCK_PACK=0
```

## 4. 最终运行配置

```bash
cd /workspace/FlagGems-vllm/BaselineBenchmark

export TRITON_ROOT=/workspace/FlagTree
export PYTHONPATH=$TRITON_ROOT/build/lib.linux-x86_64-cpython-310:$TRITON_ROOT/python:$PWD/backends/_hygon/ops/nsa:$PYTHONPATH
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
export MAX_JOBS=8

export NSA_HYGON_USE_TRITON=1
export NSA_HYGON_TRITON_STRICT=1
export NSA_HYGON_TRITON_WARPS=2
export NSA_HYGON_TRITON_QTILE=1
export NSA_HYGON_TRITON_BLOCK_PACK=0
```

`NSA_HYGON_TRITON_STRICT=1` 用于让 Triton 编译/运行错误直接暴露；正式集成前建议保留一次 strict 测试，生产环境也可以改为 `0` 让不支持的输入自动 fallback。

## 5. 正确性测试

### 5.1 标准 HIP/NSA 测试

代码文件：`backends/_hygon/test/nsa/test_nsa_hip.py`。

该测试覆盖：

- FP16/BF16 selected attention；
- compression output 和 LSE；
- packed variable-length 输入；
- GQA (`HQ/H>1`)；
- `block_size=32/64` 等 fallback 场景；
- finite 输出和相对 RMSE 阈值。

执行：

```bash
python backends/_hygon/test/nsa/test_nsa_hip.py
```

期望输出最后包含：

```text
NSA HIP correctness: PASS
```

### 5.2 Triton selected 专项测试

代码文件：`backends/_hygon/test/nsa/test_nsa_triton.py`。

该测试构造 `B=1,T=127,H=2,HQ=32,D=64,block_size=64`，使用朴素 PyTorch selected-attention 作为参考，并依次测试：

```python
for qtile in (1, 2, 4):
    os.environ["NSA_HYGON_TRITON_QTILE"] = str(qtile)
    out = parallel_nsa(..., block_size=64)
    assert torch.isfinite(out).all()
    assert relative_rmse(out, reference) < 3e-2
```

执行：

```bash
python backends/_hygon/test/nsa/test_nsa_triton.py
```

期望输出包含：

```text
NSA Triton GQA/query-tile correctness: PASS
```

如果暂时没有复制专项测试文件，也可以执行已有临时测试：

```bash
python /tmp/test_nsa_triton_fixed.py
```

## 6. Benchmark 代码和命令

代码文件：`backends/_hygon/benchmarks/benchmark_nsa_hygon.py`。

默认官方 shape：

```python
SHAPES = [
    (1, 16384, 4, 64, 64),
    (1, 8192, 16, 256, 64),
    (1, 16384, 16, 256, 64),
    (1, 65536, 16, 256, 64),
    (1, 16384, 32, 512, 64),
    (1, 16384, 16, 256, 128),
    (4, 8192, 16, 256, 64),
]
```

Benchmark 统计 `warmup` 后多次运行的 p50 延迟、峰值显存，并与脚本内的 Hygon native/TLE 参考数据计算比值：

```python
for _ in range(warmup):
    run()
torch.cuda.synchronize()
samples = []
for _ in range(iterations):
    start = time.perf_counter()
    run()
    torch.cuda.synchronize()
    samples.append((time.perf_counter() - start) * 1e3)
latency_ms = statistics.median(samples)
```

完整测试命令：

```bash
python backends/_hygon/benchmarks/benchmark_nsa_hygon.py \
  --dtype both \
  --start 0 \
  --end 7 \
  --block-size 64 \
  --topk 16 \
  --warmup 10 \
  --iterations 30
```

单 workload 快速测试：

```bash
python backends/_hygon/benchmarks/benchmark_nsa_hygon.py \
  --dtype fp16 \
  --shape 1 16384 4 64 64 \
  --shape 1 8192 16 256 64 \
  --warmup 10 \
  --iterations 30
```

## 7. Query-tile A/B 命令（可选）

最终提交不依赖该实验；如需复测：

```bash
for q in 1 2 4; do
  export NSA_HYGON_TRITON_QTILE=$q
  echo "===== QTILE=$q ====="
  python backends/_hygon/benchmarks/benchmark_nsa_hygon.py \
    --dtype fp16 \
    --shape 1 16384 4 64 64 \
    --shape 1 8192 16 256 64 \
    --warmup 10 \
    --iterations 30
done
```

对比结束后必须恢复：

```bash
export NSA_HYGON_TRITON_QTILE=1
export NSA_HYGON_TRITON_BLOCK_PACK=0
```

## 8. 最终结论

最终保留的是 Triton-on-HIP GQA selected kernel 加原生 HIP fallback，而不是仅使用标量 HIP kernel。最终默认配置为：

```text
NSA_HYGON_USE_TRITON=1
NSA_HYGON_TRITON_STRICT=1
NSA_HYGON_TRITON_WARPS=2
NSA_HYGON_TRITON_QTILE=1
NSA_HYGON_TRITON_BLOCK_PACK=0
```

该实现保留 NSA 的压缩、selected、滑窗三路功能语义；Triton 只替换满足条件的 selected 固定长度路径，其余输入仍由 `nsa_hip.py` 负责 fallback，因此不会改变公共 Python 接口。
