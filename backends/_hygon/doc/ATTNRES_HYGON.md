# ATTNRES_HYGON

AttnRes（Attention Residual）在海光 BW1000/HCU 上的 HIP/Triton 适配与优化记录。

## 1. 适配目标

- 在海光 HCU 上复现官方 AttnRes 数学语义；
- 保持 `output_norm`、多 source residual、padding 和 mask 行为一致；
- 支持 BF16 输入，并兼容 FlagTree Triton HCU 后端；
- 在正确性通过后，针对 BW1000 的 `warp_size=64`、软件流水和内存访问进行优化。

## 2. 目录结构

```text
backends/_hygon/
├── benchmarks/
│   └── benchmark_attnres_hygon.py
├── doc/
│   └── ATTNRES_HYGON.md
├── ops/attnres/
│   ├── attnres_hip.py
│   └── attnres_triton.py
└── test/attnres/
    └── test_attnres_hip.py
```

## 3. 第一步：导入与正确性基线

先固定官方 AttnRes 版本和参考实现，避免仓库更新引入无关后端导致导入失败。正确性基线应覆盖：

- 不带 output RMSNorm；
- 带 fused output RMSNorm；
- 不同 source 数量；
- padding residual；
- BF16 输入与 FP32 累加。

正确性命令：

```bash
cd /workspace/FlagAttention
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0
export DCU_VISIBLE_DEVICES=0
export PYTHONPATH=/workspace/FlagAttention/src

/usr/bin/python3.10 -m pytest tests/flag_attn/test_attnres.py -q -s
```

## 4. 第二步：关闭不兼容的异步加载

H100 路径中的 `tl.load(is_async=True)` 依赖 CUDA `cp.async/TMA`，在 HCU 后端没有对应的 LLVM lower，典型错误为：

```text
failed to translate module to LLVM IR
```

海光上的处理方式：

- 关闭 `is_async=True`；
- 使用 Triton `num_stages` 软件流水提前加载下一轮数据；
- 不在 wrapper 调用侧单独修补，因为 benchmark 可能绕过 wrapper 直接调用 kernel；
- 在 kernel 内部统一忽略非 NVIDIA 后端的 `ASYNC_LOAD`。

这样 wrapper、benchmark 和直接 kernel 调用都会使用同一条 HCU 兼容路径。

## 5. 第三步：修复 autotune 配置

AttnRes 在 `BD=8192` 的大布局上可能出现：

```text
No valid autotuner configs after pruning
```

需要针对 BW1000 调整配置：

- 去除依赖 CUDA warp/TMA 的配置；
- 使用 `warp_size=64` 对应的 wave 数；
- 删除 HCU 不支持的 `maxnreg` 等选项；
- 为 `BD=8192` 保留较浅的 `num_stages`；
- 将需要编译期判断的全局常量声明为 `tl.constexpr`，不能直接引用普通 Python 全局变量。

## 6. 第四步：消除 padding 带来的冗余 load

关闭异步路径后，编译器不会自动删除 padding source 的无效访存。原实现使用：

```python
for source in tl.static_range(...):
    residual = tl.load(..., mask=mask_d)
```

当真实 source 数为 `L`、padding 后长度更大时，`source=L..` 的 load 仍会实际执行，最后才被 `where` 丢弃。

优化为编译期剪枝：

```python
for source in tl.static_range(...):
    if source < L:
        residual = tl.load(...)
```

因为 `L` 和 `source` 都是编译期常量，越界 source 会被完全删除，不再产生无效 HBM 访问。

## 7. 第五步：autotune 专项调优

对 `BD=8192` 进行实扫，重点比较：

- `BLOCK_L` / `BLOCK_D`；
- `num_warps`；
- `num_stages`；
- 是否启用 output RMSNorm；
- source 数量和 residual 长度。

已观察到 `L9_N8192` 选择 `BLOCK_L=16、num_warps=16` 时约为 `2451.84 us`，相较原配置 `2586 us` 下降约 `5.2%`，并与实扫最优配置一致。

## 8. 正确性与 Benchmark 流程

完整正确性：

```bash
cd /workspace/FlagAttention
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0
export DCU_VISIBLE_DEVICES=0
export PYTHONPATH=/workspace/FlagAttention/src:/workspace/flash-linear-attention-attnres-baseline

/usr/bin/python3.10 -m pytest tests/flag_attn/test_attnres.py -q -p no:warnings
```

只运行 `L=9,N=8192` benchmark：

```bash
export FLAG_ATTN_RUN_EXTERNAL_BENCHMARKS=1
/usr/bin/python3.10 -m pytest tests/flag_attn/test_attnres.py \
  -k "kernel_benchmark and 9-8192" -q -s -p no:warnings \
  2>&1 | grep -E "speedup|passed|failed"
```

完整 benchmark 前，必须先准备指定的 FLA baseline commit：

```bash
git clone https://github.com/fla-org/flash-linear-attention.git \
  /workspace/flash-linear-attention-attnres-baseline
cd /workspace/flash-linear-attention-attnres-baseline
git checkout 5aea42b7740f9968f6418c6c60b78ea785ce6140
git rev-parse HEAD
```

## 9. 当前问题与下一步

- `is_async=True` 不能在 HCU 上使用，必须长期保持软件流水替代；
- `BD=8192` 是最容易触发编译器和 autotune 问题的形状，应保留专用配置；
- padding source 的无效 load 会显著影响大布局，应持续检查所有 `static_range` 循环；
- 后续可继续融合 residual load、加法和 RMSNorm，减少中间 tensor；
- 可针对 `L=2/4/9` 建立专用 kernel，避免统一 padding；
- 可按 `N=8192/16384` 分离 autotune 配置；
- 在确认 Triton 生成矩阵/向量指令后，再尝试更深软件流水，不能直接恢复 CUDA 异步原语。

## 10. 验收标准

只有同时满足以下条件，才记录为最终版本：

1. 完整正确性测试通过；
2. `BD=8192` 和普通布局均能编译运行；
3. benchmark 不再出现 `No valid autotuner configs`；
4. 关闭 async 后没有冗余 padding load；
5. 优化版性能相对基线不回退，并记录 shape、dtype、warmup、iterations 和配置。
