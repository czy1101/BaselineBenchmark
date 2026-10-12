# Enflame Chunk GDN2 优化记录

更新时间：2026-08-28

## 1. 范围与环境

本文记录 GDN2 forward 在 Enflame S60 上的适配和优化过程，重点说明 BT=64 融合推理路径、S60 launch grid 适配、局部矩阵求解、最终性能以及当前限制。

当前实现涉及以下文件：

- S60 融合实现：`FlagAttention/src/flag_attn/runtime/backend/_enflame/gdn2/chunk_gdn2.py`
- S60 native Triton fallback：`FlagAttention/src/flag_attn/runtime/backend/_enflame/gdn2/native/chunk_fwd.py`
- S60 索引处理：`FlagAttention/src/flag_attn/runtime/backend/_enflame/gdn2/index.py`
- FlagAttention 正确性测试：`FlagAttention/tests/flag_attn/test_chunk_gdn2.py`
- FlagAttention benchmark：`FlagAttention/benchmark/test_chunk_gdn2.py`
- Torch GCU 对比实现：`BaselineBenchmark/backends/_enflame/ops/gdn2/forward.py`
- Torch GCU 输入验证：`BaselineBenchmark/backends/_enflame/ops/gdn2/validation.py`
- Torch/Triton 对比 benchmark：`BaselineBenchmark/backends/_enflame/benchmarks/test_chunk_gdn2_torch_gcu.py`
- Torch GCU 正确性测试：`BaselineBenchmark/backends/_enflame/test/test_chunk_gdn2_torch_gcu.py`

主要测试环境和输入约定如下：

| 项目 | 配置 |
| --- | --- |
| S60 设备 | Enflame S60 / GCU300 |
| 输入 dtype | FP16、BF16 |
| 输入布局 | `[B, T, H, K]` 和 `[B, T, H, V]` |
| 融合时间块 | `BT=64` |
| 支持的 K | `K <= 256` |
| initial state | FP32 |
| 性能表 shape 数 | FP16 12 组、BF16 12 组 |

附件中的阶段性记录将 K1 配置记为 `num_stages=1`，当前检入源码中的实际配置为 `num_stages=2`。本文的实现描述以当前源码为准；第 7 节性能表保留附件中的原始测试数据，不将其重新归因到未经复测的配置。

## 2. 数学定义与正确性红线

对于每个 token，GDN2 的状态递推可写为：

```text
S_decay = exp(g_t) * S_(t-1)
erase_t = sum_K((k_t * b_t) * S_decay)
update_t = v_t * w_t - erase_t
S_t = S_decay + k_t^T @ update_t
o_t = scale * q_t @ S_t
```

其中：

- `b_t` 是 K 维度上的 erase gate；
- `w_t` 是 V 维度上的 write gate；
- `g_t` 控制状态衰减；
- `S_t` 使用 FP32 保存和累加。

当启用 Kernel 内 gate 处理时：

```text
rate = exp(A_log)
g_t = -rate * softplus(g_t + dt_bias)
```

safe gate 路径则按 `lower_bound` 限制 gate 范围。启用 Q/K L2 normalize 时，normalize 也在融合 Kernel 内完成。

优化必须保持以下行为：

- erase gate 和 write gate 分别作用于 K、V 维度；
- fixed-length 和 packed varlen 的序列边界不互相污染；
- 有或无 `initial_state` 均可正确计算；
- `state_v_first=True/False` 只改变状态布局，不改变结果；
- 尾 chunk 的无效位置不得参与计算；
- output 和 final state 均通过独立 FP32 token recurrence 的精度门限；
- 不降低 FP32 state 累加精度换取性能。

## 3. 当前实现结构

GDN2 forward 使用 K1/K2 两阶段 chunk 算法。

### 3.1 K1：chunk 内融合预处理

K1 在不同 chunk、batch 和 head 间并行，主要完成：

1. 可选 Q/K L2 normalize；
2. 可选 gate activation 和 base-2 gate cumsum；
3. 计算 chunk 内 `Aqk` 和严格下三角 `Akk`；
4. 计算 `Akk` 对应的局部 inverse；
5. 生成 `w_wy`、`u_wy`、`qg` 和 `kg` 工作区。

K1 的主要矩阵计算为：

```text
Aqk = (q * gate_q) @ (k * gate_k)^T
Akk = (k * erase_gate * gate_q) @ (k * gate_k)^T
u_wy = inverse(Akk) @ (write_gate * v)
w_wy = inverse(Akk) @ (erase_gate * k * gate_q)
```

### 3.2 K2：状态传播与输出

K2 对每个 batch/head/value tile 按 chunk 顺序执行，主要完成：

```text
state_projection = w_wy @ state
v_new = u_wy - state_projection
output = scale * qg @ state + Aqk @ v_new
state = state * state_decay + kg^T @ v_new
```

不同 chunk 的 state 存在真实依赖，因此 chunk 维度保持顺序；同一 chunk 内的状态投影、输出和更新由矩阵计算完成。

### 3.3 融合路径与 fallback

公共入口 `chunk_gdn2` 当前提供两条路径：

| 路径 | 启用条件 | 作用 |
| --- | --- | --- |
| BT64 融合路径 | TLE 扩展可用且 `chunk_size=64` | S60 优化的 K1/K2 融合推理实现 |
| Native Triton fallback | TLE 不可用或没有选择 BT64 | 保留更通用的多 Kernel 实现 |

兼容旧调用时，如果 TLE 扩展可用且调用方传入 `chunk_size=16`，公共入口会自动映射为 64；varlen 的 `chunk_indices` 会按照新的 BT 重新生成。

当前 `chunk_gdn2.py` 虽然仍将优化路径称为 TLE path，并使用 TLE 扩展是否存在作为 dispatch 条件，但当前 S60 Kernel 主体使用的是 `tl.load`、`tl.store` 和 `tl.dot` 等 Triton 原语，文件内没有直接调用 `tle.pipe` 或 `tle.gpu.*`。

## 4. S60 功能适配

以下修改主要用于解决跨后端编译和 launch 兼容问题，属于适配工作。

### 4.1 移除 NVIDIA PTX softplus

原实现中的 NVIDIA PTX 汇编 softplus 无法在 Enflame 后端编译，替换为可移植 Triton 实现：

```python
@triton.jit
def softplus(x):
    return tl.where(
        x > 20.0,
        x,
        tl.log(1.0 + exp(x)),
    )
```

`x > 20` 时直接返回 `x`，既保持 softplus 的大数近似，也避免不必要的指数溢出。

### 4.2 固定配置替代大规模 autotune

Triton-GCU 在某个 autotune candidate 编译失败时，可能中断整组 autotune。S60 路径因此为每个 Kernel 使用一套已验证的固定配置，避免候选配置失败导致算子不可用。

当前检入源码的配置为：

```python
_S60_INTRA_BK = 256
_S60_INTRA_BV = 256
_S60_INTRA_NUM_WARPS = 1
_S60_INTRA_NUM_STAGES = 2

_S60_STATE_BK_SMALL = 64
_S60_STATE_BK_MEDIUM = 128
_S60_STATE_BK_LARGE = 256
_S60_STATE_BV = 256
_S60_STATE_NUM_WARPS = 1
_S60_STATE_NUM_STAGES = 2

_S60_FUSED_BT = 64
```

固定配置既是稳定性适配，也包含基于 S60 实测选择 tile 的性能优化。

### 4.3 Enflame launch grid 适配

S60 的 `grid.y` 和 `grid.z` 上限为 255。K1 实现三种 grid 模式：

```text
_GRID_MODE_ORIGINAL = 0
_GRID_MODE_SWAP_BH = 1
_GRID_MODE_SPLIT_BH = 2
```

K1 根据 `NT` 和 `B×H` 选择：

| 模式 | Grid |
| --- | --- |
| 正常 | `(NT, B×H)` |
| 交换 | `(B×H, NT)` |
| 拆分 B/H | `(NT, B, H)` |

当 `B×H > 255` 且 `NT <= 255` 时交换前两维；当 `B×H > 255` 且 `NT > 255` 时，将 B 和 H 拆到不同维度。Kernel 按相同的 `GRID_MODE` 解码 `tl.program_id()`，保证索引映射不变。

K2 对 `N×H > 255` 的情况同样交换 value tile 和 sequence/head 两个 grid 维度。

## 5. 最终保留的优化方案

### 5.1 融合时间块固定为 BT=64

S60 融合推理路径统一使用：

```text
BT = 64
```

与更小的 BT 相比，BT64 可以减少 chunk 数量和跨 chunk 状态传播次数，同时增大 chunk 内矩阵形状，更适合 S60 的矩阵计算路径。

### 5.2 扩展 BT64 局部矩阵求解

`Akk` 是 `64×64` 严格下三角矩阵。由于 `L^64=0`，有限 Neumann 展开需要覆盖到 `L^63`。

当前 inverse 计算包含：

```python
b_L2 = tl.dot(b_L, b_L)
b_L4 = tl.dot(b_L2, b_L2)
b_L8 = tl.dot(b_L4, b_L4)
b_L16 = tl.dot(b_L8, b_L8)
b_L32 = tl.dot(b_L16, b_L16)
```

并依次将对应项累加到 inverse 中。加入 `L16` 和 `L32` 后，BT64 的严格下三角矩阵求解才是完整的；只保留 BT16 所需的 `L2/L4/L8` 会遗漏高阶项。

### 5.3 K1 使用固定大 tile

K1 当前使用：

```text
BK = 256
BV = 256
num_warps = 1
num_stages = 2
```

大 tile 减少 K/V 维度循环次数，并提高 `Aqk`、`Akk`、inverse、`u_wy` 和 `w_wy` 阶段的单次矩阵计算规模。

### 5.4 K2 按 K 选择 BK

K2 根据实际 K 维度选择 state tile：

```python
if K <= 64:
    state_bk = 64
elif K <= 128:
    state_bk = 128
else:
    state_bk = 256
```

其余参数固定为：

```text
BV = 256
num_warps = 1
num_stages = 2
```

这样小 K 不会被过大的 tile 增加资源占用，K=128/256 时则可减少 state 和 output 阶段的 K 分块数量。

### 5.5 低精度矩阵输入与 FP32 累加

QK、KK 以及状态相关 dot 的输入保持 FP16/BF16，使 S60 使用低精度矩阵计算路径；关键 score 和 state update 使用 FP32 累加，避免为了速度降低递推状态精度。

## 6. 最终 baseline 使用什么实现

当前生产入口的选择逻辑为：

```text
TLE 扩展可用 + BT64
        ↓
S60 融合 K1/K2 路径

否则
        ↓
Native Triton multi-kernel fallback
```

BaselineBenchmark 中的 Torch GCU 实现是独立、inference-only 的对比实现，不注册为生产 GDN2 后端。它同样采用 K1 并行准备工作区、K2 只保留跨 chunk recurrence 串行的结构，因此性能表中的 Torch GCU 不是逐 token 的慢速参考实现。

## 7. 最终性能

以下表格保留附件中的原始测试结果。`S60/Torch` 和 `S60/TopsCC` 均以对方延迟除以 S60 延迟计算：

```text
S60/Torch = Torch GCU latency / S60 latency
S60/TopsCC = TopsCC latency / S60 latency
```

数值大于 1 表示 S60 Triton 更快。

附件原表只给出一个 D 列；结合当前 benchmark shape，本文将不同的 K/V 维度分别列出。

### 7.1 FP16

| B | T | H | K | V | H100（ms） | S60（ms） | Torch GCU（ms） | TopsCC（ms） | S60/Torch | S60/TopsCC |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 512 | 8 | 64 | 64 | 0.214 | 0.575070 | 3.321886 | 3.864765 | 5.776x | 6.721x |
| 4 | 1024 | 8 | 64 | 64 | 0.254 | 1.944825 | 5.151272 | 14.518738 | 2.649x | 7.465x |
| 1 | 2048 | 8 | 64 | 64 | 0.208 | 1.297290 | 8.132935 | 10.855675 | 6.269x | 8.368x |
| 1 | 4096 | 16 | 64 | 64 | 0.293 | 3.705375 | 14.890671 | 28.807878 | 4.019x | 7.775x |
| 1 | 8192 | 96 | 128 | 128 | 2.001 | 40.786049 | 125.489473 | 470.413685 | 3.077x | 11.534x |
| 2 | 2048 | 16 | 256 | 512 | 0.619 | 5.900190 | 35.141468 | 116.223335 | 5.956x | 19.698x |
| 2 | 16384 | 16 | 128 | 128 | 1.901 | 30.769304 | 98.295689 | 332.566261 | 3.195x | 10.808x |
| 4 | 1024 | 8 | 256 | 512 | 0.328 | 3.070665 | 17.302990 | 58.317661 | 5.635x | 18.992x |
| 4 | 2048 | 16 | 128 | 128 | 0.390 | 7.144290 | 25.764227 | 83.568811 | 3.606x | 11.697x |
| 4 | 4096 | 64 | 128 | 128 | 2.400 | 54.614521 | 163.052559 | 636.720657 | 2.986x | 11.658x |
| 8 | 1024 | 8 | 64 | 64 | 0.244 | 3.463365 | 8.505344 | 25.430441 | 2.456x | 7.343x |
| 8 | 2048 | 32 | 256 | 256 | 2.891 | 32.216476 | 169.025660 | 580.204964 | 5.247x | 18.010x |

### 7.2 BF16

| B | T | H | K | V | H100（ms） | S60（ms） | Torch GCU（ms） | TopsCC（ms） | S60/Torch | S60/TopsCC |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 512 | 8 | 64 | 64 | 0.306 | 0.577875 | 3.320217 | 3.846407 | 5.746x | 6.656x |
| 4 | 1024 | 8 | 64 | 64 | 0.216 | 1.953900 | 5.141020 | 14.482021 | 2.631x | 7.412x |
| 1 | 2048 | 8 | 64 | 64 | 0.223 | 1.303440 | 8.174181 | 10.810375 | 6.271x | 8.294x |
| 1 | 4096 | 16 | 64 | 64 | 0.299 | 3.756780 | 14.649153 | 28.640509 | 3.899x | 7.624x |
| 1 | 8192 | 96 | 128 | 128 | 2.009 | 41.044289 | 126.404762 | 469.621420 | 3.080x | 11.442x |
| 2 | 2048 | 16 | 256 | 512 | 0.620 | 5.880105 | 35.003901 | 116.271019 | 5.953x | 19.774x |
| 2 | 16384 | 16 | 128 | 128 | 1.888 | 30.748739 | 98.164797 | 332.270384 | 3.192x | 10.806x |
| 4 | 1024 | 8 | 256 | 512 | 0.328 | 3.049845 | 17.214298 | 58.340549 | 5.644x | 19.129x |
| 4 | 2048 | 16 | 128 | 128 | 0.390 | 7.109535 | 25.830746 | 83.508730 | 3.633x | 11.746x |
| 4 | 4096 | 64 | 128 | 128 | 2.409 | 55.045155 | 164.199352 | 635.727167 | 2.983x | 11.549x |
| 8 | 1024 | 8 | 64 | 64 | 0.213 | 3.478200 | 8.500338 | 25.274754 | 2.444x | 7.267x |
| 8 | 2048 | 32 | 256 | 256 | 2.926 | 32.404228 | 169.959784 | 576.631784 | 5.245x | 17.795x |

从完整数据可以得到：

- 24 组 FP16/BF16 shape 中，S60 Triton 均快于 Torch GCU 和 TopsCC；
- 相对 Torch GCU 的 speedup 为 2.444x～6.271x；
- 相对 TopsCC 的 speedup 为 6.656x～19.774x；
- K/V 较大的 shape 上，固定大 tile 和 K2 自适应 BK 的收益更明显；
- S60 与 H100 仍存在明显的跨硬件性能差距。

## 8. 正确性验证

### 8.1 融合路径对 Native Triton

`FlagAttention/tests/flag_attn/test_chunk_gdn2.py` 使用 12 组正式 shape，并覆盖：

- FP16 和 BF16；
- 融合路径和 native Triton 路径；
- `K=64/128/256`；
- `V=64/128/256/512`；
- 不同 B、T 和 H；
- output 和 final state。

误差比门限为：

```text
error_ratio < 0.01
```

### 8.2 Torch GCU 对独立 FP32 recurrence

`BaselineBenchmark/backends/_enflame/test/test_chunk_gdn2_torch_gcu.py` 覆盖：

- FP16 和 BF16；
- fixed `[64]` 和 varlen `[29, 35, 17]`；
- 有或无 initial state；
- `state_v_first=True/False`；
- output 和 final state。

Torch GCU 对独立 FP32 recurrence 的 output/state error ratio 均要求小于 0.01。两种 state layout 的等价性测试要求 error ratio 小于 `1e-6`。

### 8.3 Torch GCU 与融合路径对比

对比 benchmark 在计时前先计算 output 和 final state 的误差，并要求：

```text
output_error < 0.01
state_error < 0.01
```

## 9. 已替换或未采用的方案

| 方案 | 问题 | 当前处理 |
| --- | --- | --- |
| NVIDIA PTX softplus | 只能在 NVIDIA PTX 后端使用 | 替换为可移植 Triton softplus |
| 大规模 autotune | 任一候选编译失败可能中断整组调优 | 使用已验证固定配置 |
| 融合 BT16 | chunk 数多，矩阵规模偏小 | S60 融合路径固定为 BT64 |
| 仅计算到 `L8` 的 inverse | 不能完整覆盖 BT64 的严格下三角矩阵 | 增加 `L16`、`L32` |
| K2 固定单一 BK | 不能同时兼顾小 K 资源占用和大 K 分块次数 | 按 K 选择 64/128/256 |
| 超限二维 grid | `grid.y` 或 `grid.z` 可能超过 255 | 交换 grid 或拆分 B/H |

## 10. 当前不足与后续优化空间

### 10.1 S60 与 H100 的差距

当前 S60 已经能够通过大 tile 和 BT64 使用低精度矩阵计算，但相同 shape 与 H100 仍有明显差距。后续需要从 GCU IR/LLIR 检查：

- K1 的 `64×256` 相关 dot 是否稳定映射到高效矩阵指令；
- K2 state tile 是否产生额外 transpose 或本地存储物化；
- `tl.load` 与 `tl.dot` 是否存在较长 DMA wait；
- K1 的 gate cumsum、inverse 和矩阵计算是否争用过多片上资源。

### 10.2 真正的异步流水

当前源码中的融合 Kernel 使用普通 Triton load/dot。若 Enflame 后端能够提供并稳定 lowering 异步 load、双缓冲和 warp specialization，可以进一步重叠 K/V 工作区加载、矩阵计算和状态写回。

### 10.3 K1 `num_stages` 复测

附件记录为 `num_stages=1`，当前源码为 2。需要在同一软件版本和独立进程条件下对 1/2 做完整 24-shape A/B 测试，再确认最终默认值。不能直接用附件旧数据证明当前 `num_stages=2` 更快。

### 10.4 Varlen 与超大 grid

三种 grid mode 已覆盖 `B×H` 和 `NT` 超过 255 的情况，后续仍需补充较大 packed varlen 的边界测试，确认 K1/K2 使用相同 chunk 编号和 sequence offset。

## 11. 测试与复现

### 11.1 完整 12 shape benchmark

在 FlagAttention 仓库中执行：

```bash
cd /workspace/FlagAttention
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

python -m pytest -q -s benchmark/test_chunk_gdn2.py \
  --level core \
  --warmup 25 \
  --iter 100
```

该 benchmark 默认覆盖 12 组 shape 和 FP16/BF16，并对比 public 融合路径与 native Triton baseline。

如需强制 public 入口使用 native Triton：

```bash
FLAG_ATTN_GDN2_FORCE_NATIVE=1 \
python -m pytest -q -s benchmark/test_chunk_gdn2.py \
  --level core \
  --warmup 25 \
  --iter 100
```

### 11.2 Torch GCU 与融合路径对比

在 BaselineBenchmark 仓库中执行：

```bash
cd /workspace/BaselineBenchmark
export PYTHONPATH="$PWD:$PWD/../FlagAttention/src:${PYTHONPATH:-}"

GDN2_BENCH_WARMUP=25 \
GDN2_BENCH_ITERS=100 \
python -m pytest -q -s \
  backends/_enflame/benchmarks/test_chunk_gdn2_torch_gcu.py
```

该对比脚本默认测试以下三组 shape，并同时覆盖 FP16/BF16：

```text
(1, 512, 8, 64, 64)
(1, 4096, 16, 64, 64)
(1, 8192, 96, 128, 128)
```

快速检查：

```bash
GDN2_BENCH_QUICK=1 \
GDN2_BENCH_WARMUP=2 \
GDN2_BENCH_ITERS=10 \
python -m pytest -q -s \
  backends/_enflame/benchmarks/test_chunk_gdn2_torch_gcu.py
```

### 11.3 正确性测试

```bash
cd /workspace/FlagAttention
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
python -m pytest -q tests/flag_attn/test_chunk_gdn2.py
```

```bash
cd /workspace/BaselineBenchmark
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python -m pytest -q backends/_enflame/test/test_chunk_gdn2_torch_gcu.py
```

## 12. 后续优化验收标准

后续修改只有同时满足以下条件才应替换当前实现：

1. 12 组 shape 的 FP16/BF16 均完成复测；
2. output 和 final state 的 error ratio 均小于 0.01；
3. fixed、varlen、initial state 和两种 state layout 均保持正确；
4. 不能删除 BT64 inverse 的 `L16/L32` 阶段；
5. 多轮测试中位数稳定，不依据单次最快结果；
6. 小 K 和大 K 都不能因统一 tile 而出现明显回退；
7. 不通过降低 FP32 state 累加精度换取速度；
8. 融合路径修改后仍需保留 native Triton fallback。
