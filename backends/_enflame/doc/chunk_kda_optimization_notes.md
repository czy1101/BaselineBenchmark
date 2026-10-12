# Enflame Chunk KDA 优化记录

更新时间：2026-08-28

## 1. 范围与环境

本文记录 Flash KDA（Kimi Delta Attention，下文简称 KDA）forward 在 Enflame S60 上的适配和优化过程，重点说明 S60 兼容路径、K1/K2 两阶段优化、最终性能以及当前限制。

当前实现涉及以下文件：

- S60 Triton/TLE 实现：`FlagAttention/src/flag_attn/runtime/backend/_enflame/kda/chunk_kda.py`
- S60 索引适配：`FlagAttention/src/flag_attn/runtime/backend/_enflame/kda/index.py`
- Triton 正确性测试：`FlagAttention/tests/flag_attn/test_chunk_kda.py`
- Triton benchmark：`FlagAttention/benchmark/test_chunk_kda.py`
- Torch GCU 对比实现：`BaselineBenchmark/backends/_enflame/ops/kda/forward.py`
- Torch GCU 输入验证：`BaselineBenchmark/backends/_enflame/ops/kda/validation.py`
- Torch/Triton 对比 benchmark：`BaselineBenchmark/backends/_enflame/benchmarks/test_chunk_kda_torch_gcu.py`
- Torch GCU 正确性测试：`BaselineBenchmark/backends/_enflame/test/test_chunk_kda_torch_gcu.py`

主要性能测试配置如下：

| 项目 | 配置 |
| --- | --- |
| S60 设备 | Enflame S60 / GCU300 |
| 输入 dtype | BF16 |
| 输入布局 | `[B, T, H, D]` |
| Head 数 | `H=96` |
| Key/Value 维度 | `K=V=128` |
| Chunk 大小 | `BT=32` |
| 输出 final state | 开启 |
| Fixed workload | `[8192]` |
| Irregular varlen | `[1300, 547, 2048, 963, 271, 3063]` |
| Regular varlen | `[1024] × 8` |

## 2. 数学定义与正确性红线

KDA forward 首先处理 Q/K、gate 和 beta：

```text
q_t = normalize(q_t)
k_t = normalize(k_t)
g_t = lower_bound * sigmoid(exp(A_log) * (g_t + dt_bias))
beta_t = sigmoid(beta_t)
```

随后按照 token 递推状态：

```text
r_t = v_t - k_t @ S_(t-1)
S_t = exp(g_t) * S_(t-1) + (beta_t * k_t)^T @ r_t
o_t = scale * q_t @ S_t
```

实际实现将 token recurrence 转换为 chunk 算法，以减少严格串行的 token 循环，但不能改变上述状态更新语义。

优化必须保持以下行为：

- Q/K L2 normalize、gate 变换和 beta sigmoid 的语义不变；
- fixed-length 和 packed varlen 均可用；
- 支持有或无 `initial_state`；
- `state_v_first` 对 final state 布局的约定不变；
- `output_final_state=False` 时不额外返回状态；
- 尾 chunk 不得越界，无效 token 不参与计算；
- output 和 final state 均需通过独立 FP32 recurrence 参考实现的精度门限。

## 3. 当前实现结构

当前 KDA 使用 K1/K2 两阶段 chunk 算法。

### 3.1 K1：chunk 内并行预处理

K1 对不同 chunk 并行执行，主要生成：

- gate 作用后的 `w`、`qg`、`kg`；
- chunk 内 QK 项 `Aqk`；
- chunk 内 KK 严格下三角项 `Akk`；
- `(I + L)^-1` 对应的三角 inverse；
- 每个 chunk 末尾的 state decay。

这一阶段不存在跨 chunk 状态依赖，因此可以充分并行。

### 3.2 K2：跨 chunk 状态递推与输出

K2 按 chunk 顺序维护 state，核心计算包括：

```text
kh = w @ state
v_new = Akk_inverse @ (beta_v - kh)
qh = qg @ state
output = scale * qh + Aqk @ v_new
state = state * state_decay + kg^T @ v_new
```

不同 chunk 的 state 存在前后依赖，因此 chunk 循环必须保持顺序；同一 chunk 内的矩阵运算仍由 `tl.dot` 执行。

### 3.3 三条后端路径

公共入口 `chunk_kda` 提供三条语义一致的 forward 路径：

| 路径 | 作用 |
| --- | --- |
| `strict_tle` | 面向 BF16、K=V=128 等生产 shape 的优化路径 |
| `tle` | 支持更宽输入集合的 Generic TLE 路径 |
| `triton` | TLE 不可用或输入不满足 TLE 约束时的普通 Triton fallback |

默认 `auto` 按 `strict_tle → tle → triton` 的顺序选择。可通过以下环境变量强制指定：

```text
FLAG_ATTN_CHUNK_KDA_BACKEND=strict_tle
FLAG_ATTN_CHUNK_KDA_BACKEND=tle
FLAG_ATTN_CHUNK_KDA_BACKEND=triton
```

需要注意：S60 上的 `strict_tle` 名称不表示 K2 仍使用 H100 的 TLE pipe。该路径的 K1 保留可用的 TLE load，K2 已替换为 S60 可编译的顺序 Triton Kernel。

## 4. S60 功能适配

以下修改的主要目标是解决 S60/GCU300 的编译、launch 和索引兼容问题，属于适配工作。

### 4.1 删除 `maxnreg`

H100/NVIDIA 原实现的 autotune config 使用 `maxnreg` 限制每个线程的寄存器数。Enflame `GCUOptions` 没有该字段，Kernel 在生成前即会报错，因此从 KDA 的 S60 autotune 配置中删除该参数。

### 4.2 移除 K1 中不兼容的 TLE shared/local 对象

原始路径依赖：

```text
tle.gpu.alloc
tle.gpu.local_ptr
```

当前 Enflame lowering 无法正确处理 K1 中这些 shared/local-memory 方言对象。S60 K1 保留能够正常 lowering 的 TLE load，同时移除不兼容的 shared/local 对象使用。

### 4.3 用顺序 K2 替代不可用的 TLE pipe

H100 strict K2 使用：

```text
tle.pipe
tle.gpu.warp_specialize
tle.gpu.alloc
tle.gpu.local_ptr
```

S60 编译时会出现：

```text
Dialect `tle' not found for custom op 'tle.pipe.create'
RuntimeError: Pipeline run failed: Failed to parse input MLIR
```

因此 S60 分别使用：

```text
_strict_kda_fwd_state_output_s60_kernel
_kda_fwd_state_output_s60_kernel
```

这两个 Kernel 保留相同的 K2 数学过程，以普通顺序 Triton 循环代替当前 Enflame MLIR 无法解析的 pipe 和 warp specialization。

### 4.4 K2 launch grid 一维展平

原二维 launch 会把 `N × HV` 放入 `grid.y`。较大的 varlen batch 曾出现：

```text
grid.y required: 576
hardware limit: 255
```

S60 strict 和 generic K2 均改为一维扁平索引：

```python
i_pid = tl.program_id(0)
n_v_tiles = tl.cdiv(V, BV)
i_v = i_pid % n_v_tiles
i_nh = i_pid // n_v_tiles
i_n = i_nh // HV
i_h = i_nh % HV
```

对应 launcher 为：

```python
grid = (triton.cdiv(V, BV) * N * HV,)
```

该修改绕开 `grid.y <= 255` 的限制，同时完整覆盖 value tile、sequence 和 head。

### 4.5 Varlen 元数据统一为 INT32

GCU300 对普通 INT64 索引支持受限，并会提示：

```text
GCU not support Long use Int replace
```

最终修改包括：

- `cu_seqlens` 明确要求 `torch.int32`；
- strict/generic K1、K2 的变长索引转为 `tl.int32`；
- `prepare_chunk_indices()` 和 `prepare_chunk_offsets()` 保持输入 dtype；
- 生成 `arange`、sequence id 和 chunk offset 时沿用 `cu_seqlens.dtype`；
- 输入验证拒绝 S60 varlen 路径上的 INT64 元数据。

该修改避免 GCU300 隐式替换 INT64、潜在溢出和编译失败。

## 5. K1 最终保留的优化

### 5.1 合并两个小 QK dot

BT16 原实现分别计算 `Aqk` 和 `Akk`：

```text
[16, 128] @ [128, 16]
[16, 128] @ [128, 16]
```

S60 对 `N=16` 的小矩阵利用率较低。BT16 路径使用 `tl.join`、`tl.reshape` 和 `tl.split` 打包两个右操作数，合并成一次：

```text
[16, 128] @ [128, 32]
```

代表性消融结果：

| 项目 | 优化前 | 优化后 |
| --- | ---: | ---: |
| QK 部分 | 约 2085 ms | 约 36 ms |
| K1 整体 | 约 2955 ms | 约 938 ms |

### 5.2 成对执行 BT16 inverse

原 inverse 在每个 program 内多次执行细粒度 `16×16` FP16 dot，S60 会生成大量小矩阵原语。

最终 BT16 路径先保存未完成的 inverse 数据，再启动独立 inverse Kernel，将两个独立 `16×16` 矩阵组成一个 `32×32` 分块对角矩阵，一次处理两个 inverse。

代表性结果：

| 项目 | 优化前 | 优化后 |
| --- | ---: | ---: |
| inverse 部分 | 约 862 ms | 约 100 ms |
| K1 整体 | 约 938 ms | 约 177 ms |

### 5.3 BT32 数值稳定的三角分块 QK

直接将 BT 从 16 改为 32 时，累计 gate 的指数幅度可能达到：

```text
32 × 5 × 1.4427 ≈ 231
```

直接计算 `exp2(-b_gc)` 会超过 FP32/BF16 指数范围，导致 `Aqk/Akk` 在 inverse 前出现 Inf 或 NaN。

最终 BT32 路径采用：

1. 以第 16 行的累计 gate 为指数平移中心；
2. 将 `32×32` 矩阵划分为上下两个 16-row 区域；
3. 只计算最终严格下三角矩阵需要的三个分块；
4. 跳过指数可能溢出且最终会被 mask 掉的分块；
5. inverse 增加 `L16` 阶段，完整覆盖 `32×32` 严格下三角矩阵。

BT32 K1 的代表性结果为约 `2954 ms → 64 ms`，约 46 倍提升。

## 6. K2 最终保留的优化

### 6.1 消融定位 state update dot

BT16 K2 消融结果如下：

| 路径 | 延迟（ms） |
| --- | ---: |
| 完整 K2 | 11184 |
| 去掉 state update dot | 2030 |
| 差值 | 9154 |

主要瓶颈为：

```text
kg^T @ v_new
```

该矩阵计算约占原 K2 时间的 82%。

### 6.2 BT16 改为 BT32

BT16 状态更新矩阵为：

```text
[128, 16] @ [16, 128]
```

S60 对 reduction dimension 为 16 的矩阵微内核效率很低。BT32 后变为：

```text
[128, 32] @ [32, 128]
```

同时 T=8192 时的 chunk 数从 512 降为 256，因此同时获得：

- 更高效的 K=32 矩阵微内核；
- chunk 循环次数减半；
- state update、`Aqk`、`Akk`、`w` 和 `qg` 等每 chunk 固定开销减半。

代表性结果为：

| 配置 | K2 延迟（ms） |
| --- | ---: |
| BT16 | 11321 |
| BT32 | 124 |

K2 提升约 91 倍。该结果也说明 KDA 在 S60 上的关键不是单纯增加并行 program，而是将矩阵 shape 调整到 S60 更高效的微内核范围。

## 7. 最终性能

以下数据来自附件中的统一 workload 对比。前三列为 H100 结果，后三列为 S60 对应实现；跨硬件数据用于展示实现差距，不等同于设备峰值性能的直接比较。

`Strict TLE S60/Torch` 定义为：

```text
Torch GCU latency / Strict TLE S60 latency
```

数值大于 1 表示 Strict TLE S60 更快，数值小于 1 表示 Torch GCU 更快。

| Workload | Strict TLE H100（ms） | Generic TLE H100（ms） | Triton fused H100（ms） | Strict TLE S60（ms） | Generic TLE S60（ms） | Triton fused S60（ms） | Torch GCU（ms） | TopsCC（ms） | Strict S60/Torch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed length：8192 | 1.103 | 1.323 | 2.004 | 188.742638 | 188.655615 | 189.146519 | 186.131954 | 815.037124 | 0.986x |
| Irregular varlen：1300, 547, 2048, 963, 271, 3063 | 1.100 | 1.310 | 1.908 | 190.226555 | 189.988136 | 189.470053 | 214.277029 | — | 1.126x |
| Regular varlen：8 × 1024 | 1.074 | 1.273 | 1.831 | 188.311815 | 188.700676 | 188.234329 | 207.880020 | — | 1.104x |
| Average | 1.092 | 1.302 | 1.914 | 189.093669 | 189.114809 | 188.950300 | 202.763001 | 815.037124* | 1.072x |

`*` TopsCC 只提供 fixed-length 数据，因此表中的 815.037124 ms 不是三个 workload 的平均值。

从结果可以得到：

- S60 三条 Triton/TLE 路径整体非常接近，约 188～190 ms；
- Strict TLE S60 在 fixed-length 上比 Torch GCU 慢约 1.4%；
- Strict TLE S60 在两组 varlen 上分别比 Torch GCU 快约 12.6% 和 10.4%；
- 三组平均后，Strict TLE S60 相对 Torch GCU 为 1.072x；
- S60 路径相对 H100 仍存在非常大的跨硬件性能差距。

## 8. 正确性验证

### 8.1 Triton/TLE 对独立 recurrence

`FlagAttention/tests/flag_attn/test_chunk_kda.py` 使用独立 Torch FP32 recurrence 作为参考，并覆盖：

- dense 和 varlen；
- D64 和 D128；
- BF16 和 FP16；
- strict TLE compatible shape；
- GQA；
- 有或无 initial state；
- `state_v_first=True/False`；
- 返回或不返回 final state。

output 和 final state 的误差比门限均为：

```text
error_ratio < 0.005
```

### 8.2 Torch GCU 对独立 recurrence

`BaselineBenchmark/backends/_enflame/test/test_chunk_kda_torch_gcu.py` 覆盖：

- fixed `[64]`；
- varlen `[29, 35, 17]`；
- 有、无 initial state；
- BF16、D128、INT32 `cu_seqlens`；
- output 和 final state 误差比均小于 0.005。

### 8.3 Torch GCU 与 Strict TLE 对比

对比 benchmark 在计时前检查两条路径的结果均为有限值，并要求：

```text
output_error < 0.005
state_error < 0.005
```

## 9. H100 路径为何无法直接迁移到 S60

H100 strict K2 的高性能来自 TLE pipe、异步 load/store、shared/local buffer 和 warp specialization 的协同。它能够将数据搬运、矩阵计算和结果写回组织成流水线。

当前 S60/GCU300 Triton 后端的主要限制为：

1. 无法解析 K2 使用的 `tle.pipe` 方言；
2. 无法直接 lowering 原路径中的部分 `tle.gpu.alloc/local_ptr` 对象；
3. 缺少等价的 warp specialization 执行路径；
4. 普通顺序 Triton K2 必须显式等待 load、dot 和 state update；
5. 原 BT16 的小矩阵 shape 无法充分利用 S60 矩阵计算单元；
6. 跨 chunk 的真实数据依赖限制了 K2 的并行方式。

因此，本轮优化通过 BT32 和矩阵合并大幅改善了单次矩阵计算效率，但没有恢复 H100 TLE K2 的异步流水能力。这也是 S60 结果仍明显落后于 H100 的主要原因。

## 10. Torch GCU 对比路径说明

BaselineBenchmark 中的 Torch GCU KDA 是隔离的 inference-only 实验实现，不注册为生产 KDA 后端。它不是逐 token 的朴素参考，而是采用与 S60 Triton 相同的 K1/K2 两阶段 chunk 算法：

- K1 使用 Torch GCU tensor op 和 `torch.bmm` 并行生成 chunk 工作区；
- K2 只保留跨 chunk state recurrence 的串行循环；
- 默认 chunk size 为 32；
- 目前只支持 BF16、K=V=128 和受限的生产输入集合。

因此表中的 Torch GCU 是有意义的同算法端到端对比，而不是慢速 Python recurrence reference。真正的独立 FP32 recurrence 仅用于小 shape 正确性验证。

## 11. 当前不足与后续优化空间

### 11.1 S60 原生 TLE pipeline

如果 Enflame Triton 后端补齐 GCU300 上 `tle.pipe`、shared/local buffer 和 warp specialization 的 lowering，可以重新评估 H100 strict K2。目标是让当前顺序过程：

```text
load chunk n → dot → state update → store → load chunk n+1
```

变成：

```text
计算 chunk n，同时加载 chunk n+1，并写回 chunk n-1
```

### 11.2 K2 state update dot

`kg^T @ v_new` 已由 BT32 从低效的 K=16 变成 K=32，但它仍是 K2 的核心成本。后续应重点检查：

- `128×32 @ 32×128` 在 GCU IR 中选择的矩阵指令；
- BF16 输入和 FP32 累加是否发生额外中间物化；
- state 的加载、转换和写回是否可以与 dot 重叠；
- BV tile 是否存在更适合 S60 的选择。

### 11.3 Fixed-length 小幅回退

当前 Strict TLE S60 在 fixed 8192 上为 188.743 ms，Torch GCU 为 186.132 ms，仍慢约 1.4%。后续优化应单独检查 fixed 路径，不能只依据三组 workload 平均值替换实现。

### 11.4 Varlen 索引与调度开销

一维 grid 已解决硬件上限，但每个 program 仍需从扁平 id 恢复 sequence/head/value tile，并读取 INT32 offset。可以在不增加 grid.y 风险的前提下继续评估索引预计算和固定 shape 特化。

## 12. 测试与复现

### 12.1 Torch GCU 与 Strict TLE 性能对比

在 BaselineBenchmark 仓库中执行：

```bash
cd /workspace/BaselineBenchmark
export PYTHONPATH="$PWD:$PWD/../FlagAttention/src:${PYTHONPATH:-}"

KDA_BENCH_WARMUP=25 \
KDA_BENCH_ITERS=100 \
python -m pytest -q -s \
  backends/_enflame/benchmarks/test_chunk_kda_torch_gcu.py
```

该 benchmark 固定使用 `strict_tle`，覆盖 fixed 8192、irregular varlen 和 regular varlen 三组 workload，并同时打印 output/state error ratio。

快速检查可执行：

```bash
KDA_BENCH_QUICK=1 \
KDA_BENCH_WARMUP=2 \
KDA_BENCH_ITERS=10 \
python -m pytest -q -s \
  backends/_enflame/benchmarks/test_chunk_kda_torch_gcu.py
```

### 12.2 FlagAttention KDA benchmark

```bash
cd /workspace/FlagAttention
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

python -m pytest -q -s benchmark/test_chunk_kda.py
```

可分别设置 `FLAG_ATTN_CHUNK_KDA_BACKEND=strict_tle`、`tle` 或 `triton` 进行受控消融。强制后端时需要确保输入满足对应路径约束。

### 12.3 正确性测试

```bash
cd /workspace/FlagAttention
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
python -m pytest -q tests/flag_attn/test_chunk_kda.py
```

```bash
cd /workspace/BaselineBenchmark
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python -m pytest -q backends/_enflame/test/test_chunk_kda_torch_gcu.py
```

## 13. 后续优化验收标准

后续修改只有同时满足以下条件才应替换当前路径：

1. fixed、irregular varlen 和 regular varlen 三组 workload 均完成复测；
2. output 与 final state 的 error ratio 均小于 0.005；
3. 有无 initial state、尾 chunk 和 INT32 varlen 元数据均保持正确；
4. 多轮中位数稳定，不能依据单次最快结果；
5. fixed-length 不因提高 varlen 平均性能而出现不可接受的回退；
6. 不通过删除 FP32 state 累加或降低 inverse 数值稳定性换取速度；
7. 新增后端特化必须保留 generic TLE 和普通 Triton fallback。
