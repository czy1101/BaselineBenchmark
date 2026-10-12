# Enflame SageAttention 优化记录

更新时间：2026-08-28

## 1. 范围与环境

本文记录 SageAttention（下文简称 SGA）在 Enflame S60 上的适配和优化过程，目标算子为 Q/K INT8 per-block 量化的 Attention forward。

当前实现涉及以下文件：

- S60 Triton 实现：`FlagAttention/src/flag_attn/runtime/backend/_enflame/sage_attention/attn_qk_int8_per_block.py`
- S60 Torch 原生基线：`BaselineBenchmark/backends/_enflame/ops/sage_attention/attn_qk_int8_per_block_native.py`
- 基准测试：`BaselineBenchmark/backends/_enflame/benchmarks/sage_attention_native_benchmark.py`
- 正确性测试：`BaselineBenchmark/backends/_enflame/test/test_sage_attention_native.py`

主要测试环境如下：

| 项目 | 配置 |
| --- | --- |
| 设备 | Enflame S60 48 GB |
| Triton | 3.6.0 |
| Triton target | `gcu / dtu-enflame-tops--gcu300` |
| Triton warp size | 12 |
| Q/K dtype | INT8 |
| V dtype | FP16 或 BF16 |
| Attention layout | HND、NHD |
| 默认 block 配置 | `BLOCK_M=128`、`BLOCK_N=128` |
| 默认 launch 配置 | `num_warps=1`、`num_stages=1` |
| 性能测试配置 | `warmup=25`、`rep=100` |

文中 FP16/BF16 表示 V 和输出的数据类型；Q、K 始终为 INT8，并使用 per-block scale 完成反量化。

## 2. 数学定义与正确性红线

Q/K 的 per-block 反量化形式为：

```text
Q_fp = Q_int8 * Q_scale
K_fp = K_int8 * K_scale
```

随后计算：

```text
S = Q_fp @ K_fp^T * sm_scale
P = softmax(S + mask)
O = P @ V
```

Online Softmax 使用 base-2 指数路径。最终 LSE 为：

```text
LSE = log2(l_i) + m_i
```

其中 `m_i` 是 Online Softmax 维护的行最大值，`l_i` 是经过最大值修正后的指数累加值。

优化必须保持以下行为不变：

- HND、NHD 两种布局均可用；
- MHA、GQA 和 MQA 使用一致的 head 映射语义；
- BOOL mask 和浮点加法 mask 的语义不变；
- Query/KV 长度不整除 block 时不得越界，并且无效 token 不得参与计算；
- 输出和 LSE 与基线保持一致；
- 不以降低 FP32 Online Softmax 累加精度换取性能。

## 3. 当前 Triton 实现结构

最终实现由 K 反量化 Kernel 和 Attention Kernel 两部分组成。

### 3.1 K 预反量化

`_dequant_k_fp16` 将 INT8 K 和 per-block scale 转换成 FP16 临时张量。这样可以避免每个 Query block 在 Attention 主循环中重复执行 K 的 INT8 到 FP16 转换。

对于 K 复用次数较低的 shape，预反量化带来的额外全局内存写入不一定能够回收，因此实现按照 K 的复用程度选择预反量化或循环内反量化路径。

### 3.2 Attention 主 Kernel

`_attn_fwd` 在一个 Kernel 中完成：

1. 加载并反量化 Q；
2. 分块加载 K；
3. 使用 FP16 `tl.dot` 计算 QK；
4. 执行 base-2 Online Softmax；
5. 加载 V；
6. 使用 `tl.dot` 累加 PV；
7. 写回输出和可选 LSE。

QK 和 PV 的 `tl.dot` 均能够在 S60 Triton 编译产物中下降到 `gcu.matmul`。当前主要问题不是完全没有使用矩阵计算指令，而是数据转换、加载等待、Softmax 归约和中间值物化会降低矩阵计算单元的持续占用率。

## 4. S60 功能适配

以下内容属于保证算子在 S60 上可编译、可运行并保持接口语义的适配工作。

### 4.1 INT8 转 FP16 后执行 `tl.dot`

S60 当前 Triton 后端的 INT8 `tl.dot` 路径没有表现出可用的高性能矩阵计算 lowering。当前实现先将 Q/K 转换并缩放为 FP16，再使用已验证可用的 FP16 矩阵计算路径。

这属于 S60 的兼容性路径，不代表 INT8 算法本身不适用于 S60；限制来自当前 Triton INT8 dot 的编译质量和硬件指令映射。

### 4.2 一维 grid 展平

将 batch、head 和 Query block 映射到一维 program id，避免依赖 S60 后端对高维 grid 的额外处理：

```text
program_id = (batch, query_head, query_block)
```

### 4.3 `maxnreg` 仅在 CUDA 传递

S60 GCU 后端不识别 CUDA 专用的 `maxnreg` launch 参数。实现只在 CUDA 路径传递该参数，避免 S60 编译或启动时出现 `KeyError`。

### 4.4 HND/NHD stride 适配

Kernel 通过输入张量的 stride 访问数据，不额外执行整张量转置，因此同一套实现可处理 HND 和 NHD 布局。

### 4.5 GQA/MQA head 映射

Query Head 到 KV Head 的映射为：

```text
kv_head = off_h // num_kv_groups
```

多个 Query Head 可以共享同一 K/V Head，无需分别维护 MHA、GQA 和 MQA Kernel。

### 4.6 BOOL Mask 与加法 Mask

- BOOL mask：对无效位置加入足够大的负值，使其 Softmax 概率趋近于 0；
- 浮点加法 mask：直接将 mask 值叠加到 QK score。

### 4.7 Partial Block 边界处理

Q/K/V 加载、Attention 计算和结果写回均包含有效位置判断，从而支持 Query 或 KV 长度不能整除 `BLOCK_M`、`BLOCK_N` 的尾块。

### 4.8 Base-2 LSE

保留 SageAttention 原有的 `exp2` Online Softmax 和 base-2 LSE 约定，确保 S60 实现与上层接口兼容。

## 5. 最终保留的优化方案

### 5.1 Q/K scale 前移

将 Q/K scale 应用于 dot 输入：

```text
Q_fp16 = Q_int8.to(fp16) * Q_scale
K_fp16 = K_int8.to(fp16) * K_scale
QK = tl.dot(Q_fp16, K_fp16)
```

这样可以稳定使用 S60 当前更成熟的 FP16 矩阵计算路径，并避免先执行 INT8 dot、再对 INT32 score 进行缩放的低效路径。

### 5.2 `BLOCK_N=128`

将 KV tile 从 64 增加到 128，减少主循环次数、Softmax 状态更新次数和循环控制开销。`BLOCK_M=64` 的实验出现明显回退，因此基础 Query tile 保持 128。

### 5.3 `num_warps=1`、`num_stages=1`

S60 实测中，提高 warp 数量没有带来类似 H100 上的矩阵计算并行收益，反而增加调度、同步和片上资源开销。多 stage 同样没有形成稳定的加载与计算重叠，并提高了资源占用。因此最终采用：

```text
num_warps = 1
num_stages = 1
```

### 5.4 K 预反量化与 D128 tile-major 布局

当同一 K tile 会被多个 Query block 重复使用时，先将 K 反量化为 FP16 临时张量，再由 Attention Kernel 读取。

D128 使用与后续矩阵计算访问顺序更匹配的 tile-major 临时布局，降低主循环内的数据重排成本。D64 则根据 K 的复用次数选择是否启用预反量化，避免短序列上额外 Kernel 和临时张量成本超过收益。

在 D64、B=1、H=32、T=8192 的单独实验中：

| 路径 | 延迟（ms） | 性能（TFLOPS） |
| --- | ---: | ---: |
| 循环内反量化 | 37.7115 | 14.58 |
| K 预反量化 | 32.7350 | 16.79 |

该 shape 上预反量化延迟降低约 13.2%，因此保留基于复用次数的选择逻辑。

### 5.5 PV 融合累加

PV 阶段直接使用已有累加器：

```python
acc = tl.dot(p, v, acc)
```

同时将 Online Softmax 状态更新安排在 PV 之后，减少额外中间矩阵和累加器赋值。

### 5.6 D64 双 Query block

对于 D64、无 mask 且边界完整的 dense 路径，将两个 128-row Query block 合并，使有效 `BLOCK_M` 达到 256。这样可以让同一 K/V tile 服务更多 Query row，提高 K/V 复用并减少 program 数量。

带 mask 或存在不规则边界时仍使用通用的 128-row 路径，避免改变功能语义。

### 5.7 V 预取

在计算当前 QK/Softmax 阶段提前发起 V 的加载，使部分 V 搬运和矩阵计算重叠。

- D128 dense/even 路径默认启用；
- D64 只在双 Query block 且预反量化收益足够稳定时启用；
- D64、T=8192 的无条件 V 预取出现双峰性能，因此最终选择逻辑会在第一档 K 复用阈值处关闭该路径。

### 5.8 D128 K-next 软件流水

D128 dense/even 路径在处理当前 K tile 时提前加载下一 K tile，形成软件级流水：

```text
计算 tile n 的 QK/PV，同时准备 tile n+1 的 K
```

该优化仅保留在实测稳定获益的 D128 路径。D64 K-next 会增加资源压力并造成回退，因此不启用。

### 5.9 删除无效的 K scale 指针更新

使用预反量化 K 时，Attention 主循环不再需要逐 tile 更新 `K_scale_ptr`。删除该冗余循环依赖获得约 0.23% 的小幅收益，并简化了编译器的数据依赖。

## 6. 最终性能

### 6.1 Triton 优化前后

以下为本轮优化前后的代表性 FP16 数据。D64 主要收益来自双 Query block、选择性 V 预取和 K 预反量化；D128 主要收益来自 V 预取、K-next 软件流水和 tile-major K 预反量化。

| B | SeqLen | H | HeadDim | 优化前（ms） | 最终（ms） | 延迟变化 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1024 | 32 | 64 | 0.7520 | 0.7191 | -4.38% |
| 4 | 1024 | 32 | 64 | 2.7080 | 2.6010 | -3.95% |
| 1 | 4096 | 32 | 64 | 10.3433 | 9.7559 | -5.68% |
| 1 | 8192 | 32 | 64 | 36.2068 | 35.1468 | -2.93% |
| 1 | 16384 | 32 | 64 | 138.0931 | 126.8277 | -8.16% |
| 1 | 1024 | 32 | 128 | 1.0668 | 0.9835 | -7.81% |
| 4 | 1024 | 32 | 128 | 4.1271 | 3.7882 | -8.21% |
| 1 | 4096 | 32 | 128 | 15.4214 | 13.7298 | -10.97% |
| 1 | 8192 | 32 | 128 | 51.0917 | 41.3840 | -19.00% |
| 1 | 16384 | 32 | 128 | 198.5047 | 160.6035 | -19.09% |

### 6.2 FP16 完整 shape

`S60/Torch` 定义为 `Torch latency / S60 Triton latency`。数值小于 1 表示当前 S60 Triton 实现仍慢于 Enflame Torch 原生实现。

| B | SeqLen | H | HeadDim | H100 原始（ms） | H100 优化（ms） | S60 Triton（ms） | Enflame Torch（ms） | S60/Torch |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1024 | 32 | 64 | 0.0313 | 0.0322 | 0.7191 | 0.4231 | 0.588x |
| 4 | 1024 | 32 | 64 | 0.1012 | 0.0955 | 2.6010 | 1.4165 | 0.545x |
| 1 | 4096 | 32 | 64 | 0.3535 | 0.3371 | 9.7559 | 4.1097 | 0.421x |
| 1 | 8192 | 32 | 64 | 1.3630 | 1.2868 | 35.1468 | 13.3257 | 0.379x |
| 1 | 16384 | 32 | 64 | 5.3013 | 4.9960 | 126.8277 | 48.7181 | 0.384x |
| 1 | 1024 | 32 | 128 | 0.0537 | 0.0516 | 0.9835 | 0.6089 | 0.619x |
| 4 | 1024 | 32 | 128 | 0.1922 | 0.1750 | 3.7882 | 2.1676 | 0.572x |
| 1 | 4096 | 32 | 128 | 0.6532 | 0.5898 | 13.7298 | 5.8835 | 0.429x |
| 1 | 8192 | 32 | 128 | 2.5321 | 2.2105 | 41.3840 | 18.7683 | 0.454x |
| 1 | 16384 | 32 | 128 | 9.9807 | 8.5175 | 160.6035 | 66.9602 | 0.417x |

### 6.3 BF16 完整 shape

| B | SeqLen | H | HeadDim | H100 原始（ms） | H100 优化（ms） | S60 Triton（ms） | Enflame Torch（ms） | S60/Torch |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1024 | 32 | 64 | 0.0316 | 0.0316 | 0.7192 | 0.4445 | 0.618x |
| 4 | 1024 | 32 | 64 | 0.1014 | 0.0955 | 2.6015 | 1.5038 | 0.578x |
| 1 | 4096 | 32 | 64 | 0.3508 | 0.3346 | 9.7576 | 4.2309 | 0.434x |
| 1 | 8192 | 32 | 64 | 1.3501 | 1.2763 | 35.1502 | 13.3861 | 0.381x |
| 1 | 16384 | 32 | 64 | 5.3009 | 4.9961 | 126.7973 | 49.1679 | 0.388x |
| 1 | 1024 | 32 | 128 | 0.0534 | 0.0518 | 0.9835 | 0.6524 | 0.663x |
| 4 | 1024 | 32 | 128 | 0.1910 | 0.1746 | 3.7882 | 2.3349 | 0.616x |
| 1 | 4096 | 32 | 128 | 0.6572 | 0.5938 | 13.7339 | 6.0453 | 0.440x |
| 1 | 8192 | 32 | 128 | 2.5318 | 2.2111 | 41.2853 | 19.1674 | 0.464x |
| 1 | 16384 | 32 | 128 | 9.9767 | 8.5200 | 160.5017 | 67.6786 | 0.422x |

## 7. 正确性验证

最终路径完成了以下验证：

- D64 默认 V 预取路径的输出和 LSE 与关闭预取的基线逐元素完全一致；
- D128 默认 K-next 路径的输出和 LSE 与关闭 K-next 的基线逐元素完全一致；
- 对上述 A/B 验证使用 `atol=0`、`rtol=0`；
- Torch 原生基线测试覆盖 HND/NHD、GQA、FP16/BF16、尾块和 fallback 行为；
- mask、LSE 或原生实现不支持的输入会回退到 Triton 通用路径。

性能优化只在满足 dense/even 等条件时启用，带 mask 和不规则尾块继续走经过边界保护的通用实现。

## 8. 已验证但未保留的方案

| 方案 | 代表性结果 | 结论 |
| --- | --- | --- |
| Enflame TLE 异步加载 | D64 约慢 0.97%，D128 约慢 1.25% | 当前形态没有形成有效重叠，回退 |
| D128 first-tile fast path | 约慢 1.26% | 回退 |
| `BLOCK_M=64` | 97.1204 ms，对照 50.9232 ms | 约慢 90.7%，回退 |
| FP16 PV accumulator | 51.5474 ms，对照约 51.0 ms | 约慢 1%，且有精度风险，回退 |
| 转置 score 后沿 axis 0 归约 | 69.5997 ms，对照约 51.0 ms | 约慢 36.5%，回退 |
| D64 K-next | 中位数 40.4262 ms，对照 36.2068 ms | 约慢 11.65%，关闭 |
| D64 无条件 V 预取 | T=8192 出现约 32～33 ms 和 48.4 ms 双峰 | 改为按 shape 选择 |
| 提高 `num_warps` | 没有获得稳定矩阵并行收益 | 保持 1 |
| 提高 `num_stages` | 没有获得稳定加载/计算重叠 | 保持 1 |

## 9. Torch 原生基线为什么更快

Torch 原生路径不是用 Python 运算逐步实现整个 Attention，而是先完成 Q/K 反量化与布局转换，再调用：

```python
torch.nn.functional.scaled_dot_product_attention(...)
```

在 torch-gcu 环境中，该调用可以进入 Enflame 已针对 S60 优化的 SDPA 实现。厂商实现能够使用更底层的矩阵指令、片上存储规划、DMA 搬运和流水调度。

当前 Triton 实现虽然已经将 QK/PV `tl.dot` 编译为 `gcu.matmul`，但仍存在以下差距：

1. 缺少高性能 INT8 `tl.dot` lowering，只能先转 FP16；
2. 普通 `tl.load`、`tl.dot` 和归约没有自动形成与厂商 SDPA 同等级别的异步流水；
3. INT8 到 FP16 的转换、scale、Softmax 和矩阵计算争用寄存器及片上资源；
4. 中间 score、转置或归约的物化会降低矩阵计算单元的连续工作时间；
5. Triton 层可控制的 `num_warps`、`num_stages` 和 tile 大小无法替代后端对 S60 指令选择及调度的深度优化。

因此，当前性能差距并不等同于 S60 矩阵计算能力不足，而是 Triton Enflame 后端尚未把该融合 Attention 稳定地编译成与厂商原生 SDPA 等价的底层执行计划。

## 10. 当前瓶颈与后续空间

### 10.1 高性能 INT8 矩阵路径

理想路径是直接使用 S60 高性能 INT8 矩阵计算，将 INT32 QK 结果与 Q/K scale 融合。该工作需要 Triton Enflame 后端提供并优化 INT8 `tl.dot` 到 S60 矩阵指令的 lowering，单靠算子 Python 文件难以完成。

### 10.2 真正的异步加载流水

当前 V 预取和 K-next 是源码级软件流水，收益受编译器调度影响。若后端提供可控的异步 DMA、等待组和片上双缓冲机制，可以进一步重叠 K/V 加载与 QK/PV 计算。

### 10.3 减少中间物化和归约成本

需要继续检查 GCU IR/LLIR 中的 score 存储、转置、Softmax 归约和 DMA wait，目标是在不降低 FP32 Softmax 精度的前提下减少本地存储占用和同步等待。

### 10.4 按 shape 选择策略

D64 T=8192 的双峰结果说明，单一策略不能覆盖所有 shape。后续选择规则应只基于稳定复现的数据，并同时检查：

- B、H、SeqLen、HeadDim；
- K tile 的 Query-block 复用次数；
- 预反量化临时张量成本；
- 预取引入的寄存器和本地存储压力。

## 11. 测试与复现

### 11.1 S60 Triton 全部 shape

在 FlagAttention 仓库中执行：

```bash
cd /workspace/FlagAttention
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

for dtype in float16 bfloat16; do
  for head_dim in 64 128; do
    python benchmark/sage_attention_benchmark.py \
      --batch-size 1 \
      --num-heads 32 \
      --head-dim "$head_dim" \
      --seq-lens 1024 4096 8192 16384 \
      --dtype "$dtype" \
      --warmup 25 \
      --rep 100

    python benchmark/sage_attention_benchmark.py \
      --batch-size 4 \
      --num-heads 32 \
      --head-dim "$head_dim" \
      --seq-lens 1024 \
      --dtype "$dtype" \
      --warmup 25 \
      --rep 100
  done
done
```

如果仓库中 benchmark 文件名为 `sage_attention_benchmark.py` 的兼容入口，则使用该文件；当前 BaselineBenchmark 对比入口见下一节。

### 11.2 Torch 原生与 Triton 对比

在 BaselineBenchmark 仓库中执行：

```bash
cd /workspace/BaselineBenchmark
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python backends/_enflame/benchmarks/sage_attention_native_benchmark.py \
  --backend torch-gcu \
  --compare-triton \
  --warmup 25 \
  --rep 100
```

该脚本的原生延迟包含 Q/K 反量化、布局转换、SDPA 和输出布局转换，因此表中的 Torch 与 Triton 数据按端到端口径对比。

### 11.3 正确性测试

```bash
cd /workspace/BaselineBenchmark
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python -m pytest -q backends/_enflame/test/test_sage_attention_native.py

FLAG_ATTN_RUN_LONG_GCU_TESTS=1 \
python -m pytest -q backends/_enflame/test/test_sage_attention_native.py
```

## 12. 优化开关与验收标准

当前实现保留以下实验开关，用于回归和消融测试：

| 环境变量 | 作用 | 默认值 |
| --- | --- | ---: |
| `FLAG_ATTN_S60_PREFETCH_V` | 控制 V 预取 | 1 |
| `FLAG_ATTN_S60_PREFETCH_K_NEXT` | 控制 D128 K-next | 1 |
| `FLAG_ATTN_S60_PREFETCH_D64` | 控制 D64 预取选择 | 1 |

后续优化只有同时满足以下条件才应替换当前路径：

1. 完整 20 个 FP16/BF16 shape 不出现不可解释的性能回退；
2. 多轮测试中位数稳定，不能只依据一次最快结果；
3. HND/NHD、GQA/MQA、mask、partial block 和 LSE 语义保持正确；
4. 输出精度不通过降低 Online Softmax 或 PV 累加精度换取；
5. 对启用条件、回退条件和额外临时空间有明确记录。
