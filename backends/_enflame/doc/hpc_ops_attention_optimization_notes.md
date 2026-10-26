# Enflame HPC Ops Attention 适配与优化记录

更新日期：2026-09-28

## 1. 范围与环境

本文记录 HPC Ops Attention 在燧原 S60/GCU300 上的适配过程、生产 Triton 优化、TopsCC baseline、性能结果、失败实验和后续方向。覆盖的算子包括：

- FP8 Block-Sparse Prefill；
- BF16 Dynamic Decode；
- FP8 Dynamic Decode；
- NHD/HND 两种 KV cache 布局；
- MTP、多 Query Head/GQA、Paged KV Cache 和动态 Split-K。

相关生产实现：

- `FlagAttention/src/flag_attn/runtime/backend/_enflame/hpc_ops_attention/prefill/attention_blocksparse_prefill_fp8.py`
- `FlagAttention/src/flag_attn/runtime/backend/_enflame/hpc_ops_attention/decode/assign_task.py`
- `FlagAttention/src/flag_attn/runtime/backend/_enflame/hpc_ops_attention/decode/dynamic/bf16_dynamic.py`
- `FlagAttention/src/flag_attn/runtime/backend/_enflame/hpc_ops_attention/decode/dynamic/fp8_qpertoken_perhead_kvpertensor_dynamic.py`
- `FlagAttention/src/flag_attn/runtime/backend/_enflame/hpc_ops_attention/decode/dynamic/fp8_qkpertoken_perhead_vperhead_dynamic.py`

相关 baseline 实现：

- TopsCC 设备源码：`BaselineBenchmark/backends/_enflame/ops/hpc_ops_attention/topscc/attention.cpp`
- TopsCC host launcher：`BaselineBenchmark/backends/_enflame/ops/hpc_ops_attention/topscc_decode_extension.cpp`
- BF16 Decode 封装：`BaselineBenchmark/backends/_enflame/ops/hpc_ops_attention/topscc_decode.py`
- FP8 Decode 封装：`BaselineBenchmark/backends/_enflame/ops/hpc_ops_attention/topscc_decode_fp8.py`
- FP8 Prefill 封装：`BaselineBenchmark/backends/_enflame/ops/hpc_ops_attention/topscc_prefill.py`
- FP8 公共辅助：`BaselineBenchmark/backends/_enflame/ops/hpc_ops_attention/fp8_utils.py`

TopsCC 设备源码来自 EnflameTechnology/Ubridge 的固定提交 `04c17896adbff718369b44cc56501d52d34e3662`，编译目标为 `gcu300`。Python 和 PyTorch C++ Extension 只负责参数准备、JIT 构建、当前 stream 获取和 kernel launch，不承担 Attention 热路径数学计算。

当前 BaselineBenchmark 已删除 Torch eager reference/provider。性能对比口径统一为：

```text
TopsCC baseline  vs.  FlagAttention adapted Triton
```

## 2. 数学语义与正确性红线

对每个 query，Attention 保持标准的缩放点积与在线 Softmax 语义：

```text
score(q, k) = (q @ k^T) / sqrt(head_dim)
p           = softmax(score + causal/sparse mask)
output      = p @ v
```

FP8 路径在反量化后计算：

```text
q_real = decode_e4m3fn(q_bits) * q_scale
k_real = decode_e4m3fn(k_bits) * k_scale
v_real = decode_e4m3fn(v_bits) * v_scale
```

所有优化必须满足以下条件：

1. 不能通过放宽容差掩盖 FP8 解码、scale、stride、page table 或 mask 错误。
2. NHD/HND 必须保持同一逻辑 KV 序列，不能因物理布局不同改变结果。
3. Decode 必须遵守每个 sequence 的实际 `kv_len` 和 causal 上界。
4. Block-Sparse Prefill 必须只跳过 mask 未选中的 KV block，不能跳过有效块。
5. Split-K combine 必须按 online-softmax 公式合并 partial max、sum 和 accumulator。
6. 输出以及中间 LSE/partial 必须有限；不得出现 NaN、Inf、非法访问或 SIP 异常。
7. FP8 原始字节传输必须保持 E4M3FN bit pattern，不得先在 GCU 上执行不受支持的 dtype 转换。

## 3. S60 适配问题与解决方案

### 3.1 GCU 不支持随机浮点张量直接转换为 FP8

原测试会执行：

```python
q = float_tensor.to(torch.float8_e4m3fn)
```

其失败链路为：

```text
Float/BF16 随机数据
    -> .to(torch.float8_e4m3fn)
    -> torch_gcu / topsatenCopy
    -> S60 不支持该 dtype conversion
    -> 进程 Abort
```

最终处理方式是在 CPU 上完成相同的 FP8 E4M3FN 量化，再将 FP8 原始字节复制到 GCU。生产 kernel 和 TopsCC baseline 都按原始 bit pattern 解释数据，因此绕过 eager 转换后数学语义不变。公共字节解码逻辑集中在 `fp8_utils.py`，不再保留完整 Torch Attention baseline。

### 3.2 动态任务调度避免大 Batch 编译卡死

原实现把 `B`、`MAX_TASKS` 等规模参数放入编译期，并静态展开任务循环。`batch=200` 时会产生过大的 IR 和目标代码，导致编译耗时或内存失控。

S60 适配后：

- 输入规模尽量改为运行时参数；
- 静态展开改为动态循环；
- 在 GCU 上动态生成 task map；
- task map 和 Split-K partial 放入可复用 workspace；
- 调度依据实际 KV 长度分布，不依据 benchmark case 名称特化。

这些变化只修改任务生成、并行度和中间结果组织，不改变 Attention 计算。

### 3.3 替换 Hopper 专用机制

原路径依赖下列 Hopper 专用能力：

- TLE/TMA；
- CTA Cluster 和 Cluster Barrier；
- PDL；
- `maxnreg` 等寄存器控制；
- Hopper 专属流水和加载方式。

这些机制不能直接在 S60 上运行，最终替换为：

- S60 专用 tile、warp 和 stage 配置；
- 普通 Triton block pointer 或显式指针加载；
- S60 动态 Split-K 与 task map；
- 独立的 partial output/LSE combine kernel；
- 较浅的间接分页加载 pipeline；
- 能避免 legalization failure、SIP 和非法访问的保守启动参数。

## 4. 当前生产 Triton 实现结构

### 4.1 FP8 Block-Sparse Prefill

```text
CPU FP8 量化/原始字节准备
    -> page table、block mask、scale metadata
    -> 一个 KV 循环处理一个 64-token physical page
    -> FP8 E4M3FN 位解码
    -> sparse block early skip / dense tile fast path
    -> online softmax
    -> output
```

保留的关键优化：

- Q/K/V 以 `uint8` 原始字节进入 kernel，在 Triton 内通过位运算解码 E4M3FN。
- KV 计算与 64-token 物理页对齐，每轮只读取一个 `physical_block_id`。
- Q/K/V 使用连续 block pointer，减少分页缓存的零散 gather 和地址计算。
- per-token/per-head K scale 在调用前重排成连续 page 数据。
- per-tensor K scale 在 KV 循环外并入 Q scale。
- 将 `log2(e) / sqrt(head_dim)` 提前融合进 Q scale。
- 将 Softmax 概率的 FP8 `×256` 缩放融合为 `exp2(... + 8)`。
- 只在行最大值变化时重缩放 online-softmax accumulator。
- 未选中的 sparse block 在页表查询前跳过，避免 K/V load、FP8 decode 和 dot。
- 完整有效且完全位于 causal 范围内的 page 走 dense fast path。
- 默认 `num_warps=1`、`num_stages=1`，KV pipeline 深度以 constexpr 控制。

### 4.2 BF16/FP8 Dynamic Decode

```text
运行时 KV 长度
    -> 动态 task map
    -> 多个 Split-K producer
    -> split_out / split_lse
    -> combine kernel
    -> 最终 output
```

每个 producer 使用安全的 M8 QK/PV tile。同一 program 只加载一次 K/V tile，然后依次计算多个 MTP query：

```text
load K/V -> Q0 attention -> Q1 attention -> Q2/Q3 attention
```

每个 query 保留独立的 causal 上界和 online-softmax 状态。实测 MTP3 相比 MTP1 通常只增加约 27%～52%，而不是线性增加到 3 倍。

## 5. 已保留的 Decode 优化

### 5.1 S60 安全执行配置

- 使用一维扁平 grid，规避 GCU 多维 grid 限制。
- 最大 CTA 数为 2048，workspace 上限为 8 GiB。
- QK/PV 固定使用 M8；M16/M32 会触发 legalization failure 或 SIP。
- `num_warps=1`。
- MTP1 使用 `num_stages=2`；MTP2/MTP3 使用 `num_stages=3`。

### 5.2 物理页对齐与 K/V 复用

- 计算 tile 与 64-token physical page 对齐。
- 每页只读取一次 physical block id。
- 用 `tl.make_block_ptr` 表达连续 `64×128` K/V 块。
- 根据真实 stride 访问缓存，NHD/HND 共用 kernel，无需转置或复制。

代表性 MTP1 消融结果：

| Case | 优化前 | 优化后 | 加速 |
|---|---:|---:|---:|
| uniform_512 | 21.36 ms | 2.76 ms | 7.7x |
| uniform_4096 | 152.55 ms | 3.74 ms | 40.8x |
| one_128k_31x4k | 152.64 ms | 3.79 ms | 40.2x |

其中 64-token 页对齐单独带来约 9%～10% 的长序列收益，连续 block pointer 是整体收益最大的基础优化。

### 5.3 Split-K 粒度与并行度

- 每 CTA 的初始最小粒度由 128 token 提高到 512 token。
- 输入包含短序列时最终粒度取 2048 token。
- 所有序列均不少于 4096 token 时取 4096 token。
- 未采用 8192 token，因为会损失部分 128K 长序列的并行度。

`uniform_512` NHD/MTP1 从 2.7560 ms 降到 1.0754 ms，约提升 2.56 倍。

### 5.4 并行 task-map

原 task-map 由单 warp、`grid=(1,)` 串行构建，固定开销约 2.2～2.45 ms。当前改为 `grid=(num_ctas,)`，每个 CTA 构造自己的连续任务列表：

- `uniform_512`：1.0754 ms -> 0.7881 ms；
- `skewed_extreme`：1.6038 ms -> 0.3690 ms。

另外保留：

- 复用 host/workspace 已计算的 `total_tiles`；
- uniform batch 由扁平 tile id 直接推导 sequence/head/CTA；
- batch 32～128 的非均匀输入使用整数 `tl.cumsum` 构造 vector prefix；
- 其他 batch 大小使用标量路径。

完整 54 个配置中，减少重复 batch 扫描的几何平均收益约为 1.045 倍。

### 5.5 Split-K Combine

一个 combine program 同时归并多个相邻 query head：

- 支持 HPP2、HPP4、HPP8；
- MTP2/MTP3 优先 HPP8；
- MTP1 通常 HPP4；
- 任务充足，或 `batch <= 16` 且 GQA group 足够大时，MTP1 也使用 HPP8。

该策略减少 combine grid、partial 状态读取和 program 数。历史 c54 覆盖中，MTP1 小 batch 使用 HPP8 曾把 TopsCC 95% 阈值覆盖从 40/54 提高到 41/54。

### 5.6 完整物理页快速路径

当运行时最短 KV 长度不少于 4096 时，task map 可以证明 producer 读取完整 64-token 页：

- K/V block pointer load 移除 `boundary_check`；
- 短序列和混合短序列保留安全边界路径；
- `FULL_PAGE_LOADS and NUM_SEQ_Q <= 2` 时进一步移除重复的 `valid_n` predicate；
- 仍然保留 causal 和总长度约束；
- 当前仅对 MTP1/MTP2 启用，MTP3 继续保留 mask。

## 6. 当前性能

以下数据来自原始适配记录。`TopsCC/Triton` 大于 1 表示生产 Triton 更快，小于 1 表示 TopsCC baseline 更快。

### 6.1 总览

| 路径 | 配置 | TopsCC/Triton 范围 | 结论 |
|---|---|---:|---|
| FP8 Block-Sparse Prefill | NHD | 1.670x～121.214x | Triton 全部更快 |
| FP8 Block-Sparse Prefill | HND | 1.825x～123.186x | Triton 全部更快 |
| BF16 Dynamic Decode | MTP1 | 0.460x～2.975x | 形状相关，存在 TopsCC 更快区间 |
| BF16 Dynamic Decode | MTP2 | 0.695x～4.535x | 形状相关 |
| BF16 Dynamic Decode | MTP3 | 0.703x～4.830x | 形状相关 |
| FP8 Dynamic Decode | MTP1 | 5.544x～147.169x | Triton 全部显著更快 |
| FP8 Dynamic Decode | MTP2 | 3.213x～86.310x | Triton 全部显著更快 |
| FP8 Dynamic Decode | MTP4 | 1.275x～48.158x | Triton 全部更快 |

### 6.2 FP8 Block-Sparse Prefill 代表性结果

| Layout | Q | KV | Quant | Mask | Skip | Triton | TopsCC | TopsCC/Triton |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| NHD | 127 | 127 | 0 | False | 0.0% | 0.394970 ms | 3.869938 ms | 9.798x |
| NHD | 512 | 32768 | 0 | False | 0.0% | 69.136936 ms | 115.485230 ms | 1.670x |
| NHD | 512 | 4096 | 1 | True | 73.3% | 3.187974 ms | 236.565512 ms | 74.206x |
| NHD | 127 | 127 | 1 | True | 0.0% | 0.313842 ms | 38.042166 ms | 121.214x |
| HND | 127 | 127 | 0 | False | 0.0% | 0.379768 ms | 4.065490 ms | 10.705x |
| HND | 512 | 32768 | 0 | False | 0.0% | 69.073624 ms | 126.051025 ms | 1.825x |
| HND | 512 | 4096 | 1 | True | 73.3% | 3.183683 ms | 237.923341 ms | 74.732x |
| HND | 127 | 127 | 1 | True | 0.0% | 0.311316 ms | 38.349785 ms | 123.186x |

### 6.3 BF16 Dynamic Decode 代表性结果

| MTP | Workload | Layout | Triton | TopsCC | TopsCC/Triton |
|---:|---|---|---:|---:|---:|
| 1 | uniform_512 | NHD | 0.301104 ms | 0.895863 ms | 2.975x |
| 1 | uniform_4096 | NHD | 1.367343 ms | 1.040153 ms | 0.761x |
| 1 | two_32k_30x4k | NHD | 1.379373 ms | 0.634658 ms | 0.460x |
| 1 | one_128k_31x4k | HND | 1.528512 ms | 3.305193 ms | 2.162x |
| 2 | uniform_512 | HND | 0.377107 ms | 1.710007 ms | 4.535x |
| 2 | skewed_extreme | NHD | 0.615704 ms | 0.427934 ms | 0.695x |
| 3 | uniform_512 | NHD | 0.521972 ms | 2.521363 ms | 4.830x |
| 3 | two_32k_30x4k | NHD | 2.463702 ms | 1.733014 ms | 0.703x |

BF16 Decode 尚未在所有长度分布上稳定超过 TopsCC。主要回退集中于中等长度、极端混合长度和 Split-K 固定开销占比较高的场景。

### 6.4 FP8 Dynamic Decode 代表性结果

| MTP | Workload | Quant Type | Layout | Triton | TopsCC | TopsCC/Triton |
|---:|---|---|---|---:|---:|---:|
| 1 | uniform_512 | qpertoken_perhead_kvpertensor | HND | 0.929832 ms | 136.842508 ms | 147.169x |
| 1 | one_64k_7x4k | qkpertoken_perhead_vperhead | NHD | 4.268707 ms | 23.664995 ms | 5.544x |
| 2 | uniform_512 | qpertoken_perhead_kvpertensor | NHD | 1.635792 ms | 141.185143 ms | 86.310x |
| 2 | one_64k_7x4k | qkpertoken_perhead_vperhead | HND | 7.933254 ms | 25.489203 ms | 3.213x |
| 4 | uniform_512 | qpertoken_perhead_kvpertensor | NHD | 2.934234 ms | 141.306669 ms | 48.158x |
| 4 | one_64k_7x4k | qkpertoken_perhead_vperhead | NHD | 20.863954 ms | 26.609436 ms | 1.275x |

## 7. 已验证但未保留的方案

### 7.1 GCU eager FP8 转换

S60 的 `topsatenCopy` 不支持 Float/BF16 到 E4M3FN 的该转换路径，会导致进程 Abort。该方案已由 CPU 量化和原始字节复制替代。

### 7.2 把输入规模作为编译期常量

对大 batch 静态展开 task loop 会导致编译卡死或生成代码爆炸。当前只保留运行时 task map 和动态循环。

### 7.3 M16/M32 Decode tile

在 S60 上会触发编译 legalization failure 或运行时 SIP，不能用理论上更大的矩阵 tile 替代已验证的 M8 配置。

### 7.4 Split-K 粒度 8192

虽然能减少 partial 和 combine 数量，但会削弱部分 128K 长序列的 producer 并行度，因此未采用。

### 7.5 深 pipeline 和激进 warp/stage

间接分页加载叠加深流水可能导致非法访问、NaN 或设备失效。当前保留的 warp/stage 是稳定性边界，不应只凭单一形状的速度扩大。

## 8. TLE/CTA 内归约实验

### 8.1 目标

当前中等长度和混合序列的主要瓶颈不是 Attention 数学本身，而是 Split-K 的调度、全局 partial 和二次归约：

```text
task map
    -> 多个 Split-K producer
    -> split_out / split_lse 写全局内存
    -> combine kernel
    -> 读取全部 partial
    -> online-softmax 合并
```

尝试方向是把多个计算 worker 组织到同一 CTA，在 CTA 内共享 online-softmax 状态，每个 CTA 只生成一个 partial，再进行全局 combine。

### 8.2 `tle.gpu.alloc(scope=smem) + local_ptr`

用途：为多个计算 worker 分配 CTA shared memory。

编译错误：

```text
error: operation being parsed with an unregistered dialect
```

结论：Python 和 Triton 前端能接受接口，但生成的 TLE 内存操作无法被 Enflame MLIR pipeline 解析，未进入 S60 shared-memory lowering。由于探针必须同时使用 `alloc + local_ptr` 才能产生真实读写，目前不能进一步区分哪个操作先缺少注册。

### 8.3 `tle.gpu.warp_specialize`

用途：在同一 CTA 内划分多个执行 partition。

编译错误：

```text
error: failed to legalize operation 'ttg.warp_specialize'
that was explicitly marked illegal
```

结论：前端已经生成 `ttg.warp_specialize`，但 Enflame GCU 后端没有将其 lowering 为 S60 worker/SIP partition 的 legalization。此失败与 shared memory 和 pipe 无关。

### 8.4 `tle.pipe(scope="cta")`

用途：建立同一 CTA 内 producer/consumer worker 的同步关系。

编译错误：

```text
error: Dialect `tle` not found for custom op 'tle.pipe.create'
RuntimeError: Pipeline run failed: Failed to parse input MLIR
```

结论：前端能生成 `tle.pipe.create`，但 Enflame MLIR context 没有注册对应 dialect，编译在解析阶段失败，尚未进入 S60 同步原语 lowering。

因此当前 S60 软件栈不能用这些 TLE 原语完成 CTA 内多 worker 归约。除非编译器新增 dialect 注册和 lowering，否则不应继续在同一方向堆叠实现。

## 9. 主要不足与瓶颈

### 9.1 BF16 Decode

- 中等长度和混合长度下，task-map、partial 写回和 combine 占比过高。
- 不同 workload 的最优 Split-K 粒度不同，固定阈值仍会产生局部回退。
- `two_32k_30x4k` 和 `skewed_extreme` 等场景 TopsCC 仍更快。
- S60 缺少已验证的 CTA 内 worker 协作原语，限制了 producer/combine 融合。

### 9.2 FP8 Decode

- per-token/per-head K scale 和 per-head V scale 增加额外读取及地址计算。
- MTP4、长序列、复杂量化组合的领先幅度明显小于短序列 per-tensor 路径。
- TopsCC baseline 的 FP8 路径包含原始字节解码和元数据准备，必须确保计时边界与 adapted 路径一致。

### 9.3 Block-Sparse Prefill

- 不稀疏的 32K KV 路径仍受 O(Q×KV) 计算和 K/V 带宽约束。
- 稀疏收益依赖 skip ratio 和 block 分布；只报告平均 sparsity 不足以描述负载均衡。
- page table、scale 重排和 mask lowering 应区分准备时间与稳态 kernel 时间。

### 9.4 Baseline 集成

- baseline 热路径为 TopsCC 设备 kernel，但仍需 Python/C++ launcher 完成构建和启动。
- 首次 JIT 构建时间不能混入稳态 kernel latency。
- Torch eager Attention baseline 已删除；`fp8_utils.py` 仅提供数据表示和测试辅助，不是候选实现。

## 10. 后续优化优先级

### P0：优先验证

1. **BF16 Split-K/Combine 自适应阈值**

   以 `min/max/sum(kv_lens)`、有效 tile 数和 head 数为输入选择粒度，重点改善 `skewed_extreme` 与 `two_32k_30x4k`，同时保持长序列并行度。

2. **减少无 combine 价值的 Split-K**

   当单 producer 已能占满设备或 partial 数过小时，直接写最终输出，跳过 split workspace 和 combine launch。

3. **task-map 与 producer 元数据融合**

   继续消除对 batch prefix、physical page 和 head 映射的重复读取；必须用全矩阵 benchmark 验证，不能按 case 名称分支。

4. **计时边界审计**

   对 TopsCC 与 adapted 路径统一 metadata preparation、cache packing、warmup 和同步边界，分别记录 prepare 与 execute。

### P1：需要系统 benchmark

1. 对 BF16 MTP1/2/3 分别搜索 Split-K 粒度、HPP、stage，而不是共享单一策略。
2. 对 FP8 两类 quant mode 分别搜索 scale load tile 和 page pipeline。
3. 根据 sparse mask 的连续块分布做 page-run 合并，而不只按 skip ratio 优化。
4. 对 full-page fast path 扩展 MTP3，但必须先证明寄存器压力和 mask 删除不会引发 SIP。
5. 离线固化常用 shape 的安全配置，降低首次 autotune/JIT 成本。

### P2：依赖编译器能力

- S60 shared-memory TLE dialect 注册与 lowering；
- `ttg.warp_specialize` 的 GCU legalization；
- `tle.pipe.create` 的 CTA 同步 lowering；
- 能表达 CTA 内多 worker online-softmax 归约的稳定原语。

## 11. 测试与复现

### 11.1 正确性测试

```bash
cd /workspace/BaselineBenchmark
python -m pytest -q \
  backends/_enflame/test/test_hpc_ops_decode_topscc.py \
  backends/_enflame/test/test_hpc_ops_decode_fp8_topscc.py \
  backends/_enflame/test/test_hpc_ops_prefill_topscc.py
```

测试覆盖：

- BF16 Dynamic Decode；
- FP8 Dynamic Decode 两类量化模式；
- FP8 Block-Sparse Prefill；
- NHD/HND；
- TopsCC baseline 与 FlagAttention adapted 实现的输出对比；
- CPU 侧 FP8 量化和原始字节传输路径。

### 11.2 Decode benchmark

单配置示例：

```bash
cd /workspace/BaselineBenchmark
python backends/_enflame/benchmarks/bench_hpc_ops_decode.py \
  --implementation topscc-compare \
  --case uniform_512 \
  --quant-type bf16 \
  --layout NHD \
  --mtp 1 \
  --device 2 \
  --warmup 2 \
  --repeat 5
```

全量矩阵：

```bash
python backends/_enflame/benchmarks/bench_hpc_ops_decode.py \
  --implementation topscc-compare --all --device 2
```

BF16 支持 MTP 1/2/3；FP8 支持 MTP 1/2/4。

### 11.3 Prefill benchmark

单配置示例：

```bash
cd /workspace/BaselineBenchmark
python backends/_enflame/benchmarks/bench_hpc_ops_prefill.py \
  --implementation topscc-compare \
  --case q127_kv127 \
  --layout NHD \
  --no-masked \
  --device 2 \
  --warmup 2 \
  --repeat 5
```

全量矩阵：

```bash
python backends/_enflame/benchmarks/bench_hpc_ops_prefill.py \
  --implementation topscc-compare --all --device 2
```

若本机 case 名称发生变化，以脚本 `--help` 输出为准。

## 12. 后续优化验收标准

1. 从干净进程运行，首次编译时间与稳态执行时间分开记录。
2. warmup 后进行设备同步，报告 median，并保留 min/max 或原始 JSON。
3. 同时覆盖 NHD/HND、短/中/长/混合 KV、所有支持的 MTP。
4. FP8 同时覆盖 `qpertoken_perhead_kvpertensor` 和 `qkpertoken_perhead_vperhead`。
5. Prefill 同时覆盖 dense、masked 和高 skip-ratio 场景。
6. 正确性必须覆盖 finite、causal、page tail、非均匀 `kv_lens` 和物理页重排。
7. 不允许使用 benchmark case 名称硬编码调度策略。
8. 任何新配置必须在全量矩阵上无 SIP、无非法访问、无 NaN。
9. 对性能回退必须解释是 kernel 计算、task-map、workspace 还是 combine 所致。
10. 未稳定超过当前实现的候选应回滚，不能因为更底层或更复杂而保留。

## 13. 可复用的 S60 Attention 优化经验

1. 先解决编译与执行稳定性，再扩大 tile、warp 或 pipeline。
2. Paged Attention 的首要优化通常是物理页对齐和连续 block load，而不是盲目增加并行度。
3. MTP 应优先复用 K/V，query 保持独立 online-softmax 状态。
4. Split-K 的收益必须扣除 task-map、全局 partial 和 combine 成本。
5. 稀疏块应在页表查询与数据加载前跳过。
6. 运行时可证明的 full-page/full-tile 条件可用于删除边界判断，但不能删除 causal 和总长度约束。
7. Python API 存在不等于后端支持；TLE 能否使用必须以 MLIR dialect 注册、legalization 和设备运行三层证据为准。
8. 对 S60 不支持的 FP8 eager 转换，应在 CPU 保持 bit-exact 量化并传输原始字节。
9. baseline 的设备热路径、host launcher、数据准备和计时边界必须明确分层。
10. 性能结论必须写清比值方向；`TopsCC/Triton > 1` 才表示 Triton 更快。
