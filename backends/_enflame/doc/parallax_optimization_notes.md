# Enflame Parallax 适配与优化记录

更新日期：2026-10-13

## 1. 范围与环境

本文参考 `chunk_gla_optimization_notes.md`，记录 Parallax 在燧原 S60/GCU300 上的适配、保留的优化、未保留的实验、当前性能及复现方法。范围包括 parallel forward、forward + backward 和单 token decode。

相关实现：

- Triton parallel：`/workspace/FlagAttention/src/flag_attn/runtime/backend/_enflame/FLA/parallax/parallel.py`。
- Triton decode：`/workspace/FlagAttention/src/flag_attn/runtime/backend/_enflame/FLA/parallax/decode.py`。
- grid 工具：`/workspace/FlagAttention/src/flag_attn/runtime/backend/_enflame/FLA/utils.py`。
- Torch parallel baseline：[parallax_torch.py](../ops/parallax/parallax_torch.py)。
- Torch decode baseline：[parallax_decode_torch.py](../ops/parallax/parallax_decode_torch.py)。
- 对比入口：[bench_parallax_torch.py](../benchmarks/bench_parallax_torch.py)、[bench_parallax_decode_torch.py](../benchmarks/bench_parallax_decode_torch.py)。

历史 KDA 实验使用 SSH 别名 `koap-baai`、宿主机仓库 `/data/zw/FlagAttention`、容器 `zw-enflame`。历史环境记录为 Torch `2.10.0+cpu` 配合 Torch-GCU、Triton `3.6.0`；不能把 Torch 版本中的 `+cpu` 解释为算子在 CPU 上运行。

第 6 节完整性能表来自用户本次提供的 GCU 测量结果，均为 BF16。本次文档整理没有重新执行服务器测试、同步文件或修改算子。

## 2. 数学定义与正确性要求

对一个 query head 和一个 query token，记 `q` 为主查询、`r` 为辅助查询。GQA 中，同组 query heads 共享 K/V。令允许参与计算的 key 集合为因果范围与可选窗口的交集：

```text
score_j = scale * dot(q, k_j)
u_j     = dot(r, k_j)
p_j     = softmax(score)_j
mu      = sum_j p_j * u_j
o       = sum_j p_j * (1 + mu - u_j) * v_j
```

`scale` 只作用于 `q @ k.T`，不作用于 `r @ k.T`。`fwd+bwd` 表示一次前向与其反向的合计时间，不是单独 backward。

优化必须保持：

1. 因果 mask、窗口边界和 packed/varlen 序列之间的隔离。
2. GQA query head 到 KV head 的映射，以及共享 K/V 的梯度求和。
3. 输出和 `dq/dr/dk/dv` 的 shape、dtype、device 与数值精度。
4. 非整 tile 的尾部 mask，特别是 `T=111`、`D=100` 等边界。
5. decode 的有效 KV 长度、窗口、非连续输入和 `out` 写回语义。

改变 FP32 累加顺序后，应使用原测试阈值验证输出与全部梯度，不通过放宽阈值接受候选。

## 3. 燧原适配

### 3.1 GQA backward reduction 的 grid 超限

GQA 非 grouped DKV 路径先生成按 query head 存储的临时 `dK/dV`，再由 `parallel_parallax_bwd_kernel_reduce_gqa` 归约到共享 KV heads。

归约按元素数划分 program：

```text
BLOCK = 256
reduction_programs = ceil(grad_k.numel() / BLOCK)
```

遇到 `grid.x = 65536` 时，超过目标 S60 的 `grid.x <= 65535` 限制。最终实现使用 `requires_grid_remap` 检查，再用 `make_grid_2d` 映射物理 grid：

- 未超过限制的小 shape 保留原来的逻辑一维 grid。
- 65536 个 program 映射为 `(32768, 2, 1)`。
- kernel 在重映射分支把三维 program ID 恢复为线性 ID：

```python
program_id = tl.program_id(0) + tl.num_programs(0) * (
    tl.program_id(1) + tl.num_programs(1) * tl.program_id(2)
)
```

重映射只改变调度位置，不改变元素索引或 GQA 求和定义；尾部仍由 `offsets < N` 掩码保护。

### 3.2 适配阶段的性能修正：GQA 归约使用单 warp

根据用户提供的分阶段计时结论，`reduce_gqa` 曾是主要瓶颈，因此将该归约 kernel 的启动参数从 `num_warps=4` 改为 `num_warps=1`。当前源码的归约启动点已固定为单 warp。

这是一项性能优化，不是编译兼容性修复。它仅作用于需要最终 GQA 归约的路径：`G > 1` 且没有使用 grouped DKV。不能据此声称其他 kernel 也应统一改为单 warp，或把全部反向收益归因于该改动。本次输入未包含该分阶段实验的完整原始表，因此不补写独立收益百分比。

## 4. 保留的 Triton 优化

### 4.1 普通与 grouped DKV 反向计算重排

在 `parallel_parallax_bwd_kernel_dkv` 和 `parallel_parallax_bwd_kernel_dkv_grouped` 中，调整独立 score 阶段的生成和消费顺序：

1. 先计算 `GU`，累加 `GU/R` 对 `dK` 的贡献，再生成 `RK`。
2. 生成混合权重后，先完成 `dV` 的累加，再生成 `GL`，累加另一部分 `dK`。

对应源码数据流可概括为：

```text
p / a / delta
    -> GU -> dK += GU.T @ R
    -> RK -> weights -> dV += weights.T @ dO
    -> GL -> dK += GL.T @ Q
```

目的是缩短中间分数矩阵的存活时间，减少同时存活的临时数据。降低寄存器压力是源码层面的优化假设；没有硬件 profiler 证据证明寄存器占用下降或 spill 被消除。该重排改变了部分 FP32 累加顺序，因此正确性验证必须包含反向梯度。

### 4.2 该改动本身的历史收益

2026-10-09 KDA 的比较对象是保存的原始 Triton，而不是 Torch。保留候选 `c001` 的最佳正式用例为：

| Shape `(B,T,H,HQ,D)` | 阶段 | 原始 Triton ms | c001 ms | 延迟降低 |
|---|---|---:|---:|---:|
| `(2,2048,2,8,64)` | fwd+bwd | 19.602489 | 19.079550 | 2.668% |

该轮最差用例延迟增加约 1.659%，满足 2% 回退保护；但没有达到“任意正式用例降低至少 10%”的目标。最终恢复并保留 `c001`，正确性复测为 `32 passed`。上述是历史单轮 baseline/candidate 对比，不声称统计显著性。

DKV 重排属于 backward 优化，不用于解释 forward 或 decode 的加速。第 6 节对 Torch 的 speedup 是完整实现之间的比较，不能当作该重排的单项优化收益。

## 5. 当前 Torch baseline

当前对比使用纯 Torch-GCU baseline，通过 Torch-GCU/TopsAten 执行计算，不调用 Triton 或 TopsCC。

parallel 默认采用 `implementation="optimized"`：

- FP32 score、归一化和矩阵计算。
- 按 query tile 计算，裁剪因果或窗口范围内需要读取的 K/V。
- 合并 GQA group/query 行，共享 K/V 矩阵。
- 将输出写为 `weights @ V`，避免分别计算两个输出矩阵乘法。
- 默认合并 Q/R 矩阵行，通过一次 matmul 生成两组 score；该方式仍需要拼接，不改变 FLOPs。
- 大训练图使用 checkpoint/recompute；满足条件时使用 streaming backward，减少保留的全量 score 激活。高阶梯度和其他不满足条件的调用保留对应 autograd 路径。

decode 使用 FP32 主/辅助 score 和稳定指数归一化，K/V 在 GQA heads 之间共享；支持有效长度、窗口和预分配 `out`。当前 baseline 不实现 paged KV `page_table`。

当前对比入口只接受 Torch baseline。此前 Parallax TopsCC 实现已移除，不作为现行复现入口。

## 6. 当前性能

数据来源：用户本次提供的完整 benchmark 表；本次未重新测量。单位为 ms，`speedup = Torch-GCU / FlagAttention`，大于 1 表示 Triton 更快。所有表均为 `bfloat16`。

### 6.1 Forward：1.661x～3.901x

`H` 为 KV head 数，`HQ` 为 query head 数。

| B | T | H | HQ | D | Torch-GCU fwd ms | FlagAttention fwd ms | Speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 15 | 2 | 2 | 64 | 0.787680 | 0.238560 | 3.302x |
| 1 | 63 | 1 | 1 | 64 | 0.690407 | 0.228766 | 3.018x |
| 1 | 111 | 2 | 2 | 64 | 0.792830 | 0.283300 | 2.799x |
| 2 | 200 | 2 | 8 | 64 | 0.876562 | 0.436167 | 2.010x |
| 2 | 256 | 2 | 8 | 64 | 0.881403 | 0.452631 | 1.947x |
| 2 | 512 | 2 | 8 | 64 | 1.444662 | 0.869953 | 1.661x |
| 2 | 1024 | 2 | 2 | 64 | 1.419793 | 0.787445 | 1.803x |
| 2 | 2048 | 2 | 8 | 64 | 11.053940 | 5.136140 | 2.152x |
| 2 | 4096 | 2 | 2 | 64 | 10.976687 | 4.832961 | 2.271x |
| 2 | 8192 | 2 | 8 | 64 | 172.312113 | 59.229796 | 2.909x |
| 3 | 111 | 2 | 2 | 100 | 0.774696 | 0.350708 | 2.209x |
| 3 | 1024 | 2 | 8 | 128 | 6.165860 | 2.966482 | 2.079x |
| 2 | 2048 | 2 | 8 | 128 | 11.648099 | 6.020683 | 1.935x |
| 2 | 8192 | 8 | 16 | 128 | 540.949498 | 138.660641 | 3.901x |

### 6.2 Forward + backward：1.641x～11.076x

| B | T | H | HQ | D | Torch-GCU fwd+bwd ms | FlagAttention fwd+bwd ms | Speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 15 | 2 | 2 | 64 | 2.394239 | 0.683927 | 3.501x |
| 1 | 63 | 1 | 1 | 64 | 2.144739 | 0.668622 | 3.208x |
| 1 | 111 | 2 | 2 | 64 | 2.460027 | 0.722866 | 3.403x |
| 2 | 200 | 2 | 8 | 64 | 3.828486 | 2.332930 | 1.641x |
| 2 | 256 | 2 | 8 | 64 | 4.984429 | 2.711567 | 1.838x |
| 2 | 512 | 2 | 8 | 64 | 19.685963 | 5.460099 | 3.605x |
| 2 | 1024 | 2 | 2 | 64 | 26.276241 | 2.372389 | 11.076x |
| 2 | 2048 | 2 | 8 | 64 | 126.302685 | 19.671903 | 6.420x |
| 2 | 4096 | 2 | 2 | 64 | 108.341903 | 17.723992 | 6.113x |
| 2 | 8192 | 2 | 8 | 64 | 1474.631610 | 233.137173 | 6.325x |
| 3 | 111 | 2 | 2 | 100 | 2.522306 | 0.887242 | 2.843x |
| 3 | 1024 | 2 | 8 | 128 | 59.010641 | 25.636828 | 2.302x |
| 2 | 2048 | 2 | 8 | 128 | 155.068447 | 24.547869 | 6.317x |
| 2 | 8192 | 8 | 16 | 128 | 5750.102878 | 522.574369 | 11.003x |

### 6.3 Decode：1.045x～9.398x

`L` 为 KV 长度，`HQ/HKV` 分别为 query/KV head 数。

| B | L | HQ | HKV | D | Torch-GCU decode ms | FlagAttention decode ms | Speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 512 | 8 | 8 | 64 | 0.298420 | 0.089575 | 3.332x |
| 1 | 512 | 8 | 8 | 128 | 0.332518 | 0.149687 | 2.221x |
| 1 | 1024 | 8 | 8 | 128 | 0.410525 | 0.178844 | 2.295x |
| 1 | 2048 | 8 | 8 | 128 | 0.590640 | 0.341141 | 1.731x |
| 1 | 4096 | 8 | 8 | 128 | 0.930857 | 0.664194 | 1.401x |
| 1 | 16384 | 8 | 8 | 128 | 2.713812 | 2.596766 | 1.045x |
| 2 | 1000 | 8 | 2 | 128 | 2.497729 | 0.271262 | 9.208x |
| 2 | 2048 | 8 | 2 | 64 | 2.534295 | 0.469925 | 5.393x |
| 2 | 2048 | 8 | 2 | 128 | 4.880421 | 0.519278 | 9.398x |
| 1 | 8192 | 8 | 2 | 128 | 9.490059 | 1.311187 | 7.238x |

### 6.4 计时口径

当前 `_parallax_compare.py` / `_parallax_timing.py` 的默认协议：

- parallel：`--timing eager`，同步墙钟计时，包含主机调用和同步开销。
- decode：`--timing graph`，预分配输出、GCU Graph 回放、Event 计时，除以每个 graph 的调用次数。
- 两边交替预热，使用 ABBA/BAAB 顺序配对采样，报告中位数与 JSON 中的 MAD。
- 默认 `warmup=10`、`iter=12`；这里表示次数/循环数，与 AttnRes 的毫秒预算不同。decode 默认 `calls-per-graph=100`。

本次输入只给出了结果表，没有完整运行参数或原始 JSON；以上描述的是当前脚本默认行为，不补写该次测量未提供的参数。graph 与 eager、历史 KDA kernel 计时不能混为同一延迟口径。

## 7. 已验证但未保留的实验

历史 KDA 记录位于 `E:/Zw/kda_workspaces/parallax_kda_20261009/`。

| 候选 | 改动 | 验证与决定 |
|---|---|---|
| c001 | 普通/grouped DKV score 阶段重排 | 32 项正确性通过；28 个性能项完成；保留为该轮最好正确版本 |
| c002 | dense multi-tile 路径统一使用独立 DQR preprocessing | 32 项正确性通过；性能未超过 c001，撤回 |
| c003 | DQR/DKV autotune 增加 `BS16/w1/s1` 候选 | 32 项正确性通过；性能运行被用户终止，没有完整性能表；撤回并恢复原配置 |

不为未完成的 c003 推断延迟或 speedup。该轮 decode 不在优化范围内。历史报告记录恢复到当时的 c001 源码与原配置，不等同于今天所有本地文件的完整版本状态；复现时应保存实际运行文件的 hash。

## 8. 主要不足与后续方向

1. 长序列 forward 仍较昂贵，DKV 反向重排并不改善这部分计算。
2. GQA 非 grouped 路径需要临时梯度缓冲和最终归约，可继续按阶段对比 grouped 与非 grouped 的成本。
3. Torch 的分块/checkpoint 控制激活内存，但增加重算和调度；应保持一个正确且经过优化的 baseline。
4. 部分 decode 大 KV、非 GQA 用例仅略快于 Torch，例如 `B1,L16384,HQ8,HKV8,D128` 为 1.045x。
5. 只有源码与 benchmark 证据，不能以寄存器压力、occupancy 或 spill 作为已证实的瓶颈结论。

后续先测各阶段，再针对瓶颈提候选。DKV 顺序、tile/warp 参数及 grouped 分派都应分别进行 A/B，不能叠加多项改动后无法归因。

## 9. 测试与复现

Torch baseline 正确性：

```bash
cd /workspace/BaselineBenchmark
pytest backends/_enflame/test/test_parallax_torch.py -vs
pytest backends/_enflame/test/test_parallax_decode_torch.py -vs
```

Triton 正确性：

```bash
cd /workspace/FlagAttention
pytest tests/flag_attn/test_parallax.py -vs
pytest tests/flag_attn/test_parallax_decode.py -vs
```

全部正式 shape 对比：

```bash
cd /workspace/BaselineBenchmark
pytest backends/_enflame/benchmarks/bench_parallax_torch.py -vs --all
pytest backends/_enflame/benchmarks/bench_parallax_decode_torch.py -vs --all
```

必须带 `--all` 才运行完整 shape 表；不带时默认只测第一个 shape。parallel 默认同时测 forward 与 fwd+bwd，可用 `--mode forward` 或 `--mode fwdbwd` 单独选择。decode 可用 `--timing eager` 单独测主机调用口径，但不能直接与 graph 结果比较。

当前测试文件覆盖前向、反向、GQA、非连续输入、packed/varlen、窗口及部分梯度等；decode 测试覆盖有效长度和 `out`。历史 KDA 的 `32 passed` 仅属于当时的固定合同，本次没有声称现行所有测试重新通过。

## 10. 可复用优化流程与验收

1. 保存原始算子、配置和 hash，明确优化的是 forward、fwd+bwd 还是 decode。
2. 建立数学参考，先通过正确性，再测原始 Triton 与优化后的 Torch baseline。
3. 固定 shapes、dtype、seed、窗口、设备、计时方式和编译/预热策略。
4. 分阶段定位瓶颈，记录源码假设与实测证据的边界。
5. 每次只验证一个候选；检查输出和全部梯度，再运行完整性能矩阵。
6. 对照原始 Triton 计算优化收益，对照 Torch 计算外部 speedup，分别报告。
7. 保留所有慢结果和中断记录；失败候选回滚到最好正确版本。
8. 新 shape、varlen、窗口与不同 dtype 需要重新验收，不能由本表外推最佳性能。

通用任务、实验记录与报告模板可参考 `../workflow/operator_optimization_workflow.yaml` 及同目录模板。若开展新的 KDA 远程优化，远程命令、同步和 Wiki 读取继续使用 KDA 受控接口。
