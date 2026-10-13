# Enflame AttnRes 适配与优化记录

更新日期：2026-10-13

## 1. 范围与环境

本文参考 `chunk_gla_optimization_notes.md`，记录 Attention Residuals（AttnRes）在燧原 S60/GCU300 上的适配、Torch baseline、Triton 优化、失败实验、当前性能及复现方法。

相关实现：

- Triton：`/workspace/FlagAttention/src/flag_attn/runtime/backend/_enflame/FLA/attnres/fused.py`。
- 数学参考：`/workspace/FlagAttention/src/flag_attn/testing/attnres.py`。
- Torch baseline：[torch_reference.py](../ops/attnres/torch_reference.py)。
- Torch/Triton 对比：[bench_attnres_torch.py](../benchmarks/bench_attnres_torch.py)。
- Torch 独立测试：[test_attnres_torch.py](../test/test_attnres_torch.py)。
- Triton benchmark：`/workspace/FlagAttention/benchmark/test_fused_attnres_benchmark.py`。
- Triton 测试：`/workspace/FlagAttention/tests/flag_attn/test_attnres.py`。

历史 KDA 使用 SSH 别名 `koap-baai`、宿主机仓库 `/data/zw/FlagAttention`、容器 `zw-enflame`。环境记录为 Torch `2.10.0+cpu` 配合 Torch-GCU、Triton `3.6.0`。最终 Triton 为前向推理实现，不提供 autograd 公式；支持 FP16/BF16、最多 16 个残差源、隐藏维度不超过 8192。

第 6 节性能来自用户本次提供的结果。本次仅整理本地文档，没有重新执行服务器 benchmark 或同步文件。

## 2. 数学定义与正确性要求

对每一行，`r_l` 为第 `l` 个残差源，`q` 为共享 query，`w` 为 scoring RMSNorm 权重：

```text
rstd_l = rsqrt(mean_d(r_l[d]^2) + eps)
logit_l = sum_d(q[d] * w[d] * r_l[d]) * rstd_l
p_l = softmax_l(scale * logit_l)
o[d] = sum_l(p_l * r_l[d])
```

可选输出归一化：

```text
o[d] = o[d] * rsqrt(mean_d(o[d]^2) + eps) * output_rms_weight[d]
```

评分使用 RMS-normalized residual；混合的是原始 residual。`return_weights=True` 返回 FP32 的源概率，不能把归一化后的 residual 误用于输出混合。

必须保持：

1. softmax 沿残差源维度计算，所有行之间相互独立。
2. RMS、logits、在线 softmax 和混合累加使用 FP32，最终输出转回输入 dtype。
3. 非整幂隐藏维度的 mask，特别是 `D=7168` 对应 `BD=8192`。
4. 多维输入展平与输出还原，以及输出归一化开关。
5. `return_weights` 两条路径的结果一致；输入不能被修改。
6. 使用原有误差阈值，不通过放宽容差接受性能候选。

## 3. 燧原适配

### 3.1 launcher 不支持 Tensor tuple 启动参数

目标燧原 launcher 不能将张量 tuple 作为一个内核启动参数传入。适配方式：

- 将 residual tuple 拆成 `residual_0`～`residual_15` 共 16 个独立指针。
- 在 JIT kernel 内重新组装 tuple，通过编译期索引选择来源。
- 未使用的指针槽位传入第一个 residual 的指针作为占位；循环上限为真实源数量 `L`，不会计算占位来源。

API 仍接受 residual 序列，启动参数的变化不传递给调用者。

### 3.2 指针上的 `tl.where` 生成不支持的 `arith.select`

原选择残差指针的 `tl.where` 生成 pointer-valued `arith.select`，目标编译器报错：

```text
failed to legalize operation 'arith.select' that was explicitly marked illegal
```

改为编译期静态循环，直接从当前源的基地址加载：

```python
for source in tl.static_range(0, L):
    residual_values = tle.load(residuals[source] + row * D + offsets_d, ...)
```

这项修改首先是编译兼容性修复。后续 KDA 的原始 baseline 已包含它，因此不能把它再次计为 KDA 新增性能收益。

## 4. 当前 Triton 结构与保留的优化

### 4.1 已有融合结构

一个 program 处理一行，依次读取所有残差源，融合 RMS scoring、稳定在线 softmax、原始残差混合及可选输出 RMSNorm。无需物化完整的 normalized residual 或源概率后再做混合。

默认 `return_weights=False`，不分配 logits/LSE 工作区；请求权重时 kernel 写 FP32 logits/LSE，再通过 Torch `exp` 还原概率。这一可选后处理会增加分配与调用成本，正式性能表不包含它。

当前实现必须有 TLE，否则直接报错，没有纯 Triton fallback。只使用一个 TLE 接口 `tle.load` 来读取残差，参数保留 `eviction_policy="evict_first"`。`L <= 2` 且没有输出 RMSNorm 时 `is_async=True`，其他情况为 False。query 和权重读取、归约、指数和写回使用标准 `tl` 接口。

上述融合与 TLE 加载设计在 KDA 优化前已存在，本轮保留它们，不宣称由本轮新实现。

### 4.2 调整默认启动配置

根据 S60 实测，将默认 `num_warps=4` 改为 `num_warps=1`，`num_stages=1`。八个批量正式用例在最终 KDA 候选中相对原始 Triton 延迟降低约 51%～73%。

这是启动参数敏感性的实测结果。没有硬件 profiler 证据，不解释为已经证实 occupancy 提升、寄存器占用降低或 spill 消除；也不保证所有其他 shape 都适用。

### 4.3 绕过单配置 autotuner，减少主机调度开销

仅有一个固定配置时，仍经过 `triton.autotune` 会执行参数处理、配置缓存键构造、缓存查询和配置转发。最终实现默认直接启动 JIT kernel：

```python
launch_options = {"num_warps": 1, "num_stages": 1}
```

`FLAG_ATTN_ATTNRES_AUTOTUNE` 未设置或为 `0` 时走这条路径；为 `1` 时保留 autotuner。环境开关在模块导入时确定，切换时应启动新进程。

该优化主要减少主机调度成本，特别有利于短 kernel、小行数的完整 API 调用。Event 和同步墙钟测量的时间范围不同，不能将墙钟实验中的相对收益直接套用到 Event 表。

### 4.4 可选 autotune 配置

开启时搜索四组 `(num_warps, num_stages)`：

```text
(1,1), (2,1), (4,1), (4,2)
```

键为 `L、D、N_BUCKET、HAS_ONORM、RETURN_WEIGHTS、ASYNC_LOAD`。`N_BUCKET` 按 `N <= 128`、`N <= 1024`、更大行数分为三档，只用于配置缓存键，不参与 kernel 数学计算。

历史实际选择中，`L2_N1_D7168_ON0` 为 `(4,1)`，`L2_N1_D7168_ON1` 为 `(4,2)`，其他八项为 `(1,1)`。这是一次调优运行的观测，不保证每次选择相同；调优器测量 kernel 调用范围，选中的配置也不保证完整 API 更快。

## 5. 当前 Torch baseline

当前基线为纯 Torch-GCU，通过 Torch-GCU/TopsAten 计算，不调用 Triton；CPU 输入用于独立参考验证。

默认 `ATTNRES_TORCH_OPT_LEVEL=1`、`ATTNRES_TORCH_SCORE_MODE=auto`：

- residual 堆叠为 FP32 `[N,L,D]`，直接用于加权输出 BMM，减少大中间张量的布局转换。
- `N <= 128` 使用 native RMSNorm + MM 计算 score。
- 更大 N 使用 factorized scoring：计算 `dot(residual, query * rms_weight * scale)`，再乘 inverse RMS，避免完整 normalized keys 工作区。
- 使用 `vector_norm` 避免物化 residual 平方张量。
- 源维度 softmax 后，用 BMM 得到混合输出；可选输出 RMSNorm 使用 native 接口。
- 输出新分配，输入不变，不缓存结果。

行数分界是当前启发式，不是已证明的任意 shape 最优规则。`OPT_LEVEL=2` 的直接 FP32 stack、scaled MM 与 norm 算术候选仅有本地正确性验证，不能当作已证明比默认更快的 GCU 实现。

旧的低性能 Torch 原始版本已从活动实现中移除，当前 benchmark 只比较优化后的 Torch 与 Triton。TopsCC AttnRes 虽然修复后能运行，但正式用例性能明显较差，已按用户要求移除；不保留其运行入口。

## 6. 当前性能与计时协议

### 6.1 当前 Torch / Triton 对比：1.953x～5.222x

数据来源：用户本次提供的结果；不是本次文档编写重新测量。所有用例为 BF16、`D=7168`、`return_weights=False`，Triton 默认关闭 autotune、固定 `(1,1)`。

`L` 为残差源数，`N` 为展平后的行数，`ON0/ON1` 分别表示关闭/开启输出归一化。单位 ms，`speedup = Torch / Triton`。

| Case | Torch ms | Triton ms | Speedup |
|---|---:|---:|---:|
| L2_N1_D7168_ON0 | 0.356685 | 0.082095 | 4.345x |
| L2_N1_D7168_ON1 | 0.470573 | 0.090120 | 5.222x |
| L5_N128_D7168_ON0 | 0.719340 | 0.232935 | 3.088x |
| L5_N128_D7168_ON1 | 0.829567 | 0.275295 | 3.013x |
| L5_N1024_D7168_ON0 | 2.451120 | 1.172025 | 2.091x |
| L5_N1024_D7168_ON1 | 2.973195 | 1.469910 | 2.023x |
| L9_N128_D7168_ON0 | 1.126470 | 0.335205 | 3.361x |
| L9_N128_D7168_ON1 | 1.232153 | 0.377385 | 3.265x |
| L9_N8192_D7168_ON0 | 29.260216 | 14.354535 | 2.038x |
| L9_N8192_D7168_ON1 | 33.127884 | 16.959968 | 1.953x |

最后一行用户提供的 speedup 为 1.984x，与两列延迟不一致；本文保留原始延迟，按 `33.127884 / 16.959968` 修正为 1.953x，并相应修正汇总范围。

### 6.2 与 Triton benchmark 一致的当前协议

`bench_attnres_torch.py` 已改为直接调用 `flag_attn.testing.backend.do_bench`：

- 默认 7 轮，每轮预热 1000 ms、采样 100 ms，`return_mode="median"`。
- GCU 有 `Event` 时使用设备 Event，否则回退同步墙钟。
- 编译在 timer 采样前完成；Torch/Triton 交替测量，两边使用相同协议。
- 最终取各轮中位数的中位数；JSON 保留双方每轮结果、计时类型与环境信息。
- 两边调用完整 forward API，并自行分配输出；不是预分配内部 kernel 的独立计时。
- Event 记录设备流上的时间区间，不等于 CPU 开始调用到同步返回的完整墙钟延迟，也不保证包含全部 Python 验证和分配耗时。
- 计时前双方输出均与 FlagAttention 的 Torch 数学参考比较，阈值为 `atol=5e-3, rtol=1e-2`。

`--warmup`、`--rep` 的单位是 ms；`--iter` 或 `--repeat` 表示轮数；`--inner` 必须为 1。此前墙钟脚本的 `--warmup 10 --iter 25` 不再表示相同采样预算，复现当前默认协议无需传这些参数。

### 6.3 历史 autotune 消融：单独的墙钟证据

2026-10-13 的开关验证使用同步墙钟协议，按 off/on/off 顺序运行。关闭侧取前后两次各 shape 中位数的平均，开启侧在预热后测量，不含首次搜索耗时：

| Case | 固定 `(1,1)` ms | autotune ms | 开启后延迟增加 |
|---|---:|---:|---:|
| L2_N1_D7168_ON0 | 0.180749 | 0.210621 | 16.53% |
| L2_N1_D7168_ON1 | 0.187422 | 0.207592 | 10.76% |
| L5_N128_D7168_ON0 | 0.338429 | 0.359643 | 6.27% |
| L5_N128_D7168_ON1 | 0.385173 | 0.397452 | 3.19% |
| L5_N1024_D7168_ON0 | 1.283005 | 1.302730 | 1.54% |
| L5_N1024_D7168_ON1 | 1.585265 | 1.602279 | 1.07% |
| L9_N128_D7168_ON0 | 0.449374 | 0.459071 | 2.16% |
| L9_N128_D7168_ON1 | 0.490593 | 0.507570 | 3.46% |
| L9_N8192_D7168_ON0 | 14.579883 | 14.632784 | 0.36% |
| L9_N8192_D7168_ON1 | 16.951442 | 17.008927 | 0.34% |

该轮 7 项正确性和 10 个 benchmark case 通过。全部实测墙钟中位数在关闭时较低，但大 shape 的约 0.3% 差距不能当作对所有输入都成立的显著优势。首次候选编译/搜索还有额外时间，不计入本表。

不能把本表与第 6.1 节 Event 数字相减来计算优化收益，也不能用它代替当前 Event 协议下的新一轮 autotune 消融。

## 7. KDA 候选流程与未保留的方案

2026-10-13 KDA 先保存原始四-warp baseline，再逐个验证候选。正式目标比较对象为原始 Triton，目标是任意 shape 延迟降低至少 10%，同时所有 shape 不超过 2% 回退。

| 候选 | 改动 | 验证与决定 |
|---|---|---|
| c001 | 全局改为两 warp | 正确性通过；两轮完整性能测量均有单行回退超限，撤回 |
| c002 | 全局改为一 warp，保留单配置 autotuner | 正确性通过；输出归一化单行用例回退 2.84%，撤回 |
| c003 | 一 warp、一 stage，并绕过固定配置 autotuner | 7 项正确性通过；3 轮完整 benchmark 均保留，以各 shape 三轮中位数验收；保存并恢复为最好正确版本 |

最终 c003 相对原始 Triton：八个批量用例延迟降低约 51%～73%，最大 shape 比值为 0.993318，满足原合同的目标与回退保护。该数据采用旧同步墙钟协议，与第 6.1 节分开解释。

本轮没有新增精确 benchmark shape 分派，没有改数学公式、TLE 加载或残差指针选择。原始和最好版本均已保存，历史恢复后再次通过 7 项正确性测试。

## 8. 覆盖范围、瓶颈与后续方向

已使用的分支包括 TLE 同步/异步加载、输出归一化开关及返回/不返回权重。仍有未充分覆盖的范围：

- 正式性能表只测 BF16、`D=7168`，不证明 FP16 或其他 D 的性能。
- `return_weights=True` 由正确性测试覆盖，但不在正式性能表中。
- 当前 test 与正式 benchmark 最大源数为 9，10～16 源尚无该组性能证据。
- Triton 的非连续输入复制路径尚未由当前用例验证。
- 固定 `(1,1)` 是已测用例的默认方案，不保证任意新 shape 最优；autotune 的四组候选也不保证覆盖全局最优。

没有匹配的 GCU profiler 证据，当前结论来自源码和 benchmark。后续可以优先验证更多 D/L/N、FP16 与返回权重场景，再考虑离线调优并直接选择启动参数。所有配置规则必须验证完整 API 延迟，避免 kernel 的局部收益被主机调度成本抵消。

## 9. 测试与复现

当前默认 benchmark：

```bash
cd /workspace/BaselineBenchmark
pytest backends/_enflame/benchmarks/bench_attnres_torch.py -vs
```

Torch baseline 独立测试：

```bash
cd /workspace/BaselineBenchmark
pytest backends/_enflame/test/test_attnres_torch.py -vs
```

Triton 正确性与自身 benchmark：

```bash
cd /workspace/FlagAttention
pytest tests/flag_attn/test_attnres.py -vs
pytest benchmark/test_fused_attnres_benchmark.py -vs
```

开启 autotune 对比：

```bash
cd /workspace/BaselineBenchmark
FLAG_ATTN_ATTNRES_AUTOTUNE=1 pytest backends/_enflame/benchmarks/bench_attnres_torch.py -vs
```

未设置环境变量时默认关闭；若终端此前 export 了 `1`，应在命令前显式设置 `FLAG_ATTN_ATTNRES_AUTOTUNE=0`。历史 FLA 对比仅支持 CUDA，所以 Triton 自身 benchmark 在 GCU 上显示 `FLA=N/A`；Torch speedup 使用 BaselineBenchmark 入口。

Triton 历史验证为 `7 passed`；其 test 覆盖返回概率、BF16/FP16、D=7168/4096、1/5/9 源、多维输入和输出归一化。Torch 独立测试声明 29 项（19 CPU、10 GCU），使用独立 CPU FP64 定义式；本地曾直接运行通过 19 项 CPU 检查，不能表述为 GCU 测试或完整 pytest 会话通过。当前计时改版已完成本地语法与协议逻辑检查，本次文档工作未重新运行 GCU。

## 10. 证据与可复用验收流程

历史 workspace：

- `E:/Zw/attnres_kda_s60_20261013/`：原始/最好 checkpoint、候选、完整测量、正确性与最终报告。
- `E:/Zw/attnres_autotune_verify_20261013/`：off/on/off 墙钟原始结果、每个 shape 的配置选择与最终报告。
- `E:/Zw/attnres_topscc_remove_20261013/`：TopsCC 移除记录；旧失败实现不作为活动候选恢复。

历史 KDA 原始 `fused.py` SHA256：

```text
2b09ac7443674f9063f20c0720fc94cf5a8498631df909781f2eb86714ad36ca
```

最终 c003 `fused.py` SHA256：

```text
45cce35411c07c7e436fe2ec77c246313c7bc0810a64a49c125ef5f0427a7812
```

后续优化按以下步骤验收：

1. 保存当前最好正确版本、原始 baseline 和 hash。
2. 固定数学参考、误差阈值、输入矩阵与计时协议。
3. 先优化并验证 Torch baseline，再测量原始 Triton，避免与弱基线比较。
4. 每次只修改一个候选，先正确性，再完整 benchmark；保留慢结果。
5. 编译/搜索时间与稳态延迟分开记录；同一张性能表不混用 Event 与墙钟。
6. 分别报告相对原始 Triton 的优化收益和相对 Torch 的 speedup。
7. 回滚失败候选，恢复并复测最好正确版本；新 shape 重新验证配置。

通用流程与模板参考 `../workflow/operator_optimization_workflow.yaml`。新的 KDA 远程任务继续通过受控接口执行；本次文档不触发远程同步或优化。
