# MoBA 实验基线（摩尔线程）

## 定位与来源

本目录提供用于性能研究的 MoBA 实验基线。它由项目自行组合 **MATE/MUSA 与 Triton** 实现，并非摩尔线程官方发布的完整 MoBA 算子，也不直接调用 `mate.msa_interface.sparse_msa`。正式提交的纯 Triton 算子与此实验基线相互独立。

| 上游项目 | 用途 |
| --- | --- |
| [MooreThreads/MATE](https://github.com/MooreThreads/mate) | 提供 FMHA 模板、JIT 构建能力以及 TME/SQMMA 内核结构。 |
| [MoonshotAI/MoBA](https://github.com/MoonshotAI/MoBA) | 提供 MoBA 算法和路由语义参考；本实现不调用其 CUDA 算子。 |
| [FlagAttention](https://github.com/flagos-ai/FlagAttention) 与 [MoBA PR #71](https://github.com/flagos-ai/FlagAttention/pull/71) | 提供算子集成背景和纯 Triton 实现的对照。 |

MATE 生成的内核使用 `mutlass`、`mute` 等类型和硬件抽象；本目录不复制这些依赖的源码。基线使用 MATE 的内部模板接口，升级 MATE 后须重新验证兼容性。

## 实现组成

| 文件 | 职责 |
| --- | --- |
| `parallel.py` | 编排路由、注意力、输出合并及自动求导。 |
| `routing.py` | 用 Triton 计算块均值、逐 query head 的 top-k 路由和数据整理。 |
| `native_moba.py` | 从 MATE FMHA 模板生成 C++/MUSA 前向内核，加入 MoBA 专用调度器，并完成 JIT 加载。 |
| `attention.py` | 实现 Triton 当前块因果注意力、稀疏回退、LSE 合并和反向。 |
| `benchmark.py` | 提供独立的推理性能对照入口。 |
| `__init__.py` | 导出本目录的 `parallel_moba`。 |

前向流程为：计算块级路由 → 按历史块和 query head 准备输入 → 执行历史块稀疏注意力及当前块因果注意力 → 使用 LSE 合并输出。反向使用 Triton。GQA/MQA 可以共享 KV head，但每个 query head 仍独立选择历史块。

原生前向内核以 MATE FMHA 模板为基础，使用自定义工作调度器；每个工作描述符对应一个块及一个 query head 的 query tile。历史块注意力的热点会编译为 C++/MUSA 内核，因此本基线不是纯 Triton 实现。

## 已验证配置与适用范围

- 已验证设备型号：**MTT S5000（MP31）**。
- 主要软件环境：Python 3.10.12、PyTorch 2.7.1、torch_musa 2.7.1、Triton 3.6.0、MATE `0.2.0+mu437torch2.7`。Triton 来自当前 FlagTree 开发环境；未确认独立的 FlagTree 发行版本，因此不另填版本号。
- 原生稀疏路径支持 FP16/BF16，以及 head dimension 64/128。
- 不符合原生路径条件的输入使用 Triton 回退；当前块因果注意力及反向也使用 Triton。
- 首次调用包含 JIT 编译，不能将编译时间计入稳态推理延迟。
- 保留 FP16 合并路径的正确性保护，排除已知会丢失输出列的配置。

## 性能评估

性能对照脚本默认运行 **40 个去重用例**，覆盖序列长度、batch 分布、head dimension、chunk size、top-k、MHA/GQA/MQA、FP16/BF16、变长序列及尾块。用例用于比较完整推理调用，而非单个内核。计时包含调度、分配、路由、输入整理、注意力、合并与同步；排除输入生成、首次编译和热身。

脚本先完成初始化与热身，再交替测量两个实现：每例 4 轮、每轮每版 25 次。结果报告中 `Speedup = Baseline ms / FlagAttention ms`；大于 1 表示 FlagAttention 更快。性能测试本身不检查数值正确性。

**比较 MATE 原生路径时，应只统计标记为 MATE 的用例。** D96/D256 会走基线的 Triton 回退；top-k=1 只执行当前块注意力。不能把这几类结果解释为与 MATE 原生稀疏内核的比较。不同版本或测量设置的历史结果也不应直接合并。

正确性验证应独立覆盖前向、反向、top-k 边界、逐头路由、变长序列和因果边界。算法合同、数值差异及官方实现边界另见 `SEMANTICS.md`。
