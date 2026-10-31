# MoBA 高性能实验基线（摩尔线程）

## 定位与来源

本目录保存本项目自行实现的 **MATE/MUSA + Triton 混合 MoBA baseline**，由 FlagAttention 中原来的 `src/flag_attn/runtime/backend/_mthreads/FLA/moba-baseline/` 迁移而来。
它不是摩尔线程官方发布的完整 MoBA 算子，也不是直接调用 `mate.msa_interface.sparse_msa`。我们复用 MATE 的 FMHA 内核模板，加入 MoBA 专用调度器，保留每个 query head 独立选择历史块的语义。

本目录属于 BaselineBenchmark，用于记录、实验和性能对照；不注册到 FlagAttention 公共 API，不属于 FlagAttention 的纯 Triton 提交。纯 Triton 版本仍位于 `src/flag_attn/runtime/backend/_mthreads/FLA/moba/`，公开入口为 `flag_attn.parallel_moba`。

## 上游仓库与各自作用

| 来源 | 在本实现中的作用 |
| --- | --- |
| [MooreThreads/MATE](https://github.com/MooreThreads/mate) | 实际运行依赖。复用 FMHA C++/MUSA 模板、JIT 构建和 TME/SQMMA 计算结构。 |
| [MoonshotAI/MoBA](https://github.com/MoonshotAI/MoBA) | MoBA 算法及路由语义参考；本实现不调用其 CUDA 算子。 |
| [FlagAttention](https://github.com/flagos-ai/FlagAttention)、[MoBA PR #71](https://github.com/flagos-ai/FlagAttention/pull/71) | MoBA 实现背景。本目录的 Triton 支持代码来自本工作区的 MThreads 适配与优化版本，并非未经修改的上游快照。 |

MATE 生成代码使用 `mutlass` / `mute` 类型与硬件抽象；相关头文件由当前 MATE 安装及其构建依赖提供。本目录没有另行复制这些官方库的源码，也不修改安装中的 MATE 文件。

## 实现组成

| 文件 | 内容 |
| --- | --- |
| `__init__.py` | 导出本目录独立的 `parallel_moba`。 |
| `parallel.py` | Python/PyTorch 编排，组合路由、历史块稀疏注意力、当前块因果注意力、输出合并和 autograd。 |
| `routing.py` | Triton 块均值、路由打分与 top-k、路由描述符、数据整理；GQA 下每个 query head 独立路由。 |
| `native_moba.py` | Python JIT 桥接、内嵌 C++ `MoBARouteScheduler`，以及 Triton Q/K/V 打包内核；生成并编译 `.mu` 内核。 |
| `attention.py` | Triton 局部因果注意力、稀疏回退实现、softmax/LSE 合并及反向。 |
| `benchmark.py` | 独立比较本目录与 FlagAttention 纯 Triton 入口，不修改公共注册。 |

执行流程：块级路由 → 按历史块与 query head 整理 Q/K/V → MATE 原生稀疏前向和 Triton 当前块因果前向 → 通过 LSE 合并输出。反向使用 Triton。

`native_moba.py` 从 `mate.jit.attention.fmha.fmha_fwd._render_fmha_fwd_source` 生成源码，用自定义调度器替换默认调度器，再通过 `mate.jit.core.gen_jit_spec` 编译加载。每个工作描述符对应一个块与 query head 的 query tile，不把 GQA 多个 query head 的路由合并。

硬件结构复用 MATE 的 TME 数据搬运和 SQMMA 矩阵计算；当前配置为 query tile 32、key tile 64，K/V stage 各 1，QK/PV consumer 各 1。无 split 的专用路径移除了无用的部分结果分配。Python 文件因此不代表纯 Python/Triton：实际热点会编译成 C++/MUSA 内核。

## 已验证环境和边界

- 容器：`lgh_flagtree`；设备：本机 S5000（MP31），使用物理卡 6 或 7。
- 原生路径要求 MATE 包版本精确为 `0.2.0+mu437torch2.7`，dtype 为 FP16/BF16，head dimension 为 64/128。
- 不满足原生路径条件时使用本目录的 Triton 回退。因果局部路径和反向仍为 Triton。
- 使用了 MATE 私有模板 API，升级 MATE 后需要重新适配与验证；这里没有固定可追溯的 MATE 上游 commit，不能将包版本视为 Git commit。
- 保留 FP16 merge 正确性修复：排除已验证会丢失输出列的特定 autotune 配置。

## 调用与测试

以下命令均在 `lgh_flagtree` 容器内执行，使用已有环境，无需安装或修改宿主机：

```bash
cd /workspace/czy/FlagAttention
export PYTHONDONTWRITEBYTECODE=1
export MUSA_VISIBLE_DEVICES=6
export MATE_WORKSPACE_BASE=$PWD
export TMPDIR=$PWD/.cache/tmp
export XDG_CACHE_HOME=$PWD/.cache
export TRITON_CACHE_DIR=$PWD/.cache/triton-moba-native-route6
export PYTHONPATH=$PWD/src:$PWD/BaselineBenchmark:$PWD

# 原有 MoBA 正确性测试，测试进程内选择本目录实现
python runs/moba-layout/validate.py baseline

# 热身后的端到端推理对比
python BaselineBenchmark/backends/_mthreads/benchmarks/moba_compare.py --card 6
```

直接导入算子（已设置上述 PYTHONPATH）：

```python
from backends._mthreads.ops.moba import parallel_moba
# q: [total_tokens, query_heads, head_dim]
# k/v: [total_tokens, kv_heads, head_dim]
# cu_seqlens: 设备上的 int32 序列边界
out = parallel_moba(q, k, v, cu_seqlens, max_seqlen, chunk_size, topk)
```

推理应在 `torch.inference_mode()` 或 `torch.no_grad()` 下调用。首次调用包含 JIT 编译，不能当作稳态延迟。正确性测试脚本依赖外层 FlagAttention 工作区及其原有测试，并非 BaselineBenchmark 独立安装测试。

`benchmark.py` 使用 BF16、B=2、Hq=Hkv=16、D=128、chunk_size=128、topk=8，序列长度为 4096/8192；4 轮交替测量两个实现，warmup=25、rep=150，输出各轮中位延迟，单位毫秒。计时包含路由、打包、注意力与合并，不包含首次编译。

## 历史验证记录

迁移前两版原有正确性测试均为 39 passed、6 skipped，未放宽原有误差阈值。
此前上述 BF16 配置的端到端推理记录如下；“纯 Triton”是本工作区的 MThreads 适配版本，不是官方 MATE MoBA：

| 卡 | 序列长度 | 纯 Triton ms | 本混合实现 ms | 加速倍数 |
| --- | ---: | ---: | ---: | ---: |
| 6 | 4096 | 6.1757 | 4.2101 | 1.467 |
| 6 | 8192 | 13.3033 | 9.1605 | 1.452 |
| 7 | 4096 | 6.1379 | 4.1878 | 1.466 |
| 7 | 8192 | 13.0636 | 9.0639 | 1.441 |

原始记录位于外层工作区 `runs/moba-native/REPORT.md`、`paired-card6.json`、`paired-card7.json`。这些是迁移前的历史测量，不代表本次迁移重新测得的性能，也不能外推为反向速度提升。

源码保留原有版权声明。生成的编译产物、缓存和测量输出不应纳入算子源码。

## 性能对比

入口为 `BaselineBenchmark/backends/_mthreads/benchmarks/moba_compare.py`。每次运行固定执行全部 **40 个去重用例**，无需选择 suite 或 case。覆盖长度、batch/token 分布、head dim、chunk/top-k、MHA/GQA/MQA、FP16/BF16、变长序列和尾块；包括 4K～64K 长度扫描、32K/64K GQA、chunk=512/1024 的代表性组合，以及 topk=1/2/64 边界。

40 组不是把参数做完整笛卡尔积：每组覆盖一个性能维度或关键交互点。与主集重叠的长度扫描配置会去重；128K/256K/512K 压力点和重复的长序列 chunk/top-k 组合已移除，保留 32K/64K GQA，以及三个能区分 chunk 和 top-k 影响的长序列组合。默认运行时间仍较长，尤其是 topk=64 和长序列大 chunk 组合。所有结果都报告实际 latency，不能只用单个 shape 的结果代表整体。

其中 D=96/256 不满足 MATE 原生路径，保留作 Triton 回退对照；判断 FlagAttention 是否追上 MATE 时应看表格的 `Base path`，只将 `MATE` 行作为原生内核比较。表格的 `Speedup = Baseline ms / FlagAttn ms`，大于 1 表示 FlagAttention 更快，小于 1 表示仍慢于 baseline。

脚本只做性能测试，不执行精度比较。每例先在计时区外触发初始化/JIT，再热身 10 次，每轮测量 25 次，共 4 轮并交替实现顺序。计时包括 Python 调度、分配、路由、打包、attention、merge 和同步；排除输入生成与编译。结果写入 `BaselineBenchmark/results/` JSON，终端最后输出对齐表格。`Run` 列已移除，执行错误的计时显示 `--` 并在表格后报告。

从容器项目目录运行全部用例：

```bash
cd /workspace/czy/FlagAttention
python BaselineBenchmark/backends/_mthreads/benchmarks/moba_compare.py --card 6
```

`--card` 只选择允许使用的设备；不提供用例选择参数，每次都会执行完整 40 组。需要减少耗时时，可编辑脚本中的固定 case 列表后再运行。

### Shape 清单

- 5 组基准长度、2 组 batch、3 组 head dim、3 组 chunk size、3 组 top-k。
- 3 组头布局、3 组 FP16、5 组变长/边界输入。
- 2 组 32K/64K 长序列 GQA，3 组长序列 chunk/top-k 组合。
- 3 组固定 16K token 的 batch/变长分布。
- 2 组去重后新增的长序列 MHA 长度扫描（32K、64K）。
- 3 组 topk=1/64 边界。

固定 token 数组还与现有 length-8192（B=2）和 batch-4（4×4096）构成五种分布。它们的历史可见块数不同，因此代表不同工作负载，不是等 FLOPs 微基准。

## Correctness 与语义审计

性能脚本不校验输出。独立正确性测试包括原有 forward/backward 测试，以及 `BaselineBenchmark/backends/_mthreads/tests/test_moba_topk.py` 的 K=1/2/64 专项覆盖。详细算法对照、数值误差原因和官方实现边界见 [MoBA 语义审计](SEMANTICS.md)。之前 topk=2/4 的精度失败属于历史诊断记录，不能作为本版 benchmark 的性能过滤条件。

## 历史测量提示

`results/` 下旧 JSON 可能来自不同 schema、卡或计时方法；比较时以 JSON 的 source hashes、设备、参数和 speedup 定义为准。旧版字段方向不同的记录应直接按 latency 重新计算。历史小规模性能结果不能外推到当前 40 组 shape，也不是本次改动的重测结果。
