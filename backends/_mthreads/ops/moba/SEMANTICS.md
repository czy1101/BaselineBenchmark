# MoBA 语义审计（2026-10-26）

## 结论

在已检查的输入和支持范围内，两版实现均保持 MoBA 的算法语义。MATE 版复用的是 FMHA 内核结构，未替换 MoBA 路由，也未把逐 query head 路由合并为共享路由。使用 Triton 或 C++/MUSA 不决定算法语义；实际的选块、掩码与归一化才决定计算内容。

这不是所有输入的形式化证明，也不表示与官方逐比特一致。下面区分算法合同、数值误差和官方实现自身的边界差异。本次未修改两版算子，也未修改 benchmark 的严格校验阈值。

## 官方依据

- [MoonshotAI/MoBA naive 参考](https://github.com/MoonshotAI/MoBA/blob/master/moba/moba_naive.py)
- [MoonshotAI/MoBA efficient 实现](https://github.com/MoonshotAI/MoBA/blob/master/moba/moba_efficient.py)
- [MooreThreads/MATE](https://github.com/MooreThreads/mate)：本地混合实现复用其 FMHA 模板，并非官方完整 MoBA。

两份官方源码已下载到外层 FlagAttention 工作区 `runs/moba-semantics/official_moba_naive.py` 和 `official_moba_efficient.py`，SHA256 记录于同目录 `audit.json`，避免后续 master 更新混淆。移除注释/docstring 后，本工作区 `src/flag_attn/FLA/moba/naive_moba.py` 与下载的官方 naive AST 完全相同。已有 forward/backward 测试的参考来源因此已核实。

## 核查项

| 项目 | 两版行为与检查结论 |
| --- | --- |
| 分块 | 各序列独立按 chunk_size 分块，处理尾块，不跨 batch。 |
| 路由 | 用块内 K 均值与 query 打分，每个 token、query head 独立选择。 |
| top-k | 当前块始终包含在预算中，额外选择至多 K-1 个历史块；不足时只选有效块。 |
| 因果性 | 当前块内部使用 token 级因果掩码；历史块只允许完整位于 query 之前的块，因此内部可使用非因果 FMHA。 |
| GQA/MQA | KV 可共享，每个 query head 的路由仍独立，映射为 kv_head = q_head // group_size；这是对官方同头数接口的扩展，参考通过重复 KV 头建立。 |
| 输出合并 | 使用各分支 LSE 加权，数学上等价于在所有选中 token 的并集上做一次 softmax，不是块输出简单平均。 |
| MATE 专用调度 | 每个 descriptor 对应一个块、query head 和 query tile；query tile=32 与上游描述符一致。无效槽位不参与合并。 |
| padding/尾块 | 最后一个块不作为历史候选；其 token 由局部因果分支处理。MATE 中填充的无效 KV 不构成有效历史路由。 |
| 反向 | 固定离散 top-k 路由，对注意力和 Q/K/V 求导；与官方对离散路由的处理一致。 |

两版 `routing.py`、`attention.py` 当前字节一致；`parallel.py` 的功能差异仅为历史稀疏 forward 函数替换为 native adapter。MATE 版的局部注意力、输出合并和反向仍使用 Triton。

## 实际验证

1. 重新执行原始正确性测试：纯 Triton 卡 6 为 **39 passed、6 skipped**，MATE 混合版卡 7 为 **39 passed、6 skipped**；覆盖前向、反向及已有边界输入。日志在 `runs/moba-semantics/pure-tests.log`、`mate-tests.log`。
2. BF16、B=2、S=4096、H=16、D=128、C=128，K=2/4/8：分别检查 276/273/272 个查询行，共 821 行，包括此前所有两版超差查询行。使用 CPU float64 直接计算块打分、选中 token 并集、softmax 和输出；块均值按官方输入 dtype 舍入。采样路由与参考 **零不一致**。
3. 针对共享 KV 的 query heads 构造相反方向、分数明确分离的查询，变长序列为 257/193，Hq=4/Hkv=2、D=64、C=64、K=4；全量 1800 查询行的路由匹配，确认共享 KV 不意味着共享路由。
4. 同时改变指定前缀之后的 Q/K/V 和后续 batch 数据，两版已检查前缀的输出变化均为 **0**，验证此构造下因果性和 batch 隔离。

### 之前 topk=2/4 的严格比较失败

| top-k | 两版超差查询行数 | Triton 对 float64 最大/平均绝对误差 | MATE 对 float64 最大/平均绝对误差 |
| --- | ---: | --- | --- |
| 2 | 4 | 0.0062501 / 0.00023851 | 0.0062501 / 0.00023970 |
| 4 | 1 | 0.0038603 / 0.00017236 | 0.0040312 / 0.00017454 |
| 8 | 0 | 0.0038603 / 0.00013324 | 0.0038603 / 0.00013370 |

两版都通过原 correctness 测试使用的 max_abs<0.04、mean_abs<0.0004 标准。采样中没有发现错路由或错误可见 token 集合，结果支持误差来自低精度计算/舍入，不能据此声称算法已变成另一种注意力。

但 benchmark 的逐元素阈值为 atol=0.004、rtol=0.001，这比原 correctness 标准严格，两版之间 K=2/4 的失败仍成立。**没有放宽或隐藏失败，也没有将数值差异证明为零。** 后续针对性诊断已定位到分块顺序与中间概率 BF16 量化的组合，见下文。

## 不能声称完全等同官方的边界

- 官方 efficient 文件在 `moba_topk-1<=0` 分支直接执行整序列因果注意力；官方 naive 的 K=1 则仅允许当前块。我们两版采用 naive 的当前块语义。因此 K=1 多块场景不与该 efficient 特殊分支一致，虽然符合所采用的 MoBA 选块定义。该参数已按用户要求移出性能集合，但仍属于语义说明范围。
- 路由分数相等时，本实现用较小块索引优先；官方 `torch.topk` 未固定同分选择顺序。两者都可返回有效 top-k，但选中集合和输出未必相同。近似相等时，浮点求和顺序也可能改变名次。此次随机参考采样的路由无不一致，不能外推为所有临界同分输入都完全相同。
- 本实现限制 head_dim<=256、topk<=64、FP16/BF16；GQA/MQA 是扩展。MATE 原生路径只对已验证版本、MP31、D=64/128 开启，其余回退 Triton。
- 输入需满足有效的 cu_seqlens 和 max_seqlen 合同；推理路径为避免同步，不逐项在 CPU 验证这些输入值。
- 本次未穷尽所有 shape、超长序列、所有浮点临界值或模型训练质量；不能将有限验证称为全域保证。

复现脚本：外层工作区 `runs/moba-semantics/audit.py`；结果与源文件哈希：`audit.json`。在 lgh_flagtree 中使用已有卡 6 环境、MATE_WORKSPACE_BASE 和缓存设置运行，所有产物留在 FlagAttention 内。

## 针对性失败定位（2026-10-26 补充）

捕获 K=2 的 4 个失败元素及 K=4 的 1 个失败元素，对照局部注意力、历史块输出、LSE、合并结果，并用 CPU float64 重建：

- 两版当前块输出及 LSE 逐比特相同；选中的历史块一致。
- Triton 历史注意力以 BLOCK_N=32 正序在线 softmax；当前 MATE 模板以 tile_n=64 逆序在线 softmax。两者在 PV 矩阵乘之前都把未归一化概率转换为输入类型（这里是 BF16）。
- 在线 softmax 的运行最大值随遍历顺序变化，量化前概率的缩放因此不同。BF16 中间量化不可逆，即使最终 LSE 相同，PV 累积也可不同；历史块输出落在相邻 BF16 值，最终合并再次舍入，形成严格阈值失败。
- 受控 CPU 模拟分别使用 32-token 正序和 64-token 逆序，并保留 BF16 概率量化，**复现了全部 5 个失败元素所涉及的两版历史块 BF16 输出**。保留高精度概率时，两种遍历都回到同一个 float64 注意力结果（浮点尾数误差范围内）。这比单纯猜测“tile 不同”多了一步因果验证。
- 从实际历史输出/LSE 重建合并，复现全部最终失败值。只交换两版 LSE，不改变这些失败点的最终 BF16 值；交换历史块输出则复现另一版结果。

例：token=1889、head=1、dimension=99，历史块 6：

| 阶段 | Triton | MATE |
| --- | ---: | ---: |
| CPU 模拟，历史输出舍入前 | 1.2617387851 | 1.2616554406 |
| 实际历史块 BF16 输出 | 1.265625 | 1.2578125 |
| 实际历史 LSE | 6.9029369354 | 6.9029369354 |
| K=4 最终 BF16 输出 | 0.69921875 | 0.69140625 |

该历史注意力 float64 值为 1.2617445272，两个相邻 BF16 值的中点为 1.26171875。中间量化把结果推到了舍入边界两侧。最终差 0.0078125，超过当前比较阈值。并非路由/掩码错误，也不能据此把任一版本视为逐元素金标准。

复现：`runs/moba-semantics/diagnose.py`；完整中间值：`diagnose.json`。模拟用于隔离数值机制，不是逐条硬件指令模拟。本次没有修改计算内核或放宽比较阈值；如需消除差异，应另行评估更高精度中间结果或统一数值路径及其性能成本。

## topk=1 是否需要补逻辑

当前两版已有完整分支：历史路由预算为 0，返回当前块内部的 causal attention；训练和推理均如此。以官方 naive 的 MoBA 定义为准，无需添加“全序列 causal attention”分支，后者会让第 2 块以后的查询看到预算外的历史块。与官方 efficient 的该特殊分支逐行为兼容是另一种目标，不应默默改变现有语义。

新增专门回归测试 `BaselineBenchmark/backends/_mthreads/tests/test_moba_topk.py`：两版 × FP16/BF16 × 推理/训练入口，共 **8 passed**。使用全零 Q/K、各块不同的常数 V，变长多序列及 GQA，输出必须精确等于当前块常数，且显著区别于全序列因果注意力。日志为 `runs/moba-semantics/topk1.log`。这里的训练检查针对 autograd 入口选择，不替代此前已通过的完整梯度测试。

截至 2026-10-30，活动性能测试集合已加入 topk=1、2、64 边界用例；语义回归测试仍独立保留。
