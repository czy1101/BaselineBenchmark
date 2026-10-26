# MetaX 六算子成功与失败经验

整理日期：2026-09-07。本文是证据复盘，不是一次新的 GPU 验收。

## 范围与来源

对象为 GDN2、SageAttention、MiniMax Sparse Attention、KDA、GLA、NSA。
实验主要发生于 `czy1101/FlagAttention:dev-metax` 的 WZB/C550 环境；本仓库
`ops/` 中的独立基线有各自的来源和依赖，不能把适配实现的测试结果自动记到基线上。
统一测试入口仍为 [`../test/`](../test/README.md)，本次只增加文档和结构化记录。

FlashMLA、FlashMLA Sparse、FlashMLA KV-cache 已从当前 FlagAttention 交付树移除。
本次不删除 BaselineBenchmark 保留的历史实现、benchmark 或测试，也不重写历史提交。

## 结果总览

| 算子 | 可据实记录的成功 | 失败/限制与下一步 | 复盘 |
| --- | --- | --- | --- |
| GDN2 | 完整 6+48 项通过；热缓存 pytest 4.640 s、脚本 36.414 s；保留 12 Shape benchmark | 最新空缓存试跑在探测脚本中失败，尚无修复后冷启动结果，不能量化冷编译提速 | [GDN2](gdn2_experience.md) |
| SageAttention | 正式集成候选 42 通过、0 跳过；5 个 HND Shape 配对几何平均加速 1.9128×；原生可选后端已发布 | 只对已验证的原生输入域成立；NHD 等仍走 Triton，不能泛化为全部场景提速 | [SageAttention](sage_attention_experience.md) |
| MiniMax | 27 项中 14 通过、13 预期跳过；BF16 benchmark 6 prefill+8 decode 完成 | 跳过不算通过；无 vLLM 对照或跨容器性能结论 | [MiniMax](minimax_experience.md) |
| KDA | 6 个 Shape 获得绝对时延，benchmark 进程成功 | 前 3 个 Shape 标记 TARGETED_REVIEW；本证据集没有完整正确性用例计数 | [KDA](kda_experience.md) |
| GLA | BF16 的 8 Shape 前向、反向共 16 行 benchmark 完成 | 无本次独立基线加速比；本证据集没有完整正确性用例计数 | [GLA](gla_experience.md) |
| NSA | 两个前向算子、两种精度、7 Shape，共 28 行 benchmark 完成 | 不覆盖反向；本证据集没有完整正确性用例计数或具体失败内核实验 | [NSA](nsa_experience.md) |

“未记录”不表示算子失败，也不表示从未测试；表示当前可追溯材料不足以确认。
因此不能据此笼统宣布“六算子的所有测试均已通过”。

## 已发布源码与证据边界

- GDN2 完整覆盖和 tuning/cache 改动：FlagAttention
  [2238c36](https://github.com/czy1101/FlagAttention/commit/2238c36ed139bfb36bc0b09cffb62417e8af6cd9)。
- SageAttention 可选原生集成：FlagAttention
  [52f7c60](https://github.com/czy1101/FlagAttention/commit/52f7c6096bd6c433ee51d142e7526185fb9d4e25)，
  为 `2238c36` 的后继，9 文件变更；整理时远端 `dev-metax` 指向该提交。
- 本文档基于 BaselineBenchmark
  [7940d62](https://github.com/czy1101/BaselineBenchmark/commit/7940d623c346650ca1f10c1de6fafea29d834615)，
  保留此前统一到 `test/` 的目录结果。

GPU 数字来自用户返回的 C550 执行日志，不是文档整理机重新运行所得。
证据原件留在 WZB 归档目录；此处保存定位信息、日志中报告的摘要及其限制，不上传缓存、
原生二进制、Token 或完整终端输出。日志报告的 SHA256 不等于本次已下载原件重新校验。

查阅 [证据索引](../workflow/runs/20260907/evidence_index.json)、
[实验决策记录](../workflow/runs/20260907/experiment_record.json) 和
[环境/工作流复盘](environment_and_workflow.md)。
