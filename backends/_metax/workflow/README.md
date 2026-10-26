# MetaX 实验沉淀与提交约定

一次实验记录一个可判定假设；成功与失败都保留，失败代码可以撤回，原因不能丢失。
文档目录按算子组织，原始输出保存在实验归档/仓库外目录，不把构建缓存或二进制加入 Git。

## 本次归档

- [2026-09-07 证据索引](runs/20260907/evidence_index.json)：源码提交、结果定位、报告 hash、限制。
- [实验记录](runs/20260907/experiment_record.json)：成功方案、撤回/失败尝试及待办。
- [六算子复盘](../doc/README.md)：便于评审阅读的结论和性能表。

`archive_root` 是用户 WZB 容器中的历史归档位置，不是仓库相对路径，也不保证当前机器可访问。
`reported_sha256` 来自返回日志；`artifact_bytes_reverified=false` 表示本次没有重新读取
原始 result/log 文件核验该 hash。远端 commit 是代码身份，不证明 GPU 日志本身完整。

## 记录模板

复制 [experiment_record.template.json](experiment_record.template.json)，填写字段：

- `decision`：`accepted`（在明确范围采用）、`rejected`（不采用）、`pending`（缺证据）。
- `stage`：`static`、`build`、`correctness`、`benchmark`、`integration`、`workflow`。
- `evidence_ids` 必须引用同批次证据索引；未知指标填 `null`，不能填 0 或 PASS。
- `observed_result` 是观察事实；`interpretation` 是有边界的解释；`next_action` 是尚未发生的动作。
- `source_revision` 固定完整 SHA；若日志没有绑定具体 revision，写明限制，不自行补造。

证据类别：`user_returned_log`（用户执行回执）、`remote_git_verified`（远端代码身份核查）、
`local_static_check`（本地静态/CPU 检查）。没有本次 GPU 执行证据就不标为新的 GPU 验收。
原始失败和后续成功使用不同记录或注明明确阶段，不能用最后一次 PASS 覆盖历史。

## 提交检查清单

1. 固定设备、容器、导入路径、源码 commit/tree；区分适配实现与独立基线。
2. 给每次尝试记录假设、修改范围、输入矩阵、计时方法、缓存状态、比较对象和阈值来源。
3. correctness 记录 collected/passed/failed/errors/skipped；性能记录原始样本/摘要及测量范围。
4. 保存失败阶段和错误摘要；未执行的阶段显式为未执行，而不是失败或通过。
5. 冷/热缓存及编译/稳态分别报告；有比较证据才写“提升了多少”。
6. 源码、包内容、运行时路由、正确性、性能分别验收；记录尚未满足的输入域或依赖。
7. 只提交评审范围内的文档/源码。先查 diff 和工作树，不混入其他会话改动。
8. 分支更新使用快进；更新前检查远端，之后验证 ref、父提交及最终 tree。失败/超时先读回。

当前批次仅沉淀经验，不执行上述未来 GPU 任务；测试和 benchmark 文件字节不变。
