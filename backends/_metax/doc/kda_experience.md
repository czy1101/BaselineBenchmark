# KDA：区分计时稳定性、算子结果与外层脚本状态

证据 ID：`kda_benchmark`、`kda_wrapper_cleanup`，见
[索引](../workflow/runs/20260907/evidence_index.json)。

## 成功记录

WZB/C550 的 `public_auto` 前向绝对性能，BF16、B=1、H=96、K=V=128、initial=None。
使用设备事件稳态摊销计时：warmup=10、blocks=21、target_block_ms=40、max_inner=64，
seed=20260828。结果来自适配实现，不是 `mcoplib._C` 独立基线。

| T | Event p50 ms | 状态 |
| ---: | ---: | --- |
| 256 | 0.447053 | TARGETED_REVIEW |
| 1024 | 1.130820 | TARGETED_REVIEW |
| 2048 | 2.858277 | TARGETED_REVIEW |
| 4096 | 6.437998 | ROBUST_STABLE |
| 6144 | 9.460429 | ROBUST_STABLE |
| 8192 | 12.606784 | ROBUST_STABLE |

benchmark 进程 rc=0、墙钟 140 s，整体 `COMPLETE_WITH_REVIEW_SHAPES`。
可确认 6 个 Shape 完成测量，不能写成全部稳定通过。复核项表示计时质量需继续检查，
不是已证实 correctness 或内核性能回退。

## 失败与处理

结果生成后外层 gate 返回 rc=18。只读诊断发现唯一变化是未跟踪的零字节文件 `9`，
tracked/staged 均无变动。核验精确路径和大小后清理该文件，工作树恢复干净，
原 benchmark 结果仍有效。不应因外层 rc 覆盖已有成功测量，也不能不查原因就忽略 rc。
早先 preflight rc=10 没有足够细节，此处不推断根因。

经验：分别保留 benchmark_rc、结果质量状态、仓库完整性状态；Shell 文件描述符/锁重定向
和日志输出不要混写，避免数字被当作文件名。只清理已验证的意外文件，不使用宽泛清理命令。

## 待办

对 T=256/1024/2048 在设备空闲、相同 timer 条件下做针对性稳定性复核，保留样本、CI 和漂移。
当前材料没有完整 KDA correctness 的 JUnit 计数或本轮独立基线对照，不能由 benchmark
成功反推它们已通过。独立基线应按 [`../test/kda/`](../test/kda/) 的依赖和来源单独验收。
