# 环境、测量和提交工作流复盘

## 环境失败不等于数值错误

WZB 的已验证组合为 C550、MetaX Torch 2.8.0+metax3.7.2.0、FlagTree/Triton 3.6.0。
同事 CYC 容器使用定制 FlagTree/TLE；即使 Torch 版本和设备名相同，Python 层、
IR builder 与编译后端仍可能不匹配。

| 现象 | 发生阶段 | 已知原因/处理 |
| --- | --- | --- |
| CYC GDN2 缺 `autotuning.adjust_block_size` | Python autotuner，非数值比较 | 定制运行时接口不一致；不能据此判为算子数学错误 |
| WZB MiniMax 缺 builder API | 路径选择/编译能力 | 同时检查 Python GPU API 与 builder，使用已支持路径 |
| Sage 缺 setuptools-scm | 包构建检查 | 隔离构建工具，不替换工作的 Torch/Triton |
| `ModuleNotFoundError: ...gdn2` | import | 核验实际 `__file__`；明确目标 checkout 的 src |
| `docker: command not found`、共享卷不存在 | 容器入口/文件传输 | docker 操作需在具备 CLI 的宿主入口执行；不能假设两容器共享路径 |
| 脚本不存在、heredoc 截断、mktemp 父目录不存在 | 启动前 | 先确认文件/父目录；上传完整脚本并校验，不让长命令承载整个流程 |

选择 WZB 作为集成环境是基于其既有可复现证据和尽量不扰动已验收算子，不是断言它天然
支持所有 CYC 路径。需记录 runtime 模块路径、版本/源码身份、设备、能力及路由；
`hasattr` 成功只是一层预检，不能替代编译和正确性验收。

## 测量分层

1. 源码/环境身份：目标 commit、tree、文件 hash、实际导入路径、dtype、layout。
2. 冷启动：新建且确认使用独立空缓存；保留旧缓存。分别计 import、调优、编译、测试、墙钟。
3. 热缓存：确认缓存命中与新搜索次数，再报告日常验收时间。
4. 算子稳态：预热后使用原 benchmark timer；固定测量范围、Shape、路由、输入。
5. 性能比较：同条件配对 ABBA/BAAB、多轮样本、离散度/漂移；筛查阈值与正式验收线分开。

单卡不并行开 6 个 benchmark。共用文件锁只能约束遵守同一锁的进程，不能证明整卡没有
其他任务。锁忙先只读检查持有者，不删锁文件或凭猜测终止进程。新消息/失败后保留父 shell，
不要在用户容器会话中执行 `exit`。

不能用进程耗时推导内核延迟；不能用热缓存提升替代冷编译提升；也不能将不同用例数量、
不同布局或不同计时范围的结果直接比较。test 调用 TLE 时可能还计算 Native oracle，
应按阶段定位，不按测试名称推断全部耗时。

## 门禁与状态

明确区分 `prepared`、`validated`、`applied`、`committed`、`published`。
没有发生 GPU 测试的静态 PASS 不算 correctness；skip 不算 pass；外层仓库检查失败不应
抹掉真实 benchmark 数值，但在修复/解释前也不能宣布整体门禁通过。

精确保护需要正确依据：优先冻结 manifest 和 AST，不用有歧义的子串搜索或过期 SHA。
记录完整 stdout/stderr 与 rc，避免只看到“git diff failed”而漏掉 EOF 空行等真实诊断。
仅格式/注释修复若 AST、位置和其他源文件保持一致，可复用准确绑定的既有 GPU 证据，
但不要冒充重新执行了测试。

## 正确提交经验

- 失败实验记录现象、阶段、证据、撤回原因；不提交损坏候选、缓存、日志海量输出或凭证。
- 本次是 BaselineBenchmark 文档提交；不顺手移植 FlagAttention 源码，不动已经统一的 `test/`。
- 提交前精确文件清单、`git diff --check`、JSON/链接检查；未授权内容保持原状。
- Git 网络不通时可使用 GitHub Git Data API；固定仓库/分支，检查 base 和最终 tree，
  仅非强制更新 ref。更新前再读远端；远端变化则停，不覆盖同事提交。
- API 超时可能发生在服务器成功写入之后，先读回 ref/commit/tree，再决定是否已成功；
  不盲目重发 PATCH。对象创建成功不等于分支已经更新。
- Token 仅在交互终端隐藏提示中输入，不写进聊天、命令参数、脚本、归档或提交。

这些要求可通过[实验模板和检查清单](../workflow/README.md)复用。
