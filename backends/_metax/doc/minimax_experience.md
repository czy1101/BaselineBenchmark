# MiniMax Sparse Attention：能力分流与 benchmark 兼容性

证据 ID：`minimax_correctness`、`minimax_benchmark`、`minimax_commits`，见
[索引](../workflow/runs/20260907/evidence_index.json)。

## 成功方案

WZB 的 TLE Python 层有部分 API，不代表底层 IR builder 已具备异步 MMA 路径所需能力。
最终同时检测 GPU API 和 builder API，在 WZB 得到 `_HAS_TLE_SYNC=True`、
`_HAS_TLE_ASYNC_MMA=False`，选择可用同步路径。同步分配显式使用
`nv_mma_shared_layout=False`，没有整体替换 FlagTree/TLE。

早期曾尝试给 6 个 alloc 全部增加 generic layout 参数；这不是最终方案。
它不足以补齐 async 路径的 memdesc/slot 等能力。最终提交只保留同步路径所需覆盖和
异步能力检测，不应将旧六处 patch 当成当前源码。

正确性文件：27 项收集，14 通过、13 预期跳过，0 failure/error，125.26 s。
该回执没有完整的 13 项 skip reason 清单，因此这里不臆测全部由 FP8 或 vLLM 导致。
验收其他能力前需补齐逐项 skip 原因并单独测试，不能宣称 27/27 通过。

## benchmark 修复链

| 报错位置 | 原因 | 最终处理 |
| --- | --- | --- |
| `Expected 1 values, got (B,T,Hkv,Hq)` | 一个 x_name 却传四个坐标 | 改为 batch、seq_len、num_kv_heads、num_heads 四个数值轴，同步函数签名 |
| `_CachedPlatform` 缺 `attention_launch_kwargs` | wrapper 仅缓存 PDL，遮蔽其他平台 API | 保存原平台并用 `__getattr__` 转发未知属性 |
| `float()` 收到 tuple，matplotlib 失败 | 中间方案 `[(shape,)]` 只修好了入参，没解决绘图轴 | 放弃 tuple 轴，保留四个数值轴 |

这三项是 benchmark 驱动/展示层问题，不等于已通过的算子 correctness 失效。
先做 1 prefill+1 decode smoke，再跑完整文件；修复仅提交到 benchmark 文件。

## 当前 BF16 自测性能

当前只有 `flag_attn` provider；vLLM 不可用。单位 ms，shape 为 B,T,Hkv,Hq。

| 模式 | Shape | latency ms |
| --- | --- | ---: |
| prefill | 1,8192,16,96 | 487.121399 |
| prefill | 2,16384,8,96 | 1060.219849 |
| prefill | 1,32768,16,96 | 2245.593750 |
| prefill | 2,8192,8,96 | 487.046661 |
| prefill | 4,4096,16,384 | 852.993774 |
| prefill | 4,4096,16,256 | 837.738220 |
| decode | 1,4096,16,96 | 0.122112 |
| decode | 1,16384,16,96 | 0.124160 |
| decode | 1,65536,16,96 | 0.137472 |
| decode | 4,4096,8,96 | 0.191232 |
| decode | 4,16384,8,96 | 0.201472 |
| decode | 16,4096,8,96 | 0.594944 |
| decode | 32,2048,4,48 | 0.565504 |
| decode | 64,1024,4,48 | 0.569856 |

6+8 行完整，时延均为正；整个 benchmark 114 s。这里的 PASS 表示运行及结果审计通过，
不是相对性能达标。prefill/decode 工作量不同，不能据此互算加速比；也没有证据证明两个
容器的 MSA 性能相同。后续性能优化优先调查 prefill，但需分阶段 profiling 和同条件对照。
