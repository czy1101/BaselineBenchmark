# GDN2：保留完整覆盖，缩小调优搜索并复用缓存

证据 ID：`gdn2_full`、`gdn2_paired`、`gdn2_warm`、`gdn2_cold_failed`、
`gdn2_apply`、`gdn2_commit`，见[索引](../workflow/runs/20260907/evidence_index.json)。

## 最终保留的方案

完整 correctness 为 6 个独立 Torch 参考用例，加上 12 Shape × FP16/BF16 ×
TLE/Native 的 48 个扩展组合。扩展文件以内部 Native Triton 为参考，不能称为
“54 项全部使用独立 Torch oracle”。benchmark 保留 12 Shape、2 dtype、2 实现，
即 48 次时延测量、24 行对照结果。

将 GDN2 整理为 `gdn2/` 包；通过 YAML 维护候选配置、持久化 autotuning 结果，
不改变内核主体、调优 key 或必要 hooks。三种东西必须区分：YAML 是候选集合，
autotuning 磁盘缓存保存选择结果，Triton 编译缓存保存编译产物。

| 调优入口 | 原候选数 | curated 候选数 |
| --- | ---: | ---: |
| `_chunk_gdn2_fwd_intra_infer_kernel` | 216 | 4 |
| `chunk_gdn2_fwd_kernel_inter_solve_fused` | 6 | 4 |
| `chunk_gdn2_fwd_kernel_intra_token_parallel` | 16 | 4 |
| `recompute_w_u_fwd_gdn2_kernel` | 9 | 3 |
| GDN2 私有 Native output tuner | 108 | 7 |

私有 output tuner 复用共享 JIT 内核主体，但不复用可变 tuner/config 对象；
共享 generic GLA 的 108 个、MetaX GLA 的 36 个候选保持不变。
未覆盖的输入域回到完整候选路径；`FLAG_ATTN_GDN2_FULL_TUNING=1` 用于完整搜索，
不是启用完整测试的开关。最终默认测试已恢复全部内容。

## 成功结果及可解释范围

最终热缓存复跑使用一个新进程，禁止新的 autotuning，54/54 通过：

| 计时/计数 | 结果 |
| --- | ---: |
| runtime import | 22.063 s |
| pytest | 4.640 s |
| test call 合计 | 4.343 s |
| 子进程 | 36.272 s |
| 整个脚本 | 36.414 s |
| 私有 output 的不同 key / 磁盘命中 | 17 / 17 |
| 所有 tuning 磁盘命中 | 156 |
| 新搜索 / 新候选试跑 | 0 / 0 |

这证明当前缓存下日常验收已可快速完成，不是纯编译耗时测量。此前完整候选验证也记录了
54 项和 24 行 benchmark，但复用了已有编译缓存，不能当作完全空缓存数据。

benchmark 完成也不表示 TLE 在所有输入上比 Native 快：该完整日志中 BF16 的
`(8,2048,32,256,256)` 为 Native 30.775040 ms、TLE 37.190529 ms，比例 0.827×。
这是两种实现的绝对对照，不是调优改动前后的回退证据；应保留慢项而非只展示有加速的行。

历史跨次测量中，BF16 Native 的 Shape `(2,512,8,64,64)` 曾出现约 +10.16% 的
时延差异。随后对同输入、同缓存做 12 轮 ABBA/BAAB：共享路径中位数 0.244224 ms，
私有路径 0.244480 ms，逐轮配对变化中位数 +0.0524%，范围 -0.1832% 至 +1.0428%。
输出和 final state 完全一致，无新搜索。原差异未在配对实验中复现；不能进一步断言
已找到噪声来源或所有 Shape 永无回退。5% 仅是筛查线，不是正式验收阈值。

## 失败、撤回与修正

| 尝试/现象 | 判断 | 处理与经验 |
| --- | --- | --- |
| 默认 test/benchmark 缩为 3 个代表项 | 减少覆盖，不等于编译提速；最终不采用 | 恢复 6+48 用例和 12 Shape，优化搜索而非删覆盖 |
| 只缩 TLE 候选，测试仍慢 | 用例也会运行 Native 参考，共享 output 搜索仍重 | 按阶段计时、检查真实 tuner；不能只按测试名称归因 |
| import 审计断言 MetaX GLA 有 108 个候选 | 审计器混淆 generic 108 和 MetaX 36 | 对源仓与候选按模块逐项比较，不能改算子去迎合错误断言 |
| `from fla` 子串审计、旧 SHA/精确文本替换失败 | 容易误伤 `flag_attn` 或因空白变化失败 | 采用 AST、冻结 manifest、明确差异；失败即停，不跳过保护 |
| Ruff E721 与 EOF 空行 | 精确 builtin int 校验有意排除 bool/子类；另有格式问题 | 一行说明性 E721 例外，AST 相同、26 个 CPU 类型用例通过；删除多余末尾 LF，不全局关闭 lint |
| bare `python benchmark/...` 找不到 `gdn2` | 导入了旧安装或其他 checkout | 明确 `PYTHONPATH`，核对 `module.__file__` 和提交 |
| 空缓存 probe 向 `get_cache_manager` 传普通字符串 | `bytes.fromhex` 在 GPU 测试前报错 | 改为 SHA256 十六进制 key；修复后的 C550 结果仍待补齐 |

最后一项失败运行仅 0.109 s，`correctness=null`，不能写成“冷测试 0.109 s”。
4 个本地 CPU 脚本测试通过只能证明脚本修复的静态/协议行为，不能替代 C550 54 项结果。

## 复现与待办

在已核验的 WZB FlagAttention checkout 根目录执行，先确认空闲设备和源码导入身份：

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest -q tests/flag_attn/test_chunk_gdn2.py tests/flag_attn/test_chunk_gdn2_extended.py
python benchmark/chunk_gdn2_benchmark.py
```

以上是完整入口，不承诺任意环境均为 36 s。复现热缓存还需相同源码、运行时、
profile、编译缓存及 autotuning 缓存设置。

待办：修复后的冷启动用新建独立空缓存运行 54 项，验证运行时真正使用该目录；保留旧缓存。
记录 import、搜索次数/候选次数、pytest、总墙钟。若要计算优化前后提速，需相同硬件、
源码对照和分别为空的缓存，不能拿旧 3 项时间与新 54 项直接计算倍数。
