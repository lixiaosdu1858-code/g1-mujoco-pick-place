# 证据索引

## v0.2.0

- [v02-static-verification.json](v02-static-verification.json)：此前候选入口的静态与保存数据验证，新增物理步为 0；保留当时状态，不改写为动态验收。
- [v02-user-replay-verification.json](v02-user-replay-verification.json)：候选包异目录用户回放 `replay_user_20261023_210302_965_7541bb` 的只读审核摘录，23 项检查通过；用户已确认完整抓放。62.240 仿真秒、15,560 步、15,561 个状态，最大位置误差 2.909046 mm，连续落盘 6.464 秒，非预期接触 0。该运行使用已有第二组示范，不另计独立示范。
- [v02-release-verification.json](v02-release-verification.json)：本地发布副本与已验证候选的对应关系及分发核对；不代表新增动力学试验或 GitHub 已发布。
- [runtime-compatibility.json](runtime-compatibility.json)：仅两个冻结元数据文件的精确迁移。
- [真实完整示例清单](../examples/recording-001/example-manifest.json)：原始状态、动作、初态与保存回放，注明派生元数据及外置场景要求。

## v0.1 历史证据

本目录收录 2026-10-21 三个成功人工抓放回合的轻量证据摘录。原始数据保持只读存档；摘录去除了本机绝对路径，保留源文件哈希与量化结果。

- [comparison.json](comparison.json)：三回合结果及一致性结论。
- [common_configuration.json](common_configuration.json)：24 项公共配置与初态字段。
- [source_snapshots.json](source_snapshots.json)：各原始回合内 84 项源码/资产快照路径及 SHA-256；不代表这些原始文件全部包含于本轻量证据目录。
- `runs/<回合>/summary.json`：明确标记为摘录的运行摘要。
- `runs/<回合>/audit.json`：此前逐行重算审核的摘录，包括末 2 秒物理条件。
- `runs/<回合>/curve_provenance.json`：包内原样 PNG 与原绘图来源文件的哈希、采样语义。

三回合 ID 为 `175535_099`、`212238_372`、`212557_670`，均对应 `manual_user_20261021_` 前缀。哈希核对原件身份，不能替代原件本身；原始 CSV、输入事件和逐段参考未随轻量证据包完整分发。仅用本目录无法重新执行逐样本审核。

软件层面的 `manual_task_completed`、抓取/运输/释放/落盘/跟踪字段均通过；后续复验审核确认人工功能与三次独立启动复验。运行原记录中的 `full_v01_accepted=false`、`learner_verified=false` 没有被改写，完整发布和学习掌握是另外的判断层。
