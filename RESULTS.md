# v0.2.0 结果入口

[示范与动力学回放结果](docs/results.md)记录原工程三组配对、完整新格式的朝向对比，以及候选包在本机新目录中的用户回放审核。

新入口回放 `replay_user_20261023_210302_965_7541bb` 已完成并经用户确认：62.240 仿真秒、15,560 步，最大位置跟踪误差 2.909046 mm、最大朝向跟踪误差 0.008958375 rad、连续有效落盘 6.464 s、非预期接触 0。23 项只读审核通过；15,561 个状态中除 `wall_s` 外的 196 列与示范逐项一致。见[新包回放证据](evidence/v02-user-replay-verification.json)。

此次使用已有第二组示范，不新增一条独立示范。结果只覆盖本机现有环境及相同模型/初态；另机全新安装和 v0.3 学习策略尚未验证。v0.2.0 本地功能验收已完成；GitHub 发布状态以 [Releases 页面](https://github.com/lixiaosdu1858-code/g1-mujoco-pick-place/releases)为准。

v0.1 的三次人工抓放结果保留于[历史结果](docs/v01-results.md)，独立 v0.1 候选入口的人工抓放记录保留于[历史入口审核](docs/candidate-user-verification.md)。
