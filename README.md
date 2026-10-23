# G1 示范采集与动力学回放 · v0.2.0

在 Windows + MuJoCo 中，用键盘操纵固定身体的 G1 单臂完成方块入托盘，记录完整示范；重新启动后，以保存的关节参考和夹爪意图驱动同一控制器，重新计算接触动力学，再比较两次实际运动。

**v0.2.0 本地功能验收已完成；GitHub 发布状态以 [Releases 页面](https://github.com/lixiaosdu1858-code/g1-mujoco-pick-place/releases)为准。** 原工程三组示范的独立动力学回放已通过，其中一组采用完整新格式，并完成工具与方块朝向对比。候选包解压到新目录后，用户又通过新入口完整回放了这条新格式示范，并确认看到了抓取、搬运、松爪和落盘。详见[结果与证据](docs/results.md)及[包验证](docs/package-verification.md)。

本机异目录回放覆盖 **62.240 仿真秒、15,560 个物理步**，最大位置跟踪误差 **2.909 mm**，连续有效落盘 **6.464 秒**，非预期接触数为 **0**。23 项只读审核通过；全部 15,561 个保存状态中，除现实计时外的 196 列均与示范一致。此次是同一示范的新包运行验证，不另计为第四条独立示范。见[新包用户回放审核](evidence/v02-user-replay-verification.json)。

此前的[候选静态摘要](evidence/v02-static-verification.json)保留原有日期和“新增物理步为 0”的历史范围；新动态证据单独追加。发布副本与已验证候选的文件对应关系见[发布核对记录](evidence/v02-release-verification.json)。

## 先回放随包示例

本包带一个真实新格式示范及其已验证回放记录。无需重新手动采集。准备条件为 **Windows PowerShell 5.1**、已有隔离 Python 环境，以及按[场景准备](docs/scene-preparation.md)在本地生成的冻结场景。已验证原工程依赖为 Python 3.13.9、MuJoCo 3.3.7、NumPy 2.3.5；脚本不安装或下载软件。

在包根目录执行：

```powershell
$g1Python = '你的隔离环境路径\Scripts\python.exe'
.\Prepare-Example.ps1 -Python $g1Python
```

它核对示例清单，在包内 `outputs/example_*` 创建运行副本，补入**已在本地生成且哈希匹配的场景**，完成格式检查，然后打印 `EPISODE CHECK` 目录。将下一条命令中的路径替换为该输出：

```powershell
.\Run-Replay.ps1 -EpisodeCheckPath '刚才打印的EPISODE CHECK目录' -Python $g1Python
```

点击场景，按 **7** 开始。运动与夹爪会自动回放；完成后物理暂停，按 **9** 关闭并保存。终端返回新的 `REPLAY RUN` 目录。先只想检查文件、模型与初态时，在回放命令末尾加 `-Inspect`，不会开窗或推进物理。

环境位于本包 `.venv` 时可省略 `-Python`。各 v0.2 根入口支持 `-Python` 和 `-CacheDirectory`；输出默认进入本包 `outputs`，可控缓存默认进入本包 `.cache`。环境、项目和缓存应放在有空间的工作盘。

## 比较刚才的回放

```powershell
.\Show-ReplayComparison.ps1 -RunPath '刚才的REPLAY RUN目录' -Python $g1Python
.\Show-OrientationComparison.ps1 -RunPath '刚才的REPLAY RUN目录' -Python $g1Python
```

两条命令只读取保存记录，输出 PNG、统计和来源哈希。`-NoShow` 可只保存图片。位置图比较工具、方块、夹爪、接触与落盘；朝向图比较世界坐标下的旋转和两次运行的角距离。**控制跟踪误差**是实际状态相对当前参考；**回放差异**是回放实际状态相对示范实际状态，两者分别统计。

## 采集自己的示范

需要新示范时运行 `.\Collect-Demonstration.ps1 -Python $g1Python`；仅静态检查可加 `-Inspect`。操作与 v0.1 相同：7 开始，按住方向键沿世界 X/Y 轴移动，PageUp/PageDown 沿 Z 轴移动，6 独立开合爪，8 切换相机，9 保存。无需固定路线或按键次数。

采集入口自动保存状态、控制参考、实际力矩、输入事件、初态和来源快照，并检查完整回合格式。再次检查已有回合可用 `.\Inspect-Demonstration.ps1 -RunPath 'DEMONSTRATION RUN目录' -Python $g1Python`。数据字段、终端状态与回放语义见[示范格式](docs/demonstration-format.md)。

## 已有结果与范围

| 原工程配对 | 示范格式 | 完整仿真时长 | 最大位置跟踪误差 | 连续有效落盘 |
|---|---|---:|---:|---:|
| 234130_681654 → 224350_060_63b41b | 旧格式 | 53.728 s | 2.890 mm | 3.120 s |
| 113143_694_d01c01 → 114617_265_105da2 | 完整新格式 | 62.240 s | 2.909 mm | 6.464 s |
| 134350_832723 → 121745_214_716d4e | 旧格式 | 140.976 s | 2.912 mm | 8.224 s |

三组均完整回放并落盘；这些配对的共同已保存轨迹字段一致。新格式配对中工具、方块的回放朝向差均为零，工具跟踪角误差峰值约 0.513°。旧格式缺少的朝向和速度不补造。候选新入口对第二组示范的再次回放已独立核验，结果见上方摘要。

验证覆盖本机现有依赖环境、同模型、同初态与当前固定场景；另机全新安装尚未验证，也未验证未见初态泛化。v0.3 的 CPU 状态输入模仿学习尚未训练。

原创代码与文档采用 [MIT](LICENSE)，39 个网格遵循上游 BSD-3-Clause。上游及派生场景 XML 不随源码分发，许可边界见[许可状态](LICENSE-STATUS.md)与[第三方声明](THIRD_PARTY_NOTICES.md)。

- [控制链路](docs/control.md)：人工参考、反馈控制与动力学回放。
- [示范格式与来源映射](docs/demonstration-format.md)：完整记录及可迁移示例。
- [结果与证据](docs/results.md)、[绘图口径](docs/plot-verification.md)：三组验证与两类误差。
- [包检查](docs/package-verification.md)、[排错](docs/troubleshooting.md)：环境、文件与失败记录。

`runtime` 为控制代码，`tools/demonstration` 为采集、检查、回放和比较实现，`examples/recording-001` 为一个可迁移示例。完整历史日志和全部原始回合保留在原工程；本包包含指定示例和结果摘要。这里的任务均为固定身体、单臂、单方块的离线仿真。

