# 模型、资产与来源

源码候选只整理成功人工抓放所需的网格与配置。上游XML与派生场景不随源码分发，使用者需提供指定本地输入后生成。本轮整理没有下载新模型、修改原运行记录或新增仿真。

| 内容 | 包内位置 | 来源与处理 |
|---|---|---|
| 固定基座、单臂夹爪、桌面方块和托盘场景 | 本地生成 `assets/scene.xml` 与 `configs/scene.xml` | 源码不附这两份文件；目标为已验证场景的原字节，共 33,755 字节 |
| G1 身体与 Dex1-1 夹爪网格 | `assets/meshes/`、`assets/dex1_1/meshes/` | 场景引用的 39 个 STL，18,255,276 字节，逐文件 SHA-256 与原冻结下载清单一致 |
| 虚拟文件系统映射 | `assets/vfs_map.json` | XML 中网格文件名 → 包内相对路径；无需访问原项目绝对路径 |
| 资产追溯清单 | `assets/manifest.json` | 来源仓库、固定 revision、固定 URL、字节数、SHA-256、许可核查状态 |
| 官方对应表 | `assets/upstream-license-map.json` | 39个网格逐一对应宇树官方固定提交；含Git blob、大小和SHA-256 |
| 上游许可与声明 | `licenses/`、`THIRD_PARTY_NOTICES.md` | 保留已取得的原始许可文本，区分软件与模型资产 |

网格总计约 17.41 MiB。包内不含原训练仓库、预训练策略或整套 Python 环境。

## 为什么保留场景原字节

场景来自 `manual_user_20261021_175535_099/source/scene.xml`，其 SHA-256 为：

```text
9d71754df03ab06c4fbed3010ef9a883ce71a3e6ca04de01815d7d1c7ca2630c
```

XML 的网格引用本来就是相对路径。运行包通过虚拟文件系统把这些文件名映射到本地资产，因此无需修改 XML。质量、惯量、几何尺寸、关节范围和场景配置均可对应已成功运行的模型。MuJoCo 编译加载与动力学复现属于不同验证层；包内加载检查本身不等于又完成了一次抓放。

## 来源与许可边界

直接来源是 Hugging Face `lerobot/unitree-g1-mujoco`，固定 revision `905111699ef28bdeb1c69c6fe8ada8405e963a48`。该来源的 README/NOTICE 将 Dex1-1 资产进一步追溯到 HIW-500 和 Unitree ROS。HIW-500 的 Apache-2.0 不重新授权 Unitree 网格。

39个冻结网格的Git blob与字节数均与宇树官方 `unitree_ros` 提交 `5994d4faef0a9cadd3287f8de0199a67eeb2a259` 对应文件一致，附该提交的BSD-3-Clause许可证原文。因此这些网格可按已核对的上游条款随源码材料分发。

HF上游两个XML的许可缺项仍未闭合，不能把网格BSD或项目MIT套用于它们。源码材料明确排除这两个输入和两份派生场景副本；原创生成器只完成本地转换，不赋予输入/输出新的许可。详见[模型准备](scene-preparation.md)与[第三方声明](../THIRD_PARTY_NOTICES.md)。
