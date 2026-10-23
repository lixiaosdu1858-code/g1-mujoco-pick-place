# 许可与交付状态

项目所有者已明确选择：本项目原创代码与文档采用 MIT 许可证，署名 `Copyright (c) 2026 lixiaosdu1858-code`。完整文本见 [LICENSE](LICENSE)。此决定适用于本项目原创部分，不对第三方作品重新授权。

- 机器人与夹爪的 39 个 STL 已与宇树官方 `unitree_ros` 固定提交中的文件核对，遵循其 BSD-3-Clause 许可并保留相应版权及声明。
- 派生场景 `assets/scene.xml` 与 `configs/scene.xml` 的来源许可缺项尚未闭合。源码发布材料不包含这两份文件，也不包含上游两个输入 XML。使用者提供指定本地输入后，原创生成器可重建已验证场景；生成过程不会改变上游许可状态，也不赋予场景 MIT 许可。
- 软件依赖分别遵循自身许可证；本目录不分发 Python 环境或第三方软件二进制。

v0.1.0 已公开发布。v0.2.0 源码材料已完成本机异目录入口的用户动力学回放核验，见[包验证](docs/package-verification.md)；公开发布状态以仓库的 [Releases](https://github.com/lixiaosdu1858-code/g1-mujoco-pick-place/releases) 为准。随包示例不含 XML：原始 CSV 保持字节不变；为迁移而改写的路径及元数据列入来源映射，不能冒充原件。`Prepare-Example.ps1` 只在本地输出中补入哈希匹配的生成场景。许可追溯见 [第三方声明](THIRD_PARTY_NOTICES.md)。
