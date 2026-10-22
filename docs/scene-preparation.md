# 本地场景准备

源码材料包含原创控制代码、场景生成器、冻结配置和39个已核对上游BSD-3-Clause许可的网格。**上游XML与派生场景XML不附带**；只有下载仓库，尚不能直接启动仿真。

本步骤需要使用者已持有下表指定的两个本地输入。它们的HF固定版本没有明确根许可证，因此本项目不代其授权、不分发它们，也不提供自动下载。没有这些文件时，本版可用于阅读源码与结果；完整模型再分发仍有待上游许可明确。

## 输入与输出

输入来自 [lerobot/unitree-g1-mujoco 固定 revision](https://huggingface.co/lerobot/unitree-g1-mujoco/tree/905111699ef28bdeb1c69c6fe8ada8405e963a48/assets)，只读使用：

| 文件 | 必须匹配的SHA-256 |
|---|---|
| `g1_29dof_with_grippers.xml` | `f177bbfecb2bc9e1cbae9dee875a121e4d19de7a34523bf3ba8db527edc8fa54` |
| `scene_33dof.xml` | `f9f18b6ed53688b3bfe8b0c5a2597fc9b7480cc2313b48035c32e7c006b6cb1d` |

生成器固定已有ABC布局、桌面/托盘几何和居中平移，只做XML变换。它不搜索新姿态、不调整控制器、不导入MuJoCo、不运行物理。输入哈希不匹配、输出与冻结场景不同、输出路径已有文件时均停止。

## 已有输入时如何准备

在仓库根目录的PowerShell中执行。将Python和两个输入路径替换为自己的位置；本机验证使用Python3.13.9、NumPy2.3.5。项目、环境和缓存放在选定的工作盘。下面不会下载或安装软件。

```powershell
$g1Python = '你的环境路径\Scripts\python.exe'
$g1Temp = Join-Path (Get-Location) '.cache\temp'
New-Item -ItemType Directory -Path $g1Temp -Force | Out-Null
$env:TEMP=$g1Temp; $env:TMP=$g1Temp; $env:TMPDIR=$g1Temp
$env:PYTHONDONTWRITEBYTECODE='1'; $env:PYTHONIOENCODING='utf-8'
& $g1Python -B .\tools\prepare_scene.py `
  --robot-xml '你的本地输入目录\g1_29dof_with_grippers.xml' `
  --scene-xml '你的本地输入目录\scene_33dof.xml' `
  --output '.\configs\scene.xml' `
  --receipt '.\outputs\scene-generation.json'
if ($LASTEXITCODE -ne 0) { throw 'Scene preparation failed; do not continue.' }
Copy-Item -LiteralPath '.\configs\scene.xml' -Destination '.\assets\scene.xml'
.\Run.ps1 -Python $g1Python -Inspect
```

成功生成的场景为33,755字节，SHA-256必须为：

```text
9d71754df03ab06c4fbed3010ef9a883ce71a3e6ca04de01815d7d1c7ca2630c
```

已实际核对生成结果与先前成功候选的两份场景逐字节相同，见[生成证据](../evidence/scene-generation.json)。这是原场景的字节重建，不是新抓放试验。已有用户完整抓放证据见[候选入口验证](candidate-user-verification.md)。

两个生成场景及 `upstream/` 已列入 `.gitignore`，源码打包工具也明确排除它们。GitHub网页上传不会根据本地 `.gitignore` 自动替你排除文件，因此不要把含本地输入/生成场景的工作目录整包拖入。生成步骤不改变输入和输出的上游许可边界。
