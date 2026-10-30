# KnowBody 合作者文档与绘图素材

本分支收录 2026-10-30 整理的 **v4-runtime37** 架构指南、经验案例与真实素材。

| 内容 | 入口 |
| --- | --- |
| 详细中文 Word（24页） | [下载 Word](collaborator_guide_20261030/KnowBody_Collaborator_Guide.docx?raw=true) |
| 固定排版预览 | [PDF](collaborator_guide_20261030/KnowBody_Collaborator_Guide.pdf) |
| 架构与案例图文指南 | [guide.html](collaborator_guide_20261030/guide.html) |
| 可筛选素材库 | [index.html](collaborator_guide_20261030/index.html) |
| 逐图任务、阶段、标注建议与出处 | [CSV](collaborator_guide_20261030/catalog/assets.csv) · [JSON](collaborator_guide_20261030/catalog/assets.json) |
| 33张精选总览 | [PDF](collaborator_guide_20261030/catalog/精选总览.pdf) |
| 完整包使用说明 | [先读我](collaborator_guide_20261030/README_先读我.txt) |

## 下载与打开

[下载整个 assets 分支](https://github.com/Loule0-0/KnowBody/archive/refs/heads/assets.zip)，解压后进入 `collaborator_guide_20261030/`：

1. 用 Word 打开 `KnowBody_Collaborator_Guide.docx`；PDF提供固定版式预览。
2. 用浏览器打开 `index.html`，按任务、相机、案例、分辨率和关键词选图。
3. 打开 `guide.html`，从整体分工读到经验组织、检索、更新和具体案例。

HTML离线可用；请保留解压后的目录结构。

## 素材内容

- **152张原分辨率图片**：真机73张、仿真及用户示范79张，默认精选33张。
- **21个任务组**，逐图附任务、阶段、标注建议、来源和hash。
- **三条实验线的3个主要案例与3个辅助案例**，区分提炼、送达、实际使用和观察反馈。
- **5张机制图**：PNG与可编辑SVG；**7组时序图**附原图ID和裁切记录。
- 实际知识记录、48个公开事件摘要、冻结代码来源索引，以及参考PPT/PDF。

```text
collaborator_guide_20261030/
├── KnowBody_Collaborator_Guide.docx
├── KnowBody_Collaborator_Guide.pdf
├── guide.html / index.html
├── assets/
│   ├── originals/real/ 与 originals/simulation/
│   ├── previews/
│   ├── sequences/
│   └── diagrams/             # PNG + SVG
├── catalog/                  # CSV、JSON与精选总览
├── evidence/                 # 案例、知识与代码来源
├── reference_figures/        # 参考PPT/PDF
└── FILE_MANIFEST.json        # 相对路径、大小与SHA256
```

正式绘图从 `assets/originals/` 取图；时序图的组成与裁切记录见 `catalog/sequences.json`。真机开发演示与当前v4实验案例分别标明证据身份。

正文、案例链和关键图已由Astra复核；文件来源、Word/PDF排版及离线网页功能检查通过。详见 [检查说明](collaborator_guide_20261030/REVIEW_AND_VALIDATION.txt)。
