# v4 runtime41：新服务器复线包

这是 **v4-runtime41-20261101** 的完整运行源码覆盖层、固定上游版本和新服务器部署工具。与此前只供阅读的 runtime38 快照不同，本目录覆盖实验协调器、经验管理、原生 Codex 会话、动作执行、仿真接入和必要配置。**每条新实验线独立空库起步，线内持续积累**；不复制现有实验的经验、布局状态或调用额度。

先读 [部署步骤](docs/01_SETUP.md)，再读 [建线、启动、日志与恢复](docs/02_EXPERIMENTS.md)。理解代码看 [架构与文件地图](docs/03_CODE_MAP.md)。Fig.2 / Fig.3 的可编辑 PPT 在上一级的 [paper_figures](../paper_figures/)。

| 已包含 | 单独取得 |
| --- | --- |
| 401 份原始源码/配置及逐文件 SHA256；其中 301 份匹配原冻结清单 | Isaac Sim、IsaacLab / cuRobo 的运行依赖与 NVIDIA 驱动 |
| 四个上游仓库的准确 commit；本地修改覆盖层 | 场景、机器人 USD / mesh：按冻结布局从固定数据集下载 |
| 42 任务顺序、公共布局目录元数据、空库建线工具 | 两组固定 Kit 扩展：从有使用权的安装目录私下复制 |
| RGB 与 RGB-D 两种观察配置、两模型独立预算和 token 记录 | 固定 Codex CLI 与本地编码器：工具下载并校验 |
| 精确来源、依赖观察清单、105 项可移植离线检查 | 目标服务器上登录自己的 Codex 账号 |

当前固定配置：主模型 **gpt-6-astra / medium**；管理模型 **gpt-6.1-sol / xhigh**；检索编码器 **Qwen3-Embedding-0.6B**，CPU、1024 维；语义检索 **BM25 + dense**。准备工具要求显式给出主/管调用预算、布局抽样种子、GPU、端口、观察配置；默认不启动模型或物理仿真。

```text
source/workspace/              当前 harness、协调器、提示、schema、测试
source/robodojo_overlay/       原生 RoboDojo / XPolicyLab 本地修改及导入闭包
source/asset_overlay/          小型机器人描述与配置
source/project_overlay/        演示服务器、原始碰撞 padding 配置
catalog/                       上游、数据、CLI、编码器、Kit 与材质身份
requirements/                  安装要求、约束和原服务器观察清单
tools/                         安装、空库建线、下载、检查、启动、状态工具
docs/                          中文逐步部署及运维说明
SOURCE_MANIFEST.json           原始来源与源码 hash
PACKAGE_MANIFEST.json          本复线包全部发布文件 hash
VALIDATION.json                检查范围与结果
```

## 版本与验证范围

原始发布清单 SHA256：`ae6eb6b5689184cfdf488dce27fb776364967bd0c9b586cd941b77de5b9697a7`。本包保留原运行源码字节；安装器另外生成新服务器的 `BASE_HARNESS.json`，每条线再冻结自己的协议、空库、布局和运行资源。历史发布的 905 文件审查档案不伪装成本服务器文件。

本机检查：99 项核心离线回归、6 项部署测试通过；后者用临时目录、合成公共资源，验证两种空库建线能进入原始协调器的 `--preflight-only`。下载与模型/物理执行均为替身或未运行。**另一台服务器的依赖安装、Kit 渲染、实际任务与账号模型访问尚未实测**；按部署文档在目标机验收。原始版本的 549 项测试结果保存在 `RELEASE_INFO.json`，不算作本包新跑的检查。

包内不含认证文件、私有会话、既有实验数据库、原生思考记录、权重或大型仿真依赖。
