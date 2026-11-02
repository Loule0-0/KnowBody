# 1. 在新服务器安装

本说明只适用于新的部署。当前运行源码使用固定路径，目标服务器需保留以下两个可写目录；登录账号不必名为 `lzy`。不要批量替换源码里的路径，也不要在已有实验服务器执行安装器。

```text
/home/lzy/KnowYourBody_RoboDojo       代码、冻结协议、实验摘要
/data/KnowYourBody_RoboDojo          Python 环境、资产、运行轨迹、缓存
```

## 1.1 系统准备与取得代码

基线是 Linux x86_64、Ubuntu 22.04、Python 3.11、CUDA Toolkit 12.8，NVIDIA GPU/驱动需满足 [Isaac Sim 5.1 安装要求](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html)。系统需有 `git`、`git-lfs`、`python3.11-venv`、编译工具、CMake、FFmpeg。Isaac Sim 及编译环境需要较多磁盘空间；先为环境和资产预留约 100 GB，并为后续轨迹单独留空间，实际用量取决于任务和记录长度。

如果这两个目录尚不存在，由管理员创建并交给运行账号：

```bash
sudo mkdir -p /home/lzy/KnowYourBody_RoboDojo /data/KnowYourBody_RoboDojo
sudo chown "$(id -un):$(id -gn)" /home/lzy/KnowYourBody_RoboDojo /data/KnowYourBody_RoboDojo
git clone --branch assets --single-branch https://github.com/Loule0-0/KnowBody.git "$HOME/KnowBody-assets"
cd "$HOME/KnowBody-assets/reproduction_v4_runtime41_20261102"
export KYB_BUNDLE="$PWD"
python3 tools/verify_package.py
python3 tools/install_code.py
```

最后一行只展示安装计划。检查路径后执行：

```bash
python3 tools/install_code.py --apply
```

安装器依次拉取 `catalog/UPSTREAM_PINS.json` 的四个 commit，然后覆盖本包中实际修改的文件。它不递归拉取训练权重，不执行 `submodule update --remote`，也不选取最新上游。完成后生成：

```text
/home/lzy/KnowYourBody_RoboDojo/releases/v4/runtime41_reproduction_20261102/BASE_HARNESS.json
```

已有相同 workspace / 基础清单时拒绝覆盖；安装中断后应检查具体目录，不能在运行中的部署上重装。上游仓库许可证见原仓库及本包 `licenses/`；大型组件继续按各自许可取得。

## 1.2 创建三个隔离环境

```bash
cd "$KYB_BUNDLE"
export KYB_PYTHON311=python3.11
bash tools/setup_environments.sh
```

安装脚本会下载依赖并接受 NVIDIA EULA；执行前应具备相应使用权。脚本不启动 Kit、模型或任务。

| 环境 | 用途 | 关键版本 |
| --- | --- | --- |
| `/data/KnowYourBody_RoboDojo/venv` | Isaac Sim、机器人控制、主 harness | Python 3.11、Isaac Sim 5.1.0、Torch 2.7.0+cu128、NumPy 1.26.0、websockets 12.0 |
| `…/policy_venv` | 本地演示策略服务器、实验协调器 | NumPy 1.26.0、websockets 15.0.1、Pillow 11.3.0、OpenAI SDK 3.8.0 |
| `…/retrieval_probe_venv` | CPU 编码器工作进程 | Torch 2.9.1、Transformers 4.57.3、sentence-transformers 5.1.2、NumPy 1.26.4 |

三个环境不能合并：仿真扩展与编码器需要不同依赖。`requirements/*_observed.json` 是原环境的观察记录，含系统/可编辑包，不应直接改名作为全量 pip requirements。安装使用 `.txt` 中的关键依赖及约束；它不是所有传递依赖的 wheel 锁文件。失败时保留安装输出、定位冲突，不自行升级 Sim / Torch 或改 harness 绕过。

本复线统一使用这里的安装脚本和 `run_line.py` 入口。底层策略目录还保留该原生组件自身的独立 HTTP 安装说明与脚本；它们不用于当前 Codex CLI 闭环，不要另行执行。

## 1.3 固定 Codex CLI 和独立登录

```bash
python3 "$KYB_BUNDLE/tools/install_codex.py"
export CODEX_HOME="$HOME/.codex-knowbody"
mkdir -p "$CODEX_HOME"
/home/lzy/KnowYourBody_RoboDojo/releases/harness-v1/tools/codex login --device-auth
/home/lzy/KnowYourBody_RoboDojo/releases/harness-v1/tools/codex login status
```

下载的是官方 `codex-cli 0.161.0` Linux x86_64 产物，归档和解压后二进制都检查 SHA256。已下载同一归档时可传 `--archive /path/to/codex.tar.gz`。不使用系统里版本不同的 `codex` 替代。

此 harness 原生调用 `codex exec` / `exec resume`，固定 ChatGPT 登录方式，忽略用户额外配置；不是切换为 OpenAI SDK 请求。账号必须可访问协议里的两个模型。`login status` 不能证明模型权限或剩余额度；第一次正式调用才会实际验证，工具不另发收费探测。官方参考：[无交互执行与续接](https://learn.chatgpt.com/docs/non-interactive-mode)、[认证](https://learn.chatgpt.com/docs/auth)。不要把原服务器 `auth.json`、会话或整个 `.codex` 上传 GitHub。

同一台服务器的多条线可以在同一个 `CODEX_HOME` 下由原生调度协调，但跨服务器仍可能共享账号容量。任务进入暂停后不自动换模型或重发未知结果。

## 1.4 取得固定本地编码器

```bash
python3 "$KYB_BUNDLE/tools/install_encoder.py"
```

它按 `source/workspace/research/kyb/retrieval_encoder.json` 下载并校验全部 11 份文件。ModelScope URL 使用 `master`，**内容身份由固定 SHA256 决定**；若上游已改变则拒绝安装，不自动接受新权重。也可把原服务器该权重目录私下复制到暂存目录，然后：

```bash
python3 "$KYB_BUNDLE/tools/install_encoder.py" --source /path/to/staged-encoder
```

准确安装位置：`/data/KnowYourBody_RoboDojo/retrieval_models/Qwen3-Embedding-0.6B-modelscope`。CPU、float32、2 线程、1024 维。不要把模型权重提交仓库。

## 1.5 取得两组固定 Kit 扩展

这一步是必须的；GitHub 不含 NVIDIA 二进制。需从有使用权的安装/缓存目录得到完整两组文件：

```text
omni.kit.pip_archive-0.0.0+69cbf6ad.lx64.cp311/
isaacsim.asset.importer.urdf-2.4.31+107.3.3.lx64.r.cp311/
```

合计 3344 个文件，约 181 MB；准确相对路径与 hash 见 `catalog/KIT_EXTENSIONS.json`。原服务器已经核验的目录为：

```text
/data/KnowYourBody_RoboDojo/runtime/main42_v4_round04_20261028/extensions/
```

例如在**新服务器**从旧服务器私下复制；把 `OLD_HOST` 替换成自己的 SSH 地址：

```bash
mkdir -p "$HOME/knowbody-kit-stage"
rsync -a --exclude='__pycache__' --exclude='*.pyc' \
  OLD_HOST:/data/KnowYourBody_RoboDojo/runtime/main42_v4_round04_20261028/extensions/ \
  "$HOME/knowbody-kit-stage/"
python3 "$KYB_BUNDLE/tools/seed_extensions.py" "$HOME/knowbody-kit-stage"
```

若已有另一处合法安装缓存，传其父目录即可；校验不通过先补齐准确文件。种子目录是 `/data/KnowYourBody_RoboDojo/kit_extensions`；后续每条线各自复制，缓存互不覆盖。

## 1.6 检查后再建线

```bash
cd "$KYB_BUNDLE"
/data/KnowYourBody_RoboDojo/policy_venv/bin/python tools/run_offline_tests.py
/data/KnowYourBody_RoboDojo/policy_venv/bin/python tools/test_deployment.py
python3 tools/preflight.py --gpu 0 --encoder
```

前两项为离线替身测试；最后一项验证安装源码、包版本、CLI、登录状态、权重 hash，并显式加载 CPU 编码器、枚举 CUDA。**不启动 Kit，不调用托管模型，不证明渲染和真实任务成功**。结果为 `PASS` 后按 [实验操作](02_EXPERIMENTS.md) 准备精确布局与资产，再做原协调器静态检查。
