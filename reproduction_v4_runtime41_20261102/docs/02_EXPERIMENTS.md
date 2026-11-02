# 2. 独立空库实验线：准备、执行与记录

每条线有唯一名称、独立空库、布局表、向量缓存、Kit 扩展副本、运行缓存、端口和结果目录。相同线内部才接续上一 episode 的最终库。RGB 与 RGB-D 也各自空库起步；不从旧线导入特权感知知识。

## 2.1 三条线示例

下面命令只准备协议、空库与公共资源清单，不执行动作或托管模型调用。执行前先完成安装，确认 `KYB_BUNDLE` 指向本包。布局抽样种子只是示例；正式比较需先确定并冻结。每条线主/管模型都是每 episode 各 100 次，官方原始环境步数不延长。

```bash
cd "$KYB_BUNDLE"
PY=/data/KnowYourBody_RoboDojo/policy_venv/bin/python
"$PY" tools/prepare_line.py --name new_main42_rgbd_01 --kind main42 \
  --profile rgbd_calibrated --episodes 42 --seed 2026110201 \
  --gpu 0 --port 19301 --primary-calls 100 --manager-calls 100
"$PY" tools/prepare_line.py --name new_main42_rgb_01 --kind main42 \
  --profile rgb_only --episodes 42 --seed 2026110202 \
  --gpu 0 --port 19302 --primary-calls 100 --manager-calls 100
"$PY" tools/prepare_line.py --name new_single50_rgbd_01 --kind single50 \
  --profile rgbd_calibrated --episodes 50 --seed 2026110203 \
  --gpu 0 --port 19303 --primary-calls 100 --manager-calls 100
```

`main42` 按官方任务目录顺序各跑一次，准确列表见 `catalog/TASK_ORDER.json`。`single50` 沿用当前单任务协调器，任务固定为 `play_stacking_toy`；它不是任意任务适配器。任务名与观察边界在建线时固定。

布局来自固定 Hugging Face 数据集 revision `35efbc7dedfdbeeb6e95fb749bd885d73d483e41`，每个任务内无放回抽样。`candidate_layouts` 保留完整公共目录，`native_reset_seed` 是所选布局在完整目录中的索引，**不能拿布局文件编号直接替代**。公共布局与对象配置只供宿主建立环境，不作为模型观察。

本包目录中 `play_stacking_toy` 有 55 个布局；一条 50 集线已用 50 个后，不能再要求另抽 50 个且完全不重叠。跨线是否使用相同布局需由实验设计决定。若要排除其他线或上一轮已用布局，准备 JSON `{"task_name": [layout_id, ...]}`，在建线时加 `--exclude /path/to/used_layouts.json`。不足时明确失败，不自动复用或补抽。

独立线可能抽到相同任务布局，这是空库独立性与布局抽样的不同问题。若做配对的 RGB / RGB-D 比较，用相同 seed、相同 exclude；如果需要完全不同布局，先导出已用 ID 再排除。每条线创建后不覆盖、不重新抽样。

## 2.2 下载精确资产并预检

以第一条线为例：

```bash
E=/home/lzy/KnowYourBody_RoboDojo/experiments/new_main42_rgbd_01
python3 "$KYB_BUNDLE/tools/fetch_assets.py" "$E"
python3 "$KYB_BUNDLE/tools/preflight.py" --experiment "$E" --gpu 0 --encoder
python3 "$KYB_BUNDLE/tools/run_line.py" "$E" --preflight-only
```

`prepare_line.py` 会取得小型布局文件和目录元数据；`fetch_assets.py` 再按清单下载所需场景、物体、机器人资源及准确材质镜像，不拉全部训练集。已有同名文件必须符合 hash。资源总量取决于布局，下载中断留下的 `.download` 需要先检查，再决定重试。

相机支架 USD 有三处已审计的材质 URL 替换，指向本地内容一致的 NVIDIA 镜像；准确原始/运行 hash 与替换串在 `ASSET_OVERRIDES.json` 和 `catalog/ASSET_PATCHES.json`，不修改几何或物理参数。机器人小型配置来自原运行覆盖层。

预检会建立 `STATE.json` / 初始快照等运行账本，但无模型调用和物理动作。它检查原协调器实际读取的资源、协议和内存 schema。初始 `evidence / cases / knowledge` 及版本号应全部为 0。公开机器人配置会在正式 episode 初始化时记录，不是从旧经验库复制。

## 2.3 正式启动

实际消耗模型额度并推进仿真从下面 `--start` 开始。首次建议前台观察启动输出；需要后台长跑时使用 `tmux`，保留外层日志。

```bash
export CODEX_HOME="$HOME/.codex-knowbody"
E=/home/lzy/KnowYourBody_RoboDojo/experiments/new_main42_rgbd_01
python3 "$KYB_BUNDLE/tools/run_line.py" "$E" --start
```

其他线重复其各自的资产下载与预检，再从另一个终端启动。端口必须各不相同。当前资源机制在同一 GPU 上最多放行两个 Kit episode，第三条线在边界等待且不消耗模型调用。这个上限来自原 24 GB GPU 环境，不能保证每台 GPU 都能同时承载两份实例；显存较少时手动顺序启动，或将建线时的 `--gpu` 分配到不同 GPU。共享代码只读，各线的证据、向量和缓存分离。

不配置额外管理模型替代项；账号容量或调用结果未知时保留账本并暂停。主模型查询、管理请求和终局复查都计入主模型额度；管理模型各次原生调用独立记账。模型调用额度耗尽或任务失败可按协议封账进入下一任务；软件异常或未知结果不会当成正常失败补跑。

## 2.4 去哪里看结果、经验和 token

```bash
python3 "$KYB_BUNDLE/tools/status.py" "$E"
```

| 位置 | 内容 |
| --- | --- |
| `E/PROTOCOL.json`、`LAYOUTS.json`、`HARNESS.json` | 协议、布局、源文件与配置身份 |
| `E/round_01/STATE.json`（42任务）或 `E/STATE.json`（单任务） | 已封账条目、当前任务、下一顺序、累计库指针 |
| 同级 `RESULTS.md`、`TOKEN_TOTALS.json` | 成功/失败、主/管调用、分角色 token；单任务另有 `EPISODES.csv`、曲线 |
| episode 目录的 `primary_budget.jsonl` / `manager_budget.jsonl` | 跨重启持久化的分角色调用额度 |
| `…/episodes/…/runs/…/RESULT.json` | 本集封账结果及最终库 hash |
| 同级 `MEMORY_UPDATES.json/.md` | 本集发布的案例、可复用知识、适用条件与来源 |
| 单任务同级 `MEMORY_USE.json` | 实际查询、默认知识选择、精确提示输入路径 |
| `/data/KnowYourBody_RoboDojo/runs/RUN_ID/` | 本集完整原始轨迹与最终 SQLite |
| 该运行目录 `report.json` | `model_transport` / `management_transport`，分别为主 / 管用量 |
| 该运行目录 `events.jsonl`、`steps/`、`management/` | 图像、状态、动作、执行回执、检索、管理与发布事件 |
| 该运行目录 `cli/` 及 `management/cli/` | 可审计的实际提示、原生会话和调用记录 |

token 统计以本集独立原生会话的可核验回传计数为准，区分两个角色。缓存输入是输入的子集，reasoning 是输出的子集，不再次相加；未知或冲突保留为空，不补成 0。调用数与 token 数分开，也不据此声称最终账单金额。`status.py` 只读取已封账结果，不把活跃集的局部计数当成完整成本。

长期归档需同时保留实验协议、`/data/…/runs`、最终快照及引用的旧证据。只复制 SQLite 会使图片和来源链接失效。轨迹和原生会话保留在私有实验存储，不提交到公开分支。

## 2.5 停止、边界续接与故障

```bash
touch "$E/STOP"
```

这是**停止进入下一 episode** 的请求，不保证立即中断当前动作。等待当前集封账和进程退出；先查看 `STATE.json`。若 `status=PAUSED`、`active=null`，全部已完成快照完整且调用结果明确，决定继续后可执行：

```bash
rm "$E/STOP"
python3 "$KYB_BUNDLE/tools/reconcile_boundary.py" "$E"
python3 "$KYB_BUNDLE/tools/run_line.py" "$E" --start
```

工具只重开干净的集间边界，记录 `BOUNDARY_RESUME_*.json`，不重置预算或重跑完成任务。`active` 非空、调用结果未知、GPU 崩溃等情况不适用：保留进程信息、调用账本、当前会话、轨迹和 stdout，先分析能否从现有状态恢复。不能删除 `STATE.json`、换名字重新开同一集、把预算文件清空，或把未知结果记成失败后继续。

如果准备阶段失败，检查尚未完成的实验/运行目录和下载残留；准备器不会自动覆盖或重新抽样。这里只提供固定当前版本的空库复线，不加入旧库转换、运行中热切换或通用重放恢复机制。
