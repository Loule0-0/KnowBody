# KnowBody V4 核心源码快照

**当前版本：v4-runtime38-20261030。** 代码从当前默认冻结版本提取，保留原始目录、模块名和文件字节；精确来源与SHA256见 [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json)。

本目录供合作者阅读架构、查方法细节及运行选定离线测试。包含经验管理核心、直接EEF控制层源码和必要的小型共享工具；仿真引擎、RoboDojo完整仓库、场景/机器人资产、环境依赖包、模型权重、登录凭据、运行日志和经验数据库不在本快照中。

## 建议阅读顺序

| 顺序 | 入口 | 主要问题 |
| --- | --- | --- |
| 1 | [research/kyb/session.py](research/kyb/session.py) | 执行模型能调用哪些记忆工具？当前状态、原生会话和经验准备如何分工？`instructions()`包含常驻行为指令。 |
| 2 | [research/kyb/memory.py](research/kyb/memory.py) · [library.py](research/kyb/library.py) | E/C/K怎样存储、绑定版本、按Body/External组织并发布？ |
| 3 | [research/kyb/knowledge.py](research/kyb/knowledge.py) · [hybrid.py](research/kyb/hybrid.py) · [readout.py](research/kyb/readout.py) | 查询schema、可见范围、双路检索、完整陈述、导航和分页怎样实现？ |
| 4 | [research/kyb/preparation.py](research/kyb/preparation.py) · [manager.py](research/kyb/manager.py) | 任务前怎样准备材料？管理模型怎样取证、比较、提炼、修订和分组提交？管理提示也在代码中。 |
| 5 | [research/kyb/episodes.py](research/kyb/episodes.py) · [episode_index.py](research/kyb/episode_index.py) · [body_index.py](research/kyb/body_index.py) | 跨episode过程、测量指针与数值身体端点怎样复用？ |
| 6 | [research/kyb/codex_cli.py](research/kyb/codex_cli.py) · [native_dispatch.py](research/kyb/native_dispatch.py) · [native_usage.py](research/kyb/native_usage.py) | 原生CLI连续会话、调用调度和双模型用量怎样记录？ |
| 7 | [research/kyb/policy.py](research/kyb/policy.py) · [motion_core.py](research/kyb/motion_core.py) · [execution.py](research/kyb/execution.py) | 数值EEF/夹爪请求如何变成规划、前缀执行与真实覆盖回执？ |
| 8 | [XPolicyLab/policy/GPT_6_Astra_Direct_EEF/policy.py](XPolicyLab/policy/GPT_6_Astra_Direct_EEF/policy.py) · [pose.py](XPolicyLab/policy/GPT_6_Astra_Direct_EEF/pose.py) | 末端参数、工具/抓持点坐标、姿态组合和底层动作接口。 |
| 9 | [research/run_episode.py](research/run_episode.py) | 这些模块怎样接入一次真实实验。它保留了原部署路径与环境启动逻辑，作为接线参考阅读。 |

## 模块边界

- **记忆与管理**：`memory / library / knowledge / qualification / manager / preparation / readout`。
- **历史组织与身体**：`episodes / episode_index / body_index / instances / segments / state`。
- **上下文与模型传输**：`session / context / codex_cli / native_dispatch / native_usage / budget / terminal`。
- **感知与控制接口**：`policy / motion_core / execution / surface / body_visual / body_guard / tracking`。
- **环境边界**：`robodojo_adapter.py`与`run_episode.py`保留接口源码；`XPolicyLab`内提供控制层和共享工具源码。具体仿真、规划器与机器人资产需由实际运行环境提供。

源码中的主模型/管理模型提示、工具schema、经验字段与更新规则均保留在原模块中，没有另写一套示意实现。

## runtime38 相对图文指南的增量

同分支的合作者指南记录runtime37；这里提供更新的runtime38代码。核心增量集中在：

1. `source_episode`可选查询：沿可见的精确案例来源/直接证据寻找同一历史episode的K；与domain/family/roles取交集，不沿related-K递归扩大范围。
2. 读取结果保留support/counter/related/direct-evidence的来源角色计数；来源匹配不授予原始证据或写入权限。
3. 过程提纲独立按紧凑UTF-8阶段内容计量，最多4096字节/6阶段；来源、顺序和结果门槛仍保留。

本次提取未改控制器、提示或经验库。版本自身的冻结审查与验证范围见 [RELEASE_INFO.json](RELEASE_INFO.json)；本快照的交付检查见 `VALIDATION.json`。

## 本地核验与选定离线测试

在本目录执行：

```bash
python3 verify_snapshot.py

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/research:$PWD/research/tests:$PWD" \
python3 -m unittest test_memory test_hybrid_retrieval test_source_scope \
  test_outline_size test_manager_continuity
```

这组测试使用临时SQLite、合成证据与脚本化管理答复，不调用在线模型、不启动仿真或机器人。它们覆盖来源/版本与读取资格、检索边界、提纲计量和管理事务连续性。

完整控制层运行还会引用NumPy、Pillow、OpenCV等图像/数值工具，以及实际部署中的规划与环境接口；dense检索使用配置声明的本地编码器。这些外部依赖和权重按本次要求不打包。此处没有提供自动启动在线模型或仿真的安装脚本。

## 来源与许可证

`SOURCE_MANIFEST.json`逐文件列出来源、SHA256与是否命中冻结发布清单。共享工具、测试和许可证若未被原发布清单单独固定，会明确标为额外源码快照。上游RoboDojo和XPolicyLab的原始许可证位于 [licenses/](licenses/)。
