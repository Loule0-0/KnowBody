# 3. 从宏观到代码：复线时需要理解的接口

一条实验线的协调器依次选择冻结任务，继承自己的上一集快照，启动一次闭环并封账。每个 episode 的执行模型保留一个原生 Codex 会话，输出末端目标/夹爪动作或请求证据；管理模型负责准备和更新工作材料，并把有来源的交互整理为条件案例与可复用知识。宿主负责当前状态、工具执行、证据保存、版本校验与预算。

```text
main_experiment.py / continual_experiment.py
  → run_research.sh：隔离缓存、GPU租约、演示策略服务器
  → run_episode.py：布局绑定、感知、库快照、循环、终局复查
       ├─ EefAgentPolicy → CodexCliClient → exec / resume
       ├─ MemorySession → 查询 / manager / 校验发布
       └─ execute_chunk → DirectEEF → 原生控制器 → 图像与实测反馈
  ← report.json + final_memory.sqlite + 原始轨迹
  → 分角色账本与本集经验更新 → 下一 episode
```

## 3.1 源码地图

以下路径均相对本包；安装器的准确目标位置在 `SOURCE_MANIFEST.json`。

| 阅读顺序 | 文件 | 职责 |
| --- | --- | --- |
| 1 | `source/workspace/research/main_experiment.py`、`continual_experiment.py` | 42任务 / 单任务协调、冻结验证、封账、记账与接续 |
| 2 | `source/workspace/run_research.sh`、`research/run_episode.py` | 实际运行入口；观察边界、模型、仿真、终局和最终库 |
| 3 | `research/kyb/policy.py`、`codex_cli.py`、`terminal.py` | 主模型动作/查询协议、原生连续会话、终局学习机会 |
| 4 | `research/kyb/session.py`、`preparation.py`、`context.py` | 工具分发、准备材料、动态上下文和按需证据 |
| 5 | `research/kyb/manager.py`、`knowledge.py`、`qualification.py`、`witness.py` | 管理推理、候选知识、来源/适用性/可见证据检查 |
| 6 | `research/kyb/memory.py`、`library.py`、`body_index.py`、`episodes.py`、`leads.py` | E/C/K 持久化、正文、身体端点、episode 与待处理线索 |
| 7 | `research/kyb/hybrid.py`、`embeddings.py`、`embedding_worker.py` | BM25+dense、可见语料、派生向量缓存与独立 CPU 工作进程 |
| 8 | `research/kyb/readout.py`、`media.py`、`tracking.py` | 紧凑读取、历史图像身份、条件及动态状态 |
| 9 | `research/kyb/execution.py`、`body_guard.py`、`motion_core.py`、`surface.py` | 动作 chunk、路径检查、几何与深度工具 |
| 10 | `research/kyb/budget.py`、`native_usage.py`、`native_dispatch.py`、`resources.py` | 持久化额度、token 归并、原生调用调度和 GPU 租约 |
| 11 | `source/robodojo_overlay/XPolicyLab/policy/GPT_6_Astra_Direct_EEF/` | 末端位姿转换、原生执行和执行回执 |

表中省略前缀的 `research/…` 均位于 `source/workspace/`。它包含实际提示文本与 JSON schema；不要拿过去文章中的示意接口替代当前工具定义。

## 3.2 三层内容与几类读取

**E 原始证据**保存观察、动作、实测反馈、执行回执与出处；**C 条件案例**描述特定情境中做了什么、观察到什么；**K 可复用知识**提出以后可怎样使用、什么条件成立、尚有哪些未知，并引用来源案例/证据。三者是内容与来源关系，不是必须逐层展开的同一种查询。身体端点索引、episode 导航与待处理线索是独立入口，不构成“第四层知识”。

语义知识查询先按实验库、感知配置、domain/family/角色与可见历史限定候选，再在同一可见 K 集上做 BM25 与固定编码器 dense 检索，各取 20 条并按 B1/D1/B2/D2 交错去重。分类字段和相似度是找候选的方法，不能代替适用条件核实。向量缓存是派生数据，不是经验的权威库。

身体端点查询按机体、手臂安装、控制点、坐标约定与目标附近空间筛选，提供历史命令/实测误差和来源；某个历史端点可达，不保证当前关节状态到它的路径可达。按 ID 读取则定位已知条目与版本，不做语义排序。需要细节时再读取案例、episode、原始图像和测量，主模型不必每步读全部来源。

精确工具名称、字段和主/管模型可见工具集合以 `policy.py`、`manager.py`、`session.py` 与 schema 为准。上述入口职责不同，执行模型按当前困惑选择；遇到条件冲突或需要比较多段证据时再触发管理。

## 3.3 一次 episode 中知识怎样被用到

开始前，管理模型从本条线已有库组织准备材料；第一集的 E/C/K 全空，只能依赖公开配置和本集新观察。主模型看任务、准备材料、当前图像与状态输出动作，普通步骤不自动重做一次管理推理。具体问题可以直接读知识或案例；关键变化、矛盾或提炼请求才进入管理事务。

宿主每步完整记录证据。管理模型可在被调用时同时整理近期成功和失败交互，发布有条件与引用的 C/K，替换失效工作材料。历史图片须保留原 episode / 事件身份，不能冒充当前观察；runtime41 对此增加了媒体与 witness 校验。终局主模型回合禁止动作，但可决定整理最后一段经验。封账后下一集只继承这条线的最终快照。

例如“抬起后物体还在夹爪里”是一次观察；“某抓持条件下抬升保持，但运输后发生失持”是条件案例；“这类抓持在运输到位后应再检查物体—夹爪关系，抬升成功不足以确认运输保持”才是可复用的观察选择。新线不会预先注入这个例子；它需要由新交互和发布链实际产生。

## 3.4 修改、审计和边界

本部署包不改经验管理算法；新增 `tools/` 负责固定路径安装、空库协议和资源准备。原始文件逐项可查来源与 hash。新服务器协议身份与历史原发布身份分开，避免把不存在的旧实验路径算成部署依赖。

改模型、观察配置、动作接口、提示或预算都改变实验设置；开跑前冻结。运行中有重要正确性问题时先停、留证据再修订，不对当前会话热换代码。普通优化可记入报告，供下一条线或下一轮统一评估。
