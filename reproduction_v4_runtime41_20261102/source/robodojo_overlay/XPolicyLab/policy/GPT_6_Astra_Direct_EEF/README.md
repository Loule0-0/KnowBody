# GPT_6_Astra_Direct_EEF

**Contributor:** RoboProbe / RoboDojo Team | **Paper:** An Unexpected Robot Policy: Early Evaluations of GPT-6 Astra on RoboDojo and Beyond | **arXiv:** https://arxiv.org/abs/2609.24170 | **Original code:** https://github.com/RoboProbe/RoboProbe

Evaluation-only, RGB-only GPT-6 Astra agent. The model (`L4_INSPECT_PLANNER=astra`, model id `gpt-6-astra`) calls one `move_eef` tool that names world-frame grasp-point dimensions. RoboDojo's local Cartesian planner turns those poses into time-aligned joint paths. There is no VLA checkpoint and no training entry. Shared Inspect runtime lives in `inspect/`.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

The policy server loads no weights. It borrows a Python that can import XPolicyLab (`uv` resolves `policy_uv_env_path` in `deploy.yml`, default `../Pi_05/openpi`). The eval client owns the conversation and launches one HTTP worker per request. Install `openai==3.8.0`, `pillow==11.3.0`, and `socksio==1.0.0` in the worker interpreter:

```bash
bash policy/GPT_6_Astra_Direct_EEF/install.sh \
  /path/to/http-worker/bin/python
```

Set the authorized API credential before evaluating and set `L4_INSPECT_WORKER_PYTHON` to that worker interpreter. SDK calls always use the worker; there is no simulator-process fallback. Isaac Sim 5.1 pins typing_extensions==4.12.2 while this SDK requires >=4.14, so these dependencies must remain isolated. Uncertain remote outcomes are not automatically resent.

## Data Processing

Unsupported. This is an eval-only adapter: `process_data.sh` is omitted, and it does not convert datasets.

## Training

Unsupported. `train.sh` is omitted. There is no checkpoint directory and no training release to schedule. Actions are produced at eval time by the provider API.

## Evaluation

Layout-range entry, same arguments as the original EEF adapter. There is no separate policy GPU: the simulator uses `env_gpu`, and the only inference is the API call. Default task is `arrange_largest_number`. The research runner sets a 240-second hard timeout and accounts every HTTP attempt. There is no mechanical model-call termination.

```bash
export ARK_API_KEY=...
export ROBODOJO_ROOT=/path/to/RoboDojo-eval

bash policy/GPT_6_Astra_Direct_EEF/run_fixed_layout.sh \
  0 0 general_pickup uv
```

Positional arguments are `<layouts> <env_gpu> <task> <eval_env>`. `layouts` is one index (`16`), an inclusive range (`20-39`), or a mix (`1-3,5,12-14`).

Standard harness entry:

```bash
bash policy/GPT_6_Astra_Direct_EEF/eval.sh \
  RoboDojo general_pickup no-checkpoint arx_x5 joint 0 0 0 uv uv
```

`policy_env` (arg 9) is `uv` or a uv project directory that contains `.venv/bin/python`. `eval_env` (arg 10) is `uv` (the `ROBODOJO_ROOT/.venv`), a virtualenv directory, a Python binary, or a conda env that already has the pinned `openai` and `pillow` versions.

`action_type` must stay `joint`: EEF is the model-facing action, and RoboDojo receives the planner's joint path. `env_cfg_type` is `arx_x5`. `ckpt_name` is ignored; `no-checkpoint` is the conventional placeholder. Batch evaluation is not supported.

## Configuration

Extra `deploy.yml` keys beyond the shared set:

| Key | Role |
| --- | --- |
| `policy_uv_env_path` | uv project used when `policy_env` is `uv` (default `../Pi_05/openpi`) |
| `result_dir` | default result directory name |
| `obs_transform_pipeline` | observation transform tag (`xspark-v1.0`) |

Planner overrides (environment):

| Variable | Default | Role |
| --- | --- | --- |
| `L4_INSPECT_PLANNER` | `astra` | `astra` → `gpt-6-astra`; `gpt55` → `gpt-5.5-2026-04-24` |
| `L4_INSPECT_MODEL` | planner default | override the model id |
| `L4_INSPECT_WORKER_PYTHON` | current interpreter | explicit HTTP worker interpreter; use the isolated SDK environment |
| `L4_INSPECT_DEPTH` | `off` | must stay `off` or unset |

## Notes

- The model estimates Cartesian targets from RGB and proprioception. It does not receive metric depth.
- Data processing and training are unsupported.
- A real episode needs `ROBODOJO_ROOT`, a RoboDojo client with CuRobo, and `ARK_API_KEY`.
