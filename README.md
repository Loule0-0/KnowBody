<div align="center">

# Know Your Body

### A Harness for Direct and Self-Improving Robot Control with VLMs

Zeyu Lou<sup>1,2</sup> · Yanhong Zeng<sup>2</sup> · Yong Wang<sup>2,3</sup> · Chenyang Si<sup>1,†</sup>

<sup>1</sup>Nanjing University · <sup>2</sup>Ant Group · <sup>3</sup>Zhejiang University

[Paper](https://loule0-0.github.io/KnowBody/paper.pdf) · [arXiv](https://arxiv.org/abs/2609.28530) · [Project Page](https://loule0-0.github.io/KnowBody/) · [Overview Video](https://loule0-0.github.io/KnowBody/assets/knowbody-demo-v9.mp4)

</div>

![KnowBody relates the commanded end-effector pose to the functional point that acts on the world.](https://loule0-0.github.io/KnowBody/assets/fig1-teaser-formal.png)

## Overview

A general-purpose vision–language model can understand a task goal without knowing how a particular robot's motion and functional parts produce the intended effect. **KnowBody** is a harness that makes these action-relevant body relations explicit, queryable, and revisable while keeping the model weights frozen.

Initialized from one off-task trajectory, a partial body model guides action selection and the interpretation of past interactions. New evidence refines the model, and knowledge dependent on revised body estimates is rechecked before reuse.

## Method

KnowBody connects four parts of the interaction loop:

- **Decision context:** the task, current observations, current state, eligible rules, and selected original evidence.
- **Queryable body model:** action-relevant motion, functional-geometry, and whole-arm feasibility relations.
- **Execution feedback:** robot actions, observations, and outcome checks update the current episode state.
- **Validated revision:** candidate body and rule updates are independently checked before reuse in later episodes.

![KnowBody harness architecture.](https://loule0-0.github.io/KnowBody/assets/fig2-harness-formal.png)

## Real-robot evaluation

We evaluate KnowBody on four manipulation tasks: duck placement, apple pushing, writing, and pouring. Under the matched fixed-initialization protocol, both methods use the same frozen VLM and cross-episode updates are disabled.

| Method | Completed trials | Completion |
| --- | ---: | ---: |
| **KnowBody** | **12 / 16** | **75%** |
| Native harness | 4 / 16 | 25% |

![KnowBody real-robot task examples and evaluation results.](https://loule0-0.github.io/KnowBody/assets/fig10-evaluation-formal.png)

## Repository status

This repository is the public home of KnowBody. The project page, paper, and overview video are available now. The public release of the source code, prompts, configuration files, and reproducibility materials is undergoing approval. We will make them available as soon as the approval process is complete.

## Citation

```bibtex
@article{lou2026knowbody,
  title   = {Know Your Body: A Harness for Direct and
             Self-Improving Robot Control with VLMs},
  author  = {Lou, Zeyu and Zeng, Yanhong and
             Wang, Yong and Si, Chenyang},
  year    = {2026},
  journal = {arXiv preprint},
  eprint  = {2609.28530},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  doi     = {10.48550/arXiv.2609.28530}
}
```
