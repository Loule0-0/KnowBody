#!/usr/bin/env bash
set -euo pipefail
TASK_BUNDLE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_ROOT=/home/lzy/KnowYourBody_RoboDojo
TASK_DATA=/data/KnowYourBody_RoboDojo
TASK_PYTHON="${KYB_PYTHON311:-python3.11}"
"$TASK_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3,11)'
test -f "$TASK_ROOT/releases/v4/runtime41_reproduction_20261102/BASE_HARNESS.json"
test -x /usr/local/cuda-12.8/bin/nvcc
for TASK_ENV in policy_venv retrieval_probe_venv venv; do
    test ! -e "$TASK_DATA/$TASK_ENV"
done
export PYTHONNOUSERSITE=1
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export TERM=xterm-256color
"$TASK_PYTHON" -m venv "$TASK_DATA/policy_venv"
"$TASK_DATA/policy_venv/bin/python" -m pip install --upgrade pip
"$TASK_DATA/policy_venv/bin/python" -m pip install -r "$TASK_BUNDLE/requirements/policy.txt"
"$TASK_PYTHON" -m venv "$TASK_DATA/retrieval_probe_venv"
"$TASK_DATA/retrieval_probe_venv/bin/python" -m pip install --upgrade pip
"$TASK_DATA/retrieval_probe_venv/bin/python" -m pip install -r "$TASK_BUNDLE/requirements/retrieval.txt"
"$TASK_PYTHON" -m venv "$TASK_DATA/venv"
"$TASK_DATA/venv/bin/python" -m pip install --upgrade pip
"$TASK_DATA/venv/bin/python" -m pip install 'setuptools<82' wheel
"$TASK_DATA/venv/bin/python" -m pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
"$TASK_DATA/venv/bin/python" -m pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com -c "$TASK_BUNDLE/requirements/sim_constraints.txt"
# Install the pinned editable source packages directly: no --remote and no RL extras.
"$TASK_DATA/venv/bin/python" -m pip install \
    -e "$TASK_ROOT/RoboDojo/third_party/IsaacLab/source/isaaclab" \
    -e "$TASK_ROOT/RoboDojo/third_party/IsaacLab/source/isaaclab_assets" \
    -e "$TASK_ROOT/RoboDojo/third_party/IsaacLab/source/isaaclab_tasks" \
    -e "$TASK_ROOT/RoboDojo/third_party/IsaacLab/source/isaaclab_rl[none]" \
    -e "$TASK_ROOT/RoboDojo/third_party/IsaacLab/source/isaaclab_mimic[none]" \
    -c "$TASK_BUNDLE/requirements/sim_constraints.txt"
"$TASK_DATA/venv/bin/python" -m pip install -e "$TASK_ROOT/RoboDojo/third_party/curobo[cu12]" --no-build-isolation -c "$TASK_BUNDLE/requirements/sim_constraints.txt"
"$TASK_DATA/venv/bin/python" -m pip install -r "$TASK_ROOT/RoboDojo/scripts/requirements.txt" \
    opencv-python-headless==4.11.0.86 Pillow==11.3.0 PyYAML==6.0.2 omegaconf==2.3.1 trimesh==4.5.1 pyglet==1.5.31 \
    -c "$TASK_BUNDLE/requirements/sim_constraints.txt"
printf '%s\n' 'Environment installation finished. Run preflight.py; no model, Kit or physics was started by this script.'
