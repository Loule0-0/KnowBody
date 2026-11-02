#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=/home/lzy/KnowYourBody_RoboDojo
TASK_DATA=/data/KnowYourBody_RoboDojo
TASK_WORKSPACE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
test "${1:-}" = "--experiment-dir"
TASK_EXPERIMENT="${2:?Experiment directory required}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export KYB_RUN_ID="${KYB_RUN_ID:?Main experiment executor must assign one run identity}"
export PATH="$TASK_ROOT/releases/harness-v1/tools:$PATH"
export PYTHONPATH="$TASK_WORKSPACE/research:$TASK_ROOT/RoboDojo:$TASK_ROOT/RoboDojo/XPolicyLab"
export ROBODOJO_ROOT="$TASK_ROOT/RoboDojo"
mapfile -t TASK_RUNTIME < <(python3 - "$TASK_EXPERIMENT" <<'PY'
import json,sys
from pathlib import Path
from kyb.runtime import configuration
root,port,gpu=configuration(json.loads((Path(sys.argv[1])/'PROTOCOL.json').read_text()))
print(root)
print(port)
PY
)
test "${#TASK_RUNTIME[@]}" -eq 2
TASK_RUNTIME_ROOT="${TASK_RUNTIME[0]}"
TASK_SERVER_PORT="${TASK_RUNTIME[1]}"
mkdir -p "$TASK_RUNTIME_ROOT/cache/tmp" "$TASK_RUNTIME_ROOT/logs"
export XDG_CACHE_HOME="$TASK_RUNTIME_ROOT/cache"
export XDG_DATA_HOME="$TASK_RUNTIME_ROOT/share"
export XDG_CONFIG_HOME="$TASK_RUNTIME_ROOT/config"
export TMPDIR="$TASK_RUNTIME_ROOT/cache/tmp"
export CUDA_CACHE_PATH="$TASK_RUNTIME_ROOT/cache/cuda"
export __GL_SHADER_DISK_CACHE_PATH="$TASK_RUNTIME_ROOT/cache/gl_shader"
export TORCH_EXTENSIONS_DIR="$TASK_RUNTIME_ROOT/cache/torch_extensions"
export WARP_CACHE_PATH="$TASK_RUNTIME_ROOT/cache/warp"
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export TERM=xterm-256color
TASK_SERVER_LOG="$TASK_RUNTIME_ROOT/logs/server_$KYB_RUN_ID.log"
python3 - "$TASK_SERVER_PORT" <<'PY'
import socket,sys
with socket.socket() as stream:stream.bind(('127.0.0.1',int(sys.argv[1])))
PY
cd "$TASK_ROOT/RoboDojo"
"$TASK_DATA/policy_venv/bin/python" -c 'import os,runpy;os.close(int(os.environ.pop("KYB_GPU_EPISODE_LEASE_FD")));runpy.run_module("XPolicyLab.client_server.ws.model_server",run_name="__main__",alter_sys=True)' \
    --config-path "$TASK_ROOT/demo_server.yml" --host 127.0.0.1 --port "$TASK_SERVER_PORT" \
    > "$TASK_SERVER_LOG" 2>&1 &
TASK_SERVER_PID=$!
trap 'kill "$TASK_SERVER_PID" 2>/dev/null || true' EXIT
"$TASK_DATA/policy_venv/bin/python" - "$TASK_SERVER_PORT" "$TASK_SERVER_PID" <<'PY'
import time,sys,os
from websockets.sync.client import connect
port,pid=map(int,sys.argv[1:])
for attempt in range(60):
    os.kill(pid,0)
    try:
        with connect(f'ws://127.0.0.1:{port}', open_timeout=1, proxy=None):
            break
    except OSError:
        time.sleep(1)
else:
    raise RuntimeError(f'Public demo server did not open loopback port {port}')
PY
"$TASK_DATA/venv/bin/python" "$TASK_WORKSPACE/research/run_episode.py" "$@"
"$TASK_DATA/policy_venv/bin/python" - <<'PY'
import os,json
from pathlib import Path
path=Path('/data/KnowYourBody_RoboDojo/runs')/os.environ['KYB_RUN_ID']/'report.json'
report=json.loads(path.read_text())
assert report['status']=='PASS' and report['sources_unchanged'], f'Run needs work: {path}'
print(f'[KYB] recorded: {path}',flush=True)
PY
