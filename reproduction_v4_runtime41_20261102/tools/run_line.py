"""Start or statically preflight exactly the previously prepared study command."""
import argparse
import os
import subprocess
from pathlib import Path
from common import PROJECT, WORKSPACE, load


def run(experiment,*,preflight=False):
    root=Path(experiment).resolve()
    if not (root/'ASSETS_VERIFIED.json').is_file():raise ValueError('Fetch and verify the exact study assets first')
    command=load(root/'PREPARED.json')['launch_command']
    if preflight:command=command+['--preflight-only']
    env={**os.environ,'PYTHONNOUSERSITE':'1','PYTHONUNBUFFERED':'1',
        'PYTHONPATH':os.pathsep.join(map(str,[WORKSPACE/'research',PROJECT/'RoboDojo',PROJECT/'RoboDojo/XPolicyLab'])),
        'PATH':str(PROJECT/'releases/harness-v1/tools')+os.pathsep+os.environ.get('PATH','')}
    # Authentication and the native session directory remain the caller's CODEX_HOME.
    return subprocess.run(command,env=env,check=False).returncode


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('experiment',type=Path)
    mode=p.add_mutually_exclusive_group(required=True);mode.add_argument('--preflight-only',action='store_true');mode.add_argument('--start',action='store_true')
    a=p.parse_args();raise SystemExit(run(a.experiment,preflight=a.preflight_only))
