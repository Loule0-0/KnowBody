"""Local installation checks. GPU/encoder checks are explicit, never hosted calls."""
import argparse
import json
import os
import socket
import subprocess
from pathlib import Path
from common import BUNDLE, PROJECT, DATA, WORKSPACE, base_files, digest, load


def metadata(python,names):
    code='import json,importlib.metadata as m;print(json.dumps({n:m.version(n) for n in '+repr(names)+'}))'
    return json.loads(subprocess.check_output([str(python),'-c',code],text=True))


def check(experiment=None,gpu=None,encoder=False):
    base_files()
    for env,expected in [
        ('venv',{'torch':'2.7.0+cu128','numpy':'1.26.0','isaacsim':'5.1.0.0','isaaclab':'0.54.3','websockets':'12.0','pillow':'11.3.0'}),
        ('policy_venv',{'numpy':'1.26.0','websockets':'15.0.1','pillow':'11.3.0','openai':'3.8.0'})]:
        actual=metadata(DATA/env/'bin/python',list(expected))
        if actual!=expected:raise ValueError(f'{env} package mismatch: {actual}')
    c=load(BUNDLE/'source/workspace/research/kyb/retrieval_encoder.json')
    if metadata(c['python'],list(c['packages']))!=c['packages']:raise ValueError('Encoder environment differs')
    for rel,value in c['files'].items():
        if digest(Path(c['model_path'])/rel)!=value:raise ValueError('Encoder file differs: '+rel)
    codex=load(BUNDLE/'catalog/CODEX_BINARY.json');exe=PROJECT/codex['target']
    if digest(exe)!=codex['binary_sha256']:raise ValueError('Codex executable differs')
    help_text=subprocess.check_output([str(exe),'exec','--help'],text=True)
    if any(flag not in help_text for flag in codex['required_flags']):raise ValueError('Codex CLI contract differs')
    auth=subprocess.run([str(exe),'login','status'],capture_output=True,text=True)
    if auth.returncode:raise ValueError('Native Codex login missing: run login --device-auth under the same CODEX_HOME')
    if experiment:
        e=Path(experiment).resolve()
        for path,value in load(e/'HARNESS.json')['files'].items():
            if digest(path)!=value:raise ValueError('Study source/config differs: '+path)
        port=load(e/'PROTOCOL.json')['runtime']['server_port']
        with socket.socket() as test:test.bind(('127.0.0.1',port))
        if not (e/'ASSETS_VERIFIED.json').is_file():raise ValueError('Run fetch_assets.py for this study first')
    if gpu is not None:
        code='import torch; assert torch.cuda.is_available(); assert '+str(gpu)+' < torch.cuda.device_count(); print(torch.cuda.get_device_name('+str(gpu)+'))'
        subprocess.run([str(DATA/'venv/bin/python'),'-c',code],check=True)
    if encoder:
        code='from kyb.embeddings import LocalEncoder; e=LocalEncoder(); e.encode([]); e.close(); print("CPU encoder READY")'
        subprocess.run([str(DATA/'policy_venv/bin/python'),'-c',code],env={**os.environ,'PYTHONPATH':str(WORKSPACE/'research')},check=True)
    print(json.dumps({'verdict':'PASS','scope':'Local source/dependency/login/readiness only; no hosted model or Kit/physics calls',
        'cuda_enumeration_checked':gpu is not None,'encoder_cpu_load_checked':encoder,
        'renderer_and_task_execution':'NOT_TESTED','model_account_access':'Not proven by login status'},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',type=Path)
    p.add_argument('--gpu',type=int);p.add_argument('--encoder',action='store_true')
    a=p.parse_args();check(a.experiment,a.gpu,a.encoder)
