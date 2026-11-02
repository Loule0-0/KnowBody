"""Install exact code on a fresh target server. No environments or experiments run."""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
from common import BUNDLE, PROJECT, DATA, WORKSPACE, BASE_MANIFEST, digest, load, verify_sources, write_new


def install(apply=False):
    rows=verify_sources()
    pins=load(BUNDLE/'catalog/UPSTREAM_PINS.json')
    plan={'project':str(PROJECT),'data':str(DATA),'workspace':str(WORKSPACE),
          'upstream':pins,'source_files':len(rows),'model_calls':0,'simulation_actions':0}
    if not apply:
        print(json.dumps({'status':'PLAN_ONLY',**plan},ensure_ascii=False,indent=2));return
    if WORKSPACE.exists() or BASE_MANIFEST.exists():
        raise ValueError('Target runtime already exists; do not install over an existing study/server')
    PROJECT.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    # Clone direct dependencies only; never use submodule --remote or recursively fetch policy weights.
    for name in ['robodojo','xpolicylab','isaaclab','curobo']:
        p=pins[name];target=PROJECT/p['path']
        if not (target/'.git').exists():
            if target.exists() and any(target.iterdir()):
                raise ValueError('Nonempty non-repository target: '+str(target))
            target.parent.mkdir(parents=True,exist_ok=True)
            subprocess.run(['git','clone','--no-checkout',p['url'],str(target)],check=True)
            subprocess.run(['git','-C',str(target),'checkout','--detach',p['commit']],check=True)
        actual=subprocess.check_output(['git','-C',str(target),'rev-parse','HEAD'],text=True).strip()
        if actual!=p['commit']:raise ValueError('Upstream revision differs: '+str(target))
        dirty=subprocess.check_output(['git','-C',str(target),'status','--porcelain','--untracked-files=no'],text=True)
        # Nested submodules may be uninitialized or at their pinned revisions; own files must be clean.
        if any(line[3:] not in ['XPolicyLab','third_party/IsaacLab','third_party/curobo'] for line in dirty.splitlines()):
            raise ValueError('Upstream checkout has local edits: '+str(target))
    assets=PROJECT/'RoboDojo/Assets';(DATA/'Assets').mkdir(exist_ok=True)
    if assets.exists() and not (assets.is_symlink() and assets.resolve()==DATA/'Assets'):
        raise ValueError('Expected absent Assets path or the canonical data symlink')
    if not assets.exists():assets.symlink_to(DATA/'Assets',target_is_directory=True)
    installed={}
    for row in rows:
        if row['target_root'] is None:continue
        root={'project':PROJECT,'data':DATA}[row['target_root']]
        target=root/row['target'];target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(BUNDLE/row['path'],target)
        assert digest(target)==row['sha256']
        installed[str(target)]=row['sha256']
    info=load(BUNDLE/'RELEASE_INFO.json')
    write_new(BASE_MANIFEST,{'version':'v4','source_version':info['source_version'],
        'original_release_sha256':info['original_release_sha256'],'workspace':str(WORKSPACE),
        'files':installed,'upstream_pins':pins,
        'scope':'New-server source deployment identity; current source bytes retained, not the historical 905-file review archive.'})
    print(json.dumps({'status':'PASS','scope':'Exact source installation only',**plan},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--apply',action='store_true')
    install(p.parse_args().apply)
