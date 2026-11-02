"""Prepare, never launch, an empty-memory RGB42 study paired to a frozen layout list."""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from datetime import datetime, timezone

from kyb.memory import MemoryStore
from kyb import body_index, leads, library, episodes
from kyb.perception import RGB_ONLY
from kyb.trace import write_once
from main_experiment import memory_counts

R=Path('/home/lzy/KnowYourBody_RoboDojo')
W=Path(__file__).resolve().parent.parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def copy_extensions(source, runtime):
    """Use the source study's exact installed bundles; no registry fallback."""
    metadata=json.loads((source/'EXTENSIONS.json').read_text())
    result={'status':'PASS','scope':'Copied and verified exact frozen installed runtime dependencies; no registry download',
        'packages':[],'files':{}}
    for package in metadata['packages']:
        old=Path(package['destination']);new=Path(runtime)/'extensions'/package['name']
        for path,digest in metadata['files'].items():
            p=Path(path)
            if p.is_relative_to(old) and sha(p)!=digest:raise ValueError('Pinned extension changed: '+path)
        shutil.copytree(old,new,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        for path,digest in metadata['files'].items():
            p=Path(path)
            if not p.is_relative_to(old):continue
            target=new/p.relative_to(old)
            if sha(target)!=digest:raise ValueError('Copied extension differs: '+str(target))
            result['files'][str(target)]=digest
        result['packages'].append({'name':package['name'],'source':str(old),'destination':str(new)})
    return result


def prepare(source, destination, runtime, port, gpu):
    source=Path(source).resolve(); destination=Path(destination).resolve()
    pointer=json.loads((R/'CURRENT_HARNESS.json').read_text())
    manifest=Path(pointer['manifest']); harness=json.loads(manifest.read_text())
    if pointer['version']!='v4' or Path(pointer['workspace'])!=W or sha(manifest)!=pointer['manifest_sha256']:
        raise ValueError('Use the frozen latest v4 workspace selected by CURRENT_HARNESS.json')
    for path,digest in harness['files'].items():
        if sha(path)!=digest:raise ValueError('Frozen source changed: '+path)
    if destination.exists():raise ValueError('Destination already exists; never overwrite an experiment')
    source_harness=json.loads((source/'HARNESS.json').read_text())
    required=('PROTOCOL.json','LAYOUTS.json','LAYOUT_FILES.json','layout_inventory.json','TASK_ASSETS.json','ASSET_OVERRIDES.json','EXTENSIONS.json')
    for name in required:
        path=source/name
        if source_harness['files'].get(str(path))!=sha(path):
            raise ValueError('Paired source manifest does not pin '+str(path))
    for path,digest in source_harness['files'].items():
        if sha(path)!=digest:raise ValueError('Paired source changed: '+path)
    for row in json.loads((source/'LAYOUT_FILES.json').read_text()):
        path=Path(row['path'])
        if not path.is_absolute() or sha(path)!=row['sha256']:
            raise ValueError('Paired layout changed: '+str(path))
    layouts=json.loads((source/'LAYOUTS.json').read_text())
    original=json.loads((source/'PROTOCOL.json').read_text())
    tasks=layouts['tasks']
    if len(tasks)!=42 or [t['order'] for t in tasks]!=list(range(1,43)) or [t['task'] for t in tasks]!=original['task_order']:
        raise ValueError('Source must contain the exact frozen 42-task order')
    destination.mkdir(parents=True)
    for name in ('LAYOUTS.json','LAYOUT_FILES.json','layout_inventory.json','TASK_ASSETS.json','ASSET_OVERRIDES.json'):
        shutil.copyfile(source/name,destination/name)
    extensions=copy_extensions(source,Path(runtime).resolve())
    write_once(destination/'EXTENSIONS.json',extensions)
    with tempfile.TemporaryDirectory(prefix='kyb_rgb42_empty_') as tmp:
        store=MemoryStore(Path(tmp)/'memory.sqlite')
        library.initialize(store);body_index.initialize(store);leads.initialize(store);episodes.initialize(store)
        RGB_ONLY.check_memory(store)
        store.freeze(destination/'empty_memory.sqlite');store.close()
    seed=destination/'empty_memory.sqlite'
    counts=memory_counts(seed)
    if any(counts.values()):raise ValueError('New RGB chain must start empty')
    protocol={
        'study':'RGB-only sequential42; empty library then own-chain accumulation',
        'round':1,'rounds_authorized':1,'episodes':42,'episodes_per_task':1,'prior_episodes':0,
        'task_order':original['task_order'],'harness_version':'v4','harness_id':destination.name,
        'perception_profile':'rgb_only',
        'primary':{'model':'gpt-6-astra','effort':'medium','calls_per_episode':100},
        'manager':{'model':'gpt-6.1-sol','effort':'xhigh','calls_per_episode':100},
        'runtime':{'root':str(Path(runtime).resolve()),'server_port':port,'gpu':gpu},
        'state_relative_path':'round_01/STATE.json',
        'input_memory':str(seed),'input_memory_sha256':sha(seed),'initial_memory_counts':counts,
        'algorithm_workspace':str(W),'algorithm_manifest':str(manifest),'algorithm_manifest_sha256':sha(manifest),
        'dataset_revision':original['dataset_revision'],'simulation_seed':original['simulation_seed'],
        'environment_horizon':'Official per-task action-step limits; no extensions',
        'paired_source':str(source),'paired_layouts_sha256':sha(source/'LAYOUTS.json'),
        'paired_source_harness_sha256':sha(source/'HARNESS.json'),
        'human_action_guidance':False,'manual_memory_edits':False,
        'comparison_caution':'Same task/layout/order/seed, but enhanced line inherits prior rounds. This is not an isolated perception or memory ablation, nor official42x50/170-call evaluation.',
        'allowed_information':'Three raw RGB views, measured proprioception/EEF, public instruction/recipe and public body/action geometry. No depth or camera intrinsics/extrinsics/projections.',
        'normal_failure':'Seal actual outcome and continue',
        'unknown_or_runtime_error':'Pause without retry, reset or new allowance',
        'prepared_at_utc':datetime.now(timezone.utc).isoformat()}
    from kyb.runtime import configuration
    configuration(protocol)
    write_once(destination/'PROTOCOL.json',protocol)
    files={**harness['files'],str(manifest):sha(manifest),**extensions['files']}
    for name in ('PROTOCOL.json','LAYOUTS.json','LAYOUT_FILES.json','layout_inventory.json','TASK_ASSETS.json','ASSET_OVERRIDES.json','EXTENSIONS.json','empty_memory.sqlite','empty_memory.manifest.json'):
        files[str(destination/name)]=sha(destination/name)
    write_once(destination/'HARNESS.json',{**harness,'files':files,'study':str(destination)})
    write_once(destination/'PREPARED.json',{'verdict':'PASS','scope':'Configuration and empty-library preparation only; no simulator/model calls or launch authorization',
        'counts':counts,'layout_manifest_matches':sha(destination/'LAYOUTS.json')==sha(source/'LAYOUTS.json'),
        'launch_command':['/data/KnowYourBody_RoboDojo/policy_venv/bin/python',str(W/'research/main_experiment.py'),
            '--experiment-dir',str(destination),'--round','1','--initial-memory',str(seed)]})
    return destination


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-experiment',type=Path,required=True)
    parser.add_argument('--experiment-dir',type=Path,required=True)
    parser.add_argument('--runtime-root',type=Path,required=True)
    parser.add_argument('--server-port',type=int,required=True)
    parser.add_argument('--gpu',type=int,required=True)
    args=parser.parse_args()
    print(prepare(args.source_experiment,args.experiment_dir,args.runtime_root,args.server_port,args.gpu))
