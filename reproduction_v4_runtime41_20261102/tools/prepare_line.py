"""Freeze one independent empty-start study. Public asset metadata only; no Kit/model."""
from pathlib import Path
import argparse
import datetime
import json
import random
import re
import shutil
import sys
import tempfile
import urllib.request
from urllib.parse import quote

from common import BUNDLE, PROJECT, DATA, WORKSPACE, BASE_MANIFEST, DATASET, REVISION
from common import base_files, digest, fetch_metadata_row, load, relative_path, write_new


def list_directory(directory):
    url=f'https://huggingface.co/api/datasets/{DATASET}/tree/{REVISION}/'+quote(directory,safe='/')+'?recursive=true&expand=false&limit=1000'
    result=[]
    while url:
        req=urllib.request.Request(url,headers={'User-Agent':'KnowBody-runtime41-reproduction'})
        with urllib.request.urlopen(req,timeout=90) as response:
            result.extend(x for x in json.load(response) if x['type']=='file')
            links=response.headers.get('Link','')
        url=next((x.split('>')[0].strip().lstrip('<') for x in links.split(',') if 'rel="next"' in x),None)
    if any(not str(relative_path(x['path'])).startswith(directory+'/') for x in result):
        raise ValueError('Dataset listing outside the requested directory')
    return result


def select_layouts(inventory, *, kind, episodes, seed, exclude):
    tasks=load(BUNDLE/'catalog/TASK_ORDER.json') if kind=='main42' else ['play_stacking_toy']
    catalog={task:[] for task in tasks}
    for metadata in inventory['files']:
        for task in tasks:
            m=re.fullmatch('Assets/Eval_Layout/RoboDojo/arx_x5/0/'+re.escape(task)+r'_(\d+)\.json',metadata['path'])
            if m:catalog[task].append((int(m[1]),metadata))
    rng=random.Random(seed);selected=[]
    for task in tasks:
        all_rows=sorted(catalog[task]);all_ids=[n for n,_ in all_rows]
        eligible=[r for r in all_rows if r[0] not in exclude.get(task,[])]
        needed=1 if kind=='main42' else episodes
        if len(eligible)<needed:raise ValueError(f'{task}: need {needed} unused layouts, have {len(eligible)}')
        for layout,metadata in rng.sample(eligible,needed):
            selected.append({'order':len(selected)+1,'task':task,'layout':layout,
                'path':metadata['path'],'metadata':metadata,'candidate_layouts':all_ids,
                'native_reset_seed':all_ids.index(layout),
                'excluded_layouts':sorted(set(exclude.get(task,[])))})
    return selected


def asset_plan(tasks, *, listing=list_directory):
    import yaml
    index=yaml.safe_load((PROJECT/'RoboDojo/task/RoboDojo/config/_task.yml').read_text())
    directories={};specific={};by_order_dirs={}
    for task in tasks:
        scene=load(DATA/task['path']);name=task['task'];dirs=set()
        for kind in ['Rigid','Dynamic','Geometry','Articulation','Garment','Fluid']:
            for category,items in scene.get(kind,{}).items():
                for item in items:
                    typ='Clutter' if item.get('type')=='cluttered' else kind
                    dirs.add(f'Assets/Object/RoboDojo/{typ}/{category}/{int(item["category_idx"]):05d}')
        dirs.update([f'Assets/Room/{scene["Room"]["default"]}','Assets/Background',
                     f'Assets/Material/{scene["Ground"]["materials"]["default"]}'])
        if 'Table' in scene:dirs.add('Assets/Material/'+scene['Table']['default'])
        robot=index['tasks'].get(name,{}).get('robot_config',index['common']['robot_config'])
        cfg=yaml.safe_load((PROJECT/f'RoboDojo/env_cfg/robot/{robot}.yml').read_text())
        for entry in cfg['robots']:
            if entry.get('type')=='support':dirs.add('Assets/Robots/'+entry['robot_name'])
        if name=='imitate_sorting_sequence':
            directory='Assets/Traj/RoboDojo/imitate_sorting_sequence/0'
            dirs.add(directory)
            specific.setdefault(name,set()).add(f'{directory}/{task["native_reset_seed"]}.pkl')
        elif name in ['play_tic_tac_toe','make_kong']:dirs.add('Assets/Traj/RoboDojo/'+name)
        directories.setdefault(name,set()).update(dirs)
        by_order_dirs[task['order']]=(name,dirs)
    cache={d:listing(d) for d in sorted(set().union(*directories.values()))}
    robot_rows=[{'path':r['path'],'size':r['bytes'],'lfs':{'oid':r['sha256']}} for r in load(BUNDLE/'catalog/ROBOT_ASSETS.json')]
    def rows_for(name,dirs):
        rows={r['path']:r for d in dirs for r in cache[d]
              if not r['path'].startswith('Assets/Traj/RoboDojo/imitate_sorting_sequence/') or r['path'] in specific[name]}
        rows.update({r['path']:r for r in robot_rows})
        return [rows[p] for p in sorted(rows)]
    return ({name:rows_for(name,dirs) for name,dirs in directories.items()},
            {order:rows_for(name,dirs) for order,(name,dirs) in by_order_dirs.items()})


def prepare(args):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{2,100}',args.name):raise ValueError('Use a unique simple experiment name')
    if args.primary_calls<1 or args.manager_calls<1:raise ValueError('Both call budgets must be positive')
    if args.episodes<1:raise ValueError('Episode count must be positive')
    if not 1024<=args.port<=65535 or args.gpu<0:raise ValueError('Explicit GPU and unprivileged port required')
    base=base_files();destination=PROJECT/'experiments'/args.name;runtime=DATA/'runtime'/args.name
    if destination.exists() or runtime.exists():raise ValueError('Experiment/runtime exists; never overwrite or reroll')
    inventory=load(BUNDLE/'catalog/layout_inventory.json')
    if inventory['revision']!=REVISION:raise ValueError('Pinned catalog revision mismatch')
    if args.kind=='main42' and args.episodes!=42:raise ValueError('main42 requires --episodes 42')
    exclude=load(args.exclude) if args.exclude else {}
    tasks=select_layouts(inventory,kind=args.kind,episodes=args.episodes,seed=args.seed,exclude=exclude)
    for task in tasks:fetch_metadata_row(task['metadata'])
    plan,order_assets=asset_plan(tasks)
    if args.kind=='single50':
        # The existing single-task runner requires SHA256 even for small Git blobs.
        # Fetch those exact public files now; large LFS assets remain a separate step.
        for row in {r['path']:r for entries in plan.values() for r in entries if not r.get('lfs')}.values():
            fetch_metadata_row(row)
    ext=load(BUNDLE/'catalog/KIT_EXTENSIONS.json')['files'];seed_root=DATA/'kit_extensions'
    for rel,expected in ext.items():
        if not (seed_root/rel).is_file() or digest(seed_root/rel)!=expected:
            raise ValueError('Run seed_extensions.py with exact installed bundles first: '+rel)
    sys.path.insert(0,str(WORKSPACE/'research'))
    from kyb.memory import MemoryStore
    from kyb import library,body_index,leads,episodes
    from kyb.perception import select
    perception=select(args.profile)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.prepare-'+args.name+'-',dir=destination.parent) as tmp:
        stage=Path(tmp)/'experiment';stage.mkdir()
        store=MemoryStore(stage/'working.sqlite')
        library.initialize(store);body_index.initialize(store);leads.initialize(store);episodes.initialize(store)
        perception.check_memory(store)
        store.freeze(stage/'empty_memory.sqlite');store.close()
        # The working database is a preparation temporary, never used as the seed.
        for name in ['working.sqlite','working.sqlite-wal','working.sqlite-shm']:
            path=stage/name
            if path.exists():path.unlink()
        from main_experiment import memory_counts as main_counts
        from continual_experiment import memory_counts as single_counts
        counts=(main_counts if args.kind=='main42' else single_counts)(stage/'empty_memory.sqlite')
        if any(counts.values()):raise ValueError('Each independent line must start with zero evidence/cases/K')
        protocol=load(BUNDLE/'catalog/TEMPLATE_PROTOCOL.json')
        protocol.update(study=args.kind,harness_id=args.name,perception_profile=args.profile,
            episodes=args.episodes,prior_episodes=0,round=1,rounds_authorized=1,episodes_per_task=1,
            runtime={'root':str(runtime),'server_port':args.port,'gpu':args.gpu},
            input_memory=str(destination/'empty_memory.sqlite'),input_memory_sha256=digest(stage/'empty_memory.sqlite'),
            initial_memory_counts=counts,initial_memory_snapshot=str(destination/'empty_memory.sqlite'),
            initial_memory_sha256=digest(stage/'empty_memory.sqlite'),
            algorithm_workspace=str(WORKSPACE),algorithm_manifest=str(BASE_MANIFEST),algorithm_manifest_sha256=digest(BASE_MANIFEST),
            layout_sampling_seed=args.seed,task_order=load(BUNDLE/'catalog/TASK_ORDER.json') if args.kind=='main42' else ['play_stacking_toy'],
            state_relative_path='round_01/STATE.json' if args.kind=='main42' else 'STATE.json',
            retrieval={'line_id':args.name,'cache_path':str(DATA/'retrieval_indexes'/args.name/'vectors.sqlite')},
            prepared_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            comparison_caution='Independent empty-start fixed-runtime replicate; shared layouts alone do not establish a causal memory benefit.')
        protocol['primary']['calls_per_episode']=args.primary_calls
        protocol['manager']['calls_per_episode']=args.manager_calls
        if args.kind=='single50':protocol['task']='play_stacking_toy'
        layouts={'dataset_revision':REVISION,'simulation_seed':0,'sampling_seed':args.seed,
                 'selection':'Uniform without replacement within each task; frozen before any model/Kit call','tasks':tasks}
        for name,value in [('PROTOCOL.json',protocol),('LAYOUTS.json',layouts),('layout_inventory.json',inventory),('TASK_ASSETS.json',plan)]:write_new(stage/name,value)
        patches=load(BUNDLE/'catalog/ASSET_PATCHES.json')
        overrides={str(Path(p['asset']).relative_to(DATA)):{'original_sha256':p['original_sha256'],
            'runtime_sha256':p['local_sha256'],'reason':'Three exact audited material-source URL substitutions only'} for p in patches}
        write_new(stage/'ASSET_OVERRIDES.json',overrides)
        layout_files=[{'path':str(DATA/t['path']) if args.kind=='main42' else t['path'],'sha256':digest(DATA/t['path'])} for t in tasks]
        write_new(stage/'LAYOUT_FILES.json',layout_files)
        if args.kind=='single50':
            by_order={str(t['order']):[{'path':r['path'],'runtime_sha256':overrides.get(r['path'],{}).get('runtime_sha256',r.get('lfs',{}).get('oid') or digest(DATA/r['path']))} for r in order_assets[t['order']]] for t in tasks}
            write_new(stage/'EPISODE_ASSETS.json',by_order)
        runtime.mkdir(parents=True)
        shutil.copytree(seed_root,runtime/'extensions',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        write_new(stage/'EXTENSIONS.json',{'files':{str(runtime/'extensions'/p):v for p,v in ext.items()},'scope':'Exact vendor copies; each line owns runtime caches'})
        files={**base['files'],str(BASE_MANIFEST):digest(BASE_MANIFEST),
               **{str(runtime/'extensions'/p):v for p,v in ext.items()}}
        codex=load(BUNDLE/'catalog/CODEX_BINARY.json');files[str(PROJECT/codex['target'])]=codex['binary_sha256']
        files.update({str(DATA/r['path']):r['sha256'] for r in load(BUNDLE/'catalog/NVIDIA_MATERIALS.json')['files']})
        for p in stage.iterdir():
            if p.is_file():files[str(destination/p.name)]=digest(p)
        write_new(stage/'HARNESS.json',{'version':'v4','source_version':base['source_version'],
            'workspace':str(WORKSPACE),'original_release_sha256':base['original_release_sha256'],
            'study':str(destination),'files':files})
        command=([str(DATA/'policy_venv/bin/python'),str(WORKSPACE/'research/main_experiment.py'),'--experiment-dir',str(destination),'--round','1','--initial-memory',str(destination/'empty_memory.sqlite')]
            if args.kind=='main42' else [str(DATA/'policy_venv/bin/python'),str(WORKSPACE/'research/continual_experiment.py'),str(destination)])
        write_new(stage/'PREPARED.json',{'verdict':'PASS','scope':'Preparation only, zero model/Kit/physics calls','initial_memory_counts':counts,'launch_command':command,'preflight_command':command+['--preflight-only']})
        stage.rename(destination)
    print(json.dumps({'verdict':'PASS','experiment':str(destination),'prepared_episodes':len(tasks),'launch':command,'started':False},ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name',required=True);p.add_argument('--kind',required=True,choices=['main42','single50'])
    p.add_argument('--profile',required=True,choices=['rgb_only','rgbd_calibrated'])
    p.add_argument('--episodes',type=int,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--port',type=int,required=True)
    p.add_argument('--primary-calls',type=int,required=True);p.add_argument('--manager-calls',type=int,required=True)
    p.add_argument('--exclude',type=Path,help='JSON task→layout IDs to exclude, including other frozen assignments')
    prepare(p.parse_args())
