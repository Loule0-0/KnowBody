"""Serial, auditable 42-task experiment. No task hints, retries or manual memory edits."""
from __future__ import annotations
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
from datetime import datetime,timezone
from kyb.memory import MemoryStore
from kyb import library
from kyb.trace import append,write_once
from kyb.runtime import call_limits, configuration
from kyb.resources import episode_lease
from kyb.snapshots import MAIN_COUNTS, initial_sealed_counts, sealed_counts, sha256_file as sha

R=Path('/home/lzy/KnowYourBody_RoboDojo')
D=Path('/data/KnowYourBody_RoboDojo')
E=None
W=Path(__file__).resolve().parent.parent
ROUND=None

def now():return datetime.now(timezone.utc).isoformat()
def load(path):return json.loads(Path(path).read_text())
def save(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:
        json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def memory_counts(path):
    with sqlite3.connect(f'file:{path}?mode=ro&immutable=1',uri=True) as db:
        assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
        return {'case_revision':db.execute('SELECT COALESCE(MAX(revision),0) FROM claims').fetchone()[0],
            'knowledge_revision':db.execute('SELECT COALESCE(MAX(revision),0) FROM reusable_knowledge').fetchone()[0],
            'evidence':db.execute('SELECT COUNT(*) FROM evidence').fetchone()[0],
            'cases':db.execute('SELECT COUNT(DISTINCT id) FROM claims').fetchone()[0],
            'knowledge':db.execute('SELECT COUNT(DISTINCT id) FROM reusable_knowledge').fetchone()[0]}

def validate_closed(report,primary_used,manager_used,limits):
    if report.get('status')!='PASS':raise ValueError('Execution/terminal recording failed; no automatic next episode')
    if not report.get('sources_unchanged'):raise ValueError('Runtime sources changed')
    if report.get('task_outcome') not in ('PASS','NEEDS_WORK'):raise ValueError('Physical outcome unknown')
    for role,key,used in [('primary','model_transport',primary_used),('manager','management_transport',manager_used)]:
        transport=report[key]
        if transport.get('outcome_unknown',0):raise ValueError(role+' call outcome unknown')
        if transport['cli_invocations']!=used or used>limits[role]:raise ValueError(role+' allowance/transport mismatch')
        if transport.get('structured_responses_accepted',0)+transport.get('native_failed_turns',0)!=used:
            raise ValueError(role+' response or explicit native failure missing; inspect before continuation')
    if 'terminal_review' not in report:raise ValueError('Final review missing')

def export_memory(snapshot,output,initial_revision,initial_knowledge_revision):
    with sqlite3.connect(f'file:{snapshot}?mode=ro&immutable=1',uri=True) as db:
        db.row_factory=sqlite3.Row
        cases=[dict(r) for r in db.execute('SELECT * FROM claims WHERE revision>? ORDER BY revision',(initial_revision,))]
        knowledge=[dict(r) for r in db.execute('SELECT * FROM reusable_knowledge WHERE revision>? ORDER BY revision',(initial_knowledge_revision,))]
    for r in cases:
        for key in ('conditions','dependencies'):r[key]=json.loads(r[key])
    for r in knowledge:
        for key in ('conditions','params','dependencies','unknowns','refs'):r[key]=json.loads(r[key])
    write_once(output/'MEMORY_UPDATES.json',{'snapshot':str(snapshot),'cases':cases,'knowledge':knowledge})
    lines=['# 本 episode 的经验更新','',f'完整权威库：[SQLite]({snapshot})','']
    for row in knowledge:
        lines.extend([f"## 知识 {row['id']} · v{row['revision']} · {row['domain']}/{row['family']}",'',row['question'],'',row['text'],'',
            '使用：'+row['use_hint'],'','条件：`'+json.dumps(row['conditions'],ensure_ascii=False)+'`','',
            '未知：`'+json.dumps(row['unknowns'],ensure_ascii=False)+'`','',
            '来源：`'+json.dumps(row['refs'],ensure_ascii=False)+'`',''])
    for row in cases:lines.extend([f"## 案例 {row['id']} · v{row['revision']} · {row['kind']}/{row['module']}",'',row['text'],'',
        '条件：`'+json.dumps(row['conditions'],ensure_ascii=False)+'`',''])
    if not cases and not knowledge:lines.extend(['本 episode 未发布新案例或可复用知识。原始轨迹仍完整保留。',''])
    (output/'MEMORY_UPDATES.md').write_text('\n'.join(lines))

def cumulative(rows):
    from kyb.native_usage import reported_total
    out={role:{'calls':0,'tokens':{},'usage_scopes':[],'complete_total_tokens':0} for role in ('primary','manager')}
    for row in rows:
        for role in out:
            out[role]['calls']+=row[role]['cli_invocations']
            total=reported_total(row[role])
            if total is None:out[role]['complete_total_tokens']=None
            elif out[role]['complete_total_tokens'] is not None:out[role]['complete_total_tokens']+=total
            scope=row[role].get('usage_scope','unspecified_historical_report')
            if scope not in out[role]['usage_scopes']:out[role]['usage_scopes'].append(scope)
            for k,v in row[role].get('usage_reported',{}).items():
                if isinstance(v,int):out[role]['tokens'][k]=out[role]['tokens'].get(k,0)+v
    for role in out:
        out[role]['qualification']='Observed reported counters only; missing or inconsistent coverage makes the complete total unknown. These are not certified lower bounds when metadata conflict. Scopes may differ; historical native-response audit is separate and final billing is unknown.'
    return out

def render(state):
    rows=state['completed'];totals=cumulative(rows)
    lines=[f"# 第 {state['round']} 轮运行结果",'',f"状态：**{state['status']}**；已闭合 {len(rows)}/42 个 episode。",'',
        '每任务仅一个 episode；失败也进入累计库。结果为本顺序学习协议，非官方完整多布局评测。','',
        '| 顺序 | 任务 | 布局 | 任务结果 | 主/管调用 | 环境步数 | 累计案例/知识 | 记录 |',
        '| ---: | --- | ---: | --- | --- | --- | --- | --- |']
    for r in rows:
        c=r['memory_counts'];out=Path(r['artifact'])
        outcome=r['task_outcome']+('（同布局重跑）' if r.get('infrastructure_restart') else '')
        lines.append(f"| {r['order']} | {r['task']} | {r['layout']} | {outcome} | {r['primary']['cli_invocations']}/{r['manager']['cli_invocations']} | {r['env_steps']}/{r['env_step_limit']} | {c['cases']}/{c['knowledge']} | [结果]({out/'RESULT.json'}) · [经验]({out/'MEMORY_UPDATES.md'}) |")
    lines.extend(['','分角色累计（缓存/推理分项不能重复加到总量）：','', '```json',json.dumps(totals,ensure_ascii=False,indent=2),'```',''])
    if state.get('active'):lines.extend(['当前任务：`'+json.dumps(state['active'],ensure_ascii=False)+'`',''])
    if state.get('pause_reason'):lines.extend(['暂停原因：'+state['pause_reason'],''])
    (ROUND/'RESULTS.md').write_text('\n'.join(lines))
    save(ROUND/'TOKEN_TOTALS.json',totals)

def verify_frozen():
    h=load(E/'HARNESS.json')
    for path,digest in h['files'].items():
        if sha(path)!=digest:raise ValueError('Frozen source/settings changed: '+path)
    for row in load(E/'LAYOUT_FILES.json'):
        if sha(row['path'])!=row['sha256']:raise ValueError('Pinned layout changed')
    return h

def ensure_assets(task):
    entries=load(E/'TASK_ASSETS.json')[task['task']]
    overrides=load(E/'ASSET_OVERRIDES.json')
    waiting=False
    while True:
        missing=[]
        for row in entries:
            path=D/row['path']
            if not path.exists():missing.append(row['path']);continue
            raw=path.read_bytes()
            expected=overrides.get(row['path'],{}).get('runtime_sha256')
            valid=(hashlib.sha256(raw).hexdigest()==expected if expected else
                len(raw)==row['size'] and (hashlib.sha256(raw).hexdigest()==row['lfs']['oid'] if row.get('lfs') else hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()==row['oid']))
            if not valid:raise ValueError('Task asset differs from pinned/audited identity: '+row['path'])
        if not missing:return
        if not waiting:
            print(json.dumps({'event':'waiting_assets','order':task['order'],'task':task['task'],'missing_files':len(missing)}),flush=True)
            waiting=True
        if (E/'STOP').exists():raise RuntimeError('Stopped at pre-episode asset boundary; no calls sent')
        time.sleep(5)

def main():
    import argparse
    global E,ROUND
    parser=argparse.ArgumentParser(description='Run an explicitly frozen 42-task study; importing never starts an experiment.')
    parser.add_argument('--experiment-dir',required=True,type=Path)
    parser.add_argument('--round',required=True,type=int)
    parser.add_argument('--initial-memory',required=True,type=Path)
    parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args()
    if args.round<1:parser.error('Round must be a positive integer')
    E=args.experiment_dir.resolve();ROUND=E/f'round_{args.round:02d}'
    if E==R/'experiments/main_42':parser.error('Use a new explicit continuation experiment directory')
    ROUND.mkdir(exist_ok=True,parents=True)
    with (ROUND/'executor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        h=verify_frozen();state_path=ROUND/'STATE.json'
        protocol=load(E/'PROTOCOL.json');limits=call_limits(protocol);configuration(protocol)
        if args.round!=protocol['round'] or args.initial_memory.resolve()!=Path(protocol['input_memory']).resolve():
            raise ValueError('Round/source must match the frozen authorized protocol')
        state=load(state_path) if state_path.exists() else None
        if state is not None and (state['status']!='READY' or state.get('active')):
            raise RuntimeError('Existing run is not cleanly ready; reconcile without retry before continuation')
        if 'entry_order' in protocol:
            actual=initial_sealed_counts(state,protocol)
        else:
            if sha(args.initial_memory)!=protocol['input_memory_sha256']:
                raise ValueError('Authorized initial memory changed')
            actual=memory_counts(args.initial_memory)
        if actual!=protocol['initial_memory_counts']:
            raise ValueError('Authorized initial memory counts differ')
        tasks=load(E/'LAYOUTS.json')['tasks']
        if len(tasks)!=42 or [t['order'] for t in tasks]!=list(range(1,43)) or [t['task'] for t in tasks]!=protocol['task_order']:
            raise ValueError('Expected the exact authorized 42-task order')
        if len({t['task'] for t in tasks})!=42 or len({(t['task'],t['layout']) for t in tasks})!=42:
            raise ValueError('Expected one unique task/layout entry per task')
        if state is None:
            seed=ROUND/'initial_memory.sqlite'
            branch,_=MemoryStore.fork_snapshot(args.initial_memory,ROUND/'initial_work.sqlite')
            branch.freeze(seed);branch.close()
            state={'status':'READY','round':args.round,'completed':[],'active':None,'next_order':1,
                'memory':str(seed),'input_memory':str(args.initial_memory),'initial_memory_snapshot':str(seed),'harness_version':h['version']}
            save(state_path,state);render(state)
        if args.preflight_only:
            for task in tasks:ensure_assets(task)
            validation='sealed entry certificate' if 'entry_order' in protocol else 'full SQLite input validation'
            print(json.dumps({'status':'PASS','scope':f'Static 42-task protocol/source/resource and {validation}; zero model/simulator calls'}));return
        try:
            for task in tasks[state['next_order']-1:]:
                if (E/'STOP').exists():raise RuntimeError('Operator stop at episode boundary; no next call sent')
                verify_frozen()
                ensure_assets(task)
                before=(sealed_counts(state, MAIN_COUNTS) if state['completed']
                        else memory_counts(Path(state['memory'])))
                with episode_lease(protocol['runtime']['gpu'], E) as lease_fd:
                    episode_dir=ROUND/'episodes'/f"{task['order']:02d}_{task['task']}"
                    episode_dir.mkdir(parents=True,exist_ok=True)
                    run_id=f"{E.name}_r{args.round:02d}_t{task['order']:02d}_{task['task']}_"+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
                    artifact=episode_dir/'runs'/run_id
                    artifact.mkdir(parents=True,exist_ok=False)
                    run=D/'runs'/run_id
                    state.update(status='RUNNING',active={'order':task['order'],'task':task['task'],'layout':task['layout'],
                        'run':str(run),'artifact':str(artifact),'input_memory':state['memory'],'started_at':now()})
                    save(state_path,state);append(ROUND/'JOURNAL.jsonl',{'event':'episode_started',**state['active']});render(state)
                    command=['bash',str(W/'run_research.sh'),'--experiment-dir',str(E),'--task',task['task'],'--layout',str(task['layout']),
                        '--input-memory',state['memory'],'--call-budget',str(limits['primary']),'--budget-ledger',str(episode_dir/'primary_budget.jsonl'),
                        '--manager-call-budget',str(limits['manager']),'--manager-budget-ledger',str(episode_dir/'manager_budget.jsonl')]
                    write_once(artifact/'LAUNCH.json',{'command':command,'env_overrides':{'KYB_RUN_ID':run_id,'KYB_GPU_EPISODE_LEASE_FD':str(lease_fd)},'input_memory_sha256':sha(state['memory']),'harness_version':h['version']})
                    print(json.dumps({'event':'episode_started',**state['active']}),flush=True)
                    with (artifact/'runtime.log').open('w') as log:
                        process=subprocess.Popen(command,cwd=R,env={**os.environ,'KYB_RUN_ID':run_id,'KYB_GPU_EPISODE_LEASE_FD':str(lease_fd)},stdout=log,stderr=subprocess.STDOUT,pass_fds=(lease_fd,))
                        save(artifact/'PROCESS.json',{'pid':process.pid,'started_at':now()});code=process.wait()
                if code!=0:raise RuntimeError(f'Runtime exited {code}; inspect {artifact}/runtime.log; no retry issued')
                report=load(run/'report.json')
                used=lambda role:len((episode_dir/f'{role}_budget.jsonl').read_text().splitlines())
                validate_closed(report,used('primary'),used('manager'),limits)
                snapshot=run/'final_memory.sqlite';manifest=load(snapshot.with_suffix('.manifest.json'))
                if sha(snapshot)!=manifest['sha256']:raise ValueError('Closed memory hash mismatch')
                counts=memory_counts(snapshot)
                if counts['case_revision']<before['case_revision'] or counts['knowledge_revision']<before['knowledge_revision']:raise ValueError('Memory chain regressed')
                export_memory(snapshot,artifact,before['case_revision'],before['knowledge_revision'])
                result={'order':task['order'],'task':task['task'],'layout':task['layout'],'task_outcome':report['task_outcome'],
                    'recording':'PASS','run':str(run),'artifact':str(artifact),'input_memory':state['memory'],
                    'input_memory_sha256':sha(state['memory']),'final_memory':str(snapshot),'final_memory_sha256':manifest['sha256'],
                    'memory_counts':counts,'primary':report['model_transport'],'manager':report['management_transport'],
                    'env_steps':report['env_steps'],'env_step_limit':report['env_step_limit'],'closed_at':now(),'harness_version':h['version']}
                write_once(artifact/'RESULT.json',result)
                state['completed'].append(result);state.update(status='READY',active=None,next_order=task['order']+1,memory=str(snapshot))
                save(state_path,state);append(ROUND/'JOURNAL.jsonl',{'event':'episode_closed',**result});render(state)
                print(json.dumps({'event':'episode_closed','order':task['order'],'task':task['task'],'outcome':report['task_outcome'],
                    'primary_calls':used('primary'),'manager_calls':used('manager'),'memory_counts':counts}),flush=True)
            state['status']='PASS';state['completed_at']=now();save(state_path,state);render(state)
            print(json.dumps({'event':'round_closed','status':'PASS','task_successes':sum(r['task_outcome']=='PASS' for r in state['completed']),'episodes':len(state['completed'])}),flush=True)
        except BaseException as error:
            state.update(status='PAUSED',pause_reason=str(error),paused_at=now())
            save(state_path,state);append(ROUND/'JOURNAL.jsonl',{'event':'paused','reason':str(error),'active':state.get('active')});render(state)
            raise

if __name__=='__main__':main()
