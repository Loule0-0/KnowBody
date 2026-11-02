"""One isolated, serial continual-learning replicate with a frozen episode count."""
from __future__ import annotations
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import sqlite3
import subprocess
from datetime import datetime, timezone
from kyb.memory import MemoryStore
from kyb import body_index, leads, library
from kyb.trace import append, write_once
from kyb.runtime import call_limits, configuration
from kyb.resources import episode_lease
from kyb.snapshots import SINGLE_COUNTS, sealed_counts, sha256_file as sha

R=Path("/home/lzy/KnowYourBody_RoboDojo")
D=Path("/data/KnowYourBody_RoboDojo")
W=Path(__file__).resolve().parent.parent

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
            'knowledge':db.execute('SELECT COUNT(DISTINCT id) FROM reusable_knowledge').fetchone()[0],
            'knowledge_active':db.execute("SELECT count(*) FROM reusable_knowledge k JOIN (SELECT id,max(revision) revision FROM reusable_knowledge GROUP BY id) x USING(id,revision) WHERE k.status='active'").fetchone()[0]}

def validate_closed(report,primary_used,manager_used,limits):
    if report.get('status')!='PASS':raise ValueError('Execution/terminal recording failed; no automatic next episode')
    if not report.get('sources_unchanged'):raise ValueError('Runtime sources changed')
    if report.get('task_outcome') not in ('PASS','NEEDS_WORK'):raise ValueError('Physical outcome unknown')
    for role,key,used in [('primary','model_transport',primary_used),('manager','management_transport',manager_used)]:
        transport=report[key]
        if transport.get('outcome_unknown',0):raise ValueError(role+' call outcome unknown')
        if transport['cli_invocations']!=used or used>limits[role]:raise ValueError(role+' allowance/transport mismatch')
        if transport.get('structured_responses_accepted',0)+transport.get('native_failed_turns',0)!=used:
            raise ValueError(role+' response missing; inspect before continuation')
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


def verify_frozen(root):
    identity=load(root/'HARNESS.json')
    for path,digest in identity['files'].items():
        if sha(path)!=digest:raise ValueError('Frozen source/config changed: '+path)
    for row in load(root/'LAYOUT_FILES.json'):
        if sha(D/row['path'])!=row['sha256']:raise ValueError('Frozen layout changed')
    return identity


def verify_assets(root,order):
    for row in load(root/'EPISODE_ASSETS.json')[str(order)]:
        if sha(D/row['path'])!=row['runtime_sha256']:
            raise ValueError('Pinned resource changed: '+row['path'])


def trace_summary(run):
    requests=[];queries=[];decisions=[];evaluation={};receipts=[]
    for line in (run/'events.jsonl').read_text().splitlines():
        ref=json.loads(line);kind=ref['kind']
        if kind not in {'policy_decision','evaluation','manager_group_publication','experience_query','receipt'}:continue
        data=load(ref['path'])['payload']
        if kind=='policy_decision':decisions.append({'event':ref['id'],'phase':data.get('phase','action'),'tool':data['tool']})
        elif kind=='evaluation':evaluation=data
        elif kind=='receipt':receipts.append({'event':ref['id'],'execution':data['execution'],'policy_stop':data['policy_stop']})
        elif kind=='manager_group_publication':requests.append({'event':ref['id'],'purpose':data['purpose'],'default_ids':data.get('default_ids',[]),
            'knowledge_updates':data.get('knowledge_updates',[]),'memory_updates':data.get('memory_updates',[]),
            'processed':data.get('processed',[])})
        elif kind=='experience_query':
            answer=data.get('answer',{})
            queries.append({'event':ref['id'],'question':data['question'],'status':answer.get('status'),
                'level':answer.get('level'),'scope':answer.get('scope'),
                'records':[{'id':c['id'],'revision':c.get('revision'),'source_episodes':c.get('source_episodes',[])} for c in answer.get('records',[])]})
    inputs={};sessions={}
    for role,directory in [('primary',run),('manager',run/'management')]:
        inputs[role]=[{'instructions':str(p/'input/instructions.txt'),'prompt':str(p/'input/prompt.txt')}
            for p in sorted((directory/'cli').glob('*')) if (p/'input/instructions.txt').exists()]
        latest={}
        for line in (directory/'cli_calls.jsonl').read_text().splitlines():
            event=json.loads(line)
            if event.get('thread_id') and event.get('usage') is not None:
                latest[event['thread_id']]=event['usage']
        sessions[role]=latest
    return {'scope':'Actual retrievals, default selections and exact archived prompt files; no adoption or causal benefit inferred',
        'queries':queries,'publications':requests,'primary_decisions':decisions,
        'receipts':receipts,'evaluation':evaluation,'prompt_inputs':inputs,'native_sessions':sessions}


def termination(report,usage):
    if report['task_outcome']=='PASS':return 'official_success'
    if report['env_steps']>=report['env_step_limit']:return 'official_environment_horizon'
    if report.get('allowance_action_stop'):return 'primary_allowance_boundary'
    if any(d['tool']=='give_up' and d['phase']=='action' for d in usage['primary_decisions']):return 'policy_give_up'
    if usage['evaluation'].get('official_end_flag'):return 'official_end_without_success'
    return 'policy_stop_before_environment_horizon'


def metric_rows(completed, prior_episodes=0):
    from kyb.native_usage import observed_total,reported_total
    rows=[];successes=0;total=0;complete=True
    for result in completed:
        successes+=int(result['task_outcome']=='PASS')
        row={k:result[k] for k in ('order','layout','native_reset_seed','task_outcome','recording','termination','env_steps','env_step_limit')}
        row['continual_order']=prior_episodes+row['order']
        row['success']=int(result['task_outcome']=='PASS')
        row['infrastructure_restarts']=int('infrastructure_restart' in result)
        for role in ('primary','manager'):
            row[role+'_calls']=result[role]['cli_invocations']
            row[role+'_models']=json.dumps(result[role].get('models') or [
                {'model':result[role].get('model'),'effort':result[role].get('effort')}],separators=(',',':'))
            row[role+'_usage_scope']=result[role].get('usage_scope','unspecified_historical_report')
            usage=result[role]['usage_reported']
            for field in ('input_tokens','cached_input_tokens','output_tokens','reasoning_output_tokens'):
                row[role+'_'+field]=usage.get(field)
            row[role+'_total_tokens']=reported_total(result[role])
            row[role+'_observed_total_tokens']=observed_total(result[role])
            row[role+'_usage_metadata_consistent']=result[role].get('captured_metadata_consistent')
        role_totals=[row[role+'_total_tokens'] for role in ('primary','manager')]
        row['total_tokens']=sum(role_totals) if all(n is not None for n in role_totals) else None
        if row['total_tokens'] is None:complete=False
        else:total+=row['total_tokens']
        row.update(cumulative_successes=successes,cumulative_tokens=total if complete else None,
            cumulative_successes_per_million_tokens=successes*1e6/total if complete and total else None,
            cases=result['memory_counts']['cases'],knowledge=result['memory_counts']['knowledge'],knowledge_active=result['memory_counts']['knowledge_active'])
        rows.append(row)
    return rows


def plot_curve(root,rows,total_episodes,prior_episodes=0):
    from PIL import Image,ImageDraw,ImageFont
    width,height=1650,1050
    image=Image.new('RGB',(width,height),'white');draw=ImageDraw.Draw(image)
    font=ImageFont.load_default(size=22);small=ImageFont.load_default(size=17)
    colors={'primary':'#4C78A8','manager':'#F58518','success':'#54A24B','case':'#B279A2','knowledge':'#E45756'}
    title='Play Stacking Toy: one continual-learning trajectory (descriptive)'
    if any(r['infrastructure_restarts'] for r in rows):title+='; shaded = infrastructure restart'
    draw.text((50,25),title,fill='black',font=font)
    panels=[(70,100,800,500),(850,100,1580,500),(70,570,800,970),(850,570,1580,970)]
    first=prior_episodes+1;last=prior_episodes+total_episodes
    def panel(box,label,series,maximum=None,bars=False):
        x0,y0,x1,y1=box;left=x0+85;right=x1-25;top=y0+35;bottom=y1-65
        draw.rectangle((x0,y0,x1,y1),outline='#C8C8C8',width=2);draw.text((x0+16,y0+8),label,fill='black',font=small)
        values=[value for _,points in series for value in points if value is not None]
        ymax=max(max(values,default=1),1e-9) if maximum is None else maximum
        def px(order):return left+(order-first)/(max(last-first,1))*(right-left)
        def py(value):return bottom-value/ymax*(bottom-top)
        for step in range(5):
            yy=top+(bottom-top)*step/4;draw.line((left,yy,right,yy),fill='#E8E8E8',width=1)
            draw.text((x0+8,yy-8),f'{ymax*(4-step)/4:.2g}',fill='#555555',font=small)
        draw.line((left,top,left,bottom),fill='black',width=2);draw.line((left,bottom,right,bottom),fill='black',width=2)
        ticks=sorted(set([first,last,*range(((first+4)//5)*5,last+1,5)]))
        for tick in ticks:
            xx=px(tick);draw.line((xx,bottom,xx,bottom+6),fill='black',width=1);draw.text((xx-10,bottom+10),str(tick),fill='#444444',font=small)
        for name,points in series:
            color=colors[name]
            if bars:
                bar=max(5,int((right-left)/max(total_episodes,1)*.55))
                previous=[0.0]*len(points)
                if name=='manager':previous=series[0][1]
                for row,value,base in zip(rows,points,previous):
                    if value is None or base is None:continue
                    xx=px(row['continual_order']);draw.rectangle((xx-bar/2,py(value+base),xx+bar/2,py(base)),fill=color)
            else:
                previous=None
                for row,value in zip(rows,points):
                    if value is None:previous=None;continue
                    point=(px(row['continual_order']),py(value))
                    if previous is not None:draw.line([previous,point],fill=color,width=3)
                    xx,yy=point;draw.ellipse((xx-5,yy-5,xx+5,yy+5),fill=color);previous=point
            lx=right-220;ly=top+8+26*list(n for n,_ in series).index(name)
            draw.rectangle((lx,ly+4,lx+18,ly+18),fill=color);draw.text((lx+26,ly),name,fill='#333333',font=small)
        if any(value is None for _,points in series for value in points):
            draw.text((left+5,bottom-23),'Gaps: usage metadata unavailable/inconsistent',fill='#777777',font=small)
        for row in rows:
            if row['infrastructure_restarts']:
                xx=px(row['continual_order']);draw.rectangle((xx-8,top,xx+8,bottom),fill='#F2D7D5')
    x_success=[r['success'] for r in rows]
    panel(panels[0],'Episode outcome (0/1)',[('success',x_success)],maximum=1)
    panel(panels[1],'Reported input + output tokens (million)',[
        (role,[r[role+'_total_tokens']/1e6 if r['total_tokens'] is not None else None for r in rows]) for role in ('primary','manager')],bars=True)
    panel(panels[2],'Cumulative reported tokens (million)',[('primary',[r['cumulative_tokens']/1e6 if r['cumulative_tokens'] is not None else None for r in rows])])
    panel(panels[3],'Cumulative memory records',[('case',[r['cases'] for r in rows]),('knowledge',[r['knowledge'] for r in rows])])
    image.save(root/'LEARNING_CURVE.png')


def render(root,state):
    protocol=load(root/'PROTOCOL.json');prior_episodes=protocol['prior_episodes']
    completed=state['completed'];rows=metric_rows(completed,prior_episodes);totals=cumulative(completed)
    total_episodes=protocol['episodes']
    save(root/'TOKEN_TOTALS.json',totals)
    save(root/'METRICS.json',{'scope':'One replicate; no per-index population success rate or causal claim','episodes':rows})
    if rows:
        with (root/'EPISODES.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        plot_curve(root,rows,total_episodes,prior_episodes)
    lines=['# Play Stacking Toy：连续学习轨迹','',f"状态：**{state['status']}**；续跑封存 {len(completed)}/{total_episodes}，对应累计序号 {prior_episodes+1}–{prior_episodes+total_episodes}，本段官方成功 {sum(r['success'] for r in rows)} 次。",'',
        '本段的初始经验与版本以冻结的PROTOCOL.json为准；每集重置环境与原生会话，之后只继承前一集快照。失败与终局学习成本保留。',
        '单条轨迹只展示0/1结果和预定5次分块比例；后续独立harness按相同episode索引汇总成功率。不同布局难度未被本轨迹独自排除。','',
        '| 次数 | 布局 | 任务结果 | 终止 | 主/管调用 | 输入+输出token | 累计案例/知识 | 记录 |',
        '| ---: | ---: | --- | --- | --- | ---: | --- | --- |']
    for result,row in zip(completed,rows):
        artifact=Path(result['artifact'])
        outcome=row['task_outcome']+('（基础设施重启）' if row['infrastructure_restarts'] else '')
        token_label=f"{row['total_tokens']:,}" if row['total_tokens'] is not None else '未知'
        lines.append(f"| {row['continual_order']} | {row['layout']} | {outcome} | {row['termination']} | {row['primary_calls']}/{row['manager_calls']} | {token_label} | {row['cases']}/{row['knowledge_active']} active | [结果]({artifact/'RESULT.json'}) · [经验]({artifact/'MEMORY_UPDATES.md'}) · [读取/提示]({artifact/'MEMORY_USE.json'}) |")
    if any(r['infrastructure_restarts'] for r in rows):
        lines.extend(['','标注“基础设施重启”的episode经用户同意在原布局重新开始，保留原预算和中断调用成本；不代表精确物理续接。实际模型与版本段见该episode结果，不能当作全程相同管理模型的样本。'])
    if rows:lines.extend(['',f'![连续学习曲线]({root/"LEARNING_CURVE.png"})',''])
    lines.extend(['','缓存token是输入子集，推理token是输出子集，不再次相加；token数量不等同计费金额。计量缺失或不一致时，完整总量和累计效率留空，图中为缺口；已观察到的计数与口径保留。完整分角色统计见TOKEN_TOTALS.json与EPISODES.csv。',''])
    if state.get('active'):lines.extend(['当前：`'+json.dumps(state['active'],ensure_ascii=False)+'`',''])
    if state.get('pause_reason'):lines.extend(['暂停原因：'+state['pause_reason'],''])
    (root/'RESULTS.md').write_text('\n'.join(lines))


def initialize(root,identity,protocol):
    path=root/'STATE.json'
    if path.exists():
        state=load(path)
        if state['status']!='READY' or state.get('active'):
            raise RuntimeError('Existing execution is not cleanly ready; inspect without retry')
        return state
    snapshot=Path(protocol['initial_memory_snapshot']).resolve()
    manifest=load(snapshot.with_suffix('.manifest.json'))
    if sha(snapshot)!=manifest['sha256']:
        raise ValueError('Initial cumulative memory snapshot hash mismatch')
    validation=root/'initial_memory_validation.sqlite'
    for suffix in ('','-wal','-shm'):
        candidate=Path(str(validation)+suffix)
        if candidate.exists():candidate.unlink()
    store,_=MemoryStore.fork_snapshot(snapshot,validation)
    try:
        library.initialize(store);body_index.initialize(store);leads.initialize(store)
        required={'body_endpoints','learning_leads','publication_groups'}
        tables={r[0] for r in store.db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not required<=tables:raise ValueError('Initial cumulative memory schema is incomplete')
    finally:
        store.close()
        for suffix in ('','-wal','-shm'):
            candidate=Path(str(validation)+suffix)
            if candidate.exists():candidate.unlink()
    counts=memory_counts(snapshot)
    expected=protocol['initial_memory_counts']
    if any(counts[k]!=expected[k] for k in expected):
        raise ValueError('Initial cumulative memory counts differ from frozen protocol')
    state={'status':'READY','harness_id':load(root/'PROTOCOL.json')['harness_id'],'completed':[],
        'active':None,'next_order':1,'memory':str(snapshot),'input_memory':str(snapshot),
        'initial_memory_snapshot':str(snapshot),'harness_version':identity['version']}
    save(path,state);render(root,state)
    return state


def run(root,preflight_only=False):
    with (root/'executor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        identity=verify_frozen(root);protocol=load(root/'PROTOCOL.json');tasks=load(root/'LAYOUTS.json')['tasks']
        total_episodes=protocol.get('episodes')
        if not isinstance(total_episodes,int) or total_episodes<1:
            raise ValueError('Protocol must freeze a positive integer episode count')
        if len(tasks)!=total_episodes or [r['order'] for r in tasks]!=list(range(1,total_episodes+1)) or any(r['task']!='play_stacking_toy' for r in tasks):
            raise ValueError('Frozen Play Stacking Toy entries must exactly match protocol episode count')
        limits=call_limits(protocol)
        configuration(protocol)
        for task in tasks:verify_assets(root,task['order'])
        state=initialize(root,identity,protocol)
        if preflight_only:
            print(json.dumps({'status':'PASS','scope':'static/inherited-memory/resource preflight only; no simulator or model calls'}));return
        try:
            for task in tasks[state['next_order']-1:]:
                if (root/'STOP').exists():raise RuntimeError('Operator stop at episode boundary; no next call sent')
                verify_frozen(root);verify_assets(root,task['order'])
                before=(sealed_counts(state, SINGLE_COUNTS) if state['completed']
                        else memory_counts(state['memory']))
                with episode_lease(protocol['runtime']['gpu'], root) as lease_fd:
                    episode=root/'episodes'/f"{task['order']:02d}";episode.mkdir(parents=True,exist_ok=True)
                    run_id=f"stacktoy_{protocol['harness_id']}_e{task['order']:02d}_"+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
                    artifact=episode/'runs'/run_id;artifact.mkdir(parents=True,exist_ok=False);run_dir=D/'runs'/run_id
                    state.update(status='RUNNING',active={'order':task['order'],'task':task['task'],'layout':task['layout'],
                        'run':str(run_dir),'artifact':str(artifact),'input_memory':state['memory'],'started_at':now()})
                    save(root/'STATE.json',state);append(root/'JOURNAL.jsonl',{'event':'episode_started',**state['active']});render(root,state)
                    command=['bash',str(W/'run_research.sh'),'--experiment-dir',str(root),
                        '--task',task['task'],'--layout',str(task['layout']),'--input-memory',state['memory'],
                        '--call-budget',str(limits['primary']),'--budget-ledger',str(episode/'primary_budget.jsonl'),
                        '--manager-call-budget',str(limits['manager']),'--manager-budget-ledger',str(episode/'manager_budget.jsonl')]
                    write_once(artifact/'LAUNCH.json',{'command':command,'env_overrides':{'KYB_RUN_ID':run_id,'KYB_GPU_EPISODE_LEASE_FD':str(lease_fd)},
                        'input_memory_sha256':sha(state['memory']),'harness_version':identity['version']})
                    print(json.dumps({'event':'episode_started',**state['active']}),flush=True)
                    with (artifact/'runtime.log').open('w') as log:
                        process=subprocess.Popen(command,cwd=R,env={**os.environ,'KYB_RUN_ID':run_id,'KYB_GPU_EPISODE_LEASE_FD':str(lease_fd)},stdout=log,stderr=subprocess.STDOUT,pass_fds=(lease_fd,))
                        save(artifact/'PROCESS.json',{'pid':process.pid,'started_at':now()});code=process.wait()
                if code:raise RuntimeError(f'Runtime exited {code}; inspect {artifact}; no automatic retry')
                report=load(run_dir/'report.json')
                used=lambda role:len((episode/f'{role}_budget.jsonl').read_text().splitlines())
                validate_closed(report,used('primary'),used('manager'),limits)
                snapshot=run_dir/'final_memory.sqlite';manifest=load(snapshot.with_suffix('.manifest.json'))
                if sha(snapshot)!=manifest['sha256']:raise ValueError('Final memory hash mismatch')
                counts=memory_counts(snapshot)
                if any(counts[k]<before[k] for k in ('case_revision','knowledge_revision','evidence')):
                    raise ValueError('Own cumulative memory regressed')
                export_memory(snapshot,artifact,before['case_revision'],before['knowledge_revision'])
                usage=trace_summary(run_dir);write_once(artifact/'MEMORY_USE.json',usage)
                result={'order':task['order'],'task':task['task'],'layout':task['layout'],'native_reset_seed':task['native_reset_seed'],
                    'task_outcome':report['task_outcome'],'recording':'PASS','termination':termination(report,usage),
                    'run':str(run_dir),'artifact':str(artifact),'input_memory':state['memory'],
                    'input_memory_sha256':sha(state['memory']),'final_memory':str(snapshot),'final_memory_sha256':manifest['sha256'],
                    'memory_before':before,'memory_counts':counts,'primary':report['model_transport'],'manager':report['management_transport'],
                    'native_sessions':usage['native_sessions'],'env_steps':report['env_steps'],'env_step_limit':report['env_step_limit'],
                    'wall_seconds':report['wall_seconds'],'closed_at':now(),'harness_version':identity['version']}
                write_once(artifact/'RESULT.json',result)
                state['completed'].append(result);state.update(status='READY',active=None,next_order=task['order']+1,memory=str(snapshot))
                save(root/'STATE.json',state)
                append(root/'JOURNAL.jsonl',{'event':'episode_closed',**result});render(root,state)
                print(json.dumps({'event':'episode_closed','order':task['order'],'outcome':report['task_outcome'],
                    'primary_calls':used('primary'),'manager_calls':used('manager'),'memory_counts':counts}),flush=True)
            state.update(status='PASS',completed_at=now());save(root/'STATE.json',state);render(root,state)
            print(json.dumps({'event':'study_closed','episodes':len(state['completed']),'status':'PASS'}),flush=True)
        except BaseException as error:
            state.update(status='PAUSED',pause_reason=str(error),paused_at=now())
            save(root/'STATE.json',state);append(root/'JOURNAL.jsonl',{'event':'paused','reason':str(error),'active':state.get('active')});render(root,state)
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment_root',type=Path)
    parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args()
    run(args.experiment_root.resolve(),args.preflight_only)
