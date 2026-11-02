"""Read-only health and memory-use accounting; no model calls or recovery actions."""
from pathlib import Path
import argparse,hashlib,json,os,sqlite3,subprocess,time
from kyb.runtime import call_limits
from datetime import datetime,timezone
ROOT=None
OUT=None
KINDS={'policy_decision','action_intent','receipt','experience_query','manager_tool_result','manager_group_publication'}

def load(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines()] if Path(p).exists() else []
def save(p,v):
    temporary=p.with_suffix('.tmp');temporary.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');os.replace(temporary,p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def usage(path):
    attempts={};sessions={}
    for row in lines(path):
        attempts[row['attempt_id']]=row
        if row.get('thread_id') and row.get('usage') is not None:sessions[row['thread_id']]=row['usage']
    tokens={}
    for u in sessions.values():
        for k,v in u.items():
            if type(v) is int:tokens[k]=tokens.get(k,0)+v
    return {'attempts':len(attempts),'unknown':sum(x['state']=='outcome_unknown' for x in attempts.values()),'tokens_reported':tokens}

def update_public(run,state):
    for ref in lines(run/'events.jsonl'):
        if ref['sequence']<=state['through']:continue
        if ref.get('visibility')=='private' or ref['kind'] not in KINDS:
            state['through']=ref['sequence'];continue
        p=Path(ref['path']);assert p.parent==run/'events' and sha(p)==ref['sha256']
        data=load(p)['payload'];kind=ref['kind']
        if kind=='policy_decision':
            tool=data['tool'];state['tools'][tool]=state['tools'].get(tool,0)+1
            state['last_policy_tool']=tool
        elif kind=='action_intent':
            state['call_tools'][ref['id']]=state.get('last_policy_tool','unattributed')
        elif kind=='receipt':
            tool=state['call_tools'].get(data['intent'],'unattributed');state['physical_steps'][tool]=state['physical_steps'].get(tool,0)+data['executed_actions']
        elif kind=='experience_query':
            q=' '.join(str(data.get('question','')).lower().split());state['questions'][q]=state['questions'].get(q,0)+1
            answer=data.get('answer',{});state['query_views'].extend({'event':ref['id'],'id':r.get('id'),'revision':r.get('revision'),'view':r.get('view'),'status':r.get('status')} for r in answer.get('records',[]))
        elif kind=='manager_group_publication':
            state['knowledge_updates']+=len(data.get('knowledge_updates',[]));state['case_updates']+=len(data.get('memory_updates',[]))
        elif kind=='manager_tool_result' and data.get('result',{}).get('status')=='NEEDS_WORK':
            tool=data['tool'];state['manager_rejections'][tool]=state['manager_rejections'].get(tool,0)+1
        state['through']=ref['sequence']
    return state

def check_memory_edge(source, target, order, edges):
    """Verify an explicit, user-authorized library-maintenance snapshot edge."""
    if source == target:return
    matches=[e for e in edges if e['source']==source and e['target']==target and e['before_order']==order]
    if len(matches)!=1:raise ValueError('Missing unique authorized memory-maintenance edge')
    edge=matches[0]
    if sha(edge['proof'])!=edge['proof_sha256']:raise ValueError('Memory-maintenance proof changed')
    proof=load(edge['proof'])
    if proof.get('verdict')!='PASS' or proof.get('source')!=source or proof.get('target')!=target or proof.get('before_order')!=order:
        raise ValueError('Memory-maintenance proof does not bind this edge')
    if sha(source)!=edge['source_sha256'] or sha(target)!=edge['target_sha256']:
        raise ValueError('Memory-maintenance snapshot changed')
    if proof.get('source_sha256')!=edge['source_sha256'] or proof.get('target_sha256')!=edge['target_sha256']:
        raise ValueError('Memory-maintenance proof hashes differ')
    if proof.get('authorization')!='User2026-10-26: adapt previous own-chain experience to v4':
        raise ValueError('Memory-maintenance authorization absent')

def main(root):
    global ROOT,OUT
    ROOT=Path(root).resolve();OUT=ROOT/'monitor'
    protocol=load(ROOT/'PROTOCOL.json');state=load(ROOT/protocol['state_relative_path']);limits=call_limits(protocol);total=len(load(ROOT/'LAYOUTS.json')['tasks']);issues=[]
    OUT.mkdir(exist_ok=True)
    service=subprocess.run(['systemctl','--user','show',protocol['coordinator_service'],'--property=ActiveState'],capture_output=True,text=True,check=True).stdout.strip()
    done=state['status']=='PASS' and len(state['completed'])==total
    if state['status']=='PAUSED':issues.append(state.get('pause_reason','Paused'))
    if not done and service!='ActiveState=active':issues.append('Coordinator service is not active')
    cache=load(OUT/'CACHE.json') if (OUT/'CACHE.json').exists() else {'snapshots':{},'runs':{}}
    previous=state['initial_memory_snapshot']
    for row in state['completed']:
        try:check_memory_edge(previous,row['input_memory'],row['order'],state.get('memory_adaptations',[]))
        except ValueError as error:issues.append('Snapshot chain differs at '+str(row['order'])+': '+str(error))
        previous=row['final_memory']
        if row['final_memory_sha256'] not in cache['snapshots']:
            if sha(row['final_memory'])!=row['final_memory_sha256']:issues.append('Changed snapshot '+str(row['order']))
            else:
                with sqlite3.connect('file:'+row['final_memory']+'?mode=ro&immutable=1',uri=True) as db:
                    valid=db.execute('PRAGMA integrity_check').fetchone()[0]=='ok' and not db.execute('PRAGMA foreign_key_check').fetchall()
                    if not valid:issues.append('Snapshot integrity failed')
                if valid:cache['snapshots'][row['final_memory_sha256']]=row['order']
        for role in ('primary','manager'):
            manifest=load(Path(row['run'])/'manifest.json')
            used=len(lines(manifest['budget_ledger' if role=='primary' else 'manager_budget_ledger']))
            if used!=row[role]['cli_invocations'] or used>limits[role] or row[role].get('outcome_unknown'):issues.append('Closed role budget mismatch')
        run=Path(row['run']);entry=cache['runs'].setdefault(run.name,{'through':0,'tools':{},'call_tools':{},'physical_steps':{},'questions':{},'query_views':[],'knowledge_updates':0,'case_updates':0,'manager_rejections':{}})
        update_public(run,entry)
    try:
        check_memory_edge(previous,state['memory'],(state.get('active') or {}).get('order',state['next_order']),state.get('memory_adaptations',[]))
    except ValueError as error:issues.append('Current memory chain differs: '+str(error))
    roles={};clarification=None;capacity_wait=[]
    if state.get('active'):
        active=state['active'];run=Path(active['run']);manifest=load(run/'manifest.json') if (run/'manifest.json').exists() else {}
        if active['input_memory']!=state['memory']:issues.append('Active input differs from last sealed snapshot')
        for role,log in [('primary',run/'cli_calls.jsonl'),('manager',run/'management/cli_calls.jsonl')]:
            roles[role]=usage(log);roles[role]['reserved_calls']=len(lines(manifest['budget_ledger' if role=='primary' else 'manager_budget_ledger'])) if manifest else 0
            if roles[role]['reserved_calls']>limits[role] or roles[role]['unknown']:issues.append('Active role unknown or over budget')
        if (run/'events.jsonl').exists():
            entry=cache['runs'].setdefault(run.name,{'through':0,'tools':{},'call_tools':{},'physical_steps':{},'questions':{},'query_views':[],'knowledge_updates':0,'case_updates':0,'manager_rejections':{}})
            update_public(run,entry)
        activity=[p.stat().st_mtime for p in [run/'events.jsonl',run/'cli_calls.jsonl',run/'management/cli_calls.jsonl',Path(active['artifact'])/'runtime.log'] if p.exists()]
        clarification_path=run/'clarification'/'STATE.json'
        if clarification_path.exists():clarification=load(clarification_path)
        for p in (run/'CAPACITY_WAIT.json',run/'management/CAPACITY_WAIT.json'):
            if p.exists():
                capacity=load(p)
                if capacity['status']=='WAITING_FOR_CAPACITY_RESUME':capacity_wait.append(capacity)
        waiting=bool(capacity_wait) or (clarification and clarification['status']=='WAITING_FOR_CLARIFICATION')
        if activity and time.time()-max(activity)>1800 and not waiting:issues.append('No progress for30minutes')
    result={'at_utc':datetime.now(timezone.utc).isoformat(),'health':'NEEDS_WORK' if issues else 'PASS','status':state['status'],
        'closed':len(state['completed']),'planned':total,'successes':sum(r['task_outcome']=='PASS' for r in state['completed']),
        'active':state.get('active'),'clarification':clarification,'capacity_wait':capacity_wait,'roles':roles,'issues':issues,'quality':cache['runs'],
        'qualification':'Delivery/repeated queries/rejections are descriptive signals, not adoption or causal inefficiency. No private reasoning read.'}
    save(OUT/'CACHE.json',cache);save(OUT/'LATEST.json',result)
    with (OUT/'CHECKS.jsonl').open('a') as f:f.write(json.dumps({k:v for k,v in result.items() if k!='quality'},ensure_ascii=False)+'\n')
    (OUT/'STATUS.md').write_text(f'# v4实验监测\n\n健康：{result["health"]}；封账{result["closed"]}/{total}，成功{result["successes"]}。\n\n当前：{json.dumps(result["active"],ensure_ascii=False)}\n\n澄清：{json.dumps(clarification,ensure_ascii=False)}\n\n问题：{json.dumps(issues,ensure_ascii=False)}\n\n检索、写入、调用/token及物理move/hold统计见LATEST.json；交付不等于采用。\n')
    if done:subprocess.run(['systemctl','--user','disable','--now',protocol['monitor_timer']],check=True,capture_output=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('experiment_root',type=Path);args=parser.parse_args();main(args.experiment_root)
