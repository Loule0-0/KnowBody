"""Whole-episode navigation and explicitly unreviewed controller interpretations.

The index points into immutable events. It never grants evidence reads or turns an
episode outcome into support for a physical claim or a pruned action sequence.
"""
from __future__ import annotations

import json
import hashlib
from collections import Counter
from pathlib import Path
from .memory import encoded
from .retrieval import content_terms
from .contracts import current_contract


def initialize(store):
    store.db.execute('''CREATE TABLE IF NOT EXISTS episode_records(
        run TEXT PRIMARY KEY,task TEXT NOT NULL,chain_ref TEXT NOT NULL REFERENCES evidence(id),
        outline_ref TEXT REFERENCES evidence(id),result_ref TEXT REFERENCES evidence(id),
        cost_ref TEXT REFERENCES evidence(id),summary TEXT NOT NULL,
        result_run TEXT,result_sequence INTEGER,official_success INTEGER,
        outline_run TEXT,outline_sequence INTEGER,has_outline INTEGER NOT NULL DEFAULT 0)''')
    store.db.execute('CREATE INDEX IF NOT EXISTS episode_task ON episode_records(task)')


def termination_reason(*, official_end, env_steps, env_limit, allowance_stop, last_receipt):
    if official_end or env_steps>=env_limit:return 'official_environment_end'
    if allowance_stop:return 'model_allowance_boundary'
    if last_receipt and last_receipt['data'].get('policy_stop'):return 'policy_stop_before_environment_end'
    return 'ended_without_official_environment_boundary'


def begin(session, manifest, official_success, termination='episode_ended'):
    if official_success is not None and type(official_success) is not bool:
        raise ValueError('Only a terminal boolean outcome is permitted')
    initialize(session.store)
    session.episode_ended=True
    session._terminal_outcome=official_success
    if session.store.db.execute('SELECT 1 FROM episode_records WHERE run=?',(session.episode,)).fetchone():
        return
    sources=[dict(r) for r in session.store.db.execute('SELECT id,kind,sequence,sha256,payload,path FROM evidence WHERE run=? ORDER BY sequence',(session.episode,))]
    public_refs=[]
    roots={session.trace.root,*(Path(r['path']).parent.parent for r in sources)}
    for root in sorted(roots):
        log=root/'events.jsonl'
        for line in log.read_text().splitlines() if log.exists() else []:
            ref=json.loads(line)
            if ref['visibility']=='private':continue
            path=Path(ref['path'])
            if path.parent!=root/'events' or hashlib.sha256(path.read_bytes()).hexdigest()!=ref['sha256']:
                raise ValueError('Public episode trace changed')
            public_refs.append(ref)
    posts={}
    for r in sources:
        if r['kind']=='observation':
            payload=json.loads(r['payload'])
            if payload.get('after_receipt'):posts.setdefault(payload['after_receipt'],r)
    receipts=[]
    for row in sources:
        if row['kind']!='receipt':continue
        data=json.loads(row['payload'])
        post=posts.get(row['id'])
        receipts.append({'id':row['id'],'sequence':row['sequence'],'before':data.get('before_observation'),
            'post':post['id'] if post else None,'executed_actions':data.get('executed_actions'),
            'execution':data.get('execution'),'env_start':data.get('env_start'),'env_end':data.get('env_end'),
            'measured_fingers_post_m':json.loads(post['payload']).get('measured_finger_joints_m') if post else None})
    gaps=[]
    for i,receipt in enumerate(receipts):
        if receipt['execution']=='outcome_unknown':gaps.append({'receipt':receipt['id'],'kind':'unknown_execution_outcome'})
        if receipt['post'] is None:gaps.append({'receipt':receipt['id'],'kind':'missing_registered_post'})
        if receipt['env_start'] is None or receipt['env_end'] is None:gaps.append({'receipt':receipt['id'],'kind':'unknown_environment_interval'})
        if i and receipts[i-1]['env_end'] is not None and receipt['env_start'] is not None and receipts[i-1]['env_end']!=receipt['env_start']:
            gaps.append({'receipt':receipt['id'],'kind':'environment_interval_gap','previous_receipt':receipts[i-1]['id']})
        if i and receipts[i-1]['post']!=receipt['before']:gaps.append({'receipt':receipt['id'],'kind':'observation_boundary_changed','previous_receipt':receipts[i-1]['id']})
    summary={'receipt_count':len(receipts),'executed_actions':sum(r['executed_actions'] or 0 for r in receipts),
        'contract_id':current_contract(session),'contract_amendments':list(session.contract_amendments),
        'first_receipt':receipts[0]['id'] if receipts else None,'last_receipt':receipts[-1]['id'] if receipts else None,
        'source_frontier':{session.episode:max((r['sequence'] for r in sources),default=0)},
        'gap_counts':dict(Counter(g['kind'] for g in gaps)),'public_event_count':len(public_refs),
        'prefix_replay':manifest.get('prefix_replay'),'infrastructure_restart':manifest.get('infrastructure_restart'),
        'entry_configuration':{k:v for k,v in (session.body_prior or {}).get('data',{}).get('dependencies',{}).items() if k in ('body','cameras','depth','execution','planner')},
        'termination':termination}
    chain=session.evidence('episode_chain',{'run':session.episode,'task':session.context['task'],
        'sources':[{k:r[k] for k in ('id','kind','sequence','sha256')} for r in sources],
        'public_trace_refs':public_refs,'mechanical_gaps':gaps,'summary':summary,
        'receipts':receipts,'final_observation':session.current_observation['id'] if session.current_observation else None,
        'prefix_replay':manifest.get('prefix_replay'),'infrastructure_restart':manifest.get('infrastructure_restart'),
        'entry_configuration':{k:v for k,v in (session.body_prior or {}).get('data',{}).get('dependencies',{}).items() if k in ('body','cameras','depth','execution','planner')},
        'termination':termination,
        'qualification':'Registered sources and captured public trace references through this ended-physics boundary, including commands, queries, failures and recovery. Later terminal events/costs have separate refs. Mechanical gaps are not inferred failures; no new raw/RGB read permission.'})
    with session.store.db:
        session.store.db.execute('INSERT INTO episode_records(run,task,chain_ref,summary) VALUES(?,?,?,?)',
            (session.episode,session.context['task'],chain['id'],encoded(summary)))


def seal(session, outline):
    initialize(session.store)
    row=session.store.db.execute('SELECT * FROM episode_records WHERE run=?',(session.episode,)).fetchone()
    if row is None or not getattr(session,'episode_ended',False):raise ValueError('Episode outline requires an ended episode')
    if not isinstance(outline,list):
        raise ValueError('Outline must be a list; an empty outline is valid')
    outline_bytes=len(json.dumps(outline,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode('utf-8'))
    if len(outline)>6 or outline_bytes>4096:
        raise ValueError('Outline must contain at most six stages and 4KiB of compact serialized stages; '
            f'actual_stages={len(outline)}/6; actual_bytes={outline_bytes}/4096; '
            f'excess_bytes={max(0,outline_bytes-4096)}. Canonical compact UTF-8 bytes include receipt IDs and all stage fields, excluding outer tool/event/context envelopes; '
            'shorten the interpretation, preserving material conditions and unknowns. An empty outline is valid.')
    chain=session.store.evidence_payload(row['chain_ref'])
    positions={r['id']:i for i,r in enumerate(chain['receipts'])}
    prior=-1
    for stage in outline:
        if not isinstance(stage,dict) or set(stage)-{'summary','first_receipt','last_receipt','conditions','unknowns'}:
            raise ValueError('Stage fields: summary, first_receipt, last_receipt, conditions, unknowns')
        if not isinstance(stage.get('summary'),str) or not stage['summary'].strip():raise ValueError('Stage summary required')
        start,end=positions.get(stage.get('first_receipt')),positions.get(stage.get('last_receipt'))
        if start is None or end is None or not prior<=start<=end:
            raise ValueError('Stage bounds must use this episode receipts with start<=end, ordered by nondecreasing first receipt; concurrent spans may overlap')
        prior=start
        if not isinstance(stage.get('conditions'),dict) or any(not isinstance(v,str) for v in stage['conditions'].values()):raise ValueError('Explicit stage conditions required')
        if not isinstance(stage.get('unknowns'),list) or any(not isinstance(v,str) for v in stage['unknowns']):raise ValueError('Explicit stage unknowns required')
    if row['outline_ref']:
        old=session.store.evidence_payload(row['outline_ref'])
        if old['stages']!=outline:raise ValueError('The pre-result outline is sealed; it cannot be rewritten after seeing the outcome')
        return row['outline_ref']
    if row['result_ref']:
        if outline:raise ValueError('A result was already published; no retrospective pre-result outline is permitted')
        return None
    ref=session.evidence('episode_outline',{'run':session.episode,'chain_ref':row['chain_ref'],
        'stages':outline,'interpretation':'controller_interpretation','reviewed':False,
        'qualification':'Unreviewed interpretation made before outcome delivery; stages are not certified successful skills.'})
    with session.store.db:session.store.db.execute('UPDATE episode_records SET outline_ref=?,outline_run=?,outline_sequence=?,has_outline=? WHERE run=?',
        (ref['id'],ref['id'].rsplit(':e',1)[0],ref['sequence'],bool(outline),session.episode))
    attach_result(session)
    return ref['id']


def attach_result(session):
    row=session.store.db.execute('SELECT * FROM episode_records WHERE run=?',(session.episode,)).fetchone()
    if row is None or row['result_ref']:return
    outcome=getattr(session,'_terminal_outcome',None)
    ref=session.evidence('episode_result',{'run':session.episode,'chain_ref':row['chain_ref'],
        'outline_ref':row['outline_ref'],'official_success':outcome,
        'qualification':'Terminal label binds the complete real trace only. It does not validate stages, causal explanations or a pruned/spliced path.'})
    with session.store.db:session.store.db.execute('UPDATE episode_records SET result_ref=?,result_run=?,result_sequence=?,official_success=? WHERE run=?',
        (ref['id'],ref['id'].rsplit(':e',1)[0],ref['sequence'],outcome,session.episode))


def attach_cost(session, primary, manager, primary_allowance, manager_allowance):
    row=session.store.db.execute('SELECT cost_ref FROM episode_records WHERE run=?',(session.episode,)).fetchone()
    if row is None or row['cost_ref']:return
    ref=session.evidence('episode_cost',{'run':session.episode,'primary':primary,'manager':manager,
        'primary_allowance':primary_allowance,'manager_allowance':manager_allowance,
        'qualification':'Actual cumulative role accounting, including terminal/rejected/failed requests; unknown remote usage is not zero. Token subcategories are not added twice.'})
    with session.store.db:session.store.db.execute('UPDATE episode_records SET cost_ref=? WHERE run=?',(ref['id'],session.episode))


def close(session, manifest, primary, manager, primary_allowance, manager_allowance, *, termination):
    """One mechanical archive even after interruption. No inference or action retry."""
    initialize(session.store)
    if session.store.db.execute('SELECT 1 FROM episode_records WHERE run=?',(session.episode,)).fetchone() is None:
        begin(session,manifest,None,termination)
    attach_result(session)
    attach_cost(session,primary,manager,primary_allowance,manager_allowance)


def read(store, *, task=None, question='', ids=None, frontier=None, limit=2, exclude_run=None, dependencies=None,contract_id=None,process_only=False):
    initialize(store)
    params={'task':task,'frontier':encoded(frontier or {}),'dependencies':encoded(dependencies or {}),'contract_id':contract_id}
    changed_sql='+'.join("CASE WHEN json_extract(summary,'$.entry_configuration."+k+"') IS NOT NULL AND "
        "json_extract(summary,'$.entry_configuration."+k+"') IS NOT json_extract(:dependencies,'$."+k+"') THEN 1 ELSE 0 END"
        for k in ('body','cameras','depth','execution','planner')) if dependencies is not None else '0+0'
    outcome_sql="official_success" if frontier is None else "CASE WHEN result_sequence<=COALESCE(json_extract(:frontier,'$.'||json_quote(result_run)),-1) THEN official_success ELSE NULL END"
    outline_sql="has_outline" if frontier is None else "CASE WHEN outline_sequence<=COALESCE(json_extract(:frontier,'$.'||json_quote(outline_run)),-1) THEN has_outline ELSE 0 END"
    where=[]
    if task is not None:where.append('task=:task')
    if ids is not None:
        where.append('run IN ('+','.join(':id'+str(i) for i in range(len(ids)))+')')
        params.update({'id'+str(i):v for i,v in enumerate(ids)})
    rows=store.db.execute('SELECT * FROM episode_records'+(' WHERE '+' AND '.join(where) if where else '')+
        " ORDER BY CASE WHEN :contract_id IS NOT NULL AND json_extract(summary,'$.contract_id')=:contract_id THEN 0 "
        "WHEN :contract_id IS NULL OR json_extract(summary,'$.contract_id') IS NULL THEN 1 ELSE 2 END,"+
        '('+changed_sql+') ASC, CASE WHEN ('+outcome_sql+')=1 AND ('+outline_sql+')=1 THEN 0 '
        'WHEN ('+outcome_sql+')=1 THEN 1 ELSE 2 END ASC, rowid DESC',params)
    terms=set(content_terms(question or ''));result=[]
    for row in rows:
        if row['run']==exclude_run:continue
        if ids is not None and row['run'] not in ids:continue
        def visible(source):
            r=store.db.execute('SELECT run,sequence FROM evidence WHERE id=?',(source,)).fetchone() if source else None
            return r is not None and (frontier is None or r['sequence']<=frontier.get(r['run'],-1))
        if not visible(row['chain_ref']):continue
        chain=json.loads(row['summary'])  # Never load whole historical per-tick chains in a decision hot path.
        if frontier is not None and any(seq>frontier.get(run,-1) for run,seq in chain['source_frontier'].items()):continue
        outline=store.evidence_payload(row['outline_ref']) if visible(row['outline_ref']) else None
        label=store.evidence_payload(row['result_ref']) if visible(row['result_ref']) else None
        cost=store.evidence_payload(row['cost_ref']) if visible(row['cost_ref']) else None
        if (process_only and ids is None and (outline is None or not outline['stages']) and chain['receipt_count']==0
                and chain['executed_actions']==0 and (label is None or label['official_success'] is None)):
            continue
        different=(contract_id!=chain['contract_id'] if contract_id is not None and chain['contract_id'] is not None else None)
        contract_rank=1 if different is None else (2 if different else 0)
        changed=[k for k,v in chain['entry_configuration'].items() if dependencies is not None and dependencies.get(k)!=v]
        text=row['task']+' '+encoded(outline['stages'] if outline else [])
        score=len(terms & set(content_terms(text)))
        if question and terms and not score:continue
        success=bool(label and label['official_success'] is True)
        process_rank=0 if success and outline and outline['stages'] else 1 if success else 2
        result.append(((contract_rank,len(changed),process_rank,-score),{'id':row['run'],'revision':int(row['chain_ref'].rsplit(':e',1)[1]),
            'layer':'episode_candidate','task':row['task'],'interpretation':'controller_interpretation' if outline else 'mechanical_index',
            'contract_id':chain['contract_id'],'contract_amendments':chain['contract_amendments'],
            'different_public_contract':different,
            'reviewed':False,'stages':outline['stages'] if outline else [],
            'official_success':label['official_success'] if label else None,
            'termination':chain['termination'],'receipt_count':chain['receipt_count'],'mechanical_gaps':chain['gap_counts'],
            'changed_entry_configuration':changed,
            'role_costs':{role:{k:cost[role].get(k) for k in ('cli_invocations','usage_reported','outcome_unknown',
                'usage_scope','captured_metadata_consistent','qualification')} for role in ('primary','manager')} if cost else None,
            'executed_actions':chain['executed_actions'],
            'replay':chain['prefix_replay'],'restart':chain['infrastructure_restart'],
            'trace_entry':{'chain':row['chain_ref'],'outline':row['outline_ref'] if outline else None,'result':row['result_ref'] if label else None,'cost':row['cost_ref'] if cost else None,
                           'first_receipt':chain['first_receipt']},
            'qualification':'Conditional candidate. Check current object, grasp, task and configuration; stage interpretations are unreviewed. Spans may overlap and are ordered by first receipt, not an executable sequence. No evidence-read or knowledge-support permission.'}))
        if not terms and len(result)>=limit:break
    result.sort(key=lambda item:item[0])
    if ids is not None:result.sort(key=lambda item:ids.index(item[1]['id']))
    return [item for _,item in result[:limit]]
