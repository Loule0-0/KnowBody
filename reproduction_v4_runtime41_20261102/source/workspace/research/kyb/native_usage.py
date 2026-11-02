"""Reported native response usage only; never inspect reasoning or summary text.

Rollout archives contain cumulative copies of earlier events. Count a response
once per native session, and classify compression only by its explicit link.
Neither reported tokens nor visible reconnect notices establish a final bill.
"""
from pathlib import Path
import json
import re

FIELDS=('input_tokens','cached_input_tokens','cache_write_input_tokens',
        'output_tokens','reasoning_output_tokens')


def token_values(value):
    if not isinstance(value,dict):raise ValueError('Missing native usage object')
    if any(type(value.get(k)) is not int or value[k]<0 for k in FIELDS):
        raise ValueError('Native usage requires five nonnegative integer counters')
    result={k:value[k] for k in FIELDS}
    if (result['cached_input_tokens']>result['input_tokens'] or
            result['reasoning_output_tokens']>result['output_tokens']):
        raise ValueError('Native usage subset exceeds its parent counter')
    if 'total_tokens' in value and value['total_tokens']!=result['input_tokens']+result['output_tokens']:
        raise ValueError('Native total tokens differ from input plus output')
    return result


def add_usage(target,value):
    for key,n in value.items():target[key]=target.get(key,0)+n


def observed_total(summary):
    """Arithmetic on reported counters, without certifying their completeness."""
    usage=summary.get('usage_reported',{})
    values=[usage.get(key) for key in ('input_tokens','output_tokens')]
    return sum(values) if all(type(n) is int and n>=0 for n in values) else None


def reported_total(summary):
    """A complete reported total needs explicit consistent metadata, not silence."""
    return observed_total(summary) if summary.get('captured_metadata_consistent') is True else None


def read_rollout_usage(path):
    """Return metadata, excluding every response/compaction message body."""
    out={'records':[],'compactions':[],'thread_totals':{},'issues':[]}
    contexts={}
    session_id=None
    with Path(path).open() as stream:
        for number,line in enumerate(stream,1):
            try:event=json.loads(line)
            except ValueError:
                out['issues'].append({'code':'malformed_native_line','line':number});continue
            if not isinstance(event,dict):
                out['issues'].append({'code':'invalid_native_event','line':number});continue
            kind=event.get('type')
            if kind not in ('session_meta','turn_context','token_usage_record','compacted'):continue
            data=event.get('payload',{})
            if not isinstance(data,dict):
                out['issues'].append({'code':'invalid_native_metadata','line':number});continue
            if kind=='session_meta':
                session_id=data.get('session_id') or data.get('id')
                continue
            if kind=='turn_context':
                if isinstance(data.get('turn_id'),str):
                    contexts[data['turn_id']]={'model':data.get('model'),'effort':data.get('effort')}
                continue
            if kind=='compacted':
                response=data.get('compaction_response_id')
                if isinstance(response,str) and response and isinstance(session_id,str):
                    out['compactions'].append([session_id,response])
                else:out['issues'].append({'code':'missing_compaction_response_id','line':number})
                continue
            sid=data.get('session_id');response=data.get('response_id')
            if (not isinstance(sid,str) or not sid or data.get('thread_id')!=sid or
                    not isinstance(response,str) or not response):
                out['issues'].append({'code':'invalid_native_response_identity','line':number});continue
            try:
                usage=token_values(data.get('usage'))
            except ValueError as error:
                out['issues'].append({'code':'invalid_native_usage','line':number,'reason':str(error)});continue
            out['records'].append({'session_id':sid,'response_id':response,
                'turn_id':data.get('turn_id'),'root_turn_id':data.get('root_turn_id'),'usage':usage})
            try:out['thread_totals'][sid]=token_values(data.get('thread_token_usage'))
            except ValueError as error:
                out['issues'].append({'code':'invalid_native_thread_total','line':number,'reason':str(error)})
    for record in out['records']:
        context=contexts.get(record['turn_id']) or contexts.get(record['root_turn_id'])
        record.update(context or {'model':None,'effort':None})
    return out


def merge_usage_snapshots(snapshots):
    """Union captured responses; do not sum copied rollouts or cumulative totals."""
    records={};compactions=set();totals={};issues=[];conflicts=set();attribution_conflicts=set()
    for snapshot in snapshots:
        issues.extend(snapshot.get('issues',[]))
        compactions.update(tuple(p) for p in snapshot.get('compactions',[]))
        for sid,total in snapshot.get('thread_totals',{}).items():
            old=totals.get(sid)
            if old is None or all(total[k]>=old[k] for k in FIELDS):totals[sid]=total
            elif not all(old[k]>=total[k] for k in FIELDS):
                issues.append({'code':'incomparable_native_thread_totals','session_id':sid})
        for record in snapshot.get('records',[]):
            key=(record['session_id'],record['response_id'])
            old=records.get(key)
            if old is not None and any(old.get(k)!=record.get(k) for k in
                    ('usage','turn_id','root_turn_id')):
                conflicts.add(key)
                issues.append({'code':'conflicting_native_response','session_id':key[0],'response_id':key[1]})
            elif old is not None:
                for field in ('model','effort'):
                    if old.get(field) is not None and record.get(field) is not None and old[field]!=record[field]:
                        attribution_conflicts.add(key)
                        issues.append({'code':'conflicting_native_model_context','session_id':key[0],'response_id':key[1]})
                    elif record.get(field) is not None:old[field]=record[field]
            else:records[key]=dict(record)
    for key in conflicts:records.pop(key,None)
    for key in attribution_conflicts:
        if key in records:records[key].update(model=None,effort=None)
    all_usage={};compressed={};other={};by_session={};models={};compression_count=0
    for record in records.values():
        value=record['usage'];add_usage(all_usage,value)
        is_compaction=(record['session_id'],record['response_id']) in compactions
        add_usage(compressed if is_compaction else other,value)
        compression_count+=int(is_compaction)
        add_usage(by_session.setdefault(record['session_id'],{}),value)
        key=(record.get('model'),record.get('effort'))
        model=models.setdefault(key,{'model':key[0],'effort':key[1],'usage_reported':{},
            'compaction_usage_reported':{},'reported_native_responses':0})
        add_usage(model['usage_reported'],value)
        if is_compaction:add_usage(model['compaction_usage_reported'],value)
        model['reported_native_responses']+=1
    for sid,total in totals.items():
        if by_session.get(sid)!=total:issues.append({'code':'native_thread_total_mismatch','session_id':sid})
    missing=compactions-set(records)
    if missing:issues.append({'code':'compaction_usage_not_reported','responses':sorted(missing)})
    if not records:issues.append({'code':'no_reported_native_response_usage'})
    return {'usage_reported':all_usage,'compaction_usage_reported':compressed,
        'non_compaction_usage_reported':other,'reported_native_responses':len(records),
        'reported_compaction_responses':compression_count,'models':list(models.values()),
        'usage_issues':issues,'captured_metadata_consistent':not issues,
        'usage_scope':'native_response_usage_deduplicated_by_session_and_response',
        'qualification':'Sum usage once; cached input and reasoning output are subsets. '
            'Model attribution follows the native turn context, not independently reported compactor billing. '
            'Internal failed requests without usage and actual charges remain unknown.'}


def transport_notices(events):
    """Only public transport notices, never a count of actual billed requests."""
    ordinals=set();notices=0;https=False
    for event in events:
        message=None
        if event.get('type')=='error':message=event.get('message')
        elif event.get('item',{}).get('type')=='error':message=event['item'].get('message')
        if not isinstance(message,str):continue
        match=re.search(r'Reconnecting(?:\.{3}|…)?\s+(\d+)\s*/\s*(\d+)',message,re.I)
        if match:
            notices+=1;ordinals.add((int(match[1]),int(match[2])))
        if 'falling back from websockets to https' in message.lower():https=True
    return {'reconnect_notices':notices,'reconnect_ordinals':[list(p) for p in sorted(ordinals)],
        'https_fallback':https,'actual_transport_or_model_request_count':None}


def summarize_role(root):
    """Meter one role from immutable CLI archives; usable for a read-only audit.

    Select the latest cumulative archive per session. Stdout's cumulative turn
    counters are retained separately for reconciliation, never used as a silent
    replacement for missing native response records.
    """
    root=Path(root);ledger=root/'cli_calls.jsonl';attempts={}
    for line in ledger.read_text().splitlines() if ledger.exists() else []:
        event=json.loads(line);attempts.setdefault(event['attempt_id'],[]).append(event)
    latest={};stdout={};issues=[];invocations={};transport=[]
    for attempt,events in attempts.items():
        directory=root/'cli'/attempt;terminal=events[-1]
        invocation_path=directory/'invocation.json'
        invocation=json.loads(invocation_path.read_text()) if invocation_path.exists() else {}
        key=(invocation.get('model'),invocation.get('effort'))
        invocations[key]=invocations.get(key,0)+1
        sid=terminal.get('thread_id')
        if sid:
            latest[sid]=(directory,terminal)
            if terminal.get('usage'):stdout[sid]=terminal['usage']
        elif any(e['state']=='process_started' for e in events):
            issues.append({'code':'dispatched_attempt_missing_session','attempt_id':attempt})
        public=[];path=directory/'events.jsonl'
        if path.exists():
            for line in path.read_text().splitlines():
                try:event=json.loads(line)
                except ValueError:continue
                if event.get('type')=='error' or event.get('item',{}).get('type')=='error':public.append(event)
        notices=transport_notices(public)
        if notices['reconnect_notices'] or notices['https_fallback']:
            transport.append({'attempt_id':attempt,**notices})
    snapshots=[]
    for sid,(directory,terminal) in latest.items():
        path=directory/'native_rollout.jsonl'
        if not path.exists():
            issues.append({'code':'native_archive_unavailable','session_id':sid});continue
        try:
            snapshot=read_rollout_usage(path)
        except (OSError,ValueError,TypeError) as error:
            issues.append({'code':'native_archive_unreadable','session_id':sid,'error_type':type(error).__name__});continue
        if any(r['session_id']!=sid for r in snapshot['records']):
            issues.append({'code':'native_archive_session_mismatch','session_id':sid});continue
        expected=sum(e[-1].get('native_completed_turns',0) for e in attempts.values()
            if e[-1].get('thread_id')==sid)
        unique={r['response_id'] for r in snapshot['records']}
        if len(unique)<expected:
            issues.append({'code':'native_response_coverage_below_completed_turns','session_id':sid})
        compression={tuple(pair) for pair in snapshot['compactions']};plain={};seen=set()
        for record in snapshot['records']:
            identity=(record['session_id'],record['response_id'])
            if identity not in compression and identity not in seen:add_usage(plain,record['usage'])
            seen.add(identity)
        if any(type(v) is int and plain.get(k,0)<v for k,v in stdout.get(sid,{}).items()):
            issues.append({'code':'native_usage_below_stdout_reported_counter','session_id':sid})
        snapshots.append(snapshot)
    meter=merge_usage_snapshots(snapshots)
    meter['usage_issues'].extend(issues)
    meter['captured_metadata_consistent']=not meter['usage_issues']
    groups={(m['model'],m['effort']):m for m in meter['models']}
    for key,n in invocations.items():
        group=groups.setdefault(key,{'model':key[0],'effort':key[1],
            'usage_reported':{},'compaction_usage_reported':{},'reported_native_responses':0})
        group['cli_invocations']=n
    meter['models']=list(groups.values())
    stdout_total={}
    for value in stdout.values():add_usage(stdout_total,{k:v for k,v in value.items() if type(v) is int})
    meter.update(stdout_usage_reported=stdout_total,
        stdout_usage_scope='Latest stdout cumulative per session; excludes separately reported compression',
        transport_diagnostics={'reconnect_notices':sum(e['reconnect_notices'] for e in transport),
            'attempts_with_https_fallback':sum(e['https_fallback'] for e in transport),
            'affected_cli_attempts':transport,'actual_transport_or_model_request_count':None})
    complete=[es[-1] for es in attempts.values() if es[-1]['state']=='turn_complete']
    meter.update(cli_invocations=len(attempts),
        completed_turns=sum(es[-1].get('native_completed_turns',0) for es in attempts.values()),
        structured_responses_accepted=len(complete),
        outcome_unknown=sum(es[-1]['state']=='outcome_unknown' for es in attempts.values()),
        native_failed_turns=sum(es[-1]['state']=='turn_failed' for es in attempts.values()),
        http_attempts=None,internal_model_request_count=None,bill_amount=None)
    return meter
