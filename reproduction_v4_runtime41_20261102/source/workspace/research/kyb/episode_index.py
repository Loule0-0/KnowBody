"""Deterministic navigation of public episode decisions, not physical skill labels."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

from .state import tracking_residual


KINDS = {'policy_decision', 'planning_result', 'receipt', 'observation', 'experience_query', 'surface_inspection'}


def _size(value):
    return len(json.dumps(value, ensure_ascii=False).encode())


def build(root: Path, *, through_sequence=None):
    """No model, semantic inference, evaluator, images or opaque native history."""
    root = Path(root)
    references = [json.loads(line) for line in (root / 'events.jsonl').read_text().splitlines()]
    if through_sequence is not None:
        references = [r for r in references if r['sequence'] <= through_sequence]
    events = []
    cutoff_time = 0.
    first_time = None
    for ref in references:
        # All boundary times are host trace times, never private/native reasoning.
        path = Path(ref['path'])
        if path.parent != root / 'events':
            continue
        if ref.get('visibility') == 'private' or ref['kind'] not in KINDS:
            continue
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != ref['sha256']:
            raise ValueError('Episode index source changed')
        event = json.loads(raw)
        cutoff_time = max(cutoff_time, event['wall_time'])
        first_time = event['wall_time'] if first_time is None else min(first_time, event['wall_time'])
        if ref['kind'] in KINDS:
            events.append(event)
    attempts = {}
    ledger = root / 'cli_calls.jsonl'
    for line in ledger.read_text().splitlines() if ledger.exists() else []:
        row = json.loads(line)
        if row['role'] != 'policy':
            raise ValueError('Episode index requires the primary-only native ledger')
        if row['time'] <= cutoff_time:
            attempts[row['attempt_id']] = row
    completed = {key: value for key, value in attempts.items() if value['state'] == 'turn_complete'}
    available = set(completed)
    decisions, questions = [], []
    for event in events:
        if event['kind'] != 'policy_decision':
            continue
        payload = event['payload']
        attempt = payload.get('call_id')
        ledger_match = 'exact_id' if attempt in available else 'not_confirmed'
        if attempt is None:
            prior = [(row['time'], key) for key, row in completed.items()
                     if key in available and row['time'] <= event['wall_time']]
            attempt = max(prior)[1] if prior else None
            ledger_match = 'historical_time_order' if attempt else 'not_confirmed'
        if attempt in available:
            available.remove(attempt)
        item = {'event': event['id'], 'sequence': event['sequence'],
                'attempt': attempt, 'ledger_match':ledger_match, 'tool': payload['tool'],
                'phase': payload.get('phase', 'online'),
                'observation': payload.get('observation_id'),
                'preview_requested': payload.get('arguments', {}).get('preview') is True}
        decisions.append(item)
        question = payload.get('arguments', {}).get('question')
        if item['phase'] != 'terminal_review' and question:
            questions.append({'event': event['id'], 'sequence': event['sequence'],
                              'question': question, 'tool': item['tool'],
                              'purpose':payload.get('arguments',{}).get('purpose')})
    online = [d for d in decisions if d['phase'] != 'terminal_review']
    terminal = [d for d in decisions if d['phase'] == 'terminal_review']
    observations = {e['id']: e['payload'] for e in events if e['kind'] == 'observation'}
    posts = {}
    for event in events:
        if event['kind'] == 'observation' and event['payload'].get('after_receipt'):
            posts.setdefault(event['payload']['after_receipt'], event['payload'])
    receipts = [e for e in events if e['kind'] == 'receipt']
    distributions = Counter()
    distances = []
    windows = []
    # Eight decisions per chronological window prevents one-receipt/one-row degeneration.
    # These are navigation windows, never inferred task phases or positive outcomes.
    for start in range(0, len(online), 8):
        group = online[start:start + 8]
        ids = {d['attempt'] for d in group if d['attempt']}
        linked = [e for e in receipts if e['payload'].get('call_id') in ids]
        positive = {e['payload']['call_id'] for e in linked if e['payload']['executed_actions'] > 0}
        failures = [e for e in events if e['kind'] == 'planning_result'
                    and group[0]['sequence'] <= e['sequence']
                    <= (online[start + 8]['sequence'] - 1 if start + 8 < len(online) else references[-1]['sequence'])
                    and e['payload'].get('status') != 'Success']
        end = online[start + 8]['sequence'] - 1 if start + 8 < len(online) else references[-1]['sequence']
        stop_reasons = Counter(e['payload'].get('plan', {}).get('selected_stop_reason', 'unspecified') for e in linked)
        amplitudes = []
        for event in linked:
            data = event['payload']
            before = observations.get(data.get('before_observation'))
            difference = tracking_residual(data, before) if before else {'measured_differences':{}}
            for detail in difference['measured_differences'].values():
                xyz = detail.get('grasp_position_current_minus_source_m')
                if xyz is not None:amplitudes.append(math.sqrt(sum(value ** 2 for value in xyz)))
        # Every motion decision is counted once, including requests with no receipt.
        for decision in group:
            if decision['tool'] not in ('move_eef', 'hold'):
                continue
            matched = [e['payload'] for e in linked if e['payload'].get('call_id') == decision['attempt']]
            if not matched:
                distributions['preview_requested_no_receipt' if decision['preview_requested'] else 'no_receipt_motion_request'] += 1
            else:
                for receipt in matched:
                    reason = receipt.get('plan', {}).get('selected_stop_reason', 'unspecified')
                    distributions[reason] += 1
        distances.extend(amplitudes)
        window = {'id': f'w{len(windows) + 1:02d}',
                  'events': [group[0]['sequence'], end],
                  'calls': len(group), 'tools': dict(Counter(d['tool'] for d in group)),
                  'motion_without_positive_receipt': sum(d['tool'] in ('move_eef', 'hold')
                                                        and d['attempt'] not in positive for d in group),
                  'planner_non_success': len(failures),
                  'executed_actions': sum(e['payload']['executed_actions'] for e in linked),
                  'receipt_counts': dict(Counter(e['payload']['execution'] for e in linked)),
                  'zero_action_receipts': sum(e['payload']['executed_actions'] == 0 for e in linked),
                  'selected_stop_reasons': dict(stop_reasons),
                  'requested_world_translation_m': _distribution(amplitudes),
                  'receipt_refs': [e['id'] for e in linked],
                  'measurement_sequences':[e['sequence'] for e in events if e['kind']=='surface_inspection'
                       and group[0]['sequence']<=e['sequence']<=end],
                  'read_trajectory_after': min((e['sequence'] for e in linked), default=group[0]['sequence']) - 1}
        windows.append(window)
    residuals = []
    for event in receipts:
        receipt = event['payload']
        post = posts.get(event['id'])
        if post is None:
            continue
        plan = receipt.get('plan', {})
        scope = plan.get('selected_stop_reason', 'unspecified')
        # A deliberately selected prefix is not an arrival failure against the full target.
        if (scope != 'full_plan_selected' or receipt.get('execution') != 'complete'
            or receipt.get('unknown_command') is not None or not receipt.get('executed_actions')
            or receipt.get('executed_actions') != plan.get('selected_actions')):
            continue
        delta = tracking_residual(receipt, post)
        values = {}
        for arm, difference in delta['measured_differences'].items():
            xyz = difference.get('grasp_position_current_minus_source_m')
            if xyz is not None:
                values[arm] = sum(v * v for v in xyz) ** .5
        if values:
            residuals.append({'receipt': event['id'], 'sequence': event['sequence'],
                              'selection_scope': scope, 'position_residual_m': values,
                              'coordinate_comparison': delta['coordinate_comparison']})
    residuals.sort(key=lambda row: (-max(row['position_residual_m'].values()), row['sequence']))
    # Query answers are only navigation: the actual records must still be read.
    for question in questions:
        next_sequence = next((d['sequence'] for d in decisions if d['sequence'] > question['sequence']),
                             references[-1]['sequence'] + 1)
        views = Counter()
        for event in events:
            if event['kind'] == 'experience_query' and question['sequence'] < event['sequence'] < next_sequence:
                answer = event['payload'].get('answer', {})
                views.update(card.get('view', 'unspecified') for card in answer.get('records', []))
        question['delivered_views'] = dict(views)
    online_counts = dict(Counter(d['tool'] for d in online))
    terminal_counts = dict(Counter(d['tool'] for d in terminal))
    carried = sum(completed[key]['time'] < (first_time or 0) for key in available)
    accounting = {'recorded_attempts': len(attempts),
                  'states': dict(Counter(row['state'] for row in attempts.values())),
                  'online_tools': online_counts, 'terminal_tools': terminal_counts,
                  'mapped_completed_decisions': sum(d['ledger_match'] != 'not_confirmed' for d in decisions),
                  'historical_time_matched_decisions': sum(d['ledger_match'] == 'historical_time_order' for d in decisions),
                  'host_decisions_without_completed_attempt': sum(d['ledger_match'] == 'not_confirmed' for d in decisions),
                  'carried_completed_attempts': carried,
                  'completed_without_host_decision': len(available) - carried,
                  'motion_selection': dict(distributions),
                  'requested_world_translation_m': _distribution(distances),
                  'planner_non_success': sum(e['kind'] == 'planning_result' and e['payload'].get('status') != 'Success' for e in events),
                  'receipts': len(receipts),
                  'executed_actions': sum(e['payload']['executed_actions'] for e in receipts),
                  'replayed_actions': sum(e['payload']['executed_actions'] for e in receipts
                                          if e['payload'].get('call_id') is None)}
    decision_tools={d['attempt']:d['tool'] for d in decisions if d['attempt']}
    steps_by_tool=Counter()
    holds=[]
    for receipt in receipts:
        p=receipt['payload'];tool=decision_tools.get(p.get('call_id'),'unattributed')
        steps_by_tool[tool]+=p['executed_actions']
        if tool=='hold':holds.append({'receipt':receipt['id'],'sequence':receipt['sequence'],
                                     'executed_actions':p['executed_actions']})
    accounting['environment_steps_by_tool']=dict(steps_by_tool)
    holds.sort(key=lambda row:(-row['executed_actions'],row['sequence']))
    return {'run': root.name, 'through_sequence': references[-1]['sequence'] if references else 0,
            'accounting': accounting, 'windows': windows, 'memory_questions': questions,
            'largest_full_plan_residuals': residuals,'largest_holds':holds,
            'qualification': 'Navigation only, not read evidence. Requested world displacement is not measured motion. '
                             'Counts and complete execution do not prove contact, holding or mechanism. Replay is not a new trial.'}


def _distribution(values):
    return {'samples': len(values), 'min': min(values), 'median': statistics.median(values), 'max': max(values)} if values else {'samples': 0}


def directory(store,run,after,through,kind,*,budget_bytes=6000):
    """A page of registered public pointers, never raw read permission."""
    if kind=='questions':
        rows=store.db.execute("SELECT id,sequence,payload FROM evidence WHERE run=? AND kind='policy_decision' AND sequence>? AND sequence<=? "
            "AND json_extract(payload,'$.arguments.question') IS NOT NULL ORDER BY sequence",(run,after,through))
    elif kind=='leads':
        rows=store.db.execute('SELECT id,sequence,payload FROM learning_leads WHERE run=? AND sequence>? AND sequence<=? ORDER BY sequence',
            (run,after,through))
    elif kind=='motion':
        rows=store.db.execute("SELECT id,sequence,kind,payload FROM evidence WHERE run=? AND sequence>? AND sequence<=? "
            "AND (kind IN ('planning_result','path_validation','planning_feedback') OR (kind='policy_decision' "
            "AND json_extract(payload,'$.tool') IN ('move_eef','hold'))) ORDER BY sequence",(run,after,through))
    elif kind=='measurements':
        rows=store.db.execute("SELECT id,sequence,payload FROM evidence WHERE run=? AND kind='surface_inspection' "
            "AND sequence>? AND sequence<=? ORDER BY sequence",(run,after,through))
    else:raise ValueError('Unknown trajectory directory')
    answer={'run':run,'through':through,'directory':kind,'entries':[],'next_after':None,
        'qualification':'Navigation only, not read evidence or resolved questions. Planner/path status is not execution, physical contact or feasibility.' if kind=='motion' else 'Navigation only, not read evidence or resolved questions.'}
    if kind=='measurements':
        answer['qualification']='Navigation only. Labels and cameras are original sampling requests, not certified object identity, geometry or read permission. Read the source observation and measurement before using a physical claim.'
    if _size(answer)>budget_bytes:raise ValueError('Directory budget cannot fit its envelope')
    for row in rows:
        data=json.loads(row['payload'])
        record={'id':row['id'],'sequence':row['sequence']}
        if kind=='motion':
            record.update(kind=row['kind'])
            record.update({'tool':data['tool']} if row['kind']=='policy_decision' else {'status':data.get('status')})
        elif kind=='measurements':
            observation=data.get('before_observation_id')
            if not isinstance(observation,str) or not observation:
                raise ValueError('Measurement requires an earlier registered observation in the same visible run')
            source=store.db.execute('SELECT run,sequence,kind FROM evidence WHERE id=?',(observation,)).fetchone()
            if source is None or source['run']!=run or source['kind']!='observation' or not source['sequence']<row['sequence']<=through:
                raise ValueError('Measurement requires an earlier registered observation in the same visible run')
            samples=data.get('samples')
            if not isinstance(samples,list) or any(not isinstance(s,dict) or any(
                    not isinstance(s.get(k),str) for k in ('label','camera')) for s in samples):
                raise ValueError('Measurement navigation requires the original sample labels and cameras')
            record.update(observation_id=observation,samples=[{k:s[k] for k in ('label','camera')} for s in samples])
        else:record['question_excerpt']=data.get('arguments',{}).get('question','')[:180] if kind=='questions' else data.get('question','')[:180]
        candidate={**answer,'entries':[*answer['entries'],record]}
        # Account for both the final null and a continued cursor, including
        # UTF-8 and escaped identifiers. Never split a navigation entry.
        if max(_size(candidate),_size({**candidate,'next_after':row['sequence']}))>budget_bytes:
            if not answer['entries']:raise ValueError('Directory budget cannot fit one complete navigation entry')
            answer['next_after']=answer['entries'][-1]['sequence']
            break
        answer=candidate
    return answer


def readout(index, *, budget_bytes=2800, completed=False):
    """Keep accounting intact and pack whole navigation entries, never cut conditions."""
    result = {key: index[key] for key in ['run', 'through_sequence', 'accounting', 'qualification']}
    result.update(windows=[], memory_questions=[], largest_full_plan_residuals=[],
                  total_windows=len(index['windows']), omitted_windows=0, omitted_questions=0)
    result['question_sequences']=[q['sequence'] for q in index['memory_questions'][:32]]
    result['omitted_question_sequences']=max(0,len(index['memory_questions'])-32)
    result['question_id_format']=index['run']+':e{sequence:06d}'
    result['question_directory']={'run':index['run'],'after':0,'through':index['through_sequence'],'directory':'questions'}
    result['read_trajectory']={'run':index['run'],'through':index['through_sequence'],'include_processed':True}
    result['continuation_instruction']='Set after to a question sequence or window read_trajectory_after; navigation only.'
    if _size(result) > budget_bytes:
        raise ValueError('Episode index budget cannot hold its accounting contract')
    window_keys=('id','events','calls','tools','executed_actions','read_trajectory_after','measurement_sequences')
    protected_window=None;reserved_cost_bytes=0
    if completed and any(w['executed_actions']>0 for w in index['windows']):
        highest=max(index['windows'],key=lambda w:w['executed_actions'])
        compact={k:highest[k] for k in window_keys}
        before=_size(result)
        result['windows']=[compact];result['omitted_windows']=len(index['windows'])-1
        if _size(result)>budget_bytes:
            raise ValueError('Episode index budget cannot hold the complete highest-execution-cost navigation window')
        reserved_cost_bytes=_size(result)-before
        protected_window=highest['id']
    # Explicit comparisons precede ordinary lookups. All question IDs remain retrievable.
    ordered=sorted(index['memory_questions'],key=lambda q:(q['tool']!='manage_memory',q['sequence']))
    for question in ordered:
        compact = {k:v for k,v in question.items() if k not in ('delivered_views','event')}
        compact.update(question=question['question'][:180],excerpt=len(question['question'])>180)
        candidate = {**result, 'memory_questions': [*result['memory_questions'], compact]}
        if _size(candidate) <= budget_bytes - max(0,650-reserved_cost_bytes):
            result = candidate
        else:
            result['omitted_questions'] += 1
    if index['largest_holds']:
        candidate={**result,'largest_hold':index['largest_holds'][0]}
        if _size(candidate)<=budget_bytes-max(0,400-reserved_cost_bytes):result=candidate
    for residual in index['largest_full_plan_residuals'][:1]:
        candidate = {**result, 'largest_full_plan_residuals': [residual]}
        if _size(candidate) <= budget_bytes - max(0,400-reserved_cost_bytes):
            result = candidate
    windows = index['windows']
    order = []
    if windows:
        diagnostics=[]
        for question in ordered:
            if question['tool']!='manage_memory':continue
            for i,w in enumerate(windows):
                if w['events'][0]<=question['sequence']<=w['events'][1]:
                    diagnostics.extend(windows[i:i+2]);break
        order = [*diagnostics,windows[0], max(windows, key=lambda w: (w['motion_without_positive_receipt'],
                 w['tools'].get('inspect_surface', 0), w['calls'])), windows[-1], *windows]
    seen = {protected_window} if protected_window is not None else set()
    for window in order:
        if window['id'] in seen:
            continue
        seen.add(window['id'])
        compact = {k:window[k] for k in window_keys}
        candidate = {**result, 'windows': [*result['windows'], compact]}
        if _size(candidate) <= budget_bytes:
            result = candidate
    result['omitted_windows'] = len(windows) - len(result['windows'])
    # omitted counts may add a digit; the bound applies to the actual serialized packet.
    while _size(result) > budget_bytes and result['windows']:
        if result['windows'][-1]['id']==protected_window:
            raise ValueError('Episode index budget cannot hold its accounting and protected cost navigation')
        result['windows'].pop()
        result['omitted_windows'] += 1
    motion=index['accounting'].get('motion_selection',{})
    if motion.get('no_receipt_motion_request',0) or motion.get('preview_requested_no_receipt',0):
        candidate={**result,'motion_directory':{'run':index['run'],'after':0,'through':index['through_sequence'],'directory':'motion'}}
        if _size(candidate)<=budget_bytes:result=candidate
    return result
