"""An explicit operator amendment to a genuinely incomplete public task contract."""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from .knowledge import function
from .trace import write_once
from .contracts import current_contract


class ClarificationStopped(RuntimeError):
    pass


def _state(path, value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    os.replace(temporary,path)


def tool():
    return function('request_task_clarification',
        'Only for genuinely missing or conflicting PUBLIC task requirements. Quote the current instruction/recipe, '
        'state competing goal interpretations and the one missing fact. Not for perception, grasp strategy, '
        'hidden evaluator information or an easier task. This suspends the live scene with zero actions/model calls until the operator answers or stops.',
        {'question':{'type':'string','maxLength':800},'missing_information':{'type':'string','maxLength':800},
         'interpretations':{'type':'array','minItems':2,'maxItems':3,'items':{'type':'string','maxLength':600}},
         'quotes':{'type':'array','minItems':1,'maxItems':3,'items':{'type':'object','properties':{
             'source_id':{'type':'string'},'quote':{'type':'string','maxLength':1000}},
             'required':['source_id','quote'],'additionalProperties':False}}},
        ['question','missing_information','interpretations','quotes'])


def request(session, arguments, *, poll_seconds=1.):
    if session.episode_ended:raise ValueError('Task clarification is unavailable after episode end')
    if set(arguments)!={'question','missing_information','interpretations','quotes'}:raise ValueError('Explicit contract ambiguity fields required')
    for key in ('question','missing_information'):
        if not isinstance(arguments[key],str) or not 0<len(arguments[key].strip())<=800:raise ValueError('A concise missing-contract question is required')
    alternatives=arguments['interpretations']
    if not isinstance(alternatives,list) or not 2<=len(alternatives)<=3 or any(not isinstance(s,str) or not 0<len(s.strip())<=600 for s in alternatives) or len(set(alternatives))!=len(alternatives):raise ValueError('Two or three distinct goal interpretations are required')
    refs=[session.task_specification['id']] if session.task_specification else []
    refs.extend(getattr(session,'contract_amendments',[]))
    quotes=arguments['quotes']
    if not isinstance(quotes,list) or not 1<=len(quotes)<=3:raise ValueError('Quote exact public task sources')
    for quote in quotes:
        if not isinstance(quote,dict) or set(quote)!={'source_id','quote'} or quote['source_id'] not in refs:raise ValueError('Quote a supplied current task specification/amendment')
        source=session.store.evidence_payload(quote['source_id'])
        text='\n'.join(v for k in ('instruction','public_recipe','answer') if isinstance((v:=source.get(k)),str))
        if not isinstance(quote['quote'],str) or not 0<len(quote['quote'])<=1000 or quote['quote'] not in text:raise ValueError('Quoted text is absent from the public contract')
    ref=session.evidence('task_clarification_request',{'task_specification':session.task_specification['id'],
        'contract_id':current_contract(session),
        'observation_id':session.current_observation['id'] if session.current_observation else None,
        **arguments,'qualification':'Controller-declared ambiguity. Quote presence is checked; the host does not certify semantic ambiguity.'})
    folder=session.trace.root/'clarification';folder.mkdir(exist_ok=True)
    answer_path=folder/(ref['id'].rsplit(':',1)[1]+'.answer.json')
    state={'status':'WAITING_FOR_CLARIFICATION','request_id':ref['id'],'request_path':ref['path'],
           'answer_path':str(answer_path),'simulation_actions_while_waiting':0,'model_calls_while_waiting':0}
    pending=folder/'STATE.json'
    _state(pending,state)
    rejected=set()
    while True:
        if session.stop_requested():
            state['status']='STOPPED_BY_OPERATOR';_state(pending,state)
            raise ClarificationStopped('Operator stopped the live clarification wait; no automatic replay')
        if answer_path.exists():
            raw=answer_path.read_bytes()
            try:
                answer=json.loads(raw)
                if not isinstance(answer,dict) or set(answer)!={'request_id','answer','provided_by','reuse_scope'} or answer['request_id']!=ref['id'] or answer['provided_by']!='operator' or answer['reuse_scope'] not in ('episode','same_public_contract') or not isinstance(answer['answer'],str) or not 0<len(answer['answer'].strip().encode())<=4096:raise ValueError('Invalid operator amendment')
                break
            except (ValueError,UnicodeError):
                sha=hashlib.sha256(raw).hexdigest()
                if sha not in rejected:
                    session.trace.event('clarification_answer_rejected',{'request_id':ref['id'],'answer_sha256':sha,
                        'reason':'Malformed or mismatched operator amendment; no actions/model calls; live scene still waiting'})
                    rejected.add(sha)
        time.sleep(poll_seconds)
    amendment=session.evidence('task_contract_amendment',{'request_id':ref['id'],
        'contract_id':current_contract(session),'episode':session.episode,'question':arguments['question'],
        'parent_contract':session.task_specification['id'],'prior_amendments':refs[1:],**answer,'answer_file_sha256':hashlib.sha256(raw).hexdigest(),
        'qualification':'Operator-provided public contract clarification; not a changed evaluator or physical observation.'})
    if not hasattr(session,'contract_amendments'):session.contract_amendments=[]
    session.contract_amendments.append(amendment['id'])
    state.update(status='ANSWERED',amendment_id=amendment['id'])
    _state(pending,state)
    return json.dumps({'status':'PASS','contract_amendment':amendment['id'],'answer':answer['answer'],
        'executed_actions':0,'qualification':'Continue from the unchanged live observation and cumulative budget.'},ensure_ascii=False)


def answer(run, text, scope='episode'):
    state=json.loads((run/'clarification'/'STATE.json').read_text())
    if state['status']!='WAITING_FOR_CLARIFICATION':raise ValueError('No pending live clarification')
    if not isinstance(text,str) or not 0<len(text.strip().encode())<=4096:raise ValueError('Operator answer must be 1..4096 UTF8 bytes')
    if scope not in ('episode','same_public_contract'):raise ValueError('Explicit answer scope required')
    payload={'request_id':state['request_id'],'answer':text,'provided_by':'operator','reuse_scope':scope}
    inbox=Path(state['answer_path'])
    archive=inbox.parent/('operator_answer_'+uuid.uuid4().hex+'.json')
    write_once(archive,payload)
    temporary=inbox.parent/('inbox_'+uuid.uuid4().hex+'.tmp')
    temporary.write_bytes(archive.read_bytes())
    os.replace(temporary,inbox)
