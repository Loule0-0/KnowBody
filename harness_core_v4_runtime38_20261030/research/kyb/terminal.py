"""A policy-owned, motion-free review of a completed episode."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

from .knowledge import function, policy_memory_tools, outline_schema
from .session import MemorySession
from .state import evidence_state
from . import episode_index


INSTRUCTIONS = '''The robot episode has ended. Review its final observable outcome and decide whether
there is useful experience to retain for future tasks. You have no motion tools and cannot resume the robot.
Use query_experience for one concrete historical question. If a worthwhile batch needs extraction or
comparison, you alone may trigger manage_memory with a small question and purpose learn or review.
Include useful positive and neutral responses as well as failures; do not manufacture a lesson merely to
exercise the interface. If the endpoint images cannot establish a lesson, the manager can examine earlier
registered interactions. A worthwhile unexplained response difference is enough to ask for comparison;
you need not prove a causal rule before asking, and the manager may find insufficient evidence to write.
The manager finds the source records; you need not choose a list of evidence IDs.
Before the first terminal learning request, include episode_outline (at most six stages/4KiB) describing
the useful process: grasp region and relative pose -> transformations/transport -> interaction -> release,
with exact first/last receipt bounds, conditions and unknowns. Order by first receipt; concurrent spans may overlap.
Explain how the grasp enabled later work,
not just whether an object was still held. This is an unreviewed interpretation, not a verified script.
Keep failures/recoveries in the full trace and mark gaps; do not imply a pruned path was executed successfully.
The 4KiB limit includes receipt IDs, all fields and JSON formatting; leave room for this overhead.
An empty outline is valid when evidence is insufficient. The host seals it before giving the manager
the terminal success boolean; that label validates only the whole real episode, never individual stages.
For an already organized answer, prefer direct reads. Request management only if interpretation/publication
is consequential; no extra manager call is compulsory at episode end.
The bounded episode_index is navigation across the whole episode, including ordinary moves, measurements
and earlier questions. Use it to choose one consequential unresolved question if warranted, rather than
assuming the last response is the only learning opportunity. Counts are not task phases or evidence of a mechanism.
Consider what an earlier action or measurement distinguished and which assumptions remain unresolved.
Retain a useful scoped response or counterexample without requiring a causal law; do not default to another
retention/release narrative if it adds no decision information. Existing knowledge may already suffice.
Use prefix_replay to identify execution provenance: null means this episode began from reset with no
replayed trajectory. Only when a prefix is provided, distinguish it from new policy-controlled interactions.
Prior preparation and
working notes are conditional interpretations. Receipts prove execution, not contact, retention, continuous
no-lift compliance or task success. The private evaluator is not supplied. When done, call finish_review
with a concise observable conclusion and remaining uncertainty. Extra policy and manager calls are counted.'''


def _events(root, kinds):
    result = []
    for line in (root / 'events.jsonl').read_text().splitlines():
        ref = json.loads(line)
        if ref['kind'] not in kinds:
            continue
        path = Path(ref['path'])
        if path.parent != root / 'events' or hashlib.sha256(path.read_bytes()).hexdigest() != ref['sha256']:
            raise ValueError('Terminal source evidence changed')
        result.append(json.loads(path.read_text()))
    return result


def restore_session(trace, store, source_run, manager_client):
    """Restore the current management state, without replaying physics or old writes."""
    source_run = Path(source_run).resolve()
    if not (source_run / 'report.json').is_file():
        raise ValueError('Terminal review requires a closed episode report')
    manifest = json.loads((source_run / 'manifest.json').read_text())

    def record(kind, *, last=True):
        order = 'DESC' if last else 'ASC'
        row = store.db.execute(f'SELECT * FROM evidence WHERE run=? AND kind=? ORDER BY sequence {order} LIMIT 1',
                               (source_run.name, kind)).fetchone()
        if row is None:
            raise ValueError(f'Missing terminal source: {kind}')
        return {**dict(row), 'data': store.evidence_payload(row['id'])}

    current = record('observation')
    session = MemorySession(trace, store, task=manifest['task'], dependencies=current['data']['dependencies'],
        perception_profile=manifest['perception_profile'])
    session.episode = source_run.name
    session.current_observation = current
    session.body_prior = record('known_config', last=False)
    session.task_specification = record('task_specification', last=False)
    session.task_instruction = session.task_specification['data']['instruction']
    rid = current['data'].get('after_receipt')
    if rid:
        session.last_receipt = {'id': rid, 'data': store.evidence_payload(rid)}
    session.attach_manager(manager_client)
    events = _events(source_run, {'manager_group_publication', 'manager_tool_result',
                                  'attachment_change', 'attachment_observation'})
    publications = [event for event in events if event['kind'] == 'manager_group_publication']
    if not publications:
        raise ValueError('Completed episode has no task preparation')
    publication = publications[-1]
    saved = publication['payload']
    manager = session.manager
    if 'default_ids' not in saved:raise ValueError('Archived management uses an unsupported publication format')
    manager.default_ids=saved['default_ids']
    manager.default_versions=saved['default_versions']
    manager.history_ids=saved['history_ids']
    manager.history_versions=saved['history_versions']
    manager.version,manager.prepared=saved['version'],True
    manager.frontier = saved['source_frontier']
    manager.publication = publication['id']
    # Committed defaults/version may come from a partial group publication.
    # The working interpretation independently needs a matching completed PASS.
    completions=[event for event in events if event['kind']=='manager_tool_result']
    for candidate in reversed(publications):
        if any(manager.remember_completed_result(candidate,event) for event in reversed(completions)):
            break
    for event in events:
        payload = event['payload']
        if event['kind'] in {'attachment_change', 'attachment_observation'}:
            session.attachment[payload['arm']] = payload['new']
    # The latest observation's pins remain authoritative; old declarations are only interpretations.
    return session, manifest


def _observation(record):
    payload = record['data']
    state = evidence_state(payload)
    parts = [{'type': 'text', 'text': json.dumps({
        'observation_id': record['id'], 'env_step': payload.get('env_step'),
        'remaining_steps': payload.get('remaining_steps'),
        'grasp_point_dimensions': state['grasp_point_dimensions'],
        'coordinate_contract': state['coordinate_contract'],
        'measured_finger_joints_m': payload.get('measured_finger_joints_m', {}),
        'qualification': 'Recorded RGB and measured robot state; not official task success.'}, ensure_ascii=False)}]
    for camera, source in payload['images'].items():
        raw = Path(source['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != source['sha256']:
            raise ValueError('Terminal RGB evidence changed')
        parts.extend([{'type': 'text', 'text': 'RGB_REFERENCE ' + json.dumps({
            'observation_id': record['id'], 'camera': camera})},
            {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + base64.b64encode(raw).decode()}}])
    return {'role': 'user', 'content': parts}


def review_terminal(session, client, manifest, *, official_success=None, termination='episode_ended'):
    if session.manager is None or not session.manager.prepared:raise ValueError('Prepared manager required')
    from . import episodes
    episodes.begin(session,manifest,official_success,termination)
    public={'episode_ended':True,'source_run':session.episode,'task':session.task_instruction,
        'final_observation_ref':session.current_observation['id'],'prefix_replay':manifest['prefix_replay'],
        'execution_intervals':session.store.db.execute("SELECT COUNT(*) FROM evidence WHERE run=? AND kind='receipt'",(session.episode,)).fetchone()[0],
        'qualification':'No private task success provided. Reflect on observable outcomes, and optionally request one useful batch learning.'}
    entry=session.store.db.execute('SELECT chain_ref FROM episode_records WHERE run=?',(session.episode,)).fetchone()
    chain=session.store.evidence_payload(entry['chain_ref'])
    public['trace_receipt_bounds']={'first':chain['receipts'][0]['id'] if chain['receipts'] else None,
        'last':chain['receipts'][-1]['id'] if chain['receipts'] else None,
        'qualification':'Whole trace bounds, not inferred task phases. Stage boundaries must be exact known receipt IDs.'}
    source=session.store.db.execute('SELECT path FROM evidence WHERE id=?',(session.current_observation['id'],)).fetchone()
    if source:
        public['episode_index']=episode_index.readout(episode_index.build(Path(source['path']).parent.parent,
            through_sequence=session.source_frontier()[session.episode]),completed=True)
    if session.last_receipt:
        receipt=session.last_receipt['data']
        public['final_receipt']={'evidence_id':session.last_receipt['id'],
            **{key:receipt.get(key) for key in ('before_observation','execution',
                'executed_actions','accepted_actions','planned_actions','env_start','env_end',
                'failure_type','unknown_command','last_recorded_execution_frame')}}
    messages=[{'role':'system','content':INSTRUCTIONS},{'role':'user','content':json.dumps(public,ensure_ascii=False)},
              _observation(session.current_observation)]
    finish=function('finish_review','Finish this motion-free review.',
        {'conclusion':{'type':'string'},'uncertainty':{'type':'string'},'episode_outline':outline_schema()},['conclusion','uncertainty'])
    tools=[*policy_memory_tools(),finish]
    while True:
        if session.stop_requested():raise RuntimeError('Stopped before terminal inference')
        available=tools
        last_primary_call=client.budget is not None and client.budget.view()['remaining']==1
        if last_primary_call:
            manager_budget=session._manager_call_budget()
            if manager_budget is not None and manager_budget['remaining']==0:
                available=[finish]
                notice='One primary call remains and the manager allowance is exhausted. Finish the review using the supplied evidence; state unresolved uncertainty.'
            else:
                available=[t for t in tools if t['function']['name'] in ('manage_memory','finish_review')]
                notice='One primary call remains. Choose finish_review, or explicitly request one worthwhile manage_memory learn/review with a valid outline when required. Management uses its own remaining allowance. After that request the host closes the record without another primary review; rejected requests are not retried.'
            messages.append({'role':'user','content':notice})
        prefix=session.manager.prefix_view()
        session.trace.event('decision_materials',{'phase':'terminal_review','knowledge_version':session.manager.version,
            'preparation_sha256':hashlib.sha256(json.dumps(prefix,ensure_ascii=False).encode()).hexdigest(),
            'material_scope':session.manager.material_scope(),'robot_actions':0})
        response=client.complete(messages,available)
        message=response['choices'][0]['message'];call=message['tool_calls'][0]
        name=call['function']['name'];args=json.loads(call['function']['arguments'])
        session.evidence('policy_decision',{'phase':'terminal_review','tool':name,'arguments':args,
            'call_id':client.last_attempt_id,'robot_actions':0})
        messages.append(message)
        if name not in {t['function']['name'] for t in available}:
            raise ValueError('No motion tool or unoffered terminal tool may be dispatched')
        if name=='finish_review':
            try:
                if set(args)-{'conclusion','uncertainty','episode_outline'} or not {'conclusion','uncertainty'}<=args.keys():raise ValueError('Explicit terminal result required')
                row=session.store.db.execute('SELECT outline_ref FROM episode_records WHERE run=?',(session.episode,)).fetchone()
                if 'episode_outline' in args or not row[0]:episodes.seal(session,args.get('episode_outline',[]))
            except (KeyError,TypeError,ValueError) as error:
                messages.append({'role':'tool','tool_call_id':call['id'],'content':json.dumps({'status':'NEEDS_WORK','reason':str(error),'robot_actions':0})})
                continue
            return args,messages
        if name=='query_body_history':result=session.query_body_history(args)
        elif name=='query_experience':result=session.query_experience(args)
        elif name=='manage_memory':result=session.manage_memory(args)
        else:raise ValueError('No motion tool available after episode end')
        messages.append({'role':'tool','tool_call_id':call['id'],'content':result})
        if last_primary_call:
            # Preserve the actual request/result; do not invent a policy conclusion
            # or turn a rejected request into a successful learning operation.
            return {'completion':'host_closed_after_last_primary_request','last_primary_request':name,
                'primary_followup':False,'tool_result':result},messages
