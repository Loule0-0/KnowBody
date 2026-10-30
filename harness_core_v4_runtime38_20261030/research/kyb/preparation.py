"""One bounded decision packet; complete claims or explicit navigation, never cut IFs."""
from __future__ import annotations

import copy
from .readout import size

PREPARATION_BYTES = 8192


def history_stub(record):
    """A qualified query entrance, never a shortened claim or a completed read."""
    from .readout import pointer,statement,subject
    record=statement(record)
    fields=('id','revision','domain','family','status','strength','default_eligible','applicability',
            'source_contested','source_revised','related_warnings','counter_cases')
    stub=copy.deepcopy({k:record[k] for k in fields if k in record})
    stub.update(subject(record))
    stub.update(navigation_only=True,complete_record=False,statement_complete=False,
        qualification='Historical navigation only; no statement, parameters, source-read or support permission. Query the complete visible statement before considering it.',
        counter_sources=[pointer(ref) for ref in record.get('drill_down',{}).get('source_refs',[])
                         if ref.get('relation')=='counter'],
        read={'level':'knowledge','ids':[record['id']],'view':'statement'})
    return stub


def add_history(packet, records):
    """Use leftover space only. No accepted stub means byte-identical old material."""
    for record in records:
        trial=copy.deepcopy(packet)
        trial.setdefault('history_index',[]).append(history_stub(record))
        for _ in range(4):
            trial['packing']['experience_serialized_bytes']=size({k:v for k,v in trial.items() if k!='operator_clarifications'})
            trial['packing']['serialized_bytes']=size(trial)
        if trial['packing']['experience_serialized_bytes']<=PREPARATION_BYTES:packet=trial
    return packet


def delivery_manifest(packet, defaults, histories, episodes=()):
    """Packing evidence only; neither selection nor a forecast means model adoption."""
    import hashlib,json
    def rows(records,full,index):
        complete={(r['id'],r['revision']) for r in packet.get(full,[])} if full else set()
        navigation={(r['id'],r['revision']) for r in packet.get(index,[])}
        return [{'id':r['id'],'revision':r['revision'],
                 'delivery':'complete' if (r['id'],r['revision']) in complete else 'navigation' if (r['id'],r['revision']) in navigation else 'omitted',
                 'reason':None if (r['id'],r['revision']) in complete|navigation else 'packing_budget'} for r in records]
    return {'version':packet['version'],'packet_sha256':hashlib.sha256(json.dumps(packet,ensure_ascii=False).encode()).hexdigest(),
            'default_pool':rows(defaults,'knowledge','memory_index'),
            'history_refs':rows(histories,None,'history_index'),
            'episode_candidates':[{'id':e['id'],'delivery':'complete' if any(r['id']==e['id'] for r in packet['episode_candidates']) else
                'navigation' if any(r['id']==e['id'] for r in packet['episode_index']) else 'omitted'} for e in episodes],
            'scope':'Rendered material at this version/scope; not an actual model input or adoption until matched to archived CLI input.'}


def episode_for_decision(episode):
    """Keep physical qualifications; leave exact span IDs in the full episode read."""
    result=copy.deepcopy({k:v for k,v in episode.items()
        if k not in ('role_costs','executed_actions','receipt_count','trace_entry')})
    result['stages']=[{**{k:v for k,v in stage.items() if k not in ('first_receipt','last_receipt')},
                       'stage_index':index} for index,stage in enumerate(episode.get('stages',[]),1)]
    if isinstance(result.get('restart'),dict):result['restart'].pop('source_run',None)
    result['read']={'level':'episode','ids':[episode['id']]}
    result['span_scope']='Selected spans may overlap and are ordered by first receipt, not an executable sequence. Exact receipt ranges and provenance remain in the full episode read.'
    return result


def pack(base, knowledge, episodes, amendments):
    packet = {**base, 'knowledge': [], 'memory_index': [], 'episode_candidates': [], 'episode_index': [],
        'operator_clarifications': copy.deepcopy(amendments),
        'packing': {'limit_bytes': PREPARATION_BYTES, 'serialized_bytes': 0,
            'experience_serialized_bytes':0,'operator_contract_bytes':size(amendments),
            'omitted_knowledge': 0, 'omitted_episodes': 0,
            'rule': 'Complete conditional statements only. Index entries are navigation; query the missing question.'}}

    def experience_size():
        return size({k:v for k,v in packet.items() if k!='operator_clarifications'})

    def fits():
        # Space for final counters and the byte count itself.
        return experience_size() + 96 <= PREPARATION_BYTES

    def add(container, item):
        container.append(copy.deepcopy(item))
        if fits():
            return True
        container.pop()
        return False

    def card_for(record):
        # Leave provenance, counts and costs behind an exact query. Keep every
        # decision-bearing condition, limitation and source qualification.
        keys = ('id', 'revision', 'question', 'text', 'conditions', 'params', 'strength',
                'use', 'unknowns', 'depends_on', 'applicability', 'counterexamples', 'related_warnings')
        return {k: record[k] for k in keys if k in record}

    episode_pointer=None
    if episodes:
        candidate=episodes[0]
        episode_pointer={k:candidate[k] for k in ('id','revision','task','reviewed','interpretation','official_success') if k in candidate}
        episode_pointer.update(read={'level':'episode','ids':[candidate['id']]},
            navigation_only=True,physical_support=False,
            qualification='Unreviewed process navigation, not physical support. Read the complete candidate and conditions before use.')
        # Reserve navigation before optional K. A growing archive must never
        # turn a failure to fit an optional experience into a runtime failure.
        if not add(packet['episode_index'],episode_pointer):episode_pointer=None

    ordered = [('knowledge', card_for(k)) for k in knowledge[:1]]
    for episode in episodes[:1]:
        ordered.append(('episode_candidates', episode_for_decision(episode)))
    ordered += [('knowledge', card_for(k)) for k in knowledge[1:2]]
    delivered = set()
    for kind, card in ordered:
        if kind=='episode_candidates':
            packet['episode_index']=[]
        if add(packet[kind], card):
            delivered.add((kind, card['id']))
        elif kind == 'episode_candidates':
            if episode_pointer is not None:add(packet['episode_index'],episode_pointer)
            packet['packing']['omitted_episodes'] += 1
    packet['packing']['omitted_episodes'] += max(0, len(episodes)-1)
    for record in knowledge:
        if ('knowledge', record['id']) in delivered:
            continue
        pointer = {k: record[k] for k in ('id', 'revision', 'family', 'strength', 'related_warnings') if k in record}
        question = record['question']
        pointer.update(question=question[:100], question_complete=len(question) <= 100,
            qualification='Navigation only; query this exact ID for complete conditions.')
        if len(packet['memory_index']) >= 4 or not add(packet['memory_index'], pointer):
            packet['packing']['omitted_knowledge'] += 1
    # Count the actual serialization emitted by the CLI, not sum of sub-limits.
    for _ in range(4):
        packet['packing']['experience_serialized_bytes'] = experience_size()
        packet['packing']['serialized_bytes'] = size(packet)
    if experience_size() > PREPARATION_BYTES:
        raise ValueError('Stable preparation metadata exceeds its total byte allowance')
    return packet
