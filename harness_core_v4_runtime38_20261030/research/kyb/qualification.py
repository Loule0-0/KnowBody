"""Physical reading qualifications; task labels never stand in for identities."""
from __future__ import annotations

import json

ROLES = {'intrinsic_candidate', 'binding_geometry', 'scene_observation', 'interaction'}
METADATA_KEYS = {'task', 'phase', 'episode'}


def unknown(value):
    return value is None or (isinstance(value,str) and
        (value in ('','unknown','unspecified') or value.endswith('_unknown')))


def compare_conditions(conditions, current):
    """Equality of two model declarations is weaker than a measured match."""
    return {key: ('unknown' if unknown(value) or unknown(current.get(key)) else
                  'match_declared' if value == current[key] else 'unknown')
            for key, value in conditions.items() if key not in METADATA_KEYS}


def reading_role(kind, entity):
    if kind == 'external':
        return 'interaction'
    return {'robot_part': 'intrinsic_candidate', 'tool_part': 'binding_geometry',
            'scene_object': 'scene_observation'}.get(entity, 'scene_observation')


def opening_arms(receipt, before):
    """An increased opening command invalidates retention, never proves release."""
    opened = []
    for arm, detail in receipt.get('plan', {}).get('trace', {}).get('arms', {}).items():
        command = detail.get('named', {}).get(f'{arm}_gripper')
        previous = before.get('state', {}).get(f'{arm}_ee_joint_state')
        if command is not None and (previous and command > float(previous[0])):
            opened.append(arm)
    return opened


def binding_snapshot(db, observation_id, cutoff=None):
    row = db.execute('SELECT run,sequence,payload FROM evidence WHERE id=?', (observation_id,)).fetchone()
    if row is None:
        raise ValueError('Unknown binding observation')
    payload = json.loads(row['payload'])
    pins = payload.get('dependencies', {})
    arms = {key.split(':', 1)[1] for key in pins if key.startswith('attachment:')}
    frontier = (cutoff or {}).get(row['run'], row['sequence'])
    result = {arm: {'binding_id': None, 'qualification': 'unknown',
                    'reason': 'no_sourced_uninvalidated_visual_declaration'} for arm in arms}
    timeline=[]
    for event in db.execute("SELECT id,sequence,kind,payload FROM evidence WHERE run=? AND sequence<=? "
                            "AND kind IN ('attachment_change','attachment_observation','receipt') ORDER BY sequence",
                            (row['run'], frontier)):
        data=json.loads(event['payload'])
        if event['kind']=='receipt':
            domain_sequence=event['sequence']
            source_payload=None
        else:
            source=db.execute('SELECT sequence,payload FROM evidence WHERE id=?',(data.get('observation_id'),)).fetchone()
            if source is None:
                continue
            domain_sequence=source['sequence'];source_payload=json.loads(source['payload'])
        if domain_sequence<=row['sequence']:
            timeline.append((domain_sequence,event['sequence'],dict(event),data,source_payload))
    # Availability uses publication time; physical order uses the observation's
    # time. A late annotation of an old frame cannot resurrect a released hold.
    for _,_,event,data,source_payload in sorted(timeline,key=lambda item:item[:2]):
        if event['kind']=='receipt':
            if not data.get('executed_actions'):
                continue
            before=db.execute('SELECT payload FROM evidence WHERE id=?',(data.get('before_observation'),)).fetchone()
            for arm in opening_arms(data,json.loads(before[0]) if before else {}):
                result[arm]={'binding_id':None,'qualification':'unknown','invalidated_by':event['id'],
                             'reason':'opening_executed_not_release_proof'}
        else:
            arm,declaration=data.get('arm'),data.get('new',{})
            if arm not in arms:
                continue
            if data.get('binding_changed'):
                pin=event['id']
            else:
                pin=result[arm].get('binding_id') or source_payload.get('dependencies',{}).get(f'attachment:{arm}')
            result[arm]={'binding_id':pin if declaration.get('state')=='visually_supported' else None,
                'qualification':'declared' if declaration.get('state')=='visually_supported' else 'unknown',
                'declaration_source':event['id'],'observation_source':data.get('observation_id'),
                'declaration':declaration}
    return result


def qualify(*, role, conditions, pins, context, dependencies, source_episode, episode,
            source_bindings):
    comparison = compare_conditions(conditions, context)
    different_descriptions = {key:{'source':value,'current':context[key]}
        for key,value in conditions.items() if key not in METADATA_KEYS and
        not unknown(value) and not unknown(context.get(key)) and value!=context[key]}
    physical = {key: ('unknown' if unknown(value) or unknown(dependencies.get(key)) else
                     'match' if value == dependencies[key] else 'differ')
                for key, value in pins.items() if key == 'body' or key.startswith(('attachment:','part:'))}
    # Unknown run-local attachment placeholders are never compared as identities.
    changed_contracts = [key for key, value in pins.items()
                         if key not in {'body', 'tool'} and not key.startswith(('attachment:','part:'))
                         and dependencies.get(key) != value]
    bindings = [item for snapshot in source_bindings.values() for item in snapshot.values()]
    current_ids = {value for key, value in dependencies.items() if key.startswith('attachment:') and not unknown(value)}
    same_binding = bool(bindings) and all(item.get('binding_id') in current_ids
        and item.get('qualification') == 'declared' for item in bindings if item.get('binding_id')) \
        and any(item.get('binding_id') for item in bindings)
    if role == 'binding_geometry':
        category = 'historical_binding_geometry' if source_episode == episode or same_binding else 'identity_unresolved'
    elif role == 'scene_observation':
        category = 'historical_scene_observation'
    else:
        category = 'analogy' if 'differ' in {*comparison.values(), *physical.values()} else 'conditional_candidate'
    return {'category': category, 'physical_conditions': comparison, 'body_and_binding_comparison': physical,
        'unknown_conditions': [key for key, value in comparison.items() if value == 'unknown'],
        'different_conditions': [key for key, value in physical.items() if value == 'differ'],
        'different_descriptions':different_descriptions,
        'changed_derivation_contracts': changed_contracts,
        'metric_qualification': 'incomparable_current_contract' if changed_contracts else 'historical_contract_only',
        'readout_qualification': 'declared', 'source_bindings': source_bindings,
        'geometry_equivalence': 'unknown_not_proved_by_binding_identity', 'current_fact': False}
