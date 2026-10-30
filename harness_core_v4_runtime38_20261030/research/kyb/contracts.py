"""Resolve requested configuration pins from immutable, source-time records only."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from .qualification import unknown

STATIC_KEYS = frozenset(('body', 'cameras', 'depth', 'execution', 'planner'))
INTERPRETATION_KINDS = frozenset(('episode_outline', 'episode_result', 'episode_chain', 'episode_cost','task_contract_amendment'))


def task_contract_id(task, instruction, recipe):
    return hashlib.sha256(json.dumps({'task':task,'instruction':instruction,'public_recipe':recipe},
        ensure_ascii=False,sort_keys=True).encode()).hexdigest()


def current_contract(session):
    if not session.task_specification:return None
    return session.store.evidence_payload(session.task_specification['id']).get('contract_id')


def amendment_cards(session, frontier=None):
    contract=current_contract(session)
    if contract is None:return []
    frontier=session.source_frontier() if frontier is None else frontier
    cards=[]
    for row in session.store.db.execute("SELECT id,run,sequence,payload FROM evidence WHERE kind='task_contract_amendment' AND json_extract(payload,'$.contract_id')=? ORDER BY sequence DESC",(contract,)):
        if row['sequence']>frontier.get(row['run'],-1):continue
        payload=json.loads(row['payload'])
        if payload['reuse_scope']!='same_public_contract' and payload['episode']!=session.episode:continue
        cards.append({'id':row['id'],'revision':row['sequence'],'layer':'public_contract',
            'contract_id':contract,'question':payload['question'],'answer':payload['answer'],
            'reuse_scope':payload['reuse_scope'],'request_id':payload['request_id'],
            'qualification':'Explicit operator clarification within this scope. Multiple amendments are separate statements; differences are not silently overwritten. Not physical evidence.'})
    return cards


def amendment_view(session):
    # Goal authority must not compete with optional learned experience for space.
    # Every returned entry matches the exact public contract and reuse scope.
    return {'statements':amendment_cards(session),
        'qualification':'Complete applicable operator amendments; authoritative task data. Resolve differences explicitly. Separate from the experience byte allowance.'}


def resolve_case_contracts(store, cases, keys):
    pins, basis, cache = {}, {}, {}
    for key in keys:
        declared = [(case, json.loads(case['dependencies'])[key]) for case in cases
                    if key in json.loads(case['dependencies'])]
        if not declared or any(unknown(value) for _, value in declared):
            raise ValueError(f'{key}: at least one supported case must declare a known pin')
        value = declared[0][1]
        if any(v != value for _, v in declared):
            raise ValueError(f'{key}: source contracts differ')
        origins = []
        for case in cases:
            declared_here = key in json.loads(case['dependencies'])
            if declared_here:
                continue
            if key not in STATIC_KEYS:
                raise ValueError(f'{key}: dynamic pins cannot be supplied from historical context')
            sources = []
            for source in store.provenance_at(case['revision'])['evidence']:
                row = store.db.execute('SELECT kind,payload,path,sha256 FROM evidence WHERE id=?', (source,)).fetchone()
                if row is None:
                    raise ValueError('Missing immutable case source')
                if source not in cache:
                    raw=Path(row['path']).read_bytes()
                    if hashlib.sha256(raw).hexdigest()!=row['sha256']:raise ValueError('Historical configuration source changed: '+source)
                    original=json.loads(raw)
                    if original['id']!=source or original['payload']!=json.loads(row['payload']):raise ValueError('Historical configuration payload differs from its immutable source')
                    cache[source]=original['payload']
                payload = cache[source]  # Not mutable annotations or evidence_payload overlays.
                if row['kind'] not in ('observation', 'receipt', 'known_config'):
                    continue
                source_value = payload.get('dependencies', {}).get(key)
                if unknown(source_value) or source_value != value:
                    raise ValueError(f'{key}: historical context is missing or conflicts at {source}')
                sources.append(source)
            if not sources:
                raise ValueError(f'{key}: no exact historical configuration context')
            origins.append({'case': case['id'], 'revision': case['revision'], 'sources': sources})
        pins[key] = value
        basis[key] = {'declared_by': [{'id': c['id'], 'revision': c['revision']} for c, _ in declared],
                      'context_from': origins}
    return pins, basis
