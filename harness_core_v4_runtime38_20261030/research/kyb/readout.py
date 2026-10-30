"""Budgeted views of immutable records. A preview is discovery, never a full read."""
from __future__ import annotations
import copy
import hashlib
import json


def size(value):
    return len(json.dumps(value, ensure_ascii=False).encode())


def pointer(ref):
    return {k: ref[k] for k in ('type', 'id', 'revision') if k in ref}


def subject(record):
    """A bounded discovery label, not a substitute for the qualified statement."""
    key = next((k for k in ('question', 'text', 'task') if isinstance(record.get(k), str) and record[k]), None)
    if key is None:
        return {}
    raw = record[key].encode('utf-8')
    return {'title': raw[:240].decode('utf-8', errors='ignore'), 'title_source': key,
            'title_truncated': len(raw) > 240}


def statement(record):
    card = copy.deepcopy(record)
    card['view'] = 'statement'
    card['complete_record'] = False
    card['statement_complete'] = True
    if card.get('layer') == 'episode_candidate':
        # Costs remain in the immutable detail and trace_entry.cost. Ordinary
        # physical-process reading must not acquire a second page of accounting.
        card.pop('role_costs', None)
    if 'dependencies' in card:
        card['depends_on'] = sorted(card.pop('dependencies'))
    for param in card.get('params', []):
        if isinstance(param.get('source'), dict):
            param['source'] = pointer(param['source'])
    drill = card.get('drill_down', {})
    if 'refs' in drill:
        refs = drill.pop('refs')
        drill['source_refs'] = [dict(pointer(r), relation=r.get('relation')) for r in refs]
    if card.get('layer', 'knowledge') == 'knowledge' and 'strength' in card and 'drill_down' in card:
        refs = drill.get('source_refs', [])
        omitted = sum(ref.get('relation') == 'support' for ref in refs)
        if omitted:
            # Statement completeness concerns the qualified physical answer, not
            # full audit permission. The immutable detail keeps every pointer.
            drill['source_refs'] = [ref for ref in refs if ref.get('relation') != 'support']
            drill['support_source_refs_omitted'] = omitted
    if card.get('layer')!='case':
        # Decision input keeps the statement and all qualifications; audit counts and source
        # reasons remain accessible through the immutable detail continuation.
        for key in ('different_descriptions','different_conditions'):
            if not card.get('applicability',{}).get(key):card.get('applicability',{}).pop(key,None)
        for key in ('source_revised','related_warnings'):
            if card.get(key)==[]:card.pop(key,None)
    # Keep counterexample text/conditions too; provenance compression must not erase a limitation.
    return card


def preview(record):
    result = {**subject(record), **{k: copy.deepcopy(record[k]) for k in ('id', 'revision', 'layer', 'domain', 'family', 'strength',
                 'status','source_contested') if k in record},
            'conditions': copy.deepcopy(record.get('conditions', {})),
            'dependency_warnings': {k: record.get('applicability', {}).get(k, [])
                                    for k in ('changed_dependencies','unknown_dependencies')},
            'counterexample_refs': [r.get('revision') for r in record.get('counterexamples', [])],
            'view': 'preview', 'complete_record': False, 'statement_complete': False,
            'interpretation': 'Navigation only; read the full statement before use.'}
    for key in ('source_revised','related_warnings','counter_cases'):
        if record.get(key):result[key]=copy.deepcopy(record[key])
    if record.get('review'):result['review']=record['review']
    if record.get('source_episodes'):result['source_episodes']=copy.deepcopy(record['source_episodes'])
    counter_refs=[pointer(ref) for ref in record.get('drill_down',{}).get('refs',[]) if ref.get('relation')=='counter']
    if counter_refs:result['counter_source_refs']=counter_refs
    if size(result['conditions']) > 300:
        result['conditions_omitted_bytes'] = size(result.pop('conditions'))
        result['condition_keys'] = list(record.get('conditions', {}))
    return result


def _continuation(session, record, offset=0, *, owner=None):
    if not hasattr(session, '_read_continuations'):
        session._read_continuations = {}
    identity = {'record':record,'offset':offset}
    if owner is not None:
        identity['owner'] = owner
    token = hashlib.sha256(json.dumps(identity, ensure_ascii=False,
        sort_keys=True,separators=(',', ':')).encode()).hexdigest()[:32]
    session._read_continuations[token] = (copy.deepcopy(record), offset)
    return token


def delivered_statement(session, owner, layer, identifier, revision):
    return (owner,layer,identifier,revision) in getattr(session,'_statement_deliveries',set())


def note_delivery(session, owner, answer):
    """A complete tool output is delivery, never evidence review or write permission."""
    if not hasattr(session,'_statement_deliveries'):session._statement_deliveries=set()
    if not hasattr(session,'_statement_fragments'):session._statement_fragments={}
    for record in answer.get('records',[]):
        if record.get('complete_record') or record.get('statement_complete'):
            session._statement_deliveries.add((owner,record.get('layer','knowledge'),record['id'],record['revision']))
        elif record.get('view') in ('statement_page','detail_page'):
            key=(owner,record['content_sha256'])
            fragments=session._statement_fragments.setdefault(key,{})
            fragments[record['offset']]=record['fragment']
            parts=[];offset=0
            for start,fragment in sorted(fragments.items()):
                if start!=offset:break
                parts.append(fragment);offset+=len(fragment)
            raw=''.join(parts)
            if offset==record['total_characters'] and hashlib.sha256(raw.encode()).hexdigest()==record['content_sha256']:
                value=json.loads(raw)
                session._statement_deliveries.add((owner,value.get('layer','knowledge'),value['id'],value['revision']))


def page(session, token, budget_bytes, *, owner=None):
    if token not in getattr(session, '_read_continuations', {}):
        raise ValueError('Unknown continuation in this session')
    record, start = session._read_continuations[token]
    raw = json.dumps(record, ensure_ascii=False, separators=(',', ':'))
    result = {'id': record['id'], 'revision': record['revision'], 'layer': record.get('layer', 'knowledge'),
              'view': record['view'] + '_page', 'complete_record': False, 'offset': start,
              'content_sha256': hashlib.sha256(raw.encode()).hexdigest(),
              'total_characters': len(raw), 'fragment': '', 'continuation': None, 'status': 'page_complete'}
    # Bound the actual serialized result, including escaped JSON and token.
    lo, hi = start, len(raw)
    while lo < hi:
        end = (lo + hi + 1) // 2
        test = {**result, 'fragment': raw[start:end], 'continuation': 'x' * 32}
        if size(test) <= budget_bytes:
            lo = end
        else:
            hi = end - 1
    if lo == start:
        raise ValueError('Read budget too small for one detail page; narrow the selection or increase budget_bytes')
    result['fragment'] = raw[start:lo]
    if lo < len(raw):
        result['continuation'] = _continuation(session, record, lo, owner=owner)
    result['status'] = 'partial' if result['continuation'] else 'page_complete'
    return result


def pack(session, records, answer, *, budget_bytes, limit, view='auto', selection_scope=None,
         complete_only=False, offset=0):
    """Deliver whole requested views first; defer remaining records as one selection.

    Only a view that cannot fit an otherwise empty response is paged. The
    selection stores the original qualified records, never a rerun of retrieval.
    """
    if view not in ('auto', 'statement', 'detail'):
        raise ValueError('view must be auto, statement or detail')
    base = copy.deepcopy(answer)
    for key in ('records', 'selection_continuation', 'unread_records', 'more_available',
                'status', 'unread_qualification'):
        base.pop(key, None)

    def envelope(cards, cursor):
        unread = len(records) - cursor
        incomplete = any(not (r.get('complete_record') or r.get('statement_complete')) for r in cards)
        result = {**base, 'records': cards, 'unread_records': unread,
                  'more_available': bool(unread) or incomplete,
                  'status': 'no_match' if not records else 'partial' if unread or incomplete else 'records_found'}
        if unread:
            result['selection_continuation'] = 'x' * 32
            result['unread_qualification'] = 'Remaining records, including their conditions and warnings, have not been delivered.'
        return result

    cards = []
    cursor = offset
    while cursor < len(records) and len(cards) < limit:
        record = records[cursor]
        full = {**copy.deepcopy(record), 'view': 'detail', 'complete_record': True}
        versions = [full, statement(record)] if view == 'auto' else [statement(record)] if view == 'statement' else [full]
        for candidate in versions:
            if not candidate.get('complete_record'):
                candidate['continuation'] = _continuation(session, full)
        chosen = next((c for c in versions if size(envelope([*cards, c], cursor + 1)) <= budget_bytes), None)
        if chosen is not None:
            cards.append(chosen)
            cursor += 1
            continue
        # A whole view that fits by itself waits for the next response. Never
        # consume leftover space with fragments of another otherwise small card.
        fits_alone = any(size(envelope([c], cursor + 1)) <= budget_bytes for c in versions)
        if cards or fits_alone or complete_only:
            break
        target = versions[-1] if view == 'statement' else full
        remaining = budget_bytes - size(envelope([], cursor + 1)) - 4
        fragment = page(session, _continuation(session, target), remaining)
        if size(envelope([fragment], cursor + 1)) > budget_bytes:
            raise ValueError('Read budget cannot fit a detail page; narrow the selection or increase budget_bytes')
        cards.append(fragment)
        cursor += 1
        break
    result = envelope(cards, cursor)
    if size(result) > budget_bytes:
        raise ValueError('Read budget cannot fit selection envelope; narrow the selection or increase budget_bytes')
    if cursor < len(records):
        selected = {'view': 'record_selection', 'records': records, 'answer': base,
                    'requested_view': view, 'scope': copy.deepcopy(selection_scope)}
        result['selection_continuation'] = _continuation(session, selected, cursor)
    return result
