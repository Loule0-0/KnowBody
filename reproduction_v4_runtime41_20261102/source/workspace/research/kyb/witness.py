"""Bounded visual drill-down into complete cases already delivered to this policy."""
import copy
import hashlib
import json
from pathlib import Path

from . import library, readout
from .media import read_rgb, rgb_parts, rgb_reference


def _witness_rows(store, identifier, revision):
    return [tuple(r) for r in store.db.execute(
        'SELECT c.revision,i.capsule,d.patch FROM claims c JOIN interaction_evidence i '
        'ON i.revision=c.revision JOIN decisions d ON d.revision=c.revision AND d.accepted=1 '
        'WHERE c.id=? AND c.revision<=? ORDER BY c.revision', (identifier, revision))]


def _source_identities(store, revision):
    sources = store.provenance_at(revision)['evidence']
    return [tuple(store.db.execute('SELECT id,run,sequence,kind,path,sha256 FROM evidence WHERE id=?',
                                   (source,)).fetchone()) for source in sources]


def tool():
    from .knowledge import function
    return function('view_case_witness',
        'Read registered historical before/post RGB for exact case statements already completely '
        'delivered by query_experience in this episode. No manager call or motion. '
        'Returns full case conditions/limitations and receipt coverage, at most6 images and24576 UTF-8 '
        'bytes of result plus RGB-reference text (JPEGs/native history are separate). '
        'These are historical witnesses, never current state or a grasp template guarantee. '
        'An oversized request returns its receipt/frame manifest without images; select cameras '
        'or an explicit receipt subset. No arbitrary file, depth, calibration or writer access.',
        {'cases': {'type': 'array', 'minItems': 1, 'maxItems': 2,
                   'items': {'type': 'object', 'properties': {
                       'id': {'type': 'string'}, 'revision': {'type': 'integer', 'minimum': 1}},
                       'required': ['id', 'revision'], 'additionalProperties': False}},
         'cameras': {'type': 'array', 'minItems': 1, 'maxItems': 3, 'uniqueItems': True,
                     'items': {'type': 'string'}},
         'receipts': {'type': 'array', 'minItems': 1, 'maxItems': 4, 'uniqueItems': True,
                      'items': {'type': 'string'}}}, ['cases', 'cameras'])


def remember(session, owner, card):
    """Keep the completed reader view; no re-search, new KB fields or raw grants."""
    if not owner.startswith('policy:') or card.get('layer') != 'case':
        return
    if owner != 'policy:' + session.episode:
        raise ValueError('Policy case delivery belongs to another episode')
    if not hasattr(session, '_case_witness_reads'):
        session._case_witness_reads = {}
    row = session.store.db.execute('SELECT * FROM claims WHERE id=? AND revision=?',
                                  (card['id'], card['revision'])).fetchone()
    if row is None:
        raise ValueError('Delivered case has no registered revision')
    session._case_witness_reads[(card['id'], card['revision'])] = {
        'card': copy.deepcopy(card), 'store': session.store,
        'profile': session.perception.name, 'frontier': session.source_frontier(),
        'case_at': session.frozen_revision if session.frozen_revision is not None else session.store.revision,
        'claim': tuple(row), 'witnesses': _witness_rows(session.store, card['id'], card['revision']),
        'sources': _source_identities(session.store, card['revision']),
        'reviews': [tuple(r) for r in session.store.db.execute(
            'SELECT * FROM claim_reviews WHERE revision=? ORDER BY rowid', (card['revision'],))]}


def _source(session, identifier, kind, frontier):
    row = session.store.db.execute('SELECT * FROM evidence WHERE id=?', (identifier,)).fetchone()
    if row is None or row['kind'] != kind or not library.source_visible(session.store, identifier, frontier):
        raise ValueError('Witness source not visible or of the required kind')
    raw = Path(row['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != row['sha256']:
        raise ValueError('Witness source file changed')
    event = json.loads(raw)
    if (event.get('id') != identifier or event.get('kind') != kind
            or event.get('sequence') != row['sequence']
            or event.get('payload') != json.loads(row['payload'])):
        raise ValueError('Witness source registration changed')
    data = session.store.evidence_payload(identifier)
    session.perception.check_payload(kind, data)
    return row, data


def read(session, arguments):
    if session.episode_ended:
        raise ValueError('Historical visual reads are unavailable after episode end')
    if set(arguments) - {'cases', 'cameras', 'receipts'}:
        raise ValueError('Use exact cases, cameras and optional receipts only')
    refs, cameras = arguments.get('cases'), arguments.get('cameras')
    if (not isinstance(refs, list) or not 1 <= len(refs) <= 2
            or any(not isinstance(r, dict) or set(r) != {'id', 'revision'}
                   or not isinstance(r['id'], str) or type(r['revision']) is not int
                   or r['revision'] < 1 for r in refs)
            or len({r['id'] for r in refs}) != len(refs)):
        raise ValueError('Read one or two distinct exact case ID/revision pairs')
    if (not isinstance(cameras, list) or not 1 <= len(cameras) <= 3
            or any(not isinstance(c, str) or not c for c in cameras)
            or len(set(cameras)) != len(cameras)):
        raise ValueError('Choose one to three distinct registered cameras')
    chosen = arguments.get('receipts')
    if chosen is not None and (not isinstance(chosen, list) or not 1 <= len(chosen) <= 4
            or any(not isinstance(r, str) for r in chosen) or len(set(chosen)) != len(chosen)):
        raise ValueError('Choose one to four distinct registered receipt IDs')
    frontier = session.source_frontier()
    current_at = session.frozen_revision if session.frozen_revision is not None else session.store.revision
    cards, groups = [], []
    owner = 'policy:' + session.episode
    for ref in refs:
        key = (ref['id'], ref['revision'])
        saved = getattr(session, '_case_witness_reads', {}).get(key)
        if not saved or not readout.delivered_statement(session, owner, 'case', *key):
            raise ValueError('Read the complete exact case statement in this episode first')
        row = session.store.db.execute('SELECT * FROM claims WHERE id=? AND revision=?', key).fetchone()
        if (saved['store'] is not session.store or saved['profile'] != session.perception.name
                or row is None or tuple(row) != saved['claim']
                or _witness_rows(session.store, *key) != saved['witnesses']
                or _source_identities(session.store, ref['revision']) != saved['sources']
                or [tuple(r) for r in session.store.db.execute(
                    'SELECT * FROM claim_reviews WHERE revision=? ORDER BY rowid', (ref['revision'],))]
                != saved['reviews']):
            raise ValueError('Delivered case store, profile or revision changed')
        if not (library.case_visible(session.store, ref['revision'], saved['case_at'], saved['frontier'])
                and library.case_visible(session.store, ref['revision'], current_at, frontier)):
            raise ValueError('Case beyond the delivered or current source frontier')
        cards.append(copy.deepcopy(saved['card']))
        evidence = set(session.store.provenance_at(ref['revision'])['evidence'])
        found = False
        for witness_revision, raw_capsule, raw_patch in saved['witnesses']:
            capsule = json.loads(raw_capsule)
            witness = capsule.get('witness', capsule)
            if not isinstance(witness, dict):
                continue
            before = witness.get('before_observation_id', witness.get('before_observation'))
            receipt, post = witness.get('receipt_id'), witness.get('post_observation_id')
            if not before or not receipt or not post or not {before, receipt, post} <= evidence:
                continue
            # Validate the entire registered set before filtering a requested subset.
            data = {}
            rows = []
            for identifier, kind in [(before, 'observation'), (receipt, 'receipt'), (post, 'observation')]:
                if not library.source_visible(session.store, identifier, saved['frontier']):
                    raise ValueError('Witness beyond the case delivery frontier')
                source, payload = _source(session, identifier, kind, frontier)
                rows.append(source)
                data[identifier] = payload
            if (len({r['run'] for r in rows}) != 1
                    or not rows[0]['sequence'] < rows[1]['sequence'] < rows[2]['sequence']
                    or data[receipt].get('before_observation') != before
                    or data[post].get('after_receipt') != receipt):
                raise ValueError('Registered witness before/receipt/post binding changed')
            found = True
            groups.append({'case': dict(ref), 'witness_revision': witness_revision,
                           'case_update': json.loads(raw_patch)['op'],
                           'receipt_id': receipt, 'source_episode': rows[0]['run'],
                           'before': before, 'post': post, 'sequence': rows[1]['sequence'],
                           'data': data})
        if not found:
            raise ValueError('Case has no registered complete interaction witness')
    all_receipts = {g['receipt_id'] for g in groups}
    if chosen is not None and not set(chosen) <= all_receipts:
        raise ValueError('Receipt subset must belong to the selected cases')
    selected = [g for g in groups if chosen is None or g['receipt_id'] in chosen]
    if any(not any(g['case'] == ref for g in selected) for ref in refs):
        raise ValueError('Receipt subset must include a witness for every selected case')
    frames, binaries, receipts = {}, {}, {}
    for group in sorted(groups, key=lambda g: (g['source_episode'], g['sequence'], g['witness_revision'])):
        for phase in ('before', 'post'):
            observation = group[phase]
            if group not in selected:
                continue
            images = group['data'][observation].get('images', {})
            if any(camera not in images for camera in cameras):
                raise ValueError('Unknown camera in a selected case witness')
            for camera in cameras:
                key = (observation, camera)
                frame = frames.setdefault(key, {'observation_id': observation, 'camera': camera,
                    'source_episode': group['source_episode'], 'sha256': images[camera]['sha256'],
                    'links': []})
                link = {'case': group['case'], 'witness_revision': group['witness_revision'],
                        'case_update': group['case_update'],
                        'receipt_id': group['receipt_id'], 'phase': phase}
                if link not in frame['links']:
                    frame['links'].append(link)
                binaries[key] = images
        if group in selected:
            receipts[group['receipt_id']] = {'id': group['receipt_id'], **{
                k: group['data'][group['receipt_id']].get(k)
                for k in ('execution', 'executed_actions', 'planned_actions', 'failure_type')}}
    manifest = [{'case': g['case'], 'witness_revision': g['witness_revision'],
                 'case_update': g['case_update'],
                 'receipt_id': g['receipt_id'], 'before': g['before'], 'post': g['post'],
                 'cameras': cameras} for g in groups]
    result = {'status': 'PASS', 'historical': True, 'cases': cards,
              'receipts': list(receipts.values()), 'frames': list(frames.values()),
              'frames_scope': 'complete' if chosen is None else 'selected',
              'omitted': [m for g, m in zip(groups, manifest) if g not in selected],
              'executed_actions': 0,
              'qualification': 'Previously delivered case interpretations and registered historical RGB, '
                  'not current state, an acquisition recipe, a grasp guarantee or writer authority. '
                  'Check current conditions; counterexample text remains in the complete cases. '
                  'Only the listed witness frames are supplied, not every provenance image.'}
    text_bytes = readout.size(result) + sum(len(rgb_reference(frame,
        temporal_role='historical_case_witness').encode()) for frame in frames.values())
    if len(frames) > 6 or text_bytes > 24576:
        # A bounded retry hint, never an unbounded directory or truncated case.
        rejection = {'status': 'NEEDS_WORK', 'reason': 'Witness exceeds6 RGB/24576 JSON bytes; '
                     'select fewer cases/cameras or an explicit receipt subset.',
                     'required_images': len(frames), 'required_text_bytes': text_bytes,
                     'witnesses': manifest, 'executed_actions': 0}
        if readout.size(rejection) > 24576:
            rejection = {'status': 'NEEDS_WORK', 'reason': 'Witness directory exceeds24576 bytes; '
                         'read one case at a time with fewer cameras.', 'executed_actions': 0}
        return rejection, []
    # Oversized requests never load all their JPEGs merely to count them.
    supplied = [(frame, read_rgb(binaries[key], [key[1]])[0][1])
                for key, frame in frames.items()]
    parts = rgb_parts(supplied,
                      temporal_role='historical_case_witness')
    return result, parts
