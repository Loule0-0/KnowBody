"""Keep a candidate proposition separate from each dated supporting observation."""
from __future__ import annotations

import json

from .state import body_transition


def load_payload(db, source):
    payload = json.loads(db.execute('SELECT payload FROM evidence WHERE id=?', (source,)).fetchone()[0])
    coordinate = db.execute('SELECT canonical,derivation FROM observation_coordinates WHERE observation_id=?', (source,)).fetchone()
    if coordinate:
        payload['canonical_state'] = json.loads(coordinate[0])
        payload['coordinate_derivation'] = json.loads(coordinate[1])
    return payload


def version_sources(db, revision):
    """Reconstruct delta links; revise explicitly replaces the current source set."""
    claim = db.execute('SELECT id FROM claims WHERE revision=?', (revision,)).fetchone()
    if claim is None:
        raise ValueError('Unknown claim revision')
    sources = set()
    for row in db.execute('SELECT c.revision,d.patch FROM claims c JOIN decisions d ON d.revision=c.revision '
                           'AND d.accepted=1 WHERE c.id=? AND c.revision<=? ORDER BY c.revision', (claim['id'], revision)):
        op = json.loads(row['patch']).get('op', 'new')
        if op in ('new', 'revise'):
            sources.clear()
        sources.update(r[0] for r in db.execute('SELECT evidence_id FROM claim_evidence WHERE revision=?', (row['revision'],)))
    return sorted(sources)


def source_physics_dt(db, run):
    row = db.execute("SELECT json_extract(payload,'$.planner_setup.physics_dt') FROM evidence "
                     "WHERE run=? AND kind='known_config' ORDER BY sequence LIMIT 1", (run,)).fetchone()
    return float(row[0]) if row is not None and row[0] is not None else None


def dated_instance(db, revision):
    row = db.execute('SELECT c.id,c.revision,c.conditions,c.dependencies,c.epistemic,c.status,d.patch, '
        'i.relation_type,i.capsule FROM claims c JOIN decisions d ON d.revision=c.revision AND d.accepted=1 '
        'LEFT JOIN interaction_evidence i ON i.revision=c.revision WHERE c.revision=?', (revision,)).fetchone()
    if row is None:
        return None
    patch = json.loads(row['patch'])
    capsule = json.loads(row['capsule']) if row['capsule'] else {}
    witness = capsule.get('witness') if row['relation_type'] != 'transition' else capsule
    relation_type=row['relation_type']
    if row['capsule'] is None:
        # A body-labelled visual update can still have an unambiguous physical
        # action skeleton. This never upgrades its text to retention or causality.
        pairs=[]
        sources=patch.get('evidence',[])
        for source in sources:
            receipt_row=db.execute("SELECT payload FROM evidence WHERE id=? AND kind='receipt'",(source,)).fetchone()
            if not receipt_row:
                continue
            receipt=json.loads(receipt_row[0])
            if (receipt.get('execution') not in {'complete','partial'} or not receipt.get('executed_actions')
                or receipt.get('unknown_command') is not None or receipt.get('before_observation') not in sources):
                continue
            if patch.get('receipt_id') and patch['receipt_id']!=source:
                continue
            for post_id in sources:
                post_row=db.execute("SELECT payload FROM evidence WHERE id=? AND kind='observation'",(post_id,)).fetchone()
                if post_row and json.loads(post_row[0]).get('after_receipt')==source and (
                    not patch.get('post_observation_id') or patch['post_observation_id']==post_id):
                    pairs.append({'before_observation':receipt['before_observation'],'receipt_id':source,'post_observation_id':post_id})
        if len(pairs)==1:
            witness=pairs[0];relation_type='observed_action_skeleton_not_a_stable_response_or_slip_transition'
    if witness is None:
        return None  # An explicit hypothesis has no observed interaction witness.
    before_id = witness.get('before_observation_id', witness.get('before_observation'))
    receipt_id, post_id = witness['receipt_id'], witness['post_observation_id']
    ids = [before_id, receipt_id, post_id]
    records = [db.execute('SELECT run,sequence,payload FROM evidence WHERE id=?', (source,)).fetchone() for source in ids]
    if any(record is None for record in records):
        raise ValueError('Published observed instance is missing its bound evidence')
    before, receipt, post = [load_payload(db, source) for source in ids]
    return {'claim_id': row['id'], 'revision': row['revision'], 'operation': patch['op'],
        'observation_text': patch['text'], 'text_role': 'this_accepted_update_observation_not_preserved_proposition',
        'before_observation_id': before_id, 'receipt_id': receipt_id, 'post_observation_id': post_id,
        'conditions': json.loads(row['conditions']), 'claim_dependencies': json.loads(row['dependencies']),
        'execution_dependencies': receipt.get('dependencies', {}), 'epistemic': row['epistemic'],
        'execution_receipt': {key:receipt.get(key) for key in ('execution','executed_actions','accepted_actions',
            'planned_actions','env_start','env_end','unknown_command')},
        'relation_type': relation_type, 'status': row['status'],
        'action_record':{'requested_named':{arm:detail.get('named',{}) for arm,detail in receipt.get('plan',{}).get('trace',{}).get('arms',{}).items()},
            'submitted_commands':{'count':len(receipt.get('command_refs',[])), 'first':receipt.get('command_refs',[])[:1],
                                  'last':receipt.get('command_refs',[])[-1:]},
            'realized_readout_field':'measured_body_transition', 'action_coverage':'unknown_not_inferred_from_one_instance'},
        'measured_body_transition': body_transition(before, receipt, post, source_physics_dt(db, records[1]['run'])),
        'evidence': sorted(set(ids+patch.get('evidence', []))),
        'qualification': 'Dated visual interpretation remains provisional; host measured body values are separate from scene-effect text.'}


def claim_provenance(db, revision):
    """Negative witnesses keep their original conditions through later rewrites."""
    claim = db.execute('SELECT id FROM claims WHERE revision=?', (revision,)).fetchone()
    if claim is None:
        raise ValueError('Unknown claim revision')
    counterexamples = []
    sources = set(version_sources(db, revision))
    rows = db.execute("SELECT d.patch,c.revision,c.conditions,c.dependencies FROM decisions d "
        "JOIN claims c ON c.revision=d.revision WHERE d.accepted=1 AND c.id=? AND c.revision<=? "
        "AND json_extract(d.patch,'$.op')='counterexample' ORDER BY c.revision", (claim['id'], revision))
    for row in rows:
        patch = json.loads(row['patch'])
        negative_sources = patch.get('evidence', [])
        sources.update(negative_sources)
        counterexamples.append({'at_revision': row['revision'], 'text': patch['text'],
            'conditions': json.loads(row['conditions']), 'dependencies': json.loads(row['dependencies']),
            'evidence': negative_sources,
            'application': 'historical_negative_under_original_conditions_not_automatically_same_current_conditions'})
    return {'evidence': sorted(sources), 'counterexamples': counterexamples}



def proposition_view(db, claim_id, at_revision, cutoff):
    """Latest proposition whose complete provenance was available at both frontiers."""
    for row in db.execute('SELECT * FROM claims WHERE id=? AND revision<=? ORDER BY revision DESC',
                          (claim_id, at_revision)):
        provenance = claim_provenance(db, row['revision'])
        def available(source):
            ref = db.execute('SELECT run,sequence FROM evidence WHERE id=?', (source,)).fetchone()
            return ref is not None and ref['sequence'] <= cutoff.get(ref['run'], -1)
        if not all(available(source) for source in provenance['evidence']):
            continue
        view = {key: row[key] for key in ('id', 'revision', 'text', 'status', 'epistemic')}
        view.update(conditions=json.loads(row['conditions']), dependencies=json.loads(row['dependencies']),
                    evidence=provenance['evidence'], historical_counterexamples=provenance['counterexamples'],
                    text_role='candidate_proposition_at_knowledge_and_source_frontiers_not_dated_instance')
        review = db.execute('SELECT verdict,reason FROM claim_reviews WHERE revision=?', (row['revision'],)).fetchone()
        if review is not None:
            view['review_required'] = dict(review)
        return view
    return None
