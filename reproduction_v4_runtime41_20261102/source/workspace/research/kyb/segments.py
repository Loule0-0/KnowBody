"""Bounded views of executed history. Evidence remains the SQLite authority."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

from .state import compare_states, evidence_receipt, evidence_state, tracking_residual, body_transition, requested_change
from .instances import dated_instance, source_physics_dt, proposition_view, load_payload


class SegmentReader:
    def __init__(self, db):
        self.db = db
        self.key = None
        self.anchors = {}
        db.executescript('''
        CREATE TEMP TABLE segment_frontier(run TEXT PRIMARY KEY, sequence INTEGER NOT NULL);
        CREATE VIRTUAL TABLE temp.segment_search USING fts5(text);
        ''')

    def _payload(self, identifier):
        return load_payload(self.db, identifier)

    def _available(self, identifier, cutoff):
        row=self.db.execute('SELECT run,sequence FROM evidence WHERE id=?',(identifier,)).fetchone()
        return row is not None and row['sequence']<=cutoff.get(row['run'],-1)

    def _cost(self, rows):
        receipts = [self._payload(row['id']) for row in rows]
        ids = {r['call_id'] for r in receipts if r.get('call_id') is not None}
        source_path = Path(self.db.execute('SELECT path FROM evidence WHERE id=?', (rows[0]['id'],)).fetchone()[0])
        ledger = source_path.parent.parent/'cli_calls.jsonl'
        result = {'executed_receipts':len(rows), 'motion_decision_attempt_ids':sorted(ids),
                  'confirmed_distinct_motion_turns':None, 'total_CLI_invocations_in_span':None,
                  'known_completed_turns_in_span':None, 'outcome_unknown_in_span':None}
        if not ids:
            return {**result, 'confirmed_distinct_motion_turns':0, 'decision_origin':'replayed_commands_no_new_motion_decision'}
        if not ledger.is_file():
            return {**result, 'decision_origin':'source_native_ledger_unavailable_not_assumed_equal_to_receipt_count'}
        events = [json.loads(line) for line in ledger.read_text().splitlines()]
        started = list(dict.fromkeys(e['attempt_id'] for e in events if e['state']=='process_started'))
        completed = {e['attempt_id'] for e in events if e['state']=='turn_complete'}
        result['confirmed_distinct_motion_turns'] = len(ids & completed)
        if ids <= completed and ids <= set(started):
            positions=[started.index(identifier) for identifier in ids]
            span=set(started[min(positions):max(positions)+1])
            result.update(total_CLI_invocations_in_span=len(span), known_completed_turns_in_span=len(span & completed),
                outcome_unknown_in_span=len({e['attempt_id'] for e in events if e['state']=='outcome_unknown'} & span),
                source_ledger=str(ledger), decision_origin='matched_actual_native_attempt_ids_inclusive_first_to_last_motion_span')
        return result

    def _summary(self, rows, entry):
        before=self._payload(rows[0]['before_id']);post=self._payload(rows[-1]['post_id'])
        receipts=[self._payload(row['id']) for row in rows]
        dt=source_physics_dt(self.db,rows[0]['run'])
        changes=[];lengths={arm:0. for arm in ('left','right')}
        for row,receipt in zip(rows,receipts):
            old=evidence_state(self._payload(row['before_id']))['grasp_point_dimensions']
            new=evidence_state(self._payload(row['post_id']))['grasp_point_dimensions']
            delta=requested_change(receipt,self._payload(row['before_id']))
            changes.append({'receipt_id':row['id'],'requested_change_from_measured_before':delta})
            for arm in lengths:
                keys=[f'{arm}_{axis}' for axis in 'xyz']
                if all(key in old and key in new for key in keys):
                    lengths[arm]+=math.sqrt(sum((new[key]-old[key])**2 for key in keys))
        steps=sum(r.get('physics_end',0)-r.get('physics_start',0) for r in receipts)
        return {'before_observation_id':rows[0]['before_id'],'post_observation_id':rows[-1]['post_id'],
            'whole_window_receipt_ids':[row['id'] for row in rows], 'closest_entry_index':entry,
            'measured_start_to_end':body_transition(before,{'physics_start':0,'physics_end':steps},post,dt),
            'sum_formal_endpoint_chords_m':lengths,'endpoint_chords_are_not_full_controller_path':True,
            'requested_changes':changes,'same_named_dimensions':len({tuple(sorted(c['requested_change_from_measured_before'])) for c in changes})==1,
            'actual_native_decision_cost':self._cost(rows),
            'attachment_change_ids':[identifier for row in rows for identifier in row['annotation_ids']
                if self._payload(identifier).get('binding_changed')],
            'qualification':'Measured cumulative body progress and separately dated scene interpretations. '
                'Same requested dimensions do not prove same contact mode; merging these decisions without feedback is untested.'}

    def _effects(self, rows, cutoff, at_revision):
        receipt_ids={row['id'] for row in rows}
        candidates=self.db.execute('SELECT c.revision FROM claims c JOIN interaction_evidence i ON i.revision=c.revision '
            'WHERE c.revision<=? AND (json_extract(i.capsule,\'$.witness.receipt_id\') IN ('+
            ','.join('?' for _ in receipt_ids)+') OR json_extract(i.capsule,\'$.receipt_id\') IN ('+
            ','.join('?' for _ in receipt_ids)+')) ORDER BY c.revision',
            (at_revision,*receipt_ids,*receipt_ids)).fetchall()
        result=[]
        for row in candidates:
            instance=dated_instance(self.db,row['revision'])
            if instance is not None and all(self._available(source,cutoff) for source in instance['evidence']):
                review=self.db.execute('SELECT verdict,reason FROM claim_reviews WHERE revision=?',(row['revision'],)).fetchone()
                if review is not None:instance['review_required']=dict(review)
                instance['proposition_at_read'] = proposition_view(self.db, instance['claim_id'], at_revision, cutoff)
                result.append(instance)
        return result

    def _index(self, cutoff, dependencies, task):
        key = tuple(sorted(cutoff.items()))
        if key == self.key:
            return
        self.db.execute('DELETE FROM temp.segment_frontier')
        self.db.executemany('INSERT INTO temp.segment_frontier VALUES(?,?)', cutoff.items())
        self.db.execute('DELETE FROM temp.segment_search')
        self.anchors = {}
        rows = self.db.execute('''
          SELECT r.id,r.run,r.sequence,b.id AS before_id,p.id AS post_id,p.sequence AS post_sequence,
                 json_extract(b.payload,'$.instruction') AS instruction,
                 json_extract(b.payload,'$.state') AS before_state,
                 json_extract(r.payload,'$.plan.trace.arms') AS targets,
                 json_group_array(a.id) AS annotation_ids,
                 group_concat(a.payload,' ') AS annotations
          FROM evidence r
          JOIN temp.segment_frontier fr ON fr.run=r.run AND r.sequence<=fr.sequence
          JOIN evidence b ON b.id=json_extract(r.payload,'$.before_observation') AND b.kind='observation'
          JOIN evidence p ON json_extract(p.payload,'$.after_receipt')=r.id AND p.kind='observation'
          LEFT JOIN evidence a ON a.kind IN ('attachment_change','attachment_observation')
            AND json_extract(a.payload,'$.observation_id')=p.id
            AND EXISTS(SELECT 1 FROM temp.segment_frontier fa WHERE fa.run=a.run AND a.sequence<=fa.sequence)
          WHERE r.kind='receipt' AND b.run=r.run AND p.run=r.run
            AND b.sequence<r.sequence AND p.sequence>r.sequence AND p.sequence<=fr.sequence
            AND json_extract(r.payload,'$.execution')='complete'
            AND json_extract(r.payload,'$.unknown_command') IS NULL
            AND json_extract(r.payload,'$.executed_actions')>0
            AND json_extract(r.payload,'$.executed_actions')=json_extract(r.payload,'$.accepted_actions')
            AND json_extract(r.payload,'$.executed_actions')=json_extract(r.payload,'$.planned_actions')
          GROUP BY r.id,p.id ORDER BY r.run,r.sequence,p.sequence
        ''').fetchall()
        seen = set()
        for row in rows:
            if row['id'] in seen:
                continue  # First formal post observation, not a later stale receipt reference.
            seen.add(row['id'])
            target = json.loads(row['targets'] or '{}')
            state = json.loads(row['before_state'] or '{}')
            terms = []
            for arm, detail in target.items():
                named = detail.get('named', {})
                if f'{arm}_gripper' in named:
                    terms.append('requested close gripper' if named[f'{arm}_gripper']<.5 else 'requested open gripper')
                dimensions = evidence_state(self._payload(row['before_id']))['grasp_point_dimensions']
                if f'{arm}_z' in named and f'{arm}_z' in dimensions:
                    prior = dimensions[f'{arm}_z']
                    if named[f'{arm}_z']>prior+.001:
                        terms.append('requested lift raise upward')
                    elif named[f'{arm}_z']<prior-.001:
                        terms.append('requested lower descend downward')
                if any(axis in label for label in named for axis in ('pitch_deg','roll_deg','yaw_deg')):
                    terms.append('requested rotate orientation')
            index = len(self.anchors)+1
            annotations=[]
            for identifier in json.loads(row['annotation_ids']):
                if identifier and all(self._available(ref,cutoff) for ref in self._payload(identifier).get('evidence',[])):
                    annotations.append(identifier)
            self.anchors[index] = {**dict(row), 'before_state':state,'annotation_ids':annotations}
            self.db.execute('INSERT INTO temp.segment_search(rowid,text) VALUES(?,?)', (index,
                ' '.join([row['instruction'] or '',row['targets'] or '',
                    *[json.dumps(self._payload(ref)) for ref in annotations],*terms])))
        self.db.commit()
        self.key = key

    def read(self, query, *, context, dependencies, current_observation, episode,
             evidence_cutoff=None, current_history_before=None, at_revision=None, max_receipts=5, limit=1, run=None):
        if limit != 1 or not 1<=max_receipts<=5:
            raise ValueError('Segment readout supports one segment of at most five receipts')
        cutoff = (dict(evidence_cutoff) if evidence_cutoff is not None else {
            row['run']:row['sequence'] for row in self.db.execute(
                'SELECT run,MAX(sequence) AS sequence FROM evidence WHERE run!=? GROUP BY run',(episode,))})
        if current_history_before is None:
            cutoff.pop(episode, None)
        elif evidence_cutoff is None or episode in cutoff:
            # Index known annotations to the real cutoff; only completed old blocks
            # may enter a chain. An interpretation can be published after its post.
            latest_sequence = self.db.execute('SELECT COALESCE(MAX(sequence),0) FROM evidence WHERE run=?', (episode,)).fetchone()[0]
            cutoff[episode] = min(cutoff.get(episode, latest_sequence), latest_sequence)
        if run is not None:
            if run not in cutoff:
                raise ValueError('Requested segment run is not visible at this source frontier')
            cutoff = {run: cutoff[run]}
        current_observation=current_observation or {}
        self._index(cutoff, dependencies, context.get('task'))
        anchors = {key: row for key, row in self.anchors.items()
                   if row['run'] != episode or row['post_sequence'] <= current_history_before}
        terms = re.findall(r'\w+', query, re.UNICODE)[:24]
        if not terms:
            return {'episodic_segments':[], 'retrieval_mode':'no_query'}
        fts_query = ' OR '.join('"'+term.replace('"','""')+'"' for term in terms)
        matches = self.db.execute('SELECT rowid,bm25(segment_search) AS score FROM temp.segment_search '
            'WHERE segment_search MATCH ? ORDER BY score,rowid DESC',(fts_query,)).fetchall()
        if not matches:
            return {'episodic_segments':[], 'retrieval_mode':'no_matched_segment'}
        by_post = {row['post_id']:row for row in anchors.values()}
        by_before = {row['before_id']:row for row in anchors.values()}
        candidates=[]
        for match in matches:
            if match['rowid'] not in anchors:
                continue
            anchor=anchors[match['rowid']]
            chain=[anchor]
            def changed_binding(row):
                return any(self._payload(ref).get('binding_changed') for ref in row['annotation_ids'])
            while len(chain)<max_receipts and chain[0]['before_id'] in by_post:
                prior=by_post[chain[0]['before_id']]
                if changed_binding(prior):
                    break
                chain.insert(0,prior)
            while len(chain)<max_receipts and chain[-1]['post_id'] in by_before:
                if changed_binding(chain[-1]):
                    break
                chain.append(by_before[chain[-1]['post_id']])
            distances=[]
            for row in chain:
                old,new=[],[]
                source_body=self._payload(row['before_id']).get('dependencies',{}).get('body')
                arms=({key.removesuffix('_arm_joint_state') for key in row['before_state'] if key.endswith('_arm_joint_state')}
                      if source_body==dependencies.get('body') else set())
                for arm in arms:
                    key=f'{arm}_arm_joint_state'
                    a,b=row['before_state'].get(key),current_observation.get('state',{}).get(key)
                    if a is not None and b is not None and len(a)==len(b):
                        old.extend(a);new.extend(b)
                distances.append(math.sqrt(sum((a-b)**2 for a,b in zip(old,new))) if old else float('inf'))
            entry=min(range(len(chain)),key=lambda i:distances[i])
            candidates.append((distances[entry],match['score'],-anchor['sequence'],match,anchor,chain,distances,entry))
        if not candidates:
            return {'episodic_segments': [], 'retrieval_mode': 'no_matched_closed_segment'}
        _,_,_,match,anchor,chain,distances,entry=min(candidates,key=lambda item:(item[1],item[0],item[2]))
        whole_window=chain
        summary=self._summary(whole_window,entry)
        at_revision=self.db.execute('SELECT COALESCE(MAX(revision),0) FROM claims').fetchone()[0] if at_revision is None else at_revision
        effects=self._effects(whole_window,cutoff,at_revision)
        chain=chain[entry:]
        states, actions, sources = {}, [], set()
        for row in chain:
            for identifier in (row['before_id'],row['post_id']):
                payload=self._payload(identifier)
                view=evidence_state(payload)
                states[identifier]={key:view[key] for key in ('raw_state','grasp_point_dimensions',
                    'measured_finger_joint_displacements_m')}
                sources.add(identifier)
            receipt=evidence_receipt(self._payload(row['id']))
            interpretations=[{'id':identifier,**self._payload(identifier)} for identifier in row['annotation_ids']]
            sources.update(row['annotation_ids']);sources.add(row['id'])
            for interpretation in interpretations:
                sources.update(interpretation.get('evidence',[]))
            actions.append({'receipt_id':row['id'],'before_observation_id':row['before_id'],
                'post_observation_id':row['post_id'],'executed_actions':receipt['executed_actions'],
                'historical_request':receipt.get('plan',{}).get('trace',{}).get('arms',{}),
                'measured_target_residual':tracking_residual(receipt,self._payload(row['post_id'])),
                'attachment_interpretations':interpretations,
                'post_attachment_inference':{arm:next((item['new'] for item in reversed(interpretations)
                    if item.get('arm')==arm),{'state':'unknown_no_dated_visual_interpretation'})
                    for arm in ('left','right')},
                'source_planner':receipt.get('dependencies',{}).get('planner','not_recorded'),
                'current_planner_comparison':'same' if receipt.get('dependencies',{}).get('planner')==dependencies.get('planner')
                                             and 'planner' in dependencies else 'different_or_unknown_replan_required'})
        available_after={}
        for row in whole_window:
            sources.update((row['before_id'],row['id'],row['post_id'],*row['annotation_ids']))
        for instance in effects:
            sources.update(instance['evidence'])
            if instance['proposition_at_read'] is not None:
                sources.update(instance['proposition_at_read']['evidence'])
        for identifier in sources:
            row=self.db.execute('SELECT run,sequence FROM evidence WHERE id=?',(identifier,)).fetchone()
            available_after[row['run']]=max(available_after.get(row['run'],0),row['sequence'])
        start=self._payload(chain[0]['before_id'])
        segment={'id':anchor['id']+':history','query_bm25':match['score'],
            'whole_window_intervals':[{'receipt_id':row['id'],'before_observation_id':row['before_id'],
                                      'post_observation_id':row['post_id']} for row in whole_window],
            'available_after':available_after,'evidence':sorted(sources),'actions':actions,'states':states,
            'summary':summary, 'dated_effect_refs':effects,
            'entry_comparison':compare_states(start,current_observation),
            'entry_joint_distance_rad_l2':distances[entry] if math.isfinite(distances[entry]) else None,
            'selection':'BM25 question relevance, then closest measured joint entry; no applicability threshold',
            'coordinate_conventions':evidence_state(start)['conventions'],
            'application':'past executed observations, not a causal claim or executable macro; '
                'world targets are historical, current scene alignment and new planning are required; '
                'no retention or rigidity beyond the explicit dated visual interpretations'}
        return {'episodic_segments':[segment], 'retrieval_mode':'fts_bm25_executed_history'}
