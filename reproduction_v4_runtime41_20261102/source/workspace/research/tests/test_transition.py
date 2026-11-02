"""Use the real slip's immutable records in an isolated DB; no new policy/action."""
import hashlib
import json
import tempfile
import sqlite3
import unittest
from pathlib import Path

from kyb.memory import MemoryStore
from kyb.session import MemorySession
from manager_fixtures import PreparationClient
from kyb.trace import RunTrace
from kyb.state import evidence_state

SOURCE = Path('/data/KnowYourBody_RoboDojo/runs/20261018T121514_538047197')


class TransitionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'memory.sqlite'
        self.store = MemoryStore(self.path)
        self.refs, self.data = {}, {}
        for short in ['e000618', 'e000627', 'e000628', 'e000630']:
            path = SOURCE / 'events' / f'{short}.json'
            event = json.loads(path.read_text())
            self.refs[short] = {k: event[k] for k in ['id', 'sequence', 'kind', 'visibility']}
            self.refs[short].update(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            self.data[short] = event['payload']
            if short != 'e000630':
                self.store.register(self.refs[short], event['payload'])
        self.start = self.data['e000627']['dependencies']
        self.end = {**self.start, 'attachment:left': self.refs['e000630']['id']}
        self.capsule = {
            'before_observation_id': self.refs['e000618']['id'],
            'receipt_id': self.refs['e000627']['id'],
            'post_observation_id': self.refs['e000628']['id'],
            'attachment_event_id': self.refs['e000630']['id'],
            'before_attachment': self.data['e000630']['old'], 'after_attachment': self.data['e000630']['new'],
            'execution_dependencies': self.start, 'post_dependencies': self.end,
            'evidence': [r['id'] for r in self.refs.values()]}
        self.store.publish_attachment(self.refs['e000630'], self.data['e000630'],
            observation_id=self.refs['e000628']['id'], dependencies=self.end, capsule=self.capsule)
        self.allowed = {r['id'] for r in self.refs.values()}
        self.visible = {}

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def patch(self, **changes):
        return {'op': 'new', 'reading_role':'interaction', 'kind': 'external', 'module': 'retention',
            'relation_type': 'observed_interaction', 'pending_event_id': self.refs['e000630']['id'],
            'text': 'After this observed vertical rise, the key was no longer retained; causality is unresolved.',
            'conditions': {'task': 'insert_key', 'grip_region': 'toothed end'},
            'reference_frame': 'world', 'epistemic': 'visually_inferred',
            'evidence': [self.refs['e000630']['id']], 'depends_on': [], **changes}

    def submit(self, patch, cutoff=630):
        return self.store.submit(patch, allowed_sources=self.allowed,
            cutoff={SOURCE.name: cutoff}, dependencies=self.end, episode=SOURCE.name,
            visible_revisions=self.visible)

    def test_stable_effect_cannot_hide_binding_change_by_omitting_dependency(self):
        result = self.submit(self.patch(relation_type='stable_response',
            evidence=sorted(self.allowed), receipt_id=self.refs['e000627']['id'],
            post_observation_id=self.refs['e000628']['id']))
        self.assertFalse(result['accepted'])
        self.assertEqual(result['reason'], 'external_requires_observed_interaction')
        self.assertEqual(len(self.store.pending()), 1)

    def test_observation_is_not_a_pending_marker_and_feedback_explains_historical_omission(self):
        wrong = self.refs['e000628']['id']
        result = self.submit(self.patch(pending_event_id=wrong))
        self.assertFalse(result['accepted'])
        self.assertIn('pending_event_id=' + wrong, result['reason'])
        self.assertIn('omit this optional field', result['reason'])
        self.assertIn('relation_type=observed_interaction', result['reason'])
        self.assertEqual(len(self.store.pending()), 1)

    def test_live_pending_and_retrieved_source_share_the_same_coordinate_view(self):
        expected = evidence_state(self.data['e000628'])
        pending = self.store.pending()[0]
        self.assertEqual(pending['post_state'], expected)
        trace = RunTrace(Path(self.temp.name) / 'view', {})
        session = MemorySession(trace, self.store, task='insert_key', dependencies=self.end)
        session.attach_manager(PreparationClient())
        session.current_observation = {**self.refs['e000628'], 'data': self.data['e000628']}
        messages = [{'role':'user','content':[{'type':'text','text':'current'},
                                             {'type':'text','text':'RGB_REFERENCE '+json.dumps({'observation_id':session.current_observation['id'],'camera':'head','temporal_role':'new_observation'})},
                                             {'type':'image_url','image_url':{'url':'dummy'}}]}]
        session.refresh_context(messages, None)
        from kyb.context import latest_context
        context=latest_context(messages)
        self.assertNotIn('objects',context)
        self.assertEqual(session.current_observation['data'],self.data['e000628'])
        result = self.submit(self.patch())
        self.assertTrue(result['accepted'], result)
        read = self.store.read('key', context={'task':'insert_key'}, dependencies=self.end, episode='new')
        self.assertEqual(read['historical_events'][0]['interaction']['post_state'], expected)

    def test_transition_remains_historical_after_regrasp_and_is_consumed_once(self):
        result = self.submit(self.patch())
        self.assertTrue(result['accepted'], result)
        self.assertFalse(self.store.pending())
        stored = self.store.db.execute('SELECT capsule FROM interaction_evidence').fetchone()[0]
        self.assertEqual(json.loads(stored)['execution_dependencies'], self.start)
        self.assertEqual(json.loads(stored)['post_dependencies'], self.end)
        read = self.store.read('key', context={'task': 'insert_key', 'grip_region': 'bow'},
            dependencies={**self.end, 'attachment:left': 'new_bow_grasp'}, episode='new_layout')
        self.assertFalse(read['selected'])
        self.assertEqual(len(read['historical_events']), 1)
        item=read['historical_events'][0]
        self.assertEqual(item['qualification']['physical_conditions']['grip_region'],'unknown')
        self.assertIn('grip_region',item['qualification']['different_descriptions'])
        self.assertFalse(self.submit(self.patch())['accepted'])
        self.assertEqual(self.store.revision, 1)

    def test_pending_survives_reopen_and_rejection_does_not_resolve(self):
        self.store.close()
        self.store = MemoryStore(self.path)
        self.assertEqual(len(self.store.pending()), 1)
        self.assertFalse(self.submit(self.patch(), cutoff=628)['accepted'])
        self.assertFalse(self.submit(self.patch(epistemic='hypothesis'))['accepted'])
        self.assertEqual(len(self.store.pending()), 1)
        self.assertEqual(self.store.revision, 0)

    def test_atomic_batch_rejection_restores_pending_slip_and_claims(self):
        decisions,committed=self.store.submit_batch([self.patch(),self.patch(evidence=['unprovided:e000001'])],
            allowed_sources=self.allowed,cutoff={SOURCE.name:630},dependencies=self.end,
            episode=SOURCE.name,visible_revisions=self.visible)
        self.assertFalse(committed);self.assertTrue(all(not d['accepted'] for d in decisions))
        self.assertEqual(self.store.revision,0);self.assertEqual(len(self.store.pending()),1)

    def test_description_revision_does_not_change_physical_binding(self):
        trace = RunTrace(Path(self.temp.name) / 'description', {})
        session = MemorySession(trace, self.store, task='insert_key', dependencies=self.start)
        session.attachment['left'] = {'tool': 'key', 'state': 'visually_supported', 'grip_region': 'toothed end'}
        observation = session.evidence('observation', {'dependencies': self.start})
        session.consume({'attachment_updates': [{'arm': 'left', 'tool': 'key',
            'state': 'visually_supported', 'grip_region': 'toothed end; rigidity unknown',
            'binding_event': 'unchanged', 'evidence': [observation['id']]}]}, None)
        self.assertEqual(session.dependencies['attachment:left'], self.start['attachment:left'])
        self.assertEqual(len(self.store.pending()), 1)  # Only the real fixture slip remains.

    def test_obsolete_relation_input_is_rejected_without_compatibility(self):
        result=self.submit(self.patch(relation_type='stable_response'))
        self.assertFalse(result['accepted'])
        self.assertEqual(result['reason'],'external_requires_observed_interaction')
        self.assertEqual(self.store.revision,0)

    def test_pending_resolution_failure_rolls_back_claim_publication(self):
        self.store.db.executescript("CREATE TRIGGER fail_resolve BEFORE UPDATE ON pending_events "
                                   "BEGIN SELECT RAISE(ABORT,'synthetic transaction interruption'); END;")
        with self.assertRaises(sqlite3.IntegrityError):
            self.submit(self.patch())
        self.assertEqual(self.store.revision, 0)
        self.assertEqual(len(self.store.pending()), 1)

    def test_repeated_loss_is_not_a_new_event_and_regain_changes_binding(self):
        trace = RunTrace(Path(self.temp.name) / 'regain', {})
        session = MemorySession(trace, self.store, task='insert_key', dependencies=self.start)
        session.attachment['left'] = {'tool':'key','state':'slipped','grip_region':'unknown'}
        lost = session.evidence('observation', {'dependencies':self.start})
        session.consume({'attachment_updates':[{'arm':'left','tool':'key','state':'slipped',
            'evidence':[lost['id']]}]}, None)
        self.assertEqual(session.dependencies['attachment:left'], self.start['attachment:left'])
        retained = session.evidence('observation', {'dependencies':self.start})
        update = {'arm':'left','tool':'key','state':'visually_supported','grip_region':'bow',
                  'evidence':[retained['id']]}
        session.consume({'attachment_updates':[update]}, None)
        regained = session.dependencies['attachment:left']
        self.assertNotEqual(regained, self.start['attachment:left'])
        session.consume({'attachment_updates':[{**update,'binding_event':'regrasp'}]}, None)
        self.assertEqual(session.dependencies['attachment:left'], regained)
        self.assertEqual(session.last_decisions[0]['reason'], 'duplicate_physical_interpretation_of_same_observation')
