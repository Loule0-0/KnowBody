"""History reaches the actor without changing its current state or writer rights."""
import base64
import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
import test_physical_memory as physical
import test_codex_cli as transport
from manager_fixtures import PreparationClient
from kyb import readout
from kyb.codex_cli import CodexCliClient, CodexCliFailure
from kyb.perception import RGB_ONLY
from kyb.policy import EefAgentPolicy
from kyb.session import MemorySession
from kyb.trace import RunTrace
from kyb.witness import tool


def payload(parts):
    return json.loads(parts[0]['text'])


class CaseWitnessTests(unittest.TestCase):
    add = physical.PhysicalMemoryTests.add
    patch = physical.PhysicalMemoryTests.patch
    submit = physical.PhysicalMemoryTests.submit

    def observation(self, x, **extra):
        from kyb.robodojo_adapter import canonical_state
        state = {'a_ee_pose': [x, 0., .9, 1., 0., 0., 0.],
                 'a_arm_joint_state': [0., 0.], 'a_ee_joint_state': [0.]}
        images = {}
        for camera in ('head', 'wrist', 'third'):
            path = self.root / f'{self.trace.sequence}_{camera}.jpg'
            Image.new('RGB', (8, 8), (int(x * 100) % 255, 40, 80)).save(path)
            images[camera] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        return self.add('observation', {'state': state, 'dependencies': self.deps,
            'canonical_state': canonical_state(state, {}, 'body1'), 'images': images, **extra})

    def setUp(self):
        physical.PhysicalMemoryTests.setUp(self)
        self.add('known_config', {'perception_profile': 'rgb_only'})
        first = self.submit(self.patch())
        self.assertTrue(first['accepted'], first)
        self.first = {'id': first['claim_id'], 'revision': first['revision']}
        self.original = self.before
        self.shared = self.post
        self.before = self.post
        self.receipt = self.add('receipt', {'before_observation': self.before['id'],
            'dependencies': self.deps, 'executed_actions': 2, 'planned_actions': 3,
            'execution': 'partial', 'unknown_command': None})
        self.post = self.observation(.2, after_receipt=self.receipt['id'])
        self.sources = {self.before['id'], self.receipt['id'], self.post['id']}
        second = self.submit(self.patch(text='The retained object moved further.'))
        self.assertTrue(second['accepted'], second)
        self.second = {'id': second['claim_id'], 'revision': second['revision']}
        self.session = MemorySession(RunTrace(self.root / 'reader', {}), self.store,
            task='new task', dependencies=self.deps)
        from kyb.library import organize_archive
        organize_archive(self.store)

    tearDown = physical.PhysicalMemoryTests.tearDown

    def authorize(self, refs=None, **extra):
        from kyb.library import organize_archive
        organize_archive(self.store)
        result = json.loads(self.session.query_experience({'level': 'case',
            'case_refs': refs or [self.first, self.second], **extra}))
        self.assertTrue(all(r.get('statement_complete') for r in result['records']), result)
        return result['records']

    def read(self, **extra):
        return self.session.view_case_witness({'cases': [self.first, self.second],
            'cameras': ['head', 'wrist'], **extra})

    def test_complete_cards_six_frames_and_shared_endpoint_links(self):
        cards = self.authorize()
        before_changes = self.store.db.total_changes
        old_allowed = set(self.session.allowed)
        self.session.manager = SimpleNamespace(allowed=set(), seen_images=set(),
            seen_telemetry=set(), sent_contracts=set())
        result = self.read()
        value = payload(result)
        self.assertEqual(value['status'], 'PASS')
        self.assertEqual(value['cases'], cards)
        self.assertEqual(len(value['frames']), 6)
        self.assertEqual(sum(p['type'] == 'image_url' for p in result), 6)
        shared = [f for f in value['frames'] if f['observation_id'] == self.shared['id']]
        self.assertEqual(len(shared), 2)
        self.assertTrue(all(len(f['links']) == 2 for f in shared))
        self.assertLessEqual(sum(len(p['text'].encode()) for p in result if p['type']=='text'), 24576)
        self.assertEqual(self.store.db.total_changes, before_changes)
        self.assertEqual(self.session.allowed, old_allowed)
        self.assertTrue(all(not v for v in vars(self.session.manager).values()))
        for part in result[1::2]:
            self.assertEqual(json.loads(part['text'][14:])['temporal_role'], 'historical_case_witness')

    def test_unread_preview_wrong_revision_and_foreign_store_are_rejected(self):
        self.assertEqual(payload(self.read())['status'], 'NEEDS_WORK')
        cards = self.authorize()
        self.assertEqual(payload(self.read(cases=[{**self.first, 'revision': 999}]))['status'], 'NEEDS_WORK')
        saved = self.session._case_witness_reads
        self.session._case_witness_reads = {}
        readout.note_delivery(self.session, 'policy:' + self.session.episode,
                             {'records': [readout.preview(cards[0])]})
        self.assertEqual(payload(self.read(cases=[self.first]))['status'], 'NEEDS_WORK')
        self.session._case_witness_reads = saved
        from kyb.memory import MemoryStore
        original_store = self.session.store
        foreign = MemoryStore(self.root / 'foreign.sqlite')
        self.session.store = foreign
        try:
            self.assertEqual(payload(self.read())['status'], 'NEEDS_WORK')
        finally:
            self.session.store = original_store
            foreign.close()

    def test_completed_statement_pages_allow_reading(self):
        answer = json.loads(self.session.query_experience({'level': 'case',
            'case_refs': [self.first], 'budget_bytes': 1100}))
        self.assertEqual(payload(self.read(cases=[self.first]))['status'], 'NEEDS_WORK')
        while not readout.delivered_statement(self.session, 'policy:' + self.session.episode,
                                               'case', self.first['id'], self.first['revision']):
            token = answer['records'][0].get('continuation') or answer['records'][0].get('read')
            self.assertIsInstance(token, str, answer)
            answer = json.loads(self.session.query_experience({'continuation': token, 'budget_bytes': 1100}))
        self.assertEqual(payload(self.read(cases=[self.first]))['status'], 'PASS')

    def test_frontier_camera_and_extra_fields_fail_without_db_or_media_effects(self):
        self.authorize()
        saved = self.session._case_witness_reads[(self.first['id'], self.first['revision'])]
        saved['frontier'][self.trace.root.name] = 0
        before = self.store.db.total_changes
        self.assertEqual(payload(self.read())['status'], 'NEEDS_WORK')
        self.assertEqual(self.store.db.total_changes, before)
        saved['frontier'] = self.session.source_frontier()
        for extra in ({'cameras': ['missing']}, {'budget': 100}, {'receipts': ['foreign:receipt']}):
            answer = self.read(**extra)
            self.assertEqual(payload(answer)['status'], 'NEEDS_WORK')
            self.assertEqual(len(answer), 1)
        with patch.object(self.session, 'source_frontier', return_value={self.trace.root.name: 0}):
            self.assertEqual(payload(self.read())['status'], 'NEEDS_WORK')

    def test_event_and_image_tampering_are_rejected(self):
        self.authorize()
        row = self.store.db.execute('SELECT path FROM evidence WHERE id=?', (self.original['id'],)).fetchone()
        path = Path(row['path']); raw = path.read_bytes()
        path.write_bytes(raw + b' ')
        self.assertIn('source file changed', payload(self.read())['reason'])
        path.write_bytes(raw)
        image = Path(self.store.evidence_payload(self.original['id'])['images']['head']['path'])
        image.write_bytes(b'changed')
        self.assertIn('RGB changed', payload(self.read())['reason'])

    def test_consistent_event_and_registration_rewrite_is_not_old_delivery(self):
        self.authorize()
        row = self.store.db.execute('SELECT * FROM evidence WHERE id=?', (self.original['id'],)).fetchone()
        path = Path(row['path']); event = json.loads(path.read_text())
        event['payload']['rewritten_after_delivery'] = True
        raw = json.dumps(event).encode(); path.write_bytes(raw)
        with self.store.db:
            self.store.db.execute('UPDATE evidence SET payload=?,sha256=? WHERE id=?',
                (json.dumps(event['payload']), hashlib.sha256(raw).hexdigest(), self.original['id']))
        answer = self.read()
        self.assertEqual(payload(answer)['status'], 'NEEDS_WORK')
        self.assertEqual(len(answer), 1)

    def test_counter_frames_remain_labelled_and_card_is_not_shortened(self):
        result = self.submit(self.patch(op='counterexample', claim_id=self.first['id'],
            claim_revision=self.first['revision'], depends_on=['attachment:a'], text='Changed response.'),
            visible={self.first['id']: self.first['revision']})
        self.assertTrue(result['accepted'], result)
        ref = {'id': self.first['id'], 'revision': result['revision']}
        cards = self.authorize([ref])
        value = payload(self.read(cases=[ref]))
        self.assertEqual(value['status'], 'PASS')
        self.assertEqual(value['cases'], cards)
        self.assertTrue(value['cases'][0]['counterexamples'])
        self.assertIn('counterexample', {link['case_update'] for frame in value['frames'] for link in frame['links']})

    def test_budget_includes_rgb_reference_text(self):
        self.authorize([self.first])
        saved = self.session._case_witness_reads[(self.first['id'], self.first['revision'])]
        saved['card']['conditions']['boundary'] = ''
        plain = payload(self.read(cases=[self.first]))
        self.assertEqual(plain['status'], 'PASS')
        filler = 'x' * (24500 - readout.size(plain))
        plain['cases'][0]['conditions']['boundary'] = filler
        self.assertLess(readout.size(plain), 24576)
        saved['card']['conditions']['boundary'] = filler
        answer = self.read(cases=[self.first])
        self.assertEqual(payload(answer)['status'], 'NEEDS_WORK')
        self.assertGreater(payload(answer)['required_text_bytes'], 24576)
        self.assertEqual(len(answer), 1)

    def test_oversize_has_no_images_and_subset_is_explicit(self):
        self.authorize()
        value = payload(self.read(cameras=['head', 'wrist', 'third']))
        self.assertEqual(value['status'], 'NEEDS_WORK')
        self.assertEqual(value['required_images'], 9)
        self.assertEqual(len(value['witnesses']), 2)
        # Add a second registered witness to the same proposition via support.
        result = self.submit(self.patch(op='support', claim_id=self.first['id'],
            claim_revision=self.first['revision'], depends_on=['attachment:a']),
            visible={self.first['id']:self.first['revision']})
        self.assertTrue(result['accepted'], result)
        supported = {'id': self.first['id'], 'revision': result['revision']}
        self.authorize([supported])
        value = payload(self.read(cases=[supported], receipts=[self.receipt['id']]))
        self.assertEqual(value['status'], 'PASS')
        self.assertEqual(value['frames_scope'], 'selected')
        self.assertEqual(len(value['omitted']), 1)

    def test_rgb_only_own_chain_positive_and_privileged_source_negative(self):
        self.session = MemorySession(RunTrace(self.root / 'rgb_reader', {}), self.store,
            task='new task', dependencies=self.deps, perception_profile='rgb_only')
        self.authorize()
        self.assertEqual(payload(self.read())['status'], 'PASS')
        original_payload = self.store.evidence_payload
        def contaminated(source):
            data = original_payload(source)
            if source == self.original['id']:
                data['body_visual_reference'] = {'privileged': True}
            return data
        with patch.object(self.store, 'evidence_payload', side_effect=contaminated):
            self.assertIn('RGB-only', payload(self.read())['reason'])

    def test_explicit_subset_does_not_require_unselected_cameras(self):
        result = self.submit(self.patch(op='support', claim_id=self.first['id'],
            claim_revision=self.first['revision'], depends_on=['attachment:a']),
            visible={self.first['id']:self.first['revision']})
        self.assertTrue(result['accepted'], result)
        ref = {'id':self.first['id'], 'revision':result['revision']}
        row = self.store.db.execute('SELECT * FROM evidence WHERE id=?', (self.original['id'],)).fetchone()
        path = Path(row['path']); event = json.loads(path.read_text())
        event['payload']['images'].pop('wrist')
        raw = json.dumps(event).encode(); path.write_bytes(raw)
        with self.store.db:
            self.store.db.execute('UPDATE evidence SET payload=?,sha256=? WHERE id=?',
                (json.dumps(event['payload']), hashlib.sha256(raw).hexdigest(), self.original['id']))
        self.authorize([ref])
        value = payload(self.read(cases=[ref], cameras=['wrist'], receipts=[self.receipt['id']]))
        self.assertEqual(value['status'], 'PASS')
        self.assertEqual(len(value['frames']), 2)
        self.assertEqual(len(value['omitted']), 1)

    def test_terminal_tool_not_offered_or_dispatched(self):
        from kyb.knowledge import policy_memory_tools
        self.assertNotIn('view_case_witness', [t['function']['name'] for t in policy_memory_tools()])
        self.authorize(); self.session.episode_ended = True
        self.assertEqual(payload(self.read())['status'], 'NEEDS_WORK')

    def test_real_dispatch_fake_cli_and_current_context_stay_separate(self):
        self.authorize()
        self.session.attach_manager(PreparationClient())
        current = self.session.evidence('observation', {'state': {}, 'images':
            self.store.evidence_payload(self.post['id'])['images'], 'dependencies': self.deps})
        def current_message(ref):
            from kyb.media import rgb_parts, read_rgb
            data = self.store.evidence_payload(ref['id'])
            return {'role': 'user', 'content': [{'type': 'text', 'text': 'CURRENT STATE'},
                *rgb_parts([({'observation_id': ref['id'], 'camera': c}, raw)
                    for c, raw in read_rgb(data['images'], ['head'])], temporal_role='new_observation')]}
        policy = object.__new__(EefAgentPolicy); policy.research = self.session
        executable = self.root / 'fake-codex'; executable.write_text(transport.FAKE); executable.chmod(0o700)
        with patch.dict(os.environ, {'CODEX_HOME': str(self.root / 'fake-home')}), \
                patch('kyb.native_dispatch.MIN_START_INTERVAL_SECONDS', .001):
            client = CodexCliClient(self.root / 'transport', executable=str(executable), timeout=3)
            messages = [current_message(current)]
            self.session.refresh_context(messages, None)
            response = client.complete(messages, [tool()]); sid = client.session_id
            outcome = policy._handle_motion('view_case_witness', {'cases': [self.first, self.second],
                'cameras': ['head', 'wrist']}, None)
            call = response['choices'][0]['message']['tool_calls'][0]
            messages.extend([response['choices'][0]['message'], policy._tool_result(call['id'], outcome.tool_result)])
            old_historical = copy.deepcopy(messages[-1])
            self.session.refresh_context(messages, None)
            self.assertEqual(messages[-1], old_historical)
            self.assertIn(current['id'], messages[1]['content'][0]['text'])
            client.complete(messages, [tool()])
            folder = client.root / 'cli' / client.last_attempt_id / 'input'
            media = json.loads((folder / 'image_manifest.json').read_text())
            self.assertTrue(all(m['temporal_role'] == 'historical_case_witness' for m in media))
            self.assertEqual(client.session_id, sid)
            prompt = json.loads((folder / 'prompt.txt').read_text())
            self.assertEqual(prompt['current_readout']['current_observation_id'], current['id'])
            metrics = json.loads((folder / 'context_metrics.json').read_text())
            self.assertEqual(metrics['new_experience_result_utf8_bytes_by_tool']['view_case_witness'],
                sum(len(p['text'].encode()) for p in outcome.tool_result if p['type']=='text'))
            self.assertEqual(prompt['conversation'][0]['role'], 'tool')
            self.assertEqual(prompt['conversation'][0]['tool_call_id'], call['id'])
            new = self.session.evidence('observation', {'state': {}, 'images':
                self.store.evidence_payload(self.post['id'])['images'], 'dependencies': self.deps})
            messages.append(current_message(new)); self.session.refresh_context(messages, None)
            self.assertIn(new['id'], messages[-1]['content'][0]['text'])
            self.assertEqual(old_historical, messages[-2])
            client.complete(messages, [tool()])
            self.assertEqual(client.session_id, sid)
            prompt = json.loads((client.root/'cli'/client.last_attempt_id/'input/prompt.txt').read_text())
            self.assertEqual(prompt['current_readout']['current_observation_id'], new['id'])
            from kyb.terminal import _observation
            client.complete([_observation(self.session.current_observation)],
                            [{'function': {'name': 'finish_review', 'parameters': {}}}])
            self.assertEqual(client.session_id, sid)


class WitnessIdentityTests(unittest.TestCase):
    setUp = transport.CodexCliTests.setUp
    tearDown = transport.CodexCliTests.tearDown
    observation = transport.CodexCliTests.observation

    def test_unlabelled_or_misplaced_history_rejected_before_cache_write(self):
        current = self.observation(0)
        bad = copy.deepcopy(current)
        bad['content'][1]['text'] = 'RGB_REFERENCE ' + json.dumps({
            'observation_id': 'old', 'camera': 'head', 'temporal_role': 'historical_case_witness'})
        folder = self.root / 'input'; folder.mkdir()
        with self.assertRaises(CodexCliFailure):
            self.client._input([current, bad], folder)
        self.assertFalse(list((self.client.workspace / 'images').iterdir()))
        # A second image cannot inherit the first image's reference.
        repeated = copy.deepcopy(current); repeated['content'].append(current['content'][-1])
        with self.assertRaises(CodexCliFailure):
            self.client._input([repeated], folder)
        self.assertFalse(list((self.client.workspace / 'images').iterdir()))

    def test_identical_pixels_keep_distinct_current_and_historical_identity(self):
        current = self.observation(0)
        historical = copy.deepcopy(current)
        historical.update(role='tool', tool_call_id='history')
        reference = {'observation_id': 'old:e1', 'camera': 'head',
            'temporal_role': 'historical_case_witness', 'source_episode': 'old',
            'sha256': hashlib.sha256(b'image0').hexdigest(),
            'links': [{'case': {'id': 'case', 'revision': 1}, 'receipt_id': 'old:e0', 'phase': 'post'}]}
        historical['content'][1]['text'] = 'RGB_REFERENCE ' + json.dumps(reference)
        folder = self.root / 'input_identity'; folder.mkdir()
        _, images = self.client._input([current, historical], folder)
        media = json.loads((folder/'image_manifest.json').read_text())
        self.assertEqual(len(images), 1)
        self.assertEqual(len(media), 2)
        self.assertEqual({m['temporal_role'] for m in media}, {'new_observation', 'historical_case_witness'})

    def test_readout_cannot_be_bound_to_another_current_image(self):
        from kyb.context import latest_context
        message = self.observation(0)
        message['content'][0]['text'] += '\n<KYB_CONTEXT>\n' + json.dumps(
            {'current_observation_id': 'different'}) + '\n</KYB_CONTEXT>'
        with self.assertRaisesRegex(ValueError, 'Current readout'):
            latest_context([message])


if __name__ == '__main__':
    unittest.main()
