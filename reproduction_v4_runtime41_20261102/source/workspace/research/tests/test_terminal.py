"""Terminal authority and restoration checks; no real model or simulator."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from kyb.memory import MemoryStore
from kyb.session import MemorySession
from kyb.terminal import restore_session, review_terminal
from kyb.trace import RunTrace, write_once
from manager_fixtures import ScriptedManager, response


class Policy:
    def __init__(self, turns):
        self.turns = iter(turns)
        self.inputs = []
        self.last_attempt_id = 'offline_policy'
        self.session_id = None
        self.budget = None

    def complete(self, messages, tools):
        self.inputs.append((copy.deepcopy(messages), copy.deepcopy(tools)))
        return response(*next(self.turns))


class TerminalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.store = MemoryStore(root / 'memory.sqlite')
        self.trace = RunTrace(root / 'episode', {'task': 'arbitrary_task', 'prefix_replay': None,
            'perception_profile': 'rgbd_calibrated'})
        self.session = MemorySession(self.trace, self.store, task='arbitrary_task', dependencies={'body': 'body1'})
        self.session.evidence('known_config', {'action_spec': 'static config'})
        self.session.task_specification = self.session.evidence('task_specification', {'instruction': 'original full task'})
        self.session.task_instruction = 'original full task'
        raw = b'offline_pixels'
        image = root / 'view.jpg'
        image.write_bytes(raw)
        self.obs = self.session.evidence('observation', {'state': {}, 'dependencies': {'body': 'body1'},
            'images': {'camera': {'path': str(image), 'sha256': hashlib.sha256(raw).hexdigest()}},
            'instruction': 'original full task', 'env_step': 3})
        self.manager = ScriptedManager([('publish_materials', {'expected_version': 0, 'answer': 'prepared',
            'default_knowledge': []})])
        self.session.attach_manager(self.manager)
        self.session.manager.run('Prepare stable material', 'prepare')
        write_once(self.trace.root / 'report.json', {'status': 'PASS_episode_recorded_not_v1_acceptance'})

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def test_finish_does_not_automatically_call_manager_or_offer_motion(self):
        policy = Policy([('finish_review', {'conclusion': 'Visible outcome only', 'uncertainty': 'Contact unknown'})])
        result, _ = review_terminal(self.session, policy, {'prefix_replay': None})
        self.assertEqual(result['uncertainty'], 'Contact unknown')
        self.assertEqual(len(self.manager.inputs), 1)  # task preparation only
        names = {t['function']['name'] for t in policy.inputs[0][1]}
        self.assertEqual(names, {'query_body_history','query_experience', 'manage_memory', 'finish_review'})
        self.assertEqual(self.store.revision, 0)

    def test_only_policy_request_triggers_manager_then_returns_to_policy(self):
        manager = ScriptedManager([('publish_materials', {'expected_version': 1, 'answer': 'Uncertainty retained',
            'memory_patch': []})])
        self.session.manager.client = manager
        policy = Policy([('manage_memory', {'purpose': 'learn', 'question': 'Compare positive and neutral responses.','episode_outline':[]}),
                         ('finish_review', {'conclusion': 'Reviewed', 'uncertainty': 'Physical mechanism remains unknown'})])
        review_terminal(self.session, policy, {'prefix_replay': None})
        self.assertEqual(len(manager.inputs), 1)
        self.assertEqual(len(policy.inputs), 2)
        self.assertEqual(self.session.manager.version, 2)
        self.assertTrue(any(m.get('role') == 'tool' and 'Uncertainty retained' in m['content']
                            for m in policy.inputs[1][0]))
    def test_last_primary_call_offers_only_explicit_management_or_finish(self):
        from kyb.budget import CallBudget
        policy=Policy([('finish_review',{'conclusion':'Visible result reviewed','uncertainty':'Contact unknown'})])
        policy.budget=CallBudget(Path(self.temp.name)/'primary.jsonl',1)
        review_terminal(self.session,policy,{'prefix_replay':None})
        self.assertEqual([t['function']['name'] for t in policy.inputs[0][1]],['manage_memory','finish_review'])

    def budgeted_policy(self, turns, limit=1):
        from kyb.budget import CallBudget
        policy=Policy(turns)
        policy.budget=CallBudget(Path(self.temp.name)/'last_primary.jsonl',limit)
        complete=policy.complete
        def bounded(messages, tools):
            policy.budget.reserve('last_'+str(len(policy.inputs)), 'policy')
            return complete(messages, tools)
        policy.complete=bounded
        return policy

    def test_last_call_can_explicitly_learn_without_a_fabricated_policy_conclusion(self):
        manager=ScriptedManager([('publish_materials',{'expected_version':1,'answer':'Scoped response retained','memory_patch':[]})])
        self.session.manager.client=manager
        policy=self.budgeted_policy([('manage_memory',{'purpose':'learn','question':'Retain one useful response.','episode_outline':[]})])
        result,messages=review_terminal(self.session,policy,{'prefix_replay':None},official_success=True)
        self.assertEqual(policy.budget.view()['remaining'],0)
        self.assertEqual(len(policy.inputs),1)
        self.assertEqual(len(manager.inputs),1)
        self.assertEqual(self.session.manager.version,2)
        self.assertFalse(result['primary_followup'])
        self.assertNotIn('conclusion',result)
        self.assertEqual(result['last_primary_request'],'manage_memory')
        self.assertIn('Scoped response retained',result['tool_result'])
        self.assertEqual(messages[-1]['role'],'tool')
        row=self.store.db.execute('SELECT outline_ref,result_ref FROM episode_records').fetchone()
        self.assertTrue(row['outline_ref']);self.assertTrue(row['result_ref'])
        kinds=[json.loads(line)['kind'] for line in (self.trace.root/'events.jsonl').read_text().splitlines()]
        self.assertLess(kinds.index('episode_outline'),kinds.index('episode_result'))

    def test_invalid_last_learn_preserves_rejection_without_manager_or_result_delivery(self):
        manager=ScriptedManager([])
        self.session.manager.client=manager
        policy=self.budgeted_policy([('manage_memory',{'purpose':'learn','question':'Retain a response.',
            'episode_outline':[{'summary':'界'*1600}]})])
        result,_=review_terminal(self.session,policy,{'prefix_replay':None},official_success=True)
        self.assertEqual(policy.budget.view()['used'],1)
        self.assertFalse(manager.inputs)
        rejected=json.loads(result['tool_result'])
        self.assertEqual(rejected['status'],'NEEDS_WORK')
        self.assertIn('actual_bytes=',rejected['reason'])
        self.assertIn('excess_bytes=',rejected['reason'])
        row=self.store.db.execute('SELECT outline_ref,result_ref FROM episode_records').fetchone()
        self.assertIsNone(row['outline_ref']);self.assertIsNone(row['result_ref'])
        self.assertNotIn('conclusion',result)

    def test_last_review_without_outline_does_not_gain_a_pre_result_label(self):
        manager=ScriptedManager([('publish_materials',{'expected_version':1,'answer':'Only raw observations compared','memory_patch':[]})])
        self.session.manager.client=manager
        policy=self.budgeted_policy([('manage_memory',{'purpose':'review','question':'What is still unknown?'})])
        review_terminal(self.session,policy,{'prefix_replay':None},official_success=True)
        self.assertEqual(len(manager.inputs),1)
        row=self.store.db.execute('SELECT outline_ref,result_ref FROM episode_records').fetchone()
        self.assertIsNone(row['outline_ref']);self.assertIsNone(row['result_ref'])
        self.assertNotIn('"official_success": true',json.dumps(manager.inputs))

    def test_last_call_with_no_manager_allowance_only_offers_finish(self):
        from kyb.budget import CallBudget
        manager=ScriptedManager([])
        manager.budget=CallBudget(Path(self.temp.name)/'manager_limit.jsonl',1)
        manager.budget.reserve('already_used','manager')
        self.session.manager.client=manager
        policy=self.budgeted_policy([('finish_review',{'conclusion':'No additional management requested','uncertainty':'Unknown response'})])
        review_terminal(self.session,policy,{'prefix_replay':None})
        self.assertEqual([t['function']['name'] for t in policy.inputs[0][1]],['finish_review'])
        self.assertFalse(manager.inputs)

    def test_last_management_runtime_exception_is_not_a_completed_host_record(self):
        manager=ScriptedManager([])
        def failure(*args):
            raise RuntimeError('scripted unknown remote outcome')
        manager.complete=failure
        self.session.manager.client=manager
        policy=self.budgeted_policy([('manage_memory',{'purpose':'learn','question':'Retain a response.','episode_outline':[]})])
        with self.assertRaisesRegex(RuntimeError,'scripted unknown'):
            review_terminal(self.session,policy,{'prefix_replay':None})
        self.assertEqual(policy.budget.view()['used'],1)

    def test_existing_native_session_receives_final_pixels_and_partial_receipt(self):
        receipt=self.session.evidence('receipt',{'before_observation':self.obs['id'],
            'execution':'partial','executed_actions':3,'planned_actions':5,'env_start':3,'env_end':6})
        final=self.session.evidence('observation',{**self.session.current_observation['data'],
            'after_receipt':receipt['id'],'env_step':6,'remaining_steps':94})
        policy=Policy([('finish_review',{'conclusion':'Visible endpoint only','uncertainty':'No private result'})])
        policy.session_id='existing_episode_session'
        review_terminal(self.session,policy,{'prefix_replay':None,'official_success':True})
        messages=policy.inputs[0][0];public=json.loads(messages[1]['content'])
        self.assertEqual(public['final_receipt']['evidence_id'],receipt['id'])
        self.assertEqual(public['final_receipt']['executed_actions'],3)
        self.assertEqual(public['final_receipt']['planned_actions'],5)
        self.assertNotIn('official_success',json.dumps(messages))
        parts=messages[2]['content']
        self.assertEqual(sum(part.get('type')=='image_url' for part in parts),1)
        self.assertIn(final['id'],parts[0]['text'])

    def test_terminal_provenance_follows_manifest_without_private_result(self):
        for prefix in (None, {'source_run': 'archived_episode', 'endpoint': 'archived_episode:e000123',
                              'env_actions': 20, 'prefix_only': False}):
            with self.subTest(prefix=prefix):
                policy = Policy([('finish_review', {'conclusion': 'Observable outcome', 'uncertainty': 'Unknown contact'})])
                review_terminal(self.session, policy, {'prefix_replay': prefix, 'official_success': True})
                public = json.loads(policy.inputs[0][0][1]['content'])
                self.assertEqual(public['prefix_replay'], prefix)
                self.assertNotIn('official_success', public)

    def test_restore_preserves_original_task_and_management_version_without_reprepare(self):
        new_trace = RunTrace(Path(self.temp.name) / 'terminal', {})
        manager = ScriptedManager([])
        restored, manifest = restore_session(new_trace, self.store, self.trace.root, manager)
        self.assertEqual(restored.episode, self.trace.root.name)
        self.assertEqual(restored.task_instruction, 'original full task')
        self.assertEqual(restored.current_observation['id'], self.obs['id'])
        self.assertEqual(restored.manager.version, 1)
        self.assertEqual(restored.manager.default_ids, [])
        review_terminal(restored, Policy([('finish_review', {'conclusion': 'reviewed', 'uncertainty': 'unknown'})]), manifest)
        self.assertFalse(manager.inputs)

    def test_unoffered_motion_cannot_be_dispatched(self):
        policy = Policy([('move_eef', {'targets': {'left_x': 1}, 'note': 'must not execute'})])
        with self.assertRaisesRegex(ValueError, 'No motion tool'):
            review_terminal(self.session, policy, {'prefix_replay': None})
        self.assertEqual(self.store.revision, 0)

    def test_changed_final_pixels_are_rejected_before_model_request(self):
        Path(self.session.current_observation['data']['images']['camera']['path']).write_bytes(b'changed')
        policy = Policy([])
        with self.assertRaisesRegex(ValueError, 'RGB evidence changed'):
            review_terminal(self.session, policy, {'prefix_replay': None})
        self.assertFalse(policy.inputs)


if __name__ == '__main__':
    unittest.main()
