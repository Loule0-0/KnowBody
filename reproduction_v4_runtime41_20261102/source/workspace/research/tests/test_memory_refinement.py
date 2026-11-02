"""Offline checks of source boundaries and semantics, without model or simulator calls."""
import json
from pathlib import Path
import tempfile
import time
import unittest

from kyb import episode_index
from kyb.action_digest import digest
from kyb.readout import preview, statement
from kyb.trace import RunTrace, append
import test_manager
import test_segments
import test_terminal


class EpisodeNavigation(unittest.TestCase):
    def test_decisions_conserve_calls_and_window_covers_last_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace = RunTrace(Path(tmp)/'run', {})
            append(trace.root/'cli_calls.jsonl', {'attempt_id':'carried','role':'policy','state':'turn_complete','time':0})
            append(trace.root/'cli_calls.jsonl', {'attempt_id':'unknown','role':'policy','state':'outcome_unknown','time':0})
            for i in range(9):
                obs=trace.event('observation', {'state':{}})
                append(trace.root/'cli_calls.jsonl', {'attempt_id':str(i),'role':'policy','state':'turn_complete','time':time.time()})
                trace.event('policy_decision', {'call_id':str(i),'tool':'move_eef','observation_id':obs['id']})
                receipt=trace.event('receipt', {'call_id':str(i),'before_observation':obs['id'],
                    'executed_actions':1,'execution':'complete','plan':{'selected_stop_reason':'observation_boundary'}})
                trace.event('observation', {'after_receipt':receipt['id']})
            # A public reference to a non-allowlisted payload must not be opened.
            blocked=trace.event('private_sentinel', {'never':'load this'})
            Path(blocked['path']).write_text('not JSON')
            index=episode_index.build(trace.root)
            a=index['accounting']
            self.assertEqual(a['recorded_attempts'],11)
            self.assertEqual(a['mapped_completed_decisions'],9)
            self.assertEqual(a['carried_completed_attempts'],1)
            self.assertEqual(a['states']['outcome_unknown'],1)
            self.assertEqual(sum(w['calls'] for w in index['windows']),9)
            self.assertEqual(index['windows'][-1]['calls'],1)
            self.assertLessEqual(receipt['sequence'],index['windows'][-1]['events'][1])
            self.assertFalse(index['largest_full_plan_residuals'])
            self.assertEqual(index,episode_index.build(trace.root))
            packet=episode_index.readout(index)
            self.assertLessEqual(len(json.dumps(packet).encode()),2800)


class ReceiptAndNavigation(unittest.TestCase):
    setUp=test_manager.ManagerTests.setUp
    tearDown=test_manager.ManagerTests.tearDown
    observation=test_manager.ManagerTests.observation
    attach=test_manager.ManagerTests.attach
    add_body_case=test_manager.ManagerTests.add_body_case

    def test_primary_defaults_to_statement_without_changing_manager_detail(self):
        c=self.add_body_case()
        args={'question':'gripper marker','level':'case','case_refs':[{'id':c['claim_id'],'revision':c['revision']}]}
        result=json.loads(self.session.query_experience(args))
        self.assertEqual(result['records'][0]['view'],'statement')
        result=json.loads(self.session.query_experience({**args,'view':'detail'}))
        self.assertEqual(result['records'][0]['view'],'detail')
        m=self.attach([]);m.frontier=self.session.source_frontier();m.revision=self.store.revision
        m.knowledge_revision=0;m.purpose='review'
        self.assertEqual(m._search(args)['records'][0]['view'],'detail')
    def test_digest_does_not_silently_read_endpoints(self):
        other=RunTrace(Path(self.temp.name)/'other_episode',{})
        foreign={'state':{},'after_receipt':self.receipt['id']}
        ref=other.event('observation',foreign);self.store.register(ref,foreign)
        m=self.attach([]);m.frontier=self.session.source_frontier();m.purpose='review'
        m.seen_images=set();m.sent_contracts=set()
        m.allowed=set();m.seen_telemetry=set()
        result,_=m._evidence({'ids':[self.receipt['id']],'body':'ee'})
        card=result['records'][0]['data']['action_digest']
        self.assertIsNone(card['measured']);self.assertFalse(m.allowed)
        result,_=m._evidence({'ids':[self.before['id'],self.receipt['id'],self.post['id']],'body':'ee'})
        card=result['records'][1]['data']['action_digest']
        self.assertIsNotNone(card['measured'])
        self.assertEqual(card['source_refs']['receipt'],self.receipt['id'])
        self.assertEqual(card['source_refs']['post'],self.post['id'])
        self.assertEqual(card['measured']['world_body_transition']['readout_qualification'],'unknown')
        self.assertFalse(m.allowed)  # Admission occurs only after successful tool delivery.
        m._admit(result);self.assertEqual(m.allowed,{self.before['id'],self.receipt['id'],self.post['id']})

    def test_statement_and_navigation_keep_material_limitations(self):
        record={'id':'c','revision':1,'layer':'case','domain':'body','family':'geometry',
            'text':'A scoped measured relationship','strength':'observation',
            'conditions':{'mount':'the original attachment only'},'dependencies':{'body':'hash1','cameras':'hash2'},
            'applicability':{'changed_dependencies':['cameras'],'unknown_dependencies':['depth'],
                'different_descriptions':{'mount':{'source':'old','current':'new'}}},
            'counterexamples':[{'revision':2,'text':'Opposing observation','conditions':{'load':'heavy'}}]}
        card=statement(record)
        self.assertEqual(card['conditions'],record['conditions'])
        self.assertEqual(card['applicability'],record['applicability'])
        self.assertEqual(card['counterexamples'],record['counterexamples'])
        self.assertEqual(card['depends_on'],['body','cameras']);self.assertNotIn('hash1',json.dumps(card))
        nav=preview(record);self.assertEqual(nav['conditions'],record['conditions'])
        self.assertEqual(nav['dependency_warnings']['changed_dependencies'],['cameras'])
        self.assertEqual(nav['counterexample_refs'],[2]);self.assertFalse(nav['statement_complete'])
        m=self.attach([]);m.visible_knowledge={};m.visible_revisions={};m.allowed=set()
        m._admit({'records':[nav]});self.assertFalse(m.allowed);self.assertFalse(m.visible_revisions)
        long={**record,'conditions':{'scope':'material condition '*100}}
        self.assertNotIn('conditions',preview(long));self.assertEqual(preview(long)['condition_keys'],['scope'])


class SegmentNavigation(unittest.TestCase):
    setUp=test_segments.SegmentTests.setUp
    tearDown=test_segments.SegmentTests.tearDown
    def test_manager_window_is_navigation_and_respects_frontier(self):
        from kyb.manager import MemoryManager
        from kyb.session import MemorySession
        from kyb.trace import RunTrace
        session=MemorySession(RunTrace(Path(self.temp.name)/'reader',{}),self.store,task='insert_key',dependencies=self.deps)
        m=MemoryManager(session,None);m.purpose='prepare';m.frontier={test_segments.SOURCE.name:478}
        m.request_dependencies=self.deps;m.revision=self.store.revision
        m.allowed=set();m.visible_revisions={};m.visible_knowledge={}
        result=m._segments({'question':'key bow lift'})
        self.assertTrue(result['motion_windows']);m._admit(result);self.assertFalse(m.allowed)
        self.assertLess(len(json.dumps(result).encode()),6000)


class ActualInputDelivery(unittest.TestCase):
    setUp=test_terminal.TerminalTests.setUp
    tearDown=test_terminal.TerminalTests.tearDown

    def test_terminal_first_request_delivers_navigation_without_private_result(self):
        from kyb.terminal import review_terminal
        policy=test_terminal.Policy([('finish_review',{'conclusion':'Observed endpoint','uncertainty':'Unknown contact'})])
        allowed=set(self.session.manager.allowed)
        review_terminal(self.session,policy,{'prefix_replay':None,'official_success':True})
        public=json.loads(policy.inputs[0][0][1]['content'])
        self.assertIn('episode_index',public)
        self.assertEqual(public['episode_index']['accounting']['executed_actions'],0)
        self.assertNotIn('official_success',json.dumps(policy.inputs))
        self.assertEqual(self.session.manager.allowed,allowed)
        self.assertFalse(self.session.manager.visible_revisions)
        self.assertFalse(self.session.manager.visible_knowledge)

    def test_review_first_request_delivers_navigation_without_read_permissions(self):
        from manager_fixtures import ScriptedManager
        manager=self.session.manager
        manager.client=ScriptedManager([('publish_materials',{'expected_version':1,'answer':'No new evidence read.'})])
        manager.run('Compare an unresolved observable response.','review')
        public=json.loads(manager.client.inputs[0][1]['content'])
        self.assertIn('episode_index',public)
        self.assertEqual(public['episode_index']['accounting']['executed_actions'],0)
        self.assertFalse(manager.allowed)
        self.assertFalse(manager.visible_revisions)
        self.assertFalse(manager.visible_knowledge)
        self.assertFalse(manager.seen_telemetry)


class CoverageSemantics(unittest.TestCase):
    def test_prefix_has_no_invented_selected_endpoint_or_limits(self):
        r={'execution':'partial','executed_actions':2,'plan':{'selected_actions':2,
            'full_planned_actions':10,'selected_stop_reason':'observation_boundary'}}
        card=digest(r,before={'state':{}},post={'state':{}},request={})
        self.assertIsNone(card['plan']['full_duration_s'])
        self.assertIn('do not treat a deliberate prefix',card['measured']['goal_comparison_scope'])
        self.assertEqual(card['measured']['selected_endpoint_pose_comparison'],'unknown_no_selected_endpoint_EEF_pose')
        self.assertEqual(len(card['requested']['limits_absent_from_request']),4)
        self.assertIsNone(card['requested']['limits']['execute_max_s'])


if __name__=='__main__':unittest.main()
