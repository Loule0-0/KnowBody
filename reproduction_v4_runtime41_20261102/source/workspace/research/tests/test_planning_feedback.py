"""Zero-execution cost feedback remains readable without becoming physical evidence."""
import copy
import json
import unittest
from dataclasses import replace
import numpy as np

import test_policy_pipeline as fixture
from kyb import episode_index
from kyb.policy import EefAgentPolicy


class PlanningFeedback(unittest.TestCase):
    tearDown=fixture.PolicyPipelineTests.tearDown

    def setUp(self):
        fixture.PolicyPipelineTests.setUp(self)
        self.deps={**self.deps,'execution':'exec1'}
        self.session.dependencies.update(self.deps)
        self.after=self.session.evidence('observation',{'dependencies':dict(self.deps),
            'state':self.session.current_observation['data']['state'],'after_receipt':self.receipt['id']})
        self.observation=replace(self.observation,extra={**self.observation.extra,'evidence_id':self.after['id']})

    def policy(self,*,guard=True,success=True):
        def planner(**kwargs):
            return {'status':'Success','position':np.array([np.ones(6)*.2,np.ones(6)*.4])} if success else {'status':'Fail'}
        def validate(actions,observation):
            self.session.evidence('path_validation',{'status':'PASS','executed_actions':0,
                'before_observation_id':observation.extra['evidence_id'],'dependencies':dict(self.deps)})
            return {'status':'PASS'}
        return EefAgentPolicy(action_spec=self.spec,env={},client=fixture.Client([]),planner=planner,
            research=self.session,path_validator=validate if guard else None)

    def feedback(self,**arguments):
        p=self.policy();p._gripper_goals={'left':1.,'right':1.};p._last_targets={}
        old_receipt=self.session.last_receipt['id']
        request={'targets':{'left_z':.97,'left_gripper':0},**arguments}
        result=p._handle_motion('move_eef',request,self.observation)
        self.assertIsNone(result.chunk);self.assertFalse(result.repairable)
        self.assertEqual(p._gripper_goals,{'left':1.,'right':1.});self.assertEqual(p._last_targets,{})
        self.assertEqual(self.session.last_receipt['id'],old_receipt);self.assertIsNone(p._last_executed_action)
        value=json.loads(result.tool_result);data=self.store.evidence_payload(value['evidence_id'])
        self.assertEqual(data['executed_actions'],0);self.assertEqual(data['physics_steps'],0)
        self.assertEqual(data['before_observation_id'],self.after['id'])
        self.assertEqual(data['arguments'],request);self.assertEqual(data['dependencies'],self.deps)
        self.assertEqual(data['max_single_joint_travel_deg'],value['max_single_joint_travel_deg'])
        return value,data

    def test_preview_feedback_is_registered_and_does_not_execute_or_latch(self):
        value,data=self.feedback(preview=True)
        self.assertEqual(value['status'],'preview');self.assertEqual(data['status'],'preview')
        self.assertEqual(len(data['planning_refs']),1);self.assertEqual(len(data['path_validation_refs']),1)
        self.assertIn('not planner failure',data['interpretation'])

    def test_duration_limit_is_cost_rejection_not_planner_failure(self):
        value,data=self.feedback(max_duration_s=.001)
        self.assertEqual(value['status'],'cost_limit_rejected')
        self.assertGreater(data['full_duration_s'],data['arguments']['max_duration_s'])
        self.assertEqual(self.store.evidence_payload(data['planning_refs'][0])['status'],'Success')

    def test_joint_limit_overrides_preview_but_has_no_execution(self):
        value,data=self.feedback(preview=True,max_joint_travel_deg=.1)
        self.assertEqual(value['status'],'cost_limit_rejected')
        self.assertGreater(data['max_single_joint_travel_deg'],.1)

    def test_same_observation_repeated_calls_bind_only_their_own_plans_and_guards(self):
        _,first=self.feedback(preview=True);_,second=self.feedback(max_joint_travel_deg=.1)
        self.assertTrue(set(first['planning_refs']).isdisjoint(second['planning_refs']))
        self.assertTrue(set(first['path_validation_refs']).isdisjoint(second['path_validation_refs']))
        for data in (first,second):
            for ref in data['planning_refs']+data['path_validation_refs']:
                self.assertEqual(self.store.evidence_payload(ref)['before_observation_id'],self.after['id'])

    def test_hold_without_new_planner_or_guard_does_not_reuse_stale_refs(self):
        self.feedback(preview=True)
        p=self.policy(guard=False)
        outcome=p._handle_motion('hold',{'seconds':.2,'preview':True},self.observation)
        data=self.store.evidence_payload(json.loads(outcome.tool_result)['evidence_id'])
        self.assertEqual(data['planning_refs'],[]);self.assertEqual(data['path_validation_refs'],[])
        self.assertEqual(data['tool'],'hold');self.assertEqual(data['full_planned_actions'],5)

    def test_accepted_plan_and_failed_planner_do_not_create_cost_feedback(self):
        p=self.policy();outcome=p._handle_motion('move_eef',{'targets':{'left_z':.97}},self.observation)
        self.assertIsNotNone(outcome.chunk)
        p=self.policy(success=False)
        outcome=p._handle_motion('move_eef',{'targets':{'left_z':.98},'preview':True},self.observation)
        self.assertIsNone(outcome.chunk)
        self.assertEqual(self.store.db.execute("SELECT COUNT(*) FROM evidence WHERE kind='planning_feedback'").fetchone()[0],0)

    def test_mismatched_source_observation_fails_without_feedback(self):
        obs=replace(self.observation,extra={**self.observation.extra,'evidence_id':'wrong_observation'})
        with self.assertRaisesRegex(ValueError,'actual source observation'):
            self.policy(guard=False)._handle_motion('move_eef',{'targets':{'left_z':.97},'preview':True},obs)
        self.assertEqual(self.store.db.execute("SELECT COUNT(*) FROM evidence WHERE kind='planning_feedback'").fetchone()[0],0)

    def manager(self):
        m=self.session.manager;m.run('Prepare','prepare');m.purpose='review'
        return m

    def test_motion_navigation_pages_feedback_but_only_exact_read_grants_evidence(self):
        refs=[self.feedback(preview=True)[0]['evidence_id'] for _ in range(8)]
        m=self.manager();before=set(m.allowed);cursor=self.after['sequence'];found=[]
        while cursor is not None:
            page=episode_index.directory(self.store,self.session.episode,cursor,m.frontier[self.session.episode],
                'motion',budget_bytes=1100)
            self.assertLessEqual(len(json.dumps(page,ensure_ascii=False).encode()),1100)
            m._admit(page);self.assertEqual(m.allowed,before)
            found.extend(e['id'] for e in page['entries'] if e['kind']=='planning_feedback')
            self.assertNotIn('cumulative_joint_travel_deg',json.dumps(page))
            cursor=page['next_after']
        self.assertEqual(found,refs)
        result,_=m._evidence({'ids':[refs[-1]],'cameras':[],'body':'none'})
        self.assertEqual(m.allowed,before);m._admit(result)
        self.assertEqual(m.allowed,before|{refs[-1]})
        self.assertEqual(result['records'][0]['data']['executed_actions'],0)
        self.assertFalse(m.seen_images);self.assertFalse(m.seen_telemetry)

    def test_feedback_exact_read_still_rejects_prepare_and_future_frontier(self):
        ref=self.feedback(preview=True)[0]['evidence_id']
        m=self.session.manager;m.run('Prepare','prepare')
        with self.assertRaisesRegex(ValueError,'Preparation cannot'):m._record(ref)
        m.purpose='review';future=self.feedback(preview=True)[0]['evidence_id']
        with self.assertRaisesRegex(ValueError,'frontier'):m._record(future)

    def body_patch(self,feedback,observation):
        return {'op':'new','kind':'body','module':'geometry','reading_role':'intrinsic_candidate',
            'observed_entity':'robot_part','observable':'configuration','text':'A source-scoped computational plan cost, no physical effect.',
            'conditions':{},'reference_frame':'world','epistemic':'hypothesis',
            'evidence':[feedback,observation]}

    def submit(self,patch):
        return self.store.submit(patch,allowed_sources=self.session.allowed,cutoff=self.session.source_frontier(),
            dependencies=self.deps,episode=self.session.episode)

    def test_body_cost_case_automatically_pins_planner_and_execution(self):
        ref=self.feedback(preview=True)[0]['evidence_id']
        result=self.submit(self.body_patch(ref,self.after['id']))
        self.assertTrue(result['accepted'],result)
        dependencies=json.loads(self.store.db.execute('SELECT dependencies FROM claims WHERE revision=?',(result['revision'],)).fetchone()[0])
        self.assertEqual(dependencies['planner'],self.deps['planner'])
        self.assertEqual(dependencies['execution'],self.deps['execution'])

    def test_body_feedback_requires_matching_before_and_planner_version(self):
        _,data=self.feedback(preview=True)
        for change in ({'before_observation_id':'unread_observation'},
                {'dependencies':{**self.deps,'planner':'other'}},
                {'dependencies':{**self.deps,'execution':'other'}}):
            ref=self.session.evidence('planning_feedback',{**copy.deepcopy(data),**change})
            result=self.submit(self.body_patch(ref['id'],self.after['id']))
            self.assertFalse(result['accepted']);self.assertIn('planning_source_requires',result['reason'])

    def test_feedback_cannot_replace_executed_receipt_for_physical_effect(self):
        ref=self.feedback(preview=True)[0]['evidence_id']
        patch=fixture.PolicyPipelineTests.patch(self)
        patch.pop('receipt_id');patch.pop('post_observation_id');patch['evidence']=[ref,self.after['id']]
        result=self.submit(patch)
        self.assertFalse(result['accepted']);self.assertIn('effect_requires',result['reason'])


if __name__=='__main__':unittest.main()
