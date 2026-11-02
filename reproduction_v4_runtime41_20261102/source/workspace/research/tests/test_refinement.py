"""Observable regressions on budgeted read permissions, independent writes and timing."""
import copy
import json
import unittest
from dataclasses import replace
import numpy as np
import test_promotion as promotion
import test_policy_pipeline as pipeline
from kyb import library
from kyb.knowledge import read_experience
from kyb.policy import EefAgentPolicy


class ReadoutAndGroups(unittest.TestCase):
    setUp=promotion.PromotionTests.setUp
    tearDown=promotion.PromotionTests.tearDown
    observation=promotion.PromotionTests.observation
    attach=promotion.PromotionTests.attach
    add_body_case=promotion.PromotionTests.add_body_case
    manager=promotion.PromotionTests.manager
    op=promotion.PromotionTests.op
    case_patch=promotion.PromotionTests.case_patch
    canonical_ref=promotion.PromotionTests.canonical_ref
    write=promotion.PromotionTests.write
    read_case=promotion.PromotionTests.read_case

    def test_preview_and_pages_do_not_grant_unseen_interpretation(self):
        case=self.add_body_case()
        op=self.op([self.canonical_ref(case)],text='A'*1700,unknowns=['Material unknown '+str(i)*100 for i in range(8)])
        with self.store.db:k=self.write(op,[case])
        m=self.manager()
        answer,_=read_experience(self.session,'relation',ids=[k['id']],budget_bytes=1100)
        self.assertEqual(answer['status'],'partial')
        card=answer['records'][0];self.assertEqual(card['view'],'detail_page')
        m._admit(answer);self.assertNotIn(k['id'],m.visible_knowledge)
        token=card['continuation'];seen=0
        while token:
            result,_=read_experience(self.session,'relation',continuation=token,budget_bytes=1100)
            self.assertLessEqual(len(json.dumps(result,ensure_ascii=False).encode()),1100)
            m._admit(result);seen+=1
            token=result['records'][0]['continuation']
            if token:self.assertNotIn(k['id'],m.visible_knowledge)
        self.assertGreater(seen,1);self.assertEqual(m.visible_knowledge[k['id']],k['revision'])
        self.assertFalse(m.allowed);self.assertFalse(m.visible_revisions)

    def test_statement_preserves_conditions_and_does_not_grant_write(self):
        c=self.add_body_case()
        with self.store.db:k=self.write(self.op([self.canonical_ref(c)],conditions={'arm':'left','support':'unknown'}),[c])
        m=self.manager()
        answer,_=read_experience(self.session,'relation',ids=[k['id']],view='statement')
        m._admit(answer)
        self.assertEqual(answer['records'][0]['conditions'],{'arm':'left','support':'unknown'})
        self.assertNotIn(k['id'],m.visible_knowledge)
        self.assertEqual(m.selectable_knowledge[k['id']],k['revision'])

    def test_independent_group_success_survives_invalid_knowledge_and_retry_is_idempotent(self):
        m=self.manager();m.allowed={self.post['id']}
        patch=self.case_patch()
        good={'memory_patch':[patch],'dispositions':[{'case':'c1','action':'keep','reason':'One observation only'}]}
        bad={'knowledge_ops':[self.op([{'type':'knowledge','id':'unread'}])]}
        revision=m.revision;knowledge_revision=m.knowledge_revision
        result,done=m._publish_groups({'expected_version':m.version,'answer':'Keep the scoped case','groups':[good,bad]})
        self.assertFalse(done);self.assertEqual(result['status'],'NEEDS_WORK');self.assertEqual(len(result['memory_updates']),1)
        saved=self.store.revision
        result,done=m._publish_groups({'expected_version':m.version,'answer':'No new facts','groups':[good]})
        self.assertTrue(done);self.assertEqual(self.store.revision,saved)
        self.assertTrue(result['committed_groups'][0]['already_committed'])
        self.assertEqual((m.revision,m.knowledge_revision),(revision,knowledge_revision))

    def test_case_and_invalid_dependent_knowledge_roll_back_together(self):
        m=self.manager();m.allowed={self.post['id']};before=self.store.revision
        result,done=m._publish_groups({'expected_version':m.version,'answer':'Unresolved','groups':[{
            'memory_patch':[self.case_patch()], 'knowledge_ops':[self.op([{'type':'case','local':'c1','relation':'support','reason':'Observed'}],strength='invalid')]}]})
        self.assertFalse(done);self.assertEqual(self.store.revision,before)
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM publication_groups').fetchone()[0],0)

    def test_invalid_default_retains_legal_group_and_target_retry_needs_only_missing(self):
        c=self.add_body_case();m=self.manager();self.read_case(m,c);m.targets={c['claim_id']:c['revision']}
        group={'dispositions':[{'case':c['claim_id'],'action':'keep','reason':'Scoped instance only'}]}
        result,done=m._publish_groups({'expected_version':m.version,'answer':'Keep it','groups':[group],'default_knowledge':['unread']})
        self.assertFalse(done);self.assertEqual(m.targets,{})
        count=self.store.db.execute('SELECT COUNT(*) FROM case_disposition').fetchone()[0]
        result,done=m._publish_groups({'expected_version':m.version,'answer':'Keep defaults','groups':[]})
        self.assertTrue(done);self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM case_disposition').fetchone()[0],count)

    def test_cross_group_local_dependency_is_rejected(self):
        m=self.manager();m.allowed={self.post['id']}
        result,done=m._publish_groups({'expected_version':m.version,'answer':'Must merge dependent groups','groups':[
            {'knowledge_ops':[self.op([{'type':'case','local':'c1','relation':'support','reason':'Observed'}])]}]})
        self.assertFalse(done);self.assertEqual(result['issues'][0]['code'],'cross_group_dependency')

    def test_two_groups_cannot_dispose_or_revise_the_same_case(self):
        c=self.add_body_case();m=self.manager();self.read_case(m,c)
        group={'dispositions':[{'case':c['claim_id'],'action':'keep','reason':'Scoped instance'}]}
        result,done=m._publish_groups({'expected_version':m.version,'answer':'Merge overlapping writes','groups':[group,copy.deepcopy(group)]})
        self.assertFalse(done)
        self.assertTrue(all(i['code']=='cross_group_write_dependency' for i in result['issues']))
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM case_disposition').fetchone()[0],0)

    def test_independent_knowledge_can_share_an_already_read_historical_source(self):
        c=self.add_body_case();m=self.manager();self.read_case(m,c)
        ref={k:v for k,v in self.canonical_ref(c).items() if k!='revision'}
        result,done=m._publish_groups({'expected_version':m.version,'answer':'Two distinct limited propositions','groups':[
            {'knowledge_ops':[self.op([ref])]},
            {'knowledge_ops':[self.op([ref],text='A distinct candidate relation, not a universal gain.')]}]})
        self.assertTrue(done,result);self.assertEqual(len(result['knowledge_updates']),2)
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM case_disposition').fetchone()[0],0)

    def test_group_commit_receipt_survives_new_manager_object(self):
        m=self.manager();m.allowed={self.post['id']}
        good={'memory_patch':[self.case_patch()],'dispositions':[{'case':'c1','action':'keep','reason':'Observed once'}]}
        result,done=m._publish_groups({'expected_version':m.version,'answer':'Keep candidate','groups':[good]})
        before=self.store.revision
        m.group_commits={};m.request_handles={}
        result,done=m._publish_groups({'expected_version':m.version,'answer':'Do not duplicate','groups':[good]})
        self.assertTrue(done);self.assertEqual(self.store.revision,before)
        self.assertTrue(result['committed_groups'][0]['already_committed'])

    def test_explicit_browse_does_not_require_translating_stored_language(self):
        c=self.add_body_case()
        with self.store.db:k=self.write(self.op([self.canonical_ref(c)]),[c])
        search,_=read_experience(self.session,'完全不匹配的中文问题',domain='body',family='geometry')
        # Dense candidates may exist even without lexical overlap; retrieval is
        # not a certificate that this unrelated question has an answer.
        self.assertIn('not verified relevance',search['interpretation'])
        listed,_=read_experience(self.session,mode='browse',domain='body',family='geometry',view='statement')
        self.assertEqual(listed['records'][0]['id'],k['id'])
        self.assertIn('category listing',listed['interpretation'])
        with self.assertRaises(ValueError):read_experience(self.session,mode='browse',domain='body')

    def test_body_telemetry_requires_exact_actual_read_not_index_discovery(self):
        from kyb.robodojo_adapter import canonical_state
        state={'left_ee_pose':[0.,0.,.9,1.,0.,0.,0.],'left_arm_joint_state':[0.]*6,'left_ee_joint_state':[1.]}
        obs=self.session.evidence('observation',{'state':state,'dependencies':self.deps,
            'canonical_state':canonical_state(state,{},'b1')})
        m=self.manager();m.frontier=self.session.source_frontier();m.allowed={obs['id']}
        patch={**self.case_patch(),'epistemic':'telemetry_inferred','observable':'relative_geometry','evidence':[obs['id']]}
        group={'memory_patch':[patch],'dispositions':[{'case':'c1','action':'keep','reason':'Source-time measured endpoint, no current path guarantee'}]}
        result,done=m._publish_groups({'expected_version':m.version,'answer':'Need actual body read','groups':[group]})
        self.assertFalse(done);self.assertEqual(self.store.revision,0)
        result,parts=m._evidence({'ids':[obs['id']],'body':'ee','cameras':[]});m._admit(result)
        self.assertFalse(m.seen_images)
        result,done=m._publish_groups({'expected_version':m.version,'answer':'Store limited telemetry interpretation','groups':[group]})
        self.assertTrue(done,result);self.assertEqual(self.store.revision,1)

    def test_telemetry_cannot_certify_external_interaction(self):
        patch={**self.case_patch(),'kind':'external','module':'contact_response','reading_role':'interaction',
            'relation_type':'observed_interaction','epistemic':'telemetry_inferred','evidence':[self.post['id']]}
        patch.pop('local')
        result=self.store.submit(patch,allowed_sources={self.post['id']},cutoff=self.session.source_frontier(),
            dependencies=self.deps,episode=self.session.episode)
        self.assertFalse(result['accepted']);self.assertIn('telemetry_inferred_only',result['reason'])

    def test_successful_primary_lookup_is_a_lead_not_physical_support(self):
        from kyb import leads
        c=self.add_body_case()
        with self.store.db:k=self.write(self.op([self.canonical_ref(c)]),[c])
        answer=json.loads(self.session.query_experience({'question':'relation','ids':[k['id']]}))
        self.assertTrue(answer['records'])
        queued=leads.pending(self.store,self.session.episode,self.session.source_frontier())
        self.assertEqual(len(queued['records']),1)
        self.assertEqual(queued['records'][0]['candidate_refs'][0]['id'],k['id'])
        self.assertIn('no adoption',queued['records'][0]['qualification'])
        self.assertEqual(self.store.revision,c['revision'])


class Timing(unittest.TestCase):
    setUp=pipeline.PolicyPipelineTests.setUp
    tearDown=pipeline.PolicyPipelineTests.tearDown

    def policy(self):
        def planner(**kw):
            return {'status':'Success','position':np.array([np.ones(6)*.2,np.ones(6)*.4])}
        return EefAgentPolicy(action_spec=self.spec,env={},client=pipeline.Client([]),planner=planner,research=self.session)

    def test_prefix_keeps_old_grip_until_actual_closure_phase(self):
        p=self.policy();p._gripper_goals={'left':1.,'right':1.}
        result=p._handle_motion('move_eef',{'targets':{'left_z':.97,'left_gripper':0},'execute_max_s':.04},self.observation)
        self.assertIsNotNone(result.chunk)
        self.assertEqual(len(result.chunk.actions),1)
        self.assertGreater(result.chunk.meta['full_planned_actions'],1)
        self.assertNotIn('chunk_final',result.chunk.actions[-1].meta)
        self.assertEqual(p._gripper_goals['left'],1.)
        p.confirm_executed(1);self.assertEqual(p._gripper_goals['left'],1.)

    def test_preview_and_cost_rejection_do_not_latch_future_jaw(self):
        for args in ({'preview':True},{'max_joint_travel_deg':.1}):
            p=self.policy();p._gripper_goals={'left':1.,'right':1.}
            result=p._handle_motion('move_eef',{'targets':{'left_z':.97,'left_gripper':0},**args},self.observation)
            self.assertIsNone(result.chunk);self.assertEqual(p._gripper_goals['left'],1.)
            self.assertIn('executed_actions',result.tool_result)

    def test_hold_advances_requested_ticks_and_retains_last_goal_not_measured_gap(self):
        p=self.policy();p._gripper_goals={'left':0.,'right':1.}
        obs=replace(self.observation,state={**self.observation.state,'left_ee_joint_state':np.array([.3])})
        result=p._handle_motion('hold',{'seconds':1.},obs)
        self.assertEqual(len(result.chunk.actions),25)
        self.assertTrue(all(float(a.data['left_ee_joint_state'][0])==0. for a in result.chunk.actions))
        p.confirm_executed(12);self.assertEqual(p._gripper_goals['left'],0.)

    def test_hold_repeats_confirmed_arm_setpoint_despite_tracking_residual(self):
        p=self.policy()
        first=p._handle_motion('move_eef',{'targets':{'left_z':.97},'execute_max_s':.04},self.observation)
        last=np.asarray(first.chunk.actions[0].data['left_arm_joint_state']).copy()
        p.confirm_executed(1)
        actual=replace(self.observation,state={**self.observation.state,'left_arm_joint_state':np.full(6,-.5)})
        hold=p._handle_motion('hold',{'seconds':.2},actual)
        for action in hold.chunk.actions:np.testing.assert_array_equal(action.data['left_arm_joint_state'],last)

    def test_hold_respects_remaining_environment_steps_and_body_guard(self):
        p=self.policy();obs=replace(self.observation,remaining_steps=3)
        result=p._handle_motion('hold',{'seconds':1.},obs)
        self.assertEqual(len(result.chunk.actions),3)
        p._path_validator=lambda *args:{'status':'NEEDS_WORK'}
        self.assertIsNone(p._handle_motion('hold',{'seconds':1.},obs).chunk)
