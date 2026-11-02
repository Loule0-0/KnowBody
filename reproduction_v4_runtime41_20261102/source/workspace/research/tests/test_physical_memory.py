"""Counterexamples to identity conflation, source leakage and stale geometry."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from kyb.memory import MemoryStore
from kyb.session import MemorySession
from kyb.trace import RunTrace
from kyb.qualification import binding_snapshot, compare_conditions, qualify
from kyb.robodojo_adapter import canonical_state
from kyb.state import evidence_state, compare_states, tracking_residual
from kyb.context import surface_readout
from kyb.knowledge import conditional_card


class PhysicalMemoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.store=MemoryStore(self.root/'memory.sqlite');self.trace=RunTrace(self.root/'episode_A',{})
        self.deps={'body':'body1','cameras':'K1','part:a':'part1','part:b':'part1','attachment:a':'episode_A:a_unknown'}
        self.before=self.observation(0.)
        self.receipt=self.add('receipt',{'before_observation':self.before['id'],'dependencies':self.deps,
            'executed_actions':1,'accepted_actions':1,'planned_actions':1,'execution':'complete','unknown_command':None,
            'plan':{'trace':{'arms':{'a':{'named':{'a_x':.2},'target_pose':[.0499,0.,.9,1.,0.,0.,0.]}}}}})
        self.post=self.observation(.1,after_receipt=self.receipt['id'])
        self.sources={self.before['id'],self.receipt['id'],self.post['id']}

    def tearDown(self):self.store.close();self.tmp.cleanup()

    def add(self,kind,data):
        ref=self.trace.event(kind,data);self.store.register(ref,data);return ref

    def observation(self,x,**extra):
        state={'a_ee_pose':[x,0.,.9,1.,0.,0.,0.],'a_arm_joint_state':[0.,0.],'a_ee_joint_state':[0.]}
        return self.add('observation',{'state':state,'dependencies':self.deps,
            'canonical_state':canonical_state(state,{},'body1'),**extra})

    def patch(self,**extra):
        return {'op':'new','reading_role':'interaction','kind':'external','relation_type':'observed_interaction',
            'module':'interaction','text':'A lateral push visibly moved an object.','conditions':{'support':'table','direction':'+x'},
            'reference_frame':'world','epistemic':'visually_inferred','evidence':sorted(self.sources),
            'receipt_id':self.receipt['id'],'post_observation_id':self.post['id'],**extra}

    def submit(self,p,**extra):
        return self.store.submit(p,allowed_sources=extra.get('sources',self.sources),cutoff={self.trace.root.name:self.trace.sequence},
            dependencies=self.deps,episode=self.trace.root.name,visible_revisions=extra.get('visible',{}))

    def read(self,**extra):
        return self.store.read('lateral push',context=extra.pop('context',{}),dependencies=extra.pop('dependencies',self.deps),
            episode=extra.pop('episode',self.trace.root.name),**extra)

    def test_unknown_and_declared_are_distinct_from_different(self):
        self.assertEqual(compare_conditions({'task':'one','phase':'x','support':'unknown','mass':'heavy','direction':'+x'},
            {'task':'two','phase':'y','support':'table','mass':'heavy','direction':'-x'}),
            {'support':'unknown','mass':'match_declared','direction':'unknown'})

    def test_different_free_descriptions_are_not_known_physical_differences(self):
        q=qualify(role='interaction',conditions={'support':'tabletop'},context={'support':'table'},
            pins={},dependencies={},source_episode='old',episode='new',source_bindings={})
        self.assertEqual(q['category'],'conditional_candidate')
        self.assertEqual(q['physical_conditions']['support'],'unknown')
        self.assertEqual(q['different_descriptions']['support'],{'source':'tabletop','current':'table'})

    def test_task_label_change_does_not_change_physical_candidate(self):
        self.assertTrue(self.submit(self.patch(conditions={'task':'one','phase':'old','support':'table'}))['accepted'])
        a=self.read(context={'task':'one','phase':'old','support':'table'})
        b=self.read(context={'task':'two','phase':'new','support':'table'})
        self.assertEqual(a['selected'],b['selected'])
        unmatched=self.store.read('unmatched_label',context={'task':'one'},dependencies=self.deps,episode='episode_A')
        self.assertFalse(unmatched['selected'])

    def test_run_local_unknown_attachment_does_not_reject_cross_episode_response(self):
        self.assertTrue(self.submit(self.patch(depends_on=['attachment:a']))['accepted'])
        item=self.read(episode='new_episode',dependencies={**self.deps,'attachment:a':'other:a_unknown'})['selected'][0]
        self.assertEqual(item['qualification']['category'],'conditional_candidate')
        self.assertEqual(item['qualification']['body_and_binding_comparison']['attachment:a'],'unknown')

    def test_unrelated_part_update_does_not_invalidate_local_relation(self):
        self.assertTrue(self.submit(self.patch(depends_on=['part:a']))['accepted'])
        item=self.read(dependencies={**self.deps,'part:b':'part2'})['selected'][0]
        self.assertEqual(item['qualification']['body_and_binding_comparison']['part:a'],'match')
        self.assertNotIn('part:b',item['qualification']['body_and_binding_comparison'])
        changed=self.read(dependencies={**self.deps,'part:a':'part2'})['selected'][0]
        self.assertEqual(changed['qualification']['category'],'analogy')
        self.assertEqual(changed['qualification']['geometry_equivalence'],'unknown_not_proved_by_binding_identity')

    def test_support_delta_revise_reset_and_negative_survival(self):
        first=self.submit(self.patch());self.assertTrue(first['accepted'])
        new=self.observation(.1,after_receipt=self.receipt['id']);all_sources=self.sources|{new['id']}
        supported=self.submit(self.patch(op='support',claim_id=first['claim_id'],claim_revision=first['revision'],
            evidence=sorted(all_sources),post_observation_id=new['id']),sources=all_sources,visible={first['claim_id']:first['revision']})
        self.assertTrue(supported['accepted'],supported)
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM claim_evidence WHERE revision=?',(supported['revision'],)).fetchone()[0],1)
        negative=self.submit(self.patch(op='counterexample',claim_id=first['claim_id'],claim_revision=supported['revision'],
            text='Contrary observed lateral push',evidence=sorted(all_sources),post_observation_id=new['id']),
            sources=all_sources,visible={first['claim_id']:supported['revision']})
        self.assertTrue(negative['accepted'],negative)
        revised=self.submit(self.patch(op='revise',claim_id=first['claim_id'],claim_revision=negative['revision'],text='Narrowed lateral push rule'),
            visible={first['claim_id']:negative['revision']})
        self.assertTrue(revised['accepted'],revised)
        self.assertNotIn(new['id'],self.store.evidence_ids(revised['revision']))
        self.assertIn(new['id'],self.store.provenance_at(revised['revision'])['evidence'])
        self.assertEqual(len(self.store.provenance_at(revised['revision'])['counterexamples']),1)

    def test_source_frontier_retrieves_older_version_and_rebuilds_ranking(self):
        first=self.submit(self.patch());old_cutoff=self.trace.sequence
        new=self.observation(.1,after_receipt=self.receipt['id']);sources=self.sources|{new['id']}
        second=self.submit(self.patch(op='support',claim_id=first['claim_id'],claim_revision=first['revision'],evidence=sorted(sources),
            post_observation_id=new['id']),sources=sources,visible={first['claim_id']:first['revision']})
        a=self.read(at=second['revision'],evidence_cutoff={'episode_A':old_cutoff})
        self.assertEqual(a['selected'][0]['revision'],first['revision'])
        b=self.read(at=second['revision'],evidence_cutoff={'episode_A':self.trace.sequence})
        self.assertEqual(b['selected'][0]['revision'],second['revision'])
        self.assertEqual(a,self.read(at=second['revision'],evidence_cutoff={'episode_A':old_cutoff}))

    def test_unknown_identity_hides_tool_geometry_but_not_physical_analogy(self):
        q=dict(conditions={},pins={},context={},dependencies={},source_episode='old',episode='new',source_bindings={})
        self.assertEqual(qualify(role='binding_geometry',**q)['category'],'identity_unresolved')
        self.assertEqual(qualify(role='interaction',**q)['category'],'conditional_candidate')
        self.assertFalse(qualify(role='intrinsic_candidate',**q)['current_fact'])

    def test_opening_invalidates_declared_binding_without_proving_release(self):
        source=self.before['id']
        change=self.add('attachment_change',{'arm':'a','old':{'state':'unknown'},'new':{'state':'visually_supported','tool':'object'},
            'observation_id':source,'binding_changed':True,'evidence':[source]})
        before=self.observation(.1)
        opened=self.add('receipt',{'before_observation':before['id'],'executed_actions':1,
            'plan':{'trace':{'arms':{'a':{'named':{'a_gripper':.6}}}}}})
        after=self.observation(.1,after_receipt=opened['id'])
        old=binding_snapshot(self.store.db,before['id'],{'episode_A':self.trace.sequence})['a']
        new=binding_snapshot(self.store.db,after['id'],{'episode_A':self.trace.sequence})['a']
        self.assertEqual(old['qualification'],'declared');self.assertEqual(old['binding_id'],change['id'])
        self.assertEqual(new['qualification'],'unknown');self.assertEqual(new['invalidated_by'],opened['id'])
        self.assertNotIn('released',new)

    def test_missing_contract_and_changed_live_constant_do_not_reinterpret_history(self):
        raw={'state':{'a_ee_pose':[0.,0.,.9,1.,0.,0.,0.]}}
        self.assertFalse(evidence_state(raw)['grasp_point_dimensions'])
        recorded=self.store.evidence_payload(self.before['id']);expected=evidence_state(recorded)
        from XPolicyLab.policy.GPT_6_Astra_Direct_EEF import pose
        saved=pose.GRASP_POINT_OFFSET_M
        try:
            pose.GRASP_POINT_OFFSET_M=99.
            self.assertEqual(evidence_state(recorded),expected)
        finally:pose.GRASP_POINT_OFFSET_M=saved
        changed=copy.deepcopy(recorded);changed['canonical_state']['coordinate_contract']['id']='new_contract'
        self.assertFalse(compare_states(recorded,changed)['measured_differences'])

    def test_completed_execution_does_not_certify_tracking(self):
        residual=tracking_residual(self.store.evidence_payload(self.receipt['id']),self.store.evidence_payload(self.post['id']))
        self.assertAlmostEqual(residual['measured_differences']['a']['grasp_position_current_minus_source_m'][0],.0501)
        self.assertEqual(residual['tracking_verdict'],'not_certified_by_command_execution')

    def test_hashed_source_rejects_forged_payload(self):
        ref=self.trace.event('observation',{'state':{}})
        with self.assertRaisesRegex(ValueError,'differs'):
            self.store.register(ref,{'state':{'forged':True}})

    def test_conditional_answer_preserves_negative_conditions_and_unknowns(self):
        first=self.submit(self.patch());negative=self.submit(self.patch(op='counterexample',claim_id=first['claim_id'],
            claim_revision=first['revision'],text='Observed contrary lateral push'),visible={first['claim_id']:first['revision']})
        self.assertTrue(negative['accepted'])
        item=self.read(context={})['selected'][0]
        from kyb.library import organize_archive
        organize_archive(self.store)
        card=conditional_card(item,self.store,self.deps)
        self.assertEqual(card['counterexamples'][0]['text'],'Observed contrary lateral push')
        self.assertEqual(card['counterexamples'][0]['conditions'],self.patch()['conditions'])
        self.assertEqual(set(card['applicability']['unknown_conditions']),{'support','direction'})
        self.assertNotIn('objects',card)

    def test_surface_view_expands_only_selected_arm_and_keeps_full_archive(self):
        result={'samples':[{'camera':'view','rgb_sha256':'rgb','depth_sha256':'depth','qualification':'qualified',
            'body_coordinates':{'a':{'relative_to_named_grasp_in_flange_axes_m':[1,0,0]},'b':{'relative_to_named_grasp_in_flange_axes_m':[2,0,0]}},'semantic_assignment':'declared'}]}
        original=copy.deepcopy(result)
        world=surface_readout(result,None);local=surface_readout(result,'a')
        self.assertNotIn('body_coordinates',world['samples'][0])
        self.assertEqual(local['samples'][0]['acting_frame']['arm'],'a')
        self.assertEqual(local['samples'][0]['acting_frame']['relative_to_grasp_flange_axes_m'],[1,0,0])
        self.assertEqual(result,original)

    def test_reset_revision_does_not_leak_future_old_text_into_fts(self):
        later=self.observation(.2)
        body=self.patch(kind='body',reading_role='intrinsic_candidate',text='futureonlysentinel',evidence=[later['id']],receipt_id=None,post_observation_id=None)
        first=self.submit(body,sources={later['id']});self.assertTrue(first['accepted'])
        revised=self.submit({**body,'op':'revise','claim_id':first['claim_id'],'claim_revision':first['revision'],
            'text':'Available safe observation','evidence':[self.before['id']]},sources={self.before['id']},
            visible={first['claim_id']:first['revision']})
        self.assertTrue(revised['accepted'])
        def read(sequence):
            return self.store.read('futureonlysentinel',context={},dependencies=self.deps,episode='episode_A',
                at=revised['revision'],evidence_cutoff={'episode_A':sequence})
        self.assertFalse(read(self.before['sequence'])['selected'])
        self.assertTrue(read(later['sequence'])['selected'])
        self.assertFalse(read(self.before['sequence'])['selected'])

    def test_future_annotation_cannot_change_same_prefix_read(self):
        body=self.patch(kind='body',reading_role='binding_geometry',depends_on=['attachment:a'])
        self.assertTrue(self.submit(body)['accepted']);cutoff={'episode_A':self.trace.sequence}
        before=self.read(evidence_cutoff=cutoff)
        annotation=self.add('attachment_change',{'arm':'a','observation_id':self.post['id'],'new':{'state':'visually_supported'},
            'binding_changed':True,'evidence':[self.post['id']]})
        self.store.annotate_observation(self.post['id'],{**self.deps,'attachment:a':annotation['id']})
        self.assertEqual(before,self.read(evidence_cutoff=cutoff))

    def test_late_old_annotation_cannot_resurrect_hold_after_opening(self):
        self.add('attachment_change',{'arm':'a','observation_id':self.before['id'],'new':{'state':'visually_supported'},
            'binding_changed':True,'evidence':[self.before['id']]})
        opened=self.add('receipt',{'before_observation':self.post['id'],'executed_actions':1,
            'plan':{'trace':{'arms':{'a':{'named':{'a_gripper':.6}}}}}})
        after=self.observation(.1,after_receipt=opened['id'])
        self.add('attachment_observation',{'arm':'a','observation_id':self.before['id'],
            'new':{'state':'visually_supported'},'binding_changed':False,'evidence':[self.before['id']]})
        binding=binding_snapshot(self.store.db,after['id'],{'episode_A':self.trace.sequence})['a']
        self.assertEqual(binding['qualification'],'unknown')
        self.assertEqual(binding['invalidated_by'],opened['id'])
