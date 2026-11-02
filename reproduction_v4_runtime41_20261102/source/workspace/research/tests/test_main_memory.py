"""Offline witness persistence and online cross-episode inheritance."""
import hashlib
import json
import unittest
from pathlib import Path
from kyb import library
from kyb.instances import dated_instance
from kyb.memory import MemoryStore
from kyb.session import MemorySession
from kyb.trace import RunTrace
import test_physical_memory as physical
import test_manager as manager_fixture
publish=manager_fixture.publish
from manager_fixtures import ScriptedManager


class WitnessTests(unittest.TestCase):
    setUp=physical.PhysicalMemoryTests.setUp
    tearDown=physical.PhysicalMemoryTests.tearDown
    add=physical.PhysicalMemoryTests.add
    observation=physical.PhysicalMemoryTests.observation
    patch=physical.PhysicalMemoryTests.patch
    submit=physical.PhysicalMemoryTests.submit

    def body(self,**kwargs):
        return self.patch(**{'kind':'body','module':'frame_map','reading_role':'intrinsic_candidate',
            'text':'A measured executed prefix, without a reachability guarantee.',**kwargs})

    def test_partial_explicit_pair_is_persisted_in_sources_patch_and_instance(self):
        receipt=self.add('receipt',{'before_observation':self.before['id'],'dependencies':self.deps,
            'execution':'partial','executed_actions':35,'accepted_actions':43,'planned_actions':43,
            'env_start':0,'env_end':35,'unknown_command':None})
        post=self.observation(.2,after_receipt=receipt['id'])
        sources=self.sources|{receipt['id'],post['id']}
        result=self.submit(self.body(evidence=[self.before['id']],receipt_id=receipt['id'],post_observation_id=post['id']),sources=sources)
        self.assertTrue(result['accepted'],result)
        expected={self.before['id'],receipt['id'],post['id']}
        self.assertEqual(set(self.store.evidence_ids(result['revision'])),expected)
        patch=json.loads(self.store.db.execute('SELECT patch FROM decisions WHERE accepted=1').fetchone()[0])
        self.assertEqual(set(patch['evidence']),expected)
        instance=dated_instance(self.store.db,result['revision'])
        self.assertEqual(instance['receipt_id'],receipt['id'])
        self.assertIn('not_a_stable_response',instance['relation_type'])
        transition=instance['execution_receipt']
        self.assertEqual(transition['executed_actions'],35)
        self.assertEqual(transition['planned_actions'],43)

    def test_unread_before_or_post_and_wrong_pair_are_rejected(self):
        for omitted in (self.before['id'],self.post['id']):
            r=self.submit(self.body(evidence=[self.receipt['id']]),sources=self.sources-{omitted})
            self.assertFalse(r['accepted']);self.assertEqual(r['reason'],'source_not_exposed_to_policy')
        unrelated=self.observation(.2)
        result=self.submit(self.body(post_observation_id=unrelated['id']),sources=self.sources|{unrelated['id']})
        self.assertFalse(result['accepted'])
        result=self.store.submit(self.body(),allowed_sources=self.sources,cutoff={self.trace.root.name:self.receipt['sequence']},dependencies=self.deps,episode=self.trace.root.name)
        self.assertFalse(result['accepted'])

    def test_unknown_execution_and_batch_second_error_do_not_publish(self):
        receipt=self.add('receipt',{'before_observation':self.before['id'],'execution':'outcome_unknown','executed_actions':1})
        post=self.observation(.2,after_receipt=receipt['id'])
        result=self.submit(self.body(receipt_id=receipt['id'],post_observation_id=post['id']),sources=self.sources|{receipt['id'],post['id']})
        self.assertFalse(result['accepted'])
        decisions,committed=self.store.submit_batch([self.body(),self.body(text='invalid',post_observation_id=self.before['id'])],
            allowed_sources=self.sources,cutoff={self.trace.root.name:self.trace.sequence},dependencies=self.deps,episode=self.trace.root.name)
        self.assertFalse(committed);self.assertEqual(self.store.revision,0)

    def test_pure_config_requires_no_motion_witness(self):
        result=self.submit(self.body(evidence=[self.before['id']],receipt_id=None,post_observation_id=None))
        self.assertTrue(result['accepted'],result)
        self.assertIsNone(dated_instance(self.store.db,result['revision']))


class ManagerWitnessTests(unittest.TestCase):
    setUp=manager_fixture.ManagerTests.setUp
    tearDown=manager_fixture.ManagerTests.tearDown
    attach=manager_fixture.ManagerTests.attach
    observation=manager_fixture.ManagerTests.observation

    def test_normalized_post_requires_seen_rgb_before_publish(self):
        m=self.attach([publish(0)]);m.run('prepare','prepare');m.purpose='learn'
        m.allowed={self.before['id'],self.receipt['id'],self.post['id']};m.seen_images={self.before['id']}
        p={'op':'new','local':'c1','kind':'body','module':'frame_map','reading_role':'intrinsic_candidate',
            'text':'A source-bound robot motion.', 'conditions':{},'reference_frame':'world',
            'epistemic':'visually_inferred','evidence':[self.before['id']],
            'receipt_id':self.receipt['id'],'post_observation_id':self.post['id']}
        with self.assertRaisesRegex(ValueError,'RGB'):
            m._publish({'expected_version':1,'answer':'x','memory_patch':[p],
                'dispositions':[{'case':'c1','action':'keep','reason':'Single observed instance.'}]})
        self.assertEqual(self.store.revision,0)

    def test_exhausted_manager_reports_without_another_model_call(self):
        from kyb.budget import CallBudget
        m=self.attach([publish(0)])
        budget=CallBudget(Path(self.temp.name)/'manager_budget.jsonl',1)
        m.client.budget=budget;budget.reserve('spent','manager')
        result=m.run('prepare','prepare')
        self.assertEqual(result['reason'],'manager_call_allowance_exhausted')
        self.assertFalse(m.prepared);self.assertFalse(m.client.inputs)
        result=m.run('compare uncertainty','review')
        self.assertEqual(result['reason'],'manager_call_allowance_exhausted')
        self.assertEqual(budget.view()['used'],1)
        result=json.loads(self.session.manage_memory({'purpose':'review','question':'A remaining uncertainty.'}))
        self.assertEqual(result['status'],'NEEDS_WORK');self.assertEqual(result['knowledge_updates'],[])
        self.assertIsNone(result['publication'])

    def test_unpublished_preparation_cannot_start_primary_or_repeat(self):
        from kyb.budget import CallBudget
        m=self.attach([]);m.client.budget=CallBudget(Path(self.temp.name)/'manager_budget.jsonl',1)
        m.client.budget.reserve('last-rejected-prepare','manager')
        with self.assertRaisesRegex(RuntimeError,'no primary request'):
            self.session.refresh_context([{'role':'user','content':'Original goal'}],None)
        self.assertFalse(m.prepared);self.assertFalse(m.client.inputs)

    def test_receipt_does_not_derive_from_unread_before(self):
        m=self.attach([publish(0)]);m.run('prepare','prepare');m.purpose='review'
        record=m._record(self.receipt['id'])
        self.assertNotIn('measured_body_change',record['data'])

    def test_empty_seed_then_terminal_case_knowledge_then_online_next_episode(self):
        seed=Path(self.temp.name)/'seed.sqlite'
        empty=MemoryStore(Path(self.temp.name)/'empty.sqlite');library.initialize(empty)
        empty.freeze(seed);empty.close()
        ep1,manifest=MemoryStore.fork_snapshot(seed,Path(self.temp.name)/'ep1.sqlite')
        self.assertEqual(ep1.revision,0)
        self.assertEqual(ep1.db.execute('SELECT COUNT(*) FROM evidence').fetchone()[0],0)
        t1=RunTrace(Path(self.temp.name)/'ep1',{})
        s1=MemorySession(t1,ep1,task='one',dependencies=self.deps)
        obs=s1.evidence('observation',{'state':{},'dependencies':self.deps})
        s1.attach_manager(ScriptedManager([publish(0)]));s1.manager.run('prepare','prepare')
        m=s1.manager;m.purpose='learn';m.frontier=s1.source_frontier();m.allowed={obs['id']}
        p={'op':'new','local':'c1','kind':'body','module':'geometry','reading_role':'intrinsic_candidate',
            'text':'The visible robot geometry is a conditional candidate.','conditions':{},'reference_frame':'world',
            'epistemic':'hypothesis','evidence':[obs['id']]}
        k={'op':'add','local':'k1','domain':'body','family':'geometry','question':'Which robot geometry needs checking?',
            'text':'Check this conditional geometry candidate before use.','conditions':{},'strength':'hypothesis',
            'use':'Check geometry before choosing motion.','refs':[{'type':'case','local':'c1','relation':'support','reason':'Supplies the limited candidate.'}]}
        m._publish({'expected_version':1,'answer':'Conditional geometry.','memory_patch':[p],'knowledge_ops':[k]})
        handoff=Path(self.temp.name)/'after_ep1.sqlite';ep1.freeze(handoff);ep1.close()
        before=hashlib.sha256(handoff.read_bytes()).hexdigest()
        ep2,_=MemoryStore.fork_snapshot(handoff,Path(self.temp.name)/'ep2.sqlite')
        try:
            s2=MemorySession(RunTrace(Path(self.temp.name)/'ep2',{}),ep2,task='two',dependencies=self.deps)
            self.assertIsNone(s2.frozen_revision);self.assertIsNone(s2.frozen_knowledge_revision)
            s2.attach_manager(ScriptedManager([publish(0)]));s2.manager.run('prepare','prepare')
            initial=json.loads(s2.manager.client.inputs[0][1]['content'])
            self.assertEqual(initial['navigation']['case_counts'][0]['cases'],1)
            self.assertEqual(initial['navigation']['case_counts'][0]['knowledge_items'],1)
            frontier=s2.source_frontier();at=ep2.revision
            obs2=s2.evidence('observation',{'state':{},'dependencies':self.deps})
            case=ep2.submit({key:value for key,value in {**p,'text':'New in this same episode','evidence':[obs2['id']]}.items() if key!='local'},
                allowed_sources={obs2['id']},cutoff=s2.source_frontier(),dependencies=self.deps,episode=s2.episode)
            self.assertTrue(case['accepted'],case)
            self.assertEqual(ep2.revision,2)
            with self.assertRaisesRegex(ValueError,'frontier'):s2.manager._record(obs2['id'])
            self.assertEqual(ep2.read('same episode',context={},dependencies=self.deps,episode=s2.episode)['selected'][0]['revision'],2)
            self.assertFalse(ep2.read('same episode',context={},dependencies=self.deps,episode=s2.episode,at=at,evidence_cutoff=frontier)['selected'])
        finally:ep2.close()
        self.assertEqual(hashlib.sha256(handoff.read_bytes()).hexdigest(),before)
