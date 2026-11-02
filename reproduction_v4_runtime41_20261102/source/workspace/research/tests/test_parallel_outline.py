"""Concurrent process summaries preserve independent stage conditions without granting authority."""
import copy
from pathlib import Path
import tempfile
import unittest

from kyb import episodes
from kyb.memory import MemoryStore
from kyb.session import MemorySession
from kyb.trace import RunTrace
from kyb.knowledge import read_experience
from manager_fixtures import PreparationClient


class ParallelOutlineTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();root=Path(self.temp.name)
        self.store=MemoryStore(root/'memory.sqlite');self.trace=RunTrace(root/'run',{})
        self.session=MemorySession(self.trace,self.store,task='stacking',dependencies={})
        self.receipts=[]
        before=self.session.evidence('observation',{'state':{}})
        for i in range(8):
            receipt=self.session.evidence('receipt',{'before_observation':before['id'],
                'execution':'complete','executed_actions':1,'env_start':i,'env_end':i+1})
            self.receipts.append(receipt['id'])
            before=self.session.evidence('observation',{'state':{},'after_receipt':receipt['id']})
        episodes.begin(self.session,{},True)

    def tearDown(self):
        self.store.close();self.temp.cleanup()

    def stage(self,a,b):
        return {'summary':'Independent handling thread','first_receipt':self.receipts[a],
                'last_receipt':self.receipts[b],'conditions':{'grasp':'kept visible'},'unknowns':['Continuous slip unknown']}

    def test_overlapping_and_nested_spans_are_readable_without_extra_actions_or_permissions(self):
        stages=[self.stage(0,6),self.stage(1,2),self.stage(5,7)]
        episodes.seal(self.session,stages)
        record=episodes.read(self.store,ids=['run'])[0]
        self.assertEqual(record['stages'],stages)
        self.assertEqual(record['receipt_count'],8);self.assertEqual(record['executed_actions'],8)
        self.assertIn('may overlap',record['qualification']);self.assertFalse(record['reviewed'])
        self.session.attach_manager(PreparationClient());manager=self.session.manager
        manager.run('prepare','prepare')
        result,_=read_experience(self.session,level='episode',ids=['run'])
        manager._admit(result)
        self.assertFalse(manager.allowed);self.assertFalse(manager.visible_revisions);self.assertFalse(manager.visible_knowledge)
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM claims').fetchone()[0],0)
        changed=copy.deepcopy(stages);changed[1]['summary']='Rewritten after result'
        with self.assertRaisesRegex(ValueError,'sealed'):
            episodes.seal(self.session,changed)

    def test_invalid_sources_backwards_order_and_size_remain_rejected(self):
        for stages in ([self.stage(4,6),self.stage(2,7)],[self.stage(4,2)],
                       [{**self.stage(0,1),'first_receipt':'other:e000001'}],
                       [{**self.stage(0,1),'first_receipt':self.session.current_observation['id']}],
                       [self.stage(0,1)]*7,[{**self.stage(0,1),'summary':'x'*5000}]):
            with self.assertRaises(ValueError):episodes.seal(self.session,stages)
        episodes.attach_result(self.session)
        with self.assertRaisesRegex(ValueError,'already published'):
            episodes.seal(self.session,[self.stage(0,1)])
