"""Exact detail/page repairs remain possible within the original shared allowance."""
import json
from pathlib import Path
import unittest

from kyb import library
from kyb.budget import CallBudget
from manager_fixtures import response
import test_promotion as promotion


class PublicationRepairBudget(unittest.TestCase):
    setUp=promotion.PromotionTests.setUp
    tearDown=promotion.PromotionTests.tearDown
    observation=promotion.PromotionTests.observation
    attach=promotion.PromotionTests.attach
    add_body_case=promotion.PromotionTests.add_body_case
    op=promotion.PromotionTests.op
    write=promotion.PromotionTests.write
    canonical_ref=promotion.PromotionTests.canonical_ref

    def create_knowledge(self,large=False):
        case=self.add_body_case()
        extra={'text':'A'*1700,'unknowns':['Unresolved '+str(i)+'x'*600 for i in range(8)]} if large else {}
        with self.store.db:return self.write(self.op([self.canonical_ref(case)],**extra),[case])

    def counted(self, turns, remaining):
        manager=self.attach([('publish_materials',{'expected_version':0,'answer':'Prepared','default_knowledge':[]})])
        manager.run('Prepare','prepare')
        budget=CallBudget(Path(self.temp.name)/'repair_budget.jsonl',10)
        for i in range(10-remaining):budget.reserve('prior_attempt_'+str(i),'manager')
        self.client.budget=budget;self.client.turns=list(turns);original=self.client.complete
        def complete(messages,tools):
            before=budget.view()['remaining'];names={t['function']['name'] for t in tools}
            if before==1:self.assertEqual(names,{'publish_materials'})
            else:self.assertIn('search_experience',names)
            budget.reserve('new_'+str(budget.view()['used']),'manager')
            return original(messages,tools)
        self.client.complete=complete
        return manager,budget

    def test_four_remaining_allows_detail_then_knowledge_publication(self):
        knowledge=self.create_knowledge()
        turns=[('search_experience',{'level':'knowledge','view':'detail','ids':[knowledge['id']]}),
               ('publish_materials',{'expected_version':1,'answer':'Retired an explicitly reviewed test candidate',
                 'groups':[{'knowledge_ops':[{'op':'retire','id':knowledge['id'],'reason':'Test retirement after full read'}]}]})]
        manager,budget=self.counted(turns,4)
        result=manager.run('Read the exact candidate and retire it','review')
        self.assertEqual(result['status'],'PASS');self.assertEqual(budget.view()['remaining'],2)
        self.assertEqual(result['knowledge_updates'][0]['status'],'retired')

    def test_two_remaining_allows_one_exact_read_and_last_publication(self):
        knowledge=self.create_knowledge()
        turns=[('search_experience',{'level':'knowledge','view':'detail','ids':[knowledge['id']]}),
               ('publish_materials',{'expected_version':1,'answer':'Complete current view read',
                 'groups':[{'knowledge_ops':[{'op':'retire','id':knowledge['id'],'reason':'Test retirement'}]}]})]
        manager,budget=self.counted(turns,2)
        result=manager.run('Complete the targeted repair','review')
        self.assertEqual(result['status'],'PASS');self.assertEqual(budget.view()['remaining'],0)

    def test_three_remaining_allows_two_pages_and_last_publication(self):
        knowledge=self.create_knowledge(large=True)
        def continuation(messages):
            last_tool=next(message for message in reversed(messages) if message.get('role')=='tool')
            card=json.loads(last_tool['content'])['records'][0]
            self.assertEqual(card['view'],'detail_page');self.assertTrue(card['continuation'])
            return response('search_experience',{'continuation':card['continuation']})
        turns=[('search_experience',{'level':'knowledge','view':'detail','ids':[knowledge['id']]}),continuation,
               ('publish_materials',{'expected_version':1,'answer':'Current detail completed, exact repair committed',
                 'groups':[{'knowledge_ops':[{'op':'retire','id':knowledge['id'],'reason':'Test retirement after pages'}]}]})]
        manager,budget=self.counted(turns,3)
        result=manager.run('Read the exact missing detail pages','review')
        self.assertEqual(result['status'],'PASS');self.assertEqual(budget.view()['remaining'],0)
        self.assertFalse(manager.allowed)

    def test_update_diagnostic_identifies_missing_current_read_without_granting_it(self):
        knowledge=self.create_knowledge();revision=knowledge['revision']
        args=dict(allowed=set(),visible_cases={},visible_knowledge={},dependencies=self.deps,frontier=self.session.source_frontier())
        with self.assertRaisesRegex(ValueError,'current actually read full detail') as error:
            library.publish_knowledge(self.store,{'op':'retire','id':knowledge['id'],'reason':'Not read yet'},**args)
        detail=json.loads(str(error.exception).split(': ',1)[1])
        self.assertEqual(detail['knowledge_id'],knowledge['id']);self.assertEqual(detail['current_revision'],revision)
        self.assertIsNone(detail['actually_read_revision'])
        self.assertEqual(detail['read'],{'level':'knowledge','view':'detail','ids':[knowledge['id']]})
        self.assertEqual(self.store.db.execute('SELECT MAX(revision) FROM reusable_knowledge').fetchone()[0],revision)
        with self.assertRaisesRegex(ValueError,'Unknown knowledge ID'):
            library.publish_knowledge(self.store,{'op':'retire','id':'invented','reason':'Bad ID'},**args)
