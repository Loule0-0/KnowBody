"""Requested decision views must not acquire audit material or write permission."""
import copy
import hashlib
import json
import unittest
import test_promotion as p
from kyb.knowledge import read_experience


class StatementPaging(unittest.TestCase):
    setUp=p.PromotionTests.setUp
    tearDown=p.PromotionTests.tearDown
    observation=p.PromotionTests.observation
    attach=p.PromotionTests.attach
    manager=p.PromotionTests.manager
    add_body_case=p.PromotionTests.add_body_case
    op=p.PromotionTests.op
    write=p.PromotionTests.write
    canonical_ref=p.PromotionTests.canonical_ref

    def candidate(self):
        support=self.add_body_case()
        counter=self.add_body_case(text='A distinct scoped counter observation.')
        with self.store.db:
            return self.write(self.op([self.canonical_ref(support),self.canonical_ref(counter,'counter')],
                text='A retained conditional statement. '*35,
                conditions={'scope':'条件不可删除'*60},
                unknowns=['An unresolved premise '+str(i)+'x'*450 for i in range(6)]),[support,counter])

    def fresh_manager(self):
        manager=self.manager()
        manager.allowed=set();manager.visible_revisions={};manager.visible_knowledge={}
        manager.selectable_knowledge={};manager.read_pages={};manager.read_page_lengths={}
        return manager

    def finish(self,manager,result,expected):
        fragments={};total=None
        while True:
            self.assertLessEqual(len(json.dumps(result,ensure_ascii=False).encode()),1100)
            record=result['records'][0]
            self.assertEqual(record['view'],expected+'_page')
            fragments[record['offset']]=record['fragment'];total=record['total_characters']
            manager._admit(result)
            token=record['continuation']
            if not token:break
            result,_=read_experience(self.session,continuation=token,budget_bytes=1100)
        raw=''.join(fragments[offset] for offset in sorted(fragments))
        self.assertEqual(len(raw),total)
        return json.loads(raw)

    def test_statement_pages_preserve_qualifications_without_full_read_permission(self):
        knowledge=self.candidate();manager=self.fresh_manager()
        result,_=read_experience(self.session,level='knowledge',ids=[knowledge['id']],view='statement',budget_bytes=1100)
        first=result['records'][0];self.assertEqual(first['view'],'statement_page')
        manager._admit(result);self.assertNotIn(knowledge['id'],manager.selectable_knowledge)
        record=self.finish(manager,result,'statement')
        self.assertEqual(record['conditions'],{'scope':'条件不可删除'*60})
        self.assertEqual(len(record['unknowns']),6);self.assertTrue(record['source_contested'])
        self.assertEqual(record['counter_cases'],1);self.assertFalse(record['default_eligible'])
        self.assertTrue(record['statement_complete']);self.assertFalse(record['complete_record'])
        self.assertNotIn('refs',record['drill_down']);self.assertIn('source_refs',record['drill_down'])
        self.assertEqual(manager.selectable_knowledge[knowledge['id']],knowledge['revision'])
        self.assertFalse(manager.visible_knowledge);self.assertFalse(manager.visible_revisions);self.assertFalse(manager.allowed)

    def test_explicit_detail_after_statement_has_separate_pages_and_original_permission(self):
        knowledge=self.candidate();manager=self.fresh_manager()
        result,_=read_experience(self.session,level='knowledge',ids=[knowledge['id']],view='statement',budget_bytes=1100)
        statement=self.finish(manager,result,'statement')
        self.assertNotIn(knowledge['id'],manager.visible_knowledge)
        result,_=read_experience(self.session,continuation=statement['continuation'],budget_bytes=1100)
        detail=self.finish(manager,result,'detail')
        self.assertTrue(detail['complete_record']);self.assertIn('refs',detail['drill_down'])
        self.assertEqual(manager.visible_knowledge[knowledge['id']],knowledge['revision'])
        self.assertEqual({key[3] for key in manager.read_pages},{'statement','detail'})

    def test_case_statement_pages_never_grant_case_write_permission(self):
        case=self.add_body_case(text='A scoped observation. '*70);manager=self.fresh_manager()
        result,_=read_experience(self.session,level='case',case_refs=[{'id':case['claim_id'],'revision':case['revision']}],view='statement',budget_bytes=1100)
        record=self.finish(manager,result,'statement')
        self.assertEqual(record['id'],case['claim_id']);self.assertFalse(manager.visible_revisions)

    def test_conflicting_lengths_and_fragments_are_rejected(self):
        knowledge=self.candidate();manager=self.fresh_manager()
        result,_=read_experience(self.session,ids=[knowledge['id']],view='statement',budget_bytes=1100)
        manager._admit(result)
        bad=copy.deepcopy(result);bad['records'][0]['total_characters']+=1
        with self.assertRaisesRegex(ValueError,'Conflicting immutable page length'):manager._admit(bad)
        bad=copy.deepcopy(result);bad['records'][0]['fragment']='!'+bad['records'][0]['fragment'][1:]
        with self.assertRaisesRegex(ValueError,'Conflicting immutable page fragment'):manager._admit(bad)

    def test_statement_page_cannot_reconstruct_a_full_detail_record(self):
        manager=self.fresh_manager();raw=json.dumps({'id':'test','revision':1,'layer':'knowledge','view':'detail','complete_record':True})
        result={'records':[{'id':'test','revision':1,'layer':'knowledge','view':'statement_page','offset':0,'total_characters':len(raw),'fragment':raw,'continuation':None,'content_sha256':hashlib.sha256(raw.encode()).hexdigest()}]}
        with self.assertRaisesRegex(ValueError,'identity or view mismatch'):manager._admit(result)
        self.assertFalse(manager.visible_knowledge);self.assertFalse(manager.selectable_knowledge)

    def test_repeating_same_statement_does_not_conflict_with_optional_detail_token(self):
        knowledge=self.candidate();manager=self.fresh_manager();records=[]
        for _ in range(2):
            result,_=read_experience(self.session,ids=[knowledge['id']],view='statement',budget_bytes=1100)
            records.append(self.finish(manager,result,'statement'))
        self.assertEqual(records[0],records[1]);self.assertEqual(len(manager.read_pages),1)
        self.assertFalse(manager.visible_knowledge)

    def test_different_qualification_views_of_same_version_are_not_spliced(self):
        knowledge=self.candidate();manager=self.fresh_manager();records=[];pending=[]
        for value in ['first','second']:
            result,_=read_experience(self.session,ids=[knowledge['id']],view='statement',conditions={'scope':value},budget_bytes=1100)
            pending.append(result);manager._admit(result)
        self.assertFalse(manager.selectable_knowledge)
        for result in pending:
            records.append(self.finish(manager,result,'statement'))
        self.assertNotEqual(records[0]['applicability'],records[1]['applicability'])
        self.assertEqual(len(manager.read_pages),2);self.assertFalse(manager.visible_knowledge)

    def test_compatible_overlapping_pages_from_different_read_budgets_merge(self):
        knowledge=self.candidate();manager=self.fresh_manager()
        result,_=read_experience(self.session,ids=[knowledge['id']],view='statement',budget_bytes=1100)
        larger,_=read_experience(self.session,ids=[knowledge['id']],view='statement',budget_bytes=1500)
        self.assertGreater(len(larger['records'][0]['fragment']),len(result['records'][0]['fragment']))
        manager._admit(result);manager._admit(larger)
        self.assertFalse(manager.selectable_knowledge)
        statement=self.finish(manager,result,'statement')
        self.assertEqual(statement['revision'],knowledge['revision']);self.assertEqual(len(manager.read_pages),1)
        self.assertFalse(manager.visible_knowledge)

