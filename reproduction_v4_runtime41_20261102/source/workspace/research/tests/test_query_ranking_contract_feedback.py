"""Question-first case retrieval and precise conservative secondary-contract errors."""
import json,unittest
import test_promotion as p
from kyb import library
from kyb.knowledge import read_experience

class CaseRanking(unittest.TestCase):
    setUp=p.PromotionTests.setUp
    tearDown=p.PromotionTests.tearDown
    add_body_case=p.PromotionTests.add_body_case
    observation=p.PromotionTests.observation
    def pair(self):
        egg=self.add_body_case(reading_role='scene_observation',observed_entity='scene_object',
             text='Egg case lid closure: the egg case lid remains open above the egg container.')
        doll=self.add_body_case(text='Doll container case lid sits near the gripper during a nested doll transfer.')
        return egg,doll
    def test_relevant_historical_case_survives_first_limit_with_qualifications(self):
        egg,doll=self.pair()
        out=self.store.read('egg case lid',context={},dependencies=self.deps,episode='different_layout',limit=1)
        self.assertEqual(out['selected'][0]['id'],egg['claim_id'])
        self.assertEqual(out['selected'][0]['qualification']['category'],'historical_scene_observation')
        self.assertFalse(out['selected'][0]['qualification']['current_fact'])
    def test_full_query_keeps_question_order_after_case_merge(self):
        egg,doll=self.pair()
        result,_=read_experience(self.session,question='egg case lid',level='case',limit=2)
        self.assertEqual(result['records'][0]['id'],egg['claim_id'])
        self.assertEqual(result['records'][0]['applicability']['category'],'historical_scene_observation')
    def test_browse_keeps_prior_qualification_order(self):
        egg,doll=self.pair()
        out=self.store.read('',context={},dependencies=self.deps,episode='different_layout',limit=1,
                            browse=True,structured=True,domain='body',family='geometry')
        self.assertEqual(out['selected'][0]['id'],doll['claim_id'])
    def test_exact_case_order_ignores_question_and_no_permission_is_granted(self):
        egg,doll=self.pair()
        refs=[{'id':x['claim_id'],'revision':x['revision']} for x in (doll,egg)]
        result,_=read_experience(self.session,question='egg case lid',level='case',case_refs=refs,limit=2,view='statement')
        self.assertEqual([r['id'] for r in result['records']],[doll['claim_id'],egg['claim_id']])
        self.assertTrue(all(r.get('view')!='detail' for r in result['records']))

class SecondaryDiagnostics(unittest.TestCase):
    setUp=p.PromotionTests.setUp
    tearDown=p.PromotionTests.tearDown
    add_body_case=p.PromotionTests.add_body_case
    observation=p.PromotionTests.observation
    canonical_ref=p.PromotionTests.canonical_ref
    op=p.PromotionTests.op
    write=p.PromotionTests.write
    def test_direct_secondary_reports_each_missing_and_changed_derived_contract(self):
        raw=self.session.evidence('surface_inspection',{'dependencies':{'body':'b1'}})
        obs=self.observation(dependencies={**self.deps,'body':'b-new','execution':'execution-new'})
        current=self.add_body_case(obs,text='A distinct synthetic body observation under a new source contract.',depends_on=['body','execution'])
        # Case pins are source-derived, not the supplied current request dependency.
        operation=self.op([self.canonical_ref(current),{'type':'evidence','id':raw['id']}],depends_on=['body','execution'])
        before=self.store.db.execute('SELECT count(*) FROM reusable_knowledge').fetchone()[0]
        with self.assertRaisesRegex(ValueError,'Secondary source contracts differ or are missing') as got:
            library.publish_knowledge(self.store,operation,allowed={raw['id']},visible_cases={current['claim_id']:current['revision']},
                visible_knowledge={},dependencies={'body':'unrelated-current-request'},frontier=self.session.source_frontier())
        detail=json.loads(str(got.exception).split(': ',1)[1]);self.assertEqual(detail['basis'],'supporting_case_derived_contract_not_current_request')
        self.assertEqual(len(detail['sources']),1);conflict=detail['sources'][0]
        self.assertEqual(conflict['id'],raw['id']);self.assertIsNone(conflict['revision'])
        self.assertEqual(set(conflict['mismatches']),{'body','execution'})
        self.assertEqual(conflict['mismatches']['body']['expected'],'b-new')
        self.assertTrue(conflict['mismatches']['execution']['missing_or_unknown'])
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM reusable_knowledge').fetchone()[0],before)
