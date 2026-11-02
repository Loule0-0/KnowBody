"""Physical-process statements preserve authority while cost detail stays readable."""
import copy,json,unittest
import test_promotion as p
from kyb import readout


class EpisodeStatementCosts(unittest.TestCase):
    tearDown=p.PromotionTests.tearDown
    observation=p.PromotionTests.observation
    attach=p.PromotionTests.attach
    manager=p.PromotionTests.manager

    def setUp(self):
        p.PromotionTests.setUp(self)
        self.reader=self.manager()

    def candidate(self):
        return {'id':'previous_episode','revision':50,'layer':'episode_candidate',
            'task':'A public manipulation task','official_success':True,'reviewed':False,
            'stages':[{'summary':'Observed retained transport and opening; not a causal rule.',
                       'conditions':{'grasp':'This observed grasp only'},
                       'unknowns':['New-scene feasibility and hidden contacts']}],
            'mechanical_gaps':['No continuous contact observation'],
            'prefix_replay':None,'infrastructure_restart':{'kind':'explicitly reconstructed'},
            'trace_entry':{'chain':'old:chain','outline':'old:outline','result':'old:result','cost':'old:cost'},
            'qualification':'Whole outcome does not validate a stage or grant physical evidence.',
            'role_costs':{'primary':{'cli_invocations':91,'outcome_unknown':1,
                'usage_reported':None,'usage_scope':'native_responses',
                'qualification':'Partial usage; missing metadata is not zero. '*70},
                'manager':{'cli_invocations':17,'captured_metadata_consistent':False}}}

    def pack(self,record,view='statement',budget=6000):
        return readout.pack(self.session,[record],{'question':'What process was observed?'},
                            budget_bytes=budget,limit=1,view=view)

    def reconstruct(self,result,budget=1100):
        fragments={};manager=self.reader
        while True:
            record=result['records'][0] if 'records' in result else result
            manager._admit({'records':[record]})
            if 'fragment' not in record:return record,manager
            self.assertFalse(record['complete_record'])
            fragments[record['offset']]=record['fragment']
            token=record['continuation']
            if not token:break
            result=readout.page(self.session,token,budget)
        raw=''.join(fragments[key] for key in sorted(fragments))
        self.assertEqual(len(raw),record['total_characters'])
        return json.loads(raw),manager

    def test_statement_preserves_entire_physical_contract_and_original_record(self):
        original=self.candidate();saved=copy.deepcopy(original)
        delivered=self.pack(original)['records'][0]
        expected=copy.deepcopy(original);expected.pop('role_costs')
        expected.update(view='statement',complete_record=False,statement_complete=True)
        token=delivered.pop('continuation')
        self.assertEqual(delivered,expected)
        self.assertEqual(original,saved)
        self.assertEqual(self.session._read_continuations[token][0]['role_costs'],original['role_costs'])

    def test_direct_and_continued_cost_details_retain_unknowns_without_evidence_permission(self):
        original=self.candidate()
        statement=self.pack(original)['records'][0]
        manager=self.reader;manager._admit({'records':[statement]})
        self.assertFalse(manager.allowed);self.assertFalse(manager.visible_revisions)
        self.assertFalse(manager.visible_knowledge);self.assertFalse(manager.selectable_knowledge)
        continued,manager=self.reconstruct(readout.page(self.session,statement['continuation'],1100))
        direct,unused=self.reconstruct(self.pack(original,view='detail',budget=1100))
        self.assertEqual(continued['role_costs'],original['role_costs'])
        self.assertEqual(direct['role_costs'],original['role_costs'])
        self.assertTrue(continued['complete_record'])
        self.assertFalse(manager.allowed);self.assertFalse(manager.visible_revisions)
        self.assertFalse(manager.visible_knowledge);self.assertFalse(manager.selectable_knowledge)

    def test_non_episode_and_detail_keep_original_cost(self):
        original=self.candidate();original['layer']='case'
        self.assertEqual(readout.statement(original)['role_costs'],original['role_costs'])
        original['layer']='episode_candidate'
        detail,_=self.reconstruct(self.pack(original,view='detail',budget=6000))
        self.assertEqual(detail['role_costs'],original['role_costs'])

    def test_small_budget_still_requires_complete_statement_pages(self):
        original=self.candidate();original['stages'][0]['unknowns']=['Unresolved physical premise. '*140]
        result=self.pack(original,budget=1100)
        self.assertEqual(result['records'][0]['view'],'statement_page')
        self.assertFalse(result['records'][0]['complete_record'])
        record,manager=self.reconstruct(result)
        self.assertEqual(record['stages'],original['stages'])
        self.assertNotIn('role_costs',record)
        self.assertTrue(record['statement_complete']);self.assertFalse(record['complete_record'])
        self.assertFalse(manager.allowed);self.assertFalse(manager.visible_knowledge)
