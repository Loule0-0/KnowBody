"""Support-audit growth must not crowd qualified knowledge statements."""
import copy,json,unittest
from types import SimpleNamespace
import test_promotion as p
from kyb import library,readout
from kyb.knowledge import read_experience


class KnowledgeSupportProjection(unittest.TestCase):
    setUp=p.PromotionTests.setUp
    tearDown=p.PromotionTests.tearDown
    observation=p.PromotionTests.observation
    attach=p.PromotionTests.attach
    add_body_case=p.PromotionTests.add_body_case
    manager=p.PromotionTests.manager
    op=p.PromotionTests.op
    write=p.PromotionTests.write
    canonical_ref=p.PromotionTests.canonical_ref

    def knowledge(self):
        cases=[self.add_body_case(text=text) for text in ('First limited observation.',
            'Second limited observation.','A scoped counterexample.','Navigation-only related observation.')]
        refs=[self.canonical_ref(cases[0]),self.canonical_ref(cases[1]),
              self.canonical_ref(cases[2],'counter'),self.canonical_ref(cases[3],'related')]
        with self.store.db:result=self.write(self.op(refs,conditions={'scope':'These observations only'},
            unknowns=['Unmeasured force and new-scene feasibility']),cases)
        card=library.read_knowledge(self.store,'',ids=[result['id']],context={},dependencies=self.deps)[0]
        return result,card

    def test_only_support_pointers_are_hidden_while_physical_answer_and_navigation_remain(self):
        _,card=self.knowledge();card['related_warnings']=[{'id':'related_version','counter_cases':2}]
        card['source_revised']=['changed_exact_source']
        card['applicability']['different_conditions']=['grasp']
        card['applicability']['different_descriptions']={'grasp':{'source':'A','current':'B'}}
        card['params']=[{'name':'sample','value':.1,'unit':'m','frame':'world',
                        'source':{'type':'case','id':'exact_parameter_case','revision':7}}]
        original=copy.deepcopy(card);statement=readout.statement(card)
        self.assertEqual(card,original)
        for key in ['text','conditions','unknowns','source_contested','source_revised','related_warnings',
                    'applicability','default_eligible','status','strength','params']:
            self.assertEqual(statement[key],original[key])
        self.assertEqual(statement['drill_down']['support_source_refs_omitted'],2)
        kept=statement['drill_down']['source_refs']
        self.assertEqual([ref['relation'] for ref in kept],['counter','related'])
        self.assertEqual([ref['id'] for ref in kept],[r['id'] for r in original['drill_down']['refs'] if r['relation']!='support'])
        self.assertEqual(statement['drill_down']['qualification'],original['drill_down']['qualification'])
        self.assertEqual(readout.statement(statement),statement)

    def test_statement_does_not_admit_hidden_supports_but_detail_restores_exact_original(self):
        result,card=self.knowledge();manager=self.manager()
        answer,_=read_experience(self.session,level='knowledge',ids=[result['id']],view='statement')
        statement=answer['records'][0];self.assertTrue(statement['statement_complete'])
        manager._admit(answer)
        self.assertEqual(manager.selectable_knowledge[result['id']],result['revision'])
        self.assertFalse(manager.allowed);self.assertFalse(manager.visible_revisions);self.assertFalse(manager.visible_knowledge)
        detail,_=read_experience(self.session,continuation=statement['continuation'],budget_bytes=20000)
        restored=detail
        if 'records' in detail:restored=detail['records'][0]
        if 'fragment' in restored:restored=json.loads(restored['fragment'])
        self.assertEqual(restored['drill_down']['refs'],card['drill_down']['refs'])
        self.assertNotIn('support_source_refs_omitted',restored['drill_down'])
        self.assertEqual(restored['text'],card['text']);self.assertEqual(restored['params'],card['params'])

    def test_nonknowledge_views_and_direct_detail_are_unchanged(self):
        _,card=self.knowledge()
        for layer in ('case','episode_candidate','public_contract','observation'):
            sample=copy.deepcopy(card);sample['layer']=layer
            self.assertEqual(len(readout.statement(sample)['drill_down']['source_refs']),4)
            self.assertNotIn('support_source_refs_omitted',readout.statement(sample)['drill_down'])
        detail=readout.pack(SimpleNamespace(),[card],{},budget_bytes=20000,limit=1,view='detail')['records'][0]
        self.assertEqual(detail['drill_down'],card['drill_down'])
        self.assertTrue(detail['complete_record'])

    def test_added_support_pointers_have_only_count_digit_cost_in_statement(self):
        _,card=self.knowledge();sizes=[]
        for count in (1,1001):
            sample=copy.deepcopy(card)
            sample['drill_down']['refs']=[{'type':'case','id':f'case_{i:04d}',
                'revision':i+1,'relation':'support','reason':'Audit only.'} for i in range(count)]
            statement=readout.statement(sample)
            self.assertEqual(statement['drill_down']['support_source_refs_omitted'],count)
            self.assertEqual(statement['drill_down']['source_refs'],[])
            sizes.append(readout.size(statement))
        self.assertLess(sizes[1]-sizes[0],10)
