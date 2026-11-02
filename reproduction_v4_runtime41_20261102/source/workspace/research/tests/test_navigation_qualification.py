"""Navigation cannot become support; its warnings survive real read/packing paths."""
import copy,hashlib,json,unittest
from pathlib import Path
import test_promotion as p
import test_material_selection as selection
from kyb import library
from kyb.knowledge import read_experience
from kyb.preparation import pack,history_stub
from kyb.readout import preview,size


class NavigationQualification(unittest.TestCase):
    setUp=p.PromotionTests.setUp
    tearDown=p.PromotionTests.tearDown
    add_body_case=p.PromotionTests.add_body_case
    canonical_ref=p.PromotionTests.canonical_ref
    case_patch=p.PromotionTests.case_patch
    op=p.PromotionTests.op
    write=p.PromotionTests.write
    observation=p.PromotionTests.observation
    attach=p.PromotionTests.attach
    manager=p.PromotionTests.manager
    read=selection.MaterialSelection.read
    publish=selection.MaterialSelection.publish
    config=selection.MaterialSelection.config

    def related(self,k):
        return {'type':'knowledge','id':k['id'],'revision':k['revision'],'relation':'related','reason':'Historical comparison, not evidence for this claim.'}

    def configuration(self,refs,**fields):
        return {'op':'add','domain':'body','family':'geometry','question':'Which public body configuration applies?',
            'text':'The exact public configuration source defines this body relation.','conditions':{},
            'strength':'configuration','use':'Use only within the declared source contract.','unknowns':[],
            'depends_on':['body'],'refs':refs,**fields}

    def submit(self,op,*,knowledge=(),cases=(),allowed=None):
        with self.store.db:
            return library.publish_knowledge(self.store,op,allowed={self.prior['id']} if allowed is None else allowed,
                visible_cases={c['claim_id']:c['revision'] for c in cases},
                visible_knowledge={k['id']:k['revision'] for k in knowledge},
                dependencies=self.session.dependencies,frontier=self.session.source_frontier())

    def view(self,k,**kwargs):
        return library.read_knowledge(self.store,'',context={},dependencies=self.session.dependencies,ids=[k['id']],**kwargs)[0]

    def warned_configuration(self):
        a=self.add_body_case();b=self.add_body_case(text='A contrasting observed scope.')
        prior=self.write(self.op([self.canonical_ref(a),self.canonical_ref(b,'counter')]),[a,b])
        k=self.submit(self.configuration([{'type':'evidence','id':self.prior['id']},self.related(prior)]),knowledge=[prior])
        return k,prior

    def test_different_contract_navigation_keeps_own_source_pins_and_read_guard(self):
        prior=self.config('old')
        self.session.dependencies={**self.deps,'body':'new-body'}
        raw=self.session.evidence('known_config',{'action_spec':'New exact body configuration'})
        op=self.configuration([{'type':'evidence','id':raw['id']},self.related(prior)])
        with self.assertRaisesRegex(ValueError,'complete actually delivered statement or detail'):
            self.submit(op,allowed={raw['id']})
        k=self.submit(op,knowledge=[prior],allowed={raw['id']});view=self.view(k)
        self.assertTrue(view['default_eligible']);self.assertEqual(view['support_cases'],0)
        self.assertEqual(view['related_warnings'][0]['flags'],['dependency_changed'])
        pins=json.loads(self.store.db.execute('SELECT dependencies FROM reusable_knowledge WHERE revision=?',(k['revision'],)).fetchone()[0])
        self.assertEqual(pins,{'body':'new-body'})

    def test_knowledge_requires_explicit_navigation_relation_and_reason(self):
        prior=self.config('old')
        for relation,reason in [(None,'Navigation'),('support','Navigation'),('counter','Navigation'),('related',' ')]:
            ref=self.related(prior);ref['reason']=reason
            if relation is None:ref.pop('relation')
            else:ref['relation']=relation
            with self.assertRaisesRegex(ValueError,'relation=related'):
                self.submit(self.configuration([{'type':'evidence','id':self.prior['id']},ref]),knowledge=[prior])

    def test_navigation_alone_cannot_certify_configuration_or_supply_parameters(self):
        prior=self.config('old');case=self.add_body_case()
        with self.assertRaisesRegex(ValueError,'actual known configuration'):
            self.submit(self.configuration([self.related(prior)],domain='external',family='scene_geometry',depends_on=[]),knowledge=[prior])
        for ref in [self.related(prior),self.canonical_ref(case,'related')]:
            op=self.configuration([{'type':'evidence','id':self.prior['id']},ref],params=[{
                'name':'unsupported_value','value':.5,'unit':'m','frame':'world','source':ref}])
            with self.assertRaisesRegex(ValueError,'Navigation references cannot source parameters'):
                self.submit(op,knowledge=[prior],cases=[case])

    def test_direct_evidence_contract_remains_strict(self):
        case=self.add_body_case(depends_on=['body','depth'])
        raw=self.session.evidence('surface_inspection',{'dependencies':{'body':'b1'}})
        with self.assertRaisesRegex(ValueError,'Secondary source contracts differ or are missing'):
            self.submit(self.op([self.canonical_ref(case),{'type':'evidence','id':raw['id']}],depends_on=['body','depth']),cases=[case],allowed={raw['id']})

    def test_direct_case_revision_is_warned_one_hop_without_old_frontier_leak(self):
        a=self.add_body_case();prior=self.write(self.op([self.canonical_ref(a)]),[a])
        k=self.submit(self.configuration([{'type':'evidence','id':self.prior['id']},self.related(prior)]),knowledge=[prior])
        old_at=self.store.revision
        patch={**self.case_patch(),'op':'revise','claim_id':a['claim_id'],'claim_revision':a['revision'],'text':'A revised interpretation.'};patch.pop('local')
        revised=self.store.submit(patch,allowed_sources=self.session.allowed,cutoff=self.session.cutoff,
            dependencies=self.deps,episode=self.session.episode,visible_revisions={a['claim_id']:a['revision']})
        self.assertTrue(revised['accepted'],revised)
        current=self.view(k);self.assertTrue(current['default_eligible'])
        self.assertEqual(current['related_warnings'][0]['flags'],['revised']);self.assertNotIn('basis_stale',current)
        self.assertNotIn('related_warnings',self.view(k,case_at=old_at))

    def test_related_case_revised_warns_but_never_becomes_support(self):
        a=self.add_body_case();ref=self.canonical_ref(a,'related')
        k=self.submit(self.configuration([{'type':'evidence','id':self.prior['id']},ref]),cases=[a]);old_at=self.store.revision
        patch={**self.case_patch(),'op':'revise','claim_id':a['claim_id'],'claim_revision':a['revision'],'text':'Reinterpreted navigation.'};patch.pop('local')
        revised=self.store.submit(patch,allowed_sources=self.session.allowed,cutoff=self.session.cutoff,
            dependencies=self.deps,episode=self.session.episode,visible_revisions={a['claim_id']:a['revision']})
        self.assertTrue(revised['accepted'],revised);view=self.view(k)
        self.assertTrue(view['default_eligible']);self.assertEqual(view['source_revised'],[]);self.assertEqual(view['support_cases'],0)
        self.assertEqual(view['related_warnings'][0]['current'],revised['revision'])
        self.assertNotIn('related_warnings',self.view(k,case_at=old_at))

    def test_warning_survives_complete_packet_preview_history_and_statement_pages(self):
        k,prior=self.warned_configuration();view=self.view(k);warnings=view['related_warnings']
        self.assertTrue(view['default_eligible']);self.assertFalse(view['source_contested'])
        self.assertEqual(warnings[0]['flags'],['contested'])
        self.assertEqual(preview(view)['related_warnings'],warnings)
        self.assertEqual(history_stub(view)['related_warnings'],warnings)
        m=self.manager();self.read(m,[k['id']]);self.assertTrue(self.publish(m,default_knowledge=[k['id']])[1])
        packet=m.prefix_view();self.assertEqual(packet['knowledge'][0]['related_warnings'],warnings)
        self.assertLessEqual(packet['packing']['experience_serialized_bytes'],8192)
        self.assertEqual(m.material_delivery()['packet_sha256'],hashlib.sha256(json.dumps(packet,ensure_ascii=False).encode()).hexdigest())
        result,_=read_experience(self.session,level='knowledge',ids=[k['id']],view='statement',budget_bytes=1100)
        fragments={}
        while result['records'][0]['view']=='statement_page':
            row=result['records'][0];fragments[row['offset']]=row['fragment'];m._admit(result)
            if not row['continuation']:break
            result,_=read_experience(self.session,continuation=row['continuation'],budget_bytes=1100)
        record=json.loads(''.join(fragments[k] for k in sorted(fragments))) if fragments else result['records'][0]
        self.assertEqual(record['related_warnings'],warnings)
        for padding in range(6000,9000,100):
            large=copy.deepcopy(view);large['unknowns']=['x'*padding]
            packed=pack({},[large],[],[])
            if packed['knowledge']:self.assertEqual(packed['knowledge'][0]['related_warnings'],warnings)
            for pointer in packed['memory_index']:self.assertEqual(pointer['related_warnings'],warnings)
            self.assertLessEqual(packed['packing']['experience_serialized_bytes'],8192)

    def test_unread_selection_keeps_warnings_and_declares_they_were_not_delivered(self):
        from types import SimpleNamespace
        from kyb.readout import pack as pack_readout
        k,_=self.warned_configuration();view=self.view(k)
        session=SimpleNamespace()
        answer=pack_readout(session,[view],{},budget_bytes=2000,limit=0,view='statement')
        self.assertFalse(answer['records']);self.assertEqual(answer['unread_records'],1)
        saved,_=session._read_continuations[answer['selection_continuation']]
        self.assertEqual(saved['records'][0]['related_warnings'],view['related_warnings'])
        self.assertIn('not been delivered',answer['unread_qualification'])
        oversized=copy.deepcopy(view);oversized['related_warnings']*=12
        answer=pack_readout(session,[oversized],{},budget_bytes=1000,limit=0,view='statement')
        self.assertLessEqual(len(json.dumps(answer,ensure_ascii=False).encode()),1000)
        saved,_=session._read_continuations[answer['selection_continuation']]
        self.assertEqual(saved['records'][0]['related_warnings'],oversized['related_warnings'])

    def test_recursive_source_visibility_is_preserved(self):
        old_frontier=self.session.source_frontier();raw=self.session.evidence('known_config',{'action_spec':'Future public source'})
        prior=self.submit(self.configuration([{'type':'evidence','id':raw['id']}]),allowed={raw['id']})
        k=self.submit(self.configuration([{'type':'evidence','id':self.prior['id']},self.related(prior)]),knowledge=[prior])
        self.assertEqual(library.read_knowledge(self.store,'',context={},dependencies=self.deps,ids=[k['id']],frontier=old_frontier),[])

    def test_warning_is_in_archived_cli_preparation_with_matching_sha(self):
        import test_codex_cli as cli
        k,_=self.warned_configuration();view=self.view(k);packet=pack({},[view],[],[])
        fake=cli.CodexCliTests();fake.setUp()
        try:
            fake.client.knowledge_provider=lambda:packet
            fake.client.complete([fake.observation(0)],fake.tools)
            folder=fake.client.root/'cli'/fake.client.last_attempt_id/'input'
            actual=json.loads((folder/'preparation.json').read_text());metrics=json.loads((folder/'context_metrics.json').read_text())
            self.assertEqual(actual['knowledge'][0]['related_warnings'],view['related_warnings'])
            self.assertEqual(metrics['preparation_sha256'],hashlib.sha256(json.dumps(actual,ensure_ascii=False).encode()).hexdigest())
            self.assertIn('related_warnings',(folder/'instructions.txt').read_text())
        finally:fake.tearDown()
