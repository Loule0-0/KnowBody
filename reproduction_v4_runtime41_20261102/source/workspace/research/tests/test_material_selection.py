"""Selection is atomic; historical navigation never becomes a claim or read right."""
import copy,json,unittest
import test_promotion as p
from kyb import library
from kyb.knowledge import read_experience
from kyb.preparation import pack,add_history,history_stub,delivery_manifest

class MaterialSelection(unittest.TestCase):
    setUp=p.PromotionTests.setUp
    tearDown=p.PromotionTests.tearDown
    observation=p.PromotionTests.observation
    attach=p.PromotionTests.attach
    manager=p.PromotionTests.manager
    add_body_case=p.PromotionTests.add_body_case
    op=p.PromotionTests.op
    write=p.PromotionTests.write
    canonical_ref=p.PromotionTests.canonical_ref
    case_patch=p.PromotionTests.case_patch

    def config(self,name,execution=False):
        op={'op':'add','domain':'body','family':'geometry','question':'Which recorded '+name+' configuration?','text':'Known source configuration '+name,
            'conditions':{},'strength':'configuration','use':'Read the source frame before comparison.',
            'unknowns':[],'depends_on':['body','execution'] if execution else ['body'],
            'refs':[{'type':'evidence','id':self.prior['id']}]}
        with self.store.db:
            return library.publish_knowledge(self.store,op,allowed={self.prior['id']},visible_cases={},visible_knowledge={},dependencies=self.deps,frontier=self.session.source_frontier())

    def pair(self):
        eligible=self.config('stable');historical=self.config('execution',True)
        self.session.dependencies={**self.deps,'execution':'different-execution'}
        manager=self.manager()
        self.read(manager,[eligible['id'],historical['id']])
        return manager,eligible,historical

    def read(self,manager,ids,view='statement',budget=16000):
        result=manager._search({'level':'knowledge','ids':ids,'view':view,'budget_bytes':budget})
        manager._admit(result);return result

    def publish(self,manager,**kwargs):
        return manager._publish_groups({'expected_version':manager.version,'answer':'Scoped offline test','groups':[],**kwargs})

    def test_mixed_default_selection_rejects_without_silent_filter_then_legal_retry(self):
        m,good,old=self.pair();before=m.version
        result,done=self.publish(m,default_knowledge=[good['id'],old['id']])
        self.assertFalse(done);self.assertEqual(result['status'],'NEEDS_WORK');self.assertEqual(m.default_ids,[]);self.assertEqual(m.version,before)
        details=result['issues'][0]['details'];self.assertEqual(details['items'][0]['qualification']['id'],old['id'])
        self.assertFalse(details['legal_subset_forecast']['submitted'])
        result,done=self.publish(m,default_knowledge=[good['id']],history_refs=[old['id']])
        self.assertTrue(done);self.assertEqual(m.default_ids,[good['id']]);self.assertEqual(m.history_versions,{old['id']:old['revision']})
        packet=m.prefix_view();stub=packet['history_index'][0]
        self.assertEqual(stub['id'],old['id']);self.assertFalse(stub['statement_complete']);self.assertFalse(stub['complete_record'])
        self.assertEqual(stub['title'],'Which recorded execution configuration?')
        self.assertEqual(stub['title_source'],'question');self.assertFalse(stub['title_truncated'])
        self.assertTrue(stub['navigation_only']);self.assertEqual(stub['applicability']['changed_dependencies'],['execution'])
        for key in ['text','params','conditions','resolution','question']:self.assertNotIn(key,stub)
        self.assertFalse(m.visible_knowledge);self.assertFalse(m.visible_revisions);self.assertFalse(m.allowed)
        self.assertEqual(result['delivery_manifest']['packet_sha256'],m.material_delivery()['packet_sha256'])
        self.assertEqual(result['delivery_manifest']['history_refs'][0]['delivery'],'navigation')

    def test_history_discovery_title_is_utf8_bounded_without_changing_qualifications(self):
        record={'id':'historical','revision':7,'domain':'external','family':'retention_release',
            'status':'active','strength':'observation','default_eligible':False,
            'question':'钥匙转姿时夹持与扫掠空间需要核对哪些条件？'*40,
            'text':'This physical answer must not enter a navigation-only title.',
            'conditions':{'grasp':'Unverified current grasp'},'params':[{'value':123}],
            'applicability':{'changed_dependencies':['execution'],'unknown_conditions':['grasp']},
            'counter_cases':1,'source_contested':True,
            'drill_down':{'refs':[{'type':'case','id':'counter','revision':4,'relation':'counter'}]}}
        stub=history_stub(record)
        self.assertLessEqual(len(stub['title'].encode()),240)
        self.assertTrue(record['question'].startswith(stub['title']))
        self.assertTrue(stub['title_truncated']);self.assertEqual(stub['title_source'],'question')
        self.assertEqual(stub['applicability'],record['applicability'])
        self.assertTrue(stub['source_contested']);self.assertEqual(stub['counter_cases'],1)
        self.assertEqual(stub['counter_sources'],[{'type':'case','id':'counter','revision':4}])
        self.assertFalse(stub['complete_record']);self.assertFalse(stub['statement_complete'])
        self.assertEqual(stub['read'],{'level':'knowledge','ids':['historical'],'view':'statement'})
        for field in ('question','text','conditions','params'):
            self.assertNotIn(field,stub)

    def test_unread_eligible_hypothesis_and_retired_history_are_rejected(self):
        m,good,old=self.pair()
        m.read_knowledge_versions.pop(old['id'])
        self.assertFalse(self.publish(m,history_refs=[old['id']])[1]);self.assertEqual(m.history_ids,[])
        self.read(m,[old['id']])
        self.assertFalse(self.publish(m,history_refs=[good['id']])[1])
        case=self.add_body_case()
        with self.store.db:hyp=self.write(self.op([self.canonical_ref(case)]),[case])
        m.knowledge_revision=hyp['revision'];m.revision=self.store.revision
        self.read(m,[hyp['id']])
        self.assertFalse(self.publish(m,history_refs=[hyp['id']])[1])
        with self.store.db:
            retired=library.publish_knowledge(self.store,{'op':'retire','id':old['id'],'reason':'Explicit offline retirement'},
                allowed=set(),visible_cases={},visible_knowledge={old['id']:old['revision']},dependencies=self.session.dependencies,frontier=self.session.source_frontier())
        m.knowledge_revision=retired['revision'];self.read(m,[old['id']])
        self.assertFalse(self.publish(m,history_refs=[old['id']])[1])

    def test_partial_statement_never_authorizes_history_selection(self):
        m,good,old=self.pair();m.read_knowledge_versions={};m.selectable_knowledge={};m.visible_knowledge={}
        result=self.read(m,[old['id']],budget=1000)
        self.assertEqual(result['records'][0]['view'],'statement_page')
        self.assertFalse(self.publish(m,history_refs=[old['id']])[1]);self.assertEqual(m.history_ids,[])

    def test_stale_read_revision_is_not_absorbed(self):
        m,good,old=self.pair()
        with self.store.db:
            changed=library.publish_knowledge(self.store,{'op':'retire','id':old['id'],'reason':'A new current version'},
                allowed=set(),visible_cases={},visible_knowledge={old['id']:old['revision']},dependencies=self.session.dependencies,frontier=self.session.source_frontier())
        m.knowledge_revision=changed['revision']
        self.assertFalse(self.publish(m,history_refs=[old['id']])[1]);self.assertEqual(m.read_knowledge_versions[old['id']],old['revision'])

    def test_invalid_selection_preserves_independent_committed_group_and_retry_is_idempotent(self):
        m,good,old=self.pair();m.allowed={self.post['id']}
        group={'memory_patch':[self.case_patch()],'dispositions':[{'case':'c1','action':'keep','reason':'Scoped body hypothesis.'}]}
        result,done=m._publish_groups({'expected_version':m.version,'answer':'Independent case','groups':[group],'default_knowledge':[old['id']]})
        self.assertFalse(done);self.assertEqual(len(result['memory_updates']),1);self.assertEqual(m.default_ids,[])
        before=self.store.revision
        result,done=self.publish(m,default_knowledge=[good['id']],history_refs=[old['id']])
        self.assertTrue(done);self.assertEqual(self.store.revision,before)
        result,done=m._publish_groups({'expected_version':m.version,'answer':'Same group','groups':[group]})
        self.assertTrue(done);self.assertTrue(result['committed_groups'][0]['already_committed']);self.assertEqual(self.store.revision,before)

    def test_new_committed_local_knowledge_does_not_count_as_history_read(self):
        m,good,old=self.pair();m.allowed={self.prior['id']}
        op={'op':'add','local':'k1','domain':'body','family':'geometry','question':'New source?','text':'A new configuration source.',
            'conditions':{},'strength':'configuration','use':'Read it explicitly.','depends_on':['body','execution'],
            'refs':[{'type':'evidence','id':self.prior['id']}]}
        result,done=m._publish_groups({'expected_version':m.version,'answer':'New source','groups':[{'knowledge_ops':[op]}],'history_refs':['k1']})
        self.assertFalse(done);new=result['knowledge_updates'][0]
        self.assertNotIn(new['id'],m.read_knowledge_versions)
        self.assertFalse(self.publish(m,history_refs=[new['id']])[1])
        self.read(m,[new['id']])
        self.assertTrue(self.publish(m,history_refs=[new['id']])[1])

    def test_no_history_and_budget_omission_leave_original_packet_identical(self):
        m,good,old=self.pair();result,_=self.publish(m,default_knowledge=[good['id']])
        original=m.prefix_view();self.assertEqual(add_history(original,[]),original)
        card=library.read_knowledge(self.store,'',context=self.session.context,dependencies=self.session.dependencies,ids=[old['id']])[0]
        huge=copy.deepcopy(card);huge['applicability']['unknown_conditions']=['x'*9000]
        self.assertEqual(json.dumps(add_history(original,[huge])),json.dumps(original))
        appended=add_history(original,[card]);self.assertEqual(appended['knowledge'],original['knowledge']);self.assertEqual(appended['memory_index'],original['memory_index'])
        self.assertEqual(appended['episode_candidates'],original['episode_candidates']);self.assertLessEqual(appended['packing']['experience_serialized_bytes'],8192)

    def test_all_case_reference_issues_have_exact_original_handles(self):
        m,good,old=self.pair();before=self.store.revision
        op=self.op([{'type':'case','local':'c1','relation':'support'},{'type':'case','local':'c2','reason':'Concrete comparison'}])
        result,done=m._publish_groups({'expected_version':m.version,'answer':'Bad reference schema','groups':[{
            'memory_patch':[self.case_patch(),{**self.case_patch(),'local':'c2'}],'knowledge_ops':[op]}]})
        self.assertFalse(done);reason=result['issues'][0]['reason'];self.assertIn('knowledge_ops[0].refs[0]',reason);self.assertIn('knowledge_ops[0].refs[1]',reason)
        self.assertIn('c1',reason);self.assertIn('c2',reason);self.assertEqual(self.store.revision,before)

    def test_history_statement_and_detail_have_identical_qualifications_and_fit(self):
        from kyb.readout import statement,size
        m,good,old=self.pair()
        full=library.read_knowledge(self.store,'',context=self.session.context,dependencies=self.session.dependencies,ids=[old['id']])[0]
        full['source_revised']=[];full['related_warnings']=[]
        full['applicability'].update(different_descriptions=[],different_conditions=[])
        full['drill_down']['refs']=[{'type':'case','id':'counter-source','revision':37,'relation':'counter','reason':'A scoped counter observation.'}]
        selected=statement(full)
        self.assertEqual(history_stub(full),history_stub(selected))
        self.assertEqual(history_stub(full)['counter_sources'],[{'type':'case','id':'counter-source','revision':37}])
        self.publish(m,default_knowledge=[good['id']])
        base=m.prefix_view()
        # Exercise the one-byte packing boundary, including final byte counters.
        last_fit=None;first_omitted=None
        for width in range(5000,8193):
            packet=copy.deepcopy(base);packet['padding']='x'*width
            a=add_history(packet,[full]);b=add_history(packet,[selected])
            self.assertEqual(a,b)
            if 'history_index' in a:last_fit=a
            elif last_fit is not None:first_omitted=a;break
        self.assertIsNotNone(last_fit);self.assertIsNotNone(first_omitted)
        self.assertLessEqual(last_fit['packing']['experience_serialized_bytes'],8192)
        self.assertEqual(len(first_omitted['padding']),len(last_fit['padding'])+1)
        forced=copy.deepcopy(first_omitted);forced['history_index']=[history_stub(full)]
        for _ in range(4):
            forced['packing']['experience_serialized_bytes']=size({k:v for k,v in forced.items() if k!='operator_clarifications'})
            forced['packing']['serialized_bytes']=size(forced)
        self.assertGreater(forced['packing']['experience_serialized_bytes'],8192)
