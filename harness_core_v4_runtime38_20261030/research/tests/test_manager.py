"""New transaction contract: thin inputs, real reads, atomic writes, no legacy packets."""
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch as mock_patch
from pathlib import Path
from kyb.memory import MemoryStore
from kyb.session import MemorySession
from kyb.trace import RunTrace
from kyb.context import latest_context
from kyb import library
from manager_fixtures import ScriptedManager, PreparationClient

def publish(version,**kwargs):return ('publish_materials',{'expected_version':version,'answer':'Evidence remains conditional.',**kwargs})

class ManagerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();root=Path(self.temp.name)
        self.store=MemoryStore(root/'memory.sqlite');self.trace=RunTrace(root/'episode',{})
        self.deps={'body':'b1','cameras':'c1','depth':'d1','execution':'e1','attachment:left':'g1','attachment:right':'g2'}
        self.session=MemorySession(self.trace,self.store,task='arbitrary',dependencies=self.deps)
        self.prior=self.session.evidence('known_config',{'action_spec':'Static source','coordinate_contract':{'units':'m'}})
        self.before=self.observation()
        self.receipt=self.session.evidence('receipt',{'before_observation':self.before['id'],'executed_actions':1,'execution':'complete'})
        self.post=self.observation(after_receipt=self.receipt['id'])
    def tearDown(self):self.store.close();self.temp.cleanup()
    def observation(self,**extra):
        return self.session.evidence('observation',{'state':{},'dependencies':dict(self.deps),**extra})
    def attach(self,turns):
        self.client=ScriptedManager(turns);self.session.attach_manager(self.client);return self.session.manager
    def messages(self):return [{'role':'user','content':'Goal and TASK RECIPE remain authoritative'},
        {'role':'user','content':[{'type':'text','text':'Current state sentinel'}, {'type':'image_url','image_url':{'url':'dummy'}}]}]
    def test_prepare_once_and_does_not_inject_scene_or_case_archive(self):
        manager=self.attach([publish(0,default_knowledge=[])])
        messages=self.messages();self.session.refresh_context(messages,None)
        self.observation();self.session.refresh_context(messages,None)
        self.assertEqual(len(self.client.inputs),1)
        request=json.loads(self.client.inputs[0][1]['content'])
        self.assertNotIn('records',request);self.assertNotIn('working',request)
        self.assertNotIn(self.post['id'],json.dumps(request))
        self.assertEqual(request['task'],'arbitrary')
        forecast=request['decision_packet_forecast']
        self.assertEqual(forecast['max_inline_knowledge'],2)
        self.assertEqual(forecast['experience_limit_bytes'],8192)
        self.assertEqual(forecast['complete_knowledge_ids'],[])
        self.assertEqual(forecast['experience_serialized_bytes'],manager.prefix_view()['packing']['experience_serialized_bytes'])
        self.assertNotIn('interaction_history',latest_context(messages))
        self.assertNotIn('materials',latest_context(messages))
    def test_prepare_goal_is_sourced_observation_not_earlier_operator_text(self):
        actual=self.observation(instruction='Stack the currently specified objects, then reset.')
        self.attach([publish(0,default_knowledge=[])])
        self.session.refresh_context([{'role':'user','content':'OPERATOR: unrelated note'},
            *self.messages()],None)
        request=json.loads(self.client.inputs[0][1]['content'])
        self.assertEqual(request['task'],'Stack the currently specified objects, then reset.')
        specification=self.store.evidence_payload(self.session.task_specification['id'])
        self.assertEqual(specification['source_observation_id'],actual['id'])
    def test_review_has_thin_index_not_old_raw_first_page(self):
        manager=self.attach([publish(0),publish(1)])
        manager.run('prepare','prepare');manager.run('Compare this response','review')
        request=json.loads(self.client.inputs[1][1]['content'])
        self.assertNotIn('unprocessed_interval',request);self.assertNotIn('records',request)
        self.assertEqual(request['recent_intervals']['intervals'],[])
        self.assertEqual(request['unprocessed']['intervals'][0]['receipt'],self.receipt['id'])
        self.assertNotIn('state',json.dumps(request['recent_intervals']))
        self.assertFalse(manager.allowed)
    def test_index_does_not_grant_permission_or_process_an_interval(self):
        manager=self.attach([publish(0)])
        manager.run('prepare','prepare');manager.frontier=self.session.source_frontier();manager.purpose='review'
        index=manager._intervals(self.session.episode)
        manager._admit(index)
        self.assertFalse(manager.allowed)
        with self.assertRaisesRegex(ValueError,'complete before'):
            manager._publish({'expected_version':1,'answer':'Nothing verified',
                'processed':[{'receipt_id':self.receipt['id'],'disposition':'no_lesson'}]})
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM processed_intervals').fetchone()[0],0)
    def test_review_can_process_a_complete_read_without_marking_other_intervals(self):
        manager=self.attach([publish(0),('read_evidence',{'ids':[self.before['id'],self.receipt['id'],self.post['id']]}),
            publish(1,processed=[{'receipt_id':self.receipt['id'],'disposition':'no_lesson'}])])
        manager.run('prepare','prepare');manager.run('Compare available response','review')
        self.assertEqual(self.store.db.execute('SELECT receipt_id FROM processed_intervals').fetchone()[0],self.receipt['id'])
        self.assertEqual(manager._intervals(self.session.episode)['intervals'],[])
    def test_future_and_current_scene_prepare_reads_rejected(self):
        manager=self.attach([publish(0)]);manager.run('prepare','prepare')
        with self.assertRaisesRegex(ValueError,'Preparation cannot'):manager._record(self.post['id'])
        manager.purpose='review';future=self.observation()
        with self.assertRaisesRegex(ValueError,'frontier'):manager._record(future['id'])
    def test_failed_multi_source_or_image_read_grants_no_sources(self):
        manager=self.attach([publish(0)]);manager.run('prepare','prepare');manager.purpose='review'
        with self.assertRaises(ValueError):manager._evidence({'ids':[self.post['id'],'missing']})
        self.assertFalse(manager.allowed);self.assertFalse(manager.seen_images)
        path=Path(self.temp.name)/'image.jpg';path.write_bytes(b'changed')
        obs=self.observation(images={'cam':{'path':str(path),'sha256':hashlib.sha256(b'original').hexdigest()}})
        manager.frontier=self.session.source_frontier()
        with self.assertRaisesRegex(ValueError,'RGB changed'):manager._evidence({'ids':[obs['id']],'cameras':['cam']})
        self.assertFalse(manager.allowed);self.assertFalse(manager.seen_images)
    def test_no_legacy_work_packet_and_stale_version(self):
        manager=self.attach([publish(0)]);manager.run('prepare','prepare')
        with self.assertRaisesRegex(ValueError,'obsolete'):manager._publish({'expected_version':1,'answer':'x','working':[]})
        with self.assertRaisesRegex(ValueError,'Stale'):manager._publish({'expected_version':0,'answer':'x'})
    def test_secondary_case_can_form_knowledge_without_claiming_raw_read(self):
        patch={'op':'new','kind':'body','module':'geometry','reading_role':'intrinsic_candidate','observed_entity':'robot_part',
            'observable':'relative_geometry','text':'A measured gripper-to-marker relation under these conditions.',
            'conditions':{},'reference_frame':'world','epistemic':'hypothesis','evidence':[self.post['id']]}
        case=self.store.submit(patch,allowed_sources=self.session.allowed,cutoff=self.session.cutoff,dependencies=self.deps,episode='episode')
        self.assertTrue(case['accepted']);library.organize_archive(self.store)
        op={'op':'add','domain':'body','family':'geometry','text':'A narrow candidate relation for matching this body.',
            'question':'Which relation should be checked?', 'use':'Compare the source-scoped relation before use.',
            'strength':'hypothesis','conditions':{},'refs':[{'type':'case','id':case['claim_id'],'relation':'support','reason':'Supports this narrow candidate.'}]}
        manager=self.attach([publish(0)]);manager.run('prepare','prepare');manager.purpose='review'
        result=manager._search({'question':'measured gripper marker','level':'case','case_refs':[{'id':case['claim_id'],'revision':case['revision']}]})
        manager._admit(result)
        self.assertFalse(manager.allowed)
        result,_=manager._publish({'expected_version':1,'answer':'Candidate only','knowledge_ops':[op]})
        self.assertEqual(len(result['knowledge_updates']),1)
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM reusable_knowledge').fetchone()[0],1)
    def test_processing_failure_rolls_back_knowledge(self):
        manager=self.attach([publish(0)]);manager.run('prepare','prepare');manager.purpose='review'
        manager.allowed={self.prior['id']}
        op={'op':'add','domain':'body','family':'frame_map','text':'Known body config relation','strength':'configuration',
            'question':'Which body frame applies?', 'use':'Compute the declared frame relation.',
            'conditions':{},'refs':[{'type':'evidence','id':self.prior['id']}]}
        with self.assertRaises(ValueError):manager._publish({'expected_version':1,'answer':'x','knowledge_ops':[op],
            'processed':[{'receipt_id':self.receipt['id'],'disposition':'knowledge'}]})
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM reusable_knowledge').fetchone()[0],0)
    def test_direct_default_read_does_not_substitute_cases_or_call_manager(self):
        manager=self.attach([])
        answer=json.loads(self.session.query_experience({'question':'gripper'}))
        self.assertEqual(answer['records'],[]);self.assertEqual(answer['level'],'knowledge')
        self.assertFalse(self.client.inputs)
    def add_body_case(self,source=None,**extra):
        proposal={'op':'new','kind':'body','module':'geometry','reading_role':'intrinsic_candidate',
            'observed_entity':'robot_part','observable':'relative_geometry','text':'Measured gripper marker geometry.',
            'conditions':{},'reference_frame':'world','epistemic':'hypothesis','evidence':[(source or self.post)['id']],**extra}
        result=self.store.submit(proposal,allowed_sources=self.session.allowed,cutoff=self.session.cutoff,
            dependencies=self.deps,episode=self.session.episode)
        self.assertTrue(result['accepted'],result);library.organize_archive(self.store)
        return result
    def test_unknown_role_key_or_value_is_navigation_error_without_lexical_search(self):
        self.add_body_case()
        for roles,reason in [({'object':'T-shaped block'},'unknown_role_key'),
                             ({'entity':'invented identity'},'unknown_role_value')]:
            with mock_patch.object(self.store,'read',side_effect=AssertionError('Invalid filters reached lexical search')):
                result=json.loads(self.session.query_experience({'question':'gripper marker','level':'case',
                    'domain':'body','family':'geometry','roles':roles}))
            self.assertEqual(result['status'],'invalid_navigation')
            self.assertEqual(result['records'],[])
            self.assertEqual(result['invalid_filters'][0]['reason'],reason)
            self.assertEqual(result['navigation']['role_facets']['entity']['values'][0]['value'],'robot_part')
    def test_valid_role_filter_retains_physical_qualification(self):
        case=self.add_body_case()
        answer=json.loads(self.session.query_experience({'question':'gripper marker','level':'case',
            'domain':'body','family':'geometry','roles':{'entity':'robot_part'}}))
        self.assertEqual(answer['status'],'records_found')
        self.assertEqual(answer['records'][0]['revision'],case['revision'])
        self.assertIn('applicability',answer['records'][0])
    def test_navigation_respects_case_revision_and_source_frontier(self):
        first=self.add_body_case();frontier=self.session.source_frontier()
        future=self.observation();self.add_body_case(future,observable='configuration',text='A new gripper configuration observed only after the frontier.')
        for kwargs in ({'at':first['revision']},{'at':self.store.revision,'frontier':frontier}):
            nav=library.navigation(self.store,domain='body',family='geometry',**kwargs)
            self.assertEqual([v['value'] for v in nav['role_facets']['observable']['values']],['relative_geometry'])
    def test_measurement_index_exposes_exact_refs_without_raw_content_or_read_grants(self):
        before=self.observation()
        measurement=self.session.evidence('surface_inspection',{'before_observation_id':before['id'],'raw_marker':'not index data'})
        self.session.evidence('surface_inspection',{'before_observation_id':self.before['id']})
        receipt=self.session.evidence('receipt',{'before_observation':before['id'],'executed_actions':1,'execution':'complete'})
        self.observation(after_receipt=receipt['id'])
        self.session.evidence('surface_inspection',{'before_observation_id':before['id']})
        manager=self.attach([publish(0)]);manager.run('prepare','prepare');manager.purpose='review'
        index=manager._intervals(self.session.episode)
        interval=next(i for i in index['intervals'] if i['receipt']==receipt['id'])
        self.assertEqual(interval['measurement_refs'],[{'id':measurement['id'],'kind':'surface_inspection','observation_id':before['id']}])
        self.assertNotIn('raw_marker',json.dumps(index));manager._admit(index)
        self.assertFalse(manager.allowed)
    def test_rejected_batch_names_unregistered_and_future_sources_without_partial_grants(self):
        manager=self.attach([publish(0)]);manager.run('prepare','prepare');manager.purpose='review'
        future=self.observation()
        with self.assertRaises(ValueError) as error:
            manager._evidence({'ids':[self.post['id'],'missing',future['id']]})
        self.assertIn('unregistered',str(error.exception));self.assertIn('missing',str(error.exception))
        self.assertIn('beyond_request_frontier',str(error.exception));self.assertIn(future['id'],str(error.exception))
        self.assertFalse(manager.allowed);self.assertFalse(manager.seen_images)
    def test_stop_prevents_another_manager_call(self):
        manager=self.attach([]);(self.trace.root/'STOP').write_text('stop')
        with self.assertRaises(RuntimeError):manager.run('x','review')
        self.assertFalse(self.client.inputs)

    def test_external_static_geometry_is_not_forced_into_an_interaction_case(self):
        patch={'op':'new','kind':'external','module':'scene_geometry','reading_role':'scene_observation',
            'observed_entity':'scene_object','observable':'relative_geometry','text':'A dated observed support surface.',
            'conditions':{'setting':'bench1'},'reference_frame':'world','epistemic':'visually_inferred','evidence':[self.post['id']]}
        result=self.store.submit(patch,allowed_sources=self.session.allowed,cutoff=self.session.cutoff,
            dependencies=self.deps,episode='episode')
        self.assertTrue(result['accepted'],result)
        self.assertFalse(self.store.db.execute('SELECT 1 FROM interaction_evidence WHERE revision=?',(result['revision'],)).fetchone())
