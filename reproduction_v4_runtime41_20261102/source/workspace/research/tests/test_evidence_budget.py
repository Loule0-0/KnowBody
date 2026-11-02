"""Whole evidence delivery, exact raw pages and transaction authority (no model)."""
import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import test_manager
from kyb import readout
from kyb.manager import MemoryManager
from kyb.perception import RGB_ONLY


class EvidenceBudgetTests(unittest.TestCase):
    setUp=test_manager.ManagerTests.setUp
    tearDown=test_manager.ManagerTests.tearDown
    observation=test_manager.ManagerTests.observation
    attach=test_manager.ManagerTests.attach

    def manager(self):
        m=MemoryManager(self.session,None);m.frontier=self.session.source_frontier();m.purpose='review'
        m.allowed=set();m.seen_images=set();m.seen_telemetry=set();m.sent_contracts=set()
        m.visible_revisions={};m.visible_knowledge={};m.selectable_knowledge={}
        return m

    def state(self,m):
        return copy.deepcopy((m.allowed,m.seen_images,m.seen_telemetry,m.sent_contracts))

    def image_observation(self,**extra):
        path=Path(self.temp.name)/'rgb.jpg';path.write_bytes(b'archived RGB bytes')
        ref={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        return self.observation(images={k:ref for k in ('cam_head','cam_left_wrist','cam_right_wrist')},**extra)

    def test_small_response_matches_existing_shape(self):
        m=self.manager();ids=[self.before['id'],self.receipt['id'],self.post['id']]
        original={'records':[m._record(i,body='ee',delivered_ids=ids) for i in ids],'delivered_sources':ids}
        result,parts=m._evidence({'receipts':[self.receipt['id']],'body':'ee'})
        self.assertEqual(result,original);self.assertEqual(parts,[]);self.assertFalse(m.allowed)
        m._admit(result);self.assertEqual(m.allowed,set(ids))

    def test_direct_image_bound_returns_whole_records_and_remaining(self):
        obs=[self.image_observation() for _ in range(4)];m=self.manager()
        args={'ids':[x['id'] for x in obs],'cameras':['cam_head','cam_left_wrist'],'body':'ee'}
        result,parts=m._evidence(args)
        self.assertEqual([r['id'] for r in result['records']],args['ids'][:3])
        self.assertEqual(sum(p['type']=='image_url' for p in parts),6)
        self.assertEqual(result['remaining_selectors'],{**args,'ids':args['ids'][3:]})
        self.assertNotIn(obs[-1]['id'],m.seen_images);self.assertFalse(m.allowed)
        m._admit(result);self.assertEqual(m.allowed,set(args['ids'][:3]))
        last,_=m._evidence(result['remaining_selectors']);m._admit(last)
        self.assertEqual(m.allowed,set(args['ids']));self.assertLessEqual(readout.size(result),24576)

    def test_direct_text_prefix_does_not_fragment_small_next_record(self):
        refs=[self.session.evidence('policy_decision',{'note':'x'*13000}) for _ in range(3)];m=self.manager()
        result,_=m._evidence({'ids':[r['id'] for r in refs]})
        self.assertEqual(len(result['records']),1)
        self.assertNotIn('view',result['records'][0])
        self.assertEqual(result['remaining_selectors']['ids'],[r['id'] for r in refs[1:]])
        self.assertLessEqual(readout.size(result),24576)

    def test_full_group_prefix_is_atomic(self):
        rec2=self.session.evidence('receipt',{'before_observation':self.post['id'],'executed_actions':1})
        post2=self.observation(after_receipt=rec2['id'])
        measure=self.session.evidence('surface_inspection',{'before_observation_id':self.post['id']})
        m=self.manager();old=m._record
        def render(source,**kw):
            r=old(source,**kw)
            if r['kind']=='receipt':r['padding']='r'*7000
            if r['kind']=='surface_inspection':r['padding']='m'*10000
            return r
        with patch.object(m,'_record',side_effect=render):
            result,_=m._evidence({'receipts':[self.receipt['id'],rec2['id']], 'measurements':[measure['id']], 'body':'ee'})
            expected=[self.before['id'],self.receipt['id'],self.post['id'],rec2['id'],post2['id']]
            self.assertEqual(result['delivered_sources'],expected)
            self.assertEqual(result['remaining_selectors'],{'measurements':[measure['id']],'body':'ee','cameras':[]})
            self.assertNotIn(measure['id'],str(result['records']));self.assertFalse(m.allowed)
            m._admit(result);self.assertEqual(m.allowed,set(expected))

    def test_single_plan_pages_losslessly_and_authorizes_only_at_end(self):
        source=self.session.evidence('validated_action_plan',{'actions':[{'q':[i/1000]*6,'note':'逐命令'} for i in range(600)]})
        m=self.manager();original=m._record(source['id']);result,_=m._evidence({'ids':[source['id']]})
        fragments=[];pages=0
        while True:
            self.assertLessEqual(readout.size(result),24576)
            record=result['records'][0];self.assertEqual(record['layer'],'evidence');self.assertFalse(record['complete_record'])
            self.assertEqual(record['view'],'detail_page');fragments.append(record['fragment']);pages+=1
            m._admit(result);token=record['continuation']
            if not token:break
            self.assertNotIn(source['id'],m.allowed);self.assertFalse(m.seen_telemetry)
            result,_=m._evidence({'continuation':token})
        reconstructed=json.loads(''.join(fragments))
        self.assertEqual(reconstructed,{**original,'layer':'evidence','revision':source['sequence'],'view':'detail','complete_record':True})
        self.assertEqual(hashlib.sha256(''.join(fragments).encode()).hexdigest(),record['content_sha256'])
        self.assertGreater(pages,1);self.assertIn(source['id'],m.allowed)

    def test_observation_page_images_contract_and_telemetry_have_separate_delivery(self):
        canonical={'coordinate_contract':{'id':'coord','flange_to_action_translation_m':[0,0,0]},
            'arms':{'left':{'action_pose_world_m_wxyz':[0,0,0,1,0,0,0],'action_dimensions':{}}}}
        obs=self.image_observation(canonical_state=canonical,body_visual_reference={'large':'x'*50000})
        m=self.manager();result,parts=m._evidence({'ids':[obs['id']],'body':'full','cameras':['cam_head']})
        self.assertEqual(sum(p['type']=='image_url' for p in parts),1)
        self.assertEqual(m.seen_images,{obs['id']});self.assertEqual(m.sent_contracts,{'coord'})
        m._admit(result);self.assertFalse(m.seen_telemetry);self.assertFalse(m.allowed)
        while result['records'][0]['continuation']:
            result,parts=m._evidence({'continuation':result['records'][0]['continuation']})
            self.assertEqual(parts,[]);self.assertNotIn('coordinate_contracts',result)
            m._admit(result)
        self.assertEqual(m.allowed,{obs['id']});self.assertEqual(m.seen_telemetry,{obs['id']})

    def test_direct_receipt_cannot_measure_an_unselected_post(self):
        m=self.manager();old=m._record
        def render(source,**kw):
            r=old(source,**kw)
            if source==self.post['id']:r['large']='p'*23000
            return r
        with patch.object(m,'_record',side_effect=render):
            result,_=m._evidence({'ids':[self.before['id'],self.receipt['id'],self.post['id']],'body':'ee'})
        self.assertEqual([r['id'] for r in result['records']],[self.before['id'],self.receipt['id']])
        self.assertIsNone(result['records'][1]['data']['action_digest']['measured'])
        self.assertIn('measurement_read_required',result['records'][1]['data']['action_digest'])
        self.assertNotIn(self.post['id'],m.seen_telemetry)

    def test_invalid_future_forbidden_and_prepare_tail_reject_before_partial(self):
        good=self.session.evidence('policy_decision',{'note':'x'*20000});m=self.manager()
        for bad in ['unregistered',self.observation()['id']]:
            before=self.state(m)
            with self.assertRaises(ValueError):m._evidence({'ids':[good['id'],bad]})
            self.assertEqual(self.state(m),before)
        m.frontier=self.session.source_frontier();m.purpose='prepare'
        with self.assertRaisesRegex(ValueError,'Preparation cannot'):m._evidence({'ids':[self.prior['id'],self.post['id']]})
        m.purpose='review';forbidden=self.session.evidence('surface_inspection',{'before_observation_id':self.post['id']})
        m.frontier=self.session.source_frontier();self.session.perception=RGB_ONLY
        with self.assertRaisesRegex(ValueError,'RGB-only'):m._evidence({'ids':[good['id'],forbidden['id']]})
        self.assertEqual(self.state(m),(set(),set(),set(),set()))

    def test_continuation_owner_frontier_profile_replacement_and_parameters(self):
        source=self.session.evidence('validated_action_plan',{'actions':'x'*50000});m=self.manager()
        result,_=m._evidence({'ids':[source['id']]});token=result['records'][0]['continuation']
        before=self.state(m)
        other=self.manager()
        with self.assertRaisesRegex(ValueError,'Unknown raw'):other._evidence({'continuation':token})
        for extra in ({'body':'none'},{'ids':[source['id']]},{'cameras':[]}):
            with self.assertRaises(ValueError):m._evidence({'continuation':token,**extra})
        frontier=m.frontier.copy();m.frontier={}
        with self.assertRaises(ValueError):m._evidence({'continuation':token})
        m.frontier={**frontier,self.session.episode:frontier[self.session.episode]+1}
        with self.assertRaisesRegex(ValueError,'scope changed'):m._evidence({'continuation':token})
        m.frontier=frontier;profile=self.session.perception;self.session.perception=RGB_ONLY
        with self.assertRaisesRegex(ValueError,'scope changed'):m._evidence({'continuation':token})
        self.session.perception=profile
        with self.store.db:self.store.db.execute('UPDATE evidence SET payload=? WHERE id=?',(json.dumps({'actions':'new'}),source['id']))
        with self.assertRaisesRegex(ValueError,'source changed'):m._evidence({'continuation':token})
        self.assertEqual(self.state(m),before)

    def test_manager_tokens_not_shared_with_session_or_another_owner(self):
        source=self.session.evidence('validated_action_plan',{'actions':'x'*50000});m=self.manager();other=self.manager()
        a,_=m._evidence({'ids':[source['id']]});b,_=other._evidence({'ids':[source['id']]})
        ta=a['records'][0]['continuation'];tb=b['records'][0]['continuation']
        self.assertNotEqual(ta,tb);self.assertNotIn(ta,getattr(self.session,'_read_continuations',{}))
        with self.assertRaises(ValueError):other._evidence({'continuation':ta})

    def test_skipped_raw_fragment_never_grants_permission(self):
        source=self.session.evidence('validated_action_plan',{'actions':'x'*50000});m=self.manager()
        first,_=m._evidence({'ids':[source['id']]});result=first
        while result['records'][0]['continuation']:
            result,_=m._evidence({'continuation':result['records'][0]['continuation']})
            m._admit(result)
        self.assertNotIn(source['id'],m.allowed)
        m._admit(first)
        self.assertIn(source['id'],m.allowed)

    def test_invalid_image_in_unselected_tail_rejects_all(self):
        a=self.image_observation();bad=self.observation(images={})
        m=self.manager();state=self.state(m)
        with self.assertRaisesRegex(ValueError,'Unknown camera'):
            m._evidence({'ids':[a['id'],bad['id']],'cameras':['cam_head']})
        self.assertEqual(self.state(m),state)

    def test_longer_complete_prefix_can_fit_after_remaining_metadata_disappears(self):
        a=self.session.evidence('policy_decision',{'note':'a'})
        b=self.session.evidence('policy_decision',{'note':'b'})
        m=self.manager();old=m._record;ra=old(a['id']);rb=old(b['id'])
        padding='x'*(24570-readout.size({'records':[dict(ra,padding=''),rb]}))
        cameras=['cam_head','cam_left_wrist','cam_right_wrist']
        first={'records':[dict(ra,padding=padding)],'remaining_selectors':{'ids':[b['id']],'cameras':cameras,'body':'full'},'complete_request':False}
        self.assertGreater(readout.size(first),24576)
        def render(identifier,**kwargs):
            r=old(identifier,**kwargs)
            if identifier==a['id']:r['padding']=padding
            return r
        with patch.object(m,'_record',side_effect=render):
            result,_=m._evidence({'ids':[a['id'],b['id']],'cameras':cameras,'body':'full'})
        self.assertEqual(len(result['records']),2);self.assertLessEqual(readout.size(result),24576)
        self.assertNotIn('remaining_selectors',result)
        self.assertFalse(getattr(m,'_evidence_page_context',{}))

    def test_record_that_fits_alone_is_not_paged_for_tail_metadata(self):
        source=self.session.evidence('policy_decision',{'note':'x'})
        tail=self.session.evidence('policy_decision',{'note':'tail'});m=self.manager();old=m._record
        record=old(source['id']);overhead=readout.size({'records':[dict(record,padding='')]})
        padding='x'*(24550-overhead)
        def render(identifier,**kwargs):
            r=old(identifier,**kwargs)
            if identifier==source['id']:r['padding']=padding
            return r
        with patch.object(m,'_record',side_effect=render):
            with self.assertRaisesRegex(ValueError,'fits alone'):
                m._evidence({'ids':[source['id'],tail['id']]})
            alone,_=m._evidence({'ids':[source['id']]})
        self.assertNotIn('view',alone['records'][0]);self.assertLessEqual(readout.size(alone),24576)
        self.assertFalse(getattr(m,'_evidence_page_context',{}))

    def test_oversize_group_rejects_without_any_grants(self):
        before=self.observation(body_visual_reference={'large':'x'*30000})
        receipt=self.session.evidence('receipt',{'before_observation':before['id'],'executed_actions':1})
        self.observation(after_receipt=receipt['id']);m=self.manager();state=self.state(m)
        with self.assertRaisesRegex(ValueError,'expanded exact IDs'):
            m._evidence({'receipts':[receipt['id']],'body':'full'})
        self.assertEqual(self.state(m),state)

if __name__=='__main__':unittest.main()
