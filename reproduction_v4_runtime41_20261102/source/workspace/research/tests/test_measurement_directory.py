"""Discover independent sensor records without certifying labels or reading evidence."""
import json,unittest
from pathlib import Path
import test_manager
from kyb import episode_index
from kyb.perception import RGB_ONLY
from kyb.readout import size
from kyb.trace import RunTrace


class MeasurementDirectory(unittest.TestCase):
    setUp=test_manager.ManagerTests.setUp
    tearDown=test_manager.ManagerTests.tearDown
    observation=test_manager.ManagerTests.observation
    attach=test_manager.ManagerTests.attach

    def sample(self,label='unverified aperture',**changes):
        data={'before_observation_id':self.post['id'],'samples':[
            {'label':label,'camera':'cam_head','world_xyz_m':[987.654,2,3],
             'sampled_pixel_uv':[400,300],'body_coordinates':{'private_numeric_detail':1}}]}
        data.update(changes)
        return self.session.evidence('surface_inspection',data)

    def test_final_measurement_is_discoverable_without_any_following_action(self):
        record=self.sample()
        page=episode_index.directory(self.store,self.session.episode,self.post['sequence'],record['sequence'],'measurements')
        self.assertEqual(page['entries'],[{'id':record['id'],'sequence':record['sequence'],
            'observation_id':self.post['id'],'samples':[{'label':'unverified aperture','camera':'cam_head'}]}])
        self.assertIsNone(page['next_after'])
        for forbidden in ['world_xyz_m','sampled_pixel_uv','body_coordinates','987.654']:
            self.assertNotIn(forbidden,json.dumps(page))
        self.assertIn('not certified object identity',page['qualification'])
        m=self.attach([test_manager.publish(0)]);m.run('Prepare','prepare');m.purpose='review'
        self.assertEqual(m._intervals(self.session.episode,after=self.post['sequence'],through=record['sequence'])['intervals'],[])

    def test_measurement_before_the_next_post_does_not_relax_interval_bounds(self):
        record=self.sample()
        receipt=self.session.evidence('receipt',{'before_observation':self.post['id'],'executed_actions':1,'execution':'complete'})
        post=self.observation(after_receipt=receipt['id'])
        m=self.attach([test_manager.publish(0)]);m.run('Prepare','prepare');m.purpose='review'
        self.assertFalse(m._intervals(self.session.episode,after=self.post['sequence'],through=receipt['sequence'])['intervals'])
        page=episode_index.directory(self.store,self.session.episode,self.post['sequence'],receipt['sequence'],'measurements')
        self.assertEqual([e['id'] for e in page['entries']],[record['id']])
        self.assertEqual(m._intervals(self.session.episode,after=self.post['sequence'],through=post['sequence'])['intervals'][0]['receipt'],receipt['id'])

    def test_complete_utf8_entries_page_once_with_original_labels_and_camera_order(self):
        records=[self.sample('孔口🙂 '+str(i),samples=[{'label':'孔口🙂 '+str(i),'camera':'cam_left_wrist'},
            {'label':'unconfirmed white surface','camera':'cam_head'}]) for i in range(15)]
        cursor=self.post['sequence'];end=records[-1]['sequence'];seen=[];pages=0
        while cursor is not None:
            page=episode_index.directory(self.store,self.session.episode,cursor,end,'measurements',budget_bytes=1100)
            self.assertLessEqual(size(page),1100);self.assertTrue(page['entries'])
            seen.extend(page['entries']);cursor=page['next_after'];pages+=1
        self.assertGreater(pages,1)
        self.assertEqual([e['id'] for e in seen],[r['id'] for r in records])
        self.assertEqual([e['samples'][0]['label'] for e in seen],['孔口🙂 '+str(i) for i in range(15)])
        self.assertTrue(all([s['camera'] for s in e['samples']]==['cam_left_wrist','cam_head'] for e in seen))
        self.assertNotIn(records[-1]['id'],json.dumps(episode_index.directory(
            self.store,self.session.episode,self.post['sequence'],records[-2]['sequence'],'measurements')))

    def test_oversized_sample_and_invalid_observation_fail_explicitly(self):
        record=self.sample('界'*2000)
        with self.assertRaisesRegex(ValueError,'one complete'):
            episode_index.directory(self.store,self.session.episode,self.post['sequence'],record['sequence'],'measurements',budget_bytes=1100)
        bad=self.sample(before_observation_id='not_registered')
        with self.assertRaisesRegex(ValueError,'registered observation'):
            episode_index.directory(self.store,self.session.episode,record['sequence'],bad['sequence'],'measurements')
        wrong=self.session.evidence('surface_inspection',{'before_observation_id':record['id'],'samples':[]})
        with self.assertRaisesRegex(ValueError,'registered observation'):
            episode_index.directory(self.store,self.session.episode,bad['sequence'],wrong['sequence'],'measurements')

    def test_foreign_or_future_observation_cannot_be_presented_as_a_visible_source(self):
        trace=RunTrace(Path(self.temp.name)/'foreign',{})
        data={'state':{},'dependencies':dict(self.deps)}
        foreign=trace.event('observation',data);self.store.register(foreign,data)
        record=self.sample(before_observation_id=foreign['id'])
        with self.assertRaisesRegex(ValueError,'same visible run'):
            episode_index.directory(self.store,self.session.episode,self.post['sequence'],record['sequence'],'measurements')
        after=record['sequence'];future_id=self.session.episode+f':e{self.trace.sequence+2:06d}'
        future_link=self.sample(before_observation_id=future_id)
        observation=self.observation();self.assertEqual(observation['id'],future_id)
        with self.assertRaisesRegex(ValueError,'earlier registered observation'):
            episode_index.directory(self.store,self.session.episode,after,observation['sequence'],'measurements')

    def test_original_empty_label_is_not_filled_or_treated_as_object_identity(self):
        record=self.sample('')
        page=episode_index.directory(self.store,self.session.episode,self.post['sequence'],record['sequence'],'measurements')
        self.assertEqual(page['entries'][0]['samples'],[{'label':'','camera':'cam_head'}])

    def test_navigation_grants_no_evidence_or_processed_permission(self):
        record=self.sample();m=self.attach([test_manager.publish(0)]);m.run('Prepare','prepare');m.purpose='review'
        def permissions():return (set(m.allowed),set(m.seen_images),set(m.seen_telemetry),dict(m.visible_revisions),dict(m.visible_knowledge))
        before=permissions();page=episode_index.directory(self.store,self.session.episode,self.post['sequence'],record['sequence'],'measurements')
        m._admit(page);self.assertEqual(permissions(),before)
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM processed_intervals').fetchone()[0],0)
        result,_=m._evidence({'measurements':[record['id']],'cameras':[],'body':'none'})
        self.assertEqual(permissions(),before)
        m._admit(result);self.assertIn(record['id'],m.allowed)

    def test_real_manager_dispatch_keeps_prepare_frontier_and_rgb_guards(self):
        record=self.sample();args={'run':self.session.episode,'after':self.post['sequence'],
            'through':record['sequence'],'directory':'measurements'}
        m=self.attach([('read_trajectory',args),test_manager.publish(0),
            ('read_trajectory',args),('read_trajectory',{**args,'through':1000000}),test_manager.publish(1),
            ('read_trajectory',args),test_manager.publish(2)])
        m.run('Prepare','prepare');m.run('Locate old samples','review')
        self.session.perception=RGB_ONLY;m.run('RGB cannot read depth-derived samples','review')
        results=[]
        for line in (self.trace.root/'events.jsonl').read_text().splitlines():
            ref=json.loads(line)
            if ref['kind']=='manager_tool_result':
                p=json.loads(Path(ref['path']).read_text())['payload']
                if p['tool']=='read_trajectory':results.append(p['result'])
        self.assertIn('Preparation reads historical',results[0]['reason'])
        self.assertEqual([e['id'] for e in results[1]['entries']],[record['id']])
        self.assertIn('bounds',results[2]['reason'])
        self.assertIn('RGB-only',results[3]['reason'])
        self.assertNotIn('unverified aperture',json.dumps(results[3]))
        self.assertFalse(m.allowed);self.assertFalse(m.seen_images)
