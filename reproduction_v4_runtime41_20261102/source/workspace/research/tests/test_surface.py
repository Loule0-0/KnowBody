import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from types import SimpleNamespace

from kyb.surface import inspect_surface, pad_relation
from kyb.body_visual import BodyVisualReference
from kyb.memory import MemoryStore
from kyb.session import MemorySession
from kyb.trace import RunTrace
from XPolicyLab.policy.GPT_6_Astra_Direct_EEF.inspect.deploy import _observation


class SurfaceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.path=Path(self.temp.name)/'depth.npy'
        v=np.eye(4);v[0,3]=-1
        self.payload={'images':{'cam_head':{'sha256':'rgb'}},
            'body_visual_reference':{'inner_face_windows_flange_m':{arm:{finger:{'polygon_xz_m':[[.14,-.01],[.16,-.01],[.16,.01],[.14,.01]],'plane_y_m':y,'inward_y_sign':sign} for finger,y,sign in [('finger7',.01,-1),('finger8',-.01,1)]} for arm in ('left','right')},'views':{'cam_head':{'resolution_wh':[640,480],
                'intrinsics_K':[[100,0,320],[0,100,240],[0,0,1]],'world_to_optical_ros':v.tolist()}}},
            'state':{'left_ee_pose':[0,0,0,1,0,0,0],'right_ee_pose':[1,2,0,2**-.5,0,0,2**-.5]}}
        self.save(np.full((480,640),2,dtype=np.float32))

    def save(self,depth):
        np.save(self.path,depth,allow_pickle=False)
        self.payload['depth_snapshots']={'cam_head':{'path':str(self.path),'sha256':hashlib.sha256(self.path.read_bytes()).hexdigest()}}

    def tearDown(self):self.temp.cleanup()

    def query(self):return inspect_surface(self.payload,[{'camera':'cam_head','pixel_uv':[420,240],'radius':1,'label':'visible point'}])['samples'][0]

    def test_optical_z_is_not_ray_distance_and_world_flange_frames_are_distinct(self):
        p=self.query()
        np.testing.assert_allclose(p['world_xyz_m'],[3,0,2],atol=1e-10)
        np.testing.assert_allclose(p['body_coordinates']['right']['flange_xyz_m'],[-2,-2,2],atol=1e-10)
        self.assertEqual(p['optical_z_m'],2.)

    def test_thin_object_center_is_not_replaced_by_background_median(self):
        d=np.full((480,640),3,dtype=np.float32);d[240,420]=2.;self.save(d)
        p=self.query();self.assertEqual(p['optical_z_m'],2.)
        self.assertTrue(p['boundary_or_mixed_surface_warning']);self.assertEqual(p['patch_depth_range_m'],1.)
        self.assertEqual(p['valid_patch_values'],9)

    def test_actual_key_tip_inside_pad_does_not_imply_shaft_coverage(self):
        run=Path('/data/KnowYourBody_RoboDojo/runs/20261018T144640_364604902')
        measurement=json.loads((run/'events/e000979.json').read_text())['payload']
        observation=json.loads((run/'events'/f"{measurement['before_observation_id'].split(':')[-1]}.json").read_text())['payload']
        model=BodyVisualReference(Path('/home/lzy/KnowYourBody_RoboDojo/RoboDojo/Assets/Robots/x5/X5A.urdf'),.1501)
        windows=model.inner_face_windows(observation['measured_finger_joints_m'])['right']
        margins=[min(face['minimum_projected_edge_margin_m'] for face in
            pad_relation(sample['body_coordinates']['right']['flange_xyz_m'],windows).values())
            for sample in measurement['samples']]
        self.assertAlmostEqual(margins[0],-.005518,delta=.000003)
        self.assertAlmostEqual(margins[1],.004022,delta=.000003)

    def test_pad_plane_gap_uses_each_actual_finger_and_outside_corner_is_halfplane_distance(self):
        windows=self.payload['body_visual_reference']['inner_face_windows_flange_m']['left']
        relations=pad_relation([.15,.003,0],windows)
        self.assertAlmostEqual(relations['finger7']['inward_distance_from_face_plane_m'],.007)
        self.assertAlmostEqual(relations['finger8']['inward_distance_from_face_plane_m'],.013)
        self.assertAlmostEqual(relations['finger7']['minimum_projected_edge_margin_m'],.01)
        outside=pad_relation([.17,0,.02],windows)['finger7']
        self.assertAlmostEqual(outside['minimum_projected_edge_margin_m'],-.01)

    def test_invalid_center_and_modified_snapshot_are_not_silently_substituted(self):
        for value in (0,np.inf,np.nan):
            d=np.full((480,640),2,dtype=np.float32);d[240,420]=value;self.save(d)
            p=self.query();self.assertEqual(p['status'],'NEEDS_WORK_no_valid_center_depth');self.assertIsNone(p['world_xyz_m'])
        self.path.write_bytes(b'changed')
        with self.assertRaises(ValueError):self.query()

    def test_rgb_and_depth_come_from_one_raw_capture(self):
        class Environment:
            def __init__(self):self.calls=0
            def get_obs(self):
                self.calls+=1
                return {'vision':{'cam_head':{'color':np.zeros((480,640,3),dtype=np.uint8),
                    'depth':np.full((480,640),2,dtype=np.float32)}},'state':{}}
        env=Environment();observation=_observation(env,0)
        self.assertEqual(env.calls,1);self.assertEqual(observation.extra['depth_frames']['cam_head'].dtype,np.float32)
        self.assertEqual(float(observation.extra['depth_frames']['cam_head'][0,0]),2.)

    def test_depth_claim_pins_channel_while_rgb_claim_survives_depth_change(self):
        root=Path(self.temp.name);trace=RunTrace(root/'run',{});store=MemoryStore(root/'memory.sqlite')
        deps={'body':'body','cameras':'rgb','depth':'depth1','tool':'key'}
        session=MemorySession(trace,store,task='insert_key',dependencies=deps)
        observation=session.evidence('observation',{**self.payload,'dependencies':deps})
        session.inspect_surface({'observation_id':observation['id'],
            'samples':[{'camera':'cam_head','pixel_uv':[420,240],'label':'visible face'}]},
            SimpleNamespace(extra={'evidence_id':observation['id']}))
        measurement=store.db.execute("SELECT id FROM evidence WHERE kind='surface_inspection'").fetchone()[0]
        common={'op':'new','reading_role':'intrinsic_candidate','kind':'body','module':'geometry','observed_entity':'scene_object',
            'observable':'relative_geometry','reference_frame':'world','epistemic':'visually_inferred'}
        def publish(patch):return store.submit(patch,allowed_sources=session.allowed,cutoff=session.cutoff,
            dependencies=deps,episode=session.episode,visible_revisions={})
        depth_claim=publish({**common,'text':'A model-selected visible point has a depth sample.',
            'evidence':[observation['id'],measurement]})
        self.assertTrue(depth_claim['accepted'])
        rgb_claim=publish({**common,'text':'An RGB corner is visible.','reference_frame':'image','evidence':[observation['id']]})
        self.assertTrue(rgb_claim['accepted'])
        result=store.read('visible',context={'task':'insert_key'},dependencies={**deps,'depth':'depth2'},episode=session.episode)
        self.assertEqual(len(result['selected']),2)
        by_id={c['id']:c for c in result['selected']}
        self.assertFalse(by_id[rgb_claim['claim_id']]['qualification']['changed_derivation_contracts'])
        self.assertEqual(by_id[depth_claim['claim_id']]['qualification']['changed_derivation_contracts'],['depth'])
        self.assertEqual(by_id[depth_claim['claim_id']]['qualification']['metric_qualification'],'incomparable_current_contract')
        store.close()

    def test_external_surface_sources_require_same_versions_and_interaction_endpoint(self):
        root=Path(self.temp.name);trace=RunTrace(root/'run',{});store=MemoryStore(root/'memory.sqlite')
        deps={'body':'body','cameras':'rgb','depth':'depth1','tool':'key'}
        session=MemorySession(trace,store,task='insert_key',dependencies=deps)
        before=session.evidence('observation',{**self.payload,'dependencies':deps})
        receipt=session.evidence('receipt',{'before_observation':before['id'],'executed_actions':1,
            'execution':'complete','unknown_command':None,'dependencies':deps})
        post=session.evidence('observation',{**self.payload,'dependencies':deps,'after_receipt':receipt['id']})
        foreign=session.evidence('observation',{**self.payload,'dependencies':deps})
        def publish(measurement,additional=()):
            patch={'op':'new','reading_role':'interaction','kind':'external','relation_type':'observed_interaction','module':'interaction',
                'observed_entity':'scene_object','observable':'interaction_effect','reference_frame':'world',
                'epistemic':'visually_inferred','text':'Synthetic observed surface response.',
                'evidence':[receipt['id'],post['id'],measurement['id'],*additional]}
            return store.submit(patch,allowed_sources=session.allowed,cutoff=session.cutoff,
                dependencies=deps,episode=session.episode,visible_revisions={})
        wrong_version=session.evidence('surface_inspection',{'before_observation_id':post['id'],
            'dependencies':{**deps,'depth':'depth2'}})
        result=publish(wrong_version);self.assertFalse(result['accepted'])
        self.assertIn('matching_geometry_and_depth_versions',result['reason'])
        wrong_endpoint=session.evidence('surface_inspection',{'before_observation_id':foreign['id'],'dependencies':deps})
        result=publish(wrong_endpoint,[foreign['id']]);self.assertFalse(result['accepted'])
        self.assertIn('selected_interaction_endpoint',result['reason'])
        correct=session.evidence('surface_inspection',{'before_observation_id':post['id'],'dependencies':deps})
        self.assertTrue(publish(correct)['accepted'])
        store.close()
