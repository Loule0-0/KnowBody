import unittest
from pathlib import Path

import numpy as np

from kyb.body_visual import BodyVisualReference, project, render_intrinsics

URDF = Path('/home/lzy/KnowYourBody_RoboDojo/RoboDojo/Assets/Robots/x5/X5A.urdf')


class BodyVisualTests(unittest.TestCase):
    def test_render_contract_rejects_unverified_projection_and_lens(self):
        args = dict(lens_model='pinhole', projection_mode='perspective', aperture_offsets=[0, 0])
        k = render_intrinsics((640, 480), 10, 22.212, **args)
        self.assertEqual(k[0, 0], k[1, 1])
        np.testing.assert_allclose(k[:2, 2], [319.5, 239.5])
        for changed in ({'lens_model':'opencvPinhole'}, {'projection_mode':'orthographic'},
                        {'aperture_offsets':[.01,0]}, {'aperture_offsets':[np.nan,0]}):
            with self.assertRaises(ValueError):
                render_intrinsics((640, 480), 10, 22.212, **{**args, **changed})

    def test_real_stored_table_and_pixel_roundtrip_exposes_sdk_contract_error(self):
        import json
        from kyb.surface import inspect_surface
        p = Path('/data/KnowYourBody_RoboDojo/runs/20261018T150052_126803907/events/e002514.json')
        payload = json.loads(p.read_text())['payload']
        pixels = [[480,260],[490,280],[500,300],[490,240],[470,245],[450,280]]
        samples = [{'camera':'cam_head','pixel_uv':uv,'radius':0} for uv in pixels]
        old = inspect_surface(payload, samples)['samples']
        self.assertGreater(max(abs(s['world_xyz_m'][2]-.765) for s in old), .008)
        view = payload['body_visual_reference']['views']['cam_head']
        # Runtime USD exposes these public configured physical parameters, not table-fit K.
        view['intrinsics_K'] = render_intrinsics((640,480),10,22.212,
            lens_model='pinhole',projection_mode='perspective',aperture_offsets=[0,0]).tolist()
        fixed = inspect_surface(payload, samples)['samples']
        self.assertLess(max(abs(s['world_xyz_m'][2]-.765) for s in fixed), .0000002)
        reconstructed = project([s['world_xyz_m'] for s in fixed],view['intrinsics_K'],
                                view['world_to_optical_ros'],view['resolution_wh'])
        self.assertEqual([s['pixel_uv'] for s in reconstructed],pixels)

    def test_projection_handles_non_square_intrinsics_depth_and_image_bounds(self):
        k = [[400,0,320],[0,300,240],[0,0,1]]
        result = project([[.1,.2,1],[0,0,-1],[2,0,1]],k,np.eye(4),(640,480))
        self.assertEqual(result[0]['pixel_uv'],[360.,300.])
        self.assertTrue(result[0]['in_image'])
        self.assertIsNone(result[1]['pixel_uv'])
        self.assertFalse(result[2]['in_image'])
        with self.assertRaises(ValueError):
            project([[np.nan,0,1]],k,np.eye(4),(640,480))

    def test_public_distal_faces_follow_actual_fingers_and_flange_rotation(self):
        model = BodyVisualReference(URDF,.1501)
        state = {f'{a}_ee_pose':[0,0,1,1,0,0,0] for a in ('left','right')}
        body = model.geometry(state,{'left':[.01,.02],'right':[0,0]})
        self.assertAlmostEqual(body['left']['finger7_inner_face_centroid'][1]-.01,
            body['right']['finger7_inner_face_centroid'][1])
        self.assertAlmostEqual(body['left']['finger8_inner_face_centroid'][1]+.02,
            body['right']['finger8_inner_face_centroid'][1])
        for _,_,face,_ in model.fingers:
            area = np.linalg.norm(np.cross(face[:,1]-face[:,0],face[:,2]-face[:,0]),axis=1).sum()/2
            self.assertAlmostEqual(area,.000212459419,places=10)
        state['left_ee_pose']=[0,0,1,2**-.5,0,0,2**-.5]
        rotated=model.geometry(state,{'left':[.01,.02],'right':[0,0]})['left']['named_grasp']
        np.testing.assert_allclose(rotated,[0,.1501,1],atol=1e-12)

    def test_rigidly_shared_camera_body_translation_preserves_own_projection(self):
        k = [[400,0,320],[0,300,240],[0,0,1]]
        p=np.array([.1,.2,1]);v=np.eye(4);delta=np.array([.3,-.1,.4])
        before=project(p,k,v,(640,480))[0]['pixel_uv']
        v[:3,3]=-delta
        self.assertEqual(before,project(p+delta,k,v,(640,480))[0]['pixel_uv'])
        self.assertNotEqual(before,project(p,k,v,(640,480))[0]['pixel_uv'])
