import copy
import unittest
from kyb.context import surface_readout

class SurfaceContextTests(unittest.TestCase):
    def test_depth_patch_archive_preserved_and_only_requested_geometry_returned(self):
        original={'samples':[{'camera':'head','label':'selected surface','world_xyz_m':[.123456789,.2,.78],
            'patch_depth_m':[[1,2,3]],'patch_depth_range_m':.003,'status':'PASS_sensor_sample_not_identity',
            'body_coordinates':{'left':{'relative_to_named_grasp_in_flange_axes_m':[.01,.02,.03],
                'distal_pad_relation':{'finger7':{'minimum_projected_edge_margin_m':-.004,
                'inward_distance_from_face_plane_m':.02,'projected_edge_margins_m':[1,2,3,4]}}},'right':{'unused':True}}}]}
        saved=copy.deepcopy(original);result=surface_readout(original,'left')
        self.assertEqual(original,saved)
        self.assertNotIn('patch_depth_m',result['samples'][0]);self.assertNotIn('right',str(result))
        self.assertEqual(result['samples'][0]['acting_frame']['inner_pad']['finger7']['minimum_projected_edge_margin_m'],-.004)
        self.assertEqual(result['samples'][0]['world_xyz_m'][0],.123457)
        self.assertIn('not certified',result['qualification'])
