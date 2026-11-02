import unittest
import numpy as np
from kyb.geometry_relations import planar_correspondence

class CorrespondenceTests(unittest.TestCase):
    def fit(self,a,b):
        samples=[{'label':prefix+str(i),'world_xyz_m':[*point,0]} for prefix,points in [('a',a),('b',b)] for i,point in enumerate(points)]
        pairs=[{'source':'a'+str(i),'target':'b'+str(i)} for i in range(len(a))]
        return planar_correspondence(samples,pairs)
    def test_rotation_and_selected_center_are_separate(self):
        a=np.array([[0,0],[1,0],[0,1]])
        b=a@np.array([[0,1],[-1,0]])+[2,3]
        result=self.fit(a,b)
        self.assertAlmostEqual(result['rotation_deg_ccw_about_world_positive_z'],90)
        self.assertLess(result['fit_rms_m'],1e-12)
        np.testing.assert_allclose(result['landmark_center_delta_xy_m'],b.mean(0)-a.mean(0))
    def test_reversed_identity_can_have_zero_residual(self):
        result=self.fit([[0,0],[1,0]],[[1,0],[0,0]])
        self.assertAlmostEqual(abs(result['rotation_deg_ccw_about_world_positive_z']),180)
        self.assertLess(result['fit_rms_m'],1e-12)
        self.assertIn('does not certify identity',result['qualification'])
    def test_scale_mismatch_has_residual(self):self.assertGreater(self.fit([[0,0],[1,0]],[[0,0],[2,0]])['fit_rms_m'],.4)
    def test_one_wrong_third_correspondence_increases_residual(self):
        self.assertGreater(self.fit([[0,0],[1,0],[0,1]],[[0,0],[1,0],[2,2]])['fit_rms_m'],.2)
    def test_duplicate_and_degenerate_points_rejected(self):
        with self.assertRaises(ValueError):self.fit([[0,0],[0,0]],[[1,0],[2,0]])
        with self.assertRaises(ValueError):planar_correspondence([{'label':'a','world_xyz_m':[0,0,0]}]*4,[{'source':'a','target':'a'}]*2)
