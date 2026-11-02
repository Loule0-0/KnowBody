import unittest
from types import SimpleNamespace
import numpy as np

from kyb.body_guard import synchronized_samples, closest_cross_arm_pair


class BodyGuardTests(unittest.TestCase):
    def test_synchronized_interpolation_detects_midsegment_crossing(self):
        current={'left_arm_joint_state':[-1.,0,0,0,0,0],
                 'right_arm_joint_state':[1.,0,0,0,0,0]}
        action=SimpleNamespace(data={'left_arm_joint_state':[1.,0,0,0,0,0],
                                     'right_arm_joint_state':[-1.,0,0,0,0,0]})
        dense,mapping=synchronized_samples([action],current,step_rad=.25)
        def spheres(q):
            a=np.zeros((len(q),1,4));a[:,0,0]=q[:,0];a[:,0,3]=.1;return a
        left,right=spheres(dense['left']),spheres(dense['right'])
        self.assertGreater(closest_cross_arm_pair(left[[0,-1]],right[[0,-1]])[0],0)
        gap,index=closest_cross_arm_pair(left,right)
        self.assertAlmostEqual(gap,-.2)
        self.assertEqual(mapping[index[0]],{'command_index':0,'alpha':.5})
        np.testing.assert_allclose(dense['left'][:,0],-dense['right'][:,0])

    def test_invalid_active_geometry_fails_instead_of_being_ignored(self):
        valid=np.array([[[0.,0,0,.1]]]);invalid=valid.copy();invalid[0,0,0]=np.nan
        with self.assertRaises(ValueError):closest_cross_arm_pair(valid,invalid)
        disabled=np.array([[[np.nan,0,0,0],[1.,0,0,.1]]])
        self.assertAlmostEqual(closest_cross_arm_pair(valid,disabled)[0],.8)


if __name__=='__main__':unittest.main()
