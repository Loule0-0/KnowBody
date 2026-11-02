"""The public parameter description matches the unchanged pose composition."""
import unittest
import numpy as np

from kyb.policy import EefAgentPolicy
from kyb.perception import select
from XPolicyLab.policy.GPT_6_Astra_Direct_EEF import pose
import test_policy_pipeline as fixture


class AngleContract(unittest.TestCase):
    setUp=fixture.PolicyPipelineTests.setUp
    tearDown=fixture.PolicyPipelineTests.tearDown

    def test_real_policy_tools_clarify_absolute_composition_in_both_profiles(self):
        for profile in ('rgbd_calibrated','rgb_only'):
            self.session.perception=select(profile)
            policy=EefAgentPolicy(action_spec=self.spec,env={'L4_INSPECT_DEPTH':'off'} if profile=='rgb_only' else {},client=fixture.Client([]),
                planner=lambda **_:None,research=self.session)
            move=next(t['function'] for t in policy._build_tools(self.spec) if t['function']['name']=='move_eef')
            description=move['description']
            self.assertIn('Rz(yaw_deg) Ry(roll_deg) Rx(pitch_deg) Rref',description)
            self.assertIn('not independent world-axis increments',description)
            self.assertNotIn('roll_deg about world +y',description)
            self.assertIn('absolute world-frame grasp-point targets',description)
            self.assertIn('Per-dimension bounds:',description)
            self.assertIn('execute_max_s',move['parameters']['properties'])

    def test_documented_marginal_axes_match_real_quaternion_changes_at_nonzero_attitudes(self):
        # This checks the advertised meanings, not a second Euler implementation.
        for pitch,roll,yaw in ((0.,0.,90.),(20.,-30.,-45.),(-55.,65.,123.)):
            before=pose.angles_to_quat(pitch,roll,yaw)
            # Rotation axes are spatial vectors, independent of the grasp reference.
            rz=pose.quat_to_matrix(pose._axis_quat(2,np.deg2rad(yaw)))
            ry=pose.quat_to_matrix(pose._axis_quat(1,np.deg2rad(roll)))
            expected=[rz@ry@np.array([1.,0.,0.]),rz@np.array([0.,1.,0.]),np.array([0.,0.,1.])]
            for axis in range(3):
                parameters=[pitch,roll,yaw];parameters[axis]+=2.
                after=pose.angles_to_quat(*parameters)
                difference=pose.quat_normalize(pose.quat_multiply(after,pose.quat_conjugate(before)))
                if difference[0]<0:difference=-difference
                actual=difference[1:]/np.linalg.norm(difference[1:])
                np.testing.assert_allclose(actual,expected[axis],atol=1e-12)
