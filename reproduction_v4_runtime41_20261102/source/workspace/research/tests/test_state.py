import json
import unittest
from pathlib import Path

import numpy as np
from XPolicyLab.policy.GPT_6_Astra_Direct_EEF import pose

from kyb.state import evidence_state, evidence_receipt, compare_states
from kyb.robodojo_adapter import canonical_state

def declared(payload):
    return {**payload, "canonical_state":canonical_state(payload.get("state",{}),payload.get("measured_finger_joints_m",{}),"test_body")}


class CoordinateEvidenceTests(unittest.TestCase):
    def test_recorded_missed_grasp_has_distinct_consistent_reference_points(self):
        path = Path('/data/KnowYourBody_RoboDojo/runs/20261018T121514_538047197/events/e000731.json')
        payload = json.loads(path.read_text())['payload']
        original = json.dumps(payload, sort_keys=True)
        view = evidence_state(declared(payload))
        self.assertAlmostEqual(view['raw_state']['left_ee_pose'][2], .92260092, places=7)
        self.assertAlmostEqual(view['grasp_point_dimensions']['left_z'], .77250175, places=7)
        self.assertEqual(json.dumps(payload, sort_keys=True), original)

    def test_rotated_tool_offset_is_not_fixed_world_z_subtraction(self):
        flange = [.2, -.1, .9, 1., 0., 0., 0.]
        view = evidence_state(declared({'state': {'left_ee_pose': flange}}))
        dimensions = view['grasp_point_dimensions']
        self.assertAlmostEqual(dimensions['left_x'], .3501)
        self.assertAlmostEqual(dimensions['left_z'], .9)
        restored = pose.values_to_pose({k: dimensions['left_'+k]
            for k in ('x', 'y', 'z', 'pitch_deg', 'roll_deg', 'yaw_deg')})
        np.testing.assert_allclose(restored[:3], flange[:3], atol=1e-10)
        np.testing.assert_allclose(pose.quat_to_matrix(restored[3:]), np.eye(3), atol=1e-10)

    def test_receipt_planner_and_action_fields_have_explicit_referents(self):
        receipt = {'plan': {'trace': {'arms': {'left': {'target_pose': [0,0,1,1,0,0,0],
            'named': {'left_z': .8}}}}}}
        view = evidence_receipt(receipt)
        self.assertIn('source coordinate contract', view['coordinate_conventions']['plan.trace.arms.*.target_pose'])
        self.assertIn('not realized', view['coordinate_conventions']['plan.trace.arms.*.named'])
        self.assertNotIn('coordinate_conventions', receipt)

    def test_known_finger_and_pose_differences_are_not_called_applicability(self):
        source = {'state': {'left_ee_pose':[.2,-.1,.9,1.,0.,0.,0.]},
                  'measured_finger_joints_m': {'left':[0.,0.]}}
        current = {'state': {'left_ee_pose':[.3,-.1,.9,1.,0.,0.,0.]},
                   'measured_finger_joints_m': {'left':[.044,.044]}}
        result = compare_states(declared(source), declared(current))
        differences = result['measured_differences']['left']
        self.assertAlmostEqual(differences['grasp_position_current_minus_source_m'][0], .1)
        self.assertEqual(differences['orientation_difference_deg'], 0.)
        self.assertEqual(differences['finger_displacement_current_minus_source_m'], [.044,.044])
        self.assertNotIn('applicable', result)


if __name__ == '__main__':
    unittest.main()
