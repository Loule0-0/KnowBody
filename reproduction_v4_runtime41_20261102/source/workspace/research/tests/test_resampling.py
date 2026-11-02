"""Check the declared step contract on curved and actual submitted trajectories."""
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from XPolicyLab.policy.GPT_6_Astra_Direct_EEF.policy import EefAgentPolicy, InfrastructureFailure


class ResamplingTests(unittest.TestCase):
    def sample(self, path, limits=None):
        limits = np.asarray(limits if limits is not None else [.05]*6)
        coordinates = EefAgentPolicy._path_step_coordinates(path, limits)
        count = max(1, int(np.ceil(coordinates[-1])))
        result = EefAgentPolicy._resample_path(path, count, limits)
        actual = np.vstack((path[0], result))
        self.assertLessEqual(float(np.max(np.abs(np.diff(actual, axis=0))/limits)), 1.00001)
        np.testing.assert_allclose(result[-1], path[-1], atol=1e-7)
        return count, result

    def test_nonuniform_path_indices_cannot_bypass_per_joint_budget(self):
        path=np.zeros((101,6));path[99:,0]=[.01,.30]
        self.sample(path)
        source=np.linspace(0,1,len(path));old=np.interp(np.linspace(0,1,7)[1:],source,path[:,0])
        self.assertGreater(np.max(np.abs(np.diff(np.r_[0,old]))),.05)

    def test_joint_switches_reversals_and_zero_length_knots(self):
        path=np.array([[0,0,0,0,0,0],[.2,0,0,0,0,0],[.2,0,0,0,0,0],
            [.2,-.3,0,0,0,0],[0,-.3,0,0,0,0],[0,0,0,0,0,0]])
        count,_=self.sample(path,[.05,.03,.05,.05,.05,.05]);self.assertEqual(count,28)
        self.sample(np.repeat(path[:1],5,axis=0))
        with self.assertRaises(InfrastructureFailure):
            EefAgentPolicy._resample_path(path,1,np.full(6,.05))

    def test_actual_successful_and_slipping_submitted_paths_both_had_old_violations(self):
        run=Path('/data/KnowYourBody_RoboDojo/runs/20261018T160232_895713878')
        for number in (1624,1656,1686):
            receipt=json.loads((run/'events'/f'e{number:06}.json').read_text())['payload']
            before=json.loads((run/'events'/(receipt['before_observation'].split(':')[-1]+'.json')).read_text())['payload']
            commands=[json.loads((run/'events'/(ref.split(':')[-1]+'.json')).read_text())['payload']['action']
                      for ref in receipt['command_refs']]
            path=np.array([before['state']['right_arm_joint_state'],*[c['right_arm_joint_state'] for c in commands]])
            self.assertGreater(float(np.max(np.abs(np.diff(path,axis=0)))),.07)
            self.sample(path)

    def test_bimanual_count_uses_max_budget_length_and_includes_actual_start(self):
        first=np.zeros((3,6));first[1,0]=.1;first[2,1]=.1
        second=np.zeros((2,6));second[1,2]=.4
        stub=SimpleNamespace(_arm_step_limits={'left':np.full(6,.05),'right':np.full(6,.05)},
                             _path_step_coordinates=EefAgentPolicy._path_step_coordinates)
        steps=EefAgentPolicy._arm_steps(stub,{'left':first,'right':second})
        for arm,path in [('left',first),('right',second)]:
            sampled=EefAgentPolicy._resample_path(path,steps,stub._arm_step_limits[arm])
            self.assertLessEqual(np.max(np.abs(np.diff(np.vstack((path[0],sampled)),axis=0))),.050001)
