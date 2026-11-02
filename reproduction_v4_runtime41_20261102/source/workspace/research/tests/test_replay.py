import copy
import unittest
from pathlib import Path

from kyb.replay import load_prefix, compare_body

SOURCE=Path('/data/KnowYourBody_RoboDojo/runs/20261018T135626_290364267')


class PrefixTests(unittest.TestCase):
    def test_real_prefix_is_complete_scene_commands_not_robot_teleport(self):
        prefix=load_prefix(SOURCE,SOURCE.name+':e001030',task='insert_key',layout=0)
        self.assertEqual(prefix['env_actions'],272)
        self.assertEqual(len(prefix['chunks']),24)
        self.assertEqual(sum(len(c['actions']) for c in prefix['chunks']),272)
        self.assertEqual(prefix['chunks'][-1]['post']['id'],SOURCE.name+':e001030')
        self.assertEqual(prefix['chunks'][-1]['actions'][-1]['left_ee_joint_state'],[0.])
        with self.assertRaises(ValueError):
            load_prefix(SOURCE,SOURCE.name+':e001030',task='insert_key',layout=1)
        actual=prefix['chunks'][-1]['post']['payload']
        self.assertEqual(compare_body(actual,actual)['status'],'PASS')
        divergent=copy.deepcopy(actual);divergent['measured_finger_joints_m']['left'][0]=0
        self.assertEqual(compare_body(actual,divergent)['status'],'NEEDS_WORK')
