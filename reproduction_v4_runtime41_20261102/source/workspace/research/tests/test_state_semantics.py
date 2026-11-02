"""Actual archived counterexample and local profile views; no paid calls/actions."""
import copy,json,math,unittest
from pathlib import Path
from types import SimpleNamespace
from kyb.policy import EefAgentPolicy
from kyb.perception import RGB_ONLY,RGBD_CALIBRATED
from kyb.operating_notes import for_perception,PROJECTION_CLAIM
from kyb.replay import load_prefix
from XPolicyLab.policy.GPT_6_Astra_Direct_EEF.docs import eef_docs_from_joint_docs
from XPolicyLab.policy.GPT_6_Astra_Direct_EEF import pose

SOURCE=Path('/data/KnowYourBody_RoboDojo/runs/main42_rgb_v4_01_20261026_r01_t01_stack_bowls_20261026T113926')
class StateSemanticsTests(unittest.TestCase):
    def fixture(self):
        refs=[json.loads(x) for x in (SOURCE/'events.jsonl').read_text().splitlines()]
        obs=next(x for x in refs if x['id']==SOURCE.name+':e000383')
        command=next(x for x in refs if x['id']==SOURCE.name+':e000019')
        observed=json.loads(Path(obs['path']).read_text())['payload']
        action=json.loads(Path(command['path']).read_text())['payload']['action']
        p=EefAgentPolicy.__new__(EefAgentPolicy)
        p.action_spec=SimpleNamespace(labels=tuple([f'left_joint{i}' for i in range(1,7)]+['left_gripper']+[f'right_joint{i}' for i in range(1,7)]+['right_gripper']))
        p._last_executed_action=action
        return p,SimpleNamespace(state=copy.deepcopy(observed['state']),instruction='stack bowls')
    def test_real_reading_and_actual_confirmed_command_are_distinct(self):
        p,obs=self.fixture();text=p._state_block(obs)
        self.assertIn('left_gripper=0.517 (normalized state reading); last_confirmed_command=1.000',text)
        self.assertNotIn('previous normalized command',text)
        self.assertIn('not commandable',text)
    def test_unknown_and_nonfinite_readings_are_never_reported_as_closed(self):
        p,obs=self.fixture();p._last_executed_action=None
        obs.state.pop('left_ee_joint_state');obs.state['right_ee_joint_state']=[math.nan]
        text=p._state_block(obs)
        self.assertIn('left_gripper=unavailable',text);self.assertIn('right_gripper=unavailable',text)
        self.assertIn('last_confirmed_command=unavailable',text)
    def test_rgb_profile_has_no_unavailable_body_query_claim(self):
        source=eef_docs_from_joint_docs('');view=for_perception(source,RGB_ONLY)
        self.assertNotIn('available through read_body',view);self.assertIn('No body projection',view)
        self.assertEqual(for_perception(source,RGBD_CALIBRATED),source)
        runner=(Path(__file__).parents[1]/'run_episode.py').read_text()
        self.assertIn('action_contract=sdk_action_docs',runner)
        self.assertIn('docs=for_perception(sdk_action_docs, perception)',runner)
    def test_new_public_query_contract_requires_review(self):
        with self.assertRaises(ValueError):for_perception('Unreviewed changed source',RGB_ONLY)
    def test_real_selected_observation_prefixes_have_complete_known_commands(self):
        prefix=load_prefix(SOURCE,SOURCE.name+':e001989',task='stack_bowls',layout=39)
        self.assertEqual((prefix['env_actions'],len(prefix['chunks'])),(564,29))
        selected=[x for x in prefix['chunks'] if x['receipt']['payload']['execution']=='partial']
        self.assertEqual(len(selected),5)
        self.assertTrue(all(x['receipt']['payload']['stop_reason']=='observation_boundary' for x in selected))
    def test_pose_readout_uses_same_existing_conversion(self):
        p,obs=self.fixture();text=p._state_block(obs)
        for arm in pose.ARMS:
            values=p._measured_values(obs,arm)
            for axis in ('x','y','z'):self.assertIn(f'{arm}_{axis}={values[axis]:.4f}',text)
