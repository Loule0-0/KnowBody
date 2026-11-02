import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
import numpy as np

from kyb.memory import MemoryStore
from kyb.session import MemorySession
from kyb.trace import RunTrace, plain
from manager_fixtures import PreparationClient, ScriptedManager
from kyb.policy import EefAgentPolicy
from XPolicyLab.policy.GPT_6_Astra_Direct_EEF import pose
from XPolicyLab.policy.GPT_6_Astra_Direct_EEF.inspect.policy import RoboDojoActionSpec
from XPolicyLab.policy.GPT_6_Astra_Direct_EEF.inspect.types import Observation


class Client:
    def __init__(self, calls):
        self.calls = list(calls)
        self.inputs = []

    def complete(self, messages, _tools):
        self.inputs.append(copy.deepcopy(messages))
        return {"choices": [{"message": {"role": "assistant", "content": None,
                "tool_calls": [self.calls.pop(0)]}, "finish_reason": "tool_calls"}],
                "usage": {"input_tokens": 0, "output_tokens": 0}}


def call(name, arguments):
    return {"id": f"test_{name}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)}}


class PolicyPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.trace = RunTrace(root / "run", {})
        self.store = MemoryStore(root / "memory.sqlite")
        self.deps = {"body": "x5", "cameras": "cam", "tool": "tool", 'planner':'planner1',
                     "attachment:left": "grip", "attachment:right": "none"}
        self.session = MemorySession(self.trace, self.store, task="push_T", dependencies=self.deps)
        self.manager_client=PreparationClient()
        self.session.attach_manager(self.manager_client)
        before = self.session.evidence("observation", {"dependencies": self.deps})
        self.receipt = self.session.evidence("receipt", {"executed_actions": 5, "execution":"complete",
            "before_observation": before["id"], "dependencies": self.deps})
        self.observation = Observation(images={"head": np.zeros((8, 8, 3), dtype=np.uint8)},
            state={
                "left_arm_joint_state": np.zeros(6), "right_arm_joint_state": np.zeros(6),
                "left_ee_joint_state": np.ones(1), "right_ee_joint_state": np.ones(1),
                "left_ee_pose": pose.values_to_pose({"x": -.3, "y": -.35, "z": .95,
                    "pitch_deg": 0, "roll_deg": 0, "yaw_deg": 0}),
                "right_ee_pose": pose.values_to_pose({"x": .3, "y": -.35, "z": .95,
                    "pitch_deg": 0, "roll_deg": 0, "yaw_deg": 0}),
            }, instruction="Synthetic pipeline test", remaining_steps=750,
            extra={"task": "push_T"})
        self.after = self.session.evidence("observation", {
            "after_receipt": self.receipt["id"], "dependencies": self.deps,
            'state':plain(self.observation.state)})
        self.observation = replace(self.observation, extra={**self.observation.extra,'evidence_id':self.after['id']})
        labels = tuple([f"left_joint{i}" for i in range(1, 7)] + ["left_gripper"] +
                       [f"right_joint{i}" for i in range(1, 7)] + ["right_gripper"])
        low, high = np.full(14, -3.0), np.full(14, 3.0)
        low[[6, 13]], high[[6, 13]] = 0, 1
        self.spec = RoboDojoActionSpec(labels, low, high, 25, "Synthetic config", (0.1,) * 14)

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def patch(self):
        return {"op": "new", "reading_role":"interaction", "kind": "external", "relation_type": "observed_interaction", "module": "interaction",
            "text": "A visibly offset push rotated the object", "conditions": {"task": "push_T"},
            "reference_frame": "tool_local", "epistemic": "visually_inferred",
            "receipt_id": self.receipt["id"], "post_observation_id": self.after["id"],
            "evidence": [self.receipt["id"], self.after["id"]]}

    def test_replay_branch_preserves_closed_goal_instead_of_observed_gap(self):
        policy=EefAgentPolicy(action_spec=self.spec,env={},client=Client([]),
            planner=lambda **_:None,research=self.session)
        policy.resume_executed_gripper_goals({'left_ee_joint_state':[0.],'right_ee_joint_state':[.4]})
        observed=replace(self.observation,state={**self.observation.state,'left_ee_joint_state':np.array([.15])})
        held=policy._held_action_data(observed)
        self.assertEqual(float(held['left_ee_joint_state'][0]),0.)
        self.assertAlmostEqual(float(held['right_ee_joint_state'][0]),.4)

    def test_surface_query_is_readonly_then_policy_still_selects_eef(self):
        depth_path=Path(self.temp.name)/'depth.npy';np.save(depth_path,np.ones((480,640),dtype=np.float32))
        import hashlib
        payload={**self.session.current_observation['data'],
            'images':{'cam_head':{'sha256':'rgb'}},'depth_snapshots':{'cam_head':{'path':str(depth_path),'sha256':hashlib.sha256(depth_path.read_bytes()).hexdigest()}},
            'body_visual_reference':{'inner_face_windows_flange_m':{arm:{finger:{'polygon_xz_m':[[.14,-.01],[.16,-.01],[.16,.01],[.14,.01]],'plane_y_m':y,'inward_y_sign':sign} for finger,y,sign in [('finger7',.01,-1),('finger8',-.01,1)]} for arm in ('left','right')},'views':{'cam_head':{'resolution_wh':[640,480],
                'intrinsics_K':[[400,0,320],[0,400,240],[0,0,1]],'world_to_optical_ros':np.eye(4).tolist()}}}}
        ref=self.session.evidence('observation',payload)
        observation=replace(self.observation,extra={**self.observation.extra,'evidence_id':ref['id']})
        receipt=self.session.last_receipt['id']
        client=Client([call('inspect_surface',{'observation_id':ref['id'],'samples':[{'camera':'cam_head','pixel_uv':[320,240],'label':'visible surface'}]}),
            call('move_eef',{'targets':{'left_gripper':0},'note':'Synthetic movement after geometry'})])
        policy=EefAgentPolicy(action_spec=self.spec,env={},client=client,planner=lambda **_:None,research=self.session)
        chunk=policy.act(observation)
        self.assertEqual(policy.calls,2);self.assertTrue(chunk.actions)
        self.assertEqual(self.session.last_receipt['id'],receipt)
        record=self.store.db.execute("SELECT payload FROM evidence WHERE kind='surface_inspection'").fetchone()
        data=json.loads(record[0]);self.assertEqual(data['executed_actions'],0);self.assertEqual(data['physics_steps'],0)
        self.assertIn('world_xyz_m',json.dumps(client.inputs[1]))
        stale=self.session.inspect_surface({'observation_id':'stale','samples':[]},observation)
        self.assertIn('NEEDS_WORK',stale)
        schema=next(t for t in policy._build_tools(self.spec) if t['function']['name']=='inspect_surface')
        self.assertNotIn('need',schema['function']['parameters']['properties'])
        invalid=self.session.inspect_surface({'observation_id':ref['id'],'samples':[],
            'need':'Measure the current geometry after withdrawal.'},observation)
        self.assertIn('NEEDS_WORK',invalid)
        self.assertFalse(self.store.db.execute("SELECT 1 FROM evidence WHERE kind='experience_query'").fetchone())

    def test_policy_trigger_updates_current_visual_state_before_manager_then_selects_action(self):
        publication={'expected_version':0,'answer':'Prepared','default_knowledge':[]}
        manager=ScriptedManager([('publish_materials',publication),
            ('publish_materials',{'expected_version':1,'answer':'Compare these grasp conditions, not a motion plan.',
                'memory_patch':[]})])
        self.session.manager.client=manager
        client=Client([call('manage_memory',{'question':'Compare this changed grasp with historical retention.',
                'purpose':'review','attachment_updates':[{'arm':'left','tool':'tool','state':'visually_supported'}]}),
            call('move_eef',{'targets':{'left_gripper':0},'note':'Policy chooses the target after review'})])
        policy=EefAgentPolicy(action_spec=self.spec,env={},client=client,planner=lambda **_:None,research=self.session)
        chunk=policy.act(self.observation)
        self.assertTrue(chunk.actions);self.assertEqual(policy.calls,2)
        self.assertEqual(len(manager.inputs),2)
        review=json.loads(manager.inputs[1][1]['content'])
        self.assertEqual(review['current_attachment_declarations']['left']['state'],'visually_supported')
        self.assertEqual(self.session.last_receipt['id'],self.receipt['id'])
        self.assertEqual(self.store.db.execute("SELECT COUNT(*) FROM evidence WHERE kind='policy_decision'").fetchone()[0],2)

    def test_host_stop_prevents_another_readonly_query_or_model_request(self):
        client=Client([])
        policy=EefAgentPolicy(action_spec=self.spec,env={},client=client,planner=lambda **_:None,research=self.session)
        (self.trace.root/'STOP').touch()
        result=policy.act(self.observation)
        self.assertTrue(result.actions[0].meta['request_stop'])
        self.assertEqual(policy.calls,0);self.assertEqual(client.inputs,[])

    def test_research_capabilities_agree_with_depth_tool_and_audit_mode(self):
        policy=EefAgentPolicy(action_spec=self.spec,env={},client=Client([]),planner=lambda **_:None,research=self.session)
        self.assertEqual(policy._depth,'surface_queries')
        self.assertIn('inspect_surface',policy._system_message())
        self.assertNotIn('no depth or world-coordinate query is available',policy._system_message())
        self.assertNotIn('exactly one move_eef tool call',policy._system_message())

    def test_failed_next_plan_reports_no_execution_without_automatic_learning(self):
        client = Client([
            call("move_eef", {"targets": {"left_z": .99}, "note": "Synthetic",
                "memory_context": {"support":"table"}}),
            call("give_up", {"reason": "Stop synthetic test", "hindsight": "No physical execution"}),
        ])
        policy = EefAgentPolicy(action_spec=self.spec, env={}, client=client,
            planner=lambda **_kwargs: {"status": "Fail"}, research=self.session)
        result = policy.act(self.observation)
        self.assertTrue(result.actions[0].meta["request_stop"])
        self.assertEqual(self.store.revision, 0)
        self.assertNotIn('planning_samples', json.dumps(client.inputs[1]))
        self.assertIn('did not find a usable path',json.dumps(client.inputs[1]))
        planning = self.store.db.execute("SELECT payload FROM evidence WHERE kind='planning_result'").fetchone()
        self.assertEqual(json.loads(planning[0])['status'], 'Fail')
        self.assertEqual(json.loads(planning[0])['executed_actions'], 0)
        self.assertNotIn('pose unreachable', json.dumps(client.inputs[1]))
        self.assertEqual(len(self.manager_client.inputs),1)

    def test_planning_fact_is_not_physical_effect_and_pins_source_planner(self):
        ref = self.session.record_plan('left', self.observation.state['left_ee_pose'], {'status':'Fail'}, 1)
        sources = {ref['id'],self.after['id']}
        kwargs = dict(allowed_sources=sources,cutoff={'run':self.trace.sequence},
            dependencies={**self.deps,'planner':'planner2'},episode='run')
        external = {**self.patch(),'evidence':sorted(sources)}
        self.assertFalse(self.store.submit(external,**kwargs)['accepted'])
        body = {'op':'new','reading_role':'intrinsic_candidate','kind':'body','observed_entity':'robot_part','observable':'configuration',
            'module':'planning observation','text':'This planner did not find a path for this target and start.',
            'conditions':{'arm':'left'},'reference_frame':'world','epistemic':'hypothesis','evidence':sorted(sources)}
        result = self.store.submit(body,**kwargs)
        self.assertTrue(result['accepted'],result)
        row = self.store.db.execute('SELECT dependencies FROM claims WHERE revision=?',(result['revision'],)).fetchone()
        self.assertEqual(json.loads(row[0])['planner'],'planner1')

    def test_invalid_attachment_is_rejected_without_database_write(self):
        self.session.consume({'attachment_updates':[None]},self.observation)
        self.assertFalse(self.session.last_decisions[0]['accepted'])
        self.assertEqual(self.store.revision,0)

    def test_native_episode_continuity_has_no_dead_progress_card_api(self):
        self.assertNotIn('task_continuation',self.session.tool_properties())
        self.assertIn('Native session history',self.session.instructions())
        self.assertFalse(hasattr(self.session,'progress_view'))

    def test_question_read_returns_conditions_without_manager_or_physics(self):
        decision=self.store.submit(self.patch(),allowed_sources=self.session.allowed,
            cutoff=self.session.cutoff,dependencies=self.deps,episode='run')
        self.assertTrue(decision['accepted'],decision)
        before_receipt=self.session.last_receipt['id']
        from kyb.library import organize_archive
        organize_archive(self.store)
        result=json.loads(self.session.query_experience({'question':'offset push rotated','level':'case'}))
        self.assertEqual(len(result['records']),1)
        self.assertIn('conditions',result['records'][0])
        self.assertEqual(self.session.last_receipt['id'],before_receipt)
        self.assertEqual(len(self.manager_client.inputs),0)

    def test_large_historical_call_count_does_not_terminate(self):
        client = Client([call("move_eef", {"targets": {"left_gripper": 1.0}, "note": "Hold"})])
        policy = EefAgentPolicy(action_spec=self.spec, env={}, client=client,
            planner=lambda **_kwargs: {"status": "Success", "position": np.zeros((1, 6))},
            research=self.session)
        policy._calls = 10000
        result = policy.act(self.observation)
        self.assertFalse(result.actions[0].meta.get("request_stop", False))
        self.assertEqual(len(client.inputs), 1)

    def test_rejected_joint_path_does_not_latch_unexecuted_gripper_goal(self):
        client=Client([
            call('move_eef',{'targets':{'left_z':.99,'left_gripper':0},
                'note':'Synthetic rejected path'}),
            call('give_up',{'reason':'Stop synthetic test','hindsight':'No execution'})])
        validator=lambda *_args:{'status':'NEEDS_WORK','links':{'left':'link4','right':'link6'},
            'minimum_cross_arm_gap_m':-.01,'evidence_id':'synthetic_path_check'}
        policy=EefAgentPolicy(action_spec=self.spec,env={},client=client,research=self.session,
            planner=lambda **_kwargs:{'status':'Success','position':np.zeros((1,6))},path_validator=validator)
        result=policy.act(self.observation)
        self.assertTrue(result.actions[0].meta['request_stop'])
        self.assertEqual(policy._gripper_goals,{})
        self.assertEqual(policy._last_targets,{})
        self.assertEqual(self.store.revision,0)
        self.assertIn('Neither arm moved',json.dumps(client.inputs[1]))


if __name__ == "__main__":
    unittest.main()
