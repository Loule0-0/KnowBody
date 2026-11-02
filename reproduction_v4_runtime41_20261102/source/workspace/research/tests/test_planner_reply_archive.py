"""Received-path provenance without changing native motion or current state."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import MethodType
import unittest
import numpy as np

from kyb import execution_contract
from kyb.policy import EefAgentPolicy
from kyb.session import MemorySession
import test_policy_pipeline as fixture

OLD=Path('/home/lzy/KnowYourBody_RoboDojo/workspaces/v4-runtime16-20261027/research/kyb')

class PlannerReplyArchiveTests(unittest.TestCase):
    setUp=fixture.PolicyPipelineTests.setUp
    tearDown=fixture.PolicyPipelineTests.tearDown

    def reply(self, planned):
        ref=self.session.record_plan('left',self.observation.state['left_ee_pose'],planned,4)
        result=self.store.evidence_payload(ref['id'])
        return result,result

    def test_branch_counterexample_preserves_raw_dtype_precision_and_capture_identity(self):
        # Synthetic branch ambiguity, not a reconstruction of main29's missing raw.
        self.session.current_observation['data']['state']['left_arm_joint_state'][0]=np.deg2rad(45)
        raw=np.zeros((3,6),dtype=np.float64);raw[:,0]=np.deg2rad([-315,-315.123456789,-454.11179]);saved=raw.copy()
        result,reply=self.reply({'status':'Success','position':raw})
        a=reply['received_planner_position'];path=Path(a['path']);loaded=np.load(path,allow_pickle=False)
        self.assertEqual(a['sha256'],hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(a['dtype'],'float64');self.assertEqual(a['shape'],[3,6]);np.testing.assert_array_equal(loaded,saved)
        np.testing.assert_array_equal(raw,saved)
        self.assertEqual(reply['before_observation_id'],self.session.current_observation['id'])
        self.assertEqual(reply['arm'],'left');self.assertAlmostEqual(reply['start_joint_angles_rad'][0],np.deg2rad(45))
        self.assertAlmostEqual(loaded[0,0],np.deg2rad(-315))
        self.assertIn('not an internal solver-input dump',reply['planner_reply_qualification'])
        self.assertNotIn('position',result)

    def test_failed_none_reply_is_explicit_and_creates_no_array(self):
        result,reply=self.reply({'status':'Fail','position':None})
        self.assertFalse(result['path_returned']);self.assertIsNone(reply['received_planner_position']);self.assertEqual(reply['status'],'Fail')
        self.assertFalse((self.trace.root/'planner').exists())

    def test_existing_archive_is_never_overwritten(self):
        folder=self.trace.root/'planner';folder.mkdir()
        path=folder/f'reply_{self.trace.sequence+1:06d}_left.npy';path.write_bytes(b'preserve')
        with self.assertRaises(FileExistsError):self.reply({'status':'Success','position':np.zeros((2,6))})
        self.assertEqual(path.read_bytes(),b'preserve')

    def test_same_fake_planner_gives_identical_native_ticks_and_unchanged_motion_pin_sources(self):
        spec=importlib.util.spec_from_file_location('kyb._archive_baseline16',OLD/'session.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        methods=[module.MemorySession.record_plan,MemorySession.record_plan];runs=[]
        raw=np.stack([np.zeros(6),np.full(6,.1)])
        for method in methods:
            self.session.record_plan=MethodType(method,self.session)
            policy=EefAgentPolicy(action_spec=self.spec,env={},client=fixture.Client([]),
                planner=lambda **kwargs:{'status':'Success','position':raw.copy()},research=self.session)
            outcome=policy._handle_motion('move_eef',{'targets':{'left_z':1.0}},self.observation)
            self.assertIsNotNone(outcome.chunk)
            runs.append([{k:np.array(v).copy() for k,v in a.data.items()} for a in outcome.chunk.actions])
        self.assertEqual(len(runs[0]),len(runs[1]))
        for old,new in zip(*runs):
            self.assertEqual(old.keys(),new.keys())
            for key in old:np.testing.assert_array_equal(old[key],new[key])
        package=Path(execution_contract.__file__).parent
        closure=execution_contract.local_closure([package/'motion_core.py',package/'execution.py',package/'body_guard.py'])
        for name,digest in closure.items():self.assertEqual(digest,hashlib.sha256((OLD/Path(name).name).read_bytes()).hexdigest())
