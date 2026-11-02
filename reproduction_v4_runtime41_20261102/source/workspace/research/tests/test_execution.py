import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from kyb.execution import execute_chunk
from kyb.memory import MemoryStore
from kyb.session import MemorySession
from kyb.trace import RunTrace


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.trace = RunTrace(root / "run", {})
        self.store = MemoryStore(root / "memory.sqlite")
        self.session = MemorySession(self.trace, self.store, task="insert_key",
            dependencies={"body": "x5", "cameras": "c", "tool": "key",
                          "attachment:left": "old", "attachment:right": "none"})
        ref = self.session.evidence("observation", {"images": {}, "dependencies": self.session.dependencies})
        self.observation = SimpleNamespace(step=0, extra={"evidence_id": ref["id"]})
        self.env = SimpleNamespace(take_action_cnt=[0], sim=SimpleNamespace(sim=SimpleNamespace(current_time_step_index=0)))
        self.env.is_episode_end = lambda: False
        self.env.take_action = self.act
        self.chunk = SimpleNamespace(actions=[SimpleNamespace(data={}, meta={}) for _ in range(3)], meta={})

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def act(self, _action):
        self.env.take_action_cnt[0] += 1
        self.env.sim.sim.current_time_step_index += 10

    def capture(self, _step, expose):
        ref = self.trace.event("execution_frame", {}, visibility="recorded_not_exposed")
        return SimpleNamespace(extra={"evidence_id": ref["id"]})

    def run_chunk(self, capture=None):
        return execute_chunk(self.env, self.chunk, trace=self.trace, session=self.session,
            capture=capture or self.capture, observation=self.observation, call_id="local")

    def test_success_receipt_has_actual_steps_and_source_binding(self):
        self.assertEqual(self.run_chunk(), (3, False))
        data = self.session.last_receipt["data"]
        self.assertEqual((data["executed_actions"], data["physics_end"]), (3, 30))
        self.assertEqual(data["before_observation"], self.observation.extra["evidence_id"])
        self.assertEqual(len(data["command_refs"]), 3)

    def test_native_end_on_last_selected_step_has_priority_over_prefix_boundary(self):
        self.chunk.meta={'full_planned_actions':8,'selected_stop_reason':'observation_boundary'}
        self.env.is_episode_end=lambda:self.env.take_action_cnt[0]>=3
        self.assertEqual(self.run_chunk(),(3,False))
        receipt=self.session.last_receipt['data']
        self.assertEqual(receipt['stop_reason'],'native_episode_end')
        self.assertEqual(receipt['execution'],'partial')
        self.assertEqual((receipt['full_planned_actions'],receipt['selected_actions'],receipt['executed_actions']),(8,3,3))

    def test_camera_failure_preserves_executed_first_action(self):
        def failed_capture(*_args, **_kwargs):
            raise RuntimeError("synthetic camera failure")
        with self.assertRaises(RuntimeError):
            self.run_chunk(failed_capture)
        data = self.session.last_receipt["data"]
        self.assertEqual(data["executed_actions"], 1)
        self.assertEqual(data["physics_end"], 10)
        self.assertEqual(data["execution"], "partial")
        self.assertIsNone(data["last_recorded_execution_frame"])

    def test_command_midflight_failure_is_unknown_not_zero_motion(self):
        def partial(_action):
            self.env.take_action_cnt[0] += 1
            self.env.sim.sim.current_time_step_index += 4
            raise RuntimeError("synthetic mid-control failure")
        self.env.take_action = partial
        with self.assertRaises(RuntimeError):
            self.run_chunk()
        data = self.session.last_receipt["data"]
        self.assertEqual(data["executed_actions"], 0)
        self.assertEqual(data["accepted_actions"], 1)
        self.assertEqual(data["physics_end"], 4)
        self.assertEqual(data["execution"], "outcome_unknown")
        self.assertIsNotNone(data["unknown_command"])

    def test_partial_unknown_receipt_cannot_label_an_effect(self):
        def partial_second(action):
            if self.env.take_action_cnt[0] == 0:
                self.act(action)
                return
            self.env.take_action_cnt[0] += 1
            self.env.sim.sim.current_time_step_index += 4
            raise RuntimeError("second command interrupted")
        self.env.take_action = partial_second
        with self.assertRaises(RuntimeError):
            self.run_chunk()
        receipt = self.session.last_receipt
        self.assertEqual(receipt["data"]["executed_actions"], 1)
        after = self.session.evidence("observation", {
            "after_receipt": receipt["id"], "dependencies": self.session.dependencies})
        decision = self.store.submit({"op": "new", "kind": "external", "relation_type": "observed_interaction",
            "text": "This must not become a known physical effect", "conditions": {},
            "reference_frame": "image", "epistemic": "visually_inferred",
            "evidence": [receipt["id"], after["id"]]},
            allowed_sources=self.session.allowed, cutoff=self.session.cutoff,
            dependencies=self.session.dependencies, episode=self.session.episode)
        self.assertFalse(decision["accepted"])


if __name__ == "__main__":
    unittest.main()
