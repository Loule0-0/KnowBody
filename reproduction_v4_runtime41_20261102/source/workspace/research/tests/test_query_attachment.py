"""A schema-valid memory query must consume attachment metadata exactly once."""
import importlib.util
import json
from pathlib import Path
import unittest
import test_policy_pipeline as pipeline
from kyb.policy import EefAgentPolicy


class QueryAttachmentTests(unittest.TestCase):
    setUp=pipeline.PolicyPipelineTests.setUp
    tearDown=pipeline.PolicyPipelineTests.tearDown

    def arguments(self):
        return json.loads(Path('/home/lzy/KnowYourBody_RoboDojo/experiments/dual_refinement_20261023/incident_query_20261024/FAILING_PUBLIC_QUERY.json').read_text())['arguments']

    def policy(self):
        return EefAgentPolicy(action_spec=self.spec,env={},client=pipeline.Client([]),
            planner=lambda **_:None,research=self.session)

    def test_original_frozen_method_reproduces_the_recorded_error(self):
        path='/home/lzy/KnowYourBody_RoboDojo/workspaces/refinement-20261023/research/kyb/session.py'
        spec=importlib.util.spec_from_file_location('kyb.original_session_for_test',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with self.assertRaisesRegex(TypeError,'attachment_updates'):
            module.MemorySession.query_experience(self.session,self.arguments())

    def test_real_dispatch_accepts_promised_metadata_without_duplicate_consumption(self):
        args=self.arguments();policy=self.policy()
        tool=next(t['function'] for t in policy._build_tools(self.spec) if t['function']['name']=='query_experience')
        self.assertIn('attachment_updates',tool['parameters']['properties'])
        before=self.store.db.execute("SELECT COUNT(*) FROM evidence WHERE kind='attachment_change'").fetchone()[0]
        outcome=policy._handle_motion('query_experience',args,self.observation)
        self.assertIsNone(outcome.chunk)
        self.assertEqual(json.loads(outcome.tool_result)['status'],'no_match')
        self.assertEqual(self.session.attachment['right']['state'],'slipped')
        after=self.store.db.execute("SELECT COUNT(*) FROM evidence WHERE kind='attachment_change'").fetchone()[0]
        self.assertEqual(after-before,1)
        self.assertEqual(args,self.arguments())


if __name__=='__main__':unittest.main()
