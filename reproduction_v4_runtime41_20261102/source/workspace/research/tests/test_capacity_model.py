"""Synthetic CLI transport tests; no model calls or network."""
import json,os,tempfile,unittest
from pathlib import Path
from test_codex_cli import FAKE
from kyb.codex_cli import CodexCliClient,CodexCliFailure
from kyb.budget import CallBudget

class CapacityModelTests(unittest.TestCase):
    def setUp(self):
        from unittest.mock import patch
        stop=patch.object(CodexCliClient,'_capacity_resume',return_value=False)
        stop.start();self.addCleanup(stop.stop)
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        fake=FAKE.replace("if os.environ.get('KYB_FAKE_MODE') in ('capacity','interrupted','malformed_failure'):",
            "if os.environ.get('KYB_FAKE_MODE') in ('capacity','interrupted','malformed_failure') and ('gpt-6.1-sol' in args or os.environ.get('KYB_FAKE_ALWAYS')=='yes'):")
        self.exe=self.root/'codex';self.exe.write_text(fake);self.exe.chmod(0o700)
        self.budget=CallBudget(self.root/'manager.jsonl',5)
        self.client=CodexCliClient(self.root/'run',executable=str(self.exe),role='manager',budget=self.budget)
        self.tools=[{'function':{'name':'publish_materials','parameters':{}}}]
        self.messages=[{'role':'user','content':'bounded manager request'}]
    def tearDown(self):
        for key in ('KYB_FAKE_MODE','KYB_FAKE_ALWAYS'):os.environ.pop(key,None)
        self.tmp.cleanup()
    def test_capacity_changes_model_and_preserves_native_sid_and_budget(self):
        os.environ['KYB_FAKE_MODE']='capacity'
        self.client.complete(self.messages,self.tools)
        summary=self.client.summary();self.assertEqual(self.budget.view()['used'],2)
        self.assertEqual((summary['native_failed_turns'],summary['outcome_unknown']),(1,0))
        self.assertEqual(summary['model'],'mixed')
        self.assertCountEqual([(x['model'],x['effort']) for x in summary['models']], [('gpt-6.1-sol','xhigh'),('gpt-6-astra','high')])
        switch=json.loads((self.client.root/'model_switches.jsonl').read_text())
        self.assertEqual(switch['native_session_id'],self.client.session_id)
        received=json.loads((self.client.root/'cli'/self.client.last_attempt_id/'received.json').read_text())
        self.assertIn('resume',received['argv'])
        self.assertEqual(summary['usage_reported'],{'input_tokens':7,'cached_input_tokens':2,'cache_write_input_tokens':0,'output_tokens':9,'reasoning_output_tokens':3})
    def test_capacity_alternative_failure_is_not_cycled_or_retried(self):
        os.environ['KYB_FAKE_MODE']='capacity';os.environ['KYB_FAKE_ALWAYS']='yes'
        with self.assertRaises(CodexCliFailure):self.client.complete(self.messages,self.tools)
        self.assertEqual(self.budget.view()['used'],2)
        with self.assertRaises(CodexCliFailure):self.client.complete(self.messages,self.tools)
        self.assertEqual(self.budget.view()['used'],2)
    def test_unknown_or_malformed_never_changes_model(self):
        for mode in ('interrupted','malformed_failure'):
            with self.subTest(mode=mode):
                c=CodexCliClient(self.root/mode,executable=str(self.exe),role='manager',budget=self.budget)
                os.environ['KYB_FAKE_MODE']=mode
                with self.assertRaises(CodexCliFailure):c.complete(self.messages,self.tools)
                self.assertEqual(c.model,'gpt-6.1-sol');self.assertEqual(c.summary()['outcome_unknown'],1)
                self.assertFalse((c.root/'model_switches.jsonl').exists())
    def test_unrelated_terminal_error_never_switches(self):
        raw=self.exe.read_text().replace('Selected model is at capacity. Please try a different model.', 'Permission denied for this model.')
        self.exe.write_text(raw);os.environ['KYB_FAKE_MODE']='capacity'
        with self.assertRaises(CodexCliFailure):self.client.complete(self.messages,self.tools)
        self.assertEqual(self.budget.view()['used'],1);self.assertEqual(self.client.model,'gpt-6.1-sol')
    def test_policy_never_uses_management_alternative(self):
        self.client.role='policy';self.client.model='gpt-6.1-sol';os.environ['KYB_FAKE_MODE']='capacity'
        with self.assertRaises(CodexCliFailure):self.client.complete(self.messages,self.tools)
        self.assertEqual(self.budget.view()['used'],1);self.assertEqual(self.client.model,'gpt-6.1-sol')

    def test_exhausted_budget_never_sends_alternative(self):
        self.client.budget=CallBudget(self.root/'limited.jsonl',1);os.environ['KYB_FAKE_MODE']='capacity'
        with self.assertRaises(CodexCliFailure):self.client.complete(self.messages,self.tools)
        self.assertEqual(self.client.summary()['cli_invocations'],1)
        self.assertEqual(self.client.model,'gpt-6.1-sol')
    def test_mixed_session_totals_deduplicate_same_sid_across_models(self):
        self.client.complete(self.messages,self.tools)
        self.messages.append({'role':'user','content':'next management step'})
        os.environ['KYB_FAKE_MODE']='capacity';self.client.complete(self.messages,self.tools)
        self.messages.append({'role':'user','content':'publish result'})
        self.client.complete(self.messages,self.tools)
        summary=self.client.summary()
        self.assertEqual(self.budget.view()['used'],4)
        self.assertEqual(summary['usage_reported'],{'input_tokens':21,'cached_input_tokens':6,'cache_write_input_tokens':0,'output_tokens':27,'reasoning_output_tokens':9})
        for field,value in summary['usage_reported'].items():
            self.assertEqual(sum(x['usage_reported'].get(field,0) for x in summary['models']),value)
