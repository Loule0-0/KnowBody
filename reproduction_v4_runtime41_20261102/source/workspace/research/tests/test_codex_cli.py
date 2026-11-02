"""Subprocess-boundary tests with a local fake; no real model/auth/network."""
import base64
import json
import os
import tempfile
import unittest
from pathlib import Path
from kyb.codex_cli import CodexCliClient,CodexCliFailure
from kyb.budget import CallBudget,BudgetExhausted

FAKE=r'''#!/usr/bin/env python3
import sys,json,os,time,uuid
from pathlib import Path
if '--version' in sys.argv:print('synthetic-codex');raise SystemExit
args=sys.argv[1:];out=Path(args[args.index('--output-last-message')+1]);workspace=Path(args[args.index('-C')+1])
prompt=json.loads(sys.stdin.read())
sid=args[args.index('resume')+1] if 'resume' in args else str(uuid.uuid4())
state=workspace/(sid+'.json')
prior=json.loads(state.read_text()) if state.exists() else {}
config=next(v for v in args if v.startswith('model_instructions_file='))
instructions=Path(json.loads(config.split('=',1)[1])).read_text()
tools=[{'function':f} for f in json.loads(instructions.split('AVAILABLE HOST TOOLS (name, description, complete parameter schema):\n')[1])]
turn=prior.get('turn',0)+1
state.write_text(json.dumps({'tools':tools,'turn':turn,'reported':prior.get('reported',0)}))
out.with_name('received.json').write_text(json.dumps({'prompt':prompt,'argv':args,
 'instructions':instructions,'parent_payload_inherited':'CODEX_REMOTE_PAYLOAD' in os.environ}))
print(json.dumps({'type':'thread.started','thread_id':sid}),flush=True)
if os.environ.get('KYB_FAKE_MODE')=='timeout':time.sleep(10)
if os.environ.get('KYB_FAKE_MODE') in ('capacity','interrupted','malformed_failure'):
 print(json.dumps({'type':'turn.started'}),flush=True)
 print(json.dumps({'type':'error','message':'Selected model is at capacity. Please try a different model.'}),flush=True)
 if os.environ['KYB_FAKE_MODE']!='interrupted':
  print(json.dumps({'type':'turn.failed','error':{'message':'Selected model is at capacity. Please try a different model.'}}),flush=True)
 if os.environ['KYB_FAKE_MODE']=='malformed_failure':print('{interrupted',flush=True)
 raise SystemExit(1)
name=tools[0]['function']['name'];result={'tool':name,'arguments_json':json.dumps({'targets':{'left_z':.9},'note':'Synthetic'} if name=='move_eef' else {})}
out.write_text(json.dumps(result))
reported=prior.get('reported',0)+1
state.write_text(json.dumps({'tools':tools,'turn':turn,'reported':reported}))
turn_id=str(uuid.uuid4());model=args[args.index('-m')+1]
effort=json.loads(next(v.split('=',1)[1] for v in args if v.startswith('model_reasoning_effort=')))
native=Path(os.environ['CODEX_HOME'])/'sessions'/'2026'/'10'/'27'/('rollout-'+sid+'.jsonl')
native.parent.mkdir(parents=True,exist_ok=True)
usage={'input_tokens':7,'cached_input_tokens':2,'cache_write_input_tokens':0,'output_tokens':9,'reasoning_output_tokens':3}
total={k:v*reported for k,v in usage.items()}
with native.open('a') as stream:
 if reported==1:stream.write(json.dumps({'type':'session_meta','payload':{'id':sid}})+'\n')
 stream.write(json.dumps({'type':'turn_context','payload':{'turn_id':turn_id,'model':model,'effort':effort}})+'\n')
 stream.write(json.dumps({'type':'token_usage_record','payload':{'session_id':sid,'thread_id':sid,'turn_id':turn_id,'root_turn_id':turn_id,'response_id':str(uuid.uuid4()),'usage':usage,'thread_token_usage':total}})+'\n')
if os.environ.get('KYB_FAKE_MODE')=='tool':print(json.dumps({'type':'item.completed','item':{'type':'command_execution'}}),flush=True)
print(json.dumps({'type':'turn.completed','usage':{'input_tokens':7*reported,'output_tokens':9*reported}}),flush=True)
'''

class CodexCliTests(unittest.TestCase):
    def setUp(self):
        from unittest.mock import patch
        pacing=patch('kyb.native_dispatch.MIN_START_INTERVAL_SECONDS',.001)
        pacing.start();self.addCleanup(pacing.stop)
        stop=patch.object(CodexCliClient,'_capacity_resume',return_value=False)
        stop.start();self.addCleanup(stop.stop)
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        env=patch.dict(os.environ,{'CODEX_HOME':str(self.root/'codex-home')});env.start();self.addCleanup(env.stop)
        executable=self.root/'fake-codex';executable.write_text(FAKE);executable.chmod(0o700)
        self.executable=str(executable);self.client=CodexCliClient(self.root/'run',executable=self.executable,timeout=3)
        self.tools=[{'function':{'name':'move_eef','parameters':{}}}]
    def tearDown(self):os.environ.pop('KYB_FAKE_MODE',None);os.environ.pop('CODEX_REMOTE_PAYLOAD',None);self.temp.cleanup()
    def received(self,client=None):
        client=client or self.client
        return json.loads((client.root/'cli'/client.last_attempt_id/'received.json').read_text())
    def observation(self,n):
        uri='data:image/jpeg;base64,'+base64.b64encode(f'image{n}'.encode()).decode()
        return {'role':'user','content':[{'type':'text','text':f'state{n}'},
            {'type':'text','text':'RGB_REFERENCE '+json.dumps({'observation_id':f'obs{n}','camera':'head','temporal_role':'new_observation'})},
            {'type':'image_url','image_url':{'url':uri}}]}
    def test_resume_same_sid_only_appends_new_images_and_host_results(self):
        messages=[{'role':'system','content':'action contract'},self.observation(0)]
        response=self.client.complete(messages,self.tools);sid=self.client.session_id
        messages.extend([response['choices'][0]['message'],{'role':'tool','tool_call_id':self.client.last_attempt_id,'content':'actual receipt'},self.observation(1)])
        self.client.complete(messages,self.tools)
        received=self.received();serialized=json.dumps(received['prompt'])
        self.assertEqual(self.client.session_id,sid);self.assertIn('resume',received['argv'])
        self.assertNotIn('robot_tools',received['prompt']);self.assertNotIn('state0',serialized)
        self.assertIn('actual receipt',serialized);self.assertIn('state1',serialized)
        self.assertFalse(any(m['role']=='assistant' for m in received['prompt']['conversation']))
        manifest=json.loads((self.client.root/'cli'/self.client.last_attempt_id/'input/image_manifest.json').read_text())
        self.assertEqual({r['observation_id'] for r in manifest},{'obs1'})
        self.assertEqual(self.client.summary()['cli_invocations'],2)
        self.assertEqual(self.client.summary()['usage_reported'],{'input_tokens':14,'cached_input_tokens':4,'cache_write_input_tokens':0,'output_tokens':18,'reasoning_output_tokens':6})
        self.assertFalse(any(arg.startswith('model_auto_compact_token_limit=') for arg in received['argv']))
    def test_tools_change_and_terminal_new_input_keep_episode_sid(self):
        self.client.complete([self.observation(0)],self.tools);sid=self.client.session_id
        self.client.complete([{'role':'user','content':'Terminal review'}],[{'function':{'name':'finish_review','parameters':{}}}])
        received=self.received();self.assertEqual(self.client.session_id,sid)
        self.assertNotIn('robot_tools',received['prompt'])
        self.assertIn('finish_review',received['instructions'])
    def test_manager_one_sid_per_transaction_and_chosen_model(self):
        client=CodexCliClient(self.root/'management',role='manager',executable=self.executable)
        tools=[{'function':{'name':'publish_materials','parameters':{}}}]
        client.start_transaction();client.complete([{'role':'user','content':'request one'}],tools);sid=client.session_id
        client.start_transaction();client.complete([{'role':'user','content':'request two'}],tools)
        self.assertNotEqual(sid,client.session_id)
        invocation=json.loads((client.root/'cli'/client.last_attempt_id/'invocation.json').read_text())
        self.assertEqual(invocation['model'],'gpt-6.1-sol');self.assertEqual(invocation['effort'],'xhigh')
        self.assertEqual(client.summary()['usage_reported'],{'input_tokens':14,'cached_input_tokens':4,'cache_write_input_tokens':0,'output_tokens':18,'reasoning_output_tokens':6})
    def test_unexpected_native_tool_rejected_without_action_or_retry(self):
        os.environ['KYB_FAKE_MODE']='tool'
        with self.assertRaises(CodexCliFailure):self.client.complete([self.observation(0)],self.tools)
        with self.assertRaises(CodexCliFailure):self.client.complete([self.observation(0)],self.tools)
        self.assertEqual(self.client.summary()['cli_invocations'],1)
    def test_unknown_is_not_retried_or_refunded(self):
        budget=CallBudget(self.root/'allowance.jsonl',2);self.client.budget=budget;self.client.timeout=.2
        os.environ['KYB_FAKE_MODE']='timeout'
        with self.assertRaises(CodexCliFailure):self.client.complete([self.observation(0)],self.tools)
        self.assertEqual(budget.view()['used'],1);self.assertEqual(self.client.summary()['outcome_unknown'],1)
        with self.assertRaises(CodexCliFailure):self.client.complete([self.observation(0)],self.tools)
        self.assertEqual(budget.view()['used'],1)

    def test_explicit_failed_turn_is_charged_but_not_unknown_or_accepted(self):
        budget=CallBudget(self.root/'allowance.jsonl',2);self.client.budget=budget
        os.environ['KYB_FAKE_MODE']='capacity'
        with self.assertRaises(CodexCliFailure):self.client.complete([self.observation(0)],self.tools)
        summary=self.client.summary()
        self.assertEqual((summary['cli_invocations'],summary['completed_turns'],
            summary['structured_responses_accepted'],summary['outcome_unknown'],
            summary['native_failed_turns']),(1,0,0,0,1))
        self.assertEqual(summary['usage_reported'],{})
        with self.assertRaises(CodexCliFailure):self.client.complete([self.observation(0)],self.tools)
        self.assertEqual(budget.view()['used'],1)

    def test_error_without_terminal_failure_remains_unknown(self):
        os.environ['KYB_FAKE_MODE']='interrupted'
        with self.assertRaises(CodexCliFailure):self.client.complete([self.observation(0)],self.tools)
        self.assertEqual(self.client.summary()['outcome_unknown'],1)
        self.assertEqual(self.client.summary()['native_failed_turns'],0)

    def test_malformed_failure_trace_is_not_promoted_to_known_failure(self):
        os.environ['KYB_FAKE_MODE']='malformed_failure'
        with self.assertRaises(CodexCliFailure):self.client.complete([self.observation(0)],self.tools)
        self.assertEqual(self.client.summary()['outcome_unknown'],1)
        self.assertEqual(self.client.summary()['native_failed_turns'],0)
    def test_shared_persistent_budget_includes_both_roles_and_survives_restart(self):
        budget=CallBudget(self.root/'allowance.jsonl',2);self.client.budget=budget
        manager=CodexCliClient(self.root/'manager',role='manager',executable=self.executable,budget=budget)
        self.client.complete([self.observation(0)],self.tools)
        manager.complete([{'role':'user','content':'management'}],[{'function':{'name':'publish_materials','parameters':{}}}])
        restarted=CallBudget(budget.path,2);self.assertEqual(restarted.view()['remaining'],0)
        with self.assertRaises(BudgetExhausted):self.client.complete([self.observation(1)],self.tools)
        self.assertEqual(self.client.summary()['cli_invocations'],1)
        with self.assertRaises(ValueError):CallBudget(budget.path,3)
    def test_manager_accounting_does_not_consume_primary_only_allowance(self):
        budget=CallBudget(self.root/'primary.jsonl',1);self.client.budget=budget
        manager=CodexCliClient(self.root/'manager',role='manager',executable=self.executable)
        self.client.complete([self.observation(0)],self.tools)
        manager.complete([{'role':'user','content':'management'}],[{'function':{'name':'publish_materials','parameters':{}}}])
        self.assertEqual(budget.view()['used'],1);self.assertEqual(budget.view()['roles_recorded'],['policy'])
        self.assertEqual(manager.summary()['cli_invocations'],1)
        with self.assertRaises(BudgetExhausted):self.client.complete([self.observation(1)],self.tools)
    def test_parent_auth_payload_not_inherited_and_context_not_duplicated(self):
        os.environ['CODEX_REMOTE_PAYLOAD']='private_parent_payload'
        message=self.observation(0);message['content'][0]['text']+='\n<KYB_CONTEXT>\n'+json.dumps({'current_observation_id':'obs0'})+'\n</KYB_CONTEXT>'
        self.client.complete([message],self.tools)
        received=self.received();self.assertFalse(received['parent_payload_inherited'])
        self.assertNotIn('<KYB_CONTEXT>',json.dumps(received['prompt']['conversation']))
        self.assertEqual(received['prompt']['current_readout']['current_observation_id'],'obs0')

    def test_stable_contract_and_active_knowledge_are_resident_not_repeated_user_text(self):
        import hashlib
        self.client.knowledge_provider=lambda:{'knowledge':[{'id':'k1','text':'Current scoped relation'}]}
        messages=[{'role':'system','content':'Permanent action contract'},self.observation(0)]
        self.client.complete(messages,self.tools)
        self.client.complete([{'role':'user','content':'Next result'}],self.tools)
        directory=self.client.root/'cli'/self.client.last_attempt_id
        instructions=(directory/'input/instructions.txt').read_text()
        self.assertIn('Permanent action contract',instructions)
        self.assertIn('Current scoped relation',instructions)
        self.assertNotIn('Permanent action contract',json.dumps(self.received()['prompt']))
        preparation=json.loads((directory/'input/preparation.json').read_text())
        encoded=json.dumps(preparation,ensure_ascii=False)
        metrics=json.loads((directory/'input/context_metrics.json').read_text())
        self.assertEqual(metrics['preparation_sha256'],hashlib.sha256(encoded.encode()).hexdigest())
        actual=instructions.split('CURRENT STABLE PREPARATION (not current physical state):\n',1)[1].split('\nAVAILABLE HOST TOOLS',1)[0]
        self.assertEqual(actual,encoded)
        self.assertNotIn('delivery_manifest',instructions)
