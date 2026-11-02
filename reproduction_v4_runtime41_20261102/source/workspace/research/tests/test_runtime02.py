"""Changed budget, source-binding, serial-boundary and passive-accounting behavior."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import main_experiment as main
import continual_experiment as single
import experiment_monitor as monitor
from kyb.runtime import call_limits
from kyb.memory import MemoryStore
from kyb.trace import RunTrace

def transport(n=0):
    return {'cli_invocations':n,'structured_responses_accepted':n,'outcome_unknown':0,'usage_reported':{'input_tokens':0,'output_tokens':0}}

class Runtime02(unittest.TestCase):
    def setup_experiment(self,base):
        root=base/'experiment';root.mkdir();store=MemoryStore(base/'source_work.sqlite');main.library.initialize(store)
        trace=RunTrace(base/'source_trace',{});store.register(trace.event('known_config',{'source':True}),{'source':True});seed=base/'source.sqlite';store.freeze(seed);store.close()
        tasks=[{'order':i,'task':'task'+str(i),'layout':i} for i in range(1,43)]
        main.save(root/'PROTOCOL.json',{'round':3,'primary':{'calls_per_episode':100},'manager':{'calls_per_episode':100},
            'runtime':{'root':str(base/'runtime'),'server_port':19099,'gpu':0},'input_memory':str(seed),
            'input_memory_sha256':main.sha(seed),'initial_memory_counts':main.memory_counts(seed),'task_order':[t['task'] for t in tasks]})
        main.save(root/'HARNESS.json',{'version':'TEST_ONLY','files':{}});main.save(root/'LAYOUTS.json',{'tasks':tasks});main.save(root/'LAYOUT_FILES.json',[])
        return root,seed
    def invoke(self,root,seed,extra=()):
        with patch('sys.argv',['main_experiment','--experiment-dir',str(root),'--round','3','--initial-memory',str(seed),*extra]):main.main()
    def fake_process(self,base,seen,unknown=False):
        def launch(command,*,cwd,env,stdout,stderr,pass_fds):
            self.assertEqual(len(pass_fds),1)
            self.assertEqual(command[command.index('--call-budget')+1],'100');self.assertEqual(command[command.index('--manager-call-budget')+1],'100')
            source=Path(command[command.index('--input-memory')+1]);seen.append((str(source),main.sha(source)))
            run=base/'data/runs'/env['KYB_RUN_ID'];trace=RunTrace(run,{});store,_=MemoryStore.fork_snapshot(source,run/'memory.sqlite')
            payload={'sequence':len(seen)};store.register(trace.event('known_config',payload),payload);store.freeze(run/'final_memory.sqlite');store.close()
            for flag in ['--budget-ledger','--manager-budget-ledger']:Path(command[command.index(flag)+1]).touch()
            p=transport();p['outcome_unknown']=int(unknown)
            main.save(run/'report.json',{'status':'PASS','sources_unchanged':True,'task_outcome':'PASS' if len(seen)%2 else 'NEEDS_WORK',
                'model_transport':p,'management_transport':transport(),'terminal_review':{},'env_steps':10,'env_step_limit':100})
            class Process:
                pid=12345
                def wait(self):return 0
            return Process()
        return launch
    def test_actual_main_driver_finishes_42_and_uses_own_predecessors(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);root,seed=self.setup_experiment(base);seen=[]
            with patch.object(main,'D',base/'data'),patch.object(main,'ensure_assets'),patch.object(main.subprocess,'Popen',side_effect=self.fake_process(base,seen)):
                self.invoke(root,seed)
            state=main.load(root/'round_03/STATE.json');self.assertEqual(state['status'],'PASS');self.assertEqual(len(seen),42)
            for previous,next_input in zip(state['completed'],seen[1:]):self.assertEqual((previous['final_memory'],previous['final_memory_sha256']),next_input)
            with self.assertRaisesRegex(RuntimeError,'not cleanly ready'):self.invoke(root,seed)
    def test_preflight_has_no_calls_and_stop_is_before_launch(self):
        with tempfile.TemporaryDirectory() as td:
            root,seed=self.setup_experiment(Path(td))
            with patch.object(main,'ensure_assets'),patch.object(main.subprocess,'Popen') as process:
                self.invoke(root,seed,['--preflight-only']);process.assert_not_called()
                (root/'STOP').touch()
                with self.assertRaisesRegex(RuntimeError,'episode boundary'):self.invoke(root,seed)
                process.assert_not_called()
            state=main.load(root/'round_03/STATE.json');self.assertEqual(state['status'],'PAUSED');self.assertIsNone(state['active'])
    def test_main_unknown_outcome_does_not_advance(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);root,seed=self.setup_experiment(base);seen=[]
            with patch.object(main,'D',base/'data'),patch.object(main,'ensure_assets'),patch.object(main.subprocess,'Popen',side_effect=self.fake_process(base,seen,True)):
                with self.assertRaisesRegex(ValueError,'unknown'):self.invoke(root,seed)
            state=main.load(root/'round_03/STATE.json');self.assertEqual(len(seen),1);self.assertEqual(state['status'],'PAUSED');self.assertEqual(state['completed'],[])
    def test_source_round_and_exact_task_order_are_bound_to_protocol(self):
        with tempfile.TemporaryDirectory() as td:
            root,seed=self.setup_experiment(Path(td));p=main.load(root/'PROTOCOL.json')
            for key,value,reason in [('round',4,'Round/source'),('input_memory_sha256','changed','initial memory changed'),('task_order',list(reversed(p['task_order'])),'42-task order')]:
                changed={**p,key:value};main.save(root/'PROTOCOL.json',changed)
                with patch.object(main.subprocess,'Popen') as process:
                    with self.assertRaisesRegex(ValueError,reason):self.invoke(root,seed)
                    process.assert_not_called()
    def test_150_role_gate_is_distinct_from_physics_timeout(self):
        limits=call_limits({'primary':{'calls_per_episode':150},'manager':{'calls_per_episode':100}})
        report={'status':'PASS','sources_unchanged':True,'task_outcome':'NEEDS_WORK','model_transport':transport(150),'management_transport':transport(100),'terminal_review':{},'env_steps':900,'env_step_limit':1200,'allowance_action_stop':True}
        single.validate_closed(report,150,100,limits)
        self.assertEqual(single.termination(report,{'primary_decisions':[],'evaluation':{}}),'primary_allowance_boundary')
        report['env_steps']=1200;self.assertEqual(single.termination(report,{'primary_decisions':[],'evaluation':{}}),'official_environment_horizon')
        with self.assertRaises(ValueError):single.validate_closed(report,150,100,{'primary':100,'manager':100})
        with self.assertRaises(ValueError):call_limits({'primary':{'calls_per_episode':True},'manager':{'calls_per_episode':100}})
    def test_passive_steps_join_the_actual_intent_and_do_not_read_private(self):
        with tempfile.TemporaryDirectory() as td:
            run=Path(td)/'run';trace=RunTrace(run,{})
            trace.event('policy_decision',{'tool':'move_eef'});intent=trace.event('action_intent',{'call_id':'a'});trace.event('receipt',{'intent':intent['id'],'executed_actions':17})
            trace.event('policy_decision',{'tool':'hold'});intent=trace.event('action_intent',{'call_id':'b'});trace.event('receipt',{'intent':intent['id'],'executed_actions':3})
            trace.event('policy_decision',{'tool':'PRIVATE_MUST_NOT_COUNT'},visibility='private')
            state={'through':0,'tools':{},'call_tools':{},'physical_steps':{},'questions':{},'query_views':[],'knowledge_updates':0,'case_updates':0,'manager_rejections':{}}
            monitor.update_public(run,state);self.assertEqual(state['physical_steps'],{'move_eef':17,'hold':3});self.assertNotIn('PRIVATE_MUST_NOT_COUNT',state['tools'])
            monitor.update_public(run,state);self.assertEqual(sum(state['physical_steps'].values()),20)
