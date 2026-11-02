"""Orchestration with real SQLite snapshots, mocked simulator transport only."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import continual_experiment as study
from kyb.memory import MemoryStore
from kyb.trace import RunTrace

def transport(unknown=0):
    return {'cli_invocations':0,'completed_turns':0,'structured_responses_accepted':0,'native_failed_turns':0,
            'outcome_unknown':unknown,'usage_reported':{'input_tokens':0,'cached_input_tokens':0,'output_tokens':0,'reasoning_output_tokens':0}}

class Continual(unittest.TestCase):
    def make_root(self,root):
        root.mkdir();study.save(root/'HARNESS.json',{'version':'TEST_ONLY','files':{}})
        study.save(root/'LAYOUT_FILES.json',[]);study.save(root/'EPISODE_ASSETS.json',{str(i):[] for i in range(1,21)})
        study.save(root/'LAYOUTS.json',{'tasks':[{'order':i,'task':'play_stacking_toy','layout':i,'native_reset_seed':i} for i in range(1,21)]})
        store=MemoryStore(root/'seed_work.sqlite');study.library.initialize(store);study.body_index.initialize(store);study.leads.initialize(store)
        trace=RunTrace(root/'seed_trace',{});payload={'scope':'SIMULATED_PRIOR'};store.register(trace.event('known_config',payload),payload)
        seed=root/'initial.sqlite';store.freeze(seed);store.close()
        study.save(root/'PROTOCOL.json',{'harness_id':'test','primary':{'calls_per_episode':150},'manager':{'calls_per_episode':100},
            'episodes':20,'prior_episodes':30,'runtime':{'root':str(root/'runtime'),'server_port':19101,'gpu':0},'initial_memory_snapshot':str(seed),'initial_memory_counts':study.memory_counts(seed)})
    def launch(self,data,seen,unknown=False):
        def fake(command,*,cwd,env,stdout,stderr,pass_fds):
            self.assertEqual(len(pass_fds),1)
            self.assertNotIn('--episode',command)
            self.assertEqual(command[command.index('--call-budget')+1],'150')
            self.assertEqual(command[command.index('--manager-call-budget')+1],'100')
            index=int(command[command.index('--layout')+1]);source=Path(command[command.index('--input-memory')+1])
            seen.append((index,str(source),study.sha(source),study.memory_counts(source)))
            root=data/'runs'/env['KYB_RUN_ID'];trace=RunTrace(root,{})
            store,_=MemoryStore.fork_snapshot(source,root/'memory.sqlite');payload={'index':index}
            store.register(trace.event('known_config',payload),payload);store.freeze(root/'final_memory.sqlite');store.close()
            for flag in ('--budget-ledger','--manager-budget-ledger'):Path(command[command.index(flag)+1]).touch()
            study.save(root/'report.json',{'status':'PASS','sources_unchanged':True,'task_outcome':'PASS' if index%2 else 'NEEDS_WORK',
                'model_transport':transport(),'management_transport':transport(int(unknown)),'terminal_review':{},
                'env_steps':index,'env_step_limit':1200,'wall_seconds':0})
            class Process:
                pid=12345
                def wait(self):return 0
            return Process()
        return fake
    def test_twenty_serial_episodes_preserve_own_snapshot_chain_and_stop_at_twenty(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);root=base/'experiment';self.make_root(root);seen=[]
            with patch.object(study,'D',base/'data'),patch.object(study,'plot_curve'),patch.object(study,'trace_summary',return_value={
                'native_sessions':{'primary':{},'manager':{}},'primary_decisions':[],'evaluation':{'official_end_flag':True}}), \
                patch.object(study.subprocess,'Popen',side_effect=self.launch(base/'data',seen)):
                study.run(root)
            state=study.load(root/'STATE.json');self.assertEqual(state['status'],'PASS')
            self.assertEqual([r[0] for r in seen],list(range(1,21)));self.assertEqual([r[3]['evidence'] for r in seen],list(range(1,21)))
            for previous,next_input in zip(state['completed'],seen[1:]):
                self.assertEqual(previous['final_memory'],next_input[1]);self.assertEqual(previous['final_memory_sha256'],next_input[2])
            with self.assertRaises(RuntimeError):study.run(root)  # No eleventh/restarted study.
    def test_unknown_outcome_pauses_without_next_episode_or_retry(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);root=base/'experiment';self.make_root(root);seen=[]
            with patch.object(study,'D',base/'data'),patch.object(study,'plot_curve'), \
                patch.object(study.subprocess,'Popen',side_effect=self.launch(base/'data',seen,True)):
                with self.assertRaisesRegex(ValueError,'unknown'):study.run(root)
            self.assertEqual(len(seen),1);self.assertEqual(study.load(root/'STATE.json')['status'],'PAUSED')
    def test_role_tokens_include_failures_without_double_counting_subfields(self):
        row={'order':1,'layout':1,'native_reset_seed':1,'task_outcome':'NEEDS_WORK','recording':'PASS',
            'termination':'primary_allowance_boundary','env_steps':950,'env_step_limit':1200,
            'memory_counts':{'cases':105,'knowledge':20,'knowledge_active':8},
            'primary':{'cli_invocations':100,'usage_reported':{'input_tokens':100,'cached_input_tokens':60,'output_tokens':10,'reasoning_output_tokens':3}},
            'manager':{'cli_invocations':20,'usage_reported':{'input_tokens':50,'cached_input_tokens':30,'output_tokens':20,'reasoning_output_tokens':5}}}
        for role in ('primary','manager'):row[role]['captured_metadata_consistent']=True
        metrics=study.metric_rows([row],30)[0]
        self.assertEqual(metrics['total_tokens'],180);self.assertEqual(metrics['continual_order'],31)
        self.assertEqual(metrics['knowledge_active'],8);self.assertEqual(metrics['success'],0)
