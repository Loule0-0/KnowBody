"""Observation authority and bounded packets, without models or simulator."""
import json
import tempfile
import unittest
from pathlib import Path
from kyb.memory import MemoryStore
from kyb.trace import RunTrace
from kyb.session import MemorySession
from kyb.preparation import pack, PREPARATION_BYTES


class PackTests(unittest.TestCase):
    def test_library_growth_keeps_complete_conditions_in_one_bounded_packet(self):
        cards=[{'id':str(i),'revision':i,'family':'geometry','strength':'observation',
            'question':'抓持转姿'*60,'text':'evidence'*150,'conditions':{'grip':'flat head','when':'clearance checked'},
            'use':'Check orientation','unknowns':['Unobserved clearance'],'params':[]} for i in range(1000)]
        candidate={'id':'old_episode','stages':[{'summary':'transport '*100,'conditions':{'grip':'head'},'unknowns':[]}]*4,
            'qualification':'Unreviewed; not a certified successful route.'}
        packet=pack({'version':1},cards,[candidate],{'statements':[],'index':[]})
        actual=len(json.dumps(packet,ensure_ascii=False).encode())
        self.assertLessEqual(actual,PREPARATION_BYTES)
        self.assertEqual(actual,packet['packing']['serialized_bytes'])
        self.assertLessEqual(len(packet['memory_index']),4)
        self.assertGreater(packet['packing']['omitted_knowledge'],990)
        for card in packet['knowledge']:
            self.assertEqual(card['conditions'],cards[int(card['id'])]['conditions'])
            self.assertEqual(card['unknowns'],['Unobserved clearance'])
        for episode in packet['episode_candidates']:
            for actual,source in zip(episode['stages'],candidate['stages']):
                self.assertEqual(actual['summary'],source['summary'])
                self.assertEqual(actual['conditions'],source['conditions'])
                self.assertEqual(actual['unknowns'],source['unknowns'])

    def test_operator_goal_information_precedes_learned_material(self):
        amendment={'id':'clarification','answer':'Press 2 then 1','question':'Which order?','reuse_scope':'episode'}
        p=pack({},[],[],{'statements':[amendment],'index':[]})
        self.assertEqual(p['operator_clarifications']['statements'],[amendment])

    def test_authoritative_contract_is_complete_outside_experience_allowance(self):
        amendments={'statements':[{'id':str(i),'answer':'明确要求'*1000,'question':'How?',
            'reuse_scope':'episode'} for i in range(3)]}
        p=pack({},[],[],amendments)
        self.assertEqual(p['operator_clarifications'],amendments)
        self.assertGreater(len(json.dumps(p,ensure_ascii=False).encode()),PREPARATION_BYTES)
        self.assertLessEqual(p['packing']['experience_serialized_bytes'],PREPARATION_BYTES)

    def test_oversized_process_retains_explicit_conditional_navigation(self):
        candidate={'id':'episode','revision':1,'task':'stack','reviewed':False,
            'stages':[{'summary':'x'*9000,'conditions':{'grip':'head'},'unknowns':[]}],
            'qualification':'Unreviewed candidate, not physical evidence.'}
        p=pack({},[],[candidate],{'statements':[]})
        self.assertEqual(p['episode_candidates'],[])
        self.assertEqual(p['episode_index'][0]['read'],{'level':'episode','ids':['episode']})
        self.assertFalse(p['episode_index'][0]['physical_support'])

    def test_nearly_full_knowledge_cannot_prevent_process_navigation_or_pause(self):
        k={'id':'k','revision':1,'question':'A question','text':'x'*1800,
            'conditions':{'clearance':'x'*5560},'unknowns':['Unknown response'],'use':'Check before use.'}
        episode={'id':'e','revision':1,'task':'stack','reviewed':False,'stages':[{'summary':'x'*9000}]}
        p=pack({},[k],[episode],{'statements':[]})
        self.assertLessEqual(p['packing']['experience_serialized_bytes'],PREPARATION_BYTES)
        self.assertTrue(p['episode_index'])


class RgbAuthorityTests(unittest.TestCase):
    def test_import_and_manager_reads_reject_privileged_rows_even_with_rgb_label(self):
        from kyb.perception import RGB_ONLY
        from manager_fixtures import PreparationClient
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);store=MemoryStore(root/'memory.sqlite')
            session=MemorySession(RunTrace(root/'rgb',{}),store,task='stack',dependencies={},perception_profile='rgb_only')
            session.evidence('known_config',{'action_spec':'public'})
            session.attach_manager(PreparationClient())
            m=session.manager;m.frontier=session.source_frontier();m.purpose='review'
            for kind,payload in [('known_config',{'perception_profile':'rgb_only','RGB_camera_identity':{'K':123}}),
                ('surface_inspection',{'world_xyz_m':[1,2,3]})]:
                # Bypass the new-write guard, as a contaminated imported DB could.
                ref=session.trace.event(kind,payload);store.register(ref,payload)
                m.frontier=session.source_frontier()
                with self.assertRaises(ValueError):RGB_ONLY.check_memory(store)
                with self.assertRaises(ValueError):m._record(ref['id'],body='full')
            store.close()


class ContractAndIdentityTests(unittest.TestCase):
    def test_full_operator_amendment_reaches_both_roles_and_resident_cli_on_resume(self):
        from manager_fixtures import PreparationClient
        from kyb.codex_cli import CodexCliClient
        from test_codex_cli import FAKE
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);store=MemoryStore(root/'memory.sqlite')
            session=MemorySession(RunTrace(root/'episode',{}),store,task='stack',dependencies={})
            session.task_specification=session.evidence('task_specification',{'contract_id':'exact_contract'})
            answer='Goal text '*700+'FINAL_GOAL_REQUIREMENT'
            session.evidence('task_contract_amendment',{'contract_id':'exact_contract','episode':session.episode,
                'reuse_scope':'episode','question':'Which goal?','answer':answer,'request_id':'explicit_operator_answer'})
            client=PreparationClient();session.attach_manager(client)
            session.manager.run('Prepare the task','prepare')
            manager_contract=client.inputs[0][0]['content']
            self.assertIn(answer,manager_contract)
            self.assertNotIn(answer,client.inputs[0][1]['content'])
            fake=root/'fake-codex';fake.write_text(FAKE);fake.chmod(0o700)
            cli=CodexCliClient(root/'cli_run',executable=str(fake))
            cli.knowledge_provider=session.manager.prefix_view
            tools=[{'function':{'name':'finish_review','parameters':{}}}]
            messages=[{'role':'user','content':'Current public task'}]
            cli.complete(messages,tools);sid=cli.session_id
            messages.append({'role':'user','content':'A later step after native compaction may use resident context.'})
            cli.complete(messages,tools)
            self.assertEqual(cli.session_id,sid)
            instructions=(cli.root/'cli'/cli.last_attempt_id/'input/instructions.txt').read_text()
            self.assertIn(answer,instructions)
            management=CodexCliClient(root/'manager_run',executable=str(fake),role='manager')
            management.start_transaction()
            manager_messages=[{'role':'system','content':manager_contract},{'role':'user','content':'Request materials'}]
            management.complete(manager_messages,tools);manager_sid=management.session_id
            manager_messages.append({'role':'user','content':'Continue the same management transaction.'})
            management.complete(manager_messages,tools)
            self.assertEqual(manager_sid,management.session_id)
            manager_instructions=(management.root/'cli'/management.last_attempt_id/'input/instructions.txt').read_text()
            self.assertIn(answer,manager_instructions)
            other=MemorySession(RunTrace(root/'another',{}),store,task='stack',dependencies={})
            other.task_specification=other.evidence('task_specification',{'contract_id':'exact_contract'})
            from kyb.contracts import amendment_view
            self.assertEqual(amendment_view(other)['statements'],[])
            store.close()

    def test_execution_identity_covers_kernel_closure_and_rejects_interface_roots(self):
        from kyb.execution_contract import build,local_closure
        import kyb.policy
        args=dict(control_hz=25,physics_dt=.004,effective_parameters={'arm_step':.05},action_contract='public frame')
        contract,identity=build(**args)
        self.assertIn('kyb/motion_core.py',contract['motion_sources'])
        self.assertIn('kyb/body_guard.py',contract['motion_sources'])
        self.assertNotIn('kyb/policy.py',contract['motion_sources'])
        self.assertEqual(build(**args)[1],identity)
        self.assertNotEqual(build(**{**args,'control_hz':30})[1],identity)
        with self.assertRaisesRegex(ValueError,'interaction'):
            local_closure([Path(kyb.policy.__file__)])

    def test_empty_chain_collects_rgb_proprio_but_rejects_privileged_payloads(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);store=MemoryStore(root/'memory.sqlite')
            session=MemorySession(RunTrace(root/'rgb',{}),store,task='stack',dependencies={'body':'b'},perception_profile='rgb_only')
            session.evidence('known_config',{'action_spec':'public EEF state and robot geometry'})
            session.evidence('observation',{'images':{},'state':{},'dependencies':{'body':'b'}})
            self.assertIn('No depth or camera',session.instructions())
            self.assertNotIn('Use inspect_surface',session.instructions())
            self.assertIn('unavailable',session.inspect_surface({},None))
            self.assertIn('unavailable',session.read_body({}))
            for kind,data in [('observation',{'body_visual_reference':{'K':123}}),
                ('observation',{'depth_snapshots':{'head':123}}),('known_config',{'RGB_camera_identity':{'extrinsics':123}}),
                ('surface_inspection',{'samples':[]})]:
                with self.assertRaises(ValueError):session.evidence(kind,data)
            store.close()

    def test_rgb_cannot_import_enhanced_library_even_if_queries_are_hidden(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);store=MemoryStore(root/'memory.sqlite')
            enhanced=MemorySession(RunTrace(root/'enhanced',{}),store,task='stack',dependencies={})
            enhanced.evidence('known_config',{'action_spec':'public'})
            with self.assertRaisesRegex(ValueError,'independent'):
                MemorySession(RunTrace(root/'rgb',{}),store,task='stack',dependencies={},perception_profile='rgb_only')
            store.close()
