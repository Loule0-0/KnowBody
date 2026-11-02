"""v4 authority, incremental updates and complete offline loop. No paid calls or simulator."""
import hashlib
import json
import threading
import time
import unittest
from unittest.mock import patch
from pathlib import Path

import test_promotion as p
import test_policy_pipeline as pipeline
from manager_fixtures import PreparationClient
from manager_fixtures import ScriptedManager
from kyb import episodes, library
from kyb.clarification import request, answer
from kyb.knowledge import read_experience
from kyb.memory import MemoryStore
from kyb.policy import EefAgentPolicy
from kyb.session import MemorySession
from kyb.terminal import review_terminal
from kyb.trace import RunTrace


class V4Contracts(unittest.TestCase):
    setUp=p.PromotionTests.setUp
    tearDown=p.PromotionTests.tearDown
    observation=p.PromotionTests.observation
    add_body_case=p.PromotionTests.add_body_case
    op=p.PromotionTests.op
    canonical_ref=p.PromotionTests.canonical_ref
    write=p.PromotionTests.write
    attach=p.PromotionTests.attach
    manager=p.PromotionTests.manager
    read_case=p.PromotionTests.read_case

    def test_missing_declared_static_pin_resolves_only_exact_historical_context(self):
        a=self.add_body_case(depends_on=['body','depth'])
        b=self.add_body_case(depends_on=['body'],text='Second limited relation without a declared depth dependency.')
        self.store.annotate_observation(self.post['id'],{'body':'new','depth':'mutable_wrong'})
        with self.store.db:k=self.write(self.op([self.canonical_ref(a),self.canonical_ref(b)],depends_on=['body','depth']),[a,b])
        row=self.store.db.execute('SELECT dependencies FROM reusable_knowledge WHERE revision=?',(k['revision'],)).fetchone()
        self.assertEqual(json.loads(row[0]),{'body':'b1','depth':'d1'})
        basis=json.loads(self.store.db.execute('SELECT basis FROM knowledge_contract_basis WHERE revision=?',(k['revision'],)).fetchone()[0])
        self.assertEqual(basis['depth']['declared_by'],[{'id':a['claim_id'],'revision':a['revision']}])
        self.assertEqual(basis['depth']['context_from'][0]['sources'],[self.post['id']])
        with self.assertRaisesRegex(ValueError,'at least one'):
            self.write(self.op([self.canonical_ref(b)],depends_on=['body','depth']),[b])
        source=self.observation(dependencies={**self.deps,'depth':'conflicting'})
        c=self.add_body_case(source,depends_on=['body'],text='Third relation in a conflicting historical depth context.')
        with self.assertRaisesRegex(ValueError,'conflicts'):
            self.write(self.op([self.canonical_ref(a),self.canonical_ref(c)],depends_on=['body','depth']),[a,c])

    def test_dynamic_binding_missing_or_unknown_is_never_context_completed(self):
        a=self.add_body_case(depends_on=['body','attachment:right']);b=self.add_body_case(depends_on=['body'],text='Another relation without a binding dependency.')
        with self.assertRaisesRegex(ValueError,'dynamic'):
            self.write(self.op([self.canonical_ref(a),self.canonical_ref(b)],depends_on=['body','attachment:right']),[a,b])

    def test_append_preserves_statement_rejects_replacement_and_contests_counter(self):
        a=self.add_body_case();b=self.add_body_case(text='Counterexample under the claimed conditions.')
        with self.store.db:k=self.write(self.op([self.canonical_ref(a)]),[a])
        old=dict(self.store.db.execute('SELECT * FROM reusable_knowledge WHERE revision=?',(k['revision'],)).fetchone())
        def append(ref,version=k['revision']):
            return library.publish_knowledge(self.store,{'op':'append_refs','id':k['id'],'refs':[ref]},
                allowed=set(),visible_cases={b['claim_id']:b['revision']},visible_knowledge={k['id']:version},
                dependencies={'body':'wrong_current'},frontier=self.session.source_frontier())
        with self.store.db:same=append(self.canonical_ref(a))
        self.assertTrue(same['unchanged'])
        with self.assertRaisesRegex(ValueError,'cannot change'):append({**self.canonical_ref(a),'relation':'related'})
        with self.store.db:updated=append(self.canonical_ref(b,'counter'))
        new=dict(self.store.db.execute('SELECT * FROM reusable_knowledge WHERE revision=?',(updated['revision'],)).fetchone())
        for key in old.keys()-{'revision','refs'}:self.assertEqual(new[key],old[key])
        self.assertEqual(json.loads(new['refs'])[0],self.canonical_ref(a))
        card=library.read_knowledge(self.store,'',ids=[k['id']],context={},dependencies=self.deps)[0]
        self.assertTrue(card['source_contested']);self.assertFalse(card['default_eligible'])
        self.assertNotIn(a['claim_id'],{b['claim_id']:b['revision']})

    def test_receipt_pack_delivers_exact_sources_and_failed_pack_grants_nothing(self):
        m=self.manager();m.purpose='review'
        result,_=m._evidence({'receipts':[self.receipt['id']]})
        self.assertEqual(result['delivered_sources'],[self.before['id'],self.receipt['id'],self.post['id']])
        self.assertFalse(m.allowed)
        m._admit(result)
        self.assertEqual(m.allowed,set(result['delivered_sources']))
        before=set(m.allowed)
        with self.assertRaises(ValueError):m._evidence({'receipts':[self.receipt['id'],'not_registered']})
        self.assertEqual(m.allowed,before);self.assertFalse(m.seen_images)

    def test_receipt_pack_rejects_another_runs_spurious_post(self):
        m=self.manager();m.purpose='review'
        second=RunTrace(Path(self.temp.name)/'other_run',{})
        other=MemorySession(second,self.store,task='other',dependencies=self.deps)
        receipt=self.session.evidence('receipt',{'before_observation':self.post['id'],'executed_actions':1})
        other.evidence('observation',{'after_receipt':receipt['id'],'dependencies':self.deps})
        m.frontier=self.session.source_frontier()
        with self.assertRaisesRegex(ValueError,'no registered post'):m._evidence({'receipts':[receipt['id']]})
        self.assertFalse(m.allowed)

    def test_full_pack_cost_hint_preserves_requested_view_and_permissions(self):
        before=self.observation(body_visual_reference={'padding':'x'*30000})
        receipt=self.session.evidence('receipt',{'before_observation':before['id'],'executed_actions':1})
        self.observation(after_receipt=receipt['id'],body_visual_reference={'padding':'x'*30000})
        m=self.manager();m.purpose='review';m.frontier=self.session.source_frontier()
        with self.assertRaisesRegex(ValueError,'explicitly choose body=ee'):
            m._evidence({'receipts':[receipt['id']],'body':'full'})
        self.assertFalse(m.allowed);self.assertFalse(m.seen_images);self.assertFalse(m.seen_telemetry)
        result,_=m._evidence({'receipts':[receipt['id']],'body':'ee'})
        self.assertNotIn('padding',json.dumps(result))

    def test_secondary_missing_contract_stays_strict_and_utf8_error_names_actual_size(self):
        case=self.add_body_case(depends_on=['body','depth'])
        raw=self.session.evidence('surface_inspection',{'dependencies':{'body':'b1'}})
        operation=self.op([self.canonical_ref(case),{'type':'evidence','id':raw['id']}],depends_on=['body','depth'])
        with self.assertRaisesRegex(ValueError,'Secondary source contracts differ or are missing'):
            library.publish_knowledge(self.store,operation,allowed={raw['id']},visible_cases={case['claim_id']:case['revision']},visible_knowledge={},dependencies=self.deps,frontier=self.session.source_frontier())
        with self.assertRaisesRegex(ValueError,'question: 603 UTF-8 bytes; maximum600'):
            self.write(self.op([self.canonical_ref(case)],question='问'*201),[case])

    def test_omitted_terminal_outline_does_not_consume_later_interpretation(self):
        m=self.manager();self.session.manager.client=ScriptedManager([('publish_materials',{'expected_version':1,'answer':'Recorded interpretation','groups':[]})])
        episodes.begin(self.session,{'prefix_replay':None},True)
        result=json.loads(self.session.manage_memory({'purpose':'learn','question':'Review the useful process'}))
        self.assertEqual(result['status'],'NEEDS_WORK');self.assertFalse(self.session.manager.client.inputs)
        row=self.store.db.execute('SELECT outline_ref,result_ref FROM episode_records').fetchone()
        self.assertEqual(tuple(row),(None,None))
        outline=[{'summary':'A limited test interaction','first_receipt':self.receipt['id'],'last_receipt':self.receipt['id'],'conditions':{},'unknowns':['Physical object response unverified']}]
        result=json.loads(self.session.manage_memory({'purpose':'learn','question':'Review the useful process','episode_outline':outline}))
        self.assertEqual(result['status'],'PASS')
        self.assertEqual(episodes.read(self.store)[0]['stages'],outline)

    def test_old_unknown_facets_remain_candidates_but_different_known_object_does_not(self):
        a=self.add_body_case()
        with self.store.db:k=self.write(self.op([self.canonical_ref(a)]),[a])
        kwargs=dict(context={},dependencies=self.deps,domain='body',family='geometry',roles={'object':'key'})
        self.assertEqual(library.read_knowledge(self.store,'',**kwargs)[0]['id'],k['id'])
        self.store.db.execute('UPDATE case_navigation SET roles=? WHERE revision=?',(json.dumps({'object':'alarm_clock'}),a['revision']))
        self.assertEqual(library.read_knowledge(self.store,'',**kwargs),[])


class V4Clarification(unittest.TestCase):
    setUp=p.PromotionTests.setUp
    tearDown=p.PromotionTests.tearDown
    observation=p.PromotionTests.observation

    def test_live_wait_keeps_same_observation_no_calls_and_accepts_exact_operator_answer(self):
        self.session.task_specification=self.session.evidence('task_specification',{'instruction':'Press buttons in the specified order.','public_recipe':'Read the ordered digits.'})
        args={'question':'Which public order applies?','missing_information':'The digit order is not supplied.',
            'interpretations':['Order printed left to right','Order specified by a separate list'],
            'quotes':[{'source_id':self.session.task_specification['id'],'quote':'Read the ordered digits.'}]}
        before=self.session.current_observation['id'];receipt=self.session.last_receipt['id']
        def provide_answer():
            state=self.trace.root/'clarification'/'STATE.json'
            for _ in range(300):
                if state.exists():break
                time.sleep(.01)
            answer(self.trace.root,'Use the displayed sequence from left to right.')
        thread=threading.Thread(target=provide_answer);thread.start()
        result=json.loads(request(self.session,args,poll_seconds=.01));thread.join()
        self.assertEqual(result['executed_actions'],0);self.assertEqual(self.session.current_observation['id'],before)
        self.assertEqual(self.session.last_receipt['id'],receipt)
        self.assertEqual(self.store.db.execute("SELECT COUNT(*) FROM evidence WHERE kind='policy_decision'").fetchone()[0],0)
        self.assertEqual(self.store.evidence_payload(result['contract_amendment'])['provided_by'],'operator')
        bad={**args,'quotes':[{'source_id':self.session.task_specification['id'],'quote':'a hidden evaluator target'}]}
        with self.assertRaisesRegex(ValueError,'absent'):request(self.session,bad)

    def test_only_explicit_scope_and_same_public_contract_can_cross_episodes(self):
        from kyb.contracts import task_contract_id,amendment_cards
        contract=task_contract_id('arbitrary','Do the task','Public recipe')
        self.session.task_specification=self.session.evidence('task_specification',{'contract_id':contract})
        for scope,text in [('episode','Local specific value'),('same_public_contract','Reusable public instruction rule')]:
            self.session.evidence('task_contract_amendment',{'contract_id':contract,'episode':self.session.episode,
                'question':'Clarify the public rule','answer':text,'request_id':'source_request','reuse_scope':scope})
        next_trace=RunTrace(Path(self.temp.name)/'next',{})
        next_session=MemorySession(next_trace,self.store,task='arbitrary',dependencies=self.deps)
        next_session.task_specification=next_session.evidence('task_specification',{'contract_id':contract})
        cards=amendment_cards(next_session)
        self.assertEqual([c['answer'] for c in cards],['Reusable public instruction rule'])
        result,_=read_experience(next_session,level='contract',ids=[cards[0]['id']])
        self.assertEqual(result['records'][0]['answer'],cards[0]['answer'])
        next_session.task_specification=next_session.evidence('task_specification',{'contract_id':task_contract_id('arbitrary','Changed instruction','Public recipe')})
        self.assertFalse(amendment_cards(next_session))

    def test_amendment_directory_can_page_past_the_inline_index(self):
        from kyb.contracts import task_contract_id
        contract=task_contract_id('arbitrary','Instruction','Recipe')
        self.session.task_specification=self.session.evidence('task_specification',{'contract_id':contract})
        sources=[]
        for i in range(20):
            sources.append(self.session.evidence('task_contract_amendment',{'contract_id':contract,'episode':self.session.episode,
                'question':'Independent public instruction clarification '+str(i)+' '+('x'*120),'answer':'Rule '+str(i),
                'request_id':'request'+str(i),'reuse_scope':'same_public_contract'})['id'])
        self.session.attach_manager(PreparationClient());self.session.manager.run('Prepare','prepare')
        result,_=read_experience(self.session,level='contract',mode='browse',budget_bytes=1300)
        self.session.manager._admit(result)
        first=result['records'][0];self.assertEqual(first['view'],'detail_page')
        fragments={first['offset']:first['fragment']};token=first['continuation']
        self.session.evidence('task_contract_amendment',{'contract_id':contract,'episode':self.session.episode,
            'question':'Future rule must not enter an existing page','answer':'Future rule','request_id':'later','reuse_scope':'same_public_contract'})
        while token:
            part,_=read_experience(self.session,continuation=token,budget_bytes=1300)
            self.session.manager._admit(part)
            card=part['records'][0];fragments[card['offset']]=card['fragment'];token=card['continuation']
        directory=json.loads(''.join(fragments[k] for k in sorted(fragments)))
        self.assertEqual({r['id'] for r in directory['entries']},set(sources))
        self.assertFalse(self.session.manager.selectable_knowledge);self.assertFalse(self.session.manager.visible_knowledge)
        self.assertFalse(self.session.manager.allowed)


class V4WholeLoop(unittest.TestCase):
    setUp=pipeline.PolicyPipelineTests.setUp
    tearDown=pipeline.PolicyPipelineTests.tearDown

    def test_prepare_action_terminal_next_episode_candidate_without_sol_or_support_promotion(self):
        client=pipeline.Client([pipeline.call('move_eef',{'targets':{'left_gripper':0},'note':'Synthetic protocol test'})])
        client.budget=None;client.last_attempt_id='scripted_call';client.session_id='one_scripted_episode'
        policy=EefAgentPolicy(action_spec=self.spec,env={},client=client,planner=lambda **_:None,research=self.session)
        chunk=policy.act(self.observation)
        self.assertTrue(chunk.actions);policy.confirm_executed(len(chunk.actions))
        raw_command=self.trace.event('command',{'qualification':'Scripted protocol-only command marker; never executed in a simulator'})
        receipt=self.session.evidence('receipt',{'before_observation':self.after['id'],'execution':'complete',
            'executed_actions':len(chunk.actions),'env_start':0,'env_end':len(chunk.actions),'dependencies':self.deps})
        image=Path(self.temp.name)/'terminal.jpg';image.write_bytes(b'scripted pixels')
        self.session.evidence('observation',{**self.session.current_observation['data'],'after_receipt':receipt['id'],
            'images':{'cam_head':{'path':str(image),'sha256':hashlib.sha256(image.read_bytes()).hexdigest()}}})
        outline=[{'summary':'A commanded jaw closure; no physical object claim in this scripted test.',
            'first_receipt':receipt['id'],'last_receipt':receipt['id'],'conditions':{},'unknowns':['No real simulation execution.']}]
        client.calls.append(pipeline.call('finish_review',{'conclusion':'Synthetic closure recorded','uncertainty':'No object retention proven','episode_outline':outline}))
        result,_=review_terminal(self.session,client,{'prefix_replay':None},official_success=True)
        self.assertEqual(result['episode_outline'],outline);self.assertEqual(len(self.manager_client.inputs),1)
        row=self.store.db.execute('SELECT * FROM episode_records').fetchone()
        full_chain=self.store.evidence_payload(row['chain_ref'])
        self.assertIn(raw_command['id'],[r['id'] for r in full_chain['public_trace_refs']])
        self.assertFalse(self.store.db.execute('SELECT 1 FROM evidence WHERE id=?',(raw_command['id'],)).fetchone())
        outline_event=self.store.db.execute('SELECT sequence FROM evidence WHERE id=?',(row['outline_ref'],)).fetchone()[0]
        label_event=self.store.db.execute('SELECT sequence FROM evidence WHERE id=?',(row['result_ref'],)).fetchone()[0]
        self.assertLess(outline_event,label_event)
        old=episodes.read(self.store,task='push_T',frontier={'run':outline_event})[0]
        self.assertIsNone(old['official_success']);self.assertIsNone(old['trace_entry']['result'])
        chain_sequence=int(row['chain_ref'].rsplit(':e',1)[1])
        self.assertFalse(episodes.read(self.store,frontier={'run':chain_sequence-1}))
        next_trace=RunTrace(Path(self.temp.name)/'next_episode',{})
        next_session=MemorySession(next_trace,self.store,task='push_T',dependencies=self.deps)
        next_session.attach_manager(PreparationClient());next_session.manager.run('Prepare task','prepare')
        reader=self.store.evidence_payload
        def bounded(source):
            if source==row['chain_ref']:raise AssertionError('Hot path loaded the full per-tick chain')
            return reader(source)
        with patch.object(self.store,'evidence_payload',side_effect=bounded):candidate=next_session.manager.prefix_view()['episode_candidates'][0]
        self.assertEqual(candidate['stages'][0]['summary'],outline[0]['summary'])
        self.assertEqual(candidate['stages'][0]['conditions'],outline[0]['conditions'])
        self.assertEqual(candidate['stages'][0]['unknowns'],outline[0]['unknowns'])
        self.assertNotIn('first_receipt',candidate['stages'][0])
        self.assertTrue(candidate['official_success']);self.assertFalse(candidate['reviewed'])
        self.assertFalse(next_session.manager.allowed);self.assertFalse(next_session.manager.visible_revisions)
        answer,_=read_experience(next_session,level='episode',ids=[self.session.episode])
        self.assertEqual(answer['records'][0]['stages'],outline)
        next_session.manager._admit(answer)
        self.assertFalse(next_session.manager.visible_knowledge);self.assertFalse(next_session.manager.selectable_knowledge)
        proposal={'op':'new','kind':'body','module':'geometry','reading_role':'intrinsic_candidate',
            'observed_entity':'robot_part','observable':'relative_geometry','text':'Unsupported from candidate',
            'conditions':{},'reference_frame':'world','epistemic':'hypothesis','evidence':[row['outline_ref']]}
        rejected=self.store.submit(proposal,allowed_sources={row['outline_ref']},cutoff=next_session.source_frontier(),dependencies=self.deps,episode='next_episode')
        self.assertFalse(rejected['accepted']);self.assertIn('not_physical',rejected['reason'])
        with self.assertRaisesRegex(ValueError,'sealed'):
            episodes.seal(self.session,[{**outline[0],'summary':'Rewritten after the result'}])
        # A later success without an interpretation must not hide a complete positive candidate.
        empty_trace=RunTrace(Path(self.temp.name)/'later_success_without_outline',{})
        empty=MemorySession(empty_trace,self.store,task='push_T',dependencies=self.deps)
        empty.body_prior=self.session.body_prior
        episodes.begin(empty,{'prefix_replay':None},True);episodes.seal(empty,[])
        self.assertEqual(episodes.read(self.store,task='push_T',dependencies=self.deps,limit=1)[0]['id'],self.session.episode)


class V4InterruptedArchive(unittest.TestCase):
    setUp=p.PromotionTests.setUp
    tearDown=p.PromotionTests.tearDown
    observation=p.PromotionTests.observation

    def test_episode_cost_preserves_native_coverage_and_legacy_unknown(self):
        primary={'cli_invocations':2,'usage_reported':{'input_tokens':10,'output_tokens':1},
            'outcome_unknown':0,'usage_scope':'native_responses','captured_metadata_consistent':False,
            'qualification':'Partial captured metadata; final billing unknown.'}
        manager={'cli_invocations':1,'usage_reported':{'input_tokens':2,'output_tokens':1},'outcome_unknown':0}
        episodes.close(self.session,{},primary,manager,{'used':2},{'used':1},termination='offline_test')
        card=episodes.read(self.store,task='arbitrary')[0]
        self.assertFalse(card['role_costs']['primary']['captured_metadata_consistent'])
        self.assertEqual(card['role_costs']['primary']['usage_scope'],'native_responses')
        self.assertIn('Partial',card['role_costs']['primary']['qualification'])
        self.assertIsNone(card['role_costs']['manager']['captured_metadata_consistent'])
        self.assertIsNone(card['role_costs']['manager']['usage_scope'])

    def test_interrupted_run_has_unknown_outcome_and_failed_cost_without_extra_inference(self):
        primary={'cli_invocations':7,'usage_reported':{'input_tokens':100,'output_tokens':20},'outcome_unknown':0}
        manager={'cli_invocations':4,'usage_reported':None,'outcome_unknown':1}
        self.trace.event('command',{'recorded_only':'No simulator in this test'})
        episodes.close(self.session,{'prefix_replay':None},primary,manager,{'used':7},{'used':4},termination='run_exception:CodexCliFailure')
        card=episodes.read(self.store,task='arbitrary')[0]
        self.assertIsNone(card['official_success']);self.assertEqual(card['interpretation'],'mechanical_index')
        self.assertEqual(card['role_costs']['manager']['outcome_unknown'],1)
        self.assertIsNone(card['role_costs']['manager']['usage_reported'])
        self.assertIn('run_exception',card['termination'])
        total=self.store.db.execute('SELECT COUNT(*) FROM episode_records').fetchone()[0]
        episodes.close(self.session,{},primary,manager,{'used':7},{'used':4},termination='must_not_replace')
        self.assertEqual(total,self.store.db.execute('SELECT COUNT(*) FROM episode_records').fetchone()[0])
