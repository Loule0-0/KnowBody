"""End-to-end read, delivery, navigation and transaction boundary regressions."""
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import test_manager
from manager_fixtures import ScriptedManager
from kyb import episode_index, leads, library
from kyb.knowledge import read_experience
from kyb.readout import size
from kyb.trace import RunTrace, append


class Contracts(unittest.TestCase):
    setUp=test_manager.ManagerTests.setUp
    tearDown=test_manager.ManagerTests.tearDown
    observation=test_manager.ManagerTests.observation
    attach=test_manager.ManagerTests.attach
    add_body_case=test_manager.ManagerTests.add_body_case

    def config(self,text,conditions=None):
        with self.store.db:
            return library.publish_knowledge(self.store,{
                'op':'add','domain':'body','family':'frame_map','question':text,'text':text,
                'conditions':conditions or {},'strength':'configuration','use':'Check the declared relation.',
                'refs':[{'type':'evidence','id':self.prior['id']}]},
                allowed={self.prior['id']},visible_cases={},visible_knowledge={},
                dependencies=self.deps,frontier=self.session.source_frontier())

    def test_exact_case_ids_do_not_fall_back_to_other_cases(self):
        a=self.add_body_case(text='First geometry');b=self.add_body_case(text='Second geometry')
        answer,_=read_experience(self.session,level='case',ids=[a['claim_id']])
        self.assertEqual([r['id'] for r in answer['records']],[a['claim_id']])
        reverse,_=read_experience(self.session,level='case',ids=[b['claim_id'],a['claim_id']])
        self.assertEqual([r['id'] for r in reverse['records']],[b['claim_id'],a['claim_id']])
        with self.assertRaisesRegex(ValueError,'not visible'):
            read_experience(self.session,level='case',ids=['missing'])

    def test_exact_selector_conflicts_and_frontier_are_explicit(self):
        a=self.add_body_case()
        with self.assertRaisesRegex(ValueError,'only one'):
            read_experience(self.session,level='case',ids=[a['claim_id']],case_refs=[{'id':a['claim_id'],'revision':a['revision']}])
        with self.assertRaisesRegex(ValueError,'not visible'):
            read_experience(self.session,level='case',ids=[a['claim_id']],frontier={self.session.episode:0})
        with self.assertRaisesRegex(ValueError,'question'):
            read_experience(self.session)
        with self.assertRaisesRegex(ValueError,'filters'):
            read_experience(self.session,level='case',ids=[a['claim_id']],domain='body')

    def test_six_exact_ids_deliver_six_complete_statements_and_leave_search_corpus_intact(self):
        ids=[self.config('Short declared relation '+str(i))['id'] for i in range(6)]
        kwargs=dict(context={},dependencies=self.deps,case_at=self.store.revision,frontier=self.session.source_frontier())
        library.read_knowledge(self.store,'declared',**kwargs)
        before=[tuple(r) for r in self.store.db.execute('SELECT * FROM temp.reusable_search ORDER BY rowid')]
        answer,_=read_experience(self.session,ids=ids,budget_bytes=50000,view='statement')
        self.assertEqual([r['id'] for r in answer['records']],ids)
        self.assertTrue(all(r['statement_complete'] for r in answer['records']))
        self.assertEqual(before,[tuple(r) for r in self.store.db.execute('SELECT * FROM temp.reusable_search ORDER BY rowid')])
        with self.assertRaisesRegex(ValueError,'truncated'):
            read_experience(self.session,ids=ids,limit=4)

    def test_exact_knowledge_and_pinned_case_requests_never_silently_omit_hidden_members(self):
        a=self.config('Visible relation');b=self.config('Future relation')
        for kwargs in ({'ids':[a['id'],'missing']},{'ids':[b['id']],'knowledge_revision':a['revision']}):
            with self.assertRaisesRegex(ValueError,'not visible'):read_experience(self.session,**kwargs)
        first=self.add_body_case();frontier=self.session.source_frontier()
        later=self.observation();hidden=self.add_body_case(source=later,text='A different future camera marker geometry.')
        with self.assertRaisesRegex(ValueError,'not visible'):
            read_experience(self.session,level='case',case_refs=[{'id':c['claim_id'],'revision':c['revision']} for c in (first,hidden)],frontier=frontier)

    def test_retired_knowledge_is_absent_from_search_and_browse_but_exactly_auditable(self):
        a=self.config('Specific old declared relation')
        with self.store.db:library.publish_knowledge(self.store,{'op':'retire','id':a['id'],'reason':'Duplicate current material'},
            allowed=set(),visible_cases={},visible_knowledge={a['id']:a['revision']},dependencies=self.deps,frontier=self.session.source_frontier())
        for kwargs in ({'question':'specific declared'},{'mode':'browse','domain':'body','family':'frame_map'}):
            answer,_=read_experience(self.session,**kwargs)
            self.assertEqual(answer['records'],[])
        exact,_=read_experience(self.session,ids=[a['id']])
        self.assertEqual(exact['records'][0]['status'],'retired')

    def test_selected_order_reaches_actual_inline_and_applicability_stays_live(self):
        a=self.config('Older important relation',{'load':'declared-only'})
        b=self.config('Newer relation')
        m=self.attach([test_manager.publish(0)]);m.run('Prepare','prepare')
        answer=m._search({'ids':[a['id'],b['id']],'view':'statement'});m._admit(answer)
        result,done=m._publish_groups({'expected_version':m.version,'answer':'Use this order.','default_knowledge':[a['id'],b['id']]})
        self.assertTrue(done,result)
        self.assertEqual([k['id'] for k in m.prefix_view()['knowledge']],[a['id'],b['id']])
        self.session.dependencies['body']='changed'
        self.assertFalse(m.prefix_view()['knowledge'])

    def test_missing_selection_error_names_only_unread_item(self):
        a=self.config('Already read relation');b=self.config('Not yet read relation')
        m=self.attach([test_manager.publish(0)])
        from unittest.mock import patch
        # This test isolates read authorization, independent of whether the
        # automatic hybrid preparation happens to discover both fixture cards.
        with patch.object(m,'_search',return_value={'records':[]}):m.run('Prepare','prepare')
        m._admit(m._search({'ids':[a['id']],'view':'statement'}))
        result,done=m._publish_groups({'expected_version':m.version,'answer':'Select both.','default_knowledge':[a['id'],b['id']]})
        self.assertFalse(done)
        reason=result['issues'][0]['reason']
        self.assertIn(b['id'],reason);self.assertNotIn(a['id'],reason)
        self.assertFalse(m.allowed);self.assertFalse(m.visible_knowledge)

    def test_fts_reuse_respects_scope_and_rollback(self):
        a=self.config('Declared relation one');b=self.config('Declared relation two')
        kwargs=dict(context={},dependencies=self.deps,case_at=self.store.revision,frontier=self.session.source_frontier())
        full=library.read_knowledge(self.store,'declared',**kwargs)
        self.assertEqual(full,library.read_knowledge(self.store,'declared',**kwargs))
        self.store.db.commit()
        with self.assertRaises(RuntimeError):
            with self.store.db:
                library.read_knowledge(self.store,'declared',ids=[a['id']],**kwargs)
                raise RuntimeError('Publication rolled back')
        only=library.read_knowledge(self.store,'declared',ids=[a['id']],**kwargs)
        self.assertEqual([r['id'] for r in only],[a['id']])
        # A newer known record must not change the old visible BM25 corpus.
        old=library.read_knowledge(self.store,'declared',at=b['revision'],**kwargs)
        self.config('Declared future relation')
        self.assertEqual(old,library.read_knowledge(self.store,'declared',at=b['revision'],**kwargs))

    def test_navigation_cache_invalidates_on_case_version_not_ordinary_observation(self):
        m=self.attach([test_manager.publish(0)]);m.run('Prepare','prepare')
        # Publication now audits the rendered packet and may warm this cache.
        # Start cold to check invalidation independently of that audit read.
        m._navigation_cache=None
        with patch.object(library,'navigation',wraps=library.navigation) as nav:
            m.prefix_view();self.observation();m.prefix_view()
            self.assertEqual(nav.call_count,1)
            self.add_body_case();m.prefix_view()
            self.assertEqual(nav.call_count,2)

    def test_navigation_never_caches_a_rolled_back_revision_reused_by_a_later_commit(self):
        case=self.add_body_case();m=self.attach([test_manager.publish(0)]);m.run('Prepare','prepare')
        m.prefix_view();revision=self.store.revision+1
        def insert(identifier,family):
            row=dict(self.store.db.execute('SELECT * FROM claims WHERE revision=?',(case['revision'],)).fetchone())
            row.update(id=identifier,revision=revision)
            self.store.db.execute('INSERT INTO claims VALUES('+','.join('?' for _ in row)+')',tuple(row.values()))
            self.store.db.execute('INSERT INTO decisions(patch,accepted,reason,revision) VALUES(?,1,?,?)',
                (json.dumps({'op':'new','evidence':[]}),identifier,revision))
            self.store.db.execute('INSERT INTO case_navigation VALUES(?,?,?,?)',(revision,'body',family,'{}'))
        self.store.db.execute('BEGIN');insert('rolled-back','tool_mount')
        self.assertTrue(any(r['family']=='tool_mount' for r in m.prefix_view()['case_routes']))
        self.store.db.rollback()
        with self.store.db:insert('committed','visual_action')
        routes=m.prefix_view()['case_routes']
        self.assertFalse(any(r['family']=='tool_mount' for r in routes))
        self.assertTrue(any(r['family']=='visual_action' for r in routes))

    def test_nonprepare_manager_budget_is_source_time_and_not_read_permission(self):
        self.observation(env_step=38,remaining_steps=62)
        m=self.attach([test_manager.publish(0),test_manager.publish(1)])
        m.run('Prepare','prepare');m.run('Compare the observed response','review')
        prepare=json.loads(m.client.inputs[0][1]['content'])
        review=json.loads(m.client.inputs[1][1]['content'])
        self.assertNotIn('simulation_budget',prepare)
        self.assertEqual(review['simulation_budget']['step_limit'],100)
        self.assertEqual(review['simulation_budget']['remaining_steps'],62)
        self.assertFalse(m.allowed);self.assertFalse(m.seen_images);self.assertFalse(m.seen_telemetry)

    def test_lead_interpretation_and_processed_ranges_have_real_continuations(self):
        m=self.attach([test_manager.publish(0)]);m.run('Prepare','prepare');m.purpose='learn'
        pub=self.trace.event('manager_group_publication',{'answer':'Interpretation, not evidence.'})
        with self.store.db:leads.add(self.store,pub,{'question':'What changed?',
            'answer':'Full qualified answer. '*100,'review_publication':pub['id']})
        m.frontier={self.session.episode:pub['sequence']}
        packet=leads.pending(self.store,self.session.episode,m.frontier)
        self.assertLessEqual(size(packet),2800)
        self.assertFalse(packet['records'][0]['answer_complete'])
        full=leads.read(self.store,self.session.episode,pub['id'],m.frontier)
        self.assertEqual(full['answer'],'Full qualified answer. '*100)
        full['candidate_refs']=[{'id':'unread_knowledge','revision':7,'view':'statement','statement_complete':True}]
        before=(dict(m.visible_revisions),dict(m.visible_knowledge),dict(m.selectable_knowledge))
        m._admit({'learning_lead':full})
        self.assertEqual(before,(m.visible_revisions,m.visible_knowledge,m.selectable_knowledge))
        self.assertFalse(m.allowed)
        with self.assertRaisesRegex(ValueError,'Unregistered'):
            m._record(pub['id'])
        with self.assertRaises(ValueError):leads.read(self.store,'other',pub['id'],m.frontier)
        with self.assertRaises(ValueError):leads.read(self.store,self.session.episode,pub['id'],{self.session.episode:0})
        with self.store.db:self.store.db.execute('INSERT INTO processed_intervals VALUES(?,?,?)',(self.receipt['id'],'old','case'))
        self.assertFalse(m._intervals(self.session.episode)['intervals'])
        indexed=m._intervals(self.session.episode,unprocessed=False)
        self.assertEqual(indexed['intervals'][0]['receipt'],self.receipt['id'])
        self.assertFalse(m._intervals(self.session.episode,through=self.receipt['sequence'],unprocessed=False)['intervals'])


    def test_question_and_lead_directory_pages_grant_no_read_or_selection_permissions(self):
        leads.initialize(self.store)
        for i in range(14):
            ref=self.session.evidence('policy_decision',{'tool':'manage_memory','arguments':{'question':f'Compare {i}'}})
            with self.store.db:leads.add(self.store,ref,{'question':f'Compare {i}','candidate_refs':[{'id':'unread','revision':1,'view':'statement'}]})
        m=self.attach([test_manager.publish(0)]);m.run('Prepare','prepare')
        m.purpose='review';m.frontier=self.session.source_frontier()
        def permissions():return (set(m.allowed),set(m.seen_images),set(m.seen_telemetry),dict(m.visible_revisions),dict(m.visible_knowledge),dict(m.selectable_knowledge))
        before=permissions()
        for kind in ('questions','leads'):
            cursor=0;seen=[]
            while cursor is not None:
                page=episode_index.directory(self.store,self.session.episode,cursor,m.frontier[self.session.episode],kind)
                m._admit(page);self.assertEqual(permissions(),before)
                seen.extend(e['id'] for e in page['entries']);cursor=page['next_after']
            self.assertEqual(len(seen),14);self.assertEqual(len(set(seen)),14)


class Navigation(unittest.TestCase):
    def test_maximum_call_question_directories_remain_bounded_and_reachable(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace=RunTrace(Path(tmp)/'episode_with_a_realistic_long_identifier_20261024T120000',{})
            import test_manager
            f=test_manager.ManagerTests();f.setUp()
            try:
                for i in range(1000):
                    trace.event('policy_decision',{'tool':'manage_memory','arguments':{'question':'q'*1200,'purpose':'review'}})
                    ref={'id':trace.root.name+f':e{i+1:06d}','sequence':i+1}
                    with f.store.db:leads.add(f.store,ref,{'question':'q'*1200,'answer':'a'*9000,'review_publication':ref['id']})
                packet=episode_index.readout(episode_index.build(trace.root))
                self.assertLessEqual(size(packet),2800)
                self.assertEqual(len(packet['question_sequences'])+packet['omitted_question_sequences'],1000)
                self.assertNotIn('after',packet['read_trajectory'])
                pending=leads.pending(f.store,trace.root.name,{trace.root.name:1000})
                self.assertLessEqual(size(pending),2800)
                self.assertEqual(len(pending['lead_sequences'])+pending['omitted_lead_sequences'],1000)
                cursor=0;seen=[]
                while cursor is not None:
                    page=episode_index.directory(f.store,trace.root.name,cursor,1000,'leads')
                    seen.extend(r['sequence'] for r in page['entries']);cursor=page['next_after']
                self.assertEqual(seen,list(range(1,1001)))
                self.assertEqual(leads.read(f.store,trace.root.name,trace.root.name+':e000100',{trace.root.name:100})['answer'],'a'*9000)
            finally:f.tearDown()

    def test_diagnostic_middle_window_survives_and_costs_are_conserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace=RunTrace(Path(tmp)/'run',{})
            question=None;measurement=None
            for i in range(32):
                obs=trace.event('observation',{'state':{}})
                append(trace.root/'cli_calls.jsonl',{'attempt_id':str(i),'role':'policy','state':'turn_complete','time':time.time()})
                args={'question':'Distinguish the remaining alternatives. '*8,'purpose':'review'} if i==16 else {}
                tool='manage_memory' if i==16 else 'hold' if i==20 else 'move_eef'
                ref=trace.event('policy_decision',{'call_id':str(i),'tool':tool,'arguments':args,'observation_id':obs['id']})
                if i==16:question=ref
                if i==17:measurement=trace.event('surface_inspection',{'before_observation_id':obs['id']})
                if tool!='manage_memory':
                    receipt=trace.event('receipt',{'call_id':str(i),'before_observation':obs['id'],
                        'executed_actions':7 if tool=='hold' else 1,'execution':'complete'})
                    trace.event('observation',{'after_receipt':receipt['id']})
            index=episode_index.build(trace.root);packet=episode_index.readout(index)
            self.assertLessEqual(size(packet),2800)
            self.assertEqual(packet['accounting']['environment_steps_by_tool']['hold'],7)
            self.assertEqual(sum(packet['accounting']['environment_steps_by_tool'].values()),37)
            self.assertIn(question['sequence'],packet['question_sequences'])
            self.assertEqual(packet['memory_questions'][0]['sequence'],question['sequence'])
            self.assertTrue(any(w['events'][0]<=measurement['sequence']<=w['events'][1] for w in packet['windows']))


if __name__=='__main__':unittest.main()
