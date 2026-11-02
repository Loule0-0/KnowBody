"""End-to-end tool delivery invariants with a deterministic local encoder double."""
from pathlib import Path
from types import SimpleNamespace
import copy
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch

from kyb import hybrid, library, readout
from kyb.knowledge import read_experience
from kyb.memory import MemoryStore
from kyb.session import MemorySession
from kyb.trace import RunTrace
import test_physical_memory as physical
import test_manager as manager_fixture


class Index:
    def __init__(self):self.last_documents={}
    def rank(self,question,documents,**kwargs):
        self.last_documents=documents
        return sorted(documents,reverse=True),{'encoder':'test-double','documents':len(documents)}
    def close(self):pass


class HybridTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.store=MemoryStore(self.root/'memory.sqlite');library.initialize(self.store)
        self.index=Index()
        self.session=MemorySession(RunTrace(self.root/'episode',{}),self.store,task='test',
            dependencies={},perception_profile='rgb_only',retrieval_index=self.index)

    def tearDown(self):
        self.session.close_retrieval();self.store.close();self.temp.cleanup()

    def add(self,identifier,*,extra=0,conditions=None,domain='body',family='geometry'):
        with self.store.db:
            cur=self.store.db.execute('INSERT INTO reusable_knowledge(id,domain,family,question,text,conditions,params,'
                'dependencies,strength,use_hint,unknowns,refs,status) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)',
                (identifier,domain,family,'Rotation clearance '+identifier,'Rotation may sweep a long object past a rim.',
                 json.dumps(conditions or {}),'[]','{}','configuration','Check the held object and surrounding clearance.',
                 json.dumps(['Missing condition '+('x'*extra)] if extra else []),'[]','active'))
        return cur.lastrowid

    def chosen(self, identifiers):
        chosen=hybrid.selection(self.session,'rotation clearance',{},context=self.session.context,
            frontier={},case_at=0,knowledge_at=len(identifiers),snapshot=None,domain=None,family=None,roles=None,owner='policy:episode')
        chosen['ordered']=[{'id':identifier,'revision':i+1,'bm_rank':i+1,
                            'dense_rank':len(identifiers)-i} for i,identifier in enumerate(identifiers)]
        return chosen

    def test_both_branches_only_receive_the_explicit_visible_route(self):
        self.add('inside');self.add('outside',domain='external',family='contact_response')
        answer,_=read_experience(self.session,'rotation',domain='body',family='geometry',view='statement')
        self.assertEqual(set(self.index.last_documents),{('inside',1)})
        self.assertEqual({r['id'] for r in answer['records']},{'inside'})
        self.assertLessEqual(readout.size(answer),6000)

    def test_interleave_preserves_both_heads_and_exact_identity_dedup(self):
        self.assertEqual(hybrid.interleave(['b1','both','b3'],['d1','both','d3']),
                         ['b1','d1','both','b3','d3'])

    def test_optional_navigation_cannot_displace_a_large_fitting_head(self):
        self.add('b1',extra=2900)
        for identifier in ('b2','b3','d1'):self.add(identifier)
        answer,audit=hybrid.render(self.session,self.chosen(['b1','b2','b3','d1']),
                                  offset=0,budget_bytes=6000,limit=4)
        self.assertIn('b1',[r['id'] for r in answer['records']])
        self.assertNotIn(0,audit['oversized_navigation_indices'])
        self.assertLessEqual(readout.size(answer),6000)

    def test_normally_full_page_does_not_skip_next_large_card_for_small_one(self):
        self.add('b1',extra=1600);self.add('d1',extra=2000);self.add('small')
        chosen=self.chosen(['b1','d1','small'])
        chosen['ordered'][1]['dense_rank']=1;chosen['ordered'][0]['dense_rank']=2
        answer,audit=hybrid.render(self.session,chosen,offset=0,budget_bytes=6000,limit=4)
        self.assertEqual([r['id'] for r in answer['records']],['b1'])
        self.assertEqual(audit['cursor_after'],1)
        self.assertNotIn(1,audit['oversized_navigation_indices'])

    def test_all_oversized_candidates_are_eventually_exposed_without_loss(self):
        ids=['k'+str(i) for i in range(8)]
        for identifier in ids:self.add(identifier,extra=9000)
        chosen=self.chosen(ids);offset=0;seen=set();last=-1
        for _ in range(10):
            answer,audit=hybrid.render(self.session,chosen,offset=offset,budget_bytes=6000,limit=4)
            seen.update(r['id'] for r in answer['navigation'])
            self.assertFalse(answer['records']);self.assertLessEqual(readout.size(answer),6000)
            token=answer.get('candidate_continuation')
            if not token:break
            chosen,offset=self.session._read_continuations[token]
            self.assertGreater(offset,last);last=offset
        self.assertEqual(seen,set(ids))
        self.assertNotIn('candidate_continuation',answer)
        self.assertEqual(answer['omitted_candidates'],0)

    def test_cursor_preserves_query_conditions_before_and_after_native_sid(self):
        self.add('a',conditions={'grasp':'neck'});self.add('b',conditions={'grasp':'neck'})
        answer,_=read_experience(self.session,'rotation',conditions={'grasp':'neck'},view='statement',limit=1,
                                 reader_owner='manager:stable')
        token=answer['candidate_continuation']
        self.session.manager=SimpleNamespace(client=SimpleNamespace(session_id='native-created-later'))
        next_answer,_=read_experience(self.session,continuation=token,reader_owner='manager:stable')
        self.assertTrue(next_answer['records'])
        self.assertFalse(next_answer['records'][0]['applicability']['unknown_conditions'])
        self.assertEqual(self.session.context,{'task':'test'})

    def test_cursor_rejects_other_owner_and_changed_base_context(self):
        self.add('a');self.add('b')
        answer,_=read_experience(self.session,'rotation',view='statement',limit=1)
        token=answer['candidate_continuation']
        with self.assertRaisesRegex(ValueError,'another reader'):
            read_experience(self.session,continuation=token,reader_owner='manager:other')
        self.session.context['support']='table'
        with self.assertRaisesRegex(ValueError,'stale_selection'):
            read_experience(self.session,continuation=token)

    def test_primary_candidate_cursor_survives_its_navigation_log_but_not_new_evidence(self):
        self.add('a');self.add('b')
        first=json.loads(self.session.query_experience({'question':'rotation','limit':1}))
        token=first['candidate_continuation']
        second=json.loads(self.session.query_experience({'continuation':token}))
        self.assertNotEqual(second.get('status'),'NEEDS_WORK',second)
        self.assertTrue(second['records'])
        self.session.evidence('known_config',{'action_spec':'new public source'})
        stale=json.loads(self.session.query_experience({'continuation':token}))
        self.assertEqual(stale['status'],'NEEDS_WORK');self.assertIn('stale_selection',stale['reason'])

    def test_ignoring_navigation_logs_does_not_allow_a_narrowed_frontier(self):
        self.add('a');self.add('b')
        self.session.evidence('known_config',{'action_spec':'original public source'})
        physical_frontier=self.session.source_frontier()
        self.session.evidence('lookup_lead',{'question':'Navigation only'})
        first,_=read_experience(self.session,'rotation',view='statement',limit=1)
        with self.assertRaisesRegex(ValueError,'frontier was narrowed'):
            read_experience(self.session,continuation=first['candidate_continuation'],frontier=physical_frontier)

    def test_exact_statement_pages_remain_lossless_when_candidate_scope_changes(self):
        self.add('huge',extra=10000)
        answer,_=read_experience(self.session,'rotation',view='statement')
        token=answer['navigation'][0]['continuation'];self.session.context['support']='table'
        fragments=[];digest=None
        for _ in range(20):
            result,_=read_experience(self.session,continuation=token)
            page=result['records'][0];fragments.append(page['fragment']);digest=page['content_sha256']
            token=page.get('continuation')
            if not token:break
        raw=''.join(fragments)
        self.assertEqual(hashlib.sha256(raw.encode()).hexdigest(),digest)
        self.assertEqual(json.loads(raw)['id'],'huge')
        self.assertTrue(readout.delivered_statement(self.session,'policy:episode','knowledge','huge',1))
        self.assertFalse(self.session.allowed)

    def test_static_embedding_projection_excludes_dynamic_and_forbidden_fields(self):
        self.add('a');row=dict(self.store.db.execute('SELECT * FROM reusable_knowledge').fetchone())
        expected=hybrid.index_text(row)
        row.update(depth_snapshots='POISON',actual_RGBD_camera_config='POISON',
                   applicability={'changed_dependencies':['body']},read_state='POISON')
        self.assertEqual(hybrid.index_text(row),expected)
        self.assertNotIn('POISON',expected)

    def test_staged_statement_does_not_count_as_consumed_output(self):
        self.add('a')
        answer,_=read_experience(self.session,'rotation',view='statement',
            reader_owner='manager:A',track_delivery=False)
        self.assertTrue(answer['records'])
        self.assertFalse(readout.delivered_statement(self.session,'manager:A','knowledge','a',1))
        self.assertFalse(readout.delivered_statement(self.session,'policy:episode','knowledge','a',1))


class CaseRouteTests(unittest.TestCase):
    setUp=physical.PhysicalMemoryTests.setUp
    tearDown=physical.PhysicalMemoryTests.tearDown
    observation=physical.PhysicalMemoryTests.observation
    add=physical.PhysicalMemoryTests.add
    patch=physical.PhysicalMemoryTests.patch
    submit=physical.PhysicalMemoryTests.submit

    def knowledge(self,refs):
        library.initialize(self.store)
        with self.store.db:
            return self.store.db.execute('INSERT INTO reusable_knowledge(id,domain,family,question,text,conditions,params,'
                'dependencies,strength,use_hint,unknowns,refs,status) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)',
                ('K','external','contact_response','How does lateral contact respond?',
                 'Observed contact can move the object.','{}','[]','{}','configuration',
                 'Compare the scoped responses.','[]',json.dumps(refs),'active')).lastrowid

    def test_counter_append_reason_is_attributed_not_automatic_delivery(self):
        case=self.submit(self.patch());library.organize_archive(self.store)
        self.knowledge([])
        reason='This source author contrasts a different contact condition.'
        current=self.knowledge([{'type':'case','id':case['claim_id'],'revision':case['revision'],
                                  'relation':'counter','reason':reason}])
        row=self.store.db.execute('SELECT * FROM reusable_knowledge WHERE revision=?',(current,)).fetchone()
        card=library.knowledge_view(self.store,row,{},self.deps,
            frontier={'episode_A':self.trace.sequence})
        self.assertEqual(card['counter_sources'][0]['author_reason'],reason)
        self.assertTrue(card['counter_sources'][0]['added_with_unchanged_statement'])
        session=SimpleNamespace()
        self.assertFalse(readout.delivered_statement(session,'manager:A','case',case['claim_id'],case['revision']))
        readout.note_delivery(session,'manager:A',{'records':[{'id':case['claim_id'],'revision':case['revision'],
            'layer':'case','statement_complete':True,'complete_record':False}]})
        self.assertTrue(readout.delivered_statement(session,'manager:A','case',case['claim_id'],case['revision']))
        self.assertFalse(readout.delivered_statement(session,'policy:A','case',case['claim_id'],case['revision']))

    def test_counter_append_audit_does_not_inspect_invisible_prior_reference(self):
        case=self.submit(self.patch());library.organize_archive(self.store);cutoff=self.trace.sequence
        future=self.observation(.2)
        self.knowledge([{'type':'evidence','id':future['id']}])
        current=self.knowledge([{'type':'case','id':case['claim_id'],'revision':case['revision'],
                                  'relation':'counter','reason':'A visible authored contrast.'}])
        row=self.store.db.execute('SELECT * FROM reusable_knowledge WHERE revision=?',(current,)).fetchone()
        card=library.knowledge_view(self.store,row,{},self.deps,frontier={'episode_A':cutoff})
        self.assertIsNone(card['counter_sources'][0]['added_with_unchanged_statement'])
        self.assertNotIn(future['id'],json.dumps(card))

    def test_route_uses_old_visible_revision_not_future_revision_labels(self):
        first=self.submit(self.patch());cutoff=self.trace.sequence
        newer=self.observation(.1,after_receipt=self.receipt['id']);sources=self.sources|{newer['id']}
        second=self.submit(self.patch(op='support',claim_id=first['claim_id'],claim_revision=first['revision'],
            evidence=sorted(sources),post_observation_id=newer['id']),sources=sources,visible={first['claim_id']:first['revision']})
        self.assertTrue(second['accepted']);library.organize_archive(self.store)
        with self.store.db:
            self.store.db.execute("UPDATE case_navigation SET family='contact_response' WHERE revision=?",(first['revision'],))
            self.store.db.execute("UPDATE case_navigation SET family='retention_release' WHERE revision=?",(second['revision'],))
        def read(family,frontier):
            r=self.store.read('lateral push',context={},dependencies=self.deps,episode='episode_A',
                structured=True,include_unresolved=True,domain='external',family=family,evidence_cutoff={'episode_A':frontier})
            return r['selected']+r['historical_events']+r['review_required']
        self.assertEqual([r['revision'] for r in read('contact_response',cutoff)],[first['revision']])
        self.assertFalse(read('retention_release',cutoff))
        self.assertFalse(read('contact_response',self.trace.sequence))
        self.assertEqual([r['revision'] for r in read('retention_release',self.trace.sequence)],[second['revision']])


class ManagerPrefetchDeliveryTests(unittest.TestCase):
    setUp=manager_fixture.ManagerTests.setUp
    tearDown=manager_fixture.ManagerTests.tearDown
    observation=manager_fixture.ManagerTests.observation
    attach=manager_fixture.ManagerTests.attach
    add_body_case=manager_fixture.ManagerTests.add_body_case

    def manager(self):
        m=self.attach([manager_fixture.publish(0)]);m.run('Prepare','prepare')
        m.frontier=self.session.source_frontier();m.revision=self.store.revision
        m.knowledge_revision=0;m.purpose='learn'
        return m

    def test_case_batch_is_delivered_only_to_manager_owner(self):
        c=self.add_body_case();m=self.manager()
        actual=m._case_batch([c['claim_id']])
        self.assertTrue(actual)
        self.assertTrue(readout.delivered_statement(self.session,m.retrieval_owner,'case',c['claim_id'],c['revision']))
        self.assertFalse(readout.delivered_statement(self.session,'policy:'+self.session.episode,'case',c['claim_id'],c['revision']))

    def test_discarded_case_statement_is_not_admitted_or_delivered(self):
        c=self.add_body_case();m=self.manager()
        result={'records':[{'id':c['claim_id'],'revision':c['revision'],'layer':'case',
            'view':'statement','statement_complete':True,'complete_record':False}]}
        with patch('kyb.manager.read_experience',return_value=(result,[])):
            self.assertEqual(m._case_batch([c['claim_id']])['records'],[])
        self.assertNotIn(c['claim_id'],m.visible_revisions)
        self.assertFalse(readout.delivered_statement(self.session,m.retrieval_owner,'case',c['claim_id'],c['revision']))


if __name__=='__main__':unittest.main()
