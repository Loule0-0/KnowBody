"""Exact-source discovery never broadens visibility or grants raw authority."""
import copy
import hashlib
import json
import unittest
from unittest.mock import patch

from kyb import hybrid, library, readout
from kyb.knowledge import read_experience
from kyb.session import MemorySession
from kyb.trace import RunTrace
import test_hybrid_retrieval as retrieval
import test_physical_memory as physical


class SourceScopeTests(unittest.TestCase):
    add=physical.PhysicalMemoryTests.add
    observation=physical.PhysicalMemoryTests.observation
    patch=physical.PhysicalMemoryTests.patch

    def submit(self,value,**extra):
        cutoff=dict(self.store.db.execute('SELECT run,MAX(sequence) FROM evidence GROUP BY run'))
        return self.store.submit(value,allowed_sources=extra.get('sources',self.sources),cutoff=cutoff,
            dependencies=self.deps,episode=self.trace.root.name,visible_revisions=extra.get('visible',{}))

    def setUp(self):
        physical.PhysicalMemoryTests.setUp(self)
        library.initialize(self.store)
        self.index=retrieval.Index()
        self.session=MemorySession(RunTrace(self.root/'reader',{}),self.store,task='test',
            dependencies=self.deps,retrieval_index=self.index)
        self.case=self.submit(self.patch())
        self.assertTrue(self.case['accepted'],self.case)
        library.organize_archive(self.store)

    def tearDown(self):
        self.session.close_retrieval()
        physical.PhysicalMemoryTests.tearDown(self)

    def put(self,identifier,refs,**kw):
        revision=retrieval.HybridTests.add(self,identifier,**kw)
        with self.store.db:
            self.store.db.execute('UPDATE reusable_knowledge SET refs=? WHERE revision=?',
                (json.dumps(refs),revision))
        return revision

    def ref(self,case=None,relation='support',**kw):
        c=case or self.case
        return {'type':'case','id':c['claim_id'],'revision':c['revision'],
            'relation':relation,'reason':'Scoped test observation.',**kw}

    def query(self,**kw):
        return read_experience(self.session,'rotation clearance',view='statement',**kw)[0]

    def test_relation_roles_are_separate_and_knowledge_links_are_not_traversed(self):
        self.put('support',[self.ref(),self.ref(reason='Duplicate wording')])
        self.put('related',[self.ref(relation='related')])
        self.put('counter',[self.ref(relation='counter')])
        self.put('direct',[{'type':'evidence','id':self.before['id']}])
        self.put('transitive',[{'type':'knowledge','id':'support','revision':1,'relation':'related','reason':'Navigation'}])
        answer=self.query(source_episode='episode_A',limit=6,budget_bytes=16000)
        self.assertEqual({r['id'] for r in answer['records']},{'support','related','counter','direct'})
        for row in answer['records']:
            counts=row['source_relation_counts']
            self.assertEqual(counts['source_episode'],'episode_A')
            expected='direct_evidence' if row['id']=='direct' else row['id']
            self.assertEqual({k:v for k,v in counts.items() if k!='source_episode'},
                {k:int(k==expected) for k in ('support','counter','related','direct_evidence')})
        self.assertEqual(set(self.index.last_documents),{(r['id'],r['revision']) for r in answer['records']})
        self.assertFalse(self.session.allowed)
        self.assertTrue(next(r for r in answer['records'] if r['id']=='counter')['source_contested'])

    def test_unknown_future_or_other_capture_never_encodes_or_falls_back(self):
        self.put('known',[self.ref()])
        before=copy.deepcopy(self.index.last_documents)
        for source,frontier in [('foreign_library',None),('episode_A_restart',None),
                                ('episode_A',{'episode_A':0})]:
            answer=self.query(source_episode=source,frontier=frontier)
            self.assertEqual(answer['reason'],'source_not_found')
            self.assertFalse(answer['records']);self.assertFalse(answer['navigation'])
        self.assertEqual(before,self.index.last_documents)
        self.assertTrue(self.query(source_episode='episode_A')['records'])

    def test_route_is_intersection_and_empty_means_only_requested_scope(self):
        self.put('body',[self.ref()])
        answer=self.query(source_episode='episode_A',domain='external',family='contact_response')
        self.assertEqual(answer['reason'],'no_source_reference_in_requested_scope')
        self.assertFalse(answer['records'])
        self.assertEqual(self.query(source_episode='episode_A',domain='body',family='geometry')['records'][0]['id'],'body')

    def test_invalid_combinations_do_not_log_or_grant_delivery(self):
        self.put('known',[self.ref()]);seq=self.session.trace.sequence
        wrong=[{'ids':['known']},{'case_refs':[{'id':self.case['claim_id'],'revision':1}]},
            {'continuation':'any'},{'mode':'browse'},{'level':'episode'},{'level':'case'}]
        for extra in wrong:
            with self.assertRaises(ValueError):self.query(source_episode='episode_A',**extra)
        for value in ('',' episode_A','episode_A ','x'*301,[],42):
            with self.assertRaises(ValueError):self.query(source_episode=value)
        self.assertEqual(self.session.trace.sequence,seq)
        self.assertFalse(self.session.allowed)
        self.assertFalse(getattr(self.session,'_read_continuations',{}))

    def test_exact_old_case_ref_does_not_upgrade_and_support_delta_is_inherited(self):
        old=self.case
        source2=RunTrace(self.root/'episode_B',{})
        new=source2.event('observation',{'state':{}});self.store.register(new,{'state':{}})
        all_sources=self.sources|{new['id']}
        supported=self.submit(self.patch(op='support',claim_id=old['claim_id'],claim_revision=old['revision'],
            evidence=sorted(all_sources)),sources=all_sources,visible={old['claim_id']:old['revision']})
        self.assertTrue(supported['accepted'],supported)
        self.put('old_ref',[self.ref(old)])
        self.put('supported',[self.ref(supported)])
        a=self.query(source_episode='episode_A',budget_bytes=12000)
        b=self.query(source_episode='episode_B',budget_bytes=12000)
        self.assertEqual({r['id'] for r in a['records']},{'old_ref','supported'})
        self.assertEqual([r['id'] for r in b['records']],['supported'])

    def test_revise_replaces_support_but_preserves_counter_provenance(self):
        old=self.case
        source2=RunTrace(self.root/'episode_B',{})
        new=source2.event('observation',{'state':{}});self.store.register(new,{'state':{}})
        all_sources=self.sources|{new['id']}
        negative=self.submit(self.patch(op='counterexample',claim_id=old['claim_id'],claim_revision=old['revision'],
            evidence=sorted(all_sources)),sources=all_sources,visible={old['claim_id']:old['revision']})
        self.assertTrue(negative['accepted'],negative)
        revised=self.submit(self.patch(op='revise',claim_id=old['claim_id'],claim_revision=negative['revision']),
            visible={old['claim_id']:negative['revision']})
        self.assertTrue(revised['accepted'],revised)
        self.assertNotIn(new['id'],self.store.evidence_ids(revised['revision']))
        self.put('revised',[self.ref(revised)])
        answer=self.query(source_episode='episode_B')
        self.assertEqual([r['id'] for r in answer['records']],['revised'])
        self.assertEqual(answer['records'][0]['source_relation_counts']['support'],1)
        self.assertTrue(answer['records'][0]['source_contested'])

    def test_latest_retired_does_not_revive_but_old_cut_and_own_overlay_work(self):
        rev=self.put('known',[self.ref()])
        with self.store.db:
            cols=[r[1] for r in self.store.db.execute('PRAGMA table_info(reusable_knowledge)') if r[1]!='revision']
            self.store.db.execute('INSERT INTO reusable_knowledge('+','.join(cols)+') SELECT '+','.join(cols)+
                ' FROM reusable_knowledge WHERE revision=?',(rev,))
            self.store.db.execute("UPDATE reusable_knowledge SET status='retired' WHERE revision=?",(rev+1,))
        self.assertFalse(self.query(source_episode='episode_A')['records'])
        old=self.query(source_episode='episode_A',knowledge_revision=rev)
        self.assertEqual(old['records'][0]['revision'],rev)
        over=self.query(source_episode='episode_A',knowledge_revision=0,
            version_snapshot={'cases':{self.case['claim_id']:self.case['revision']},'knowledge':{'known':rev}})
        self.assertEqual(over['records'][0]['revision'],rev)

    def test_candidate_cursor_freezes_source_and_rejects_owner_or_context_changes(self):
        self.put('one',[self.ref()]);self.put('two',[self.ref()])
        first=self.query(source_episode='episode_A',limit=1,reader_owner='manager:stable')
        token=first['candidate_continuation']
        with self.assertRaises(ValueError):self.query(source_episode='episode_A',continuation=token)
        with self.assertRaisesRegex(ValueError,'another reader'):
            read_experience(self.session,continuation=token,reader_owner='manager:other')
        second,_=read_experience(self.session,continuation=token,reader_owner='manager:stable')
        self.assertEqual(second['scope']['source_episode'],'episode_A')
        self.assertTrue(second['records'])
        self.session.dependencies['body']='changed'
        with self.assertRaisesRegex(ValueError,'stale_selection'):
            read_experience(self.session,continuation=token,reader_owner='manager:stable')

    def test_oversized_card_is_lossless_and_preserves_source_in_detail(self):
        self.put('huge',[self.ref()],extra=10000)
        first=self.query(source_episode='episode_A')
        self.assertFalse(first['records']);self.assertLessEqual(readout.size(first),6000)
        nav=first['navigation'][0];self.assertEqual(nav['source_relation_counts']['support'],1)
        self.assertFalse(readout.delivered_statement(self.session,'policy:reader','knowledge','huge',1))
        token=nav['continuation'];fragments=[]
        for _ in range(20):
            answer,_=read_experience(self.session,continuation=token)
            self.assertLessEqual(readout.size(answer),6000)
            page=answer['records'][0];fragments.append(page['fragment'])
            token=page.get('continuation')
            if not token:break
        raw=''.join(fragments)
        self.assertEqual(hashlib.sha256(raw.encode()).hexdigest(),page['content_sha256'])
        card=json.loads(raw)
        self.assertEqual(card['source_relation_counts']['source_episode'],'episode_A')
        self.assertEqual(card['source_relation_counts']['support'],1)
        self.assertTrue(card['unknowns'])
        self.assertFalse(self.session.allowed)


class RGBSourceTests(unittest.TestCase):
    setUp=retrieval.HybridTests.setUp
    tearDown=retrieval.HybridTests.tearDown
    add=retrieval.HybridTests.add

    def test_real_rgb_profile_uses_only_own_registered_sources(self):
        evidence=self.session.evidence('observation',{'state':{}})
        rev=self.add('own')
        with self.store.db:self.store.db.execute('UPDATE reusable_knowledge SET refs=? WHERE revision=?',
            (json.dumps([{'type':'evidence','id':evidence['id']}]),rev))
        answer,_=read_experience(self.session,'rotation',source_episode='episode')
        self.assertEqual(answer['records'][0]['id'],'own')
        self.assertEqual(answer['records'][0]['source_relation_counts']['direct_evidence'],1)
        foreign,_=read_experience(self.session,'rotation',source_episode='enhanced_only')
        self.assertEqual(foreign['reason'],'source_not_found')
        with self.assertRaises(ValueError):
            self.session.evidence('surface_inspection',{'depth':1.})
