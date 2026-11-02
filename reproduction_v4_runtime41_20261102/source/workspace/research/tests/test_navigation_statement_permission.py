"""Navigation read permission never becomes physical or target-update authority."""
import copy,json,unittest
from unittest.mock import patch

import test_manager as fixture
from kyb import library,readout


class NavigationStatementPermission(unittest.TestCase):
    setUp=fixture.ManagerTests.setUp
    tearDown=fixture.ManagerTests.tearDown
    observation=fixture.ManagerTests.observation
    attach=fixture.ManagerTests.attach
    add_body_case=fixture.ManagerTests.add_body_case

    def config(self,text='A scoped known configuration',unknowns=None):
        op=self.op([]);op['question']=op['text']=text;op['unknowns']=unknowns or []
        with self.store.db:
            return library.publish_knowledge(self.store,op,allowed={self.prior['id']},visible_cases={},
                visible_knowledge={},dependencies=self.deps,frontier=self.session.source_frontier())

    def op(self,related):
        return {'op':'add','domain':'body','family':'geometry','question':'A separate configuration question',
            'text':'The actual public source establishes this configuration, not related links.',
            'conditions':{},'params':[],'strength':'configuration','use':'Check its source scope.',
            'unknowns':[],'depends_on':['body'],'refs':[{'type':'evidence','id':self.prior['id']},*related]}

    def manager(self):
        m=self.attach([fixture.publish(0),fixture.publish(1)])
        with patch.object(m,'_search',return_value={'records':[]}):m.run('Prepare','prepare')
        m.purpose='learn';m.allowed={self.prior['id']}
        return m

    def ref(self,k):
        return {'type':'knowledge','id':k['id'],'relation':'related','reason':'A related scoped proposition; not physical support.'}

    def read(self,m,k,view='statement'):
        answer=m._search({'ids':[k['id']],'view':view});m._admit(answer);return answer

    def publish(self,m,op):
        return m._publish_groups({'expected_version':m.version,'answer':'Keep the authority distinction.',
            'groups':[{'knowledge_ops':[op]}]})

    def test_complete_statement_allows_only_related_exact_reference_without_target_write(self):
        k=self.config();m=self.manager();before=tuple(self.store.db.execute('SELECT * FROM reusable_knowledge WHERE revision=?',(k['revision'],)).fetchone())
        self.read(m,k);self.assertNotIn(k['id'],m.visible_knowledge)
        result,done=self.publish(m,self.op([self.ref(k)]));self.assertTrue(done,result)
        target=result['knowledge_updates'][0]
        row=self.store.db.execute('SELECT refs,dependencies FROM reusable_knowledge WHERE revision=?',(target['revision'],)).fetchone()
        ref=next(r for r in json.loads(row['refs']) if r['type']=='knowledge')
        self.assertEqual(ref['revision'],k['revision']);self.assertEqual(json.loads(row['dependencies']),{'body':'b1'})
        self.assertEqual(tuple(self.store.db.execute('SELECT * FROM reusable_knowledge WHERE revision=?',(k['revision'],)).fetchone()),before)
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM reusable_knowledge WHERE id=?',(k['id'],)).fetchone()[0],1)

    def test_preview_and_partial_statement_cannot_resolve_a_navigation_ref(self):
        k=self.config(unknowns=['x'*11000]);m=self.manager()
        card=library.read_knowledge(self.store,'',context={},dependencies=self.deps,ids=[k['id']])[0]
        m._admit({'records':[readout.preview(card)]})
        with self.assertRaisesRegex(ValueError,'complete statement'):
            m._resolve_ref(self.ref(k),{},[],{},m.visible_knowledge,related_statement_versions=m.read_knowledge_versions)
        first=self.read(m,k);self.assertEqual(first['records'][0]['view'],'statement_page')
        self.assertNotIn(k['id'],m.read_knowledge_versions)
        with self.assertRaisesRegex(ValueError,'complete statement'):
            m._resolve_ref(self.ref(k),{},[],{},m.visible_knowledge,related_statement_versions=m.read_knowledge_versions)
        token=first['records'][0]['continuation']
        while token:
            answer=m._search({'continuation':token});m._admit(answer)
            token=answer['records'][0].get('continuation')
        resolved=m._resolve_ref(self.ref(k),{},[],{},m.visible_knowledge,related_statement_versions=m.read_knowledge_versions)
        self.assertEqual(resolved['revision'],k['revision']);self.assertNotIn(k['id'],m.visible_knowledge)

    def test_primary_statement_and_previous_transaction_do_not_grant_manager_permission(self):
        k=self.config();m=self.manager()
        self.session.query_experience({'ids':[k['id']],'view':'statement'})
        rejected,done=self.publish(m,self.op([self.ref(k)]));self.assertFalse(done)
        self.read(m,k)
        with patch.object(m,'_search',return_value={'records':[]}):m.run('A new question','review')
        m.allowed={self.prior['id']};m.purpose='learn'
        self.assertNotIn(k['id'],m.read_knowledge_versions)
        rejected,done=self.publish(m,self.op([self.ref(k)]));self.assertFalse(done)

    def test_statement_does_not_authorize_retire_revise_or_parameter_supply(self):
        k=self.config();m=self.manager();self.read(m,k)
        retired,done=self.publish(m,{'op':'retire','id':k['id'],'reason':'Not authorized by statement'})
        self.assertFalse(done);self.assertIn('full detail',str(retired))
        source=self.ref(k)
        op=self.op([source]);op['params']=[{'name':'invented','value':1.,'unit':'m','frame':'world','source':k['id']}]
        rejected,done=self.publish(m,op);self.assertFalse(done)
        self.assertIn('Navigation references cannot source parameters',str(rejected))
        bad=self.op([{**source,'relation':'support'}])
        rejected,done=self.publish(m,bad);self.assertFalse(done)
        self.assertIn('relation=related',str(rejected))

    def test_related_only_never_establishes_configuration_or_observed_knowledge(self):
        k=self.config();m=self.manager();self.read(m,k)
        for strength in ('configuration','observation'):
            op=self.op([self.ref(k)]);op['refs']=[self.ref(k)];op['strength']=strength
            result,done=self.publish(m,op);self.assertFalse(done)
        self.assertNotIn(k['id'],m.visible_knowledge)

    def test_statement_source_is_independently_watermark_and_modality_checked(self):
        k=self.config();ref={**self.ref(k),'revision':k['revision']}
        for scope in [
            {'case_at':self.store.revision,'knowledge_at':0,'profile':'rgbd_calibrated','snapshot':{}},
            {'case_at':self.store.revision,'knowledge_at':k['revision'],'profile':'rgb_only','snapshot':{}}]:
            with self.assertRaisesRegex(ValueError,'outside|matching-profile'):
                with self.store.db:library.publish_knowledge(self.store,self.op([ref]),
                    allowed={self.prior['id']},visible_cases={},visible_knowledge={},
                    related_statement_versions={k['id']:k['revision']},related_scope=scope,
                    dependencies=self.deps,frontier=self.session.source_frontier())

    def test_a_read_historical_version_is_not_upgraded_to_unread_latest(self):
        k=self.config();m=self.manager();self.read(m,k)
        old=dict(self.store.db.execute('SELECT * FROM reusable_knowledge WHERE revision=?',(k['revision'],)).fetchone())
        with self.store.db:
            columns=[c for c in old if c!='revision'];values=[old[c] if c!='text' else 'A newer unread description' for c in columns]
            self.store.db.execute('INSERT INTO reusable_knowledge('+','.join(columns)+') VALUES('+','.join('?' for _ in columns)+')',values)
        result,done=self.publish(m,self.op([self.ref(k)]));self.assertTrue(done,result)
        refs=json.loads(self.store.db.execute('SELECT refs FROM reusable_knowledge WHERE revision=?',
            (result['knowledge_updates'][0]['revision'],)).fetchone()[0])
        self.assertEqual(next(r['revision'] for r in refs if r['type']=='knowledge'),k['revision'])

    def newer_version(self,k):
        old=dict(self.store.db.execute('SELECT * FROM reusable_knowledge WHERE revision=?',(k['revision'],)).fetchone())
        with self.store.db:
            columns=[c for c in old if c!='revision']
            values=[old[c] if c!='text' else 'A newer scoped description' for c in columns]
            self.store.db.execute('INSERT INTO reusable_knowledge('+','.join(columns)+') VALUES('+','.join('?' for _ in columns)+')',values)
        return self.store.db.execute('SELECT * FROM reusable_knowledge WHERE id=? ORDER BY revision DESC',(k['id'],)).fetchone()

    def test_multiple_actually_delivered_versions_resolve_to_highest_in_either_order(self):
        k=self.config();old=self.store.db.execute('SELECT * FROM reusable_knowledge WHERE revision=?',(k['revision'],)).fetchone()
        new=self.newer_version(k);m=self.manager()
        for rows in ([old,new],[new,old]):
            m.read_knowledge_versions.clear()
            for row in rows:
                card=library.knowledge_view(self.store,row,{},self.deps,frontier=m.frontier)
                m._admit({'records':[readout.statement(card)]})
            self.assertEqual(m.read_knowledge_versions[k['id']],new['revision'])
            resolved=m._resolve_ref(self.ref(k),{},[],{},m.visible_knowledge,
                related_statement_versions=m.read_knowledge_versions)
            self.assertEqual(resolved['revision'],new['revision'])

    def test_revision_warning_is_computed_when_read_after_later_source_revision(self):
        k=self.config();m=self.manager();self.read(m,k)
        result,done=self.publish(m,self.op([self.ref(k)]));self.assertTrue(done,result)
        target=result['knowledge_updates'][0]
        before=tuple(self.store.db.execute('SELECT * FROM reusable_knowledge WHERE revision=?',(target['revision'],)).fetchone())
        self.newer_version(k)
        card=library.read_knowledge(self.store,'',context={},dependencies=self.deps,
            ids=[target['id']],frontier=self.session.source_frontier())[0]
        warning=next(w for w in card['related_warnings'] if w['id']==k['id'])
        self.assertIn('revised',warning['flags']);self.assertEqual(warning['cited'],k['revision'])
        self.assertGreater(warning['current'],warning['cited'])
        self.assertEqual(tuple(self.store.db.execute('SELECT * FROM reusable_knowledge WHERE revision=?',(target['revision'],)).fetchone()),before)


if __name__=='__main__':unittest.main()
