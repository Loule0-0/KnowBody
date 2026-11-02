"""Complete views precede unread records; continuations preserve scope and authority."""
import copy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from kyb import readout
from kyb.knowledge import read_experience
import test_promotion as fixture


class CompleteReadout(unittest.TestCase):
    def record(self, identifier, length=3000):
        return {'id':identifier,'revision':1,'layer':'case','text':'x'*length,
                'conditions':{'grasp':'observed shaft only'},
                'applicability':{'unknown_conditions':['rotation clearance']},
                'related_warnings':[{'reason':'Unreviewed counterexample'}]}

    def test_remaining_warnings_never_displace_first_complete_case(self):
        records=[self.record('first'),self.record('second')]
        records[1]['related_warnings']=[{'reason':'unknown '*3000}]
        s=SimpleNamespace()
        out=readout.pack(s,records,{'question':'a physical question'},budget_bytes=6000,limit=4,view='detail')
        self.assertEqual([r['id'] for r in out['records']],['first'])
        self.assertTrue(out['records'][0]['complete_record'])
        selected,offset=s._read_continuations[out['selection_continuation']]
        self.assertEqual(offset,1);self.assertEqual(selected['records'],records)
        self.assertLessEqual(readout.size(out),6000)
        self.assertNotIn('omitted',out)

    def test_unread_records_preserve_order_and_repeated_token_is_reproducible(self):
        records=[self.record(str(i)) for i in range(4)]
        s=SimpleNamespace();seen=[]
        out=readout.pack(s,records,{'question':'physical'},budget_bytes=6000,limit=4,view='detail')
        while True:
            self.assertTrue(all(r['complete_record'] for r in out['records']))
            seen.extend(r['id'] for r in out['records'])
            if not out.get('selection_continuation'):break
            selected,offset=s._read_continuations[out['selection_continuation']]
            args=dict(budget_bytes=6000,limit=4,view=selected['requested_view'],offset=offset)
            out=readout.pack(s,selected['records'],selected['answer'],**args)
            self.assertEqual(out,readout.pack(s,selected['records'],selected['answer'],**args))
        self.assertEqual(seen,['0','1','2','3'])

    def test_staged_oversized_case_stays_unread_then_pages_losslessly(self):
        original=self.record('large',9000);s=SimpleNamespace()
        staged=readout.pack(s,[original],{},budget_bytes=1100,limit=4,view='detail',complete_only=True)
        self.assertFalse(staged['records'])
        selected,offset=s._read_continuations[staged['selection_continuation']]
        actual=readout.pack(s,selected['records'],selected['answer'],budget_bytes=1100,
                            limit=4,view=selected['requested_view'],offset=offset)
        card=actual['records'][0];parts=[card['fragment']];token=card['continuation']
        self.assertFalse(card['complete_record'])
        while token:
            card=readout.page(s,token,1100);parts.append(card['fragment']);token=card['continuation']
        expected={**original,'view':'detail','complete_record':True}
        self.assertEqual(json.loads(''.join(parts)),expected)


class ScopedReadout(unittest.TestCase):
    setUp=fixture.PromotionTests.setUp
    tearDown=fixture.PromotionTests.tearDown
    observation=fixture.PromotionTests.observation
    attach=fixture.PromotionTests.attach
    manager=fixture.PromotionTests.manager
    add_body_case=fixture.PromotionTests.add_body_case

    def selection(self,view='detail'):
        batch=self.store.revision
        cases=[self.add_body_case(text='Observed conditional geometry '+str(batch)+'/'+str(i)+' '+('x'*1250)) for i in range(4)]
        result,_=read_experience(self.session,level='case',view=view,
            case_refs=[{'id':c['claim_id'],'revision':c['revision']} for c in cases],budget_bytes=4200)
        self.assertIn('selection_continuation',result)
        return cases,result['selection_continuation']

    def test_owner_and_view_cannot_change(self):
        _,token=self.selection('statement')
        with self.assertRaisesRegex(ValueError,'another reader'):
            read_experience(self.session,continuation=token,reader_owner='another-manager')
        with self.assertRaisesRegex(ValueError,'requested view'):
            read_experience(self.session,continuation=token,view='detail')
        with self.assertRaisesRegex(ValueError,'freezes'):
            read_experience(self.session,continuation=token,conditions={})
        result,_=read_experience(self.session,continuation=token)
        self.assertTrue(all(r.get('view')=='statement' for r in result['records']))

    def test_primary_entrypoint_keeps_view_and_its_lookup_log_does_not_stale_selection(self):
        cases=[self.add_body_case(text='Primary geometry '+str(i)+' '+('x'*1250)) for i in range(4)]
        first=json.loads(self.session.query_experience({'level':'case','view':'detail','budget_bytes':4200,
            'case_refs':[{'id':c['claim_id'],'revision':c['revision']} for c in cases]}))
        token=first['selection_continuation']
        second=json.loads(self.session.query_experience({'continuation':token,'budget_bytes':4200}))
        self.assertNotEqual(second.get('status'),'NEEDS_WORK',second)
        self.assertTrue(second['records'][0]['complete_record'])
        self.assertNotEqual(first['records'][0]['id'],second['records'][0]['id'])
        self.observation()
        stale=json.loads(self.session.query_experience({'continuation':token}))
        self.assertEqual(stale['status'],'NEEDS_WORK');self.assertIn('stale_selection',stale['reason'])

    def test_context_dependency_and_snapshot_changes_reject_instead_of_requery(self):
        for attribute in ('context','dependencies'):
            _,token=self.selection()
            original=copy.deepcopy(getattr(self.session,attribute))
            getattr(self.session,attribute)['changed']='new'
            with self.assertRaisesRegex(ValueError,'stale_selection'):
                read_experience(self.session,continuation=token)
            setattr(self.session,attribute,original)
        _,token=self.selection()
        with self.assertRaisesRegex(ValueError,'stale_selection'):
            read_experience(self.session,continuation=token,version_snapshot={'cases':{},'knowledge':{}})

    def test_removing_original_navigation_source_does_not_pass_as_an_unchanged_scope(self):
        physical_frontier=self.session.source_frontier()
        self.session.evidence('lookup_lead',{'question':'A navigation question only'})
        _,token=self.selection()
        with self.assertRaisesRegex(ValueError,'frontier was narrowed'):
            read_experience(self.session,continuation=token,frontier=physical_frontier)

    def test_initial_batch_continuation_uses_same_manager_snapshot_and_only_actual_delivery(self):
        cases=[self.add_body_case(text='Observed geometry '+str(i)+' '+('x'*1250)) for i in range(4)]
        m=self.manager();m.frontier=self.session.source_frontier();m.revision=self.store.revision
        def smaller(*args,**kwargs):
            kwargs['budget_bytes']=4200
            return read_experience(*args,**kwargs)
        with patch('kyb.manager.read_experience',side_effect=smaller):
            batch=m._case_batch([c['claim_id'] for c in cases])
        self.assertIn('selection_continuation',batch)
        delivered={r['id'] for r in batch['records']}
        self.assertEqual(set(m.visible_revisions),delivered)
        result=m._search({'continuation':batch['selection_continuation']})
        m._admit(result)
        new={r['id'] for r in result['records'] if r.get('complete_record')}
        self.assertTrue(new-delivered)
        self.assertEqual(set(m.visible_revisions),delivered|new)
        m.own_cases['new_own_case']=999
        with self.assertRaisesRegex(ValueError,'stale_selection'):
            m._search({'continuation':batch['selection_continuation']})

    def test_navigation_and_partial_case_do_not_grant_detail_permission(self):
        m=self.manager()
        for view in ('navigation','record_selection','detail_page'):
            record={'id':'unread','revision':99,'layer':'case','view':view,'complete_record':False}
            if view=='detail_page':continue  # SHA page shape is separately covered by StatementPaging.
            m._admit({'records':[record]})
        self.assertNotIn('unread',m.visible_revisions)
        with self.assertRaisesRegex(ValueError,'four'):
            m._case_batch(['a','b','c','d','e'])
