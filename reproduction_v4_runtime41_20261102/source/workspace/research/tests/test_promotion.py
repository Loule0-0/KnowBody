"""Observable organization, provenance and snapshot behavior; no model calls."""
import copy
import hashlib
import json
import unittest
from pathlib import Path
from kyb import library
import test_manager as fixtures
publish=fixtures.publish

class PromotionTests(unittest.TestCase):
    setUp=fixtures.ManagerTests.setUp
    tearDown=fixtures.ManagerTests.tearDown
    observation=fixtures.ManagerTests.observation
    attach=fixtures.ManagerTests.attach
    add_body_case=fixtures.ManagerTests.add_body_case

    def manager(self):
        m=self.attach([publish(0)]);m.run('prepare','prepare');m.purpose='learn'
        return m
    def case_patch(self):
        return {'op':'new','local':'c1','kind':'body','module':'geometry','reading_role':'intrinsic_candidate',
            'observed_entity':'robot_part','observable':'relative_geometry','text':'A candidate relation at the sampled pose.',
            'conditions':{},'reference_frame':'world','epistemic':'hypothesis','evidence':[self.post['id']]}
    def op(self,refs,**kwargs):
        return {'op':'add','domain':'body','family':'geometry','question':'Which relation needs checking?',
            'text':'A scoped candidate relation, with no calibrated gain established.','conditions':{},
            'strength':'hypothesis','use':'Check the matching body relation before applying a gain.',
            'refs':refs,**kwargs}
    def read_case(self,m,case):
        m.frontier=self.session.source_frontier();m.revision=self.store.revision
        answer=m._search({'question':'relation','level':'case','case_refs':[{'id':case['claim_id'],'revision':case['revision']}]})
        m._admit(answer)
        return answer
    def canonical_ref(self,case,relation='support',**extra):
        return {'type':'case','id':case['claim_id'],'revision':case['revision'],'relation':relation,'reason':'Supports the narrow candidate statement.',**extra}
    def write(self,op,cases):
        return library.publish_knowledge(self.store,op,allowed={self.prior['id']},
            visible_cases={c['claim_id']:c['revision'] for c in cases},visible_knowledge={},
            dependencies={'body':'different_current_body'},frontier=self.session.source_frontier())
    def test_same_batch_local_case_and_two_knowledge_links_commit_once(self):
        m=self.manager();m.allowed={self.post['id']}
        ref={'type':'case','local':'c1','relation':'support','reason':'The case supplies this limited candidate.'}
        result,_=m._publish({'expected_version':1,'answer':'Candidate only',
            'memory_patch':[self.case_patch()],
            'knowledge_ops':[self.op([ref],local='k1'),self.op([ref,{'type':'knowledge','local':'k1','relation':'related','reason':'Historical candidate navigation.'}],text='A second candidate using the earlier one.')]})
        case=result['memory_updates'][0];disposition=result['case_dispositions'][case['claim_id']]
        self.assertEqual(disposition['action'],'linked');self.assertEqual(len(disposition['knowledge_revisions']),2)
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM case_disposition').fetchone()[0],1)
        self.assertEqual(json.loads(self.store.db.execute('SELECT refs FROM reusable_knowledge ORDER BY revision LIMIT 1').fetchone()[0])[0]['revision'],case['revision'])
    def test_same_batch_revised_case_id_resolves_to_accepted_new_version(self):
        case=self.add_body_case();m=self.manager();self.read_case(m,case);m.allowed={self.post['id']}
        patch={**self.case_patch(),'op':'revise','claim_id':case['claim_id'],'text':'A refined relation at the sampled pose.'}
        patch.pop('local')
        ref={k:v for k,v in self.canonical_ref(case).items() if k!='revision'}
        result,_=m._publish({'expected_version':1,'answer':'Use the explicitly revised source.',
            'memory_patch':[patch],'knowledge_ops':[self.op([ref])]})
        revised=result['memory_updates'][0]['revision']
        stored=json.loads(self.store.db.execute('SELECT refs FROM reusable_knowledge').fetchone()[0])[0]
        self.assertGreater(revised,case['revision']);self.assertEqual(stored['revision'],revised)
        self.assertEqual(result['case_dispositions'][case['claim_id']]['revision'],revised)
    def test_same_batch_reference_failure_rolls_back_revision_and_permissions(self):
        case=self.add_body_case();m=self.manager();self.read_case(m,case);m.allowed={self.post['id']}
        patch={**self.case_patch(),'op':'revise','claim_id':case['claim_id'],'text':'A proposed refined relation.'};patch.pop('local')
        ref={k:v for k,v in self.canonical_ref(case).items() if k!='revision'}
        with self.assertRaises(ValueError):m._publish({'expected_version':1,'answer':'Reject atomically.',
            'memory_patch':[patch],'knowledge_ops':[self.op([ref]),self.op([{'type':'knowledge','id':'unread'}])]})
        self.assertEqual(self.store.revision,case['revision'])
        self.assertEqual(m.visible_revisions,{case['claim_id']:case['revision']})
        self.assertEqual(m.visible_knowledge,{})
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM reusable_knowledge').fetchone()[0],0)
    def test_case_id_does_not_absorb_an_unread_concurrent_revision(self):
        case=self.add_body_case();m=self.manager();self.read_case(m,case)
        patch={**self.case_patch(),'op':'revise','claim_id':case['claim_id'],'claim_revision':case['revision'],
            'text':'Another transaction changed the source.'};patch.pop('local')
        concurrent=self.store.submit(patch,allowed_sources=self.session.allowed,cutoff=self.session.cutoff,
            dependencies=self.deps,episode=self.session.episode,visible_revisions={case['claim_id']:case['revision']})
        self.assertTrue(concurrent['accepted'])
        ref={k:v for k,v in self.canonical_ref(case).items() if k!='revision'}
        m._publish({'expected_version':1,'answer':'Preserve the actually read source.','knowledge_ops':[self.op([ref])]})
        stored=json.loads(self.store.db.execute('SELECT refs FROM reusable_knowledge').fetchone()[0])[0]
        self.assertEqual(stored['revision'],case['revision']);self.assertNotEqual(stored['revision'],concurrent['revision'])
    def test_same_batch_revised_knowledge_id_resolves_to_accepted_new_version(self):
        case=self.add_body_case();op=self.op([self.canonical_ref(case)])
        with self.store.db:first=self.write(op,[case])
        m=self.manager();self.read_case(m,case);m.visible_knowledge={first['id']:first['revision']}
        ref={k:v for k,v in self.canonical_ref(case).items() if k!='revision'}
        result,_=m._publish({'expected_version':1,'answer':'Use the explicitly revised knowledge.',
            'knowledge_ops':[{**op,'op':'revise','id':first['id'],'refs':[ref],'text':'An explicitly refined candidate relation.'},
                self.op([ref,{'type':'knowledge','id':first['id'],'relation':'related','reason':'Historical candidate navigation.'}],text='A second candidate using the refined relation.')]})
        revised,dependent=result['knowledge_updates']
        refs=json.loads(self.store.db.execute('SELECT refs FROM reusable_knowledge WHERE revision=?',(dependent['revision'],)).fetchone()[0])
        self.assertEqual(next(r for r in refs if r['type']=='knowledge')['revision'],revised['revision'])
    def test_missing_disposition_and_bad_ref_roll_back_entire_new_case(self):
        m=self.manager();m.allowed={self.post['id']};before=self.store.revision
        for extra in ({},{'knowledge_ops':[self.op([{'type':'case','local':'c9','relation':'support','reason':'Bad'}])]}):
            with self.assertRaises(ValueError):m._publish({'expected_version':1,'answer':'x','memory_patch':[self.case_patch()],**extra})
            self.assertEqual(self.store.revision,before)
            self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM reusable_knowledge').fetchone()[0],0)
            self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM case_disposition').fetchone()[0],0)
    def test_target_requires_disposition_and_keep_is_valid(self):
        case=self.add_body_case();m=self.manager();self.read_case(m,case);m.targets={case['claim_id']:case['revision']}
        with self.assertRaisesRegex(ValueError,'Every new/targeted'):m._publish({'expected_version':1,'answer':'x'})
        result,_=m._publish({'expected_version':1,'answer':'No reusable relation yet','dispositions':[
            {'case':case['claim_id'],'action':'keep','reason':'Single pose does not establish a transferable gain.'}]})
        self.assertEqual(result['case_dispositions'][case['claim_id']]['action'],'keep')
    def test_offline_targeted_learning_needs_no_current_episode_intervals(self):
        case=self.add_body_case();self.session.episode='offline_maintenance';self.session.current_observation=None
        m=self.attach([publish(0,dispositions=[{'case':case['claim_id'],'action':'keep','reason':'Preserve the scoped observation.'}])])
        result=m.run('Organize this archived relation.','learn',targets=[case['claim_id']])
        initial=json.loads(self.client.inputs[0][1]['content'])
        self.assertNotIn('recent_intervals',initial);self.assertNotIn('unprocessed',initial)
        self.assertEqual(initial['case_batch'][0]['id'],case['claim_id'])
        self.assertEqual(result['case_dispositions'][case['claim_id']]['action'],'keep')
    def test_related_is_not_support_and_unknown_material_premise_rejects_support(self):
        a=self.add_body_case();b=self.add_body_case(text='Another marker under uncertain contact.')
        op=self.op([self.canonical_ref(a),self.canonical_ref(b,'related')])
        with self.store.db:k=self.write(op,[a,b])
        view=library.read_knowledge(self.store,'',context={},dependencies=self.deps,ids=[k['id']])[0]
        self.assertEqual(view['support_cases'],1);self.assertEqual(view['related_cases'],1)
        op=self.op([self.canonical_ref(a,unverified_conditions=['contact'])],conditions={'contact':'loaded'})
        with self.assertRaisesRegex(ValueError,'related, not support'):self.write(op,[a])
    def test_current_pins_do_not_relabel_historical_body_knowledge(self):
        a=self.add_body_case()
        with self.store.db:k=self.write(self.op([self.canonical_ref(a)]),[a])
        pins=json.loads(self.store.db.execute('SELECT dependencies FROM reusable_knowledge WHERE id=?',(k['id'],)).fetchone()[0])
        self.assertEqual(pins,{'body':'b1'})
    def test_source_revision_rechecks_but_frozen_view_keeps_original_state(self):
        a=self.add_body_case()
        with self.store.db:k=self.write(self.op([self.canonical_ref(a)]),[a])
        old_at=self.store.revision;frontier=self.session.source_frontier()
        patch={**self.case_patch(),'op':'revise','claim_id':a['claim_id'],'claim_revision':a['revision'],'text':'The relation was reinterpreted.'};patch.pop('local')
        revision=self.store.submit(patch,allowed_sources=self.session.allowed,cutoff=self.session.cutoff,
            dependencies=self.deps,episode=self.session.episode,visible_revisions={a['claim_id']:a['revision']})
        self.assertTrue(revision['accepted'],revision)
        frozen=library.read_knowledge(self.store,'',context={},dependencies=self.deps,ids=[k['id']],case_at=old_at,frontier=frontier)[0]
        current=library.read_knowledge(self.store,'',context={},dependencies=self.deps,ids=[k['id']])[0]
        self.assertEqual(frozen['source_revised'],[]);self.assertEqual(current['source_revised'],[a['claim_id']])
    def test_latest_visible_version_survives_future_source_and_retirement_does_not_revive(self):
        a=self.add_body_case();op=self.op([self.canonical_ref(a)])
        with self.store.db:first=self.write(op,[a])
        old_at=self.store.revision;frontier=self.session.source_frontier()
        future=self.observation();b=self.add_body_case(future,text='A later source not available in the old snapshot.')
        revised={**op,'op':'revise','id':first['id'],'refs':[self.canonical_ref(a),self.canonical_ref(b)]}
        with self.store.db:
            second=library.publish_knowledge(self.store,revised,allowed=set(),visible_cases={a['claim_id']:a['revision'],b['claim_id']:b['revision']},visible_knowledge={first['id']:first['revision']},dependencies=self.deps,frontier=self.session.source_frontier())
        frozen=library.read_knowledge(self.store,'',context={},dependencies=self.deps,ids=[first['id']],case_at=old_at,frontier=frontier)[0]
        self.assertEqual(frozen['revision'],first['revision'])
        with self.store.db:
            retired=library.publish_knowledge(self.store,{'op':'retire','id':first['id'],'reason':'Superseded'},allowed=set(),visible_cases={},visible_knowledge={first['id']:second['revision']},dependencies=self.deps,frontier=self.session.source_frontier())
        latest=library.read_knowledge(self.store,'',context={},dependencies=self.deps,ids=[first['id']])[0]
        self.assertEqual(latest['revision'],retired['revision']);self.assertEqual(latest['status'],'retired');self.assertFalse(latest['default_eligible'])
    def test_counter_cannot_silently_disappear(self):
        a=self.add_body_case();b=self.add_body_case(text='A contrary interpretation.')
        op=self.op([self.canonical_ref(a),self.canonical_ref(b,'counter')])
        with self.store.db:first=self.write(op,[a,b])
        revised={**op,'op':'revise','id':first['id'],'refs':[self.canonical_ref(a)]}
        with self.assertRaisesRegex(ValueError,'Retain counter'):
            library.publish_knowledge(self.store,revised,allowed=set(),visible_cases={a['claim_id']:a['revision']},visible_knowledge={first['id']:first['revision']},dependencies=self.deps,frontier=self.session.source_frontier())
    def test_inherited_refs_do_not_grant_case_or_raw_read(self):
        a=self.add_body_case();m=self.manager();ref=self.canonical_ref(a)
        stripped={k:v for k,v in ref.items() if k!='revision'}
        self.assertEqual(m._resolve_ref(stripped,{},[ref],m.visible_revisions,m.visible_knowledge),ref)
        self.assertFalse(m.allowed);self.assertFalse(m.visible_revisions)
        with self.assertRaises(ValueError):m._resolve_ref({**stripped,'reason':'Changed interpretation'}, {},[ref],m.visible_revisions,m.visible_knowledge)
    def test_read_card_roundtrips_dependency_names_for_ref_only_revision(self):
        a=self.add_body_case(depends_on=['body','cameras']);b=self.add_body_case(text='A separate related grip observation.')
        op=self.op([self.canonical_ref(a)],domain='external',family='retention_release',depends_on=['body','cameras'],
            params=[],unknowns=['Grip equivalence is unestablished.'])
        with self.store.db:first=self.write(op,[a])
        m=self.manager();self.read_case(m,b)
        answer=m._search({'question':'Read this existing item.','level':'knowledge','ids':[first['id']]});m._admit(answer)
        card=answer['records'][0]
        revised={k:card[k] for k in ('domain','family','question','text','conditions','strength','use','unknowns','params','depends_on')}
        revised.update(op='revise',id=card['id'],refs=[{k:v for k,v in r.items() if k!='revision'} for r in card['drill_down']['refs']]
            +[{'type':'case','id':b['claim_id'],'relation':'related','reason':'Different grip, not equivalent support.'}])
        m._publish({'expected_version':1,'answer':'Retain the qualified related pointer.','knowledge_ops':[revised]})
        latest=self.store.db.execute('SELECT * FROM reusable_knowledge ORDER BY revision DESC LIMIT 1').fetchone()
        self.assertEqual(json.loads(latest['dependencies']),{'body':'b1','cameras':'c1'})
        self.assertNotIn(a['claim_id'],m.visible_revisions)
        refs=json.loads(latest['refs']);self.assertEqual(refs[0]['revision'],a['revision']);self.assertEqual(refs[1]['relation'],'related')
    def test_camera_selection_and_compact_body_do_not_expand_full_projections(self):
        path=Path(self.temp.name)/'rgb.jpg';path.write_bytes(b'RGB')
        image={'path':str(path),'sha256':hashlib.sha256(b'RGB').hexdigest()}
        obs=self.observation(images={'head':image,'wrist':image},body_visual_reference={'large_projection':'DO_NOT_EXPAND'})
        m=self.manager();m.frontier=self.session.source_frontier()
        result,parts=m._evidence({'ids':[obs['id']],'cameras':['head'],'body':'ee'})
        self.assertEqual(len([p for p in parts if p['type']=='image_url']),1)
        self.assertNotIn('DO_NOT_EXPAND',json.dumps(result));self.assertIn(obs['id'],m.seen_images)
        self.assertFalse(m.allowed)
        with self.assertRaises(ValueError):m._evidence({'ids':[obs['id']],'cameras':['absent']})

    def test_new_case_revision_is_pending_after_old_version_kept(self):
        a=self.add_body_case();m=self.manager();self.read_case(m,a)
        m._publish({'expected_version':1,'answer':'Keep local case','dispositions':[
            {'case':a['claim_id'],'action':'keep','reason':'Insufficient transferable relation.'}]})
        self.assertEqual(m._case_batch(None)['records'],[])
        patch={**self.case_patch(),'op':'revise','claim_id':a['claim_id'],'claim_revision':a['revision'],'text':'Updated local relation.'};patch.pop('local')
        r=self.store.submit(patch,allowed_sources=self.session.allowed,cutoff=self.session.cutoff,
            dependencies=self.deps,episode=self.session.episode,visible_revisions={a['claim_id']:a['revision']})
        self.assertTrue(r['accepted']);library.organize_archive(self.store);m.revision=self.store.revision
        self.assertEqual([c['revision'] for c in m._case_batch(None)['records']],[r['revision']])
    def test_retired_related_knowledge_warns_without_recursive_qualification(self):
        a=self.add_body_case();op=self.op([self.canonical_ref(a)])
        with self.store.db:first=self.write(op,[a])
        dependent=self.op([self.canonical_ref(a),{'type':'knowledge','id':first['id'],'revision':first['revision'],'relation':'related','reason':'Historical candidate navigation.'}])
        with self.store.db:
            second=library.publish_knowledge(self.store,dependent,allowed=set(),visible_cases={a['claim_id']:a['revision']},
                visible_knowledge={first['id']:first['revision']},dependencies=self.deps,frontier=self.session.source_frontier())
            library.publish_knowledge(self.store,{'op':'retire','id':first['id'],'reason':'Reinterpreted'},allowed=set(),visible_cases={},
                visible_knowledge={first['id']:first['revision']},dependencies=self.deps,frontier=self.session.source_frontier())
        view=library.read_knowledge(self.store,'',context={},dependencies=self.deps,ids=[second['id']])[0]
        self.assertNotIn('basis_stale',view)
        self.assertEqual(view['related_warnings'][0]['id'],first['id'])

    def test_preserving_ref_does_not_license_broadening_unread_case_interpretation(self):
        a=self.add_body_case();op=self.op([self.canonical_ref(a)])
        with self.store.db:first=self.write(op,[a])
        broad={**op,'op':'revise','id':first['id'],'text':'The effect necessarily occurs for every pose.'}
        with self.assertRaisesRegex(ValueError,'actual case reads'):
            library.publish_knowledge(self.store,broad,allowed=set(),visible_cases={},visible_knowledge={first['id']:first['revision']},
                dependencies=self.deps,frontier=self.session.source_frontier())
        unchanged={**op,'op':'revise','id':first['id']}
        with self.store.db:
            result=library.publish_knowledge(self.store,unchanged,allowed=set(),visible_cases={},visible_knowledge={first['id']:first['revision']},
                dependencies=self.deps,frontier=self.session.source_frontier())
        self.assertGreater(result['revision'],first['revision'])

    def test_different_free_text_descriptions_remain_unknown_not_incompatible(self):
        a=self.add_body_case();op=self.op([self.canonical_ref(a)],conditions={'support':'tabletop'})
        with self.store.db:k=self.write(op,[a])
        view=library.read_knowledge(self.store,'',context={'support':'table'},dependencies=self.deps,ids=[k['id']])[0]
        self.assertEqual(view['applicability']['different_conditions'],[])
        self.assertEqual(view['applicability']['unknown_conditions'],['support'])
        self.assertIn('support',view['applicability']['different_descriptions'])
    def test_prepare_cannot_absorb_a_later_unread_knowledge_version(self):
        op=self.op([{'type':'evidence','id':self.prior['id']}],strength='configuration')
        with self.store.db:first=self.write(op,[])
        m=self.manager();m.purpose='prepare';m.visible_knowledge={first['id']:first['revision']}
        original_at=m.knowledge_revision
        with self.store.db:
            second=library.publish_knowledge(self.store,{**op,'op':'revise','id':first['id'],'text':'New unpublished preparation wording.'},
                allowed={self.prior['id']},visible_cases={},visible_knowledge={first['id']:first['revision']},dependencies=self.deps,frontier=m.frontier)
        result,_=m._publish({'expected_version':1,'answer':'Select read version','default_knowledge':[first['id']]})
        self.assertEqual(result['default_knowledge'][0]['revision'],original_at)
        self.assertNotEqual(result['default_knowledge'][0]['revision'],second['revision'])
        self.assertEqual(m.prefix_view()['knowledge'],[])  # Later unreviewed version is not silently adopted.

    def test_related_unknown_contract_is_warning_while_own_contract_remains_strict(self):
        a=self.add_body_case();op=self.op([self.canonical_ref(a)])
        with self.store.db:first=self.write(op,[a])
        second_op={**op,'op':'add','domain':'external','family':'contact_response','depends_on':['body'],'refs':[
            self.canonical_ref(a),{'type':'knowledge','id':first['id'],'revision':first['revision'],'relation':'related','reason':'Historical candidate navigation.'}]}
        with self.store.db:
            second=library.publish_knowledge(self.store,second_op,allowed=set(),visible_cases={a['claim_id']:a['revision']},
                visible_knowledge={first['id']:first['revision']},dependencies=self.deps,frontier=self.session.source_frontier())
        view=library.read_knowledge(self.store,'',context={},dependencies={},ids=[second['id']])[0]
        self.assertNotIn('basis_stale',view)
        self.assertIn('dependency_unknown',view['related_warnings'][0]['flags'])
        self.assertEqual(view['applicability']['unknown_dependencies'],['body'])
    def test_manager_response_does_not_reinject_full_publication(self):
        from unittest.mock import patch
        m=self.manager()
        internal={'status':'PASS','answer':'A short scoped answer','publication':'p1','version':2,'knowledge_updates':[],
            'default_knowledge':[{'text':'VERBOSE_AUDIT_SENTINEL'}],'case_dispositions':{'c':{'reason':'VERBOSE_AUDIT_SENTINEL'}}}
        with patch.object(m,'run',return_value=internal):answer=json.loads(self.session.manage_memory({'purpose':'review','question':'Which condition differs?'}))
        self.assertEqual(answer['answer'],internal['answer']);self.assertNotIn('VERBOSE_AUDIT_SENTINEL',json.dumps(answer))

    def test_specific_response_is_not_displaced_by_fewer_recorded_conditions(self):
        strong=self.add_body_case(text='Gripper marker withdrawal response changed in the recorded interval.',
            conditions={'motion':'withdrawal','camera':'head','orientation':'recorded pose','support':'unknown'})
        self.add_body_case(text='A gripper marker appears among other landmarks in a general visual description without an interaction observation.',conditions={})
        answer=json.loads(self.session.query_experience({'question':'gripper marker withdrawal response','level':'case','domain':'body','family':'geometry'}))
        self.assertEqual(answer['records'][0]['id'],strong['claim_id'])
    def test_preparation_inventory_exposes_empty_knowledge_families_without_hiding_cases(self):
        self.add_body_case();m=self.manager();initial=json.loads(self.client.inputs[0][1]['content'])
        route=next(c for c in initial['navigation']['case_counts'] if c['family']=='geometry')
        self.assertEqual(route['knowledge_items'],0);self.assertGreater(route['cases'],0)
