"""Unexecuted requests remain discoverable without granting evidence or changing packing."""
import copy,json,unittest
from pathlib import Path
import test_manager
from kyb import episode_index
from kyb.readout import size
from test_terminal_cost_navigation import index


class MotionDirectory(unittest.TestCase):
    setUp=test_manager.ManagerTests.setUp
    tearDown=test_manager.ManagerTests.tearDown
    observation=test_manager.ManagerTests.observation
    attach=test_manager.ManagerTests.attach

    def request(self):
        decision=self.session.evidence('policy_decision',{'tool':'move_eef','observation_id':self.post['id'],'arguments':{'note':'Conditional request'}})
        plan=self.session.evidence('planning_result',{'status':'Success','executed_actions':0,'before_observation_id':self.post['id']})
        check=self.session.evidence('path_validation',{'status':'NEEDS_WORK','executed_actions':0,'before_observation_id':self.post['id']})
        return decision,plan,check

    def test_pages_only_registered_motion_ids_in_exact_range(self):
        expected=[]
        for _ in range(5):
            expected.extend(self.request())
            self.session.evidence('policy_decision',{'tool':'query_experience','arguments':{'question':'Unrelated'}})
            self.trace.event('planning_result',{'status':'NOT_REGISTERED'})
        cursor=self.post['sequence'];end=self.session.source_frontier()[self.session.episode];seen=[]
        while cursor is not None:
            page=episode_index.directory(self.store,self.session.episode,cursor,end,'motion')
            self.assertLessEqual(size(page),6000)
            self.assertNotIn('records',page);self.assertNotIn('data',page)
            seen.extend(page['entries']);cursor=page['next_after']
        self.assertEqual([r['id'] for r in seen],[r['id'] for r in expected])
        self.assertEqual([r['status'] for r in seen if r['kind']!='policy_decision'],['Success','NEEDS_WORK']*5)
        self.assertTrue(all(set(row)<= {'id','sequence','kind','status','tool'} for row in seen))
        short=episode_index.directory(self.store,self.session.episode,self.post['sequence'],expected[1]['sequence'],'motion')
        self.assertEqual([r['id'] for r in short['entries']],[r['id'] for r in expected[:2]])
        self.assertNotIn(expected[2]['id'],json.dumps(short))
        self.assertFalse(episode_index.directory(self.store,'foreign',0,end,'motion')['entries'])

    def test_byte_budget_paginates_complete_utf8_entries_without_skipping_a_source(self):
        expected=[]
        for _ in range(18):expected.extend(self.request())
        end=self.session.source_frontier()[self.session.episode];cursor=self.post['sequence'];seen=[];pages=0
        while cursor is not None:
            page=episode_index.directory(self.store,self.session.episode,cursor,end,'motion',budget_bytes=1000)
            self.assertLessEqual(size(page),1000);self.assertTrue(page['entries'])
            self.assertGreater(page['entries'][0]['sequence'],cursor)
            self.assertLessEqual(page['entries'][-1]['sequence'],end)
            seen.extend(r['id'] for r in page['entries']);cursor=page['next_after'];pages+=1
        self.assertGreater(pages,1)
        self.assertEqual(seen,[r['id'] for r in expected])
        # A large envelope or one unusually long ID cannot produce a non-advancing page.
        with self.assertRaisesRegex(ValueError,'envelope'):
            episode_index.directory(self.store,self.session.episode,0,end,'motion',budget_bytes=100)
        self.session.evidence('planning_result',{'status':'界' * 1000})
        with self.assertRaisesRegex(ValueError,'one complete'):
            episode_index.directory(self.store,self.session.episode,end,self.session.source_frontier()[self.session.episode],
                                    'motion',budget_bytes=1000)

    def test_ten_motion_pointers_fit_one_page_when_bytes_allow_more_than_six(self):
        expected=[]
        for _ in range(3):expected.extend(self.request())
        expected.append(self.session.evidence('policy_decision',{'tool':'hold','arguments':{}}))
        page=episode_index.directory(self.store,self.session.episode,self.post['sequence'],expected[-1]['sequence'],'motion')
        self.assertEqual([r['id'] for r in page['entries']],[r['id'] for r in expected])
        self.assertIsNone(page['next_after']);self.assertLessEqual(size(page),6000)

    def test_navigation_grants_no_permission_then_explicit_read_is_required(self):
        records=self.request();m=self.attach([test_manager.publish(0)]);m.run('Prepare','prepare')
        m.purpose='review'
        def permissions():return (set(m.allowed),set(m.seen_images),set(m.seen_telemetry),dict(m.visible_revisions),dict(m.visible_knowledge),dict(m.selectable_knowledge))
        before=permissions()
        page=episode_index.directory(self.store,self.session.episode,self.post['sequence'],m.frontier[self.session.episode],'motion')
        m._admit(page);self.assertEqual(permissions(),before)
        result,_=m._evidence({'ids':[r['id'] for r in records],'cameras':[],'body':'none'})
        self.assertEqual(permissions(),before)
        m._admit(result);self.assertEqual(m.allowed,{r['id'] for r in records})
        self.assertFalse(m.seen_images);self.assertFalse(m.seen_telemetry)
        self.assertEqual(result['records'][-1]['data']['executed_actions'],0)

    def test_host_dispatch_preserves_prepare_and_frontier_boundaries(self):
        records=self.request();args={'run':self.session.episode,'after':self.post['sequence'],'through':records[1]['sequence'],'directory':'motion'}
        m=self.attach([('read_trajectory',args),test_manager.publish(0),('read_trajectory',args),
            ('read_trajectory',{**args,'through':1000000}),test_manager.publish(1)])
        m.run('Prepare','prepare');m.run('Compare planning and rejection','review')
        results=[]
        for ref in map(json.loads,(self.trace.root/'events.jsonl').read_text().splitlines()):
            if ref['kind']=='manager_tool_result':
                p=json.loads(Path(ref['path']).read_text())['payload']
                if p['tool']=='read_trajectory':results.append(p['result'])
        self.assertIn('Preparation reads historical episodes only',results[0]['reason'])
        self.assertEqual([r['id'] for r in results[1]['entries']],[r['id'] for r in records[:2]])
        self.assertIn('bounds',results[2]['reason'])
        self.assertFalse(m.allowed);self.assertFalse(m.visible_revisions)

    def test_optional_hint_uses_leftover_budget_without_displacing_old_fields(self):
        original=index();original['accounting']['motion_selection']={'no_receipt_motion_request':0}
        request=copy.deepcopy(original);request['accounting']['motion_selection']['no_receipt_motion_request']=1
        for budget in [1250,1600,2000,2800]:
            base=episode_index.readout(original,budget_bytes=budget,completed=True)
            actual=episode_index.readout(request,budget_bytes=budget,completed=True)
            base['accounting']=copy.deepcopy(actual['accounting'])
            hint=actual.pop('motion_directory',None)
            self.assertEqual(actual,base)
            if hint:
                self.assertEqual(hint,{'run':'episode','after':0,'through':600,'directory':'motion'})
                self.assertLessEqual(size({**actual,'motion_directory':hint}),budget)
        full=episode_index.readout(request)
        self.assertIn('motion_directory',full)
        empty=copy.deepcopy(original);empty['windows']=[];empty['memory_questions']=[]
        exact=size(episode_index.readout(empty))
        empty['accounting']['motion_selection']['no_receipt_motion_request']=1
        self.assertNotIn('motion_directory',episode_index.readout(empty,budget_bytes=exact))
