"""Completed interpretations survive a transaction; source authority does not."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from kyb.memory import MemoryStore
from kyb.session import MemorySession
from kyb.terminal import restore_session
from kyb.trace import RunTrace, write_once
from kyb.readout import size
from manager_fixtures import ScriptedManager


def publish(version, answer, **extra):
    return 'publish_materials', dict(expected_version=version, answer=answer, **extra)


class ManagerContinuityTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.store=MemoryStore(self.root/'memory.sqlite')
        self.trace=RunTrace(self.root/'episode',dict(task='task',perception_profile='rgbd_calibrated'))
        self.session=MemorySession(self.trace,self.store,task='task',dependencies={'body':'b1'})
        self.session.evidence('known_config',{'action_spec':'public config'})
        self.session.task_specification=self.session.evidence('task_specification',{'instruction':'Complete task'})
        self.session.task_instruction='Complete task'
        self.obs=self.session.evidence('observation',{'state':{},'dependencies':{'body':'b1'},'env_step':1})

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def attach(self, turns):
        self.client=ScriptedManager(turns)
        self.session.attach_manager(self.client)
        return self.session.manager

    def events(self):
        return [json.loads(Path(json.loads(line)['path']).read_text())
                for line in (self.trace.root/'events.jsonl').read_text().splitlines()]

    def test_one_complete_result_replaces_previous_without_entering_primary_prefix(self):
        manager=self.attach([publish(0,'No applicable panel result in the returned candidates.',default_knowledge=[]),
                             publish(1,'New question remains unresolved.'),publish(2,'Final interpretation.')])
        manager.run('Select task preparation','prepare')
        prefix=copy.deepcopy(manager.prefix_view())
        manager.run('Compare a new response','review')
        request=json.loads(self.client.inputs[1][1]['content'])
        memo=request['previous_manager_result']
        self.assertEqual(memo['answer'],'No applicable panel result in the returned candidates.')
        self.assertEqual(request['previous_manager_result_bytes'],size(memo))
        self.assertEqual(memo['request_read_view'],{'case_revision':0,'knowledge_revision':0})
        self.assertEqual(memo['current_read_view'],{'case_revision':0,'knowledge_revision':0})
        self.assertIn('not proof of library-wide absence',memo['qualification'])
        self.assertIn('not current physical facts',memo['qualification'])
        self.assertEqual(manager.prefix_view()['knowledge'],prefix['knowledge'])
        self.assertNotIn('No applicable panel',json.dumps(manager.prefix_view()))
        manager.run('One later question','review')
        later=json.loads(self.client.inputs[2][1]['content'])['previous_manager_result']
        self.assertEqual(later['answer'],'New question remains unresolved.')
        self.assertNotIn('No applicable panel',json.dumps(later))

    def test_prior_answer_cannot_grant_raw_sources_or_write_a_case(self):
        patch={'op':'new','kind':'body','module':'geometry','reading_role':'intrinsic_candidate',
            'observed_entity':'robot_part','observable':'relative_geometry','text':'A purported observation.',
            'conditions':{},'reference_frame':'world','epistemic':'observation','evidence':[self.obs['id']]}
        manager=self.attach([publish(0,'Previously mentioned source '+self.obs['id'],default_knowledge=[]),
            publish(1,'An unsupported write.',groups=[{'memory_patch':[patch]}]),
            publish(1,'No source read; no write.')])
        manager.run('Prepare','prepare')
        manager.allowed.add(self.obs['id'])
        manager.seen_telemetry.add(self.obs['id'])
        manager.run('Review without reading','review')
        self.assertFalse(manager.allowed)
        self.assertFalse(manager.seen_telemetry)
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM claims').fetchone()[0],0)
        rejected=[e for e in self.events() if e['kind']=='manager_tool_result'
                  and e['payload']['result'].get('status')=='NEEDS_WORK']
        self.assertTrue(rejected)
        self.assertEqual(manager.previous_result['answer'],'No source read; no write.')

    def test_partial_unknown_or_unmatched_result_does_not_replace_completed(self):
        manager=self.attach([publish(0,'Complete answer.',default_knowledge=[])])
        manager.run('Prepare','prepare')
        events=self.events()
        pub=next(e for e in events if e['kind']=='manager_group_publication')
        completion=next(e for e in events if e['kind']=='manager_tool_result')
        old=copy.deepcopy(manager.previous_result)
        variants=[]
        a=copy.deepcopy(pub);a['payload']['issues']=[{'reason':'rejected group'}];variants.append((a,completion))
        a=copy.deepcopy(pub);a['payload']['status']='NEEDS_WORK';variants.append((a,completion))
        for status in ['NEEDS_WORK','UNKNOWN']:
            c=copy.deepcopy(completion);c['payload']['result']['status']=status;variants.append((pub,c))
        c=copy.deepcopy(completion);c['payload']['result']['publication']='another:e000001';variants.append((pub,c))
        for p,c in variants:
            self.assertFalse(manager.remember_completed_result(p,c))
            self.assertEqual(manager.previous_result,old)

    def test_recovery_keeps_committed_version_and_only_confirmed_pass_interpretation(self):
        manager=self.attach([publish(0,'Completed earlier result.',default_knowledge=[])])
        manager.run('Prepare','prepare')
        previous=next(e for e in self.events() if e['kind']=='manager_group_publication')
        payload=copy.deepcopy(previous['payload'])
        payload.update(version=2,answer='Unconfirmed later answer.')
        latest=self.trace.event('manager_group_publication',payload)
        write_once(self.trace.root/'report.json',{'status':'PASS'})
        recovered=RunTrace(self.root/'recovery',{})
        session,_=restore_session(recovered,self.store,self.trace.root,ScriptedManager([]))
        self.assertEqual(session.manager.version,2)
        self.assertEqual(session.manager.publication,latest['id'])
        self.assertEqual(session.manager.previous_result['publication'],previous['id'])
        self.assertEqual(session.manager.previous_result['answer'],'Completed earlier result.')
        self.assertEqual(session.manager.previous_result['logical_episode'],self.trace.root.name)
        session.manager.client=ScriptedManager([publish(2,'Restored, without source reads.')])
        session.manager.run('Continue the restored review','review')
        self.assertFalse(session.manager.allowed)

    def test_memo_does_not_cross_episode_or_perception_profile(self):
        manager=self.attach([publish(0,'Private to this episode.',default_knowledge=[]),
                             publish(1,'Another question.'),publish(2,'Another profile.')])
        manager.run('Prepare','prepare')
        self.session.episode='different_episode'
        manager.run('New episode question','review')
        self.assertNotIn('previous_manager_result',json.loads(self.client.inputs[1][1]['content']))
        from kyb.perception import select
        self.session.perception=select('rgb_only')
        manager.run('New profile question','review')
        self.assertNotIn('previous_manager_result',json.loads(self.client.inputs[2][1]['content']))

    def test_full_answer_and_unicode_question_label_have_constant_serialized_bound(self):
        answer='\x00'*1800
        manager=self.attach([publish(0,answer,default_knowledge=[]),publish(1,'Later.')])
        manager.run('问'*1200,'prepare')
        manager.run('Later question','review')
        memo=json.loads(self.client.inputs[1][1]['content'])['previous_manager_result']
        self.assertEqual(memo['answer'],answer)
        self.assertTrue(memo['question_truncated'])
        self.assertLessEqual(len(memo['question_title'].encode()),240)
        self.assertLessEqual(size(memo),16384)


if __name__=='__main__':
    unittest.main()
