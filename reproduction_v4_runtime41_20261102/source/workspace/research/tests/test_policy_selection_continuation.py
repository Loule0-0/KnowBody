"""Real act→registered decision→dispatch must not invalidate its own read."""
import json
import unittest
from kyb import library
from kyb.policy import EefAgentPolicy
import test_manager as manager_fixture
import test_policy_pipeline as fixture
from test_hybrid_retrieval import Index


class QueryClient:
    def __init__(self,session,question,cursor_field,change=None):
        self.session,self.question,self.cursor_field,self.change=session,question,cursor_field,change
        self.calls=0;self.results=[]

    def complete(self,messages,tools):
        if self.calls:
            result=json.loads(next(m['content'] for m in reversed(messages) if m['role']=='tool'))
            self.results.append(result)
        if self.calls==0:
            call=fixture.call('query_experience',self.question)
        elif self.calls==1:
            arguments={'continuation':self.results[0][self.cursor_field]}
            if self.change=='attachment':arguments['attachment_updates']=[{'arm':'left','tool':'test tool','state':'visually_supported'}]
            elif self.change in ('planning_result','surface_inspection'):
                self.session.evidence(self.change,{'status':'Fail','executed_actions':0,
                    'before_observation_id':self.session.current_observation['id']})
            call=fixture.call('query_experience',arguments)
        else:call=fixture.call('give_up',{'reason':'End offline scripted test'})
        self.calls+=1
        return {'choices':[{'message':{'role':'assistant','content':None,'tool_calls':[call]},'finish_reason':'tool_calls'}],
                'usage':{'input_tokens':0,'output_tokens':0}}


class PolicySelectionContinuation(unittest.TestCase):
    setUp=fixture.PolicyPipelineTests.setUp
    tearDown=fixture.PolicyPipelineTests.tearDown
    add_body_case=manager_fixture.ManagerTests.add_body_case

    def run_case(self,change=None):
        self.post=self.after
        cases=[self.add_body_case(text='A scoped body relation '+str(i)+' '+('x'*1300)) for i in range(4)]
        query={'level':'case','view':'statement','case_refs':[{'id':c['claim_id'],'revision':c['revision']} for c in cases]}
        client=QueryClient(self.session,query,'selection_continuation',change)
        policy=EefAgentPolicy(action_spec=self.spec,env={},client=client,planner=lambda **_:None,research=self.session)
        policy.act(self.observation)
        return client

    def test_primary_case_cursor_survives_registered_query_decisions(self):
        client=self.run_case()
        self.assertEqual(len(client.results),2)
        self.assertNotEqual(client.results[1]['status'],'NEEDS_WORK',client.results[1])
        self.assertTrue(client.results[1]['records'])
        first={r['id'] for r in client.results[0]['records']};second={r['id'] for r in client.results[1]['records']}
        self.assertFalse(first&second)
        registered=[json.loads(r[0]) for r in self.store.db.execute("SELECT payload FROM evidence WHERE kind='policy_decision' ORDER BY sequence")]
        self.assertEqual([p['tool'] for p in registered[:2]],['query_experience','query_experience'])

    def test_primary_knowledge_cursor_uses_the_same_real_dispatch_boundary(self):
        self.session.retrieval_index=Index()
        with self.store.db:
            for i in range(4):
                self.store.db.execute('INSERT INTO reusable_knowledge(id,domain,family,question,text,conditions,params,dependencies,strength,use_hint,unknowns,refs,status) '
                    'VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)',(f'k{i}','body','geometry','Rotation clearance',
                    'A conditional physical response. '+('x'*1800),'{}','[]','{}','configuration','Compare the actual body.','[]','[]','active'))
        client=QueryClient(self.session,{'question':'rotation clearance'},'candidate_continuation')
        policy=EefAgentPolicy(action_spec=self.spec,env={},client=client,planner=lambda **_:None,research=self.session)
        policy.act(self.observation)
        self.assertNotEqual(client.results[1]['status'],'NEEDS_WORK',client.results[1])
        self.assertTrue(client.results[1]['records'])

    def test_valid_attachment_update_in_the_continuation_call_still_invalidates(self):
        client=self.run_case('attachment')
        self.assertEqual(client.results[1]['status'],'NEEDS_WORK')
        self.assertIn('stale_selection',client.results[1]['reason'])

    def test_new_unexecuted_plan_is_new_evidence_even_without_physical_progress(self):
        client=self.run_case('planning_result')
        self.assertEqual(client.results[1]['status'],'NEEDS_WORK')
        self.assertIn('stale_selection',client.results[1]['reason'])

    def test_new_measurement_still_invalidates_without_a_motion(self):
        client=self.run_case('surface_inspection')
        self.assertEqual(client.results[1]['status'],'NEEDS_WORK')
        self.assertIn('stale_selection',client.results[1]['reason'])
