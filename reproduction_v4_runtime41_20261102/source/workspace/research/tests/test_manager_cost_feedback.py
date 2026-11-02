"""The primary must see real manager expenditure, independently of its own allowance."""
import json
from pathlib import Path
import unittest

from kyb.budget import CallBudget
import test_manager


class ManagerCostFeedback(unittest.TestCase):
    setUp=test_manager.ManagerTests.setUp
    tearDown=test_manager.ManagerTests.tearDown
    observation=test_manager.ManagerTests.observation
    attach=test_manager.ManagerTests.attach
    messages=test_manager.ManagerTests.messages

    def counted_client(self, turns, limit=100):
        manager=self.attach(turns)
        self.client.budget=CallBudget(Path(self.temp.name)/'manager_budget.jsonl',limit)
        complete=self.client.complete
        def counted(messages,tools):
            n=self.client.budget.view()['used']
            self.client.budget.reserve('manager_native_'+str(n),'manager')
            return complete(messages,tools)
        self.client.complete=counted
        return manager

    def test_prepare_cost_is_in_episode_total_but_not_next_review_delta(self):
        manager=self.counted_client([test_manager.publish(0),
            ('read_evidence',{'ids':[]}),test_manager.publish(1)])
        manager.run('prepare','prepare')
        answer=json.loads(self.session.manage_memory({'purpose':'review','question':'One unresolved condition'}))
        self.assertEqual(answer['manager_call_usage'],{'native_calls_this_invocation':2,'limit':100,'used':3,'remaining':97})
        messages=self.messages();self.session.refresh_context(messages,None)
        # The dynamic budget survives native context replacement without becoming static knowledge.
        dynamic=json.loads(messages[-1]['content'][0]['text'].split('<KYB_CONTEXT>\n')[1].split('\n</KYB_CONTEXT>')[0])
        self.assertEqual(dynamic['manager_call_budget'],{'limit':100,'used':3,'remaining':97})
        self.assertNotIn('manager_call_budget',json.dumps(manager.prefix_view()))

    def test_exhausted_review_retains_spent_calls_and_unknown_budget_is_not_zero(self):
        manager=self.attach([test_manager.publish(0)])
        manager.run('prepare','prepare')
        # An unconfigured offline client reports unknown; do not turn it into a zero-cost success.
        self.client.turns.append(test_manager.publish(1))
        result=json.loads(self.session.manage_memory({'purpose':'review','question':'One question'}))
        self.assertIsNone(result['manager_call_usage'])
        self.client.budget=CallBudget(Path(self.temp.name)/'exhausted.jsonl',1)
        self.client.budget.reserve('already_failed_turn','manager')
        result=json.loads(self.session.manage_memory({'purpose':'review','question':'Cannot spend again'}))
        self.assertEqual(result['status'],'NEEDS_WORK')
        self.assertEqual(result['manager_call_usage'],{'native_calls_this_invocation':0,'limit':1,'used':1,'remaining':0})
        self.assertEqual(len(self.client.inputs),2)
        # Invalid arguments never invoke a manager or reserve budget.
        rejected=self.session.manage_memory({'purpose':'invalid','question':'Bad request'})
        self.assertIn('no manager called',rejected)
        self.assertEqual(self.client.budget.view()['used'],1)
