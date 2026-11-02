"""Budget and result gates for serial experiments, using no real calls."""
import copy
import unittest
from main_experiment import validate_closed,cumulative

class ExperimentTests(unittest.TestCase):
    def report(self):
        p={'cli_invocations':5,'structured_responses_accepted':5,'outcome_unknown':0}
        m={'cli_invocations':100,'structured_responses_accepted':100,'outcome_unknown':0}
        return {'status':'PASS','sources_unchanged':True,'task_outcome':'NEEDS_WORK',
            'model_transport':p,'management_transport':m,'terminal_review':{'robot_actions':0}}
    def test_normal_task_failure_and_exact_budget_end_are_closed(self):
        validate_closed(self.report(),5,100,{'primary':100,'manager':100})
    def test_known_native_failure_consumes_allowance_without_becoming_accepted(self):
        r=self.report();r['model_transport'].update(
            structured_responses_accepted=4,native_failed_turns=1)
        validate_closed(r,5,100,{'primary':100,'manager':100})
        r['model_transport']['outcome_unknown']=1
        with self.assertRaises(ValueError):validate_closed(r,5,100,{'primary':100,'manager':100})
    def test_unknown_missing_response_or_terminal_cannot_advance(self):
        for mode in ('unknown','missing','terminal','source','count'):
            r=self.report()
            if mode=='unknown':r['model_transport']['outcome_unknown']=1
            elif mode=='missing':r['management_transport']['structured_responses_accepted']=99
            elif mode=='terminal':r.pop('terminal_review')
            elif mode=='source':r['sources_unchanged']=False
            elif mode=='count':r['model_transport']['cli_invocations']=6
            with self.assertRaises(ValueError):validate_closed(r,5,100,{'primary':100,'manager':100})
    def test_aggregates_roles_without_double_counting_cache(self):
        r=self.report();p={**r['model_transport'],'usage_reported':{'input_tokens':100,'cached_input_tokens':80,'output_tokens':10}}
        m={**r['management_transport'],'usage_reported':{'input_tokens':30,'output_tokens':5}}
        out=cumulative([{'primary':p,'manager':m}])
        self.assertEqual(out['primary']['tokens']['input_tokens'],100)
        self.assertEqual(out['primary']['tokens']['cached_input_tokens'],80)
        self.assertEqual(out['manager']['calls'],100)
