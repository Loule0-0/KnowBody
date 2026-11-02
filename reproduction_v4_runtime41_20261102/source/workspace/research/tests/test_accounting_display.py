"""Unknown usage must remain unknown through reports, CSV and plots."""
from pathlib import Path
import copy,csv,json,tempfile,unittest
import continual_experiment as study
import main_experiment as main


def row(order=1):
    return {'order':order,'layout':order,'native_reset_seed':order,'task_outcome':'PASS',
        'recording':'PASS','termination':'official_success','env_steps':100,'env_step_limit':1200,
        'artifact':'/nonexistent/offline-artifact','memory_counts':{'cases':1,'knowledge':1,'knowledge_active':1},
        'primary':{'cli_invocations':2,'usage_reported':{'input_tokens':100,'output_tokens':10},
            'captured_metadata_consistent':True,'usage_scope':'native_responses'},
        'manager':{'cli_invocations':1,'usage_reported':{'input_tokens':50,'output_tokens':20},
            'captured_metadata_consistent':True,'usage_scope':'native_responses'}}


class AccountingDisplay(unittest.TestCase):
    def test_complete_then_missing_then_complete_cumulative_stays_unknown(self):
        rows=[row(i) for i in (1,2,3)]
        rows[1]['primary'].update(usage_reported={},captured_metadata_consistent=False)
        before=copy.deepcopy(rows);result=study.metric_rows(rows)
        self.assertEqual([r['total_tokens'] for r in result],[180,None,180])
        self.assertEqual([r['cumulative_tokens'] for r in result],[180,None,None])
        self.assertIsNone(result[-1]['cumulative_successes_per_million_tokens'])
        self.assertEqual(rows,before)
        for coordinator in (study,main):
            totals=coordinator.cumulative(rows)
            self.assertEqual(totals['primary']['calls'],6)
            self.assertEqual(totals['primary']['tokens']['input_tokens'],200)
            self.assertIsNone(totals['primary']['complete_total_tokens'])
            self.assertEqual(totals['manager']['complete_total_tokens'],210)

    def test_partial_or_conflicting_metadata_does_not_certify_observed_numbers(self):
        value=row();value['primary']['captured_metadata_consistent']=False
        result=study.metric_rows([value])[0]
        self.assertEqual(result['primary_observed_total_tokens'],110)
        self.assertIsNone(result['primary_total_tokens']);self.assertIsNone(result['total_tokens'])

    def test_historical_coverage_missing_stays_unknown_with_original_scope(self):
        value=row();value['primary'].pop('captured_metadata_consistent')
        value['primary']['usage_scope']='historical_stdout'
        result=study.metric_rows([value])[0]
        self.assertEqual(result['primary_usage_scope'],'historical_stdout')
        self.assertEqual(result['primary_observed_total_tokens'],110)
        self.assertIsNone(result['primary_usage_metadata_consistent'])
        self.assertIsNone(result['total_tokens'])

    def test_explicit_reported_zero_is_preserved(self):
        value=row()
        for role in ('primary','manager'):value[role]['usage_reported']={'input_tokens':0,'output_tokens':0}
        result=study.metric_rows([value])[0]
        self.assertEqual(result['total_tokens'],0);self.assertEqual(result['cumulative_tokens'],0)
        self.assertIsNone(result['cumulative_successes_per_million_tokens'])

    def test_all_unknown_and_single_unknown_role_render_without_fake_zero(self):
        for missing in (('primary',),('primary','manager')):
            with self.subTest(missing=missing),tempfile.TemporaryDirectory() as root:
                root=Path(root);value=row()
                for role in missing:value[role].update(usage_reported={},captured_metadata_consistent=False)
                (root/'PROTOCOL.json').write_text(json.dumps({'prior_episodes':0,'episodes':3}))
                study.render(root,{'status':'RUNNING','completed':[value]})
                data=json.loads((root/'METRICS.json').read_text())['episodes'][0]
                self.assertIsNone(data['total_tokens']);self.assertIn('未知',(root/'RESULTS.md').read_text())
                with (root/'EPISODES.csv').open() as f:self.assertEqual(next(csv.DictReader(f))['total_tokens'],'')
                self.assertGreater((root/'LEARNING_CURVE.png').stat().st_size,100)
