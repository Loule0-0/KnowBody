"""Native accounting correctness without inference, private content or network."""
import json
import tempfile
import unittest
from pathlib import Path
from kyb.native_usage import read_rollout_usage,merge_usage_snapshots,transport_notices,summarize_role


def usage(n=1):
    return {'input_tokens':10*n,'cached_input_tokens':4*n,'cache_write_input_tokens':0,
        'output_tokens':3*n,'reasoning_output_tokens':n}


def snapshot(sid='s',response='r',n=1,compressed=False,model='m'):
    return {'records':[{'session_id':sid,'response_id':response,'turn_id':'t',
        'root_turn_id':'t','usage':usage(n),'model':model,'effort':'high' if model else None}],
        'thread_totals':{sid:usage(n)},'compactions':[[sid,response]] if compressed else [],'issues':[]}


class NativeUsageTests(unittest.TestCase):
    def test_repeated_cumulative_copies_count_once_in_either_order(self):
        old=snapshot();new=snapshot(response='r2')
        new['records']=old['records']+new['records'];new['thread_totals']['s']=usage(2)
        for snapshots in ([old,new,new],[new,old]):
            result=merge_usage_snapshots(snapshots)
            self.assertTrue(result['captured_metadata_consistent'])
            self.assertEqual(result['usage_reported'],usage(2))
            self.assertEqual(result['reported_native_responses'],2)

    def test_same_response_name_in_other_session_is_not_compression(self):
        result=merge_usage_snapshots([snapshot(compressed=True),snapshot(sid='other')])
        self.assertEqual(result['reported_compaction_responses'],1)
        self.assertEqual(result['compaction_usage_reported'],usage())
        self.assertEqual(result['non_compaction_usage_reported'],usage())

    def test_missing_model_can_be_enriched_without_losing_tokens(self):
        result=merge_usage_snapshots([snapshot(model=None),snapshot()])
        self.assertTrue(result['captured_metadata_consistent'])
        self.assertEqual(result['usage_reported'],usage())
        self.assertEqual(result['models'][0]['model'],'m')

    def test_model_conflict_keeps_known_tokens_with_unknown_attribution(self):
        result=merge_usage_snapshots([snapshot(),snapshot(model='different')])
        self.assertFalse(result['captured_metadata_consistent'])
        self.assertEqual(result['usage_reported'],usage())
        self.assertIsNone(result['models'][0]['model'])

    def test_usage_conflict_is_not_silently_selected(self):
        result=merge_usage_snapshots([snapshot(),snapshot(n=2)])
        self.assertFalse(result['captured_metadata_consistent'])
        self.assertEqual(result['reported_native_responses'],0)

    def test_empty_and_missing_compression_usage_are_explicit(self):
        self.assertFalse(merge_usage_snapshots([])['captured_metadata_consistent'])
        s=snapshot();s['compactions']=[['s','missing']]
        result=merge_usage_snapshots([s])
        self.assertIn('compaction_usage_not_reported',{i['code'] for i in result['usage_issues']})

    def test_parse_only_metadata_including_partial_tail(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/'native.jsonl'
            events=[{'type':'session_meta','payload':{'id':'s'}},
                {'type':'turn_context','payload':{'turn_id':'t','model':'m','effort':'high'}},
                {'type':'token_usage_record','payload':{'thread_id':'s','session_id':'s',
                    'response_id':'r','turn_id':'t','usage':usage(),'thread_token_usage':usage(),
                    'turn_token_usage':usage(100)}},
                {'type':'compacted','payload':{'compaction_response_id':'r','message':'PRIVATE_SENTINEL'}},
                {'type':'response_item','payload':{'type':'reasoning','text':'PRIVATE_SENTINEL'}}]
            p.write_text('\n'.join(map(json.dumps,events))+'\n{partial')
            data=read_rollout_usage(p);result=merge_usage_snapshots([data])
            self.assertNotIn('PRIVATE_SENTINEL',json.dumps(data))
            self.assertEqual(result['compaction_usage_reported'],usage())
            self.assertFalse(result['captured_metadata_consistent'])

    def test_native_subsets_are_not_added_to_input_output(self):
        result=merge_usage_snapshots([snapshot()])
        self.assertEqual(result['usage_reported']['input_tokens']+result['usage_reported']['output_tokens'],13)

    def test_transport_notices_are_not_request_counts(self):
        result=transport_notices([{'type':'error','message':'Reconnecting 2/5'},
            {'type':'item.completed','item':{'type':'error','message':'Falling back from WebSockets to HTTPS'}},
            {'type':'item.completed','item':{'type':'reasoning','text':'Reconnecting 3/5'}}])
        self.assertEqual(result['reconnect_notices'],1);self.assertTrue(result['https_fallback'])
        self.assertIsNone(result['actual_transport_or_model_request_count'])

    def test_real_reconnect_punctuation_is_counted_only_in_public_errors(self):
        result=transport_notices([{'type':'error','message':'Reconnecting... 2/5 (transport error)'},
            {'type':'item.completed','item':{'type':'error','message':'Reconnecting… 3/5'}},
            {'type':'item.completed','item':{'type':'reasoning','text':'Reconnecting... 4/5'}},
            {'type':'error','message':'exceeded retry limit, last status:429'}])
        self.assertEqual(result['reconnect_notices'],2)
        self.assertEqual(result['reconnect_ordinals'],[[2,5],[3,5]])
        self.assertFalse(result['https_fallback'])
        self.assertIsNone(result['actual_transport_or_model_request_count'])

    def test_completed_attempt_without_archive_cannot_claim_complete_coverage(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root);(p/'cli_calls.jsonl').write_text(json.dumps({'attempt_id':'a',
                'state':'turn_complete','thread_id':'s','native_completed_turns':1,
                'usage':{'input_tokens':10,'output_tokens':3}})+'\n')
            result=summarize_role(p)
            self.assertFalse(result['captured_metadata_consistent'])
            self.assertEqual(result['usage_reported'],{})
            self.assertEqual(result['stdout_usage_reported']['input_tokens'],10)
            self.assertIsNone(result['bill_amount'])

    def test_compression_record_cannot_mask_missing_ordinary_response(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root);directory=p/'cli'/'a';directory.mkdir(parents=True)
            (p/'cli_calls.jsonl').write_text(json.dumps({'attempt_id':'a','state':'turn_complete',
                'thread_id':'s','native_completed_turns':1,'usage':{'input_tokens':10,'output_tokens':3}})+'\n')
            events=[{'type':'session_meta','payload':{'id':'s'}},
                {'type':'token_usage_record','payload':{'thread_id':'s','session_id':'s',
                    'response_id':'compression','turn_id':'t','usage':usage(),'thread_token_usage':usage()}},
                {'type':'compacted','payload':{'compaction_response_id':'compression'}}]
            (directory/'native_rollout.jsonl').write_text('\n'.join(map(json.dumps,events)))
            result=summarize_role(p)
            self.assertEqual(result['usage_reported'],usage())
            self.assertIn('native_usage_below_stdout_reported_counter',
                {i['code'] for i in result['usage_issues']})


if __name__=='__main__':unittest.main()
