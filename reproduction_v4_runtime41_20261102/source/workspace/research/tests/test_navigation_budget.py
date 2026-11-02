"""Bounded discovery labels and frontier-safe diagnostic windows."""
import copy
from types import SimpleNamespace
import unittest

from kyb.readout import pack, page, size, subject
import test_segments
import test_manager


class NavigationBudgetTests(unittest.TestCase):
    def test_many_oversized_candidates_remain_reachable_without_crowding_first_view(self):
        records=[{'id':f'case{i}', 'revision':i, 'layer':'case', 'question':f'key bow subject {i}',
                  'text':'physical limitation ' * 900, 'conditions':{'mount':'unchanged'}} for i in range(9)]
        session=SimpleNamespace()
        answer=pack(session,records,{'question':'key bow'},budget_bytes=3500,limit=2)
        self.assertLessEqual(size(answer),3500)
        self.assertEqual(len(answer['records']),1)
        self.assertEqual(answer['unread_records'],8)
        selection,offset=session._read_continuations[answer['selection_continuation']]
        self.assertEqual(offset,1);self.assertEqual(selection['records'],records)
        self.assertIn('warnings',answer['unread_qualification'])
        # A remaining record is still its original qualified view, not a preview
        # with silently omitted limitations. Read it through the saved selection.
        last=pack(session,selection['records'],selection['answer'],budget_bytes=3500,
                  limit=1,offset=8,view=selection['requested_view'])
        card=last['records'][0];parts=[card['fragment']];token=card['continuation']
        while token:
            p=page(session,token,1100);parts.append(p['fragment']);token=p['continuation']
        import json
        recovered=json.loads(''.join(parts))
        self.assertEqual(recovered['conditions'],records[-1]['conditions'])
        self.assertEqual(recovered['text'],records[-1]['text'])
        self.assertTrue(recovered['complete_record'])

    def test_utf8_title_and_too_small_budget_are_explicit(self):
        record={'id':'case', 'revision':1,'question':'钥匙抓持' * 100, 'text':'evidence'}
        saved=copy.deepcopy(record);label=subject(record)
        self.assertLessEqual(len(label['title'].encode()),240)
        self.assertTrue(label['title_truncated']);self.assertEqual(record,saved)
        with self.assertRaisesRegex(ValueError,'narrow'):
            pack(SimpleNamespace(),[record],{},budget_bytes=180,limit=1)


class SegmentRunTests(unittest.TestCase):
    setUp=test_segments.SegmentTests.setUp
    tearDown=test_segments.SegmentTests.tearDown

    def test_explicit_run_retains_matched_anchor_in_whole_window(self):
        from test_segments import SOURCE
        result=self.store.read_segments('key bow lift',context={'task':'insert_key'},dependencies=self.deps,
            current_observation=self.current,episode='new_run',evidence_cutoff={SOURCE.name:478},run=SOURCE.name)
        segment=result['episodic_segments'][0]
        matched=segment['id'].removesuffix(':history')
        self.assertIn(matched,[a['receipt_id'] for a in segment['whole_window_intervals']])
        self.assertTrue(all(a['receipt_id'].startswith(SOURCE.name+':') for a in segment['whole_window_intervals']))
        with self.assertRaisesRegex(ValueError,'frontier'):
            self.store.read_segments('key',context={},dependencies={},current_observation={},episode='new_run',
                evidence_cutoff={SOURCE.name:478},run='future_run')
        with self.assertRaisesRegex(ValueError,'frontier'):
            self.store.read_segments('key',context={},dependencies={},current_observation={},episode=SOURCE.name,
                evidence_cutoff={},current_history_before=478,run=SOURCE.name)


class ManagerSegmentScopeTests(unittest.TestCase):
    setUp=test_manager.ManagerTests.setUp
    tearDown=test_manager.ManagerTests.tearDown
    observation=test_manager.ManagerTests.observation
    attach=test_manager.ManagerTests.attach

    def test_prepare_rejects_current_run_even_if_frontier_contains_it(self):
        manager=self.attach([]);manager.frontier=self.session.source_frontier();manager.purpose='prepare'
        with self.assertRaisesRegex(ValueError,'historical'):
            manager._segments({'question':'key','run':self.session.episode})
