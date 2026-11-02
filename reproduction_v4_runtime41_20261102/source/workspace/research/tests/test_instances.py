"""Actual support-instance provenance and body-transition regressions."""
import json
import sqlite3
import unittest
from pathlib import Path

from kyb.instances import dated_instance, load_payload
from kyb.segments import SegmentReader


SNAPSHOT=Path('/data/KnowYourBody_RoboDojo/memory/checkpoints/physical_reflection_20261019_rev291.sqlite')
PUSH='20261018T154043_237718419'
KEY='20261018T160232_895713878'


class InstanceTests(unittest.TestCase):
    def setUp(self):
        self.db=sqlite3.connect(f'file:{SNAPSHOT}?mode=ro&immutable=1',uri=True)
        self.db.row_factory=sqlite3.Row

    def tearDown(self):self.db.close()

    def payload(self,identifier):
        return load_payload(self.db,identifier)

    def rows(self):
        return [{'id':f'{PUSH}:e{n:06}','run':PUSH,
            'before_id':self.payload(f'{PUSH}:e{n:06}')['before_observation'],
            'post_id':self.db.execute("SELECT id FROM evidence WHERE kind='observation' AND json_extract(payload,'$.after_receipt')=? ORDER BY sequence LIMIT 1",(f'{PUSH}:e{n:06}',)).fetchone()[0],
            'annotation_ids':[]} for n in (333,344,355,366,377)]

    def test_support_text_uses_its_own_witness_instead_of_original_proposition_example(self):
        original=dated_instance(self.db,222);support=dated_instance(self.db,237)
        self.assertIn('-0.4476',original['observation_text'])
        self.assertNotIn('-0.4476',support['observation_text'])
        self.assertIn('4.4 mm',support['observation_text'])
        self.assertEqual(support['before_observation_id'],f'{PUSH}:e000356')
        self.assertEqual(support['receipt_id'],f'{PUSH}:e000366')
        self.assertEqual(support['post_observation_id'],f'{PUSH}:e000367')
        change=support['measured_body_transition']
        self.assertAlmostEqual(change['before_grasp']['left_x'],-.408233094,places=8)
        self.assertAlmostEqual(change['post_grasp']['left_x'],-.403831622,places=8)
        self.assertAlmostEqual(change['physics_duration_s'],.04)

    def test_whole_window_summary_retains_pre_entry_progress_and_real_call_cost(self):
        reader=SegmentReader(self.db);summary=reader._summary(self.rows(),4)
        self.assertEqual(summary['closest_entry_index'],4)
        self.assertEqual(len(summary['whole_window_receipt_ids']),5)
        transition=summary['measured_start_to_end']
        self.assertAlmostEqual(transition['post_grasp']['left_x']-transition['before_grasp']['left_x'],.0220073467,places=8)
        cost=summary['actual_native_decision_cost']
        self.assertEqual(cost['confirmed_distinct_motion_turns'],5)
        self.assertGreaterEqual(cost['total_CLI_invocations_in_span'],5)
        self.assertIn('merging',summary['qualification'])

    def test_dated_effects_respect_both_knowledge_revision_and_source_frontier(self):
        reader=SegmentReader(self.db);rows=self.rows()
        now=reader._effects(rows,{PUSH:378},264)
        self.assertTrue(any(x['revision']==237 for x in now))
        earlier=reader._effects(rows,{PUSH:378},236)
        self.assertFalse(any(x['revision']==237 for x in earlier))
        source_earlier=reader._effects(rows,{PUSH:366},264)
        self.assertFalse(any(x['revision']==237 for x in source_earlier))
        self.assertFalse(any('-0.4476' in x['observation_text'] for x in now if x['revision']==237))

    def test_actual_transport_loss_body_change_is_not_automatically_a_slip_label(self):
        from kyb.state import body_transition
        receipt=self.payload(f'{KEY}:e001686');before=self.payload(receipt['before_observation']);post=self.payload(f'{KEY}:e001687')
        transition=body_transition(before,receipt,post,.004)
        self.assertAlmostEqual(transition['physics_duration_s'],.28)
        self.assertGreater(sum(transition['before_finger_joints_m']['right']),.019)
        self.assertEqual(transition['post_finger_joints_m']['right'],[0.,0.])
        self.assertNotIn('attachment_state',transition)
        self.assertIn('visual attachment review',transition['qualification'])

    def test_body_labelled_handover_has_bound_skeleton_without_retention_upgrade(self):
        for revision,receipt in ((265,982),(267,1000),(268,1029)):
            instance=dated_instance(self.db,revision)
            self.assertTrue(instance['receipt_id'].endswith(f'e{receipt:06}'))
            self.assertEqual(instance['relation_type'],'observed_action_skeleton_not_a_stable_response_or_slip_transition')
            self.assertNotIn('holding_certified',instance)
            source=self.payload(instance['post_observation_id'])
            self.assertEqual(source['coordinate_derivation']['kind'],'explicit_offline_derivation_not_original_policy_input')
