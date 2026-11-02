"""Exact measurement projections, provenance scope and the RGB observation boundary."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from kyb.context import surface_evidence_view
from kyb.perception import RGB_ONLY
import test_manager

SOURCE=Path('/data/KnowYourBody_RoboDojo/runs/stacktoy_v4_repeat50_runtime12_20261027_e14_20261026T192845/events/e000298.json')

class SurfaceEvidenceViewTests(unittest.TestCase):
    setUp=test_manager.ManagerTests.setUp
    tearDown=test_manager.ManagerTests.tearDown
    observation=test_manager.ManagerTests.observation
    attach=test_manager.ManagerTests.attach

    def measurement(self):
        data=json.loads(SOURCE.read_text())['payload']
        data['before_observation_id']=self.before['id'];data['dependencies']=dict(self.deps)
        ref=self.session.evidence('surface_inspection',data)
        manager=self.attach([test_manager.publish(0)])
        manager.run('prepare','prepare');manager.purpose='review';manager.frontier=self.session.source_frontier()
        return manager,ref,data

    def test_sensor_values_precision_and_qualifications_survive_both_projections(self):
        manager,ref,data=self.measurement();saved=copy.deepcopy(data)
        for body in ('none','ee'):
            row=manager._record(ref['id'],body=body)
            self.assertEqual(row['body_view'],body);self.assertTrue(row['omitted_fields'])
            self.assertEqual({k:v for k,v in row['data'].items() if k!='samples'},
                             {k:v for k,v in data.items() if k not in ('samples','pad_relation_conventions')})
            for got,original in zip(row['data']['samples'],data['samples']):
                self.assertEqual({k:v for k,v in got.items() if k!='body_coordinates'},
                                 {k:v for k,v in original.items() if k!='body_coordinates'})
                if body=='none':self.assertNotIn('body_coordinates',got)
                else:
                    for arm,fields in original.get('body_coordinates',{}).items():
                        self.assertEqual(got['body_coordinates'][arm],{k:v for k,v in fields.items() if k!='distal_pad_relation'})
            self.assertIn('not mean every original field',row['readout_qualification'])
        self.assertEqual(data,saved);self.assertEqual(self.store.evidence_payload(ref['id']),saved)

    def test_explicit_full_read_preserves_the_complete_original_measurement(self):
        manager,ref,data=self.measurement();row=manager._record(ref['id'],body='full')
        self.assertEqual(row['data'],data);self.assertEqual(row['omitted_fields'],[])
        self.assertIn('distal_pad_relation',row['data']['samples'][0]['body_coordinates']['left'])

    def test_failed_sensor_sample_does_not_gain_hidden_values_or_zero_geometry(self):
        original={'samples':[{'status':'NEEDS_WORK','label':'unknown_surface','world_xyz_m':None}],
                  'limitations':['No valid centre depth']}
        for body in ('none','ee','full'):
            result,omitted=surface_evidence_view(original,body)
            self.assertEqual(result,original);self.assertEqual(omitted,[])
        with self.assertRaises(ValueError):surface_evidence_view(original,'unknown')

    def test_rgb_guard_runs_before_even_the_sensor_only_projection(self):
        manager,ref,data=self.measurement();self.session.perception=RGB_ONLY
        with patch('kyb.context.surface_evidence_view',side_effect=AssertionError('Projection ran before profile check')):
            with self.assertRaisesRegex(ValueError,'unavailable in RGB-only'):
                manager._record(ref['id'],body='none')

    def test_surface_view_alone_grants_no_observation_telemetry_or_future_access(self):
        manager,ref,data=self.measurement()
        result,_=manager._evidence({'ids':[ref['id']],'body':'ee'})
        manager._admit(result)
        self.assertIn(ref['id'],manager.allowed)
        self.assertFalse(manager.seen_telemetry)
        later=self.session.evidence('surface_inspection',data)
        with self.assertRaisesRegex(ValueError,'frontier'):manager._record(later['id'],body='none')
