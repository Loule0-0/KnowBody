"""Real executed history: continuity, source cutoff and snapshot isolation."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from kyb.memory import MemoryStore, encoded
from kyb.robodojo_adapter import canonical_state

SOURCE=Path('/data/KnowYourBody_RoboDojo/runs/20261018T124930_219848964')


class SegmentTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.store=MemoryStore(Path(self.temp.name)/'memory.sqlite')
        for n in range(337,479):
            p=SOURCE/'events'/f'e{n:06d}.json'
            d=json.loads(p.read_text())
            if d['kind'] not in ('observation','receipt','attachment_change','attachment_observation'):
                continue
            ref={key:d[key] for key in ('id','sequence','kind','visibility')}
            ref.update(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            self.store.register(ref,d['payload'])
            if d['kind']=='observation':
                payload=d['payload']; canonical=canonical_state(payload['state'],payload.get('measured_finger_joints_m',{}),payload['dependencies']['body'])
                with self.store.db:
                    self.store.db.execute('INSERT INTO observation_coordinates VALUES(?,?,?)',(d['id'],encoded(canonical),encoded({'kind':'explicit_test_derivation'})))
        self.current=self.store.evidence_payload(SOURCE.name+':e000384')
        self.deps={**self.current['dependencies'],'planner':'new_profile'}

    def tearDown(self):
        self.store.close();self.temp.cleanup()

    def read(self,query='key bow lift',cutoff=478,current=None):
        return self.store.read_segments(query,context={'task':'insert_key'},dependencies=self.deps,
            current_observation=current or self.current,episode='new_run',
            evidence_cutoff={SOURCE.name:cutoff})

    def test_connected_lift_history_is_factual_and_not_a_macro(self):
        result=self.read();self.assertTrue(result['episodic_segments'])
        s=result['episodic_segments'][0]
        self.assertEqual([a['receipt_id'].rsplit(':',1)[-1] for a in s['actions']],['e000414','e000475'])
        measured=[]
        for a in s['actions']:
            before=s['states'][a['before_observation_id']]['grasp_point_dimensions']['left_z']
            after=s['states'][a['post_observation_id']]['grasp_point_dimensions']['left_z']
            measured.append(after-before)
        self.assertAlmostEqual(measured[0],.096414,places=5)
        self.assertAlmostEqual(measured[1],.150482,places=5)
        self.assertIn('not a causal claim or executable macro',s['application'])
        for a,b in zip(s['actions'][:-1],s['actions'][1:]):
            self.assertEqual(a['post_observation_id'],b['before_observation_id'])
        self.assertTrue(all(a['current_planner_comparison']=='different_or_unknown_replan_required' for a in s['actions']))
        self.assertTrue(all(self.store.db.execute('SELECT kind FROM evidence WHERE id=?',(i,)).fetchone()[0]
            !='execution_frame' for i in s['evidence']))
        self.assertEqual(self.store.revision,0)

    def test_future_retention_interpretation_does_not_leak_into_earlier_read(self):
        earlier=self.read(cutoff=476)
        text=json.dumps(earlier)
        self.assertNotIn(':e000478',text)
        last=earlier['episodic_segments'][0]['actions'][-1]
        self.assertEqual(last['receipt_id'].rsplit(':',1)[-1],'e000475')
        self.assertEqual(last['post_attachment_inference']['left']['state'],'unknown_no_dated_visual_interpretation')
        self.assertTrue(all(s['available_after'][SOURCE.name]<=476 for s in earlier['episodic_segments']))
        earlier_again=self.read(cutoff=476)
        self.assertEqual(earlier,earlier_again)
        later=self.read()
        self.assertTrue(later['episodic_segments'])

    def test_closure_is_unknown_until_dated_visual_lift_evidence(self):
        result=self.read(query='close gripper bow',cutoff=386,
            current=self.store.evidence_payload(SOURCE.name+':e000337'))
        s=result['episodic_segments'][0]
        closure=next(a for a in s['actions'] if a['receipt_id'].endswith('e000383'))
        self.assertEqual(closure['post_attachment_inference']['left']['state'],'unknown')
        self.assertNotIn('visually_supported',json.dumps(s))
        self.assertEqual(s['available_after'][SOURCE.name],386)

    def test_query_and_body_scope_affect_readout(self):
        self.assertFalse(self.read(query='nonexistent_lexical_token')['episodic_segments'])
        self.deps['body']='other_robot'
        segment=self.read()['episodic_segments'][0]
        self.assertIsNone(segment['entry_joint_distance_rad_l2'])
        self.assertIn('current scene alignment',segment['application'])

    def test_current_episode_recall_keeps_late_known_annotation_but_not_newer_block(self):
        result = self.store.read_segments('key bow lift', context={'task':'insert_key'}, dependencies=self.deps,
            current_observation=self.current, episode=SOURCE.name,
            current_history_before=476, evidence_cutoff={SOURCE.name:478})
        segment = result['episodic_segments'][0]
        self.assertTrue(all(int(a['post_observation_id'].rsplit('e',1)[1]) <= 476 for a in segment['actions']))
        self.assertIn(':e000478', json.dumps(segment))
        earlier = self.store.read_segments('key bow lift', context={'task':'insert_key'}, dependencies=self.deps,
            current_observation=self.current, episode=SOURCE.name,
            current_history_before=384, evidence_cutoff={SOURCE.name:478})
        self.assertNotIn(':e000475',json.dumps(earlier))


if __name__=='__main__':unittest.main()
