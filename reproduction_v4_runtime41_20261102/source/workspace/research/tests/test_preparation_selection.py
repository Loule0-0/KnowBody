"""Legacy success must remain discoverable without a retrospective outline."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from kyb import episodes
from kyb.memory import MemoryStore
from kyb.trace import RunTrace
from kyb.session import MemorySession
from kyb.preparation import episode_for_decision


class EpisodeSelectionTests(unittest.TestCase):
    def test_success_without_outline_precedes_later_imported_failure_with_snapshot_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);store=MemoryStore(root/'memory.sqlite')
            histories=[]
            for name,label in [('known_success',True),('imported_failure',False),('empty_interruption',None)]:
                trace=RunTrace(root/name,{})
                session=MemorySession(trace,store,task='build_tower',dependencies={})
                episodes.begin(session,{},label)
                episodes.attach_result(session)
                histories.append(session)
            records=episodes.read(store,task='build_tower',process_only=True,limit=3)
            self.assertEqual([r['id'] for r in records],['known_success','imported_failure'])
            self.assertTrue(records[0]['official_success']);self.assertEqual(records[0]['stages'],[])
            self.assertFalse(records[0]['reviewed'])
            self.assertEqual(episodes.read(store,ids=['empty_interruption'],process_only=True)[0]['id'],'empty_interruption')
            # A future label may not affect ranking or the displayed outcome.
            first=store.db.execute('SELECT * FROM episode_records WHERE run=?',('known_success',)).fetchone()
            before_label=int(first['chain_ref'].rsplit(':e',1)[1])
            old=episodes.read(store,task='build_tower',frontier={'known_success':before_label},limit=1)
            self.assertEqual(old[0]['id'],'known_success');self.assertIsNone(old[0]['official_success'])
            store.close()

    def test_default_projection_keeps_physical_conditions_and_no_raw_read_authority(self):
        episode={'id':'old_run','reviewed':False,'official_success':True,'different_public_contract':None,
            'contract_amendments':['old:clarification'],'qualification':'Unreviewed interpretation only',
            'stages':[{'summary':'Head grasp permitted the observed turn, with clearance checked',
                       'first_receipt':'old:r1','last_receipt':'old:r2',
                       'conditions':{'grasp':'flat head only'},'unknowns':['Other grips untested']}],
            'trace_entry':{'first_receipt':'old:r1','result':'old:outcome'},
            'restart':{'independent_trial':False,'mode':'restored','source_run':'long/path'},
            'role_costs':{'manager':{'cli_invocations':50}}}
        saved=copy.deepcopy(episode);projected=episode_for_decision(episode)
        self.assertEqual(episode,saved)
        for key in ('summary','conditions','unknowns'):
            self.assertEqual(projected['stages'][0][key],episode['stages'][0][key])
        self.assertEqual(projected['read'],{'level':'episode','ids':['old_run']})
        self.assertNotIn('trace_entry',projected);self.assertNotIn('first_receipt',projected['stages'][0])
        self.assertEqual(projected['restart'],{'independent_trial':False,'mode':'restored'})
        for key in ('reviewed','official_success','different_public_contract','contract_amendments','qualification'):
            self.assertEqual(projected[key],episode[key])
