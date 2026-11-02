"""Temporary-directory deployment tests; no network, Kit, encoder or hosted calls."""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import common
import prepare_line as prep
import reconcile_boundary as boundary
import status as status_tool

sys.path.insert(0,str(common.BUNDLE/'source/workspace/research'))
import main_experiment as main42
import continual_experiment as single


class DeploymentTests(unittest.TestCase):
    def test_layouts_are_frozen_unique_and_exclude_previous_assignments(self):
        inventory=common.load(common.BUNDLE/'catalog/layout_inventory.json')
        kwargs=dict(kind='single50',episodes=50,seed=20261102,exclude={})
        rows=prep.select_layouts(inventory,**kwargs)
        self.assertEqual(rows,prep.select_layouts(inventory,**kwargs))
        self.assertEqual(len({r['layout'] for r in rows}),50)
        used=[r['layout'] for r in rows]
        more=prep.select_layouts(inventory,**{**kwargs,'episodes':5,'exclude':{'play_stacking_toy':used}})
        self.assertFalse(set(used)&{r['layout'] for r in more})
        with self.assertRaisesRegex(ValueError,'need 50 unused layouts'):
            prep.select_layouts(inventory,**{**kwargs,'exclude':{'play_stacking_toy':used}})
        for row in rows:self.assertEqual(row['candidate_layouts'][row['native_reset_seed']],row['layout'])

    def test_existing_download_is_never_replaced_on_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'artifact';p.write_bytes(b'original')
            with patch('urllib.request.urlopen',side_effect=AssertionError('Network must not run')):
                with self.assertRaisesRegex(ValueError,'Existing artifact differs'):
                    common.fetch('https://example.invalid',p,sha256='0'*64)
            self.assertEqual(p.read_bytes(),b'original')

    def prepare_and_preflight(self,kind,profile,episodes):
        with tempfile.TemporaryDirectory() as tmp,contextlib.ExitStack() as stack:
            root=Path(tmp);project=root/'project';data=root/'data';project.mkdir();data.mkdir()
            bundle=root/'bundle';(bundle/'catalog').mkdir(parents=True)
            for name in ['TASK_ORDER.json','TEMPLATE_PROTOCOL.json','layout_inventory.json']:
                shutil.copy2(common.BUNDLE/'catalog'/name,bundle/'catalog'/name)
            def put(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
            extension=data/'kit_extensions/vendor/file';extension.parent.mkdir(parents=True);extension.write_bytes(b'test extension')
            put(bundle/'catalog/KIT_EXTENSIONS.json',{'files':{'vendor/file':common.digest(extension)}})
            put(bundle/'catalog/ASSET_PATCHES.json',[])
            put(bundle/'catalog/NVIDIA_MATERIALS.json',{'files':[]})
            codex=project/'tools/codex';codex.parent.mkdir();codex.write_bytes(b'not executable; never invoked')
            put(bundle/'catalog/CODEX_BINARY.json',{'target':'tools/codex','binary_sha256':common.digest(codex)})
            base=project/'BASE_HARNESS.json';put(base,{'source_version':'v4-runtime41-20261101','original_release_sha256':'a'*64,'files':{}})
            fixture=data/'Assets/fixture.bin';fixture.parent.mkdir();fixture.write_bytes(b'public asset fixture')
            entry={'path':'Assets/fixture.bin','size':fixture.stat().st_size,'lfs':{'oid':common.digest(fixture)}}
            def assets(tasks):
                return ({t['task']:[entry] for t in tasks},{t['order']:[entry] for t in tasks})
            def layouts(row):put(data/row['path'],{'fixture_only':True})
            for key,value in {'BUNDLE':bundle,'PROJECT':project,'DATA':data,'BASE_MANIFEST':base,
                              'WORKSPACE':common.BUNDLE/'source/workspace'}.items():
                stack.enter_context(patch.object(prep,key,value))
            stack.enter_context(patch.object(prep,'base_files',lambda:common.load(base)))
            stack.enter_context(patch.object(prep,'asset_plan',assets))
            stack.enter_context(patch.object(prep,'fetch_metadata_row',layouts))
            args=argparse.Namespace(name='test-'+kind,kind=kind,profile=profile,episodes=episodes,seed=42,
                gpu=0,port=19231,primary_calls=100,manager_calls=100,exclude=None)
            with contextlib.redirect_stdout(io.StringIO()):prep.prepare(args)
            e=project/'experiments'/args.name
            protocol=common.load(e/'PROTOCOL.json')
            self.assertTrue(all(v==0 for v in protocol['initial_memory_counts'].values()))
            self.assertEqual(protocol['perception_profile'],profile)
            self.assertEqual(protocol['manager']['effort'],'xhigh')
            self.assertTrue((e/'empty_memory.manifest.json').is_file())
            if kind=='main42':
                with patch.object(main42,'R',project),patch.object(main42,'D',data),patch.object(sys,'argv',
                    ['main_experiment.py','--experiment-dir',str(e),'--round','1','--initial-memory',str(e/'empty_memory.sqlite'),'--preflight-only']),contextlib.redirect_stdout(io.StringIO()):
                    main42.main()
            else:
                with patch.object(single,'R',project),patch.object(single,'D',data),contextlib.redirect_stdout(io.StringIO()):
                    single.run(e,preflight_only=True)
            state=common.load(e/protocol['state_relative_path'])
            self.assertEqual(state['status'],'READY');self.assertEqual(state['completed'],[])
            self.assertIsNone(state['active'])
            with self.assertRaisesRegex(ValueError,'never overwrite or reroll'):prep.prepare(args)

    def test_main42_empty_rgbd_preparation_reaches_original_coordinator(self):
        self.prepare_and_preflight('main42','rgbd_calibrated',42)

    def test_single50_empty_rgb_preparation_reaches_original_coordinator(self):
        self.prepare_and_preflight('single50','rgb_only',50)

    def test_boundary_resume_refuses_active_or_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            e=Path(tmp);common.write_new(e/'PROTOCOL.json',{'study':'single50','state_relative_path':'STATE.json'})
            def state(value):(e/'STATE.json').write_text(json.dumps(value))
            state({'status':'PAUSED','active':{'order':1}})
            with self.assertRaisesRegex(ValueError,'Only PAUSED'):boundary.reconcile(e)
            common.write_new(e/'HARNESS.json',{'files':{}})
            state({'status':'PAUSED','active':None,'next_order':2,
                'completed':[{'primary':{'outcome_unknown':1},'manager':{}}]})
            with self.assertRaisesRegex(ValueError,'outcome remains unknown'):boundary.reconcile(e)
            state({'status':'PAUSED','active':None,'next_order':1,'completed':[]})
            with contextlib.redirect_stdout(io.StringIO()):boundary.reconcile(e)
            self.assertEqual(common.load(e/'STATE.json')['status'],'READY')
            self.assertFalse(common.load(e/'BOUNDARY_RESUME_001.json')['budget_changes'])

    def test_role_usage_stays_separate_and_missing_coverage_is_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            e=Path(tmp);common.write_new(e/'PROTOCOL.json',{'state_relative_path':'STATE.json'})
            row={'order':1,'task':'fixture','layout':0,'task_outcome':'NEEDS_WORK','env_steps':9,'memory_counts':{},'run':'fixture',
                'primary':{'cli_invocations':2,'captured_metadata_consistent':True,'usage_reported':{'input_tokens':100,'output_tokens':20,'cached_input_tokens':30}},
                'manager':{'cli_invocations':1,'usage_reported':{'input_tokens':10}}}
            common.write_new(e/'STATE.json',{'status':'READY','completed':[row]})
            result=status_tool.status(e)['episodes'][0]['roles']
            self.assertEqual(result['primary']['input_plus_output'],120)
            self.assertIsNone(result['manager']['input_plus_output'])


if __name__=='__main__':unittest.main(verbosity=2)
