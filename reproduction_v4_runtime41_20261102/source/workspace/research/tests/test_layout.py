"""Exercise native SeedManager with a sparse local layout installation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from env.seed_manager.seed_manager import SeedManager
from kyb.layout import bind_layout

class LayoutTests(unittest.TestCase):
    def test_sparse_installation_preserves_full_catalog_numeric_rank(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);directory=root/'Eval_Layout/RoboDojo/arx_x5/0';directory.mkdir(parents=True)
            selected_file=directory/'task_2.json';selected_file.write_text('{"sentinel":"selected frozen file"}')
            with patch('env.seed_manager.seed_manager.ASSETS_PATH',str(root)):
                manager=SeedManager({'num_envs':1,'task_name':'task','config_name':'arx_x5','seed':0});manager.init_eval()
            self.assertEqual(len(manager.seed_info),1)
            selected={'task':'task','layout':2,'candidate_layouts':[1,2,10],'path':str(selected_file.relative_to(root))}
            inv=[{'type':'file','path':str(directory.relative_to(root)/f'task_{n}.json')} for n in (10,1,2)]
            inv.extend([{'type':'file','path':'Eval_Layout/RoboDojo/arx_x5/1/task_2.json'},
                {'type':'file','path':str(directory.relative_to(root)/'task_random_2.json')}])
            index=bind_layout(manager,selected,inv,root)
            self.assertEqual(index,1)
            self.assertEqual(manager.get_seed_scene_info(index),{'sentinel':'selected frozen file'})
            self.assertFalse((directory/'task_1.json').exists())
    def test_frozen_catalog_or_selected_path_cannot_be_changed(self):
        class Manager:pass
        s={'task':'x','layout':1,'candidate_layouts':[0,1],'path':'p/x_1.json'}
        with self.assertRaises(ValueError):bind_layout(Manager(),s,[{'type':'file','path':'p/x_1.json'}],Path('/tmp'))
