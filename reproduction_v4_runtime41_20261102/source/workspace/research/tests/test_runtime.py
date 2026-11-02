import unittest
import tempfile,hashlib
from pathlib import Path
from kyb.runtime import configuration,select
from kyb.memory import MemoryStore

class Runtime(unittest.TestCase):
    def test_same_task_selects_its_exact_layout_and_rejects_ambiguity(self):
        rows={'tasks':[{'order':1,'task':'same','layout':22},{'order':2,'task':'same','layout':3}]}
        self.assertEqual(select(rows,'same',3)['order'],2)
        with self.assertRaises(ValueError):select(rows,'same',9)
        with self.assertRaises(ValueError):select({'tasks':rows['tasks']*2},'same',3)
    def test_resource_roots_ports_and_devices_are_explicit(self):
        a=configuration({'runtime':{'root':'/data/runtime/a','server_port':19099,'gpu':0}})
        b=configuration({'runtime':{'root':'/data/runtime/b','server_port':19101,'gpu':0}})
        self.assertNotEqual(a[0],b[0]);self.assertNotEqual(a[1],b[1])
        with self.assertRaises(KeyError):configuration({})
        with self.assertRaises(ValueError):configuration({'runtime':{'root':'relative','server_port':19101,'gpu':0}})
    def test_actual_fork_initializes_indexes_after_copy_and_preserves_sealed_source(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);source=MemoryStore(root/'source_work.sqlite')
            source.db.execute('DROP INDEX decisions_revision');source.db.commit()
            sealed=root/'sealed.sqlite';source.freeze(sealed);source.close()
            before=hashlib.sha256(sealed.read_bytes()).hexdigest()
            fork,_=MemoryStore.fork_snapshot(sealed,root/'branch.sqlite')
            plan=[tuple(r) for r in fork.db.execute('EXPLAIN QUERY PLAN SELECT patch FROM decisions WHERE revision=? AND accepted=1',(1,))]
            self.assertTrue(any('decisions_revision' in r[-1] for r in plan))
            self.assertEqual(fork.revision,0);fork.close()
            self.assertEqual(hashlib.sha256(sealed.read_bytes()).hexdigest(),before)
