"""Physical extra attempts and restored scenes retain explicit logical ancestry."""
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest

from kyb.library import lineage_roots


class RestartLineage(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.db=sqlite3.connect(':memory:')
        self.db.row_factory=sqlite3.Row
        self.db.execute('CREATE TABLE evidence(run TEXT,path TEXT)')
        self.store=SimpleNamespace(db=self.db)
        self.original=self.create_run('original',None)

    def tearDown(self):
        self.db.close()
        self.temp.cleanup()

    def create_run(self,name,restart):
        path=self.root/name
        (path/'observations').mkdir(parents=True)
        (path/'manifest.json').write_text(json.dumps({'infrastructure_restart':restart}))
        self.db.execute('INSERT INTO evidence VALUES(?,?)',(name,str(path/'observations/000001.json')))
        return path

    def test_explicit_extra_attempt_and_restoration_share_only_logical_root(self):
        extra=self.create_run('extra',{'additional_physical_attempt':True,'prior_source_run':str(self.original)})
        self.create_run('restored',{'source_run':str(extra)})
        before=list(self.db.execute('SELECT * FROM evidence'))
        self.assertEqual(lineage_roots(self.store,['original','extra','restored']),{'original'})
        self.assertEqual(list(self.db.execute('SELECT * FROM evidence')),before)
        self.assertTrue(json.loads((extra/'manifest.json').read_text())['infrastructure_restart']['additional_physical_attempt'])

    def test_each_event_requires_its_own_field_without_aliases(self):
        for index,event in enumerate([
            {'additional_physical_attempt':True,'source_run':str(self.original)},
            {'prior_source_run':str(self.original)},
            {'additional_physical_attempt':False,'prior_source_run':str(self.original)},
        ]):
            with self.subTest(event=event):
                path=self.create_run('missing'+str(index),event)
                with self.assertRaises(KeyError):lineage_roots(self.store,[path.name])

    def test_required_paths_reject_empty_or_nonstring_values(self):
        for extra in (False,True):
            field='prior_source_run' if extra else 'source_run'
            for index,value in enumerate(['','   ',None,42,[],{}]):
                with self.subTest(extra=extra,value=value):
                    path=self.create_run(('extra' if extra else 'restored')+str(index),
                        {'additional_physical_attempt':extra,field:value})
                    with self.assertRaises(ValueError):lineage_roots(self.store,[path.name])
