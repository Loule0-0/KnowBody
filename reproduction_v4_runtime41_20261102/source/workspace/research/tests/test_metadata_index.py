"""Metadata reads stay covered; a new branch never rewrites its frozen source."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,json,sqlite3,tempfile,unittest
from kyb.memory import MemoryStore
from kyb import hybrid


class MetadataIndex(unittest.TestCase):
    def test_frontier_and_scope_queries_use_metadata_without_reading_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=MemoryStore(Path(tmp)/'memory.sqlite')
            try:
                with store.db:
                    for n,(run,kind) in enumerate([('old.a','observation'),('old.a','lookup_lead'),
                                                   ('新run','known_config'),('新run','lookup_lead')],1):
                        store.db.execute('INSERT INTO evidence VALUES(?,?,?,?,?,?,?)',
                            (f'{run}:e{n:06d}',run,n,kind,'record.json','source_sha',json.dumps({'opaque':'x'*5000})))
                session=SimpleNamespace(store=store,episode='current',perception=SimpleNamespace(name='rgb_only'),context={},dependencies={})
                frontier=dict(store.db.execute('SELECT run,MAX(sequence) FROM evidence GROUP BY run'))
                stamp=hybrid.scope_state(session,frontier=frontier,case_at=0,knowledge_at=0,snapshot=None)
                queries=[('SELECT run,MAX(sequence) FROM evidence GROUP BY run',()),
                    ("SELECT run,MAX(sequence) FROM evidence WHERE kind NOT IN ('lookup_lead','policy_decision') "
                     "AND sequence<=json_extract(?, '$.'||json_quote(run)) GROUP BY run",(json.dumps(frontier),))]
                for sql,params in queries:
                    plans=' '.join(row[3] for row in store.db.execute('EXPLAIN QUERY PLAN '+sql,params))
                    self.assertIn('COVERING INDEX',plans)
                original=[tuple(r) for r in store.db.execute('SELECT * FROM evidence ORDER BY id')]
                with store.db:store.db.execute('DROP INDEX evidence_run_sequence_kind')
                self.assertEqual(hybrid.scope_state(session,frontier=frontier,case_at=0,knowledge_at=0,snapshot=None),stamp)
                self.assertEqual([tuple(r) for r in store.db.execute('SELECT * FROM evidence ORDER BY id')],original)
            finally:store.close()

    def test_fork_adds_its_required_index_without_mutating_an_unindexed_frozen_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=MemoryStore(root/'work.sqlite')
            with source.db:source.db.execute('DROP INDEX evidence_run_sequence_kind')
            snapshot=root/'frozen.sqlite';metadata=source.freeze(snapshot);source.close()
            before=hashlib.sha256(snapshot.read_bytes()).hexdigest()
            branch,returned=MemoryStore.fork_snapshot(snapshot,root/'branch.sqlite')
            try:
                self.assertEqual(returned,metadata)
                self.assertIn('evidence_run_sequence_kind',{r[1] for r in branch.db.execute('PRAGMA index_list(evidence)')})
                original=sqlite3.connect('file:'+str(snapshot)+'?mode=ro',uri=True)
                try:self.assertNotIn('evidence_run_sequence_kind',{r[1] for r in original.execute('PRAGMA index_list(evidence)')})
                finally:original.close()
                self.assertEqual(hashlib.sha256(snapshot.read_bytes()).hexdigest(),before)
                self.assertEqual(branch.revision,metadata['revision'])
            finally:branch.close()
