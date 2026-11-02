"""Sealed custody reuse, bounded hashing and unchanged fresh/final validation."""
import copy
import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import main_experiment as main
import continual_experiment as single
from kyb import library
from kyb.memory import MemoryStore
from kyb.snapshots import MAIN_COUNTS, SINGLE_COUNTS, sealed_counts, sha256_file


class Snapshots(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.serial = 0

    def tearDown(self):
        self.temp.cleanup()

    def fixture(self, single_line=False):
        self.serial += 1
        root = self.root / str(self.serial)
        root.mkdir()
        store = MemoryStore(root / 'work.sqlite')
        library.initialize(store)
        # Two revisions of one ID ensure IDs are never confused with revisions.
        with store.db:
            for text in ('first', 'revised'):
                store.db.execute("INSERT INTO claims(id,kind,module,text,conditions,dependencies,epistemic,status) "
                    "VALUES('one','external','retention_release',?,'{}','{}','hypothesis','provisional')", (text,))
        snapshot = root / 'final_memory.sqlite'
        manifest = store.freeze(snapshot)
        store.close()
        fields, count = (SINGLE_COUNTS, single.memory_counts) if single_line else (MAIN_COUNTS, main.memory_counts)
        counts = count(snapshot)  # Actual full check precedes recording=PASS.
        artifact = root / 'artifact'
        artifact.mkdir()
        result = {'order': 1, 'recording': 'PASS', 'final_memory': str(snapshot),
                  'final_memory_sha256': manifest['sha256'], 'artifact': str(artifact), 'memory_counts': counts}
        self.save(artifact / 'RESULT.json', result)
        state = {'status': 'READY', 'active': None, 'next_order': 2,
                 'memory': str(snapshot), 'completed': [copy.deepcopy(result)]}
        return state, fields, snapshot

    def save(self, path, value):
        path.write_text(json.dumps(value))

    def update_result(self, state):
        self.save(Path(state['completed'][-1]['artifact']) / 'RESULT.json', state['completed'][-1])

    def test_authenticated_main_and_single_counts_need_no_database_recheck(self):
        for line in (False, True):
            state, fields, _ = self.fixture(line)
            with patch.object(sqlite3, 'connect', side_effect=AssertionError('Unexpected database recheck')):
                counts = sealed_counts(state, fields)
            self.assertEqual(counts, state['completed'][-1]['memory_counts'])
            self.assertEqual((counts['case_revision'], counts['cases']), (2, 1))
            self.assertEqual('knowledge_active' in counts, line)
            counts['cases'] = 0
            self.assertEqual(state['completed'][-1]['memory_counts']['cases'], 1)

    def test_state_result_path_status_and_journal_mismatch_stop(self):
        for change in ('path', 'active', 'status', 'order', 'recording', 'result', 'wal', 'journal'):
            with self.subTest(change=change):
                state, fields, snapshot = self.fixture()
                if change == 'path':
                    state['memory'] = str(snapshot.parent / 'other.sqlite')
                elif change == 'active':
                    state['active'] = {'order': 2}
                elif change == 'status':
                    state['status'] = 'RUNNING'
                elif change == 'order':
                    state['next_order'] = 3
                elif change == 'recording':
                    state['completed'][-1]['recording'] = 'NEEDS_WORK'
                elif change == 'result':
                    path = Path(state['completed'][-1]['artifact']) / 'RESULT.json'
                    result = json.loads(path.read_text())
                    result['memory_counts']['cases'] = 0
                    self.save(path, result)
                else:
                    Path(str(snapshot) + ('-wal' if change == 'wal' else '-journal')).write_bytes(b'unsealed changes')
                with self.assertRaises(ValueError):
                    sealed_counts(state, fields)

    def test_readonly_wal_index_is_not_pending_database_content(self):
        state, fields, snapshot = self.fixture()
        reader = sqlite3.connect('file:' + str(snapshot) + '?mode=ro', uri=True)
        try:
            reader.execute('SELECT COUNT(*) FROM claims').fetchone()
            wal = Path(str(snapshot) + '-wal')
            shm = Path(str(snapshot) + '-shm')
            self.assertTrue(wal.exists())
            self.assertEqual(wal.stat().st_size, 0)
            self.assertTrue(shm.exists())
            self.assertEqual(sealed_counts(state, fields), state['completed'][-1]['memory_counts'])
        finally:
            reader.close()

    def test_bytes_manifest_and_count_schema_mismatch_stop_without_fallback(self):
        for change in ('bytes', 'sha', 'scope', 'revision', 'fields', 'negative', 'boolean', 'count', 'line_schema'):
            with self.subTest(change=change):
                state, fields, snapshot = self.fixture(True)
                manifest_path = snapshot.with_suffix('.manifest.json')
                manifest = json.loads(manifest_path.read_text())
                counts = state['completed'][-1]['memory_counts']
                if change == 'bytes':
                    with snapshot.open('ab') as out:
                        out.write(b'changed')
                elif change in ('sha', 'scope', 'revision'):
                    manifest[{'sha': 'sha256', 'scope': 'scope', 'revision': 'revision'}[change]] = 'changed'
                    self.save(manifest_path, manifest)
                elif change == 'fields':
                    del counts['knowledge_active']
                elif change == 'negative':
                    counts['evidence'] = -1
                elif change == 'boolean':
                    counts['evidence'] = True
                elif change == 'count':
                    counts['cases'] = 3
                else:
                    fields = MAIN_COUNTS
                self.update_result(state)
                with self.assertRaises(ValueError):
                    sealed_counts(state, fields)

    def test_streamed_hash_and_snapshot_fork_preserve_custody(self):
        state, _, snapshot = self.fixture()
        expected = state['completed'][-1]['final_memory_sha256']
        with patch.object(Path, 'read_bytes', side_effect=AssertionError('Whole-file allocation')):
            self.assertEqual(sha256_file(snapshot), expected)
            branch, manifest = MemoryStore.fork_snapshot(snapshot, self.root / 'fork.sqlite')
            try:
                frozen = branch.freeze(self.root / 'second.sqlite')
            finally:
                branch.close()
        self.assertEqual(manifest['sha256'], expected)
        self.assertEqual(frozen['sha256'], hashlib.sha256((self.root / 'second.sqlite').read_bytes()).hexdigest())

    def test_fresh_or_corrupt_new_final_is_never_accepted_by_a_count_proof(self):
        state, fields, snapshot = self.fixture()
        state['completed'] = []
        with self.assertRaises(ValueError):
            sealed_counts(state, fields)
        for coordinator in (main, single):
            bad = snapshot.parent / (coordinator.__name__ + '.sqlite')
            bad.write_bytes(b'not a database')
            with self.assertRaises(sqlite3.DatabaseError):
                coordinator.memory_counts(bad)


if __name__ == '__main__':
    unittest.main()
