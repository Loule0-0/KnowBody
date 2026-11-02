"""Real small-snapshot custody through the actual main startup/preflight entry."""
import copy
from contextlib import ExitStack
import json
from pathlib import Path
import sqlite3
import sys
import unittest
from unittest.mock import patch

import main_experiment as main
from kyb import library, snapshots
from kyb.memory import MemoryStore
from kyb.snapshots import MAIN_COUNTS, initial_sealed_counts
import test_snapshots as snapshot_fixture


class StartupSnapshots(unittest.TestCase):
    setUp = snapshot_fixture.Snapshots.setUp
    tearDown = snapshot_fixture.Snapshots.tearDown
    fixture = snapshot_fixture.Snapshots.fixture
    save = snapshot_fixture.Snapshots.save
    update_result = snapshot_fixture.Snapshots.update_result

    def continuation(self, *, later=False):
        state, _, snapshot = self.fixture()
        protocol = {'entry_order': 2, 'input_memory': str(snapshot),
                    'input_memory_sha256': state['completed'][0]['final_memory_sha256'],
                    'initial_memory_counts': dict(state['completed'][0]['memory_counts'])}
        if later:
            root = self.root / ('later_' + str(self.serial))
            root.mkdir()
            store = MemoryStore(root / 'work.sqlite')
            library.initialize(store)
            with store.db:
                for identifier in ('a', 'b', 'c'):
                    store.db.execute("INSERT INTO claims(id,kind,module,text,conditions,dependencies,epistemic,status) "
                                     "VALUES(?,'body','geometry','Later geometry','{}','{}','hypothesis','provisional')",
                                     (identifier,))
            latest = root / 'final_memory.sqlite'
            manifest = store.freeze(latest)
            store.close()
            counts = main.memory_counts(latest)
            artifact = root / 'artifact'
            artifact.mkdir()
            row = {'order': 2, 'recording': 'PASS', 'final_memory': str(latest),
                   'final_memory_sha256': manifest['sha256'], 'artifact': str(artifact),
                   'memory_counts': counts}
            self.save(artifact / 'RESULT.json', row)
            state['completed'].append(row)
            state.update(next_order=3, memory=str(latest))
        return state, protocol

    def entrypoint(self, state, protocol, *, preflight=True):
        self.serial += 1
        root = self.root / ('experiment_' + str(self.serial))
        round_dir = root / 'round_01'
        round_dir.mkdir(parents=True)
        tasks = [{'order': i, 'task': 'task_' + str(i), 'layout': i} for i in range(1, 43)]
        protocol = {**protocol, 'round': 1, 'task_order': [t['task'] for t in tasks]}
        self.save(root / 'PROTOCOL.json', protocol)
        self.save(root / 'LAYOUTS.json', {'tasks': tasks})
        if state is not None:
            self.save(round_dir / 'STATE.json', state)
        if not preflight:
            (root / 'STOP').write_text('Synthetic pre-episode stop, no model or simulator.')
        args = ['main_experiment.py', '--experiment-dir', str(root), '--round', '1',
                '--initial-memory', protocol['input_memory']]
        if preflight:
            args.append('--preflight-only')
        with ExitStack() as stack:
            stack.enter_context(patch.object(sys, 'argv', args))
            stack.enter_context(patch.object(main, 'E', None))
            stack.enter_context(patch.object(main, 'ROUND', None))
            stack.enter_context(patch.object(main, 'verify_frozen', return_value={'version': 'v4'}))
            stack.enter_context(patch.object(main, 'configuration', return_value={}))
            stack.enter_context(patch.object(main, 'call_limits', return_value={'primary': 100, 'manager': 100}))
            stack.enter_context(patch.object(main, 'ensure_assets'))
            stack.enter_context(patch.object(main, 'render'))
            stack.enter_context(patch.object(main, 'episode_lease', side_effect=AssertionError('No GPU/model step allowed')))
            main.main()
        return root

    def test_certified_initial_counts_use_real_sha_without_sql_and_preserve_state(self):
        state, protocol = self.continuation()
        original = copy.deepcopy(state)
        with patch.object(sqlite3, 'connect', side_effect=AssertionError('Repeated integrity check')):
            with patch.object(snapshots, 'sha256_file', wraps=snapshots.sha256_file) as hashing:
                actual = initial_sealed_counts(state, protocol)
        self.assertEqual(actual, protocol['initial_memory_counts'])
        self.assertEqual(state, original)
        hashing.assert_called_once_with(self.root / '1/final_memory.sqlite')

    def test_later_completed_state_still_authenticates_the_fixed_entry_input(self):
        state, protocol = self.continuation(later=True)
        expected = state['completed'][0]['memory_counts']
        self.assertNotEqual(expected, state['completed'][-1]['memory_counts'])
        with patch.object(sqlite3, 'connect', side_effect=AssertionError('Repeated integrity check')):
            self.assertEqual(initial_sealed_counts(state, protocol), expected)
        self.assertEqual(state['next_order'], 3)
        self.assertEqual(state['memory'], state['completed'][-1]['final_memory'])

    def test_actual_preflight_and_normal_startup_do_not_repeat_full_validation(self):
        state, protocol = self.continuation(later=True)
        for preflight in (True, False):
            with self.subTest(preflight=preflight):
                with patch.object(main, 'memory_counts', side_effect=AssertionError('Repeated full check')):
                    with patch.object(sqlite3, 'connect', side_effect=AssertionError('Unexpected database connect')):
                        if preflight:
                            self.entrypoint(state, protocol)
                        else:
                            with self.assertRaisesRegex(RuntimeError, 'Operator stop at episode boundary'):
                                self.entrypoint(state, protocol, preflight=False)

    def test_missing_continuation_state_never_forks_or_falls_back(self):
        _, protocol = self.continuation()
        with patch.object(main, 'memory_counts', side_effect=AssertionError('No fallback')):
            with patch.object(MemoryStore, 'fork_snapshot', side_effect=AssertionError('No new episode seed')):
                with self.assertRaisesRegex(ValueError, 'actual clean READY'):
                    self.entrypoint(None, protocol)

    def test_bad_entry_identity_and_completed_sequence_hard_stop(self):
        for mode in ('entry_one', 'entry_boolean', 'entry_missing_row', 'path', 'sha',
                     'duplicate_path', 'sequence', 'next_order', 'active', 'status'):
            with self.subTest(mode=mode):
                state, protocol = self.continuation(later=True)
                if mode == 'entry_one':
                    protocol['entry_order'] = 1
                elif mode == 'entry_boolean':
                    protocol['entry_order'] = True
                elif mode == 'entry_missing_row':
                    protocol['entry_order'] = 4
                elif mode == 'path':
                    protocol['input_memory'] = str(self.root / 'missing.sqlite')
                elif mode == 'sha':
                    protocol['input_memory_sha256'] = '0' * 64
                elif mode == 'duplicate_path':
                    state['completed'][1]['final_memory'] = protocol['input_memory']
                elif mode == 'sequence':
                    state['completed'][1]['order'] = 1
                elif mode == 'next_order':
                    state['next_order'] = 2
                elif mode == 'active':
                    state['active'] = {'order': 3}
                else:
                    state['status'] = 'RUNNING'
                with patch.object(sqlite3, 'connect', side_effect=AssertionError('No database fallback')):
                    with self.assertRaises(ValueError):
                        initial_sealed_counts(state, protocol)

    def test_entry_result_manifest_revision_and_source_bytes_remain_authoritative(self):
        for mode in ('result', 'scope', 'revision', 'bytes', 'recording'):
            with self.subTest(mode=mode):
                state, protocol = self.continuation(later=True)
                entry = state['completed'][0]
                # fixture() supplies the exact path; do not accidentally test the latest.
                path = Path(entry['final_memory'])
                if mode == 'result':
                    result = copy.deepcopy(entry)
                    result['memory_counts']['cases'] = 0
                    self.save(Path(entry['artifact']) / 'RESULT.json', result)
                elif mode == 'recording':
                    entry['recording'] = 'NEEDS_WORK'
                elif mode == 'bytes':
                    with path.open('ab') as out:
                        out.write(b'changed')
                else:
                    manifest_path = path.with_suffix('.manifest.json')
                    manifest = json.loads(manifest_path.read_text())
                    manifest[mode] = 'changed'
                    self.save(manifest_path, manifest)
                with self.assertRaises(ValueError):
                    initial_sealed_counts(state, protocol)

    def test_nonempty_wal_and_journal_stop_but_readonly_empty_wal_remains_legal(self):
        for suffix in ('-wal', '-journal'):
            state, protocol = self.continuation()
            Path(protocol['input_memory'] + suffix).write_bytes(b'unsealed')
            with self.assertRaisesRegex(ValueError, 'nonempty journal'):
                initial_sealed_counts(state, protocol)
        state, protocol = self.continuation()
        reader = sqlite3.connect('file:' + protocol['input_memory'] + '?mode=ro', uri=True)
        try:
            reader.execute('SELECT COUNT(*) FROM claims').fetchone()
            self.assertEqual(Path(protocol['input_memory'] + '-wal').stat().st_size, 0)
            self.assertTrue(Path(protocol['input_memory'] + '-shm').exists())
            self.assertEqual(initial_sealed_counts(state, protocol), protocol['initial_memory_counts'])
        finally:
            reader.close()

    def test_protocol_count_schema_and_values_must_match_in_actual_entrypoint(self):
        for mode in ('missing', 'extra', 'wrong'):
            state, protocol = self.continuation()
            if mode == 'missing':
                protocol['initial_memory_counts'].pop('evidence')
            elif mode == 'extra':
                protocol['initial_memory_counts']['knowledge_active'] = 0
            else:
                protocol['initial_memory_counts']['cases'] += 1
            with patch.object(main, 'memory_counts', side_effect=AssertionError('No fallback')):
                with self.assertRaisesRegex(ValueError, 'counts differ'):
                    self.entrypoint(state, protocol)

    def test_recorded_entry_counts_keep_the_existing_strict_schema(self):
        for mode in ('missing', 'boolean', 'negative'):
            state, protocol = self.continuation()
            entry = state['completed'][0]
            counts = entry['memory_counts']
            if mode == 'missing':
                counts.pop('evidence')
            elif mode == 'boolean':
                counts['evidence'] = True
            else:
                counts['evidence'] = -1
            self.save(Path(entry['artifact']) / 'RESULT.json', entry)
            with self.assertRaisesRegex(ValueError, 'count schema'):
                initial_sealed_counts(state, protocol)

    def test_fresh_source_still_has_real_full_check_and_snapshot_fork(self):
        state, protocol = self.continuation()
        protocol.pop('entry_order')
        with patch.object(main, 'memory_counts', wraps=main.memory_counts) as counting:
            with patch.object(MemoryStore, 'fork_snapshot', wraps=MemoryStore.fork_snapshot) as forking:
                self.entrypoint(None, protocol)
        counting.assert_called_once()
        forking.assert_called_once()

    def test_corrupt_fresh_input_still_fails_full_sqlite_validation(self):
        state, protocol = self.continuation()
        protocol.pop('entry_order')
        bad = self.root / 'bad.sqlite'
        bad.write_bytes(b'not a database')
        protocol.update(input_memory=str(bad), input_memory_sha256=snapshots.sha256_file(bad))
        with patch.object(main, 'memory_counts', wraps=main.memory_counts) as counting:
            with self.assertRaises(sqlite3.DatabaseError):
                self.entrypoint(None, protocol)
        counting.assert_called_once()


if __name__ == '__main__':
    unittest.main()
