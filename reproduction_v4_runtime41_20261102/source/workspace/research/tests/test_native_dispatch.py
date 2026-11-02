"""Real local process/lock boundaries; no network, credentials or model calls."""
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from kyb import native_dispatch
from kyb.codex_cli import CodexCliClient
from kyb.budget import CallBudget
from test_codex_cli import FAKE


def launch_worker(home, records, interval, ready):
    native_dispatch.MIN_START_INTERVAL_SECONDS = interval
    ready.wait()
    with native_dispatch.NativeDispatch(Path(home)) as dispatch:
        process = subprocess.Popen([sys.executable, '-c',
            'import os,time;from pathlib import Path;'
            'Path(__import__("sys").argv[1]).write_text(str(time.monotonic()));time.sleep(.7)',
            str(Path(records) / str(os.getpid()))])
        dispatch.process_started()
        process.wait(timeout=3)


def crash_worker(home):
    with native_dispatch.NativeDispatch(Path(home)):
        os._exit(0)


def crash_pending_worker(home):
    with native_dispatch.NativeDispatch(Path(home)) as dispatch:
        dispatch.prepare_launch('original-pending-attempt')
        process = subprocess.Popen([sys.executable, '-c', 'pass'])
        process.wait(timeout=2)
        os._exit(0)  # Host lost between actual Popen and its confirmation.


class NativeDispatchTests(unittest.TestCase):
    def setUp(self):
        interval = patch.object(native_dispatch, 'MIN_START_INTERVAL_SECONDS', 60.0)
        interval.start()
        self.addCleanup(interval.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.home = self.root / 'native'

    def client(self, role='policy'):
        executable = self.root / 'fake-codex'
        executable.write_text(FAKE)
        executable.chmod(0o700)
        budget = CallBudget(self.root / (role + '_budget.jsonl'), 3)
        with patch.dict(os.environ, {'CODEX_HOME': str(self.home)}):
            client = CodexCliClient(self.root / role, executable=str(executable),
                                    role=role, budget=budget, timeout=.25)
        return client, budget

    def test_processes_share_start_interval_without_serializing_generation(self):
        records = self.root / 'records'
        records.mkdir()
        context = multiprocessing.get_context('fork')
        ready = context.Event()
        workers = [context.Process(target=launch_worker,
                   args=(str(self.home), str(records), .18, ready)) for _ in range(2)]
        for process in workers:
            process.start()
            self.addCleanup(lambda p=process: p.kill() if p.is_alive() else None)
        ready.set()
        for process in workers:
            process.join(timeout=4)
            self.assertEqual(process.exitcode, 0)
        starts = sorted(float(p.read_text()) for p in records.iterdir())
        self.assertEqual(len(starts), 2)
        self.assertGreaterEqual(starts[1] - starts[0], .15)
        self.assertLess(starts[1] - starts[0], .65)

    def test_primary_and_manager_wait_outside_request_timeout(self):
        primary, pb = self.client()
        manager, mb = self.client('manager')
        tools = [{'function': {'name': 'finish_review', 'parameters': {}}}]
        with patch.dict(os.environ, {'CODEX_HOME': str(self.home)}), \
                patch.object(native_dispatch, 'MIN_START_INTERVAL_SECONDS', .4):
            primary.complete([{'role': 'user', 'content': 'first'}], tools)
            manager.complete([{'role': 'user', 'content': 'second'}], tools)
        self.assertEqual(pb.view()['used'], 1)
        self.assertEqual(mb.view()['used'], 1)
        rows = [json.loads(line) for line in (manager.root / 'cli_calls.jsonl').read_text().splitlines()]
        self.assertEqual(rows[-1]['state'], 'turn_complete')
        self.assertGreater(next(r['pre_dispatch_wait_seconds'] for r in rows if r['state'] == 'process_started'), .25)
        self.assertLess(rows[-1]['wall_seconds'], .25)

    def test_wait_interruption_neither_reserves_nor_marks_unknown_or_changes_sid(self):
        client, budget = self.client()
        client.session_id = 'retained-native-session'
        with native_dispatch.NativeDispatch(self.home) as dispatch:
            dispatch.process_started()
        with patch.object(native_dispatch.time, 'sleep', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                client.complete([{'role': 'user', 'content': 'queued'}],
                                [{'function': {'name': 'finish_review', 'parameters': {}}}])
        self.assertEqual(budget.view()['used'], 0)
        self.assertFalse((client.root / 'cli_calls.jsonl').exists())
        waits=[json.loads(line) for line in (client.root/'native_dispatch_waits.jsonl').read_text().splitlines()]
        self.assertEqual([r['state'] for r in waits],['waiting','aborted_before_dispatch'])
        self.assertFalse(waits[-1]['model_call_reserved'])
        self.assertFalse(client._failed)
        self.assertEqual(client.session_id, 'retained-native-session')
        self.assertEqual(client.summary()['cli_invocations'], 0)
        with patch.object(native_dispatch, 'MIN_START_INTERVAL_SECONDS', 0):
            with native_dispatch.NativeDispatch(self.home):
                pass

    def test_lock_wait_interruption_releases_only_own_handle(self):
        with native_dispatch.NativeDispatch(self.home) as owner:
            with patch.object(native_dispatch.time, 'sleep', side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    with native_dispatch.NativeDispatch(self.home):
                        self.fail('A second owner cannot enter')
            self.assertIsNotNone(owner.handle)
            owner.process_started()

    def test_crashed_owner_does_not_keep_lock(self):
        context = multiprocessing.get_context('fork')
        process = context.Process(target=crash_worker, args=(str(self.home),))
        process.start()
        process.join(timeout=2)
        self.assertEqual(process.exitcode, 0)
        with native_dispatch.NativeDispatch(self.home) as dispatch:
            self.assertLess(dispatch.wait_seconds, .2)

    def test_unconfirmed_spawn_is_not_reinterpreted_as_no_launch_after_crash(self):
        context = multiprocessing.get_context('fork')
        process = context.Process(target=crash_pending_worker, args=(str(self.home),))
        process.start()
        process.join(timeout=3)
        self.assertEqual(process.exitcode, 0)
        client, budget = self.client()
        with self.assertRaisesRegex(RuntimeError, 'Unconfirmed native launch'):
            client.complete([{'role':'user','content':'another study'}],
                            [{'function':{'name':'finish_review','parameters':{}}}])
        self.assertEqual(budget.view()['used'], 0)
        self.assertFalse((client.root / 'cli_calls.jsonl').exists())
        state=json.loads((self.home / 'kyb_native_dispatch' / 'last_launch.json').read_text())
        self.assertEqual(state['attempt'], 'original-pending-attempt')

    def test_pause_appearing_during_cooldown_releases_lock_and_waits_outside(self):
        pause=self.root / 'PAUSE'
        entered=threading.Event()
        errors=[]
        with native_dispatch.NativeDispatch(self.home) as dispatch:
            dispatch.process_started()
        def worker():
            try:
                with native_dispatch.NativeDispatch(self.home,pause_path=pause):entered.set()
            except BaseException as error:errors.append(error)
        with patch.object(native_dispatch, 'MIN_START_INTERVAL_SECONDS', .4):
            thread=threading.Thread(target=worker,daemon=True)
            thread.start()
            time.sleep(.05)
            pause.touch()
            time.sleep(.15)
            with patch.object(native_dispatch,'MIN_START_INTERVAL_SECONDS',0):
                with native_dispatch.NativeDispatch(self.home) as other:
                    self.assertLess(other.wait_seconds,.2)
            time.sleep(.25)
            self.assertFalse(entered.is_set())
            pause.unlink()
            thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertFalse(errors)
        self.assertTrue(entered.is_set())

    def test_rejected_spawn_clears_only_its_pending_without_retry(self):
        client,budget=self.client()
        with patch('kyb.codex_cli.subprocess.Popen',side_effect=OSError('synthetic spawn rejection')):
            with self.assertRaises(OSError):
                client.complete([{'role':'user','content':'dispatch'}],
                                [{'function':{'name':'finish_review','parameters':{}}}])
        self.assertEqual(budget.view()['used'],1)
        state=json.loads((self.home / 'kyb_native_dispatch' / 'last_launch.json').read_text())
        self.assertEqual(state['state'],'started')
        self.assertIn('rejected',state['qualification'])
        rows=[json.loads(line) for line in (client.root/'cli_calls.jsonl').read_text().splitlines()]
        self.assertEqual(rows[-1]['state'],'pre_dispatch_rejected')

    def test_different_home_and_previous_boot_do_not_share_old_deadline(self):
        with native_dispatch.NativeDispatch(self.home) as dispatch:
            dispatch.process_started()
        with native_dispatch.NativeDispatch(self.root / 'other') as dispatch:
            self.assertLess(dispatch.wait_seconds, .2)
        state = self.home / 'kyb_native_dispatch' / 'last_launch.json'
        state.write_text(json.dumps({'state':'started','boot_id': 'old-boot', 'monotonic_started': 1e15}))
        with native_dispatch.NativeDispatch(self.home) as dispatch:
            self.assertLess(dispatch.wait_seconds, .2)

    def test_invalid_shared_state_fails_before_model_accounting(self):
        client, budget = self.client()
        state = self.home / 'kyb_native_dispatch' / 'last_launch.json'
        state.parent.mkdir(parents=True)
        state.write_text('{broken')
        with self.assertRaises(json.JSONDecodeError):
            client.complete([{'role': 'user', 'content': 'queued'}],
                            [{'function': {'name': 'finish_review', 'parameters': {}}}])
        self.assertEqual(budget.view()['used'], 0)
        self.assertFalse((client.root / 'cli_calls.jsonl').exists())
        self.assertFalse(client._failed)


if __name__ == '__main__':
    unittest.main()
