"""Space native model process launches within this user's harness pool.

Only harness clients using the same CODEX_HOME participate. This gate neither
retries nor counts HTTP requests inside Codex and is released at process launch,
not after generation. Waiting happens before the caller reserves a model call.
"""
from __future__ import annotations

import fcntl
import json
import time
from pathlib import Path


MIN_START_INTERVAL_SECONDS = 30.0


class NativeDispatch:
    def __init__(self, codex_home: Path, *, pause_path: Path | None = None):
        self.root = Path(codex_home) / 'kyb_native_dispatch'
        self.handle = None
        self.wait_seconds = 0.0
        self.interval_seconds = MIN_START_INTERVAL_SECONDS
        self.pause_path = pause_path

    def _paused(self):
        return self.pause_path is not None and self.pause_path.exists()

    def __enter__(self):
        before = time.monotonic()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            while True:
                while self._paused():
                    time.sleep(.1)
                self.handle = (self.root / 'launch.lock').open('a+')
                while not self._paused():
                    try:
                        fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        time.sleep(.1)
                if self._paused():
                    self.release()
                    continue
                self.boot_id = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
                state = self.root / 'last_launch.json'
                previous = json.loads(state.read_text()) if state.exists() else None
                if previous is not None and previous['state'] == 'pending':
                    raise RuntimeError(f'Unconfirmed native launch; inspect {state} and its original attempt before admitting another request')
                if previous is not None and previous['state'] != 'started':
                    raise ValueError(f'Invalid native launch state in {state}; no request admitted')
                deadline = (previous['monotonic_started'] + self.interval_seconds
                            if previous is not None and previous['boot_id'] == self.boot_id else 0)
                while not self._paused():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self.wait_seconds = time.monotonic() - before
                        return self
                    time.sleep(min(.1, remaining))
                self.release()
        except BaseException:
            self.release()
            raise

    def _save(self, state):
        if self.handle is None:
            raise RuntimeError('Native launch gate is not held')
        temporary = self.root / 'last_launch.tmp'
        temporary.write_text(json.dumps(state))
        temporary.replace(self.root / 'last_launch.json')

    def prepare_launch(self, attempt):
        """Fail closed if the host dies across the actual spawn boundary."""
        self._save({'state': 'pending', 'attempt': attempt, 'boot_id': self.boot_id,
                    'wall_prepared': time.time()})

    def process_started(self):
        """Commit immediately after successful Popen, then let peers wait/launch."""
        self._save({'state': 'started', 'boot_id': self.boot_id,
                    'monotonic_started': time.monotonic(), 'wall_started': time.time()})
        self.release()

    def spawn_rejected(self):
        """Popen raised OSError: no native model process accepted this launch."""
        self._save({'state': 'started', 'boot_id': self.boot_id,
                    'monotonic_started': time.monotonic(), 'wall_started': time.time(),
                    'qualification': 'Spawn rejected; conservative cooldown, no model request'})
        self.release()

    def release(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None

    def __exit__(self, *_):
        self.release()
