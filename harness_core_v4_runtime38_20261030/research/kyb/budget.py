"""A durable, fail-closed allowance for the CLI clients assigned to it."""
from __future__ import annotations
import fcntl
import json
import time
from pathlib import Path

class BudgetExhausted(RuntimeError):
    pass

class CallBudget:
    def __init__(self, path: Path, limit: int):
        if type(limit) is not int or limit < 1:
            raise ValueError('Positive total call budget required')
        self.path, self.limit = Path(path), limit
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self.view()

    def _rows(self, stream):
        stream.seek(0)
        rows=[json.loads(line) for line in stream if line.strip()]
        if any(row['limit'] != self.limit for row in rows):
            raise ValueError('Persisted budget limit differs; no allowance reset')
        return rows

    def view(self):
        with self.path.open('r') as stream:
            fcntl.flock(stream, fcntl.LOCK_SH)
            rows=self._rows(stream);used=len(rows)
        return {'limit':self.limit,'used':used,'remaining':max(0,self.limit-used),
                'roles_recorded':sorted({row['role'] for row in rows}),
                'scope':'all native CLI attempts assigned to this ledger; internal HTTP retries unobservable'}

    def reserve(self, attempt: str, role: str):
        with self.path.open('r+') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            rows=self._rows(stream)
            if len(rows)>=self.limit:
                raise BudgetExhausted('Assigned model call allowance exhausted; no request sent')
            if any(row['attempt']==attempt for row in rows):
                raise ValueError('Attempt already reserved')
            stream.seek(0,2)
            stream.write(json.dumps({'attempt':attempt,'role':role,'limit':self.limit,
                'number':len(rows)+1,'reserved_at':time.time()})+'\n')
            stream.flush()
            import os
            os.fsync(stream.fileno())
        return self.view()
