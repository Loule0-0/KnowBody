"""Immutable local evidence and policy trace files."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any


def plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    return value


def write_once(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(plain(value), ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode()
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(data).hexdigest()


def append(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.write(json.dumps(plain(value), ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class RunTrace:
    def __init__(self, root: Path, manifest: dict[str, Any]):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False, mode=0o700)
        self.sequence = 0
        write_once(self.root / "manifest.json", manifest)

    def event(self, kind: str, payload: dict[str, Any], *, visibility: str = "policy") -> dict:
        self.sequence += 1
        event_id = f"{self.root.name}:e{self.sequence:06d}"
        relative = Path("evaluation" if visibility == "private" else "events") / f"e{self.sequence:06d}.json"
        event = dict(id=event_id, sequence=self.sequence, kind=kind,
                     visibility=visibility, wall_time=time.time(), payload=plain(payload))
        sha = write_once(self.root / relative, event)
        reference = dict(id=event_id, sequence=self.sequence, kind=kind,
                         visibility=visibility, path=str(self.root / relative), sha256=sha)
        append(self.root / ("evaluation/events.jsonl" if visibility == "private" else "events.jsonl"), reference)
        return reference
