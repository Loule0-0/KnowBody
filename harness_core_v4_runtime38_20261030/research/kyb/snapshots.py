"""Custody checks for immutable, already sealed experiment snapshots."""
import hashlib
import json
from pathlib import Path

MAIN_COUNTS = frozenset(('case_revision', 'knowledge_revision', 'evidence', 'cases', 'knowledge'))
SINGLE_COUNTS = MAIN_COUNTS | {'knowledge_active'}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _no_pending_journal(snapshot):
    # A read-only WAL client can leave an empty WAL and its SHM index. SHM has
    # no database content. Never ignore transaction bytes in WAL or a journal.
    for suffix in ('-wal', '-journal'):
        try:
            size = Path(str(snapshot) + suffix).stat().st_size
        except FileNotFoundError:
            continue
        if size:
            raise ValueError('Sealed snapshot has nonempty journal content')


def sealed_counts(state, fields):
    """Reuse the counts certified before the predecessor's RESULT was written.

    Experiment coordinators emit recording=PASS only after checking the new
    final database's integrity and counts. The manifest alone proves no such
    check. Fresh snapshots still go through the coordinator's full check.
    """
    if state['status'] != 'READY' or state.get('active') or not state['completed']:
        raise ValueError('Snapshot count reuse requires a sealed predecessor')
    previous = state['completed'][-1]
    if previous['recording'] != 'PASS' or previous['order'] != state['next_order'] - 1:
        raise ValueError('Predecessor is not the last sealed episode')
    snapshot = Path(state['memory'])
    if not snapshot.is_absolute() or str(snapshot) != previous['final_memory']:
        raise ValueError('Predecessor snapshot path differs')
    result = json.loads((Path(previous['artifact']) / 'RESULT.json').read_text())
    if result != previous:
        raise ValueError('STATE predecessor differs from sealed RESULT')
    _no_pending_journal(snapshot)
    manifest = json.loads(snapshot.with_suffix('.manifest.json').read_text())
    if manifest['scope'] != 'complete_database_including_evidence_interpretations':
        raise ValueError('Snapshot manifest has another scope')
    if sha256_file(snapshot) != manifest['sha256'] or manifest['sha256'] != previous['final_memory_sha256']:
        raise ValueError('Sealed predecessor snapshot hash differs')
    _no_pending_journal(snapshot)
    counts = previous['memory_counts']
    if set(counts) != fields or any(type(value) is not int or value < 0 for value in counts.values()):
        raise ValueError('Sealed predecessor count schema differs')
    if (counts['case_revision'] != manifest['revision'] or counts['cases'] > counts['case_revision']
            or counts['knowledge'] > counts['knowledge_revision']
            or ('knowledge_active' in counts and counts['knowledge_active'] > counts['knowledge'])):
        raise ValueError('Sealed predecessor counts are inconsistent')
    return dict(counts)
