"""Episode-owned local encoder and content-addressed derived vector cache."""
from __future__ import annotations

import array
import hashlib
import json
import math
import os
import select
import sqlite3
import subprocess
import time
from pathlib import Path


class EmbeddingError(ValueError):
    pass


class LocalEncoder:
    def __init__(self):
        self.path = Path(__file__).with_name('retrieval_encoder.json')
        self.config = json.loads(self.path.read_text())
        self.signature = hashlib.sha256(self.path.read_bytes()+
            self.path.with_name('embedding_worker.py').read_bytes()).hexdigest()
        self.process = None
        self.buffer = b''

    def _response(self, timeout=1200):
        deadline = time.monotonic() + timeout
        while b'\n' not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                self.close()
                raise EmbeddingError('Local embedding response timed out; index not ready')
            part = os.read(self.process.stdout.fileno(), 65536)
            if not part:
                self.close()
                raise EmbeddingError('Local embedding worker exited; index not ready')
            self.buffer += part
        line, self.buffer = self.buffer.split(b'\n', 1)
        try:
            return json.loads(line)
        except (ValueError, UnicodeDecodeError) as error:
            self.close()
            raise EmbeddingError('Invalid local embedding protocol') from error

    def encode(self, items):
        if self.process is None:
            try:
                self.process = subprocess.Popen([self.config['python'],
                    str(Path(__file__).with_name('embedding_worker.py'))],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    env={**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                         'TOKENIZERS_PARALLELISM': 'false', 'PYTHONNOUSERSITE': '1'})
            except OSError as error:
                raise EmbeddingError('Pinned local encoder could not start; index not ready') from error
            hello = self._response()
            if hello.get('status')!='READY':
                self.close()
                raise EmbeddingError(hello.get('reason','Local encoder startup did not pass'))
            if (hello.get('status'), hello.get('signature'), hello.get('dimension')) != (
                    'READY', self.signature, self.config['dimension']):
                self.close()
                raise EmbeddingError('Local encoder identity mismatch')
        try:
            self.process.stdin.write((json.dumps({'items': items}, ensure_ascii=False) + '\n').encode())
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            self.close()
            raise EmbeddingError('Local embedding worker unavailable') from error
        response = self._response()
        if response.get('status') != 'PASS' or response.get('signature') != self.signature:
            raise EmbeddingError(response.get('reason', 'Local encoder did not return PASS'))
        vectors = response.get('vectors', [])
        if len(vectors) != len(items):
            raise EmbeddingError('Embedding count mismatch')
        for vector in vectors:
            if (len(vector) != self.config['dimension'] or
                    any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in vector) or
                    abs(sum(x*x for x in vector) - 1) > .001):
                raise EmbeddingError('Invalid normalized embedding')
        return vectors

    def close(self):
        if self.process is not None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            self.process.stdout.close()
            self.process = None
            self.buffer = b''


class VectorIndex:
    """The caller supplies only its authorized visible corpus, on every request."""
    def __init__(self, cache_path, *, line_id, profile, encoder=None):
        if not isinstance(line_id,str) or not line_id or profile not in ('rgb_only','rgbd_calibrated'):
            raise EmbeddingError('An explicit experiment line and supported perception profile are required')
        self.profile = profile
        self.encoder = encoder or LocalEncoder()
        path = Path(cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        expected = {'line_id': line_id, 'profile': profile, 'encoder': self.encoder.signature}
        if path.exists():
            prior=sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True)
            try:
                tables={r[0] for r in prior.execute('SELECT name FROM sqlite_master WHERE type=\'table\'')}
                actual=dict(prior.execute('SELECT key,value FROM identity')) if 'identity' in tables else None
            finally:prior.close()
            if actual!=expected:
                raise EmbeddingError('Derived vector cache line/profile/encoder identity differs or header missing')
        self.db = sqlite3.connect(path)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('CREATE TABLE IF NOT EXISTS identity(key TEXT PRIMARY KEY,value TEXT NOT NULL)')
        actual = dict(self.db.execute('SELECT key,value FROM identity'))
        if actual and actual != expected:
            self.db.close()
            raise EmbeddingError('Derived vector cache line/profile/encoder identity differs')
        with self.db:
            self.db.executemany('INSERT OR IGNORE INTO identity VALUES(?,?)', expected.items())
        self.db.execute('CREATE TABLE IF NOT EXISTS retrieval_vectors('
            'encoder TEXT NOT NULL,profile TEXT NOT NULL,schema TEXT NOT NULL,'
            'digest TEXT NOT NULL,vector BLOB NOT NULL,PRIMARY KEY(encoder,profile,schema,digest))')
        self.db.execute('CREATE TABLE IF NOT EXISTS document_versions('
            'version_key TEXT NOT NULL,schema TEXT NOT NULL,digest TEXT NOT NULL,'
            'PRIMARY KEY(version_key,schema))')

    def prepare(self, documents, *, schema):
        # Documents contain exact authorized version keys and their canonical text.
        # Cached vectors never determine visibility, applicability or read permission.
        namespace = (self.encoder.signature, self.profile, schema)
        if not documents:
            return {}, {}, {'encoder': self.encoder.signature, 'documents': 0,
                            'encoded_texts': 0, 'cached_texts': 0}
        digests = {key: hashlib.sha256(text.encode()).hexdigest() for key, text in documents.items()}
        vectors, missing = {}, {}
        for key, digest in digests.items():
            version_key = json.dumps(key, sort_keys=True)
            previous = self.db.execute('SELECT digest FROM document_versions WHERE version_key=? AND schema=?',
                                       (version_key, schema)).fetchone()
            if previous and previous[0] != digest:
                raise EmbeddingError('Immutable document projection changed under the same schema')
            row = self.db.execute('SELECT vector FROM retrieval_vectors '
                'WHERE encoder=? AND profile=? AND schema=? AND digest=?', (*namespace, digest)).fetchone()
            if row:
                vector = array.array('f'); vector.frombytes(row[0]); vectors[digest] = vector
                if (len(vector) != self.encoder.config['dimension'] or any(not math.isfinite(x) for x in vector)
                        or abs(sum(x*x for x in vector) - 1) > .001):
                    raise EmbeddingError('Corrupt derived vector cache; rebuild before searching')
            else:
                missing[digest] = documents[key]
        items = list(missing.items())
        for start in range(0, len(items), 32):
            batch = items[start:start+32]
            encoded = self.encoder.encode([{'key': digest, 'text': text} for digest, text in batch])
            with self.db:
                for (digest, _), vector in zip(batch, encoded):
                    values = array.array('f', vector)
                    self.db.execute('INSERT INTO retrieval_vectors VALUES(?,?,?,?,?)',
                        (*namespace, digest, values.tobytes()))
                    vectors[digest] = values
        with self.db:
            self.db.executemany('INSERT OR IGNORE INTO document_versions VALUES(?,?,?)',
                [(json.dumps(key, sort_keys=True), schema, digest) for key,digest in digests.items()])
        return vectors, digests, {'encoder': self.encoder.signature, 'documents': len(documents),
            'encoded_texts': len(missing), 'cached_texts': len(set(digests.values()))-len(missing)}

    def rank(self, question, documents, *, schema, limit=20):
        vectors, digests, stats = self.prepare(documents, schema=schema)
        if not documents:return [],stats
        query = 'Instruct: ' + self.encoder.config['query_instruction'] + '\nQuery: ' + question
        qvector = self.encoder.encode([{'key': 'query', 'text': query}])[0]
        scores = {key: sum(x*y for x,y in zip(vectors[digest],qvector)) for key,digest in digests.items()}
        ranking = sorted(scores, key=lambda key: (-scores[key], str(key)))[:limit]
        return ranking,stats

    def close(self):
        self.encoder.close()
        self.db.close()
