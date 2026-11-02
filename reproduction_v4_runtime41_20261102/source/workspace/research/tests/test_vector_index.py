"""Derived-cache identity, visibility and incremental correctness."""
from pathlib import Path
import json
import hashlib
import sqlite3
import tempfile
import unittest

from kyb.embeddings import EmbeddingError, VectorIndex


class Encoder:
    signature = 'test-encoder-v1'
    config = {'dimension': 2, 'query_instruction': 'Retrieve'}

    def __init__(self):
        self.calls = []

    def encode(self, items):
        self.calls.append(items)
        return [[1., 0.] if 'rotate' in item['text'] else [0., 1.] for item in items]

    def close(self):
        pass


class VectorIndexTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)/'derived.sqlite'
        self.encoder = Encoder()
        self.index = VectorIndex(self.path, line_id='single', profile='rgb_only', encoder=self.encoder)

    def tearDown(self):
        self.index.close()
        self.tmp.cleanup()

    def rank(self, docs):
        return self.index.rank('rotate a key', docs, schema='knowledge-v1')

    def test_cached_future_item_is_not_a_candidate_without_authority(self):
        self.rank({('future', 10): 'rotate', ('visible', 2): 'release'})
        ranking, _ = self.rank({('visible', 2): 'release'})
        self.assertEqual(ranking, [('visible', 2)])

    def test_second_query_reuses_document_vectors(self):
        self.rank({('a', 1): 'rotate'})
        _, stats = self.rank({('a', 1): 'rotate'})
        self.assertEqual(stats['encoded_texts'], 0)
        self.assertEqual(len(self.encoder.calls), 3)  # doc+query, then query only

    def test_line_and_profile_must_match_on_reopen(self):
        for line, profile in [('enhanced', 'rgb_only'), ('single', 'rgbd_calibrated')]:
            with self.assertRaisesRegex(EmbeddingError, 'identity differs'):
                VectorIndex(self.path, line_id=line, profile=profile, encoder=Encoder())

    def test_same_version_cannot_silently_change_projection(self):
        self.rank({('a', 1): 'rotate'})
        with self.assertRaisesRegex(EmbeddingError, 'Immutable document'):
            self.rank({('a', 1): 'release'})

    def test_new_revision_encodes_only_new_text(self):
        self.rank({('a', 1): 'release'})
        result, stats = self.rank({('a', 2): 'rotate'})
        self.assertEqual(result, [('a', 2)])
        self.assertEqual(stats['encoded_texts'], 1)

    def test_incremental_and_rebuilt_rankings_agree(self):
        self.rank({('a', 1): 'release'})
        docs = {('a', 1): 'release', ('b', 2): 'rotate'}
        incremental, _ = self.rank(docs)
        other = VectorIndex(Path(self.tmp.name)/'cold.sqlite', line_id='single',
                            profile='rgb_only', encoder=Encoder())
        try:
            cold, _ = other.rank('rotate a key', docs, schema='knowledge-v1')
            self.assertEqual(incremental, cold)
            self.assertEqual(list(self.index.db.execute('SELECT digest,vector FROM retrieval_vectors ORDER BY digest')),
                             list(other.db.execute('SELECT digest,vector FROM retrieval_vectors ORDER BY digest')))
        finally:
            other.close()

    def test_empty_authorized_set_never_starts_encoder(self):
        ranking, _ = self.rank({})
        self.assertEqual(ranking, [])
        self.assertEqual(self.encoder.calls, [])

    def test_wrong_existing_sqlite_file_is_never_modified_as_a_cache(self):
        path=Path(self.tmp.name)/'authority.sqlite'
        db=sqlite3.connect(path);db.execute('CREATE TABLE important(value TEXT)')
        db.execute("INSERT INTO important VALUES('preserve')");db.commit();db.close()
        before=hashlib.sha256(path.read_bytes()).hexdigest()
        with self.assertRaisesRegex(EmbeddingError,'header missing'):
            VectorIndex(path,line_id='single',profile='rgb_only',encoder=Encoder())
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),before)


if __name__ == '__main__':
    unittest.main()
