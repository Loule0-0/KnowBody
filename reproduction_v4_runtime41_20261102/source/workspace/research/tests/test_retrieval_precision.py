"""Question relevance, default process selection and truthful contract comparison."""
from pathlib import Path
import unittest
import copy
from unittest.mock import patch
import test_promotion as p
from kyb import episodes, library
from kyb.session import MemorySession
from kyb.trace import RunTrace


class RetrievalPrecision(unittest.TestCase):
    setUp = p.PromotionTests.setUp
    tearDown = p.PromotionTests.tearDown
    observation = p.PromotionTests.observation
    add_body_case = p.PromotionTests.add_body_case
    op = p.PromotionTests.op
    canonical_ref = p.PromotionTests.canonical_ref
    write = p.PromotionTests.write
    attach = p.PromotionTests.attach
    manager = p.PromotionTests.manager

    def knowledge_pair(self):
        old = self.add_body_case(text='Mallet holder mounting is a conditional geometry observation.')
        new_obs = self.observation(dependencies={**self.deps, 'body': 'b2'})
        unrelated = self.add_body_case(new_obs, text='Egg holder mounting is a different geometry observation.')
        target = self.write(self.op([self.canonical_ref(old)], question='Mallet holder mounting?',
            text='Mallet holder geometry under the observed mounting.', use='Inspect the mallet before applying this relation.'), [old])
        other = self.write(self.op([self.canonical_ref(unrelated)], question='Egg holder mounting?',
            text='Egg holder geometry under its observed mounting.', use='Inspect the egg holder.'), [unrelated])
        return target, other

    def test_search_subject_precedes_unchanged_unrelated_pin_without_certifying_transfer(self):
        target, other = self.knowledge_pair()
        cards = library.read_knowledge(self.store, 'mallet holder', context={}, dependencies={**self.deps, 'body': 'b2'})
        self.assertEqual(cards[0]['id'], target['id'])
        self.assertIn('body', cards[0]['applicability']['changed_dependencies'])
        self.assertFalse(cards[0]['default_eligible'])
        browse = library.read_knowledge(self.store, '', context={}, dependencies={**self.deps, 'body': 'b2'})
        self.assertEqual(browse[0]['id'], other['id'])

    def test_partial_lexical_body_recall_and_exact_order_survive(self):
        target, other = self.knowledge_pair()
        cards = library.read_knowledge(self.store, 'xylophone mallet', context={}, dependencies=self.deps)
        self.assertEqual(cards[0]['id'], target['id'])
        exact = library.read_knowledge(self.store, 'unmatched question', ids=[other['id'], target['id']],
            context={}, dependencies=self.deps)
        self.assertEqual([c['id'] for c in exact], [other['id'], target['id']])

    def close(self, session):
        role = {'cli_invocations': 0, 'usage_reported': {}, 'outcome_unknown': 0}
        episodes.close(session, {}, role, role, {'used': 0}, {'used': 0}, termination='fixture_interruption')

    def test_default_omits_unknown_empty_alias_but_exact_read_keeps_it(self):
        self.close(self.session)  # This fixture has one real receipt.
        trace = RunTrace(Path(self.temp.name) / 'empty_interrupted_alias', {})
        alias = MemorySession(trace, self.store, task='arbitrary', dependencies=self.deps)
        self.close(alias)
        candidates = episodes.read(self.store, task='arbitrary', limit=1, process_only=True)
        self.assertEqual(candidates[0]['id'], self.session.episode)
        exact = episodes.read(self.store, ids=[alias.episode], process_only=True)
        self.assertEqual(exact[0]['id'], alias.episode)
        self.assertIsNone(exact[0]['official_success'])

    def test_unknown_contract_is_not_known_different_and_known_comparisons_remain(self):
        self.close(self.session)
        unknown = episodes.read(self.store, ids=[self.session.episode], contract_id='current')[0]
        self.assertIsNone(unknown['different_public_contract'])
        trace = RunTrace(Path(self.temp.name) / 'known_contract', {})
        known = MemorySession(trace, self.store, task='arbitrary', dependencies=self.deps)
        known.task_specification = known.evidence('task_specification', {'contract_id': 'known', 'instruction': 'Fixture'})
        self.close(known)
        self.assertFalse(episodes.read(self.store, ids=[known.episode], contract_id='known')[0]['different_public_contract'])
        self.assertTrue(episodes.read(self.store, ids=[known.episode], contract_id='other')[0]['different_public_contract'])
        self.assertIsNone(episodes.read(self.store, ids=[known.episode], contract_id=None)[0]['different_public_contract'])

    def test_statement_reference_error_identifies_required_detail_without_granting_it(self):
        case = self.add_body_case()
        manager = self.manager()
        manager.frontier = self.session.source_frontier()
        manager.revision = self.store.revision
        statement = manager._search({'level':'case','ids':[case['claim_id']],'view':'statement'})
        manager._admit(statement)
        self.assertNotIn(case['claim_id'], manager.visible_revisions)
        with self.assertRaisesRegex(ValueError, case['claim_id'] + '.*complete detail permission') as error:
            manager._resolve_ref({'type':'case','id':case['claim_id'],'relation':'related'}, {}, [], manager.visible_revisions, {})
        self.assertIn('"view": "detail"', str(error.exception))
        detail = manager._search({'level':'case','ids':[case['claim_id']],'view':'detail'})
        manager._admit(detail)
        resolved = manager._resolve_ref({'type':'case','id':case['claim_id'],'relation':'related'}, {}, [], manager.visible_revisions, {})
        self.assertEqual(resolved['revision'], case['revision'])

    def test_oversized_text_pack_reports_cause_and_preserves_all_read_permissions(self):
        manager = self.manager()
        manager.purpose = 'review'
        before = (copy.deepcopy(manager.allowed), copy.deepcopy(manager.seen_images),
            copy.deepcopy(manager.seen_telemetry), copy.deepcopy(manager.sent_contracts))
        original = manager._record
        def large_record(source, **kwargs):
            record = original(source, **kwargs)
            return {**record, 'large_selected_record': 'x' * 26000}
        with patch.object(manager, '_record', side_effect=large_record):
            with self.assertRaisesRegex(ValueError, 'text_bytes=.*rgb_images=0/6') as error:
                manager._evidence({'receipts':[self.receipt['id']], 'cameras':[]})
        self.assertIn('reducing cameras does not reduce', str(error.exception))
        self.assertEqual(before, (manager.allowed, manager.seen_images, manager.seen_telemetry, manager.sent_contracts))
