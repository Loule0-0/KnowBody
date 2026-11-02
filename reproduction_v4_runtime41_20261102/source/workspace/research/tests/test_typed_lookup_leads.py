"""Query-to-manager navigation retains type without acquiring read authority."""
import json
import unittest
from unittest.mock import patch

from kyb import leads, readout
import test_promotion as fixture


class TypedLookupLeads(unittest.TestCase):
    setUp = fixture.PromotionTests.setUp
    tearDown = fixture.PromotionTests.tearDown
    observation = fixture.PromotionTests.observation
    attach = fixture.PromotionTests.attach
    manager = fixture.PromotionTests.manager
    add_body_case = fixture.PromotionTests.add_body_case
    op = fixture.PromotionTests.op
    canonical_ref = fixture.PromotionTests.canonical_ref
    write = fixture.PromotionTests.write

    def knowledge(self):
        case = self.add_body_case(text='Knowledge source geometry ' + str(self.store.revision))
        with self.store.db:
            return self.write(self.op([self.canonical_ref(case)]), [case])

    def latest_lead(self):
        return json.loads(self.store.db.execute(
            'SELECT payload FROM learning_leads ORDER BY sequence DESC LIMIT 1').fetchone()[0])

    def check_query(self, arguments, expected_level):
        answer = json.loads(self.session.query_experience(arguments))
        self.assertNotEqual(answer.get('status'), 'NEEDS_WORK', answer)
        self.assertTrue(answer['records'], answer)
        expected = [{**{k: record[k] for k in ('id', 'revision', 'view') if k in record},
                     'level': expected_level} for record in answer['records']]
        self.assertEqual(self.latest_lead()['candidate_refs'], expected)
        pending = leads.pending(self.store, self.session.episode, self.session.source_frontier())
        self.assertLessEqual(readout.size(pending), 2800)
        self.assertEqual(pending['records'][0]['candidate_refs'], expected[:2])
        return answer

    def test_effective_types_survive_all_four_query_levels(self):
        case = self.add_body_case()
        self.check_query({'level': 'case', 'ids': [case['claim_id']]}, 'case')
        knowledge = self.knowledge()
        self.check_query({'level': 'knowledge', 'ids': [knowledge['id']]}, 'knowledge')
        self.session.task_specification = self.session.evidence('task_specification',
            {'contract_id': 'public-operator-contract', 'instruction': 'A public task'})
        amendment = self.session.evidence('task_contract_amendment', {
            'contract_id': 'public-operator-contract', 'question': 'Which direction?',
            'answer': 'Publicly clarified direction.', 'reuse_scope': 'same_public_contract',
            'request_id': 'operator-request', 'episode': self.session.episode})
        self.check_query({'level': 'contract', 'ids': [amendment['id']]}, 'contract')
        # A recorded historical process is a read backend fixture, not simulated execution.
        # Its record layer is episode_candidate; the tool's addressing level is episode.
        historical = {'id': 'previous-episode', 'revision': 10, 'layer': 'episode_candidate',
                      'task': 'A public task', 'official_success': True, 'reviewed': False,
                      'stages': [{'summary': 'A conditional retained transport.'}],
                      'qualification': 'Not current facts or authority to cite raw sources.'}
        with patch('kyb.episodes.read', return_value=[historical]):
            self.check_query({'level': 'episode', 'ids': [historical['id']]}, 'episode')

    def test_implicit_knowledge_level_comes_from_effective_answer(self):
        knowledge = self.knowledge()
        self.check_query({'ids': [knowledge['id']]}, 'knowledge')

    def test_partial_case_navigation_never_grants_manager_read_permissions(self):
        manager = self.manager()
        case = self.add_body_case(text='Conditional geometry. ' + 'x' * 1700)
        answer = self.check_query({'level': 'case', 'ids': [case['claim_id']],
                                   'budget_bytes': 1800}, 'case')
        self.assertEqual(answer['records'][0]['view'], 'statement_page')
        self.assertFalse(answer['records'][0]['complete_record'])
        pending = leads.pending(self.store, self.session.episode, self.session.source_frontier())
        manager._admit(pending)
        self.assertFalse(manager.allowed)
        self.assertFalse(manager.visible_revisions)
        self.assertFalse(manager.visible_knowledge)
        self.assertFalse(manager.selectable_knowledge)

    def test_continuation_survives_typed_lead_and_does_not_duplicate_it(self):
        cases = [self.add_body_case(text='Scoped geometry ' + str(i) + ' ' + 'x' * 1250)
                 for i in range(4)]
        first = self.check_query({'level': 'case', 'view': 'detail', 'budget_bytes': 4200,
                                 'case_refs': [{'id': c['claim_id'], 'revision': c['revision']}
                                               for c in cases]}, 'case')
        before = self.store.db.execute('SELECT COUNT(*) FROM learning_leads').fetchone()[0]
        second = json.loads(self.session.query_experience(
            {'continuation': first['selection_continuation'], 'budget_bytes': 4200}))
        self.assertNotEqual(second.get('status'), 'NEEDS_WORK', second)
        self.assertNotEqual(first['records'][0]['id'], second['records'][0]['id'])
        self.assertEqual(before, self.store.db.execute(
            'SELECT COUNT(*) FROM learning_leads').fetchone()[0])

    def test_tight_pending_budget_omits_whole_records_and_retains_type(self):
        case = self.add_body_case()
        for i in range(4):
            self.check_query({'level': 'case', 'ids': [case['claim_id']],
                              'question': 'A bounded physical question ' + str(i)}, 'case')
        pending = leads.pending(self.store, self.session.episode, self.session.source_frontier(),
                                budget_bytes=1600)
        self.assertLessEqual(readout.size(pending), 1600)
        self.assertGreater(pending['omitted_records'], 0)
        for record in pending['records']:
            self.assertTrue(record['candidate_refs'])
            self.assertEqual(record['candidate_refs'][0]['level'], 'case')


if __name__ == '__main__':
    unittest.main()
