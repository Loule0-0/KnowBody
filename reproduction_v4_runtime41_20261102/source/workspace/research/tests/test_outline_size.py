"""Admission uses compact stages while retaining source and ordering gates."""
import json
import unittest

from kyb import episodes
import test_terminal as terminal


class OutlineSizeTests(unittest.TestCase):
    setUp=terminal.TerminalTests.setUp
    tearDown=terminal.TerminalTests.tearDown

    def begin(self):
        self.r1=self.session.evidence('receipt',{'executed_actions':1})
        self.r2=self.session.evidence('receipt',{'executed_actions':1})
        episodes.begin(self.session,{},True)

    def stage(self,**extra):
        return {'summary':'Lift, turn, then verify after release.',
            'first_receipt':self.r1['id'],'last_receipt':self.r2['id'],
            'conditions':{'grasp':'flat head'},'unknowns':['Hidden contact'],**extra}

    def test_exact_4096_compact_utf8_keeps_unicode_and_escaping(self):
        self.begin();stages=[self.stage(summary='汉字"\\\n')]
        compact=lambda:len(json.dumps(stages,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode())
        stages[0]['summary']+='x'*(4096-compact())
        self.assertEqual(compact(),4096)
        self.assertGreater(len(json.dumps(stages,ensure_ascii=False).encode()),4096)
        ref=episodes.seal(self.session,stages)
        self.assertEqual(self.store.evidence_payload(ref)['stages'],stages)

    def test_4097_rejects_without_result_or_partial_outline(self):
        self.begin();stages=[self.stage()]
        size=len(json.dumps(stages,ensure_ascii=False,separators=(',',':')).encode())
        stages[0]['summary']+='x'*(4097-size)
        with self.assertRaisesRegex(ValueError,'actual_bytes=4097/4096'):
            episodes.seal(self.session,stages)
        row=self.store.db.execute('SELECT outline_ref,result_ref FROM episode_records').fetchone()
        self.assertIsNone(row['outline_ref']);self.assertIsNone(row['result_ref'])

    def test_seven_stages_foreign_receipt_and_reverse_order_still_reject(self):
        self.begin()
        for stages in [[self.stage()]*7,[self.stage(first_receipt='foreign:e1')],
            [self.stage(first_receipt=self.r2['id']),self.stage(last_receipt=self.r1['id'])]]:
            with self.assertRaises(ValueError):episodes.seal(self.session,stages)
        self.assertIsNone(self.store.db.execute('SELECT outline_ref FROM episode_records').fetchone()[0])
