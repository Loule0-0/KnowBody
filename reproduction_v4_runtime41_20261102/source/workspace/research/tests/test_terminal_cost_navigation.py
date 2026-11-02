"""Whole-episode learning sees a costly interval without expanding online context."""
import copy
import unittest

from kyb.episode_index import readout, _size


def index():
    windows=[]
    for i,cost in enumerate((0,20,4,159,30,10)):
        windows.append({'id':f'w{i}', 'events':[i*100+1,(i+1)*100], 'calls':8,
                        'tools':{'move_eef':8},'motion_without_positive_receipt':0,
                        'executed_actions':cost,'read_trajectory_after':i*100,
                        'measurement_sequences':[]})
    return {'run':'episode','through_sequence':600,'accounting':{'executed_actions':223},
            'windows':windows,'memory_questions':[{'event':'episode:e000010','sequence':10,
                'question':'A completed earlier comparison','tool':'manage_memory','purpose':'review'}],
            'largest_holds':[],'largest_full_plan_residuals':[],
            'qualification':'Navigation only, not physical evidence.'}


class TerminalCostNavigation(unittest.TestCase):
    def test_completed_prioritizes_actual_cost_without_changing_accounting_or_authority(self):
        original=index();saved=copy.deepcopy(original)
        ordinary=readout(original,budget_bytes=1250)
        terminal=readout(original,budget_bytes=1250,completed=True)
        self.assertEqual(original,saved)
        self.assertEqual(terminal['windows'][0]['id'],'w3')
        self.assertEqual(terminal['windows'][0]['executed_actions'],159)
        self.assertEqual(terminal['accounting'],ordinary['accounting'])
        self.assertEqual(terminal['qualification'],ordinary['qualification'])
        self.assertLessEqual(_size(terminal),1250)
        self.assertEqual(terminal['omitted_windows'],6-len(terminal['windows']))

    def test_zero_action_empty_and_online_context_keep_prior_selection(self):
        zero=index()
        for w in zero['windows']:w['executed_actions']=0
        self.assertEqual(readout(zero),readout(zero,completed=True))
        zero['windows']=[]
        self.assertEqual(readout(zero),readout(zero,completed=True))
        self.assertEqual(readout(index()),readout(index(),completed=False))

    def test_cost_window_cannot_be_silently_dropped_when_budget_is_too_small(self):
        huge=index();huge['windows'][3]['measurement_sequences']=list(range(1000))
        with self.assertRaisesRegex(ValueError,'highest-execution-cost'):
            readout(huge,budget_bytes=1250,completed=True)
        huge['accounting']['audit']='x'*5000
        with self.assertRaisesRegex(ValueError,'accounting contract'):
            readout(huge,budget_bytes=1250,completed=True)
