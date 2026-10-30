import tempfile
import unittest
from pathlib import Path

from kyb.memory import MemoryStore
from kyb.trace import RunTrace
from kyb.session import MemorySession
from manager_fixtures import PreparationClient
import json


class MemoryBehaviorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.trace = RunTrace(self.root / "runA", {"test": True})
        self.store = MemoryStore(self.root / "memory.sqlite")
        self.dependencies = {"body": "x5", "cameras": "cam1", "tool": "key",
                             "attachment:left": "grip1", "attachment:right": "empty"}
        self.before = self.add("observation", {"images": ["before.jpg"], "dependencies": self.dependencies})
        self.receipt = self.add("receipt", {"executed_actions": 3, "execution":"complete", "dependencies": self.dependencies,
                                          "before_observation": self.before["id"]})
        self.obs = self.add("observation", {"images": ["frame.jpg"], "dependencies": self.dependencies,
                                          "after_receipt": self.receipt["id"]})
        self.sources = {self.before["id"], self.obs["id"], self.receipt["id"]}
        self.visible = {}

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def add(self, kind, payload, visibility="policy"):
        reference = self.trace.event(kind, payload, visibility=visibility)
        if visibility == "policy":
            self.store.register(reference, payload)
        return reference

    def patch(self, **changes):
        return {"op": "new", "reading_role": "interaction", "kind": "external", "relation_type": "observed_interaction", "module": "interaction",
                "text": "Offset push visibly rotates the object",
                "conditions": {"task": "push_T"}, "reference_frame": "grasp",
                "epistemic": "visually_inferred", "evidence": sorted(self.sources),
                "receipt_id": self.receipt["id"], "post_observation_id": self.obs["id"],
                **changes}

    def submit(self, patch, **changes):
        return self.store.submit(patch, allowed_sources=changes.get("sources", self.sources),
            cutoff=changes.get("cutoff", {"runA": self.trace.sequence}),
            dependencies=changes.get("dependencies", self.dependencies), episode="runA",
            visible_revisions=self.visible)

    def read(self, **changes):
        result = self.store.read("push", context={"task": "push_T", "arm": "left"},
            dependencies=changes.get("dependencies", self.dependencies),
            episode="runA", at=changes.get("at"), limit=changes.get("limit", 6))
        self.visible.update({row["id"]: row["revision"] for row in result["selected"]})
        return result

    def valid_patch(self, **changes):
        changes.setdefault('reading_role','intrinsic_candidate' if changes.get('kind')=='body' else 'interaction')
        return self.patch(conditions={"task": "push_T", "arm": "left"}, **changes)

    def test_observation_receipt_publish_then_read(self):
        result = self.submit(self.valid_patch())
        self.assertTrue(result["accepted"])
        record = self.read()["selected"][0]
        self.assertEqual(record["evidence"], sorted(self.sources))
        self.assertEqual(record["status"], "provisional")
        self.assertEqual(record["epistemic"], "visually_inferred")

    def test_ordinary_word_overlap_does_not_fill_an_unrelated_history_answer(self):
        claim=self.submit(self.valid_patch(text='After left-hand withdrawal, the released broom lies across the dustpan.'))
        self.assertTrue(claim['accepted'])
        result=self.store.read('Measure the exposed pad and block outlines after withdrawal to distinguish residual planar displacement and rotation from height parallax.',
            context={},dependencies=self.dependencies,episode='different_task')
        self.assertFalse(result['selected']);self.assertFalse(result['historical_events'])

    def test_condition_field_names_are_not_physical_search_content(self):
        claim=self.submit(self.patch(text='The contacted surface moved.', conditions={'arm':'left','accidentalsentinel':'table'}))
        self.assertTrue(claim['accepted'])
        result=self.store.read('accidentalsentinel',context={},dependencies=self.dependencies,episode='runA')
        self.assertFalse(result['selected'])

    def test_execution_dependency_is_source_bound_and_old_dynamics_are_explicitly_unknown(self):
        old=self.submit(self.valid_patch());self.assertTrue(old['accepted'])
        view=self.read(dependencies={**self.dependencies,'execution':'new_resampler'})['selected'][0]
        self.assertEqual(view['execution_protocol_comparison'],'unknown_historical_execution_recheck_dynamics')
        source_deps={**self.dependencies,'execution':'source_resampler'}
        before=self.add('observation',{'dependencies':source_deps})
        receipt=self.add('receipt',{'before_observation':before['id'],'executed_actions':1,'execution':'complete','dependencies':source_deps})
        post=self.add('observation',{'after_receipt':receipt['id'],'dependencies':source_deps})
        patch=self.valid_patch(text='Observed response under the source execution protocol',
            receipt_id=receipt['id'],post_observation_id=post['id'],evidence=[receipt['id'],post['id']])
        published=self.submit(patch,sources={before['id'],receipt['id'],post['id']},dependencies={**source_deps,'execution':'new_resampler'})
        self.assertTrue(published['accepted'],published)
        pins=json.loads(self.store.db.execute('SELECT dependencies FROM claims WHERE revision=?',(published['revision'],)).fetchone()[0])
        self.assertEqual(pins['execution'],'source_resampler')
        retained=self.store.read('execution protocol',context={},dependencies={**source_deps,'execution':'new_resampler'},episode='runA')['selected']
        item=next(r for r in retained if r['id']==published['claim_id'])
        self.assertIn('execution',item['qualification']['changed_derivation_contracts'])
        self.assertEqual(item['qualification']['metric_qualification'],'incomparable_current_contract')

    def test_flagged_revision_is_suspended_then_model_revision_can_replace_it(self):
        first = self.submit(self.valid_patch())
        self.store.flag_review(first['revision'], reason='frame_semantics_conflict', details={'source': self.obs['id']})
        result = self.read()
        self.assertFalse(result['selected'])
        self.assertEqual(result['review_required'][0]['revision'], first['revision'])
        self.visible.update({c['id']:c['revision'] for c in result['review_required']})
        for op in ('support', 'counterexample'):
            attempt = self.submit(self.valid_patch(op=op, claim_id=first['claim_id'],
                claim_revision=first['revision']))
            self.assertFalse(attempt['accepted'])
            self.assertEqual(attempt['reason'], 'suspended_claim_requires_explicit_revision')
        revised = self.submit(self.valid_patch(op='revise', claim_id=first['claim_id'],
            claim_revision=first['revision'], text='Narrowed reference-point interpretation'))
        self.assertTrue(revised['accepted'], revised)
        self.assertEqual(self.read()['selected'][0]['revision'], revised['revision'])
        self.assertFalse(self.read()['review_required'])

    def test_query_can_select_old_body_instead_of_latest_six(self):
        old = self.submit(self.valid_patch(kind='body', text='The wrist camera reveals the gripper.' ))
        for number in range(7):
            self.assertTrue(self.submit(self.valid_patch(kind='body',
                text=f'Joint configuration sample number {number}'))['accepted'])
        result = self.store.read('wrist camera', context={'task':'push_T'},
            dependencies=self.dependencies, episode='runA', limit=6)
        self.assertEqual(result['retrieval_mode'], 'question_relevance_then_qualification')
        self.assertEqual([c['revision'] for c in result['selected']], [old['revision']])
        self.assertEqual(len(result['source_observations']), 2)
        no_match = self.store.read('unindexedtask', context={'task':'push_T'},
            dependencies=self.dependencies, episode='runA', limit=6)
        self.assertEqual(no_match['retrieval_mode'], 'no_lexical_match')
        self.assertFalse(no_match['selected'])
        self.assertEqual(no_match['omission_index']['lexically_unmatched']['count'],8)

    def test_frozen_ranking_corpus_excludes_later_candidates(self):
        self.submit(self.valid_patch(kind='body', text='A wrist camera sees the hand.'))
        self.submit(self.valid_patch(kind='body', text='A camera sees the table.'))
        at = self.store.revision
        def frozen_read():
            return self.store.read('wrist camera', context={'task':'push_T'},
                dependencies=self.dependencies, episode='runA', at=at)['selected']
        before = [(r['revision'],r['retrieval_score_bm25']) for r in frozen_read()]
        self.submit(self.valid_patch(kind='body', text='wrist camera '*50))
        after = [(r['revision'],r['retrieval_score_bm25']) for r in frozen_read()]
        self.assertEqual(before, after)

    def test_invalid_history_stays_in_audit_without_growing_policy_payload(self):
        for number in range(32):
            self.assertTrue(self.submit(self.valid_patch(kind='body', reference_frame='image',
                text=f'Wrist camera relation sample {number}'))['accepted'])
        changed = {**self.dependencies,'cameras':'cam2'}
        trace = RunTrace(self.root/'runB',{})
        session = MemorySession(trace,self.store,task='push_T',dependencies=changed)
        session.evidence('observation',{'dependencies':changed,'state':{}})
        session.attach_manager(PreparationClient())
        messages = [{'role':'user','content':[{'type':'text','text':'current'},
                                             {'type':'image_url','image_url':{'url':'dummy'}}]}]
        session.refresh_context(messages,None)
        from kyb.context import latest_context
        context=latest_context(messages)
        self.assertNotIn('rejected',context)
        self.assertNotIn('objects',context)
        self.assertNotIn('selected',context)
        from kyb.library import organize_archive
        organize_archive(self.store)
        answer=json.loads(session.query_experience({'question':'wrist camera','level':'case'}))
        self.assertTrue(answer['records'])
        self.assertTrue(all(c['applicability']['metric_qualification']=='incomparable_current_contract'
            for c in answer['records']))

    def test_unexecuted_or_future_or_private_cannot_be_effect(self):
        zero = self.add("receipt", {"executed_actions": 0})
        zero_obs = self.add("observation", {"images": ["no_move.jpg"], "dependencies": self.dependencies,
                                           "after_receipt": zero["id"]})
        sources = {zero["id"], zero_obs["id"]}
        self.assertFalse(self.submit(self.valid_patch(evidence=sorted(sources),
            receipt_id=zero["id"], post_observation_id=zero_obs["id"]), sources=sources)["accepted"])
        self.assertFalse(self.submit(self.valid_patch(), cutoff={"runA": 1})["accepted"])
        private = self.add("evaluation", {"success": True}, "private")
        with self.assertRaises(ValueError):
            self.store.register(private, {"success": True})
        self.assertFalse(self.submit(self.valid_patch(evidence=[private["id"]]))["accepted"])
        self.assertFalse(self.submit(self.valid_patch(epistemic="measured"))["accepted"])

    def test_support_deduplicates_and_adds_real_evidence(self):
        record = self.submit(self.valid_patch())
        self.read()
        support = self.valid_patch(op="support", claim_id=record["claim_id"], claim_revision=record["revision"])
        self.assertEqual(self.submit(support)["reason"], "duplicate_evidence")
        newer = self.add("observation", {"images": ["after2.jpg"], "dependencies": self.dependencies,
                                        "after_receipt": self.receipt["id"]})
        sources = self.sources | {newer["id"]}
        self.assertTrue(self.submit({**support, "evidence": sorted(sources),
                                     "post_observation_id": newer["id"]}, sources=sources)["accepted"])
        self.assertEqual(len(self.read()["selected"][0]["evidence"]), 4)

    def test_changed_attachment_invalidates_inside_same_phase(self):
        self.submit(self.valid_patch())
        self.assertEqual(len(self.read()["selected"]), 1)
        changed = {**self.dependencies, "attachment:left": "grip2"}
        retained=self.read(dependencies=changed)['selected'][0]
        self.assertEqual(retained['qualification']['category'],'analogy')
        self.assertEqual(retained['qualification']['body_and_binding_comparison']['attachment:left'],'differ')
        self.assertFalse(retained['qualification']['current_fact'])

    def test_tool_local_relation_can_survive_grasp_change(self):
        self.submit(self.valid_patch(reference_frame="tool_local"))
        changed = {**self.dependencies, "attachment:left": "grip2"}
        self.assertEqual(len(self.read(dependencies=changed)["selected"]), 1)

    def test_body_candidate_uses_source_camera_not_current_camera(self):
        result = self.submit(self.valid_patch(kind="body", reference_frame="image",
            observed_entity="robot_part", observable="pixel_position"),
            dependencies={**self.dependencies, "cameras": "cam2"})
        self.assertTrue(result["accepted"], result)
        record = self.read()["selected"][0]
        self.assertEqual(record["dependencies"]["cameras"], "cam1")
        retained=self.read(dependencies={**self.dependencies, "cameras": "cam2"})['selected'][0]
        self.assertEqual(retained['qualification']['changed_derivation_contracts'],['cameras'])
        self.assertFalse(retained['qualification']['current_fact'])

    def test_body_candidate_cannot_merge_different_source_versions(self):
        different = self.add("observation", {"dependencies": {**self.dependencies, "cameras": "cam2"}})
        sources = self.sources | {different["id"]}
        result = self.submit(self.valid_patch(kind="body", reference_frame="image", evidence=sorted(sources)), sources=sources)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "body_sources_mix_dependency_versions")

    def test_counterexamples_require_same_conditions_and_pins(self):
        record = self.submit(self.valid_patch())
        self.read()
        counter = self.valid_patch(op="counterexample", claim_id=record["claim_id"],
                                  claim_revision=record["revision"],
                                  text="A comparable push visibly translates")
        changed = {**self.dependencies, "attachment:left": "grip2"}
        self.store.annotate_observation(self.obs["id"], changed)
        self.assertFalse(self.submit(counter, dependencies=changed)["accepted"])
        self.store.annotate_observation(self.obs["id"], self.dependencies)
        self.assertFalse(self.submit({**counter, "conditions": {"task": "push_T", "arm": "right"}})["accepted"])
        self.assertTrue(self.submit(counter)["accepted"])
        result = self.read(limit=0)
        self.assertEqual(result["relevant_counterexamples"], 1)
        self.assertEqual(result["selected"][0]["status"], "contested")

    def test_frozen_revision_and_legal_old_episode_evidence(self):
        old = self.submit(self.valid_patch())
        self.read()
        frozen = self.store.revision
        self.submit(self.valid_patch(op="revise", claim_id=old["claim_id"],
                                    claim_revision=old["revision"], text="Revised narrower relation"))
        self.assertIn("Offset push", self.read(at=frozen)["selected"][0]["text"])
        self.assertIn("Revised", self.read()["selected"][0]["text"])
        # Evidence remains legal across episodes when explicitly exposed with its old cutoff.
        result = self.store.submit(self.valid_patch(text="Transferred old observation"),
            allowed_sources=self.sources, cutoff={"runA": 3},
            dependencies=self.dependencies, episode="runB")
        self.assertTrue(result["accepted"])

    def test_unpaired_after_image_cannot_witness_old_action(self):
        unrelated = self.add("observation", {"images": ["unrelated.jpg"], "dependencies": self.dependencies})
        sources = {self.receipt["id"], unrelated["id"]}
        self.assertFalse(self.submit(self.valid_patch(evidence=sorted(sources)), sources=sources)["accepted"])

    def test_segment_keeps_later_negative_outside_window_without_leaking_future_evidence(self):
        first=self.submit(self.valid_patch());self.assertTrue(first['accepted']);self.read()
        old_cutoff=self.trace.sequence
        before=self.add('observation',{'dependencies':self.dependencies})
        receipt=self.add('receipt',{'dependencies':self.dependencies,'executed_actions':1,'execution':'complete',
                                    'before_observation':before['id']})
        post=self.add('observation',{'dependencies':self.dependencies,'after_receipt':receipt['id']})
        negative=self.submit(self.valid_patch(op='counterexample',claim_id=first['claim_id'],
            claim_revision=first['revision'],text='Later comparable push visibly translates',
            receipt_id=receipt['id'],post_observation_id=post['id'],evidence=[receipt['id'],post['id']]),
            sources={before['id'],receipt['id'],post['id']})
        self.assertTrue(negative['accepted'],negative)
        rows=[{'id':self.receipt['id']}]
        reader=self.store._segments
        def effect(revision,cutoff):
            return reader._effects(rows,{'runA':cutoff},revision)[0]
        current=effect(negative['revision'],self.trace.sequence)
        self.assertEqual(current['revision'],first['revision'])
        self.assertEqual(current['status'],'provisional')  # Dated instance retains its own past status.
        proposition=current['proposition_at_read']
        self.assertEqual(proposition['status'],'contested')
        self.assertEqual(proposition['historical_counterexamples'][0]['text'],'Later comparable push visibly translates')
        self.assertIn(post['id'],proposition['evidence'])
        self.assertFalse(effect(first['revision'],self.trace.sequence)['proposition_at_read']['historical_counterexamples'])
        earlier=effect(negative['revision'],old_cutoff)['proposition_at_read']
        self.assertEqual(earlier['revision'],first['revision'])
        self.assertEqual(earlier['status'],'provisional')
        self.read()
        revised=self.submit(self.valid_patch(op='revise',claim_id=first['claim_id'],
            claim_revision=negative['revision'],text='Narrowed candidate after the contrary push'))
        self.assertTrue(revised['accepted'],revised)
        self.assertIn(post['id'],effect(revised['revision'],self.trace.sequence)['proposition_at_read']['evidence'])
        self.assertEqual(effect(revised['revision'],old_cutoff)['proposition_at_read']['revision'],first['revision'])

    def test_support_cannot_remove_contested_status(self):
        first = self.submit(self.valid_patch())
        self.read()
        counter = self.valid_patch(op="counterexample", claim_id=first["claim_id"],
                                  claim_revision=first["revision"], text="Observed comparable contrary result")
        contested = self.submit(counter)
        self.read()
        newer = self.add("observation", {"images": ["newer.jpg"], "dependencies": self.dependencies,
                                        "after_receipt": self.receipt["id"]})
        sources = self.sources | {newer["id"]}
        support = self.valid_patch(op="support", claim_id=first["claim_id"],
                                  claim_revision=contested["revision"], evidence=sorted(sources),
                                  post_observation_id=newer["id"])
        self.assertTrue(self.submit(support, sources=sources)["accepted"])
        self.assertEqual(self.read(limit=0)["selected"][0]["status"], "contested")

    def test_revise_cannot_hide_negative_sources_from_provenance_or_frozen_read(self):
        first=self.submit(self.valid_patch());self.read()
        negative=self.submit(self.valid_patch(op='counterexample',claim_id=first['claim_id'],
            claim_revision=first['revision'],text='Comparable observed contrary result'))
        self.assertTrue(negative['accepted']);self.read()
        newer=self.add('observation',{'dependencies':self.dependencies})
        # A body relation permits a disjoint source, exercising the original body case.
        body={'op':'new','reading_role':'intrinsic_candidate','kind':'body','observed_entity':'robot_part','observable':'configuration',
            'module':'kinematics','text':'Push configuration candidate','conditions':{},'reference_frame':'world',
            'epistemic':'visually_inferred','evidence':[self.obs['id']]}
        initial=self.submit(body);self.assertTrue(initial['accepted']);self.read()
        contrary=self.submit({**body,'op':'counterexample','claim_id':initial['claim_id'],
            'claim_revision':initial['revision'],'text':'Observed contrary body configuration','evidence':[newer['id']]},sources={newer['id']})
        self.assertTrue(contrary['accepted']);self.read()
        rewrite=self.submit({**body,'op':'revise','claim_id':initial['claim_id'],
            'claim_revision':contrary['revision'],'text':'Narrowed rewritten candidate'})
        self.assertTrue(rewrite['accepted'])
        self.assertNotIn(newer['id'],self.store.evidence_ids(rewrite['revision']))
        provenance=self.store.provenance_at(rewrite['revision'])
        self.assertIn(newer['id'],provenance['evidence'])
        self.assertEqual(provenance['counterexamples'][0]['text'],'Observed contrary body configuration')
        row=next(r for r in self.read()['selected'] if r['id']==initial['claim_id'])
        self.assertIn(newer['id'],row['counterexample_evidence'])
        self.assertEqual(row['historical_counterexample_index']['total_records'],1)
        self.assertFalse(self.store.provenance_at(initial['revision'])['counterexamples'])

    def test_update_requires_read_revision_and_rejects_stale_version(self):
        first = self.submit(self.valid_patch())
        update = self.valid_patch(op="revise", claim_id=first["claim_id"],
                                 claim_revision=first["revision"], text="Narrower")
        self.assertFalse(self.submit(update)["accepted"])
        self.read()
        self.assertTrue(self.submit(update)["accepted"])
        self.assertEqual(self.submit(update)["reason"], "stale_claim_revision")

    def test_complete_snapshot_keeps_old_observation_interpretation(self):
        self.submit(self.valid_patch())
        snapshot = self.root / "frozen.sqlite"
        self.store.freeze(snapshot)
        self.store.annotate_observation(self.obs["id"],
            {**self.dependencies, "attachment:left": "reinterpreted"})
        self.assertEqual(self.read()['selected'][0]['qualification']['source_interpretation_revised'],[self.obs['id']])
        branch, metadata = MemoryStore.fork_snapshot(snapshot, self.root / "branch.sqlite")
        try:
            result = branch.read("push", context={"task": "push_T", "arm": "left"},
                                 dependencies=self.dependencies, episode="runA", at=metadata["revision"])
            self.assertEqual(len(result["selected"]), 1)
            self.assertFalse(result['selected'][0]['qualification']['source_interpretation_revised'])
        finally:
            branch.close()

    def test_invalid_claim_does_not_prevent_later_valid_claim(self):
        self.assertFalse(self.submit(self.valid_patch(text=None))["accepted"])
        self.assertTrue(self.submit(self.valid_patch())["accepted"])


if __name__ == "__main__":
    unittest.main()
