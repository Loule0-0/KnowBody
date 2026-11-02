"""One transactional authority for claims, provenance and read-time dependencies."""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from .trace import write_once
from .state import evidence_state, evidence_receipt, compare_states
from .segments import SegmentReader
from .instances import dated_instance, claim_provenance, version_sources
from .qualification import ROLES, qualify, binding_snapshot
from .retrieval import content_terms, sufficient_overlap
from .snapshots import sha256_file


def encoded(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


class MemoryStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS evidence(
          id TEXT PRIMARY KEY, run TEXT NOT NULL, sequence INTEGER NOT NULL,
          kind TEXT NOT NULL, path TEXT NOT NULL, sha256 TEXT NOT NULL,
          payload TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS evidence_kind ON evidence(kind);
        CREATE INDEX IF NOT EXISTS evidence_run_sequence_kind ON evidence(run,sequence,kind);
        CREATE INDEX IF NOT EXISTS evidence_contract_amendment
          ON evidence(json_extract(payload,'$.contract_id')) WHERE kind='task_contract_amendment';
        CREATE INDEX IF NOT EXISTS evidence_after_receipt
          ON evidence(json_extract(payload,'$.after_receipt')) WHERE kind='observation';
        CREATE INDEX IF NOT EXISTS evidence_attachment_observation
          ON evidence(json_extract(payload,'$.observation_id'))
          WHERE kind IN ('attachment_change','attachment_observation');
        CREATE TABLE IF NOT EXISTS claims(
          revision INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL,
          kind TEXT NOT NULL, module TEXT NOT NULL, text TEXT NOT NULL,
          conditions TEXT NOT NULL, dependencies TEXT NOT NULL,
          epistemic TEXT NOT NULL, status TEXT NOT NULL, scope TEXT,
          UNIQUE(id,revision));
        CREATE TABLE IF NOT EXISTS claim_evidence(
          revision INTEGER REFERENCES claims(revision),
          evidence_id TEXT REFERENCES evidence(id),
          PRIMARY KEY(revision,evidence_id));
        CREATE TABLE IF NOT EXISTS observation_dependencies(
          observation_id TEXT PRIMARY KEY REFERENCES evidence(id), pins TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS decisions(
          n INTEGER PRIMARY KEY, patch TEXT NOT NULL, accepted INTEGER NOT NULL,
          reason TEXT NOT NULL, revision INTEGER);
        CREATE INDEX IF NOT EXISTS decisions_revision ON decisions(revision) WHERE accepted=1;
        CREATE TABLE IF NOT EXISTS pending_events(
          event_id TEXT PRIMARY KEY REFERENCES evidence(id), episode TEXT NOT NULL,
          event_kind TEXT NOT NULL, capsule TEXT NOT NULL,
          status TEXT NOT NULL DEFAULT 'pending', resolved_revision INTEGER REFERENCES claims(revision));
        CREATE TABLE IF NOT EXISTS interaction_evidence(
          revision INTEGER PRIMARY KEY REFERENCES claims(revision),
          relation_type TEXT NOT NULL, capsule TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS claim_grounding(
          revision INTEGER PRIMARY KEY REFERENCES claims(revision),
          entity TEXT NOT NULL, observable TEXT NOT NULL, source_episode TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS claim_reading(
          revision INTEGER PRIMARY KEY REFERENCES claims(revision),
          role TEXT NOT NULL, reference_frame TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS observation_coordinates(
          observation_id TEXT PRIMARY KEY REFERENCES evidence(id),
          canonical TEXT NOT NULL, derivation TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS claim_reviews(
          revision INTEGER PRIMARY KEY REFERENCES claims(revision), verdict TEXT NOT NULL,
          reason TEXT NOT NULL, details TEXT NOT NULL);
        CREATE VIRTUAL TABLE temp.knowledge_search USING fts5(text, conditions);
        """)
        self._search_revision = None
        self._segments = SegmentReader(self.db)
        self._batch_depth = 0

    @contextmanager
    def transaction(self):
        if self._batch_depth:
            yield
        else:
            with self.db:
                yield

    def submit_batch(self, patches, **kwargs):
        """A management publication cannot partially commit claims or pending events."""
        class RejectedBatch(Exception):
            pass
        decisions=[]
        try:
            with self.db:
                self._batch_depth += 1
                try:
                    decisions=[self.submit(patch, **kwargs) for patch in patches]
                    if any(not item['accepted'] for item in decisions):
                        raise RejectedBatch()
                finally:
                    self._batch_depth -= 1
        except RejectedBatch:
            self._search_revision = None
            return [{**item, 'validation_accepted':item['accepted'], 'accepted':False,
                'revision':None,'reason':'batch_rolled_back' if item['accepted'] else item['reason']}
                for item in decisions],False
        return decisions,True

    def read_segments(self, query: str, **kwargs) -> dict:
        return self._segments.read(query, **kwargs)

    @property
    def revision(self) -> int:
        return self.db.execute("SELECT COALESCE(MAX(revision),0) FROM claims").fetchone()[0]

    def freeze(self, path: Path) -> dict:
        if path.exists():
            raise ValueError("Snapshot path already exists")
        path.parent.mkdir(parents=True, exist_ok=True)
        target = sqlite3.connect(path)
        self.db.backup(target)
        revision = target.execute("SELECT COALESCE(MAX(revision),0) FROM claims").fetchone()[0]
        target.close()
        manifest = {"revision": revision, "sha256": sha256_file(path),
                    "scope": "complete_database_including_evidence_interpretations"}
        write_once(path.with_suffix(".manifest.json"), manifest)
        return manifest

    @classmethod
    def fork_snapshot(cls, snapshot: Path, target: Path) -> tuple["MemoryStore", dict]:
        manifest = json.loads(snapshot.with_suffix(".manifest.json").read_text())
        if sha256_file(snapshot) != manifest["sha256"]:
            raise ValueError("Frozen database hash mismatch")
        if target.exists():
            raise ValueError("Evaluation branch already exists")
        source = sqlite3.connect(f"file:{snapshot}?mode=ro&immutable=1", uri=True)
        copied=sqlite3.connect(target)
        try:source.backup(copied)
        finally:
            copied.close();source.close()
        # Initialize the copied schema: backup replaces indexes made before it.
        branch = cls(target)
        if branch.revision != manifest["revision"]:
            raise ValueError("Frozen database revision mismatch")
        return branch, manifest

    def _register(self, reference: dict, payload: dict) -> None:
        if not (reference["visibility"]=="policy" or (reference["visibility"]=="recorded_not_exposed" and reference["kind"]=="execution_frame")):
            raise ValueError("Private evaluation is not eligible memory evidence")
        path = Path(reference["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != reference["sha256"]:
            raise ValueError("Evidence hash mismatch")
        original=json.loads(path.read_text())
        if (encoded(original['payload'])!=encoded(payload) or
            any(original[key]!=reference[key] for key in ('id','kind','sequence','visibility'))):
            raise ValueError('Evidence payload or identity differs from its hashed original')
        self.db.execute("INSERT OR IGNORE INTO evidence VALUES(?,?,?,?,?,?,?)",
                (reference["id"], reference["id"].rsplit(":e", 1)[0],
                 reference["sequence"], reference["kind"], str(path),
                 reference["sha256"], encoded(payload)))
        if reference["kind"] == "observation" and "dependencies" in payload:
            self.db.execute("INSERT OR IGNORE INTO observation_dependencies VALUES(?,?)",
                                (reference["id"], encoded(payload["dependencies"])))

    def register(self, reference: dict, payload: dict) -> None:
        with self.transaction():
            self._register(reference, payload)

    def evidence_payload(self, source: str) -> dict:
        row = self.db.execute("SELECT payload FROM evidence WHERE id=?", (source,)).fetchone()
        if row is None:
            raise ValueError("Unknown evidence")
        payload = json.loads(row[0])
        coordinate = self.db.execute('SELECT canonical,derivation FROM observation_coordinates WHERE observation_id=?', (source,)).fetchone()
        if coordinate is not None:
            payload['canonical_state'] = json.loads(coordinate[0])
            payload['coordinate_derivation'] = json.loads(coordinate[1])
        return payload

    def flag_review(self, revision: int, *, reason: str, details: dict) -> None:
        with self.transaction():
            self.db.execute('INSERT INTO claim_reviews VALUES(?,?,?,?)',
                            (revision, 'NEEDS_WORK', reason, encoded(details)))

    def _capsule_view(self, capsule: dict) -> dict:
        return {**capsule,
            'before_state': evidence_state(self.evidence_payload(capsule['before_observation_id'])),
            'post_state': evidence_state(self.evidence_payload(capsule['post_observation_id'])),
            'execution': evidence_receipt(self.evidence_payload(capsule['receipt_id']))}

    def publish_attachment(self, reference: dict, payload: dict, *, observation_id: str,
                           dependencies: dict, capsule: dict | None) -> None:
        with self.transaction():
            self._register(reference, payload)
            self.db.execute("INSERT OR REPLACE INTO observation_dependencies VALUES(?,?)",
                            (observation_id, encoded(dependencies)))
            if capsule is not None:
                self.db.execute("INSERT INTO pending_events(event_id,episode,event_kind,capsule) VALUES(?,?,?,?)",
                    (reference["id"], reference["id"].rsplit(":e", 1)[0], "slipped", encoded(capsule)))

    def pending(self, focus_id: str | None = None) -> list[dict]:
        return [{"pending_event_id": r["event_id"], **self._capsule_view(json.loads(r["capsule"]))}
                for r in self.db.execute("SELECT * FROM pending_events WHERE status='pending' "
                    "ORDER BY (event_id=?) DESC,rowid LIMIT 2", (focus_id,))]

    @property
    def pending_count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM pending_events WHERE status='pending'").fetchone()[0]

    def attachment_event_at(self, observation_id: str, arm: str):
        return self.db.execute("SELECT id FROM evidence WHERE kind='attachment_change' "
            "AND json_extract(payload,'$.observation_id')=? AND json_extract(payload,'$.arm')=?",
            (observation_id, arm)).fetchone()

    def annotate_observation(self, observation_id: str, dependencies: dict[str, str]) -> None:
        """Revise derived binding interpretation; the original observation stays immutable."""
        with self.transaction():
            self.db.execute("INSERT OR REPLACE INTO observation_dependencies VALUES(?,?)",
                            (observation_id, encoded(dependencies)))

    def evidence_ids(self, revision: int) -> list[str]:
        return version_sources(self.db, revision)

    def provenance_at(self, revision: int) -> dict:
        return claim_provenance(self.db, revision)

    def _latest(self, claim_id: str, at: int | None = None) -> sqlite3.Row | None:
        return self.db.execute(
            "SELECT * FROM claims WHERE id=? AND revision<=? ORDER BY revision DESC LIMIT 1",
            (claim_id, self.revision if at is None else at)).fetchone()

    def normalize_patch(self, patch: dict) -> dict:
        """Explicit action witnesses are sources, including their real starting observation."""
        patch = dict(patch)
        sources = patch['evidence']
        if not isinstance(sources, list) or not all(isinstance(s, str) for s in sources):
            raise ValueError('evidence_must_be_string_ids')
        pair = (patch.get('receipt_id'), patch.get('post_observation_id'))
        if any(s is not None for s in pair):
            if not all(isinstance(s, str) and s for s in pair):
                raise ValueError('explicit_receipt_requires_post_observation')
            receipt = self.db.execute("SELECT payload FROM evidence WHERE id=? AND kind='receipt'", (pair[0],)).fetchone()
            if receipt is None:
                raise ValueError('explicit_receipt_not_found')
            before = json.loads(receipt[0]).get('before_observation')
            if not isinstance(before, str):
                raise ValueError('explicit_receipt_requires_before_observation')
            sources = [*sources, before, *pair]
        if patch.get('pending_event_id'):
            pending = self.db.execute('SELECT capsule FROM pending_events WHERE event_id=?', (patch.get('pending_event_id'),)).fetchone()
            if pending is not None:
                sources = [*sources, *json.loads(pending[0])['evidence']]
        patch['evidence'] = sorted(set(sources))
        return patch

    def submit(self, patch: dict, *, allowed_sources: set[str], cutoff: dict[str, int],
               dependencies: dict[str, str], episode: str,
               visible_revisions: dict[str, int] | None = None) -> dict:
        """Accept an auditable provisional claim, never certify VLM text as measurement."""
        try:
            patch = self.normalize_patch(patch)
            role = patch['reading_role']
            if role not in ROLES:
                raise ValueError('unknown_reading_role')
            kind = patch["kind"]
            if kind not in {"body", "external"}:
                raise ValueError("unknown_kind")
            physical_effect=kind=='external' and role=='interaction'
            if kind=='external' and role not in {'interaction','scene_observation'}:
                raise ValueError('external_requires_interaction_reading_role')
            epistemic = patch.get("epistemic", "visually_inferred")
            if epistemic not in {"visually_inferred", "telemetry_inferred", "hypothesis"}:
                raise ValueError("model_claim_is_not_a_host_measurement")
            if epistemic=='telemetry_inferred' and not (kind=='body' and role=='intrinsic_candidate'
                and patch.get('observed_entity')=='robot_part' and patch.get('observable') in ('configuration','relative_geometry')
                and patch.get('reference_frame')=='world'):
                raise ValueError('telemetry_inferred_only_source_time_world_body_configuration_or_geometry')
            relation = patch.get("relation_type") if physical_effect else None
            if physical_effect and relation != "observed_interaction":
                raise ValueError("external_requires_observed_interaction")
            pending = None
            capsule = None
            if physical_effect and patch.get("pending_event_id"):
                if epistemic != "visually_inferred":
                    raise ValueError("transition_requires_observed_event_not_hypothesis")
                pending = self.db.execute("SELECT * FROM pending_events WHERE event_id=?",
                                         (patch.get("pending_event_id"),)).fetchone()
                if pending is None:
                    raise ValueError("transition_requires_host_pending_event: pending_event_id="+str(patch.get('pending_event_id'))+
                        "; use an exact registered host pending-event ID, not an observation or receipt ID. "
                        "For a historical receipt-backed interaction, omit this optional field and retain "
                        "relation_type=observed_interaction with its before/receipt/post evidence; all evidence checks still apply.")
                capsule = json.loads(pending["capsule"])
                if patch.get("op", "new") not in {"new", "revise"}:
                    raise ValueError("transition_is_an_event_not_statistical_support_or_counterexample")
                if pending["status"] == "resolved" and patch.get("op", "new") == "new":
                    raise ValueError("event_already_resolved_not_a_second_experience")
            if not isinstance(patch["text"], str):
                raise ValueError("claim_text_must_be_string")
            text = patch["text"].strip()
            if patch.get("observed_entity", "unknown") not in {"robot_part", "tool_part", "scene_object", "unknown"}:
                raise ValueError("unknown_observed_entity")
            if patch.get("observable", "unknown") not in {"configuration", "relative_geometry", "pixel_position", "visibility", "local_response", "interaction_effect", "unknown"}:
                raise ValueError("unknown_observable")
            if not text or len(text) > 1800:
                raise ValueError('text: '+str(len(text))+' Unicode characters; required1..1800 characters')
            conditions = patch.get("conditions", {})
            if not isinstance(conditions, dict) or any(not isinstance(v, (str, bool, int, float)) for v in conditions.values()):
                raise ValueError("conditions_must_be_scalar_fields")
            if not isinstance(patch["evidence"], list) or not all(isinstance(s, str) for s in patch["evidence"]):
                raise ValueError("evidence_must_be_string_ids")
            sources = sorted(set(patch["evidence"]))
            if capsule is not None:
                sources = sorted(set(sources) | set(capsule["evidence"]))
            patch['evidence'] = sources
            if not sources or not set(sources) <= allowed_sources:
                decision = self._decision(patch, False, "source_not_exposed_to_policy", None)
                decision["invalid_sources"] = sorted(set(sources)-allowed_sources)
                return decision
            evidence = []
            for source in sources:
                row = self.db.execute("SELECT * FROM evidence WHERE id=?", (source,)).fetchone()
                if row is None or row["run"] not in cutoff or row["sequence"] > cutoff[row["run"]]:
                    raise ValueError("source_outside_visible_lineage_or_cutoff")
                from .contracts import INTERPRETATION_KINDS
                if row['kind'] in INTERPRETATION_KINDS:
                    raise ValueError('episode_candidate_is_not_physical_case_evidence')
                if hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest() != row["sha256"]:
                    raise ValueError("source_modified")
                evidence.append(row)
            if epistemic=='telemetry_inferred':
                observations=[json.loads(e['payload']) for e in evidence if e['kind']=='observation']
                if not observations or any(not o.get('canonical_state',{}).get('coordinate_contract',{}).get('id') or not o['canonical_state'].get('arms') for o in observations):
                    raise ValueError('telemetry_inferred_requires_recorded_EEF_coordinate_contract')
                if len({o['canonical_state']['coordinate_contract']['id'] for o in observations})!=1:
                    raise ValueError('telemetry_inferred_sources_mix_coordinate_contracts')
            witnessed = []
            for receipt in [r for r in evidence if r["kind"] == "receipt"]:
                data = json.loads(receipt["payload"])
                for observation in [r for r in evidence if r["kind"] == "observation"]:
                    observed = json.loads(observation["payload"])
                    if (data.get("executed_actions", 0) > 0
                        and data.get("execution") != "outcome_unknown"
                        and data.get("unknown_command") is None
                        and observed.get("after_receipt") == receipt["id"]
                        and observation["run"] == receipt["run"]
                        and observation["sequence"] > receipt["sequence"]):
                        annotation = self.db.execute(
                            "SELECT pins FROM observation_dependencies WHERE observation_id=?",
                            (observation["id"],)).fetchone()
                        if annotation is not None:
                            witnessed.append({**data, "receipt_id": receipt["id"],
                                "post_observation_id": observation["id"],
                                "post_dependencies": json.loads(annotation["pins"])})
            if epistemic=='telemetry_inferred' and any(e['kind']=='receipt' for e in evidence) and not (patch.get('receipt_id') and patch.get('post_observation_id')):
                raise ValueError('telemetry_motion_requires_explicit_before_receipt_post_pair')
            explicit = ((capsule["receipt_id"], capsule["post_observation_id"]) if capsule else
                        (patch.get("receipt_id"), patch.get("post_observation_id")))
            if any(explicit):
                witnessed = [w for w in witnessed if (w['receipt_id'], w['post_observation_id']) == explicit]
                if len(witnessed) != 1:
                    raise ValueError('explicit_pair_requires_known_executed_action_and_matching_post')
                w = witnessed[0]
                before = next((r for r in evidence if r['id'] == w.get('before_observation') and r['kind'] == 'observation'), None)
                receipt_row = next(r for r in evidence if r['id'] == w['receipt_id'])
                if (before is None or before['run'] != receipt_row['run'] or before['sequence'] >= receipt_row['sequence']
                    or w.get('execution') not in {'complete', 'partial'}):
                    raise ValueError('explicit_pair_requires_true_before_and_known_execution')
            if physical_effect and epistemic != "hypothesis":
                if any(explicit):
                    witnessed = [w for w in witnessed if (w["receipt_id"], w["post_observation_id"]) == explicit]
                if len(witnessed) != 1:
                    raise ValueError("effect_requires_one_explicit_or_unique_interaction_pair")
            if physical_effect and epistemic != "hypothesis" and not witnessed:
                raise ValueError("no_bound_executed_action_and_post_observation")
            module = patch.get("module", "interaction")
            frame = patch.get("reference_frame", "image")
            depends_on = patch.get("depends_on", [])
            if not isinstance(depends_on, list) or not all(isinstance(key, str) for key in depends_on):
                raise ValueError("depends_on_must_be_string_keys")
            # Physical applicability is compared at read time, not a global hard gate.
            dependency_keys = set(depends_on)
            if kind == 'body':
                dependency_keys.add('body')
            if physical_effect and witnessed:
                # Derive participating/changed arms from actual execution, not model omissions.
                w = witnessed[0]
                arms = set(w.get("plan", {}).get("trace", {}).get("arms", {}))
                arms.update(k.split(":", 1)[1] for k, v in w.get("dependencies", {}).items()
                            if k.startswith("attachment:") and v != w["post_dependencies"].get(k))
                dependency_keys.update(f"attachment:{arm}" for arm in arms)
                if 'execution' in w.get('dependencies', {}):
                    dependency_keys.add('execution')
            if frame == "image":
                dependency_keys.add("cameras")
            elif frame not in {"world", "tool_local", "grasp"}:
                raise ValueError("unknown_reference_frame")
            # Tool-local is a reference convention, not a task-derived identity.
            if any(e['kind']=='surface_inspection' for e in evidence):
                dependency_keys.update(('depth','cameras'))
            if frame == "grasp" or module == "attachment":
                arm = conditions.get("arm")
                if arm not in {"left", "right"}:
                    raise ValueError("attachment_requires_arm")
                dependency_keys.add(f"attachment:{arm}")
            if not dependency_keys <= dependencies.keys():
                raise ValueError("unknown_dependency")
            witness = witnessed[0] if physical_effect and epistemic != "hypothesis" and witnessed else None
            source_dependencies = witness.get("dependencies", {}) if witness else dependencies
            if kind == "body" or role=="scene_observation":
                planning_sources = [json.loads(e['payload']) for e in evidence if e['kind'] in ('planning_result','path_validation','planning_feedback')]
                if planning_sources:
                    dependency_keys.add('planner')
                if any(e['kind']=='planning_feedback' for e in evidence):
                    # Costs use the execution contract's resampling, step limits and frequency.
                    dependency_keys.add('execution')
                source_states = [json.loads(e["payload"]).get("dependencies", {})
                                 for e in evidence if e["kind"] == "observation"]
                if not source_states or any(not dependency_keys <= s.keys() for s in source_states):
                    raise ValueError("body_requires_source_observation_dependency_versions")
                if any(any(s[k] != source_states[0][k] for k in dependency_keys) for s in source_states):
                    raise ValueError("body_sources_mix_dependency_versions")
                if any(any(p['dependencies'].get(k)!=source_states[0][k] for k in dependency_keys)
                       or p['before_observation_id'] not in sources for p in planning_sources):
                    raise ValueError('planning_source_requires_matching_before_observation_and_versions')
                source_dependencies = source_states[0]
            surface_sources=[(e['id'],json.loads(e['payload'])) for e in evidence if e['kind']=='surface_inspection']
            for measurement_id,measurement in surface_sources:
                observation_sources=[json.loads(e['payload']) for e in evidence
                    if e['kind']=='observation' and e['id']==measurement['before_observation_id']]
                if len(observation_sources)!=1:
                    raise ValueError('surface_source_requires_its_exact_observation')
                observed_pins=observation_sources[0]['dependencies']
                if any(measurement['dependencies'].get(k) is None
                    or measurement['dependencies'].get(k)!=observed_pins.get(k)
                    or measurement['dependencies'].get(k)!=source_dependencies.get(k)
                    for k in ('body','cameras','depth')):
                    raise ValueError('surface_source_requires_matching_geometry_and_depth_versions')
                if kind=='external' and witness and measurement['before_observation_id'] not in {
                    witness['before_observation'],witness['post_observation_id']}:
                    raise ValueError('surface_effect_requires_selected_interaction_endpoint: '+json.dumps({
                        'measurement_id':measurement_id,'measurement_observation':measurement['before_observation_id'],
                        'selected_receipt':witness['receipt_id'],
                        'allowed_observations':[witness['before_observation'],witness['post_observation_id']]})+
                        '; select the actually corresponding read interaction, or split distinct stages into separate cases. '
                        'Do not change IDs merely to pass validation; endpoint agreement does not certify the full narrative.')
            if not dependency_keys <= source_dependencies.keys():
                raise ValueError("missing_execution_dependency_snapshot")
            pins = {key: source_dependencies[key] for key in sorted(dependency_keys)}
            claim_id = patch.get("claim_id")
            op = patch.get("op", "new")
            transitions = [key for key in pins if key.startswith("attachment:")
                           and witness and pins[key] != witness["post_dependencies"].get(key)]
            if op in {"support", "counterexample"} and transitions:
                raise ValueError("transition_cannot_support_or_refute_stable_relation")
            if op == "counterexample" and epistemic == "hypothesis":
                raise ValueError("a_hypothesis_is_not_an_observed_counterexample")
            previous = None
            if claim_id:
                if op == "new":
                    previous = self._latest(claim_id)
                else:
                    read_revision = patch.get("claim_revision")
                    if visible_revisions is None or visible_revisions.get(claim_id) != read_revision:
                        raise ValueError("update_requires_actually_read_claim_revision")
                    previous = self.db.execute("SELECT * FROM claims WHERE id=? AND revision=?",
                                               (claim_id, read_revision)).fetchone()
                    latest = self._latest(claim_id)
                    if previous is not None and latest["revision"] != read_revision:
                        raise ValueError("stale_claim_revision")
            if op not in {"new", "support", "revise", "counterexample"}:
                raise ValueError("unknown_update_op")
            if op != "new" and previous is None:
                raise ValueError("missing_claim_to_update")
            if op == "new" and previous is not None:
                raise ValueError("existing_claim_requires_explicit_update")
            if previous is not None:
                if op != 'revise' and self.db.execute(
                        'SELECT 1 FROM claim_reviews WHERE revision=?', (previous['revision'],)).fetchone():
                    raise ValueError('suspended_claim_requires_explicit_revision')
                if previous["kind"] != kind:
                    raise ValueError("cannot_change_claim_kind")
                previous_reading=self.db.execute('SELECT role,reference_frame FROM claim_reading WHERE revision=?',(previous['revision'],)).fetchone()
                if op in {'support','counterexample'} and (previous_reading['role']!=role or previous_reading['reference_frame']!=frame):
                    raise ValueError('changed_reading_role_or_frame_requires_revision')
                if kind=="external" and (previous_reading["role"]=="interaction")!=physical_effect:
                    raise ValueError("changed_static_or_interaction_semantics_requires_new_case")
                if physical_effect:
                    previous_interaction = self.db.execute("SELECT * FROM interaction_evidence WHERE revision=?",
                                                          (previous["revision"],)).fetchone()
                    if previous_interaction is None or previous_interaction["relation_type"] != relation:
                        raise ValueError("cannot_change_external_relation_type")
                    if physical_effect and patch.get("pending_event_id"):
                        old_capsule = json.loads(previous_interaction["capsule"])
                        if old_capsule["attachment_event_id"] != pending["event_id"]:
                            raise ValueError("cannot_change_transition_event_identity")
                        if pending["status"] != "resolved":
                            raise ValueError("event_first_publication_requires_new")
                if pending is not None and pending["resolved_revision"] is not None:
                    recorded = self.db.execute("SELECT id FROM claims WHERE revision=?",
                                               (pending["resolved_revision"],)).fetchone()
                    if recorded["id"] != previous["id"]:
                        raise ValueError("transition_revision_must_revise_same_event_claim")
                if op in {"support", "counterexample"} and json.loads(previous["conditions"]) != conditions:
                    raise ValueError("different_conditions_require_a_new_cell")
                if op in {"support", "counterexample"} and json.loads(previous["dependencies"]) != pins:
                    raise ValueError("changed_dependency_is_not_comparable_evidence")
                if op == "support":
                    if json.loads(previous["dependencies"]) != pins:
                        raise ValueError("changed_dependency_is_not_same_support")
                    old_sources = self.evidence_ids(previous["revision"])
                    if set(sources) <= set(old_sources):
                        return self._decision(patch, False, "duplicate_evidence", previous["revision"])
                    sources = sorted(set(sources) - set(old_sources))
                    text = previous["text"]
                elif op == "counterexample":
                    sources = sorted(set(sources) - set(self.evidence_ids(previous["revision"])))
                    text = previous["text"] + "\nCounterexample: " + text
            if not claim_id:
                claim_id = "k" + hashlib.sha256(encoded([kind, module, text, conditions]).encode()).hexdigest()[:16]
                if self._latest(claim_id):
                    return self._decision(patch, False, "duplicate_claim_use_support", None)
            status = ("contested" if op == "counterexample" or
                      (previous is not None and previous["status"] == "contested") else "provisional")
            scope = episode if module == "attachment" else None
            with self.transaction():
                cur = self.db.execute(
                    "INSERT INTO claims(id,kind,module,text,conditions,dependencies,epistemic,status,scope) VALUES(?,?,?,?,?,?,?,?,?)",
                    (claim_id, kind, module, text, encoded(conditions), encoded(pins), epistemic, status, scope))
                revision = cur.lastrowid
                self.db.executemany("INSERT INTO claim_evidence VALUES(?,?)",
                                   [(revision, source) for source in sources])
                self.db.execute('INSERT INTO claim_reading VALUES(?,?,?)', (revision, role, frame))
                self._search_revision = None
                self.db.execute("INSERT INTO claim_grounding VALUES(?,?,?,?)", (revision,
                    patch.get("observed_entity", "unknown"), patch.get("observable", "unknown"), episode))
                if physical_effect:
                    interaction = capsule if capsule is not None else {"witness": witness,
                        "execution_dependencies": source_dependencies,
                        "before_observation_id":witness.get("before_observation") if witness else None,
                        "receipt_id":witness.get("receipt_id") if witness else None,
                        "post_observation_id":witness.get("post_observation_id") if witness else None,
                        "changed_attachment_keys":transitions,
                        "qualification":"Observed interval, attachment declarations are not contact or causal proof"}
                    self.db.execute("INSERT INTO interaction_evidence VALUES(?,?,?)",
                                    (revision, relation, encoded(interaction)))
                if pending is not None:
                    self.db.execute("UPDATE pending_events SET status='resolved',resolved_revision=? WHERE event_id=?",
                                    (revision, pending["event_id"]))
                self.db.execute("INSERT INTO decisions(patch,accepted,reason,revision) VALUES(?,1,?,?)",
                                (encoded(patch), "published_provisional_not_certified", revision))
            return dict(accepted=True, reason="published_provisional_not_certified",
                        claim_id=claim_id, revision=revision, status=status)
        except (KeyError, TypeError, ValueError) as error:
            return self._decision(patch, False, str(error), None)

    def _decision(self, patch: dict, accepted: bool, reason: str, revision: int | None) -> dict:
        with self.transaction():
            self.db.execute("INSERT INTO decisions(patch,accepted,reason,revision) VALUES(?,?,?,?)",
                            (encoded(patch), int(accepted), reason, revision))
        return dict(accepted=accepted, reason=reason, revision=revision)

    def read(self, query: str, *, context: dict, dependencies: dict[str, str],
             episode: str, at: int | None = None, limit: int = 6,
             current_observation: dict | None = None, evidence_cutoff: dict | None = None,
             domain: str | None = None, family: str | None = None, roles: dict | None = None,
             structured: bool = False, include_unresolved: bool = False, claim_refs: list | None = None, browse: bool = False) -> dict:
        at = self.revision if at is None else at
        if self.db.execute('SELECT COUNT(*) FROM claims WHERE revision<=?',(at,)).fetchone()[0] != self.db.execute(
            'SELECT COUNT(*) FROM claim_reading WHERE revision<=?',(at,)).fetchone()[0]:
            raise ValueError('Unsupported old memory; use an explicitly derived snapshot, not a runtime fallback')
        latest = self.db.execute("""SELECT c.*,g.entity,g.observable,g.source_episode,cr.role
            FROM claims c JOIN claim_reading cr ON cr.revision=c.revision
            LEFT JOIN claim_grounding g ON g.revision=c.revision
            WHERE c.revision IN (SELECT MAX(revision) FROM claims WHERE revision<=? GROUP BY id)
            ORDER BY c.revision DESC""", (at,)).fetchall()
        requested={ref['id']:ref['revision'] for ref in claim_refs} if claim_refs is not None else None
        if requested is not None:
            latest=[]
            for identifier,revision in requested.items():
                r=self.db.execute('SELECT c.*,g.entity,g.observable,g.source_episode,cr.role FROM claims c '
                    'JOIN claim_reading cr ON cr.revision=c.revision LEFT JOIN claim_grounding g ON g.revision=c.revision '
                    'WHERE c.id=? AND c.revision=? AND c.revision<=?',(identifier,revision,at)).fetchone()
                if r is None:raise ValueError('Unknown case revision')
                latest.append(r)
        outside=[]
        def available(source):
            ref=self.db.execute('SELECT run,sequence FROM evidence WHERE id=?',(source,)).fetchone()
            return ref is not None and (evidence_cutoff is None or ref['sequence']<=evidence_cutoff.get(ref['run'],-1))
        if evidence_cutoff is not None:
            readable=[]
            identifiers={r['id'] for r in latest}
            for identifier in sorted(identifiers):
                for r in self.db.execute('SELECT c.*,g.entity,g.observable,g.source_episode,cr.role FROM claims c '
                    'JOIN claim_reading cr ON cr.revision=c.revision LEFT JOIN claim_grounding g ON g.revision=c.revision '
                    'WHERE c.id=? AND c.revision<=? ORDER BY c.revision DESC',(identifier,at)):
                    if requested is not None and r['revision']!=requested[identifier]:continue
                    if all(available(source) for source in self.provenance_at(r['revision'])['evidence']):
                        readable.append(r)
                        break
                    outside.append(f"{r['id']}@{r['revision']}")
            latest=readable
        # Navigation belongs to the selected visible revision. An unavailable
        # newer revision must neither hide an old route nor route its old text.
        if structured:
            clauses=[];values=[]
            for field,value in [('domain',domain),('family',family)]:
                if value is not None:clauses.append(field+'=?');values.append(value)
            for key,value in (roles or {}).items():
                clauses.append("(json_extract(roles,?) IS NULL OR json_extract(roles,?) IN ('unknown',?))")
                values.extend(['$.'+key,'$.'+key,value])
            sql='SELECT revision FROM case_navigation'+(' WHERE '+' AND '.join(clauses) if clauses else '')
            revisions={r[0] for r in self.db.execute(sql,values)}
            latest=[r for r in latest if r['revision'] in revisions]
        search_key=(at,tuple(sorted(r['revision'] for r in latest)),
                    tuple(sorted(evidence_cutoff.items())) if evidence_cutoff is not None else None)
        if self._search_revision != search_key:
            with self.transaction():
                self.db.execute('DELETE FROM temp.knowledge_search')
                indexed=[]
                for r in latest:
                    dated=self.db.execute('SELECT d.patch FROM decisions d JOIN claims c ON c.revision=d.revision '
                        'WHERE d.accepted=1 AND c.id=? AND c.revision<=? ORDER BY c.revision',(r['id'],r['revision']))
                    texts=[r['text']]
                    for d in dated:
                        patch=json.loads(d[0])
                        if evidence_cutoff is None or all(available(source) for source in patch['evidence']):
                            texts.append(patch['text'])
                    texts=list(dict.fromkeys(texts))
                    indexed.append((r['revision'],'\n'.join(texts),' '.join(
                        str(v) for k,v in json.loads(r['conditions']).items() if k not in {'task','phase','episode'})))
                self.db.executemany('INSERT INTO temp.knowledge_search(rowid,text,conditions) VALUES(?,?,?)',indexed)
            self._search_revision = search_key
        if browse and (not structured or (domain is None and family is None)):
            raise ValueError('Structured browse requires an explicit route')
        terms = content_terms(query)[:24]
        matching = {}
        if terms:
            q = ' OR '.join('"' + term.replace('"', '""') + '"' for term in terms)
            matching = {row[0]:row[1] for row in self.db.execute(
                'SELECT rowid,bm25(knowledge_search),text,conditions FROM temp.knowledge_search '
                'WHERE knowledge_search MATCH ?', (q,)) if sufficient_overlap(terms, row[2]+' '+row[3])}
        if requested is not None or browse:matching={row['revision']:0 for row in latest}
        candidates, historical, reviews, index = [], [], [], ({'outside_source_frontier':outside} if outside else {})
        for row in latest:
            if row['revision'] not in matching:
                index.setdefault('lexically_unmatched', []).append(f"{row['id']}@{row['revision']}")
                continue
            provenance = self.provenance_at(row['revision'])
            pins, conditions = json.loads(row['dependencies']), json.loads(row['conditions'])
            observations = [source for source in provenance['evidence'] if self.db.execute(
                'SELECT kind FROM evidence WHERE id=?', (source,)).fetchone()[0] == 'observation']
            bindings = {source: binding_snapshot(self.db, source, evidence_cutoff) for source in observations}
            source_runs=sorted({self.db.execute('SELECT run FROM evidence WHERE id=?',(source,)).fetchone()[0]
                                for source in observations})
            qualification = qualify(role=row['role'], conditions=conditions, pins=pins, context=context,
                dependencies=dependencies, source_episode=source_runs[0] if len(source_runs)==1 else None, episode=episode,
                source_bindings=bindings)
            revised_sources=[]
            for source in observations:
                if evidence_cutoff is None:
                    annotation=self.db.execute('SELECT pins FROM observation_dependencies WHERE observation_id=?',(source,)).fetchone()
                    observed=json.loads(annotation[0]) if annotation else {}
                else:
                    # Mutable latest annotations are not an as-of ledger. Only
                    # source-bounded declarations may revise an old binding.
                    observed=self.evidence_payload(source).get('dependencies',{}).copy()
                    for arm,binding in bindings[source].items():
                        observed[f'attachment:{arm}']=binding.get('binding_id')
                if any(observed.get(k) is not None and observed[k]!=v for k,v in pins.items()
                       if k=='body' or k.startswith(('attachment:','part:'))):
                    revised_sources.append(source)
            qualification['source_interpretation_revised']=revised_sources
            if revised_sources:
                qualification['category']='source_interpretation_revised'
                qualification['metric_qualification']='incomparable_revised_source_interpretation'
            review = self.db.execute('SELECT * FROM claim_reviews WHERE revision=?', (row['revision'],)).fetchone()
            relation = self.db.execute('SELECT * FROM interaction_evidence WHERE revision=?', (row['revision'],)).fetchone()
            item = dict(id=row['id'], revision=row['revision'], kind=row['kind'], module=row['module'],
                reading_role=row['role'], text=row['text'], conditions=conditions, epistemic=row['epistemic'],
                status=row['status'], dependencies=pins, qualification=qualification,
                unknown_conditions=qualification['unknown_conditions'], different_conditions=qualification['different_conditions'],
                evidence=provenance['evidence'], retrieval_score_bm25=matching.get(row['revision']),
                grounding={'entity':row['entity'],'observable':row['observable'],'publication_episode':row['source_episode'],
                           'source_episodes':source_runs},
                application='historical_conditional_evidence_not_current_state_or_metric_guarantee')
            if review:
                item.update(review={**dict(review), 'details':json.loads(review['details'])})
                reviews.append(item)
                continue
            if relation and 'attachment_event_id' in json.loads(relation['capsule']):
                item.update(interaction=self._capsule_view(json.loads(relation['capsule'])),
                            application='historical_transition_not_current_state_or_general_law',
                            changed_dependencies=[k for k,v in pins.items() if dependencies.get(k)!=v])
                historical.append(item)
                continue
            if qualification['category'] == 'identity_unresolved' and not include_unresolved:
                index.setdefault('identity_unresolved', []).append(f"{row['id']}@{row['revision']}")
                continue
            item['execution_protocol_comparison'] = ('unknown_historical_execution_recheck_dynamics' if 'execution' not in pins
                else 'same' if pins['execution'] == dependencies.get('execution') else 'different')
            item['text_role'] = 'candidate_proposition_not_latest_support_observation'
            item['latest_dated_instance'] = dated_instance(self.db, row['revision'])
            item['historical_counterexample_index'] = {'total_records':len(provenance['counterexamples']),
                'latest':provenance['counterexamples'][-1:], 'access':'management read_evidence retains original negative conditions'}
            item['counterexample_evidence'] = provenance['counterexamples'][-1]['evidence'] if provenance['counterexamples'] else []
            candidates.append(item)
        # A contradiction stays attached to its candidate; collect before the ordinary cap.
        def rank(item):
            q = item['qualification']; values = q['physical_conditions'].values()
            qualification=(q['category'] != 'conditional_candidate', -sum(v=='match_declared' for v in values),
                len(q['unknown_conditions']), -item['revision'])
            if terms and requested is None and not browse:
                return (item['retrieval_score_bm25'], *qualification)
            return (qualification[0], qualification[1], item['retrieval_score_bm25'], *qualification[2:])
        candidates.sort(key=rank)
        negatives = [c for c in candidates if c['historical_counterexample_index']['total_records']]
        ordinary = [c for c in candidates if c not in negatives]
        pool=sorted(ordinary+historical+reviews,key=rank)
        chosen={c['revision'] for c in pool[:limit]}
        selected = negatives + [c for c in ordinary if c['revision'] in chosen]
        for c in pool[limit:]:
            index.setdefault('budget', []).append(f"{c['id']}@{c['revision']}")
        historical=sorted([c for c in historical if c['revision'] in chosen],key=rank)
        reviews=sorted([c for c in reviews if c['revision'] in chosen],key=rank)
        source_observations = {}
        for item in selected + historical[:limit] + reviews[:limit]:
            sources = set(item['evidence'])
            if item.get('latest_dated_instance'):
                sources.update(item['latest_dated_instance']['evidence'])
            for source in sources:
                if source in source_observations or self.db.execute('SELECT kind FROM evidence WHERE id=?',(source,)).fetchone()[0]!='observation':
                    continue
                data = self.evidence_payload(source)
                source_observations[source] = {'state':evidence_state(data), 'dependencies':data.get('dependencies',{})}
                if current_observation is not None:
                    source_observations[source]['source_state_comparison'] = compare_states(data,current_observation)
        return dict(knowledge_revision=at, query=query, selected=selected, historical_events=historical[:limit],
            review_required=reviews[:limit], source_observations=source_observations,
            omission_index={k:{'count':len(v),'ids':v} for k,v in index.items()},
            retrieval_mode=('no_cases_for_navigation' if not latest else
                            ('question_relevance_then_qualification' if terms and requested is None and not browse else 'physical_qualification_then_lexical') if matching else 'no_lexical_match'),
            relevant_counterexamples=len(negatives),
            caution='Model declarations remain provisional; unknown differs from known different. Task labels are not physical identity.')

    def close(self) -> None:
        self.db.close()
