"""Connect existing policy turns to body state, readouts and transactional updates."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from .memory import MemoryStore
from .trace import RunTrace, plain
from .context import surface_readout
from .knowledge import read_experience


class MemorySession:
    def __init__(self, trace: RunTrace, store: MemoryStore, *, task: str,
                 dependencies: dict[str, str], frozen_revision: int | None = None,
                 perception_profile: str = 'rgbd_calibrated', retrieval_index=None):
        self.trace, self.store = trace, store
        self.episode = trace.root.name
        self.context: dict[str, Any] = {"task": task}
        self.dependencies = dict(dependencies)
        from .perception import select
        self.perception = select(perception_profile)
        self.retrieval_index = retrieval_index
        self.perception.check_memory(store)
        self.frozen_revision = frozen_revision
        from .library import initialize
        initialize(store)
        self.frozen_knowledge_revision=store.db.execute('SELECT COALESCE(MAX(revision),0) FROM reusable_knowledge').fetchone()[0] if frozen_revision is not None else None
        self.budget=None
        self.pause_after_preparation=False
        self.allowed: set[str] = set()
        self.cutoff: dict[str, int] = {}
        self.current_observation: dict | None = None
        self.last_receipt: dict | None = None
        self.attachment = {arm: {"tool": "unknown", "state": "unknown"} for arm in ("left", "right")}
        self.task_instruction = task
        self.manager = None
        self.task_specification = None
        self.last_decisions: list[dict] = []
        self.body_prior: dict | None = None
        self.last_path_validation: dict | None = None
        self.previous_observation: dict | None = None
        self.public_task_recipe=None
        self.episode_ended=False
        self.contract_amendments=[]

    def get_retrieval_index(self):
        if self.retrieval_index is None:
            from .embeddings import VectorIndex
            self.retrieval_index=VectorIndex(self.trace.root/'retrieval_index.sqlite',
                line_id=self.episode,profile=self.perception.name)
        return self.retrieval_index

    def close_retrieval(self):
        if self.retrieval_index is not None:
            self.retrieval_index.close()
            self.retrieval_index=None

    def action_allowance_low(self) -> bool:
        return self.budget is not None and self.budget.view()['remaining']<=2

    def stop_requested(self) -> bool:
        return (self.trace.root/'STOP').exists()

    def evidence(self, kind: str, payload: dict) -> dict:
        self.perception.check_payload(kind, payload)
        if kind=='known_config':
            payload={**payload,'dependencies':dict(self.dependencies),'perception_profile':self.perception.name}
        reference = self.trace.event(kind, payload)
        self.store.register(reference, plain(payload))
        self.allowed.add(reference["id"])
        self.cutoff[self.trace.root.name] = reference["sequence"]
        if kind == "observation":
            if self.current_observation is None and payload.get('instruction'):
                self.task_instruction = payload['instruction']
            self.previous_observation = self.current_observation
            self.current_observation = {**reference, "data": plain(payload)}
            from .body_index import index_observation
            with self.store.db:index_observation(self.store,reference['id'])
        elif kind == "receipt":
            self.last_receipt = {**reference, "data": plain(payload)}
        elif kind == "known_config":
            self.body_prior = {**reference, "data": plain(payload)}
        elif kind == 'path_validation':
            self.last_path_validation = {**reference,'data':plain(payload)}
        return reference

    def record_plan(self, arm: str, target_pose: Any, planned: dict, policy_call_index: int) -> dict:
        from XPolicyLab.policy.GPT_6_Astra_Direct_EEF import pose
        if self.current_observation is None:
            raise ValueError('Planning evidence requires the actual source observation')
        # Preserve the received path before BasePolicy converts/prepends/resamples.
        # The observed start below is not an internal solver-input dump.
        import numpy as np
        import os,hashlib
        position=planned.get('position');artifact=None
        if position is not None:
            raw=np.array(position,copy=True,order='K')
            path=self.trace.root/'planner'/f"reply_{self.trace.sequence+1:06d}_{arm}.npy"
            path.parent.mkdir(exist_ok=True)
            with path.open('xb') as stream:
                np.save(stream,raw,allow_pickle=False)
                stream.flush();os.fsync(stream.fileno())
            artifact={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'dtype':str(raw.dtype),'shape':list(raw.shape),'units':'radian',
                'column_order':'Unmodified columns from this arm planner reply'}
        reference = self.evidence('planning_result', {
            'before_observation_id': self.current_observation['id'], 'policy_call_index': policy_call_index,
            'arm': arm, 'resolved_grasp_target': pose.pose_to_values(target_pose),
            'flange_target_xyz_m_quat_wxyz': plain(target_pose),
            'start_joint_angles_rad': self.current_observation['data']['state'][f'{arm}_arm_joint_state'],
            'dependencies': dict(self.dependencies), 'status': str(planned.get('status', 'unspecified')),
            'path_returned': planned.get('position') is not None,
            'received_planner_position':artifact,
            'planner_reply_qualification':'Wrapper-received planner return before float32 conversion, start prepend and resampling; not executed actions. '
                'start_joint_angles_rad is from the capture, not an internal solver-input dump. No branch normalization applied.',
            'reason': 'unspecified_by_existing_planner', 'executed_actions': 0,
            'interpretation': 'Computational planning result under these settings; failure does not prove '
                              'physical impossibility, and success does not prove actual tracking.'})
        return reference

    def inspect_surface(self, arguments: dict, observation: Any) -> str:
        if not self.perception.depth:
            return 'NEEDS_WORK: RGB-only observation contract; depth queries are unavailable. No action executed.'
        from .surface import inspect_surface
        if set(arguments)-{'observation_id','samples','acting_arm','attachment_updates','correspondences'}:
            return 'NEEDS_WORK: inspect_surface accepts only current measurement fields; use query_experience for history. No action executed.'
        if (self.current_observation is None or arguments.get('observation_id')!=self.current_observation['id']
            or observation.extra.get('evidence_id')!=self.current_observation['id']):
            return 'NEEDS_WORK: surface queries require the exact current observation ID; no action executed.'
        try:
            result=inspect_surface(self.current_observation['data'],arguments['samples'])
            if arguments.get('correspondences'):
                from .geometry_relations import planar_correspondence
                result['planar_alignment']=planar_correspondence(result['samples'],arguments['correspondences'])
        except (KeyError,TypeError,ValueError) as error:
            return f'NEEDS_WORK: invalid surface query ({type(error).__name__}); no action executed.'
        ref=self.evidence('surface_inspection',{**result,'before_observation_id':self.current_observation['id'],
            'dependencies':dict(self.dependencies)})
        output={'evidence_id':ref['id'],**surface_readout(result,arguments.get('acting_arm'))}
        return json.dumps(output)

    def source_frontier(self) -> dict:
        return {row['run']:row['sequence'] for row in self.store.db.execute(
            'SELECT run,MAX(sequence) sequence FROM evidence GROUP BY run')}

    def attach_manager(self, client) -> None:
        from .manager import MemoryManager
        if self.manager is not None:
            raise ValueError('Manager already attached')
        self.manager = MemoryManager(self, client)

    def refresh_context(self, messages: list[dict], observation: Any) -> None:
        if self.manager is None:raise RuntimeError('Task preparation manager required')
        if not self.manager.prepared:
            from .contracts import task_contract_id,amendment_cards
            source=self.current_observation
            if source and source['data'].get('instruction'):
                self.task_instruction=source['data']['instruction']
            self.task_specification=self.evidence('task_specification',{
                'instruction':self.task_instruction,
                'public_recipe':self.public_task_recipe,
                'contract_id':task_contract_id(self.context['task'],self.task_instruction,self.public_task_recipe),
                'source_observation_id':source['id'] if source else None})
            self.contract_amendments=[c['id'] for c in amendment_cards(self)]
            self.manager.run('Select concise applicable reusable knowledge and query routes. '
                'Do not expand individual cases or current state into preparation. Empty knowledge is valid.','prepare')
            if not self.manager.prepared:
                raise RuntimeError('Task preparation ended without a committed publication; no primary request permitted')
            if self.pause_after_preparation:
                (self.trace.root/'PAUSE').write_text('Internal input review; no model or physics advance. Remove to continue same episode.\n')
                self.trace.event('preparation_review_pause',{'knowledge':self.manager.prefix_view()})
        prefix=self.manager.prefix_view()
        signature=json.dumps(prefix,sort_keys=True)
        if not hasattr(self,'_prefix_signature'):
            first_rgb=next((i for i,m in enumerate(messages) if isinstance(m.get('content'),list)),len(messages))
            messages.insert(first_rgb,{'role':'user','content':'<KYB_PREPARATION>\n'+json.dumps(prefix,ensure_ascii=False)+'\n</KYB_PREPARATION>'})
        elif signature!=self._prefix_signature:
            messages.append({'role':'user','content':'KNOWLEDGE_REPLACEMENT version='+str(prefix['version'])+
                ' Previous default versions not in this set are no longer current; old history is not erased.'})
        self._prefix_signature=signature
        payload={'current_observation_id':self.current_observation['id'],
            'manager_call_budget':self._manager_call_budget(),
            'measured_finger_joints_m':self.current_observation['data'].get('measured_finger_joints_m',{}),
            'attachment_declarations':self.attachment,
            'qualification':'Prior attachment interpretations need confirmation in current RGB.',
            'knowledge_version':self.manager.version}
        if self.last_receipt:
            data=self.last_receipt['data']
            payload['previous_execution']={'id':self.last_receipt['id'],
                **{k:data.get(k) for k in ('execution','executed_actions','planned_actions','failure_type')}}
        self.trace.event('decision_materials',{**payload,
            'preparation_sha256':hashlib.sha256(json.dumps(prefix,ensure_ascii=False).encode()).hexdigest(),
            'material_scope':self.manager.material_scope()})
        for message in reversed(messages):
            parts=message.get('content')
            if message.get('role')=='user' and isinstance(parts,list) and any(p.get('type')=='image_url' for p in parts):
                parts[0]['text']=parts[0]['text'].split('\n<KYB_CONTEXT>',1)[0]+'\n<KYB_CONTEXT>\n'+json.dumps(payload,ensure_ascii=False)+'\n</KYB_CONTEXT>'
                return
        raise ValueError('No RGB observation')

    def query_experience(self, arguments: dict) -> str:
        try:
            query={k:v for k,v in arguments.items() if k!='attachment_updates'}
            if not query.get('continuation'):query={'view':'statement',**query}
            answer,_=read_experience(self,**query)
            if answer.get('records') and not arguments.get('continuation'):
                from .leads import add
                payload={'question':answer['question'],
                    'lookup_handle':answer.get('read_handle'),'candidate_refs':[{k:r[k] for k in ('id','revision','view') if k in r} for r in answer['records']],
                    'qualification':'Candidates delivered; no adoption, action outcome or causal benefit established'}
                ref=self.evidence('lookup_lead',payload)
                with self.store.db:add(self.store,ref,payload)
            return json.dumps(answer,ensure_ascii=False)
        except ValueError as error:
            return json.dumps({'status':'NEEDS_WORK','reason':str(error),'executed_actions':0})



    def _manager_call_budget(self):
        budget=self.manager.client.budget if self.manager is not None else None
        if budget is None:return None
        view=budget.view()
        return {key:view[key] for key in ('limit','used','remaining')}

    def manage_memory(self, arguments: dict) -> str:
        if self.manager is None:raise RuntimeError('No configured manager')
        if (arguments.get('purpose') not in ('review','learn') or not isinstance(arguments.get('question'),str)
            or not 0<len(arguments['question'])<=1200):
            return 'NEEDS_WORK: provide one question and purpose review or learn; no manager called.'
        conditions=arguments.get('conditions')
        if conditions is not None:
            self.consume({'memory_context':conditions},None)
        if 'episode_outline' in arguments:
            from .episodes import seal
            try:seal(self,arguments['episode_outline'])
            except ValueError as error:return json.dumps({'status':'NEEDS_WORK','reason':str(error),'executed_actions':0})
        if self.episode_ended and arguments['purpose']=='learn':
            from .episodes import initialize
            initialize(self.store)
            row=self.store.db.execute('SELECT outline_ref FROM episode_records WHERE run=?',(self.episode,)).fetchone()
            if row and not row[0]:
                return json.dumps({'status':'NEEDS_WORK','reason':'Before first terminal learning supply episode_outline explicitly ([] is valid); omission does not mean no useful process. No outcome delivered or manager called.','executed_actions':0})
        before=self._manager_call_budget()
        result=self.manager.run(arguments.get('question'),arguments.get('purpose'))
        after=self._manager_call_budget()
        # The complete publication stays in the audit. Policy receives the answer and
        # update handles; selected material is already in the resident preparation.
        answer={key:result[key] for key in ('status','answer','publication','version','knowledge_updates')}
        answer['manager_call_usage']=None if before is None or after is None else {
            'native_calls_this_invocation':after['used']-before['used'],**after}
        return json.dumps(answer,ensure_ascii=False)

    def query_body_history(self, arguments: dict) -> str:
        from .body_index import query
        try:
            result=query(self.store,config=self.body_prior['data'] if self.body_prior else {},
                run=self.episode,frontier=self.source_frontier(),**arguments)
        except (TypeError,ValueError) as error:
            return json.dumps({'status':'NEEDS_WORK','reason':str(error),'executed_actions':0})
        ref=self.trace.event('body_history_query',result)
        if result['records']:
            from .leads import add
            payload={'question':'Body endpoint lookup '+json.dumps(arguments,ensure_ascii=False),
                'lookup_handle':ref['id'],'source_refs':[r['post'] for r in result['records']],
                'qualification':'Numerical candidates delivered; not adopted, current reachability or retention proof'}
            lead=self.evidence('lookup_lead',payload)
            with self.store.db:add(self.store,lead,payload)
        return json.dumps(result,ensure_ascii=False)

    def read_body(self, arguments: dict) -> str:
        if not self.perception.calibrated_camera:
            return 'NEEDS_WORK: RGB-only observation contract; camera projections are unavailable. No action executed.'
        if arguments.get('observation_id')!=self.current_observation['id']:
            return 'NEEDS_WORK: body geometry requires the current observation ID; no motion.'
        body=self.current_observation['data'].get('body_visual_reference',{})
        camera=arguments.get('camera')
        if not isinstance(camera,str) or camera not in body.get('views',{}):
            return 'NEEDS_WORK: select an available current camera.'
        result={'observation_id':self.current_observation['id'], 'camera':camera,
            'projection':body['views'][camera], 'inner_face_windows_flange_m':body.get('inner_face_windows_flange_m'),
            'interpretation':body.get('interpretation'),'occlusion':'unknown'}
        self.trace.event('body_query',result)
        return json.dumps(result,ensure_ascii=False)

    def consume(self, arguments: dict, observation: Any) -> None:
        """Updates concern past observed actions even if the next plan is refused."""
        self.last_decisions = []
        attachments = arguments.get("attachment_updates", [])
        if not isinstance(attachments, list):
            self.last_decisions.append({"accepted": False, "update_type": "attachment",
                                        "reason": "attachment_updates_must_be_list"})
            attachments = []
        for index, change in enumerate(attachments):
            if isinstance(change,dict) and 'evidence' not in change and self.current_observation:
                change={**change,'evidence':[self.current_observation['id']]}
            if (not isinstance(change, dict) or not isinstance(change.get("evidence"), list)
                or not all(isinstance(s, str) for s in change["evidence"])
                or not all(isinstance(change.get(k, "unknown"), str) for k in ("arm", "tool", "state", "grip_region"))):
                self.last_decisions.append({"accepted": False, "update_type": "attachment",
                                            "index": index, "reason": "malformed_attachment_update"})
                continue
            sources = set(change["evidence"])
            valid = (self.current_observation is not None
                     and self.current_observation["id"] in sources
                     and sources <= self.allowed
                     and change.get("arm") in self.attachment
                     and change.get("state") in {"visually_supported", "unknown", "released", "slipped"})
            if not valid:
                self.last_decisions.append({"accepted": False, "update_type": "attachment", "index": index,
                                            "reason": "attachment_requires_current_visual_source"})
                continue
            arm = change["arm"]
            state = {k: change.get(k, "unknown") for k in ("tool", "state", "grip_region")}
            old = self.attachment[arm]
            lost = {"slipped", "released"}
            loss_event = state["state"] in lost and old["state"] not in lost
            regained = state["state"] == "visually_supported" and old["state"] in lost
            physical = (loss_event or regained or
                        state["tool"] != old["tool"] or
                        change.get("binding_event") in {"regrasp", "contact_region_changed"})
            if physical and self.store.attachment_event_at(self.current_observation["id"], arm):
                self.last_decisions.append({"accepted": False, "update_type": "attachment", "index": index,
                                            "reason": "duplicate_physical_interpretation_of_same_observation"})
                continue
            if state != old or physical or state["state"] == "visually_supported":
                payload = {
                    "arm": arm, "old": self.attachment[arm], "new": state,
                    "epistemic": "visually_inferred", "rigidity": "unknown", "evidence": sorted(sources),
                    "observation_id": self.current_observation["id"],
                    "binding_changed": physical, "binding_event": change.get("binding_event", "not_declared")}
                ref = self.trace.event("attachment_change" if physical else "attachment_observation", payload)
                if physical:
                    self.dependencies[f"attachment:{arm}"] = ref["id"]
                self.attachment[arm] = state
                capsule = None
                if physical and loss_event and state["state"] == "slipped" and self.last_receipt is not None:
                    receipt = self.last_receipt["data"]
                    post = self.current_observation
                    if post["data"].get("after_receipt") == self.last_receipt["id"]:
                        before_id = receipt["before_observation"]
                        before = self.store.evidence_payload(before_id)
                        view_names = {"cam_head", f"cam_{arm}_wrist"}
                        images = []
                        for oid, data in [(before_id, before), (post["id"], post["data"])]:
                            images += [{"observation_id": oid, "camera": name,
                                "policy_step": data.get("policy_step"), "env_step": data.get("env_step"), **image}
                                for name, image in data.get("images", {}).items() if name in view_names]
                        capsule = {"before_observation_id": before_id, "receipt_id": self.last_receipt["id"],
                            "post_observation_id": post["id"], "attachment_event_id": ref["id"], "arm": arm,
                            "before_attachment": old, "after_attachment": state,
                            "execution_dependencies": receipt["dependencies"], "post_dependencies": dict(self.dependencies),
                            "execution": receipt, "before_state": before.get("state", {}),
                            "post_state": post["data"].get("state", {}),
                            "before_finger_joints_m": before.get("measured_finger_joints_m", {}),
                            "post_finger_joints_m": post["data"].get("measured_finger_joints_m", {}),
                            "evidence_images": images,
                            "evidence": [before_id, self.last_receipt["id"], post["id"], ref["id"]],
                            "qualification": "visually_inferred_state_transition_after_execution_not_single_step_causality"}
                self.store.publish_attachment(ref, payload, observation_id=self.current_observation["id"],
                                              dependencies=self.dependencies, capsule=capsule)
                self.allowed.add(ref["id"])
                self.cutoff[self.episode] = ref["sequence"]
            self.last_decisions.append({"accepted": True, "update_type": "attachment", "index": index,
                "reason": "current_visual_inference_not_rigidity_measurement", "arm": arm})
        if self.last_decisions:
            self.trace.event("memory_update", {"decisions": self.last_decisions,
                                             "knowledge_revision": self.store.revision})
        if 'memory_context' in arguments:
            declared = arguments['memory_context']
            valid = (isinstance(declared, dict) and len(declared) <= 24
                and all(isinstance(k,str) and isinstance(v,str) and len(k)<=80 and len(v)<=300
                        and k not in {'task','phase','episode'} for k,v in declared.items()))
            if valid:
                # Replace, rather than silently carrying old contact conditions forward.
                self.context = {k:v for k,v in self.context.items() if k in {'task','phase'}} | declared
            self.last_decisions.append({'accepted':valid,'update_type':'memory_context',
                'reason':'current_model_declaration_not_physical_verification' if valid else 'invalid_physical_context'})
        if isinstance(arguments.get("phase"), str):
            self.context["phase"] = arguments["phase"][:120]
        if "predict" in arguments:
            self.trace.event("prediction", {
                "about": "next_action_not_observed", "prediction": arguments["predict"],
                "targets": arguments.get("targets"), "observation": self.current_observation["id"]})

    def invalidate_opened_grips(self, action: dict, receipt_id: str) -> None:
        if self.current_observation is None:
            return
        before = self.current_observation['data']
        for arm in self.attachment:
            value=action.get(f'{arm}_ee_joint_state')
            previous=before.get('state',{}).get(f'{arm}_ee_joint_state')
            if value is not None and previous and float(value[0])>float(previous[0]) and self.attachment[arm]['state']=='visually_supported':
                self.dependencies[f'attachment:{arm}'] = receipt_id
                self.attachment[arm] = {'tool':'unknown','state':'unknown',
                    'reason':'opening_command_executed;release_not_visually_confirmed'}

    def instructions(self) -> str:
        text = (
            'Native session history maintains episode continuity. Default preparation is short reusable '
            'knowledge, not an action script. Current RGB and measured state take precedence. '
            'No database patch or progress boilerplate is required for an ordinary action. '
            'Before committing an uncertain interaction, query if historical body relations or responses could '
            'change the action, chunk size or next observation. Do not repeat a query for facts already supplied. '
            'query_experience defaults to knowledge; level=case drills into finer conditional history, even '
            'when knowledge is empty or the topic is absent from the short index. '
            'A supplied episode_candidate is an unreviewed whole-process interpretation: reuse the grasp/relative-pose '
            'relationships and downstream transformations only after checking current conditions; query level=episode '
            'for a specific missing process. Whole-episode success does not certify each phase or a pruned sequence. '
            'level=contract searches applicable explicit operator amendments only; no match does not mean experience has no answer. '
            'Before requesting clarification, use supplied relevant candidates or query the knowledge/cases/episode evidence '
            'needed to test the competing interpretations. Historical evidence may support a provisional working interpretation '
            'without certifying an operator rule or general law. Respect clarification scope; ask if conflicting evidence '
            'or a genuinely missing goal condition still prevents choosing the intended outcome. '
            'When only the public task contract is genuinely ambiguous or omits information needed to define the goal, '
            'use request_task_clarification with exact public quotes and competing interpretations. Do not ask for '
            'perception, motor strategy, easier success or unavailable evaluator signals. The scene waits for the operator. '
            'query_body_history retrieves measured body endpoints with the same mount/TCP/frame contract; it is not a reachability guarantee. '
            'Use inspect_surface for current selected pixels and read_body '
            'for current body projections. Historical coordinates/contact labels are not current state. '
            'Use manage_memory only for one unresolved comparison, changed conditions or useful batch learning. '
            'It finds evidence; you need not list source IDs. Every primary round, including reads and manager triggers, '
            'counts toward the primary allowance. manage_memory can spend multiple manager calls; its usage and '
            'the current manager_call_budget share one episode ledger, including failed turns. Null means unknown. '
            'Weigh that cost against the information needed for the next choice; current diagnostic tests need not wait for historical certainty. '
            'Use hold for timed dynamics rather than repeated unchanged EEF requests. '
            'Choose execute_max_s when contact or retention uncertainty requires an earlier observation; '
            'the prefix stops before target arrival and must be replanned from measured state. '
            'A small Cartesian rotation can have a long joint path; use plan preview or cost limits when consequential. '
            'Choose efficient large moves through already-understood free space, including simultaneous xyz '
            'or both arms when suitable. Retain observation at uncertain contact, grasp and precision boundaries. '
            'Distinguish position and orientation residuals using corresponding object/target landmarks; '
            'a changed translation with unchanged angular error is still a response. A sampled top/interior '
            'point does not certify an edge or loaded contact. '
            'When outcomes contradict predictions or progress stalls, identify the assumption\ngoverning your next decision. Separate observations from explanations, and required\noutcomes from chosen means. Memory provides evidence, not permission. A success\nclaim needs sufficient evidence; a justified exploratory test need not wait for\ncertainty. Choose an informative observation or action that distinguishes plausible\ncauses; prefer reversible tests. Predict what result would change your next choice,\nthen revise or abandon the assumption accordingly. Keep this check brief; do not\nrequire an extra query, manager call or long narrative each step. '
            'Model-call allowance and remaining environment steps are separate. Reads, management and rejected plans '
            'advance no simulation; executed moves and hold consume its step budget. Repeated waiting or motion '
            'needs a reason to expect new observable feedback or a task/physical dwell requirement; '
            'waiting cannot expose an unavailable completion signal. '
            'Budget physical steps for the remaining task and any required home return. '
            'You alone choose all EEF targets. Completed execution and finger closure do not prove contact or holding.'
        )
        if not self.perception.depth:
            text = text.replace('Use inspect_surface for current selected pixels and read_body '
                'for current body projections.', 'Estimate visual relationships from RGB and measured proprioception. '
                'No depth or camera calibration/projection query is available.')
        return text

    @staticmethod
    def tool_properties() -> dict:
        return {
            'memory_context':{'type':'object','additionalProperties':{'type':'string'},
                'description':'Optional current physical-condition declarations. Missing conditions remain unknown.'},
            'attachment_updates':{'type':'array','items':{'type':'object','properties':{
                'arm':{'type':'string','enum':['left','right']},'tool':{'type':'string'},
                'state':{'type':'string','enum':['visually_supported','unknown','released','slipped']},
                'grip_region':{'type':'string'},
                'binding_event':{'type':'string','enum':['unchanged','regrasp','contact_region_changed','uncertain']}},
                'required':['arm','tool','state']},
                'description':'Your current RGB interpretation; host binds this observation and records uncertainty.'}}
