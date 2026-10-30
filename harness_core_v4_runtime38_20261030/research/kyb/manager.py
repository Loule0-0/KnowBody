"""Question-scoped memory transactions; no raw-interval or working-packet injection."""
from __future__ import annotations
import base64
import hashlib
import json
import sqlite3
import uuid
from pathlib import Path
from .knowledge import function, patch_schema, knowledge_schema, query_properties, read_experience
from .state import evidence_state, compact_body_state
from . import library, episode_index, readout
from .action_digest import digest
from .instances import source_physics_dt
from .contracts import amendment_view,current_contract

MANAGER_INSTRUCTIONS='''Organize evidence-grounded body relations and external interaction experience. The primary model alone chooses future EEF targets.
Authority: raw evidence -> conditional cases -> reusable knowledge K. Episode candidates are unreviewed controller interpretations; a terminal boolean labels the complete real trace, not phases or a pruned/spliced route. Neither can support an observed physical claim.
Answer the caller's one question. Begin with supplied complete statements and same-task process candidates; do not reread them. Query knowledge, then cases, then only the missing/disputed raw evidence. Complete K statements permit related-K navigation references only; detail is required to revise/retire K or cite case-detail sources. Navigation, episode_index, body endpoints and measurement pointers grant no RGB/raw read permission. Current unorganized actions need trajectory/segment/evidence reads, not a search for unwritten cases. Current coordinates and holding are not preparation.
Prefer positive decision information when evidence supports it: grasp region + relative pose -> feasible observed transformation/transport -> object/tool interaction -> release/result. Ask how an early grasp enabled or obstructed later work. Preserve conditions, response, feedback boundary and unknowns. Independent useful counterexamples remain valid; failure of A does not establish success of untested B. A single uncontrolled result cannot establish causality, preference, typical gain or zero cost.
Body means robot/tool geometry, mounting, frames and camera/action relations; an object handling route is external even when both arms participate. Use query_body_history to discover historical measured endpoint precedents. For current or source-time numerical comparisons, read the exact observations/receipts with body=ee/full; use the configured episode-start source for return comparisons. Nearby historical poses are not that baseline. Domain/family and actual role_facets navigate; object/part/operation labels are source-backed navigation, not success certificates. Missing old labels remain unknown.
A case records a scoped observation. K answers a reusable decision question with conditional response, observable disconfirmation and concrete use; it may span stages when their relation changes that decision. Keep instance coordinates and chronology in cases/episode sources. No sample pose tables in K; prefer at most two essential parameters and five material IF conditions. Do not union case fields, turn missing evidence into prerequisites, or drop conditions to fit. Distinguish what observations establish from action prerequisites: a successful sequence is a scoped candidate, not proof each step was necessary. Necessity or causality needs relevant contrasting evidence under comparable conditions; preserve positive response information and unresolved alternatives. Inspect decision_bytes/readout_warning; split only when useful. Missing material premises mean related, not support.
Before adding K, compare supplied/same-family K. Repeated unchanged response or assessment boundaries belong in an existing applicable K via append_refs {id,refs}; a shared boundary may stand separately, but never merge incompatible grasps/scopes or force an abstraction. Host preserves old fields/sources. Exact duplicate is no-op; changed source revision/relation/reason or meaning requires full revise. Preserve counters; reclassification requires explicit scope resolution. Related case/K refs are navigation only: no support, pins or parameter values; cite underlying read evidence/cases to use a value. Historical warnings remain visible, not inherited as the current claim's qualification. Source counts/chronology are not causal replication or stage order; trace gaps do not inherit success.
Dependency names (body/cameras/depth/execution/planner/attachment:arm), never hashes. Choose only dependencies needed for use. Host resolves requested static pins from exact historical sources; at least one support case must declare each key. Missing/conflicting context cannot be repaired using current pins. Dynamic attachment is never filled from a whole-episode context; episode-local bindings need not be mechanically inherited into general interaction K.
New cases need local=c1,c2. Knowledge may use {type:case,local:c1}; never invent IDs/revisions. Existing IDs resolve to actually read versions. Preserving inherited refs grants no raw/case read permission. Full revise supplies domain/family/question/text/conditions/strength/use/unknowns/params/depends_on/refs; byte limits are question600/text1800/use900, answer1800. Case text limit1800 characters. New/targeted cases need a knowledge link or keep/defer with a concrete reason; these are alternatives, including when the link is only related. Never also keep/defer a linked case. No quota to publish.
Use receipt action_digest to separate requested controls, executed coverage and measured world changes. EEF motion is not object motion. New visual cases require cited RGB actually viewed this transaction; read only necessary cameras. telemetry_inferred is only body/intrinsic_candidate/robot_part configuration or relative_geometry, non-image frame, with exact EEF sources delivered body=ee/full; never contact or holding. New interaction cases need complete before/receipt/post with exact witness endpoints and observed effects. Binding changes do not erase an interval or certify causality.
read_evidence receipts/measurements expands exact endpoints. Both selector forms return at most6 RGB and24KiB result JSON (separate RGB-reference text is not included). Continue remaining_selectors without changing body/cameras. A single oversized raw record uses immutable detail pages; finish its continuation before claiming the source was read. Receipt action_digest already supplies planned cost, executed coverage, requests and measured endpoints; read a large plan action array only for a per-command question. Use the measurement directory to locate samples by time and original labels, including samples without a following action. Labels do not certify object identity or replace action/context reads. Raw ids remain an explicit read choice. Mark processed only complete actually-read intervals, including useful successful/neutral portions. read_trajectory/read_segments are navigation, not evidence.
Publish independent atomic groups: cases and their dependent K/disposition/processed operations stay together. Several cases affecting the same existing K belong in one group with one combined knowledge operation. Use append_refs only for unchanged meaning; changed meaning or conditions require full revise and its detail permissions. Separate groups cannot write the same target. Inspect committed_groups/issues; repair only rejected groups at returned version, never replay committed groups. Return a concise answer with evidence limits/unknowns, not future EEF targets. Insufficient evidence and no writes are valid.
'''

MODE_INSTRUCTIONS={
 'prepare':'Read complete statement views to select up to 12 useful knowledge IDs, most useful first; at most two are inlined. No writes. '
   'Only default_eligible entries belong in default_knowledge. Optionally select up to two already-read, supported, active nonhypothesis ineligible IDs as history_refs: leftover-space navigation only, not extra claims. Do not search merely to fill navigation slots. '
   'Use decision_packet_forecast to judge the actual byte allowance and supplied candidates that fit. Further discovery should change the useful selection, not fill an unseen list. '
   'Selection does not require full provenance; use detail only for a specific missing source question. Do not reread already complete statements. '
   'Choose task preparation, not a script, old coordinates, current holding or scene state. The navigation reports visible knowledge counts; do not search empty knowledge families. Empty knowledge is valid; case routes remain available.',
 'review':'Answer the one consequential uncertainty. Start with qualified statement views; detail is for a specific missing provenance question or an intended write requiring full-source permission. '
   'A complete statement already supports conditional reading; do not finish its optional detail continuation just because it exists. '
   'Knowledge first, then relevant cases, raw only for a missing or disputed detail. '
   'Use the shared episode manager allowance to resolve the caller\'s next decision. Once the evidence supports a bounded answer, '
   'or establishes what the available record cannot distinguish, publish that answer and the missing observation that could change it. '
   'Organize already-read evidence when useful; do not extend raw retrieval solely to complete a historical route. '
   'An insufficient-evidence answer with no writes is valid.',
 'learn':'Organize each targeted case and any new cases. Compare same-family candidates supplied or query them. '
   'Prefer a small correct synthesis over trial narratives. Preserve what an actual action or measurement distinguished, '
   'including scoped counterexamples and unresolved variables from earlier diagnostic questions, rather than only the last retention/release result. '
   'Observed response contrasts may remain observations without a causal mechanism. Untested explanations or variants remain hypotheses, not action preferences. '
   'Ask whether the statement changes a future decision or measurement boundary; keep instance coordinates in cases. '
   'When a complete, actually observed operation has a supported response, lead with identifiable IF conditions -> the tried operation as a conditional candidate -> visible feedback or disconfirmation, even when its individual contributions are not isolated; leave future EEF targets to the primary model. '
   'Preserve useful cross-stage context in the applicable cases/episode. Include a grasp -> transformation -> interaction -> release candidate in K only when that relation answers its decision question; retain conditions, gaps and unknowns rather than retelling each trial. '
   'Preserve single-trial scope, changed-condition comparisons and unknown mechanism/repeatability in conditions and unknowns, without promoting the candidate to a preference or guarantee, proposing untried variants, or changing its epistemic strength. '
   'Failure of A does not establish success of untested B. Compare subsequent responses with previously delivered knowledge where relevant; delivery alone is not adoption. '
   'No obligation to create knowledge or reread raw pixels already represented by cases.'}

def manager_tools():
    from .body_index import tool
    return [tool(),function('search_experience','Read knowledge or explicitly selected cases via structured navigation. '
        'Writing a new case reference requires complete view=detail in this transaction, including its pages; statement/preview only supports reading.',query_properties(),[]),
        function('read_trajectory','Index complete interaction intervals, not raw data or successful outcomes. Continue with next_after. '
            'For interval reads use an explicit through bound and include_processed=true to inspect previously organized intervals. '
            'For directory reads omit include_processed and lead; these modes are mutually exclusive.',
            {'run':{'type':'string'},'after':{'type':'integer','minimum':0},
             'through':{'type':'integer','minimum':0},'include_processed':{'type':'boolean','default':False},
             'directory':{'type':'string','enum':['questions','leads','motion','measurements'],'description':'Page question/lead pointers, motion requests/plans/checks, or measurement pointers with original labels/cameras (depth-enabled profiles only). Measurements do not require a following action. Navigation only, no XYZ or read permission; continue next_after.'},
             'lead':{'type':'string','description':'Optional exact learning-lead ID from review_followups: return its complete interpretation, not physical evidence or read permission.'}},['run','after']),
        function('read_segments','Find one bounded historical motion window by question and measured entry configuration. '
            'Navigation only; read its exact before/receipt/post before writing. Use run to isolate a known episode.',
            {'question':{'type':'string','maxLength':1200},
             'run':{'type':'string','description':'Optional exact source run from the episode/index pointers.'},
             'max_receipts':{'type':'integer','minimum':1,'maximum':5,'default':1}},['question']),
        function('read_evidence','Read exact sources. Choose ids OR receipts/measurements OR continuation alone. Receipt groups expand exact before/receipt/post; measurements include their observation. Both selector forms return at most6 RGB and24576 bytes of result JSON, excluding separate RGB-reference text. Complete groups/records are delivered first; retry remaining_selectors unchanged. A single oversized ID is paged losslessly; incomplete pages grant no raw/telemetry read permission. A receipt/measurement group too large alone returns its expanded IDs for explicit reading. Navigation grants no read permission.',
            {'continuation':{'type':'string','description':'Exact manager-owned raw detail-page token; use alone.'},
             'ids':{'type':'array','items':{'type':'string'},'maxItems':6},
             'receipts':{'type':'array','items':{'type':'string'},'maxItems':2},
             'measurements':{'type':'array','items':{'type':'string'},'maxItems':2},
             'cameras':{'type':'array','items':{'type':'string','enum':['cam_head','cam_left_wrist','cam_right_wrist']},'default':[]},
             'body':{'type':'string','enum':['none','ee','full'],'default':'none','description':'Observation ee adds compact EEF state and enables measured endpoint action digests; full additionally includes body_visual_reference and depth_snapshots. Surface none preserves sensor samples; ee adds flange/grasp coordinates, omitting pad details. Choose full only for needed pad/camera/depth reference fields; it can greatly enlarge a pack. Cite only delivered fields.'}},[]),
        function('publish_materials','Commit explicit independent groups. Each group is atomic, with c/k handles unique throughout this management request. '
            'Put dependent cases and all writes to the same knowledge target in one group. '
            'Valid unrelated groups survive a rejected group. Repair only rejected groups at the returned version. '
            'Default selection is separate and atomic; failure retains the old selection.',
            {'expected_version':{'type':'integer'},'answer':{'type':'string','maxLength':1800,'description':'At most1800 UTF-8 bytes; include limits/unknowns, no future EEF targets.'},
             'default_knowledge':{'type':'array','maxItems':12,'items':{'type':'string'},'description':'Ranked eligible knowledge pool; unchanged packing selects at most two complete cards. Ineligible choices reject the selection component.'},
             'history_refs':{'type':'array','maxItems':2,'items':{'type':'string'},'description':'Optional explicit IDs of active, supported, nonhypothesis but default-ineligible knowledge fully read this transaction. Host pins read revisions. Navigation only, uses leftover packet space; may be omitted. Empty clears, omission retains prior selection. Never put a new local handle here.'},
             'groups':{'type':'array','maxItems':8,'description':'Independent writes. Cases affecting the same existing knowledge belong in one group with one appropriate knowledge operation per target. Normal append/revise/retire permissions still apply. Repair rejected groups only; do not replay committed cases.','items':{'type':'object','properties':{
                 'memory_patch':patch_schema(),'knowledge_ops':knowledge_schema(),
                 'dispositions':{'type':'array','items':{'type':'object','properties':{
                     'case':{'type':'string'},'action':{'type':'string','enum':['keep','defer']},'reason':{'type':'string'}},
                     'required':['case','action','reason'],'additionalProperties':False}},
                 'processed':{'type':'array','items':{'type':'object','properties':{
                     'receipt_id':{'type':'string'},'disposition':{'type':'string','enum':['case','knowledge','no_lesson']}},
                     'required':['receipt_id','disposition'],'additionalProperties':False}}},'additionalProperties':False}}},
            ['expected_version','answer'])]


class SelectionRejected(ValueError):
    def __init__(self,details):
        self.details=details
        super().__init__('Material selection rejected')


class MemoryManager:
    def __init__(self,session,client):
        self.session,self.client=session,client
        library.initialize(session.store)
        self.version=0;self.prepared=False;self.default_ids=[];self.default_versions={};self.publication=None
        self.history_ids=[];self.history_versions={};self.read_knowledge_versions={}
        self.previous_result=None
        self._navigation_cache=None
        self.retrieval_owner='manager:'+uuid.uuid4().hex

    def remember_completed_result(self, publication, completion):
        """Retain one completed interpretation, never a source/read permission."""
        pub=publication['payload'];done=completion['payload'];result=done.get('result',{})
        if (publication.get('kind')!='manager_group_publication' or pub.get('status')!='PASS'
                or pub.get('issues') or completion.get('kind')!='manager_tool_result'
                or done.get('tool')!='publish_materials' or result.get('status')!='PASS'
                or result.get('issues') or result.get('publication')!=publication['id']
                or result.get('version')!=pub.get('version')
                or completion['sequence']<=publication['sequence']
                or completion['id'].rsplit(':',1)[0]!=publication['id'].rsplit(':',1)[0]):
            return False
        answer=pub['answer'];question=pub['question']
        if not isinstance(answer,str) or len(answer.encode())>1800 or not isinstance(question,str):
            raise ValueError('Invalid completed management interpretation')
        title=question.encode()[:240].decode('utf-8',errors='ignore')
        record={'logical_episode':self.session.episode,'perception_profile':self.session.perception.name,
            'publication':publication['id'],'completion':completion['id'],'purpose':pub['purpose'],
            'question_title':title,'question_truncated':title!=question,'answer':answer,
            'request_read_view':{'case_revision':pub['case_revision_at_request'],
                                 'knowledge_revision':pub['knowledge_revision_at_request']},
            'publication_material_scope':pub['delivery_manifest']['render_scope'],
            'qualification':'Previous completed manager interpretation, not current physical facts or source/write permission. '
                'A search finding covers its returned material, not proof of library-wide absence. '
                'Read-view revisions do not cover later writes in that transaction; see the publication for those. '
                'Current revisions are a comparison, not proof that state or applicability is unchanged. Recheck when consequential.'}
        # One full <=1800-byte answer plus a bounded question label and fixed metadata.
        # No growing transcript, truncation of the answer, or negative-search cache.
        from .readout import size
        if size(record)+128>16384:raise ValueError('Completed management interpretation exceeds its bounded record')
        self.previous_result=record
        return True

    def _available(self,source):
        row=self.session.store.db.execute('SELECT run,sequence FROM evidence WHERE id=?',(source,)).fetchone()
        return row is not None and row['sequence']<=self.frontier.get(row['run'],-1)

    def _record(self,source,*,body='none',delivered_ids=()):
        row=self.session.store.db.execute('SELECT * FROM evidence WHERE id=?',(source,)).fetchone()
        if row is None:raise ValueError(f'Unregistered evidence ID: {source}')
        if not self._available(source):raise ValueError(f'Source beyond request frontier: {source}')
        if self.purpose=='prepare' and row['run']==self.session.episode and row['kind'] not in ('known_config','task_specification'):
            raise ValueError('Preparation cannot inspect current scene')
        data=self.session.store.evidence_payload(source)
        self.session.perception.check_payload(row['kind'],data)
        readout={}
        if row['kind']=='observation':
            state=evidence_state(data)
            view={'grasp_point_dimensions':state['grasp_point_dimensions'],
                'measured_finger_joints_m':data.get('measured_finger_joints_m'),
                'policy_step':data.get('policy_step'),'env_step':data.get('env_step'),'remaining_steps':data.get('remaining_steps'),
                'after_receipt':data.get('after_receipt'),'available_cameras':list(data.get('images',{}))}
            if body in ('ee','full'):view['eef']=compact_body_state(data)
            if body=='full':view.update(body_visual_reference=data.get('body_visual_reference'),depth_snapshots=data.get('depth_snapshots'))
        elif row['kind']=='receipt':
            view={k:data.get(k) for k in ('before_observation','execution','executed_actions','planned_actions','failure_type')}
            post=self.session.store.db.execute("SELECT id FROM evidence WHERE run=? AND kind='observation' AND json_extract(payload,'$.after_receipt')=? ORDER BY sequence LIMIT 1",(row['run'],source)).fetchone()
            if post and self._available(post['id']):
                view['post_observation_id']=post['id']
            decision=self.session.store.db.execute("SELECT id,payload FROM evidence WHERE run=? AND kind='policy_decision' "
                "AND sequence<? AND (json_extract(payload,'$.call_id')=? OR json_extract(payload,'$.observation_id')=?) "
                "ORDER BY sequence DESC LIMIT 1",(row['run'],row['sequence'],data.get('call_id'),data.get('before_observation'))).fetchone()
            request=json.loads(decision['payload']).get('arguments',{}) if decision else None
            endpoints={data.get('before_observation'),post['id'] if post else None}
            can_measure=(body in ('ee','full') and None not in endpoints and
                all(self._available(identifier) for identifier in endpoints) and endpoints <= (set(delivered_ids)|self.seen_telemetry))
            view['action_digest']=digest(data,
                before=self.session.store.evidence_payload(data['before_observation']) if can_measure else None,
                post=self.session.store.evidence_payload(post['id']) if can_measure else None,
                physics_dt_s=source_physics_dt(self.session.store.db,row['run']),request=request)
            view['action_digest']['source_refs']={'before':data.get('before_observation'),
                'receipt':source,'post':post['id'] if post and self._available(post['id']) else None,
                'request':decision['id'] if decision else None}
            if not can_measure:view['action_digest']['measurement_read_required']='Read before and post with body=ee/full.'
        elif row['kind']=='surface_inspection':
            from .context import surface_evidence_view
            view,omitted=surface_evidence_view(data,body)
            readout={'body_view':body,'omitted_fields':omitted,
                'readout_qualification':'Use only the delivered fields. body=full reads omitted pad geometry; '
                    'a source ID does not mean every original field was delivered. Source guards do not certify textual entailment.'}
        elif row['kind']=='known_config' and body!='full':
            view={k:data.get(k) for k in ('action_spec','coordinate_contract','episode_start_reference')}
        elif row['kind']=='episode_chain':
            view={'summary':data['summary'],'source_run':data['run'],
                'qualification':'Chain navigation only. Use read_trajectory for bounded receipts and read_evidence for exact physical sources; per-tick trace refs are not decision input.'}
        else:view=data
        return {'id':source,'kind':row['kind'],'sequence':row['sequence'],'data':view,**readout}

    def _intervals(self,run,after=0,*,limit=6,recent=False,unprocessed=True,through=None):
        if run not in self.frontier or type(after) is not int or after<0:raise ValueError('Available run and nonnegative cursor required')
        if through is not None and (type(through) is not int or through<after or through>self.frontier[run]):
            raise ValueError('Trajectory upper bound must lie between after and the request frontier')
        end=self.frontier[run] if through is None else through
        if self.purpose=='prepare' and run==self.session.episode:raise ValueError('Preparation reads historical episodes only')
        sql="SELECT e.* FROM evidence e LEFT JOIN processed_intervals p ON p.receipt_id=e.id WHERE e.run=? AND e.kind='receipt' AND e.sequence>? AND e.sequence<=?"
        if unprocessed:sql+=' AND p.receipt_id IS NULL'
        sql+=' ORDER BY e.sequence '+('DESC' if recent else 'ASC')+' LIMIT ?'
        rows=self.session.store.db.execute(sql,(run,after,end,limit+1)).fetchall()
        records=[]
        for row in rows[:limit]:
            data=json.loads(row['payload'])
            post=self.session.store.db.execute("SELECT id,sequence FROM evidence WHERE run=? AND kind='observation' AND json_extract(payload,'$.after_receipt')=? ORDER BY sequence LIMIT 1",(run,row['id'])).fetchone()
            if not post or post['sequence']>end or not self._available(post['id']):continue
            decision=self.session.store.db.execute("SELECT id,payload FROM evidence WHERE run=? AND kind='policy_decision' AND json_extract(payload,'$.observation_id')=? AND sequence<? ORDER BY sequence DESC LIMIT 1",(run,data['before_observation'],row['sequence'])).fetchone()
            intent=json.loads(decision['payload']).get('arguments',{}).get('note','') if decision else ''
            measurements=self.session.store.db.execute("SELECT id,kind FROM evidence WHERE run=? AND kind='surface_inspection' "
                "AND json_extract(payload,'$.before_observation_id')=? "
                "AND sequence>(SELECT sequence FROM evidence WHERE id=?) AND sequence<? AND sequence<=? ORDER BY sequence",
                (run,data['before_observation'],data['before_observation'],row['sequence'],self.frontier[run])).fetchall()
            records.append({'before':data['before_observation'],'receipt':row['id'],'post':post['id'],
                'sequence':row['sequence'],'execution':data.get('execution'),'executed_actions':data.get('executed_actions'),
                'measurement_refs':[{'id':m['id'],'kind':m['kind'],'observation_id':data['before_observation']} for m in measurements],
                'intent':intent if len(intent)<=220 else {'ref':decision['id'],'omitted_long_intention':True}})
        return {'run':run,'through':end,'include_processed':not unprocessed,'intervals':records,'next_after':rows[limit-1]['sequence'] if len(rows)>limit else None,
            'qualification':'Index only; not read sources, certified object outcomes or processed coverage.'}

    def _search(self,args):
        answer,items=read_experience(self.session,**args,frontier=self.frontier,revision=self.revision,knowledge_revision=self.knowledge_revision,version_snapshot={'cases':getattr(self,'own_cases',{}),'knowledge':getattr(self,'own_knowledge',{})},reader_owner=self.retrieval_owner)
        if self.purpose=='prepare':
            filtered=[]
            for record in answer['records']:
                refs=record.get('drill_down',{}).get('evidence',[])
                if any(source.startswith(self.session.episode+':') for source in refs):continue
                filtered.append(record)
            answer['records']=filtered
        return answer

    def _segments(self,args):
        if set(args)-{'question','max_receipts','run'} or not isinstance(args.get('question'),str) or not 0<len(args['question'])<=1200:
            raise ValueError('One bounded segment question required')
        if 'run' in args and (not isinstance(args['run'],str) or args['run'] not in self.frontier):
            raise ValueError('Requested segment run must lie within this request frontier')
        if self.purpose=='prepare' and args.get('run')==self.session.episode:
            raise ValueError('Preparation reads historical episodes only')
        result=self.session.store.read_segments(args['question'],context=self.session.context,
            dependencies=self.request_dependencies,current_observation=(self.session.current_observation or {}).get('data',{})
                if self.purpose!='prepare' else {},episode=self.session.episode,evidence_cutoff=self.frontier,
            current_history_before=(self.session.current_observation or {}).get('sequence') if self.purpose!='prepare' else None,
            at_revision=self.revision,max_receipts=args.get('max_receipts',1),run=args.get('run'))
        windows=[]
        for segment in result['episodic_segments']:
            summary=segment['summary']
            windows.append({'id':segment['id'],'selection':segment['selection'],
                'entry_joint_distance_rad_l2':segment['entry_joint_distance_rad_l2'],
                'before':summary['before_observation_id'],'post':summary['post_observation_id'],
                'receipts':summary['whole_window_receipt_ids'],
                'actual_native_decision_cost':summary['actual_native_decision_cost'],
                'read_evidence':segment['whole_window_intervals']})
        return {'motion_windows':windows,'retrieval_mode':result['retrieval_mode'],
            'qualification':'Navigation only; no source read, image-view or interval-processing permission granted. '
                            'Lexical/joint proximity does not establish applicability. Read exact evidence with body=ee as needed.'}

    def _episode_index(self):
        source=self.session.store.db.execute("SELECT path FROM evidence WHERE run=? AND kind='observation' ORDER BY sequence DESC LIMIT 1",
            (self.session.episode,)).fetchone()
        if source is None:return None
        return episode_index.readout(episode_index.build(Path(source['path']).parent.parent,
            through_sequence=self.frontier[self.session.episode]),completed=self.session.episode_ended)

    def _admit(self,result):
        for record in result.get('records',[]):
            if record.get('view') in ('detail_page','statement_page'):
                record_view=record['view'].removesuffix('_page')
                digest=record['content_sha256']
                if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
                    raise ValueError('Invalid immutable page content hash')
                key=(record['layer'],record['id'],record['revision'],record_view,digest)
                if not hasattr(self,'read_pages'):self.read_pages={}
                if not hasattr(self,'read_page_lengths'):self.read_page_lengths={}
                total=record['total_characters'];offset=record['offset'];fragment=record['fragment']
                if type(total) is not int or type(offset) is not int or not isinstance(fragment,str) or not 0<=offset<=offset+len(fragment)<=total:
                    raise ValueError('Invalid immutable page bounds')
                if key in self.read_page_lengths and self.read_page_lengths[key]!=total:
                    raise ValueError('Conflicting immutable page length')
                self.read_page_lengths[key]=total
                pages=self.read_pages.setdefault(key,{})
                merged=[]
                for start,part in sorted([*pages.items(),(offset,fragment)],key=lambda item:item[0]):
                    if not merged or start>merged[-1][0]+len(merged[-1][1]):
                        merged.append((start,part));continue
                    prior_start,prior=merged[-1];prior_end=prior_start+len(prior)
                    overlap=min(prior_end,start+len(part))-start
                    if prior[start-prior_start:start-prior_start+overlap]!=part[:overlap]:
                        raise ValueError('Conflicting immutable page fragment')
                    if start+len(part)>prior_end:merged[-1]=(prior_start,prior+part[prior_end-start:])
                pages.clear();pages.update(merged)
                end=0;parts=[]
                for offset,fragment in sorted(pages.items()):
                    if offset!=end:break
                    parts.append(fragment);end+=len(fragment)
                if end!=record['total_characters']:continue
                raw=''.join(parts)
                if hashlib.sha256(raw.encode()).hexdigest()!=digest:raise ValueError('Immutable page content hash mismatch')
                record=json.loads(raw)
                if (record.get('layer','knowledge'),record.get('id'),record.get('revision'),record.get('view'))!=key[:4]:
                    raise ValueError('Immutable page identity or view mismatch')
                if record_view=='statement' and (record.get('statement_complete') is not True or record.get('complete_record') is not False):
                    raise ValueError('Statement pages cannot grant full-detail permission')
                if record_view=='detail' and record.get('complete_record') is not True:
                    raise ValueError('Detail pages must reconstruct a complete record')
            if record.get('layer') in ('episode_candidate','public_contract','public_contract_directory'):continue
            if record.get('layer')!='case' and record.get('strength') and record.get('drill_down') is not None and (record.get('statement_complete') or record.get('complete_record')):
                self.read_knowledge_versions[record['id']]=max(
                    self.read_knowledge_versions.get(record['id'],-1),record['revision'])
            if record.get('statement_complete') and record.get('layer')!='case':self.selectable_knowledge[record['id']]=record['revision']
            if record.get('view') in ('statement','preview'):continue
            if 'kind' in record and 'data' in record:
                if record.get('layer')=='evidence' and not (record.get('view')=='detail' and record.get('complete_record') is True):
                    raise ValueError('Raw evidence pages must reconstruct complete detail')
                self.allowed.add(record['id'])
                eef=record['data'].get('eef',{})
                if record.get('layer')=='evidence' and record['kind']=='observation' and eef.get('coordinate_contract_id') and eef.get('arms'):
                    self.seen_telemetry.add(record['id'])
            elif record.get('layer')=='case' and record.get('complete_record') is True:self.visible_revisions[record['id']]=record['revision']
            elif 'strength' in record and 'drill_down' in record and record.get('complete_record') is True:self.visible_knowledge[record['id']]=record['revision']

    def _evidence_identity(self, source):
        row=self.session.store.db.execute('SELECT * FROM evidence WHERE id=?',(source,)).fetchone()
        if row is None or not self._available(source):raise ValueError('Raw continuation source is unavailable')
        identity={'row':dict(row),'effective_payload':self.session.store.evidence_payload(source)}
        return hashlib.sha256(json.dumps(identity,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

    def _evidence_media(self, records, body, cameras):
        """Validate all requested media before selection; this grants no permission."""
        media={};contracts={}
        for record in records:
            if record['kind']!='observation':continue
            data=self.session.store.evidence_payload(record['id'])
            contract=data.get('canonical_state',{}).get('coordinate_contract')
            if body!='none' and contract and contract['id'] not in self.sent_contracts:
                contracts[record['id']]=contract
            images=[]
            for camera in cameras:
                ref=data.get('images',{}).get(camera)
                if ref is None:raise ValueError('Unknown camera '+camera+'; available: '+','.join(data.get('images',{})))
                raw=Path(ref['path']).read_bytes()
                if hashlib.sha256(raw).hexdigest()!=ref['sha256']:raise ValueError('Historical RGB changed')
                images.append((camera,raw))
            media[record['id']]=images
        return media,contracts

    def _evidence_deliver(self,result,selected,media):
        """Only the selected response can update image/contract/telemetry reads."""
        if readout.size(result)>24576:raise ValueError('Evidence result JSON exceeds24576 bytes')
        images=[(source,camera,raw) for source in selected for camera,raw in media.get(source,[])]
        if len(images)>6:raise ValueError('Evidence response exceeds6 RGB images')
        parts=[]
        for source,camera,raw in images:
            parts.extend([{'type':'text','text':'RGB_REFERENCE '+json.dumps({'observation_id':source,'camera':camera})},
                {'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(raw).decode()}}])
        self.seen_images.update(source for source,_,_ in images)
        self.sent_contracts.update(result.get('coordinate_contracts',{}))
        for record in result['records']:
            # Pages contain text fragments, not yet a readable EEF observation.
            eef=record.get('data',{}).get('eef',{})
            if record.get('kind')=='observation' and eef.get('coordinate_contract_id') and eef.get('arms'):
                self.seen_telemetry.add(record['id'])
        return result,parts

    def _evidence_page(self, token, *, first=False, media=None, contracts=None):
        context=getattr(self,'_evidence_page_context',{}).get(token)
        if context is None or context['owner']!=self.retrieval_owner:
            raise ValueError('Unknown raw evidence continuation for this manager request')
        if (context['episode']!=self.session.episode or context['profile']!=self.session.perception.name
                or context['purpose']!=self.purpose or context['frontier']!=self.frontier):
            raise ValueError('Raw evidence continuation scope changed')
        source=context['source']
        if self._evidence_identity(source)!=context['identity']:raise ValueError('Raw evidence source changed')
        # Repeat the ordinary source/profile/prepare gate, without recomputing the
        # frozen read view or silently gaining newly read endpoint measurements.
        checked=self._record(source,body=context['body'],delivered_ids=())
        if not first:
            self._evidence_media([checked],context['body'],context['cameras'])
            media={};contracts={}
        base={}
        if contracts:base['coordinate_contracts']={c['id']:c for c in contracts.values()}
        if context['remaining']:
            base.update(remaining_selectors=context['remaining'],complete_request=False)
        base['read_qualification']='Immutable raw detail fragments; no raw/telemetry permission until all pages reconstruct the exact record.'
        # page() accounts for its own escaped fragment/token; include every
        # outer field as well. The empty list's brackets are shared.
        page_budget=24576-readout.size({**base,'records':[]})
        fragment=readout.page(self,token,page_budget,owner=self.retrieval_owner)
        result={**base,'records':[fragment]}
        if fragment['continuation']:
            self._evidence_page_context[fragment['continuation']]=context
        return self._evidence_deliver(result,[source] if first else [],media or {})

    def _evidence(self,args):
        if 'continuation' in args:
            if set(args)!={'continuation'} or not isinstance(args['continuation'],str):
                raise ValueError('Use raw evidence continuation alone; no changed selectors')
            return self._evidence_page(args['continuation'])
        if set(args)-{'ids','receipts','measurements','cameras','body'}:raise ValueError('Use exact source selectors')
        packed='receipts' in args or 'measurements' in args
        if packed and 'ids' in args:raise ValueError('Choose ids OR receipts/measurements')
        cameras=args.get('cameras',[]);body=args.get('body','none');groups=[]
        if not isinstance(cameras,list) or any(not isinstance(c,str) for c in cameras) or body not in ('none','ee','full'):
            raise ValueError('Invalid camera/body selectors')
        if packed:
            for key,kind in (('receipts','receipt'),('measurements','surface_inspection')):
                values=args.get(key,[])
                if not isinstance(values,list) or len(values)>2 or any(not isinstance(v,str) for v in values):raise ValueError('At most two '+key+' per evidence pack')
                for source in values:
                    row=self.session.store.db.execute('SELECT kind,payload,run FROM evidence WHERE id=?',(source,)).fetchone()
                    if row is None or row['kind']!=kind or not self._available(source):raise ValueError('Pack source must be registered, visible and of the selected kind')
                    data=json.loads(row['payload'])
                    before=data.get('before_observation' if kind=='receipt' else 'before_observation_id')
                    if not before:raise ValueError('Pack source has no observation endpoint')
                    endpoint=self.session.store.db.execute("SELECT run FROM evidence WHERE id=? AND kind='observation'",(before,)).fetchone()
                    if endpoint is None or endpoint['run']!=row['run']:raise ValueError('Pack observation must belong to the source run')
                    members=[before,source]
                    if kind=='receipt':
                        post=self.session.store.db.execute("SELECT id FROM evidence WHERE run=? AND kind='observation' AND json_extract(payload,'$.after_receipt')=? ORDER BY sequence LIMIT 1",(row['run'],source)).fetchone()
                        if not post:raise ValueError('Receipt has no registered post observation')
                        members.append(post['id'])
                    groups.append((key,source,members))
        else:
            ids=args.get('ids',[])
            if not isinstance(ids,list) or not 1<=len(ids)<=6 or any(not isinstance(i,str) for i in ids):raise ValueError('Read one to six sources')
            groups=[('ids',source,[source]) for source in ids]
        ids=list(dict.fromkeys(source for _,_,members in groups for source in members))
        if not 1<=len(ids)<=(10 if packed else 6):raise ValueError('Read one to six sources or a bounded receipt/measurement pack')
        missing=[];future=[]
        for source in ids:
            row=self.session.store.db.execute('SELECT id FROM evidence WHERE id=?',(source,)).fetchone()
            if row is None:missing.append(source)
            elif not self._available(source):future.append(source)
        if missing or future:raise ValueError('Evidence request rejected: '+json.dumps({'unregistered':missing,'beyond_request_frontier':future}))
        # Validate *all* selectors before a partial response. This intentionally
        # does not authorize endpoint-derived fields using still-unselected IDs.
        checked=[self._record(source,body=body,delivered_ids=()) for source in ids]
        media,contracts=self._evidence_media(checked,body,cameras)
        def remaining(offset):
            result={}
            for key,source,_ in groups[offset:]:result.setdefault(key,[]).append(source)
            if result:result.update(cameras=cameras,body=body)
            return result
        def response(selected,offset):
            result={'records':[self._record(source,body=body,delivered_ids=selected) for source in selected]}
            selected_contracts={contracts[source]['id']:contracts[source] for source in selected if source in contracts}
            if selected_contracts:result['coordinate_contracts']=selected_contracts
            if packed:result['delivered_sources']=selected
            if offset<len(groups):result.update(remaining_selectors=remaining(offset),complete_request=False)
            return result
        selected=[];answer=None;requested=[]
        for offset,(_,_,members) in enumerate(groups,1):
            requested=list(dict.fromkeys([*requested,*members]))
            result=response(requested,offset)
            image_count=sum(len(media.get(source,[])) for source in requested)
            if readout.size(result)<=24576 and image_count<=6:
                selected=requested;answer=result
        if answer is not None:
            return self._evidence_deliver(answer,selected,media)
        # Selector metadata shrinks with each prefix, so inspect every bounded
        # prefix before deciding that no whole response fits.
        members=groups[0][2];result=response(members,1)
        image_count=sum(len(media.get(source,[])) for source in members)
        if packed or image_count>6:
            hint=(' If compact EEF state and measured endpoint action digests suffice, explicitly choose body=ee; retain full when needed.' if body=='full' else '')
            raise ValueError('Pack rejected: text_bytes='+str(readout.size(result))+'/24576; rgb_images='+str(image_count)+'/6. '+
                'A single complete group exceeds the response budget; reducing cameras does not reduce record bytes. '+
                'Read these expanded exact IDs explicitly, preserving needed body/cameras: '+json.dumps(members)+'.'+hint+' No read permission granted.')
        # Only one record too large even on its own is paged. No ordinary
        # following record is fragmented to consume leftover response room.
        alone={k:v for k,v in result.items() if k not in ('remaining_selectors','complete_request')}
        if readout.size(alone)<=24576:
            raise ValueError('The complete record fits alone, but remaining-selector metadata exceeds the response budget. '+
                'Read the first exact ID alone, then the remaining original IDs. No read permission granted.')
        source=members[0]
        record={**result['records'][0],'layer':'evidence','revision':result['records'][0]['sequence'],
            'view':'detail','complete_record':True}
        token=readout._continuation(self,record,owner=self.retrieval_owner)
        if not hasattr(self,'_evidence_page_context'):self._evidence_page_context={}
        self._evidence_page_context[token]={'owner':self.retrieval_owner,'episode':self.session.episode,
            'profile':self.session.perception.name,'purpose':self.purpose,'frontier':dict(self.frontier),'source':source,
            'identity':self._evidence_identity(source),'body':body,'cameras':cameras,'remaining':remaining(1)}
        return self._evidence_page(token,first=True,media={source:media.get(source,[])},
            contracts={source:contracts[source]} if source in contracts else {})

    def _case_batch(self,targets):
        store=self.session.store
        if targets is None:
            targets=[]
            for row in store.db.execute('SELECT c.id,c.revision FROM claims c JOIN claim_grounding g ON g.revision=c.revision '
                'WHERE g.source_episode=? AND c.revision<=? ORDER BY c.revision DESC',(self.session.episode,self.revision)):
                latest=library.latest_case(store,row['id'],self.revision,self.frontier)
                if latest is None or latest['revision']!=row['revision']:continue
                disposition=store.db.execute('SELECT action FROM case_disposition WHERE case_revision=? ORDER BY sequence DESC LIMIT 1',(row['revision'],)).fetchone()
                if disposition is None or disposition['action']=='defer':targets.append(row['id'])
                if len(targets)>=4:break
        if not isinstance(targets,list) or len(targets)>4 or len(set(targets))!=len(targets):raise ValueError('Up to four distinct targeted cases')
        refs=[]
        for identifier in targets:
            row=library.latest_case(store,identifier,self.revision,self.frontier)
            if row is None:raise ValueError('Target outside snapshot')
            refs.append({'id':identifier,'revision':row['revision']})
        if not refs:return {'records':[]}
        answer,_=read_experience(self.session,'Organize these exact cases',level='case',case_refs=refs,
            limit=4,budget_bytes=24000,frontier=self.frontier,revision=self.revision,
            knowledge_revision=self.knowledge_revision,
            version_snapshot={'cases':self.own_cases,'knowledge':self.own_knowledge},
            reader_owner=self.retrieval_owner,track_delivery=False,view='detail',complete_only=True)
        actual={'records':[r for r in answer['records'] if r.get('complete_record')]}
        self._admit(actual)
        from .readout import note_delivery
        note_delivery(self.session,self.retrieval_owner,actual)
        # Only complete delivered cards become targets. Never truncate a case and grant its full version.
        return {**answer,'records':actual['records']}

    def _resolve_ref(self,ref,locals_,previous_refs,visible_cases,visible_knowledge,*,related_statement_versions=None):
        ref=dict(ref);typ=ref.get('type')
        if 'revision' in ref:raise ValueError('Host resolves revisions; do not supply revision')
        if typ=='knowledge' and ref.get('relation')!='related':
            raise ValueError('Knowledge references require relation=related; navigation supplies no physical support')
        if 'local' in ref:
            if typ not in ('case','knowledge') or 'id' in ref or ref['local'] not in locals_ or not ref['local'].startswith('c' if typ=='case' else 'k'):raise ValueError('Unknown or ambiguous local reference')
            identifier,revision=locals_[ref.pop('local')];ref.update(id=identifier,revision=revision)
        elif typ in ('case','knowledge'):
            versions=visible_cases if typ=='case' else visible_knowledge
            if typ=='knowledge':
                versions=dict(versions)
                for identifier,known_revision in (related_statement_versions or {}).items():
                    versions[identifier]=max(versions.get(identifier,-1),known_revision)
            revision=versions.get(ref.get('id'))
            if revision is None:
                old=next((r for r in previous_refs if r['type']==typ and r['id']==ref.get('id') and
                    {k:v for k,v in r.items() if k!='revision'}==ref),None)
                if old is None:
                    required_view='statement' if typ=='knowledge' else 'detail'
                    raise ValueError('Source must be actually delivered; inherited refs may only be preserved unchanged. '+
                        str(typ)+'/'+str(ref.get('id'))+' lacks complete '+required_view+' permission in this transaction. '+
                        'Use search_experience '+json.dumps({'level':typ,'ids':[ref.get('id')],'view':required_view})+' and finish its pages. '+
                        'Navigation/partial pages grant no source permission; related knowledge grants no support, pins, parameters or target modification.')
                revision=old['revision']
            ref['revision']=revision
        return ref

    def _publish(self,args,*,select_defaults=True,emit=True,group_digest=None):
        if set(args)-{'expected_version','answer','default_knowledge','history_refs','memory_patch','knowledge_ops','processed','dispositions'}:
            raise ValueError('Unknown publication fields; obsolete work packets are not supported')
        if args.get('expected_version')!=self.version:raise ValueError('Stale material version')
        if self.session.current_observation and self.session.current_observation['id']!=self.observation_id:
            raise ValueError('Current observation changed during management')
        answer=args.get('answer')
        if not isinstance(answer,str):raise ValueError('answer: string required; maximum1800 UTF-8 bytes')
        if len(answer.encode())>1800:raise ValueError('answer: '+str(len(answer.encode()))+' UTF-8 bytes; maximum1800 bytes')
        patches=args.get('memory_patch',[]);ops=args.get('knowledge_ops',[]);processed=args.get('processed',[]);dispositions=args.get('dispositions',[])
        if self.purpose=='prepare' and (patches or ops or processed or dispositions):raise ValueError('Preparation cannot write memory or processing state')
        if not all(isinstance(v,list) for v in (patches,ops,processed,dispositions)):raise ValueError('Operations must be arrays')
        reference_issues=[]
        for op_index,op in enumerate(ops):
            for issue in library.case_reference_issues(op.get('refs',[])):
                reference_issues.append({**issue,'path':f'knowledge_ops[{op_index}].'+issue['path']})
        if reference_issues:raise ValueError('Case references require relation and specific reason: '+json.dumps(reference_issues))
        defaults=args.get('default_knowledge',self.default_ids)
        if not isinstance(defaults,list) or len(defaults)>12 or len(set(defaults))!=len(defaults) or any(not isinstance(i,str) for i in defaults):raise ValueError('Default knowledge IDs required')
        histories=args.get('history_refs',self.history_ids)
        if not isinstance(histories,list) or len(histories)>2 or any(not isinstance(i,str) for i in histories) or len(set(histories))!=len(histories):raise ValueError('At most two distinct history knowledge IDs required')
        store=self.session.store
        decisions=[];updates=[];locals_={};organized={};required=dict(self.targets)
        with store.db:
            store._batch_depth+=1
            try:
                for patch_index,original in enumerate(patches):
                    patch=dict(original);local=patch.pop('local',None)
                    if local is not None:
                        import re
                        if not isinstance(local,str) or not re.fullmatch(r'c[1-9][0-9]*',local) or local in locals_:raise ValueError('Unique local c1,c2 case handles required')
                    if patch['op']=='new' and local is None:raise ValueError('New case needs a local handle')
                    navigation_tags=patch.pop('navigation',{})
                    if not isinstance(navigation_tags,dict) or set(navigation_tags)-{'object','part','operation'} or any(not isinstance(v,str) or not 0<len(v)<=100 for v in navigation_tags.values()):
                        raise ValueError('navigation: object/part/operation must each be1..100 Unicode characters; labels do not certify success')
                    if 'claim_revision' in patch:raise ValueError('Host resolves claim revision')
                    if patch.get('claim_id'):
                        if patch['claim_id'] not in self.visible_revisions:raise ValueError('Read case before updating')
                        patch['claim_revision']=self.visible_revisions[patch['claim_id']]
                    patch=store.normalize_patch(patch)
                    library.route(patch['kind'],patch['module'])
                    if 'roles' in patch:raise ValueError('Case role facets are host-derived; omit roles from a patch')
                    if patch.get('epistemic')=='telemetry_inferred':
                        obs={source for source in patch['evidence'] if (row:=store.db.execute('SELECT kind FROM evidence WHERE id=?',(source,)).fetchone()) and row['kind']=='observation'}
                        if not obs or not obs<=self.seen_telemetry:raise ValueError('telemetry_inferred requires exact EEF state actually delivered with body=ee/full in this transaction')
                    if patch.get('epistemic')=='visually_inferred':
                        obs={source for source in patch['evidence'] if (row:=store.db.execute('SELECT kind FROM evidence WHERE id=?',(source,)).fetchone()) is not None and row['kind']=='observation'}
                        if not obs<=self.seen_images:raise ValueError('New visual observations require cited RGB in this transaction; secondary reuse needs a knowledge op')
                    result=store.submit(patch,allowed_sources=self.allowed,cutoff=self.frontier,
                        dependencies=self.request_dependencies,episode=self.session.episode,visible_revisions=self.visible_revisions)
                    if not result['accepted']:raise ValueError(f'memory_patch[{patch_index}]: '+result['reason'])
                    facets={'entity':patch.get('observed_entity','unknown'),'observable':patch.get('observable','unknown')}
                    facets.update(navigation_tags)
                    arms=set()
                    for source in patch['evidence']:
                        row=store.db.execute("SELECT payload FROM evidence WHERE id=? AND kind='receipt'",(source,)).fetchone()
                        if row:arms.update(json.loads(row[0]).get('plan',{}).get('trace',{}).get('arms',{}))
                    if arms:facets['arm']='both' if len(arms)>1 else next(iter(arms))
                    store.db.execute('INSERT INTO case_navigation VALUES(?,?,?,?)',(result['revision'],patch['kind'],patch['module'],json.dumps(facets)))
                    decisions.append(result)
                    required[result['claim_id']]=result['revision']
                    if local:locals_[local]=(result['claim_id'],result['revision'])
                visible_cases={**self.visible_revisions,**{d['claim_id']:d['revision'] for d in decisions}}
                visible_knowledge=dict(self.visible_knowledge)
                for op_index,original in enumerate(ops):
                    op=dict(original);local=op.pop('local',None)
                    if local is not None:
                        import re
                        if op.get('op')!='add' or not re.fullmatch(r'k[1-9][0-9]*',local) or local in locals_:raise ValueError('Unique k1,k2 handles only for added knowledge')
                    previous=[]
                    if op.get('op')!='add':
                        known={**self.selectable_knowledge,**self.visible_knowledge} if op.get('op')=='append_refs' else self.visible_knowledge
                        row=store.db.execute('SELECT refs FROM reusable_knowledge WHERE id=? AND revision=?',
                            (op.get('id'),known.get(op.get('id')))).fetchone()
                        if row:previous=json.loads(row['refs'])
                    if 'expected_revision' in op:raise ValueError('Host resolves knowledge revision')
                    op['refs']=[self._resolve_ref(r,locals_,previous,visible_cases,visible_knowledge,
                        related_statement_versions=self.read_knowledge_versions) for r in op.get('refs',[])]
                    if op.get('op')!='append_refs':
                        op['params']=[{**param,'source':next((ref for ref in op['refs'] if
                            ref.get('id')==param.get('source') or locals_.get(param.get('source'))==(ref.get('id'),ref.get('revision'))),None)} for param in op.get('params',[])]
                    try:
                        update=library.publish_knowledge(store,op,allowed=self.allowed,visible_cases=visible_cases,
                            visible_knowledge={**self.selectable_knowledge,**visible_knowledge} if op.get('op')=='append_refs' else visible_knowledge,
                            related_statement_versions=self.read_knowledge_versions,
                            related_scope={'case_at':self.revision,'knowledge_at':self.knowledge_revision,
                                'profile':self.session.perception.name,
                                'snapshot':{'cases':{**getattr(self,'own_cases',{}),**{d['claim_id']:d['revision'] for d in decisions}},
                                    'knowledge':{**getattr(self,'own_knowledge',{}),**{u['id']:u['revision'] for u in updates}}}},
                            dependencies=self.request_dependencies,frontier=self.frontier)
                    except ValueError as error:
                        raise ValueError(f'knowledge_ops[{op_index}]: {error}') from error
                    updates.append(update);visible_knowledge[update['id']]=update['revision']
                    if local:locals_[local]=(update['id'],update['revision'])
                    if op['op']=='retire':continue
                    for ref in op['refs']:
                        if ref['type']!='case' or visible_cases.get(ref['id'])!=ref['revision'] or ref['id'] not in required:continue
                        slot=organized.setdefault(ref['id'],{'revision':ref['revision'],'action':'linked','knowledge_revisions':[],'reason':'Referenced by committed knowledge; relation and qualification are in exact knowledge refs.'})
                        slot['knowledge_revisions'].append(update)
                for item in dispositions:
                    identifier=item['case']
                    if identifier in locals_ and not identifier.startswith('c'):raise ValueError('Case disposition cannot reference a knowledge handle')
                    identifier,revision=locals_.get(identifier,(identifier,visible_cases.get(identifier)))
                    if revision is None:raise ValueError('Disposition requires actually read case version')
                    if identifier in organized:raise ValueError('Case cannot be both linked and kept/deferred or disposed twice')
                    if item['action'] not in ('keep','defer') or not isinstance(item.get('reason'),str) or not item['reason'].strip():raise ValueError('Keep/defer requires a concrete reason')
                    organized[identifier]={'revision':revision,'action':item['action'],'knowledge_revisions':[],'reason':item['reason']}
                if any(identifier not in organized or organized[identifier]['revision']!=revision for identifier,revision in required.items()):raise ValueError('Every new/targeted case version needs knowledge linkage or keep/defer disposition')
                for identifier,item in organized.items():
                    store.db.execute('INSERT INTO case_disposition(request_id,case_id,case_revision,action,knowledge_revisions,reason) VALUES(?,?,?,?,?,?)',
                        (self.request_id,identifier,item['revision'],item['action'],json.dumps(item['knowledge_revisions']),item['reason']))
                cards=None;history_cards=None
                if select_defaults:
                    defaults=[locals_.get(identifier,(identifier,None))[0] for identifier in defaults]
                    # Selecting an existing record must follow an actual read, not a guessed ID.
                    selectable={**getattr(self,'selectable_knowledge',{}),**visible_knowledge}
                    missing=set(defaults)-set(self.default_ids)-selectable.keys() if 'default_knowledge' in args else set()
                    if missing:
                        gaps=[]
                        for identifier in defaults:
                            if identifier not in missing:continue
                            row=library.latest_knowledge(store,identifier,self.revision,self.knowledge_revision,self.frontier)
                            gap={'id':identifier,'revision':row['revision'] if row else None,'state':'not_visible' if row is None else 'statement_not_read'}
                            for (layer,read_id,revision,read_view,view_digest),pages in self.read_pages.items():
                                if layer=='knowledge' and read_id==identifier and row is not None and revision==row['revision']:
                                    end=0
                                    for offset,fragment in sorted(pages.items()):
                                        if offset!=end:break
                                        end+=len(fragment)
                                    gap.update(state='partial_'+read_view,contiguous_characters=end)
                            gaps.append(gap)
                        raise ValueError('Default selection missing complete statements: '+json.dumps(gaps))
                    knowledge_at=self.knowledge_revision
                    case_at=self.revision
                    overlay={'cases':{**getattr(self,'own_cases',{}),**{d['claim_id']:d['revision'] for d in decisions}},
                             'knowledge':{**getattr(self,'own_knowledge',{}),**{u['id']:u['revision'] for u in updates}}}
                    cards=library.read_knowledge(store,'',context=self.session.context,dependencies=self.session.dependencies,
                        ids=defaults,limit=12,at=knowledge_at,case_at=case_at,frontier=self.frontier,snapshot=overlay)
                    approved={**self.default_versions,**selectable}
                    if 'default_knowledge' in args and any(approved.get(card['id'])!=card['revision'] for card in cards):raise ValueError('Default selection cannot absorb an unread knowledge version')
                    history_cards=library.read_knowledge(store,'',context=self.session.context,dependencies=self.session.dependencies,
                        ids=histories,limit=2,at=knowledge_at,case_at=case_at,frontier=self.frontier,snapshot=overlay) if histories else []
                    history_approved=self.read_knowledge_versions if 'history_refs' in args else self.history_versions
                    rejected=[];legal_defaults=[];legal_history=[]
                    overlap=set(defaults)&set(histories)
                    def qualification(card):
                        return {k:card.get(k) for k in ('id','revision','status','strength','support_cases','default_eligible',
                            'applicability','source_contested','source_revised','related_warnings','counter_cases')}
                    for identifier in defaults:
                        card=next((c for c in cards if c['id']==identifier),None)
                        if 'default_knowledge' not in args:
                            if card is not None and card['default_eligible'] and self.default_versions.get(identifier)==card['revision'] and identifier not in overlap:legal_defaults.append(card)
                            continue
                        if card is None:rejected.append({'id':identifier,'selection':'default','reason':'not_visible'})
                        elif not card['default_eligible'] or identifier in overlap:
                            rejected.append({'selection':'default','reason':'overlap' if identifier in overlap else 'not_default_eligible','qualification':qualification(card)})
                        else:legal_defaults.append(card)
                    for identifier in histories:
                        card=next((c for c in history_cards if c['id']==identifier),None)
                        if 'history_refs' not in args:
                            if card is not None and self.history_versions.get(identifier)==card['revision'] and card['status']=='active' and card['strength']!='hypothesis' and (card['support_cases'] or card['strength']=='configuration') and not card['default_eligible'] and identifier not in overlap:legal_history.append(card)
                            continue
                        if card is None:rejected.append({'id':identifier,'selection':'history','reason':'not_visible_or_unread_local_handle'})
                        elif (history_approved.get(identifier)!=card['revision'] or card['status']!='active' or card['strength']=='hypothesis'
                              or not (card['support_cases'] or card['strength']=='configuration') or card['default_eligible'] or identifier in overlap):
                            rejected.append({'selection':'history','reason':'requires_exact_read_active_supported_nonhypothesis_ineligible_version_without_overlap',
                                'actually_read_revision':history_approved.get(identifier),'qualification':qualification(card)})
                        else:legal_history.append(card)
                    if rejected:
                        forecast=self._selection_forecast(legal_defaults,legal_history,material_version=self.version+1)
                        raise SelectionRejected({'items':rejected,'legal_subset_forecast':forecast,
                            'retained_defaults':self.default_ids,'retained_history_refs':self.history_ids,
                            'repair':'Resubmit only the selection component; no automatic filtering or demotion. Fully read supported nonhypothesis historical entries may be explicitly selected as history_refs, never as default knowledge.'})
                    cards=legal_defaults;history_cards=legal_history
                    # Prefix packing chooses complete statements that fit; selection is not erased by an inline budget.
                for item in processed:
                    receipt=item['receipt_id'];data=store.evidence_payload(receipt)
                    source=store.db.execute("SELECT run FROM evidence WHERE id=? AND kind='receipt'",(receipt,)).fetchone()
                    if source is None:raise ValueError('Processing requires an exact receipt')
                    post=store.db.execute("SELECT id FROM evidence WHERE run=? AND kind='observation' AND json_extract(payload,'$.after_receipt')=? ORDER BY sequence LIMIT 1",(source['run'],receipt)).fetchone()
                    if not post or not {receipt,data.get('before_observation'),post['id']}<=self.allowed:
                        raise ValueError('Processing requires delivered complete before/receipt/post sources')
                    if item['disposition'] not in ('case','knowledge','no_lesson'):raise ValueError('Unknown processing disposition')
                    store.db.execute('INSERT OR REPLACE INTO processed_intervals VALUES(?,?,?)',(receipt,self.request_id,item['disposition']))
                if group_digest is not None:
                    store.db.execute('INSERT INTO publication_groups VALUES(?,?,?)',(self.request_id,group_digest,
                        json.dumps({'memory_updates':decisions,'knowledge_updates':updates,'case_dispositions':organized,
                                    'local_handles':locals_},ensure_ascii=False)))
            finally:store._batch_depth-=1
        if cards is not None:
            if 'default_knowledge' in args:
                self.default_ids=[card['id'] for card in cards];self.default_versions={card['id']:card['revision'] for card in cards}
            if 'history_refs' in args:
                self.history_ids=[card['id'] for card in history_cards];self.history_versions={card['id']:card['revision'] for card in history_cards}
        self.version+=1;self.prepared=True
        result={'status':'PASS','version':self.version,'answer':answer,'memory_updates':decisions,'knowledge_updates':updates,
            'default_knowledge':cards or [],'case_dispositions':organized,'processed':processed,'local_handles':locals_,'action_effect':'none; policy alone chooses actions'}
        if emit:
            publication=self.session.trace.event('manager_publication',{**result,'purpose':self.purpose,'question':self.question,
                'provided_sources':sorted(self.allowed),'seen_RGB':sorted(self.seen_images),
                'case_revision_at_request':self.revision,'knowledge_revision_at_request':self.knowledge_revision,'source_frontier':self.frontier,
                'default_ids':self.default_ids,'history_ids':self.history_ids,'history_versions':self.history_versions,'source_observation_id':self.observation_id})
            self.publication=publication['id'];result['publication']=publication['id']
            if self.purpose=='review' and answer:
                from .leads import add
                with store.db:add(store,publication,{'question':self.question,'answer':answer,
                    'review_publication':publication['id'],'source_refs':sorted(self.allowed),
                    'qualification':'Manager interpretation; not adopted action or observed success'})
        else:result['publication']=None
        return result,True

    def _publish_groups(self,args):
        """Each explicit group is atomic; unrelated accepted groups survive a rejection."""
        permitted={'expected_version','answer','groups','default_knowledge','history_refs'}
        if set(args)-permitted:raise ValueError('Use explicit groups; flat publication operations are not supported')
        if args.get('expected_version')!=self.version:raise ValueError('Stale material version')
        groups=args.get('groups',[])
        if not isinstance(groups,list) or len(groups)>8:raise ValueError('Use at most eight independent groups')
        original_targets=dict(self.targets);committed=[];issues=[];organized={};memory=[];knowledge=[]
        if not hasattr(self,'group_commits'):self.group_commits={};self.own_cases={};self.own_knowledge={};self.request_updates=[];self.request_memory=[];self.request_handles={}
        owners={};blocked={}
        for index,group in enumerate(groups):
            if not isinstance(group,dict) or any(not isinstance(v,list) or any(not isinstance(op,dict) for op in v) for v in group.values()):
                blocked[index]={'code':'invalid_group_shape'};continue
            if any(not isinstance(op.get('refs',[]),list) or any(not isinstance(r,dict) for r in op.get('refs',[])) for op in group.get('knowledge_ops',[])):
                blocked[index]={'code':'invalid_knowledge_refs','path_detail':'knowledge_ops.refs must be an array of source objects'};continue
            writes=set()
            for field,kind,identifier in (('memory_patch','case','claim_id'),('knowledge_ops','knowledge','id')):
                for op in group.get(field,[]):
                    if op.get(identifier):writes.add((kind,op[identifier]))
                    if op.get('local'):writes.add(('local',op['local']))
            writes.update(('disposition',op.get('case')) for op in group.get('dispositions',[]))
            writes.update(('interval',op.get('receipt_id')) for op in group.get('processed',[]))
            for op in group.get('knowledge_ops',[]):
                for ref in op.get('refs',[]):
                    if ref.get('type')=='case' and ref.get('id') in self.targets:writes.add(('disposition',ref['id']))
            for resource in writes:
                if resource in owners:
                    prior=owners[resource]
                    issue={'code':'cross_group_write_dependency','resource':list(resource),'repair':'Merge writes/dispositions for the same entity into one group'}
                    blocked[index]=issue;blocked[prior]=issue
                else:owners[resource]=index
        for index,group in enumerate(groups):
            if index in blocked:continue
            reads=set()
            for op in group.get('knowledge_ops',[]):
                reads.update((r.get('type'),r.get('id')) for r in op.get('refs',[]) if r.get('id'))
            reads.update(('case',d.get('case')) for d in group.get('dispositions',[]))
            for resource in reads:
                owner=owners.get(resource)
                if owner is not None and owner!=index:
                    issue={'code':'cross_group_write_read_dependency','resource':list(resource),'repair':'Merge the writer and dependent reader into one group'}
                    blocked[index]=issue;blocked[owner]=issue
        for index,group in enumerate(groups):
            path=f'groups[{index}]'
            if index in blocked:
                issues.append({'path':path,**blocked[index]});continue
            if not isinstance(group,dict) or set(group)-{'memory_patch','knowledge_ops','dispositions','processed'}:
                issues.append({'path':path,'code':'invalid_group_fields'});continue
            digest=hashlib.sha256(json.dumps(group,sort_keys=True).encode()).hexdigest()
            prior=self.session.store.db.execute('SELECT result FROM publication_groups WHERE request_id=? AND digest=?',(self.request_id,digest)).fetchone()
            if prior:
                saved=json.loads(prior[0])
                self.request_handles.update(saved['local_handles'])
                self.own_cases.update({r['claim_id']:r['revision'] for r in saved['memory_updates']})
                self.own_knowledge.update({r['id']:r['revision'] for r in saved['knowledge_updates']})
                self.visible_revisions.update(self.own_cases);self.visible_knowledge.update(self.own_knowledge)
                organized.update(saved['case_dispositions'])
                for field,items in (('request_memory',saved['memory_updates']),('request_updates',saved['knowledge_updates'])):
                    prior_items=getattr(self,field)
                    for item in items:
                        if item not in prior_items:prior_items.append(item)
                committed.append({'group':index,'already_committed':True,'result':saved});continue
            declared={op.get('local') for field in ('memory_patch','knowledge_ops') for op in group.get(field,[]) if op.get('local')}
            if declared & self.request_handles.keys():
                issues.append({'path':path,'code':'committed_handle_changed','handles':sorted(declared & self.request_handles.keys())});continue
            foreign=[r['local'] for op in group.get('knowledge_ops',[]) for r in op.get('refs',[]) if r.get('local') and r['local'] not in declared]
            if foreign:
                issues.append({'path':path+'.knowledge_ops.refs','code':'cross_group_dependency','handles':foreign,'repair':'Merge dependent case and knowledge into one group'});continue
            # Local handles live within one group. Shared dependencies must share that group.
            ids={p.get('claim_id') for p in group.get('memory_patch',[])}
            ids.update(d.get('case') for d in group.get('dispositions',[]))
            ids.update(r.get('id') for op in group.get('knowledge_ops',[]) for r in op.get('refs',[]) if r.get('type')=='case')
            self.targets={k:v for k,v in original_targets.items() if k in ids}
            try:
                result,_=self._publish({**group,'expected_version':self.version,'answer':args.get('answer')},select_defaults=False,emit=False,group_digest=digest)
            except (KeyError,TypeError,ValueError,sqlite3.IntegrityError) as error:
                issues.append({'path':path,'code':'group_rejected','reason':str(error),'committed':False})
                continue
            self.group_commits[digest]=result
            self.request_handles.update(result['local_handles'])
            committed.append({'group':index,'publication':result['publication']})
            self.own_cases.update({r['claim_id']:r['revision'] for r in result['memory_updates']})
            self.own_knowledge.update({r['id']:r['revision'] for r in result['knowledge_updates']})
            self.request_updates.extend(result['knowledge_updates']);self.request_memory.extend(result['memory_updates'])
            organized.update(result['case_dispositions']);memory.extend(result['memory_updates']);knowledge.extend(result['knowledge_updates'])
            self.visible_revisions.update({r['claim_id']:r['revision'] for r in result['memory_updates']})
            self.visible_knowledge.update({r['id']:r['revision'] for r in result['knowledge_updates']})
        self.targets=original_targets
        missing={k:v for k,v in original_targets.items() if k not in organized}
        if missing:issues.append({'path':'groups','code':'undisposed_targets','cases':missing})
        if 'default_knowledge' in args or 'history_refs' in args or not groups:
            self.targets={}
            try:
                final,_=self._publish({'expected_version':self.version,'answer':args.get('answer'),
                    **({'default_knowledge':[self.request_handles.get(i,(i,None))[0] for i in args['default_knowledge']]} if 'default_knowledge' in args else {}),
                    **({'history_refs':args['history_refs']} if 'history_refs' in args else {})},
                    select_defaults='default_knowledge' in args or 'history_refs' in args,emit=False)
                committed.append({'group':'default_and_answer','publication':final['publication']})
            except (KeyError,TypeError,ValueError,sqlite3.IntegrityError) as error:
                issue={'path':'default_knowledge','code':'selection_rejected','retained_defaults':self.default_ids,'retained_history_refs':self.history_ids}
                issue.update({'details':error.details} if isinstance(error,SelectionRejected) else {'reason':str(error)})
                issues.append(issue)
            finally:self.targets=missing
        self.targets=missing
        result={'status':'NEEDS_WORK' if issues else 'PASS','version':self.version,'answer':args.get('answer'),
            'publication':self.publication,'committed_groups':committed,'issues':issues,
            'memory_updates':list(self.request_memory),'knowledge_updates':list(self.request_updates),'case_dispositions':organized,
            'retry_instruction':'Resubmit only rejected groups at the returned version; committed groups must not be repeated.'}
        result['delivery_manifest']=self.material_delivery()
        publication=self.session.trace.event('manager_group_publication',{**result,'purpose':self.purpose,
            'question':self.question,'provided_sources':sorted(self.allowed),'source_frontier':self.frontier,
            'case_revision_at_request':self.revision,'knowledge_revision_at_request':self.knowledge_revision,
            'default_ids':self.default_ids,'default_versions':self.default_versions,'history_ids':self.history_ids,'history_versions':self.history_versions,'local_handles':self.request_handles})
        self.publication=publication['id'];result['publication']=publication['id']
        if self.purpose=='review' and args.get('answer'):
            from .leads import add
            with self.session.store.db:add(self.session.store,publication,{'question':self.question,
                'answer':args['answer'],'review_publication':publication['id'],'source_refs':sorted(self.allowed),
                'qualification':'Manager interpretation; not physical confirmation'})
        return result,not issues

    def run(self,question,purpose,*,targets=None):
        if self.session.stop_requested():raise RuntimeError('Stopped before management request')
        if purpose not in ('prepare','review','learn') or not isinstance(question,str) or not 0<len(question)<=1200:raise ValueError('One bounded management question required')
        if purpose=='prepare' and self.prepared:raise ValueError('Preparation already completed')
        self.question,self.purpose=question,purpose
        self.frontier=self.session.source_frontier();self.request_dependencies=dict(self.session.dependencies)
        self.revision=self.session.frozen_revision if self.session.frozen_revision is not None else self.session.store.revision
        self.observation_id=self.session.current_observation['id'] if self.session.current_observation else None
        self.session._read_continuations={}
        self._read_continuations={};self._evidence_page_context={}
        self.allowed=set();self.read_pages={};self.read_page_lengths={};self.seen_images=set();self.seen_telemetry=set();self.sent_contracts=set();self.visible_revisions={};self.visible_knowledge={};self.selectable_knowledge={}
        self.read_knowledge_versions={}
        self.knowledge_revision=self.session.frozen_knowledge_revision if self.session.frozen_knowledge_revision is not None else self.session.store.db.execute('SELECT COALESCE(MAX(revision),0) FROM reusable_knowledge').fetchone()[0]
        self.targets={}
        self.group_commits={};self.own_cases={};self.own_knowledge={};self.request_updates=[];self.request_memory=[];self.request_handles={}
        self.retrieval_owner='manager:'+uuid.uuid4().hex
        self.client.start_transaction()
        initial={'purpose':purpose,'question':question,'task':self.session.task_instruction,'expected_version':self.version,
            'perception_profile':self.session.perception.name,
            'case_revision':self.revision,'knowledge_revision':self.knowledge_revision,'navigation':library.navigation(self.session.store,at=self.revision,frontier=self.frontier),
            'body_config_ref':self.session.body_prior['id'] if self.session.body_prior else None,
            'task_contract_ref':self.session.task_specification['id'] if self.session.task_specification else None,
            'operator_clarifications':amendment_view(self.session),
            'active_knowledge_ids':self.default_ids}
        if (purpose!='prepare' and self.previous_result is not None
                and self.previous_result['logical_episode']==self.session.episode
                and self.previous_result['perception_profile']==self.session.perception.name):
            initial['previous_manager_result']={**self.previous_result,
                'current_read_view':{'case_revision':self.revision,'knowledge_revision':self.knowledge_revision}}
            from .readout import size
            initial['previous_manager_result_bytes']=size(initial['previous_manager_result'])
        if self.history_ids:initial['selected_history_refs']=[{'id':i,'revision':self.history_versions[i]} for i in self.history_ids]
        # A bounded direct read replaces repeated discovery of already organized material.
        # This delivers statements, not raw observations or permission to cite physical evidence.
        first_read=self._search({'question':question if purpose!='prepare' else self.session.task_instruction,
            'view':'statement','limit':2})
        initial['relevant_knowledge']=first_read['records']
        if first_read.get('navigation'):initial['knowledge_navigation']=first_read['navigation']
        if first_read.get('candidate_continuation'):initial['knowledge_candidates_continuation']=first_read['candidate_continuation']
        self._admit({'records':initial['relevant_knowledge']})
        from .episodes import read as read_episodes
        initial['episode_candidates']=read_episodes(self.session.store,task=self.session.context['task'],
            frontier=self.frontier,limit=1,exclude_run=self.session.episode,dependencies=self.request_dependencies,contract_id=current_contract(self.session),process_only=True)
        counts={}
        for row in self.session.store.db.execute('SELECT DISTINCT id FROM reusable_knowledge WHERE revision<=?',(self.knowledge_revision,)):
            item=library.latest_knowledge(self.session.store,row['id'],self.revision,self.knowledge_revision,self.frontier)
            if item is not None and item['status']=='active':counts[(item['domain'],item['family'])]=counts.get((item['domain'],item['family']),0)+1
        for item in initial['navigation']['case_counts']:item['knowledge_items']=counts.get((item['domain'],item['family']),0)
        initial['navigation']['knowledge_only_families']=[{'domain':d,'family':f,'knowledge_items':n} for (d,f),n in sorted(counts.items()) if not any(c['domain']==d and c['family']==f for c in initial['navigation']['case_counts'])]
        if purpose=='prepare':
            known=[r for r in initial['relevant_knowledge'] if r.get('default_eligible') and
                   (r.get('complete_record') or r.get('statement_complete'))]
            forecast=self._prefix_packet(known,initial['episode_candidates'],initial['navigation'])
            from .readout import size
            initial['decision_packet_forecast']={
                'based_on_supplied_knowledge_ids':[r['id'] for r in known],
                'complete_knowledge_ids':[r['id'] for r in forecast['knowledge']],
                'navigation_knowledge_ids':[r['id'] for r in forecast['memory_index']],
                'experience_limit_bytes':forecast['packing']['limit_bytes'],
                'experience_serialized_bytes':forecast['packing']['experience_serialized_bytes'],
                'episode_candidate_bytes':size(forecast['episode_candidates']),
                'max_inline_knowledge':2,
                'qualification':'Forecast for supplied candidates, not a selection mandate; different chosen cards may fit differently. Unseen IDs are not decision input.'}
            history=[r for r in initial['relevant_knowledge'] if r.get('default_eligible') is False
                and r.get('status')=='active' and r.get('strength')!='hypothesis'
                and (r.get('support_cases') or r.get('strength')=='configuration')
                and self.read_knowledge_versions.get(r['id'])==r['revision']]
            if history:initial['history_reference_forecast']=self._selection_forecast(known,history[:2],material_version=self.version+1)
        if purpose!='prepare':
            initial['current_observation_ref']=self.observation_id
            data=(self.session.current_observation or {}).get('data',{})
            used,remaining=data.get('env_step'),data.get('remaining_steps')
            initial['simulation_budget']={'source_observation':self.observation_id,
                'env_step':used,'remaining_steps':remaining,
                'step_limit':used+remaining if type(used) is int and type(remaining) is int and used>=0 and remaining>=0 else None,
                'qualification':'Source-time environment budget, separate from model calls; reads and management advance no simulation.'}
            initial['current_attachment_declarations']=self.session.attachment
            if self.session.episode in self.frontier:
                initial['recent_intervals']=self._intervals(self.session.episode,limit=2,recent=True,unprocessed=False)
                initial['unprocessed']=self._intervals(self.session.episode,limit=2)
                first={r['receipt'] for r in initial['unprocessed']['intervals']}
                initial['recent_intervals']['intervals']=[r for r in initial['recent_intervals']['intervals'] if r['receipt'] not in first]
                initial['episode_index']=self._episode_index()
            if getattr(self.session,'episode_ended',False):
                from .episodes import read as read_episodes
                initial['terminal_episode']=read_episodes(self.session.store,ids=[self.session.episode],frontier=self.frontier,limit=1)
        if purpose=='learn':
            from .leads import pending
            initial['review_followups']=pending(self.session.store,self.session.episode,self.frontier)
            batch=self._case_batch(targets)
            initial['case_batch']=batch['records']
            if batch.get('selection_continuation'):
                initial['case_batch_continuation']=batch['selection_continuation']
            self.targets={c['id']:c['revision'] for c in initial['case_batch']}
            families={(c['domain'],c['family']) for c in initial['case_batch']}
            candidates=[];family_navigation=[]
            for domain,family in sorted(families):
                # A specific question carries relation text, not a generic match-all archive.
                questions=' '.join(c['text'][:120] for c in initial['case_batch'] if (c['domain'],c['family'])==(domain,family))
                found=self._search({'question':questions or self.question,'domain':domain,'family':family,
                    'limit':2,'budget_bytes':3500,'view':'statement','track_delivery':False})
                candidates.extend(found['records'])
                if found.get('navigation') or found.get('candidate_continuation'):
                    family_navigation.append({'domain':domain,'family':family,'entries':found.get('navigation',[]),
                        **({'candidate_continuation':found['candidate_continuation']} if found.get('candidate_continuation') else {})})
            delivered={r['id'] for r in initial['relevant_knowledge']}
            initial['same_family_knowledge']=[r for r in candidates if r['id'] not in delivered]
            if family_navigation:initial['same_family_navigation']=family_navigation
            self._admit({'records':initial['same_family_knowledge']})
            from .readout import note_delivery
            note_delivery(self.session,self.retrieval_owner,{'records':initial['same_family_knowledge']})
        request=self.session.trace.event('management_request',initial)
        self.request_id=request['id']
        # CLI restores the system contract on every native resume. Task authority
        # must survive compaction for the manager too, without a second copy in
        # the initial user payload. The immutable trace above retains the full request.
        resident_contract={'task':self.session.task_instruction,'task_contract_ref':initial['task_contract_ref'],
            'perception_profile':self.session.perception.name,'operator_clarifications':initial['operator_clarifications']}
        self.client.operator_contract=initial['operator_clarifications']
        request_materials={k:v for k,v in initial.items() if k!='operator_clarifications'}
        messages=[{'role':'system','content':MANAGER_INSTRUCTIONS+'\nMODE: '+MODE_INSTRUCTIONS[purpose]+
            '\nRESIDENT PUBLIC TASK CONTRACT:\n'+json.dumps(resident_contract,ensure_ascii=False)},
            {'role':'user','content':json.dumps(request_materials,ensure_ascii=False)}]
        while True:
            if self.session.stop_requested():raise RuntimeError('Stopped before management inference')
            if self.client.budget is not None and self.client.budget.view()['remaining']==0:
                result={'status':'NEEDS_WORK','reason':'manager_call_allowance_exhausted',
                    'answer':'Manager allowance exhausted; committed groups remain saved. Any pending issues are unresolved.',
                    'version':self.version,'publication':self.publication,'knowledge_updates':list(self.request_updates),
                    'memory_updates':list(self.request_memory),'unresolved_targets':self.targets,
                    'preparation_completed':self.prepared,'action_effect':'none'}
                self.session.trace.event('management_allowance_exhausted',result)
                return result
            tools=manager_tools()
            if self.client.budget is not None:
                remaining=self.client.budget.view()['remaining']
                if remaining==1:
                    tools=[tool for tool in tools if tool['function']['name']=='publish_materials']
                    messages.append({'role':'user','content':'One manager call remains in the episode allowance. Publish the supported answer now; unresolved issues and no writes are valid.'})
                elif remaining<=4:
                    messages.append({'role':'user','content':f'{remaining} manager calls remain for this episode, including final publication. Use only consequential exact detail/page repairs or publish the supported answer; avoid new broad discovery and preserve a final publication call.'})
            response=self.client.complete(messages,tools)
            message=response['choices'][0]['message'];calls=message.get('tool_calls',[])
            if len(calls)!=1:raise ValueError('Manager must choose one host tool')
            call=calls[0];name=call['function']['name'];args=json.loads(call['function']['arguments'])
            messages.append(message);parts=[];done=False
            try:
                if name=='query_body_history':
                    from .body_index import query
                    result=query(self.session.store,config=self.session.body_prior['data'] if self.session.body_prior else {},
                        run=self.session.episode,frontier=self.frontier,**args)
                elif name=='search_experience':result=self._search(args)
                elif name=='read_trajectory':
                    if set(args)-{'run','after','through','include_processed','lead','directory'} or type(args.get('include_processed',False)) is not bool:
                        raise ValueError('Use run, after, optional through, lead/directory and boolean include_processed')
                    if 'directory' in args:
                        if 'lead' in args or 'include_processed' in args:raise ValueError('Directory and interaction/lead reads are separate')
                        end=args.get('through',self.frontier.get(args['run'],-1))
                        if args['run'] not in self.frontier or type(args['after']) is not int or type(end) is not int or not 0<=args['after']<=end<=self.frontier[args['run']]:
                            raise ValueError('Directory bounds must lie within the request frontier')
                        if self.purpose=='prepare' and args['run']==self.session.episode:raise ValueError('Preparation reads historical episodes only')
                        if args['directory']=='measurements' and not self.session.perception.depth:
                            raise ValueError('Measurement navigation is unavailable in RGB-only mode')
                        result=episode_index.directory(self.session.store,args['run'],args['after'],end,args['directory'])
                    else:result=self._intervals(args['run'],args['after'],through=args.get('through'),unprocessed=not args.get('include_processed',False))
                    if 'lead' in args:
                        from .leads import read
                        result['learning_lead']=read(self.session.store,args['run'],args['lead'],self.frontier)
                elif name=='read_segments':result=self._segments(args)
                elif name=='read_evidence':result,parts=self._evidence(args)
                elif name=='publish_materials':result,done=self._publish_groups(args)
                else:raise ValueError('Unknown management tool')
            except (KeyError,TypeError,ValueError,sqlite3.IntegrityError) as error:result={'status':'NEEDS_WORK','reason':str(error)}
            tool_payload={'request':request['id'],'tool':name,'arguments':args,'result':result}
            completion=self.session.trace.event('manager_tool_result',tool_payload)
            if done:
                source=result['publication']
                if source.rsplit(':',1)[0]!=self.session.trace.root.name:
                    raise ValueError('Completed publication belongs to another trace')
                publication=json.loads((self.session.trace.root/'events'/(source.rsplit(':',1)[1]+'.json')).read_text())
                if publication['id']!=source:raise ValueError('Completed publication identity mismatch')
                if not self.remember_completed_result(publication,{**completion,'payload':tool_payload}):
                    raise ValueError('Completed management publication lacks its matching PASS result')
                return result
            messages.append({'role':'tool','tool_call_id':call['id'],'content':json.dumps(result,ensure_ascii=False)})
            if parts:messages.append({'role':'user','content':[{'type':'text','text':'Explicit historical RGB evidence.'},*parts]})
            self._admit(result)

    def _prefix_sources(self):
        from .episodes import read as read_episodes
        store=self.session.store
        case_at=self.session.frozen_revision if self.session.frozen_revision is not None else store.revision
        frontier=self.frontier if hasattr(self,'frontier') else self.session.source_frontier()
        key=(case_at,tuple(sorted(frontier.items())))
        if store.db.in_transaction:
            nav=library.navigation(store,at=case_at,frontier=frontier)
        elif self._navigation_cache is None or self._navigation_cache[0]!=key:
            self._navigation_cache=(key,library.navigation(store,at=case_at,frontier=frontier))
            nav=self._navigation_cache[1]
        else:nav=self._navigation_cache[1]
        episodes=read_episodes(store,task=self.session.context['task'],frontier=self.session.source_frontier(),limit=1,
            exclude_run=self.session.episode,dependencies=self.session.dependencies,contract_id=current_contract(self.session),process_only=True)
        return episodes,nav

    def material_scope(self):
        def digest(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
        return {'version':self.version,'case_revision':self.session.frozen_revision if self.session.frozen_revision is not None else self.session.store.revision,
            'knowledge_revision':self.session.frozen_knowledge_revision if self.session.frozen_knowledge_revision is not None else
                self.session.store.db.execute('SELECT COALESCE(MAX(revision),0) FROM reusable_knowledge').fetchone()[0],
            'source_frontier_sha256':digest(self.session.source_frontier()),'dependencies_sha256':digest(self.session.dependencies),
            'context_sha256':digest(self.session.context),'observation_id':self.session.current_observation['id'] if self.session.current_observation else None}

    def _render_materials(self):
        store=self.session.store;frontier=self.session.source_frontier()
        def read(ids,limit):
            return library.read_knowledge(store,'',context=self.session.context,dependencies=self.session.dependencies,ids=ids,limit=limit,
                at=self.session.frozen_knowledge_revision,case_at=self.session.frozen_revision,frontier=frontier) if ids else []
        cards=read(self.default_ids,12);history=read(self.history_ids,2)
        eligible=[c for c in cards if c['default_eligible'] and self.default_versions.get(c['id'])==c['revision']]
        retained=[c for c in history if self.history_versions.get(c['id'])==c['revision'] and c['status']=='active'
            and c['strength']!='hypothesis' and (c['support_cases'] or c['strength']=='configuration') and not c['default_eligible']]
        episodes,nav=self._prefix_sources()
        from .readout import statement
        packet=self._prefix_packet([statement(c) for c in eligible],episodes,nav,retained)
        missing={i:'qualification_or_selected_revision_changed' for i in set(self.default_ids)-{c['id'] for c in eligible}}
        missing.update({i:'qualification_or_selected_revision_changed' for i in set(self.history_ids)-{c['id'] for c in retained}})
        return packet,episodes,missing

    def prefix_view(self):
        return self._render_materials()[0]

    def material_delivery(self):
        from .preparation import delivery_manifest
        packet,episodes,missing=self._render_materials()
        defaults=[{'id':i,'revision':self.default_versions[i]} for i in self.default_ids]
        history=[{'id':i,'revision':self.history_versions[i]} for i in self.history_ids]
        result=delivery_manifest(packet,defaults,history,episodes)
        for row in result['default_pool']+result['history_refs']:
            if row['id'] in missing:row['reason']=missing[row['id']]
        result['render_scope']=self.material_scope()
        return result

    def _selection_forecast(self, defaults, history, *, material_version=None):
        from .preparation import delivery_manifest
        from .readout import statement
        episodes,nav=self._prefix_sources()
        packet=self._prefix_packet([statement(c) for c in defaults],episodes,nav,history,material_version=material_version)
        return {**delivery_manifest(packet,defaults,history,episodes),'prediction':True,
            'based_on_default_ids':[c['id'] for c in defaults],'based_on_history_ids':[c['id'] for c in history],
            'render_scope':{**self.material_scope(),'version':packet['version']},'submitted':False}

    def _prefix_packet(self, knowledge, episodes, navigation, history=(), *, material_version=None):
        from .preparation import pack,add_history
        packet=pack({'version':self.version if material_version is None else material_version,'case_routes':[{k:r[k] for k in ('domain','family','cases')}
                                                         for r in navigation['case_counts']],
            'task_contract_ref':self.session.task_specification['id'] if self.session.task_specification else None,
            'interpretation':'Conditional IF material, not current state. Index is a hint, not exhaustive; query cases for finer historical detail.',
            'episode_start_reference':self.session.body_prior['data'].get('episode_start_reference') if self.session.body_prior else None},
            knowledge,episodes,amendment_view(self.session))
        return add_history(packet,history)

    def view(self):
        return {'version':self.version,'active_knowledge_ids':self.default_ids,'history_refs':[{'id':i,'revision':self.history_versions[i]} for i in self.history_ids],'publication':self.publication}
