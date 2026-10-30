"""Knowledge-first, structured historical reads; exact cases and evidence are explicit drill-downs."""
from __future__ import annotations
import json
from . import library

def function(name,description,properties,required):
    return {'type':'function','function':{'name':name,'description':description,
        'parameters':{'type':'object','properties':properties,'required':required,'additionalProperties':False}}}

def query_properties():
    return {'mode':{'type':'string','enum':['search','browse'],'description':'browse lists an explicit domain/family without lexical matching; use when navigating a known small category or query/source languages differ. No automatic scope relaxation.'},'view':{'type':'string','enum':['auto','statement','detail'],'description':'Knowledge text search delivers complete statements plus bounded navigation. For knowledge detail select exact IDs or a record continuation. Detail adds provenance for a specific source question or required write permission. Explicit statement stays statement when paged; no view drops conditions or grants raw permission.'},
        'continuation':{'type':'string','description':'Pass candidate_continuation or selection_continuation here for the remaining fixed selection; omit conditions/scope and keep its view. Whole requested records are returned before remaining ones; unread qualifications are not delivered. A navigation.continuation reads that single immutable statement. A statement/detail page token continues its same view; a complete statement token explicitly drills into optional detail. Question may be omitted. Navigation and partial pages do not grant raw evidence permission.'},
        'question':{'type':'string','description':'Required for text search. Omit for explicit ids/case_refs, a continuation, or structured browse.'},'level':{'type':'string','enum':['knowledge','case','episode','contract']},
        'task':{'type':'string','description':'Exact task identifier for level=episode; not a physical applicability test.'},
        'source_episode':{'type':'string','minLength':1,'maxLength':300,
            'description':'Knowledge text search only: restrict to K whose exact case provenance or direct evidence references this registered source-run ID (episode_candidate.id or case source_episodes). Exact capture only, no ancestor/restart expansion. Source overlap is not relevance or applicability; other sources are not searched. Omit with ids, case_refs, browse, other levels or any continuation.'},
        'domain':{'type':'string','enum':list(library.FAMILIES)},
        'family':{'type':'string','enum':list(sum(library.FAMILIES.values(),()))},
        'roles':{'type':'object','description':'Navigation labels in an explicit family. Knowledge derives them from support cases; unknown old labels remain candidates. Inspect actual role_facets; never guess keys/values.',
                 'additionalProperties':{'type':'string'}},
        'conditions':{'type':'object','description':'Declared applicability context, not measured facts or an episode/task locator. For current unorganized actions use trajectory/segment/evidence tools; a case search cannot retrieve a case that has not been written.',
                      'additionalProperties':{'type':'string'}},
        'case_refs':{'type':'array','maxItems':4,'items':{'type':'object','properties':{
            'id':{'type':'string'},'revision':{'type':'integer'}},'required':['id','revision']}},
        'ids':{'type':'array','items':{'type':'string'},'minItems':1,'maxItems':6,
               'description':'Exact IDs in the selected layer, in requested order; case IDs resolve at the visible snapshot. Use case_refs for pinned case revisions. Do not combine selectors.'}}

def policy_memory_tools():
    from .body_index import tool
    return [tool(),function('query_experience','Use when a historical body relation or physical response could change the next action, '
        'its extent, or its observation boundary: e.g. retention during reorientation, contact-height response, '
        'or a camera/action mapping. Ask one specific question; no Sol call or motion. '
        'Default knowledge; level=case drills into conditional observations when finer detail is needed or knowledge is absent. '
        'Default view=statement preserves text, IF conditions and limitations without full provenance. Use view=detail '
        'or a continuation when exact source history is needed; navigation previews are not statements. '
        'Case source_episode_results link exact whole-trace outcomes, not stage validation; read level=episode with source_episodes IDs. The memory index is a hint, not an exhaustive allowlist. Do not query for current pixels/depth/pose, '
        'facts already supplied, or an action adequately determined by current observations. '
        'Use domain alone if the family is uncertain; contact_response covers action-induced object responses, '
        'retention_release covers holding/release evidence. Results are candidates: check the object, action and conditions; '
        'a mismatched object or sequence does not answer the question. Deliberately revise the scope when needed. '
        'domain/family/roles are structured filters; inspect actual role_facets before using role filters.',
        query_properties(),[]),
        function('manage_memory','Ask Sol/xhigh when a consequential question needs comparison or reinterpretation of cases, '
        'a conflict or changed conditions need review, or a useful batch of success/failure experience needs organization. '
        'State the small uncertainty, not a source-ID list. Use direct query for already-organized facts and measurement '
        'for current geometry. Manager returns evidence, limits and updates, never EEF targets. '
        'Sol rounds are recorded separately from primary calls; missing knowledge alone is not a reason to invent a rule.',
        {'question':{'type':'string'},'purpose':{'type':'string','enum':['review','learn']},
         'conditions':{'type':'object','additionalProperties':{'type':'string'}},
         'episode_outline':outline_schema()},['question','purpose'])]

def outline_schema():
    return {'type':'array','maxItems':6,'description':'Terminal only, before result delivery: brief whole-process interpretation, including grasp/relative pose, downstream transformations and gaps. Order by first receipt; spans may overlap for concurrent work and are not an executable sequence. No need to repeat raw poses. Empty is valid; at most 4096 UTF-8 bytes of compact serialized stages, excluding outer tool/event/context envelopes.',
        'items':{'type':'object','properties':{'summary':{'type':'string'},'first_receipt':{'type':'string'},
            'last_receipt':{'type':'string'},'conditions':{'type':'object','additionalProperties':{'type':'string'}},
            'unknowns':{'type':'array','items':{'type':'string'}}},
            'required':['summary','first_receipt','last_receipt','conditions','unknowns'],'additionalProperties':False}}

def patch_schema():
    return {'type':'array','items':{'type':'object','properties':{
        'op':{'type':'string','enum':['new','support','revise','counterexample']},
        'local':{'type':'string','description':'New case handle, e.g. c1; not stored. Required for a new case.'},
        'claim_id':{'type':'string'},
        'reading_role':{'type':'string','enum':['intrinsic_candidate','binding_geometry','scene_observation','interaction']},
        'kind':{'type':'string','enum':['body','external']},
        'module':{'type':'string','enum':list(sum(library.FAMILIES.values(),()))},
        'observed_entity':{'type':'string','enum':['robot_part','tool_part','scene_object','unknown']},
        'observable':{'type':'string','enum':['configuration','relative_geometry','pixel_position','visibility','local_response','interaction_effect','unknown']},
        'relation_type':{'type':'string','enum':['observed_interaction']},
        'pending_event_id':{'type':'string','description':'Optional exact registered host pending-event ID, not an observation/receipt. Omit for historical receipt-backed interactions; retain relation_type=observed_interaction and source evidence.'},'text':{'type':'string','maxLength':1800,'description':'Conditional case: at most1800 Unicode characters, not a raw pose table.'},
        'conditions':{'type':'object','additionalProperties':{'type':'string'}},
        'reference_frame':{'type':'string','enum':['image','world','tool_local','grasp']},
        'depends_on':{'type':'array','description':'Source dependency field names, e.g. body, cameras or depth; never hash values.','items':{'type':'string'}},
        'epistemic':{'type':'string','enum':['visually_inferred','telemetry_inferred','hypothesis']},
        'evidence':{'type':'array','items':{'type':'string'}},
        'receipt_id':{'type':'string','description':'Select the single actually read, executed receipt witnessing this interaction, with its true before and exact registered post in evidence; this selection does not certify an entire workflow.'},
        'post_observation_id':{'type':'string','description':'Use that receipt\'s exact post observation; for an external interaction, every cited surface inspection must be captured at this post or the same receipt\'s before, so measurements from other stages need their own correctly witnessed cases.'},
        'navigation':{'type':'object','description':'Optional source-backed navigation labels, not proof of operation success. Missing labels stay unknown.',
            'properties':{k:{'type':'string','maxLength':100} for k in ('object','part','operation')},'additionalProperties':False}},
        'required':['op','reading_role','kind','module','text','conditions','reference_frame','epistemic','evidence']}}

def knowledge_schema():
    return {'type':'array','items':{'type':'object','properties':{
        'op':{'type':'string','enum':['add','revise','retire','append_refs'],'description':'append_refs supplies only id and new refs: host preserves meaning, parameters and existing refs. Exact duplicate is a no-op; changed source version/relation requires revise. add/revise supply the full record.'},'id':{'type':'string','description':'Existing knowledge ID. For append_refs, a complete current statement or detail in this transaction is sufficient for the target; each new source still needs its own required read permissions. For revise/retire, first read the complete current view=detail, including every page. Appending references cannot change meaning or existing sources.'},'local':{'type':'string','description':'New knowledge handle k1; usable by later knowledge refs and default selection in this batch.'},'reason':{'type':'string'},
        'domain':{'type':'string','enum':list(library.FAMILIES)},
        'family':{'type':'string','enum':list(sum(library.FAMILIES.values(),()))},
        'question':{'type':'string','maxLength':600,'description':'At most600 UTF-8 bytes; host checks bytes.'},'text':{'type':'string','maxLength':1800,'description':'At most1800 UTF-8 bytes; host checks bytes.'},'conditions':{'type':'object','maxProperties':5,'additionalProperties':{'type':'string'}},
        'strength':{'type':'string','enum':['configuration','observation','hypothesis']},
        'use':{'type':'string','maxLength':900,'description':'The specific decision or query influenced; at most900 UTF-8 bytes, not future EEF targets.'},
        'unknowns':{'type':'array','items':{'type':'string'}},
        'params':{'type':'array','items':{'type':'object','properties':{
            'name':{'type':'string'},'value':{'anyOf':[{'type':'number'},{'type':'array','items':{'type':'number'}}]},
            'unit':{'type':'string'},'frame':{'type':'string'},'source':{'type':'string','description':'Exact supporting evidence/case id or local case handle from refs; host resolves it. Navigation references cannot supply parameter values.'}},
            'required':['name','value','unit','frame','source']}},
        'depends_on':{'type':'array','description':'Source dependency field names, e.g. body, cameras or depth; never hash values.','items':{'type':'string'}},
        'refs':{'type':'array','items':{'type':'object','properties':{
            'type':{'type':'string','enum':['case','knowledge','evidence'],'description':'Knowledge refs require relation=related and a specific reason: navigation only, no support, dependency pins or parameter source. Related case refs are also navigation. Direct evidence must match the supporting-case contract; read permission and compatibility are independent.'},'id':{'type':'string'},
            'local':{'type':'string','description':'Use local instead of id for a newly created case or knowledge item.'},
            'relation':{'type':'string','enum':['support','counter','related'],'description':'Required for case and knowledge references; knowledge must use related.'},
            'reason':{'type':'string','minLength':1,'description':'Required for case and knowledge references: why this exact source supports, counters or only relates to this statement. Knowledge is navigation only.'},
            'unverified_conditions':{'type':'array','items':{'type':'string'}},'resolution':{'type':'string'}},'required':['type'],
            'allOf':[{'if':{'properties':{'type':{'enum':['case','knowledge']}},'required':['type']},
                      'then':{'required':['relation','reason']}},
                     {'if':{'properties':{'type':{'const':'knowledge'}},'required':['type']},
                      'then':{'properties':{'relation':{'const':'related'}}}}]}}},'required':['op']}}

def conditional_card(item,store,dependencies):
    q=item['qualification']
    nav=store.db.execute('SELECT domain,family,roles FROM case_navigation WHERE revision=?',(item['revision'],)).fetchone()
    provenance=store.provenance_at(item['revision'])
    return {'id':item['id'],'revision':item['revision'],'layer':'case',
        'domain':nav['domain'],'family':nav['family'],'roles':json.loads(nav['roles']),
        'status':item['status'],
        'dependencies':item['dependencies'],'source_episodes':item['grounding']['source_episodes'],
        'text':item['text'],'conditions':item['conditions'],'strength':item['epistemic'],
        'applicability':{**{k:q.get(k) for k in ('category','metric_qualification','unknown_conditions','different_conditions',
            'different_descriptions','changed_derivation_contracts','body_and_binding_comparison',
            'geometry_equivalence','current_fact','source_interpretation_revised')},
            'changed_dependencies':[k for k,v in item['dependencies'].items() if not library.unknown_value(v)
                and not library.unknown_value(dependencies.get(k)) and dependencies[k]!=v],
            'unknown_dependencies':[k for k,v in item['dependencies'].items()
                if library.unknown_value(v) or library.unknown_value(dependencies.get(k))]},
        'review':item.get('review',{}).get('reason'),
        'counterexamples':[{k:r.get(k) for k in ('revision','text','conditions','evidence')} for r in provenance['counterexamples']],
        'drill_down':{'evidence':item['evidence'],'qualification':'Historical source references, not newly read pixels'}}

def read_experience(session,question=None,conditions=None,*,level='knowledge',domain=None,family=None,roles=None,
                    case_refs=None,ids=None,limit=None,budget_bytes=6000,frontier=None,revision=None,knowledge_revision=None,view="auto",continuation=None,version_snapshot=None,mode="search",task=None,reader_owner=None,track_delivery=True,complete_only=False,source_episode=None):
    if source_episode is not None:
        if (not isinstance(source_episode,str) or not source_episode.strip()
                or source_episode!=source_episode.strip() or len(source_episode)>300):
            raise ValueError('source_episode requires one exact registered source-run ID')
        if level!='knowledge' or mode!='search' or any(v is not None for v in (ids,case_refs,continuation)):
            raise ValueError('source_episode is only for knowledge text search without exact selectors or continuation')
    reader_owner=reader_owner or 'policy:'+session.episode
    selectors=sum(v is not None for v in (ids,case_refs,continuation))
    if selectors>1:raise ValueError('Choose only one of ids, case_refs or continuation')
    if case_refs is not None and level!='case':raise ValueError('case_refs requires level=case')
    if (ids is not None or case_refs is not None) and any(x is not None for x in (domain,family,roles)):
        raise ValueError('Exact record selectors cannot be combined with domain/family/roles filters')
    if mode=='browse' and selectors:raise ValueError('Browse and exact record selectors are separate operations')
    if ids is not None and (not isinstance(ids,list) or not 1<=len(ids)<=6 or
            any(not isinstance(i,str) or not i for i in ids) or len(set(ids))!=len(ids)):
        raise ValueError('Provide one to six distinct exact IDs')
    if case_refs is not None and (not isinstance(case_refs,list) or not 1<=len(case_refs)<=4 or
            any(not isinstance(r,dict) or set(r)!={'id','revision'} or not isinstance(r['id'],str) or
                type(r['revision']) is not int or r['revision']<1 for r in case_refs) or
            len({r['id'] for r in case_refs})!=len(case_refs)):
        raise ValueError('Provide one to four distinct exact case ID/revision pairs')
    exact_count=len(ids if ids is not None else case_refs) if ids is not None or case_refs is not None else None
    if exact_count is not None and limit is not None and limit<exact_count:
        raise ValueError('Exact record selection cannot be truncated by limit')
    if limit is None:limit=exact_count or 4
    if continuation is not None and question is None:question='Continue selected immutable record'
    if (ids is not None or case_refs is not None) and question is None:question='Read selected immutable records'
    if mode=='browse' and (question is None or question==''):question=f'Browse {domain}/{family}'
    if not isinstance(question,str) or not 0<len(question.strip())<=1200:raise ValueError('One specific question required for search')
    from .readout import pack, page, size
    def selection_scope():
        from .hybrid import scope_state
        current_frontier=session.source_frontier() if frontier is None else frontier
        current_case=(session.frozen_revision if session.frozen_revision is not None else session.store.revision) if revision is None else revision
        current_knowledge=(session.frozen_knowledge_revision if session.frozen_knowledge_revision is not None else
            session.store.db.execute('SELECT COALESCE(MAX(revision),0) FROM reusable_knowledge').fetchone()[0]) if knowledge_revision is None else knowledge_revision
        import copy
        return copy.deepcopy({'owner':reader_owner,'episode':session.episode,'profile':session.perception.name,
            'frontier':current_frontier,'case_at':current_case,'knowledge_at':current_knowledge,
            'snapshot':version_snapshot,'context':session.context,'dependencies':session.dependencies,
            'stamp':scope_state(session,frontier=current_frontier,
                case_at=current_case,knowledge_at=current_knowledge,snapshot=version_snapshot)})
    if continuation is not None:
        entry=getattr(session,'_read_continuations',{}).get(continuation)
        if entry and entry[0].get('view')=='record_selection':
            if any(v is not None for v in (conditions,domain,family,roles,ids,case_refs,task)):
                raise ValueError('Record selection freezes its question/conditions/scope; provide only its token')
            chosen,offset=entry
            if chosen['scope']['owner']!=reader_owner:
                raise ValueError('Record selection belongs to another reader transaction')
            current_scope=selection_scope()
            if any(current_scope['frontier'].get(run,-1)<through for run,through in chosen['scope']['frontier'].items()):
                raise ValueError('stale_selection: original source frontier was narrowed; issue a new question')
            if chosen['scope']['stamp']!=current_scope['stamp']:
                raise ValueError('stale_selection: context, dependencies or visible snapshot changed; issue a new question')
            if view!='auto' and view!=chosen['requested_view']:
                raise ValueError('Record selection cannot change its requested view')
            answer=pack(session,chosen['records'],chosen['answer'],budget_bytes=budget_bytes,limit=limit,
                view=chosen['requested_view'],selection_scope=chosen['scope'],offset=offset,
                complete_only=complete_only)
            answer.pop('read_handle',None)
            ref=session.trace.event('experience_query',{'question':chosen['answer'].get('question'),
                'answer':answer,'reader_owner':reader_owner,'selection_continuation':continuation,
                'consumer_delivery_deferred':not track_delivery})
            answer['read_handle']=ref['id']
            if size(answer)>budget_bytes:raise ValueError('Read budget cannot fit final handle')
            from .readout import note_delivery
            if track_delivery:note_delivery(session,reader_owner,answer)
            return answer,[]
        if entry and entry[0].get('view')=='candidate_set':
            from . import hybrid
            if any(v is not None for v in (conditions,domain,family,roles,ids,case_refs)):
                raise ValueError('Candidate continuation freezes its question/conditions/scope; provide only its token')
            chosen,offset=entry
            current_frontier=session.source_frontier() if frontier is None else frontier
            current_case=(session.frozen_revision if session.frozen_revision is not None else session.store.revision) if revision is None else revision
            current_knowledge=(session.frozen_knowledge_revision if session.frozen_knowledge_revision is not None else
                session.store.db.execute('SELECT COALESCE(MAX(revision),0) FROM reusable_knowledge').fetchone()[0]) if knowledge_revision is None else knowledge_revision
            hybrid.validate_selection(session,chosen,owner=reader_owner,frontier=current_frontier,
                case_at=current_case,knowledge_at=current_knowledge,snapshot=version_snapshot)
            return hybrid.deliver(session,chosen,offset=offset,budget_bytes=budget_bytes,limit=limit,track_delivery=track_delivery)
        envelope={'question':question,'status':'page_complete','records':[]}
        result=page(session,continuation,budget_bytes-size(envelope)-4)
        session.trace.event('experience_query',{'question':question,'answer':result})
        answer={'question':question,'status':result['status'],'records':[result]}
        if size(answer)>budget_bytes:raise ValueError('Read budget cannot fit continuation envelope')
        from .readout import note_delivery
        if track_delivery:note_delivery(session,reader_owner,answer)
        return answer,[]
    conditions={} if conditions is None else conditions
    if not isinstance(conditions,dict) or any(not isinstance(v,str) for v in conditions.values()):raise ValueError('String conditions required')
    if roles is not None and (not isinstance(roles,dict) or any(not isinstance(v,str) for v in roles.values())):raise ValueError('String roles required')
    library.initialize(session.store);library.route(domain,family)
    if mode not in ('search','browse'):raise ValueError('mode must be search or browse')
    if mode=='browse' and level not in ('episode','contract') and (domain is None or family is None):raise ValueError('browse requires explicit domain and family')
    retrieval_question='' if mode=='browse' else question
    frontier=session.source_frontier() if frontier is None else frontier
    revision=(session.frozen_revision if session.frozen_revision is not None else session.store.revision) if revision is None else revision
    if level=='case' and ids is not None:
        exact=[]
        for identifier in ids:
            row=library.latest_case(session.store,identifier,revision,frontier,version_snapshot)
            if row is None:raise ValueError('Case ID not visible at this snapshot: '+identifier)
            exact.append({'id':identifier,'revision':row['revision']})
        case_refs=exact
    context={**session.context,**conditions}
    if task is not None and level!='episode':raise ValueError('task filter requires level=episode')
    if roles and (level not in ('case','knowledge') or family is None):
        raise ValueError('Role filters require a knowledge/case family; inspect its role_facets first')
    if level=='contract':
        if domain is not None or family is not None or roles or conditions:raise ValueError('Contract reads use the current public contract and explicit ids or text; no physical case filters')
        from .contracts import amendment_cards
        from .retrieval import content_terms
        terms=set(content_terms(retrieval_question)) if ids is None else set()
        records=[c for c in amendment_cards(session,frontier) if (ids is None or c['id'] in ids) and
            (not terms or terms & set(content_terms(c['question']+' '+c['answer'])))]
        if ids is not None:records.sort(key=lambda c:ids.index(c['id']))
        if mode=='browse':
            from .contracts import current_contract
            records=[{'id':current_contract(session) or 'no_current_contract','revision':max((c['revision'] for c in records),default=0),
                'layer':'public_contract_directory','entries':[{k:c[k] for k in ('id','question','reuse_scope')} for c in records],
                'qualification':'Complete immutable directory at this source frontier. Entries are pointers, not read amendment content; use exact ids for answers.'}]
        else:records=records[:limit]
        selected=[];diagnostic={'layer':'public_contract'}
    elif level=='episode':
        if domain is not None or family is not None or roles or conditions:raise ValueError('Episode candidates use exact task/ids or a text question, not physical case filters')
        from .episodes import read
        from .contracts import current_contract
        records=read(session.store,task=task,question='' if ids is not None or mode=='browse' else retrieval_question,
                     ids=ids,frontier=frontier,limit=limit,dependencies=session.dependencies,
                     contract_id=current_contract(session),process_only=ids is None)
        selected=[];diagnostic={'layer':'episode_candidate'}
    elif level=='knowledge':
        if ids is None and mode=='search':
            if view=='detail':raise ValueError('Knowledge search delivers statements; use an exact ID or record continuation for detail')
            from . import hybrid
            current_knowledge=(session.frozen_knowledge_revision if session.frozen_knowledge_revision is not None else
                session.store.db.execute('SELECT COALESCE(MAX(revision),0) FROM reusable_knowledge').fetchone()[0]) if knowledge_revision is None else knowledge_revision
            chosen=hybrid.selection(session,retrieval_question,conditions,context=context,
                frontier=frontier,case_at=revision,knowledge_at=current_knowledge,snapshot=version_snapshot,
                domain=domain,family=family,roles=roles,owner=reader_owner,
                **({'source_episode':source_episode} if source_episode is not None else {}))
            return hybrid.deliver(session,chosen,offset=0,budget_bytes=budget_bytes,limit=limit,track_delivery=track_delivery)
        records=library.read_knowledge(session.store,retrieval_question,context=context,dependencies=session.dependencies,
            domain=domain,family=family,roles=roles,ids=ids,limit=limit,at=session.frozen_knowledge_revision if knowledge_revision is None else knowledge_revision,case_at=revision,frontier=frontier,snapshot=version_snapshot)
        if view=='detail':
            for record in records:
                record['drill_down']['source_coverage']=library.source_coverage(session.store,record['drill_down']['refs'])
                basis=session.store.db.execute('SELECT basis FROM knowledge_contract_basis WHERE revision=?',(record['revision'],)).fetchone()
                if basis:record['drill_down']['configuration_basis']=json.loads(basis[0])
        selected=[];diagnostic={'layer':'knowledge'}
    elif level=='case':
        if roles:
            nav=library.navigation(session.store,domain=domain,family=family,at=revision,
                                   frontier=frontier,requested_roles=roles)
            invalid=[]
            for key,value in roles.items():
                facet=nav['role_facets'].get(key)
                if facet is None:invalid.append({'key':key,'value':value,'reason':'unknown_role_key'})
                elif value not in {v['value'] for v in facet['values']}:
                    invalid.append({'key':key,'value':value,'reason':'unknown_role_value'})
            if invalid:
                answer={'question':question,'level':'case','status':'invalid_navigation','records':[],
                        'more_available':False,'invalid_filters':invalid,'navigation':nav,
                        'interpretation':'Filters were not applied. Select actual navigation values; no filter was dropped and lexical search was not attempted.'}
                ref=session.trace.event('experience_query',{'question':question,'conditions':conditions,
                    'source_frontier':frontier,'readout':{'retrieval_mode':'invalid_navigation'},'answer':answer})
                answer['read_handle']=ref['id']
                return answer,[]
        result=session.store.read(retrieval_question,context=context,dependencies=session.dependencies,episode=session.episode,
            at=revision,limit=limit,evidence_cutoff=frontier,domain=domain,family=family,roles=roles,
            structured=True,include_unresolved=True,claim_refs=case_refs,browse=mode=='browse')
        pool=result['selected']+result['historical_events']+result['review_required']
        def rank(item):
            q=item['qualification']
            qualification=(q['category']!='conditional_candidate',bool(q['different_conditions']),
                    len(q['unknown_conditions']),-item['revision'])
            if mode=='search' and case_refs is None and retrieval_question:
                return (item['retrieval_score_bm25'], *qualification)
            return (qualification[0], qualification[1], item['retrieval_score_bm25'], *qualification[2:])
        if case_refs is not None:
            order={ref['id']:i for i,ref in enumerate(case_refs)}
            selected=sorted(pool,key=lambda item:order[item['id']])
        else:selected=sorted(pool,key=rank)[:limit]
        records=[conditional_card(item,session.store,session.dependencies) for item in selected]
        from .episodes import read as read_episodes
        for record in records:
            sources=record['source_episodes']
            if not sources:continue
            historical=read_episodes(session.store,ids=sources,frontier=frontier,limit=len(sources),
                exclude_run=session.episode,dependencies=session.dependencies)
            if historical:
                record['source_episode_results']=[{'id':e['id'],'official_success':e['official_success'],
                    'result_ref':e['trace_entry']['result']} for e in historical]
                record['episode_result_scope']='Whole-trace labels only; not validation of this case, a stage or a rule. Read level=episode with exact source IDs for history; no raw-read permission.'
        diagnostic=result
    else:raise ValueError('Read level must be knowledge or case')
    if ids is not None or case_refs is not None:
        visible={(r['id'],r['revision']) for r in records}
        missing=([identifier for identifier in ids if not any(i==identifier for i,v in visible)] if level in ('knowledge','episode','contract')
                 else [ref for ref in case_refs if (ref['id'],ref['revision']) not in visible])
        if missing:raise ValueError('Exact selectors not visible at this snapshot: '+json.dumps(missing))
    answer={'question':question,'level':level,'scope':{'domain':domain,'family':family,'roles':roles or {},'mode':mode},'records':[],'more_available':False,
        'interpretation':'Candidate historical material, not a certified answer. Check object, action and IF scope; revise query scope if mismatched. Current observations take precedence.'}
    if mode=='browse':answer['interpretation']='Explicit category listing, not an answer to a physical question. Check conditions and qualifiers before use.'
    if not records and level=='case':answer['reason']=diagnostic['retrieval_mode']
    # Reserve the actual handle field in the same serialized envelope that is fitted.
    answer['read_handle']=session.episode+':e'+'9'*20
    if level=='case':answer['navigation_available']=True  # reserve any final navigation fallback before packing
    answer=pack(session,records,answer,budget_bytes=budget_bytes,limit=limit,view=view,
        selection_scope=selection_scope(),complete_only=complete_only)
    complete={c['id'] for c in answer['records'] if c.get('complete_record')}
    delivered=[item for item in selected if item['id'] in complete]
    if not answer['records'] or (level=='case' and family is not None):
        nav=library.navigation(session.store,domain=domain,family=family,at=revision,frontier=frontier,requested_roles=roles)
        proposed={**answer,'navigation':nav};proposed.pop('navigation_available',None)
        if size(proposed)<=budget_bytes:answer=proposed
    if size(answer)>budget_bytes:raise ValueError('Read budget cannot fit final navigation envelope; narrow the selection')
    del answer['read_handle']
    ref=session.trace.event('experience_query',{'question':question,'conditions':conditions,'source_frontier':frontier,
        'readout':diagnostic,'answer':answer,'reader_owner':reader_owner,'consumer_delivery_deferred':not track_delivery})
    answer['read_handle']=ref['id']
    if size(answer)>budget_bytes:raise ValueError('Read budget cannot fit final handle')
    from .readout import note_delivery
    if track_delivery:note_delivery(session,reader_owner,answer)
    return answer,delivered
