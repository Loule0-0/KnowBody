"""Visible candidates, two retrieval branches, and bounded complete statements."""
from __future__ import annotations

import copy
import hashlib
import json

from . import library, readout
from .retrieval import content_terms

DEPTH = 20
SCHEMA = 'knowledge-static-v1'
POLICY = 'bm_first-interleave20-statement-v1'


def index_text(row):
    """Static allowed semantics only; no current qualifications or audit hashes."""
    params=[{k:p[k] for k in ('name','value','unit','frame')} for p in json.loads(row['params'])]
    return json.dumps({'question':row['question'],'text':row['text'],'use':row['use_hint'],
        'conditions':json.loads(row['conditions']),'unknowns':json.loads(row['unknowns']),
        'parameters':params},ensure_ascii=False,sort_keys=True,separators=(',',':'))


def interleave(bm, dense):
    result=[];seen=set()
    for offset in range(max(len(bm),len(dense))):
        for branch in (bm,dense):
            if offset<len(branch) and branch[offset] not in seen:
                key=branch[offset];seen.add(key);result.append(key)
    return result


def scope_state(session, *, frontier, case_at, knowledge_at, snapshot):
    # A recorded request or lookup lead is not its execution or a new observed
    # fact. Real dispatch records policy_decision before every query, too. The
    # original full frontier still gates every frozen candidate and source;
    # measurements/plans/receipts/observations and configuration changes remain
    # in this stamp, along with independently checked context and dependencies.
    # JSON1 is already required by the evidence store. One scan avoids a full
    # evidence-table scan per historical run when only the primary ID is indexed.
    evidence_frontier=dict(session.store.db.execute(
        "SELECT run,MAX(sequence) FROM evidence WHERE kind NOT IN ('lookup_lead','policy_decision') "
        "AND sequence<=json_extract(?, '$.'||json_quote(run)) GROUP BY run",
        (json.dumps(frontier),)))
    value={'episode':session.episode,'profile':session.perception.name,
        'context':session.context,'dependencies':session.dependencies,'frontier':evidence_frontier,
        'case_at':case_at,'knowledge_at':knowledge_at,'snapshot':snapshot}
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True).encode()).hexdigest()


def selection(session, question, conditions, *, context, frontier, case_at, knowledge_at,
              snapshot, domain, family, roles, owner, source_episode=None):
    source_ids=None;source_counts={}
    if source_episode is not None:
        source_ids={r[0] for r in session.store.db.execute(
            'SELECT id FROM evidence WHERE run=? AND sequence<=?',
            (source_episode,frontier.get(source_episode,-1)))}
    rows=(library.visible_knowledge(session.store,domain=domain,family=family,roles=roles,
        at=knowledge_at,case_at=case_at,frontier=frontier,snapshot=snapshot)
        if source_ids is None or source_ids else [])
    if source_ids is not None:
        source_counts=library.knowledge_source_counts(session.store,rows,source_ids)
        rows=[row for row in rows if (row['id'],row['revision']) in source_counts]
    keys={(row['id'],row['revision']):row for row in rows}
    bmrows=(library.lexical_knowledge(session.store,rows,question,context=context,
        dependencies=session.dependencies,roles=roles,limit=DEPTH) if content_terms(question) else [])
    bm=[(row['id'],row['revision']) for row in bmrows]
    dense=[];stats={'documents':0,'encoded_texts':0,'cached_texts':0}
    if rows:
        physical={k:v for k,v in context.items() if k not in ('task','phase','episode')}
        query=question+('\nDeclared conditions, not measurements: '+json.dumps(physical,ensure_ascii=False,sort_keys=True)
                        if physical else '')
        dense,stats=session.get_retrieval_index().rank(query,
            {key:index_text(row) for key,row in keys.items()},schema=SCHEMA,limit=DEPTH)
    if not set(bm+dense)<=set(keys):raise ValueError('Retriever returned a version outside its authorized visible set')
    ordered=interleave(bm,dense)
    rank_b={key:i+1 for i,key in enumerate(bm)};rank_d={key:i+1 for i,key in enumerate(dense)}
    chosen={'view':'candidate_set','owner':owner,'policy_id':POLICY,'question':question,
        'conditions':copy.deepcopy(conditions),'context':copy.deepcopy(context),
        'frontier':copy.deepcopy(frontier),'case_at':case_at,'knowledge_at':knowledge_at,
        'snapshot':copy.deepcopy(snapshot),'scope':{'domain':domain,'family':family,'roles':roles or {},'mode':'search'},
        'stamp':scope_state(session,frontier=frontier,case_at=case_at,knowledge_at=knowledge_at,snapshot=snapshot),
        'ordered':[{'id':key[0],'revision':key[1],'bm_rank':rank_b.get(key),'dense_rank':rank_d.get(key),
                    **({'source_relation_counts':{'source_episode':source_episode,**source_counts[key]}}
                       if source_episode is not None else {})} for key in ordered],
        'delivered':[],'navigated':[],'only_exact':[],'encoder_stats':stats}
    if source_episode is not None:
        chosen['scope']['source_episode']=source_episode
        chosen['source_found']=bool(source_ids)
        chosen['source_has_reference']=bool(rows)
    return chosen


def validate_selection(session, chosen, *, owner, frontier, case_at, knowledge_at, snapshot):
    if chosen['owner']!=owner:raise ValueError('Candidate continuation belongs to another reader transaction')
    if any(frontier.get(run,-1)<through for run,through in chosen['frontier'].items()):
        raise ValueError('stale_selection: original source frontier was narrowed; issue a new question')
    if chosen['stamp']!=scope_state(session,frontier=frontier,case_at=case_at,
                                    knowledge_at=knowledge_at,snapshot=snapshot):
        raise ValueError('stale_selection: context, dependencies or visible snapshot changed; issue a new question')
    for ref in chosen['ordered']:
        row=session.store.db.execute('SELECT * FROM reusable_knowledge WHERE id=? AND revision=?',
                                     (ref['id'],ref['revision'])).fetchone()
        if row is None or not library.knowledge_visible(session.store,row,chosen['case_at'],chosen['knowledge_at'],
                                                         chosen['frontier'],snapshot=chosen['snapshot']):
            raise ValueError('Candidate continuation source no longer passes its original visibility gate')


def render(session, chosen, *, offset, budget_bytes, limit):
    """Search never fragments a statement. Exact reads still page losslessly."""
    ordered=chosen['ordered'];past={tuple(k) for k in chosen['delivered']}
    prior_navigation=set(chosen.get('navigated',[]))
    cache={};statements={}

    def record(index):
        if index not in cache:
            ref=ordered[index]
            row=session.store.db.execute('SELECT * FROM reusable_knowledge WHERE id=? AND revision=?',
                                          (ref['id'],ref['revision'])).fetchone()
            card=library.knowledge_view(session.store,row,chosen['context'],session.dependencies,
                case_at=chosen['case_at'],knowledge_at=chosen['knowledge_at'],frontier=chosen['frontier'],snapshot=chosen['snapshot'])
            card['retrieval']={k:ref[k] for k in ('bm_rank','dense_rank')}
            if 'source_relation_counts' in ref:
                card['source_relation_counts']=copy.deepcopy(ref['source_relation_counts'])
            for counter in card.get('counter_sources',[]):
                counter['statement_delivered_to_reader']=readout.delivered_statement(session,chosen['owner'],
                    counter['type'],counter['id'],counter['revision'])
            cache[index]=card
        return cache[index]

    def statement(index):
        if index not in statements:
            card=record(index)
            result=readout.statement(card)
            result['continuation']=readout._continuation(session,{**copy.deepcopy(card),'view':'detail','complete_record':True})
            statements[index]=result
        return statements[index]

    def navigation(index, oversized=False):
        card=record(index);ref=ordered[index];q=card['applicability']
        flags=[k for k in ('source_contested','source_revised','related_warnings') if card.get(k)]
        title=card['question'][:100]
        return {**ref,'question':title,'question_complete':title==card['question'],
            'statement_bytes':readout.size(statement(index)),
            'changed_pins':q.get('changed_dependencies',[]),'unknown_conditions':len(q.get('unknown_conditions',[])),
            'source_flags':flags,'explicit_counter_refs':card['counter_cases'],
            'view':'navigation','statement_complete':False,'complete_record':False,
            **({'exact_read_required':True} if oversized else {}),
            'continuation':readout._continuation(session,{**statement(index),'view':'statement'})}

    base={'question':chosen['question'],'level':'knowledge','scope':chosen['scope'],
        'records':[],'navigation':[],'more_available':False,'selection_policy':POLICY,
        'interpretation':'Ranked historical candidates, not verified relevance or applicability. Declared conditions are not measurements; explicit counter references are not exhaustive or reviewed.',
        'read_handle':session.episode+':e'+'9'*20}
    if 'source_episode' in chosen['scope']:
        base['interpretation']+=' Only this source and requested scope were searched. Ref-role counts are mechanical provenance overlap, not independent trials, relevance, applicability or stage validation.'
        if not chosen['source_found']:base['reason']='source_not_found'
        elif not chosen['source_has_reference']:base['reason']='no_source_reference_in_requested_scope'

    def envelope(full, cursor, oversized, *, extra=False,navigation_covered=()):
        done=past|set(navigation_covered)|{(c['id'],c['revision']) for c in full}
        remaining=[i for i,r in enumerate(ordered) if (r['id'],r['revision']) not in done]
        heads=[next((i for i in remaining if ordered[i][branch]==1 and i not in prior_navigation),None)
               for branch in ('bm_rank','dense_rank')]
        wanted=list(dict.fromkeys(i for i in heads if i is not None))
        if extra:
            wanted=list(dict.fromkeys([*wanted,*(i for i in remaining if i>=offset and i not in prior_navigation)]))[:3]
        nav=[navigation(i,i in oversized) for i in wanted]
        answer={**base,'records':full,'navigation':nav,'omitted_candidates':max(0,len(remaining)-len(nav)),
            'more_available':bool(remaining),'status':'records_found' if full else 'navigation_only' if nav else 'no_match'}
        if remaining:answer['candidate_continuation']='x'*32
        return answer

    full=[];cursor=offset;oversized=set(chosen.get('only_exact',[]));new_oversized=0
    while cursor<len(ordered) and len(full)<limit:
        ref=ordered[cursor]
        if (ref['id'],ref['revision']) in past:cursor+=1;continue
        card=statement(cursor)
        # Empty response test distinguishes intrinsically oversized from merely
        # losing remaining room. Only the former may be skipped as navigation.
        if readout.size(envelope([card],cursor+1,oversized,
                navigation_covered={(c['id'],c['revision']) for c in full}))>budget_bytes:
            oversized.add(cursor);cursor+=1;new_oversized+=1
            if new_oversized==3:break
            continue
        trial=envelope([*full,card],cursor+1,oversized)
        if readout.size(trial)>budget_bytes:break
        full.append(card);cursor+=1
    answer=envelope(full,cursor,oversized)
    if readout.size(answer)>budget_bytes:raise ValueError('Query byte budget cannot fit bounded candidate navigation')
    # Only leftover bytes fund optional navigation. It must never displace a
    # full statement that fits with the actual branch-head minimum.
    extended=envelope(full,cursor,oversized,extra=True)
    for nav in extended['navigation']:
        if any(n['id']==nav['id'] and n['revision']==nav['revision'] for n in answer['navigation']):continue
        trial=copy.deepcopy(answer);trial['navigation'].append(nav);trial['omitted_candidates']-=1
        if readout.size(trial)<=budget_bytes:answer=trial
    nav_keys={(n['id'],n['revision']) for n in answer['navigation']}
    shown=prior_navigation|{i for i,r in enumerate(ordered) if (r['id'],r['revision']) in nav_keys}
    complete_keys=past|{(c['id'],c['revision']) for c in full}
    uncovered=[i for i in range(offset,cursor) if i not in shown and
               (ordered[i]['id'],ordered[i]['revision']) not in complete_keys]
    if uncovered:cursor=min(uncovered)
    answer['omitted_candidates']=sum((r['id'],r['revision']) not in complete_keys and i not in shown
                                     for i,r in enumerate(ordered))
    if cursor<len(ordered):
        following=copy.deepcopy(chosen)
        following['delivered']=[list(k) for k in sorted(past|{(c['id'],c['revision']) for c in full})]
        following['only_exact']=sorted(oversized)
        following['navigated']=sorted(shown)
        answer['candidate_continuation']=readout._continuation(session,following,cursor)
    else:answer.pop('candidate_continuation',None)
    return answer,{'layer':'knowledge','policy_id':POLICY,'ordered_candidates':ordered,
        'cursor_before':offset,'cursor_after':cursor,'oversized_navigation_indices':sorted(oversized),
        'fully_delivered':[{'id':r['id'],'revision':r['revision']} for r in full],
        'encoder':chosen['encoder_stats'],'case_revision':chosen['case_at'],'knowledge_revision':chosen['knowledge_at'],
        'reader_owner':chosen['owner'],'selection_scope_sha256':chosen['stamp']}


def deliver(session,chosen,*,offset,budget_bytes,limit,track_delivery=True):
    answer,audit=render(session,chosen,offset=offset,budget_bytes=budget_bytes,limit=limit)
    del answer['read_handle']
    ref=session.trace.event('experience_query',{'question':chosen['question'],'conditions':chosen['conditions'],
        'source_frontier':chosen['frontier'],'version_snapshot':chosen['snapshot'],
        'readout':audit,'answer':answer,'consumer_delivery_deferred':not track_delivery})
    answer['read_handle']=ref['id']
    if readout.size(answer)>budget_bytes:raise ValueError('Candidate read budget cannot fit committed handle')
    if track_delivery:readout.note_delivery(session,chosen['owner'],answer)
    return answer,[]
