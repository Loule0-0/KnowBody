"""Structured navigation and versioned reusable knowledge in the evidence SQLite authority."""
from __future__ import annotations
import json
import uuid
from collections import Counter
from .memory import encoded
from .retrieval import content_terms
from .qualification import compare_conditions, unknown as unknown_value

FAMILIES={'body':('frame_map','geometry','tool_mount','visual_action'),
          'external':('scene_geometry','contact_response','retention_release')}

def initialize(store):
    if getattr(store,'_library_initialized',False):return
    store.db.executescript('''
      CREATE TABLE IF NOT EXISTS case_navigation(
        revision INTEGER PRIMARY KEY REFERENCES claims(revision),domain TEXT NOT NULL,
        family TEXT NOT NULL,roles TEXT NOT NULL);
      CREATE INDEX IF NOT EXISTS case_route ON case_navigation(domain,family,revision);
      CREATE TABLE IF NOT EXISTS reusable_knowledge(
        revision INTEGER PRIMARY KEY AUTOINCREMENT,id TEXT NOT NULL,domain TEXT NOT NULL,
        family TEXT NOT NULL,question TEXT NOT NULL,text TEXT NOT NULL,conditions TEXT NOT NULL,params TEXT NOT NULL,dependencies TEXT NOT NULL,
        strength TEXT NOT NULL,use_hint TEXT NOT NULL,unknowns TEXT NOT NULL,refs TEXT NOT NULL,
        status TEXT NOT NULL,UNIQUE(id,revision));
      CREATE INDEX IF NOT EXISTS knowledge_route ON reusable_knowledge(domain,family,id,revision);
      CREATE TABLE IF NOT EXISTS case_disposition(
        sequence INTEGER PRIMARY KEY AUTOINCREMENT,request_id TEXT NOT NULL,
        case_id TEXT NOT NULL,case_revision INTEGER NOT NULL REFERENCES claims(revision),
        action TEXT NOT NULL,knowledge_revisions TEXT NOT NULL,reason TEXT NOT NULL,
        UNIQUE(request_id,case_revision));
      CREATE TABLE IF NOT EXISTS publication_groups(
        request_id TEXT NOT NULL,digest TEXT NOT NULL,result TEXT NOT NULL,
        PRIMARY KEY(request_id,digest));
      CREATE TABLE IF NOT EXISTS processed_intervals(
        receipt_id TEXT PRIMARY KEY REFERENCES evidence(id),request_id TEXT NOT NULL,
        disposition TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS knowledge_contract_basis(
        revision INTEGER PRIMARY KEY REFERENCES reusable_knowledge(revision),basis TEXT NOT NULL);
    ''')
    store._library_initialized=True

def classify_archived_case(row):
    """One-time coarse data curation; physical qualifications/evidence never change."""
    text=row['module'].lower()
    subject=(row['text']+' '+row['conditions']).lower()
    entity=row['entity']
    if row['role']=='interaction':
        domain='external'; family='retention_release' if (any(w in text for w in ('retent','handover','rotation','reorient','lift','transport','release')) or ('key' in subject and any(w in subject for w in ('held','suspend','grasp','fingers'))) or ('broom' in subject and any(w in subject for w in ('donor','receiv'))) or ('dustpan' in subject and 'released' in subject)) else 'contact_response'
    elif entity=='scene_object' and row['role']=='scene_observation':
        domain='external';family='scene_geometry'
    else:
        domain='body'
        family=('visual_action' if any(w in text for w in ('camera','wrist','visibility','occlusion')) else
                'tool_mount' if row['role']=='binding_geometry' else
                'frame_map' if 'translation' in text or 'home' in text else 'geometry')
    return domain,family,{'entity':entity or 'unknown','observable':row['observable'] or 'unknown'}

def organize_archive(store):
    initialize(store)
    rows=store.db.execute('SELECT c.*,r.role,g.entity,g.observable FROM claims c '
        'JOIN claim_reading r ON r.revision=c.revision LEFT JOIN claim_grounding g ON g.revision=c.revision').fetchall()
    with store.db:
        for row in rows:
            domain,family,roles=classify_archived_case(row)
            store.db.execute('INSERT OR IGNORE INTO case_navigation VALUES(?,?,?,?)',
                (row['revision'],domain,family,encoded(roles)))
    return {'indexed_case_revisions':len(rows),'physical_claims_changed':False}

def route(domain=None,family=None):
    if domain is not None and domain not in FAMILIES:raise ValueError('Unknown memory domain')
    if family is not None and family not in (FAMILIES[domain] if domain else sum(FAMILIES.values(),())):
        raise ValueError('Unknown memory family')

def navigation(store,*,domain=None,family=None,at=None,frontier=None,requested_roles=None):
    initialize(store)
    route(domain,family)
    at=store.revision if at is None else at
    seen=set();visible=[]
    for row in store.db.execute('SELECT c.id,n.* FROM claims c JOIN case_navigation n ON n.revision=c.revision '
                                'WHERE c.revision<=? ORDER BY c.revision DESC',(at,)):
        if row['id'] in seen:continue
        if frontier is not None:
            sources=store.provenance_at(row['revision'])['evidence']
            if any((ref:=store.db.execute('SELECT run,sequence FROM evidence WHERE id=?',(source,)).fetchone()) is None
                   or ref['sequence']>frontier.get(ref['run'],-1) for source in sources):continue
        seen.add(row['id']);visible.append(row)
    counts=Counter((row['domain'],row['family']) for row in visible)
    result={'families':FAMILIES,'case_counts':[{'domain':d,'family':f,'cases':n} for (d,f),n in sorted(counts.items())],
            'layers':['knowledge','case','evidence']}
    # Role values are an explicit family-level entry, never an unbounded startup catalog.
    if family is not None:
        facets={}
        for row in visible:
            if row['family']!=family or (domain is not None and row['domain']!=domain):continue
            for key,value in json.loads(row['roles']).items():facets.setdefault(key,Counter())[value]+=1
        result['scope']={'domain':domain,'family':family,'case_revision':at}
        result['role_facets']={}
        for key,values in sorted(facets.items()):
            ordered=sorted(values.items(),key=lambda v:(v[0]!=(requested_roles or {}).get(key),-v[1],v[0]))
            result['role_facets'][key]={'values':[{'value':v,'cases':n} for v,n in ordered[:8]],
                                       'other_values':max(0,len(ordered)-8)}
    return result

def source_visible(store,source,frontier):
    row=store.db.execute('SELECT run,sequence FROM evidence WHERE id=?',(source,)).fetchone()
    return row is not None and (frontier is None or row['sequence']<=frontier.get(row['run'],-1))

def case_visible(store,revision,case_at,frontier,snapshot=None):
    row=store.db.execute('SELECT id FROM claims WHERE revision=?',(revision,)).fetchone()
    own=(snapshot or {}).get('cases',{})
    return row is not None and (revision<=case_at or own.get(row['id'])==revision) and all(source_visible(store,s,frontier) for s in store.provenance_at(revision)['evidence'])

def latest_case(store,identifier,case_at,frontier,snapshot=None):
    own=(snapshot or {}).get('cases',{}).get(identifier,-1)
    return next((r for r in store.db.execute('SELECT * FROM claims WHERE id=? AND (revision<=? OR revision=?) ORDER BY revision DESC',
        (identifier,case_at,own)) if case_visible(store,r['revision'],case_at,frontier,snapshot)),None)

def knowledge_visible(store,row,case_at,knowledge_at,frontier,seen=None,snapshot=None):
    seen=set() if seen is None else set(seen)
    key=(row['id'],row['revision'])
    if key in seen:return False
    seen.add(key)
    for ref in json.loads(row['refs']):
        if ref['type']=='case':
            if not case_visible(store,ref['revision'],case_at,frontier,snapshot):return False
        elif ref['type']=='evidence':
            if not source_visible(store,ref['id'],frontier):return False
        else:
            own=(snapshot or {}).get('knowledge',{}).get(ref['id'],-1)
            source=store.db.execute('SELECT * FROM reusable_knowledge WHERE id=? AND revision=? AND (revision<=? OR revision=?)',
                (ref['id'],ref['revision'],knowledge_at,own)).fetchone()
            if source is None or not knowledge_visible(store,source,case_at,knowledge_at,frontier,seen,snapshot):return False
    return True

def latest_knowledge(store,identifier,case_at,knowledge_at,frontier,snapshot=None):
    own=(snapshot or {}).get('knowledge',{}).get(identifier,-1)
    return next((r for r in store.db.execute('SELECT * FROM reusable_knowledge WHERE id=? AND (revision<=? OR revision=?) ORDER BY revision DESC',
        (identifier,knowledge_at,own)) if knowledge_visible(store,r,case_at,knowledge_at,frontier,snapshot=snapshot)),None)

def lineage_roots(store,runs):
    from pathlib import Path
    roots=set()
    for run in runs:
        evidence=store.db.execute('SELECT path FROM evidence WHERE run=? LIMIT 1',(run,)).fetchone()
        path=Path(evidence['path']).parent.parent;seen=set()
        while path.name not in seen:
            seen.add(path.name);manifest=path/'manifest.json'
            if not manifest.exists():break
            provenance=json.loads(manifest.read_text())
            prefix=provenance.get('prefix_replay')
            restart=provenance.get('infrastructure_restart')
            if prefix:
                path=Path(prefix.get('source_run') or prefix.get('run'))
            elif restart:
                field='prior_source_run' if restart.get('additional_physical_attempt') is True else 'source_run'
                source=restart[field]
                if not isinstance(source,str) or not source.strip():
                    raise ValueError('Restart lineage requires a nonempty '+field)
                path=Path(source)
            else:
                break
        roots.add(path.name)
    return roots

def navigation_warnings(store,refs,dependencies,case_at,knowledge_at,frontier,snapshot):
    """One-hop qualifications of exact navigation targets; never support or eligibility.

    Flags describe the cited version, plus revision/retirement of its latest
    visible version. A knowledge target's direct supporting/counter cases are
    checked, but its own navigation links are not recursively qualified.
    """
    warnings=[]
    for ref in refs:
        typ=ref['type']
        if typ!='knowledge' and not (typ=='case' and ref['relation']=='related'):continue
        table='reusable_knowledge' if typ=='knowledge' else 'claims'
        source=store.db.execute('SELECT * FROM '+table+' WHERE id=? AND revision=?',(ref['id'],ref['revision'])).fetchone()
        current=(latest_knowledge(store,ref['id'],case_at,knowledge_at,frontier,snapshot) if typ=='knowledge'
                 else latest_case(store,ref['id'],case_at,frontier,snapshot))
        if source is None or current is None:raise ValueError('Navigation source is not visible at the read frontier')
        flags=set()
        if current['revision']!=ref['revision']:flags.add('revised')
        if typ=='case':
            if source['status']=='contested' or store.db.execute('SELECT 1 FROM claim_reviews WHERE revision=?',(source['revision'],)).fetchone():flags.add('contested')
        else:
            if source['status']!='active' or current['status']!='active':flags.add('retired')
            for case_ref in json.loads(source['refs']):
                if case_ref['type']!='case' or case_ref['relation']=='related':continue
                case=store.db.execute('SELECT * FROM claims WHERE id=? AND revision=?',(case_ref['id'],case_ref['revision'])).fetchone()
                latest=latest_case(store,case_ref['id'],case_at,frontier,snapshot)
                if latest is None or latest['revision']!=case_ref['revision']:flags.add('revised')
                if case_ref['relation']=='counter' or case is None or case['status']=='contested' or store.db.execute('SELECT 1 FROM claim_reviews WHERE revision=?',(case_ref['revision'],)).fetchone():flags.add('contested')
            for key,value in json.loads(source['dependencies']).items():
                if unknown_value(value) or unknown_value(dependencies.get(key)):flags.add('dependency_unknown')
                elif dependencies[key]!=value:flags.add('dependency_changed')
        if flags:warnings.append({'type':typ,'id':ref['id'],'cited':ref['revision'],'current':current['revision'],'flags':sorted(flags)})
    return warnings


def knowledge_view(store,row,context,dependencies,*,case_at=None,knowledge_at=None,frontier=None,snapshot=None):
    case_at=store.revision if case_at is None else case_at
    knowledge_at=store.db.execute('SELECT COALESCE(MAX(revision),0) FROM reusable_knowledge').fetchone()[0] if knowledge_at is None else knowledge_at
    conditions=json.loads(row['conditions']);pins=json.loads(row['dependencies']);refs=json.loads(row['refs'])
    changed=[k for k,v in pins.items() if not unknown_value(v) and not unknown_value(dependencies.get(k)) and dependencies[k]!=v]
    unverified_pins=[k for k,v in pins.items() if unknown_value(v) or unknown_value(dependencies.get(k))]
    comparison=compare_conditions(conditions,context)
    unknown=[k for k,v in comparison.items() if v=='unknown']
    descriptions={k:{'source':v,'current':context[k]} for k,v in conditions.items() if k in context and v!=context[k]}
    different=[]
    runs=set();support=set();counters=set();related=set();stale=[];contested=False
    for ref in refs:
        if ref['type']=='case':
            case=store.db.execute('SELECT * FROM claims WHERE id=? AND revision=?',(ref['id'],ref['revision'])).fetchone()
            latest=latest_case(store,ref['id'],case_at,frontier,snapshot)
            role=ref['relation']
            if role!='related' and (latest is None or latest['revision']!=ref['revision']):stale.append(ref['id'])
            if role=='counter':counters.add(ref['id']);contested=True
            elif role=='related':related.add(ref['id'])
            else:
                support.add(ref['id'])
                contested |= case is None or case['status']=='contested' or bool(store.db.execute('SELECT 1 FROM claim_reviews WHERE revision=?',(ref['revision'],)).fetchone())
                for source in store.provenance_at(ref['revision'])['evidence']:
                    runs.add(store.db.execute('SELECT run FROM evidence WHERE id=?',(source,)).fetchone()[0])
    warnings=navigation_warnings(store,refs,dependencies,case_at,knowledge_at,frontier,snapshot)
    roots=lineage_roots(store,runs)
    supported=bool(support) or row['strength']=='configuration'
    counter_sources=[]
    claim_fields=('domain','family','question','text','conditions','params','dependencies','strength','use_hint','unknowns')
    base=row;history_known=True
    if any(ref.get('type')=='case' and ref.get('relation')=='counter' for ref in refs):
        for previous in store.db.execute('SELECT * FROM reusable_knowledge WHERE id=? AND revision<? ORDER BY revision DESC',
                                         (row['id'],row['revision'])):
            if ((previous['revision']>knowledge_at and (snapshot or {}).get('knowledge',{}).get(row['id'])!=previous['revision']) or
                    not knowledge_visible(store,previous,case_at,knowledge_at,frontier,snapshot=snapshot)):
                history_known=False;break
            if any(previous[k]!=row[k] for k in claim_fields):break
            base=previous
    original_counters={(r.get('type'),r.get('id'),r.get('revision'),r.get('reason'))
                       for r in json.loads(base['refs']) if r.get('relation')=='counter'}
    for ref in refs:
        if ref.get('type')=='case' and ref.get('relation')=='counter':
            counter_sources.append({'type':ref['type'],'id':ref['id'],'revision':ref['revision'],
                'author_reason':ref.get('reason',''),
                'added_with_unchanged_statement':((ref.get('type'),ref.get('id'),ref.get('revision'),ref.get('reason')) not in original_counters)
                    if history_known else None})
    return {'id':row['id'],'revision':row['revision'],'domain':row['domain'],'family':row['family'],
        'question':row['question'],'text':row['text'],'conditions':conditions,'params':json.loads(row['params']),
        'depends_on':sorted(pins),
        'strength':row['strength'],'use':row['use_hint'],'unknowns':json.loads(row['unknowns']),'status':row['status'],
        'applicability':{'changed_dependencies':changed,'unknown_conditions':unknown,'different_conditions':different,'different_descriptions':descriptions,'unknown_dependencies':unverified_pins},
        'support_cases':len(support),'support_episodes':len(runs),'source_lineages':len(roots),
        'counter_cases':len(counters),**({'counter_sources':counter_sources} if counter_sources else {}),
        'related_cases':len(related),'source_revised':sorted(set(stale)),
        'source_contested':bool(contested),**({'related_warnings':warnings} if warnings else {}),
        'default_eligible':supported and row['status']=='active' and row['strength']!='hypothesis'
            and not (changed or unverified_pins or different or contested or stale),
        'drill_down':{'refs':refs,'qualification':'Secondary evidence; IF premises are not established current state or verified physics'}}

def knowledge_facets(store,row):
    """Labels stay attached to each support case; union values never create a new relation."""
    return [json.loads(nav[0]) if nav else {} for ref in json.loads(row['refs']) if ref['type']=='case' and ref['relation']=='support'
            for nav in [store.db.execute('SELECT roles FROM case_navigation WHERE revision=?',(ref['revision'],)).fetchone()]]

def source_coverage(store,refs):
    """Source chronology, not a route, stage order, or an inference of failed gaps."""
    runs={}
    for ref in refs:
        if ref['type']!='case':continue
        for source in store.provenance_at(ref['revision'])['evidence']:
            row=store.db.execute("SELECT run,sequence FROM evidence WHERE id=? AND kind='receipt'",(source,)).fetchone()
            if row:runs.setdefault(row['run'],{})[source]=row['sequence']
    result=[]
    for run,covered in sorted(runs.items()):
        receipts=store.db.execute("SELECT id,sequence FROM evidence WHERE run=? AND kind='receipt' AND sequence BETWEEN ? AND ? ORDER BY sequence",(run,min(covered.values()),max(covered.values()))).fetchall()
        result.append({'run':run,'covered_receipts':[r['id'] for r in receipts if r['id'] in covered],
            'uncovered_between_sources':[r['id'] for r in receipts if r['id'] not in covered]})
    return {'traces':result,'qualification':'Coverage navigation only. List/ref order is not phase order. Gaps are not automatically failures; separate traces are not one continuous demonstrated procedure.'}

def visible_knowledge(store,*,domain=None,family=None,roles=None,ids=None,at=None,case_at=None,frontier=None,snapshot=None):
    initialize(store);route(domain,family)
    case_at=store.revision if case_at is None else case_at
    at=store.db.execute('SELECT COALESCE(MAX(revision),0) FROM reusable_knowledge').fetchone()[0] if at is None else at
    # Pick the latest *visible* version, rather than hiding an id whose newer version is beyond the source cutoff.
    identifiers=ids if ids is not None else [r[0] for r in store.db.execute('SELECT DISTINCT id FROM reusable_knowledge WHERE revision<=?',(at,))]
    if ids is None:identifiers=list(dict.fromkeys([*identifiers,*(snapshot or {}).get('knowledge',{})]))
    rows=[]
    for identifier in identifiers:
        row=latest_knowledge(store,identifier,case_at,at,frontier,snapshot)
        if ids is None and row is not None and row['status']!='active':continue
        if row is not None and (domain is None or row['domain']==domain) and (family is None or row['family']==family):
            if roles:
                facets=knowledge_facets(store,row)
                if facets and not any(all(f.get(k) in (None,'unknown',v) for k,v in roles.items()) for f in facets):continue
            rows.append(row)
    return rows

def knowledge_source_counts(store,rows,source_ids):
    """Count exact K ref roles that intersect visible sources, not independent trials."""
    provenance={};matched={}
    for row in rows:
        counts={'support':0,'counter':0,'related':0,'direct_evidence':0};seen=set()
        for ref in json.loads(row['refs']):
            kind=ref['type']
            if kind=='knowledge':continue
            identity=(kind,ref['id'],ref.get('revision'),ref.get('relation'))
            if identity in seen:continue
            seen.add(identity)
            if kind=='case':
                revision=ref['revision']
                if revision not in provenance:
                    provenance[revision]=set(store.provenance_at(revision)['evidence'])
                present=bool(provenance[revision]&source_ids)
                relation=ref['relation']
            else:
                present=ref['id'] in source_ids;relation='direct_evidence'
            if present:counts[relation]+=1
        if any(counts.values()):matched[(row['id'],row['revision'])]=counts
    return matched


def lexical_knowledge(store,rows,question,*,context,dependencies,roles=None,ids=None,limit=20):
    terms=content_terms(question or '');results=[]
    # SQLite FTS5, rebuilt only over this request's visible versions. No future-version rank leakage.
    scores={}
    if ids is None and terms:
        versions=tuple(sorted(row['revision'] for row in rows))
        store.db.execute('CREATE VIRTUAL TABLE IF NOT EXISTS temp.reusable_search USING fts5(question,text,use_hint,conditions,params)')
        indexed=tuple(r[0] for r in store.db.execute('SELECT rowid FROM temp.reusable_search ORDER BY rowid'))
        # Actual SQLite contents make rollback safe; exact reads never disturb this corpus.
        if indexed!=versions:
            store.db.execute('DELETE FROM temp.reusable_search')
            for row in rows:
                parameters=[{k:p[k] for k in ('name','value','unit','frame')} for p in json.loads(row['params'])]
                store.db.execute('INSERT INTO temp.reusable_search(rowid,question,text,use_hint,conditions,params) VALUES(?,?,?,?,?,?)',
                    (row['revision'],row['question'],row['text'],row['use_hint'],row['conditions'],json.dumps(parameters)))
        query=' OR '.join('"'+term.replace('"','""')+'"' for term in terms)
        scores={row[0]:row[1] for row in store.db.execute('SELECT rowid,bm25(reusable_search) FROM temp.reusable_search WHERE reusable_search MATCH ?',(query,))}
    for row in rows:
        if ids is None and terms and row['revision'] not in scores:continue
        score=scores.get(row['revision'],0.)
        pins=json.loads(row['dependencies'])
        changed=sum(not unknown_value(v) and not unknown_value(dependencies.get(k)) and dependencies[k]!=v for k,v in pins.items())
        unknown=sum(v=='unknown' for v in compare_conditions(json.loads(row['conditions']),context).values())
        role_unknown=bool(roles) and not any(all(f.get(k)==v for k,v in roles.items()) for f in knowledge_facets(store,row))
        # A question first selects its subject. Unchanged pins cannot make an
        # unrelated subject relevant; all applicability warnings remain intact.
        rank=((score,changed,role_unknown,unknown,-row['revision']) if terms and ids is None
              else (changed,role_unknown,score,unknown,-row['revision']))
        results.append((rank,row))
    if ids is not None:
        order={identifier:i for i,identifier in enumerate(ids)}
        results.sort(key=lambda item:order[item[1]['id']])
    else:results.sort(key=lambda item:item[0])
    return [row for _,row in results[:limit]]

def read_knowledge(store,question,*,context,dependencies,domain=None,family=None,roles=None,ids=None,limit=4,at=None,case_at=None,frontier=None,snapshot=None):
    initialize(store)
    case_at=store.revision if case_at is None else case_at
    at=store.db.execute('SELECT COALESCE(MAX(revision),0) FROM reusable_knowledge').fetchone()[0] if at is None else at
    rows=visible_knowledge(store,domain=domain,family=family,roles=roles,ids=ids,
        at=at,case_at=case_at,frontier=frontier,snapshot=snapshot)
    ordered=lexical_knowledge(store,rows,question,context=context,dependencies=dependencies,
        roles=roles,ids=ids,limit=limit)
    return [knowledge_view(store,row,context,dependencies,case_at=case_at,knowledge_at=at,
                           frontier=frontier,snapshot=snapshot) for row in ordered]

def case_reference_issues(refs):
    """All missing case relation fields, with the caller's original ID/local handle."""
    issues=[]
    for index,ref in enumerate(refs):
        if ref.get('type')!='case':continue
        missing=[]
        if ref.get('relation') not in ('support','counter','related'):missing.append('relation')
        if not isinstance(ref.get('reason'),str) or not ref['reason'].strip():missing.append('reason')
        if missing:issues.append({'path':f'refs[{index}]',
            'source':{k:ref[k] for k in ('id','local','revision') if k in ref},'missing_or_invalid':missing})
    return issues


def publish_knowledge(store,op,*,allowed,visible_cases,visible_knowledge,dependencies,frontier,
                      related_statement_versions=None,related_scope=None):
    """Input refs are host-resolved exact versions. No current pins are substituted for historical sources."""
    initialize(store)
    action=op.get('op');identifier=op.get('id');previous=None
    if action not in ('add','revise','retire','append_refs'):raise ValueError('Unknown knowledge operation')
    if action=='add' and identifier is not None:raise ValueError('Knowledge IDs are host allocated')
    if action!='add':
        previous=store.db.execute('SELECT * FROM reusable_knowledge WHERE id=? ORDER BY revision DESC LIMIT 1',(identifier,)).fetchone()
        if previous is None:
            raise ValueError('Unknown knowledge ID for update: '+json.dumps({'knowledge_id':identifier,
                'repair':'Find the existing knowledge ID; add allocates a new ID. Do not invent a revision or replacement source.'}))
        if visible_knowledge.get(identifier)!=previous['revision']:
            append=action=='append_refs'
            raise ValueError(('Reference append requires a current actually read complete statement or detail: ' if append else
                              'Knowledge update requires current actually read full detail: ')+json.dumps({
                'knowledge_id':identifier,'current_revision':previous['revision'],
                'actually_read_revision':visible_knowledge.get(identifier),
                'read':{'level':'knowledge','view':'statement' if append else 'detail','ids':[identifier]},
                'repair':('Read the complete current target statement or detail; new references retain their own source read requirements. ' if append else
                          'Read the complete current detail, including every continuation page; a statement does not permit revise/retire. ')+
                         'If this snapshot cannot expose the current revision, leave the update unresolved; do not substitute version numbers. Read completeness and source contract compatibility are independent checks.'}))
    if action=='append_refs':
        if previous['status']!='active':raise ValueError('append_refs requires active knowledge; use revise for an explicit status change')
        if set(op)-{'op','id','refs'}:raise ValueError('append_refs accepts only id and new refs; use revise to change meaning')
        old_refs=json.loads(previous['refs']);new_refs=[]
        for ref in op.get('refs',[]):
            old=next((r for r in old_refs if (r['type'],r['id'])==(ref.get('type'),ref.get('id'))),None)
            if old is not None:
                if ref!=old:raise ValueError('append_refs cannot change a source revision, relation or reason; use revise')
            elif ref not in new_refs:new_refs.append(ref)
        if not new_refs:return {'id':identifier,'revision':previous['revision'],'status':previous['status'],'unchanged':True}
        op={'op':'revise','id':identifier,**{k:previous[k] for k in ('domain','family','question','text','strength')},
            'conditions':json.loads(previous['conditions']),'params':json.loads(previous['params']),
            'depends_on':list(json.loads(previous['dependencies'])),'use':previous['use_hint'],
            'unknowns':json.loads(previous['unknowns']),'refs':old_refs+new_refs}
    contract_basis={}
    if action=='retire':
        if not op.get('reason'):raise ValueError('Retirement needs a reason')
        data=dict(previous);data['status']='retired'
    else:
        domain,family=op.get('domain'),op.get('family');route(domain,family)
        if domain is None or family is None:raise ValueError('Knowledge needs a domain/family')
        for key,maximum in [('question',600),('text',1800),('use',900)]:
            value=op.get(key)
            if not isinstance(value,str) or not value:raise ValueError(key+': nonempty string required; maximum'+str(maximum)+' UTF-8 bytes')
            if len(value.encode())>maximum:raise ValueError(key+': '+str(len(value.encode()))+' UTF-8 bytes; maximum'+str(maximum)+' bytes')
        conditions=op.get('conditions',{});unknowns=op.get('unknowns',[])
        if not isinstance(conditions,dict) or len(conditions)>5 or any(not isinstance(v,str) or not v for v in conditions.values()):raise ValueError('At most five material IF conditions required')
        if not isinstance(unknowns,list) or any(not isinstance(v,str) for v in unknowns):raise ValueError('Explicit unknowns required')
        refs=op.get('refs',[]);strength=op.get('strength')
        if strength not in ('configuration','observation','hypothesis'):raise ValueError('Unknown evidence strength')
        if not refs:raise ValueError('Knowledge must have exact sources')
        if len({(r.get('type'),r.get('id')) for r in refs})!=len(refs):raise ValueError('Each source appears once per knowledge item')
        reference_issues=case_reference_issues(refs)
        if reference_issues:raise ValueError('Case references require relation and specific reason: '+json.dumps(reference_issues))
        previous_refs=json.loads(previous['refs']) if previous else []
        contract_keys=set(op.get('depends_on',[])) | ({'body'} if domain=='body' else set())
        same_statement=bool(previous) and all((
            previous['domain']==domain,previous['family']==family,previous['question']==op['question'],
            previous['text']==op['text'],json.loads(previous['conditions'])==conditions,
            previous['use_hint']==op['use'],previous['strength']==strength,
            json.loads(previous['unknowns'])==unknowns,json.loads(previous['params'])==op.get('params',[]),
            set(json.loads(previous['dependencies']))==contract_keys))
        pin_sources=[];support_cases=[];has_support=False;configuration_only=True
        for ref_index,ref in enumerate(refs):
            if ref.get('type')=='case':
                configuration_only &= ref.get('relation')=='related'
                if visible_cases.get(ref.get('id'))!=ref.get('revision') and not (same_statement and ref in previous_refs):raise ValueError('Changed knowledge interpretation requires actual case reads; inherited refs only preserve an unchanged statement')
                if ref.get('relation') not in ('support','counter','related') or not ref.get('reason'):raise ValueError('Case relation and specific reason required')
                unknown=ref.get('unverified_conditions',[])
                if not isinstance(unknown,list) or not set(unknown)<=conditions.keys():raise ValueError('refs['+str(ref_index)+'].unverified_conditions: allowed condition keys='+json.dumps(sorted(conditions))+'; got='+json.dumps(unknown))
                if ref['relation']=='support':
                    if unknown:raise ValueError('Unverified material premises may be related, not support')
                    case=store.db.execute('SELECT * FROM claims WHERE id=? AND revision=?',(ref['id'],ref['revision'])).fetchone()
                    if case is None:raise ValueError('Unknown case version')
                    if case['epistemic']=='hypothesis' and strength!='hypothesis':raise ValueError('A hypothesis cannot become observed support')
                    has_support=True;support_cases.append(case);pin_sources.append(json.loads(case['dependencies']))
            elif ref.get('type')=='knowledge':
                if ref.get('relation')!='related' or not isinstance(ref.get('reason'),str) or not ref['reason'].strip():raise ValueError('Knowledge references require relation=related and a specific reason; navigation is not support')
                if ref.get('id')==identifier:raise ValueError('Knowledge cannot cite itself')
                full=visible_knowledge.get(ref.get('id'))==ref.get('revision')
                statement=(related_statement_versions or {}).get(ref.get('id'))==ref.get('revision')
                if not(full or statement) and ref not in previous_refs:raise ValueError('Related knowledge requires a complete actually delivered statement or detail')
                source=store.db.execute('SELECT * FROM reusable_knowledge WHERE id=? AND revision=?',(ref['id'],ref['revision'])).fetchone()
                if source is None:raise ValueError('Unknown knowledge version')
                if statement and not full and related_scope is None:raise ValueError('Statement navigation requires explicit request watermarks and perception profile')
                if related_scope is not None:
                    own=(related_scope.get('snapshot') or {}).get('knowledge',{})
                    if (source['revision']>related_scope['knowledge_at'] and own.get(source['id'])!=source['revision']) or not knowledge_visible(
                            store,source,related_scope['case_at'],related_scope['knowledge_at'],frontier,snapshot=related_scope.get('snapshot')):
                        raise ValueError('Related knowledge is outside the request source frontier or authorized versions')
                    from .perception import select
                    select(related_scope['profile']).check_memory(store)
            elif ref.get('type')=='evidence':
                if ref.get('id') not in allowed:raise ValueError('Raw source must have been supplied this transaction')
                row=store.db.execute('SELECT kind FROM evidence WHERE id=?',(ref['id'],)).fetchone()
                if row is None or not source_visible(store,ref['id'],frontier):raise ValueError('Raw source beyond frontier')
                from .contracts import INTERPRETATION_KINDS
                if row['kind'] in INTERPRETATION_KINDS:raise ValueError('Episode interpretations/results are candidates, not physical knowledge support')
                configuration_only &= row['kind']=='known_config'
                pin_sources.append(store.evidence_payload(ref['id']).get('dependencies',{}))
            else:raise ValueError('Unknown source type')
        if strength=='configuration' and (not configuration_only or not pin_sources):raise ValueError('Only actual known configuration sources establish configuration facts')
        if strength!='configuration' and not has_support:raise ValueError('Nonconfiguration knowledge requires at least one supported case')
        # A counterexample cannot silently vanish when wording is revised. Scope changes must explain reclassification.
        for old in previous_refs:
            if old.get('relation')!='counter':continue
            replacement=next((r for r in refs if r['type']=='case' and r['id']==old['id']),None)
            if replacement is None or (replacement.get('relation')!='counter' and not replacement.get('resolution')):raise ValueError('Retain counter source or explicitly resolve its scope')
        keys=op.get('depends_on',[])
        if not isinstance(keys,list) or any(not isinstance(k,str) for k in keys):raise ValueError('Dependency keys required')
        if domain=='body' and 'body' not in keys:keys=[*keys,'body']
        pins={}
        if strength!='configuration':
            from .contracts import resolve_case_contracts
            pins,contract_basis=resolve_case_contracts(store,support_cases,keys)
            # Direct evidence may not override the supporting-case contract.
            # Related knowledge is navigation and supplies neither pins nor parameters.
            conflicts=[]
            for ref in refs:
                if ref['type']!='evidence':continue
                declared=json.loads(store.db.execute('SELECT payload FROM evidence WHERE id=?',(ref['id'],)).fetchone()[0]).get('dependencies',{})
                mismatch={k:{'expected':v,'actual':declared.get(k),'missing_or_unknown':unknown_value(declared.get(k))}
                          for k,v in pins.items() if unknown_value(declared.get(k)) or declared[k]!=v}
                if mismatch:conflicts.append({'type':ref['type'],'id':ref['id'],'revision':ref.get('revision'),'mismatches':mismatch})
            if conflicts:raise ValueError('Secondary source contracts differ or are missing: '+json.dumps({
                'basis':'supporting_case_derived_contract_not_current_request','sources':conflicts,
                'repair':'Read permission does not establish compatibility. Direct evidence sources must match the derived contract; navigation provides no support. Do not invent pins or weaken conditions.'},ensure_ascii=False))
        else:
            for key in keys:
                values=[s.get(key) for s in pin_sources]
                if not values or any(unknown_value(v) or v!=values[0] for v in values):raise ValueError(key+': source contracts differ or are missing')
                pins[key]=values[0]
        parameters=op.get('params',[])
        import math
        if not isinstance(parameters,list):raise ValueError('Parameters must be an array')
        for param in parameters:
            if not isinstance(param,dict) or not {'name','value','unit','frame','source'}<=param.keys():raise ValueError('Every parameter needs name/value/unit/frame/source')
            values=param['value'] if isinstance(param['value'],list) else [param['value']]
            if not values or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in values):raise ValueError('Parameter numbers must be finite')
            if any(not isinstance(param[k],str) or not param[k] for k in ('name','unit','frame')):raise ValueError('Parameter frame and units required')
            if param['source'] not in refs:raise ValueError('Parameter source must be an exact provided reference')
            if param['source']['type']=='knowledge' or param['source'].get('relation')=='related':raise ValueError('Navigation references cannot source parameters; read and cite supporting evidence or a supported case')
        data={'id':identifier or 'n'+uuid.uuid4().hex[:15],'domain':domain,'family':family,'question':op['question'],'text':op['text'],
            'conditions':encoded(conditions),'params':encoded(parameters),'dependencies':encoded(pins),'strength':strength,
            'use_hint':op['use'],'unknowns':encoded(unknowns),'refs':encoded(refs),'status':'active'}
    columns=('id','domain','family','question','text','conditions','params','dependencies','strength','use_hint','unknowns','refs','status')
    cursor=store.db.execute('INSERT INTO reusable_knowledge('+','.join(columns)+') VALUES('+','.join('?' for _ in columns)+')',tuple(data[k] for k in columns))
    result={'id':data['id'],'revision':cursor.lastrowid,'status':data['status']}
    store.db.execute('INSERT INTO knowledge_contract_basis VALUES(?,?)',(cursor.lastrowid,encoded(contract_basis)))
    if action!='retire':
        from .readout import pointer,size
        decision={k:op.get(k) for k in ('question','text','conditions','use','unknowns','strength','depends_on')}
        decision['params']=[{**p,'source':pointer(p['source'])} for p in parameters]
        result['decision_bytes']=size(decision)
        if result['decision_bytes']>3500:
            result['readout_warning']='Long decision statement may require preview/paging. Retain material premises; move instance detail into cases or narrow/split the question instead of deleting conditions.'
    return result
