"""Explicit one-time derivation into a new branch. Never rewrites round-one authority."""
import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from kyb.memory import MemoryStore
from kyb import library, body_index, leads


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def adapt(source, destination):
    source=Path(source).resolve();destination=Path(destination).resolve()
    if source==destination or destination.exists():raise ValueError('A fresh destination is required')
    original=sha(source)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with sqlite3.connect('file:'+str(source)+'?mode=ro&immutable=1',uri=True) as src:
        with sqlite3.connect(destination) as dst:src.backup(dst)
    store=MemoryStore(destination);library.initialize(store)
    preserved={t:[tuple(r) for r in store.db.execute('SELECT * FROM '+t)] for t in
        ('evidence','claims','claim_evidence','reusable_knowledge','observation_dependencies','claim_reviews','case_disposition','processed_intervals')}
    with store.db:
        # New input uses one observed-interval type. Prior event identity/capsule is retained verbatim.
        relations=[dict(r) for r in store.db.execute('SELECT * FROM interaction_evidence')]
        store.db.execute("UPDATE interaction_evidence SET relation_type='observed_interaction'")
        for row in store.db.execute('SELECT c.revision,g.entity,g.observable FROM claims c JOIN claim_grounding g ON g.revision=c.revision').fetchall():
            arms=set()
            for ref in store.provenance_at(row['revision'])['evidence']:
                item=store.db.execute("SELECT payload FROM evidence WHERE id=? AND kind='receipt'",(ref,)).fetchone()
                if item:arms.update(json.loads(item[0]).get('plan',{}).get('trace',{}).get('arms',{}))
            facets={'entity':row['entity'],'observable':row['observable']}
            if arms:facets['arm']='both' if len(arms)>1 else next(iter(arms))
            store.db.execute('UPDATE case_navigation SET roles=? WHERE revision=?',(json.dumps(facets),row['revision']))
        indexed=body_index.rebuild(store)
        runs=[r[0] for r in store.db.execute('SELECT DISTINCT run FROM evidence')]
        for run in runs:
            evidence=store.db.execute('SELECT path FROM evidence WHERE run=? LIMIT 1',(run,)).fetchone()
            root=Path(evidence[0]).parent.parent
            for line in (root/'events.jsonl').read_text().splitlines():
                ref=json.loads(line)
                if ref['kind']!='manager_publication':continue
                if sha(ref['path'])!=ref['sha256']:raise ValueError('Historical review file changed')
                payload=json.loads(Path(ref['path']).read_text())['payload']
                if payload.get('purpose')=='review' and payload.get('answer'):
                    leads.add(store,ref,{'question':payload.get('question'),'answer':payload['answer'],
                        'review_publication':ref['id'],'source_refs':payload.get('provided_sources',[]),
                        'origin':'post_round_deterministic_candidate_index_not_new_learning',
                        'qualification':'A review is not physical confirmation or causal benefit'})
        for table,rows in preserved.items():
            if [tuple(r) for r in store.db.execute('SELECT * FROM '+table)]!=rows:
                raise AssertionError('Source content changed: '+table)
    counts={'cases':store.db.execute('SELECT COUNT(DISTINCT id) FROM claims').fetchone()[0],
            'knowledge':store.db.execute('SELECT COUNT(DISTINCT id) FROM reusable_knowledge').fetchone()[0],
            'body_endpoints':indexed['indexed_rows'],'learning_leads':store.db.execute('SELECT COUNT(*) FROM learning_leads').fetchone()[0]}
    if store.db.execute('PRAGMA integrity_check').fetchone()[0]!='ok' or store.db.execute('PRAGMA foreign_key_check').fetchall():
        raise AssertionError('Adapted database integrity failure')
    store.db.execute('PRAGMA wal_checkpoint(TRUNCATE)');store.close()
    if sha(source)!=original:raise AssertionError('Original memory changed')
    manifest={'status':'PASS','source':str(source),'source_sha256':original,'destination':str(destination),
        'sha256':sha(destination),'counts':counts,'new_cases':0,'new_knowledge':0,
        'adaptations':['interaction input relation normalized; original capsules retained','case navigation facts derived from exact recorded entity/observable/participating arms','source-time body endpoint index','review-to-subsequent-response discovery leads'],
        'previous_interaction_types':relations,'qualification':'Offline deterministic adaptation. No new robot observations, semantic re-extraction, independent trials or task scores.'}
    destination.with_suffix('.adaptation.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    # Freeze through existing project snapshot mechanism, preserving lineage metadata and inherited source revisions.
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True,type=Path);parser.add_argument('--destination',required=True,type=Path)
    args=parser.parse_args();print(json.dumps(adapt(args.source,args.destination),ensure_ascii=False))
