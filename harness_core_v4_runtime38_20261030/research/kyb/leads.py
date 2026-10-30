"""Bounded question navigation. Interpretations never grant evidence permissions."""
import json


def initialize(store):
    store.db.execute('''CREATE TABLE IF NOT EXISTS learning_leads(
        id TEXT PRIMARY KEY,run TEXT NOT NULL,sequence INTEGER NOT NULL,payload TEXT NOT NULL)''')


def add(store, ref, payload):
    initialize(store)
    store.db.execute('INSERT OR IGNORE INTO learning_leads VALUES(?,?,?,?)',
        (ref['id'],ref['id'].rsplit(':e',1)[0],ref['sequence'],json.dumps(payload,ensure_ascii=False)))


def read(store, run, identifier, frontier):
    """A trace publication need not be registered as raw physical evidence."""
    if not isinstance(identifier,str):raise ValueError('One exact learning-lead ID required')
    row=store.db.execute('SELECT * FROM learning_leads WHERE id=? AND run=? AND sequence<=?',
                        (identifier,run,frontier.get(run,-1))).fetchone()
    if row is None:raise ValueError('Learning lead not visible in this run/frontier')
    return {'lead':row['id'],**json.loads(row['payload']),
            'read_qualification':'Complete recorded interpretation, not an independent observation or permission to cite unread sources.'}


def pending(store, run, frontier, limit=4, *, budget_bytes=2800):
    initialize(store)
    end=frontier.get(run,-1)
    rows=store.db.execute('SELECT * FROM learning_leads WHERE run=? AND sequence<=? ORDER BY sequence DESC',
                         (run,end)).fetchall()
    rows=sorted(rows,key=lambda row:('review_publication' not in json.loads(row['payload']),-row['sequence']))
    result={'run':run,'through':end,'lead_sequences':[r['sequence'] for r in rows[:32]],
            'omitted_lead_sequences':max(0,len(rows)-32),
            'lead_directory':{'run':run,'after':0,'through':end,'directory':'leads'},
            'id_format':run+':e{sequence:06d}','records':[],
            'closure_status':'not_recorded; processed intervals and case dispositions do not close questions',
            'qualification':'Navigation only. Subsequent means later in time, not a diagnostic test or benefit. Read exact sources before writing.'}
    result['total_leads']=len(rows)
    for row in rows[:limit]:
        data=json.loads(row['payload'])
        measurements=store.db.execute("SELECT sequence FROM evidence WHERE run=? AND kind='surface_inspection' AND sequence>? AND sequence<=? ORDER BY sequence",
                                     (run,row['sequence'],end)).fetchall()
        record={'lead':row['id'],'question_excerpt':data.get('question','')[:180],
                'question_complete':len(data.get('question',''))<=180,
                'answer_excerpt':data.get('answer','')[:180],
                'answer_complete':len(data.get('answer',''))<=180,
                'candidate_refs':data.get('candidate_refs',[])[:2],
                'more_candidate_refs':max(0,len(data.get('candidate_refs',[]))-2),
                'qualification':data.get('qualification','Recorded interpretation, not physical evidence'),
                'measurement_sequences':[m['sequence'] for m in measurements[:8]],
                'more_measurements':max(0,len(measurements)-8),
                'read_trajectory':{'run':run,'after':row['sequence'],'through':end,'include_processed':True,'lead':row['id']}}
        candidate={**result,'records':[*result['records'],record]}
        if len(json.dumps(candidate,ensure_ascii=False).encode())<=budget_bytes-40:result=candidate
    result['omitted_records']=len(rows)-len(result['records'])
    if len(json.dumps(result,ensure_ascii=False).encode())>budget_bytes:
        raise ValueError('Learning-lead directory exceeds navigation budget')
    return result
