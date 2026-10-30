"""Deterministic index over source-time body telemetry, not learned physical laws."""
from __future__ import annotations
import hashlib
import json
import math
from .state import compact_body_state, tracking_residual


def initialize(store):
    store.db.execute('''CREATE TABLE IF NOT EXISTS body_endpoints(
        id TEXT PRIMARY KEY,run TEXT NOT NULL,sequence INTEGER NOT NULL,arm TEXT NOT NULL,
        embodiment TEXT NOT NULL,observation_id TEXT NOT NULL REFERENCES evidence(id),
        receipt_id TEXT NOT NULL REFERENCES evidence(id),payload TEXT NOT NULL)''')
    store.db.execute('CREATE INDEX IF NOT EXISTS body_endpoints_scope ON body_endpoints(embodiment,arm,run,sequence)')


def embodiment(config, arm, run):
    contract = config.get('coordinate_contract', {})
    mount = config.get('planner_setup', {}).get('mounts', {}).get(arm + '_arm')
    identity = {'body': config.get('body_hash') or config.get('dependencies', {}).get('body'),
                'arm': arm, 'mount': mount, 'coordinate_contract': contract}
    if not identity['body'] or not mount or not contract.get('id'):
        identity['unknown_contract_run_scope'] = run
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def configuration(store, run):
    row = store.db.execute("SELECT payload FROM evidence WHERE run=? AND kind='known_config' ORDER BY sequence LIMIT 1", (run,)).fetchone()
    return json.loads(row[0]) if row else {}


def index_observation(store, source):
    initialize(store)
    row = store.db.execute("SELECT * FROM evidence WHERE id=? AND kind='observation'", (source,)).fetchone()
    if row is None:
        return 0
    post = json.loads(row['payload'])
    receipt_id = post.get('after_receipt')
    receipt_row = store.db.execute("SELECT * FROM evidence WHERE id=? AND kind='receipt'", (receipt_id,)).fetchone()
    if not receipt_row or receipt_row['run'] != row['run'] or receipt_row['sequence'] >= row['sequence']:
        return 0
    receipt = json.loads(receipt_row['payload'])
    before_id = receipt.get('before_observation')
    before_row = store.db.execute('SELECT * FROM evidence WHERE id=?', (before_id,)).fetchone()
    if not before_row or before_row['run'] != row['run'] or before_row['sequence'] >= receipt_row['sequence']:
        return 0
    before = json.loads(before_row['payload'])
    config = configuration(store, row['run'])
    measured = compact_body_state(post)
    initial = compact_body_state(before)
    residual = tracking_residual(receipt, post).get('measured_differences', {})
    hz = config.get('execution_protocol', {}).get('control_hz')
    count = 0
    for arm, values in measured.get('arms', {}).items():
        detail = receipt.get('plan', {}).get('trace', {}).get('arms', {}).get(arm)
        # An untouched arm is not another independent reaching trial.
        if detail is None:
            continue
        r = residual.get(arm, {})
        delta = r.get('grasp_position_current_minus_source_m')
        start = initial.get('arms', {}).get(arm, {})
        pose = values['grasp_world_m_wxyz']
        old_pose = start.get('grasp_world_m_wxyz')
        motion = []
        if old_pose:
            motion = [pose[i]-old_pose[i] for i in range(3)]
        payload = {'id': source + '/' + arm, 'arm': arm, 'run': row['run'],
            'before': before_id, 'receipt': receipt_id, 'post': source,
            'embodiment': embodiment(config, arm, row['run']), 'sequence': row['sequence'],
            'sources': [{'id': r['id'], 'sha256': r['sha256']} for r in (before_row, receipt_row, row)],
            'execution': receipt.get('execution'), 'unknown_command': receipt.get('unknown_command'),
            'source_dependencies': receipt.get('dependencies', {}),
            'planned_actions': receipt.get('full_planned_actions', receipt.get('planned_actions')),
            'executed_actions': receipt.get('executed_actions'),
            'duration_s': receipt.get('executed_actions', 0)/hz if hz else None,
            'requested': detail.get('named'), 'requested_flange_pose': detail.get('target_pose'),
            'measured': values, 'start': start,
            'position_error_m': math.sqrt(sum(x*x for x in delta)) if delta else None,
            'orientation_error_deg': r.get('orientation_difference_deg'),
            'motion_signature': {'translation_m': motion, 'finger_before_m': start.get('finger_displacements_m'),
                                 'finger_after_m': values.get('finger_displacements_m')},
            'qualification': 'Measured endpoint in the recorded scene/start/configuration; not holding, contact, future reachability or path proof.'}
        store.db.execute('INSERT OR IGNORE INTO body_endpoints VALUES(?,?,?,?,?,?,?,?)',
            (payload['id'], row['run'], row['sequence'], arm, payload['embodiment'], source,
             receipt_id, json.dumps(payload, ensure_ascii=False)))
        count += 1
    return count


def rebuild(store):
    initialize(store)
    count = 0
    for row in store.db.execute("SELECT id FROM evidence WHERE kind='observation' AND json_extract(payload,'$.after_receipt') IS NOT NULL").fetchall():
        count += index_observation(store, row['id'])
    return {'indexed_rows': store.db.execute('SELECT COUNT(*) FROM body_endpoints').fetchone()[0],
            'new_semantic_knowledge': 0}


def query(store, *, config, run, frontier, arm, target, point='grasp', tolerance_m=0.05, limit=4):
    if arm not in ('left', 'right') or point not in ('grasp', 'flange'):
        raise ValueError('Choose left/right and grasp/flange')
    if not isinstance(target, dict) or not target or not set(target) <= {'x','y','z'}:
        raise ValueError('target requires one or more world metre coordinates x/y/z')
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in target.values()):
        raise ValueError('Finite target coordinates required')
    if isinstance(tolerance_m,bool) or not isinstance(tolerance_m,(int,float)) or not math.isfinite(tolerance_m) or tolerance_m<=0:
        raise ValueError('Positive finite tolerance_m required')
    if type(limit) is not int or not 1<=limit<=8:
        raise ValueError('limit must be 1..8')
    identity = embodiment(config, arm, run)
    matches = []
    for row in store.db.execute('SELECT * FROM body_endpoints WHERE embodiment=? AND arm=?', (identity,arm)):
        if row['sequence'] > frontier.get(row['run'], -1):
            continue
        fact = json.loads(row['payload'])
        if fact['executed_actions'] <= 0 or fact['unknown_command'] or fact['execution'] not in ('complete','partial'):
            continue
        pose = fact['measured'][point + '_world_m_wxyz']
        distances = [abs(pose['xyz'.index(k)]-v) for k,v in target.items()]
        if max(distances) <= tolerance_m:
            matches.append((sum(d*d for d in distances), fact))
    matches.sort(key=lambda item: (item[0], -item[1]['sequence']))
    # Prefer distinct orientation witnesses over repeating the same held pose.
    selected=[]
    for _,fact in matches:
        q=fact['measured'][point+'_world_m_wxyz'][3:7]
        if all(abs(sum(a*b for a,b in zip(q,f['measured'][point+'_world_m_wxyz'][3:7]))) < math.cos(math.radians(10)) for f in selected):
            selected.append(fact)
        if len(selected)>=limit:break
    if len(selected)<limit:
        ids={r['id'] for r in selected}
        selected.extend(f for _,f in matches if f['id'] not in ids)
        selected=selected[:limit]
    return {'status':'records_found' if selected else 'no_match','derived_index':'body telemetry within E; not a fourth semantic layer',
            'embodiment':identity,'target':target,'tolerance_m':tolerance_m,'matching_observations':len(matches),
            'records':[{k:f[k] for k in ('id','arm','run','before','receipt','post','execution','executed_actions','duration_s','requested','measured','position_error_m','orientation_error_deg')} for f in selected],'qualification':'Endpoints are observations, not a calibrated workspace envelope or current collision/retention guarantee. Failed plans are separate computational evidence; ask manager to interpret needed sources. matching_observations counts recorded endpoints, not independent trials; replay belongs to the source lineage.'}


def tool():
    from .knowledge import function
    return function('query_body_history','Find source-time measured body endpoints by arm and world target region. No manager or motion. '
        'Use for high reach, grasp-vs-flange geometry or previous tracking. It does not prove current paths/holding. Then query cases/manage for physical interpretation.',
        {'arm':{'type':'string','enum':['left','right']},'point':{'type':'string','enum':['grasp','flange']},
         'target':{'type':'object','properties':{k:{'type':'number'} for k in 'xyz'},'additionalProperties':False},
         'tolerance_m':{'type':'number','exclusiveMinimum':0},'limit':{'type':'integer','minimum':1,'maximum':8}},['arm','target'])
