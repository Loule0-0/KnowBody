"""Reopen only a stopped, sealed episode boundary. Never recover an active episode."""
import argparse
import fcntl
import json
import os
from pathlib import Path
from common import digest,load,write_new


def reconcile(experiment):
    e=Path(experiment);p=load(e/'PROTOCOL.json');state_path=e/p['state_relative_path']
    lock=e/'executor.lock' if p['study']=='single50' else state_path.parent/'executor.lock'
    with lock.open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        s=load(state_path)
        if s['status']!='PAUSED' or s.get('active') is not None:
            raise ValueError('Only PAUSED with no active episode can be reconciled; live/unknown episodes require separate recovery')
        if (e/'STOP').exists():raise ValueError('Remove the operator STOP after deciding to continue')
        if s['next_order']!=len(s['completed'])+1:raise ValueError('Boundary order mismatch')
        for name,value in load(e/'HARNESS.json')['files'].items():
            if digest(name)!=value:raise ValueError('Frozen file changed: '+name)
        for row in s['completed']:
            for role in ['primary','manager']:
                if row[role].get('outcome_unknown',0):raise ValueError('An earlier outcome remains unknown')
            if digest(row['final_memory'])!=row['final_memory_sha256']:raise ValueError('Sealed memory changed')
        suffix=1
        while (e/f'BOUNDARY_RESUME_{suffix:03d}.json').exists():suffix+=1
        write_new(e/f'BOUNDARY_RESUME_{suffix:03d}.json',{'prior_state_sha256':digest(state_path),
            'reason':'Explicit operator invocation; sealed idle boundary only','completed':len(s['completed']),
            'new_model_calls':0,'physics_steps':0,'budget_changes':False})
        s['status']='READY';s.pop('pause_reason',None)
        temp=state_path.with_suffix('.operator.tmp')
        with temp.open('x') as f:json.dump(s,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
        os.replace(temp,state_path)
    print('PASS: sealed idle boundary ready; no budget reset, model call or simulator launch')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('experiment',type=Path);reconcile(p.parse_args().experiment)
