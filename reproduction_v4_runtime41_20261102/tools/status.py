"""Read sealed outcomes and separate-role costs. Unknown usage remains unknown."""
import argparse
import json
from pathlib import Path
from common import load


def status(experiment):
    e=Path(experiment);p=load(e/'PROTOCOL.json');state=e/p['state_relative_path']
    if not state.exists():return {'experiment':str(e),'status':'PREPARED_NOT_STARTED','completed':0}
    s=load(state);episodes=[]
    for row in s['completed']:
        roles={}
        for role in ['primary','manager']:
            m=row[role];u=m.get('usage_reported',{})
            roles[role]={'calls':m.get('cli_invocations'),'usage_reported':u,
                'input_plus_output':u['input_tokens']+u['output_tokens'] if m.get('captured_metadata_consistent') is True and all(k in u for k in ['input_tokens','output_tokens']) else None,
                'scope':m.get('usage_scope')}
        episodes.append({'order':row['order'],'task':row['task'],'layout':row['layout'],
            'outcome':row['task_outcome'],'env_steps':row['env_steps'],
            'memory_counts':row['memory_counts'],'roles':roles,'run':row['run']})
    return {'experiment':str(e),'status':s['status'],'completed':len(episodes),
        'successes':sum(x['outcome']=='PASS' for x in episodes),'active':s.get('active'),
        'pause_reason':s.get('pause_reason'),'episodes':episodes,
        'qualification':'Sealed outcomes only; cached input/reasoning output are subsets, not extra tokens; not a bill.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('experiment',type=Path)
    print(json.dumps(status(p.parse_args().experiment),ensure_ascii=False,indent=2))
