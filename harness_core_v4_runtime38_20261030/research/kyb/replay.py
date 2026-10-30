"""Replay a fully receipted scene-action prefix from reset; no teleport or model call."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .execution import execute_chunk


def load_prefix(root: Path, observation_id: str, *, task: str, layout: int):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest['task'] != task or manifest['layout'] != layout:
        raise ValueError('Replay prefix must use the identical task and layout')
    index = [json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()]
    events = {}
    for ref in index:
        p = Path(ref['path'])
        if p.parent != root/'events' or hashlib.sha256(p.read_bytes()).hexdigest() != ref['sha256']:
            raise ValueError('Replay source evidence changed')
        event = json.loads(p.read_text())
        events[event['id']] = event
    target = events[observation_id]
    if target['kind'] != 'observation' or target['visibility'] != 'policy':
        raise ValueError('Replay endpoint must be a formal public observation')
    initial = next(e for e in events.values() if e['kind'] == 'observation')
    before = initial
    prefix, count = [], initial['payload']['env_step']
    physics_cursor = initial['payload']['capture_timing']['physics_step_after']
    for event in events.values():
        if event['sequence'] > target['sequence'] or event['kind'] != 'receipt':
            continue
        receipt = event['payload']
        selected_complete = (receipt['execution'] == 'partial'
            and receipt.get('stop_reason') == 'observation_boundary'
            and receipt.get('selected_actions') == receipt['planned_actions']
            and receipt.get('full_planned_actions', 0) > receipt['planned_actions']
            and receipt.get('policy_stop') is False)
        if ((receipt['execution'] != 'complete' and not selected_complete)
            or receipt['unknown_command'] is not None
            or len({receipt['accepted_actions'], receipt['executed_actions'], receipt['planned_actions']}) != 1):
            raise ValueError('Replay cannot cross an incomplete selected prefix or unknown receipt')
        if receipt['executed_actions'] != receipt['planned_actions']:
            raise ValueError('Replay requires the entire recorded chunk')
        if receipt['before_observation'] != before['id'] or receipt['env_start'] != count:
            raise ValueError('Replay prefix is not continuous from reset')
        if receipt['physics_start'] != physics_cursor:
            raise ValueError('Unrecorded physics steps interrupt the replay prefix')
        commands = []
        for ref in receipt['command_refs']:
            command = events[ref]
            confirmations = [e for e in events.values() if e['kind'] == 'command_receipt'
                and e['payload']['command'] == ref]
            if len(confirmations) != 1 or confirmations[0]['payload']['env_step'] != count+1:
                raise ValueError('Replay command lacks its sequential acceptance receipt')
            accepted=confirmations[0]['payload']
            if accepted['physics_start'] != physics_cursor or accepted['physics_end'] <= physics_cursor:
                raise ValueError('Replay command physics coverage is not continuous')
            physics_cursor=accepted['physics_end']
            action = command['payload']['action']
            for arm in ('left','right'):
                if len(action[f'{arm}_arm_joint_state']) != 6 or len(action[f'{arm}_ee_joint_state']) != 1:
                    raise ValueError('Replay requires full explicit two-arm commands')
            commands.append(action)
            count += 1
        if len(commands) != receipt['executed_actions'] or count != receipt['env_end']:
            raise ValueError('Replay command coverage differs from receipt')
        if receipt['physics_end'] != physics_cursor:
            raise ValueError('Replay receipt physics end differs from its commands')
        posts = [e for e in events.values() if e['kind'] == 'observation'
                 and e['payload'].get('after_receipt') == event['id']]
        if len(posts) != 1 or posts[0]['sequence'] > target['sequence']:
            raise ValueError('Replay requires the formal post-observation for every chunk')
        prefix.append({'receipt': event, 'actions': commands, 'post': posts[0]})
        before = posts[0]
    if before['id'] != target['id'] or count != target['payload']['env_step']:
        raise ValueError('Replay did not reach the requested endpoint')
    return {'run':str(root),'endpoint':observation_id,'initial':initial,'chunks':prefix,
            'env_actions':count,'source_hashes':json.loads((root/'source_hashes.json').read_text())}


def compare_body(source, actual):
    from XPolicyLab.policy.GPT_6_Astra_Direct_EEF.pose import orientation_error_deg, pose_to_values
    result = {}
    for arm in ('left','right'):
        old, new = source['state'], actual['state']
        p, q = np.asarray(old[f'{arm}_ee_pose']), np.asarray(new[f'{arm}_ee_pose'])
        xyz_old, xyz_new = pose_to_values(p), pose_to_values(q)
        result[arm] = {'max_joint_difference_rad':float(np.max(np.abs(np.asarray(old[f'{arm}_arm_joint_state'])-new[f'{arm}_arm_joint_state']))),
            'max_finger_difference_m':float(np.max(np.abs(np.asarray(source['measured_finger_joints_m'][arm])-actual['measured_finger_joints_m'][arm]))),
            'grasp_position_difference_m':float(np.linalg.norm([xyz_old[a]-xyz_new[a] for a in 'xyz'])),
            'orientation_difference_deg':orientation_error_deg(p[3:],q[3:])}
    passed = all(r['max_joint_difference_rad']<=.001 and r['max_finger_difference_m']<=.0001
        and r['grasp_position_difference_m']<=.0005 and r['orientation_difference_deg']<=.1 for r in result.values())
    return {'status':'PASS' if passed else 'NEEDS_WORK', 'measured_body_differences':result,
        'interpretation':'Engineering replay gate only; matching body state does not prove identical object/contact state.'}


def replay_prefix(environment, prefix, *, trace, session, capture, observation):
    from XPolicyLab.policy.GPT_6_Astra_Direct_EEF.inspect.types import Action, ActionChunk
    initial = compare_body(prefix['initial']['payload'],session.current_observation['data'])
    trace.event('prefix_replay_initial_comparison',initial)
    if initial['status'] != 'PASS':
        raise RuntimeError('Reset body state differs from source prefix')
    final_action = None
    for index, entry in enumerate(prefix['chunks'],1):
        if (trace.root/'STOP').exists():
            trace.event('operator_stop',{'boundary':'before_next_complete_prefix_receipt'})
            raise RuntimeError('Prefix replay stopped at a receipt boundary')
        physics_before = int(environment.sim.sim.current_time_step_index)
        actions=[Action(data={k:np.asarray(v,dtype=float) for k,v in action.items()}) for action in entry['actions']]
        if not actions:
            raise ValueError('Zero-action receipt cannot define a prefix chunk')
        chunk=ActionChunk(actions=actions,meta={'source_receipt':entry['receipt']['id'],
            'source_run':prefix['run'],'mode':'complete_scene_action_prefix_replay_no_new_plan_or_model'})
        played, stopped=execute_chunk(environment,chunk,trace=trace,session=session,
            capture=capture,observation=observation,call_id=None)
        observation=capture(index,expose=True)
        comparison=compare_body(entry['post']['payload'],session.current_observation['data'])
        trace.event('prefix_replay_comparison',{**comparison,'source_post_observation':entry['post']['id'],
            'actual_post_observation':observation.extra['evidence_id'],
            'source_images':entry['post']['payload']['images'],
            'actual_images':session.current_observation['data']['images']})
        expected_physics = entry['receipt']['payload']['physics_end']-entry['receipt']['payload']['physics_start']
        actual_physics = int(environment.sim.sim.current_time_step_index)-physics_before
        if played != len(actions) or stopped or comparison['status'] != 'PASS' or actual_physics != expected_physics:
            raise RuntimeError('Scene-action prefix replay diverged; do not continue policy')
        final_action=entry['actions'][-1]
    reference=session.evidence('prefix_replay_complete',{'source_run':prefix['run'],
        'source_endpoint':prefix['endpoint'],'actual_observation':observation.extra['evidence_id'],
        'env_actions':prefix['env_actions'],'model_calls':0,'attachment':'unknown_until_current_RGB_review',
        'interpretation':'All recorded robot commands replayed from reset with body state gates; '
            'not an arbitrary scene checkpoint or proof of identical object/contact/physics state.'})
    return observation,final_action,reference
