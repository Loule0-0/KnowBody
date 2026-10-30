"""Consume observation-time coordinates. Never reinterpret history with live constants."""
from __future__ import annotations

import math
from .trace import plain


def evidence_state(payload: dict) -> dict:
    canonical = payload.get('canonical_state')
    return plain({'raw_state':payload.get('state', {}),
        'coordinate_contract':canonical.get('coordinate_contract') if canonical else None,
        'canonical_arms':canonical.get('arms', {}) if canonical else {},
        'grasp_point_dimensions':{k:v for arm in (canonical or {}).get('arms', {}).values()
                                   for k,v in arm['action_dimensions'].items()},
        'measured_finger_joint_displacements_m':payload.get('measured_finger_joints_m', {}),
        'coordinate_derivation':payload.get('coordinate_derivation'),
        'qualification':('explicit_offline_derivation_not_original_policy_input' if payload.get('coordinate_derivation')
                          else 'observation_time_contract') if canonical else 'unknown_no_coordinate_contract',
        'conventions':{'raw_ee_pose':'Adapter raw telemetry; do not infer its reference point from its key.',
            'action_positions':'Use the recorded action frame and units in coordinate_contract.',
            'warning':'Missing contract forbids derived pose comparison. Finger opening does not prove holding.'}})


def evidence_receipt(receipt: dict | None) -> dict | None:
    if receipt is None:
        return None
    return {**receipt, 'coordinate_conventions': {
        'plan.trace.arms.*.target_pose':'Adapter planner target; use its source coordinate contract.',
        'plan.trace.arms.*.named':'Requested action dimensions, not realized movement.',
        'warning':'Requested, submitted and measured actions are distinct.'}}


def _rotate(q, vector):
    w,x,y,z=q; norm=math.sqrt(sum(v*v for v in q)); w,x,y,z=[v/norm for v in (w,x,y,z)]
    a,b,c=vector
    return [(1-2*(y*y+z*z))*a+2*(x*y-z*w)*b+2*(x*z+y*w)*c,
            2*(x*y+z*w)*a+(1-2*(x*x+z*z))*b+2*(y*z-x*w)*c,
            2*(x*z-y*w)*a+2*(y*z+x*w)*b+(1-2*(x*x+y*y))*c]


def tracking_residual(receipt: dict, post: dict) -> dict:
    canonical = post.get('canonical_state')
    target_arms = {}
    if canonical:
        contract = canonical['coordinate_contract']
        for arm, detail in receipt.get('plan',{}).get('trace',{}).get('arms',{}).items():
            target=detail.get('target_pose')
            if target is None:
                continue
            offset = _rotate(target[3:7], contract['flange_to_action_translation_m'])
            target_arms[arm]={'action_pose_world_m_wxyz':[*(a+b for a,b in zip(target[:3],offset)),*target[3:7]],
                              'action_dimensions':{},'part_id':f'arm:{arm}'}
    target_payload={'canonical_state':{'coordinate_contract':canonical['coordinate_contract'],'arms':target_arms}} if canonical else {}
    return {**compare_states(target_payload, post),
        'reference':'actual_post_minus_requested_target; recorded action frame',
        'execution':receipt.get('execution','unknown'), 'tracking_verdict':'not_certified_by_command_execution'}


def body_transition(before: dict, receipt: dict, post: dict, physics_dt_s: float | None) -> dict:
    start,end=evidence_state(before),evidence_state(post)
    comparison=compare_states(before,post)
    steps=receipt.get('physics_end',0)-receipt.get('physics_start',0)
    return {'before_grasp':start['grasp_point_dimensions'],'post_grasp':end['grasp_point_dimensions'],
        'before_finger_joints_m':before.get('measured_finger_joints_m',{}),
        'post_finger_joints_m':post.get('measured_finger_joints_m',{}),
        'measured_change':comparison,'physics_steps':steps,
        'physics_duration_s':steps*physics_dt_s if physics_dt_s is not None else None,'physics_dt_s':physics_dt_s,
        'measured_target_residual':tracking_residual(receipt,post),
        'readout_qualification':'host_measured' if comparison['coordinate_comparison']=='same_recorded_contract' else 'unknown',
        'qualification':'Body sensor transition, not object motion, contact or holding proof. '
            'A finger-gap change with unchanged closure deserves visual attachment review; deformation/contact identity remain unknown.'}


def compare_states(source: dict, current: dict) -> dict:
    a,b=source.get('canonical_state'),current.get('canonical_state')
    comparable=bool(a and b and a['coordinate_contract']['id']==b['coordinate_contract']['id'])
    differences={}
    if comparable:
        for arm in sorted(a['arms'].keys() & b['arms'].keys()):
            old,new=a['arms'][arm]['action_pose_world_m_wxyz'],b['arms'][arm]['action_pose_world_m_wxyz']
            norm=math.sqrt(sum(v*v for v in old[3:7])*sum(v*v for v in new[3:7]))
            dot=min(1.,max(0.,abs(sum(x*y for x,y in zip(old[3:7],new[3:7])))/norm))
            differences[arm]={'orientation_difference_deg':math.degrees(2*math.acos(dot)),
                'grasp_position_current_minus_source_m':[y-x for x,y in zip(old[:3],new[:3])]}
    for arm, old in source.get('measured_finger_joints_m',{}).items():
        new=current.get('measured_finger_joints_m',{}).get(arm)
        source_body=source.get('dependencies',{}).get('body') or (a or {}).get('arms',{}).get(arm,{}).get('part_version')
        current_body=current.get('dependencies',{}).get('body') or (b or {}).get('arms',{}).get(arm,{}).get('part_version')
        # Fingers on different bodies are not the same sensor channels.
        if new is not None and source_body is not None and source_body == current_body:
            differences.setdefault(arm,{})['finger_displacement_current_minus_source_m']=[y-x for x,y in zip(old,new)]
    return {'measured_differences':differences,'coordinate_comparison':'same_recorded_contract' if comparable else 'unknown_or_different_contract',
        'interpretation':'Known numerical differences only; no applicability threshold or scene/contact/retention equivalence.'}


def compact_body_state(payload: dict) -> dict:
    """Small source-time end-effector read; no cameras, projections or live constants."""
    canonical=payload.get('canonical_state')
    if not canonical:return {'qualification':'unknown_no_coordinate_contract','arms':{}}
    contract=canonical['coordinate_contract'];arms={}
    for arm,data in canonical['arms'].items():
        grasp=data['action_pose_world_m_wxyz']
        offset=_rotate(grasp[3:7],contract['flange_to_action_translation_m'])
        arms[arm]={'grasp_world_m_wxyz':grasp,
            'flange_world_m_wxyz':[*(x-y for x,y in zip(grasp[:3],offset)),*grasp[3:7]],
            'finger_displacements_m':data.get('finger_displacements_m')}
    return {'coordinate_contract_id':contract['id'],'arms':arms,
        'qualification':'source_time_body_measurement_not_contact_or_holding_proof'}


def requested_change(receipt: dict, before: dict) -> dict:
    """Named requested dimensions minus measured start, using the source contract."""
    old = evidence_state(before)['grasp_point_dimensions']
    return {key: value - old[key]
            for detail in receipt.get('plan', {}).get('trace', {}).get('arms', {}).values()
            for key, value in detail.get('named', {}).items() if key in old}
