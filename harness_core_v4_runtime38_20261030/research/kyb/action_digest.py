"""Historical request, execution coverage and body measurements, kept distinct."""
from __future__ import annotations

from .state import body_transition, requested_change


def digest(receipt, *, before=None, post=None, physics_dt_s=None, request=None):
    plan = receipt.get('plan', {})
    result = {
        'requested': {
            'named_dimensions': {arm: detail.get('named', {}) for arm, detail in plan.get('trace', {}).get('arms', {}).items()},
            'planner_targets': {arm: detail.get('target_pose') for arm, detail in plan.get('trace', {}).get('arms', {}).items()},
            'limits': {key: (request or {}).get(key) for key in
                       ('preview', 'execute_max_s', 'max_duration_s', 'max_joint_travel_deg')},
            'limits_source': 'recorded_policy_arguments' if request is not None else 'unknown_no_linked_policy_decision',
            'limits_absent_from_request': [key for key in ('preview', 'execute_max_s', 'max_duration_s', 'max_joint_travel_deg')
                                          if request is not None and key not in request],
            'frame': 'Recorded action contract; named world positions and planner flange poses are distinct.'},
        'plan': {key: plan.get(key) for key in ('full_planned_actions', 'full_duration_s',
                    'cumulative_joint_travel_deg', 'max_single_joint_travel_deg',
                    'selected_actions', 'selected_stop_reason', 'full_plan_ref')},
        'execution': {key: receipt.get(key) for key in ('execution', 'executed_actions',
                      'accepted_actions', 'planned_actions', 'stop_reason', 'failure_type', 'unknown_command')},
        'source_dependencies': receipt.get('dependencies', {}),
        'measured': None,
        'qualification': 'Historical observations, not a current action plan, preference, contact or holding proof. '
                         'Command completion does not certify target arrival. Missing fields remain unknown.'}
    if before is not None and post is not None:
        transition = body_transition(before, receipt, post, physics_dt_s)
        goal_difference = transition.pop('measured_target_residual')
        scope = plan.get('selected_stop_reason')
        whole = (scope == 'full_plan_selected' and receipt.get('execution') == 'complete'
            and receipt.get('unknown_command') is None and bool(receipt.get('executed_actions'))
            and receipt.get('executed_actions') == plan.get('selected_actions'))
        result['requested']['world_change_from_measured_before'] = requested_change(receipt, before)
        result['measured'] = {
            'pose_sources': 'Exact source-time poses are in the separately delivered before/post body observations.',
            'arm_joint_angles_rad': {
                phase: {key: value for key, value in payload.get('state', {}).items()
                        if key.endswith('_arm_joint_state')}
                for phase, payload in (('before', before), ('post', post))},
            'world_body_transition': {key:transition[key] for key in ('measured_change','physics_steps','physics_duration_s',
                 'physics_dt_s','readout_qualification','qualification')},
            'requested_goal_difference': goal_difference,
            'goal_comparison_scope': 'full_selected_plan' if whole else
                'full_request_only_not_selected_endpoint; arrival_not_assessed; do not treat a deliberate prefix as tracking failure',
            'selected_endpoint_pose_comparison': 'unknown_no_selected_endpoint_EEF_pose' if not whole
                else 'same_as_full_requested_goal'}
    return result
