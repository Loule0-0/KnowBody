"""Robot-only replay/replan diagnosis; never a task-state checkpoint."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .trace import append, plain, write_once


def audit_path(manager, actions, start_action):
    """Check all segments in the configured planner model, not the full PhysX world."""
    import numpy as np
    import torch
    from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg

    arms = {}
    for robot in manager.robot_list:
        if robot.type != 'target':
            continue
        arm = robot.arm_name.removesuffix('_arm')
        key = f'{arm}_arm_joint_state'
        path = np.asarray([start_action[key]]+[action[key] for action in actions])
        if np.max(np.abs(path-path[0])) < 1e-8:
            continue
        cp = manager.planner[robot.robot_name]
        # Densify the actual policy-resampled polyline, including initial state.
        # This remains sampled validation, not a continuous collision certificate.
        dense = [path[0]]
        for a,b in zip(path[:-1],path[1:]):
            count = max(1, int(np.ceil(np.max(np.abs(b-a))/.001)))
            dense.extend(a+(b-a)*t/count for t in range(1,count+1))
        q = torch.as_tensor(np.asarray([cp._build_cspace_joint_values(v) for v in dense]),
                            dtype=torch.float32, device=cp.device_cfg.device).unsqueeze(0)
        checker = RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(
            robot_config=cp.robot_cfg, device_cfg=cp.device_cfg,
            scene_collision_checker=cp.motion_planner.scene_collision_checker,
            collision_activation_distance=0., self_collision_activation_distance=0.))
        if checker.self_collision_cost is None or checker.collision_constraint is None:
            raise ValueError('Diagnostic collision model is incomplete')
        valid = checker.validate_trajectory(q)
        world_cost,self_cost = checker.get_scene_self_collision_distance_from_joint_trajectory(q)
        arms[arm] = {'status':'PASS' if bool(valid.all().item()) else 'NEEDS_WORK',
            'samples':len(dense),'maximum_sample_joint_delta_rad':.001,
            'invalid_sample_indices':torch.where(~valid.reshape(-1))[0].cpu().tolist(),
            'max_world_collision_cost':float(world_cost.max().item()),
            'max_self_collision_cost':float(self_cost.max().item()),
            'scene_model':cp.scene_model,'planner_profile':cp.yml_path,
            'collision_cost_semantics':'positive means configured model overlap; zero is not a clearance measurement'}
    return {'status':'PASS' if arms and all(v['status']=='PASS' for v in arms.values()) else 'NEEDS_WORK',
        'scope':'sampled joint bounds + enabled self pairs + configured table only; '
                'not other arm, task objects, held tool, or a full PhysX collision certificate',
        'arms':arms}


def diagnose_tracking(environment, trace, source_run: Path, receipt_id: str, capture,
                      *, mode, planner, action_spec, session):
    import torch
    import omni.usd
    import carb.settings
    from omni.physx import get_physx_simulation_interface
    from pxr import UsdPhysics, PhysxSchema, PhysicsSchemaTools
    import numpy as np
    from XPolicyLab.policy.GPT_6_Astra_Direct_EEF.policy import EefAgentPolicy
    from XPolicyLab.policy.GPT_6_Astra_Direct_EEF import pose

    if mode not in ('replay', 'replan'):
        raise ValueError('Unknown tracking diagnostic mode')

    source_hashes = {}

    def source(identifier):
        path = source_run / 'events' / (identifier.rsplit(':', 1)[-1]+'.json')
        raw = path.read_bytes()
        source_hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    receipt = source(receipt_id)
    assert receipt['kind'] == 'receipt'
    receipt = receipt['payload']
    assert receipt['execution'] == 'complete' and receipt['unknown_command'] is None
    before = source(receipt['before_observation'])['payload']
    commands = [source(ref)['payload']['action'] for ref in receipt['command_refs']]
    assert len(commands) == receipt['executed_actions'] > 0
    write_once(trace.root/'diagnostic_source_hashes.json', source_hashes)
    manager = environment.robot_manager
    settings = carb.settings.get_settings()
    contact_processing_before = settings.get_as_bool('/physics/disableContactProcessing')
    settings.set_bool('/physics/disableContactProcessing', False)
    manager.control_manager.reset()
    articulations = {}
    held = {}
    for robot in manager.robot_list:
        if robot.type != 'target':
            continue
        arm = robot.arm_name.removesuffix('_arm')
        art = manager.robot_key[manager.robot_list.index(robot)]
        articulations[arm] = (robot, art)
        env_ids = torch.tensor([0], dtype=torch.long, device=art.device)
        q = torch.tensor([before['state'][f'{arm}_arm_joint_state']], device=art.device)
        art.write_joint_state_to_sim(q, torch.zeros_like(q), joint_ids=robot.arm_joint_indices, env_ids=env_ids)
        art.set_joint_position_target(q, joint_ids=robot.arm_joint_indices, env_ids=env_ids)
        art.set_joint_velocity_target(torch.zeros_like(q), joint_ids=robot.arm_joint_indices, env_ids=env_ids)
        # No key is attached. Closed left fingers and open right fingers are explicit interventions.
        opening = 0. if arm == 'left' else .044
        fingers = torch.full((1,len(robot.gripper_joint_indices)), opening, device=art.device)
        art.write_joint_state_to_sim(fingers, torch.zeros_like(fingers), joint_ids=robot.gripper_joint_indices, env_ids=env_ids)
        art.set_joint_position_target(fingers, joint_ids=robot.gripper_joint_indices, env_ids=env_ids)
        art.set_joint_velocity_target(torch.zeros_like(fingers), joint_ids=robot.gripper_joint_indices, env_ids=env_ids)
        art.write_data_to_sim()
        held[f'{arm}_arm_joint_state'] = before['state'][f'{arm}_arm_joint_state']
        held[f'{arm}_ee_joint_state'] = [0. if arm == 'left' else 1.]

    stage = omni.usd.get_context().get_stage()
    roots = [art.root_physx_view.prim_paths[0] for _,art in articulations.values()]
    reporting = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) and any(str(prim.GetPath()).startswith(root+'/') for root in roots):
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(0.)
            reporting.append(str(prim.GetPath()))
    phase, command_index = 'initial', -1
    contact_count = 0

    def on_contacts(headers, data):
        nonlocal contact_count
        for header in headers:
            actors = [str(PhysicsSchemaTools.intToSdfPath(getattr(header,k))) for k in ('actor0','actor1')]
            if not any(any(actor.startswith(root+'/') for root in roots) for actor in actors):
                continue
            contacts = data[header.contact_data_offset:header.contact_data_offset+header.num_contact_data]
            append(trace.root/'contacts.jsonl', {'phase':phase,'command_index':command_index,
                'physics_step':int(environment.sim.sim.current_time_step_index),
                'type':int(header.type), 'actors':actors,
                'colliders':[str(PhysicsSchemaTools.intToSdfPath(getattr(header,k))) for k in ('collider0','collider1')],
                'contacts':[{'impulse':list(c.impulse),'separation':float(c.separation)} for c in contacts]})
            contact_count += 1

    subscription = get_physx_simulation_interface().subscribe_contact_report_events(on_contacts)
    trace.event('diagnostic_setup', {'source_run':str(source_run), 'source_receipt':receipt_id, 'mode':mode,
        'intervention':'robot joint positions only; zero velocities; no held key; left fingers closed; '
                       'original task objects remain at reset positions; simulation collision/gains unchanged',
        'not_a_claim':'not complete scene restoration, not a policy-visible task trajectory',
        'disable_contact_processing_before':contact_processing_before,
        'disable_contact_processing_during_diagnostic':settings.get_as_bool('/physics/disableContactProcessing'),
        'contact_reporting_rigid_bodies':reporting}, visibility='recorded_not_exposed')

    def telemetry(label):
        result = {}
        for arm,(robot,art) in articulations.items():
            indices = robot.arm_joint_indices
            result[arm] = plain({'joint_position_rad':art.data.joint_pos[0,indices],
                'joint_velocity_rad_s':art.data.joint_vel[0,indices],
                'runtime_position_limits_rad':art.data.joint_pos_limits[0,indices],
                'physx_position_targets_rad':art.root_physx_view.get_dof_position_targets()[0,indices],
                'implicit_pd_estimated_torque_Nm':art.data.computed_torque[0,indices],
                'implicit_pd_clipped_estimate_Nm':art.data.applied_torque[0,indices],
                'physx_projected_joint_forces':art.root_physx_view.get_dof_projected_joint_forces()[0,indices]})
        trace.event('tracking_telemetry', {'phase':label,'command_index':command_index,
            'env_step':int(environment.take_action_cnt[0]), 'arms':result}, visibility='recorded_not_exposed')
        return result

    try:
        initial = telemetry('initialized')
        capture(0, expose=False)
        def run_actions(label, actions):
            nonlocal phase, command_index
            phase = label
            for command_index, action in enumerate(actions):
                trace.event('diagnostic_command', {'phase':phase,'command_index':command_index,
                    'action':action}, visibility='recorded_not_exposed')
                environment.take_action(action)
                final = telemetry(phase)
                if environment.is_episode_end():
                    raise RuntimeError('Diagnostic environment ended before sequence completed')
            capture(0, expose=False)
            return final

        initial_hold = run_actions('initial_hold', [held]*5)
        diagnostic_plan = {'mode':mode,'source_goal_flange_world':receipt['plan']['trace']['arms']['left']['target_pose']}
        actions = commands
        if mode == 'replan':
            class NoModelClient:
                def complete(self, *_args, **_kwargs):
                    raise AssertionError('Tracking diagnostic must not request a model')

            observation = capture(0, expose=True)
            target = np.asarray(diagnostic_plan['source_goal_flange_world'])
            named = {f'left_{axis}':value for axis,value in pose.pose_to_values(target).items()}
            from .body_guard import BodyPathGuard
            policy = EefAgentPolicy(action_spec=action_spec, client=NoModelClient(), env={},
                                    planner=planner, research=session,
                                    path_validator=BodyPathGuard(manager,session))
            outcome = policy._handle_motion('move_eef', {'targets':named,
                'note':'Robot-only diagnostic: same Cartesian goal as source; no model inference'}, observation)
            diagnostic_plan.update(named_grasp_goal=named,tool_result=outcome.tool_result,
                                   planner_returned_chunk=outcome.chunk is not None)
            actions = [] if outcome.chunk is None else [plain(action.data) for action in outcome.chunk.actions]
            if actions:
                diagnostic_plan['path_audit'] = audit_path(manager, actions, observation.state)
                if diagnostic_plan['path_audit']['status'] != 'PASS':
                    actions = []
            trace.event('diagnostic_replan', {**diagnostic_plan,'qualified_actions':actions},
                        visibility='recorded_not_exposed')
        if actions:
            run_actions('source_replay' if mode == 'replay' else 'replanned_execution', actions)
            final = run_actions('final_hold', [actions[-1]]*15)
        else:
            final = initial_hold
        final_observation = capture(0, expose=False)
        target = np.asarray(diagnostic_plan['source_goal_flange_world'])
        actual = np.asarray(final_observation.state['left_ee_pose'])
        diagnostic_plan.update(arrival_flange_position_error_m=float(np.linalg.norm(actual[:3]-target[:3])),
            arrival_rotation_error_deg=pose.orientation_error_deg(actual[3:],target[3:]),
            actual_final_flange_world=actual.tolist(),executed_path_actions=len(actions))
        return {'status':'PASS_tracking_diagnostic_recorded_not_task_acceptance',
            'robot_only_replay':mode=='replay', 'robot_only_replan':mode=='replan',
            'diagnostic_plan':diagnostic_plan,
            'model_requests':0, 'source_command_count':len(commands),
            'env_steps':int(environment.take_action_cnt[0]),
            'initial_hold_env_steps':5,'final_hold_env_steps':15 if actions else 0,
            'initial':initial,'after_initial_hold':initial_hold,'after_final_hold':final,
            'contact_headers_recorded':contact_count,
            'source_records_unchanged':all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in source_hashes.items())}
    finally:
        subscription.unsubscribe()
