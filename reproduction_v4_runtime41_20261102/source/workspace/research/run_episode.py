"""Run the existing DirectEEF policy with grounded trace/memory instrumentation."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import shutil
import signal
import time
from dataclasses import replace
from pathlib import Path

from kyb.trace import RunTrace, plain, write_once
from kyb.codex_cli import CodexCliClient
from kyb.memory import MemoryStore
from kyb.session import MemorySession
from kyb.execution import execute_chunk
from kyb.body_visual import BodyVisualReference
from kyb.robodojo_adapter import canonical_state

ROOT = Path("/home/lzy/KnowYourBody_RoboDojo")
REPO = ROOT / "RoboDojo"
DATA = Path("/data/KnowYourBody_RoboDojo")
PLANNER_CONFIG = ROOT / 'artifacts/body_collision_candidate/original_padding/curobo.yml'
EXPERIMENT = None
parser = argparse.ArgumentParser()
parser.add_argument("--experiment-dir", required=True,type=Path)
parser.add_argument("--task", required=True)
parser.add_argument("--layout", type=int, required=True)
parser.add_argument('--call-budget', type=int, required=True)
parser.add_argument('--budget-ledger', type=Path, required=True)
parser.add_argument('--manager-call-budget', type=int, required=True)
parser.add_argument('--manager-budget-ledger', type=Path, required=True)
parser.add_argument('--input-memory', type=Path, required=True)
args = parser.parse_args()
from kyb.resources import seal_kit_lease
seal_kit_lease(int(os.environ.pop('KYB_GPU_EPISODE_LEASE_FD')))
EXPERIMENT=args.experiment_dir.resolve()
protocol=json.loads((EXPERIMENT/'PROTOCOL.json').read_text())
from kyb.perception import select as select_perception
perception=select_perception(protocol['perception_profile'])
layouts=json.loads((EXPERIMENT/'LAYOUTS.json').read_text())
from kyb.runtime import configuration,select
try:
    selected=select(layouts,args.task,args.layout)
    runtime_root,server_port,runtime_gpu=configuration(protocol)
except (KeyError,TypeError,ValueError) as error:parser.error(str(error))
runtime_root.mkdir(parents=True,exist_ok=True)
(runtime_root/'kit_logs').mkdir(exist_ok=True)
if (args.call_budget,args.manager_call_budget)!=(protocol['primary']['calls_per_episode'],protocol['manager']['calls_per_episode']):
    parser.error('Both budgets must match the authorized protocol')
from kyb.budget import CallBudget, BudgetExhausted
budget=CallBudget(args.budget_ledger,args.call_budget)
manager_budget=CallBudget(args.manager_budget_ledger,args.manager_call_budget)
run_id = os.environ["KYB_RUN_ID"]
env_vars = dict(os.environ)
env_vars.update(L4_INSPECT_MODEL="gpt-6-astra", L4_INSPECT_PLANNER="astra",
    L4_INSPECT_API_STYLE="responses", L4_INSPECT_REASONING_EFFORT="medium",
    L4_INSPECT_DEPTH="surface_queries" if perception.depth else "off")
os.environ.update(env_vars)
os.environ["ROBODOJO_RUN_ID"] = run_id

input_manifest=json.loads(args.input_memory.with_suffix('.manifest.json').read_text())
initial_revision=input_manifest['revision']
trace = RunTrace(DATA / "runs" / run_id, {
    "task":args.task,"layout":args.layout,"protocol":"official",
    "experiment_order":selected['order'],"runtime":protocol['runtime'],
    "model":"gpt-6-astra","effort":"medium","transport":"codex_cli",
    "manager_model":"gpt-6.1-sol","manager_effort":"xhigh",
    "authentication":"native_cli_saved_login",
    "perception_profile":perception.name,
    "perception_protocol":("RGB-D_local_readonly_surface_queries;unmodified_RGB_no_task_object_GT" if perception.depth
        else "RGB_proprioception_public_instruction;no_depth_camera_calibration_or_task_object_GT"),
    "initial_knowledge_revision":initial_revision,"input_snapshot":input_manifest,
    "input_memory":str(args.input_memory),"memory_read_mode":"online_per_transaction_frontiers",
    "reasoning_visibility":"opaque_protocol_items_stored_locally_never_rendered",
    "research_source":str(Path(inspect.getfile(MemorySession)).parent.parent),
    "call_allowance":budget.view(),"budget_ledger":str(args.budget_ledger),
    "manager_call_allowance":manager_budget.view(),"manager_budget_ledger":str(args.manager_budget_ledger),
    "memory_protocol":"knowledge_first_structured_cases_native_episode_session",
    "retrieval":protocol['retrieval'],
    "prefix_replay":None,
    "experiment_protocol_sha256":hashlib.sha256((EXPERIMENT/'PROTOCOL.json').read_bytes()).hexdigest(),
    "layout_manifest_sha256":hashlib.sha256((EXPERIMENT/'LAYOUTS.json').read_bytes()).hexdigest(),
    "harness_version":json.loads((EXPERIMENT/'HARNESS.json').read_text())['version'],
})
store, _ = MemoryStore.fork_snapshot(args.input_memory, trace.root / "memory.sqlite")
perception.check_memory(store)
from kyb.embeddings import VectorIndex
from kyb import hybrid,library
retrieval_index=VectorIndex(protocol['retrieval']['cache_path'],
    line_id=protocol['retrieval']['line_id'],profile=perception.name)
try:
    retrieval_frontier={row[0]:row[1] for row in store.db.execute('SELECT run,MAX(sequence) FROM evidence GROUP BY run')}
    retrieval_rows=library.visible_knowledge(store,frontier=retrieval_frontier)
    retrieval_index.encoder.encode([])  # validate/load even an empty library, before Kit/hosted requests
    _,_,retrieval_preparation=retrieval_index.prepare(
        {(row['id'],row['revision']):hybrid.index_text(row) for row in retrieval_rows},schema=hybrid.SCHEMA)
except Exception as error:
    trace.event('retrieval_index_not_ready',{'error':type(error).__name__,'reason':str(error),
        'hosted_model_calls':0,'simulation_actions':0})
    retrieval_index.close()
    store.close()
    raise
trace.event('retrieval_index_preparation',{**retrieval_preparation,'policy_id':hybrid.POLICY,
    'scope':'Derived local CPU index only; no new facts, hosted requests or physics',
    'line_id':protocol['retrieval']['line_id'],'profile':perception.name})

source_files = [Path(__file__)]
source_files += list(Path(inspect.getfile(MemorySession)).parent.glob("*.py"))
source_files += [Path(inspect.getfile(MemorySession)).parent/'retrieval_encoder.json']
from kyb.execution_contract import NATIVE_SOURCES
source_files += [REPO/name for name in NATIVE_SOURCES]
source_files += list((REPO / "XPolicyLab/policy/GPT_6_Astra_Direct_EEF").glob("*.py"))
source_files += list((REPO / "XPolicyLab/policy/GPT_6_Astra_Direct_EEF/inspect").glob("*.py"))
planner_sources = [REPO/'env/planner_manager/curobo_planner.py',
                   REPO/'env/robot_manager/robot_manager.py',
                   REPO/'env/robot_manager/robot_class/x5.py',
                   PLANNER_CONFIG, REPO/'Assets/Robots/x5/X5A.urdf']
source_files += planner_sources
source_files += [REPO/'env/camera_manager/camera_manager.py', REPO/'utils/rotations.py',
    REPO/'env_cfg/camera/template.py', REPO/'Assets/Robots/x5/robot_config.yml',
    REPO/'Assets/Robots/x5/meshes/link7.STL', REPO/'Assets/Robots/x5/meshes/link8.STL',
    REPO/'env/camera_manager/capture/camera_view.py', REPO/'env/camera_manager/capture/tiled_capture_manager.py',
    REPO/'env/observation_manager/obs_manager.py', REPO/'env_cfg/arx_x5.yml',
    REPO/'env_cfg/camera/camera_config.yml']
source_hashes = {}
for source in source_files:
    source_hashes[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
    destination = trace.root / "sources" / source.relative_to(ROOT)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
write_once(trace.root / "source_hashes.json", source_hashes)
print(f"[KYB] run: {trace.root}", flush=True)

from isaaclab.app import AppLauncher
from env.camera_manager.capture.render_sync import zero_delay_kit_args, zero_delay_active, dispatcher_time
app = AppLauncher(headless=True, enable_cameras=True, device=f"cuda:{runtime_gpu}",
    kit_args=f"--portable --portable-root {runtime_root}/kit "
             f"--/app/logDir={runtime_root}/kit_logs "
             f"--ext-folder {runtime_root}/extensions "
             "--/telemetry/enable=false " + zero_delay_kit_args()).app
import numpy as np
from omegaconf import OmegaConf
from env.global_configs import ENV_CONFIG_PATH
from src.eval_client.eval_env import create_eval_env
from utils.load_file import load_yaml
from utils.pipeline_utils import process_config, process_randomization
from XPolicyLab.utils.process_data import encode_image_bit
from XPolicyLab.policy.GPT_6_Astra_Direct_EEF import deploy as eef_deploy
from XPolicyLab.policy.GPT_6_Astra_Direct_EEF.inspect import deploy as joint_deploy
from kyb.policy import EefAgentPolicy
from XPolicyLab.policy.GPT_6_Astra_Direct_EEF.docs import eef_docs_from_joint_docs

environment = None
session = None
policy = None
client = None
manager_client = None
report = {"status": "NEEDS_WORK", "run": str(trace.root), "model_policy_test": False}
started = time.monotonic()
image_index = 0
body_visual = BodyVisualReference(REPO/'Assets/Robots/x5/X5A.urdf', .1501) if perception.calibrated_camera else None


def capture(policy_step: int, *, expose: bool):
    global image_index
    image_index += 1
    physics_before = environment.sim.sim.current_time_step_index
    dispatched_before = dispatcher_time()
    observation = joint_deploy._observation(environment, policy_step)
    timing = {"zero_delay_active": zero_delay_active(),
              "dispatcher_before": dispatched_before, "dispatcher_after": dispatcher_time(),
              "physics_step_before": physics_before,
              "physics_step_after": environment.sim.sim.current_time_step_index,
              "physics_time": environment.sim.sim.current_time}
    assert timing["zero_delay_active"] and timing["physics_step_before"] == timing["physics_step_after"]
    assert timing["dispatcher_after"] is not None and timing["dispatcher_after"] != dispatched_before
    images = {}
    for name, color in observation.images.items():
        assert color.dtype == np.uint8 and color.shape == (480, 640, 3) and np.isfinite(color).all()
        path = trace.root / "observations" / f"{image_index:06d}_{name}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = encode_image_bit(color, quality=95)
        with path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        images[name] = {"path": str(path), "sha256": hashlib.sha256(data).hexdigest()}
    fingers = {robot.arm_name.removesuffix("_arm"): plain(
        environment.robot_manager.get_end_effector_real_val(robot, env_idx_list=[0])[0])
        for robot in environment.robot_manager.robot_list if robot.type == "target"}
    depths={}
    if perception.depth and set(observation.extra['depth_frames'])!=set(images):
        raise RuntimeError('Required synchronized RGB-D camera sets differ')
    if not perception.depth and observation.extra.get('depth_frames'):
        raise RuntimeError('RGB-only simulator unexpectedly produced depth')
    for name,depth in observation.extra['depth_frames'].items():
        if depth.shape!=(480,640) or depth.dtype!=np.float32:
            raise RuntimeError('Required float32 optical-Z depth shape changed')
        path=trace.root/'observations'/f'{image_index:06d}_{name}_depth.npy'
        with path.open('xb') as stream:
            np.save(stream,depth,allow_pickle=False);stream.flush();os.fsync(stream.fileno())
        depths[name]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'encoding':'float32_optical_Z_metres','aligned_RGB_sha256':images[name]['sha256']}
    payload = {"state": plain(observation.state), "images": images,
               "canonical_state":canonical_state(plain(observation.state),fingers,session.dependencies["body"]),
               "measured_finger_joints_m": fingers, "capture_timing": timing,
               "instruction": observation.instruction, "policy_step": policy_step,
               "env_step": int(environment.take_action_cnt[0]),
               "remaining_steps": observation.remaining_steps,
               "after_receipt": session.last_receipt["id"] if session.last_receipt else None,
               "dependencies": dict(session.dependencies)}
    if perception.depth:
        payload['depth_snapshots']=depths
    if perception.calibrated_camera:
        payload['body_visual_reference']=body_visual.capture(environment.camera_manager,observation.state,fingers,images)
    if expose:
        reference = session.evidence("observation", payload)
    else:
        reference = trace.event("execution_frame", payload, visibility="recorded_not_exposed")
        session.store.register(reference,plain(payload))
    return replace(observation, extra={**observation.extra,
                   "measured_finger_joints_m": fingers, "evidence_id": reference["id"],
                   "env_step": payload["env_step"]})


signal.signal(signal.SIGINT, signal.default_int_handler)
try:
    eval_cfg = load_yaml(Path(ENV_CONFIG_PATH) / "arx_x5.yml")
    eval_cfg.update(task_name=args.task, num_envs=1, device_id=runtime_gpu, eval_batch=False,
        policy_name="demo_policy", additional_info=run_id, seed=0, eval_num=1,
        physx_monitor_enabled=False)
    eval_cfg['observation']['vision'].update(depth=perception.depth,approximate_depth=False)
    cfg = OmegaConf.create({
        "sim": load_yaml(Path(ENV_CONFIG_PATH) / "sim/sim_config.yml"),
        "scene": load_yaml(Path(ENV_CONFIG_PATH) / "scene/default.yml"),
        "camera": load_yaml(Path(ENV_CONFIG_PATH) / "camera/camera_config.yml"),
        "robot": load_yaml(Path(ENV_CONFIG_PATH) / "robot/dual_x5.yml"),
        "task_env": load_yaml(REPO / f"task/RoboDojo/config/{args.task}.yml"),
        "eval_cfg": eval_cfg,
        "deploy_cfg": {"policy_name": "demo_policy", "host": "127.0.0.1", "port": server_port}})
    cfg.sim.scene.num_envs = 1
    cfg.sim.seed = [0]
    cfg.camera.default_frequency = 25
    for annotators in cfg.camera.annotator.values():
        for name in list(annotators):
            if isinstance(annotators[name],dict) and annotators[name].get('type') in ('distance_to_image_plane','distance_to_camera'):
                del annotators[name]
        if perception.depth and annotators.get('enabled',False):
            annotators['depth_capture']={'type':'distance_to_image_plane','device':'cpu'}
    cfg = process_randomization(cfg)
    cfg, _ = process_config(cfg, task_name=args.task)
    for robot_config in cfg.robot.robots:
        robot_config.planner_config_path = str(PLANNER_CONFIG)
    environment = create_eval_env(cfg, app)
    # Official SeedManager enumerates the complete numeric filename catalog.
    # Restore that metadata mapping when only selected layout files are installed.
    inventory=json.loads((EXPERIMENT/'layout_inventory.json').read_text())['files']
    from kyb.layout import bind_layout
    native_layout_index=bind_layout(environment.seed_manager,selected,inventory,DATA)
    trace.event('layout_binding',{'catalog_filename_id':args.layout,'native_layout_index':native_layout_index,
        'selected_path':str(DATA/selected['path']),'catalog_source_sha256':hashlib.sha256((EXPERIMENT/'layout_inventory.json').read_bytes()).hexdigest()})
    environment.reset(seed=[native_layout_index])
    environment.run_reward()
    spec = joint_deploy._action_spec(environment)
    sdk_action_docs = eef_docs_from_joint_docs(spec.docs)
    from kyb.operating_notes import for_perception
    spec = replace(spec, docs=for_perception(sdk_action_docs, perception))
    trace.event('numeric_action_contract',{'labels':list(spec.labels),'low':plain(spec.low),
        'high':plain(spec.high),'max_step':plain(spec.max_step),'control_hz':spec.control_hz,
        'qualification':'Host-only numeric execution configuration for reproducible command tests; not a model observation.'},visibility='private')
    body_hash = hashlib.sha256((REPO / "Assets/Robots/x5/X5A.urdf").read_bytes()).hexdigest()
    actual_camera_config=OmegaConf.to_container(cfg.camera,resolve=True)
    rgb_camera_identity=json.loads(json.dumps(actual_camera_config))
    for annotators in rgb_camera_identity['annotator'].values():
        for name in list(annotators):
            if isinstance(annotators[name],dict) and annotators[name].get('type')=='distance_to_image_plane':
                del annotators[name]
    camera_hash = hashlib.sha256(json.dumps(rgb_camera_identity,sort_keys=True).encode()).hexdigest()
    depth_protocol={'type':'distance_to_image_plane','encoding':'float32_optical_Z_metres',
        'approximate_depth':False,'capture':'same_raw_snapshot_as_RGB',
        'surface_query_code':source_hashes[str(Path(inspect.getfile(MemorySession)).parent/'surface.py')],
        'body_surface_geometry_code':source_hashes[str(Path(inspect.getfile(MemorySession)).parent/'body_visual.py')]}
    depth_hash=hashlib.sha256(json.dumps(depth_protocol,sort_keys=True).encode()).hexdigest()
    planner_setup = {'files': {str(p.relative_to(ROOT)):source_hashes[str(p)] for p in planner_sources},
        'mounts':{robot.arm_name:[float(v) for v in robot.entity_origin_pose] for robot in environment.robot_manager.robot_list
                  if robot.type=='target'}, 'control_hz':spec.control_hz,
        'physics_dt':environment.sim.sim.get_physics_dt()}
    from kyb.execution_contract import build as execution_contract
    execution_protocol,execution_hash=execution_contract(control_hz=spec.control_hz,
        physics_dt=environment.sim.sim.get_physics_dt(),
        effective_parameters=EefAgentPolicy.execution_parameters(spec,env_vars),action_contract=sdk_action_docs)
    session = MemorySession(trace, store, task=args.task, perception_profile=perception.name,retrieval_index=retrieval_index, dependencies={
        "body": body_hash, "cameras": camera_hash, **({'depth':depth_hash} if perception.depth else {}),
        'execution':execution_hash,
        'planner': hashlib.sha256(json.dumps(planner_setup,sort_keys=True).encode()).hexdigest(),
        "attachment:left": f"{run_id}:left_unknown", "attachment:right": f"{run_id}:right_unknown"},
        frozen_revision=None)
    session.budget=budget
    from XPolicyLab.policy.GPT_6_Astra_Direct_EEF.inspect.policy import task_recipe,use_task_recipe
    recipe=task_recipe(args.task) if use_task_recipe(env_vars) else None
    session.public_task_recipe=recipe[1] if recipe else None
    observation = capture(0, expose=True)
    from kyb.state import evidence_state
    reset_state=evidence_state(session.current_observation['data'])
    known_config={"action_spec": spec.docs, "body_hash": body_hash,
                                     'perception_profile':perception.name,
                                     'coordinate_contract':reset_state['coordinate_contract'],
                                     'episode_start_reference':{
                                         'source_observation':session.current_observation['id'],
                                         'grasp_point_dimensions':reset_state['grasp_point_dimensions'],
                                         'qualification':'Fixed episode-start return reference, not current state.'},
                                     'execution_protocol':execution_protocol,
                                     'planner_setup':planner_setup,
                                     "source": "public_configuration_not_trajectory_learning"}
    if perception.calibrated_camera:
        known_config.update(actual_RGBD_camera_config=actual_camera_config,RGB_camera_identity=rgb_camera_identity)
    if perception.depth:
        known_config['depth_protocol']=depth_protocol
    session.evidence('known_config',known_config)
    planner = eef_deploy._planner(environment)
    client = CodexCliClient(trace.root,budget=budget,timeout=600)
    manager_client = CodexCliClient(trace.root / "management", role="manager", budget=manager_budget, timeout=600)
    session.attach_manager(manager_client)
    client.knowledge_provider=session.manager.prefix_view
    from kyb.body_guard import BodyPathGuard
    policy = EefAgentPolicy(action_spec=spec, env=env_vars, client=client,
                            planner=planner, research=session,
                            path_validator=BodyPathGuard(environment.robot_manager,session))
    step = observation.step
    while not environment.is_episode_end():
        if session.action_allowance_low():
            trace.event('allowance_action_boundary',budget.view())
            report['allowance_action_stop']=True
            break
        if (trace.root / "STOP").exists():
            trace.event("operator_stop", {"boundary": "post_observation_before_policy"})
            report["diagnostic_stop"] = True
            break
        chunk = policy.act(observation)
        if (trace.root / "STOP").exists():
            trace.event("operator_stop", {"boundary": "policy_return_before_execution",
                                         "ignored_call": client.last_attempt_id})
            report["diagnostic_stop"] = True
            break
        played, stop = execute_chunk(environment, chunk, trace=trace, session=session,
            capture=capture, observation=observation, call_id=client.last_attempt_id)
        policy.confirm_executed(played)
        step += 1
        observation = capture(step, expose=True)
        if stop:
            break
    trace.event("terminal_pending", {"observation": observation.extra["evidence_id"],
                                    "receipt": session.last_receipt["id"] if session.last_receipt else None})
    trace.event("evaluation", {"official_success": bool(environment.end_flag[0] and environment.success[0]),
        "official_end_flag": bool(environment.end_flag[0]),
        "task_instruction_process_verified": False}, visibility="private")
    report.update(status=("NEEDS_WORK" if report.get("diagnostic_stop") else "PASS"),
                  policy_calls=policy.calls, env_steps=int(environment.take_action_cnt[0]),
                  env_step_limit=environment.step_lim)
    report['task_outcome']='PASS' if bool(environment.end_flag[0] and environment.success[0]) else 'NEEDS_WORK'
    write_once(trace.root/'terminal_checkpoint.json', {
        'report_before_review':report,
        'final_observation_ref':session.current_observation['id'],
        'last_receipt_ref':session.last_receipt['id'] if session.last_receipt else None,
        'policy_native_session_id':client.session_id,
        'manager_native_session_id':manager_client.session_id,
        'source_hashes_sha256':hashlib.sha256((trace.root/'source_hashes.json').read_bytes()).hexdigest(),
        'qualification':'Physics has ended; private evaluation is for report recovery only, never model input.'})
    if not report.get('diagnostic_stop') and not (trace.root/'STOP').exists():
        from kyb.terminal import review_terminal
        from kyb.episodes import termination_reason
        before_terminal=client.summary()['cli_invocations']
        result,transcript=review_terminal(session,client,json.loads((trace.root/'manifest.json').read_text()),
            official_success=bool(environment.end_flag[0] and environment.success[0]),
            termination=termination_reason(official_end=bool(environment.end_flag[0]),
                env_steps=int(environment.take_action_cnt[0]),env_limit=environment.step_lim,
                allowance_stop=bool(report.get('allowance_action_stop')),last_receipt=session.last_receipt))
        write_once(trace.root/'terminal_transcript.json',transcript)
        report['terminal_review']={'result':result,'robot_actions':0,
            'policy_cli_invocations':client.summary()['cli_invocations']-before_terminal}
except BaseException as error:
    import traceback
    report.update(status="NEEDS_WORK", error_type=type(error).__name__)
    report.setdefault('task_outcome','UNKNOWN')
    # Never echo an SDK exception or a whole request, which could include credentials.
    trace.event("run_failure", {"error_type": type(error).__name__, "frames": [
        {"file": frame.filename, "line": frame.lineno, "function": frame.name}
        for frame in traceback.extract_tb(error.__traceback__)]})
    if session is not None and session.last_receipt is not None:
        trace.event("terminal_pending", {"reason": "exception_no_automatic_extra_request",
            "receipt": session.last_receipt["id"],
            "observation": session.current_observation["id"] if session.current_observation else None,
            "post_observation_status": ("available" if session.current_observation and
                session.current_observation["data"].get("after_receipt") == session.last_receipt["id"] else "missing")})
    if isinstance(error,BudgetExhausted):report['allowance_exhausted']=True
    print(f"[KYB] failure: {type(error).__name__}; inspect local evidence", flush=True)
finally:
    if policy is not None:
        write_once(trace.root / "policy_transcript.json", policy.transcript())
    transport = client.summary() if client else {"transport": "codex_cli", "cli_invocations": 0}
    management_transport=manager_client.summary() if manager_client else {'transport':'codex_cli','cli_invocations':0}
    report['call_allowance']=budget.view()
    report['budget_role']='policy'
    if session is not None:
        from kyb.contracts import current_contract
        report['public_task_contract_id']=current_contract(session)
        report['operator_clarification_refs']=list(session.contract_amendments)
        report['operator_clarification_note']='Public requirement amendments only; preserve original evaluator. Assistance is explicitly identified, not silently counted as unassisted.'
    report['manager_call_allowance']=manager_budget.view()
    report['management_transport']=management_transport
    if session is not None and session.current_observation is not None:
        from kyb.episodes import close
        termination='operator_stop' if report.get('diagnostic_stop') or session.stop_requested() else 'run_exception:'+report.get('error_type','closed_before_terminal_review')
        try:
            close(session,json.loads((trace.root/'manifest.json').read_text()),transport,management_transport,
                  budget.view(),manager_budget.view(),termination=termination)
            report['episode_archive_status']='PASS'
        except Exception as archive_error:
            report['episode_archive_status']='NEEDS_WORK';report['status']='NEEDS_WORK'
            trace.event('episode_archive_failure',{'error_type':type(archive_error).__name__,
                'qualification':'Archive did not pass verification; no inference/action retried. Preserve original failure, ledgers and snapshot.'})
    report['total_cli_invocations']=transport['cli_invocations']+management_transport['cli_invocations']
    report.update(model_transport=transport, model_policy_test=transport['cli_invocations'] > 0, knowledge_revision=store.revision,
                  wall_seconds=time.monotonic()-started,
                  sources_unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha
                                        for p,sha in source_hashes.items()))
    if not report["sources_unchanged"]:
        report["status"] = "NEEDS_WORK"
        report["source_changed"] = True
    report['final_memory']=store.freeze(trace.root/'final_memory.sqlite')
    write_once(trace.root / "report.json", report)
    store.close()
    retrieval_index.close()
    if environment is not None:
        environment.close()
    app.close()
