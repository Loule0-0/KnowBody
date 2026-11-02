"""Execute an existing action chunk and close its receipt even on partial failure."""
from __future__ import annotations

from typing import Any, Callable

from .trace import plain


def execute_chunk(environment: Any, chunk: Any, *, trace: Any, session: Any,
                  capture: Callable, observation: Any, call_id: str | None) -> tuple[int, bool]:
    intent = trace.event("action_intent", {
        "call_id": call_id, "policy_step": observation.step,
        "plan": plain(chunk.meta), "actions": [plain(action.data) for action in chunk.actions]})
    first_step = int(environment.take_action_cnt[0])
    physics_start = int(environment.sim.sim.current_time_step_index)
    execution_dependencies = dict(session.dependencies)
    if chunk.meta.get('mode') == 'complete_scene_action_prefix_replay_no_new_plan_or_model':
        # Exact old commands bypass the new resampler; do not relabel their dynamics.
        execution_dependencies.pop('execution', None)
    played, stop = 0, False
    stop_reason="selected_chunk_complete"
    command_refs = []
    in_flight = None
    failure = None
    last_frame = None
    try:
        for index, action in enumerate(chunk.actions):
            if action.meta.get("request_stop"):
                stop = True
                stop_reason="policy_stop"
                break
            before = int(environment.sim.sim.current_time_step_index)
            command = trace.event("command", {"intent": intent["id"], "index": index,
                                             "action": plain(action.data)})
            command_refs.append(command["id"])
            in_flight = command["id"]
            count_before = int(environment.take_action_cnt[0])
            environment.take_action(dict(action.data))
            if int(environment.take_action_cnt[0]) != count_before + 1:
                raise RuntimeError("Command returned without an accepted environment action")
            played += 1
            in_flight = None
            trace.event("command_receipt", {"command": command["id"],
                "physics_start": before,
                "physics_end": int(environment.sim.sim.current_time_step_index),
                "env_step": int(environment.take_action_cnt[0])})
            frame = capture(observation.step, expose=False)
            last_frame = frame.extra["evidence_id"]
            if environment.is_episode_end():
                stop_reason="native_episode_end"
                break
    except BaseException as error:
        failure = type(error).__name__
        stop_reason="execution_exception"
        raise
    finally:
        data = {
            "intent": intent["id"], "call_id": call_id, "before_observation": observation.extra["evidence_id"],
            "dependencies": execution_dependencies, "executed_actions": played,
            "accepted_actions": int(environment.take_action_cnt[0]) - first_step,
            "planned_actions": len(chunk.actions),
            "full_planned_actions":chunk.meta.get("full_planned_actions",len(chunk.actions)),
            "selected_actions":len(chunk.actions),
            "stop_reason":stop_reason if stop_reason!="selected_chunk_complete" else chunk.meta.get("selected_stop_reason",stop_reason), "command_refs": command_refs,
            "env_start": first_step, "env_end": int(environment.take_action_cnt[0]),
            "physics_start": physics_start,
            "physics_end": int(environment.sim.sim.current_time_step_index),
            "unknown_command": in_flight, "policy_stop": stop,
            "execution": ("outcome_unknown" if in_flight else "partial" if failure or
                          (played < chunk.meta.get("full_planned_actions",len(chunk.actions)) and not stop) else "complete"),
            "failure_type": failure, "last_recorded_execution_frame": last_frame,
            "plan": plain(chunk.meta),
        }
        receipt = session.evidence("receipt", data)
        if played:
            session.invalidate_opened_grips(dict(chunk.actions[played - 1].data), receipt["id"])
        if failure:
            trace.event("terminal_pending", {"receipt": receipt["id"],
                "post_observation_status": "missing", "last_recorded_execution_frame": last_frame,
                "unknown_command": in_flight, "reason": "partial_execution_exception"})
    return played, stop
