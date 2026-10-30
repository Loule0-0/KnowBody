"""Cartesian action surface for the RGB-only Inspect agent."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from XPolicyLab.policy.GPT_6_Astra_Direct_EEF.inspect.policy import (
    InfrastructureFailure,
    JointAgentPolicy,
    MotionOutcome,
    RoboDojoActionSpec,
    _optional_float,
)
from XPolicyLab.policy.GPT_6_Astra_Direct_EEF.inspect.trace import named_state, source_revision
from XPolicyLab.policy.GPT_6_Astra_Direct_EEF.inspect.types import (
    JOINT_CHANNELS,
    Action,
    ActionChunk,
    ActionSpace,
    Observation,
)

from . import pose

Planner = Callable[..., Mapping[str, Any]]

# How far one env step may carry a motion comes from the action spec's
# per-dimension ``max_step``, the way the upstream Inspect agent takes it from
# the embodiment's declaration rather than deciding a speed for itself. A fixed
# count cannot work: resampling is uniform, so a 480-waypoint CuRobo plan
# replayed as 15 joint targets asks for a jump the controller cannot track, and
# the arm lands hundreds of millimetres away.
#
# Upstream's playout cap. Nothing in the recorded sweep comes close, so this
# only exists to refuse a motion rather than silently compress it.
_DEFAULT_MAX_DURATION_S = 10.0
# The jaw is not limited by tracking but by dwell: it is one position-controlled
# degree of freedom, so what matters is how long its setpoint is held, not how
# finely its travel is divided. The embodiment declares what the jaw *can* do,
# its full range in four env steps; this asks for half again as many, leaving a
# full close at six steps and a small correction at one. Nothing measured so
# far separates four steps from six on grasp success, so this keeps the count
# the recorded sweep ran with rather than speeding the jaw up on a refactor.
_DEFAULT_GRIPPER_DWELL_STEP = 0.18

_LEGACY_KEYS = ("xyz", "quat", "arm", "left", "right")


def _function(name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


class EefAgentPolicy(JointAgentPolicy):
    """Inspect policy that asks the model for world-frame grasp-point dimensions."""

    motion_tool_name = "move_eef"

    def __init__(self, *, planner: Planner, **kwargs: Any) -> None:
        self.research = kwargs.pop("research", None)
        if self.research is not None:
            kwargs['env']=dict(kwargs.get('env') or {})
            kwargs['env'].setdefault('L4_INSPECT_DEPTH','surface_queries')
        self._path_validator = kwargs.pop("path_validator", None)
        self._planner = planner
        # The tool description quotes the playout cap, so the limits have to be
        # known before the base constructor builds the tools.
        parameters = self.execution_parameters(kwargs['action_spec'], kwargs.get('env') or {})
        self._max_duration_s = parameters['max_duration_s']
        self._gripper_dwell_step = parameters['gripper_dwell_step']
        super().__init__(**kwargs)
        self._arm_step_limits = {arm: np.asarray(values) for arm, values in parameters['arm_step_limits'].items()}
        self._gripper_step_limits = parameters['gripper_step_limits']

    @classmethod
    def execution_parameters(cls, action_spec, env):
        """Resolve the exact parameters used by this policy, also saved in research provenance."""
        duration = _optional_float(env, 'L4_INSPECT_EEF_MAX_DURATION_S', _DEFAULT_MAX_DURATION_S)
        dwell = _optional_float(env, 'L4_INSPECT_EEF_GRIPPER_DWELL_STEP', _DEFAULT_GRIPPER_DWELL_STEP)
        for key, value in (('L4_INSPECT_EEF_MAX_DURATION_S', duration),
                           ('L4_INSPECT_EEF_GRIPPER_DWELL_STEP', dwell)):
            if not np.isfinite(value) or value <= 0.0:
                raise InfrastructureFailure(f'{key} must be finite and > 0')
        limits = cls._declared_step_limits(action_spec)
        located = ActionSpace(JOINT_CHANNELS).offsets()
        arm_limits, gripper_limits = {}, {}
        for arm in pose.ARMS:
            start, size = located[f'{arm}_arm_joint_state']
            arm_limits[arm] = limits[start:start+size].tolist()
            start, _ = located[f'{arm}_ee_joint_state']
            gripper_limits[arm] = min(float(limits[start]), dwell)
        return {'max_duration_s':duration, 'gripper_dwell_step':dwell,
                'arm_step_limits':arm_limits, 'gripper_step_limits':gripper_limits}

    @property
    def _max_steps(self) -> int:
        return max(1, int(np.ceil(self._max_duration_s * self.action_spec.control_hz)))

    def reset(self) -> None:
        self._last_targets: dict[str, np.ndarray] = {}
        self._gripper_goals: dict[str, float] = {}
        super().reset()

    def _validate_depth_mode(self) -> None:
        if self.research is None:
            return super()._validate_depth_mode()
        if self._env.get('L4_INSPECT_DEPTH')!='surface_queries':
            raise ValueError('Research EEF policy requires explicit read-only surface depth queries')

    def resume_executed_gripper_goals(self, last_action: dict) -> None:
        """A fresh development branch keeps actual replayed jaw goals, not observed gap."""
        goals = {}
        for arm in pose.ARMS:
            value = np.asarray(last_action[f'{arm}_ee_joint_state'],dtype=float)
            if value.shape != (1,) or not np.isfinite(value).all() or not 0 <= value[0] <= 1:
                raise ValueError('Replay requires the last executed normalized gripper command')
            goals[arm] = float(value[0])
        self._gripper_goals = goals

    # ------------------------------------------------------------------ #
    # Tool surface
    # ------------------------------------------------------------------ #

    def _build_tools(self, action_spec: RoboDojoActionSpec) -> list[dict[str, Any]]:
        terminal_tools = super()._build_tools(action_spec)[1:]
        move = _function(
            "move_eef",
            (
                "Move to absolute world-frame grasp-point targets. Name only the "
                "dimensions you intend to change; every unnamed dimension holds "
                "its observed value, so lowering one hand 2 cm is a single "
                "entry. Positions are metres. Orientation is three angles in "
                "degrees measured from a straight-down grasp, not absolute "
                "world angles: at "
                "pitch_deg=roll_deg=yaw_deg=0 the outward tool axis points at "
                "world -z. pitch_deg turns about world +x (0 is straight down, "
                "+90 is horizontal pointing at world +y), roll_deg about world "
                "+y, and yaw_deg about world +z, which is the jaw rotation of a "
                "straight-down grasp. Naming dimensions of both arms moves them "
                "simultaneously along time-aligned paths. The motion plays at a "
                "fixed safe speed, so how long it takes follows from how far it "
                f"goes; a move needing over {self._max_duration_s:g}s is refused "
                "and has to be split. A gripper value named alongside a pose "
                "moves the jaw after the planned arm-path phase. Physical "
                "arrival and successful grasp still need observation. "
                f"Per-dimension bounds: {pose.bounds_text()}."
            ),
            {
                "type": "object",
                "properties": {
                    "targets": {
                        "type": "object",
                        "additionalProperties": {"type": "number"},
                        "description": (
                            "Map of dimension name to value. Valid names: "
                            + ", ".join(pose.labels())
                        ),
                    },
                    "note": {
                        "type": "string",
                        "description": (
                            "What you observe right now and why you chose this "
                            "motion, in one or two plain sentences."
                        ),
                    },
                },
                "required": ["targets", "note"],
            },
        )
        if self.research is not None:
            move["function"]["parameters"]["properties"].update(self.research.tool_properties())
        move["function"]["strict"] = False
        if self.research is not None:
            from kyb.surface import tool_schema
            from kyb.knowledge import policy_memory_tools, function
            surface=tool_schema()
            memory_tools=policy_memory_tools()
            for tool in [surface,memory_tools[1]]:
                tool['function']['parameters']['properties']['attachment_updates']=self.research.tool_properties()['attachment_updates']
            body=function('read_body','Read current body/finger projections in one camera. No motion or model call.',
                {'observation_id':{'type':'string'},'camera':{'type':'string'}},['observation_id','camera'])
            return [move,surface,*memory_tools,body,*terminal_tools]
        return [move, *terminal_tools]

    def _system_message(self) -> str:
        if self.research is not None:
            return (
                'You control a robot by choosing one supplied tool per turn. You alone choose every EEF target. '
                'Use current raw RGB, measured body state, task instruction and conditional task materials. '
                'The host plans and executes motions; a rejected plan executes nothing. After execution, '
                'inspect actual arrival and scene evidence before treating an intended effect as observed. '
                'Aim for task success with few total decisions, reads and management calls. '
                'Choose feedback when its possible outcomes can change the next action; a fixed small step '
                'is not required merely because a physical property is unknown. '
                + self.research.instructions() + '\n\nBody/action facts:\n' + self.action_spec.docs.strip())
        perception=('You may query visible pixels with inspect_surface to obtain sourced RGB-D surface coordinates. '
            'The query advances no simulation and generates no motion; you still choose every move_eef target. '
            if self.research is not None else
            'Cartesian targets must be estimated from RGB; no depth or world-coordinate query is available. ')
        return (
            "You are controlling a simulated robot embodiment named "
            "'robodojo-arx-x5'. You receive RGB camera images, the current "
            "world-frame grasp-point state of both arms in the same 14 dimensions "
            "move_eef takes, arm joint angles as context you cannot command, "
            "and a task instruction. Move with move_eef by "
            "naming only the world-frame dimensions you want to change. "
            + perception + "Respond with exactly one tool "
            "call per turn. After each motion the next observation reports how "
            "far the grasp point ended up from what you asked for, so check it "
            "before assuming a motion landed. Model calls are accounted and "
            "reviewed, with no mechanical call-count termination. The "
            "environment's action-step limit is reported with each observation "
            "as the env steps remaining. A "
            "motion spends env steps in proportion to how far it travels. "
            "Model decisions are a separate expensive resource: aim for high task success "
            "with few decisions. When intermediate feedback would not change the action, "
            "make meaningful progress in one justified motion rather than repeated tiny moves. "
            "When feedback is needed, say what the next observation can distinguish and "
            "how its outcomes change the next choice. Unknown tool rigidity does not alone "
            "require repeated fixed-size probes; consider which uncertainty matters for the "
            "specific next motion, its clearance, and retention."
            f"\n\nEmbodiment notes:\n{self.action_spec.docs.strip()}"
            + ("\n\n" + self.research.instructions() if self.research is not None else "")
        )

    # ------------------------------------------------------------------ #
    # Observation
    # ------------------------------------------------------------------ #

    def _state_block(self, observation: Observation) -> str:
        """Render the 14 Cartesian dimensions, then joints as read-only context.

        The Cartesian vector leads and is emitted in ``pose.labels()`` order, so
        it is dimension-for-dimension the space ``move_eef`` targets are
        absolute in. That alignment is the whole point: an absolute target only
        says where to go, so the model needs its starting point named the same
        way to reason about a correction at all.

        Joint angles follow as context the model cannot command. Their two
        gripper dimensions are the same quantity as ``<arm>_gripper`` above, so
        they are left out rather than repeated under a second name.
        """
        measured = {arm: self._measured_values(observation, arm) for arm in pose.ARMS}
        lines = [
            f"Instruction: {observation.instruction or ''}",
            "World-frame grasp-point state, named as move_eef takes them "
            "(metres; degrees from the straight-down reference):",
        ]
        for label in pose.labels():
            arm, _, axis = label.partition("_")
            if axis == "gripper":
                lines.append(f"{label}={self._measured_gripper(observation, arm):.2f} (previous normalized command; not measured jaw opening)")
            elif measured[arm] is None:
                # Never silently drop a dimension: a missing line reads as
                # "nothing to say about it" when it means the reference the
                # model is about to aim from is not there.
                lines.append(f"{label}=unavailable")
            elif axis in pose.ANGLE_AXES:
                lines.append(f"{label}={measured[arm][axis]:.1f}")
            else:
                lines.append(f"{label}={measured[arm][axis]:.4f}")
        joints = named_state(self.action_spec.labels, observation.state)
        for arm in pose.ARMS:
            joints.pop(f"{arm}_gripper", None)
        lines.append("Arm joint angles, radians, for context; not commandable:")
        lines.extend(f"{label}={value:.4f}" for label, value in joints.items())
        return "\n".join(lines)

    def _observation_message(self, observation: Observation) -> dict[str, Any]:
        message = super()._observation_message(observation)
        arrival = self._arrival_text(observation)
        if arrival:
            message["content"][0]["text"] += "\n" + arrival
        return message

    def _arrival_text(self, observation: Observation) -> str:
        """Report how far the previous motion ended up from what was requested."""
        pending, self._last_targets = self._last_targets, {}
        parts = []
        for arm, target in sorted(pending.items()):
            measured = self._measured_pose(observation, arm)
            if measured is None:
                continue
            # Both sides are stored as flange poses because that is what the
            # planner takes, but the model reasons in grasp points, and the
            # two distances differ whenever the orientation missed as well.
            reached = pose.pose_to_values(measured)
            wanted = pose.pose_to_values(target)
            offset = float(
                np.linalg.norm(
                    [reached[axis] - wanted[axis] for axis in pose.POSITION_AXES]
                )
            )
            angle = pose.orientation_error_deg(measured[3:7], target[3:7])
            parts.append(f"{arm} {offset * 1000:.0f} mm and {angle:.1f} deg away")
        if not parts:
            return ""
        return (
            "Arrival check for the previous move_eef target: "
            + "; ".join(parts)
            + "."
        )

    @staticmethod
    def _measured_pose(observation: Observation, arm: str) -> np.ndarray | None:
        raw = np.asarray(
            observation.state.get(f"{arm}_ee_pose", []), dtype=np.float64
        ).reshape(-1)
        if raw.size < 7 or not np.isfinite(raw[:7]).all():
            return None
        if float(np.linalg.norm(raw[3:7])) <= 1e-8:
            return None
        return raw[:7]

    def _measured_values(
        self, observation: Observation, arm: str
    ) -> dict[str, float] | None:
        measured = self._measured_pose(observation, arm)
        if measured is None:
            return None
        return pose.pose_to_values(measured)

    @staticmethod
    def _measured_gripper(observation: Observation, arm: str) -> float:
        raw = np.asarray(
            observation.state.get(f"{arm}_ee_joint_state", [0.0]), dtype=np.float64
        ).reshape(-1)
        return float(raw[0]) if raw.size else 0.0

    # ------------------------------------------------------------------ #
    # Motion
    # ------------------------------------------------------------------ #

    def _is_motion_tool(self,name: str) -> bool:
        return name == self.motion_tool_name or (self.research is not None and name in {
            'inspect_surface','query_experience','manage_memory','read_body'})

    def _handle_motion(
        self,
        name: str,
        arguments: Mapping[str, Any],
        observation: Observation,
    ) -> MotionOutcome:
        if self.research is not None and any(key in arguments for key in ('memory_patch','memory_query')):
            return self._repair('Database patches and next-turn memory_query are not action arguments. '
                'Use query_experience for a current historical question or manage_memory for extraction/reorganization. No motion executed.')
        if self.research is not None:
            self.research.consume(arguments, observation)
        if name=='inspect_surface':
            return MotionOutcome(chunk=None,repairable=False,
                tool_result=self.research.inspect_surface(arguments,observation))
        if name in {'query_experience','manage_memory','read_body'}:
            result=getattr(self.research,name)(arguments)
            return MotionOutcome(chunk=None,repairable=False,tool_result=result)
        legacy = [key for key in _LEGACY_KEYS if key in arguments]
        if legacy:
            return self._repair(
                f"move_eef no longer takes {legacy[0]!r}. Name world-frame "
                'dimensions instead, for example {"targets": {"left_z": 0.90, '
                '"left_yaw_deg": -90}}.'
            )
        by_arm, error = self._parse_targets(arguments.get("targets"))
        if error:
            return self._repair(error)
        assert by_arm is not None

        paths: dict[str, np.ndarray] = {}
        target_poses: dict[str, np.ndarray] = {}
        for arm in sorted(by_arm):
            axes = {
                axis: value
                for axis, value in by_arm[arm].items()
                if axis != "gripper"
            }
            if not axes:
                continue
            measured = self._measured_values(observation, arm)
            if measured is None:
                raise InfrastructureFailure(
                    f"RoboDojo reported no usable {arm}_ee_pose; an absolute "
                    "Cartesian target cannot be built from a missing flange pose"
                )
            target_pose = pose.values_to_pose({**measured, **axes})
            path, failure = self._plan_arm(arm, target_pose)
            if failure is not None:
                return MotionOutcome(
                    chunk=None,
                    tool_result=f"{arm} arm {failure.tool_result} Neither arm moved.",
                    repairable=failure.repairable,
                )
            assert path is not None
            paths[arm] = path
            target_poses[arm] = target_pose

        grippers = {
            arm: (self._measured_gripper(observation, arm), overrides["gripper"])
            for arm, overrides in by_arm.items()
            if overrides.get("gripper") is not None
        }
        # The jaw moves only once the arm has arrived. Closing it while the
        # hand is still travelling sweeps the object out of the way before the
        # grasp, which is the most common complaint in the recorded give_up
        # reasons. Sequencing costs the jaw's own waypoints on top of the
        # arm's, which is why the derivation adds the two phases.
        paths = {arm: np.vstack((np.asarray(observation.state[f'{arm}_arm_joint_state'], dtype=float), path))
                 for arm, path in paths.items()}
        arm_steps = self._arm_steps(paths)
        gripper_steps = self._gripper_steps(grippers)
        steps = arm_steps + gripper_steps
        if steps > self._max_steps:
            return self._repair(
                f"this motion needs {steps / self.action_spec.control_hz:.1f}s at "
                f"the safe speed, over the {self._max_duration_s:g}s playout cap; "
                "split it into smaller motions"
            )
        if steps == 0:
            # Every named dimension already holds the value it was asked for.
            arm_steps = steps = 1
        # The count is never clamped to the planner's own point count.
        # Resampling interpolates along the planned polyline, so asking for
        # more waypoints than CuRobo returned adds points on that same
        # collision-checked path; clamping instead would be the compression
        # this derivation exists to avoid, only in miniature.
        synchronized = {
            arm: self._resample_path(path, max(1, arm_steps), self._arm_step_limits[arm])
            for arm, path in paths.items()
        }
        actions = []
        for index in range(steps):
            data = self._held_action_data(observation)
            for arm in by_arm:
                if arm in synchronized:
                    # Both arms travel over the same waypoints and arrive
                    # together, then hold while the jaws move.
                    data[f"{arm}_arm_joint_state"] = synchronized[arm][
                        min(index, len(synchronized[arm]) - 1)
                    ]
                if arm in grippers and index >= arm_steps:
                    data[f"{arm}_ee_joint_state"][:] = grippers[arm][1]
            actions.append(
                Action(
                    data=data,
                    meta={"chunk_final": True} if index == steps - 1 else {},
                )
            )

        if self._path_validator is not None:
            validation = self._path_validator(actions, observation)
            if validation['status'] != 'PASS':
                return MotionOutcome(chunk=None, repairable=False, tool_result=(
                    'The planned joint path overlaps the other arm in the configured body geometry. '
                    f"Closest links: {validation['links']}; gap {validation['minimum_cross_arm_gap_m']:.6f} m. "
                    'Neither arm moved. Choose another approach, orientation, or intermediate pose. '
                    f"Path evidence: {validation['evidence_id']}. This does not prove the goal physically impossible."))

        self._last_targets = dict(target_poses)
        # Latched after the waypoints are built, so the arm travels to the
        # release pose still holding the previous grip and only lets go on
        # arrival.
        for arm, (_, target) in grippers.items():
            self._gripper_goals[arm] = float(target)
        trace = {
            "tool": "move_eef",
            "mode": "bimanual" if len(by_arm) == 2 else "single_arm",
            "arms": {
                arm: {
                    "named": {
                        f"{arm}_{axis}": value
                        for axis, value in sorted(by_arm[arm].items())
                    },
                    "target_pose": (
                        target_poses[arm].round(6).tolist()
                        if arm in target_poses
                        else None
                    ),
                    "source_waypoints": len(paths[arm]) if arm in paths else 0,
                }
                for arm in sorted(by_arm)
            },
            "plan_status": "Success",
            "planned_waypoints": steps,
            "arm_waypoints": arm_steps,
            "gripper_waypoints": gripper_steps,
            "normalized_from_parallel_calls": bool(
                arguments.get("_normalized_from_parallel_calls")
            ),
        }
        return MotionOutcome(
            chunk=ActionChunk(
                actions=actions,
                control_hz=self.action_spec.control_hz,
                meta={"trace": trace},
            ),
            tool_result=self._accepted_text(steps, gripper_steps, target_poses),
        )

    def _accepted_text(
        self,
        steps: int,
        gripper_steps: int,
        target_poses: Mapping[str, np.ndarray],
    ) -> str:
        text = f"Accepted: playing {steps} waypoints"
        text += f" ({steps / self.action_spec.control_hz:.1f}s)."
        if gripper_steps:
            text += (
                f" The last {gripper_steps} move the gripper, once the arm has "
                "arrived."
            )
        if target_poses:
            text += " The next observation reports how close the grasp point landed."
        return text

    def _parse_targets(
        self, raw: Any
    ) -> tuple[dict[str, dict[str, float]] | None, str | None]:
        if not isinstance(raw, Mapping) or not raw:
            return None, (
                "move_eef.targets must be a non-empty object of dimension "
                "name to value"
            )
        valid = pose.labels()
        by_arm: dict[str, dict[str, float]] = {}
        for raw_label, raw_value in raw.items():
            label = str(raw_label)
            if label not in valid:
                return None, (
                    f"unknown dimension {label!r}; valid names: {', '.join(valid)}"
                )
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                return None, (
                    f"value for {label!r} must be a finite number, got {raw_value!r}"
                )
            value = float(raw_value)
            if not np.isfinite(value):
                return None, (
                    f"value for {label!r} must be a finite number, got {raw_value!r}"
                )
            low, high = pose.bounds(label)
            if not low <= value <= high:
                return None, (
                    f"target for {label!r} is outside [{low:.4g}, {high:.4g}]"
                )
            arm, _, axis = label.partition("_")
            by_arm.setdefault(arm, {})[axis] = value
        return by_arm, None

    def _plan_arm(
        self, arm: str, target_pose: np.ndarray
    ) -> tuple[np.ndarray | None, MotionOutcome | None]:
        planned = self._planner(arm=arm, target_pose=target_pose)
        planning_ref = (self.research.record_plan(arm, target_pose, planned,
            self.calls) if self.research is not None else None)
        if planned.get("status") != "Success" or planned.get("position") is None:
            status = str(planned.get("status", "Fail"))
            return None, MotionOutcome(
                chunk=None,
                tool_result=(
                    f"planner did not find a usable path (status: {status}). The robot did "
                    "not move; this does not establish physical impossibility. Choose another pose."
                    + (f" Planning evidence: {planning_ref['id']}." if planning_ref else "")
                ),
                repairable=False,
            )
        path = np.asarray(planned["position"], dtype=np.float32)
        if path.ndim != 2 or path.shape[1] != 6 or not np.isfinite(path).all():
            raise InfrastructureFailure(
                f"EEF planner returned path shape {path.shape}; expected finite (N, 6)"
            )
        if len(path) == 0:
            raise InfrastructureFailure("EEF planner returned an empty joint path")
        return path, None

    @staticmethod
    def _declared_step_limits(action_spec) -> np.ndarray:
        """The per-step budget for each of the 14 joint-space dimensions.

        The embodiment's declaration governs. A dimension that declares nothing
        is one whose whole range already fits inside a step, so its range is the
        only bound left to apply.
        """
        labels = action_spec.labels
        declared = action_spec.max_step or (None,) * len(labels)
        span = action_spec.high - action_spec.low
        limits = np.asarray(
            [
                span[index] if entry is None else entry
                for index, entry in enumerate(declared)
            ],
            dtype=np.float64,
        )
        if not np.isfinite(limits).all() or bool(np.any(limits <= 0.0)):
            offenders = [
                label for label, limit in zip(labels, limits, strict=True) if limit <= 0.0
            ]
            raise InfrastructureFailure(
                "every action dimension needs a positive per-step budget to "
                f"derive a waypoint count from; {offenders} have none"
            )
        return limits

    def _arm_steps(self, paths: Mapping[str, np.ndarray]) -> int:
        """Weighted polyline arc length bounds each joint in every interval."""
        steps = 0
        for arm, path in paths.items():
            coordinates = self._path_step_coordinates(path, self._arm_step_limits[arm])
            steps = max(steps, int(np.ceil(coordinates[-1])))
        return steps

    def _gripper_steps(self, grippers: Mapping[str, tuple[float, float]]) -> int:
        """Env steps the jaw's setpoint is held for, scaled to its travel."""
        steps = 0
        for arm, (start, target) in grippers.items():
            travel = abs(target - start)
            steps = max(
                steps, int(np.ceil(travel / self._gripper_step_limits[arm]))
            )
        return steps

    @staticmethod
    def _path_step_coordinates(path: np.ndarray, limits: np.ndarray) -> np.ndarray:
        path, limits = np.asarray(path, dtype=float), np.asarray(limits, dtype=float)
        if (path.ndim != 2 or path.shape[1] != 6 or not len(path) or limits.shape != (6,)
            or not np.isfinite(path).all() or not np.isfinite(limits).all() or np.any(limits <= 0)):
            raise InfrastructureFailure('Malformed joint path or per-joint step limits')
        lengths = np.max(np.abs(np.diff(path, axis=0))/limits, axis=1)
        return np.r_[0., np.cumsum(lengths)]

    @staticmethod
    def _resample_path(path: np.ndarray, steps: int, limits: np.ndarray) -> np.ndarray:
        """Sample equal weighted-arc intervals, excluding the measured start.

        Each interval has L-infinity joint-budget length <=1. By the triangle
        inequality, every joint target difference is bounded by its own limit
        (up to float32 representation). Uniform path indices do not give this bound.
        """
        path = np.asarray(path, dtype=float)
        source = EefAgentPolicy._path_step_coordinates(path, limits)
        if steps < 1 or steps < int(np.ceil(source[-1])):
            raise InfrastructureFailure('Waypoint count cannot satisfy the declared joint step limits')
        if source[-1] == 0:
            return np.repeat(path[-1:], steps, axis=0).astype(np.float32)
        keep = np.r_[True, np.diff(source) > 0]
        target = np.linspace(0.0, source[-1], steps + 1)[1:]
        return np.stack(
            [np.interp(target, source[keep], path[keep, joint]) for joint in range(6)],
            axis=1,
        ).astype(np.float32)

    def _normalize_tool_calls(self, tool_calls):
        # Multiple calls cannot be folded without losing experience/attachment updates.
        # The base policy enforces the single-tool contract.
        return tool_calls

    @staticmethod
    def _repair(message: str) -> MotionOutcome:
        return MotionOutcome(chunk=None, tool_result=message, repairable=True)

    def _held_action_data(self, observation: Observation) -> dict[str, np.ndarray]:
        """Hold the arms where they are and the jaws on what they were told.

        The jaw cannot hold by echoing its own reading. RoboDojo reports
        ``<arm>_ee_joint_state`` from the previous command after its
        20%-of-range clamp, not from the finger position, so re-sending it
        parks the jaw at the width it already has: the position error, and with
        it the grip force, goes to zero and the object slides out of a closed
        hand during the next transfer. Re-sending the value the model last
        asked for keeps the clamp a full 20% ahead of the fingers, which is the
        same squeeze the closing phase applied. Until the model has named a
        jaw, the reading is all there is.
        """
        data = {
            name: np.asarray(observation.state[name], dtype=np.float32).copy()
            for name in (
                "left_arm_joint_state",
                "left_ee_joint_state",
                "right_arm_joint_state",
                "right_ee_joint_state",
            )
        }
        for arm, goal in self._gripper_goals.items():
            data[f"{arm}_ee_joint_state"][:] = goal
        return data

    def audit_config(self) -> dict[str, Any]:
        config = super().audit_config()
        config["adapter"] = "robodojo-agent-l4-inspect-eef"
        # The base adapter hashes only itself. This surface is the other half
        # of what drives the run, so a trace from it has to cover both.
        config["code"] = source_revision(Path(__file__).resolve().parent)
        labels = pose.labels()
        config["embodiment"] = {
            **config["embodiment"],
            "labels": list(labels),
            "low": [pose.bounds(label)[0] for label in labels],
            "high": [pose.bounds(label)[1] for label in labels],
            # A Cartesian dimension has no honest per-step budget: how far one
            # env step carries it depends on the arm's pose. The budget that
            # governs is declared in joint space and reported below.
            "max_step": None,
            "rotation_reference_quat_wxyz": list(pose.TOP_DOWN_QUAT_WXYZ),
            # Per dimension, where the reference the absolute targets are
            # measured against was read from. Aligned with "labels".
            "state_reference_keys": list(pose.state_reference_keys()),
            "joint_labels": list(self.action_spec.labels),
            # The per-step budget every waypoint count is derived from, so a
            # trace shows what actually governed the run. Aligned with
            # "joint_labels".
            "joint_max_step": [float(value) for value in self._declared_step_limits(self.action_spec)],
        }
        config["policy_config"]["model_action_space"] = "eef"
        config["policy_config"]["max_duration_s"] = self._max_duration_s
        config["policy_config"]["gripper_dwell_step"] = self._gripper_dwell_step
        return config
