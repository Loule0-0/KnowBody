"""Perception, memory and operator dispatch; motion arguments pass through unchanged."""
import json
from dataclasses import replace
from XPolicyLab.policy.GPT_6_Astra_Direct_EEF.inspect.policy import MotionOutcome
from .knowledge import function
from .motion_core import MotionCore


class EefAgentPolicy(MotionCore):
    def _validate_depth_mode(self):
        if self.research is not None and not self.research.perception.depth:
            from XPolicyLab.policy.GPT_6_Astra_Direct_EEF.inspect.policy import validate_rgb_only_depth
            return validate_rgb_only_depth(self._env)
        return super()._validate_depth_mode()

    def _state_block(self, observation):
        """Keep measured state and acknowledged commands distinct in every input."""
        from XPolicyLab.policy.GPT_6_Astra_Direct_EEF import pose
        from XPolicyLab.policy.GPT_6_Astra_Direct_EEF.inspect.trace import named_state
        measured={arm:self._measured_values(observation,arm) for arm in pose.ARMS}
        lines=[f"Instruction: {observation.instruction or ''}",
            "World-frame grasp-point poses (metres; degrees from the straight-down reference). "
            "Gripper readings are normalized robot telemetry, separate from confirmed commands; neither proves holding:"]
        import math
        def format_reading(value):
            try:raw=float(value[0])
            except (TypeError,ValueError,IndexError):return 'unavailable'
            return f'{raw:.3f}' if math.isfinite(raw) else 'unavailable'
        confirmed=self._last_executed_action
        for label in pose.labels():
            arm,_,axis=label.partition('_')
            if axis=='gripper':
                raw=observation.state.get(f'{arm}_ee_joint_state')
                reading=format_reading(raw)
                value=confirmed.get(f'{arm}_ee_joint_state') if confirmed is not None else None
                command=format_reading(value)
                lines.append(f'{label}={reading} (normalized state reading); last_confirmed_command={command}')
            elif measured[arm] is None:lines.append(f'{label}=unavailable')
            elif axis in pose.ANGLE_AXES:lines.append(f'{label}={measured[arm][axis]:.1f}')
            else:lines.append(f'{label}={measured[arm][axis]:.4f}')
        joints=named_state(self.action_spec.labels,observation.state)
        for arm in pose.ARMS:joints.pop(f'{arm}_gripper',None)
        lines.append('Arm joint angles, radians, for context; not commandable:')
        lines.extend(f'{label}={value:.4f}' for label,value in joints.items())
        return '\n'.join(lines)

    def _build_tools(self, action_spec):
        tools=super()._build_tools(action_spec)
        if self.research is not None:
            tools=[t for t in tools if t['function']['name'] not in self.research.perception.forbidden_tools]
        for tool in tools:
            if tool['function']['name']=='inspect_surface':
                tool['function']['description'] += (
                    ' When inferring a particular face orientation from several samples, first check in RGB that they lie on that same functional face; '
                    'points on different faces or a small fit residual do not establish its normal.')
        move=next(t['function'] for t in tools if t['function']['name']=='move_eef')
        original=('pitch_deg turns about world +x (0 is straight down, '
            '+90 is horizontal pointing at world +y), roll_deg about world '
            '+y, and yaw_deg about world +z, which is the jaw rotation of a '
            'straight-down grasp.')
        if move['description'].count(original)!=1:
            raise ValueError('Inherited move_eef angle description changed; verify its composition before use')
        move['description']=move['description'].replace(original,
            'These are absolute Euler parameters composed as '
            'Rz(yaw_deg) Ry(roll_deg) Rx(pitch_deg) Rref, applied right to left; '
            'Rref is the fixed straight-down reference. They are not independent '
            'world-axis increments. With the other parameters fixed, changing '
            'roll turns about Rz(yaw) applied to world +y; changing pitch turns '
            'about Rz(yaw) Ry(roll) applied to world +x; changing yaw turns about '
            'world +z. At roll=yaw=0, pitch=+90 points the tool at world +y.')
        move['parameters']['properties'].update({
            'preview':{'type':'boolean','description':'Return validated plan cost without executing it. No future command is latched.'},
            'execute_max_s':{'type':'number','exclusiveMinimum':0,'description':'Execute only this initial duration, observe, then replan from measured state. No automatic continuation.'},
            'max_duration_s':{'type':'number','exclusiveMinimum':0,'description':'Reject the full plan if longer; no actions are executed.'},
            'max_joint_travel_deg':{'type':'number','exclusiveMinimum':0,'description':'Reject if any single joint cumulative path travel exceeds this. End-effector angle alone can hide a long joint path.'}})
        tools.append(function('hold','Advance simulation by continuing the last confirmed joint/jaw commands, then observe. '
            'This does not guarantee physical stillness or retention; use a short duration at uncertain contact.',
            {'seconds':{'type':'number','exclusiveMinimum':0,'maximum':5},'note':{'type':'string'}},['seconds']))
        from .clarification import tool
        tools.append(tool())
        return tools

    def _is_motion_tool(self,name):
        return name in ('hold','query_body_history','request_task_clarification') or super()._is_motion_tool(name)

    def _handle_motion(self,name,arguments,observation):
        if self.research is not None and name in self.research.perception.forbidden_tools:
            return MotionOutcome(chunk=None,repairable=False,tool_result='NEEDS_WORK: tool unavailable under the frozen perception profile; no action executed.')
        if name=='request_task_clarification':
            from .clarification import request
            try:result=request(self.research,arguments)
            except ValueError as error:result=json.dumps({'status':'NEEDS_WORK','reason':str(error),'executed_actions':0})
            return MotionOutcome(chunk=None,repairable=False,tool_result=result)
        if name=='query_body_history':
            return MotionOutcome(chunk=None,repairable=False,tool_result=self.research.query_body_history(arguments))
        planning_start=self.research.trace.sequence if self.research is not None else None
        outcome=super()._handle_motion(name,arguments,observation)
        if self.research is None or name not in ('move_eef','hold') or outcome.chunk is not None or outcome.repairable:
            return outcome
        if not any(arguments.get(k) for k in ('preview','max_duration_s','max_joint_travel_deg')):
            return outcome
        # Native planner/path refusals are plain text; preserve those outcomes.
        if not outcome.tool_result.startswith('{'):
            return outcome
        feedback=json.loads(outcome.tool_result)
        if feedback.get('status') not in ('preview','cost_limit_rejected'):
            return outcome
        before=observation.extra.get('evidence_id')
        if self.research.current_observation is None or before!=self.research.current_observation['id']:
            raise ValueError('Planning feedback requires the actual source observation')
        rows=self.research.store.db.execute(
            "SELECT id,kind,payload FROM evidence WHERE run=? AND sequence>? "
            "AND kind IN ('planning_result','path_validation') ORDER BY sequence",
            (self.research.episode,planning_start)).fetchall()
        if any(json.loads(row['payload'])['before_observation_id']!=before for row in rows):
            raise ValueError('Planning feedback sources must belong to this invocation and observation')
        from .trace import plain
        source=self.research.evidence('planning_feedback',{
            **feedback,'physics_steps':0,'before_observation_id':before,
            'dependencies':dict(self.research.dependencies),'policy_call_index':self.calls,
            'tool':name,'arguments':plain(arguments),
            'planning_refs':[row['id'] for row in rows if row['kind']=='planning_result'],
            'path_validation_refs':[row['id'] for row in rows if row['kind']=='path_validation'],
            'interpretation':'Computed from proposed joint commands at the source observation. '
                'No action was executed. Cost rejection is not planner failure, collision, '
                'physical infeasibility or a measured physical response.'})
        return replace(outcome,tool_result=json.dumps({**feedback,'evidence_id':source['id']}))
