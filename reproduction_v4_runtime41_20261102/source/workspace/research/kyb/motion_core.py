"""Motion kernel for a given command, measured state and planner response.

No task retrieval, operator dialogue or perception-profile dispatch lives here.
"""
import json
import math
from dataclasses import replace
import numpy as np
from XPolicyLab.policy.GPT_6_Astra_Direct_EEF.policy import EefAgentPolicy as BasePolicy
from XPolicyLab.policy.GPT_6_Astra_Direct_EEF.inspect.policy import MotionOutcome
from XPolicyLab.policy.GPT_6_Astra_Direct_EEF.inspect.types import Action, ActionChunk
from .trace import plain


class MotionCore(BasePolicy):
    def reset(self):
        self._last_executed_action=None
        super().reset()

    def _handle_motion(self,name,arguments,observation):
        if 'preview' in arguments and not isinstance(arguments['preview'],bool):
            return self._repair('preview must be a boolean')
        timing=('execute_max_s','max_duration_s','max_joint_travel_deg')
        for key in timing:
            value=arguments.get(key)
            if value is not None and (isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0):
                return self._repair(key+' must be positive and finite; no motion executed')
        hz=self.action_spec.control_hz
        if arguments.get('execute_max_s') is not None and arguments['execute_max_s']<1/hz:
            return self._repair('execute_max_s must cover at least one control tick')
        old_goals=dict(self._gripper_goals)
        old_targets=dict(self._last_targets)
        if name=='hold':
            seconds=arguments.get('seconds')
            if isinstance(seconds,bool) or not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or not 0<seconds<=5:
                return self._repair('hold seconds must be in (0,5]')
            count=max(1,math.floor(seconds*hz+1e-9))
            held=getattr(self,'_last_executed_action',None) or self._held_action_data(observation)
            actions=[Action(data={k:np.asarray(v).copy() for k,v in held.items()}) for _ in range(count)]
            if self._path_validator is not None:
                result=self._path_validator(actions,observation)
                if result['status']!='PASS':return self._repair('Hold commands did not pass the configured body path guard')
            outcome=MotionOutcome(chunk=ActionChunk(actions=actions,control_hz=hz,meta={'trace':{'tool':'hold'}}),tool_result='Hold selected')
        else:
            outcome=super()._handle_motion(name,arguments,observation)
        if outcome.chunk is None:return outcome
        chunk=outcome.chunk
        actions=list(chunk.actions)
        # Base planning latches the final jaw target. Planning is not execution.
        self._gripper_goals=old_goals
        travel={}
        for arm in ('left','right'):
            last=np.asarray(observation.state[arm+'_arm_joint_state'],dtype=float)
            total=np.zeros_like(last)
            for action in actions:
                current=np.asarray(action.data[arm+'_arm_joint_state'],dtype=float)
                total+=np.abs(current-last);last=current
            travel[arm]=np.rad2deg(total).tolist()
        stats={'full_planned_actions':len(actions),'full_duration_s':len(actions)/hz,
               'cumulative_joint_travel_deg':travel,'max_single_joint_travel_deg':max(v for values in travel.values() for v in values)}
        cost_failure=(arguments.get('max_duration_s') is not None and stats['full_duration_s']>arguments['max_duration_s']) or (
            arguments.get('max_joint_travel_deg') is not None and stats['max_single_joint_travel_deg']>arguments['max_joint_travel_deg'])
        if arguments.get('preview') or cost_failure:
            self._last_targets=old_targets
            return MotionOutcome(chunk=None,repairable=False,tool_result=json.dumps({'status':'cost_limit_rejected' if cost_failure else 'preview',
                **stats,'executed_actions':0,'qualification':'Plan only; no contact/tracking guarantee.'}))
        count=len(actions)
        if arguments.get('execute_max_s') is not None:count=min(count,max(1,math.floor(arguments['execute_max_s']*hz+1e-9)))
        if observation.remaining_steps is not None:count=min(count,observation.remaining_steps)
        if count<=0:return self._repair('No environment steps remain')
        if self.research is not None:
            source=self.research.evidence('validated_action_plan',{'source_observation':observation.extra.get('evidence_id'),
                'stats':stats,'actions':[plain(a.data) for a in actions],'qualification':'Validated full path, not executed actions'})
            stats['full_plan_ref']=source['id']
        chosen=actions[:count]
        # A prefix boundary is not arrival at the requested target.
        if count<len(actions):chosen[-1]=replace(chosen[-1],meta={k:v for k,v in chosen[-1].meta.items() if k!='chunk_final'})
        self._awaiting_confirmation=chosen
        meta={**chunk.meta,**stats,'selected_actions':count,
              'selected_stop_reason':'observation_boundary' if count<len(actions) else 'full_plan_selected'}
        return MotionOutcome(chunk=replace(chunk,actions=chosen,meta=meta),tool_result=json.dumps({
            'status':'accepted','selected_actions':count,'selected_duration_s':count/hz,**stats,
            'qualification':'A prefix is not target arrival; after observation choose a new plan.'}))

    def confirm_executed(self,played):
        pending=getattr(self,'_awaiting_confirmation',[])
        if not 0<=played<=len(pending):raise ValueError('Executed count outside selected prefix')
        if played:
            confirmed=dict(pending[played-1].data)
            self.resume_executed_gripper_goals(confirmed)
            self._last_executed_action={k:np.asarray(v).copy() for k,v in confirmed.items()}
        self._awaiting_confirmation=[]
