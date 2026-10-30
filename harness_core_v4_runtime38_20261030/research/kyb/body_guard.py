"""Check nominal synchronized paths against the other arm's configured spheres."""
from __future__ import annotations

import numpy as np


def synchronized_samples(actions, current_state, step_rad=.001):
    paths={arm:np.asarray([current_state[f'{arm}_arm_joint_state']]+
        [a.data[f'{arm}_arm_joint_state'] for a in actions],dtype=np.float64) for arm in ('left','right')}
    if any(not np.isfinite(q).all() or q.shape[1:]!=(6,) for q in paths.values()):
        raise ValueError('Joint path must contain finite six-joint samples for both arms')
    dense={arm:[q[0]] for arm,q in paths.items()}; mapping=[{'command_index':-1,'alpha':0.}]
    for i in range(len(actions)):
        count=max(1,int(np.ceil(max(np.max(np.abs(q[i+1]-q[i])) for q in paths.values())/step_rad)))
        for j in range(1,count+1):
            alpha=j/count
            for arm,q in paths.items():dense[arm].append(q[i]+alpha*(q[i+1]-q[i]))
            mapping.append({'command_index':i,'alpha':alpha})
    return {arm:np.asarray(q) for arm,q in dense.items()},mapping


def closest_cross_arm_pair(left_spheres, right_spheres):
    """Raw radii only: self-collision padding/ignore never applies between arms."""
    if left_spheres.shape[0]!=right_spheres.shape[0]:
        raise ValueError('Both arms must use the same trajectory sample times')
    for spheres in (left_spheres,right_spheres):
        if not np.isfinite(spheres[...,3]).all() or not (spheres[...,3]>0).any():
            raise ValueError('Missing or nonfinite collision spheres')
        if not np.isfinite(spheres[...,:3][spheres[...,3]>0]).all():
            raise ValueError('Active collision sphere has nonfinite geometry')
    best=(float('inf'),None)
    for start in range(0,len(left_spheres),128):
        a,b=left_spheres[start:start+128],right_spheres[start:start+128]
        gap=np.linalg.norm(a[:,:,None,:3]-b[:,None,:,:3],axis=-1)-a[:,:,None,3]-b[:,None,:,3]
        active=(a[:,:,None,3]>0)&(b[:,None,:,3]>0)
        gap=np.where(active,gap,np.inf)
        flat=int(np.argmin(gap));index=np.unravel_index(flat,gap.shape)
        if float(gap[index])<best[0]:best=(float(gap[index]),(int(index[0]+start),int(index[1]),int(index[2])))
    return best


class BodyPathGuard:
    def __init__(self, manager, session):
        from curobo.kinematics import Kinematics
        self.session=session
        self.arms={}
        for robot in manager.robot_list:
            if robot.type!='target':continue
            arm=robot.arm_name.removesuffix('_arm')
            cp=manager.planner[robot.robot_name]
            # Independent FK buffers: do not resize the planner's captured CUDA buffers.
            kin=Kinematics(cp.motion_planner.kinematics.config)
            if list(kin.joint_names)!=cp.cspace_joint_names:
                raise ValueError('Guard/planner joint order mismatch')
            cfg=kin.config.kinematics_config
            inverse={int(v):k for k,v in cfg.link_name_to_idx_map.items()}
            names=[inverse[int(v)] for v in cfg.link_sphere_idx_map.cpu().reshape(-1)]
            self.arms[arm]=(robot,cp,kin,names)
        if set(self.arms)!={'left','right'}:
            raise ValueError('This guard requires the supported two X5 arms')

    def _world_spheres(self, arm, q):
        import torch
        import transforms3d as t3d
        from curobo.types import JointState
        robot,cp,kin,_=self.arms[arm]
        position=torch.as_tensor(np.asarray([cp._build_cspace_joint_values(v) for v in q]),
            device=cp.device_cfg.device,dtype=torch.float32).unsqueeze(0)
        state=kin.compute_kinematics(JointState.from_position(position,joint_names=kin.joint_names))
        spheres=state.get_link_spheres()[0].detach().cpu().numpy().copy()
        mount=np.asarray(robot.entity_origin_pose,dtype=float)
        rotation=t3d.quaternions.quat2mat(mount[3:])
        spheres[...,:3]=spheres[...,:3]@rotation.T+mount[:3]
        return spheres

    def __call__(self, actions, observation):
        dense,mapping=synchronized_samples(actions,observation.state)
        spheres={arm:self._world_spheres(arm,dense[arm]) for arm in ('left','right')}
        gap,index=closest_cross_arm_pair(spheres['left'],spheres['right'])
        h,i,j=index
        result={'status':'PASS' if gap>=0 else 'NEEDS_WORK','before_observation_id':observation.extra['evidence_id'],
            'dependencies':dict(self.session.dependencies),'executed_actions':0,
            'minimum_cross_arm_gap_m':gap,'minimum_at':mapping[h],
            'sphere_indices':{'left':i,'right':j},
            'closest_spheres_world_xyz_radius_m':{'left':spheres['left'][h,i].tolist(),
                                                 'right':spheres['right'][h,j].tolist()},
            'nominal_joint_sample_rad':{arm:dense[arm][h].tolist() for arm in ('left','right')},
            'links':{'left':self.arms['left'][3][i],'right':self.arms['right'][3][j]},
            'samples':len(mapping),'maximum_sample_joint_delta_rad':.001,
            'scope':'sampled nominal synchronized arms and configured rigid spheres; '
                    'not continuous safety, timing/tracking uncertainty, fingers, held tools or task objects',
            'interpretation':'computational represented-geometry check, not a measured physical contact or feasibility law'}
        reference=self.session.evidence('path_validation',result)
        return {**result,'evidence_id':reference['id']}
