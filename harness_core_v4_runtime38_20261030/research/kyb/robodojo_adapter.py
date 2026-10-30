"""ARX-specific capture boundary. The memory manager never imports this adapter."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def canonical_state(state, fingers, body_version, *, pose_module=None):
    if pose_module is None:
        from XPolicyLab.policy.GPT_6_Astra_Direct_EEF import pose as pose_module
    contract = {'adapter': 'RoboDojo/ARX_X5_Direct_EEF', 'world_frame': 'RoboDojo/world',
        'pose_code_sha256': hashlib.sha256(Path(pose_module.__file__).read_bytes()).hexdigest(),
        'flange_to_action_translation_m': [float(pose_module.GRASP_POINT_OFFSET_M), 0., 0.],
        'orientation_reference_wxyz': list(pose_module.TOP_DOWN_QUAT_WXYZ),
        'units': {'position':'metre','quaternion':'wxyz','joint':'radian','action_angle':'degree'}}
    contract['id'] = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    arms = {}
    for key, raw in state.items():
        if not key.endswith('_ee_pose'):
            continue
        arm = key.removesuffix('_ee_pose')
        dimensions = {f'{arm}_{k}': float(v) for k, v in pose_module.pose_to_values(raw).items()}
        arms[arm] = {'part_id': f'arm:{arm}', 'part_version': body_version,
            'action_pose_world_m_wxyz': [*[dimensions[f'{arm}_{k}'] for k in 'xyz'], *[float(v) for v in raw[3:7]]],
            'action_dimensions': dimensions, 'joint_angles_rad': state.get(f'{arm}_arm_joint_state'),
            'finger_displacements_m': fingers.get(arm), 'qualification': 'host_measured_with_declared_adapter_contract'}
    return {'coordinate_contract':contract, 'arms':arms}
