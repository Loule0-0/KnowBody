"""Read-only RGB-D surface queries on an immutable observation, without object GT."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .body_visual import flange_transform
from XPolicyLab.policy.GPT_6_Astra_Direct_EEF.pose import GRASP_POINT_OFFSET_M


def pad_relation(local_point, windows):
    """Signed distances to actual pad supporting edges/planes; no contact inference."""
    result = {}
    for finger, window in windows.items():
        polygon = np.asarray(window['polygon_xz_m'], dtype=float)
        edges = np.roll(polygon, -1, axis=0)-polygon
        offset = np.asarray(local_point)[[0, 2]]-polygon
        margins = (edges[:, 0]*offset[:, 1]-edges[:, 1]*offset[:, 0])/np.linalg.norm(edges, axis=1)
        result[finger] = {'projected_edge_margins_m': margins.tolist(),
            'minimum_projected_edge_margin_m': float(margins.min()),
            'inward_distance_from_face_plane_m': float(window['inward_y_sign']*(local_point[1]-window['plane_y_m']))}
    return result


def inspect_surface(payload, samples):
    if not isinstance(samples,list) or not 1 <= len(samples) <= 6:
        raise ValueError('Request between one and six visible surface samples')
    results=[]
    loaded={}
    for sample in samples:
        camera=sample['camera'];uv=np.asarray(sample['pixel_uv'],dtype=float)
        radius=sample.get('radius',1)
        if (uv.shape!=(2,) or not np.isfinite(uv).all() or not isinstance(radius,int)
            or isinstance(radius,bool) or not 0<=radius<=4):
            raise ValueError('Each sample needs finite u,v and an integer radius from zero to four')
        view=payload['body_visual_reference']['views'][camera]
        width,height=view['resolution_wh']
        if not 0<=uv[0]<width or not 0<=uv[1]<height:
            raise ValueError('Requested pixel is outside the source RGB')
        ref=payload['depth_snapshots'][camera]
        if camera not in loaded:
            raw=Path(ref['path']).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=ref['sha256']:
                raise ValueError('Immutable depth snapshot changed')
            depth=np.load(ref['path'],allow_pickle=False)
            if depth.dtype!=np.float32 or depth.shape!=(height,width):
                raise ValueError('Expected aligned float32 optical-Z depth')
            loaded[camera]=depth
        depth=loaded[camera]
        u,v=np.floor(uv+.5).astype(int)
        if u>=width or v>=height:
            raise ValueError('Nearest depth pixel is outside the source image')
        patch=depth[max(0,v-radius):min(height,v+radius+1),max(0,u-radius):min(width,u+radius+1)]
        valid=np.isfinite(patch)&(patch>0)
        values=patch[valid]
        z=float(depth[v,u])
        spread=float(np.ptp(values)) if len(values) else None
        result={'label':str(sample.get('label','unspecified'))[:100], 'semantic_assignment':'model_selected_RGB_pixel_not_host_object_identity',
            'camera':camera,'requested_pixel_uv':uv.tolist(),'sampled_pixel_uv':[int(u),int(v)],
            'radius':radius,'depth_sha256':ref['sha256'],'rgb_sha256':payload['images'][camera]['sha256'],
            'valid_patch_values':int(valid.sum()),'patch_size':int(patch.size),
            'patch_depth_range_m':spread,'patch_depth_m':[[float(x) if np.isfinite(x) and x>0 else None for x in row] for row in patch],
            'boundary_or_mixed_surface_warning':bool(not valid.all() or (spread is not None and spread>.005)),
            'qualification':'A valid centre pixel is a sensor surface sample, not an object centre, hidden point, contact or retention certificate. '
                'Patch variation above 5mm is a warning, not a universal surface classifier or calibrated uncertainty.'}
        if not np.isfinite(z) or z<=0:
            result.update(status='NEEDS_WORK_no_valid_center_depth',world_xyz_m=None)
        else:
            k=np.asarray(view['intrinsics_K']);world_to_optical=np.asarray(view['world_to_optical_ros'])
            optical=z*np.linalg.solve(k,[int(u),int(v),1])
            world=(np.linalg.inv(world_to_optical)@[*optical,1])[:3]
            body={}
            for arm in ('left','right'):
                local=(np.linalg.inv(flange_transform(payload['state'][f'{arm}_ee_pose']))@[*world,1])[:3]
                body[arm]={'flange_xyz_m':local.tolist(),'relative_to_named_grasp_in_flange_axes_m':(local-[GRASP_POINT_OFFSET_M,0,0]).tolist(),
                    'distal_pad_relation':pad_relation(local,payload['body_visual_reference']['inner_face_windows_flange_m'][arm])}
            result.update(status='PASS_sensor_sample_not_semantic_geometry_truth',optical_z_m=z,
                world_xyz_m=world.tolist(),body_coordinates=body)
        results.append(result)
    return {'samples':results,'executed_actions':0,'physics_steps':0,
        'source':'same_capture_float32_distance_to_image_plane_and_actual_K_V',
        'pad_relation_conventions':'Flange X/Z projection onto the public distal inner-face outlines. Positive minimum edge margin means '
            'the projection is inside that pad outline; negative means outside at least one supporting edge. '
            'Positive inward plane distance means the sample lies on the gap side of that finger plane. '
            'Neither projection nor distance proves contact, opposite-face geometry, grasp retention, or that a different surface point is covered. '
            'Inspect separated points along a proposed contact region; one tip point inside is not shaft coverage.',
        'limitations':'Simulator optical-Z sensor data, without a realistic noise model. No task object asset, target pose, segmentation, hidden geometry, evaluator or controller is consulted.'}


def tool_schema():
    return {'type':'function','function':{'name':'inspect_surface','description':
        'Read depth at up to six visible RGB pixels in the CURRENT observation before choosing an EEF pose. '
        'Use for uncertain approach distance, tool shaft/handle or surface location rather than repeated blind millimetre moves. '
        'Choose pixels inside a visible surface; labels are your hypotheses. This tool advances no simulation and returns no action. '
        'It reports world/flange surface coordinates, local depth variation, and signed distances to the current distal pad edges and planes. '
        'Sample separated points along the intended contact region, not only its tip. The next move_eef pose is still yours to choose. '
        'Do not select hidden points, silhouettes outside the object or infer holding from a depth sample.',
        'parameters':{'type':'object','properties':{'observation_id':{'type':'string'},
            'correspondences':{'type':'array','minItems':2,'maxItems':3,'items':{'type':'object','properties':{
                'source':{'type':'string'},'target':{'type':'string'}},'required':['source','target'],'additionalProperties':False},
                'description':'Optional explicit source/target sample-label pairs. Computes conditional world-XY rotation, '
                    'selected-landmark center delta and fit residual, never object identity, COM or EEF action.'},
            'acting_arm':{'type':'string','enum':['left','right'],
                'description':'Expand local grasp/pad coordinates only for this acting arm. Omit for world-only geometry.'},
            'samples':{'type':'array','minItems':1,'maxItems':6,'items':{'type':'object','properties':{
                'label':{'type':'string'},'camera':{'type':'string','enum':['cam_head','cam_left_wrist','cam_right_wrist']},
                'pixel_uv':{'type':'array','items':{'type':'number'},'minItems':2,'maxItems':2},
                'radius':{'type':'integer','minimum':0,'maximum':4}},'required':['label','camera','pixel_uv'],'additionalProperties':False}}},
            'required':['observation_id','samples'],'additionalProperties':False}}}
