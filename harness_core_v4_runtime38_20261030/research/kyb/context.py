"""Rebuildable policy views. Full conversations and SQLite evidence stay archived."""
from __future__ import annotations

import copy
import json


def rgb_turns(messages):
    return [i for i, m in enumerate(messages) if isinstance(m.get('content'), list)
            and any(p.get('type') == 'image_url' for p in m['content'])]


def latest_context(messages):
    turns = rgb_turns(messages)
    if not turns:
        return {}
    text = messages[turns[-1]]['content'][0].get('text', '')
    return json.loads(text.split('\n<KYB_CONTEXT>\n', 1)[1].split('\n</KYB_CONTEXT>', 1)[0]) \
        if '\n<KYB_CONTEXT>\n' in text else {}


def surface_readout(result, acting_arm):
    """Decision-sized measurements; complete patches/geometry remain immutable evidence."""
    samples=[]
    for original in result['samples']:
        sample={k:original[k] for k in ('label','camera','sampled_pixel_uv','status','world_xyz_m',
            'boundary_or_mixed_surface_warning','patch_depth_range_m','valid_patch_values','patch_size') if k in original}
        body=original.get('body_coordinates',{})
        if acting_arm is not None and body and acting_arm not in body:raise ValueError('Unknown acting arm')
        if acting_arm in body:
            detail=body[acting_arm]
            sample['acting_frame']={
                'arm':acting_arm,
                'relative_to_grasp_flange_axes_m':detail.get('relative_to_named_grasp_in_flange_axes_m'),
                'inner_pad':{finger:{k:relation[k] for k in ('minimum_projected_edge_margin_m',
                    'inward_distance_from_face_plane_m')} for finger,relation in detail.get('distal_pad_relation',{}).items()}}
        samples.append(sample)
    def rounded(value):
        if isinstance(value,float):return round(value,6)
        if isinstance(value,list):return [rounded(v) for v in value]
        if isinstance(value,dict):return {k:rounded(v) for k,v in value.items()}
        return value
    return rounded({'samples':samples,
        'qualification':'RGB-selected surface measurements, not certified object/part identity or contact. '
            'Pad relations describe inner finger faces, not exterior pushing surfaces. Full depth patches and '
            'geometry stay in the archived surface_inspection record.',
        **{k:v for k,v in result.items() if k not in ('samples','pad_relation_conventions')}})


def surface_evidence_view(result, body):
    """Preserve exact sensor values while exposing only the requested body geometry."""
    if body not in ('none', 'ee', 'full'):
        raise ValueError('Unknown surface body view')
    view=copy.deepcopy(result);omitted=set()
    if body!='full':
        if 'pad_relation_conventions' in view:
            del view['pad_relation_conventions'];omitted.add('pad_relation_conventions')
        for sample in view.get('samples',[]):
            if body=='none':
                if 'body_coordinates' in sample:
                    del sample['body_coordinates'];omitted.add('samples[].body_coordinates')
            else:
                for detail in sample.get('body_coordinates',{}).values():
                    if 'distal_pad_relation' in detail:
                        del detail['distal_pad_relation'];omitted.add('samples[].body_coordinates.*.distal_pad_relation')
    return view,sorted(omitted)
