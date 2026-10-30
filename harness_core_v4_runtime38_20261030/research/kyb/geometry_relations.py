"""Conditional geometry of explicitly paired observed landmarks; no action solver."""
import numpy as np

def planar_correspondence(samples,pairs):
    if not isinstance(pairs,list) or not 2<=len(pairs)<=3:raise ValueError('Provide two or three explicit source/target pairs')
    labels=[sample['label'] for sample in samples]
    if len(set(labels))!=len(labels):raise ValueError('Correspondence labels must be unique')
    by_label={sample['label']:sample for sample in samples}
    source=[pair['source'] for pair in pairs];target=[pair['target'] for pair in pairs]
    if len(set(source+target))!=2*len(pairs):raise ValueError('Every correspondence endpoint must use a different selected sample')
    def points(names):
        result=[]
        for name in names:
            if name not in by_label or by_label[name].get('world_xyz_m') is None:raise ValueError('Correspondence needs valid current samples')
            result.append(by_label[name]['world_xyz_m'][:2])
        array=np.asarray(result,dtype=float)
        if not np.isfinite(array).all():raise ValueError('Finite observed points required')
        return array
    a,b=points(source),points(target);ac,bc=a.mean(axis=0),b.mean(axis=0)
    def spread(p):return float(max(np.linalg.norm(p[i]-p[j]) for i in range(len(p)) for j in range(i)))
    ba,bb=spread(a),spread(b)
    if min(ba,bb)<1e-6:raise ValueError('Both point sets need a nonzero baseline')
    u,_,vt=np.linalg.svd((a-ac).T@(b-bc))
    correction=np.eye(2);correction[-1,-1]=np.linalg.det(vt.T@u.T)
    rotation=vt.T@correction@u.T
    translated=(a-ac)@rotation.T+bc
    errors=np.linalg.norm(translated-b,axis=1)
    return {'mapping':'declared source→target, world XY projection, proper rotation only',
        'rotation_deg_ccw_about_world_positive_z':float(np.degrees(np.arctan2(rotation[1,0],rotation[0,0]))),
        'landmark_center_delta_xy_m':(bc-ac).tolist(),
        'source_baseline_m':ba,'target_baseline_m':bb,'fit_rms_m':float(np.sqrt(np.mean(errors**2))),
        'pair_residuals_m':errors.tolist(),
        'qualification':'Conditional on model-declared pixel identities/correspondences. Low RMS does not certify identity, contact, '
            'COM, task completion or an EEF target. '+('Two pairs have no extra correspondence for an independent consistency check.' if len(pairs)==2 else 'Check individual residuals and visible correspondence.')}
