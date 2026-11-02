"""Identity of command execution, separate from policy, memory and dialogue identity."""
from __future__ import annotations
import ast
import hashlib
import inspect
import json
from pathlib import Path

FORBIDDEN = frozenset(('policy', 'manager', 'knowledge', 'session', 'clarification',
    'perception', 'preparation', 'library', 'readout'))
NATIVE_SOURCES = ('src/eval_client/eval_env.py','env/environment/base_env.py',
    'env/robot_manager/control_manager.py','env/robot_manager/robot_manager.py')


def local_closure(roots):
    package=Path(__file__).resolve().parent
    pending=[Path(p).resolve() for p in roots];result={}
    while pending:
        path=pending.pop()
        if path in result:continue
        if path.parent!=package or path.stem in FORBIDDEN:
            raise ValueError('Motion kernel imports interaction code: '+str(path))
        raw=path.read_bytes();result[path]=hashlib.sha256(raw).hexdigest()
        for node in ast.walk(ast.parse(raw)):
            modules=[]
            if isinstance(node,ast.ImportFrom):
                if node.level==1:
                    modules=[node.module] if node.module else [a.name for a in node.names]
                elif node.module and node.module.startswith('kyb.'):
                    modules=[node.module[4:]]
                elif node.module=='kyb':
                    modules=[a.name for a in node.names]
                elif node.level:
                    raise ValueError('Motion imports outside its explicit package boundary')
            elif isinstance(node,ast.Import):
                modules=[a.name[4:] for a in node.names if a.name.startswith('kyb.')]
            for module in modules:
                dependency=package.joinpath(*module.split('.')).with_suffix('.py')
                if not dependency.is_file():raise ValueError('Unresolved motion dependency: '+module)
                pending.append(dependency)
    return {'kyb/'+p.name:sha for p,sha in sorted(result.items())}


def build(*, control_hz, physics_dt, effective_parameters, action_contract):
    from XPolicyLab.policy.GPT_6_Astra_Direct_EEF.policy import EefAgentPolicy as FrozenBase
    package=Path(__file__).resolve().parent
    sources=local_closure([package/'motion_core.py',package/'execution.py',package/'body_guard.py'])
    adapter=Path(inspect.getfile(FrozenBase)).resolve().parent
    # The mature base includes its own non-motion dispatch. Pin that whole
    # subsystem; do not recursively refactor it to manufacture a pure closure.
    for p in sorted(adapter.rglob('*.py')):
        sources['frozen_eef/'+str(p.relative_to(adapter))]=hashlib.sha256(p.read_bytes()).hexdigest()
    repository=adapter.parents[2]
    for name in NATIVE_SOURCES:
        sources['native/'+name]=hashlib.sha256((repository/name).read_bytes()).hexdigest()
    protocol={
        'scope':'Given command, measured state and planner response: native actions, rejection and execution-state updates. Not whole-policy or physical-outcome equivalence.',
        'resampling':'weighted_Linf_polyline_arc_length_including_measured_start',
        'control_hz':float(control_hz),'physics_dt':float(physics_dt),
        'effective_parameters':effective_parameters,'motion_sources':sources,
        'public_action_contract_sha256':hashlib.sha256(action_contract.encode()).hexdigest(),
        'qualification':'Historical pins are never rewritten or mapped to this identity. Planner settings retain their separate pin.'}
    return protocol,hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
