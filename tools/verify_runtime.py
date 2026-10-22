"""Static-only package integrity/model/initial-reference check. No viewer or step."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'))
from lesson_common import guard,save_json,stamp

def main():
    guard()
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    import numpy as np
    import mujoco
    def forbidden(*a,**k):raise AssertionError('Physics integration is forbidden in static verification')
    mujoco.mj_step=mujoco.mj_step1=mujoco.mj_step2=forbidden
    import manual_pickplace as manual
    ctx=manual.setup(mujoco,np)
    reference=json.loads((ROOT/'configs/static_baseline.json').read_text(encoding='utf-8'))
    mismatches=[]
    for name,expected in reference['model_arrays'].items():
        value=getattr(ctx['model'],name)
        actual=dict(shape=list(value.shape),dtype=str(value.dtype),sha256=hashlib.sha256(value.tobytes()).hexdigest())
        if actual!=expected:mismatches.append(name)
    if mismatches:raise ValueError('Compiled model mismatch: '+','.join(mismatches))
    planner,inspection=manual.inspect(ctx,mujoco,np)
    row=inspection['initial_sample'];old=reference['initial_sample']
    differences=[]
    for key,value in old.items():
        if isinstance(value,(int,float)) and not isinstance(value,bool):
            if not np.isclose(row[key],value,rtol=0,atol=1e-11):differences.append(key)
        elif row[key]!=value:differences.append(key)
    if differences:raise ValueError('Initial sample mismatch: '+','.join(differences))
    from manual_task_target import TaskPlanner
    for name,expected in reference['reference_probes'].items():
        p=TaskPlanner(ctx,mujoco,np);q=ctx['data'].qpos[ctx['qa']].copy()
        result=p.plan_jog([.0015,0.,0.],q) if name=='world_x_jog' else p.plan_manual_fingers(manual.base.CLOSED,q)
        actual=dict(accepted=result['accepted'],reason=result['reason'],samples=len(result.get('rows',[])),
            rows_sha256=hashlib.sha256(json.dumps(result.get('rows',[]),sort_keys=True,separators=(',',':')).encode()).hexdigest())
        if actual!=expected:raise ValueError('Static reference mismatch: '+name)
    if ctx['data'].time!=0.:raise AssertionError('Unexpected time advancement')
    result=dict(status='static_package_verified',physics_steps=0,viewer_opened=False,
        model_arrays_identical=len(reference['model_arrays']),initial_sample_fields_identical=len(old),
        reference_probes_identical=len(reference['reference_probes']),
        model_loaded=True,planner_constructed=True,mujoco_version=mujoco.__version__,numpy_version=np.__version__,
        baseline_source_run_id='manual_user_20261021_175535_099',
        limitation='This invocation checks static equivalence only. Separate user-run dynamics evidence is in docs/candidate-user-verification.md.')
    out=args.output or ROOT/'outputs'/('static_check_'+stamp()+'.json')
    out.parent.mkdir(parents=True,exist_ok=True);save_json(out,result)
    print(json.dumps(result,indent=2));print('STATIC REPORT:',out)
    return 0

if __name__=='__main__':raise SystemExit(main())
