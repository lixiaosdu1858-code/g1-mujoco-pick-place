"""One-factor contact-solver diagnostic, independently reset and started by the user."""
import argparse
import csv
import json
import math
import numbers
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback

from lesson_common import ROOT, guard, save_json, sha, stamp
import centered_grasp_lift as centered
import grasp_lift as base
import reachable_motion as motion

BASE_RUN = ROOT/'outputs/centered_grasp_user_20261019_105256_607'
SOURCE_LINKS = [
    'https://mujoco.readthedocs.io/en/3.3.7/modeling.html#preventing-slip',
    'https://mujoco.readthedocs.io/en/3.3.7/XMLreference.html#option-noslip_iterations',
]


def options_dict(opt):
    result = {}
    for name in dir(opt):
        if name.startswith('_'): continue
        value = getattr(opt, name)
        if isinstance(value, numbers.Integral): result[name] = int(value)
        elif isinstance(value, numbers.Real): result[name] = float(value)
        elif hasattr(value, 'tolist'): result[name] = value.tolist()
    return result


def setup(noslip, mujoco, np):
    if noslip not in (0, 3): raise ValueError('This registered comparison permits only 0 and 3')
    ctx = centered.setup(mujoco, np)
    before = options_dict(ctx['model'].opt)
    if before['noslip_iterations'] != 0: raise ValueError('Frozen baseline is no longer noslip=0')
    ctx['model'].opt.noslip_iterations = noslip
    after = options_dict(ctx['model'].opt)
    differences = [k for k in before if before[k] != after[k]]
    if differences != ([] if noslip == 0 else ['noslip_iterations']):
        raise AssertionError('Unregistered option differences: '+str(differences))
    ctx.update(solver_baseline_options=before, solver_effective_options=after, noslip_iterations=noslip)
    return ctx


def load_run(path):
    path = Path(path).resolve()
    if path.parent != (ROOT/'outputs').resolve() or not path.name.startswith('solver_grasp_user_'):
        raise ValueError('Expected a solver_grasp_user_ project run')
    s = json.loads((path/'summary.json').read_text(encoding='utf-8-sig'))
    if sha(path/'grasp.csv') != s['grasp_csv_sha256']: raise ValueError('CSV hash mismatch')
    if not all(sha(path/x['path']) == x['sha256'] for x in s['snapshots']):
        raise ValueError('Snapshot hash mismatch')
    with (path/'grasp.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    return s, rows


def metrics(s, rows):
    ready = [r for r in rows if r['phase'] == 'grip_ready']
    lift = next((r for r in rows if r['phase'] == 'lifting'), None)
    result = dict(complete_wait_window=False, wait_duration_s=None, wait_relative_drift_mm=None,
                  wait_drift_rate_mm_s=None, max_clearance_mm=None, rotation_at_stop_rad=None)
    if rows:
        result['max_clearance_mm'] = max(float(r['cube_bottom_clearance_m'])*1000 for r in rows)
        if rows[-1]['relative_rotation_rad'] != '': result['rotation_at_stop_rad'] = float(rows[-1]['relative_rotation_rad'])
    if ready:
        first, last = ready[0], lift if lift is not None else ready[-1]
        duration = float(last['sim_s'])-float(first['sim_s'])
        drift = (float(last['relative_slip_m'])-float(first['relative_slip_m']))*1000
        result.update(complete_wait_window=lift is not None and abs(duration-10.)<1e-8,
                      wait_duration_s=duration, wait_relative_drift_mm=drift,
                      wait_drift_rate_mm_s=drift/duration if duration>0 else None)
    return result


def summarize(compare_path):
    out = Path(compare_path).resolve()
    if out.parent != (ROOT/'outputs').resolve() or not out.name.startswith('grasp_solver_compare_user_'):
        raise ValueError('Expected a project comparison folder')
    index = json.loads((out/'runs.json').read_text(encoding='utf-8-sig'))
    report = dict(kind='grasp_solver_comparison', cases=[], checks={}, comparison_valid=False,
                  scope='Two user-started reset diagnostic runs; one per option, no success-rate estimate or full pick/place claim.')
    loaded=[]
    for item in index['runs']:
        p = Path(item['run_path'])
        try:
            s, rows = load_run(p)
        except (OSError, KeyError, ValueError) as exc:
            report['cases'].append(dict(noslip_iterations=item['noslip_iterations'],run_path=str(p),
                status='artifact_error',end_reason='saved_data_unavailable_or_invalid',grasp_hold_passed=False,
                data_available=False,artifact_error=str(exc),metrics=None,
                summary_sha256=sha(p/'summary.json') if (p/'summary.json').exists() else None,csv_sha256=None))
            continue
        if s.get('noslip_iterations') != item['noslip_iterations']: raise ValueError('Case label mismatch')
        case = dict(noslip_iterations=s['noslip_iterations'], run_path=str(p), status=s['status'],
                    end_reason=s.get('end_reason'), grasp_hold_passed=s['grasp_hold_passed'],data_available=True,
                    summary_sha256=sha(p/'summary.json'), csv_sha256=sha(p/'grasp.csv'),
                    metrics=metrics(s, rows))
        report['cases'].append(case); loaded.append((s,rows))
    report['checks']['artifact_integrity']=all(c['data_available'] for c in report['cases']) and bool(report['cases'])
    if len(loaded)==2 and len(report['cases'])==2:
        a,b = loaded[0][0],loaded[1][0]
        ao,bo = a['effective_solver_options'].copy(),b['effective_solver_options'].copy()
        ao.pop('noslip_iterations');bo.pop('noslip_iterations')
        report['checks'] = dict(
            artifact_integrity=True,
            registered_options=[a['noslip_iterations'],b['noslip_iterations']]==[0,3],
            other_solver_options_equal=ao==bo,
            initial_qpos_equal=a['initial_qpos']==b['initial_qpos'],
            initial_qvel_equal=a['initial_qvel']==b['initial_qvel'],
            controller_and_thresholds_equal=all(a[k]==b[k] for k in ('thresholds','kp','kd','software_caps','finger_closed_target_m','finger_open_m')),
            timing_protocol_equal=a['timing_protocol']==b['timing_protocol'],
            scene_and_reference_equal=a['frozen_inputs']==b['frozen_inputs'],
            both_wait_windows_complete=all(c['metrics']['complete_wait_window'] for c in report['cases']),
            neither_execution_error=all(s['status'] not in ('error','interrupted') for s,_ in loaded),
        )
        report['comparison_valid']=all(report['checks'].values())
    report['interpretation'] = 'Compare wait drift first, then lift/hold. Noslip affects all frictional contacts, including cube/table. A changed outcome is not proof of hardware realism or a unique cause.'
    save_json(out/'comparison.json',report)
    print('SOLVER COMPARE:',out)
    for c in report['cases']:
        print('NOSLIP',c['noslip_iterations'],c['status'],c['end_reason'],json.dumps(c['metrics']))
    print('COMPARISON VALID:',report['comparison_valid'])
    return report


def main():
    guard()
    parser=argparse.ArgumentParser()
    parser.add_argument('--noslip',type=int,choices=(0,3),default=0)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--inspect',action='store_true')
    parser.add_argument('--compare',type=Path)
    args=parser.parse_args()
    if args.compare:
        summarize(args.compare); return 0
    import numpy as np
    import mujoco
    import grasp_solver_runtime as runtime
    out=args.output.resolve() if args.output else ROOT/'outputs'/('solver_grasp_user_'+stamp())
    if out.parent!=(ROOT/'outputs').resolve() or not out.name.startswith('solver_grasp_user_'):
        raise ValueError('Use a new solver_grasp_user_ project output')
    if not args.inspect: out.mkdir(exist_ok=False)
    os.chdir(ROOT/'outputs' if args.inspect else out)
    report=dict(kind='grasp_lift_dynamics',variant='scheduled_noslip_'+str(args.noslip),status='started',
        model_loaded=False,viewer_started=False,physics_started=False,steps=0,data_saved_samples=0,simulation_s=0.,
        events=[],phases=[],grasp_hold_passed=False,tracking_thresholds_passed=False,learner_verified=False,
        transport_validated=False,release_validated=False,thresholds=base.THRESHOLDS,
        object_stop_thresholds=dict(translation_m=.015,rotation_rad=.25),controller='PD_plus_zero_velocity_gravity',
        finger_open_m=base.OPEN,finger_closed_target_m=base.CLOSED,kp=motion.KP,kd=motion.KD,software_caps=motion.CAP,
        executable=sys.executable,temp=tempfile.gettempdir(),snapshots=[],noslip_iterations=args.noslip,
        timing_protocol=dict(approach_s=6.,B_ready_s=2.,closing_s=2.,minimum_grip_settle_s=1.,
                             grip_ready_wait_s=10.,lifting_s=4.,lift_hold_s=3.),
        sample_semantics='Original same-state pre-step sample; terminal future controls blank.',
        scope='One-factor solver diagnostic with scheduled phases. User starts and observes; not free keyboard pick/place.',
        hypothesis='Soft friction constraints contribute to drift; noslip=3 might reduce it. Grasp success is unknown.',
        official_sources=SOURCE_LINKS)
    start=time.perf_counter();code=0
    try:
        ctx=setup(args.noslip,mujoco,np);inspection=base.inspect(ctx,mujoco,np)
        report.update(model_loaded=True,mujoco_version=mujoco.__version__,numpy_version=np.__version__,dt_s=base.DT,
            static_inspection=inspection,effective_solver_options=ctx['solver_effective_options'],
            baseline_solver_options=ctx['solver_baseline_options'],
            initial_qpos=ctx['data'].qpos.tolist(),initial_qvel=ctx['data'].qvel.tolist(),
            frozen_inputs=dict(scene=sha(ctx['folder']/'scene.xml'),reference=sha(ctx['reference_file']),
                               base_controller=sha(Path(base.__file__)),motion=sha(Path(motion.__file__)),
                               runtime=sha(Path(runtime.__file__)),entry=sha(Path(__file__))),
            layout_translation_world_m=ctx['comparison_index']['layout_translation_world_m'])
        if args.inspect:
            print(json.dumps(dict(status='prepared_static_only',noslip=args.noslip,physics_steps=0,
                                  options=ctx['solver_effective_options'],inspection=inspection),indent=2));return 0
        centered.snapshot(ctx,out,report)
        # Record the effective runtime override separately from the unchanged MJCF.
        save_json(out/'source/effective_solver_options.json',ctx['solver_effective_options'])
        report['snapshots'].append(dict(path='source/effective_solver_options.json',sha256=sha(out/'source/effective_solver_options.json')))
        for name in ('Run-GraspSolverCompare.ps1','Show-GraspSolverCompare.ps1'):
            import shutil
            src=Path(__file__).with_name(name);dest=out/'code'/name;shutil.copy2(src,dest)
            report['snapshots'].append(dict(path='code/'+name,sha256=sha(dest)))
        print('SOLVER RUN:',out,flush=True)
        print('Noslip iterations:',args.noslip,'| Scheduled diagnostic; physical outcome unknown.',flush=True)
        code=runtime.run(ctx,report,out,mujoco,np)
        if 'stop_diagnostics' in report:
            row=report['stop_diagnostics']['row']
            report['object_stop_components']=dict(translation_exceeded=row['relative_slip_m']!='' and row['relative_slip_m']>.015,
                                                  rotation_exceeded=row['relative_rotation_rad']!='' and row['relative_rotation_rad']>.25)
    except KeyboardInterrupt:
        report.update(status='interrupted',end_reason='KeyboardInterrupt',grasp_hold_passed=False);code=130
    except Exception:
        report.update(status='error',end_reason='exception',grasp_hold_passed=False,error=traceback.format_exc());code=1
        print(report['error'],file=sys.stderr)
        if not args.inspect:(out/'error.txt').write_text(report['error'],encoding='utf-8')
    finally:
        report['wall_s']=time.perf_counter()-start
        if not args.inspect:
            if (out/'grasp.csv').exists():report['grasp_csv_sha256']=sha(out/'grasp.csv')
            save_json(out/'summary.json',report);print('SUMMARY:',out/'summary.json',flush=True)
    return code


if __name__=='__main__':raise SystemExit(main())
