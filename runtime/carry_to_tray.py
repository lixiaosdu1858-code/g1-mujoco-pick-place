"""Fresh user-run grasp/carry/release using the accepted noslip=3 configuration."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import traceback

from lesson_common import ROOT, guard, save_json, sha, stamp
import grasp_lift as base
import grasp_solver_compare as solver
import grasp_solver_runtime as original
import centered_grasp_lift as centered
import carry_route
import carry_runtime as runtime

SUCCESS = ROOT/'outputs/solver_grasp_user_20261020_155236_233_repeat_n3'
SUCCESS_SHA = '73ace9b44044367f747f3dd70639e6ae20bd7e9265783fc8db9c98fe6e9e3e5b'
CORE_FILES = ('grasp_lift.py', 'reachable_motion.py', 'grasp_solver_compare.py',
              'grasp_solver_runtime.py', 'centered_grasp_lift.py', 'lesson_common.py',
              'reachable_pose_preview.py', 'pickplace_layout_preview.py', 'grasp_contact_geometry.py')


def setup(mujoco, np):
    if sha(SUCCESS/'summary.json') != SUCCESS_SHA:
        raise ValueError('Accepted repeat summary changed')
    accepted = json.loads((SUCCESS/'summary.json').read_text(encoding='utf-8-sig'))
    if (accepted['status'] != 'completed' or not accepted['grasp_hold_passed']
            or not accepted['tracking_thresholds_passed'] or accepted['steps'] != 7000):
        raise ValueError('Expected the accepted 28 s independent repeat')
    if sha(SUCCESS/'grasp.csv') != accepted['grasp_csv_sha256']:
        raise ValueError('Accepted repeat CSV changed')
    snapshots = {item['path']: item['sha256'] for item in accepted['snapshots']}
    for name in CORE_FILES:
        relative = 'code/'+name
        if sha(SUCCESS/relative) != snapshots[relative] or sha(Path(__file__).with_name(name)) != snapshots[relative]:
            raise ValueError('Accepted implementation dependency changed: '+name)
    ctx = solver.setup(3, mujoco, np)
    for key, current in (
        ('thresholds', base.THRESHOLDS), ('kp', base.motion.KP), ('kd', base.motion.KD),
        ('software_caps', base.motion.CAP), ('finger_open_m', base.OPEN),
        ('finger_closed_target_m', base.CLOSED), ('effective_solver_options', ctx['solver_effective_options']),
        ('diagnostic_schedule', original.SCHEDULE),
    ):
        if current != accepted[key]:
            raise ValueError('Accepted configuration changed: '+key)
    if (mujoco.__version__ != accepted['mujoco_version'] or np.__version__ != accepted['numpy_version']
            or not np.array_equal(ctx['data'].qpos, accepted['initial_qpos'])
            or not np.array_equal(ctx['data'].qvel, accepted['initial_qvel'])
            or ctx['data'].time != 0.):
        raise ValueError('Accepted software versions or initial state changed')
    before = (ctx['data'].qpos.copy(), ctx['data'].qvel.copy(), float(ctx['data'].time))
    ctx['carry_route'] = carry_route.load_route(ctx, mujoco, np)
    if (not np.array_equal(ctx['data'].qpos, before[0])
            or not np.array_equal(ctx['data'].qvel, before[1]) or ctx['data'].time != before[2]):
        raise AssertionError('Loading carry reference changed live integrated state')
    ctx['accepted_summary'] = accepted
    return ctx


def inspect(ctx, mujoco, np):
    before = (ctx['data'].qpos.copy(), ctx['data'].qvel.copy(), float(ctx['data'].time))
    initial = base.inspect(ctx, mujoco, np)
    stage = runtime.CarryStages({name: ctx['carry_route']['segments'][name]['duration_s']
                                for name in ('transport', 'descend', 'retract')})
    # Keep the original preparation's forward-call count and warm-start state.
    # Task metrics below are read-only additions to its already evaluated row.
    row = runtime.add_task_fields(ctx, stage, initial['initial_row'].copy(), mujoco, np)
    if (not np.array_equal(ctx['data'].qpos, before[0])
            or not np.array_equal(ctx['data'].qvel, before[1]) or ctx['data'].time != before[2]):
        raise AssertionError('Static inspection changed integrated state')
    return dict(status='static_carry_input_check_passed', physics_steps=0,
        initial_state_matches_accepted_repeat=True, first_grasp_protocol_unchanged=True,
        original_inspection=initial, initial_row=row,
        route_path=str(ctx['carry_route']['path']), route_sha256=ctx['carry_route']['sha256'],
        route_audit=ctx['carry_route']['audit'], placement_thresholds=runtime.PLACEMENT_THRESHOLDS,
        limitation='No GUI, physical transport or release was run. User outcome remains unknown.')


def snapshot(ctx, out, report):
    centered.snapshot(ctx, out, report)
    files = [(SUCCESS/'summary.json', 'accepted_repeat/summary.json'),
             (ctx['carry_route']['path'], 'carry_reference/path.json')]
    route_dir = ctx['carry_route']['path'].parent
    for name in ('receipt.json', 'static_samples.json', 'carry_route.py'):
        files.append((route_dir/name, 'carry_reference/'+name))
    for name in ('Run-CarryToTray.ps1', 'Show-CarryToTray.ps1'):
        files.append((Path(__file__).with_name(name), 'code/'+name))
    for source, relative in files:
        dest = out/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        report['snapshots'].append(dict(path=relative, sha256=sha(dest)))
    save_json(out/'source/effective_solver_options.json', ctx['solver_effective_options'])
    report['snapshots'].append(dict(path='source/effective_solver_options.json',
                                    sha256=sha(out/'source/effective_solver_options.json')))


def main():
    guard()
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    parser.add_argument('--inspect', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve() if args.output else ROOT/'outputs'/('carry_user_'+stamp())
    if out.parent != (ROOT/'outputs').resolve() or not out.name.startswith('carry_user_'):
        raise ValueError('Use a fresh flat outputs/carry_user_ directory')
    if not args.inspect:
        out.mkdir(exist_ok=False)
    os.chdir(ROOT/'outputs' if args.inspect else out)
    report = dict(kind='grasp_carry_release_dynamics', variant='noslip3_user_stage_confirmations',
        status='started', model_loaded=False, viewer_started=False, physics_started=False,
        steps=0, data_saved_samples=0, simulation_s=0., events=[], phases=[], snapshots=[],
        grasp_hold_passed=False, tracking_thresholds_passed=False, transfer_completed=False,
        release_started=False, placement_passed=False, learner_verified=False,
        full_cartesian_manual_control=False, full_v01_accepted=False, noslip_iterations=3,
        thresholds=base.THRESHOLDS, placement_thresholds=runtime.PLACEMENT_THRESHOLDS,
        object_stop_thresholds=dict(translation_m=.015, rotation_rad=.25),
        controller='PD_plus_zero_velocity_gravity', kp=base.motion.KP, kd=base.motion.KD,
        software_caps=base.motion.CAP, finger_open_m=base.OPEN, finger_closed_target_m=base.CLOSED,
        executable=sys.executable, temp=tempfile.gettempdir(),
        scope='Original grasp then user-confirmed transport, descent and release/retract. No attachment/state playback. Not free Cartesian keyboard control.',
        sample_semantics='Same-state pre-step actual/reference/motor/force samples; terminal future controls blank.',
        relative_motion_semantics='Captured original grasp reference; logged after opening but not a stop/hold criterion after release_active=1.',
        clearance_check_revision='box_frame_mesh_enclosure_lower_bound_v1',
        finger_clearance_semantics='No finger contacts and minimum conservative box-frame mesh enclosure distance >= unchanged 3 mm. Raw mj_geomDistance is logged separately and cannot certify separation.',
        tracking_result_semantics='Position/orientation thresholds over observed samples with at least one physics step; independent of task completion.',
        slip_metric_semantics='max_relative_slip_m includes released separation for legacy compatibility; max_held_relative_slip_m/rotation only include release_active=0.',
        tray_containment_semantics='Rotated full-box projected half extents, equivalent to all eight cube corners inside the tray XY interior.',
        accepted_repeat_path=str(SUCCESS), accepted_repeat_summary_sha256=SUCCESS_SHA)
    started, code = time.perf_counter(), 0
    try:
        import numpy as np
        import mujoco
        ctx = setup(mujoco, np)
        inspection = inspect(ctx, mujoco, np)
        cfg = ctx['cfg']
        report.update(model_loaded=True, mujoco_version=mujoco.__version__, numpy_version=np.__version__,
            dt_s=base.DT, static_inspection=inspection,
            initial_qpos=ctx['data'].qpos.tolist(), initial_qvel=ctx['data'].qvel.tolist(),
            effective_solver_options=ctx['solver_effective_options'],
            table_top_z_m=cfg['table']['top_z_m'], tray_floor_z_m=cfg['tray']['floor_top_z_m'],
            tray_rim_z_m=cfg['tray']['wall_top_z_m'], tray_center_xy_m=cfg['tray']['center_xy_m'],
            tray_yaw_rad=float(np.arctan2(ctx['carry_route']['tray']['rotation_world'][1][0],
                                        ctx['carry_route']['tray']['rotation_world'][0][0])),
            tray_inner_half_xy_m=cfg['tray']['interior_half_size_xy_m'],
            route_sha256=ctx['carry_route']['sha256'],
            carry_segment_durations_s={k:ctx['carry_route']['segments'][k]['duration_s'] for k in ('transport','descend','retract')})
        if args.inspect:
            print(json.dumps(inspection, ensure_ascii=False, indent=2))
            return 0
        snapshot(ctx, out, report)
        print('CARRY RUN:', out, flush=True)
        code = runtime.run(ctx, report, out, mujoco, np)
    except KeyboardInterrupt:
        report.update(status='interrupted', end_reason='KeyboardInterrupt', placement_passed=False)
        code = 130
    except Exception:
        report.update(status='error', end_reason='exception', placement_passed=False, error=traceback.format_exc())
        code = 1
        print(report['error'], file=sys.stderr)
        if not args.inspect:
            (out/'error.txt').write_text(report['error'], encoding='utf-8')
    finally:
        report['wall_s'] = time.perf_counter()-started
        if not args.inspect:
            if (out/'grasp.csv').exists():
                report['grasp_csv_sha256'] = sha(out/'grasp.csv')
            save_json(out/'summary.json', report)
            print('SUMMARY:', out/'summary.json', flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
