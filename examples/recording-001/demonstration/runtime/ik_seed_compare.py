"""User-run, bounded IK seed comparison; numerical kinematics, never dynamics.

The six real trials belong to the learner. Maintainer checks may use --inspect
(hashes and known A/C FK only) or synthetic callbacks with solve_trial().
"""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

from lesson_common import ROOT, guard, save_json, sha, stamp

SOURCE_RUN = ROOT / 'outputs/task_pose_user_20261015_194500_360'
SOURCE = SOURCE_RUN / 'source'
SOURCE_SUMMARY_SHA = 'b96ee4b88dd3cec42d2cd1e598a4676518a3c78b48780b7a26ae502b5105ae38'
SOURCE_HASHES = {
    'config.json': '8bb009d5b7d25062fb387815c70fa642181841dcb09349d905b4ae2d2ce33b94',
    'scene.xml': '91b5c579b0b09fdf388bc3d8f37c8efd2e7dab106295f675bf40db502268fe56',
    'static_audit.json': 'dfa0b0c47398f8be27815ba14b94522878d8dd56094011e939c5d1eae41f302d',
    'summary.json': 'f8835616e439c52bc158d4d01d8dbd68861eb275c4176e2931332e1ce84efaeb',
}
LEGACY_PATH = ROOT / 'scripts/grasp/prepare_tray_shift.py'
LEGACY_SHA = '975b719675895d0a2e8aab4c199073fd5c74705fb4f1d1c311c4a9ff8677bea2'
KIND = 'ik_seed_initialization_compare'
TRACE_KIND = 'accepted_update_states'
SOLVER_SETTINGS = {
    'algorithm': 'legacy clipped damped least squares with first improving line-search candidate',
    'max_attempts': 180, 'orientation_weight': 0.15, 'damping': 1e-6,
    'max_abs_joint_step_rad': 0.12, 'line_search_scales': [1.0, 0.5, 0.25, 0.125, 0.0625],
    'joint_limit_inset_rad': 0.03, 'strict_position_tolerance_m': 2e-5,
    'strict_orientation_tolerance_rad': 2e-4,
    'task_position_tolerance_m': 0.003, 'task_orientation_tolerance_rad': 0.05,
    'collision_aware_solver': False, 'seed_projection': False,
    'weighted_cost_definition': 'norm([target_position-FK_position, 0.15*orientation_error_rotation_vector])',
    'weighted_cost_note': 'Position in metres, orientation in radians, scale 0.15 m/rad; not unweighted pose error.',
}
CSV_FIELDS = ['trial_id', 'target_id', 'seed_id', 'iteration', 'position_error_mm',
              'orientation_error_rad', 'weighted_cost', 'min_solver_margin_rad',
              'nearest_joint', 'row_kind']
POINT_IDS = {'A': 'pregrasp', 'B': 'grasp', 'C': 'lift', 'D': 'above_tray', 'E': 'release'}
JOINT_ORDER = ['right_shoulder_pitch_joint', 'right_shoulder_roll_joint',
               'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint',
               'right_wrist_pitch_joint', 'right_wrist_yaw_joint']


def boundary_report(np, q, limits, joint_names):
    physical = np.column_stack((q - limits[:, 0], limits[:, 1] - q))
    solver = physical - 0.03
    index, side = np.unravel_index(int(np.argmin(solver)), solver.shape)
    return {
        'physical_margins_rad': physical.tolist(), 'solver_margins_rad': solver.tolist(),
        'margin_column_order': ['lower', 'upper'],
        'min_model_margin_rad': float(np.min(physical)),
        'min_solver_margin_rad': float(solver[index, side]),
        'nearest_joint': joint_names[index], 'nearest_side': ['lower', 'upper'][side],
        'within_physical_limits': bool(np.all(physical >= 0)),
        'within_solver_limits': bool(np.all(solver >= -1e-12)),
    }


def solve_trial(np, evaluate, rot_error, limits, joint_names, target, seed,
                target_id, seed_id, max_attempts=180):
    """Legacy arithmetic and acceptance rule, with observable state transitions.

    evaluate(q) returns position, rotation matrix, position Jacobian, rotation
    Jacobian. The callback may be synthetic for QA. Real CLI fixes 180 attempts.
    The final evaluate(q) is essential: a rejected candidate leaves MjData at a
    different pose even though the solver's returned q has not changed.
    """
    trial_id = target_id + '_from_' + seed_id
    q = np.asarray(seed, dtype=float).copy()
    target = np.asarray(target, dtype=float)
    rows, states, attempts = [], [], []
    accepted_updates = 0
    clip_candidate_count = 0
    stop_reason = 'iteration_budget'
    termination_checked_strict = False

    def measure(q, result):
        p, rotation, jp, jr = result
        ep, er = target - p, rot_error(rotation)
        error = np.r_[ep, .15 * er]
        boundaries = boundary_report(np, q, limits, joint_names)
        state = {
            'iteration': accepted_updates, 'q_arm_rad': q.tolist(),
            'fk_position_world_m': p.tolist(), 'fk_rotation_world': rotation.tolist(),
            'position_error_world_m': ep.tolist(), 'orientation_error_vector_rad': er.tolist(),
            'position_error_mm': float(np.linalg.norm(ep) * 1000),
            'orientation_error_rad': float(np.linalg.norm(er)),
            'weighted_cost': float(np.linalg.norm(error)), **boundaries,
        }
        if not np.isfinite(q).all() or not np.isfinite(error).all():
            raise FloatingPointError('Non-finite solver state')
        return state, ep, er, error, jp, jr

    def record_state(state):
        kind = 'initial' if state['iteration'] == 0 else 'accepted'
        row = {k: state[k] for k in CSV_FIELDS if k in state}
        row.update(trial_id=trial_id, target_id=target_id, seed_id=seed_id, row_kind=kind)
        rows.append(row)
        states.append(dict(state, row_kind=kind))

    for attempt_index in range(max_attempts):
        state, ep, er, error, jp, jr = measure(q, evaluate(q))
        if not rows or rows[-1]['iteration'] != accepted_updates:
            record_state(state)
        attempt = {'attempt_index': attempt_index, 'accepted_updates_before': accepted_updates,
                   'q_start_rad': q.tolist(), 'position_error_mm': state['position_error_mm'],
                   'orientation_error_rad': state['orientation_error_rad'],
                   'weighted_cost': state['weighted_cost'], 'candidates': [], 'accepted': False}
        attempts.append(attempt)
        if np.linalg.norm(ep) < 2e-5 and np.linalg.norm(er) < 2e-4:
            stop_reason = 'strict_converged'
            termination_checked_strict = True
            attempt['strict_termination_check_passed'] = True
            break
        # Same expressions, constants, operation order and strict cost inequality
        # as the frozen prepare_tray_shift.py solver. Do not add collision costs.
        jacobian = np.vstack([jp, .15 * jr])
        if not np.isfinite(jacobian).all():
            raise FloatingPointError('Non-finite solver Jacobian')
        step = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 1e-6 * np.eye(6), error)
        if not np.isfinite(step).all():
            raise FloatingPointError('Non-finite solver step')
        attempt['raw_step_rad'] = step.tolist()
        step *= min(1., .12 / max(np.max(np.abs(step)), 1e-12))
        attempt['bounded_step_rad'] = step.tolist()
        accepted = False
        for scale in (1., .5, .25, .125, .0625):
            proposed = q + scale * step
            trial = np.clip(proposed, limits[:, 0] + .03, limits[:, 1] - .03)
            tp, tR, _, _ = evaluate(trial)
            trial_error = np.r_[target - tp, .15 * rot_error(tR)]
            cost = np.linalg.norm(trial_error)
            if not np.isfinite(trial_error).all() or not np.isfinite(cost):
                raise FloatingPointError('Non-finite line-search candidate error')
            clipped = trial != proposed
            clip_candidate_count += int(bool(np.any(clipped)))
            candidate = {
                'scale': scale, 'proposed_q_arm_rad': proposed.tolist(),
                'candidate_q_arm_rad': trial.tolist(), 'clipped': bool(np.any(clipped)),
                'clipped_joints': [joint_names[i] for i in np.flatnonzero(clipped)],
                'clip_delta_rad': (trial - proposed).tolist(),
                'fk_position_world_m': tp.tolist(), 'position_error_mm': float(np.linalg.norm(target - tp) * 1000),
                'orientation_error_rad': float(np.linalg.norm(rot_error(tR))),
                'weighted_cost': float(cost), 'cost_decrease': float(np.linalg.norm(error) - cost),
                'accepted': bool(cost < np.linalg.norm(error)),
                **boundary_report(np, trial, limits, joint_names),
            }
            attempt['candidates'].append(candidate)
            if cost < np.linalg.norm(error):
                q = trial
                accepted = True
                accepted_updates += 1
                attempt['accepted'] = True
                attempt['accepted_scale'] = scale
                break
        if not accepted:
            stop_reason = 'no_acceptable_step'
            break
    # Re-establish returned q, including after rejected candidates. The final
    # accepted update of attempt 179 is recorded even without a following loop.
    final, ep, er, _, _, _ = measure(q, evaluate(q))
    if not rows or rows[-1]['iteration'] != accepted_updates:
        record_state(final)
    final_kind = 'initial_final' if accepted_updates == 0 else 'final'
    rows[-1]['row_kind'] = final_kind
    states[-1] = dict(final, row_kind=final_kind)
    strict_satisfied = bool(np.linalg.norm(ep) < 2e-5 and np.linalg.norm(er) < 2e-4)
    task_pass = bool(np.linalg.norm(ep) <= .003 and np.linalg.norm(er) <= .05)
    summary = {
        'trial_id': trial_id, 'target_id': target_id, 'seed_id': seed_id,
        'target_config_id': POINT_IDS.get(target_id, target_id),
        'seed_config_id': POINT_IDS.get(seed_id, seed_id), 'target_position_world_m': target.tolist(),
        'initial_q_arm_rad': np.asarray(seed, dtype=float).tolist(),
        'stop_reason': stop_reason, 'accepted_updates': accepted_updates, 'attempts': len(attempts),
        'strict_converged': termination_checked_strict,
        'termination_checked_strict': termination_checked_strict,
        'strict_threshold_satisfied': strict_satisfied, 'task_tolerance_pass': task_pass,
        'task_tolerance_scope': 'Final position/orientation residual only; excludes collision/path/dynamics/grasp.',
        'clip_candidate_count': clip_candidate_count,
        'candidate_count': sum(len(a['candidates']) for a in attempts), **final,
    }
    detail = {'trial_id': trial_id, 'target_id': target_id, 'seed_id': seed_id,
              'states': states, 'attempts': attempts, 'final': summary}
    return q, summary, rows, detail


def load_frozen(np, mujoco):
    """Verify immutable inputs and load meshes through the in-memory VFS."""
    if mujoco.__version__ != '3.3.7':
        raise RuntimeError('Expected audited MuJoCo 3.3.7')
    if sha(SOURCE_RUN / 'summary.json') != SOURCE_SUMMARY_SHA:
        raise RuntimeError('Frozen task-pose run summary changed')
    parent_summary = json.loads((SOURCE_RUN / 'summary.json').read_text(encoding='utf-8-sig'))
    snapshots = {item['path'].replace('\\', '/'): item['sha256'] for item in parent_summary['snapshot_files']}
    for name, digest in SOURCE_HASHES.items():
        if sha(SOURCE / name) != digest or snapshots.get('source/' + name) != digest:
            raise RuntimeError('Frozen source hash or run binding mismatch: ' + name)
    cfg = json.loads((SOURCE / 'config.json').read_text(encoding='utf-8-sig'))
    audit = json.loads((SOURCE / 'static_audit.json').read_text(encoding='utf-8-sig'))
    if cfg['scene_sha256'] != SOURCE_HASHES['scene.xml'] or audit['scene_sha256'] != cfg['scene_sha256']:
        raise RuntimeError('Scene/config/audit binding mismatch')
    if cfg['audit_sha256'] != SOURCE_HASHES['static_audit.json']:
        raise RuntimeError('Config does not bind frozen audit')
    if cfg['arm_joint_order'] != JOINT_ORDER or cfg['layout_accepted_for_control'] is not False:
        raise RuntimeError('Unexpected joint order or source acceptance status')
    if sha(LEGACY_PATH) != LEGACY_SHA:
        raise RuntimeError('Legacy solver reference changed; review before comparing')
    manifest = ROOT / 'repos/g1_grasp_assets/asset_manifest.json'
    if sha(manifest) != cfg['asset_manifest_sha256']:
        raise RuntimeError('Asset manifest changed')
    assets = {}
    asset_root = (ROOT / 'repos/g1_grasp_assets/assets').resolve()
    for item in cfg['mesh_files']:
        path = (ROOT / item['path']).resolve()
        if not path.is_relative_to(asset_root) or sha(path) != item['sha256']:
            raise RuntimeError('Mesh path/hash mismatch: ' + item['path'])
        assets[path.relative_to(asset_root).as_posix()] = path.read_bytes()
    model = mujoco.MjModel.from_xml_string((SOURCE / 'scene.xml').read_text(encoding='utf-8'), assets=assets)
    data = mujoco.MjData(model)
    joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_ORDER]
    if min(joints) < 0 or not all(model.jnt_limited[j] for j in joints):
        raise RuntimeError('Expected seven physically limited arm joints')
    qa = [int(model.jnt_qposadr[j]) for j in joints]
    va = [int(model.jnt_dofadr[j]) for j in joints]
    limits = model.jnt_range[joints].copy()
    initial = np.asarray(cfg['preview_qpos'], dtype=float)
    if initial.shape != (model.nq,) or not np.isfinite(initial).all():
        raise RuntimeError('Malformed frozen initial pose')
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, cfg['site'])
    if site < 0:
        raise RuntimeError('Missing grasp site')
    target_quat = np.asarray(cfg['constraints']['hand_orientation_wxyz'])

    def evaluate(q):
        mujoco.mj_resetData(model, data)
        data.qpos[:] = initial
        data.qpos[qa] = q
        mujoco.mj_forward(model, data)
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jp, jr, site)
        if data.time != 0:
            raise RuntimeError('Static FK must not advance simulation time')
        return data.site_xpos[site].copy(), data.site_xmat[site].reshape(3, 3).copy(), jp[:, va], jr[:, va]

    def rot_error(rotation):
        quat, inv, err, vector = np.empty(4), np.empty(4), np.empty(4), np.empty(3)
        mujoco.mju_mat2Quat(quat, rotation.ravel())
        mujoco.mju_negQuat(inv, quat)
        mujoco.mju_mulQuat(err, target_quat, inv)
        mujoco.mju_quat2Vel(vector, err, 1.)
        return vector

    points = {letter: next(p for p in cfg['task_points'] if p['id'] == name)
              for letter, name in POINT_IDS.items()}
    known = []
    # Only A/C are evaluated by inspection. B/D/E are targets in the user-run
    # experiment; their six independent numerical solves are not preparation QA.
    for letter in ('A', 'C'):
        point = points[letter]
        q = np.asarray(point['q_arm_rad'], dtype=float)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise RuntimeError('Invalid known seed ' + letter)
        p, rotation, _, _ = evaluate(q)
        bounds = boundary_report(np, q, limits, JOINT_ORDER)
        error = float(np.linalg.norm(p - point['position_world_m']))
        rerror = float(np.linalg.norm(rot_error(rotation)))
        if not bounds['within_solver_limits'] or error >= 2e-5 or rerror >= 2e-4:
            raise RuntimeError('Known seed no longer passes its recorded pose check: ' + letter)
        if not np.allclose(p, point['fk_position_world_m'], rtol=0, atol=1e-9):
            raise RuntimeError('Known seed FK differs from frozen result: ' + letter)
        known.append({'seed_id': letter, 'q_arm_rad': q.tolist(), 'fk_position_world_m': p.tolist(),
                      'position_error_mm': error * 1000, 'orientation_error_rad': rerror, **bounds})
    metadata = {
        'source_run': str(SOURCE_RUN), 'source_directory': str(SOURCE),
        'source_run_summary_sha256': SOURCE_SUMMARY_SHA, 'source_files_sha256': SOURCE_HASHES,
        'scene_sha256': cfg['scene_sha256'], 'config_sha256': SOURCE_HASHES['config.json'],
        'asset_manifest_sha256': cfg['asset_manifest_sha256'], 'mesh_files': cfg['mesh_files'],
        'legacy_solver_reference': str(LEGACY_PATH), 'legacy_solver_sha256': LEGACY_SHA,
        'joint_order': JOINT_ORDER, 'physical_joint_limits_rad': limits.tolist(),
        'solver_joint_limits_rad': (limits + np.array([.03, -.03])).tolist(),
        'qpos_addresses': qa, 'dof_addresses': va, 'site': cfg['site'],
        'fixed_orientation_wxyz': target_quat.tolist(), 'known_seed_checks': known,
        'nq': model.nq, 'nv': model.nv, 'nu': model.nu,
    }
    return cfg, points, model, data, limits, evaluate, rot_error, metadata


def contact_report(mujoco, model, data):
    result = []
    for contact in data.contact[:data.ncon]:
        geoms = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g))
                 for g in (contact.geom1, contact.geom2)]
        bodies = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[int(g)]))
                  for g in (contact.geom1, contact.geom2)]
        expected = set(geoms) == {'task_cube_geom', 'task_table_top'}
        result.append({'geoms': geoms, 'bodies': bodies, 'distance_m': float(contact.dist),
                       'expected_cube_table_contact': expected,
                       'unintended_penetration': bool(not expected and contact.dist <= 0)})
    return result


def freeze_run(out):
    copies = []
    entries = [(SOURCE / name, 'source/' + name) for name in SOURCE_HASHES]
    entries.append((SOURCE_RUN / 'summary.json', 'source/task_pose_run_summary.json'))
    entries.extend((ROOT / rel, 'code/' + rel) for rel in [
        'scripts/grasp/ik_seed_compare.py', 'scripts/grasp/lesson_common.py',
        'scripts/grasp/prepare_tray_shift.py', 'scripts/grasp/pickplace_layout_preview.py',
        'scripts/grasp/Run-IKSeedCompare.ps1', 'scripts/grasp/Show-IKSeedCompare.ps1',
        'scripts/启用E盘环境.ps1', 'scripts/grasp/IK初值对照带练.md'])
    entries.append((ROOT / 'repos/g1_grasp_assets/asset_manifest.json', 'source/asset_manifest.json'))
    for source, relative in entries:
        if not source.is_file():
            raise RuntimeError('Required reproducibility source missing: ' + str(source))
        destination = out / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        digest = sha(source)
        if sha(destination) != digest:
            raise RuntimeError('Snapshot copy mismatch: ' + str(source))
        copies.append({'path': relative, 'sha256': digest, 'original': str(source)})
    return copies


def save_progress(out, summary, rows, details):
    trace_path = out / 'trace.csv'
    with trace_path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    save_json(out / 'trace.json', {
        'kind': KIND, 'trace_kind': TRACE_KIND, 'solver_settings': SOLVER_SETTINGS,
        'iteration_semantics': 'Initial state 0; each accepted update increments by 1. Final state is retained once.',
        'attempt_index_semantics': 'Zero-based attempted outer-loop iteration; may stop without accepting an update.',
        'trials': details,
    })
    summary['trace_sha256'] = sha(trace_path)
    summary['trace_json_sha256'] = sha(out / 'trace.json')
    save_json(out / 'summary.json', summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inspect', action='store_true', help='Verify source and known A/C FK; do not solve B/D/E.')
    parser.add_argument('--output', type=Path, help='New directory within this project outputs, for a user-run experiment.')
    args = parser.parse_args()
    guard()
    if args.inspect and args.output is not None:
        parser.error('--inspect is read-only; do not specify --output')
    if args.inspect:
        import numpy as np
        import mujoco
        _, _, _, _, _, _, _, metadata = load_frozen(np, mujoco)
        runtime = {'python_executable': sys.executable, 'numpy_version': np.__version__,
                   'mujoco_version': mujoco.__version__, 'temp': os.environ['TEMP'],
                   'script_sha256': sha(__file__), 'lesson_common_sha256': sha(Path(__file__).with_name('lesson_common.py'))}
        print(json.dumps({'status': 'inspection_passed', 'kind': 'ik_seed_source_inspection',
                          'physics_steps': 0, 'simulation_s': 0., 'real_target_solve_calls': 0,
                          'trials': [], 'metadata': metadata, 'runtime': runtime}, ensure_ascii=False, indent=2))
        return 0
    out = (args.output if args.output is not None else ROOT / 'outputs' / ('ik_seed_user_' + stamp())).resolve()
    allowed = (ROOT / 'outputs').resolve()
    if out == allowed or not out.is_relative_to(allowed) or out.drive.casefold() != 'e:':
        raise RuntimeError('Output must be a new directory within project E-drive outputs')
    out.mkdir(parents=True, exist_ok=False)
    os.chdir(out)
    started = time.perf_counter()
    summary = {'status': 'running', 'kind': KIND, 'trace_kind': TRACE_KIND,
               'physics_steps': 0, 'simulation_s': 0., 'real_target_solve_calls': 0,
               'trials': [], 'solver_settings': SOLVER_SETTINGS,
               'runtime': {'python_executable': sys.executable, 'python_version': sys.version,
                           'temp': os.environ['TEMP'], 'script_sha256': sha(__file__),
                           'lesson_common_sha256': sha(Path(__file__).with_name('lesson_common.py'))},
               'layout_accepted_for_control': False, 'grasp_validated': False,
               'scope': 'Six independent local IK trials at frozen B/D/E targets from frozen A/C arm seeds.',
               'limitations': ['Local numerical convergence is not proof of global reachability or unreachability.',
                               'No collision avoidance cost; final contacts are reported separately.',
                               'No interpolated path, contact dynamics, PD tracking, or grasp execution is validated.',
                               'The cube, fingers and all non-arm coordinates remain at the frozen initial pose.'],
               'trace_columns': CSV_FIELDS}
    rows, details = [], []
    try:
        save_progress(out, summary, rows, details)
        import numpy as np
        import mujoco
        summary['runtime'].update(numpy_version=np.__version__, mujoco_version=mujoco.__version__,
                                  numpy_file=np.__file__, mujoco_file=mujoco.__file__)
        cfg, points, model, data, limits, evaluate, rot_error, metadata = load_frozen(np, mujoco)
        summary['metadata'] = metadata
        summary['snapshot_files'] = freeze_run(out)
        save_progress(out, summary, rows, details)
        for target_id in ('B', 'D', 'E'):
            for seed_id in ('A', 'C'):
                summary['active_trial'] = target_id + '_from_' + seed_id
                summary['real_target_solve_calls'] += 1
                q, trial, new_rows, detail = solve_trial(
                    np, evaluate, rot_error, limits, JOINT_ORDER,
                    points[target_id]['position_world_m'], points[seed_id]['q_arm_rad'], target_id, seed_id)
                # solve_trial already re-evaluates returned q. Verify that the
                # shared MjData agrees before taking contacts from it.
                qa = metadata['qpos_addresses']
                if not np.array_equal(data.qpos[qa], q) or data.time != 0:
                    raise RuntimeError('Returned-q / final MjData mismatch')
                contacts = contact_report(mujoco, model, data)
                trial.update(contacts=contacts, unintended_contacts=[c for c in contacts if c['unintended_penetration']],
                             simulation_s=float(data.time), physics_steps=0,
                             warning_counts=[int(w.number) for w in data.warning],
                             final_qpos=data.qpos.tolist(), final_qvel=data.qvel.tolist(),
                             grasp_validated=False, layout_accepted_for_control=False)
                detail['final'] = trial
                summary['trials'].append(trial)
                rows.extend(new_rows)
                details.append(detail)
                save_progress(out, summary, rows, details)
                print(f"{trial['trial_id']}: {trial['stop_reason']}; accepted={trial['accepted_updates']}; "
                      f"position={trial['position_error_mm']:.6f} mm; orientation={trial['orientation_error_rad']:.6f} rad; "
                      f"task_tol={trial['task_tolerance_pass']}", flush=True)
        summary.pop('active_trial', None)
        summary['status'] = 'completed'
        summary['diagnostic_completed'] = True
        summary['task_completed'] = False
        summary['source_hashes_unchanged'] = all(sha(SOURCE / n) == digest for n, digest in SOURCE_HASHES.items())
        if not summary['source_hashes_unchanged']:
            raise RuntimeError('Frozen sources changed during experiment')
        return_code = 0
    except BaseException as exc:
        summary.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                       diagnostic_completed=False, task_completed=False,
                       error_type=type(exc).__name__, error=str(exc))
        (out / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        return_code = 1
    finally:
        summary['wall_s'] = time.perf_counter() - started
        save_progress(out, summary, rows, details)
        print('OUTPUT:', out, flush=True)
        print('STATUS:', summary['status'], '; physics_steps=0; task_completed=false', flush=True)
    return return_code


if __name__ == '__main__':
    raise SystemExit(main())
