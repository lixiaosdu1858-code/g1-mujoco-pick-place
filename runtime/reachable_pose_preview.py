"""Inspect three FK-defined poses and their derived layout; no IK or dynamics."""
import argparse
import json
import os
from pathlib import Path
import queue
import shutil
import sys
import tempfile
import time
import traceback

from lesson_common import ROOT, guard, save_json, sha, stamp
from build_shoulder_elbow_scene import checked_assets
from pickplace_layout_preview import robot_opacity, set_camera, write_png

INDEX = Path(__file__).with_name('reachable_preview_source.json')
POSE_KEYS = {ord('4'): 'A', ord('5'): 'B', ord('6'): 'C'}
CAMERA_KEYS = {ord('1'): 'oblique', ord('2'): 'top', ord('3'): 'side'}
ROLES = {'A': 'initial', 'B': 'grasp', 'C': 'release'}


def read_frozen_source():
    index = json.loads(INDEX.read_text(encoding='utf-8-sig'))
    folder = (ROOT / index['source_directory']).resolve()
    if folder.parent != (ROOT / 'outputs').resolve():
        raise ValueError('Candidate source must be an approved project output.')
    for name in ('scene.xml', 'config.json', 'static_audit.json'):
        if sha(folder / name) != index['sha256'][name]:
            raise ValueError('Candidate source changed: ' + name)
    cfg = json.loads((folder / 'config.json').read_text(encoding='utf-8-sig'))
    audit = json.loads((folder / 'static_audit.json').read_text(encoding='utf-8-sig'))
    if not cfg.get('static_preview_checks_passed'):
        raise ValueError('Candidate layout is not ready even for static preview.')
    if cfg['scene_sha256'] != sha(folder / 'scene.xml'):
        raise ValueError('Configuration and scene hash disagree.')
    if audit.get('config_sha256') != sha(folder / 'config.json') or audit.get('scene_sha256') != cfg['scene_sha256']:
        raise ValueError('Preparation audit does not bind this configuration and scene.')
    points = {p['id']: p for p in cfg['poses']}
    if set(points) != set(ROLES):
        raise ValueError('Expected exactly the new A/B/C roles.')
    for key, role in ROLES.items():
        if points[key]['role'] != role:
            raise ValueError('New ABC roles do not match source.')
    return folder, cfg, audit, points


def model_context(folder, cfg, mujoco, np):
    manifest, meshes, assets = checked_assets()
    if sha(manifest) != cfg['asset_manifest_sha256'] or meshes != cfg['mesh_files']:
        raise ValueError('Source assets differ from frozen candidate design.')
    model = mujoco.MjModel.from_xml_string((folder / 'scene.xml').read_text(encoding='utf-8'), assets=assets)
    if (model.nq, model.nv, model.nu) != (16, 15, 9):
        raise ValueError('Unexpected single-arm, two-finger, free-cube model.')
    arm_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in cfg['arm_joint_order']]
    finger_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in cfg['finger_joint_order']]
    cube_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'task_cube_free')
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, cfg['site'])
    if min(arm_ids + finger_ids + [cube_joint, site]) < 0:
        raise ValueError('A required model element is missing.')
    base_qpos = np.asarray(cfg['preview_qpos'], dtype=float)
    if base_qpos.shape != (model.nq,) or not np.isfinite(base_qpos).all():
        raise ValueError('Invalid base configuration.')
    return {'model': model, 'data': mujoco.MjData(model), 'site': site,
            'arm_ids': arm_ids, 'arm_adr': model.jnt_qposadr[arm_ids].copy(),
            'finger_adr': model.jnt_qposadr[finger_ids].copy(),
            'cube_adr': int(model.jnt_qposadr[cube_joint]), 'base_qpos': base_qpos.copy()}


def apply_and_check(context, point, mujoco, np):
    model, data = context['model'], context['data']
    q = np.asarray(point['q_arm_rad'], dtype=float)
    limits = model.jnt_range[context['arm_ids']]
    if q.shape != (7,) or not np.isfinite(q).all() or np.any(q < limits[:, 0]) or np.any(q > limits[:, 1]):
        raise ValueError('Invalid candidate joint angles.')
    mujoco.mj_resetData(model, data)
    data.qpos[:] = context['base_qpos']
    data.qpos[context['arm_adr']] = q
    data.qpos[context['finger_adr']] = .0245
    mujoco.mj_forward(model, data)
    pos = data.site_xpos[context['site']].copy()
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, data.site_xmat[context['site']])
    source_quat = np.asarray(point['orientation_wxyz'])
    same_orientation = min(np.linalg.norm(quat-source_quat), np.linalg.norm(quat+source_quat))
    residual = float(np.linalg.norm(pos-np.asarray(point['position_world_m'])))
    if residual > 1e-8 or same_orientation > 1e-8:
        raise ValueError('FK no longer matches the saved candidate pose.')
    cube_adr = context['cube_adr']
    if not np.array_equal(data.qpos[cube_adr:cube_adr+7], context['base_qpos'][cube_adr:cube_adr+7]):
        raise ValueError('The cube moved during a static pose switch.')
    contacts, unintended = [], []
    for contact in data.contact[:data.ncon]:
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) for g in (contact.geom1, contact.geom2)]
        expected = set(names) == {'task_cube_geom', 'task_table_top'}
        row = {'geoms': names, 'distance_m': float(contact.dist), 'expected_cube_table': expected}
        contacts.append(row)
        if not expected and contact.dist <= 0:
            unintended.append(row)
    if unintended or data.time != 0 or any(w.number for w in data.warning):
        raise ValueError('Unexpected contact, warning or time advance in static preview.')
    margin = float(np.min(np.minimum(q-limits[:, 0], limits[:, 1]-q)))
    return {'id': point['id'], 'role': point['role'], 'q_arm_rad': q.tolist(),
            'position_world_m': pos.tolist(), 'orientation_wxyz': quat.tolist(),
            'source_fk_recheck_m': residual, 'min_model_margin_rad': margin,
            'contacts': contacts, 'cube_qpos': data.qpos[cube_adr:cube_adr+7].tolist(),
            'simulation_s': float(data.time), 'IK_recovery_tested': False}


def markers(scene, points, selected, mujoco, np):
    colors = {'A': [0.2, .65, 1, .7], 'B': [1, .7, .1, .7], 'C': [.75, .3, 1, .7]}
    for key, point in points.items():
        pos = np.asarray(point['position_world_m'])
        for kind in ('sphere', 'label'):
            if scene.ngeom >= scene.maxgeom:
                raise RuntimeError('Preview marker capacity exceeded.')
            geom = scene.geoms[scene.ngeom]
            scene.ngeom += 1
            label = kind == 'label'
            loc = pos + np.array([0, -.025, .022]) if label else pos
            radius = .006 if key == selected else .004
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LABEL if label else mujoco.mjtGeom.mjGEOM_SPHERE,
                               np.zeros(3) if label else np.array([radius, 0, 0]), loc,
                               np.eye(3).ravel(), np.array(colors[key], dtype=np.float32))
            if label:
                geom.label = key + ': ' + ROLES[key]


def overlay(row, np):
    x, y, z = row['position_world_m']
    return ('ABC POSE PREVIEW | STATIC FK\n'
            '4:A initial   5:B grasp   6:C release\n'
            '1:angle  2:top  3:side  H:hide body  Q:finish\n'
            f"Selected {row['id']} | XYZ (m): {x:.3f}, {y:.3f}, {z:.3f}\n"
            f"Joint-limit margin: {np.degrees(row['min_model_margin_rad']):.1f} deg\n"
            'Pose switches are instant. Cube stays on table.\n'
            'Path / IK recovery / PD / grasp: not validated.')


def render(context, cfg, points, out, mujoco, np):
    model, data = context['model'], context['data']
    model.vis.global_.offwidth, model.vis.global_.offheight = 1200, 900
    option = mujoco.MjvOption()
    option.label = mujoco.mjtLabel.mjLABEL_NONE
    images = []
    with mujoco.Renderer(model, height=900, width=1200) as renderer:
        for key in ROLES:
            apply_and_check(context, points[key], mujoco, np)
            camera = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(camera)
            set_camera(camera, cfg, 'oblique')
            renderer.update_scene(data, camera=camera, scene_option=option)
            markers(renderer.scene, points, key, mujoco, np)
            path = out / (key + '_static_pose.png')
            write_png(path, renderer.render())
            images.append({'path': path.name, 'sha256': sha(path), 'pose': key, 'static_only': True})
    return images


def show(context, cfg, points, rows, report, mujoco, np):
    import mujoco.viewer
    model, data = context['model'], context['data']
    commands = queue.SimpleQueue()
    selected, camera_mode, hidden = 'A', 'oblique', False
    apply_and_check(context, points[selected], mujoco, np)
    rgba = model.geom_rgba.copy()
    start = time.perf_counter()
    reason = 'window_closed'
    with mujoco.viewer.launch_passive(model, data, key_callback=commands.put,
                                     show_left_ui=False, show_right_ui=False) as viewer:
        report['viewer_started'] = True
        with viewer.lock():
            set_camera(viewer.cam, cfg, camera_mode)
            viewer.opt.label = mujoco.mjtLabel.mjLABEL_NONE
        while viewer.is_running():
            now = time.perf_counter()
            if now-start >= 600:
                reason = 'preview_wall_time_limit'
                break
            stop = False
            with viewer.lock():
                while not commands.empty():
                    key = commands.get()
                    if key == ord('Q'):
                        reason, stop = 'user_Q', True
                    elif key in POSE_KEYS:
                        selected = POSE_KEYS[key]
                        apply_and_check(context, points[selected], mujoco, np)
                        print('POSE:', selected, ROLES[selected], flush=True)
                    elif key in CAMERA_KEYS:
                        camera_mode = CAMERA_KEYS[key]
                        set_camera(viewer.cam, cfg, camera_mode)
                    elif key == ord('H'):
                        hidden = not hidden
                        robot_opacity(model, mujoco, rgba, hidden)
                    else:
                        continue
                    report['events'].append({'wall_s': now-start, 'key': chr(key), 'pose': selected,
                                             'camera': camera_mode, 'robot_hidden': hidden})
                    if stop:
                        break
                if stop:
                    break
                # Re-assert the displayed state after any viewer edits; never integrate.
                apply_and_check(context, points[selected], mujoco, np)
                viewer.user_scn.ngeom = 0
                markers(viewer.user_scn, points, selected, mujoco, np)
            viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                              overlay(rows[selected], np), ''))
            viewer.sync()
            time.sleep(1/30)
    report.update(viewer_end_reason=reason, final_pose=selected, preview_wall_s=time.perf_counter()-start)


def main():
    guard()
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--inspect', action='store_true')
    modes.add_argument('--render-only', action='store_true')
    args = parser.parse_args()
    prefix = 'reachable_pose_toolcheck_' if args.render_only else 'reachable_pose_user_'
    out = args.output.resolve() if args.output else ROOT / 'outputs' / (prefix+stamp())
    if out.parent != (ROOT/'outputs').resolve() or not out.name.startswith(prefix):
        raise ValueError('Use a new approved E-drive project output directory.')
    if not args.inspect:
        out.mkdir(exist_ok=False)
    os.chdir(ROOT / 'outputs' if args.inspect else out)
    report = {'status': 'started', 'kind': 'FK_defined_ABC_static_preview', 'events': [],
              'physics_steps': 0, 'simulation_s': 0, 'IK_solve_calls': 0,
              'model_loaded': False, 'viewer_started': False, 'learner_verified': False,
              'path_validated': False, 'PD_validated': False, 'grasp_validated': False,
              'roles': ROLES, 'snapshots': [], 'executable': sys.executable, 'temp': tempfile.gettempdir()}
    start = time.perf_counter()
    code = 0
    try:
        import numpy as np
        import mujoco
        if mujoco.__version__ != '3.3.7':
            raise ValueError('Expected registered MuJoCo 3.3.7.')
        folder, cfg, audit, points = read_frozen_source()
        report['source_directory'] = str(folder)
        if not args.inspect:
            sources = [(folder/name, 'source/'+name) for name in ('scene.xml','config.json','static_audit.json')]
            sources += [(Path(__file__).with_name(name), 'code/'+name) for name in (
                'reachable_pose_preview.py','reachable_pose_design.py','reachable_preview_source.json',
                'Run-ReachablePoses.ps1','lesson_common.py','pickplace_layout_preview.py',
                'build_shoulder_elbow_scene.py','build_elbow_scene.py','build_pickplace_preview.py')]
            sources += [(ROOT/'repos/g1_grasp_assets/asset_manifest.json', 'source/asset_manifest.json')]
            for source, rel in sources:
                dest = out/rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)
                report['snapshots'].append({'path': rel, 'sha256': sha(dest)})
        context = model_context(folder, cfg, mujoco, np)
        report['model_loaded'] = True
        rows = {key: apply_and_check(context, point, mujoco, np) for key, point in points.items()}
        report.update(poses=rows, static_fk_recheck_passed=True,
                      geometry={k: cfg[k] for k in ('table', 'cube', 'tray', 'proposed_grasp_relation', 'proposed_release_gap_m')},
                      source_audit_sha256=sha(folder/'static_audit.json'), mujoco_version=mujoco.__version__)
        for key, row in rows.items():
            print(f"{key} ({row['role']}): XYZ (m)={row['position_world_m']}; min joint margin={np.degrees(row['min_model_margin_rad']):.2f} deg")
        if args.inspect:
            report['status'] = 'inspection_passed'
        else:
            report['images'] = render(context, cfg, points, out, mujoco, np)
            if not args.render_only:
                print('Click scene: 4=A, 5=B, 6=C. Camera 1/2/3. Q=finish.', flush=True)
                show(context, cfg, points, rows, report, mujoco, np)
            report['status'] = 'static_render_checked' if args.render_only else 'preview_ended'
    except KeyboardInterrupt:
        report.update(status='interrupted', error='KeyboardInterrupt')
        code = 130
    except Exception as error:
        report.update(status='error', error=str(error), traceback=traceback.format_exc())
        print(report['traceback'], file=sys.stderr)
        code = 1
    finally:
        report['wall_s'] = time.perf_counter()-start
        if args.inspect:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            save_json(out/'summary.json', report)
            print('SUMMARY:', out/'summary.json', flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
