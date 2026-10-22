"""v0.1 task-layout preview. Static FK only; never steps physics or moves a robot."""
import argparse
import json
import os
from pathlib import Path
import queue
import shutil
import struct
import sys
import tempfile
import time
import traceback
import zlib

from lesson_common import ROOT, guard, save_json, sha, stamp


def load_scene(mujoco, np, folder=None):
    folder = Path(folder) if folder is not None else ROOT / 'models/g1_pickplace_preview'
    cfg_path, scene_path = folder / 'config.json', folder / 'scene.xml'
    cfg = json.loads(cfg_path.read_text(encoding='utf-8-sig'))
    if not cfg.get('preview_checks_passed'):
        raise RuntimeError('Even the displayed static pose has not passed preview checks.')
    audit_path = (ROOT / cfg['audit_path']).resolve()
    if not audit_path.is_relative_to((ROOT / 'outputs').resolve()) or sha(audit_path) != cfg['audit_sha256']:
        raise RuntimeError('Static audit path/hash mismatch.')
    manifest = ROOT / 'repos/g1_grasp_assets/asset_manifest.json'
    if sha(scene_path) != cfg['scene_sha256']:
        raise RuntimeError('Preview scene changed since its static audit.')
    if sha(manifest) != cfg['asset_manifest_sha256']:
        raise RuntimeError('Asset manifest changed since preparation.')
    assets = {}
    asset_root = (ROOT / 'repos/g1_grasp_assets/assets').resolve()
    for item in cfg['mesh_files']:
        path = (ROOT / item['path']).resolve()
        if not path.is_relative_to(asset_root) or sha(path) != item['sha256']:
            raise RuntimeError('Mesh path/hash mismatch: ' + item['path'])
        assets[path.relative_to(asset_root).as_posix()] = path.read_bytes()
    model = mujoco.MjModel.from_xml_string(scene_path.read_text(encoding='utf-8'), assets=assets)
    data = mujoco.MjData(model)
    pose = np.array(cfg['preview_qpos'], dtype=float)
    if pose.shape != (model.nq,) or not np.isfinite(pose).all():
        raise RuntimeError('Invalid preview pose.')
    data.qpos[:] = pose
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    for point in cfg['task_points']:
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'task_' + point['id'])
        if site < 0:
            raise RuntimeError('Missing task marker: ' + point['id'])
        passed = point['position_residual_m'] < 2e-5 and point['orientation_residual_rad'] < 2e-4 and not point['unintended_contacts']
        model.site_rgba[site] = [.15, .85, .25, 1.] if passed else [1., .12, .1, 1.]
    return cfg, cfg_path, scene_path, manifest, model, data, pose


def add_labels(scene, cfg, mujoco, np, mode):
    """Label only task waypoints, not every route/IMU/hidden model site."""
    points = cfg['task_points']
    if mode == 'top':
        labels = [('A/B/C', np.array(points[0]['position_world_m']) + [.055, .025, .02]),
                  ('D/E', np.array(points[3]['position_world_m']) + [.055, -.02, .02])]
    else:
        offsets = [[.035,.03,0], [.035,.02,-.015], [0,0,.035], [0,0,.035], [-.03,-.03,-.005]]
        labels = [(point['label'][0], np.array(point['position_world_m']) + offsets[i]) for i, point in enumerate(points)]
    for label, position in labels:
        if scene.ngeom >= scene.maxgeom:
            raise RuntimeError('No room for task labels.')
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LABEL, np.zeros(3), position,
                           np.eye(3).ravel(), np.array([1, 1, 1, 1], dtype=np.float32))
        geom.label = label
        scene.ngeom += 1


def robot_opacity(model, mujoco, original, ghost):
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')
    for geom in range(model.ngeom):
        body = int(model.geom_bodyid[geom])
        while body and body != pelvis:
            body = int(model.body_parentid[body])
        if body == pelvis and pelvis > 0:
            model.geom_rgba[geom, 3] = original[geom, 3] * (0. if ghost else 1.)


def set_camera(cam, cfg, mode):
    # Frame the right-hand work area; full robot remains inspectable by mouse.
    points = cfg['task_points']
    x = sum(p['position_world_m'][0] for p in points) / len(points)
    y = sum(p['position_world_m'][1] for p in points) / len(points)
    z = sum(p['position_world_m'][2] for p in points) / len(points)
    cam.lookat[:] = [x - 0.06, y + 0.06, z - 0.06]
    cam.distance = 1.18
    if mode == 'oblique':
        cam.azimuth, cam.elevation = 135, -28
    elif mode == 'top':
        cam.azimuth, cam.elevation, cam.distance = 90, -89, 1.05
    elif mode == 'side':
        cam.azimuth, cam.elevation = -90, -8
    else:
        raise ValueError('Unknown camera mode')


def write_png(path, pixels):
    """Save renderer output with stdlib, without new image dependencies."""
    height, width, channels = pixels.shape
    if channels != 3 or pixels.dtype.name != 'uint8':
        raise ValueError('Expected RGB uint8 renderer output.')
    def chunk(kind, data):
        return struct.pack('!I', len(data)) + kind + data + struct.pack('!I', zlib.crc32(kind + data) & 0xffffffff)
    raw = b''.join(b'\0' + row.tobytes() for row in pixels)
    header = struct.pack('!IIBBBBB', width, height, 8, 2, 0, 0, 0)
    path.write_bytes(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', header) +
                     chunk(b'IDAT', zlib.compress(raw, 6)) + chunk(b'IEND', b''))


def render_views(model, data, cfg, out, mujoco, np):
    # Rendering a fixed state is a tool check, not a physics/control experiment.
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, 1200)
    model.vis.global_.offheight = max(model.vis.global_.offheight, 900)
    option = mujoco.MjvOption()
    option.label = mujoco.mjtLabel.mjLABEL_NONE
    files = []
    original_rgba = model.geom_rgba.copy()
    with mujoco.Renderer(model, height=900, width=1200) as renderer:
        for mode in ('oblique', 'top', 'side'):
            robot_opacity(model, mujoco, original_rgba, mode == 'top')
            cam = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(cam)
            set_camera(cam, cfg, mode)
            renderer.update_scene(data, camera=cam, scene_option=option)
            add_labels(renderer.scene, cfg, mujoco, np, mode)
            target = out / (mode + '_STATIC_PREVIEW.png')
            write_png(target, renderer.render())
            files.append({'path': str(target), 'sha256': sha(target), 'robot_hidden': mode == 'top',
                          'kind': 'static_layout_not_grasp_result'})
    robot_opacity(model, mujoco, original_rgba, False)
    return files


def show(model, data, pose, cfg, result, mujoco, np):
    import mujoco.viewer
    commands = queue.SimpleQueue()
    start, mode, reason, ghost = time.perf_counter(), 'oblique', 'window_closed', False
    original_rgba = model.geom_rgba.copy()
    events = result['camera_events']
    keys = {ord('1'): 'oblique', ord('2'): 'top', ord('3'): 'side'}
    with mujoco.viewer.launch_passive(model, data, key_callback=commands.put,
                                     show_left_ui=False, show_right_ui=False) as viewer:
        result['viewer_started'] = True
        with viewer.lock():
            set_camera(viewer.cam, cfg, mode)
            viewer.opt.label = mujoco.mjtLabel.mjLABEL_NONE
            viewer.user_scn.ngeom = 0
            add_labels(viewer.user_scn, cfg, mujoco, np, mode)
        while viewer.is_running():
            now = time.perf_counter()
            if now - start >= 300:
                reason = 'preview_wall_time_limit'
                break
            stop = False
            while not commands.empty():
                key = commands.get()
                if key == ord('Q'):
                    stop, reason = True, 'user_Q'
                    break
                if key in keys:
                    mode = keys[key]
                    with viewer.lock():
                        set_camera(viewer.cam, cfg, mode)
                        viewer.user_scn.ngeom = 0
                        add_labels(viewer.user_scn, cfg, mujoco, np, mode)
                    events.append({'wall_s': now - start, 'camera': mode})
                    print('CAMERA:', mode, flush=True)
                if key == ord('H'):
                    ghost = not ghost
                    with viewer.lock():
                        robot_opacity(model, mujoco, original_rgba, ghost)
                    events.append({'wall_s': now - start, 'robot_hidden': ghost})
            if stop:
                break
            # This is deliberately a frozen layout, with no mj_step or control law.
            data.qpos[:] = pose
            data.qvel[:] = 0
            data.ctrl[:] = 0
            data.time = 0
            data.qfrc_applied[:] = 0
            data.xfrc_applied[:] = 0
            mujoco.mj_forward(model, data)
            viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                              'V0.1 CANDIDATE LAYOUT | NOT READY FOR CONTROL\n'
                              '1:oblique  2:top  3:side  H:hide/show robot  Q:finish\n'
                              'Green: checked pose. Red: candidate needs revision.\n'
                              'World +X:robot front  +Y:robot left  +Z:up\n'
                              'Markers are planned hand targets. Cube stays on table.\n'
                              'No physics stepping / no grasp execution.', ''))
            viewer.sync()
            time.sleep(1 / 30)
    result.update(viewer_end_reason=reason, preview_wall_s=time.perf_counter() - start)


def main():
    guard()
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    parser.add_argument('--inspect', action='store_true')
    parser.add_argument('--render-only', action='store_true', help='Static screenshot QA; no interactive viewer.')
    args = parser.parse_args()
    prefix = 'pickplace_layout_toolcheck_' if args.render_only else 'pickplace_layout_user_'
    out = args.output.resolve() if args.output else ROOT / 'outputs' / (prefix + stamp())
    if out.parent != (ROOT / 'outputs').resolve() or not out.name.startswith(prefix):
        raise RuntimeError('Use a unique approved project outputs directory.')
    if not args.inspect:
        out.mkdir(exist_ok=False)
    os.chdir(ROOT / 'outputs' if args.inspect else out)
    result = dict(status='started', kind='static_task_layout_preview', physics_steps=0,
                  simulation_s=0, model_loaded=False, viewer_started=False, learner_verified=False, camera_events=[],
                  snapshot_files=[], missing_snapshot_sources=[],
                  executable=sys.executable, temp=tempfile.gettempdir(),
                  scope='Static scene and candidate waypoints only. No control, object transport, or grasp validation.',
                  output=str(out) if not args.inspect else None)
    start = time.perf_counter()
    try:
        if not args.inspect:
            # Retain readable inputs even when the subsequent model load fails.
            sources = (Path(__file__), Path(__file__).with_name('lesson_common.py'),
                       ROOT / 'models/g1_pickplace_preview/config.json',
                       ROOT / 'models/g1_pickplace_preview/scene.xml',
                       ROOT / 'models/g1_pickplace_preview/static_audit.json',
                       ROOT / 'repos/g1_grasp_assets/asset_manifest.json')
            for source in sources:
                if source.is_file():
                    shutil.copy2(source, out / source.name)
                    result['snapshot_files'].append({'name': source.name, 'sha256': sha(out / source.name)})
                else:
                    result['missing_snapshot_sources'].append(str(source))
        import mujoco
        import numpy as np
        cfg, cfg_path, scene, manifest, model, data, pose = load_scene(mujoco, np)
        result.update(model_loaded=True, mujoco_version=mujoco.__version__,
                      scene_sha256=sha(scene), config_sha256=sha(cfg_path), script_sha256=sha(__file__),
                      asset_manifest_sha256=sha(manifest), nq=int(model.nq), nv=int(model.nv),
                      initial_contact_count=int(data.ncon), task_points=cfg['task_points'],
                      preview_checks_passed=cfg['preview_checks_passed'],
                      full_layout_checks_passed=cfg['static_checks_passed'],
                      layout_accepted_for_control=cfg['layout_accepted_for_control'], known_issues=cfg['known_issues'],
                      display_semantics='Green=individual static IK/contact checks passed; red=candidate failed current checks. No dynamic execution.')
        print('STATIC LAYOUT: no dynamics, no grasp success claim.', flush=True)
        for point in cfg['task_points']:
            print(point['id'], point['label'], 'world xyz (m):', point['position_world_m'])
        if args.inspect:
            print('INSPECT PASSED; nq/nv:', model.nq, model.nv, flush=True)
            return
        print('OUTPUT:', out, flush=True)
        if args.render_only:
            result['images'] = render_views(model, data, cfg, out, mujoco, np)
            result['status'] = 'static_render_completed'
        else:
            print('Click scene. 1=oblique, 2=top, 3=side; H=hide/show robot; Q=finish. Robot/cube stay still.', flush=True)
            show(model, data, pose, cfg, result, mujoco, np)
            result['status'] = 'preview_ended'
        if data.time != 0:
            raise RuntimeError('Unexpected simulation time advance in static preview.')
    except BaseException:
        result.update(status='error', error=traceback.format_exc())
        if not args.inspect:
            (out / 'error.txt').write_text(result['error'], encoding='utf-8')
        raise
    finally:
        if not args.inspect:
            result['wall_s'] = time.perf_counter() - start
            save_json(out / 'summary.json', result)
            print('SUMMARY:', out / 'summary.json', flush=True)


if __name__ == '__main__':
    main()
