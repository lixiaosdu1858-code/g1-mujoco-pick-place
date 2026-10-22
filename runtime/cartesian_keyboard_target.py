"""Bounded, scratch-data Cartesian references for the first empty-hand lesson.

Commands are relative to the last accepted WORLD target near a frozen high
working posture. A command is
accepted atomically after every reference sample passes its static checks.
The live MjData is only read. This module never integrates or opens a viewer.
The runtime must reject keys while executing a segment, then supply its held
reference endpoint as q_start9; actual joint feedback is not a new reference.
"""
import itertools
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

from lesson_common import ROOT, guard, save_json, sha, stamp
from ik_bounded_step import solve_box_qp

DT = .004
SOURCE = ROOT/'configs/accepted_baseline.json'
SOURCE_SHA = '5f710296f86b1e9a489b293408678e63ee7ac39b2183a9c59f0c22485f3625fa'
OPEN = .0245
CLOSED = -.0065
ANCHOR = ROOT/'configs/keyboard_anchor.json'
ANCHOR_SHA = '8a79353a724c7173d6da7da3f9e14987988fb0c1097c4e425cf8a464bc6b4efb'
SETTINGS = dict(maximum_delta_m=.005, workspace_half_extent_m=.010,
    position_tolerance_m=.00005, orientation_tolerance_rad=.0005,
    path_orientation_tolerance_rad=.05, path_line_deviation_m=.001,
    arm_margin_rad=.03, minimum_environment_gap_m=.010,
    maximum_arm_reference_speed_rad_s=.5, maximum_finger_reference_speed_m_s=.035,
    maximum_static_gravity_to_cap=.7, motion_duration_s=1., finger_duration_s=2.,
    ik_iterations=20, qp_iterations=128, ik_step_bound_rad=.05,
    damping=1e-6, orientation_weight_m=.15, backtracking_scales=8)


def load_initial_pose(ctx, mujoco, np):
    """Read frozen derived posture, checking it in independent data only."""
    if sha(ANCHOR) != ANCHOR_SHA:
        raise ValueError('Frozen keyboard anchor changed')
    record = json.loads(ANCHOR.read_text(encoding='utf-8-sig'))
    if (record['source_summary_sha256'] != SOURCE_SHA or sha(SOURCE) != SOURCE_SHA
            or record['source_scene_sha256'] != sha(ctx['folder']/'scene.xml')
            or record['source_config_sha256'] != sha(ctx['folder']/'config.json')
            or record['source_reference_sha256'] != sha(ctx['reference_file'])
            or record['settings'] != SETTINGS):
        raise ValueError('Keyboard anchor provenance or settings changed')
    q = np.asarray(record['q_ref9'], dtype=float)
    if q.shape != (9,) or not np.isfinite(q).all() or not np.array_equal(q[7:], [OPEN, OPEN]):
        raise ValueError('Malformed keyboard initial joints')
    d = mujoco.MjData(ctx['model'])
    d.qpos[:] = ctx['base_qpos']
    d.qpos[ctx['qa']] = q
    mujoco.mj_forward(ctx['model'], d)
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, d.site_xmat[ctx['site']])
    angle = 2*np.arccos(np.clip(abs(float(quat@np.asarray(record['orientation_wxyz']))), 0., 1.))
    if (not np.allclose(d.site_xpos[ctx['site']], record['position_world_m'], atol=1e-10, rtol=0)
            or angle > SETTINGS['orientation_tolerance_rad']):
        raise ValueError('Frozen initial posture FK differs')
    return dict(record, path=str(ANCHOR), sha256=ANCHOR_SHA)


def prepare_initial_pose(ctx, mujoco, np):
    """Initialize the NEW keyboard lesson once, before any physics or viewer.

    This is an explicitly different starting pose, not a claim that the robot
    physically moved from A. No original file, model property or carry setup is
    changed. Reinitialization during an exercise is forbidden.
    """
    d = ctx['data']
    if (d.time != 0. or ctx.get('keyboard_initialization') is not None
            or not np.array_equal(d.qpos, ctx['base_qpos'])):
        raise ValueError('Keyboard initialization is allowed only once at fresh setup')
    record = load_initial_pose(ctx, mujoco, np)
    mujoco.mj_resetData(ctx['model'], d)
    d.qpos[:] = ctx['base_qpos']
    d.qpos[ctx['qa']] = record['q_ref9']
    mujoco.mj_forward(ctx['model'], d)
    ctx['keyboard_initialization'] = record
    ctx['keyboard_anchor'] = np.asarray(record['position_world_m']).copy()
    ctx['keyboard_orientation'] = np.asarray(record['orientation_wxyz']).copy()
    return record


class TargetPlanner:
    def __init__(self, ctx, mujoco, np):
        self.ctx, self.mj, self.np = ctx, mujoco, np
        self.model = ctx['model']
        self.scratch = mujoco.MjData(self.model)
        self.forward_calls = 0
        if sha(SOURCE) != SOURCE_SHA:
            raise ValueError('Accepted carry summary changed')
        accepted = json.loads(SOURCE.read_text(encoding='utf-8-sig'))
        if not all(accepted[k] for k in ('grasp_hold_passed', 'placement_passed', 'tracking_thresholds_passed')):
            raise ValueError('Expected successful physical baseline')
        snapshots = {item['path']: item['sha256'] for item in accepted['snapshots']}
        if (accepted['status'] != 'completed' or
                sha(ctx['folder']/'scene.xml') != snapshots['source/scene.xml'] or
                sha(ctx['folder']/'config.json') != snapshots['source/config.json'] or
                sha(ctx['reference_file']) != snapshots['source/path.json']):
            raise ValueError('Keyboard model/reference differs from accepted baseline')
        if abs(float(self.model.opt.timestep)-DT) > 1e-12:
            raise ValueError('Expected the unchanged 0.004 s model')
        record = load_initial_pose(ctx, mujoco, np)
        self.initial_q9 = np.asarray(record['q_ref9']).copy()
        self.last_q9 = self.initial_q9.copy()
        self._frozen_state = ctx['data'].qpos.copy()
        self._forward(self.initial_q9)
        self.origin_xyz = self.scratch.site_xpos[ctx['site']].copy()
        self.target_xyz = self.origin_xyz.copy()
        self.orientation_wxyz = np.asarray(record['orientation_wxyz']).copy()
        self.lo = ctx['limits'][:, 0]+SETTINGS['arm_margin_rad']
        self.hi = ctx['limits'][:, 1]-SETTINGS['arm_margin_rad']
        self.jp = np.zeros((3, self.model.nv))
        self.jr = np.zeros((3, self.model.nv))
        self.meshes = self._moving_geometry()
        self.obstacles = [g for g in range(self.model.ngeom)
            if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) or '')
                in ['task_table_top', 'task_cube_geom', 'tray_floor']+
                   ['tray_wall_'+str(i) for i in range(4)]+['table_leg_'+str(i) for i in range(4)]]
        if len(self.obstacles) != 11:
            raise ValueError('Unexpected task collision geometry')
        self.allowed_pair = {int(ctx['cube_geom']), int(ctx['table_geom'])}

    def _quaternion(self):
        q = self.np.empty(4)
        self.mj.mju_mat2Quat(q, self.scratch.site_xmat[self.ctx['site']])
        return q

    def _rotation_error(self):
        q = self._quaternion()
        inverse, error, vec = self.np.empty(4), self.np.empty(4), self.np.empty(3)
        self.mj.mju_negQuat(inverse, q)
        self.mj.mju_mulQuat(error, self.orientation_wxyz, inverse)
        if error[0] < 0:
            error *= -1
        self.mj.mju_quat2Vel(vec, error, 1.)
        return vec

    def _forward(self, q9):
        d = self.scratch
        d.qpos[:] = self._frozen_state
        d.qpos[self.ctx['qa']] = q9
        d.qvel[:] = 0.
        d.ctrl[:] = 0.
        d.qfrc_applied[:] = 0.
        d.xfrc_applied[:] = 0.
        self.mj.mj_forward(self.model, d)
        self.forward_calls += 1

    def _moving_geometry(self):
        """Convex enclosing vertices of every collidable right-arm shape."""
        m, mj, np = self.model, self.mj, self.np
        root_body = int(m.jnt_bodyid[self.ctx['arm_ids'][0]])
        signs = np.asarray(list(itertools.product((-1., 1.), repeat=3)))
        found = []
        for g in range(m.ngeom):
            if not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            body = int(m.geom_bodyid[g])
            while body not in (0, root_body):
                body = int(m.body_parentid[body])
            if body != root_body:
                continue
            kind, size = int(m.geom_type[g]), m.geom_size[g]
            if kind == int(mj.mjtGeom.mjGEOM_MESH):
                mesh = int(m.geom_dataid[g])
                start, count = int(m.mesh_vertadr[mesh]), int(m.mesh_vertnum[mesh])
                vertices = m.mesh_vert[start:start+count].astype(float).copy()
            elif kind == int(mj.mjtGeom.mjGEOM_BOX):
                vertices = signs*size
            elif kind == int(mj.mjtGeom.mjGEOM_CYLINDER):
                vertices = signs*np.array([size[0], size[0], size[1]])
            elif kind == int(mj.mjtGeom.mjGEOM_CAPSULE):
                vertices = signs*np.array([size[0], size[0], size[0]+size[1]])
            elif kind == int(mj.mjtGeom.mjGEOM_SPHERE):
                vertices = signs*size[0]
            else:
                raise ValueError('Unsupported arm collision shape')
            # A local box contains the complete compiled collision hull. Eight
            # corners suffice for conservative projected separation; never use
            # an inscribed subset of mesh vertices as a clearance certificate.
            low, high = vertices.min(axis=0), vertices.max(axis=0)
            enclosing = (low+high)/2+signs*(high-low)/2
            found.append((g, enclosing))
        if len(found) < 3:
            raise ValueError('Missing moving arm geometry')
        return found

    def _start(self, q_start9):
        np = self.np
        q = np.asarray(q_start9, dtype=float)
        if q.shape != (9,) or not np.isfinite(q).all():
            return None, 'invalid_start_reference'
        if not np.allclose(q, self.last_q9, atol=1e-10, rtol=0):
            return None, 'start_is_not_last_accepted_endpoint'
        # Current obstacle state is copied once per command; only scratch is set.
        self._frozen_state = self.ctx['data'].qpos.copy()
        self._forward(q)
        self.obstacle_position = self.scratch.geom_xpos[self.obstacles].copy()
        self.obstacle_rotation = self.scratch.geom_xmat[self.obstacles].reshape(-1, 3, 3).copy()
        self.obstacle_size = self.model.geom_size[self.obstacles].copy()
        return q.copy(), None

    def _solve(self, q9, target):
        np, mj, m = self.np, self.mj, self.model
        q = q9.copy()
        history = []
        for iteration in range(SETTINGS['ik_iterations']+1):
            self._forward(q)
            ep = target-self.scratch.site_xpos[self.ctx['site']]
            er = self._rotation_error()
            pos, ang = float(np.linalg.norm(ep)), float(np.linalg.norm(er))
            history.append(dict(iteration=iteration, position_error_m=pos, orientation_error_rad=ang))
            if pos <= SETTINGS['position_tolerance_m'] and ang <= SETTINGS['orientation_tolerance_rad']:
                return q, dict(status='converged', iterations=iteration, history=history)
            if iteration == SETTINGS['ik_iterations']:
                break
            mj.mj_jacSite(m, self.scratch, self.jp, self.jr, self.ctx['site'])
            weight = SETTINGS['orientation_weight_m']
            J = np.vstack((self.jp[:, self.ctx['va'][:7]], weight*self.jr[:, self.ctx['va'][:7]]))
            e = np.r_[ep, weight*er]
            H = J.T@J+SETTINGS['damping']*np.eye(7)
            bound = SETTINGS['ik_step_bound_rad']
            step, info = solve_box_qp(H, -J.T@e,
                np.maximum(self.lo-q[:7], -bound), np.minimum(self.hi-q[:7], bound),
                max_iterations=SETTINGS['qp_iterations'])
            if info['status'] != 'optimal' or not info['success']:
                return None, dict(status='qp_rejected', qp_status=info['status'], history=history)
            accepted = False
            for k in range(SETTINGS['backtracking_scales']):
                trial = q.copy()
                trial[:7] += (2.**-k)*step
                self._forward(trial)
                residual = np.r_[target-self.scratch.site_xpos[self.ctx['site']], weight*self._rotation_error()]
                if float(residual@residual) < float(e@e)-1e-18:
                    q, accepted = trial, True
                    history[-1]['accepted_scale'] = 2.**-k
                    break
            if not accepted:
                return None, dict(status='line_search_rejected', history=history)
        return None, dict(status='ik_budget', history=history)

    def _geometry(self):
        np, mj, m, d = self.np, self.mj, self.model, self.scratch
        unexpected = []
        for c in d.contact[:d.ncon]:
            if c.dist <= 0. and {int(c.geom1), int(c.geom2)} != self.allowed_pair:
                unexpected.append(dict(geom_ids=[int(c.geom1), int(c.geom2)], distance_m=float(c.dist)))
        best, best_pair = float('inf'), None
        finger_gap = float('inf')
        for g, vertices in self.meshes:
            world = vertices@d.geom_xmat[g].reshape(3, 3).T+d.geom_xpos[g]
            if not np.isfinite(world).all():
                return dict(minimum_environment_gap_lower_bound_m=-1., nearest_geom_ids=[g],
                    minimum_finger_table_gap_m=-1., unexpected_contacts=unexpected,
                    invalid_geometry='nonfinite_arm_vertices')
            if g in self.ctx['finger_geoms']:
                finger_gap = min(finger_gap, float(world[:, 2].min())-self.ctx['cfg']['table']['top_z_m'])
            local = np.einsum('ovi,oij->ovj', world[None, :, :]-self.obstacle_position[:, None, :],
                              self.obstacle_rotation)
            gaps = np.maximum(-self.obstacle_size-local.max(axis=1),
                              local.min(axis=1)-self.obstacle_size)
            bounds = np.linalg.norm(np.maximum(gaps, 0.), axis=1)
            if not np.isfinite(bounds).all():
                return dict(minimum_environment_gap_lower_bound_m=-1., nearest_geom_ids=[g],
                    minimum_finger_table_gap_m=-1., unexpected_contacts=unexpected,
                    invalid_geometry='nonfinite_clearance_bound')
            j = int(np.argmin(bounds))
            if bounds[j] < best:
                best, best_pair = float(bounds[j]), [g, self.obstacles[j]]
        return dict(minimum_environment_gap_lower_bound_m=best, nearest_geom_ids=best_pair,
            minimum_finger_table_gap_m=finger_gap, unexpected_contacts=unexpected)

    def _segment(self, start, end, target, duration, kind, ik):
        np = self.np
        steps = round(duration/DT)
        rows = []
        minimum_gap, minimum_table = float('inf'), float('inf')
        max_angle = max_line = max_gravity = 0.
        self._forward(start)
        p0 = self.scratch.site_xpos[self.ctx['site']].copy()
        for i in range(steps+1):
            u = i/steps
            s, ds = u**3*(10+u*(-15+6*u)), 30*u*u*(1-u)**2/duration
            q, dq = start+s*(end-start), ds*(end-start)
            # Exact held endpoint avoids any accumulation of interpolation roundoff.
            if i == 0: q = start.copy()
            if i == steps: q = end.copy()
            self._forward(q)
            p = self.scratch.site_xpos[self.ctx['site']].copy()
            angle = float(np.linalg.norm(self._rotation_error()))
            line = float(np.linalg.norm(p-(p0+s*(target-p0))))
            geometry = self._geometry()
            gap = geometry['minimum_environment_gap_lower_bound_m']
            gravity = float(np.max(np.abs(self.scratch.qfrc_bias[self.ctx['va']])/self.ctx['cap']))
            reasons = []
            if not np.isfinite(np.r_[q, dq, p, angle, line, gap, gravity]).all(): reasons.append('nonfinite_reference')
            if np.any(q[:7] < self.lo) or np.any(q[:7] > self.hi): reasons.append('joint_limit_margin')
            if np.any(q[7:] < self.ctx['finger_limits'][:, 0]) or np.any(q[7:] > self.ctx['finger_limits'][:, 1]+1e-12): reasons.append('finger_limits')
            if np.max(np.abs(dq[:7])) > SETTINGS['maximum_arm_reference_speed_rad_s']: reasons.append('arm_speed')
            if np.max(np.abs(dq[7:])) > SETTINGS['maximum_finger_reference_speed_m_s']: reasons.append('finger_speed')
            if angle > SETTINGS['path_orientation_tolerance_rad']: reasons.append('reference_orientation')
            if line > SETTINGS['path_line_deviation_m']: reasons.append('reference_line_deviation')
            if np.any(np.abs(p-self.origin_xyz) > SETTINGS['workspace_half_extent_m']+SETTINGS['position_tolerance_m']): reasons.append('reference_workspace')
            if gap < SETTINGS['minimum_environment_gap_m']: reasons.append('environment_clearance')
            if geometry['unexpected_contacts']: reasons.append('unexpected_contact')
            if gravity > SETTINGS['maximum_static_gravity_to_cap']: reasons.append('gravity_headroom')
            if reasons:
                return self._reject('reference_sample_rejected', dict(kind=kind, rejected_step=i,
                    checks=reasons, geometry=geometry, orientation_error_rad=angle,
                    line_deviation_m=line, gravity_to_cap=gravity, ik=ik))
            rows.append(dict(step=i, sim_s=i*DT, q_ref9=q.tolist(), dq_ref9=dq.tolist(),
                position_world_m=p.tolist(), orientation_wxyz=self._quaternion().tolist(),
                minimum_environment_gap_lower_bound_m=gap,
                minimum_finger_table_gap_m=geometry['minimum_finger_table_gap_m']))
            minimum_gap, minimum_table = min(minimum_gap, gap), min(minimum_table, geometry['minimum_finger_table_gap_m'])
            max_angle, max_line, max_gravity = max(max_angle, angle), max(max_line, line), max(max_gravity, gravity)
        endpoint_error = float(np.linalg.norm(np.asarray(rows[-1]['position_world_m'])-target))
        if endpoint_error > SETTINGS['position_tolerance_m'] or angle > SETTINGS['orientation_tolerance_rad']:
            return self._reject('endpoint_pose_rejected', dict(position_error_m=endpoint_error, orientation_error_rad=angle))
        self.target_xyz, self.last_q9 = target.copy(), end.copy()
        return dict(accepted=True, reason='static_reference_accepted', kind=kind,
            target_xyz=target.tolist(), duration_s=duration, rows=rows,
            planning_wall_s=time.perf_counter()-self._command_started,
            diagnostics=dict(ik=ik, endpoint_position_error_m=endpoint_error,
                endpoint_orientation_error_rad=angle, maximum_path_orientation_error_rad=max_angle,
                maximum_line_deviation_m=max_line, minimum_environment_gap_lower_bound_m=minimum_gap,
                minimum_finger_table_gap_m=minimum_table, maximum_gravity_to_cap=max_gravity,
                checked_samples=len(rows), discrete_only=True))

    def _reject(self, reason, diagnostics=None):
        return dict(accepted=False, reason=reason, target_xyz=self.target_xyz.tolist(),
                    duration_s=0., rows=[], diagnostics=diagnostics or {},
                    planning_wall_s=time.perf_counter()-self._command_started)

    def plan_delta(self, delta_world_m, q_start9):
        """Accept one <=5 mm world translation, else retain the held target."""
        self._command_started = time.perf_counter()
        np = self.np
        start, error = self._start(q_start9)
        if error: return self._reject(error)
        delta = np.asarray(delta_world_m, dtype=float)
        if delta.shape != (3,) or not np.isfinite(delta).all(): return self._reject('invalid_delta')
        norm = float(np.linalg.norm(delta))
        if norm <= 1e-12 or norm > SETTINGS['maximum_delta_m']+1e-12: return self._reject('delta_size')
        target = self.target_xyz+delta
        if np.any(np.abs(target-self.origin_xyz) > SETTINGS['workspace_half_extent_m']+1e-12): return self._reject('workspace_boundary')
        end, info = self._solve(start, target)
        if end is None: return self._reject('ik_failed', info)
        return self._segment(start, end, target, SETTINGS['motion_duration_s'], 'translate', info)

    def plan_fingers(self, finger_target, q_start9):
        """Both unloaded fingers follow the accepted open/closed values in 2 s."""
        self._command_started = time.perf_counter()
        start, error = self._start(q_start9)
        if error: return self._reject(error)
        if not isinstance(finger_target, (float, int, self.np.floating)):
            return self._reject('invalid_finger_target')
        if not any(abs(float(finger_target)-value) < 1e-12 for value in (OPEN, CLOSED)):
            return self._reject('finger_target_not_open_or_closed')
        if self.np.allclose(start[7:], finger_target, atol=1e-12, rtol=0):
            return self._reject('fingers_already_at_target')
        end = start.copy()
        end[7:] = finger_target
        return self._segment(start, end, self.target_xyz.copy(), SETTINGS['finger_duration_s'], 'fingers', None)


def prepare(out):
    """Bounded synthetic key sequences; no dynamics, no viewer."""
    import numpy as np
    import mujoco
    import carry_to_tray
    def forbidden(*args, **kwargs):
        raise AssertionError('Keyboard preparation must not integrate')
    mujoco.mj_step = forbidden
    ctx = carry_to_tray.setup(mujoco, np)
    initialization = prepare_initial_pose(ctx, mujoco, np)
    d = ctx['data']
    before = [d.qpos.copy(), d.qvel.copy(), d.ctrl.copy(), d.qacc_warmstart.copy(), float(d.time)]
    planner = TargetPlanner(ctx, mujoco, np)
    commands, rejected = [], []
    started = time.perf_counter()
    # Six axis extremes and their complete return paths; small bounded lesson.
    for axis in range(3):
        for sign in (-1, 1):
            delta = np.eye(3)[axis]*sign*.005
            for direction in [1]*2+[-1]*2:
                result = planner.plan_delta(direction*delta, planner.last_q9)
                commands.append(result)
                if not result['accepted']:
                    raise RuntimeError('Axis/return reference rejected: '+json.dumps(result))
    for value in (CLOSED, OPEN):
        result = planner.plan_fingers(value, planner.last_q9)
        commands.append(result)
        if not result['accepted']:
            raise RuntimeError('Empty-hand finger reference rejected: '+json.dumps(result))
    # No runtime state or accepted target may change for malformed/outside input.
    for delta, expected in [([.00501, 0, 0], 'delta_size'), ([float('nan'), 0, 0], 'invalid_delta'), ([0, 0, 0], 'delta_size')]:
        oldq, oldp = planner.last_q9.copy(), planner.target_xyz.copy()
        result = planner.plan_delta(delta, oldq)
        assert not result['accepted'] and result['reason'] == expected
        assert np.array_equal(planner.last_q9, oldq) and np.array_equal(planner.target_xyz, oldp)
        rejected.append(dict(test=expected, result=result))
    for _ in range(2):
        result = planner.plan_delta([.005, 0, 0], planner.last_q9)
        commands.append(result)
        assert result['accepted']
    oldq, oldp = planner.last_q9.copy(), planner.target_xyz.copy()
    result = planner.plan_delta([.005, 0, 0], oldq)
    assert result['reason'] == 'workspace_boundary'
    assert np.array_equal(planner.last_q9, oldq) and np.array_equal(planner.target_xyz, oldp)
    rejected.append(dict(test='workspace_boundary_atomic', result=result))
    result = planner.plan_delta([-.005, 0, 0], oldq+1e-4)
    assert result['reason'] == 'start_is_not_last_accepted_endpoint'
    rejected.append(dict(test='reference_continuity', result=result))
    unchanged = all(np.array_equal(current, old) for current, old in zip(
        [d.qpos, d.qvel, d.ctrl, d.qacc_warmstart], before[:4])) and d.time == before[4]
    assert unchanged
    rows = [r for command in commands for r in command['rows']]
    receipt = dict(status='discrete_static_keyboard_preparation_passed', physics_steps=0,
        no_viewer=True, live_state_unchanged=unchanged, source_summary=str(SOURCE), source_sha256=SOURCE_SHA,
        scene_sha256=sha(ctx['folder']/'scene.xml'), source_code_sha256=sha(Path(__file__)),
        settings=SETTINGS, executable=sys.executable, temp=tempfile.gettempdir(),
        initialization=initialization,
        origin_xyz=planner.origin_xyz.tolist(), orientation_wxyz=planner.orientation_wxyz.tolist(),
        accepted_command_count=len(commands), rejected_tests=rejected, checked_samples=len(rows),
        forward_calls=planner.forward_calls, wall_s=time.perf_counter()-started,
        minimum_environment_gap_lower_bound_m=min(r['minimum_environment_gap_lower_bound_m'] for r in rows),
        minimum_finger_table_gap_m=min(r['minimum_finger_table_gap_m'] for r in rows),
        maximum_endpoint_position_error_m=max(c['diagnostics']['endpoint_position_error_m'] for c in commands),
        maximum_path_orientation_error_rad=max(c['diagnostics']['maximum_path_orientation_error_rad'] for c in commands),
        maximum_line_deviation_m=max(c['diagnostics']['maximum_line_deviation_m'] for c in commands),
        maximum_gravity_to_cap=max(c['diagnostics']['maximum_gravity_to_cap'] for c in commands),
        maximum_planning_wall_s=max(c['planning_wall_s'] for c in commands),
        limitations=['Six axes and returns tested, not every 3D workspace combination.',
            'Endpoint orientation is held; joint interpolation may vary intermediate orientation.',
            'Every future command is checked independently; finite samples are not continuous collision proof.',
            'No user dynamic run, contact grasp, IK global reachability or v0.1 completion is claimed.'])
    save_json(out/'commands.json', commands)
    save_json(out/'receipt.json', receipt)
    shutil.copy2(Path(__file__), out/Path(__file__).name)
    return receipt


if __name__ == '__main__':
    guard()
    out = ROOT/'outputs'/('cartesian_keyboard_target_preparation_'+stamp())
    out.mkdir(exist_ok=False)
    try:
        result = prepare(out)
    except Exception as exc:
        save_json(out/'failure.json', dict(status='preparation_failed', error=repr(exc), physics_steps=0))
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print('TARGET PREPARATION:', out)
