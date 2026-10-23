"""Manually commanded contact pick/place; no scheduled Cartesian task route.

Only setup initializes the live state. Later references follow the user's held
direction or gripper press. Inspection/planning use separate data and no mj_step.
"""
import argparse
import csv
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
import cartesian_keyboard as keyboard
import carry_to_tray as carry
import carry_runtime as task
import grasp_lift as base
import grasp_solver_runtime as frozen
from manual_task_observer import FreeManualTaskObserver
from manual_task_target import TaskPlanner, prepare_initial_pose, SETTINGS as PLANNER_SETTINGS
from manual_jog_input import WindowsJogInput
from pickplace_layout_preview import set_camera

DT = base.DT
DELTAS, KEY_COMMANDS, KEY_NAMES = keyboard.DELTAS, keyboard.KEY_COMMANDS, keyboard.KEY_NAMES
THRESHOLDS = dict(keyboard.THRESHOLDS, maximum_simulation_s=420., maximum_physics_wall_s=900.)
# The cube is intentionally transported here; empty-hand immobility limits do
# not belong in this task's reported thresholds.
THRESHOLDS.pop('maximum_cube_xy_displacement_m')
THRESHOLDS.pop('maximum_cube_z_displacement_m')
SUCCESS = ROOT/'configs'
SUCCESS_SHA = '35ab20b50e204f38b21bdf82270a329cd959f70824068315480747095c8f7514'
STATIC_HELP = ('MANUAL PICK AND PLACE | FIXED BASE | OFFLINE SIMULATION\n'
    '7:start  8:camera  9:finish/save  6:close/open\n'
    'Up/Down: world +/-X; Left/Right: world +/-Y\n'
    'PageUp/PageDown: world +/-Z. HOLD one direction to move.\n'
    'Release to slow/stop. 6: manual OPEN/CLOSE at any task stage.\n'
    'Task observations do not lock the gripper. No fixed key sequence.\n'
    'All movements require your keys. 9 also saves an incomplete attempt.')


def setup(mujoco, np):
    from package_runtime import setup as package_setup
    return package_setup(mujoco, np)



def ready_valid(row, session):
    closed = session.closed_requested and not session.checks.release_started
    fingers = (row['max_finger_error_m'] <= THRESHOLDS['ready_finger_error_m'] or
               (closed and row['finger1_contact_count']+row['finger2_contact_count'] > 0))
    held = (not session.checks.baseline_active or session.checks.release_started or
            (row['relative_slip_m'] <= base.THRESHOLDS['maximum_relative_slip_m'] and
             row['relative_rotation_rad'] <= base.THRESHOLDS['maximum_relative_rotation_rad']))
    return bool(fingers and held and
        row['position_error_m'] <= THRESHOLDS['position_warning_m'] and
        row['orientation_error_rad'] <= THRESHOLDS['orientation_warning_rad'] and
        row['target_position_error_m'] <= THRESHOLDS['position_warning_m'] and
        row['target_orientation_error_rad'] <= THRESHOLDS['orientation_warning_rad'] and
        row['max_joint_speed_rad_s'] <= THRESHOLDS['ready_arm_speed_rad_s'] and
        row['max_finger_speed_m_s'] <= THRESHOLDS['ready_finger_speed_m_s'] and
        row['unexpected_contact_count'] == 0)


def input_hint(result, gripper=False):
    """A refusal should explain the adjustment, not expose an internal gate."""
    reason=result['reason']
    hints={
        'busy_or_not_ready_no_queue':'Release direction keys; let the hand settle',
        'received_busy_or_stale_no_queue':'Release direction keys; let the hand settle',
        'floor_margin':'Near support surface: move UP; 6 still controls the gripper',
        'workspace_boundary':'Workspace edge: release key and change direction',
    }
    checks=result.get('diagnostics',{}).get('checks',[])
    if any(c in checks for c in ('environment_clearance','payload_clearance','unexpected_contact')):
        return ('Gripper motion blocked by nearby geometry' if gripper else
                'Obstacle nearby: release key and change direction')
    return hints.get(reason,reason)


class ManualSession(keyboard.KeyboardSession):
    def __init__(self, ctx, planner, np):
        super().__init__(dict(ctx, keyboard_anchor=ctx['manual_anchor']), planner, np)
        self.ctx = ctx
        self.closed_requested = False
        self.pending_gripper_closed = None
        self.notice = ''
        self.jog_reference_completed_ids = []
        self.last_plan_kind = None
        self.checks = FreeManualTaskObserver(ctx['cfg']['tray']['wall_top_z_m']-ctx['cfg']['tray']['floor_top_z_m'])

    @property
    def mode(self):
        if self.checks.release_started:
            return 'released'
        if self.checks.baseline_active:
            return 'held'
        return 'closing' if self.closed_requested else 'empty'

    def observe(self, row, step):
        if self.phase not in ('settling', 'ready'):
            self.ready_since = None
            return
        if ready_valid(row, self):
            if self.ready_since is None:
                self.ready_since = step
            if (step-self.ready_since)*DT >= THRESHOLDS['ready_stable_s']-1e-10:
                self.phase = 'ready'
                if self.pending_id is not None:
                    self.completed_ids.append(self.pending_id)
                    self.pending_id = None
        else:
            if self.phase == 'ready':
                self.settle_step = step
            self.phase = 'settling'
            self.ready_since = None

    def command_gate(self, command, row, jogging=False):
        # Task observations never authorize/forbid operator commands.
        if self.pending_gripper_closed is not None:
            return 'gripper_request_pending'
        continuation = (jogging and self.phase == 'settling' and self.plan is None
                        and self.last_plan_kind == 'jog')
        if self.phase != 'ready' and not continuation:
            return 'busy_or_not_ready_no_queue'
        return None

    def queue_gripper_toggle(self):
        if self.phase=='waiting_start':
            return dict(accepted=False,reason='press_7_to_start')
        desired=(self.closed_requested if self.pending_gripper_closed is None
                 else self.pending_gripper_closed)
        self.pending_gripper_closed=not desired
        return dict(accepted=True,reason='gripper_intent_accepted',
                    desired_closed=self.pending_gripper_closed)

    def advance(self, step):
        if self.plan is not None and step-self.start_step >= len(self.plan['rows'])-1:
            self.last_plan_kind = self.plan['kind']
            if self.last_plan_kind == 'jog':
                self.jog_reference_completed_ids.append(self.command_id)
        super().advance(step)

    def request_jog(self, command, step, row, still_held):
        """One short checked segment; latest held direction, never a key queue."""
        rejected = self.command_gate(command, row, jogging=True)
        if rejected:
            return dict(accepted=False, reason=rejected, command=command)
        if (row['position_error_m'] > THRESHOLDS['position_warning_m'] or
                row['orientation_error_rad'] > THRESHOLDS['orientation_warning_rad']):
            return dict(accepted=False, reason='wait_for_tracking_to_settle', command=command)
        previous_phase = self.phase
        previous_target = self.planner.target_xyz.copy()
        previous_q = self.planner.last_q9.copy()
        self.phase = 'planning'
        self.generation += 1
        delta = self.np.asarray(DELTAS[command]) * (.0015/.005)
        plan = self.planner.plan_jog(delta, self.q_hold.copy(), mode=self.mode,
            grip_baseline=self.ctx['grip_baseline'], grasp_confirmed=self.checks.grasp_hold_passed)
        if not plan['accepted']:
            self.phase = previous_phase
            return plan
        # Planning pauses simulation. Releasing/focus loss during planning must
        # not turn its now-stale target into a future physical command.
        if not still_held():
            self.planner.target_xyz = previous_target
            self.planner.last_q9 = previous_q
            self.phase = previous_phase
            return dict(accepted=False, reason='released_before_commit', command=command)
        rows = plan['rows']
        if len(rows) < 2 or not self.np.allclose(rows[0]['q_ref9'],self.q_hold,rtol=0,atol=1e-10):
            raise ValueError('Jog must start from current reference, without a jump')
        self.command_id += 1
        self.pending_id = self.command_id
        self.command_records.append(dict(command_id=self.command_id,command=command,
            input_mode='held_direction_segment',finger_target_m=None))
        self.plan = plan
        self.goal = self.np.asarray(plan['target_xyz'],dtype=float).copy()
        self.start_step = step
        self.phase = 'moving'
        self.ready_since = None
        if command == 'Z+' and self.checks.release_started:
            self.checks.notify_retraction(step)
        elif command == 'Z-' and self.checks.transfer_completed:
            self.checks.notify_descent(step)
        return plan

    def apply_pending_gripper(self, step, row):
        if self.pending_gripper_closed is None or self.plan is not None:
            return None
        closing=self.pending_gripper_closed
        self.pending_gripper_closed=None
        destination=base.CLOSED if closing else base.OPEN
        if self.np.allclose(self.q_hold[7:],destination,atol=1e-12,rtol=0):
            return dict(accepted=False,reason='latest_gripper_intent_already_satisfied')
        previous_phase=self.phase
        self.phase = 'planning'
        self.generation += 1
        plan = self.planner.plan_manual_fingers(destination,self.q_hold.copy())
        if not plan['accepted']:
            self.phase = previous_phase
            return plan
        rows = plan['rows']
        if len(rows) < 2 or not self.np.allclose(rows[0]['q_ref9'], self.q_hold, rtol=0, atol=1e-10):
            raise ValueError('New reference must start at the held reference')
        self.command_id += 1
        self.pending_id = self.command_id
        self.command_records.append(dict(command_id=self.command_id, command='GRIPPER_TARGET', finger_target_m=destination))
        self.plan = plan
        self.goal = self.np.asarray(plan['target_xyz'], dtype=float).copy()
        self.start_step = step
        self.phase = 'moving'
        self.ready_since = None
        self.checks.notify_gripper(closing,step,row)
        self.ctx['grip_baseline']=None
        self.closed_requested=closing
        return plan


def sample(ctx, session, step, wall_s, mujoco, np):
    m, d, r = ctx['model'], ctx['data'], ctx['reference']
    if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
        raise RuntimeError('Nonfinite actual state')
    qref, qdref = session.reference(step)
    pd, gravity, raw, tau = base.motion.control_terms(ctx, qref, qdref, mujoco, np)
    d.ctrl[ctx['aids']] = tau
    mujoco.mj_forward(m, d)
    if not np.allclose(d.qfrc_actuator[ctx['va']], tau, rtol=0, atol=1e-9):
        raise RuntimeError('Actuator force and recorded command differ')
    r.qpos[:] = ctx['base_qpos']; r.qpos[ctx['qa']] = qref; r.qvel[:] = 0
    mujoco.mj_forward(m, r)
    q, qd = d.qpos[ctx['qa']].copy(), d.qvel[ctx['va']].copy()
    actual, ref = d.site_xpos[ctx['site']].copy(), r.site_xpos[ctx['site']].copy()
    quat, refquat = np.empty(4), np.empty(4)
    mujoco.mju_mat2Quat(quat, d.site_xmat[ctx['site']]); mujoco.mju_mat2Quat(refquat, r.site_xmat[ctx['site']])
    angle = float(2*np.arccos(np.clip(abs(float(quat@refquat)), 0., 1.)))
    target_angle = float(2*np.arccos(np.clip(abs(float(quat@ctx['manual_orientation'])), 0., 1.)))
    forces, counts, table_contacts = [0., 0.], [0, 0], 0
    unexpected, contact_details = [], []
    cube, table = ctx['cube_geom'], ctx['table_geom']
    tray = ctx['carry_route']['tray']
    release = session.checks.release_started
    for ci, contact in enumerate(d.contact[:d.ncon]):
        pair = {int(contact.geom1), int(contact.geom2)}
        finger_pair = cube in pair and any(g in pair for g in ctx['finger_geoms'])
        table_pair = cube in pair and table in pair
        floor_pair = cube in pair and tray['floor_geom_id'] in pair
        # Cube/support contact is a normal physical outcome, including an
        # early voluntary drop. It affects task score, not input permission.
        allowed = ((finger_pair and (session.closed_requested or release)) or table_pair or floor_pair)
        if finger_pair or table_pair or floor_pair:
            wrench = np.zeros(6); mujoco.mj_contactForce(m, d, ci, wrench)
            active = contact.efc_address >= 0 and wrench[0] > 0
            if active:
                if table_pair:
                    table_contacts += 1
                for i, g in enumerate(ctx['finger_geoms']):
                    if g in pair:
                        counts[i] += 1; forces[i] += float(wrench[0])
            contact_details.append(dict(pair=sorted(pair), dist_m=float(contact.dist), normal_n=float(wrench[0]), active=bool(active), allowed=allowed))
        if not allowed and contact.dist <= 0:
            unexpected.append(dict(geoms=[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in sorted(pair)], distance_m=float(contact.dist)))
    finger_gap = min(float((v@d.geom_xmat[g].reshape(3, 3).T+d.geom_xpos[g])[:, 2].min()) for g, v in ctx['finger_meshes'])-ctx['cfg']['table']['top_z_m']
    cube_position, cube_rotation = d.geom_xpos[cube].copy(), d.geom_xmat[cube].reshape(3, 3)
    bottom = float(cube_position[2]-np.abs(cube_rotation[2])@m.geom_size[cube])
    slip = rotation_error = ''
    baseline_valid = session.checks.baseline_active and ctx['grip_baseline'] is not None and not release
    if baseline_valid:
        relpos, relrot = base.cube_relative(ctx, np)
        oldpos, oldrot = ctx['grip_baseline']
        slip = float(np.linalg.norm(relpos-oldpos))
        rotation_error = float(np.arccos(np.clip((np.trace(oldrot.T@relrot)-1)/2., -1., 1.)))
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'task_cube_free')
    if jid < 0:
        raise ValueError('Missing cube free joint')
    cv = int(m.jnt_dofadr[jid])
    row = dict(sim_s=float(d.time), wall_s=wall_s, phase=session.phase, terminal_state=0,
        command_id=session.command_id, busy=int(session.phase in ('moving', 'settling')), ready=int(session.phase=='ready'),
        position_error_m=float(np.linalg.norm(actual-ref)), orientation_error_rad=angle,
        target_position_error_m=float(np.linalg.norm(actual-session.goal)), target_orientation_error_rad=target_angle,
        finger_ref_m=float(qref[7]), finger1_q_m=float(q[7]), finger2_q_m=float(q[8]),
        finger1_normal_n=forces[0], finger2_normal_n=forces[1], finger1_contact_count=counts[0], finger2_contact_count=counts[1],
        cube_table_contact_count=table_contacts, cube_bottom_clearance_m=bottom-ctx['cfg']['table']['top_z_m'],
        cube_lift_from_start_m=float(cube_position[2]-ctx['cube_initial'][2]),
        cube_speed_m_s=float(np.linalg.norm(d.qvel[cv:cv+3])), cube_angular_speed_rad_s=float(np.linalg.norm(d.qvel[cv+3:cv+6])),
        relative_slip_m=slip, relative_rotation_rad=rotation_error, grip_reference_valid=int(baseline_valid),
        release_active=int(release), unexpected_contact_count=len(unexpected),
        max_torque_ratio=float(np.max(np.abs(raw[:7])/ctx['cap'][:7])),
        max_finger_force_ratio=float(np.max(np.abs(raw[7:])/ctx['cap'][7:])), min_finger_table_gap_m=finger_gap,
        max_joint_error_rad=float(np.max(np.abs(q[:7]-qref[:7]))), max_finger_error_m=float(np.max(np.abs(q[7:]-qref[7:]))),
        max_joint_speed_rad_s=float(np.max(np.abs(qd[:7]))), max_finger_speed_m_s=float(np.max(np.abs(qd[7:]))))
    for prefix, values in (('cube', cube_position), ('actual', actual), ('ref', ref), ('target', session.goal)):
        row.update({prefix+'_'+axis+'_m':float(value) for axis, value in zip('xyz', values)})
    for i in range(9):
        for key, values in (('q_ref',qref), ('q',q), ('qd_ref',qdref), ('qd',qd), ('pd',pd), ('gravity',gravity), ('tau_raw',raw), ('tau',tau)):
            row[f'{key}_{i}'] = float(values[i])
    row.update(task.tray_fields(ctx, row, mujoco, np))
    problems = []
    if abs(d.time-step*DT) > 1e-8: problems.append('unexpected_time')
    if row['position_error_m'] > THRESHOLDS['position_stop_m'] or angle > THRESHOLDS['orientation_stop_rad']: problems.append('arm_pose_tracking_stop')
    if row['max_joint_error_rad'] > .15 or row['max_joint_speed_rad_s'] > 2.: problems.append('arm_joint_tracking_stop')
    if np.any(q[:7] <= ctx['limits'][:,0]+.02) or np.any(q[:7] >= ctx['limits'][:,1]-.02): problems.append('arm_limit_stop')
    if np.any(q[7:] < ctx['finger_limits'][:,0]-.0005) or np.any(q[7:] > ctx['finger_limits'][:,1]+.0005) or row['max_finger_speed_m_s'] > .25: problems.append('finger_limit_or_speed_stop')
    if finger_gap < THRESHOLDS['min_finger_table_gap_m']: problems.append('finger_table_clearance_stop')
    if unexpected: problems.append('unexpected_contact')
    half = np.abs(cube_rotation)@m.geom_size[cube]
    if (row['cube_bottom_clearance_m'] < -.003 or cube_position[2] > ctx['cfg']['table']['top_z_m']+.4 or
            np.any(np.abs(cube_position[:2]-np.asarray(ctx['cfg']['table']['center_xy_m']))+half[:2] > np.asarray(ctx['cfg']['table']['top_half_size_xy_m'])+.005)):
        problems.append('cube_left_task_workspace')
    if baseline_valid and (slip > .015 or rotation_error > .25): problems.append('object_slip_stop')
    if any(w.number for w in d.warning): problems.append('mujoco_warning')
    return row, problems, dict(unexpected=unexpected, contacts=contact_details)


def observe_task(ctx, session, row, step, mujoco, np):
    session.observe(row, step)
    row.update(phase=session.phase, busy=int(session.phase in ('moving','settling')), ready=int(session.phase=='ready'))
    result = session.checks.observe(row, step)
    if result['need_capture_baseline']:
        ctx['grip_baseline'] = tuple(x.copy() for x in base.cube_relative(ctx, np))
        session.checks.baseline_captured(step)
        row.update(grip_reference_valid=1, relative_slip_m=0., relative_rotation_rad=0.)
        result = session.checks.observe(row, step)
    row.update(result['row_fields'])
    return result['stop_reasons']


def inspect(ctx, mujoco, np):
    planner = TaskPlanner(ctx, mujoco, np)
    before = (ctx['data'].qpos.copy(), ctx['data'].qvel.copy(), float(ctx['data'].time))
    session = ManualSession(ctx, planner, np)
    row, problems, contacts = sample(ctx, session, 0, 0., mujoco, np)
    if problems:
        raise ValueError('Manual initial state failed: '+str(problems))
    if not np.array_equal(before[0],ctx['data'].qpos) or not np.array_equal(before[1],ctx['data'].qvel) or before[2] != ctx['data'].time:
        raise RuntimeError('Inspection changed integrated state')
    return planner, dict(status='prepared_static_only', physics_steps=0, initial_sample=row,
        contacts=contacts, anchor_xyz_m=ctx['manual_anchor'].tolist(),
        targets=keyboard.jsonable(ctx['manual_targets']), settings=PLANNER_SETTINGS,
        limitation='No manual contact motion or user acceptance has been run.')


def update_scene(viewer, ctx, session, row, mujoco, np):
    viewer.perturb.select=0; viewer.perturb.active=0
    scene=viewer.user_scn; scene.ngeom=0
    if scene.maxgeom < 7: raise RuntimeError('Viewer needs seven annotation slots')
    points=[(ctx['manual_targets']['B'],[.2,1.,.3,1.]),
            (ctx['manual_targets']['C'],[.7,.3,1.,1.]),(session.goal,[1.,.1,.5,1.])]
    for i,(point,color) in enumerate(points):
        mujoco.mjv_initGeom(scene.geoms[i],mujoco.mjtGeom.mjGEOM_SPHERE,np.full(3,.0035),np.asarray(point),np.eye(3).ravel(),np.asarray(color,dtype=np.float32))
    lines=[f'{session.phase.upper()} | {session.checks.task_stage}',
           session.notice or '6: OPEN/CLOSE | task score does not lock input',
           f'Contact N: {row["finger1_normal_n"]:.2f}/{row["finger2_normal_n"]:.2f} | cube above tray floor {row["cube_bottom_tray_clearance_m"]*1000:.1f}mm',
           'PLACEMENT PASS - press 9 to save' if row.get('placement_continuous_s',0.) >= 2.-1e-10 else
           f'Gripper target: {"CLOSED" if session.closed_requested else "OPEN"} | placed {row.get("placement_continuous_s",0.):.1f}/2s']
    actual=np.asarray([row['actual_'+a+'_m'] for a in 'xyz'])
    for j,text in enumerate(lines):
        geom=scene.geoms[3+j]
        mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_LABEL,np.zeros(3),actual+[0.,0.,.14-j*.025],np.eye(3).ravel(),np.ones(4,dtype=np.float32))
        geom.label=text[:95]
    scene.ngeom=7


def completion(session, reason, metrics, step, last):
    tracking=bool(step>0 and metrics['max_position_error_m']<=THRESHOLDS['position_warning_m'] and
                  metrics['max_orientation_error_rad']<=THRESHOLDS['orientation_warning_rad'])
    passed=bool(reason=='user_9' and session.phase=='ready' and tracking and
        session.checks.placement_passed and not session.checks.stop_reasons and
        last and last['placement_valid'] and
        last['placement_continuous_s'] >= task.PLACEMENT_THRESHOLDS['hold_s']-1e-10 and
        session.pending_id is None and session.pending_gripper_closed is None)
    return dict(manual_task_completed=passed, tracking_thresholds_passed=tracking,
        full_v01_accepted=False, learner_verified=False, **session.checks.summary())


def run(ctx, planner, report, out, mujoco, np):
    import mujoco.viewer
    m, d = ctx['model'], ctx['data']
    session = ManualSession(ctx, planner, np)
    jog_input = WindowsJogInput()
    blocked_direction = None
    previous_input = None
    commands = queue.SimpleQueue()
    started = time.perf_counter(); physics_started = None
    step = 0; last = None; terminal_saved = False; camera_index = 0
    status, reason = 'interrupted', 'window_closed'
    saturation_s = 0.; stop = False
    options = frozen.option_snapshot(m, np)
    report['frozen_physics_options'] = options
    first, problems, _ = sample(ctx, session, 0, 0., mujoco, np)
    if problems: raise RuntimeError('Initial sample failed: '+str(problems))
    first.update(session.checks.observe(first, 0)['row_fields'])
    metrics = dict(max_position_error_m=0., max_orientation_error_rad=0., max_joint_error_rad=0.,
        max_torque_ratio=0., max_finger_force_ratio=0., max_unexpected_contact_count=0,
        max_held_relative_slip_m=0., max_held_relative_rotation_rad=0.)

    def write(writer, row, terminal=False):
        nonlocal last, terminal_saved
        if terminal:
            row = base.motion.terminal_row(row); terminal_saved = True
            for key in row:
                if key.startswith('qd_ref_'):
                    row[key] = ''
        writer.writerow(row); last = row
        report['data_saved_samples'] += 1
        for key, col in (('max_position_error_m','position_error_m'), ('max_orientation_error_rad','orientation_error_rad'),
                ('max_joint_error_rad','max_joint_error_rad'), ('max_torque_ratio','max_torque_ratio'),
                ('max_finger_force_ratio','max_finger_force_ratio'), ('max_unexpected_contact_count','unexpected_contact_count'),
                ('max_held_relative_slip_m','relative_slip_m'), ('max_held_relative_rotation_rad','relative_rotation_rad')):
            if row[col] != '': metrics[key] = max(metrics[key], row[col])

    def callback(key):
        if int(key) in KEY_COMMANDS and KEY_COMMANDS[int(key)] in DELTAS:
            return  # Direction keys are sampled as held state, never queued.
        commands.put((int(key), time.perf_counter(), session.phase, session.generation))

    def event_write(stage, event):
        with (out/'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(keyboard.jsonable(dict(stage=stage, **event)), ensure_ascii=False)+'\n')

    with (out/'manual.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(first)); writer.writeheader()
        try:
            with mujoco.viewer.launch_passive(m,d,key_callback=callback,show_left_ui=False,show_right_ui=False) as viewer:
                report['viewer_started'] = True
                with viewer.lock():
                    set_camera(viewer.cam, ctx['cfg'], frozen.CAMERAS[camera_index])
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
                # One native text call only; subsequent status uses locked scene labels.
                viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100,mujoco.mjtGridPos.mjGRID_TOPLEFT,STATIC_HELP,''))
                report['viewer_status_mode'] = 'one_static_overlay_and_locked_scene_labels'
                print('7 starts. HOLD one direction to move; release to slow/stop. 6 commands CLOSE/OPEN at any task stage.',flush=True)
                print('A pending gripper request has priority at the current segment end. Latest requested state wins.',flush=True)
                print('Short checked segments: <=1.5 mm / 0.12 simulation s. No per-step READY or key counts.',flush=True)
                print('9 ends and saves at any time. After PLACEMENT PASS, press 9 to finish.',flush=True)
                last_phase = last_task = None
                while viewer.is_running():
                    tick = time.perf_counter()
                    frozen.assert_options_unchanged(m,options,np)
                    if ((physics_started is None and tick-started>600.) or
                            (physics_started is not None and tick-physics_started>THRESHOLDS['maximum_physics_wall_s'])):
                        reason='wall_time_limit'; break
                    while not commands.empty():
                        key, received, received_phase, received_generation = commands.get()
                        event = dict(key=KEY_NAMES.get(key,str(key)), command=KEY_COMMANDS.get(key,'OTHER'),
                            sim_s=float(d.time), wall_s=received-started, accepted=False, reason='unsupported_key',
                            command_id=session.command_id, accepted_target_xyz=session.goal.tolist(),
                            received_phase=received_phase, received_generation=received_generation,
                            process_delay_wall_s=time.perf_counter()-received)
                        event_write('received',event)
                        if key == ord('9'):
                            event.update(accepted=True, command='END', reason='user_end'); reason='user_9'; stop=True
                        elif key == ord('8'):
                            camera_index=(camera_index+1)%len(frozen.CAMERAS)
                            with viewer.lock(): set_camera(viewer.cam,ctx['cfg'],frozen.CAMERAS[camera_index])
                            event.update(accepted=True,command='CAMERA',reason='camera_changed')
                        elif key == ord('7'):
                            if session.phase=='waiting_start':
                                binding=jog_input.arm()
                                event['jog_input']=binding
                                if jog_input.armed:
                                    accepted,note=session.start(step)
                                else:
                                    accepted,note=False,'click_scene_then_press_7'
                            else:
                                accepted,note=session.start(step)
                            event.update(accepted=accepted,command='START',reason=note)
                            if accepted:
                                physics_started=tick; report['physics_started']=True
                        elif key == ord('6'):
                            result=session.queue_gripper_toggle()
                            event.update(result)
                            if result['accepted']:
                                session.notice=('CLOSE' if result['desired_closed'] else 'OPEN')+' requested'
                            else:
                                session.notice='6: '+input_hint(result)
                        report['events'].append(event); event_write('processed',event)
                        print('KEY:',event['key'],event['reason'],flush=True)
                        if stop: break
                    if stop: break
                    # Sample only the six direction keys in our own foreground
                    # viewer window. New input replaces old input; no backlog.
                    held = jog_input.poll()
                    direction = held['command']
                    input_state = (direction, held['reason'])
                    if input_state != previous_input:
                        event_write('jog_input',dict(sim_s=float(d.time),wall_s=time.perf_counter()-started,**held))
                        previous_input = input_state
                    if direction is None:
                        blocked_direction = None
                    session.advance(step)
                    # Accept 6 in any started phase. Commit at the current
                    # segment endpoint, ahead of another held-direction segment.
                    if session.pending_gripper_closed is not None and session.plan is None:
                        current,current_problems,_=sample(ctx,session,step,time.perf_counter()-started,mujoco,np)
                        if not current_problems:
                            planning_start=time.perf_counter()
                            result=session.apply_pending_gripper(step,current)
                            event=dict(key='6 reference',command='GRIPPER_REFERENCE',sim_s=float(d.time),
                                wall_s=time.perf_counter()-started,accepted=bool(result['accepted']),
                                reason=result['reason'],command_id=session.command_id,
                                planning_wall_s=time.perf_counter()-planning_start,
                                diagnostics=keyboard.jsonable(result.get('diagnostics',{})),
                                input_mode='manual_gripper_reference')
                            if result['accepted']:
                                path=out/'command_references'/f'{session.command_id:04d}.json'
                                save_json(path,keyboard.jsonable(dict(command='GRIPPER_TARGET',start_step=step,**result)))
                                event.update(reference_path=str(path.relative_to(out)),reference_sha256=sha(path))
                                session.notice=('CLOSING' if session.closed_requested else 'OPENING')
                            elif result['reason']=='latest_gripper_intent_already_satisfied':
                                session.notice='Latest gripper request already satisfied'
                            else:
                                session.notice='6 blocked by geometry: '+input_hint(result,gripper=session.closed_requested)
                            report['events'].append(event);event_write('reference_commit',event)
                            print('GRIPPER:',event['reason'],flush=True)
                    can_jog = (session.phase=='ready' or
                        (session.phase=='settling' and session.plan is None and session.last_plan_kind=='jog'))
                    if direction is not None and direction != blocked_direction and can_jog:
                        current,current_problems,_=sample(ctx,session,step,time.perf_counter()-started,mujoco,np)
                        if not current_problems:
                            planning_start=time.perf_counter()
                            result=session.request_jog(direction,step,current,
                                lambda: jog_input.poll()['command']==direction)
                            event=dict(key='HOLD '+direction,command=direction,sim_s=float(d.time),
                                wall_s=time.perf_counter()-started,accepted=bool(result['accepted']),
                                reason=result['reason'],command_id=session.command_id,
                                accepted_target_xyz=session.goal.tolist(),
                                planning_wall_s=time.perf_counter()-planning_start,
                                diagnostics=keyboard.jsonable(result.get('diagnostics',{})),
                                input_mode='held_direction_segment')
                            if result['accepted']:
                                reference=dict(command_id=session.command_id,command=direction,start_step=step,**result)
                                with (out/'jog_references.jsonl').open('a',encoding='utf-8') as refs:
                                    refs.write(json.dumps(keyboard.jsonable(reference),ensure_ascii=True)+'\n')
                                event['reference_path']='jog_references.jsonl'
                                session.notice=''
                            elif result['reason'] not in ('wait_for_tracking_to_settle','released_before_commit'):
                                blocked_direction=direction
                                checks=result.get('diagnostics',{}).get('checks',[])
                                detail=', '.join(checks) if checks else result['reason']
                                session.notice='BLOCKED: '+input_hint(result)
                                print('JOG BLOCKED:',direction,detail,flush=True)
                            # Waiting for tracking is transient; do not flood the
                            # event log on every frame of the same physical hold.
                            if result['reason']!='wait_for_tracking_to_settle':
                                report['events'].append(event);event_write('processed',event)
                    if session.phase != 'waiting_start':
                        for _ in range(4):
                            frozen.assert_options_unchanged(m,options,np)
                            session.advance(step)
                            row,problems,contacts=sample(ctx,session,step,time.perf_counter()-started,mujoco,np)
                            problems.extend(observe_task(ctx,session,row,step,mujoco,np))
                            saturation_s=saturation_s+DT if row['max_torque_ratio']>1. else 0.
                            if saturation_s>.25: problems.append('sustained_arm_saturation')
                            if session.phase=='settling' and (step-session.settle_step)*DT>THRESHOLDS['settling_timeout_s']:
                                problems.append('settling_timeout')
                            if problems:
                                report['stop_diagnostics']=dict(problems=problems,contacts=contacts,row=row)
                                status,reason='stopped',','.join(problems); write(writer,row,True); stop=True; break
                            if d.time>=THRESHOLDS['maximum_simulation_s']-1e-10:
                                reason='simulation_time_limit'; write(writer,row,True); stop=True; break
                            write(writer,row)
                            mujoco.mj_step(m,d)
                            step+=1; report['steps']=step
                        stream.flush()
                    display=last if last is not None else first
                    with viewer.lock(): update_scene(viewer,ctx,session,display,mujoco,np)
                    if session.phase != last_phase and session.phase not in ('moving','settling'):
                        print('PHASE:',session.phase.upper(),f'| sim {d.time:.3f}s',flush=True); last_phase=session.phase
                    if session.checks.task_stage != last_task:
                        print('TASK:',session.checks.task_stage, f'| sim {d.time:.3f}s',flush=True); last_task=session.checks.task_stage
                    before=(d.qpos.copy(),d.qvel.copy(),d.ctrl.copy(),float(d.time))
                    viewer.sync()
                    frozen.assert_options_unchanged(m,options,np)
                    if (not np.array_equal(d.qpos,before[0]) or not np.array_equal(d.qvel,before[1]) or
                        not np.array_equal(d.ctrl,before[2]) or d.time!=before[3] or
                        np.any(d.xfrc_applied) or np.any(d.qfrc_applied)):
                        raise RuntimeError('Viewer changed actual state or applied forces')
                    if stop: break
                    time.sleep(max(0.,(4*DT if physics_started is not None else .02)-(time.perf_counter()-tick)))
        finally:
            jog_input.disarm()
            if not terminal_saved:
                try:
                    row,problems,contacts=sample(ctx,session,step,time.perf_counter()-started,mujoco,np)
                    problems.extend(observe_task(ctx,session,row,step,mujoco,np))
                    if problems:
                        status,reason='stopped','terminal:'+','.join(problems)
                        report['stop_diagnostics']=dict(problems=problems,contacts=contacts,row=row)
                    write(writer,row,True)
                except Exception:
                    status,reason='error','terminal_record_failed'; report['terminal_record_error']=traceback.format_exc()
            report.update(steps=step,simulation_s=float(d.time),metrics=metrics,
                completed_command_ids=session.completed_ids,command_records=session.command_records,final_phase=session.phase,
                jog_reference_completed_ids=session.jog_reference_completed_ids,
                pending_gripper_closed=session.pending_gripper_closed,
                final_qpos=d.qpos.tolist(),final_qvel=d.qvel.tolist(),final_qacc_warmstart=d.qacc_warmstart.tolist(),
                physics_loop_wall_s=None if physics_started is None else time.perf_counter()-physics_started)
    checks=completion(session,reason,metrics,step,last)
    if checks['manual_task_completed']: status='completed'
    elif reason=='user_9': status='partial'
    report.update(status=status,end_reason=reason,**checks,
        planning_wall_s_total=sum(e.get('planning_wall_s',0.) for e in report['events']))
    return 0 if status in ('completed','partial','interrupted') else 2


def main():
    guard()
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path)
    parser.add_argument('--inspect',action='store_true')
    args=parser.parse_args()
    out=args.output.resolve() if args.output else ROOT/'outputs'/('manual_user_'+stamp())
    if out.parent != (ROOT/'outputs').resolve() or not out.name.startswith('manual_user_'):
        raise ValueError('Use a fresh flat outputs/manual_user_ directory')
    if not args.inspect: out.mkdir(exist_ok=False)
    os.chdir(ROOT/'outputs' if args.inspect else out)
    report=dict(kind='manual_pickplace_dynamics',status='started',model_loaded=False,viewer_started=False,
        physics_started=False,steps=0,data_saved_samples=0,simulation_s=0.,events=[],snapshots=[],
        manual_task_completed=False,grasp_hold_passed=False,transfer_completed=False,release_started=False,
        placement_passed=False,tracking_thresholds_passed=False,full_v01_accepted=False,
        thresholds=THRESHOLDS,placement_thresholds=task.PLACEMENT_THRESHOLDS,noslip_iterations=3,
        input_mode='foreground_hold_to_jog',jog_maximum_segment_m=.0015,jog_segment_s=.12,
        gripper_input_mode='latest_intent_at_segment_boundary_no_task_milestone_lock',
        task_scoring_mode='physical_observer_v2_independent_of_input_permission',
        jog_maximum_nominal_mean_speed_m_s=.0125,jog_nominal_quintic_peak_speed_m_s=.0234375,
        release_behavior='Finish current <=0.12 simulation second segment; discard uncommitted plans on release/focus loss.',
        controller='PD_plus_zero_velocity_gravity',kp=base.motion.KP,kd=base.motion.KD,
        software_caps=base.motion.CAP,executable=sys.executable,temp=tempfile.gettempdir(),
        package_version='0.1.0',accepted_manual_baseline='manual_user_20261021_175535_099',
        scope='User holds Cartesian directions and commands both gripper actions; checked short segments follow current key state, no scheduled carry route or physical attachment.',
        sample_semantics='Same-state pre-step samples; terminal future motor/reference velocities blank. Planned endpoint differs from interpolated FK.',
        relative_motion_semantics='One actual grip baseline; slip/rotation blank before capture and after user opening.',
        diagnostic_mode='csv_and_events_no_async_stack_sampler')
    started=time.perf_counter(); code=0
    try:
        import numpy as np
        import mujoco
        ctx=setup(mujoco,np)
        planner,inspection=inspect(ctx,mujoco,np)
        cfg=ctx['cfg']
        report.update(model_loaded=True,mujoco_version=mujoco.__version__,numpy_version=np.__version__,dt_s=DT,
            static_inspection=inspection,anchor_position_world_m=ctx['manual_anchor'].tolist(),
            target_fixed_orientation_wxyz=ctx['manual_orientation'].tolist(),planner_settings=PLANNER_SETTINGS,
            initial_qpos=ctx['data'].qpos.tolist(),initial_qvel=ctx['data'].qvel.tolist(),
            effective_solver_options=ctx['solver_effective_options'],manual_initialization=ctx['manual_initialization'],
            table_top_z_m=cfg['table']['top_z_m'],tray_floor_z_m=cfg['tray']['floor_top_z_m'],
            tray_rim_z_m=cfg['tray']['wall_top_z_m'],tray_center_xy_m=cfg['tray']['center_xy_m'],
            tray_inner_half_xy_m=cfg['tray']['interior_half_size_xy_m'])
        if args.inspect:
            print(json.dumps(keyboard.jsonable(inspection),ensure_ascii=False,indent=2)); return 0
        from package_runtime import snapshot
        snapshot(ctx, out, report)
        save_json(out/'manual_initialization.json',keyboard.jsonable(ctx['manual_initialization']))
        report['snapshots'].append(dict(path='manual_initialization.json',sha256=sha(out/'manual_initialization.json')))
        (out/'command_references').mkdir()
        print('MANUAL RUN:',out,flush=True)
        code=run(ctx,planner,report,out,mujoco,np)
    except KeyboardInterrupt:
        report.update(status='interrupted',end_reason='KeyboardInterrupt',manual_task_completed=False); code=130
    except Exception:
        report.update(status='error',end_reason='exception',manual_task_completed=False,error=traceback.format_exc()); code=1
        print(report['error'],file=sys.stderr)
        if not args.inspect: (out/'error.txt').write_text(report['error'],encoding='utf-8')
    finally:
        report['wall_s']=time.perf_counter()-started
        if not args.inspect:
            if (out/'manual.csv').exists(): report['manual_csv_sha256']=sha(out/'manual.csv')
            if (out/'jog_references.jsonl').exists(): report['jog_references_sha256']=sha(out/'jog_references.jsonl')
            save_json(out/'summary.json',keyboard.jsonable(report)); print('SUMMARY:',out/'summary.json',flush=True)
    return code


if __name__=='__main__':
    raise SystemExit(main())
