"""User-confirmed carry/release stages after the unchanged 28 s grasp protocol.

Only setup initializes the live state. References command motors; they never
overwrite the simulated arm or cube. Post-release cube motion remains physical.
"""
import csv
import queue
import time
import traceback

import grasp_lift as base
import grasp_solver_runtime as original
from grasp_contact_geometry import separated_box_lower_bound
from pickplace_layout_preview import set_camera

motion = base.motion
cube_relative = base.cube_relative

DT = base.DT
THRESHOLDS = base.THRESHOLDS
PLACEMENT_THRESHOLDS = dict(hold_s=2., evaluation_window_s=3.,
    minimum_tray_xy_margin_m=.002, minimum_floor_normal_n=.1,
    minimum_floor_clearance_m=-.001, maximum_floor_clearance_m=.002,
    minimum_finger_cube_distance_m=.003, maximum_cube_speed_m_s=.02,
    maximum_cube_angular_speed_rad_s=.2, maximum_wait_phase_s=45.,
    opening_duration_s=2., minimum_release_floor_clearance_m=.002,
    release_settle_minimum_s=1., release_settle_timeout_s=3.,
    release_settle_stable_s=.25)
PRE_PHASES = {'waiting_start', 'approach', 'B_ready', 'closing', 'grip_settle',
              'grip_ready', 'lifting', 'lift_hold'}
POST_PHASES = {'lift_ready', 'transport', 'above_tray_ready', 'descending',
               'release_ready', 'opening', 'release_settle', 'retracting', 'placement_hold'}
RELEASE_PHASES = {'opening', 'release_settle', 'retracting', 'placement_hold'}
READY_PHASES = {'lift_ready', 'above_tray_ready', 'release_ready'}


class CarryStages(original.ScheduledStages):
    def __init__(self, durations):
        super().__init__()
        self.durations = durations

    def advance(self, step):
        if self.phase in PRE_PHASES:
            super().advance(step)
        for old, new, duration in (
            ('transport', 'above_tray_ready', self.durations['transport']),
            ('descending', 'release_ready', self.durations['descend']),
            ('opening', 'release_settle', PLACEMENT_THRESHOLDS['opening_duration_s']),
            ('retracting', 'placement_hold', self.durations['retract']),
        ):
            if self.phase == old and self.age(step) >= duration-1e-10:
                self.enter(new, step)
                break

    def command(self, key, step, ready=False):
        if key != '7':
            return False, 'unsupported_command'
        if self.phase == 'waiting_start':
            self.enter('approach', step)
            return True, 'accepted_start'
        target = {'lift_ready': 'transport', 'above_tray_ready': 'descending',
                  'release_ready': 'opening'}.get(self.phase)
        if target is None:
            return False, 'wrong_phase_no_queue'
        if not ready:
            return False, 'current_joint_gate_not_ready'
        self.enter(target, step)
        return True, 'accepted_' + target


def references(ctx, stage, step, np):
    if stage.phase in PRE_PHASES:
        return base.references(ctx, stage, step, np)
    mapping = {'lift_ready': ('transport', 0), 'transport': ('transport', None),
        'above_tray_ready': ('transport', -1), 'descending': ('descend', None),
        'release_ready': ('descend', -1), 'opening': ('descend', -1), 'release_settle': ('descend', -1),
        'retracting': ('retract', None), 'placement_hold': ('retract', -1)}
    name, fixed = mapping[stage.phase]
    rows = ctx['carry_route']['segments'][name]['rows']
    index = min(step-stage.start, len(rows)-1) if fixed is None else fixed
    record = rows[index]
    q = np.asarray(record['q_ref'], dtype=float)
    qd = np.asarray(record['dq_ref'], dtype=float) if fixed is None else np.zeros(7)
    finger, velocity = base.CLOSED, 0.
    if stage.phase == 'opening':
        duration = PLACEMENT_THRESHOLDS['opening_duration_s']
        u = min(1., max(0., stage.age(step)/duration))
        finger = base.CLOSED+(base.OPEN-base.CLOSED)*u**3*(10+u*(-15+6*u))
        velocity = (base.OPEN-base.CLOSED)*30*u*u*(1-u)**2/duration
    elif stage.phase in ('release_settle', 'retracting', 'placement_hold'):
        finger = base.OPEN
    return np.r_[q, [finger, finger]], np.r_[qd, [velocity, velocity]]


def tray_fields(ctx, row, mujoco, np):
    """Read-only task geometry/contact metrics after a force evaluation."""
    m, d, tray = ctx['model'], ctx['data'], ctx['carry_route']['tray']
    cube = ctx['cube_geom']
    tr = d.xmat[tray['body_id']].reshape(3, 3)
    cr = d.geom_xmat[cube].reshape(3, 3)
    local = tr.T @ (d.geom_xpos[cube] - d.xpos[tray['body_id']])
    extent = np.abs(tr.T @ cr) @ m.geom_size[cube]
    margin = float(np.min(np.asarray(tray['inner_half_size_xy_m']) - np.abs(local[:2]) - extent[:2]))
    bottom = float(d.geom_xpos[cube][2] - np.abs(cr[2]) @ m.geom_size[cube])
    floor_count = wall_count = 0
    floor_force = 0.
    for ci, contact in enumerate(d.contact[:d.ncon]):
        pair = {int(contact.geom1), int(contact.geom2)}
        if cube not in pair:
            continue
        wrench = np.zeros(6)
        mujoco.mj_contactForce(m, d, ci, wrench)
        active = contact.efc_address >= 0 and wrench[0] > 0
        if active and tray['floor_geom_id'] in pair:
            floor_count += 1
            floor_force += float(wrench[0])
        if active and any(g in pair for g in tray['wall_geom_ids']):
            wall_count += 1
    distances = [float(mujoco.mj_geomDistance(m, d, g, cube, .1, np.zeros(6)))
                 for g in ctx['finger_geoms']]
    minimum_distance = min(distances)
    # A positive box-frame enclosing-box gap proves separation of the whole
    # compiled finger mesh (and its convex collision hull). Keep the engine
    # distance as raw diagnostics: separated mesh/box pairs can return zero.
    # Zero lower bound is inconclusive and must never certify clearance.
    lower_bounds = [separated_box_lower_bound(
        vertices @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g], m, d, cube, np)
        for g, vertices in ctx['finger_meshes']]
    minimum_lower_bound = float(np.min(lower_bounds))
    clear = (row['finger1_contact_count'] == 0 and row['finger2_contact_count'] == 0
             and np.isfinite(lower_bounds).all()
             and minimum_lower_bound >= PLACEMENT_THRESHOLDS['minimum_finger_cube_distance_m'])
    return dict(cube_tray_base_contact_count=floor_count,
        cube_tray_wall_contact_count=wall_count, cube_tray_base_normal_n=floor_force,
        cube_bottom_tray_clearance_m=bottom-tray['floor_top_z_m'],
        cube_inside_tray_xy=int(margin >= 0.), cube_tray_xy_margin_m=margin,
        minimum_finger_cube_distance_m=minimum_distance,
        minimum_finger_cube_clearance_lower_bound_m=minimum_lower_bound,
        finger_clear_of_cube=int(clear))


def placement_valid(row):
    p = PLACEMENT_THRESHOLDS
    return bool(row['release_active'] and row['cube_inside_tray_xy']
        and row['cube_tray_xy_margin_m'] >= p['minimum_tray_xy_margin_m']
        and row['cube_tray_base_contact_count'] > 0
        and row['cube_tray_base_normal_n'] >= p['minimum_floor_normal_n']
        and p['minimum_floor_clearance_m'] <= row['cube_bottom_tray_clearance_m'] <= p['maximum_floor_clearance_m']
        and row['cube_table_contact_count'] == 0 and row['cube_tray_wall_contact_count'] == 0
        and row['finger_clear_of_cube'] and row['cube_speed_m_s'] <= p['maximum_cube_speed_m_s']
        and row['cube_angular_speed_rad_s'] <= p['maximum_cube_angular_speed_rad_s']
        and row['position_error_m'] <= THRESHOLDS['position_warning_m']
        and row['orientation_error_rad'] <= THRESHOLDS['orientation_warning_rad']
        and row['unexpected_contact_count'] == 0)


def release_settle_transition(stage, row, step, timer):
    """Require current physical support before allowing the arm to withdraw."""
    valid_s = timer.observe(stage.phase == 'release_settle' and placement_valid(row), step)
    if stage.phase != 'release_settle':
        return None
    if (stage.age(step) >= PLACEMENT_THRESHOLDS['release_settle_minimum_s']-1e-10
            and valid_s >= PLACEMENT_THRESHOLDS['release_settle_stable_s']-1e-10):
        stage.enter('retracting', step)
        return None
    if stage.age(step) >= PLACEMENT_THRESHOLDS['release_settle_timeout_s']-1e-10:
        return 'release_not_settled'
    return None


def ready_valid(row, phase):
    """Carry readiness retains grasp criteria; release uses its local floor gap."""
    common = (base.bilateral(row) and row['grip_reference_valid'] == 1
        and row['relative_slip_m'] <= THRESHOLDS['maximum_relative_slip_m']
        and row['relative_rotation_rad'] <= THRESHOLDS['maximum_relative_rotation_rad']
        and row['cube_speed_m_s'] <= THRESHOLDS['max_hold_cube_speed_m_s']
        and row['cube_angular_speed_rad_s'] <= THRESHOLDS['max_hold_cube_angular_speed_rad_s']
        and row['position_error_m'] <= THRESHOLDS['position_warning_m']
        and row['orientation_error_rad'] <= THRESHOLDS['orientation_warning_rad']
        and row['unexpected_contact_count'] == 0 and row['cube_table_contact_count'] == 0
        and row['cube_tray_base_contact_count'] == 0 and row['cube_tray_wall_contact_count'] == 0)
    if phase == 'release_ready':
        return bool(common and row['cube_tray_xy_margin_m'] >= PLACEMENT_THRESHOLDS['minimum_tray_xy_margin_m']
            and row['cube_bottom_tray_clearance_m'] >= PLACEMENT_THRESHOLDS['minimum_release_floor_clearance_m'])
    return bool(common and row['cube_bottom_clearance_m'] >= THRESHOLDS['minimum_lift_m'])


def add_task_fields(ctx, stage, row, mujoco, np):
    row.update(tray_fields(ctx, row, mujoco, np))
    row.update(carry_active=int(stage.phase in POST_PHASES),
               release_active=int(stage.phase in RELEASE_PHASES),
               placement_continuous_s=0., phase_ready=0, placement_valid=0)
    row['placement_valid'] = int(placement_valid(row))
    if stage.phase in READY_PHASES:
        row['phase_ready'] = int(ready_valid(row, stage.phase))
    return row


def sample(ctx, stage, step, wall_s, mujoco, np):
    if stage.phase in PRE_PHASES:
        row, problems, contacts = base.sample(ctx, stage, step, wall_s, mujoco, np)
    else:
        row, problems, contacts = post_sample(ctx, stage, step, wall_s, mujoco, np)
    return add_task_fields(ctx, stage, row, mujoco, np), problems, contacts


def display_instruction(stage, step, ctx):
    text = {'lift_ready': 'Press 7 to carry above the tray when the joint gate is ready.',
            'above_tray_ready': 'Press 7 to descend toward the release pose when ready.',
            'release_ready': 'Press 7 to open fingers, then retract and check placement.'}
    if stage.phase in text:
        return text[stage.phase]
    if stage.phase in PRE_PHASES:
        return original.countdown_text(stage, step)
    if stage.phase == 'release_settle':
        return 'Awaiting actual tray support and finger separation before retracting.'
    duration = {'transport': ctx['carry_route']['segments']['transport']['duration_s'],
                'descending': ctx['carry_route']['segments']['descend']['duration_s'],
                'opening': PLACEMENT_THRESHOLDS['opening_duration_s'],
                'retracting': ctx['carry_route']['segments']['retract']['duration_s'],
                'placement_hold': PLACEMENT_THRESHOLDS['evaluation_window_s']}[stage.phase]
    return f'Phase time remaining: {max(0.,duration-stage.age(step)):.2f} simulation s'


def post_sample(ctx,stage,step,wall_s,mujoco,np):
    m,d,r=ctx['model'],ctx['data'],ctx['reference']
    if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
        raise RuntimeError('Nonfinite actual state')
    qref,qdref=references(ctx,stage,step,np)
    pd,gravity,raw,tau=motion.control_terms(ctx,qref,qdref,mujoco,np)
    d.ctrl[ctx['aids']]=tau
    # Same-state solver forces with this row's command; no time integration.
    mujoco.mj_forward(m,d)
    if not np.allclose(d.qfrc_actuator[ctx['va']],tau,rtol=0,atol=1e-9):
        raise RuntimeError('Actuator force and recorded motor command differ')
    r.qpos[:]=ctx['base_qpos'];r.qpos[ctx['qa']]=qref;r.qvel[:]=0
    mujoco.mj_forward(m,r)
    q=d.qpos[ctx['qa']].copy();qd=d.qvel[ctx['va']].copy()
    actual=d.site_xpos[ctx['site']].copy();ref=r.site_xpos[ctx['site']].copy()
    quat=np.empty(4);qrefquat=np.empty(4)
    mujoco.mju_mat2Quat(quat,d.site_xmat[ctx['site']]);mujoco.mju_mat2Quat(qrefquat,r.site_xmat[ctx['site']])
    angle=float(2*np.arccos(np.clip(abs(float(quat@qrefquat)),0.,1.)))
    forces=[0.,0.];counts=[0,0];table_contacts=0;unexpected=[];contact_details=[]
    cube=ctx['cube_geom'];table=ctx['table_geom']
    tray=ctx['carry_route']['tray']; release=stage.phase in RELEASE_PHASES
    for ci,c in enumerate(d.contact[:d.ncon]):
        pair={int(c.geom1),int(c.geom2)}
        if cube in pair and (table in pair or any(g in pair for g in ctx['finger_geoms']) or (release and tray['floor_geom_id'] in pair)):
            wrench=np.zeros(6);mujoco.mj_contactForce(m,d,ci,wrench)
            # efc_address excludes inactive gap contacts; positive normal indicates support.
            active=c.efc_address>=0 and wrench[0]>0
            if active:
                if table in pair:table_contacts+=1
            if table in pair and c.dist<=0:
                unexpected.append(dict(geoms=['task_cube_geom','task_table_top'],distance_m=float(c.dist),reason='table_contact_after_lift'))
            if active:
                for i,g in enumerate(ctx['finger_geoms']):
                    if g in pair:counts[i]+=1;forces[i]+=float(wrench[0])
            contact_details.append(dict(pair=sorted(pair),dist_m=float(c.dist),normal_n=float(wrench[0]),active=bool(active)))
        elif c.dist<=0:
            unexpected.append(dict(geoms=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) for g in sorted(pair)],distance_m=float(c.dist)))
    finger_gap=min(float((v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g])[:,2].min()) for g,v in ctx['finger_meshes'])-ctx['cfg']['table']['top_z_m']
    cube_position=d.geom_xpos[cube].copy();cube_rotation=d.geom_xmat[cube].reshape(3,3)
    bottom=float(cube_position[2]-np.abs(cube_rotation[2])@m.geom_size[cube])
    relpos,relrot=cube_relative(ctx,np)
    slip=rotation_error=''
    if ctx['grip_baseline'] is not None:
        oldpos,oldrot=ctx['grip_baseline']
        slip=float(np.linalg.norm(relpos-oldpos))
        rotation_error=float(np.arccos(np.clip((np.trace(oldrot.T@relrot)-1)/2.,-1.,1.)))
    cube_jid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'task_cube_free')
    if cube_jid<0:raise ValueError('Missing free cube joint')
    cv=int(m.jnt_dofadr[cube_jid]);cube_speed=float(np.linalg.norm(d.qvel[cv:cv+3]));cube_spin=float(np.linalg.norm(d.qvel[cv+3:cv+6]))
    row=dict(sim_s=float(d.time),wall_s=wall_s,phase=stage.phase,terminal_state=0,
        position_error_m=float(np.linalg.norm(actual-ref)),orientation_error_rad=angle,
        finger_ref_m=float(qref[7]),finger1_q_m=float(q[7]),finger2_q_m=float(q[8]),
        finger1_normal_n=forces[0],finger2_normal_n=forces[1],finger1_contact_count=counts[0],finger2_contact_count=counts[1],
        cube_table_contact_count=table_contacts,cube_bottom_clearance_m=bottom-ctx['cfg']['table']['top_z_m'],
        cube_lift_from_start_m=float(cube_position[2]-ctx['cube_initial'][2]),
        relative_slip_m=slip,relative_rotation_rad=rotation_error,grip_reference_valid=int(ctx['grip_baseline'] is not None),
        unexpected_contact_count=len(unexpected),max_torque_ratio=float(np.max(np.abs(raw[:7])/ctx['cap'][:7])),
        max_finger_force_ratio=float(np.max(np.abs(raw[7:])/ctx['cap'][7:])),min_finger_table_gap_m=finger_gap,
        max_joint_error_rad=float(np.max(np.abs(q[:7]-qref[:7]))),cube_speed_m_s=cube_speed,cube_angular_speed_rad_s=cube_spin,
        **dict(zip(('cube_x_m','cube_y_m','cube_z_m'),map(float,cube_position))),
        **dict(zip(('actual_x_m','actual_y_m','actual_z_m'),map(float,actual))),
        **dict(zip(('ref_x_m','ref_y_m','ref_z_m'),map(float,ref))))
    for i in range(9):
        for key,vector in (('q_ref',qref),('q',q),('qd_ref',qdref),('qd',qd),('pd',pd),('gravity',gravity),('tau_raw',raw),('tau',tau)):
            row[f'{key}_{i}']=float(vector[i])
    problems=[]
    if abs(d.time-step*DT)>1e-8:problems.append('unexpected_time')
    if row['position_error_m']>THRESHOLDS['position_stop_m'] or angle>THRESHOLDS['orientation_stop_rad']:problems.append('arm_pose_tracking_stop')
    if row['max_joint_error_rad']>.15 or np.max(np.abs(qd[:7]))>2.:problems.append('arm_joint_tracking_stop')
    if np.any(q[:7]<=ctx['limits'][:,0]+.02) or np.any(q[:7]>=ctx['limits'][:,1]-.02):problems.append('arm_limit_stop')
    if np.any(q[7:]<ctx['finger_limits'][:,0]-.0005) or np.any(q[7:]>ctx['finger_limits'][:,1]+.0005) or np.max(np.abs(qd[7:]))>.25:problems.append('finger_limit_or_speed_stop')
    if finger_gap<THRESHOLDS['min_finger_table_gap_m']:problems.append('finger_table_clearance_stop')
    if unexpected:problems.append('unexpected_contact')
    half=np.abs(cube_rotation)@m.geom_size[cube]
    table_center=np.asarray(ctx['cfg']['table']['center_xy_m'])
    table_half=np.asarray(ctx['cfg']['table']['top_half_size_xy_m'])
    if (row['cube_bottom_clearance_m']<-.003 or cube_position[2]>ctx['cfg']['table']['top_z_m']+.4
            or np.any(np.abs(cube_position[:2]-table_center)+half[:2]>table_half+.005)):
        problems.append('cube_left_carry_workspace')
    if not release and ctx['grip_baseline'] is not None and (slip>.015 or rotation_error>.25):problems.append('object_slip_stop')
    if any(w.number for w in d.warning):problems.append('mujoco_warning')
    return row,problems,dict(unexpected=unexpected,allowed=contact_details)




def run(ctx, report, out, mujoco, np):
    """Run one freshly initialized model; preserves all original safety gates."""
    import mujoco.viewer
    m, d = ctx['model'], ctx['data']
    stage = CarryStages({k: ctx['carry_route']['segments'][k]['duration_s'] for k in ('transport','descend','retract')})
    commands = queue.SimpleQueue()
    stable = base.Continuous()
    hold = base.Continuous()
    placed = base.Continuous()
    release_settled = base.Continuous()
    ready_timer = base.Continuous()
    step = 0
    start = time.perf_counter()
    loop_start = None
    last = None
    terminal_saved = False
    last_phase = None
    status, reason = 'interrupted', 'window_closed'
    arm_saturation_s = contact_loss_s = 0.
    hold_good_last = False
    camera_index = 0
    options = original.option_snapshot(m, np)
    if options['noslip_iterations'] != 3:
        raise ValueError('Carry requires the accepted noslip_iterations=3 configuration.')
    report.update(diagnostic_schedule=original.SCHEDULE.copy(),
                  diagnostic_keys={'start': '7', 'camera_cycle': '8', 'end': '9'},
                  phase_boundary_states=[], frozen_physics_options=options,
                  diagnostic_scope='User-confirmed carry stages after the unchanged grasp; not free Cartesian teleoperation.')
    first, _, _ = sample(ctx, stage, 0, 0., mujoco, np)
    metrics = dict(max_position_error_m=0., max_orientation_error_rad=0.,
                   max_relative_slip_m=0., max_held_relative_slip_m=0.,
                   max_held_relative_rotation_rad=0., maximum_cube_bottom_clearance_m=0.,
                   minimum_finger_table_gap_m=first['min_finger_table_gap_m'],
                   max_torque_ratio=0., max_finger_force_ratio=0.,
                   max_unexpected_contact_count=0)

    def write(writer, row, terminal=False):
        nonlocal terminal_saved, last
        if terminal:
            row = base.motion.terminal_row(row)
            terminal_saved = True
        writer.writerow(row)
        last = row
        report['data_saved_samples'] += 1
        for key, col in [
            ('max_position_error_m', 'position_error_m'),
            ('max_orientation_error_rad', 'orientation_error_rad'),
            ('max_relative_slip_m', 'relative_slip_m'),
            ('maximum_cube_bottom_clearance_m', 'cube_bottom_clearance_m'),
            ('max_torque_ratio', 'max_torque_ratio'),
            ('max_finger_force_ratio', 'max_finger_force_ratio'),
            ('max_unexpected_contact_count', 'unexpected_contact_count'),
        ]:
            if row[col] != '':
                metrics[key] = max(metrics[key], row[col])
        metrics['minimum_finger_table_gap_m'] = min(
            metrics['minimum_finger_table_gap_m'], row['min_finger_table_gap_m'])
        if not row['release_active'] and row['grip_reference_valid']:
            metrics['max_held_relative_slip_m'] = max(
                metrics['max_held_relative_slip_m'], row['relative_slip_m'])
            metrics['max_held_relative_rotation_rad'] = max(
                metrics['max_held_relative_rotation_rad'], row['relative_rotation_rad'])

    with (out / 'grasp.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(first))
        writer.writeheader()
        try:
            with mujoco.viewer.launch_passive(
                m, d, key_callback=commands.put, show_left_ui=False, show_right_ui=False,
            ) as viewer:
                report['viewer_started'] = True
                with viewer.lock():
                    set_camera(viewer.cam, ctx['cfg'], original.CAMERAS[camera_index])
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
                print(f'USER-CONFIRMED CARRY: noslip_iterations={options["noslip_iterations"]}', flush=True)
                print('Click scene. Top-row 7 starts; 9 ends; 8 cycles camera. Mouse camera is available.', flush=True)
                print('First complete the accepted grasp. Press 7 again only when each ready phase asks.', flush=True)
                stop = False
                while viewer.is_running():
                    tick = time.perf_counter()
                    original.assert_options_unchanged(m, options, np)
                    if tick - start > 600 or (
                        loop_start is not None and tick - loop_start > THRESHOLDS['maximum_physics_wall_s']
                    ):
                        reason = 'wall_time_limit'
                        break
                    while not commands.empty():
                        key = commands.get()
                        if key == original.CAMERA_KEY:
                            camera_index = (camera_index + 1) % len(original.CAMERAS)
                            with viewer.lock():
                                set_camera(viewer.cam, ctx['cfg'], original.CAMERAS[camera_index])
                        elif key == original.END_KEY:
                            report['events'].append(dict(key='9', accepted=True, reason='user_end',
                                                         sim_s=float(d.time), wall_s=tick-start))
                            reason, stop = 'user_9', True
                            break
                        elif key == original.START_KEY:
                            previous_phase = stage.phase
                            ready_now = False
                            if previous_phase in READY_PHASES:
                                current, current_problems, _ = sample(ctx, stage, step, tick-start, mujoco, np)
                                ready_now = (not current_problems and ready_valid(current, previous_phase)
                                             and ready_timer.since is not None
                                             and (step-ready_timer.since)*DT >= THRESHOLDS['contact_stable_s'])
                            accepted, note = stage.command('7', step, ready_now)
                            report['events'].append(dict(key='7', accepted=accepted, reason=note,
                                from_phase=previous_phase, to_phase=stage.phase,
                                sim_s=float(d.time), wall_s=tick-start))
                            print('KEY: 7', note, flush=True)
                            if accepted and previous_phase == 'waiting_start':
                                loop_start = tick
                                report['physics_started'] = True
                            if accepted and stage.phase == 'opening':
                                report['release_started'] = True
                    if stop:
                        break
                    if stage.phase != 'waiting_start':
                        for _ in range(4):
                            original.assert_options_unchanged(m, options, np)
                            enter_lift_ready = False
                            old_phase = stage.phase
                            stage.advance(step)
                            row, problems, contacts = sample(
                                ctx, stage, step, time.perf_counter()-start, mujoco, np)
                            both = base.bilateral(row)
                            stable_s = stable.observe(both and row['cube_speed_m_s'] < .01, step)
                            # Scheduled L retains the original current-bilateral and stability prerequisite.
                            if old_phase == 'grip_ready' and stage.phase == 'lifting':
                                if not (both and ctx['grip_baseline'] is not None
                                        and stable_s >= THRESHOLDS['contact_stable_s']):
                                    problems.append('scheduled_lift_bilateral_not_ready')
                            if (stage.phase == 'grip_settle' and stage.age(step) >= 1.
                                    and stable_s >= THRESHOLDS['contact_stable_s']):
                                ctx['grip_baseline'] = tuple(x.copy() for x in base.cube_relative(ctx, np))
                                report['grip_baseline'] = dict(
                                    sim_s=float(d.time), position_in_tool_m=ctx['grip_baseline'][0].tolist(),
                                    rotation_in_tool=ctx['grip_baseline'][1].tolist())
                                stage.enter('grip_ready', step)
                                row, problems, contacts = sample(
                                    ctx, stage, step, time.perf_counter()-start, mujoco, np)
                            if stage.phase == 'grip_settle' and stage.age(step) >= 3.:
                                problems.append('bilateral_grip_not_established')
                            before_settle_phase = stage.phase
                            settle_problem = release_settle_transition(stage, row, step, release_settled)
                            if before_settle_phase != stage.phase:
                                row, problems, contacts = sample(
                                    ctx, stage, step, time.perf_counter()-start, mujoco, np)
                            if settle_problem:
                                problems.append(settle_problem)
                            if (stage.phase in ({'B_ready','grip_ready'} | READY_PHASES)
                                    and stage.age(step) > THRESHOLDS['maximum_wait_phase_s']):
                                problems.append('key_wait_timeout')
                            if d.time >= THRESHOLDS['maximum_simulation_s']:
                                problems.append('simulation_time_limit')
                            arm_saturation_s = arm_saturation_s + DT if row['max_torque_ratio'] > 1 else 0.
                            if arm_saturation_s > .25:
                                problems.append('sustained_arm_saturation')
                            contact_loss_s = contact_loss_s + DT if ctx['grip_baseline'] is not None and stage.phase not in RELEASE_PHASES and not both else 0.
                            if contact_loss_s > .1:
                                problems.append('bilateral_contact_lost')
                            hold_good_last = (stage.phase == 'lift_hold' and stage.age(step) >= 1.-1e-10
                                              and base.hold_valid(row))
                            hold.observe(hold_good_last, step)
                            ready_seconds = ready_timer.observe(stage.phase in READY_PHASES and ready_valid(row,stage.phase),step)
                            if stage.phase == 'above_tray_ready' and ready_seconds >= THRESHOLDS['contact_stable_s']:
                                report['transfer_completed'] = True
                            placement_ok = (stage.phase == 'placement_hold' and stage.age(step)>=1.-1e-10
                                            and placement_valid(row))
                            placement_seconds = placed.observe(placement_ok, step)
                            row['placement_continuous_s'] = placement_seconds
                            if stage.phase != last_phase:
                                print(f'PHASE: {stage.phase} | sim {d.time:.3f} s', flush=True)
                                last_phase = stage.phase
                                report['phases'].append(dict(phase=stage.phase, sim_s=float(d.time), step=step))
                                report['phase_boundary_states'].append(original.phase_state(ctx, stage, step))
                                if stage.phase in ('closing', 'lifting'):
                                    report['events'].append(dict(
                                        key=None, accepted=not problems, reason='fixed_schedule',
                                        phase=stage.phase, sim_s=float(d.time),
                                        wall_s=time.perf_counter()-start))
                            if problems:
                                report['stop_diagnostics'] = dict(problems=problems, contacts=contacts, row=row)
                                status, reason = 'stopped', ','.join(problems)
                                write(writer, row, True)
                                stop = True
                                break
                            if stage.phase == 'lift_hold' and stage.age(step) >= 3.-1e-10:
                                report['grasp_hold_passed'] = bool(
                                    hold_good_last and hold.since is not None
                                    and (step-hold.since)*DT >= THRESHOLDS['hold_s']-1e-10)
                                if not report['grasp_hold_passed']:
                                    status, reason = 'stopped', 'initial_lift_hold_not_passed'
                                    report['stop_diagnostics'] = dict(problems=[reason], contacts=contacts, row=row)
                                    write(writer, row, True)
                                    stop = True
                                    break
                                report['initial_lift_acceptance'] = dict(step=step,sim_s=float(d.time),row=row.copy())
                                enter_lift_ready = True
                            if stage.phase == 'placement_hold' and stage.age(step) >= 3.-1e-10:
                                report['placement_passed'] = bool(placement_ok and placement_seconds>=PLACEMENT_THRESHOLDS['hold_s']-1e-10)
                                status = 'completed' if report['placement_passed'] else 'stopped'
                                reason = 'tray_placement_hold_passed' if report['placement_passed'] else 'tray_placement_hold_not_passed'
                                if not report['placement_passed']:
                                    report['stop_diagnostics'] = dict(problems=[reason], contacts=contacts, row=row)
                                write(writer,row,True)
                                stop=True
                                break
                            write(writer, row)
                            mujoco.mj_step(m, d)
                            step += 1
                            report['steps'] = step
                            if enter_lift_ready:
                                stage.enter('lift_ready', step)
                        stream.flush()
                    with viewer.lock():
                        base.motion.draw(viewer, ctx, mujoco, np)
                    display = last if last is not None else first
                    viewer.set_texts((
                        mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        f'USER-CONFIRMED CARRY | noslip_iterations={options["noslip_iterations"]}\n'
                        '7:start  9:end  8:cycle camera | Mouse camera available\n'
                        f'{stage.phase} | sim {d.time:.2f} s\n'
                        f'{display_instruction(stage, step, ctx)}\n'
                        f'Finger normal: {display["finger1_normal_n"]:.2f} / {display["finger2_normal_n"]:.2f} N\n'
                        f'Cube bottom above table: {display["cube_bottom_clearance_m"]*1000:.1f} mm\n'
                        'No attachment or state playback. Stage keys, not free Cartesian control.', ''))
                    before = (d.qpos.copy(), d.qvel.copy(), d.ctrl.copy(), float(d.time))
                    viewer.sync()
                    original.assert_options_unchanged(m, options, np)
                    if (not np.array_equal(d.qpos, before[0])
                            or not np.array_equal(d.qvel, before[1])
                            or not np.array_equal(d.ctrl, before[2]) or d.time != before[3]
                            or np.any(d.xfrc_applied) or np.any(d.qfrc_applied)):
                        raise RuntimeError('Viewer changed model/state; use only diagnostic keys/camera.')
                    if stop:
                        time.sleep(1.2)
                        break
                    time.sleep(max(0., (4*DT if loop_start is not None else .02)
                                   - (time.perf_counter()-tick)))
        finally:
            if not terminal_saved:
                try:
                    original.assert_options_unchanged(m, options, np)
                    row, _, _ = sample(ctx, stage, step, time.perf_counter()-start, mujoco, np)
                    write(writer, row, True)
                except Exception:
                    report['terminal_record_error'] = traceback.format_exc()
            report['phase_boundary_states'].append(original.phase_state(ctx, stage, step, label='terminal'))
            report.update(steps=step, simulation_s=float(d.time), metrics=metrics,
                          hold_longest_valid_s=hold.maximum_s, placement_longest_valid_s=placed.maximum_s,
                          physics_loop_wall_s=None if loop_start is None else time.perf_counter()-loop_start,
                          final_qpos=d.qpos.tolist(), final_qvel=d.qvel.tolist(),
                          final_qacc_warmstart=d.qacc_warmstart.tolist())
    report.update(status=status, end_reason=reason)
    report['tracking_thresholds_passed'] = bool(
        step > 0 and metrics['max_position_error_m'] <= THRESHOLDS['position_warning_m']
        and metrics['max_orientation_error_rad'] <= THRESHOLDS['orientation_warning_rad'])
    return 0 if status in ('completed', 'interrupted') else 2
