"""Fixed-timing contact-solver diagnostic; user starts each independent run.

Uses the frozen grasp controller, reference path, safety checks and hold gates.
This runtime changes only lesson scheduling, display keys and evidence capture.
It is not a completed keyboard-manual pick-and-place implementation.
"""
import csv
import queue
import time
import traceback

import grasp_lift as base
from pickplace_layout_preview import set_camera


DT = base.DT
THRESHOLDS = base.THRESHOLDS
START_KEY = ord('7')
CAMERA_KEY = ord('8')
END_KEY = ord('9')
CAMERAS = ('oblique', 'top', 'side')
SCHEDULE = dict(approach_s=6., B_ready_s=2., closing_s=2.,
                grip_settle_min_s=1., grip_settle_timeout_s=3.,
                grip_ready_s=10., lifting_s=4., lift_hold_s=3.)

# Complete mjOption field list in the pinned MuJoCo 3.3.7 mjmodel.h.
# Capture after setup selected noslip_iterations; do not silently restore changes.
OPTION_FIELDS = (
    'timestep', 'impratio', 'tolerance', 'ls_tolerance', 'noslip_tolerance',
    'ccd_tolerance', 'gravity', 'wind', 'magnetic', 'density', 'viscosity',
    'o_margin', 'o_solref', 'o_solimp', 'o_friction', 'integrator', 'cone',
    'jacobian', 'solver', 'iterations', 'ls_iterations', 'noslip_iterations',
    'ccd_iterations', 'disableflags', 'enableflags', 'disableactuator',
    'sdf_initpoints', 'sdf_iterations',
)


class ScheduledStages(base.Stages):
    """Same mechanical phases; fixed waits replace C/L command timings."""

    def advance(self, step):
        super().advance(step)
        if self.phase == 'B_ready' and self.age(step) >= SCHEDULE['B_ready_s'] - 1e-10:
            self.enter('closing', step)
        elif self.phase == 'grip_ready' and self.age(step) >= SCHEDULE['grip_ready_s'] - 1e-10:
            self.enter('lifting', step)

    def command(self, key, step, current_bilateral=False):
        if key == '7' and self.phase == 'waiting_start':
            self.enter('approach', step)
            return True, 'accepted'
        return False, 'wrong_phase_or_not_a_diagnostic_command'


def option_snapshot(model, np):
    return {name: np.asarray(getattr(model.opt, name)).tolist() for name in OPTION_FIELDS}


def assert_options_unchanged(model, expected, np):
    changed = [name for name in OPTION_FIELDS
               if not np.array_equal(np.asarray(getattr(model.opt, name)),
                                     np.asarray(expected[name]))]
    if changed:
        raise RuntimeError('Physics options changed during diagnostic: ' + ', '.join(changed))


def phase_state(ctx, stage, step, label='phase_entry'):
    """Record current state; never assign these snapshots back to live mjData."""
    d = ctx['data']
    baseline = ctx['grip_baseline']
    return dict(label=label, phase=stage.phase, phase_start_step=stage.start,
                step=int(step), time=float(d.time), qpos=d.qpos.tolist(),
                qvel=d.qvel.tolist(), ctrl=d.ctrl.tolist(),
                qacc_warmstart=d.qacc_warmstart.tolist(),
                grip_baseline=None if baseline is None else dict(
                    position_in_tool_m=baseline[0].tolist(),
                    rotation_in_tool=baseline[1].tolist()),
                semantics='Current state after this sample forward/control evaluation; no state playback.')


def countdown_text(stage, step):
    duration = {'approach': 6., 'B_ready': 2., 'closing': 2.,
                'grip_settle': 1., 'grip_ready': 10., 'lifting': 4., 'lift_hold': 3.}
    if stage.phase == 'waiting_start':
        return 'Ready: press top-row 7 once to start this run.'
    remaining = max(0., duration.get(stage.phase, 0.) - stage.age(step))
    if stage.phase == 'grip_settle':
        return f'Settle minimum remaining: {remaining:.2f} s; bilateral gate required.'
    return f'Phase time remaining: {remaining:.2f} simulation s'


def run(ctx, report, out, mujoco, np):
    """Run one freshly initialized model; preserves all original safety gates."""
    import mujoco.viewer
    m, d = ctx['model'], ctx['data']
    stage = ScheduledStages()
    commands = queue.SimpleQueue()
    stable = base.Continuous()
    hold = base.Continuous()
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
    options = option_snapshot(m, np)
    if options['noslip_iterations'] not in (0, 3):
        raise ValueError('This diagnostic accepts only noslip_iterations 0 or 3.')
    report.update(diagnostic_schedule=SCHEDULE.copy(),
                  diagnostic_keys={'start': '7', 'camera_cycle': '8', 'end': '9'},
                  phase_boundary_states=[], frozen_physics_options=options,
                  diagnostic_scope='Fixed-timing solver comparison; not full manual pick-and-place.')
    first, _, _ = base.sample(ctx, stage, 0, 0., mujoco, np)
    metrics = dict(max_position_error_m=0., max_orientation_error_rad=0.,
                   max_relative_slip_m=0., maximum_cube_bottom_clearance_m=0.,
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

    with (out / 'grasp.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(first))
        writer.writeheader()
        try:
            with mujoco.viewer.launch_passive(
                m, d, key_callback=commands.put, show_left_ui=False, show_right_ui=False,
            ) as viewer:
                report['viewer_started'] = True
                with viewer.lock():
                    set_camera(viewer.cam, ctx['cfg'], CAMERAS[camera_index])
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
                print(f'FIXED-TIMING DIAGNOSTIC: noslip_iterations={options["noslip_iterations"]}', flush=True)
                print('Click scene. Top-row 7 starts; 9 ends; 8 cycles camera. Mouse camera is available.', flush=True)
                print('Closing/lifting follow the saved schedule automatically. No S/C/L or 1/2/3 commands.', flush=True)
                stop = False
                while viewer.is_running():
                    tick = time.perf_counter()
                    assert_options_unchanged(m, options, np)
                    if tick - start > 600 or (
                        loop_start is not None and tick - loop_start > THRESHOLDS['maximum_physics_wall_s']
                    ):
                        reason = 'wall_time_limit'
                        break
                    while not commands.empty():
                        key = commands.get()
                        if key == CAMERA_KEY:
                            camera_index = (camera_index + 1) % len(CAMERAS)
                            with viewer.lock():
                                set_camera(viewer.cam, ctx['cfg'], CAMERAS[camera_index])
                        elif key == END_KEY:
                            report['events'].append(dict(key='9', accepted=True, reason='user_end',
                                                         sim_s=float(d.time), wall_s=tick-start))
                            reason, stop = 'user_9', True
                            break
                        elif key == START_KEY:
                            accepted, note = stage.command('7', step)
                            report['events'].append(dict(key='7', accepted=accepted, reason=note,
                                                         sim_s=float(d.time), wall_s=tick-start))
                            print('KEY: 7', note, flush=True)
                            if accepted:
                                loop_start = tick
                                report['physics_started'] = True
                    if stop:
                        break
                    if stage.phase != 'waiting_start':
                        for _ in range(4):
                            assert_options_unchanged(m, options, np)
                            old_phase = stage.phase
                            stage.advance(step)
                            row, problems, contacts = base.sample(
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
                                row, problems, contacts = base.sample(
                                    ctx, stage, step, time.perf_counter()-start, mujoco, np)
                            if stage.phase == 'grip_settle' and stage.age(step) >= 3.:
                                problems.append('bilateral_grip_not_established')
                            if (stage.phase in ('B_ready', 'grip_ready')
                                    and stage.age(step) > THRESHOLDS['maximum_wait_phase_s']):
                                problems.append('key_wait_timeout')
                            if d.time >= THRESHOLDS['maximum_simulation_s']:
                                problems.append('simulation_time_limit')
                            arm_saturation_s = arm_saturation_s + DT if row['max_torque_ratio'] > 1 else 0.
                            if arm_saturation_s > .25:
                                problems.append('sustained_arm_saturation')
                            contact_loss_s = contact_loss_s + DT if ctx['grip_baseline'] is not None and not both else 0.
                            if contact_loss_s > .1:
                                problems.append('bilateral_contact_lost')
                            hold_good_last = (stage.phase == 'lift_hold' and stage.age(step) >= 1.-1e-10
                                              and base.hold_valid(row))
                            hold.observe(hold_good_last, step)
                            if stage.phase != last_phase:
                                print(f'PHASE: {stage.phase} | sim {d.time:.3f} s', flush=True)
                                last_phase = stage.phase
                                report['phases'].append(dict(phase=stage.phase, sim_s=float(d.time), step=step))
                                report['phase_boundary_states'].append(phase_state(ctx, stage, step))
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
                                status, reason = 'completed', 'lift_hold_window_completed'
                                report['grasp_hold_passed'] = bool(
                                    hold_good_last and hold.since is not None
                                    and (step-hold.since)*DT >= THRESHOLDS['hold_s']-1e-10)
                                write(writer, row, True)
                                stop = True
                                break
                            write(writer, row)
                            mujoco.mj_step(m, d)
                            step += 1
                            report['steps'] = step
                        stream.flush()
                    with viewer.lock():
                        base.motion.draw(viewer, ctx, mujoco, np)
                    display = last if last is not None else first
                    viewer.set_texts((
                        mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        f'SOLVER DIAGNOSTIC | noslip_iterations={options["noslip_iterations"]}\n'
                        '7:start  9:end  8:cycle camera | Mouse camera available\n'
                        f'{stage.phase} | sim {d.time:.2f} s\n'
                        f'{countdown_text(stage, step)}\n'
                        f'Finger normal: {display["finger1_normal_n"]:.2f} / {display["finger2_normal_n"]:.2f} N\n'
                        f'Cube bottom above table: {display["cube_bottom_clearance_m"]*1000:.1f} mm\n'
                        'Fixed schedule; not full manual grasp/transfer/release.', ''))
                    before = (d.qpos.copy(), d.qvel.copy(), d.ctrl.copy(), float(d.time))
                    viewer.sync()
                    assert_options_unchanged(m, options, np)
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
                    assert_options_unchanged(m, options, np)
                    row, _, _ = base.sample(ctx, stage, step, time.perf_counter()-start, mujoco, np)
                    write(writer, row, True)
                except Exception:
                    report['terminal_record_error'] = traceback.format_exc()
            report['phase_boundary_states'].append(phase_state(ctx, stage, step, label='terminal'))
            report.update(steps=step, simulation_s=float(d.time), metrics=metrics,
                          hold_longest_valid_s=hold.maximum_s,
                          physics_loop_wall_s=None if loop_start is None else time.perf_counter()-loop_start,
                          final_qpos=d.qpos.tolist(), final_qvel=d.qvel.tolist(),
                          final_qacc_warmstart=d.qacc_warmstart.tolist())
    report.update(status=status, end_reason=reason)
    report['tracking_thresholds_passed'] = bool(
        status == 'completed' and metrics['max_position_error_m'] <= THRESHOLDS['position_warning_m']
        and metrics['max_orientation_error_rad'] <= THRESHOLDS['orientation_warning_rad'])
    return 0 if status in ('completed', 'interrupted') else 2
