"""Ordered, read-only acceptance for manually commanded contact pick-and-place.

This module commands no motion and never touches mjData. The caller notifies it
only of ACCEPTED user commands and feeds one same-state row per physics step.
The established grasp/placement gates are imported, rather than weakened here.
"""
import math

import grasp_lift as grasp
from carry_runtime import PLACEMENT_THRESHOLDS, placement_valid


class ContinuousSamples:
    """Elapsed simulation time between consecutive valid sampled states."""

    def __init__(self, dt):
        self.dt = dt
        self.since = None
        self.maximum_s = 0.0

    def reset(self):
        self.since = None

    def observe(self, valid, step):
        if not valid:
            self.reset()
            return 0.0
        if self.since is None:
            self.since = step
        duration = (step - self.since) * self.dt
        self.maximum_s = max(self.maximum_s, duration)
        return duration


class ManualTaskChecks:
    """Acceptance state; notifications record intent, observations prove results.

    Integration order for a physical sample:
      result = checks.observe(row, step)
      if result['need_capture_baseline']:
          # Caller copies current cube/tool relative transform into ctx.
          checks.baseline_captured(step)
          # Re-sample relative fields if needed, then observe again at same step.
      row.update(result['row_fields'])

    notify_gripper(False, ...) means a user-accepted OPEN command, never initial
    open initialization. notify_retraction() means a user-accepted upward command
    after opening. An actual >=10 mm tool rise must also be observed afterward.
    Accepted flags are historical milestones; stop_reasons always veto success.
    """

    REQUIRED = (
        'position_error_m', 'orientation_error_rad', 'unexpected_contact_count',
        'finger1_contact_count', 'finger2_contact_count',
        'finger1_normal_n', 'finger2_normal_n', 'finger_ref_m',
        'max_joint_speed_rad_s', 'max_finger_speed_m_s',
        'cube_speed_m_s', 'cube_angular_speed_rad_s',
        'cube_table_contact_count', 'cube_bottom_clearance_m',
        'cube_tray_base_contact_count', 'cube_tray_wall_contact_count',
        'cube_tray_base_normal_n', 'cube_bottom_tray_clearance_m',
        'cube_inside_tray_xy', 'cube_tray_xy_margin_m',
        'finger_clear_of_cube', 'actual_z_m',
    )

    def __init__(self, tray_rim_above_floor_m, dt=grasp.DT,
                 above_rim_gap_m=0.005, minimum_retract_rise_m=0.010):
        if not all(math.isfinite(v) and v > 0 for v in
                   (dt, tray_rim_above_floor_m, above_rim_gap_m,
                    minimum_retract_rise_m)):
            raise ValueError('Task geometry and timestep must be positive finite values')
        self.dt = float(dt)
        self.tray_rim_above_floor_m = float(tray_rim_above_floor_m)
        self.above_rim_gap_m = float(above_rim_gap_m)
        self.minimum_retract_rise_m = float(minimum_retract_rise_m)
        self.contact_timer = ContinuousSamples(dt)
        self.hold_timer = ContinuousSamples(dt)
        self.above_timer = ContinuousSamples(dt)
        self.placement_timer = ContinuousSamples(dt)
        self.loss_timer = ContinuousSamples(dt)
        self.last_step = None
        self.manual_close_requested = False
        self.baseline_active = False
        self.capture_requested_step = None
        self.grasp_hold_passed = False
        self.transfer_completed = False
        self.descent_requested = False
        self.descent_observed = False
        self.above_tool_z_m = None
        self.release_started = False
        self.retraction_requested = False
        self.retraction_observed = False
        self.placement_passed = False
        self.release_tool_z_m = None
        self.milestones = {}
        self.stop_reasons = []

    def _mark(self, name, step):
        self.milestones.setdefault(name, dict(step=int(step), sim_s=step*self.dt))

    def _fault(self, reason):
        if reason not in self.stop_reasons:
            self.stop_reasons.append(reason)

    def notify_gripper(self, closed, step, row=None):
        """Record accepted manual close/open commands; this moves nothing."""
        if closed:
            if self.manual_close_requested or self.release_started:
                self._fault('manual_grasp_sequence_restarted')
                return
            self.manual_close_requested = True
            self._mark('manual_close', step)
        else:
            if self.release_started:
                self._fault('manual_release_sequence_restarted')
                return
            if not (self.baseline_active and self.grasp_hold_passed
                    and self.transfer_completed):
                self._fault('manual_release_before_verified_transfer')
            allowed, reason = self.can_open(row) if row is not None else (False, 'missing_release_row')
            if not allowed:
                self._fault('manual_release_gate:' + reason)
            self.release_started = True
            self._mark('manual_open', step)
            if row is not None:
                value = float(row['actual_z_m'])
                if not math.isfinite(value):
                    self._fault('invalid_release_tool_height')
                else:
                    self.release_tool_z_m = value

    def notify_descent(self, step):
        """Record accepted user Z- after carrying above the tray; move nothing."""
        if not self.transfer_completed or self.release_started:
            self._fault('manual_descent_out_of_sequence')
            return
        self.descent_requested = True
        self._mark('manual_descent', step)

    def can_open(self, row):
        """Return (allowed, reason) for manual release, using the current row.

        The user must first descend from the verified above-rim location. The
        whole cube must be inside the tray, resting in the grasp at 2--10 mm
        above its floor. This is a command gate, never an automatic OPEN.
        """
        if self.stop_reasons:
            return False, 'task_stop_recorded'
        if self.release_started:
            return False, 'release_already_started'
        if not (self.baseline_active and self.grasp_hold_passed and self.transfer_completed):
            return False, 'grasp_or_transfer_not_verified'
        if not (self.descent_requested and self.descent_observed):
            return False, 'manual_descent_not_verified'
        try:
            for key in self.REQUIRED + ('relative_slip_m', 'relative_rotation_rad', 'grip_reference_valid'):
                if not math.isfinite(float(row[key])):
                    return False, 'invalid_release_fields'
            if str(row.get('phase', '')).lower() != 'ready':
                return False, 'not_ready'
            if not grasp.bilateral(row) or row['grip_reference_valid'] != 1:
                return False, 'bilateral_grasp_not_ready'
            t = grasp.THRESHOLDS
            if (row['relative_slip_m'] > t['maximum_relative_slip_m']
                    or row['relative_rotation_rad'] > t['maximum_relative_rotation_rad']):
                return False, 'grasp_relative_motion_not_ready'
            if (row['position_error_m'] > t['position_warning_m']
                    or row['orientation_error_rad'] > t['orientation_warning_rad']
                    or row['unexpected_contact_count'] != 0):
                return False, 'tracking_or_contact_not_ready'
            if (row['cube_speed_m_s'] > t['max_hold_cube_speed_m_s']
                    or row['cube_angular_speed_rad_s'] > t['max_hold_cube_angular_speed_rad_s']
                    or row['max_joint_speed_rad_s'] > .03
                    or row['max_finger_speed_m_s'] > .002):
                return False, 'release_state_not_settled'
            if (row['cube_inside_tray_xy'] != 1
                    or row['cube_tray_xy_margin_m'] < PLACEMENT_THRESHOLDS['minimum_tray_xy_margin_m']):
                return False, 'whole_cube_not_inside_tray'
            if (row['cube_table_contact_count'] != 0
                    or row['cube_tray_base_contact_count'] != 0
                    or row['cube_tray_wall_contact_count'] != 0):
                return False, 'release_cube_already_in_environment_contact'
            if not .002 <= row['cube_bottom_tray_clearance_m'] <= .010:
                return False, 'release_height_outside_2_to_10_mm'
        except (KeyError, ValueError, TypeError):
            return False, 'invalid_release_fields'
        return True, 'manual_release_ready'

    def notify_retraction(self, step):
        """Notify an accepted upward user move; actual rise is checked later."""
        if not self.release_started:
            self._fault('manual_retraction_before_release')
            return
        self.retraction_requested = True
        self._mark('manual_retraction', step)

    def baseline_captured(self, step):
        """Caller confirms copying the actual relative transform, exactly once."""
        if self.baseline_active or self.capture_requested_step != step:
            raise ValueError('Grip baseline capture must follow a fresh capture request')
        self.baseline_active = True
        self._mark('grip_baseline', step)
        self.capture_requested_step = None

    def observe(self, row, step):
        """Pure-field checks, with continuous durations measured in physics steps.

        phase must be ready/moving/settling (case-insensitive). Re-observing the
        same step after baseline capture is supported and cannot add elapsed
        time. Missing steps reset all running timers, never bridge unseen data.
        """
        if not isinstance(step, int) or step < 0:
            raise ValueError('step must be a nonnegative integer')
        if self.last_step is not None:
            if step < self.last_step:
                raise ValueError('Task samples must not go backwards')
            if step > self.last_step + 1:
                for timer in self._timers():
                    timer.reset()
        self.last_step = step
        fields = dict(grip_contact_stable_s=0.0, grasp_hold_continuous_s=0.0,
                      above_tray_continuous_s=0.0, placement_continuous_s=0.0,
                      placement_valid=0, grasp_hold_valid=0, above_tray_valid=0,
                      release_active=int(self.release_started))
        if self.release_started:
            fields.update(relative_slip_m='', relative_rotation_rad='',
                          grip_reference_valid=0)
        invalid = []
        for key in self.REQUIRED:
            try:
                if not math.isfinite(float(row[key])):
                    invalid.append(key)
            except (KeyError, ValueError, TypeError):
                invalid.append(key)
        phase = str(row.get('phase', '')).lower()
        if phase not in {'ready', 'moving', 'settling', 'waiting_start', 'done'}:
            invalid.append('phase')
        if self.baseline_active and not self.release_started:
            for key in ('relative_slip_m', 'relative_rotation_rad', 'grip_reference_valid'):
                try:
                    if not math.isfinite(float(row[key])):
                        invalid.append(key)
                except (KeyError, ValueError, TypeError):
                    invalid.append(key)
            if not invalid and row['grip_reference_valid'] != 1:
                invalid.append('grip_reference_valid')
        if invalid:
            self._fault('invalid_task_fields:' + ','.join(invalid))
            for timer in self._timers():
                timer.reset()
            return self._result(fields, False)

        t = grasp.THRESHOLDS
        p = PLACEMENT_THRESHOLDS
        both = grasp.bilateral(row)
        tracking = (row['position_error_m'] <= t['position_warning_m']
                    and row['orientation_error_rad'] <= t['orientation_warning_rad']
                    and row['unexpected_contact_count'] == 0)
        still = (row['cube_speed_m_s'] <= t['max_hold_cube_speed_m_s']
                 and row['cube_angular_speed_rad_s'] <= t['max_hold_cube_angular_speed_rad_s']
                 and row['max_joint_speed_rad_s'] <= 0.03
                 and row['max_finger_speed_m_s'] <= 0.002)
        # A blocked physical close is expected: do not compare actual finger
        # coordinates against CLOSED to decide whether contact is established.
        close_finished = abs(row['finger_ref_m'] - grasp.CLOSED) <= 1e-9
        contact_ok = (self.manual_close_requested and not self.release_started
                      and phase in {'ready', 'settling'} and close_finished
                      and both and tracking and still)
        stable_s = self.contact_timer.observe(contact_ok, step)
        fields['grip_contact_stable_s'] = stable_s
        need_capture = (not self.baseline_active and contact_ok
                        and stable_s >= t['contact_stable_s'] - 1e-10
                        and not self.stop_reasons)
        if need_capture:
            self.capture_requested_step = step

        hold_ok = False
        above_ok = False
        if self.baseline_active and not self.release_started:
            loss_s = self.loss_timer.observe(not both, step)
            if loss_s > 0.1 + 1e-10:
                self._fault('bilateral_contact_lost')
            if row['relative_slip_m'] > 0.015 or row['relative_rotation_rad'] > 0.25:
                self._fault('object_slip_stop')
            hold_ok = (phase == 'ready' and still and grasp.hold_valid(row)
                       and row['cube_tray_base_contact_count'] == 0
                       and row['cube_tray_wall_contact_count'] == 0)
            hold_s = self.hold_timer.observe(hold_ok, step)
            fields['grasp_hold_continuous_s'] = hold_s
            if hold_ok and hold_s >= t['hold_s'] - 1e-10 and not self.stop_reasons:
                self.grasp_hold_passed = True
                self._mark('grasp_hold', step)
            above_ok = (self.grasp_hold_passed and hold_ok
                        and row['cube_inside_tray_xy'] == 1
                        and row['cube_tray_xy_margin_m'] >= p['minimum_tray_xy_margin_m']
                        and row['cube_bottom_tray_clearance_m'] >=
                        self.tray_rim_above_floor_m + self.above_rim_gap_m)
            above_s = self.above_timer.observe(above_ok, step)
            fields['above_tray_continuous_s'] = above_s
            if above_ok and above_s >= t['contact_stable_s'] - 1e-10 and not self.stop_reasons:
                self.transfer_completed = True
                if self.above_tool_z_m is None:
                    self.above_tool_z_m = float(row['actual_z_m'])
                self._mark('above_tray', step)
            if (self.descent_requested and self.above_tool_z_m is not None
                    and row['actual_z_m'] <= self.above_tool_z_m - .005):
                self.descent_observed = True
                self._mark('descent_observed', step)
        else:
            self.loss_timer.reset()
            self.hold_timer.reset()
            self.above_timer.reset()
        fields['grasp_hold_valid'] = int(hold_ok)
        fields['above_tray_valid'] = int(above_ok)

        if self.release_started:
            if self.release_tool_z_m is None:
                self.release_tool_z_m = float(row['actual_z_m'])
            if (self.retraction_requested and row['actual_z_m'] - self.release_tool_z_m
                    >= self.minimum_retract_rise_m - 1e-10):
                self.retraction_observed = True
                self._mark('retraction_observed', step)
            opened = abs(row['finger_ref_m'] - grasp.OPEN) <= 1e-9
            placement_row = dict(row, release_active=1)
            placed_ok = (self.transfer_completed and self.retraction_observed
                         and phase == 'ready' and opened and still
                         and placement_valid(placement_row) and not self.stop_reasons)
            duration = self.placement_timer.observe(placed_ok, step)
            fields['placement_valid'] = int(placed_ok)
            fields['placement_continuous_s'] = duration
            if placed_ok and duration >= p['hold_s'] - 1e-10:
                self.placement_passed = True
                self._mark('placement', step)
        else:
            self.placement_timer.reset()
        return self._result(fields, need_capture)

    def _timers(self):
        return (self.contact_timer, self.hold_timer, self.above_timer,
                self.placement_timer, self.loss_timer)

    @property
    def task_stage(self):
        if self.placement_passed:
            return 'placement_complete'
        if self.retraction_requested:
            return 'retract_and_settle'
        if self.release_started:
            return 'release'
        if self.transfer_completed:
            return 'descend_for_release'
        if self.grasp_hold_passed:
            return 'carry_above_tray'
        if self.baseline_active:
            return 'lift_and_hold'
        if self.manual_close_requested:
            return 'close_and_settle'
        return 'approach_cube'

    def _result(self, fields, need_capture):
        fields.update(task_stage=self.task_stage,
                      grasp_hold_passed=int(self.grasp_hold_passed),
                      transfer_completed=int(self.transfer_completed),
                      placement_passed=int(self.placement_passed),
                      grip_baseline_captured=int(self.baseline_active),
                      descent_observed=int(self.descent_observed),
                      retraction_observed=int(self.retraction_observed))
        return dict(need_capture_baseline=bool(need_capture),
                    stop_reasons=list(self.stop_reasons),
                    grasp_hold_passed=self.grasp_hold_passed,
                    transfer_completed=self.transfer_completed,
                    placement_passed=self.placement_passed,
                    row_fields=fields)

    def summary(self):
        return dict(grasp_hold_passed=self.grasp_hold_passed,
                    transfer_completed=self.transfer_completed,
                    release_started=self.release_started,
                    descent_observed=self.descent_observed,
                    placement_passed=self.placement_passed,
                    retraction_observed=self.retraction_observed,
                    hold_longest_valid_s=self.hold_timer.maximum_s,
                    placement_longest_valid_s=self.placement_timer.maximum_s,
                    manual_task_milestones=dict(self.milestones),
                    manual_task_stop_reasons=list(self.stop_reasons),
                    minimum_retract_rise_m=self.minimum_retract_rise_m,
                    minimum_above_rim_gap_m=self.above_rim_gap_m)
