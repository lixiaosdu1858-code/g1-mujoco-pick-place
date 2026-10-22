"""Physical evidence observer for freely commanded manual pick-and-place.

This module never grants or denies a gripper/motion command. Opening, closing
empty fingers, releasing early, and starting another attempt are valid intent.
They are recorded, while success still requires measured contact, lift, carry,
release, and two seconds of the original physical tray-placement conditions.
No model state is changed here. The runtime owns the actual grip transform.
"""
import math

import grasp_lift as grasp
from carry_runtime import PLACEMENT_THRESHOLDS, placement_valid
from manual_task_checks import ContinuousSamples, ManualTaskChecks


class FreeManualTaskObserver(ManualTaskChecks):
    """API-compatible observations, without the former command-sequence gates.

    After *each accepted* gripper command, the caller must clear its old grip
    transform. ``notify_gripper`` returns ``reset_grip_baseline=True`` to make
    that contract explicit. A later ``observe`` requests a fresh capture only
    after measured bilateral contact has remained stable for 0.25 seconds.

    ``grasp_hold_passed`` remains the separate 2-second lifted hold statistic.
    A human need not pause for this statistic before carrying or opening.
    ``placement_passed`` instead requires the same attempt's actual grasp,
    >=20 mm lift, >=5 mm horizontal carried displacement into the tray, user
    opening, and the original tray support/clearance/velocity test for 2 s.
    """

    REQUIRED = ManualTaskChecks.REQUIRED + ('cube_x_m', 'cube_y_m', 'cube_z_m')

    def __init__(self, tray_rim_above_floor_m, dt=grasp.DT, **kwargs):
        super().__init__(tray_rim_above_floor_m, dt=dt, **kwargs)
        self.attempt_id = 0
        self.attempt_history = []
        self.intent_history = []
        self.grasp_observed = False
        self.lift_observed = False
        self.grasp_origin_xyz_m = None
        self.capture_origin_xyz_m = None
        self.release_had_verified_transport = False
        self.minimum_carried_xy_displacement_m = 0.005

    def _attempt_summary(self):
        return dict(attempt_id=self.attempt_id,
                    grasp_observed=self.grasp_observed,
                    lift_observed=self.lift_observed,
                    grasp_hold_passed=self.grasp_hold_passed,
                    transfer_completed=self.transfer_completed,
                    release_started=self.release_started,
                    release_had_verified_transport=self.release_had_verified_transport,
                    placement_passed=self.placement_passed,
                    hold_longest_valid_s=self.hold_timer.maximum_s,
                    placement_longest_valid_s=self.placement_timer.maximum_s,
                    milestones=dict(self.milestones))

    def _start_attempt(self):
        if self.attempt_id:
            self.attempt_history.append(self._attempt_summary())
        self.attempt_id += 1
        self.grasp_observed = False
        self.lift_observed = False
        self.grasp_origin_xyz_m = None
        self.capture_origin_xyz_m = None
        self.grasp_hold_passed = False
        self.transfer_completed = False
        self.descent_requested = False
        self.descent_observed = False
        self.above_tool_z_m = None
        self.release_started = False
        self.release_had_verified_transport = False
        self.retraction_requested = False
        self.retraction_observed = False
        self.placement_passed = False
        self.release_tool_z_m = None
        self.milestones = {}
        self.contact_timer = ContinuousSamples(self.dt)
        self.hold_timer = ContinuousSamples(self.dt)
        self.above_timer = ContinuousSamples(self.dt)
        self.placement_timer = ContinuousSamples(self.dt)
        self.loss_timer = ContinuousSamples(self.dt)
        # Actual safety faults remain recorded across attempts; user toggling
        # must not erase a physical stop that the runtime has already detected.

    def notify_gripper(self, closed, step, row=None):
        """Record user intent; any accepted OPEN/CLOSE is allowed by scoring."""
        if closed:
            self._start_attempt()
            self.manual_close_requested = True
            self._mark('manual_close', step)
        else:
            # Capture eligibility before invalidating the active grip reference.
            if not self.release_started:
                self.release_had_verified_transport = bool(
                    self.baseline_active and self.grasp_observed
                    and self.lift_observed and self.transfer_completed)
            self.manual_close_requested = False
            self.release_started = True
            self._mark('manual_open', step)
            if row is not None:
                try:
                    height = float(row['actual_z_m'])
                    if math.isfinite(height):
                        self.release_tool_z_m = height
                except (KeyError, ValueError, TypeError):
                    pass
        self.baseline_active = False
        self.capture_requested_step = None
        self.capture_origin_xyz_m = None
        self.contact_timer.reset()
        self.loss_timer.reset()
        self.hold_timer.reset()
        self.above_timer.reset()
        self.intent_history.append(dict(command='close' if closed else 'open',
                                        step=int(step), sim_s=step*self.dt,
                                        attempt_id=self.attempt_id))
        return dict(reset_grip_baseline=True, attempt_id=self.attempt_id)

    def can_open(self, row=None):
        """Scoring is deliberately not a command permission mechanism."""
        return True, 'manual_open_allowed'

    def notify_descent(self, step):
        self.descent_requested = True
        self._mark('manual_descent', step)

    def notify_retraction(self, step):
        self.retraction_requested = True
        self._mark('manual_retraction', step)

    def baseline_captured(self, step):
        if (self.baseline_active or self.capture_requested_step != step
                or self.capture_origin_xyz_m is None
                or not self.manual_close_requested or self.release_started):
            raise ValueError('Grip baseline capture must follow a fresh contact request')
        self.baseline_active = True
        self.grasp_observed = True
        self.grasp_origin_xyz_m = tuple(self.capture_origin_xyz_m)
        self.capture_requested_step = None
        self.capture_origin_xyz_m = None
        self._mark('grip_baseline', step)

    def observe(self, row, step):
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
        if not self.baseline_active or self.release_started:
            fields.update(relative_slip_m='', relative_rotation_rad='',
                          grip_reference_valid=0)
        invalid = []
        for key in self.REQUIRED:
            try:
                if not math.isfinite(float(row[key])):
                    invalid.append(key)
            except (KeyError, ValueError, TypeError):
                invalid.append(key)
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
                 and row['max_joint_speed_rad_s'] <= .03
                 and row['max_finger_speed_m_s'] <= .002)
        contact_ok = (self.manual_close_requested and not self.release_started
                      and abs(row['finger_ref_m'] - grasp.CLOSED) <= 1e-9
                      and both and tracking and row['max_finger_speed_m_s'] <= .002)
        stable_s = self.contact_timer.observe(contact_ok, step)
        fields['grip_contact_stable_s'] = stable_s
        need_capture = (not self.baseline_active and contact_ok
                        and stable_s >= t['contact_stable_s'] - 1e-10
                        and not self.stop_reasons)
        if need_capture:
            self.capture_requested_step = step
            self.capture_origin_xyz_m = tuple(float(row['cube_'+axis+'_m'])
                                               for axis in 'xyz')

        hold_ok = False
        above_ok = False
        if self.baseline_active and not self.release_started:
            loss_s = self.loss_timer.observe(not both, step)
            if loss_s > .1 + 1e-10:
                self._fault('bilateral_contact_lost')
            if row['relative_slip_m'] > .015 or row['relative_rotation_rad'] > .25:
                self._fault('object_slip_stop')
            stable_grip = (both and tracking
                           and row['relative_slip_m'] <= t['maximum_relative_slip_m']
                           and row['relative_rotation_rad'] <= t['maximum_relative_rotation_rad'])
            off_support = (row['cube_table_contact_count'] == 0
                           and row['cube_tray_base_contact_count'] == 0
                           and row['cube_tray_wall_contact_count'] == 0)
            raised = (self.grasp_origin_xyz_m is not None
                      and row['cube_z_m'] - self.grasp_origin_xyz_m[2]
                      >= t['minimum_lift_m'] - 1e-10
                      and row['cube_bottom_clearance_m'] >= t['minimum_lift_m'])
            if stable_grip and off_support and raised and not self.stop_reasons:
                self.lift_observed = True
                self._mark('lift_observed', step)
            hold_ok = (still and grasp.hold_valid(row) and off_support)
            hold_s = self.hold_timer.observe(hold_ok, step)
            fields['grasp_hold_continuous_s'] = hold_s
            if hold_ok and hold_s >= t['hold_s'] - 1e-10 and not self.stop_reasons:
                self.grasp_hold_passed = True
                self._mark('grasp_hold', step)
            carried_xy = (math.hypot(row['cube_x_m'] - self.grasp_origin_xyz_m[0],
                                     row['cube_y_m'] - self.grasp_origin_xyz_m[1])
                          if self.grasp_origin_xyz_m is not None else 0.0)
            above_ok = (self.lift_observed and stable_grip
                        and carried_xy >= self.minimum_carried_xy_displacement_m
                        and row['cube_inside_tray_xy'] == 1
                        and row['cube_tray_xy_margin_m'] >= p['minimum_tray_xy_margin_m'])
            fields['above_tray_continuous_s'] = self.above_timer.observe(above_ok, step)
            if above_ok and not self.stop_reasons:
                self.transfer_completed = True
                self.above_tool_z_m = float(row['actual_z_m'])
                self._mark('transfer_observed', step)
            if (self.descent_requested and self.above_tool_z_m is not None
                    and row['actual_z_m'] <= self.above_tool_z_m - .005):
                self.descent_observed = True
        else:
            self.loss_timer.reset()
            self.hold_timer.reset()
            self.above_timer.reset()
        fields['grasp_hold_valid'] = int(hold_ok)
        fields['above_tray_valid'] = int(above_ok)

        if self.release_started:
            if (self.retraction_requested and self.release_tool_z_m is not None
                    and row['actual_z_m'] > self.release_tool_z_m):
                self.retraction_observed = True
            placed_ok = (self.release_had_verified_transport
                         and abs(row['finger_ref_m'] - grasp.OPEN) <= 1e-9
                         and placement_valid(dict(row, release_active=1))
                         and not self.stop_reasons)
            duration = self.placement_timer.observe(placed_ok, step)
            fields['placement_valid'] = int(placed_ok)
            fields['placement_continuous_s'] = duration
            if placed_ok and duration >= p['hold_s'] - 1e-10:
                self.placement_passed = True
                self._mark('placement', step)
        else:
            self.placement_timer.reset()
        return self._result(fields, need_capture)

    @property
    def task_stage(self):
        if self.placement_passed:
            return 'placement_observed'
        if self.release_started:
            return 'gripper_open'
        if self.transfer_completed:
            return 'cube_inside_tray_xy'
        if self.lift_observed:
            return 'lift_observed'
        if self.baseline_active:
            return 'bilateral_grasp_observed'
        if self.manual_close_requested:
            return 'gripper_closed'
        return 'gripper_open'

    def _result(self, fields, need_capture):
        result = super()._result(fields, need_capture)
        result['row_fields'].update(attempt_id=self.attempt_id,
                                    grasp_observed=int(self.grasp_observed),
                                    lift_observed=int(self.lift_observed))
        return result

    def summary(self):
        return dict(grasp_hold_passed=self.grasp_hold_passed,
                    grasp_observed=self.grasp_observed,
                    lift_observed=self.lift_observed,
                    transfer_completed=self.transfer_completed,
                    release_started=self.release_started,
                    placement_passed=self.placement_passed,
                    descent_observed=self.descent_observed,
                    retraction_observed=self.retraction_observed,
                    hold_longest_valid_s=self.hold_timer.maximum_s,
                    placement_longest_valid_s=self.placement_timer.maximum_s,
                    manual_task_milestones=dict(self.milestones),
                    manual_task_stop_reasons=list(self.stop_reasons),
                    manual_attempt_id=self.attempt_id,
                    manual_attempt_history=list(self.attempt_history),
                    manual_gripper_intents=list(self.intent_history),
                    manual_observer_semantics='physical_evidence_independent_of_command_permissions',
                    minimum_carried_xy_displacement_m=self.minimum_carried_xy_displacement_m)
