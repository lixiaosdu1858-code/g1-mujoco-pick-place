"""NumPy-only box-constrained IK steps, prepared for a learner-run comparison.

This is a small local active-set implementation, not Mink. It changes the
linearized step optimization, not the task, geometry, pose metric or outer
line search. No model is loaded and no experiment runs on import.
"""
import numpy as np

from ik_seed_compare import CSV_FIELDS, POINT_IDS, SOLVER_SETTINGS, boundary_report


BOX_SETTINGS = {
    **SOLVER_SETTINGS,
    'algorithm': 'box_qp_active_set',
    'step_objective': '0.5 * step.T @ (J.T @ J + 1e-6*I) @ step - (J.T @ error).T @ step',
    'step_bounds': 'max(qmin+0.03-q,-0.12) <= step <= min(qmax-0.03-q,0.12)',
    'qp_max_iterations': 256, 'qp_kkt_atol': 1e-10, 'qp_kkt_rtol': 1e-9,
    'method_difference': 'Joint and infinity-norm step bounds enter the quadratic problem; no post-solve direction rescaling or substantive clipping.',
    'unchanged': ['model', 'target pose', 'initial arm seed', 'non-arm coordinates',
                  'pose error and Jacobian convention', 'orientation weight',
                  'damping', 'strict/task tolerances', 'outer budget', 'line-search scales'],
    'not_implemented': ['posture/nullspace regularization', 'collision avoidance',
                        'global IK', 'trajectory planning', 'dynamics', 'grasp control'],
}


def solve_box_qp(H, g, lower, upper, *, max_iterations=256,
                 kkt_atol=1e-10, kkt_rtol=1e-9):
    """Solve a strictly convex box QP using a feasible primal active set.

    Minimize .5*x.T@H@x + g.T@x subject to lower <= x <= upper.
    H must be symmetric positive definite. Invalid/non-finite input raises;
    numerical or iteration failure returns a feasible iterate with status other
    than 'optimal', which callers MUST NOT use as a certified solution.

    On a lower bound the gradient must be nonnegative, on an upper bound
    nonpositive, and at an interior coordinate zero. Equal bounds fix a variable
    and impose no gradient-sign condition. Tiny roundoff snapping is reported.
    """
    H, g = np.asarray(H, dtype=float), np.asarray(g, dtype=float)
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if g.ndim != 1 or not 1 <= g.size <= 7:
        raise ValueError('Expected one to seven QP variables')
    n = g.size
    if H.shape != (n, n) or lower.shape != (n,) or upper.shape != (n,):
        raise ValueError('QP dimensions do not agree')
    if not all(np.isfinite(v).all() for v in (H, g, lower, upper)):
        raise FloatingPointError('Non-finite QP input')
    if np.any(lower > upper):
        raise ValueError('Empty box: lower bound exceeds upper bound')
    if not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1:
        raise ValueError('max_iterations must be a positive integer')
    if not np.isfinite([kkt_atol, kkt_rtol]).all() or kkt_atol < 0 or kkt_rtol < 0:
        raise ValueError('KKT tolerances must be finite and nonnegative')
    eps = np.finfo(float).eps
    hscale = max(1., float(np.linalg.norm(H, ord=np.inf)))
    symmetry_error = float(np.max(np.abs(H - H.T)))
    if symmetry_error > 64 * eps * hscale:
        raise ValueError('Hessian is not symmetric')
    H = (H + H.T) * .5
    try:
        np.linalg.cholesky(H)
    except np.linalg.LinAlgError as exc:
        raise ValueError('Hessian must be positive definite') from exc
    x = np.clip(np.zeros(n), lower, upper)
    fixed = lower == upper
    # -1: active lower, +1: active upper, 2: equality/fixed, 0: free.
    active = np.zeros(n, dtype=np.int8)
    active[x == lower] = -1
    active[x == upper] = 1
    active[fixed] = 2
    feas_tol = 128 * eps * max(1., float(np.max(np.abs(lower))), float(np.max(np.abs(upper))))
    history = []
    roundoff_snap_count = 0

    def diagnostics(status, iterations, reason=None):
        gradient = H @ x + g
        low, high = x == lower, x == upper
        interior = ~(low | high)
        violation = np.zeros(n)
        violation[interior] = np.abs(gradient[interior])
        violation[low & ~fixed] = np.maximum(-gradient[low & ~fixed], 0.)
        violation[high & ~fixed] = np.maximum(gradient[high & ~fixed], 0.)
        primal = float(max(0., np.max(lower - x), np.max(x - upper)))
        scale = max(1., float(np.linalg.norm(g, ord=np.inf)),
                    float(np.linalg.norm(H, ord=np.inf)) * max(1., float(np.linalg.norm(x, ord=np.inf))))
        tolerance = float(kkt_atol + kkt_rtol * scale)
        kkt = float(np.max(violation))
        passed = bool(np.isfinite(gradient).all() and primal <= feas_tol and kkt <= tolerance)
        return {
            'status': status, 'iterations': iterations, 'success': status == 'optimal' and passed,
            'kkt_pass': passed, 'kkt_residual_inf': kkt, 'kkt_tolerance': tolerance,
            'primal_violation_inf': primal, 'feasibility_tolerance': feas_tol,
            'gradient': gradient.tolist(), 'coordinate_kkt_violations': violation.tolist(),
            'objective': float(.5 * x @ H @ x + g @ x), 'x': x.tolist(),
            'lower': lower.tolist(), 'upper': upper.tolist(),
            'active_lower_indices': np.flatnonzero(low & ~fixed).tolist(),
            'active_upper_indices': np.flatnonzero(high & ~fixed).tolist(),
            'fixed_indices': np.flatnonzero(fixed).tolist(),
            'roundoff_snap_count': roundoff_snap_count, 'symmetry_error_inf': symmetry_error,
            'reason': reason, 'history': history,
        }

    for iteration in range(1, max_iterations + 1):
        gradient = H @ x + g
        if not np.isfinite(gradient).all():
            return x, diagnostics('numerical_failure', iteration, 'Non-finite gradient')
        event = {'iteration': iteration, 'x_before': x.tolist(), 'active_before': active.tolist()}
        history.append(event)
        free = np.flatnonzero(active == 0)
        direction = np.zeros(n)
        if free.size:
            try:
                direction[free] = np.linalg.solve(H[np.ix_(free, free)], -gradient[free])
            except np.linalg.LinAlgError:
                return x, diagnostics('numerical_failure', iteration, 'Free-face linear solve failed')
            if not np.isfinite(direction).all():
                return x, diagnostics('numerical_failure', iteration, 'Non-finite free-face direction')
        # Move towards the free-face minimizer only until the first box face.
        ratios = np.full(n, np.inf)
        positive, negative = direction > 0., direction < 0.
        ratios[positive] = (upper[positive] - x[positive]) / direction[positive]
        ratios[negative] = (lower[negative] - x[negative]) / direction[negative]
        first_ratio = float(np.min(ratios))
        if first_ratio < -64 * eps:
            return x, diagnostics('numerical_failure', iteration, 'Feasible iterate lost before step')
        alpha = max(0., min(1., first_ratio))
        trial = x + alpha * direction
        hits = np.flatnonzero(ratios <= alpha + 64 * eps) if first_ratio <= 1. else np.array([], dtype=int)
        for i in hits:
            trial[i] = upper[i] if direction[i] > 0 else lower[i]
        if (np.max(lower - trial) > feas_tol or np.max(trial - upper) > feas_tol
                or not np.isfinite(trial).all()):
            return x, diagnostics('numerical_failure', iteration, 'Step violated box beyond roundoff')
        snapped = np.clip(trial, lower, upper)
        roundoff_snap_count += int(np.any(snapped != trial))
        x = snapped
        event.update(alpha=alpha, direction=direction.tolist(), hit_indices=hits.tolist(), x_after=x.tolist())
        for i in hits:
            active[i] = 1 if direction[i] > 0 else -1
        active[fixed] = 2
        if hits.size:
            event['action'] = 'activate_first_hit'
            continue
        # At the face optimum, all free gradients should vanish. Bound gradient
        # signs decide whether an active constraint should be released.
        diag = diagnostics('checking_kkt', iteration)
        if diag['kkt_pass']:
            event['action'] = 'optimal'
            return x, diagnostics('optimal', iteration)
        gradient = H @ x + g
        violation = np.zeros(n)
        violation[active == -1] = np.maximum(-gradient[active == -1], 0.)
        violation[active == 1] = np.maximum(gradient[active == 1], 0.)
        release = int(np.argmax(violation))
        if violation[release] <= diag['kkt_tolerance']:
            event['action'] = 'free_face_kkt_failure'
            return x, diagnostics('numerical_failure', iteration, 'Free-face stationarity not resolved')
        active[release] = 0
        event.update(action='release_constraint', release_index=release,
                     release_kkt_violation=float(violation[release]))
    return x, diagnostics('iteration_limit', max_iterations, 'Active-set budget exhausted')


def solve_trial_bounded(np, evaluate, rot_error, limits, joint_names, target, seed,
                        target_id, seed_id, max_attempts=180):
    """Match solve_trial's return schema while replacing only its step rule."""
    trial_id = target_id + '_from_' + seed_id
    q, target = np.asarray(seed, dtype=float).copy(), np.asarray(target, dtype=float)
    limits = np.asarray(limits, dtype=float)
    n = len(joint_names)
    if q.shape != (n,) or limits.shape != (n, 2) or target.shape != (3,):
        raise ValueError('Malformed bounded IK input')
    if not all(np.isfinite(v).all() for v in (q, target, limits)):
        raise FloatingPointError('Non-finite bounded IK input')
    lo, hi = limits[:, 0] + .03, limits[:, 1] - .03
    if np.any(lo > hi) or np.any(q < lo) or np.any(q > hi):
        raise ValueError('Seed must be feasible inside the unchanged solver inset')
    if not isinstance(max_attempts, (int, np.integer)) or max_attempts < 1:
        raise ValueError('max_attempts must be a positive integer')
    rows, states, attempts = [], [], []
    accepted_updates = qp_solve_count = qp_failure_count = 0
    stop_reason, termination_checked_strict = 'iteration_budget', False
    roundoff_snap_count = 0

    def measure(q, result):
        p, rotation, jp, jr = result
        ep, er = target - p, rot_error(rotation)
        error = np.r_[ep, .15 * er]
        if not all(np.isfinite(v).all() for v in (q, p, rotation, error)):
            raise FloatingPointError('Non-finite bounded IK state')
        state = {
            'iteration': accepted_updates, 'q_arm_rad': q.tolist(),
            'fk_position_world_m': p.tolist(), 'fk_rotation_world': rotation.tolist(),
            'position_error_world_m': ep.tolist(), 'orientation_error_vector_rad': er.tolist(),
            'position_error_mm': float(np.linalg.norm(ep) * 1000),
            'orientation_error_rad': float(np.linalg.norm(er)),
            'weighted_cost': float(np.linalg.norm(error)), **boundary_report(np, q, limits, joint_names),
        }
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
            stop_reason, termination_checked_strict = 'strict_converged', True
            attempt['strict_termination_check_passed'] = True
            break
        jacobian = np.vstack([jp, .15 * jr])
        if jacobian.shape != (6, n) or not np.isfinite(jacobian).all():
            raise FloatingPointError('Invalid bounded IK Jacobian')
        H, g = jacobian.T @ jacobian + 1e-6 * np.eye(n), -jacobian.T @ error
        lower, upper = np.maximum(lo - q, -.12), np.minimum(hi - q, .12)
        qp_solve_count += 1
        attempt['step_bounds_rad'] = {'lower': lower.tolist(), 'upper': upper.tolist()}
        step, qp = solve_box_qp(H, g, lower, upper)
        attempt['qp_diagnostics'] = qp
        attempt['bounded_step_rad'] = step.tolist()
        if qp['status'] != 'optimal' or not qp['success']:
            qp_failure_count += 1
            stop_reason = 'qp_failure'
            break
        accepted = False
        for scale in (1., .5, .25, .125, .0625):
            proposed = q + scale * step
            # Convex combinations of feasible q and q+step are feasible. Only
            # floating-point roundoff may be snapped; no substantive post-clip.
            candidate_violation = float(max(0., np.max(lo - proposed), np.max(proposed - hi)))
            snap_tolerance = 128 * np.finfo(float).eps * max(1., float(np.max(np.abs(limits))))
            if candidate_violation > snap_tolerance:
                raise FloatingPointError('QP line-search candidate escaped joint bounds')
            trial = np.clip(proposed, lo, hi)
            snapped = trial != proposed
            roundoff_snap_count += int(bool(np.any(snapped)))
            tp, tR, _, _ = evaluate(trial)
            trial_error = np.r_[target - tp, .15 * rot_error(tR)]
            cost = np.linalg.norm(trial_error)
            if not np.isfinite(trial_error).all() or not np.isfinite(cost):
                raise FloatingPointError('Non-finite bounded IK candidate')
            candidate = {
                'scale': scale, 'proposed_q_arm_rad': proposed.tolist(),
                'candidate_q_arm_rad': trial.tolist(), 'clipped': False, 'clipped_joints': [],
                'clip_delta_rad': [0.] * n, 'roundoff_snapped': bool(np.any(snapped)),
                'roundoff_snap_delta_rad': (trial - proposed).tolist(),
                'pre_snap_violation_rad': candidate_violation,
                'fk_position_world_m': tp.tolist(), 'position_error_mm': float(np.linalg.norm(target - tp) * 1000),
                'orientation_error_rad': float(np.linalg.norm(rot_error(tR))),
                'weighted_cost': float(cost), 'cost_decrease': float(np.linalg.norm(error) - cost),
                'accepted': bool(cost < np.linalg.norm(error)), **boundary_report(np, trial, limits, joint_names),
            }
            attempt['candidates'].append(candidate)
            if cost < np.linalg.norm(error):
                q = trial
                accepted = True
                accepted_updates += 1
                attempt.update(accepted=True, accepted_scale=scale)
                break
        if not accepted:
            stop_reason = 'no_acceptable_step'
            break
    final, ep, er, _, _, _ = measure(q, evaluate(q))
    if not rows or rows[-1]['iteration'] != accepted_updates:
        record_state(final)
    final_kind = 'initial_final' if accepted_updates == 0 else 'final'
    rows[-1]['row_kind'] = final_kind
    states[-1] = dict(final, row_kind=final_kind)
    summary = {
        'trial_id': trial_id, 'target_id': target_id, 'seed_id': seed_id, 'method': 'box_qp_active_set',
        'target_config_id': POINT_IDS.get(target_id, target_id), 'seed_config_id': POINT_IDS.get(seed_id, seed_id),
        'target_position_world_m': target.tolist(), 'initial_q_arm_rad': np.asarray(seed, dtype=float).tolist(),
        'stop_reason': stop_reason, 'accepted_updates': accepted_updates, 'attempts': len(attempts),
        'strict_converged': termination_checked_strict, 'termination_checked_strict': termination_checked_strict,
        'strict_threshold_satisfied': bool(np.linalg.norm(ep) < 2e-5 and np.linalg.norm(er) < 2e-4),
        'task_tolerance_pass': bool(np.linalg.norm(ep) <= .003 and np.linalg.norm(er) <= .05),
        'task_tolerance_scope': 'Final position/orientation only; excludes collision/path/dynamics/grasp.',
        'clip_candidate_count': 0, 'roundoff_snap_candidate_count': roundoff_snap_count,
        'candidate_count': sum(len(a['candidates']) for a in attempts),
        'qp_solve_count': qp_solve_count, 'qp_failure_count': qp_failure_count,
        'all_qp_solves_succeeded': qp_failure_count == 0,
        'method_settings': BOX_SETTINGS, **final,
    }
    detail = {'trial_id': trial_id, 'target_id': target_id, 'seed_id': seed_id,
              'method': 'box_qp_active_set', 'states': states, 'attempts': attempts, 'final': summary}
    return q, summary, rows, detail
