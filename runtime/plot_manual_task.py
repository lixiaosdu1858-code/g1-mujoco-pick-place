"""Plot saved manual pick-place records; never advances simulation or changes input."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone, timedelta
import uuid

from lesson_common import ROOT, guard

REQUIRED = ('sim_s', 'position_error_m', 'orientation_error_rad',
            'cube_bottom_clearance_m', 'finger_ref_m', 'finger1_q_m', 'finger2_q_m',
            'finger1_normal_n', 'finger2_normal_n', 'placement_continuous_s',
            'cube_tray_xy_margin_m', 'cube_speed_m_s') + tuple(
                f'{prefix}_{axis}_m' for prefix in ('target', 'ref', 'actual') for axis in 'xyz')
FLAGS = ('terminal_state', 'grip_reference_valid', 'release_active', 'placement_valid')
RELATIVE = ('relative_slip_m', 'relative_rotation_rad')
MAX_BYTES = 256 * 1024 * 1024
MAX_ROWS = 250001
MAX_POINTS = 5000
JOG_MAX_POINTS = 20000


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def directory(path):
    path = Path(path).absolute()
    if not path.is_dir() or not path.resolve().is_relative_to((ROOT / 'outputs').resolve()):
        raise ValueError('Directory must exist under project outputs.')
    for item in (path, *path.parents):
        if item.is_symlink() or item.is_junction():
            raise ValueError('Reparse directories are not allowed.')
    return path.resolve()


def regular(path):
    if not path.is_file() or path.is_symlink() or path.is_junction():
        raise ValueError(f'Expected a regular input file: {path}')
    if path.stat().st_size > MAX_BYTES:
        raise ValueError('Input exceeds the 256 MiB plotting bound.')


def number(value, name):
    if isinstance(value, bool) or value is None or value == '':
        raise ValueError(f'Missing/Boolean numeric measurement: {name}')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'Nonfinite measurement: {name}')
    return value


def flag(value, name):
    if value in ('1', 'True', 'true', True):
        return True
    if value in ('0', 'False', 'false', False):
        return False
    raise ValueError(f'Invalid Boolean: {name}')


def threshold(summary, group, name):
    value = number(summary[group][name], f'{group}.{name}')
    if value <= 0:
        raise ValueError(f'Nonpositive threshold: {group}.{name}')
    return value


def prepare(run, destination):
    run, destination = directory(run), directory(destination)
    summary_path, csv_path = run / 'summary.json', run / 'manual.csv'
    regular(summary_path)
    summary_hash = sha(summary_path)
    summary = json.loads(summary_path.read_text(encoding='utf-8-sig'))
    if summary.get('kind') not in ('manual_pickplace_dynamics', 'synthetic_manual_plot_toolcheck'):
        raise ValueError('Expected a manual pick-place record or explicitly synthetic plot fixture.')
    for key in ('status', 'end_reason', 'steps', 'simulation_s'):
        if key not in summary:
            raise ValueError(f'Missing summary field: {key}')
    synthetic = summary['kind'] == 'synthetic_manual_plot_toolcheck'
    held_mode = summary.get('input_mode') == 'foreground_hold_to_jog'
    max_points = JOG_MAX_POINTS if held_mode else MAX_POINTS
    incomplete = summary['status'] in ('error', 'stopped', 'interrupted', 'partial')
    recording_issues = []
    gates = {key: summary.get(key) for key in ('manual_task_completed', 'grasp_hold_passed',
                                               'placement_passed', 'tracking_thresholds_passed')}
    if any(v is not None and not isinstance(v, bool) for v in gates.values()):
        raise ValueError('Outcome fields must be Boolean or null.')
    thresholds = dict(
        position_mm=1000 * threshold(summary, 'thresholds', 'position_warning_m'),
        orientation_rad=threshold(summary, 'thresholds', 'orientation_warning_rad'),
        slip_mm=1000 * threshold(summary, 'thresholds', 'maximum_relative_slip_m'),
        rotation_rad=threshold(summary, 'thresholds', 'maximum_relative_rotation_rad'),
        lift_mm=1000 * threshold(summary, 'thresholds', 'minimum_lift_m'),
        placement_s=threshold(summary, 'placement_thresholds', 'hold_s'))
    heights = {}
    for key in ('table_top_z_m', 'tray_floor_z_m', 'tray_rim_z_m'):
        if key in summary:
            heights[key] = number(summary[key], key)
    rows, csv_hash = [], None
    if csv_path.exists():
        regular(csv_path)
        csv_hash = sha(csv_path)
        if summary.get('manual_csv_sha256') and summary['manual_csv_sha256'].lower() != csv_hash:
            raise ValueError('CSV SHA256 differs from summary.')
        with csv_path.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames:
                for field in (*REQUIRED, *FLAGS, *RELATIVE, 'phase'):
                    if field not in reader.fieldnames:
                        raise ValueError(f'Missing CSV column: {field}')
            for raw in reader:
                if len(rows) >= MAX_ROWS:
                    raise ValueError('Record exceeds the 250001-row plotting bound.')
                row = {name: number(raw[name], name) for name in REQUIRED}
                row.update({name: flag(raw[name], name) for name in FLAGS})
                row['phase'] = raw['phase']
                if not row['phase']:
                    raise ValueError('Missing phase.')
                if row['sim_s'] < 0 or (rows and row['sim_s'] <= rows[-1]['sim_s']):
                    raise ValueError('Sample times must be nonnegative and strictly increasing.')
                if rows and rows[-1]['terminal_state']:
                    raise ValueError('Only the last row can be terminal.')
                for name in RELATIVE:
                    if row['release_active']:
                        # Runtime may omit released separation entirely; it is not held slip.
                        if raw[name].strip():
                            number(raw[name], name)
                        row[name] = None
                    elif row['grip_reference_valid']:
                        row[name] = number(raw[name], name)
                    else:
                        if raw[name].strip():
                            raise ValueError('Relative measurements must be blank before grip reference.')
                        row[name] = None
                for name in ('position_error_m', 'orientation_error_rad', 'finger1_normal_n',
                             'finger2_normal_n', 'placement_continuous_s', 'cube_speed_m_s', *RELATIVE):
                    if row[name] is not None and row[name] < 0:
                        raise ValueError(f'Negative nonnegative measurement: {name}')
                if (row['placement_valid'] or row['placement_continuous_s'] > 0) and not row['release_active']:
                    raise ValueError('Placement validity/time before release is inconsistent.')
                # Held-object slip is unavailable after opening; do not draw separation as slip.
                for name in RELATIVE:
                    if row['release_active']:
                        row[name] = None
                rows.append(row)
    if not rows and int(summary['steps']) > 0:
        if not incomplete:
            raise ValueError('Nonzero step count has no saved samples.')
        recording_issues.append('Summary reports physics steps but no state samples survived; no curves inferred.')
    if rows and abs(rows[-1]['sim_s'] - float(summary['simulation_s'])) > 1e-7:
        if not incomplete or rows[-1]['sim_s'] > float(summary['simulation_s']) + 1e-7:
            raise ValueError('Summary duration differs from the final recorded sample.')
        recording_issues.append('Incomplete tail: curves end at last saved sample, earlier than summary simulation time.')
    if rows and len(rows) != int(summary['steps']) + 1:
        if not incomplete:
            raise ValueError('Completed record row count differs from steps plus terminal state.')
        recording_issues.append('Saved row count differs from steps plus terminal state; no missing rows synthesized.')
    if rows and not rows[-1]['terminal_state']:
        if not incomplete:
            raise ValueError('Completed record lacks its terminal sample.')
        recording_issues.append('No saved terminal-state row; latest point is the last observed pre-step state.')
    if gates['placement_passed']:
        historical = any(r['placement_valid'] and r['placement_continuous_s'] + 1e-8 >= thresholds['placement_s'] for r in rows)
        final_valid = bool(rows and rows[-1]['placement_valid'] and rows[-1]['placement_continuous_s'] + 1e-8 >= thresholds['placement_s'])
        if not historical:
            raise ValueError('Placement PASS has no recorded continuous-hold evidence.')
        if not final_valid:
            if not incomplete or gates['manual_task_completed']:
                raise ValueError('Placement PASS conflicts with the final recorded stability state.')
            recording_issues.append('Placement milestone was reached earlier, but the final continuous hold is not valid; manual completion is not accepted.')
    if gates['manual_task_completed'] and not all(gates[k] for k in ('grasp_hold_passed', 'placement_passed', 'tracking_thresholds_passed')):
        raise ValueError('Manual completion conflicts with summary component gates.')
    if recording_issues and gates['manual_task_completed']:
        raise ValueError('Manual completion cannot be shown for an incomplete recording.')

    origin = [rows[0][f'ref_{a}_m'] for a in 'xyz'] if rows else None
    for row in rows:
        for axis, offset in zip('xyz', origin):
            for prefix in ('target', 'ref', 'actual'):
                row[f'{prefix}_{axis}_mm'] = 1000 * (row[f'{prefix}_{axis}_m'] - offset)
        for name in ('position_error_m', 'cube_bottom_clearance_m', 'finger_ref_m', 'finger1_q_m', 'finger2_q_m'):
            row[name.removesuffix('_m') + '_mm'] = 1000 * row[name]
        row['held_slip_mm'] = 1000 * row['relative_slip_m'] if row['relative_slip_m'] is not None else None
        row['held_rotation_rad'] = row['relative_rotation_rad']
        row['postrelease_hold_s'] = row['placement_continuous_s'] if row['release_active'] else None
        row['postrelease_valid'] = int(row['placement_valid']) if row['release_active'] else None
    display = ['position_error_mm', 'orientation_error_rad', 'cube_bottom_clearance_mm',
               'finger_ref_mm', 'finger1_q_mm', 'finger2_q_mm', 'finger1_normal_n', 'finger2_normal_n',
               'held_slip_mm', 'held_rotation_rad', 'postrelease_hold_s', 'postrelease_valid'] + [
                   f'{p}_{a}_mm' for p in ('target', 'ref', 'actual') for a in 'xyz']
    mandatory = {0, len(rows)-1} if rows else set()
    # Keep both sides of every transition and every displayed global extremum.
    transitions = ['phase', *FLAGS, 'target_x_m', 'target_y_m', 'target_z_m', 'finger_ref_m']
    for i in range(1, len(rows)):
        if any(rows[i][k] != rows[i-1][k] for k in transitions):
            mandatory.update((i-1, i))
    for name in display:
        indices = [i for i, row in enumerate(rows) if row[name] is not None]
        if indices:
            mandatory.update((min(indices, key=lambda i: rows[i][name]), max(indices, key=lambda i: rows[i][name])))
    if len(mandatory) >= max_points:
        raise ValueError(f'Too many transitions to draw faithfully within the {max_points}-point bound.')
    stride = max(1, math.ceil(len(rows) / (max_points-len(mandatory))))
    kept = sorted(mandatory | set(range(0, len(rows), stride)))
    if len(kept) > max_points:
        raise ValueError('Plot point bound exceeded.')
    processed = [e for e in summary.get('events', []) if e.get('stage', 'processed') == 'processed']
    accepted = sum(e.get('accepted') is True for e in processed)
    refused = sum(e.get('accepted') is False for e in processed)
    # A held direction emits several accepted checked references. Their count
    # is not the number of physical key presses by the user.
    jog_events = [e for e in processed if e.get('input_mode') == 'held_direction_segment']
    jog_accepted = sum(e.get('accepted') is True for e in jog_events)
    jog_refused = sum(e.get('accepted') is False for e in jog_events)
    if sha(summary_path) != summary_hash or (csv_hash is not None and sha(csv_path) != csv_hash):
        raise ValueError('Inputs changed during parsing.')
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
    stem = 'manual_task_' + stamp + '_' + uuid.uuid4().hex[:8]
    payload = dict(kind='manual_pickplace_plot_payload', synthetic_tool_check=synthetic,
        run_status=summary['status'], end_reason=summary['end_reason'], steps=summary['steps'],
        simulation_s=summary['simulation_s'], gates=gates, thresholds=thresholds, geometry=heights,
        diagnostic_only=bool(recording_issues), recording_issues=recording_issues,
        recorded_time_window_s=[rows[0]['sim_s'], rows[-1]['sim_s']] if rows else None,
        source_csv=str(csv_path), source_csv_sha256=csv_hash, summary_json=str(summary_path),
        summary_sha256=summary_hash, source_rows=len(rows), plotted_rows=len(kept), row_stride=stride,
        effective_point_budget=max_points, mandatory_transition_extrema_rows=len(mandatory),
        retained_zero_based_indices=kept, rows=[rows[i] for i in kept], origin_world_m=origin,
        accepted_events=accepted, refused_events=refused, output_png=str(destination/(stem+'.png')),
        input_mode=summary.get('input_mode', 'discrete_key_commands'),
        held_direction_mode=held_mode, accepted_jog_reference_segments=jog_accepted,
        refused_jog_reference_proposals=jog_refused,
        accepted_discrete_commands=accepted-jog_accepted,
        refused_discrete_commands=refused-jog_refused,
        event_count_semantics=('Held-direction reference segments are not key-press counts; discrete commands are counted separately.'
                               if held_mode else 'Saved processed command events, including start/end/camera and refused keys.'),
        output_source_json=str(destination/(stem+'.source.json')), script=str(Path(__file__).resolve()),
        script_sha256=sha(Path(__file__)), sample_semantics=summary.get('sample_semantics'),
        generated_beijing=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        source_files_modified=False, no_new_physics=True,
        sampling='Uniform stride plus endpoints, phase/flag/target/finger-reference transitions and global displayed extrema; no smoothing.',
        relative_measurement_semantics='Blank before saved grip baseline and after release starts; never substitute zero.',
        success_semantics='Outcome labels quote saved summary gates; no success is inferred by plotting.')
    output = destination / (stem+'.plotdata.json')
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(payload, stream, ensure_ascii=False, allow_nan=False, indent=2)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output-directory', type=Path)
    parser.add_argument('--no-open', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--receipt-json', action='store_true',
                        help='With --prepare-only, emit an ASCII JSON path/hash receipt for native-shell capture.')
    args = parser.parse_args()
    if args.receipt_json and not args.prepare_only:
        parser.error('--receipt-json requires --prepare-only')
    guard()
    run = directory(args.run)
    destination = directory(args.output_directory or run)
    if args.prepare_only:
        output = prepare(run, destination)
        if args.receipt_json:
            # Windows PowerShell 5.1 may decode redirected native stdout using a
            # legacy console code page. ASCII JSON survives that boundary; the
            # actual payload is still read from an explicitly UTF-8 JSON file.
            print(json.dumps(dict(kind='manual_plot_prepare_receipt', protocol=1,
                                  plotdata_json=str(output), plotdata_sha256=sha(output)),
                             ensure_ascii=True))
        else:
            print(output)
        return 0
    powershell = Path(os.environ['SystemRoot']) / 'System32/WindowsPowerShell/v1.0/powershell.exe'
    command = [str(powershell), '-NoProfile', '-File', str(ROOT/'tools/Show-Result.ps1'),
               '-RunPath', str(run), '-OutputDirectory', str(destination),
               '-Python', os.environ['G1_PACKAGE_PYTHON'], '-CacheDirectory', os.environ['G1_PACKAGE_CACHE']]
    if args.no_open:
        command.append('-NoShow')
    child_env = {key: value for key, value in os.environ.items() if key.upper() != 'PSMODULEPATH'}
    return subprocess.run(command, cwd=ROOT, env=child_env, check=False).returncode


if __name__ == '__main__':
    raise SystemExit(main())

