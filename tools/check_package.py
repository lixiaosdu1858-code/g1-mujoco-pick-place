"""Verify the v0.2 source distribution; no simulation or installation."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile

SOURCE_ROOT = Path(__file__).absolute().parents[1]
ROOT = SOURCE_ROOT.resolve()
MANIFEST = ROOT / 'PACKAGE-MANIFEST.json'
PACKAGE = 'g1-mujoco-pick-place-v0.2.0'
STATUS = 'release_prepared_pending_publication'
LOCAL_ROOTS = {'.venv', '.cache', 'outputs', '.git', 'upstream'}
SCENE_SHA = '9d71754df03ab06c4fbed3010ef9a883ce71a3e6ca04de01815d7d1c7ca2630c'
GENERATED = {name: SCENE_SHA for name in ('assets/scene.xml', 'configs/scene.xml')}
MAX_FILE_BYTES = 256 * 1024 * 1024
MAX_TOTAL_BYTES = 768 * 1024 * 1024
ROOT_FILES = {
    '.gitignore', 'LICENSE', 'LICENSE-STATUS.md', 'README.md', 'RELEASE-NOTES.md',
    'RESULTS.md', 'THIRD_PARTY_NOTICES.md', 'requirements.txt', 'Run.ps1',
    'Collect-Demonstration.ps1', 'Inspect-Demonstration.ps1', 'Run-Replay.ps1',
    'Show-ReplayComparison.ps1', 'Show-OrientationComparison.ps1', 'Prepare-Example.ps1',
}
DOCS = {
    'assets.md', 'candidate-user-verification.md', 'control.md', 'demo.md',
    'demonstration-format.md', 'package-verification.md', 'plot-verification.md',
    'results.md', 'scene-preparation.md', 'troubleshooting.md', 'v01-results.md',
}
RUNTIME = {
    'build_elbow_scene.py', 'build_shoulder_elbow_scene.py', 'carry_route.py',
    'carry_runtime.py', 'carry_to_tray.py', 'cartesian_keyboard.py',
    'cartesian_keyboard_target.py', 'centered_grasp_lift.py', 'grasp_contact_geometry.py',
    'grasp_lift.py', 'grasp_solver_compare.py', 'grasp_solver_runtime.py',
    'ik_bounded_step.py', 'ik_seed_compare.py', 'lesson_common.py', 'manual_jog_input.py',
    'manual_pickplace.py', 'manual_task_checks.py', 'manual_task_observer.py',
    'manual_task_target.py', 'package_runtime.py', 'pickplace_layout_preview.py',
    'plot_manual_task.py', 'reachable_motion.py', 'reachable_pose_preview.py',
}
CONFIGS = {
    'accepted_baseline.json', 'config.json', 'dependency_versions.json',
    'frozen_context.json', 'keyboard_anchor.json', 'plot_source_equivalence.json',
    'reference.json', 'runtime_integrity.json', 'source_equivalence.json',
    'static_baseline.json',
}
TOOLS = {'check_package.py', 'prepare_scene.py', 'prepare_example.py',
         'verify_runtime.py', 'task_parameters.json', 'Show-Result.ps1', 'test_portable.py',
         'Enable-PackageEnvironment.ps1'}
DEMONSTRATION = {
    'Collect-Demonstration.ps1', 'Inspect-Demonstration.ps1', 'Run-Replay.ps1',
    'Show-ReplayComparison.ps1', 'Show-OrientationComparison.ps1',
    'collect_episode.py', 'compare_orientation.py', 'compare_replay.py',
    'episode_schema.py', 'episode_state.py', 'inspect_episode.py', 'replay_episode.py',
    'portable_paths.py', 'check_episode.py',
}
ASSET_JSON = {'manifest.json', 'upstream-license-map.json', 'vfs_map.json'}
LICENSES = {'HIW-500-Apache-2.0.txt', 'HIW-500-NOTICE.txt',
            'unitree_rl_gym-BSD-3-Clause.txt', 'unitree_ros-BSD-3-Clause.txt'}
EVIDENCE = {'README.md', 'candidate-user-run.json', 'common_configuration.json',
            'comparison.json', 'package-static-verification.json', 'plot-verification.json',
            'scene-generation.json', 'source_snapshots.json', 'runtime-compatibility.json',
            'v02-static-verification.json', 'v02-user-replay-verification.json',
            'v02-release-verification.json'}
RUN_IDS = {'175535_099', '212238_372', '212557_670'}
RUN_EVIDENCE = {'audit.json', 'curve_provenance.json', 'summary.json'}
EXAMPLE = 'examples/recording-001'
EXAMPLE_MANIFEST = EXAMPLE + '/example-manifest.json'
REQUIRED = ROOT_FILES | {'docs/' + name for name in DOCS} | {
    'tools/check_package.py', 'tools/prepare_example.py', 'tools/prepare_scene.py',
    'tools/verify_runtime.py', 'tools/Enable-PackageEnvironment.ps1',
    EXAMPLE_MANIFEST, 'configs/runtime_integrity.json',
    'evidence/runtime-compatibility.json',
    'evidence/v02-user-replay-verification.json',
    'evidence/v02-release-verification.json',
} | {'tools/demonstration/' + name for name in DEMONSTRATION}
TEXT_SUFFIXES = {'.md', '.py', '.ps1', '.json', '.jsonl', '.csv', '.txt'}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def linked(path):
    info = path.lstat()
    return (path.is_symlink() or getattr(path, 'is_junction', lambda: False)()
            or bool(getattr(info, 'st_file_attributes', 0)
                    & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)))


def relative_name(name):
    if not isinstance(name, str) or not name or '\\' in name or ':' in name:
        raise ValueError(f'Invalid relative name: {name!r}')
    p = PurePosixPath(name)
    if p.is_absolute() or any(part in ('', '.', '..') for part in name.split('/')):
        raise ValueError(f'Invalid relative name: {name!r}')
    return p


def example_inventory():
    path = ROOT / EXAMPLE_MANIFEST
    for parent in (path.parent, path.parent.parent):
        if not parent.is_dir() or linked(parent):
            raise ValueError('Example directories must be regular local directories.')
    if not path.is_file() or linked(path):
        raise ValueError('Regular example-manifest.json is required.')
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    if data.get('schema') != 'g1.portable_example.v0.2.0':
        raise ValueError('Unexpected example manifest schema.')
    items = data.get('files')
    if not isinstance(items, dict) or not items:
        raise ValueError('Example files must be a non-empty exact mapping.')
    external = data.get('external_files')
    if not isinstance(external, dict) or set(external) != {'demonstration/configs/scene.xml'}:
        raise ValueError('Example must name only the expected external scene.')
    scene = external['demonstration/configs/scene.xml']
    if not isinstance(scene, dict) or scene.get('sha256') != SCENE_SHA:
        raise ValueError('External example scene hash differs.')
    allowed = {}
    for name, record in items.items():
        p = relative_name(name)
        if p.parts[0] not in {'demonstration', 'saved-replay'} and name not in {
            'README.md', 'provenance-map.json', 'source-map.json', 'export-provenance.json'}:
            raise ValueError(f'Unexpected example entry: {name}')
        if p.suffix.lower() not in {'.py', '.json', '.jsonl', '.csv', '.md'}:
            raise ValueError(f'Unexpected example file type: {name}')
        if any(part.startswith('.') or part in {'node_modules', '__pycache__', 'venv',
                'cache', 'outputs', 'upstream', 'downloads'} for part in p.parts):
            raise ValueError(f'Unexpected generated example path: {name}')
        if p.parts[0] == 'saved-replay' and name not in {
                'saved-replay/manual.csv', 'saved-replay/summary.json',
                'saved-replay/replay_episode.py', 'saved-replay/replay_code/episode_state.py',
                'saved-replay/replay_code/inspect_episode.py'}:
            raise ValueError(f'Unexpected saved replay entry: {name}')
        if (not isinstance(record, dict) or type(record.get('bytes')) is not int
                or not 0 <= record['bytes'] <= MAX_FILE_BYTES
                or not re.fullmatch(r'[0-9a-f]{64}', record.get('sha256', ''))):
            raise ValueError(f'Invalid example integrity record: {name}')
        allowed[EXAMPLE + '/' + name] = record
    for required in ('demonstration/manual.csv', 'demonstration/summary.json',
                     'saved-replay/manual.csv', 'saved-replay/summary.json'):
        if EXAMPLE + '/' + required not in allowed:
            raise ValueError(f'Missing required example file: {required}')
    return allowed


def allowed_file(name, examples):
    p = PurePosixPath(name)
    if name in ROOT_FILES or name == EXAMPLE_MANIFEST or name in examples:
        return True
    parts = p.parts
    if len(parts) == 2:
        return (parts[0] == 'docs' and parts[1] in DOCS
                or parts[0] == 'runtime' and parts[1] in RUNTIME
                or parts[0] == 'configs' and parts[1] in CONFIGS
                or parts[0] == 'tools' and parts[1] in TOOLS
                or parts[0] == 'licenses' and parts[1] in LICENSES
                or parts[0] == 'evidence' and parts[1] in EVIDENCE
                or parts[0] == 'assets' and parts[1] in ASSET_JSON
                or parts[0] == 'media' and parts[1] in {
                    f'run-{run}-curves.png' for run in RUN_IDS})
    if len(parts) == 3:
        return (parts[:2] == ('tools', 'demonstration') and parts[2] in DEMONSTRATION
                or parts[:2] == ('assets', 'meshes') and p.suffix.lower() == '.stl')
    if len(parts) == 4:
        return (parts[:3] == ('assets', 'dex1_1', 'meshes') and p.suffix.lower() == '.stl'
                or parts[:2] == ('evidence', 'runs') and parts[2] in RUN_IDS
                and parts[3] in RUN_EVIDENCE)
    return False


def allowed_dir(name, examples):
    fixed = {'docs', 'runtime', 'configs', 'tools', 'tools/demonstration', 'licenses',
             'evidence', 'evidence/runs', 'assets', 'assets/meshes', 'assets/dex1_1',
             'assets/dex1_1/meshes', 'media', 'examples', EXAMPLE}
    fixed.update('evidence/runs/' + run for run in RUN_IDS)
    return name in fixed or any(file.startswith(name + '/') for file in examples)


def inventory():
    for parent in (SOURCE_ROOT, *SOURCE_ROOT.parents):
        if linked(parent):
            raise ValueError('Package ancestry must not contain links or junctions.')
    examples = example_inventory()
    files, case_names = {}, set()
    total = 0
    for base, dirs, names in os.walk(ROOT, topdown=True, followlinks=False):
        base = Path(base)
        for name in list(dirs):
            path = base / name
            rel = path.relative_to(ROOT).as_posix()
            if linked(path):
                raise ValueError(f'Linked directory: {rel}')
            if base == ROOT and name in LOCAL_ROOTS:
                dirs.remove(name)
                continue
            if not allowed_dir(rel, examples):
                raise ValueError(f'Unexpected directory: {rel}')
        for name in sorted(names):
            path = base / name
            rel = path.relative_to(ROOT).as_posix()
            if linked(path) or not path.is_file():
                raise ValueError(f'Non-regular or linked file: {rel}')
            if rel.casefold() in case_names:
                raise ValueError(f'Case-insensitive name collision: {rel}')
            case_names.add(rel.casefold())
            if path == MANIFEST:
                continue
            if rel in GENERATED:
                if sha(path) != GENERATED[rel]:
                    raise ValueError(f'Unexpected locally generated scene hash: {rel}')
                continue
            if path.suffix.lower() == '.xml' or not allowed_file(rel, examples):
                raise ValueError(f'Unexpected distribution file: {rel}')
            size = path.stat().st_size
            total += size
            if size > MAX_FILE_BYTES or total > MAX_TOTAL_BYTES:
                raise ValueError(f'Inventory byte budget exceeded: {rel}')
            record = {'bytes': size, 'sha256': sha(path)}
            if rel in examples and record != {k: examples[rel][k] for k in ('bytes', 'sha256')}:
                raise ValueError(f'Example file integrity differs: {rel}')
            files[rel] = record
    missing = (REQUIRED | set(examples)) - set(files)
    if missing:
        raise ValueError('Required files missing: ' + ', '.join(sorted(missing)))
    meshes = [name for name in files if name.lower().endswith('.stl')]
    if len(meshes) != 39:
        raise ValueError(f'Expected 39 model meshes, found {len(meshes)}.')
    mesh_manifest = json.loads((ROOT / 'assets/manifest.json').read_text(encoding='utf-8-sig'))
    declared_meshes = {item['package_path']: item for item in mesh_manifest['files']}
    if set(meshes) != set(declared_meshes):
        raise ValueError('Model mesh names differ from their source manifest.')
    for name, item in declared_meshes.items():
        if files[name] != {key: item[key] for key in ('bytes', 'sha256')}:
            raise ValueError(f'Model mesh differs from source manifest: {name}')
    return dict(sorted(files.items()))


def inspect(files):
    errors = []
    for name in files:
        path = ROOT / name
        if path.suffix.lower() not in TEXT_SUFFIXES and name not in {'LICENSE', '.gitignore'}:
            continue
        text = path.read_text(encoding='utf-8-sig')
        if re.search(r'(?<![A-Za-z0-9])[A-Za-z]:[\\/]', text):
            errors.append(f'Absolute drive path: {name}')
        if re.search(r'/(?:Users|home)/[^/\s]+/', text):
            errors.append(f'Private home path: {name}')
        if path.suffix.lower() == '.md':
            for target in re.findall(r'\]\(([^)]+)\)', text):
                if re.match(r'^[a-z]+:', target, re.I) or target.startswith('#'):
                    continue
                target = target.split('#', 1)[0]
                if target and not (path.parent / target).exists():
                    errors.append(f'Missing link in {name}: {target}')
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-manifest', action='store_true')
    parser.add_argument('--archive', type=Path)
    args = parser.parse_args()
    files, errors = {}, []
    try:
        files = inventory()
        errors.extend(inspect(files))
        if not errors and args.write_manifest:
            MANIFEST.write_text(json.dumps({
                'package': PACKAGE, 'status': STATUS,
                'inventory_excludes': sorted(LOCAL_ROOTS) + [MANIFEST.name],
                'local_generated_excluded_from_distribution': GENERATED,
                'files': files,
            }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        if not errors:
            if not MANIFEST.exists():
                errors.append('Manifest missing; use --write-manifest after preparation.')
            else:
                expected = json.loads(MANIFEST.read_text(encoding='utf-8-sig'))
                if expected.get('package') != PACKAGE or expected.get('status') != STATUS:
                    errors.append('Manifest package/version/status differs.')
                if expected.get('files') != files:
                    errors.append('Manifest inventory differs from package files.')
    except (ValueError, OSError, TypeError, KeyError) as error:
        errors.append(str(error))
    result = {'passed': not errors, 'package': PACKAGE, 'status': STATUS,
              'files': len(files), 'bytes': sum(v['bytes'] for v in files.values()),
              'errors': errors, 'physics_steps': 0, 'gui_opened': False,
              'scene_files_bundled': False, 'dynamic_validation_performed_by_this_check': False,
              'recorded_dynamic_evidence': 'evidence/v02-user-replay-verification.json',
              'note': 'This command checks distribution files only. Historical user dynamics and public release status remain separately recorded.'}
    if not errors and args.archive:
        archive_input = args.archive.absolute()
        for parent in (archive_input.parent, *archive_input.parent.parents):
            if parent.exists() and linked(parent):
                raise ValueError('Archive ancestry must not contain links or junctions.')
        archive = archive_input.resolve()
        if archive.exists():
            raise FileExistsError(f'Archive already exists: {archive}')
        if archive.is_relative_to(ROOT) or archive.suffix.lower() != '.zip':
            raise ValueError('Archive must be a new ZIP outside the package directory.')
        # Recheck after manifest creation, before opening an archive.
        if inventory() != files:
            raise ValueError('Package changed before archive creation.')
        archive.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as bundle:
            for name in sorted(list(files) + [MANIFEST.name]):
                bundle.write(ROOT / name, f'{ROOT.name}/{name}')
        with zipfile.ZipFile(archive) as bundle:
            bad = bundle.testzip()
            if bad:
                raise RuntimeError(f'Archive checksum error: {bad}')
            for name, item in files.items():
                if hashlib.sha256(bundle.read(f'{ROOT.name}/{name}')).hexdigest() != item['sha256']:
                    raise RuntimeError(f'Archive source changed while reading: {name}')
        result.update(archive=archive.name, archive_bytes=archive.stat().st_size,
                      archive_sha256=sha(archive))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == '__main__':
    raise SystemExit(main())
