"""Static portability/tamper regression checks; no simulator or installations.

Run through the configured package Python environment. Optionally pass a fully
materialized example episode index for one read-only positive replay-input check.
All tiny fixtures and the JSON report are retained under outputs/test_portable_*.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import stat
import sys
import traceback
from types import SimpleNamespace
from unittest import mock
import uuid


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'runtime'))
sys.path.insert(0, str(PROJECT / 'tools/demonstration'))
from lesson_common import guard
from portable_paths import output_path, contained_path, resolve_source, relative_source
import replay_episode as replay


RECORDED_MANIFEST_SHA = 'f6976fe149098b6dbfd757a8aae2f9412b131b20343364e9bb86fd3d646e0c56'
PACKAGE_MANIFEST_SHA = '0e4de1b785fe7823bd0395f1adf3f4fe898ce06675078c26f6b65826755c976b'
RECORDED_INTEGRITY_SHA = '59da1d5e730bebbf3339058c47274a5f14805a7001d831b88d72bd4ac4109aaa'
PACKAGE_INTEGRITY_SHA = '7b2d39d8bb195a3306d6032785faa88fef5fd449314585d96cbeeaf751ef4c3b'


def write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def expect(condition, message):
    if not condition:
        raise AssertionError(message)


def rejected(action, message_fragment, exception_types=(ValueError, FileNotFoundError)):
    try:
        action()
    except exception_types as exc:
        expect(message_fragment in str(exc), 'Wrong rejection: ' + str(exc))
        return {'rejection_type': type(exc).__name__, 'reason': str(exc)}
    raise AssertionError('Invalid input was accepted')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode-index', type=Path)
    args = parser.parse_args()
    guard()
    expect(not any(n == 'mujoco' or n.startswith('mujoco.') for n in sys.modules),
           'Static test must not import MuJoCo')
    outputs = output_path(PROJECT / 'outputs', exists=False)
    outputs.mkdir(exist_ok=True)
    folder = output_path(outputs / ('test_portable_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                                   + '_' + uuid.uuid4().hex[:6]), exists=False)
    folder.mkdir(exist_ok=False)
    source = folder / 'demonstration'
    metadata_dir = folder / 'index'
    source.mkdir()
    metadata_dir.mkdir()
    metadata = metadata_dir / 'episode-index.json'
    write_json(metadata, {'result': 'REFERENCE_TIMELINE_CHECK_PASSED', 'source_run': '../demonstration'})
    (source / 'small.txt').write_text('static portability fixture\n', encoding='utf-8')
    (source / 'assets').mkdir()
    write_json(source / 'assets/manifest.json', {'files': [], 'synthetic_fixture': True})
    (source / 'configs').mkdir()
    write_json(source / 'configs/runtime_integrity.json', {'files': {}, 'synthetic_fixture': True})
    cases = []

    def check(name, action):
        try:
            detail = action()
            cases.append({'name': name, 'status': 'passed', 'detail': detail})
        except Exception:
            cases.append({'name': name, 'status': 'failed', 'error': traceback.format_exc()})

    def positive_source():
        result = resolve_source(metadata, '../demonstration')
        expect(result == source, 'Relative provenance did not resolve to its sibling source')
        relative = relative_source(metadata_dir, source)
        expect(relative == '../demonstration', 'Source locator is not relocation-safe')
        expect(resolve_source(metadata, relative) == source, 'Relative source round trip failed')
        expect(contained_path(source, 'small.txt') == source / 'small.txt', 'Ordinary manifest path rejected')
        return {'relative_source': relative}

    check('relative_source_and_manifest_round_trip', positive_source)
    check('existing_absolute_source_inside_package_outputs',
          lambda: expect(resolve_source(metadata, str(source)) == source, 'Local absolute source rejected'))
    check('outside_outputs_source_rejected',
          lambda: rejected(lambda: resolve_source(metadata, str(PROJECT / 'runtime')), 'package outputs'))
    check('parent_traversal_outside_outputs_rejected',
          lambda: rejected(lambda: resolve_source(metadata, '../../../runtime'), 'package outputs'))
    check('missing_metadata_rejected',
          lambda: rejected(lambda: resolve_source(metadata_dir / 'missing.json', '../demonstration'),
                           'missing.json'))
    check('directory_metadata_rejected',
          lambda: rejected(lambda: resolve_source(metadata_dir, '../demonstration'), 'Metadata must be a file'))
    check('empty_source_rejected', lambda: rejected(lambda: resolve_source(metadata, ''), 'Missing source_run'))
    check('whitespace_source_rejected',
          lambda: rejected(lambda: resolve_source(metadata, '  '), 'Missing source_run'))
    check('file_used_as_source_rejected',
          lambda: rejected(lambda: resolve_source(metadata, '../demonstration/small.txt'), 'must be a directory'))
    check('absolute_manifest_path_rejected',
          lambda: rejected(lambda: contained_path(source, str(source / 'small.txt')), 'must be relative'))
    check('manifest_parent_escape_rejected',
          lambda: rejected(lambda: contained_path(source, '../index/episode-index.json'), 'escaped root'))
    check('missing_manifest_file_rejected',
          lambda: rejected(lambda: contained_path(source, 'absent.xml'), 'absent.xml'))
    if os.name == 'nt':
        check('drive_relative_source_rejected',
              lambda: rejected(lambda: resolve_source(metadata, PROJECT.drive + 'relative'), 'Drive-relative'))
    else:
        cases.append({'name': 'drive_relative_source_rejected', 'status': 'skipped',
                      'reason': 'Windows drive-relative path syntax only'})

    def mocked_reparse():
        original = Path.lstat

        def marked(path, *a, **kw):
            value = original(path, *a, **kw)
            if Path(os.path.abspath(path)) == source:
                return SimpleNamespace(st_mode=value.st_mode,
                    st_file_attributes=getattr(value, 'st_file_attributes', 0) |
                    getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400))
            return value

        with mock.patch.object(Path, 'lstat', marked):
            return rejected(lambda: resolve_source(metadata, '../demonstration'), 'Reparse paths')

    check('junction_or_reparse_source_rejected_mock', mocked_reparse)

    def replay_source_escape():
        bad = metadata_dir / 'escaping-index.json'
        write_json(bad, {'result': 'REFERENCE_TIMELINE_CHECK_PASSED', 'source_run': str(PROJECT / 'runtime')})
        return rejected(lambda: replay.verify_input(bad), 'package outputs')

    check('replay_rejects_source_escape_before_loading_records', replay_source_escape)
    check('unmapped_runtime_change_rejected',
          lambda: rejected(lambda: replay.verify_snapshot(source, {
              'path': 'runtime/manual_pickplace.py', 'sha256': '0' * 64}), 'Package differs'))
    check('exact_current_manifest_snapshot_accepted',
          lambda: replay.verify_snapshot(source, {'path': 'assets/manifest.json',
                                                  'sha256': PACKAGE_MANIFEST_SHA}))
    check('exact_current_integrity_snapshot_accepted',
          lambda: replay.verify_snapshot(source, {'path': 'configs/runtime_integrity.json',
                                                  'sha256': PACKAGE_INTEGRITY_SHA}))
    check('unknown_recorded_manifest_hash_rejected',
          lambda: rejected(lambda: replay.verify_snapshot(source, {
              'path': 'assets/manifest.json', 'sha256': '0' * 64}), 'Package differs'))

    def mapping_rejected(mapping):
        original = replay.read_json
        compatibility_path = PROJECT / 'evidence/runtime-compatibility.json'

        def altered(path):
            if Path(path) == compatibility_path:
                return {'snapshot_mappings': mapping}
            return original(path)

        with mock.patch.object(replay, 'read_json', side_effect=altered):
            return rejected(lambda: replay.verify_snapshot(source, {
                'path': 'assets/manifest.json', 'sha256': RECORDED_MANIFEST_SHA}),
                'Unsupported runtime compatibility mapping')

    canonical_mapping = {'path': 'assets/manifest.json', 'recorded_sha256': RECORDED_MANIFEST_SHA,
                         'package_sha256': PACKAGE_MANIFEST_SHA}
    integrity_mapping = {'path': 'configs/runtime_integrity.json', 'recorded_sha256': RECORDED_INTEGRITY_SHA,
                         'package_sha256': PACKAGE_INTEGRITY_SHA}
    check('missing_compatibility_mapping_rejected', lambda: mapping_rejected([]))
    check('wrong_recorded_hash_in_mapping_rejected', lambda: mapping_rejected([
        dict(canonical_mapping, recorded_sha256='1' * 64), integrity_mapping]))
    check('wrong_package_hash_in_mapping_rejected', lambda: mapping_rejected([
        dict(canonical_mapping, package_sha256='2' * 64), integrity_mapping]))
    check('wrong_integrity_hash_in_mapping_rejected', lambda: mapping_rejected([
        canonical_mapping, dict(integrity_mapping, package_sha256='3' * 64)]))
    check('extra_compatibility_exception_rejected', lambda: mapping_rejected([
        canonical_mapping, integrity_mapping, {'path': 'configs/config.json', 'recorded_sha256': '0' * 64,
                            'package_sha256': '0' * 64}]))
    check('changed_historical_manifest_rejected',
          lambda: rejected(lambda: replay.verify_snapshot(source, {'path': 'assets/manifest.json',
              'sha256': RECORDED_MANIFEST_SHA}), 'Historical metadata snapshot changed'))
    check('changed_historical_integrity_rejected',
          lambda: rejected(lambda: replay.verify_snapshot(source, {'path': 'configs/runtime_integrity.json',
              'sha256': RECORDED_INTEGRITY_SHA}), 'Historical metadata snapshot changed'))

    if args.episode_index:
        def valid_example():
            index_path = output_path(args.episode_index)
            index_before = replay.digest(index_path)
            locator = replay.read_json(index_path)
            run = resolve_source(index_path, locator['source_run'])
            watched = [index_path, run / 'summary.json', run / 'assets/manifest.json']
            before = {str(p): replay.digest(p) for p in watched}
            index, actual_run, summary, refs, states, commands = replay.verify_input(index_path)
            expect(actual_run == run and len(refs) == index['action_count'] and
                   len(states) == index['state_count'], 'Example export does not match its metadata')
            expect(replay.digest(index_path) == index_before and
                   all(replay.digest(Path(p)) == value for p, value in before.items()),
                   'Read-only input verification changed its inputs')
            return {'source_run': relative_source(folder, run), 'actions': len(refs), 'states': len(states),
                    'command_count': len(commands), 'historical_status': summary['status'],
                    'no_csv_copy': True, 'metadata_unchanged': True}

        check('real_example_input_and_manifest_migration', valid_example)

        def non_manifest_integrity_edit():
            index_path = output_path(args.episode_index)
            run = resolve_source(index_path, replay.read_json(index_path)['source_run'])
            integrity_path = PROJECT / 'configs/runtime_integrity.json'
            original = replay.read_json

            def altered(path):
                value = original(path)
                if Path(path) == integrity_path:
                    value['files']['runtime/manual_pickplace.py'] = '4' * 64
                return value

            with mock.patch.object(replay, 'read_json', side_effect=altered):
                return rejected(lambda: replay.verify_snapshot(run, {
                    'path': 'configs/runtime_integrity.json', 'sha256': RECORDED_INTEGRITY_SHA}),
                    'Integrity migration changed fields beyond')

        check('integrity_mapping_cannot_change_runtime_hashes_mock', non_manifest_integrity_edit)
    else:
        cases.append({'name': 'real_example_input_and_manifest_migration', 'status': 'skipped',
                      'reason': 'Pass --episode-index for a materialized example'})
    check('no_mujoco_imported', lambda: expect(
        not any(n == 'mujoco' or n.startswith('mujoco.') for n in sys.modules), 'MuJoCo was imported'))
    report = {'kind': 'g1_v02_portable_boundary_tests', 'new_physics_steps': 0,
              'new_windows': 0, 'fixtures': folder.relative_to(PROJECT).as_posix(),
              'passed': sum(c['status'] == 'passed' for c in cases),
              'failed': sum(c['status'] == 'failed' for c in cases),
              'skipped': sum(c['status'] == 'skipped' for c in cases), 'cases': cases}
    report_path = folder / 'tests.json'
    write_json(report_path, report)
    print(json.dumps({'passed': report['passed'], 'failed': report['failed'], 'skipped': report['skipped'],
                      'new_physics_steps': 0, 'report': str(report_path)}, ensure_ascii=False))
    return 1 if report['failed'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
