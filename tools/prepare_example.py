"""Verify and materialize the bundled recording without stepping physics."""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime'))
sys.path.insert(0, str(ROOT / 'tools/demonstration'))
from lesson_common import guard
from portable_paths import contained_path, output_path, checked_path
from episode_schema import finalize_episode

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def require(ok, message):
    if not ok:
        raise ValueError(message)

def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def verify_bundle():
    base = checked_path(ROOT / 'examples/recording-001')
    manifest_path = contained_path(base, 'example-manifest.json')
    manifest = read(manifest_path)
    require(manifest.get('schema') == 'g1.portable_example.v0.2.0', 'Unknown example format')
    entries = manifest['files']
    require(isinstance(entries, dict) and 0 < len(entries) < 200, 'Unexpected example file count')
    actual = set()
    for path in base.rglob('*'):
        checked_path(path)
        if path.is_file():
            actual.add(path.relative_to(base).as_posix())
    require(actual == set(entries) | {'example-manifest.json'}, 'Example inventory differs')
    total = 0
    for name, item in entries.items():
        path = contained_path(base, name)
        require(path.is_file(), 'Expected regular file: ' + name)
        size = path.stat().st_size
        require(0 <= size == item['bytes'] <= 256*1024*1024, 'Example size differs: ' + name)
        total += size
        require(total <= 512*1024*1024, 'Example exceeds bounded total size')
        require(digest(path) == item['sha256'], 'Example hash differs: ' + name)
    external = manifest['external_files']
    require(set(external) == {'demonstration/configs/scene.xml'}, 'Unknown external example input')
    spec = external['demonstration/configs/scene.xml']
    require(spec['package_path'] == 'configs/scene.xml', 'Unknown external scene location')
    scene = contained_path(ROOT, spec['package_path'], exists=False)
    require(scene.is_file() and digest(scene) == spec['sha256'],
            'Prepare the exact local scene described in docs/scene-preparation.md')
    return base, manifest, scene, digest(manifest_path)

def materialize(destination):
    guard()
    dest = output_path(destination, exists=False)
    require(not dest.exists(), 'Example output already exists; choose a fresh path')
    base, manifest, scene, manifest_hash = verify_bundle()
    dest.mkdir(parents=True, exist_ok=False)
    for name in manifest['files']:
        source = contained_path(base, name)
        target = contained_path(dest, name, exists=False)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    shutil.copyfile(scene, dest / 'demonstration/configs/scene.xml')
    index = finalize_episode(dest / 'demonstration')
    require(index['complete_v02_episode'], 'Materialized episode failed: ' + index.get('validation_error', 'unknown'))
    require(index['references_sha256'] == manifest['references_sha256'], 'Reference bytes changed during export')
    require(index['action_count'] == manifest['action_count'] and index['state_count'] == manifest['state_count'],
            'Episode counts differ from bundled evidence')
    replay_path = dest / 'saved-replay/summary.json'
    original = read(replay_path)
    replay = dict(original)
    replay.update(source_run='../demonstration', episode_index='../demonstration/episode/episode-index.json',
                  source_summary_sha256=digest(dest / 'demonstration/summary.json'),
                  episode_index_sha256=digest(dest / 'demonstration/episode/episode-index.json'),
                  source_files_sha256=index['source_files_sha256'])
    changed = [key for key in replay if replay[key] != original.get(key)]
    require(set(changed) <= {'source_run', 'episode_index', 'source_summary_sha256', 'episode_index_sha256', 'source_files_sha256'},
            'Unexpected historical result edit')
    save(replay_path, replay)
    for name, item in manifest['files'].items():
        require(digest(contained_path(base, name)) == item['sha256'], 'Bundled input changed: ' + name)
        if name != 'saved-replay/summary.json':
            require(digest(contained_path(dest, name)) == item['sha256'], 'Copied data changed: ' + name)
    require(digest(base / 'example-manifest.json') == manifest_hash, 'Manifest changed')
    save(dest / 'materialization.json', {
        'schema': 'g1.example_materialization.v0.2.0', 'example_manifest_sha256': manifest_hash,
        'original_episode_index_sha256': manifest['original_episode_index_sha256'],
        'materialized_episode_index_sha256': replay['episode_index_sha256'],
        'bundled_replay_summary_sha256': manifest['files']['saved-replay/summary.json']['sha256'],
        'materialized_replay_summary_sha256': digest(replay_path),
        'changed_replay_metadata_fields': changed,
        'input_files_unchanged': len(manifest['files']) + 1,
        'physics_steps': 0, 'new_windows': 0,
        'note': 'Copied historical measurements; rebuilt path/hash metadata only. No new replay performed.',
    })
    print('EXAMPLE FORMAT: PASS | states: %d | actions: %d | 0 physics steps' % (index['state_count'], index['action_count']))
    print('EPISODE CHECK: ' + str(dest / 'demonstration/episode'))
    print('SAVED REPLAY: ' + str(dest / 'saved-replay'))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    materialize(args.output)

if __name__ == '__main__':
    main()
