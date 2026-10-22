"""Verify the portable bundle and its SHA-256 inventory; no simulation imports."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'PACKAGE-MANIFEST.json'
IGNORED = {'.venv', '.cache', '__pycache__', 'outputs', '.git', 'upstream'}
GENERATED = {name: '9d71754df03ab06c4fbed3010ef9a883ce71a3e6ca04de01815d7d1c7ca2630c'
             for name in ('assets/scene.xml', 'configs/scene.xml')}


def inventory():
    return {
        p.relative_to(ROOT).as_posix(): {
            'bytes': p.stat().st_size,
            'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
        }
        for p in sorted(ROOT.rglob('*'))
        if p.is_file() and p != MANIFEST
        and p.relative_to(ROOT).as_posix() not in GENERATED
        and not any(part in IGNORED for part in p.relative_to(ROOT).parts)
    }


def inspect(files):
    errors = []
    for name in files:
        p = ROOT / name
        if p.suffix.lower() not in {'.md', '.py', '.ps1', '.json', '.txt', '.xml'}:
            continue
        text = p.read_text(encoding='utf-8-sig')
        if re.search(r'[A-Z]:[\\/](?:CSC|Users)[\\/]', text, re.I):
            errors.append(f'Private absolute path: {name}')
        if p.suffix == '.md':
            for target in re.findall(r'\]\(([^)]+)\)', text):
                if re.match(r'^[a-z]+:', target, re.I) or target.startswith('#'):
                    continue
                target = target.split('#', 1)[0]
                if target and not (p.parent / target).exists():
                    errors.append(f'Missing link in {name}: {target}')
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-manifest', action='store_true')
    parser.add_argument('--archive', type=Path)
    args = parser.parse_args()
    files = inventory()
    errors = inspect(files)
    if not errors and args.write_manifest:
        MANIFEST.write_text(json.dumps({
            'package': 'g1-manual-pick-place-v0.1.0',
            'status': 'source_candidate_requires_separate_xml_inputs',
            'inventory_excludes': sorted(IGNORED) + [MANIFEST.name],
            'local_generated_excluded_from_distribution': GENERATED,
            'files': files,
        }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if not errors:
        if not MANIFEST.exists():
            errors.append('Manifest missing; use --write-manifest after preparation.')
        elif json.loads(MANIFEST.read_text(encoding='utf-8'))['files'] != files:
            errors.append('Manifest inventory differs from package files.')
    result = {'passed': not errors, 'files': len(files),
              'bytes': sum(v['bytes'] for v in files.values()), 'errors': errors,
              'physics_steps': 0, 'gui_opened': False,
              'scene_files_bundled': False,
              'note': 'Source inventory only; supply local XML inputs and prepare the scene before runtime inspection.'}
    if not errors and args.archive:
        archive = args.archive.resolve()
        if archive.exists():
            raise FileExistsError(f'Archive already exists: {archive}')
        if archive.is_relative_to(ROOT):
            raise ValueError('Archive must be outside the package directory.')
        archive.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as bundle:
            for name in sorted(list(files) + [MANIFEST.name]):
                bundle.write(ROOT / name, f'{ROOT.name}/{name}')
        with zipfile.ZipFile(archive) as bundle:
            bad = bundle.testzip()
            if bad:
                raise RuntimeError(f'Archive checksum error: {bad}')
        result['archive'] = archive.name
        result['archive_bytes'] = archive.stat().st_size
        result['archive_sha256'] = hashlib.sha256(archive.read_bytes()).hexdigest()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == '__main__':
    raise SystemExit(main())
