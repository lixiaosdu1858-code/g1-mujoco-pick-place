"""Portable package storage guard and unchanged JSON/hash helpers."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from datetime import datetime, timezone, timedelta

ROOT = Path(__file__).resolve().parents[1]
TEMP = Path(os.environ.get('G1_PACKAGE_CACHE', ROOT/'.cache'))/'temp'


def guard():
    expected = os.environ.get('G1_PACKAGE_PYTHON')
    if not expected or Path(sys.executable).resolve() != Path(expected).resolve():
        raise RuntimeError('Start through package Run.ps1.')
    if sys.prefix == sys.base_prefix:
        raise RuntimeError('Use an isolated virtual environment.')
    for name in ('TEMP', 'TMP', 'TMPDIR'):
        if not os.environ.get(name) or Path(os.environ[name]).resolve() != TEMP.resolve():
            raise RuntimeError('Package temporary directory not configured: ' + name)
    with tempfile.TemporaryFile(dir=TEMP) as probe:
        probe.write(b'arm lesson E-drive probe')
    if Path(tempfile.gettempdir()).resolve() != TEMP.resolve():
        raise RuntimeError('Temporary directory fallback detected.')


def stamp():
    return datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
