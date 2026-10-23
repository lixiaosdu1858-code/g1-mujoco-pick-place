"""Resolve portable provenance within this package, rejecting reparse escapes.

Relative source_run values are relative to the metadata file containing them.
They may contain '..', but both their lexical ancestors and resolved destination
must remain ordinary paths and the destination must be in package outputs.
"""
import os
from pathlib import Path
import stat


PROJECT = Path(__file__).resolve().parents[2]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_path(path, *, exists=True):
    """Check the original path before resolve() can hide a junction/symlink."""
    original = Path(path)
    if not original.is_absolute():
        original = Path.cwd() / original
    for item in (original, *original.parents):
        try:
            info = item.lstat()
        except FileNotFoundError:
            continue
        require(not stat.S_ISLNK(info.st_mode) and
                not (getattr(info, 'st_file_attributes', 0) &
                     getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)),
                'Reparse paths are not allowed: ' + str(item))
    return original.resolve(strict=exists)


def output_path(path, *, exists=True):
    """A run or metadata path under this package's outputs, including examples."""
    result = checked_path(path, exists=exists)
    root = checked_path(PROJECT / 'outputs', exists=False)
    require(result.is_relative_to(root), 'Path must stay in this package outputs: ' + str(path))
    return result


def contained_path(root, relative, *, exists=True):
    """Manifest entry confined to an ordinary root; absolute entries rejected."""
    base = checked_path(root)
    value = Path(relative)
    require(not value.is_absolute() and not value.drive and str(value) not in ('', '.'),
            'Manifest path must be relative: ' + str(relative))
    result = checked_path(base / value, exists=exists)
    require(result.is_relative_to(base), 'Manifest path escaped root: ' + str(relative))
    return result


def resolve_source(metadata_path, value):
    """Resolve source_run relative to its index/summary file, never the cwd."""
    metadata = output_path(metadata_path)
    require(metadata.is_file(), 'Metadata must be a file: ' + str(metadata))
    require(isinstance(value, str) and value.strip(), 'Missing source_run')
    candidate = Path(value)
    require(not candidate.drive or candidate.is_absolute(), 'Drive-relative source_run is not supported')
    source = output_path(candidate if candidate.is_absolute() else metadata.parent / candidate)
    require(source.is_dir(), 'Source run must be a directory')
    return source


def relative_source(from_dir, to_run):
    """Create a relocation-safe path from a metadata directory to its source."""
    base = output_path(from_dir)
    target = output_path(to_run)
    require(base.is_dir() and target.is_dir(), 'Source and metadata directory must exist')
    return Path(os.path.relpath(target, base)).as_posix()
