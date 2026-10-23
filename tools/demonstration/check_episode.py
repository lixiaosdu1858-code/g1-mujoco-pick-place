"""Validate saved records and export references without changing the source run.

Complete episodes use the complete schema; legacy records retain their explicit
missing-data labels. This entry point imports no MuJoCo and runs no physics.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'runtime'))
from lesson_common import guard

from inspect_episode import inspect, read_json, require
from episode_schema import FORMAT, finalize_episode
from portable_paths import output_path, contained_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    guard()
    run = output_path(args.run)
    output = output_path(args.output, exists=False)
    require(run.is_dir() and not output.is_relative_to(run),
            'Check output must be separate from the unchanged source run')
    require(not output.exists(), 'Use a fresh check output directory')
    summary = read_json(contained_path(run, 'summary.json'))
    if summary.get('episode_format') == FORMAT:
        index = finalize_episode(run, output)
        print('REFERENCE TIMELINE:', 'PASS' if index['complete_v02_episode'] else 'FAIL')
        print('Complete v0.2 episode:', index['complete_v02_episode'], '| dynamic replay: NOT RUN')
        print('EPISODE INDEX:', output / 'episode-index.json')
        print('REPORT:', output / '检查报告.md')
        return 0 if index['complete_v02_episode'] else 2
    require(summary.get('episode_format') in (None, ''), 'Unsupported episode format; no legacy downgrade')
    inspect(run, output)
    print('EPISODE INDEX:', output / 'episode-index.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
