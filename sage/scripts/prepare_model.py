#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

from figure_inputs import MANIFEST_FORMAT, checked_path, content_sha256, file_sha256, measurement_files, verify_plot_inputs

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    framework = ROOT.name
    cache = Path(os.environ.get('REGUARD_REFERENCE_CACHE', '/tmp/.7f9c2a64e18b43d0a5c806f12bd973ee')) / framework
    references = json.loads((ROOT / 'configs/network_traces.json').read_text())['files']
    for item in references:
        path = checked_path(cache, item['path'])
        if not path.is_file() or file_sha256(path) != item['sha256']:
            raise RuntimeError(f'Required reference asset missing or invalid: {path}')
    if args.check_only:
        print(f'{framework}: reference assets verified')
        return
    results = args.results_dir.expanduser().resolve()
    measured = measurement_files(results, framework)
    root = results / 'evaluation'
    for item in references:
        destination = checked_path(root, item['path'])
        if destination.exists():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(checked_path(cache, item['path']), destination)
    files = measured + [r['path'] for r in references]
    manifest = {'format': MANIFEST_FORMAT, 'framework': framework, 'content_files': files,
                'content_sha256': content_sha256(root, files),
                'measured_files': measured, 'reference_files': [r['path'] for r in references]}
    with (root / 'manifest.json').open('x') as stream:
        json.dump(manifest, stream, indent=2)
        stream.write('\n')
    verify_plot_inputs(results, framework)
    print(f'{framework}: complete plotting inputs verified')


if __name__ == '__main__':
    main()
