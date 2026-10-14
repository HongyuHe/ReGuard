from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import tempfile

from _episode_identity import validate_episode_ids

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_FORMAT = 'reguard-measured-figure-data-v2'


def checked_path(root, relative):
    path = PurePosixPath(relative)
    if not relative or path.is_absolute() or '..' in path.parts:
        raise RuntimeError(f'Invalid result path: {relative!r}')
    candidate = root / path
    candidate.resolve().relative_to(root.resolve())
    return candidate


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def content_sha256(root, relative_paths):
    digest = hashlib.sha256()
    for relative in sorted(relative_paths, key=lambda value: PurePosixPath(value).parts):
        digest.update((relative + '\0' + file_sha256(checked_path(root, relative)) + '\n').encode())
    return digest.hexdigest()


def csv_rows(path, required):
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        if not set(required).issubset(reader.fieldnames or []):
            raise RuntimeError(f'Missing metric columns in {path}: {required}')
        rows = list(reader)
    if not rows:
        raise RuntimeError(f'Empty measurement file: {path}')
    return rows


def finite(rows, fields, path):
    for row in rows:
        for field in fields:
            if not math.isfinite(float(row[field])):
                raise RuntimeError(f'Nonfinite {field} in {path}')


def measurement_files(results, framework):
    root = results / 'evaluation'
    matrix = json.loads((root / 'evaluation_matrix.json').read_text())
    config = json.loads((ROOT / 'configs/paper_evaluation.json').read_text())
    expected = {(c.get('group', 'main'), c['name']): c for c in config['cases']}
    actual = {(c['group'], c['name']): c for c in matrix['cases']}
    if matrix.get('framework') != framework or actual.keys() != expected.keys() or len(actual) != len(matrix['cases']):
        raise RuntimeError('Incomplete or duplicate evaluation matrix')
    limit = matrix.get('trace_limit', 0)
    if not isinstance(limit, int) or limit < 0:
        raise RuntimeError('Invalid evaluation trace limit')
    for key, case in actual.items():
        count = min(expected[key]['count'], limit) if limit else expected[key]['count']
        if case['count'] != count:
            raise RuntimeError(f'Unexpected evaluation count for {key}')
    files = ['evaluation_matrix.json']
    if framework == 'pensieve':
        for group in sorted({key[0] for key in expected}):
            names = {name: actual[(g, name)]['count'] for g, name in expected if g == group}
            trace = f'eval_oracle_regret_traces_{group}.csv'
            step = f'eval_oracle_regret_steps_{group}.csv'
            rows = csv_rows(root / trace, ['trace_type', 'trace_name', 'num_steps', 'oracle_qoe_total', 'qoe_gap_total'])
            if dict(Counter(r['trace_type'] for r in rows)) != names:
                raise RuntimeError(f'Missing or unexpected traces in {trace}')
            finite(rows, ['oracle_qoe_total', 'qoe_gap_total'], trace)
            counts = {(r['trace_type'], r['trace_name']): int(r['num_steps']) for r in rows}
            steps = csv_rows(root / step, ['trace_type', 'trace_name', 'chunk_idx', 'oracle_k', 'regret'])
            if len(counts) != len(rows) or dict(Counter((r['trace_type'], r['trace_name']) for r in steps)) != counts:
                raise RuntimeError(f'Missing, duplicate, or truncated episode steps in {step}')
            for row in steps:
                n = counts[(row['trace_type'], row['trace_name'])]
                if not 0 <= int(row['chunk_idx']) < n:
                    raise RuntimeError(f'Invalid chunk index in {step}')
                if int(row['chunk_idx']) <= n - int(row['oracle_k']):
                    finite([row], ['regret'], step)
            if len({(r['trace_type'], r['trace_name'], r['chunk_idx']) for r in steps}) != len(steps):
                raise RuntimeError(f'Duplicate video chunks in {step}')
            files.extend([trace, step])
    elif framework == 'park':
        for case in actual.values():
            base = 'v2/' + case['name'] + '/'
            ep, step = base + 'episode_metrics.csv', base + 'step_metrics.csv'
            rows = csv_rows(root / ep, ['trace_id', 'num_jobs', 'oracle_total_reward', 'oracle_gap_total'])
            if len(rows) != case['count']:
                raise RuntimeError(f'Expected {case["count"]} episodes in {ep}; found {len(rows)}')
            finite(rows, ['oracle_total_reward', 'oracle_gap_total'], ep)
            counts = {r['trace_id']: int(r['num_jobs']) for r in rows}
            steps = csv_rows(root / step, ['trace_id', 'step_idx', 'regret'])
            if len(counts) != len(rows) or dict(Counter(r['trace_id'] for r in steps)) != counts:
                raise RuntimeError(f'Missing, duplicate, or truncated episode steps in {step}')
            finite(steps, ['regret'], step)
            if len({(r['trace_id'], r['step_idx']) for r in steps}) != len(steps):
                raise RuntimeError(f'Duplicate job steps in {step}')
            if any(not 0 <= int(r['step_idx']) < counts[r['trace_id']] for r in steps):
                raise RuntimeError(f'Invalid job step index in {step}')
            files.extend([ep, step])
    else:
        for case in actual.values():
            relative = 'eval-300k-50ms-new/' + case['name'] + '/eval_summary.json'
            payload = json.loads((root / relative).read_text())
            rows = payload['per_episode']
            if len(rows) != case['count']:
                raise RuntimeError(f'Expected {case["count"]} episodes in {relative}; found {len(rows)}')
            validate_episode_ids((row.get('episode_id') for row in rows), relative)
            finite(rows, ['gap_best_baseline_gap_mean', 'gap_best_baseline_score_mean'], relative)
            files.append(relative)
            timing = str(Path(relative).with_name('controller_timing_summary.json'))
            if (root / timing).is_file():
                files.append(timing)
    return files


def verify_plot_inputs(results_dir, framework):
    root = results_dir / 'evaluation'
    path = root / 'manifest.json'
    manifest = json.loads(path.read_text())
    if manifest.get('format') != MANIFEST_FORMAT or manifest.get('framework') != framework:
        raise RuntimeError(f'Unsupported plotting manifest: {path}')
    measured = measurement_files(results_dir, framework)
    references = json.loads((ROOT / 'configs/network_traces.json').read_text())['files']
    files = measured + [r['path'] for r in references]
    if sorted(files) != sorted(manifest['content_files']) or len(files) != len(set(files)):
        raise RuntimeError('Incomplete plotting file list')
    for item in references:
        if file_sha256(checked_path(root, item['path'])) != item['sha256']:
            raise RuntimeError(f'Invalid reference asset: {item["path"]}')
    if content_sha256(root, files) != manifest['content_sha256']:
        raise RuntimeError('Plot input checksum mismatch')
    return manifest


@contextmanager
def plot_input_snapshot(results_dir, framework):
    manifest = verify_plot_inputs(results_dir, framework)
    with tempfile.TemporaryDirectory(prefix=f'reguard-{framework}-plots-') as directory:
        workspace = Path(directory)
        inputs = workspace / 'inputs'
        for relative in manifest['content_files']:
            destination = checked_path(inputs, relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(checked_path(results_dir / 'evaluation', relative), destination)
        if content_sha256(inputs, manifest['content_files']) != manifest['content_sha256']:
            raise RuntimeError('Files changed while preparing plotting inputs')
        yield inputs, workspace / 'plots', manifest
