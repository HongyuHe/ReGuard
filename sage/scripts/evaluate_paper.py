#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK = ROOT.name


def run_case(case, results, limit, jobs):
    name = case['name']
    count = min(case['count'], limit) if limit else case['count']
    case_id = case.get('group', 'main') + '/' + name
    work = results / 'measurements' / case_id
    work.mkdir(parents=True, exist_ok=False)
    args = list(case['args'])
    if FRAMEWORK == 'park':
        script = 'eval_trace_dir_oracle_regret.py'
        output = results / 'evaluation' / 'v2' / name
        args += ['--eval-count', str(count), '--trace-limit', str(count), '--out-dir', str(output)]
    elif FRAMEWORK == 'pensieve':
        script = 'eval_oracle_regret.py'
        args += ['--trace-num', str(count), '--jobs', str(jobs),
                 '--out-trace-csv', str(work / 'traces.csv'),
                 '--out-step-csv', str(work / 'steps.csv')]
    else:
        script = 'eval_sage_clean_vs_adv.py'
        output = results / 'evaluation' / 'eval-300k-50ms-new'
        for flag, field in [('--generated-manifest', 'generated_entries'), ('--test-manifest', 'entries')]:
            if flag not in args:
                continue
            index = args.index(flag) + 1
            payload = json.loads((ROOT / args[index]).read_text())
            if len(payload[field]) < count:
                raise RuntimeError(f'{name}: fewer input traces than required ({count})')
            payload[field] = payload[field][:count]
            path = work / (field + '.json')
            path.write_text(json.dumps(payload, indent=2) + '\n')
            args[index] = str(path)
        #* Share the native port/actor lease lock across all concurrent Sage cases.
        args += ['--out-dir', str(output), '--runtime-dir', str(results / 'runtime' / 'paper_evaluation')]
    command = [sys.executable, str(ROOT / 'scripts' / script), *args]
    record = {'case': case_id, 'trace_count': count, 'command': command,
              'source': 'fresh_evaluation', 'started_at': time.time()}
    (work / 'command.json').write_text(json.dumps(record, indent=2) + '\n')
    print(f'Evaluating {case_id}: {count} traces', flush=True)
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', WANDB_MODE='disabled')
    with (work / 'evaluation.log').open('w') as log:
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode:
        record['returncode'] = completed.returncode
        record['elapsed_seconds'] = time.time() - record['started_at']
        (work / 'command.json').write_text(json.dumps(record, indent=2) + '\n')
        raise RuntimeError(f'{case_id} exited with {completed.returncode}; inspect {work / "evaluation.log"}')
    record['elapsed_seconds'] = time.time() - record['started_at']
    (work / 'command.json').write_text(json.dumps(record, indent=2) + '\n')
    return {'name': name, 'group': case.get('group', 'main'), 'count': count}


def merge_pensieve(results, cases):
    for group in sorted({case['group'] for case in cases}):
        for kind in ['traces', 'steps']:
            destination = results / 'evaluation' / f'eval_oracle_regret_{kind}_{group}.csv'
            with destination.open('x', newline='') as out:
                writer = None
                for case in cases:
                    if case['group'] != group:
                        continue
                    source = results / 'measurements' / group / case['name'] / f'{kind}.csv'
                    with source.open(newline='') as stream:
                        reader = csv.DictReader(stream)
                        if writer is None:
                            writer = csv.DictWriter(out, fieldnames=reader.fieldnames)
                            writer.writeheader()
                        if reader.fieldnames != writer.fieldnames:
                            raise RuntimeError(f'Inconsistent CSV schema: {source}')
                        writer.writerows(reader)


def main():
    parser = argparse.ArgumentParser(description='Evaluate the complete controller/trace/shield matrix.')
    parser.add_argument('--results-dir', type=Path, required=True)
    parser.add_argument('--trace-limit', type=int, default=0)
    parser.add_argument('--workers', type=int, default=1 if FRAMEWORK == 'sage' else 4)
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--check-inputs', action='store_true')
    args = parser.parse_args()
    if args.trace_limit < 0 or args.workers < 1 or args.jobs < 1:
        parser.error('Trace limit must be nonnegative; workers and jobs must be positive.')
    config = json.loads((ROOT / 'configs/paper_evaluation.json').read_text())
    path_flags = {'--model', '--victim-checkpoint', '--trace', '--trace-dir',
                  '--shield-rules-prefix', '--shield-min-rules-file', '--shield-max-rules-file',
                  '--generated-manifest', '--test-manifest', '--config-path',
                  '--sage-eval-ckpt-dir', '--shield-rules-file'}
    for case in config['cases']:
        for index, flag in enumerate(case['args']):
            if flag in path_flags:
                path = ROOT / case['args'][index + 1]
                if flag == '--shield-rules-prefix':
                    paths = [Path(str(path) + '_risk.pl')] + [Path(str(path) + f'_allow_server{i}.pl') for i in range(10)]
                else:
                    paths = [path]
                for required in paths:
                    if not required.exists():
                        raise FileNotFoundError(f'{case["name"]}: required evaluation input {required}')
    if args.check_inputs:
        print(f'{FRAMEWORK}: inputs for {len(config["cases"])} evaluation cases verified')
        return
    results = args.results_dir.expanduser().resolve()
    (results / 'evaluation').mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_case, case, results, args.trace_limit, args.jobs) for case in config['cases']]
        completed = []
        try:
            for future in as_completed(futures):
                completed.append(future.result())
        except Exception:
            for future in futures:
                future.cancel()
            raise
    if FRAMEWORK == 'pensieve':
        merge_pensieve(results, config['cases'])
    report = {'framework': FRAMEWORK, 'cases': completed,
              'trace_limit': args.trace_limit, 'source': config['source']}
    (results / 'evaluation/evaluation_matrix.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Completed {len(completed)} measured evaluation cases', flush=True)


if __name__ == '__main__':
    main()
