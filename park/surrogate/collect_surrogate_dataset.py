#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from surrogate.common import collect_trace_files, infer_control_bounds_from_jobs, parse_service_rates, resolve_path
from surrogate.data import collect_transition_dataset, save_transition_dataset
from online.sim_core import parse_job_trace


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect exact Park transition data for the PGD surrogate.")
    p.add_argument("--victim-checkpoint", type=str, required=True)
    p.add_argument("--trace-dir", action="append", default=None, help="Trace directory to include")
    p.add_argument("--trace-path", action="append", default=None, help="Individual trace path to include")
    p.add_argument("--out", type=str, required=True, help="Output .npz dataset path")
    p.add_argument("--num-servers", type=int, default=10)
    p.add_argument("--service-rates", type=str, default=None)
    p.add_argument("--obs-high", type=float, default=500000.0)
    p.add_argument("--reward-time-scale", type=float, default=1000.0)
    p.add_argument("--oracle-k", type=int, default=5)
    p.add_argument("--dt-min", type=float, default=None)
    p.add_argument("--dt-max", type=float, default=None)
    p.add_argument("--size-min", type=float, default=None)
    p.add_argument("--size-max", type=float, default=None)
    p.add_argument("--max-traces", type=int, default=None)
    p.add_argument("--max-steps-per-trace", type=int, default=None)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def _resolve_trace_files(args: argparse.Namespace) -> list[Path]:
    trace_files: list[Path] = []
    for trace_dir in list(args.trace_dir or []):
        trace_files.extend(collect_trace_files(resolve_path(str(trace_dir))))
    for trace_path in list(args.trace_path or []):
        trace_files.append(resolve_path(str(trace_path)))
    if len(trace_files) == 0:
        raise ValueError("provide at least one --trace-dir or --trace-path")
    trace_files = sorted(dict.fromkeys(trace_files))
    if args.max_traces is not None:
        trace_files = trace_files[: int(args.max_traces)]
    return trace_files


def _infer_bounds(trace_files: list[Path], args: argparse.Namespace) -> dict[str, float]:
    if (
        args.dt_min is not None
        and args.dt_max is not None
        and args.size_min is not None
        and args.size_max is not None
    ):
        return {
            "dt_min": float(args.dt_min),
            "dt_max": float(args.dt_max),
            "size_min": float(args.size_min),
            "size_max": float(args.size_max),
        }
    merged_jobs = []
    for trace_path in list(trace_files):
        merged_jobs.extend(parse_job_trace(str(trace_path)))
    inferred = infer_control_bounds_from_jobs(merged_jobs)
    if args.dt_min is not None:
        inferred["dt_min"] = float(args.dt_min)
    if args.dt_max is not None:
        inferred["dt_max"] = float(args.dt_max)
    if args.size_min is not None:
        inferred["size_min"] = float(args.size_min)
    if args.size_max is not None:
        inferred["size_max"] = float(args.size_max)
    return inferred


def main() -> None:
    args = parse_args()
    trace_files = _resolve_trace_files(args)
    service_rates = parse_service_rates(
        args.service_rates,
        num_servers=int(args.num_servers),
        checkpoint_config=None,
    )
    bounds = _infer_bounds(trace_files, args)
    arrays, meta = collect_transition_dataset(
        trace_files=trace_files,
        victim_checkpoint=str(resolve_path(str(args.victim_checkpoint))),
        service_rates=list(service_rates),
        num_servers=int(args.num_servers),
        obs_high=float(args.obs_high),
        reward_time_scale=float(args.reward_time_scale),
        oracle_k=int(args.oracle_k),
        dt_min=float(bounds["dt_min"]),
        dt_max=float(bounds["dt_max"]),
        size_min=float(bounds["size_min"]),
        size_max=float(bounds["size_max"]),
        max_steps_per_trace=None if args.max_steps_per_trace is None else int(args.max_steps_per_trace),
        device=str(args.device),
    )
    out_path = resolve_path(str(args.out))
    save_transition_dataset(
        out_npz=out_path,
        arrays=arrays,
        meta=meta,
    )
    print(f"saved dataset: {out_path}")
    print(f"samples: {meta['num_samples']}")


if __name__ == "__main__":
    main()
