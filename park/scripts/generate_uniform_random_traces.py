#!/usr/bin/env python3
"""
Example uniform random traces from the repo root:
.venv/bin/python scripts/generate_uniform_random_traces.py \
  --out-dir traces/uniform_random \
  --num-traces 100 \
  --num-jobs 1000 \
  --dt-min-ms 1 \
  --dt-max-ms 100 \
  --job-size-min-bytes 100 \
  --job-size-max-bytes 1000 \
  --service-rate 1.0 \
  --seed 42

Example random per-trace job counts and service rates:
.venv/bin/python scripts/generate_uniform_random_traces.py \
  --out-dir traces/uniform_random_varied \
  --num-traces 100 \
  --num-jobs-min 100 \
  --num-jobs-max 2000 \
  --dt-min-ms 0 \
  --dt-max-ms 10 \
  --job-size-min-bytes 1 \
  --job-size-max-bytes 10000 \
  --service-rate-min 0.1 \
  --service-rate-max 10 \
  --queue-shuffled-probability 0.5 \
  --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TraceConfig:
    num_jobs: int
    dt_min_ms: float
    dt_max_ms: float
    job_size_min_bytes: int
    job_size_max_bytes: int
    service_rate: float | None
    queue_shuffled_probability: float | None
    inter_arrival_distribution: str = "uniform"
    job_size_distribution: str = "uniform"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate uniformly random load-balance workload traces.")
    p.add_argument("--out-dir", type=str, required=True, help="Output directory")
    p.add_argument("--num-traces", type=int, default=100, help="How many traces to generate")
    p.add_argument("--num-servers", type=int, default=10, help="Server count for service_rates metadata")
    p.add_argument("--num-jobs", type=int, default=1000, help="Fixed jobs per trace")
    p.add_argument("--num-jobs-min", type=int, default=None, help="Uniform lower bound for per-trace job count")
    p.add_argument("--num-jobs-max", type=int, default=None, help="Uniform upper bound for per-trace job count")
    p.add_argument("--dt-min-ms", type=float, default=0.0, help="Uniform lower bound for inter-arrival time")
    p.add_argument("--dt-max-ms", type=float, default=100.0, help="Uniform upper bound for inter-arrival time")
    p.add_argument("--job-size-min-bytes", type=int, default=100, help="Uniform lower bound for job size")
    p.add_argument("--job-size-max-bytes", type=int, default=1000, help="Uniform upper bound for job size")
    p.add_argument("--service-rate", type=float, default=None, help="Fixed identical service rate metadata")
    p.add_argument("--service-rate-min", type=float, default=None, help="Uniform lower bound for service rate metadata")
    p.add_argument("--service-rate-max", type=float, default=None, help="Uniform upper bound for service rate metadata")
    p.add_argument("--queue-shuffled-probability", type=float, default=None, help="Queue shuffle probability metadata")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument(
        "--time-mode",
        type=str,
        default="absolute",
        choices=["absolute", "delta"],
        help="Legacy CSV time alias: arrival_time or dt",
    )
    return p.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if int(args.num_traces) <= 0:
        raise ValueError("num_traces must be positive")
    if int(args.num_jobs) <= 0:
        raise ValueError("num_jobs must be positive")
    if args.num_jobs_min is not None and int(args.num_jobs_min) <= 0:
        raise ValueError("num_jobs_min must be positive")
    if args.num_jobs_max is not None and int(args.num_jobs_max) <= 0:
        raise ValueError("num_jobs_max must be positive")
    if (args.num_jobs_min is None) != (args.num_jobs_max is None):
        raise ValueError("num_jobs_min and num_jobs_max must be provided together")
    if args.num_jobs_min is not None and int(args.num_jobs_min) > int(args.num_jobs_max):
        raise ValueError("num_jobs_min must be at most num_jobs_max")
    if float(args.dt_min_ms) < 0:
        raise ValueError("dt_min_ms must be non-negative")
    if float(args.dt_min_ms) > float(args.dt_max_ms):
        raise ValueError("dt_min_ms must be at most dt_max_ms")
    if int(args.job_size_min_bytes) <= 0:
        raise ValueError("job_size_min_bytes must be positive")
    if int(args.job_size_min_bytes) > int(args.job_size_max_bytes):
        raise ValueError("job_size_min_bytes must be at most job_size_max_bytes")
    if args.service_rate is not None and float(args.service_rate) <= 0:
        raise ValueError("service_rate must be positive")
    if (args.service_rate_min is None) != (args.service_rate_max is None):
        raise ValueError("service_rate_min and service_rate_max must be provided together")
    if args.service_rate_min is not None and float(args.service_rate_min) <= 0:
        raise ValueError("service_rate_min must be positive")
    if args.service_rate_max is not None and float(args.service_rate_max) <= 0:
        raise ValueError("service_rate_max must be positive")
    if args.service_rate_min is not None and float(args.service_rate_min) > float(args.service_rate_max):
        raise ValueError("service_rate_min must be at most service_rate_max")
    if args.service_rate is not None and args.service_rate_min is not None:
        raise ValueError("use either service_rate or service_rate_min/service_rate_max, not both")
    if args.queue_shuffled_probability is not None:
        q = float(args.queue_shuffled_probability)
        if q < 0.0 or q > 1.0:
            raise ValueError("queue_shuffled_probability must be in [0, 1]")


def _sample_num_jobs(args: argparse.Namespace, rng: np.random.Generator) -> int:
    if args.num_jobs_min is None:
        return int(args.num_jobs)
    return int(rng.integers(int(args.num_jobs_min), int(args.num_jobs_max) + 1))


def _sample_service_rate(args: argparse.Namespace, rng: np.random.Generator) -> float | None:
    if args.service_rate is not None:
        return float(args.service_rate)
    if args.service_rate_min is not None:
        return float(rng.uniform(float(args.service_rate_min), float(args.service_rate_max)))
    return None


def sample_trace_config(args: argparse.Namespace, rng: np.random.Generator) -> TraceConfig:
    return TraceConfig(
        num_jobs=_sample_num_jobs(args, rng),
        dt_min_ms=float(args.dt_min_ms),
        dt_max_ms=float(args.dt_max_ms),
        job_size_min_bytes=int(args.job_size_min_bytes),
        job_size_max_bytes=int(args.job_size_max_bytes),
        service_rate=_sample_service_rate(args, rng),
        queue_shuffled_probability=(
            None if args.queue_shuffled_probability is None else float(args.queue_shuffled_probability)
        ),
    )


def sample_jobs(rng: np.random.Generator, cfg: TraceConfig) -> list[dict[str, float | int]]:
    jobs: list[dict[str, float | int]] = []
    arrival_time_ms = 0.0
    for i in range(int(cfg.num_jobs)):
        if i == 0:
            dt_ms = 0.0
        else:
            dt_ms = float(rng.uniform(float(cfg.dt_min_ms), float(cfg.dt_max_ms)))
            arrival_time_ms += max(0.0, dt_ms)
        job_size_bytes = int(rng.integers(int(cfg.job_size_min_bytes), int(cfg.job_size_max_bytes) + 1))
        jobs.append(
            {
                "job_id": int(i),
                "arrival_time_ms": float(arrival_time_ms),
                "dt_ms": float(dt_ms),
                "job_size_bytes": int(job_size_bytes),
                "arrival_time": int(round(arrival_time_ms)),
                "dt": int(round(dt_ms)),
                "job_size": int(job_size_bytes),
            }
        )
    return jobs


def _csv_fieldnames(time_mode: str) -> list[str]:
    if time_mode == "absolute":
        return ["job_id", "arrival_time_ms", "dt_ms", "job_size_bytes", "arrival_time", "job_size"]
    return ["job_id", "arrival_time_ms", "dt_ms", "job_size_bytes", "dt", "job_size"]


def _metadata(trace_id: str, cfg: TraceConfig, num_servers: int) -> dict[str, Any]:
    data = asdict(cfg)
    data.update(
        {
            "trace_id": trace_id,
            "arrival_process": "uniform_random_inter_arrival",
            "service_model": "identical_service_rates",
            "service_rate_unit": "bytes_per_ms",
            "num_servers": int(num_servers),
            "service_rates": (
                None if cfg.service_rate is None else [float(cfg.service_rate) for _ in range(int(num_servers))]
            ),
            "legacy_time_unit": "rounded_ms",
        }
    )
    return data


def _write_csv(path: Path, jobs: list[dict[str, float | int]], time_mode: str) -> None:
    fieldnames = _csv_fieldnames(time_mode)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for job in jobs:
            writer.writerow({k: job[k] for k in fieldnames})


def _write_json(path: Path, trace_id: str, cfg: TraceConfig, jobs: list[dict[str, float | int]], num_servers: int) -> None:
    payload = _metadata(trace_id, cfg, num_servers)
    payload["jobs"] = jobs
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _index_row(trace_id: str, csv_path: Path, json_path: Path, cfg: TraceConfig) -> dict[str, Any]:
    row = _metadata(trace_id, cfg, num_servers=0)
    row["csv_path"] = csv_path.name
    row["json_path"] = json_path.name
    row.pop("service_rates", None)
    row.pop("num_servers", None)
    return row


def main() -> None:
    args = parse_args()
    _validate_args(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    index_rows: list[dict[str, Any]] = []
    for i in range(int(args.num_traces)):
        trace_id = f"uniform_trace_{i:05d}"
        cfg = sample_trace_config(args, rng)
        jobs = sample_jobs(rng, cfg)

        csv_path = out_dir / f"{trace_id}.csv"
        json_path = out_dir / f"{trace_id}.json"

        _write_csv(csv_path, jobs, str(args.time_mode))
        _write_json(json_path, trace_id, cfg, jobs, int(args.num_servers))
        index_rows.append(_index_row(trace_id, csv_path, json_path, cfg))

    index_path = out_dir / "index.csv"
    index_fieldnames = list(index_rows[0].keys()) if len(index_rows) > 0 else []
    with index_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=index_fieldnames)
        writer.writeheader()
        writer.writerows(index_rows)

    print(f"Generated {len(index_rows)} uniform random workload traces in: {out_dir}")
    print(f"Index: {index_path}")


if __name__ == "__main__":
    main()
