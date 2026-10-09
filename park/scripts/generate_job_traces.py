#!/usr/bin/env python3
"""
Example RL1 traces from the repo root:
.venv/bin/python scripts/generate_job_traces.py \
  --out-dir traces/rl1 \
  --num-traces 100 \
  --workload-family rl1 \
  --seed 42

Example all named workload families in one directory:
.venv/bin/python scripts/generate_job_traces.py \
  --out-dir traces/clean \
  --num-traces 20 \
  --workload-family all \
  --seed 42

Example default-family traces:
.venv/bin/python scripts/generate_job_traces.py \
  --out-dir traces/default \
  --num-traces 100 \
  --workload-family default \
  --seed 42

Example legacy-compatible manual traces:
.venv/bin/python scripts/generate_job_traces.py \
  --out-dir traces/workload_benign_stable \
  --num-traces 200 \
  --num-jobs 1000 \
  --job-interval 55 \
  --job-size-pareto-shape 1.5 \
  --job-size-pareto-scale 100 \
  --seed 42 \
  --time-mode absolute
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
    workload_family: str
    num_jobs: int
    service_rate: float | None
    job_interval_ms: float
    queue_shuffled_probability: float | None
    pareto_shape: float
    pareto_scale_bytes: float
    job_size_min_bytes: float | None
    job_size_max_bytes: float | None
    job_size_distribution: str


WORKLOAD_FAMILIES: dict[str, dict[str, Any]] = {
    "rl1": {
        "service_rate": (0.1, 2.0),
        "job_size_range_bytes": (100.0, 200.0),
        "job_interval_ms": (0.01, 0.05),
        "num_jobs": (10, 100),
        "queue_shuffled_probability": (0.1, 0.2),
    },
    "rl2": {
        "service_rate": (0.1, 5.0),
        "job_size_range_bytes": (100.0, 1000.0),
        "job_interval_ms": (0.01, 0.1),
        "num_jobs": (10, 1000),
        "queue_shuffled_probability": (0.1, 0.5),
    },
    "rl3": {
        "service_rate": (0.1, 10.0),
        "job_size_range_bytes": (1.0, 10000.0),
        "job_interval_ms": (0.1, 1.0),
        "num_jobs": (10, 5000),
        "queue_shuffled_probability": (0.1, 1.0),
    },
    "default": {
        "service_rate_choices": (0.5, 1.0, 2.0),
        "job_size_scale_bytes": 2000.0,
        "job_interval_ms": 0.1,
        "num_jobs": 2000,
        "queue_shuffled_probability": 0.5,
    },
    "original": {
        "service_rate": (2.0, 4.0),
        "job_size_range_bytes": (100.0, 1000.0),
        "job_interval_ms": 0.2,
        "num_jobs": 1000,
        "queue_shuffled_probability": None,
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate workload traces for the load-balance env.")
    p.add_argument("--out-dir", type=str, required=True, help="Output directory")
    p.add_argument("--num-traces", type=int, default=100, help="How many traces to generate")
    p.add_argument(
        "--workload-family",
        type=str,
        default="manual",
        choices=["manual", "all", *WORKLOAD_FAMILIES.keys()],
        help="Named workload family to sample, all named families, or manual for fixed legacy parameters",
    )
    p.add_argument("--num-servers", type=int, default=10, help="Server count for service_rates metadata")
    p.add_argument("--num-jobs", type=int, default=1000, help="Manual-mode jobs per trace")
    p.add_argument("--job-interval", type=float, default=55.0, help="Manual-mode exponential inter-arrival mean")
    p.add_argument("--job-interval-ms", type=float, default=None, help="Manual-mode inter-arrival mean in ms")
    p.add_argument("--job-size-pareto-shape", type=float, default=1.5, help="Pareto shape")
    p.add_argument("--job-size-pareto-scale", type=float, default=100.0, help="Manual-mode Pareto scale")
    p.add_argument("--pareto-shape", type=float, default=None, help="Pareto shape alias")
    p.add_argument("--pareto-scale-bytes", type=float, default=None, help="Manual-mode Pareto scale alias")
    p.add_argument("--job-size-min-bytes", type=float, default=None, help="Manual-mode lower support hint")
    p.add_argument("--job-size-max-bytes", type=float, default=None, help="Manual-mode upper support hint")
    p.add_argument("--service-rate", type=float, default=None, help="Manual-mode identical service rate metadata")
    p.add_argument(
        "--queue-shuffled-probability",
        type=float,
        default=None,
        help="Manual-mode queue shuffle probability metadata",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument(
        "--time-mode",
        type=str,
        default="absolute",
        choices=["absolute", "delta"],
        help="Legacy CSV time alias: arrival_time or dt",
    )
    return p.parse_args()


def _uniform_float(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    lo, hi = bounds
    return float(rng.uniform(float(lo), float(hi)))


def _uniform_int_inclusive(rng: np.random.Generator, bounds: tuple[int, int]) -> int:
    lo, hi = bounds
    return int(rng.integers(int(lo), int(hi) + 1))


def _pareto_shape(args: argparse.Namespace) -> float:
    if args.pareto_shape is not None:
        return float(args.pareto_shape)
    return float(args.job_size_pareto_shape)


def _manual_config(args: argparse.Namespace) -> TraceConfig:
    pareto_scale = args.pareto_scale_bytes
    if pareto_scale is None:
        pareto_scale = float(args.job_size_pareto_scale)
    max_bytes = args.job_size_max_bytes
    distribution = "truncated_pareto" if max_bytes is not None else "pareto"
    return TraceConfig(
        workload_family="manual",
        num_jobs=int(args.num_jobs),
        service_rate=None if args.service_rate is None else float(args.service_rate),
        job_interval_ms=float(args.job_interval if args.job_interval_ms is None else args.job_interval_ms),
        queue_shuffled_probability=(
            None if args.queue_shuffled_probability is None else float(args.queue_shuffled_probability)
        ),
        pareto_shape=_pareto_shape(args),
        pareto_scale_bytes=float(pareto_scale),
        job_size_min_bytes=None if args.job_size_min_bytes is None else float(args.job_size_min_bytes),
        job_size_max_bytes=None if max_bytes is None else float(max_bytes),
        job_size_distribution=distribution,
    )


def _family_config(args: argparse.Namespace, rng: np.random.Generator, workload_family: str | None = None) -> TraceConfig:
    family = str(args.workload_family if workload_family is None else workload_family)
    spec = WORKLOAD_FAMILIES[family]
    service_rate: float | None
    if "service_rate_choices" in spec:
        service_rate = float(rng.choice(spec["service_rate_choices"]))
    else:
        service_rate = _uniform_float(rng, spec["service_rate"])

    num_jobs_spec = spec["num_jobs"]
    if isinstance(num_jobs_spec, tuple):
        num_jobs = _uniform_int_inclusive(rng, num_jobs_spec)
    else:
        num_jobs = int(num_jobs_spec)

    interval_spec = spec["job_interval_ms"]
    if isinstance(interval_spec, tuple):
        job_interval_ms = _uniform_float(rng, interval_spec)
    else:
        job_interval_ms = float(interval_spec)

    queue_spec = spec["queue_shuffled_probability"]
    if isinstance(queue_spec, tuple):
        queue_shuffled_probability = _uniform_float(rng, queue_spec)
    elif queue_spec is None:
        queue_shuffled_probability = None
    else:
        queue_shuffled_probability = float(queue_spec)

    if "job_size_range_bytes" in spec:
        min_bytes, max_bytes = spec["job_size_range_bytes"]
        pareto_scale_bytes = float(min_bytes)
        distribution = "truncated_pareto"
    else:
        min_bytes = float(spec["job_size_scale_bytes"])
        max_bytes = None
        pareto_scale_bytes = float(spec["job_size_scale_bytes"])
        distribution = "pareto"

    return TraceConfig(
        workload_family=family,
        num_jobs=int(num_jobs),
        service_rate=None if service_rate is None else float(service_rate),
        job_interval_ms=float(job_interval_ms),
        queue_shuffled_probability=queue_shuffled_probability,
        pareto_shape=_pareto_shape(args),
        pareto_scale_bytes=float(pareto_scale_bytes),
        job_size_min_bytes=None if min_bytes is None else float(min_bytes),
        job_size_max_bytes=None if max_bytes is None else float(max_bytes),
        job_size_distribution=distribution,
    )


def sample_trace_config(
    args: argparse.Namespace,
    rng: np.random.Generator,
    workload_family: str | None = None,
) -> TraceConfig:
    family = str(args.workload_family if workload_family is None else workload_family)
    if family == "manual":
        cfg = _manual_config(args)
    else:
        cfg = _family_config(args, rng, family)
    if cfg.num_jobs <= 0:
        raise ValueError("num_jobs must be positive")
    if cfg.job_interval_ms <= 0:
        raise ValueError("job_interval_ms must be positive")
    if cfg.pareto_shape <= 0:
        raise ValueError("pareto_shape must be positive")
    if cfg.pareto_scale_bytes <= 0:
        raise ValueError("pareto_scale_bytes must be positive")
    if cfg.job_size_max_bytes is not None and cfg.job_size_max_bytes < cfg.pareto_scale_bytes:
        raise ValueError("job_size_max_bytes must be at least pareto_scale_bytes")
    return cfg


def _run_workload_families(args: argparse.Namespace) -> list[str]:
    if str(args.workload_family) == "all":
        return list(WORKLOAD_FAMILIES.keys())
    return [str(args.workload_family)]


def _sample_pareto_bytes(rng: np.random.Generator, cfg: TraceConfig) -> int:
    alpha = float(cfg.pareto_shape)
    scale = float(cfg.pareto_scale_bytes)
    max_bytes = cfg.job_size_max_bytes
    if max_bytes is not None:
        if float(max_bytes) <= scale:
            size = scale
        else:
            cdf_max = 1.0 - (scale / float(max_bytes)) ** alpha
            u = float(rng.uniform(0.0, cdf_max))
            size = scale / ((1.0 - u) ** (1.0 / alpha))
    else:
        size = float((rng.pareto(alpha) + 1.0) * scale)
    if cfg.job_size_min_bytes is not None:
        size = max(size, float(cfg.job_size_min_bytes))
    if cfg.job_size_max_bytes is not None:
        size = min(size, float(cfg.job_size_max_bytes))
    return int(max(1, round(size)))


def sample_jobs(rng: np.random.Generator, cfg: TraceConfig) -> list[dict[str, float | int]]:
    jobs: list[dict[str, float | int]] = []
    arrival_time_ms = 0.0
    for i in range(int(cfg.num_jobs)):
        if i == 0:
            dt_ms = 0.0
        else:
            dt_ms = float(rng.exponential(float(cfg.job_interval_ms)))
            arrival_time_ms += max(0.0, dt_ms)
        job_size_bytes = _sample_pareto_bytes(rng, cfg)
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
            "arrival_process": "poisson",
            "inter_arrival_distribution": "exponential",
            "service_model": "identical_service_rates",
            "service_rate_unit": "bytes_per_ms",
            "job_interval": float(cfg.job_interval_ms),
            "job_size_pareto_shape": float(cfg.pareto_shape),
            "job_size_pareto_scale": float(cfg.pareto_scale_bytes),
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
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    index_rows: list[dict[str, Any]] = []

    workload_families = _run_workload_families(args)
    for family in workload_families:
        for i in range(int(args.num_traces)):
            if len(workload_families) > 1:
                trace_id = f"{family}_trace_{i:05d}"
            else:
                trace_id = f"trace_{i:05d}"
            cfg = sample_trace_config(args, rng, family)
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

    print(f"Generated {len(index_rows)} workload traces in: {out_dir}")
    print(f"Index: {index_path}")


if __name__ == "__main__":
    main()
