from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent


def repo_root() -> Path:
    return ROOT_DIR


def resolve_path(path: str) -> Path:
    expanded = Path(os.path.expandvars(os.path.expanduser(str(path))))
    if expanded.is_absolute():
        return expanded
    return (repo_root() / expanded).resolve()


def default_service_rates(num_servers: int) -> list[float]:
    if int(num_servers) == 10:
        return [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05]
    return np.linspace(0.15, 1.05, num=int(num_servers), dtype=np.float64).tolist()


def parse_service_rates(
    text: str | None,
    *,
    num_servers: int,
    checkpoint_config: dict[str, Any] | None = None,
) -> list[float]:
    if text is not None:
        vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
        if len(vals) != int(num_servers):
            raise ValueError("--service-rates must have exactly num_servers comma-separated values")
        return vals
    if checkpoint_config is not None:
        cfg_rates = checkpoint_config.get("service_rates")
        if isinstance(cfg_rates, list) and len(cfg_rates) == int(num_servers):
            return [float(x) for x in cfg_rates]
    return default_service_rates(int(num_servers))


def collect_trace_files(trace_dir: Path) -> list[Path]:
    csv_files = [p for p in sorted(trace_dir.glob("*.csv")) if p.name.lower() != "index.csv"]
    csv_stems = {p.stem for p in csv_files}
    json_files = [
        p
        for p in sorted(trace_dir.glob("*.json"))
        if p.name.lower() != "meta.json" and p.stem not in csv_stems
    ]
    return sorted(list(csv_files) + list(json_files))


def choose_trace_files(
    *,
    trace_files: list[Path],
    episodes: int,
    select_mode: str,
    rng: np.random.Generator,
) -> list[Path]:
    if len(trace_files) == 0:
        raise ValueError("trace_files must be non-empty")
    if int(episodes) <= 0:
        raise ValueError("episodes must be positive")
    mode = str(select_mode)
    if mode not in {"random", "round_robin"}:
        raise ValueError("select_mode must be one of: random, round_robin")
    if mode == "round_robin":
        return [trace_files[int(i % len(trace_files))] for i in range(int(episodes))]
    indices = rng.integers(0, len(trace_files), size=int(episodes))
    return [trace_files[int(i)] for i in np.asarray(indices).reshape(-1)]


def range_to_log01(value: float, vmin: float, vmax: float) -> float:
    lo = float(max(vmin, 1e-9))
    hi = float(max(vmax, lo + 1e-9))
    x = float(np.clip(float(value), lo, hi))
    num = float(np.log(x) - np.log(lo))
    den = float(np.log(hi) - np.log(lo))
    if den <= 0.0:
        return 0.0
    return float(np.clip(num / den, 0.0, 1.0))


def log01_to_range(x01: float, vmin: float, vmax: float) -> float:
    x = float(np.clip(float(x01), 0.0, 1.0))
    lo = float(np.log(max(vmin, 1e-9)))
    hi = float(np.log(max(vmax, max(vmin, 1e-9) + 1e-9)))
    return float(np.exp(lo + x * (hi - lo)))


def infer_control_bounds_from_jobs(
    jobs: list[dict[str, int]],
    *,
    dt_floor: float = 1.0,
) -> dict[str, float]:
    if len(jobs) < 2:
        return {
            "dt_min": float(dt_floor),
            "dt_max": float(max(dt_floor, 2.0)),
            "size_min": float(max(1, int(jobs[0]["job_size"]))),
            "size_max": float(max(2, int(jobs[0]["job_size"]))),
        }
    dt_vals = []
    size_vals = []
    for idx in range(1, len(jobs)):
        dt_vals.append(max(0.0, float(int(jobs[idx]["arrival_time"]) - int(jobs[idx - 1]["arrival_time"]))))
        size_vals.append(float(int(jobs[idx]["job_size"])))
    positive_dt = [x for x in dt_vals if float(x) > 0.0]
    dt_min = float(max(float(dt_floor), min(positive_dt) if len(positive_dt) > 0 else float(dt_floor)))
    dt_max = float(max(dt_min + 1.0, max(dt_vals) if len(dt_vals) > 0 else dt_min + 1.0))
    size_min = float(max(1.0, min(size_vals)))
    size_max = float(max(size_min + 1.0, max(size_vals)))
    return {
        "dt_min": float(dt_min),
        "dt_max": float(dt_max),
        "size_min": float(size_min),
        "size_max": float(size_max),
    }


def jobs_to_future_arrivals(jobs: list[dict[str, int]]) -> list[tuple[int | None, int | None]]:
    future_arrivals: list[tuple[int | None, int | None]] = []
    n = len(jobs)
    for i in range(int(n)):
        if int(i) < int(n) - 1:
            next_dt = int(jobs[int(i + 1)]["arrival_time"]) - int(jobs[int(i)]["arrival_time"])
            next_size = int(jobs[int(i + 1)]["job_size"])
            future_arrivals.append((max(0, int(next_dt)), int(next_size)))
        else:
            future_arrivals.append((None, None))
    return future_arrivals


def jobs_to_control_sequence(jobs: list[dict[str, int]]) -> tuple[np.ndarray, np.ndarray]:
    if len(jobs) < 2:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    dt_vals = []
    size_vals = []
    for idx in range(1, len(jobs)):
        dt_vals.append(max(0, int(jobs[idx]["arrival_time"]) - int(jobs[idx - 1]["arrival_time"])))
        size_vals.append(max(1, int(jobs[idx]["job_size"])))
    return np.asarray(dt_vals, dtype=np.float32), np.asarray(size_vals, dtype=np.float32)


def controls_to_jobs(
    *,
    first_job_size: int,
    dt_values: np.ndarray,
    size_values: np.ndarray,
    first_arrival_time: int = 0,
) -> list[dict[str, int]]:
    jobs = [{"arrival_time": int(first_arrival_time), "job_size": int(first_job_size)}]
    arrival_time = int(first_arrival_time)
    for dt_value, size_value in zip(np.asarray(dt_values).reshape(-1), np.asarray(size_values).reshape(-1)):
        arrival_time += int(max(0, round(float(dt_value))))
        jobs.append(
            {
                "arrival_time": int(arrival_time),
                "job_size": int(max(1, round(float(size_value)))),
            }
        )
    return jobs


def write_trace_csv(path: Path, jobs: list[dict[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["job_id", "arrival_time", "job_size"])
        writer.writeheader()
        for idx, job in enumerate(list(jobs)):
            writer.writerow(
                {
                    "job_id": int(idx),
                    "arrival_time": int(job["arrival_time"]),
                    "job_size": int(job["job_size"]),
                }
            )


def write_trace_json(path: Path, jobs: list[dict[str, int]], *, kind: str, extra: dict[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "kind": str(kind),
        "num_jobs": int(len(jobs)),
        "jobs": [
            {
                "arrival_time": int(job["arrival_time"]),
                "job_size": int(job["job_size"]),
            }
            for job in list(jobs)
        ],
    }
    if isinstance(extra, dict):
        payload.update(dict(extra))
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
