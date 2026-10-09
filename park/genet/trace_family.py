from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class JobTraceParams:
    num_jobs: int
    base_dt_ms: float
    base_size_scale_bytes: float
    pareto_shape: float
    burst_start_frac: float
    burst_duration_frac: float
    burst_dt_multiplier: float
    burst_size_multiplier: float


@dataclass(frozen=True)
class GeneratedJobTrace:
    jobs: list[dict[str, float | int]]
    params: JobTraceParams
    trace_seed: int

    def plain_jobs(self) -> list[dict[str, int]]:
        return [
            {
                "arrival_time": int(job["arrival_time"]),
                "job_size": int(job["job_size"]),
            }
            for job in list(self.jobs)
        ]


@dataclass(frozen=True)
class JobTraceSearchSpace:
    num_jobs: int
    base_dt_ms_range: tuple[float, float]
    base_size_scale_bytes_range: tuple[float, float]
    pareto_shape_range: tuple[float, float]
    burst_start_frac_range: tuple[float, float]
    burst_duration_frac_range: tuple[float, float]
    burst_dt_multiplier_range: tuple[float, float]
    burst_size_multiplier_range: tuple[float, float]

    def _specs(self) -> list[tuple[str, tuple[float, float], str]]:
        return [
            ("base_dt_ms", tuple(self.base_dt_ms_range), "log"),
            ("base_size_scale_bytes", tuple(self.base_size_scale_bytes_range), "log"),
            ("pareto_shape", tuple(self.pareto_shape_range), "linear"),
            ("burst_start_frac", tuple(self.burst_start_frac_range), "linear"),
            ("burst_duration_frac", tuple(self.burst_duration_frac_range), "linear"),
            ("burst_dt_multiplier", tuple(self.burst_dt_multiplier_range), "log"),
            ("burst_size_multiplier", tuple(self.burst_size_multiplier_range), "log"),
        ]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "JobTraceSearchSpace":
        return cls(
            num_jobs=int(payload["num_jobs"]),
            base_dt_ms_range=(
                float(payload["base_dt_ms_range"][0]),
                float(payload["base_dt_ms_range"][1]),
            ),
            base_size_scale_bytes_range=(
                float(payload["base_size_scale_bytes_range"][0]),
                float(payload["base_size_scale_bytes_range"][1]),
            ),
            pareto_shape_range=(
                float(payload["pareto_shape_range"][0]),
                float(payload["pareto_shape_range"][1]),
            ),
            burst_start_frac_range=(
                float(payload["burst_start_frac_range"][0]),
                float(payload["burst_start_frac_range"][1]),
            ),
            burst_duration_frac_range=(
                float(payload["burst_duration_frac_range"][0]),
                float(payload["burst_duration_frac_range"][1]),
            ),
            burst_dt_multiplier_range=(
                float(payload["burst_dt_multiplier_range"][0]),
                float(payload["burst_dt_multiplier_range"][1]),
            ),
            burst_size_multiplier_range=(
                float(payload["burst_size_multiplier_range"][0]),
                float(payload["burst_size_multiplier_range"][1]),
            ),
        )

    def sample_vector(self, *, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(0.0, 1.0, size=len(self._specs())).astype(np.float64)

    def sample_params(self, *, rng: np.random.Generator) -> JobTraceParams:
        return self.vector_to_params(self.sample_vector(rng=rng))

    def vector_to_params(self, vector: np.ndarray) -> JobTraceParams:
        values: dict[str, float] = {}
        vec = np.asarray(vector, dtype=np.float64).reshape(-1)
        if vec.shape[0] != len(self._specs()):
            raise ValueError("search vector has wrong dimension")
        for idx, (name, bounds, scale_kind) in enumerate(self._specs()):
            x = float(np.clip(vec[int(idx)], 0.0, 1.0))
            low = float(bounds[0])
            high = float(bounds[1])
            if scale_kind == "log":
                low_safe = max(low, 1e-9)
                high_safe = max(high, low_safe)
                value = float(np.exp(math.log(low_safe) + x * (math.log(high_safe) - math.log(low_safe))))
            else:
                value = float(low + x * (high - low))
            values[str(name)] = float(value)
        return self.clamp_params(
            JobTraceParams(
                num_jobs=int(self.num_jobs),
                base_dt_ms=float(values["base_dt_ms"]),
                base_size_scale_bytes=float(values["base_size_scale_bytes"]),
                pareto_shape=float(values["pareto_shape"]),
                burst_start_frac=float(values["burst_start_frac"]),
                burst_duration_frac=float(values["burst_duration_frac"]),
                burst_dt_multiplier=float(values["burst_dt_multiplier"]),
                burst_size_multiplier=float(values["burst_size_multiplier"]),
            )
        )

    def params_to_vector(self, params: JobTraceParams) -> np.ndarray:
        params = self.clamp_params(params)
        vector: list[float] = []
        for name, bounds, scale_kind in self._specs():
            value = float(getattr(params, str(name)))
            low = float(bounds[0])
            high = float(bounds[1])
            if scale_kind == "log":
                low_safe = max(low, 1e-9)
                high_safe = max(high, low_safe)
                numer = math.log(max(value, low_safe)) - math.log(low_safe)
                denom = max(math.log(high_safe) - math.log(low_safe), 1e-12)
                coord = numer / denom
            else:
                denom = max(high - low, 1e-12)
                coord = (value - low) / denom
            vector.append(float(np.clip(coord, 0.0, 1.0)))
        return np.asarray(vector, dtype=np.float64)

    def clamp_params(self, params: JobTraceParams) -> JobTraceParams:
        start_low, start_high = tuple(self.burst_start_frac_range)
        dur_low, dur_high = tuple(self.burst_duration_frac_range)
        burst_start = float(np.clip(float(params.burst_start_frac), float(start_low), float(start_high)))
        burst_duration = float(np.clip(float(params.burst_duration_frac), float(dur_low), float(dur_high)))
        burst_duration = float(min(burst_duration, max(1e-6, 1.0 - burst_start)))
        if burst_start + burst_duration > 1.0:
            burst_start = float(max(0.0, 1.0 - burst_duration))
        return JobTraceParams(
            num_jobs=max(1, int(params.num_jobs)),
            base_dt_ms=float(
                np.clip(float(params.base_dt_ms), float(self.base_dt_ms_range[0]), float(self.base_dt_ms_range[1]))
            ),
            base_size_scale_bytes=float(
                np.clip(
                    float(params.base_size_scale_bytes),
                    float(self.base_size_scale_bytes_range[0]),
                    float(self.base_size_scale_bytes_range[1]),
                )
            ),
            pareto_shape=float(
                np.clip(float(params.pareto_shape), float(self.pareto_shape_range[0]), float(self.pareto_shape_range[1]))
            ),
            burst_start_frac=float(burst_start),
            burst_duration_frac=float(burst_duration),
            burst_dt_multiplier=float(
                np.clip(
                    float(params.burst_dt_multiplier),
                    float(self.burst_dt_multiplier_range[0]),
                    float(self.burst_dt_multiplier_range[1]),
                )
            ),
            burst_size_multiplier=float(
                np.clip(
                    float(params.burst_size_multiplier),
                    float(self.burst_size_multiplier_range[0]),
                    float(self.burst_size_multiplier_range[1]),
                )
            ),
        )


def _sample_pareto_bytes(*, rng: np.random.Generator, alpha: float, scale: float) -> int:
    value = float((rng.pareto(float(alpha)) + 1.0) * float(scale))
    return int(max(1, round(value)))


def _in_burst(params: JobTraceParams, job_idx: int) -> bool:
    if int(params.num_jobs) <= 1:
        frac = 0.0
    else:
        frac = float(job_idx) / float(max(int(params.num_jobs) - 1, 1))
    burst_end = float(min(1.0, float(params.burst_start_frac) + float(params.burst_duration_frac)))
    return float(params.burst_start_frac) <= float(frac) < float(burst_end)


def generate_job_trace(
    *,
    params: JobTraceParams,
    rng: np.random.Generator,
    trace_seed: int,
) -> GeneratedJobTrace:
    jobs: list[dict[str, float | int]] = []
    arrival_time_ms = 0.0
    for idx in range(int(params.num_jobs)):
        in_burst = _in_burst(params, int(idx))
        dt_mean = float(params.base_dt_ms) * (float(params.burst_dt_multiplier) if in_burst else 1.0)
        size_scale = float(params.base_size_scale_bytes) * (float(params.burst_size_multiplier) if in_burst else 1.0)
        if int(idx) == 0:
            dt_ms = 0.0
        else:
            dt_ms = float(rng.exponential(max(dt_mean, 1e-6)))
            arrival_time_ms += max(0.0, float(dt_ms))
        size_bytes = _sample_pareto_bytes(
            rng=rng,
            alpha=float(params.pareto_shape),
            scale=max(float(size_scale), 1e-6),
        )
        jobs.append(
            {
                "job_id": int(idx),
                "arrival_time_ms": float(arrival_time_ms),
                "dt_ms": float(dt_ms),
                "job_size_bytes": int(size_bytes),
                "arrival_time": int(round(arrival_time_ms)),
                "dt": int(round(dt_ms)),
                "job_size": int(size_bytes),
                "phase": "burst" if in_burst else "background",
            }
        )
    return GeneratedJobTrace(
        jobs=jobs,
        params=params,
        trace_seed=int(trace_seed),
    )


def trace_metadata(*, trace_id: str, trace: GeneratedJobTrace) -> dict[str, Any]:
    payload = asdict(trace.params)
    payload.update(
        {
            "trace_id": str(trace_id),
            "trace_seed": int(trace.trace_seed),
            "generator": "park_lb_genet",
            "trace_family": "two_phase_burst",
            "service_rate_unit": "bytes_per_ms",
        }
    )
    return payload


def write_generated_trace_csv(path: Path, trace: GeneratedJobTrace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["job_id", "arrival_time_ms", "dt_ms", "job_size_bytes", "arrival_time", "job_size", "phase"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in list(trace.jobs):
            writer.writerow({key: row[key] for key in fieldnames})


def write_generated_trace_json(path: Path, *, trace_id: str, trace: GeneratedJobTrace) -> None:
    payload = trace_metadata(trace_id=str(trace_id), trace=trace)
    payload["jobs"] = list(trace.jobs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
