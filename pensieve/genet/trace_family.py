from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


DEFAULT_DURATION_S = 330.0
DEFAULT_TIME_NOISE_LOW_S = 0.1
DEFAULT_TIME_NOISE_HIGH_S = 3.5


@dataclass(frozen=True)
class BandwidthTraceParams:
    min_bw_mbps: float
    max_bw_mbps: float
    bw_change_interval: float

    def normalized(self, *, bw_floor_mbps: float, bw_ceil_mbps: float) -> "BandwidthTraceParams":
        min_bw = float(np.clip(float(self.min_bw_mbps), float(bw_floor_mbps), float(bw_ceil_mbps)))
        max_bw = float(np.clip(float(self.max_bw_mbps), float(bw_floor_mbps), float(bw_ceil_mbps)))
        max_bw = max(float(max_bw), float(min_bw))
        interval = max(float(self.bw_change_interval), 1.0)
        return BandwidthTraceParams(
            min_bw_mbps=float(min_bw),
            max_bw_mbps=float(max_bw),
            bw_change_interval=float(interval),
        )


@dataclass(frozen=True)
class CookedTrace:
    timestamps_s: list[float]
    bandwidths_mbps: list[float]
    params: BandwidthTraceParams
    trace_seed: int

    def to_metadata(self) -> dict[str, Any]:
        return {
            "timestamps_s": [float(x) for x in self.timestamps_s],
            "bandwidths_mbps": [float(x) for x in self.bandwidths_mbps],
            "params": asdict(self.params),
            "trace_seed": int(self.trace_seed),
        }


def _sample_hold_steps(*, bw_change_interval: float, rng: np.random.RandomState) -> int:
    max_hold_steps = max(int(math.ceil(float(bw_change_interval))), 1)
    return int(rng.randint(1, max_hold_steps + 1))


def generate_cooked_trace(
    *,
    params: BandwidthTraceParams,
    rng: np.random.RandomState,
    trace_seed: int,
    duration_s: float = DEFAULT_DURATION_S,
    time_noise_low_s: float = DEFAULT_TIME_NOISE_LOW_S,
    time_noise_high_s: float = DEFAULT_TIME_NOISE_HIGH_S,
) -> CookedTrace:
    if float(duration_s) <= 0.0:
        raise ValueError("* duration_s must be positive")
    if float(time_noise_low_s) <= 0.0 or float(time_noise_high_s) < float(time_noise_low_s):
        raise ValueError("* invalid time-noise bounds")

    timestamps_s: list[float] = [0.0]
    bandwidths_mbps: list[float] = []

    current_bw = float(rng.uniform(float(params.min_bw_mbps), float(params.max_bw_mbps)))
    bandwidths_mbps.append(float(current_bw))

    t_s = 0.0
    hold_steps_remaining = 0
    while float(t_s) < float(duration_s):
        if int(hold_steps_remaining) <= 0:
            current_bw = float(rng.uniform(float(params.min_bw_mbps), float(params.max_bw_mbps)))
            hold_steps_remaining = _sample_hold_steps(
                bw_change_interval=float(params.bw_change_interval),
                rng=rng,
            )

        hold_steps_remaining -= 1
        dt_s = float(rng.uniform(float(time_noise_low_s), float(time_noise_high_s)))
        t_s = float(min(float(duration_s), float(t_s + dt_s)))
        timestamps_s.append(float(t_s))
        bandwidths_mbps.append(float(current_bw))

    if len(timestamps_s) < 2:
        timestamps_s.append(float(duration_s))
        bandwidths_mbps.append(float(current_bw))

    return CookedTrace(
        timestamps_s=timestamps_s,
        bandwidths_mbps=bandwidths_mbps,
        params=params,
        trace_seed=int(trace_seed),
    )


def write_cooked_trace(path: str, trace: CookedTrace) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for t_s, bw_mbps in zip(trace.timestamps_s, trace.bandwidths_mbps):
            f.write(f"{float(t_s):.6f}\t{float(bw_mbps):.6f}\n")


def write_trace_metadata(path: str, trace: CookedTrace, *, extra: dict[str, Any] | None = None) -> None:
    payload = trace.to_metadata()
    if extra:
        payload.update(extra)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

