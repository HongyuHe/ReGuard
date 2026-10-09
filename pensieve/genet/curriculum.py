from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from .trace_family import BandwidthTraceParams, CookedTrace, generate_cooked_trace


def _sample_uniform(rng: np.random.RandomState, low: float, high: float) -> float:
    if float(high) <= float(low):
        return float(low)
    return float(rng.uniform(float(low), float(high)))


def _sample_log_uniform(rng: np.random.RandomState, low: float, high: float) -> float:
    low_safe = max(float(low), 1e-6)
    high_safe = max(float(high), float(low_safe))
    if float(high_safe) <= float(low_safe):
        return float(low_safe)
    return float(np.exp(rng.uniform(math.log(float(low_safe)), math.log(float(high_safe)))))


@dataclass(frozen=True)
class BandwidthRangeComponent:
    min_bw_range_mbps: tuple[float, float]
    max_bw_range_mbps: tuple[float, float]
    bw_change_interval_range: tuple[float, float]
    duration_s: float
    weight: float

    def sample_params(self, *, rng: np.random.RandomState, bw_floor_mbps: float, bw_ceil_mbps: float) -> BandwidthTraceParams:
        min_bw_mbps = _sample_uniform(
            rng,
            float(self.min_bw_range_mbps[0]),
            float(self.min_bw_range_mbps[1]),
        )
        max_bw_mbps = _sample_log_uniform(
            rng,
            float(self.max_bw_range_mbps[0]),
            float(self.max_bw_range_mbps[1]),
        )
        bw_change_interval = _sample_uniform(
            rng,
            float(self.bw_change_interval_range[0]),
            float(self.bw_change_interval_range[1]),
        )
        return BandwidthTraceParams(
            min_bw_mbps=float(min_bw_mbps),
            max_bw_mbps=float(max_bw_mbps),
            bw_change_interval=float(bw_change_interval),
        ).normalized(
            bw_floor_mbps=float(bw_floor_mbps),
            bw_ceil_mbps=float(bw_ceil_mbps),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_bw_range_mbps": [float(self.min_bw_range_mbps[0]), float(self.min_bw_range_mbps[1])],
            "max_bw_range_mbps": [float(self.max_bw_range_mbps[0]), float(self.max_bw_range_mbps[1])],
            "bw_change_interval_range": [
                float(self.bw_change_interval_range[0]),
                float(self.bw_change_interval_range[1]),
            ],
            "duration_s": float(self.duration_s),
            "weight": float(self.weight),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BandwidthRangeComponent":
        return cls(
            min_bw_range_mbps=(
                float(payload["min_bw_range_mbps"][0]),
                float(payload["min_bw_range_mbps"][1]),
            ),
            max_bw_range_mbps=(
                float(payload["max_bw_range_mbps"][0]),
                float(payload["max_bw_range_mbps"][1]),
            ),
            bw_change_interval_range=(
                float(payload["bw_change_interval_range"][0]),
                float(payload["bw_change_interval_range"][1]),
            ),
            duration_s=float(payload["duration_s"]),
            weight=float(payload["weight"]),
        )


@dataclass(frozen=True)
class SampledTrace:
    trace: CookedTrace
    params: BandwidthTraceParams
    component_index: int


class BandwidthCurriculum:
    def __init__(
        self,
        *,
        bw_floor_mbps: float,
        bw_ceil_mbps: float,
        components: list[BandwidthRangeComponent],
    ) -> None:
        self.bw_floor_mbps = float(bw_floor_mbps)
        self.bw_ceil_mbps = float(bw_ceil_mbps)
        self.components = [component for component in list(components)]
        self._validate()

    def _validate(self) -> None:
        if self.bw_floor_mbps <= 0.0 or self.bw_ceil_mbps < self.bw_floor_mbps:
            raise ValueError("* invalid curriculum bandwidth bounds")
        if len(self.components) == 0:
            raise ValueError("* curriculum must contain at least one component")

        total_weight = 0.0
        for component in self.components:
            if float(component.weight) < 0.0:
                raise ValueError("* curriculum weights must be non-negative")
            if float(component.duration_s) <= 0.0:
                raise ValueError("* curriculum duration must be positive")
            total_weight += float(component.weight)
        if float(total_weight) <= 0.0:
            raise ValueError("* curriculum weights must sum to a positive value")

    @classmethod
    def from_search_space(
        cls,
        *,
        bw_floor_mbps: float,
        bw_ceil_mbps: float,
        bw_change_interval: float,
        duration_s: float,
    ) -> "BandwidthCurriculum":
        component = BandwidthRangeComponent(
            min_bw_range_mbps=(float(bw_floor_mbps), float(bw_ceil_mbps)),
            max_bw_range_mbps=(float(bw_floor_mbps), float(bw_ceil_mbps)),
            bw_change_interval_range=(float(bw_change_interval), float(bw_change_interval)),
            duration_s=float(duration_s),
            weight=1.0,
        )
        return cls(
            bw_floor_mbps=float(bw_floor_mbps),
            bw_ceil_mbps=float(bw_ceil_mbps),
            components=[component],
        )

    def normalized_weights(self) -> np.ndarray:
        weights = np.asarray([float(component.weight) for component in self.components], dtype=np.float64)
        weight_sum = float(np.sum(weights))
        if float(weight_sum) <= 0.0:
            raise ValueError("* curriculum weights must sum to a positive value")
        return weights / float(weight_sum)

    def sample_trace(self, *, rng: np.random.RandomState, trace_seed: int) -> SampledTrace:
        weights = self.normalized_weights()
        component_index = int(rng.choice(len(self.components), p=weights))
        component = self.components[int(component_index)]
        params = component.sample_params(
            rng=rng,
            bw_floor_mbps=float(self.bw_floor_mbps),
            bw_ceil_mbps=float(self.bw_ceil_mbps),
        )
        trace_rng = np.random.RandomState(int(trace_seed))
        trace = generate_cooked_trace(
            params=params,
            rng=trace_rng,
            trace_seed=int(trace_seed),
            duration_s=float(component.duration_s),
        )
        return SampledTrace(
            trace=trace,
            params=params,
            component_index=int(component_index),
        )

    def add_point_component(
        self,
        *,
        params: BandwidthTraceParams,
        add_probability: float,
        duration_s: float,
    ) -> "BandwidthCurriculum":
        prob = float(add_probability)
        if float(prob) < 0.0 or float(prob) > 1.0:
            raise ValueError("* add_probability must be in [0, 1]")
        if float(prob) == 0.0:
            return BandwidthCurriculum(
                bw_floor_mbps=float(self.bw_floor_mbps),
                bw_ceil_mbps=float(self.bw_ceil_mbps),
                components=[
                    BandwidthRangeComponent(
                        min_bw_range_mbps=tuple(component.min_bw_range_mbps),
                        max_bw_range_mbps=tuple(component.max_bw_range_mbps),
                        bw_change_interval_range=tuple(component.bw_change_interval_range),
                        duration_s=float(component.duration_s),
                        weight=float(component.weight),
                    )
                    for component in self.components
                ],
            )

        current_weights = self.normalized_weights()
        next_components: list[BandwidthRangeComponent] = []
        for component, weight in zip(self.components, list(current_weights)):
            next_components.append(
                BandwidthRangeComponent(
                    min_bw_range_mbps=tuple(component.min_bw_range_mbps),
                    max_bw_range_mbps=tuple(component.max_bw_range_mbps),
                    bw_change_interval_range=tuple(component.bw_change_interval_range),
                    duration_s=float(component.duration_s),
                    weight=float(weight) * (1.0 - float(prob)),
                )
            )

        next_components.append(
            BandwidthRangeComponent(
                min_bw_range_mbps=(float(params.min_bw_mbps), float(params.min_bw_mbps)),
                max_bw_range_mbps=(float(params.max_bw_mbps), float(params.max_bw_mbps)),
                bw_change_interval_range=(float(params.bw_change_interval), float(params.bw_change_interval)),
                duration_s=float(duration_s),
                weight=float(prob) if len(next_components) > 0 else 1.0,
            )
        )
        return BandwidthCurriculum(
            bw_floor_mbps=float(self.bw_floor_mbps),
            bw_ceil_mbps=float(self.bw_ceil_mbps),
            components=next_components,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "bw_floor_mbps": float(self.bw_floor_mbps),
            "bw_ceil_mbps": float(self.bw_ceil_mbps),
            "components": [component.to_dict() for component in self.components],
        }

    def dump(self, path: str) -> None:
        os.makedirs(os.path.dirname(str(path)), exist_ok=True)
        with open(str(path), "w") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str) -> "BandwidthCurriculum":
        with open(str(path), "r") as f:
            payload = json.load(f)
        return cls(
            bw_floor_mbps=float(payload["bw_floor_mbps"]),
            bw_ceil_mbps=float(payload["bw_ceil_mbps"]),
            components=[
                BandwidthRangeComponent.from_dict(component_payload)
                for component_payload in list(payload["components"])
            ],
        )
