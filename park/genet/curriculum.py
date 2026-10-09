from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .trace_family import GeneratedJobTrace, JobTraceParams, JobTraceSearchSpace, generate_job_trace


@dataclass(frozen=True)
class JobTraceRangeComponent:
    search_space: JobTraceSearchSpace
    weight: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "search_space": self.search_space.to_dict(),
            "weight": float(self.weight),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "JobTraceRangeComponent":
        return cls(
            search_space=JobTraceSearchSpace.from_dict(dict(payload["search_space"])),
            weight=float(payload["weight"]),
        )


@dataclass(frozen=True)
class SampledTrace:
    trace: GeneratedJobTrace
    params: JobTraceParams
    component_index: int


class JobTraceCurriculum:
    def __init__(self, *, components: list[JobTraceRangeComponent]) -> None:
        if len(components) == 0:
            raise ValueError("curriculum must contain at least one component")
        self.components = [component for component in list(components)]
        total_weight = float(sum(float(component.weight) for component in self.components))
        if float(total_weight) <= 0.0:
            raise ValueError("curriculum weights must sum to a positive value")

    @classmethod
    def from_search_space(cls, search_space: JobTraceSearchSpace) -> "JobTraceCurriculum":
        return cls(
            components=[
                JobTraceRangeComponent(
                    search_space=search_space,
                    weight=1.0,
                )
            ]
        )

    def normalized_weights(self) -> np.ndarray:
        weights = np.asarray([float(component.weight) for component in self.components], dtype=np.float64)
        total = float(np.sum(weights))
        if float(total) <= 0.0:
            raise ValueError("curriculum weights must sum to a positive value")
        return weights / float(total)

    def sample_trace(self, *, rng: np.random.Generator, trace_seed: int) -> SampledTrace:
        weights = self.normalized_weights()
        component_index = int(rng.choice(len(self.components), p=weights))
        component = self.components[int(component_index)]
        params = component.search_space.sample_params(rng=rng)
        trace = generate_job_trace(
            params=params,
            rng=np.random.default_rng(int(trace_seed)),
            trace_seed=int(trace_seed),
        )
        return SampledTrace(
            trace=trace,
            params=params,
            component_index=int(component_index),
        )

    def add_point_component(
        self,
        *,
        params: JobTraceParams,
        add_probability: float,
        relative_radius: float = 0.1,
        fraction_radius: float = 0.05,
    ) -> "JobTraceCurriculum":
        prob = float(np.clip(float(add_probability), 0.0, 1.0))
        if float(prob) == 0.0:
            return JobTraceCurriculum(
                components=[
                    JobTraceRangeComponent(
                        search_space=component.search_space,
                        weight=float(component.weight),
                    )
                    for component in self.components
                ]
            )
        weights = self.normalized_weights()
        next_components: list[JobTraceRangeComponent] = []
        for component, weight in zip(self.components, list(weights)):
            next_components.append(
                JobTraceRangeComponent(
                    search_space=component.search_space,
                    weight=float(weight) * (1.0 - float(prob)),
                )
            )

        rr = float(max(relative_radius, 1e-6))
        fr = float(max(fraction_radius, 1e-6))
        point_space = JobTraceSearchSpace(
            num_jobs=int(params.num_jobs),
            base_dt_ms_range=(
                max(1e-6, float(params.base_dt_ms) * (1.0 - rr)),
                float(params.base_dt_ms) * (1.0 + rr),
            ),
            base_size_scale_bytes_range=(
                max(1e-6, float(params.base_size_scale_bytes) * (1.0 - rr)),
                float(params.base_size_scale_bytes) * (1.0 + rr),
            ),
            pareto_shape_range=(
                max(1.01, float(params.pareto_shape) * (1.0 - rr)),
                float(params.pareto_shape) * (1.0 + rr),
            ),
            burst_start_frac_range=(
                max(0.0, float(params.burst_start_frac) - fr),
                min(1.0, float(params.burst_start_frac) + fr),
            ),
            burst_duration_frac_range=(
                max(1e-4, float(params.burst_duration_frac) - fr),
                min(1.0, float(params.burst_duration_frac) + fr),
            ),
            burst_dt_multiplier_range=(
                max(1e-6, float(params.burst_dt_multiplier) * (1.0 - rr)),
                float(params.burst_dt_multiplier) * (1.0 + rr),
            ),
            burst_size_multiplier_range=(
                max(1e-6, float(params.burst_size_multiplier) * (1.0 - rr)),
                float(params.burst_size_multiplier) * (1.0 + rr),
            ),
        )
        next_components.append(
            JobTraceRangeComponent(
                search_space=point_space,
                weight=float(prob),
            )
        )
        return JobTraceCurriculum(components=next_components)

    def to_dict(self) -> dict[str, Any]:
        return {
            "components": [component.to_dict() for component in self.components],
        }

    def dump(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "JobTraceCurriculum":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            components=[JobTraceRangeComponent.from_dict(component) for component in list(payload["components"])],
        )
