from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np

from attacks.envs import DEFAULT_BASELINE_METHODS, ParallelGapAttackEnv
from attacks.online import SageLaunchConfig, acquire_run_namespace


if __package__ in (None, ""):
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts._trace_attack_common import (
    attack_bounds_from_config,
    load_trace_entries,
    materialize_trace_splits,
    repo_root_from_script,
    resolve_repo_path,
    run_online_policy_episode,
    save_json,
    utc_now_iso,
    write_bandwidth_trace,
)


def _require_bayes_opt():
    try:
        from bayes_opt import BayesianOptimization  # type: ignore
        from bayes_opt.util import UtilityFunction  # type: ignore

        return BayesianOptimization, UtilityFunction
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency: install `examples/Pensieve/genet/requirements.txt` "
            "to run Sage Genet BO search."
        ) from exc


def tf_checkpoint_exists(path: str) -> bool:
    checkpoint_dir = Path(path)
    checkpoint_file = checkpoint_dir / "checkpoint"
    if not checkpoint_file.exists():
        return False
    return any(checkpoint_dir.glob("*.index"))


def load_config_payload(repo_root: str, config_path: str) -> tuple[str, dict[str, Any]]:
    resolved = resolve_repo_path(repo_root, config_path)
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"missing config file: {resolved}")
    with open(resolved, "r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {resolved}")
    return resolved, dict(payload)


def ensure_test_manifest(repo_root: str, manifest_path: str) -> str:
    resolved = resolve_repo_path(repo_root, manifest_path)
    if os.path.exists(resolved):
        return resolved
    materialize_trace_splits(
        repo_root=repo_root,
        train_root=resolve_repo_path(repo_root, "attacks/train"),
        test_root=resolve_repo_path(repo_root, "attacks/test"),
    )
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"missing test manifest: {resolved}")
    return resolved


def attack_mode_from_config(config_payload: dict[str, Any]) -> str:
    return str(config_payload.get("attack_mode", "trace_conditioned"))


def trace_set_name_from_out_dir(out_dir: str) -> str:
    return os.path.basename(os.path.abspath(out_dir.rstrip(os.sep))) or "generated"


def resolved_launch_config(
    *,
    config_payload: dict[str, Any],
    run_namespace,
    log_prefix: str,
) -> SageLaunchConfig:
    return replace(
        SageLaunchConfig(
            sage_script="sage_rl/sage.sh",
            latency_ms=int(config_payload.get("latency_ms", 25)),
            port=int(config_payload.get("port", 5101)),
            downlink_trace="wired48",
            uplink_trace="wired48",
            iteration_id=int(config_payload.get("iteration_id", 0)),
            qsize_packets=int(config_payload.get("qsize_packets", 128)),
            env_bw_mbps=int(config_payload.get("env_bw_mbps", 48)),
            bw2_mbps=int(config_payload.get("bw2_mbps", 48)),
            trace_period_s=int(config_payload.get("trace_period_s", 7)),
            first_time_mode=int(config_payload.get("sage_mode", 0)),
            log_prefix=str(log_prefix),
            duration_seconds=int(config_payload.get("duration_seconds", 60)),
            actor_id=int(config_payload.get("actor_id", 900)),
            duration_steps=int(config_payload.get("duration_steps", 6000)),
            num_flows=int(config_payload.get("num_flows", 1)),
            save_logs=int(config_payload.get("save_logs", 0)),
            analyze_logs=int(config_payload.get("analyze_logs", 0)),
            mm_adv_bin=config_payload.get("mm_adv_bin"),
            initial_uplink_loss=float(config_payload.get("init_uplink_loss", 0.0)),
            initial_downlink_loss=float(config_payload.get("init_downlink_loss", 0.0)),
            initial_uplink_delay_ms=config_payload.get("init_uplink_delay_ms"),
            initial_downlink_delay_ms=config_payload.get("init_downlink_delay_ms"),
            initial_uplink_queue_packets=config_payload.get("init_uplink_queue_packets"),
            initial_downlink_queue_packets=config_payload.get("init_downlink_queue_packets"),
        ),
        port=int(run_namespace.port_base),
        actor_id=int(run_namespace.actor_id_base),
    )


@dataclass(frozen=True)
class SharedScheduleSearchBounds:
    bw_min_mbps: float
    bw_max_mbps: float
    loss_min: float
    loss_max: float
    segment_steps_min: float
    segment_steps_max: float


def _resolve_optional_float(value: float | None, fallback: float) -> float:
    return float(fallback if value is None else value)


def infer_search_bounds(
    *,
    config_payload: dict[str, Any],
    bw_min_mbps: float | None,
    bw_max_mbps: float | None,
    loss_min: float | None,
    loss_max: float | None,
    segment_steps_min: float,
    segment_steps_max: float,
) -> SharedScheduleSearchBounds:
    if attack_mode_from_config(config_payload) != "independent_gap":
        raise ValueError("Sage Genet search currently supports only independent_gap configs")

    bounds = attack_bounds_from_config(config_payload)
    shared_bw_min = max(float(bounds.uplink_bw_mbps[0]), float(bounds.downlink_bw_mbps[0]))
    shared_bw_max = min(float(bounds.uplink_bw_mbps[1]), float(bounds.downlink_bw_mbps[1]))
    shared_loss_min = max(float(bounds.uplink_loss[0]), float(bounds.downlink_loss[0]))
    shared_loss_max = min(float(bounds.uplink_loss[1]), float(bounds.downlink_loss[1]))

    if shared_bw_min > shared_bw_max:
        raise ValueError("shared bandwidth bounds do not overlap")
    if shared_loss_min > shared_loss_max:
        raise ValueError("shared loss bounds do not overlap")

    resolved_bw_min = _resolve_optional_float(bw_min_mbps, shared_bw_min)
    resolved_bw_max = _resolve_optional_float(bw_max_mbps, shared_bw_max)
    resolved_loss_min = _resolve_optional_float(loss_min, shared_loss_min)
    resolved_loss_max = _resolve_optional_float(loss_max, shared_loss_max)
    if resolved_bw_min < shared_bw_min - 1e-9 or resolved_bw_max > shared_bw_max + 1e-9:
        raise ValueError(
            f"bandwidth search range [{resolved_bw_min}, {resolved_bw_max}] "
            f"lies outside config bounds [{shared_bw_min}, {shared_bw_max}]"
        )
    if resolved_loss_min < shared_loss_min - 1e-9 or resolved_loss_max > shared_loss_max + 1e-9:
        raise ValueError(
            f"loss search range [{resolved_loss_min}, {resolved_loss_max}] "
            f"lies outside config bounds [{shared_loss_min}, {shared_loss_max}]"
        )
    if resolved_bw_min > resolved_bw_max:
        raise ValueError("bandwidth min must be <= max")
    if resolved_loss_min > resolved_loss_max:
        raise ValueError("loss min must be <= max")
    if float(segment_steps_min) < 1.0:
        raise ValueError("segment_steps_min must be >= 1")
    if float(segment_steps_max) < float(segment_steps_min):
        raise ValueError("segment_steps_max must be >= segment_steps_min")
    return SharedScheduleSearchBounds(
        bw_min_mbps=float(resolved_bw_min),
        bw_max_mbps=float(resolved_bw_max),
        loss_min=float(resolved_loss_min),
        loss_max=float(resolved_loss_max),
        segment_steps_min=float(segment_steps_min),
        segment_steps_max=float(segment_steps_max),
    )


@dataclass(frozen=True)
class SharedScheduleFamilyParams:
    min_shared_bw_mbps: float
    max_shared_bw_mbps: float
    min_shared_loss: float
    max_shared_loss: float
    segment_steps: float

    def normalized(self, *, bounds: SharedScheduleSearchBounds) -> "SharedScheduleFamilyParams":
        min_bw = float(np.clip(float(self.min_shared_bw_mbps), float(bounds.bw_min_mbps), float(bounds.bw_max_mbps)))
        max_bw = float(np.clip(float(self.max_shared_bw_mbps), float(bounds.bw_min_mbps), float(bounds.bw_max_mbps)))
        max_bw = max(float(max_bw), float(min_bw))
        min_loss = float(np.clip(float(self.min_shared_loss), float(bounds.loss_min), float(bounds.loss_max)))
        max_loss = float(np.clip(float(self.max_shared_loss), float(bounds.loss_min), float(bounds.loss_max)))
        max_loss = max(float(max_loss), float(min_loss))
        segment_steps = float(
            np.clip(float(self.segment_steps), float(bounds.segment_steps_min), float(bounds.segment_steps_max))
        )
        segment_steps = max(float(segment_steps), 1.0)
        return SharedScheduleFamilyParams(
            min_shared_bw_mbps=float(min_bw),
            max_shared_bw_mbps=float(max_bw),
            min_shared_loss=float(min_loss),
            max_shared_loss=float(max_loss),
            segment_steps=float(segment_steps),
        )


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
class SharedScheduleRangeComponent:
    min_shared_bw_range_mbps: tuple[float, float]
    max_shared_bw_range_mbps: tuple[float, float]
    min_shared_loss_range: tuple[float, float]
    max_shared_loss_range: tuple[float, float]
    segment_steps_range: tuple[float, float]
    weight: float

    def sample_params(
        self,
        *,
        rng: np.random.RandomState,
        bounds: SharedScheduleSearchBounds,
    ) -> SharedScheduleFamilyParams:
        params = SharedScheduleFamilyParams(
            min_shared_bw_mbps=_sample_uniform(
                rng,
                float(self.min_shared_bw_range_mbps[0]),
                float(self.min_shared_bw_range_mbps[1]),
            ),
            max_shared_bw_mbps=_sample_log_uniform(
                rng,
                float(self.max_shared_bw_range_mbps[0]),
                float(self.max_shared_bw_range_mbps[1]),
            ),
            min_shared_loss=_sample_uniform(
                rng,
                float(self.min_shared_loss_range[0]),
                float(self.min_shared_loss_range[1]),
            ),
            max_shared_loss=_sample_uniform(
                rng,
                float(self.max_shared_loss_range[0]),
                float(self.max_shared_loss_range[1]),
            ),
            segment_steps=_sample_uniform(
                rng,
                float(self.segment_steps_range[0]),
                float(self.segment_steps_range[1]),
            ),
        )
        return params.normalized(bounds=bounds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_shared_bw_range_mbps": [
                float(self.min_shared_bw_range_mbps[0]),
                float(self.min_shared_bw_range_mbps[1]),
            ],
            "max_shared_bw_range_mbps": [
                float(self.max_shared_bw_range_mbps[0]),
                float(self.max_shared_bw_range_mbps[1]),
            ],
            "min_shared_loss_range": [
                float(self.min_shared_loss_range[0]),
                float(self.min_shared_loss_range[1]),
            ],
            "max_shared_loss_range": [
                float(self.max_shared_loss_range[0]),
                float(self.max_shared_loss_range[1]),
            ],
            "segment_steps_range": [
                float(self.segment_steps_range[0]),
                float(self.segment_steps_range[1]),
            ],
            "weight": float(self.weight),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SharedScheduleRangeComponent":
        return cls(
            min_shared_bw_range_mbps=(
                float(payload["min_shared_bw_range_mbps"][0]),
                float(payload["min_shared_bw_range_mbps"][1]),
            ),
            max_shared_bw_range_mbps=(
                float(payload["max_shared_bw_range_mbps"][0]),
                float(payload["max_shared_bw_range_mbps"][1]),
            ),
            min_shared_loss_range=(
                float(payload["min_shared_loss_range"][0]),
                float(payload["min_shared_loss_range"][1]),
            ),
            max_shared_loss_range=(
                float(payload["max_shared_loss_range"][0]),
                float(payload["max_shared_loss_range"][1]),
            ),
            segment_steps_range=(
                float(payload["segment_steps_range"][0]),
                float(payload["segment_steps_range"][1]),
            ),
            weight=float(payload["weight"]),
        )


@dataclass(frozen=True)
class SampledFamily:
    params: SharedScheduleFamilyParams
    component_index: int


class SharedScheduleCurriculum:
    def __init__(
        self,
        *,
        bounds: SharedScheduleSearchBounds,
        components: list[SharedScheduleRangeComponent],
    ) -> None:
        self.bounds = bounds
        self.components = list(components)
        self._validate()

    def _validate(self) -> None:
        if not self.components:
            raise ValueError("curriculum must contain at least one component")
        total_weight = 0.0
        for component in self.components:
            if float(component.weight) < 0.0:
                raise ValueError("curriculum weights must be non-negative")
            total_weight += float(component.weight)
        if total_weight <= 0.0:
            raise ValueError("curriculum weights must sum to a positive value")

    @classmethod
    def from_search_space(
        cls,
        *,
        bounds: SharedScheduleSearchBounds,
    ) -> "SharedScheduleCurriculum":
        return cls(
            bounds=bounds,
            components=[
                SharedScheduleRangeComponent(
                    min_shared_bw_range_mbps=(float(bounds.bw_min_mbps), float(bounds.bw_max_mbps)),
                    max_shared_bw_range_mbps=(float(bounds.bw_min_mbps), float(bounds.bw_max_mbps)),
                    min_shared_loss_range=(float(bounds.loss_min), float(bounds.loss_max)),
                    max_shared_loss_range=(float(bounds.loss_min), float(bounds.loss_max)),
                    segment_steps_range=(float(bounds.segment_steps_min), float(bounds.segment_steps_max)),
                    weight=1.0,
                )
            ],
        )

    def normalized_weights(self) -> np.ndarray:
        weights = np.asarray([float(component.weight) for component in self.components], dtype=np.float64)
        weight_sum = float(np.sum(weights))
        if weight_sum <= 0.0:
            raise ValueError("curriculum weights must sum to a positive value")
        return weights / weight_sum

    def sample_family(self, *, rng: np.random.RandomState) -> SampledFamily:
        weights = self.normalized_weights()
        component_index = int(rng.choice(len(self.components), p=weights))
        params = self.components[component_index].sample_params(
            rng=rng,
            bounds=self.bounds,
        )
        return SampledFamily(
            params=params,
            component_index=component_index,
        )

    def add_point_component(
        self,
        *,
        params: SharedScheduleFamilyParams,
        add_probability: float,
    ) -> "SharedScheduleCurriculum":
        probability = float(add_probability)
        if probability < 0.0 or probability > 1.0:
            raise ValueError("add_probability must be in [0, 1]")
        if probability == 0.0:
            return SharedScheduleCurriculum(
                bounds=self.bounds,
                components=[
                    SharedScheduleRangeComponent.from_dict(component.to_dict()) for component in self.components
                ],
            )

        current_weights = self.normalized_weights()
        next_components: list[SharedScheduleRangeComponent] = []
        for component, weight in zip(self.components, list(current_weights)):
            payload = component.to_dict()
            payload["weight"] = float(weight) * (1.0 - probability)
            next_components.append(SharedScheduleRangeComponent.from_dict(payload))
        normalized_params = params.normalized(bounds=self.bounds)
        next_components.append(
            SharedScheduleRangeComponent(
                min_shared_bw_range_mbps=(
                    float(normalized_params.min_shared_bw_mbps),
                    float(normalized_params.min_shared_bw_mbps),
                ),
                max_shared_bw_range_mbps=(
                    float(normalized_params.max_shared_bw_mbps),
                    float(normalized_params.max_shared_bw_mbps),
                ),
                min_shared_loss_range=(
                    float(normalized_params.min_shared_loss),
                    float(normalized_params.min_shared_loss),
                ),
                max_shared_loss_range=(
                    float(normalized_params.max_shared_loss),
                    float(normalized_params.max_shared_loss),
                ),
                segment_steps_range=(
                    float(normalized_params.segment_steps),
                    float(normalized_params.segment_steps),
                ),
                weight=float(probability),
            )
        )
        return SharedScheduleCurriculum(
            bounds=self.bounds,
            components=next_components,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "bounds": asdict(self.bounds),
            "components": [component.to_dict() for component in self.components],
        }

    def dump(self, path: str) -> None:
        save_json(path, self.to_dict())

    @classmethod
    def load(cls, path: str) -> "SharedScheduleCurriculum":
        with open(path, "r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
        bounds = SharedScheduleSearchBounds(**dict(payload["bounds"]))
        return cls(
            bounds=bounds,
            components=[
                SharedScheduleRangeComponent.from_dict(component_payload)
                for component_payload in list(payload["components"])
            ],
        )


class SharedSchedulePolicy:
    def __init__(
        self,
        *,
        env: ParallelGapAttackEnv,
        rng: np.random.RandomState,
        params: SharedScheduleFamilyParams,
    ) -> None:
        self._env = env
        self._rng = rng
        self._params = params
        self._remaining_hold_steps = 0
        self._current_policy_action: np.ndarray | None = None

    def _sample_hold_steps(self) -> int:
        max_hold_steps = max(int(math.ceil(float(self._params.segment_steps))), 1)
        return int(self._rng.randint(1, max_hold_steps + 1))

    def _sample_effective_action(self) -> np.ndarray:
        shared_bw = _sample_uniform(
            self._rng,
            float(self._params.min_shared_bw_mbps),
            float(self._params.max_shared_bw_mbps),
        )
        shared_loss = _sample_uniform(
            self._rng,
            float(self._params.min_shared_loss),
            float(self._params.max_shared_loss),
        )
        return np.asarray(
            [
                float(shared_bw),
                float(shared_bw),
                float(shared_loss),
                float(shared_loss),
                float(self._env._fixed_uplink_delay_ms),
                float(self._env._fixed_downlink_delay_ms),
            ],
            dtype=np.float32,
        )

    def _policy_action_from_effective(self, effective_action: np.ndarray) -> np.ndarray:
        _, policy_action = self._env._expand_effective_action(np.asarray(effective_action, dtype=np.float32))
        return np.asarray(policy_action, dtype=np.float32)

    def __call__(self, observation: np.ndarray, info: dict[str, Any], step: int) -> np.ndarray:
        if self._current_policy_action is None or self._remaining_hold_steps <= 0:
            effective_action = self._sample_effective_action()
            self._current_policy_action = self._policy_action_from_effective(effective_action)
            self._remaining_hold_steps = self._sample_hold_steps()
        self._remaining_hold_steps -= 1
        return np.asarray(self._current_policy_action, dtype=np.float32)


def build_gap_env(
    *,
    repo_root: str,
    config_payload: dict[str, Any],
    launch_config: SageLaunchConfig,
    runtime_dir: str,
    baseline_methods: tuple[str, ...] | None = None,
) -> ParallelGapAttackEnv:
    if attack_mode_from_config(config_payload) != "independent_gap":
        raise ValueError("Sage Genet utilities currently support only independent_gap configs")
    return ParallelGapAttackEnv(
        repo_root=repo_root,
        launch_config=launch_config,
        bounds=attack_bounds_from_config(config_payload),
        obs_history_len=int(config_payload.get("obs_history_len", 4)),
        attack_interval_ms=float(config_payload.get("attack_interval_ms", 100.0)),
        max_episode_steps=int(config_payload.get("episode_steps", 6000)),
        launch_timeout_s=float(config_payload.get("launch_timeout_s", 90.0)),
        step_timeout_s=float(config_payload.get("step_timeout_s", 10.0)),
        runtime_dir=runtime_dir,
        baseline_gap_alpha=float(config_payload.get("baseline_gap_alpha", 2.0)),
        baseline_hard_max=bool(config_payload.get("baseline_hard_max", False)),
        baseline_methods=tuple(baseline_methods or DEFAULT_BASELINE_METHODS),
        smooth_penalty_scale=float(config_payload.get("smooth_penalty_scale", 0.0)),
        sync_guard_ms=float(config_payload.get("sync_guard_ms", 25.0)),
        launch_retries=int(config_payload.get("gap_launch_retries", 6)),
        shared_bin_loss_enabled=bool(config_payload.get("shared_bin_loss_enabled", False)),
        shared_bin_loss_bin_ms=float(config_payload.get("shared_bin_loss_bin_ms", 5.0)),
    )


def prepare_output_dir(out_dir: str, *, overwrite: bool) -> None:
    if os.path.isdir(out_dir):
        has_entries = len(os.listdir(out_dir)) > 0
        if has_entries and not overwrite:
            raise RuntimeError(f"output dir is not empty: {out_dir}. Re-run with --overwrite.")
        if has_entries and overwrite:
            shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)


def _bundle_payload(
    *,
    repo_root: str,
    out_dir: str,
    trace_index: int,
    trace_id: str,
    result,
    config_path: str,
    config_payload: dict[str, Any],
    generator_type: str,
    family_params: SharedScheduleFamilyParams,
    trace_seed: int,
    component_index: int | None = None,
) -> dict[str, Any]:
    bundle_dir = os.path.join(out_dir, f"{trace_index:03d}-{trace_id}")
    os.makedirs(bundle_dir, exist_ok=True)

    uplink_trace_path = os.path.join(bundle_dir, "uplink.trace")
    downlink_trace_path = os.path.join(bundle_dir, "downlink.trace")
    write_bandwidth_trace(
        bandwidth_mbps=[
            float(
                record.get(
                    "attacker_uplink_bw_mbps",
                    record.get(
                        "uplink_bw_mbps",
                        (
                            record.get("effective_action", record["action"])[0]
                            if record.get("effective_action", record["action"])
                            else 0.0
                        ),
                    ),
                )
            )
            for record in result.step_records
        ],
        interval_ms=float(config_payload.get("attack_interval_ms", 100.0)),
        out_path=uplink_trace_path,
    )
    write_bandwidth_trace(
        bandwidth_mbps=[
            float(
                record.get(
                    "attacker_downlink_bw_mbps",
                    record.get(
                        "downlink_bw_mbps",
                        (
                            record.get("effective_action", record["action"])[1]
                            if len(record.get("effective_action", record["action"])) > 1
                            else record.get("effective_action", record["action"])[0]
                        ),
                    ),
                )
            )
            for record in result.step_records
        ],
        interval_ms=float(config_payload.get("attack_interval_ms", 100.0)),
        out_path=downlink_trace_path,
    )

    schedule_payload = {
        "created_at_utc": utc_now_iso(),
        "attack_mode": attack_mode_from_config(config_payload),
        "trace_id": trace_id,
        "trace_name": trace_id,
        "model_path": None,
        "training_config_path": config_path,
        "attack_interval_ms": float(config_payload.get("attack_interval_ms", 100.0)),
        "shared_bin_loss_enabled": bool(config_payload.get("shared_bin_loss_enabled", False)),
        "shared_bin_loss_bin_ms": float(config_payload.get("shared_bin_loss_bin_ms", 5.0)),
        "num_steps": int(result.num_steps),
        "metrics": result.metrics,
        "steps": result.step_records,
        "generator_type": str(generator_type),
        "family_params": asdict(family_params),
        "trace_seed": int(trace_seed),
    }
    if component_index is not None:
        schedule_payload["curriculum_component_index"] = int(component_index)
    schedule_path = os.path.join(bundle_dir, "schedule.json")
    save_json(schedule_path, schedule_payload)

    entry = {
        "trace_id": trace_id,
        "trace_name": trace_id,
        "bundle_dir": os.path.relpath(bundle_dir, repo_root),
        "schedule_path": os.path.relpath(schedule_path, repo_root),
        "uplink_trace_path": os.path.relpath(uplink_trace_path, repo_root),
        "downlink_trace_path": os.path.relpath(downlink_trace_path, repo_root),
        "metrics": result.metrics,
        "family_params": asdict(family_params),
        "trace_seed": int(trace_seed),
    }
    if component_index is not None:
        entry["curriculum_component_index"] = int(component_index)
    return entry


def write_generated_manifest(
    *,
    repo_root: str,
    out_dir: str,
    config_path: str,
    config_payload: dict[str, Any],
    test_manifest_path: str,
    trace_set_name: str,
    generated_entries: list[dict[str, Any]],
    generator_type: str,
    sage_model_dir: str,
    baseline_methods: tuple[str, ...],
    extra_payload: dict[str, Any] | None = None,
) -> str:
    test_entries = load_trace_entries(test_manifest_path)
    payload = {
        "created_at_utc": utc_now_iso(),
        "repo_root": repo_root,
        "trace_set_name": trace_set_name,
        "attack_mode": attack_mode_from_config(config_payload),
        "baseline_methods": list(baseline_methods),
        "model_path": None,
        "training_config_path": config_path,
        "test_manifest_resolved": test_manifest_path,
        "attack_interval_ms": float(config_payload.get("attack_interval_ms", 100.0)),
        "num_reference_test_traces": len(test_entries),
        "num_generated_traces": len(generated_entries),
        "generated_entries": generated_entries,
        "generator_type": str(generator_type),
        "sage_model_dir": str(sage_model_dir),
    }
    if extra_payload:
        payload.update(dict(extra_payload))
    manifest_path = os.path.join(out_dir, "generated_manifest.json")
    save_json(manifest_path, payload)
    return manifest_path


class SageHardGapBOObjective:
    def __init__(
        self,
        *,
        env: ParallelGapAttackEnv,
        bounds: SharedScheduleSearchBounds,
        seed: int,
        traces_per_eval: int,
        episode_steps: int,
        log_path: str,
    ) -> None:
        self.env = env
        self.bounds = bounds
        self.seed = int(seed)
        self.traces_per_eval = int(traces_per_eval)
        self.episode_steps = int(episode_steps)
        self.log_path = str(log_path)
        self.eval_id = 0

    def resolve_params(
        self,
        *,
        min_shared_bw_mbps: float,
        log_bw_span_mbps: float,
        min_shared_loss: float,
        loss_span: float,
        segment_steps: float | None = None,
    ) -> SharedScheduleFamilyParams:
        resolved_segment_steps = (
            float(self.bounds.segment_steps_min) if segment_steps is None else float(segment_steps)
        )
        bw_span_mbps = max(float(np.exp(float(log_bw_span_mbps)) - 1.0), 0.0)
        loss_span_resolved = max(float(loss_span), 0.0)
        params = SharedScheduleFamilyParams(
            min_shared_bw_mbps=float(min_shared_bw_mbps),
            max_shared_bw_mbps=float(min_shared_bw_mbps) + float(bw_span_mbps),
            min_shared_loss=float(min_shared_loss),
            max_shared_loss=float(min_shared_loss) + float(loss_span_resolved),
            segment_steps=float(resolved_segment_steps),
        )
        return params.normalized(bounds=self.bounds)

    def __call__(
        self,
        min_shared_bw_mbps: float,
        log_bw_span_mbps: float,
        min_shared_loss: float,
        loss_span: float,
        segment_steps: float | None = None,
    ) -> float:
        eval_id = int(self.eval_id)
        self.eval_id += 1
        params = self.resolve_params(
            min_shared_bw_mbps=float(min_shared_bw_mbps),
            log_bw_span_mbps=float(log_bw_span_mbps),
            min_shared_loss=float(min_shared_loss),
            loss_span=float(loss_span),
            segment_steps=segment_steps,
        )

        hard_gap_values: list[float] = []
        trace_rows: list[dict[str, Any]] = []
        for trace_idx in range(int(self.traces_per_eval)):
            trace_seed = int(self.seed + eval_id * 10_000 + trace_idx)
            rng = np.random.RandomState(int(trace_seed))
            policy = SharedSchedulePolicy(
                env=self.env,
                rng=rng,
                params=params,
            )
            result = run_online_policy_episode(
                self.env,
                action_fn=policy,
                max_steps=int(self.episode_steps),
                episode_id=f"bo-e{int(eval_id):04d}-t{int(trace_idx):03d}",
            )
            hard_gap = float(result.metrics.get("gap_best_baseline_gap_mean", 0.0))
            hard_gap_values.append(float(hard_gap))
            trace_rows.append(
                {
                    "trace_idx": int(trace_idx),
                    "trace_seed": int(trace_seed),
                    "hard_gap_mean": float(hard_gap),
                    "hard_best_baseline_score_mean": float(result.metrics.get("gap_best_baseline_score_mean", 0.0)),
                    "sage_score_mean": float(result.metrics.get("gap_score_sage_mean", 0.0)),
                    "num_steps": int(result.num_steps),
                    "episode_total_reward": float(result.total_reward),
                }
            )

        target = float(np.mean(np.asarray(hard_gap_values, dtype=np.float64)))
        log_payload = {
            "eval_id": int(eval_id),
            "objective": "hard_absolute_gap",
            "target": float(target),
            "hard_gap_mean": float(target),
            "hard_gap_std": float(np.std(np.asarray(hard_gap_values, dtype=np.float64))),
            "params": asdict(params),
            "raw_params": {
                "min_shared_bw_mbps": float(min_shared_bw_mbps),
                "log_bw_span_mbps": float(log_bw_span_mbps),
                "bw_span_mbps": float(max(np.exp(float(log_bw_span_mbps)) - 1.0, 0.0)),
                "max_shared_bw_mbps": float(params.max_shared_bw_mbps),
                "min_shared_loss": float(min_shared_loss),
                "loss_span": float(max(float(loss_span), 0.0)),
                "max_shared_loss": float(params.max_shared_loss),
                "segment_steps": float(params.segment_steps if segment_steps is None else segment_steps),
            },
            "traces": trace_rows,
        }
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as file_obj:
            file_obj.write(json.dumps(log_payload, sort_keys=True) + "\n")
        return float(target)


@dataclass(frozen=True)
class SageGenetSearchResult:
    best_params: SharedScheduleFamilyParams
    best_target: float
    best_raw_params: dict[str, float]
    generated_manifest_path: str


def _generate_trace_entries_from_family(
    *,
    env: ParallelGapAttackEnv,
    repo_root: str,
    out_dir: str,
    config_path: str,
    config_payload: dict[str, Any],
    family_params: SharedScheduleFamilyParams,
    num_traces: int,
    seed: int,
    generator_type: str,
    component_index: int | None = None,
) -> list[dict[str, Any]]:
    generated_entries: list[dict[str, Any]] = []
    for trace_index in range(int(num_traces)):
        trace_id = f"generated-{trace_index:03d}"
        trace_seed = int(seed + 1_000_000 + trace_index)
        rng = np.random.RandomState(int(trace_seed))
        policy = SharedSchedulePolicy(
            env=env,
            rng=rng,
            params=family_params,
        )
        result = run_online_policy_episode(
            env,
            action_fn=policy,
            max_steps=int(config_payload.get("episode_steps", 6000)),
            episode_id=trace_id,
        )
        generated_entries.append(
            _bundle_payload(
                repo_root=repo_root,
                out_dir=out_dir,
                trace_index=trace_index,
                trace_id=trace_id,
                result=result,
                config_path=config_path,
                config_payload=config_payload,
                generator_type=generator_type,
                family_params=family_params,
                trace_seed=trace_seed,
                component_index=component_index,
            )
        )
    return generated_entries


def _generate_trace_entries_from_curriculum(
    *,
    env: ParallelGapAttackEnv,
    repo_root: str,
    out_dir: str,
    config_path: str,
    config_payload: dict[str, Any],
    curriculum: SharedScheduleCurriculum,
    num_traces: int,
    seed: int,
    generator_type: str,
) -> list[dict[str, Any]]:
    generated_entries: list[dict[str, Any]] = []
    for trace_index in range(int(num_traces)):
        trace_id = f"generated-{trace_index:03d}"
        trace_seed = int(seed + 2_000_000 + trace_index)
        rng = np.random.RandomState(int(trace_seed))
        sampled_family = curriculum.sample_family(rng=rng)
        policy = SharedSchedulePolicy(
            env=env,
            rng=rng,
            params=sampled_family.params,
        )
        result = run_online_policy_episode(
            env,
            action_fn=policy,
            max_steps=int(config_payload.get("episode_steps", 6000)),
            episode_id=trace_id,
        )
        generated_entries.append(
            _bundle_payload(
                repo_root=repo_root,
                out_dir=out_dir,
                trace_index=trace_index,
                trace_id=trace_id,
                result=result,
                config_path=config_path,
                config_payload=config_payload,
                generator_type=generator_type,
                family_params=sampled_family.params,
                trace_seed=trace_seed,
                component_index=sampled_family.component_index,
            )
        )
    return generated_entries


class SageGenetBOTraceGenerator:
    def __init__(
        self,
        *,
        repo_root: str,
        config_path: str,
        config_payload: dict[str, Any],
        sage_model_dir: str,
        test_manifest_path: str,
        out_dir: str,
        runtime_dir: str,
        bounds: SharedScheduleSearchBounds,
        seed: int,
        num_traces: int,
        traces_per_eval: int,
        bo_init_points: int,
        bo_iter: int,
        optimize_segment_steps: bool,
        fixed_segment_steps: float,
        overwrite: bool,
    ) -> None:
        self.repo_root = os.path.abspath(repo_root)
        self.config_path = str(config_path)
        self.config_payload = dict(config_payload)
        self.sage_model_dir = resolve_repo_path(self.repo_root, str(sage_model_dir))
        self.test_manifest_path = ensure_test_manifest(self.repo_root, str(test_manifest_path))
        self.out_dir = resolve_repo_path(self.repo_root, str(out_dir))
        self.runtime_dir = str(runtime_dir)
        self.bounds = bounds
        self.seed = int(seed)
        self.num_traces = int(num_traces)
        self.traces_per_eval = int(traces_per_eval)
        self.bo_init_points = int(bo_init_points)
        self.bo_iter = int(bo_iter)
        self.optimize_segment_steps = bool(optimize_segment_steps)
        self.fixed_segment_steps = float(fixed_segment_steps)
        self.overwrite = bool(overwrite)
        self.trace_set_name = trace_set_name_from_out_dir(self.out_dir)
        self.baseline_methods = tuple(DEFAULT_BASELINE_METHODS)

        if not tf_checkpoint_exists(self.sage_model_dir):
            raise FileNotFoundError(f"missing TensorFlow checkpoint in Sage model dir: {self.sage_model_dir}")
        if self.num_traces <= 0:
            self.num_traces = len(load_trace_entries(self.test_manifest_path))
        if self.num_traces <= 0:
            raise ValueError("num_traces must be positive")
        if self.traces_per_eval <= 0:
            raise ValueError("traces_per_eval must be positive")
        if self.bo_init_points < 0 or self.bo_iter < 0:
            raise ValueError("bo_init_points and bo_iter must be non-negative")

    def _write_best_config(
        self,
        *,
        best_params: SharedScheduleFamilyParams,
        best_raw: dict[str, float],
        best_target: float,
    ) -> None:
        save_json(
            os.path.join(self.out_dir, "best_config.json"),
            {
                "objective": "hard_absolute_gap",
                "best_target": float(best_target),
                "best_params": asdict(best_params),
                "best_raw_params": dict(best_raw),
                "optimize_segment_steps": bool(self.optimize_segment_steps),
                "baseline_methods": list(self.baseline_methods),
            },
        )

    def run(self) -> SageGenetSearchResult:
        prepare_output_dir(self.out_dir, overwrite=self.overwrite)
        BayesianOptimization, UtilityFunction = _require_bayes_opt()
        run_namespace = acquire_run_namespace(
            repo_root=self.repo_root,
            runtime_dir=str(self.runtime_dir),
            actor_id=int(self.config_payload.get("actor_id", 900)),
            port=int(self.config_payload.get("port", 5101)),
            label=str(self.trace_set_name or "sage-genet-search"),
            ports_per_run=len(self.baseline_methods) + 1,
        )
        previous_ckpt_dir = os.environ.get("SAGE_EVAL_CKPT_DIR")
        env: ParallelGapAttackEnv | None = None
        try:
            os.environ["SAGE_EVAL_CKPT_DIR"] = str(self.sage_model_dir)
            launch_config = resolved_launch_config(
                config_payload=self.config_payload,
                run_namespace=run_namespace,
                log_prefix="sage-genet-search",
            )
            env = build_gap_env(
                repo_root=self.repo_root,
                config_payload=self.config_payload,
                launch_config=launch_config,
                runtime_dir=run_namespace.runtime_dir,
                baseline_methods=self.baseline_methods,
            )
            objective = SageHardGapBOObjective(
                env=env,
                bounds=self.bounds,
                seed=int(self.seed),
                traces_per_eval=int(self.traces_per_eval),
                episode_steps=int(self.config_payload.get("episode_steps", 6000)),
                log_path=os.path.join(self.out_dir, "bo_log.jsonl"),
            )
            max_bw_span_mbps = max(float(self.bounds.bw_max_mbps) - float(self.bounds.bw_min_mbps), 0.0)
            max_loss_span = max(float(self.bounds.loss_max) - float(self.bounds.loss_min), 0.0)
            pbounds: dict[str, tuple[float, float]] = {
                "min_shared_bw_mbps": (float(self.bounds.bw_min_mbps), float(self.bounds.bw_max_mbps)),
                "log_bw_span_mbps": (
                    0.0,
                    float(math.log1p(max_bw_span_mbps)),
                ),
                "min_shared_loss": (float(self.bounds.loss_min), float(self.bounds.loss_max)),
                "loss_span": (0.0, float(max_loss_span)),
            }
            if self.optimize_segment_steps:
                pbounds["segment_steps"] = (
                    float(self.bounds.segment_steps_min),
                    float(self.bounds.segment_steps_max),
                )

            optimizer = BayesianOptimization(
                f=objective,
                pbounds=pbounds,
                random_state=int(self.seed),
            )
            acquisition = UtilityFunction(
                kind="ucb",
                kappa=20,
                xi=0.1,
                kappa_decay=1,
                kappa_decay_delay=0,
            )
            optimizer.maximize(
                init_points=int(self.bo_init_points),
                n_iter=int(self.bo_iter),
                acquisition_function=acquisition,
            )
            if not optimizer.max:
                raise RuntimeError("BO search did not produce a best result")

            best_raw = dict(optimizer.max["params"])
            best_params = objective.resolve_params(
                min_shared_bw_mbps=float(best_raw["min_shared_bw_mbps"]),
                log_bw_span_mbps=float(best_raw["log_bw_span_mbps"]),
                min_shared_loss=float(best_raw["min_shared_loss"]),
                loss_span=float(best_raw["loss_span"]),
                segment_steps=best_raw.get("segment_steps", self.fixed_segment_steps),
            )
            best_target = float(optimizer.max["target"])
            best_raw_payload = {
                "min_shared_bw_mbps": float(best_raw["min_shared_bw_mbps"]),
                "log_bw_span_mbps": float(best_raw["log_bw_span_mbps"]),
                "bw_span_mbps": float(max(np.exp(float(best_raw["log_bw_span_mbps"])) - 1.0, 0.0)),
                "max_shared_bw_mbps": float(best_params.max_shared_bw_mbps),
                "min_shared_loss": float(best_raw["min_shared_loss"]),
                "loss_span": float(max(float(best_raw["loss_span"]), 0.0)),
                "max_shared_loss": float(best_params.max_shared_loss),
                "segment_steps": float(best_raw.get("segment_steps", best_params.segment_steps)),
            }
            self._write_best_config(
                best_params=best_params,
                best_raw=best_raw_payload,
                best_target=best_target,
            )

            generated_entries = _generate_trace_entries_from_family(
                env=env,
                repo_root=self.repo_root,
                out_dir=self.out_dir,
                config_path=self.config_path,
                config_payload=self.config_payload,
                family_params=best_params,
                num_traces=int(self.num_traces),
                seed=int(self.seed),
                generator_type="sage_genet_bo_round1",
            )
            manifest_path = write_generated_manifest(
                repo_root=self.repo_root,
                out_dir=self.out_dir,
                config_path=self.config_path,
                config_payload=self.config_payload,
                test_manifest_path=self.test_manifest_path,
                trace_set_name=self.trace_set_name,
                generated_entries=generated_entries,
                generator_type="sage_genet_bo_round1",
                sage_model_dir=self.sage_model_dir,
                baseline_methods=self.baseline_methods,
                extra_payload={
                    "objective": "hard_absolute_gap",
                    "best_target": float(best_target),
                    "best_params": asdict(best_params),
                    "best_raw_params": best_raw_payload,
                    "bo_init_points": int(self.bo_init_points),
                    "bo_iter": int(self.bo_iter),
                    "traces_per_eval": int(self.traces_per_eval),
                    "optimize_segment_steps": bool(self.optimize_segment_steps),
                },
            )
            return SageGenetSearchResult(
                best_params=best_params,
                best_target=float(best_target),
                best_raw_params=best_raw_payload,
                generated_manifest_path=manifest_path,
            )
        finally:
            if env is not None:
                env.close()
            run_namespace.release()
            if previous_ckpt_dir is None:
                os.environ.pop("SAGE_EVAL_CKPT_DIR", None)
            else:
                os.environ["SAGE_EVAL_CKPT_DIR"] = previous_ckpt_dir


def sample_curriculum_trace_set(
    *,
    repo_root: str,
    config_path: str,
    config_payload: dict[str, Any],
    sage_model_dir: str,
    test_manifest_path: str,
    out_dir: str,
    runtime_dir: str,
    curriculum: SharedScheduleCurriculum,
    seed: int,
    num_traces: int,
    overwrite: bool,
) -> str:
    repo_root_resolved = os.path.abspath(repo_root)
    sage_model_dir_resolved = resolve_repo_path(repo_root_resolved, str(sage_model_dir))
    if not tf_checkpoint_exists(sage_model_dir_resolved):
        raise FileNotFoundError(f"missing TensorFlow checkpoint in Sage model dir: {sage_model_dir_resolved}")

    out_dir_resolved = resolve_repo_path(repo_root_resolved, str(out_dir))
    prepare_output_dir(out_dir_resolved, overwrite=bool(overwrite))
    trace_set_name = trace_set_name_from_out_dir(out_dir_resolved)
    baseline_methods = tuple(DEFAULT_BASELINE_METHODS)
    test_manifest_resolved = ensure_test_manifest(repo_root_resolved, str(test_manifest_path))
    if int(num_traces) <= 0:
        num_traces = len(load_trace_entries(test_manifest_resolved))
    if int(num_traces) <= 0:
        raise ValueError("num_traces must be positive")

    run_namespace = acquire_run_namespace(
        repo_root=repo_root_resolved,
        runtime_dir=str(runtime_dir),
        actor_id=int(config_payload.get("actor_id", 900)),
        port=int(config_payload.get("port", 5101)),
        label=str(trace_set_name or "sage-genet-curriculum"),
        ports_per_run=len(baseline_methods) + 1,
    )
    previous_ckpt_dir = os.environ.get("SAGE_EVAL_CKPT_DIR")
    env: ParallelGapAttackEnv | None = None
    try:
        os.environ["SAGE_EVAL_CKPT_DIR"] = str(sage_model_dir_resolved)
        launch_config = resolved_launch_config(
            config_payload=config_payload,
            run_namespace=run_namespace,
            log_prefix="sage-genet-curriculum",
        )
        env = build_gap_env(
            repo_root=repo_root_resolved,
            config_payload=config_payload,
            launch_config=launch_config,
            runtime_dir=run_namespace.runtime_dir,
            baseline_methods=baseline_methods,
        )
        generated_entries = _generate_trace_entries_from_curriculum(
            env=env,
            repo_root=repo_root_resolved,
            out_dir=out_dir_resolved,
            config_path=config_path,
            config_payload=config_payload,
            curriculum=curriculum,
            num_traces=int(num_traces),
            seed=int(seed),
            generator_type="sage_genet_curriculum",
        )
        return write_generated_manifest(
            repo_root=repo_root_resolved,
            out_dir=out_dir_resolved,
            config_path=config_path,
            config_payload=config_payload,
            test_manifest_path=test_manifest_resolved,
            trace_set_name=trace_set_name,
            generated_entries=generated_entries,
            generator_type="sage_genet_curriculum",
            sage_model_dir=sage_model_dir_resolved,
            baseline_methods=baseline_methods,
            extra_payload={
                "curriculum": curriculum.to_dict(),
                "objective": "hard_absolute_gap",
            },
        )
    finally:
        if env is not None:
            env.close()
        run_namespace.release()
        if previous_ckpt_dir is None:
            os.environ.pop("SAGE_EVAL_CKPT_DIR", None)
        else:
            os.environ["SAGE_EVAL_CKPT_DIR"] = previous_ckpt_dir
