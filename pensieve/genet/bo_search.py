from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .scoring import ObjectiveName, TraceScorer
from .trace_family import (
    BandwidthTraceParams,
    CookedTrace,
    DEFAULT_DURATION_S,
    generate_cooked_trace,
    write_cooked_trace,
    write_trace_metadata,
)


def _require_bayes_opt():
    try:
        from bayes_opt import BayesianOptimization  # type: ignore
        from bayes_opt.util import UtilityFunction  # type: ignore

        return BayesianOptimization, UtilityFunction
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "* Missing dependency: install packages from `genet/requirements.txt` to run Genet BO trace generation."
        ) from exc


@dataclass(frozen=True)
class SearchResult:
    best_params: BandwidthTraceParams
    best_target: float
    best_raw_params: dict[str, float]


def _write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _append_jsonl(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def _cleanup_output_dir(out_dir: str) -> None:
    if not os.path.isdir(out_dir):
        return

    for name in os.listdir(out_dir):
        if str(name).startswith("episode_") and (
            str(name).endswith(".trace") or str(name).endswith(".json")
        ):
            os.remove(os.path.join(out_dir, str(name)))
        elif str(name) in {"best_config.json", "manifest.json", "bo_log.jsonl"}:
            os.remove(os.path.join(out_dir, str(name)))


class BOObjective:
    def __init__(
        self,
        *,
        scorer: TraceScorer,
        objective: ObjectiveName,
        seed: int,
        traces_per_eval: int,
        duration_s: float,
        bw_min_mbps: float,
        bw_max_mbps: float,
        bw_change_interval: float,
        optimize_bw_change_interval: bool,
        log_path: str,
    ) -> None:
        self.scorer = scorer
        self.objective = str(objective)
        self.seed = int(seed)
        self.traces_per_eval = int(traces_per_eval)
        self.duration_s = float(duration_s)
        self.bw_min_mbps = float(bw_min_mbps)
        self.bw_max_mbps = float(bw_max_mbps)
        self.bw_change_interval = float(bw_change_interval)
        self.optimize_bw_change_interval = bool(optimize_bw_change_interval)
        self.log_path = str(log_path)
        self.eval_id = 0

    def resolve_params(
        self,
        *,
        min_bw_mbps: float,
        log_max_bw_mbps: float,
        bw_change_interval: float | None = None,
    ) -> BandwidthTraceParams:
        resolved_bw_change_interval = self.bw_change_interval
        if bw_change_interval is not None:
            resolved_bw_change_interval = float(bw_change_interval)
        params = BandwidthTraceParams(
            min_bw_mbps=float(min_bw_mbps),
            max_bw_mbps=float(np.exp(float(log_max_bw_mbps))),
            bw_change_interval=float(resolved_bw_change_interval),
        )
        return params.normalized(
            bw_floor_mbps=float(self.bw_min_mbps),
            bw_ceil_mbps=float(self.bw_max_mbps),
        )

    def __call__(
        self,
        min_bw_mbps: float,
        log_max_bw_mbps: float,
        bw_change_interval: float | None = None,
    ) -> float:
        eval_id = int(self.eval_id)
        self.eval_id += 1

        params = self.resolve_params(
            min_bw_mbps=float(min_bw_mbps),
            log_max_bw_mbps=float(log_max_bw_mbps),
            bw_change_interval=bw_change_interval,
        )

        trace_rows: list[dict[str, Any]] = []
        gaps: list[float] = []
        pensieve_qoes: list[float] = []
        baseline_qoes: list[float] = []
        for trace_idx in range(int(self.traces_per_eval)):
            trace_seed = int(self.seed + eval_id * 10_000 + trace_idx)
            rng = np.random.RandomState(int(trace_seed))
            trace = generate_cooked_trace(
                params=params,
                rng=rng,
                trace_seed=int(trace_seed),
                duration_s=float(self.duration_s),
            )
            score = self.scorer.score_trace(
                trace,
                objective=self.objective,
                seed=int(trace_seed),
            )
            gaps.append(float(score.gap))
            pensieve_qoes.append(float(score.pensieve_qoe))
            baseline_qoes.append(float(score.baseline_qoe))
            trace_rows.append(
                {
                    "trace_idx": int(trace_idx),
                    "trace_seed": int(trace_seed),
                    "gap": float(score.gap),
                    "pensieve_qoe": float(score.pensieve_qoe),
                    "baseline_qoe": float(score.baseline_qoe),
                    "baseline_name": str(score.baseline_name),
                }
            )

        target = float(np.mean(np.asarray(gaps, dtype=np.float64)))
        _append_jsonl(
            self.log_path,
            {
                "eval_id": int(eval_id),
                "target": float(target),
                "objective": str(self.objective),
                "params": asdict(params),
                "raw_params": {
                    "min_bw_mbps": float(min_bw_mbps),
                    "log_max_bw_mbps": float(log_max_bw_mbps),
                    "max_bw_mbps": float(np.exp(float(log_max_bw_mbps))),
                    "bw_change_interval": float(params.bw_change_interval),
                },
                "bw_change_interval_source": "optimized"
                if bool(self.optimize_bw_change_interval)
                else "fixed",
                "gap_mean": float(target),
                "gap_std": float(np.std(np.asarray(gaps, dtype=np.float64))),
                "pensieve_qoe_mean": float(np.mean(np.asarray(pensieve_qoes, dtype=np.float64))),
                "baseline_qoe_mean": float(np.mean(np.asarray(baseline_qoes, dtype=np.float64))),
                "traces": trace_rows,
            },
        )
        return float(target)


class GenetBOTraceGenerator:
    def __init__(
        self,
        *,
        scorer: TraceScorer,
        objective: ObjectiveName,
        seed: int,
        episodes: int,
        out_dir: str,
        bw_min_mbps: float,
        bw_max_mbps: float,
        bw_change_interval: float,
        optimize_bw_change_interval: bool,
        bw_change_interval_min: float,
        bw_change_interval_max: float,
        duration_s: float = DEFAULT_DURATION_S,
        traces_per_eval: int = 10,
        bo_init_points: int = 10,
        bo_iter: int = 5,
        overwrite: bool = False,
    ) -> None:
        self.scorer = scorer
        self.objective = str(objective)
        self.seed = int(seed)
        self.episodes = int(episodes)
        self.out_dir = str(out_dir)
        self.bw_min_mbps = float(bw_min_mbps)
        self.bw_max_mbps = float(bw_max_mbps)
        self.bw_change_interval = float(bw_change_interval)
        self.optimize_bw_change_interval = bool(optimize_bw_change_interval)
        self.bw_change_interval_min = float(bw_change_interval_min)
        self.bw_change_interval_max = float(bw_change_interval_max)
        self.duration_s = float(duration_s)
        self.traces_per_eval = int(traces_per_eval)
        self.bo_init_points = int(bo_init_points)
        self.bo_iter = int(bo_iter)
        self.overwrite = bool(overwrite)

        if self.episodes <= 0:
            raise ValueError("* episodes must be positive")
        if self.traces_per_eval <= 0:
            raise ValueError("* traces_per_eval must be positive")
        if self.bo_init_points < 0 or self.bo_iter < 0:
            raise ValueError("* bo_init_points and bo_iter must be non-negative")
        if self.bw_min_mbps <= 0.0 or self.bw_max_mbps < self.bw_min_mbps:
            raise ValueError("* invalid bandwidth bounds")
        if self.bw_change_interval <= 0.0:
            raise ValueError("* bw_change_interval must be positive")
        if self.optimize_bw_change_interval and (
            self.bw_change_interval_min <= 0.0 or self.bw_change_interval_max < self.bw_change_interval_min
        ):
            raise ValueError("* invalid bw_change_interval bounds")

    def _prepare_output_dir(self) -> None:
        os.makedirs(self.out_dir, exist_ok=True)
        existing_trace_files = [name for name in os.listdir(self.out_dir) if str(name).endswith(".trace")]
        if existing_trace_files and not bool(self.overwrite):
            raise RuntimeError(f"* output dir already contains trace files: {self.out_dir}. Re-run with --overwrite.")
        if bool(self.overwrite):
            _cleanup_output_dir(self.out_dir)

    def run(self) -> SearchResult:
        self._prepare_output_dir()
        BayesianOptimization, UtilityFunction = _require_bayes_opt()
        objective = BOObjective(
            scorer=self.scorer,
            objective=self.objective,
            seed=int(self.seed),
            traces_per_eval=int(self.traces_per_eval),
            duration_s=float(self.duration_s),
            bw_min_mbps=float(self.bw_min_mbps),
            bw_max_mbps=float(self.bw_max_mbps),
            bw_change_interval=float(self.bw_change_interval),
            optimize_bw_change_interval=bool(self.optimize_bw_change_interval),
            log_path=os.path.join(self.out_dir, "bo_log.jsonl"),
        )

        pbounds: dict[str, tuple[float, float]] = {
            "min_bw_mbps": (float(self.bw_min_mbps), float(self.bw_max_mbps)),
            "log_max_bw_mbps": (float(math.log(self.bw_min_mbps)), float(math.log(self.bw_max_mbps))),
        }
        if bool(self.optimize_bw_change_interval):
            pbounds["bw_change_interval"] = (
                float(self.bw_change_interval_min),
                float(self.bw_change_interval_max),
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
            raise RuntimeError("* BO search did not produce a best result")

        best_raw = dict(optimizer.max["params"])
        best_params = objective.resolve_params(
            min_bw_mbps=float(best_raw["min_bw_mbps"]),
            log_max_bw_mbps=float(best_raw["log_max_bw_mbps"]),
            bw_change_interval=best_raw.get("bw_change_interval"),
        )
        best_target = float(optimizer.max["target"])

        self._write_outputs(best_params=best_params, best_raw=best_raw, best_target=best_target)
        return SearchResult(
            best_params=best_params,
            best_target=float(best_target),
            best_raw_params={
                "min_bw_mbps": float(best_raw["min_bw_mbps"]),
                "log_max_bw_mbps": float(best_raw["log_max_bw_mbps"]),
                "max_bw_mbps": float(np.exp(float(best_raw["log_max_bw_mbps"]))),
                "bw_change_interval": float(best_params.bw_change_interval),
            },
        )

    def _write_outputs(
        self,
        *,
        best_params: BandwidthTraceParams,
        best_raw: dict[str, float],
        best_target: float,
    ) -> None:
        _write_json(
            os.path.join(self.out_dir, "best_config.json"),
            {
                "objective": str(self.objective),
                "best_target": float(best_target),
                "best_params": asdict(best_params),
                "best_raw_params": {
                    "min_bw_mbps": float(best_raw["min_bw_mbps"]),
                    "log_max_bw_mbps": float(best_raw["log_max_bw_mbps"]),
                    "max_bw_mbps": float(np.exp(float(best_raw["log_max_bw_mbps"]))),
                    "bw_change_interval": float(best_params.bw_change_interval),
                },
                "bw_change_interval_source": "optimized"
                if bool(self.optimize_bw_change_interval)
                else "fixed",
            },
        )

        for episode in range(int(self.episodes)):
            trace_seed = int(self.seed + 1_000_000 + episode)
            rng = np.random.RandomState(int(trace_seed))
            trace: CookedTrace = generate_cooked_trace(
                params=best_params,
                rng=rng,
                trace_seed=int(trace_seed),
                duration_s=float(self.duration_s),
            )
            trace_path = os.path.join(self.out_dir, f"episode_{episode:04d}.trace")
            write_cooked_trace(trace_path, trace)
            write_trace_metadata(
                os.path.join(self.out_dir, f"episode_{episode:04d}.json"),
                trace,
                extra={
                    "objective": str(self.objective),
                    "best_target": float(best_target),
                    "frozen_link_rtt_ms": 80.0,
                    "frozen_buffer_thresh_s": 60.0,
                },
            )

        _write_json(
            os.path.join(self.out_dir, "manifest.json"),
            {
                "kind": "genet_bo_trace_set",
                "objective": str(self.objective),
                "episodes": int(self.episodes),
                "seed": int(self.seed),
                "duration_s": float(self.duration_s),
                "traces_per_eval": int(self.traces_per_eval),
                "bo_init_points": int(self.bo_init_points),
                "bo_iter": int(self.bo_iter),
                "bandwidth_bounds_mbps": {
                    "min": float(self.bw_min_mbps),
                    "max": float(self.bw_max_mbps),
                },
                "bw_change_interval": {
                    "mode": "optimized" if bool(self.optimize_bw_change_interval) else "fixed",
                    "value": float(best_params.bw_change_interval),
                    "bounds": {
                        "min": float(self.bw_change_interval_min),
                        "max": float(self.bw_change_interval_max),
                    },
                },
                "frozen_link_rtt_ms": 80.0,
                "frozen_buffer_thresh_s": 60.0,
                "best_target": float(best_target),
                "best_params": asdict(best_params),
            },
        )
