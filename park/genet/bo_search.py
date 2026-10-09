from __future__ import annotations

import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from .scoring import ObjectiveName, ParkTraceScorer
from .trace_family import (
    GeneratedJobTrace,
    JobTraceParams,
    JobTraceSearchSpace,
    generate_job_trace,
    trace_metadata,
    write_generated_trace_csv,
    write_generated_trace_json,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def _cleanup_output_dir(out_dir: Path) -> None:
    if not out_dir.exists():
        return
    for child in list(out_dir.iterdir()):
        if child.is_dir():
            shutil.rmtree(child)
            continue
        child.unlink()


def _normal_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * np.square(z)) / math.sqrt(2.0 * math.pi)


def _normal_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


def _rbf_kernel(x1: np.ndarray, x2: np.ndarray, *, length_scale: float) -> np.ndarray:
    diff = x1[:, None, :] - x2[None, :, :]
    sqdist = np.sum(np.square(diff), axis=-1)
    scale = max(float(length_scale), 1e-6)
    return np.exp(-0.5 * sqdist / float(scale * scale))


def _expected_improvement(
    *,
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    x_cand: np.ndarray,
    length_scale: float,
    exploration: float,
) -> np.ndarray:
    if x_obs.shape[0] < 2:
        return np.ones((x_cand.shape[0],), dtype=np.float64)
    y_mean = float(np.mean(y_obs))
    y_std = float(np.std(y_obs))
    y_scale = max(y_std, 1e-8)
    y_norm = (y_obs - y_mean) / y_scale
    noise = 1e-6
    kernel_xx = _rbf_kernel(x_obs, x_obs, length_scale=float(length_scale))
    kernel_xx = kernel_xx + noise * np.eye(kernel_xx.shape[0], dtype=np.float64)
    try:
        chol = np.linalg.cholesky(kernel_xx)
    except np.linalg.LinAlgError:
        return np.ones((x_cand.shape[0],), dtype=np.float64)
    kernel_xs = _rbf_kernel(x_obs, x_cand, length_scale=float(length_scale))
    alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y_norm))
    mu_norm = kernel_xs.T @ alpha
    solved = np.linalg.solve(chol, kernel_xs)
    var_norm = np.maximum(1.0 - np.sum(np.square(solved), axis=0), 1e-9)
    mu = y_mean + y_scale * mu_norm
    sigma = y_scale * np.sqrt(var_norm)
    best = float(np.max(y_obs))
    improvement = mu - best - float(exploration)
    z = improvement / np.maximum(sigma, 1e-9)
    ei = improvement * _normal_cdf(z) + sigma * _normal_pdf(z)
    ei[sigma <= 1e-9] = 0.0
    return ei.astype(np.float64)


@dataclass(frozen=True)
class SearchResult:
    best_params: JobTraceParams
    best_target: float
    best_raw_vector: list[float]
    best_eval_id: int
    export_dir: str


@dataclass(frozen=True)
class _EvalSummary:
    eval_id: int
    target: float
    params: JobTraceParams
    raw_vector: np.ndarray
    trace_rows: list[dict[str, Any]]


class BOObjective:
    def __init__(
        self,
        *,
        scorer: ParkTraceScorer,
        objective: ObjectiveName,
        search_space: JobTraceSearchSpace,
        seed: int,
        traces_per_eval: int,
        log_path: Path,
    ) -> None:
        self.scorer = scorer
        self.objective = str(objective)
        self.search_space = search_space
        self.seed = int(seed)
        self.traces_per_eval = int(traces_per_eval)
        self.log_path = log_path
        self.eval_id = 0

    def evaluate_vector(self, vector: np.ndarray) -> _EvalSummary:
        eval_id = int(self.eval_id)
        self.eval_id += 1
        params = self.search_space.vector_to_params(np.asarray(vector, dtype=np.float64))
        trace_rows: list[dict[str, Any]] = []
        targets: list[float] = []
        for trace_idx in range(int(self.traces_per_eval)):
            trace_seed = int(self.seed + eval_id * 10_000 + trace_idx)
            rng = np.random.default_rng(int(trace_seed))
            trace = generate_job_trace(
                params=params,
                rng=rng,
                trace_seed=int(trace_seed),
            )
            score = self.scorer.score_trace(
                trace,
                objective=cast(ObjectiveName, self.objective),
                trace_id=f"eval_{int(eval_id):04d}_trace_{int(trace_idx):02d}",
                trace_path="",
                source="genet_search",
            )
            targets.append(float(score.objective_value))
            trace_rows.append(
                {
                    "trace_idx": int(trace_idx),
                    "trace_seed": int(trace_seed),
                    "objective_value": float(score.objective_value),
                    "gap_total": float(score.gap_total),
                    "gap_per_job": float(score.gap_per_job),
                    "regret_mean": float(score.regret_mean),
                    "regret_p95": float(score.regret_p95),
                    "dominant_reference_policy": str(score.dominant_reference_policy),
                    "reference_policy_histogram": dict(score.reference_policy_histogram),
                }
            )
        target = float(np.mean(np.asarray(targets, dtype=np.float64)))
        payload = {
            "eval_id": int(eval_id),
            "target": float(target),
            "objective": str(self.objective),
            "params": asdict(params),
            "raw_vector": [float(x) for x in np.asarray(vector, dtype=np.float64).tolist()],
            "traces_per_eval": int(self.traces_per_eval),
            "trace_rows": list(trace_rows),
        }
        _append_jsonl(self.log_path, payload)
        return _EvalSummary(
            eval_id=int(eval_id),
            target=float(target),
            params=params,
            raw_vector=np.asarray(vector, dtype=np.float64).copy(),
            trace_rows=trace_rows,
        )


class GenetBOTraceGenerator:
    def __init__(
        self,
        *,
        scorer: ParkTraceScorer,
        objective: ObjectiveName,
        search_space: JobTraceSearchSpace,
        seed: int,
        export_traces: int,
        out_dir: str,
        traces_per_eval: int = 5,
        init_points: int = 8,
        search_iters: int = 12,
        candidate_pool_size: int = 1024,
        gp_length_scale: float = 0.35,
        gp_exploration: float = 0.01,
        overwrite: bool = False,
    ) -> None:
        self.scorer = scorer
        self.objective = str(objective)
        self.search_space = search_space
        self.seed = int(seed)
        self.export_traces = int(export_traces)
        self.out_dir = Path(out_dir)
        self.traces_per_eval = int(traces_per_eval)
        self.init_points = int(init_points)
        self.search_iters = int(search_iters)
        self.candidate_pool_size = int(candidate_pool_size)
        self.gp_length_scale = float(gp_length_scale)
        self.gp_exploration = float(gp_exploration)
        self.overwrite = bool(overwrite)
        if int(self.export_traces) <= 0:
            raise ValueError("export_traces must be positive")
        if int(self.traces_per_eval) <= 0:
            raise ValueError("traces_per_eval must be positive")

    def _prepare_output_dir(self) -> None:
        if self.out_dir.exists():
            if not bool(self.overwrite) and any(self.out_dir.iterdir()):
                raise RuntimeError(f"output dir is not empty: {self.out_dir}")
            if bool(self.overwrite):
                _cleanup_output_dir(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _candidate_pool(self, *, rng: np.random.Generator, incumbent: np.ndarray | None) -> np.ndarray:
        pool = rng.uniform(
            0.0,
            1.0,
            size=(max(int(self.candidate_pool_size), 1), len(self.search_space._specs())),
        ).astype(np.float64)
        if incumbent is not None:
            local_count = max(1, int(self.candidate_pool_size // 4))
            noise = rng.normal(loc=0.0, scale=0.1, size=(int(local_count), incumbent.shape[0])).astype(np.float64)
            local_pool = np.clip(incumbent[None, :] + noise, 0.0, 1.0)
            pool[: int(local_count)] = local_pool
        return pool

    def _pick_next_vector(
        self,
        *,
        rng: np.random.Generator,
        observed_x: list[np.ndarray],
        observed_y: list[float],
    ) -> np.ndarray:
        incumbent = None
        if len(observed_x) > 0:
            incumbent = np.asarray(observed_x[int(np.argmax(np.asarray(observed_y, dtype=np.float64)))], dtype=np.float64)
        pool = self._candidate_pool(rng=rng, incumbent=incumbent)
        if len(observed_x) < 2:
            return np.asarray(pool[0], dtype=np.float64)
        x_obs = np.asarray(observed_x, dtype=np.float64)
        y_obs = np.asarray(observed_y, dtype=np.float64)
        acquisition = _expected_improvement(
            x_obs=x_obs,
            y_obs=y_obs,
            x_cand=pool,
            length_scale=float(self.gp_length_scale),
            exploration=float(self.gp_exploration),
        )
        order = np.argsort(-acquisition)
        for idx in list(order):
            cand = np.asarray(pool[int(idx)], dtype=np.float64)
            if np.min(np.linalg.norm(x_obs - cand[None, :], axis=1)) > 1e-6:
                return cand
        return np.asarray(pool[int(order[0])], dtype=np.float64)

    def _export_trace(self, *, trace_id: str, trace: GeneratedJobTrace) -> dict[str, Any]:
        csv_path = self.out_dir / f"{trace_id}.csv"
        json_path = self.out_dir / f"{trace_id}.json"
        write_generated_trace_csv(csv_path, trace)
        write_generated_trace_json(json_path, trace_id=str(trace_id), trace=trace)
        payload = trace_metadata(trace_id=str(trace_id), trace=trace)
        payload["csv_path"] = str(csv_path)
        payload["json_path"] = str(json_path)
        return payload

    def run(self) -> SearchResult:
        self._prepare_output_dir()
        rng = np.random.default_rng(int(self.seed))
        objective = BOObjective(
            scorer=self.scorer,
            objective=cast(ObjectiveName, self.objective),
            search_space=self.search_space,
            seed=int(self.seed),
            traces_per_eval=int(self.traces_per_eval),
            log_path=self.out_dir / "bo_log.jsonl",
        )
        observed_x: list[np.ndarray] = []
        observed_y: list[float] = []
        best_summary: _EvalSummary | None = None

        for _ in range(int(self.init_points)):
            vector = self.search_space.sample_vector(rng=rng)
            summary = objective.evaluate_vector(vector)
            observed_x.append(np.asarray(summary.raw_vector, dtype=np.float64))
            observed_y.append(float(summary.target))
            if best_summary is None or float(summary.target) > float(best_summary.target):
                best_summary = summary

        for _ in range(int(self.search_iters)):
            vector = self._pick_next_vector(rng=rng, observed_x=observed_x, observed_y=observed_y)
            summary = objective.evaluate_vector(vector)
            observed_x.append(np.asarray(summary.raw_vector, dtype=np.float64))
            observed_y.append(float(summary.target))
            if best_summary is None or float(summary.target) > float(best_summary.target):
                best_summary = summary

        if best_summary is None:
            raise RuntimeError("Genet search did not produce any evaluations")

        manifest_rows: list[dict[str, Any]] = []
        for ep in range(int(self.export_traces)):
            trace_seed = int(self.seed + 1_000_000 + ep)
            trace_rng = np.random.default_rng(int(trace_seed))
            trace = generate_job_trace(
                params=best_summary.params,
                rng=trace_rng,
                trace_seed=int(trace_seed),
            )
            manifest_rows.append(self._export_trace(trace_id=f"episode_{int(ep):04d}", trace=trace))

        _write_json(
            self.out_dir / "best_config.json",
            {
                "best_target": float(best_summary.target),
                "best_eval_id": int(best_summary.eval_id),
                "objective": str(self.objective),
                "search_space": self.search_space.to_dict(),
                "params": asdict(best_summary.params),
                "raw_vector": [float(x) for x in best_summary.raw_vector.tolist()],
                "trace_rows": list(best_summary.trace_rows),
            },
        )
        _write_json(
            self.out_dir / "manifest.json",
            {
                "objective": str(self.objective),
                "best_target": float(best_summary.target),
                "best_eval_id": int(best_summary.eval_id),
                "export_trace_count": int(len(manifest_rows)),
                "search_space": self.search_space.to_dict(),
                "exported_traces": list(manifest_rows),
            },
        )
        return SearchResult(
            best_params=best_summary.params,
            best_target=float(best_summary.target),
            best_raw_vector=[float(x) for x in best_summary.raw_vector.tolist()],
            best_eval_id=int(best_summary.eval_id),
            export_dir=str(self.out_dir),
        )
