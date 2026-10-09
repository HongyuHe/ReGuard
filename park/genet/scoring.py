from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Literal

import numpy as np

from scripts.lb_oracle_regret_utils import evaluate_jobs_with_oracle_regret

from .runtime import ParkRuntime
from .trace_family import GeneratedJobTrace


ObjectiveName = Literal[
    "reference_gap_total",
    "reference_gap_per_job",
    "regret_mean",
    "regret_p95",
]


@dataclass(frozen=True)
class TraceScore:
    objective_name: str
    objective_value: float
    baseline_name: str
    victim_total_reward: float
    reference_total_reward: float
    gap_total: float
    gap_per_job: float
    regret_mean: float
    regret_p95: float
    num_jobs: int
    dominant_reference_policy: str
    reference_policy_histogram: dict[str, int]


def _objective_value(*, objective: str, episode_row: dict[str, float | int | str]) -> float:
    if str(objective) == "reference_gap_total":
        return float(episode_row["oracle_gap_total"])
    if str(objective) == "reference_gap_per_job":
        return float(episode_row["oracle_gap_per_job"])
    if str(objective) == "regret_mean":
        return float(episode_row["regret_mean"])
    if str(objective) == "regret_p95":
        return float(episode_row["regret_p95"])
    raise ValueError(f"unsupported objective: {objective}")


class ParkTraceScorer:
    def __init__(self, runtime: ParkRuntime) -> None:
        self.runtime = runtime

    def score_jobs(
        self,
        jobs: list[dict[str, int]],
        *,
        objective: ObjectiveName,
        trace_id: str,
        trace_path: str,
        source: str = "genet",
    ) -> TraceScore:
        episode_row, step_rows = evaluate_jobs_with_oracle_regret(
            jobs=[{"arrival_time": int(job["arrival_time"]), "job_size": int(job["job_size"])} for job in list(jobs)],
            victim=self.runtime.victim,
            service_rates=list(self.runtime.service_rates),
            obs_high=float(self.runtime.obs_high),
            reward_time_scale=float(self.runtime.reward_time_scale),
            oracle_k=int(self.runtime.reference_window),
            trace_id=str(trace_id),
            trace_path=str(trace_path),
            source=str(source),
            shield=self.runtime.shield,
        )
        policy_counter: Counter[str] = Counter()
        for row in list(step_rows):
            policy_name = str(row.get("oracle_trace_policy_name", ""))
            if policy_name:
                policy_counter[str(policy_name)] += 1
        dominant_reference_policy = ""
        if policy_counter:
            dominant_reference_policy = max(
                policy_counter.items(),
                key=lambda item: (int(item[1]), str(item[0])),
            )[0]
        return TraceScore(
            objective_name=str(objective),
            objective_value=float(_objective_value(objective=str(objective), episode_row=episode_row)),
            baseline_name="best_of_heuristics",
            victim_total_reward=float(episode_row["victim_total_reward"]),
            reference_total_reward=float(episode_row["oracle_total_reward"]),
            gap_total=float(episode_row["oracle_gap_total"]),
            gap_per_job=float(episode_row["oracle_gap_per_job"]),
            regret_mean=float(episode_row["regret_mean"]),
            regret_p95=float(episode_row["regret_p95"]),
            num_jobs=int(episode_row["num_jobs"]),
            dominant_reference_policy=str(dominant_reference_policy),
            reference_policy_histogram={str(k): int(v) for k, v in sorted(policy_counter.items())},
        )

    def score_trace(
        self,
        trace: GeneratedJobTrace,
        *,
        objective: ObjectiveName,
        trace_id: str,
        trace_path: str = "",
        source: str = "genet",
    ) -> TraceScore:
        return self.score_jobs(
            trace.plain_jobs(),
            objective=objective,
            trace_id=str(trace_id),
            trace_path=str(trace_path),
            source=str(source),
        )

    def score_trace_dir(
        self,
        *,
        trace_paths: list[str],
        objective: ObjectiveName,
    ) -> dict[str, float]:
        scores: list[TraceScore] = []
        for idx, path in enumerate(list(trace_paths)):
            from online.sim_core import parse_job_trace

            jobs = parse_job_trace(str(path))
            scores.append(
                self.score_jobs(
                    jobs,
                    objective=objective,
                    trace_id=f"trace_{int(idx):04d}",
                    trace_path=str(path),
                    source="genet_eval",
                )
            )
        if len(scores) == 0:
            return {
                "count": 0.0,
                "objective_mean": float("nan"),
                "objective_std": float("nan"),
                "gap_total_mean": float("nan"),
                "gap_per_job_mean": float("nan"),
            }
        objective_values = np.asarray([float(score.objective_value) for score in list(scores)], dtype=np.float64)
        gap_total_values = np.asarray([float(score.gap_total) for score in list(scores)], dtype=np.float64)
        gap_per_job_values = np.asarray([float(score.gap_per_job) for score in list(scores)], dtype=np.float64)
        return {
            "count": float(int(len(scores))),
            "objective_mean": float(np.mean(objective_values)),
            "objective_std": float(np.std(objective_values)),
            "gap_total_mean": float(np.mean(gap_total_values)),
            "gap_per_job_mean": float(np.mean(gap_per_job_values)),
        }
