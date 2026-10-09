#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from online.sim_core import (
    LoadBalanceSystem,
    best_of_heuristics_reference_reward,
    reference_policy_names,
    rollout_reference_policy_reward,
)
from scripts.lb_oracle_regret_utils import (
    VictimActor,
    collect_trace_files,
    jobs_to_future_arrivals,
    load_jobs_from_trace,
    parse_service_rates,
    resolve_path,
    rollout_policy_k_step_reward,
)
from shield.obs_features import feature_columns, feature_row_from_obs


def _policy_logits(victim: VictimActor, obs: np.ndarray) -> np.ndarray:
    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.inference_mode():
        logits, _value = victim.model(obs_t)
    return logits.detach().cpu().numpy().reshape(-1).astype(np.float64)


def _forced_first_reference_q(
    *,
    snapshot: LoadBalanceSystem,
    action: int,
    future_arrivals: list[tuple[int | None, int | None]],
) -> float:
    if len(future_arrivals) == 0:
        return 0.0
    system = snapshot.clone()
    next_dt, next_size = future_arrivals[0]
    _obs, reward, done, _info = system.dispatch(
        int(action),
        next_arrival_dt=next_dt,
        next_job_size=next_size,
    )
    total = float(reward)
    if bool(done) or len(future_arrivals) <= 1:
        return float(total)
    ref = best_of_heuristics_reference_reward(
        snapshot=system.clone(),
        future_arrivals=list(future_arrivals[1:]),
    )
    return float(total + float(ref.reward))


def _episode_rows(
    *,
    jobs: list[dict[str, int]],
    trace_path: Path,
    trace_idx: int,
    source: str,
    victim: VictimActor,
    service_rates: list[float],
    obs_high: float,
    reward_time_scale: float,
    reference_window: int,
    include_derived: bool,
    max_rows: int = 0,
) -> list[dict[str, Any]]:
    jobs_sorted = sorted(list(jobs), key=lambda x: int(x["arrival_time"]))
    if len(jobs_sorted) == 0:
        return []
    future_arrivals = jobs_to_future_arrivals(jobs_sorted)
    num_servers = int(victim.num_servers)
    system = LoadBalanceSystem(
        service_rates=list(service_rates),
        obs_high=float(obs_high),
        reward_time_scale=float(reward_time_scale),
    )
    obs = system.reset(
        first_job_size=int(jobs_sorted[0]["job_size"]),
        first_arrival_time=int(jobs_sorted[0]["arrival_time"]),
    )
    rows: list[dict[str, Any]] = []
    policies = list(reference_policy_names())
    for step_idx in range(len(jobs_sorted)):
        obs_before = np.asarray(obs, dtype=np.float32).copy()
        snapshot = system.clone()
        horizon = int(min(int(reference_window), int(len(jobs_sorted)) - int(step_idx)))
        future_window = list(future_arrivals[int(step_idx) : int(step_idx) + int(horizon)])
        logits = _policy_logits(victim, obs_before)
        policy_action = int(np.argmax(logits))
        policy_reward = rollout_policy_k_step_reward(
            snapshot=snapshot,
            start_obs=obs_before,
            start_action=int(policy_action),
            future_arrivals=list(future_window),
            victim=victim,
        )
        reference_results = {
            str(name): rollout_reference_policy_reward(
                snapshot=snapshot.clone(),
                future_arrivals=list(future_window),
                policy_name=str(name),
            )
            for name in list(policies)
        }
        best_reference = max(
            reference_results.values(),
            key=lambda result: (float(result.reward), tuple(-int(a) for a in result.actions)),
        )
        q_by_action = [
            _forced_first_reference_q(
                snapshot=snapshot,
                action=int(action),
                future_arrivals=list(future_window),
            )
            for action in range(int(num_servers))
        ]
        gap = float(best_reference.reward - float(policy_reward))
        denom = max(1e-9, abs(float(best_reference.reward)))
        row: dict[str, Any] = {
            "source": str(source),
            "trace_name": str(trace_path.name),
            "trace_path": str(trace_path),
            "trace_idx": int(trace_idx),
            "step_idx": int(step_idx),
            "num_jobs": int(len(jobs_sorted)),
            "arrival_time": int(jobs_sorted[int(step_idx)]["arrival_time"]),
            "job_size": int(jobs_sorted[int(step_idx)]["job_size"]),
            "reference_window": int(reference_window),
            "reference_horizon": int(horizon),
            "policy_action": int(policy_action),
            "policy_reward": float(policy_reward),
            "best_reference_policy": str(best_reference.policy_name),
            "best_reference_action": int(best_reference.actions[0]) if len(best_reference.actions) > 0 else -1,
            "best_reference_reward": float(best_reference.reward),
            "reference_gap": float(gap),
            "reference_gap_pct": float(100.0 * float(gap) / float(denom)),
        }
        for action in range(int(num_servers)):
            row[f"policy_logit_{action}"] = float(logits[int(action)])
            row[f"reference_q_action_{action}"] = float(q_by_action[int(action)])
        for name, result in sorted(reference_results.items()):
            row[f"reference_action/{name}"] = int(result.actions[0]) if len(result.actions) > 0 else -1
            row[f"reference_reward/{name}"] = float(result.reward)
        row.update(
            feature_row_from_obs(
                obs_before,
                num_servers=int(num_servers),
                include_derived=bool(include_derived),
            )
        )

        rows.append(row)
        if int(max_rows) > 0 and len(rows) >= int(max_rows):
            break
        next_dt, next_size = future_arrivals[int(step_idx)]
        obs, _reward, done, _info = system.dispatch(
            int(policy_action),
            next_arrival_dt=next_dt,
            next_job_size=next_size,
        )
        if bool(done):
            break
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an observation-only Park LB shield rule-mining dataset.")
    parser.add_argument("--victim-checkpoint", "--model", dest="victim_checkpoint", type=str, required=True)
    parser.add_argument("--trace-dir", action="append", required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--reference-window", "--oracle-k", dest="reference_window", type=int, default=5)
    parser.add_argument("--num-servers", type=int, default=None)
    parser.add_argument("--service-rates", type=str, default=None)
    parser.add_argument("--obs-high", type=float, default=None)
    parser.add_argument("--reward-time-scale", type=float, default=None)
    parser.add_argument("--trace-limit", type=int, default=0)
    parser.add_argument("--max-logged-states", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--raw-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.reference_window) <= 0:
        raise ValueError("--reference-window must be positive")
    out_path = resolve_path(str(args.out))
    if out_path.exists() and not bool(args.overwrite):
        raise RuntimeError(f"output exists: {out_path}. Re-run with --overwrite.")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    victim = VictimActor(checkpoint_path=str(resolve_path(str(args.victim_checkpoint))), num_servers=args.num_servers)
    cfg = dict(victim.config)
    num_servers = int(victim.num_servers)
    service_rates = parse_service_rates(
        args.service_rates,
        num_servers=int(num_servers),
        checkpoint_config=cfg,
    )
    obs_high = float(args.obs_high if args.obs_high is not None else cfg.get("obs_high", 500000.0))
    reward_time_scale = float(
        args.reward_time_scale if args.reward_time_scale is not None else cfg.get("reward_time_scale", 1000.0)
    )
    include_derived = bool(not args.raw_only)

    all_rows: list[dict[str, Any]] = []
    trace_sources: list[tuple[str, Path]] = []
    for trace_dir_text in list(args.trace_dir):
        trace_dir = resolve_path(str(trace_dir_text))
        for path in collect_trace_files(trace_dir):
            trace_sources.append((str(trace_dir.name), path))
    if int(args.trace_limit) > 0:
        trace_sources = trace_sources[: int(args.trace_limit)]
    if len(trace_sources) == 0:
        raise RuntimeError("no valid traces found")

    started = time.perf_counter()
    for trace_idx, (source, trace_path) in enumerate(list(trace_sources)):
        jobs = load_jobs_from_trace(trace_path)
        rows = _episode_rows(
            jobs=list(jobs),
            trace_path=trace_path,
            trace_idx=int(trace_idx),
            source=str(source),
            victim=victim,
            service_rates=list(service_rates),
            obs_high=float(obs_high),
            reward_time_scale=float(reward_time_scale),
            reference_window=int(args.reference_window),
            include_derived=bool(include_derived),
            max_rows=0
            if int(args.max_logged_states) <= 0
            else max(0, int(args.max_logged_states) - int(len(all_rows))),
        )
        for row in list(rows):
            all_rows.append(row)
            if int(args.max_logged_states) > 0 and len(all_rows) >= int(args.max_logged_states):
                break
        if int(args.progress_every) > 0 and (int(trace_idx) + 1) % int(args.progress_every) == 0:
            elapsed = float(time.perf_counter() - started)
            print(
                f"progress traces={int(trace_idx) + 1} rows={len(all_rows)} elapsed_s={elapsed:.2f}",
                flush=True,
            )
        if int(args.max_logged_states) > 0 and len(all_rows) >= int(args.max_logged_states):
            break

    base_cols = [
        "source",
        "trace_name",
        "trace_path",
        "trace_idx",
        "step_idx",
        "num_jobs",
        "arrival_time",
        "job_size",
        "reference_window",
        "reference_horizon",
        "policy_action",
        "policy_reward",
        "best_reference_policy",
        "best_reference_action",
        "best_reference_reward",
        "reference_gap",
        "reference_gap_pct",
    ]
    policy_cols = [f"policy_logit_{i}" for i in range(int(num_servers))]
    q_cols = [f"reference_q_action_{i}" for i in range(int(num_servers))]
    reference_cols = []
    for name in list(reference_policy_names()):
        reference_cols.append(f"reference_action/{name}")
        reference_cols.append(f"reference_reward/{name}")
    fields = list(base_cols) + list(policy_cols) + list(q_cols) + list(reference_cols)
    fields.extend(feature_columns(int(num_servers), include_derived=bool(include_derived)))

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        for row in list(all_rows):
            writer.writerow({key: row.get(key, "") for key in list(fields)})
    elapsed = float(time.perf_counter() - started)
    print(f"wrote {len(all_rows)} rows to {out_path} from {len(trace_sources)} traces elapsed_s={elapsed:.2f}")


if __name__ == "__main__":
    main()
