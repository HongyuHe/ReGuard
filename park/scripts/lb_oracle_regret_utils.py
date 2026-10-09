from __future__ import annotations

import csv
import json
import os
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
    parse_job_trace,
    reference_policy_names,
    select_reference_action,
)
from train_ppo import ActorCritic


def repo_root() -> Path:
    return ROOT_DIR


def resolve_path(path: str) -> Path:
    expanded = Path(os.path.expandvars(os.path.expanduser(str(path))))
    if expanded.is_absolute():
        return expanded
    return (repo_root() / expanded).resolve()


def mean(values: list[float]) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def percentile(values: list[float], q: float) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), float(q)))


def default_service_rates(num_servers: int) -> list[float]:
    if int(num_servers) == 10:
        return [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05]
    return np.linspace(0.15, 1.05, num=int(num_servers), dtype=np.float64).tolist()


def parse_service_rates(
    text: str | None,
    *,
    num_servers: int,
    checkpoint_config: dict | None = None,
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
    json_files = [p for p in sorted(trace_dir.glob("*.json")) if p.name.lower() != "meta.json" and p.stem not in csv_stems]
    candidates = sorted(list(csv_files) + list(json_files))
    valid_files: list[Path] = []
    for path in list(candidates):
        try:
            jobs = parse_job_trace(str(path))
        except Exception:
            continue
        if len(jobs) <= 0:
            continue
        valid_files.append(path)
    return sorted(valid_files)


def try_import_wandb() -> Any | None:
    try:
        import wandb  # type: ignore
    except Exception:
        return None
    return wandb


def init_wandb_run(
    *,
    enabled: bool,
    project: str,
    name: str | None,
    config: dict[str, Any],
    entity: str | None = None,
    mode: str = "online",
    tags: list[str] | None = None,
):
    if not bool(enabled):
        return None
    wandb = try_import_wandb()
    if wandb is None:
        raise RuntimeError("wandb logging requested but wandb is not installed")
    run = wandb.init(
        project=str(project),
        entity=entity,
        name=name,
        mode=str(mode),
        config=dict(config),
        tags=list(tags) if tags is not None else None,
    )
    return run


def normalize_wandb_tags(tags: list[str] | None) -> list[str]:
    if tags is None:
        return []
    out: list[str] = []
    for tag in list(tags):
        for part in str(tag).split(","):
            item = str(part).strip()
            if item:
                out.append(item)
    return out


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


def write_trace_json(path: Path, jobs: list[dict[str, int]], *, kind: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": str(kind),
        "num_jobs": int(len(jobs)),
        "jobs": [{"arrival_time": int(job["arrival_time"]), "job_size": int(job["job_size"])} for job in list(jobs)],
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


class VictimActor:
    def __init__(self, checkpoint_path: str, num_servers: int | None = None) -> None:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        cfg = dict(ckpt.get("config", {}))
        self.config = cfg
        self.num_servers = int(num_servers if num_servers is not None else cfg.get("num_servers", 10))
        hidden = int(cfg.get("hidden", 128))
        self.model = ActorCritic(obs_dim=int(self.num_servers + 1), act_dim=int(self.num_servers), hidden=hidden)
        state_dict = ckpt.get("model_state_dict", ckpt.get("model"))
        if state_dict is None:
            raise RuntimeError(f"checkpoint missing model weights: {checkpoint_path}")
        self.model.load_state_dict(state_dict)
        self.model.eval()

    @torch.inference_mode()
    def logits(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits, _value = self.model(obs_t)
        return logits.detach().cpu().numpy().reshape(-1).astype(np.float64)

    @torch.inference_mode()
    def act(self, obs: np.ndarray) -> int:
        logits = self.logits(obs)
        return int(np.argmax(logits))


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


def rollout_policy_k_step_reward(
    *,
    snapshot: LoadBalanceSystem,
    start_obs: np.ndarray,
    start_action: int,
    future_arrivals: list[tuple[int | None, int | None]],
    victim: VictimActor,
    shield: Any | None = None,
) -> float:
    system = snapshot.clone()
    obs = np.asarray(start_obs, dtype=np.float32).copy()
    total_reward = 0.0
    shield_snapshot = None
    if shield is not None and hasattr(shield, "snapshot_runtime_state"):
        shield_snapshot = shield.snapshot_runtime_state()
    try:
        for depth, (next_dt, next_size) in enumerate(list(future_arrivals)):
            if int(depth) == 0:
                action = int(start_action)
            elif shield is None:
                action = int(victim.act(obs))
            else:
                logits = victim.logits(obs)
                base_action = int(np.argmax(logits))
                action, _shield_stats = shield.select_action(
                    obs=obs,
                    scores=logits,
                    base_action=int(base_action),
                    action_selection="argmax",
                )
            obs, reward, done, _info = system.dispatch(
                int(action),
                next_arrival_dt=next_dt,
                next_job_size=next_size,
            )
            total_reward += float(reward)
            if bool(done):
                break
    finally:
        if shield is not None and hasattr(shield, "restore_runtime_state"):
            shield.restore_runtime_state(shield_snapshot)
    return float(total_reward)


def oracle_episode_rollout_receding(
    *,
    jobs: list[dict[str, int]],
    service_rates: list[float],
    obs_high: float,
    reward_time_scale: float,
    oracle_k: int,
) -> tuple[float, list[int], list[str]]:
    jobs_sorted = sorted(list(jobs), key=lambda x: int(x["arrival_time"]))
    future_arrivals = jobs_to_future_arrivals(jobs_sorted)
    oracle_sys = LoadBalanceSystem(
        service_rates=list(service_rates),
        obs_high=float(obs_high),
        reward_time_scale=float(reward_time_scale),
    )
    oracle_sys.reset(
        first_job_size=int(jobs_sorted[0]["job_size"]),
        first_arrival_time=int(jobs_sorted[0]["arrival_time"]),
    )

    total_reward = 0.0
    actions: list[int] = []
    policy_names: list[str] = []
    n = len(jobs_sorted)
    for i in range(int(n)):
        horizon = int(min(int(oracle_k), int(n) - int(i)))
        oracle_result = best_of_heuristics_reference_reward(
            snapshot=oracle_sys.clone(),
            future_arrivals=future_arrivals[int(i) : int(i) + int(horizon)],
        )
        action = int(oracle_result.actions[0])
        _obs, reward, _done, _info = oracle_sys.dispatch(
            int(action),
            next_arrival_dt=future_arrivals[int(i)][0],
            next_job_size=future_arrivals[int(i)][1],
        )
        total_reward += float(reward)
        actions.append(int(action))
        policy_names.append(str(oracle_result.policy_name))
    return float(total_reward), actions, policy_names


def collect_system_state(system: LoadBalanceSystem, *, prefix: str, suffix: str) -> dict[str, float | int]:
    obs = system.observe()
    out: dict[str, float | int] = {}
    total_backlog_proxy = 0.0
    total_queue_jobs = 0
    total_running_jobs = 0
    total_queue_size = 0
    for server_idx, server in enumerate(list(system.servers)):
        load_proxy = float(obs[int(server_idx)])
        queue_len = int(len(server.queue))
        running = int(server.curr_job is not None)
        queue_size = int(sum(int(job.size) for job in list(server.queue)))
        out[f"{prefix}/server_load_proxy_{server_idx}_{suffix}"] = float(load_proxy)
        out[f"{prefix}/queue_len_{server_idx}_{suffix}"] = int(queue_len)
        out[f"{prefix}/running_{server_idx}_{suffix}"] = int(running)
        out[f"{prefix}/queue_size_bytes_{server_idx}_{suffix}"] = int(queue_size)
        total_backlog_proxy += float(load_proxy)
        total_queue_jobs += int(queue_len)
        total_running_jobs += int(running)
        total_queue_size += int(queue_size)
    out[f"{prefix}/incoming_size_{suffix}"] = float(obs[-1]) if len(obs) > 0 else 0.0
    out[f"{prefix}/backlog_proxy_total_{suffix}"] = float(total_backlog_proxy)
    out[f"{prefix}/queue_jobs_total_{suffix}"] = int(total_queue_jobs)
    out[f"{prefix}/running_jobs_total_{suffix}"] = int(total_running_jobs)
    out[f"{prefix}/active_jobs_total_{suffix}"] = int(total_queue_jobs + total_running_jobs)
    out[f"{prefix}/queue_size_bytes_total_{suffix}"] = int(total_queue_size)
    return out


def evaluate_jobs_with_oracle_regret(
    *,
    jobs: list[dict[str, int]],
    victim: VictimActor,
    service_rates: list[float],
    obs_high: float,
    reward_time_scale: float,
    oracle_k: int,
    trace_id: str,
    trace_path: str,
    source: str,
    shield: Any | None = None,
) -> tuple[dict[str, float | int | str], list[dict[str, float | int | str]]]:
    if len(jobs) == 0:
        raise ValueError("jobs must be non-empty")

    jobs_sorted = sorted(list(jobs), key=lambda x: int(x["arrival_time"]))
    future_arrivals = jobs_to_future_arrivals(jobs_sorted)
    _oracle_total_reward, oracle_trace_actions, oracle_trace_policy_names = oracle_episode_rollout_receding(
        jobs=jobs_sorted,
        service_rates=service_rates,
        obs_high=float(obs_high),
        reward_time_scale=float(reward_time_scale),
        oracle_k=int(oracle_k),
    )

    victim_sys = LoadBalanceSystem(
        service_rates=list(service_rates),
        obs_high=float(obs_high),
        reward_time_scale=float(reward_time_scale),
    )
    victim_obs = victim_sys.reset(
        first_job_size=int(jobs_sorted[0]["job_size"]),
        first_arrival_time=int(jobs_sorted[0]["arrival_time"]),
    )
    oracle_sys = LoadBalanceSystem(
        service_rates=list(service_rates),
        obs_high=float(obs_high),
        reward_time_scale=float(reward_time_scale),
    )
    oracle_obs = oracle_sys.reset(
        first_job_size=int(jobs_sorted[0]["job_size"]),
        first_arrival_time=int(jobs_sorted[0]["arrival_time"]),
    )
    baseline_policy_names = list(reference_policy_names())
    baseline_systems: dict[str, LoadBalanceSystem] = {}
    for policy_name in list(baseline_policy_names):
        baseline_system = LoadBalanceSystem(
            service_rates=list(service_rates),
            obs_high=float(obs_high),
            reward_time_scale=float(reward_time_scale),
        )
        baseline_system.reset(
            first_job_size=int(jobs_sorted[0]["job_size"]),
            first_arrival_time=int(jobs_sorted[0]["arrival_time"]),
        )
        baseline_systems[str(policy_name)] = baseline_system

    total_victim_reward = 0.0
    total_oracle_reward = 0.0
    baseline_total_rewards = {str(policy_name): 0.0 for policy_name in list(baseline_policy_names)}
    regret_values: list[float] = []
    model_time_ms_values: list[float] = []
    victim_backlog_values: list[float] = []
    oracle_backlog_values: list[float] = []
    victim_active_jobs_values: list[float] = []
    oracle_active_jobs_values: list[float] = []
    decision_diff_trace_values: list[float] = []
    decision_diff_counterfactual_values: list[float] = []
    baseline_backlog_values = {str(policy_name): [] for policy_name in list(baseline_policy_names)}
    baseline_active_jobs_values = {str(policy_name): [] for policy_name in list(baseline_policy_names)}
    baseline_decision_diff_values = {str(policy_name): [] for policy_name in list(baseline_policy_names)}
    shield_time_ms_values: list[float] = []
    shield_intervene_values: list[float] = []
    shield_action_changed_values: list[float] = []
    shield_risky_values: list[float] = []
    shield_allowed_count_values: list[float] = []
    shield_empty_allowed_values: list[float] = []
    shield_fallback_values: list[float] = []
    step_rows: list[dict[str, float | int | str]] = []
    if shield is not None and hasattr(shield, "reset_runtime_state"):
        shield.reset_runtime_state()

    n = len(jobs_sorted)
    for i in range(int(n)):
        snapshot = victim_sys.clone()
        obs_before = np.asarray(victim_obs, dtype=np.float32).copy()
        victim_before = collect_system_state(victim_sys, prefix="victim", suffix="before")
        oracle_before = collect_system_state(oracle_sys, prefix="oracle", suffix="before")
        baseline_before: dict[str, dict[str, float | int]] = {}
        baseline_actions: dict[str, int] = {}
        for policy_name in list(baseline_policy_names):
            baseline_system = baseline_systems[str(policy_name)]
            prefix = f"baseline/{policy_name}"
            baseline_before[str(policy_name)] = collect_system_state(
                baseline_system,
                prefix=prefix,
                suffix="before",
            )
            baseline_actions[str(policy_name)] = int(select_reference_action(baseline_system, str(policy_name)))
        model_start_ns = time.perf_counter_ns()
        victim_logits = victim.logits(obs_before)
        victim_base_action = int(np.argmax(victim_logits))
        model_time_ms = float(time.perf_counter_ns() - int(model_start_ns)) / 1_000_000.0
        model_time_ms_values.append(float(model_time_ms))
        shield_stats: dict[str, float | int] = {}
        victim_action = int(victim_base_action)
        if shield is not None:
            victim_action, shield_stats = shield.select_action(
                obs=obs_before,
                scores=victim_logits,
                base_action=int(victim_base_action),
                action_selection="argmax",
            )
            shield_time_ms_values.append(float(shield_stats.get("shield/time_ms", 0.0)))
            shield_intervene_values.append(float(shield_stats.get("shield/intervene", 0.0)))
            shield_action_changed_values.append(float(shield_stats.get("shield/action_changed", 0.0)))
            shield_risky_values.append(float(shield_stats.get("shield/risky_state", 0.0)))
            shield_allowed_count_values.append(float(shield_stats.get("shield/allowed_count", 0.0)))
            shield_empty_allowed_values.append(float(shield_stats.get("shield/empty_allowed", 0.0)))
            shield_fallback_values.append(float(shield_stats.get("shield/fallback_used", 0.0)))

        next_dt, next_size = future_arrivals[int(i)]
        step_start_time = int(victim_sys.curr_time)
        oracle_action = int(oracle_trace_actions[int(i)])
        victim_obs, victim_step_reward, _done, victim_info = victim_sys.dispatch(
            int(victim_action),
            next_arrival_dt=next_dt,
            next_job_size=next_size,
        )
        oracle_obs, oracle_step_reward, _oracle_done, oracle_info = oracle_sys.dispatch(
            int(oracle_action),
            next_arrival_dt=next_dt,
            next_job_size=next_size,
        )
        baseline_after: dict[str, dict[str, float | int]] = {}
        baseline_step_rewards: dict[str, float] = {}
        baseline_infos: dict[str, dict[str, float | int]] = {}
        for policy_name in list(baseline_policy_names):
            baseline_system = baseline_systems[str(policy_name)]
            _baseline_obs, baseline_step_reward, _baseline_done, baseline_info = baseline_system.dispatch(
                int(baseline_actions[str(policy_name)]),
                next_arrival_dt=next_dt,
                next_job_size=next_size,
            )
            baseline_step_rewards[str(policy_name)] = float(baseline_step_reward)
            baseline_infos[str(policy_name)] = dict(baseline_info)
            baseline_total_rewards[str(policy_name)] += float(baseline_step_reward)
            baseline_after[str(policy_name)] = collect_system_state(
                baseline_system,
                prefix=f"baseline/{policy_name}",
                suffix="after",
            )
        total_victim_reward += float(victim_step_reward)
        total_oracle_reward += float(oracle_step_reward)
        victim_after = collect_system_state(victim_sys, prefix="victim", suffix="after")
        oracle_after = collect_system_state(oracle_sys, prefix="oracle", suffix="after")

        horizon = int(min(int(oracle_k), int(n) - int(i)))
        oracle_future = future_arrivals[int(i) : int(i) + int(horizon)]
        oracle_result = best_of_heuristics_reference_reward(
            snapshot=snapshot,
            future_arrivals=oracle_future,
        )
        policy_k_reward = rollout_policy_k_step_reward(
            snapshot=snapshot,
            start_obs=obs_before,
            start_action=int(victim_action),
            future_arrivals=oracle_future,
            victim=victim,
            shield=shield,
        )
        regret = float(oracle_result.reward - float(policy_k_reward))
        regret_values.append(float(regret))
        victim_backlog_values.append(float(victim_before["victim/backlog_proxy_total_before"]))
        oracle_backlog_values.append(float(oracle_before["oracle/backlog_proxy_total_before"]))
        victim_active_jobs_values.append(float(victim_before["victim/active_jobs_total_before"]))
        oracle_active_jobs_values.append(float(oracle_before["oracle/active_jobs_total_before"]))
        decision_diff_trace = int(victim_action != oracle_action)
        decision_diff_counterfactual = int(victim_action != int(oracle_result.actions[0]))
        decision_diff_trace_values.append(float(decision_diff_trace))
        decision_diff_counterfactual_values.append(float(decision_diff_counterfactual))
        for policy_name in list(baseline_policy_names):
            baseline_before_row = baseline_before[str(policy_name)]
            baseline_backlog_values[str(policy_name)].append(
                float(baseline_before_row[f"baseline/{policy_name}/backlog_proxy_total_before"])
            )
            baseline_active_jobs_values[str(policy_name)].append(
                float(baseline_before_row[f"baseline/{policy_name}/active_jobs_total_before"])
            )
            baseline_decision_diff_values[str(policy_name)].append(
                float(int(victim_action != int(baseline_actions[str(policy_name)])))
            )

        row: dict[str, float | int | str] = {
            "source": str(source),
            "trace_id": str(trace_id),
            "trace_path": str(trace_path),
            "step_idx": int(i),
            "num_jobs": int(n),
            "oracle_k": int(oracle_k),
            "reference_window": int(oracle_k),
            "oracle_horizon": int(horizon),
            "arrival_time": int(jobs_sorted[int(i)]["arrival_time"]),
            "step_start_time": int(step_start_time),
            "victim_step_end_time": int(victim_info.get("curr_time", 0)),
            "oracle_step_end_time": int(oracle_info.get("curr_time", 0)),
            "job_size": int(jobs_sorted[int(i)]["job_size"]),
            "next_arrival_dt": None if next_dt is None else int(next_dt),
            "next_job_size": None if next_size is None else int(next_size),
            "victim_action": int(victim_action),
            "victim_base_action": int(victim_base_action),
            "oracle_trace_action": int(oracle_action),
            "oracle_best_first_action": int(oracle_result.actions[0]),
            "oracle_trace_policy_name": str(oracle_trace_policy_names[int(i)]),
            "oracle_counterfactual_policy_name": str(oracle_result.policy_name),
            "decision_diff_trace": int(decision_diff_trace),
            "decision_diff_counterfactual": int(decision_diff_counterfactual),
            "victim_step_reward": float(victim_step_reward),
            "oracle_step_reward": float(oracle_step_reward),
            "victim_finished_jobs": int(victim_info.get("num_finished_jobs", 0)),
            "oracle_finished_jobs": int(oracle_info.get("num_finished_jobs", 0)),
            "victim_cumulative_reward": float(total_victim_reward),
            "oracle_cumulative_reward": float(total_oracle_reward),
            "oracle_gap_cumulative": float(total_oracle_reward - total_victim_reward),
            "policy_k_reward": float(policy_k_reward),
            "oracle_k_reward": float(oracle_result.reward),
            "regret": float(regret),
            "model_time_ms": float(model_time_ms),
            "shielded": int(shield is not None),
            "shield_time_ms": float(shield_stats.get("shield/time_ms", 0.0)) if shield is not None else 0.0,
        }
        if shield is not None:
            for key, value in dict(shield_stats).items():
                row[str(key)] = value
        for policy_name in list(baseline_policy_names):
            row[f"baseline_action/{policy_name}"] = int(baseline_actions[str(policy_name)])
            row[f"baseline_step_reward/{policy_name}"] = float(baseline_step_rewards[str(policy_name)])
            row[f"baseline_finished_jobs/{policy_name}"] = int(
                baseline_infos[str(policy_name)].get("num_finished_jobs", 0)
            )
            row[f"baseline_step_end_time/{policy_name}"] = int(
                baseline_infos[str(policy_name)].get("curr_time", 0)
            )
            row[f"baseline_cumulative_reward/{policy_name}"] = float(baseline_total_rewards[str(policy_name)])
        row.update(victim_before)
        row.update(oracle_before)
        for policy_name in list(baseline_policy_names):
            row.update(baseline_before[str(policy_name)])
        row.update(victim_after)
        row.update(oracle_after)
        for policy_name in list(baseline_policy_names):
            row.update(baseline_after[str(policy_name)])
        step_rows.append(row)

    oracle_gap_total = float(total_oracle_reward - float(total_victim_reward))
    episode_row: dict[str, float | int | str] = {
        "source": str(source),
        "trace_id": str(trace_id),
        "trace_path": str(trace_path),
        "num_jobs": int(n),
        "oracle_k": int(oracle_k),
        "reference_window": int(oracle_k),
        "victim_total_reward": float(total_victim_reward),
        "oracle_total_reward": float(total_oracle_reward),
        "oracle_gap_total": float(oracle_gap_total),
        "victim_reward_per_job": float(total_victim_reward / max(1, int(n))),
        "oracle_reward_per_job": float(total_oracle_reward / max(1, int(n))),
        "oracle_gap_per_job": float(oracle_gap_total / max(1, int(n))),
        "regret_count": int(len(regret_values)),
        "regret_mean": float(mean(regret_values)),
        "regret_p50": float(percentile(regret_values, 50.0)),
        "regret_p95": float(percentile(regret_values, 95.0)),
        "regret_sum": float(sum(regret_values)),
        "victim_backlog_mean": float(mean(victim_backlog_values)),
        "oracle_backlog_mean": float(mean(oracle_backlog_values)),
        "victim_active_jobs_mean": float(mean(victim_active_jobs_values)),
        "oracle_active_jobs_mean": float(mean(oracle_active_jobs_values)),
        "decision_diff_trace_rate": float(mean(decision_diff_trace_values)),
        "decision_diff_counterfactual_rate": float(mean(decision_diff_counterfactual_values)),
        "model_time_ms_mean": float(mean(model_time_ms_values)),
        "model_time_ms_p50": float(percentile(model_time_ms_values, 50.0)),
        "model_time_ms_p95": float(percentile(model_time_ms_values, 95.0)),
        "shielded": int(shield is not None),
        "shield_time_ms_mean": float(mean(shield_time_ms_values)),
        "shield_time_ms_p50": float(percentile(shield_time_ms_values, 50.0)),
        "shield_time_ms_p95": float(percentile(shield_time_ms_values, 95.0)),
        "shield_intervention_rate": float(mean(shield_intervene_values)),
        "shield_action_changed_rate": float(mean(shield_action_changed_values)),
        "shield_risk_rate": float(mean(shield_risky_values)),
        "shield_allowed_count_mean": float(mean(shield_allowed_count_values)),
        "shield_empty_allowed_rate": float(mean(shield_empty_allowed_values)),
        "shield_fallback_rate": float(mean(shield_fallback_values)),
        "first_arrival_time": int(jobs_sorted[0]["arrival_time"]),
        "last_arrival_time": int(jobs_sorted[-1]["arrival_time"]),
        "victim_final_time": int(victim_sys.curr_time),
        "oracle_final_time": int(oracle_sys.curr_time),
    }
    for policy_name in list(baseline_policy_names):
        episode_row[f"baseline_total_reward/{policy_name}"] = float(baseline_total_rewards[str(policy_name)])
        episode_row[f"baseline_reward_per_job/{policy_name}"] = float(
            baseline_total_rewards[str(policy_name)] / max(1, int(n))
        )
        episode_row[f"baseline_backlog_mean/{policy_name}"] = float(mean(baseline_backlog_values[str(policy_name)]))
        episode_row[f"baseline_active_jobs_mean/{policy_name}"] = float(
            mean(baseline_active_jobs_values[str(policy_name)])
        )
        episode_row[f"decision_diff_rate/baseline/{policy_name}"] = float(
            mean(baseline_decision_diff_values[str(policy_name)])
        )
        episode_row[f"baseline_final_time/{policy_name}"] = int(baseline_systems[str(policy_name)].curr_time)
    return episode_row, step_rows


def write_metrics_csvs(
    *,
    out_dir: Path,
    episode_rows: list[dict[str, float | int | str]],
    step_rows: list[dict[str, float | int | str]],
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    episode_csv = out_dir / "episode_metrics.csv"
    step_csv = out_dir / "step_metrics.csv"

    episode_base_fieldnames = [
        "source",
        "trace_id",
        "trace_path",
        "num_jobs",
        "oracle_k",
        "reference_window",
        "victim_total_reward",
        "oracle_total_reward",
        "oracle_gap_total",
        "victim_reward_per_job",
        "oracle_reward_per_job",
        "oracle_gap_per_job",
        "regret_count",
        "regret_mean",
        "regret_p50",
        "regret_p95",
        "regret_sum",
        "model_time_ms_mean",
        "model_time_ms_p50",
        "model_time_ms_p95",
        "first_arrival_time",
        "last_arrival_time",
    ]
    step_base_fieldnames = [
        "source",
        "trace_id",
        "trace_path",
        "step_idx",
        "num_jobs",
        "oracle_k",
        "reference_window",
        "oracle_horizon",
        "arrival_time",
        "step_start_time",
        "step_end_time",
        "job_size",
        "next_arrival_dt",
        "next_job_size",
        "victim_action",
        "victim_step_reward",
        "victim_finished_jobs",
        "policy_k_reward",
        "oracle_k_reward",
        "regret",
        "oracle_best_first_action",
        "oracle_trace_action",
        "oracle_trace_policy_name",
        "oracle_counterfactual_policy_name",
        "model_time_ms",
    ]
    episode_fieldnames = list(episode_base_fieldnames)
    step_fieldnames = list(step_base_fieldnames)
    episode_extra = sorted({str(k) for row in list(episode_rows) for k in row.keys() if str(k) not in set(episode_base_fieldnames)})
    step_extra = sorted({str(k) for row in list(step_rows) for k in row.keys() if str(k) not in set(step_base_fieldnames)})
    episode_fieldnames.extend(list(episode_extra))
    step_fieldnames.extend(list(step_extra))

    with episode_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=episode_fieldnames)
        writer.writeheader()
        for row in list(episode_rows):
            writer.writerow({k: row.get(k, "") for k in episode_fieldnames})

    with step_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=step_fieldnames)
        writer.writeheader()
        for row in list(step_rows):
            writer.writerow({k: row.get(k, "") for k in step_fieldnames})

    return episode_csv, step_csv


def load_jobs_from_trace(path: Path) -> list[dict[str, int]]:
    return parse_job_trace(str(path))


def log_eval_to_wandb(
    run,
    *,
    episode_rows: list[dict[str, float | int | str]],
    step_rows: list[dict[str, float | int | str]],
    max_trace_plots: int = 5,
) -> None:
    if run is None:
        return
    wandb = try_import_wandb()
    if wandb is None:
        return

    summary_metrics: dict[str, float | int] = {
        "eval/num_episodes": int(len(episode_rows)),
        "eval/num_steps": int(len(step_rows)),
    }
    episode_metric_keys = [
        "victim_total_reward",
        "oracle_total_reward",
        "oracle_gap_total",
        "victim_reward_per_job",
        "oracle_reward_per_job",
        "oracle_gap_per_job",
        "regret_mean",
        "regret_p95",
        "regret_sum",
        "victim_backlog_mean",
        "oracle_backlog_mean",
        "victim_active_jobs_mean",
        "oracle_active_jobs_mean",
        "decision_diff_trace_rate",
        "decision_diff_counterfactual_rate",
        "model_time_ms_mean",
        "shield_time_ms_mean",
        "shield_intervention_rate",
        "shield_action_changed_rate",
        "shield_risk_rate",
        "shield_allowed_count_mean",
        "shield_empty_allowed_rate",
        "shield_fallback_rate",
    ]
    for key in list(episode_metric_keys):
        vals = [float(row[key]) for row in list(episode_rows) if key in row]
        if len(vals) > 0:
            summary_metrics[f"eval/{key}_mean"] = float(mean(vals))
            summary_metrics[f"eval/{key}_p95"] = float(percentile(vals, 95.0))
    dynamic_episode_prefixes = (
        "baseline_total_reward/",
        "baseline_reward_per_job/",
        "baseline_backlog_mean/",
        "baseline_active_jobs_mean/",
        "decision_diff_rate/baseline/",
        "baseline_final_time/",
    )
    dynamic_episode_keys = sorted(
        {
            str(key)
            for row in list(episode_rows)
            for key in row.keys()
            if any(str(key).startswith(str(prefix)) for prefix in list(dynamic_episode_prefixes))
        }
    )
    for key in list(dynamic_episode_keys):
        vals = [float(row[key]) for row in list(episode_rows) if key in row]
        if len(vals) > 0:
            summary_metrics[f"eval/{key}_mean"] = float(mean(vals))
            summary_metrics[f"eval/{key}_p95"] = float(percentile(vals, 95.0))
    run.log(summary_metrics)

    episode_columns = sorted({str(k) for row in list(episode_rows) for k in row.keys()})
    step_columns = sorted({str(k) for row in list(step_rows) for k in row.keys()})
    episode_table = wandb.Table(columns=episode_columns)
    step_table = wandb.Table(columns=step_columns)
    for row in list(episode_rows):
        episode_table.add_data(*[row.get(col, "") for col in list(episode_columns)])
    for row in list(step_rows):
        step_table.add_data(*[row.get(col, "") for col in list(step_columns)])
    run.log({"eval/episode_table": episode_table, "eval/step_table": step_table})

    trace_ids = sorted({str(row.get("trace_id", "")) for row in list(step_rows)})
    for trace_id in list(trace_ids)[: int(max_trace_plots)]:
        rows = [row for row in list(step_rows) if str(row.get("trace_id", "")) == str(trace_id)]
        rows.sort(key=lambda row: (int(row.get("step_idx", 0)), int(row.get("step_start_time", 0))))
        if len(rows) == 0:
            continue
        xs = [float(row.get("step_start_time", 0.0)) for row in list(rows)]
        baseline_action_keys = sorted(
            [str(key) for key in rows[0].keys() if str(key).startswith("baseline_action/")]
        )
        baseline_policy_names = [str(key).split("/", 1)[1] for key in list(baseline_action_keys)]
        decision_specs = [
            ("victim_action", "victim_action"),
            ("oracle_trace_action", "oracle_trace_action"),
            ("oracle_best_first_action", "oracle_best_first_action"),
        ]
        decision_specs.extend(
            [(f"baseline_action/{policy_name}", f"baseline_action/{policy_name}") for policy_name in list(baseline_policy_names)]
        )
        backlog_specs = [
            ("victim/backlog_proxy_total_before", "victim_backlog"),
            ("oracle/backlog_proxy_total_before", "oracle_backlog"),
        ]
        backlog_specs.extend(
            [
                (
                    f"baseline/{policy_name}/backlog_proxy_total_before",
                    f"baseline_backlog/{policy_name}",
                )
                for policy_name in list(baseline_policy_names)
            ]
        )
        active_specs = [
            ("victim/active_jobs_total_before", "victim_active_jobs"),
            ("oracle/active_jobs_total_before", "oracle_active_jobs"),
        ]
        active_specs.extend(
            [
                (
                    f"baseline/{policy_name}/active_jobs_total_before",
                    f"baseline_active_jobs/{policy_name}",
                )
                for policy_name in list(baseline_policy_names)
            ]
        )
        queue_specs = [
            ("victim/queue_jobs_total_before", "victim_queue_jobs"),
            ("oracle/queue_jobs_total_before", "oracle_queue_jobs"),
        ]
        queue_specs.extend(
            [
                (
                    f"baseline/{policy_name}/queue_jobs_total_before",
                    f"baseline_queue_jobs/{policy_name}",
                )
                for policy_name in list(baseline_policy_names)
            ]
        )
        cumulative_reward_specs = [
            ("victim_cumulative_reward", "victim_cumulative_reward"),
            ("oracle_cumulative_reward", "oracle_cumulative_reward"),
            ("oracle_gap_cumulative", "oracle_gap_cumulative"),
        ]
        cumulative_reward_specs.extend(
            [
                (
                    f"baseline_cumulative_reward/{policy_name}",
                    f"baseline_cumulative_reward/{policy_name}",
                )
                for policy_name in list(baseline_policy_names)
            ]
        )
        run.log(
            {
                f"trace/{trace_id}/decisions": wandb.plot.line_series(
                    xs=list(xs),
                    ys=[[float(row.get(key, 0.0)) for row in list(rows)] for key, _label in list(decision_specs)],
                    keys=[str(label) for _key, label in list(decision_specs)],
                    title=f"{trace_id} decisions",
                    xname="simulation_time",
                ),
                f"trace/{trace_id}/backlog": wandb.plot.line_series(
                    xs=list(xs),
                    ys=[[float(row.get(key, 0.0)) for row in list(rows)] for key, _label in list(backlog_specs)],
                    keys=[str(label) for _key, label in list(backlog_specs)],
                    title=f"{trace_id} backlog",
                    xname="simulation_time",
                ),
                f"trace/{trace_id}/active_jobs": wandb.plot.line_series(
                    xs=list(xs),
                    ys=[[float(row.get(key, 0.0)) for row in list(rows)] for key, _label in list(active_specs)],
                    keys=[str(label) for _key, label in list(active_specs)],
                    title=f"{trace_id} active jobs",
                    xname="simulation_time",
                ),
                f"trace/{trace_id}/queue_jobs": wandb.plot.line_series(
                    xs=list(xs),
                    ys=[[float(row.get(key, 0.0)) for row in list(rows)] for key, _label in list(queue_specs)],
                    keys=[str(label) for _key, label in list(queue_specs)],
                    title=f"{trace_id} queue jobs",
                    xname="simulation_time",
                ),
                f"trace/{trace_id}/cumulative_reward": wandb.plot.line_series(
                    xs=list(xs),
                    ys=[
                        [float(row.get(key, 0.0)) for row in list(rows)]
                        for key, _label in list(cumulative_reward_specs)
                    ],
                    keys=[str(label) for _key, label in list(cumulative_reward_specs)],
                    title=f"{trace_id} cumulative reward",
                    xname="simulation_time",
                ),
            }
        )
        entity_specs = [("victim", "victim"), ("oracle", "oracle")]
        entity_specs.extend(
            [
                (f"baseline/{policy_name}", f"baseline_{policy_name}")
                for policy_name in list(baseline_policy_names)
            ]
        )
        entity_payload: dict[str, Any] = {}
        for prefix, label in list(entity_specs):
            server_keys = sorted(
                [
                    str(key)
                    for key in rows[0].keys()
                    if str(key).startswith(f"{prefix}/server_load_proxy_") and str(key).endswith("_before")
                ]
            )
            if len(server_keys) > 0:
                entity_payload[f"trace/{trace_id}/{label}_server_loads"] = wandb.plot.line_series(
                    xs=list(xs),
                    ys=[[float(row.get(key, 0.0)) for row in list(rows)] for key in list(server_keys)],
                    keys=[
                        str(key).replace(f"{prefix}/", "").replace("_before", "")
                        for key in list(server_keys)
                    ],
                    title=f"{trace_id} {label} server loads",
                    xname="simulation_time",
                )
            queue_keys = sorted(
                [
                    str(key)
                    for key in rows[0].keys()
                    if str(key).startswith(f"{prefix}/queue_len_") and str(key).endswith("_before")
                ]
            )
            if len(queue_keys) > 0:
                entity_payload[f"trace/{trace_id}/{label}_queue_lengths"] = wandb.plot.line_series(
                    xs=list(xs),
                    ys=[[float(row.get(key, 0.0)) for row in list(rows)] for key in list(queue_keys)],
                    keys=[
                        str(key).replace(f"{prefix}/", "").replace("_before", "")
                        for key in list(queue_keys)
                    ],
                    title=f"{trace_id} {label} queue lengths",
                    xname="simulation_time",
                )
        if len(entity_payload) > 0:
            run.log(entity_payload)

    for key, value in list(summary_metrics.items()):
        run.summary[str(key)] = value
