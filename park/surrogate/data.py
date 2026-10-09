from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from online.sim_core import LoadBalanceSystem, best_of_heuristics_reference_reward, parse_job_trace
from surrogate.common import jobs_to_future_arrivals
from surrogate.features import extract_system_summary
from surrogate.model import TorchVictimPolicy
from scripts.lb_oracle_regret_utils import rollout_policy_k_step_reward


def _control_to_log01(value: float, *, vmin: float, vmax: float) -> float:
    lo = float(max(vmin, 1e-9))
    hi = float(max(vmax, lo + 1e-9))
    x = float(np.clip(float(value), lo, hi))
    return float(np.clip((np.log(x) - np.log(lo)) / (np.log(hi) - np.log(lo)), 0.0, 1.0))


def collect_transition_dataset(
    *,
    trace_files: list[Path],
    victim_checkpoint: str,
    service_rates: list[float],
    num_servers: int,
    obs_high: float,
    reward_time_scale: float,
    oracle_k: int,
    dt_min: float,
    dt_max: float,
    size_min: float,
    size_max: float,
    max_steps_per_trace: int | None = None,
    device: str = "cpu",
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    victim = TorchVictimPolicy(
        victim_checkpoint,
        num_servers=int(num_servers),
        device=str(device),
    )
    summary_rows = []
    action_prob_rows = []
    control_rows = []
    reward_rows = []
    next_summary_rows = []
    oracle_gap_rows = []
    step_rows = []
    trace_rows = []
    total_steps = 0
    for trace_path in list(trace_files):
        jobs = parse_job_trace(str(trace_path))
        if len(jobs) < 2:
            continue
        future_arrivals = jobs_to_future_arrivals(jobs)
        system = LoadBalanceSystem(
            service_rates=list(service_rates),
            obs_high=float(obs_high),
            reward_time_scale=float(reward_time_scale),
        )
        obs = system.reset(
            first_job_size=int(jobs[0]["job_size"]),
            first_arrival_time=int(jobs[0]["arrival_time"]),
        )
        step_limit = int(len(jobs) - 1)
        if max_steps_per_trace is not None:
            step_limit = int(min(step_limit, int(max_steps_per_trace)))
        for step_idx in range(step_limit):
            snapshot = system.clone()
            summary = extract_system_summary(system)
            obs_before = np.asarray(obs, dtype=np.float32).copy()
            probs = victim.predict_probs(obs_before).detach().cpu().numpy().astype(np.float32, copy=False)
            action = int(torch.argmax(torch.as_tensor(probs), dim=-1).item())
            next_dt, next_size = future_arrivals[step_idx]
            if next_dt is None or next_size is None:
                break
            control01 = np.asarray(
                [
                    _control_to_log01(float(next_dt), vmin=float(dt_min), vmax=float(dt_max)),
                    _control_to_log01(float(next_size), vmin=float(size_min), vmax=float(size_max)),
                ],
                dtype=np.float32,
            )
            obs, reward, done, _info = system.dispatch(
                int(action),
                next_arrival_dt=int(next_dt),
                next_job_size=int(next_size),
            )
            next_summary = extract_system_summary(system)
            horizon = int(min(int(oracle_k), int(len(jobs) - 1 - step_idx)))
            if horizon > 0:
                future_window = future_arrivals[int(step_idx) : int(step_idx) + int(horizon)]
                oracle_result = best_of_heuristics_reference_reward(
                    snapshot=snapshot,
                    future_arrivals=future_window,
                )
                policy_reward = rollout_policy_k_step_reward(
                    snapshot=snapshot,
                    start_obs=obs_before,
                    start_action=int(action),
                    future_arrivals=future_window,
                    victim=victim,
                )
                oracle_gap = float(oracle_result.reward - float(policy_reward))
            else:
                oracle_gap = 0.0
            summary_rows.append(summary)
            action_prob_rows.append(probs)
            control_rows.append(control01)
            reward_rows.append(float(reward))
            next_summary_rows.append(next_summary)
            oracle_gap_rows.append(float(oracle_gap))
            step_rows.append(int(step_idx))
            trace_rows.append(str(trace_path))
            total_steps += 1
            if bool(done):
                break
    if len(summary_rows) == 0:
        raise RuntimeError("no transition samples collected")
    arrays = {
        "summary": np.asarray(summary_rows, dtype=np.float32),
        "action_probs": np.asarray(action_prob_rows, dtype=np.float32),
        "control01": np.asarray(control_rows, dtype=np.float32),
        "reward": np.asarray(reward_rows, dtype=np.float32),
        "next_summary": np.asarray(next_summary_rows, dtype=np.float32),
        "oracle_gap": np.asarray(oracle_gap_rows, dtype=np.float32),
        "step_idx": np.asarray(step_rows, dtype=np.int64),
        "trace_name": np.asarray(trace_rows).astype(str),
    }
    meta = {
        "num_samples": int(total_steps),
        "num_servers": int(num_servers),
        "obs_high": float(obs_high),
        "reward_time_scale": float(reward_time_scale),
        "oracle_k": int(oracle_k),
        "dt_min": float(dt_min),
        "dt_max": float(dt_max),
        "size_min": float(size_min),
        "size_max": float(size_max),
        "service_rates": [float(x) for x in list(service_rates)],
        "victim_checkpoint": str(victim_checkpoint),
        "trace_files": [str(x) for x in list(trace_files)],
    }
    return arrays, meta


def save_transition_dataset(
    *,
    out_npz: Path,
    arrays: dict[str, np.ndarray],
    meta: dict[str, Any],
) -> None:
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        **arrays,
        meta_json=np.asarray([json.dumps(meta)], dtype=object),
    )


def load_transition_dataset(path: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=True) as data:
        arrays = {
            "summary": np.asarray(data["summary"], dtype=np.float32),
            "action_probs": np.asarray(data["action_probs"], dtype=np.float32),
            "control01": np.asarray(data["control01"], dtype=np.float32),
            "reward": np.asarray(data["reward"], dtype=np.float32),
            "next_summary": np.asarray(data["next_summary"], dtype=np.float32),
            "oracle_gap": np.asarray(data["oracle_gap"], dtype=np.float32),
            "step_idx": np.asarray(data["step_idx"], dtype=np.int64),
            "trace_name": np.asarray(data["trace_name"]).astype(str),
        }
        meta_json = data["meta_json"]
        meta = json.loads(str(np.asarray(meta_json).reshape(-1)[0]))
    return arrays, meta


class TransitionDataset(torch.utils.data.Dataset):
    def __init__(self, arrays: dict[str, np.ndarray]) -> None:
        self.summary = torch.from_numpy(np.asarray(arrays["summary"], dtype=np.float32))
        self.action_probs = torch.from_numpy(np.asarray(arrays["action_probs"], dtype=np.float32))
        self.control01 = torch.from_numpy(np.asarray(arrays["control01"], dtype=np.float32))
        self.reward = torch.from_numpy(np.asarray(arrays["reward"], dtype=np.float32))
        self.next_summary = torch.from_numpy(np.asarray(arrays["next_summary"], dtype=np.float32))

    def __len__(self) -> int:
        return int(self.summary.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "summary": self.summary[int(idx)],
            "action_probs": self.action_probs[int(idx)],
            "control01": self.control01[int(idx)],
            "reward": self.reward[int(idx)],
            "next_summary": self.next_summary[int(idx)],
        }
