from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from online.sim_core import LoadBalanceSystem, best_of_heuristics_reference_reward, parse_job_trace
from surrogate.common import (
    collect_trace_files,
    controls_to_jobs,
    jobs_to_control_sequence,
    jobs_to_future_arrivals,
    log01_to_range,
    range_to_log01,
)
from surrogate.features import extract_system_summary
from surrogate.model import ParkSurrogateRollout, TorchVictimPolicy
from scripts.lb_oracle_regret_utils import rollout_policy_k_step_reward


@dataclass(frozen=True)
class AttackConfig:
    window_k: int
    stride: int
    iters: int
    restarts: int
    alpha: float
    dt_min: float
    dt_max: float
    size_min: float
    size_max: float
    dt_eps_abs: float
    dt_eps_rel: float
    size_eps_abs: float
    size_eps_rel: float
    reward_weight: float
    backlog_weight: float
    active_weight: float


def _clamp_actual_with_budget(
    *,
    values: np.ndarray,
    base: np.ndarray,
    global_min: float,
    global_max: float,
    eps_abs: float,
    eps_rel: float,
) -> np.ndarray:
    low = np.full_like(values, float(global_min), dtype=np.float32)
    high = np.full_like(values, float(global_max), dtype=np.float32)
    if float(eps_abs) > 0.0:
        low = np.maximum(low, base - float(eps_abs))
        high = np.minimum(high, base + float(eps_abs))
    if float(eps_rel) > 0.0:
        low = np.maximum(low, base * float(max(0.0, 1.0 - float(eps_rel))))
        high = np.minimum(high, base * float(1.0 + float(eps_rel)))
    return np.clip(values, low, high)


def _control01_to_actual(
    control01: torch.Tensor,
    *,
    dt_min: float,
    dt_max: float,
    size_min: float,
    size_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    dt01 = torch.clamp(control01[..., 0], 0.0, 1.0)
    size01 = torch.clamp(control01[..., 1], 0.0, 1.0)
    dt_actual = torch.exp(torch.log(torch.tensor(float(dt_min), device=control01.device)) + dt01 * (torch.log(torch.tensor(float(dt_max), device=control01.device)) - torch.log(torch.tensor(float(dt_min), device=control01.device))))
    size_actual = torch.exp(torch.log(torch.tensor(float(size_min), device=control01.device)) + size01 * (torch.log(torch.tensor(float(size_max), device=control01.device)) - torch.log(torch.tensor(float(size_min), device=control01.device))))
    return dt_actual, size_actual


def _project_control01(
    *,
    control01: torch.Tensor,
    base_dt: np.ndarray,
    base_size: np.ndarray,
    config: AttackConfig,
) -> torch.Tensor:
    with torch.no_grad():
        dt_actual, size_actual = _control01_to_actual(
            control01,
            dt_min=float(config.dt_min),
            dt_max=float(config.dt_max),
            size_min=float(config.size_min),
            size_max=float(config.size_max),
        )
        dt_np = dt_actual.detach().cpu().numpy().astype(np.float32, copy=False)
        size_np = size_actual.detach().cpu().numpy().astype(np.float32, copy=False)
        dt_proj = _clamp_actual_with_budget(
            values=dt_np,
            base=np.asarray(base_dt, dtype=np.float32),
            global_min=float(config.dt_min),
            global_max=float(config.dt_max),
            eps_abs=float(config.dt_eps_abs),
            eps_rel=float(config.dt_eps_rel),
        )
        size_proj = _clamp_actual_with_budget(
            values=size_np,
            base=np.asarray(base_size, dtype=np.float32),
            global_min=float(config.size_min),
            global_max=float(config.size_max),
            eps_abs=float(config.size_eps_abs),
            eps_rel=float(config.size_eps_rel),
        )
        out = np.zeros_like(dt_np, dtype=np.float32)
        control_proj = np.zeros((*dt_proj.shape, 2), dtype=np.float32)
        for idx in np.ndindex(dt_proj.shape):
            control_proj[idx + (0,)] = float(
                range_to_log01(
                    float(dt_proj[idx]),
                    float(config.dt_min),
                    float(config.dt_max),
                )
            )
            control_proj[idx + (1,)] = float(
                range_to_log01(
                    float(size_proj[idx]),
                    float(config.size_min),
                    float(config.size_max),
                )
            )
        del out
    return torch.as_tensor(control_proj, dtype=control01.dtype, device=control01.device)


def _warmup_trace_prefix(
    *,
    jobs: list[dict[str, int]],
    start_step: int,
    victim: TorchVictimPolicy,
    service_rates: list[float],
    obs_high: float,
    reward_time_scale: float,
) -> tuple[LoadBalanceSystem, np.ndarray, int]:
    system = LoadBalanceSystem(
        service_rates=list(service_rates),
        obs_high=float(obs_high),
        reward_time_scale=float(reward_time_scale),
    )
    obs = system.reset(
        first_job_size=int(jobs[0]["job_size"]),
        first_arrival_time=int(jobs[0]["arrival_time"]),
    )
    for step_idx in range(int(start_step)):
        action = int(victim.act(obs))
        next_dt = int(jobs[int(step_idx + 1)]["arrival_time"]) - int(jobs[int(step_idx)]["arrival_time"])
        next_size = int(jobs[int(step_idx + 1)]["job_size"])
        obs, _reward, done, _info = system.dispatch(
            int(action),
            next_arrival_dt=max(0, int(next_dt)),
            next_job_size=int(next_size),
        )
        if bool(done):
            break
    current_action = int(victim.act(obs))
    return system.clone(), np.asarray(obs, dtype=np.float32).copy(), int(current_action)


def _window_score_exact(
    *,
    snapshot: LoadBalanceSystem,
    obs_before: np.ndarray,
    start_action: int,
    future_arrivals: list[tuple[int | None, int | None]],
    victim: TorchVictimPolicy,
) -> tuple[float, dict[str, Any]]:
    oracle_result = best_of_heuristics_reference_reward(
        snapshot=snapshot.clone(),
        future_arrivals=list(future_arrivals),
    )
    policy_reward = rollout_policy_k_step_reward(
        snapshot=snapshot.clone(),
        start_obs=np.asarray(obs_before, dtype=np.float32).copy(),
        start_action=int(start_action),
        future_arrivals=list(future_arrivals),
        victim=victim,
    )
    gap = float(oracle_result.reward - float(policy_reward))
    return gap, {
        "oracle_reward": float(oracle_result.reward),
        "policy_reward": float(policy_reward),
        "oracle_policy_name": str(oracle_result.policy_name),
        "oracle_first_action": int(oracle_result.actions[0]) if len(oracle_result.actions) > 0 else -1,
    }


def attack_trace_jobs(
    *,
    jobs: list[dict[str, int]],
    victim: TorchVictimPolicy,
    rollout: ParkSurrogateRollout,
    service_rates: list[float],
    obs_high: float,
    reward_time_scale: float,
    config: AttackConfig,
    device: str = "cpu",
) -> tuple[list[dict[str, int]], dict[str, Any]]:
    if len(jobs) < 2:
        return list(jobs), {"windows": [], "num_jobs": int(len(jobs))}
    dt_all, size_all = jobs_to_control_sequence(jobs)
    windows_meta = []
    first_job_size = int(jobs[0]["job_size"])
    for start_step in range(0, len(dt_all), int(max(1, config.stride))):
        horizon = int(min(int(config.window_k), int(len(dt_all) - start_step)))
        if horizon <= 0:
            break
        current_jobs = controls_to_jobs(
            first_job_size=int(first_job_size),
            dt_values=dt_all,
            size_values=size_all,
            first_arrival_time=int(jobs[0]["arrival_time"]),
        )
        snapshot, obs_before, start_action = _warmup_trace_prefix(
            jobs=current_jobs,
            start_step=int(start_step),
            victim=victim,
            service_rates=list(service_rates),
            obs_high=float(obs_high),
            reward_time_scale=float(reward_time_scale),
        )
        base_dt = np.asarray(dt_all[int(start_step) : int(start_step) + int(horizon)], dtype=np.float32)
        base_size = np.asarray(size_all[int(start_step) : int(start_step) + int(horizon)], dtype=np.float32)
        control_init = np.zeros((1, horizon, 2), dtype=np.float32)
        for idx in range(int(horizon)):
            control_init[0, idx, 0] = float(
                range_to_log01(float(base_dt[idx]), float(config.dt_min), float(config.dt_max))
            )
            control_init[0, idx, 1] = float(
                range_to_log01(float(base_size[idx]), float(config.size_min), float(config.size_max))
            )
        initial_summary = torch.as_tensor(
            extract_system_summary(snapshot),
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        best_gap = None
        best_dt = None
        best_size = None
        best_info: dict[str, Any] = {}
        for restart_idx in range(int(config.restarts)):
            control01 = torch.as_tensor(control_init, dtype=torch.float32, device=device)
            if int(restart_idx) > 0:
                noise = 0.1 * torch.randn_like(control01)
                control01 = torch.clamp(control01 + noise, 0.0, 1.0)
                control01 = _project_control01(
                    control01=control01,
                    base_dt=base_dt.reshape(1, -1),
                    base_size=base_size.reshape(1, -1),
                    config=config,
                )
            control01 = control01.detach().clone().requires_grad_(True)
            restart_best_gap = None
            restart_best_dt = None
            restart_best_size = None
            restart_best_info: dict[str, Any] = {}
            for iter_idx in range(int(config.iters)):
                control01.data = _project_control01(
                    control01=control01.data,
                    base_dt=base_dt.reshape(1, -1),
                    base_size=base_size.reshape(1, -1),
                    config=config,
                )
                total_reward, step_info = rollout(
                    initial_summary,
                    control01,
                )
                proxy = (
                    float(config.reward_weight) * (-total_reward.mean())
                    + float(config.backlog_weight) * step_info["backlog_total"].mean()
                    + float(config.active_weight) * step_info["active_jobs_total"].mean()
                )
                if control01.grad is not None:
                    control01.grad.zero_()
                proxy.backward()
                with torch.no_grad():
                    grad = control01.grad
                    if grad is not None:
                        control01.add_(float(config.alpha) * grad.sign())
                    control01.copy_(
                        _project_control01(
                            control01=control01,
                            base_dt=base_dt.reshape(1, -1),
                            base_size=base_size.reshape(1, -1),
                            config=config,
                        )
                    )
                    dt_actual = np.zeros((horizon,), dtype=np.float32)
                    size_actual = np.zeros((horizon,), dtype=np.float32)
                    for idx in range(int(horizon)):
                        dt_actual[idx] = float(
                            log01_to_range(
                                float(control01[0, idx, 0].detach().cpu().item()),
                                float(config.dt_min),
                                float(config.dt_max),
                            )
                        )
                        size_actual[idx] = float(
                            log01_to_range(
                                float(control01[0, idx, 1].detach().cpu().item()),
                                float(config.size_min),
                                float(config.size_max),
                            )
                        )
                    future_window = [
                        (
                            int(max(0, round(float(dt_actual[idx])))),
                            int(max(1, round(float(size_actual[idx])))),
                        )
                        for idx in range(int(horizon))
                    ]
                    gap, info = _window_score_exact(
                        snapshot=snapshot,
                        obs_before=obs_before,
                        start_action=int(start_action),
                        future_arrivals=future_window,
                        victim=victim,
                    )
                    info["iter_idx"] = int(iter_idx)
                    info["restart_idx"] = int(restart_idx)
                    if restart_best_gap is None or float(gap) > float(restart_best_gap):
                        restart_best_gap = float(gap)
                        restart_best_dt = dt_actual.copy()
                        restart_best_size = size_actual.copy()
                        restart_best_info = dict(info)
            if restart_best_gap is not None and (best_gap is None or float(restart_best_gap) > float(best_gap)):
                best_gap = float(restart_best_gap)
                best_dt = restart_best_dt
                best_size = restart_best_size
                best_info = dict(restart_best_info)
        if best_dt is not None and best_size is not None:
            dt_all[int(start_step) : int(start_step) + int(horizon)] = np.asarray(best_dt, dtype=np.float32)
            size_all[int(start_step) : int(start_step) + int(horizon)] = np.asarray(best_size, dtype=np.float32)
        windows_meta.append(
            {
                "start_step": int(start_step),
                "horizon": int(horizon),
                "best_gap": None if best_gap is None else float(best_gap),
                "best_info": dict(best_info),
            }
        )
    adv_jobs = controls_to_jobs(
        first_job_size=int(first_job_size),
        dt_values=dt_all,
        size_values=size_all,
        first_arrival_time=int(jobs[0]["arrival_time"]),
    )
    meta = {
        "num_jobs": int(len(adv_jobs)),
        "windows": list(windows_meta),
        "attack_config": {
            "window_k": int(config.window_k),
            "stride": int(config.stride),
            "iters": int(config.iters),
            "restarts": int(config.restarts),
            "alpha": float(config.alpha),
            "dt_min": float(config.dt_min),
            "dt_max": float(config.dt_max),
            "size_min": float(config.size_min),
            "size_max": float(config.size_max),
            "dt_eps_abs": float(config.dt_eps_abs),
            "dt_eps_rel": float(config.dt_eps_rel),
            "size_eps_abs": float(config.size_eps_abs),
            "size_eps_rel": float(config.size_eps_rel),
            "reward_weight": float(config.reward_weight),
            "backlog_weight": float(config.backlog_weight),
            "active_weight": float(config.active_weight),
        },
    }
    return adv_jobs, meta
