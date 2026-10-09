from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from scripts.lb_oracle_regret_utils import VictimActor, parse_service_rates


def resolve_repo_path(path: str) -> str:
    expanded = os.path.expandvars(os.path.expanduser(str(path)))
    if os.path.isabs(expanded):
        return expanded
    repo_root = Path(__file__).resolve().parent.parent
    return str((repo_root / expanded).resolve())


@dataclass(frozen=True)
class ParkRuntime:
    checkpoint_path: str
    checkpoint_config: dict[str, Any]
    victim: VictimActor
    num_servers: int
    service_rates: list[float]
    obs_high: float
    reward_time_scale: float
    reference_window: int
    shield: Any | None = None


def load_runtime(
    model_path: str,
    *,
    num_servers: int | None = None,
    service_rates: str | list[float] | None = None,
    obs_high: float | None = None,
    reward_time_scale: float | None = None,
    reference_window: int = 5,
    shield_rules_prefix: str | None = None,
    shield_risk_trigger_steps: int = 1,
    shield_cooldown_steps: int = 0,
    shield_empty_fallback: str = "least_load_proxy",
) -> ParkRuntime:
    checkpoint_path = resolve_repo_path(str(model_path))
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    cfg = dict(ckpt.get("config", {}))
    resolved_num_servers = int(num_servers if num_servers is not None else cfg.get("num_servers", 10))
    if isinstance(service_rates, str) or service_rates is None:
        resolved_rates = parse_service_rates(
            service_rates,
            num_servers=int(resolved_num_servers),
            checkpoint_config=cfg,
        )
    else:
        if len(service_rates) != int(resolved_num_servers):
            raise ValueError("service_rates must match num_servers")
        resolved_rates = [float(x) for x in list(service_rates)]
    resolved_obs_high = float(obs_high if obs_high is not None else cfg.get("obs_high", 500000.0))
    resolved_reward_time_scale = float(
        reward_time_scale if reward_time_scale is not None else cfg.get("reward_time_scale", 1000.0)
    )
    victim = VictimActor(str(checkpoint_path), num_servers=int(resolved_num_servers))
    shield = None
    if shield_rules_prefix is not None:
        from shield.shield_runtime import load_shield_from_prefix

        shield = load_shield_from_prefix(
            resolve_repo_path(str(shield_rules_prefix)),
            num_servers=int(resolved_num_servers),
            risk_trigger_steps=int(shield_risk_trigger_steps),
            intervention_cooldown_steps=int(shield_cooldown_steps),
            empty_fallback=str(shield_empty_fallback),
        )
    return ParkRuntime(
        checkpoint_path=str(checkpoint_path),
        checkpoint_config=cfg,
        victim=victim,
        num_servers=int(resolved_num_servers),
        service_rates=list(resolved_rates),
        obs_high=float(resolved_obs_high),
        reward_time_scale=float(resolved_reward_time_scale),
        reference_window=max(1, int(reference_window)),
        shield=shield,
    )
