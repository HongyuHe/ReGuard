from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from surrogate.features import (
    active_jobs_total_from_summary_tensor,
    backlog_total_from_summary_tensor,
    project_summary_tensor,
    summary_feature_dim,
    summary_to_victim_obs_tensor,
)
from train_ppo import ActorCritic


@dataclass(frozen=True)
class SurrogateCheckpointConfig:
    num_servers: int
    hidden: int
    dt_min: float
    dt_max: float
    size_min: float
    size_max: float
    reward_mean: float
    reward_std: float
    obs_high: float
    reward_time_scale: float
    service_rates: list[float]


class TorchVictimPolicy(nn.Module):
    def __init__(self, checkpoint_path: str, *, num_servers: int | None = None, device: str = "cpu") -> None:
        super().__init__()
        ckpt = torch.load(checkpoint_path, map_location=device)
        cfg = dict(ckpt.get("config", {}))
        self.config = cfg
        self.num_servers = int(num_servers if num_servers is not None else cfg.get("num_servers", 10))
        hidden = int(cfg.get("hidden", 128))
        self.model = ActorCritic(
            obs_dim=int(self.num_servers + 1),
            act_dim=int(self.num_servers),
            hidden=hidden,
        ).to(device)
        state_dict = ckpt.get("model_state_dict", ckpt.get("model"))
        if state_dict is None:
            raise RuntimeError(f"checkpoint missing model weights: {checkpoint_path}")
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.device = str(device)

    @torch.inference_mode()
    def act(self, obs) -> int:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, _value = self.model(obs_t)
        return int(torch.argmax(logits, dim=-1).item())

    @torch.inference_mode()
    def predict_probs(self, obs) -> torch.Tensor:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, _value = self.model(obs_t)
        return torch.softmax(logits, dim=-1).squeeze(0)

    def predict_probs_tensor(self, obs_t: torch.Tensor) -> torch.Tensor:
        logits, _value = self.model(obs_t)
        return torch.softmax(logits, dim=-1)


class LearnedParkStepSurrogate(nn.Module):
    def __init__(self, *, num_servers: int, hidden: int = 256) -> None:
        super().__init__()
        self.num_servers = int(num_servers)
        self.hidden = int(hidden)
        self.summary_dim = int(summary_feature_dim(self.num_servers))
        input_dim = int(self.summary_dim + self.num_servers + 2)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
        )
        self.summary_head = nn.Linear(self.hidden, self.summary_dim)
        self.reward_head = nn.Linear(self.hidden, 1)

    def forward(
        self,
        summary: torch.Tensor,
        action_probs: torch.Tensor,
        control01: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([summary, action_probs, control01], dim=-1)
        h = self.encoder(x)
        residual = torch.tanh(self.summary_head(h))
        next_summary = project_summary_tensor(summary + 0.25 * residual, num_servers=self.num_servers)
        reward_norm = self.reward_head(h).squeeze(-1)
        return next_summary, reward_norm


class ParkSurrogateRollout(nn.Module):
    def __init__(
        self,
        *,
        victim_policy: TorchVictimPolicy,
        step_model: LearnedParkStepSurrogate,
        reward_mean: float,
        reward_std: float,
    ) -> None:
        super().__init__()
        self.victim_policy = victim_policy
        self.step_model = step_model
        self.reward_mean = float(reward_mean)
        self.reward_std = float(max(reward_std, 1e-6))
        self.num_servers = int(step_model.num_servers)

    def forward(
        self,
        initial_summary: torch.Tensor,
        controls01: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if controls01.ndim != 3 or controls01.shape[-1] != 2:
            raise ValueError("controls01 must have shape [batch, horizon, 2]")
        summary = initial_summary
        batch = int(summary.shape[0])
        total_reward = summary.new_zeros((batch,))
        reward_list = []
        backlog_list = []
        active_list = []
        probs_list = []
        for step_idx in range(int(controls01.shape[1])):
            obs_t = summary_to_victim_obs_tensor(summary, num_servers=self.num_servers)
            probs = self.victim_policy.predict_probs_tensor(obs_t)
            next_summary, reward_norm = self.step_model(summary, probs, controls01[:, step_idx, :])
            reward = reward_norm * float(self.reward_std) + float(self.reward_mean)
            total_reward = total_reward + reward
            reward_list.append(reward)
            probs_list.append(probs)
            backlog_list.append(backlog_total_from_summary_tensor(next_summary, num_servers=self.num_servers))
            active_list.append(active_jobs_total_from_summary_tensor(next_summary, num_servers=self.num_servers))
            summary = next_summary
        step_info = {
            "reward": torch.stack(reward_list, dim=1) if len(reward_list) > 0 else summary.new_zeros((batch, 0)),
            "backlog_total": torch.stack(backlog_list, dim=1)
            if len(backlog_list) > 0
            else summary.new_zeros((batch, 0)),
            "active_jobs_total": torch.stack(active_list, dim=1)
            if len(active_list) > 0
            else summary.new_zeros((batch, 0)),
            "action_probs": torch.stack(probs_list, dim=1)
            if len(probs_list) > 0
            else summary.new_zeros((batch, 0, self.num_servers)),
            "final_summary": summary,
        }
        return total_reward, step_info


def build_surrogate_rollout_from_checkpoint(
    *,
    checkpoint_path: str,
    victim_policy: TorchVictimPolicy,
    device: str = "cpu",
) -> tuple[ParkSurrogateRollout, SurrogateCheckpointConfig, dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg_raw = dict(ckpt.get("surrogate_config", {}))
    config = SurrogateCheckpointConfig(
        num_servers=int(cfg_raw.get("num_servers", victim_policy.num_servers)),
        hidden=int(cfg_raw.get("hidden", 256)),
        dt_min=float(cfg_raw.get("dt_min", 1.0)),
        dt_max=float(cfg_raw.get("dt_max", 500.0)),
        size_min=float(cfg_raw.get("size_min", 10.0)),
        size_max=float(cfg_raw.get("size_max", 5000.0)),
        reward_mean=float(cfg_raw.get("reward_mean", 0.0)),
        reward_std=float(cfg_raw.get("reward_std", 1.0)),
        obs_high=float(cfg_raw.get("obs_high", 500000.0)),
        reward_time_scale=float(cfg_raw.get("reward_time_scale", 1000.0)),
        service_rates=[float(x) for x in cfg_raw.get("service_rates", [])],
    )
    model = LearnedParkStepSurrogate(
        num_servers=int(config.num_servers),
        hidden=int(config.hidden),
    ).to(device)
    state_dict = ckpt.get("model_state_dict", ckpt.get("model"))
    if state_dict is None:
        raise RuntimeError(f"surrogate checkpoint missing model weights: {checkpoint_path}")
    model.load_state_dict(state_dict)
    model.eval()
    rollout = ParkSurrogateRollout(
        victim_policy=victim_policy,
        step_model=model,
        reward_mean=float(config.reward_mean),
        reward_std=float(config.reward_std),
    ).to(device)
    rollout.eval()
    return rollout, config, ckpt
