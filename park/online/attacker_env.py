from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from online.sim_core import LoadBalanceSystem, best_of_heuristics_reference_reward
from train_ppo import ActorCritic


def _log_map01_to_range(x01: float, vmin: float, vmax: float) -> float:
    x = float(np.clip(x01, 0.0, 1.0))
    lo = float(np.log(max(vmin, 1e-12)))
    hi = float(np.log(max(vmax, 1e-12)))
    return float(np.exp(lo + x * (hi - lo)))


class FixedVictimPolicy:
    def __init__(
        self,
        checkpoint_path: str,
        num_servers: int,
        *,
        shield_rules_prefix: str | None = None,
        shield_risk_trigger_steps: int = 1,
        shield_cooldown_steps: int = 0,
        shield_empty_fallback: str = "least_load_proxy",
    ) -> None:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        cfg = dict(ckpt.get("config", {}))
        hidden = int(cfg.get("hidden", 128))
        obs_dim = int(num_servers + 1)
        act_dim = int(num_servers)
        self.model = ActorCritic(obs_dim=obs_dim, act_dim=act_dim, hidden=hidden)
        state_dict = ckpt.get("model_state_dict")
        if state_dict is None:
            state_dict = ckpt.get("model")
        if state_dict is None:
            raise RuntimeError(f"checkpoint missing model weights: {checkpoint_path}")
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.shield = None
        self.last_shield_stats: dict[str, float | int] = {}
        if shield_rules_prefix is not None:
            from shield.shield_runtime import load_shield_from_prefix

            self.shield = load_shield_from_prefix(
                str(shield_rules_prefix),
                num_servers=int(num_servers),
                risk_trigger_steps=int(shield_risk_trigger_steps),
                intervention_cooldown_steps=int(shield_cooldown_steps),
                empty_fallback=str(shield_empty_fallback),
            )

    def reset_runtime_state(self) -> None:
        self.last_shield_stats = {}
        if self.shield is not None and hasattr(self.shield, "reset_runtime_state"):
            self.shield.reset_runtime_state()

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> int:
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits, _v = self.model(obs_t)
        logits_np = logits.detach().cpu().numpy().reshape(-1)
        base_action = int(np.argmax(logits_np))
        if self.shield is None:
            self.last_shield_stats = {}
            return int(base_action)
        action, stats = self.shield.select_action(
            obs=np.asarray(obs, dtype=np.float32),
            scores=logits_np,
            base_action=int(base_action),
            action_selection="argmax",
        )
        self.last_shield_stats = dict(stats)
        return int(action)


class OnlineTraceAttackerEnv(gym.Env):
    """RL attacker env that crafts exogenous workload traces against a fixed victim policy.

    Action space:
      a[0] in [0,1] -> next inter-arrival dt via log mapping
      a[1] in [0,1] -> next job size via log mapping

    Reward:
      rolling_oracle_gap = reference_window_reward - victim_window_reward
    """

    metadata = {"render_modes": ["human", None]}

    def __init__(
        self,
        *,
        victim_checkpoint: str,
        num_servers: int = 10,
        service_rates: list[float] | None = None,
        episode_jobs: int = 1000,
        obs_high: float = 500000.0,
        reward_time_scale: float = 1000.0,
        dt_min: float = 1.0,
        dt_max: float = 500.0,
        size_min: float = 10.0,
        size_max: float = 5000.0,
        first_job_size: int = 100,
        oracle_window_len: int = 5,
        compute_oracle_gap: bool = True,
        include_last_gap: bool = True,
        shield_rules_prefix: str | None = None,
        shield_risk_trigger_steps: int = 1,
        shield_cooldown_steps: int = 0,
        shield_empty_fallback: str = "least_load_proxy",
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.num_servers = int(num_servers)
        if service_rates is None:
            if self.num_servers == 10:
                service_rates = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05]
            else:
                service_rates = np.linspace(0.15, 1.05, num=self.num_servers, dtype=np.float64).tolist()
        if len(service_rates) != self.num_servers:
            raise ValueError("service_rates length must match num_servers")

        self.service_rates = [float(x) for x in service_rates]
        self.episode_jobs = int(episode_jobs)
        self.obs_high = float(obs_high)
        self.reward_time_scale = float(reward_time_scale)
        if self.reward_time_scale <= 0:
            raise ValueError("reward_time_scale must be positive")

        self.dt_min = float(dt_min)
        self.dt_max = float(dt_max)
        self.size_min = float(size_min)
        self.size_max = float(size_max)
        self.first_job_size = int(first_job_size)
        self.oracle_window_len = int(oracle_window_len)
        if self.oracle_window_len <= 0:
            raise ValueError("oracle_window_len must be positive")
        self.compute_oracle_gap = bool(compute_oracle_gap)
        self.include_last_gap = bool(include_last_gap)

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        #* victim_obs + prev_action(2) + progress + optional last_gap
        obs_dim = (self.num_servers + 1) + 3 + (1 if self.include_last_gap else 0)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.np_random = np.random.RandomState(int(seed))
        self.victim_policy = FixedVictimPolicy(
            checkpoint_path=victim_checkpoint,
            num_servers=self.num_servers,
            shield_rules_prefix=shield_rules_prefix,
            shield_risk_trigger_steps=int(shield_risk_trigger_steps),
            shield_cooldown_steps=int(shield_cooldown_steps),
            shield_empty_fallback=str(shield_empty_fallback),
        )

        self.victim_sys = LoadBalanceSystem(
            service_rates=self.service_rates,
            obs_high=self.obs_high,
            reward_time_scale=self.reward_time_scale,
        )

        self._step_idx = 0
        self._prev_action = np.array([0.5, 0.5], dtype=np.float32)
        self._last_gap = 0.0
        self._trace_jobs: list[dict[str, int]] = []
        self._victim_obs = np.zeros((self.num_servers + 1,), dtype=np.float32)
        self._snapshot_hist: list[LoadBalanceSystem] = []
        self._future_arrival_hist: list[tuple[int | None, int | None]] = []
        self._victim_reward_hist: list[float] = []

    def _victim_state_metrics(self) -> dict[str, float | int]:
        obs = np.asarray(self.victim_sys.observe(), dtype=np.float64)
        backlog_proxy_total = float(np.sum(obs[:-1])) if obs.size > 1 else 0.0
        queue_jobs_total = 0
        running_jobs_total = 0
        queue_size_bytes_total = 0
        for server in list(self.victim_sys.servers):
            queue_jobs_total += int(len(server.queue))
            running_jobs_total += int(server.curr_job is not None)
            queue_size_bytes_total += int(sum(int(job.size) for job in list(server.queue)))
        active_jobs_total = int(queue_jobs_total + running_jobs_total)
        return {
            "sim/backlog_proxy_total": float(backlog_proxy_total),
            "sim/queue_jobs_total": int(queue_jobs_total),
            "sim/running_jobs_total": int(running_jobs_total),
            "sim/active_jobs_total": int(active_jobs_total),
            "sim/queue_size_bytes_total": int(queue_size_bytes_total),
        }

    def seed(self, seed: int | None = None):
        if seed is None:
            return
        self.np_random = np.random.RandomState(int(seed))

    def _build_obs(self) -> np.ndarray:
        progress = float(self._step_idx) / float(max(1, self.episode_jobs - 1))
        parts = [
            self._victim_obs.astype(np.float32),
            self._prev_action.astype(np.float32),
            np.asarray([progress], dtype=np.float32),
        ]
        if self.include_last_gap:
            parts.append(np.asarray([float(self._last_gap)], dtype=np.float32))
        obs = np.concatenate(parts)
        return obs.astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        del options
        if seed is not None:
            self.seed(seed)

        self._step_idx = 0
        self._prev_action = np.array([0.5, 0.5], dtype=np.float32)
        self._last_gap = 0.0

        self._trace_jobs = [{"arrival_time": 0, "job_size": int(self.first_job_size)}]

        self._victim_obs = self.victim_sys.reset(first_job_size=int(self.first_job_size), first_arrival_time=0)
        self.victim_policy.reset_runtime_state()
        self._snapshot_hist = []
        self._future_arrival_hist = []
        self._victim_reward_hist = []

        return self._build_obs(), {}

    def _action_to_job(self, action: np.ndarray) -> tuple[int, int]:
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.shape[0] != 2:
            raise ValueError("attacker action must have shape (2,)")
        dt = int(round(_log_map01_to_range(float(a[0]), self.dt_min, self.dt_max)))
        size = int(round(_log_map01_to_range(float(a[1]), self.size_min, self.size_max)))
        return max(0, dt), max(1, size)

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        a_clipped = np.clip(action, 0.0, 1.0)
        dt, size = self._action_to_job(a_clipped)

        has_next = self._step_idx < (self.episode_jobs - 1)
        next_dt = int(dt) if has_next else None
        next_size = int(size) if has_next else None

        victim_act = int(self.victim_policy.act(self._victim_obs))

        prev_t = int(self.victim_sys.curr_time)
        victim_snapshot = self.victim_sys.clone() if self.compute_oracle_gap else None
        self._victim_obs, r_victim, done_v, info_v = self.victim_sys.dispatch(
            victim_act, next_arrival_dt=next_dt, next_job_size=next_size
        )

        if has_next:
            self._trace_jobs.append({"arrival_time": int(prev_t + dt), "job_size": int(size)})

        if self.compute_oracle_gap:
            self._snapshot_hist.append(victim_snapshot)
            self._future_arrival_hist.append((next_dt, next_size))
            self._victim_reward_hist.append(float(r_victim))

            window_end = int(len(self._victim_reward_hist))
            window_start = max(0, window_end - int(self.oracle_window_len))
            oracle_result = best_of_heuristics_reference_reward(
                snapshot=self._snapshot_hist[int(window_start)],
                future_arrivals=self._future_arrival_hist[int(window_start) : int(window_end)],
            )
            r_oracle = float(oracle_result.reward)
            r_protocol = float(sum(self._victim_reward_hist[int(window_start) : int(window_end)]))
            gap = float(r_oracle - r_protocol)
            oracle_window_len = int(oracle_result.horizon)
            best_first_action = int(oracle_result.actions[0]) if len(oracle_result.actions) > 0 else -1
            reference_policy_name = str(oracle_result.policy_name)
        else:
            window_start = -1
            r_oracle = 0.0
            r_protocol = 0.0
            gap = 0.0
            oracle_window_len = 0
            best_first_action = -1
            reference_policy_name = ""

        self._last_gap = float(gap)
        self._prev_action = a_clipped.astype(np.float32)
        self._step_idx += 1

        done = bool(done_v or (self._step_idx >= self.episode_jobs))

        info = {
            "attacker/action_dt01": float(a_clipped[0]),
            "attacker/action_size01": float(a_clipped[1]),
            "oracle/r_opt": float(r_oracle),
            "oracle/r_protocol": float(r_protocol),
            "oracle/gap": float(gap),
            "oracle/window_len": int(oracle_window_len),
            "oracle/window_start": int(window_start),
            "oracle/best_first_action": int(best_first_action),
            "oracle/policy_name": str(reference_policy_name),
            "protocol/action": int(victim_act),
            "protocol/step_reward": float(r_victim),
            "env/reward": float(gap),
            "trace/next_dt": int(dt),
            "trace/next_size": int(size),
            "trace/num_jobs": int(len(self._trace_jobs)),
            "sim/victim_time": int(info_v.get("curr_time", 0)),
        }
        if self.victim_policy.last_shield_stats:
            for key, value in dict(self.victim_policy.last_shield_stats).items():
                info[str(key)] = value
        info.update(self._victim_state_metrics())

        return self._build_obs(), float(gap), bool(done), False, info

    def get_trace_jobs(self) -> list[dict[str, int]]:
        return [{"arrival_time": int(x["arrival_time"]), "job_size": int(x["job_size"])} for x in self._trace_jobs]

    def save_last_trace(self, path: str) -> None:
        payload = {
            "num_jobs": len(self._trace_jobs),
            "jobs": self.get_trace_jobs(),
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
