from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np


def _parse_service_rates(text: str | None, num_servers: int) -> list[float] | None:
    if text is None:
        return None
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(vals) != int(num_servers):
        raise ValueError("--service-rates must have exactly num_servers comma-separated values")
    return vals


def _parse_wandb_tags(text: str | None) -> list[str]:
    if text is None or str(text).strip() == "":
        return []
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _resolve_path(text: str) -> str:
    path = Path(os.path.expandvars(os.path.expanduser(str(text))))
    return str(path if path.is_absolute() else path.resolve())


def _resolve_optional_path(text: str | None) -> str | None:
    if text is None or str(text).strip() == "":
        return None
    return _resolve_path(str(text))


def _validate_shield_rules_prefix(prefix: str | None, *, num_servers: int) -> None:
    if prefix is None:
        return
    manifest_path = Path(str(prefix) + "_manifest.json")
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_num_servers = manifest.get("num_servers")
        if manifest_num_servers is not None and int(manifest_num_servers) != int(num_servers):
            raise ValueError(
                f"shield rule manifest num_servers={int(manifest_num_servers)} does not match --num-servers={int(num_servers)}"
            )
    required_paths = [Path(str(prefix) + "_risk.pl")]
    required_paths.extend([Path(str(prefix) + f"_allow_server{int(server_id)}.pl") for server_id in range(int(num_servers))])
    missing = [str(path) for path in list(required_paths) if not path.is_file()]
    if len(missing) > 0:
        preview = ", ".join(list(missing[:5]))
        suffix = "" if len(missing) <= 5 else f", ... ({len(missing)} missing)"
        raise FileNotFoundError(f"shield rules prefix is incomplete; missing: {preview}{suffix}")


def _series_payload(name: str, values: list[float], *, include_sum: bool = False) -> dict[str, float]:
    if len(values) == 0:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    payload: dict[str, float] = {
        f"{name}_mean": float(np.mean(arr)),
        f"{name}_std": float(np.std(arr)),
        f"{name}_p50": float(np.percentile(arr, 50.0)),
        f"{name}_p95": float(np.percentile(arr, 95.0)),
        f"{name}_min": float(np.min(arr)),
        f"{name}_max": float(np.max(arr)),
        f"{name}_last": float(arr[-1]),
    }
    if include_sum:
        payload[f"{name}_sum"] = float(np.sum(arr))
    return payload


def _has_tensorboard() -> bool:
    try:
        import tensorboard as _tensorboard
    except Exception:
        return False
    return _tensorboard is not None


def _init_wandb(args: argparse.Namespace, config: dict):
    if str(args.wandb_mode) == "disabled":
        return None
    try:
        import wandb
    except Exception as e:
        raise RuntimeError("Missing dependency: install wandb or run with --wandb-mode disabled") from e

    run = wandb.init(
        project=str(args.wandb_project),
        entity=args.wandb_entity,
        name=args.wandb_name,
        mode=str(args.wandb_mode),
        tags=_parse_wandb_tags(args.wandb_tags),
        group=args.wandb_group,
        dir=args.wandb_dir,
        sync_tensorboard=bool(config.get("tensorboard_available", False)),
        config=config,
    )
    wandb.define_metric("train/timesteps")
    wandb.define_metric("train/*", step_metric="train/timesteps")
    wandb.define_metric("eval/*", step_metric="train/timesteps")
    wandb.define_metric("time/*", step_metric="train/timesteps")
    wandb.define_metric("rollout/*", step_metric="train/timesteps")
    wandb.define_metric("attacker/*", step_metric="train/timesteps")
    wandb.define_metric("oracle/*", step_metric="train/timesteps")
    wandb.define_metric("protocol/*", step_metric="train/timesteps")
    wandb.define_metric("trace/*", step_metric="train/timesteps")
    wandb.define_metric("sim/*", step_metric="train/timesteps")
    wandb.define_metric("checkpoint/*", step_metric="train/timesteps")
    return run


def _make_wandb_callback(base_callback_cls, wandb_run, log_freq: int):
    class WandbAttackerCallback(base_callback_cls):
        def __init__(self):
            super().__init__()
            self.log_freq = int(max(1, int(log_freq)))
            self.last_log_step = 0
            self.start_time = time.perf_counter()
            self.rewards: list[float] = []
            self.episode_rewards: list[float] = []
            self.episode_lengths: list[float] = []
            self.running_episode_rewards: list[float] = []
            self.running_episode_lengths: list[int] = []
            self.policy_counts: dict[str, int] = {}
            self.values: dict[str, list[float]] = {
                "attacker/action_dt01": [],
                "attacker/action_size01": [],
                "oracle/r_opt": [],
                "oracle/r_protocol": [],
                "oracle/gap": [],
                "oracle/window_len": [],
                "oracle/window_start": [],
                "oracle/best_first_action": [],
                "protocol/step_reward": [],
                "protocol/action": [],
                "trace/next_dt": [],
                "trace/next_size": [],
                "trace/num_jobs": [],
                "sim/backlog_proxy_total": [],
                "sim/queue_jobs_total": [],
                "sim/running_jobs_total": [],
                "sim/active_jobs_total": [],
                "sim/queue_size_bytes_total": [],
                "sim/victim_time": [],
                "shield/risky_state": [],
                "shield/allowed_count": [],
                "shield/intervene": [],
                "shield/action_changed": [],
                "shield/rl_action_allowed": [],
                "shield/empty_allowed": [],
                "shield/fallback_used": [],
                "shield/time_ms": [],
                "shield/base_action": [],
                "shield/selected_action": [],
            }

        def _log_sb3_metrics(self, payload: dict[str, float | int]) -> None:
            logger_values = getattr(self.model.logger, "name_to_value", {})
            if not isinstance(logger_values, dict):
                return
            metric_map = {
                "train/approx_kl": "train/approx_kl",
                "train/clip_fraction": "train/clip_fraction",
                "train/clip_range": "train/clip_range",
                "train/entropy_loss": "train/entropy_loss",
                "train/explained_variance": "train/explained_variance",
                "train/learning_rate": "train/learning_rate",
                "train/loss": "train/loss",
                "train/policy_gradient_loss": "train/policy_gradient_loss",
                "train/std": "train/std",
                "train/value_loss": "train/value_loss",
                "rollout/ep_len_mean": "rollout/ep_len_mean",
                "rollout/ep_rew_mean": "rollout/ep_rew_mean",
                "time/fps": "time/fps",
                "time/iterations": "time/iterations",
                "time/time_elapsed": "time/time_elapsed",
                "time/total_timesteps": "time/total_timesteps",
                "eval/mean_reward": "eval/mean_reward",
                "eval/mean_ep_length": "eval/mean_ep_length",
            }
            for sb3_key, wandb_key in metric_map.items():
                if sb3_key not in logger_values:
                    continue
                value = logger_values[sb3_key]
                if value is None:
                    continue
                try:
                    payload[wandb_key] = float(value)
                except Exception:
                    continue
            entropy_loss = logger_values.get("train/entropy_loss")
            if entropy_loss is not None:
                try:
                    payload["train/entropy"] = float(-float(entropy_loss))
                except Exception:
                    pass
            policy_gradient_loss = logger_values.get("train/policy_gradient_loss")
            if policy_gradient_loss is not None:
                try:
                    payload["train/policy_loss"] = float(policy_gradient_loss)
                except Exception:
                    pass

        def _on_step(self) -> bool:
            rewards = self.locals.get("rewards", [])
            if rewards is not None:
                for reward in np.asarray(rewards, dtype=np.float64).reshape(-1):
                    self.rewards.append(float(reward))

            infos = self.locals.get("infos", [])
            for info in infos:
                if not isinstance(info, dict):
                    continue
                policy_name = info.get("oracle/policy_name")
                if policy_name is not None:
                    policy_name_text = str(policy_name).strip()
                    if policy_name_text != "":
                        self.policy_counts[policy_name_text] = int(self.policy_counts.get(policy_name_text, 0)) + 1
                for key, buf in self.values.items():
                    val = info.get(key)
                    if val is not None:
                        buf.append(float(val))
            dones = self.locals.get("dones", [])
            reward_arr = np.asarray(rewards if rewards is not None else [], dtype=np.float64).reshape(-1)
            done_arr = np.asarray(dones if dones is not None else [], dtype=np.bool_).reshape(-1)
            if reward_arr.size > 0:
                if len(self.running_episode_rewards) != int(reward_arr.size):
                    self.running_episode_rewards = [0.0] * int(reward_arr.size)
                    self.running_episode_lengths = [0] * int(reward_arr.size)
                for idx, reward in enumerate(reward_arr):
                    self.running_episode_rewards[int(idx)] += float(reward)
                    self.running_episode_lengths[int(idx)] += 1
                    is_done = bool(done_arr[int(idx)]) if int(idx) < int(done_arr.size) else False
                    if is_done:
                        self.episode_rewards.append(float(self.running_episode_rewards[int(idx)]))
                        self.episode_lengths.append(float(self.running_episode_lengths[int(idx)]))
                        self.running_episode_rewards[int(idx)] = 0.0
                        self.running_episode_lengths[int(idx)] = 0

            if int(self.num_timesteps) - int(self.last_log_step) >= int(self.log_freq):
                payload = {
                    "train/timesteps": int(self.num_timesteps),
                    "rollout/reward_count": int(len(self.rewards)),
                    "rollout/episode_count": int(len(self.episode_rewards)),
                    "time/wall_clock_ms": float((time.perf_counter() - self.start_time) * 1000.0),
                }
                payload.update(_series_payload("rollout/reward", self.rewards, include_sum=True))
                if len(self.episode_rewards) > 0:
                    payload.update(_series_payload("rollout/episode_reward", self.episode_rewards))
                    payload.update(_series_payload("rollout/episode_length", self.episode_lengths))
                for key, buf in self.values.items():
                    payload.update(_series_payload(key, buf))
                if len(self.values["oracle/gap"]) > 0:
                    gap_arr = np.asarray(self.values["oracle/gap"], dtype=np.float64)
                    payload["oracle/gap_positive_rate"] = float(np.mean(gap_arr > 0.0))
                    payload["oracle/gap_negative_rate"] = float(np.mean(gap_arr < 0.0))
                policy_total = int(sum(self.policy_counts.values()))
                if int(policy_total) > 0:
                    payload["oracle/policy_count"] = int(policy_total)
                    for name, count in sorted(self.policy_counts.items()):
                        payload[f"oracle/policy_rate/{name}"] = float(float(count) / float(policy_total))
                self._log_sb3_metrics(payload)
                wandb_run.log(payload, step=int(self.num_timesteps))
                self.last_log_step = int(self.num_timesteps)
                self.rewards.clear()
                self.episode_rewards.clear()
                self.episode_lengths.clear()
                self.policy_counts.clear()
                for buf in self.values.values():
                    buf.clear()
            return True

    return WandbAttackerCallback()


def main() -> None:
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
    except Exception as e:
        raise RuntimeError("Missing dependency: install stable-baselines3 to train attacker") from e

    from online.attacker_env import OnlineTraceAttackerEnv

    p = argparse.ArgumentParser(description="Train PPO attacker to generate challenging load-balance traces")
    p.add_argument("--victim-checkpoint", type=str, required=True)
    p.add_argument("--num-servers", type=int, default=10)
    p.add_argument("--service-rates", type=str, default=None)
    p.add_argument("--episode-jobs", type=int, default=1000)
    p.add_argument("--obs-high", type=float, default=500000.0)

    p.add_argument("--dt-min", type=float, default=1.0)
    p.add_argument("--dt-max", type=float, default=500.0)
    p.add_argument("--size-min", type=float, default=10.0)
    p.add_argument("--size-max", type=float, default=5000.0)
    p.add_argument("--first-job-size", type=int, default=100)
    p.add_argument("--reference-window", "--oracle-window", dest="reference_window", type=int, default=5)
    p.add_argument("--disable-last-gap-feature", action="store_true")
    p.add_argument(
        "--shield-rules-prefix",
        "--victim-shield-rules-prefix",
        dest="shield_rules_prefix",
        type=str,
        default=None,
        help="Prefix for shield rule files used to protect the fixed victim during attacker training and eval.",
    )
    p.add_argument("--shield-risk-trigger-steps", type=int, default=1)
    p.add_argument("--shield-cooldown-steps", type=int, default=0)
    p.add_argument(
        "--shield-empty-fallback",
        type=str,
        default="least_load_proxy",
        choices=["least_load_proxy", "keep_rl", "open_all"],
    )

    p.add_argument("--total-timesteps", type=int, default=300000)
    p.add_argument("--n-steps", type=int, default=20480)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, default="online/models")
    p.add_argument("--eval-freq", type=int, default=10000)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--wandb-mode", type=str, default="disabled", choices=["disabled", "offline", "online"])
    p.add_argument("--wandb-project", type=str, default="park-load-balance-torch")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-name", type=str, default=None)
    p.add_argument("--wandb-tags", type=str, default="")
    p.add_argument("--wandb-group", type=str, default=None)
    p.add_argument("--wandb-dir", type=str, default=None)
    p.add_argument("--wandb-log-freq", type=int, default=1000)
    args = p.parse_args()
    if int(args.reference_window) <= 0:
        raise ValueError("--reference-window must be positive")
    if int(args.shield_risk_trigger_steps) <= 0:
        raise ValueError("--shield-risk-trigger-steps must be positive")
    if int(args.shield_cooldown_steps) < 0:
        raise ValueError("--shield-cooldown-steps must be >= 0")

    victim_checkpoint_path = _resolve_path(str(args.victim_checkpoint))
    if not Path(str(victim_checkpoint_path)).is_file():
        raise FileNotFoundError(f"--victim-checkpoint not found: {victim_checkpoint_path}")
    shield_rules_prefix = _resolve_optional_path(args.shield_rules_prefix)
    _validate_shield_rules_prefix(shield_rules_prefix, num_servers=int(args.num_servers))

    service_rates = _parse_service_rates(args.service_rates, args.num_servers)
    tensorboard_available = _has_tensorboard()
    wandb_run = _init_wandb(
        args,
        {
            **vars(args),
            "victim_checkpoint": str(victim_checkpoint_path),
            "shield_rules_prefix": shield_rules_prefix,
            "protected_victim": bool(shield_rules_prefix is not None),
            "model_kind": "park_lb_online_attacker",
            "service_rates_parsed": service_rates,
            "tensorboard_available": bool(tensorboard_available),
            "include_last_gap": bool(not args.disable_last_gap_feature),
            "reference_window": int(args.reference_window),
        },
    )

    env = OnlineTraceAttackerEnv(
        victim_checkpoint=str(victim_checkpoint_path),
        num_servers=int(args.num_servers),
        service_rates=service_rates,
        episode_jobs=int(args.episode_jobs),
        obs_high=float(args.obs_high),
        dt_min=float(args.dt_min),
        dt_max=float(args.dt_max),
        size_min=float(args.size_min),
        size_max=float(args.size_max),
        first_job_size=int(args.first_job_size),
        oracle_window_len=int(args.reference_window),
        include_last_gap=bool(not args.disable_last_gap_feature),
        shield_rules_prefix=shield_rules_prefix,
        shield_risk_trigger_steps=int(args.shield_risk_trigger_steps),
        shield_cooldown_steps=int(args.shield_cooldown_steps),
        shield_empty_fallback=str(args.shield_empty_fallback),
        seed=int(args.seed),
    )
    eval_env = OnlineTraceAttackerEnv(
        victim_checkpoint=str(victim_checkpoint_path),
        num_servers=int(args.num_servers),
        service_rates=service_rates,
        episode_jobs=int(args.episode_jobs),
        obs_high=float(args.obs_high),
        dt_min=float(args.dt_min),
        dt_max=float(args.dt_max),
        size_min=float(args.size_min),
        size_max=float(args.size_max),
        first_job_size=int(args.first_job_size),
        oracle_window_len=int(args.reference_window),
        include_last_gap=bool(not args.disable_last_gap_feature),
        shield_rules_prefix=shield_rules_prefix,
        shield_risk_trigger_steps=int(args.shield_risk_trigger_steps),
        shield_cooldown_steps=int(args.shield_cooldown_steps),
        shield_empty_fallback=str(args.shield_empty_fallback),
        seed=int(args.seed) + 100000,
    )
    os.makedirs(args.out_dir, exist_ok=True)
    tb_log_dir = os.path.join(args.out_dir, "tb")

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        seed=int(args.seed),
        n_steps=int(args.n_steps),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.0,
        tensorboard_log=(tb_log_dir if bool(tensorboard_available) else None),
    )

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_path = os.path.join(args.out_dir, f"adv_ppo_lb_{stamp}.zip")
    best_model_path = os.path.join(args.out_dir, "best_model.zip")
    cfg_path = os.path.join(args.out_dir, f"adv_ppo_lb_{stamp}.config.json")
    best_cfg_path = os.path.join(args.out_dir, "best_model.config.json")
    if os.path.exists(best_model_path):
        os.remove(best_model_path)
    if wandb_run is not None:
        wandb_run.config.update(
            {
                "model_path": str(model_path),
                "best_model_path": str(best_model_path),
                "config_path": str(cfg_path),
                "best_config_path": str(best_cfg_path),
            },
            allow_val_change=True,
        )

    callbacks = []
    if int(args.eval_freq) > 0:
        callbacks.append(
            EvalCallback(
                eval_env,
                best_model_save_path=str(args.out_dir),
                log_path=str(args.out_dir),
                eval_freq=int(args.eval_freq),
                n_eval_episodes=int(args.eval_episodes),
                deterministic=True,
                render=False,
                warn=False,
            )
        )
    if wandb_run is not None:
        callbacks.append(_make_wandb_callback(BaseCallback, wandb_run, int(args.wandb_log_freq)))
    callback = CallbackList(callbacks) if len(callbacks) > 0 else None

    model.learn(total_timesteps=int(args.total_timesteps), callback=callback)
    model.save(model_path)
    if not os.path.exists(best_model_path):
        shutil.copyfile(model_path, best_model_path)

    payload = {
        "created_at": stamp,
        "victim_checkpoint": str(victim_checkpoint_path),
        "protected_victim": bool(shield_rules_prefix is not None),
        "shield_rules_prefix": shield_rules_prefix,
        "shield_risk_trigger_steps": int(args.shield_risk_trigger_steps),
        "shield_cooldown_steps": int(args.shield_cooldown_steps),
        "shield_empty_fallback": str(args.shield_empty_fallback),
        "num_servers": int(args.num_servers),
        "service_rates": service_rates,
        "episode_jobs": int(args.episode_jobs),
        "obs_high": float(args.obs_high),
        "dt_min": float(args.dt_min),
        "dt_max": float(args.dt_max),
        "size_min": float(args.size_min),
        "size_max": float(args.size_max),
        "first_job_size": int(args.first_job_size),
        "reference_window": int(args.reference_window),
        "oracle_window": int(args.reference_window),
        "include_last_gap": bool(not args.disable_last_gap_feature),
        "total_timesteps": int(args.total_timesteps),
        "n_steps": int(args.n_steps),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "eval_freq": int(args.eval_freq),
        "eval_episodes": int(args.eval_episodes),
        "seed": int(args.seed),
        "model_path": str(model_path),
        "best_model_path": str(best_model_path),
        "tensorboard_dir": (str(tb_log_dir) if bool(tensorboard_available) else None),
        "tensorboard_available": bool(tensorboard_available),
        "wandb_mode": str(args.wandb_mode),
        "wandb_project": str(args.wandb_project),
        "wandb_name": args.wandb_name,
    }
    _write_json(cfg_path, payload)
    _write_json(best_cfg_path, payload)

    if wandb_run is not None:
        wandb_run.summary["checkpoint/model_path"] = str(model_path)
        wandb_run.summary["checkpoint/best_model_path"] = str(best_model_path)
        wandb_run.summary["checkpoint/config_path"] = str(cfg_path)
        wandb_run.summary["checkpoint/best_config_path"] = str(best_cfg_path)
        if bool(tensorboard_available):
            wandb_run.summary["checkpoint/tensorboard_dir"] = str(tb_log_dir)
        wandb_run.summary["train/total_timesteps"] = int(args.total_timesteps)
        wandb_run.finish()

    print(f"saved attacker model: {model_path}")
    print(f"saved best attacker model: {best_model_path}")
    print(f"saved config: {cfg_path}")
    print(f"saved best config: {best_cfg_path}")


if __name__ == "__main__":
    main()
