from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any


def _require_sb3():
    try:
        from stable_baselines3 import PPO  # type: ignore
        from stable_baselines3.common.callbacks import BaseCallback  # type: ignore
        from stable_baselines3.common.monitor import Monitor  # type: ignore
        from stable_baselines3.common.vec_env import DummyVecEnv  # type: ignore

        return PPO, BaseCallback, Monitor, DummyVecEnv
    except Exception as e:  # pragma: no cover
        raise RuntimeError("* Missing dependency: install `stable-baselines3` to train the online adversary") from e


def _try_import_wandb() -> Any | None:
    try:
        import wandb  # type: ignore

        if not hasattr(wandb, "init"):
            raise ImportError("* `wandb` import resolved to a non-package directory; install the wandb pip package")
        return wandb
    except Exception:
        return None


def _resolve_path(repo_relative_or_abs: str) -> str:
    expanded = os.path.expandvars(os.path.expanduser(str(repo_relative_or_abs)))
    if os.path.isabs(expanded):
        return expanded
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(repo_root, expanded)


def _resolve_existing_input_path(repo_relative_or_abs: str, *, arg_name: str) -> str:
    expanded = os.path.expandvars(os.path.expanduser(str(repo_relative_or_abs)))
    candidates: list[str] = []
    if os.path.isabs(expanded):
        candidates.append(os.path.abspath(expanded))
    else:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        candidates.extend(
            [
                os.path.abspath(expanded),
                os.path.join(repo_root, "Pensieve-PPO", "src", expanded),
                os.path.join(repo_root, expanded),
            ]
        )

    seen: set[str] = set()
    for candidate in list(candidates):
        candidate_abs = os.path.abspath(str(candidate))
        if str(candidate_abs) in set(seen):
            continue
        seen.add(str(candidate_abs))
        if os.path.isfile(str(candidate_abs)):
            return str(candidate_abs)

    searched = ", ".join(str(path) for path in list(candidates))
    raise FileNotFoundError(f"* {arg_name} not found: {repo_relative_or_abs}; searched: {searched}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pensieve-model", type=str, default="ppo/nn_model_ep_426300.pth")
    parser.add_argument("--total-steps", type=int, default=600_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bw-min", type=float, default=0.8)
    parser.add_argument("--bw-max", type=float, default=4.8)
    parser.add_argument("--obs-history", type=int, default=10)
    parser.add_argument("--oracle-window", type=int, default=4)
    parser.add_argument("--smooth-penalty-scale", type=float, default=1.0)
    parser.add_argument(
        "--reward-objective",
        type=str,
        default="weighted_max_regret",
        choices=["max_regret", "weighted_max_regret", "oracle_floor_max_regret"],
    )
    parser.add_argument("--reward-tau", type=float, default=0.0)
    parser.add_argument("--reward-beta", type=float, default=1.0)
    parser.add_argument("--out-dir", type=str, default="online/models")
    parser.add_argument("--shield-min-rules-file", type=str, default=None)
    parser.add_argument("--shield-max-rules-file", type=str, default=None)
    parser.add_argument("--shield-action-selection", type=str, default="argmax", choices=["argmax", "gumbel"])

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="pensieve-online-adv")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default="online")
    parser.add_argument("--wandb-tags", type=str, default="")
    parser.add_argument("--wandb-log-every", type=int, default=100)

    args = parser.parse_args()

    if (args.shield_min_rules_file is None) != (args.shield_max_rules_file is None):
        raise RuntimeError("* Provide both --shield-min-rules-file and --shield-max-rules-file, or neither")

    pensieve_model_path = _resolve_existing_input_path(str(args.pensieve_model), arg_name="--pensieve-model")
    shield_min_rules_file = None
    shield_max_rules_file = None
    if args.shield_min_rules_file is not None and args.shield_max_rules_file is not None:
        shield_min_rules_file = _resolve_existing_input_path(
            str(args.shield_min_rules_file),
            arg_name="--shield-min-rules-file",
        )
        shield_max_rules_file = _resolve_existing_input_path(
            str(args.shield_max_rules_file),
            arg_name="--shield-max-rules-file",
        )
    victim_shield_enabled = bool(shield_min_rules_file is not None and shield_max_rules_file is not None)

    from online.adv_env import OnlineBandwidthAdversaryEnv

    PPO, BaseCallback, Monitor, DummyVecEnv = _require_sb3()

    wandb = None
    wandb_run = None
    if bool(args.wandb):
        wandb = _try_import_wandb()
        if wandb is None:
            raise RuntimeError("* --wandb set but `wandb` is not importable; install `wandb` and retry")
        tags = [t.strip() for t in str(args.wandb_tags).split(",") if t.strip()]
        wandb_run = wandb.init(
            project=str(args.wandb_project),
            entity=args.wandb_entity,
            name=args.wandb_name,
            mode=str(args.wandb_mode),
            tags=tags,
            config={
                "pensieve_model": args.pensieve_model,
                "pensieve_model_resolved": str(pensieve_model_path),
                "total_steps": int(args.total_steps),
                "lr": float(args.lr),
                "seed": int(args.seed),
                "bw_min": float(args.bw_min),
                "bw_max": float(args.bw_max),
                "obs_history": int(args.obs_history),
                "oracle_window": int(args.oracle_window),
                "smooth_penalty_scale": float(args.smooth_penalty_scale),
                "reward_objective": str(args.reward_objective),
                "reward_tau": float(args.reward_tau),
                "reward_beta": float(args.reward_beta),
                "shield_min_rules_file": args.shield_min_rules_file,
                "shield_max_rules_file": args.shield_max_rules_file,
                "shield_min_rules_file_resolved": shield_min_rules_file,
                "shield_max_rules_file_resolved": shield_max_rules_file,
                "shield_action_selection": str(args.shield_action_selection),
                "victim_shield_enabled": bool(victim_shield_enabled),
            },
        )

    class WandbInfoCallback(BaseCallback):
        def __init__(self, *, log_every: int = 100) -> None:
            super().__init__()
            self._log_every = max(int(log_every), 1)

        def _on_step(self) -> bool:
            if wandb is None:
                return True
            if (int(self.num_timesteps) % int(self._log_every)) != 0:
                return True
            infos = self.locals.get("infos", [])
            if not infos:
                return True

            info = infos[0] if isinstance(infos, list) else infos
            if not isinstance(info, dict):
                return True

            metrics: dict[str, float | int] = {}
            for k, v in info.items():
                if isinstance(v, (int, float)):
                    metrics[str(k)] = v
            metrics["train/num_timesteps"] = int(self.num_timesteps)
            wandb.log(metrics, step=int(self.num_timesteps))
            return True

    env = OnlineBandwidthAdversaryEnv(
        pensieve_model_path=str(pensieve_model_path),
        seed=int(args.seed),
        bw_min_mbps=float(args.bw_min),
        bw_max_mbps=float(args.bw_max),
        obs_history_len=int(args.obs_history),
        oracle_window_len=int(args.oracle_window),
        smooth_penalty_scale=float(args.smooth_penalty_scale),
        reward_objective=str(args.reward_objective),
        reward_tau=float(args.reward_tau),
        reward_beta=float(args.reward_beta),
        deterministic_protocol=True,
        shield_min_rules_file=str(shield_min_rules_file) if shield_min_rules_file is not None else None,
        shield_max_rules_file=str(shield_max_rules_file) if shield_max_rules_file is not None else None,
        shield_action_selection=str(args.shield_action_selection),
    )
    venv = DummyVecEnv([lambda: Monitor(env)])

    policy_kwargs = {"net_arch": dict(pi=[32, 16], vf=[32, 16])}
    model = PPO(
        "MlpPolicy",
        venv,
        learning_rate=float(args.lr),
        seed=int(args.seed),
        policy_kwargs=policy_kwargs,
        verbose=1,
    )

    callback = WandbInfoCallback(log_every=int(args.wandb_log_every))
    model.learn(total_timesteps=int(args.total_steps), callback=callback)

    out_dir = _resolve_path(str(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    model_path = os.path.join(out_dir, f"adv_ppo_bw_{stamp}.zip")
    model.save(model_path)

    cfg_path = os.path.join(out_dir, f"adv_ppo_bw_{stamp}.config.json")
    config_payload = dict(vars(args))
    config_payload["pensieve_model_resolved"] = str(pensieve_model_path)
    config_payload["shield_min_rules_file_resolved"] = shield_min_rules_file
    config_payload["shield_max_rules_file_resolved"] = shield_max_rules_file
    config_payload["victim_shield_enabled"] = bool(victim_shield_enabled)
    with open(cfg_path, "w") as f:
        json.dump(config_payload, f, indent=2, sort_keys=True)

    if wandb_run is not None:
        wandb_run.log({"artifact/model_path": model_path})
        wandb_run.finish()

    print(model_path)


if __name__ == "__main__":  # pragma: no cover
    main()
