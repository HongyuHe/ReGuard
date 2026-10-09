from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Any

import numpy as np


def _require_sb3():
    try:
        from stable_baselines3 import PPO  # type: ignore

        return PPO
    except Exception as e:  # pragma: no cover
        raise RuntimeError("* Missing dependency: install `stable-baselines3` to generate traces from a trained adversary") from e


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


def _write_segments_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _write_trace_file_from_segments(path: str, *, segments: list[dict[str, Any]]) -> None:
    if not segments:
        raise ValueError("* segments must be non-empty")

    times: list[float] = [0.0]
    bws: list[float] = [float(segments[0]["bw_mbps"])]
    t = 0.0
    for seg in segments:
        dur = float(seg["duration_s"])
        bw = float(seg["bw_mbps"])
        t += max(dur, 0.0)
        times.append(float(t))
        bws.append(float(bw))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for tm, bw in zip(times, bws):
            f.write(f"{tm:.6f}\t{bw:.6f}\n")


def _rollout_one_episode(*, env, adv_model, rng: np.random.RandomState, deterministic: bool, random_policy: bool) -> dict[str, Any]:
    obs, _ = env.reset()

    done = False
    last_info: dict[str, Any] = {}
    while not done:
        if bool(random_policy):
            action_value = float(rng.uniform(0.0, 1.0))
            action = np.asarray([action_value], dtype=np.float32)
        else:
            action, _ = adv_model.predict(obs, deterministic=bool(deterministic))
        obs, _reward, done, _trunc, info = env.step(action)
        if isinstance(info, dict):
            last_info = info

    if "episode/segments" not in last_info:
        raise RuntimeError("* env did not return episode metadata; expected `episode/segments` in info on done")

    return last_info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adv-model", type=str, default=None)
    parser.add_argument("--pensieve-model", type=str, default="ppo/nn_model_ep_426300.pth")
    parser.add_argument("--episodes", type=int, default=200)
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
    parser.add_argument("--out-dir", type=str, default="online/adv-traces")
    parser.add_argument("--flat-out-dir", action="store_true")
    parser.add_argument("--shield-min-rules-file", type=str, default=None)
    parser.add_argument("--shield-max-rules-file", type=str, default=None)
    parser.add_argument("--shield-action-selection", type=str, default="argmax", choices=["argmax", "gumbel"])

    parser.add_argument("--no-adv", action="store_true")
    parser.add_argument("--no-random", action="store_true")
    parser.add_argument("--deterministic", action="store_true")

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="pensieve-online-adv")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default="online")
    parser.add_argument("--wandb-tags", type=str, default="")

    args = parser.parse_args()

    from online.adv_env import OnlineBandwidthAdversaryEnv

    PPO = _require_sb3()

    #* Stable (seeded) stochastic trace generation.
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    try:
        import torch  # type: ignore

        torch.manual_seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))
    except Exception:
        pass

    out_dir = _resolve_path(str(args.out_dir))
    #* Output layout: by default keep the nested `pensieve/` + `random/` dirs for backwards compatibility.
    #* When `--flat-out-dir` is set, write adversarial traces directly under `--out-dir`.
    adv_dir = out_dir if bool(args.flat_out_dir) else os.path.join(out_dir, "pensieve")
    rnd_dir = os.path.join(out_dir, "random")

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
                "episodes": int(args.episodes),
                "seed": int(args.seed),
                "bw_min": float(args.bw_min),
                "bw_max": float(args.bw_max),
                "obs_history": int(args.obs_history),
                "oracle_window": int(args.oracle_window),
                "smooth_penalty_scale": float(args.smooth_penalty_scale),
                "reward_objective": str(args.reward_objective),
                "reward_tau": float(args.reward_tau),
                "reward_beta": float(args.reward_beta),
                "adv_model": args.adv_model,
                "shield_min_rules_file": args.shield_min_rules_file,
                "shield_max_rules_file": args.shield_max_rules_file,
                "shield_action_selection": str(args.shield_action_selection),
            },
        )

    rng = np.random.RandomState(int(args.seed))

    env = OnlineBandwidthAdversaryEnv(
        pensieve_model_path=str(args.pensieve_model),
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
        shield_min_rules_file=str(args.shield_min_rules_file) if args.shield_min_rules_file else None,
        shield_max_rules_file=str(args.shield_max_rules_file) if args.shield_max_rules_file else None,
        shield_action_selection=str(args.shield_action_selection),
    )

    adv_model = None
    if not bool(args.no_adv):
        if bool(args.deterministic) and int(args.episodes) > 1:
            print(
                "* Warning: `--deterministic` uses the mean action; with a deterministic env, all adversarial episodes "
                "will be identical. Omit `--deterministic` to sample diverse traces.",
                file=sys.stderr,
            )
        if args.adv_model is None:
            raise RuntimeError("* --adv-model is required unless --no-adv is set")
        adv_model = PPO.load(_resolve_path(str(args.adv_model)))
        try:
            adv_model.set_random_seed(int(args.seed))
        except Exception:
            pass

    if adv_model is not None:
        for ep in range(int(args.episodes)):
            info = _rollout_one_episode(
                env=env,
                adv_model=adv_model,
                rng=rng,
                deterministic=bool(args.deterministic),
                random_policy=False,
            )
            segs = list(info["episode/segments"])
            payload = {
                "kind": "pensieve_adversary",
                "episode": int(ep),
                "segments": segs,
                "episode_qoe_protocol": float(info.get("episode/qoe_protocol", 0.0)),
                "episode_rebuf_s_total": float(info.get("episode/rebuf_s_total", 0.0)),
                "bitrate_idxs": list(info.get("episode/bitrate_idxs", [])),
                "bw_values_mbps": list(info.get("episode/bw_values_mbps", [])),
            }
            _write_segments_json(os.path.join(adv_dir, f"episode_{ep:04d}.segments.json"), payload)
            _write_trace_file_from_segments(os.path.join(adv_dir, f"episode_{ep:04d}.trace"), segments=segs)

            if wandb is not None:
                wandb.log(
                    {
                        "eval/pensieve_qoe_protocol": float(payload["episode_qoe_protocol"]),
                        "eval/pensieve_rebuf_s_total": float(payload["episode_rebuf_s_total"]),
                        "eval/episode": int(ep),
                    },
                    step=int(ep),
                )

    if not bool(args.no_random):
        for ep in range(int(args.episodes)):
            info = _rollout_one_episode(
                env=env,
                adv_model=adv_model,
                rng=rng,
                deterministic=True,
                random_policy=True,
            )
            segs = list(info["episode/segments"])
            payload = {
                "kind": "random_baseline",
                "episode": int(ep),
                "segments": segs,
                "episode_qoe_protocol": float(info.get("episode/qoe_protocol", 0.0)),
                "episode_rebuf_s_total": float(info.get("episode/rebuf_s_total", 0.0)),
                "bitrate_idxs": list(info.get("episode/bitrate_idxs", [])),
                "bw_values_mbps": list(info.get("episode/bw_values_mbps", [])),
            }
            _write_segments_json(os.path.join(rnd_dir, f"episode_{ep:04d}.segments.json"), payload)
            _write_trace_file_from_segments(os.path.join(rnd_dir, f"episode_{ep:04d}.trace"), segments=segs)

            if wandb is not None:
                wandb.log(
                    {
                        "eval/random_qoe_protocol": float(payload["episode_qoe_protocol"]),
                        "eval/random_rebuf_s_total": float(payload["episode_rebuf_s_total"]),
                        "eval/episode": int(ep),
                    },
                    step=int(ep),
                )

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":  # pragma: no cover
    main()
