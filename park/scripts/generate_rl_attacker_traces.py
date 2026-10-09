#!/usr/bin/env python3
"""
#* Example from the repo root with a trained attacker model:

time python scripts/generate_rl_attacker_traces.py \
  --victim-checkpoint models/parklb_v1_350k.pt \
  --attacker-model online/models/v2/win5-12m-nolastgap-shielded03/best_model.zip \
  --episodes 100 \
  --episode-jobs 1000 \
  --reference-window 5 \
  --out-dir traces/shielded-iter3
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from online.attacker_config import infer_include_last_gap


def _parse_service_rates(text: str | None, num_servers: int) -> list[float] | None:
    if text is None:
        return None
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(vals) != int(num_servers):
        raise ValueError("--service-rates must have exactly num_servers comma-separated values")
    return vals


def _write_trace_csv(path: str, jobs: list[dict[str, int]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["job_id", "arrival_time", "job_size"])
        writer.writeheader()
        for i, j in enumerate(jobs):
            writer.writerow(
                {
                    "job_id": int(i),
                    "arrival_time": int(j["arrival_time"]),
                    "job_size": int(j["job_size"]),
                }
            )


def _write_trace_json(path: str, jobs: list[dict[str, int]], kind: str) -> None:
    payload = {
        "kind": kind,
        "num_jobs": int(len(jobs)),
        "jobs": [{"arrival_time": int(x["arrival_time"]), "job_size": int(x["job_size"])} for x in jobs],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    try:
        from stable_baselines3 import PPO
    except Exception as e:
        raise RuntimeError("Missing dependency: install stable-baselines3 to generate traces") from e

    from online.attacker_env import OnlineTraceAttackerEnv

    p = argparse.ArgumentParser(
        description="Roll out a trained RL attacker policy to generate workload traces."
    )
    p.add_argument("--victim-checkpoint", type=str, required=True)
    p.add_argument("--attacker-model", "--adv-model", dest="attacker_model", type=str, required=True)
    p.add_argument("--episodes", type=int, default=20)
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
    p.add_argument("--shield-rules-prefix", type=str, default=None)
    p.add_argument("--shield-risk-trigger-steps", type=int, default=1)
    p.add_argument("--shield-cooldown-steps", type=int, default=0)
    p.add_argument(
        "--shield-empty-fallback",
        type=str,
        default="least_load_proxy",
        choices=["least_load_proxy", "keep_rl", "open_all"],
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, default="online/adv-traces/RL-attacker")
    p.add_argument("--deterministic", action="store_true")
    args = p.parse_args()
    if int(args.reference_window) <= 0:
        raise ValueError("--reference-window must be positive")
    if int(args.shield_risk_trigger_steps) <= 0:
        raise ValueError("--shield-risk-trigger-steps must be positive")
    if int(args.shield_cooldown_steps) < 0:
        raise ValueError("--shield-cooldown-steps must be >= 0")

    service_rates = _parse_service_rates(args.service_rates, args.num_servers)
    include_last_gap = bool(infer_include_last_gap(str(args.attacker_model), default=True))

    #* Caveat: this script only rolls out one saved attacker policy and writes the resulting traces.
    #* Caveat: it does not score, rank, or search over generated traces after rollout.
    #* Caveat: generation disables oracle-gap computation for speed, so any last-gap feature stays identically 0.0.
    #* Caveat: reference_window is retained for config compatibility, but it does not affect jobs when gap computation is disabled.
    env = OnlineTraceAttackerEnv(
        victim_checkpoint=str(args.victim_checkpoint),
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
        compute_oracle_gap=False,
        include_last_gap=bool(include_last_gap),
        shield_rules_prefix=args.shield_rules_prefix,
        shield_risk_trigger_steps=int(args.shield_risk_trigger_steps),
        shield_cooldown_steps=int(args.shield_cooldown_steps),
        shield_empty_fallback=str(args.shield_empty_fallback),
        seed=int(args.seed),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    attacker_model = PPO.load(str(args.attacker_model))

    for ep in range(int(args.episodes)):
        obs, _ = env.reset(seed=int(args.seed + ep))
        done = False
        while not done:
            action, _ = attacker_model.predict(obs, deterministic=bool(args.deterministic))
            obs, _r, done, _trunc, _info = env.step(action)
        jobs = env.get_trace_jobs()
        _write_trace_csv(str(out_dir / f"episode_{ep:04d}.csv"), jobs)
        _write_trace_json(str(out_dir / f"episode_{ep:04d}.json"), jobs, kind="rl_attacker")

    meta = {
        "episodes": int(args.episodes),
        "victim_checkpoint": str(args.victim_checkpoint),
        "attacker_model": str(args.attacker_model),
        "adv_model": str(args.attacker_model),
        "include_last_gap": bool(include_last_gap),
        "seed": int(args.seed),
        "episode_jobs": int(args.episode_jobs),
        "reference_window": int(args.reference_window),
        "oracle_window": int(args.reference_window),
        "deterministic": bool(args.deterministic),
        "num_servers": int(args.num_servers),
        "service_rates": service_rates,
        "obs_high": float(args.obs_high),
        "dt_min": float(args.dt_min),
        "dt_max": float(args.dt_max),
        "size_min": float(args.size_min),
        "size_max": float(args.size_max),
        "first_job_size": int(args.first_job_size),
        "shield_rules_prefix": args.shield_rules_prefix,
        "shield_risk_trigger_steps": int(args.shield_risk_trigger_steps),
        "shield_cooldown_steps": int(args.shield_cooldown_steps),
        "shield_empty_fallback": str(args.shield_empty_fallback),
        "generation_mode": "pretrained_attacker_rollout",
        "generation_caveats": [
            "This script only rolls out a saved attacker policy and does not evaluate trace strength.",
            "Gap computation is disabled during generation, so the optional last-gap feature remains zero.",
            "reference_window is recorded for compatibility only and does not affect generated jobs in this mode.",
        ],
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"saved traces under: {out_dir}")


if __name__ == "__main__":
    main()
