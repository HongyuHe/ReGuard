#!/usr/bin/env python3
from __future__ import annotations

"""
Example from the repo root:
time python scripts/eval_generated_attacker_oracle_regret.py \
  --victim-checkpoint models/parklb_v1_350k.pt \
  --adv-model online/models/parklb_v1_350k-attacker_win2oracle_best.zip \
  --episodes 200 \
  --eval-count 100 \
  --episode-jobs 1000 \
  --wandb \
  --wandb-name maxgap \
  --wandb-project park-lb-eval \
  --reference-window 2 \
  --out-dir online/eval/maxgap \
  --trace-out-dir traces/maxgap
"""

import argparse
import json

from lb_oracle_regret_utils import (
    VictimActor,
    init_wandb_run,
    log_eval_to_wandb,
    normalize_wandb_tags,
    parse_service_rates,
    resolve_path,
    write_trace_csv,
    write_trace_json,
    write_metrics_csvs,
)
from online.attacker_config import infer_include_last_gap
from shield.shield_runtime import load_shield_from_prefix


def main() -> None:
    try:
        from stable_baselines3 import PPO
    except Exception as e:
        raise RuntimeError("missing dependency: install stable-baselines3 to evaluate attacker traces") from e

    from online.attacker_env import OnlineTraceAttackerEnv

    p = argparse.ArgumentParser(description="Generate attacker traces and evaluate victim oracle regret on them")
    p.add_argument("--victim-checkpoint", type=str, required=True)
    p.add_argument("--eval-victim-checkpoint", type=str, default=None)
    p.add_argument("--attacker-victim-checkpoint", type=str, default=None)
    p.add_argument("--adv-model", type=str, required=True)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--eval-count", type=int, default=0)
    p.add_argument("--episode-jobs", type=int, default=1000)
    p.add_argument("--reference-window", "--oracle-k", dest="reference_window", type=int, default=5)
    p.add_argument("--num-servers", type=int, default=None)
    p.add_argument("--service-rates", type=str, default=None)
    p.add_argument("--obs-high", type=float, default=None)
    p.add_argument("--reward-time-scale", type=float, default=None)
    p.add_argument("--shield-rules-prefix", type=str, default=None)
    p.add_argument("--eval-shield-rules-prefix", type=str, default=None)
    p.add_argument("--attacker-shield-rules-prefix", type=str, default=None)
    p.add_argument("--shield-risk-trigger-steps", type=int, default=1)
    p.add_argument("--shield-cooldown-steps", type=int, default=0)
    p.add_argument(
        "--shield-empty-fallback",
        type=str,
        default="least_load_proxy",
        choices=["least_load_proxy", "keep_rl", "open_all"],
    )
    p.add_argument("--dt-min", type=float, default=1.0)
    p.add_argument("--dt-max", type=float, default=500.0)
    p.add_argument("--size-min", type=float, default=10.0)
    p.add_argument("--size-max", type=float, default=5000.0)
    p.add_argument("--first-job-size", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--trace-out-dir", type=str, default=None)
    p.add_argument("--wandb", dest="wandb", action="store_true")
    p.add_argument("--no-wandb", dest="wandb", action="store_false")
    p.add_argument("--wandb-project", type=str, default="park-lb-eval")
    p.add_argument("--wandb-name", type=str, default=None)
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-tags", nargs="*", default=None)
    p.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    p.set_defaults(wandb=True)
    args = p.parse_args()
    if int(args.eval_count) < 0:
        raise ValueError("--eval-count must be >= 0")
    if int(args.reference_window) <= 0:
        raise ValueError("--reference-window must be positive")
    if int(args.shield_risk_trigger_steps) <= 0:
        raise ValueError("--shield-risk-trigger-steps must be positive")
    if int(args.shield_cooldown_steps) < 0:
        raise ValueError("--shield-cooldown-steps must be >= 0")

    eval_victim_path = str(args.eval_victim_checkpoint or args.victim_checkpoint)
    attacker_victim_path = str(args.attacker_victim_checkpoint or args.victim_checkpoint)

    victim = VictimActor(checkpoint_path=str(resolve_path(eval_victim_path)))
    cfg = dict(victim.config)
    num_servers = int(args.num_servers if args.num_servers is not None else cfg.get("num_servers", victim.num_servers))
    service_rates = parse_service_rates(
        args.service_rates,
        num_servers=int(num_servers),
        checkpoint_config=cfg,
    )
    obs_high = float(args.obs_high if args.obs_high is not None else cfg.get("obs_high", 500000.0))
    reward_time_scale = float(
        args.reward_time_scale if args.reward_time_scale is not None else cfg.get("reward_time_scale", 1000.0)
    )
    include_last_gap = bool(infer_include_last_gap(str(resolve_path(str(args.adv_model))), default=True))
    eval_shield_prefix = args.eval_shield_rules_prefix or args.shield_rules_prefix
    attacker_shield_prefix = args.attacker_shield_rules_prefix or args.shield_rules_prefix
    eval_shield = None
    if eval_shield_prefix is not None:
        eval_shield = load_shield_from_prefix(
            str(resolve_path(str(eval_shield_prefix))),
            num_servers=int(num_servers),
            risk_trigger_steps=int(args.shield_risk_trigger_steps),
            intervention_cooldown_steps=int(args.shield_cooldown_steps),
            empty_fallback=str(args.shield_empty_fallback),
        )

    env = OnlineTraceAttackerEnv(
        victim_checkpoint=str(resolve_path(attacker_victim_path)),
        num_servers=int(num_servers),
        service_rates=list(service_rates),
        episode_jobs=int(args.episode_jobs),
        obs_high=float(obs_high),
        reward_time_scale=float(reward_time_scale),
        dt_min=float(args.dt_min),
        dt_max=float(args.dt_max),
        size_min=float(args.size_min),
        size_max=float(args.size_max),
        first_job_size=int(args.first_job_size),
        compute_oracle_gap=False,
        include_last_gap=bool(include_last_gap),
        seed=int(args.seed),
        shield_rules_prefix=None if attacker_shield_prefix is None else str(resolve_path(str(attacker_shield_prefix))),
        shield_risk_trigger_steps=int(args.shield_risk_trigger_steps),
        shield_cooldown_steps=int(args.shield_cooldown_steps),
        shield_empty_fallback=str(args.shield_empty_fallback),
    )
    adv_model = PPO.load(str(resolve_path(str(args.adv_model))))

    from lb_oracle_regret_utils import evaluate_jobs_with_oracle_regret

    episode_rows: list[dict[str, float | int | str]] = []
    step_rows: list[dict[str, float | int | str]] = []
    out_dir = resolve_path(str(args.out_dir))
    trace_out_dir = resolve_path(str(args.trace_out_dir)) if args.trace_out_dir is not None else out_dir / "generated_traces"
    trace_out_dir.mkdir(parents=True, exist_ok=True)
    eval_count = int(args.eval_count) if int(args.eval_count) > 0 else int(args.episodes)
    wandb_tags = normalize_wandb_tags(args.wandb_tags)
    wandb_run = init_wandb_run(
        enabled=bool(args.wandb),
        project=str(args.wandb_project),
        name=args.wandb_name,
        entity=args.wandb_entity,
        mode=str(args.wandb_mode),
        tags=list(wandb_tags),
        config={
            "script": "eval_generated_attacker_oracle_regret",
            "victim_checkpoint": str(resolve_path(eval_victim_path)),
            "attacker_victim_checkpoint": str(resolve_path(attacker_victim_path)),
            "adv_model": str(resolve_path(str(args.adv_model))),
            "episodes": int(args.episodes),
            "eval_count": int(min(int(eval_count), int(args.episodes))),
            "episode_jobs": int(args.episode_jobs),
            "reference_window": int(args.reference_window),
            "oracle_k": int(args.reference_window),
            "deterministic": bool(args.deterministic),
            "trace_out_dir": str(trace_out_dir),
            "metrics_out_dir": str(out_dir),
            "include_last_gap": bool(include_last_gap),
            "service_rates": list(service_rates),
            "obs_high": float(obs_high),
            "reward_time_scale": float(reward_time_scale),
            "shield_rules_prefix": None if args.shield_rules_prefix is None else str(resolve_path(str(args.shield_rules_prefix))),
            "eval_shield_rules_prefix": None if eval_shield_prefix is None else str(resolve_path(str(eval_shield_prefix))),
            "attacker_shield_rules_prefix": None if attacker_shield_prefix is None else str(resolve_path(str(attacker_shield_prefix))),
            "shield_risk_trigger_steps": int(args.shield_risk_trigger_steps),
            "shield_cooldown_steps": int(args.shield_cooldown_steps),
            "shield_empty_fallback": str(args.shield_empty_fallback),
            "seed": int(args.seed),
            "wandb_tags": list(wandb_tags),
        },
    )

    for ep in range(int(args.episodes)):
        obs, _info = env.reset(seed=int(args.seed + ep))
        done = False
        while not done:
            action, _state = adv_model.predict(obs, deterministic=bool(args.deterministic))
            obs, _reward, done, _trunc, _step_info = env.step(action)
        jobs = env.get_trace_jobs()
        trace_id = f"episode_{int(ep):04d}"
        trace_csv_path = trace_out_dir / f"{trace_id}.csv"
        trace_json_path = trace_out_dir / f"{trace_id}.json"
        write_trace_csv(trace_csv_path, jobs)
        write_trace_json(trace_json_path, jobs, kind="generated_attacker")
        if int(ep) < int(eval_count):
            episode_row, trace_step_rows = evaluate_jobs_with_oracle_regret(
                jobs=jobs,
                victim=victim,
                service_rates=list(service_rates),
                obs_high=float(obs_high),
                reward_time_scale=float(reward_time_scale),
                oracle_k=int(args.reference_window),
                trace_id=str(trace_id),
                trace_path=str(trace_csv_path),
                source="generated_attacker",
                shield=eval_shield,
            )
            episode_rows.append(episode_row)
            step_rows.extend(trace_step_rows)

    meta = {
        "victim_checkpoint": str(resolve_path(eval_victim_path)),
        "attacker_victim_checkpoint": str(resolve_path(attacker_victim_path)),
        "adv_model": str(resolve_path(str(args.adv_model))),
        "episodes_generated": int(args.episodes),
        "episodes_evaluated": int(min(int(eval_count), int(args.episodes))),
        "episode_jobs": int(args.episode_jobs),
        "reference_window": int(args.reference_window),
        "oracle_k": int(args.reference_window),
        "deterministic": bool(args.deterministic),
        "trace_out_dir": str(trace_out_dir),
        "metrics_out_dir": str(out_dir),
        "include_last_gap": bool(include_last_gap),
        "service_rates": list(service_rates),
        "obs_high": float(obs_high),
        "reward_time_scale": float(reward_time_scale),
        "shield_rules_prefix": None if args.shield_rules_prefix is None else str(resolve_path(str(args.shield_rules_prefix))),
        "eval_shield_rules_prefix": None if eval_shield_prefix is None else str(resolve_path(str(eval_shield_prefix))),
        "attacker_shield_rules_prefix": None if attacker_shield_prefix is None else str(resolve_path(str(attacker_shield_prefix))),
        "shield_risk_trigger_steps": int(args.shield_risk_trigger_steps),
        "shield_cooldown_steps": int(args.shield_cooldown_steps),
        "shield_empty_fallback": str(args.shield_empty_fallback),
        "seed": int(args.seed),
    }
    with (trace_out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    episode_csv, step_csv = write_metrics_csvs(
        out_dir=out_dir,
        episode_rows=episode_rows,
        step_rows=step_rows,
    )
    log_eval_to_wandb(
        wandb_run,
        episode_rows=episode_rows,
        step_rows=step_rows,
    )
    if wandb_run is not None:
        wandb_run.summary["paths/episode_metrics_csv"] = str(episode_csv)
        wandb_run.summary["paths/step_metrics_csv"] = str(step_csv)
        wandb_run.summary["paths/trace_out_dir"] = str(trace_out_dir)
        wandb_run.finish()
    print(f"wrote: {episode_csv}")
    print(f"wrote: {step_csv}")
    print(f"wrote traces under: {trace_out_dir}")


if __name__ == "__main__":
    main()
