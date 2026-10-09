#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from online.sim_core import parse_job_trace
from surrogate.common import (
    choose_trace_files,
    collect_trace_files,
    infer_control_bounds_from_jobs,
    parse_service_rates,
    resolve_path,
    write_trace_csv,
    write_trace_json,
)
from surrogate.data import collect_transition_dataset, save_transition_dataset
from surrogate.model import TorchVictimPolicy, build_surrogate_rollout_from_checkpoint
from surrogate.pgd import AttackConfig, attack_trace_jobs
from surrogate.train import save_surrogate_checkpoint, train_surrogate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the Park PGD-surrogate trace generation pipeline.")
    p.add_argument("--victim-checkpoint", type=str, required=True)
    p.add_argument("--seed-trace-dir", type=str, default=None)
    p.add_argument("--seed-trace-path", type=str, default=None)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--trace-select-mode", type=str, default="round_robin", choices=["random", "round_robin"])
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--surrogate-checkpoint", type=str, default=None)
    p.add_argument("--fit-trace-dir", type=str, default=None)
    p.add_argument("--fit-trace-path", type=str, default=None)
    p.add_argument("--fit-max-traces", type=int, default=20)
    p.add_argument("--fit-max-steps-per-trace", type=int, default=200)
    p.add_argument("--fit-hidden", type=int, default=256)
    p.add_argument("--fit-epochs", type=int, default=10)
    p.add_argument("--fit-batch-size", type=int, default=256)
    p.add_argument("--fit-lr", type=float, default=3e-4)
    p.add_argument("--fit-val-fraction", type=float, default=0.1)
    p.add_argument("--num-servers", type=int, default=10)
    p.add_argument("--service-rates", type=str, default=None)
    p.add_argument("--obs-high", type=float, default=500000.0)
    p.add_argument("--reward-time-scale", type=float, default=1000.0)
    p.add_argument("--oracle-k", type=int, default=5)
    p.add_argument("--window-k", type=int, default=5)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--restarts", type=int, default=3)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--dt-min", type=float, default=None)
    p.add_argument("--dt-max", type=float, default=None)
    p.add_argument("--size-min", type=float, default=None)
    p.add_argument("--size-max", type=float, default=None)
    p.add_argument("--dt-eps-abs", type=float, default=50.0)
    p.add_argument("--dt-eps-rel", type=float, default=0.5)
    p.add_argument("--size-eps-abs", type=float, default=1000.0)
    p.add_argument("--size-eps-rel", type=float, default=0.5)
    p.add_argument("--reward-weight", type=float, default=1.0)
    p.add_argument("--backlog-weight", type=float, default=0.05)
    p.add_argument("--active-weight", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def _resolve_seed_trace_files(args: argparse.Namespace) -> list[Path]:
    if args.seed_trace_dir is None and args.seed_trace_path is None:
        raise ValueError("provide --seed-trace-dir or --seed-trace-path")
    trace_files: list[Path] = []
    if args.seed_trace_dir is not None:
        trace_files.extend(collect_trace_files(resolve_path(str(args.seed_trace_dir))))
    if args.seed_trace_path is not None:
        trace_files.append(resolve_path(str(args.seed_trace_path)))
    trace_files = sorted(dict.fromkeys(trace_files))
    if len(trace_files) == 0:
        raise RuntimeError("no seed trace files found")
    return trace_files


def _resolve_fit_trace_files(args: argparse.Namespace, seed_trace_files: list[Path]) -> list[Path]:
    trace_files: list[Path] = []
    if args.fit_trace_dir is not None:
        trace_files.extend(collect_trace_files(resolve_path(str(args.fit_trace_dir))))
    if args.fit_trace_path is not None:
        trace_files.append(resolve_path(str(args.fit_trace_path)))
    if len(trace_files) == 0:
        trace_files = list(seed_trace_files)
    trace_files = sorted(dict.fromkeys(trace_files))
    if args.fit_max_traces is not None:
        trace_files = trace_files[: int(args.fit_max_traces)]
    return trace_files


def _infer_bounds(args: argparse.Namespace, trace_files: list[Path]) -> dict[str, float]:
    if (
        args.dt_min is not None
        and args.dt_max is not None
        and args.size_min is not None
        and args.size_max is not None
    ):
        return {
            "dt_min": float(args.dt_min),
            "dt_max": float(args.dt_max),
            "size_min": float(args.size_min),
            "size_max": float(args.size_max),
        }
    merged_jobs = []
    for trace_path in list(trace_files):
        merged_jobs.extend(parse_job_trace(str(trace_path)))
    inferred = infer_control_bounds_from_jobs(merged_jobs)
    if args.dt_min is not None:
        inferred["dt_min"] = float(args.dt_min)
    if args.dt_max is not None:
        inferred["dt_max"] = float(args.dt_max)
    if args.size_min is not None:
        inferred["size_min"] = float(args.size_min)
    if args.size_max is not None:
        inferred["size_max"] = float(args.size_max)
    return inferred


def _maybe_fit_surrogate(
    *,
    args: argparse.Namespace,
    seed_trace_files: list[Path],
    service_rates: list[float],
    bounds: dict[str, float],
    out_dir: Path,
) -> Path:
    if args.surrogate_checkpoint is not None:
        return resolve_path(str(args.surrogate_checkpoint))
    fit_trace_files = _resolve_fit_trace_files(args, seed_trace_files)
    dataset_path = out_dir / "artifacts" / "surrogate_dataset.npz"
    checkpoint_path = out_dir / "artifacts" / "surrogate_step_model.pt"
    arrays, meta = collect_transition_dataset(
        trace_files=fit_trace_files,
        victim_checkpoint=str(resolve_path(str(args.victim_checkpoint))),
        service_rates=list(service_rates),
        num_servers=int(args.num_servers),
        obs_high=float(args.obs_high),
        reward_time_scale=float(args.reward_time_scale),
        oracle_k=int(args.oracle_k),
        dt_min=float(bounds["dt_min"]),
        dt_max=float(bounds["dt_max"]),
        size_min=float(bounds["size_min"]),
        size_max=float(bounds["size_max"]),
        max_steps_per_trace=None
        if args.fit_max_steps_per_trace is None
        else int(args.fit_max_steps_per_trace),
        device=str(args.device),
    )
    save_transition_dataset(
        out_npz=dataset_path,
        arrays=arrays,
        meta=meta,
    )
    train_result = train_surrogate(
        arrays=arrays,
        num_servers=int(meta["num_servers"]),
        hidden=int(args.fit_hidden),
        epochs=int(args.fit_epochs),
        batch_size=int(args.fit_batch_size),
        lr=float(args.fit_lr),
        val_fraction=float(args.fit_val_fraction),
        seed=int(args.seed),
        device=str(args.device),
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    save_surrogate_checkpoint(
        path=str(checkpoint_path),
        train_result=train_result,
        meta=meta,
        hidden=int(args.fit_hidden),
    )
    return checkpoint_path


def main() -> None:
    args = parse_args()
    seed_trace_files = _resolve_seed_trace_files(args)
    bounds = _infer_bounds(args, seed_trace_files)
    service_rates = parse_service_rates(
        args.service_rates,
        num_servers=int(args.num_servers),
        checkpoint_config=None,
    )
    out_dir = resolve_path(str(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    surrogate_checkpoint = _maybe_fit_surrogate(
        args=args,
        seed_trace_files=seed_trace_files,
        service_rates=list(service_rates),
        bounds=bounds,
        out_dir=out_dir,
    )
    rng = np.random.default_rng(int(args.seed))
    selected_trace_files = choose_trace_files(
        trace_files=seed_trace_files,
        episodes=int(args.episodes),
        select_mode=str(args.trace_select_mode),
        rng=rng,
    )
    victim = TorchVictimPolicy(
        str(resolve_path(str(args.victim_checkpoint))),
        num_servers=int(args.num_servers),
        device=str(args.device),
    )
    rollout, surrogate_cfg, _ckpt = build_surrogate_rollout_from_checkpoint(
        checkpoint_path=str(surrogate_checkpoint),
        victim_policy=victim,
        device=str(args.device),
    )
    attack_config = AttackConfig(
        window_k=int(args.window_k),
        stride=int(args.stride),
        iters=int(args.iters),
        restarts=int(args.restarts),
        alpha=float(args.alpha),
        dt_min=float(bounds["dt_min"]),
        dt_max=float(bounds["dt_max"]),
        size_min=float(bounds["size_min"]),
        size_max=float(bounds["size_max"]),
        dt_eps_abs=float(args.dt_eps_abs),
        dt_eps_rel=float(args.dt_eps_rel),
        size_eps_abs=float(args.size_eps_abs),
        size_eps_rel=float(args.size_eps_rel),
        reward_weight=float(args.reward_weight),
        backlog_weight=float(args.backlog_weight),
        active_weight=float(args.active_weight),
    )
    trace_records = []
    for episode_idx, trace_path in enumerate(list(selected_trace_files)):
        jobs = parse_job_trace(str(trace_path))
        adv_jobs, attack_meta = attack_trace_jobs(
            jobs=jobs,
            victim=victim,
            rollout=rollout,
            service_rates=list(service_rates),
            obs_high=float(args.obs_high),
            reward_time_scale=float(args.reward_time_scale),
            config=attack_config,
            device=str(args.device),
        )
        trace_id = f"episode_{int(episode_idx):04d}"
        csv_path = out_dir / f"{trace_id}.csv"
        json_path = out_dir / f"{trace_id}.json"
        write_trace_csv(csv_path, adv_jobs)
        write_trace_json(
            json_path,
            adv_jobs,
            kind="pgd_surrogate",
            extra={
                "trace_id": str(trace_id),
                "seed_trace_path": str(trace_path),
                "surrogate_checkpoint": str(surrogate_checkpoint),
                "attack_meta": attack_meta,
            },
        )
        trace_records.append(
            {
                "trace_id": str(trace_id),
                "seed_trace_path": str(trace_path),
                "csv_path": str(csv_path),
                "json_path": str(json_path),
                "num_jobs": int(len(adv_jobs)),
                "window_count": int(len(list(attack_meta.get("windows", [])))),
            }
        )
    meta = {
        "script": "surrogate/generate_pgd_surrogate_traces.py",
        "victim_checkpoint": str(resolve_path(str(args.victim_checkpoint))),
        "surrogate_checkpoint": str(surrogate_checkpoint),
        "episodes": int(args.episodes),
        "trace_select_mode": str(args.trace_select_mode),
        "seed": int(args.seed),
        "num_servers": int(args.num_servers),
        "service_rates": [float(x) for x in list(service_rates)],
        "obs_high": float(args.obs_high),
        "reward_time_scale": float(args.reward_time_scale),
        "oracle_k": int(args.oracle_k),
        "window_k": int(args.window_k),
        "stride": int(args.stride),
        "iters": int(args.iters),
        "restarts": int(args.restarts),
        "alpha": float(args.alpha),
        "dt_min": float(bounds["dt_min"]),
        "dt_max": float(bounds["dt_max"]),
        "size_min": float(bounds["size_min"]),
        "size_max": float(bounds["size_max"]),
        "dt_eps_abs": float(args.dt_eps_abs),
        "dt_eps_rel": float(args.dt_eps_rel),
        "size_eps_abs": float(args.size_eps_abs),
        "size_eps_rel": float(args.size_eps_rel),
        "reward_weight": float(args.reward_weight),
        "backlog_weight": float(args.backlog_weight),
        "active_weight": float(args.active_weight),
        "surrogate_config": {
            "num_servers": int(surrogate_cfg.num_servers),
            "hidden": int(surrogate_cfg.hidden),
            "dt_min": float(surrogate_cfg.dt_min),
            "dt_max": float(surrogate_cfg.dt_max),
            "size_min": float(surrogate_cfg.size_min),
            "size_max": float(surrogate_cfg.size_max),
            "reward_mean": float(surrogate_cfg.reward_mean),
            "reward_std": float(surrogate_cfg.reward_std),
        },
        "traces": list(trace_records),
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote traces under: {out_dir}")
    print(f"meta: {out_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
