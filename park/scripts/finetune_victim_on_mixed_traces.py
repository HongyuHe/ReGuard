#!/usr/bin/env python3
"""
#* Example from the repo root:

time python scripts/finetune_victim_on_mixed_traces.py \
  --checkpoint models/parklb_v1_350k.pt \
  --normal-trace-dir traces/mixed \
  --adversarial-trace-dir traces/mixed \
  --out-dir finetune/parklb_v1_mixed \
  --normal-fraction 0.3 \
  --total-iters 1000 \
  --overwrite

time python scripts/finetune_victim_on_mixed_traces.py \
  --checkpoint models/parklb_v1_350k.pt \
  --normal-trace-dir traces/clean \
  --adversarial-trace-dir traces/clean \
  --out-dir finetune/parklb_v1_clean \
  --normal-fraction 0.3 \
  --total-iters 1000 \
  --overwrite

time python scripts/finetune_victim_on_mixed_traces.py \
  --checkpoint models/parklb_v1_350k.pt \
  --normal-trace-dir traces/clean \
  --adversarial-trace-dir traces/genet \
  --out-dir finetune/parklb_v1_genet \
  --normal-fraction 0.3 \
  --total-iters 1000 \
  --overwrite

time python scripts/finetune_victim_on_mixed_traces.py \
  --checkpoint models/parklb_v1_350k.pt \
  --normal-trace-dir traces/clean \
  --adversarial-trace-dir traces/maxgap \
  --out-dir finetune/parklb_v1_maxgap \
  --normal-fraction 0.3 \
  --total-iters 1000 \
  --overwrite
  
time python scripts/finetune_victim_on_mixed_traces.py \
  --checkpoint models/parklb_v1_350k.pt \
  --normal-trace-dir traces/clean \
  --adversarial-trace-dir traces/pgd-surrogate \
  --out-dir finetune/parklb_v1_pgd-surrogate \
  --normal-fraction 0.3 \
  --total-iters 1000 \
  --overwrite

time python scripts/finetune_victim_on_mixed_traces.py \
  --checkpoint models/parklb_v1_350k.pt \
  --normal-trace-dir traces/clean \
  --adversarial-trace-dir traces/random \
  --out-dir finetune/parklb_v1_random \
  --normal-fraction 0.3 \
  --total-iters 1000 \
  --overwrite

time python scripts/finetune_victim_on_mixed_traces.py \
  --checkpoint models/parklb_v1_350k.pt \
  --normal-trace-dir traces/clean \
  --adversarial-trace-dir traces/oracle2step \
  --out-dir finetune/parklb_v1_oracle2step \
  --normal-fraction 0.3 \
  --total-iters 1000 \
  --overwrite
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.lb_oracle_regret_utils import collect_trace_files, resolve_path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_checkpoint(path: Path) -> dict[str, Any]:
    return torch.load(str(path), map_location="cpu")


def _checkpoint_config(path: Path) -> dict[str, Any]:
    checkpoint = _load_checkpoint(path)
    return dict(checkpoint.get("config", {}))


def _validate_resume_state(path: Path, *, allow_missing_optimizer_state: bool) -> dict[str, Any]:
    checkpoint = _load_checkpoint(path)
    model_state = checkpoint.get("model_state_dict", checkpoint.get("model"))
    if model_state is None:
        raise RuntimeError(f"checkpoint missing model weights: {path}")
    optimizer_state = checkpoint.get("optimizer_state_dict", checkpoint.get("optimizer"))
    if optimizer_state is None and not bool(allow_missing_optimizer_state):
        raise RuntimeError(
            "checkpoint missing optimizer state. Re-run with --allow-missing-optimizer-state only if this is intentional."
        )
    return {
        "has_model_state": bool(model_state is not None),
        "has_optimizer_state": bool(optimizer_state is not None),
        "has_entropy_weight": bool(checkpoint.get("entropy_weight") is not None),
        "entropy_weight": checkpoint.get("entropy_weight"),
        "checkpoint_iter": checkpoint.get("iter"),
        "eval_mean_reward": checkpoint.get("eval_mean_reward"),
        "eval_mean_reward_per_job": checkpoint.get("eval_mean_reward_per_job"),
    }


def _choose_trace_subset(
    *,
    rng: np.random.Generator,
    files: list[Path],
    count: int,
    with_replacement: bool,
) -> list[Path]:
    if int(count) <= 0:
        return []
    if len(files) == 0:
        raise RuntimeError("cannot sample from an empty trace list")
    replace = bool(with_replacement or int(count) > len(files))
    indices = rng.choice(np.arange(len(files)), size=int(count), replace=bool(replace))
    return [files[int(idx)] for idx in list(indices)]


def _link_or_copy_trace(*, src: Path, dst: Path, copy_mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if str(copy_mode) == "copy":
        shutil.copy2(str(src), str(dst))
        return
    os.symlink(str(src.resolve()), str(dst))


def _materialize_mixed_traces(
    *,
    normal_files: list[Path],
    adversarial_files: list[Path],
    out_dir: Path,
    normal_fraction: float,
    normal_trace_count: int | None,
    adversarial_trace_count: int,
    seed: int,
    copy_mode: str,
) -> dict[str, Any]:
    if not (0.0 <= float(normal_fraction) < 1.0):
        raise ValueError("--normal-fraction must be in [0, 1)")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(seed))

    adv_count = int(adversarial_trace_count if int(adversarial_trace_count) > 0 else len(adversarial_files))
    if adv_count <= 0:
        raise ValueError("adversarial trace count must be positive")
    if normal_trace_count is None:
        frac = float(normal_fraction)
        normal_count = int(round(float(adv_count) * frac / max(1.0 - frac, 1e-9)))
    else:
        normal_count = int(normal_trace_count)
    if normal_count < 0:
        raise ValueError("normal trace count must be non-negative")

    selected_adv = _choose_trace_subset(
        rng=rng,
        files=list(adversarial_files),
        count=int(adv_count),
        with_replacement=False,
    )
    selected_normal = _choose_trace_subset(
        rng=rng,
        files=list(normal_files),
        count=int(normal_count),
        with_replacement=False,
    )
    rows: list[dict[str, Any]] = []
    combined = [("adv", path) for path in list(selected_adv)] + [("normal", path) for path in list(selected_normal)]
    rng.shuffle(combined)
    for idx, (source_kind, src_path) in enumerate(list(combined)):
        suffix = src_path.suffix.lower()
        dst_name = f"mixed_{int(idx):05d}_{str(source_kind)}{suffix}"
        dst_path = out_dir / dst_name
        _link_or_copy_trace(src=src_path, dst=dst_path, copy_mode=str(copy_mode))
        rows.append(
            {
                "mixed_index": int(idx),
                "source_kind": str(source_kind),
                "source_path": str(src_path.resolve()),
                "mixed_path": str(dst_path),
            }
        )
    manifest = {
        "normal_available": int(len(normal_files)),
        "adversarial_available": int(len(adversarial_files)),
        "normal_selected": int(len(selected_normal)),
        "adversarial_selected": int(len(selected_adv)),
        "normal_fraction_requested": float(normal_fraction),
        "normal_fraction_actual": float(len(selected_normal) / max(1, len(rows))),
        "copy_mode": str(copy_mode),
        "seed": int(seed),
        "traces": rows,
    }
    _write_json(out_dir / "manifest.json", manifest)
    return manifest


def _arg_or_cfg(args: argparse.Namespace, cfg: dict[str, Any], name: str, default: Any) -> Any:
    value = getattr(args, name)
    if value is not None:
        return value
    return cfg.get(name.replace("_", "-"), cfg.get(name, default))


def _default_finetune_lr(cfg: dict[str, Any], explicit_lr: float | None) -> float:
    if explicit_lr is not None:
        return float(explicit_lr)
    base_lr = float(cfg.get("lr", 3e-4))
    return float(base_lr * 0.1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a Park LB PPO victim on adversarial traces mixed with normal traces."
    )
    parser.add_argument("--checkpoint", "--pretrained-checkpoint", dest="checkpoint", type=str, required=True)
    parser.add_argument("--normal-trace-dir", type=str, required=True)
    parser.add_argument("--adversarial-trace-dir", "--adv-trace-dir", dest="adversarial_trace_dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--normal-fraction", type=float, default=0.3)
    parser.add_argument("--normal-trace-count", type=int, default=None)
    parser.add_argument("--adversarial-trace-count", type=int, default=0)
    parser.add_argument("--copy-mode", type=str, choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-iters", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--ppo-epochs", type=int, default=None)
    parser.add_argument("--mini-batch", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--gae-lambda", type=float, default=None)
    parser.add_argument("--clip", type=float, default=None)
    parser.add_argument("--vf-coef", type=float, default=None)
    parser.add_argument("--ent-coef", type=float, default=None)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--wandb-mode", type=str, default="disabled", choices=["disabled", "offline", "online"])
    parser.add_argument("--wandb-project", type=str, default="park-load-balance-torch")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, default="finetune,mixed-traces")
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-dir", type=str, default=None)
    parser.add_argument("--allow-missing-optimizer-state", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve_path(str(args.checkpoint))
    normal_dir = resolve_path(str(args.normal_trace_dir))
    adversarial_dir = resolve_path(str(args.adversarial_trace_dir))
    out_dir = resolve_path(str(args.out_dir))
    if out_dir.exists() and any(out_dir.iterdir()) and not bool(args.overwrite):
        raise RuntimeError(f"output directory is not empty: {out_dir}. Re-run with --overwrite.")
    if out_dir.exists() and bool(args.overwrite):
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    resume_state = _validate_resume_state(
        checkpoint_path,
        allow_missing_optimizer_state=bool(args.allow_missing_optimizer_state),
    )
    cfg = _checkpoint_config(checkpoint_path)
    normal_files = collect_trace_files(normal_dir)
    adversarial_files = collect_trace_files(adversarial_dir)
    if len(normal_files) == 0:
        raise RuntimeError(f"no valid normal traces found in: {normal_dir}")
    if len(adversarial_files) == 0:
        raise RuntimeError(f"no valid adversarial traces found in: {adversarial_dir}")

    mixed_trace_dir = out_dir / "mixed_traces"
    manifest = _materialize_mixed_traces(
        normal_files=list(normal_files),
        adversarial_files=list(adversarial_files),
        out_dir=mixed_trace_dir,
        normal_fraction=float(args.normal_fraction),
        normal_trace_count=args.normal_trace_count,
        adversarial_trace_count=int(args.adversarial_trace_count),
        seed=int(args.seed),
        copy_mode=str(args.copy_mode),
    )

    lr = _default_finetune_lr(cfg, args.lr)
    command = [
        sys.executable,
        str(ROOT_DIR / "train_ppo.py"),
        "--seed",
        str(int(args.seed)),
        "--num-servers",
        str(int(cfg.get("num_servers", 10))),
        "--num-stream-jobs",
        str(int(cfg.get("num_stream_jobs", 1000))),
        "--job-size-pareto-shape",
        str(float(cfg.get("job_size_pareto_shape", 1.5))),
        "--job-size-pareto-scale",
        str(float(cfg.get("job_size_pareto_scale", 100.0))),
        "--job-interval",
        str(float(cfg.get("job_interval", 55.0))),
        "--obs-high",
        str(float(cfg.get("obs_high", 500000.0))),
        "--reward-time-scale",
        str(float(cfg.get("reward_time_scale", 1000.0))),
        "--workload-trace-dir",
        str(mixed_trace_dir),
        "--trace-select-mode",
        "random",
        "--total-iters",
        str(int(args.total_iters)),
        "--rollout-steps",
        str(int(_arg_or_cfg(args, cfg, "rollout_steps", 4096))),
        "--ppo-epochs",
        str(int(_arg_or_cfg(args, cfg, "ppo_epochs", 10))),
        "--mini-batch",
        str(int(_arg_or_cfg(args, cfg, "mini_batch", 512))),
        "--gamma",
        str(float(_arg_or_cfg(args, cfg, "gamma", 0.99))),
        "--gae-lambda",
        str(float(_arg_or_cfg(args, cfg, "gae_lambda", 0.95))),
        "--clip",
        str(float(_arg_or_cfg(args, cfg, "clip", 0.2))),
        "--vf-coef",
        str(float(_arg_or_cfg(args, cfg, "vf_coef", 0.5))),
        "--ent-coef",
        str(float(_arg_or_cfg(args, cfg, "ent_coef", 0.01))),
        "--normalize-value-loss",
        str(int(cfg.get("normalize_value_loss", 1))),
        "--lr",
        str(float(lr)),
        "--hidden",
        str(int(cfg.get("hidden", 128))),
        "--eval-every",
        str(int(args.eval_every)),
        "--eval-episodes",
        str(int(args.eval_episodes)),
        "--save-dir",
        str(out_dir / "checkpoints"),
        "--resume-from",
        str(checkpoint_path),
        "--resume-optimizer",
        "1",
        "--resume-entropy-weight",
        "1",
        "--wandb-mode",
        str(args.wandb_mode),
        "--wandb-project",
        str(args.wandb_project),
        "--wandb-tags",
        str(args.wandb_tags),
    ]
    if args.wandb_name is not None:
        command.extend(["--wandb-name", str(args.wandb_name)])
    if args.wandb_entity is not None:
        command.extend(["--wandb-entity", str(args.wandb_entity)])
    if args.wandb_group is not None:
        command.extend(["--wandb-group", str(args.wandb_group)])
    if args.wandb_dir is not None:
        command.extend(["--wandb-dir", str(args.wandb_dir)])

    run_config = {
        "script": "finetune_victim_on_mixed_traces",
        "checkpoint": str(checkpoint_path),
        "normal_trace_dir": str(normal_dir),
        "adversarial_trace_dir": str(adversarial_dir),
        "out_dir": str(out_dir),
        "mixed_trace_dir": str(mixed_trace_dir),
        "resume_state": dict(resume_state),
        "checkpoint_config": dict(cfg),
        "finetune_lr": float(lr),
        "normal_fraction": float(args.normal_fraction),
        "manifest": dict(manifest),
        "train_command": list(command),
    }
    _write_json(out_dir / "finetune_config.json", run_config)
    subprocess.run(command, cwd=str(ROOT_DIR), check=True)
    print(f"finetuning complete. outputs under: {out_dir}")
    print(f"mixed traces: {mixed_trace_dir}")
    print(f"checkpoints: {out_dir / 'checkpoints'}")


if __name__ == "__main__":
    main()
