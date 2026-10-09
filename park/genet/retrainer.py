from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .curriculum import JobTraceCurriculum
from .runtime import resolve_repo_path
from .trace_family import trace_metadata, write_generated_trace_csv, write_generated_trace_json


@dataclass(frozen=True)
class FineTuneResult:
    model_path: str
    best_model_path: str
    trace_dir: str
    save_dir: str
    stats: dict[str, Any]


def load_checkpoint_config(model_path: str) -> dict[str, Any]:
    checkpoint = torch.load(resolve_repo_path(str(model_path)), map_location="cpu")
    return dict(checkpoint.get("config", {}))


def materialize_curriculum_traces(
    *,
    curriculum: JobTraceCurriculum,
    out_dir: str,
    num_traces: int,
    seed: int,
) -> str:
    if int(num_traces) <= 0:
        raise ValueError("num_traces must be positive")
    trace_dir = Path(out_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    np_rng = np.random.default_rng(int(seed))
    manifest_rows: list[dict[str, Any]] = []
    for idx in range(int(num_traces)):
        trace_seed = int(seed + idx)
        sampled = curriculum.sample_trace(
            rng=np_rng,
            trace_seed=int(trace_seed),
        )
        trace_id = f"curriculum_{int(idx):04d}"
        csv_path = trace_dir / f"{trace_id}.csv"
        json_path = trace_dir / f"{trace_id}.json"
        write_generated_trace_csv(csv_path, sampled.trace)
        write_generated_trace_json(json_path, trace_id=str(trace_id), trace=sampled.trace)
        row = trace_metadata(trace_id=str(trace_id), trace=sampled.trace)
        row["component_index"] = int(sampled.component_index)
        row["csv_path"] = str(csv_path)
        row["json_path"] = str(json_path)
        manifest_rows.append(row)
    meta_path = trace_dir / "meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "generator": "park_lb_genet_curriculum",
                "num_traces": int(num_traces),
                "seed": int(seed),
                "manifest": manifest_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return str(trace_dir)


def fine_tune_on_curriculum(
    *,
    model_path: str,
    curriculum: JobTraceCurriculum,
    out_dir: str,
    seed: int,
    num_traces: int,
    total_iters: int,
    eval_every: int,
    eval_episodes: int,
    python_executable: str | None = None,
) -> FineTuneResult:
    save_dir = Path(out_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = materialize_curriculum_traces(
        curriculum=curriculum,
        out_dir=str(save_dir / "curriculum_traces"),
        num_traces=int(num_traces),
        seed=int(seed),
    )
    cfg = load_checkpoint_config(str(model_path))
    script_path = resolve_repo_path("train_ppo.py")
    python_bin = str(python_executable or sys.executable)
    command = [
        str(python_bin),
        str(script_path),
        "--seed",
        str(int(seed)),
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
        str(trace_dir),
        "--trace-select-mode",
        "round_robin",
        "--total-iters",
        str(int(total_iters)),
        "--rollout-steps",
        str(int(cfg.get("rollout_steps", 4096))),
        "--ppo-epochs",
        str(int(cfg.get("ppo_epochs", 10))),
        "--mini-batch",
        str(int(cfg.get("mini_batch", 512))),
        "--gamma",
        str(float(cfg.get("gamma", 0.99))),
        "--gae-lambda",
        str(float(cfg.get("gae_lambda", 0.95))),
        "--clip",
        str(float(cfg.get("clip", 0.2))),
        "--vf-coef",
        str(float(cfg.get("vf_coef", 0.5))),
        "--ent-coef",
        str(float(cfg.get("ent_coef", 0.01))),
        "--normalize-value-loss",
        str(int(cfg.get("normalize_value_loss", 1))),
        "--lr",
        str(float(cfg.get("lr", 3e-4))),
        "--hidden",
        str(int(cfg.get("hidden", 128))),
        "--eval-every",
        str(int(eval_every)),
        "--eval-episodes",
        str(int(eval_episodes)),
        "--save-dir",
        str(save_dir),
        "--resume-from",
        str(resolve_repo_path(str(model_path))),
        "--resume-optimizer",
        "1",
        "--resume-entropy-weight",
        "1",
    ]
    subprocess.run(
        command,
        cwd=str(Path(script_path).resolve().parent),
        check=True,
    )
    last_path = save_dir / "last.pt"
    best_path = save_dir / "best.pt"
    return FineTuneResult(
        model_path=str(last_path),
        best_model_path=str(best_path),
        trace_dir=str(trace_dir),
        save_dir=str(save_dir),
        stats={
            "command": command,
            "trace_count": int(num_traces),
            "total_iters": int(total_iters),
            "eval_every": int(eval_every),
            "eval_episodes": int(eval_episodes),
        },
    )
