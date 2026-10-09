from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from online.sim_core import parse_job_trace
from scripts.lb_oracle_regret_utils import collect_trace_files

from .bo_search import GenetBOTraceGenerator
from .curriculum import JobTraceCurriculum
from .retrainer import FineTuneResult, fine_tune_on_curriculum
from .runtime import load_runtime, resolve_repo_path
from .scoring import ObjectiveName, ParkTraceScorer
from .trace_family import JobTraceSearchSpace


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _prepare_out_dir(out_dir: Path, *, overwrite: bool) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        if not bool(overwrite):
            raise RuntimeError(f"output dir is not empty: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def _copy_round_traces(*, src_dir: Path, dst_dir: Path, round_idx: int) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for trace_path in list(_resolved_trace_files(src_dir)):
        dst_name = f"round_{int(round_idx):02d}_{trace_path.name}"
        shutil.copyfile(str(trace_path), str(dst_dir / dst_name))
        copied += 1
    return int(copied)


def _resolved_trace_files(trace_dir: Path) -> list[Path]:
    valid_paths: list[Path] = []
    for path in list(collect_trace_files(trace_dir)):
        try:
            jobs = parse_job_trace(str(path))
        except Exception:
            continue
        if len(jobs) == 0:
            continue
        valid_paths.append(path)
    return valid_paths


def _trace_paths(trace_dir: Path, *, limit: int = 0) -> list[str]:
    paths = [str(path) for path in list(_resolved_trace_files(trace_dir))]
    if int(limit) > 0:
        return list(paths[: int(limit)])
    return paths


def _score_trace_dir(
    *,
    model_path: str,
    trace_dir: Path,
    objective: ObjectiveName,
    reference_window: int,
    shield_rules_prefix: str | None = None,
    shield_risk_trigger_steps: int = 1,
    shield_cooldown_steps: int = 0,
    shield_empty_fallback: str = "least_load_proxy",
    limit: int = 0,
) -> dict[str, float]:
    scorer = ParkTraceScorer(
        load_runtime(
            str(model_path),
            reference_window=int(reference_window),
            shield_rules_prefix=shield_rules_prefix,
            shield_risk_trigger_steps=int(shield_risk_trigger_steps),
            shield_cooldown_steps=int(shield_cooldown_steps),
            shield_empty_fallback=str(shield_empty_fallback),
        )
    )
    return scorer.score_trace_dir(
        trace_paths=_trace_paths(trace_dir, limit=int(limit)),
        objective=objective,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Park LB Genet iterative search and retraining loop.")
    parser.add_argument("--model", type=str, default="models/parklb_v1_350k.pt")
    parser.add_argument(
        "--objective",
        type=str,
        default="reference_gap_per_job",
        choices=["reference_gap_total", "reference_gap_per_job", "regret_mean", "regret_p95"],
    )
    parser.add_argument("--reference-window", "--oracle-k", dest="reference_window", type=int, default=5)
    parser.add_argument("--shield-rules-prefix", type=str, default=None)
    parser.add_argument("--shield-risk-trigger-steps", type=int, default=1)
    parser.add_argument("--shield-cooldown-steps", type=int, default=0)
    parser.add_argument(
        "--shield-empty-fallback",
        type=str,
        default="least_load_proxy",
        choices=["least_load_proxy", "keep_rl", "open_all"],
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-jobs", type=int, default=300)
    parser.add_argument("--base-dt-min", type=float, default=10.0)
    parser.add_argument("--base-dt-max", type=float, default=120.0)
    parser.add_argument("--base-size-scale-min", type=float, default=40.0)
    parser.add_argument("--base-size-scale-max", type=float, default=400.0)
    parser.add_argument("--pareto-shape-min", type=float, default=1.1)
    parser.add_argument("--pareto-shape-max", type=float, default=2.5)
    parser.add_argument("--burst-start-min", type=float, default=0.05)
    parser.add_argument("--burst-start-max", type=float, default=0.8)
    parser.add_argument("--burst-duration-min", type=float, default=0.05)
    parser.add_argument("--burst-duration-max", type=float, default=0.4)
    parser.add_argument("--burst-dt-mult-min", type=float, default=0.05)
    parser.add_argument("--burst-dt-mult-max", type=float, default=0.8)
    parser.add_argument("--burst-size-mult-min", type=float, default=1.0)
    parser.add_argument("--burst-size-mult-max", type=float, default=12.0)
    parser.add_argument("--export-traces-per-round", type=int, default=10)
    parser.add_argument("--traces-per-eval", type=int, default=5)
    parser.add_argument("--bo-init-points", type=int, default=8)
    parser.add_argument("--bo-iter", type=int, default=12)
    parser.add_argument("--candidate-pool-size", type=int, default=1024)
    parser.add_argument("--curriculum-add-prob", type=float, default=0.3)
    parser.add_argument("--training-trace-count", type=int, default=64)
    parser.add_argument("--updates-per-round", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--val-trace-dir", type=str, default="")
    parser.add_argument("--val-trace-limit", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="genet/runs/default")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(resolve_repo_path(str(args.out_dir)))
    _prepare_out_dir(out_dir, overwrite=bool(args.overwrite))
    if int(args.rounds) <= 0:
        raise ValueError("rounds must be positive")
    if int(args.reference_window) <= 0:
        raise ValueError("reference_window must be positive")
    if int(args.shield_risk_trigger_steps) <= 0:
        raise ValueError("shield_risk_trigger_steps must be positive")
    if int(args.shield_cooldown_steps) < 0:
        raise ValueError("shield_cooldown_steps must be >= 0")

    search_space = JobTraceSearchSpace(
        num_jobs=int(args.num_jobs),
        base_dt_ms_range=(float(args.base_dt_min), float(args.base_dt_max)),
        base_size_scale_bytes_range=(float(args.base_size_scale_min), float(args.base_size_scale_max)),
        pareto_shape_range=(float(args.pareto_shape_min), float(args.pareto_shape_max)),
        burst_start_frac_range=(float(args.burst_start_min), float(args.burst_start_max)),
        burst_duration_frac_range=(float(args.burst_duration_min), float(args.burst_duration_max)),
        burst_dt_multiplier_range=(float(args.burst_dt_mult_min), float(args.burst_dt_mult_max)),
        burst_size_multiplier_range=(float(args.burst_size_mult_min), float(args.burst_size_mult_max)),
    )
    curriculum = JobTraceCurriculum.from_search_space(search_space)
    curriculum.dump(out_dir / "curriculum_initial.json")

    config_payload = {
        "model": str(resolve_repo_path(str(args.model))),
        "objective": str(args.objective),
        "reference_window": int(args.reference_window),
        "shield_rules_prefix": args.shield_rules_prefix,
        "shield_risk_trigger_steps": int(args.shield_risk_trigger_steps),
        "shield_cooldown_steps": int(args.shield_cooldown_steps),
        "shield_empty_fallback": str(args.shield_empty_fallback),
        "rounds": int(args.rounds),
        "seed": int(args.seed),
        "search_space": search_space.to_dict(),
        "export_traces_per_round": int(args.export_traces_per_round),
        "traces_per_eval": int(args.traces_per_eval),
        "bo_init_points": int(args.bo_init_points),
        "bo_iter": int(args.bo_iter),
        "candidate_pool_size": int(args.candidate_pool_size),
        "curriculum_add_prob": float(args.curriculum_add_prob),
        "training_trace_count": int(args.training_trace_count),
        "updates_per_round": int(args.updates_per_round),
        "eval_every": int(args.eval_every),
        "eval_episodes": int(args.eval_episodes),
        "val_trace_dir": str(args.val_trace_dir),
        "val_trace_limit": int(args.val_trace_limit),
        "out_dir": str(out_dir),
    }
    _write_json(out_dir / "config.json", config_payload)

    current_model_path = str(resolve_repo_path(str(args.model)))
    discovered_trace_dir = out_dir / "discovered_traces"
    round_summaries: list[dict[str, Any]] = []
    val_dir = Path(resolve_repo_path(str(args.val_trace_dir))) if str(args.val_trace_dir).strip() else None

    for round_idx in range(int(args.rounds)):
        round_seed = int(args.seed) + int(round_idx) * 10_000
        round_dir = out_dir / f"round_{int(round_idx):02d}"
        search_dir = round_dir / "search"
        round_dir.mkdir(parents=True, exist_ok=True)

        runtime = load_runtime(
            str(current_model_path),
            reference_window=int(args.reference_window),
            shield_rules_prefix=args.shield_rules_prefix,
            shield_risk_trigger_steps=int(args.shield_risk_trigger_steps),
            shield_cooldown_steps=int(args.shield_cooldown_steps),
            shield_empty_fallback=str(args.shield_empty_fallback),
        )
        scorer = ParkTraceScorer(runtime)
        generator = GenetBOTraceGenerator(
            scorer=scorer,
            objective=cast(ObjectiveName, str(args.objective)),
            search_space=search_space,
            seed=int(round_seed),
            export_traces=int(args.export_traces_per_round),
            out_dir=str(search_dir),
            traces_per_eval=int(args.traces_per_eval),
            init_points=int(args.bo_init_points),
            search_iters=int(args.bo_iter),
            candidate_pool_size=int(args.candidate_pool_size),
            overwrite=True,
        )
        search_result = generator.run()
        copied_traces = _copy_round_traces(
            src_dir=search_dir,
            dst_dir=discovered_trace_dir,
            round_idx=int(round_idx),
        )
        discovered_eval_before = scorer.score_trace_dir(
            trace_paths=_trace_paths(search_dir),
            objective=cast(ObjectiveName, str(args.objective)),
        )
        val_eval_before = None
        if val_dir is not None:
            val_eval_before = scorer.score_trace_dir(
                trace_paths=_trace_paths(val_dir, limit=int(args.val_trace_limit)),
                objective=cast(ObjectiveName, str(args.objective)),
            )

        curriculum = curriculum.add_point_component(
            params=search_result.best_params,
            add_probability=float(args.curriculum_add_prob),
        )
        curriculum.dump(round_dir / "curriculum_after_search.json")

        train_result: FineTuneResult = fine_tune_on_curriculum(
            model_path=str(current_model_path),
            curriculum=curriculum,
            out_dir=str(round_dir / "train"),
            seed=int(round_seed + 5_000),
            num_traces=int(args.training_trace_count),
            total_iters=int(args.updates_per_round),
            eval_every=int(args.eval_every),
            eval_episodes=int(args.eval_episodes),
        )
        current_model_path = str(train_result.model_path)
        post_scorer = ParkTraceScorer(
            load_runtime(
                str(current_model_path),
                reference_window=int(args.reference_window),
                shield_rules_prefix=args.shield_rules_prefix,
                shield_risk_trigger_steps=int(args.shield_risk_trigger_steps),
                shield_cooldown_steps=int(args.shield_cooldown_steps),
                shield_empty_fallback=str(args.shield_empty_fallback),
            )
        )
        discovered_eval_after = post_scorer.score_trace_dir(
            trace_paths=_trace_paths(search_dir),
            objective=cast(ObjectiveName, str(args.objective)),
        )
        val_eval_after = None
        if val_dir is not None:
            val_eval_after = post_scorer.score_trace_dir(
                trace_paths=_trace_paths(val_dir, limit=int(args.val_trace_limit)),
                objective=cast(ObjectiveName, str(args.objective)),
            )

        round_summary = {
            "round_idx": int(round_idx),
            "round_seed": int(round_seed),
            "start_model_path": str(runtime.checkpoint_path),
            "end_model_path": str(current_model_path),
            "best_search_target": float(search_result.best_target),
            "best_search_eval_id": int(search_result.best_eval_id),
            "best_params": asdict(search_result.best_params),
            "best_raw_vector": list(search_result.best_raw_vector),
            "copied_trace_count": int(copied_traces),
            "search_dir": str(search_dir),
            "train_result": {
                "model_path": str(train_result.model_path),
                "best_model_path": str(train_result.best_model_path),
                "trace_dir": str(train_result.trace_dir),
                "save_dir": str(train_result.save_dir),
                "stats": dict(train_result.stats),
            },
            "discovered_eval_before": dict(discovered_eval_before),
            "discovered_eval_after": dict(discovered_eval_after),
            "val_eval_before": None if val_eval_before is None else dict(val_eval_before),
            "val_eval_after": None if val_eval_after is None else dict(val_eval_after),
        }
        round_summaries.append(round_summary)
        _write_json(round_dir / "round_summary.json", round_summary)
        curriculum.dump(round_dir / "curriculum_after_train.json")

    _write_json(
        out_dir / "run_summary.json",
        {
            "config": config_payload,
            "final_model_path": str(current_model_path),
            "round_summaries": round_summaries,
        },
    )
    print(f"saved Genet run under: {out_dir}")


if __name__ == "__main__":
    main()
