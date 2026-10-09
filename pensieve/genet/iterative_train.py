from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import asdict
from typing import Any

from .bo_search import GenetBOTraceGenerator
from .cooked_trace_io import list_cooked_trace_files
from .curriculum import BandwidthCurriculum
from .generate_traces import DEFAULT_BW_CHANGE_INTERVAL, DEFAULT_BW_MAX_MBPS, DEFAULT_BW_MIN_MBPS
from .retrainer import FineTuneResult, LoadedTraceSet, fine_tune_on_curriculum, load_trace_set
from .runtime import load_runtime, resolve_repo_path
from .scoring import TraceScorer
from .trace_family import BandwidthTraceParams, CookedTrace, DEFAULT_DURATION_S


def _write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    with open(str(path), "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _prepare_out_dir(out_dir: str, *, overwrite: bool) -> None:
    if os.path.isdir(str(out_dir)):
        has_entries = len(os.listdir(str(out_dir))) > 0
        if has_entries and not bool(overwrite):
            raise RuntimeError(f"* output dir is not empty: {out_dir}. Re-run with --overwrite.")
        if has_entries and bool(overwrite):
            shutil.rmtree(str(out_dir))
    os.makedirs(str(out_dir), exist_ok=True)


def _trace_to_cooked(trace_time: list[float], trace_bw: list[float], *, trace_seed: int) -> CookedTrace:
    bw_floor = max(min([float(x) for x in trace_bw]), 1e-6)
    bw_ceil = max(max([float(x) for x in trace_bw]), float(bw_floor))
    params = BandwidthTraceParams(
        min_bw_mbps=float(bw_floor),
        max_bw_mbps=float(bw_ceil),
        bw_change_interval=1.0,
    ).normalized(
        bw_floor_mbps=float(bw_floor),
        bw_ceil_mbps=float(bw_ceil),
    )
    return CookedTrace(
        timestamps_s=[float(x) for x in trace_time],
        bandwidths_mbps=[float(x) for x in trace_bw],
        params=params,
        trace_seed=int(trace_seed),
    )


def _evaluate_trace_set(
    *,
    model_path: str,
    trace_set: LoadedTraceSet,
    objective: str,
    oracle_k: int,
    seed: int,
) -> dict[str, float]:
    runtime = load_runtime(str(model_path))
    scorer = TraceScorer(runtime, oracle_k=int(oracle_k))

    gaps: list[float] = []
    pensieve_qoes: list[float] = []
    baseline_qoes: list[float] = []
    for idx, (trace_time, trace_bw) in enumerate(zip(trace_set.cooked_time, trace_set.cooked_bw)):
        trace = _trace_to_cooked(
            [float(x) for x in trace_time],
            [float(x) for x in trace_bw],
            trace_seed=int(seed) + int(idx),
        )
        score = scorer.score_trace(
            trace,
            objective=str(objective),
            seed=int(seed) + int(idx),
        )
        gaps.append(float(score.gap))
        pensieve_qoes.append(float(score.pensieve_qoe))
        baseline_qoes.append(float(score.baseline_qoe))

    if len(gaps) == 0:
        return {
            "eval/num_traces": 0.0,
            "eval/gap_mean": float("nan"),
            "eval/pensieve_qoe_mean": float("nan"),
            "eval/baseline_qoe_mean": float("nan"),
        }

    import numpy as np

    return {
        "eval/num_traces": float(int(len(gaps))),
        "eval/gap_mean": float(np.mean(np.asarray(gaps, dtype=np.float64))),
        "eval/pensieve_qoe_mean": float(np.mean(np.asarray(pensieve_qoes, dtype=np.float64))),
        "eval/baseline_qoe_mean": float(np.mean(np.asarray(baseline_qoes, dtype=np.float64))),
    }


def _aggregate_round_traces(*, round_trace_dir: str, aggregate_dir: str, round_idx: int) -> int:
    os.makedirs(str(aggregate_dir), exist_ok=True)
    trace_files = list_cooked_trace_files(str(round_trace_dir))
    for trace_name in trace_files:
        src = os.path.join(str(round_trace_dir), str(trace_name))
        dst = os.path.join(str(aggregate_dir), f"round_{int(round_idx):02d}_{str(trace_name)}")
        shutil.copyfile(str(src), str(dst))
    return int(len(trace_files))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Genet iterative BO + retraining loop for Pensieve.")
    parser.add_argument("--model", type=str, default="ppo/nn_model_ep_426300.pth")
    parser.add_argument(
        "--objective",
        type=str,
        default="mpc_gap",
        choices=["mpc_gap", "oracle_gap"],
        help="BO objective used to discover hard environments before each retraining round.",
    )
    parser.add_argument("--oracle-k", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bw-min", type=float, default=DEFAULT_BW_MIN_MBPS)
    parser.add_argument("--bw-max", type=float, default=DEFAULT_BW_MAX_MBPS)
    parser.add_argument("--duration-s", type=float, default=DEFAULT_DURATION_S)
    parser.add_argument("--bw-change-interval", type=float, default=DEFAULT_BW_CHANGE_INTERVAL)
    parser.add_argument(
        "--optimize-bw-change-interval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include bw_change_interval in BO. Disabled by default to match the current setup.",
    )
    parser.add_argument("--bw-change-interval-min", type=float, default=1.0)
    parser.add_argument("--bw-change-interval-max", type=float, default=100.0)
    parser.add_argument("--traces-per-eval", type=int, default=10)
    parser.add_argument("--bo-init-points", type=int, default=10)
    parser.add_argument("--bo-iter", type=int, default=5)
    parser.add_argument("--curriculum-add-prob", type=float, default=0.3)
    parser.add_argument("--updates-per-round", type=int, default=5000)
    parser.add_argument("--batch-episodes", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--init-entropy-weight", type=float, default=None)
    parser.add_argument(
        "--freeze-entropy-weight",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--export-traces-per-round", type=int, default=10)
    parser.add_argument("--real-trace-dir", type=str, default="")
    parser.add_argument("--real-trace-prob", type=float, default=0.0)
    parser.add_argument("--real-trace-num", type=int, default=0)
    parser.add_argument("--val-trace-dir", type=str, default="")
    parser.add_argument("--val-trace-num", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="genet/runs/mpc_gap")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if int(args.rounds) <= 0:
        raise ValueError("* rounds must be positive")
    if int(args.updates_per_round) <= 0:
        raise ValueError("* updates_per_round must be positive")
    if int(args.export_traces_per_round) <= 0:
        raise ValueError("* export_traces_per_round must be positive")

    out_dir = resolve_repo_path(str(args.out_dir))
    if str(args.out_dir) == "genet/runs/mpc_gap" and str(args.objective) != "mpc_gap":
        out_dir = resolve_repo_path(os.path.join("genet", "runs", str(args.objective)))
    _prepare_out_dir(str(out_dir), overwrite=bool(args.overwrite))

    initial_model_path = resolve_repo_path(str(args.model))
    discovered_trace_dir = os.path.join(str(out_dir), "discovered_traces")
    os.makedirs(str(discovered_trace_dir), exist_ok=True)

    curriculum = BandwidthCurriculum.from_search_space(
        bw_floor_mbps=float(args.bw_min),
        bw_ceil_mbps=float(args.bw_max),
        bw_change_interval=float(args.bw_change_interval),
        duration_s=float(args.duration_s),
    )
    curriculum.dump(os.path.join(str(out_dir), "curriculum_initial.json"))

    real_trace_set = None
    if str(args.real_trace_dir).strip():
        real_trace_dir = resolve_repo_path(str(args.real_trace_dir))
        real_trace_set = load_trace_set(str(real_trace_dir), limit=int(args.real_trace_num))

    val_trace_set = None
    if str(args.val_trace_dir).strip():
        val_trace_dir = resolve_repo_path(str(args.val_trace_dir))
        val_trace_set = load_trace_set(str(val_trace_dir), limit=int(args.val_trace_num))

    run_config = {
        "model": str(initial_model_path),
        "objective": str(args.objective),
        "oracle_k": int(args.oracle_k),
        "rounds": int(args.rounds),
        "seed": int(args.seed),
        "bw_min": float(args.bw_min),
        "bw_max": float(args.bw_max),
        "duration_s": float(args.duration_s),
        "bw_change_interval": float(args.bw_change_interval),
        "optimize_bw_change_interval": bool(args.optimize_bw_change_interval),
        "bw_change_interval_min": float(args.bw_change_interval_min),
        "bw_change_interval_max": float(args.bw_change_interval_max),
        "traces_per_eval": int(args.traces_per_eval),
        "bo_init_points": int(args.bo_init_points),
        "bo_iter": int(args.bo_iter),
        "curriculum_add_prob": float(args.curriculum_add_prob),
        "updates_per_round": int(args.updates_per_round),
        "batch_episodes": int(args.batch_episodes),
        "learning_rate": float(args.learning_rate),
        "init_entropy_weight": None if args.init_entropy_weight is None else float(args.init_entropy_weight),
        "freeze_entropy_weight": bool(args.freeze_entropy_weight),
        "episodes_per_round": int(args.updates_per_round) * int(args.batch_episodes),
        "export_traces_per_round": int(args.export_traces_per_round),
        "real_trace_dir": str(args.real_trace_dir),
        "real_trace_prob": float(args.real_trace_prob),
        "real_trace_num": int(args.real_trace_num),
        "val_trace_dir": str(args.val_trace_dir),
        "val_trace_num": int(args.val_trace_num),
        "out_dir": str(out_dir),
    }
    _write_json(os.path.join(str(out_dir), "config.json"), run_config)

    current_model_path = str(initial_model_path)
    round_summaries: list[dict[str, Any]] = []
    for round_idx in range(int(args.rounds)):
        round_seed = int(args.seed) + int(round_idx) * 10_000
        round_dir = os.path.join(str(out_dir), f"round_{int(round_idx):02d}")
        bo_dir = os.path.join(str(round_dir), "bo_search")
        os.makedirs(str(round_dir), exist_ok=True)

        scorer = TraceScorer(load_runtime(str(current_model_path)), oracle_k=int(args.oracle_k))
        generator = GenetBOTraceGenerator(
            scorer=scorer,
            objective=str(args.objective),
            seed=int(round_seed),
            episodes=int(args.export_traces_per_round),
            out_dir=str(bo_dir),
            bw_min_mbps=float(args.bw_min),
            bw_max_mbps=float(args.bw_max),
            bw_change_interval=float(args.bw_change_interval),
            optimize_bw_change_interval=bool(args.optimize_bw_change_interval),
            bw_change_interval_min=float(args.bw_change_interval_min),
            bw_change_interval_max=float(args.bw_change_interval_max),
            duration_s=float(args.duration_s),
            traces_per_eval=int(args.traces_per_eval),
            bo_init_points=int(args.bo_init_points),
            bo_iter=int(args.bo_iter),
            overwrite=True,
        )
        search_result = generator.run()

        round_trace_set = load_trace_set(str(bo_dir), limit=int(args.export_traces_per_round))
        pre_eval = _evaluate_trace_set(
            model_path=str(current_model_path),
            trace_set=round_trace_set,
            objective=str(args.objective),
            oracle_k=int(args.oracle_k),
            seed=int(round_seed) + 97,
        )
        exported_trace_count = _aggregate_round_traces(
            round_trace_dir=str(bo_dir),
            aggregate_dir=str(discovered_trace_dir),
            round_idx=int(round_idx),
        )

        curriculum = curriculum.add_point_component(
            params=search_result.best_params,
            add_probability=float(args.curriculum_add_prob),
            duration_s=float(args.duration_s),
        )
        curriculum_path = os.path.join(str(round_dir), "curriculum.json")
        curriculum.dump(str(curriculum_path))

        round_model_out = os.path.join(str(round_dir), f"model_round_{int(round_idx):02d}.pth")
        retrain_result: FineTuneResult = fine_tune_on_curriculum(
            model_path=str(current_model_path),
            out_model_path=str(round_model_out),
            curriculum=curriculum,
            episodes=int(args.updates_per_round) * int(args.batch_episodes),
            batch_episodes=int(args.batch_episodes),
            learning_rate=float(args.learning_rate),
            seed=int(round_seed) + 1,
            real_trace_set=real_trace_set,
            real_trace_prob=float(args.real_trace_prob),
            init_entropy_weight=None if args.init_entropy_weight is None else float(args.init_entropy_weight),
            freeze_entropy_weight=bool(args.freeze_entropy_weight),
        )

        post_eval = _evaluate_trace_set(
            model_path=str(retrain_result.model_path),
            trace_set=round_trace_set,
            objective=str(args.objective),
            oracle_k=int(args.oracle_k),
            seed=int(round_seed) + 211,
        )
        val_eval = None
        if val_trace_set is not None:
            val_eval = _evaluate_trace_set(
                model_path=str(retrain_result.model_path),
                trace_set=val_trace_set,
                objective=str(args.objective),
                oracle_k=int(args.oracle_k),
                seed=int(round_seed) + 307,
            )

        round_summary = {
            "round_idx": int(round_idx),
            "round_seed": int(round_seed),
            "model_in": str(current_model_path),
            "model_out": str(retrain_result.model_path),
            "best_target": float(search_result.best_target),
            "best_params": asdict(search_result.best_params),
            "best_raw_params": dict(search_result.best_raw_params),
            "exported_trace_count": int(exported_trace_count),
            "pre_retrain_eval": dict(pre_eval),
            "post_retrain_eval": dict(post_eval),
            "train_stats": dict(retrain_result.stats),
            "curriculum_path": str(curriculum_path),
        }
        if val_eval is not None:
            round_summary["val_eval"] = dict(val_eval)
        round_summaries.append(round_summary)
        _write_json(os.path.join(str(round_dir), "round_summary.json"), round_summary)

        current_model_path = str(retrain_result.model_path)
        print(
            f"* round={int(round_idx)} target={float(search_result.best_target):.6f} "
            f"pre_gap={float(pre_eval['eval/gap_mean']):.6f} "
            f"post_gap={float(post_eval['eval/gap_mean']):.6f} "
            f"model={current_model_path}"
        )

    final_model_path = os.path.join(str(out_dir), "final_model.pth")
    shutil.copyfile(str(current_model_path), str(final_model_path))
    curriculum.dump(os.path.join(str(out_dir), "curriculum_final.json"))

    summary_payload = {
        "config": run_config,
        "initial_model_path": str(initial_model_path),
        "final_model_path": str(final_model_path),
        "discovered_trace_dir": str(discovered_trace_dir),
        "rounds": round_summaries,
    }
    _write_json(os.path.join(str(out_dir), "summary.json"), summary_payload)

    print(f"* Final model: {final_model_path}")
    print(f"* Discovered traces: {discovered_trace_dir}")
    print(f"* Summary: {os.path.join(str(out_dir), 'summary.json')}")


if __name__ == "__main__":
    main()
