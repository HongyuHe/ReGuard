#!/usr/bin/env python3
"""
Run the Sage Genet loop:
1. Genet BO discovery of a hard trace family.
2. Add the discovered family into a curriculum.
3. Sample training traces from the curriculum.
4. Fine-tune Sage offline on those traces.

Example usage:

time python scripts/run_sage_genet_loop.py \
  --config-path attacks/models/gap_adv_20260321_gap-constrained-all-loss_50ms_300k.config.json \
  --pretrained-model-dir sage_rl/rl_module/sage_models/final \
  --test-manifest attacks/test/manifest.json \
  --out-dir attacks/genet/gap-constrained-all-loss_50ms_300k \
  --rounds 5 \
  --num-discovery-traces 55 \
  --num-training-traces 55 \
  --traces-per-eval 10 \
  --segment-steps 8 \
  --curriculum-add-prob 0.3

Notes:
- Fine-tuning uses the dataset-size-aware auto budget from
  scripts/finetune_sage_on_adv_traces.py by default.
- Restarting the same command reuses completed rounds plus generated
  discovery/training manifests. Use --overwrite only when you intentionally
  want to regenerate everything.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


if __package__ in (None, ""):
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from genet import (
        SageGenetBOTraceGenerator,
        SharedScheduleCurriculum,
        SharedScheduleFamilyParams,
        infer_search_bounds,
        load_config_payload,
        repo_root_from_script,
        resolve_repo_path,
        sample_curriculum_trace_set,
        save_json,
        tf_checkpoint_exists,
    )
else:
    from genet import (
        SageGenetBOTraceGenerator,
        SharedScheduleCurriculum,
        SharedScheduleFamilyParams,
        infer_search_bounds,
        load_config_payload,
        repo_root_from_script,
        resolve_repo_path,
        sample_curriculum_trace_set,
        save_json,
        tf_checkpoint_exists,
    )


def _add_bool_flag(parser: argparse.ArgumentParser, *, name: str, default: bool, help_text: str) -> None:
    destination = str(name).replace("-", "_")
    parser.add_argument(f"--{name}", dest=destination, action="store_true", help=help_text)
    parser.add_argument(f"--no-{name}", dest=destination, action="store_false")
    parser.set_defaults(**{destination: bool(default)})


def _python_env(repo_root: str, extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath_items = [str(repo_root)]
    if env.get("PYTHONPATH"):
        pythonpath_items.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_items)
    if extra:
        env.update({str(key): str(value) for key, value in extra.items()})
    return env


def _run_command(cmd: list[str], *, cwd: str, env: dict[str, str] | None = None) -> None:
    print(f"\n$ cd {cwd}", flush=True)
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return dict(payload)


def _maybe_load_completed_round(
    *,
    round_dir: str,
    round_idx: int,
) -> dict[str, Any] | None:
    summary_path = os.path.join(round_dir, "round_summary.json")
    if not os.path.exists(summary_path):
        return None
    summary = _load_json(summary_path)
    model_out = str(summary.get("model_out", ""))
    if not model_out or not tf_checkpoint_exists(model_out):
        print(
            f"* round={int(round_idx)} has a summary but no reusable checkpoint; "
            "continuing from the first incomplete phase.",
            flush=True,
        )
        return None
    print(f"* round={int(round_idx)} already complete; reusing {model_out}", flush=True)
    return summary


def _load_discovery_manifest(path: str) -> dict[str, Any]:
    manifest = _load_json(path)
    best_params_payload = manifest.get("best_params")
    if not isinstance(best_params_payload, dict):
        raise ValueError(f"missing best_params in discovery manifest: {path}")
    return {
        "best_params": SharedScheduleFamilyParams(**dict(best_params_payload)),
        "best_target": float(manifest.get("best_target", 0.0)),
        "best_raw_params": dict(manifest.get("best_raw_params", {})),
        "generated_manifest_path": str(path),
    }


def _load_finished_finetune_summary(path: str) -> dict[str, Any] | None:
    if not os.path.exists(path):
        return None
    summary = _load_json(path)
    export_policy_dir = str(summary.get("export_policy_dir", ""))
    if export_policy_dir and tf_checkpoint_exists(export_policy_dir):
        return summary
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Genet-style BO discovery + curriculum-sampled offline CRR fine-tuning loop for Sage."
        )
    )
    parser.add_argument("--repo-root", type=str, default=repo_root_from_script(__file__))
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument(
        "--pretrained-model-dir",
        type=str,
        default="sage_rl/rl_module/sage_models/final",
        help="Initial Sage TensorFlow checkpoint dir.",
    )
    parser.add_argument("--test-manifest", type=str, default="attacks/test/manifest.json")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--runtime-dir", type=str, default="attacks/runtime")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bw-min-mbps", type=float, default=None)
    parser.add_argument("--bw-max-mbps", type=float, default=None)
    parser.add_argument("--loss-min", type=float, default=None)
    parser.add_argument("--loss-max", type=float, default=None)
    parser.add_argument("--segment-steps", type=float, default=8.0)
    _add_bool_flag(
        parser,
        name="optimize-segment-steps",
        default=False,
        help_text="Include segment_steps in the BO search space instead of fixing it.",
    )
    parser.add_argument("--segment-steps-min", type=float, default=1.0)
    parser.add_argument("--segment-steps-max", type=float, default=8.0)
    parser.add_argument("--traces-per-eval", type=int, default=10)
    parser.add_argument("--bo-init-points", type=int, default=10)
    parser.add_argument("--bo-iter", type=int, default=5)
    parser.add_argument("--curriculum-add-prob", type=float, default=0.3)
    parser.add_argument("--num-discovery-traces", type=int, default=0)
    parser.add_argument("--num-training-traces", type=int, default=0)
    parser.add_argument("--behavior-policy", type=str, default="best_baseline")
    parser.add_argument(
        "--num-training-steps",
        type=int,
        default=0,
        help=(
            "Fine-tuning learner steps per round. Use 0 to let "
            "finetune_sage_on_adv_traces.py derive a dataset-size-aware budget."
        ),
    )
    parser.add_argument(
        "--target-effective-epochs",
        type=float,
        default=64.0,
        help="Forwarded to fine-tuning when --num-training-steps=0.",
    )
    parser.add_argument(
        "--min-training-steps",
        type=int,
        default=32,
        help="Lower bound for auto-derived fine-tuning steps.",
    )
    parser.add_argument(
        "--max-training-steps",
        type=int,
        default=1024,
        help="Upper bound for auto-derived fine-tuning steps. Use 0 to disable the cap.",
    )
    parser.add_argument("--sequence-length", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--policy-lr", type=float, default=5e-6)
    parser.add_argument("--critic-lr", type=float, default=5e-5)
    parser.add_argument("--shuffle-buffer-size", type=int, default=10000)
    parser.add_argument("--block-length", type=int, default=1)
    parser.add_argument("--interleave-type", type=str, default="full", choices=["full", "none"])
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument(
        "--progress-interval-seconds",
        type=float,
        default=60.0,
        help="Forwarded to fine-tuning progress heartbeats.",
    )
    parser.add_argument(
        "--show-finetune-child-logs",
        action="store_true",
        help="Stream fine-tuning child logs instead of using the fine-tuner's quiet default.",
    )
    parser.add_argument("--eval-python", type=str, default=sys.executable)
    parser.add_argument(
        "--offline-python",
        type=str,
        default=str(Path.home() / "venvpy36" / "bin" / "python"),
    )
    parser.add_argument(
        "--finetune-extra-arg",
        action="append",
        default=[],
        help="Extra token forwarded verbatim to scripts/finetune_sage_on_adv_traces.py.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the whole loop output directory and regenerate all rounds from scratch.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = os.path.abspath(os.path.expanduser(args.repo_root))
    if int(args.rounds) <= 0:
        raise ValueError("rounds must be positive")

    out_dir = resolve_repo_path(repo_root, str(args.out_dir))
    if os.path.isdir(out_dir):
        if len(os.listdir(out_dir)) > 0 and bool(args.overwrite):
            shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    config_path, config_payload = load_config_payload(repo_root, str(args.config_path))
    search_bounds = infer_search_bounds(
        config_payload=config_payload,
        bw_min_mbps=args.bw_min_mbps,
        bw_max_mbps=args.bw_max_mbps,
        loss_min=args.loss_min,
        loss_max=args.loss_max,
        segment_steps_min=args.segment_steps_min if bool(args.optimize_segment_steps) else float(args.segment_steps),
        segment_steps_max=args.segment_steps_max if bool(args.optimize_segment_steps) else float(args.segment_steps),
    )
    curriculum = SharedScheduleCurriculum.from_search_space(bounds=search_bounds)
    curriculum.dump(os.path.join(out_dir, "curriculum_initial.json"))

    initial_model_dir = resolve_repo_path(repo_root, str(args.pretrained_model_dir))
    if not tf_checkpoint_exists(initial_model_dir):
        raise FileNotFoundError(f"missing TensorFlow checkpoint in pretrained model dir: {initial_model_dir}")

    run_config = {
        "config_path": str(config_path),
        "pretrained_model_dir": str(initial_model_dir),
        "test_manifest": str(args.test_manifest),
        "out_dir": str(out_dir),
        "runtime_dir": str(args.runtime_dir),
        "rounds": int(args.rounds),
        "seed": int(args.seed),
        "search_bounds": search_bounds.__dict__,
        "segment_steps": float(args.segment_steps),
        "optimize_segment_steps": bool(args.optimize_segment_steps),
        "traces_per_eval": int(args.traces_per_eval),
        "bo_init_points": int(args.bo_init_points),
        "bo_iter": int(args.bo_iter),
        "curriculum_add_prob": float(args.curriculum_add_prob),
        "num_discovery_traces": int(args.num_discovery_traces),
        "num_training_traces": int(args.num_training_traces),
        "behavior_policy": str(args.behavior_policy),
        "num_training_steps": int(args.num_training_steps),
        "target_effective_epochs": float(args.target_effective_epochs),
        "min_training_steps": int(args.min_training_steps),
        "max_training_steps": int(args.max_training_steps),
        "sequence_length": int(args.sequence_length),
        "batch_size": int(args.batch_size),
        "policy_lr": float(args.policy_lr),
        "critic_lr": float(args.critic_lr),
        "shuffle_buffer_size": int(args.shuffle_buffer_size),
        "block_length": int(args.block_length),
        "interleave_type": str(args.interleave_type),
        "tensorboard": bool(args.tensorboard),
        "progress_interval_seconds": float(args.progress_interval_seconds),
        "show_finetune_child_logs": bool(args.show_finetune_child_logs),
        "eval_python": str(args.eval_python),
        "offline_python": str(args.offline_python),
    }
    save_json(os.path.join(out_dir, "config.json"), run_config)

    current_model_dir = str(initial_model_dir)
    round_summaries: list[dict[str, Any]] = []
    for round_idx in range(int(args.rounds)):
        round_seed = int(args.seed) + int(round_idx) * 10_000
        round_dir = os.path.join(out_dir, f"round_{int(round_idx):02d}")
        discovery_dir = os.path.join(round_dir, "discovery_traces")
        training_dir = os.path.join(round_dir, "training_traces")
        finetune_dir = os.path.join(round_dir, "finetune")
        os.makedirs(round_dir, exist_ok=True)

        completed_round = None if bool(args.overwrite) else _maybe_load_completed_round(
            round_dir=round_dir,
            round_idx=int(round_idx),
        )
        if completed_round is not None:
            curriculum_path = str(completed_round.get("curriculum_path", ""))
            if curriculum_path and os.path.exists(curriculum_path):
                curriculum = SharedScheduleCurriculum.load(curriculum_path)
            else:
                best_params_payload = completed_round.get("best_params")
                if not isinstance(best_params_payload, dict):
                    raise ValueError(
                        f"completed round {int(round_idx)} is missing both curriculum_path and best_params"
                    )
                curriculum = curriculum.add_point_component(
                    params=SharedScheduleFamilyParams(**dict(best_params_payload)),
                    add_probability=float(args.curriculum_add_prob),
                )
            round_summaries.append(completed_round)
            current_model_dir = str(completed_round["model_out"])
            continue

        discovery_manifest_path = os.path.join(discovery_dir, "generated_manifest.json")
        if os.path.exists(discovery_manifest_path) and not bool(args.overwrite):
            print(f"* round={int(round_idx)} reusing discovery manifest: {discovery_manifest_path}", flush=True)
            discovery_result = _load_discovery_manifest(discovery_manifest_path)
        else:
            discovery_generator = SageGenetBOTraceGenerator(
                repo_root=repo_root,
                config_path=config_path,
                config_payload=config_payload,
                sage_model_dir=current_model_dir,
                test_manifest_path=str(args.test_manifest),
                out_dir=discovery_dir,
                runtime_dir=str(args.runtime_dir),
                bounds=search_bounds,
                seed=int(round_seed),
                num_traces=int(args.num_discovery_traces),
                traces_per_eval=int(args.traces_per_eval),
                bo_init_points=int(args.bo_init_points),
                bo_iter=int(args.bo_iter),
                optimize_segment_steps=bool(args.optimize_segment_steps),
                fixed_segment_steps=float(args.segment_steps),
                overwrite=True,
            )
            raw_discovery_result = discovery_generator.run()
            discovery_result = {
                "best_params": raw_discovery_result.best_params,
                "best_target": float(raw_discovery_result.best_target),
                "best_raw_params": dict(raw_discovery_result.best_raw_params),
                "generated_manifest_path": str(raw_discovery_result.generated_manifest_path),
            }

        curriculum = curriculum.add_point_component(
            params=discovery_result["best_params"],
            add_probability=float(args.curriculum_add_prob),
        )
        curriculum_path = os.path.join(round_dir, "curriculum.json")
        curriculum.dump(curriculum_path)

        training_manifest_path = os.path.join(training_dir, "generated_manifest.json")
        if os.path.exists(training_manifest_path) and not bool(args.overwrite):
            print(f"* round={int(round_idx)} reusing training manifest: {training_manifest_path}", flush=True)
        else:
            training_manifest_path = sample_curriculum_trace_set(
                repo_root=repo_root,
                config_path=config_path,
                config_payload=config_payload,
                sage_model_dir=current_model_dir,
                test_manifest_path=str(args.test_manifest),
                out_dir=training_dir,
                runtime_dir=str(args.runtime_dir),
                curriculum=curriculum,
                seed=int(round_seed) + 1,
                num_traces=int(args.num_training_traces),
                overwrite=True,
            )

        finetune_cmd = [
            str(args.eval_python),
            os.path.join(repo_root, "scripts", "finetune_sage_on_adv_traces.py"),
            "--generated-manifest",
            str(training_manifest_path),
            "--pretrained-model-dir",
            str(current_model_dir),
            "--out-dir",
            str(finetune_dir),
            "--behavior-policy",
            str(args.behavior_policy),
            "--target-effective-epochs",
            str(float(args.target_effective_epochs)),
            "--min-training-steps",
            str(int(args.min_training_steps)),
            "--max-training-steps",
            str(int(args.max_training_steps)),
            "--sequence-length",
            str(int(args.sequence_length)),
            "--batch-size",
            str(int(args.batch_size)),
            "--policy-lr",
            str(float(args.policy_lr)),
            "--critic-lr",
            str(float(args.critic_lr)),
            "--shuffle-buffer-size",
            str(int(args.shuffle_buffer_size)),
            "--block-length",
            str(int(args.block_length)),
            "--interleave-type",
            str(args.interleave_type),
            "--eval-python",
            str(args.eval_python),
            "--offline-python",
            str(args.offline_python),
            "--progress-interval-seconds",
            str(float(args.progress_interval_seconds)),
        ]
        if int(args.num_training_steps) > 0:
            finetune_cmd.extend(
                [
                    "--num-training-steps",
                    str(int(args.num_training_steps)),
                ]
            )
        if bool(args.tensorboard):
            finetune_cmd.append("--tensorboard")
        if bool(args.show_finetune_child_logs):
            finetune_cmd.append("--show-child-logs")
        finetune_cmd.extend([str(token) for token in list(args.finetune_extra_arg)])

        finetune_summary_path = os.path.join(finetune_dir, "finetune_summary.json")
        finetune_summary = None if bool(args.overwrite) else _load_finished_finetune_summary(finetune_summary_path)
        if finetune_summary is not None:
            print(f"* round={int(round_idx)} reusing fine-tuned model: {finetune_summary['export_policy_dir']}", flush=True)
        else:
            _run_command(
                finetune_cmd,
                cwd=repo_root,
                env=_python_env(repo_root),
            )
            finetune_summary = _load_json(finetune_summary_path)
        next_model_dir = str(finetune_summary["export_policy_dir"])
        if not tf_checkpoint_exists(next_model_dir):
            raise RuntimeError(f"fine-tune did not export a TensorFlow checkpoint dir: {next_model_dir}")
        round_summary = {
            "round_idx": int(round_idx),
            "round_seed": int(round_seed),
            "model_in": str(current_model_dir),
            "model_out": str(next_model_dir),
            "best_target": float(discovery_result["best_target"]),
            "best_params": asdict(discovery_result["best_params"]),
            "best_raw_params": dict(discovery_result["best_raw_params"]),
            "discovery_generated_manifest": str(discovery_result["generated_manifest_path"]),
            "training_generated_manifest": str(training_manifest_path),
            "curriculum_path": str(curriculum_path),
            "finetune_summary_path": str(finetune_summary_path),
        }
        round_summaries.append(round_summary)
        save_json(os.path.join(round_dir, "round_summary.json"), round_summary)
        current_model_dir = next_model_dir
        print(
            f"* round={int(round_idx)} target={float(discovery_result['best_target']):.6f} "
            f"model_out={current_model_dir}"
        )

    final_model_dir = os.path.join(out_dir, "final_model")
    if os.path.isdir(final_model_dir):
        shutil.rmtree(final_model_dir)
    shutil.copytree(current_model_dir, final_model_dir)
    curriculum.dump(os.path.join(out_dir, "curriculum_final.json"))
    save_json(
        os.path.join(out_dir, "summary.json"),
        {
            "config": run_config,
            "initial_model_dir": str(initial_model_dir),
            "final_model_dir": str(final_model_dir),
            "rounds": round_summaries,
        },
    )
    print(f"* Final model dir: {final_model_dir}")
    print(f"* Summary: {os.path.join(out_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
