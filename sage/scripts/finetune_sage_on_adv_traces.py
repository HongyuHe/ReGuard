#!/usr/bin/env python3
"""
Fine-tune the pretrained Sage policy on generated adversarial traces.

Example usage:

time python scripts/finetune_sage_on_adv_traces.py \
  --generated-manifest attacks/adv_traces/mixed/generated_manifest.json \
  --pretrained-model-dir sage_rl/rl_module/sage_models/final \
  --out-dir attacks/output/sage-finetune/mixed

time python scripts/finetune_sage_on_adv_traces.py \
  --generated-manifest attacks/train/manifest.json \
  --pretrained-model-dir sage_rl/rl_module/sage_models/final \
  --out-dir attacks/output/sage-finetune/clean

time python scripts/finetune_sage_on_adv_traces.py \
  --generated-manifest attacks/adv_traces/gap-constrained-all-loss_50ms_300k/generated_manifest.json \
  --pretrained-model-dir sage_rl/rl_module/sage_models/final \
  --out-dir attacks/output/sage-finetune/gap-constrained-all-loss_50ms_300k

time python scripts/finetune_sage_on_adv_traces.py \
  --generated-manifest attacks/adv_traces/hotnets19-loss_50ms_300k/generated_manifest.json \
  --pretrained-model-dir sage_rl/rl_module/sage_models/final \
  --out-dir attacks/output/sage-finetune/hotnets19-loss_50ms_300k
  
time python scripts/finetune_sage_on_adv_traces.py \
  --generated-manifest attacks/adv_traces/pgd_gap-constrained-all-loss_50ms_300k/generated_manifest.json \
  --pretrained-model-dir sage_rl/rl_module/sage_models/final \
  --out-dir attacks/output/sage-finetune/pgd_gap-constrained-all-loss_50ms_300k

time python scripts/finetune_sage_on_adv_traces.py \
  --generated-manifest attacks/adv_traces/random_gap-constrained-all-loss_50ms_300k/generated_manifest.json \
  --pretrained-model-dir sage_rl/rl_module/sage_models/final \
  --out-dir attacks/output/sage-finetune/random_gap-constrained-all-loss_50ms_300k

time python scripts/finetune_sage_on_adv_traces.py \
  --generated-manifest attacks/adv_traces/random_gap-constrained-all-loss_50ms_300k/generated_manifest.json \
  --pretrained-model-dir sage_rl/rl_module/sage_models/final \
  --out-dir attacks/output/sage-finetune/genet_gap-constrained-all-loss_50ms_300k

The script first replays generated traces and collects an offline dataset in the
original Sage 69-D observation format using a chosen behavior policy (best baseline
by default), then launches the existing offline CRR learner from that dataset while
initializing the policy from the pretrained checkpoint. Unless
--num-training-steps is set explicitly, the training budget is derived from the
dataset size so small offline datasets do not trigger multi-day CRR runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

try:
    import yaml as _pyyaml
except ModuleNotFoundError:
    _pyyaml = None
    from ruamel.yaml import YAML

    _ruamel_yaml = YAML()
else:
    _ruamel_yaml = None


DEFAULT_CLEAN_COLLECTION_CONFIG = "attacks/models/gap_adv_20260320_gap-constrained-all_50ms_300k.config.json"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_path(repo_root: Path, path: str | os.PathLike[str]) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (repo_root / candidate).resolve()


def _run_command(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    dry_run: bool = False,
    progress_label: str | None = None,
    progress_interval_seconds: float = 60.0,
    progress_fn: Any | None = None,
    log_path: Path | None = None,
) -> None:
    printable = " ".join(cmd)
    print(f"\n$ cd {cwd}", flush=True)
    print(f"$ {printable}", flush=True)
    if dry_run:
        return
    log_file = None
    popen_kwargs: dict[str, Any] = {
        "cwd": str(cwd),
        "env": env,
    }
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("a", encoding="utf-8")
        popen_kwargs.update(
            {
                "stdout": log_file,
                "stderr": subprocess.STDOUT,
            }
        )
    process = subprocess.Popen(cmd, **popen_kwargs)
    started_at = time.time()
    last_report_at = started_at
    label = str(progress_label or Path(cmd[0]).name)
    print(f"[{label}] started (pid={process.pid})", flush=True)
    if log_path is not None:
        print(f"[{label}] child output -> {log_path}", flush=True)
    poll_interval = 1.0 if float(progress_interval_seconds) > 0 else 5.0
    try:
        while True:
            return_code = process.poll()
            now = time.time()
            if return_code is not None:
                elapsed = _format_duration(now - started_at)
                if int(return_code) != 0:
                    if log_path is not None:
                        print(f"[{label}] failed; see child log at {log_path}", flush=True)
                    raise subprocess.CalledProcessError(int(return_code), cmd)
                print(f"[{label}] finished in {elapsed}", flush=True)
                return
            if float(progress_interval_seconds) > 0 and (now - last_report_at) >= float(progress_interval_seconds):
                details = ""
                if progress_fn is not None:
                    try:
                        details = str(progress_fn() or "").strip()
                    except Exception as exc:
                        details = f"progress unavailable ({exc})"
                message = f"[{label}] still running after {_format_duration(now - started_at)}"
                if details:
                    message += f" | {details}"
                print(message, flush=True)
                last_report_at = now
            time.sleep(poll_interval)
    finally:
        if log_file is not None:
            log_file.close()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _yaml_load(file_obj) -> dict[str, Any]:
    if _pyyaml is not None:
        payload = _pyyaml.safe_load(file_obj)
    else:
        payload = _ruamel_yaml.load(file_obj)
    if not isinstance(payload, dict):
        raise ValueError("expected YAML mapping")
    return payload


def _yaml_dump(payload: dict[str, Any], file_obj) -> None:
    if _pyyaml is not None:
        _pyyaml.safe_dump(payload, file_obj, sort_keys=False)
    else:
        _ruamel_yaml.dump(payload, file_obj)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, sort_keys=True)


def _format_duration(total_seconds: float) -> str:
    seconds = max(0, int(round(float(total_seconds))))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes > 0:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _manifest_kind(manifest_path: Path) -> str:
    payload = _load_json(manifest_path)
    if isinstance(payload.get("generated_entries"), list) and payload.get("generated_entries"):
        return "generated"
    if isinstance(payload.get("entries"), list) and payload.get("entries"):
        return "clean"
    raise ValueError(
        f"manifest must contain either non-empty generated_entries or entries: {manifest_path}"
    )


def _collection_manifest(
    *,
    manifest_path: Path,
    out_dir: Path,
    num_traces: int | None,
    manifest_kind: str,
) -> Path:
    if num_traces is None or int(num_traces) <= 0:
        return manifest_path
    payload = _load_json(manifest_path)
    entries_key = "generated_entries" if str(manifest_kind) == "generated" else "entries"
    entries = list(payload.get(entries_key, []))
    if not entries:
        raise ValueError(f"manifest has no {entries_key}: {manifest_path}")
    limited_payload = {
        **payload,
        entries_key: entries[: int(num_traces)],
        "source_manifest": str(manifest_path),
        "fine_tune_limited_num_traces": int(num_traces),
    }
    limited_manifest = out_dir / "collection_manifest.json"
    _write_json(limited_manifest, limited_payload)
    return limited_manifest


def _dataset_files(dataset_dir: Path) -> list[Path]:
    if not dataset_dir.exists():
        return []
    return sorted(path for path in dataset_dir.rglob("*.txt") if path.is_file())


def _dataset_row_count(dataset_dir: Path) -> int:
    total = 0
    for path in _dataset_files(dataset_dir):
        with path.open("r", encoding="utf-8") as file_obj:
            line_count = sum(1 for _ in file_obj)
        total += max(line_count - 1, 0)
    return total


def _dataset_sequence_count(dataset_dir: Path, sequence_length: int) -> int:
    if int(sequence_length) <= 0:
        return 0
    total = 0
    for path in _dataset_files(dataset_dir):
        with path.open("r", encoding="utf-8") as file_obj:
            line_count = sum(1 for _ in file_obj)
        total += max((line_count - 1) // int(sequence_length), 0)
    return total


def _dataset_size_bytes(dataset_dir: Path) -> int:
    return int(sum(path.stat().st_size for path in _dataset_files(dataset_dir)))


def _collection_progress_message(dataset_dir: Path) -> str:
    file_count = len(_dataset_files(dataset_dir))
    size_mb = _dataset_size_bytes(dataset_dir) / (1024.0 * 1024.0)
    return f"offline dataset files={file_count}, size={size_mb:.1f} MB"


def _latest_training_log_path(offline_run_dir: Path) -> Path | None:
    log_paths = sorted(
        (offline_run_dir / "models" / "crr_learner").glob("*/logs/logs.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return log_paths[0] if log_paths else None


def _last_csv_row(path: Path) -> list[str] | None:
    if not path.exists():
        return None
    last_row: list[str] | None = None
    with path.open("r", encoding="utf-8") as file_obj:
        reader = csv.reader(file_obj)
        for row in reader:
            if row:
                last_row = row
    return last_row


def _last_nonempty_line(path: Path, max_chars: int = 160) -> str:
    if not path.exists():
        return ""
    last_line = ""
    with path.open("r", encoding="utf-8", errors="replace") as file_obj:
        for raw_line in file_obj:
            line = raw_line.strip()
            if line:
                last_line = line
    if len(last_line) > int(max_chars):
        return f"{last_line[:max_chars]}..."
    return last_line


def _training_progress_message(offline_run_dir: Path, total_steps: int, child_log_path: Path | None = None) -> str:
    log_path = _latest_training_log_path(offline_run_dir)
    if log_path is None:
        if child_log_path is not None:
            last_line = _last_nonempty_line(child_log_path)
            if last_line:
                return f"waiting for learner log initialization | latest child log: {last_line}"
        return "waiting for learner log initialization"
    last_row = _last_csv_row(log_path)
    if not last_row or last_row[0] == "LR_steps":
        if child_log_path is not None:
            last_line = _last_nonempty_line(child_log_path)
            if last_line:
                return f"waiting for first learner step | latest child log: {last_line}"
        return "waiting for first learner step"
    step = int(float(last_row[0]))
    progress_pct = (100.0 * float(step) / float(max(total_steps, 1)))
    return f"learner_step={step}/{total_steps} ({progress_pct:.1f}%), log={log_path.parent.parent.name}"


def _latest_logged_training_step(offline_run_dir: Path) -> int:
    log_path = _latest_training_log_path(offline_run_dir)
    if log_path is None:
        return 0
    last_row = _last_csv_row(log_path)
    if not last_row or last_row[0] == "LR_steps":
        return 0
    try:
        return int(float(last_row[0]))
    except Exception:
        return 0


def _per_run_training_progress_message(
    offline_run_dir: Path,
    total_steps: int,
    start_step: int,
    child_log_path: Path | None = None,
) -> str:
    base_message = _training_progress_message(
        offline_run_dir,
        total_steps,
        child_log_path,
    )
    log_path = _latest_training_log_path(offline_run_dir)
    if log_path is None:
        return base_message
    last_row = _last_csv_row(log_path)
    if not last_row or last_row[0] == "LR_steps":
        return base_message
    try:
        absolute_step = int(float(last_row[0]))
    except Exception:
        return base_message
    run_step = max(0, absolute_step - int(start_step))
    progress_pct = 100.0 * float(run_step) / float(max(total_steps, 1))
    return (
        f"run_step={run_step}/{total_steps} ({progress_pct:.1f}%), "
        f"absolute_learner_step={absolute_step}, log={log_path.parent.parent.name}"
    )


def _archive_path(path: Path, archive_dir: Path) -> Path:
    archive_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    destination = archive_dir / f"{path.name}.previous-{timestamp}"
    if destination.exists():
        suffix_counter = 1
        while True:
            candidate = archive_dir / f"{path.name}.previous-{timestamp}-{suffix_counter}"
            if not candidate.exists():
                destination = candidate
                break
            suffix_counter += 1
    shutil.move(str(path), str(destination))
    return destination


def _line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as file_obj:
        return sum(1 for _ in file_obj)


def _quarantine_path(rejected_dir: Path, source_path: Path) -> Path:
    rejected_dir.mkdir(parents=True, exist_ok=True)
    candidate = rejected_dir / source_path.name
    if not candidate.exists():
        return candidate
    stem = source_path.stem
    suffix = source_path.suffix
    counter = 1
    while True:
        candidate = rejected_dir / f"{stem}-{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def _sanitize_dataset_files(dataset_dir: Path, rejected_dir: Path, *, dry_run: bool) -> dict[str, int]:
    kept = 0
    rejected_empty = 0
    rejected_header_only = 0
    rejected_unreadable = 0
    for path in _dataset_files(dataset_dir):
        reject_reason = ""
        try:
            lines = _line_count(path)
        except Exception:
            lines = 0
            reject_reason = "unreadable"
        if not reject_reason and lines <= 0:
            reject_reason = "empty"
        elif not reject_reason and lines <= 1:
            reject_reason = "header_only"

        if not reject_reason:
            kept += 1
            continue

        if reject_reason == "empty":
            rejected_empty += 1
        elif reject_reason == "header_only":
            rejected_header_only += 1
        else:
            rejected_unreadable += 1
        destination = _quarantine_path(rejected_dir, path)
        print(f"Quarantining {reject_reason} dataset file: {path} -> {destination}")
        if not dry_run:
            shutil.move(str(path), str(destination))

    return {
        "kept_files": int(kept),
        "rejected_empty_files": int(rejected_empty),
        "rejected_header_only_files": int(rejected_header_only),
        "rejected_unreadable_files": int(rejected_unreadable),
    }


def _prepare_offline_config(
    *,
    template_path: Path,
    out_path: Path,
    pretrained_model_dir: Path,
    export_policy_dir: Path,
    args: argparse.Namespace,
    resolved_batch_size: int,
    resolved_num_training_steps: int,
) -> Path:
    with template_path.open("r", encoding="utf-8") as file_obj:
        config = _yaml_load(file_obj)
    if not isinstance(config, dict):
        raise ValueError(f"expected YAML mapping in {template_path}")
    offline_config = config.setdefault("offline_config", {})
    offline_config["use_offline_model"] = True
    offline_config["load_ckpt_dir"] = str(pretrained_model_dir)
    offline_config["export_policy_dir"] = str(export_policy_dir)
    offline_config["num_training_step"] = int(resolved_num_training_steps)
    offline_config["sequence_length"] = int(args.sequence_length)
    offline_config["batch_size"] = int(resolved_batch_size)
    offline_config["policy_lr"] = float(args.policy_lr)
    offline_config["critic_lr"] = float(args.critic_lr)
    offline_config["should_shuffle"] = True
    offline_config["interleave_type"] = str(args.interleave_type)
    offline_config["shuffle_buffer_size"] = int(args.shuffle_buffer_size)
    offline_config["block_length"] = int(args.block_length)
    offline_config["add_uid"] = False
    config["use_learner_ckpt"] = True
    config["use_learner_tb"] = bool(args.tensorboard)
    config["save_every_n_hours"] = float(args.save_every_n_hours)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as file_obj:
        _yaml_dump(config, file_obj)
    return out_path


def _resolved_batch_size(sequence_count: int, requested_batch_size: int) -> int:
    if int(requested_batch_size) <= 0:
        return 1
    if int(sequence_count) <= 0:
        return int(requested_batch_size)
    return max(1, min(int(requested_batch_size), int(sequence_count)))


def _resolved_num_training_steps(
    *,
    requested_num_training_steps: int,
    sequence_count: int,
    batch_size: int,
    target_effective_epochs: float,
    min_training_steps: int,
    max_training_steps: int,
) -> tuple[int, bool]:
    if int(requested_num_training_steps) > 0:
        return int(requested_num_training_steps), False
    if int(sequence_count) <= 0 or int(batch_size) <= 0:
        return max(1, int(min_training_steps)), True
    steps_per_epoch = float(sequence_count) / float(batch_size)
    auto_steps = int(round(steps_per_epoch * float(target_effective_epochs)))
    auto_steps = max(int(min_training_steps), auto_steps)
    if int(max_training_steps) > 0:
        auto_steps = min(int(max_training_steps), auto_steps)
    return max(1, int(auto_steps)), True


def _training_budget_summary(
    *,
    sequence_length: int,
    dataset_transition_rows: int,
    dataset_sequence_count: int,
    requested_batch_size: int,
    resolved_batch_size: int,
    requested_num_training_steps: int,
    resolved_num_training_steps: int,
    auto_training_steps: bool,
    target_effective_epochs: float,
) -> dict[str, float | int | bool]:
    steps_per_epoch = (
        float(dataset_sequence_count) / float(resolved_batch_size)
        if int(resolved_batch_size) > 0 and int(dataset_sequence_count) > 0
        else 0.0
    )
    effective_epochs = (
        float(resolved_num_training_steps) / steps_per_epoch
        if steps_per_epoch > 0.0
        else 0.0
    )
    batch_exceeds_epoch = bool(
        int(requested_batch_size) > 0
        and int(dataset_sequence_count) > 0
        and int(requested_batch_size) >= int(dataset_sequence_count)
    )
    warning = bool(
        batch_exceeds_epoch
        or (steps_per_epoch > 0.0 and effective_epochs >= 1000.0)
    )
    return {
        "sequence_length": int(sequence_length),
        "dataset_transition_rows": int(dataset_transition_rows),
        "dataset_sequence_count": int(dataset_sequence_count),
        "requested_batch_size": int(requested_batch_size),
        "resolved_batch_size": int(resolved_batch_size),
        "requested_num_training_steps": int(requested_num_training_steps),
        "resolved_num_training_steps": int(resolved_num_training_steps),
        "auto_training_steps": bool(auto_training_steps),
        "target_effective_epochs": float(target_effective_epochs),
        "steps_per_epoch": float(steps_per_epoch),
        "effective_epochs": float(effective_epochs),
        "batch_exceeds_epoch": bool(batch_exceeds_epoch),
        "warning": bool(warning),
    }


def _python_env(repo_root: Path, extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath_items = [str(repo_root)]
    if env.get("PYTHONPATH"):
        pythonpath_items.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_items)
    if extra:
        env.update({str(key): str(value) for key, value in extra.items()})
    return env


def _default_out_dir(generated_manifest: Path, repo_root: Path) -> Path:
    trace_dir = generated_manifest.parent
    return repo_root / "attacks" / "output" / "sage-finetune" / trace_dir.name


def _default_sage_rl_python() -> str:
    candidate = Path.home() / "venvpy36" / "bin" / "python"
    if candidate.exists():
        return str(candidate)
    return sys.executable


def parse_args() -> argparse.Namespace:
    repo_root = _repo_root()
    parser = argparse.ArgumentParser(
        description=(
            "Collect baseline-guided Sage-format offline data on adversarial generated traces "
            "or clean trace manifests "
            "and fine-tune the pretrained Sage policy with the existing offline CRR learner."
        )
    )
    parser.add_argument(
        "--generated-manifest",
        required=True,
        help="Path to either an adversarial generated_manifest.json or a clean trace manifest.",
    )
    parser.add_argument(
        "--collection-config-path",
        default=DEFAULT_CLEAN_COLLECTION_CONFIG,
        help=(
            "Eval config used when collecting from a clean trace manifest. "
            "Ignored for adversarial generated manifests."
        ),
    )
    parser.add_argument(
        "--pretrained-model-dir",
        default="sage_rl/rl_module/sage_models/final",
        help="Initial Sage policy checkpoint directory.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Fine-tuning output directory. Defaults under attacks/output/sage-finetune/<trace-set>.",
    )
    parser.add_argument(
        "--offline-config-template",
        default="sage_rl/rl_module/config-rl-eval.yaml",
        help="YAML template for the offline CRR fine-tuning config.",
    )
    parser.add_argument("--dataset-dir", default=None, help="Existing or output offline dataset directory.")
    parser.add_argument(
        "--behavior-policy",
        default="best_baseline",
        choices=["best_baseline", "sage", "reno", "bbr", "cubic"],
        help=(
            "Behavior policy used to populate the offline dataset. "
            "'best_baseline' is the default because it is closest to the original "
            "offline Sage setup of learning from strong controller trajectories."
        ),
    )
    parser.add_argument("--num-traces", type=int, default=0, help="Optional first-N trace limit for collection.")
    parser.add_argument("--skip-collection", action="store_true", help="Reuse an existing offline dataset.")
    parser.add_argument("--skip-training", action="store_true", help="Only collect the offline dataset.")
    parser.add_argument("--append-dataset", action="store_true", help="Allow collection into a non-empty dataset dir.")
    parser.add_argument(
        "--resume-training",
        action="store_true",
        help="Resume from existing offline_crr checkpoints/logs in the output directory instead of starting a fresh fine-tune run.",
    )
    parser.add_argument(
        "--num-training-steps",
        type=int,
        default=0,
        help=(
            "Number of offline CRR learner steps. "
            "Use 0 to derive a dataset-size-aware budget automatically."
        ),
    )
    parser.add_argument(
        "--target-effective-epochs",
        type=float,
        default=64.0,
        help="When --num-training-steps=0, choose a step budget matching about this many effective epochs.",
    )
    parser.add_argument(
        "--min-training-steps",
        type=int,
        default=32,
        help="Lower bound for auto-derived training steps.",
    )
    parser.add_argument(
        "--max-training-steps",
        type=int,
        default=1024,
        help="Upper bound for auto-derived training steps. Use 0 to disable the cap.",
    )
    parser.add_argument("--sequence-length", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--policy-lr", type=float, default=5e-6)
    parser.add_argument("--critic-lr", type=float, default=5e-5)
    parser.add_argument("--shuffle-buffer-size", type=int, default=10000)
    parser.add_argument("--block-length", type=int, default=1)
    parser.add_argument("--interleave-type", default="full", choices=["full", "none"])
    parser.add_argument("--save-every-n-hours", type=float, default=1.0)
    parser.add_argument("--tensorboard", action="store_true", help="Enable learner TensorBoard logs.")
    parser.add_argument(
        "--python",
        default=None,
        help="Deprecated convenience override: use this Python for both collection and offline training.",
    )
    parser.add_argument(
        "--eval-python",
        default=sys.executable,
        help="Python executable for eval_sage_clean_vs_adv.py collection.",
    )
    parser.add_argument(
        "--offline-python",
        default=_default_sage_rl_python(),
        help="Python executable for sage_rl/offline/offline_agent.py training.",
    )
    parser.add_argument(
        "--eval-extra-arg",
        action="append",
        default=[],
        help="Extra argument passed verbatim to eval_sage_clean_vs_adv.py; repeat for each token.",
    )
    parser.add_argument(
        "--progress-interval-seconds",
        type=float,
        default=60.0,
        help="Print a terminal progress heartbeat this often while collection or training is running.",
    )
    parser.set_defaults(quiet_child_logs=True)
    parser.add_argument(
        "--quiet-child-logs",
        dest="quiet_child_logs",
        action="store_true",
        help="Redirect child-process stdout/stderr to files under <out-dir>/child_logs. This is the default.",
    )
    parser.add_argument(
        "--show-child-logs",
        dest="quiet_child_logs",
        action="store_false",
        help="Stream child-process stdout/stderr directly to the terminal.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands and write config/manifest only.")
    args = parser.parse_args()
    if args.python:
        args.eval_python = args.python
        args.offline_python = args.python

    args.generated_manifest = str(_resolve_path(repo_root, args.generated_manifest))
    if args.out_dir is None:
        args.out_dir = str(_default_out_dir(Path(args.generated_manifest), repo_root))
    else:
        args.out_dir = str(_resolve_path(repo_root, args.out_dir))
    return args


def main() -> None:
    args = parse_args()
    repo_root = _repo_root()
    generated_manifest = Path(args.generated_manifest)
    out_dir = Path(args.out_dir)
    pretrained_model_dir = _resolve_path(repo_root, args.pretrained_model_dir)
    config_template = _resolve_path(repo_root, args.offline_config_template)
    collection_config_path = _resolve_path(repo_root, args.collection_config_path)
    dataset_dir = _resolve_path(repo_root, args.dataset_dir) if args.dataset_dir else out_dir / "offline_dataset"
    rejected_dataset_dir = out_dir / "offline_dataset_rejected"
    collection_eval_dir = out_dir / "collection_eval"
    offline_run_dir = out_dir / "offline_crr"
    export_policy_dir = out_dir / "policy_finetuned"
    offline_config_path = out_dir / "config-rl-finetune.yaml"
    child_logs_dir = out_dir / "child_logs"
    collection_log_path = child_logs_dir / "collection.log"
    training_log_path = child_logs_dir / "training.log"
    training_archive_dir = out_dir / "training_history"

    if not generated_manifest.exists():
        raise FileNotFoundError(f"missing generated manifest: {generated_manifest}")
    if not tf_checkpoint_exists(pretrained_model_dir):
        raise FileNotFoundError(f"missing TensorFlow checkpoint in pretrained model dir: {pretrained_model_dir}")
    if not config_template.exists():
        raise FileNotFoundError(f"missing offline config template: {config_template}")
    if not collection_config_path.exists():
        raise FileNotFoundError(f"missing collection config template: {collection_config_path}")

    manifest_kind = _manifest_kind(generated_manifest)

    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    collection_manifest = _collection_manifest(
        manifest_path=generated_manifest,
        out_dir=out_dir,
        num_traces=int(args.num_traces) if int(args.num_traces) > 0 else None,
        manifest_kind=manifest_kind,
    )

    reused_existing_dataset = False
    archived_training_artifacts: list[str] = []
    training_resume_start_step = 0
    existing_files = _dataset_files(dataset_dir)
    if not args.skip_collection:
        if existing_files and not args.append_dataset:
            reused_existing_dataset = True
            print(
                f"Reusing existing offline dataset with {len(existing_files)} .txt files: {dataset_dir}. "
                "Pass --append-dataset if you intentionally want to add more rollouts.",
                flush=True,
            )
        else:
            collect_cmd = [
                args.eval_python,
                str(repo_root / "scripts" / "eval_sage_clean_vs_adv.py"),
            ]
            if str(manifest_kind) == "generated":
                collect_cmd.extend(
                    [
                        "--generated-manifest",
                        str(collection_manifest),
                        "--skip-clean-rollout",
                    ]
                )
            else:
                collect_cmd.extend(
                    [
                        "--test-manifest",
                        str(collection_manifest),
                        "--config-path",
                        str(collection_config_path),
                        "--clean-only-rollout",
                    ]
                )
            collect_cmd.extend(
                [
                    "--out-dir",
                    str(collection_eval_dir),
                    "--offline-dataset-dir",
                    str(dataset_dir),
                    "--offline-behavior-policy",
                    str(args.behavior_policy),
                    *list(args.eval_extra_arg),
                ]
            )
            collect_env = _python_env(
                repo_root,
                {
                    "SAGE_EVAL_CKPT_DIR": str(pretrained_model_dir),
                },
            )
            _run_command(
                collect_cmd,
                cwd=repo_root,
                env=collect_env,
                dry_run=bool(args.dry_run),
                progress_label="collection",
                progress_interval_seconds=float(args.progress_interval_seconds),
                progress_fn=lambda dataset_dir=dataset_dir: _collection_progress_message(dataset_dir),
                log_path=collection_log_path if bool(args.quiet_child_logs) else None,
            )

    dataset_sanitize_summary = _sanitize_dataset_files(
        dataset_dir,
        rejected_dataset_dir,
        dry_run=bool(args.dry_run),
    )
    row_count = _dataset_row_count(dataset_dir)
    if not args.dry_run and row_count <= 0:
        raise RuntimeError(f"no offline transition rows found in {dataset_dir}")
    sequence_count = _dataset_sequence_count(dataset_dir, int(args.sequence_length))
    resolved_batch_size = _resolved_batch_size(int(sequence_count), int(args.batch_size))
    resolved_num_training_steps, auto_training_steps = _resolved_num_training_steps(
        requested_num_training_steps=int(args.num_training_steps),
        sequence_count=int(sequence_count),
        batch_size=int(resolved_batch_size),
        target_effective_epochs=float(args.target_effective_epochs),
        min_training_steps=int(args.min_training_steps),
        max_training_steps=int(args.max_training_steps),
    )
    training_budget = _training_budget_summary(
        sequence_length=int(args.sequence_length),
        dataset_transition_rows=int(row_count),
        dataset_sequence_count=int(sequence_count),
        requested_batch_size=int(args.batch_size),
        resolved_batch_size=int(resolved_batch_size),
        requested_num_training_steps=int(args.num_training_steps),
        resolved_num_training_steps=int(resolved_num_training_steps),
        auto_training_steps=bool(auto_training_steps),
        target_effective_epochs=float(args.target_effective_epochs),
    )
    print("\nOffline fine-tuning budget:")
    print(json.dumps(training_budget, indent=2, sort_keys=True))
    if int(resolved_batch_size) != int(args.batch_size):
        print(
            f"Adjusted batch size from {args.batch_size} to {resolved_batch_size} to match the available sequence count.",
            flush=True,
        )
    if bool(training_budget["warning"]):
        print(
            "Warning: this fine-tuning budget is very large relative to the collected dataset. "
            "With the current sequence_length and batch_size, one learner step may already cover about one epoch "
            "of data, so large num_training_steps can turn into thousands of effective epochs and multi-day CPU jobs.",
            flush=True,
        )

    _prepare_offline_config(
        template_path=config_template,
        out_path=offline_config_path,
        pretrained_model_dir=pretrained_model_dir,
        export_policy_dir=export_policy_dir,
        args=args,
        resolved_batch_size=int(resolved_batch_size),
        resolved_num_training_steps=int(resolved_num_training_steps),
    )

    if not args.skip_training:
        if bool(args.resume_training):
            training_resume_start_step = _latest_logged_training_step(offline_run_dir)
            print(
                f"Resuming offline training from absolute learner step {training_resume_start_step}.",
                flush=True,
            )
        else:
            for stale_path in [offline_run_dir, export_policy_dir, training_log_path]:
                if stale_path.exists():
                    if bool(args.dry_run):
                        print(
                            f"Would archive previous training artifact: {stale_path} -> {training_archive_dir}/...",
                            flush=True,
                        )
                    else:
                        archived_path = _archive_path(stale_path, training_archive_dir)
                        archived_training_artifacts.append(str(archived_path))
                        print(
                            f"Archived previous training artifact: {stale_path} -> {archived_path}",
                            flush=True,
                        )
        offline_run_dir.mkdir(parents=True, exist_ok=True)
        train_cmd = [
            args.offline_python,
            str(repo_root / "sage_rl" / "offline" / "offline_agent.py"),
            "--dataset_name",
            str(dataset_dir),
            "--config",
            str(offline_config_path),
        ]
        _run_command(
            train_cmd,
            cwd=offline_run_dir,
            env=_python_env(repo_root),
            dry_run=bool(args.dry_run),
            progress_label="training",
            progress_interval_seconds=float(args.progress_interval_seconds),
            progress_fn=lambda offline_run_dir=offline_run_dir, total_steps=int(resolved_num_training_steps), training_log_path=training_log_path, start_step=int(training_resume_start_step): _per_run_training_progress_message(
                offline_run_dir,
                total_steps,
                start_step,
                training_log_path if bool(args.quiet_child_logs) else None,
            ),
            log_path=training_log_path if bool(args.quiet_child_logs) else None,
        )

    summary = {
        "generated_manifest": str(generated_manifest),
        "collection_manifest_kind": str(manifest_kind),
        "collection_manifest": str(collection_manifest),
        "collection_config_path": str(collection_config_path),
        "pretrained_model_dir": str(pretrained_model_dir),
        "dataset_dir": str(dataset_dir),
        "dataset_rejected_dir": str(rejected_dataset_dir),
        "behavior_policy": str(args.behavior_policy),
        "dataset_sanitize_summary": dataset_sanitize_summary,
        "dataset_transition_rows": int(row_count),
        "dataset_sequence_count": int(sequence_count),
        "training_budget": training_budget,
        "resume_training": bool(args.resume_training),
        "training_resume_start_step": int(training_resume_start_step),
        "archived_training_artifacts": archived_training_artifacts,
        "reused_existing_dataset": bool(args.skip_collection or reused_existing_dataset),
        "collection_skipped_due_to_existing_dataset": bool(reused_existing_dataset),
        "offline_config_path": str(offline_config_path),
        "offline_run_dir": str(offline_run_dir),
        "export_policy_dir": str(export_policy_dir),
        "child_logs_dir": str(child_logs_dir) if bool(args.quiet_child_logs) else "",
        "eval_python": str(args.eval_python),
        "offline_python": str(args.offline_python),
        "eval_with_finetuned_model": (
            f"SAGE_EVAL_CKPT_DIR={export_policy_dir} "
            "python scripts/eval_sage_clean_vs_adv.py --generated-manifest <manifest> ..."
        ),
    }
    _write_json(out_dir / "finetune_summary.json", summary)
    print("\nFine-tuning workspace:")
    print(json.dumps(summary, indent=2, sort_keys=True))


def tf_checkpoint_exists(path: Path) -> bool:
    checkpoint_file = path / "checkpoint"
    if not checkpoint_file.exists():
        return False
    return any(path.glob("*.index"))


if __name__ == "__main__":
    main()
