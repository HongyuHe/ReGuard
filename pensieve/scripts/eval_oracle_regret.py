#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any


"""
Evaluate a pretrained Pensieve model on trace sets and report:
- per-trace oracle QoE (policy-independent, receding-horizon K-step oracle)
- per-step regret (policy-dependent K-step oracle gap)

Supports:
- `--trace`: a single leaf directory of traces
- `--trace-dir`: a directory of subdirectories of traces (`./<trace-type>/{trace0,trace1,...}`)

Example:
python scripts/eval_oracle_regret.py \
  --model models/genet_20iters_goodinit.pth \
  --trace-dir traces/online \
  --trace-num 142 \
  --oracle-k 5 \
  --suffix genet-trained \
  --wandb --wandb-project pensieve-trace-weighted --wandb-tags genet-trained

python scripts/eval_oracle_regret.py \
  --model models/robustified-0.01entropy/nn_model_ep_426300_robust_finetune-mixed-finetune-mixed.pt \
  --trace-dir traces/online \
  --trace-num 142 \
  --oracle-k 5 \
  --suffix finetune \
  --wandb --wandb-project pensieve-trace-weighted --wandb-tags mixed-finetune

python scripts/eval_oracle_regret.py \
  --finetuned-dir models/robustified-0.01entropy \
  --trace-dir traces/online \
  --trace-num 142 \
  --oracle-k 5 \
  --wandb --wandb-project pensieve-trace-weighted --wandb-tags finetune-0.01entropy

python scripts/eval_oracle_regret.py \
  --model Pensieve-PPO/src/ppo/nn_model_ep_426300.pth \
  --trace-dir traces/online \
  --trace-num 142 \
  --oracle-k 5 \
  --wandb-mode disabled --append
  
  --wandb --wandb-project pensieve-trace-weighted --wandb-tags final-eval


#* Eval shielded:

python scripts/eval_oracle_regret.py \
  --model Pensieve-PPO/src/ppo/nn_model_ep_426300.pth \
  --trace traces/online/clean \
  --trace-num 142 \
  --oracle-k 5 \
  --wandb-mode disabled --append

python scripts/eval_oracle_regret.py \
  --model Pensieve-PPO/src/ppo/nn_model_ep_426300.pth \
  --trace traces/shielded/iter0 \
  --trace-num 142 \
  --oracle-k 5 \
  --suffix shielded \
  --wandb-mode disabled --append

python scripts/eval_oracle_regret.py \
  --model Pensieve-PPO/src/ppo/nn_model_ep_426300.pth \
  --shield-min-rules-file rules/shield/out/maxregret-mixed_bounds_h2o_gap75_delta10_dmin0p3_dmax0p5_min.pl \
  --shield-max-rules-file rules/shield/out/maxregret-mixed_bounds_h2o_gap75_delta10_dmin0p3_dmax0p5_max.pl \
  --shield-only \
  --trace traces/shielded/iter1 \
  --trace-num 142 \
  --oracle-k 5 \
  --suffix shielded \
  --wandb-mode disabled --append

python scripts/eval_oracle_regret.py \
  --model Pensieve-PPO/src/ppo/nn_model_ep_426300.pth \
  --shield-min-rules-file rules/shield/out/shielded-iter3_bounds_h2o_gap75_delta10_dmin0p3_dmax0p5_min.pl \
  --shield-max-rules-file rules/shield/out/shielded-iter3_bounds_h2o_gap75_delta10_dmin0p3_dmax0p5_max.pl \
  --shield-only \
  --trace traces/shielded/iter4 \
  --trace-num 142 \
  --oracle-k 5 \
  --suffix shielded \
  --wandb-mode disabled --append
  
  --wandb-project pensieve-trace-weighted --wandb-tags random-shield-75pct-genet
"""


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _pensieve_src_dir() -> str:
    return os.path.join(_repo_root(), "Pensieve-PPO", "src")


def _add_repo_root_to_path() -> None:
    root = _repo_root()
    if root not in sys.path:
        sys.path.insert(0, root)


def _add_pensieve_src_to_path() -> None:
    src_dir = _pensieve_src_dir()
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


def _resolve_path(path: str) -> str:
    expanded = os.path.expandvars(os.path.expanduser(str(path)))
    if os.path.isabs(expanded):
        return expanded
    if os.path.exists(expanded):
        return os.path.abspath(expanded)

    root = _repo_root()
    under_root = os.path.join(root, expanded)
    if os.path.exists(under_root):
        return os.path.abspath(under_root)

    under_src = os.path.join(root, "Pensieve-PPO", "src", expanded)
    if os.path.exists(under_src):
        return os.path.abspath(under_src)

    return os.path.abspath(under_root)


def _list_trace_files(trace_dir: str) -> list[str]:
    out: list[str] = []
    for name in sorted(os.listdir(str(trace_dir))):
        if str(name).startswith("."):
            continue
        path = os.path.join(str(trace_dir), str(name))
        if not _is_cooked_trace_file(path):
            continue
        out.append(str(name))
    return out


def _is_cooked_trace_file(path: str) -> bool:
    #* Accepts legacy cooked traces with arbitrary suffixes while excluding sidecars like `.jsonl`.
    name = os.path.basename(str(path))
    if str(name).endswith(".json") or str(name).endswith(".jsonl"):
        return False
    if not os.path.isfile(path):
        return False

    with open(str(path), "rb") as f:
        for line in f:
            parse = line.split()
            if not parse:
                continue
            if len(parse) < 2:
                return False
            try:
                float(parse[0])
                float(parse[1])
            except ValueError:
                return False
            return True
    return False


def _list_trace_type_dirs(trace_root: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for name in sorted(os.listdir(str(trace_root))):
        if str(name).startswith("."):
            continue
        path = os.path.join(str(trace_root), str(name))
        if not os.path.isdir(path):
            continue
        if _list_trace_files(path):
            out.append((str(name), str(path)))
    return out


def _normalize_trace_type_suffix(raw_suffix: str | None) -> str:
    if raw_suffix is None:
        return ""
    suffix = str(raw_suffix).strip()
    if not suffix:
        return ""
    return str(suffix) if str(suffix).startswith("_") else f"_{str(suffix)}"


def _format_result_trace_type(*, base_trace_type: str, result_suffix: str, variant_suffix: str = "") -> str:
    return f"{str(base_trace_type)}{str(result_suffix)}{str(variant_suffix)}"


def _shield_variant_trace_type_suffix(*, result_suffix: str, include_unshielded_variant: bool) -> str:
    #* If a caller supplied `--suffix` for a shield-only run, that suffix should be the full reported label.
    if str(result_suffix) and not bool(include_unshielded_variant):
        return ""
    return "_shield"


def _model_robustification_suffix(model_path: str) -> str:
    stem = os.path.splitext(os.path.basename(str(model_path)))[0]
    if "_robust_" in stem:
        return str(stem.split("_robust_", 1)[1])
    return str(stem)


def _list_model_files(model_dir: str) -> list[str]:
    exts = {".pt", ".pth"}
    out: list[str] = []
    for name in sorted(os.listdir(str(model_dir))):
        if str(name).startswith("."):
            continue
        path = os.path.join(str(model_dir), str(name))
        if not os.path.isfile(path):
            continue
        _base, ext = os.path.splitext(str(name))
        if str(ext).lower() not in exts:
            continue
        out.append(str(path))
    return out


def _infer_finetuned_trace_type(*, model_path: str, available_trace_types: list[str]) -> str:
    stem = os.path.splitext(os.path.basename(str(model_path)))[0]
    robust_suffix = _model_robustification_suffix(str(model_path))
    ranked_matches: list[tuple[int, int, str]] = []
    for trace_type in list(available_trace_types):
        trace_type_s = str(trace_type)
        score = -1
        if str(robust_suffix) == str(trace_type_s) or str(stem) == str(trace_type_s):
            score = 2
        elif (
            str(robust_suffix).endswith(f"-{str(trace_type_s)}")
            or str(robust_suffix).endswith(f"_{str(trace_type_s)}")
            or str(stem).endswith(f"-{str(trace_type_s)}")
            or str(stem).endswith(f"_{str(trace_type_s)}")
        ):
            score = 1
        if int(score) >= 0:
            ranked_matches.append((int(score), int(len(str(trace_type_s))), str(trace_type_s)))

    if len(ranked_matches) == 0:
        available_joined = ", ".join(sorted([str(x) for x in list(available_trace_types)]))
        raise RuntimeError(
            f"* Fine-tuned model has no matching trace type under --trace-dir:\n"
            f"  model={model_path}\n"
            f"  robust_suffix={robust_suffix}\n"
            f"  available_trace_types={available_joined}"
        )

    ranked_matches.sort(key=lambda item: (int(item[0]), int(item[1]), str(item[2])), reverse=True)
    best_score = int(ranked_matches[0][0])
    best_len = int(ranked_matches[0][1])
    best_candidates = [str(trace_type) for score, length, trace_type in ranked_matches if int(score) == int(best_score) and int(length) == int(best_len)]
    if len(best_candidates) != 1:
        joined = ", ".join(sorted([str(x) for x in list(best_candidates)]))
        raise RuntimeError(
            f"* Fine-tuned model name is ambiguous; it matches multiple trace types under --trace-dir:\n"
            f"  model={model_path}\n"
            f"  robust_suffix={robust_suffix}\n"
            f"  candidates={joined}"
        )
    return str(best_candidates[0])


def _resolve_finetuned_models_by_trace_type(
    *,
    finetuned_dir: str,
    trace_sets: list[tuple[str, str]],
    ignored_trace_types: set[str] | None = None,
) -> dict[str, str]:
    model_files = _list_model_files(str(finetuned_dir))
    if len(model_files) == 0:
        raise RuntimeError(f"* No model files found under --finetuned-dir: {finetuned_dir}")

    ignored = set() if ignored_trace_types is None else set([str(x) for x in list(ignored_trace_types)])
    available_trace_types = [str(trace_type) for trace_type, _trace_dir in list(trace_sets)]
    required_trace_types = [str(trace_type) for trace_type in list(available_trace_types) if str(trace_type) not in ignored]
    trace_type_to_model: dict[str, str] = {}
    for model_path in list(model_files):
        trace_type = _infer_finetuned_trace_type(
            model_path=str(model_path),
            available_trace_types=list(available_trace_types),
        )
        if str(trace_type) in ignored:
            continue
        if str(trace_type) in set(trace_type_to_model.keys()):
            prev_model = str(trace_type_to_model[str(trace_type)])
            raise RuntimeError(
                f"* Multiple fine-tuned models match the same trace type under --trace-dir:\n"
                f"  trace_type={trace_type}\n"
                f"  model_a={prev_model}\n"
                f"  model_b={model_path}"
            )
        trace_type_to_model[str(trace_type)] = str(model_path)

    missing_trace_types = [str(trace_type) for trace_type in list(required_trace_types) if str(trace_type) not in set(trace_type_to_model.keys())]
    if len(missing_trace_types) > 0:
        joined = ", ".join(sorted([str(x) for x in list(missing_trace_types)]))
        raise RuntimeError(
            f"* --trace-dir contains trace types without a matching fine-tuned model under --finetuned-dir:\n"
            f"  missing_trace_types={joined}\n"
            f"  finetuned_dir={finetuned_dir}"
        )

    return dict(trace_type_to_model)


def _resolve_trace_files(*, trace_dir: str, trace_num: int) -> list[str]:
    files = _list_trace_files(str(trace_dir))
    if not files:
        raise RuntimeError(f"* No trace files found under: {trace_dir}")

    n = int(trace_num)
    if n > 0:
        if len(files) < n:
            raise RuntimeError(f"* Trace dir has only {len(files)} trace(s) (< {n}): {trace_dir}")
        files = list(files[:n])
    return list(files)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    import numpy as np

    return float(np.percentile(np.asarray(values, dtype=np.float64), float(q)))


def _mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    import numpy as np

    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _finite(values: list[float]) -> list[float]:
    return [float(x) for x in values if math.isfinite(float(x))]


def _throughput_mbps(video_chunk_size_bytes: float, delay_ms: float) -> float:
    delay_ms_f = float(delay_ms)
    if delay_ms_f <= 0.0:
        return 0.0
    return float(video_chunk_size_bytes) * 8.0 / (delay_ms_f * 1000.0)


def _avg_bandwidth_mbps(
    video_chunk_size_bytes: float,
    delay_ms: float,
    *,
    link_rtt_ms: float,
    payload_portion: float,
) -> float:
    """
    * Average link bandwidth (Mbps) implied by this chunk transfer, excluding RTT and undoing
      the PACKET_PAYLOAD_PORTION overhead so it's comparable to `cooked_bw` units.
    """
    delay_ms_f = float(delay_ms)
    rtt_free_delay_ms = float(max(delay_ms_f - float(link_rtt_ms), 0.0))
    if rtt_free_delay_ms <= 0.0:
        return 0.0
    payload_portion_f = float(payload_portion)
    if payload_portion_f <= 0.0:
        return 0.0
    return float(video_chunk_size_bytes) * 8.0 / (rtt_free_delay_ms * 1000.0) / payload_portion_f


def _entropy(probs: Any, eps: float = 1e-12) -> float:
    import numpy as np

    p = np.clip(np.asarray(probs, dtype=np.float64), float(eps), 1.0)
    return float(-np.sum(p * np.log(p)))


def _agg_stats(
    values: list[float],
    *,
    include_geo_mean: bool = True,
    geo_mean_eps: float = 1e-12,
) -> dict[str, float]:
    import numpy as np

    if len(values) == 0:
        return {"mean": float("nan"), "geo_mean": float("nan"), "p50": float("nan"), "p95": float("nan")}

    arr = np.asarray(values, dtype=np.float64)
    out = {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "geo_mean": float("nan"),
    }
    if include_geo_mean:
        arr_pos = np.clip(arr, float(geo_mean_eps), None)
        out["geo_mean"] = float(np.exp(np.mean(np.log(arr_pos))))
    return out


def _write_agg_stats(
    summary: dict[str, float],
    metric_name: str,
    values: list[float],
    *,
    include_geo_mean: bool = True,
) -> None:
    #* Helper: write `{metric_name}-{mean,geo_mean,p50,p95}` using `_agg_stats`.
    stats = _agg_stats(values, include_geo_mean=include_geo_mean)
    summary[f"{metric_name}-mean"] = float(stats["mean"])
    summary[f"{metric_name}-p50"] = float(stats["p50"])
    summary[f"{metric_name}-p95"] = float(stats["p95"])
    if include_geo_mean:
        summary[f"{metric_name}-geo_mean"] = float(stats["geo_mean"])


def _write_csv_rows(
    out_csv: str,
    fieldnames: list[str],
    rows: list[dict[str, Any]],
    *,
    append: bool,
) -> None:
    write_header = True
    if bool(append) and os.path.exists(out_csv):
        if os.path.getsize(out_csv) > 0:
            with open(out_csv, newline="") as f:
                existing_header = next(csv.reader(f), [])
            if list(existing_header) != list(fieldnames):
                raise RuntimeError(
                    f"* Existing CSV header does not match requested schema for --append:\n"
                    f"  path={out_csv}\n"
                    f"  existing_header={existing_header}\n"
                    f"  expected_header={fieldnames}"
                )
            write_header = False

    mode = "a" if bool(append) else "w"
    with open(out_csv, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if bool(write_header):
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _update_state(
    state: Any,
    *,
    bit_rate: int,
    buffer_s: float,
    video_chunk_size: int,
    delay_ms: float,
    next_video_chunk_sizes: list[int],
    video_chunk_remain: int,
    util_mod: Any,
) -> Any:
    import numpy as np

    #* Matches `Pensieve-PPO/src/inference.py:_update_state`.
    new_state = np.roll(state, -1, axis=1).astype(np.float64, copy=False)
    new_state[0, -1] = float(util_mod.VIDEO_BIT_RATE[int(bit_rate)]) / float(max(util_mod.VIDEO_BIT_RATE))
    new_state[1, -1] = float(buffer_s) / float(util_mod.BUFFER_NORM_FACTOR)
    new_state[2, -1] = float(video_chunk_size) / float(delay_ms) / float(util_mod.M_IN_K)
    new_state[3, -1] = float(delay_ms) / float(util_mod.M_IN_K) / float(util_mod.BUFFER_NORM_FACTOR)
    new_state[4, : int(util_mod.A_DIM)] = (
        np.asarray(next_video_chunk_sizes, dtype=np.float64) / float(util_mod.M_IN_K) / float(util_mod.M_IN_K)
    )
    new_state[5, -1] = float(min(float(video_chunk_remain), float(util_mod.CHUNK_TIL_VIDEO_END_CAP))) / float(
        util_mod.CHUNK_TIL_VIDEO_END_CAP
    )
    return new_state


def _oracle_k_step_qoe(
    *,
    env_snapshot: Any,
    last_br: int,
    k_steps: int,
    seed: int,
    util_mod: Any,
) -> tuple[float | None, float | None, float | None, float | None, int | None, float | None, float | None]:
    k = int(k_steps)
    if k <= 0:
        return None, None, None, None, None, None, None

    best_r, best_seq, _worst_r, _worst_seq = util_mod.find_optimal_k_step_reward_oracle(env_snapshot, int(last_br), int(k))
    if best_r is None or best_seq is None:
        return None, None, None, None, None, None, None

    #* Derive QoE component totals for the best sequence (raw values, no penalties).
    env_sim = util_mod.copy_env(env_snapshot, RANDOM_SEED=int(seed))
    total_quality = 0.0
    total_rebuf_s = 0.0
    total_smoothness = 0.0
    oracle_action_idx: int | None = None
    oracle_action_kbps: float | None = None
    oracle_buffer_s: float | None = None
    last = int(last_br)
    for idx, action in enumerate(best_seq):
        a = int(action)
        if idx == 0:
            oracle_action_idx = int(a)
            oracle_action_kbps = float(util_mod.VIDEO_BIT_RATE[int(a)])
        try:
            _delay_ms, _sleep_ms, _buffer_s, rebuf_s, _chunk_size, _next_sizes, end, _remain = env_sim.get_video_chunk(a)
        except Exception:
            return None, None, None, None, None, None, None

        if idx == 0:
            oracle_buffer_s = float(_buffer_s)

        #* Match the oracle validity rule: sequences that end early are invalid for K-step comparison.
        if bool(end) and idx < int(k) - 1:
            return None, None, None, None, None, None, None

        total_quality += float(util_mod.VIDEO_BIT_RATE[a]) / float(util_mod.M_IN_K)
        total_rebuf_s += float(rebuf_s)
        total_smoothness += float(abs(util_mod.VIDEO_BIT_RATE[a] - util_mod.VIDEO_BIT_RATE[int(last)])) / float(util_mod.M_IN_K)
        last = int(a)

    denom = float(k)
    return (
        float(best_r) / denom,
        float(total_quality) / denom,
        float(total_rebuf_s) / denom,
        float(total_smoothness) / denom,
        oracle_action_idx,
        oracle_action_kbps,
        oracle_buffer_s,
    )


def _oracle_episode_rollout_receding(
    *,
    env_snapshot: Any,
    last_br: int,
    k_steps: int,
    util_mod: Any,
) -> tuple[float | None, list[int], list[float], list[float]]:
    #* Policy-independent oracle baseline: receding-horizon exact K-step search.
    k = int(k_steps)
    if k <= 0:
        return None, [], [], []

    total_chunks = int(getattr(util_mod, "CHUNK_TIL_VIDEO_END_CAP", 48))
    if total_chunks <= 0:
        total_chunks = 48

    env_sim = env_snapshot
    last = int(last_br)
    total_reward = 0.0
    action_values: list[int] = []
    bitrate_kbps_values: list[float] = []
    buffer_s_values: list[float] = []

    while True:
        remaining = int(total_chunks) - int(getattr(env_sim, "video_chunk_counter", 0))
        k_eff = int(min(int(k), int(max(1, remaining))))
        best_r, best_seq, _worst_r, _worst_seq = util_mod.find_optimal_k_step_reward_oracle(env_sim, int(last), int(k_eff))
        if best_r is None or best_seq is None:
            return None, action_values, bitrate_kbps_values, buffer_s_values

        a0 = int(best_seq[0])
        try:
            _delay_ms, _sleep_ms, buffer_s, rebuf_s, _chunk_size, _next_sizes, end, _remain = env_sim.get_video_chunk(int(a0))
        except Exception:
            return None, action_values, bitrate_kbps_values, buffer_s_values

        total_reward += float(util_mod.calculate_reward(int(a0), int(last), float(rebuf_s)))
        action_values.append(int(a0))
        bitrate_kbps_values.append(float(util_mod.VIDEO_BIT_RATE[int(a0)]))
        buffer_s_values.append(float(buffer_s))
        last = int(a0)
        if bool(end):
            break

    return float(total_reward), action_values, bitrate_kbps_values, buffer_s_values


def _policy_k_step_qoe(
    *,
    actor: Any,
    env_snapshot: Any,
    state_snapshot: Any,
    last_br: int,
    start_action: int,
    k_steps: int,
    seed: int,
    action_selection: str,
    rng_seed: int,
    shield: Any | None,
    util_mod: Any,
) -> tuple[float | None, float | None, float | None, float | None]:
    import numpy as np

    k = int(k_steps)
    if k <= 0:
        return None, None, None, None

    env_sim = util_mod.copy_env(env_snapshot, RANDOM_SEED=int(seed))
    state_sim = np.array(state_snapshot, copy=True)

    total_reward = 0.0
    total_quality = 0.0
    total_rebuf_s = 0.0
    total_smoothness = 0.0

    rng = np.random.RandomState(int(rng_seed))
    shield_snapshot = None if shield is None or not hasattr(shield, "snapshot_runtime_state") else shield.snapshot_runtime_state()

    last = int(last_br)
    a = int(start_action)

    try:
        for i in range(int(k)):
            try:
                (
                    delay_ms,
                    _sleep_ms,
                    buffer_s,
                    rebuf_s,
                    chunk_size,
                    next_sizes,
                    end,
                    remain,
                ) = env_sim.get_video_chunk(int(a))
            except Exception:
                return None, None, None, None

            #* Match the oracle validity rule: sequences that end early are invalid for K-step comparison.
            if bool(end) and i < int(k) - 1:
                return None, None, None, None

            total_reward += float(util_mod.calculate_reward(int(a), int(last), float(rebuf_s)))
            total_quality += float(util_mod.VIDEO_BIT_RATE[int(a)]) / float(util_mod.M_IN_K)
            total_rebuf_s += float(rebuf_s)
            total_smoothness += float(abs(util_mod.VIDEO_BIT_RATE[int(a)] - util_mod.VIDEO_BIT_RATE[int(last)])) / float(util_mod.M_IN_K)

            state_sim = _update_state(
                state_sim,
                bit_rate=int(a),
                buffer_s=float(buffer_s),
                video_chunk_size=int(chunk_size),
                delay_ms=float(delay_ms),
                next_video_chunk_sizes=[int(x) for x in next_sizes],
                video_chunk_remain=int(remain),
                util_mod=util_mod,
            )
            last = int(a)

            if i < int(k) - 1:
                p = actor.predict(np.reshape(state_sim, (1, int(util_mod.S_INFO), int(util_mod.S_LEN)))).astype(np.float64).reshape(-1)
                if shield is None:
                    if str(action_selection) == "argmax":
                        a = int(np.argmax(p))
                    elif str(action_selection) == "gumbel":
                        noise = rng.gumbel(size=p.shape)
                        a = int(np.argmax(np.log(p) + noise))
                    else:
                        raise ValueError(f"* Unknown action_selection: {action_selection}")
                else:
                    a, _stats = shield.select_action(
                        p=p,
                        state=state_sim,
                        last_bitrate_idx=int(last),
                        video_chunk_size_bytes=int(chunk_size),
                        action_selection=str(action_selection),
                        rng=rng,
                    )
    finally:
        if shield is not None and hasattr(shield, "restore_runtime_state"):
            shield.restore_runtime_state(shield_snapshot)

    denom = float(k)
    return (
        float(total_reward) / denom,
        float(total_quality) / denom,
        float(total_rebuf_s) / denom,
        float(total_smoothness) / denom,
    )


def _try_import_wandb() -> Any | None:
    try:
        import wandb  # type: ignore

        if not hasattr(wandb, "init"):
            raise ImportError("* `wandb` import resolved to a non-package directory; install the wandb pip package")
        return wandb
    except Exception:
        return None


#* Worker globals: avoid re-loading the PPO actor for each trace in parallel execution.
_WORKER_ACTOR: Any | None = None
_WORKER_UTIL: Any | None = None
_WORKER_ENV: Any | None = None
_WORKER_MODEL_PATH: str | None = None
_WORKER_SHIELD: Any | None = None


def _worker_init(
    model_path: str,
    shield_min_rules_file: str | None,
    shield_max_rules_file: str | None,
    shield_risk_trigger_steps: int,
    shield_cooldown_steps: int,
) -> None:
    global _WORKER_ACTOR, _WORKER_UTIL, _WORKER_ENV, _WORKER_MODEL_PATH, _WORKER_SHIELD

    _add_repo_root_to_path()
    _add_pensieve_src_to_path()
    import ppo2 as network  # type: ignore
    import util as util_mod  # type: ignore
    import fixed_env as env_mod  # type: ignore

    actor = network.Network(
        state_dim=[int(util_mod.S_INFO), int(util_mod.S_LEN)],
        action_dim=int(util_mod.A_DIM),
        learning_rate=float(util_mod.ACTOR_LR_RATE),
    )
    actor.load_model(str(model_path))

    shield = None
    if shield_min_rules_file is not None and shield_max_rules_file is not None:
        from rules.shield.shield_runtime import ShieldBounds, load_rules

        min_rules = load_rules(str(shield_min_rules_file), output_var="MinBitrateKbps")
        max_rules = load_rules(str(shield_max_rules_file), output_var="MaxBitrateKbps")
        shield = ShieldBounds(
            min_rules=min_rules,
            max_rules=max_rules,
            risk_trigger_steps=int(shield_risk_trigger_steps),
            intervention_cooldown_steps=int(shield_cooldown_steps),
        )

    _WORKER_ACTOR = actor
    _WORKER_UTIL = util_mod
    _WORKER_ENV = env_mod
    _WORKER_MODEL_PATH = str(model_path)
    _WORKER_SHIELD = shield


def _eval_one_trace_file(
    *,
    trace_type: str,
    trace_dir: str,
    trace_name: str,
    trace_idx: int,
    seed: int,
    oracle_k: int,
    action_selection: str,
    use_shield: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    #* Runs in a worker process; relies on `_worker_init` for model loading.
    if _WORKER_ACTOR is None or _WORKER_UTIL is None or _WORKER_ENV is None or _WORKER_MODEL_PATH is None:
        raise RuntimeError("* worker is not initialized (missing actor/util/env)")
    if bool(use_shield) and _WORKER_SHIELD is None:
        raise RuntimeError("* worker shield requested but shield rules were not initialized")

    actor = _WORKER_ACTOR
    util_mod = _WORKER_UTIL
    env_mod = _WORKER_ENV
    model_path = _WORKER_MODEL_PATH
    shield = _WORKER_SHIELD if bool(use_shield) else None
    if shield is not None and hasattr(shield, "reset_runtime_state"):
        shield.reset_runtime_state()

    file_path = os.path.join(str(trace_dir), str(trace_name))
    cooked_time: list[float] = []
    cooked_bw: list[float] = []
    with open(str(file_path), "rb") as f:
        for line in f:
            parse = line.split()
            cooked_time.append(float(parse[0]))
            cooked_bw.append(float(parse[1]))

    env_inst = env_mod.Environment(all_cooked_time=[cooked_time], all_cooked_bw=[cooked_bw], random_seed=int(seed))

    import numpy as np

    state = np.zeros((int(util_mod.S_INFO), int(util_mod.S_LEN)), dtype=np.float64)
    last_bit_rate = int(util_mod.DEFAULT_QUALITY)
    bit_rate = int(util_mod.DEFAULT_QUALITY)

    oracle_env_snapshot = util_mod.copy_env(env_inst, RANDOM_SEED=int(seed))
    oracle_episode_qoe_total, oracle_trace_action_values, oracle_trace_bitrate_kbps_values, oracle_trace_buffer_s_values = (
        _oracle_episode_rollout_receding(
            env_snapshot=oracle_env_snapshot,
            last_br=int(util_mod.DEFAULT_QUALITY),
            k_steps=int(oracle_k),
            util_mod=util_mod,
        )
    )

    per_step_rows: list[dict[str, Any]] = []

    episode_return = 0.0
    episode_regret_values: list[float] = []
    episode_steps = 0
    model_time_ms_values: list[float] = []
    shield_time_ms_values: list[float] = []
    delay_ms_values: list[float] = []
    sleep_time_ms_values: list[float] = []
    decision_time_ms_values: list[float] = []
    decision_time_rel_values: list[float] = []
    sim_time_ms = 0.0

    rng_episode = np.random.RandomState(int(seed) + int(trace_idx))

    while True:
        chunk_idx = int(getattr(env_inst, "video_chunk_counter", -1))

        #* Snapshot trace bandwidth at the current Mahimahi pointer (Mbps).
        trace_bw = (
            float(env_inst.cooked_bw[env_inst.mahimahi_ptr]) if int(env_inst.mahimahi_ptr) < len(env_inst.cooked_bw) else float("nan")
        )

        (
            oracle_reward,
            oracle_quality,
            oracle_rebuf_s,
            oracle_smoothness,
            oracle_action_idx,
            oracle_action_kbps,
            oracle_buffer_s,
        ) = _oracle_k_step_qoe(
            env_snapshot=env_inst,
            last_br=int(last_bit_rate),
            k_steps=int(oracle_k),
            seed=int(seed),
            util_mod=util_mod,
        )
        policy_reward, policy_quality, policy_rebuf_s, policy_smoothness = _policy_k_step_qoe(
            actor=actor,
            env_snapshot=env_inst,
            state_snapshot=state,
            last_br=int(last_bit_rate),
            start_action=int(bit_rate),
            k_steps=int(oracle_k),
            seed=int(seed),
            action_selection=str(action_selection),
            rng_seed=int(seed) + int(trace_idx) * 10_000 + int(chunk_idx),
            shield=shield,
            util_mod=util_mod,
        )

        oracle_gap_reward = None
        oracle_gap_quality = None
        oracle_gap_rebuf_s = None
        oracle_gap_smoothness = None
        if (
            oracle_reward is not None
            and oracle_quality is not None
            and oracle_rebuf_s is not None
            and oracle_smoothness is not None
            and policy_reward is not None
            and policy_quality is not None
            and policy_rebuf_s is not None
            and policy_smoothness is not None
        ):
            oracle_gap_reward = float(oracle_reward) - float(policy_reward)
            oracle_gap_quality = float(oracle_quality) - float(policy_quality)
            oracle_gap_rebuf_s = float(policy_rebuf_s) - float(oracle_rebuf_s)
            oracle_gap_smoothness = float(policy_smoothness) - float(oracle_smoothness)
            episode_regret_values.append(float(oracle_gap_reward))
        else:
            episode_regret_values.append(float("nan"))

        (
            delay_ms,
            sleep_time_ms,
            buffer_s,
            rebuf_s,
            video_chunk_size,
            next_video_chunk_sizes,
            end_of_video,
            video_chunk_remain,
        ) = env_inst.get_video_chunk(int(bit_rate))

        qoe_step = float(util_mod.calculate_reward(int(bit_rate), int(last_bit_rate), float(rebuf_s)))
        episode_return += float(qoe_step)
        episode_steps += 1

        #* QoE breakdown components (raw values, no penalties).
        qoe_quality = float(util_mod.VIDEO_BIT_RATE[int(bit_rate)]) / float(util_mod.M_IN_K)
        qoe_smoothness = float(abs(util_mod.VIDEO_BIT_RATE[int(bit_rate)] - util_mod.VIDEO_BIT_RATE[int(last_bit_rate)])) / float(
            util_mod.M_IN_K
        )
        qoe_rebuf_s = float(rebuf_s)

        state = _update_state(
            state,
            bit_rate=int(bit_rate),
            buffer_s=float(buffer_s),
            video_chunk_size=int(video_chunk_size),
            delay_ms=float(delay_ms),
            next_video_chunk_sizes=[int(x) for x in next_video_chunk_sizes],
            video_chunk_remain=int(video_chunk_remain),
            util_mod=util_mod,
        )

        model_start_ns = time.perf_counter_ns()
        p = actor.predict(np.reshape(state, (1, int(util_mod.S_INFO), int(util_mod.S_LEN)))).astype(np.float64).reshape(-1)
        model_time_ms = float(time.perf_counter_ns() - int(model_start_ns)) / 1_000_000.0
        entropy = float(_entropy(p))
        max_prob = float(np.max(p))
        shield_time_ms = 0.0
        if shield is None:
            if str(action_selection) == "argmax":
                next_bit_rate = int(np.argmax(p))
            elif str(action_selection) == "gumbel":
                noise = rng_episode.gumbel(size=p.shape)
                next_bit_rate = int(np.argmax(np.log(p) + noise))
            else:
                raise ValueError(f"* Unknown action_selection: {action_selection}")
        else:
            shield_start_ns = time.perf_counter_ns()
            next_bit_rate, _shield_stats = shield.select_action(
                p=p,
                state=state,
                last_bitrate_idx=int(last_bit_rate),
                video_chunk_size_bytes=int(video_chunk_size),
                action_selection=str(action_selection),
                rng=rng_episode,
            )
            shield_time_ms = float(time.perf_counter_ns() - int(shield_start_ns)) / 1_000_000.0
        decision_time_ms = float(model_time_ms + shield_time_ms)
        sim_time_start_ms = float(sim_time_ms)
        env_interval_ms = float(delay_ms + sleep_time_ms)
        sim_time_ms = float(sim_time_ms + env_interval_ms)
        decision_time_rel = float("nan") if env_interval_ms <= 0.0 else float(decision_time_ms / env_interval_ms)
        model_time_ms_values.append(float(model_time_ms))
        shield_time_ms_values.append(float(shield_time_ms))
        delay_ms_values.append(float(delay_ms))
        sleep_time_ms_values.append(float(sleep_time_ms))
        decision_time_ms_values.append(float(decision_time_ms))
        if math.isfinite(float(decision_time_rel)):
            decision_time_rel_values.append(float(decision_time_rel))

        oracle_trace_action: int | None = None
        oracle_trace_bitrate_kbps: float | None = None
        oracle_trace_buffer_s: float | None = None
        if 0 <= int(chunk_idx) < len(oracle_trace_action_values):
            oracle_trace_action = int(oracle_trace_action_values[int(chunk_idx)])
            oracle_trace_bitrate_kbps = float(oracle_trace_bitrate_kbps_values[int(chunk_idx)])
            oracle_trace_buffer_s = float(oracle_trace_buffer_s_values[int(chunk_idx)])

        per_step_row: dict[str, Any] = {
            "trace_type": str(trace_type),
            "trace_dir": str(trace_dir),
            "model_path": str(model_path),
            "seed": int(seed),
            "oracle_k": int(oracle_k),
            "trace_name": str(trace_name),
            "trace_idx": int(trace_idx),
            "chunk_idx": int(chunk_idx),
            "sim_time_start_ms": float(sim_time_start_ms),
            "sim_time_start_s": float(sim_time_start_ms / 1000.0),
            "sim_interval_ms": float(env_interval_ms),
            "sim_interval_s": float(env_interval_ms / 1000.0),
            "sim_time_ms": float(sim_time_ms),
            "sim_time_s": float(sim_time_ms / 1000.0),
            "video_chunk_remain": int(video_chunk_remain),
            #* Two-pass Phase-2 parity: use `adv/*` metric names.
            "adv/trace": str(trace_name),
            "adv/trace_bw_mbps": float(trace_bw),
            "adv/action": int(bit_rate),
            "adv/action_next": int(next_bit_rate),
            "adv/bitrate_kbps": float(util_mod.VIDEO_BIT_RATE[int(bit_rate)]),
            "adv/video_chunk_size_bytes": float(video_chunk_size),
            "adv/delay_ms": float(delay_ms),
            "adv/sleep_time_ms": float(sleep_time_ms),
            "adv/buffer_s": float(buffer_s),
            "adv/rebuf_s": float(rebuf_s),
            "adv/realized_bw_mbps": float(
                _avg_bandwidth_mbps(
                    float(video_chunk_size),
                    float(delay_ms),
                    link_rtt_ms=float(getattr(env_mod, "LINK_RTT", 0.0)),
                    payload_portion=float(getattr(env_mod, "PACKET_PAYLOAD_PORTION", 1.0)),
                )
            ),
            "adv/throughput_mbps": float(_throughput_mbps(float(video_chunk_size), float(delay_ms))),
            "adv/qoe_reward": float(qoe_step),
            "adv/qoe/quality": float(qoe_quality),
            "adv/qoe/rebuf_s": float(qoe_rebuf_s),
            "adv/qoe/smoothness": float(qoe_smoothness),
            "adv/entropy": float(entropy),
            "adv/max_prob": float(max_prob),
            #* Keep legacy per-step CSV columns for downstream analysis scripts.
            "policy_action_idx": int(bit_rate),
            "policy_action_kbps": int(util_mod.VIDEO_BIT_RATE[int(bit_rate)]),
            "last_action_idx": int(last_bit_rate),
            "last_action_kbps": int(util_mod.VIDEO_BIT_RATE[int(last_bit_rate)]),
            "delay_ms": float(delay_ms),
            "throughput_mbps": float(_throughput_mbps(float(video_chunk_size), float(delay_ms))),
            "buffer_s": float(buffer_s),
            "rebuf_s": float(rebuf_s),
            "video_chunk_size_bytes": int(video_chunk_size),
            "sleep_time_ms": float(sleep_time_ms),
            "model_time_ms": float(model_time_ms),
            "shield_time_ms": float(shield_time_ms),
            "decision_time_ms": float(decision_time_ms),
            "decision_time_rel": float(decision_time_rel),
            "qoe_step": float(qoe_step),
            "oracle_reward_mean": float("nan") if oracle_reward is None else float(oracle_reward),
            "policy_reward_mean": float("nan") if policy_reward is None else float(policy_reward),
            "regret": float("nan") if oracle_gap_reward is None else float(oracle_gap_reward),
        }

        if oracle_action_idx is not None:
            per_step_row["adv/oracle/action"] = int(oracle_action_idx)
            if oracle_action_kbps is not None:
                per_step_row["adv/oracle/bitrate_kbps"] = float(oracle_action_kbps)
            if oracle_buffer_s is not None:
                per_step_row["adv/oracle/buffer_s"] = float(oracle_buffer_s)

        if oracle_trace_action is not None:
            per_step_row["adv/oracle_trace/action"] = int(oracle_trace_action)
            if oracle_trace_bitrate_kbps is not None:
                per_step_row["adv/oracle_trace/bitrate_kbps"] = float(oracle_trace_bitrate_kbps)
            if oracle_trace_buffer_s is not None:
                per_step_row["adv/oracle_trace/buffer_s"] = float(oracle_trace_buffer_s)

        if oracle_gap_reward is not None:
            per_step_row["adv/oracle/qoe_reward"] = float(oracle_reward)
            per_step_row["adv/oracle/qoe/quality"] = float(oracle_quality)
            per_step_row["adv/oracle/qoe/rebuf_s"] = float(oracle_rebuf_s)
            per_step_row["adv/oracle/qoe/smoothness"] = float(oracle_smoothness)
            per_step_row["adv/oracle_gap/qoe_reward"] = float(oracle_gap_reward)
            per_step_row["adv/oracle_gap/qoe/quality"] = float(oracle_gap_quality)
            per_step_row["adv/oracle_gap/qoe/rebuf_s"] = float(oracle_gap_rebuf_s)
            per_step_row["adv/oracle_gap/qoe/smoothness"] = float(oracle_gap_smoothness)

        per_step_rows.append(per_step_row)

        last_bit_rate = int(bit_rate)
        bit_rate = int(next_bit_rate)

        if bool(end_of_video):
            break

    regret_finite = _finite(episode_regret_values)
    regret_mean = float("nan") if not regret_finite else float(_mean(regret_finite))
    per_trace_row: dict[str, Any] = {
        "trace_type": str(trace_type),
        "trace_dir": str(trace_dir),
        "model_path": str(model_path),
        "seed": int(seed),
        "oracle_k": int(oracle_k),
        "trace_name": str(trace_name),
        "trace_idx": int(trace_idx),
        "num_steps": int(episode_steps),
        "sim_time_ms_total": float(sim_time_ms),
        "sim_time_s_total": float(sim_time_ms / 1000.0),
        "policy_qoe_total": float(episode_return),
        "oracle_qoe_total": float("nan") if oracle_episode_qoe_total is None else float(oracle_episode_qoe_total),
        "qoe_gap_total": float("nan") if oracle_episode_qoe_total is None else float(oracle_episode_qoe_total) - float(episode_return),
        "regret_count_valid": int(len(regret_finite)),
        "regret_mean": float(regret_mean),
        "regret_p50": float("nan") if not regret_finite else float(_percentile(regret_finite, 50)),
        "regret_p95": float("nan") if not regret_finite else float(_percentile(regret_finite, 95)),
        "regret_sum": float("nan") if not regret_finite else float(sum(regret_finite)),
        "model_time_ms_mean": float("nan") if not model_time_ms_values else float(_mean(model_time_ms_values)),
        "model_time_ms_p50": float("nan") if not model_time_ms_values else float(_percentile(model_time_ms_values, 50)),
        "model_time_ms_p95": float("nan") if not model_time_ms_values else float(_percentile(model_time_ms_values, 95)),
        "shield_time_ms_mean": float("nan") if not shield_time_ms_values else float(_mean(shield_time_ms_values)),
        "shield_time_ms_p50": float("nan") if not shield_time_ms_values else float(_percentile(shield_time_ms_values, 50)),
        "shield_time_ms_p95": float("nan") if not shield_time_ms_values else float(_percentile(shield_time_ms_values, 95)),
        "delay_ms_mean": float("nan") if not delay_ms_values else float(_mean(delay_ms_values)),
        "delay_ms_p50": float("nan") if not delay_ms_values else float(_percentile(delay_ms_values, 50)),
        "delay_ms_p95": float("nan") if not delay_ms_values else float(_percentile(delay_ms_values, 95)),
        "sleep_time_ms_mean": float("nan") if not sleep_time_ms_values else float(_mean(sleep_time_ms_values)),
        "sleep_time_ms_p50": float("nan") if not sleep_time_ms_values else float(_percentile(sleep_time_ms_values, 50)),
        "sleep_time_ms_p95": float("nan") if not sleep_time_ms_values else float(_percentile(sleep_time_ms_values, 95)),
        "decision_time_ms_mean": float("nan") if not decision_time_ms_values else float(_mean(decision_time_ms_values)),
        "decision_time_ms_p50": float("nan") if not decision_time_ms_values else float(_percentile(decision_time_ms_values, 50)),
        "decision_time_ms_p95": float("nan") if not decision_time_ms_values else float(_percentile(decision_time_ms_values, 95)),
        "decision_time_rel_mean": float("nan") if not decision_time_rel_values else float(_mean(decision_time_rel_values)),
        "decision_time_rel_p50": float("nan") if not decision_time_rel_values else float(_percentile(decision_time_rel_values, 50)),
        "decision_time_rel_p95": float("nan") if not decision_time_rel_values else float(_percentile(decision_time_rel_values, 95)),
    }
    return per_trace_row, per_step_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate one model, or a matched set of fine-tuned models, on traces; write per-trace oracle QoE and per-step regret to CSV and (optionally) wandb."
    )
    parser.add_argument("--model", type=str, default=None, help="* PPO model path (repo-relative or absolute)")
    parser.add_argument("--result-label", default=None, help="Explicit label for a single trace/model evaluation.")
    parser.add_argument(
        "--finetuned-dir",
        type=str,
        default=None,
        help="* Directory of fine-tuned models; adversarial trace types under --trace-dir are matched strictly, and every matched model is also evaluated on the shared clean trace.",
    )
    trace_grp = parser.add_mutually_exclusive_group(required=True)
    trace_grp.add_argument("--trace", type=str, default=None, help="* Directory containing ONE type of traces (leaf dir)")
    trace_grp.add_argument(
        "--trace-dir",
        type=str,
        default=None,
        help="* Directory containing subdirectories of traces (`./<trace-type>/{trace0,trace1,...}`)",
    )

    parser.add_argument(
        "--trace-num",
        type=int,
        default=-1,
        help="* Number of traces to evaluate per trace type (-1 = all; if >0, uses first N and errors if fewer exist).",
    )
    parser.add_argument(
        "--oracle-k",
        type=int,
        default=5,
        help="* K for the K-step oracle/regret. Per-trace oracle QoE uses receding horizon with effective K <= oracle-k.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--action-selection",
        type=str,
        default="argmax",
        choices=["argmax", "gumbel"],
        help="* Policy action selection rule (matches inference.py behavior).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="* Max concurrent trace rollouts (0 = all CPU cores).",
    )
    parser.add_argument(
        "--out-trace-csv",
        type=str,
        default="traces/eval_oracle_regret_traces.csv",
        help="* Output CSV path for per-trace metrics.",
    )
    parser.add_argument(
        "--out-step-csv",
        type=str,
        default="traces/eval_oracle_regret_steps.csv",
        help="* Output CSV path for per-step metrics.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="* Append rows to existing CSV outputs instead of overwriting them.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="* Optional suffix appended to every reported trace type, e.g. --suffix genet -> clean_genet, hotnets19_genet. For shield-only runs, a non-empty suffix replaces the default `_shield` label.",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="* Print progress info to stderr",
    )
    parser.add_argument("--shield-min-rules-file", type=str, default=None, help="* Optional shield min-rules file.")
    parser.add_argument("--shield-max-rules-file", type=str, default=None, help="* Optional shield max-rules file.")
    parser.add_argument(
        "--shield-risk-trigger-steps",
        type=int,
        default=1,
        help="* Shield hysteresis: require this many consecutive risky states before intervening.",
    )
    parser.add_argument(
        "--shield-cooldown-steps",
        type=int,
        default=0,
        help="* Shield hysteresis: suppress further interventions for this many steps after an intervention.",
    )
    parser.add_argument(
        "--shield-only",
        action="store_true",
        help="* If shield rules are provided, evaluate only the shielded policy variants.",
    )

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="pensieve-eval")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, default="")
    parser.add_argument("--wandb-mode", type=str, default="online")

    args = parser.parse_args()
    if args.result_label and (not args.trace or args.finetuned_dir):
        parser.error("--result-label requires --trace and --model")
    if args.result_label and args.shield_min_rules_file and not args.shield_only:
        parser.error("--result-label with shield rules requires --shield-only")

    #* Force CPU for reproducibility and to avoid multi-process GPU contention.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

    model_path = None if args.model is None else _resolve_path(str(args.model))
    finetuned_dir = None if args.finetuned_dir is None else _resolve_path(str(args.finetuned_dir))
    out_trace_csv = _resolve_path(str(args.out_trace_csv))
    out_step_csv = _resolve_path(str(args.out_step_csv))
    shield_min_rules_file = None if args.shield_min_rules_file is None else _resolve_path(str(args.shield_min_rules_file))
    shield_max_rules_file = None if args.shield_max_rules_file is None else _resolve_path(str(args.shield_max_rules_file))
    result_trace_type_suffix = _normalize_trace_type_suffix(str(args.suffix))

    if args.model is None and args.finetuned_dir is None:
        raise RuntimeError("* Provide either --model or --finetuned-dir")
    if args.model is not None and args.finetuned_dir is not None:
        raise RuntimeError("* Provide either --model or --finetuned-dir, not both")
    if model_path is not None and not os.path.isfile(str(model_path)):
        raise FileNotFoundError(f"* --model not found: {model_path}")
    if finetuned_dir is not None and not os.path.isdir(str(finetuned_dir)):
        raise FileNotFoundError(f"* --finetuned-dir not found: {finetuned_dir}")
    if finetuned_dir is not None and args.trace is not None:
        raise RuntimeError("* --finetuned-dir requires --trace-dir because models are matched to trace-type subdirectories")
    if (shield_min_rules_file is None) != (shield_max_rules_file is None):
        raise RuntimeError("* Provide both --shield-min-rules-file and --shield-max-rules-file, or neither")
    if shield_min_rules_file is not None and not os.path.isfile(str(shield_min_rules_file)):
        raise FileNotFoundError(f"* --shield-min-rules-file not found: {shield_min_rules_file}")
    if shield_max_rules_file is not None and not os.path.isfile(str(shield_max_rules_file)):
        raise FileNotFoundError(f"* --shield-max-rules-file not found: {shield_max_rules_file}")
    if int(args.shield_risk_trigger_steps) <= 0:
        raise RuntimeError("* --shield-risk-trigger-steps must be >= 1")
    if int(args.shield_cooldown_steps) < 0:
        raise RuntimeError("* --shield-cooldown-steps must be >= 0")
    if bool(args.shield_only) and (shield_min_rules_file is None or shield_max_rules_file is None):
        raise RuntimeError("* --shield-only requires both --shield-min-rules-file and --shield-max-rules-file")

    trace_sets: list[tuple[str, str]] = []
    if args.trace is not None:
        trace_dir = _resolve_path(str(args.trace))
        if not os.path.isdir(trace_dir):
            raise FileNotFoundError(f"* --trace not found: {trace_dir}")
        trace_type = os.path.basename(os.path.abspath(trace_dir)) or "trace"
        trace_sets = [(str(trace_type), str(trace_dir))]
    else:
        trace_root = _resolve_path(str(args.trace_dir))
        if not os.path.isdir(trace_root):
            raise FileNotFoundError(f"* --trace-dir not found: {trace_root}")
        trace_sets = _list_trace_type_dirs(str(trace_root))
        if not trace_sets:
            hint = ""
            root_files = _list_trace_files(str(trace_root))
            if root_files:
                hint = "\n* Note: `--trace-dir` expects subdirectories; you probably want `--trace` for a leaf dir."
            raise RuntimeError(f"* No trace subdirectories found under: {trace_root}{hint}")

    finetuned_models_by_trace_type: dict[str, str] | None = None
    clean_trace_set: tuple[str, str] | None = None
    adv_trace_sets: list[tuple[str, str]] = list(trace_sets)
    if finetuned_dir is not None:
        for trace_type, trace_dir in list(trace_sets):
            if str(trace_type) == "clean":
                clean_trace_set = (str(trace_type), str(trace_dir))
                break
        adv_trace_sets = [(str(trace_type), str(trace_dir)) for trace_type, trace_dir in list(trace_sets) if str(trace_type) != "clean"]
        if clean_trace_set is None:
            raise RuntimeError("* --finetuned-dir requires a shared clean trace subdirectory named `clean` under --trace-dir")
        if len(adv_trace_sets) == 0:
            raise RuntimeError("* --finetuned-dir requires at least one adversarial trace type under --trace-dir in addition to `clean`")
        finetuned_models_by_trace_type = _resolve_finetuned_models_by_trace_type(
            finetuned_dir=str(finetuned_dir),
            trace_sets=list(trace_sets),
            ignored_trace_types={"clean"},
        )

    include_unshielded_variant = not bool(args.shield_only)
    eval_variants: list[tuple[str, bool]] = []
    if not bool(args.shield_only):
        eval_variants.append(("", False))
    if shield_min_rules_file is not None and shield_max_rules_file is not None:
        eval_variants.append(
            (
                _shield_variant_trace_type_suffix(
                    result_suffix=str(result_trace_type_suffix),
                    include_unshielded_variant=bool(include_unshielded_variant),
                ),
                True,
            )
        )

    tasks: list[tuple[str, str, str, int, bool, str]] = []
    expected_by_type: dict[str, int] = {}
    use_shield_by_trace_type: dict[str, bool] = {}
    if finetuned_models_by_trace_type is None:
        for trace_type, trace_dir in trace_sets:
            files = _resolve_trace_files(trace_dir=str(trace_dir), trace_num=int(args.trace_num))
            task_model_path = str(model_path)
            trace_type_label = str(trace_type)
            for suffix, use_shield in eval_variants:
                trace_type_variant = _format_result_trace_type(
                    base_trace_type=str(trace_type_label),
                    result_suffix=str(result_trace_type_suffix),
                    variant_suffix=str(suffix),
                )
                if args.result_label:
                    trace_type_variant = args.result_label
                expected_by_type[str(trace_type_variant)] = int(len(files))
                prev_use_shield = use_shield_by_trace_type.get(str(trace_type_variant))
                if prev_use_shield is not None and bool(prev_use_shield) != bool(use_shield):
                    raise RuntimeError(
                        f"* Trace type label is ambiguous across shield variants:\n"
                        f"  trace_type={trace_type_variant}\n"
                        f"  suffix={args.suffix!r}\n"
                        f"  hint=use --shield-only or choose a distinct --suffix"
                    )
                use_shield_by_trace_type[str(trace_type_variant)] = bool(use_shield)
                for i, name in enumerate(files):
                    tasks.append((str(trace_type_variant), str(trace_dir), str(name), int(i), bool(use_shield), str(task_model_path)))
    else:
        assert clean_trace_set is not None
        clean_trace_dir = str(clean_trace_set[1])
        clean_files = _resolve_trace_files(trace_dir=str(clean_trace_dir), trace_num=int(args.trace_num))
        for trace_type, trace_dir in adv_trace_sets:
            adv_files = _resolve_trace_files(trace_dir=str(trace_dir), trace_num=int(args.trace_num))
            task_model_path = str(finetuned_models_by_trace_type[str(trace_type)])
            adv_trace_type_label = f"{str(trace_type)}_finetune"
            clean_trace_type_label = f"{str(trace_type)}_finetune-clean"
            for suffix, use_shield in eval_variants:
                adv_trace_type_variant = _format_result_trace_type(
                    base_trace_type=str(adv_trace_type_label),
                    result_suffix=str(result_trace_type_suffix),
                    variant_suffix=str(suffix),
                )
                expected_by_type[str(adv_trace_type_variant)] = int(len(adv_files))
                prev_adv_use_shield = use_shield_by_trace_type.get(str(adv_trace_type_variant))
                if prev_adv_use_shield is not None and bool(prev_adv_use_shield) != bool(use_shield):
                    raise RuntimeError(
                        f"* Trace type label is ambiguous across shield variants:\n"
                        f"  trace_type={adv_trace_type_variant}\n"
                        f"  suffix={args.suffix!r}\n"
                        f"  hint=use --shield-only or choose a distinct --suffix"
                    )
                use_shield_by_trace_type[str(adv_trace_type_variant)] = bool(use_shield)
                for i, name in enumerate(adv_files):
                    tasks.append(
                        (
                            str(adv_trace_type_variant),
                            str(trace_dir),
                            str(name),
                            int(i),
                            bool(use_shield),
                            str(task_model_path),
                        )
                    )

                clean_trace_type_variant = _format_result_trace_type(
                    base_trace_type=str(clean_trace_type_label),
                    result_suffix=str(result_trace_type_suffix),
                    variant_suffix=str(suffix),
                )
                expected_by_type[str(clean_trace_type_variant)] = int(len(clean_files))
                prev_clean_use_shield = use_shield_by_trace_type.get(str(clean_trace_type_variant))
                if prev_clean_use_shield is not None and bool(prev_clean_use_shield) != bool(use_shield):
                    raise RuntimeError(
                        f"* Trace type label is ambiguous across shield variants:\n"
                        f"  trace_type={clean_trace_type_variant}\n"
                        f"  suffix={args.suffix!r}\n"
                        f"  hint=use --shield-only or choose a distinct --suffix"
                    )
                use_shield_by_trace_type[str(clean_trace_type_variant)] = bool(use_shield)
                for i, name in enumerate(clean_files):
                    tasks.append(
                        (
                            str(clean_trace_type_variant),
                            str(clean_trace_dir),
                            str(name),
                            int(i),
                            bool(use_shield),
                            str(task_model_path),
                        )
                    )

    if not tasks:
        raise RuntimeError("* No traces to evaluate")

    jobs = int(args.jobs)
    if jobs <= 0:
        jobs = max(int(os.cpu_count() or 1), 1)

    if bool(args.progress):
        total_traces = int(len(tasks))
        shown = ", ".join([f"{k}={v}" for k, v in list(expected_by_type.items())[:8]])
        suffix = "..." if len(expected_by_type) > 8 else ""
        if finetuned_dir is None:
            print(f"* Model: {model_path}", file=sys.stderr, flush=True)
        else:
            print(f"* Finetuned dir: {finetuned_dir}", file=sys.stderr, flush=True)
            mapping_parts = [f"{trace_type}->{os.path.basename(path)}" for trace_type, path in sorted(finetuned_models_by_trace_type.items())]
            mapping_suffix = " ..." if len(mapping_parts) > 6 else ""
            print(
                f"* Finetuned matches: {'; '.join(mapping_parts[:6])}{mapping_suffix}",
                file=sys.stderr,
                flush=True,
            )
        print(f"* Trace types: {shown}{suffix} (total_traces={total_traces})", file=sys.stderr, flush=True)
        if shield_min_rules_file is not None and shield_max_rules_file is not None:
            print(
                f"* Shield rules: min={shield_min_rules_file} max={shield_max_rules_file}",
                file=sys.stderr,
                flush=True,
            )
        print(
            f"* oracle_k={int(args.oracle_k)} seed={int(args.seed)} action_selection={str(args.action_selection)} jobs={int(jobs)}",
            file=sys.stderr,
            flush=True,
        )

    done_by_type: dict[str, int] = {k: 0 for k in expected_by_type}
    start_t = time.perf_counter()
    last_report_t = start_t

    per_trace_rows: list[dict[str, Any]] = []
    per_step_rows: list[dict[str, Any]] = []
    tasks_by_model: dict[str, list[tuple[str, str, str, int, bool]]] = {}
    for trace_type, trace_dir, trace_name, trace_idx, use_shield, task_model_path in tasks:
        tasks_by_model.setdefault(str(task_model_path), []).append(
            (str(trace_type), str(trace_dir), str(trace_name), int(trace_idx), bool(use_shield))
        )

    done_total = 0
    for task_model_path in sorted(tasks_by_model.keys()):
        model_tasks = list(tasks_by_model[str(task_model_path)])
        with ProcessPoolExecutor(
            max_workers=int(jobs),
            initializer=_worker_init,
            initargs=(
                str(task_model_path),
                shield_min_rules_file,
                shield_max_rules_file,
                int(args.shield_risk_trigger_steps),
                int(args.shield_cooldown_steps),
            ),
        ) as pool:
            futures = [
                pool.submit(
                    _eval_one_trace_file,
                    trace_type=trace_type,
                    trace_dir=trace_dir,
                    trace_name=trace_name,
                    trace_idx=int(trace_idx),
                    seed=int(args.seed),
                    oracle_k=int(args.oracle_k),
                    action_selection=str(args.action_selection),
                    use_shield=bool(use_shield),
                )
                for trace_type, trace_dir, trace_name, trace_idx, use_shield in model_tasks
            ]

            for fut in as_completed(futures):
                trace_row, step_rows = fut.result()
                per_trace_rows.append(trace_row)
                per_step_rows.extend(step_rows)

                ttype = str(trace_row.get("trace_type", "unknown"))
                done_by_type[ttype] = int(done_by_type.get(ttype, 0)) + 1
                done_total += 1

                if bool(args.progress):
                    now = time.perf_counter()
                    if done_total == len(tasks) or (now - last_report_t) >= 1.0:
                        elapsed = float(now - start_t)
                        rate = float(done_total) / max(1e-9, elapsed)
                        parts: list[str] = []
                        for k in sorted(expected_by_type.keys()):
                            if int(done_by_type.get(k, 0)) > 0 and len(parts) < 6:
                                parts.append(f"{k}={done_by_type.get(k, 0)}/{expected_by_type.get(k, 0)}")
                        suffix = " ..." if len(expected_by_type) > len(parts) else ""
                        detail = (" ".join(parts) + suffix) if parts else ""
                        print(
                            f"* progress traces={done_total}/{len(tasks)} rate={rate:.2f} traces/s {detail}",
                            file=sys.stderr,
                            flush=True,
                        )
                        last_report_t = now

    #* Deterministic ordering for CSV + wandb logging.
    per_trace_rows.sort(key=lambda r: (str(r.get("trace_type", "")), int(r.get("trace_idx", 0))))
    per_step_rows.sort(key=lambda r: (str(r.get("trace_type", "")), int(r.get("trace_idx", 0)), int(r.get("chunk_idx", 0))))

    os.makedirs(os.path.dirname(out_trace_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_step_csv) or ".", exist_ok=True)

    trace_fieldnames = [
        "trace_type",
        "trace_dir",
        "model_path",
        "seed",
        "oracle_k",
        "trace_name",
        "trace_idx",
        "num_steps",
        "sim_time_ms_total",
        "sim_time_s_total",
        "policy_qoe_total",
        "oracle_qoe_total",
        "qoe_gap_total",
        "regret_count_valid",
        "regret_mean",
        "regret_p50",
        "regret_p95",
        "regret_sum",
        "model_time_ms_mean",
        "model_time_ms_p50",
        "model_time_ms_p95",
        "shield_time_ms_mean",
        "shield_time_ms_p50",
        "shield_time_ms_p95",
        "delay_ms_mean",
        "delay_ms_p50",
        "delay_ms_p95",
        "sleep_time_ms_mean",
        "sleep_time_ms_p50",
        "sleep_time_ms_p95",
        "decision_time_ms_mean",
        "decision_time_ms_p50",
        "decision_time_ms_p95",
        "decision_time_rel_mean",
        "decision_time_rel_p50",
        "decision_time_rel_p95",
    ]
    _write_csv_rows(
        out_trace_csv,
        trace_fieldnames,
        per_trace_rows,
        append=bool(args.append),
    )

    step_fieldnames = [
        "trace_type",
        "trace_dir",
        "model_path",
        "seed",
        "oracle_k",
        "trace_name",
        "trace_idx",
        "chunk_idx",
        "sim_time_start_ms",
        "sim_time_start_s",
        "sim_interval_ms",
        "sim_interval_s",
        "sim_time_ms",
        "sim_time_s",
        "policy_action_idx",
        "policy_action_kbps",
        "last_action_idx",
        "last_action_kbps",
        "delay_ms",
        "sleep_time_ms",
        "model_time_ms",
        "shield_time_ms",
        "decision_time_ms",
        "decision_time_rel",
        "throughput_mbps",
        "buffer_s",
        "rebuf_s",
        "video_chunk_size_bytes",
        "video_chunk_remain",
        "qoe_step",
        "oracle_reward_mean",
        "policy_reward_mean",
        "regret",
    ]
    _write_csv_rows(
        out_step_csv,
        step_fieldnames,
        per_step_rows,
        append=bool(args.append),
    )

    if bool(args.wandb):
        wandb = _try_import_wandb()
        if wandb is None:
            raise RuntimeError("* --wandb set but `wandb` is not importable; install `wandb` and retry")
        tags = [t.strip() for t in str(args.wandb_tags).split(",") if t.strip()]

        #* Group rows by trace type (subdirectory name).
        steps_by_type: dict[str, list[dict[str, Any]]] = {}
        traces_by_type: dict[str, list[dict[str, Any]]] = {}
        for row in per_step_rows:
            ttype = str(row.get("trace_type", "unknown"))
            steps_by_type.setdefault(ttype, []).append(row)
        for row in per_trace_rows:
            ttype = str(row.get("trace_type", "unknown"))
            traces_by_type.setdefault(ttype, []).append(row)

        for trace_type in sorted(steps_by_type.keys()):
            trace_rows = list(traces_by_type.get(str(trace_type), []))
            trace_rows.sort(key=lambda r: int(r.get("trace_idx", 0)))
            run_model_path = "" if len(trace_rows) == 0 else str(trace_rows[0].get("model_path", ""))
            run = wandb.init(
                project=str(args.wandb_project),
                entity=args.wandb_entity,
                group=args.wandb_group,
                name=str(trace_type),
                mode=str(args.wandb_mode),
                tags=tags,
                config={
                    "model_path": str(run_model_path),
                    "trace_type": str(trace_type),
                    "trace_type_suffix": str(result_trace_type_suffix),
                    "use_shield": bool(use_shield_by_trace_type.get(str(trace_type), False)),
                    "shield_only": bool(args.shield_only),
                    "shield_min_rules_file": shield_min_rules_file,
                    "shield_max_rules_file": shield_max_rules_file,
                    "shield_risk_trigger_steps": int(args.shield_risk_trigger_steps),
                    "shield_cooldown_steps": int(args.shield_cooldown_steps),
                    "seed": int(args.seed),
                    "oracle_k": int(args.oracle_k),
                    "trace_num": int(expected_by_type.get(str(trace_type), 0)),
                    "action_selection": str(args.action_selection),
                },
            )

            step_rows = list(steps_by_type.get(str(trace_type), []))
            step_rows.sort(key=lambda r: (int(r.get("trace_idx", 0)), int(r.get("chunk_idx", 0))))

            #* Summary time series buffers (mirror inference end2end-two-pass Phase 2 summaries).
            adv_qoe_reward_values: list[float] = []
            adv_qoe_quality_values: list[float] = []
            adv_qoe_rebuf_s_values: list[float] = []
            adv_qoe_smoothness_values: list[float] = []
            adv_delay_ms_values: list[float] = []
            adv_throughput_mbps_values: list[float] = []
            adv_realized_bw_mbps_values: list[float] = []
            adv_buffer_s_values: list[float] = []
            adv_rebuf_s_values: list[float] = []
            adv_entropy_values: list[float] = []
            adv_max_prob_values: list[float] = []
            oracle_bitrate_kbps_values: list[float] = []
            oracle_buffer_s_values: list[float] = []
            oracle_reward_values: list[float] = []
            oracle_quality_values: list[float] = []
            oracle_rebuf_s_values: list[float] = []
            oracle_smoothness_values: list[float] = []
            oracle_gap_reward_values: list[float] = []
            oracle_gap_quality_values: list[float] = []
            oracle_gap_rebuf_s_values: list[float] = []
            oracle_gap_smoothness_values: list[float] = []
            oracle_trace_bitrate_kbps_values: list[float] = []
            oracle_trace_buffer_s_values: list[float] = []

            per_trace_by_idx: dict[int, dict[str, Any]] = {int(r.get("trace_idx", -1)): r for r in trace_rows}

            #* Per-step time series: mimic end2end two-pass Phase-2 metrics (adv/* only).
            global_step = 0
            global_sim_time_ms = 0.0
            cur_trace_idx: int | None = None
            for row in step_rows:
                trace_idx = int(row.get("trace_idx", -1))
                if cur_trace_idx is None:
                    cur_trace_idx = trace_idx
                if trace_idx != cur_trace_idx:
                    prev = per_trace_by_idx.get(int(cur_trace_idx), {})
                    wandb.log(
                        {
                            "adv/episode/cumulative_reward": float(prev.get("policy_qoe_total", float("nan"))),
                            "env/episode": int(cur_trace_idx),
                            "eval/oracle_trace_qoe_total": float(prev.get("oracle_qoe_total", float("nan"))),
                            "eval/qoe_gap_total": float(prev.get("qoe_gap_total", float("nan"))),
                            "eval/regret_mean": float(prev.get("regret_mean", float("nan"))),
                            "env/sim_time_ms": float(global_sim_time_ms),
                            "env/sim_time_s": float(global_sim_time_ms / 1000.0),
                            "env/episode_time_ms": float(prev.get("sim_time_ms_total", float("nan"))),
                            "env/episode_time_s": float(prev.get("sim_time_s_total", float("nan"))),
                        },
                        step=int(global_step),
                    )
                    cur_trace_idx = trace_idx

                sim_interval_ms = float(row.get("sim_interval_ms", float("nan")))
                if not math.isfinite(float(sim_interval_ms)):
                    sim_interval_ms = float(row.get("delay_ms", 0.0)) + float(row.get("sleep_time_ms", 0.0))
                if math.isfinite(float(sim_interval_ms)):
                    global_sim_time_ms = float(global_sim_time_ms + sim_interval_ms)

                metrics: dict[str, Any] = {
                    "env/step": int(global_step),
                    "env/episode": int(trace_idx),
                    "env/sim_time_ms": float(global_sim_time_ms),
                    "env/sim_time_s": float(global_sim_time_ms / 1000.0),
                    "env/episode_time_ms": float(row.get("sim_time_ms", float("nan"))),
                    "env/episode_time_s": float(row.get("sim_time_s", float("nan"))),
                    "env/chunk_start_time_ms": float(row.get("sim_time_start_ms", float("nan"))),
                    "env/chunk_start_time_s": float(row.get("sim_time_start_s", float("nan"))),
                    "env/chunk_end_time_ms": float(row.get("sim_time_ms", float("nan"))),
                    "env/chunk_end_time_s": float(row.get("sim_time_s", float("nan"))),
                    "env/sim_interval_ms": float(sim_interval_ms),
                    "env/sim_interval_s": float(sim_interval_ms / 1000.0) if math.isfinite(float(sim_interval_ms)) else float("nan"),
                }
                for k in [
                    "adv/trace",
                    "adv/trace_bw_mbps",
                    "adv/action",
                    "adv/action_next",
                    "adv/bitrate_kbps",
                    "adv/video_chunk_size_bytes",
                    "adv/delay_ms",
                    "adv/sleep_time_ms",
                    "adv/buffer_s",
                    "adv/rebuf_s",
                    "adv/realized_bw_mbps",
                    "adv/throughput_mbps",
                    "adv/qoe_reward",
                    "adv/qoe/quality",
                    "adv/qoe/rebuf_s",
                    "adv/qoe/smoothness",
                    "adv/entropy",
                    "adv/max_prob",
                    "adv/oracle/action",
                    "adv/oracle/bitrate_kbps",
                    "adv/oracle/buffer_s",
                    "adv/oracle/qoe_reward",
                    "adv/oracle/qoe/quality",
                    "adv/oracle/qoe/rebuf_s",
                    "adv/oracle/qoe/smoothness",
                    "adv/oracle_gap/qoe_reward",
                    "adv/oracle_gap/qoe/quality",
                    "adv/oracle_gap/qoe/rebuf_s",
                    "adv/oracle_gap/qoe/smoothness",
                    "adv/oracle_trace/action",
                    "adv/oracle_trace/bitrate_kbps",
                    "adv/oracle_trace/buffer_s",
                ]:
                    if k in row:
                        metrics[str(k)] = row[str(k)]

                wandb.log(metrics, step=int(global_step))

                adv_qoe_reward_values.append(float(row.get("adv/qoe_reward", float("nan"))))
                adv_qoe_quality_values.append(float(row.get("adv/qoe/quality", float("nan"))))
                adv_qoe_rebuf_s_values.append(float(row.get("adv/qoe/rebuf_s", float("nan"))))
                adv_qoe_smoothness_values.append(float(row.get("adv/qoe/smoothness", float("nan"))))
                adv_delay_ms_values.append(float(row.get("adv/delay_ms", float("nan"))))
                adv_throughput_mbps_values.append(float(row.get("adv/throughput_mbps", float("nan"))))
                adv_realized_bw_mbps_values.append(float(row.get("adv/realized_bw_mbps", float("nan"))))
                adv_buffer_s_values.append(float(row.get("adv/buffer_s", float("nan"))))
                adv_rebuf_s_values.append(float(row.get("adv/rebuf_s", float("nan"))))
                adv_entropy_values.append(float(row.get("adv/entropy", float("nan"))))
                adv_max_prob_values.append(float(row.get("adv/max_prob", float("nan"))))

                if "adv/oracle/bitrate_kbps" in row:
                    oracle_bitrate_kbps_values.append(float(row.get("adv/oracle/bitrate_kbps", float("nan"))))
                if "adv/oracle/buffer_s" in row:
                    oracle_buffer_s_values.append(float(row.get("adv/oracle/buffer_s", float("nan"))))
                if "adv/oracle_trace/bitrate_kbps" in row:
                    oracle_trace_bitrate_kbps_values.append(float(row.get("adv/oracle_trace/bitrate_kbps", float("nan"))))
                if "adv/oracle_trace/buffer_s" in row:
                    oracle_trace_buffer_s_values.append(float(row.get("adv/oracle_trace/buffer_s", float("nan"))))

                if "adv/oracle/qoe_reward" in row:
                    oracle_reward_values.append(float(row.get("adv/oracle/qoe_reward", float("nan"))))
                    oracle_quality_values.append(float(row.get("adv/oracle/qoe/quality", float("nan"))))
                    oracle_rebuf_s_values.append(float(row.get("adv/oracle/qoe/rebuf_s", float("nan"))))
                    oracle_smoothness_values.append(float(row.get("adv/oracle/qoe/smoothness", float("nan"))))
                    oracle_gap_reward_values.append(float(row.get("adv/oracle_gap/qoe_reward", float("nan"))))
                    oracle_gap_quality_values.append(float(row.get("adv/oracle_gap/qoe/quality", float("nan"))))
                    oracle_gap_rebuf_s_values.append(float(row.get("adv/oracle_gap/qoe/rebuf_s", float("nan"))))
                    oracle_gap_smoothness_values.append(float(row.get("adv/oracle_gap/qoe/smoothness", float("nan"))))

                global_step += 1

            if cur_trace_idx is not None:
                prev = per_trace_by_idx.get(int(cur_trace_idx), {})
                wandb.log(
                    {
                        "adv/episode/cumulative_reward": float(prev.get("policy_qoe_total", float("nan"))),
                        "env/episode": int(cur_trace_idx),
                        "eval/oracle_trace_qoe_total": float(prev.get("oracle_qoe_total", float("nan"))),
                        "eval/qoe_gap_total": float(prev.get("qoe_gap_total", float("nan"))),
                        "eval/regret_mean": float(prev.get("regret_mean", float("nan"))),
                        "env/sim_time_ms": float(global_sim_time_ms),
                        "env/sim_time_s": float(global_sim_time_ms / 1000.0),
                        "env/episode_time_ms": float(prev.get("sim_time_ms_total", float("nan"))),
                        "env/episode_time_s": float(prev.get("sim_time_s_total", float("nan"))),
                    },
                    step=int(global_step),
                )

            run.summary["env/steps-sum"] = int(max(int(global_step), 1))
            run.summary["env/sim_time_ms-sum"] = float(global_sim_time_ms)
            run.summary["env/sim_time_s-sum"] = float(global_sim_time_ms / 1000.0)
            _write_agg_stats(run.summary, "adv/qoe_reward", _finite(adv_qoe_reward_values), include_geo_mean=False)
            _write_agg_stats(run.summary, "adv/qoe/quality", _finite(adv_qoe_quality_values), include_geo_mean=True)
            _write_agg_stats(run.summary, "adv/qoe/rebuf_s", _finite(adv_qoe_rebuf_s_values), include_geo_mean=True)
            _write_agg_stats(run.summary, "adv/qoe/smoothness", _finite(adv_qoe_smoothness_values), include_geo_mean=True)
            _write_agg_stats(run.summary, "adv/delay_ms", _finite(adv_delay_ms_values), include_geo_mean=True)
            _write_agg_stats(run.summary, "adv/throughput_mbps", _finite(adv_throughput_mbps_values), include_geo_mean=True)
            _write_agg_stats(run.summary, "adv/realized_bw_mbps", _finite(adv_realized_bw_mbps_values), include_geo_mean=True)
            _write_agg_stats(run.summary, "adv/buffer_s", _finite(adv_buffer_s_values), include_geo_mean=True)
            _write_agg_stats(run.summary, "adv/rebuf_s", _finite(adv_rebuf_s_values), include_geo_mean=True)
            _write_agg_stats(run.summary, "adv/entropy", _finite(adv_entropy_values), include_geo_mean=True)
            _write_agg_stats(run.summary, "adv/max_prob", _finite(adv_max_prob_values), include_geo_mean=True)
            _write_agg_stats(run.summary, "adv/oracle/bitrate_kbps", _finite(oracle_bitrate_kbps_values), include_geo_mean=True)
            _write_agg_stats(run.summary, "adv/oracle/buffer_s", _finite(oracle_buffer_s_values), include_geo_mean=True)
            _write_agg_stats(
                run.summary, "adv/oracle_trace/bitrate_kbps", _finite(oracle_trace_bitrate_kbps_values), include_geo_mean=True
            )
            _write_agg_stats(run.summary, "adv/oracle_trace/buffer_s", _finite(oracle_trace_buffer_s_values), include_geo_mean=True)
            _write_agg_stats(run.summary, "adv/oracle/qoe_reward", _finite(oracle_reward_values), include_geo_mean=False)
            _write_agg_stats(run.summary, "adv/oracle/qoe/quality", _finite(oracle_quality_values), include_geo_mean=True)
            _write_agg_stats(run.summary, "adv/oracle/qoe/rebuf_s", _finite(oracle_rebuf_s_values), include_geo_mean=True)
            _write_agg_stats(run.summary, "adv/oracle/qoe/smoothness", _finite(oracle_smoothness_values), include_geo_mean=True)
            _write_agg_stats(run.summary, "adv/oracle_gap/qoe_reward", _finite(oracle_gap_reward_values), include_geo_mean=False)
            _write_agg_stats(run.summary, "adv/oracle_gap/qoe/quality", _finite(oracle_gap_quality_values), include_geo_mean=False)
            _write_agg_stats(run.summary, "adv/oracle_gap/qoe/rebuf_s", _finite(oracle_gap_rebuf_s_values), include_geo_mean=False)
            _write_agg_stats(run.summary, "adv/oracle_gap/qoe/smoothness", _finite(oracle_gap_smoothness_values), include_geo_mean=False)

            run.finish()

    print(f"* Wrote per-trace CSV: {out_trace_csv}")
    print(f"* Wrote per-step CSV: {out_step_csv}")


if __name__ == "__main__":
    main()
