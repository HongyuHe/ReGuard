import os
import re
import json
import argparse
import time
import random
import hashlib
import math
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np
import load_trace
#import a2c as network
import ppo2 as network
import fixed_env as env
from util import *
from pgd import PGDAttack
from symbolic_policy_attack import enumerate_delay_perturbation_candidates, load_policy_rules
import wandb

# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size delay reward
# NN_MODEL = sys.argv[1]
# NN_MODEL = "pretrain/nn_model_ep_155400.pth"
NN_MODEL = "ppo/nn_model_ep_426300.pth"

#* Path helpers: make scripts runnable from repo root or from `src/`.
def _resolve_under_src(path: str) -> str:
    #* Accept user-provided relative paths from either the repo root or from `src/`.
    expanded = os.path.expandvars(os.path.expanduser(str(path)))
    if os.path.isabs(expanded):
        return expanded

    #* Prefer paths that already exist relative to the current working directory.
    cwd_path = os.path.abspath(expanded)
    if os.path.exists(cwd_path):
        return cwd_path

    #* Fall back to resolving under `src/`.
    under_src = os.path.abspath(os.path.join(os.path.dirname(__file__), expanded))
    if os.path.exists(under_src):
        return under_src

    #* Finally, try resolving under the repo root (two levels above `src/`).
    under_root = os.path.abspath(_resolve_repo_root(expanded))
    if os.path.exists(under_root):
        return under_root

    return under_src


def _resolve_repo_root(path: str) -> str:
    #* Resolve paths relative to the repo root (two levels up from `src/`).
    if os.path.isabs(path):
        return path
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(repo_root, path)


def _safe_path_component(value: str) -> str:
    s = str(value).strip()
    if not s:
        return "unknown"
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def _wandb_subrun_name(*, base_name: str | None, subrun: str, num_subruns: int) -> str:
    #* Robustify/shield create multiple W&B runs (one per trace subdirectory); allow overriding the run name.
    if base_name is None:
        return str(subrun)
    base = str(base_name).strip()
    if not base:
        return str(subrun)
    if int(num_subruns) <= 1:
        return str(base)
    return f"{base}-{subrun}"


def _ensure_trailing_sep(path: str) -> str:
    return path if path.endswith(os.sep) else path + os.sep


def _parse_tags(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    tags = [t.strip() for t in str(raw).split(",") if t.strip()]
    return tags if tags else None


def _sample_traces(
    all_cooked_time: list,
    all_cooked_bw: list,
    all_file_names: list,
    trace_num: int,
    *,
    seed: int = 42,
) -> tuple[list, list, list]:
    #* When `trace_num > 0`, sample a deterministic subset of traces for fair comparisons.
    if trace_num <= 0:
        return all_cooked_time, all_cooked_bw, all_file_names

    total = len(all_file_names)
    if trace_num >= total:
        return all_cooked_time, all_cooked_bw, all_file_names

    rng = np.random.RandomState(int(seed))
    idx = rng.choice(total, size=int(trace_num), replace=False).tolist()
    return (
        [all_cooked_time[i] for i in idx],
        [all_cooked_bw[i] for i in idx],
        [all_file_names[i] for i in idx],
    )
    # #* Return the first `trace_num` traces for simplicity.
    # return (
    #     all_cooked_time[:trace_num],
    #     all_cooked_bw[:trace_num],
    #     all_file_names[:trace_num],
    # )


def _segments_to_cooked_trace(
    bw_segments_mbps: list[float], duration_segments_s: list[float]
) -> tuple[list[float], list[float]]:
    if len(bw_segments_mbps) != len(duration_segments_s):
        raise ValueError("* bw_segments_mbps and duration_segments_s must have the same length")
    cooked_time: list[float] = [0.0]
    cooked_bw: list[float] = [0.0]
    for bw_mbps, dur_s in zip(bw_segments_mbps, duration_segments_s):
        dur = float(dur_s)
        if not np.isfinite(dur) or dur <= 0.0:
            continue
        cooked_time.append(float(cooked_time[-1]) + float(dur))
        cooked_bw.append(float(bw_mbps))
    return cooked_time, cooked_bw


def _write_cooked_trace_file(
    *,
    out_dir: str,
    file_name: str,
    cooked_time: list[float],
    cooked_bw: list[float],
) -> str:
    if len(cooked_time) != len(cooked_bw):
        raise ValueError("* cooked_time and cooked_bw must have the same length")
    os.makedirs(str(out_dir), exist_ok=True)
    out_path = os.path.join(str(out_dir), str(file_name))
    with open(out_path, "w") as f:
        for t, bw in zip(cooked_time, cooked_bw):
            f.write(f"{float(t):.9f}\t{float(bw):.9f}\n")
    return out_path


def _extract_download_prefix_segments(
    *,
    bandwidth_trace_mbps: list[float],
    duration_trace_s: list[float],
    video_chunk_size_bytes: int,
) -> tuple[list[float], list[float], float, bool]:
    #* Match `fixed_env.Environment.get_video_chunk` semantics: treat exact-equality as completion.
    sent_bytes = 0.0
    used_bw: list[float] = []
    used_t: list[float] = []
    delay_s = 0.0

    for bw_mbps, dur_s in zip(bandwidth_trace_mbps, duration_trace_s):
        bw = float(bw_mbps)
        dur = float(dur_s)
        if not np.isfinite(bw) or not np.isfinite(dur) or dur <= 0.0:
            continue

        throughput_Bps = bw * env.B_IN_MB / env.BITS_IN_BYTE
        payload_bytes = throughput_Bps * dur * env.PACKET_PAYLOAD_PORTION

        #* Match `fixed_env.Environment.get_video_chunk` semantics (fractional last segment).
        if throughput_Bps > 0.0 and float(sent_bytes) + float(payload_bytes) >= float(video_chunk_size_bytes):
            remaining_bytes = float(video_chunk_size_bytes) - float(sent_bytes)
            frac_s = remaining_bytes / (throughput_Bps * env.PACKET_PAYLOAD_PORTION)
            frac_s = float(max(min(frac_s, dur), 0.0))
            used_bw.append(float(bw))
            used_t.append(float(frac_s))
            delay_s += float(frac_s)
            sent_bytes = float(video_chunk_size_bytes)
            break

        used_bw.append(float(bw))
        used_t.append(float(dur))
        delay_s += float(dur)
        sent_bytes += float(payload_bytes)

    complete = float(sent_bytes) > float(video_chunk_size_bytes) - 1e-9
    return used_bw, used_t, float(delay_s), bool(complete)


#* Metric helpers for feature-space attack strength evaluation.
def _entropy(probs: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(probs, eps, 1.0)
    return float(-np.sum(p * np.log(p)))


def _kl_div(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p_ = np.clip(p, eps, 1.0)
    q_ = np.clip(q, eps, 1.0)
    return float(np.sum(p_ * np.log(p_ / q_)))


def _agg_stats(
    values: list[float],
    *,
    include_geo_mean: bool = True,
    geo_mean_eps: float = 1e-12,
) -> dict[str, float]:
    """
    Returns: {"mean": ..., "geo_mean": ..., "p50": ..., "p95": ...}
    """
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


def _write_agg_stats_or_zero(
    summary: dict[str, float],
    metric_name: str,
    values: list[float],
    *,
    include_geo_mean: bool = True,
) -> None:
    if len(values) == 0:
        summary[f"{metric_name}-mean"] = 0.0
        summary[f"{metric_name}-p50"] = 0.0
        summary[f"{metric_name}-p95"] = 0.0
        if include_geo_mean:
            summary[f"{metric_name}-geo_mean"] = 0.0
        return
    _write_agg_stats(summary, metric_name, values, include_geo_mean=include_geo_mean)


def _rules_fully_relaxed_metrics(
    pgd_attacker: PGDAttack,
    *,
    totals_before: tuple[int, int] | None,
) -> dict[str, float | int]:
    total_calls, total_fully_relaxed = pgd_attacker.get_rules_projection_totals()
    if totals_before is None:
        step_calls = 0
        step_fully_relaxed = 0
    else:
        before_calls, before_fully_relaxed = totals_before
        step_calls = max(int(total_calls) - int(before_calls), 0)
        step_fully_relaxed = max(int(total_fully_relaxed) - int(before_fully_relaxed), 0)

    step_rate = float(step_fully_relaxed) / float(step_calls) if step_calls > 0 else 0.0
    total_rate = float(total_fully_relaxed) / float(total_calls) if total_calls > 0 else 0.0
    return {
        "pgd/rules_projection_calls": int(step_calls),
        "pgd/rules_fully_relaxed_calls": int(step_fully_relaxed),
        "pgd/rules_fully_relaxed_rate": float(step_rate),
        "pgd/rules_projection_calls_total": int(total_calls),
        "pgd/rules_fully_relaxed_calls_total": int(total_fully_relaxed),
        "pgd/rules_fully_relaxed_rate_total": float(total_rate),
    }


def _rules_projection_metrics(pgd_attacker: PGDAttack) -> dict[str, float | int]:
    return pgd_attacker.get_rules_projection_last_attack_stats()


def _rules_projection_total_metrics(pgd_attacker: PGDAttack) -> dict[str, float | int]:
    success_total, fail_total, fallbacks_total = pgd_attacker.get_rules_projection_success_totals()
    attempts_total = int(success_total) + int(fail_total)
    return {
        "pgd/rules_projection_attempts_total": int(attempts_total),
        "pgd/rules_projection_success_total": int(success_total),
        "pgd/rules_projection_fail_total": int(fail_total),
        "pgd/rules_projection_fallbacks_total": int(fallbacks_total),
        "pgd/rules_projection_success_rate_total": float(success_total) / float(max(attempts_total, 1)),
    }


def _implied_chunk_size_bytes(
    throughput_norm: float,
    delay_norm: float,
    *,
    m_in_k: float,
    delay_norm_factor: float,
) -> float:
    """
    * `throughput_norm` is the state feature (MB/s), i.e. throughput_mbps / 8.
    """
    delay_ms = delay_norm * m_in_k * delay_norm_factor
    return float(throughput_norm) * float(delay_ms) * float(m_in_k)


def _throughput_mbps(video_chunk_size_bytes: float, delay_ms: float) -> float:
    """
    * Average payload throughput over the whole chunk download window (including RTT in `delay_ms`).
    """
    delay_ms_f = float(delay_ms)
    if delay_ms_f <= 0.0:
        return 0.0
    return float(video_chunk_size_bytes) * 8.0 / (delay_ms_f * 1000.0)


def _avg_bandwidth_mbps(video_chunk_size_bytes: float, delay_ms: float) -> float:
    """
    * Average link bandwidth (Mbps) implied by this chunk transfer, excluding RTT and undoing
      the PACKET_PAYLOAD_PORTION overhead so it's comparable to `cooked_bw` units.
    """
    delay_ms_f = float(delay_ms)
    rtt_free_delay_ms = float(max(delay_ms_f - float(env.LINK_RTT), 0.0))
    if rtt_free_delay_ms <= 0.0:
        return 0.0
    payload_portion = float(getattr(env, "PACKET_PAYLOAD_PORTION", 1.0))
    if payload_portion <= 0.0:
        return 0.0
    return float(video_chunk_size_bytes) * 8.0 / (rtt_free_delay_ms * 1000.0) / payload_portion


def _state_bitrate_kbps_hist(state: np.ndarray) -> list[int]:
    max_br = float(np.max(VIDEO_BIT_RATE))
    bitrate_values = np.array(VIDEO_BIT_RATE, dtype=np.float64)
    out: list[int] = []
    for val in np.array(state[0, :], dtype=np.float64, copy=False).tolist():
        if not np.isfinite(val):
            out.append(int(VIDEO_BIT_RATE[int(DEFAULT_QUALITY)]))
            continue
        br_kbps = float(val) * float(max_br)
        idx = int(np.argmin(np.abs(bitrate_values - br_kbps)))
        out.append(int(VIDEO_BIT_RATE[idx]))
    return out


def _policy_rule_inputs_from_state(
    state: np.ndarray,
    *,
    last_bitrate_idx: int,
    video_chunk_size_bytes: int,
) -> tuple[dict[str, float], list[float], list[float], list[int]]:
    history_len = int(state.shape[1])

    delay_ms_hist = [float(state[3, i]) * float(M_IN_K) * float(BUFFER_NORM_FACTOR) for i in range(history_len)]
    throughput_mbps_hist = [float(state[2, i]) * 8.0 for i in range(history_len)]

    chosen_chunk_bytes_hist: list[int] = []
    for i in range(history_len):
        chunk_bytes = float(state[2, i]) * float(delay_ms_hist[i]) * float(M_IN_K)
        if not np.isfinite(chunk_bytes):
            chunk_bytes = 0.0
        chosen_chunk_bytes_hist.append(int(max(1.0, round(chunk_bytes))))
    if history_len > 0:
        chosen_chunk_bytes_hist[-1] = int(video_chunk_size_bytes)

    values: dict[str, float] = {
        "BufferSeconds": float(state[1, -1]) * float(BUFFER_NORM_FACTOR),
        "LastBitrateKbps": float(VIDEO_BIT_RATE[int(last_bitrate_idx)]),
        "ChunksRemaining": float(state[5, -1]) * float(CHUNK_TIL_VIDEO_END_CAP),
    }
    for j in range(int(A_DIM)):
        values[f"NextChunkBytes{j}"] = float(state[4, j]) * float(M_IN_K) * float(M_IN_K)

    return values, delay_ms_hist, throughput_mbps_hist, chosen_chunk_bytes_hist


def _policy_k_step_reward_mean(
    *,
    actor: network.Network,
    env_snapshot: env.Environment,
    state_snapshot: np.ndarray,
    last_br: int,
    start_action: int,
    k_steps: int,
    action_selection: str,
    rng: np.random.RandomState,
) -> float | None:
    k = int(k_steps)
    if k <= 0:
        return None

    env_sim = copy_env(env_snapshot, RANDOM_SEED=int(rng.randint(0, 2**31 - 1)))
    state_sim = np.array(state_snapshot, copy=True)

    total_reward = 0.0
    last = int(last_br)
    a = int(start_action)

    for i in range(int(k)):
        try:
            delay_ms, _sleep_ms, buffer_s, rebuf_s, chunk_size, next_sizes, end, remain = env_sim.get_video_chunk(int(a))
        except Exception:
            return None

        #* Match the oracle validity rule: sequences that end early are invalid for K-step comparison.
        if bool(end) and i < int(k) - 1:
            return None

        total_reward += float(calculate_reward(int(a), int(last), float(rebuf_s)))
        state_sim = _update_state(
            state_sim,
            bit_rate=int(a),
            buffer_s=float(buffer_s),
            video_chunk_size=int(chunk_size),
            delay_ms=float(delay_ms),
            next_video_chunk_sizes=[int(x) for x in next_sizes],
            video_chunk_remain=int(remain),
        )
        last = int(a)

        if i < int(k) - 1:
            p = actor.predict(np.reshape(state_sim, (1, S_INFO, S_LEN))).astype(np.float64).reshape(-1)
            if str(action_selection) == "argmax":
                a = int(np.argmax(p))
            elif str(action_selection) == "gumbel":
                noise = rng.gumbel(size=p.shape)
                a = int(np.argmax(np.log(p + 1e-12) + noise))
            else:
                raise ValueError(f"Unknown action_selection: {action_selection}")

    return float(total_reward) / float(k)


def _select_target_action_by_oracle_gap(
    *,
    actor: network.Network,
    env_snapshot: env.Environment,
    state_snapshot: np.ndarray,
    last_br: int,
    k_steps: int,
    action_selection: str,
    rng: np.random.RandomState,
) -> tuple[int | None, dict[str, float]]:
    k = int(k_steps)
    if k <= 0:
        return None, {"sym/oracle_best_reward": float("nan")}

    #* `oracle_horizon` is kept for CLI compatibility; DT uses the exact K-step oracle.
    best_r, _best_seq, _worst_r, _worst_seq = find_optimal_k_step_reward_oracle(env_snapshot, int(last_br), int(k))
    if best_r is None:
        return None, {"sym/oracle_best_reward": float("nan")}

    oracle_best_mean = float(best_r) / float(k)
    gaps: list[tuple[float, int]] = []
    for a in range(int(A_DIM)):
        pr = _policy_k_step_reward_mean(
            actor=actor,
            env_snapshot=env_snapshot,
            state_snapshot=state_snapshot,
            last_br=int(last_br),
            start_action=int(a),
            k_steps=int(k),
            action_selection=str(action_selection),
            rng=rng,
        )
        if pr is None:
            continue
        gap = float(oracle_best_mean) - float(pr)
        if not np.isfinite(gap):
            continue
        gaps.append((float(gap), int(a)))

    if len(gaps) == 0:
        return None, {"sym/oracle_best_reward": float(oracle_best_mean)}

    max_gap = max(g[0] for g in gaps)
    best_actions = [a for g, a in gaps if abs(float(g) - float(max_gap)) <= 1e-9 * max(1.0, abs(float(max_gap)))]
    chosen = int(rng.choice(best_actions))
    return chosen, {"sym/oracle_best_reward": float(oracle_best_mean), "sym/target_gap": float(max_gap)}


def _symbolic_dt_attack_state(
    args: argparse.Namespace,
    *,
    actor: network.Network,
    pgd_attacker: PGDAttack,
    clean_state_next: np.ndarray,
    oracle_env_snapshot: env.Environment,
    last_bitrate_idx: int,
    video_chunk_size_bytes: int,
    allow_delay_idx: list[int],
    rng: np.random.RandomState,
) -> tuple[np.ndarray, dict[str, float | int]]:
    metrics: dict[str, float | int] = {
        "sym/attempted": 1,
        "sym/applied": 0,
        "sym/rule_found": 0,
        "sym/projection_success": 0,
        "sym/teacher_check_pass": 0,
        "sym/tries": 0,
    }

    p_clean = actor.predict(np.reshape(clean_state_next, (1, S_INFO, S_LEN))).astype(np.float64).reshape(-1)
    teacher_clean_action = int(np.argmax(p_clean))
    metrics["sym/teacher_action_clean"] = int(teacher_clean_action)

    target_action, target_stats = _select_target_action_by_oracle_gap(
        actor=actor,
        env_snapshot=oracle_env_snapshot,
        state_snapshot=clean_state_next,
        last_br=int(last_bitrate_idx),
        k_steps=K_steps,
        action_selection=str(args.action_selection),
        rng=rng,
            )
    for k, v in target_stats.items():
        metrics[str(k)] = float(v)
    if target_action is None:
        metrics["sym/target_action"] = -1
        return np.array(clean_state_next, copy=True), metrics

    metrics["sym/target_action"] = int(target_action)
    target_bitrate_kbps = int(VIDEO_BIT_RATE[int(target_action)])

    rules_file = _resolve_under_src(str(getattr(args, "policy_rules_file")))
    rules_by_bitrate = load_policy_rules(str(rules_file))

    values, delay_ms_hist, thr_mbps_hist, chosen_chunk_bytes_hist = _policy_rule_inputs_from_state(
        clean_state_next,
        last_bitrate_idx=int(last_bitrate_idx),
        video_chunk_size_bytes=int(video_chunk_size_bytes),
    )

    py_rng = random.Random(int(rng.randint(0, 2**31 - 1)))
    rule_candidates = enumerate_delay_perturbation_candidates(
        rules_by_bitrate=rules_by_bitrate,
        target_bitrate_kbps=int(target_bitrate_kbps),
        values=values,
        delay_ms_hist=delay_ms_hist,
        throughput_mbps_hist=thr_mbps_hist,
        chosen_chunk_bytes_hist=chosen_chunk_bytes_hist,
        allow_delay_idx=[int(i) for i in allow_delay_idx],
        eps_delay_ms=float(getattr(args, "sym_eps_delay_ms", 1e-6)),
        rng=py_rng,
        max_rules=None,
    )
    metrics["sym/rule_candidates"] = int(len(rule_candidates))
    if len(rule_candidates) == 0:
        return np.array(clean_state_next, copy=True), metrics

    metrics["sym/rule_found"] = 1
    history_len = int(clean_state_next.shape[1])
    bitrate_kbps_hist = _state_bitrate_kbps_hist(clean_state_next)
    buffer_seconds_hist = [float(clean_state_next[1, i]) * float(BUFFER_NORM_FACTOR) for i in range(history_len)]
    fix_buffer_hist = [True] * int(history_len)

    #* Sparse (L0-first) candidate search: minimize the number of perturbed delay history values.
    max_rules = int(getattr(args, "sym_max_rules", 50))
    max_tries = int(getattr(args, "sym_max_tries", 20))
    samples_per_rule = int(getattr(args, "sym_samples_per_rule", 2))
    min_delay_ms = float(getattr(args, "sym_min_delay_ms", 1e-6))
    eps_delay_ms = float(getattr(args, "sym_eps_delay_ms", 1e-6))
    eps_change_ms = float(max(float(eps_delay_ms), 1e-6))

    allow = sorted({int(i) for i in allow_delay_idx if 0 <= int(i) < int(history_len)})
    metrics["sym/l0_allow_k"] = int(len(allow))
    if len(allow) == 0:
        return np.array(clean_state_next, copy=True), metrics

    cur_delay = [float(x) for x in delay_ms_hist]
    cur_thr = [float(x) for x in thr_mbps_hist]

    prefer_increase_delay = int(target_action) < int(teacher_clean_action)
    prefer_decrease_delay = int(target_action) > int(teacher_clean_action)
    direction = 1 if prefer_increase_delay else (-1 if prefer_decrease_delay else 0)
    metrics["sym/delay_direction"] = int(direction)

    def _interval(rule_cand, idx: int) -> tuple[float, float]:
        interval = rule_cand.intervals.get(int(idx), (-float("inf"), float("inf")))
        return float(interval[0]), float(interval[1])

    def _l1_delay_ms(delay_list: list[float]) -> float:
        l1 = 0.0
        for i in allow:
            l1 += abs(float(delay_list[i]) - float(cur_delay[i]))
        return float(l1)

    def _support(delay_list: list[float]) -> list[int]:
        out: list[int] = []
        for i in allow:
            if abs(float(delay_list[i]) - float(cur_delay[i])) > float(eps_change_ms):
                out.append(int(i))
        return out

    def _mask(indices: list[int]) -> int:
        m = 0
        for i in indices:
            if 0 <= int(i) < 63:
                m |= 1 << int(i)
        return int(m)

    #* Build per-rule metadata: required support size (L0) and minimal L1 to satisfy the rule.
    rule_infos: list[tuple[int, float, float, str, list[int], object]] = []
    for cand in rule_candidates:
        md5 = hashlib.md5(str(cand.rule.text).encode("utf-8")).hexdigest()[:8]
        required: list[int] = []
        req_l1 = 0.0
        for i in allow:
            d = float(cand.delay_ms_hist[int(i)])
            if abs(d - float(cur_delay[int(i)])) > float(eps_change_ms):
                required.append(int(i))
                req_l1 += abs(d - float(cur_delay[int(i)]))
        rule_infos.append((int(len(required)), float(req_l1), float(py_rng.random()), str(md5), required, cand))

    rule_infos.sort(key=lambda x: (int(x[0]), float(x[1]), float(x[2])))
    if max_rules > 0:
        rule_infos = list(rule_infos[: int(max_rules)])
    metrics["sym/rules_considered"] = int(len(rule_infos))
    if len(rule_infos) == 0:
        return np.array(clean_state_next, copy=True), metrics

    l0_min = min(int(k) for k, *_ in rule_infos)
    metrics["sym/l0_k_min"] = int(l0_min)
    metrics["sym/delay_l1_ms_min"] = float(min(l1 for _k, l1, *_ in rule_infos))

    def _can_perturb_optional(rule_cand, idx: int) -> bool:
        lb, ub = _interval(rule_cand, int(idx))
        lo = max(float(lb), float(min_delay_ms))
        hi = float(ub)
        if lo > hi:
            return False
        cur = float(cur_delay[int(idx)])
        if cur + float(eps_change_ms) <= hi:
            return True
        if cur - float(eps_change_ms) >= lo:
            return True
        return False

    def _optional_cost(rule_cand, idx: int) -> float:
        lb, ub = _interval(rule_cand, int(idx))
        cur = float(cur_delay[int(idx)])
        costs: list[float] = []
        if math.isfinite(lb):
            costs.append(abs(cur - float(lb)))
        if math.isfinite(ub):
            costs.append(abs(float(ub) - cur))
        if len(costs) > 0:
            return float(min(costs))
        return float(max(1.0, abs(cur) * 0.25))

    def _pick_min_optional(rule_cand, idx: int) -> float | None:
        lb, ub = _interval(rule_cand, int(idx))
        lo = max(float(lb), float(min_delay_ms))
        hi = float(ub)
        if lo > hi:
            return None
        cur = float(cur_delay[int(idx)])

        if direction >= 0:
            if cur + float(eps_change_ms) <= hi:
                return float(cur + float(eps_change_ms))
            if cur - float(eps_change_ms) >= lo:
                return float(cur - float(eps_change_ms))
            return None

        if cur - float(eps_change_ms) >= lo:
            return float(cur - float(eps_change_ms))
        if cur + float(eps_change_ms) <= hi:
            return float(cur + float(eps_change_ms))
        return None

    def _pick_low(rule_cand, idx: int) -> float | None:
        lb, ub = _interval(rule_cand, int(idx))
        cur = float(cur_delay[int(idx)])
        if math.isfinite(lb):
            val = float(max(float(lb), float(min_delay_ms)))
        else:
            step = float(max(1.0, abs(cur) * 0.25))
            val = float(max(float(min_delay_ms), float(cur) - float(step)))
        if val < float(lb) or val > float(ub):
            return None
        return float(val)

    def _pick_high(rule_cand, idx: int) -> float | None:
        lb, ub = _interval(rule_cand, int(idx))
        cur = float(cur_delay[int(idx)])
        if math.isfinite(ub):
            val = float(max(float(min_delay_ms), float(ub)))
        else:
            step = float(max(1.0, abs(cur) * 0.25))
            val = float(max(float(min_delay_ms), float(cur) + float(step)))
        if val < float(lb) or val > float(ub):
            return None
        return float(val)

    def _pick_sample(rule_cand, idx: int) -> float | None:
        lb, ub = _interval(rule_cand, int(idx))
        cur = float(cur_delay[int(idx)])
        scale = float(max(1.0, abs(cur) * 0.25))

        if math.isfinite(lb) and math.isfinite(ub):
            lo = float(max(float(lb), float(min_delay_ms)))
            hi = float(ub)
            if lo > hi:
                return None
            return float(py_rng.uniform(lo, hi))

        if math.isfinite(ub) and not math.isfinite(lb):
            hi = float(ub)
            lo = float(max(float(min_delay_ms), float(hi) - float(scale)))
            if lo > hi:
                return None
            return float(py_rng.uniform(lo, hi))

        if math.isfinite(lb) and not math.isfinite(ub):
            lo = float(max(float(lb), float(min_delay_ms)))
            hi = float(lo + float(scale))
            if lo > hi:
                return None
            return float(py_rng.uniform(lo, hi))

        lo = float(max(float(min_delay_ms), float(cur) - float(scale)))
        hi = float(max(lo, float(cur) + float(scale)))
        return float(py_rng.uniform(lo, hi))

    def _candidate_throughput(delay_list: list[float]) -> list[float] | None:
        out: list[float] = []
        for d_ms, chunk_bytes in zip(delay_list, chosen_chunk_bytes_hist):
            d = float(d_ms)
            if d <= float(min_delay_ms):
                return None
            out.append(float(chunk_bytes) * 8.0 / (1000.0 * d))
        return out

    tries = 0
    projection_success_any = False

    best_teacher_p = float("-inf")
    best_teacher_action = -1
    best_rule_md5: str | None = None
    best_l1: float | None = None
    best_k_pre: int | None = None
    best_k_post: int | None = None
    best_mask_pre: int | None = None
    best_mask_post: int | None = None

    #* Rank candidates across all rules and all L0 budgets together (rather than per-rule or per-k_target).
    #* New ranking:
    #*   1) maximize p(target_action) under the pre-projection policy
    #*   2) minimize L0 (number of perturbed indices)
    #*   3) minimize L1 (sum of absolute delay changes)
    #* Also filter out candidates that do not already induce the target argmax pre-projection.
    scored_all: list[
        tuple[
            tuple[float, int, float, float],
            float,
            int,
            float,
            str,
            list[float],
            list[float],
            list[int],
            int,
        ]
    ] = []
    #* (sort_key, p_pre_target, l0_k, l1_ms, rule_md5, delay_ms_hist, thr_mbps_hist, support_idx, kind_code)

    total_generated = 0
    total_filtered_non_target = 0

    for k_target in range(int(l0_min), int(len(allow)) + 1):
        candidates: list[tuple[float, str, list[float], list[int], int]] = []
        #* (l1_ms, rule_md5, delay_ms_hist, support_idx, kind_code)

        for req_k, _, _, rule_md5, required_idx, rule_cand in rule_infos:
            if int(req_k) > int(k_target):
                continue

            extras_needed = int(k_target) - int(req_k)
            optional_idx = [
                int(i) for i in allow if int(i) not in set(required_idx) and _can_perturb_optional(rule_cand, int(i))
            ]
            optional_idx.sort(key=lambda i: _optional_cost(rule_cand, int(i)))
            if int(len(optional_idx)) < int(extras_needed):
                continue
            extras = list(optional_idx[: int(extras_needed)])

            def _build(kind: str) -> list[float] | None:
                out_delay = list(cur_delay)
                for i in required_idx:
                    val = float(rule_cand.delay_ms_hist[int(i)])
                    lb, ub = _interval(rule_cand, int(i))
                    if val < float(min_delay_ms):
                        if float(lb) <= float(min_delay_ms) <= float(ub):
                            val = float(min_delay_ms)
                        else:
                            return None
                    if val < float(lb) or val > float(ub):
                        return None
                    out_delay[int(i)] = float(val)

                for i in extras:
                    if kind == "min":
                        pick = _pick_min_optional(rule_cand, int(i))
                        if pick is None:
                            return None
                        out_delay[int(i)] = float(pick)
                        continue

                    if kind == "low":
                        pick = _pick_low(rule_cand, int(i))
                        if pick is None:
                            return None
                        out_delay[int(i)] = float(pick)
                        continue

                    if kind == "high":
                        pick = _pick_high(rule_cand, int(i))
                        if pick is None:
                            return None
                        out_delay[int(i)] = float(pick)
                        continue

                    if kind == "sample":
                        pick = _pick_sample(rule_cand, int(i))
                        if pick is None:
                            return None
                        out_delay[int(i)] = float(pick)
                        continue

                    return None

                #* For required indices, allow magnitude escalation too (still rule-satisfying).
                if kind in ("low", "high", "sample"):
                    for i in required_idx:
                        if kind == "low":
                            pick = _pick_low(rule_cand, int(i))
                        elif kind == "high":
                            pick = _pick_high(rule_cand, int(i))
                        else:
                            pick = _pick_sample(rule_cand, int(i))
                        if pick is None:
                            continue
                        out_delay[int(i)] = float(pick)

                return out_delay

            #* Always include the minimal-magnitude candidate for this (rule, k_target) pair.
            for kind_code, kind in [(0, "min"), (1, "low"), (2, "high")]:
                d = _build(kind)
                if d is None:
                    continue
                supp = _support(d)
                if int(len(supp)) != int(k_target):
                    continue
                l1 = _l1_delay_ms(d)
                candidates.append((float(l1), str(rule_md5), d, supp, int(kind_code)))

            for _ in range(max(0, int(samples_per_rule))):
                d = _build("sample")
                if d is None:
                    continue
                supp = _support(d)
                if int(len(supp)) != int(k_target):
                    continue
                l1 = _l1_delay_ms(d)
                candidates.append((float(l1), str(rule_md5), d, supp, 3))

        total_generated += int(len(candidates))
        if len(candidates) == 0:
            continue

        #* Score by pre-projection teacher output and keep only candidates that already hit the target argmax.
        for l1_ms, rule_md5, d_ms_hist, supp, kind_code in candidates:
            thr_hist = _candidate_throughput(d_ms_hist)
            if thr_hist is None:
                continue

            st = np.array(clean_state_next, copy=True)
            for i in range(history_len):
                st[3, i] = float(d_ms_hist[i]) / (float(M_IN_K) * float(BUFFER_NORM_FACTOR))
                st[2, i] = float(thr_hist[i]) / 8.0
            p_pre = actor.predict(np.reshape(st, (1, S_INFO, S_LEN))).astype(np.float64).reshape(-1)
            if p_pre.shape[0] != int(A_DIM):
                continue
            a_pre = int(np.argmax(p_pre))
            if int(a_pre) != int(target_action):
                total_filtered_non_target += 1
                continue
            p_t = float(p_pre[int(target_action)])
            l0_k = int(len(supp))
            sort_key = (-float(p_t), int(l0_k), float(l1_ms), float(py_rng.random()))
            scored_all.append(
                (sort_key, float(p_t), int(l0_k), float(l1_ms), str(rule_md5), d_ms_hist, thr_hist, supp, int(kind_code))
            )

    scored_all.sort(key=lambda x: x[0])
    metrics["sym/candidates_generated"] = int(total_generated)
    metrics["sym/candidates_scored"] = int(len(scored_all))
    metrics["sym/candidates_filtered_non_target"] = int(total_filtered_non_target)

    for _, _p_pre_t, _l0_k, l1_ms, rule_md5, d_ms_hist, cand_thr_mbps_hist, supp, kind_code in scored_all:
        if max_tries > 0 and int(tries) >= int(max_tries):
            break
        tries += 1

        if cand_thr_mbps_hist is None:
            continue

        fix_delay_hist = [True] * int(history_len)
        for idx in supp:
            if 0 <= int(idx) < int(history_len):
                fix_delay_hist[int(idx)] = False

        sol_delay_ms_hist: list[float] | None = None
        sol_thr_mbps_hist: list[float] | None = None
        if bool(getattr(args, "use_rules", False)):
            sol = pgd_attacker._project_with_rules(
                rules_file=str(args.rules_file),
                bitrate_kbps_hist=[int(x) for x in bitrate_kbps_hist],
                buffer_seconds_hist=buffer_seconds_hist,
                fix_buffer_hist=fix_buffer_hist,
                fix_delay_hist=fix_delay_hist,
                buffer_seconds_max=float(BUFFER_NORM_FACTOR) * float(BUFFER_NORM_FACTOR),
                chosen_chunk_bytes_hist=[int(x) for x in chosen_chunk_bytes_hist],
                delay_ms_hist=[float(x) for x in d_ms_hist],
                throughput_mbps_hist=[float(x) for x in cand_thr_mbps_hist],
            )
            if sol is None:
                continue
            projection_success_any = True
            sol_delay_ms_hist, sol_thr_mbps_hist, _sol_buf_s_hist = sol
        else:
            #* When `--no-rules`, skip the environment-rule projection stage entirely.
            #* The candidate still enforces per-step delay/throughput coupling via `_candidate_throughput`.
            sol_delay_ms_hist = [float(x) for x in d_ms_hist]
            sol_thr_mbps_hist = [float(x) for x in cand_thr_mbps_hist]
        assert sol_delay_ms_hist is not None
        assert sol_thr_mbps_hist is not None

        attacked_state = np.array(clean_state_next, copy=True)
        for i in range(history_len):
            attacked_state[3, i] = float(sol_delay_ms_hist[i]) / (float(M_IN_K) * float(BUFFER_NORM_FACTOR))
            attacked_state[2, i] = float(sol_thr_mbps_hist[i]) / 8.0

        p = actor.predict(np.reshape(attacked_state, (1, S_INFO, S_LEN))).astype(np.float64).reshape(-1)
        teacher_action = int(np.argmax(p))
        p_t = float(p[int(target_action)]) if p.shape[0] == int(A_DIM) else float("nan")

        k_pre = int(len(supp))
        mask_pre = _mask([int(x) for x in supp])

        post_supp: list[int] = []
        for i in allow:
            if abs(float(sol_delay_ms_hist[int(i)]) - float(cur_delay[int(i)])) > float(eps_change_ms):
                post_supp.append(int(i))
        k_post = int(len(post_supp))
        mask_post = _mask([int(x) for x in post_supp])

        if math.isfinite(p_t) and float(p_t) > float(best_teacher_p):
            best_teacher_p = float(p_t)
            best_teacher_action = int(teacher_action)
            best_rule_md5 = str(rule_md5)
            best_l1 = float(l1_ms)
            best_k_pre = int(k_pre)
            best_k_post = int(k_post)
            best_mask_pre = int(mask_pre)
            best_mask_post = int(mask_post)

        if int(teacher_action) == int(target_action):
            metrics["sym/projection_success"] = 1
            metrics["sym/teacher_action"] = int(teacher_action)
            metrics["sym/teacher_check_pass"] = 1
            metrics["sym/applied"] = 1
            metrics["sym/tries"] = int(tries)
            metrics["sym/rule_md5"] = str(rule_md5)
            metrics["sym/delay_l1_ms"] = float(l1_ms)
            metrics["sym/l0_k_pre"] = int(k_pre)
            metrics["sym/l0_k_post"] = int(k_post)
            metrics["sym/l0_mask_pre"] = int(mask_pre)
            metrics["sym/l0_mask_post"] = int(mask_post)
            metrics["sym/candidate_kind"] = int(kind_code)
            return attacked_state, metrics
            # else:
            #     metrics["sym/projection_success"] = 0
            #     metrics["sym/teacher_action"] = int(teacher_action)
            #     metrics["sym/teacher_check_pass"] = 0
            #     metrics["sym/applied"] = 1
            #     metrics["sym/tries"] = int(tries)
            #     metrics["sym/rule_md5"] = str(rule_md5)
            #     metrics["sym/delay_l1_ms"] = float(l1_ms)
            #     metrics["sym/l0_k_pre"] = int(k_pre)
            #     metrics["sym/l0_k_post"] = int(k_post)
            #     metrics["sym/l0_mask_pre"] = int(mask_pre)
            #     metrics["sym/l0_mask_post"] = int(mask_post)
            #     metrics["sym/candidate_kind"] = int(kind_code)
            #     return attacked_state, metrics

    metrics["sym/tries"] = int(tries)
    metrics["sym/projection_success"] = int(projection_success_any)
    metrics["sym/teacher_check_pass"] = 0
    metrics["sym/teacher_action"] = int(best_teacher_action)
    if best_rule_md5 is not None:
        metrics["sym/rule_md5"] = str(best_rule_md5)
    if best_l1 is not None:
        metrics["sym/delay_l1_ms"] = float(best_l1)
    if best_k_pre is not None:
        metrics["sym/l0_k_pre"] = int(best_k_pre)
    if best_k_post is not None:
        metrics["sym/l0_k_post"] = int(best_k_post)
    if best_mask_pre is not None:
        metrics["sym/l0_mask_pre"] = int(best_mask_pre)
    if best_mask_post is not None:
        metrics["sym/l0_mask_post"] = int(best_mask_post)
    return np.array(clean_state_next, copy=True), metrics


def _policy_start_action_from_state(
    *,
    actor: network.Network,
    state_snapshot: np.ndarray,
    action_selection: str,
    rng: np.random.RandomState,
) -> tuple[int, np.ndarray]:
    p = actor.predict(np.reshape(state_snapshot, (1, S_INFO, S_LEN))).astype(np.float64).reshape(-1)
    if p.shape[0] != int(A_DIM):
        return int(DEFAULT_QUALITY), np.array(p, copy=True)

    if str(action_selection) == "argmax":
        return int(np.argmax(p)), np.array(p, copy=True)
    if str(action_selection) == "gumbel":
        noise = rng.gumbel(size=p.shape)
        return int(np.argmax(np.log(p + 1e-12) + noise)), np.array(p, copy=True)

    raise ValueError(f"Unknown action_selection: {action_selection}")


def _policy_k_step_reward_mean_follow_policy(
    *,
    actor: network.Network,
    env_snapshot: env.Environment,
    state_snapshot: np.ndarray,
    last_br: int,
    k_steps: int,
    action_selection: str,
    rng: np.random.RandomState,
) -> tuple[float | None, int, float]:
    start_action, p0 = _policy_start_action_from_state(
        actor=actor, state_snapshot=state_snapshot, action_selection=str(action_selection), rng=rng
    )
    p_start = float(p0[int(start_action)]) if p0.shape[0] == int(A_DIM) else float("nan")
    pr = _policy_k_step_reward_mean(
        actor=actor,
        env_snapshot=env_snapshot,
        state_snapshot=state_snapshot,
        last_br=int(last_br),
        start_action=int(start_action),
        k_steps=int(k_steps),
        action_selection=str(action_selection),
        rng=rng,
    )
    return pr, int(start_action), float(p_start)


def _symbolic_dt_truegap_attack_state(
    args: argparse.Namespace,
    *,
    actor: network.Network,
    pgd_attacker: PGDAttack,
    clean_state_next: np.ndarray,
    clean_env_snapshot: env.Environment,
    env_before_chunk: env.Environment,
    state_before_chunk: np.ndarray,
    last_bitrate_idx: int,
    video_chunk_size_bytes: int,
    allow_delay_idx: list[int],
    rng: np.random.RandomState,
) -> tuple[np.ndarray, dict[str, float | int], list[float] | None, list[float] | None]:
    """
    * Symbolic DT candidate search, but rank candidates by the *true* oracle gap computed on the
      realized attacked environment (SMT + simulator oracle), with an oracle-retention safeguard.

    Returns:
      (target_state_next, metrics, selected_bw_trace, selected_t_trace)
    """
    try:
        from smt import synthesize_bandwidth_trace
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "* Failed to import SMT solver. Install deps (e.g., `pip install z3-solver`) and retry."
        ) from e

    metrics: dict[str, float | int] = {
        "sym/attempted": 1,
        "sym/applied": 0,
        "sym/rule_found": 0,
        "sym/projection_success": 0,
        "sym/teacher_check_pass": 0,
        "sym/tries": 0,
        "symtrue/evals_attempted": 0,
        "symtrue/evals_smt_success": 0,
        "symtrue/evals_override_success": 0,
        "symtrue/evals_oracle_valid": 0,
        "symtrue/evals_policy_valid": 0,
        "symtrue/evals_oracle_filter_pass": 0,
    }

    p_clean = actor.predict(np.reshape(clean_state_next, (1, S_INFO, S_LEN))).astype(np.float64).reshape(-1)
    teacher_clean_action = int(np.argmax(p_clean))
    metrics["sym/teacher_action_clean"] = int(teacher_clean_action)

    best_r_clean, _best_seq, _worst_r_clean, _worst_seq = find_optimal_k_step_reward_oracle(
        clean_env_snapshot, int(last_bitrate_idx), int(K_steps)
    )
    if best_r_clean is None:
        metrics["symtrue/clean_oracle_valid"] = 0
        return np.array(clean_state_next, copy=True), metrics, None, None

    clean_oracle_best_mean = float(best_r_clean) / float(K_steps)
    metrics["symtrue/clean_oracle_best_reward_mean"] = float(clean_oracle_best_mean)

    oracle_min_ratio = float(getattr(args, "sym_truegap_oracle_min_ratio", 0.0))
    oracle_max_drop = float(getattr(args, "sym_truegap_oracle_max_drop", -1.0))
    metrics["symtrue/oracle_min_ratio"] = float(oracle_min_ratio)
    metrics["symtrue/oracle_max_drop"] = float(oracle_max_drop)

    oracle_min = -float("inf")
    if float(oracle_max_drop) >= 0.0:
        oracle_min = max(float(oracle_min), float(clean_oracle_best_mean) - float(oracle_max_drop))
    if float(oracle_min_ratio) > 0.0 and float(clean_oracle_best_mean) > 0.0:
        oracle_min = max(float(oracle_min), float(clean_oracle_best_mean) * float(oracle_min_ratio))
    metrics["symtrue/oracle_min_required"] = float(oracle_min)

    clean_policy_mean, clean_policy_start_action, clean_policy_start_p = _policy_k_step_reward_mean_follow_policy(
        actor=actor,
        env_snapshot=clean_env_snapshot,
        state_snapshot=clean_state_next,
        last_br=int(last_bitrate_idx),
        k_steps=int(K_steps),
        action_selection=str(args.action_selection),
        rng=rng,
    )
    metrics["symtrue/clean_policy_reward_mean"] = (
        float(clean_policy_mean) if clean_policy_mean is not None else float("nan")
    )
    metrics["symtrue/clean_policy_start_action"] = int(clean_policy_start_action)
    metrics["symtrue/clean_policy_start_p"] = float(clean_policy_start_p)
    clean_gap_true: float | None = None
    if clean_policy_mean is not None:
        clean_gap_true = float(clean_oracle_best_mean) - float(clean_policy_mean)
        metrics["symtrue/clean_gap_true"] = float(clean_gap_true)
    else:
        metrics["symtrue/clean_gap_true"] = float("nan")

    history_len = int(clean_state_next.shape[1])
    allow = sorted({int(i) for i in allow_delay_idx if 0 <= int(i) < int(history_len)})
    metrics["sym/l0_allow_k"] = int(len(allow))
    if len(allow) == 0:
        return np.array(clean_state_next, copy=True), metrics, None, None

    if int(history_len) - 1 not in allow:
        raise ValueError("* attack-method dt-truegap requires allow_delay_idx to include the latest delay index")

    rules_file = _resolve_under_src(str(getattr(args, "policy_rules_file")))
    rules_by_bitrate = load_policy_rules(str(rules_file))

    values, delay_ms_hist, thr_mbps_hist, chosen_chunk_bytes_hist = _policy_rule_inputs_from_state(
        clean_state_next,
        last_bitrate_idx=int(last_bitrate_idx),
        video_chunk_size_bytes=int(video_chunk_size_bytes),
    )

    bitrate_kbps_hist = _state_bitrate_kbps_hist(clean_state_next)
    buffer_seconds_hist = [float(clean_state_next[1, i]) * float(BUFFER_NORM_FACTOR) for i in range(history_len)]
    fix_buffer_hist = [True] * int(history_len)

    max_rules = int(getattr(args, "sym_max_rules", 50))
    samples_per_rule = int(getattr(args, "sym_samples_per_rule", 2))
    min_delay_ms = float(getattr(args, "sym_min_delay_ms", 1e-6))
    eps_delay_ms = float(getattr(args, "sym_eps_delay_ms", 1e-6))
    eps_change_ms = float(max(float(eps_delay_ms), 1e-6))

    max_candidates_per_action = int(getattr(args, "sym_truegap_max_candidates_per_action", 5))
    max_candidates_total = 0
    if int(max_candidates_per_action) > 0:
        max_candidates_total = int(max_candidates_per_action) * int(A_DIM)
    metrics["symtrue/max_candidates"] = int(max_candidates_total)
    metrics["symtrue/max_candidates_per_action"] = int(max_candidates_per_action)
    #* Backward-compatible metric aliases.
    metrics["symtrue/max_evals"] = int(max_candidates_total)
    metrics["symtrue/max_evals_per_action"] = int(max_candidates_per_action)

    cur_delay = [float(x) for x in delay_ms_hist]

    def _candidate_throughput(delay_list: list[float]) -> list[float] | None:
        out: list[float] = []
        for d_ms, chunk_bytes in zip(delay_list, chosen_chunk_bytes_hist):
            d = float(d_ms)
            if d <= float(min_delay_ms):
                return None
            out.append(float(chunk_bytes) * 8.0 / (1000.0 * d))
        return out

    def _l1_delay_ms(delay_list: list[float]) -> float:
        l1 = 0.0
        for i in allow:
            l1 += abs(float(delay_list[i]) - float(cur_delay[i]))
        return float(l1)

    def _support(delay_list: list[float]) -> list[int]:
        out: list[int] = []
        for i in allow:
            if abs(float(delay_list[i]) - float(cur_delay[i])) > float(eps_change_ms):
                out.append(int(i))
        return out

    def _mask(indices: list[int]) -> int:
        m = 0
        for i in indices:
            if 0 <= int(i) < 63:
                m |= 1 << int(i)
        return int(m)

    best_gap = float("-inf")
    best_oracle_mean = float("-inf")
    best_candidate: tuple[
        np.ndarray, list[float] | None, list[float] | None, dict[str, float | int]
    ] | None = None

    #* Always score "no perturbation" for the teacher's clean action as a candidate.
    if clean_gap_true is not None and np.isfinite(float(clean_gap_true)):
        best_gap = float(clean_gap_true)
        best_oracle_mean = float(clean_oracle_best_mean)
        best_candidate = (
            np.array(clean_state_next, copy=True),
            None,
            None,
            {
                "sym/teacher_action": int(teacher_clean_action),
                "sym/teacher_check_pass": 1,
                "sym/rule_md5": "clean",
                "sym/delay_l1_ms": 0.0,
                "sym/l0_k_pre": 0,
                "sym/l0_k_post": 0,
                "sym/l0_mask_pre": 0,
                "sym/l0_mask_post": 0,
                "sym/candidate_kind": -1,
                "symtrue/gap_true": float(clean_gap_true),
                "symtrue/oracle_best_reward_mean": float(clean_oracle_best_mean),
                "symtrue/policy_reward_mean": float(clean_policy_mean),
                "symtrue/policy_start_action": int(clean_policy_start_action),
                "symtrue/policy_start_p": float(clean_policy_start_p),
                "symtrue/smt_solve_time_ms_scoring": 0.0,
                **(
                    {"symtrue/smt_solve_time_ms_selected": 0.0}
                    if not bool(getattr(args, "smt_trace_rules", True))
                    else {}
                ),
            },
        )

    total_evals = 0
    total_candidates_generated = 0
    total_candidates_scored = 0
    total_candidates_filtered_non_target = 0
    projection_success_any = False

    initial_buffer_ms = float(getattr(env_before_chunk, "buffer_size", 0.0))
    #* dt-truegap: trace rules are expensive; only enforce them during the final trace realization
    #* (outside oracle-gap scoring). During scoring we synthesize without mined trace rules.
    score_enforce_trace_rules = False

    for target_action in range(int(A_DIM)):
        target_bitrate_kbps = int(VIDEO_BIT_RATE[int(target_action)])

        py_rng = random.Random(int(rng.randint(0, 2**31 - 1)))
        rule_candidates = enumerate_delay_perturbation_candidates(
            rules_by_bitrate=rules_by_bitrate,
            target_bitrate_kbps=int(target_bitrate_kbps),
            values=values,
            delay_ms_hist=delay_ms_hist,
            throughput_mbps_hist=thr_mbps_hist,
            chosen_chunk_bytes_hist=chosen_chunk_bytes_hist,
            allow_delay_idx=[int(i) for i in allow],
            eps_delay_ms=float(eps_delay_ms),
            rng=py_rng,
            max_rules=None,
        )
        if len(rule_candidates) == 0:
            continue

        metrics["sym/rule_found"] = 1

        #* Build per-rule metadata: required support size (L0) and minimal L1 to satisfy the rule.
        rule_infos: list[tuple[int, float, float, str, list[int], object]] = []
        for cand in rule_candidates:
            md5 = hashlib.md5(str(cand.rule.text).encode("utf-8")).hexdigest()[:8]
            required: list[int] = []
            req_l1 = 0.0
            for i in allow:
                d = float(cand.delay_ms_hist[int(i)])
                if abs(d - float(cur_delay[int(i)])) > float(eps_change_ms):
                    required.append(int(i))
                    req_l1 += abs(d - float(cur_delay[int(i)]))
            rule_infos.append((int(len(required)), float(req_l1), float(py_rng.random()), str(md5), required, cand))

        rule_infos.sort(key=lambda x: (int(x[0]), float(x[1]), float(x[2])))
        if max_rules > 0:
            rule_infos = list(rule_infos[: int(max_rules)])
        if len(rule_infos) == 0:
            continue

        l0_min = min(int(k) for k, *_ in rule_infos)

        def _interval(rule_cand, idx: int) -> tuple[float, float]:
            interval = rule_cand.intervals.get(int(idx), (-float("inf"), float("inf")))
            return float(interval[0]), float(interval[1])

        prefer_increase_delay = int(target_action) < int(teacher_clean_action)
        prefer_decrease_delay = int(target_action) > int(teacher_clean_action)
        direction = 1 if prefer_increase_delay else (-1 if prefer_decrease_delay else 0)

        def _pick_min_optional(rule_cand, idx: int) -> float | None:
            lb, ub = _interval(rule_cand, int(idx))
            lo = max(float(lb), float(min_delay_ms))
            hi = float(ub)
            if lo > hi:
                return None
            cur = float(cur_delay[int(idx)])

            if direction >= 0:
                if cur + float(eps_change_ms) <= hi:
                    return float(cur + float(eps_change_ms))
                if cur - float(eps_change_ms) >= lo:
                    return float(cur - float(eps_change_ms))
                return None

            if cur - float(eps_change_ms) >= lo:
                return float(cur - float(eps_change_ms))
            if cur + float(eps_change_ms) <= hi:
                return float(cur + float(eps_change_ms))
            return None

        def _pick_low(rule_cand, idx: int) -> float | None:
            lb, ub = _interval(rule_cand, int(idx))
            cur = float(cur_delay[int(idx)])
            if math.isfinite(lb):
                val = float(max(float(lb), float(min_delay_ms)))
            else:
                step = float(max(1.0, abs(cur) * 0.25))
                val = float(max(float(min_delay_ms), float(cur) - float(step)))
            if val < float(lb) or val > float(ub):
                return None
            return float(val)

        def _pick_high(rule_cand, idx: int) -> float | None:
            lb, ub = _interval(rule_cand, int(idx))
            cur = float(cur_delay[int(idx)])
            if math.isfinite(ub):
                val = float(max(float(min_delay_ms), float(ub)))
            else:
                step = float(max(1.0, abs(cur) * 0.25))
                val = float(max(float(min_delay_ms), float(cur) + float(step)))
            if val < float(lb) or val > float(ub):
                return None
            return float(val)

        def _pick_sample(rule_cand, idx: int) -> float | None:
            lb, ub = _interval(rule_cand, int(idx))
            cur = float(cur_delay[int(idx)])
            scale = float(max(1.0, abs(cur) * 0.25))

            if math.isfinite(lb) and math.isfinite(ub):
                lo = float(max(float(lb), float(min_delay_ms)))
                hi = float(ub)
                if lo > hi:
                    return None
                return float(py_rng.uniform(lo, hi))

            if math.isfinite(ub) and not math.isfinite(lb):
                hi = float(ub)
                lo = float(max(float(min_delay_ms), float(hi) - float(scale)))
                if lo > hi:
                    return None
                return float(py_rng.uniform(lo, hi))

            if math.isfinite(lb) and not math.isfinite(ub):
                lo = float(max(float(lb), float(min_delay_ms)))
                hi = float(lo + float(scale))
                if lo > hi:
                    return None
                return float(py_rng.uniform(lo, hi))

            lo = float(max(float(min_delay_ms), float(cur) - float(scale)))
            hi = float(max(lo, float(cur) + float(scale)))
            return float(py_rng.uniform(lo, hi))

        scored_all: list[
            tuple[
                tuple[float, int, float, float],
                float,
                int,
                float,
                str,
                list[float],
                list[float],
                list[int],
                int,
            ]
        ] = []
        #* (sort_key, p_pre_target, l0_k, l1_ms, rule_md5, delay_ms_hist, thr_mbps_hist, support_idx, kind_code)

        for k_target in range(int(l0_min), int(len(allow)) + 1):
            candidates: list[tuple[float, str, list[float], list[int], int]] = []
            #* (l1_ms, rule_md5, delay_ms_hist, support_idx, kind_code)

            for req_k, _, _, rule_md5, required_idx, rule_cand in rule_infos:
                if int(req_k) > int(k_target):
                    continue

                extras_needed = int(k_target) - int(req_k)
                optional_idx = [int(i) for i in allow if int(i) not in set(required_idx)]
                if int(len(optional_idx)) < int(extras_needed):
                    continue
                extras = list(optional_idx[: int(extras_needed)])

                def _build(kind: str) -> list[float] | None:
                    out_delay = list(cur_delay)
                    for i in required_idx:
                        val = float(rule_cand.delay_ms_hist[int(i)])
                        lb, ub = _interval(rule_cand, int(i))
                        if val < float(min_delay_ms):
                            if float(lb) <= float(min_delay_ms) <= float(ub):
                                val = float(min_delay_ms)
                            else:
                                return None
                        if val < float(lb) or val > float(ub):
                            return None
                        out_delay[int(i)] = float(val)

                    for i in extras:
                        if kind == "min":
                            pick = _pick_min_optional(rule_cand, int(i))
                            if pick is None:
                                return None
                            out_delay[int(i)] = float(pick)
                            continue
                        if kind == "low":
                            pick = _pick_low(rule_cand, int(i))
                            if pick is None:
                                return None
                            out_delay[int(i)] = float(pick)
                            continue
                        if kind == "high":
                            pick = _pick_high(rule_cand, int(i))
                            if pick is None:
                                return None
                            out_delay[int(i)] = float(pick)
                            continue
                        if kind == "sample":
                            pick = _pick_sample(rule_cand, int(i))
                            if pick is None:
                                return None
                            out_delay[int(i)] = float(pick)
                            continue
                        return None

                    #* For required indices, allow magnitude escalation too (still rule-satisfying).
                    if kind in ("low", "high", "sample"):
                        for i in required_idx:
                            if kind == "low":
                                pick = _pick_low(rule_cand, int(i))
                            elif kind == "high":
                                pick = _pick_high(rule_cand, int(i))
                            else:
                                pick = _pick_sample(rule_cand, int(i))
                            if pick is None:
                                continue
                            out_delay[int(i)] = float(pick)

                    return out_delay

                for kind_code, kind in [(0, "min"), (1, "low"), (2, "high")]:
                    d = _build(kind)
                    if d is None:
                        continue
                    supp = _support(d)
                    if int(len(supp)) != int(k_target):
                        continue
                    l1 = _l1_delay_ms(d)
                    candidates.append((float(l1), str(rule_md5), d, supp, int(kind_code)))

                for _ in range(max(0, int(samples_per_rule))):
                    d = _build("sample")
                    if d is None:
                        continue
                    supp = _support(d)
                    if int(len(supp)) != int(k_target):
                        continue
                    l1 = _l1_delay_ms(d)
                    candidates.append((float(l1), str(rule_md5), d, supp, 3))

            total_candidates_generated += int(len(candidates))
            if len(candidates) == 0:
                continue

            for l1_ms, rule_md5, d_ms_hist, supp, kind_code in candidates:
                thr_hist = _candidate_throughput(d_ms_hist)
                if thr_hist is None:
                    continue

                st = np.array(clean_state_next, copy=True)
                for i in range(history_len):
                    st[3, i] = float(d_ms_hist[i]) / (float(M_IN_K) * float(BUFFER_NORM_FACTOR))
                    st[2, i] = float(thr_hist[i]) / 8.0
                p_pre = actor.predict(np.reshape(st, (1, S_INFO, S_LEN))).astype(np.float64).reshape(-1)
                if p_pre.shape[0] != int(A_DIM):
                    continue
                a_pre = int(np.argmax(p_pre))
                if int(a_pre) != int(target_action):
                    total_candidates_filtered_non_target += 1
                    continue
                p_t = float(p_pre[int(target_action)])
                l0_k = int(len(supp))
                sort_key = (-float(p_t), int(l0_k), float(l1_ms), float(py_rng.random()))
                scored_all.append(
                    (
                        sort_key,
                        float(p_t),
                        int(l0_k),
                        float(l1_ms),
                        str(rule_md5),
                        d_ms_hist,
                        thr_hist,
                        supp,
                        int(kind_code),
                    )
                )

        scored_all.sort(key=lambda x: x[0])
        total_candidates_scored += int(len(scored_all))
        if len(scored_all) == 0:
            continue

        candidates_considered_for_action = 0
        for _, _p_pre_t, _l0_k, l1_ms, rule_md5, d_ms_hist, cand_thr_mbps_hist, supp, kind_code in scored_all:
            if int(max_candidates_per_action) > 0 and int(candidates_considered_for_action) >= int(max_candidates_per_action):
                break
            candidates_considered_for_action += 1

            metrics["sym/tries"] = int(metrics["sym/tries"]) + 1

            fix_delay_hist = [True] * int(history_len)
            for idx in supp:
                if 0 <= int(idx) < int(history_len):
                    fix_delay_hist[int(idx)] = False

            sol_delay_ms_hist: list[float] | None = None
            sol_thr_mbps_hist: list[float] | None = None
            if bool(getattr(args, "use_rules", False)):
                sol = pgd_attacker._project_with_rules(
                    rules_file=str(args.rules_file),
                    bitrate_kbps_hist=[int(x) for x in bitrate_kbps_hist],
                    buffer_seconds_hist=buffer_seconds_hist,
                    fix_buffer_hist=fix_buffer_hist,
                    fix_delay_hist=fix_delay_hist,
                    buffer_seconds_max=float(BUFFER_NORM_FACTOR) * float(BUFFER_NORM_FACTOR),
                    chosen_chunk_bytes_hist=[int(x) for x in chosen_chunk_bytes_hist],
                    delay_ms_hist=[float(x) for x in d_ms_hist],
                    throughput_mbps_hist=[float(x) for x in cand_thr_mbps_hist],
                )
                if sol is None:
                    continue
                projection_success_any = True
                sol_delay_ms_hist, sol_thr_mbps_hist, _sol_buf_s_hist = sol
            else:
                sol_delay_ms_hist = [float(x) for x in d_ms_hist]
                sol_thr_mbps_hist = [float(x) for x in cand_thr_mbps_hist]

            attacked_state_target = np.array(clean_state_next, copy=True)
            for i in range(history_len):
                attacked_state_target[3, i] = float(sol_delay_ms_hist[i]) / (float(M_IN_K) * float(BUFFER_NORM_FACTOR))
                attacked_state_target[2, i] = float(sol_thr_mbps_hist[i]) / 8.0

            p_target = actor.predict(np.reshape(attacked_state_target, (1, S_INFO, S_LEN))).astype(np.float64).reshape(-1)
            teacher_action_target = int(np.argmax(p_target))
            if int(teacher_action_target) != int(target_action):
                continue
            #* A candidate passes the teacher check (post-projection) even if SMT/oracle evaluation later fails.
            metrics["sym/teacher_check_pass"] = 1

            total_evals += 1
            metrics["symtrue/evals_attempted"] = int(metrics["symtrue/evals_attempted"]) + 1

            tgt_buf_norm = float(attacked_state_target[1, -1])
            tgt_thr_norm = float(attacked_state_target[2, -1])
            tgt_dly_norm = float(attacked_state_target[3, -1])
            if not (np.isfinite(tgt_buf_norm) and np.isfinite(tgt_thr_norm) and np.isfinite(tgt_dly_norm)):
                continue

            smt_t0 = time.perf_counter()
            bw_trace, t_trace = synthesize_bandwidth_trace(
                video_chunk_size=int(video_chunk_size_bytes),
                initial_buffer_ms=float(initial_buffer_ms),
                target_buffer_norm=float(tgt_buf_norm),
                target_throughput_norm=float(tgt_thr_norm),
                target_delay_norm=float(tgt_dly_norm),
                num_bw_samples=int(args.smt_bw_samples),
                enforce_buffer_target=bool(args.pgd_buffer),
                enforce_throughput_target=True,
                enforce_trace_rules=bool(score_enforce_trace_rules),
                rel_tol=float(args.smt_window_rel_tol),
            )
            smt_solve_ms = float((time.perf_counter() - smt_t0) * 1000.0)
            if bw_trace is None or t_trace is None:
                continue
            metrics["symtrue/evals_smt_success"] = int(metrics["symtrue/evals_smt_success"]) + 1

            bw = [float(x) for x in bw_trace]
            tt = [float(x) for x in t_trace]

            env_attacked = copy_env(env_before_chunk, RANDOM_SEED=int(rng.randint(0, 2**31 - 1)))
            overridden = _get_video_chunk_override(
                env_attacked,
                int(last_bitrate_idx),
                bandwidth_trace_mbps=bw,
                time_trace_s=tt,
            )
            if overridden is None:
                continue
            metrics["symtrue/evals_override_success"] = int(metrics["symtrue/evals_override_success"]) + 1

            (
                delay_ms,
                _sleep_time_ms,
                buffer_s,
                _rebuf_s,
                chunk_size_bytes,
                next_sizes,
                end_of_video,
                video_chunk_remain,
            ) = overridden
            if bool(end_of_video):
                continue

            realized_state_next = _update_state(
                np.array(state_before_chunk, copy=True),
                bit_rate=int(last_bitrate_idx),
                buffer_s=float(buffer_s),
                video_chunk_size=int(chunk_size_bytes),
                delay_ms=float(delay_ms),
                next_video_chunk_sizes=[int(x) for x in next_sizes],
                video_chunk_remain=int(video_chunk_remain),
            )

            best_r_att, _best_seq_att, _worst_r_att, _worst_seq_att = find_optimal_k_step_reward_oracle(
                env_attacked, int(last_bitrate_idx), int(K_steps)
            )
            if best_r_att is None:
                continue
            metrics["symtrue/evals_oracle_valid"] = int(metrics["symtrue/evals_oracle_valid"]) + 1
            oracle_best_mean = float(best_r_att) / float(K_steps)

            policy_mean, policy_start_action, policy_p_start = _policy_k_step_reward_mean_follow_policy(
                actor=actor,
                env_snapshot=env_attacked,
                state_snapshot=realized_state_next,
                last_br=int(last_bitrate_idx),
                k_steps=int(K_steps),
                action_selection=str(args.action_selection),
                rng=rng,
            )
            if policy_mean is None:
                continue
            metrics["symtrue/evals_policy_valid"] = int(metrics["symtrue/evals_policy_valid"]) + 1

            if float(oracle_best_mean) < float(oracle_min):
                continue
            metrics["symtrue/evals_oracle_filter_pass"] = int(metrics["symtrue/evals_oracle_filter_pass"]) + 1

            gap_true = float(oracle_best_mean) - float(policy_mean)

            post_supp: list[int] = []
            for i in allow:
                if abs(float(sol_delay_ms_hist[int(i)]) - float(cur_delay[int(i)])) > float(eps_change_ms):
                    post_supp.append(int(i))
            k_post = int(len(post_supp))
            mask_pre = _mask([int(x) for x in supp])
            mask_post = _mask([int(x) for x in post_supp])

            update = False
            if not math.isfinite(float(best_gap)) or float(gap_true) > float(best_gap) + 1e-12:
                update = True
            elif abs(float(gap_true) - float(best_gap)) <= 1e-12 and float(oracle_best_mean) > float(best_oracle_mean) + 1e-12:
                update = True
            elif (
                abs(float(gap_true) - float(best_gap)) <= 1e-12
                and abs(float(oracle_best_mean) - float(best_oracle_mean)) <= 1e-12
                and best_candidate is not None
                and float(l1_ms) < float(best_candidate[3].get("sym/delay_l1_ms", float("inf"))) - 1e-12
            ):
                update = True

            if update:
                best_gap = float(gap_true)
                best_oracle_mean = float(oracle_best_mean)
                best_candidate_metrics: dict[str, float | int] = {
                    "sym/target_action": int(target_action),
                    "sym/teacher_action": int(teacher_action_target),
                    "sym/teacher_check_pass": 1,
                    "sym/rule_md5": str(rule_md5),
                    "sym/delay_l1_ms": float(l1_ms),
                    "sym/l0_k_pre": int(len(supp)),
                    "sym/l0_k_post": int(k_post),
                    "sym/l0_mask_pre": int(mask_pre),
                    "sym/l0_mask_post": int(mask_post),
                    "sym/candidate_kind": int(kind_code),
                    "symtrue/gap_true": float(gap_true),
                    "symtrue/oracle_best_reward_mean": float(oracle_best_mean),
                    "symtrue/policy_reward_mean": float(policy_mean),
                    "symtrue/policy_start_action": int(policy_start_action),
                    "symtrue/policy_start_p": float(policy_p_start),
                    "symtrue/smt_solve_time_ms_scoring": float(smt_solve_ms),
                    **(
                        {"symtrue/smt_solve_time_ms_selected": float(smt_solve_ms)}
                        if not bool(getattr(args, "smt_trace_rules", True))
                        else {}
                    ),
                }
                best_candidate = (np.array(attacked_state_target, copy=True), bw, tt, best_candidate_metrics)

    metrics["sym/candidates_generated"] = int(total_candidates_generated)
    metrics["sym/candidates_scored"] = int(total_candidates_scored)
    metrics["sym/candidates_filtered_non_target"] = int(total_candidates_filtered_non_target)
    metrics["sym/projection_success"] = int(projection_success_any)

    if best_candidate is None:
        return np.array(clean_state_next, copy=True), metrics, None, None

    best_state, best_bw, best_tt, best_m = best_candidate
    metrics.update(best_m)
    metrics["sym/applied"] = int(int(best_m.get("sym/l0_k_post", 0)) > 0)
    if int(metrics["sym/applied"]) == 0:
        return np.array(clean_state_next, copy=True), metrics, None, None
    if bool(getattr(args, "smt_trace_rules", True)):
        #* Force final SMT realization (with trace rules) to happen outside dt-truegap scoring.
        #* This avoids paying the trace-rule cost per candidate.
        return np.array(best_state, copy=True), metrics, None, None

    if best_bw is None or best_tt is None:
        return np.array(best_state, copy=True), metrics, None, None
    return np.array(best_state, copy=True), metrics, list(best_bw), list(best_tt)


def _simulate_chunk_from_piecewise_bw(
    *,
    bandwidth_trace_mbps: list[float],
    time_trace_s: list[float],
    video_chunk_size_bytes: int,
    initial_buffer_ms: float,
) -> tuple[float, float, float, float, bool]:
    """
    * Simulate `fixed_env.Environment.get_video_chunk` for one chunk, but using a synthesized
      piecewise-constant bandwidth trace (bw_i Mbps for time_i seconds).

    Returns:
        delay_ms, sleep_time_ms, buffer_s, rebuf_s, complete
    """
    #* Match `fixed_env.Environment.get_video_chunk` semantics (treat exact-equality as completion).
    sent_bytes = 0.0
    delay_s = 0.0

    for bw_mbps, dur_s in zip(bandwidth_trace_mbps, time_trace_s):
        dur = float(dur_s)
        if not np.isfinite(dur) or dur <= 0.0:
            continue

        throughput_Bps = float(bw_mbps) * env.B_IN_MB / env.BITS_IN_BYTE
        if not np.isfinite(throughput_Bps) or throughput_Bps <= 0.0:
            #* Match `fixed_env.Environment.get_video_chunk`: time still elapses even at 0 Mbps.
            delay_s += float(dur)
            continue

        payload_bytes = throughput_Bps * float(dur) * env.PACKET_PAYLOAD_PORTION

        if float(sent_bytes) + float(payload_bytes) >= float(video_chunk_size_bytes):
            remaining_bytes = float(video_chunk_size_bytes) - float(sent_bytes)
            fractional_time_s = remaining_bytes / (throughput_Bps * env.PACKET_PAYLOAD_PORTION)
            fractional_time_s = float(max(min(fractional_time_s, dur), 0.0))
            delay_s += float(fractional_time_s)
            sent_bytes = float(video_chunk_size_bytes)
            break

        sent_bytes += float(payload_bytes)
        delay_s += float(dur)

    complete = float(sent_bytes) > float(video_chunk_size_bytes) - 1e-9
    delay_ms = delay_s * env.MILLISECONDS_IN_SECOND + env.LINK_RTT

    rebuf_ms = float(np.maximum(delay_ms - float(initial_buffer_ms), 0.0))
    buffer_ms = float(np.maximum(float(initial_buffer_ms) - delay_ms, 0.0))
    buffer_ms += env.VIDEO_CHUNCK_LEN

    sleep_time_ms = 0.0
    if buffer_ms > env.BUFFER_THRESH:
        drain_ms = buffer_ms - env.BUFFER_THRESH
        sleep_time_ms = float(np.ceil(drain_ms / env.DRAIN_BUFFER_SLEEP_TIME) * env.DRAIN_BUFFER_SLEEP_TIME)
        buffer_ms -= sleep_time_ms

    buffer_s = buffer_ms / env.MILLISECONDS_IN_SECOND
    rebuf_s = rebuf_ms / env.MILLISECONDS_IN_SECOND
    return float(delay_ms), float(sleep_time_ms), float(buffer_s), float(rebuf_s), bool(complete)


def _advance_trace_time(net_env: env.Environment, delta_s: float) -> None:
    """
    * Advance `fixed_env.Environment`'s Mahimahi pointer by `delta_s` seconds (trace time).
      This mirrors how `get_video_chunk` consumes `cooked_time` during download/sleep.
    """
    remaining_s = float(delta_s)
    eps = 1e-12

    while remaining_s > eps:
        if net_env.mahimahi_ptr >= len(net_env.cooked_bw):
            net_env.mahimahi_ptr = 1
            net_env.last_mahimahi_time = 0.0

        duration = float(net_env.cooked_time[net_env.mahimahi_ptr] - net_env.last_mahimahi_time)
        if duration <= eps:
            net_env.last_mahimahi_time = float(net_env.cooked_time[net_env.mahimahi_ptr])
            net_env.mahimahi_ptr += 1
            continue

        if duration > remaining_s:
            net_env.last_mahimahi_time = float(net_env.last_mahimahi_time + remaining_s)
            remaining_s = 0.0
            break

        remaining_s -= duration
        net_env.last_mahimahi_time = float(net_env.cooked_time[net_env.mahimahi_ptr])
        net_env.mahimahi_ptr += 1


def _get_video_chunk_record_trace(
    net_env: env.Environment, quality: int
) -> tuple[tuple[float, float, float, float, int, list[int], bool, int], list[float], list[float]]:
    """
    * `fixed_env.Environment.get_video_chunk` but also record the raw-trace segments consumed
      (bandwidth Mbps + duration seconds) during BOTH download and sleep.

    Returns:
      ((delay_ms, sleep_time_ms, buffer_s, rebuf_s, video_chunk_size, next_sizes, end_of_video, remain),
       bw_segments_mbps,
       duration_segments_s)
    """
    assert 0 <= quality < env.BITRATE_LEVELS

    video_chunk_size = int(net_env.video_size[quality][net_env.video_chunk_counter])

    delay_s = 0.0
    video_chunk_counter_sent = 0.0
    bw_segments: list[float] = []
    dur_segments: list[float] = []

    #* Download loop over Mahimahi raw trace (may stop fractionally).
    while True:
        if net_env.mahimahi_ptr >= len(net_env.cooked_bw):
            net_env.mahimahi_ptr = 1
            net_env.last_mahimahi_time = 0.0

        bw_mbps = float(net_env.cooked_bw[net_env.mahimahi_ptr])
        duration = float(net_env.cooked_time[net_env.mahimahi_ptr] - net_env.last_mahimahi_time)
        if duration <= 1e-12:
            net_env.last_mahimahi_time = float(net_env.cooked_time[net_env.mahimahi_ptr])
            net_env.mahimahi_ptr += 1
            continue

        throughput_Bps = bw_mbps * env.B_IN_MB / env.BITS_IN_BYTE
        packet_payload = throughput_Bps * duration * env.PACKET_PAYLOAD_PORTION

        if video_chunk_counter_sent + packet_payload > float(video_chunk_size) and throughput_Bps > 0.0:
            fractional_time = (float(video_chunk_size) - video_chunk_counter_sent) / throughput_Bps / env.PACKET_PAYLOAD_PORTION
            fractional_time = float(max(min(fractional_time, duration), 0.0))
            bw_segments.append(float(bw_mbps))
            dur_segments.append(float(fractional_time))
            delay_s += float(fractional_time)
            net_env.last_mahimahi_time = float(net_env.last_mahimahi_time + fractional_time)
            break

        bw_segments.append(float(bw_mbps))
        dur_segments.append(float(duration))
        video_chunk_counter_sent += float(packet_payload)
        delay_s += float(duration)
        net_env.last_mahimahi_time = float(net_env.cooked_time[net_env.mahimahi_ptr])
        net_env.mahimahi_ptr += 1

    delay_ms = float(delay_s * env.MILLISECONDS_IN_SECOND + env.LINK_RTT)

    #* Rebuffer and buffer update (ms).
    rebuf_ms = float(np.maximum(delay_ms - float(net_env.buffer_size), 0.0))
    net_env.buffer_size = float(np.maximum(float(net_env.buffer_size) - delay_ms, 0.0))
    net_env.buffer_size = float(net_env.buffer_size + env.VIDEO_CHUNCK_LEN)

    #* Sleep if buffer exceeds threshold (sleep consumes trace time).
    sleep_time_ms = 0.0
    if net_env.buffer_size > env.BUFFER_THRESH:
        drain_buffer_time = float(net_env.buffer_size - env.BUFFER_THRESH)
        sleep_time_ms = float(np.ceil(drain_buffer_time / env.DRAIN_BUFFER_SLEEP_TIME) * env.DRAIN_BUFFER_SLEEP_TIME)
        net_env.buffer_size = float(net_env.buffer_size - sleep_time_ms)

        remaining_sleep_s = float(sleep_time_ms / env.MILLISECONDS_IN_SECOND)
        while remaining_sleep_s > 1e-12:
            if net_env.mahimahi_ptr >= len(net_env.cooked_time):
                net_env.mahimahi_ptr = 1
                net_env.last_mahimahi_time = 0.0

            duration = float(net_env.cooked_time[net_env.mahimahi_ptr] - net_env.last_mahimahi_time)
            if duration <= 1e-12:
                net_env.last_mahimahi_time = float(net_env.cooked_time[net_env.mahimahi_ptr])
                net_env.mahimahi_ptr += 1
                continue

            bw_mbps = float(net_env.cooked_bw[net_env.mahimahi_ptr])
            if duration > remaining_sleep_s:
                bw_segments.append(float(bw_mbps))
                dur_segments.append(float(remaining_sleep_s))
                net_env.last_mahimahi_time = float(net_env.last_mahimahi_time + remaining_sleep_s)
                remaining_sleep_s = 0.0
                break

            bw_segments.append(float(bw_mbps))
            dur_segments.append(float(duration))
            remaining_sleep_s -= float(duration)
            net_env.last_mahimahi_time = float(net_env.cooked_time[net_env.mahimahi_ptr])
            net_env.mahimahi_ptr += 1

            if net_env.mahimahi_ptr >= len(net_env.cooked_bw):
                net_env.mahimahi_ptr = 1
                net_env.last_mahimahi_time = 0.0

    return_buffer_size_s = float(net_env.buffer_size / env.MILLISECONDS_IN_SECOND)

    net_env.video_chunk_counter += 1
    video_chunk_remain = int(env.TOTAL_VIDEO_CHUNCK - net_env.video_chunk_counter)

    end_of_video = False
    if net_env.video_chunk_counter >= env.TOTAL_VIDEO_CHUNCK:
        end_of_video = True
        net_env.buffer_size = 0.0
        net_env.video_chunk_counter = 0

        net_env.trace_idx += 1
        if net_env.trace_idx >= len(net_env.all_cooked_time):
            net_env.trace_idx = 0

        net_env.cooked_time = net_env.all_cooked_time[net_env.trace_idx]
        net_env.cooked_bw = net_env.all_cooked_bw[net_env.trace_idx]
        net_env.mahimahi_ptr = net_env.mahimahi_start_ptr
        net_env.last_mahimahi_time = net_env.cooked_time[net_env.mahimahi_ptr - 1]
        video_chunk_remain = int(env.TOTAL_VIDEO_CHUNCK - net_env.video_chunk_counter)

    next_video_chunk_sizes = [int(net_env.video_size[i][net_env.video_chunk_counter]) for i in range(env.BITRATE_LEVELS)]

    return (
        float(delay_ms),
        float(sleep_time_ms),
        float(return_buffer_size_s),
        float(rebuf_ms / env.MILLISECONDS_IN_SECOND),
        int(video_chunk_size),
        next_video_chunk_sizes,
        bool(end_of_video),
        int(video_chunk_remain),
    ), bw_segments, dur_segments


def _get_video_chunk_override(
    net_env: env.Environment,
    quality: int,
    *,
    bandwidth_trace_mbps: list[float],
    time_trace_s: list[float],
) -> tuple[float, float, float, float, int, list[int], bool, int] | None:
    """
    * Download one chunk like `fixed_env.Environment.get_video_chunk`, but override the
      bandwidth evolution during the download with an SMT-synthesized piecewise trace.

    Returns:
        Same tuple as `get_video_chunk`, or None if the synthesized trace doesn't complete the chunk.
    """
    assert 0 <= quality < env.BITRATE_LEVELS

    video_chunk_size = int(net_env.video_size[quality][net_env.video_chunk_counter])
    initial_buffer_ms = float(net_env.buffer_size)

    delay_ms, sleep_time_ms, buffer_s, rebuf_s, complete = _simulate_chunk_from_piecewise_bw(
        bandwidth_trace_mbps=bandwidth_trace_mbps,
        time_trace_s=time_trace_s,
        video_chunk_size_bytes=video_chunk_size,
        initial_buffer_ms=initial_buffer_ms,
    )
    if not complete:
        return None

    #* Consume trace time during download (RTT does not advance Mahimahi time in `fixed_env.py`).
    rtt_free_delay_s = float(max(delay_ms - env.LINK_RTT, 0.0) / env.MILLISECONDS_IN_SECOND)
    _advance_trace_time(net_env, rtt_free_delay_s)

    #* Apply buffer update results.
    net_env.buffer_size = float(buffer_s * env.MILLISECONDS_IN_SECOND)

    #* Consume trace time during sleep (this mirrors the sleep-time loop in `fixed_env.py`).
    _advance_trace_time(net_env, float(sleep_time_ms / env.MILLISECONDS_IN_SECOND))

    return_buffer_size_s = float(net_env.buffer_size / env.MILLISECONDS_IN_SECOND)

    net_env.video_chunk_counter += 1
    video_chunk_remain = env.TOTAL_VIDEO_CHUNCK - net_env.video_chunk_counter

    end_of_video = False
    if net_env.video_chunk_counter >= env.TOTAL_VIDEO_CHUNCK:
        end_of_video = True
        net_env.buffer_size = 0.0
        net_env.video_chunk_counter = 0

        net_env.trace_idx += 1
        if net_env.trace_idx >= len(net_env.all_cooked_time):
            net_env.trace_idx = 0

        net_env.cooked_time = net_env.all_cooked_time[net_env.trace_idx]
        net_env.cooked_bw = net_env.all_cooked_bw[net_env.trace_idx]

        net_env.mahimahi_ptr = net_env.mahimahi_start_ptr
        net_env.last_mahimahi_time = net_env.cooked_time[net_env.mahimahi_ptr - 1]

        video_chunk_remain = env.TOTAL_VIDEO_CHUNCK - net_env.video_chunk_counter

    next_video_chunk_sizes = [int(net_env.video_size[i][net_env.video_chunk_counter]) for i in range(env.BITRATE_LEVELS)]

    return (
        float(delay_ms),
        float(sleep_time_ms),
        float(return_buffer_size_s),
        float(rebuf_s),
        int(video_chunk_size),
        next_video_chunk_sizes,
        bool(end_of_video),
        int(video_chunk_remain),
    )


def _update_state(
    state: np.ndarray,
    *,
    bit_rate: int,
    buffer_s: float,
    video_chunk_size: int,
    delay_ms: float,
    next_video_chunk_sizes: list[int],
    video_chunk_remain: int,
) -> np.ndarray:
    #* Matches the state construction in `test.py`.
    new_state = np.roll(state, -1, axis=1).astype(np.float64, copy=False)
    new_state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))
    new_state[1, -1] = float(buffer_s) / BUFFER_NORM_FACTOR
    new_state[2, -1] = float(video_chunk_size) / float(delay_ms) / M_IN_K
    new_state[3, -1] = float(delay_ms) / M_IN_K / BUFFER_NORM_FACTOR
    new_state[4, :A_DIM] = np.array(next_video_chunk_sizes, dtype=np.float64) / M_IN_K / M_IN_K
    new_state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
    return new_state


def end_to_end_eval(args: argparse.Namespace) -> None:
    #* End-to-end impact: actually apply PGD->SMT as an environment attack, then roll out the policy on the attacked env.
    try:
        from smt import synthesize_bandwidth_trace
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "* Failed to import SMT solver. Install deps (e.g., `pip install z3-solver`) and retry."
        ) from e

    np.random.seed(args.seed)

    history_delay = bool(args.pgd_history_delay)
    history_throughput = bool(args.pgd_history_throughput)

    trace_dir = _ensure_trailing_sep(_resolve_under_src(args.trace_dir))
    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(trace_dir)
    if args.trace_num > 0:
        all_cooked_time, all_cooked_bw, all_file_names = _sample_traces(
            all_cooked_time, all_cooked_bw, all_file_names, int(args.trace_num), seed=42
        )

    net_env = env.Environment(all_cooked_time=all_cooked_time, all_cooked_bw=all_cooked_bw)

    actor = network.Network(state_dim=[S_INFO, S_LEN], action_dim=A_DIM, learning_rate=ACTOR_LR_RATE)
    actor.load_model(_resolve_under_src(args.model))

    wandb_run = None
    if args.wandb_mode != "disabled":
        tags = _parse_tags(args.wandb_tags)
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            mode=args.wandb_mode,
            tags=tags,
            config={
                "trace_dir": args.trace_dir,
                "trace_num": args.trace_num,
                "model": args.model,
                "attack_method": getattr(args, "attack_method", "pgd"),
                "policy_rules_file": getattr(args, "policy_rules_file", None),
                "sym_eps_delay_ms": getattr(args, "sym_eps_delay_ms", 1e-6),
                "sym_max_rules": getattr(args, "sym_max_rules", 0),
                "sym_samples_per_rule": getattr(args, "sym_samples_per_rule", 0),
                "sym_min_delay_ms": getattr(args, "sym_min_delay_ms", 0.0),
                "sym_truegap_max_candidates_per_action": getattr(
                    args, "sym_truegap_max_candidates_per_action", 0
                ),
                "sym_truegap_oracle_min_ratio": getattr(args, "sym_truegap_oracle_min_ratio", 0.0),
                "sym_truegap_oracle_max_drop": getattr(args, "sym_truegap_oracle_max_drop", 0.0),
                "pgd_alpha": args.pgd_alpha,
                "pgd_iters": args.pgd_iters,
                "pgd_objective_id": args.pgd_objective_id,
                "attack_enabled": args.attack,
                "enforce_coupling": args.coupling,
                "use_rules": args.use_rules,
                "rules_file": args.rules_file,
                "rules_project_every_step": args.rules_project_every_step,
                "rules_perturb_both": args.rules_perturb_both,
                "pgd_perturb_buffer": args.pgd_buffer,
                "pgd_history_delay": args.pgd_history_delay,
                "pgd_history_throughput": args.pgd_history_throughput,
                "pgd_history_sparse": args.pgd_history_sparse,
                "pgd_history_sparse_tol": args.pgd_history_sparse_tol,
                "smt_bw_samples": args.smt_bw_samples,
                "smt_trace_rules": args.smt_trace_rules,
                "smt_window_rel_tol": args.smt_window_rel_tol,
                "action_selection": args.action_selection,
                "max_steps": args.max_steps,
                "seed": args.seed,
            },
        )

    pgd_attacker = None
    if args.attack:
        pgd_attacker = PGDAttack(actor, alpha=args.pgd_alpha, num_iterations=args.pgd_iters)
    uniform = np.ones(A_DIM, dtype=np.float64) / float(A_DIM)
    buffer_norm_max = float(env.BUFFER_THRESH / env.MILLISECONDS_IN_SECOND / BUFFER_NORM_FACTOR)

    time_stamp_ms = 0.0
    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY
    state = np.zeros((S_INFO, S_LEN), dtype=np.float64)

    step = 0
    video_count = 0
    episode_return = 0.0
    rng = np.random.RandomState(int(args.seed))

    agg = {
        "env_steps": 0,
        "env_rebuf_s_values": [],
        "env_bitrate_kbps_values": [],
        "smoothness_kbps_values": [],
        "entropy_values": [],
        "max_prob_values": [],
        #* QoE component distributions (for percentiles / geo_mean).
        "qoe_reward_values": [],
        "qoe_quality_values": [],
        "qoe_rebuf_s_values": [],
        "qoe_smoothness_values": [],
        "pgd_attack_time_ms_values": [],
        "pgd_iters_run_values": [],
        "sym_attempts": 0,
        "sym_applied_sum": 0,
        "sym_rule_found_sum": 0,
        "sym_projection_success_sum": 0,
        "sym_teacher_check_pass_sum": 0,
        #* History sparsification diagnostics (only populated when enabled + history attack is used).
        "pgd_history_sparse_attempts": 0,
        "pgd_history_sparse_applied_sum": 0,
        "pgd_history_sparse_k_values": [],
        "pgd_history_sparse_k_unique_values": [],
        "pgd_history_sparse_candidates_total_values": [],
        "smt_attempts": 0,
        "smt_successes": 0,
        "smt_skipped": 0,
        "smt_applied": 0,
        "smt_complete": 0,
        "smt_solve_time_ms_values": [],
        "smt_correction_buffer_norm_values": [],
        "smt_correction_throughput_norm_values": [],
        "smt_correction_delay_norm_values": [],
        "smt_correction_l1_values": [],
        "smt_correction_l2_values": [],
        "smt_kl_pgd_realized_values": [],
        "smt_action_flip_sum": 0,
        "smt_action_match_pgd_sum": 0,
    }

    use_window_plan = bool(history_delay or history_throughput)
    # When history attacks are enabled, we optionally build a multi-chunk SMT plan and
    # apply it chunk-by-chunk (so the environment attack spans multiple chunks).
    window_plan = None  # list[dict[str, object]] | None
    window_plan_idx = 0
    window_plan_solve_time_ms: float | None = None

    while True:
        rules_totals_before = None
        if args.attack and args.use_rules and pgd_attacker is not None:
            rules_totals_before = pgd_attacker.get_rules_projection_totals()

        #* Environment bandwidth snapshot (trace Mbps) before downloading this chunk.
        try:
            trace_bw_mbps = float(net_env.cooked_bw[net_env.mahimahi_ptr])
        except Exception:
            trace_bw_mbps = float("nan")

        initial_buffer_ms = float(net_env.buffer_size)

        #* Shadow step on the real trace (no attack) to get the baseline next-state for PGD.
        clean_env = copy_env(net_env, RANDOM_SEED=args.seed)
        delay_ms_clean, sleep_time_ms_clean, buffer_s_clean, rebuf_s_clean, video_chunk_size, next_video_chunk_sizes, end_of_video_clean, video_chunk_remain = (
            clean_env.get_video_chunk(bit_rate)
        )

        clean_state_next = _update_state(
            state,
            bit_rate=bit_rate,
            buffer_s=float(buffer_s_clean),
            video_chunk_size=int(video_chunk_size),
            delay_ms=float(delay_ms_clean),
            next_video_chunk_sizes=[int(x) for x in next_video_chunk_sizes],
            video_chunk_remain=int(video_chunk_remain),
        )

        # Default: no environment attack applied (use clean transition).
        used_env = clean_env
        delay_ms = float(delay_ms_clean)
        sleep_time_ms = float(sleep_time_ms_clean)
        buffer_s = float(buffer_s_clean)
        rebuf_s = float(rebuf_s_clean)
        end_of_video = bool(end_of_video_clean)
        next_sizes = [int(x) for x in next_video_chunk_sizes]
        smt_applied = 0

        metrics: dict[str, float | int | str] = {
            "env/step": step,
            "env/episode": video_count,
            "env/trace": all_file_names[net_env.trace_idx],
            "env/buffer_before_s": float(initial_buffer_ms / 1000.0),
            "env/video_chunk_size_bytes": float(video_chunk_size),
            "env/trace_bw_mbps": float(trace_bw_mbps),
            "env/clean_bw_mbps": float(_avg_bandwidth_mbps(float(video_chunk_size), float(delay_ms_clean))),
            "pgd/history_delay": int(history_delay),
            "pgd/history_throughput": int(history_throughput),
            "pgd/history_applied": 0,
            "smt/window": int(use_window_plan),
        }

        attack_method = str(getattr(args, "attack_method", "pgd"))
        if args.attack:
            if pgd_attacker is None:
                raise RuntimeError("* PGD attacker was not initialized, but --attack is enabled")

            #* Attack chooses a target state in feature space (starting from the clean next-state).
            pgd_t0 = time.perf_counter()
            sym_metrics: dict[str, float | int] | None = None
            pre_bw: list[float] | None = None
            pre_tt: list[float] | None = None
            if attack_method == "dt":
                pgd_state, sym_metrics = _symbolic_dt_attack_state(
                    args,
                    actor=actor,
                    pgd_attacker=pgd_attacker,
                    clean_state_next=clean_state_next,
                    oracle_env_snapshot=clean_env,
                    last_bitrate_idx=int(bit_rate),
                    video_chunk_size_bytes=int(video_chunk_size),
                    allow_delay_idx=[int(clean_state_next.shape[1]) - 1],
                    rng=rng,
                )
            elif attack_method == "dt-truegap":
                pgd_state, sym_metrics, pre_bw, pre_tt = _symbolic_dt_truegap_attack_state(
                    args,
                    actor=actor,
                    pgd_attacker=pgd_attacker,
                    clean_state_next=clean_state_next,
                    clean_env_snapshot=clean_env,
                    env_before_chunk=net_env,
                    state_before_chunk=state,
                    last_bitrate_idx=int(bit_rate),
                    video_chunk_size_bytes=int(video_chunk_size),
                    allow_delay_idx=[int(clean_state_next.shape[1]) - 1],
                    rng=rng,
                )
            else:
                pgd_state = pgd_attacker.pgd_attack(
                    clean_state_next,
                    target_bitrate=bit_rate,
                    BUFFER_NORM_FACTOR=buffer_norm_max,
                    M_IN_K=M_IN_K,
                    video_chunk_size=int(video_chunk_size),
                    enforce_coupling=bool(args.coupling),
                    perturb_buffer=bool(args.pgd_buffer),
                    use_rules=bool(args.use_rules),
                    rules_file=str(args.rules_file),
                    rules_project_every_step=bool(args.rules_project_every_step),
                    rules_perturb_both=bool(args.rules_perturb_both),
                    perturb_delay_history=history_delay,
                    perturb_throughput_history=history_throughput,
                    history_sparse=bool(args.pgd_history_sparse),
                    history_sparse_tol=float(args.pgd_history_sparse_tol),
                    oracle_env_snapshot=clean_env,
                    oracle_k_steps=K_steps,
                    objective_id=int(args.pgd_objective_id),
                )
            pgd_attack_time_ms = (time.perf_counter() - pgd_t0) * 1000.0
            p_clean = actor.predict(np.reshape(clean_state_next, (1, S_INFO, S_LEN))).astype(np.float64)
            p_pgd = actor.predict(np.reshape(pgd_state, (1, S_INFO, S_LEN))).astype(np.float64)

            pgd_action = int(np.argmax(p_pgd))
            clean_action = int(np.argmax(p_clean))
            kl_uniform_pgd = float(_kl_div(uniform, p_pgd))
            implied_chunk = _implied_chunk_size_bytes(
                float(pgd_state[2, -1]),
                float(pgd_state[3, -1]),
                m_in_k=M_IN_K,
                delay_norm_factor=BUFFER_NORM_FACTOR,
            )
            chunk_rel_err = abs(implied_chunk - float(video_chunk_size)) / (float(video_chunk_size) + 1e-9)

            metrics.update(
                {
                    "pgd/attack_time_ms": float(pgd_attack_time_ms),
                    "pgd/buffer_norm": float(pgd_state[1, -1]),
                    "pgd/throughput_norm": float(pgd_state[2, -1]),
                    "pgd/delay_norm": float(pgd_state[3, -1]),
                    "pgd/kl_clean_pgd": float(_kl_div(p_clean, p_pgd)),
                    "pgd/kl_uniform_pgd": kl_uniform_pgd,
                    "pgd/action_flip": int(pgd_action != clean_action),
                    "pgd/chunk_size_rel_err": float(chunk_rel_err),
                }
            )
            if attack_method not in ("dt", "dt-truegap"):
                early_stop = pgd_attacker.get_early_stop_last_attack()
                iters_run = int(early_stop.get("pgd/early_stop_iters_run", args.pgd_iters))
                metrics["pgd/iters_run"] = int(iters_run)
                agg["pgd_iters_run_values"].append(float(iters_run))
            if sym_metrics is not None:
                metrics.update(sym_metrics)
                agg["sym_attempts"] += 1
                agg["sym_applied_sum"] += int(sym_metrics.get("sym/applied", 0))
                agg["sym_rule_found_sum"] += int(sym_metrics.get("sym/rule_found", 0))
                agg["sym_projection_success_sum"] += int(sym_metrics.get("sym/projection_success", 0))
                agg["sym_teacher_check_pass_sum"] += int(sym_metrics.get("sym/teacher_check_pass", 0))
            agg["pgd_attack_time_ms_values"].append(float(pgd_attack_time_ms))

            # ^ SMT: synthesize a bandwidth trace to realize the PGD state.
            # - Default (no history attack): single-chunk SMT, as before.
            # - With history attack: build a multi-chunk window plan once, then apply it chunk-by-chunk.
            solve_time_ms = 0.0
            bw: list[float] | None = pre_bw
            tt: list[float] | None = pre_tt
            if sym_metrics is not None and bw is not None and tt is not None:
                solve_time_ms = float(sym_metrics.get("symtrue/smt_solve_time_ms_selected", 0.0))
            if use_window_plan:
                #* Always drive history attacks via a multi-chunk SMT plan (ignore per-step dt-truegap traces).
                bw = None
                tt = None

            metrics["smt/window"] = int(use_window_plan)

            skip_smt = bool(
                attack_method == "dt-truegap"
                and sym_metrics is not None
                and int(sym_metrics.get("sym/applied", 0)) == 0
            )
            if (bw is None or tt is None) and not bool(skip_smt):
                if use_window_plan:
                    #* Validate the current plan entry against the current action/chunk size.
                    if window_plan is not None and window_plan_idx < len(window_plan):
                        expected = window_plan[window_plan_idx]
                        expected_action = int(expected["action"])
                        expected_chunk_bytes = int(expected["chunk_size_bytes"])
                        actual_chunk_bytes = int(net_env.video_size[int(bit_rate)][int(net_env.video_chunk_counter)])
                        if expected_action != int(bit_rate) or expected_chunk_bytes != actual_chunk_bytes:
                            window_plan = None
                            window_plan_idx = 0

                    #* Build a fresh plan if needed (starting at the current chunk).
                    if window_plan is None:
                        plan_actions: list[int] = [int(bit_rate)]
                        plan_chunk_sizes: list[int] = [int(video_chunk_size)]

                        target_delay_norms: list[float] = [float(pgd_state[3, -1])]
                        target_throughput_norms_plan: list[float] = [float(pgd_state[2, -1])]
                        last_pgd_state = np.array(pgd_state, copy=True)

                        #* Predict the remaining actions in this plan using the PGD-attacked states
                        # (more likely to match the agent's actual actions under attack).
                        pred_state_for_action = np.array(pgd_state, copy=True)
                        pred_state_clean = np.array(clean_state_next, copy=True)
                        pred_env = copy_env(clean_env, RANDOM_SEED=args.seed)
                        for _ in range(1, S_LEN):
                            p_pred = actor.predict(np.reshape(pred_state_for_action, (1, S_INFO, S_LEN))).astype(np.float64).reshape(-1)
                            if args.action_selection == "argmax":
                                pred_action = int(np.argmax(p_pred))
                            elif args.action_selection == "gumbel":
                                noise = np.random.gumbel(size=p_pred.shape)
                                pred_action = int(np.argmax(np.log(p_pred) + noise))
                            else:
                                raise ValueError(f"Unknown action_selection: {args.action_selection}")

                            env_before_chunk_p = copy_env(pred_env, RANDOM_SEED=args.seed)
                            state_before_chunk_p = np.array(pred_state_clean, copy=True)

                            delay_ms_p, sleep_ms_p, buffer_s_p, _, chunk_size_p, next_sizes_p, end_p, remain_p = (
                                pred_env.get_video_chunk(pred_action)
                            )
                            clean_state_next_p = _update_state(
                                pred_state_clean,
                                bit_rate=int(pred_action),
                                buffer_s=float(buffer_s_p),
                                video_chunk_size=int(chunk_size_p),
                                delay_ms=float(delay_ms_p),
                                next_video_chunk_sizes=[int(x) for x in next_sizes_p],
                                video_chunk_remain=int(remain_p),
                            )

                            oracle_snapshot = copy_env(pred_env, RANDOM_SEED=args.seed)
                            sym_p_metrics: dict[str, float | int] | None = None
                            if attack_method == "dt":
                                chunk_idx = int(len(plan_chunk_sizes))
                                allow_from = max(0, int(S_LEN) - 1 - int(chunk_idx))
                                allow_idx = list(range(int(allow_from), int(S_LEN)))
                                pgd_state_p, sym_p_metrics = _symbolic_dt_attack_state(
                                    args,
                                    actor=actor,
                                    pgd_attacker=pgd_attacker,
                                    clean_state_next=clean_state_next_p,
                                    oracle_env_snapshot=oracle_snapshot,
                                    last_bitrate_idx=int(pred_action),
                                    video_chunk_size_bytes=int(chunk_size_p),
                                    allow_delay_idx=[int(i) for i in allow_idx],
                                    rng=rng,
                                )
                            elif attack_method == "dt-truegap":
                                chunk_idx = int(len(plan_chunk_sizes))
                                allow_from = max(0, int(S_LEN) - 1 - int(chunk_idx))
                                allow_idx = list(range(int(allow_from), int(S_LEN)))
                                pgd_state_p, sym_p_metrics, _bw_p, _tt_p = _symbolic_dt_truegap_attack_state(
                                    args,
                                    actor=actor,
                                    pgd_attacker=pgd_attacker,
                                    clean_state_next=clean_state_next_p,
                                    clean_env_snapshot=oracle_snapshot,
                                    env_before_chunk=env_before_chunk_p,
                                    state_before_chunk=state_before_chunk_p,
                                    last_bitrate_idx=int(pred_action),
                                    video_chunk_size_bytes=int(chunk_size_p),
                                    allow_delay_idx=[int(i) for i in allow_idx],
                                    rng=rng,
                                )
                            else:
                                #* Per-step PGD target for the window: use the latest-column (delay, throughput).
                                pgd_state_p = pgd_attacker.pgd_attack(
                                    clean_state_next_p,
                                    target_bitrate=int(pred_action),
                                    BUFFER_NORM_FACTOR=buffer_norm_max,
                                    M_IN_K=M_IN_K,
                                    video_chunk_size=int(chunk_size_p),
                                    enforce_coupling=bool(args.coupling),
                                    perturb_buffer=bool(args.pgd_buffer),
                                    use_rules=bool(args.use_rules),
                                    rules_file=str(args.rules_file),
                                    rules_project_every_step=bool(args.rules_project_every_step),
                                    rules_perturb_both=bool(args.rules_perturb_both),
                                    perturb_delay_history=history_delay,
                                    perturb_throughput_history=history_throughput,
                                    history_sparse=bool(args.pgd_history_sparse),
                                    history_sparse_tol=float(args.pgd_history_sparse_tol),
                                    oracle_env_snapshot=oracle_snapshot,
                                    oracle_k_steps=K_steps,
                                    objective_id=int(args.pgd_objective_id),
                                )

                            plan_actions.append(int(pred_action))
                            plan_chunk_sizes.append(int(chunk_size_p))

                            if sym_p_metrics is not None and int(sym_p_metrics.get("sym/applied", 0)) != 0:
                                #^ History-delay DT attack can modify multiple delay coordinates; project them onto the
                                #  per-chunk plan targets (chunk j maps to history index 7-(i-j)).
                                i = int(len(plan_chunk_sizes) - 1)
                                for j in range(int(i)):
                                    k = int(S_LEN) - 1 - (int(i) - int(j))
                                    if 0 <= k < int(S_LEN):
                                        target_delay_norms[int(j)] = float(pgd_state_p[3, int(k)])
                                        target_throughput_norms_plan[int(j)] = float(pgd_state_p[2, int(k)])

                            target_delay_norms.append(float(pgd_state_p[3, -1]))
                            target_throughput_norms_plan.append(float(pgd_state_p[2, -1]))

                            last_pgd_state = np.array(pgd_state_p, copy=True)
                            pred_state_for_action = np.array(pgd_state_p, copy=True)
                            pred_state_clean = np.array(clean_state_next_p, copy=True)

                            if bool(end_p):
                                break

                        #* Sequential per-chunk SMT solves: each chunk gets its own `--bw-samples` segments.
                        smt_t0 = time.perf_counter()
                        buffer_ms_plan = float(initial_buffer_ms)
                        plan_ok = True
                        plan_entries: list[dict[str, object]] = []

                        for j in range(len(plan_chunk_sizes)):
                            action_j = int(plan_actions[j])
                            chunk_bytes_j = int(plan_chunk_sizes[j])
                            tgt_delay_norm_j = float(target_delay_norms[j])
                            tgt_thr_norm_j = float(target_throughput_norms_plan[j])
                            enforce_buf_j = bool(args.pgd_buffer) and (j == len(plan_chunk_sizes) - 1)

                            bw_j, tt_j = synthesize_bandwidth_trace(
                                video_chunk_size=int(chunk_bytes_j),
                                initial_buffer_ms=float(buffer_ms_plan),
                                target_buffer_norm=float(last_pgd_state[1, -1]),
                                target_throughput_norm=float(tgt_thr_norm_j),
                                target_delay_norm=float(tgt_delay_norm_j),
                                num_bw_samples=int(args.smt_bw_samples),
                                enforce_buffer_target=bool(enforce_buf_j),
                                enforce_throughput_target=bool(history_throughput),
                                enforce_trace_rules=bool(args.smt_trace_rules),
                                rel_tol=float(args.smt_window_rel_tol),
                            )
                            if bw_j is None or tt_j is None:
                                plan_ok = False
                                break

                            delay_ms_j, sleep_ms_j, buffer_s_j, _, complete_j = _simulate_chunk_from_piecewise_bw(
                                bandwidth_trace_mbps=[float(x) for x in bw_j],
                                time_trace_s=[float(x) for x in tt_j],
                                video_chunk_size_bytes=int(chunk_bytes_j),
                                initial_buffer_ms=float(buffer_ms_plan),
                            )
                            if not bool(complete_j):
                                plan_ok = False
                                break

                            buffer_ms_plan = float(buffer_s_j * env.MILLISECONDS_IN_SECOND)
                            plan_entries.append(
                                {
                                    "action": int(action_j),
                                    "chunk_size_bytes": int(chunk_bytes_j),
                                    "bw": [float(x) for x in bw_j],
                                    "t": [float(x) for x in tt_j],
                                }
                            )

                        solve_time_ms = (time.perf_counter() - smt_t0) * 1000.0
                        window_plan_solve_time_ms = float(solve_time_ms)
                        if plan_ok and len(plan_entries) == len(plan_chunk_sizes):
                            window_plan = plan_entries
                            window_plan_idx = 0

                    if window_plan is not None and window_plan_idx < len(window_plan):
                        cur = window_plan[window_plan_idx]
                        bw = [float(x) for x in cur["bw"]]
                        tt = [float(x) for x in cur["t"]]
                        metrics["smt/plan_len"] = int(len(window_plan))
                        metrics["smt/plan_idx"] = int(window_plan_idx)
                        metrics["smt/plan_solve_time_ms"] = float(window_plan_solve_time_ms or 0.0)
                else:
                    smt_t0 = time.perf_counter()
                    bw_trace, t_trace = synthesize_bandwidth_trace(
                        video_chunk_size=int(video_chunk_size),
                        initial_buffer_ms=float(initial_buffer_ms),
                        target_buffer_norm=float(pgd_state[1, -1]),
                        target_throughput_norm=float(pgd_state[2, -1]),
                        target_delay_norm=float(pgd_state[3, -1]),
                        num_bw_samples=int(args.smt_bw_samples),
                        enforce_buffer_target=bool(args.pgd_buffer),
                        enforce_trace_rules=bool(args.smt_trace_rules),
                        rel_tol=float(args.smt_window_rel_tol),
                    )
                    solve_time_ms = (time.perf_counter() - smt_t0) * 1000.0
                    if bw_trace is not None and t_trace is not None:
                        bw = [float(x) for x in bw_trace]
                        tt = [float(x) for x in t_trace]

            smt_success = bw is not None and tt is not None
            smt_attempted = int(not bool(skip_smt))
            agg["smt_skipped"] += int(bool(skip_smt))
            if smt_attempted:
                agg["smt_attempts"] += 1
                agg["smt_successes"] += int(smt_success)
                agg["smt_solve_time_ms_values"].append(float(solve_time_ms))

            metrics["smt/skipped"] = int(bool(skip_smt))
            metrics["smt/attempted"] = int(smt_attempted)
            metrics["smt/success"] = int(smt_success) if smt_attempted else 0
            metrics["smt/solve_time_ms"] = float(solve_time_ms) if smt_attempted else 0.0

            if smt_success:
                metrics["smt/trace_total_time_s"] = float(np.sum(tt))
                metrics["smt/trace_avg_bw_mbps"] = float(np.mean(bw))
                metrics["smt/trace_min_bw_mbps"] = float(np.min(bw))
                metrics["smt/trace_max_bw_mbps"] = float(np.max(bw))

                overridden = _get_video_chunk_override(
                    net_env,
                    bit_rate,
                    bandwidth_trace_mbps=bw,
                    time_trace_s=tt,
                )
                if overridden is not None:
                    (
                        delay_ms,
                        sleep_time_ms,
                        buffer_s,
                        rebuf_s,
                        video_chunk_size,
                        next_sizes,
                        end_of_video,
                        video_chunk_remain,
                    ) = overridden
                    used_env = net_env
                    smt_applied = 1
                    agg["smt_applied"] += 1
                    if use_window_plan and window_plan is not None:
                        window_plan_idx += 1
                        if window_plan_idx >= len(window_plan):
                            window_plan = None
                            window_plan_idx = 0

                    #* Additional "complete" is always true if `overridden` is not None.
                    metrics["smt/complete"] = 1
                    agg["smt_complete"] += 1

                    realized_state_next = _update_state(
                        state,
                        bit_rate=bit_rate,
                        buffer_s=float(buffer_s),
                        video_chunk_size=int(video_chunk_size),
                        delay_ms=float(delay_ms),
                        next_video_chunk_sizes=[int(x) for x in next_sizes],
                        video_chunk_remain=int(video_chunk_remain),
                    )
                    observed_state_next = realized_state_next
                    metrics["pgd/history_applied"] = 0

                    p_realized = actor.predict(np.reshape(observed_state_next, (1, S_INFO, S_LEN))).astype(np.float64)
                    realized_action = int(np.argmax(p_realized))

                    correction_buf = float(realized_state_next[1, -1] - pgd_state[1, -1])
                    correction_thr = float(realized_state_next[2, -1] - pgd_state[2, -1])
                    correction_dly = float(realized_state_next[3, -1] - pgd_state[3, -1])
                    correction_l1 = float(abs(correction_buf) + abs(correction_thr) + abs(correction_dly))
                    correction_l2 = float(
                        np.sqrt(correction_buf * correction_buf + correction_thr * correction_thr + correction_dly * correction_dly)
                    )
                    kl_pgd_realized = float(_kl_div(p_pgd, p_realized))

                    agg["smt_action_flip_sum"] += int(realized_action != clean_action)
                    agg["smt_action_match_pgd_sum"] += int(realized_action == pgd_action)
                    metrics.update(
                        {
                            "smt/applied": 1,
                            "smt/delay_ms": float(delay_ms),
                            "smt/sleep_time_ms": float(sleep_time_ms),
                            "smt/buffer_s": float(buffer_s),
                            "smt/rebuf_s": float(rebuf_s),
                            "smt/buffer_norm": float(realized_state_next[1, -1]),
                            "smt/throughput_norm": float(realized_state_next[2, -1]),
                            "smt/delay_norm": float(realized_state_next[3, -1]),
                            "smt/correction_buffer_norm": correction_buf,
                            "smt/correction_throughput_norm": correction_thr,
                            "smt/correction_delay_norm": correction_dly,
                            "smt/correction_l1": correction_l1,
                            "smt/correction_l2": correction_l2,
                            "smt/kl_pgd_realized": kl_pgd_realized,
                            "smt/action_flip": int(realized_action != clean_action),
                            "smt/action_match_pgd": int(realized_action == pgd_action),
                        }
                    )
                    agg["smt_correction_buffer_norm_values"].append(float(correction_buf))
                    agg["smt_correction_throughput_norm_values"].append(float(correction_thr))
                    agg["smt_correction_delay_norm_values"].append(float(correction_dly))
                    agg["smt_correction_l1_values"].append(float(correction_l1))
                    agg["smt_correction_l2_values"].append(float(correction_l2))
                    agg["smt_kl_pgd_realized_values"].append(float(kl_pgd_realized))

                    #* The agent will see the (optionally history-perturbed) realized state.
                    state = observed_state_next
                else:
                    metrics["smt/applied"] = 0
                    metrics["smt/complete"] = 0
                    if use_window_plan:
                        window_plan = None
                        window_plan_idx = 0
                    used_env = clean_env
                    state = clean_state_next
            else:
                metrics["smt/applied"] = 0
                metrics["smt/complete"] = 0
                if use_window_plan:
                    window_plan = None
                    window_plan_idx = 0
                used_env = clean_env
                state = clean_state_next
        else:
            # No attack: just use the clean transition.
            state = clean_state_next

        #* Commit chosen transition env.
        net_env = used_env
        time_stamp_ms += float(delay_ms + sleep_time_ms)

        metrics.update(
            {
                "env/realized_bw_mbps": float(_avg_bandwidth_mbps(float(video_chunk_size), float(delay_ms))),
                "env/throughput_mbps": float(_throughput_mbps(float(video_chunk_size), float(delay_ms))),
            }
        )

        #* QoE reward for the chunk that was just downloaded at `bit_rate`.
        qoe_quality = float(VIDEO_BIT_RATE[bit_rate]) / float(M_IN_K)
        qoe_rebuf_s = float(rebuf_s)
        reward = float(calculate_reward(bit_rate, last_bit_rate, rebuf_s))
        smoothness_kbps = float(abs(VIDEO_BIT_RATE[bit_rate] - VIDEO_BIT_RATE[last_bit_rate]))
        qoe_smoothness = float(smoothness_kbps) / float(M_IN_K)

        episode_return += reward
        agg["env_steps"] += 1
        agg["qoe_reward_values"].append(float(reward))
        agg["qoe_quality_values"].append(float(qoe_quality))
        agg["qoe_rebuf_s_values"].append(float(qoe_rebuf_s))
        agg["qoe_smoothness_values"].append(float(qoe_smoothness))
        agg["env_rebuf_s_values"].append(float(rebuf_s))
        agg["env_bitrate_kbps_values"].append(float(VIDEO_BIT_RATE[bit_rate]))
        agg["smoothness_kbps_values"].append(float(smoothness_kbps))

        #* Policy metrics on the state the agent actually observes (after attack if applied).
        p_obs = actor.predict(np.reshape(state, (1, S_INFO, S_LEN))).astype(np.float64)
        entropy = _entropy(p_obs)
        max_prob = float(np.max(p_obs))

        metrics.update(
            {
                "env/time_s": float(time_stamp_ms / 1000.0),
                "env/bitrate_kbps": float(VIDEO_BIT_RATE[bit_rate]),
                "env/buffer_s": float(buffer_s),
                "env/rebuf_s": float(rebuf_s),
                "env/delay_ms": float(delay_ms),
                "qoe_reward": reward,
                "qoe/quality": float(qoe_quality),
                "qoe/rebuf_s": float(qoe_rebuf_s),
                "qoe/smoothness": float(qoe_smoothness),
                "smoothness_kbps": smoothness_kbps,
                "entropy": float(entropy),
                "max_prob": float(max_prob),
            }
        )
        agg["entropy_values"].append(float(entropy))
        agg["max_prob_values"].append(float(max_prob))

        if args.attack and pgd_attacker is not None:
            sparse = pgd_attacker.get_history_sparse_last_attack()
            metrics.update(sparse)
            agg["pgd_history_sparse_attempts"] += 1
            agg["pgd_history_sparse_applied_sum"] += int(sparse.get("pgd/history_sparse_applied", 0))
            if int(sparse.get("pgd/history_sparse_applied", 0)) != 0:
                agg["pgd_history_sparse_k_values"].append(float(sparse.get("pgd/history_sparse_k", 0)))
                agg["pgd_history_sparse_k_unique_values"].append(float(sparse.get("pgd/history_sparse_k_unique", 0)))
                agg["pgd_history_sparse_candidates_total_values"].append(
                    float(sparse.get("pgd/history_sparse_candidates_total", 0))
                )
            if args.use_rules:
                metrics.update(_rules_fully_relaxed_metrics(pgd_attacker, totals_before=rules_totals_before))
                metrics.update(_rules_projection_metrics(pgd_attacker))

        if wandb_run is not None:
            wandb.log(metrics, step=step)

        # Choose next action based on the observed state.
        if args.action_selection == "argmax":
            next_bit_rate = int(np.argmax(p_obs))
        elif args.action_selection == "gumbel":
            noise = np.random.gumbel(size=len(p_obs))
            next_bit_rate = int(np.argmax(np.log(p_obs) + noise))
        else:
            raise ValueError(f"Unknown action_selection: {args.action_selection}")

        last_bit_rate = bit_rate
        bit_rate = next_bit_rate
        step += 1

        if end_of_video or (args.max_steps is not None and step >= args.max_steps):
            if end_of_video:
                if wandb_run is not None:
                    wandb.log({"episode/cumulative_reward": float(episode_return), "env/episode": video_count}, step=step)
                episode_return = 0.0
                state[:] = 0.0
                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY
                if use_window_plan:
                    window_plan = None
                    window_plan_idx = 0
                    window_plan_solve_time_ms = None
                video_count += 1

            if (args.max_steps is not None and step >= args.max_steps) or video_count >= len(all_file_names):
                break

    if bool(args.pgd_history_sparse):
        ks = [int(x) for x in agg.get("pgd_history_sparse_k_values", [])]
        print(f"* pgd/history_sparse_k values (n={len(ks)}): {ks}", flush=True)

    if wandb_run is not None:
        total = max(agg["env_steps"], 1)
        wandb_run.summary["env/steps-sum"] = total

        _write_agg_stats(wandb_run.summary, "qoe_reward", agg["qoe_reward_values"], include_geo_mean=False)
        _write_agg_stats(wandb_run.summary, "qoe/quality", agg["qoe_quality_values"], include_geo_mean=True)
        _write_agg_stats(wandb_run.summary, "qoe/rebuf_s", agg["qoe_rebuf_s_values"], include_geo_mean=True)
        _write_agg_stats(wandb_run.summary, "qoe/smoothness", agg["qoe_smoothness_values"], include_geo_mean=True)
        _write_agg_stats(wandb_run.summary, "env/rebuf_s", agg["env_rebuf_s_values"], include_geo_mean=True)
        _write_agg_stats(wandb_run.summary, "env/bitrate_kbps", agg["env_bitrate_kbps_values"], include_geo_mean=True)
        _write_agg_stats(wandb_run.summary, "smoothness_kbps", agg["smoothness_kbps_values"], include_geo_mean=True)
        _write_agg_stats(wandb_run.summary, "entropy", agg["entropy_values"], include_geo_mean=True)
        _write_agg_stats(wandb_run.summary, "max_prob", agg["max_prob_values"], include_geo_mean=True)

        #* Always log SMT rates. When SMT is not invoked (e.g., --no-attack), treat success/complete
        #* as vacuously true for comparability across runs.
        smt_attempts = int(agg.get("smt_attempts", 0))
        smt_successes = int(agg.get("smt_successes", 0))
        smt_applied = int(agg.get("smt_applied", 0))
        smt_complete = int(agg.get("smt_complete", 0))
        denom = float(max(smt_attempts, 1))
        if smt_attempts > 0:
            wandb_run.summary["smt/success-rate"] = float(smt_successes) / denom
            wandb_run.summary["smt/applied-rate"] = float(smt_applied) / denom
            wandb_run.summary["smt/complete-rate"] = float(smt_complete) / denom
        else:
            #* No SMT attempts happened in this run.
            wandb_run.summary["smt/success-rate"] = 1.0
            wandb_run.summary["smt/applied-rate"] = 0.0
            wandb_run.summary["smt/complete-rate"] = 1.0
        wandb_run.summary["smt/skipped-rate"] = float(agg.get("smt_skipped", 0)) / float(total)

        if args.attack:
            attempts = max(agg["smt_attempts"], 1)
            applied = agg["smt_applied"]
            completes = agg["smt_complete"]
            _write_agg_stats(wandb_run.summary, "pgd/attack_time_ms", agg["pgd_attack_time_ms_values"], include_geo_mean=True)
            wandb_run.summary["pgd/attack_time_ms-sum"] = float(np.sum(agg["pgd_attack_time_ms_values"]))
            if str(getattr(args, "attack_method", "pgd")) not in ("dt", "dt-truegap"):
                _write_agg_stats(wandb_run.summary, "pgd/iters_run", agg["pgd_iters_run_values"], include_geo_mean=False)
            _write_agg_stats(wandb_run.summary, "smt/solve_time_ms", agg["smt_solve_time_ms_values"], include_geo_mean=True)
            if completes > 0:
                _write_agg_stats(
                    wandb_run.summary,
                    "smt/correction_buffer_norm",
                    agg["smt_correction_buffer_norm_values"],
                    include_geo_mean=False,
                )
                _write_agg_stats(
                    wandb_run.summary,
                    "smt/correction_throughput_norm",
                    agg["smt_correction_throughput_norm_values"],
                    include_geo_mean=False,
                )
                _write_agg_stats(
                    wandb_run.summary,
                    "smt/correction_delay_norm",
                    agg["smt_correction_delay_norm_values"],
                    include_geo_mean=False,
                )
                _write_agg_stats(wandb_run.summary, "smt/correction_l1", agg["smt_correction_l1_values"], include_geo_mean=True)
                _write_agg_stats(wandb_run.summary, "smt/correction_l2", agg["smt_correction_l2_values"], include_geo_mean=True)
                _write_agg_stats(
                    wandb_run.summary, "smt/kl_pgd_realized", agg["smt_kl_pgd_realized_values"], include_geo_mean=True
                )
                wandb_run.summary["smt/action_flip-rate"] = float(agg["smt_action_flip_sum"]) / float(completes)
                wandb_run.summary["smt/action_match_pgd-rate"] = float(agg["smt_action_match_pgd_sum"]) / float(completes)

        sparse_attempts = int(agg.get("pgd_history_sparse_attempts", 0))
        sparse_applied = int(agg.get("pgd_history_sparse_applied_sum", 0))
        wandb_run.summary["pgd/history_sparse_applied-rate"] = float(sparse_applied) / float(max(sparse_attempts, 1))
        _write_agg_stats_or_zero(
            wandb_run.summary, "pgd/history_sparse_k", agg["pgd_history_sparse_k_values"], include_geo_mean=False
        )
        _write_agg_stats_or_zero(
            wandb_run.summary,
            "pgd/history_sparse_k_unique",
            agg["pgd_history_sparse_k_unique_values"],
            include_geo_mean=False,
        )
        _write_agg_stats_or_zero(
            wandb_run.summary,
            "pgd/history_sparse_candidates_total",
            agg["pgd_history_sparse_candidates_total_values"],
            include_geo_mean=False,
        )

        #* Always include these in summary (0 when rules are not used or attack is disabled).
        total_calls = 0
        total_fully_relaxed = 0
        if pgd_attacker is not None:
            total_calls, total_fully_relaxed = pgd_attacker.get_rules_projection_totals()
        wandb_run.summary["pgd/rules_projection_calls_total"] = int(total_calls)
        wandb_run.summary["pgd/rules_fully_relaxed_calls_total"] = int(total_fully_relaxed)
        wandb_run.summary["pgd/rules_fully_relaxed_rate_total"] = float(total_fully_relaxed) / float(max(total_calls, 1))
        if pgd_attacker is not None:
            wandb_run.summary.update(_rules_projection_total_metrics(pgd_attacker))

        wandb.finish()


def _build_trace_out_dir(args: argparse.Namespace) -> str:
    tags = _parse_tags(args.wandb_tags) or []
    project = _safe_path_component(str(args.wandb_project))
    tags_part = _safe_path_component("-".join(tags)) if tags else "notags"
    trace_run_name = getattr(args, "trace_run_name", None)
    if trace_run_name is not None:
        name_part = _safe_path_component(str(trace_run_name))
    else:
        name_part = _safe_path_component(args.wandb_name or f"run_{int(time.time())}")
    trace_model_suffix = getattr(args, "trace_model_suffix", None)
    if trace_model_suffix is not None and str(trace_model_suffix).strip():
        suffix_part = _safe_path_component(str(trace_model_suffix))
    else:
        suffix_part = None
    base_dir = _resolve_repo_root(getattr(args, "trace_out_dir", "traces"))
    if suffix_part is not None:
        return os.path.join(str(base_dir), project, tags_part, name_part, suffix_part)
    return os.path.join(str(base_dir), project, tags_part, name_part)


def _model_suffix_from_path(model_path: str) -> str:
    stem = os.path.splitext(os.path.basename(str(model_path)))[0]
    if "_robust_" in stem:
        return str(stem.split("_robust_", 1)[1])
    return str(stem)


def _list_models_in_dir(model_dir: str) -> list[str]:
    #* Only include known model weight extensions.
    exts = {".pt", ".pth"}
    out: list[str] = []
    for name in sorted(os.listdir(str(model_dir))):
        if str(name).startswith("."):
            continue
        path = os.path.join(str(model_dir), str(name))
        if not os.path.isfile(path):
            continue
        _, ext = os.path.splitext(str(name))
        if str(ext).lower() not in exts:
            continue
        out.append(str(path))
    return out


def _generate_adversarial_traces(
    args: argparse.Namespace,
    *,
    actor: network.Network,
    pgd_attacker: PGDAttack | None,
    all_cooked_time: list,
    all_cooked_bw: list,
    all_file_names: list[str],
) -> tuple[str, list, list, list[str], dict[str, float | int]]:
    """
    * Phase 1 (trace synthesis): run the PGD->SMT environment attack to generate raw traces,
      but do NOT use the generated traces for evaluation yet.

    Returns:
      (out_dir, adv_all_cooked_time, adv_all_cooked_bw, adv_file_names, tracegen_stats)
    """
    out_dir = _build_trace_out_dir(args)
    os.makedirs(str(out_dir), exist_ok=True)

    manifest: dict[str, object] = {
        "clean_trace_dir": str(args.trace_dir),
        "seed": int(args.seed),
        "mode": "end2end-two-pass",
        "attack_enabled": bool(args.attack),
        "generated_dir": str(out_dir),
        "traces": [],
    }

    tracegen_stats: dict[str, float | int] = {
        "smt/attempts": 0,
        "smt/successes": 0,
        "smt/applied": 0,
        "smt/complete": 0,
        "smt/skipped": 0,
        #* dt-truegap diagnostics (aggregated; present only when that method is used).
        "symtrue/evals_attempted_sum": 0,
        "symtrue/evals_smt_success_sum": 0,
        "symtrue/evals_override_success_sum": 0,
        "symtrue/evals_oracle_valid_sum": 0,
        "symtrue/evals_policy_valid_sum": 0,
        "symtrue/evals_oracle_filter_pass_sum": 0,
        "symtrue/gap_true_sum": 0.0,
        "symtrue/oracle_best_reward_mean_sum": 0.0,
        "symtrue/policy_reward_mean_sum": 0.0,
        "symtrue/smt_solve_time_ms_selected_sum": 0.0,
        #* PGD-side rule projection diagnostics (always recorded; will stay 0 if rules aren't used).
        "pgd/rules_projection_calls_total": 0,
        "pgd/rules_fully_relaxed_calls_total": 0,
        "pgd/rules_fully_relaxed_rate_total": 0.0,
        "pgd/rules_projection_success_sum": 0,
        "pgd/rules_projection_fail_sum": 0,
        "pgd/rules_projection_attempts_sum": 0,
        "pgd/rules_projection_fallbacks_sum": 0,
        "pgd/rules_projection_success-rate": 0.0,
    }

    #* PGD history sparsification diagnostics (aggregated over trace generation steps).
    sparse_attempts = 0
    sparse_applied_sum = 0
    sparse_k_values: list[float] = []
    sparse_k_unique_values: list[float] = []
    sparse_candidates_total_values: list[float] = []

    #* If attack is disabled, just write out the sampled clean traces in the SAME order.
    if not bool(args.attack):
        adv_all_cooked_time: list = []
        adv_all_cooked_bw: list = []
        adv_file_names: list[str] = []
        for i, (t, bw, name) in enumerate(zip(all_cooked_time, all_cooked_bw, all_file_names)):
            adv_name = f"{i:03d}__{str(name)}"
            _write_cooked_trace_file(out_dir=str(out_dir), file_name=str(adv_name), cooked_time=list(t), cooked_bw=list(bw))
            adv_all_cooked_time.append(list(t))
            adv_all_cooked_bw.append(list(bw))
            adv_file_names.append(str(adv_name))
            manifest["traces"].append(
                {
                    "idx": int(i),
                    "clean_name": str(name),
                    "adv_name": str(adv_name),
                }
            )
        tracegen_stats["pgd/history_sparse_applied-rate"] = 0.0
        _write_agg_stats_or_zero(tracegen_stats, "pgd/history_sparse_k", sparse_k_values, include_geo_mean=False)
        _write_agg_stats_or_zero(
            tracegen_stats, "pgd/history_sparse_k_unique", sparse_k_unique_values, include_geo_mean=False
        )
        _write_agg_stats_or_zero(
            tracegen_stats, "pgd/history_sparse_candidates_total", sparse_candidates_total_values, include_geo_mean=False
        )
        manifest["tracegen_stats"] = dict(tracegen_stats)
        with open(os.path.join(str(out_dir), "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        if bool(args.pgd_history_sparse):
            print("* tracegen pgd/history_sparse_k values (n=0): []", flush=True)
        return str(out_dir), adv_all_cooked_time, adv_all_cooked_bw, adv_file_names, tracegen_stats

    if pgd_attacker is None:
        raise RuntimeError("* --two-pass requires PGD attacker when --attack is enabled")

    try:
        from smt import synthesize_bandwidth_trace
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "* Failed to import SMT solver. Install deps (e.g., `pip install z3-solver`) and retry."
        ) from e

    history_delay = bool(args.pgd_history_delay)
    history_throughput = bool(args.pgd_history_throughput)
    use_window_plan = bool(history_delay or history_throughput)

    buffer_norm_max = float(env.BUFFER_THRESH / env.MILLISECONDS_IN_SECOND / BUFFER_NORM_FACTOR)

    net_env = env.Environment(all_cooked_time=all_cooked_time, all_cooked_bw=all_cooked_bw)

    state = np.zeros((S_INFO, S_LEN), dtype=np.float64)
    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY

    step = 0
    video_count = 0
    rng = np.random.RandomState(int(args.seed))
    attack_method = str(getattr(args, "attack_method", "pgd"))

    cur_trace_name: str | None = None
    cur_bw_segments: list[float] = []
    cur_dur_segments: list[float] = []
    #* Per-chunk stats captured during trace synthesis (Phase 1) for later comparison in Phase 2.
    cur_expected_realized_bw_mbps: list[float] = []
    cur_expected_action_exec: list[int] = []
    cur_expected_action_next: list[int] = []
    cur_expected_delay_ms: list[float] = []
    cur_expected_throughput_mbps: list[float] = []
    cur_expected_video_chunk_size_bytes: list[int] = []
    cur_expected_pgd_iters_run: list[int] = []

    adv_all_cooked_time: list = []
    adv_all_cooked_bw: list = []
    adv_file_names: list[str] = []

    window_plan = None  # list[dict[str, object]] | None
    window_plan_idx = 0
    window_plan_solve_time_ms: float | None = None

    while True:
        trace_name = str(all_file_names[net_env.trace_idx])
        if cur_trace_name is None:
            cur_trace_name = trace_name
            cur_expected_realized_bw_mbps = []
            cur_expected_action_exec = []
            cur_expected_action_next = []
            cur_expected_delay_ms = []
            cur_expected_throughput_mbps = []
            cur_expected_video_chunk_size_bytes = []
            cur_expected_pgd_iters_run = []

        initial_buffer_ms = float(net_env.buffer_size)

        #* Clean shadow step for PGD baseline.
        clean_env = copy_env(net_env, RANDOM_SEED=args.seed)
        delay_ms_clean, sleep_time_ms_clean, buffer_s_clean, _, video_chunk_size, next_sizes_clean, end_clean, remain_clean = (
            clean_env.get_video_chunk(int(bit_rate))
        )
        clean_state_next = _update_state(
            state,
            bit_rate=int(bit_rate),
            buffer_s=float(buffer_s_clean),
            video_chunk_size=int(video_chunk_size),
            delay_ms=float(delay_ms_clean),
            next_video_chunk_sizes=[int(x) for x in next_sizes_clean],
            video_chunk_remain=int(remain_clean),
        )

        #* Default: clean transition (used if SMT fails).
        applied_env_attack = False
        delay_ms = float(delay_ms_clean)
        sleep_time_ms = float(sleep_time_ms_clean)
        buffer_s = float(buffer_s_clean)
        rebuf_s = 0.0
        end_of_video = bool(end_clean)
        next_sizes = [int(x) for x in next_sizes_clean]
        video_chunk_remain = int(remain_clean)

        #^ Attack chooses a target state (feature space), then SMT synthesizes a per-chunk trace.
        sym_metrics: dict[str, float | int] | None = None
        pgd_iters_run = 0
        pre_bw: list[float] | None = None
        pre_tt: list[float] | None = None
        if attack_method == "dt":
            pgd_state, sym_metrics = _symbolic_dt_attack_state(
                args,
                actor=actor,
                pgd_attacker=pgd_attacker,
                clean_state_next=clean_state_next,
                oracle_env_snapshot=clean_env,
                last_bitrate_idx=int(bit_rate),
                video_chunk_size_bytes=int(video_chunk_size),
                allow_delay_idx=[int(clean_state_next.shape[1]) - 1],
                rng=rng,
            )
        elif attack_method == "dt-truegap":
            pgd_state, sym_metrics, pre_bw, pre_tt = _symbolic_dt_truegap_attack_state(
                args,
                actor=actor,
                pgd_attacker=pgd_attacker,
                clean_state_next=clean_state_next,
                clean_env_snapshot=clean_env,
                env_before_chunk=net_env,
                state_before_chunk=state,
                last_bitrate_idx=int(bit_rate),
                video_chunk_size_bytes=int(video_chunk_size),
                allow_delay_idx=[int(clean_state_next.shape[1]) - 1],
                rng=rng,
            )
        else:
            pgd_state = pgd_attacker.pgd_attack(
                clean_state_next,
                target_bitrate=int(bit_rate),
                BUFFER_NORM_FACTOR=buffer_norm_max,
                M_IN_K=M_IN_K,
                video_chunk_size=int(video_chunk_size),
                enforce_coupling=bool(args.coupling),
                perturb_buffer=bool(args.pgd_buffer),
                use_rules=bool(args.use_rules),
                rules_file=str(args.rules_file),
                rules_project_every_step=bool(args.rules_project_every_step),
                rules_perturb_both=bool(args.rules_perturb_both),
                perturb_delay_history=bool(history_delay),
                perturb_throughput_history=bool(history_throughput),
                history_sparse=bool(args.pgd_history_sparse),
                history_sparse_tol=float(args.pgd_history_sparse_tol),
                oracle_env_snapshot=clean_env,
                oracle_k_steps=K_steps,
                objective_id=int(args.pgd_objective_id),
            )
            early_stop = pgd_attacker.get_early_stop_last_attack()
            pgd_iters_run = int(early_stop.get("pgd/early_stop_iters_run", args.pgd_iters))
            if bool(args.use_rules):
                rules_stats = pgd_attacker.get_rules_projection_last_attack_stats()
                tracegen_stats["pgd/rules_projection_success_sum"] = int(
                    tracegen_stats.get("pgd/rules_projection_success_sum", 0)
                ) + int(rules_stats.get("pgd/rules_projection_success", 0))
                tracegen_stats["pgd/rules_projection_fail_sum"] = int(
                    tracegen_stats.get("pgd/rules_projection_fail_sum", 0)
                ) + int(rules_stats.get("pgd/rules_projection_fail", 0))
                tracegen_stats["pgd/rules_projection_attempts_sum"] = int(
                    tracegen_stats.get("pgd/rules_projection_attempts_sum", 0)
                ) + int(rules_stats.get("pgd/rules_projection_attempts", 0))
                tracegen_stats["pgd/rules_projection_fallbacks_sum"] = int(
                    tracegen_stats.get("pgd/rules_projection_fallbacks_sum", 0)
                ) + int(rules_stats.get("pgd/rules_projection_fallbacks", 0))

        if attack_method not in ("dt", "dt-truegap"):
            sparse = pgd_attacker.get_history_sparse_last_attack()
            sparse_attempts += 1
            sparse_applied_sum += int(sparse.get("pgd/history_sparse_applied", 0))
            if int(sparse.get("pgd/history_sparse_applied", 0)) != 0:
                sparse_k_values.append(float(sparse.get("pgd/history_sparse_k", 0)))
                sparse_k_unique_values.append(float(sparse.get("pgd/history_sparse_k_unique", 0)))
                sparse_candidates_total_values.append(float(sparse.get("pgd/history_sparse_candidates_total", 0)))

        bw: list[float] | None = pre_bw
        tt: list[float] | None = pre_tt
        if use_window_plan:
            #* Always drive history attacks via a multi-chunk SMT plan (ignore per-step dt-truegap traces).
            bw = None
            tt = None

        skip_smt = bool(
            attack_method == "dt-truegap" and sym_metrics is not None and int(sym_metrics.get("sym/applied", 0)) == 0
        )
        if (bw is None or tt is None) and not bool(skip_smt):
            if use_window_plan:
                #* Validate the current plan entry against the current action/chunk size.
                if window_plan is not None and window_plan_idx < len(window_plan):
                    expected = window_plan[window_plan_idx]
                    expected_action = int(expected["action"])
                    expected_chunk_bytes = int(expected["chunk_size_bytes"])
                    actual_chunk_bytes = int(net_env.video_size[int(bit_rate)][int(net_env.video_chunk_counter)])
                    if expected_action != int(bit_rate) or expected_chunk_bytes != actual_chunk_bytes:
                        window_plan = None
                        window_plan_idx = 0

                #* Build a fresh plan if needed (starting at the current chunk).
                if window_plan is None:
                    plan_actions: list[int] = [int(bit_rate)]
                    plan_chunk_sizes: list[int] = [int(video_chunk_size)]

                    target_delay_norms: list[float] = [float(pgd_state[3, -1])]
                    target_throughput_norms_plan: list[float] = [float(pgd_state[2, -1])]
                    last_pgd_state = np.array(pgd_state, copy=True)
                    pred_state_for_action = np.array(pgd_state, copy=True)
                    pred_state_clean = np.array(clean_state_next, copy=True)
                    pred_env = copy_env(clean_env, RANDOM_SEED=args.seed)
                    for _ in range(1, S_LEN):
                        p_pred = actor.predict(np.reshape(pred_state_for_action, (1, S_INFO, S_LEN))).astype(np.float64).reshape(-1)
                        if args.action_selection == "argmax":
                            pred_action = int(np.argmax(p_pred))
                        elif args.action_selection == "gumbel":
                            noise = np.random.gumbel(size=p_pred.shape)
                            pred_action = int(np.argmax(np.log(p_pred) + noise))
                        else:
                            raise ValueError(f"Unknown action_selection: {args.action_selection}")

                        env_before_chunk_p = copy_env(pred_env, RANDOM_SEED=args.seed)
                        state_before_chunk_p = np.array(pred_state_clean, copy=True)

                        delay_ms_p, sleep_ms_p, buffer_s_p, _, chunk_size_p, next_sizes_p, end_p, remain_p = pred_env.get_video_chunk(
                            int(pred_action)
                        )
                        clean_state_next_p = _update_state(
                            pred_state_clean,
                            bit_rate=int(pred_action),
                            buffer_s=float(buffer_s_p),
                            video_chunk_size=int(chunk_size_p),
                            delay_ms=float(delay_ms_p),
                            next_video_chunk_sizes=[int(x) for x in next_sizes_p],
                            video_chunk_remain=int(remain_p),
                        )

                        oracle_snapshot = copy_env(pred_env, RANDOM_SEED=args.seed)
                        sym_p_metrics: dict[str, float | int] | None = None
                        if attack_method == "dt":
                            chunk_idx = int(len(plan_chunk_sizes))
                            allow_from = max(0, int(S_LEN) - 1 - int(chunk_idx))
                            allow_idx = list(range(int(allow_from), int(S_LEN)))
                            pgd_state_p, sym_p_metrics = _symbolic_dt_attack_state(
                                args,
                                actor=actor,
                                pgd_attacker=pgd_attacker,
                                clean_state_next=clean_state_next_p,
                                oracle_env_snapshot=oracle_snapshot,
                                last_bitrate_idx=int(pred_action),
                                video_chunk_size_bytes=int(chunk_size_p),
                                allow_delay_idx=[int(i) for i in allow_idx],
                                rng=rng,
                            )
                        elif attack_method == "dt-truegap":
                            chunk_idx = int(len(plan_chunk_sizes))
                            allow_from = max(0, int(S_LEN) - 1 - int(chunk_idx))
                            allow_idx = list(range(int(allow_from), int(S_LEN)))
                            pgd_state_p, sym_p_metrics, _bw_p, _tt_p = _symbolic_dt_truegap_attack_state(
                                args,
                                actor=actor,
                                pgd_attacker=pgd_attacker,
                                clean_state_next=clean_state_next_p,
                                clean_env_snapshot=oracle_snapshot,
                                env_before_chunk=env_before_chunk_p,
                                state_before_chunk=state_before_chunk_p,
                                last_bitrate_idx=int(pred_action),
                                video_chunk_size_bytes=int(chunk_size_p),
                                allow_delay_idx=[int(i) for i in allow_idx],
                                rng=rng,
                            )
                        else:
                            pgd_state_p = pgd_attacker.pgd_attack(
                                clean_state_next_p,
                                target_bitrate=int(pred_action),
                                BUFFER_NORM_FACTOR=buffer_norm_max,
                                M_IN_K=M_IN_K,
                                video_chunk_size=int(chunk_size_p),
                                enforce_coupling=bool(args.coupling),
                                perturb_buffer=bool(args.pgd_buffer),
                                use_rules=bool(args.use_rules),
                                rules_file=str(args.rules_file),
                                rules_project_every_step=bool(args.rules_project_every_step),
                                rules_perturb_both=bool(args.rules_perturb_both),
                                perturb_delay_history=bool(history_delay),
                                perturb_throughput_history=bool(history_throughput),
                                history_sparse=bool(args.pgd_history_sparse),
                                history_sparse_tol=float(args.pgd_history_sparse_tol),
                                oracle_env_snapshot=oracle_snapshot,
                                oracle_k_steps=K_steps,
                                objective_id=int(args.pgd_objective_id),
                            )
                            if bool(args.use_rules):
                                rules_stats = pgd_attacker.get_rules_projection_last_attack_stats()
                                tracegen_stats["pgd/rules_projection_success_sum"] = int(
                                    tracegen_stats.get("pgd/rules_projection_success_sum", 0)
                                ) + int(rules_stats.get("pgd/rules_projection_success", 0))
                                tracegen_stats["pgd/rules_projection_fail_sum"] = int(
                                    tracegen_stats.get("pgd/rules_projection_fail_sum", 0)
                                ) + int(rules_stats.get("pgd/rules_projection_fail", 0))
                                tracegen_stats["pgd/rules_projection_attempts_sum"] = int(
                                    tracegen_stats.get("pgd/rules_projection_attempts_sum", 0)
                                ) + int(rules_stats.get("pgd/rules_projection_attempts", 0))
                                tracegen_stats["pgd/rules_projection_fallbacks_sum"] = int(
                                    tracegen_stats.get("pgd/rules_projection_fallbacks_sum", 0)
                                ) + int(rules_stats.get("pgd/rules_projection_fallbacks", 0))

                        plan_actions.append(int(pred_action))
                        plan_chunk_sizes.append(int(chunk_size_p))
                        if sym_p_metrics is not None and int(sym_p_metrics.get("sym/applied", 0)) != 0:
                            i = int(len(plan_chunk_sizes) - 1)
                            for j in range(int(i)):
                                k = int(S_LEN) - 1 - (int(i) - int(j))
                                if 0 <= k < int(S_LEN):
                                    target_delay_norms[int(j)] = float(pgd_state_p[3, int(k)])
                                    target_throughput_norms_plan[int(j)] = float(pgd_state_p[2, int(k)])

                        target_delay_norms.append(float(pgd_state_p[3, -1]))
                        target_throughput_norms_plan.append(float(pgd_state_p[2, -1]))

                        last_pgd_state = np.array(pgd_state_p, copy=True)
                        pred_state_for_action = np.array(pgd_state_p, copy=True)
                        pred_state_clean = np.array(clean_state_next_p, copy=True)

                        if bool(end_p):
                            break

                    #* Sequential per-chunk SMT solves: each chunk gets its own `--bw-samples` segments.
                    smt_t0 = time.perf_counter()
                    buffer_ms_plan = float(initial_buffer_ms)
                    plan_ok = True
                    plan_entries: list[dict[str, object]] = []

                    for j in range(len(plan_chunk_sizes)):
                        action_j = int(plan_actions[j])
                        chunk_bytes_j = int(plan_chunk_sizes[j])
                        tgt_delay_norm_j = float(target_delay_norms[j])
                        tgt_thr_norm_j = float(target_throughput_norms_plan[j])
                        enforce_buf_j = bool(args.pgd_buffer) and (j == len(plan_chunk_sizes) - 1)

                        bw_j, tt_j = synthesize_bandwidth_trace(
                            video_chunk_size=int(chunk_bytes_j),
                            initial_buffer_ms=float(buffer_ms_plan),
                            target_buffer_norm=float(last_pgd_state[1, -1]),
                            target_throughput_norm=float(tgt_thr_norm_j),
                            target_delay_norm=float(tgt_delay_norm_j),
                            num_bw_samples=int(args.smt_bw_samples),
                            enforce_buffer_target=bool(enforce_buf_j),
                            enforce_throughput_target=bool(history_throughput),
                            enforce_trace_rules=bool(args.smt_trace_rules),
                            rel_tol=float(args.smt_window_rel_tol),
                        )
                        if bw_j is None or tt_j is None:
                            plan_ok = False
                            break

                        delay_ms_j, sleep_ms_j, buffer_s_j, _, complete_j = _simulate_chunk_from_piecewise_bw(
                            bandwidth_trace_mbps=[float(x) for x in bw_j],
                            time_trace_s=[float(x) for x in tt_j],
                            video_chunk_size_bytes=int(chunk_bytes_j),
                            initial_buffer_ms=float(buffer_ms_plan),
                        )
                        if not bool(complete_j):
                            plan_ok = False
                            break

                        buffer_ms_plan = float(buffer_s_j * env.MILLISECONDS_IN_SECOND)
                        plan_entries.append(
                            {
                                "action": int(action_j),
                                "chunk_size_bytes": int(chunk_bytes_j),
                                "bw": [float(x) for x in bw_j],
                                "t": [float(x) for x in tt_j],
                            }
                        )

                    window_plan_solve_time_ms = float((time.perf_counter() - smt_t0) * 1000.0)
                    if plan_ok and len(plan_entries) == len(plan_chunk_sizes):
                        window_plan = plan_entries
                        window_plan_idx = 0

                if window_plan is not None and window_plan_idx < len(window_plan):
                    cur = window_plan[window_plan_idx]
                    bw = [float(x) for x in cur["bw"]]
                    tt = [float(x) for x in cur["t"]]
            else:
                bw_trace, t_trace = synthesize_bandwidth_trace(
                    video_chunk_size=int(video_chunk_size),
                    initial_buffer_ms=float(initial_buffer_ms),
                    target_buffer_norm=float(pgd_state[1, -1]),
                    target_throughput_norm=float(pgd_state[2, -1]),
                    target_delay_norm=float(pgd_state[3, -1]),
                    num_bw_samples=int(args.smt_bw_samples),
                    enforce_buffer_target=bool(args.pgd_buffer),
                    enforce_throughput_target=True,
                    enforce_trace_rules=bool(args.smt_trace_rules),
                    rel_tol=float(args.smt_window_rel_tol),
                )
                if bw_trace is not None and t_trace is not None:
                    bw = [float(x) for x in bw_trace]
                    tt = [float(x) for x in t_trace]

        smt_attempted = int(not bool(skip_smt))
        tracegen_stats["smt/skipped"] = int(tracegen_stats["smt/skipped"]) + int(bool(skip_smt))
        tracegen_stats["smt/attempts"] = int(tracegen_stats["smt/attempts"]) + int(smt_attempted)
        smt_success = bw is not None and tt is not None
        tracegen_stats["smt/successes"] = int(tracegen_stats["smt/successes"]) + int(bool(smt_success)) * int(smt_attempted)

        if bw is not None and tt is not None:
            overridden = _get_video_chunk_override(net_env, int(bit_rate), bandwidth_trace_mbps=bw, time_trace_s=tt)
            if overridden is not None:
                (
                    delay_ms,
                    sleep_time_ms,
                    buffer_s,
                    rebuf_s,
                    video_chunk_size,
                    next_sizes,
                    end_of_video,
                    video_chunk_remain,
                ) = overridden
                applied_env_attack = True
                if use_window_plan and window_plan is not None:
                    window_plan_idx += 1
                    if window_plan_idx >= len(window_plan):
                        window_plan = None
                        window_plan_idx = 0
                        window_plan_solve_time_ms = None
                tracegen_stats["smt/applied"] = int(tracegen_stats["smt/applied"]) + 1
                tracegen_stats["smt/complete"] = int(tracegen_stats["smt/complete"]) + 1

                #* Record the realized per-chunk raw trace (download prefix + sleep filler).
                dl_bw, dl_tt, _, dl_complete = _extract_download_prefix_segments(
                    bandwidth_trace_mbps=[float(x) for x in bw],
                    duration_trace_s=[float(x) for x in tt],
                    video_chunk_size_bytes=int(video_chunk_size),
                )
                if dl_complete and len(dl_bw) == len(dl_tt) and len(dl_bw) > 0:
                    cur_bw_segments.extend([float(x) for x in dl_bw])
                    cur_dur_segments.extend([float(x) for x in dl_tt])
                    sleep_s = float(sleep_time_ms) / env.MILLISECONDS_IN_SECOND
                    if sleep_s > 1e-12:
                        cur_bw_segments.append(float(dl_bw[-1]))
                        cur_dur_segments.append(float(sleep_s))
                else:
                    #* Defensive fallback: export a 1-segment trace matching this realized delay.
                    rtt_free_s = float(max(float(delay_ms) - float(env.LINK_RTT), 0.0) / env.MILLISECONDS_IN_SECOND)
                    if rtt_free_s > 1e-12:
                        avg_bw = float(int(video_chunk_size)) * 8.0 / (
                            float(rtt_free_s) * float(env.B_IN_MB) * float(env.PACKET_PAYLOAD_PORTION)
                        )
                        cur_bw_segments.append(float(max(avg_bw, 1e-6)))
                        cur_dur_segments.append(float(rtt_free_s))
                    sleep_s = float(sleep_time_ms) / env.MILLISECONDS_IN_SECOND
                    if sleep_s > 1e-12:
                        bw_sleep = float(cur_bw_segments[-1]) if cur_bw_segments else 1.0
                        cur_bw_segments.append(float(bw_sleep))
                        cur_dur_segments.append(float(sleep_s))

        if sym_metrics is not None:
            tracegen_stats["sym/attempts"] = int(tracegen_stats.get("sym/attempts", 0)) + 1
            tracegen_stats["sym/rule_found_sum"] = int(tracegen_stats.get("sym/rule_found_sum", 0)) + int(
                sym_metrics.get("sym/rule_found", 0)
            )
            tracegen_stats["sym/projection_success_sum"] = int(
                tracegen_stats.get("sym/projection_success_sum", 0)
            ) + int(sym_metrics.get("sym/projection_success", 0))
            tracegen_stats["sym/teacher_check_pass_sum"] = int(
                tracegen_stats.get("sym/teacher_check_pass_sum", 0)
            ) + int(sym_metrics.get("sym/teacher_check_pass", 0))

            #* `sym/applied-rate` should reflect end-to-end application (feature attack + successful SMT/override).
            sym_feature_applied = int(sym_metrics.get("sym/applied", 0))
            sym_env_applied = int(bool(sym_feature_applied) and bool(applied_env_attack))
            tracegen_stats["sym/applied_sum"] = int(tracegen_stats.get("sym/applied_sum", 0)) + int(sym_env_applied)

            tracegen_stats["symtrue/evals_attempted_sum"] = int(tracegen_stats.get("symtrue/evals_attempted_sum", 0)) + int(
                sym_metrics.get("symtrue/evals_attempted", 0)
            )
            tracegen_stats["symtrue/evals_smt_success_sum"] = int(
                tracegen_stats.get("symtrue/evals_smt_success_sum", 0)
            ) + int(sym_metrics.get("symtrue/evals_smt_success", 0))
            tracegen_stats["symtrue/evals_override_success_sum"] = int(
                tracegen_stats.get("symtrue/evals_override_success_sum", 0)
            ) + int(sym_metrics.get("symtrue/evals_override_success", 0))
            tracegen_stats["symtrue/evals_oracle_valid_sum"] = int(
                tracegen_stats.get("symtrue/evals_oracle_valid_sum", 0)
            ) + int(sym_metrics.get("symtrue/evals_oracle_valid", 0))
            tracegen_stats["symtrue/evals_policy_valid_sum"] = int(
                tracegen_stats.get("symtrue/evals_policy_valid_sum", 0)
            ) + int(sym_metrics.get("symtrue/evals_policy_valid", 0))
            tracegen_stats["symtrue/evals_oracle_filter_pass_sum"] = int(
                tracegen_stats.get("symtrue/evals_oracle_filter_pass_sum", 0)
            ) + int(sym_metrics.get("symtrue/evals_oracle_filter_pass", 0))
            tracegen_stats["symtrue/smt_solve_time_ms_selected_sum"] = float(
                tracegen_stats.get("symtrue/smt_solve_time_ms_selected_sum", 0.0)
            ) + float(sym_metrics.get("symtrue/smt_solve_time_ms_selected", 0.0))
            if int(sym_env_applied) != 0:
                tracegen_stats["symtrue/gap_true_sum"] = float(tracegen_stats.get("symtrue/gap_true_sum", 0.0)) + float(
                    sym_metrics.get("symtrue/gap_true", 0.0)
                )
                tracegen_stats["symtrue/oracle_best_reward_mean_sum"] = float(
                    tracegen_stats.get("symtrue/oracle_best_reward_mean_sum", 0.0)
                ) + float(sym_metrics.get("symtrue/oracle_best_reward_mean", 0.0))
                tracegen_stats["symtrue/policy_reward_mean_sum"] = float(
                    tracegen_stats.get("symtrue/policy_reward_mean_sum", 0.0)
                ) + float(sym_metrics.get("symtrue/policy_reward_mean", 0.0))

        if not bool(applied_env_attack):
            #* Fallback: use the clean trace segments directly.
            (ret, bw_seg, dur_seg) = _get_video_chunk_record_trace(net_env, int(bit_rate))
            (
                delay_ms,
                sleep_time_ms,
                buffer_s,
                rebuf_s,
                video_chunk_size,
                next_sizes,
                end_of_video,
                video_chunk_remain,
            ) = ret
            cur_bw_segments.extend([float(x) for x in bw_seg])
            cur_dur_segments.extend([float(x) for x in dur_seg])
            if use_window_plan:
                window_plan = None
                window_plan_idx = 0
                window_plan_solve_time_ms = None

        #* Expected bandwidth implied by the chunk transfer *during trace generation*.
        cur_expected_realized_bw_mbps.append(float(_avg_bandwidth_mbps(float(video_chunk_size), float(delay_ms))))
        #* Persist Phase-1 per-chunk execution details for Phase-2 comparison.
        cur_expected_action_exec.append(int(bit_rate))
        cur_expected_delay_ms.append(float(delay_ms))
        cur_expected_throughput_mbps.append(float(_throughput_mbps(float(video_chunk_size), float(delay_ms))))
        cur_expected_video_chunk_size_bytes.append(int(video_chunk_size))
        cur_expected_pgd_iters_run.append(int(pgd_iters_run))

        state = _update_state(
            state,
            bit_rate=int(bit_rate),
            buffer_s=float(buffer_s),
            video_chunk_size=int(video_chunk_size),
            delay_ms=float(delay_ms),
            next_video_chunk_sizes=[int(x) for x in next_sizes],
            video_chunk_remain=int(video_chunk_remain),
        )

        p_obs = actor.predict(np.reshape(state, (1, S_INFO, S_LEN))).astype(np.float64).reshape(-1)
        if args.action_selection == "argmax":
            next_bit_rate = int(np.argmax(p_obs))
        elif args.action_selection == "gumbel":
            noise = np.random.gumbel(size=len(p_obs))
            next_bit_rate = int(np.argmax(np.log(p_obs) + noise))
        else:
            raise ValueError(f"Unknown action_selection: {args.action_selection}")

        cur_expected_action_next.append(int(next_bit_rate))

        last_bit_rate = int(bit_rate)
        bit_rate = int(next_bit_rate)
        step += 1

        if end_of_video or (args.max_steps is not None and step >= args.max_steps):
            adv_time, adv_bw = _segments_to_cooked_trace(cur_bw_segments, cur_dur_segments)
            adv_name = f"{int(video_count):03d}__{cur_trace_name}"
            _write_cooked_trace_file(out_dir=str(out_dir), file_name=str(adv_name), cooked_time=adv_time, cooked_bw=adv_bw)
            manifest["traces"].append(
                {
                    "idx": int(video_count),
                    "clean_name": str(cur_trace_name),
                    "adv_name": str(adv_name),
                    "steps": int(net_env.video_chunk_counter) if not bool(end_of_video) else int(env.TOTAL_VIDEO_CHUNCK),
                    #* Logged in Phase 2 under `tracegen/adv/realized_bw_mbps` so it shares the same x-axis as rollout metrics.
                    "expected_realized_bw_mbps": [float(x) for x in cur_expected_realized_bw_mbps],
                    #* Additional Phase-1 per-chunk values to diagnose Phase-1 vs Phase-2 divergence.
                    "expected_action_exec": [int(x) for x in cur_expected_action_exec],
                    "expected_action_next": [int(x) for x in cur_expected_action_next],
                    "expected_delay_ms": [float(x) for x in cur_expected_delay_ms],
                    "expected_throughput_mbps": [float(x) for x in cur_expected_throughput_mbps],
                    "expected_video_chunk_size_bytes": [int(x) for x in cur_expected_video_chunk_size_bytes],
                    "expected_pgd_iters_run": [int(x) for x in cur_expected_pgd_iters_run],
                }
            )

            adv_all_cooked_time.append(list(adv_time))
            adv_all_cooked_bw.append(list(adv_bw))
            adv_file_names.append(str(adv_name))

            #* Reset episode accumulator.
            cur_trace_name = None
            cur_bw_segments = []
            cur_dur_segments = []
            cur_expected_realized_bw_mbps = []
            cur_expected_action_exec = []
            cur_expected_action_next = []
            cur_expected_delay_ms = []
            cur_expected_throughput_mbps = []
            cur_expected_video_chunk_size_bytes = []
            cur_expected_pgd_iters_run = []

            if bool(end_of_video):
                state[:] = 0.0
                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY
                if use_window_plan:
                    window_plan = None
                    window_plan_idx = 0
                    window_plan_solve_time_ms = None
                video_count += 1

            if (args.max_steps is not None and step >= args.max_steps) or video_count >= len(all_file_names):
                break

    if pgd_attacker is not None:
        total_calls, total_fully_relaxed = pgd_attacker.get_rules_projection_totals()
        tracegen_stats["pgd/rules_projection_calls_total"] = int(total_calls)
        tracegen_stats["pgd/rules_fully_relaxed_calls_total"] = int(total_fully_relaxed)
        tracegen_stats["pgd/rules_fully_relaxed_rate_total"] = float(total_fully_relaxed) / float(max(total_calls, 1))
        attempts_sum = int(tracegen_stats.get("pgd/rules_projection_attempts_sum", 0))
        success_sum = int(tracegen_stats.get("pgd/rules_projection_success_sum", 0))
        tracegen_stats["pgd/rules_projection_success-rate"] = float(success_sum) / float(max(attempts_sum, 1))

    tracegen_stats["pgd/history_sparse_applied-rate"] = float(sparse_applied_sum) / float(max(sparse_attempts, 1))
    _write_agg_stats_or_zero(tracegen_stats, "pgd/history_sparse_k", sparse_k_values, include_geo_mean=False)
    _write_agg_stats_or_zero(tracegen_stats, "pgd/history_sparse_k_unique", sparse_k_unique_values, include_geo_mean=False)
    _write_agg_stats_or_zero(
        tracegen_stats, "pgd/history_sparse_candidates_total", sparse_candidates_total_values, include_geo_mean=False
    )
    if bool(args.pgd_history_sparse):
        ks = [int(x) for x in sparse_k_values]
        print(f"* tracegen pgd/history_sparse_k values (n={len(ks)}): {ks}", flush=True)

    sym_attempts = int(tracegen_stats.get("sym/attempts", 0))
    if sym_attempts > 0:
        tracegen_stats["sym/applied-rate"] = float(tracegen_stats.get("sym/applied_sum", 0)) / float(sym_attempts)
        tracegen_stats["sym/rule_found-rate"] = float(tracegen_stats.get("sym/rule_found_sum", 0)) / float(sym_attempts)
        tracegen_stats["sym/projection_success-rate"] = float(tracegen_stats.get("sym/projection_success_sum", 0)) / float(
            sym_attempts
        )
        tracegen_stats["sym/teacher_check_pass-rate"] = float(tracegen_stats.get("sym/teacher_check_pass_sum", 0)) / float(
            sym_attempts
        )
    else:
        tracegen_stats["sym/applied-rate"] = 0.0
        tracegen_stats["sym/rule_found-rate"] = 0.0
        tracegen_stats["sym/projection_success-rate"] = 0.0
        tracegen_stats["sym/teacher_check_pass-rate"] = 0.0

    sym_applied = int(tracegen_stats.get("sym/applied_sum", 0))
    if sym_applied > 0:
        tracegen_stats["symtrue/gap_true-mean"] = float(tracegen_stats.get("symtrue/gap_true_sum", 0.0)) / float(sym_applied)
        tracegen_stats["symtrue/oracle_best_reward_mean-mean"] = float(
            tracegen_stats.get("symtrue/oracle_best_reward_mean_sum", 0.0)
        ) / float(sym_applied)
        tracegen_stats["symtrue/policy_reward_mean-mean"] = float(tracegen_stats.get("symtrue/policy_reward_mean_sum", 0.0)) / float(
            sym_applied
        )
    else:
        tracegen_stats["symtrue/gap_true-mean"] = 0.0
        tracegen_stats["symtrue/oracle_best_reward_mean-mean"] = 0.0
        tracegen_stats["symtrue/policy_reward_mean-mean"] = 0.0

    symtrue_evals_attempted_sum = int(tracegen_stats.get("symtrue/evals_attempted_sum", 0))
    if symtrue_evals_attempted_sum > 0:
        tracegen_stats["symtrue/evals_smt_success-rate"] = float(tracegen_stats.get("symtrue/evals_smt_success_sum", 0)) / float(
            symtrue_evals_attempted_sum
        )
        tracegen_stats["symtrue/evals_override_success-rate"] = float(
            tracegen_stats.get("symtrue/evals_override_success_sum", 0)
        ) / float(symtrue_evals_attempted_sum)
        tracegen_stats["symtrue/evals_oracle_filter_pass-rate"] = float(
            tracegen_stats.get("symtrue/evals_oracle_filter_pass_sum", 0)
        ) / float(symtrue_evals_attempted_sum)
    else:
        tracegen_stats["symtrue/evals_smt_success-rate"] = 0.0
        tracegen_stats["symtrue/evals_override_success-rate"] = 0.0
        tracegen_stats["symtrue/evals_oracle_filter_pass-rate"] = 0.0

    manifest["tracegen_stats"] = dict(tracegen_stats)
    with open(os.path.join(str(out_dir), "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    return str(out_dir), adv_all_cooked_time, adv_all_cooked_bw, adv_file_names, tracegen_stats


def _paired_clean_vs_adv_eval(
    args: argparse.Namespace,
    *,
    actor: network.Network,
    clean_all_cooked_time: list,
    clean_all_cooked_bw: list,
    clean_file_names: list[str],
    adv_all_cooked_time: list,
    adv_all_cooked_bw: list,
    adv_file_names: list[str],
    tracegen_expected_realized_bw_by_trace: dict[str, list[float]] | None = None,
    tracegen_expected_action_exec_by_trace: dict[str, list[float]] | None = None,
    tracegen_expected_action_next_by_trace: dict[str, list[float]] | None = None,
    tracegen_expected_delay_ms_by_trace: dict[str, list[float]] | None = None,
    tracegen_expected_throughput_mbps_by_trace: dict[str, list[float]] | None = None,
    tracegen_expected_video_chunk_size_bytes_by_trace: dict[str, list[float]] | None = None,
    tracegen_expected_pgd_iters_run_by_trace: dict[str, list[float]] | None = None,
    wandb_run: wandb.sdk.wandb_run.Run | None,
) -> None:
    #* Phase 2 (evaluation): roll out the policy on both clean and synthesized traces and compare.
    if tracegen_expected_realized_bw_by_trace is None:
        tracegen_expected_realized_bw_by_trace = {}
    if tracegen_expected_action_exec_by_trace is None:
        tracegen_expected_action_exec_by_trace = {}
    if tracegen_expected_action_next_by_trace is None:
        tracegen_expected_action_next_by_trace = {}
    if tracegen_expected_delay_ms_by_trace is None:
        tracegen_expected_delay_ms_by_trace = {}
    if tracegen_expected_throughput_mbps_by_trace is None:
        tracegen_expected_throughput_mbps_by_trace = {}
    if tracegen_expected_video_chunk_size_bytes_by_trace is None:
        tracegen_expected_video_chunk_size_bytes_by_trace = {}
    if tracegen_expected_pgd_iters_run_by_trace is None:
        tracegen_expected_pgd_iters_run_by_trace = {}

    #* K-step oracle QoE (exact) used for oracle-gap diagnostics in Phase 2.
    #* Uses the same `K_steps` horizon as the PGD trace generation objective.
    def _oracle_k_step_qoe(
        env_snapshot: env.Environment, last_br: int, k_steps: int
    ) -> tuple[float | None, float | None, float | None, float | None]:
        k = int(k_steps)
        if k <= 0:
            return None, None, None, None

        best_r, best_seq, _, _ = find_optimal_k_step_reward_oracle(env_snapshot, int(last_br), int(k))
        if best_r is None or best_seq is None:
            return None, None, None, None

        #* Derive QoE component totals for the best sequence (raw values, no penalties).
        env_sim = copy_env(env_snapshot, RANDOM_SEED=int(args.seed))
        total_quality = 0.0
        total_rebuf_s = 0.0
        total_smoothness = 0.0
        last = int(last_br)
        for idx, action in enumerate(best_seq):
            a = int(action)
            try:
                _, _, _, rebuf_s, _, _, end, _ = env_sim.get_video_chunk(a)
            except Exception:
                return None, None, None, None

            #* Match the oracle validity rule: sequences that end early are invalid for K-step comparison.
            if bool(end) and idx < int(k) - 1:
                return None, None, None, None

            total_quality += float(VIDEO_BIT_RATE[a]) / float(M_IN_K)
            total_rebuf_s += float(rebuf_s)
            total_smoothness += float(abs(VIDEO_BIT_RATE[a] - VIDEO_BIT_RATE[int(last)])) / float(M_IN_K)
            last = int(a)

        denom = float(k)
        return float(best_r) / denom, float(total_quality) / denom, float(total_rebuf_s) / denom, float(total_smoothness) / denom

    def _policy_k_step_qoe(
        env_snapshot: env.Environment,
        state_snapshot: np.ndarray,
        *,
        last_br: int,
        start_action: int,
        k_steps: int,
        rng: np.random.RandomState,
    ) -> tuple[float | None, float | None, float | None, float | None]:
        k = int(k_steps)
        if k <= 0:
            return None, None, None, None

        env_sim = copy_env(env_snapshot, RANDOM_SEED=int(args.seed))
        state_sim = np.array(state_snapshot, copy=True)

        total_reward = 0.0
        total_quality = 0.0
        total_rebuf_s = 0.0
        total_smoothness = 0.0

        last = int(last_br)
        a = int(start_action)

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

            total_reward += float(calculate_reward(int(a), int(last), float(rebuf_s)))
            total_quality += float(VIDEO_BIT_RATE[int(a)]) / float(M_IN_K)
            total_rebuf_s += float(rebuf_s)
            total_smoothness += float(abs(VIDEO_BIT_RATE[int(a)] - VIDEO_BIT_RATE[int(last)])) / float(M_IN_K)

            state_sim = _update_state(
                state_sim,
                bit_rate=int(a),
                buffer_s=float(buffer_s),
                video_chunk_size=int(chunk_size),
                delay_ms=float(delay_ms),
                next_video_chunk_sizes=[int(x) for x in next_sizes],
                video_chunk_remain=int(remain),
            )
            last = int(a)

            if i < int(k) - 1:
                p = actor.predict(np.reshape(state_sim, (1, S_INFO, S_LEN))).astype(np.float64).reshape(-1)
                if args.action_selection == "argmax":
                    a = int(np.argmax(p))
                elif args.action_selection == "gumbel":
                    noise = rng.gumbel(size=p.shape)
                    a = int(np.argmax(np.log(p) + noise))
                else:
                    raise ValueError(f"Unknown action_selection: {args.action_selection}")

        denom = float(k)
        return (
            float(total_reward) / denom,
            float(total_quality) / denom,
            float(total_rebuf_s) / denom,
            float(total_smoothness) / denom,
        )

    env_clean = env.Environment(all_cooked_time=clean_all_cooked_time, all_cooked_bw=clean_all_cooked_bw)
    env_adv = env.Environment(all_cooked_time=adv_all_cooked_time, all_cooked_bw=adv_all_cooked_bw)

    state_clean = np.zeros((S_INFO, S_LEN), dtype=np.float64)
    state_adv = np.zeros((S_INFO, S_LEN), dtype=np.float64)

    last_br_clean = DEFAULT_QUALITY
    br_clean = DEFAULT_QUALITY
    last_br_adv = DEFAULT_QUALITY
    br_adv = DEFAULT_QUALITY

    episode_return_clean = 0.0
    episode_return_adv = 0.0

    rng = np.random.RandomState(int(args.seed))

    step = 0
    episode = 0

    action_flip_sum = 0
    kl_clean_adv_values: list[float] = []
    qoe_reward_clean_values: list[float] = []
    qoe_reward_adv_values: list[float] = []
    qoe_reward_diff_values: list[float] = []
    qoe_quality_clean_values: list[float] = []
    qoe_quality_adv_values: list[float] = []
    qoe_rebuf_s_clean_values: list[float] = []
    qoe_rebuf_s_adv_values: list[float] = []
    qoe_smoothness_clean_values: list[float] = []
    qoe_smoothness_adv_values: list[float] = []
    entropy_clean_values: list[float] = []
    entropy_adv_values: list[float] = []
    max_prob_clean_values: list[float] = []
    max_prob_adv_values: list[float] = []

    tracegen_expected_adv_realized_bw_mbps_values: list[float] = []
    tracegen_expected_adv_action_exec_values: list[float] = []
    tracegen_expected_adv_action_next_values: list[float] = []
    tracegen_expected_adv_delay_ms_values: list[float] = []
    tracegen_expected_adv_throughput_mbps_values: list[float] = []
    tracegen_expected_adv_video_chunk_size_bytes_values: list[float] = []
    tracegen_expected_adv_pgd_iters_run_values: list[float] = []

    oracle_reward_clean_values: list[float] = []
    oracle_reward_adv_values: list[float] = []
    oracle_quality_clean_values: list[float] = []
    oracle_quality_adv_values: list[float] = []
    oracle_rebuf_s_clean_values: list[float] = []
    oracle_rebuf_s_adv_values: list[float] = []
    oracle_smoothness_clean_values: list[float] = []
    oracle_smoothness_adv_values: list[float] = []

    oracle_gap_reward_clean_values: list[float] = []
    oracle_gap_reward_adv_values: list[float] = []
    oracle_gap_quality_clean_values: list[float] = []
    oracle_gap_quality_adv_values: list[float] = []
    oracle_gap_rebuf_s_clean_values: list[float] = []
    oracle_gap_rebuf_s_adv_values: list[float] = []
    oracle_gap_smoothness_clean_values: list[float] = []
    oracle_gap_smoothness_adv_values: list[float] = []

    oracle_k_steps = int(K_steps)

    while True:
        #* Capture trace names before `get_video_chunk` advances to the next trace on end-of-video.
        clean_trace_name = str(clean_file_names[int(env_clean.trace_idx)]) if clean_file_names else "unknown"
        adv_trace_name = str(adv_file_names[int(env_adv.trace_idx)]) if adv_file_names else "unknown"

        #* K-step oracle + policy rollouts for oracle-gap diagnostics (averaged over `K_steps`).
        oracle_reward_clean, oracle_quality_clean, oracle_rebuf_s_clean, oracle_smoothness_clean = _oracle_k_step_qoe(
            env_clean, int(last_br_clean), oracle_k_steps
        )
        oracle_reward_adv, oracle_quality_adv, oracle_rebuf_s_adv, oracle_smoothness_adv = _oracle_k_step_qoe(
            env_adv, int(last_br_adv), oracle_k_steps
        )

        policy_reward_clean, policy_quality_clean, policy_rebuf_s_clean, policy_smoothness_clean = (None, None, None, None)
        policy_reward_adv, policy_quality_adv, policy_rebuf_s_adv, policy_smoothness_adv = (None, None, None, None)
        try:
            rng_state = rng.get_state()
            rng_clean = np.random.RandomState()
            rng_clean.set_state(rng_state)
            rng_adv = np.random.RandomState()
            rng_adv.set_state(rng_state)
            policy_reward_clean, policy_quality_clean, policy_rebuf_s_clean, policy_smoothness_clean = _policy_k_step_qoe(
                env_clean,
                state_clean,
                last_br=int(last_br_clean),
                start_action=int(br_clean),
                k_steps=oracle_k_steps,
                rng=rng_clean,
            )
            policy_reward_adv, policy_quality_adv, policy_rebuf_s_adv, policy_smoothness_adv = _policy_k_step_qoe(
                env_adv,
                state_adv,
                last_br=int(last_br_adv),
                start_action=int(br_adv),
                k_steps=oracle_k_steps,
                rng=rng_adv,
            )
        except Exception:
            policy_reward_clean, policy_quality_clean, policy_rebuf_s_clean, policy_smoothness_clean = (None, None, None, None)
            policy_reward_adv, policy_quality_adv, policy_rebuf_s_adv, policy_smoothness_adv = (None, None, None, None)

        #* Snapshot trace bandwidth at the current Mahimahi pointer (Mbps).
        trace_bw_clean = float(env_clean.cooked_bw[env_clean.mahimahi_ptr]) if env_clean.mahimahi_ptr < len(env_clean.cooked_bw) else float("nan")
        trace_bw_adv = float(env_adv.cooked_bw[env_adv.mahimahi_ptr]) if env_adv.mahimahi_ptr < len(env_adv.cooked_bw) else float("nan")

        (
            delay_ms_c,
            sleep_ms_c,
            buffer_s_c,
            rebuf_s_c,
            chunk_size_c,
            next_sizes_c,
            end_c,
            remain_c,
        ) = env_clean.get_video_chunk(int(br_clean))
        (
            delay_ms_a,
            sleep_ms_a,
            buffer_s_a,
            rebuf_s_a,
            chunk_size_a,
            next_sizes_a,
            end_a,
            remain_a,
        ) = env_adv.get_video_chunk(int(br_adv))

        reward_c = float(calculate_reward(int(br_clean), int(last_br_clean), float(rebuf_s_c)))
        reward_a = float(calculate_reward(int(br_adv), int(last_br_adv), float(rebuf_s_a)))

        #* QoE breakdown components (raw values, no penalties).
        qoe_quality_clean = float(VIDEO_BIT_RATE[int(br_clean)]) / float(M_IN_K)
        qoe_quality_adv = float(VIDEO_BIT_RATE[int(br_adv)]) / float(M_IN_K)
        qoe_smoothness_clean = float(abs(VIDEO_BIT_RATE[int(br_clean)] - VIDEO_BIT_RATE[int(last_br_clean)])) / float(M_IN_K)
        qoe_smoothness_adv = float(abs(VIDEO_BIT_RATE[int(br_adv)] - VIDEO_BIT_RATE[int(last_br_adv)])) / float(M_IN_K)
        qoe_rebuf_s_clean = float(rebuf_s_c)
        qoe_rebuf_s_adv = float(rebuf_s_a)

        episode_return_clean += float(reward_c)
        episode_return_adv += float(reward_a)

        state_clean = _update_state(
            state_clean,
            bit_rate=int(br_clean),
            buffer_s=float(buffer_s_c),
            video_chunk_size=int(chunk_size_c),
            delay_ms=float(delay_ms_c),
            next_video_chunk_sizes=[int(x) for x in next_sizes_c],
            video_chunk_remain=int(remain_c),
        )
        state_adv = _update_state(
            state_adv,
            bit_rate=int(br_adv),
            buffer_s=float(buffer_s_a),
            video_chunk_size=int(chunk_size_a),
            delay_ms=float(delay_ms_a),
            next_video_chunk_sizes=[int(x) for x in next_sizes_a],
            video_chunk_remain=int(remain_a),
        )

        p_clean = actor.predict(np.reshape(state_clean, (1, S_INFO, S_LEN))).astype(np.float64).reshape(-1)
        p_adv = actor.predict(np.reshape(state_adv, (1, S_INFO, S_LEN))).astype(np.float64).reshape(-1)

        entropy_clean = _entropy(p_clean)
        entropy_adv = _entropy(p_adv)
        max_prob_clean = float(np.max(p_clean))
        max_prob_adv = float(np.max(p_adv))

        if args.action_selection == "argmax":
            next_br_clean = int(np.argmax(p_clean))
            next_br_adv = int(np.argmax(p_adv))
        elif args.action_selection == "gumbel":
            noise = rng.gumbel(size=p_clean.shape)
            next_br_clean = int(np.argmax(np.log(p_clean) + noise))
            next_br_adv = int(np.argmax(np.log(p_adv) + noise))
        else:
            raise ValueError(f"Unknown action_selection: {args.action_selection}")

        action_flip = int(next_br_clean != next_br_adv)
        kl_clean_adv = float(_kl_div(p_clean, p_adv))
        reward_diff = float(reward_a - reward_c)

        action_flip_sum += int(action_flip)
        kl_clean_adv_values.append(float(kl_clean_adv))
        qoe_reward_clean_values.append(float(reward_c))
        qoe_reward_adv_values.append(float(reward_a))
        qoe_reward_diff_values.append(float(reward_diff))
        qoe_quality_clean_values.append(float(qoe_quality_clean))
        qoe_quality_adv_values.append(float(qoe_quality_adv))
        qoe_rebuf_s_clean_values.append(float(qoe_rebuf_s_clean))
        qoe_rebuf_s_adv_values.append(float(qoe_rebuf_s_adv))
        qoe_smoothness_clean_values.append(float(qoe_smoothness_clean))
        qoe_smoothness_adv_values.append(float(qoe_smoothness_adv))
        entropy_clean_values.append(float(entropy_clean))
        entropy_adv_values.append(float(entropy_adv))
        max_prob_clean_values.append(float(max_prob_clean))
        max_prob_adv_values.append(float(max_prob_adv))

        expected_adv_realized_bw_mbps = float("nan")
        expected_adv_action_exec = float("nan")
        expected_adv_action_next = float("nan")
        expected_adv_delay_ms = float("nan")
        expected_adv_throughput_mbps = float("nan")
        expected_adv_video_chunk_size_bytes = float("nan")
        expected_adv_pgd_iters_run = float("nan")
        try:
            chunk_idx_a = int(env.TOTAL_VIDEO_CHUNCK) - int(remain_a) - 1
        except Exception:
            chunk_idx_a = -1
        idx_ok = 0 <= int(chunk_idx_a)
        expected_list = tracegen_expected_realized_bw_by_trace.get(str(adv_trace_name))
        if idx_ok and isinstance(expected_list, list) and int(chunk_idx_a) < len(expected_list):
            try:
                expected_adv_realized_bw_mbps = float(expected_list[int(chunk_idx_a)])
            except Exception:
                expected_adv_realized_bw_mbps = float("nan")
        expected_list = tracegen_expected_action_exec_by_trace.get(str(adv_trace_name))
        if idx_ok and isinstance(expected_list, list) and int(chunk_idx_a) < len(expected_list):
            try:
                expected_adv_action_exec = float(expected_list[int(chunk_idx_a)])
            except Exception:
                expected_adv_action_exec = float("nan")
        expected_list = tracegen_expected_action_next_by_trace.get(str(adv_trace_name))
        if idx_ok and isinstance(expected_list, list) and int(chunk_idx_a) < len(expected_list):
            try:
                expected_adv_action_next = float(expected_list[int(chunk_idx_a)])
            except Exception:
                expected_adv_action_next = float("nan")
        expected_list = tracegen_expected_delay_ms_by_trace.get(str(adv_trace_name))
        if idx_ok and isinstance(expected_list, list) and int(chunk_idx_a) < len(expected_list):
            try:
                expected_adv_delay_ms = float(expected_list[int(chunk_idx_a)])
            except Exception:
                expected_adv_delay_ms = float("nan")
        expected_list = tracegen_expected_throughput_mbps_by_trace.get(str(adv_trace_name))
        if idx_ok and isinstance(expected_list, list) and int(chunk_idx_a) < len(expected_list):
            try:
                expected_adv_throughput_mbps = float(expected_list[int(chunk_idx_a)])
            except Exception:
                expected_adv_throughput_mbps = float("nan")
        expected_list = tracegen_expected_video_chunk_size_bytes_by_trace.get(str(adv_trace_name))
        if idx_ok and isinstance(expected_list, list) and int(chunk_idx_a) < len(expected_list):
            try:
                expected_adv_video_chunk_size_bytes = float(expected_list[int(chunk_idx_a)])
            except Exception:
                expected_adv_video_chunk_size_bytes = float("nan")
        expected_list = tracegen_expected_pgd_iters_run_by_trace.get(str(adv_trace_name))
        if idx_ok and isinstance(expected_list, list) and int(chunk_idx_a) < len(expected_list):
            try:
                expected_adv_pgd_iters_run = float(expected_list[int(chunk_idx_a)])
            except Exception:
                expected_adv_pgd_iters_run = float("nan")
        if np.isfinite(expected_adv_realized_bw_mbps):
            tracegen_expected_adv_realized_bw_mbps_values.append(float(expected_adv_realized_bw_mbps))
        if np.isfinite(expected_adv_action_exec):
            tracegen_expected_adv_action_exec_values.append(float(expected_adv_action_exec))
        if np.isfinite(expected_adv_action_next):
            tracegen_expected_adv_action_next_values.append(float(expected_adv_action_next))
        if np.isfinite(expected_adv_delay_ms):
            tracegen_expected_adv_delay_ms_values.append(float(expected_adv_delay_ms))
        if np.isfinite(expected_adv_throughput_mbps):
            tracegen_expected_adv_throughput_mbps_values.append(float(expected_adv_throughput_mbps))
        if np.isfinite(expected_adv_video_chunk_size_bytes):
            tracegen_expected_adv_video_chunk_size_bytes_values.append(float(expected_adv_video_chunk_size_bytes))
        if np.isfinite(expected_adv_pgd_iters_run):
            tracegen_expected_adv_pgd_iters_run_values.append(float(expected_adv_pgd_iters_run))

        oracle_gap_reward_clean = None
        oracle_gap_quality_clean = None
        oracle_gap_rebuf_s_clean = None
        oracle_gap_smoothness_clean = None
        if (
            oracle_reward_clean is not None
            and oracle_quality_clean is not None
            and oracle_rebuf_s_clean is not None
            and oracle_smoothness_clean is not None
            and policy_reward_clean is not None
            and policy_quality_clean is not None
            and policy_rebuf_s_clean is not None
            and policy_smoothness_clean is not None
        ):
            oracle_reward_clean_values.append(float(oracle_reward_clean))
            oracle_quality_clean_values.append(float(oracle_quality_clean))
            oracle_rebuf_s_clean_values.append(float(oracle_rebuf_s_clean))
            oracle_smoothness_clean_values.append(float(oracle_smoothness_clean))
            oracle_gap_reward_clean = float(oracle_reward_clean) - float(policy_reward_clean)
            oracle_gap_quality_clean = float(oracle_quality_clean) - float(policy_quality_clean)
            oracle_gap_rebuf_s_clean = float(policy_rebuf_s_clean) - float(oracle_rebuf_s_clean)
            oracle_gap_smoothness_clean = float(policy_smoothness_clean) - float(oracle_smoothness_clean)
            oracle_gap_reward_clean_values.append(float(oracle_gap_reward_clean))
            oracle_gap_quality_clean_values.append(float(oracle_gap_quality_clean))
            oracle_gap_rebuf_s_clean_values.append(float(oracle_gap_rebuf_s_clean))
            oracle_gap_smoothness_clean_values.append(float(oracle_gap_smoothness_clean))

        oracle_gap_reward_adv = None
        oracle_gap_quality_adv = None
        oracle_gap_rebuf_s_adv = None
        oracle_gap_smoothness_adv = None
        if (
            oracle_reward_adv is not None
            and oracle_quality_adv is not None
            and oracle_rebuf_s_adv is not None
            and oracle_smoothness_adv is not None
            and policy_reward_adv is not None
            and policy_quality_adv is not None
            and policy_rebuf_s_adv is not None
            and policy_smoothness_adv is not None
        ):
            oracle_reward_adv_values.append(float(oracle_reward_adv))
            oracle_quality_adv_values.append(float(oracle_quality_adv))
            oracle_rebuf_s_adv_values.append(float(oracle_rebuf_s_adv))
            oracle_smoothness_adv_values.append(float(oracle_smoothness_adv))
            oracle_gap_reward_adv = float(oracle_reward_adv) - float(policy_reward_adv)
            oracle_gap_quality_adv = float(oracle_quality_adv) - float(policy_quality_adv)
            oracle_gap_rebuf_s_adv = float(policy_rebuf_s_adv) - float(oracle_rebuf_s_adv)
            oracle_gap_smoothness_adv = float(policy_smoothness_adv) - float(oracle_smoothness_adv)
            oracle_gap_reward_adv_values.append(float(oracle_gap_reward_adv))
            oracle_gap_quality_adv_values.append(float(oracle_gap_quality_adv))
            oracle_gap_rebuf_s_adv_values.append(float(oracle_gap_rebuf_s_adv))
            oracle_gap_smoothness_adv_values.append(float(oracle_gap_smoothness_adv))

        metrics: dict[str, float | int | str] = {
            "env/step": int(step),
            "env/episode": int(episode),
            "clean/trace": str(clean_trace_name),
            "adv/trace": str(adv_trace_name),
            "clean/trace_bw_mbps": float(trace_bw_clean),
            "adv/trace_bw_mbps": float(trace_bw_adv),
            "clean/action": int(br_clean),
            "adv/action": int(br_adv),
            "clean/action_next": int(next_br_clean),
            "adv/action_next": int(next_br_adv),
            "clean/bitrate_kbps": float(VIDEO_BIT_RATE[int(br_clean)]),
            "adv/bitrate_kbps": float(VIDEO_BIT_RATE[int(br_adv)]),
            "clean/video_chunk_size_bytes": float(chunk_size_c),
            "adv/video_chunk_size_bytes": float(chunk_size_a),
            "clean/delay_ms": float(delay_ms_c),
            "adv/delay_ms": float(delay_ms_a),
            "clean/sleep_time_ms": float(sleep_ms_c),
            "adv/sleep_time_ms": float(sleep_ms_a),
            "clean/buffer_s": float(buffer_s_c),
            "adv/buffer_s": float(buffer_s_a),
            "clean/rebuf_s": float(rebuf_s_c),
            "adv/rebuf_s": float(rebuf_s_a),
            "clean/realized_bw_mbps": float(_avg_bandwidth_mbps(float(chunk_size_c), float(delay_ms_c))),
            "adv/realized_bw_mbps": float(_avg_bandwidth_mbps(float(chunk_size_a), float(delay_ms_a))),
            "tracegen/adv/realized_bw_mbps": float(expected_adv_realized_bw_mbps),
            "tracegen/adv/action_exec": float(expected_adv_action_exec),
            "tracegen/adv/action_next": float(expected_adv_action_next),
            "tracegen/adv/delay_ms": float(expected_adv_delay_ms),
            "tracegen/adv/throughput_mbps": float(expected_adv_throughput_mbps),
            "tracegen/adv/video_chunk_size_bytes": float(expected_adv_video_chunk_size_bytes),
            "tracegen/pgd/iters_run": float(expected_adv_pgd_iters_run),
            "clean/throughput_mbps": float(_throughput_mbps(float(chunk_size_c), float(delay_ms_c))),
            "adv/throughput_mbps": float(_throughput_mbps(float(chunk_size_a), float(delay_ms_a))),
            "clean/qoe_reward": float(reward_c),
            "adv/qoe_reward": float(reward_a),
            "clean/qoe/quality": float(qoe_quality_clean),
            "adv/qoe/quality": float(qoe_quality_adv),
            "clean/qoe/rebuf_s": float(qoe_rebuf_s_clean),
            "adv/qoe/rebuf_s": float(qoe_rebuf_s_adv),
            "clean/qoe/smoothness": float(qoe_smoothness_clean),
            "adv/qoe/smoothness": float(qoe_smoothness_adv),
            "compare/qoe_reward_diff": float(reward_diff),
            "compare/action_flip": int(action_flip),
            "compare/kl_clean_adv": float(kl_clean_adv),
            "clean/entropy": float(entropy_clean),
            "adv/entropy": float(entropy_adv),
            "clean/max_prob": float(max_prob_clean),
            "adv/max_prob": float(max_prob_adv),
        }

        if oracle_gap_quality_clean is not None:
            metrics["clean/oracle/qoe_reward"] = float(oracle_reward_clean)
            metrics["clean/oracle/qoe/quality"] = float(oracle_quality_clean)
            metrics["clean/oracle/qoe/rebuf_s"] = float(oracle_rebuf_s_clean)
            metrics["clean/oracle/qoe/smoothness"] = float(oracle_smoothness_clean)
            metrics["clean/oracle_gap/qoe_reward"] = float(oracle_gap_reward_clean)
            metrics["clean/oracle_gap/qoe/quality"] = float(oracle_gap_quality_clean)
            metrics["clean/oracle_gap/qoe/rebuf_s"] = float(oracle_gap_rebuf_s_clean)
            metrics["clean/oracle_gap/qoe/smoothness"] = float(oracle_gap_smoothness_clean)

        if oracle_gap_quality_adv is not None:
            metrics["adv/oracle/qoe_reward"] = float(oracle_reward_adv)
            metrics["adv/oracle/qoe/quality"] = float(oracle_quality_adv)
            metrics["adv/oracle/qoe/rebuf_s"] = float(oracle_rebuf_s_adv)
            metrics["adv/oracle/qoe/smoothness"] = float(oracle_smoothness_adv)
            metrics["adv/oracle_gap/qoe_reward"] = float(oracle_gap_reward_adv)
            metrics["adv/oracle_gap/qoe/quality"] = float(oracle_gap_quality_adv)
            metrics["adv/oracle_gap/qoe/rebuf_s"] = float(oracle_gap_rebuf_s_adv)
            metrics["adv/oracle_gap/qoe/smoothness"] = float(oracle_gap_smoothness_adv)

        if wandb_run is not None:
            wandb.log(metrics, step=step)

        last_br_clean = int(br_clean)
        br_clean = int(next_br_clean)
        last_br_adv = int(br_adv)
        br_adv = int(next_br_adv)

        step += 1

        if bool(end_c) or bool(end_a) or (args.max_steps is not None and step >= args.max_steps):
            if bool(end_c) or bool(end_a):
                if wandb_run is not None:
                    wandb.log(
                        {
                            "clean/episode/cumulative_reward": float(episode_return_clean),
                            "adv/episode/cumulative_reward": float(episode_return_adv),
                            "compare/episode/cumulative_reward_diff": float(episode_return_adv - episode_return_clean),
                            "env/episode": int(episode),
                        },
                        step=step,
                    )
                episode_return_clean = 0.0
                episode_return_adv = 0.0
                state_clean[:] = 0.0
                state_adv[:] = 0.0
                last_br_clean = DEFAULT_QUALITY
                br_clean = DEFAULT_QUALITY
                last_br_adv = DEFAULT_QUALITY
                br_adv = DEFAULT_QUALITY
                episode += 1

            max_episodes = min(int(len(clean_file_names)), int(len(adv_file_names)))
            if (args.max_steps is not None and step >= args.max_steps) or episode >= max_episodes:
                break

    if wandb_run is not None:
        total = max(int(step), 1)
        wandb_run.summary["env/steps-sum"] = int(total)
        wandb_run.summary["compare/action_flip-rate"] = float(action_flip_sum) / float(total)
        _write_agg_stats(wandb_run.summary, "compare/kl_clean_adv", kl_clean_adv_values, include_geo_mean=True)
        _write_agg_stats(wandb_run.summary, "clean/qoe_reward", qoe_reward_clean_values, include_geo_mean=False)
        _write_agg_stats(wandb_run.summary, "adv/qoe_reward", qoe_reward_adv_values, include_geo_mean=False)
        _write_agg_stats(wandb_run.summary, "compare/qoe_reward_diff", qoe_reward_diff_values, include_geo_mean=False)
        _write_agg_stats(wandb_run.summary, "clean/qoe/quality", qoe_quality_clean_values, include_geo_mean=True)
        _write_agg_stats(wandb_run.summary, "adv/qoe/quality", qoe_quality_adv_values, include_geo_mean=True)
        _write_agg_stats(wandb_run.summary, "clean/qoe/rebuf_s", qoe_rebuf_s_clean_values, include_geo_mean=True)
        _write_agg_stats(wandb_run.summary, "adv/qoe/rebuf_s", qoe_rebuf_s_adv_values, include_geo_mean=True)
        _write_agg_stats(wandb_run.summary, "clean/qoe/smoothness", qoe_smoothness_clean_values, include_geo_mean=True)
        _write_agg_stats(wandb_run.summary, "adv/qoe/smoothness", qoe_smoothness_adv_values, include_geo_mean=True)
        _write_agg_stats(wandb_run.summary, "clean/entropy", entropy_clean_values, include_geo_mean=True)
        _write_agg_stats(wandb_run.summary, "adv/entropy", entropy_adv_values, include_geo_mean=True)
        _write_agg_stats(wandb_run.summary, "clean/max_prob", max_prob_clean_values, include_geo_mean=True)
        _write_agg_stats(wandb_run.summary, "adv/max_prob", max_prob_adv_values, include_geo_mean=True)
        _write_agg_stats(
            wandb_run.summary,
            "tracegen/adv/realized_bw_mbps",
            tracegen_expected_adv_realized_bw_mbps_values,
            include_geo_mean=True,
        )
        _write_agg_stats(
            wandb_run.summary,
            "tracegen/adv/action_exec",
            tracegen_expected_adv_action_exec_values,
            include_geo_mean=False,
        )
        _write_agg_stats(
            wandb_run.summary,
            "tracegen/adv/action_next",
            tracegen_expected_adv_action_next_values,
            include_geo_mean=False,
        )
        _write_agg_stats(
            wandb_run.summary,
            "tracegen/adv/delay_ms",
            tracegen_expected_adv_delay_ms_values,
            include_geo_mean=True,
        )
        _write_agg_stats(
            wandb_run.summary,
            "tracegen/adv/throughput_mbps",
            tracegen_expected_adv_throughput_mbps_values,
            include_geo_mean=True,
        )
        _write_agg_stats(
            wandb_run.summary,
            "tracegen/adv/video_chunk_size_bytes",
            tracegen_expected_adv_video_chunk_size_bytes_values,
            include_geo_mean=True,
        )
        _write_agg_stats(
            wandb_run.summary,
            "tracegen/pgd/iters_run",
            tracegen_expected_adv_pgd_iters_run_values,
            include_geo_mean=False,
        )
        _write_agg_stats(wandb_run.summary, "clean/oracle/qoe_reward", oracle_reward_clean_values, include_geo_mean=False)
        _write_agg_stats(wandb_run.summary, "adv/oracle/qoe_reward", oracle_reward_adv_values, include_geo_mean=False)
        _write_agg_stats(wandb_run.summary, "clean/oracle/qoe/quality", oracle_quality_clean_values, include_geo_mean=True)
        _write_agg_stats(wandb_run.summary, "adv/oracle/qoe/quality", oracle_quality_adv_values, include_geo_mean=True)
        _write_agg_stats(wandb_run.summary, "clean/oracle/qoe/rebuf_s", oracle_rebuf_s_clean_values, include_geo_mean=True)
        _write_agg_stats(wandb_run.summary, "adv/oracle/qoe/rebuf_s", oracle_rebuf_s_adv_values, include_geo_mean=True)
        _write_agg_stats(
            wandb_run.summary, "clean/oracle/qoe/smoothness", oracle_smoothness_clean_values, include_geo_mean=True
        )
        _write_agg_stats(wandb_run.summary, "adv/oracle/qoe/smoothness", oracle_smoothness_adv_values, include_geo_mean=True)
        _write_agg_stats(
            wandb_run.summary,
            "clean/oracle_gap/qoe_reward",
            oracle_gap_reward_clean_values,
            include_geo_mean=False,
        )
        _write_agg_stats(
            wandb_run.summary,
            "adv/oracle_gap/qoe_reward",
            oracle_gap_reward_adv_values,
            include_geo_mean=False,
        )
        _write_agg_stats(
            wandb_run.summary,
            "clean/oracle_gap/qoe/quality",
            oracle_gap_quality_clean_values,
            include_geo_mean=False,
        )
        _write_agg_stats(
            wandb_run.summary,
            "adv/oracle_gap/qoe/quality",
            oracle_gap_quality_adv_values,
            include_geo_mean=False,
        )
        _write_agg_stats(
            wandb_run.summary,
            "clean/oracle_gap/qoe/rebuf_s",
            oracle_gap_rebuf_s_clean_values,
            include_geo_mean=False,
        )
        _write_agg_stats(
            wandb_run.summary,
            "adv/oracle_gap/qoe/rebuf_s",
            oracle_gap_rebuf_s_adv_values,
            include_geo_mean=False,
        )
        _write_agg_stats(
            wandb_run.summary,
            "clean/oracle_gap/qoe/smoothness",
            oracle_gap_smoothness_clean_values,
            include_geo_mean=False,
        )
        _write_agg_stats(
            wandb_run.summary,
            "adv/oracle_gap/qoe/smoothness",
            oracle_gap_smoothness_adv_values,
            include_geo_mean=False,
        )


def _looks_like_cooked_trace_file(path: str) -> bool:
    path = str(path)
    if not os.path.isfile(path):
        return False
    try:
        with open(path, "rb") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    return False
                try:
                    float(parts[0])
                    float(parts[1])
                except ValueError:
                    return False
                return True
    except OSError:
        return False
    return False


def _load_cooked_traces_dir_filtered(cooked_trace_dir: str) -> tuple[list, list, list[str]]:
    #* Trace dirs can include BO logs / manifests; accept only files whose first non-empty line parses as two floats.
    cooked_trace_dir = str(cooked_trace_dir)
    cooked_files = []
    for name in sorted(os.listdir(cooked_trace_dir)):
        if name.startswith("."):
            continue
        path = os.path.join(cooked_trace_dir, name)
        if not _looks_like_cooked_trace_file(path):
            continue
        cooked_files.append(str(name))

    all_cooked_time = []
    all_cooked_bw = []
    all_file_names = []
    for cooked_file in cooked_files:
        file_path = os.path.join(cooked_trace_dir, cooked_file)
        cooked_time = []
        cooked_bw = []
        with open(file_path, "rb") as f:
            for line in f:
                parse = line.split()
                cooked_time.append(float(parse[0]))
                cooked_bw.append(float(parse[1]))
        all_cooked_time.append(cooked_time)
        all_cooked_bw.append(cooked_bw)
        all_file_names.append(cooked_file)

    return all_cooked_time, all_cooked_bw, all_file_names


def _count_cooked_trace_files(cooked_trace_dir: str) -> int:
    #* Count cooked trace files using the same content-based filtering as `_load_cooked_traces_dir_filtered`.
    cooked_trace_dir = str(cooked_trace_dir)
    count = 0
    for name in sorted(os.listdir(cooked_trace_dir)):
        if name.startswith("."):
            continue
        path = os.path.join(cooked_trace_dir, name)
        if not _looks_like_cooked_trace_file(path):
            continue
        count += 1
    return int(count)


_SHIELD_RULES_CACHE: dict[tuple[str, float, str], list[tuple[list[object], int]]] = {}


class _ShieldVar:
    def __init__(self, name: str):
        self.name = str(name)


class _ShieldCmp:
    def __init__(self, op: str, left: object, right: object):
        self.op = str(op)
        self.left = left
        self.right = right


class _ShieldImplies:
    def __init__(self, antecedent: object, consequent: object):
        self.antecedent = antecedent
        self.consequent = consequent


class _ShieldOr:
    def __init__(self, args: tuple[object, ...]):
        self.args = tuple(args)


def _shield_flip_op(op: str) -> str:
    if op == "<":
        return ">"
    if op == "<=":
        return ">="
    if op == ">":
        return "<"
    if op == ">=":
        return "<="
    return str(op)


def _shield_flatten_and(expr: object) -> list[object]:
    if isinstance(expr, list):
        out: list[object] = []
        for item in expr:
            out.extend(_shield_flatten_and(item))
        return out
    return [expr]


def _shield_cmp_to_atom(expr: object) -> tuple[str, str, float] | None:
    if not isinstance(expr, _ShieldCmp):
        return None
    left = expr.left
    right = expr.right
    if isinstance(left, _ShieldVar) and isinstance(right, (int, float)):
        return str(left.name), str(expr.op), float(right)
    if isinstance(right, _ShieldVar) and isinstance(left, (int, float)):
        return str(right.name), _shield_flip_op(str(expr.op)), float(left)
    return None


def _shield_parse_indexed(var_name: str, prefix: str) -> int | None:
    if not str(var_name).startswith(str(prefix)):
        return None
    suffix = str(var_name)[len(str(prefix)) :]
    return int(suffix) if suffix.isdigit() else None


def _shield_cmp(a: float, op: str, b: float) -> bool:
    if not (math.isfinite(float(a)) and math.isfinite(float(b))):
        return False
    if op == "<":
        return float(a) < float(b)
    if op == "<=":
        return float(a) <= float(b)
    if op == ">":
        return float(a) > float(b)
    if op == ">=":
        return float(a) >= float(b)
    if op == "==":
        return float(a) == float(b)
    if op == "!=":
        return float(a) != float(b)
    return False


def _load_shield_rules(rules_file: str, *, output_var: str) -> list[tuple[list[object], int]]:
    path = os.path.abspath(str(rules_file))
    mtime = 0.0
    try:
        mtime = float(os.path.getmtime(path))
    except Exception:
        mtime = 0.0

    cache_key = (path, mtime, str(output_var))
    if cache_key in _SHIELD_RULES_CACHE:
        return _SHIELD_RULES_CACHE[cache_key]

    variable_cache: dict[str, _ShieldVar] = {}

    def Symbol(name: str, *args, **kwargs):
        if str(name) not in variable_cache:
            variable_cache[str(name)] = _ShieldVar(str(name))
        return variable_cache[str(name)]

    def Integer(v):
        return int(v)

    def Float(v, precision=None):
        return float(v)

    def And(*args):
        return list(args)

    def Or(*args):
        return _ShieldOr(tuple(args))

    def Implies(a, b):
        return _ShieldImplies(a, b)

    def Equality(a, b):
        return _ShieldCmp("==", a, b)

    def Unequality(a, b):
        return _ShieldCmp("!=", a, b)

    def StrictLessThan(a, b):
        return _ShieldCmp("<", a, b)

    def StrictGreaterThan(a, b):
        return _ShieldCmp(">", a, b)

    def LessThan(a, b):
        return _ShieldCmp("<=", a, b)

    def GreaterThan(a, b):
        return _ShieldCmp(">=", a, b)

    eval_env = {
        "__builtins__": {},
        "Symbol": Symbol,
        "Integer": Integer,
        "Float": Float,
        "And": And,
        "Or": Or,
        "Implies": Implies,
        "Equality": Equality,
        "Unequality": Unequality,
        "StrictLessThan": StrictLessThan,
        "StrictGreaterThan": StrictGreaterThan,
        "LessThan": LessThan,
        "GreaterThan": GreaterThan,
    }

    rules: list[tuple[list[object], int]] = []
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            expr = eval(line, eval_env)  # nosec - trusted local rules file
            if not isinstance(expr, _ShieldImplies):
                continue

            consequent_atom = _shield_cmp_to_atom(expr.consequent)
            if consequent_atom is None:
                continue
            cons_var, cons_op, cons_val = consequent_atom
            if str(cons_var) != str(output_var) or str(cons_op) != "==":
                continue

            try:
                out_kbps = int(cons_val)
            except Exception:
                continue

            atoms: list[object] = []
            ok = True
            for atom_expr in _shield_flatten_and(expr.antecedent):
                if isinstance(atom_expr, _ShieldOr):
                    or_atoms: list[tuple[str, str, float]] = []
                    for sub in atom_expr.args:
                        atom = _shield_cmp_to_atom(sub)
                        if atom is None:
                            ok = False
                            break
                        v, op, val = atom
                        if str(op) not in ("<", "<=", ">", ">=", "==", "!="):
                            ok = False
                            break
                        or_atoms.append((str(v), str(op), float(val)))
                    if not ok or len(or_atoms) == 0:
                        ok = False
                        break
                    atoms.append(("or", or_atoms))
                    continue

                atom = _shield_cmp_to_atom(atom_expr)
                if atom is None:
                    ok = False
                    break
                v, op, val = atom
                if str(op) not in ("<", "<=", ">", ">=", "==", "!="):
                    ok = False
                    break
                atoms.append((str(v), str(op), float(val)))
            if not ok:
                continue

            rules.append((atoms, int(out_kbps)))

    _SHIELD_RULES_CACHE[cache_key] = rules
    return rules


def _bitrate_kbps_to_action_idx(bitrate_kbps: float) -> int:
    bitrate_values = np.array(VIDEO_BIT_RATE, dtype=np.float64)
    if bitrate_values.size == 0:
        return 0
    target = float(bitrate_kbps)
    if not np.isfinite(target):
        return 0
    return int(np.argmin(np.abs(bitrate_values - float(target))))


def _policy_action_idx(
    *,
    p: np.ndarray,
    action_selection: str,
    rng: np.random.RandomState,
    lo: int = 0,
    hi: int | None = None,
    noise: np.ndarray | None = None,
) -> tuple[int, np.ndarray | None]:
    upper = int(A_DIM) - 1 if hi is None else int(hi)
    lower = int(max(0, min(int(A_DIM) - 1, int(lo))))
    upper = int(max(0, min(int(A_DIM) - 1, int(upper))))
    if lower > upper:
        lower, upper = 0, int(A_DIM) - 1

    probs = np.asarray(p, dtype=np.float64).reshape(-1)
    if str(action_selection) == "argmax":
        chosen = int(lower + int(np.argmax(np.array(probs[lower : upper + 1], copy=False))))
        return int(chosen), noise

    if str(action_selection) == "gumbel":
        gumbel_noise = noise
        if gumbel_noise is None:
            gumbel_noise = rng.gumbel(size=probs.shape)
        logits = np.log(probs + 1e-12) + np.asarray(gumbel_noise, dtype=np.float64)
        masked = np.full_like(logits, -np.inf, dtype=np.float64)
        masked[lower : upper + 1] = logits[lower : upper + 1]
        if not np.any(np.isfinite(masked[lower : upper + 1])):
            return int(lower), np.asarray(gumbel_noise, dtype=np.float64)
        return int(np.argmax(masked)), np.asarray(gumbel_noise, dtype=np.float64)

    raise ValueError(f"Unknown action_selection: {action_selection}")


def _shield_rule_satisfied(
    atoms: list[object],
    *,
    values: dict[str, float],
    delay_ms_hist: list[float],
    throughput_mbps_hist: list[float],
) -> bool:
    for atom in atoms:
        if isinstance(atom, tuple) and len(atom) == 2 and str(atom[0]) == "or":
            ok_any = False
            for v, op, val in list(atom[1]):
                if _shield_atom_satisfied(
                    str(v),
                    str(op),
                    float(val),
                    values=values,
                    delay_ms_hist=delay_ms_hist,
                    throughput_mbps_hist=throughput_mbps_hist,
                ):
                    ok_any = True
                    break
            if not ok_any:
                return False
            continue

        if not (isinstance(atom, tuple) and len(atom) == 3):
            return False
        v, op, val = atom
        if not _shield_atom_satisfied(
            str(v),
            str(op),
            float(val),
            values=values,
            delay_ms_hist=delay_ms_hist,
            throughput_mbps_hist=throughput_mbps_hist,
        ):
            return False
    return True


def _shield_atom_satisfied(
    var: str,
    op: str,
    val: float,
    *,
    values: dict[str, float],
    delay_ms_hist: list[float],
    throughput_mbps_hist: list[float],
) -> bool:
    if str(var) in values:
        return _shield_cmp(float(values[str(var)]), str(op), float(val))

    idx_delay = _shield_parse_indexed(str(var), "DelayMs")
    if idx_delay is not None:
        if 0 <= int(idx_delay) < len(delay_ms_hist):
            return _shield_cmp(float(delay_ms_hist[int(idx_delay)]), str(op), float(val))
        return False

    idx_thr = _shield_parse_indexed(str(var), "ThroughputMbps")
    if idx_thr is not None:
        if 0 <= int(idx_thr) < len(throughput_mbps_hist):
            return _shield_cmp(float(throughput_mbps_hist[int(idx_thr)]), str(op), float(val))
        return False

    return False


class ShieldBounds:
    def __init__(
        self,
        *,
        min_rules: list[tuple[list[object], int]],
        max_rules: list[tuple[list[object], int]],
        risk_trigger_steps: int = 1,
        intervention_cooldown_steps: int = 0,
    ):
        self._min_rules = list(min_rules)
        self._max_rules = list(max_rules)
        self._risk_trigger_steps = int(max(1, int(risk_trigger_steps)))
        self._intervention_cooldown_steps = int(max(0, int(intervention_cooldown_steps)))
        self.reset_runtime_state()

    def reset_runtime_state(self) -> None:
        self._risk_streak = 0
        self._cooldown_remaining = 0

    def snapshot_runtime_state(self) -> dict[str, int]:
        return {
            "risk_streak": int(self._risk_streak),
            "cooldown_remaining": int(self._cooldown_remaining),
        }

    def restore_runtime_state(self, snapshot: dict[str, int] | None) -> None:
        if snapshot is None:
            self.reset_runtime_state()
            return
        self._risk_streak = int(max(0, int(snapshot.get("risk_streak", 0))))
        self._cooldown_remaining = int(max(0, int(snapshot.get("cooldown_remaining", 0))))

    def bounds_from_state(
        self,
        *,
        state: np.ndarray,
        last_bitrate_idx: int,
        video_chunk_size_bytes: int,
    ) -> tuple[int, int, dict[str, float | int]]:
        values, delay_ms_hist, throughput_mbps_hist, _chosen_chunk_bytes_hist = _policy_rule_inputs_from_state(
            state,
            last_bitrate_idx=int(last_bitrate_idx),
            video_chunk_size_bytes=int(video_chunk_size_bytes),
        )

        min_kbps = float(min(VIDEO_BIT_RATE)) if VIDEO_BIT_RATE else 0.0
        max_kbps = float(max(VIDEO_BIT_RATE)) if VIDEO_BIT_RATE else 0.0
        matched_min = 0
        matched_max = 0

        for atoms, out_kbps in self._min_rules:
            if _shield_rule_satisfied(
                atoms,
                values=values,
                delay_ms_hist=delay_ms_hist,
                throughput_mbps_hist=throughput_mbps_hist,
            ):
                matched_min += 1
                min_kbps = max(float(min_kbps), float(out_kbps))

        for atoms, out_kbps in self._max_rules:
            if _shield_rule_satisfied(
                atoms,
                values=values,
                delay_ms_hist=delay_ms_hist,
                throughput_mbps_hist=throughput_mbps_hist,
            ):
                matched_max += 1
                max_kbps = min(float(max_kbps), float(out_kbps))

        min_idx = _bitrate_kbps_to_action_idx(float(min_kbps))
        max_idx = _bitrate_kbps_to_action_idx(float(max_kbps))

        inconsistent = int(min_idx > max_idx)
        if inconsistent:
            min_idx = 0
            max_idx = int(A_DIM) - 1

        stats: dict[str, float | int] = {
            "shield/matched_min": int(matched_min),
            "shield/matched_max": int(matched_max),
            "shield/min_kbps": float(min_kbps),
            "shield/max_kbps": float(max_kbps),
            "shield/min_idx": int(min_idx),
            "shield/max_idx": int(max_idx),
            "shield/inconsistent": int(inconsistent),
        }
        return int(min_idx), int(max_idx), stats

    def select_action(
        self,
        *,
        p: np.ndarray,
        state: np.ndarray,
        last_bitrate_idx: int,
        video_chunk_size_bytes: int,
        action_selection: str,
        rng: np.random.RandomState,
    ) -> tuple[int, dict[str, float | int]]:
        mi, ma, stats = self.bounds_from_state(
            state=state,
            last_bitrate_idx=int(last_bitrate_idx),
            video_chunk_size_bytes=int(video_chunk_size_bytes),
        )
        lo = int(max(0, min(int(A_DIM) - 1, int(mi))))
        hi = int(max(0, min(int(A_DIM) - 1, int(ma))))
        if lo > hi:
            lo, hi = 0, int(A_DIM) - 1

        base_action_idx, sampled_noise = _policy_action_idx(
            p=p,
            action_selection=str(action_selection),
            rng=rng,
        )
        risky = bool(int(base_action_idx) < int(lo) or int(base_action_idx) > int(hi))
        if bool(risky):
            self._risk_streak = int(self._risk_streak) + 1
        else:
            self._risk_streak = 0

        cooldown_active = bool(int(self._cooldown_remaining) > 0)
        intervene = bool(risky and int(self._risk_streak) >= int(self._risk_trigger_steps) and not cooldown_active)
        if bool(intervene):
            chosen, _ = _policy_action_idx(
                p=p,
                action_selection=str(action_selection),
                rng=rng,
                lo=int(lo),
                hi=int(hi),
                noise=sampled_noise,
            )
            self._cooldown_remaining = int(self._intervention_cooldown_steps)
        else:
            chosen = int(base_action_idx)
            if cooldown_active:
                self._cooldown_remaining = int(max(0, int(self._cooldown_remaining) - 1))

        stats["shield/base_action_idx"] = int(base_action_idx)
        stats["shield/risky_state"] = int(bool(risky))
        stats["shield/risk_streak"] = int(self._risk_streak)
        stats["shield/intervene"] = int(bool(intervene))
        stats["shield/cooldown_active"] = int(bool(cooldown_active))
        stats["shield/cooldown_remaining"] = int(self._cooldown_remaining)
        stats["shield/risk_trigger_steps"] = int(self._risk_trigger_steps)
        stats["shield/intervention_cooldown_steps"] = int(self._intervention_cooldown_steps)
        stats["shield/selected_action_idx"] = int(chosen)
        return int(chosen), stats


def _eval_collect_qoe_values(
    actor: network.Network,
    *,
    trace_dir: str,
    trace_num: int,
    max_steps: int | None,
    action_selection: str,
    seed: int,
) -> dict[str, object]:
    trace_dir = _ensure_trailing_sep(_resolve_under_src(trace_dir))
    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(trace_dir)
    if int(trace_num) > 0:
        all_cooked_time, all_cooked_bw, all_file_names = _sample_traces(
            all_cooked_time, all_cooked_bw, all_file_names, int(trace_num), seed=42
        )

    return _eval_collect_qoe_values_from_traces(
        actor,
        all_cooked_time=all_cooked_time,
        all_cooked_bw=all_cooked_bw,
        all_file_names=all_file_names,
        max_steps=max_steps,
        action_selection=action_selection,
        seed=seed,
        shield=None,
    )


def _eval_collect_qoe_values_from_traces(
    actor: network.Network,
    *,
    all_cooked_time: list,
    all_cooked_bw: list,
    all_file_names: list[str],
    max_steps: int | None,
    action_selection: str,
    seed: int,
    shield: ShieldBounds | None,
) -> dict[str, object]:
    #* Minimal, side-effect-free evaluation loop: returns metric value lists for summary logging.
    #* Includes K-step oracle-gap diagnostics (oracle best vs policy rollout) for robustify/shield modes.
    net_env = env.Environment(all_cooked_time=all_cooked_time, all_cooked_bw=all_cooked_bw)
    rng = np.random.RandomState(int(seed))

    state = np.zeros((S_INFO, S_LEN), dtype=np.float64)
    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY
    step = 0
    video_count = 0
    time_stamp_ms = 0.0

    shield_min_idx_values: list[float] = []
    shield_max_idx_values: list[float] = []
    shield_inconsistent_values: list[float] = []
    shield_matched_min_values: list[float] = []
    shield_matched_max_values: list[float] = []

    env_time_s_values: list[float] = []
    env_episode_values: list[int] = []
    env_trace_values: list[str] = []
    trace_bw_mbps_values: list[float] = []
    realized_bw_mbps_values: list[float] = []
    throughput_mbps_values: list[float] = []
    action_values: list[int] = []
    bitrate_kbps_values: list[float] = []
    buffer_s_values: list[float] = []
    qoe_reward_values: list[float] = []
    qoe_quality_values: list[float] = []
    qoe_rebuf_s_values: list[float] = []
    qoe_smoothness_values: list[float] = []
    entropy_values: list[float] = []
    max_prob_values: list[float] = []

    #* K-step oracle-gap diagnostics (averaged over K_steps).
    oracle_reward_values: list[float] = []
    oracle_quality_values: list[float] = []
    oracle_rebuf_s_values: list[float] = []
    oracle_smoothness_values: list[float] = []

    oracle_gap_reward_values: list[float] = []
    oracle_gap_quality_values: list[float] = []
    oracle_gap_rebuf_s_values: list[float] = []
    oracle_gap_smoothness_values: list[float] = []

    oracle_action_values: list[int] = []
    oracle_bitrate_kbps_values: list[float] = []
    oracle_buffer_s_values: list[float] = []
    oracle_throughput_mbps_values: list[float] = []

    oracle_k_steps = int(K_steps)

    def _oracle_k_step_qoe(
        env_snapshot: env.Environment, last_br: int, k_steps: int
    ) -> tuple[float | None, float | None, float | None, float | None, int | None]:
        k = int(k_steps)
        if k <= 0:
            return None, None, None, None, None

        best_r, best_seq, _, _ = find_optimal_k_step_reward_oracle(env_snapshot, int(last_br), int(k))
        if best_r is None or best_seq is None:
            return None, None, None, None, None

        first_action = int(best_seq[0]) if len(best_seq) > 0 else None

        env_sim = copy_env(env_snapshot, RANDOM_SEED=int(seed))
        total_quality = 0.0
        total_rebuf_s = 0.0
        total_smoothness = 0.0
        last = int(last_br)
        for idx, action in enumerate(best_seq):
            a = int(action)
            try:
                _, _, _, rebuf_s, _, _, end, _ = env_sim.get_video_chunk(a)
            except Exception:
                return None, None, None, None, None

            if bool(end) and idx < int(k) - 1:
                return None, None, None, None, None

            total_quality += float(VIDEO_BIT_RATE[a]) / float(M_IN_K)
            total_rebuf_s += float(rebuf_s)
            total_smoothness += float(abs(VIDEO_BIT_RATE[a] - VIDEO_BIT_RATE[int(last)])) / float(M_IN_K)
            last = int(a)

        denom = float(k)
        return (
            float(best_r) / denom,
            float(total_quality) / denom,
            float(total_rebuf_s) / denom,
            float(total_smoothness) / denom,
            first_action,
        )

    def _policy_k_step_qoe(
        env_snapshot: env.Environment,
        state_snapshot: np.ndarray,
        *,
        last_br: int,
        start_action: int,
        k_steps: int,
        rng: np.random.RandomState,
    ) -> tuple[float | None, float | None, float | None, float | None]:
        k = int(k_steps)
        if k <= 0:
            return None, None, None, None

        env_sim = copy_env(env_snapshot, RANDOM_SEED=int(seed))
        state_sim = np.array(state_snapshot, copy=True)

        total_reward = 0.0
        total_quality = 0.0
        total_rebuf_s = 0.0
        total_smoothness = 0.0
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

                if bool(end) and i < int(k) - 1:
                    return None, None, None, None

                total_reward += float(calculate_reward(int(a), int(last), float(rebuf_s)))
                total_quality += float(VIDEO_BIT_RATE[int(a)]) / float(M_IN_K)
                total_rebuf_s += float(rebuf_s)
                total_smoothness += float(abs(VIDEO_BIT_RATE[int(a)] - VIDEO_BIT_RATE[int(last)])) / float(M_IN_K)

                state_sim = _update_state(
                    state_sim,
                    bit_rate=int(a),
                    buffer_s=float(buffer_s),
                    video_chunk_size=int(chunk_size),
                    delay_ms=float(delay_ms),
                    next_video_chunk_sizes=[int(x) for x in next_sizes],
                    video_chunk_remain=int(remain),
                )
                last = int(a)

                if i < int(k) - 1:
                    p = actor.predict(np.reshape(state_sim, (1, S_INFO, S_LEN))).astype(np.float64).reshape(-1)
                    if shield is None:
                        if action_selection == "argmax":
                            a = int(np.argmax(p))
                        elif action_selection == "gumbel":
                            noise = rng.gumbel(size=p.shape)
                            a = int(np.argmax(np.log(p) + noise))
                        else:
                            raise ValueError(f"Unknown action_selection: {action_selection}")
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

    while True:
        #* Capture trace name before `get_video_chunk` advances to the next trace on end-of-video.
        trace_name = str(all_file_names[int(net_env.trace_idx)]) if all_file_names else "unknown"

        #* Snapshot trace bandwidth at the current Mahimahi pointer (Mbps) before downloading this chunk.
        try:
            trace_bw_mbps = float(net_env.cooked_bw[net_env.mahimahi_ptr])
        except Exception:
            trace_bw_mbps = float("nan")

        policy_action_exec = int(bit_rate)
        policy_bitrate_kbps = float(VIDEO_BIT_RATE[int(policy_action_exec)])

        oracle_reward, oracle_quality, oracle_rebuf_s, oracle_smoothness, oracle_action_exec = _oracle_k_step_qoe(
            net_env, int(last_bit_rate), oracle_k_steps
        )
        oracle_reward_values.append(float(oracle_reward) if oracle_reward is not None else float("nan"))
        oracle_quality_values.append(float(oracle_quality) if oracle_quality is not None else float("nan"))
        oracle_rebuf_s_values.append(float(oracle_rebuf_s) if oracle_rebuf_s is not None else float("nan"))
        oracle_smoothness_values.append(float(oracle_smoothness) if oracle_smoothness is not None else float("nan"))

        oracle_buffer_s = float("nan")
        oracle_throughput_mbps = float("nan")
        if oracle_action_exec is not None:
            oracle_action_values.append(int(oracle_action_exec))
            oracle_bitrate_kbps_values.append(float(VIDEO_BIT_RATE[int(oracle_action_exec)]))
            try:
                env_oracle_step = copy_env(net_env, RANDOM_SEED=int(seed))
                delay_ms_o, _sleep_ms_o, buffer_s_o, _rebuf_s_o, chunk_size_o, _next_sizes_o, _end_o, _remain_o = (
                    env_oracle_step.get_video_chunk(int(oracle_action_exec))
                )
                oracle_buffer_s = float(buffer_s_o)
                oracle_throughput_mbps = float(_throughput_mbps(float(chunk_size_o), float(delay_ms_o)))
            except Exception:
                oracle_buffer_s = float("nan")
                oracle_throughput_mbps = float("nan")
            oracle_buffer_s_values.append(float(oracle_buffer_s))
            oracle_throughput_mbps_values.append(float(oracle_throughput_mbps))
        else:
            oracle_action_values.append(-1)
            oracle_bitrate_kbps_values.append(float("nan"))
            oracle_buffer_s_values.append(float("nan"))
            oracle_throughput_mbps_values.append(float("nan"))

        policy_reward = None
        policy_quality = None
        policy_rebuf_s = None
        policy_smoothness = None
        try:
            rng_state = rng.get_state()
            rng_roll = np.random.RandomState()
            rng_roll.set_state(rng_state)
            policy_reward, policy_quality, policy_rebuf_s, policy_smoothness = _policy_k_step_qoe(
                net_env,
                state,
                last_br=int(last_bit_rate),
                start_action=int(bit_rate),
                k_steps=oracle_k_steps,
                rng=rng_roll,
            )
        except Exception:
            policy_reward = None
            policy_quality = None
            policy_rebuf_s = None
            policy_smoothness = None

        oracle_gap_reward = float("nan")
        oracle_gap_quality = float("nan")
        oracle_gap_rebuf_s = float("nan")
        oracle_gap_smoothness = float("nan")
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

        oracle_gap_reward_values.append(float(oracle_gap_reward))
        oracle_gap_quality_values.append(float(oracle_gap_quality))
        oracle_gap_rebuf_s_values.append(float(oracle_gap_rebuf_s))
        oracle_gap_smoothness_values.append(float(oracle_gap_smoothness))

        (
            delay_ms,
            sleep_time_ms,
            buffer_s,
            rebuf_s,
            video_chunk_size,
            next_video_chunk_sizes,
            end_of_video,
            video_chunk_remain,
        ) = net_env.get_video_chunk(int(bit_rate))
        time_stamp_ms += float(delay_ms) + float(sleep_time_ms)

        trace_bw_mbps_values.append(float(trace_bw_mbps))
        realized_bw_mbps_values.append(float(_avg_bandwidth_mbps(float(video_chunk_size), float(delay_ms))))
        throughput_mbps_values.append(float(_throughput_mbps(float(video_chunk_size), float(delay_ms))))
        action_values.append(int(policy_action_exec))
        bitrate_kbps_values.append(float(policy_bitrate_kbps))
        buffer_s_values.append(float(buffer_s))

        qoe_reward = float(calculate_reward(int(bit_rate), int(last_bit_rate), float(rebuf_s)))
        qoe_quality = float(VIDEO_BIT_RATE[int(bit_rate)]) / float(M_IN_K)
        smoothness_kbps = float(abs(VIDEO_BIT_RATE[int(bit_rate)] - VIDEO_BIT_RATE[int(last_bit_rate)]))
        qoe_smoothness = float(smoothness_kbps) / float(M_IN_K)

        env_time_s_values.append(float(time_stamp_ms / 1000.0))
        env_episode_values.append(int(video_count))
        env_trace_values.append(str(trace_name))
        qoe_reward_values.append(float(qoe_reward))
        qoe_quality_values.append(float(qoe_quality))
        qoe_rebuf_s_values.append(float(rebuf_s))
        qoe_smoothness_values.append(float(qoe_smoothness))

        state = _update_state(
            state,
            bit_rate=int(bit_rate),
            buffer_s=float(buffer_s),
            video_chunk_size=int(video_chunk_size),
            delay_ms=float(delay_ms),
            next_video_chunk_sizes=[int(x) for x in next_video_chunk_sizes],
            video_chunk_remain=int(video_chunk_remain),
        )

        p_obs = actor.predict(np.reshape(state, (1, S_INFO, S_LEN))).astype(np.float64).reshape(-1)
        entropy_values.append(float(_entropy(p_obs)))
        max_prob_values.append(float(np.max(p_obs)))

        if shield is None:
            if action_selection == "argmax":
                next_bit_rate = int(np.argmax(p_obs))
            elif action_selection == "gumbel":
                noise = rng.gumbel(size=len(p_obs))
                next_bit_rate = int(np.argmax(np.log(p_obs) + noise))
            else:
                raise ValueError(f"Unknown action_selection: {action_selection}")
            shield_min_idx_values.append(float("nan"))
            shield_max_idx_values.append(float("nan"))
            shield_inconsistent_values.append(float("nan"))
            shield_matched_min_values.append(float("nan"))
            shield_matched_max_values.append(float("nan"))
        else:
            next_bit_rate, stats = shield.select_action(
                p=p_obs,
                state=state,
                last_bitrate_idx=int(last_bit_rate),
                video_chunk_size_bytes=int(video_chunk_size),
                action_selection=str(action_selection),
                rng=rng,
            )
            shield_min_idx_values.append(float(stats.get("shield/min_idx", float("nan"))))
            shield_max_idx_values.append(float(stats.get("shield/max_idx", float("nan"))))
            shield_inconsistent_values.append(float(stats.get("shield/inconsistent", float("nan"))))
            shield_matched_min_values.append(float(stats.get("shield/matched_min", float("nan"))))
            shield_matched_max_values.append(float(stats.get("shield/matched_max", float("nan"))))

        last_bit_rate = int(bit_rate)
        bit_rate = int(next_bit_rate)
        step += 1

        if end_of_video or (max_steps is not None and step >= int(max_steps)):
            video_count += int(bool(end_of_video))
            if bool(end_of_video):
                state[:] = 0.0
                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY
                if shield is not None and hasattr(shield, "reset_runtime_state"):
                    shield.reset_runtime_state()

            if (max_steps is not None and step >= int(max_steps)) or video_count >= len(all_file_names):
                break

    return {
        "env/steps": int(step),
        "env/episodes": int(video_count),
        "env/time_s_values": env_time_s_values,
        "env/episode_values": env_episode_values,
        "env/trace_values": env_trace_values,
        "trace_bw_mbps_values": trace_bw_mbps_values,
        "realized_bw_mbps_values": realized_bw_mbps_values,
        "throughput_mbps_values": throughput_mbps_values,
        "action_values": action_values,
        "bitrate_kbps_values": bitrate_kbps_values,
        "buffer_s_values": buffer_s_values,
        "qoe_reward_values": qoe_reward_values,
        "qoe_quality_values": qoe_quality_values,
        "qoe_rebuf_s_values": qoe_rebuf_s_values,
        "qoe_smoothness_values": qoe_smoothness_values,
        "entropy_values": entropy_values,
        "max_prob_values": max_prob_values,
        "oracle_reward_values": oracle_reward_values,
        "oracle_quality_values": oracle_quality_values,
        "oracle_rebuf_s_values": oracle_rebuf_s_values,
        "oracle_smoothness_values": oracle_smoothness_values,
        "oracle_gap_reward_values": oracle_gap_reward_values,
        "oracle_gap_quality_values": oracle_gap_quality_values,
        "oracle_gap_rebuf_s_values": oracle_gap_rebuf_s_values,
        "oracle_gap_smoothness_values": oracle_gap_smoothness_values,
        "oracle_action_values": oracle_action_values,
        "oracle_bitrate_kbps_values": oracle_bitrate_kbps_values,
        "oracle_buffer_s_values": oracle_buffer_s_values,
        "oracle_throughput_mbps_values": oracle_throughput_mbps_values,
        "shield_min_idx_values": shield_min_idx_values,
        "shield_max_idx_values": shield_max_idx_values,
        "shield_inconsistent_values": shield_inconsistent_values,
        "shield_matched_min_values": shield_matched_min_values,
        "shield_matched_max_values": shield_matched_max_values,
    }


def _log_eval_time_series(
    wandb_run,
    *,
    prefix: str,
    values: dict[str, object],
) -> None:
    #* Log per-step evaluation values so W&B can render line charts (not just summary bars).
    p = str(prefix).rstrip("/")

    qoe_reward_values = list(values.get("qoe_reward_values", []))
    qoe_quality_values = list(values.get("qoe_quality_values", []))
    qoe_rebuf_s_values = list(values.get("qoe_rebuf_s_values", []))
    qoe_smoothness_values = list(values.get("qoe_smoothness_values", []))
    entropy_values = list(values.get("entropy_values", []))
    max_prob_values = list(values.get("max_prob_values", []))

    trace_bw_mbps_values = list(values.get("trace_bw_mbps_values", []))
    realized_bw_mbps_values = list(values.get("realized_bw_mbps_values", []))
    throughput_mbps_values = list(values.get("throughput_mbps_values", []))
    action_values = list(values.get("action_values", []))
    bitrate_kbps_values = list(values.get("bitrate_kbps_values", []))
    buffer_s_values = list(values.get("buffer_s_values", []))

    oracle_reward_values = list(values.get("oracle_reward_values", []))
    oracle_quality_values = list(values.get("oracle_quality_values", []))
    oracle_rebuf_s_values = list(values.get("oracle_rebuf_s_values", []))
    oracle_smoothness_values = list(values.get("oracle_smoothness_values", []))
    oracle_gap_reward_values = list(values.get("oracle_gap_reward_values", []))
    oracle_gap_quality_values = list(values.get("oracle_gap_quality_values", []))
    oracle_gap_rebuf_s_values = list(values.get("oracle_gap_rebuf_s_values", []))
    oracle_gap_smoothness_values = list(values.get("oracle_gap_smoothness_values", []))

    oracle_action_values = list(values.get("oracle_action_values", []))
    oracle_bitrate_kbps_values = list(values.get("oracle_bitrate_kbps_values", []))
    oracle_buffer_s_values = list(values.get("oracle_buffer_s_values", []))
    oracle_throughput_mbps_values = list(values.get("oracle_throughput_mbps_values", []))

    env_time_s_values = list(values.get("env/time_s_values", []))
    env_episode_values = list(values.get("env/episode_values", []))
    env_trace_values = list(values.get("env/trace_values", []))

    n = int(len(qoe_reward_values))
    for i in range(n):
        metrics: dict[str, object] = {
            f"{p}/env/step": int(i),
            f"{p}/qoe_reward": float(qoe_reward_values[i]),
            f"{p}/qoe/quality": float(qoe_quality_values[i]) if i < len(qoe_quality_values) else float("nan"),
            f"{p}/qoe/rebuf_s": float(qoe_rebuf_s_values[i]) if i < len(qoe_rebuf_s_values) else float("nan"),
            f"{p}/qoe/smoothness": float(qoe_smoothness_values[i]) if i < len(qoe_smoothness_values) else float("nan"),
            f"{p}/entropy": float(entropy_values[i]) if i < len(entropy_values) else float("nan"),
            f"{p}/max_prob": float(max_prob_values[i]) if i < len(max_prob_values) else float("nan"),
        }
        if i < len(env_time_s_values):
            metrics[f"{p}/env/time_s"] = float(env_time_s_values[i])
        if i < len(env_episode_values):
            metrics[f"{p}/env/episode"] = int(env_episode_values[i])
        if i < len(env_trace_values):
            metrics[f"{p}/env/trace"] = str(env_trace_values[i])
        if i < len(trace_bw_mbps_values):
            metrics[f"{p}/trace_bw_mbps"] = float(trace_bw_mbps_values[i])
        if i < len(realized_bw_mbps_values):
            metrics[f"{p}/realized_bw_mbps"] = float(realized_bw_mbps_values[i])
        if i < len(throughput_mbps_values):
            metrics[f"{p}/throughput_mbps"] = float(throughput_mbps_values[i])
        if i < len(action_values):
            metrics[f"{p}/action"] = int(action_values[i])
        if i < len(bitrate_kbps_values):
            metrics[f"{p}/bitrate_kbps"] = float(bitrate_kbps_values[i])
        if i < len(buffer_s_values):
            metrics[f"{p}/buffer_s"] = float(buffer_s_values[i])
        if i < len(oracle_reward_values):
            metrics[f"{p}/oracle/qoe_reward"] = float(oracle_reward_values[i])
        if i < len(oracle_quality_values):
            metrics[f"{p}/oracle/qoe/quality"] = float(oracle_quality_values[i])
        if i < len(oracle_rebuf_s_values):
            metrics[f"{p}/oracle/qoe/rebuf_s"] = float(oracle_rebuf_s_values[i])
        if i < len(oracle_smoothness_values):
            metrics[f"{p}/oracle/qoe/smoothness"] = float(oracle_smoothness_values[i])
        if i < len(oracle_gap_reward_values):
            metrics[f"{p}/oracle_gap/qoe_reward"] = float(oracle_gap_reward_values[i])
        if i < len(oracle_gap_quality_values):
            metrics[f"{p}/oracle_gap/qoe/quality"] = float(oracle_gap_quality_values[i])
        if i < len(oracle_gap_rebuf_s_values):
            metrics[f"{p}/oracle_gap/qoe/rebuf_s"] = float(oracle_gap_rebuf_s_values[i])
        if i < len(oracle_gap_smoothness_values):
            metrics[f"{p}/oracle_gap/qoe/smoothness"] = float(oracle_gap_smoothness_values[i])
        if i < len(oracle_action_values):
            metrics[f"{p}/oracle/action"] = int(oracle_action_values[i])
        if i < len(oracle_bitrate_kbps_values):
            metrics[f"{p}/oracle/bitrate_kbps"] = float(oracle_bitrate_kbps_values[i])
        if i < len(oracle_buffer_s_values):
            metrics[f"{p}/oracle/buffer_s"] = float(oracle_buffer_s_values[i])
        if i < len(oracle_throughput_mbps_values):
            metrics[f"{p}/oracle/throughput_mbps"] = float(oracle_throughput_mbps_values[i])
        wandb_run.log(metrics)


def _write_eval_values_to_summary(
    summary: dict[str, float],
    *,
    prefix: str,
    values: dict[str, object],
) -> None:
    p = str(prefix).rstrip("/")
    summary[f"{p}/env_steps-sum"] = float(values.get("env/steps", 0))
    summary[f"{p}/env_episodes-sum"] = float(values.get("env/episodes", 0))
    trace_bw_vals = [float(x) for x in list(values.get("trace_bw_mbps_values", [])) if np.isfinite(float(x))]
    realized_bw_vals = [float(x) for x in list(values.get("realized_bw_mbps_values", [])) if np.isfinite(float(x))]
    throughput_vals = [float(x) for x in list(values.get("throughput_mbps_values", [])) if np.isfinite(float(x))]
    buffer_vals = [float(x) for x in list(values.get("buffer_s_values", [])) if np.isfinite(float(x))]
    _write_agg_stats(summary, f"{p}/trace_bw_mbps", trace_bw_vals, include_geo_mean=True)
    _write_agg_stats(summary, f"{p}/realized_bw_mbps", realized_bw_vals, include_geo_mean=True)
    _write_agg_stats(summary, f"{p}/throughput_mbps", throughput_vals, include_geo_mean=True)
    _write_agg_stats(summary, f"{p}/buffer_s", buffer_vals, include_geo_mean=True)
    _write_agg_stats(summary, f"{p}/qoe_reward", list(values["qoe_reward_values"]), include_geo_mean=False)
    _write_agg_stats(summary, f"{p}/qoe/quality", list(values["qoe_quality_values"]), include_geo_mean=True)
    _write_agg_stats(summary, f"{p}/qoe/rebuf_s", list(values["qoe_rebuf_s_values"]), include_geo_mean=True)
    _write_agg_stats(summary, f"{p}/qoe/smoothness", list(values["qoe_smoothness_values"]), include_geo_mean=True)
    _write_agg_stats(summary, f"{p}/entropy", list(values["entropy_values"]), include_geo_mean=True)
    _write_agg_stats(summary, f"{p}/max_prob", list(values["max_prob_values"]), include_geo_mean=True)

    oracle_reward_values = [float(x) for x in list(values.get("oracle_reward_values", [])) if np.isfinite(float(x))]
    oracle_quality_values = [float(x) for x in list(values.get("oracle_quality_values", [])) if np.isfinite(float(x))]
    oracle_rebuf_s_values = [float(x) for x in list(values.get("oracle_rebuf_s_values", [])) if np.isfinite(float(x))]
    oracle_smoothness_values = [
        float(x) for x in list(values.get("oracle_smoothness_values", [])) if np.isfinite(float(x))
    ]
    oracle_buffer_s_values = [float(x) for x in list(values.get("oracle_buffer_s_values", [])) if np.isfinite(float(x))]
    oracle_throughput_mbps_values = [
        float(x) for x in list(values.get("oracle_throughput_mbps_values", [])) if np.isfinite(float(x))
    ]
    oracle_gap_reward_values = [
        float(x) for x in list(values.get("oracle_gap_reward_values", [])) if np.isfinite(float(x))
    ]
    oracle_gap_quality_values = [
        float(x) for x in list(values.get("oracle_gap_quality_values", [])) if np.isfinite(float(x))
    ]
    oracle_gap_rebuf_s_values = [
        float(x) for x in list(values.get("oracle_gap_rebuf_s_values", [])) if np.isfinite(float(x))
    ]
    oracle_gap_smoothness_values = [
        float(x) for x in list(values.get("oracle_gap_smoothness_values", [])) if np.isfinite(float(x))
    ]

    _write_agg_stats(summary, f"{p}/oracle/qoe_reward", oracle_reward_values, include_geo_mean=False)
    _write_agg_stats(summary, f"{p}/oracle/qoe/quality", oracle_quality_values, include_geo_mean=True)
    _write_agg_stats(summary, f"{p}/oracle/qoe/rebuf_s", oracle_rebuf_s_values, include_geo_mean=True)
    _write_agg_stats(summary, f"{p}/oracle/qoe/smoothness", oracle_smoothness_values, include_geo_mean=True)
    _write_agg_stats(summary, f"{p}/oracle/buffer_s", oracle_buffer_s_values, include_geo_mean=True)
    _write_agg_stats(summary, f"{p}/oracle/throughput_mbps", oracle_throughput_mbps_values, include_geo_mean=True)

    _write_agg_stats(summary, f"{p}/oracle_gap/qoe_reward", oracle_gap_reward_values, include_geo_mean=False)
    _write_agg_stats(summary, f"{p}/oracle_gap/qoe/quality", oracle_gap_quality_values, include_geo_mean=False)
    _write_agg_stats(summary, f"{p}/oracle_gap/qoe/rebuf_s", oracle_gap_rebuf_s_values, include_geo_mean=False)
    _write_agg_stats(summary, f"{p}/oracle_gap/qoe/smoothness", oracle_gap_smoothness_values, include_geo_mean=False)


def _ppo_finetune_on_traces(
    actor: network.Network,
    *,
    adv_cooked_time: list,
    adv_cooked_bw: list,
    episodes: int,
    seed: int,
    clean_cooked_time: list | None = None,
    clean_cooked_bw: list | None = None,
    clean_ratio: float = 0.7,
    batch_episodes: int = 16,
    ensure_adv_coverage: bool = True,
    wandb_run=None,
) -> dict[str, float]:
    #* On-policy PPO fine-tuning with mixed clean/adv rollouts.
    def _pop_shuffled_index(order: list[int], size: int, rng_local: np.random.RandomState) -> int:
        if int(size) <= 0:
            raise ValueError("* trace set is empty")
        if len(order) == 0:
            order.extend([int(i) for i in range(int(size))])
            rng_local.shuffle(order)
        return int(order.pop())

    def _rollout_on_trace(
        trace_time: list[float],
        trace_bw: list[float],
        *,
        rollout_seed: int,
        rng_local: np.random.RandomState,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[float], float]:
        net_env = env.Environment(
            all_cooked_time=[list(trace_time)],
            all_cooked_bw=[list(trace_bw)],
            random_seed=int(rollout_seed),
        )

        state = np.zeros((S_INFO, S_LEN), dtype=np.float64)
        last_bit_rate = DEFAULT_QUALITY
        bit_rate = DEFAULT_QUALITY

        (
            delay_ms,
            _sleep_time_ms,
            buffer_s,
            _rebuf_s,
            video_chunk_size,
            next_video_chunk_sizes,
            _end_of_video,
            video_chunk_remain,
        ) = net_env.get_video_chunk(int(bit_rate))
        state = _update_state(
            state,
            bit_rate=int(bit_rate),
            buffer_s=float(buffer_s),
            video_chunk_size=int(video_chunk_size),
            delay_ms=float(delay_ms),
            next_video_chunk_sizes=[int(x) for x in next_video_chunk_sizes],
            video_chunk_remain=int(video_chunk_remain),
        )

        s_batch: list[np.ndarray] = []
        a_batch: list[np.ndarray] = []
        p_batch: list[np.ndarray] = []
        r_batch: list[float] = []
        episode_return = 0.0

        while True:
            s_batch.append(np.array(state, copy=True))
            pi = actor.predict(np.reshape(state, (1, S_INFO, S_LEN))).astype(np.float64).reshape(-1)
            noise = rng_local.gumbel(size=len(pi))
            bit_rate = int(np.argmax(np.log(pi + 1e-12) + noise))

            action_vec = np.zeros(A_DIM, dtype=np.float64)
            action_vec[int(bit_rate)] = 1.0

            (
                delay_ms,
                _sleep_time_ms,
                buffer_s,
                rebuf_s,
                video_chunk_size,
                next_video_chunk_sizes,
                end_of_video,
                video_chunk_remain,
            ) = net_env.get_video_chunk(int(bit_rate))
            reward = float(calculate_reward(int(bit_rate), int(last_bit_rate), float(rebuf_s)))
            last_bit_rate = int(bit_rate)
            episode_return += float(reward)

            a_batch.append(action_vec)
            p_batch.append(np.array(pi, copy=True))
            r_batch.append(float(reward))

            state = _update_state(
                state,
                bit_rate=int(bit_rate),
                buffer_s=float(buffer_s),
                video_chunk_size=int(video_chunk_size),
                delay_ms=float(delay_ms),
                next_video_chunk_sizes=[int(x) for x in next_video_chunk_sizes],
                video_chunk_remain=int(video_chunk_remain),
            )

            if bool(end_of_video):
                break

        return s_batch, a_batch, p_batch, r_batch, float(episode_return)

    rng = np.random.RandomState(int(seed))
    adv_count = int(len(adv_cooked_time))
    if adv_count <= 0:
        raise ValueError("* robustify training requires at least one adversarial trace")
    if int(len(adv_cooked_bw)) != int(adv_count):
        raise ValueError("* adv_cooked_time and adv_cooked_bw must have the same length")

    clean_time = list(clean_cooked_time) if clean_cooked_time is not None else []
    clean_bw = list(clean_cooked_bw) if clean_cooked_bw is not None else []
    if int(len(clean_time)) != int(len(clean_bw)):
        raise ValueError("* clean_cooked_time and clean_cooked_bw must have the same length")
    clean_count = int(len(clean_time))

    total_episodes = int(episodes)
    if total_episodes <= 0:
        raise ValueError("* robustify episodes must be > 0")
    if bool(ensure_adv_coverage) and total_episodes < adv_count:
        raise ValueError(
            f"* robustify_episodes={total_episodes} is smaller than adv traces={adv_count}; "
            f"increase episodes or disable --no-robustify-ensure-adv-coverage"
        )

    batch_ep = int(max(1, int(batch_episodes)))
    clean_ratio_clamped = float(min(max(float(clean_ratio), 0.0), 1.0))
    adv_prob = float(1.0 - clean_ratio_clamped)

    t0 = time.perf_counter()
    episode_returns: list[float] = []
    episode_returns_adv: list[float] = []
    episode_returns_clean: list[float] = []
    update_loss_total: list[float] = []
    update_loss_policy: list[float] = []
    update_loss_value: list[float] = []
    update_loss_entropy: list[float] = []

    adv_cover_order = [int(i) for i in range(adv_count)]
    rng.shuffle(adv_cover_order)
    adv_cycle_order: list[int] = []
    clean_cycle_order: list[int] = []

    train_s: list[np.ndarray] = []
    train_a: list[np.ndarray] = []
    train_p: list[np.ndarray] = []
    train_v: list[float] = []
    update_episode_returns: list[float] = []
    update_episode_adv: int = 0
    update_idx = 0
    adv_episode_count = 0
    clean_episode_count = 0
    interrupted = False
    current_episode = -1

    def _run_update(*, flush_reason: str) -> None:
        nonlocal update_idx
        nonlocal train_s, train_a, train_p, train_v
        nonlocal update_episode_returns, update_episode_adv
        if len(update_episode_returns) == 0:
            return

        update_idx += 1
        train_stats = actor.train(
            np.stack(train_s, axis=0),
            np.vstack(train_a),
            np.vstack(train_p),
            np.vstack(train_v),
            int(max(current_episode, 0)),
        )

        if isinstance(train_stats, dict):
            if "train/loss/total" in train_stats:
                update_loss_total.append(float(train_stats["train/loss/total"]))
            if "train/loss/policy" in train_stats:
                update_loss_policy.append(float(train_stats["train/loss/policy"]))
            if "train/loss/value" in train_stats:
                update_loss_value.append(float(train_stats["train/loss/value"]))
            if "train/loss/entropy" in train_stats:
                update_loss_entropy.append(float(train_stats["train/loss/entropy"]))

        if wandb_run is not None:
            payload = {
                "train/update": int(update_idx),
                "train/update_batch_episodes": int(len(update_episode_returns)),
                "train/update_batch_adv_episodes": int(update_episode_adv),
                "train/update_batch_clean_episodes": int(len(update_episode_returns) - int(update_episode_adv)),
                "train/update_batch_return_mean": float(np.mean(update_episode_returns)),
                "train/update_batch_return_p50": float(np.percentile(update_episode_returns, 50)),
                "train/update_batch_return_p95": float(np.percentile(update_episode_returns, 95)),
                "train/update_adv_coverage_rate": float(1.0 - float(len(adv_cover_order)) / float(adv_count)),
                "train/update_flush_reason": str(flush_reason),
                "train/time_ms": float((time.perf_counter() - t0) * 1000.0),
            }
            if isinstance(train_stats, dict):
                payload.update({str(k): float(v) for k, v in train_stats.items()})
            wandb_run.log(payload)

        train_s = []
        train_a = []
        train_p = []
        train_v = []
        update_episode_returns = []
        update_episode_adv = 0

    try:
        for ep in range(total_episodes):
            current_episode = int(ep)
            episodes_left = int(total_episodes - ep)
            pending_adv_unseen = int(len(adv_cover_order))
            must_use_adv = bool(
                bool(ensure_adv_coverage)
                and int(pending_adv_unseen) > 0
                and int(pending_adv_unseen) >= int(episodes_left)
            )

            if clean_count <= 0:
                use_adv = True
            elif must_use_adv:
                use_adv = True
            else:
                use_adv = bool(rng.rand() < float(adv_prob))

            if use_adv:
                if len(adv_cover_order) > 0:
                    trace_idx = int(adv_cover_order.pop())
                else:
                    trace_idx = _pop_shuffled_index(adv_cycle_order, adv_count, rng)
                trace_time = list(adv_cooked_time[int(trace_idx)])
                trace_bw = list(adv_cooked_bw[int(trace_idx)])
                source_adv = 1
                adv_episode_count += 1
            else:
                trace_idx = _pop_shuffled_index(clean_cycle_order, clean_count, rng)
                trace_time = list(clean_time[int(trace_idx)])
                trace_bw = list(clean_bw[int(trace_idx)])
                source_adv = 0
                clean_episode_count += 1

            s_batch, a_batch, p_batch, r_batch, episode_return = _rollout_on_trace(
                trace_time=trace_time,
                trace_bw=trace_bw,
                rollout_seed=int(rng.randint(0, 2**31 - 1)),
                rng_local=rng,
            )
            v_batch = actor.compute_v(s_batch, a_batch, r_batch, terminal=True)

            train_s.extend(s_batch)
            train_a.extend(a_batch)
            train_p.extend(p_batch)
            train_v.extend([float(v) for v in v_batch])
            update_episode_returns.append(float(episode_return))
            update_episode_adv += int(source_adv)

            episode_returns.append(float(episode_return))
            if int(source_adv) == 1:
                episode_returns_adv.append(float(episode_return))
            else:
                episode_returns_clean.append(float(episode_return))

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/episode": int(ep),
                        "train/episode/return": float(episode_return),
                        "train/episode/is_adv": int(source_adv),
                        "train/episode/is_clean": int(1 - int(source_adv)),
                        "train/episode/adv_remaining_unseen": int(len(adv_cover_order)),
                        "train/episode/adv_coverage_rate": float(1.0 - float(len(adv_cover_order)) / float(adv_count)),
                        "train/time_ms": float((time.perf_counter() - t0) * 1000.0),
                    }
                )

            if int(len(update_episode_returns)) >= int(batch_ep):
                _run_update(flush_reason="batch_full")
    except KeyboardInterrupt:
        interrupted = True
        print("* Robustify interrupted by Ctrl+C; flushing pending batch and saving checkpoint.", flush=True)

    _run_update(flush_reason="interrupt" if bool(interrupted) else "train_end")

    dt_ms = (time.perf_counter() - t0) * 1000.0
    final_adv_coverage_rate = float(1.0 - float(len(adv_cover_order)) / float(adv_count))
    final_all_adv_seen = float(1.0 if len(adv_cover_order) == 0 else 0.0)

    def _safe_mean(values: list[float]) -> float:
        if len(values) == 0:
            return float("nan")
        return float(np.mean(values))

    def _safe_pctl(values: list[float], q: float) -> float:
        if len(values) == 0:
            return float("nan")
        return float(np.percentile(values, float(q)))

    episodes_completed = int(len(episode_returns))
    return {
        "train/episodes": float(int(episodes_completed)),
        "train/episodes_target": float(int(total_episodes)),
        "train/time_ms": float(dt_ms),
        "train/updates": float(int(update_idx)),
        "train/episode_return-mean": _safe_mean(episode_returns),
        "train/episode_return-p50": _safe_pctl(episode_returns, 50),
        "train/episode_return-p95": _safe_pctl(episode_returns, 95),
        "train/episode_return_adv-mean": _safe_mean(episode_returns_adv),
        "train/episode_return_clean-mean": _safe_mean(episode_returns_clean),
        "train/adv_episode_count": float(int(adv_episode_count)),
        "train/clean_episode_count": float(int(clean_episode_count)),
        "train/adv_coverage_rate": float(final_adv_coverage_rate),
        "train/all_adv_seen": float(final_all_adv_seen),
        "train/loss/total-mean": _safe_mean(update_loss_total),
        "train/loss/policy-mean": _safe_mean(update_loss_policy),
        "train/loss/value-mean": _safe_mean(update_loss_value),
        "train/loss/entropy-mean": _safe_mean(update_loss_entropy),
        "train/interrupted": float(1.0 if bool(interrupted) else 0.0),
    }


def robustify_eval(args: argparse.Namespace) -> None:
    #* Robustify a pretrained model by fine-tuning on saved two-pass adversarial traces.
    if args.adv_trace_dir is None:
        raise ValueError("* robustify mode requires --adv-trace-dir")

    pretrained_model = _resolve_under_src(str(args.model))
    configured_out_dir = getattr(args, "robustify_out_dir", None)
    models_dir = (
        _resolve_repo_root(str(configured_out_dir))
        if configured_out_dir is not None
        else _resolve_repo_root("models")
    )
    os.makedirs(str(models_dir), exist_ok=True)

    adv_trace_dir = _resolve_under_src(str(args.adv_trace_dir))
    if not os.path.isdir(adv_trace_dir):
        raise FileNotFoundError(f"* Adversarial trace dir not found: {adv_trace_dir}")

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    adv_trace_rel = os.path.relpath(str(adv_trace_dir), str(repo_root))
    adv_trace_rel = adv_trace_rel if not adv_trace_rel.startswith("..") else str(adv_trace_dir)

    adv_trace_base = os.path.basename(os.path.normpath(str(adv_trace_dir)))
    adv_trace_parent = os.path.basename(os.path.dirname(os.path.normpath(str(adv_trace_dir))))
    adv_trace_slug = str(adv_trace_base) if adv_trace_parent in ("", "traces") else f"{adv_trace_parent}-{adv_trace_base}"

    run_names = [
        d
        for d in sorted(os.listdir(adv_trace_dir))
        if os.path.isdir(os.path.join(adv_trace_dir, d)) and not d.startswith(".")
    ]
    run_dirs_by_name = {str(name): os.path.join(adv_trace_dir, str(name)) for name in run_names}
    if not run_names:
        #* Convenience: allow passing a single run dir that directly contains cooked traces.
        single_time, _, _ = _load_cooked_traces_dir_filtered(adv_trace_dir)
        if len(single_time) == 0:
            raise RuntimeError(f"* No runs (subdirs) or trace files found under {adv_trace_dir}")
        run_names = [str(adv_trace_base)]
        run_dirs_by_name = {str(adv_trace_base): str(adv_trace_dir)}

    #* Enforce a common trace budget per run when requested.
    adv_n = int(getattr(args, "adv_n", 0))
    if adv_n > 0:
        insufficient = []
        for run_name in run_names:
            run_dir = str(run_dirs_by_name[str(run_name)])
            count = _count_cooked_trace_files(run_dir)
            if int(count) < int(adv_n):
                insufficient.append((str(run_name), int(count)))
        if insufficient:
            joined = ", ".join([f"{name}={count}" for name, count in insufficient])
            raise RuntimeError(
                f"* --adv-trace-num={adv_n} requires every run under {adv_trace_dir} to have >= {adv_n} traces; "
                f"insufficient runs: {joined}"
            )

    tags = _parse_tags(args.wandb_tags) or []
    if str(adv_trace_base) not in tags:
        tags.insert(0, str(adv_trace_base))

    clean_ratio = float(getattr(args, "robustify_clean_ratio", 0.7))
    if not np.isfinite(float(clean_ratio)):
        raise ValueError("* --robustify-clean-ratio must be finite")
    clean_ratio = float(min(max(float(clean_ratio), 0.0), 1.0))

    clean_trace_dir = _resolve_under_src(str(getattr(args, "robustify_clean_trace_dir", "./train/")))
    clean_time: list = []
    clean_bw: list = []
    if float(clean_ratio) > 0.0:
        if not os.path.isdir(clean_trace_dir):
            raise FileNotFoundError(f"* Clean trace dir not found: {clean_trace_dir}")
        clean_time, clean_bw, _clean_names = _load_cooked_traces_dir_filtered(clean_trace_dir)
        if len(clean_time) == 0:
            raise RuntimeError(f"* Clean trace dir has no trace files: {clean_trace_dir}")

    for run_name in run_names:
        run_dir = str(run_dirs_by_name[str(run_name)])
        adv_time, adv_bw, adv_names = _load_cooked_traces_dir_filtered(run_dir)
        if int(adv_n) > 0:
            if len(adv_time) < int(adv_n):
                raise RuntimeError(f"* Run {run_name} has only {len(adv_time)} traces, need {adv_n}: {run_dir}")
            adv_time = adv_time[: int(adv_n)]
            adv_bw = adv_bw[: int(adv_n)]
            adv_names = adv_names[: int(adv_n)]
        if len(adv_time) == 0:
            print(f"* Skip empty trace run dir: {run_dir}", flush=True)
            continue

        wandb_run = None
        if args.wandb_mode != "disabled":
            wandb_run = wandb.init(
                project=str(args.wandb_project),
                entity=args.wandb_entity,
                name=str(run_name),
                mode=str(args.wandb_mode),
                tags=tags,
                config={
                    "mode": "robustify",
                    "adv_trace_dir": str(adv_trace_rel),
                    "adv_trace_slug": str(adv_trace_slug),
                    "adv_run": str(run_name),
                    "adv_n": int(getattr(args, "adv_n", 0)),
                    "pretrained_model": str(args.model),
                    "robustify_lr": float(args.robustify_lr),
                    "robustify_episodes": int(args.robustify_episodes),
                    "robustify_batch_episodes": int(args.robustify_batch_episodes),
                    "robustify_init_entropy_weight": (
                        None
                        if getattr(args, "robustify_init_entropy_weight", None) is None
                        else float(args.robustify_init_entropy_weight)
                    ),
                    "robustify_freeze_entropy_weight": int(bool(args.robustify_freeze_entropy_weight)),
                    "robustify_clean_trace_dir": str(clean_trace_dir),
                    "robustify_clean_ratio": float(clean_ratio),
                    "robustify_ensure_adv_coverage": int(bool(args.robustify_ensure_adv_coverage)),
                    "action_selection": str(args.action_selection),
                    "max_steps": args.max_steps,
                    "seed": int(args.seed),
                },
            )

            #* Use explicit step metrics so W&B can plot multiple time series in one run.
            wandb.define_metric("train/episode")
            wandb.define_metric("train/episode/*", step_metric="train/episode")
            wandb.define_metric("train/update")
            wandb.define_metric("train/loss/*", step_metric="train/update")
            wandb.define_metric("train/ratio_mean", step_metric="train/update")
            wandb.define_metric("train/clip_frac", step_metric="train/update")
            wandb.define_metric("train/approx_kl", step_metric="train/update")
            wandb.define_metric("train/adv_mean", step_metric="train/update")
            wandb.define_metric("train/entropy_weight_before", step_metric="train/update")
            wandb.define_metric("train/entropy_weight_after", step_metric="train/update")
            wandb.define_metric("train/update_*", step_metric="train/update")

        actor = network.Network(state_dim=[S_INFO, S_LEN], action_dim=A_DIM, learning_rate=float(args.robustify_lr))
        checkpoint_info = actor.load_model(
            pretrained_model,
            restore_optimizer=True,
            restore_entropy_weight=True,
        )
        entropy_init_applied = False
        if (
            not bool(checkpoint_info.get("restored_entropy_weight", False))
            and getattr(args, "robustify_init_entropy_weight", None) is not None
        ):
            actor.set_entropy_weight(float(args.robustify_init_entropy_weight))
            entropy_init_applied = True
        actor.set_freeze_entropy_weight(bool(args.robustify_freeze_entropy_weight))
        checkpoint_metrics = {
            "train/checkpoint_legacy_format": float(1.0 if bool(checkpoint_info.get("legacy_checkpoint", False)) else 0.0),
            "train/checkpoint_restored_optimizer": float(
                1.0 if bool(checkpoint_info.get("restored_optimizer", False)) else 0.0
            ),
            "train/checkpoint_restored_entropy_weight": float(
                1.0 if bool(checkpoint_info.get("restored_entropy_weight", False)) else 0.0
            ),
            "train/checkpoint_entropy_weight_initialized": float(1.0 if bool(entropy_init_applied) else 0.0),
            "train/checkpoint_entropy_weight_frozen": float(
                1.0 if bool(args.robustify_freeze_entropy_weight) else 0.0
            ),
            "train/checkpoint_entropy_weight_start": float(actor._entropy_weight),
        }

        train_stats = _ppo_finetune_on_traces(
            actor,
            adv_cooked_time=adv_time,
            adv_cooked_bw=adv_bw,
            episodes=int(args.robustify_episodes),
            seed=int(args.seed),
            clean_cooked_time=clean_time,
            clean_cooked_bw=clean_bw,
            clean_ratio=float(clean_ratio),
            batch_episodes=int(args.robustify_batch_episodes),
            ensure_adv_coverage=bool(args.robustify_ensure_adv_coverage),
            wandb_run=wandb_run,
        )
        train_stats.update(checkpoint_metrics)
        interrupted_run = bool(float(train_stats.get("train/interrupted", 0.0)) > 0.5)
        episodes_done = int(float(train_stats.get("train/episodes", 0.0)))

        orig_base = os.path.basename(str(args.model))
        orig_stem = os.path.splitext(orig_base)[0]
        if bool(interrupted_run):
            out_name = (
                f"{_safe_path_component(orig_stem)}_robust_"
                f"{_safe_path_component(str(adv_trace_slug))}-{_safe_path_component(run_name)}"
                f"_interrupt-ep{int(episodes_done)}.pt"
            )
        else:
            out_name = (
                f"{_safe_path_component(orig_stem)}_robust_"
                f"{_safe_path_component(str(adv_trace_slug))}-{_safe_path_component(run_name)}.pt"
            )
        out_path = os.path.join(str(models_dir), str(out_name))
        actor.save_model(out_path)
        print(f"* Saved robustified model: {out_path}", flush=True)

        if wandb_run is not None:
            wandb_run.summary.update({str(k): float(v) for k, v in train_stats.items()})
            wandb_run.summary["robustified_model"] = str(out_path)
            wandb.finish()


def shield_eval(args: argparse.Namespace) -> None:
    #* Evaluate a pretrained model with mined shield bounds (min/max bitrate) on trace runs; no training.
    if args.adv_trace_dir is None:
        raise ValueError("* shield mode requires --adv-trace-dir")

    pretrained_model = _resolve_under_src(str(args.model))
    shield_trace_dir = _resolve_under_src(str(args.adv_trace_dir))
    if not os.path.isdir(shield_trace_dir):
        raise FileNotFoundError(f"* Shield trace dir not found: {shield_trace_dir}")

    min_rules_file = _resolve_under_src(str(getattr(args, "shield_min_rules_file", "")))
    max_rules_file = _resolve_under_src(str(getattr(args, "shield_max_rules_file", "")))
    if not os.path.isfile(min_rules_file):
        raise FileNotFoundError(f"* Shield min rules file not found: {min_rules_file}")
    if not os.path.isfile(max_rules_file):
        raise FileNotFoundError(f"* Shield max rules file not found: {max_rules_file}")
    if int(getattr(args, "shield_risk_trigger_steps", 1)) <= 0:
        raise ValueError("* Shield: --shield-risk-trigger-steps must be >= 1")
    if int(getattr(args, "shield_cooldown_steps", 0)) < 0:
        raise ValueError("* Shield: --shield-cooldown-steps must be >= 0")

    min_rules = _load_shield_rules(min_rules_file, output_var="MinBitrateKbps")
    max_rules = _load_shield_rules(max_rules_file, output_var="MaxBitrateKbps")
    shield = ShieldBounds(
        min_rules=min_rules,
        max_rules=max_rules,
        risk_trigger_steps=int(getattr(args, "shield_risk_trigger_steps", 1)),
        intervention_cooldown_steps=int(getattr(args, "shield_cooldown_steps", 0)),
    )

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    shield_trace_rel = os.path.relpath(str(shield_trace_dir), str(repo_root))
    shield_trace_rel = shield_trace_rel if not shield_trace_rel.startswith("..") else str(shield_trace_dir)

    shield_trace_base = os.path.basename(os.path.normpath(str(shield_trace_dir)))
    shield_trace_parent = os.path.basename(os.path.dirname(os.path.normpath(str(shield_trace_dir))))
    shield_trace_slug = (
        str(shield_trace_base)
        if shield_trace_parent in ("", "traces")
        else f"{shield_trace_parent}-{shield_trace_base}"
    )

    run_names = [
        d
        for d in sorted(os.listdir(shield_trace_dir))
        if os.path.isdir(os.path.join(shield_trace_dir, d)) and not d.startswith(".")
    ]
    run_dirs_by_name = {str(name): os.path.join(shield_trace_dir, str(name)) for name in run_names}
    if not run_names:
        #* Convenience: allow passing a single run dir that directly contains cooked traces.
        single_time, _, _ = _load_cooked_traces_dir_filtered(shield_trace_dir)
        if len(single_time) == 0:
            raise RuntimeError(f"* No runs (subdirs) or trace files found under {shield_trace_dir}")
        run_names = [str(shield_trace_base)]
        run_dirs_by_name = {str(shield_trace_base): str(shield_trace_dir)}

    adv_n = int(getattr(args, "adv_n", 0))
    if adv_n > 0:
        insufficient = []
        for run_name in run_names:
            run_dir = str(run_dirs_by_name[str(run_name)])
            count = _count_cooked_trace_files(run_dir)
            if int(count) < int(adv_n):
                insufficient.append((str(run_name), int(count)))
        if insufficient:
            joined = ", ".join([f"{name}={count}" for name, count in insufficient])
            raise RuntimeError(
                f"* --adv-trace-num={adv_n} requires every run under {shield_trace_dir} to have >= {adv_n} traces; "
                f"insufficient runs: {joined}"
            )

    actor = network.Network(state_dim=[S_INFO, S_LEN], action_dim=A_DIM, learning_rate=ACTOR_LR_RATE)
    actor.load_model(pretrained_model)

    tags = _parse_tags(args.wandb_tags) or []
    if str(shield_trace_base) not in tags:
        tags.insert(0, str(shield_trace_base))

    for run_name in run_names:
        run_dir = str(run_dirs_by_name[str(run_name)])
        cooked_time, cooked_bw, cooked_names = _load_cooked_traces_dir_filtered(run_dir)
        if int(adv_n) > 0:
            cooked_time = cooked_time[: int(adv_n)]
            cooked_bw = cooked_bw[: int(adv_n)]
            cooked_names = cooked_names[: int(adv_n)]
        if len(cooked_time) == 0:
            print(f"* Skip empty trace run dir: {run_dir}", flush=True)
            continue

        wandb_run = None
        if args.wandb_mode != "disabled":
            wandb_run_name = _wandb_subrun_name(
                base_name=args.wandb_name,
                subrun=str(run_name),
                num_subruns=int(len(run_names)),
            )
            wandb_run = wandb.init(
                project=str(args.wandb_project),
                entity=args.wandb_entity,
                name=str(wandb_run_name),
                mode=str(args.wandb_mode),
                tags=tags,
                config={
                    "mode": "shield",
                    "shield_trace_dir": str(shield_trace_rel),
                    "shield_trace_slug": str(shield_trace_slug),
                    "shield_run": str(run_name),
                    "adv_n": int(getattr(args, "adv_n", 0)),
                    "pretrained_model": str(args.model),
                    "shield_min_rules_file": os.path.relpath(min_rules_file, str(repo_root))
                    if os.path.isabs(min_rules_file)
                    else str(min_rules_file),
                    "shield_max_rules_file": os.path.relpath(max_rules_file, str(repo_root))
                    if os.path.isabs(max_rules_file)
                    else str(max_rules_file),
                    "shield_risk_trigger_steps": int(getattr(args, "shield_risk_trigger_steps", 1)),
                    "shield_cooldown_steps": int(getattr(args, "shield_cooldown_steps", 0)),
                    "action_selection": str(args.action_selection),
                    "max_steps": args.max_steps,
                    "seed": int(args.seed),
                },
            )

            #* Match robustify metric schema: log baseline under `test/pretrained/*` and shielded under `test/robust/*`.
            wandb.define_metric("test/pretrained/env/step")
            wandb.define_metric("test/pretrained/*", step_metric="test/pretrained/env/step")
            wandb.define_metric("test/robust/env/step")
            wandb.define_metric("test/robust/*", step_metric="test/robust/env/step")

        baseline_values = _eval_collect_qoe_values_from_traces(
            actor,
            all_cooked_time=cooked_time,
            all_cooked_bw=cooked_bw,
            all_file_names=cooked_names,
            max_steps=args.max_steps,
            action_selection=str(args.action_selection),
            seed=int(args.seed),
            shield=None,
        )
        shield_values = _eval_collect_qoe_values_from_traces(
            actor,
            all_cooked_time=cooked_time,
            all_cooked_bw=cooked_bw,
            all_file_names=cooked_names,
            max_steps=args.max_steps,
            action_selection=str(args.action_selection),
            seed=int(args.seed),
            shield=shield,
        )

        if wandb_run is not None:
            _log_eval_time_series(wandb_run, prefix="test/pretrained", values=baseline_values)
            _log_eval_time_series(wandb_run, prefix="test/robust", values=shield_values)

            _write_eval_values_to_summary(wandb_run.summary, prefix="test/pretrained", values=baseline_values)
            _write_eval_values_to_summary(wandb_run.summary, prefix="test/robust", values=shield_values)

            #* Shield application stats.
            mi_vals = [float(x) for x in list(shield_values.get("shield_min_idx_values", [])) if np.isfinite(float(x))]
            ma_vals = [float(x) for x in list(shield_values.get("shield_max_idx_values", [])) if np.isfinite(float(x))]
            inc_vals = [
                float(x) for x in list(shield_values.get("shield_inconsistent_values", [])) if np.isfinite(float(x))
            ]
            if mi_vals and ma_vals:
                restricted = 0
                for mi, ma in zip(mi_vals, ma_vals):
                    if int(mi) > 0 or int(ma) < int(A_DIM) - 1:
                        restricted += 1
                wandb_run.summary["shield/restricted_rate"] = float(restricted) / float(max(len(mi_vals), 1))
            if inc_vals:
                wandb_run.summary["shield/inconsistent_rate"] = float(np.mean([1.0 if x > 0.0 else 0.0 for x in inc_vals]))

            wandb.finish()


def end_to_end_model_dir_eval(args: argparse.Namespace) -> None:
    #* End2end: run the same attack config against every model in `--model-dir`.
    if not bool(getattr(args, "two_pass", False)):
        raise ValueError("* --model-dir currently supports only --two-pass end2end runs")

    model_dir = _resolve_under_src(str(getattr(args, "model_dir", "")))
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"* --model-dir not found: {model_dir}")

    model_paths = _list_models_in_dir(model_dir)
    if not model_paths:
        raise RuntimeError(f"* No model files found under --model-dir: {model_dir}")

    base_name = args.wandb_name
    if base_name is None or not str(base_name).strip():
        trace_group_name = f"run_{int(time.time())}"
    else:
        trace_group_name = str(base_name).strip()
    for model_path in model_paths:
        suffix = _model_suffix_from_path(model_path)
        sub_args = argparse.Namespace(**vars(args))
        sub_args.model = str(model_path)
        if base_name is None or not str(base_name).strip():
            sub_args.wandb_name = str(suffix)
        else:
            sub_args.wandb_name = f"{str(base_name).strip()}_{suffix}"
        #* Ensure traces from different `--name` runs don't overwrite each other under the same tags:
        #  write under `.../<name>/<model-suffix>/` instead of `.../<model-suffix>/`.
        sub_args.trace_run_name = str(trace_group_name)
        sub_args.trace_model_suffix = str(suffix)

        print(f"\n====== Model: {sub_args.model} ======", flush=True)
        end_to_end_two_pass_eval(sub_args)


def end_to_end_two_pass_eval(args: argparse.Namespace) -> None:
    #* Two-pass end2end: (1) generate traces, (2) evaluate clean vs generated.
    np.random.seed(args.seed)

    trace_dir = _ensure_trailing_sep(_resolve_under_src(args.trace_dir))
    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(trace_dir)
    if args.trace_num > 0:
        all_cooked_time, all_cooked_bw, all_file_names = _sample_traces(
            all_cooked_time, all_cooked_bw, all_file_names, int(args.trace_num), seed=42
        )

    actor = network.Network(state_dim=[S_INFO, S_LEN], action_dim=A_DIM, learning_rate=ACTOR_LR_RATE)
    actor.load_model(_resolve_under_src(args.model))

    wandb_run = None
    if args.wandb_mode != "disabled":
        tags = _parse_tags(args.wandb_tags)
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            mode=args.wandb_mode,
            tags=tags,
            config={
                "trace_dir": args.trace_dir,
                "trace_num": args.trace_num,
                "model": args.model,
                "seed": args.seed,
                "max_steps": args.max_steps,
                "action_selection": args.action_selection,
                "attack_enabled": args.attack,
                "attack_method": getattr(args, "attack_method", "pgd"),
                "policy_rules_file": getattr(args, "policy_rules_file", None),
                "sym_eps_delay_ms": getattr(args, "sym_eps_delay_ms", 1e-6),
                "sym_max_rules": getattr(args, "sym_max_rules", 0),
                "sym_samples_per_rule": getattr(args, "sym_samples_per_rule", 0),
                "sym_min_delay_ms": getattr(args, "sym_min_delay_ms", 0.0),
                "sym_truegap_max_candidates_per_action": getattr(
                    args, "sym_truegap_max_candidates_per_action", 0
                ),
                "sym_truegap_oracle_min_ratio": getattr(args, "sym_truegap_oracle_min_ratio", 0.0),
                "sym_truegap_oracle_max_drop": getattr(args, "sym_truegap_oracle_max_drop", 0.0),
                "enforce_coupling": args.coupling,
                "use_rules": args.use_rules,
                "rules_file": args.rules_file,
                "rules_project_every_step": args.rules_project_every_step,
                "rules_perturb_both": args.rules_perturb_both,
                "pgd_alpha": args.pgd_alpha,
                "pgd_iters": args.pgd_iters,
                "pgd_objective_id": args.pgd_objective_id,
                "pgd_perturb_buffer": args.pgd_buffer,
                "pgd_history_delay": args.pgd_history_delay,
                "pgd_history_throughput": args.pgd_history_throughput,
                "pgd_history_sparse": args.pgd_history_sparse,
                "pgd_history_sparse_tol": args.pgd_history_sparse_tol,
                "smt_bw_samples": args.smt_bw_samples,
                "smt_trace_rules": args.smt_trace_rules,
                "smt_window_rel_tol": args.smt_window_rel_tol,
                "trace_out_dir": getattr(args, "trace_out_dir", "traces"),
            },
        )

    pgd_attacker = None
    if bool(args.attack):
        pgd_attacker = PGDAttack(actor, alpha=args.pgd_alpha, num_iterations=args.pgd_iters)

    out_dir, adv_time, adv_bw, adv_names, tracegen_stats = _generate_adversarial_traces(
        args,
        actor=actor,
        pgd_attacker=pgd_attacker,
        all_cooked_time=all_cooked_time,
        all_cooked_bw=all_cooked_bw,
        all_file_names=[str(x) for x in all_file_names],
    )
    if wandb_run is not None:
        wandb_run.summary["tracegen/out_dir"] = str(out_dir)
        smt_attempts = int(tracegen_stats.get("smt/attempts", 0))
        smt_successes = int(tracegen_stats.get("smt/successes", 0))
        smt_applied = int(tracegen_stats.get("smt/applied", 0))
        smt_complete = int(tracegen_stats.get("smt/complete", 0))
        denom = float(max(smt_attempts, 1))
        if smt_attempts > 0:
            wandb_run.summary["smt/success-rate"] = float(smt_successes) / denom
            wandb_run.summary["smt/applied-rate"] = float(smt_applied) / denom
            wandb_run.summary["smt/complete-rate"] = float(smt_complete) / denom
        else:
            #* No SMT attempts happened in this run.
            wandb_run.summary["smt/success-rate"] = 1.0
            wandb_run.summary["smt/applied-rate"] = 0.0
            wandb_run.summary["smt/complete-rate"] = 1.0
        wandb_run.summary["smt/skipped"] = int(tracegen_stats.get("smt/skipped", 0))
        #* Recorded during trace generation; 0 when rules are not used (or attack is disabled).
        wandb_run.summary["pgd/rules_projection_calls_total"] = int(
            tracegen_stats.get("pgd/rules_projection_calls_total", 0)
        )
        wandb_run.summary["pgd/rules_fully_relaxed_calls_total"] = int(
            tracegen_stats.get("pgd/rules_fully_relaxed_calls_total", 0)
        )
        wandb_run.summary["pgd/rules_fully_relaxed_rate_total"] = float(
            tracegen_stats.get("pgd/rules_fully_relaxed_rate_total", 0.0)
        )
        for key, value in tracegen_stats.items():
            if str(key).startswith("pgd/history_sparse"):
                wandb_run.summary[str(key)] = value
            if str(key).startswith("pgd/rules_projection"):
                wandb_run.summary[str(key)] = value
            if str(key).startswith("sym/") and (str(key).endswith("-rate") or str(key) == "sym/attempts"):
                wandb_run.summary[str(key)] = value
            if str(key).startswith("symtrue/"):
                wandb_run.summary[str(key)] = value

    #* Phase 2 wants access to Phase 1 per-chunk expectations stored in the manifest.
    tracegen_expected_realized_bw_by_trace: dict[str, list[float]] = {}
    tracegen_expected_action_exec_by_trace: dict[str, list[float]] = {}
    tracegen_expected_action_next_by_trace: dict[str, list[float]] = {}
    tracegen_expected_delay_ms_by_trace: dict[str, list[float]] = {}
    tracegen_expected_throughput_mbps_by_trace: dict[str, list[float]] = {}
    tracegen_expected_video_chunk_size_bytes_by_trace: dict[str, list[float]] = {}
    tracegen_expected_pgd_iters_run_by_trace: dict[str, list[float]] = {}
    try:
        manifest_path = os.path.join(str(out_dir), "manifest.json")
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        for entry in manifest.get("traces", []):
            if not isinstance(entry, dict):
                continue
            adv_name = entry.get("adv_name")
            expected_bw = entry.get("expected_realized_bw_mbps")
            if not isinstance(adv_name, str):
                continue
            if isinstance(expected_bw, list):
                vals: list[float] = []
                for x in expected_bw:
                    try:
                        vals.append(float(x))
                    except Exception:
                        continue
                tracegen_expected_realized_bw_by_trace[str(adv_name)] = vals

            expected_action_exec = entry.get("expected_action_exec")
            if isinstance(expected_action_exec, list):
                vals: list[float] = []
                for x in expected_action_exec:
                    try:
                        vals.append(float(int(x)))
                    except Exception:
                        continue
                tracegen_expected_action_exec_by_trace[str(adv_name)] = vals

            expected_action_next = entry.get("expected_action_next")
            if isinstance(expected_action_next, list):
                vals: list[float] = []
                for x in expected_action_next:
                    try:
                        vals.append(float(int(x)))
                    except Exception:
                        continue
                tracegen_expected_action_next_by_trace[str(adv_name)] = vals

            expected_delay_ms = entry.get("expected_delay_ms")
            if isinstance(expected_delay_ms, list):
                vals: list[float] = []
                for x in expected_delay_ms:
                    try:
                        vals.append(float(x))
                    except Exception:
                        continue
                tracegen_expected_delay_ms_by_trace[str(adv_name)] = vals

            expected_throughput_mbps = entry.get("expected_throughput_mbps")
            if isinstance(expected_throughput_mbps, list):
                vals: list[float] = []
                for x in expected_throughput_mbps:
                    try:
                        vals.append(float(x))
                    except Exception:
                        continue
                tracegen_expected_throughput_mbps_by_trace[str(adv_name)] = vals

            expected_video_chunk_size_bytes = entry.get("expected_video_chunk_size_bytes")
            if isinstance(expected_video_chunk_size_bytes, list):
                vals: list[float] = []
                for x in expected_video_chunk_size_bytes:
                    try:
                        vals.append(float(int(x)))
                    except Exception:
                        continue
                tracegen_expected_video_chunk_size_bytes_by_trace[str(adv_name)] = vals

            expected_pgd_iters_run = entry.get("expected_pgd_iters_run")
            if isinstance(expected_pgd_iters_run, list):
                vals: list[float] = []
                for x in expected_pgd_iters_run:
                    try:
                        vals.append(float(int(x)))
                    except Exception:
                        continue
                tracegen_expected_pgd_iters_run_by_trace[str(adv_name)] = vals
    except Exception:
        tracegen_expected_realized_bw_by_trace = {}
        tracegen_expected_action_exec_by_trace = {}
        tracegen_expected_action_next_by_trace = {}
        tracegen_expected_delay_ms_by_trace = {}
        tracegen_expected_throughput_mbps_by_trace = {}
        tracegen_expected_video_chunk_size_bytes_by_trace = {}
        tracegen_expected_pgd_iters_run_by_trace = {}

    _paired_clean_vs_adv_eval(
        args,
        actor=actor,
        clean_all_cooked_time=all_cooked_time,
        clean_all_cooked_bw=all_cooked_bw,
        clean_file_names=[str(x) for x in all_file_names],
        adv_all_cooked_time=adv_time,
        adv_all_cooked_bw=adv_bw,
        adv_file_names=adv_names,
        tracegen_expected_realized_bw_by_trace=tracegen_expected_realized_bw_by_trace,
        tracegen_expected_action_exec_by_trace=tracegen_expected_action_exec_by_trace,
        tracegen_expected_action_next_by_trace=tracegen_expected_action_next_by_trace,
        tracegen_expected_delay_ms_by_trace=tracegen_expected_delay_ms_by_trace,
        tracegen_expected_throughput_mbps_by_trace=tracegen_expected_throughput_mbps_by_trace,
        tracegen_expected_video_chunk_size_bytes_by_trace=tracegen_expected_video_chunk_size_bytes_by_trace,
        tracegen_expected_pgd_iters_run_by_trace=tracegen_expected_pgd_iters_run_by_trace,
        wandb_run=wandb_run,
    )

    if wandb_run is not None:
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pensieve-PPO inference / PGD analysis utilities")
    parser.add_argument(
        "--mode",
        choices=["end2end", "robustify", "shield"],
        default="end2end",
        help="* end2end: generate/evaluate adversarial traces; robustify: fine-tune on saved traces; shield: replay traces with shield bounds",
    )

    #* Shared evaluation flags (end2end/robustify).
    parser.add_argument(
        "-t",
        "--trace",
        "--trace-dir",
        dest="trace_dir",
        default="./test1/",
        help="* Trace folder under `src/` (e.g., ./test/, ./train/)",
    )
    parser.add_argument(
        "-n",
        "--num",
        "--trace-num",
        dest="trace_num",
        type=int,
        default=1,
        help="* Number of traces to evaluate (-1 = all)",
    )
    parser.add_argument("-m", "--model", default=NN_MODEL, help="* PPO model path under `src/`")
    parser.add_argument(
        "--model-dir",
        dest="model_dir",
        default=None,
        help="* End2end: run attacks for ALL models under this directory (overrides -m; traces saved under `<name>/<model-suffix>` when --two-pass)",
    )
    parser.add_argument(
        "--alpha", 
        "--pgd-alpha", 
        dest="pgd_alpha", type=float, 
        default=0.001, 
        help="* PGD step size")
    parser.add_argument(
        "--iters", 
        "--pgd-iters", 
        dest="pgd_iters", type=int, 
        default=20, 
        help="* PGD iterations")
    parser.add_argument(
        "--objective-id",
        "--pgd-objective-id",
        dest="pgd_objective_id",
        type=int,
        default=3,
        help="* PGD objective id (1..7). Objective 3 = surrogate oracle-gap, 5 = action-gap target, 6 = rebuffer-worst target, 7 = decision flip",
    )
    parser.add_argument(
        "--steps",
        "--max-steps",
        dest="max_steps",
        type=int,
        default=500,
        help="* Max total steps (-1 for unlimited)",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)

    parser.add_argument("--attack", action=argparse.BooleanOptionalAction, default=True, help="* Enable/disable attack")
    parser.add_argument(
        "--attack-method",
        dest="attack_method",
        choices=["pgd", "dt", "dt-truegap"],
        default="pgd",
        help="* Attack method: pgd=gradient PGD, dt=symbolic student DT rules, dt-truegap=symbolic DT ranked by true oracle gap",
    )
    parser.add_argument(
        "--policy-rules-file",
        dest="policy_rules_file",
        default="rules/pensieve_dagger_rules_sklearn_1k.pl",
        help="* Policy rules file (SymPy-like Implies rules) for --attack-method dt/dt-truegap",
    )
    parser.add_argument(
        "--sym-eps-delay-ms",
        dest="sym_eps_delay_ms",
        type=float,
        default=1e-6,
        help="* Symbolic attack: epsilon (ms) for strict delay bounds",
    )
    parser.add_argument(
        "--sym-max-rules",
        dest="sym_max_rules",
        type=int,
        default=0,
        help="* Symbolic DT attack: max feasible rules considered per step (0=all)",
    )
    parser.add_argument(
        "--sym-samples-per-rule",
        dest="sym_samples_per_rule",
        type=int,
        default=0,
        help="* Symbolic DT attack: random delay samples per rule (in addition to min+boundary)",
    )
    parser.add_argument(
        "--sym-min-delay-ms",
        dest="sym_min_delay_ms",
        type=float,
        default=200.823012174641,
        help="* Symbolic DT attack: min delay (ms) for throughput coupling computation",
    )
    parser.add_argument(
        "--sym-truegap-max-candidates-per-action",
        dest="sym_truegap_max_candidates_per_action",
        type=int,
        default=5,
        help="* Symbolic dt-truegap: max top-ranked candidates evaluated per target action per step (0=all)",
    )
    parser.add_argument(
        "--sym-truegap-oracle-min-ratio",
        dest="sym_truegap_oracle_min_ratio",
        type=float,
        default=0.9,
        help="* Symbolic dt-truegap: require oracle_mean(attacked) >= ratio*oracle_mean(clean) when oracle_mean(clean)>0 (0=disable)",
    )
    parser.add_argument(
        "--sym-truegap-oracle-max-drop",
        dest="sym_truegap_oracle_max_drop",
        type=float,
        default=-1.0,
        help="* Symbolic dt-truegap: require oracle_mean(attacked) >= oracle_mean(clean)-drop (negative=disable)",
    )
    parser.add_argument(
        "--coupling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="* Enforce throughput-delay consistency inside PGD",
    )
    parser.add_argument(
        "--buffer",
        dest="pgd_buffer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="* Allow PGD to perturb buffer feature (use --no-buffer to freeze buffer)",
    )
    parser.add_argument(
        "--history-delay",
        dest="pgd_history_delay",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="* Allow PGD to perturb delay history `state[3, :]` (default: latest only)",
    )
    parser.add_argument(
        "--history-throughput",
        "--history-download",
        dest="pgd_history_throughput",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="* Allow PGD to perturb throughput history `state[2, :]` (default: no direct throughput perturbation)",
    )
    parser.add_argument(
        "--history-sparse",
        dest="pgd_history_sparse",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="* Greedy history sparsification: rank `abs(grad*delta)` from full-history PGD, then rerun PGD allowing top-k values until within tolerance",
    )
    parser.add_argument(
        "--history-sparse-tol",
        dest="pgd_history_sparse_tol",
        type=float,
        default=0.05,
        help="* Stopping tolerance for --history-sparse (loss <= full_loss + tol*max(1,|full_loss|))",
    )
    parser.add_argument(
        "--rules",
        dest="use_rules",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="* Use mined logic rules for PGD projection (implies --no-coupling)",
    )
    parser.add_argument(
        "--rules-file",
        dest="rules_file",
        default=None,
        help="* Mined rules file (SymPy-like syntax) for --rules",
    )
    parser.add_argument(
        "--rules-step",
        "--rules-every-step",
        dest="rules_project_every_step",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="* When --rules, project onto rules at every PGD iteration (slow)",
    )
    parser.add_argument(
        "--rules-both",
        "--rules-perturb-both",
        dest="rules_perturb_both",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="* When --rules, allow PGD to perturb (delay_norm, throughput_norm) before projection",
    )
    parser.add_argument(
        "--select",
        "--action-selection",
        dest="action_selection",
        choices=["argmax", "gumbel"],
        default="argmax",
        help="* How to roll out the environment while collecting states",
    )

    parser.add_argument(
        "--two-pass",
        dest="two_pass",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="* End2end: first generate/save adversarial traces, then evaluate on them",
    )
    parser.add_argument(
        "--trace-out-dir",
        dest="trace_out_dir",
        default="traces",
        help="* End2end --two-pass: output base dir under repo root (default: ./traces)",
    )

    #* Robustify: fine-tune a pretrained model on saved traces under a directory containing run subdirectories.
    parser.add_argument(
        "--adv-trace-dir",
        dest="adv_trace_dir",
        default=None,
        help="* Robustify/shield: directory containing run subdirectories of cooked traces (or a single run dir with trace files)",
    )
    parser.add_argument(
        "--adv-trace-num",
        "--adv-n",
        dest="adv_n",
        type=int,
        default=-1,
        help="* Robustify/shield: common number of traces per run dir (-1 = all); error if any run has fewer than N",
    )
    parser.add_argument(
        "--robustify-episodes",
        dest="robustify_episodes",
        type=int,
        default=40,
        help="* Robustify: PPO fine-tune episodes per run",
    )
    parser.add_argument(
        "--robustify-lr",
        dest="robustify_lr",
        type=float,
        default=1e-5,
        help="* Robustify: PPO fine-tune learning rate",
    )
    parser.add_argument(
        "--robustify-out-dir",
        dest="robustify_out_dir",
        default=None,
        help="* Robustify: checkpoint output directory (default: repository models directory)",
    )
    parser.add_argument(
        "--robustify-clean-trace-dir",
        dest="robustify_clean_trace_dir",
        default="./train/",
        help="* Robustify: clean trace directory used for mixed training",
    )
    parser.add_argument(
        "--robustify-clean-ratio",
        dest="robustify_clean_ratio",
        type=float,
        default=0.7,
        help="* Robustify: target clean episode ratio in mixed training (0..1)",
    )
    parser.add_argument(
        "--robustify-batch-episodes",
        dest="robustify_batch_episodes",
        type=int,
        default=16,
        help="* Robustify: number of episodes collected per PPO update",
    )
    parser.add_argument(
        "--robustify-init-entropy-weight",
        dest="robustify_init_entropy_weight",
        type=float,
        default=None,
        help="* Robustify: fallback entropy weight used only when resuming from a legacy checkpoint without saved entropy state",
    )
    parser.add_argument(
        "--robustify-freeze-entropy-weight",
        dest="robustify_freeze_entropy_weight",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="* Robustify: keep the entropy weight fixed during fine-tuning",
    )
    parser.add_argument(
        "--robustify-ensure-adv-coverage",
        dest="robustify_ensure_adv_coverage",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="* Robustify: force all selected adversarial traces to be seen at least once",
    )

    #* Shield: enforce mined (min,max) bitrate bounds during inference.
    parser.add_argument(
        "--shield-min-rules-file",
        dest="shield_min_rules_file",
        default="rules/shield/out/online_adv_bounds_sklearn_min.pl",
        help="* Shield: min-bitrate bounds rules file (expects consequent `MinBitrateKbps == ...`)",
    )
    parser.add_argument(
        "--shield-max-rules-file",
        dest="shield_max_rules_file",
        default="rules/shield/out/online_adv_bounds_sklearn_max.pl",
        help="* Shield: max-bitrate bounds rules file (expects consequent `MaxBitrateKbps == ...`)",
    )
    parser.add_argument(
        "--shield-risk-trigger-steps",
        dest="shield_risk_trigger_steps",
        type=int,
        default=1,
        help="* Shield hysteresis: require this many consecutive risky states before intervening.",
    )
    parser.add_argument(
        "--shield-cooldown-steps",
        dest="shield_cooldown_steps",
        type=int,
        default=0,
        help="* Shield hysteresis: suppress further interventions for this many steps after an intervention.",
    )

    #* SMT flags (used by `--mode end2end`).
    parser.add_argument(
        "--bw-samples", 
        dest="smt_bw_samples", 
        type=int, 
        default=5, 
        help="* SMT: num raw trace segments (BandwidthMbps0..N-1, DurationSec0..N-1)"
    )
    parser.add_argument(
        "--trace-rules",
        "--smt-trace-rules",
        dest="smt_trace_rules",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="* SMT: enforce mined trace-dynamics rules during trace synthesis",
    )
    parser.add_argument(
        "--smt-window-rel-tol",
        dest="smt_window_rel_tol",
        type=float,
        default=1e-5,
        help="* SMT (window): relative tolerance band (e.g., 0.10 = ±10%%)",
    )

    #* W&B flags (short names + compatible aliases).
    parser.add_argument(
        "--project", "--wandb-project", 
        dest="wandb_project", 
        default="attack-pensieve-gap",
        help="* W&B project name",)
    parser.add_argument("--entity", "--wandb-entity", dest="wandb_entity", default=None)
    parser.add_argument("--name", "--wandb-name", dest="wandb_name", default=None)
    parser.add_argument(
        "--tags",
        "--wandb-tags",
        dest="wandb_tags",
        default=None,
        help="* Comma-separated W&B tags (e.g., tag1,tag2)",
    )
    parser.add_argument(
        "--wandb",
        "--wandb-mode",
        dest="wandb_mode",
        choices=["online", "offline", "disabled"],
        default="online",
        help="* W&B logging mode",
    )

    args = parser.parse_args()
    #* Print run name if provided.
    if args.wandb_name is not None:
        print(f"\n====== Run name: {args.wandb_name} ======")    

    if args.trace_num == -1:
        args.trace_num = 0
    if getattr(args, "adv_n", None) == -1:
        args.adv_n = 0
    if args.max_steps is not None and args.max_steps < 0:
        args.max_steps = None
    if bool(getattr(args, "attack", False)) and str(getattr(args, "attack_method", "pgd")) in ("dt", "dt-truegap"):
        #* DT attacks can optionally use the environment-rule projection stage for realizability.
        #  Enable it automatically when a rules file is provided; otherwise keep `--no-rules` behavior.
        args.coupling = False
        if args.rules_file is not None:
            args.use_rules = True
    if bool(getattr(args, "attack", False)) and str(getattr(args, "attack_method", "pgd")) == "dt-truegap":
        if bool(getattr(args, "pgd_history_throughput", False)):
            raise ValueError("* --attack-method dt-truegap currently supports only --no-history-throughput")
    if args.use_rules:
        #* Rules projection is a 3rd option; it disables the hand-coded coupling.
        args.coupling = False
        if args.rules_file is None:
            raise ValueError("* --rules requires --rules-file (or pass --no-rules to disable projection)")
        if not args.rules_perturb_both:
            args.pgd_buffer = False
    if args.mode == "robustify":
        robustify_eval(args)
    elif args.mode == "shield":
        shield_eval(args)
    elif args.mode == "end2end":
        if getattr(args, "model_dir", None) is not None:
            end_to_end_model_dir_eval(args)
        elif bool(getattr(args, "two_pass", False)):
            end_to_end_two_pass_eval(args)
        else:
            end_to_end_eval(args)
