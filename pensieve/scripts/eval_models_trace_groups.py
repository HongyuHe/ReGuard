#!/usr/bin/env python3
from __future__ import annotations

"""
Example usage:
python scripts/eval_models_trace_groups.py \
    --base-model ppo/nn_model_ep_426300.pth --robust-model-dir ./models/robustified-189e/ \
    --adv-trace-dir ./traces/test/ --clean-trace-dir Pensieve-PPO/src/test \
    --out-csv traces/robustification_eval_189e.csv
"""

import argparse
import csv
import concurrent.futures
import multiprocessing as mp
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


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


def _model_robustification_suffix(model_path: str) -> str:
    stem = os.path.splitext(os.path.basename(str(model_path)))[0]
    if "_robust_" in stem:
        return str(stem.split("_robust_", 1)[1])
    return "original"


def _list_model_files(model_dir: str) -> list[str]:
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


def _has_trace_files(trace_dir: str) -> bool:
    for name in os.listdir(str(trace_dir)):
        if name.startswith("."):
            continue
        path = os.path.join(str(trace_dir), str(name))
        if _is_cooked_trace_file(path):
            return True
    return False


def _list_trace_files(trace_dir: str) -> list[str]:
    #* Mirrors `Pensieve-PPO/src/load_trace.py` selection.
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
    #* Accepts legacy cooked traces with arbitrary suffixes while excluding metadata sidecars.
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


def _load_trace_subset(*, trace_dir: str, num_traces: int) -> tuple[list, list, list[str]]:
    #* Avoids loading all trace files when only the first N are needed.
    files = _list_trace_files(str(trace_dir))
    n = int(num_traces)
    if n <= 0:
        raise ValueError(f"* num_traces must be positive, got: {num_traces}")
    if len(files) < n:
        raise RuntimeError(f"* Trace dir has only {len(files)} trace(s) (< {n}): {trace_dir}")

    all_cooked_time: list = []
    all_cooked_bw: list = []
    all_file_names: list[str] = []
    for name in list(files[:n]):
        file_path = os.path.join(str(trace_dir), str(name))
        cooked_time: list[float] = []
        cooked_bw: list[float] = []
        with open(str(file_path), "rb") as f:
            for line in f:
                parse = line.split()
                cooked_time.append(float(parse[0]))
                cooked_bw.append(float(parse[1]))
        all_cooked_time.append(cooked_time)
        all_cooked_bw.append(cooked_bw)
        all_file_names.append(str(name))

    return all_cooked_time, all_cooked_bw, all_file_names


@dataclass(frozen=True)
class TraceGroup:
    eval_trace: str
    attack_name: str
    trace_victim_suffix: str
    trace_dir: str


def _eprint(msg: str) -> None:
    print(str(msg), file=sys.stderr, flush=True)


def _try_import_wandb() -> Any | None:
    try:
        import wandb  # type: ignore

        if not hasattr(wandb, "init"):
            raise ImportError("* `wandb` import resolved to a non-package directory; install the wandb pip package")
        return wandb
    except Exception:
        return None


def _load_pensieve_modules() -> tuple[Any, Any, Any]:
    repo_root = _repo_root()
    src_dir = os.path.join(repo_root, "Pensieve-PPO", "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    import load_trace as load_trace_mod  # type: ignore
    import fixed_env as env_mod  # type: ignore
    import ppo2 as network  # type: ignore
    import util as util_mod  # type: ignore

    return network, util_mod, env_mod


def _discover_adv_trace_groups(adv_trace_dir: str) -> list[TraceGroup]:
    groups: list[TraceGroup] = []
    for attack_name in sorted(os.listdir(str(adv_trace_dir))):
        attack_path = os.path.join(str(adv_trace_dir), str(attack_name))
        if not os.path.isdir(attack_path):
            continue
        for victim_suffix in sorted(os.listdir(attack_path)):
            group_path = os.path.join(attack_path, str(victim_suffix))
            if not os.path.isdir(group_path):
                continue
            if not _has_trace_files(group_path):
                continue
            groups.append(
                TraceGroup(
                    eval_trace=f"{attack_name}/{victim_suffix}",
                    attack_name=str(attack_name),
                    trace_victim_suffix=str(victim_suffix),
                    trace_dir=str(group_path),
                )
            )
    return groups


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


def _entropy(p: Any, *, eps: float = 1e-12) -> float:
    import numpy as np

    arr = np.asarray(p, dtype=np.float64).reshape(-1)
    if arr.size <= 0:
        return 0.0
    arr = arr / max(float(np.sum(arr)), float(eps))
    arr = np.clip(arr, float(eps), 1.0)
    return float(-np.sum(arr * np.log(arr)))


def _throughput_mbps(video_chunk_size_bytes: float, delay_ms: float) -> float:
    #* Matches `Pensieve-PPO/src/inference.py:_throughput_mbps`.
    delay_ms_f = float(delay_ms)
    if delay_ms_f <= 0.0:
        return 0.0
    return float(video_chunk_size_bytes) * 8.0 / (delay_ms_f * 1000.0)


def _avg_bandwidth_mbps(video_chunk_size_bytes: float, delay_ms: float, *, env_mod: Any) -> float:
    #* Matches `Pensieve-PPO/src/inference.py:_avg_bandwidth_mbps`.
    delay_ms_f = float(delay_ms)
    rtt_free_delay_ms = float(max(delay_ms_f - float(getattr(env_mod, "LINK_RTT", 0.0)), 0.0))
    if rtt_free_delay_ms <= 0.0:
        return 0.0
    payload_portion = float(getattr(env_mod, "PACKET_PAYLOAD_PORTION", 1.0))
    if payload_portion <= 0.0:
        return 0.0
    return float(video_chunk_size_bytes) * 8.0 / (rtt_free_delay_ms * 1000.0) / payload_portion


def _agg_stats(
    values: list[float],
    *,
    include_geo_mean: bool = True,
    geo_mean_eps: float = 1e-12,
) -> dict[str, float]:
    #* Mirrors `Pensieve-PPO/src/inference.py:_agg_stats`.
    if len(values) == 0:
        return {"mean": float("nan"), "geo_mean": float("nan"), "p50": float("nan"), "p95": float("nan")}

    import numpy as np

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
    #* Mirrors `Pensieve-PPO/src/inference.py:_write_agg_stats`.
    stats = _agg_stats(values, include_geo_mean=include_geo_mean)
    summary[f"{metric_name}-mean"] = float(stats["mean"])
    summary[f"{metric_name}-p50"] = float(stats["p50"])
    summary[f"{metric_name}-p95"] = float(stats["p95"])
    if include_geo_mean:
        summary[f"{metric_name}-geo_mean"] = float(stats["geo_mean"])


def _update_state(
    state,
    *,
    bit_rate: int,
    buffer_s: float,
    video_chunk_size: int,
    delay_ms: float,
    next_video_chunk_sizes: list[int],
    video_chunk_remain: int,
    util_mod,
):
    #* Matches `Pensieve-PPO/src/inference.py:_update_state`.
    import numpy as np

    new_state = np.roll(state, -1, axis=1).astype(np.float64, copy=False)
    new_state[0, -1] = util_mod.VIDEO_BIT_RATE[int(bit_rate)] / float(max(util_mod.VIDEO_BIT_RATE))
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
    env_snapshot,
    last_br: int,
    k_steps: int,
    seed: int,
    util_mod,
) -> tuple[float | None, float | None, float | None, float | None, int | None, float | None, float | None]:
    k = int(k_steps)
    if k <= 0:
        return None, None, None, None, None, None, None

    best_r, best_seq, _worst_r, _worst_seq = util_mod.find_optimal_k_step_reward_oracle(env_snapshot, int(last_br), int(k))
    if best_r is None or best_seq is None:
        return None, None, None, None, None, None, None

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
            _delay_ms, _sleep_ms, buffer_s, rebuf_s, _chunk_size, _next_sizes, end, _remain = env_sim.get_video_chunk(int(a))
        except Exception:
            return None, None, None, None, None, None, None

        if idx == 0:
            oracle_buffer_s = float(buffer_s)

        if bool(end) and idx < int(k) - 1:
            return None, None, None, None, None, None, None

        total_quality += float(util_mod.VIDEO_BIT_RATE[int(a)]) / float(util_mod.M_IN_K)
        total_rebuf_s += float(rebuf_s)
        total_smoothness += float(abs(util_mod.VIDEO_BIT_RATE[int(a)] - util_mod.VIDEO_BIT_RATE[int(last)])) / float(util_mod.M_IN_K)
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
    env_snapshot,
    last_br: int,
    k_steps: int,
    util_mod,
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
    actor,
    env_snapshot,
    state_snapshot,
    last_br: int,
    start_action: int,
    k_steps: int,
    seed: int,
    util_mod,
):
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
            a = int(np.argmax(p))

    denom = float(k)
    return (
        float(total_reward) / denom,
        float(total_quality) / denom,
        float(total_rebuf_s) / denom,
        float(total_smoothness) / denom,
    )


def _eval_actor_on_loaded_trace_set(
    *,
    actor,
    all_cooked_time: list,
    all_cooked_bw: list,
    all_file_names: list[str],
    seed: int,
    util_mod,
    env_mod,
    on_trace_end: Callable[[], None] | None,
    on_log: Callable[[dict[str, float | int], int], None] | None,
) -> dict[str, Any]:
    net_env = env_mod.Environment(all_cooked_time=all_cooked_time, all_cooked_bw=all_cooked_bw, random_seed=int(seed))

    import numpy as np

    state = np.zeros((int(util_mod.S_INFO), int(util_mod.S_LEN)), dtype=np.float64)
    last_bit_rate = int(util_mod.DEFAULT_QUALITY)
    bit_rate = int(util_mod.DEFAULT_QUALITY)

    per_trace_qoe_total: list[float] = []
    per_trace_regret_mean: list[float] = []
    per_trace_oracle_qoe_total: list[float] = []
    per_trace_qoe_gap_total: list[float] = []
    per_trace_names: list[str] = []

    step = 0
    episode_qoe_total = 0.0
    episode_regret_values: list[float] = []
    oracle_episode_qoe_total = None
    oracle_trace_action_values: list[int] = []
    oracle_trace_bitrate_kbps_values: list[float] = []
    oracle_trace_buffer_s_values: list[float] = []
    episode_return = 0.0

    video_count = 0
    oracle_k = int(util_mod.K_steps)

    agg: dict[str, Any] = {
        "env_steps": 0,
        "adv_trace_bw_mbps_values": [],
        "adv_realized_bw_mbps_values": [],
        "adv_throughput_mbps_values": [],
        "adv_delay_ms_values": [],
        "adv_buffer_s_values": [],
        "adv_rebuf_s_values": [],
        "adv_action_values": [],
        "adv_bitrate_kbps_values": [],
        "adv_entropy_values": [],
        "adv_max_prob_values": [],
        "adv_qoe_reward_values": [],
        "adv_qoe_quality_values": [],
        "adv_qoe_rebuf_s_values": [],
        "adv_qoe_smoothness_values": [],
        "oracle_reward_values": [],
        "oracle_quality_values": [],
        "oracle_rebuf_s_values": [],
        "oracle_smoothness_values": [],
        "oracle_gap_reward_values": [],
        "oracle_gap_quality_values": [],
        "oracle_gap_rebuf_s_values": [],
        "oracle_gap_smoothness_values": [],
        "oracle_action_values": [],
        "oracle_bitrate_kbps_values": [],
        "oracle_buffer_s_values": [],
        "oracle_trace_action_values": [],
        "oracle_trace_bitrate_kbps_values": [],
        "oracle_trace_buffer_s_values": [],
    }

    while True:
        chunk_idx = int(getattr(net_env, "video_chunk_counter", -1))
        active_trace_name = str(all_file_names[int(net_env.trace_idx)]) if all_file_names else "unknown"

        #* Snapshot trace bandwidth at the current Mahimahi pointer (Mbps).
        trace_bw_mbps = (
            float(net_env.cooked_bw[net_env.mahimahi_ptr]) if int(net_env.mahimahi_ptr) < len(net_env.cooked_bw) else float("nan")
        )

        if oracle_episode_qoe_total is None:
            oracle_env_snapshot = util_mod.copy_env(net_env, RANDOM_SEED=int(seed))
            (
                oracle_episode_qoe_total,
                oracle_trace_action_values,
                oracle_trace_bitrate_kbps_values,
                oracle_trace_buffer_s_values,
            ) = _oracle_episode_rollout_receding(
                env_snapshot=oracle_env_snapshot,
                last_br=int(util_mod.DEFAULT_QUALITY),
                k_steps=int(oracle_k),
                util_mod=util_mod,
            )

        (
            oracle_reward_mean,
            oracle_quality_mean,
            oracle_rebuf_s_mean,
            oracle_smoothness_mean,
            oracle_action_idx,
            oracle_action_kbps,
            oracle_buffer_s,
        ) = _oracle_k_step_qoe(
            env_snapshot=net_env,
            last_br=int(last_bit_rate),
            k_steps=int(oracle_k),
            seed=int(seed),
            util_mod=util_mod,
        )
        policy_reward_mean, policy_quality_mean, policy_rebuf_s_mean, policy_smoothness_mean = _policy_k_step_qoe(
            actor=actor,
            env_snapshot=net_env,
            state_snapshot=state,
            last_br=int(last_bit_rate),
            start_action=int(bit_rate),
            k_steps=int(oracle_k),
            seed=int(seed),
            util_mod=util_mod,
        )

        oracle_gap_reward = None
        oracle_gap_quality = None
        oracle_gap_rebuf_s = None
        oracle_gap_smoothness = None
        if (
            oracle_reward_mean is not None
            and oracle_quality_mean is not None
            and oracle_rebuf_s_mean is not None
            and oracle_smoothness_mean is not None
            and policy_reward_mean is not None
            and policy_quality_mean is not None
            and policy_rebuf_s_mean is not None
            and policy_smoothness_mean is not None
        ):
            oracle_gap_reward = float(oracle_reward_mean) - float(policy_reward_mean)
            oracle_gap_quality = float(oracle_quality_mean) - float(policy_quality_mean)
            oracle_gap_rebuf_s = float(policy_rebuf_s_mean) - float(oracle_rebuf_s_mean)
            oracle_gap_smoothness = float(policy_smoothness_mean) - float(oracle_smoothness_mean)
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
        ) = net_env.get_video_chunk(int(bit_rate))

        reward = float(util_mod.calculate_reward(int(bit_rate), int(last_bit_rate), float(rebuf_s)))
        episode_qoe_total += float(reward)
        episode_return += float(reward)

        qoe_quality = float(util_mod.VIDEO_BIT_RATE[int(bit_rate)]) / float(util_mod.M_IN_K)
        qoe_rebuf_s = float(rebuf_s)
        qoe_smoothness = float(abs(util_mod.VIDEO_BIT_RATE[int(bit_rate)] - util_mod.VIDEO_BIT_RATE[int(last_bit_rate)])) / float(
            util_mod.M_IN_K
        )

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

        realized_bw_mbps = float(_avg_bandwidth_mbps(float(video_chunk_size), float(delay_ms), env_mod=env_mod))
        throughput_mbps = float(_throughput_mbps(float(video_chunk_size), float(delay_ms)))

        p = actor.predict(np.reshape(state, (1, int(util_mod.S_INFO), int(util_mod.S_LEN)))).astype(np.float64).reshape(-1)
        entropy = float(_entropy(p))
        max_prob = float(np.max(p)) if p.size > 0 else 0.0
        next_bit_rate = int(np.argmax(p))

        oracle_trace_action: int | None = None
        oracle_trace_bitrate_kbps: float | None = None
        oracle_trace_buffer_s: float | None = None
        if 0 <= int(chunk_idx) < len(oracle_trace_action_values):
            oracle_trace_action = int(oracle_trace_action_values[int(chunk_idx)])
            oracle_trace_bitrate_kbps = float(oracle_trace_bitrate_kbps_values[int(chunk_idx)])
            oracle_trace_buffer_s = float(oracle_trace_buffer_s_values[int(chunk_idx)])

        agg["env_steps"] += 1
        agg["adv_trace_bw_mbps_values"].append(float(trace_bw_mbps))
        agg["adv_realized_bw_mbps_values"].append(float(realized_bw_mbps))
        agg["adv_throughput_mbps_values"].append(float(throughput_mbps))
        agg["adv_delay_ms_values"].append(float(delay_ms))
        agg["adv_buffer_s_values"].append(float(buffer_s))
        agg["adv_rebuf_s_values"].append(float(rebuf_s))
        agg["adv_action_values"].append(float(bit_rate))
        agg["adv_bitrate_kbps_values"].append(float(util_mod.VIDEO_BIT_RATE[int(bit_rate)]))
        agg["adv_entropy_values"].append(float(entropy))
        agg["adv_max_prob_values"].append(float(max_prob))
        agg["adv_qoe_reward_values"].append(float(reward))
        agg["adv_qoe_quality_values"].append(float(qoe_quality))
        agg["adv_qoe_rebuf_s_values"].append(float(qoe_rebuf_s))
        agg["adv_qoe_smoothness_values"].append(float(qoe_smoothness))

        if oracle_gap_reward is not None:
            agg["oracle_reward_values"].append(float(oracle_reward_mean))
            agg["oracle_quality_values"].append(float(oracle_quality_mean))
            agg["oracle_rebuf_s_values"].append(float(oracle_rebuf_s_mean))
            agg["oracle_smoothness_values"].append(float(oracle_smoothness_mean))
            agg["oracle_gap_reward_values"].append(float(oracle_gap_reward))
            agg["oracle_gap_quality_values"].append(float(oracle_gap_quality))
            agg["oracle_gap_rebuf_s_values"].append(float(oracle_gap_rebuf_s))
            agg["oracle_gap_smoothness_values"].append(float(oracle_gap_smoothness))

        if oracle_action_idx is not None:
            agg["oracle_action_values"].append(float(oracle_action_idx))
            if oracle_action_kbps is not None:
                agg["oracle_bitrate_kbps_values"].append(float(oracle_action_kbps))
            if oracle_buffer_s is not None:
                agg["oracle_buffer_s_values"].append(float(oracle_buffer_s))

        if oracle_trace_action is not None:
            agg["oracle_trace_action_values"].append(float(oracle_trace_action))
            if oracle_trace_bitrate_kbps is not None:
                agg["oracle_trace_bitrate_kbps_values"].append(float(oracle_trace_bitrate_kbps))
            if oracle_trace_buffer_s is not None:
                agg["oracle_trace_buffer_s_values"].append(float(oracle_trace_buffer_s))

        if on_log is not None:
            metrics: dict[str, float | int] = {
                "env/step": int(step),
                "env/episode": int(video_count),
                "adv/trace": str(active_trace_name),
                "adv/trace_bw_mbps": float(trace_bw_mbps),
                "adv/action": int(bit_rate),
                "adv/action_next": int(next_bit_rate),
                "adv/bitrate_kbps": float(util_mod.VIDEO_BIT_RATE[int(bit_rate)]),
                "adv/video_chunk_size_bytes": float(video_chunk_size),
                "adv/delay_ms": float(delay_ms),
                "adv/sleep_time_ms": float(sleep_time_ms),
                "adv/buffer_s": float(buffer_s),
                "adv/rebuf_s": float(rebuf_s),
                "adv/realized_bw_mbps": float(realized_bw_mbps),
                "adv/throughput_mbps": float(throughput_mbps),
                "adv/qoe_reward": float(reward),
                "adv/qoe/quality": float(qoe_quality),
                "adv/qoe/rebuf_s": float(qoe_rebuf_s),
                "adv/qoe/smoothness": float(qoe_smoothness),
                "adv/entropy": float(entropy),
                "adv/max_prob": float(max_prob),
            }
            if oracle_action_idx is not None:
                metrics["adv/oracle/action"] = int(oracle_action_idx)
                if oracle_action_kbps is not None:
                    metrics["adv/oracle/bitrate_kbps"] = float(oracle_action_kbps)
                if oracle_buffer_s is not None:
                    metrics["adv/oracle/buffer_s"] = float(oracle_buffer_s)
            if oracle_trace_action is not None:
                metrics["adv/oracle_trace/action"] = int(oracle_trace_action)
                if oracle_trace_bitrate_kbps is not None:
                    metrics["adv/oracle_trace/bitrate_kbps"] = float(oracle_trace_bitrate_kbps)
                if oracle_trace_buffer_s is not None:
                    metrics["adv/oracle_trace/buffer_s"] = float(oracle_trace_buffer_s)
            if oracle_gap_reward is not None:
                metrics["adv/oracle/qoe_reward"] = float(oracle_reward_mean)
                metrics["adv/oracle/qoe/quality"] = float(oracle_quality_mean)
                metrics["adv/oracle/qoe/rebuf_s"] = float(oracle_rebuf_s_mean)
                metrics["adv/oracle/qoe/smoothness"] = float(oracle_smoothness_mean)
                metrics["adv/oracle_gap/qoe_reward"] = float(oracle_gap_reward)
                metrics["adv/oracle_gap/qoe/quality"] = float(oracle_gap_quality)
                metrics["adv/oracle_gap/qoe/rebuf_s"] = float(oracle_gap_rebuf_s)
                metrics["adv/oracle_gap/qoe/smoothness"] = float(oracle_gap_smoothness)
            on_log(metrics, int(step))

        last_bit_rate = int(bit_rate)
        bit_rate = int(next_bit_rate)
        step += 1

        if bool(end_of_video):
            if on_log is not None:
                regret_finite = _finite(episode_regret_values)
                regret_mean = float("nan") if not regret_finite else float(_mean(regret_finite))
                qoe_gap_total = (
                    float("nan")
                    if oracle_episode_qoe_total is None
                    else float(oracle_episode_qoe_total) - float(episode_qoe_total)
                )
                on_log(
                    {
                        "adv/episode/cumulative_reward": float(episode_return),
                        "env/episode": int(video_count),
                        "eval/oracle_trace_qoe_total": float("nan")
                        if oracle_episode_qoe_total is None
                        else float(oracle_episode_qoe_total),
                        "eval/qoe_gap_total": float(qoe_gap_total),
                        "eval/regret_mean": float(regret_mean),
                    },
                    int(step),
                )

            per_trace_names.append(str(active_trace_name))
            per_trace_qoe_total.append(float(episode_qoe_total))

            regret_finite = _finite(episode_regret_values)
            if regret_finite:
                per_trace_regret_mean.append(float(_mean(regret_finite)))
            else:
                per_trace_regret_mean.append(float("nan"))

            per_trace_oracle_qoe_total.append(float("nan") if oracle_episode_qoe_total is None else float(oracle_episode_qoe_total))
            if oracle_episode_qoe_total is None:
                per_trace_qoe_gap_total.append(float("nan"))
            else:
                per_trace_qoe_gap_total.append(float(oracle_episode_qoe_total) - float(episode_qoe_total))

            video_count += 1
            episode_qoe_total = 0.0
            episode_regret_values = []
            episode_return = 0.0
            state[:] = 0.0
            last_bit_rate = int(util_mod.DEFAULT_QUALITY)
            bit_rate = int(util_mod.DEFAULT_QUALITY)
            oracle_episode_qoe_total = None
            oracle_trace_action_values = []
            oracle_trace_bitrate_kbps_values = []
            oracle_trace_buffer_s_values = []

            if on_trace_end is not None:
                on_trace_end()

            if video_count >= len(all_file_names):
                break

    return {
        "trace_count": int(len(per_trace_names)),
        "trace_names": per_trace_names,
        "qoe_total": per_trace_qoe_total,
        "regret_mean": per_trace_regret_mean,
        "oracle_qoe_total": per_trace_oracle_qoe_total,
        "qoe_gap_total": per_trace_qoe_gap_total,
        "end2end_agg": dict(agg),
    }

 
def _eval_one_model(
    *,
    model_path: str,
    groups: list[dict[str, str]],
    seed: int,
    progress: bool,
    traces_per_group: int,
    wandb_enabled: bool,
    wandb_project: str,
    wandb_entity: str | None,
    wandb_group: str | None,
    wandb_tags: list[str],
    wandb_mode: str,
    wandb_name_prefix: str | None,
    wandb_log_every: int,
) -> list[dict[str, Any]]:
    #* Worker entrypoint for multiprocessing: evaluate one model across all trace groups.
    import numpy as np

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    try:
        import torch  # type: ignore

        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except Exception:
            pass
    except Exception:
        pass

    network, util_mod, env_mod = _load_pensieve_modules()

    model_path = str(model_path)
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    robust_suffix = _model_robustification_suffix(model_path)

    actor = network.Network(
        state_dim=[int(util_mod.S_INFO), int(util_mod.S_LEN)],
        action_dim=int(util_mod.A_DIM),
        learning_rate=float(util_mod.ACTOR_LR_RATE),
    )
    actor.load_model(str(model_path))

    wandb = None
    if bool(wandb_enabled):
        wandb = _try_import_wandb()
        if wandb is None:
            raise RuntimeError("* --wandb enabled but `wandb` is not importable; install `wandb` and retry")

    out_rows: list[dict[str, Any]] = []
    traces_done = 0
    total_traces = int(len(groups)) * int(traces_per_group)
    report_every = max(1, int(total_traces) // 20)
    last_report_t = time.perf_counter()

    for g in groups:
        trace_dir = str(g["trace_dir"])
        eval_trace = str(g["eval_trace"])
        attack_name = str(g["attack_name"])
        trace_victim_suffix = str(g["trace_victim_suffix"])

        all_cooked_time, all_cooked_bw, all_file_names = _load_trace_subset(
            trace_dir=str(trace_dir),
            num_traces=int(traces_per_group),
        )
        if not all_file_names:
            raise RuntimeError(f"* No traces found under: {trace_dir}")

        wandb_run = None
        if wandb is not None:
            safe_eval = str(eval_trace).replace("/", "__")
            name_parts = [str(model_name), str(safe_eval)]
            run_name = "__".join([p for p in name_parts if p])
            if wandb_name_prefix:
                run_name = str(wandb_name_prefix) + str(run_name)

            tags = list(wandb_tags)
            if str(attack_name) and str(attack_name) not in tags:
                tags.append(str(attack_name))
            if str(robust_suffix) and str(robust_suffix) not in tags:
                tags.append(str(robust_suffix))

            wandb_run = wandb.init(
                project=str(wandb_project),
                entity=wandb_entity,
                group=wandb_group,
                name=str(run_name),
                mode=str(wandb_mode),
                tags=tags,
                config={
                    "script": "scripts/eval_models_trace_groups.py",
                    "model_name": str(model_name),
                    "model_path": str(model_path),
                    "robustification_trace": str(robust_suffix),
                    "eval_trace": str(eval_trace),
                    "attack_name": str(attack_name),
                    "trace_victim_suffix": str(trace_victim_suffix),
                    "trace_dir": str(trace_dir),
                    "num_traces": int(len(all_file_names)),
                    "oracle_k_steps": int(util_mod.K_steps),
                    "seed": int(seed),
                },
            )

        def _on_trace_end() -> None:
            nonlocal traces_done, last_report_t
            traces_done += 1
            now = time.perf_counter()
            if traces_done >= total_traces or (traces_done % report_every == 0) or (now - last_report_t) >= 60.0:
                if bool(progress):
                    _eprint(f"* {model_name}: {traces_done}/{total_traces} traces")
                last_report_t = now

        results: dict[str, Any] | None = None
        try:
            def _on_log(metrics: dict[str, float | int], step: int) -> None:
                if wandb_run is None:
                    return
                if (int(step) % int(max(1, wandb_log_every))) != 0 and "adv/episode/cumulative_reward" not in metrics:
                    return
                wandb.log(dict(metrics), step=int(step))

            results = _eval_actor_on_loaded_trace_set(
                actor=actor,
                all_cooked_time=all_cooked_time,
                all_cooked_bw=all_cooked_bw,
                all_file_names=[str(x) for x in list(all_file_names)],
                seed=int(seed),
                util_mod=util_mod,
                env_mod=env_mod,
                on_trace_end=_on_trace_end if bool(progress) else None,
                on_log=_on_log if wandb_run is not None else None,
            )
        finally:
            if wandb_run is not None:
                agg = dict(results.get("end2end_agg", {})) if isinstance(results, dict) else {}
                total_steps = int(agg.get("env_steps", 0))
                wandb_run.summary["env/steps-sum"] = int(max(total_steps, 1))
                _write_agg_stats(
                    wandb_run.summary, "adv/qoe_reward", _finite(list(agg.get("adv_qoe_reward_values", []))), include_geo_mean=False
                )
                _write_agg_stats(
                    wandb_run.summary, "adv/qoe/quality", _finite(list(agg.get("adv_qoe_quality_values", []))), include_geo_mean=True
                )
                _write_agg_stats(
                    wandb_run.summary, "adv/qoe/rebuf_s", _finite(list(agg.get("adv_qoe_rebuf_s_values", []))), include_geo_mean=True
                )
                _write_agg_stats(
                    wandb_run.summary,
                    "adv/qoe/smoothness",
                    _finite(list(agg.get("adv_qoe_smoothness_values", []))),
                    include_geo_mean=True,
                )
                _write_agg_stats(wandb_run.summary, "adv/delay_ms", _finite(list(agg.get("adv_delay_ms_values", []))), include_geo_mean=True)
                _write_agg_stats(
                    wandb_run.summary, "adv/throughput_mbps", _finite(list(agg.get("adv_throughput_mbps_values", []))), include_geo_mean=True
                )
                _write_agg_stats(
                    wandb_run.summary, "adv/realized_bw_mbps", _finite(list(agg.get("adv_realized_bw_mbps_values", []))), include_geo_mean=True
                )
                _write_agg_stats(
                    wandb_run.summary, "adv/buffer_s", _finite(list(agg.get("adv_buffer_s_values", []))), include_geo_mean=True
                )
                _write_agg_stats(wandb_run.summary, "adv/rebuf_s", _finite(list(agg.get("adv_rebuf_s_values", []))), include_geo_mean=True)
                _write_agg_stats(wandb_run.summary, "adv/entropy", _finite(list(agg.get("adv_entropy_values", []))), include_geo_mean=True)
                _write_agg_stats(wandb_run.summary, "adv/max_prob", _finite(list(agg.get("adv_max_prob_values", []))), include_geo_mean=True)
                _write_agg_stats(
                    wandb_run.summary, "adv/oracle/bitrate_kbps", _finite(list(agg.get("oracle_bitrate_kbps_values", []))), include_geo_mean=True
                )
                _write_agg_stats(
                    wandb_run.summary, "adv/oracle/buffer_s", _finite(list(agg.get("oracle_buffer_s_values", []))), include_geo_mean=True
                )
                _write_agg_stats(
                    wandb_run.summary,
                    "adv/oracle_trace/bitrate_kbps",
                    _finite(list(agg.get("oracle_trace_bitrate_kbps_values", []))),
                    include_geo_mean=True,
                )
                _write_agg_stats(
                    wandb_run.summary,
                    "adv/oracle_trace/buffer_s",
                    _finite(list(agg.get("oracle_trace_buffer_s_values", []))),
                    include_geo_mean=True,
                )
                _write_agg_stats(
                    wandb_run.summary, "adv/oracle/qoe_reward", _finite(list(agg.get("oracle_reward_values", []))), include_geo_mean=False
                )
                _write_agg_stats(
                    wandb_run.summary, "adv/oracle/qoe/quality", _finite(list(agg.get("oracle_quality_values", []))), include_geo_mean=True
                )
                _write_agg_stats(
                    wandb_run.summary, "adv/oracle/qoe/rebuf_s", _finite(list(agg.get("oracle_rebuf_s_values", []))), include_geo_mean=True
                )
                _write_agg_stats(
                    wandb_run.summary,
                    "adv/oracle/qoe/smoothness",
                    _finite(list(agg.get("oracle_smoothness_values", []))),
                    include_geo_mean=True,
                )
                _write_agg_stats(
                    wandb_run.summary,
                    "adv/oracle_gap/qoe_reward",
                    _finite(list(agg.get("oracle_gap_reward_values", []))),
                    include_geo_mean=False,
                )
                _write_agg_stats(
                    wandb_run.summary,
                    "adv/oracle_gap/qoe/quality",
                    _finite(list(agg.get("oracle_gap_quality_values", []))),
                    include_geo_mean=False,
                )
                _write_agg_stats(
                    wandb_run.summary, "adv/oracle_gap/qoe/rebuf_s", _finite(list(agg.get("oracle_gap_rebuf_s_values", []))), include_geo_mean=False
                )
                _write_agg_stats(
                    wandb_run.summary,
                    "adv/oracle_gap/qoe/smoothness",
                    _finite(list(agg.get("oracle_gap_smoothness_values", []))),
                    include_geo_mean=False,
                )

                #* Match inference.py end2end summary behavior when SMT is not invoked.
                wandb_run.summary["smt/success-rate"] = 1.0
                wandb_run.summary["smt/applied-rate"] = 0.0
                wandb_run.summary["smt/complete-rate"] = 1.0
                wandb_run.summary["smt/skipped-rate"] = 0.0

                wandb_run.finish()

        per_trace_qoe_total = list(results.get("qoe_total", []))
        per_trace_regret_mean = list(results.get("regret_mean", []))
        per_trace_oracle_qoe_total = list(results.get("oracle_qoe_total", []))
        per_trace_qoe_gap_total = list(results.get("qoe_gap_total", []))

        qoe_vals = [float(x) for x in per_trace_qoe_total if math.isfinite(float(x))]
        regret_vals = _finite([float(x) for x in per_trace_regret_mean])
        oracle_qoe_vals = _finite([float(x) for x in per_trace_oracle_qoe_total])
        qoe_gap_vals = _finite([float(x) for x in per_trace_qoe_gap_total])

        out_rows.append(
            {
                "model_name": str(model_name),
                "model_path": str(model_path),
                "robustification_trace": str(robust_suffix),
                "eval_trace": str(eval_trace),
                "attack_name": str(attack_name),
                "trace_victim_suffix": str(trace_victim_suffix),
                "trace_dir": str(trace_dir),
                "num_traces": int(len(per_trace_qoe_total)),
                "num_traces_regret_valid": int(len(regret_vals)),
                "qoe_total_mean": float(_mean(qoe_vals)),
                "qoe_total_p50": float(_percentile(qoe_vals, 50)),
                "qoe_total_p95": float(_percentile(qoe_vals, 95)),
                "qoe_total_sum": float(np.sum(np.asarray(qoe_vals, dtype=np.float64))) if qoe_vals else float("nan"),
                "regret_mean": float(_mean(regret_vals)),
                "regret_p50": float(_percentile(regret_vals, 50)),
                "regret_p95": float(_percentile(regret_vals, 95)),
                "regret_sum": float(np.sum(np.asarray(regret_vals, dtype=np.float64))) if regret_vals else float("nan"),
                "oracle_qoe_total_mean": float(_mean(oracle_qoe_vals)),
                "oracle_qoe_total_p50": float(_percentile(oracle_qoe_vals, 50)),
                "oracle_qoe_total_p95": float(_percentile(oracle_qoe_vals, 95)),
                "oracle_qoe_total_sum": float(np.sum(np.asarray(oracle_qoe_vals, dtype=np.float64)))
                if oracle_qoe_vals
                else float("nan"),
                "qoe_gap_mean": float(_mean(qoe_gap_vals)),
                "qoe_gap_p50": float(_percentile(qoe_gap_vals, 50)),
                "qoe_gap_p95": float(_percentile(qoe_gap_vals, 95)),
                "qoe_gap_sum": float(np.sum(np.asarray(qoe_gap_vals, dtype=np.float64))) if qoe_gap_vals else float("nan"),
            }
        )

    return out_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate original + robustified models on clean and adversarial trace groups; write aggregate CSV."
    )
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--robust-model-dir", type=str, required=True)
    parser.add_argument("--adv-trace-dir", type=str, required=True)
    parser.add_argument("--clean-trace-dir", type=str, required=True)
    parser.add_argument("--out-csv", type=str, default="robustification_eval.csv")
    parser.add_argument("--jobs", type=int, default=0, help="* Max parallel model eval workers (0 = auto)")
    parser.add_argument(
        "--traces-per-group",
        type=int,
        default=124,
        help="* Use only the first N traces from each trace group (error if a group has fewer than N traces).",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="* Print progress info to stderr",
    )
    parser.add_argument("--wandb", action="store_true", help="* Log end2end-style per-step metrics to wandb (one run per model x trace group).")
    parser.add_argument("--wandb-project", type=str, default="pensieve-eval")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None, help="* wandb group name (default: auto timestamp).")
    parser.add_argument("--wandb-tags", type=str, default="")
    parser.add_argument("--wandb-mode", type=str, default="online")
    parser.add_argument("--wandb-name-prefix", type=str, default=None)
    parser.add_argument("--wandb-log-every", type=int, default=1)
    args = parser.parse_args()

    repo_root = _repo_root()
    network, util_mod, _env_mod = _load_pensieve_modules()

    base_model = _resolve_path(str(args.base_model))
    robust_model_dir = _resolve_path(str(args.robust_model_dir))
    adv_trace_dir = _resolve_path(str(args.adv_trace_dir))
    clean_trace_dir = _resolve_path(str(args.clean_trace_dir))
    out_csv = _resolve_path(str(args.out_csv))

    if not os.path.isfile(base_model):
        raise FileNotFoundError(f"* --base-model not found: {base_model}")
    if not os.path.isdir(robust_model_dir):
        raise FileNotFoundError(f"* --robust-model-dir not found: {robust_model_dir}")
    if not os.path.isdir(adv_trace_dir):
        raise FileNotFoundError(f"* --adv-trace-dir not found: {adv_trace_dir}")
    if not os.path.isdir(clean_trace_dir):
        raise FileNotFoundError(f"* --clean-trace-dir not found: {clean_trace_dir}")

    robust_models = _list_model_files(robust_model_dir)
    if not robust_models:
        raise RuntimeError(f"* No model files found under --robust-model-dir: {robust_model_dir}")

    adv_groups = _discover_adv_trace_groups(adv_trace_dir)
    if not adv_groups:
        raise RuntimeError(
            "* No adversarial trace groups found. Expected structure: <adv-trace-dir>/<attack-name>/<model-suffix>/*"
        )

    #* Clean traces are evaluated as a separate trace type.
    groups: list[TraceGroup] = [
        TraceGroup(eval_trace="clean", attack_name="clean", trace_victim_suffix="clean", trace_dir=str(clean_trace_dir))
    ]
    groups.extend(list(adv_groups))

    groups_payload: list[dict[str, str]] = []
    for g in groups:
        #* Fail fast on insufficient trace counts.
        trace_files = _list_trace_files(str(g.trace_dir))
        if len(trace_files) < int(args.traces_per_group):
            raise RuntimeError(
                f"* Trace group has only {len(trace_files)} trace(s) (< {int(args.traces_per_group)}): {g.eval_trace} ({g.trace_dir})"
            )
        groups_payload.append(
            {
                "eval_trace": str(g.eval_trace),
                "attack_name": str(g.attack_name),
                "trace_victim_suffix": str(g.trace_victim_suffix),
                "trace_dir": str(g.trace_dir),
            }
        )

    if bool(args.progress):
        _eprint(f"* Found {len(robust_models)} robust model(s) under: {robust_model_dir}")
        _eprint(f"* Found {len(adv_groups)} adversarial trace group(s) under: {adv_trace_dir}")
        _eprint(f"* Total eval trace groups (incl. clean): {len(groups)}")

    wandb_tags = [t.strip() for t in str(args.wandb_tags).split(",") if t.strip()]
    wandb_group = str(args.wandb_group) if args.wandb_group else None
    if bool(args.wandb) and wandb_group is None:
        wandb_group = f"eval_models_trace_groups_{time.strftime('%Y%m%d-%H%M%S')}"

    all_models = [base_model] + list(robust_models)
    jobs = int(args.jobs)
    if jobs <= 0:
        jobs = min(int(os.cpu_count() or 1), int(len(all_models)))

    out_rows: list[dict[str, Any]] = []
    if jobs <= 1:
        for model_idx, model_path in enumerate(all_models):
            if bool(args.progress):
                _eprint(f"\n* [model {model_idx + 1}/{len(all_models)}] {os.path.basename(str(model_path))}")
            out_rows.extend(
                _eval_one_model(
                    model_path=str(model_path),
                    groups=groups_payload,
                    seed=int(util_mod.RANDOM_SEED),
                    progress=bool(args.progress),
                    traces_per_group=int(args.traces_per_group),
                    wandb_enabled=bool(args.wandb),
                    wandb_project=str(args.wandb_project),
                    wandb_entity=args.wandb_entity,
                    wandb_group=wandb_group,
                    wandb_tags=list(wandb_tags),
                    wandb_mode=str(args.wandb_mode),
                    wandb_name_prefix=args.wandb_name_prefix,
                    wandb_log_every=int(args.wandb_log_every),
                )
            )
    else:
        if bool(args.progress):
            _eprint(f"* Parallel eval: {len(all_models)} model(s) with jobs={jobs}")
        ctx = mp.get_context("spawn")
        future_to_model: dict[concurrent.futures.Future, str] = {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=int(jobs), mp_context=ctx) as ex:
            for idx, model_path in enumerate(all_models):
                fut = ex.submit(
                    _eval_one_model,
                    model_path=str(model_path),
                    groups=groups_payload,
                    seed=int(util_mod.RANDOM_SEED),
                    progress=bool(args.progress),
                    traces_per_group=int(args.traces_per_group),
                    wandb_enabled=bool(args.wandb),
                    wandb_project=str(args.wandb_project),
                    wandb_entity=args.wandb_entity,
                    wandb_group=wandb_group,
                    wandb_tags=list(wandb_tags),
                    wandb_mode=str(args.wandb_mode),
                    wandb_name_prefix=args.wandb_name_prefix,
                    wandb_log_every=int(args.wandb_log_every),
                )
                future_to_model[fut] = str(model_path)
                if bool(args.progress):
                    _eprint(f"* Submitted {idx + 1}/{len(all_models)}: {os.path.basename(str(model_path))}")

            done = 0
            for fut in concurrent.futures.as_completed(list(future_to_model.keys())):
                model_path = future_to_model[fut]
                try:
                    rows = fut.result()
                except Exception as e:
                    _eprint(f"* FAILED: {os.path.basename(model_path)} ({e})")
                    raise
                out_rows.extend(list(rows))
                done += 1
                if bool(args.progress):
                    _eprint(f"* Completed {done}/{len(all_models)}: {os.path.basename(model_path)}")

    out_rows.sort(key=lambda r: (str(r.get("model_name", "")), str(r.get("eval_trace", ""))))

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    fieldnames = [
        "model_name",
        "model_path",
        "robustification_trace",
        "eval_trace",
        "attack_name",
        "trace_victim_suffix",
        "trace_dir",
        "num_traces",
        "num_traces_regret_valid",
        "qoe_total_mean",
        "qoe_total_p50",
        "qoe_total_p95",
        "qoe_total_sum",
        "regret_mean",
        "regret_p50",
        "regret_p95",
        "regret_sum",
        "oracle_qoe_total_mean",
        "oracle_qoe_total_p50",
        "oracle_qoe_total_p95",
        "oracle_qoe_total_sum",
        "qoe_gap_mean",
        "qoe_gap_p50",
        "qoe_gap_p95",
        "qoe_gap_sum",
    ]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in out_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"* Wrote CSV: {out_csv}")


if __name__ == "__main__":
    main()
