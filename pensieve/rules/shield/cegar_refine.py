#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any


"""
Example usage:
python rules/shield/cegar_refine.py --base-model ppo/nn_model_ep_426300.pth \
    --clean-traces-dir Pensieve-PPO/src/test --iters 3 --wandb
"""

ONLINE_ADV_BW_FLOOR_MBPS = 1e-6


def _effective_online_bw_min(bw_min: float) -> float:
    if not math.isfinite(float(bw_min)):
        raise ValueError("* --bw-min must be finite")
    return max(float(bw_min), float(ONLINE_ADV_BW_FLOOR_MBPS))


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


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


def _resolve_path(repo_relative_or_abs: str) -> str:
    expanded = os.path.expandvars(os.path.expanduser(str(repo_relative_or_abs)))
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


def _fmt_tag_intlike(x: float) -> str:
    if float(x).is_integer():
        return str(int(round(float(x))))
    s = str(float(x))
    if "e" in s or "E" in s:
        s = f"{float(x):.6f}"
    s = s.rstrip("0").rstrip(".")
    return s.replace("-", "m").replace(".", "p")


def _fmt_tag_keep_p0(x: float) -> str:
    #* Mirror the bash `${VAR//./p}` behavior for clamp tags (e.g., 1.0 -> 1p0).
    if float(x).is_integer():
        s = f"{float(x):.1f}"
    else:
        s = str(float(x))
        if "e" in s or "E" in s:
            s = f"{float(x):.6f}"
        s = s.rstrip("0").rstrip(".")
    return s.replace("-", "m").replace(".", "p")


def _run(cmd: list[str], *, cwd: str) -> None:
    print("* cmd:", " ".join([str(x) for x in cmd]), flush=True)
    subprocess.run([str(x) for x in cmd], cwd=str(cwd), check=True)


def _latest_file(pattern: str) -> str | None:
    files = glob.glob(pattern)
    if not files:
        return None
    files = [str(f) for f in files]
    files.sort(key=lambda p: float(os.path.getmtime(p)))
    return str(files[-1])


def _count_traces(trace_dir: str) -> int:
    n = 0
    for name in os.listdir(str(trace_dir)):
        if str(name).startswith("."):
            continue
        if not str(name).endswith(".trace"):
            continue
        path = os.path.join(str(trace_dir), str(name))
        if os.path.isfile(path):
            n += 1
    return int(n)


def _list_trace_files(trace_dir: str) -> list[str]:
    out: list[str] = []
    for name in sorted(os.listdir(str(trace_dir))):
        if str(name).startswith("."):
            continue
        if str(name).endswith(".json"):
            continue
        path = os.path.join(str(trace_dir), str(name))
        if not os.path.isfile(path):
            continue
        out.append(str(name))
    return out


def _finite(values: list[float]) -> list[float]:
    return [float(x) for x in values if math.isfinite(float(x))]


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
    delay_ms_f = float(delay_ms)
    rtt_free_delay_ms = float(max(delay_ms_f - float(link_rtt_ms), 0.0))
    if rtt_free_delay_ms <= 0.0:
        return 0.0
    payload_portion_f = float(payload_portion)
    if payload_portion_f <= 0.0:
        return 0.0
    return float(video_chunk_size_bytes) * 8.0 / (rtt_free_delay_ms * 1000.0) / payload_portion_f


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


def _oracle_episode_rollout_receding(
    *,
    env_snapshot: Any,
    last_br: int,
    k_steps: int,
    util_mod: Any,
) -> tuple[float | None, list[int], list[float], list[float], list[float], list[float], list[float]]:
    k = int(k_steps)
    if k <= 0:
        return None, [], [], [], [], [], []

    total_chunks = int(getattr(util_mod, "CHUNK_TIL_VIDEO_END_CAP", 48))
    if total_chunks <= 0:
        total_chunks = 48

    env_sim = env_snapshot
    last = int(last_br)
    total_reward = 0.0
    action_values: list[int] = []
    bitrate_kbps_values: list[float] = []
    buffer_s_values: list[float] = []
    delay_ms_values: list[float] = []
    chunk_size_bytes_values: list[float] = []
    trace_bw_mbps_values: list[float] = []

    while True:
        trace_bw = (
            float(env_sim.cooked_bw[env_sim.mahimahi_ptr])
            if int(getattr(env_sim, "mahimahi_ptr", -1)) >= 0 and int(env_sim.mahimahi_ptr) < len(env_sim.cooked_bw)
            else float("nan")
        )
        remaining = int(total_chunks) - int(getattr(env_sim, "video_chunk_counter", 0))
        k_eff = int(min(int(k), int(max(1, remaining))))
        best_r, best_seq, _worst_r, _worst_seq = util_mod.find_optimal_k_step_reward_oracle(env_sim, int(last), int(k_eff))
        if best_r is None or best_seq is None:
            return None, action_values, bitrate_kbps_values, buffer_s_values, delay_ms_values, chunk_size_bytes_values, trace_bw_mbps_values

        a0 = int(best_seq[0])
        try:
            delay_ms, _sleep_ms, buffer_s, rebuf_s, chunk_size, _next_sizes, end, _remain = env_sim.get_video_chunk(int(a0))
        except Exception:
            return None, action_values, bitrate_kbps_values, buffer_s_values, delay_ms_values, chunk_size_bytes_values, trace_bw_mbps_values

        total_reward += float(util_mod.calculate_reward(int(a0), int(last), float(rebuf_s)))
        action_values.append(int(a0))
        bitrate_kbps_values.append(float(util_mod.VIDEO_BIT_RATE[int(a0)]))
        buffer_s_values.append(float(buffer_s))
        delay_ms_values.append(float(delay_ms))
        chunk_size_bytes_values.append(float(chunk_size))
        trace_bw_mbps_values.append(float(trace_bw))
        last = int(a0)
        if bool(end):
            break

    return (
        float(total_reward),
        action_values,
        bitrate_kbps_values,
        buffer_s_values,
        delay_ms_values,
        chunk_size_bytes_values,
        trace_bw_mbps_values,
    )


def _oracle_k_step_reward_mean(
    *,
    env_snapshot: Any,
    last_br: int,
    k_steps: int,
    util_mod: Any,
) -> tuple[float | None, int | None, float | None, float | None, float | None, float | None, float | None]:
    k = int(k_steps)
    if k <= 0:
        return None, None, None, None, None, None, None

    best_r, best_seq, _worst_r, _worst_seq = util_mod.find_optimal_k_step_reward_oracle(env_snapshot, int(last_br), int(k))
    if best_r is None or best_seq is None or len(best_seq) == 0:
        return None, None, None, None, None, None, None

    a0 = int(best_seq[0])
    oracle_action_kbps = float(util_mod.VIDEO_BIT_RATE[int(a0)])

    env_first = util_mod.copy_env(env_snapshot, RANDOM_SEED=int(getattr(util_mod, "RANDOM_SEED", 42)))
    oracle_buffer_s = None
    oracle_delay_ms = None
    oracle_chunk_size_bytes = None
    oracle_trace_bw_mbps = (
        float(env_first.cooked_bw[env_first.mahimahi_ptr])
        if int(getattr(env_first, "mahimahi_ptr", -1)) >= 0 and int(env_first.mahimahi_ptr) < len(env_first.cooked_bw)
        else float("nan")
    )
    try:
        delay_ms, _sleep_ms, buffer_s, _rebuf_s, chunk_size, _next_sizes, end, _remain = env_first.get_video_chunk(int(a0))
        oracle_buffer_s = float(buffer_s)
        oracle_delay_ms = float(delay_ms)
        oracle_chunk_size_bytes = float(chunk_size)
    except Exception:
        return None, None, None, None, None, None, None

    if bool(end) and int(k) > 1:
        return None, None, None, None, None, None, None

    return (
        float(best_r) / float(k),
        int(a0),
        float(oracle_action_kbps),
        float(oracle_buffer_s),
        float("nan") if oracle_delay_ms is None else float(oracle_delay_ms),
        float("nan") if oracle_chunk_size_bytes is None else float(oracle_chunk_size_bytes),
        float(oracle_trace_bw_mbps),
    )


def _policy_k_step_reward_mean(
    actor: Any,
    *,
    env_snapshot: Any,
    state_snapshot: Any,
    last_br: int,
    start_action: int,
    k_steps: int,
    action_selection: str,
    rng_seed: int,
    shield: Any | None,
    util_mod: Any,
) -> float | None:
    k = int(k_steps)
    if k <= 0:
        return None

    import numpy as np

    env_sim = util_mod.copy_env(env_snapshot, RANDOM_SEED=int(rng_seed))
    state_sim = np.array(state_snapshot, copy=True)
    rng = np.random.RandomState(int(rng_seed))

    total_reward = 0.0
    last = int(last_br)
    a = int(start_action)
    shield_snapshot = None if shield is None or not hasattr(shield, "snapshot_runtime_state") else shield.snapshot_runtime_state()

    try:
        for i in range(int(k)):
            try:
                delay_ms, _sleep_ms, buffer_s, rebuf_s, chunk_size, next_sizes, end, remain = env_sim.get_video_chunk(int(a))
            except Exception:
                return None

            #* Match the oracle validity rule: sequences that end early are invalid for K-step comparison.
            if bool(end) and i < int(k) - 1:
                return None

            total_reward += float(util_mod.calculate_reward(int(a), int(last), float(rebuf_s)))
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
                if shield is not None:
                    a, _stats = shield.select_action(
                        p=p,
                        state=state_sim,
                        last_bitrate_idx=int(last),
                        video_chunk_size_bytes=int(chunk_size),
                        action_selection=str(action_selection),
                        rng=rng,
                    )
                else:
                    if str(action_selection) == "argmax":
                        a = int(np.argmax(p))
                    elif str(action_selection) == "gumbel":
                        noise = rng.gumbel(size=p.shape)
                        a = int(np.argmax(np.log(p + 1e-12) + noise))
                    else:
                        raise ValueError(f"* Unknown action_selection: {action_selection}")
    finally:
        if shield is not None and hasattr(shield, "restore_runtime_state"):
            shield.restore_runtime_state(shield_snapshot)

    return float(total_reward) / float(k)


def _try_import_wandb() -> Any | None:
    try:
        import wandb  # type: ignore

        if not hasattr(wandb, "init"):
            raise ImportError("* `wandb` import resolved to a non-package directory; install the wandb pip package")
        return wandb
    except Exception:
        return None


_WORKER_ACTOR: Any | None = None
_WORKER_UTIL: Any | None = None
_WORKER_ENV: Any | None = None
_WORKER_SHIELD: Any | None = None
_WORKER_ACTION_SELECTION: str | None = None
_WORKER_ORACLE_K: int | None = None
_WORKER_SEED: int | None = None
_WORKER_MODEL_PATH: str | None = None


def _worker_init(
    model_path: str,
    shield_min_rules_file: str | None,
    shield_max_rules_file: str | None,
    action_selection: str,
    oracle_k: int,
    seed: int,
) -> None:
    global _WORKER_ACTOR, _WORKER_UTIL, _WORKER_ENV, _WORKER_SHIELD, _WORKER_ACTION_SELECTION, _WORKER_ORACLE_K, _WORKER_SEED, _WORKER_MODEL_PATH

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
        shield = ShieldBounds(min_rules=min_rules, max_rules=max_rules)

    _WORKER_ACTOR = actor
    _WORKER_UTIL = util_mod
    _WORKER_ENV = env_mod
    _WORKER_SHIELD = shield
    _WORKER_ACTION_SELECTION = str(action_selection)
    _WORKER_ORACLE_K = int(oracle_k)
    _WORKER_SEED = int(seed)
    _WORKER_MODEL_PATH = str(model_path)


def _eval_one_trace_file(
    *,
    trace_type: str,
    trace_dir: str,
    trace_name: str,
    trace_idx: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if (
        _WORKER_ACTOR is None
        or _WORKER_UTIL is None
        or _WORKER_ENV is None
        or _WORKER_ACTION_SELECTION is None
        or _WORKER_ORACLE_K is None
        or _WORKER_SEED is None
        or _WORKER_MODEL_PATH is None
    ):
        raise RuntimeError("* worker is not initialized")

    actor = _WORKER_ACTOR
    util_mod = _WORKER_UTIL
    env_mod = _WORKER_ENV
    shield = _WORKER_SHIELD
    action_selection = str(_WORKER_ACTION_SELECTION)
    oracle_k = int(_WORKER_ORACLE_K)
    seed = int(_WORKER_SEED)
    model_path = str(_WORKER_MODEL_PATH)
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
    (
        oracle_episode_qoe_total,
        oracle_trace_action_values,
        oracle_trace_bitrate_kbps_values,
        oracle_trace_buffer_s_values,
        oracle_trace_delay_ms_values,
        oracle_trace_chunk_size_bytes_values,
        oracle_trace_trace_bw_mbps_values,
    ) = _oracle_episode_rollout_receding(
        env_snapshot=oracle_env_snapshot,
        last_br=int(util_mod.DEFAULT_QUALITY),
        k_steps=int(oracle_k),
        util_mod=util_mod,
    )

    per_step_rows: list[dict[str, Any]] = []
    episode_return = 0.0
    episode_regret_values: list[float] = []
    episode_steps = 0

    rng_episode = np.random.RandomState(int(seed) + int(trace_idx))

    while True:
        chunk_idx = int(getattr(env_inst, "video_chunk_counter", -1))
        trace_bw = (
            float(env_inst.cooked_bw[env_inst.mahimahi_ptr])
            if int(env_inst.mahimahi_ptr) < len(env_inst.cooked_bw)
            else float("nan")
        )

        (
            oracle_reward_mean,
            oracle_action_idx,
            oracle_action_kbps,
            oracle_buffer_s,
            oracle_delay_ms,
            oracle_chunk_size_bytes,
            oracle_trace_bw,
        ) = _oracle_k_step_reward_mean(
            env_snapshot=env_inst,
            last_br=int(last_bit_rate),
            k_steps=int(oracle_k),
            util_mod=util_mod,
        )

        policy_reward_mean = _policy_k_step_reward_mean(
            actor,
            env_snapshot=env_inst,
            state_snapshot=state,
            last_br=int(last_bit_rate),
            start_action=int(bit_rate),
            k_steps=int(oracle_k),
            action_selection=str(action_selection),
            rng_seed=int(seed) + int(trace_idx) * 10_000 + int(chunk_idx),
            shield=shield,
            util_mod=util_mod,
        )

        regret = None
        if oracle_reward_mean is not None and policy_reward_mean is not None:
            regret = float(oracle_reward_mean) - float(policy_reward_mean)
            episode_regret_values.append(float(regret))
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

        p = actor.predict(np.reshape(state, (1, int(util_mod.S_INFO), int(util_mod.S_LEN)))).astype(np.float64).reshape(-1)
        if str(action_selection) == "argmax":
            next_bit_rate = int(np.argmax(p))
        elif str(action_selection) == "gumbel":
            noise = rng_episode.gumbel(size=p.shape)
            next_bit_rate = int(np.argmax(np.log(p + 1e-12) + noise))
        else:
            raise ValueError(f"* Unknown action_selection: {action_selection}")

        shield_stats: dict[str, float | int] = {}
        if shield is not None and int(chunk_idx) >= 0:
            next_bit_rate, shield_stats = shield.select_action(
                p=p,
                state=state,
                last_bitrate_idx=int(bit_rate),
                video_chunk_size_bytes=int(video_chunk_size),
                action_selection=str(action_selection),
                rng=rng_episode,
            )

        oracle_trace_action: int | None = None
        oracle_trace_bitrate_kbps: float | None = None
        oracle_trace_buffer_s: float | None = None
        oracle_trace_delay_ms: float | None = None
        oracle_trace_chunk_size_bytes: float | None = None
        oracle_trace_trace_bw: float | None = None
        if 0 <= int(chunk_idx) < len(oracle_trace_action_values):
            oracle_trace_action = int(oracle_trace_action_values[int(chunk_idx)])
            oracle_trace_bitrate_kbps = float(oracle_trace_bitrate_kbps_values[int(chunk_idx)])
            oracle_trace_buffer_s = float(oracle_trace_buffer_s_values[int(chunk_idx)])
            oracle_trace_delay_ms = float(oracle_trace_delay_ms_values[int(chunk_idx)])
            oracle_trace_chunk_size_bytes = float(oracle_trace_chunk_size_bytes_values[int(chunk_idx)])
            oracle_trace_trace_bw = float(oracle_trace_trace_bw_mbps_values[int(chunk_idx)])

        per_step_row: dict[str, Any] = {
            "trace_type": str(trace_type),
            "trace_dir": str(trace_dir),
            "model_path": str(model_path),
            "seed": int(seed),
            "oracle_k": int(oracle_k),
            "trace_name": str(trace_name),
            "trace_idx": int(trace_idx),
            "chunk_idx": int(chunk_idx),
            "video_chunk_remain": int(video_chunk_remain),
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
            "policy_action_idx": int(bit_rate),
            "policy_action_kbps": int(util_mod.VIDEO_BIT_RATE[int(bit_rate)]),
            "last_action_idx": int(last_bit_rate),
            "last_action_kbps": int(util_mod.VIDEO_BIT_RATE[int(last_bit_rate)]),
            "delay_ms": float(delay_ms),
            "throughput_mbps": float(_throughput_mbps(float(video_chunk_size), float(delay_ms))),
            "buffer_s": float(buffer_s),
            "rebuf_s": float(rebuf_s),
            "video_chunk_size_bytes": int(video_chunk_size),
            "qoe_step": float(qoe_step),
            "oracle_reward_mean": float("nan") if oracle_reward_mean is None else float(oracle_reward_mean),
            "policy_reward_mean": float("nan") if policy_reward_mean is None else float(policy_reward_mean),
            "regret": float("nan") if regret is None else float(regret),
        }

        if oracle_action_idx is not None:
            per_step_row["adv/oracle/action"] = int(oracle_action_idx)
        if oracle_action_kbps is not None:
            per_step_row["adv/oracle/bitrate_kbps"] = float(oracle_action_kbps)
        if oracle_buffer_s is not None:
            per_step_row["adv/oracle/buffer_s"] = float(oracle_buffer_s)
        if oracle_delay_ms is not None:
            per_step_row["adv/oracle/delay_ms"] = float(oracle_delay_ms)
        if oracle_chunk_size_bytes is not None:
            per_step_row["adv/oracle/video_chunk_size_bytes"] = float(oracle_chunk_size_bytes)
        if oracle_delay_ms is not None and oracle_chunk_size_bytes is not None:
            per_step_row["adv/oracle/throughput_mbps"] = float(_throughput_mbps(float(oracle_chunk_size_bytes), float(oracle_delay_ms)))
            per_step_row["adv/oracle/realized_bw_mbps"] = float(
                _avg_bandwidth_mbps(
                    float(oracle_chunk_size_bytes),
                    float(oracle_delay_ms),
                    link_rtt_ms=float(getattr(env_mod, "LINK_RTT", 0.0)),
                    payload_portion=float(getattr(env_mod, "PACKET_PAYLOAD_PORTION", 1.0)),
                )
            )
        if oracle_trace_bw is not None:
            per_step_row["adv/oracle/trace_bw_mbps"] = float(oracle_trace_bw)

        if oracle_trace_action is not None:
            per_step_row["adv/oracle_trace/action"] = int(oracle_trace_action)
        if oracle_trace_bitrate_kbps is not None:
            per_step_row["adv/oracle_trace/bitrate_kbps"] = float(oracle_trace_bitrate_kbps)
        if oracle_trace_buffer_s is not None:
            per_step_row["adv/oracle_trace/buffer_s"] = float(oracle_trace_buffer_s)
        if oracle_trace_delay_ms is not None:
            per_step_row["adv/oracle_trace/delay_ms"] = float(oracle_trace_delay_ms)
        if oracle_trace_chunk_size_bytes is not None:
            per_step_row["adv/oracle_trace/video_chunk_size_bytes"] = float(oracle_trace_chunk_size_bytes)
        if oracle_trace_delay_ms is not None and oracle_trace_chunk_size_bytes is not None:
            per_step_row["adv/oracle_trace/throughput_mbps"] = float(
                _throughput_mbps(float(oracle_trace_chunk_size_bytes), float(oracle_trace_delay_ms))
            )
            per_step_row["adv/oracle_trace/realized_bw_mbps"] = float(
                _avg_bandwidth_mbps(
                    float(oracle_trace_chunk_size_bytes),
                    float(oracle_trace_delay_ms),
                    link_rtt_ms=float(getattr(env_mod, "LINK_RTT", 0.0)),
                    payload_portion=float(getattr(env_mod, "PACKET_PAYLOAD_PORTION", 1.0)),
                )
            )
        if oracle_trace_trace_bw is not None:
            per_step_row["adv/oracle_trace/trace_bw_mbps"] = float(oracle_trace_trace_bw)

        if shield_stats:
            for k, v in shield_stats.items():
                if isinstance(v, (int, float)):
                    per_step_row[f"adv/{k}"] = v

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
        "policy_qoe_total": float(episode_return),
        "oracle_qoe_total": float("nan") if oracle_episode_qoe_total is None else float(oracle_episode_qoe_total),
        "qoe_gap_total": float("nan") if oracle_episode_qoe_total is None else float(oracle_episode_qoe_total) - float(episode_return),
        "regret_count_valid": int(len(regret_finite)),
        "regret_mean": float(regret_mean),
        "regret_p50": float("nan") if not regret_finite else float(_percentile(regret_finite, 50)),
        "regret_p95": float("nan") if not regret_finite else float(_percentile(regret_finite, 95)),
        "regret_sum": float("nan") if not regret_finite else float(sum(regret_finite)),
    }
    return per_trace_row, per_step_rows


def _write_csv(path: str, *, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _append_csv_row(path: str, *, fieldnames: list[str], row: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    has_header = os.path.isfile(path) and os.path.getsize(path) > 0
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not bool(has_header):
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def _csv_iter_values(path: str) -> set[int]:
    if not os.path.isfile(path) or os.path.getsize(path) <= 0:
        return set()
    out: set[int] = set()
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not isinstance(row, dict):
                continue
            it = row.get("iter", None)
            if it is None:
                continue
            try:
                out.add(int(it))
            except Exception:
                continue
    return out


_INTERRUPT_PENDING: dict[str, Any] = {}


def _flush_pending_results_on_interrupt() -> None:
    pending = dict(_INTERRUPT_PENDING)

    iter_stats_csv = pending.get("iter_stats_csv", None)
    iter_stats_fieldnames = pending.get("iter_stats_fieldnames", None)
    iter_row = pending.get("iter_row", None)

    if isinstance(iter_stats_csv, str) and isinstance(iter_stats_fieldnames, list) and isinstance(iter_row, dict):
        try:
            it = int(iter_row.get("iter", -1))
        except Exception:
            it = -1
        if it >= 0 and int(it) not in _csv_iter_values(str(iter_stats_csv)):
            _append_csv_row(str(iter_stats_csv), fieldnames=[str(x) for x in iter_stats_fieldnames], row=dict(iter_row))

    wandb_payload = pending.get("wandb_payload", None)
    wandb_ok = pending.get("wandb_ok", None)
    wandb_attempted = bool(pending.get("wandb_attempted", False))
    resume = bool(pending.get("resume", False))

    if bool(wandb_attempted):
        return
    if not isinstance(wandb_payload, dict):
        return
    if isinstance(wandb_ok, str) and bool(resume) and os.path.isfile(str(wandb_ok)):
        return

    wandb = _try_import_wandb()
    if wandb is None:
        return

    _log_wandb_iteration(wandb=wandb, **dict(wandb_payload))
    if isinstance(wandb_ok, str):
        with open(str(wandb_ok), "w", encoding="utf-8") as f:
            f.write(f"logged_at_unix_s={time.time():.6f}\n")


def _eval_dataset(
    *,
    trace_type: str,
    trace_dir: str,
    trace_num: int,
    model_path: str,
    shield_min_rules_file: str | None,
    shield_max_rules_file: str | None,
    oracle_k: int,
    action_selection: str,
    seed: int,
    out_trace_csv: str,
    out_step_csv: str,
    progress_prefix: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trace_files = _list_trace_files(str(trace_dir))
    if not trace_files:
        raise RuntimeError(f"* No trace files found under: {trace_dir}")
    if int(trace_num) > 0:
        if len(trace_files) < int(trace_num):
            raise RuntimeError(f"* trace_dir has only {len(trace_files)} trace(s) (< {trace_num}): {trace_dir}")
        trace_files = list(trace_files[: int(trace_num)])

    tasks = [(str(name), int(i)) for i, name in enumerate(trace_files)]

    per_trace_rows: list[dict[str, Any]] = []
    per_step_rows: list[dict[str, Any]] = []

    max_workers = max(1, min(os.cpu_count() or 1, 16))

    last_report_t = time.time()
    done = 0
    with ProcessPoolExecutor(
        max_workers=int(max_workers),
        initializer=_worker_init,
        initargs=(
            str(model_path),
            str(shield_min_rules_file) if shield_min_rules_file is not None else None,
            str(shield_max_rules_file) if shield_max_rules_file is not None else None,
            str(action_selection),
            int(oracle_k),
            int(seed),
        ),
    ) as ex:
        futs = [
            ex.submit(
                _eval_one_trace_file,
                trace_type=str(trace_type),
                trace_dir=str(trace_dir),
                trace_name=str(name),
                trace_idx=int(idx),
            )
            for name, idx in tasks
        ]
        for fut in as_completed(futs):
            tr, steps = fut.result()
            per_trace_rows.append(dict(tr))
            per_step_rows.extend([dict(r) for r in steps])
            done += 1
            now = time.time()
            if now - last_report_t >= 2.0:
                print(f"* {progress_prefix} traces_done={done}/{len(tasks)}", flush=True)
                last_report_t = now

    per_trace_rows.sort(key=lambda r: int(r.get("trace_idx", 0)))
    per_step_rows.sort(key=lambda r: (int(r.get("trace_idx", 0)), int(r.get("chunk_idx", 0))))

    trace_fieldnames = [
        "trace_type",
        "trace_dir",
        "model_path",
        "seed",
        "oracle_k",
        "trace_name",
        "trace_idx",
        "num_steps",
        "policy_qoe_total",
        "oracle_qoe_total",
        "qoe_gap_total",
        "regret_count_valid",
        "regret_mean",
        "regret_p50",
        "regret_p95",
        "regret_sum",
    ]
    step_fieldnames = [
        "trace_type",
        "trace_dir",
        "model_path",
        "seed",
        "oracle_k",
        "trace_name",
        "trace_idx",
        "chunk_idx",
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
        "adv/oracle/action",
        "adv/oracle/bitrate_kbps",
        "adv/oracle/buffer_s",
        "adv/oracle/trace_bw_mbps",
        "adv/oracle/video_chunk_size_bytes",
        "adv/oracle/delay_ms",
        "adv/oracle/realized_bw_mbps",
        "adv/oracle/throughput_mbps",
        "adv/oracle_trace/action",
        "adv/oracle_trace/bitrate_kbps",
        "adv/oracle_trace/buffer_s",
        "adv/oracle_trace/trace_bw_mbps",
        "adv/oracle_trace/video_chunk_size_bytes",
        "adv/oracle_trace/delay_ms",
        "adv/oracle_trace/realized_bw_mbps",
        "adv/oracle_trace/throughput_mbps",
        "policy_action_idx",
        "policy_action_kbps",
        "last_action_idx",
        "last_action_kbps",
        "delay_ms",
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

    _write_csv(str(out_trace_csv), fieldnames=trace_fieldnames, rows=per_trace_rows)
    _write_csv(str(out_step_csv), fieldnames=step_fieldnames, rows=per_step_rows)

    return per_trace_rows, per_step_rows


def _chunk_agg(per_step_rows: list[dict[str, Any]], *, key: str) -> dict[int, dict[str, float]]:
    by_idx: dict[int, list[float]] = {}
    for row in per_step_rows:
        idx = int(row.get("chunk_idx", -1))
        if idx < 0:
            continue
        v = row.get(str(key), None)
        if v is None:
            continue
        try:
            vf = float(v)
        except Exception:
            continue
        if not math.isfinite(vf):
            continue
        by_idx.setdefault(int(idx), []).append(float(vf))

    out: dict[int, dict[str, float]] = {}
    for idx, vals in sorted(by_idx.items(), key=lambda kv: int(kv[0])):
        finite = _finite(list(vals))
        out[int(idx)] = {
            "mean": float(_mean(finite)),
            "p50": float(_percentile(finite, 50)),
            "p95": float(_percentile(finite, 95)),
        }
    return out


def _trace_stats(per_trace_rows: list[dict[str, Any]], *, key: str) -> dict[str, float]:
    vals: list[float] = []
    for row in per_trace_rows:
        try:
            vals.append(float(row.get(str(key), float("nan"))))
        except Exception:
            vals.append(float("nan"))
    finite = _finite(vals)
    return {
        "mean": float(_mean(finite)),
        "p50": float(_percentile(finite, 50)),
        "p95": float(_percentile(finite, 95)),
    }


def _step_stats(per_step_rows: list[dict[str, Any]], *, key: str) -> dict[str, float]:
    vals: list[float] = []
    for row in per_step_rows:
        try:
            vals.append(float(row.get(str(key), float("nan"))))
        except Exception:
            vals.append(float("nan"))
    finite = _finite(vals)
    return {
        "mean": float(_mean(finite)),
        "p50": float(_percentile(finite, 50)),
        "p95": float(_percentile(finite, 95)),
    }


def _log_wandb_iteration(
    *,
    wandb: Any,
    project: str,
    entity: str | None,
    mode: str,
    tags: list[str],
    name: str,
    config: dict[str, Any],
    summary: dict[str, float],
    attack_traces: list[dict[str, Any]],
    clean_traces: list[dict[str, Any]],
    attack_steps: list[dict[str, Any]],
    clean_steps: list[dict[str, Any]],
) -> None:
    _INTERRUPT_PENDING["wandb_attempted"] = True
    _INTERRUPT_PENDING["wandb_in_progress"] = True
    run = wandb.init(project=str(project), entity=entity, name=str(name), mode=str(mode), tags=list(tags), config=dict(config))

    def _log_dataset(
        dataset: str,
        *,
        step_rows: list[dict[str, Any]],
        trace_rows: list[dict[str, Any]],
        global_step: int,
    ) -> int:
        #* Per-step time series: mimic `scripts/eval_oracle_regret.py` (end2end two-pass Phase-2 style).
        step_rows_sorted = list(step_rows)
        step_rows_sorted.sort(key=lambda r: (int(r.get("trace_idx", 0)), int(r.get("chunk_idx", 0))))
        per_trace_by_idx: dict[int, dict[str, Any]] = {int(r.get("trace_idx", -1)): r for r in trace_rows}

        keys = [
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
            "adv/oracle/action",
            "adv/oracle/bitrate_kbps",
            "adv/oracle/buffer_s",
            "adv/oracle/trace_bw_mbps",
            "adv/oracle/video_chunk_size_bytes",
            "adv/oracle/delay_ms",
            "adv/oracle/realized_bw_mbps",
            "adv/oracle/throughput_mbps",
            "adv/oracle_trace/action",
            "adv/oracle_trace/bitrate_kbps",
            "adv/oracle_trace/buffer_s",
            "adv/oracle_trace/trace_bw_mbps",
            "adv/oracle_trace/video_chunk_size_bytes",
            "adv/oracle_trace/delay_ms",
            "adv/oracle_trace/realized_bw_mbps",
            "adv/oracle_trace/throughput_mbps",
        ]

        cur_trace_idx: int | None = None
        for row in step_rows_sorted:
            trace_idx = int(row.get("trace_idx", -1))
            chunk_idx = int(row.get("chunk_idx", -1))

            if cur_trace_idx is None:
                cur_trace_idx = int(trace_idx)

            if trace_idx != cur_trace_idx:
                prev = per_trace_by_idx.get(int(cur_trace_idx), {})
                wandb.log(
                    {
                        f"{dataset}/adv/episode/cumulative_reward": float(prev.get("policy_qoe_total", float("nan"))),
                        f"{dataset}/env/episode": int(cur_trace_idx),
                        f"{dataset}/eval/oracle_trace_qoe_total": float(prev.get("oracle_qoe_total", float("nan"))),
                        f"{dataset}/eval/qoe_gap_total": float(prev.get("qoe_gap_total", float("nan"))),
                        f"{dataset}/eval/regret_mean": float(prev.get("regret_mean", float("nan"))),
                    },
                    step=int(global_step),
                )
                cur_trace_idx = int(trace_idx)

            metrics: dict[str, Any] = {
                f"{dataset}/env/step": int(global_step),
                f"{dataset}/env/episode": int(trace_idx),
                f"{dataset}/env/chunk_idx": int(chunk_idx),
            }
            for k in keys:
                if str(k) in row:
                    metrics[f"{dataset}/{k}"] = row[str(k)]

            wandb.log(metrics, step=int(global_step))
            global_step += 1

        if cur_trace_idx is not None:
            prev = per_trace_by_idx.get(int(cur_trace_idx), {})
            wandb.log(
                {
                    f"{dataset}/adv/episode/cumulative_reward": float(prev.get("policy_qoe_total", float("nan"))),
                    f"{dataset}/env/episode": int(cur_trace_idx),
                    f"{dataset}/eval/oracle_trace_qoe_total": float(prev.get("oracle_qoe_total", float("nan"))),
                    f"{dataset}/eval/qoe_gap_total": float(prev.get("qoe_gap_total", float("nan"))),
                    f"{dataset}/eval/regret_mean": float(prev.get("regret_mean", float("nan"))),
                },
                step=int(global_step),
            )
        return int(global_step)

    global_step = 0
    try:
        global_step = _log_dataset("attack", step_rows=attack_steps, trace_rows=attack_traces, global_step=int(global_step))
        global_step = _log_dataset("clean", step_rows=clean_steps, trace_rows=clean_traces, global_step=int(global_step))
    finally:
        #* Always persist the per-iteration summary, even if the run is interrupted mid-logging.
        try:
            for k, v in summary.items():
                run.summary[str(k)] = float(v)
            run.summary["env/steps-sum"] = int(max(int(global_step), 1))
        except Exception:
            pass
        try:
            run.finish()
        except Exception:
            pass
        _INTERRUPT_PENDING["wandb_in_progress"] = False


def main() -> None:
    parser = argparse.ArgumentParser(description="Iterative shield refinement (CEGAR-like) loop for Pensieve.")

    parser.add_argument("--base-model", required=True)
    parser.add_argument("--clean-traces-dir", required=True)
    parser.add_argument("--clean-trace-num", type=int, default=0)

    parser.add_argument("--iters", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--run-dir", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")

    parser.add_argument("--attacker-steps", type=int, default=600_000)
    parser.add_argument("--attacker-lr", type=float, default=3e-4)
    parser.add_argument("--bw-min", type=float, default=0.0)
    parser.add_argument("--bw-max", type=float, default=8.951518)
    parser.add_argument("--obs-history", type=int, default=10)
    parser.add_argument("--oracle-window", type=int, default=5)
    parser.add_argument("--smooth-penalty-scale", type=float, default=0.0)

    parser.add_argument("--k-steps", type=int, default=5)
    parser.add_argument("--action-selection", type=str, default="argmax", choices=["argmax", "gumbel"])

    parser.add_argument("--gap-pct-open-threshold", type=float, default=80.0)
    parser.add_argument("--oracle-best-min-abs", type=float, default=0.1)
    parser.add_argument("--regret-delta", type=float, default=10.0)
    parser.add_argument("--regret-delta-min-abs", type=float, default=0.3)
    parser.add_argument("--regret-delta-max-abs", type=float, default=1.0)
    parser.add_argument("--noncontiguous", type=str, default="best_action", choices=["best_action", "largest", "as_is"])

    parser.add_argument("--dt-backend", type=str, default="h2o", choices=["h2o"])
    parser.add_argument("--leaf-purity", type=float, default=0.999999)
    parser.add_argument("--h2o-ntrees", type=int, default=1)
    parser.add_argument("--h2o-max-depth", type=int, default=200)
    parser.add_argument("--h2o-min-rows", type=int, default=1)
    parser.add_argument("--h2o-sample-rate", type=float, default=1.0)
    parser.add_argument("--h2o-mtries", type=int, default=None)

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="pensieve-cegar-new")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default="online")
    parser.add_argument("--wandb-tags", type=str, default="")
    parser.add_argument("--wandb-name-prefix", type=str, default="cegar")

    args = parser.parse_args()

    online_bw_min = float(_effective_online_bw_min(float(args.bw_min)))
    bw_max = float(args.bw_max)
    if not math.isfinite(float(bw_max)):
        raise ValueError("* --bw-max must be finite")
    if float(bw_max) < float(online_bw_min):
        raise ValueError(f"* --bw-max must be >= effective --bw-min ({online_bw_min:g})")
    if float(online_bw_min) != float(args.bw_min):
        print(
            f"* Adjusted --bw-min from {float(args.bw_min):g} to {float(online_bw_min):g} for log-scale adversary mapping",
            flush=True,
        )

    base_model = _resolve_path(str(args.base_model))
    clean_traces_dir = _resolve_path(str(args.clean_traces_dir))
    if not os.path.isfile(base_model):
        raise FileNotFoundError(f"* base model not found: {base_model}")
    if not os.path.isdir(clean_traces_dir):
        raise FileNotFoundError(f"* clean traces dir not found: {clean_traces_dir}")

    run_dir = str(args.run_dir).strip()
    if not run_dir:
        run_dir = os.path.join(_repo_root(), "rules", "shield", "cegar_runs", time.strftime("%Y%m%d-%H%M%S"))
    run_dir = _resolve_path(run_dir)
    os.makedirs(run_dir, exist_ok=True)

    agg_dir = os.path.join(run_dir, "agg")
    os.makedirs(agg_dir, exist_ok=True)

    current_min_rules: str | None = None
    current_max_rules: str | None = None

    iter_stats_csv = os.path.join(run_dir, "cegar_iteration_stats.csv")
    iter_stats_fieldnames = [
        "iter",
        "attack_policy_qoe_mean",
        "attack_policy_qoe_p50",
        "attack_policy_qoe_p95",
        "attack_oracle_gap_mean",
        "attack_oracle_gap_p50",
        "attack_oracle_gap_p95",
        "attack_oracle_qoe_mean",
        "attack_oracle_qoe_p50",
        "attack_oracle_qoe_p95",
        "attack_regret_mean",
        "attack_regret_p50",
        "attack_regret_p95",
        "attack_base_policy_qoe_mean",
        "attack_base_policy_qoe_p50",
        "attack_base_policy_qoe_p95",
        "attack_base_oracle_gap_mean",
        "attack_base_oracle_gap_p50",
        "attack_base_oracle_gap_p95",
        "attack_base_oracle_qoe_mean",
        "attack_base_oracle_qoe_p50",
        "attack_base_oracle_qoe_p95",
        "attack_base_regret_mean",
        "attack_base_regret_p50",
        "attack_base_regret_p95",
        "clean_policy_qoe_mean",
        "clean_policy_qoe_p50",
        "clean_policy_qoe_p95",
        "clean_oracle_gap_mean",
        "clean_oracle_gap_p50",
        "clean_oracle_gap_p95",
        "clean_oracle_qoe_mean",
        "clean_oracle_qoe_p50",
        "clean_oracle_qoe_p95",
        "clean_regret_mean",
        "clean_regret_p50",
        "clean_regret_p95",
        "clean_base_policy_qoe_mean",
        "clean_base_policy_qoe_p50",
        "clean_base_policy_qoe_p95",
        "clean_base_oracle_gap_mean",
        "clean_base_oracle_gap_p50",
        "clean_base_oracle_gap_p95",
        "clean_base_oracle_qoe_mean",
        "clean_base_oracle_qoe_p50",
        "clean_base_oracle_qoe_p95",
        "clean_base_regret_mean",
        "clean_base_regret_p50",
        "clean_base_regret_p95",
    ]
    if not bool(args.resume):
        _write_csv(str(iter_stats_csv), fieldnames=list(iter_stats_fieldnames), rows=[])
        iter_stats_recorded = set()
    else:
        if not os.path.isfile(iter_stats_csv) or os.path.getsize(iter_stats_csv) <= 0:
            _write_csv(str(iter_stats_csv), fieldnames=list(iter_stats_fieldnames), rows=[])
        iter_stats_recorded = _csv_iter_values(str(iter_stats_csv))

    _INTERRUPT_PENDING["iter_stats_csv"] = str(iter_stats_csv)
    _INTERRUPT_PENDING["iter_stats_fieldnames"] = list(iter_stats_fieldnames)
    _INTERRUPT_PENDING["resume"] = bool(args.resume)
    _INTERRUPT_PENDING["wandb_attempted"] = False
    _INTERRUPT_PENDING["wandb_in_progress"] = False

    for it in range(int(args.iters)):
        _INTERRUPT_PENDING["iter_row"] = None
        _INTERRUPT_PENDING["wandb_payload"] = None
        _INTERRUPT_PENDING["wandb_ok"] = None
        _INTERRUPT_PENDING["wandb_attempted"] = False
        _INTERRUPT_PENDING["wandb_in_progress"] = False

        iter_seed = int(args.seed) + int(it)
        iter_dir = os.path.join(run_dir, f"iter_{it:02d}")
        attacker_dir = os.path.join(iter_dir, "attacker")
        traces_dir = os.path.join(iter_dir, "traces")
        states_csv = os.path.join(iter_dir, "states.csv")
        rules_dir = os.path.join(iter_dir, "rules")
        eval_dir = os.path.join(iter_dir, "eval")

        os.makedirs(iter_dir, exist_ok=True)
        os.makedirs(attacker_dir, exist_ok=True)
        os.makedirs(traces_dir, exist_ok=True)
        os.makedirs(rules_dir, exist_ok=True)
        os.makedirs(eval_dir, exist_ok=True)

        #* 1) Train attacker (skip if zip exists and --resume).
        adv_zip = _latest_file(os.path.join(attacker_dir, "*.zip"))
        if adv_zip is None or not bool(args.resume):
            cmd = [
                sys.executable,
                "-m",
                "online.train_adv",
                "--pensieve-model",
                str(base_model),
                "--total-steps",
                str(int(args.attacker_steps)),
                "--lr",
                str(float(args.attacker_lr)),
                "--seed",
                str(int(iter_seed)),
                "--bw-min",
                str(float(online_bw_min)),
                "--bw-max",
                str(float(bw_max)),
                "--obs-history",
                str(int(args.obs_history)),
                "--oracle-window",
                str(int(args.oracle_window)),
                "--smooth-penalty-scale",
                str(float(args.smooth_penalty_scale)),
                "--out-dir",
                str(attacker_dir),
            ]
            if current_min_rules is not None and current_max_rules is not None:
                cmd += [
                    "--shield-min-rules-file",
                    str(current_min_rules),
                    "--shield-max-rules-file",
                    str(current_max_rules),
                    "--shield-action-selection",
                    "argmax",
                ]
            _run(cmd, cwd=_repo_root())
            adv_zip = _latest_file(os.path.join(attacker_dir, "*.zip"))
        if adv_zip is None:
            raise RuntimeError(f"* Failed to find trained attacker model under: {attacker_dir}")

        #* 2) Generate traces (skip if already have enough and --resume).
        if not (bool(args.resume) and _count_traces(traces_dir) >= int(args.episodes)):
            cmd = [
                sys.executable,
                "-m",
                "online.generate_traces",
                "--adv-model",
                str(adv_zip),
                "--pensieve-model",
                str(base_model),
                "--episodes",
                str(int(args.episodes)),
                "--seed",
                str(int(iter_seed)),
                "--bw-min",
                str(float(online_bw_min)),
                "--bw-max",
                str(float(bw_max)),
                "--obs-history",
                str(int(args.obs_history)),
                "--oracle-window",
                str(int(args.oracle_window)),
                "--smooth-penalty-scale",
                str(float(args.smooth_penalty_scale)),
                "--out-dir",
                str(traces_dir),
                "--flat-out-dir",
                "--no-random",
            ]
            if current_min_rules is not None and current_max_rules is not None:
                cmd += [
                    "--shield-min-rules-file",
                    str(current_min_rules),
                    "--shield-max-rules-file",
                    str(current_max_rules),
                    "--shield-action-selection",
                    "argmax",
                ]
            _run(cmd, cwd=_repo_root())

        if _count_traces(traces_dir) < int(args.episodes):
            raise RuntimeError(f"* trace generation produced fewer than --episodes={args.episodes} traces: {traces_dir}")

        #* 3) Log states for mining (skip if exists and --resume).
        if not (bool(args.resume) and os.path.isfile(states_csv) and os.path.getsize(states_csv) > 0):
            cmd = [
                sys.executable,
                os.path.join(_repo_root(), "rules", "shield", "gen_rule_mining_dataset.py"),
                "--trace-dir",
                str(traces_dir),
                "--model",
                str(base_model),
                "--out",
                str(states_csv),
                "--overwrite",
                "--k-steps",
                str(int(args.k_steps)),
                "--trace-num",
                str(int(args.episodes)),
                "--action-selection",
                str(args.action_selection),
                "--seed",
                str(int(iter_seed)),
            ]
            if current_min_rules is not None and current_max_rules is not None:
                cmd += [
                    "--shield-min-rules-file",
                    str(current_min_rules),
                    "--shield-max-rules-file",
                    str(current_max_rules),
                ]
            _run(cmd, cwd=_repo_root())

        if not os.path.isfile(states_csv) or os.path.getsize(states_csv) <= 0:
            raise RuntimeError(f"* states.csv missing/empty: {states_csv}")

        #* 4) Aggregate adversarial states across iterations (rebuild each iter).
        import pandas as pd

        states_all_path = os.path.join(agg_dir, "states_all.csv")
        frames: list[pd.DataFrame] = []
        for j in range(int(it) + 1):
            p = os.path.join(run_dir, f"iter_{j:02d}", "states.csv")
            if not os.path.isfile(p):
                raise RuntimeError(f"* Missing states.csv for iter_{j:02d}: {p}")
            dfj = pd.read_csv(p)
            frames.append(dfj)
        df_all = pd.concat(frames, axis=0, ignore_index=True)
        if "trace_dir" in df_all.columns and "trace_name" in df_all.columns and "chunk_idx" in df_all.columns:
            df_all = df_all.sort_values(by=["trace_dir", "trace_name", "chunk_idx"], kind="mergesort").reset_index(drop=True)
        df_all.to_csv(states_all_path, index=False)

        #* 5) Label aggregated dataset.
        labeled_all_path = os.path.join(agg_dir, "labeled_all.csv")
        do_label = True
        if bool(args.resume) and os.path.isfile(labeled_all_path) and os.path.getsize(labeled_all_path) > 0:
            if float(os.path.getmtime(labeled_all_path)) >= float(os.path.getmtime(states_all_path)):
                do_label = False
        if do_label:
            cmd = [
                sys.executable,
                os.path.join(_repo_root(), "rules", "shield", "label_bounds.py"),
                "--in",
                str(states_all_path),
                "--out",
                str(labeled_all_path),
                "--gap-pct-open-threshold",
                str(float(args.gap_pct_open_threshold)),
                "--oracle-best-min-abs",
                str(float(args.oracle_best_min_abs)),
                "--regret-delta",
                str(float(args.regret_delta)),
                "--regret-delta-min-abs",
                str(float(args.regret_delta_min_abs)),
                "--regret-delta-max-abs",
                str(float(args.regret_delta_max_abs)),
                "--noncontiguous",
                str(args.noncontiguous),
            ]
            _run(cmd, cwd=_repo_root())

        if not os.path.isfile(labeled_all_path) or os.path.getsize(labeled_all_path) <= 0:
            raise RuntimeError(f"* labeled_all.csv missing/empty: {labeled_all_path}")

        #* 6) Train DTs and extract rules (writes to this iteration's rules dir).
        prefix = (
            f"cegar_iter{it:02d}_bounds_{args.dt_backend}"
            f"_gap{_fmt_tag_intlike(float(args.gap_pct_open_threshold))}"
            f"_delta{_fmt_tag_intlike(float(args.regret_delta))}"
            f"_dmin{_fmt_tag_keep_p0(float(args.regret_delta_min_abs))}"
            f"_dmax{_fmt_tag_keep_p0(float(args.regret_delta_max_abs))}"
        )
        out_prefix = os.path.join(rules_dir, prefix)
        out_min = out_prefix + "_min.pl"
        out_max = out_prefix + "_max.pl"

        do_train_dt = True
        if bool(args.resume) and os.path.isfile(out_min) and os.path.isfile(out_max) and os.path.getsize(out_min) > 0 and os.path.getsize(out_max) > 0:
            if float(os.path.getmtime(out_min)) >= float(os.path.getmtime(labeled_all_path)) and float(os.path.getmtime(out_max)) >= float(os.path.getmtime(labeled_all_path)):
                do_train_dt = False

        if do_train_dt:
            cmd = [
                sys.executable,
                os.path.join(_repo_root(), "rules", "shield", "train_bounds_dt.py"),
                "--data",
                str(labeled_all_path),
                "--backend",
                str(args.dt_backend),
                "--seed",
                str(int(iter_seed)),
                "--leaf-purity",
                str(float(args.leaf_purity)),
                "--out-prefix",
                str(out_prefix),
                "--h2o-ntrees",
                str(int(args.h2o_ntrees)),
                "--h2o-max-depth",
                str(int(args.h2o_max_depth)),
                "--h2o-min-rows",
                str(int(args.h2o_min_rows)),
                "--h2o-sample-rate",
                str(float(args.h2o_sample_rate)),
            ]
            if args.h2o_mtries is not None:
                cmd += ["--h2o-mtries", str(int(args.h2o_mtries))]
            _run(cmd, cwd=_repo_root())

        if not os.path.isfile(out_min) or not os.path.isfile(out_max):
            raise RuntimeError(f"* missing extracted rules: {out_min} / {out_max}")

        #* Update shield for the next iteration.
        current_min_rules = str(out_min)
        current_max_rules = str(out_max)

        #* 7) Evaluate base + shielded victim and log to W&B (one run per iteration).
        attack_trace_csv = os.path.join(eval_dir, "attack_traces.csv")
        attack_step_csv = os.path.join(eval_dir, "attack_steps.csv")
        clean_trace_csv = os.path.join(eval_dir, "clean_traces.csv")
        clean_step_csv = os.path.join(eval_dir, "clean_steps.csv")
        attack_base_trace_csv = os.path.join(eval_dir, "attack_traces_base.csv")
        attack_base_step_csv = os.path.join(eval_dir, "attack_steps_base.csv")
        clean_base_trace_csv = os.path.join(eval_dir, "clean_traces_base.csv")
        clean_base_step_csv = os.path.join(eval_dir, "clean_steps_base.csv")
        wandb_ok = os.path.join(eval_dir, "wandb_logged.ok")

        have_shielded_eval = (
            bool(args.resume)
            and os.path.isfile(attack_trace_csv)
            and os.path.isfile(attack_step_csv)
            and os.path.isfile(clean_trace_csv)
            and os.path.isfile(clean_step_csv)
            and os.path.getsize(attack_trace_csv) > 0
            and os.path.getsize(attack_step_csv) > 0
            and os.path.getsize(clean_trace_csv) > 0
            and os.path.getsize(clean_step_csv) > 0
        )
        have_base_eval = (
            bool(args.resume)
            and os.path.isfile(attack_base_trace_csv)
            and os.path.isfile(attack_base_step_csv)
            and os.path.isfile(clean_base_trace_csv)
            and os.path.isfile(clean_base_step_csv)
            and os.path.getsize(attack_base_trace_csv) > 0
            and os.path.getsize(attack_base_step_csv) > 0
            and os.path.getsize(clean_base_trace_csv) > 0
            and os.path.getsize(clean_base_step_csv) > 0
        )

        if not bool(have_base_eval):
            print(f"* eval iter={it:02d} (base)", flush=True)
            attack_base_traces, attack_base_steps = _eval_dataset(
                trace_type="attack",
                trace_dir=str(traces_dir),
                trace_num=int(args.episodes),
                model_path=str(base_model),
                shield_min_rules_file=None,
                shield_max_rules_file=None,
                oracle_k=int(args.k_steps),
                action_selection=str(args.action_selection),
                seed=int(iter_seed),
                out_trace_csv=str(attack_base_trace_csv),
                out_step_csv=str(attack_base_step_csv),
                progress_prefix=f"[iter {it:02d}] attack (base)",
            )
            clean_base_traces, clean_base_steps = _eval_dataset(
                trace_type="clean",
                trace_dir=str(clean_traces_dir),
                trace_num=int(args.clean_trace_num),
                model_path=str(base_model),
                shield_min_rules_file=None,
                shield_max_rules_file=None,
                oracle_k=int(args.k_steps),
                action_selection=str(args.action_selection),
                seed=int(iter_seed),
                out_trace_csv=str(clean_base_trace_csv),
                out_step_csv=str(clean_base_step_csv),
                progress_prefix=f"[iter {it:02d}] clean (base)",
            )
        else:
            import pandas as pd

            attack_base_traces = pd.read_csv(attack_base_trace_csv).to_dict(orient="records")
            clean_base_traces = pd.read_csv(clean_base_trace_csv).to_dict(orient="records")
            attack_base_steps = pd.read_csv(attack_base_step_csv).to_dict(orient="records")
            clean_base_steps = pd.read_csv(clean_base_step_csv).to_dict(orient="records")

        if not bool(have_shielded_eval):
            print(f"* eval iter={it:02d} (shielded)", flush=True)
            attack_traces, attack_steps = _eval_dataset(
                trace_type="attack",
                trace_dir=str(traces_dir),
                trace_num=int(args.episodes),
                model_path=str(base_model),
                shield_min_rules_file=str(out_min),
                shield_max_rules_file=str(out_max),
                oracle_k=int(args.k_steps),
                action_selection=str(args.action_selection),
                seed=int(iter_seed),
                out_trace_csv=str(attack_trace_csv),
                out_step_csv=str(attack_step_csv),
                progress_prefix=f"[iter {it:02d}] attack",
            )
            clean_traces, clean_steps = _eval_dataset(
                trace_type="clean",
                trace_dir=str(clean_traces_dir),
                trace_num=int(args.clean_trace_num),
                model_path=str(base_model),
                shield_min_rules_file=str(out_min),
                shield_max_rules_file=str(out_max),
                oracle_k=int(args.k_steps),
                action_selection=str(args.action_selection),
                seed=int(iter_seed),
                out_trace_csv=str(clean_trace_csv),
                out_step_csv=str(clean_step_csv),
                progress_prefix=f"[iter {it:02d}] clean",
            )
        else:
            import pandas as pd

            attack_traces = pd.read_csv(attack_trace_csv).to_dict(orient="records")
            clean_traces = pd.read_csv(clean_trace_csv).to_dict(orient="records")
            attack_steps = pd.read_csv(attack_step_csv).to_dict(orient="records")
            clean_steps = pd.read_csv(clean_step_csv).to_dict(orient="records")

        summary: dict[str, float] = {}
        atk_qoe = _trace_stats(attack_traces, key="policy_qoe_total")
        atk_oracle = _trace_stats(attack_traces, key="oracle_qoe_total")
        atk_gap = _trace_stats(attack_traces, key="qoe_gap_total")
        atk_regret = _trace_stats(attack_traces, key="regret_mean")
        atk_regret_step = _step_stats(attack_steps, key="regret")
        atk_base_qoe = _trace_stats(attack_base_traces, key="policy_qoe_total")
        atk_base_oracle = _trace_stats(attack_base_traces, key="oracle_qoe_total")
        atk_base_gap = _trace_stats(attack_base_traces, key="qoe_gap_total")
        atk_base_regret = _trace_stats(attack_base_traces, key="regret_mean")
        atk_base_regret_step = _step_stats(attack_base_steps, key="regret")
        cln_qoe = _trace_stats(clean_traces, key="policy_qoe_total")
        cln_oracle = _trace_stats(clean_traces, key="oracle_qoe_total")
        cln_gap = _trace_stats(clean_traces, key="qoe_gap_total")
        cln_regret = _trace_stats(clean_traces, key="regret_mean")
        cln_regret_step = _step_stats(clean_steps, key="regret")
        cln_base_qoe = _trace_stats(clean_base_traces, key="policy_qoe_total")
        cln_base_oracle = _trace_stats(clean_base_traces, key="oracle_qoe_total")
        cln_base_gap = _trace_stats(clean_base_traces, key="qoe_gap_total")
        cln_base_regret = _trace_stats(clean_base_traces, key="regret_mean")
        cln_base_regret_step = _step_stats(clean_base_steps, key="regret")

        for stat, d in [("qoe_total", atk_qoe), ("oracle_qoe_total", atk_oracle), ("qoe_gap_total", atk_gap), ("regret", atk_regret)]:
            for k, v in d.items():
                summary[f"attack/{stat}_{k}"] = float(v)
        summary["attack/num_traces"] = float(len(attack_traces))
        for k, v in atk_regret_step.items():
            summary[f"attack/regret_step_{k}"] = float(v)
        for stat, d in [
            ("qoe_total", atk_base_qoe),
            ("oracle_qoe_total", atk_base_oracle),
            ("qoe_gap_total", atk_base_gap),
            ("regret", atk_base_regret),
        ]:
            for k, v in d.items():
                summary[f"attack_base/{stat}_{k}"] = float(v)
        summary["attack_base/num_traces"] = float(len(attack_base_traces))
        for k, v in atk_base_regret_step.items():
            summary[f"attack_base/regret_step_{k}"] = float(v)

        for stat, d in [("qoe_total", cln_qoe), ("oracle_qoe_total", cln_oracle), ("qoe_gap_total", cln_gap), ("regret", cln_regret)]:
            for k, v in d.items():
                summary[f"clean/{stat}_{k}"] = float(v)
        summary["clean/num_traces"] = float(len(clean_traces))
        for k, v in cln_regret_step.items():
            summary[f"clean/regret_step_{k}"] = float(v)
        for stat, d in [
            ("qoe_total", cln_base_qoe),
            ("oracle_qoe_total", cln_base_oracle),
            ("qoe_gap_total", cln_base_gap),
            ("regret", cln_base_regret),
        ]:
            for k, v in d.items():
                summary[f"clean_base/{stat}_{k}"] = float(v)
        summary["clean_base/num_traces"] = float(len(clean_base_traces))
        for k, v in cln_base_regret_step.items():
            summary[f"clean_base/regret_step_{k}"] = float(v)

        iter_row = {
            "iter": int(it),
            "attack_policy_qoe_mean": float(atk_qoe["mean"]),
            "attack_policy_qoe_p50": float(atk_qoe["p50"]),
            "attack_policy_qoe_p95": float(atk_qoe["p95"]),
            "attack_oracle_gap_mean": float(atk_gap["mean"]),
            "attack_oracle_gap_p50": float(atk_gap["p50"]),
            "attack_oracle_gap_p95": float(atk_gap["p95"]),
            "attack_oracle_qoe_mean": float(atk_oracle["mean"]),
            "attack_oracle_qoe_p50": float(atk_oracle["p50"]),
            "attack_oracle_qoe_p95": float(atk_oracle["p95"]),
            "attack_regret_mean": float(atk_regret_step["mean"]),
            "attack_regret_p50": float(atk_regret_step["p50"]),
            "attack_regret_p95": float(atk_regret_step["p95"]),
            "attack_base_policy_qoe_mean": float(atk_base_qoe["mean"]),
            "attack_base_policy_qoe_p50": float(atk_base_qoe["p50"]),
            "attack_base_policy_qoe_p95": float(atk_base_qoe["p95"]),
            "attack_base_oracle_gap_mean": float(atk_base_gap["mean"]),
            "attack_base_oracle_gap_p50": float(atk_base_gap["p50"]),
            "attack_base_oracle_gap_p95": float(atk_base_gap["p95"]),
            "attack_base_oracle_qoe_mean": float(atk_base_oracle["mean"]),
            "attack_base_oracle_qoe_p50": float(atk_base_oracle["p50"]),
            "attack_base_oracle_qoe_p95": float(atk_base_oracle["p95"]),
            "attack_base_regret_mean": float(atk_base_regret_step["mean"]),
            "attack_base_regret_p50": float(atk_base_regret_step["p50"]),
            "attack_base_regret_p95": float(atk_base_regret_step["p95"]),
            "clean_policy_qoe_mean": float(cln_qoe["mean"]),
            "clean_policy_qoe_p50": float(cln_qoe["p50"]),
            "clean_policy_qoe_p95": float(cln_qoe["p95"]),
            "clean_oracle_gap_mean": float(cln_gap["mean"]),
            "clean_oracle_gap_p50": float(cln_gap["p50"]),
            "clean_oracle_gap_p95": float(cln_gap["p95"]),
            "clean_oracle_qoe_mean": float(cln_oracle["mean"]),
            "clean_oracle_qoe_p50": float(cln_oracle["p50"]),
            "clean_oracle_qoe_p95": float(cln_oracle["p95"]),
            "clean_regret_mean": float(cln_regret_step["mean"]),
            "clean_regret_p50": float(cln_regret_step["p50"]),
            "clean_regret_p95": float(cln_regret_step["p95"]),
            "clean_base_policy_qoe_mean": float(cln_base_qoe["mean"]),
            "clean_base_policy_qoe_p50": float(cln_base_qoe["p50"]),
            "clean_base_policy_qoe_p95": float(cln_base_qoe["p95"]),
            "clean_base_oracle_gap_mean": float(cln_base_gap["mean"]),
            "clean_base_oracle_gap_p50": float(cln_base_gap["p50"]),
            "clean_base_oracle_gap_p95": float(cln_base_gap["p95"]),
            "clean_base_oracle_qoe_mean": float(cln_base_oracle["mean"]),
            "clean_base_oracle_qoe_p50": float(cln_base_oracle["p50"]),
            "clean_base_oracle_qoe_p95": float(cln_base_oracle["p95"]),
            "clean_base_regret_mean": float(cln_base_regret_step["mean"]),
            "clean_base_regret_p50": float(cln_base_regret_step["p50"]),
            "clean_base_regret_p95": float(cln_base_regret_step["p95"]),
        }
        _INTERRUPT_PENDING["iter_row"] = dict(iter_row)
        _INTERRUPT_PENDING["wandb_ok"] = str(wandb_ok)
        if int(it) not in iter_stats_recorded:
            _append_csv_row(str(iter_stats_csv), fieldnames=list(iter_stats_fieldnames), row=dict(iter_row))
            iter_stats_recorded.add(int(it))

        if bool(args.wandb) and not (bool(args.resume) and os.path.isfile(wandb_ok)):
            wandb = _try_import_wandb()
            if wandb is None:
                raise RuntimeError("* --wandb set but `wandb` is not importable; install `wandb` and retry")

            tags = [t.strip() for t in str(args.wandb_tags).split(",") if t.strip()]
            run_name = f"{str(args.wandb_name_prefix)}_iter{it:02d}"
            cfg = {
                "iter": int(it),
                "seed": int(iter_seed),
                "base_model": str(base_model),
                "episodes": int(args.episodes),
                "clean_trace_num": int(args.clean_trace_num),
                "attacker_steps": int(args.attacker_steps),
                "attacker_lr": float(args.attacker_lr),
                "bw_min": float(online_bw_min),
                "bw_min_requested": float(args.bw_min),
                "bw_max": float(bw_max),
                "obs_history": int(args.obs_history),
                "oracle_window": int(args.oracle_window),
                "smooth_penalty_scale": float(args.smooth_penalty_scale),
                "k_steps": int(args.k_steps),
                "action_selection": str(args.action_selection),
                "gap_pct_open_threshold": float(args.gap_pct_open_threshold),
                "oracle_best_min_abs": float(args.oracle_best_min_abs),
                "regret_delta": float(args.regret_delta),
                "regret_delta_min_abs": float(args.regret_delta_min_abs),
                "regret_delta_max_abs": float(args.regret_delta_max_abs),
                "noncontiguous": str(args.noncontiguous),
                "artifact/iter_dir": str(iter_dir),
                "artifact/rules_min": str(out_min),
                "artifact/rules_max": str(out_max),
                "artifact/traces_dir": str(traces_dir),
                "artifact/clean_traces_dir": str(clean_traces_dir),
                "artifact/states_csv": str(states_csv),
                "artifact/agg_states_csv": str(states_all_path),
                "artifact/agg_labeled_csv": str(labeled_all_path),
                "artifact/eval_attack_traces_csv": str(attack_trace_csv),
                "artifact/eval_attack_steps_csv": str(attack_step_csv),
                "artifact/eval_clean_traces_csv": str(clean_trace_csv),
                "artifact/eval_clean_steps_csv": str(clean_step_csv),
                "artifact/eval_attack_base_traces_csv": str(attack_base_trace_csv),
                "artifact/eval_attack_base_steps_csv": str(attack_base_step_csv),
                "artifact/eval_clean_base_traces_csv": str(clean_base_trace_csv),
                "artifact/eval_clean_base_steps_csv": str(clean_base_step_csv),
            }
            _INTERRUPT_PENDING["wandb_payload"] = {
                "project": str(args.wandb_project),
                "entity": args.wandb_entity,
                "mode": str(args.wandb_mode),
                "tags": list(tags),
                "name": str(run_name),
                "config": dict(cfg),
                "summary": dict(summary),
                "attack_traces": list(attack_traces),
                "clean_traces": list(clean_traces),
                "attack_steps": list(attack_steps),
                "clean_steps": list(clean_steps),
            }
            _log_wandb_iteration(
                wandb=wandb,
                project=str(args.wandb_project),
                entity=args.wandb_entity,
                mode=str(args.wandb_mode),
                tags=tags,
                name=str(run_name),
                config=cfg,
                summary=summary,
                attack_traces=attack_traces,
                clean_traces=clean_traces,
                attack_steps=attack_steps,
                clean_steps=clean_steps,
            )
            with open(wandb_ok, "w", encoding="utf-8") as f:
                f.write(f"logged_at_unix_s={time.time():.6f}\n")

        _INTERRUPT_PENDING["iter_row"] = None
        _INTERRUPT_PENDING["wandb_payload"] = None
        _INTERRUPT_PENDING["wandb_ok"] = None
        _INTERRUPT_PENDING["wandb_attempted"] = False
        _INTERRUPT_PENDING["wandb_in_progress"] = False

        print(f"* iter {it:02d} done: rules={os.path.basename(out_prefix)} traces={traces_dir}", flush=True)

    print(f"* Wrote per-iteration stats CSV (incremental): {iter_stats_csv}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        #* Best-effort flush: iteration CSV is incremental, but W&B logging may be pending.
        print("\n* Ctrl+C received: flushing latest results and exiting", flush=True)
        try:
            _flush_pending_results_on_interrupt()
        except KeyboardInterrupt:
            print("\n* Ctrl+C again: exiting immediately", flush=True)
        except Exception as e:
            print(f"* WARNING: failed to flush pending results: {e}", flush=True)
        sys.exit(130)
