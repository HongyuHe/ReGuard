#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Any


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _pensieve_src_dir() -> str:
    return os.path.join(_repo_root(), "Pensieve-PPO", "src")


def _add_pensieve_src_to_path() -> None:
    src_dir = _pensieve_src_dir()
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


def _add_repo_root_to_path() -> None:
    root = _repo_root()
    if root not in sys.path:
        sys.path.insert(0, root)


def _resolve_path(repo_relative_or_abs: str) -> str:
    expanded = os.path.expandvars(os.path.expanduser(str(repo_relative_or_abs)))
    if os.path.isabs(expanded):
        return expanded
    return os.path.join(_repo_root(), expanded)


def _trace_files_in_dir(trace_dir: str) -> list[str]:
    files: list[str] = []
    for name in sorted(os.listdir(trace_dir)):
        if name.startswith("."):
            continue
        if name.endswith(".json"):
            continue
        path = os.path.join(trace_dir, name)
        if not os.path.isfile(path):
            continue
        files.append(name)
    return files


def _snap_bitrate_kbps(x_kbps: float, *, allowed: list[int]) -> int:
    if x_kbps <= 0.0:
        return 0
    if x_kbps < float(min(allowed)) * 0.5:
        return 0
    return int(min(allowed, key=lambda v: abs(float(v) - float(x_kbps))))


def _denorm_state(state_norm: Any, *, util_mod: Any) -> dict[str, Any]:
    import numpy as np

    arr = np.asarray(state_norm, dtype=np.float64)
    if arr.shape != (int(util_mod.S_INFO), int(util_mod.S_LEN)):
        raise ValueError(f"* Unexpected state shape: {arr.shape}")

    max_kbps = float(max(util_mod.VIDEO_BIT_RATE))
    allowed_kbps = [int(x) for x in util_mod.VIDEO_BIT_RATE]

    bitrate_hist_kbps = [
        _snap_bitrate_kbps(float(arr[0, i]) * max_kbps, allowed=allowed_kbps) for i in range(int(util_mod.S_LEN))
    ]
    buffer_hist_sec = [float(arr[1, i]) * float(util_mod.BUFFER_NORM_FACTOR) for i in range(int(util_mod.S_LEN))]

    #* In the Pensieve state, `state[2]` is the "throughput_norm" feature; inference.py treats it as Mbps/8.
    throughput_hist_mbps = [float(arr[2, i]) * 8.0 for i in range(int(util_mod.S_LEN))]

    delay_hist_ms = [
        float(arr[3, i]) * float(util_mod.M_IN_K) * float(util_mod.BUFFER_NORM_FACTOR) for i in range(int(util_mod.S_LEN))
    ]

    next_chunk_sizes_bytes = [
        int(round(float(arr[4, i]) * float(util_mod.M_IN_K) * float(util_mod.M_IN_K))) for i in range(int(util_mod.A_DIM))
    ]

    remain_current = int(round(float(arr[5, -1]) * float(util_mod.CHUNK_TIL_VIDEO_END_CAP)))

    return {
        "bitrate_hist_kbps": bitrate_hist_kbps,
        "buffer_hist_sec": buffer_hist_sec,
        "throughput_hist_mbps": throughput_hist_mbps,
        "delay_hist_ms": delay_hist_ms,
        "next_chunk_sizes_bytes": next_chunk_sizes_bytes,
        "video_chunk_remain": remain_current,
    }


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
    new_state[0, -1] = float(util_mod.VIDEO_BIT_RATE[int(bit_rate)]) / float(np.max(util_mod.VIDEO_BIT_RATE))
    new_state[1, -1] = float(buffer_s) / float(util_mod.BUFFER_NORM_FACTOR)
    new_state[2, -1] = float(video_chunk_size) / float(delay_ms) / float(util_mod.M_IN_K)
    new_state[3, -1] = float(delay_ms) / float(util_mod.M_IN_K) / float(util_mod.BUFFER_NORM_FACTOR)
    new_state[4, : int(util_mod.A_DIM)] = (
        np.asarray(next_video_chunk_sizes, dtype=np.float64) / float(util_mod.M_IN_K) / float(util_mod.M_IN_K)
    )
    new_state[5, -1] = float(min(int(video_chunk_remain), int(util_mod.CHUNK_TIL_VIDEO_END_CAP))) / float(
        util_mod.CHUNK_TIL_VIDEO_END_CAP
    )
    return new_state


def _policy_k_step_reward_mean(
    actor: Any,
    *,
    env_snapshot: Any,
    state_snapshot: Any,
    last_br: int,
    start_action: int,
    k_steps: int,
    action_selection: str,
    rng: Any,
    shield: Any | None,
    util_mod: Any,
) -> float | None:
    k = int(k_steps)
    if k <= 0:
        return None

    import numpy as np

    env_sim = util_mod.copy_env(env_snapshot)
    state_sim = np.array(state_snapshot, copy=True)

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
                        raise ValueError(f"Unknown action_selection: {action_selection}")
    finally:
        if shield is not None and hasattr(shield, "restore_runtime_state"):
            shield.restore_runtime_state(shield_snapshot)

    return float(total_reward) / float(k)


def _oracle_best_reward_mean(*, env_snapshot: Any, last_br: int, k_steps: int, util_mod: Any) -> float | None:
    k = int(k_steps)
    if k <= 0:
        return None

    best_r, _best_seq, _worst_r, _worst_seq = util_mod.find_optimal_k_step_reward_oracle(
        env_snapshot, int(last_br), int(k)
    )
    if best_r is None:
        return None
    return float(best_r) / float(k)


def _oracle_q_reward_mean_by_first_action(
    *,
    env_snapshot: Any,
    last_br: int,
    k_steps: int,
    util_mod: Any,
) -> list[float | None]:
    k = int(k_steps)
    if k <= 0:
        return [None for _ in range(int(util_mod.A_DIM))]

    out: list[float | None] = []
    for a0 in range(int(util_mod.A_DIM)):
        env_first = util_mod.copy_env(env_snapshot)
        try:
            _delay_ms, _sleep_ms, _buffer_s, rebuf_s, _chunk_size, _next_sizes, end, _remain = env_first.get_video_chunk(
                int(a0)
            )
        except Exception:
            out.append(None)
            continue

        #* Match the oracle validity rule: sequences that end early are invalid for K-step comparison.
        if bool(end) and int(k) > 1:
            out.append(None)
            continue

        r0 = float(util_mod.calculate_reward(int(a0), int(last_br), float(rebuf_s)))
        if int(k) == 1:
            out.append(float(r0))
            continue

        best_future, _best_seq, _worst_r, _worst_seq = util_mod.find_optimal_k_step_reward_oracle(
            env_first, int(a0), int(k) - 1
        )
        if best_future is None:
            out.append(None)
            continue

        total = float(r0) + float(best_future)
        out.append(float(total) / float(k))

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a shield rule-mining dataset by replaying traces.")
    parser.add_argument("--trace-dir", required=True, help="Directory with .trace files (time_s\\tbw_mbps).")
    parser.add_argument("--model", required=True, help="Pretrained Pensieve model path (e.g., ppo/*.pth).")
    parser.add_argument("--out", required=True, help="Output CSV path.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite --out if it already exists (default: skip generation when --out exists).",
    )
    parser.add_argument("--k-steps", type=int, default=5, help="Oracle/policy horizon K.")
    parser.add_argument("--trace-num", type=int, default=0, help="Use only the first N traces (0 means all).")
    parser.add_argument("--max-logged-states", type=int, default=0, help="Stop after logging N states (0 means all).")
    parser.add_argument(
        "--action-selection",
        type=str,
        default="argmax",
        choices=["argmax", "gumbel"],
        help="Policy action selection when rolling out and when scoring K-step policy reward.",
    )
    parser.add_argument("--shield-min-rules-file", type=str, default=None, help="Optional shield min rules (.pl).")
    parser.add_argument("--shield-max-rules-file", type=str, default=None, help="Optional shield max rules (.pl).")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for gumbel selection (if enabled).")
    parser.add_argument("--progress-every", type=int, default=200, help="Print progress every N chunks.")

    args = parser.parse_args()

    _add_pensieve_src_to_path()
    _add_repo_root_to_path()
    import numpy as np

    import fixed_env as env
    import load_trace
    import ppo2 as network
    import util as util_mod

    shield = None
    if (args.shield_min_rules_file is None) != (args.shield_max_rules_file is None):
        raise ValueError("* Provide both --shield-min-rules-file and --shield-max-rules-file, or neither")
    if args.shield_min_rules_file is not None and args.shield_max_rules_file is not None:
        from rules.shield.shield_runtime import ShieldBounds, load_rules

        min_rules = load_rules(_resolve_path(str(args.shield_min_rules_file)), output_var="MinBitrateKbps")
        max_rules = load_rules(_resolve_path(str(args.shield_max_rules_file)), output_var="MaxBitrateKbps")
        shield = ShieldBounds(min_rules=min_rules, max_rules=max_rules)

    trace_dir = _resolve_path(args.trace_dir)
    model_path = _resolve_path(args.model)
    out_path = _resolve_path(args.out)

    if not os.path.isdir(trace_dir):
        raise FileNotFoundError(f"* Trace dir not found: {trace_dir}")

    if os.path.exists(out_path) and not bool(args.overwrite):
        if os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
            print(f"* Output exists; skipping generation: {out_path}", flush=True)
            return
        if os.path.isdir(out_path):
            raise IsADirectoryError(f"* --out must be a file, got a directory: {out_path}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    np.random.seed(int(args.seed))
    rng = np.random.RandomState(int(args.seed))

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(str(trace_dir))
    if int(args.trace_num) > 0:
        n = int(args.trace_num)
        if int(n) > int(len(all_file_names)):
            raise RuntimeError(
                f"* --trace-num={n} exceeds available traces ({len(all_file_names)}): {trace_dir}"
            )
        all_cooked_time = all_cooked_time[:n]
        all_cooked_bw = all_cooked_bw[:n]
        all_file_names = all_file_names[:n]

    net_env = env.Environment(all_cooked_time=all_cooked_time, all_cooked_bw=all_cooked_bw)

    actor = network.Network(state_dim=[util_mod.S_INFO, util_mod.S_LEN], action_dim=util_mod.A_DIM, learning_rate=util_mod.ACTOR_LR_RATE)
    actor.load_model(str(model_path))

    start_time = time.time()
    created_at_unix_s = float(time.time())
    total_chunks = 0
    total_logged = 0
    total_states_considered = 0

    fieldnames: list[str] = [
        "trace_dir",
        "model",
        "created_at_unix_s",
        "k_steps",
        "action_selection",
        "seed",
        "trace_name",
        "trace_idx",
        "chunk_idx",
        "chunk_size_bytes",
        "policy_action_idx",
        "policy_action_kbps",
        "last_action_idx",
        "last_action_kbps",
        "oracle_action_kbps",
        "oracle_best_reward_mean",
        "policy_reward_mean",
        "oracle_gap_mean",
        "oracle_gap_pct",
        "shield_min_idx",
        "shield_max_idx",
        "shield_inconsistent",
        "shield_matched_min",
        "shield_matched_max",
        "shield_selected_action_idx",
        "shield_min_kbps",
        "shield_max_kbps",
    ]
    fieldnames.extend(
        [f"oracle_q_reward_mean_by_action{i}" for i in range(int(util_mod.A_DIM))]
    )
    fieldnames.extend([f"bitrate_hist_kbps_{i}" for i in range(int(util_mod.S_LEN))])
    fieldnames.extend([f"buffer_hist_sec_{i}" for i in range(int(util_mod.S_LEN))])
    fieldnames.extend([f"throughput_hist_mbps_{i}" for i in range(int(util_mod.S_LEN))])
    fieldnames.extend([f"delay_hist_ms_{i}" for i in range(int(util_mod.S_LEN))])
    fieldnames.append("video_chunk_remain")
    fieldnames.extend([f"next_chunk_size_bytes_{i}" for i in range(int(util_mod.A_DIM))])

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for trace_idx, trace_name in enumerate(all_file_names):
            #* Reset the simulator to the beginning of this trace.
            net_env.trace_idx = int(trace_idx)
            net_env.cooked_time = net_env.all_cooked_time[net_env.trace_idx]
            net_env.cooked_bw = net_env.all_cooked_bw[net_env.trace_idx]
            net_env.video_chunk_counter = 0
            net_env.buffer_size = 0.0
            net_env.mahimahi_ptr = int(net_env.mahimahi_start_ptr)
            net_env.last_mahimahi_time = net_env.cooked_time[net_env.mahimahi_ptr - 1]

            state = np.zeros((int(util_mod.S_INFO), int(util_mod.S_LEN)), dtype=np.float64)
            if shield is not None and hasattr(shield, "reset_runtime_state"):
                shield.reset_runtime_state()

            last_bit_rate = int(util_mod.DEFAULT_QUALITY)
            bit_rate = int(util_mod.DEFAULT_QUALITY)

            while True:
                try:
                    delay_ms, _sleep_ms, buffer_s, rebuf_s, chunk_size, next_sizes, end, remain = net_env.get_video_chunk(
                        int(bit_rate)
                    )
                except Exception as e:
                    raise RuntimeError(f"* Simulator failed on trace={trace_name} chunk={net_env.video_chunk_counter}") from e

                #* Update the policy state after observing the download outcome.
                state = _update_state(
                    state,
                    bit_rate=int(bit_rate),
                    buffer_s=float(buffer_s),
                    video_chunk_size=int(chunk_size),
                    delay_ms=float(delay_ms),
                    next_video_chunk_sizes=[int(x) for x in next_sizes],
                    video_chunk_remain=int(remain),
                    util_mod=util_mod,
                )

                total_chunks += 1

                if bool(end):
                    break

                #* Decision point for the next chunk: score the policy vs the exact K-step oracle.
                env_snapshot = util_mod.copy_env(net_env)
                last_br = int(bit_rate)

                p = actor.predict(np.reshape(state, (1, int(util_mod.S_INFO), int(util_mod.S_LEN)))).astype(np.float64).reshape(-1)
                shield_stats: dict[str, float | int] = {}
                if shield is not None:
                    next_action, shield_stats = shield.select_action(
                        p=p,
                        state=state,
                        last_bitrate_idx=int(last_br),
                        video_chunk_size_bytes=int(chunk_size),
                        action_selection=str(args.action_selection),
                        rng=rng,
                    )
                else:
                    if str(args.action_selection) == "argmax":
                        next_action = int(np.argmax(p))
                    else:
                        noise = rng.gumbel(size=p.shape)
                        next_action = int(np.argmax(np.log(p + 1e-12) + noise))

                best_r, best_seq, _worst_r, _worst_seq = util_mod.find_optimal_k_step_reward_oracle(
                    env_snapshot, int(last_br), int(args.k_steps)
                )
                if best_r is None or best_seq is None or len(best_seq) == 0:
                    bit_rate = int(next_action)
                    last_bit_rate = int(bit_rate)
                    continue
                oracle_action_kbps = int(util_mod.VIDEO_BIT_RATE[int(best_seq[0])])
                oracle_best_mean = float(best_r) / float(int(args.k_steps))

                policy_mean = _policy_k_step_reward_mean(
                    actor,
                    env_snapshot=env_snapshot,
                    state_snapshot=state,
                    last_br=int(last_br),
                    start_action=int(next_action),
                    k_steps=int(args.k_steps),
                    action_selection=str(args.action_selection),
                    rng=rng,
                    shield=shield,
                    util_mod=util_mod,
                )
                if policy_mean is None:
                    bit_rate = int(next_action)
                    last_bit_rate = int(bit_rate)
                    continue

                gap_mean = float(oracle_best_mean) - float(policy_mean)
                total_states_considered += 1

                denom = max(1e-12, abs(float(oracle_best_mean)))
                gap_pct = float(gap_mean) / float(denom) * 100.0

                q_oracle_by_first = _oracle_q_reward_mean_by_first_action(
                    env_snapshot=env_snapshot,
                    last_br=int(last_br),
                    k_steps=int(args.k_steps),
                    util_mod=util_mod,
                )

                sd = _denorm_state(state, util_mod=util_mod)

                row: dict[str, Any] = {
                    "trace_dir": str(args.trace_dir),
                    "model": str(args.model),
                    "created_at_unix_s": float(created_at_unix_s),
                    "k_steps": int(args.k_steps),
                    "action_selection": str(args.action_selection),
                    "seed": int(args.seed),
                    "trace_name": str(trace_name),
                    "trace_idx": int(trace_idx),
                    "chunk_idx": int(getattr(env_snapshot, "video_chunk_counter", -1)),
                    "chunk_size_bytes": int(chunk_size),
                    "policy_action_idx": int(next_action),
                    "policy_action_kbps": int(util_mod.VIDEO_BIT_RATE[int(next_action)]),
                    "last_action_idx": int(last_br),
                    "last_action_kbps": int(util_mod.VIDEO_BIT_RATE[int(last_br)]),
                    "oracle_action_kbps": int(oracle_action_kbps),
                    "oracle_best_reward_mean": float(oracle_best_mean),
                    "policy_reward_mean": float(policy_mean),
                    "oracle_gap_mean": float(gap_mean),
                    "oracle_gap_pct": float(gap_pct),
                    "shield_min_idx": int(shield_stats.get("shield/min_idx", -1)) if shield is not None else -1,
                    "shield_max_idx": int(shield_stats.get("shield/max_idx", -1)) if shield is not None else -1,
                    "shield_inconsistent": int(shield_stats.get("shield/inconsistent", 0)) if shield is not None else 0,
                    "shield_matched_min": int(shield_stats.get("shield/matched_min", 0)) if shield is not None else 0,
                    "shield_matched_max": int(shield_stats.get("shield/matched_max", 0)) if shield is not None else 0,
                    "shield_selected_action_idx": int(shield_stats.get("shield/selected_action_idx", next_action))
                    if shield is not None
                    else int(next_action),
                    "shield_min_kbps": float(shield_stats.get("shield/min_kbps", float("nan")))
                    if shield is not None
                    else float("nan"),
                    "shield_max_kbps": float(shield_stats.get("shield/max_kbps", float("nan")))
                    if shield is not None
                    else float("nan"),
                }
                for i in range(int(util_mod.A_DIM)):
                    v = q_oracle_by_first[i] if i < len(q_oracle_by_first) else None
                    row[f"oracle_q_reward_mean_by_action{i}"] = float(v) if v is not None else float("nan")

                for i in range(int(util_mod.S_LEN)):
                    row[f"bitrate_hist_kbps_{i}"] = int(sd["bitrate_hist_kbps"][i])
                    row[f"buffer_hist_sec_{i}"] = float(sd["buffer_hist_sec"][i])
                    row[f"throughput_hist_mbps_{i}"] = float(sd["throughput_hist_mbps"][i])
                    row[f"delay_hist_ms_{i}"] = float(sd["delay_hist_ms"][i])
                row["video_chunk_remain"] = int(sd["video_chunk_remain"])

                for i in range(int(util_mod.A_DIM)):
                    row[f"next_chunk_size_bytes_{i}"] = int(sd["next_chunk_sizes_bytes"][i])

                writer.writerow(row)
                total_logged += 1

                if int(args.max_logged_states) > 0 and int(total_logged) >= int(args.max_logged_states):
                    break

                bit_rate = int(next_action)
                last_bit_rate = int(bit_rate)

                if int(args.progress_every) > 0 and int(total_chunks) % int(args.progress_every) == 0:
                    elapsed = float(time.time() - start_time)
                    rate = float(total_chunks) / max(1.0, elapsed)
                    print(
                        f"* progress chunks={total_chunks} logged={total_logged} considered={total_states_considered} rate={rate:.2f} chunks/s",
                        flush=True,
                    )

            if int(args.max_logged_states) > 0 and int(total_logged) >= int(args.max_logged_states):
                break

    elapsed = float(time.time() - start_time)
    print(
        f"* wrote {total_logged} states to {out_path} from {len(all_file_names)} traces "
        f"(considered={total_states_considered}, chunks={total_chunks}, elapsed_s={elapsed:.2f})",
        flush=True,
    )


if __name__ == "__main__":
    main()
