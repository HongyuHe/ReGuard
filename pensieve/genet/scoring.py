from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .runtime import PensieveRuntime
from .trace_family import CookedTrace


ObjectiveName = Literal["mpc_gap", "oracle_gap"]
MPC_FUTURE_CHUNK_COUNT = 5
ACTION_COMBOS = np.asarray(
    list(itertools.product(range(6), repeat=MPC_FUTURE_CHUNK_COUNT)),
    dtype=np.int64,
)


@dataclass(frozen=True)
class TraceScore:
    baseline_qoe: float
    pensieve_qoe: float
    gap: float
    baseline_name: str


def _update_state(
    state: np.ndarray,
    *,
    bit_rate: int,
    buffer_s: float,
    video_chunk_size: int,
    delay_ms: float,
    next_video_chunk_sizes: list[int],
    video_chunk_remain: int,
    util_mod,
) -> np.ndarray:
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


def _make_env(trace: CookedTrace, runtime: PensieveRuntime, *, seed: int):
    return runtime.fixed_env_mod.Environment(
        all_cooked_time=[list(trace.timestamps_s)],
        all_cooked_bw=[list(trace.bandwidths_mbps)],
        random_seed=int(seed),
    )


def _rollout_pensieve(trace: CookedTrace, runtime: PensieveRuntime, *, seed: int) -> float:
    util_mod = runtime.util_mod
    env_inst = _make_env(trace, runtime, seed=seed)

    state = np.zeros((int(util_mod.S_INFO), int(util_mod.S_LEN)), dtype=np.float64)
    last_bit_rate = int(util_mod.DEFAULT_QUALITY)
    bit_rate = int(util_mod.DEFAULT_QUALITY)
    total_reward = 0.0

    while True:
        (
            delay_ms,
            _sleep_time_ms,
            buffer_s,
            rebuf_s,
            video_chunk_size,
            next_video_chunk_sizes,
            end_of_video,
            video_chunk_remain,
        ) = env_inst.get_video_chunk(int(bit_rate))

        total_reward += float(util_mod.calculate_reward(int(bit_rate), int(last_bit_rate), float(rebuf_s)))
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
        last_bit_rate = int(bit_rate)
        if bool(end_of_video):
            break

        probs = runtime.actor.predict(np.reshape(state, (1, int(util_mod.S_INFO), int(util_mod.S_LEN))))
        probs = np.asarray(probs, dtype=np.float64).reshape(-1)
        bit_rate = int(np.argmax(probs))

    return float(total_reward)


def _choose_mpc_action(
    *,
    size_video_array: np.ndarray,
    future_chunk_cnt: int,
    buffer_size_s: float,
    current_bit_rate: int,
    last_index: int,
    future_bandwidth_mbps: float,
    util_mod,
) -> int:
    if int(future_chunk_cnt) <= 0:
        return int(current_bit_rate)

    best_reward = float("-inf")
    best_action = int(current_bit_rate)
    row_len = int(size_video_array.shape[1])
    future_bandwidth = max(float(future_bandwidth_mbps), 1e-6)

    for combo in ACTION_COMBOS:
        reward = 0.0
        current_buffer_s = float(buffer_size_s)
        last_quality = int(current_bit_rate)
        for position in range(int(future_chunk_cnt)):
            chunk_quality = int(combo[int(position)])
            index = int(last_index) + int(position) + 1
            download_time_s = float(size_video_array[chunk_quality, int(index % row_len)]) / float(util_mod.env.B_IN_MB)
            download_time_s = float(download_time_s / future_bandwidth)
            if float(current_buffer_s) < float(download_time_s):
                rebuffer_s = float(download_time_s - current_buffer_s)
                current_buffer_s = 0.0
            else:
                rebuffer_s = 0.0
                current_buffer_s = float(current_buffer_s - download_time_s)

            current_buffer_s += float(util_mod.env.VIDEO_CHUNCK_LEN) / float(util_mod.env.MILLISECONDS_IN_SECOND)
            reward += float(util_mod.calculate_reward(int(chunk_quality), int(last_quality), float(rebuffer_s)))
            last_quality = int(chunk_quality)

        if float(reward) >= float(best_reward):
            best_reward = float(reward)
            best_action = int(combo[0])

    return int(best_action)


def _rollout_mpc(trace: CookedTrace, runtime: PensieveRuntime, *, seed: int) -> float:
    util_mod = runtime.util_mod
    env_inst = _make_env(trace, runtime, seed=seed)
    size_video_array = np.asarray([env_inst.video_size[i] for i in sorted(env_inst.video_size)], dtype=np.float64)

    last_bit_rate = int(util_mod.DEFAULT_QUALITY)
    bit_rate = int(util_mod.DEFAULT_QUALITY)
    total_reward = 0.0
    throughput_window: list[float] = []
    past_errors: list[float] = []
    past_bandwidth_ests: list[float] = []

    while True:
        (
            delay_ms,
            _sleep_time_ms,
            buffer_s,
            rebuf_s,
            video_chunk_size,
            _next_video_chunk_sizes,
            end_of_video,
            video_chunk_remain,
        ) = env_inst.get_video_chunk(int(bit_rate))

        total_reward += float(util_mod.calculate_reward(int(bit_rate), int(last_bit_rate), float(rebuf_s)))
        throughput_window.append(float(video_chunk_size) / float(delay_ms) / float(util_mod.M_IN_K))
        last_bit_rate = int(bit_rate)
        if bool(end_of_video):
            break

        if past_bandwidth_ests:
            current_error = abs(float(past_bandwidth_ests[-1]) - float(throughput_window[-1])) / max(
                float(throughput_window[-1]),
                1e-6,
            )
        else:
            current_error = 0.0
        past_errors.append(float(current_error))

        if len(throughput_window) > 5:
            past_bandwidths = throughput_window[-5:]
        else:
            past_bandwidths = list(throughput_window)

        harmonic_bandwidth = float(1.0 / np.mean(1.0 / np.asarray(past_bandwidths, dtype=np.float64)))
        error_pos = -5 if len(past_errors) >= 5 else -len(past_errors)
        max_error = float(max(past_errors[error_pos:]))
        future_bandwidth = float(harmonic_bandwidth / (1.0 + max_error))
        past_bandwidth_ests.append(float(harmonic_bandwidth))

        last_index = int(runtime.fixed_env_mod.TOTAL_VIDEO_CHUNCK - int(video_chunk_remain) - 1)
        future_chunk_cnt = int(
            min(
                int(MPC_FUTURE_CHUNK_COUNT),
                int(runtime.fixed_env_mod.TOTAL_VIDEO_CHUNCK) - int(last_index) - 1,
            )
        )
        bit_rate = _choose_mpc_action(
            size_video_array=size_video_array,
            future_chunk_cnt=int(future_chunk_cnt),
            buffer_size_s=float(buffer_s),
            current_bit_rate=int(bit_rate),
            last_index=int(last_index),
            future_bandwidth_mbps=float(future_bandwidth),
            util_mod=util_mod,
        )

    return float(total_reward)


def _rollout_oracle(trace: CookedTrace, runtime: PensieveRuntime, *, seed: int, oracle_k: int) -> float:
    util_mod = runtime.util_mod
    env_oracle = _make_env(trace, runtime, seed=seed)
    last_bit_rate = int(util_mod.DEFAULT_QUALITY)
    total_reward = 0.0

    while True:
        remaining = int(runtime.fixed_env_mod.TOTAL_VIDEO_CHUNCK) - int(env_oracle.video_chunk_counter)
        k_eff = int(min(max(int(oracle_k), 1), max(int(remaining), 1)))
        best_reward, best_seq, _worst_reward, _worst_seq = util_mod.find_optimal_k_step_reward_oracle(
            env_oracle,
            int(last_bit_rate),
            int(k_eff),
        )
        if best_reward is None or best_seq is None or len(best_seq) == 0:
            raise RuntimeError("* oracle could not find a valid K-step sequence")

        bit_rate = int(best_seq[0])
        (
            _delay_ms,
            _sleep_time_ms,
            _buffer_s,
            rebuf_s,
            _video_chunk_size,
            _next_video_chunk_sizes,
            end_of_video,
            _video_chunk_remain,
        ) = env_oracle.get_video_chunk(int(bit_rate))

        total_reward += float(util_mod.calculate_reward(int(bit_rate), int(last_bit_rate), float(rebuf_s)))
        last_bit_rate = int(bit_rate)
        if bool(end_of_video):
            break

    return float(total_reward)


class TraceScorer:
    def __init__(self, runtime: PensieveRuntime, *, oracle_k: int = 5) -> None:
        self.runtime = runtime
        self.oracle_k = int(max(int(oracle_k), 1))

    def score_trace(self, trace: CookedTrace, *, objective: ObjectiveName, seed: int) -> TraceScore:
        pensieve_qoe = _rollout_pensieve(trace, self.runtime, seed=int(seed))
        if str(objective) == "mpc_gap":
            baseline_qoe = _rollout_mpc(trace, self.runtime, seed=int(seed))
            baseline_name = "mpc"
        elif str(objective) == "oracle_gap":
            baseline_qoe = _rollout_oracle(trace, self.runtime, seed=int(seed), oracle_k=int(self.oracle_k))
            baseline_name = "oracle"
        else:
            raise ValueError(f"* unknown objective: {objective}")

        return TraceScore(
            baseline_qoe=float(baseline_qoe),
            pensieve_qoe=float(pensieve_qoe),
            gap=float(baseline_qoe - pensieve_qoe),
            baseline_name=str(baseline_name),
        )
