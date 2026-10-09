from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from online.pensieve_bridge import add_pensieve_src_to_path

from .cooked_trace_io import load_cooked_traces_dir
from .curriculum import BandwidthCurriculum
from .runtime import resolve_repo_path
from .scoring import _update_state


@dataclass(frozen=True)
class LoadedTraceSet:
    cooked_time: list[list[float]]
    cooked_bw: list[list[float]]
    names: list[str]


@dataclass(frozen=True)
class FineTuneResult:
    model_path: str
    stats: dict[str, float]


def load_trace_set(trace_dir: str, *, limit: int = 0) -> LoadedTraceSet:
    cooked_time, cooked_bw, names = load_cooked_traces_dir(str(trace_dir), limit=int(limit))
    return LoadedTraceSet(
        cooked_time=[list(x) for x in cooked_time],
        cooked_bw=[list(x) for x in cooked_bw],
        names=[str(x) for x in names],
    )


def _load_training_modules() -> tuple[Any, Any, Any]:
    add_pensieve_src_to_path()

    import fixed_env as env_mod  # type: ignore
    import ppo2 as network_mod  # type: ignore
    import util as util_mod  # type: ignore

    return network_mod, util_mod, env_mod


def _safe_mean(values: list[float]) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _safe_percentile(values: list[float], q: float) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), float(q)))


def _pop_shuffled_index(order: list[int], size: int, rng: np.random.RandomState) -> int:
    if int(size) <= 0:
        raise ValueError("* trace set is empty")
    if len(order) == 0:
        order.extend([int(i) for i in range(int(size))])
        rng.shuffle(order)
    return int(order.pop())


def _rollout_on_trace(
    *,
    actor: Any,
    trace_time: list[float],
    trace_bw: list[float],
    rollout_seed: int,
    rng: np.random.RandomState,
    util_mod: Any,
    env_mod: Any,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[float], float]:
    net_env = env_mod.Environment(
        all_cooked_time=[list(trace_time)],
        all_cooked_bw=[list(trace_bw)],
        random_seed=int(rollout_seed),
    )

    state = np.zeros((int(util_mod.S_INFO), int(util_mod.S_LEN)), dtype=np.float64)
    last_bit_rate = int(util_mod.DEFAULT_QUALITY)
    bit_rate = int(util_mod.DEFAULT_QUALITY)

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
        util_mod=util_mod,
    )

    s_batch: list[np.ndarray] = []
    a_batch: list[np.ndarray] = []
    p_batch: list[np.ndarray] = []
    r_batch: list[float] = []
    episode_return = 0.0

    while True:
        s_batch.append(np.array(state, copy=True))
        action_prob = actor.predict(np.reshape(state, (1, int(util_mod.S_INFO), int(util_mod.S_LEN))))
        action_prob = np.asarray(action_prob, dtype=np.float64).reshape(-1)
        noise = rng.gumbel(size=len(action_prob))
        bit_rate = int(np.argmax(np.log(action_prob + 1e-12) + noise))

        action_vec = np.zeros(int(util_mod.A_DIM), dtype=np.float64)
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
        reward = float(util_mod.calculate_reward(int(bit_rate), int(last_bit_rate), float(rebuf_s)))
        last_bit_rate = int(bit_rate)
        episode_return += float(reward)

        a_batch.append(action_vec)
        p_batch.append(np.array(action_prob, copy=True))
        r_batch.append(float(reward))

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
        if bool(end_of_video):
            break

    return s_batch, a_batch, p_batch, r_batch, float(episode_return)


def fine_tune_on_curriculum(
    *,
    model_path: str,
    out_model_path: str,
    curriculum: BandwidthCurriculum,
    episodes: int,
    batch_episodes: int,
    learning_rate: float,
    seed: int,
    real_trace_set: LoadedTraceSet | None = None,
    real_trace_prob: float = 0.0,
    init_entropy_weight: float | None = None,
    freeze_entropy_weight: bool = False,
) -> FineTuneResult:
    if int(episodes) <= 0:
        raise ValueError("* episodes must be positive")
    if int(batch_episodes) <= 0:
        raise ValueError("* batch_episodes must be positive")

    real_prob = float(min(max(float(real_trace_prob), 0.0), 1.0))
    if real_trace_set is None and float(real_prob) > 0.0:
        raise ValueError("* real_trace_prob > 0 requires real_trace_set")

    network_mod, util_mod, env_mod = _load_training_modules()
    actor = network_mod.Network(
        state_dim=[int(util_mod.S_INFO), int(util_mod.S_LEN)],
        action_dim=int(util_mod.A_DIM),
        learning_rate=float(learning_rate),
    )
    checkpoint_info = actor.load_model(
        resolve_repo_path(str(model_path)),
        restore_optimizer=True,
        restore_entropy_weight=True,
    )
    entropy_init_applied = False
    if not bool(checkpoint_info.get("restored_entropy_weight", False)) and init_entropy_weight is not None:
        actor.set_entropy_weight(float(init_entropy_weight))
        entropy_init_applied = True
    actor.set_freeze_entropy_weight(bool(freeze_entropy_weight))
    entropy_weight_start = float(actor._entropy_weight)

    rng = np.random.RandomState(int(seed))
    real_order: list[int] = []
    real_count = 0 if real_trace_set is None else int(len(real_trace_set.cooked_time))

    train_s: list[np.ndarray] = []
    train_a: list[np.ndarray] = []
    train_p: list[np.ndarray] = []
    train_v: list[float] = []
    update_episode_returns: list[float] = []
    update_idx = 0
    current_episode = -1
    interrupted = False

    episode_returns: list[float] = []
    synthetic_episode_returns: list[float] = []
    real_episode_returns: list[float] = []
    update_loss_total: list[float] = []
    update_loss_policy: list[float] = []
    update_loss_value: list[float] = []
    update_loss_entropy: list[float] = []
    synthetic_episode_count = 0
    real_episode_count = 0
    component_episode_counts = [0 for _ in range(len(curriculum.components))]
    sampled_min_bw_values: list[float] = []
    sampled_max_bw_values: list[float] = []
    sampled_interval_values: list[float] = []
    t0 = time.perf_counter()

    def _run_update() -> None:
        nonlocal update_idx
        nonlocal train_s
        nonlocal train_a
        nonlocal train_p
        nonlocal train_v
        nonlocal update_episode_returns
        if len(update_episode_returns) == 0:
            return

        update_idx += 1
        train_stats = actor.train(
            np.stack(train_s, axis=0),
            np.vstack(train_a),
            np.vstack(train_p),
            np.asarray(train_v, dtype=np.float64).reshape(-1, 1),
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

        train_s = []
        train_a = []
        train_p = []
        train_v = []
        update_episode_returns = []

    try:
        for ep in range(int(episodes)):
            current_episode = int(ep)
            use_real = bool(real_count > 0 and rng.rand() < float(real_prob))

            if bool(use_real):
                trace_idx = _pop_shuffled_index(real_order, real_count, rng)
                trace_time = list(real_trace_set.cooked_time[int(trace_idx)]) if real_trace_set is not None else []
                trace_bw = list(real_trace_set.cooked_bw[int(trace_idx)]) if real_trace_set is not None else []
                real_episode_count += 1
            else:
                sampled = curriculum.sample_trace(
                    rng=rng,
                    trace_seed=int(int(seed) + 1_000_000 + int(ep)),
                )
                trace_time = list(sampled.trace.timestamps_s)
                trace_bw = list(sampled.trace.bandwidths_mbps)
                synthetic_episode_count += 1
                component_episode_counts[int(sampled.component_index)] += 1
                sampled_min_bw_values.append(float(sampled.params.min_bw_mbps))
                sampled_max_bw_values.append(float(sampled.params.max_bw_mbps))
                sampled_interval_values.append(float(sampled.params.bw_change_interval))

            s_batch, a_batch, p_batch, r_batch, episode_return = _rollout_on_trace(
                actor=actor,
                trace_time=trace_time,
                trace_bw=trace_bw,
                rollout_seed=int(rng.randint(0, 2**31 - 1)),
                rng=rng,
                util_mod=util_mod,
                env_mod=env_mod,
            )
            v_batch = actor.compute_v(s_batch, a_batch, r_batch, terminal=True)

            train_s.extend(s_batch)
            train_a.extend(a_batch)
            train_p.extend(p_batch)
            train_v.extend([float(v) for v in v_batch])
            update_episode_returns.append(float(episode_return))

            episode_returns.append(float(episode_return))
            if bool(use_real):
                real_episode_returns.append(float(episode_return))
            else:
                synthetic_episode_returns.append(float(episode_return))

            if len(update_episode_returns) >= int(batch_episodes):
                _run_update()
    except KeyboardInterrupt:
        interrupted = True

    _run_update()

    os.makedirs(os.path.dirname(str(out_model_path)), exist_ok=True)
    actor.save_model(str(out_model_path))

    stats: dict[str, float] = {
        "train/episodes": float(int(len(episode_returns))),
        "train/episodes_target": float(int(episodes)),
        "train/updates": float(int(update_idx)),
        "train/time_ms": float((time.perf_counter() - t0) * 1000.0),
        "train/episode_return-mean": _safe_mean(episode_returns),
        "train/episode_return-p50": _safe_percentile(episode_returns, 50),
        "train/episode_return-p95": _safe_percentile(episode_returns, 95),
        "train/episode_return_synth-mean": _safe_mean(synthetic_episode_returns),
        "train/episode_return_real-mean": _safe_mean(real_episode_returns),
        "train/synthetic_episode_count": float(int(synthetic_episode_count)),
        "train/real_episode_count": float(int(real_episode_count)),
        "train/sampled_min_bw_mbps-mean": _safe_mean(sampled_min_bw_values),
        "train/sampled_max_bw_mbps-mean": _safe_mean(sampled_max_bw_values),
        "train/sampled_bw_change_interval-mean": _safe_mean(sampled_interval_values),
        "train/loss/total-mean": _safe_mean(update_loss_total),
        "train/loss/policy-mean": _safe_mean(update_loss_policy),
        "train/loss/value-mean": _safe_mean(update_loss_value),
        "train/loss/entropy-mean": _safe_mean(update_loss_entropy),
        "train/checkpoint_legacy_format": float(1.0 if bool(checkpoint_info.get("legacy_checkpoint", False)) else 0.0),
        "train/checkpoint_restored_optimizer": float(
            1.0 if bool(checkpoint_info.get("restored_optimizer", False)) else 0.0
        ),
        "train/checkpoint_restored_entropy_weight": float(
            1.0 if bool(checkpoint_info.get("restored_entropy_weight", False)) else 0.0
        ),
        "train/checkpoint_entropy_weight_initialized": float(1.0 if bool(entropy_init_applied) else 0.0),
        "train/checkpoint_entropy_weight_frozen": float(1.0 if bool(freeze_entropy_weight) else 0.0),
        "train/checkpoint_entropy_weight_start": float(entropy_weight_start),
        "train/interrupted": float(1.0 if bool(interrupted) else 0.0),
    }
    for component_index, component_count in enumerate(component_episode_counts):
        stats[f"train/component_{int(component_index):03d}_episodes"] = float(int(component_count))

    return FineTuneResult(
        model_path=str(out_model_path),
        stats=stats,
    )
