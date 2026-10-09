from __future__ import annotations

import os
from collections import deque
from typing import Any

import numpy as np

from .chunk_simulator import ChunkLevelPensieveSimulator, compute_normalized_throughput_delay
from .oracle import ExactWindowOracle
from .pensieve_bridge import add_pensieve_src_to_path, import_pensieve


def _require_gym() -> tuple[Any, Any]:
    try:
        import gymnasium as gym  # type: ignore
        from gymnasium import spaces  # type: ignore

        return gym, spaces
    except Exception:
        try:
            import gym  # type: ignore
            from gym import spaces  # type: ignore

            return gym, spaces
        except Exception as e:  # pragma: no cover
            raise RuntimeError("* Missing dependency: install `gymnasium` (or `gym`) to use online adversary env") from e


gym, spaces = _require_gym()

#* Use a positive floor to avoid division-by-zero when bw_min_mbps is 0.
BW_FLOOR_Mbps = 1e-6


def _resolve_path(repo_relative_or_abs: str) -> str:
    expanded = os.path.expandvars(os.path.expanduser(str(repo_relative_or_abs)))
    if os.path.isabs(expanded):
        return expanded
    cwd_path = os.path.abspath(expanded)
    if os.path.exists(cwd_path):
        return cwd_path

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    under_src = os.path.join(repo_root, "Pensieve-PPO", "src", expanded)
    if os.path.exists(under_src):
        return under_src

    under_root = os.path.join(repo_root, expanded)
    if os.path.exists(under_root):
        return under_root

    return under_src


def _update_pensieve_state(
    state: np.ndarray,
    *,
    bit_rate: int,
    buffer_s: float,
    video_chunk_size: int,
    delay_ms: float,
    next_video_chunk_sizes: list[int],
    video_chunk_remain: int,
) -> np.ndarray:
    modules = import_pensieve()
    pensieve_env = modules["env"]

    new_state = np.roll(state, -1, axis=1).astype(np.float64, copy=False)
    new_state[0, -1] = float(pensieve_env.VIDEO_BIT_RATE[int(bit_rate)]) / float(np.max(pensieve_env.VIDEO_BIT_RATE))
    new_state[1, -1] = float(buffer_s) / float(pensieve_env.BUFFER_NORM_FACTOR)
    new_state[2, -1] = float(video_chunk_size) / float(delay_ms) / float(pensieve_env.M_IN_K)
    new_state[3, -1] = float(delay_ms) / float(pensieve_env.M_IN_K) / float(pensieve_env.BUFFER_NORM_FACTOR)
    new_state[4, : int(pensieve_env.A_DIM)] = (
        np.asarray(next_video_chunk_sizes, dtype=np.float64) / float(pensieve_env.M_IN_K) / float(pensieve_env.M_IN_K)
    )
    new_state[5, -1] = float(min(int(video_chunk_remain), int(pensieve_env.CHUNK_TIL_VIDEO_END_CAP))) / float(
        pensieve_env.CHUNK_TIL_VIDEO_END_CAP
    )
    return new_state


def _step_obs_from_state(state: np.ndarray) -> np.ndarray:
    modules = import_pensieve()
    pensieve_env = modules["env"]

    parts = [
        np.asarray([float(state[0, -1])], dtype=np.float32),
        np.asarray([float(state[1, -1])], dtype=np.float32),
        np.asarray([float(state[2, -1])], dtype=np.float32),
        np.asarray([float(state[3, -1])], dtype=np.float32),
        np.asarray(state[4, : int(pensieve_env.A_DIM)], dtype=np.float32),
        np.asarray([float(state[5, -1])], dtype=np.float32),
    ]
    return np.concatenate(parts, axis=0)


def _window_qoe(*, bitrate_idxs: list[int], rebuf_s: list[float]) -> float:
    modules = import_pensieve()
    pensieve_env = modules["env"]

    br_mbps = (np.asarray(pensieve_env.VIDEO_BIT_RATE, dtype=np.float64) / float(pensieve_env.M_IN_K)).astype(np.float64)
    rates = br_mbps[np.asarray(bitrate_idxs, dtype=np.int64)]
    quality_sum = float(np.sum(rates))
    rebuf_sum = float(np.sum(np.asarray(rebuf_s, dtype=np.float64)))
    if len(rates) <= 1:
        smooth_sum = 0.0
    else:
        smooth_sum = float(np.sum(np.abs(rates[1:] - rates[:-1])))
    return float(quality_sum - float(pensieve_env.REBUF_PENALTY) * rebuf_sum - smooth_sum)


class OnlineBandwidthAdversaryEnv(gym.Env):
    metadata = {"render_modes": []}
    REWARD_OBJECTIVES = frozenset({"max_regret", "weighted_max_regret", "oracle_floor_max_regret"})

    def __init__(
        self,
        *,
        pensieve_model_path: str = "ppo/nn_model_ep_426300.pth",
        seed: int = 42,
        bw_min_mbps: float = 0.8,
        bw_max_mbps: float = 4.8,
        obs_history_len: int = 10,
        oracle_window_len: int = 4,
        smooth_penalty_scale: float = 1.0,
        reward_objective: str = "weighted_max_regret",
        reward_tau: float = 0.0,
        reward_beta: float = 1.0,
        deterministic_protocol: bool = True,
        shield_min_rules_file: str | None = None,
        shield_max_rules_file: str | None = None,
        shield_action_selection: str = "argmax",
    ) -> None:
        super().__init__()

        add_pensieve_src_to_path()
        modules = import_pensieve()
        pensieve_env = modules["env"]
        fixed_env = modules["fixed_env"]

        self._seed = int(seed)
        self._rng = np.random.RandomState(int(seed))

        self.bw_min_mbps = float(bw_min_mbps)
        self.bw_max_mbps = float(bw_max_mbps)
        self.obs_history_len = int(obs_history_len)
        self.oracle_window_len = int(oracle_window_len)
        self.smooth_penalty_scale = float(smooth_penalty_scale)
        self.reward_objective = str(reward_objective)
        self.reward_tau = float(reward_tau)
        self.reward_beta = float(reward_beta)
        self.deterministic_protocol = bool(deterministic_protocol)

        if (shield_min_rules_file is None) != (shield_max_rules_file is None):
            raise ValueError("* Provide both shield_min_rules_file and shield_max_rules_file, or neither")
        if str(shield_action_selection) not in {"argmax", "gumbel"}:
            raise ValueError("* shield_action_selection must be one of: argmax, gumbel")
        if self.reward_objective not in self.REWARD_OBJECTIVES:
            raise ValueError(
                "* reward_objective must be one of: " + ", ".join(sorted(str(x) for x in self.REWARD_OBJECTIVES))
            )
        if not np.isfinite(self.reward_tau):
            raise ValueError("* reward_tau must be finite")
        if not np.isfinite(self.reward_beta) or self.reward_beta < 0.0:
            raise ValueError("* reward_beta must be finite and non-negative")

        if not np.isfinite(self.bw_min_mbps) or not np.isfinite(self.bw_max_mbps):
            raise ValueError("* bw_min_mbps and bw_max_mbps must be finite")
        if self.bw_min_mbps <= 0.0:
            raise ValueError("* bw_min_mbps must be > 0.0 for log-scale action mapping")
        if self.bw_max_mbps < self.bw_min_mbps:
            raise ValueError("* bw_max_mbps must be >= bw_min_mbps")
        if self.bw_min_mbps < float(BW_FLOOR_Mbps):
            raise ValueError("* bw_min_mbps must be >= BW_FLOOR_Mbps for log-scale action mapping")
        if self.obs_history_len <= 0:
            raise ValueError("* obs_history_len must be positive")
        if self.oracle_window_len <= 0:
            raise ValueError("* oracle_window_len must be positive")

        self._log_bw_min = float(np.log(self.bw_min_mbps))
        self._log_bw_span = float(np.log(self.bw_max_mbps) - self._log_bw_min)

        self._protocol_state = np.zeros((int(pensieve_env.S_INFO), int(pensieve_env.S_LEN)), dtype=np.float64)
        self._protocol_last_bitrate = int(pensieve_env.DEFAULT_QUALITY)

        import ppo2 as network  # type: ignore

        self._actor = network.Network(
            state_dim=[int(pensieve_env.S_INFO), int(pensieve_env.S_LEN)],
            action_dim=int(pensieve_env.A_DIM),
            learning_rate=1e-4,
        )
        self._actor.load_model(_resolve_path(pensieve_model_path))

        self._sim = ChunkLevelPensieveSimulator(seed=int(seed))
        self._oracle = ExactWindowOracle(window_size=int(self.oracle_window_len))

        self._obs_step_dim = 1 + 1 + 1 + 1 + int(pensieve_env.A_DIM) + 1
        self._obs_hist: deque[np.ndarray] = deque(maxlen=int(self.obs_history_len))

        self._last_video_chunk_size_bytes = 0
        self._shield_action_selection = str(shield_action_selection)
        self._shield = None
        self._shield_stats_last: dict[str, float | int] = {}
        if shield_min_rules_file is not None and shield_max_rules_file is not None:
            from rules.shield.shield_runtime import ShieldBounds, load_rules

            min_rules = load_rules(_resolve_path(str(shield_min_rules_file)), output_var="MinBitrateKbps")
            max_rules = load_rules(_resolve_path(str(shield_max_rules_file)), output_var="MaxBitrateKbps")
            self._shield = ShieldBounds(min_rules=min_rules, max_rules=max_rules)

        self._bw_hist: list[float] = []
        self._bitrate_hist: list[int] = []
        self._rebuf_s_hist: list[float] = []
        self._delay_ms_hist: list[float] = []
        self._buffer_before_ms_hist: list[float] = []
        self._segments: list[dict[str, float | str]] = []

        self.action_space = spaces.Box(
            low=np.asarray([0.0], dtype=np.float32),
            high=np.asarray([1.0], dtype=np.float32),
            dtype=np.float32,
        )
        high = np.full((int(self.obs_history_len) * int(self._obs_step_dim),), 1e9, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self._max_chunks = int(fixed_env.TOTAL_VIDEO_CHUNCK)

    def seed(self, seed: int | None = None) -> list[int]:
        if seed is None:
            seed = self._seed
        self._seed = int(seed)
        self._rng = np.random.RandomState(int(seed))
        return [int(seed)]

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self.seed(int(seed))

        modules = import_pensieve()
        pensieve_env = modules["env"]

        self._sim.reset()
        self._protocol_state = np.zeros((int(pensieve_env.S_INFO), int(pensieve_env.S_LEN)), dtype=np.float64)
        self._protocol_last_bitrate = int(pensieve_env.DEFAULT_QUALITY)
        self._last_video_chunk_size_bytes = 0
        self._shield_stats_last = {}

        self._obs_hist.clear()
        initial = _step_obs_from_state(self._protocol_state)
        for _ in range(int(self.obs_history_len)):
            self._obs_hist.append(np.asarray(initial, dtype=np.float32))

        self._bw_hist.clear()
        self._bitrate_hist.clear()
        self._rebuf_s_hist.clear()
        self._delay_ms_hist.clear()
        self._buffer_before_ms_hist.clear()
        self._segments.clear()

        obs = np.concatenate(list(self._obs_hist), axis=0).astype(np.float32, copy=False)
        return obs, {}

    def _protocol_choose_bitrate(self) -> int:
        modules = import_pensieve()
        pensieve_env = modules["env"]

        if int(self._sim.video_chunk_counter) == 0:
            return int(pensieve_env.DEFAULT_QUALITY)

        pi = self._actor.predict(np.reshape(self._protocol_state, (1, int(pensieve_env.S_INFO), int(pensieve_env.S_LEN))))
        pi = np.asarray(pi, dtype=np.float64).reshape(-1)
        if self._shield is not None:
            chosen, stats = self._shield.select_action(
                p=pi,
                state=self._protocol_state,
                last_bitrate_idx=int(self._protocol_last_bitrate),
                video_chunk_size_bytes=int(self._last_video_chunk_size_bytes),
                action_selection=str(self._shield_action_selection),
                rng=self._rng,
            )
            self._shield_stats_last = dict(stats)
            return int(chosen)

        if not bool(self.deterministic_protocol):
            pi = pi / max(float(np.sum(pi)), 1e-12)
            return int(self._rng.choice(np.arange(int(pensieve_env.A_DIM)), p=pi))
        return int(np.argmax(pi))

    def step(self, action):
        modules = import_pensieve()
        pensieve_env = modules["env"]
        fixed_env = modules["fixed_env"]

        action_raw = float(np.asarray(action, dtype=np.float64).reshape(-1)[0])
        action_clipped = float(np.clip(float(action_raw), 0.0, 1.0))
        bw_mapped = float(np.exp(float(self._log_bw_min) + float(action_clipped) * float(self._log_bw_span)))
        bw_clipped = float(np.clip(float(bw_mapped), self.bw_min_mbps, self.bw_max_mbps))
        bw = float(max(float(bw_clipped), float(BW_FLOOR_Mbps)))
        buffer_before_ms = float(self._sim.buffer_ms)

        bit_rate = int(self._protocol_choose_bitrate())
        sim_res = self._sim.simulate_chunk(bw_mbps=float(bw), quality=int(bit_rate))

        chunk_idx = int(len(self._bw_hist))
        chunk_size_bytes = int(self._sim.video_sizes[int(bit_rate), int(chunk_idx)])

        next_sizes = self._sim.get_next_chunk_sizes()
        video_chunk_remain = int(fixed_env.TOTAL_VIDEO_CHUNCK) - int(self._sim.video_chunk_counter)

        self._protocol_state = _update_pensieve_state(
            self._protocol_state,
            bit_rate=int(bit_rate),
            buffer_s=float(sim_res.buffer_after_s),
            video_chunk_size=int(chunk_size_bytes),
            delay_ms=float(sim_res.delay_ms),
            next_video_chunk_sizes=[int(x) for x in next_sizes],
            video_chunk_remain=int(video_chunk_remain),
        )
        self._protocol_last_bitrate = int(bit_rate)
        self._last_video_chunk_size_bytes = int(chunk_size_bytes)

        obs_step = _step_obs_from_state(self._protocol_state)
        self._obs_hist.append(np.asarray(obs_step, dtype=np.float32))
        obs = np.concatenate(list(self._obs_hist), axis=0).astype(np.float32, copy=False)

        self._bw_hist.append(float(bw))
        self._bitrate_hist.append(int(bit_rate))
        self._rebuf_s_hist.append(float(sim_res.rebuf_s))
        self._delay_ms_hist.append(float(sim_res.delay_ms))
        self._buffer_before_ms_hist.append(float(buffer_before_ms))

        self._segments.append({"kind": "download", "bw_mbps": float(bw), "duration_s": float(sim_res.download_time_s)})
        if float(sim_res.sleep_time_ms) > 0.0:
            self._segments.append(
                {"kind": "sleep", "bw_mbps": float(bw), "duration_s": float(sim_res.sleep_time_ms) / 1000.0}
            )

        t = int(len(self._bw_hist) - 1)
        win = min(int(self.oracle_window_len), t + 1)
        k = t - win + 1
        chunk_indices = [int(i) for i in range(int(k), int(t) + 1)]
        bw_window = [float(x) for x in self._bw_hist[int(k) : int(t) + 1]]

        r_opt = float(self._oracle.compute(start_buffer_ms=float(self._buffer_before_ms_hist[int(k)]), chunk_indices=chunk_indices, bw_mbps=bw_window))
        r_protocol = float(
            _window_qoe(
                bitrate_idxs=[int(x) for x in self._bitrate_hist[int(k) : int(t) + 1]],
                rebuf_s=[float(x) for x in self._rebuf_s_hist[int(k) : int(t) + 1]],
            )
        )
        smooth_pen = 0.0
        if t > 0:
            smooth_pen = abs(float(self._bw_hist[t]) - float(self._bw_hist[t - 1]))
        gap = float(r_opt - r_protocol)
        weighted_gap = float(gap * r_opt)
        oracle_above_tau = float(max(float(r_opt) - float(self.reward_tau), 0.0))
        oracle_below_tau = float(max(float(self.reward_tau) - float(r_opt), 0.0))
        oracle_floor_weighted_gap = float(gap * oracle_above_tau)
        oracle_floor_penalty = float(self.reward_beta * oracle_below_tau)
        if self.reward_objective == "max_regret":
            reward_core = float(gap)
        elif self.reward_objective == "weighted_max_regret":
            reward_core = float(weighted_gap)
        else:
            reward_core = float(oracle_floor_weighted_gap - oracle_floor_penalty)
        reward = float(reward_core - float(self.smooth_penalty_scale) * float(smooth_pen))

        done = bool(int(self._sim.video_chunk_counter) >= int(self._max_chunks))

        throughput_norm, delay_norm = compute_normalized_throughput_delay(
            video_chunk_size_bytes=float(chunk_size_bytes), delay_ms=float(sim_res.delay_ms)
        )

        info: dict[str, Any] = {
            "adv/action_raw": float(action_raw),
            "adv/action_clipped": float(action_clipped),
            "adv/bw_mbps": float(bw),
            "adv/bw_mbps_clipped": float(bw_clipped),
            "adv/p_smoothing": float(smooth_pen),
            "protocol/bitrate_idx": int(bit_rate),
            "protocol/bitrate_mbps": float(pensieve_env.VIDEO_BIT_RATE[int(bit_rate)] / pensieve_env.M_IN_K),
            "protocol/buffer_s": float(sim_res.buffer_after_s),
            "protocol/rebuf_s": float(sim_res.rebuf_s),
            "protocol/delay_ms": float(sim_res.delay_ms),
            "protocol/throughput_norm": float(throughput_norm),
            "protocol/delay_norm": float(delay_norm),
            "oracle/r_opt": float(r_opt),
            "oracle/r_protocol": float(r_protocol),
            "oracle/gap": float(gap),
            "oracle/gap_weighted": float(weighted_gap),
            "oracle/r_opt_above_tau": float(oracle_above_tau),
            "oracle/r_opt_below_tau": float(oracle_below_tau),
            "oracle/gap_oracle_floor_weighted": float(oracle_floor_weighted_gap),
            "oracle/oracle_floor_penalty": float(oracle_floor_penalty),
            "env/reward_core": float(reward_core),
            "env/reward": float(reward),
            "env/chunk_idx": int(chunk_idx),
        }
        if self._shield is not None and self._shield_stats_last:
            for k, v in self._shield_stats_last.items():
                if isinstance(v, (int, float)):
                    info[f"protocol/{k}"] = v

        if done:
            full_qoe = _window_qoe(bitrate_idxs=[int(x) for x in self._bitrate_hist], rebuf_s=[float(x) for x in self._rebuf_s_hist])
            info |= {
                "episode/qoe_protocol": float(full_qoe),
                "episode/rebuf_s_total": float(np.sum(np.asarray(self._rebuf_s_hist, dtype=np.float64))),
                "episode/segments": list(self._segments),
                "episode/bw_values_mbps": list(self._bw_hist),
                "episode/bitrate_idxs": list(self._bitrate_hist),
            }

        return obs, reward, done, False, info
