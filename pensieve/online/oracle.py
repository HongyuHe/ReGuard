from __future__ import annotations

import numpy as np

from .pensieve_bridge import import_pensieve
from .video_sizes import load_video_size_table


class ExactWindowOracle:
    def __init__(self, *, window_size: int = 4) -> None:
        if int(window_size) <= 0:
            raise ValueError("* window_size must be positive")

        modules = import_pensieve()
        self._env = modules["env"]
        self._fixed = modules["fixed_env"]

        self.window_size = int(window_size)
        self.video_sizes = load_video_size_table()

        self._seqs = self._build_sequences(self.window_size)  # (n_seq, window)
        bitrate_mbps = (np.asarray(self._env.VIDEO_BIT_RATE, dtype=np.float64) / float(self._env.M_IN_K)).astype(
            np.float64
        )
        self._seq_bitrates_mbps = bitrate_mbps[self._seqs]  # (n_seq, window)

        self._quality_sum_by_len: dict[int, np.ndarray] = {}
        self._smooth_sum_by_len: dict[int, np.ndarray] = {}
        for length in range(1, self.window_size + 1):
            q = self._seq_bitrates_mbps[:, :length]
            self._quality_sum_by_len[length] = q.sum(axis=1)
            if length <= 1:
                self._smooth_sum_by_len[length] = np.zeros((q.shape[0],), dtype=np.float64)
            else:
                self._smooth_sum_by_len[length] = np.abs(q[:, 1:] - q[:, :-1]).sum(axis=1)

        self._bytes_per_sec_per_mbps = float(self._fixed.B_IN_MB) / float(self._fixed.BITS_IN_BYTE)

    @staticmethod
    def _build_sequences(window: int) -> np.ndarray:
        grids = np.indices((6,) * int(window), dtype=np.int64)
        return np.reshape(np.moveaxis(grids, 0, -1), (-1, int(window))).astype(np.int64, copy=False)

    def compute(
        self,
        *,
        start_buffer_ms: float,
        chunk_indices: list[int],
        bw_mbps: list[float],
    ) -> float:
        length = int(len(chunk_indices))
        if length <= 0:
            return 0.0
        if length != int(len(bw_mbps)):
            raise ValueError("* chunk_indices and bw_mbps must have the same length")
        if length > int(self.window_size):
            raise ValueError("* length exceeds window_size")

        buffer_ms = np.full((self._seqs.shape[0],), float(start_buffer_ms), dtype=np.float64)
        rebuf_ms_sum = np.zeros_like(buffer_ms)

        seqs_prefix = self._seqs[:, :length]

        for i in range(length):
            bw = float(bw_mbps[i])
            if bw < 0.0 or not np.isfinite(bw):
                raise ValueError("* bw_mbps values must be finite and >= 0")
            #* Avoid division-by-zero when callers allow bw_min == 0.
            bw = max(float(bw), 1e-6)

            chunk_idx = int(chunk_indices[i])
            if chunk_idx < 0 or chunk_idx >= int(self._fixed.TOTAL_VIDEO_CHUNCK):
                raise ValueError("* chunk index out of range")

            quality_idx = seqs_prefix[:, i]
            chunk_size_bytes = self.video_sizes[quality_idx, chunk_idx].astype(np.float64, copy=False)

            download_time_s = chunk_size_bytes / (bw * self._bytes_per_sec_per_mbps * float(self._fixed.PACKET_PAYLOAD_PORTION))
            delay_ms = download_time_s * float(self._fixed.MILLISECONDS_IN_SECOND) + float(self._fixed.LINK_RTT)

            rebuf_ms = np.maximum(delay_ms - buffer_ms, 0.0)
            rebuf_ms_sum += rebuf_ms

            buffer_after_delay = np.maximum(buffer_ms - delay_ms, 0.0)
            buffer_after_add = buffer_after_delay + float(self._fixed.VIDEO_CHUNCK_LEN)

            drain_ms = np.maximum(buffer_after_add - float(self._fixed.BUFFER_THRESH), 0.0)
            sleep_steps = np.ceil(drain_ms / float(self._fixed.DRAIN_BUFFER_SLEEP_TIME))
            sleep_time_ms = sleep_steps * float(self._fixed.DRAIN_BUFFER_SLEEP_TIME)

            buffer_ms = buffer_after_add - sleep_time_ms

        quality_sum = self._quality_sum_by_len[length]
        smooth_sum = self._smooth_sum_by_len[length]
        qoe = quality_sum - float(self._env.REBUF_PENALTY) * (rebuf_ms_sum / float(self._fixed.MILLISECONDS_IN_SECOND)) - smooth_sum
        return float(np.max(qoe))
