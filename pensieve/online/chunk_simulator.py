from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .pensieve_bridge import import_pensieve
from .video_sizes import load_video_size_table


@dataclass(frozen=True)
class ChunkSimResult:
    delay_ms: float
    download_time_s: float
    sleep_time_ms: float
    buffer_after_s: float
    rebuf_s: float


class ChunkLevelPensieveSimulator:
    def __init__(self, *, seed: int = 42) -> None:
        modules = import_pensieve()
        self._env = modules["env"]
        self._fixed = modules["fixed_env"]
        self._rng = np.random.RandomState(int(seed))
        self.video_sizes = load_video_size_table()
        self.reset()

    def reset(self) -> None:
        self.video_chunk_counter = 0
        self.buffer_ms = 0.0

    def get_next_chunk_sizes(self) -> list[int]:
        if int(self.video_chunk_counter) >= int(self._fixed.TOTAL_VIDEO_CHUNCK):
            return [0 for _ in range(int(self._fixed.BITRATE_LEVELS))]
        return [int(self.video_sizes[i, int(self.video_chunk_counter)]) for i in range(int(self._fixed.BITRATE_LEVELS))]

    def simulate_chunk(self, *, bw_mbps: float, quality: int) -> ChunkSimResult:
        if int(self.video_chunk_counter) >= int(self._fixed.TOTAL_VIDEO_CHUNCK):
            raise RuntimeError("* simulate_chunk called after end_of_video")

        bw = float(bw_mbps)
        if bw < 0.0 or not np.isfinite(bw):
            raise ValueError("* bw_mbps must be finite and >= 0")
        #* Avoid division-by-zero when callers allow bw_min == 0.
        bw = max(float(bw), 1e-6)

        chunk_size_bytes = float(self.video_sizes[int(quality), int(self.video_chunk_counter)])

        bytes_per_sec_per_mbps = float(self._fixed.B_IN_MB) / float(self._fixed.BITS_IN_BYTE)
        download_time_s = chunk_size_bytes / (bw * bytes_per_sec_per_mbps * float(self._fixed.PACKET_PAYLOAD_PORTION))
        delay_ms = download_time_s * float(self._fixed.MILLISECONDS_IN_SECOND) + float(self._fixed.LINK_RTT)

        rebuf_ms = max(float(delay_ms) - float(self.buffer_ms), 0.0)
        buffer_after_delay = max(float(self.buffer_ms) - float(delay_ms), 0.0)
        buffer_after_add = buffer_after_delay + float(self._fixed.VIDEO_CHUNCK_LEN)

        sleep_time_ms = 0.0
        if buffer_after_add > float(self._fixed.BUFFER_THRESH):
            drain_ms = float(buffer_after_add) - float(self._fixed.BUFFER_THRESH)
            sleep_steps = math.ceil(drain_ms / float(self._fixed.DRAIN_BUFFER_SLEEP_TIME))
            sleep_time_ms = float(sleep_steps) * float(self._fixed.DRAIN_BUFFER_SLEEP_TIME)
            buffer_after_add -= sleep_time_ms

        self.buffer_ms = float(buffer_after_add)
        self.video_chunk_counter += 1

        return ChunkSimResult(
            delay_ms=float(delay_ms),
            download_time_s=float(download_time_s),
            sleep_time_ms=float(sleep_time_ms),
            buffer_after_s=float(self.buffer_ms) / float(self._fixed.MILLISECONDS_IN_SECOND),
            rebuf_s=float(rebuf_ms) / float(self._fixed.MILLISECONDS_IN_SECOND),
        )


def compute_normalized_throughput_delay(*, video_chunk_size_bytes: float, delay_ms: float) -> tuple[float, float]:
    modules = import_pensieve()
    pensieve_env = modules["env"]
    throughput_norm = float(video_chunk_size_bytes) / float(delay_ms) / float(pensieve_env.M_IN_K)
    delay_norm = float(delay_ms) / float(pensieve_env.M_IN_K) / float(pensieve_env.BUFFER_NORM_FACTOR)
    return float(throughput_norm), float(delay_norm)
