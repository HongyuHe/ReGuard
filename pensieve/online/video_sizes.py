from __future__ import annotations

import numpy as np

from .pensieve_bridge import import_pensieve


def load_video_size_table() -> np.ndarray:
    modules = import_pensieve()
    fixed_env = modules["fixed_env"]

    video_sizes = np.zeros((int(fixed_env.BITRATE_LEVELS), int(fixed_env.TOTAL_VIDEO_CHUNCK)), dtype=np.int64)
    for bitrate in range(int(fixed_env.BITRATE_LEVELS)):
        with open(f"{fixed_env.VIDEO_SIZE_FILE}{bitrate}") as f:
            for chunk_idx, line in enumerate(f):
                if chunk_idx >= int(fixed_env.TOTAL_VIDEO_CHUNCK):
                    break
                video_sizes[bitrate, chunk_idx] = int(line.split()[0])
    return video_sizes

