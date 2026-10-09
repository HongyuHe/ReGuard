from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from online.sim_core import LoadBalanceSystem

LOAD_LOG_SCALE = 12.0
SIZE_LOG_SCALE = 12.0
TIME_LOG_SCALE = 12.0
QUEUE_JOBS_SCALE = 32.0


def _service_duration(job_size: int, service_rate: float) -> int:
    return max(1, int(int(job_size) / float(service_rate)))


def _scale_log(x: np.ndarray, scale: float) -> np.ndarray:
    return np.log1p(np.maximum(x, 0.0)) / float(scale)


def _unscale_log_tensor(x: torch.Tensor, scale: float) -> torch.Tensor:
    return torch.expm1(torch.clamp(x, min=0.0) * float(scale))


@dataclass(frozen=True)
class SummarySlices:
    loads: slice
    queue_jobs: slice
    running: slice
    queue_sizes: slice
    backlog_times: slice
    incoming_size: slice


def summary_slices(num_servers: int) -> SummarySlices:
    n = int(num_servers)
    return SummarySlices(
        loads=slice(0, n),
        queue_jobs=slice(n, 2 * n),
        running=slice(2 * n, 3 * n),
        queue_sizes=slice(3 * n, 4 * n),
        backlog_times=slice(4 * n, 5 * n),
        incoming_size=slice(5 * n, 5 * n + 1),
    )


def summary_feature_dim(num_servers: int) -> int:
    return int(5 * int(num_servers) + 1)


def extract_system_summary(system: LoadBalanceSystem) -> np.ndarray:
    obs = np.asarray(system.observe(), dtype=np.float64)
    num_servers = int(system.num_servers)
    if obs.shape[0] != int(num_servers + 1):
        raise ValueError("unexpected observation width for system summary")
    queue_jobs = np.zeros((num_servers,), dtype=np.float64)
    running = np.zeros((num_servers,), dtype=np.float64)
    queue_sizes = np.zeros((num_servers,), dtype=np.float64)
    backlog_times = np.zeros((num_servers,), dtype=np.float64)
    curr_time = int(system.curr_time)
    for idx, server in enumerate(list(system.servers)):
        queue_jobs[idx] = float(len(server.queue))
        running[idx] = float(int(server.curr_job is not None))
        queue_sizes[idx] = float(sum(int(job.size) for job in list(server.queue)))
        current_remaining = (
            max(0, int(server.curr_job.finish_time) - int(curr_time))
            if server.curr_job is not None
            else 0
        )
        queued_remaining = sum(
            _service_duration(int(job.size), float(server.service_rate))
            for job in list(server.queue)
        )
        backlog_times[idx] = float(int(current_remaining + queued_remaining))
    incoming_size = np.asarray([float(obs[-1])], dtype=np.float64)
    summary = np.concatenate(
        [
            _scale_log(obs[:-1], LOAD_LOG_SCALE),
            np.maximum(queue_jobs, 0.0) / float(QUEUE_JOBS_SCALE),
            np.clip(running, 0.0, 1.0),
            _scale_log(queue_sizes, SIZE_LOG_SCALE),
            _scale_log(backlog_times, TIME_LOG_SCALE),
            _scale_log(incoming_size, SIZE_LOG_SCALE),
        ],
        axis=0,
    )
    return np.asarray(summary, dtype=np.float32)


def project_summary_tensor(summary: torch.Tensor, *, num_servers: int) -> torch.Tensor:
    sl = summary_slices(int(num_servers))
    parts = []
    parts.append(F.softplus(summary[..., sl.loads]))
    parts.append(F.softplus(summary[..., sl.queue_jobs]))
    parts.append(torch.sigmoid(summary[..., sl.running]))
    parts.append(F.softplus(summary[..., sl.queue_sizes]))
    parts.append(F.softplus(summary[..., sl.backlog_times]))
    parts.append(F.softplus(summary[..., sl.incoming_size]))
    return torch.cat(parts, dim=-1)


def summary_to_victim_obs_tensor(summary: torch.Tensor, *, num_servers: int) -> torch.Tensor:
    sl = summary_slices(int(num_servers))
    load_obs = _unscale_log_tensor(summary[..., sl.loads], LOAD_LOG_SCALE)
    incoming = _unscale_log_tensor(summary[..., sl.incoming_size], SIZE_LOG_SCALE)
    return torch.cat([load_obs, incoming], dim=-1)


def backlog_total_from_summary_tensor(summary: torch.Tensor, *, num_servers: int) -> torch.Tensor:
    sl = summary_slices(int(num_servers))
    backlog = _unscale_log_tensor(summary[..., sl.backlog_times], TIME_LOG_SCALE)
    return backlog.sum(dim=-1)


def active_jobs_total_from_summary_tensor(summary: torch.Tensor, *, num_servers: int) -> torch.Tensor:
    sl = summary_slices(int(num_servers))
    queue_jobs = torch.clamp(summary[..., sl.queue_jobs], min=0.0) * float(QUEUE_JOBS_SCALE)
    running = torch.clamp(summary[..., sl.running], min=0.0, max=1.0)
    return (queue_jobs + running).sum(dim=-1)
