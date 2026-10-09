from __future__ import annotations

import csv
import heapq
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass
class Job:
    size: int
    arrival_time: int
    server: "Server | None" = None
    start_time: int = -1
    finish_time: int = -1


class WallTime:
    def __init__(self) -> None:
        self.curr_time = 0

    def reset(self) -> None:
        self.curr_time = 0

    def update(self, t: int) -> None:
        self.curr_time = int(t)


class Timeline:
    def __init__(self) -> None:
        self._heap: list[tuple[int, int, Any]] = []
        self._counter = 0

    def reset(self) -> None:
        self._heap.clear()
        self._counter = 0

    def push(self, t: int, obj: Any) -> None:
        heapq.heappush(self._heap, (int(t), int(self._counter), obj))
        self._counter += 1

    def pop(self) -> tuple[int, Any]:
        t, _, obj = heapq.heappop(self._heap)
        return int(t), obj

    def __len__(self) -> int:
        return len(self._heap)


class Server:
    def __init__(self, server_id: int, service_rate: float, wall_time: WallTime) -> None:
        self.server_id = int(server_id)
        self.service_rate = float(service_rate)
        self.wall_time = wall_time
        self.queue: deque[Job] = deque()
        self.curr_job: Job | None = None

    def schedule(self, job: Job) -> None:
        self.queue.append(job)
        job.server = self

    def process(self) -> Job | None:
        if (self.curr_job is None or self.curr_job.finish_time <= self.wall_time.curr_time) and len(self.queue) > 0:
            self.curr_job = self.queue.popleft()
            duration = max(1, int(self.curr_job.size / self.service_rate))
            self.curr_job.start_time = int(self.wall_time.curr_time)
            self.curr_job.finish_time = int(self.wall_time.curr_time + duration)
            return self.curr_job
        return None

    def reset(self) -> None:
        self.queue = deque()
        self.curr_job = None


class LoadBalanceTorchEnv(gym.Env):
    """Torch-friendly rewrite of Park's load_balance simulation.

    Observation: [server_0_load, ..., server_n_load, incoming_job_size]
    Action: discrete server index
    Reward: negative active-job-seconds since previous decision
    """

    metadata = {"render_modes": ["human", None]}

    def __init__(
        self,
        num_servers: int = 10,
        service_rates: list[float] | None = None,
        num_stream_jobs: int = 1000,
        job_size_pareto_shape: float = 1.5,
        job_size_pareto_scale: float = 100.0,
        job_interval: float = 55.0,
        obs_high: float = 500000.0,
        reward_time_scale: float = 1000.0,
        seed: int = 42,
        workload_trace_path: str | None = None,
        workload_trace_dir: str | None = None,
        trace_select_mode: str = "random",
    ) -> None:
        super().__init__()
        self.num_servers = int(num_servers)
        if service_rates is None:
            if self.num_servers == 10:
                service_rates = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05]
            else:
                service_rates = np.linspace(0.15, 1.05, num=self.num_servers, dtype=np.float64).tolist()
        if len(service_rates) != self.num_servers:
            raise ValueError("service_rates length must match num_servers")

        self.service_rates = [float(x) for x in service_rates]
        self.num_stream_jobs = int(num_stream_jobs)
        self.job_size_pareto_shape = float(job_size_pareto_shape)
        self.job_size_pareto_scale = float(job_size_pareto_scale)
        self.job_interval = float(job_interval)
        self.obs_high = float(obs_high)
        self.reward_time_scale = float(reward_time_scale)
        self.workload_trace_path = str(workload_trace_path) if workload_trace_path else None
        self.workload_trace_dir = str(workload_trace_dir) if workload_trace_dir else None
        self.trace_select_mode = str(trace_select_mode)
        if self.reward_time_scale <= 0:
            raise ValueError("reward_time_scale must be positive")
        if self.trace_select_mode not in ("random", "round_robin"):
            raise ValueError("trace_select_mode must be one of: random, round_robin")

        self.wall_time = WallTime()
        self.timeline = Timeline()
        self.servers = [Server(i, self.service_rates[i], self.wall_time) for i in range(self.num_servers)]
        self.incoming_job: Job | None = None
        self.finished_jobs: list[Job] = []
        self._trace_files = self._discover_trace_files()
        self._trace_rr_cursor = 0

        low = np.array([0.0] * (self.num_servers + 1), dtype=np.float32)
        high = np.array([self.obs_high] * (self.num_servers + 1), dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(self.num_servers)

        self.np_random = np.random.RandomState(int(seed))
        self.reset(seed=int(seed))

    def seed(self, seed: int | None = None):
        if seed is None:
            return
        self.np_random = np.random.RandomState(int(seed))

    def _generate_job(self) -> tuple[int, int]:
        size = int((self.np_random.pareto(self.job_size_pareto_shape) + 1.0) * self.job_size_pareto_scale)
        dt = int(self.np_random.exponential(self.job_interval))
        return max(0, dt), max(1, size)

    def _generate_jobs(self) -> None:
        t = 0
        for _ in range(self.num_stream_jobs):
            dt, size = self._generate_job()
            t += dt
            self.timeline.push(t, int(size))

    def _discover_trace_files(self) -> list[Path]:
        if self.workload_trace_path:
            p = Path(self.workload_trace_path)
            if not p.is_file():
                raise FileNotFoundError(f"workload_trace_path not found: {p}")
            return [p]

        if not self.workload_trace_dir:
            return []

        d = Path(self.workload_trace_dir)
        if not d.is_dir():
            raise FileNotFoundError(f"workload_trace_dir not found: {d}")

        candidates = sorted([*d.glob("*.csv"), *d.glob("*.json")])
        candidates = [p for p in candidates if p.name.lower() != "index.csv"]
        valid: list[Path] = []
        for p in candidates:
            try:
                self._parse_trace_file(p)
                valid.append(p)
            except Exception:
                continue
        if len(valid) == 0:
            raise RuntimeError(f"no valid workload traces found in: {d}")
        return valid

    def _select_trace_file(self) -> Path:
        if len(self._trace_files) == 0:
            raise RuntimeError("trace mode enabled but no trace files available")
        if self.workload_trace_path:
            return self._trace_files[0]
        if self.trace_select_mode == "round_robin":
            p = self._trace_files[self._trace_rr_cursor % len(self._trace_files)]
            self._trace_rr_cursor += 1
            return p
        i = int(self.np_random.randint(0, len(self._trace_files)))
        return self._trace_files[i]

    def _parse_trace_file(self, path: Path) -> list[tuple[int, int]]:
        if path.suffix.lower() == ".json":
            obj = json.loads(path.read_text(encoding="utf-8"))
            jobs_raw = obj.get("jobs")
            if not isinstance(jobs_raw, list) or len(jobs_raw) == 0:
                raise ValueError(f"invalid jobs list in trace: {path}")
            out: list[tuple[int, int]] = []
            for j in jobs_raw:
                if not isinstance(j, dict):
                    raise ValueError(f"invalid job entry in trace: {path}")
                t = int(j.get("arrival_time", 0))
                size = int(j.get("job_size", j.get("size", 0)))
                if size <= 0:
                    raise ValueError(f"invalid job_size in trace: {path}")
                out.append((max(0, t), size))
            out.sort(key=lambda x: x[0])
            return out

        if path.suffix.lower() != ".csv":
            raise ValueError(f"unsupported trace suffix: {path}")

        out: list[tuple[int, int]] = []
        t = 0
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"missing header in trace csv: {path}")
            cols = {c.strip().lower() for c in reader.fieldnames}
            has_abs = "arrival_time" in cols
            has_dt = "dt" in cols
            has_size = ("job_size" in cols) or ("size" in cols)
            if not has_size or (not has_abs and not has_dt):
                raise ValueError(f"csv must contain arrival_time or dt and job_size/size: {path}")

            for row in reader:
                if row is None:
                    continue
                size_raw = row.get("job_size", row.get("size"))
                if size_raw is None:
                    continue
                size = int(float(size_raw))
                if size <= 0:
                    continue
                if has_abs:
                    arr_raw = row.get("arrival_time")
                    if arr_raw is None:
                        continue
                    arr_t = int(float(arr_raw))
                    out.append((max(0, arr_t), size))
                else:
                    dt_raw = row.get("dt")
                    if dt_raw is None:
                        continue
                    t += int(float(dt_raw))
                    out.append((max(0, t), size))
        if len(out) == 0:
            raise ValueError(f"no valid jobs in trace csv: {path}")
        out.sort(key=lambda x: x[0])
        return out

    def _load_trace_jobs(self, path: Path) -> None:
        jobs = self._parse_trace_file(path)
        for t, size in jobs:
            self.timeline.push(int(t), int(size))

    def _initialize(self) -> None:
        if len(self._trace_files) > 0:
            trace_path = self._select_trace_file()
            self._load_trace_jobs(trace_path)
        else:
            self._generate_jobs()
        if len(self.timeline) <= 0:
            raise RuntimeError("timeline is empty at initialization")
        new_time, obj = self.timeline.pop()
        self.wall_time.update(new_time)
        self.incoming_job = Job(size=int(obj), arrival_time=int(self.wall_time.curr_time))

    def _observe(self) -> np.ndarray:
        obs_arr: list[float] = []
        for server in self.servers:
            load = float(sum(j.size for j in server.queue))
            if server.curr_job is not None:
                load += float(max(0, server.curr_job.finish_time - self.wall_time.curr_time))
            load = min(load, self.obs_high)
            obs_arr.append(load)

        if self.incoming_job is None:
            obs_arr.append(0.0)
        else:
            obs_arr.append(min(float(self.incoming_job.size), self.obs_high))

        obs = np.asarray(obs_arr, dtype=np.float32)
        return obs

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        del options
        if seed is not None:
            self.seed(int(seed))

        for server in self.servers:
            server.reset()
        self.wall_time.reset()
        self.timeline.reset()
        self.incoming_job = None
        self.finished_jobs = []

        self._initialize()
        obs = self._observe()
        return obs, {}

    def step(self, action: int):
        act = int(action)
        if not self.action_space.contains(act):
            raise ValueError(f"invalid action {action}")
        if self.incoming_job is None:
            raise RuntimeError("step() called with no incoming job")

        self.servers[act].schedule(self.incoming_job)
        running_job = self.servers[act].process()
        if running_job is not None:
            self.timeline.push(running_job.finish_time, running_job)

        self.incoming_job = None
        reward = 0.0

        while len(self.timeline) > 0:
            new_time, obj = self.timeline.pop()

            num_active_jobs = 0
            for s in self.servers:
                num_active_jobs += len(s.queue)
                if s.curr_job is not None and s.curr_job.finish_time >= self.wall_time.curr_time:
                    num_active_jobs += 1

            elapsed_time = float(new_time - self.wall_time.curr_time) / float(self.reward_time_scale)
            reward -= float(elapsed_time * num_active_jobs)
            self.wall_time.update(new_time)

            if isinstance(obj, int):
                self.incoming_job = Job(size=int(obj), arrival_time=int(self.wall_time.curr_time))
                break

            if isinstance(obj, Job):
                job = obj
                self.finished_jobs.append(job)
                if job.server is not None and job.server.curr_job == job:
                    job.server.curr_job = None
                if job.server is not None:
                    nxt = job.server.process()
                    if nxt is not None:
                        self.timeline.push(nxt.finish_time, nxt)
                continue

            raise RuntimeError("illegal event type")

        done = (len(self.timeline) == 0) and (self.incoming_job is None)
        obs = self._observe()
        info = {"curr_time": int(self.wall_time.curr_time), "num_finished_jobs": len(self.finished_jobs)}
        return obs, float(reward), bool(done), False, info

    def render(self):
        pass

    def close(self):
        pass
