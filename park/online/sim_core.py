from __future__ import annotations

import csv
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Job:
    size: int
    arrival_time: int
    server: "Server | None" = None
    start_time: int = -1
    finish_time: int = -1


@dataclass(frozen=True)
class ReferencePolicyResult:
    reward: float
    actions: tuple[int, ...]
    horizon: int
    policy_name: str


@dataclass(frozen=True)
class ServerDispatchStats:
    server_id: int
    service_rate: float
    queue_jobs: int
    active_jobs: int
    queue_size: int
    current_remaining_time: int
    backlog_time: int
    incoming_duration: int
    completion_time: int


class Server:
    def __init__(self, server_id: int, service_rate: float, curr_time_ref: list[int]) -> None:
        self.server_id = int(server_id)
        self.service_rate = float(service_rate)
        self.curr_time_ref = curr_time_ref
        self.queue: deque[Job] = deque()
        self.curr_job: Job | None = None

    def reset(self) -> None:
        self.queue.clear()
        self.curr_job = None

    def clone(self, curr_time_ref: list[int], job_map: dict[int, Job]) -> "Server":
        cloned = Server(self.server_id, self.service_rate, curr_time_ref)
        for job in self.queue:
            cloned_job = _clone_job(job, job_map)
            cloned_job.server = cloned
            cloned.queue.append(cloned_job)
        if self.curr_job is not None:
            cloned.curr_job = _clone_job(self.curr_job, job_map)
            cloned.curr_job.server = cloned
        return cloned

    @property
    def curr_time(self) -> int:
        return int(self.curr_time_ref[0])

    def schedule(self, job: Job) -> None:
        self.queue.append(job)
        job.server = self

    def process(self) -> Job | None:
        if self.curr_job is None and len(self.queue) > 0:
            job = self.queue.popleft()
            duration = _service_duration(int(job.size), float(self.service_rate))
            job.start_time = int(self.curr_time)
            job.finish_time = int(self.curr_time + duration)
            self.curr_job = job
            return job
        return None


REFERENCE_POLICY_NAMES: tuple[str, ...] = (
    "least_completion_time",
    "least_backlog_time",
    "least_active_jobs",
    "least_queue_jobs",
    "least_queue_size",
    "fastest_server",
)


def _service_duration(job_size: int, service_rate: float) -> int:
    return max(1, int(int(job_size) / float(service_rate)))


def _server_remaining_time(server: Server, *, curr_time: int) -> int:
    remaining = 0
    if server.curr_job is not None:
        remaining += max(0, int(server.curr_job.finish_time) - int(curr_time))
    remaining += sum(_service_duration(int(job.size), float(server.service_rate)) for job in list(server.queue))
    return int(remaining)


def _collect_server_dispatch_stats(system: "LoadBalanceSystem") -> list[ServerDispatchStats]:
    if system.incoming_job is None:
        raise RuntimeError("dispatch stats requested without an incoming job")

    incoming_size = int(system.incoming_job.size)
    curr_time = int(system.curr_time)
    stats: list[ServerDispatchStats] = []
    for server in list(system.servers):
        current_remaining_time = (
            max(0, int(server.curr_job.finish_time) - int(curr_time)) if server.curr_job is not None else 0
        )
        queue_jobs = int(len(server.queue))
        queue_size = int(sum(int(job.size) for job in list(server.queue)))
        backlog_time = _server_remaining_time(server, curr_time=curr_time)
        incoming_duration = _service_duration(incoming_size, float(server.service_rate))
        stats.append(
            ServerDispatchStats(
                server_id=int(server.server_id),
                service_rate=float(server.service_rate),
                queue_jobs=int(queue_jobs),
                active_jobs=int(queue_jobs + int(server.curr_job is not None)),
                queue_size=int(queue_size),
                current_remaining_time=int(current_remaining_time),
                backlog_time=int(backlog_time),
                incoming_duration=int(incoming_duration),
                completion_time=int(backlog_time + incoming_duration),
            )
        )
    return stats


def reference_policy_names() -> tuple[str, ...]:
    return tuple(REFERENCE_POLICY_NAMES)


def select_reference_action(system: "LoadBalanceSystem", policy_name: str) -> int:
    stats = _collect_server_dispatch_stats(system)
    policy = str(policy_name)
    if policy == "least_completion_time":
        return int(
            min(
                stats,
                key=lambda s: (
                    int(s.completion_time),
                    int(s.backlog_time),
                    -float(s.service_rate),
                    int(s.server_id),
                ),
            ).server_id
        )
    if policy == "least_backlog_time":
        return int(
            min(
                stats,
                key=lambda s: (
                    int(s.backlog_time),
                    int(s.completion_time),
                    -float(s.service_rate),
                    int(s.server_id),
                ),
            ).server_id
        )
    if policy == "least_active_jobs":
        return int(
            min(
                stats,
                key=lambda s: (
                    int(s.active_jobs),
                    int(s.completion_time),
                    int(s.backlog_time),
                    int(s.server_id),
                ),
            ).server_id
        )
    if policy == "least_queue_jobs":
        return int(
            min(
                stats,
                key=lambda s: (
                    int(s.queue_jobs),
                    int(s.active_jobs),
                    int(s.completion_time),
                    int(s.server_id),
                ),
            ).server_id
        )
    if policy == "least_queue_size":
        return int(
            min(
                stats,
                key=lambda s: (
                    int(s.queue_size),
                    int(s.completion_time),
                    int(s.active_jobs),
                    int(s.server_id),
                ),
            ).server_id
        )
    if policy == "fastest_server":
        return int(
            min(
                stats,
                key=lambda s: (
                    -float(s.service_rate),
                    int(s.completion_time),
                    int(s.backlog_time),
                    int(s.server_id),
                ),
            ).server_id
        )
    raise ValueError(f"unknown reference policy: {policy_name}")


def _clone_job(job: Job, job_map: dict[int, Job]) -> Job:
    key = int(id(job))
    if key not in job_map:
        job_map[key] = Job(
            size=int(job.size),
            arrival_time=int(job.arrival_time),
            server=None,
            start_time=int(job.start_time),
            finish_time=int(job.finish_time),
        )
    return job_map[key]


class LoadBalanceSystem:
    """A controllable load-balance simulator that accepts external job arrivals."""

    def __init__(
        self,
        service_rates: list[float],
        obs_high: float = 500000.0,
        reward_time_scale: float = 1000.0,
    ) -> None:
        self.service_rates = [float(x) for x in service_rates]
        self.num_servers = len(self.service_rates)
        self.obs_high = float(obs_high)
        self.reward_time_scale = float(reward_time_scale)
        if self.reward_time_scale <= 0:
            raise ValueError("reward_time_scale must be positive")

        self._curr_time_ref = [0]
        self.servers = [Server(i, self.service_rates[i], self._curr_time_ref) for i in range(self.num_servers)]
        self.timeline: list[tuple[int, int, Any]] = []
        self._event_counter = 0
        self.incoming_job: Job | None = None
        self.finished_jobs = 0

    @property
    def curr_time(self) -> int:
        return int(self._curr_time_ref[0])

    def _push_event(self, t: int, obj: Any) -> None:
        import heapq

        heapq.heappush(self.timeline, (int(t), int(self._event_counter), obj))
        self._event_counter += 1

    def _pop_event(self) -> tuple[int, Any]:
        import heapq

        t, _eid, obj = heapq.heappop(self.timeline)
        return int(t), obj

    def clone(self) -> "LoadBalanceSystem":
        cloned = LoadBalanceSystem(
            service_rates=list(self.service_rates),
            obs_high=float(self.obs_high),
            reward_time_scale=float(self.reward_time_scale),
        )
        cloned._curr_time_ref[0] = int(self.curr_time)
        cloned._event_counter = int(self._event_counter)
        cloned.finished_jobs = int(self.finished_jobs)

        job_map: dict[int, Job] = {}
        cloned.servers = [server.clone(cloned._curr_time_ref, job_map) for server in self.servers]

        if self.incoming_job is not None:
            cloned.incoming_job = _clone_job(self.incoming_job, job_map)
            if self.incoming_job.server is not None:
                cloned.incoming_job.server = cloned.servers[int(self.incoming_job.server.server_id)]
        else:
            cloned.incoming_job = None

        cloned.timeline = []
        for t, eid, obj in self.timeline:
            if isinstance(obj, Job):
                cloned_obj = _clone_job(obj, job_map)
                if obj.server is not None:
                    cloned_obj.server = cloned.servers[int(obj.server.server_id)]
            else:
                cloned_obj = obj
            cloned.timeline.append((int(t), int(eid), cloned_obj))
        return cloned

    def reset(self, *, first_job_size: int, first_arrival_time: int = 0) -> np.ndarray:
        self._curr_time_ref[0] = int(first_arrival_time)
        self.timeline = []
        self._event_counter = 0
        for s in self.servers:
            s.reset()
        self.incoming_job = Job(size=int(first_job_size), arrival_time=int(first_arrival_time))
        self.finished_jobs = 0
        return self.observe()

    def observe(self) -> np.ndarray:
        obs_arr: list[float] = []
        for s in self.servers:
            load = float(sum(j.size for j in s.queue))
            if s.curr_job is not None:
                load += float(max(0, s.curr_job.finish_time - self.curr_time))
            obs_arr.append(min(load, self.obs_high))
        if self.incoming_job is None:
            obs_arr.append(0.0)
        else:
            obs_arr.append(min(float(self.incoming_job.size), self.obs_high))
        return np.asarray(obs_arr, dtype=np.float32)

    def dispatch(
        self,
        action: int,
        *,
        next_arrival_dt: int | None,
        next_job_size: int | None,
    ) -> tuple[np.ndarray, float, bool, dict[str, float | int]]:
        if self.incoming_job is None:
            raise RuntimeError("dispatch called without incoming job")
        act = int(action)
        if act < 0 or act >= self.num_servers:
            raise ValueError(f"invalid action {action}")

        self.servers[act].schedule(self.incoming_job)
        running = self.servers[act].process()
        if running is not None:
            self._push_event(running.finish_time, running)
        self.incoming_job = None

        if next_arrival_dt is not None:
            if next_job_size is None:
                raise ValueError("next_job_size required when next_arrival_dt is set")
            self._push_event(self.curr_time + int(next_arrival_dt), int(next_job_size))

        reward = 0.0

        while len(self.timeline) > 0:
            new_time, obj = self._pop_event()

            num_active_jobs = 0
            for s in self.servers:
                num_active_jobs += len(s.queue)
                if s.curr_job is not None and s.curr_job.finish_time >= self.curr_time:
                    num_active_jobs += 1
            elapsed_time = float(new_time - self.curr_time) / float(self.reward_time_scale)
            reward -= float(elapsed_time * num_active_jobs)
            self._curr_time_ref[0] = int(new_time)

            if isinstance(obj, int):
                self.incoming_job = Job(size=int(obj), arrival_time=self.curr_time)
                break

            if isinstance(obj, Job):
                self.finished_jobs += 1
                job = obj
                if job.server is not None and job.server.curr_job == job:
                    job.server.curr_job = None
                    nxt = job.server.process()
                    if nxt is not None:
                        self._push_event(nxt.finish_time, nxt)
                continue

            raise RuntimeError("unknown event type")

        done = (len(self.timeline) == 0) and (self.incoming_job is None)
        obs = self.observe()
        info = {
            "curr_time": int(self.curr_time),
            "num_finished_jobs": int(self.finished_jobs),
        }
        return obs, float(reward), bool(done), info


def rollout_reference_policy_reward(
    *,
    snapshot: LoadBalanceSystem,
    future_arrivals: list[tuple[int | None, int | None]],
    policy_name: str,
) -> ReferencePolicyResult:
    horizon = int(len(future_arrivals))
    if horizon <= 0:
        return ReferencePolicyResult(
            reward=0.0,
            actions=tuple(),
            horizon=0,
            policy_name=str(policy_name),
        )

    system = snapshot.clone()
    total_reward = 0.0
    actions: list[int] = []
    realized_horizon = 0
    for depth, (next_arrival_dt, next_job_size) in enumerate(list(future_arrivals)):
        action = int(select_reference_action(system, str(policy_name)))
        _obs, reward, done, _info = system.dispatch(
            int(action),
            next_arrival_dt=next_arrival_dt,
            next_job_size=next_job_size,
        )
        total_reward += float(reward)
        actions.append(int(action))
        realized_horizon = int(depth) + 1
        if bool(done):
            break
    return ReferencePolicyResult(
        reward=float(total_reward),
        actions=tuple(actions),
        horizon=int(realized_horizon),
        policy_name=str(policy_name),
    )


def best_of_heuristics_reference_reward(
    *,
    snapshot: LoadBalanceSystem,
    future_arrivals: list[tuple[int | None, int | None]],
    policy_names: tuple[str, ...] | list[str] | None = None,
) -> ReferencePolicyResult:
    names = tuple(str(x) for x in (REFERENCE_POLICY_NAMES if policy_names is None else policy_names))
    if len(names) == 0:
        raise ValueError("policy_names must be non-empty")

    best_result: ReferencePolicyResult | None = None
    for name in names:
        result = rollout_reference_policy_reward(
            snapshot=snapshot,
            future_arrivals=future_arrivals,
            policy_name=str(name),
        )
        if best_result is None or float(result.reward) > float(best_result.reward):
            best_result = result
            continue
        if (
            float(result.reward) == float(best_result.reward)
            and tuple(result.actions) < tuple(best_result.actions)
        ):
            best_result = result
    if best_result is None:
        raise RuntimeError("failed to evaluate reference policy pool")
    return best_result


def exact_k_step_oracle_reward(
    *,
    snapshot: LoadBalanceSystem,
    future_arrivals: list[tuple[int | None, int | None]],
    num_servers: int | None = None,
) -> ReferencePolicyResult:
    del num_servers
    return best_of_heuristics_reference_reward(
        snapshot=snapshot,
        future_arrivals=future_arrivals,
    )


def parse_job_trace(path: str) -> list[dict[str, int]]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"trace not found: {path}")
    if p.suffix.lower() == ".json":
        obj = json.loads(p.read_text(encoding="utf-8"))
        rows = obj.get("jobs", [])
        if not isinstance(rows, list):
            raise ValueError(f"invalid jobs in {path}")
        jobs = []
        for row in rows:
            at = int(row.get("arrival_time", 0))
            js = int(row.get("job_size", row.get("size", 0)))
            if js <= 0:
                continue
            jobs.append({"arrival_time": at, "job_size": js})
        jobs.sort(key=lambda x: int(x["arrival_time"]))
        return jobs

    if p.suffix.lower() != ".csv":
        raise ValueError(f"unsupported trace suffix: {p.suffix}")

    jobs = []
    t = 0
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"missing header in csv trace: {path}")
        cols = {c.strip().lower() for c in reader.fieldnames}
        has_abs = "arrival_time" in cols
        has_dt = "dt" in cols
        if (not has_abs) and (not has_dt):
            raise ValueError(f"csv trace must contain arrival_time or dt: {path}")
        for row in reader:
            size_raw = row.get("job_size", row.get("size"))
            if size_raw is None:
                continue
            size = int(float(size_raw))
            if size <= 0:
                continue
            if has_abs:
                at = int(float(row.get("arrival_time", 0)))
            else:
                t += int(float(row.get("dt", 0)))
                at = int(t)
            jobs.append({"arrival_time": at, "job_size": int(size)})
    jobs.sort(key=lambda x: int(x["arrival_time"]))
    return jobs


def evaluate_victim_vs_reference_on_jobs(
    *,
    jobs: list[dict[str, int]],
    victim_action_fn,
    service_rates: list[float],
    obs_high: float = 500000.0,
    reward_time_scale: float = 1000.0,
    reference_window_len: int = 5,
) -> dict[str, float | int]:
    if len(jobs) == 0:
        raise ValueError("jobs must be non-empty")
    if int(reference_window_len) <= 0:
        raise ValueError("reference_window_len must be positive")

    jobs_sorted = sorted(jobs, key=lambda x: int(x["arrival_time"]))
    future_arrivals = []
    n = len(jobs_sorted)
    for i in range(int(n)):
        if int(i) < int(n) - 1:
            next_dt = int(jobs_sorted[int(i + 1)]["arrival_time"]) - int(jobs_sorted[int(i)]["arrival_time"])
            next_size = int(jobs_sorted[int(i + 1)]["job_size"])
            future_arrivals.append((max(0, int(next_dt)), int(next_size)))
        else:
            future_arrivals.append((None, None))
    first_t = int(jobs_sorted[0]["arrival_time"])

    victim_sys = LoadBalanceSystem(
        service_rates=service_rates,
        obs_high=obs_high,
        reward_time_scale=float(reward_time_scale),
    )
    reference_sys = LoadBalanceSystem(
        service_rates=service_rates,
        obs_high=obs_high,
        reward_time_scale=float(reward_time_scale),
    )

    victim_obs = victim_sys.reset(first_job_size=int(jobs_sorted[0]["job_size"]), first_arrival_time=first_t)
    reference_obs = reference_sys.reset(first_job_size=int(jobs_sorted[0]["job_size"]), first_arrival_time=first_t)

    total_victim = 0.0
    total_reference = 0.0

    for i in range(n):
        next_dt: int | None
        next_size: int | None
        if i < n - 1:
            next_dt = int(jobs_sorted[i + 1]["arrival_time"]) - int(jobs_sorted[i]["arrival_time"])
            next_dt = max(0, int(next_dt))
            next_size = int(jobs_sorted[i + 1]["job_size"])
        else:
            next_dt = None
            next_size = None

        victim_act = int(victim_action_fn(victim_obs))
        horizon = int(min(int(reference_window_len), int(n) - int(i)))
        reference_result = best_of_heuristics_reference_reward(
            snapshot=reference_sys.clone(),
            future_arrivals=future_arrivals[int(i) : int(i) + int(horizon)],
        )
        reference_act = int(reference_result.actions[0])

        victim_obs, victim_rew, _done_v, _info_v = victim_sys.dispatch(
            victim_act, next_arrival_dt=next_dt, next_job_size=next_size
        )
        reference_obs, reference_rew, _done_o, _info_o = reference_sys.dispatch(
            reference_act, next_arrival_dt=next_dt, next_job_size=next_size
        )

        total_victim += float(victim_rew)
        total_reference += float(reference_rew)

    gap = float(total_reference - total_victim)
    return {
        "num_jobs": int(n),
        "victim_total_reward": float(total_victim),
        "oracle_total_reward": float(total_reference),
        "oracle_gap": float(gap),
        "victim_reward_per_job": float(total_victim / max(1, n)),
        "oracle_reward_per_job": float(total_reference / max(1, n)),
        "oracle_gap_per_job": float(gap / max(1, n)),
        "reference_window": int(reference_window_len),
    }


def evaluate_victim_vs_oracle_on_jobs(
    *,
    jobs: list[dict[str, int]],
    victim_action_fn,
    service_rates: list[float],
    obs_high: float = 500000.0,
) -> dict[str, float | int]:
    return evaluate_victim_vs_reference_on_jobs(
        jobs=jobs,
        victim_action_fn=victim_action_fn,
        service_rates=service_rates,
        obs_high=float(obs_high),
    )
