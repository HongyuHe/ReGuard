from __future__ import annotations

import argparse

import numpy as np
import torch

from load_balance_env import LoadBalanceTorchEnv
from train_ppo import ActorCritic


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = dict(ckpt.get("config", {}))

    env = LoadBalanceTorchEnv(
        num_servers=int(cfg.get("num_servers", 10)),
        num_stream_jobs=int(cfg.get("num_stream_jobs", 1000)),
        job_size_pareto_shape=float(cfg.get("job_size_pareto_shape", 1.5)),
        job_size_pareto_scale=float(cfg.get("job_size_pareto_scale", 100.0)),
        job_interval=float(cfg.get("job_interval", 55.0)),
        obs_high=float(cfg.get("obs_high", 500000.0)),
        reward_time_scale=float(cfg.get("reward_time_scale", 1000.0)),
        seed=int(args.seed),
        workload_trace_dir=cfg.get("workload_trace_dir", None),
        workload_trace_path=cfg.get("workload_trace_path", None),
        trace_select_mode=str(cfg.get("trace_select_mode", "random")),
    )

    model = ActorCritic(
        obs_dim=int(env.observation_space.shape[0]),
        act_dim=int(env.action_space.n),
        hidden=int(cfg.get("hidden", 128)),
    )
    model.load_state_dict(ckpt["model"])
    model.eval()

    scores = []
    for ep in range(int(args.episodes)):
        obs, _ = env.reset(seed=int(args.seed + ep))
        done = False
        total = 0.0
        last_info = {}
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits, _ = model(obs_t)
                action = int(torch.argmax(logits, dim=-1).item())
            obs, rew, term, trunc, info = env.step(action)
            total += float(rew)
            last_info = info
            done = bool(term or trunc)
        scores.append(total)
        finished = int(last_info.get("num_finished_jobs", 0))
        scores_per_job = (total / max(1, finished))
        print(f"ep={ep} total_reward={total:.3f} reward_per_job={scores_per_job:.6f} jobs={finished}")

    print(f"episodes={len(scores)} mean_reward={np.mean(scores):.3f} std={np.std(scores):.3f}")


if __name__ == "__main__":
    main()
