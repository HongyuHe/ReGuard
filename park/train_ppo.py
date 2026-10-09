from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from load_balance_env import LoadBalanceTorchEnv


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.pi = nn.Linear(hidden, act_dim)
        self.v = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        z = self.body(obs)
        logits = self.pi(z)
        value = self.v(z).squeeze(-1)
        return logits, value


@dataclass
class RolloutBatch:
    obs: torch.Tensor
    act: torch.Tensor
    logp: torch.Tensor
    ret: torch.Tensor
    adv: torch.Tensor
    rew: torch.Tensor
    done: torch.Tensor


def discounted_cumsum(x: np.ndarray, gamma: float) -> np.ndarray:
    y = np.zeros_like(x, dtype=np.float64)
    run = 0.0
    for i in reversed(range(len(x))):
        run = x[i] + gamma * run
        y[i] = run
    return y


def collect_rollout(env: LoadBalanceTorchEnv, model: ActorCritic, device: torch.device, steps: int, gamma: float, lam: float) -> RolloutBatch:
    obs_buf = []
    act_buf = []
    logp_buf = []
    rew_buf = []
    val_buf = []
    done_buf = []

    obs, _ = env.reset()
    for _ in range(steps):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, value = model(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            act = dist.sample()
            logp = dist.log_prob(act)

        next_obs, rew, term, trunc, _info = env.step(int(act.item()))
        done = bool(term or trunc)

        obs_buf.append(obs.copy())
        act_buf.append(int(act.item()))
        logp_buf.append(float(logp.item()))
        rew_buf.append(float(rew))
        val_buf.append(float(value.item()))
        done_buf.append(done)

        obs = next_obs
        if done:
            obs, _ = env.reset()

    #* Bootstrap the terminal value for GAE.
    with torch.no_grad():
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        _logits, last_v = model(obs_t)
        last_v = float(last_v.item())

    rews = np.asarray(rew_buf, dtype=np.float64)
    vals = np.asarray(val_buf + [last_v], dtype=np.float64)
    dones = np.asarray(done_buf, dtype=np.float64)

    deltas = rews + gamma * vals[1:] * (1.0 - dones) - vals[:-1]
    adv = discounted_cumsum(deltas, gamma * lam)
    ret = adv + vals[:-1]

    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    return RolloutBatch(
        obs=torch.as_tensor(np.asarray(obs_buf), dtype=torch.float32, device=device),
        act=torch.as_tensor(np.asarray(act_buf), dtype=torch.int64, device=device),
        logp=torch.as_tensor(np.asarray(logp_buf), dtype=torch.float32, device=device),
        ret=torch.as_tensor(ret, dtype=torch.float32, device=device),
        adv=torch.as_tensor(adv, dtype=torch.float32, device=device),
        rew=torch.as_tensor(rews, dtype=torch.float32, device=device),
        done=torch.as_tensor(dones, dtype=torch.float32, device=device),
    )


def evaluate(env_cfg: dict, model: ActorCritic, device: torch.device, episodes: int = 5) -> tuple[float, float]:
    scores = []
    per_job_scores = []
    for _ in range(episodes):
        env = LoadBalanceTorchEnv(**env_cfg)
        obs, _ = env.reset()
        done = False
        total = 0.0
        last_info = {}
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits, _v = model(obs_t)
                act = int(torch.argmax(logits, dim=-1).item())
            obs, rew, term, trunc, info = env.step(act)
            total += float(rew)
            last_info = info
            done = bool(term or trunc)
        num_jobs = int(last_info.get("num_finished_jobs", 0))
        scores.append(total)
        per_job_scores.append(total / max(1, num_jobs))
        env.close()
    return float(np.mean(scores)), float(np.mean(per_job_scores))


def _parse_wandb_tags(text: str | None) -> list[str]:
    if text is None or str(text).strip() == "":
        return []
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _mean(values: list[float]) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.mean(values))


def _checkpoint_payload(
    *,
    args: argparse.Namespace,
    model: ActorCritic,
    opt: optim.Adam,
    iter_idx: int,
    best_score_per_job: float,
    eval_mean_reward: float,
    eval_mean_reward_per_job: float,
) -> dict[str, Any]:
    model_state = model.state_dict()
    optimizer_state = opt.state_dict()
    return {
        "checkpoint_format_version": 2,
        "model": model_state,
        "model_state_dict": model_state,
        "optimizer": optimizer_state,
        "optimizer_state_dict": optimizer_state,
        "entropy_weight": float(args.ent_coef),
        "entropy_state": {
            "type": "static_ent_coef",
            "ent_coef": float(args.ent_coef),
            "adaptive": False,
        },
        "config": vars(args),
        "eval_mean_reward": float(eval_mean_reward),
        "eval_mean_reward_per_job": float(eval_mean_reward_per_job),
        "iter": int(iter_idx),
        "training_state": {
            "iter": int(iter_idx),
            "best_score_per_job_before_update": float(best_score_per_job),
            "lr": float(args.lr),
            "gamma": float(args.gamma),
            "gae_lambda": float(args.gae_lambda),
            "clip": float(args.clip),
            "vf_coef": float(args.vf_coef),
            "ent_coef": float(args.ent_coef),
            "entropy_weight": float(args.ent_coef),
            "reward_time_scale": float(args.reward_time_scale),
            "normalize_value_loss": int(args.normalize_value_loss),
        },
    }


def _load_resume_checkpoint(path: str) -> dict[str, Any]:
    return torch.load(str(path), map_location="cpu")


def _optimizer_to_device(opt: optim.Adam, device: torch.device) -> None:
    for state in opt.state.values():
        if not isinstance(state, dict):
            continue
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device=device)


def _init_wandb(args: argparse.Namespace, config: dict[str, Any]):
    if str(args.wandb_mode) == "disabled":
        return None
    try:
        import wandb
    except Exception as e:
        raise RuntimeError("Missing dependency: install wandb or run with --wandb-mode disabled") from e

    run = wandb.init(
        project=str(args.wandb_project),
        entity=args.wandb_entity,
        name=args.wandb_name,
        mode=str(args.wandb_mode),
        tags=_parse_wandb_tags(args.wandb_tags),
        group=args.wandb_group,
        dir=args.wandb_dir,
        config=config,
    )
    wandb.define_metric("train/iter")
    wandb.define_metric("train/*", step_metric="train/iter")
    wandb.define_metric("eval/*", step_metric="train/iter")
    wandb.define_metric("checkpoint/*", step_metric="train/iter")
    return run


def main() -> None:
    p = argparse.ArgumentParser(description="Torch PPO for Park load-balance task")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-servers", type=int, default=10)
    p.add_argument("--num-stream-jobs", type=int, default=1000)
    p.add_argument("--job-size-pareto-shape", type=float, default=1.5)
    p.add_argument("--job-size-pareto-scale", type=float, default=100.0)
    p.add_argument("--job-interval", type=float, default=55.0)
    p.add_argument("--obs-high", type=float, default=500000.0)
    p.add_argument("--reward-time-scale", type=float, default=1000.0)
    p.add_argument("--workload-trace-dir", type=str, default=None)
    p.add_argument("--workload-trace-path", type=str, default=None)
    p.add_argument("--trace-select-mode", type=str, default="random", choices=["random", "round_robin"])

    p.add_argument("--total-iters", type=int, default=200)
    p.add_argument("--rollout-steps", type=int, default=4096)
    p.add_argument("--ppo-epochs", type=int, default=10)
    p.add_argument("--mini-batch", type=int, default=512)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--normalize-value-loss", type=int, default=1, choices=[0, 1])
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--save-dir", type=str, default="runs/ppo_lb_torch")
    p.add_argument("--resume-from", type=str, default=None)
    p.add_argument("--resume-optimizer", type=int, default=1, choices=[0, 1])
    p.add_argument("--resume-entropy-weight", type=int, default=1, choices=[0, 1])
    p.add_argument("--wandb-mode", type=str, default="disabled", choices=["disabled", "offline", "online"])
    p.add_argument("--wandb-project", type=str, default="park-load-balance-torch")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-name", type=str, default=None)
    p.add_argument("--wandb-tags", type=str, default="")
    p.add_argument("--wandb-group", type=str, default=None)
    p.add_argument("--wandb-dir", type=str, default=None)
    args = p.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env_cfg = dict(
        num_servers=int(args.num_servers),
        num_stream_jobs=int(args.num_stream_jobs),
        job_size_pareto_shape=float(args.job_size_pareto_shape),
        job_size_pareto_scale=float(args.job_size_pareto_scale),
        job_interval=float(args.job_interval),
        obs_high=float(args.obs_high),
        reward_time_scale=float(args.reward_time_scale),
        seed=int(args.seed),
        workload_trace_dir=args.workload_trace_dir,
        workload_trace_path=args.workload_trace_path,
        trace_select_mode=args.trace_select_mode,
    )

    env = LoadBalanceTorchEnv(**env_cfg)
    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.n)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActorCritic(obs_dim, act_dim, hidden=int(args.hidden)).to(device)
    opt = optim.Adam(model.parameters(), lr=float(args.lr))
    if args.resume_from is not None:
        resume_ckpt = _load_resume_checkpoint(str(args.resume_from))
        model_state = resume_ckpt.get("model_state_dict", resume_ckpt.get("model"))
        if model_state is None:
            raise RuntimeError(f"resume checkpoint is missing model weights: {args.resume_from}")
        model.load_state_dict(model_state)
        if int(args.resume_optimizer) != 0:
            optimizer_state = resume_ckpt.get("optimizer_state_dict", resume_ckpt.get("optimizer"))
            if optimizer_state is not None:
                opt.load_state_dict(optimizer_state)
                _optimizer_to_device(opt, device)
                for group in opt.param_groups:
                    group["lr"] = float(args.lr)
        if int(args.resume_entropy_weight) != 0 and resume_ckpt.get("entropy_weight") is not None:
            args.ent_coef = float(resume_ckpt["entropy_weight"])

    wandb_run = _init_wandb(
        args,
        {
            **vars(args),
            "model_kind": "park_lb_victim",
            "obs_dim": int(obs_dim),
            "act_dim": int(act_dim),
            "device": str(device),
        },
    )

    os.makedirs(args.save_dir, exist_ok=True)
    best_score_per_job = float("-inf")
    train_start_time = time.perf_counter()
    last_iter = 0

    for it in range(1, int(args.total_iters) + 1):
        last_iter = int(it)
        batch = collect_rollout(
            env,
            model,
            device,
            steps=int(args.rollout_steps),
            gamma=float(args.gamma),
            lam=float(args.gae_lambda),
        )

        n = batch.obs.shape[0]
        idx = np.arange(n)
        loss_values: list[float] = []
        policy_loss_values: list[float] = []
        value_loss_values: list[float] = []
        raw_value_loss_values: list[float] = []
        entropy_values: list[float] = []
        ratio_values: list[float] = []
        approx_kl_values: list[float] = []
        clip_frac_values: list[float] = []
        grad_norm_values: list[float] = []
        value_target_std = float(batch.ret.detach().std(unbiased=False).item())
        value_loss_scale = max(value_target_std, 1e-8)
        for _ in range(int(args.ppo_epochs)):
            np.random.shuffle(idx)
            for start in range(0, n, int(args.mini_batch)):
                mb = idx[start : start + int(args.mini_batch)]

                logits, v = model(batch.obs[mb])
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(batch.act[mb])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logp - batch.logp[mb])
                surr1 = ratio * batch.adv[mb]
                surr2 = torch.clamp(ratio, 1.0 - float(args.clip), 1.0 + float(args.clip)) * batch.adv[mb]
                policy_loss = -torch.min(surr1, surr2).mean()

                raw_value_loss = ((v - batch.ret[mb]) ** 2).mean()
                if int(args.normalize_value_loss) != 0:
                    value_loss = raw_value_loss / float(value_loss_scale * value_loss_scale)
                else:
                    value_loss = raw_value_loss
                loss = policy_loss + float(args.vf_coef) * value_loss - float(args.ent_coef) * entropy

                opt.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                opt.step()

                with torch.no_grad():
                    approx_kl = (batch.logp[mb] - new_logp).mean()
                    clip_frac = (torch.abs(ratio - 1.0) > float(args.clip)).to(torch.float32).mean()
                    loss_values.append(float(loss.item()))
                    policy_loss_values.append(float(policy_loss.item()))
                    value_loss_values.append(float(value_loss.item()))
                    entropy_values.append(float(entropy.item()))
                    ratio_values.append(float(ratio.mean().item()))
                    approx_kl_values.append(float(approx_kl.item()))
                    clip_frac_values.append(float(clip_frac.item()))
                    grad_norm_values.append(float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm))
                    raw_value_loss_values.append(float(raw_value_loss.item()))

        if wandb_run is not None:
            rewards_np = batch.rew.detach().cpu().numpy()
            done_np = batch.done.detach().cpu().numpy()
            wandb_run.log(
                {
                    "train/iter": int(it),
                    "train/env_steps": int(it * int(args.rollout_steps)),
                    "train/rollout_steps": int(n),
                    "train/rollout_reward_mean": float(np.mean(rewards_np)),
                    "train/rollout_reward_sum": float(np.sum(rewards_np)),
                    "train/rollout_reward_std": float(np.std(rewards_np)),
                    "train/rollout_done_count": int(np.sum(done_np)),
                    "train/return_mean": float(batch.ret.detach().cpu().mean().item()),
                    "train/return_std": float(value_target_std),
                    "train/adv_mean": float(batch.adv.detach().cpu().mean().item()),
                    "train/loss/total": _mean(loss_values),
                    "train/loss/policy": _mean(policy_loss_values),
                    "train/loss/value": _mean(value_loss_values),
                    "train/loss/value_raw": _mean(raw_value_loss_values),
                    "train/policy_loss": _mean(policy_loss_values),
                    "train/value_loss": _mean(value_loss_values),
                    "train/entropy": _mean(entropy_values),
                    "train/ratio_mean": _mean(ratio_values),
                    "train/approx_kl": _mean(approx_kl_values),
                    "train/clip_frac": _mean(clip_frac_values),
                    "train/grad_norm": _mean(grad_norm_values),
                    "train/lr": float(args.lr),
                    "train/entropy_weight": float(args.ent_coef),
                    "train/reward_time_scale": float(args.reward_time_scale),
                    "train/normalize_value_loss": int(args.normalize_value_loss),
                    "train/time_ms": float((time.perf_counter() - train_start_time) * 1000.0),
                },
                step=int(it),
            )

        if it % int(args.eval_every) == 0 or it == 1:
            score, score_per_job = evaluate(env_cfg, model, device, episodes=int(args.eval_episodes))
            print(
                f"iter={it} eval_mean_reward={score:.3f} "
                f"eval_mean_reward_per_job={score_per_job:.6f}"
            )
            ckpt_payload = _checkpoint_payload(
                args=args,
                model=model,
                opt=opt,
                iter_idx=int(it),
                best_score_per_job=float(best_score_per_job),
                eval_mean_reward=float(score),
                eval_mean_reward_per_job=float(score_per_job),
            )
            torch.save(ckpt_payload, os.path.join(args.save_dir, f"checkpoint_{it:04d}.pt"))
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/iter": int(it),
                        "eval/mean_reward": float(score),
                        "eval/mean_reward_per_job": float(score_per_job),
                        "eval/reward_per_job": float(score_per_job),
                        "checkpoint/saved": 1,
                        "checkpoint/iter": int(it),
                    },
                    step=int(it),
                )
            if float(score_per_job) > float(best_score_per_job):
                best_score_per_job = float(score_per_job)
                torch.save(ckpt_payload, os.path.join(args.save_dir, "best.pt"))
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/iter": int(it),
                            "eval/best_mean_reward_per_job": float(best_score_per_job),
                            "checkpoint/new_best": 1,
                        },
                        step=int(it),
                    )
                print(f"iter={it} new_best_per_job={best_score_per_job:.6f} saved={args.save_dir}/best.pt")

    last_payload = _checkpoint_payload(
        args=args,
        model=model,
        opt=opt,
        iter_idx=int(last_iter),
        best_score_per_job=float(best_score_per_job),
        eval_mean_reward=float("nan"),
        eval_mean_reward_per_job=float("nan"),
    )
    torch.save(last_payload, os.path.join(args.save_dir, "last.pt"))
    env.close()
    if wandb_run is not None:
        wandb_run.summary["eval/best_mean_reward_per_job"] = float(best_score_per_job)
        wandb_run.summary["checkpoint/save_dir"] = str(args.save_dir)
        wandb_run.summary["checkpoint/last_path"] = str(os.path.join(args.save_dir, "last.pt"))
        wandb_run.finish()
    print(f"done. checkpoints in: {args.save_dir}")


if __name__ == "__main__":
    main()
