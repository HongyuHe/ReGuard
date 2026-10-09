from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from online.sim_core import evaluate_victim_vs_reference_on_jobs, parse_job_trace
from train_ppo import ActorCritic


def _parse_service_rates(text: str | None, num_servers: int) -> list[float]:
    if text is None:
        if int(num_servers) == 10:
            return [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05]
        return np.linspace(0.15, 1.05, num=int(num_servers), dtype=np.float64).tolist()
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(vals) != int(num_servers):
        raise ValueError("--service-rates must have exactly num_servers comma-separated values")
    return vals


class VictimActor:
    def __init__(self, checkpoint_path: str, num_servers: int) -> None:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        cfg = dict(ckpt.get("config", {}))
        hidden = int(cfg.get("hidden", 128))
        self.model = ActorCritic(obs_dim=int(num_servers + 1), act_dim=int(num_servers), hidden=hidden)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> int:
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits, _ = self.model(obs_t)
        return int(torch.argmax(logits, dim=-1).item())


def _collect_trace_files(d: Path) -> list[Path]:
    files = sorted(d.glob("*.csv"))
    if len(files) == 0:
        files = sorted(d.glob("*.json"))
    files = [p for p in files if p.name.lower() != "index.csv"]
    return files


def _evaluate_dir(
    *,
    trace_dir: Path,
    victim: VictimActor,
    service_rates: list[float],
    obs_high: float,
    reward_time_scale: float,
    reference_window_len: int,
    out_csv_path: Path,
) -> dict[str, float | int]:
    rows = []
    files = _collect_trace_files(trace_dir)
    if len(files) == 0:
        raise RuntimeError(f"no trace files found in {trace_dir}")

    for p in files:
        jobs = parse_job_trace(str(p))
        metrics = evaluate_victim_vs_reference_on_jobs(
            jobs=jobs,
            victim_action_fn=victim.act,
            service_rates=service_rates,
            obs_high=float(obs_high),
            reward_time_scale=float(reward_time_scale),
            reference_window_len=int(reference_window_len),
        )
        rows.append({
            "trace_file": p.name,
            "num_jobs": int(metrics["num_jobs"]),
            "victim_total_reward": float(metrics["victim_total_reward"]),
            "oracle_total_reward": float(metrics["oracle_total_reward"]),
            "oracle_gap": float(metrics["oracle_gap"]),
            "victim_reward_per_job": float(metrics["victim_reward_per_job"]),
            "oracle_reward_per_job": float(metrics["oracle_reward_per_job"]),
            "oracle_gap_per_job": float(metrics["oracle_gap_per_job"]),
            "reference_window": int(metrics.get("reference_window", reference_window_len)),
        })

    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with out_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "trace_file",
                "num_jobs",
                "victim_total_reward",
                "oracle_total_reward",
                "oracle_gap",
                "victim_reward_per_job",
                "oracle_reward_per_job",
                "oracle_gap_per_job",
                "reference_window",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    gap_vals = np.asarray([float(r["oracle_gap"]) for r in rows], dtype=np.float64)
    gap_per_job_vals = np.asarray([float(r["oracle_gap_per_job"]) for r in rows], dtype=np.float64)

    return {
        "num_traces": int(len(rows)),
        "oracle_gap_mean": float(np.mean(gap_vals)),
        "oracle_gap_std": float(np.std(gap_vals)),
        "oracle_gap_per_job_mean": float(np.mean(gap_per_job_vals)),
        "oracle_gap_per_job_std": float(np.std(gap_per_job_vals)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate victim/oracle gap on clean and adversarial traces")
    p.add_argument("--victim-checkpoint", type=str, required=True)
    p.add_argument("--clean-dir", type=str, required=True)
    p.add_argument("--adv-dir", type=str, required=True)
    p.add_argument("--num-servers", type=int, default=10)
    p.add_argument("--service-rates", type=str, default=None)
    p.add_argument("--obs-high", type=float, default=500000.0)
    p.add_argument("--reward-time-scale", type=float, default=1000.0)
    p.add_argument("--reference-window", type=int, default=5)
    p.add_argument("--out-dir", type=str, default="online/eval")
    args = p.parse_args()
    if int(args.reference_window) <= 0:
        raise ValueError("--reference-window must be positive")

    service_rates = _parse_service_rates(args.service_rates, args.num_servers)
    victim = VictimActor(checkpoint_path=str(args.victim_checkpoint), num_servers=int(args.num_servers))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_summary = _evaluate_dir(
        trace_dir=Path(args.clean_dir),
        victim=victim,
        service_rates=service_rates,
        obs_high=float(args.obs_high),
        reward_time_scale=float(args.reward_time_scale),
        reference_window_len=int(args.reference_window),
        out_csv_path=out_dir / "oracle_gap_clean.csv",
    )
    adv_summary = _evaluate_dir(
        trace_dir=Path(args.adv_dir),
        victim=victim,
        service_rates=service_rates,
        obs_high=float(args.obs_high),
        reward_time_scale=float(args.reward_time_scale),
        reference_window_len=int(args.reference_window),
        out_csv_path=out_dir / "oracle_gap_adv.csv",
    )

    delta = float(adv_summary["oracle_gap_mean"]) - float(clean_summary["oracle_gap_mean"])
    delta_per_job = float(adv_summary["oracle_gap_per_job_mean"]) - float(clean_summary["oracle_gap_per_job_mean"])

    summary = {
        "clean": clean_summary,
        "adv": adv_summary,
        "delta_oracle_gap_mean": float(delta),
        "delta_oracle_gap_per_job_mean": float(delta_per_job),
        "reference_window": int(args.reference_window),
    }
    with (out_dir / "oracle_gap_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"wrote: {out_dir / 'oracle_gap_clean.csv'}")
    print(f"wrote: {out_dir / 'oracle_gap_adv.csv'}")
    print(f"wrote: {out_dir / 'oracle_gap_summary.json'}")


if __name__ == "__main__":
    main()
