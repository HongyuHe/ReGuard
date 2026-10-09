#!/usr/bin/env python3
"""
#* Example from the repo root:

time python scripts/generate_genet_first_iter_traces.py \
  --model models/parklb_v1_350k.pt \
  --reference-window 5 \
  --export-traces 100 \
  --out-dir traces/genet \
  --overwrite
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from genet.bo_search import GenetBOTraceGenerator
from genet.runtime import load_runtime, resolve_repo_path
from genet.scoring import ParkTraceScorer
from genet.trace_family import JobTraceSearchSpace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the first Genet search iteration against the original Park victim and export the resulting traces."
    )
    parser.add_argument("--model", type=str, default="models/parklb_v1_350k.pt")
    parser.add_argument("--shield-rules-prefix", type=str, default=None)
    parser.add_argument("--shield-risk-trigger-steps", type=int, default=1)
    parser.add_argument("--shield-cooldown-steps", type=int, default=0)
    parser.add_argument(
        "--shield-empty-fallback",
        type=str,
        default="least_load_proxy",
        choices=["least_load_proxy", "keep_rl", "open_all"],
    )
    parser.add_argument(
        "--objective",
        type=str,
        default="reference_gap_per_job",
        choices=["reference_gap_total", "reference_gap_per_job", "regret_mean", "regret_p95"],
    )
    parser.add_argument("--reference-window", "--oracle-k", dest="reference_window", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-jobs", type=int, default=1000)
    parser.add_argument("--base-dt-min", type=float, default=10.0)
    parser.add_argument("--base-dt-max", type=float, default=120.0)
    parser.add_argument("--base-size-scale-min", type=float, default=40.0)
    parser.add_argument("--base-size-scale-max", type=float, default=400.0)
    parser.add_argument("--pareto-shape-min", type=float, default=1.1)
    parser.add_argument("--pareto-shape-max", type=float, default=2.5)
    parser.add_argument("--burst-start-min", type=float, default=0.05)
    parser.add_argument("--burst-start-max", type=float, default=0.8)
    parser.add_argument("--burst-duration-min", type=float, default=0.05)
    parser.add_argument("--burst-duration-max", type=float, default=0.4)
    parser.add_argument("--burst-dt-mult-min", type=float, default=0.05)
    parser.add_argument("--burst-dt-mult-max", type=float, default=0.8)
    parser.add_argument("--burst-size-mult-min", type=float, default=1.0)
    parser.add_argument("--burst-size-mult-max", type=float, default=12.0)
    parser.add_argument("--export-traces", type=int, default=20)
    parser.add_argument("--traces-per-eval", type=int, default=5)
    parser.add_argument("--bo-init-points", type=int, default=8)
    parser.add_argument("--bo-iter", type=int, default=12)
    parser.add_argument("--candidate-pool-size", type=int, default=1024)
    parser.add_argument("--out-dir", type=str, default="traces/genet")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = load_runtime(
        str(args.model),
        reference_window=int(args.reference_window),
        shield_rules_prefix=args.shield_rules_prefix,
        shield_risk_trigger_steps=int(args.shield_risk_trigger_steps),
        shield_cooldown_steps=int(args.shield_cooldown_steps),
        shield_empty_fallback=str(args.shield_empty_fallback),
    )
    scorer = ParkTraceScorer(runtime)
    search_space = JobTraceSearchSpace(
        num_jobs=int(args.num_jobs),
        base_dt_ms_range=(float(args.base_dt_min), float(args.base_dt_max)),
        base_size_scale_bytes_range=(float(args.base_size_scale_min), float(args.base_size_scale_max)),
        pareto_shape_range=(float(args.pareto_shape_min), float(args.pareto_shape_max)),
        burst_start_frac_range=(float(args.burst_start_min), float(args.burst_start_max)),
        burst_duration_frac_range=(float(args.burst_duration_min), float(args.burst_duration_max)),
        burst_dt_multiplier_range=(float(args.burst_dt_mult_min), float(args.burst_dt_mult_max)),
        burst_size_multiplier_range=(float(args.burst_size_mult_min), float(args.burst_size_mult_max)),
    )
    generator = GenetBOTraceGenerator(
        scorer=scorer,
        objective=str(args.objective),
        search_space=search_space,
        seed=int(args.seed),
        export_traces=int(args.export_traces),
        out_dir=str(resolve_repo_path(str(args.out_dir))),
        traces_per_eval=int(args.traces_per_eval),
        init_points=int(args.bo_init_points),
        search_iters=int(args.bo_iter),
        candidate_pool_size=int(args.candidate_pool_size),
        overwrite=bool(args.overwrite),
    )
    result = generator.run()
    meta = {
        "generation_mode": "genet_first_iteration",
        "model": str(resolve_repo_path(str(args.model))),
        "objective": str(args.objective),
        "reference_window": int(args.reference_window),
        "shield_rules_prefix": args.shield_rules_prefix,
        "shield_risk_trigger_steps": int(args.shield_risk_trigger_steps),
        "shield_cooldown_steps": int(args.shield_cooldown_steps),
        "shield_empty_fallback": str(args.shield_empty_fallback),
        "seed": int(args.seed),
        "search_space": search_space.to_dict(),
        "export_traces": int(args.export_traces),
        "traces_per_eval": int(args.traces_per_eval),
        "bo_init_points": int(args.bo_init_points),
        "bo_iter": int(args.bo_iter),
        "candidate_pool_size": int(args.candidate_pool_size),
        "best_target": float(result.best_target),
        "best_eval_id": int(result.best_eval_id),
        "best_params": {
            "num_jobs": int(result.best_params.num_jobs),
            "base_dt_ms": float(result.best_params.base_dt_ms),
            "base_size_scale_bytes": float(result.best_params.base_size_scale_bytes),
            "pareto_shape": float(result.best_params.pareto_shape),
            "burst_start_frac": float(result.best_params.burst_start_frac),
            "burst_duration_frac": float(result.best_params.burst_duration_frac),
            "burst_dt_multiplier": float(result.best_params.burst_dt_multiplier),
            "burst_size_multiplier": float(result.best_params.burst_size_multiplier),
        },
    }
    out_dir = Path(resolve_repo_path(str(args.out_dir)))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"saved first-iteration Genet traces under: {out_dir}")


if __name__ == "__main__":
    main()
