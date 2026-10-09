from __future__ import annotations

import argparse
import os

from .bo_search import GenetBOTraceGenerator
from .runtime import load_runtime, resolve_repo_path
from .scoring import TraceScorer
from .trace_family import DEFAULT_DURATION_S


DEFAULT_BW_MIN_MBPS = 1e-3
DEFAULT_BW_MAX_MBPS = 8.951518
DEFAULT_BW_CHANGE_INTERVAL = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Genet-style BO traces in the current cooked-trace format.")
    parser.add_argument("--pensieve-model", type=str, default="ppo/nn_model_ep_426300.pth")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=4242423)
    parser.add_argument("--bw-min", type=float, default=DEFAULT_BW_MIN_MBPS)
    parser.add_argument("--bw-max", type=float, default=DEFAULT_BW_MAX_MBPS)
    parser.add_argument("--duration-s", type=float, default=DEFAULT_DURATION_S)
    parser.add_argument(
        "--bw-change-interval",
        type=float,
        default=DEFAULT_BW_CHANGE_INTERVAL,
        help="Fixed bandwidth-change interval. Used by default unless --optimize-bw-change-interval is enabled.",
    )
    parser.add_argument(
        "--optimize-bw-change-interval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include bw_change_interval in the BO search space instead of fixing it.",
    )
    parser.add_argument("--bw-change-interval-min", type=float, default=1.0)
    parser.add_argument("--bw-change-interval-max", type=float, default=100.0)
    parser.add_argument("--traces-per-eval", type=int, default=10)
    parser.add_argument("--bo-init-points", type=int, default=10)
    parser.add_argument("--bo-iter", type=int, default=5)
    parser.add_argument(
        "--objective",
        type=str,
        default="mpc_gap",
        choices=["mpc_gap", "oracle_gap"],
        help="BO objective: MPC-vs-Pensieve QoE gap or K-step-oracle-vs-Pensieve QoE gap.",
    )
    parser.add_argument("--oracle-k", type=int, default=5)
    parser.add_argument("--out-dir", type=str, default="traces/genet/mpc_gap")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = resolve_repo_path(str(args.out_dir))
    if str(args.out_dir) == "traces/genet/mpc_gap" and str(args.objective) != "mpc_gap":
        out_dir = resolve_repo_path(os.path.join("traces", "genet", str(args.objective)))

    runtime = load_runtime(str(args.pensieve_model))
    scorer = TraceScorer(runtime, oracle_k=int(args.oracle_k))
    generator = GenetBOTraceGenerator(
        scorer=scorer,
        objective=str(args.objective),
        seed=int(args.seed),
        episodes=int(args.episodes),
        out_dir=str(out_dir),
        bw_min_mbps=float(args.bw_min),
        bw_max_mbps=float(args.bw_max),
        bw_change_interval=float(args.bw_change_interval),
        optimize_bw_change_interval=bool(args.optimize_bw_change_interval),
        bw_change_interval_min=float(args.bw_change_interval_min),
        bw_change_interval_max=float(args.bw_change_interval_max),
        duration_s=float(args.duration_s),
        traces_per_eval=int(args.traces_per_eval),
        bo_init_points=int(args.bo_init_points),
        bo_iter=int(args.bo_iter),
        overwrite=bool(args.overwrite),
    )
    result = generator.run()
    print(f"* Wrote {int(args.episodes)} Genet-style traces to: {out_dir}")
    print(f"* Objective: {str(args.objective)}")
    print(
        "* Best params: "
        f"min_bw={float(result.best_params.min_bw_mbps):.6f} Mbps, "
        f"max_bw={float(result.best_params.max_bw_mbps):.6f} Mbps, "
        f"bw_change_interval={float(result.best_params.bw_change_interval):.6f}"
    )
    print(
        "* bw_change_interval mode: "
        f"{'optimized' if bool(args.optimize_bw_change_interval) else 'fixed'}"
    )
    print(f"* Best target gap: {float(result.best_target):.6f}")


if __name__ == "__main__":
    main()
