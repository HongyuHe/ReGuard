#!/usr/bin/env python3
"""
Run one Genet BO search round for Sage and export the discovered hard traces in
the same bundle / manifest format as `scripts/generate_online_adv_traces.py`.

Example usage:

time python scripts/generate_sage_genet_round1_traces.py \
  --config-path attacks/models/gap_adv_20260321_gap-constrained-all-loss_50ms_300k.config.json \
  --pretrained-model-dir sage_rl/rl_module/sage_models/final \
  --test-manifest attacks/test/manifest.json \
  --out-dir attacks/adv_traces/genet_gap-constrained-all-loss_50ms_300k \
  --num-traces 55 \
  --seed 42 \
  --segment-steps 8 \
  --traces-per-eval 10 \
  --bo-init-points 10 \
  --bo-iter 5 \
  --overwrite
"""

from __future__ import annotations

import argparse
import os
import sys


if __package__ in (None, ""):
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from genet import (
        SageGenetBOTraceGenerator,
        infer_search_bounds,
        load_config_payload,
        repo_root_from_script,
        resolve_repo_path,
    )
else:
    from genet import (
        SageGenetBOTraceGenerator,
        infer_search_bounds,
        load_config_payload,
        repo_root_from_script,
        resolve_repo_path,
    )


def _add_bool_flag(parser: argparse.ArgumentParser, *, name: str, default: bool, help_text: str) -> None:
    destination = str(name).replace("-", "_")
    parser.add_argument(f"--{name}", dest=destination, action="store_true", help=help_text)
    parser.add_argument(f"--no-{name}", dest=destination, action="store_false")
    parser.set_defaults(**{destination: bool(default)})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Genet BO search over shared bandwidth/loss schedule families for the current "
            "Sage checkpoint and export the Round-1 discovered traces."
        )
    )
    parser.add_argument("--repo-root", type=str, default=repo_root_from_script(__file__))
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument(
        "--pretrained-model-dir",
        type=str,
        default="sage_rl/rl_module/sage_models/final",
        help="TensorFlow checkpoint dir for the Sage policy being stress-tested.",
    )
    parser.add_argument("--test-manifest", type=str, default="attacks/test/manifest.json")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="attacks/adv_traces/genet_round1",
    )
    parser.add_argument("--runtime-dir", type=str, default="attacks/runtime")
    parser.add_argument("--num-traces", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bw-min-mbps", type=float, default=None)
    parser.add_argument("--bw-max-mbps", type=float, default=None)
    parser.add_argument("--loss-min", type=float, default=None)
    parser.add_argument("--loss-max", type=float, default=None)
    parser.add_argument(
        "--segment-steps",
        type=float,
        default=8.0,
        help="Fixed Genet segment-length cap, analogous to Pensieve's bw_change_interval.",
    )
    _add_bool_flag(
        parser,
        name="optimize-segment-steps",
        default=False,
        help_text="Include segment_steps in the BO search space instead of fixing it.",
    )
    parser.add_argument("--segment-steps-min", type=float, default=1.0)
    parser.add_argument("--segment-steps-max", type=float, default=8.0)
    parser.add_argument("--traces-per-eval", type=int, default=10)
    parser.add_argument("--bo-init-points", type=int, default=10)
    parser.add_argument("--bo-iter", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = os.path.abspath(os.path.expanduser(args.repo_root))
    config_path, config_payload = load_config_payload(repo_root, str(args.config_path))
    bounds = infer_search_bounds(
        config_payload=config_payload,
        bw_min_mbps=args.bw_min_mbps,
        bw_max_mbps=args.bw_max_mbps,
        loss_min=args.loss_min,
        loss_max=args.loss_max,
        segment_steps_min=args.segment_steps_min if bool(args.optimize_segment_steps) else float(args.segment_steps),
        segment_steps_max=args.segment_steps_max if bool(args.optimize_segment_steps) else float(args.segment_steps),
    )
    generator = SageGenetBOTraceGenerator(
        repo_root=repo_root,
        config_path=config_path,
        config_payload=config_payload,
        sage_model_dir=resolve_repo_path(repo_root, str(args.pretrained_model_dir)),
        test_manifest_path=str(args.test_manifest),
        out_dir=str(args.out_dir),
        runtime_dir=str(args.runtime_dir),
        bounds=bounds,
        seed=int(args.seed),
        num_traces=int(args.num_traces),
        traces_per_eval=int(args.traces_per_eval),
        bo_init_points=int(args.bo_init_points),
        bo_iter=int(args.bo_iter),
        optimize_segment_steps=bool(args.optimize_segment_steps),
        fixed_segment_steps=float(args.segment_steps),
        overwrite=bool(args.overwrite),
    )
    result = generator.run()
    print(f"Generated manifest: {result.generated_manifest_path}")
    print(
        "Best family: "
        f"bw=[{result.best_params.min_shared_bw_mbps:.6f}, {result.best_params.max_shared_bw_mbps:.6f}] Mbps, "
        f"loss=[{result.best_params.min_shared_loss:.6f}, {result.best_params.max_shared_loss:.6f}], "
        f"segment_steps={result.best_params.segment_steps:.6f}"
    )
    print(f"Best hard absolute gap target: {result.best_target:.6f}")


if __name__ == "__main__":
    main()
