#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from typing import Any


VIDEO_BIT_RATE_KBPS: tuple[int, ...] = (300, 750, 1200, 1850, 2850, 4300)


def _resolve_path(repo_relative_or_abs: str) -> str:
    expanded = os.path.expandvars(os.path.expanduser(str(repo_relative_or_abs)))
    if os.path.isabs(expanded):
        return expanded
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(repo_root, expanded)


def _default_out_path(in_path: str) -> str:
    base, ext = os.path.splitext(str(in_path))
    if ext.lower() != ".csv":
        return base + "_labeled.csv"
    return base[:-4] + "_labeled.csv"


def _contiguous_blocks(sorted_actions: list[int]) -> list[list[int]]:
    if len(sorted_actions) == 0:
        return []
    blocks: list[list[int]] = [[int(sorted_actions[0])]]
    for a in sorted_actions[1:]:
        a = int(a)
        if a == int(blocks[-1][-1]) + 1:
            blocks[-1].append(a)
        else:
            blocks.append([a])
    return blocks


def _pick_block(
    *,
    blocks: list[list[int]],
    q_by_action: list[float],
    strategy: str,
) -> list[int]:
    if len(blocks) == 0:
        return []

    if str(strategy) == "best_action":
        best_a = int(max(range(len(q_by_action)), key=lambda i: float(q_by_action[i])))
        for b in blocks:
            if best_a in b:
                return list(b)

    if str(strategy) == "largest":
        best = None
        for b in blocks:
            if best is None or len(b) > len(best):
                best = list(b)
                continue
            if len(b) < len(best):
                continue
            b_score = max(float(q_by_action[i]) for i in b)
            best_score = max(float(q_by_action[i]) for i in best)
            if b_score > best_score:
                best = list(b)
        return list(best) if best is not None else list(blocks[0])

    return list(blocks[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Label a rule-mining dataset with min/max action bounds.")
    parser.add_argument("--in", dest="in_path", required=True, help="Input CSV (from gen_rule_mining_dataset.py).")
    parser.add_argument(
        "--out",
        dest="out_path",
        default=None,
        help="Output CSV path (defaults to '<in>_labeled.csv' next to the input).",
    )
    parser.add_argument(
        "--gap-pct-open-threshold",
        type=float,
        default=40.0,
        help="If oracle_gap_pct <= threshold and oracle_best_reward_mean is stable, label full action range.",
    )
    parser.add_argument(
        "--oracle-best-min-abs",
        type=float,
        default=0.1,
        help="If oracle_best_reward_mean <= 0 or abs(oracle_best_reward_mean) < this, do not gate by oracle_gap_pct.",
    )
    parser.add_argument(
        "--regret-delta",
        type=float,
        default=10.0,
        help="Allowed actions satisfy 100*(V_oracle - Q_oracle(action))/max(abs(V_oracle), oracle_best_min_abs) <= regret_delta.",
    )
    parser.add_argument(
        "--regret-delta-min-abs",
        type=float,
        default=0.3,
        help="Lower clamp (absolute) on the derived regret threshold used by --regret-delta.",
    )
    parser.add_argument(
        "--regret-delta-max-abs",
        type=float,
        default=1.0,
        help="Upper clamp (absolute) on the derived regret threshold used by --regret-delta. Set <=0 to disable.",
    )
    parser.add_argument(
        "--noncontiguous",
        choices=["largest", "best_action", "as_is"],
        default="best_action",
        help="How to turn a non-contiguous allowed set into (min,max).",
    )

    args = parser.parse_args()

    in_path = _resolve_path(str(args.in_path))
    out_path = _resolve_path(str(args.out_path)) if args.out_path else _default_out_path(in_path)

    if not os.path.isfile(in_path):
        raise FileNotFoundError(f"* Input CSV not found: {in_path}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    try:
        import numpy as np
        import pandas as pd
    except Exception as e:  # pragma: no cover
        raise RuntimeError("* Missing dependencies (numpy/pandas). Use `venv/bin/python`.") from e

    df = pd.read_csv(in_path)

    a_dim = int(len(VIDEO_BIT_RATE_KBPS))
    q_cols = [f"oracle_q_reward_mean_by_action{i}" for i in range(a_dim)]
    missing = [c for c in q_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"* Missing oracle Q columns: {missing}")

    min_idx_out: list[int] = []
    max_idx_out: list[int] = []
    min_kbps_out: list[int] = []
    max_kbps_out: list[int] = []
    label_reason_out: list[str] = []

    gap_thr = float(args.gap_pct_open_threshold)
    oracle_best_min_abs = float(args.oracle_best_min_abs)
    regret_delta_pct = float(args.regret_delta)
    regret_delta_min_abs = float(args.regret_delta_min_abs)
    regret_delta_max_abs = float(args.regret_delta_max_abs)

    for _, row in df.iterrows():
        oracle_best = float(row.get("oracle_best_reward_mean", float("nan")))
        gap_pct = float(row.get("oracle_gap_pct", float("nan")))

        stable_oracle_best = bool(np.isfinite(oracle_best) and oracle_best > 0.0 and abs(oracle_best) >= oracle_best_min_abs)
        if stable_oracle_best and np.isfinite(gap_pct) and float(gap_pct) <= float(gap_thr):
            mi, ma = 0, a_dim - 1
            reason = "open_low_gap"
        else:
            q_raw = [float(row.get(c, float("nan"))) for c in q_cols]
            finite_actions = [i for i, q in enumerate(q_raw) if np.isfinite(float(q))]
            if len(finite_actions) == 0:
                mi, ma = 0, a_dim - 1
                reason = "open_missing_q"
            else:
                q_vals = [float(q_raw[i]) for i in range(a_dim)]
                v = max(q_vals[i] for i in finite_actions)
                denom = max(float(oracle_best_min_abs), abs(float(v)))
                regret_limit_abs = float(denom) * float(regret_delta_pct) / 100.0
                regret_limit_abs = max(float(regret_delta_min_abs), float(regret_limit_abs))
                if float(regret_delta_max_abs) > 0.0:
                    regret_limit_abs = min(float(regret_limit_abs), float(regret_delta_max_abs))
                allowed = [i for i in finite_actions if float(v) - float(q_vals[i]) <= float(regret_limit_abs)]
                if len(allowed) == 0:
                    best_a = int(max(finite_actions, key=lambda i: float(q_vals[i])))
                    allowed = [best_a]

                allowed_sorted = sorted(set(int(a) for a in allowed))
                if str(args.noncontiguous) == "as_is":
                    mi, ma = int(min(allowed_sorted)), int(max(allowed_sorted))
                else:
                    blocks = _contiguous_blocks(allowed_sorted)
                    picked = _pick_block(blocks=blocks, q_by_action=q_vals, strategy=str(args.noncontiguous))
                    if len(picked) == 0:
                        mi, ma = 0, a_dim - 1
                    else:
                        mi, ma = int(min(picked)), int(max(picked))
                reason = "regret"

        mi = int(max(0, min(a_dim - 1, mi)))
        ma = int(max(0, min(a_dim - 1, ma)))
        if mi > ma:
            mi, ma = ma, mi

        min_idx_out.append(int(mi))
        max_idx_out.append(int(ma))
        min_kbps_out.append(int(VIDEO_BIT_RATE_KBPS[int(mi)]))
        max_kbps_out.append(int(VIDEO_BIT_RATE_KBPS[int(ma)]))
        label_reason_out.append(str(reason))

    df_out = df.copy()
    df_out["label_min_action_idx"] = min_idx_out
    df_out["label_max_action_idx"] = max_idx_out
    df_out["label_min_action_kbps"] = min_kbps_out
    df_out["label_max_action_kbps"] = max_kbps_out
    df_out["label_reason"] = label_reason_out
    df_out["label_gap_pct_open_threshold"] = float(gap_thr)
    df_out["label_oracle_best_min_abs"] = float(oracle_best_min_abs)
    df_out["label_regret_delta"] = float(regret_delta_pct)
    df_out["label_regret_delta_min_abs"] = float(regret_delta_min_abs)
    df_out["label_regret_delta_max_abs"] = float(regret_delta_max_abs)
    df_out["label_noncontiguous"] = str(args.noncontiguous)

    df_out.to_csv(out_path, index=False)
    print(f"* Wrote labeled CSV: {out_path}")


if __name__ == "__main__":
    main()
