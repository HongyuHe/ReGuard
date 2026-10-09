#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _resolve_path(path: str) -> Path:
    p = Path(os.path.expandvars(os.path.expanduser(str(path))))
    if p.is_absolute():
        return p
    return Path(__file__).resolve().parent.parent / p


def _default_out_path(in_path: Path) -> Path:
    if in_path.suffix.lower() == ".csv":
        return in_path.with_name(in_path.stem + "_labeled.csv")
    return in_path.with_suffix(str(in_path.suffix) + "_labeled.csv")


def _infer_num_servers(df: pd.DataFrame) -> int:
    q_cols = [str(c) for c in df.columns if str(c).startswith("reference_q_action_")]
    if len(q_cols) == 0:
        raise RuntimeError("missing reference_q_action_* columns")
    indices = sorted(int(str(c).rsplit("_", 1)[1]) for c in list(q_cols))
    return int(max(indices) + 1)


def _reference_policy_names(df: pd.DataFrame) -> list[str]:
    names = []
    for col in df.columns:
        text = str(col)
        if text.startswith("reference_reward/"):
            names.append(text.split("/", 1)[1])
    return sorted(set(names))


def _gap_pct(row: pd.Series, *, denom_min_abs: float) -> float:
    if "reference_gap_pct" in row:
        value = float(row.get("reference_gap_pct", float("nan")))
        if np.isfinite(value):
            return float(value)
    gap = float(row.get("reference_gap", float("nan")))
    ref = float(row.get("best_reference_reward", float("nan")))
    if not np.isfinite(gap):
        return float("nan")
    denom = max(float(denom_min_abs), abs(float(ref)) if np.isfinite(ref) else 0.0)
    return float(100.0 * float(gap) / float(denom))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label Park LB shield data with risk and corrective action masks.")
    parser.add_argument("--in", dest="in_path", type=str, required=True)
    parser.add_argument("--out", dest="out_path", type=str, default=None)
    parser.add_argument("--risk-gap-pct-threshold", type=float, default=30.0)
    parser.add_argument("--risk-gap-min-abs", type=float, default=0.0)
    parser.add_argument("--gap-denom-min-abs", type=float, default=1e-6)
    parser.add_argument("--action-regret-pct", type=float, default=10.0)
    parser.add_argument("--action-regret-min-abs", type=float, default=0.0)
    parser.add_argument("--action-regret-max-abs", type=float, default=0.0)
    parser.add_argument("--allow-source", choices=["policies", "q"], default="policies")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_path = _resolve_path(str(args.in_path))
    out_path = _resolve_path(str(args.out_path)) if args.out_path is not None else _default_out_path(in_path)
    if not in_path.is_file():
        raise FileNotFoundError(f"input CSV not found: {in_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    num_servers = _infer_num_servers(df)
    q_cols = [f"reference_q_action_{i}" for i in range(int(num_servers))]
    missing = [c for c in list(q_cols) if c not in df.columns]
    if missing:
        raise RuntimeError(f"missing Q columns: {missing}")
    policy_names = _reference_policy_names(df)
    if str(args.allow_source) == "policies" and len(policy_names) == 0:
        raise RuntimeError("allow-source=policies requires reference_reward/* and reference_action/* columns")

    risk_labels: list[int] = []
    best_actions: list[int] = []
    allowed_counts: list[int] = []
    reasons: list[str] = []
    allowed_by_action: dict[int, list[int]] = {int(i): [] for i in range(int(num_servers))}
    gap_pct_values: list[float] = []
    action_regret_limits: list[float] = []

    for _, row in df.iterrows():
        gap_abs = float(row.get("reference_gap", float("nan")))
        gap_pct = _gap_pct(row, denom_min_abs=float(args.gap_denom_min_abs))
        risky = bool(
            np.isfinite(gap_abs)
            and np.isfinite(gap_pct)
            and float(gap_abs) >= float(args.risk_gap_min_abs)
            and float(gap_pct) >= float(args.risk_gap_pct_threshold)
        )
        if str(args.allow_source) == "policies":
            policy_rows = []
            for name in list(policy_names):
                reward = float(row.get(f"reference_reward/{name}", float("nan")))
                action = int(row.get(f"reference_action/{name}", -1))
                if np.isfinite(reward) and 0 <= int(action) < int(num_servers):
                    policy_rows.append((str(name), int(action), float(reward)))
            if len(policy_rows) > 0:
                best_policy_name, best_action, best_reward = max(
                    policy_rows,
                    key=lambda item: (float(item[2]), -int(item[1]), str(item[0])),
                )
                _ = best_policy_name
                denom = max(float(args.gap_denom_min_abs), abs(float(best_reward)))
                action_regret_limit = float(denom) * float(args.action_regret_pct) / 100.0
                action_regret_limit = max(float(args.action_regret_min_abs), float(action_regret_limit))
                if float(args.action_regret_max_abs) > 0.0:
                    action_regret_limit = min(float(action_regret_limit), float(args.action_regret_max_abs))
                if bool(risky):
                    allowed = sorted(
                        {
                            int(action)
                            for _name, action, reward in list(policy_rows)
                            if float(best_reward - float(reward)) <= float(action_regret_limit)
                        }
                    )
                    reason = "risky_reference_policy_mask"
                else:
                    allowed = list(range(int(num_servers)))
                    reason = "not_risky_open"
            else:
                best_action = int(row.get("best_reference_action", -1))
                action_regret_limit = float("nan")
                if bool(risky) and 0 <= int(best_action) < int(num_servers):
                    allowed = [int(best_action)]
                    reason = "risky_best_reference_action"
                else:
                    allowed = list(range(int(num_servers)))
                    reason = "missing_policy_open"
        else:
            q_vals = np.asarray([float(row.get(c, float("nan"))) for c in list(q_cols)], dtype=np.float64)
            finite = np.isfinite(q_vals)
            if np.any(finite):
                best_q = float(np.max(q_vals[finite]))
                best_action = int(np.argmax(np.where(finite, q_vals, -np.inf)))
                denom = max(float(args.gap_denom_min_abs), abs(float(best_q)))
                action_regret_limit = float(denom) * float(args.action_regret_pct) / 100.0
                action_regret_limit = max(float(args.action_regret_min_abs), float(action_regret_limit))
                if float(args.action_regret_max_abs) > 0.0:
                    action_regret_limit = min(float(action_regret_limit), float(args.action_regret_max_abs))
                if bool(risky):
                    allowed = [
                        int(i)
                        for i in range(int(num_servers))
                        if bool(finite[int(i)]) and float(best_q - float(q_vals[int(i)])) <= float(action_regret_limit)
                    ]
                    reason = "risky_reference_q_mask"
                else:
                    allowed = list(range(int(num_servers)))
                    reason = "not_risky_open"
            else:
                best_action = int(row.get("best_reference_action", -1))
                action_regret_limit = float("nan")
                if bool(risky) and 0 <= int(best_action) < int(num_servers):
                    allowed = [int(best_action)]
                    reason = "risky_best_reference_action"
                else:
                    allowed = list(range(int(num_servers)))
                    reason = "missing_q_open"
        if len(allowed) == 0:
            allowed = [int(best_action)] if 0 <= int(best_action) < int(num_servers) else list(range(int(num_servers)))
            reason = "empty_mask_repaired"

        risk_labels.append(int(bool(risky)))
        best_actions.append(int(best_action))
        allowed_counts.append(int(len(set(allowed))))
        reasons.append(str(reason))
        gap_pct_values.append(float(gap_pct))
        action_regret_limits.append(float(action_regret_limit))
        allowed_set = set(int(a) for a in list(allowed))
        for action in range(int(num_servers)):
            allowed_by_action[int(action)].append(int(int(action) in allowed_set))

    out_df = df.copy()
    out_df["label_risk"] = risk_labels
    out_df["label_best_action"] = best_actions
    out_df["label_allowed_count"] = allowed_counts
    out_df["label_reason"] = reasons
    out_df["label_reference_gap_pct"] = gap_pct_values
    out_df["label_action_regret_limit"] = action_regret_limits
    out_df["label_risk_gap_pct_threshold"] = float(args.risk_gap_pct_threshold)
    out_df["label_risk_gap_min_abs"] = float(args.risk_gap_min_abs)
    out_df["label_action_regret_pct"] = float(args.action_regret_pct)
    out_df["label_action_regret_min_abs"] = float(args.action_regret_min_abs)
    out_df["label_action_regret_max_abs"] = float(args.action_regret_max_abs)
    out_df["label_allow_source"] = str(args.allow_source)
    for action in range(int(num_servers)):
        out_df[f"label_allow_server{action}"] = allowed_by_action[int(action)]

    out_df.to_csv(out_path, index=False)
    risk_rate = float(np.mean(np.asarray(risk_labels, dtype=np.float64))) if risk_labels else float("nan")
    print(f"wrote labeled CSV: {out_path}")
    print(f"rows={len(out_df)} risk_rate={risk_rate:.6f} num_servers={num_servers}")


if __name__ == "__main__":
    main()
