#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any


"""
Example usage:
python scripts/plot_attacks_shields_heatmaps.py --csv traces/eval_models_attacks_shields.csv --out-dir traces/eval_models_attacks_shields_plots

python scripts/plot_attacks_shields_heatmaps.py --csv traces/eval_models_base_attacks_shields.csv --out-dir traces/eval_models_base_attacks_shields_plots
"""

def _eprint(msg: str) -> None:
    import sys

    print(str(msg), file=sys.stderr, flush=True)


def _require_imports() -> tuple[Any, Any, Any, Any]:
    import tempfile

    #* Make matplotlib/fontconfig caches writable on cluster nodes / sandboxed envs.
    tmp = str(tempfile.gettempdir())
    os.environ.setdefault("XDG_CACHE_HOME", os.path.join(tmp, "xdg_cache"))
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(tmp, "matplotlib"))
    os.makedirs(str(os.environ["XDG_CACHE_HOME"]), exist_ok=True)
    os.makedirs(str(os.environ["MPLCONFIGDIR"]), exist_ok=True)

    try:
        import pandas as pd  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit("* Missing dependency: pandas (try: pip install pandas)") from exc

    try:
        import matplotlib  # type: ignore

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
        import numpy as np  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit("* Missing dependency: matplotlib/numpy (try: pip install matplotlib numpy)") from exc

    try:
        import seaborn as sns  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit("* Missing dependency: seaborn (try: pip install seaborn)") from exc

    return pd, sns, plt, np


def _ordered_unique(values: list[str], preferred: list[str]) -> list[str]:
    seen = set()
    out: list[str] = []
    for v in preferred:
        if v in values and v not in seen:
            out.append(v)
            seen.add(v)
    for v in sorted(values):
        if v not in seen:
            out.append(v)
            seen.add(v)
    return out


def _symmetric_vlim(np, data, *, q: float = 0.95) -> float | None:
    flat = data.reshape(-1)
    flat = flat[np.isfinite(flat)]
    if int(flat.size) == 0:
        return None
    absvals = np.abs(flat)
    v = float(np.quantile(absvals, float(q)))
    if not float(v) > 0.0:
        return None
    return float(v)


def _mean_or_single(np, series) -> float | None:
    if series is None:
        return None
    vals = [float(x) for x in list(series) if x is not None and float(x) == float(x)]
    if not vals:
        return None
    if len(vals) == 1:
        return float(vals[0])
    return float(np.mean(np.asarray(vals, dtype=np.float64)))


def _pretty_row_label(robustification_trace: str) -> str:
    import re

    out = str(robustification_trace)
    if out.startswith("finetune-"):
        out = out[len("finetune-") :]
    out = re.sub(r"-(\d+(?:\.\d+)?)a-(\d+)i$", "", out)
    return out


def _label_newline_on_plus(label: str) -> str:
    #* Improve readability for compound labels.
    s = str(label)
    if " + " in s:
        return "\n+ ".join([p.strip() for p in s.split(" + ")])
    if "+" in s:
        return str(s).replace("+", "\n+ ")
    return s


def _policy_variant_label_for_timing(*, policy_case: str, robustification_trace: str, base_tag: str) -> str:
    r = _pretty_row_label(str(robustification_trace))
    if str(policy_case) == "base":
        return "Base"
    if str(policy_case) == "base+shield":
        return "Base + Shield"
    if str(policy_case) == "robust":
        return f"Robust({r})"
    if str(policy_case) == "robust+shield":
        return f"Robust({r}) + Shield({r})"
    if str(robustification_trace) == str(base_tag):
        return str(policy_case)
    return f"{policy_case}({r})"


def _build_improvement_matrix(
    *,
    pd,
    np,
    df,
    base_tag: str,
    robust_rows: list[str],
    attack_cols: list[str],
    variant_case: str,
    trace_victim_mode: str,
    fixed_victim_suffix: str | None,
    metric_col: str,
    lower_is_better: bool,
    eps_denom: float,
) -> Any:
    #* Positive = improvement relative to base (QoE up; gap/regret down).
    data = np.full((len(robust_rows), len(attack_cols)), np.nan, dtype=np.float64)

    for i, r in enumerate(list(robust_rows)):
        for j, a in enumerate(list(attack_cols)):
            if str(a) == "clean":
                victim_suffix = "clean"
            else:
                if str(trace_victim_mode) == "fixed":
                    if not fixed_victim_suffix:
                        continue
                    victim_suffix = str(fixed_victim_suffix)
                else:
                    victim_suffix = str(r)

            df_b = df[
                (df["robustification_trace"].astype(str) == str(base_tag))
                & (df["policy_case"].astype(str) == "base")
                & (df["attack_name"].astype(str) == str(a))
                & (df["trace_victim_suffix"].astype(str) == str(victim_suffix))
            ]
            b_val = _mean_or_single(np, df_b.get(metric_col))
            if b_val is None:
                continue

            if str(variant_case) == "base+shield":
                df_v = df[
                    (df["robustification_trace"].astype(str) == str(base_tag))
                    & (df["policy_case"].astype(str) == "base+shield")
                    & (df["attack_name"].astype(str) == str(a))
                    & (df["trace_victim_suffix"].astype(str) == str(victim_suffix))
                ]
            else:
                df_v = df[
                    (df["robustification_trace"].astype(str) == str(r))
                    & (df["policy_case"].astype(str) == str(variant_case))
                    & (df["attack_name"].astype(str) == str(a))
                    & (df["trace_victim_suffix"].astype(str) == str(victim_suffix))
                ]

            v_val = _mean_or_single(np, df_v.get(metric_col))
            if v_val is None:
                continue

            denom = float(abs(float(b_val)))
            if not float(denom) > float(eps_denom):
                continue

            if bool(lower_is_better):
                data[i, j] = 100.0 * (float(b_val) - float(v_val)) / float(denom)
            else:
                data[i, j] = 100.0 * (float(v_val) - float(b_val)) / float(denom)

    return data


def _plot_heatmap_single(
    *,
    sns,
    plt,
    np,
    out_path: Path,
    row_labels: list[str],
    col_labels: list[str],
    data,
    title: str,
    x_label: str,
    y_label: str,
    fontsize: int,
    annot: bool,
    clean_sep_x: float | None = None,
) -> None:
    cell_scale = 1.6
    fig_w = max(12.0, float(cell_scale) * float(len(col_labels)))
    fig_h = max(8.0, float(cell_scale) * float(len(row_labels)))

    fig, ax = plt.subplots(1, 1, figsize=(float(fig_w), float(fig_h)))
    vlim = _symmetric_vlim(np, data, q=0.95)
    vmin = -float(vlim) if vlim is not None else None
    vmax = float(vlim) if vlim is not None else None

    sns.heatmap(
        data,
        ax=ax,
        cmap="RdYlGn",
        center=0.0,
        vmin=vmin,
        vmax=vmax,
        square=True,
        annot=bool(annot),
        fmt=".2f",
        annot_kws={"size": int(fontsize)},
        xticklabels=list(col_labels),
        yticklabels=list(row_labels),
        cbar_kws={"label": "Δ% vs base (positive = improvement)"},
    )

    #* Enforce square-ish cells robustly even with colorbars/tight_layout.
    try:
        ax.set_box_aspect(float(len(row_labels)) / float(len(col_labels)))
    except Exception:
        pass
    try:
        ax.set_aspect("equal", adjustable="box")
    except Exception:
        pass

    ax.set_title(str(title), fontsize=int(fontsize) + 2)
    ax.tick_params(axis="x", labelrotation=0, labelsize=int(fontsize))
    ax.tick_params(axis="y", labelrotation=0, labelsize=int(fontsize))
    ax.set_xlabel(str(x_label), fontsize=int(fontsize) + 1)
    ax.set_ylabel(str(y_label), fontsize=int(fontsize) + 1)

    #* Separate clean columns from adversarial columns.
    if clean_sep_x is not None and len(col_labels) > 1:
        ax.axvline(x=float(clean_sep_x), color="black", linewidth=3)
    else:
        if col_labels and str(col_labels[-1]) == "clean" and len(col_labels) > 1:
            ax.axvline(x=len(col_labels) - 1, color="black", linewidth=3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot heatmaps from eval_models_attacks_shields.csv.")
    parser.add_argument("--csv", type=str, default="traces/eval_models_attacks_shields.csv")
    parser.add_argument("--out-dir", type=str, default="traces/eval_models_attacks_shields_plots")
    parser.add_argument("--base-tag", type=str, default="original", help="* robustification_trace tag for the base model")
    parser.add_argument(
        "--exclude-attacks",
        type=str,
        nargs="*",
        default=[],
        help="* Attack names to exclude from plots (space- or comma-separated); also filters matching robustification traces, e.g. --exclude-attacks pgd-feat-step",
    )
    parser.add_argument("--fontsize", type=int, default=20)
    parser.add_argument("--annot", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eps-denom", type=float, default=1e-9)
    args = parser.parse_args()

    pd, sns, plt, np = _require_imports()

    csv_path = Path(str(args.csv))
    if not csv_path.exists():
        raise FileNotFoundError(f"* CSV not found: {csv_path}")

    df = pd.read_csv(str(csv_path))
    required = {
        "robustification_trace",
        "policy_case",
        "attack_name",
        "trace_victim_suffix",
        "num_traces",
        "num_traces_regret_valid",
        "qoe_total_mean",
        "qoe_total_p50",
        "qoe_total_p95",
        "qoe_total_sum",
        "regret_mean",
        "regret_p50",
        "regret_p95",
        "regret_sum",
        "oracle_qoe_total_mean",
        "oracle_qoe_total_p50",
        "oracle_qoe_total_p95",
        "oracle_qoe_total_sum",
        "qoe_gap_mean",
        "qoe_gap_p50",
        "qoe_gap_p95",
        "qoe_gap_sum",
    }
    missing = sorted([c for c in list(required) if c not in set(df.columns)])
    if missing:
        raise RuntimeError(f"* Missing required columns in CSV: {missing}")

    metric_cols = [
        "num_traces",
        "num_traces_regret_valid",
        "qoe_total_mean",
        "qoe_total_p50",
        "qoe_total_p95",
        "qoe_total_sum",
        "regret_mean",
        "regret_p50",
        "regret_p95",
        "regret_sum",
        "oracle_qoe_total_mean",
        "oracle_qoe_total_p50",
        "oracle_qoe_total_p95",
        "oracle_qoe_total_sum",
        "qoe_gap_mean",
        "qoe_gap_p50",
        "qoe_gap_p95",
        "qoe_gap_sum",
    ]

    #* Defensive aggregation: handle repeated rows by averaging numeric summaries.
    df_agg = (
        df.groupby(["robustification_trace", "policy_case", "attack_name", "trace_victim_suffix"], as_index=False)[metric_cols]
        .mean(numeric_only=True)
        .copy()
    )

    attack_order = [
        "dtgap-sk-feat",
        "dtgap-sk-none",
        "hotnets19",
        "pgd-feat",
        "pgd-feat-step",
        "pgd-manual",
        "pgd-none",
        "clean",
    ]
    attack_cols = _ordered_unique([str(x) for x in list(df_agg["attack_name"].unique())], preferred=list(attack_order))

    #* Optional filtering of attack columns (keep ordering + clean-last logic).
    exclude_attacks: set[str] = set()
    for item in list(args.exclude_attacks):
        for part in str(item).split(","):
            p = str(part).strip()
            if p:
                exclude_attacks.add(p)
    if exclude_attacks:
        attack_cols = [str(a) for a in list(attack_cols) if str(a) not in exclude_attacks]
        if not attack_cols:
            raise RuntimeError(f"* All attacks were excluded via --exclude-attacks: {sorted(exclude_attacks)}")

    if "clean" in set(attack_cols):
        attack_cols = [str(x) for x in list(attack_cols) if str(x) != "clean"] + ["clean"]

    base_tag = str(args.base_tag)
    robust_traces = sorted(set(str(x) for x in list(df_agg["robustification_trace"].unique()) if str(x) != str(base_tag)))
    if not robust_traces:
        raise RuntimeError("* No robustification rows found (all rows match --base-tag?)")

    victim_suffixes = sorted(set(str(x) for x in list(df_agg["trace_victim_suffix"].unique()) if str(x) != "clean"))
    trace_victim_mode = "per_row"
    fixed_victim_suffix: str | None = None
    if len(victim_suffixes) == 1 and str(victim_suffixes[0]) in {"base", "original"}:
        trace_victim_mode = "fixed"
        fixed_victim_suffix = str(victim_suffixes[0])
        robust_rows = list(robust_traces)
    else:
        robust_rows = [str(x) for x in list(victim_suffixes) if str(x) in set(robust_traces)]
    if not robust_rows:
        raise RuntimeError("* No robust_rows found after resolving trace victim suffix mode.")

    #* Apply --exclude-attacks to robustification traces as well.
    if exclude_attacks:
        robust_rows = [str(r) for r in list(robust_rows) if _pretty_row_label(str(r)) not in exclude_attacks]
        if not robust_rows:
            raise RuntimeError(f"* All robustification traces were excluded via --exclude-attacks: {sorted(exclude_attacks)}")

    variant_cases = ["base+shield", "robust", "robust+shield"]
    variant_cases = [str(c) for c in list(variant_cases) if str(c) in set(str(x) for x in list(df_agg["policy_case"].unique()))]
    if not variant_cases:
        raise RuntimeError("* No expected policy_case values found (need at least one of: base+shield, robust, robust+shield).")

    sns.set_style("ticks", {"grid.linestyle": ":"})
    plt.rcParams["axes.grid"] = True
    plt.rcParams["savefig.transparent"] = False
    plt.rcParams.update(
        {
            "font.size": int(args.fontsize),
            "axes.titlesize": int(args.fontsize) + 2,
            "axes.labelsize": int(args.fontsize),
            "xtick.labelsize": int(args.fontsize),
            "ytick.labelsize": int(args.fontsize),
        }
    )

    out_dir = Path(str(args.out_dir))

    stats = [
        ("mean", "mean"),
        ("p50", "p50"),
        ("p95", "p95"),
        ("sum", "sum"),
    ]

    metric_specs = [
        (
            "qoe",
            "Raw QoE (per-trace)",
            False,
            {"mean": "qoe_total_mean", "p50": "qoe_total_p50", "p95": "qoe_total_p95", "sum": "qoe_total_sum"},
        ),
        (
            "regret",
            "Regret (avg over episode; per-step)",
            True,
            {"mean": "regret_mean", "p50": "regret_p50", "p95": "regret_p95", "sum": "regret_sum"},
        ),
        (
            "qoe_gap",
            "QoE gap (oracle - policy; per-trace)",
            True,
            {"mean": "qoe_gap_mean", "p50": "qoe_gap_p50", "p95": "qoe_gap_p95", "sum": "qoe_gap_sum"},
        ),
    ]
    for metric_key, metric_title, lower_is_better, col_map in list(metric_specs):
        for stat_key, stat_title in list(stats):
            metric_col = str(col_map[str(stat_key)])
            variant_mats: dict[str, Any] = {}
            for variant_case in list(variant_cases):
                variant_mats[str(variant_case)] = _build_improvement_matrix(
                    pd=pd,
                    np=np,
                    df=df_agg,
                    base_tag=str(base_tag),
                    robust_rows=list(robust_rows),
                    attack_cols=list(attack_cols),
                    variant_case=str(variant_case),
                    trace_victim_mode=str(trace_victim_mode),
                    fixed_victim_suffix=str(fixed_victim_suffix) if fixed_victim_suffix else None,
                    metric_col=str(metric_col),
                    lower_is_better=bool(lower_is_better),
                    eps_denom=float(args.eps_denom),
                )

            v = len(list(variant_cases))
            combined_cols: list[str] = []
            for robustification_trace in list(robust_rows):
                r = _pretty_row_label(str(robustification_trace))
                for variant_case in list(variant_cases):
                    if str(variant_case) == "base+shield":
                        combined_cols.append(f"Base + Shield({r})")
                    elif str(variant_case) == "robust":
                        combined_cols.append(f"Robust({r})")
                    elif str(variant_case) == "robust+shield":
                        combined_cols.append(f"Robust({r}) + Shield({r})")
                    else:
                        combined_cols.append(f"{variant_case}({r})")

            combined = np.full((len(list(attack_cols)), len(list(robust_rows)) * v), np.nan, dtype=np.float64)
            for robust_idx in range(len(list(robust_rows))):
                for variant_idx, variant_case in enumerate(list(variant_cases)):
                    mat = variant_mats.get(str(variant_case))
                    if mat is None:
                        continue
                    col_idx = int(robust_idx) * int(v) + int(variant_idx)
                    combined[:, int(col_idx)] = mat[int(robust_idx), :]

            out_path = out_dir / f"heatmap_delta_{metric_key}_{stat_key}.png"
            _plot_heatmap_single(
                sns=sns,
                plt=plt,
                np=np,
                out_path=out_path,
                row_labels=[_label_newline_on_plus(str(x)) for x in list(combined_cols)],
                col_labels=[_label_newline_on_plus(str(x)) for x in list(attack_cols)],
                data=combined.T,
                title=f"{metric_title} ({stat_title}): Δ% vs base",
                x_label="Attack",
                y_label="Policy variant",
                fontsize=int(args.fontsize),
                annot=bool(args.annot),
            )
            _eprint(f"* Wrote: {out_path}")

    decision_cols = {"decision_count_total", "decision_time_s_total"}
    if decision_cols.issubset(set(df.columns)):
        df_bar = df.copy()
        df_bar = df_bar[df_bar["attack_name"].astype(str) != "clean"].copy()
        if exclude_attacks:
            df_bar = df_bar[~df_bar["attack_name"].astype(str).isin(set(exclude_attacks))].copy()
            df_bar = df_bar[
                ~df_bar["robustification_trace"].astype(str).map(lambda s: _pretty_row_label(str(s)) in set(exclude_attacks))
            ].copy()

        df_bar["policy_variant"] = df_bar.apply(
            lambda r: _policy_variant_label_for_timing(
                policy_case=str(r.get("policy_case", "")),
                robustification_trace=str(r.get("robustification_trace", "")),
                base_tag=str(base_tag),
            ),
            axis=1,
        )

        df_bar["decision_count_total"] = pd.to_numeric(df_bar["decision_count_total"], errors="coerce")
        df_bar["decision_time_s_total"] = pd.to_numeric(df_bar["decision_time_s_total"], errors="coerce")
        df_bar = df_bar[
            (df_bar["decision_count_total"].notna())
            & (df_bar["decision_time_s_total"].notna())
            & (df_bar["decision_count_total"] > 0)
        ].copy()

        if not df_bar.empty:
            g = (
                df_bar.groupby("policy_variant", as_index=False)[["decision_time_s_total", "decision_count_total"]]
                .sum(numeric_only=True)
                .copy()
            )
            g["decision_time_ms_mean"] = g["decision_time_s_total"] / g["decision_count_total"] * 1000.0
            g = g.sort_values(["decision_time_ms_mean", "policy_variant"], ascending=[True, True])

            y_labels = [_label_newline_on_plus(str(x)) for x in list(g["policy_variant"].astype(str))]
            x_vals = [float(x) for x in list(g["decision_time_ms_mean"].astype(float))]

            fig_h = max(6.0, 0.6 * float(len(y_labels)) + 1.0)
            fig, ax = plt.subplots(1, 1, figsize=(12.0, float(fig_h)))
            ax.barh(y_labels, x_vals, color="tab:gray", edgecolor="black")
            ax.set_xlabel("Decision time (ms / decision)", fontsize=int(args.fontsize) + 1)
            ax.set_ylabel("Policy variant", fontsize=int(args.fontsize) + 1)
            ax.set_title("Decision-making time", fontsize=int(args.fontsize) + 2)
            ax.tick_params(axis="x", labelsize=int(args.fontsize))
            ax.tick_params(axis="y", labelsize=int(args.fontsize))
            fig.tight_layout()

            out_path = out_dir / "bar_decision_time_ms_by_policy_variant.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(out_path), dpi=200, bbox_inches="tight", pad_inches=0.05)
            plt.close(fig)
            _eprint(f"* Wrote: {out_path}")
        else:
            _eprint("* Skipping decision-time bar plot (no non-clean rows with decision timing).")
    else:
        _eprint("* Skipping decision-time bar plot (missing decision timing columns in CSV).")


if __name__ == "__main__":
    main()
