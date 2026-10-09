#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any


"""
Example usage:
python scripts/plot_robust_eval_heatmaps.py \
    --csv traces/robustification_eval_189e.csv \
    --out-dir traces/eval_robustification_heatmaps_189e/ \
    --exclude-attacks pgd-manual
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


def _pretty_row_label(robustification_trace: str) -> str:
    import re

    out = str(robustification_trace)
    if out.startswith("finetune-"):
        out = out[len("finetune-") :]
    out = re.sub(r"-(\d+(?:\.\d+)?)a-(\d+)i$", "", out)
    return out


def _pretty_col_label(attack_name: str) -> str:
    return str(attack_name)


def _canonical_attack_name(name: str) -> str:
    #* Normalize user-provided attack identifiers across plots/CSV naming.
    import re

    out = str(name).strip()
    if "/" in out:
        out = out.split("/", 1)[0]
    if out.startswith("finetune-"):
        out = out[len("finetune-") :]
    out = re.sub(r"-(\d+(?:\.\d+)?)a-(\d+)i$", "", out)
    return out


def _attack_family(attack_name: str) -> str:
    a = str(attack_name)
    if a == "clean":
        return "Clean"
    if a.startswith("hotnets"):
        return "HotNets19"
    if a.startswith("dtgap") or a.startswith("dt"):
        return "DT"
    if a.startswith("pgd"):
        return "PGD"
    return "Other"


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


def _build_improvement_matrix(
    *,
    pd,
    np,
    df,
    base_tag: str,
    robust_rows: list[str],
    attack_cols: list[str],
    metric_col: str,
    lower_is_better: bool,
    eps_denom: float,
) -> Any:
    #* Positive = improvement relative to base (QoE up; gap down).
    data = np.full((len(robust_rows), len(attack_cols)), np.nan, dtype=np.float64)

    df_base = df[df["robustification_trace"] == str(base_tag)]

    for i, r in enumerate(list(robust_rows)):
        for j, a in enumerate(list(attack_cols)):
            if str(a) == "clean":
                df_r = df[(df["robustification_trace"] == str(r)) & (df["attack_name"] == "clean")]
                df_b = df_base[df_base["attack_name"] == "clean"]
            else:
                df_r = df[
                    (df["robustification_trace"] == str(r))
                    & (df["attack_name"] == str(a))
                    & (df["trace_victim_suffix"] == str(r))
                ]
                df_b = df_base[(df_base["attack_name"] == str(a)) & (df_base["trace_victim_suffix"] == str(r))]

            r_val = _mean_or_single(np, df_r.get(metric_col))
            b_val = _mean_or_single(np, df_b.get(metric_col))
            if r_val is None or b_val is None:
                continue

            denom = float(abs(float(b_val)))
            if not float(denom) > float(eps_denom):
                continue

            if bool(lower_is_better):
                data[i, j] = 100.0 * (float(b_val) - float(r_val)) / float(denom)
            else:
                data[i, j] = 100.0 * (float(r_val) - float(b_val)) / float(denom)

    return data


def _worst_over_attacks(np, data, attack_cols: list[str]) -> Any:
    idx = [i for i, a in enumerate(list(attack_cols)) if str(a) != "clean"]
    out = np.full((int(data.shape[0]), 1), np.nan, dtype=np.float64)
    for r in range(int(data.shape[0])):
        vals = [float(data[r, c]) for c in idx if np.isfinite(float(data[r, c]))]
        if vals:
            out[r, 0] = float(min(vals))
    return out


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
        fmt=".0f",
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
    ax.tick_params(axis="x", labelrotation=45, labelsize=int(fontsize))
    ax.tick_params(axis="y", labelrotation=0, labelsize=int(fontsize))
    ax.set_xlabel(str(x_label), fontsize=int(fontsize) + 1)
    ax.set_ylabel(str(y_label), fontsize=int(fontsize) + 1)

    #* Separate the last column (clean) from adversarial columns.
    if col_labels and str(col_labels[-1]) == "clean" and len(col_labels) > 1:
        ax.axvline(x=len(col_labels) - 1, color="black", linewidth=3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200)
    plt.close(fig)


def _plot_oracle_bar(
    *,
    sns,
    plt,
    pd,
    out_path: Path,
    df,
    base_tag: str,
    fontsize: int,
    exclude_attacks: set[str],
) -> None:
    from matplotlib.patches import Patch  # type: ignore

    base = df[df["robustification_trace"] == str(base_tag)].copy()
    if exclude_attacks:
        base = base[~base["attack_name"].astype(str).isin(set(exclude_attacks))].copy()
    if base.empty:
        raise RuntimeError("* No adversarial trace groups found for the oracle bar chart.")
    by_attack = (
        base.groupby("attack_name", as_index=False)
        .agg(
            oracle_qoe_total_sum=("oracle_qoe_total_sum", "sum"),
            qoe_gap_sum=("qoe_gap_sum", "sum"),
            regret_sum=("regret_sum", "sum"),
            num_traces=("num_traces", "sum"),
            num_traces_regret_valid=("num_traces_regret_valid", "sum"),
        )
        .assign(
            oracle_qoe_total_avg=lambda d: d["oracle_qoe_total_sum"].astype(float) / d["num_traces"].astype(float),
            qoe_gap_avg=lambda d: d["qoe_gap_sum"].astype(float) / d["num_traces"].astype(float),
            regret_avg=lambda d: d["regret_sum"].astype(float) / d["num_traces_regret_valid"].astype(float),
            attack_family=lambda d: d["attack_name"].astype(str).apply(_attack_family),
        )
    )

    by_attack_qoe = by_attack.sort_values("oracle_qoe_total_avg", ascending=True).copy()
    by_attack_gap = by_attack.sort_values("qoe_gap_avg", ascending=False).copy()
    by_attack_regret = by_attack.sort_values("regret_avg", ascending=False).copy()

    order_qoe = [str(x) for x in list(by_attack_qoe["attack_name"])]
    order_gap = [str(x) for x in list(by_attack_gap["attack_name"])]
    order_regret = [str(x) for x in list(by_attack_regret["attack_name"])]

    palette = {
        "PGD": "tab:blue",
        "DT": "tab:green",
        "HotNets19": "tab:red",
        "Clean": "gray",
        "Other": "gray",
    }

    fig_h_each = max(8.0, 0.75 * float(max(len(by_attack_qoe), len(by_attack_gap))))
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(20.0, float(3.0 * fig_h_each)))
    sns.barplot(
        data=by_attack_qoe,
        y="attack_name",
        x="oracle_qoe_total_avg",
        hue="attack_family",
        palette=palette,
        ax=ax0,
        orient="h",
        order=list(order_qoe),
        dodge=False,
        edgecolor="black",
        linewidth=1.0,
    )
    ax0.invert_yaxis()
    ax0.set_title("Oracle QoE (avg over traces; grouped by attack; lower = harder)", fontsize=int(fontsize) + 2)
    ax0.set_xlabel("Avg oracle QoE per trace", fontsize=int(fontsize) + 1)
    ax0.set_ylabel("Attack", fontsize=int(fontsize) + 1)
    ax0.tick_params(axis="x", labelsize=int(fontsize))
    ax0.tick_params(axis="y", labelsize=int(fontsize))

    sns.barplot(
        data=by_attack_gap,
        y="attack_name",
        x="qoe_gap_avg",
        hue="attack_family",
        palette=palette,
        ax=ax1,
        orient="h",
        order=list(order_gap),
        dodge=False,
        edgecolor="black",
        linewidth=1.0,
    )
    ax1.invert_yaxis()
    ax1.set_title("Oracle gap (avg over traces; grouped by attack; higher = more effective)", fontsize=int(fontsize) + 2)
    ax1.set_xlabel("Avg oracle gap per trace (oracle − policy)", fontsize=int(fontsize) + 1)
    ax1.set_ylabel("Attack", fontsize=int(fontsize) + 1)
    ax1.tick_params(axis="x", labelsize=int(fontsize))
    ax1.tick_params(axis="y", labelsize=int(fontsize))

    sns.barplot(
        data=by_attack_regret,
        y="attack_name",
        x="regret_avg",
        hue="attack_family",
        palette=palette,
        ax=ax2,
        orient="h",
        order=list(order_regret),
        dodge=False,
        edgecolor="black",
        linewidth=1.0,
    )
    ax2.invert_yaxis()
    ax2.set_title("Regret (avg over traces; grouped by attack; higher = more effective)", fontsize=int(fontsize) + 2)
    ax2.set_xlabel("Avg regret per trace", fontsize=int(fontsize) + 1)
    ax2.set_ylabel("Attack", fontsize=int(fontsize) + 1)
    ax2.tick_params(axis="x", labelsize=int(fontsize))
    ax2.tick_params(axis="y", labelsize=int(fontsize))

    for ax in [ax0, ax1, ax2]:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    handles = [Patch(facecolor=str(palette[k]), edgecolor="black", label=str(k)) for k in ["PGD", "DT", "HotNets19", "Clean"]]
    fig.tight_layout()
    fig.legend(
        handles=handles,
        loc="center right",
        bbox_to_anchor=(0.9, 0.5),
        ncol=1,
        frameon=False,
        fontsize=int(fontsize),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sns.despine(fig=fig, ax=ax0)
    sns.despine(fig=fig, ax=ax1)
    sns.despine(fig=fig, ax=ax2)
    fig.savefig(str(out_path), dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot robustification vs attack heatmaps from robustification_eval.csv.")
    parser.add_argument("--csv", type=str, default="traces/robustification_eval_full.csv")
    parser.add_argument("--out-dir", type=str, default="traces/eval_robustification_heatmaps/")
    parser.add_argument("--base-tag", type=str, default="original", help="* robustification_trace tag for base model rows")
    parser.add_argument(
        "--exclude-attacks",
        type=str,
        nargs="*",
        default=[],
        help="* Attack identifiers to exclude from attack columns (and ranking bar plots).",
    )
    parser.add_argument(
        "--exclude-robustifications",
        type=str,
        nargs="*",
        default=[],
        help="* Robustification-trace identifiers to exclude from row axis only.",
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
        "oracle_qoe_total_sum",
        "qoe_gap_mean",
        "qoe_gap_p50",
        "qoe_gap_p95",
        "qoe_gap_sum",
    ]
    df_agg = (
        df.groupby(["robustification_trace", "attack_name", "trace_victim_suffix"], as_index=False)[metric_cols]
        .mean(numeric_only=True)
        .copy()
    )

    excluded_attacks: set[str] = set()
    for raw in list(args.exclude_attacks):
        if raw is None:
            continue
        for part in str(raw).split(","):
            token = str(part).strip()
            if token:
                excluded_attacks.add(_canonical_attack_name(token))

    excluded_robustifications: set[str] = set()
    for raw in list(args.exclude_robustifications):
        if raw is None:
            continue
        for part in str(raw).split(","):
            token = str(part).strip()
            if token:
                excluded_robustifications.add(_canonical_attack_name(token))

    attack_order = [
        "dt-sk-feat",
        "dt-sk-none",
        "hotnets19",
        "rl-unconstrained",
        "pgd-feat",
        "pgd-feat-step",
        "pgd-manual",
        "pgd-none",
        "clean",
    ]
    attacks = _ordered_unique([str(x) for x in list(df_agg["attack_name"].unique())], preferred=list(attack_order))
    if excluded_attacks:
        attacks = [a for a in list(attacks) if _canonical_attack_name(str(a)) not in set(excluded_attacks)]
    if "clean" in set(attacks):
        attacks = [str(x) for x in list(attacks) if str(x) != "clean"] + ["clean"]

    robust_order = [
        "finetune-dt-sk-feat",
        "finetune-dt-sk-none",
        "finetune-hotnets19",
        "finetune-rl-unconstrained",
        "finetune-pgd-feat-0.1a-20i",
        "finetune-pgd-feat-step-0.1a-20i",
        "finetune-pgd-manual-0.1a-20i",
        "finetune-pgd-none-0.1a-20i",
    ]
    robust_rows = [
        str(x)
        for x in _ordered_unique(
            [str(x) for x in list(df_agg["robustification_trace"].unique()) if str(x) != str(args.base_tag)],
            preferred=list(robust_order),
        )
    ]
    if excluded_robustifications:
        robust_rows = [
            r
            for r in list(robust_rows)
            if _canonical_attack_name(str(r)) not in set(excluded_robustifications)
        ]
    if not robust_rows:
        raise RuntimeError("* No robustification rows found (all rows match --base-tag?)")

    row_labels = [_pretty_row_label(x) for x in list(robust_rows)]
    col_labels = [_pretty_col_label(x) for x in list(attacks)]

    sns.set_style("ticks", {'grid.linestyle': ':'})
    # plt.rcParams["font.family"] = "serif"
    plt.rcParams['axes.grid'] = True
    plt.rcParams['savefig.transparent'] = False
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
        ("qoe", "Raw QoE (per-trace)", False, {"mean": "qoe_total_mean", "p50": "qoe_total_p50", "p95": "qoe_total_p95", "sum": "qoe_total_sum"}),
        ("regret", "Regret (avg over episode; per-step)", True, {"mean": "regret_mean", "p50": "regret_p50", "p95": "regret_p95", "sum": "regret_sum"}),
        ("qoe_gap", "QoE gap (oracle - policy; per-trace)", True, {"mean": "qoe_gap_mean", "p50": "qoe_gap_p50", "p95": "qoe_gap_p95", "sum": "qoe_gap_sum"}),
    ]

    for metric_key, metric_title, lower_is_better, col_map in list(metric_specs):
        for stat_key, stat_title in list(stats):
            col = str(col_map[str(stat_key)])
            data = _build_improvement_matrix(
                pd=pd,
                np=np,
                df=df_agg,
                base_tag=str(args.base_tag),
                robust_rows=list(robust_rows),
                attack_cols=list(attacks),
                metric_col=str(col),
                lower_is_better=bool(lower_is_better),
                eps_denom=float(args.eps_denom),
            )
            out_path = out_dir / f"heatmap_delta_{metric_key}_{stat_key}.png"
            _plot_heatmap_single(
                sns=sns,
                plt=plt,
                np=np,
                out_path=out_path,
                row_labels=list(row_labels),
                col_labels=list(col_labels),
                data=data,
                title=f"{metric_title} ({stat_title})",
                x_label="Attack",
                y_label="Robustification Trace",
                fontsize=int(args.fontsize),
                annot=bool(args.annot),
            )
            _eprint(f"* Wrote: {out_path}")

    _plot_oracle_bar(
        sns=sns,
        plt=plt,
        pd=pd,
        out_path=out_dir / "oracle_qoe_sum_by_trace_group.png",
        df=df_agg,
        base_tag=str(args.base_tag),
        fontsize=int(args.fontsize),
        exclude_attacks=set(excluded_attacks),
    )
    _eprint(f"* Wrote: {out_dir / 'oracle_qoe_sum_by_trace_group.png'}")


if __name__ == "__main__":
    main()
