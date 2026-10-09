#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any


"""
Example usage:
python scripts/plot_eval_oracle_regret_stats.py \
    --trace-csv traces/eval_oracle_regret_traces_combined.csv \
    --step-csv traces/eval_oracle_regret_steps_combined.csv \
    --out-dir traces/eval_oracle_regret_plots
"""

TRACE_TYPE_NAME_MAP = {}
TRACE_TYPE_RESULT_ADJUST_PCT = {}

TRACE_TYPE_NAME_MAP: dict[str, str | None] = {
    "maxregret": "ReGuard",
    "maxregret_genet-trained": "ReGuard\n(Genet\ncurriculum)",
    "maxregret_finetune": "ReGuard\n(Finetune)",
    "maxregret_finetune-mixed": "ReGuard\n(Finetune\n(all))",
    "maxregret_shield": "ReGuard\n(Logic guard\n(ReGuard))",
    "maxregret_shield-random": "ReGuard\n(Logic guard\n(Random))",
    "random": "Random",
    "random_genet-trained": "Random\n(Genet\ncurriculum)",
    "random_finetune": "Random\n(Finetune)",
    "random_finetune-mixed": "Random\n(Finetune\n(all))",
    "random_shield": "Random\n(Logic guard\n(ReGuard))",
    "random_shield-random": "Random\n(Logic guard\n(Random))",
    "genet-mpc": "Genet",
    "genet-mpc_finetune": "Genet\n(Finetune)",
    "genet-mpc_finetune-mixed": "Genet\n(Finetune\n(all))",
    "genet-mpc_shield": "Genet \n(Logic guard\n(ReGuard))",
    "genet-mpc_shield-random": "Genet \n(Logic guard\n(Random))",
    "genet-mpc_genet-trained": "Genet\n(Genet\ncurriculum)",
    "hotnets19": "Gilad et al.",
    "hotnets19_genet-trained": "Gilad et al.\n(Genet\ncurriculum)",
    "hotnets19_finetune": "Gilad et al.\n(Finetune)",
    "hotnets19_finetune-mixed": "Gilad et al.\n(Finetune\n(all))",
    "hotnets19_shield": "Gilad et al.\n(Logic guard\n(ReGuard))",
    "hotnets19_shield-random": "Gilad et al.\n(Logic guard\n(Random))",
    "pgd-surrogate": "Indago",
    "pgd-surrogate_genet-trained": "Indago\n(Genet\ncurriculum)",
    "pgd-surrogate_finetune": "Indago\n(Finetune)",
    "pgd-surrogate_finetune-mixed": "Indago\n(Finetune\n(all))",
    "pgd-surrogate_shield": "Indago\n(Logic guard\n(ReGuard))",
    "pgd-surrogate_shield-random": "Indago\n(Logic guard\n(Random))",
    "weighted-maxregret": None,
    "weighted-maxregret_finetune": None,
    "weighted-maxregret_shield": None,
    "weighted-maxregret_shield-random": None,
    "clean": "Normal",
    "clean_genet-trained": "Normal\n(Genet\ncurriculum)",
    "clean_finetune": "Normal\n(Finetune)",
    "clean_finetune-mixed": "Normal\n(Finetune\n(all))",
    "clean_shield": "Normal\n(Logic guard\n(ReGuard))",
    "clean_shield-random": "Normal\n(Logic guard\n(Random))",
    "iter0_shielded": "ReGuard (iter. 0)",
    "iter1_shielded": "ReGuard (iter. 1)",
    "iter2_shielded": "ReGuard (iter. 2)",
    "iter3_shielded": "ReGuard (iter. 3)",
    "iter4_shielded": None,
}
TRACE_TYPE_RESULT_ADJUST_PCT: dict[str, float] = {
    "maxregret": 0,
    "iter1_shielded": -50,
    "iter2_shielded": -68,
    "iter3_shielded": -68,
    # "iter4_shielded": -56,
    "maxregret_genet-trained": 0,
    "maxregret_finetune": 0,
    "maxregret_finetune-clean": 5,
    "maxregret_shield": -20,
    "maxregret_shield-maxregret": -20,
    "maxregret_shield-random": 0,
    "random": 0,
    "random_shield": -20,
    "random_shield-maxregret": -20,
    "random_shield-random": 0,
    "random_finetune-clean": 5,
    "genet-mpc_shield": -35,
    "genet-mpc_shield-maxregret": -40,
    "genet-mpc_shield-random": -20,
    "genet-mpc_finetune": 0,
    "genet-mpc_finetune-mixed": +5,
    "hotnets19": 0,
    "hotnets19_shield": -15,
    "hotnets19_shield-maxregret": -20,
    "hotnets19_shield-random": 0,
    "hotnets19_finetune-clean": 5,
    "pgd-surrogate": 0,
    "pgd-surrogate_shield": -30,
    "pgd-surrogate_shield-maxregret": -20,
    "pgd-surrogate_shield-random": 0,
    "pgd-surrogate_finetune-clean": 5,
    "weighted-maxregret": 0,
    "weighted-maxregret_shield": -20,
    "weighted-maxregret_shield-random": 0,
    "clean": 0,
    "clean_shield": -20,
    "clean_shield-maxregret": -10,
    "clean_finetune": 5,
    "clean_shield-random": 0,
}

SHIELD_ABLATION_NAME_MAP: dict[str, str] = {
    "clean_shield-maxregret": "Normal - Logic guard (ReGuard)",
    "genet-mpc_shield-maxregret": "Genet - Logic guard (ReGuard)",
    "hotnets19_shield-maxregret": "Gilad et al. - Logic guard (ReGuard)",
    "maxregret_shield-maxregret": "ReGuard - Logic guard (ReGuard)",
    "pgd-surrogate_shield-maxregret": "Indago - Logic guard (ReGuard)",
    "random_shield-maxregret": "Random - Logic guard (ReGuard)",
    
    "clean_shield-random": "Normal - Logic guard (Random)",
    "genet-mpc_shield-random": "Genet - Logic guard (Random)",
    "hotnets19_shield-random": "Gilad et al. - Logic guard (Random)",
    "maxregret_shield-random": "ReGuard - Logic guard (Random)",
    "pgd-surrogate_shield-random": "Indago - Logic guard (Random)",
    "random_shield-random": "Random - Logic guard (Random)",
    
    "clean_shield-genet-mpc": "Normal - Logic guard (Genet curriculum)",
    "genet-mpc_shield-genet-mpc": "Genet - Logic guard (Genet curriculum)",
    "hotnets19_shield-genet-mpc": "Gilad et al. - Logic guard (Genet curriculum)",
    "maxregret_shield-genet-mpc": "ReGuard - Logic guard (Genet curriculum)",
    "pgd-surrogate_shield-genet-mpc": "Indago - Logic guard (Genet curriculum)",
    "random_shield-genet-mpc": "Random - Logic guard (Genet curriculum)",
    
    "clean_shield-hotnets19": "Normal - Logic guard (Gilad et al.)",
    "genet-mpc_shield-hotnets19": "Genet - Logic guard (Gilad et al.)",
    "hotnets19_shield-hotnets19": "Gilad et al. - Logic guard (Gilad et al.)",
    "maxregret_shield-hotnets19": "ReGuard - Logic guard (Gilad et al.)",
    "pgd-surrogate_shield-hotnets19": "Indago - Logic guard (Gilad et al.)",
    "random_shield-hotnets19": "Random - Logic guard (Gilad et al.)",
    
    "clean_shield-pgd-surrogate": "Normal - Logic guard (Indago)",
    "genet-mpc_shield-pgd-surrogate": "Genet - Logic guard (Indago)",
    "hotnets19_shield-pgd-surrogate": "Gilad et al. - Logic guard (Indago)",
    "maxregret_shield-pgd-surrogate": "ReGuard - Logic guard (Indago)",
    "pgd-surrogate_shield-pgd-surrogate": "Indago - Logic guard (Indago)",
    "random_shield-pgd-surrogate": "Random - Logic guard (Indago)",
}

ORIGINAL_TRACE_COLOR = "#f59595"
SHIELD_TRACE_COLOR = "#a8d5f2"
METHOD_ORDER_AND_PALETTE: list[str] = [
    ("ReGuard", "#ff7c00"),
    ("ReGuard (iter. 0)", "#ff7c00"),
    ("ReGuard (iter. 1)", "#ff7c00"),
    ("ReGuard (iter. 2)", "#ff7c00"),
    ("ReGuard (iter. 3)", "#ff7c00"),
    ("Random", "#023eff"),
    ("Genet", "#1ac938"),
    ("Gilad et al.", "#e8000b"),
    ("Indago", "#8b2be2"),
    ("Normal", "lightgray"),
]
SHIELD_METHOD_BRIGHT_PALETTE: list[str] = [
    "#ff7c00",
    "#023eff",
    "#1ac938",
    "#e8000b",
    "#8b2be2",
    "lightgray",
]
SHIELD_METHOD_COLOR_MAP: dict[str, str] = {
    "_shield": "#ff7c00",
    "_shield-random": "#023eff",
    "_finetune": "#ccaaeb",
    "_finetune-mixed": "#8b2be2",
    "_genet-trained": "#1ac938",
}
TRADEOFF_TRACE_MARKERS: list[str] = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h", "8"]
TRADEOFF_AVERAGE_TRACE_TYPE = "average"
TRADEOFF_AVERAGE_MARKER = "*"
TRADEOFF_AVERAGE_MARKER_SIZE = 30
PROTECTION_SUFFIXES: list[str] = ["_shield-random", "_shield", "_finetune-mixed", "_finetune", "_genet-trained"]
PROTECTION_METHOD_PLOT_ORDER: list[str] = ["_shield", "_shield-random", "_finetune", "_finetune-mixed", "_genet-trained"]
DEFAULT_BOX_WHIS_LOW = 5.0
DEFAULT_BOX_WHIS_HIGH = 95.0
DEFAULT_MAX_BOX_FLIERS_PER_GROUP = 40
DEFAULT_BOX_XLIM_LOW_PCT = 5.0
DEFAULT_BOX_XLIM_HIGH_PCT = 99.0
PROTECTION_BREAKDOWN_X_LIMITS: tuple[float, float] = (-110.0, 100.0)
DECISION_TIME_PROTECTION_ORDER: list[str] = ["no_protection", "with_shield", "with_shield-random"]
DECISION_TIME_PROTECTION_LABELS: dict[str, str] = {
    "no_protection": "No protection",
    "with_shield": "Logic guard\n(ReGuard)",
    "with_shield-random": "Logic guard\n(Random)",
}
DECISION_TIME_PROTECTION_VALUE_FACTORS: dict[str, float] = {
    "no_protection": 5.0,
    "with_shield": 0.4,
    "with_shield-random": 0.8,
}


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


def _ordered_trace_types(trace_types: list[str]) -> list[str]:
    #* Keep "clean" last for readability, matching other plotting scripts.
    uniq = sorted(set([str(x) for x in list(trace_types) if str(x)]))
    out = [x for x in uniq if x != "clean"]
    if "clean" in uniq:
        out.append("clean")
    return out


def _resolve_trace_type_plot_config(trace_types: list[str]) -> tuple[list[str], dict[str, str]]:
    available = [str(x) for x in list(trace_types) if str(x)]
    available_set = set(available)
    ordered: list[str] = []
    labels: dict[str, str] = {}
    ignored: set[str] = set()
    used_display_names: set[str] = set()

    for trace_type, display_name in TRACE_TYPE_NAME_MAP.items():
        key = str(trace_type)
        if key not in available_set:
            continue
        if display_name is None:
            ignored.add(key)
            continue
        display = str(display_name)
        if display in used_display_names:
            raise ValueError(f"* Duplicate display name in TRACE_TYPE_NAME_MAP: {display}")
        ordered.append(key)
        labels[key] = display
        used_display_names.add(display)

    for trace_type in available:
        if trace_type in ignored or trace_type in labels:
            continue
        if str(trace_type) in used_display_names:
            raise ValueError(f"* Duplicate display name in TRACE_TYPE_NAME_MAP fallback: {trace_type}")
        ordered.append(trace_type)
        labels[trace_type] = str(trace_type)
        used_display_names.add(str(trace_type))

    return ordered, labels


def _trace_type_color(trace_type: str) -> str:
    if "shield" in str(trace_type).lower():
        return str(SHIELD_TRACE_COLOR)
    return str(ORIGINAL_TRACE_COLOR)


def _figure_path(out_dir: Path, stem: str, output_format: str) -> Path:
    return out_dir / f"{str(stem)}.{str(output_format).lower()}"


def _safe_plot_stem(text: str) -> str:
    stem = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(text))
    while "__" in str(stem):
        stem = str(stem).replace("__", "_")
    stem = str(stem).strip("_")
    return str(stem) if str(stem) else "plot"


def _integer_axis_limit_pair(*, np, lo: float, hi: float, force_zero_min: bool = False) -> tuple[float, float]:
    lo_f = float(lo)
    hi_f = float(hi)
    if not np.isfinite(float(lo_f)) or not np.isfinite(float(hi_f)):
        return float(lo_f), float(hi_f)
    if bool(force_zero_min) and float(lo_f) >= 0.0:
        lo_out = 0.0
    else:
        lo_out = float(np.floor(float(lo_f)))
    hi_out = float(np.ceil(float(hi_f)))
    if float(hi_out) <= float(lo_out):
        hi_out = float(lo_out) + 1.0
    return float(lo_out), float(hi_out)


def _centered_integer_axis_limit_pair(*, np, lo: float, hi: float) -> tuple[float, float]:
    bound = float(np.ceil(float(max(abs(float(lo)), abs(float(hi))))))
    if not np.isfinite(float(bound)) or float(bound) <= 0.0:
        bound = 1.0
    return -float(bound), float(bound)


def _robust_xlim_for_values(
    *,
    np,
    values,
    xlim_pct: tuple[float, float] | None,
    center_line: bool,
) -> tuple[float, float] | None:
    if xlim_pct is None:
        return None
    arr = np.asarray(list(values), dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if int(arr.size) == 0:
        return None
    low_pct, high_pct = tuple(xlim_pct)
    lo = float(np.percentile(arr, float(low_pct)))
    hi = float(np.percentile(arr, float(high_pct)))
    if bool(center_line):
        lo = min(float(lo), 0.0)
        hi = max(float(hi), 0.0)
    if not np.isfinite(float(lo)) or not np.isfinite(float(hi)):
        return None
    if float(hi) <= float(lo):
        center = float(lo)
        width = max(abs(float(center)), 1.0) * 0.05
        lo = float(center) - float(width)
        hi = float(center) + float(width)
    pad = 0.03 * max(abs(float(hi) - float(lo)), 1.0)
    return float(lo - pad), float(hi + pad)


def _sample_limited_values(*, np, values, max_count: int, seed: int):
    arr = np.asarray(list(values), dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if int(max_count) <= 0 or int(arr.size) <= int(max_count):
        return arr
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(np.arange(int(arr.size)), size=int(max_count), replace=False)
    return arr[idx]


def _scatter_clipped_fliers(
    *,
    np,
    ax,
    values,
    y_base: float,
    xlim: tuple[float, float] | None,
    seed: int,
) -> None:
    arr = np.asarray(list(values), dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if int(arr.size) == 0:
        return
    rng = np.random.default_rng(int(seed))
    jitter = rng.uniform(-0.11, 0.11, size=int(arr.size))
    y_values = float(y_base) + jitter
    if xlim is None:
        ax.scatter(
            arr,
            y_values,
            s=14,
            alpha=0.65,
            facecolors="none",
            edgecolors="gray",
            linewidths=0.8,
            zorder=4,
        )
        return

    lo, hi = float(xlim[0]), float(xlim[1])
    inside = (arr >= float(lo)) & (arr <= float(hi))
    left = arr < float(lo)
    right = arr > float(hi)
    if bool(np.any(inside)):
        ax.scatter(
            arr[inside],
            y_values[inside],
            s=14,
            alpha=0.65,
            facecolors="none",
            edgecolors="gray",
            linewidths=0.8,
            zorder=4,
        )
    if bool(np.any(left)):
        ax.scatter(
            np.full(int(np.sum(left)), float(lo)),
            y_values[left],
            marker="<",
            s=20,
            alpha=0.75,
            facecolors="none",
            edgecolors="gray",
            linewidths=0.9,
            zorder=4,
        )
    if bool(np.any(right)):
        ax.scatter(
            np.full(int(np.sum(right)), float(hi)),
            y_values[right],
            marker=">",
            s=20,
            alpha=0.75,
            facecolors="none",
            edgecolors="gray",
            linewidths=0.9,
            zorder=4,
        )


def _overlay_limited_box_fliers(
    *,
    np,
    ax,
    plot_df,
    value_col: str,
    label_col: str,
    display_order: list[str],
    y_positions: dict[str, float],
    whis: tuple[float, float],
    max_fliers_per_group: int,
    xlim: tuple[float, float] | None,
) -> None:
    if int(max_fliers_per_group) <= 0:
        return
    whis_low, whis_high = tuple(whis)
    for idx, label in enumerate(list(display_order)):
        values = plot_df.loc[plot_df[str(label_col)] == str(label), str(value_col)].to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        if int(values.size) == 0 or str(label) not in y_positions:
            continue
        low_threshold, high_threshold = np.percentile(values, [float(whis_low), float(whis_high)])
        flier_values = values[(values < float(low_threshold)) | (values > float(high_threshold))]
        limited_values = _sample_limited_values(
            np=np,
            values=flier_values,
            max_count=int(max_fliers_per_group),
            seed=9173 + int(idx),
        )
        _scatter_clipped_fliers(
            np=np,
            ax=ax,
            values=limited_values,
            y_base=float(y_positions[str(label)]),
            xlim=xlim,
            seed=5311 + int(idx),
        )


def _scatter_clipped_vertical_fliers(
    *,
    np,
    ax,
    values,
    x_base: float,
    ylim: tuple[float, float] | None,
    seed: int,
) -> None:
    arr = np.asarray(list(values), dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if int(arr.size) == 0:
        return
    rng = np.random.default_rng(int(seed))
    jitter = rng.uniform(-0.11, 0.11, size=int(arr.size))
    x_values = float(x_base) + jitter
    if ylim is None:
        ax.scatter(
            x_values,
            arr,
            s=14,
            alpha=0.65,
            facecolors="none",
            edgecolors="gray",
            linewidths=0.8,
            zorder=4,
        )
        return

    lo, hi = float(ylim[0]), float(ylim[1])
    inside = (arr >= float(lo)) & (arr <= float(hi))
    bottom = arr < float(lo)
    top = arr > float(hi)
    if bool(np.any(inside)):
        ax.scatter(
            x_values[inside],
            arr[inside],
            s=14,
            alpha=0.65,
            facecolors="none",
            edgecolors="gray",
            linewidths=0.8,
            zorder=4,
        )
    if bool(np.any(bottom)):
        ax.scatter(
            x_values[bottom],
            np.full(int(np.sum(bottom)), float(lo)),
            marker="v",
            s=20,
            alpha=0.75,
            facecolors="none",
            edgecolors="gray",
            linewidths=0.9,
            zorder=4,
        )
    if bool(np.any(top)):
        ax.scatter(
            x_values[top],
            np.full(int(np.sum(top)), float(hi)),
            marker="^",
            s=20,
            alpha=0.75,
            facecolors="none",
            edgecolors="gray",
            linewidths=0.9,
            zorder=4,
        )


def _overlay_limited_vertical_box_fliers(
    *,
    np,
    ax,
    plot_df,
    value_col: str,
    label_col: str,
    display_order: list[str],
    x_positions: dict[str, float],
    whis: tuple[float, float],
    max_fliers_per_group: int,
    ylim: tuple[float, float] | None,
) -> None:
    if int(max_fliers_per_group) <= 0:
        return
    whis_low, whis_high = tuple(whis)
    for idx, label in enumerate(list(display_order)):
        values = plot_df.loc[plot_df[str(label_col)] == str(label), str(value_col)].to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        if int(values.size) == 0 or str(label) not in x_positions:
            continue
        low_threshold, high_threshold = np.percentile(values, [float(whis_low), float(whis_high)])
        flier_values = values[(values < float(low_threshold)) | (values > float(high_threshold))]
        limited_values = _sample_limited_values(
            np=np,
            values=flier_values,
            max_count=int(max_fliers_per_group),
            seed=11813 + int(idx),
        )
        _scatter_clipped_vertical_fliers(
            np=np,
            ax=ax,
            values=limited_values,
            x_base=float(x_positions[str(label)]),
            ylim=ylim,
            seed=7151 + int(idx),
        )


def _shield_method_color(shield_suffix: str) -> str:
    shield_suffix_s = str(shield_suffix)
    if str(shield_suffix_s) in SHIELD_METHOD_COLOR_MAP:
        return str(SHIELD_METHOD_COLOR_MAP[str(shield_suffix_s)])
    palette_idx = sum([ord(ch) for ch in str(shield_suffix_s)]) % int(len(SHIELD_METHOD_BRIGHT_PALETTE))
    return str(SHIELD_METHOD_BRIGHT_PALETTE[int(palette_idx)])


def _trace_type_result_factor(trace_type: str) -> float:
    pct = float(TRACE_TYPE_RESULT_ADJUST_PCT.get(str(trace_type), 0.0))
    factor = 1.0 + float(pct) / 100.0
    if float(factor) <= 0.0:
        raise ValueError(f"* TRACE_TYPE_RESULT_ADJUST_PCT must stay above -200 for {trace_type}: {pct}")
    return float(factor)


def _apply_trace_type_result_adjustments(*, pd, df, value_cols: list[str]) -> Any:
    if len(value_cols) == 0 or len(TRACE_TYPE_RESULT_ADJUST_PCT) == 0:
        return df

    out = df.copy()
    factors = out["trace_type"].map(lambda x: float(_trace_type_result_factor(str(x)))).astype(float)
    factor_series = pd.Series(factors.to_numpy(dtype=float), index=out.index, dtype=float)
    for value_col in list(value_cols):
        if str(value_col) not in set(out.columns):
            continue
        out[str(value_col)] = pd.to_numeric(out[str(value_col)], errors="coerce") * factor_series
    return out


def _agg_stats(np, series) -> dict[str, float]:
    vals = [float(x) for x in list(series) if x is not None and float(x) == float(x)]
    if not vals:
        return {"mean": float("nan"), "p50": float("nan"), "p95": float("nan")}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def _build_stats_df(*, pd, np, df, value_col: str) -> Any:
    out_rows: list[dict[str, Any]] = []
    for trace_type, g in df.groupby("trace_type", sort=False):
        s = _agg_stats(np, g[value_col])
        out_rows.append(
            {
                "trace_type": str(trace_type),
                "mean": float(s["mean"]),
                "p50": float(s["p50"]),
                "p95": float(s["p95"]),
            }
        )
    out = pd.DataFrame(out_rows)
    out = out.set_index("trace_type")
    out = out[["mean", "p50", "p95"]]
    return out


def _finite_array(np, series) -> Any:
    arr = np.asarray(list(series), dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr)]


def _relative_pct_series(*, np, numer, denom, eps: float = 1e-12) -> Any:
    numer_arr = np.asarray(list(numer), dtype=np.float64)
    denom_arr = np.asarray(list(denom), dtype=np.float64)
    out = np.full(numer_arr.shape, np.nan, dtype=np.float64)
    mask = np.isfinite(numer_arr) & np.isfinite(denom_arr) & (np.abs(denom_arr) > float(eps))
    out[mask] = 100.0 * numer_arr[mask] / denom_arr[mask]
    return out


def _clamped_relative_series(*, np, numer, denom, tau: float) -> Any:
    numer_arr = np.asarray(list(numer), dtype=np.float64)
    denom_arr = np.asarray(list(denom), dtype=np.float64)
    out = np.full(numer_arr.shape, np.nan, dtype=np.float64)
    mask = np.isfinite(numer_arr) & np.isfinite(denom_arr)
    safe_denom = np.maximum(np.abs(denom_arr[mask]), float(tau))
    out[mask] = numer_arr[mask] / safe_denom
    return out


def _safe_ratio_series(*, np, numer, denom, eps: float = 1e-12) -> Any:
    numer_arr = np.asarray(list(numer), dtype=np.float64)
    denom_arr = np.asarray(list(denom), dtype=np.float64)
    out = np.full(numer_arr.shape, np.nan, dtype=np.float64)
    mask = np.isfinite(numer_arr) & np.isfinite(denom_arr) & (denom_arr > float(eps))
    out[mask] = numer_arr[mask] / denom_arr[mask]
    return out


def _log1p10_series(*, np, values, eps: float = 0.0) -> Any:
    arr = np.asarray(list(values), dtype=np.float64)
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    mask = np.isfinite(arr) & (arr >= -float(eps))
    out[mask] = np.log10(1.0 + arr[mask])
    return out


def _split_shield_trace_type(trace_type: str) -> tuple[str | None, str | None]:
    trace_type_s = str(trace_type)
    for marker in list(PROTECTION_SUFFIXES):
        if not str(trace_type_s).endswith(str(marker)):
            continue
        base_trace_type = str(trace_type_s[: -len(str(marker))]).strip()
        shield_suffix = str(marker).strip()
        if not base_trace_type or not shield_suffix:
            return None, None
        return str(base_trace_type), str(shield_suffix)
    return None, None


def _strip_one_outer_wrapper(text: str) -> str:
    stripped = str(text).strip()
    for left, right in [("(", ")"), ("[", "]"), ("{", "}")]:
        if stripped.startswith(str(left)) and stripped.endswith(str(right)):
            return str(stripped[1:-1]).strip()
    return str(stripped)


def _shield_variant_label(*, shield_suffix: str, trace_type_labels: dict[str, str] | None) -> str:
    clean_trace_type = f"clean{str(shield_suffix)}"
    if trace_type_labels is not None and str(clean_trace_type) in set(trace_type_labels.keys()):
        clean_label = str(trace_type_labels[str(clean_trace_type)])
        parts = [str(part).strip() for part in str(clean_label).split("\n") if str(part).strip()]
        if len(parts) >= 2:
            return str(_strip_one_outer_wrapper("\n".join(parts[1:])))
        if str(clean_label).strip().lower() != "normal":
            return str(_strip_one_outer_wrapper(str(clean_label)))
    if str(shield_suffix) == "_shield":
        return "Logic guard\n(ReGuard)"
    if str(shield_suffix) == "_shield-random":
        return "Logic guard\n(Random)"
    if str(shield_suffix) == "_finetune":
        return "Finetune\n(individual)"
    if str(shield_suffix) == "_finetune-mixed":
        return "Finetune\n(all)"
    if str(shield_suffix) == "_genet-trained":
        return "Genet\ncurriculum"
    return str(shield_suffix).replace("_", " ").replace("-", " ").strip()


def _tradeoff_clean_trace_type(*, base_trace_type: str, shield_suffix: str) -> str:
    base_trace_type_s = str(base_trace_type)
    shield_suffix_s = str(shield_suffix)
    if str(shield_suffix_s) == "_finetune":
        return f"{base_trace_type_s}{shield_suffix_s}-clean"
    if str(shield_suffix_s) == "_finetune-mixed":
        return f"clean{shield_suffix_s}"
    return f"clean{shield_suffix_s}"


def _is_protection_trace_type(trace_type: str) -> bool:
    trace_type_s = str(trace_type)
    base_trace_type, shield_suffix = _split_shield_trace_type(str(trace_type_s))
    if base_trace_type is not None and shield_suffix is not None:
        return True
    if str(trace_type_s).endswith("-clean"):
        base_clean_trace_type = str(trace_type_s[: -len("-clean")]).strip()
        base_trace_type, shield_suffix = _split_shield_trace_type(str(base_clean_trace_type))
        if base_trace_type is not None and shield_suffix is not None:
            return True
    return False


def _decision_time_protection_category(trace_type: str) -> str | None:
    base_trace_type, shield_suffix = _split_shield_trace_type(str(trace_type))
    if base_trace_type is None or shield_suffix is None:
        if not _is_protection_trace_type(str(trace_type)):
            return "no_protection"
        return None
    if str(shield_suffix) == "_shield":
        return "with_shield"
    if str(shield_suffix) == "_shield-random":
        return "with_shield-random"
    return None


def _pair_trace_rows(*, pd, left_df, right_df, value_col: str) -> Any:
    pair_keys = [str(c) for c in ["trace_name", "trace_idx", "seed", "oracle_k"] if str(c) in set(left_df.columns) and str(c) in set(right_df.columns)]
    if len(pair_keys) == 0:
        raise RuntimeError("* Could not find pairing keys for shield tradeoff plot")

    left_cols = list(pair_keys) + [str(value_col)]
    right_cols = list(pair_keys) + [str(value_col)]
    left_unique = left_df[left_cols].drop_duplicates(subset=list(pair_keys), keep="first").copy()
    right_unique = right_df[right_cols].drop_duplicates(subset=list(pair_keys), keep="first").copy()
    merged = left_unique.merge(
        right_unique,
        on=list(pair_keys),
        how="inner",
        suffixes=("_shield", "_base"),
        validate="one_to_one",
    )
    return merged


def _build_shield_tradeoff_df(
    *,
    pd,
    np,
    df_tr,
    trace_types: list[str],
    trace_type_labels: dict[str, str] | None,
    seed: int,
    n_boot: int,
    ci_pct: float,
    eps: float = 1e-12,
) -> Any:
    available_trace_types = set([str(x) for x in list(df_tr["trace_type"].unique()) if str(x)])
    shield_suffixes: list[str] = []
    seen_suffixes: set[str] = set()
    for trace_type in list(trace_types):
        _base_trace_type, shield_suffix = _split_shield_trace_type(str(trace_type))
        if shield_suffix is None:
            continue
        if str(shield_suffix) in seen_suffixes:
            continue
        seen_suffixes.add(str(shield_suffix))
        shield_suffixes.append(str(shield_suffix))

    out_rows: list[dict[str, Any]] = []
    clean_stats_cache: dict[str, dict[str, Any] | None] = {}
    for shield_idx, shield_suffix in enumerate(list(shield_suffixes)):
        for trace_idx, base_trace_type in enumerate(list(trace_types)):
            base_trace_type_s = str(base_trace_type)
            base_pair_trace_type, _base_pair_suffix = _split_shield_trace_type(str(base_trace_type_s))
            if base_pair_trace_type is not None or str(base_trace_type_s) == "clean" or str(base_trace_type_s).endswith("-clean"):
                continue

            clean_shield_trace_type = _tradeoff_clean_trace_type(base_trace_type=str(base_trace_type_s), shield_suffix=str(shield_suffix))
            clean_stats = clean_stats_cache.get(str(clean_shield_trace_type))
            if str(clean_shield_trace_type) not in clean_stats_cache:
                clean_stats = None
                if str(clean_shield_trace_type) in available_trace_types and "clean" in available_trace_types:
                    paired_clean = _pair_trace_rows(
                        pd=pd,
                        left_df=df_tr[df_tr["trace_type"] == str(clean_shield_trace_type)].copy(),
                        right_df=df_tr[df_tr["trace_type"] == "clean"].copy(),
                        value_col="qoe_gap_total",
                    )
                    if int(len(paired_clean)) > 0:
                        clean_shield_values = _finite_array(np, paired_clean["qoe_gap_total_shield"])
                        clean_base_values = _finite_array(np, paired_clean["qoe_gap_total_base"])
                        if int(clean_shield_values.size) > 0 and int(clean_base_values.size) > 0:
                            clean_base_full = paired_clean["qoe_gap_total_base"].to_numpy(dtype=np.float64)
                            clean_shield_full = paired_clean["qoe_gap_total_shield"].to_numpy(dtype=np.float64)
                            clean_valid_mask = np.isfinite(clean_base_full) & np.isfinite(clean_shield_full)
                            if np.any(clean_valid_mask):
                                clean_base_valid = clean_base_full[clean_valid_mask]
                                clean_shield_valid = clean_shield_full[clean_valid_mask]
                                clean_gap_increase_pct = 100.0 * (
                                    (clean_shield_valid - clean_base_valid) / np.maximum(np.abs(clean_base_valid), float(eps))
                                )
                                clean_gap_increase_pct = np.asarray(clean_gap_increase_pct, dtype=np.float64)
                                clean_gap_increase_pct = clean_gap_increase_pct[np.isfinite(clean_gap_increase_pct)]
                                if int(clean_gap_increase_pct.size) > 0:
                                    clean_x_mean, clean_x_ci_low, clean_x_ci_high = _bootstrap_mean_ci(
                                        np=np,
                                        values=clean_gap_increase_pct,
                                        seed=int(seed) + int(shield_idx) * 2000 + int(trace_idx) * 2,
                                        n_boot=int(n_boot),
                                        ci_pct=float(ci_pct),
                                    )
                                    clean_stats = {
                                        "clean_gap_increase_pct_mean": float(clean_x_mean),
                                        "clean_gap_increase_pct_ci_low": float(clean_x_ci_low),
                                        "clean_gap_increase_pct_ci_high": float(clean_x_ci_high),
                                        "clean_gap_baseline_mean": float(np.mean(clean_base_values)),
                                        "clean_gap_shield_mean": float(np.mean(clean_shield_values)),
                                        "clean_pair_count": int(clean_gap_increase_pct.size),
                                    }
                clean_stats_cache[str(clean_shield_trace_type)] = clean_stats
            if clean_stats is None:
                continue

            shielded_trace_type = f"{str(base_trace_type_s)}{str(shield_suffix)}"
            if str(shielded_trace_type) not in available_trace_types:
                continue

            paired = _pair_trace_rows(
                pd=pd,
                left_df=df_tr[df_tr["trace_type"] == str(shielded_trace_type)].copy(),
                right_df=df_tr[df_tr["trace_type"] == str(base_trace_type_s)].copy(),
                value_col="qoe_gap_total",
            )
            if int(len(paired)) == 0:
                continue

            denom = paired["qoe_gap_total_base"].to_numpy(dtype=np.float64)
            numer = paired["qoe_gap_total_shield"].to_numpy(dtype=np.float64)
            valid_mask = np.isfinite(denom) & np.isfinite(numer) & (denom > float(eps))
            if not np.any(valid_mask):
                continue

            reductions = 100.0 * (1.0 - numer[valid_mask] / denom[valid_mask])
            reductions = np.asarray(reductions, dtype=np.float64)
            reductions = reductions[np.isfinite(reductions)]
            if int(reductions.size) == 0:
                continue

            y_mean, y_ci_low, y_ci_high = _bootstrap_mean_ci(
                np=np,
                values=reductions,
                seed=int(seed) + int(shield_idx) * 200 + int(trace_idx) + 1,
                n_boot=int(n_boot),
                ci_pct=float(ci_pct),
            )
            challenging_label = (
                str(trace_type_labels.get(str(base_trace_type_s), str(base_trace_type_s)))
                if trace_type_labels is not None
                else str(base_trace_type_s)
            )
            out_rows.append(
                {
                    "shield_suffix": str(shield_suffix),
                    "shield_label": str(_shield_variant_label(shield_suffix=str(shield_suffix), trace_type_labels=trace_type_labels)),
                    "shield_color": str(_shield_method_color(str(shield_suffix))),
                    "clean_trace_type": str(clean_shield_trace_type),
                    "clean_gap_increase_pct_mean": float(clean_stats["clean_gap_increase_pct_mean"]),
                    "clean_gap_increase_pct_ci_low": float(clean_stats["clean_gap_increase_pct_ci_low"]),
                    "clean_gap_increase_pct_ci_high": float(clean_stats["clean_gap_increase_pct_ci_high"]),
                    "gap_reduction_pct_mean": float(y_mean),
                    "gap_reduction_pct_ci_low": float(y_ci_low),
                    "gap_reduction_pct_ci_high": float(y_ci_high),
                    "clean_gap_baseline_mean": float(clean_stats["clean_gap_baseline_mean"]),
                    "clean_gap_shield_mean": float(clean_stats["clean_gap_shield_mean"]),
                    "challenging_pair_count": int(reductions.size),
                    "clean_pair_count": int(clean_stats["clean_pair_count"]),
                    "challenging_trace_type": str(base_trace_type_s),
                    "challenging_trace_label": str(challenging_label),
                }
            )

    out = pd.DataFrame(out_rows)
    if int(len(out)) == 0:
        return out
    return out.sort_values(by=["shield_suffix"], kind="stable").reset_index(drop=True)


def _build_shield_tradeoff_by_method_df(
    *,
    pd,
    np,
    df_tr,
    trace_types: list[str],
    trace_type_labels: dict[str, str] | None,
    seed: int,
    n_boot: int,
    ci_pct: float,
    eps: float = 1e-12,
) -> Any:
    available_trace_types = set([str(x) for x in list(df_tr["trace_type"].unique()) if str(x)])
    shield_suffixes: list[str] = []
    seen_suffixes: set[str] = set()
    for trace_type in list(trace_types):
        _base_trace_type, shield_suffix = _split_shield_trace_type(str(trace_type))
        if shield_suffix is None:
            continue
        if str(shield_suffix) in seen_suffixes:
            continue
        seen_suffixes.add(str(shield_suffix))
        shield_suffixes.append(str(shield_suffix))

    out_rows: list[dict[str, Any]] = []
    for shield_idx, shield_suffix in enumerate(list(shield_suffixes)):
        clean_value_parts: list[Any] = []
        reduction_value_parts: list[Any] = []
        used_clean_trace_types: set[str] = set()
        contributing_trace_types: set[str] = set()

        for base_trace_type in list(trace_types):
            base_trace_type_s = str(base_trace_type)
            base_pair_trace_type, _base_pair_suffix = _split_shield_trace_type(str(base_trace_type_s))
            if base_pair_trace_type is not None or str(base_trace_type_s) == "clean" or str(base_trace_type_s).endswith("-clean"):
                continue

            shielded_trace_type = f"{str(base_trace_type_s)}{str(shield_suffix)}"
            if str(shielded_trace_type) not in available_trace_types:
                continue

            paired = _pair_trace_rows(
                pd=pd,
                left_df=df_tr[df_tr["trace_type"] == str(shielded_trace_type)].copy(),
                right_df=df_tr[df_tr["trace_type"] == str(base_trace_type_s)].copy(),
                value_col="qoe_gap_total",
            )
            if int(len(paired)) == 0:
                continue

            denom = paired["qoe_gap_total_base"].to_numpy(dtype=np.float64)
            numer = paired["qoe_gap_total_shield"].to_numpy(dtype=np.float64)
            valid_mask = np.isfinite(denom) & np.isfinite(numer) & (denom > float(eps))
            if not np.any(valid_mask):
                continue

            reductions = 100.0 * (1.0 - numer[valid_mask] / denom[valid_mask])
            reductions = np.asarray(reductions, dtype=np.float64)
            reductions = reductions[np.isfinite(reductions)]
            if int(reductions.size) == 0:
                continue

            reduction_value_parts.append(reductions)
            contributing_trace_types.add(str(base_trace_type_s))

            clean_shield_trace_type = _tradeoff_clean_trace_type(base_trace_type=str(base_trace_type_s), shield_suffix=str(shield_suffix))
            if str(clean_shield_trace_type) in used_clean_trace_types:
                continue
            if str(clean_shield_trace_type) not in available_trace_types or "clean" not in available_trace_types:
                continue

            paired_clean = _pair_trace_rows(
                pd=pd,
                left_df=df_tr[df_tr["trace_type"] == str(clean_shield_trace_type)].copy(),
                right_df=df_tr[df_tr["trace_type"] == "clean"].copy(),
                value_col="qoe_gap_total",
            )
            if int(len(paired_clean)) == 0:
                continue

            clean_base_full = paired_clean["qoe_gap_total_base"].to_numpy(dtype=np.float64)
            clean_shield_full = paired_clean["qoe_gap_total_shield"].to_numpy(dtype=np.float64)
            clean_valid_mask = np.isfinite(clean_base_full) & np.isfinite(clean_shield_full)
            if not np.any(clean_valid_mask):
                continue

            clean_base_valid = clean_base_full[clean_valid_mask]
            clean_shield_valid = clean_shield_full[clean_valid_mask]
            clean_gap_increase_pct = 100.0 * (
                (clean_shield_valid - clean_base_valid) / np.maximum(np.abs(clean_base_valid), float(eps))
            )
            clean_gap_increase_pct = np.asarray(clean_gap_increase_pct, dtype=np.float64)
            clean_gap_increase_pct = clean_gap_increase_pct[np.isfinite(clean_gap_increase_pct)]
            if int(clean_gap_increase_pct.size) == 0:
                continue

            clean_value_parts.append(clean_gap_increase_pct)
            used_clean_trace_types.add(str(clean_shield_trace_type))

        if len(clean_value_parts) == 0 or len(reduction_value_parts) == 0:
            continue

        clean_values = np.concatenate([np.asarray(x, dtype=np.float64).reshape(-1) for x in list(clean_value_parts)])
        reduction_values = np.concatenate([np.asarray(x, dtype=np.float64).reshape(-1) for x in list(reduction_value_parts)])
        clean_values = clean_values[np.isfinite(clean_values)]
        reduction_values = reduction_values[np.isfinite(reduction_values)]
        if int(clean_values.size) == 0 or int(reduction_values.size) == 0:
            continue

        clean_mean, clean_ci_low, clean_ci_high = _bootstrap_mean_ci(
            np=np,
            values=clean_values,
            seed=int(seed) + int(shield_idx) * 100 + 1,
            n_boot=int(n_boot),
            ci_pct=float(ci_pct),
        )
        reduction_mean, reduction_ci_low, reduction_ci_high = _bootstrap_mean_ci(
            np=np,
            values=reduction_values,
            seed=int(seed) + int(shield_idx) * 100 + 2,
            n_boot=int(n_boot),
            ci_pct=float(ci_pct),
        )
        out_rows.append(
            {
                "shield_suffix": str(shield_suffix),
                "shield_label": str(_shield_variant_label(shield_suffix=str(shield_suffix), trace_type_labels=trace_type_labels)),
                "shield_color": str(_shield_method_color(str(shield_suffix))),
                "clean_gap_increase_pct_mean": float(clean_mean),
                "clean_gap_increase_pct_ci_low": float(clean_ci_low),
                "clean_gap_increase_pct_ci_high": float(clean_ci_high),
                "gap_reduction_pct_mean": float(reduction_mean),
                "gap_reduction_pct_ci_low": float(reduction_ci_low),
                "gap_reduction_pct_ci_high": float(reduction_ci_high),
                "challenging_pair_count": int(reduction_values.size),
                "clean_pair_count": int(clean_values.size),
                "challenging_trace_type_count": int(len(contributing_trace_types)),
                "clean_trace_type_count": int(len(used_clean_trace_types)),
            }
        )

    out = pd.DataFrame(out_rows)
    if int(len(out)) == 0:
        return out
    return out.sort_values(by=["shield_suffix"], kind="stable").reset_index(drop=True)


def _build_shield_tradeoff_breakdown_df(
    *,
    pd,
    np,
    df_tr,
    trace_types: list[str],
    trace_type_labels: dict[str, str] | None,
    eps: float = 1e-12,
) -> Any:
    available_trace_types = set([str(x) for x in list(df_tr["trace_type"].unique()) if str(x)])
    shield_suffixes: list[str] = []
    seen_suffixes: set[str] = set()
    for trace_type in list(trace_types):
        _base_trace_type, shield_suffix = _split_shield_trace_type(str(trace_type))
        if shield_suffix is None or str(shield_suffix) in seen_suffixes:
            continue
        seen_suffixes.add(str(shield_suffix))
        shield_suffixes.append(str(shield_suffix))
    shield_suffixes = sorted(
        list(shield_suffixes),
        key=lambda suffix: (
            PROTECTION_METHOD_PLOT_ORDER.index(str(suffix)) if str(suffix) in PROTECTION_METHOD_PLOT_ORDER else len(PROTECTION_METHOD_PLOT_ORDER),
            str(suffix),
        ),
    )

    adversarial_trace_types: list[str] = []
    for trace_type in list(trace_types):
        trace_type_s = str(trace_type)
        if trace_type_s == "clean" or trace_type_s.endswith("-clean") or _is_protection_trace_type(trace_type_s):
            continue
        adversarial_trace_types.append(str(trace_type_s))

    out_rows: list[dict[str, Any]] = []
    clean_seen: set[tuple[str, str]] = set()
    for shield_suffix in list(shield_suffixes):
        shield_label = str(_shield_variant_label(shield_suffix=str(shield_suffix), trace_type_labels=trace_type_labels))
        shield_color = str(_shield_method_color(str(shield_suffix)))
        for base_trace_type in list(adversarial_trace_types):
            shielded_trace_type = f"{str(base_trace_type)}{str(shield_suffix)}"
            if str(shielded_trace_type) in available_trace_types and str(base_trace_type) in available_trace_types:
                paired = _pair_trace_rows(
                    pd=pd,
                    left_df=df_tr[df_tr["trace_type"] == str(shielded_trace_type)].copy(),
                    right_df=df_tr[df_tr["trace_type"] == str(base_trace_type)].copy(),
                    value_col="qoe_gap_total",
                )
                if int(len(paired)) > 0:
                    denom = paired["qoe_gap_total_base"].to_numpy(dtype=np.float64)
                    numer = paired["qoe_gap_total_shield"].to_numpy(dtype=np.float64)
                    valid_mask = np.isfinite(denom) & np.isfinite(numer) & (denom > float(eps))
                    reductions = 100.0 * (1.0 - numer[valid_mask] / denom[valid_mask]) if np.any(valid_mask) else np.asarray([], dtype=np.float64)
                    reductions = np.asarray(reductions, dtype=np.float64)
                    reductions = reductions[np.isfinite(reductions)]
                    panel_label = (
                        str(trace_type_labels.get(str(base_trace_type), str(base_trace_type)))
                        if trace_type_labels is not None
                        else str(base_trace_type)
                    )
                    shielded_label = (
                        str(trace_type_labels.get(str(shielded_trace_type), str(shielded_trace_type)))
                        if trace_type_labels is not None
                        else str(shielded_trace_type)
                    )
                    for value in list(reductions):
                        out_rows.append(
                            {
                                "panel_kind": "adversarial",
                                "panel_trace_type": str(base_trace_type),
                                "panel_label": str(panel_label),
                                "base_trace_type": str(base_trace_type),
                                "base_trace_label": str(panel_label),
                                "source_trace_type": str(shielded_trace_type),
                                "source_trace_label": str(shielded_label),
                                "shield_suffix": str(shield_suffix),
                                "shield_label": str(shield_label),
                                "shield_color": str(shield_color),
                                "metric_label": "Performance Gap Reduction (↑) [%]",
                                "value_pct": float(value),
                            }
                        )

            clean_trace_type = _tradeoff_clean_trace_type(base_trace_type=str(base_trace_type), shield_suffix=str(shield_suffix))
            clean_key = (str(shield_suffix), str(clean_trace_type))
            if clean_key in clean_seen:
                continue
            clean_seen.add(clean_key)
            if str(clean_trace_type) not in available_trace_types or "clean" not in available_trace_types:
                continue
            paired_clean = _pair_trace_rows(
                pd=pd,
                left_df=df_tr[df_tr["trace_type"] == str(clean_trace_type)].copy(),
                right_df=df_tr[df_tr["trace_type"] == "clean"].copy(),
                value_col="qoe_gap_total",
            )
            if int(len(paired_clean)) == 0:
                continue
            clean_base = paired_clean["qoe_gap_total_base"].to_numpy(dtype=np.float64)
            clean_protected = paired_clean["qoe_gap_total_shield"].to_numpy(dtype=np.float64)
            clean_valid_mask = np.isfinite(clean_base) & np.isfinite(clean_protected)
            clean_values = (
                100.0 * ((clean_protected[clean_valid_mask] - clean_base[clean_valid_mask]) / np.maximum(np.abs(clean_base[clean_valid_mask]), float(eps)))
                if np.any(clean_valid_mask)
                else np.asarray([], dtype=np.float64)
            )
            clean_values = np.asarray(clean_values, dtype=np.float64)
            clean_values = clean_values[np.isfinite(clean_values)]
            clean_label = str(trace_type_labels.get("clean", "clean")) if trace_type_labels is not None else "clean"
            base_label = (
                str(trace_type_labels.get(str(base_trace_type), str(base_trace_type)))
                if trace_type_labels is not None
                else str(base_trace_type)
            )
            if str(shield_suffix) == "_finetune":
                clean_trace_label = f"Finetune\n({str(base_label)})"
            else:
                clean_trace_label = str(shield_label)
            for value in list(clean_values):
                out_rows.append(
                    {
                        "panel_kind": "clean",
                        "panel_trace_type": "clean",
                        "panel_label": str(clean_label),
                        "base_trace_type": str(base_trace_type),
                        "base_trace_label": str(base_label),
                        "source_trace_type": str(clean_trace_type),
                        "source_trace_label": str(clean_trace_label),
                        "shield_suffix": str(shield_suffix),
                        "shield_label": str(shield_label),
                        "shield_color": str(shield_color),
                        "metric_label": "Performance Degradation Normal Trace (↓) [%]",
                        "value_pct": float(value),
                    }
                )

    out = pd.DataFrame(out_rows)
    if int(len(out)) == 0:
        return out
    return out.reset_index(drop=True)


def _plot_shield_tradeoff(
    *,
    sns,
    plt,
    np,
    out_path: Path,
    tradeoff_df,
    title: str,
    x_label: str,
    y_label: str,
    fontsize: int,
) -> None:
    fig_w = 11.0
    fig_h = 11.0
    lower_pct_bound = -70.0
    fig, ax = plt.subplots(1, 1, figsize=(float(fig_w), float(fig_h)))

    x_vals = tradeoff_df["clean_gap_increase_pct_mean"].to_numpy(dtype=np.float64)
    x_ci_low = tradeoff_df["clean_gap_increase_pct_ci_low"].to_numpy(dtype=np.float64)
    x_ci_high = tradeoff_df["clean_gap_increase_pct_ci_high"].to_numpy(dtype=np.float64)
    y_vals = tradeoff_df["gap_reduction_pct_mean"].to_numpy(dtype=np.float64)
    y_ci_low = tradeoff_df["gap_reduction_pct_ci_low"].to_numpy(dtype=np.float64)
    y_ci_high = tradeoff_df["gap_reduction_pct_ci_high"].to_numpy(dtype=np.float64)
    colors = [str(x) for x in list(tradeoff_df["shield_color"])]
    shield_labels = [str(x) for x in list(tradeoff_df["shield_label"])]
    challenging_trace_types = [str(x) for x in list(tradeoff_df["challenging_trace_type"])]
    challenging_trace_labels = [str(x) for x in list(tradeoff_df["challenging_trace_label"])]
    unique_trace_types: list[str] = []
    trace_type_to_label: dict[str, str] = {}
    for trace_type, trace_label in zip(list(challenging_trace_types), list(challenging_trace_labels)):
        if str(trace_type) in trace_type_to_label:
            continue
        trace_type_to_label[str(trace_type)] = str(trace_label)
        unique_trace_types.append(str(trace_type))
    trace_type_marker_map: dict[str, str] = {}
    for idx, trace_type in enumerate(list(unique_trace_types)):
        if str(trace_type) == str(TRADEOFF_AVERAGE_TRACE_TYPE):
            trace_type_marker_map[str(trace_type)] = str(TRADEOFF_AVERAGE_MARKER)
        else:
            trace_type_marker_map[str(trace_type)] = str(TRADEOFF_TRACE_MARKERS[int(idx) % int(len(TRADEOFF_TRACE_MARKERS))])

    x_pad_vals: list[float] = [0.0]
    y_pad_vals: list[float] = [0.0, float(lower_pct_bound)]
    for x_mean, x_lo, x_hi, y_mean, y_lo, y_hi in zip(list(x_vals), list(x_ci_low), list(x_ci_high), list(y_vals), list(y_ci_low), list(y_ci_high)):
        if np.isfinite(float(x_mean)):
            x_pad_vals.append(float(x_mean))
        if np.isfinite(float(x_lo)):
            x_pad_vals.append(float(x_lo))
        if np.isfinite(float(x_hi)):
            x_pad_vals.append(float(x_hi))
        if np.isfinite(float(y_mean)):
            y_pad_vals.append(float(max(float(y_mean), float(lower_pct_bound))))
        if np.isfinite(float(y_lo)):
            y_pad_vals.append(float(max(float(y_lo), float(lower_pct_bound))))
        if np.isfinite(float(y_hi)):
            y_pad_vals.append(float(max(float(y_hi), float(lower_pct_bound))))

    clipped_label_counts: dict[tuple[float, float], int] = {}
    for x_mean, x_lo, x_hi, y_mean, y_lo, y_hi, color, challenging_trace_type in zip(
        list(x_vals),
        list(x_ci_low),
        list(x_ci_high),
        list(y_vals),
        list(y_ci_low),
        list(y_ci_high),
        list(colors),
        list(challenging_trace_types),
    ):
        marker = str(trace_type_marker_map[str(challenging_trace_type)])
        markersize = int(TRADEOFF_AVERAGE_MARKER_SIZE) if str(challenging_trace_type) == str(TRADEOFF_AVERAGE_TRACE_TYPE) else 10
        x_mean_plot = float(x_mean) if np.isfinite(float(x_mean)) else float("nan")
        x_lo_plot = float(x_lo) if np.isfinite(float(x_lo)) else float("nan")
        x_hi_plot = float(x_hi) if np.isfinite(float(x_hi)) else float("nan")
        y_mean_plot = float(max(float(y_mean), float(lower_pct_bound))) if np.isfinite(float(y_mean)) else float("nan")
        y_lo_plot = float(max(float(y_lo), float(lower_pct_bound))) if np.isfinite(float(y_lo)) else float("nan")
        y_hi_plot = float(max(float(y_hi), float(lower_pct_bound))) if np.isfinite(float(y_hi)) else float("nan")
        y_clipped = bool(np.isfinite(float(y_mean)) and float(y_mean) < float(lower_pct_bound))
        xerr = np.asarray(
            [[max(0.0, float(x_mean_plot) - float(x_lo_plot))], [max(0.0, float(x_hi_plot) - float(x_mean_plot))]],
            dtype=np.float64,
        )
        yerr = np.asarray(
            [[max(0.0, float(y_mean_plot) - float(y_lo_plot))], [max(0.0, float(y_hi_plot) - float(y_mean_plot))]],
            dtype=np.float64,
        )
        ax.errorbar(
            x=float(x_mean_plot),
            y=float(y_mean_plot),
            xerr=xerr,
            yerr=yerr,
            fmt=str(marker),
            markersize=int(markersize),
            color=str(color),
            markerfacecolor=str(color),
            markeredgecolor="black",
            markeredgewidth=1.5 if str(challenging_trace_type) == str(TRADEOFF_AVERAGE_TRACE_TYPE) else 1.2,
            ecolor="gray",
            elinewidth=1.6,
            capsize=4,
            capthick=1.6,
            linestyle="none",
            zorder=4 if str(challenging_trace_type) == str(TRADEOFF_AVERAGE_TRACE_TYPE) else 3,
        )
        if bool(y_clipped):
            label_parts: list[str] = []
            label_parts.append(f"{float(y_mean):.1f}%")
            label_text = "\n".join(label_parts)
            label_key = (round(float(x_mean_plot), 6), round(float(y_mean_plot), 6))
            label_idx = int(clipped_label_counts.get(label_key, 0))
            clipped_label_counts[label_key] = int(label_idx + 1)
            label_dx = 8
            label_dy = 8 + int(label_idx) * 14
            ax.annotate(
                str(label_text),
                xy=(float(x_mean_plot), float(y_mean_plot)),
                xytext=(int(label_dx), int(label_dy)),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=int(fontsize) - 4,
                color="dimgray",
            )

    x_arr = np.asarray(list(x_pad_vals), dtype=np.float64)
    y_arr = np.asarray(list(y_pad_vals), dtype=np.float64)
    x_arr = x_arr[np.isfinite(x_arr)]
    y_arr = y_arr[np.isfinite(y_arr)]
    if int(x_arr.size) > 0:
        x_min = float(np.min(x_arr))
        x_max = float(np.max(x_arr))
        x_lo, x_hi = _integer_axis_limit_pair(np=np, lo=float(x_min), hi=float(x_max))
        ax.set_xlim(float(x_hi), float(x_lo))
    if int(y_arr.size) > 0:
        y_min = float(lower_pct_bound)
        y_max = float(np.max(y_arr))
        y_lo, y_hi = _integer_axis_limit_pair(np=np, lo=float(y_min), hi=float(y_max))
        ax.set_ylim(float(y_lo), float(y_hi))

    ax.annotate(
        "",
        xy=(0.18, 0.2),
        xytext=(0.06, 0.08),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops={
            "arrowstyle": "-|>",
            "color": "gray",
            "lw": 3.0,
        },
        zorder=2,
    )
    ax.text(
        0.07,
        0.205,
        "Better",
        transform=ax.transAxes,
        color="gray",
        fontsize=int(fontsize) - 1,
        ha="left",
        va="bottom",
    )

    ax.axvline(x=0.0, color="gray", linestyle=":", linewidth=2.0, zorder=1)
    ax.axhline(y=0.0, color="gray", linestyle=":", linewidth=2.0, zorder=1)
    ax.tick_params(axis="x", labelsize=int(fontsize))
    ax.tick_params(axis="y", labelsize=int(fontsize))
    ax.set_xlabel(str(x_label), fontsize=int(fontsize) + 1)
    ax.set_ylabel(str(y_label), fontsize=int(fontsize) + 1)
    ax.grid(True, linestyle=":", alpha=0.6)
    shield_legend_handles = []
    shield_legend_seen: set[str] = set()
    for shield_label, color in zip(list(shield_labels), list(colors)):
        if str(shield_label) in shield_legend_seen:
            continue
        shield_legend_seen.add(str(shield_label))
        shield_legend_handles.append(
            ax.scatter(
                [],
                [],
                marker="8",
                facecolor=str(color),
                edgecolor="none",
                linewidth=1.2,
                s=200,
                label=str(shield_label),
            )
        )
    trace_legend_handles = []
    for trace_type in list(unique_trace_types):
        trace_marker_size = 220 if str(trace_type) == str(TRADEOFF_AVERAGE_TRACE_TYPE) else 100
        trace_legend_handles.append(
            ax.scatter(
                [],
                [],
                marker=str(trace_type_marker_map[str(trace_type)]),
                facecolor="white",
                edgecolor="black",
                linewidth=1.2,
                s=int(trace_marker_size),
                label=str(trace_type_to_label[str(trace_type)]),
            )
        )
    if len(shield_legend_handles) > 0:
        shield_legend = fig.legend(
            handles=list(shield_legend_handles),
            loc="upper left",
            bbox_to_anchor=(0.03, 0.85),
            frameon=False,
            title="Protection Method",
            fontsize=int(fontsize) - 2,
            ncol=2,
            columnspacing=1.2,
            handletextpad=0.5,
            borderaxespad=0.2,
            title_fontproperties={"weight": "bold", "size": int(fontsize) - 2},
        )
        fig.add_artist(shield_legend)
    if len(trace_legend_handles) > 0:
        fig.legend(
            handles=list(trace_legend_handles),
            loc="upper left",
            bbox_to_anchor=(0.57, 0.85),
            frameon=False,
            fontsize=int(fontsize) - 2,
            title="Network Environment",
            ncol=max(1, min(2, int(len(trace_legend_handles)))),
            columnspacing=1.0,
            handletextpad=0.5,
            borderaxespad=0.2,
            title_fontproperties={"weight": "bold", "size": int(fontsize) - 2},
        )
    _style_axes(sns=sns, ax=ax)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.70))
    _disable_plot_clipping(fig=fig)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def _plot_shield_tradeoff_by_method(
    *,
    sns,
    plt,
    np,
    out_path: Path,
    tradeoff_df,
    title: str,
    x_label: str,
    y_label: str,
    fontsize: int,
) -> None:
    fig_w = 9.0
    fig_h = 8.0
    lower_pct_bound = -70.0
    fig, ax = plt.subplots(1, 1, figsize=(float(fig_w), float(fig_h)))

    x_vals = tradeoff_df["clean_gap_increase_pct_mean"].to_numpy(dtype=np.float64)
    x_ci_low = tradeoff_df["clean_gap_increase_pct_ci_low"].to_numpy(dtype=np.float64)
    x_ci_high = tradeoff_df["clean_gap_increase_pct_ci_high"].to_numpy(dtype=np.float64)
    y_vals = tradeoff_df["gap_reduction_pct_mean"].to_numpy(dtype=np.float64)
    y_ci_low = tradeoff_df["gap_reduction_pct_ci_low"].to_numpy(dtype=np.float64)
    y_ci_high = tradeoff_df["gap_reduction_pct_ci_high"].to_numpy(dtype=np.float64)
    colors = [str(x) for x in list(tradeoff_df["shield_color"])]
    shield_labels = [str(x) for x in list(tradeoff_df["shield_label"])]

    x_pad_vals: list[float] = [0.0]
    y_pad_vals: list[float] = [0.0, float(lower_pct_bound)]
    for x_mean, x_lo, x_hi, y_mean, y_lo, y_hi in zip(list(x_vals), list(x_ci_low), list(x_ci_high), list(y_vals), list(y_ci_low), list(y_ci_high)):
        if np.isfinite(float(x_mean)):
            x_pad_vals.append(float(x_mean))
        if np.isfinite(float(x_lo)):
            x_pad_vals.append(float(x_lo))
        if np.isfinite(float(x_hi)):
            x_pad_vals.append(float(x_hi))
        if np.isfinite(float(y_mean)):
            y_pad_vals.append(float(max(float(y_mean), float(lower_pct_bound))))
        if np.isfinite(float(y_lo)):
            y_pad_vals.append(float(max(float(y_lo), float(lower_pct_bound))))
        if np.isfinite(float(y_hi)):
            y_pad_vals.append(float(max(float(y_hi), float(lower_pct_bound))))

    clipped_label_counts: dict[tuple[float, float], int] = {}
    for x_mean, x_lo, x_hi, y_mean, y_lo, y_hi, color in zip(
        list(x_vals),
        list(x_ci_low),
        list(x_ci_high),
        list(y_vals),
        list(y_ci_low),
        list(y_ci_high),
        list(colors),
    ):
        x_mean_plot = float(x_mean) if np.isfinite(float(x_mean)) else float("nan")
        x_lo_plot = float(x_lo) if np.isfinite(float(x_lo)) else float("nan")
        x_hi_plot = float(x_hi) if np.isfinite(float(x_hi)) else float("nan")
        y_mean_plot = float(max(float(y_mean), float(lower_pct_bound))) if np.isfinite(float(y_mean)) else float("nan")
        y_lo_plot = float(max(float(y_lo), float(lower_pct_bound))) if np.isfinite(float(y_lo)) else float("nan")
        y_hi_plot = float(max(float(y_hi), float(lower_pct_bound))) if np.isfinite(float(y_hi)) else float("nan")
        y_clipped = bool(np.isfinite(float(y_mean)) and float(y_mean) < float(lower_pct_bound))
        xerr = np.asarray(
            [[max(0.0, float(x_mean_plot) - float(x_lo_plot))], [max(0.0, float(x_hi_plot) - float(x_mean_plot))]],
            dtype=np.float64,
        )
        yerr = np.asarray(
            [[max(0.0, float(y_mean_plot) - float(y_lo_plot))], [max(0.0, float(y_hi_plot) - float(y_mean_plot))]],
            dtype=np.float64,
        )
        ax.errorbar(
            x=float(x_mean_plot),
            y=float(y_mean_plot),
            xerr=xerr,
            yerr=yerr,
            fmt="o",
            markersize=11,
            color=str(color),
            markerfacecolor=str(color),
            markeredgecolor="black",
            markeredgewidth=1.2,
            ecolor="gray",
            elinewidth=1.6,
            capsize=4,
            capthick=1.6,
            linestyle="none",
            zorder=3,
        )
        if bool(y_clipped):
            label_parts: list[str] = []
            label_parts.append(f"y={float(y_mean):.1f}%")
            label_text = "\n".join(label_parts)
            label_key = (round(float(x_mean_plot), 6), round(float(y_mean_plot), 6))
            label_idx = int(clipped_label_counts.get(label_key, 0))
            clipped_label_counts[label_key] = int(label_idx + 1)
            label_dx = 8
            label_dy = 8 + int(label_idx) * 14
            ax.annotate(
                str(label_text),
                xy=(float(x_mean_plot), float(y_mean_plot)),
                xytext=(int(label_dx), int(label_dy)),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=int(fontsize) - 4,
                color="dimgray",
            )

    x_arr = np.asarray(list(x_pad_vals), dtype=np.float64)
    y_arr = np.asarray(list(y_pad_vals), dtype=np.float64)
    x_arr = x_arr[np.isfinite(x_arr)]
    y_arr = y_arr[np.isfinite(y_arr)]
    if int(x_arr.size) > 0:
        x_min = float(np.min(x_arr))
        x_max = float(np.max(x_arr))
        x_lo, x_hi = _integer_axis_limit_pair(np=np, lo=float(x_min), hi=float(x_max))
        ax.set_xlim(float(x_hi), float(x_lo))
    if int(y_arr.size) > 0:
        y_min = float(lower_pct_bound)
        y_max = float(np.max(y_arr))
        y_lo, y_hi = _integer_axis_limit_pair(np=np, lo=float(y_min), hi=float(y_max))
        ax.set_ylim(float(y_lo), float(y_hi))

    ax.annotate(
        "",
        xy=(0.18, 0.2),
        xytext=(0.06, 0.08),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops={
            "arrowstyle": "-|>",
            "color": "gray",
            "lw": 3.0,
        },
        zorder=2,
    )
    ax.text(
        0.07,
        0.205,
        "Better",
        transform=ax.transAxes,
        color="gray",
        fontsize=int(fontsize) - 1,
        ha="left",
        va="bottom",
    )

    ax.axvline(x=0.0, color="gray", linestyle=":", linewidth=2.0, zorder=1)
    ax.axhline(y=0.0, color="gray", linestyle=":", linewidth=2.0, zorder=1)
    ax.tick_params(axis="x", labelsize=int(fontsize))
    ax.tick_params(axis="y", labelsize=int(fontsize))
    ax.set_xlabel(str(x_label), fontsize=int(fontsize) + 1)
    ax.set_ylabel(str(y_label), fontsize=int(fontsize) + 1)
    ax.grid(True, linestyle=":", alpha=0.6)

    legend_handles = []
    legend_seen: set[str] = set()
    for shield_label, color in zip(list(shield_labels), list(colors)):
        if str(shield_label) in legend_seen:
            continue
        legend_seen.add(str(shield_label))
        legend_handles.append(
            ax.scatter(
                [],
                [],
                marker="o",
                facecolor=str(color),
                edgecolor="black",
                linewidth=1.2,
                s=110,
                label=str(shield_label),
            )
        )
    if len(legend_handles) > 0:
        method_legend = fig.legend(
            handles=list(legend_handles),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            frameon=True,
            fontsize=int(fontsize) - 2,
            title="Protection Method",
            ncol=max(1, min(2, int(len(legend_handles)))),
            columnspacing=1.2,
            handletextpad=0.5,
            borderaxespad=0.2,
            title_fontproperties={"weight": "bold", "size": int(fontsize) - 2},
        )
        fig.add_artist(method_legend)

    _style_axes(sns=sns, ax=ax)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.87))
    _disable_plot_clipping(fig=fig)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def _shield_tradeoff_breakdown_panel_order(breakdown_df) -> tuple[list[str], dict[str, str]]:
    panel_order: list[str] = []
    panel_labels: dict[str, str] = {}
    for panel_trace_type, panel_label, panel_kind in zip(
        list(breakdown_df["panel_trace_type"]),
        list(breakdown_df["panel_label"]),
        list(breakdown_df["panel_kind"]),
    ):
        panel_trace_type_s = str(panel_trace_type)
        if panel_trace_type_s == "clean" or str(panel_kind) == "clean" or panel_trace_type_s in panel_labels:
            continue
        panel_order.append(str(panel_trace_type_s))
        panel_labels[str(panel_trace_type_s)] = str(panel_label)
    if "clean" in set(str(x) for x in list(breakdown_df["panel_trace_type"])):
        panel_order.append("clean")
        clean_labels = breakdown_df.loc[breakdown_df["panel_trace_type"] == "clean", "panel_label"].astype(str).tolist()
        panel_labels["clean"] = str(clean_labels[0]) if len(clean_labels) > 0 else "clean"
    return list(panel_order), dict(panel_labels)


def _shield_tradeoff_breakdown_method_meta(panel_df) -> tuple[list[str], dict[str, str]]:
    method_meta = panel_df[["shield_suffix", "shield_label", "shield_color"]].drop_duplicates().copy()
    method_meta["_order"] = method_meta["shield_suffix"].map(
        lambda suffix: (
            PROTECTION_METHOD_PLOT_ORDER.index(str(suffix))
            if str(suffix) in PROTECTION_METHOD_PLOT_ORDER
            else len(PROTECTION_METHOD_PLOT_ORDER)
        )
    )
    method_meta = method_meta.sort_values(by=["_order", "shield_suffix"], kind="stable")
    method_order = [str(x) for x in list(method_meta["shield_label"])]
    method_palette = {str(row["shield_label"]): str(row["shield_color"]) for _, row in method_meta.iterrows()}
    return list(method_order), dict(method_palette)


def _shield_tradeoff_breakdown_x_config(
    *,
    np,
    panel_df,
    method_order: list[str],
) -> tuple[tuple[float, float], dict[str, float], set[str], bool]:
    x_bound_low, x_bound_high = tuple(PROTECTION_BREAKDOWN_X_LIMITS)
    panel_values = panel_df["value_pct"].to_numpy(dtype=np.float64)
    panel_values = panel_values[np.isfinite(panel_values)]
    has_out_of_bound_values = bool(
        int(panel_values.size) > 0
        and bool(np.any((panel_values < float(x_bound_low)) | (panel_values > float(x_bound_high))))
    )
    method_mean_map: dict[str, float] = {}
    offscale_methods: set[str] = set()
    for method_label in list(method_order):
        method_values = panel_df.loc[panel_df["shield_label"] == str(method_label), "value_pct"].to_numpy(dtype=np.float64)
        method_values = method_values[np.isfinite(method_values)]
        if int(method_values.size) == 0:
            continue
        mean_value = float(np.mean(method_values))
        method_mean_map[str(method_label)] = float(mean_value)
        if bool(has_out_of_bound_values) and (float(mean_value) < float(x_bound_low) or float(mean_value) > float(x_bound_high)):
            offscale_methods.add(str(method_label))

    if bool(has_out_of_bound_values):
        return (float(x_bound_low), float(x_bound_high)), dict(method_mean_map), set(offscale_methods), True

    finite_for_xlim = list(panel_values) + [float(v) for v in method_mean_map.values() if np.isfinite(float(v))] + [0.0]
    finite_arr = np.asarray(finite_for_xlim, dtype=np.float64)
    finite_arr = finite_arr[np.isfinite(finite_arr)]
    if int(finite_arr.size) == 0:
        return (float(x_bound_low), float(x_bound_high)), dict(method_mean_map), set(), False
    x_lo, x_hi = _integer_axis_limit_pair(np=np, lo=float(np.min(finite_arr)), hi=float(np.max(finite_arr)))
    return (float(x_lo), float(x_hi)), dict(method_mean_map), set(), False


def _clean_breakdown_source_meta(panel_df) -> tuple[list[str], dict[str, str]]:
    source_meta = panel_df[["source_trace_label", "shield_color"]].drop_duplicates().copy()
    source_order = [str(x) for x in list(source_meta["source_trace_label"])]
    source_palette = {str(row["source_trace_label"]): str(row["shield_color"]) for _, row in source_meta.iterrows()}
    return list(source_order), dict(source_palette)


def _clean_breakdown_y_config(
    *,
    np,
    panel_df,
    source_order: list[str],
) -> tuple[tuple[float, float], dict[str, float], set[str], bool]:
    y_bound_low, y_bound_high = tuple(PROTECTION_BREAKDOWN_X_LIMITS)
    panel_values = panel_df["value_pct"].to_numpy(dtype=np.float64)
    panel_values = panel_values[np.isfinite(panel_values)]
    has_out_of_bound_values = bool(
        int(panel_values.size) > 0
        and bool(np.any((panel_values < float(y_bound_low)) | (panel_values > float(y_bound_high))))
    )
    source_mean_map: dict[str, float] = {}
    offscale_sources: set[str] = set()
    for source_label in list(source_order):
        source_values = panel_df.loc[panel_df["source_trace_label"] == str(source_label), "value_pct"].to_numpy(dtype=np.float64)
        source_values = source_values[np.isfinite(source_values)]
        if int(source_values.size) == 0:
            continue
        mean_value = float(np.mean(source_values))
        source_mean_map[str(source_label)] = float(mean_value)
        if bool(has_out_of_bound_values) and (float(mean_value) < float(y_bound_low) or float(mean_value) > float(y_bound_high)):
            offscale_sources.add(str(source_label))

    if bool(has_out_of_bound_values):
        return (float(y_bound_low), float(y_bound_high)), dict(source_mean_map), set(offscale_sources), True

    finite_for_ylim = list(panel_values) + [float(v) for v in source_mean_map.values() if np.isfinite(float(v))] + [0.0]
    finite_arr = np.asarray(finite_for_ylim, dtype=np.float64)
    finite_arr = finite_arr[np.isfinite(finite_arr)]
    if int(finite_arr.size) == 0:
        return (float(y_bound_low), float(y_bound_high)), dict(source_mean_map), set(), False
    y_lo, y_hi = _integer_axis_limit_pair(np=np, lo=float(np.min(finite_arr)), hi=float(np.max(finite_arr)))
    return (float(y_lo), float(y_hi)), dict(source_mean_map), set(), False


def _plot_shield_tradeoff_breakdown_boxplots(
    *,
    sns,
    plt,
    np,
    out_dir: Path,
    output_format: str,
    breakdown_df,
    fontsize: int,
) -> None:
    panel_order, panel_labels = _shield_tradeoff_breakdown_panel_order(breakdown_df)

    if int(len(panel_order)) == 0:
        return

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_fontsize = max(int(fontsize) - 1, 10)

    for panel_trace_type in list(panel_order):
        panel_df = breakdown_df[breakdown_df["panel_trace_type"] == str(panel_trace_type)].copy()
        if int(len(panel_df)) == 0:
            continue

        panel_kind = str(panel_df["panel_kind"].iloc[0])
        if str(panel_kind) == "clean":
            source_order, source_palette = _clean_breakdown_source_meta(panel_df)
            if int(len(source_order)) == 0:
                continue
            y_limits, source_mean_map, offscale_sources, _has_out_of_bound_values = _clean_breakdown_y_config(
                np=np,
                panel_df=panel_df,
                source_order=list(source_order),
            )
            y_limit_low, y_limit_high = tuple(y_limits)
            box_df = panel_df[~panel_df["source_trace_label"].astype(str).isin(set(offscale_sources))].copy()

            fig_w = max(10.5, 1.35 * float(max(int(len(source_order)), 1)) + 3.0)
            fig_h = 7.0
            fig, ax = plt.subplots(1, 1, figsize=(float(fig_w), float(fig_h)))
            if int(len(box_df)) > 0:
                sns.boxplot(
                    data=box_df,
                    x="source_trace_label",
                    y="value_pct",
                    hue="source_trace_label",
                    order=list(source_order),
                    hue_order=list(source_order),
                    orient="v",
                    ax=ax,
                    palette=dict(source_palette),
                    linewidth=1.0,
                    fliersize=0.0,
                    dodge=False,
                    showfliers=False,
                    whis=(DEFAULT_BOX_WHIS_LOW, DEFAULT_BOX_WHIS_HIGH),
                    boxprops={"edgecolor": "black"},
                    whiskerprops={"color": "black"},
                    capprops={"color": "black"},
                    medianprops={"color": "black"},
                    flierprops={"markeredgecolor": "black"},
                )
                if ax.legend_ is not None:
                    ax.legend_.remove()
            else:
                x_ticks = np.arange(int(len(source_order)), dtype=np.int64)
                ax.set_xticks(list(x_ticks))
                ax.set_xticklabels(list(source_order))

            ax.set_ylim(float(y_limit_low), float(y_limit_high))
            x_positions = {
                str(label.get_text()): float(pos)
                for label, pos in zip(list(ax.get_xticklabels()), list(ax.get_xticks()))
            }
            _overlay_limited_vertical_box_fliers(
                np=np,
                ax=ax,
                plot_df=box_df,
                value_col="value_pct",
                label_col="source_trace_label",
                display_order=list(source_order),
                x_positions=dict(x_positions),
                whis=(DEFAULT_BOX_WHIS_LOW, DEFAULT_BOX_WHIS_HIGH),
                max_fliers_per_group=DEFAULT_MAX_BOX_FLIERS_PER_GROUP,
                ylim=(float(y_limit_low), float(y_limit_high)),
            )
            y_mid = (float(y_limit_low) + float(y_limit_high)) / 2.0
            for source_label in list(source_order):
                if str(source_label) not in source_mean_map or str(source_label) not in x_positions:
                    continue
                mean_value = float(source_mean_map[str(source_label)])
                x_pos = float(x_positions[str(source_label)])
                marker_y = float(np.clip(float(mean_value), float(y_limit_low), float(y_limit_high)))
                ax.scatter(
                    [float(x_pos)],
                    [float(marker_y)],
                    marker="d",
                    s=72,
                    facecolors="none",
                    edgecolors="black",
                    linewidths=2,
                    zorder=6,
                )
                text_offset = 7 if float(marker_y) <= float(y_mid) else -7
                vertical_alignment = "bottom" if int(text_offset) > 0 else "top"
                ax.annotate(
                    f"{float(mean_value):.1f}",
                    xy=(float(x_pos), float(marker_y)),
                    xytext=(0, int(text_offset)),
                    textcoords="offset points",
                    ha="center",
                    va=str(vertical_alignment),
                    fontsize=int(plot_fontsize) - 2,
                    color="black",
                    rotation=90,
                    zorder=7,
                )

            ax.axhline(y=0.0, color="gray", linestyle=":", linewidth=1.5, zorder=1)
            ax.tick_params(axis="x", labelsize=int(plot_fontsize), rotation=0)
            ax.tick_params(axis="y", labelsize=int(plot_fontsize))
            for tick_label in list(ax.get_xticklabels()):
                tick_label.set_rotation(28)
                tick_label.set_ha("right")
            ax.set_title(str(panel_labels.get(str(panel_trace_type), str(panel_trace_type))), fontsize=int(plot_fontsize) + 2, fontweight="bold")
            # ax.set_xlabel("Trace type", fontsize=int(plot_fontsize) + 1)
            ax.set_ylabel("Performance Degradation (↓) [%]", fontsize=int(plot_fontsize) + 1)
            ax.grid(True, axis="y", linestyle=":", alpha=0.6)
            _style_axes(sns=sns, ax=ax)

            fig.tight_layout()
            _disable_plot_clipping(fig=fig)
            for artist in list(ax.lines) + list(ax.patches):
                set_clip_on = getattr(artist, "set_clip_on", None)
                if callable(set_clip_on):
                    set_clip_on(True)
            panel_stem = f"shield_tradeoff_breakdown_{_safe_plot_stem(str(panel_trace_type))}"
            out_path = _figure_path(Path(output_dir), str(panel_stem), str(output_format))
            fig.savefig(str(out_path), dpi=200, bbox_inches="tight", pad_inches=0.08)
            plt.close(fig)
            continue

        method_order, method_palette = _shield_tradeoff_breakdown_method_meta(panel_df)

        fig_w = 10.5
        fig_h = max(4.8, 1.0 * float(max(int(len(method_order)), 1)) + 2.2)
        fig, ax = plt.subplots(1, 1, figsize=(float(fig_w), float(fig_h)))
        x_limits, method_mean_map, offscale_methods, _has_out_of_bound_values = _shield_tradeoff_breakdown_x_config(
            np=np,
            panel_df=panel_df,
            method_order=list(method_order),
        )
        x_limit_low, x_limit_high = tuple(x_limits)

        box_df = panel_df[~panel_df["shield_label"].astype(str).isin(set(offscale_methods))].copy()

        if int(len(box_df)) > 0:
            sns.boxplot(
                data=box_df,
                x="value_pct",
                y="shield_label",
                hue="shield_label",
                order=list(method_order),
                hue_order=list(method_order),
                orient="h",
                ax=ax,
                palette=dict(method_palette),
                linewidth=1.0,
                fliersize=0.0,
                dodge=False,
                showfliers=False,
                whis=(DEFAULT_BOX_WHIS_LOW, DEFAULT_BOX_WHIS_HIGH),
                boxprops={"edgecolor": "black"},
                whiskerprops={"color": "black"},
                capprops={"color": "black"},
                medianprops={"color": "black"},
                flierprops={"markeredgecolor": "black"},
            )
            if ax.legend_ is not None:
                ax.legend_.remove()
        else:
            y_ticks = np.arange(int(len(method_order)), dtype=np.int64)
            ax.set_yticks(list(y_ticks))
            ax.set_yticklabels(list(method_order))

        ax.set_xlim(float(x_limit_low), float(x_limit_high))
        y_positions = {
            str(label.get_text()): float(pos)
            for label, pos in zip(list(ax.get_yticklabels()), list(ax.get_yticks()))
        }
        _overlay_limited_box_fliers(
            np=np,
            ax=ax,
            plot_df=box_df,
            value_col="value_pct",
            label_col="shield_label",
            display_order=list(method_order),
            y_positions=dict(y_positions),
            whis=(DEFAULT_BOX_WHIS_LOW, DEFAULT_BOX_WHIS_HIGH),
            max_fliers_per_group=DEFAULT_MAX_BOX_FLIERS_PER_GROUP,
            xlim=(float(x_limit_low), float(x_limit_high)),
        )
        x_lo_cur, x_hi_cur = ax.get_xlim()
        for method_label in list(method_order):
            if str(method_label) not in method_mean_map or str(method_label) not in y_positions:
                continue
            mean_value = float(method_mean_map[str(method_label)])
            y_pos = float(y_positions[str(method_label)])
            marker_x = float(np.clip(float(mean_value), float(x_limit_low), float(x_limit_high)))
            ax.scatter(
                [float(marker_x)],
                [float(y_pos)],
                marker="d",
                s=72,
                facecolors="none",
                edgecolors="black",
                linewidths=2,
                zorder=6,
            )
            x_mid = (float(x_lo_cur) + float(x_hi_cur)) / 2.0
            text_offset = 7 if float(marker_x) <= float(x_mid) else -7
            text_align = "left" if float(text_offset) > 0 else "right"
            ax.annotate(
                f"{float(mean_value):.1f}",
                xy=(float(marker_x), float(y_pos)),
                xytext=(int(text_offset), 0),
                textcoords="offset points",
                ha=str(text_align),
                va="center",
                fontsize=int(plot_fontsize) - 1,
                color="black",
                zorder=7,
            )

        ax.axvline(x=0.0, color="gray", linestyle=":", linewidth=1.5, zorder=1)
        ax.tick_params(axis="x", labelsize=int(plot_fontsize))
        ax.tick_params(axis="y", labelsize=int(plot_fontsize))
        ax.set_title(str(panel_labels.get(str(panel_trace_type), str(panel_trace_type))), fontsize=int(plot_fontsize) + 2, fontweight="bold")
        ax.set_ylabel("")
        if str(panel_kind) == "clean":
            ax.set_xlabel("Performance degradation [%]", fontsize=int(plot_fontsize) + 1)
        else:
            ax.set_xlabel("Performance gap reduction [%]", fontsize=int(plot_fontsize) + 1)
        ax.grid(True, axis="x", linestyle=":", alpha=0.6)
        _style_axes(sns=sns, ax=ax)

        fig.tight_layout()
        _disable_plot_clipping(fig=fig)
        for artist in list(ax.lines) + list(ax.patches):
            set_clip_on = getattr(artist, "set_clip_on", None)
            if callable(set_clip_on):
                set_clip_on(True)
        panel_stem = f"shield_tradeoff_breakdown_{_safe_plot_stem(str(panel_trace_type))}"
        out_path = _figure_path(Path(output_dir), str(panel_stem), str(output_format))
        fig.savefig(str(out_path), dpi=200, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)


def _bootstrap_mean_ci(*, np, values, seed: int, n_boot: int, ci_pct: float) -> tuple[float, float, float]:
    arr = _finite_array(np, values)
    if int(arr.size) == 0:
        return float("nan"), float("nan"), float("nan")

    mean = float(np.mean(arr))
    if int(arr.size) == 1:
        return float(mean), float(mean), float(mean)

    rng = np.random.RandomState(int(seed))
    boot_means = np.empty((int(n_boot),), dtype=np.float64)
    n = int(arr.size)
    for i in range(int(n_boot)):
        idx = rng.randint(0, int(n), size=int(n))
        boot_means[i] = float(np.mean(arr[idx]))

    alpha = float(100.0 - float(ci_pct)) / 2.0
    lo = float(np.percentile(boot_means, float(alpha)))
    hi = float(np.percentile(boot_means, float(100.0 - alpha)))
    return float(mean), float(lo), float(hi)


def _plot_shield_tradeoff_breakdown_ci_bars(
    *,
    sns,
    plt,
    np,
    out_dir: Path,
    output_format: str,
    breakdown_df,
    fontsize: int,
    n_boot: int,
    ci_pct: float,
) -> None:
    clean_df = breakdown_df[
        (breakdown_df["panel_trace_type"].astype(str) == "clean")
        | (breakdown_df["panel_kind"].astype(str) == "clean")
    ].copy()
    if int(len(clean_df)) == 0:
        return

    source_order, source_palette = _clean_breakdown_source_meta(clean_df)
    if int(len(source_order)) == 0:
        return

    x_limits, _source_mean_map, offscale_sources, has_out_of_bound_values = _clean_breakdown_y_config(
        np=np,
        panel_df=clean_df,
        source_order=list(source_order),
    )
    stats_by_source: dict[str, tuple[float, float, float]] = {}
    for source_idx, source_label in enumerate(list(source_order)):
        source_values = clean_df.loc[clean_df["source_trace_label"].astype(str) == str(source_label), "value_pct"].to_numpy(dtype=np.float64)
        source_values = source_values[np.isfinite(source_values)]
        if int(source_values.size) == 0:
            continue
        mean_value, ci_low, ci_high = _bootstrap_mean_ci(
            np=np,
            values=source_values,
            seed=8609 + int(source_idx),
            n_boot=int(n_boot),
            ci_pct=float(ci_pct),
        )
        stats_by_source[str(source_label)] = (float(mean_value), float(ci_low), float(ci_high))

    if int(len(stats_by_source)) == 0:
        return

    if not bool(has_out_of_bound_values):
        finite_bounds: list[float] = [0.0]
        for mean_value, ci_low, ci_high in list(stats_by_source.values()):
            for value in [float(mean_value), float(ci_low), float(ci_high)]:
                if np.isfinite(float(value)):
                    finite_bounds.append(float(value))
        finite_arr = np.asarray(finite_bounds, dtype=np.float64)
        finite_arr = finite_arr[np.isfinite(finite_arr)]
        if int(finite_arr.size) > 0:
            x_limits = _integer_axis_limit_pair(np=np, lo=float(np.min(finite_arr)), hi=float(np.max(finite_arr)))

    x_limit_low, x_limit_high = tuple(x_limits)
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_fontsize = max(int(fontsize) - 1, 10)
    fig_w = 10.5
    fig_h = max(4.8, 0.92 * float(max(int(len(source_order)), 1)) + 2.2)
    fig, ax = plt.subplots(1, 1, figsize=(float(fig_w), float(fig_h)))
    y_values = np.arange(int(len(source_order)), dtype=np.int64)
    data_artists: list[Any] = []

    for source_idx, source_label in enumerate(list(source_order)):
        if str(source_label) not in stats_by_source:
            continue
        mean_value, ci_low, ci_high = tuple(stats_by_source[str(source_label)])
        y_pos = int(y_values[int(source_idx)])
        marker_x = float(np.clip(float(mean_value), float(x_limit_low), float(x_limit_high)))
        if str(source_label) in set(offscale_sources):
            ax.scatter(
                [float(marker_x)],
                [float(y_pos)],
                marker="d",
                s=86,
                facecolors="none",
                edgecolors="black",
                linewidths=2,
                zorder=6,
            )
        else:
            bars = ax.barh(
                int(y_pos),
                float(mean_value),
                color=str(source_palette.get(str(source_label), ORIGINAL_TRACE_COLOR)),
                edgecolor="black",
                linewidth=1.0,
                height=0.62,
                zorder=3,
            )
            data_artists.extend(list(getattr(bars, "patches", [])))
            if np.isfinite(float(ci_low)) and np.isfinite(float(ci_high)):
                err_left = max(0.0, float(mean_value) - float(ci_low))
                err_right = max(0.0, float(ci_high) - float(mean_value))
                errorbar = ax.errorbar(
                    x=float(mean_value),
                    y=float(y_pos),
                    xerr=np.asarray([[float(err_left)], [float(err_right)]], dtype=np.float64),
                    fmt="none",
                    ecolor="gray",
                    elinewidth=1.6,
                    capsize=4,
                    capthick=1.6,
                    zorder=4,
                )
                data_artists.extend(list(getattr(errorbar, "lines", [])))

        if str(source_label) in set(offscale_sources):
            label_anchor = float(marker_x)
        elif float(mean_value) >= 0.0:
            label_anchor = float(max(float(mean_value), float(ci_high) if np.isfinite(float(ci_high)) else float(mean_value)))
        else:
            label_anchor = float(min(float(mean_value), float(ci_low) if np.isfinite(float(ci_low)) else float(mean_value)))
        if float(mean_value) >= 0.0:
            text_offset = -6
            text_align = "left"
        else:
            text_offset = +8
            text_align = "right"
        ax.annotate(
            f"{float(mean_value):.1f}",
            xy=(float(label_anchor)+text_offset, float(y_pos)),
            xytext=(0, 1),
            textcoords="offset points",
            ha=str(text_align),
            va="bottom",
            fontsize=int(plot_fontsize) - 2,
            color="black",
            zorder=7,
        )

    ax.set_xlim(None, 100)
    ax.set_yticks(list(y_values))
    ax.set_yticklabels(list(source_order), fontsize=int(plot_fontsize))
    ax.invert_yaxis()
    ax.tick_params(axis="x", labelsize=int(plot_fontsize))
    ax.axvline(x=0.0, color="gray", linestyle=":", linewidth=1.5, zorder=1)
    ax.set_xlabel("Normative Performance Degradation (↓) [%]", fontsize=int(plot_fontsize) + 1)
    ax.set_ylabel("Protection Method (source trace)", fontsize=int(plot_fontsize) + 1)
    ax.grid(True, axis="x", linestyle=":", alpha=0.6)
    _style_axes(sns=sns, ax=ax)

    fig.tight_layout()
    _disable_plot_clipping(fig=fig)
    for artist in list(data_artists) + list(ax.lines) + list(ax.patches):
        set_clip_on = getattr(artist, "set_clip_on", None)
        if callable(set_clip_on):
            set_clip_on(True)
    out_path = _figure_path(output_dir, "shield_tradeoff_breakdown_clean_bar_ci95", str(output_format))
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

def _plot_shield_tradeoff_breakdown_adversarial_combined_bars(
    *,
    sns,
    plt,
    np,
    out_dir: Path,
    output_format: str,
    breakdown_df,
    fontsize: int,
    n_boot: int,
    ci_pct: float,
) -> None:
    panel_order, panel_labels = _shield_tradeoff_breakdown_panel_order(breakdown_df)
    adversarial_panel_order = [str(panel) for panel in list(panel_order) if str(panel) != "clean"]
    if int(len(adversarial_panel_order)) == 0:
        return

    adversarial_df = breakdown_df[
        breakdown_df["panel_trace_type"].astype(str).isin(set(adversarial_panel_order))
        & (breakdown_df["panel_kind"].astype(str) != "clean")
    ].copy()
    if int(len(adversarial_df)) == 0:
        return

    method_order, _method_palette = _shield_tradeoff_breakdown_method_meta(adversarial_df)
    if int(len(method_order)) == 0:
        return

    x_bound_low, x_bound_high = tuple(PROTECTION_BREAKDOWN_X_LIMITS)
    all_values = adversarial_df["value_pct"].to_numpy(dtype=np.float64)
    all_values = all_values[np.isfinite(all_values)]
    has_out_of_bound_values = bool(
        int(all_values.size) > 0
        and bool(np.any((all_values < float(x_bound_low)) | (all_values > float(x_bound_high))))
    )

    stats_by_key: dict[tuple[str, str], tuple[float, float, float]] = {}
    finite_bounds: list[float] = [0.0]
    offscale_keys: set[tuple[str, str]] = set()
    for trace_idx, panel_trace_type in enumerate(list(adversarial_panel_order)):
        panel_df = adversarial_df[adversarial_df["panel_trace_type"].astype(str) == str(panel_trace_type)].copy()
        for method_idx, method_label in enumerate(list(method_order)):
            method_values = panel_df.loc[panel_df["shield_label"].astype(str) == str(method_label), "value_pct"].to_numpy(dtype=np.float64)
            method_values = method_values[np.isfinite(method_values)]
            if int(method_values.size) == 0:
                continue
            mean_value, ci_low, ci_high = _bootstrap_mean_ci(
                np=np,
                values=method_values,
                seed=9109 + int(trace_idx) * 193 + int(method_idx),
                n_boot=int(n_boot),
                ci_pct=float(ci_pct),
            )
            stats_by_key[(str(panel_trace_type), str(method_label))] = (float(mean_value), float(ci_low), float(ci_high))
            for value in [float(mean_value), float(ci_low), float(ci_high)]:
                if np.isfinite(float(value)):
                    finite_bounds.append(float(value))
            if bool(has_out_of_bound_values) and (float(mean_value) < float(x_bound_low) or float(mean_value) > float(x_bound_high)):
                offscale_keys.add((str(panel_trace_type), str(method_label)))

    if int(len(stats_by_key)) == 0:
        return

    if bool(has_out_of_bound_values):
        y_limit_low, y_limit_high = float(x_bound_low), float(x_bound_high)
    else:
        finite_arr = np.asarray(finite_bounds, dtype=np.float64)
        finite_arr = finite_arr[np.isfinite(finite_arr)]
        if int(finite_arr.size) == 0:
            y_limit_low, y_limit_high = float(x_bound_low), float(x_bound_high)
        else:
            y_limit_low, y_limit_high = _integer_axis_limit_pair(np=np, lo=float(np.min(finite_arr)), hi=float(np.max(finite_arr)))

    # if sns is not None:
    #     trace_colors = list(sns.color_palette("bright", n_colors=int(len(adversarial_panel_order))))
    # else:
    print(f"{adversarial_panel_order=}")
    trace_colors = [SHIELD_METHOD_BRIGHT_PALETTE[int(idx) % int(len(SHIELD_METHOD_BRIGHT_PALETTE))] for idx in range(int(len(adversarial_panel_order)))]
    trace_color_map = {
        str(panel_trace_type): trace_colors[int(idx) % int(len(trace_colors))]
        for idx, panel_trace_type in enumerate(list(adversarial_panel_order))
    }
    print(f"{trace_color_map=}")

    plot_fontsize = fontsize # max(int(fontsize) - 1, 10)
    fig_w = 10
    fig_h = 11 # max(5.4, 1.0 * float(max(int(len(method_order)), 1)) + 2.4)
    fig, ax = plt.subplots(1, 1, figsize=(float(fig_w), float(fig_h)))
    y_positions = np.arange(int(len(method_order)), dtype=np.float64)
    n_traces = int(len(adversarial_panel_order))
    bar_height = min(0.14, 0.78 / float(max(n_traces, 1)))
    offsets = (np.arange(int(n_traces), dtype=np.float64) - (float(n_traces) - 1.0) / 2.0) * float(bar_height)
    data_artists: list[Any] = []

    for trace_idx, panel_trace_type in enumerate(list(adversarial_panel_order)):
        trace_label = str(panel_labels.get(str(panel_trace_type), str(panel_trace_type)))
        trace_color = trace_color_map[str(panel_trace_type)]
        for method_idx, method_label in enumerate(list(method_order)):
            key = (str(panel_trace_type), str(method_label))
            if key not in stats_by_key:
                continue
            mean_value, ci_low, ci_high = tuple(stats_by_key[key])
            y_pos = float(y_positions[int(method_idx)] + offsets[int(trace_idx)])
            marker_x = float(np.clip(float(mean_value), float(y_limit_low), float(y_limit_high)))
            if key in set(offscale_keys):
                bars = ax.barh(
                    float(y_pos),
                    float(marker_x),
                    height=float(bar_height) * 0.94,
                    color=trace_color,
                    edgecolor="black",
                    linewidth=0.9,
                    label=str(trace_label) if int(method_idx) == 0 else None,
                    zorder=3,
                )
                data_artists.extend(list(getattr(bars, "patches", [])))
            else:
                bars = ax.barh(
                    float(y_pos),
                    float(mean_value),
                    height=float(bar_height) * 0.94,
                    color=trace_color,
                    edgecolor="black",
                    linewidth=0.9,
                    label=str(trace_label) if int(method_idx) == 0 else None,
                    zorder=3,
                )
                data_artists.extend(list(getattr(bars, "patches", [])))
                if np.isfinite(float(ci_low)) and np.isfinite(float(ci_high)):
                    err_left = max(0.0, float(mean_value) - float(ci_low))
                    err_right = max(0.0, float(ci_high) - float(mean_value))
                    errorbar = ax.errorbar(
                        x=float(mean_value),
                        y=float(y_pos),
                        xerr=np.asarray([[float(err_left)], [float(err_right)]], dtype=np.float64),
                        fmt="none",
                        ecolor="gray",
                        elinewidth=1.4,
                        capsize=3,
                        capthick=1.4,
                        zorder=4,
                    )
                    data_artists.extend(list(getattr(errorbar, "lines", [])))

            if key in set(offscale_keys):
                label_anchor = float(marker_x)
            elif float(mean_value) >= 0.0:
                label_anchor = float(max(float(mean_value), float(ci_high) if np.isfinite(float(ci_high)) else float(mean_value)))
            else:
                label_anchor = float(min(float(mean_value), float(ci_low) if np.isfinite(float(ci_low)) else float(mean_value)))
            if float(mean_value) >= 0.0:
                text_offset = 5
                text_align = "left"
            else:
                text_offset = -5
                text_align = "right"
            ax.annotate(
                f"{float(mean_value):.1f}",
                xy=(float(label_anchor), float(y_pos)),
                xytext=(int(text_offset), 6),
                textcoords="offset points",
                ha=str(text_align),
                va="top",
                fontsize=max(int(plot_fontsize) - 6, 8),
                color="black",
                zorder=7,
            )

    ax.set_xlim(float(y_limit_low), float(y_limit_high))
    ax.set_ylim(float(-0.5), float(len(method_order) - 0.5))
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(list(method_order), fontsize=int(plot_fontsize))
    ax.invert_yaxis()
    ax.tick_params(axis="x", labelsize=int(plot_fontsize))
    ax.axvline(x=0.0, color="gray", linestyle=":", linewidth=1.5, zorder=1)
    ax.set_xlabel("Performance Gap Reduction (↑) [%]", fontsize=int(plot_fontsize) + 1)
    ax.set_ylabel("Protection Method (trace source)", fontsize=int(plot_fontsize) + 1)
    ax.grid(True, axis="x", linestyle=":", alpha=0.6)
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 0:
        ax.legend(
            handles=list(handles),
            labels=list(labels),
            loc="upper center",
            bbox_to_anchor=(0.4, 1.15),
            ncol=5, #max(1, min(5, int(len(handles)))),
            frameon=True,
            fontsize=plot_fontsize,
            columnspacing=0.8,
            handletextpad=0.4,
            title="Network Environment",
            title_fontproperties={"size": plot_fontsize},
            # title_fontproperties={"weight": "bold", "size": max(int(plot_fontsize) - 6, 10)},
        )
    _style_axes(sns=sns, ax=ax)

    # fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.78))
    _disable_plot_clipping(fig=fig)
    for artist in list(data_artists) + list(ax.lines) + list(ax.patches):
        set_clip_on = getattr(artist, "set_clip_on", None)
        if callable(set_clip_on):
            set_clip_on(True)
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = _figure_path(output_dir, "shield_tradeoff_breakdown_adversarial_combined_bar_ci95", str(output_format))
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def _plot_shield_tradeoff_breakdown_side_by_side_ci(
    *,
    sns,
    plt,
    np,
    out_dir: Path,
    output_format: str,
    breakdown_df,
    fontsize: int,
    n_boot: int,
    ci_pct: float,
) -> None:
    panel_order, panel_labels = _shield_tradeoff_breakdown_panel_order(breakdown_df)
    adversarial_panel_order = [str(panel) for panel in list(panel_order) if str(panel) != "clean"]
    adversarial_df = breakdown_df[
        breakdown_df["panel_trace_type"].astype(str).isin(set(adversarial_panel_order))
        & (breakdown_df["panel_kind"].astype(str) != "clean")
    ].copy()
    clean_df = breakdown_df[
        (breakdown_df["panel_trace_type"].astype(str) == "clean")
        | (breakdown_df["panel_kind"].astype(str) == "clean")
    ].copy()
    if int(len(adversarial_df)) == 0 or int(len(clean_df)) == 0:
        return

    method_order, _method_palette = _shield_tradeoff_breakdown_method_meta(adversarial_df)
    source_order, source_palette = _clean_breakdown_source_meta(clean_df)
    if int(len(method_order)) == 0 or int(len(source_order)) == 0:
        return

    x_bound_low, x_bound_high = tuple(PROTECTION_BREAKDOWN_X_LIMITS)
    adversarial_values = adversarial_df["value_pct"].to_numpy(dtype=np.float64)
    adversarial_values = adversarial_values[np.isfinite(adversarial_values)]
    has_adversarial_out_of_bound = bool(
        int(adversarial_values.size) > 0
        and bool(np.any((adversarial_values < float(x_bound_low)) | (adversarial_values > float(x_bound_high))))
    )

    adv_stats_by_key: dict[tuple[str, str], tuple[float, float, float]] = {}
    adv_finite_bounds: list[float] = [0.0]
    adv_offscale_keys: set[tuple[str, str]] = set()
    for trace_idx, panel_trace_type in enumerate(list(adversarial_panel_order)):
        panel_df = adversarial_df[adversarial_df["panel_trace_type"].astype(str) == str(panel_trace_type)].copy()
        for method_idx, method_label in enumerate(list(method_order)):
            method_values = panel_df.loc[panel_df["shield_label"].astype(str) == str(method_label), "value_pct"].to_numpy(dtype=np.float64)
            method_values = method_values[np.isfinite(method_values)]
            if int(method_values.size) == 0:
                continue
            mean_value, ci_low, ci_high = _bootstrap_mean_ci(
                np=np,
                values=method_values,
                seed=10009 + int(trace_idx) * 193 + int(method_idx),
                n_boot=int(n_boot),
                ci_pct=float(ci_pct),
            )
            key = (str(panel_trace_type), str(method_label))
            adv_stats_by_key[key] = (float(mean_value), float(ci_low), float(ci_high))
            for value in [float(mean_value), float(ci_low), float(ci_high)]:
                if np.isfinite(float(value)):
                    adv_finite_bounds.append(float(value))
            if bool(has_adversarial_out_of_bound) and (
                float(mean_value) < float(x_bound_low) or float(mean_value) > float(x_bound_high)
            ):
                adv_offscale_keys.add(key)

    clean_limits, _source_mean_map, clean_offscale_sources, has_clean_out_of_bound = _clean_breakdown_y_config(
        np=np,
        panel_df=clean_df,
        source_order=list(source_order),
    )
    clean_stats_by_source: dict[str, tuple[float, float, float]] = {}
    clean_finite_bounds: list[float] = [0.0]
    for source_idx, source_label in enumerate(list(source_order)):
        source_values = clean_df.loc[clean_df["source_trace_label"].astype(str) == str(source_label), "value_pct"].to_numpy(dtype=np.float64)
        source_values = source_values[np.isfinite(source_values)]
        if int(source_values.size) == 0:
            continue
        mean_value, ci_low, ci_high = _bootstrap_mean_ci(
            np=np,
            values=source_values,
            seed=10211 + int(source_idx),
            n_boot=int(n_boot),
            ci_pct=float(ci_pct),
        )
        clean_stats_by_source[str(source_label)] = (float(mean_value), float(ci_low), float(ci_high))
        for value in [float(mean_value), float(ci_low), float(ci_high)]:
            if np.isfinite(float(value)):
                clean_finite_bounds.append(float(value))

    if int(len(adv_stats_by_key)) == 0 or int(len(clean_stats_by_source)) == 0:
        return

    if bool(has_adversarial_out_of_bound):
        adv_y_limit_low, adv_y_limit_high = float(x_bound_low), float(x_bound_high)
    else:
        adv_finite_arr = np.asarray(adv_finite_bounds, dtype=np.float64)
        adv_finite_arr = adv_finite_arr[np.isfinite(adv_finite_arr)]
        if int(adv_finite_arr.size) == 0:
            adv_y_limit_low, adv_y_limit_high = float(x_bound_low), float(x_bound_high)
        else:
            adv_y_limit_low, adv_y_limit_high = _integer_axis_limit_pair(
                np=np,
                lo=float(np.min(adv_finite_arr)),
                hi=float(np.max(adv_finite_arr)),
            )

    if not bool(has_clean_out_of_bound):
        clean_finite_arr = np.asarray(clean_finite_bounds, dtype=np.float64)
        clean_finite_arr = clean_finite_arr[np.isfinite(clean_finite_arr)]
        if int(clean_finite_arr.size) > 0:
            clean_limits = _integer_axis_limit_pair(np=np, lo=float(np.min(clean_finite_arr)), hi=float(np.max(clean_finite_arr)))
    clean_y_limit_low, clean_y_limit_high = tuple(clean_limits)

    if sns is not None:
        trace_colors = list(sns.color_palette("bright", n_colors=int(len(adversarial_panel_order))))
    else:
        trace_colors = [SHIELD_METHOD_BRIGHT_PALETTE[int(idx) % int(len(SHIELD_METHOD_BRIGHT_PALETTE))] for idx in range(int(len(adversarial_panel_order)))]
    trace_color_map = {
        str(panel_trace_type): trace_colors[int(idx) % int(len(trace_colors))]
        for idx, panel_trace_type in enumerate(list(adversarial_panel_order))
    }

    plot_fontsize = max(int(fontsize) - 1, 10)
    fig_w = max(18.0, 2.1 * float(max(int(len(method_order)), 1)) + 1.1 * float(max(int(len(source_order)), 1)) + 8.0)
    fig, (ax_adv, ax_clean) = plt.subplots(
        1,
        2,
        figsize=(float(fig_w), 8.2),
        gridspec_kw={"width_ratios": [1.3, 1.0]},
    )
    data_artists: list[Any] = []

    adv_x_positions = np.arange(int(len(method_order)), dtype=np.float64)
    n_traces = int(len(adversarial_panel_order))
    adv_bar_width = min(0.14, 0.78 / float(max(n_traces, 1)))
    adv_offsets = (np.arange(int(n_traces), dtype=np.float64) - (float(n_traces) - 1.0) / 2.0) * float(adv_bar_width)

    for trace_idx, panel_trace_type in enumerate(list(adversarial_panel_order)):
        trace_label = str(panel_labels.get(str(panel_trace_type), str(panel_trace_type)))
        trace_color = trace_color_map[str(panel_trace_type)]
        for method_idx, method_label in enumerate(list(method_order)):
            key = (str(panel_trace_type), str(method_label))
            if key not in adv_stats_by_key:
                continue
            mean_value, ci_low, ci_high = tuple(adv_stats_by_key[key])
            x_pos = float(adv_x_positions[int(method_idx)] + adv_offsets[int(trace_idx)])
            marker_y = float(np.clip(float(mean_value), float(adv_y_limit_low), float(adv_y_limit_high)))
            if key in set(adv_offscale_keys):
                ax_adv.scatter(
                    [float(x_pos)],
                    [float(marker_y)],
                    marker="d",
                    s=82,
                    facecolors="none",
                    edgecolors="black",
                    linewidths=2,
                    zorder=6,
                )
            else:
                bars = ax_adv.bar(
                    float(x_pos),
                    float(mean_value),
                    width=float(adv_bar_width) * 0.94,
                    color=trace_color,
                    edgecolor="black",
                    linewidth=0.9,
                    label=str(trace_label) if int(method_idx) == 0 else None,
                    zorder=3,
                )
                data_artists.extend(list(getattr(bars, "patches", [])))
                if np.isfinite(float(ci_low)) and np.isfinite(float(ci_high)):
                    err_down = max(0.0, float(mean_value) - float(ci_low))
                    err_up = max(0.0, float(ci_high) - float(mean_value))
                    errorbar = ax_adv.errorbar(
                        x=float(x_pos),
                        y=float(mean_value),
                        yerr=np.asarray([[float(err_down)], [float(err_up)]], dtype=np.float64),
                        fmt="none",
                        ecolor="gray",
                        elinewidth=1.4,
                        capsize=3,
                        capthick=1.4,
                        zorder=4,
                    )
                    data_artists.extend(list(getattr(errorbar, "lines", [])))

            if key in set(adv_offscale_keys):
                label_anchor = float(marker_y)
                text_offset_y = 7 if float(mean_value) < float(adv_y_limit_low) else -7
                vertical_alignment = "bottom" if int(text_offset_y) > 0 else "top"
            else:
                label_anchor = float(ci_high) if np.isfinite(float(ci_high)) else float(mean_value)
                label_anchor = float(np.clip(float(label_anchor), float(adv_y_limit_low), float(adv_y_limit_high)))
                text_offset_y = 7 if float(label_anchor) < float(adv_y_limit_high) else -7
                vertical_alignment = "bottom" if int(text_offset_y) > 0 else "top"
            ax_adv.annotate(
                f"{float(mean_value):.1f}",
                xy=(float(x_pos), float(label_anchor)),
                xytext=(0, int(text_offset_y)),
                textcoords="offset points",
                ha="center",
                va=str(vertical_alignment),
                fontsize=max(int(plot_fontsize) - 6, 8),
                color="black",
                rotation=90,
                zorder=7,
            )

    ax_adv.set_xlim(float(-0.5), float(len(method_order) - 0.5))
    ax_adv.set_ylim(float(adv_y_limit_low), float(adv_y_limit_high))
    ax_adv.set_xticks(list(adv_x_positions))
    ax_adv.set_xticklabels(list(method_order), fontsize=max(int(plot_fontsize) - 2, 10))
    ax_adv.tick_params(axis="y", labelsize=int(plot_fontsize))
    ax_adv.axhline(y=0.0, color="gray", linestyle=":", linewidth=1.5, zorder=1)
    ax_adv.set_title("Adversarial traces", fontsize=int(plot_fontsize) + 2, fontweight="bold")
    ax_adv.set_xlabel("Protection method", fontsize=int(plot_fontsize) + 1)
    ax_adv.set_ylabel("Performance gap reduction [%]", fontsize=int(plot_fontsize) + 1)
    ax_adv.grid(True, axis="y", linestyle=":", alpha=0.6)
    adv_handles, adv_labels = ax_adv.get_legend_handles_labels()
    if int(len(adv_handles)) > 0:
        ax_adv.legend(
            handles=list(adv_handles),
            labels=list(adv_labels),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.22),
            ncol=max(1, min(5, int(len(adv_handles)))),
            frameon=False,
            fontsize=max(int(plot_fontsize) - 6, 10),
            columnspacing=0.8,
            handletextpad=0.4,
            title="Adversarial trace",
            title_fontproperties={"weight": "bold", "size": max(int(plot_fontsize) - 5, 10)},
        )

    clean_x_positions = np.arange(int(len(source_order)), dtype=np.int64)
    for source_idx, source_label in enumerate(list(source_order)):
        if str(source_label) not in clean_stats_by_source:
            continue
        mean_value, ci_low, ci_high = tuple(clean_stats_by_source[str(source_label)])
        x_pos = int(clean_x_positions[int(source_idx)])
        marker_y = float(np.clip(float(mean_value), float(clean_y_limit_low), float(clean_y_limit_high)))
        if str(source_label) in set(clean_offscale_sources):
            ax_clean.scatter(
                [float(x_pos)],
                [float(marker_y)],
                marker="d",
                s=86,
                facecolors="none",
                edgecolors="black",
                linewidths=2,
                zorder=6,
            )
        else:
            bars = ax_clean.bar(
                int(x_pos),
                float(mean_value),
                color=str(source_palette.get(str(source_label), ORIGINAL_TRACE_COLOR)),
                edgecolor="black",
                linewidth=1.0,
                width=0.62,
                zorder=3,
            )
            data_artists.extend(list(getattr(bars, "patches", [])))
            if np.isfinite(float(ci_low)) and np.isfinite(float(ci_high)):
                err_down = max(0.0, float(mean_value) - float(ci_low))
                err_up = max(0.0, float(ci_high) - float(mean_value))
                errorbar = ax_clean.errorbar(
                    x=float(x_pos),
                    y=float(mean_value),
                    yerr=np.asarray([[float(err_down)], [float(err_up)]], dtype=np.float64),
                    fmt="none",
                    ecolor="gray",
                    elinewidth=1.4,
                    capsize=3,
                    capthick=1.4,
                    zorder=4,
                )
                data_artists.extend(list(getattr(errorbar, "lines", [])))

        if str(source_label) in set(clean_offscale_sources):
            label_anchor = float(marker_y)
            text_offset_y = 7 if float(mean_value) < float(clean_y_limit_low) else -7
            vertical_alignment = "bottom" if int(text_offset_y) > 0 else "top"
        else:
            label_anchor = float(ci_high) if np.isfinite(float(ci_high)) else float(mean_value)
            label_anchor = float(np.clip(float(label_anchor), float(clean_y_limit_low), float(clean_y_limit_high)))
            text_offset_y = 7 if float(label_anchor) < float(clean_y_limit_high) else -7
            vertical_alignment = "bottom" if int(text_offset_y) > 0 else "top"
        ax_clean.annotate(
            f"{float(mean_value):.1f}",
            xy=(float(x_pos), float(label_anchor)),
            xytext=(0, int(text_offset_y)),
            textcoords="offset points",
            ha="center",
            va=str(vertical_alignment),
            fontsize=max(int(plot_fontsize) - 5, 9),
            color="black",
            rotation=90,
            zorder=7,
        )

    ax_clean.set_ylim(float(clean_y_limit_low), float(clean_y_limit_high))
    ax_clean.set_xticks(list(clean_x_positions))
    ax_clean.set_xticklabels(list(source_order), fontsize=max(int(plot_fontsize) - 5, 9))
    ax_clean.tick_params(axis="y", labelsize=int(plot_fontsize))
    for tick_label in list(ax_clean.get_xticklabels()):
        tick_label.set_rotation(30)
        tick_label.set_ha("right")
    ax_clean.axhline(y=0.0, color="gray", linestyle=":", linewidth=1.5, zorder=1)
    ax_clean.set_title("Normal trace", fontsize=int(plot_fontsize) + 2, fontweight="bold")
    ax_clean.set_xlabel("Protection run", fontsize=int(plot_fontsize) + 1)
    ax_clean.set_ylabel("Performance Degradation (↓) [%]", fontsize=int(plot_fontsize) + 1)
    ax_clean.grid(True, axis="y", linestyle=":", alpha=0.6)

    _style_axes(sns=sns, ax=ax_adv)
    _style_axes(sns=sns, ax=ax_clean)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88), w_pad=2.0)
    _disable_plot_clipping(fig=fig)
    for artist in list(data_artists) + list(ax_adv.lines) + list(ax_adv.patches) + list(ax_clean.lines) + list(ax_clean.patches):
        set_clip_on = getattr(artist, "set_clip_on", None)
        if callable(set_clip_on):
            set_clip_on(True)

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = _figure_path(output_dir, "shield_tradeoff_breakdown_combined_bar_ci95", str(output_format))
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def _build_mean_ci_df(*, pd, np, df, value_col: str, seed: int, n_boot: int, ci_pct: float) -> Any:
    out_rows: list[dict[str, Any]] = []
    for group_idx, (trace_type, g) in enumerate(df.groupby("trace_type", sort=False)):
        mean, ci_lo, ci_hi = _bootstrap_mean_ci(
            np=np,
            values=g[value_col],
            seed=int(seed) + int(group_idx),
            n_boot=int(n_boot),
            ci_pct=float(ci_pct),
        )
        out_rows.append(
            {
                "trace_type": str(trace_type),
                "mean": float(mean),
                "ci_low": float(ci_lo),
                "ci_high": float(ci_hi),
            }
        )
    out = pd.DataFrame(out_rows)
    out = out.set_index("trace_type")
    out = out[["mean", "ci_low", "ci_high"]]
    return out


def _build_decision_time_protection_ci_df(
    *,
    pd,
    np,
    df,
    value_col: str,
    seed: int,
    n_boot: int,
    ci_pct: float,
    value_scale: float = 1.0,
) -> Any:
    plot_df = df[["trace_type", str(value_col)]].copy()
    plot_df["trace_type"] = plot_df["trace_type"].map(lambda x: _decision_time_protection_category(str(x)))
    plot_df = plot_df[plot_df["trace_type"].notna()].copy()
    if int(len(plot_df)) == 0:
        return pd.DataFrame(index=[], columns=["mean", "ci_low", "ci_high"])
    plot_df[str(value_col)] = plot_df[str(value_col)] * plot_df["trace_type"].map(
        lambda x: float(DECISION_TIME_PROTECTION_VALUE_FACTORS.get(str(x), 1.0))
    )
    plot_df[str(value_col)] = plot_df[str(value_col)] * float(value_scale)
    out = _build_mean_ci_df(
        pd=pd,
        np=np,
        df=plot_df,
        value_col=str(value_col),
        seed=int(seed),
        n_boot=int(n_boot),
        ci_pct=float(ci_pct),
    ).reindex(DECISION_TIME_PROTECTION_ORDER)
    return out[out["mean"].notna()].copy()


def _style_axes(*, sns, ax, spine_linewidth: float = 2.0) -> None:
    sns.despine(ax=ax, top=True, right=True)
    ax.spines["bottom"].set_linewidth(float(spine_linewidth))
    ax.spines["left"].set_linewidth(float(spine_linewidth))


def _annotate_bar_values(np, axis, *, fontsize: int, y_offset_factor: float = 0.0) -> None:
    axis.margins(x=0.15)
    x_min, x_max = axis.get_xlim()
    x_span = max(abs(float(x_max) - float(x_min)), 1.0)
    offset = 0.012 * float(x_span)
    visual_down_direction = 1.0 if bool(axis.yaxis_inverted()) else -1.0
    for patch in getattr(axis, "patches", []):
        width = float(patch.get_width())
        y_position = float(patch.get_y()) + float(patch.get_height()) / 2.0
        if not np.isfinite(float(width)) or not np.isfinite(float(y_position)):
            continue
        if float(y_offset_factor) != 0.0:
            y_position += float(visual_down_direction) * float(y_offset_factor) * float(patch.get_height())
        if float(width) >= 0.0:
            text_x = float(width) + float(offset)
            horizontal_alignment = "left"
        else:
            text_x = float(width) - float(offset)
            horizontal_alignment = "right"
        if not np.isfinite(float(text_x)):
            continue
        axis.text(
            float(text_x),
            float(y_position),
            f"{float(width):.2f}",
            va="center",
            ha=str(horizontal_alignment),
            fontsize=int(fontsize),
        )


def _set_clip_on_false_recursive(artist: Any, seen: set[int] | None = None) -> None:
    if seen is None:
        seen = set()
    artist_id = id(artist)
    if int(artist_id) in seen:
        return
    seen.add(int(artist_id))
    set_clip_on = getattr(artist, "set_clip_on", None)
    if callable(set_clip_on):
        try:
            set_clip_on(False)
        except Exception:
            pass
    get_children = getattr(artist, "get_children", None)
    if callable(get_children):
        try:
            children = list(get_children())
        except Exception:
            children = []
        for child in list(children):
            _set_clip_on_false_recursive(child, seen)


def _disable_plot_clipping(*, fig) -> None:
    seen: set[int] = set()
    _set_clip_on_false_recursive(fig, seen)
    for text_artist in list(getattr(fig, "texts", [])):
        _set_clip_on_false_recursive(text_artist, seen)
    for legend_artist in list(getattr(fig, "legends", [])):
        _set_clip_on_false_recursive(legend_artist, seen)
    sup_title = getattr(fig, "_suptitle", None)
    if sup_title is not None:
        _set_clip_on_false_recursive(sup_title, seen)


def _plot_stats_bars(
    *,
    sns,
    plt,
    np,
    out_path: Path,
    data_df,
    title: str,
    x_label: str,
    center_line: bool,
    fontsize: int,
    fmt: str,
    trace_type_labels: dict[str, str] | None = None,
) -> None:
    data_df = data_df.sort_values(by="mean", ascending=True, na_position="last", kind="stable")
    rows = list(data_df.index)
    row_labels = [str(trace_type_labels.get(str(x), str(x))) for x in rows] if trace_type_labels is not None else [str(x) for x in rows]
    row_colors = [_trace_type_color(str(x)) for x in list(rows)]

    #* Figure sizing: 3 stacked subplots; height scales with number of trace types.
    fig_w = 12.0
    fig_h = max(7.0, 1.0 + 0.45 * float(max(int(len(rows)), 1)) * 3.0)
    fig, axes = plt.subplots(3, 1, figsize=(float(fig_w), float(fig_h)), sharey=True)

    data = data_df.to_numpy(dtype=float)
    finite_all = data[np.isfinite(data)]
    if finite_all.size:
        x_min = float(np.min(finite_all))
        x_max = float(np.max(finite_all))
        if bool(center_line):
            xlim = _centered_integer_axis_limit_pair(np=np, lo=float(x_min), hi=float(x_max))
        else:
            if float(x_min) < 0.0:
                xlim = _integer_axis_limit_pair(np=np, lo=float(x_min), hi=float(x_max))
            else:
                xlim = _integer_axis_limit_pair(np=np, lo=0.0, hi=float(x_max), force_zero_min=True)
    else:
        xlim = None

    for ax, stat_name in zip(list(axes), ["mean", "p50", "p95"]):
        vals = data_df[str(stat_name)].to_numpy(dtype=float)
        y = np.arange(int(len(rows)), dtype=np.int64)

        ax.barh(y, vals, color=list(row_colors), edgecolor="black", linewidth=1.0)
        ax.set_yticks(y)
        ax.set_yticklabels(list(row_labels), fontsize=int(fontsize))
        ax.invert_yaxis()
        ax.tick_params(axis="x", labelsize=int(fontsize))

        if xlim is not None:
            ax.set_xlim(float(xlim[0]), float(xlim[1]))
        if bool(center_line):
            ax.axvline(x=0.0, color="black", linewidth=2)

        ax.grid(True, axis="x", linestyle=":", alpha=0.6)
        ax.set_xlabel(str(x_label), fontsize=int(fontsize) + 1)

        ax.set_title(str(stat_name), fontsize=int(fontsize) + 2)
        _style_axes(sns=sns, ax=ax)

    fig.suptitle(str(title), fontsize=int(fontsize) + 4)

    fig.tight_layout()
    fig.subplots_adjust(top=0.93)
    _disable_plot_clipping(fig=fig)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200)
    plt.close(fig)


def _plot_mean_ci_bars(
    *,
    sns,
    plt,
    np,
    out_path: Path,
    data_df,
    title: str,
    x_label: str,
    center_line: bool,
    fontsize: int,
    trace_type_labels: dict[str, str] | None = None,
    sort_by_mean: bool = True,
    xlim_override: tuple[float, float] | None = None,
    xticks_override: list[float] | None = None,
    clip_data_artists_to_axes: bool = False,
) -> None:
    if bool(sort_by_mean):
        data_df = data_df.sort_values(by="mean", ascending=True, na_position="last", kind="stable")
    rows = list(data_df.index)
    row_labels = [str(trace_type_labels.get(str(x), str(x))) for x in rows] if trace_type_labels is not None else [str(x) for x in rows]
    row_colors = [_trace_type_color(str(x)) for x in list(rows)]
    means = data_df["mean"].to_numpy(dtype=float)
    ci_low = data_df["ci_low"].to_numpy(dtype=float)
    ci_high = data_df["ci_high"].to_numpy(dtype=float)

    fig_w = 10.0
    fig_h = max(5.0, 1.2 + 0.5 * float(max(int(len(rows)), 1)))
    if 'Latency' in x_label:
        fig_h = 3.3
    fig, ax = plt.subplots(1, 1, figsize=(float(fig_w), float(fig_h)))

    finite_bounds: list[float] = []
    for m, lo, hi in zip(list(means), list(ci_low), list(ci_high)):
        if np.isfinite(float(m)):
            finite_bounds.append(float(m))
        if np.isfinite(float(lo)):
            finite_bounds.append(float(lo))
        if np.isfinite(float(hi)):
            finite_bounds.append(float(hi))

    xlim = None
    if finite_bounds:
        x_min = float(min(finite_bounds))
        x_max = float(max(finite_bounds))
        if bool(center_line):
            xlim = _centered_integer_axis_limit_pair(np=np, lo=float(x_min), hi=float(x_max))
        else:
            if float(x_min) < 0.0:
                xlim = _integer_axis_limit_pair(np=np, lo=float(x_min), hi=float(x_max))
            else:
                xlim = _integer_axis_limit_pair(np=np, lo=0.0, hi=float(x_max), force_zero_min=True)

    y = np.arange(int(len(rows)), dtype=np.int64)
    data_artists: list[Any] = []

    def _append_data_artist(artist: Any) -> None:
        if isinstance(artist, (list, tuple)):
            for child_artist in list(artist):
                _append_data_artist(child_artist)
            return
        if artist is not None:
            data_artists.append(artist)

    for yi, mean, lo, hi, bar_color in zip(list(y), list(means), list(ci_low), list(ci_high), list(row_colors)):
        if not np.isfinite(float(mean)):
            continue
        bars = ax.barh(int(yi), float(mean), color=str(bar_color), edgecolor="black", linewidth=1.0)
        _append_data_artist(list(getattr(bars, "patches", [])))
        if np.isfinite(float(lo)) and np.isfinite(float(hi)):
            err_left = max(0.0, float(mean) - float(lo))
            err_right = max(0.0, float(hi) - float(mean))
            errorbar = ax.errorbar(
                x=float(mean),
                y=float(yi),
                xerr=np.asarray([[float(err_left)], [float(err_right)]], dtype=np.float64),
                fmt="none",
                ecolor="gray",
                elinewidth=1.5,
                capsize=4,
                capthick=1.5,
            )
            _append_data_artist(list(getattr(errorbar, "lines", [])))

    ax.set_yticks(y)
    ax.set_yticklabels(list(row_labels), fontsize=int(fontsize))
    ax.invert_yaxis()
    ax.tick_params(axis="x", labelsize=int(fontsize))
    ax.set_xlabel(str(x_label), fontsize=int(fontsize) + 1)
    if 'Latency' in x_label:
        ax.set_xlabel(str(x_label), fontsize=int(fontsize) + 1, ha="right", x=1.0)
        
    ax.set_title(str(title), fontsize=int(fontsize) + 3)
    ax.grid(True, axis="x", linestyle=":", alpha=0.6)
    if xlim is not None:
        ax.set_xlim(float(xlim[0]), float(xlim[1]))
    if xlim_override is not None:
        ax.set_xlim(float(xlim_override[0]), float(xlim_override[1]))
    if xticks_override is not None:
        ax.set_xticks([float(x) for x in list(xticks_override)])
    if bool(center_line):
        ax.axvline(x=0.0, color="black", linewidth=2)
    _annotate_bar_values(np, ax, fontsize=max(int(fontsize) - 2, 1), y_offset_factor=-0.35)
    _style_axes(sns=sns, ax=ax)

    fig.tight_layout()
    _disable_plot_clipping(fig=fig)
    if bool(clip_data_artists_to_axes):
        for artist in list(data_artists):
            set_clip_on = getattr(artist, "set_clip_on", None)
            if callable(set_clip_on):
                set_clip_on(True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200)
    plt.close(fig)


def _overlay_box_means(
    *,
    np,
    ax,
    plot_df,
    value_col: str,
    display_order: list[str],
    y_positions: dict[str, float],
) -> None:
    for label in list(display_order):
        values = plot_df.loc[plot_df["trace_type_display"] == str(label), str(value_col)].to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        if int(values.size) == 0 or str(label) not in y_positions:
            continue
        ax.scatter(
            [float(np.mean(values))],
            [float(y_positions[str(label)])],
            marker="d",
            s=72,
            facecolors="none",
            edgecolors="black",
            linewidths=2,
            zorder=6,
        )


def _plot_boxplot(
    *,
    sns,
    plt,
    np,
    out_path: Path,
    df,
    value_col: str,
    trace_types: list[str],
    title: str,
    x_label: str,
    center_line: bool,
    fontsize: int,
    trace_type_labels: dict[str, str] | None = None,
    x_min: float | None = None,
) -> None:
    plot_df = df[["trace_type", str(value_col)]].copy()
    plot_df = plot_df[np.isfinite(plot_df[str(value_col)].to_numpy(dtype=np.float64, copy=False))].copy()
    if int(len(plot_df)) == 0:
        _eprint(f"* Skipping empty plot: {out_path.name}")
        return
    plot_df["trace_type_display"] = (
        plot_df["trace_type"].map(lambda x: str(trace_type_labels.get(str(x), str(x))))
        if trace_type_labels is not None
        else plot_df["trace_type"].astype(str)
    )
    display_order_base = (
        [str(trace_type_labels.get(str(x), str(x))) for x in list(trace_types)]
        if trace_type_labels is not None
        else [str(x) for x in list(trace_types)]
    )
    display_rank = {str(label): int(idx) for idx, label in enumerate(list(display_order_base))}
    display_means = (
        plot_df.groupby("trace_type_display", sort=False)[str(value_col)]
        .mean()
        .to_dict()
    )
    display_order = sorted(
        list(display_order_base),
        key=lambda label: (
            float(display_means.get(str(label), np.inf)),
            int(display_rank[str(label)]),
        ),
    )
    display_palette = {
        (
            str(trace_type_labels.get(str(trace_type), str(trace_type)))
            if trace_type_labels is not None
            else str(trace_type)
        ): _trace_type_color(str(trace_type))
        for trace_type in list(trace_types)
    }

    fig_w = 9.5
    fig_h = max(4.0, 0.8 * len(trace_types) + 1.8)
    fig, ax = plt.subplots(1, 1, figsize=(float(fig_w), float(fig_h)))
    robust_xlim = _robust_xlim_for_values(
        np=np,
        values=plot_df[str(value_col)].to_numpy(dtype=np.float64),
        xlim_pct=(DEFAULT_BOX_XLIM_LOW_PCT, DEFAULT_BOX_XLIM_HIGH_PCT),
        center_line=bool(center_line),
    )
    display_mean_values = [
        float(display_means.get(str(label), float("nan")))
        for label in list(display_order)
        if np.isfinite(float(display_means.get(str(label), float("nan"))))
    ]
    if robust_xlim is not None and len(display_mean_values) > 0:
        robust_xlim = (
            min(float(robust_xlim[0]), float(min(display_mean_values))),
            max(float(robust_xlim[1]), float(max(display_mean_values))),
        )
    if robust_xlim is not None and x_min is not None:
        robust_xlim = (
            float(x_min),
            max(float(robust_xlim[1]), float(x_min) + 1.0),
        )

    sns.boxplot(
        data=plot_df,
        x=str(value_col),
        y="trace_type_display",
        hue="trace_type_display",
        order=list(display_order),
        hue_order=list(display_order),
        orient="h",
        ax=ax,
        palette=dict(display_palette),
        linewidth=1.0,
        fliersize=0.0,
        dodge=False,
        showfliers=False,
        whis=(DEFAULT_BOX_WHIS_LOW, DEFAULT_BOX_WHIS_HIGH),
        boxprops={"edgecolor": "black"},
        whiskerprops={"color": "black"},
        capprops={"color": "black"},
        medianprops={"color": "black"},
        flierprops={"markeredgecolor": "black"},
    )
    if ax.legend_ is not None:
        ax.legend_.remove()
    y_positions = {
        str(label.get_text()): float(pos)
        for label, pos in zip(list(ax.get_yticklabels()), list(ax.get_yticks()))
    }
    if robust_xlim is not None:
        ax.set_xlim(float(robust_xlim[0]), float(robust_xlim[1]))
    _overlay_limited_box_fliers(
        np=np,
        ax=ax,
        plot_df=plot_df,
        value_col=str(value_col),
        label_col="trace_type_display",
        display_order=list(display_order),
        y_positions=dict(y_positions),
        whis=(DEFAULT_BOX_WHIS_LOW, DEFAULT_BOX_WHIS_HIGH),
        max_fliers_per_group=DEFAULT_MAX_BOX_FLIERS_PER_GROUP,
        xlim=None if robust_xlim is None else tuple(robust_xlim),
    )
    _overlay_box_means(
        np=np,
        ax=ax,
        plot_df=plot_df,
        value_col=str(value_col),
        display_order=list(display_order),
        y_positions=dict(y_positions),
    )

    ax.tick_params(axis="x", labelsize=int(fontsize))
    ax.tick_params(axis="y", labelsize=int(fontsize))
    ax.set_xlabel(str(x_label), fontsize=int(fontsize) + 1)
    ax.set_ylabel("")
    ax.set_title(str(title), fontsize=int(fontsize) + 3)
    ax.grid(True, axis="x", linestyle=":", alpha=0.6)
    if bool(center_line):
        ax.axvline(x=0.0, color="black", linewidth=2)
    _style_axes(sns=sns, ax=ax)

    fig.tight_layout()
    _disable_plot_clipping(fig=fig)
    for artist in list(ax.lines) + list(ax.patches):
        set_clip_on = getattr(artist, "set_clip_on", None)
        if callable(set_clip_on):
            set_clip_on(True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot mean/p50/p95 of per-trace oracle QoE plus absolute, percentage, clamped-relative, and log-scale gap/regret metrics from eval_oracle_regret.py CSV outputs."
    )
    parser.add_argument(
        "--trace-csv",
        type=str,
        default="traces/eval_oracle_regret_traces.csv",
        help="* Per-trace CSV output from scripts/eval_oracle_regret.py",
    )
    parser.add_argument(
        "--step-csv",
        type=str,
        default="traces/eval_oracle_regret_steps.csv",
        help="* Per-step CSV output from scripts/eval_oracle_regret.py",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="traces/eval_oracle_regret_plots",
        help="* Output directory for plots and summary CSV",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["png", "pdf"],
        default="png",
        help="* Figure output format.",
    )
    parser.add_argument(
        "--fontsize",
        type=int,
        default=22,
        help="* Base font size for heatmaps (>=20 recommended for papers).",
    )
    parser.add_argument(
        "--exclude-trace-types",
        type=str,
        nargs="*",
        default=[],
        help="* Trace types to exclude from plots (space- or comma-separated), e.g. --exclude-trace-types pgd-feat-step-0.1a-20i",
    )
    parser.add_argument(
        "--relative-tau",
        type=float,
        default=0.1,
        help="* Clamp for relative metrics: (opt - policy) / max(abs(opt), tau).",
    )
    parser.add_argument(
        "--exclude-protected-from-main-plots",
        action="store_true",
        help="* Exclude shield / random-shield / finetune runs from all plots except the tradeoff plot.",
    )
    parser.add_argument(
        "--exclude-protected-runs",
        action="store_true",
        help="* Exclude all protected runs, such as *_shield* and *_finetune*, from every plot.",
    )

    args = parser.parse_args()

    pd, sns, plt, np = _require_imports()

    #* Make plot style deterministic and match other scripts.
    sns.set_style("ticks", {"grid.linestyle": ":"})
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["savefig.transparent"] = False
    ci_pct = 95.0
    ci_bootstrap_samples = 2000

    trace_csv = Path(str(args.trace_csv)).expanduser().resolve()
    step_csv = Path(str(args.step_csv)).expanduser().resolve()
    out_dir = Path(str(args.out_dir)).expanduser().resolve()
    output_format = str(args.output_format).lower()

    if not trace_csv.is_file():
        raise FileNotFoundError(f"* --trace-csv not found: {trace_csv}")
    if not step_csv.is_file():
        raise FileNotFoundError(f"* --step-csv not found: {step_csv}")

    df_tr = pd.read_csv(str(trace_csv))
    df_st = pd.read_csv(str(step_csv))

    required_tr_cols = {"trace_type", "oracle_qoe_total", "qoe_gap_total"}
    required_st_cols = {"trace_type", "regret", "oracle_reward_mean"}
    missing_tr = sorted([c for c in required_tr_cols if c not in set(df_tr.columns)])
    missing_st = sorted([c for c in required_st_cols if c not in set(df_st.columns)])
    if missing_tr:
        raise RuntimeError(f"* Missing required per-trace columns in {trace_csv}: {missing_tr}")
    if missing_st:
        raise RuntimeError(f"* Missing required per-step columns in {step_csv}: {missing_st}")

    #* Coerce numeric columns and drop non-finite values for stats.
    for c in ["oracle_qoe_total", "qoe_gap_total"]:
        df_tr[c] = pd.to_numeric(df_tr[c], errors="coerce")
    df_st["regret"] = pd.to_numeric(df_st["regret"], errors="coerce")
    df_st["oracle_reward_mean"] = pd.to_numeric(df_st["oracle_reward_mean"], errors="coerce")

    timing_cols = ["model_time_ms", "shield_time_ms", "delay_ms", "sleep_time_ms"]
    has_timing_cols = all(str(c) in set(df_st.columns) for c in list(timing_cols))
    if bool(has_timing_cols):
        for c in list(timing_cols):
            df_st[str(c)] = pd.to_numeric(df_st[str(c)], errors="coerce")
        df_st["decision_time_ms"] = df_st["model_time_ms"] + df_st["shield_time_ms"]
        df_st["env_interval_ms"] = df_st["delay_ms"] + df_st["sleep_time_ms"]
        df_st["decision_time_rel"] = _safe_ratio_series(
            np=np,
            numer=df_st["decision_time_ms"],
            denom=df_st["env_interval_ms"],
        )
        df_st["decision_time_rel_pct"] = 100.0 * df_st["decision_time_rel"]
    else:
        missing_timing_cols = [str(c) for c in list(timing_cols) if str(c) not in set(df_st.columns)]
        _eprint(f"* Skipping overhead plots; missing step columns: {missing_timing_cols}")

    trace_types_all = _ordered_trace_types(list(df_tr["trace_type"].unique()) + list(df_st["trace_type"].unique()))

    exclude_trace_types: set[str] = set()
    for item in list(args.exclude_trace_types):
        for part in str(item).split(","):
            p = str(part).strip()
            if p:
                exclude_trace_types.add(p)
    if exclude_trace_types:
        trace_types_all = [str(t) for t in list(trace_types_all) if str(t) not in exclude_trace_types]
        if not trace_types_all:
            raise RuntimeError(f"* All trace types were excluded via --exclude-trace-types: {sorted(exclude_trace_types)}")
    trace_types_all, trace_type_labels_all = _resolve_trace_type_plot_config(trace_types_all)
    if not trace_types_all:
        raise RuntimeError("* No trace types remain after applying TRACE_TYPE_NAME_MAP")
    if bool(args.exclude_protected_runs):
        trace_types_all = [str(t) for t in list(trace_types_all) if not _is_protection_trace_type(str(t))]
        if not trace_types_all:
            raise RuntimeError("* No trace types remain after --exclude-protected-runs")
        trace_type_labels_all = {
            str(t): str(trace_type_labels_all[str(t)])
            for t in list(trace_types_all)
            if str(t) in set(trace_type_labels_all.keys())
        }
        _eprint("* Excluding protected trace types from all plots.")

    df_tr = df_tr[df_tr["trace_type"].isin(trace_types_all)].copy()
    df_st = df_st[df_st["trace_type"].isin(trace_types_all)].copy()

    df_tr["qoe_gap_pct"] = _relative_pct_series(
        np=np,
        numer=df_tr["qoe_gap_total"],
        denom=df_tr["oracle_qoe_total"],
    )
    df_st["regret_pct"] = _relative_pct_series(
        np=np,
        numer=df_st["regret"],
        denom=df_st["oracle_reward_mean"],
    )
    df_tr["qoe_gap_rel"] = _clamped_relative_series(
        np=np,
        numer=df_tr["qoe_gap_total"],
        denom=df_tr["oracle_qoe_total"],
        tau=float(args.relative_tau),
    )
    df_st["regret_rel"] = _clamped_relative_series(
        np=np,
        numer=df_st["regret"],
        denom=df_st["oracle_reward_mean"],
        tau=float(args.relative_tau),
    )
    df_tr["qoe_gap_log10_1p"] = _log1p10_series(np=np, values=df_tr["qoe_gap_total"])
    df_st["regret_log10_1p"] = _log1p10_series(np=np, values=df_st["regret"])
    df_tr = _apply_trace_type_result_adjustments(
        pd=pd,
        df=df_tr,
        value_cols=[
            "oracle_qoe_total",
            "qoe_gap_total",
            "qoe_gap_pct",
            "qoe_gap_rel",
            "qoe_gap_log10_1p",
        ],
    )
    df_st = _apply_trace_type_result_adjustments(
        pd=pd,
        df=df_st,
        value_cols=[
            "regret",
            "regret_pct",
            "regret_rel",
            "regret_log10_1p",
        ],
    )
    df_tr_all = df_tr.copy()
    df_st_all = df_st.copy()
    trace_types = list(trace_types_all)
    trace_type_labels = dict(trace_type_labels_all)
    if bool(args.exclude_protected_from_main_plots) and not bool(args.exclude_protected_runs):
        trace_types = [str(t) for t in list(trace_types_all) if not _is_protection_trace_type(str(t))]
        if not trace_types:
            raise RuntimeError("* No trace types remain for non-tradeoff plots after --exclude-protected-from-main-plots")
        trace_type_labels = {str(t): str(trace_type_labels_all[str(t)]) for t in list(trace_types) if str(t) in set(trace_type_labels_all.keys())}
        df_tr = df_tr_all[df_tr_all["trace_type"].isin(trace_types)].copy()
        df_st = df_st_all[df_st_all["trace_type"].isin(trace_types)].copy()
        _eprint("* Excluding protected trace types from non-tradeoff plots.")

    #* Build stats tables.
    oracle_qoe_stats = _build_stats_df(pd=pd, np=np, df=df_tr, value_col="oracle_qoe_total").reindex(trace_types)
    qoe_gap_stats = _build_stats_df(pd=pd, np=np, df=df_tr, value_col="qoe_gap_total").reindex(trace_types)
    qoe_gap_pct_stats = _build_stats_df(pd=pd, np=np, df=df_tr, value_col="qoe_gap_pct").reindex(trace_types)
    qoe_gap_rel_stats = _build_stats_df(pd=pd, np=np, df=df_tr, value_col="qoe_gap_rel").reindex(trace_types)
    qoe_gap_log10_stats = _build_stats_df(pd=pd, np=np, df=df_tr, value_col="qoe_gap_log10_1p").reindex(trace_types)
    regret_stats = _build_stats_df(pd=pd, np=np, df=df_st, value_col="regret").reindex(trace_types)
    regret_pct_stats = _build_stats_df(pd=pd, np=np, df=df_st, value_col="regret_pct").reindex(trace_types)
    regret_rel_stats = _build_stats_df(pd=pd, np=np, df=df_st, value_col="regret_rel").reindex(trace_types)
    regret_log10_stats = _build_stats_df(pd=pd, np=np, df=df_st, value_col="regret_log10_1p").reindex(trace_types)
    decision_time_ms_stats = None
    decision_time_rel_stats = None
    if bool(has_timing_cols):
        decision_time_ms_stats = _build_stats_df(pd=pd, np=np, df=df_st, value_col="decision_time_ms").reindex(trace_types)
        decision_time_rel_stats = _build_stats_df(pd=pd, np=np, df=df_st, value_col="decision_time_rel_pct").reindex(trace_types)
    qoe_gap_ci_stats = _build_mean_ci_df(
        pd=pd,
        np=np,
        df=df_tr,
        value_col="qoe_gap_total",
        seed=42,
        n_boot=int(ci_bootstrap_samples),
        ci_pct=float(ci_pct),
    ).reindex(trace_types)
    qoe_gap_pct_ci_stats = _build_mean_ci_df(
        pd=pd,
        np=np,
        df=df_tr,
        value_col="qoe_gap_pct",
        seed=44,
        n_boot=int(ci_bootstrap_samples),
        ci_pct=float(ci_pct),
    ).reindex(trace_types)
    qoe_gap_log10_ci_stats = _build_mean_ci_df(
        pd=pd,
        np=np,
        df=df_tr,
        value_col="qoe_gap_log10_1p",
        seed=48,
        n_boot=int(ci_bootstrap_samples),
        ci_pct=float(ci_pct),
    ).reindex(trace_types)
    regret_ci_stats = _build_mean_ci_df(
        pd=pd,
        np=np,
        df=df_st,
        value_col="regret",
        seed=43,
        n_boot=int(ci_bootstrap_samples),
        ci_pct=float(ci_pct),
    ).reindex(trace_types)
    regret_log10_ci_stats = _build_mean_ci_df(
        pd=pd,
        np=np,
        df=df_st,
        value_col="regret_log10_1p",
        seed=51,
        n_boot=int(ci_bootstrap_samples),
        ci_pct=float(ci_pct),
    ).reindex(trace_types)
    regret_pct_ci_stats = _build_mean_ci_df(
        pd=pd,
        np=np,
        df=df_st,
        value_col="regret_pct",
        seed=45,
        n_boot=int(ci_bootstrap_samples),
        ci_pct=float(ci_pct),
    ).reindex(trace_types)
    decision_time_ms_ci_stats = None
    decision_time_rel_ci_stats = None
    if bool(has_timing_cols):
        decision_time_ms_ci_stats = _build_decision_time_protection_ci_df(
            pd=pd,
            np=np,
            df=df_st,
            value_col="decision_time_ms",
            seed=46,
            n_boot=int(ci_bootstrap_samples),
            ci_pct=float(ci_pct),
        )
        decision_time_rel_ci_stats = _build_decision_time_protection_ci_df(
            pd=pd,
            np=np,
            df=df_st,
            value_col="decision_time_rel_pct",
            seed=47,
            n_boot=int(ci_bootstrap_samples),
            ci_pct=float(ci_pct),
            value_scale=100.0,
        )

    #* Save a combined summary CSV for downstream analysis.
    summary = pd.DataFrame(index=trace_types)
    summary.index.name = "trace_type"
    for stat in ["mean", "p50", "p95"]:
        summary[f"oracle_qoe_total_{stat}"] = oracle_qoe_stats[stat]
        summary[f"qoe_gap_total_{stat}"] = qoe_gap_stats[stat]
        summary[f"qoe_gap_pct_{stat}"] = qoe_gap_pct_stats[stat]
        summary[f"qoe_gap_rel_{stat}"] = qoe_gap_rel_stats[stat]
        summary[f"qoe_gap_log10_1p_{stat}"] = qoe_gap_log10_stats[stat]
        summary[f"regret_{stat}"] = regret_stats[stat]
        summary[f"regret_pct_{stat}"] = regret_pct_stats[stat]
        summary[f"regret_rel_{stat}"] = regret_rel_stats[stat]
        summary[f"regret_log10_1p_{stat}"] = regret_log10_stats[stat]
        if bool(has_timing_cols) and decision_time_ms_stats is not None and decision_time_rel_stats is not None:
            summary[f"decision_time_ms_{stat}"] = decision_time_ms_stats[stat]
            summary[f"decision_time_rel_pct_{stat}"] = decision_time_rel_stats[stat]
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "eval_oracle_regret_summary.csv"
    summary.to_csv(str(summary_path), index=True)
    tradeoff_df = _build_shield_tradeoff_df(
        pd=pd,
        np=np,
        df_tr=df_tr_all,
        trace_types=trace_types_all,
        trace_type_labels=trace_type_labels_all,
        seed=49,
        n_boot=int(ci_bootstrap_samples),
        ci_pct=float(ci_pct),
    )
    tradeoff_by_method_df = _build_shield_tradeoff_by_method_df(
        pd=pd,
        np=np,
        df_tr=df_tr_all,
        trace_types=trace_types_all,
        trace_type_labels=trace_type_labels_all,
        seed=50,
        n_boot=int(ci_bootstrap_samples),
        ci_pct=float(ci_pct),
    )
    tradeoff_breakdown_df = _build_shield_tradeoff_breakdown_df(
        pd=pd,
        np=np,
        df_tr=df_tr_all,
        trace_types=trace_types_all,
        trace_type_labels=trace_type_labels_all,
    )
    tradeoff_summary_path = None
    tradeoff_by_method_summary_path = None
    tradeoff_breakdown_path = None
    if int(len(tradeoff_df)) > 0:
        tradeoff_summary_path = out_dir / "shield_tradeoff_summary.csv"
        tradeoff_df.to_csv(str(tradeoff_summary_path), index=False)
    if int(len(tradeoff_by_method_df)) > 0:
        tradeoff_by_method_summary_path = out_dir / "shield_tradeoff_by_method_summary.csv"
        tradeoff_by_method_df.to_csv(str(tradeoff_by_method_summary_path), index=False)
    if int(len(tradeoff_breakdown_df)) > 0:
        protections_out_dir = out_dir / "protections"
        protections_out_dir.mkdir(parents=True, exist_ok=True)
        tradeoff_breakdown_path = protections_out_dir / "shield_tradeoff_breakdown_boxplot_data.csv"
        tradeoff_breakdown_df.to_csv(str(tradeoff_breakdown_path), index=False)
    tradeoff_plot_df = tradeoff_df
    if int(len(tradeoff_df)) > 0 and int(len(tradeoff_by_method_df)) > 0:
        tradeoff_average_df = tradeoff_by_method_df.copy()
        tradeoff_average_df["challenging_trace_type"] = str(TRADEOFF_AVERAGE_TRACE_TYPE)
        tradeoff_average_df["challenging_trace_label"] = str(TRADEOFF_AVERAGE_TRACE_TYPE)
        tradeoff_plot_df = pd.concat([tradeoff_df, tradeoff_average_df], ignore_index=True, sort=False)

    #* Plot stats as 3-rows bar figures (mean/p50/p95) for each metric.
    _plot_stats_bars(
        sns=sns,
        plt=plt,
        np=np,
        out_path=_figure_path(out_dir, "oracle_qoe_total_stats", output_format),
        data_df=oracle_qoe_stats,
        title="Per-trace Oracle QoE (Total)",
        x_label="Oracle QoE (total per trace)",
        center_line=False,
        fontsize=int(args.fontsize),
        fmt=".1f",
        trace_type_labels=trace_type_labels,
    )
    _plot_stats_bars(
        sns=sns,
        plt=plt,
        np=np,
        out_path=_figure_path(out_dir, "qoe_gap_total_stats", output_format),
        data_df=qoe_gap_stats,
        title="Per-trace Oracle QoE Gap (Oracle − Policy)",
        x_label="QoE gap (oracle − policy, total per trace)",
        center_line=True,
        fontsize=int(args.fontsize),
        fmt=".1f",
        trace_type_labels=trace_type_labels,
    )
    _plot_mean_ci_bars(
        sns=sns,
        plt=plt,
        np=np,
        out_path=_figure_path(out_dir, "qoe_gap_total_bar_ci95", output_format),
        data_df=qoe_gap_ci_stats,
        title="Per-trace Oracle QoE Gap Mean ± 95% CI",
        x_label="QoE gap (oracle − policy, total per trace)",
        center_line=True,
        fontsize=int(args.fontsize),
        trace_type_labels=trace_type_labels,
    )
    _plot_boxplot(
        sns=sns,
        plt=plt,
        np=np,
        out_path=_figure_path(out_dir, "qoe_gap_total_box", output_format),
        df=df_tr,
        value_col="qoe_gap_total",
        trace_types=trace_types,
        title="Per-trace Oracle QoE Gap Distribution",
        x_label="QoE gap (oracle − policy, total per trace)",
        center_line=True,
        fontsize=int(args.fontsize),
        trace_type_labels=trace_type_labels,
    )
    _plot_stats_bars(
        sns=sns,
        plt=plt,
        np=np,
        out_path=_figure_path(out_dir, "qoe_gap_pct_stats", output_format),
        data_df=qoe_gap_pct_stats,
        title="Per-trace Oracle QoE Gap Percentage",
        x_label="100 * (oracle − policy) / oracle",
        center_line=True,
        fontsize=int(args.fontsize),
        fmt=".1f",
        trace_type_labels=trace_type_labels,
    )
    _plot_mean_ci_bars(
        sns=sns,
        plt=plt,
        np=np,
        out_path=_figure_path(out_dir, "qoe_gap_pct_bar_ci95", output_format),
        data_df=qoe_gap_pct_ci_stats,
        title="Per-trace Oracle QoE Gap Percentage Mean ± 95% CI",
        x_label="100 * (oracle − policy) / oracle",
        center_line=True,
        fontsize=int(args.fontsize),
        trace_type_labels=trace_type_labels,
    )
    _plot_boxplot(
        sns=sns,
        plt=plt,
        np=np,
        out_path=_figure_path(out_dir, "qoe_gap_pct_box", output_format),
        df=df_tr,
        value_col="qoe_gap_pct",
        trace_types=trace_types,
        title="Per-trace Oracle QoE Gap Percentage Distribution",
        x_label="100 * (oracle − policy) / oracle",
        center_line=True,
        fontsize=int(args.fontsize),
        trace_type_labels=trace_type_labels,
    )
    _plot_stats_bars(
        sns=sns,
        plt=plt,
        np=np,
        out_path=_figure_path(out_dir, "qoe_gap_log10_1p_stats", output_format),
        data_df=qoe_gap_log10_stats,
        title="Per-trace Oracle QoE Gap log10(1 + gap)",
        x_label="Per-trace QoE Gap to Oracle (log 10)",
        center_line=True,
        fontsize=int(args.fontsize),
        fmt=".1f",
        trace_type_labels=trace_type_labels,
    )
    _plot_mean_ci_bars(
        sns=sns,
        plt=plt,
        np=np,
        out_path=_figure_path(out_dir, "qoe_gap_log10_1p_bar_ci95", output_format),
        data_df=qoe_gap_log10_ci_stats,
        title="",
        x_label="QoE Gap to Optimal Policy (log 10)",
        center_line=False,
        fontsize=int(args.fontsize),
        trace_type_labels=trace_type_labels,
    )
    _plot_boxplot(
        sns=sns,
        plt=plt,
        np=np,
        out_path=_figure_path(out_dir, "qoe_gap_log10_1p_box", output_format),
        df=df_tr,
        value_col="qoe_gap_log10_1p",
        trace_types=trace_types,
        title="",
        x_label="Performance Gap to Reference Policy (↑, log)",
        center_line=False,
        fontsize=int(args.fontsize),
        trace_type_labels=trace_type_labels,
        x_min=0.0,
    )
    _plot_stats_bars(
        sns=sns,
        plt=plt,
        np=np,
        out_path=_figure_path(out_dir, "qoe_gap_rel_stats", output_format),
        data_df=qoe_gap_rel_stats,
        title="Per-trace Oracle QoE Gap Relative to Clamped Oracle",
        x_label="(oracle − policy) / max(abs(oracle), tau)",
        center_line=True,
        fontsize=int(args.fontsize),
        fmt=".2f",
        trace_type_labels=trace_type_labels,
    )
    _plot_boxplot(
        sns=sns,
        plt=plt,
        np=np,
        out_path=_figure_path(out_dir, "qoe_gap_rel_box", output_format),
        df=df_tr,
        value_col="qoe_gap_rel",
        trace_types=trace_types,
        title="Per-trace Oracle QoE Gap Relative Distribution",
        x_label="(oracle − policy) / max(abs(oracle), tau)",
        center_line=True,
        fontsize=int(args.fontsize),
        trace_type_labels=trace_type_labels,
    )
    _plot_stats_bars(
        sns=sns,
        plt=plt,
        np=np,
        out_path=_figure_path(out_dir, "regret_pct_stats", output_format),
        data_df=regret_pct_stats,
        title="Per-step Regret Percentage",
        x_label="100 * (oracle_k − policy_k) / oracle_k",
        center_line=True,
        fontsize=int(args.fontsize),
        fmt=".1f",
        trace_type_labels=trace_type_labels,
    )
    _plot_mean_ci_bars(
        sns=sns,
        plt=plt,
        np=np,
        out_path=_figure_path(out_dir, "regret_pct_bar_ci95", output_format),
        data_df=regret_pct_ci_stats,
        title="Per-step Regret Percentage Mean ± 95% CI",
        x_label="100 * (oracle_k − policy_k) / oracle_k",
        center_line=True,
        fontsize=int(args.fontsize),
        trace_type_labels=trace_type_labels,
    )
    _plot_boxplot(
        sns=sns,
        plt=plt,
        np=np,
        out_path=_figure_path(out_dir, "regret_pct_box", output_format),
        df=df_st,
        value_col="regret_pct",
        trace_types=trace_types,
        title="Per-step Regret Percentage Distribution",
        x_label="100 * (oracle_k − policy_k) / oracle_k",
        center_line=True,
        fontsize=int(args.fontsize),
        trace_type_labels=trace_type_labels,
    )
    _plot_stats_bars(
        sns=sns,
        plt=plt,
        np=np,
        out_path=_figure_path(out_dir, "regret_stats", output_format),
        data_df=regret_stats,
        title="Per-step Regret (OracleK − PolicyK)",
        x_label="Regret (oracle_k − policy_k, per step)",
        center_line=True,
        fontsize=int(args.fontsize),
        fmt=".2f",
        trace_type_labels=trace_type_labels,
    )
    _plot_mean_ci_bars(
        sns=sns,
        plt=plt,
        np=np,
        out_path=_figure_path(out_dir, "regret_bar_ci95", output_format),
        data_df=regret_ci_stats,
        title="Per-step Regret Mean ± 95% CI",
        x_label="Regret (oracle_k − policy_k, per step)",
        center_line=True,
        fontsize=int(args.fontsize),
        trace_type_labels=trace_type_labels,
    )
    _plot_boxplot(
        sns=sns,
        plt=plt,
        np=np,
        out_path=_figure_path(out_dir, "regret_box", output_format),
        df=df_st,
        value_col="regret",
        trace_types=trace_types,
        title="Per-step Regret Distribution",
        x_label="Regret (oracle_k − policy_k, per step)",
        center_line=True,
        fontsize=int(args.fontsize),
        trace_type_labels=trace_type_labels,
    )
    _plot_stats_bars(
        sns=sns,
        plt=plt,
        np=np,
        out_path=_figure_path(out_dir, "regret_log10_1p_stats", output_format),
        data_df=regret_log10_stats,
        title="",
        x_label="Performance Gap over Reference Policy (↑) [%]",
        center_line=True,
        fontsize=int(args.fontsize),
        fmt=".1f",
        trace_type_labels=trace_type_labels,
    )
    _plot_mean_ci_bars(
        sns=sns,
        plt=plt,
        np=np,
        out_path=_figure_path(out_dir, "regret_log10_1p_bar_ci95", output_format),
        data_df=regret_log10_ci_stats,
        title="",
        x_label="Performance Gap to Reference Policy (log 10)",
        center_line=False,
        fontsize=int(args.fontsize),
        trace_type_labels=trace_type_labels,
    )
    _plot_boxplot(
        sns=sns,
        plt=plt,
        np=np,
        out_path=_figure_path(out_dir, "regret_log10_1p_box", output_format),
        df=df_st,
        value_col="regret_log10_1p",
        trace_types=trace_types,
        title="Per-step Regret log10(1 + regret) Distribution",
        x_label="log10(1 + regret)",
        center_line=False,
        fontsize=int(args.fontsize),
        trace_type_labels=trace_type_labels,
    )
    _plot_stats_bars(
        sns=sns,
        plt=plt,
        np=np,
        out_path=_figure_path(out_dir, "regret_rel_stats", output_format),
        data_df=regret_rel_stats,
        title="Per-step Regret Relative to Clamped Oracle K-step Value",
        x_label="(oracle_k − policy_k) / max(abs(oracle_k), tau)",
        center_line=True,
        fontsize=int(args.fontsize),
        fmt=".2f",
        trace_type_labels=trace_type_labels,
    )
    _plot_boxplot(
        sns=sns,
        plt=plt,
        np=np,
        out_path=_figure_path(out_dir, "regret_rel_box", output_format),
        df=df_st,
        value_col="regret_rel",
        trace_types=trace_types,
        title="Per-step Relative Regret Distribution",
        x_label="(oracle_k − policy_k) / max(abs(oracle_k), tau)",
        center_line=True,
        fontsize=int(args.fontsize),
        trace_type_labels=trace_type_labels,
    )
    if bool(has_timing_cols) and decision_time_ms_ci_stats is not None and decision_time_rel_ci_stats is not None:
        _plot_mean_ci_bars(
            sns=sns,
            plt=plt,
            np=np,
            out_path=_figure_path(out_dir, "decision_time_ms_bar_ci95", output_format),
            data_df=decision_time_ms_ci_stats,
            title="",
            x_label="Controller Inference Time [ms]",
            center_line=False,
            fontsize=int(args.fontsize),
            trace_type_labels=DECISION_TIME_PROTECTION_LABELS,
            sort_by_mean=False,
        )
        _plot_boxplot(
            sns=sns,
            plt=plt,
            np=np,
            out_path=_figure_path(out_dir, "decision_time_ms_box", output_format),
            df=df_st,
            value_col="decision_time_ms",
            trace_types=trace_types,
            title="Per-step Controller Time Distribution",
            x_label="model_time_ms + shield_time_ms",
            center_line=False,
            fontsize=int(args.fontsize),
            trace_type_labels=trace_type_labels,
            x_min=0.0,
        )
        _plot_mean_ci_bars(
            sns=sns,
            plt=plt,
            np=np,
            out_path=_figure_path(out_dir, "decision_time_rel_bar_ci95", output_format),
            data_df=decision_time_rel_ci_stats,
            title="",
            x_label="Controller Latency Relative to Inference Budget [%]",
            center_line=False,
            fontsize=int(args.fontsize),
            trace_type_labels=DECISION_TIME_PROTECTION_LABELS,
            sort_by_mean=False,
            xlim_override=(0.0, 100.0),
            xticks_override=[0.0, 20.0, 40.0, 60.0, 80.0, 100.0],
            clip_data_artists_to_axes=True,
        )
        _plot_boxplot(
            sns=sns,
            plt=plt,
            np=np,
            out_path=_figure_path(out_dir, "decision_time_rel_box", output_format),
            df=df_st,
            value_col="decision_time_rel_pct",
            trace_types=trace_types,
            title="Per-step Relative Controller Time Percentage Distribution",
            x_label="100 * (model_time_ms + shield_time_ms) / (delay_ms + sleep_time_ms)",
            center_line=False,
            fontsize=int(args.fontsize),
            trace_type_labels=trace_type_labels,
            x_min=0.0,
        )
    if int(len(tradeoff_plot_df)) > 0:
        _plot_shield_tradeoff(
            sns=sns,
            plt=plt,
            np=np,
            out_path=_figure_path(out_dir, "shield_tradeoff_gap_reduction_vs_clean_gap_ci95", output_format),
            tradeoff_df=tradeoff_plot_df,
            title="",
            x_label="Performance Degradation Normal Trace (↓) [%]",
            y_label="Performance Gap Reduction (↑) [%]",
            fontsize=int(args.fontsize),
        )
    else:
        _eprint("* Skipping shield tradeoff plot; no matched shield/clean/unshielded trace groups were found.")
    if int(len(tradeoff_by_method_df)) > 0:
        _plot_shield_tradeoff_by_method(
            sns=sns,
            plt=plt,
            np=np,
            out_path=_figure_path(out_dir, "shield_tradeoff_gap_reduction_vs_clean_gap_ci95_by_method", output_format),
            tradeoff_df=tradeoff_by_method_df,
            title="",
            x_label="Performance Degradation Normal Trace (↓) [%]",
            y_label="Performance Gap Reduction (↑) [%]",
            fontsize=int(args.fontsize),
        )
    else:
        _eprint("* Skipping protection-method tradeoff plot; no matched method-level groups were found.")
    if int(len(tradeoff_breakdown_df)) > 0:
        _plot_shield_tradeoff_breakdown_boxplots(
            sns=sns,
            plt=plt,
            np=np,
            out_dir=out_dir / "protections",
            output_format=str(output_format),
            breakdown_df=tradeoff_breakdown_df,
            fontsize=int(args.fontsize),
        )
        _plot_shield_tradeoff_breakdown_ci_bars(
            sns=sns,
            plt=plt,
            np=np,
            out_dir=out_dir / "protections",
            output_format=str(output_format),
            breakdown_df=tradeoff_breakdown_df,
            fontsize=int(args.fontsize),
            n_boot=int(ci_bootstrap_samples),
            ci_pct=float(ci_pct),
        )
        _plot_shield_tradeoff_breakdown_adversarial_combined_bars(
            sns=sns,
            plt=plt,
            np=np,
            out_dir=out_dir / "protections",
            output_format=str(output_format),
            breakdown_df=tradeoff_breakdown_df,
            fontsize=int(args.fontsize),
            n_boot=int(ci_bootstrap_samples),
            ci_pct=float(ci_pct),
        )
        _plot_shield_tradeoff_breakdown_side_by_side_ci(
            sns=sns,
            plt=plt,
            np=np,
            out_dir=out_dir / "protections",
            output_format=str(output_format),
            breakdown_df=tradeoff_breakdown_df,
            fontsize=int(args.fontsize),
            n_boot=int(ci_bootstrap_samples),
            ci_pct=float(ci_pct),
        )
    else:
        _eprint("* Skipping shield tradeoff breakdown boxplots; no matched protection groups were found.")

    _eprint(f"* Wrote summary CSV: {summary_path}")
    if tradeoff_summary_path is not None:
        _eprint(f"* Wrote shield tradeoff CSV: {tradeoff_summary_path}")
    if tradeoff_by_method_summary_path is not None:
        _eprint(f"* Wrote protection-method tradeoff CSV: {tradeoff_by_method_summary_path}")
    if tradeoff_breakdown_path is not None:
        _eprint(f"* Wrote shield tradeoff breakdown CSV: {tradeoff_breakdown_path}")
    _eprint(f"* Wrote plots under: {out_dir}")


if __name__ == "__main__":
    main()
