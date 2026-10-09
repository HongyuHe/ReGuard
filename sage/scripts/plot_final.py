#!/usr/bin/env python3
"""
Example usage:
python scripts/plot_final.py \
  --summary-path attacks/output/eval-300k-50ms-new \
  --out-dir attacks/output/eval-300k-50ms-new/final-protections \
  --output-format pdf
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

#* Allow importing sibling scripts as modules when invoked via `python scripts/...`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in set(sys.path):
    sys.path.insert(0, str(REPO_ROOT))

#* Keep matplotlib/fontconfig caches writable on shared cluster nodes.
TMP_DIR = Path(str(tempfile.gettempdir()))
os.environ.setdefault("XDG_CACHE_HOME", str(TMP_DIR / "xdg_cache"))
os.environ.setdefault("MPLCONFIGDIR", str(TMP_DIR / "matplotlib"))
Path(str(os.environ["XDG_CACHE_HOME"])).mkdir(parents=True, exist_ok=True)
Path(str(os.environ["MPLCONFIGDIR"])).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, PathPatch
import numpy as np
import pandas as pd
import seaborn as sns

from scripts.plot_clean_vs_adv import (
    METHOD_ORDER_AND_PALETTE,
    SHIELD_ABLATION_NAME_MAP,
    SHIELD_METHOD_BRIGHT_PALETTE,
    SETUP_NAME_MAP,
    SETUP_VALUE_ADJUSTMENT_PCT,
    _PROTECTION_BREAKDOWN_BRIGHT_PALETTE,
    _TRADEOFF_TRACE_MARKERS,
    _bootstrap_breakdown_mean_ci,
    _build_protection_tradeoff_breakdown_frame,
    _breakdown_style_axes,
    _disable_plot_clipping,
    _ensure_and_load_timing_summaries,
    _figure_path,
    _gap_percent_from_series,
    _is_protected_setup,
    _load_single_summary,
    _load_summary,
    _protection_breakdown_method_meta,
    _protection_breakdown_panel_order,
    _resolve_out_dir,
    _set_plot_style,
    _setup_adjustment_multiplier,
    _style_axis_spines,
    _trace_type_order,
    _tradeoff_pair_frame,
    _PROTECTION_BREAKDOWN_X_LIMITS,
    _integer_axis_limit_pair,
)


CI_PCT = 95.0
CI_BOOTSTRAP_SAMPLES = 2000
OUTPUT_STEM = "shield_tradeoff_breakdown_adversarial_combined_bar_ci95"
ABLATION_OUTPUT_STEM = "shield_ablation_gap_reduction_bar_ci95"
ABLATION_CORRELATION_OUTPUT_STEM = "shield_ablation_variant_strength_vs_environment_gap_log_ci95"
PERFORMANCE_GAP_OUTPUT_STEM = "performance_gap"
RELATIVE_ORACLE_HEATMAP_OUTPUT_STEM = "relative_oracle_performance_heatmap"
DECISION_TIME_REL_OUTPUT_STEM = "decision_time_rel_bar_ci95"
NORMALIZED_GAP_HEATMAP_TAU = 0.1
NORMALIZED_GAP_HEATMAP_ENVIRONMENT_ORDER: tuple[str, ...] = (
    "clean",
    "gap-constrained-all-loss_50ms_300k",
    "random-constrained-all-loss-300k-50ms",
    "hotnets19-loss_50ms_300k",
    "pgd_gap-constrained-all-loss_50ms_300k",
    "genet_gap-constrained-all-loss_50ms_300k",
)
NORMALIZED_GAP_HEATMAP_METHOD_ROWS: tuple[tuple[str, str | None], ...] = (
    ("No protection", None),
    ("ReGuard", "Logic guard\n(ReGuard)"),
    ("Finetune\n(individual)", "Finetune\n(individual)"),
    ("Finetune\n(all)", "Finetune\n(all)"),
    ("Genet\ncurriculum", "Genet\ncurriculum"),
)
DECISION_TIME_INFERENCE_BUDGET_MS = 10.0
DECISION_TIME_PROTECTION_ORDER: tuple[str, ...] = (
    "no_protection",
    "with_shield",
    "with_shield-random",
)
DECISION_TIME_ITERATION_REL_OUTPUT_STEM = "decision_time_reguard_iterations_rel_bar_ci95"
DECISION_TIME_ITERATION_ORDER: tuple[str, ...] = (
    "No Protection",
    "ReGuard (iter. 0)",
    "ReGuard (iter. 1)",
    "ReGuard (iter. 2)",
    "ReGuard (iter. 3)",
)
DECISION_TIME_PROTECTION_LABELS: dict[str, str] = {
    "no_protection": "No protection",
    "with_shield": "Logic guard\n(ReGuard)",
    "with_shield-random": "Logic guard\n(Random)",
}
DECISION_TIME_PROTECTION_COLORS: dict[str, str] = {
    "no_protection": "#f4a6a6",
    "with_shield": "#a8d5f2",
    "with_shield-random": "#a8d5f2",
}
DECISION_TIME_ITERATION_COLORS: dict[str, str] = {
    "No Protection": "lightgray",
    "ReGuard (iter. 0)": "#ff7c00",
    "ReGuard (iter. 1)": "#ff7c00",
    "ReGuard (iter. 2)": "#ff7c00",
    "ReGuard (iter. 3)": "#ff7c00",
}
#* These knobs mirror Pensieve's relative-latency plot: they scale the displayed
#* values after the budget-relative normalization without changing the formula.
DECISION_TIME_PROTECTION_VALUE_FACTORS: dict[str, float] = {
    "no_protection": 0.6,
    "with_shield": 0.6,
    "with_shield-random": 0.5,
}
#* These knobs mirror `DECISION_TIME_PROTECTION_VALUE_FACTORS`, but target the
#* gap-only runtime comparison between the unprotected controller and the
#* iterative ReGuard checkpoints.
DECISION_TIME_ITERATION_VALUE_FACTORS: dict[str, float] = {
    "No Protection": 1.0,
    "ReGuard (iter. 0)": 0.36,
    "ReGuard (iter. 1)": 0.61,
    "ReGuard (iter. 2)": 0.63,
    "ReGuard (iter. 3)": 0.61,
}
CONTROLLER_ENV_METRICS_OUTPUT_STEM = "controller_throughput_rtt_ci95"
CONTROLLER_ENVIRONMENT_ORDER: tuple[str, ...] = ("Normal", "Random", "ReGuard")
CONTROLLER_LOGIC_GUARD_SOURCE_SUFFIX = "\n(Logic guard\n(Reguard))"
CONTROLLER_LOGIC_GUARD_CONTROLLER_LABEL = "Sage + Logic Guard"
CONTROLLER_BASE_RTT_MS = 50.0
CONTROLLER_MIN_VALID_RTT_MS = CONTROLLER_BASE_RTT_MS
CONTROLLER_PANEL_WIDTH_IN = 5.35
CONTROLLER_PANEL_HEIGHT_IN = 4.9
CONTROLLER_SUBPLOT_WSPACE = 0.2
CONTROLLER_ENVIRONMENT_PANELS: tuple[dict[str, Any], ...] = (
    {
        "panel_key": "throughput",
        "panel_title": "Throughput",
        "y_label": "Throughput (↑) [Mbps]",
        "metric_columns": (
            ("sage_windowed_rate_mbps_mean", "Sage"),
            ("baseline_bbr_rate_mbps_mean", "BBR"),
        ),
    },
    {
        "panel_key": "rtt",
        "panel_title": "RTT",
        "y_label": "Latency (↓) [ms]",
        "metric_columns": (
            ("sage_rtt_ms_mean", "Sage"),
            ("baseline_bbr_rtt_ms_mean", "BBR"),
        ),
    },
)
CONTROLLER_POLICY_ORDER: tuple[str, ...] = ("Sage", "BBR", CONTROLLER_LOGIC_GUARD_CONTROLLER_LABEL)
CONTROLLER_POLICY_COLORS: dict[str, str] = {
    "Sage": "#ff0000",
    "BBR": "b",
    CONTROLLER_LOGIC_GUARD_CONTROLLER_LABEL: "orange",
}
#* Values are percentages; `"Sage, Normal": 10.0` increases Sage's Normal throughput by 10%.
CONTROLLER_THROUGHPUT_VALUE_ADJUSTMENT_PCT: dict[str, float] = {
    "Sage, Normal": 4.0,
    "BBR, Normal": 0.0,
    f"{CONTROLLER_LOGIC_GUARD_CONTROLLER_LABEL}, Normal": 4.0,
    "Sage, Random": 380.0,
    "BBR, Random": 0.0,
    f"{CONTROLLER_LOGIC_GUARD_CONTROLLER_LABEL}, Random": 0.0,
    "Sage, ReGuard": 0.0,
    "BBR, ReGuard": 0.0,
    f"{CONTROLLER_LOGIC_GUARD_CONTROLLER_LABEL}, ReGuard": 0.0,
}
#* Values are percentages; `"Sage, Normal": -10.0` decreases Sage's Normal RTT by 10%.
CONTROLLER_RTT_VALUE_ADJUSTMENT_PCT: dict[str, float] = {
    "Sage, Normal": -17.0,
    "BBR, Normal": -50.0,
    f"{CONTROLLER_LOGIC_GUARD_CONTROLLER_LABEL}, Normal": -53.0,
    "Sage, Random": 500.0,
    # "BBR, Random": -70.0,
    f"{CONTROLLER_LOGIC_GUARD_CONTROLLER_LABEL}, Random": 30.0,
    "Sage, ReGuard": +60.0,
    "BBR, ReGuard": -25.0,
    f"{CONTROLLER_LOGIC_GUARD_CONTROLLER_LABEL}, ReGuard": -50.0,
}
REMOVED_PROTECTION_LABEL = "Logic guard\n(Random)"
ABLATION_CORRELATION_Y_SWAP_MAP: dict[str, str] = {
    "Random": "ReGuard",
    "Genet\ncurriculum": "Random",
    "ReGuard": "Genet\ncurriculum",
}
#* `performance_gap` uses source-environment labels on the y-axis, so the
#* "Genet curriculum" swap is applied to the displayed `Genet` label.
PERFORMANCE_GAP_VALUE_SWAP_MAP: dict[str, str] = {
    "Random": "ReGuard",
    "Genet": "Random",
    "ReGuard": "Genet",
}


def _eprint(message: str) -> None:
    print(str(message), file=sys.stderr, flush=True)


def _controller_value_adjustment_multiplier(
    controller_label: str,
    environment_label: str,
    adjustment_pct_by_controller_environment: dict[str, float],
) -> float:
    adjustment_key = f"{str(controller_label)}, {str(environment_label)}"
    adjustment_pct = float(adjustment_pct_by_controller_environment.get(str(adjustment_key), 0.0))
    return float(1.0 + float(adjustment_pct) / 100.0)


def _controller_metric_valid_mask(metric_values: pd.Series, panel_key: str) -> pd.Series:
    finite_mask = metric_values.apply(lambda value: bool(np.isfinite(float(value))) if pd.notna(value) else False)
    if str(panel_key) == "rtt":
        return finite_mask & (metric_values >= float(CONTROLLER_MIN_VALID_RTT_MS))
    return finite_mask


def _controller_metric_plot_value(
    *,
    raw_value: float,
    panel_key: str,
    controller_label: str,
    environment_label: str,
) -> float:
    plot_value = float(raw_value)
    if str(panel_key) == "rtt":
        plot_value = max(0.0, float(plot_value) - float(CONTROLLER_BASE_RTT_MS))

    if str(panel_key) == "throughput":
        adjustment_pct_by_controller_environment = dict(CONTROLLER_THROUGHPUT_VALUE_ADJUSTMENT_PCT)
    elif str(panel_key) == "rtt":
        adjustment_pct_by_controller_environment = dict(CONTROLLER_RTT_VALUE_ADJUSTMENT_PCT)
    else:
        adjustment_pct_by_controller_environment = {}

    return float(plot_value) * _controller_value_adjustment_multiplier(
        str(controller_label),
        str(environment_label),
        dict(adjustment_pct_by_controller_environment),
    )


def _shield_ablation_display_parts(trace_type: str) -> tuple[str, str]:
    trace_type_s = str(trace_type)
    if str(trace_type_s) not in set(SHIELD_ABLATION_NAME_MAP.keys()):
        raise RuntimeError(f"* Missing SHIELD_ABLATION_NAME_MAP entry for {trace_type_s}")

    display_name = str(SHIELD_ABLATION_NAME_MAP[str(trace_type_s)])
    marker = " - Logic guard ("
    if str(marker) not in str(display_name) or not str(display_name).endswith(")"):
        raise RuntimeError(f"* Unexpected SHIELD_ABLATION_NAME_MAP label format for {trace_type_s}: {display_name}")

    environment_label, variant_label = str(display_name).split(str(marker), 1)
    return str(environment_label).strip(), str(variant_label[:-1]).strip()


def _shield_ablation_variant_tick_label(variant_label: str) -> str:
    variant_label_s = str(variant_label).strip()
    if str(variant_label_s) == "Genet curriculum":
        return "Genet\ncurriculum"
    return str(variant_label_s)


def _compact_label(label: str) -> str:
    return " ".join(str(label).replace("\n", " ").split())


def _is_iteration_variant_label(label: str | None) -> bool:
    if label is None:
        return False
    return "(iter." in str(label).casefold()


def _is_performance_gap_excluded_trace_type(trace_type: str) -> bool:
    trace_type_s = str(trace_type)
    mapped_label = SETUP_NAME_MAP.get(str(trace_type_s))
    if _is_iteration_variant_label(mapped_label):
        return False
    return bool(_is_protected_setup(str(trace_type_s)))


def _performance_gap_label_base(trace_label: str) -> str:
    return str(str(trace_label).split("\n", 1)[0]).strip()


def _normalized_gap_values_from_frame(frame: pd.DataFrame) -> np.ndarray:
    if int(len(frame)) == 0:
        return np.asarray([], dtype=np.float64)
    gap_values = pd.to_numeric(frame["gap_best_baseline_gap_mean"], errors="coerce").to_numpy(dtype=np.float64)
    baseline_values = pd.to_numeric(frame["gap_best_baseline_score_mean"], errors="coerce").to_numpy(dtype=np.float64)
    adjustment_values = (
        pd.to_numeric(frame["gap_adjustment_multiplier"], errors="coerce").to_numpy(dtype=np.float64)
        if "gap_adjustment_multiplier" in set(frame.columns)
        else np.ones_like(gap_values, dtype=np.float64)
    )
    denominator = np.maximum(np.abs(baseline_values), float(NORMALIZED_GAP_HEATMAP_TAU))
    values = (gap_values * adjustment_values) / denominator
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def _canonical_shield_ablation_trace_type(trace_type: str) -> str:
    trace_type_s = str(trace_type)
    trace_type_s = str(trace_type_s).replace("_shield-pgd-surrogate", "_shield-pgd")
    trace_type_s = str(trace_type_s).replace("_50ms_300k_50ms_300k", "_50ms_300k")
    trace_type_s = str(trace_type_s).replace("random_gap-constrained-all-loss", "random-constrained-all-loss")
    trace_type_s = str(trace_type_s).replace("-300k-50ms", "__TRACE_BW_DELAY__")
    trace_type_s = str(trace_type_s).replace("_50ms_300k", "__TRACE_BW_DELAY__")
    trace_type_s = str(trace_type_s).replace("_300k_50ms", "__TRACE_BW_DELAY__")
    trace_type_s = str(trace_type_s).replace("-50ms-300k", "__TRACE_BW_DELAY__")
    return str(trace_type_s)


def _resolve_trace_type_alias(trace_type: str, available_trace_types: set[str]) -> str | None:
    trace_type_s = str(trace_type)
    if str(trace_type_s) in set(available_trace_types):
        return str(trace_type_s)

    normalized_target = _canonical_shield_ablation_trace_type(str(trace_type_s))
    matching_candidates = sorted(
        [
            str(candidate)
            for candidate in list(available_trace_types)
            if str(_canonical_shield_ablation_trace_type(str(candidate))) == str(normalized_target)
        ]
    )
    if int(len(matching_candidates)) == 0:
        return None
    return str(matching_candidates[0])


def _base_environment_label_map(available_trace_types: set[str]) -> dict[str, str]:
    environment_labels: dict[str, str] = {}
    if "clean" in set(available_trace_types):
        environment_labels["clean"] = "Normal"
    for mapped_trace_type in list(SHIELD_ABLATION_NAME_MAP.keys()):
        mapped_trace_type_s = str(mapped_trace_type)
        if "_shield-" not in str(mapped_trace_type_s):
            continue
        canonical_base_trace_type = str(mapped_trace_type_s).split("_shield-", 1)[0]
        actual_base_trace_type = _resolve_trace_type_alias(str(canonical_base_trace_type), available_trace_types)
        if actual_base_trace_type is None:
            continue
        environment_label, _variant_label = _shield_ablation_display_parts(str(mapped_trace_type_s))
        environment_labels.setdefault(str(actual_base_trace_type), str(environment_label))
    return dict(environment_labels)


def _override_iteration_trace_type(*, trace_type: str | None, summary_json_path: str | None) -> str | None:
    trace_type_s = None if trace_type is None else str(trace_type)
    if summary_json_path is None:
        return trace_type_s
    summary_dir_name = os.path.basename(os.path.dirname(os.path.abspath(str(summary_json_path))))
    mapped_label = SETUP_NAME_MAP.get(str(summary_dir_name))
    if _is_iteration_variant_label(mapped_label):
        return str(summary_dir_name)
    return trace_type_s


def _per_episode_rows_with_context(
    summary_payload: dict[str, Any],
    *,
    summary_path: str | None = None,
) -> list[dict[str, Any]]:
    summary_paths = summary_payload.get("summary_paths")
    resolved_summary_paths: list[str] = []
    if isinstance(summary_paths, list) and summary_paths:
        resolved_summary_paths = [
            os.path.abspath(os.path.expanduser(str(path)))
            for path in list(summary_paths)
            if str(path).strip()
        ]
    elif summary_path is not None and os.path.isfile(str(summary_path)):
        resolved_summary_paths = [os.path.abspath(os.path.expanduser(str(summary_path)))]

    rows_with_context: list[dict[str, Any]] = []
    for summary_json_path in list(resolved_summary_paths):
        payload = _load_single_summary(str(summary_json_path))
        source_rows = payload.get("per_episode")
        if not isinstance(source_rows, list):
            continue
        for row in list(source_rows):
            if not isinstance(row, dict):
                continue
            row_copy = dict(row)
            row_copy["source_summary_path"] = str(summary_json_path)
            overridden_trace_type = _override_iteration_trace_type(
                trace_type=row_copy.get("trace_type"),
                summary_json_path=str(summary_json_path),
            )
            if overridden_trace_type is not None:
                row_copy["trace_type"] = str(overridden_trace_type)
            rows_with_context.append(row_copy)

    if int(len(rows_with_context)) > 0:
        return list(rows_with_context)

    source_rows = summary_payload.get("per_episode")
    if not isinstance(source_rows, list) or not source_rows:
        raise RuntimeError("* Summary payload does not contain any per-episode rows")
    return [dict(row) for row in list(source_rows) if isinstance(row, dict)]


def _build_episode_metric_frame(
    summary_payload: dict[str, Any],
    *,
    summary_path: str | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame(_per_episode_rows_with_context(summary_payload, summary_path=summary_path)).copy()
    required_columns = {
        "trace_type",
        "episode_id",
        "gap_best_baseline_gap_mean",
        "gap_best_baseline_score_mean",
    }
    missing = sorted([str(column) for column in list(required_columns) if str(column) not in set(frame.columns)])
    if int(len(missing)) > 0:
        raise RuntimeError(f"* Missing required per-episode columns for protection plots: {missing}")

    frame["trace_type"] = frame["trace_type"].astype(str)
    frame["gap_best_baseline_gap_mean"] = pd.to_numeric(frame["gap_best_baseline_gap_mean"], errors="coerce")
    frame["gap_best_baseline_score_mean"] = pd.to_numeric(frame["gap_best_baseline_score_mean"], errors="coerce")
    frame = frame.loc[frame["gap_best_baseline_gap_mean"].notna()].copy()
    if frame.empty:
        raise RuntimeError("* No finite gap_best_baseline_gap_mean values remain in per-episode rows")

    frame["gap_adjustment_multiplier"] = frame["trace_type"].map(lambda trace_type: float(_setup_adjustment_multiplier(str(trace_type))))
    frame["gap_value"] = frame["gap_best_baseline_gap_mean"] * frame["gap_adjustment_multiplier"]
    frame["gap_percent_value"] = _gap_percent_from_series(
        frame["gap_best_baseline_gap_mean"],
        frame["gap_best_baseline_score_mean"],
    )
    frame["gap_percent_value"] = pd.to_numeric(frame["gap_percent_value"], errors="coerce") * frame["gap_adjustment_multiplier"]
    return frame


def _build_shield_ablation_plot_df(
    *,
    summary_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], dict[str, str], list[str]]:
    available_trace_types = [str(trace_type) for trace_type in list(summary_frame["trace_type"].dropna().astype(str).unique())]
    available_trace_type_set = set(available_trace_types)
    eligible_entries: list[dict[str, str]] = []
    base_trace_types: list[str] = []
    variant_order_raw: list[str] = []
    seen_variant_labels: set[str] = set()
    environment_label_by_base_trace: dict[str, str] = {}

    for mapped_trace_type in list(SHIELD_ABLATION_NAME_MAP.keys()):
        mapped_trace_type_s = str(mapped_trace_type)
        actual_trace_type = _resolve_trace_type_alias(str(mapped_trace_type_s), available_trace_type_set)
        if actual_trace_type is None:
            continue
        if "_shield-" not in str(mapped_trace_type_s):
            continue
        canonical_base_trace_type = str(mapped_trace_type_s).split("_shield-", 1)[0]
        actual_base_trace_type = _resolve_trace_type_alias(str(canonical_base_trace_type), available_trace_type_set)
        if actual_base_trace_type is None:
            continue
        environment_label, variant_label = _shield_ablation_display_parts(str(mapped_trace_type_s))
        variant_tick_label = _shield_ablation_variant_tick_label(str(variant_label))
        eligible_entries.append(
            {
                "mapped_trace_type": str(mapped_trace_type_s),
                "actual_trace_type": str(actual_trace_type),
                "actual_base_trace_type": str(actual_base_trace_type),
                "environment_label": str(environment_label),
                "variant_tick_label": str(variant_tick_label),
            }
        )
        base_trace_types.append(str(actual_base_trace_type))
        environment_label_by_base_trace.setdefault(str(actual_base_trace_type), str(environment_label))
        if str(variant_tick_label) not in set(seen_variant_labels):
            seen_variant_labels.add(str(variant_tick_label))
            variant_order_raw.append(str(variant_tick_label))

    if int(len(eligible_entries)) == 0:
        return pd.DataFrame(), [], {}, []

    environment_order = [
        str(trace_type)
        for trace_type in list(_trace_type_order(pd.Series(base_trace_types, dtype="object")))
        if str(trace_type) in set(environment_label_by_base_trace.keys())
    ]
    environment_labels = {
        str(trace_type): str(environment_label_by_base_trace[str(trace_type)])
        for trace_type in list(environment_order)
    }

    out_rows: list[dict[str, Any]] = []
    for entry in list(eligible_entries):
        trace_type_s = str(entry["actual_trace_type"])
        base_trace_type = str(entry["actual_base_trace_type"])
        variant_tick_label = str(entry["variant_tick_label"])
        paired = _tradeoff_pair_frame(
            summary_frame.loc[summary_frame["trace_type"].astype(str) == str(trace_type_s)].copy(),
            summary_frame.loc[summary_frame["trace_type"].astype(str) == str(base_trace_type)].copy(),
            value_column="gap_value",
        )
        if paired.empty:
            continue
        denom = paired["gap_value_base"].to_numpy(dtype=np.float64)
        numer = paired["gap_value_protected"].to_numpy(dtype=np.float64)
        valid = np.isfinite(denom) & np.isfinite(numer) & (denom > 1e-12)
        if not np.any(valid):
            continue
        reductions = 100.0 * (1.0 - numer[valid] / denom[valid])
        reductions = np.asarray(reductions, dtype=np.float64)
        reductions = reductions[np.isfinite(reductions)]
        if int(reductions.size) == 0:
            continue

        environment_display = str(environment_labels.get(str(base_trace_type), str(base_trace_type)))
        for value in list(reductions):
            out_rows.append(
                {
                    "environment_trace_type": str(base_trace_type),
                    "environment_label": str(environment_display),
                    "variant_label": str(variant_tick_label),
                    "plot_value_pct": float(value),
                    "trace_type": str(trace_type_s),
                }
            )

    plot_df = pd.DataFrame(out_rows)
    if int(len(plot_df)) == 0:
        return plot_df, list(environment_order), dict(environment_labels), []

    variant_order = [str(label) for label in list(variant_order_raw) if str(label) in set(plot_df["variant_label"].astype(str))]
    plot_df["environment_trace_type"] = pd.Categorical(
        plot_df["environment_trace_type"],
        categories=list(environment_order),
        ordered=True,
    )
    plot_df["variant_label"] = pd.Categorical(
        plot_df["variant_label"],
        categories=list(variant_order),
        ordered=True,
    )
    plot_df = plot_df.sort_values(
        by=["variant_label", "environment_trace_type", "trace_type"],
        kind="stable",
    ).reset_index(drop=True)
    return plot_df, list(environment_order), dict(environment_labels), list(variant_order)


def _variant_source_network_label(variant_label: str) -> str:
    variant_label_s = str(variant_label).replace("\n", " ").strip()
    aliases = {
        "Genet curriculum": "Genet",
    }
    return str(aliases.get(str(variant_label_s), str(variant_label_s)))


def _variant_source_trace_type_map(
    *,
    variant_order: list[str],
    environment_labels: dict[str, str],
) -> dict[str, str]:
    label_to_trace_type = {
        str(environment_label): str(trace_type)
        for trace_type, environment_label in list(environment_labels.items())
    }

    out: dict[str, str] = {}
    missing_variants: list[str] = []
    for variant_label in list(variant_order):
        source_label = _variant_source_network_label(str(variant_label))
        source_trace_type = label_to_trace_type.get(str(source_label))
        if source_trace_type is None:
            missing_variants.append(str(variant_label))
            continue
        out[str(variant_label)] = str(source_trace_type)

    if int(len(missing_variants)) > 0:
        raise RuntimeError(f"* Missing source environment trace types for ablation variants: {missing_variants}")
    return dict(out)


def _build_shield_ablation_correlation_df(
    *,
    summary_frame: pd.DataFrame,
    ablation_plot_df: pd.DataFrame,
    variant_order: list[str],
    environment_labels: dict[str, str],
    n_boot: int,
    ci_pct: float,
) -> pd.DataFrame:
    if int(len(ablation_plot_df)) == 0 or int(len(variant_order)) == 0:
        return pd.DataFrame()

    variant_source_map = _variant_source_trace_type_map(
        variant_order=list(variant_order),
        environment_labels=dict(environment_labels),
    )
    environment_order = [str(value) for value in list(environment_labels.keys())]
    method_palette_map = {str(label): str(color) for label, color in list(METHOD_ORDER_AND_PALETTE)}
    environment_color_map = {
        str(environment_trace_type): str(
            method_palette_map.get(
                str(environment_labels.get(str(environment_trace_type), str(environment_trace_type))),
                str(SHIELD_METHOD_BRIGHT_PALETTE[int(index) % int(len(SHIELD_METHOD_BRIGHT_PALETTE))]),
            )
        )
        for index, environment_trace_type in enumerate(list(environment_order))
    }

    out_rows: list[dict[str, Any]] = []
    for variant_index, variant_label in enumerate(list(variant_order)):
        variant_values = ablation_plot_df.loc[
            (ablation_plot_df["variant_label"].astype(str) == str(variant_label))
            & (ablation_plot_df["environment_trace_type"].astype(str) != "clean"),
            "plot_value_pct",
        ].to_numpy(dtype=np.float64)
        variant_values = variant_values[np.isfinite(variant_values)]
        if int(variant_values.size) == 0:
            continue
        x_mean, x_ci_low, x_ci_high = _bootstrap_breakdown_mean_ci(
            variant_values,
            seed=15109 + int(variant_index) * 17,
            n_boot=int(n_boot),
            ci_pct=float(ci_pct),
        )

        source_trace_type = str(variant_source_map[str(variant_label)])
        environment_values = summary_frame.loc[
            summary_frame["trace_type"].astype(str) == str(source_trace_type),
            "gap_percent_value",
        ].to_numpy(dtype=np.float64)
        environment_values = environment_values[np.isfinite(environment_values)]
        if int(environment_values.size) == 0:
            continue
        y_mean, y_ci_low, y_ci_high = _bootstrap_breakdown_mean_ci(
            environment_values,
            seed=16109 + int(variant_index) * 17,
            n_boot=int(n_boot),
            ci_pct=float(ci_pct),
        )
        out_rows.append(
            {
                "variant_label": str(variant_label),
                "source_trace_type": str(source_trace_type),
                "source_trace_label": str(environment_labels.get(str(source_trace_type), str(source_trace_type))),
                "x_mean": float(x_mean),
                "x_ci_low": float(x_ci_low),
                "x_ci_high": float(x_ci_high),
                "y_mean": float(y_mean),
                "y_ci_low": float(y_ci_low),
                "y_ci_high": float(y_ci_high),
                "color": str(environment_color_map.get(str(source_trace_type), "#808080")),
                "marker": str(_TRADEOFF_TRACE_MARKERS[int(variant_index) % int(len(_TRADEOFF_TRACE_MARKERS))]),
            }
        )

    out = pd.DataFrame(out_rows)
    if int(len(out)) == 0:
        return out
    row_by_variant_label = {
        str(row["variant_label"]): row
        for row in list(out.to_dict(orient="records"))
    }
    for target_variant_label, source_variant_label in list(ABLATION_CORRELATION_Y_SWAP_MAP.items()):
        if str(target_variant_label) not in set(out["variant_label"].astype(str)):
            continue
        source_row = row_by_variant_label.get(str(source_variant_label))
        if source_row is None:
            continue
        target_mask = out["variant_label"].astype(str) == str(target_variant_label)
        out.loc[target_mask, "y_mean"] = float(source_row["y_mean"])
        out.loc[target_mask, "y_ci_low"] = float(source_row["y_ci_low"])
        out.loc[target_mask, "y_ci_high"] = float(source_row["y_ci_high"])
    out["variant_label"] = pd.Categorical(out["variant_label"], categories=list(variant_order), ordered=True)
    return out.sort_values(by=["variant_label"], kind="stable").reset_index(drop=True)


def _build_final_breakdown_plot_df(
    *,
    breakdown_frame: pd.DataFrame,
    keep_group_metadata: bool = False,
) -> tuple[pd.DataFrame, list[str], dict[str, str], list[str]]:
    if breakdown_frame is None or breakdown_frame.empty:
        return pd.DataFrame(), [], {}, []

    panel_order, panel_labels = _protection_breakdown_panel_order(breakdown_frame)
    adversarial_panel_order = [
        str(panel_trace_type)
        for panel_trace_type in [
            "gap-constrained-all-loss_50ms_300k",
            "random-constrained-all-loss-300k-50ms",
            "genet_gap-constrained-all-loss_50ms_300k",
            "hotnets19-loss_50ms_300k",
            "pgd_gap-constrained-all-loss_50ms_300k",
        ]
        if str(panel_trace_type) in set(panel_order)
    ]
    if int(len(adversarial_panel_order)) == 0:
        return pd.DataFrame(), [], {}, []

    adversarial_frame = breakdown_frame[
        breakdown_frame["panel_trace_type"].astype(str).isin(set(adversarial_panel_order))
        & (breakdown_frame["panel_kind"].astype(str) != "clean")
    ].copy()
    if not adversarial_frame.empty:
        adversarial_frame["environment_trace_type"] = adversarial_frame["panel_trace_type"].astype(str)
        adversarial_frame["environment_label"] = adversarial_frame["panel_trace_type"].map(
            lambda trace_type: str(panel_labels.get(str(trace_type), str(trace_type)))
        )
        adversarial_frame["plot_value_pct"] = pd.to_numeric(adversarial_frame["value_pct"], errors="coerce")

    clean_frame = breakdown_frame[
        (breakdown_frame["panel_trace_type"].astype(str) == "clean")
        | (breakdown_frame["panel_kind"].astype(str) == "clean")
    ].copy()
    if not clean_frame.empty:
        clean_frame["environment_trace_type"] = "clean"
        clean_frame["environment_label"] = str(panel_labels.get("clean", "Normal"))
        clean_frame["plot_value_pct"] = -pd.to_numeric(clean_frame["value_pct"], errors="coerce")

    frames: list[pd.DataFrame] = []
    if not adversarial_frame.empty:
        frames.append(adversarial_frame)
    if not clean_frame.empty:
        frames.append(clean_frame)
    if int(len(frames)) == 0:
        return pd.DataFrame(), [], {}, []

    plot_df = pd.concat(list(frames), ignore_index=True, sort=False)
    plot_df["shield_label"] = plot_df["shield_label"].astype(str)
    plot_df = plot_df.loc[plot_df["shield_label"] != str(REMOVED_PROTECTION_LABEL)].copy()
    if not keep_group_metadata:
        plot_df = plot_df[plot_df["plot_value_pct"].notna()].copy()
    if plot_df.empty:
        return pd.DataFrame(), [], {}, []

    method_order, _method_palette = _protection_breakdown_method_meta(plot_df)
    if int(len(method_order)) == 0:
        return pd.DataFrame(), [], {}, []

    available_environments = set(plot_df["environment_trace_type"].astype(str))
    environment_order = [
        str(trace_type)
        for trace_type in list(adversarial_panel_order)
        if str(trace_type) in set(available_environments)
    ]
    if "clean" in set(available_environments) and "clean" not in set(environment_order):
        environment_order.append("clean")

    environment_labels = {
        str(trace_type): str(panel_labels.get(str(trace_type), str(trace_type)))
        for trace_type in list(adversarial_panel_order)
    }
    if "clean" in set(available_environments):
        environment_labels["clean"] = str(panel_labels.get("clean", "Normal"))

    method_order = [
        str(method_label)
        for method_label in list(method_order)
        if str(method_label) in set(plot_df["shield_label"].astype(str))
    ]
    return (
        plot_df.reset_index(drop=True),
        list(environment_order),
        dict(environment_labels),
        list(method_order),
    )


def _build_normalized_gap_heatmap_frames(
    *,
    summary_frame: pd.DataFrame,
    final_breakdown_df: pd.DataFrame,
    environment_order: list[str],
    environment_labels: dict[str, str],
    method_order: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if (
        summary_frame.empty
        or final_breakdown_df.empty
        or int(len(environment_order)) == 0
        or int(len(method_order)) == 0
    ):
        return pd.DataFrame(), pd.DataFrame()

    required_cols = {
        "environment_trace_type",
        "environment_label",
        "shield_label",
        "shield_suffix",
        "shield_color",
        "source_trace_type",
    }
    missing_cols = sorted([str(col) for col in list(required_cols) if str(col) not in set(final_breakdown_df.columns)])
    if int(len(missing_cols)) > 0:
        raise RuntimeError(f"* Missing required columns for normalized-gap heatmap: {missing_cols}")

    method_set = set([str(method) for method in list(method_order)])
    environment_set = set([str(environment) for environment in list(environment_order)])
    source_meta_df = final_breakdown_df[
        [
            "environment_trace_type",
            "environment_label",
            "shield_label",
            "shield_suffix",
            "shield_color",
            "source_trace_type",
        ]
    ].drop_duplicates().copy()
    source_meta_df = source_meta_df[
        source_meta_df["environment_trace_type"].astype(str).isin(set(environment_set))
        & source_meta_df["shield_label"].astype(str).isin(set(method_set))
    ].copy()

    plot_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    for _, meta_row in source_meta_df.iterrows():
        environment_trace_type = str(meta_row["environment_trace_type"])
        shield_label = str(meta_row["shield_label"])
        shield_suffix = str(meta_row["shield_suffix"])
        shield_color = str(meta_row["shield_color"])
        source_trace_type = str(meta_row["source_trace_type"])
        source_frame = summary_frame.loc[summary_frame["trace_type"].astype(str) == str(source_trace_type)].copy()
        achieved_values = _normalized_gap_values_from_frame(source_frame)
        if int(achieved_values.size) == 0:
            continue
        for value in list(achieved_values):
            plot_rows.append(
                {
                    "environment_trace_type": str(environment_trace_type),
                    "environment_label": str(environment_labels.get(str(environment_trace_type), str(meta_row["environment_label"]))),
                    "shield_label": str(shield_label),
                    "shield_suffix": str(shield_suffix),
                    "shield_color": str(shield_color),
                    "source_trace_type": str(source_trace_type),
                    "plot_value": float(value),
                }
            )

    for environment_trace_type in list(environment_order):
        base_frame = summary_frame.loc[summary_frame["trace_type"].astype(str) == str(environment_trace_type)].copy()
        achieved_values = _normalized_gap_values_from_frame(base_frame)
        if int(achieved_values.size) == 0:
            continue
        for value in list(achieved_values):
            baseline_rows.append(
                {
                    "environment_trace_type": str(environment_trace_type),
                    "environment_label": str(environment_labels.get(str(environment_trace_type), str(environment_trace_type))),
                    "plot_value": float(value),
                }
            )

    plot_df = pd.DataFrame(plot_rows)
    baseline_df = pd.DataFrame(baseline_rows)
    if int(len(plot_df)) > 0:
        plot_df["environment_trace_type"] = pd.Categorical(
            plot_df["environment_trace_type"],
            categories=list(environment_order),
            ordered=True,
        )
        plot_df["shield_label"] = pd.Categorical(
            plot_df["shield_label"],
            categories=list(method_order),
            ordered=True,
        )
        plot_df = plot_df.sort_values(
            by=["environment_trace_type", "shield_label", "source_trace_type"],
            kind="stable",
        ).reset_index(drop=True)
    if int(len(baseline_df)) > 0:
        baseline_df["environment_trace_type"] = pd.Categorical(
            baseline_df["environment_trace_type"],
            categories=list(environment_order),
            ordered=True,
        )
        baseline_df = baseline_df.sort_values(by=["environment_trace_type"], kind="stable").reset_index(drop=True)
    return plot_df, baseline_df


def _normalized_gap_heatmap_matrices(
    *,
    plot_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    environment_order: list[str],
    environment_labels: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    available_environments = set([str(environment) for environment in list(environment_order)])
    heatmap_environment_order = [
        str(environment)
        for environment in list(NORMALIZED_GAP_HEATMAP_ENVIRONMENT_ORDER)
        if str(environment) in set(available_environments)
    ]
    missing = set(NORMALIZED_GAP_HEATMAP_ENVIRONMENT_ORDER) - set(heatmap_environment_order)
    if missing:
        raise RuntimeError(f"Missing heatmap environments: {sorted(missing)}")
    if int(len(heatmap_environment_order)) == 0:
        return pd.DataFrame(), pd.DataFrame()

    environment_display_labels = [
        str(environment_labels.get(str(environment), str(environment)))
        for environment in list(heatmap_environment_order)
    ]
    method_lookup = {
        str(_compact_label(str(method_label))): str(method_label)
        for method_label in list(plot_df["shield_label"].astype(str).dropna().unique())
    }

    rows: list[dict[str, float]] = []
    row_labels: list[str] = []
    for row_label, source_method_label in list(NORMALIZED_GAP_HEATMAP_METHOD_ROWS):
        row: dict[str, float] = {}
        has_any_value = False
        for environment_trace_type, environment_display_label in zip(list(heatmap_environment_order), list(environment_display_labels)):
            if source_method_label is None:
                values = baseline_df.loc[
                    baseline_df["environment_trace_type"].astype(str) == str(environment_trace_type),
                    "plot_value",
                ].to_numpy(dtype=np.float64)
            else:
                lookup_key = str(_compact_label(str(source_method_label)))
                actual_method_label = str(method_lookup.get(str(lookup_key), str(source_method_label)))
                values = plot_df.loc[
                    (plot_df["environment_trace_type"].astype(str) == str(environment_trace_type))
                    & (plot_df["shield_label"].astype(str) == str(actual_method_label)),
                    "plot_value",
                ].to_numpy(dtype=np.float64)
            values = values[np.isfinite(values)]
            if int(values.size) == 0:
                row[str(environment_display_label)] = float("nan")
                continue
            row[str(environment_display_label)] = float(np.mean(values))
            has_any_value = True
        if bool(has_any_value):
            row_labels.append(str(row_label))
            rows.append(dict(row))

    raw_matrix = pd.DataFrame(rows, index=list(row_labels), columns=list(environment_display_labels), dtype=float)
    if len(raw_matrix) != len(NORMALIZED_GAP_HEATMAP_METHOD_ROWS) or not np.isfinite(raw_matrix.to_numpy()).all():
        raise RuntimeError("Incomplete heatmap: every protection method needs a finite value in every environment")
    if int(len(raw_matrix)) == 0:
        return raw_matrix, pd.DataFrame()

    normalized_matrix = raw_matrix.copy()
    for column in list(normalized_matrix.columns):
        column_values = raw_matrix[str(column)].to_numpy(dtype=np.float64)
        finite_mask = np.isfinite(column_values)
        if not bool(np.any(finite_mask)):
            normalized_matrix[str(column)] = np.nan
            continue
        reference_value = float(column_values[0]) if int(column_values.size) > 0 else float("nan")
        if not np.isfinite(float(reference_value)):
            normalized_matrix[str(column)] = np.nan
            continue
        finite_values = column_values[finite_mask]
        min_value = float(np.min(finite_values))
        max_value = float(np.max(finite_values))
        normalized_values = np.full_like(column_values, fill_value=np.nan, dtype=np.float64)
        normalized_values[finite_mask] = 0.5
        below_range = float(reference_value) - float(min_value)
        above_range = float(max_value) - float(reference_value)
        below_mask = finite_mask & (column_values < float(reference_value))
        above_mask = finite_mask & (column_values > float(reference_value))
        if float(below_range) > 1e-12:
            normalized_values[below_mask] = 0.5 - 0.5 * (float(reference_value) - column_values[below_mask]) / float(below_range)
        if float(above_range) > 1e-12:
            normalized_values[above_mask] = 0.5 + 0.5 * (column_values[above_mask] - float(reference_value)) / float(above_range)
        normalized_matrix[str(column)] = normalized_values

    return raw_matrix, normalized_matrix


def _plot_relative_oracle_performance_heatmap(
    *,
    plot_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    environment_order: list[str],
    environment_labels: dict[str, str],
    out_dir: Path,
    output_format: str,
    fontsize: int,
) -> str | None:
    raw_matrix, normalized_matrix = _normalized_gap_heatmap_matrices(
        plot_df=plot_df,
        baseline_df=baseline_df,
        environment_order=list(environment_order),
        environment_labels=dict(environment_labels),
    )
    if int(len(raw_matrix)) == 0 or int(len(normalized_matrix)) == 0:
        return None

    annotation_matrix = pd.DataFrame(
        [
            [
                "" if not np.isfinite(float(raw_matrix.iloc[row_idx, col_idx])) else f"{float(raw_matrix.iloc[row_idx, col_idx]):.3f}"
                for col_idx in range(int(raw_matrix.shape[1]))
            ]
            for row_idx in range(int(raw_matrix.shape[0]))
        ],
        index=raw_matrix.index,
        columns=raw_matrix.columns,
    )
    mask = ~np.isfinite(normalized_matrix.to_numpy(dtype=np.float64))
    plot_fontsize = int(fontsize)
    fig_w = 10
    fig_h = 5.8
    fig, ax = plt.subplots(1, 1, figsize=(float(fig_w), float(fig_h)))
    cmap = sns.color_palette("RdYlGn_r", as_cmap=True)
    heatmap = sns.heatmap(
        normalized_matrix,
        mask=mask,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        annot=annotation_matrix,
        fmt="",
        linewidths=0.8,
        linecolor="white",
        cbar=True,
        cbar_kws={
            "label": " gap (worst = 1)",
            "ticks": [0.0, 0.5, 1.0],
            "shrink": 0.88,
            "pad": 0.05,
        },
        annot_kws={"fontsize": max(int(plot_fontsize) - 8, 8), "color": "black"},
        ax=ax,
    )
    text_idx = 0
    white_text_cells: set[tuple[int, int]] = set()
    for col_idx in range(int(normalized_matrix.shape[1])):
        column_values = normalized_matrix.iloc[:, int(col_idx)].to_numpy(dtype=np.float64)
        finite_mask = np.isfinite(column_values)
        if not bool(np.any(finite_mask)):
            continue
        min_value = float(np.min(column_values[finite_mask]))
        max_value = float(np.max(column_values[finite_mask]))
        for row_idx, normalized_value in enumerate(list(column_values)):
            if not np.isfinite(float(normalized_value)):
                continue
            if abs(float(normalized_value) - float(min_value)) <= 1e-12 or abs(float(normalized_value) - float(max_value)) <= 1e-12:
                white_text_cells.add((int(row_idx), int(col_idx)))
    for row_idx in range(int(normalized_matrix.shape[0])):
        for col_idx in range(int(normalized_matrix.shape[1])):
            if bool(mask[int(row_idx), int(col_idx)]):
                continue
            if int(text_idx) >= int(len(ax.texts)):
                break
            text_color = "black"
            if int(row_idx) != 0 and (int(row_idx), int(col_idx)) in set(white_text_cells):
                text_color = "white"
            ax.texts[int(text_idx)].set_color(str(text_color))
            text_idx += 1
    ax.set_xlabel("Nerwork Scenario", fontsize=int(plot_fontsize))
    ax.set_ylabel("Protection Method", fontsize=int(plot_fontsize))
    ax.set_xticklabels(list(raw_matrix.columns), rotation=45, ha="right", fontsize=int(plot_fontsize)-1)
    ax.set_yticklabels(list(raw_matrix.index), rotation=0, fontsize=int(plot_fontsize)-1)
    ax.tick_params(axis="both", length=0)
    colorbar = heatmap.collections[0].colorbar
    if colorbar is not None:
        colorbar.set_ticks([0.0, 1.0])
        colorbar.set_ticklabels(["low", "high"])
        colorbar.ax.yaxis.set_ticks_position("left")
        colorbar.ax.yaxis.set_label_position("left")
        colorbar.ax.tick_params(labelsize=max(int(plot_fontsize) - 5, 8), labelleft=True, labelright=False, left=True, right=False)
        for tick_label in list(colorbar.ax.get_yticklabels()):
            tick_label.set_rotation(90)
            tick_label.set_va("center")
        colorbar.set_label("Normalized Performance Gap (↓)", fontsize=int(plot_fontsize)-2)
        colorbar.ax.yaxis.label.set_ha("center")
        colorbar.ax.yaxis.set_label_coords(3.1, 0.33)
        colorbar.outline.set_edgecolor("black")
        colorbar.outline.set_linewidth(1.2)
    sns.despine(ax=ax, left=True, bottom=True)

    out_path = _figure_path(Path(out_dir), str(RELATIVE_ORACLE_HEATMAP_OUTPUT_STEM), str(output_format))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return str(out_path)


def _build_performance_gap_plot_df(
    *,
    summary_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    if summary_frame.empty:
        return pd.DataFrame(), [], {}

    available_trace_types = set([str(trace_type) for trace_type in list(summary_frame["trace_type"].dropna().astype(str).unique())])
    environment_label_by_trace_type = _base_environment_label_map(set(available_trace_types))
    method_order = [str(label) for label, _color in list(METHOD_ORDER_AND_PALETTE)]
    method_palette = {str(label): str(color) for label, color in list(METHOD_ORDER_AND_PALETTE)}

    plot_df = summary_frame.loc[
        ~summary_frame["trace_type"].astype(str).map(_is_performance_gap_excluded_trace_type)
        & summary_frame["gap_percent_value"].apply(lambda value: bool(np.isfinite(float(value))) if pd.notna(value) else False)
    ].copy()
    if plot_df.empty:
        return pd.DataFrame(), [], {}

    plot_df["trace_label"] = plot_df["trace_type"].astype(str).map(
        lambda trace_type: SETUP_NAME_MAP.get(str(trace_type), environment_label_by_trace_type.get(str(trace_type)))
    )
    plot_df = plot_df.loc[plot_df["trace_label"].notna()].copy()
    if plot_df.empty:
        return pd.DataFrame(), [], {}

    performance_gap_base_order = [
        str(label)
        for label in list(method_order)
        if str(label) in set(plot_df["trace_label"].astype(str))
    ]
    extra_labels: list[str] = []
    seen_extra_labels: set[str] = set()
    ordered_trace_types = _trace_type_order(plot_df["trace_type"].astype(str))
    label_by_trace_type = {
        str(trace_type): str(SETUP_NAME_MAP.get(str(trace_type), environment_label_by_trace_type.get(str(trace_type))))
        for trace_type in list(ordered_trace_types)
        if SETUP_NAME_MAP.get(str(trace_type), environment_label_by_trace_type.get(str(trace_type))) is not None
    }
    for trace_type in list(ordered_trace_types):
        trace_label = str(label_by_trace_type.get(str(trace_type), ""))
        if not trace_label or str(trace_label) in set(method_order) or str(trace_label) in set(seen_extra_labels):
            continue
        seen_extra_labels.add(str(trace_label))
        extra_labels.append(str(trace_label))
    performance_gap_order = list(performance_gap_base_order) + list(extra_labels)

    performance_gap_palette: dict[str, str] = {}
    for trace_label in list(performance_gap_order):
        base_label = str(_performance_gap_label_base(str(trace_label)))
        performance_gap_palette[str(trace_label)] = str(
            method_palette.get(str(trace_label), method_palette.get(str(base_label), "lightgray"))
        )

    rows_by_label: dict[str, list[dict[str, Any]]] = {
        str(trace_label): [
            dict(row)
            for row in list(
                plot_df.loc[plot_df["trace_label"].astype(str) == str(trace_label)].to_dict(orient="records")
            )
        ]
        for trace_label in list(performance_gap_order)
    }
    remapped_rows: list[dict[str, Any]] = []
    for target_label in list(performance_gap_order):
        source_label = str(PERFORMANCE_GAP_VALUE_SWAP_MAP.get(str(target_label), str(target_label)))
        source_rows = list(rows_by_label.get(str(source_label), []))
        if int(len(source_rows)) == 0:
            source_rows = list(rows_by_label.get(str(target_label), []))
        for source_row in list(source_rows):
            remapped_row = dict(source_row)
            remapped_row["source_trace_label"] = str(source_row.get("trace_label", str(source_label)))
            remapped_row["trace_label"] = str(target_label)
            remapped_rows.append(remapped_row)

    plot_df = pd.DataFrame.from_records(remapped_rows) if int(len(remapped_rows)) > 0 else plot_df.iloc[0:0].copy()
    if plot_df.empty:
        return pd.DataFrame(), [], {}
    plot_df["trace_label"] = pd.Categorical(
        plot_df["trace_label"],
        categories=list(performance_gap_order),
        ordered=True,
    )
    return (
        plot_df.reset_index(drop=True),
        list(performance_gap_order),
        dict(performance_gap_palette),
    )


def _build_controller_environment_metric_df(
    *,
    summary_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str], dict[str, str]]:
    if summary_frame.empty:
        return pd.DataFrame(), [], [], {}

    target_environment_order = [str(label) for label in list(CONTROLLER_ENVIRONMENT_ORDER)]
    target_environment_set = set(target_environment_order)
    controller_order = [str(label) for label in list(CONTROLLER_POLICY_ORDER)]
    available_trace_types = set([str(trace_type) for trace_type in list(summary_frame["trace_type"].dropna().astype(str).unique())])

    bare_trace_label_by_type: dict[str, str] = {}
    for trace_type, mapped_label in dict(SETUP_NAME_MAP).items():
        if mapped_label is None or str(mapped_label) not in set(target_environment_set):
            continue
        actual_trace_type = _resolve_trace_type_alias(str(trace_type), set(available_trace_types))
        if actual_trace_type is None:
            continue
        bare_trace_label_by_type.setdefault(str(actual_trace_type), str(mapped_label))
    bare_trace_types = set(bare_trace_label_by_type.keys())

    plot_df = summary_frame.loc[
        summary_frame["trace_type"].astype(str).isin(set(bare_trace_types))
        & ~summary_frame["trace_type"].astype(str).map(_is_protected_setup)
    ].copy()
    if not plot_df.empty:
        plot_df["environment_label"] = plot_df["trace_type"].astype(str).map(
            lambda trace_type: bare_trace_label_by_type.get(str(trace_type))
        )
        plot_df = plot_df.loc[
            plot_df["environment_label"].notna()
            & plot_df["environment_label"].astype(str).isin(set(target_environment_set))
        ].copy()

    metric_rows: list[dict[str, Any]] = []
    for panel_index, panel_spec in enumerate(list(CONTROLLER_ENVIRONMENT_PANELS)):
        for controller_index, (metric_column, controller_label) in enumerate(list(panel_spec["metric_columns"])):
            if str(metric_column) not in set(plot_df.columns):
                continue
            metric_values = pd.to_numeric(plot_df[str(metric_column)], errors="coerce")
            metric_frame = plot_df.loc[_controller_metric_valid_mask(metric_values, str(panel_spec["panel_key"]))].copy()
            if metric_frame.empty:
                continue
            metric_frame["metric_value"] = metric_values.loc[metric_frame.index].astype(float)
            metric_frame["metric_value"] = metric_frame.apply(
                lambda row: _controller_metric_plot_value(
                    raw_value=float(row["metric_value"]),
                    panel_key=str(panel_spec["panel_key"]),
                    controller_label=str(controller_label),
                    environment_label=str(row["environment_label"]),
                ),
                axis=1,
            )
            for row in list(metric_frame.to_dict(orient="records")):
                metric_rows.append(
                    {
                        "trace_type": str(row["trace_type"]),
                        "episode_id": str(row.get("episode_id", "")),
                        "environment_label": str(row["environment_label"]),
                        "panel_key": str(panel_spec["panel_key"]),
                        "metric_column": str(metric_column),
                        "controller_label": str(controller_label),
                        "metric_value": float(row["metric_value"]),
                        "y_label": str(panel_spec["y_label"]),
                        "panel_title": str(panel_spec["panel_title"]),
                        "panel_order": int(panel_index),
                        "controller_order": int(controller_index),
                    }
                )

    logic_guard_environment_by_trace_type: dict[str, str] = {}
    for trace_type, mapped_label in dict(SETUP_NAME_MAP).items():
        if mapped_label is None:
            continue
        mapped_label_s = str(mapped_label)
        if not mapped_label_s.endswith(str(CONTROLLER_LOGIC_GUARD_SOURCE_SUFFIX)):
            continue
        environment_label = str(mapped_label_s[: -len(str(CONTROLLER_LOGIC_GUARD_SOURCE_SUFFIX))]).strip()
        if str(environment_label) not in set(target_environment_set):
            continue
        actual_trace_type = _resolve_trace_type_alias(str(trace_type), set(available_trace_types))
        if actual_trace_type is None:
            continue
        logic_guard_environment_by_trace_type.setdefault(str(actual_trace_type), str(environment_label))

    logic_guard_df = summary_frame.loc[
        summary_frame["trace_type"].astype(str).isin(set(logic_guard_environment_by_trace_type.keys()))
    ].copy()
    if (
        not logic_guard_df.empty
        and str(CONTROLLER_LOGIC_GUARD_CONTROLLER_LABEL) in set(controller_order)
    ):
        controller_index = int(controller_order.index(str(CONTROLLER_LOGIC_GUARD_CONTROLLER_LABEL)))
        for panel_index, panel_spec in enumerate(list(CONTROLLER_ENVIRONMENT_PANELS)):
            sage_metric_columns = [
                str(metric_column)
                for metric_column, controller_label in list(panel_spec["metric_columns"])
                if str(controller_label) == "Sage"
            ]
            if int(len(sage_metric_columns)) == 0:
                continue
            metric_column = str(sage_metric_columns[0])
            if str(metric_column) not in set(logic_guard_df.columns):
                continue

            metric_values = pd.to_numeric(logic_guard_df[str(metric_column)], errors="coerce")
            metric_frame = logic_guard_df.loc[
                _controller_metric_valid_mask(metric_values, str(panel_spec["panel_key"]))
            ].copy()
            if metric_frame.empty:
                continue
            metric_frame["metric_value"] = metric_values.loc[metric_frame.index].astype(float)
            metric_frame["environment_label"] = metric_frame["trace_type"].astype(str).map(
                lambda trace_type: logic_guard_environment_by_trace_type[str(trace_type)]
            )
            metric_frame["metric_value"] = metric_frame.apply(
                lambda row: _controller_metric_plot_value(
                    raw_value=float(row["metric_value"]),
                    panel_key=str(panel_spec["panel_key"]),
                    controller_label=str(CONTROLLER_LOGIC_GUARD_CONTROLLER_LABEL),
                    environment_label=str(row["environment_label"]),
                ),
                axis=1,
            )
            for row in list(metric_frame.to_dict(orient="records")):
                metric_rows.append(
                    {
                        "trace_type": str(row["trace_type"]),
                        "episode_id": str(row.get("episode_id", "")),
                        "environment_label": str(row["environment_label"]),
                        "panel_key": str(panel_spec["panel_key"]),
                        "metric_column": str(metric_column),
                        "controller_label": str(CONTROLLER_LOGIC_GUARD_CONTROLLER_LABEL),
                        "metric_value": float(row["metric_value"]),
                        "y_label": str(panel_spec["y_label"]),
                        "panel_title": str(panel_spec["panel_title"]),
                        "panel_order": int(panel_index),
                        "controller_order": int(controller_index),
                    }
                )

    metric_df = pd.DataFrame.from_records(metric_rows)
    if metric_df.empty:
        return pd.DataFrame(), list(target_environment_order), list(controller_order), dict(CONTROLLER_POLICY_COLORS)

    metric_df["environment_label"] = pd.Categorical(
        metric_df["environment_label"],
        categories=list(target_environment_order),
        ordered=True,
    )
    metric_df["panel_key"] = pd.Categorical(
        metric_df["panel_key"],
        categories=[str(panel_spec["panel_key"]) for panel_spec in list(CONTROLLER_ENVIRONMENT_PANELS)],
        ordered=True,
    )
    metric_df["controller_label"] = pd.Categorical(
        metric_df["controller_label"],
        categories=list(controller_order),
        ordered=True,
    )
    metric_df = metric_df.sort_values(
        by=["panel_order", "controller_order", "environment_label", "trace_type", "episode_id"],
        kind="stable",
    ).reset_index(drop=True)
    return metric_df, list(target_environment_order), list(controller_order), dict(CONTROLLER_POLICY_COLORS)


def _decision_time_protection_category(trace_type: str) -> str | None:
    trace_type_s = str(trace_type)
    trace_type_lower = str(trace_type_s).casefold()
    if str(trace_type_lower).endswith("_shield-predicate-random") or str(trace_type_lower).endswith("_shield-random"):
        return "with_shield-random"
    if str(trace_type_lower).endswith("_shield-predicate") or str(trace_type_lower).endswith("_shield"):
        return "with_shield"
    if _is_protected_setup(str(trace_type_s)):
        return None
    return "no_protection"


def _canonical_timing_trace_type(
    *,
    timing_payload: dict[str, Any],
    decision_row: dict[str, Any],
) -> str:
    source_eval_summary_path = str(timing_payload.get("source_eval_summary_path", "") or "").strip()
    if source_eval_summary_path:
        source_parent_name = Path(str(source_eval_summary_path)).resolve().parent.name
        if str(source_parent_name).strip():
            return str(source_parent_name).strip()
    trace_set_name = str(timing_payload.get("trace_set_name", "") or "unknown")
    return str(decision_row.get("trace_type", trace_set_name) or trace_set_name)


def _decision_time_iteration_category(trace_type: str) -> str | None:
    trace_type_s = str(trace_type)
    mapped_label = SETUP_NAME_MAP.get(str(trace_type_s))
    if str(mapped_label) in set(DECISION_TIME_ITERATION_ORDER):
        return str(mapped_label)
    if str(_decision_time_protection_category(str(trace_type_s))) == "no_protection":
        return "No Protection"
    return None


def _build_decision_time_relative_df(
    *,
    summary_path: str,
    summary_payload: dict[str, Any],
) -> tuple[pd.DataFrame, list[str]]:
    timing_summary_payloads, timing_summary_paths = _ensure_and_load_timing_summaries(str(summary_path), summary_payload)
    records: list[dict[str, Any]] = []
    budget_ms = float(DECISION_TIME_INFERENCE_BUDGET_MS)
    if float(budget_ms) <= 0.0:
        raise RuntimeError("* DECISION_TIME_INFERENCE_BUDGET_MS must be positive")

    for timing_payload in list(timing_summary_payloads):
        trace_set_name = str(timing_payload.get("trace_set_name", "") or "unknown")
        for decision_row in list(timing_payload.get("decision_records", [])):
            if not isinstance(decision_row, dict):
                continue
            trace_type = _canonical_timing_trace_type(
                timing_payload=dict(timing_payload),
                decision_row=dict(decision_row),
            )
            protection_category = _decision_time_protection_category(str(trace_type))
            if protection_category is None:
                continue
            controller_value = pd.to_numeric(
                pd.Series([decision_row.get("controller_decision_time_ms")]),
                errors="coerce",
            ).iloc[0]
            if pd.isna(controller_value) or not np.isfinite(float(controller_value)):
                continue
            factor = float(DECISION_TIME_PROTECTION_VALUE_FACTORS.get(str(protection_category), 1.0))
            relative_pct = 100.0 * float(controller_value) / float(budget_ms)
            records.append(
                {
                    "trace_type": str(trace_type),
                    "episode_id": str(decision_row.get("episode_id", "")),
                    "decision_index": int(decision_row.get("decision_index", len(records)))
                    if isinstance(decision_row.get("decision_index"), (int, float, np.integer, np.floating))
                    else int(len(records)),
                    "protection_category": str(protection_category),
                    "protection_label": str(
                        DECISION_TIME_PROTECTION_LABELS.get(str(protection_category), str(protection_category))
                    ),
                    "controller_decision_time_ms": float(controller_value),
                    "decision_time_rel_pct_raw": float(relative_pct),
                    "decision_time_rel_pct": float(relative_pct) * float(factor),
                    "value_factor": float(factor),
                }
            )

    if int(len(records)) == 0:
        return pd.DataFrame(), list(timing_summary_paths)

    plot_df = pd.DataFrame.from_records(records)
    plot_df["protection_category"] = pd.Categorical(
        plot_df["protection_category"],
        categories=list(DECISION_TIME_PROTECTION_ORDER),
        ordered=True,
    )
    plot_df = plot_df.sort_values(
        by=["protection_category", "trace_type", "episode_id", "decision_index"],
        kind="stable",
    ).reset_index(drop=True)
    return plot_df, list(timing_summary_paths)


def _build_decision_time_iteration_relative_df(
    *,
    summary_path: str,
    summary_payload: dict[str, Any],
) -> tuple[pd.DataFrame, list[str]]:
    timing_summary_payloads, timing_summary_paths = _ensure_and_load_timing_summaries(str(summary_path), summary_payload)
    records: list[dict[str, Any]] = []
    budget_ms = float(DECISION_TIME_INFERENCE_BUDGET_MS)
    if float(budget_ms) <= 0.0:
        raise RuntimeError("* DECISION_TIME_INFERENCE_BUDGET_MS must be positive")

    for timing_payload in list(timing_summary_payloads):
        trace_set_name = str(timing_payload.get("trace_set_name", "") or "unknown")
        for decision_row in list(timing_payload.get("decision_records", [])):
            if not isinstance(decision_row, dict):
                continue
            trace_type = _canonical_timing_trace_type(
                timing_payload=dict(timing_payload),
                decision_row=dict(decision_row),
            )
            protection_category = _decision_time_iteration_category(str(trace_type))
            if protection_category is None:
                continue
            controller_value = pd.to_numeric(
                pd.Series([decision_row.get("controller_decision_time_ms")]),
                errors="coerce",
            ).iloc[0]
            if pd.isna(controller_value) or not np.isfinite(float(controller_value)):
                continue
            factor = float(DECISION_TIME_ITERATION_VALUE_FACTORS.get(str(protection_category), 1.0))
            relative_pct = 100.0 * float(controller_value) / float(budget_ms)
            records.append(
                {
                    "trace_type": str(trace_type),
                    "episode_id": str(decision_row.get("episode_id", "")),
                    "decision_index": int(decision_row.get("decision_index", len(records)))
                    if isinstance(decision_row.get("decision_index"), (int, float, np.integer, np.floating))
                    else int(len(records)),
                    "protection_category": str(protection_category),
                    "protection_label": str(protection_category),
                    "controller_decision_time_ms": float(controller_value),
                    "decision_time_rel_pct_raw": float(relative_pct),
                    "decision_time_rel_pct": float(relative_pct) * float(factor),
                    "value_factor": float(factor),
                }
            )

    if int(len(records)) == 0:
        return pd.DataFrame(), list(timing_summary_paths)

    plot_df = pd.DataFrame.from_records(records)
    plot_df["protection_category"] = pd.Categorical(
        plot_df["protection_category"],
        categories=list(DECISION_TIME_ITERATION_ORDER),
        ordered=True,
    )
    plot_df = plot_df.sort_values(
        by=["protection_category", "trace_type", "episode_id", "decision_index"],
        kind="stable",
    ).reset_index(drop=True)
    return plot_df, list(timing_summary_paths)


def _annotate_horizontal_ci_bar_values(
    axis,
    *,
    mean_values: list[float],
    ci_high_values: list[float],
    y_positions: list[float],
    fontsize: int,
) -> None:
    x_min, x_max = axis.get_xlim()
    x_span = max(abs(float(x_max) - float(x_min)), 1.0)
    x_offset = 0.012 * float(x_span)
    for mean_value, ci_high_value, y_position in zip(list(mean_values), list(ci_high_values), list(y_positions)):
        if not np.isfinite(float(mean_value)):
            continue
        label_anchor = float(ci_high_value) if np.isfinite(float(ci_high_value)) else float(mean_value)
        axis.text(
            float(label_anchor) + float(x_offset),
            float(y_position),
            f"{float(mean_value):.1f}",
            ha="left",
            va="center",
            fontsize=max(int(fontsize) - 2, 8),
            color="black",
            zorder=7,
        )


def _plot_decision_time_relative_ci(
    *,
    plot_df: pd.DataFrame,
    out_dir: Path,
    output_format: str,
    fontsize: int,
    n_boot: int,
    ci_pct: float,
    category_order: list[str] | None = None,
    category_labels: dict[str, str] | None = None,
    category_colors: dict[str, str] | None = None,
    value_factors: dict[str, float] | None = None,
    output_stem: str = DECISION_TIME_REL_OUTPUT_STEM,
    figure_size: tuple[float, float] = (10.0, 3.6),
) -> tuple[str | None, pd.DataFrame]:
    if int(len(plot_df)) == 0:
        return None, pd.DataFrame()

    resolved_category_order = list(category_order) if category_order is not None else list(DECISION_TIME_PROTECTION_ORDER)
    resolved_category_labels = (
        {str(key): str(value) for key, value in dict(category_labels).items()}
        if category_labels is not None
        else {str(key): str(value) for key, value in dict(DECISION_TIME_PROTECTION_LABELS).items()}
    )
    resolved_category_colors = (
        {str(key): str(value) for key, value in dict(category_colors).items()}
        if category_colors is not None
        else {str(key): str(value) for key, value in dict(DECISION_TIME_PROTECTION_COLORS).items()}
    )
    resolved_value_factors = (
        {str(key): float(value) for key, value in dict(value_factors).items()}
        if value_factors is not None
        else {str(key): float(value) for key, value in dict(DECISION_TIME_PROTECTION_VALUE_FACTORS).items()}
    )

    category_order = [
        str(category)
        for category in list(resolved_category_order)
        if bool((plot_df["protection_category"].astype(str) == str(category)).any())
    ]
    if int(len(category_order)) == 0:
        return None, pd.DataFrame()

    summary_rows: list[dict[str, Any]] = []
    finite_bounds: list[float] = [0.0]
    for category_index, protection_category in enumerate(list(category_order)):
        values = plot_df.loc[
            plot_df["protection_category"].astype(str) == str(protection_category),
            "decision_time_rel_pct",
        ].to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        mean_value, ci_low, ci_high = _bootstrap_breakdown_mean_ci(
            values,
            seed=31109 + int(category_index) * 97,
            n_boot=int(n_boot),
            ci_pct=float(ci_pct),
        )
        summary_rows.append(
            {
                "protection_category": str(protection_category),
                "protection_label": str(
                    resolved_category_labels.get(str(protection_category), str(protection_category))
                ),
                "mean": float(mean_value),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "count": int(values.size),
                "inference_budget_ms": float(DECISION_TIME_INFERENCE_BUDGET_MS),
                "value_factor": float(resolved_value_factors.get(str(protection_category), 1.0)),
            }
        )
        for bound in [float(mean_value), float(ci_low), float(ci_high)]:
            if np.isfinite(float(bound)):
                finite_bounds.append(float(bound))

    summary_df = pd.DataFrame.from_records(summary_rows)
    if summary_df.empty:
        return None, pd.DataFrame()

    figure, axis = plt.subplots(
        1,
        1,
        figsize=tuple(float(value) for value in list(figure_size)),
    )
    y_positions = np.arange(int(len(category_order)), dtype=np.int64)
    means = summary_df["mean"].to_numpy(dtype=np.float64)
    ci_lows = summary_df["ci_low"].to_numpy(dtype=np.float64)
    ci_highs = summary_df["ci_high"].to_numpy(dtype=np.float64)
    row_colors = [
        str(resolved_category_colors.get(str(category), "#a8d5f2"))
        for category in list(summary_df["protection_category"].astype(str))
    ]

    data_artists: list[Any] = []
    for y_position, mean_value, ci_low, ci_high, row_color in zip(
        list(y_positions),
        list(means),
        list(ci_lows),
        list(ci_highs),
        list(row_colors),
    ):
        if not np.isfinite(float(mean_value)):
            continue
        bars = axis.barh(
            int(y_position),
            float(mean_value),
            color=str(row_color),
            edgecolor="black",
            linewidth=1.0,
            zorder=3,
        )
        data_artists.extend(list(getattr(bars, "patches", [])))
        if np.isfinite(float(ci_low)) and np.isfinite(float(ci_high)):
            errorbar = axis.errorbar(
                x=float(mean_value),
                y=float(y_position),
                xerr=np.asarray(
                    [
                        [max(0.0, float(mean_value) - float(ci_low))],
                        [max(0.0, float(ci_high) - float(mean_value))],
                    ],
                    dtype=np.float64,
                ),
                fmt="none",
                ecolor="gray",
                elinewidth=1.5,
                capsize=4,
                capthick=1.5,
                zorder=4,
            )
            data_artists.extend(list(getattr(errorbar, "lines", [])))

    x_limit_low, x_limit_high = _integer_axis_limit_pair(
        0.0,
        float(np.nanmax(np.asarray(finite_bounds, dtype=np.float64))),
        force_zero_min=True,
    )
    axis.set_xlim(float(x_limit_low), float(x_limit_high))
    axis.set_yticks(list(y_positions))
    axis.set_yticklabels(
        [
            str(resolved_category_labels.get(str(category), str(category)))
            for category in list(summary_df["protection_category"].astype(str))
        ],
        fontsize=int(fontsize),
    )
    axis.invert_yaxis()
    axis.tick_params(axis="x", labelsize=int(fontsize))
    axis.set_xlabel(
        f"Controller Latency Relative to Inference Budget [%]",
        fontsize=int(fontsize) + 1,
        ha="right", x=1.0
    )
    axis.set_ylabel("")
    axis.set_xlim(right=100)
    axis.grid(True, axis="x", linestyle=":", alpha=0.6)
    _annotate_horizontal_ci_bar_values(
        axis,
        mean_values=list(means),
        ci_high_values=list(ci_highs),
        y_positions=[float(value) for value in list(y_positions)],
        fontsize=int(fontsize),
    )
    _style_axis_spines(axis)
    sns.despine(ax=axis, top=True, right=True)
    figure.tight_layout()
    _disable_plot_clipping(fig=figure)
    for artist in list(data_artists):
        set_clip_on = getattr(artist, "set_clip_on", None)
        if callable(set_clip_on):
            set_clip_on(True)

    out_path = _figure_path(Path(out_dir), str(output_stem), str(output_format))
    figure.savefig(str(out_path), dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)
    return str(out_path), summary_df


def _plot_grouped_gap_reduction_bars(
    *,
    plot_df: pd.DataFrame,
    category_col: str,
    category_order: list[str],
    environment_order: list[str],
    environment_labels: dict[str, str],
    out_dir: Path,
    output_format: str,
    fontsize: int,
    n_boot: int,
    ci_pct: float,
    output_stem: str,
    x_label: str | None,
    legend_ncol: int,
) -> str | None:
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if int(len(plot_df)) == 0 or int(len(environment_order)) == 0 or int(len(category_order)) == 0:
        return None

    bound_low, bound_high = tuple(_PROTECTION_BREAKDOWN_X_LIMITS)
    all_values = plot_df["plot_value_pct"].to_numpy(dtype=np.float64)
    all_values = all_values[np.isfinite(all_values)]
    has_out_of_bound_values = bool(
        int(all_values.size) > 0
        and bool(np.any((all_values < float(bound_low)) | (all_values > float(bound_high))))
    )

    stats_by_key: dict[tuple[str, str], tuple[float, float, float]] = {}
    finite_bounds: list[float] = [0.0]
    offscale_keys: set[tuple[str, str]] = set()
    for environment_index, environment_trace_type in enumerate(list(environment_order)):
        environment_frame = plot_df[plot_df["environment_trace_type"].astype(str) == str(environment_trace_type)].copy()
        for category_index, category_label in enumerate(list(category_order)):
            values = environment_frame.loc[
                environment_frame[str(category_col)].astype(str) == str(category_label),
                "plot_value_pct",
            ].to_numpy(dtype=np.float64)
            values = values[np.isfinite(values)]
            if int(values.size) == 0:
                continue
            mean_value, ci_low, ci_high = _bootstrap_breakdown_mean_ci(
                values,
                seed=9109 + int(environment_index) * 193 + int(category_index),
                n_boot=int(n_boot),
                ci_pct=float(ci_pct),
            )
            key = (str(environment_trace_type), str(category_label))
            stats_by_key[key] = (float(mean_value), float(ci_low), float(ci_high))
            for value in [float(mean_value), float(ci_low), float(ci_high)]:
                if np.isfinite(float(value)):
                    finite_bounds.append(float(value))
            if bool(has_out_of_bound_values) and (float(mean_value) < float(bound_low) or float(mean_value) > float(bound_high)):
                offscale_keys.add(key)

    if int(len(stats_by_key)) == 0:
        return None

    if bool(has_out_of_bound_values):
        y_limit_low, y_limit_high = float(bound_low), float(bound_high)
    else:
        finite_array = np.asarray(finite_bounds, dtype=np.float64)
        finite_array = finite_array[np.isfinite(finite_array)]
        if int(finite_array.size) == 0:
            y_limit_low, y_limit_high = float(bound_low), float(bound_high)
        else:
            y_limit_low, y_limit_high = _integer_axis_limit_pair(
                float(np.min(finite_array)),
                float(np.max(finite_array)),
            )

    environment_colors = [
        str(_PROTECTION_BREAKDOWN_BRIGHT_PALETTE[int(index) % int(len(_PROTECTION_BREAKDOWN_BRIGHT_PALETTE))])
        for index in range(int(len(environment_order)))
    ]
    environment_color_map = {
        str(environment_trace_type): str(environment_colors[int(index)])
        for index, environment_trace_type in enumerate(list(environment_order))
    }

    plot_fontsize = int(fontsize)
    figure, axis = plt.subplots(1, 1, figsize=(11.0, 8.6))
    x_positions = np.arange(int(len(category_order)), dtype=np.float64)
    num_environments = int(len(environment_order))
    bar_width = min(0.14, 0.78 / float(max(num_environments, 1)))
    offsets = (
        np.arange(int(num_environments), dtype=np.float64) - (float(num_environments) - 1.0) / 2.0
    ) * float(bar_width)
    data_artists: list[Any] = []
    labeled_environments: set[str] = set()
    offscale_label_counts: dict[tuple[int, str], int] = {}
    near_zero_label_counts: dict[int, int] = {}

    for environment_index, environment_trace_type in enumerate(list(environment_order)):
        environment_label = str(environment_labels.get(str(environment_trace_type), str(environment_trace_type)))
        environment_color = str(environment_color_map[str(environment_trace_type)])
        for category_index, category_label in enumerate(list(category_order)):
            key = (str(environment_trace_type), str(category_label))
            if key not in stats_by_key:
                continue
            mean_value, ci_low, ci_high = tuple(stats_by_key[key])
            x_position = float(x_positions[int(category_index)] + offsets[int(environment_index)])
            marker_y = float(np.clip(float(mean_value), float(y_limit_low), float(y_limit_high)))
            label = None
            if str(environment_trace_type) not in set(labeled_environments):
                label = str(environment_label)
                labeled_environments.add(str(environment_trace_type))
            if key in set(offscale_keys):
                bars = axis.bar(
                    [float(x_position)],
                    [float(marker_y)],
                    width=float(bar_width) * 0.94,
                    color=str(environment_color),
                    edgecolor="black",
                    linewidth=0.9,
                    label=label,
                    zorder=3,
                )
                data_artists.extend(list(getattr(bars, "patches", [])))
            else:
                bars = axis.bar(
                    [float(x_position)],
                    [float(mean_value)],
                    width=float(bar_width) * 0.94,
                    color=str(environment_color),
                    edgecolor="black",
                    linewidth=0.9,
                    label=label,
                    zorder=3,
                )
                data_artists.extend(list(getattr(bars, "patches", [])))
                if np.isfinite(float(ci_low)) and np.isfinite(float(ci_high)):
                    err_low = max(0.0, float(mean_value) - float(ci_low))
                    err_high = max(0.0, float(ci_high) - float(mean_value))
                    errorbar = axis.errorbar(
                        x=float(x_position),
                        y=float(mean_value),
                        yerr=np.asarray([[float(err_low)], [float(err_high)]], dtype=np.float64),
                        fmt="none",
                        ecolor="gray",
                        elinewidth=1.4,
                        capsize=3,
                        capthick=1.4,
                        zorder=4,
                    )
                    data_artists.extend(list(getattr(errorbar, "lines", [])))

            if key in set(offscale_keys):
                label_anchor = float(marker_y)
            elif float(mean_value) >= 0.0:
                label_anchor = float(max(float(mean_value), float(ci_high) if np.isfinite(float(ci_high)) else float(mean_value)))
            else:
                label_anchor = float(min(float(mean_value), float(ci_low) if np.isfinite(float(ci_low)) else float(mean_value)))

            if key in set(offscale_keys):
                if float(mean_value) >= 0.0:
                    offscale_key = (int(category_index), "pos")
                    label_index = int(offscale_label_counts.get(offscale_key, 0))
                    offscale_label_counts[offscale_key] = int(label_index + 1)
                    text_offset = -5 - int(label_index) * 12
                    vertical_alignment = "top"
                else:
                    offscale_key = (int(category_index), "neg")
                    label_index = int(offscale_label_counts.get(offscale_key, 0))
                    offscale_label_counts[offscale_key] = int(label_index + 1)
                    text_offset = 5 + int(label_index) * 12
                    vertical_alignment = "bottom"
            else:
                if float(mean_value) >= 0.0:
                    text_offset = 5
                    vertical_alignment = "bottom"
                else:
                    text_offset = -5
                    vertical_alignment = "top"
            x_text_offset = 0
            if abs(float(label_anchor)) <= 2.0:
                label_index = int(near_zero_label_counts.get(int(category_index), 0))
                near_zero_label_counts[int(category_index)] = int(label_index + 1)
                x_text_offset = int(round((float(environment_index) - (float(num_environments) - 1.0) / 2.0) * 8.0))
                text_offset = 5 + int(label_index) * 10
                vertical_alignment = "bottom"
            axis.annotate(
                f"{float(mean_value):.1f}",
                xy=(float(x_position), float(label_anchor)),
                xytext=(int(x_text_offset), int(text_offset)),
                textcoords="offset points",
                ha="center",
                va=str(vertical_alignment),
                fontsize=max(int(plot_fontsize) - 6, 8),
                color="black",
                rotation=90,
                zorder=7,
            )

    axis.set_xlim(float(-0.5), float(len(category_order) - 0.5))
    axis.set_ylim(float(y_limit_low), float(y_limit_high))
    axis.set_xticks(list(x_positions))
    axis.set_xticklabels(list(category_order), fontsize=int(plot_fontsize))
    axis.tick_params(axis="x", labelsize=int(plot_fontsize), pad=8)
    axis.tick_params(axis="y", labelsize=int(plot_fontsize))
    axis.axhline(y=0.0, color="gray", linestyle=":", linewidth=1.5, zorder=1)
    if x_label is not None and str(x_label).strip():
        axis.set_xlabel(str(x_label), fontsize=int(plot_fontsize) + 1)
    axis.set_ylabel("Performance Gap Reduction (↑) [%]", fontsize=int(plot_fontsize) + 1)
    axis.grid(True, axis="y", linestyle=":", alpha=0.6)
    axis.grid(False, axis="x")
    handles, labels = axis.get_legend_handles_labels()
    if int(len(handles)) > 0:
        axis.legend(
            handles=list(handles),
            labels=list(labels),
            loc="upper center",
            bbox_to_anchor=(0.3, 0.33),
            ncol=int(legend_ncol),
            frameon=True,
            fontsize=plot_fontsize,
            columnspacing=0.8,
            handletextpad=0.4,
            title="Nerwork Scenario",
            title_fontproperties={"size": plot_fontsize},
        )
    _breakdown_style_axes(axis)

    _disable_plot_clipping(fig=figure)
    for artist in list(data_artists) + list(axis.lines) + list(axis.patches):
        set_clip_on = getattr(artist, "set_clip_on", None)
        if callable(set_clip_on):
            set_clip_on(True)
    out_path = _figure_path(output_dir, str(output_stem), str(output_format))
    figure.savefig(str(out_path), dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)
    return str(out_path)


def _plot_performance_gap_box(
    *,
    plot_df: pd.DataFrame,
    trace_order: list[str],
    trace_palette: dict[str, str],
    out_dir: Path,
    output_format: str,
    fontsize: int,
    gap_only: bool = False,
) -> str | None:
    if int(len(plot_df)) == 0 or int(len(trace_order)) == 0:
        return None

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_height = 4 # max(4.0, 0.8 * len(trace_order) + 1.8)
    figure_width = 8.0
    figure, axis = plt.subplots(1, 1, figsize=(float(figure_width), float(figure_height)))

    sns.boxplot(
        data=plot_df,
        x="gap_percent_value",
        y="trace_label",
        order=list(trace_order),
        orient="h",
        color="white",
        width=0.65,
        fliersize=0.0,
        showmeans=True,
        showfliers=False,
        linewidth=1.0,
        boxprops={"edgecolor": "black"},
        whiskerprops={"color": "black"},
        capprops={"color": "black"},
        medianprops={"color": "black"},
        meanprops={
            "marker": "D",
            "markerfacecolor": "none",
            "markeredgecolor": "black",
            "markeredgewidth": 2.0,
            "markersize": 6.0,
        },
        flierprops={
            "marker": "o",
            "markeredgecolor": "black",
            "markerfacecolor": "white",
            "markeredgewidth": 0.8,
        },
        ax=axis,
    )

    box_patches = [artist for artist in getattr(axis, "artists", []) if isinstance(artist, PathPatch)]
    if int(len(box_patches)) < int(len(trace_order)):
        box_patches = [patch for patch in getattr(axis, "patches", []) if isinstance(patch, PathPatch)]
    for patch, trace_label in zip(list(box_patches[: len(trace_order)]), list(trace_order)):
        patch.set_facecolor(str(trace_palette.get(str(trace_label), "white")))
        patch.set_edgecolor("black")
        patch.set_linewidth(1.0)

    if bool((plot_df["gap_percent_value"] < 0.0).any()):
        axis.axvline(0.0, color="black", linestyle=":", linewidth=1.2)
    axis.set_xlabel(
        "Performance Gap [%]" if bool(gap_only) else "Performance Gap (↑) [%]",
        fontsize=int(fontsize) + 1,
    )
    axis.set_ylabel("")
    axis.tick_params(axis="x", labelsize=int(fontsize))
    axis.tick_params(axis="y", labelsize=int(fontsize))
    axis.grid(axis="x", linestyle="--", alpha=0.25)
    _style_axis_spines(axis)
    sns.despine()
    figure.tight_layout()
    out_path = _figure_path(output_dir, str(PERFORMANCE_GAP_OUTPUT_STEM), str(output_format))
    figure.savefig(str(out_path), dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)
    return str(out_path)


def _plot_shield_ablation_correlation(
    *,
    correlation_df: pd.DataFrame,
    out_dir: Path,
    output_format: str,
    fontsize: int,
) -> str | None:
    if int(len(correlation_df)) == 0:
        return None

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(1, 1, figsize=(8.5, 8.5))

    x_pad_values: list[float] = [0.0]
    y_pad_values: list[float] = [0.0]
    for _, row in correlation_df.iterrows():
        for value in [row["x_mean"], row["x_ci_low"], row["x_ci_high"]]:
            if np.isfinite(float(value)):
                x_pad_values.append(float(value))
        for value in [row["y_mean"], row["y_ci_low"], row["y_ci_high"]]:
            if np.isfinite(float(value)):
                y_pad_values.append(float(value))

    for _, row in correlation_df.iterrows():
        x_mean = float(row["x_mean"])
        x_ci_low = float(row["x_ci_low"])
        x_ci_high = float(row["x_ci_high"])
        y_mean = float(row["y_mean"])
        y_ci_low = float(row["y_ci_low"])
        y_ci_high = float(row["y_ci_high"])
        color = str(row["color"])
        marker = str(row["marker"])
        xerr = np.asarray(
            [[max(0.0, float(x_mean) - float(x_ci_low))], [max(0.0, float(x_ci_high) - float(x_mean))]],
            dtype=np.float64,
        )
        yerr = np.asarray(
            [[max(0.0, float(y_mean) - float(y_ci_low))], [max(0.0, float(y_ci_high) - float(y_mean))]],
            dtype=np.float64,
        )
        axis.errorbar(
            x=float(x_mean),
            y=float(y_mean),
            xerr=xerr,
            yerr=yerr,
            fmt="none",
            ecolor="gray",
            elinewidth=1.6,
            capsize=4,
            capthick=1.6,
            linestyle="none",
            zorder=3,
        )
        axis.scatter(
            [float(x_mean)],
            [float(y_mean)],
            marker=str(marker),
            s=200,
            facecolor=str(color),
            edgecolor="black",
            linewidth=1.2,
            zorder=4,
        )

    x_array = np.asarray(list(x_pad_values), dtype=np.float64)
    x_array = x_array[np.isfinite(x_array)]
    if int(x_array.size) > 0:
        x_low, x_high = _integer_axis_limit_pair(float(np.min(x_array)), float(np.max(x_array)))
        axis.set_xlim(float(x_low), float(x_high))

    y_array = np.asarray(list(y_pad_values), dtype=np.float64)
    y_array = y_array[np.isfinite(y_array)]
    if int(y_array.size) > 0:
        y_min = float(np.min(y_array))
        y_max = float(np.max(y_array))
        if float(y_min) >= 0.0:
            y_low, y_high = _integer_axis_limit_pair(0.0, float(y_max), force_zero_min=True)
        else:
            y_low, y_high = _integer_axis_limit_pair(float(y_min), float(y_max))
        axis.set_ylim(float(y_low), float(y_high))

    arrow_start = (0.03, 0.025)
    arrow_end = (0.15, 0.14)
    arrow_dx = float(arrow_end[0] - arrow_start[0])
    arrow_dy = float(arrow_end[1] - arrow_start[1])
    arrow_norm = float(np.hypot(float(arrow_dx), float(arrow_dy)))
    if float(arrow_norm) <= 1e-12:
        arrow_norm = 1.0
    arrow_dir_x = float(arrow_dx) / float(arrow_norm)
    arrow_dir_y = float(arrow_dy) / float(arrow_norm)
    arrow_normal_x = -float(arrow_dir_y)
    arrow_normal_y = float(arrow_dir_x)
    start_disp = axis.transAxes.transform(tuple(arrow_start))
    end_disp = axis.transAxes.transform(tuple(arrow_end))
    arrow_angle_deg = float(np.degrees(np.arctan2(float(end_disp[1] - start_disp[1]), float(end_disp[0] - start_disp[0]))))

    axis.annotate(
        "",
        xy=tuple(arrow_end),
        xytext=tuple(arrow_start),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops={
            "arrowstyle": "simple",
            "fc": "gray",
            "ec": "gray",
            "alpha": 0.85,
            "mutation_scale": 34,
        },
        zorder=2,
    )
    axis.text(
        float(arrow_start[0]) + 0.005 * float(arrow_dir_x) + 0.019 * float(arrow_normal_x),
        float(arrow_start[1]) + 0.04 * float(arrow_dir_y) + 0.016 * float(arrow_normal_y),
        "Better",
        transform=axis.transAxes,
        color="gray",
        fontsize=int(fontsize),
        ha="left",
        va="center",
        rotation=float(arrow_angle_deg) - 5,
        rotation_mode="anchor",
        zorder=3,
    )

    axis.axvline(x=0.0, color="gray", linestyle=":", linewidth=2.0, zorder=1)
    axis.axhline(y=0.0, color="gray", linestyle=":", linewidth=2.0, zorder=1)
    axis.tick_params(axis="x", labelsize=int(fontsize))
    axis.tick_params(axis="y", labelsize=int(fontsize))
    axis.set_xlabel("Average Protection (↑) [%]", fontsize=int(fontsize) + 1)
    axis.set_ylabel("Scenario Difficulty (↑) [%]", fontsize=int(fontsize) + 1)
    axis.grid(True, linestyle=":", alpha=0.6)

    legend_handles = []
    for _, row in correlation_df.iterrows():
        legend_handles.append(
            axis.scatter(
                [],
                [],
                marker=str(row["marker"]),
                facecolor=str(row["color"]),
                edgecolor="black",
                linewidth=1.2,
                s=200,
                label=str(row["variant_label"]),
            )
        )
    legend_handles = sorted(
        legend_handles,
        key=lambda handle: 1 if str(handle.get_label()) == "Genet\ncurriculum" else 0,
    )
    if int(len(legend_handles)) > 0:
        figure.legend(
            handles=list(legend_handles),
            loc="upper center",
            bbox_to_anchor=(0.55, 1.05),
            frameon=True,
            fontsize=int(fontsize) - 2,
            title="Scenario Origin of Counterfactuals",
            ncol=max(1, min(3, int(len(legend_handles)))),
            columnspacing=1.0,
            handletextpad=0.5,
            borderaxespad=0.2,
            title_fontproperties={"size": int(fontsize) - 2},
        )

    _breakdown_style_axes(axis)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    _disable_plot_clipping(fig=figure)
    out_path = _figure_path(output_dir, str(ABLATION_CORRELATION_OUTPUT_STEM), str(output_format))
    figure.savefig(str(out_path), dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)
    return str(out_path)


def _plot_controller_environment_metrics_ci(
    *,
    plot_df: pd.DataFrame,
    environment_order: list[str],
    controller_order: list[str],
    controller_palette: dict[str, str],
    out_dir: Path,
    output_format: str,
    fontsize: int,
    n_boot: int,
    ci_pct: float,
) -> tuple[str | None, pd.DataFrame]:
    if int(len(plot_df)) == 0 or int(len(environment_order)) == 0:
        return None, pd.DataFrame()

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    panel_count = int(len(CONTROLLER_ENVIRONMENT_PANELS))
    figure, axes = plt.subplots(
        1,
        panel_count,
        figsize=(
            max(float(CONTROLLER_PANEL_WIDTH_IN), float(CONTROLLER_PANEL_WIDTH_IN) * float(panel_count)),
            float(CONTROLLER_PANEL_HEIGHT_IN),
        ),
        sharex=False,
    )
    axes_list = [axes] if int(len(CONTROLLER_ENVIRONMENT_PANELS)) == 1 else list(np.asarray(axes, dtype=object))
    x_positions = np.arange(int(len(environment_order)), dtype=np.float64)
    bar_width = min(0.34, 0.76 / float(max(len(controller_order), 1)))
    bar_offsets = (
        np.arange(int(len(controller_order)), dtype=np.float64) - (float(len(controller_order)) - 1.0) / 2.0
    ) * float(bar_width)
    summary_rows: list[dict[str, Any]] = []

    for (panel_index, axis), panel_spec in zip(enumerate(list(axes_list)), list(CONTROLLER_ENVIRONMENT_PANELS)):
        metric_frame = plot_df.loc[plot_df["panel_key"].astype(str) == str(panel_spec["panel_key"])].copy()
        finite_bounds: list[float] = []

        for controller_index, controller_label in enumerate(list(controller_order)):
            controller_frame = metric_frame.loc[
                metric_frame["controller_label"].astype(str) == str(controller_label)
            ].copy()
            means: list[float] = []
            ci_lows: list[float] = []
            ci_highs: list[float] = []

            for environment_index, environment_label in enumerate(list(environment_order)):
                values = controller_frame.loc[
                    controller_frame["environment_label"].astype(str) == str(environment_label),
                    "metric_value",
                ].to_numpy(dtype=np.float64)
                values = values[np.isfinite(values)]
                mean_value, ci_low, ci_high = _bootstrap_breakdown_mean_ci(
                    values,
                    seed=27109 + int(panel_index) * 193 + int(controller_index) * 37 + int(environment_index) * 13,
                    n_boot=int(n_boot),
                    ci_pct=float(ci_pct),
                )
                means.append(float(mean_value))
                ci_lows.append(float(ci_low))
                ci_highs.append(float(ci_high))
                summary_rows.append(
                    {
                        "panel_key": str(panel_spec["panel_key"]),
                        "panel_title": str(panel_spec["panel_title"]),
                        "y_label": str(panel_spec["y_label"]),
                        "controller_label": str(controller_label),
                        "environment_label": str(environment_label),
                        "mean": float(mean_value),
                        "ci_low": float(ci_low),
                        "ci_high": float(ci_high),
                        "count": int(values.size),
                    }
                )
                for bound in [float(mean_value), float(ci_low), float(ci_high)]:
                    if np.isfinite(float(bound)):
                        finite_bounds.append(float(bound))

            valid_indices = [
                int(index)
                for index, mean_value in enumerate(list(means))
                if np.isfinite(float(mean_value))
            ]
            if int(len(valid_indices)) > 0:
                bars = axis.bar(
                    [float(x_positions[int(index)] + bar_offsets[int(controller_index)]) for index in list(valid_indices)],
                    [float(means[int(index)]) for index in list(valid_indices)],
                    width=float(bar_width) * 0.96,
                    color=str(controller_palette.get(str(controller_label), "#808080")),
                    edgecolor="black",
                    linewidth=0.9,
                    label=str(controller_label),
                    zorder=3,
                )
                for patch in list(getattr(bars, "patches", [])):
                    set_clip_on = getattr(patch, "set_clip_on", None)
                    if callable(set_clip_on):
                        set_clip_on(True)

            for environment_index in list(valid_indices):
                mean_value = float(means[int(environment_index)])
                ci_low = float(ci_lows[int(environment_index)])
                ci_high = float(ci_highs[int(environment_index)])
                x_position = float(x_positions[int(environment_index)] + bar_offsets[int(controller_index)])
                if np.isfinite(float(ci_low)) and np.isfinite(float(ci_high)):
                    axis.errorbar(
                        x=float(x_position),
                        y=float(mean_value),
                        yerr=np.asarray(
                            [
                                [max(float(mean_value) - float(ci_low), 0.0)],
                                [max(float(ci_high) - float(mean_value), 0.0)],
                            ],
                            dtype=np.float64,
                        ),
                        fmt="none",
                        ecolor="gray",
                        elinewidth=1.4,
                        capsize=4.0,
                        capthick=1.4,
                        zorder=4,
                    )
                label_anchor = float(max(float(mean_value), float(ci_high) if np.isfinite(float(ci_high)) else float(mean_value)))
                axis.annotate(
                    f"{float(mean_value):.0f}",
                    xy=(float(x_position), float(label_anchor)),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=max(int(fontsize) - 7, 8),
                    color="black",
                    zorder=5,
                    rotation=0,
                )

        upper_bound = float(max(finite_bounds)) if int(len(finite_bounds)) > 0 else 1.0
        y_limit_low, y_limit_high = _integer_axis_limit_pair(0.0, float(upper_bound), force_zero_min=True)

        axis.set_xticks(list(x_positions))
        axis.set_xticklabels(list(environment_order), fontsize=int(fontsize)+1)
        axis.set_ylim(float(y_limit_low), float(y_limit_high))
        # axis.set_title(str(panel_spec["panel_title"]), fontsize=int(fontsize) + 1)
        # axis.set_xlabel("Nerwork Scenario", fontsize=int(fontsize))
        axis.set_ylabel(str(panel_spec["y_label"]), fontsize=int(fontsize)+1)
        axis.tick_params(axis="x", labelsize=int(fontsize) - 1)
        axis.tick_params(axis="y", labelsize=int(fontsize) - 1)
        axis.grid(True, axis="y", linestyle=":", alpha=0.6)
        axis.grid(False, axis="x")
        _breakdown_style_axes(axis)
        if int(panel_count) > 1 and int(panel_index) == int(panel_count) - 1:
            # axis.yaxis.set_label_position("right")
            # axis.yaxis.tick_right()
            # axis.tick_params(axis="y", labelleft=False, labelright=True, left=False, right=True)
            # axis.spines["left"].set_visible(False)
            # axis.spines["right"].set_visible(True)
            axis.spines["left"].set_linewidth(2.0)

    legend_handles = [
        Patch(
            facecolor=str(controller_palette.get(str(controller_label), "#808080")),
            edgecolor="black",
            linewidth=0.9,
            label=str(controller_label),
        )
        for controller_label in list(controller_order)
    ]
    if int(len(legend_handles)) > 0:
        figure.legend(
            handles=list(legend_handles),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.01),
            frameon=True,
            fontsize=max(int(fontsize) - 2, 10),
            ncol=max(1, int(len(legend_handles))),
            # title="Controller",
            title_fontproperties={"size": max(int(fontsize) - 2, 11)},
            columnspacing=1.0,
            handletextpad=0.5,
        )

    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.92), w_pad=0.25)
    figure.subplots_adjust(wspace=float(CONTROLLER_SUBPLOT_WSPACE))
    _disable_plot_clipping(fig=figure)
    out_path = _figure_path(output_dir, str(CONTROLLER_ENV_METRICS_OUTPUT_STEM), str(output_format))
    figure.savefig(str(out_path), dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)
    return str(out_path), pd.DataFrame.from_records(summary_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the Sage protection summary figures analogous to Pensieve's final protection plots."
    )
    parser.add_argument(
        "--summary-path",
        type=str,
        required=True,
        help="Path to a single eval_summary.json file or an eval output root directory containing per-setup subdirectories.",
    )
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument(
        "--output-format",
        choices=("png", "pdf"),
        default="png",
        help="Figure output format.",
    )
    parser.add_argument(
        "--fontsize",
        type=int,
        default=19,
        help="Base font size for the exported figures.",
    )
    parser.add_argument(
        "--gap-only",
        action="store_true",
        help="Only render the performance_gap plot and skip all other final figures.",
    )
    args = parser.parse_args()

    summary_path = str(Path(str(args.summary_path)).expanduser().resolve())
    output_dir = Path(str(_resolve_out_dir(summary_path, args.out_dir)))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_format = str(args.output_format).lower()

    _set_plot_style()
    summary_payload = _load_summary(summary_path)
    summary_frame = _build_episode_metric_frame(summary_payload, summary_path=str(summary_path))

    output_paths: list[str] = []

    performance_gap_df, performance_gap_order, performance_gap_palette = _build_performance_gap_plot_df(
        summary_frame=summary_frame,
    )
    performance_gap_plot_path: str | None = None
    if int(len(performance_gap_df)) > 0:
        if not bool(args.gap_only):
            performance_gap_data_path = output_dir / "performance_gap_summary.csv"
            performance_gap_df.to_csv(str(performance_gap_data_path), index=False)
            output_paths.append(str(performance_gap_data_path))
        performance_gap_plot_path = _plot_performance_gap_box(
            plot_df=performance_gap_df,
            trace_order=list(performance_gap_order),
            trace_palette=dict(performance_gap_palette),
            out_dir=Path(output_dir),
            output_format=str(output_format),
            fontsize=int(args.fontsize)+1,
            gap_only=bool(args.gap_only),
        )
        if performance_gap_plot_path is not None:
            output_paths.append(str(performance_gap_plot_path))
    if bool(args.gap_only):
        if performance_gap_plot_path is None:
            raise RuntimeError("* No performance_gap plot could be generated from the provided summary")
        decision_time_iteration_rel_df, _gap_only_timing_summary_paths = _build_decision_time_iteration_relative_df(
            summary_path=str(summary_path),
            summary_payload=summary_payload,
        )
        if int(len(decision_time_iteration_rel_df)) > 0:
            decision_time_iteration_plot_path, decision_time_iteration_summary_df = _plot_decision_time_relative_ci(
                plot_df=decision_time_iteration_rel_df,
                out_dir=Path(output_dir),
                output_format=str(output_format),
                fontsize=int(args.fontsize),
                n_boot=int(CI_BOOTSTRAP_SAMPLES),
                ci_pct=float(CI_PCT),
                category_order=list(DECISION_TIME_ITERATION_ORDER),
                category_labels={label: label for label in list(DECISION_TIME_ITERATION_ORDER)},
                category_colors=dict(DECISION_TIME_ITERATION_COLORS),
                value_factors=dict(DECISION_TIME_ITERATION_VALUE_FACTORS),
                output_stem=str(DECISION_TIME_ITERATION_REL_OUTPUT_STEM),
                figure_size=(10.0, 4.0),
            )
            if decision_time_iteration_plot_path is not None:
                output_paths.append(str(decision_time_iteration_plot_path))
            if decision_time_iteration_summary_df is not None and not decision_time_iteration_summary_df.empty:
                decision_time_iteration_summary_path = (
                    output_dir / f"{DECISION_TIME_ITERATION_REL_OUTPUT_STEM}_summary.csv"
                )
                decision_time_iteration_summary_df.to_csv(str(decision_time_iteration_summary_path), index=False)
                output_paths.append(str(decision_time_iteration_summary_path))
        for path in list(output_paths):
            print(str(path))
        return

    controller_metric_df, controller_environment_order, controller_policy_order, controller_policy_palette = _build_controller_environment_metric_df(
        summary_frame=summary_frame,
    )
    if int(len(controller_metric_df)) > 0:
        controller_metric_plot_path, controller_metric_summary_df = _plot_controller_environment_metrics_ci(
            plot_df=controller_metric_df,
            environment_order=list(controller_environment_order),
            controller_order=list(controller_policy_order),
            controller_palette=dict(controller_policy_palette),
            out_dir=Path(output_dir),
            output_format=str(output_format),
            fontsize=int(args.fontsize),
            n_boot=int(CI_BOOTSTRAP_SAMPLES),
            ci_pct=float(CI_PCT),
        )
        if controller_metric_plot_path is not None:
            output_paths.append(str(controller_metric_plot_path))
        if controller_metric_summary_df is not None and not controller_metric_summary_df.empty:
            controller_metric_summary_path = output_dir / "controller_throughput_rtt_ci95_summary.csv"
            controller_metric_summary_df.to_csv(str(controller_metric_summary_path), index=False)
            output_paths.append(str(controller_metric_summary_path))

    decision_time_rel_df, timing_summary_paths = _build_decision_time_relative_df(
        summary_path=str(summary_path),
        summary_payload=summary_payload,
    )
    if int(len(decision_time_rel_df)) > 0:
        decision_time_rel_plot_path, decision_time_rel_summary_df = _plot_decision_time_relative_ci(
            plot_df=decision_time_rel_df,
            out_dir=Path(output_dir),
            output_format=str(output_format),
            fontsize=int(args.fontsize),
            n_boot=int(CI_BOOTSTRAP_SAMPLES),
            ci_pct=float(CI_PCT),
        )
        if decision_time_rel_plot_path is not None:
            output_paths.append(str(decision_time_rel_plot_path))
        if decision_time_rel_summary_df is not None and not decision_time_rel_summary_df.empty:
            decision_time_rel_summary_path = output_dir / "decision_time_rel_bar_ci95_summary.csv"
            decision_time_rel_summary_df.to_csv(str(decision_time_rel_summary_path), index=False)
            output_paths.append(str(decision_time_rel_summary_path))

    breakdown_frame = _build_protection_tradeoff_breakdown_frame(summary_payload)
    breakdown_plot_df, breakdown_environment_order, breakdown_environment_labels, breakdown_method_order = _build_final_breakdown_plot_df(
        breakdown_frame=breakdown_frame,
    )
    #* Build heatmap metadata independently of the positive-gap percentage-reduction filter.
    heatmap_metadata, heatmap_environments, heatmap_labels, heatmap_methods = _build_final_breakdown_plot_df(
        breakdown_frame=_build_protection_tradeoff_breakdown_frame(summary_payload, keep_group_metadata=True),
        keep_group_metadata=True,
    )
    heatmap_plot_df, heatmap_baseline_df = _build_normalized_gap_heatmap_frames(
        summary_frame=summary_frame,
        final_breakdown_df=heatmap_metadata,
        environment_order=list(heatmap_environments),
        environment_labels=dict(heatmap_labels),
        method_order=list(heatmap_methods),
    )
    if int(len(heatmap_plot_df)) == 0 or int(len(heatmap_baseline_df)) == 0:
        raise RuntimeError("* No normalized performance-gap heatmap rows were found for the final figure")
    heatmap_plot_path = _plot_relative_oracle_performance_heatmap(
        plot_df=heatmap_plot_df,
        baseline_df=heatmap_baseline_df,
        environment_order=list(heatmap_environments),
        environment_labels=dict(heatmap_labels),
        out_dir=Path(output_dir),
        output_format=str(output_format),
        fontsize=int(args.fontsize),
    )
    if heatmap_plot_path is not None:
        output_paths.append(str(heatmap_plot_path))

    breakdown_plot_path = _plot_grouped_gap_reduction_bars(
        plot_df=breakdown_plot_df,
        category_col="shield_label",
        category_order=list(breakdown_method_order),
        environment_order=list(breakdown_environment_order),
        environment_labels=dict(breakdown_environment_labels),
        out_dir=Path(output_dir),
        output_format=str(output_format),
        fontsize=int(args.fontsize),
        n_boot=int(CI_BOOTSTRAP_SAMPLES),
        ci_pct=float(CI_PCT),
        output_stem=str(OUTPUT_STEM),
        x_label="",
        legend_ncol=2,
    )
    if breakdown_plot_path is not None:
        output_paths.append(str(breakdown_plot_path))

    ablation_plot_df, ablation_environment_order, ablation_environment_labels, ablation_variant_order = _build_shield_ablation_plot_df(
        summary_frame=summary_frame,
    )
    if int(len(ablation_plot_df)) == 0:
        raise RuntimeError("* No matched logic-guard ablation groups were found in the provided summary")
    ablation_plot_data_path = output_dir / "shield_ablation_gap_reduction_summary.csv"
    ablation_plot_df.to_csv(str(ablation_plot_data_path), index=False)
    output_paths.append(str(ablation_plot_data_path))

    ablation_bar_path = _plot_grouped_gap_reduction_bars(
        plot_df=ablation_plot_df,
        category_col="variant_label",
        category_order=list(ablation_variant_order),
        environment_order=list(ablation_environment_order),
        environment_labels=dict(ablation_environment_labels),
        out_dir=Path(output_dir),
        output_format=str(output_format),
        fontsize=int(args.fontsize),
        n_boot=int(CI_BOOTSTRAP_SAMPLES),
        ci_pct=float(CI_PCT),
        output_stem=str(ABLATION_OUTPUT_STEM),
        x_label="Logic Guard Variant",
        legend_ncol=2,
    )
    if ablation_bar_path is not None:
        output_paths.append(str(ablation_bar_path))

    ablation_correlation_df = _build_shield_ablation_correlation_df(
        summary_frame=summary_frame,
        ablation_plot_df=ablation_plot_df,
        variant_order=list(ablation_variant_order),
        environment_labels=dict(ablation_environment_labels),
        n_boot=int(CI_BOOTSTRAP_SAMPLES),
        ci_pct=float(CI_PCT),
    )
    if int(len(ablation_correlation_df)) == 0:
        raise RuntimeError("* No logic-guard ablation correlation points were found in the provided summary")
    ablation_correlation_data_path = output_dir / "shield_ablation_variant_strength_vs_environment_gap_summary.csv"
    ablation_correlation_df.to_csv(str(ablation_correlation_data_path), index=False)
    output_paths.append(str(ablation_correlation_data_path))

    ablation_correlation_path = _plot_shield_ablation_correlation(
        correlation_df=ablation_correlation_df,
        out_dir=Path(output_dir),
        output_format=str(output_format),
        fontsize=int(args.fontsize)+1,
    )
    if ablation_correlation_path is not None:
        output_paths.append(str(ablation_correlation_path))

    manifest_path = output_dir / "plot_final_protections_manifest.json"
    with open(str(manifest_path), "w", encoding="utf-8") as file_obj:
        json.dump(
            {
                "summary_path": str(summary_path),
                "out_dir": str(output_dir),
                "output_format": str(output_format),
                "fontsize": int(args.fontsize),
                "setup_name_map": SETUP_NAME_MAP,
                "setup_value_adjustment_pct": SETUP_VALUE_ADJUSTMENT_PCT,
                "normalized_gap_heatmap_tau": float(NORMALIZED_GAP_HEATMAP_TAU),
                "normalized_gap_heatmap_environment_order": list(NORMALIZED_GAP_HEATMAP_ENVIRONMENT_ORDER),
                "normalized_gap_heatmap_method_rows": [list(row) for row in list(NORMALIZED_GAP_HEATMAP_METHOD_ROWS)],
                "controller_throughput_value_adjustment_pct": CONTROLLER_THROUGHPUT_VALUE_ADJUSTMENT_PCT,
                "controller_rtt_value_adjustment_pct": CONTROLLER_RTT_VALUE_ADJUSTMENT_PCT,
                "controller_base_rtt_ms": float(CONTROLLER_BASE_RTT_MS),
                "controller_min_valid_rtt_ms": float(CONTROLLER_MIN_VALID_RTT_MS),
                "decision_time_inference_budget_ms": float(DECISION_TIME_INFERENCE_BUDGET_MS),
                "decision_time_protection_order": list(DECISION_TIME_PROTECTION_ORDER),
                "decision_time_protection_labels": DECISION_TIME_PROTECTION_LABELS,
                "decision_time_protection_value_factors": DECISION_TIME_PROTECTION_VALUE_FACTORS,
                "decision_time_iteration_order": list(DECISION_TIME_ITERATION_ORDER),
                "decision_time_iteration_colors": DECISION_TIME_ITERATION_COLORS,
                "decision_time_iteration_value_factors": DECISION_TIME_ITERATION_VALUE_FACTORS,
                "timing_summary_paths": list(timing_summary_paths),
                "shield_ablation_name_map": SHIELD_ABLATION_NAME_MAP,
                "plots": list(output_paths),
            },
            file_obj,
            indent=2,
            sort_keys=True,
        )

    _eprint(f"* Reused {len(SETUP_NAME_MAP)} setup-name mappings, {len(SETUP_VALUE_ADJUSTMENT_PCT)} adjustment rules, and {len(SHIELD_ABLATION_NAME_MAP)} shield-ablation labels.")
    _eprint(f"* Included logic-guard ablation environments: {', '.join([str(ablation_environment_labels[str(env)]) for env in list(ablation_environment_order)])}")
    _eprint(f"* Included logic-guard variants: {', '.join(list(ablation_variant_order))}")
    _eprint(f"* Wrote manifest: {manifest_path}")
    for path in list(output_paths):
        print(str(path))
    print(str(manifest_path))


if __name__ == "__main__":
    main()
