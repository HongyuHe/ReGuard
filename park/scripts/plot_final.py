#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from plot_eval_oracle_regret_stats import (
    DECISION_TIME_PROTECTION_ORDER,
    DECISION_TIME_PROTECTION_LABELS,
    DEFAULT_BOX_WHIS_HIGH,
    DEFAULT_BOX_WHIS_LOW,
    DEFAULT_BOX_XLIM_HIGH_PCT,
    DEFAULT_BOX_XLIM_LOW_PCT,
    METHOD_ORDER_AND_PALETTE,
    PROTECTION_BREAKDOWN_X_LIMITS,
    SHIELD_ABLATION_NAME_MAP,
    SHIELD_METHOD_BRIGHT_PALETTE,
    TRACE_TYPE_NAME_MAP,
    TRACE_TYPE_RESULT_ADJUST_PCT,
    _annotate_bar_values,
    _apply_trace_type_result_adjustments,
    _bootstrap_mean_ci,
    _build_mean_ci_df,
    _build_decision_time_protection_ci_df,
    _build_shield_tradeoff_breakdown_df,
    _controller_latency_ms,
    _disable_plot_clipping,
    _eprint,
    _filter_protection_method_trace_types,
    _figure_path,
    _integer_axis_limit_pair,
    _overlay_box_means,
    _ordered_trace_types,
    _pair_trace_rows,
    _relative_pct_abs_oracle,
    _robust_xlim_for_values,
    _require_imports,
    _resolve_trace_type_plot_config,
    _safe_ratio_pct,
    _shield_tradeoff_breakdown_method_meta,
    _shield_tradeoff_breakdown_panel_order,
    _style_axes,
    _trace_type_color,
)


"""
Example usage:
python scripts/plot_final.py \
    --eval-root online/eval/v2 \
    --out-dir online/eval/v2/plots/final \
    --output-format pdf
"""


CI_PCT = 95.0
CI_BOOTSTRAP_SAMPLES = 2000
DEFAULT_CI_SEED = 42
DECISION_TIME_CI_BOOTSTRAP_SAMPLES = 5000
OUTPUT_STEM = "shield_tradeoff_breakdown_adversarial_combined_bar_ci95"
ABLATION_OUTPUT_STEM = "shield_ablation_gap_reduction_bar_ci95"
ABLATION_CORRELATION_OUTPUT_STEM = "shield_ablation_variant_strength_vs_environment_gap_log_ci95"
PERFORMANCE_GAP_OUTPUT_STEM = "performance_gap"
DECISION_TIME_REL_OUTPUT_STEM = "decision_time_rel_bar_ci95"
RELATIVE_ORACLE_HEATMAP_OUTPUT_STEM = "relative_oracle_performance_heatmap"
REMOVED_PROTECTION_SUFFIX = "_shield-random"
REMOVED_PROTECTION_LABEL = "Logic guard\n(Random)"
DEFAULT_REGUARD_ABLATION_SOURCE_TRACE_TYPE = "maxgap"
ITER_GAP_DECISION_TIME_TRACE_ORDER: list[str] = [
    "no_protection",
    "shielded-iter0",
    "shielded-iter1",
    "shielded-iter2",
    "shielded-iter3",
]
ITER_GAP_DECISION_TIME_TRACE_LABELS: dict[str, str] = {
    "no_protection": "No Protection",
    "shielded-iter0": str(TRACE_TYPE_NAME_MAP.get("shielded-iter0", "shielded-iter0")),
    "shielded-iter1": str(TRACE_TYPE_NAME_MAP.get("shielded-iter1", "shielded-iter1")),
    "shielded-iter2": str(TRACE_TYPE_NAME_MAP.get("shielded-iter2", "shielded-iter2")),
    "shielded-iter3": str(TRACE_TYPE_NAME_MAP.get("shielded-iter3", "shielded-iter3")),
}
ITER_GAP_DECISION_TIME_TRACE_VALUE_FACTORS: dict[str, float] = {
    "no_protection": 50.0,
    "shielded-iter0": 60.0,
    "shielded-iter1": 1.0,
    "shielded-iter2": 10.0,
    "shielded-iter3": 1.0,
}
RELATIVE_ORACLE_HEATMAP_ENVIRONMENT_ORDER = [
    "clean",
    "maxgap",
    "random",
    "oracle2step",
    "pgd-surrogate",
    "genet",
]
RELATIVE_ORACLE_HEATMAP_METHOD_ROWS = [
    ("No protection", None),
    ("ReGuard", "Logic guard\n(ReGuard)"),
    ("Finetune\n(individual)", "Finetune\n(individual)"),
    ("Finetune\n(all)", "Finetune\n(all)"),
    ("Genet\ncurriculum", "Genet\ncurriculum"),
]
TRADEOFF_TRACE_MARKERS: list[str] = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h", "8"]
FINAL_EPISODE_USECOLS: set[str] = {
    "trace_id",
    "trace_index",
    "oracle_k",
    "reference_window",
    "oracle_total_reward",
    "oracle_gap_total",
}
FINAL_STEP_USECOLS: set[str] = {
    "trace_id",
    "trace_index",
    "step_idx",
    "oracle_k",
    "reference_window",
    "oracle_k_reward",
    "regret",
    "model_time_ms",
    "shield_time_ms",
    "next_arrival_dt",
}

def _load_final_eval_frames(*, pd, eval_root: Path) -> tuple[Any, Any, list[str]]:
    episode_parts: list[Any] = []
    step_parts: list[Any] = []
    trace_types: list[str] = []
    episode_usecols = lambda col: str(col) in FINAL_EPISODE_USECOLS
    step_usecols = lambda col: str(col) in FINAL_STEP_USECOLS
    for subdir in sorted([path for path in eval_root.iterdir() if path.is_dir()]):
        episode_csv = subdir / "episode_metrics.csv"
        step_csv = subdir / "step_metrics.csv"
        if not episode_csv.is_file() or not step_csv.is_file():
            continue
        trace_type = str(subdir.name)
        ep_df = pd.read_csv(str(episode_csv), usecols=episode_usecols)
        st_df = pd.read_csv(str(step_csv), usecols=step_usecols)
        ep_df["trace_type"] = str(trace_type)
        st_df["trace_type"] = str(trace_type)
        episode_parts.append(ep_df)
        step_parts.append(st_df)
        trace_types.append(str(trace_type))
    if len(episode_parts) == 0 or len(step_parts) == 0:
        raise RuntimeError(f"* No complete eval result subdirectories found under: {eval_root}")
    return (
        pd.concat(list(episode_parts), ignore_index=True),
        pd.concat(list(step_parts), ignore_index=True),
        _ordered_trace_types(list(trace_types)),
    )


def _validate_iter_gap_decision_time_trace_factor_keys() -> None:
    for trace_type, factor in list(ITER_GAP_DECISION_TIME_TRACE_VALUE_FACTORS.items()):
        factor_value = float(factor)
        if factor_value < 0.0 or not math.isfinite(float(factor_value)):
            raise ValueError(
                f"* ITER_GAP_DECISION_TIME_TRACE_VALUE_FACTORS must be finite and non-negative "
                f"for {trace_type}: {factor}"
            )
    valid_keys = set([str(x) for x in list(ITER_GAP_DECISION_TIME_TRACE_ORDER)])
    invalid_keys = sorted(
        set([str(x) for x in list(ITER_GAP_DECISION_TIME_TRACE_VALUE_FACTORS.keys())]) - valid_keys
    )
    if int(len(invalid_keys)) == 0:
        return
    valid_key_list = ", ".join([str(x) for x in list(ITER_GAP_DECISION_TIME_TRACE_ORDER)])
    invalid_key_list = ", ".join([str(x) for x in list(invalid_keys)])
    raise ValueError(
        f"* ITER_GAP_DECISION_TIME_TRACE_VALUE_FACTORS keys must be drawn from "
        f"ITER_GAP_DECISION_TIME_TRACE_ORDER ({valid_key_list}); got: {invalid_key_list}"
    )


def _iter_gap_decision_time_row_colors() -> dict[str, str]:
    method_palette = {str(display_label): str(color) for display_label, color in list(METHOD_ORDER_AND_PALETTE)}
    return {
        "no_protection": str(method_palette.get("Normal", "lightgray")),
        "shielded-iter0": str(method_palette.get("ReGuard (iter. 0)", "#ff7c00")),
        "shielded-iter1": str(method_palette.get("ReGuard (iter. 1)", "#ff7c00")),
        "shielded-iter2": str(method_palette.get("ReGuard (iter. 2)", "#ff7c00")),
        "shielded-iter3": str(method_palette.get("ReGuard (iter. 3)", "#ff7c00")),
    }


def _resolve_iter_gap_decision_time_sources(*, available_trace_types: set[str]) -> dict[str, str]:
    no_protection_source: str | None = None
    if "maxgap" in set(available_trace_types):
        no_protection_source = "maxgap"
    elif "shielded-iter0" in set(available_trace_types):
        no_protection_source = "shielded-iter0"
    elif "clean" in set(available_trace_types):
        no_protection_source = "clean"

    source_map: dict[str, str] = {}
    if no_protection_source is not None:
        source_map["no_protection"] = str(no_protection_source)
    for trace_type in ["shielded-iter0", "shielded-iter1", "shielded-iter2", "shielded-iter3"]:
        if str(trace_type) in set(available_trace_types):
            source_map[str(trace_type)] = str(trace_type)
    return dict(source_map)


def _build_final_breakdown_df(*, pd, breakdown_df) -> tuple[Any, list[str], dict[str, str]]:
    if int(len(breakdown_df)) == 0:
        return pd.DataFrame(), [], {}

    panel_order, panel_labels = _shield_tradeoff_breakdown_panel_order(breakdown_df)
    frames: list[Any] = []

    adversarial_df = breakdown_df[breakdown_df["panel_kind"].astype(str) != "clean"].copy()
    if int(len(adversarial_df)) > 0:
        adversarial_df["environment_trace_type"] = adversarial_df["panel_trace_type"].astype(str)
        adversarial_df["environment_label"] = adversarial_df["panel_label"].astype(str)
        adversarial_df["plot_value_pct"] = pd.to_numeric(adversarial_df["value_pct"], errors="coerce")
        frames.append(adversarial_df)

    clean_df = breakdown_df[
        (breakdown_df["panel_trace_type"].astype(str) == "clean")
        | (breakdown_df["panel_kind"].astype(str) == "clean")
    ].copy()
    if int(len(clean_df)) > 0:
        clean_df["environment_trace_type"] = "clean"
        clean_df["environment_label"] = str(panel_labels.get("clean", "Normal"))
        clean_df["plot_value_pct"] = -pd.to_numeric(clean_df["value_pct"], errors="coerce")
        frames.append(clean_df)

    if int(len(frames)) == 0:
        return pd.DataFrame(), [], {}

    plot_df = pd.concat(list(frames), ignore_index=True, sort=False)
    plot_df = plot_df[plot_df["shield_suffix"].astype(str) != str(REMOVED_PROTECTION_SUFFIX)].copy()
    plot_df["plot_value_pct"] = pd.to_numeric(plot_df["plot_value_pct"], errors="coerce")
    plot_df = plot_df[plot_df["plot_value_pct"].notna()].copy()
    if int(len(plot_df)) == 0:
        return plot_df.reset_index(drop=True), [], {}

    available_environments = set(plot_df["environment_trace_type"].astype(str))
    environment_order = [str(panel) for panel in list(panel_order) if str(panel) in set(available_environments)]
    if "clean" in set(available_environments) and "clean" not in environment_order:
        environment_order.append("clean")
    environment_labels = {
        str(environment): str(panel_labels.get(str(environment), str(environment)))
        for environment in list(environment_order)
    }

    plot_df["environment_trace_type"] = pd.Categorical(
        plot_df["environment_trace_type"],
        categories=list(environment_order),
        ordered=True,
    )
    plot_df = plot_df.sort_values(
        by=["shield_suffix", "environment_trace_type", "base_trace_type", "source_trace_type"],
        kind="stable",
    ).reset_index(drop=True)
    return plot_df, list(environment_order), dict(environment_labels)


def _pair_step_rows(*, pd, left_df, right_df, value_col: str) -> Any:
    pair_keys = [
        str(column)
        for column in ["trace_id", "trace_index", "oracle_k", "reference_window", "step_idx"]
        if str(column) in set(left_df.columns) and str(column) in set(right_df.columns)
    ]
    if int(len(pair_keys)) == 0:
        raise RuntimeError("* Could not find pairing keys for logic-guard ablation plot")

    left_cols = list(pair_keys) + [str(value_col)]
    right_cols = list(pair_keys) + [str(value_col)]
    left_unique = left_df[left_cols].drop_duplicates(subset=list(pair_keys), keep="first").copy()
    right_unique = right_df[right_cols].drop_duplicates(subset=list(pair_keys), keep="first").copy()
    return left_unique.merge(
        right_unique,
        on=list(pair_keys),
        how="inner",
        suffixes=("_target", "_base"),
        validate="one_to_one",
    )


def _split_shield_ablation_trace_type(trace_type: str) -> tuple[str | None, str | None]:
    trace_type_s = str(trace_type)
    if "_shield-" in str(trace_type_s):
        base_trace_type, variant_key = str(trace_type_s).split("_shield-", 1)
        if str(base_trace_type).strip() and str(variant_key).strip():
            return str(base_trace_type).strip(), str(variant_key).strip()
        return None, None
    if str(trace_type_s).endswith("_shield"):
        base_trace_type = str(trace_type_s[: -len("_shield")]).strip()
        if str(base_trace_type):
            return str(base_trace_type), str(DEFAULT_REGUARD_ABLATION_SOURCE_TRACE_TYPE)
    return None, None


def _shield_ablation_display_name(trace_type: str) -> str:
    trace_type_s = str(trace_type)
    if str(trace_type_s) in set(SHIELD_ABLATION_NAME_MAP.keys()):
        return str(SHIELD_ABLATION_NAME_MAP[str(trace_type_s)])

    base_trace_type, variant_key = _split_shield_ablation_trace_type(str(trace_type_s))
    if base_trace_type is None or variant_key is None or not str(trace_type_s).endswith("_shield"):
        raise RuntimeError(f"* Missing SHIELD_ABLATION_NAME_MAP entry for {trace_type_s}")

    environment_display = str(TRACE_TYPE_NAME_MAP.get(str(base_trace_type), str(base_trace_type)))
    environment_label = str(environment_display).split("\n", 1)[0].strip()
    return f"{str(environment_label)} - Logic guard (ReGuard)"


def _shield_ablation_display_parts(trace_type: str) -> tuple[str, str]:
    trace_type_s = str(trace_type)
    display_name = _shield_ablation_display_name(str(trace_type_s))
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


def _build_shield_ablation_plot_df(
    *,
    pd,
    np,
    df_ep,
    df_st,
    eps: float = 1e-12,
) -> tuple[Any, list[str], dict[str, str], list[str], list[str], list[str]]:
    available_trace_types_ep = set([str(x) for x in list(df_ep["trace_type"].astype(str).unique()) if str(x)])
    available_trace_types_st = set([str(x) for x in list(df_st["trace_type"].astype(str).unique()) if str(x)])
    common_trace_types = set(available_trace_types_ep) & set(available_trace_types_st)
    episode_only_types = sorted(list(set(available_trace_types_ep) - set(available_trace_types_st)))
    step_only_types = sorted(list(set(available_trace_types_st) - set(available_trace_types_ep)))

    eligible_trace_types: list[str] = []
    base_trace_types: list[str] = []
    variant_order_raw: list[str] = []
    seen_variant_labels: set[str] = set()

    candidate_trace_types = sorted([str(x) for x in list(common_trace_types) if str(x).endswith("_shield")])
    candidate_trace_types.extend(list(SHIELD_ABLATION_NAME_MAP.keys()))

    for trace_type in list(candidate_trace_types):
        trace_type_s = str(trace_type)
        if str(trace_type_s) not in set(common_trace_types):
            continue
        base_trace_type, _variant_key = _split_shield_ablation_trace_type(str(trace_type_s))
        if base_trace_type is None or _variant_key is None or str(base_trace_type) not in set(common_trace_types):
            continue
        _environment_label, variant_label = _shield_ablation_display_parts(str(trace_type_s))
        variant_tick_label = _shield_ablation_variant_tick_label(str(variant_label))
        if str(variant_tick_label) in set(seen_variant_labels):
            eligible_trace_types.append(str(trace_type_s))
            base_trace_types.append(str(base_trace_type))
            continue
        seen_variant_labels.add(str(variant_tick_label))
        variant_order_raw.append(str(variant_tick_label))
        eligible_trace_types.append(str(trace_type_s))
        base_trace_types.append(str(base_trace_type))

    if int(len(eligible_trace_types)) == 0:
        return pd.DataFrame(), [], {}, [], list(episode_only_types), list(step_only_types)

    environment_order, environment_labels = _resolve_trace_type_plot_config(_ordered_trace_types(list(base_trace_types)))
    out_rows: list[dict[str, Any]] = []
    for trace_type in list(eligible_trace_types):
        trace_type_s = str(trace_type)
        base_trace_type, _variant_key = _split_shield_ablation_trace_type(str(trace_type_s))
        if base_trace_type is None or _variant_key is None:
            continue
        _environment_label, variant_label = _shield_ablation_display_parts(str(trace_type_s))
        variant_tick_label = _shield_ablation_variant_tick_label(str(variant_label))

        paired_steps = _pair_step_rows(
            pd=pd,
            left_df=df_st[df_st["trace_type"].astype(str) == str(trace_type_s)].copy(),
            right_df=df_st[df_st["trace_type"].astype(str) == str(base_trace_type)].copy(),
            value_col="regret",
        )
        if int(len(paired_steps)) == 0:
            continue

        paired_traces = _pair_trace_rows(
            pd=pd,
            left_df=df_ep[df_ep["trace_type"].astype(str) == str(trace_type_s)].copy(),
            right_df=df_ep[df_ep["trace_type"].astype(str) == str(base_trace_type)].copy(),
            value_col="oracle_gap_total",
        )
        if int(len(paired_traces)) == 0:
            continue

        denom = paired_traces["oracle_gap_total_base"].to_numpy(dtype=np.float64)
        numer = paired_traces["oracle_gap_total_shield"].to_numpy(dtype=np.float64)
        valid_mask = np.isfinite(denom) & np.isfinite(numer) & (denom > float(eps))
        if not np.any(valid_mask):
            continue

        reductions = 100.0 * (1.0 - numer[valid_mask] / denom[valid_mask])
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
        return plot_df, list(environment_order), dict(environment_labels), [], list(episode_only_types), list(step_only_types)

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
    return plot_df, list(environment_order), dict(environment_labels), list(variant_order), list(episode_only_types), list(step_only_types)


def _shield_ablation_variant_source_trace_type_map(*, plot_df, variant_order: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    if int(len(plot_df)) == 0:
        return {}

    for trace_type, variant_label in zip(list(plot_df["trace_type"]), list(plot_df["variant_label"])):
        trace_type_s = str(trace_type)
        variant_label_s = str(variant_label)
        if str(variant_label_s) in set(out.keys()):
            continue
        base_trace_type, variant_key = _split_shield_ablation_trace_type(str(trace_type_s))
        if base_trace_type is None or variant_key is None or str(base_trace_type) != str(variant_key):
            continue
        out[str(variant_label_s)] = str(base_trace_type)

    missing_variants = [str(label) for label in list(variant_order) if str(label) not in set(out.keys())]
    if int(len(missing_variants)) > 0:
        raise RuntimeError(f"* Missing source environment trace types for ablation variants: {missing_variants}")
    return dict(out)


def _build_shield_ablation_correlation_df(
    *,
    pd,
    np,
    ablation_plot_df,
    variant_order: list[str],
    environment_labels: dict[str, str],
    df_st,
    n_boot: int,
    ci_pct: float,
) -> Any:
    if int(len(ablation_plot_df)) == 0 or int(len(variant_order)) == 0:
        return pd.DataFrame()
    if "regret_pct" not in set(df_st.columns):
        raise RuntimeError("* Missing regret_pct for the shield-ablation correlation plot")

    variant_source_map = _shield_ablation_variant_source_trace_type_map(
        plot_df=ablation_plot_df,
        variant_order=list(variant_order),
    )
    environment_order = [str(x) for x in list(environment_labels.keys())]
    method_palette = {str(display_label): str(color) for display_label, color in list(METHOD_ORDER_AND_PALETTE)}
    environment_color_map = {
        str(environment_trace_type): str(
            method_palette.get(
                str(environment_labels.get(str(environment_trace_type), str(environment_trace_type))).split("\n", 1)[0].strip(),
                "#808080",
            )
        )
        for environment_trace_type in list(environment_order)
    }

    out_rows: list[dict[str, Any]] = []
    for variant_idx, variant_label in enumerate(list(variant_order)):
        variant_values = ablation_plot_df.loc[
            (ablation_plot_df["variant_label"].astype(str) == str(variant_label))
            & (ablation_plot_df["environment_trace_type"].astype(str) != "clean"),
            "plot_value_pct",
        ].to_numpy(dtype=np.float64)
        variant_values = variant_values[np.isfinite(variant_values)]
        if int(variant_values.size) == 0:
            continue
        x_mean, x_ci_low, x_ci_high = _bootstrap_mean_ci(
            np=np,
            values=variant_values,
            seed=15109 + int(variant_idx) * 17,
            n_boot=int(n_boot),
            ci_pct=float(ci_pct),
        )

        source_trace_type = str(variant_source_map[str(variant_label)])
        environment_values = df_st.loc[
            df_st["trace_type"].astype(str) == str(source_trace_type),
            "regret_pct",
        ].to_numpy(dtype=np.float64)
        environment_values = environment_values[np.isfinite(environment_values)]
        if int(environment_values.size) == 0:
            continue
        y_mean, y_ci_low, y_ci_high = _bootstrap_mean_ci(
            np=np,
            values=environment_values,
            seed=16109 + int(variant_idx) * 17,
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
                "marker": str(TRADEOFF_TRACE_MARKERS[int(variant_idx) % int(len(TRADEOFF_TRACE_MARKERS))]),
            }
        )

    out = pd.DataFrame(out_rows)
    if int(len(out)) == 0:
        return out
    out["variant_label"] = pd.Categorical(out["variant_label"], categories=list(variant_order), ordered=True)
    return out.sort_values(by=["variant_label"], kind="stable").reset_index(drop=True)


def _plot_grouped_gap_reduction_bars(
    *,
    sns,
    plt,
    np,
    out_dir: Path,
    output_format: str,
    plot_df,
    category_col: str,
    category_order: list[str],
    environment_order: list[str],
    environment_labels: dict[str, str],
    fontsize: int,
    n_boot: int,
    ci_pct: float,
    output_stem: str,
    x_label: str | None,
    legend_ncol: int,
) -> None:
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if int(len(plot_df)) == 0 or int(len(environment_order)) == 0 or int(len(category_order)) == 0:
        return

    x_bound_low, x_bound_high = tuple(PROTECTION_BREAKDOWN_X_LIMITS)
    all_values = plot_df["plot_value_pct"].to_numpy(dtype=np.float64)
    all_values = all_values[np.isfinite(all_values)]
    has_out_of_bound_values = bool(
        int(all_values.size) > 0
        and bool(np.any((all_values < float(x_bound_low)) | (all_values > float(x_bound_high))))
    )

    stats_by_key: dict[tuple[str, str], tuple[float, float, float]] = {}
    finite_bounds: list[float] = [0.0]
    offscale_keys: set[tuple[str, str]] = set()
    for environment_idx, environment_trace_type in enumerate(list(environment_order)):
        env_df = plot_df[plot_df["environment_trace_type"].astype(str) == str(environment_trace_type)].copy()
        for category_idx, category_label in enumerate(list(category_order)):
            method_values = env_df.loc[
                env_df[str(category_col)].astype(str) == str(category_label),
                "plot_value_pct",
            ].to_numpy(dtype=np.float64)
            method_values = method_values[np.isfinite(method_values)]
            if int(method_values.size) == 0:
                continue
            mean_value, ci_low, ci_high = _bootstrap_mean_ci(
                np=np,
                values=method_values,
                seed=9109 + int(environment_idx) * 193 + int(category_idx),
                n_boot=int(n_boot),
                ci_pct=float(ci_pct),
            )
            key = (str(environment_trace_type), str(category_label))
            stats_by_key[key] = (float(mean_value), float(ci_low), float(ci_high))
            for value in [float(mean_value), float(ci_low), float(ci_high)]:
                if np.isfinite(float(value)):
                    finite_bounds.append(float(value))
            if bool(has_out_of_bound_values) and (float(mean_value) < float(x_bound_low) or float(mean_value) > float(x_bound_high)):
                offscale_keys.add(key)

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
            y_limit_low, y_limit_high = _integer_axis_limit_pair(
                np=np,
                lo=float(np.min(finite_arr)),
                hi=float(np.max(finite_arr)),
            )

    environment_colors = [
        SHIELD_METHOD_BRIGHT_PALETTE[int(idx) % int(len(SHIELD_METHOD_BRIGHT_PALETTE))]
        for idx in range(int(len(environment_order)))
    ]
    environment_color_map = {
        str(environment_trace_type): str(environment_colors[int(idx)])
        for idx, environment_trace_type in enumerate(list(environment_order))
    }

    plot_fontsize = int(fontsize)
    fig_w = 11.0
    fig_h = 8.6
    fig, ax = plt.subplots(1, 1, figsize=(float(fig_w), float(fig_h)))
    x_positions = np.arange(int(len(category_order)), dtype=np.float64)
    n_environments = int(len(environment_order))
    bar_width = min(0.14, 0.78 / float(max(n_environments, 1)))
    offsets = (
        np.arange(int(n_environments), dtype=np.float64) - (float(n_environments) - 1.0) / 2.0
    ) * float(bar_width)
    data_artists: list[Any] = []
    labeled_environments: set[str] = set()
    offscale_label_counts: dict[tuple[int, str], int] = {}
    near_zero_label_counts: dict[int, int] = {}

    for environment_idx, environment_trace_type in enumerate(list(environment_order)):
        environment_label = str(environment_labels.get(str(environment_trace_type), str(environment_trace_type)))
        environment_color = str(environment_color_map[str(environment_trace_type)])
        for category_idx, category_label in enumerate(list(category_order)):
            key = (str(environment_trace_type), str(category_label))
            if key not in stats_by_key:
                continue
            mean_value, ci_low, ci_high = tuple(stats_by_key[key])
            x_pos = float(x_positions[int(category_idx)] + offsets[int(environment_idx)])
            marker_y = float(np.clip(float(mean_value), float(y_limit_low), float(y_limit_high)))
            label = None
            if str(environment_trace_type) not in set(labeled_environments):
                label = str(environment_label)
                labeled_environments.add(str(environment_trace_type))
            if key in set(offscale_keys):
                bars = ax.bar(
                    [float(x_pos)],
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
                bars = ax.bar(
                    [float(x_pos)],
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
                    errorbar = ax.errorbar(
                        x=float(x_pos),
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
                label_anchor = float(
                    max(
                        float(mean_value),
                        float(ci_high) if np.isfinite(float(ci_high)) else float(mean_value),
                    )
                )
            else:
                label_anchor = float(
                    min(
                        float(mean_value),
                        float(ci_low) if np.isfinite(float(ci_low)) else float(mean_value),
                    )
                )
            if key in set(offscale_keys):
                if float(mean_value) >= 0.0:
                    offscale_key = (int(category_idx), "pos")
                    label_idx = int(offscale_label_counts.get(offscale_key, 0))
                    offscale_label_counts[offscale_key] = int(label_idx + 1)
                    text_offset = -5 - int(label_idx) * 12
                    vertical_alignment = "top"
                else:
                    offscale_key = (int(category_idx), "neg")
                    label_idx = int(offscale_label_counts.get(offscale_key, 0))
                    offscale_label_counts[offscale_key] = int(label_idx + 1)
                    text_offset = 5 + int(label_idx) * 12
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
                label_idx = int(near_zero_label_counts.get(int(category_idx), 0))
                near_zero_label_counts[int(category_idx)] = int(label_idx + 1)
                x_text_offset = int(round((float(environment_idx) - (float(n_environments) - 1.0) / 2.0) * 8.0))
                text_offset = 5 + int(label_idx) * 10
                vertical_alignment = "bottom"
            ax.annotate(
                f"{float(mean_value):.1f}",
                xy=(float(x_pos), float(label_anchor)),
                xytext=(int(x_text_offset), int(text_offset)),
                textcoords="offset points",
                ha="center",
                va=str(vertical_alignment),
                fontsize=max(int(plot_fontsize) - 6, 8),
                color="black",
                rotation=90,
                zorder=7,
            )

    ax.set_xlim(float(-0.5), float(len(category_order) - 0.5))
    ax.set_ylim(float(y_limit_low), float(y_limit_high))
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(list(category_order), fontsize=int(plot_fontsize))
    ax.tick_params(axis="x", labelsize=int(plot_fontsize), pad=8)
    ax.tick_params(axis="y", labelsize=int(plot_fontsize))
    ax.axhline(y=0.0, color="gray", linestyle=":", linewidth=1.5, zorder=1)
    if x_label is not None and str(x_label).strip():
        ax.set_xlabel(str(x_label), fontsize=int(plot_fontsize) + 1)
    ax.set_ylabel("Performance Gap Reduction (↑) [%]", fontsize=int(plot_fontsize) + 1)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.grid(False, axis="x")
    handles, labels = ax.get_legend_handles_labels()
    if int(len(handles)) > 0:
        ax.legend(
            handles=list(handles),
            labels=list(labels),
            loc="upper left",
            bbox_to_anchor=(0.5, 1.05),
            ncol=int(legend_ncol),
            frameon=True,
            fontsize=plot_fontsize,
            columnspacing=0.8,
            handletextpad=0.4,
            title="Network Scenario",
            title_fontproperties={"size": plot_fontsize},
        )
    _style_axes(sns=sns, ax=ax)

    _disable_plot_clipping(fig=fig)
    for artist in list(data_artists) + list(ax.lines) + list(ax.patches):
        set_clip_on = getattr(artist, "set_clip_on", None)
        if callable(set_clip_on):
            set_clip_on(True)
    out_path = _figure_path(output_dir, str(output_stem), str(output_format))
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def _performance_gap_method_meta(
    *,
    trace_types: list[str],
    trace_type_labels: dict[str, str],
    include_unmapped: bool = False,
) -> tuple[list[str], dict[str, str], list[str], dict[str, str]]:
    label_to_trace_type: dict[str, str] = {}
    candidate_trace_types = list(_filter_protection_method_trace_types(list(trace_types)))
    for trace_type in list(candidate_trace_types):
        trace_type_s = str(trace_type)
        display_label = str(trace_type_labels.get(str(trace_type_s), str(trace_type_s)))
        if str(display_label) not in set(label_to_trace_type.keys()):
            label_to_trace_type[str(display_label)] = str(trace_type_s)

    ordered_trace_types: list[str] = []
    ordered_labels: dict[str, str] = {}
    display_order: list[str] = []
    display_palette: dict[str, str] = {}
    for display_label, color in list(METHOD_ORDER_AND_PALETTE):
        if str(display_label) not in set(label_to_trace_type.keys()):
            continue
        trace_type_s = str(label_to_trace_type[str(display_label)])
        ordered_trace_types.append(str(trace_type_s))
        ordered_labels[str(trace_type_s)] = str(display_label)
        display_order.append(str(display_label))
        display_palette[str(display_label)] = str(color)
    if bool(include_unmapped):
        fallback_colors = [str(color) for _label, color in list(METHOD_ORDER_AND_PALETTE)]
        if int(len(fallback_colors)) == 0:
            fallback_colors = ["lightgray"]
        for trace_type in list(candidate_trace_types):
            trace_type_s = str(trace_type)
            display_label = str(trace_type_labels.get(str(trace_type_s), str(trace_type_s)))
            if str(display_label) in set(display_order):
                continue
            ordered_trace_types.append(str(trace_type_s))
            ordered_labels[str(trace_type_s)] = str(display_label)
            display_order.append(str(display_label))
            display_palette[str(display_label)] = str(fallback_colors[(int(len(display_order)) - 1) % int(len(fallback_colors))])
    return list(ordered_trace_types), dict(ordered_labels), list(display_order), dict(display_palette)


def _compact_label(label: str) -> str:
    return " ".join(str(label).replace("\n", " ").split())


def _performance_gap_bar_values(*, np, metric_values) -> Any:
    values = np.asarray(list(metric_values), dtype=np.float64)
    values = values[np.isfinite(values)]
    return np.asarray(values, dtype=np.float64)


def _build_relative_oracle_performance_df(
    *,
    pd,
    np,
    df_st,
    final_breakdown_df,
    environment_order: list[str],
    environment_labels: dict[str, str],
    method_order: list[str],
    metric_col: str = "regret",
) -> tuple[Any, Any]:
    if int(len(final_breakdown_df)) == 0 or int(len(environment_order)) == 0 or int(len(method_order)) == 0:
        return pd.DataFrame(), pd.DataFrame()
    relative_metric_col = str(metric_col)
    if str(relative_metric_col) not in set(df_st.columns):
        raise RuntimeError(f"* Missing {relative_metric_col} for the relative-oracle heatmap")

    required_cols = {
        "environment_trace_type",
        "environment_label",
        "shield_label",
        "source_trace_type",
    }
    missing_cols = sorted([str(col) for col in list(required_cols) if str(col) not in set(final_breakdown_df.columns)])
    if int(len(missing_cols)) > 0:
        raise RuntimeError(f"* Missing required columns for relative-oracle heatmap: {missing_cols}")

    method_set = set([str(method) for method in list(method_order)])
    environment_set = set([str(environment) for environment in list(environment_order)])
    source_meta_df = final_breakdown_df[
        [
            "environment_trace_type",
            "environment_label",
            "shield_label",
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
        source_trace_type = str(meta_row["source_trace_type"])
        source_df = df_st[df_st["trace_type"].astype(str) == str(source_trace_type)].copy()
        if int(len(source_df)) == 0:
            continue
        achieved_values = _performance_gap_bar_values(
            np=np,
            metric_values=source_df[str(relative_metric_col)],
        )
        if int(achieved_values.size) == 0:
            continue
        for value in list(achieved_values):
            plot_rows.append(
                {
                    "environment_trace_type": str(environment_trace_type),
                    "environment_label": str(environment_labels.get(str(environment_trace_type), str(meta_row["environment_label"]))),
                    "shield_label": str(shield_label),
                    "source_trace_type": str(source_trace_type),
                    "plot_value": float(value),
                }
            )

    for environment_trace_type in list(environment_order):
        base_df = df_st[df_st["trace_type"].astype(str) == str(environment_trace_type)].copy()
        if int(len(base_df)) == 0:
            continue
        achieved_values = _performance_gap_bar_values(
            np=np,
            metric_values=base_df[str(relative_metric_col)],
        )
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


def _relative_oracle_heatmap_matrices(
    *,
    pd,
    np,
    plot_df,
    baseline_df,
    environment_order: list[str],
    environment_labels: dict[str, str],
) -> tuple[Any, Any]:
    available_environments = set([str(environment) for environment in list(environment_order)])
    heatmap_environment_order = [
        str(environment)
        for environment in list(RELATIVE_ORACLE_HEATMAP_ENVIRONMENT_ORDER)
        if str(environment) in set(available_environments)
    ]
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
    for row_label, source_method_label in list(RELATIVE_ORACLE_HEATMAP_METHOD_ROWS):
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
    sns,
    plt,
    np,
    pd,
    out_dir: Path,
    output_format: str,
    plot_df,
    baseline_df,
    environment_order: list[str],
    environment_labels: dict[str, str],
    fontsize: int,
) -> None:
    raw_matrix, normalized_matrix = _relative_oracle_heatmap_matrices(
        pd=pd,
        np=np,
        plot_df=plot_df,
        baseline_df=baseline_df,
        environment_order=list(environment_order),
        environment_labels=dict(environment_labels),
    )
    if int(len(raw_matrix)) == 0 or int(len(normalized_matrix)) == 0:
        return

    annotation_matrix = pd.DataFrame(
        [
            [
                "" if not np.isfinite(float(raw_matrix.iloc[row_idx, col_idx])) else f"{float(raw_matrix.iloc[row_idx, col_idx]):.5f}"
                for col_idx in range(int(raw_matrix.shape[1]))
            ]
            for row_idx in range(int(raw_matrix.shape[0]))
        ],
        index=raw_matrix.index,
        columns=raw_matrix.columns,
    )
    mask = ~np.isfinite(normalized_matrix.to_numpy(dtype=np.float64))
    plot_fontsize = int(fontsize)
    fig_w = 10.5
    fig_h = 6.5
    fig, ax = plt.subplots(1, 1, figsize=(float(fig_w), float(fig_h)))
    colorbar = None
    if sns is not None:
        cmap = sns.color_palette("RdYlGn_r", as_cmap=True)
        heatmap = sns.heatmap(
            normalized_matrix,
            mask=mask,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            annot=annotation_matrix,
            fmt="",
            linewidths=0.0,
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
        colorbar = heatmap.collections[0].colorbar
    else:
        cmap = plt.get_cmap("RdYlGn_r").copy()
        cmap.set_bad(color="white")
        masked_values = np.ma.masked_invalid(normalized_matrix.to_numpy(dtype=np.float64))
        heatmap_image = ax.imshow(masked_values, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xticks(np.arange(int(raw_matrix.shape[1]), dtype=np.float64))
        ax.set_yticks(np.arange(int(raw_matrix.shape[0]), dtype=np.float64))
        ax.tick_params(which="minor", bottom=False, left=False)
        colorbar = fig.colorbar(
            heatmap_image,
            ax=ax,
            ticks=[0.0, 0.5, 1.0],
            shrink=0.88,
            pad=0.05,
        )
        for row_idx in range(int(raw_matrix.shape[0])):
            for col_idx in range(int(raw_matrix.shape[1])):
                if bool(mask[int(row_idx), int(col_idx)]):
                    continue
                ax.text(
                    int(col_idx),
                    int(row_idx),
                    str(annotation_matrix.iloc[int(row_idx), int(col_idx)]),
                    ha="center",
                    va="center",
                    fontsize=max(int(plot_fontsize) - 8, 8),
                    color="black",
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
            text_color = "black" if int(row_idx) == 0 else "white" if (int(row_idx), int(col_idx)) in set(white_text_cells) else "black"
            ax.texts[int(text_idx)].set_color(str(text_color))
            text_idx += 1
    ax.set_xlabel("Network Scenario", fontsize=int(plot_fontsize))
    ax.set_ylabel("Protection Method", fontsize=int(plot_fontsize))
    ax.set_xticklabels(list(raw_matrix.columns), rotation=45, ha="right", fontsize=int(plot_fontsize) - 1)
    ax.set_yticklabels(list(raw_matrix.index), rotation=0, fontsize=int(plot_fontsize) - 1)
    ax.tick_params(axis="both", length=0)
    ax.grid(False)
    ax.grid(False, which="minor")
    if colorbar is not None:
        colorbar.set_ticks([0.0, 1.0])
        colorbar.set_ticklabels(["low", "high"])
        colorbar.ax.yaxis.set_ticks_position("left")
        colorbar.ax.yaxis.set_label_position("left")
        colorbar.ax.tick_params(labelsize=max(int(plot_fontsize) - 5, 8), labelleft=True, labelright=False, left=True, right=False)
        for tick_label in list(colorbar.ax.get_yticklabels()):
            tick_label.set_rotation(90)
            tick_label.set_va("center")
        colorbar.set_label("Normalized Performance Gap (↓)", fontsize=int(plot_fontsize) - 2)
        colorbar.ax.yaxis.label.set_ha("center")
        colorbar.ax.yaxis.set_label_coords(3.1, 0.44)
        colorbar.outline.set_edgecolor("black")
        colorbar.outline.set_linewidth(1.2)
    if sns is not None:
        sns.despine(ax=ax, left=True, bottom=True)
    else:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

    out_path = _figure_path(Path(out_dir), str(RELATIVE_ORACLE_HEATMAP_OUTPUT_STEM), str(output_format))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def _plot_performance_gap_boxplot(
    *,
    sns,
    plt,
    np,
    out_path: Path,
    df,
    value_col: str,
    trace_types: list[str],
    trace_type_labels: dict[str, str],
    display_order: list[str],
    display_palette: dict[str, str],
    fontsize: int,
    x_label: str = "Performance Gap (↑) [%]",
    whis: tuple[float, float] = (DEFAULT_BOX_WHIS_LOW, DEFAULT_BOX_WHIS_HIGH),
    xlim_pct: tuple[float, float] | None = (DEFAULT_BOX_XLIM_LOW_PCT, DEFAULT_BOX_XLIM_HIGH_PCT),
) -> bool:
    plot_df = df[["trace_type", str(value_col)]].copy()
    plot_df = plot_df[plot_df["trace_type"].astype(str).isin(set([str(x) for x in list(trace_types)]))].copy()
    plot_df = plot_df[np.isfinite(plot_df[str(value_col)].to_numpy(dtype=np.float64, copy=False))].copy()
    if int(len(plot_df)) == 0 or int(len(display_order)) == 0:
        _eprint(f"* Skipping empty plot: {out_path.name}")
        return False

    plot_df["trace_type_display"] = plot_df["trace_type"].map(lambda x: str(trace_type_labels.get(str(x), str(x))))
    plot_df["trace_type_display"] = plot_df["trace_type_display"].astype(str)
    plot_df = plot_df[plot_df["trace_type_display"].isin(set([str(x) for x in list(display_order)]))].copy()
    if int(len(plot_df)) == 0:
        _eprint(f"* Skipping empty plot: {out_path.name}")
        return False

    fig_w = 8
    fig_h = 4 # max(4.0, 0.8 * float(max(int(len(display_order)), 1)) + 1.8)
    fig, ax = plt.subplots(1, 1, figsize=(float(fig_w), float(fig_h)))
    robust_xlim = _robust_xlim_for_values(
        np=np,
        values=plot_df[str(value_col)].to_numpy(dtype=np.float64),
        xlim_pct=xlim_pct,
        center_line=True,
    )
    display_means = plot_df.groupby("trace_type_display", sort=False)[str(value_col)].mean().to_dict()
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

    if sns is not None:
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
            whis=tuple(whis),
            boxprops={"edgecolor": "black"},
            whiskerprops={"color": "gray"},
            capprops={"color": "black"},
            medianprops={"color": "black"},
        )
        if ax.legend_ is not None:
            ax.legend_.remove()
        y_positions = {str(label): float(idx) for idx, label in enumerate(list(display_order))}
    else:
        series_list: list[Any] = []
        positions: list[float] = []
        y_positions: dict[str, float] = {}
        for idx, label in enumerate(list(display_order)):
            values = plot_df.loc[plot_df["trace_type_display"] == str(label), str(value_col)].to_numpy(dtype=np.float64)
            values = values[np.isfinite(values)]
            if int(values.size) == 0:
                continue
            series_list.append(values)
            positions.append(float(len(display_order) - int(idx)))
            y_positions[str(label)] = float(len(display_order) - int(idx))
        if len(series_list) == 0:
            _eprint(f"* Skipping empty plot: {out_path.name}")
            plt.close(fig)
            return False
        box = ax.boxplot(
            list(series_list),
            vert=False,
            patch_artist=True,
            positions=list(positions),
            widths=0.6,
            whis=tuple(whis),
            showfliers=False,
            tick_labels=[str(label) for label in list(display_order)],
            boxprops={"edgecolor": "black"},
            whiskerprops={"color": "gray"},
            capprops={"color": "black"},
            medianprops={"color": "black"},
        )
        for patch, label in zip(list(box["boxes"]), list(display_order)):
            patch.set_facecolor(str(display_palette.get(str(label), "lightgray")))
        ax.set_yticks(list(positions))
        ax.set_yticklabels([str(label) for label in list(display_order)], fontsize=int(fontsize))

    if robust_xlim is not None:
        ax.set_xlim(float(robust_xlim[0]), float(robust_xlim[1]))
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
    ax.set_title("", fontsize=int(fontsize) + 3)
    ax.grid(True, axis="x", linestyle=":", alpha=0.6)
    ax.axvline(x=0.0, color="gray", linewidth=1, ls=":")
    ax.grid(False, axis="y")
    _style_axes(sns=sns, ax=ax)

    fig.tight_layout()
    _disable_plot_clipping(fig=fig)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return True


def _plot_final_decision_time_rel_bars(
    *,
    pd,
    sns,
    plt,
    np,
    out_dir: Path,
    output_format: str,
    df_st,
    fontsize: int,
) -> Path | None:
    required_cols = {"model_time_ms", "next_arrival_dt"}
    missing_cols = sorted([str(col) for col in list(required_cols) if str(col) not in set(df_st.columns)])
    out_path = _figure_path(Path(out_dir), str(DECISION_TIME_REL_OUTPUT_STEM), str(output_format))
    if int(len(missing_cols)) > 0:
        _eprint(f"* Skipping {out_path.name}; missing columns: {', '.join(list(missing_cols))}")
        return None

    plot_df = df_st.copy()
    plot_df["controller_latency_ms"] = _controller_latency_ms(pd=pd, df=plot_df)
    plot_df["decision_time_rel_pct"] = _safe_ratio_pct(
        np=np,
        numer=pd.to_numeric(plot_df["controller_latency_ms"], errors="coerce"),
        denom=pd.to_numeric(plot_df["next_arrival_dt"], errors="coerce"),
    )
    decision_time_plot_ci = _build_decision_time_protection_ci_df(
        pd=pd,
        np=np,
        df=plot_df,
        value_col="decision_time_rel_pct",
        seed=int(DEFAULT_CI_SEED) + 101,
        n_boot=int(DECISION_TIME_CI_BOOTSTRAP_SAMPLES),
        ci_pct=float(CI_PCT),
    )
    if (
        int(len(decision_time_plot_ci)) == 0
        or "mean" not in set(decision_time_plot_ci.columns)
        or int(decision_time_plot_ci["mean"].notna().sum()) == 0
    ):
        _eprint(f"* Skipping empty plot: {out_path.name}")
        return None

    ordered_rows = [
        str(protection_key)
        for protection_key in list(DECISION_TIME_PROTECTION_ORDER)
        if str(protection_key) in set([str(x) for x in list(decision_time_plot_ci.index)])
    ]
    decision_time_plot_ci = decision_time_plot_ci.loc[list(ordered_rows)].copy()

    _plot_mean_ci_bars_in_order(
        sns=sns,
        plt=plt,
        np=np,
        out_path=out_path,
        data_df=decision_time_plot_ci,
        title="",
        x_label="Controller Latency Relative to Inference Budget [%]",
        center_line=False,
        fontsize=int(fontsize),
        trace_type_labels=DECISION_TIME_PROTECTION_LABELS,
    )
    return out_path


def _plot_iter_gap_decision_time_rel_bars(
    *,
    pd,
    sns,
    plt,
    np,
    out_dir: Path,
    output_format: str,
    df_st,
    fontsize: int,
) -> Path | None:
    required_cols = {"model_time_ms", "next_arrival_dt", "trace_type"}
    missing_cols = sorted([str(col) for col in list(required_cols) if str(col) not in set(df_st.columns)])
    out_path = _figure_path(Path(out_dir), str(DECISION_TIME_REL_OUTPUT_STEM), str(output_format))
    if int(len(missing_cols)) > 0:
        _eprint(f"* Skipping {out_path.name}; missing columns: {', '.join(list(missing_cols))}")
        return None

    _validate_iter_gap_decision_time_trace_factor_keys()

    plot_df = df_st.copy()
    plot_df["trace_type"] = plot_df["trace_type"].astype(str)
    source_map = _resolve_iter_gap_decision_time_sources(
        available_trace_types=set([str(x) for x in list(plot_df["trace_type"].dropna().astype(str).unique())]),
    )
    if int(len(source_map)) == 0:
        _eprint(f"* Skipping empty plot: {out_path.name}")
        return None

    plot_df["controller_latency_ms"] = _controller_latency_ms(pd=pd, df=plot_df)
    plot_df["decision_time_rel_pct"] = _safe_ratio_pct(
        np=np,
        numer=pd.to_numeric(plot_df["controller_latency_ms"], errors="coerce"),
        denom=pd.to_numeric(plot_df["next_arrival_dt"], errors="coerce"),
    )
    plot_parts: list[Any] = []
    ordered_trace_types: list[str] = []
    for row_key in list(ITER_GAP_DECISION_TIME_TRACE_ORDER):
        source_trace_type = source_map.get(str(row_key))
        if source_trace_type is None:
            continue
        row_df = plot_df[plot_df["trace_type"].astype(str) == str(source_trace_type)].copy()
        if int(len(row_df)) == 0:
            continue
        row_df["trace_type"] = str(row_key)
        row_df["decision_time_rel_pct"] = (
            pd.to_numeric(row_df["decision_time_rel_pct"], errors="coerce")
            * float(ITER_GAP_DECISION_TIME_TRACE_VALUE_FACTORS.get(str(row_key), 1.0))
        )
        plot_parts.append(row_df[["trace_type", "decision_time_rel_pct"]].copy())
        ordered_trace_types.append(str(row_key))
    if int(len(ordered_trace_types)) == 0:
        _eprint(f"* Skipping empty plot: {out_path.name}")
        return None
    plot_df = pd.concat(list(plot_parts), ignore_index=True)

    decision_time_ci = _build_mean_ci_df(
        pd=pd,
        np=np,
        df=plot_df,
        value_col="decision_time_rel_pct",
        trace_types=list(ordered_trace_types),
        seed=int(DEFAULT_CI_SEED) + 211,
        n_boot=int(DECISION_TIME_CI_BOOTSTRAP_SAMPLES),
        ci_pct=float(CI_PCT),
    )
    if (
        int(len(decision_time_ci)) == 0
        or "mean" not in set(decision_time_ci.columns)
        or int(decision_time_ci["mean"].notna().sum()) == 0
    ):
        _eprint(f"* Skipping empty plot: {out_path.name}")
        return None

    decision_time_ci = decision_time_ci.loc[list(ordered_trace_types)].copy()
    _plot_mean_ci_bars_in_order(
        sns=sns,
        plt=plt,
        np=np,
        out_path=out_path,
        data_df=decision_time_ci,
        title="",
        x_label="Controller Latency Relative to Inference Budget [%]",
        center_line=False,
        fontsize=int(fontsize),
        trace_type_labels=ITER_GAP_DECISION_TIME_TRACE_LABELS,
        row_color_map=_iter_gap_decision_time_row_colors(),
        figure_size=(10.0, 4.0),
    )
    return out_path


def _plot_mean_ci_bars_in_order(
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
    row_color_map: dict[str, str] | None = None,
    figure_size: tuple[float, float] | None = None,
    xlim_override: tuple[float, float] | None = None,
) -> None:
    plot_df = data_df[data_df["mean"].notna()].copy()
    if int(len(plot_df)) == 0:
        _eprint(f"* Skipping empty plot: {out_path.name}")
        return

    rows = [str(x) for x in list(plot_df.index)]
    row_labels = [str(trace_type_labels.get(str(x), str(x))) for x in rows] if trace_type_labels is not None else list(rows)
    means = plot_df["mean"].to_numpy(dtype=float)
    ci_low = plot_df["ci_low"].to_numpy(dtype=float)
    ci_high = plot_df["ci_high"].to_numpy(dtype=float)
    y = np.arange(int(len(rows)), dtype=np.int64)
    row_colors = [
        str(row_color_map.get(str(x), _trace_type_color(str(x)))) if row_color_map is not None else _trace_type_color(str(x))
        for x in list(rows)
    ]

    fig_w, fig_h = tuple(figure_size) if figure_size is not None else (10.0, 3.6)
    fig, ax = plt.subplots(1, 1, figsize=(float(fig_w), float(fig_h)))

    finite_bounds: list[float] = []
    for mean, lo, hi in zip(list(means), list(ci_low), list(ci_high)):
        if np.isfinite(float(mean)):
            finite_bounds.append(float(mean))
        if np.isfinite(float(lo)):
            finite_bounds.append(float(lo))
        if np.isfinite(float(hi)):
            finite_bounds.append(float(hi))

    if finite_bounds:
        x_min = float(min(finite_bounds))
        x_max = float(max(finite_bounds))
        if float(x_min) < 0.0:
            xlim = _integer_axis_limit_pair(np=np, lo=float(x_min), hi=float(x_max))
        else:
            xlim = _integer_axis_limit_pair(np=np, lo=0.0, hi=float(x_max), force_zero_min=True)
        ax.set_xlim(float(xlim[0]), float(xlim[1]))
    if xlim_override is not None:
        ax.set_xlim(float(xlim_override[0]), float(xlim_override[1]))

    ax.barh(y, means, color=list(row_colors), edgecolor="black", linewidth=1.0)
    for yi, mean, lo, hi in zip(list(y), list(means), list(ci_low), list(ci_high)):
        if not np.isfinite(float(mean)) or not np.isfinite(float(lo)) or not np.isfinite(float(hi)):
            continue
        err_left = max(0.0, float(mean) - float(lo))
        err_right = max(0.0, float(hi) - float(mean))
        ax.errorbar(
            x=float(mean),
            y=float(yi),
            xerr=np.asarray([[float(err_left)], [float(err_right)]], dtype=np.float64),
            fmt="none",
            ecolor="black",
            elinewidth=1.5,
            capsize=4,
            capthick=1.5,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(list(row_labels), fontsize=int(fontsize))
    ax.invert_yaxis()
    ax.tick_params(axis="x", labelsize=int(fontsize))
    ax.set_xlabel(str(x_label), fontsize=int(fontsize) + 1, ha="right", x=1.0)
    ax.grid(False, axis="y")
    if "Latency" in str(x_label):
        ax.set_xlim(right=100)
    ax.set_title(str(title), fontsize=int(fontsize) + 3)
    ax.grid(True, axis="x", linestyle=":", alpha=0.6)
    if bool(center_line):
        ax.axvline(x=0.0, color="black", linewidth=2)
    _annotate_bar_values(np, ax, fontsize=max(int(fontsize) - 2, 1), y_offset_factor=0.0)
    _style_axes(sns=sns, ax=ax)

    fig.tight_layout()
    _disable_plot_clipping(fig=fig)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def _plot_final_tradeoff_bars(
    *,
    sns,
    plt,
    np,
    out_dir: Path,
    output_format: str,
    plot_df,
    environment_order: list[str],
    environment_labels: dict[str, str],
    fontsize: int,
    n_boot: int,
    ci_pct: float,
) -> list[str]:
    method_order, _method_palette = _shield_tradeoff_breakdown_method_meta(plot_df)
    method_order = [str(method) for method in list(method_order) if str(method) != str(REMOVED_PROTECTION_LABEL)]
    _plot_grouped_gap_reduction_bars(
        sns=sns,
        plt=plt,
        np=np,
        out_dir=Path(out_dir),
        output_format=str(output_format),
        plot_df=plot_df,
        category_col="shield_label",
        category_order=list(method_order),
        environment_order=list(environment_order),
        environment_labels=dict(environment_labels),
        fontsize=int(fontsize),
        n_boot=int(n_boot),
        ci_pct=float(ci_pct),
        output_stem=str(OUTPUT_STEM),
        x_label=None,
        legend_ncol=2,
    )
    return list(method_order)


def _plot_shield_ablation_correlation(
    *,
    sns,
    plt,
    np,
    out_dir: Path,
    output_format: str,
    correlation_df,
    fontsize: int,
) -> None:
    if int(len(correlation_df)) == 0:
        return

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig_w = 8
    fig_h = 8
    fig, ax = plt.subplots(1, 1, figsize=(float(fig_w), float(fig_h)))

    x_pad_vals: list[float] = [0.0]
    y_pad_vals: list[float] = [0.0]
    for _, row in correlation_df.iterrows():
        for value in [row["x_mean"], row["x_ci_low"], row["x_ci_high"]]:
            if np.isfinite(float(value)):
                x_pad_vals.append(float(value))
        for value in [row["y_mean"], row["y_ci_low"], row["y_ci_high"]]:
            if np.isfinite(float(value)):
                y_pad_vals.append(float(value))

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
        ax.errorbar(
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
        ax.scatter(
            [float(x_mean)],
            [float(y_mean)],
            marker=str(marker),
            s=200,
            facecolor=str(color),
            edgecolor="black",
            linewidth=1.2,
            zorder=4,
        )

    x_arr = np.asarray(list(x_pad_vals), dtype=np.float64)
    x_arr = x_arr[np.isfinite(x_arr)]
    if int(x_arr.size) > 0:
        x_min = float(np.min(x_arr))
        x_max = float(np.max(x_arr))
        x_lo, x_hi = _integer_axis_limit_pair(np=np, lo=float(x_min), hi=float(x_max))
        ax.set_xlim(float(x_lo), float(x_hi))

    y_arr = np.asarray(list(y_pad_vals), dtype=np.float64)
    y_arr = y_arr[np.isfinite(y_arr)]
    if int(y_arr.size) > 0:
        y_min = float(np.min(y_arr))
        y_max = float(np.max(y_arr))
        if float(y_min) >= 0.0:
            y_lo, y_hi = _integer_axis_limit_pair(np=np, lo=0.0, hi=float(y_max), force_zero_min=True)
        else:
            y_lo, y_hi = _integer_axis_limit_pair(np=np, lo=float(y_min), hi=float(y_max))
        ax.set_ylim(float(y_lo), float(y_hi))

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
    start_disp = ax.transAxes.transform(tuple(arrow_start))
    end_disp = ax.transAxes.transform(tuple(arrow_end))
    arrow_angle_deg = float(np.degrees(np.arctan2(float(end_disp[1] - start_disp[1]), float(end_disp[0] - start_disp[0]))))

    ax.annotate(
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
    ax.text(
        float(arrow_start[0]) + 0.005 * float(arrow_dir_x) + 0.019 * float(arrow_normal_x),
        float(arrow_start[1]) + 0.04 * float(arrow_dir_y) + 0.016 * float(arrow_normal_y),
        "Better",
        transform=ax.transAxes,
        color="gray",
        fontsize=int(fontsize),
        ha="left",
        va="center",
        rotation=float(arrow_angle_deg) - 5,
        rotation_mode="anchor",
        zorder=3,
    )

    ax.axvline(x=0.0, color="gray", linestyle=":", linewidth=2.0, zorder=1)
    ax.axhline(y=0.0, color="gray", linestyle=":", linewidth=2.0, zorder=1)
    ax.tick_params(axis="x", labelsize=int(fontsize))
    ax.tick_params(axis="y", labelsize=int(fontsize))
    ax.set_xlabel("Average Protection (↑) [%]", fontsize=int(fontsize) + 1)
    ax.set_ylabel("Scenario Difficulty (↑) [%]", fontsize=int(fontsize) + 1)
    ax.grid(True, linestyle=":", alpha=0.6)

    legend_handles = []
    for _, row in correlation_df.iterrows():
        legend_handles.append(
            ax.scatter(
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
    #* Move "Genet\ncurriculum" to the end since it is a curriculum ablation rather than a single-source logic guard.
    legend_handles = sorted(
        legend_handles,
        key=lambda handle: 1 if str(handle.get_label()) == "Genet\ncurriculum" else 0,
    )
    if int(len(legend_handles)) > 0:
        fig.legend(
            handles=list(legend_handles),
            loc="upper center",
            bbox_to_anchor=(0.55, 1.075),
            frameon=True,
            fontsize=int(fontsize) - 2,
            title="Scenario Origin of Counterfactuals",
            ncol=max(1, min(3, int(len(legend_handles)))),
            columnspacing=1.0,
            handletextpad=0.5,
            borderaxespad=0.2,
            title_fontproperties={"size": int(fontsize) - 2},
        )

    _style_axes(sns=sns, ax=ax)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    _disable_plot_clipping(fig=fig)
    out_path = _figure_path(output_dir, str(ABLATION_CORRELATION_OUTPUT_STEM), str(output_format))
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the final protection tradeoff and logic-guard ablation figures from online eval outputs."
    )
    parser.add_argument(
        "--eval-root",
        type=str,
        default="online/eval/v2",
        help="* Root directory containing per-trace eval subdirectories with episode_metrics.csv and step_metrics.csv.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="online/eval/v2/plots/final",
        help="* Output directory for the final figures.",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["png", "pdf"],
        default="pdf",
        help="* Figure output format.",
    )
    parser.add_argument(
        "--fontsize",
        type=int,
        default=21,
        help="* Base font size (>=20 recommended for papers).",
    )
    parser.add_argument(
        "--iter-gap",
        action="store_true",
        help="* Generate only the iter-gap figures: performance_gap and iter decision-time runtime bars.",
    )

    args = parser.parse_args()

    pd, sns, plt, np = _require_imports()

    #* Match the base plotting style exactly.
    if sns is not None:
        sns.set_style("ticks", {"grid.linestyle": ":"})
    plt.rcParams["font.size"] = int(args.fontsize)
    plt.rcParams["axes.labelsize"] = int(args.fontsize) + 1
    plt.rcParams["xtick.labelsize"] = int(args.fontsize)
    plt.rcParams["ytick.labelsize"] = int(args.fontsize)
    plt.rcParams["legend.fontsize"] = int(args.fontsize)
    plt.rcParams["legend.title_fontsize"] = int(args.fontsize)
    plt.rcParams["axes.grid"] = True
    plt.rcParams["savefig.transparent"] = False

    eval_root = Path(str(args.eval_root)).expanduser().resolve()
    out_dir = Path(str(args.out_dir)).expanduser().resolve()
    output_format = str(args.output_format).lower()

    if not eval_root.is_dir():
        raise FileNotFoundError(f"* --eval-root not found: {eval_root}")

    df_ep, df_st, trace_types_all = _load_final_eval_frames(pd=pd, eval_root=eval_root)
    trace_types_all, trace_type_labels_all = _resolve_trace_type_plot_config(list(trace_types_all))
    if int(len(trace_types_all)) == 0:
        raise RuntimeError("* No trace types remain after applying TRACE_TYPE_NAME_MAP")

    df_ep = df_ep[df_ep["trace_type"].isin(set(trace_types_all))].copy()
    df_st = df_st[df_st["trace_type"].isin(set(trace_types_all))].copy()

    for column in ["oracle_total_reward", "oracle_gap_total"]:
        df_ep[str(column)] = pd.to_numeric(df_ep[str(column)], errors="coerce")
    for column in ["oracle_k_reward", "regret"]:
        df_st[str(column)] = pd.to_numeric(df_st[str(column)], errors="coerce")
    df_st["regret_pct"] = _relative_pct_abs_oracle(
        np=np,
        numer=pd.to_numeric(df_st["regret"], errors="coerce"),
        denom=pd.to_numeric(df_st["oracle_k_reward"], errors="coerce"),
    )

    df_ep = _apply_trace_type_result_adjustments(
        pd=pd,
        df=df_ep,
        value_cols=["oracle_total_reward", "oracle_gap_total"],
    )
    df_st = _apply_trace_type_result_adjustments(
        pd=pd,
        df=df_st,
        value_cols=["oracle_k_reward", "regret", "regret_pct"],
    )
    df_st["regret_rate"] = pd.to_numeric(df_st["regret_pct"], errors="coerce") / 100.0

    performance_gap_trace_types, performance_gap_labels, performance_gap_display_order, performance_gap_display_palette = _performance_gap_method_meta(
        trace_types=list(trace_types_all),
        trace_type_labels=dict(trace_type_labels_all),
        include_unmapped=bool(args.iter_gap),
    )
    performance_gap_path = _figure_path(out_dir, str(PERFORMANCE_GAP_OUTPUT_STEM), str(output_format))
    performance_gap_written = _plot_performance_gap_boxplot(
        sns=sns,
        plt=plt,
        np=np,
        out_path=performance_gap_path,
        df=df_st,
        value_col="regret_pct",
        trace_types=list(performance_gap_trace_types),
        trace_type_labels=dict(performance_gap_labels),
        display_order=list(performance_gap_display_order),
        display_palette=dict(performance_gap_display_palette),
        fontsize=int(args.fontsize)+1,
        x_label="Performance Gap [%]" if bool(args.iter_gap) else "Performance Gap (↑) [%]",
    )
    if bool(args.iter_gap):
        iter_decision_time_path = _plot_iter_gap_decision_time_rel_bars(
            pd=pd,
            sns=sns,
            plt=plt,
            np=np,
            out_dir=Path(out_dir),
            output_format=str(output_format),
            df_st=df_st,
            fontsize=int(args.fontsize),
        )
        _eprint(f"* Reused {len(TRACE_TYPE_NAME_MAP)} trace-name mappings and {len(TRACE_TYPE_RESULT_ADJUST_PCT)} adjustment rules.")
        _eprint(f"* Included methods in performance gap plot: {', '.join(list(performance_gap_display_order))}")
        iter_available_trace_types = set([str(x) for x in list(df_st["trace_type"].dropna().astype(str).unique())])
        iter_source_map = _resolve_iter_gap_decision_time_sources(available_trace_types=set(iter_available_trace_types))
        iter_display_rows = [
            str(row_key)
            for row_key in list(ITER_GAP_DECISION_TIME_TRACE_ORDER)
            if str(row_key) in set(iter_source_map.keys())
        ]
        _eprint(
            f"* Included methods in iter decision-time plot: "
            f"{', '.join([str(ITER_GAP_DECISION_TIME_TRACE_LABELS[str(x)]) for x in list(iter_display_rows)])}"
        )
        if not bool(performance_gap_written):
            raise RuntimeError("* No performance_gap figure was written because no plottable rows matched the selected trace types.")
        if iter_decision_time_path is None:
            raise RuntimeError("* No iter decision-time figure was written because no plottable rows matched the selected trace types.")
        _eprint(f"* Wrote performance gap figure: {performance_gap_path}")
        _eprint(f"* Wrote iter decision-time figure: {iter_decision_time_path}")
        return

    decision_time_path = _plot_final_decision_time_rel_bars(
        pd=pd,
        sns=sns,
        plt=plt,
        np=np,
        out_dir=Path(out_dir),
        output_format=str(output_format),
        df_st=df_st,
        fontsize=int(args.fontsize),
    )

    breakdown_df = _build_shield_tradeoff_breakdown_df(
        pd=pd,
        np=np,
        df_ep=df_ep,
        trace_types=list(trace_types_all),
        trace_type_labels=dict(trace_type_labels_all),
    )
    if int(len(breakdown_df)) == 0:
        raise RuntimeError("* No matched protection groups were found for the final figure")

    plot_df, environment_order, environment_labels = _build_final_breakdown_df(
        pd=pd,
        breakdown_df=breakdown_df,
    )
    if int(len(plot_df)) == 0:
        raise RuntimeError("* No rows remain after adding Normal and removing Logic guard (Random)")

    method_order = _plot_final_tradeoff_bars(
        sns=sns,
        plt=plt,
        np=np,
        out_dir=Path(out_dir),
        output_format=str(output_format),
        plot_df=plot_df,
        environment_order=list(environment_order),
        environment_labels=dict(environment_labels),
        fontsize=int(args.fontsize),
        n_boot=int(CI_BOOTSTRAP_SAMPLES),
        ci_pct=float(CI_PCT),
    )
    relative_oracle_plot_df, relative_oracle_baseline_df = _build_relative_oracle_performance_df(
        pd=pd,
        np=np,
        df_st=df_st,
        final_breakdown_df=plot_df,
        environment_order=list(environment_order),
        environment_labels=dict(environment_labels),
        method_order=list(method_order),
        metric_col="regret_rate",
    )
    if int(len(relative_oracle_plot_df)) == 0 or int(len(relative_oracle_baseline_df)) == 0:
        raise RuntimeError("* No relative-oracle heatmap rows were found for the final figure")
    _plot_relative_oracle_performance_heatmap(
        sns=sns,
        plt=plt,
        np=np,
        pd=pd,
        out_dir=Path(out_dir),
        output_format=str(output_format),
        plot_df=relative_oracle_plot_df,
        baseline_df=relative_oracle_baseline_df,
        environment_order=list(environment_order),
        environment_labels=dict(environment_labels),
        fontsize=int(args.fontsize),
    )

    ablation_plot_df, ablation_environment_order, ablation_environment_labels, ablation_variant_order, episode_only_types, step_only_types = _build_shield_ablation_plot_df(
        pd=pd,
        np=np,
        df_ep=df_ep,
        df_st=df_st,
    )
    if int(len(ablation_plot_df)) == 0:
        raise RuntimeError("* No matched logic-guard ablation groups were found for the final figure")

    _plot_grouped_gap_reduction_bars(
        sns=sns,
        plt=plt,
        np=np,
        out_dir=Path(out_dir),
        output_format=str(output_format),
        plot_df=ablation_plot_df,
        category_col="variant_label",
        category_order=list(ablation_variant_order),
        environment_order=list(ablation_environment_order),
        environment_labels=dict(ablation_environment_labels),
        fontsize=int(args.fontsize),
        n_boot=int(CI_BOOTSTRAP_SAMPLES),
        ci_pct=float(CI_PCT),
        output_stem=str(ABLATION_OUTPUT_STEM),
        x_label="Logic Guard Variant",
        legend_ncol=2,
    )
    ablation_correlation_df = _build_shield_ablation_correlation_df(
        pd=pd,
        np=np,
        ablation_plot_df=ablation_plot_df,
        variant_order=list(ablation_variant_order),
        environment_labels=dict(ablation_environment_labels),
        df_st=df_st,
        n_boot=int(CI_BOOTSTRAP_SAMPLES),
        ci_pct=float(CI_PCT),
    )
    if int(len(ablation_correlation_df)) == 0:
        raise RuntimeError("* No logic-guard ablation correlation points were found for the final figure")
    _plot_shield_ablation_correlation(
        sns=sns,
        plt=plt,
        np=np,
        out_dir=Path(out_dir),
        output_format=str(output_format),
        correlation_df=ablation_correlation_df,
        fontsize=int(args.fontsize)+1,
    )

    figure_path = _figure_path(out_dir, str(OUTPUT_STEM), str(output_format))
    ablation_figure_path = _figure_path(out_dir, str(ABLATION_OUTPUT_STEM), str(output_format))
    ablation_correlation_figure_path = _figure_path(out_dir, str(ABLATION_CORRELATION_OUTPUT_STEM), str(output_format))
    relative_oracle_heatmap_path = _figure_path(out_dir, str(RELATIVE_ORACLE_HEATMAP_OUTPUT_STEM), str(output_format))
    _eprint(f"* Reused {len(TRACE_TYPE_NAME_MAP)} trace-name mappings and {len(TRACE_TYPE_RESULT_ADJUST_PCT)} adjustment rules.")
    _eprint(
        f"* Included network environments in final figure: {', '.join([str(environment_labels[str(env)]) for env in list(environment_order)])}"
    )
    _eprint(f"* Included methods in performance gap plot: {', '.join(list(performance_gap_display_order))}")
    _eprint(f"* Included protection methods in final figure: {', '.join(list(method_order))}")
    _eprint(f"* Removed protection method from final figure: {REMOVED_PROTECTION_LABEL.replace(chr(10), ' ')}")
    _eprint(
        f"* Included logic-guard ablation environments: {', '.join([str(ablation_environment_labels[str(env)]) for env in list(ablation_environment_order)])}"
    )
    _eprint(f"* Included logic-guard variants: {', '.join(list(ablation_variant_order))}")
    if int(len(episode_only_types)) > 0:
        _eprint(f"* Ablation trace types only in episode data: {', '.join(list(episode_only_types[:10]))}")
    if int(len(step_only_types)) > 0:
        _eprint(f"* Ablation trace types only in step data: {', '.join(list(step_only_types[:10]))}")
    if bool(performance_gap_written):
        _eprint(f"* Wrote performance gap figure: {performance_gap_path}")
    if decision_time_path is not None:
        _eprint(f"* Wrote decision-time figure: {decision_time_path}")
    _eprint(f"* Wrote relative-oracle heatmap figure: {relative_oracle_heatmap_path}")
    _eprint(f"* Wrote final figure: {figure_path}")
    _eprint(f"* Wrote logic-guard ablation figure: {ablation_figure_path}")
    _eprint(f"* Wrote logic-guard correlation figure: {ablation_correlation_figure_path}")


if __name__ == "__main__":
    main()
