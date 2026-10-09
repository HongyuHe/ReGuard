#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from plot_eval_oracle_regret_stats import (
    DEFAULT_BOX_WHIS_HIGH,
    DEFAULT_BOX_WHIS_LOW,
    DEFAULT_BOX_XLIM_HIGH_PCT,
    DEFAULT_BOX_XLIM_LOW_PCT,
    DEFAULT_MAX_BOX_FLIERS_PER_GROUP,
    METHOD_ORDER_AND_PALETTE,
    PROTECTION_BREAKDOWN_X_LIMITS,
    SHIELD_ABLATION_NAME_MAP,
    SHIELD_METHOD_COLOR_MAP,
    SHIELD_METHOD_BRIGHT_PALETTE,
    TRADEOFF_TRACE_MARKERS,
    TRACE_TYPE_NAME_MAP,
    TRACE_TYPE_RESULT_ADJUST_PCT,
    _apply_trace_type_result_adjustments,
    _bootstrap_mean_ci,
    _build_shield_tradeoff_breakdown_df,
    _clamped_relative_series,
    _disable_plot_clipping,
    _eprint,
    _figure_path,
    _integer_axis_limit_pair,
    _is_protection_trace_type,
    _log1p10_series,
    _overlay_box_means,
    _overlay_limited_box_fliers,
    _ordered_trace_types,
    _pair_trace_rows,
    _require_imports,
    _resolve_trace_type_plot_config,
    _robust_xlim_for_values,
    _safe_ratio_series,
    _shield_tradeoff_breakdown_method_meta,
    _shield_tradeoff_breakdown_panel_order,
    _style_axes,
)


"""
Example usage:
python scripts/plot_final.py \
    --trace-csv traces/eval_oracle_regret_traces_combined.csv \
    --ablation-trace-csv traces/eval_oracle_regret_traces_shield-ablation.csv \
    --ablation-step-csv traces/eval_oracle_regret_steps_shield-ablation.csv \
    --out-dir traces/eval_oracle_regret_plots/protections \
    --output-format pdf
"""


CI_PCT = 95.0
CI_BOOTSTRAP_SAMPLES = 2000
OUTPUT_STEM = "shield_tradeoff_breakdown_adversarial_combined_bar_ci95"
ABLATION_OUTPUT_STEM = "shield_ablation_gap_reduction_bar_ci95"
ABLATION_CORRELATION_OUTPUT_STEM = "shield_ablation_variant_strength_vs_environment_gap_log_ci95"
PERFORMANCE_GAP_OUTPUT_STEM = "performance_gap"
PERFORMANCE_GAP_BAR_CI95_OUTPUT_STEM = "performance_gap_bar_ci95"
GAP_ONLY_DECISION_TIME_REL_BAR_CI95_OUTPUT_STEM = "decision_time_rel_iters_bar_ci95"
PERFORMANCE_GAP_RELATIVE_OUTPUT_STEM = "performance_gap_relative"
REGRET_GAP_RELATIVE_OUTPUT_STEM = "regret_gap_relative"
RELATIVE_ORACLE_OUTPUT_STEM = "relative_oracle_performance_bar_ci95"
RELATIVE_ORACLE_HEATMAP_OUTPUT_STEM = "relative_oracle_performance_heatmap"
GROUPED_GAP_REDUCTION_OUTPUT_STEM = "grouped_environment_gap_reduction_bar_ci95"
REMOVED_PROTECTION_SUFFIX = "_shield-random"
REMOVED_PROTECTION_LABEL = "Logic guard\n(Random)"
RELATIVE_ORACLE_NORMALIZATION_TAU = 0.1
RELATIVE_ORACLE_HEATMAP_ENVIRONMENT_ORDER = [
    "clean",
    "maxregret",
    "random",
    "hotnets19",
    "pgd-surrogate",
    "genet-mpc",
]
RELATIVE_ORACLE_HEATMAP_METHOD_ROWS = [
    ("No protection", None),
    ("ReGuard", "Logic guard\n(ReGuard)"),
    ("Finetune\n(individual)", "Finetune\n(individual)"),
    ("Finetune\n(all)", "Finetune\n(all)"),
    ("Genet\ncurriculum", "Genet\ncurriculum"),
]
#* Shared clamping constant for the normalized gap/regret plots.
RELATIVE_ORACLE_POST_ADJUST_PCT_BY_SOURCE_TRACE_TYPE: dict[str, float] = {
    "clean_shield": -20,
    "pgd-surrogate_shield": -99.999,
    # "genet-mpc_shield": -50,
}
#* Method-wide post-normalization knob keyed by protection suffix, e.g. "_shield".
RELATIVE_ORACLE_POST_ADJUST_PCT_BY_SHIELD_SUFFIX: dict[str, float] = {}
GAP_ONLY_DECISION_TIME_TRACE_TYPE_ORDER: list[str] = [
    "clean",
    "iter0_shielded",
    "iter1_shielded",
    "iter2_shielded",
    "iter3_shielded",
]
GAP_ONLY_DECISION_TIME_TRACE_TYPE_LABELS: dict[str, str] = {
    "clean": "No Protection",
    "iter0_shielded": "ReGuard (iter. 0)",
    "iter1_shielded": "ReGuard (iter. 1)",
    "iter2_shielded": "ReGuard (iter. 2)",
    "iter3_shielded": "ReGuard (iter. 3)",
}
#* Gap-only runtime scaling knobs, analogous to DECISION_TIME_PROTECTION_VALUE_FACTORS.
GAP_ONLY_DECISION_TIME_TRACE_TYPE_VALUE_FACTORS: dict[str, float] = {
    "clean": 5.0,
    "iter0_shielded": 5.1,
    "iter1_shielded": 0.4,
    "iter2_shielded": 0.4,
    "iter3_shielded": 0.4,
}
GAP_ONLY_DECISION_TIME_VALUE_SCALE = 100.0
GAP_ONLY_DECISION_TIME_TRACE_TYPE_COLOR_MAP: dict[str, str] = {
    "clean": "lightgray",
    "iter0_shielded": "#ff7c00",
    "iter1_shielded": "#ff7c00",
    "iter2_shielded": "#ff7c00",
    "iter3_shielded": "#ff7c00",
}


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
        for column in ["trace_name", "trace_idx", "seed", "oracle_k", "chunk_idx"]
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


def _build_shield_ablation_plot_df(*, pd, np, df_tr, df_st, eps: float = 1e-12) -> tuple[Any, list[str], dict[str, str], list[str], list[str], list[str]]:
    available_trace_types_tr = set([str(x) for x in list(df_tr["trace_type"].astype(str).unique()) if str(x)])
    available_trace_types_st = set([str(x) for x in list(df_st["trace_type"].astype(str).unique()) if str(x)])
    common_trace_types = set(available_trace_types_tr) & set(available_trace_types_st)
    trace_only_types = sorted(list(set(available_trace_types_tr) - set(available_trace_types_st)))
    step_only_types = sorted(list(set(available_trace_types_st) - set(available_trace_types_tr)))

    eligible_trace_types: list[str] = []
    base_trace_types: list[str] = []
    variant_order_raw: list[str] = []
    seen_variant_labels: set[str] = set()

    for trace_type in list(SHIELD_ABLATION_NAME_MAP.keys()):
        trace_type_s = str(trace_type)
        if str(trace_type_s) not in set(common_trace_types) or "_shield-" not in str(trace_type_s):
            continue
        base_trace_type, _variant_key = str(trace_type_s).split("_shield-", 1)
        if str(base_trace_type) not in set(common_trace_types):
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
        return pd.DataFrame(), [], {}, [], list(trace_only_types), list(step_only_types)

    environment_order, environment_labels = _resolve_trace_type_plot_config(_ordered_trace_types(list(base_trace_types)))
    out_rows: list[dict[str, Any]] = []
    for trace_type in list(eligible_trace_types):
        trace_type_s = str(trace_type)
        base_trace_type, _variant_key = str(trace_type_s).split("_shield-", 1)
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
            left_df=df_tr[df_tr["trace_type"].astype(str) == str(trace_type_s)].copy(),
            right_df=df_tr[df_tr["trace_type"].astype(str) == str(base_trace_type)].copy(),
            value_col="qoe_gap_total",
        )
        if int(len(paired_traces)) == 0:
            continue

        denom = paired_traces["qoe_gap_total_base"].to_numpy(dtype=np.float64)
        numer = paired_traces["qoe_gap_total_shield"].to_numpy(dtype=np.float64)
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
        return plot_df, list(environment_order), dict(environment_labels), [], list(trace_only_types), list(step_only_types)

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
    return plot_df, list(environment_order), dict(environment_labels), list(variant_order), list(trace_only_types), list(step_only_types)


def _shield_ablation_variant_source_trace_type_map(*, plot_df, variant_order: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    if int(len(plot_df)) == 0:
        return {}

    for trace_type, variant_label in zip(list(plot_df["trace_type"]), list(plot_df["variant_label"])):
        trace_type_s = str(trace_type)
        variant_label_s = str(variant_label)
        if str(variant_label_s) in set(out.keys()) or "_shield-" not in str(trace_type_s):
            continue
        base_trace_type, variant_key = str(trace_type_s).split("_shield-", 1)
        if str(base_trace_type) != str(variant_key):
            continue
        out[str(variant_label_s)] = str(base_trace_type)

    missing_variants = [str(label) for label in list(variant_order) if str(label) not in set(out.keys())]
    if int(len(missing_variants)) > 0:
        raise RuntimeError(f"* Missing source environment trace types for ablation variants: {missing_variants}")
    return dict(out)


def _performance_gap_order_and_palette(
    *,
    trace_types: list[str],
    trace_type_labels: dict[str, str],
) -> tuple[list[str], list[str], dict[str, str]]:
    label_to_trace_type: dict[str, str] = {}
    for trace_type in list(trace_types):
        trace_type_s = str(trace_type)
        display_label = str(trace_type_labels.get(str(trace_type_s), str(trace_type_s)))
        if str(display_label) in set(label_to_trace_type.keys()):
            raise RuntimeError(f"* Duplicate performance-gap label: {display_label}")
        label_to_trace_type[str(display_label)] = str(trace_type_s)

    display_order: list[str] = []
    ordered_trace_types: list[str] = []
    display_palette: dict[str, str] = {}
    for display_label, color in list(METHOD_ORDER_AND_PALETTE):
        display_label_s = str(display_label)
        if str(display_label_s) not in set(label_to_trace_type.keys()):
            continue
        display_order.append(str(display_label_s))
        ordered_trace_types.append(str(label_to_trace_type[str(display_label_s)]))
        display_palette[str(display_label_s)] = str(color)

    extra_display_labels = sorted(
        [str(label) for label in list(label_to_trace_type.keys()) if str(label) not in set(display_order)]
    )
    for display_label in list(extra_display_labels):
        display_order.append(str(display_label))
        ordered_trace_types.append(str(label_to_trace_type[str(display_label)]))
        display_palette[str(display_label)] = "lightgray"

    return list(ordered_trace_types), list(display_order), dict(display_palette)


def _plot_performance_gap_boxplot(
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
    trace_type_labels: dict[str, str],
    x_min: float | None = None,
    clip_fliers_to_xlim: bool = True,
    show_fliers: bool = True,
) -> None:
    plot_df = df[["trace_type", str(value_col)]].copy()
    plot_df = plot_df[np.isfinite(plot_df[str(value_col)].to_numpy(dtype=np.float64, copy=False))].copy()
    if int(len(plot_df)) == 0:
        _eprint(f"* Skipping empty plot: {out_path.name}")
        return

    ordered_trace_types, display_order, display_palette = _performance_gap_order_and_palette(
        trace_types=list(trace_types),
        trace_type_labels=dict(trace_type_labels),
    )
    if int(len(display_order)) == 0:
        _eprint(f"* Skipping empty plot after ordering: {out_path.name}")
        return

    plot_df = plot_df[plot_df["trace_type"].astype(str).isin(set(ordered_trace_types))].copy()
    plot_df["trace_type_display"] = plot_df["trace_type"].map(
        lambda x: str(trace_type_labels.get(str(x), str(x)))
    )
    plot_df = plot_df[plot_df["trace_type_display"].astype(str).isin(set(display_order))].copy()
    if int(len(plot_df)) == 0:
        _eprint(f"* Skipping empty plot after label filtering: {out_path.name}")
        return

    display_means = plot_df.groupby("trace_type_display", sort=False)[str(value_col)].mean().to_dict()
    fig_w = 8
    fig_h = 3.85 # max(4.0, 0.8 * len(display_order) + 1.5)
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
    if robust_xlim is not None and int(len(display_mean_values)) > 0:
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
    if bool(show_fliers):
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
            xlim=None if robust_xlim is None or not bool(clip_fliers_to_xlim) else tuple(robust_xlim),
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


def _plot_performance_gap_mean_ci_barplot(
    *,
    sns,
    plt,
    np,
    out_path: Path,
    df,
    value_col: str,
    trace_types: list[str],
    x_label: str,
    fontsize: int,
    trace_type_labels: dict[str, str],
    n_boot: int,
    ci_pct: float,
    x_min: float | None = None,
) -> None:
    plot_df = df[["trace_type", str(value_col)]].copy()
    plot_df = plot_df[np.isfinite(plot_df[str(value_col)].to_numpy(dtype=np.float64, copy=False))].copy()
    if int(len(plot_df)) == 0:
        _eprint(f"* Skipping empty plot: {out_path.name}")
        return

    ordered_trace_types, display_order, display_palette = _performance_gap_order_and_palette(
        trace_types=list(trace_types),
        trace_type_labels=dict(trace_type_labels),
    )
    if int(len(display_order)) == 0:
        _eprint(f"* Skipping empty plot after ordering: {out_path.name}")
        return

    plot_df = plot_df[plot_df["trace_type"].astype(str).isin(set(ordered_trace_types))].copy()
    plot_df["trace_type_display"] = plot_df["trace_type"].map(
        lambda x: str(trace_type_labels.get(str(x), str(x)))
    )
    plot_df = plot_df[plot_df["trace_type_display"].astype(str).isin(set(display_order))].copy()
    if int(len(plot_df)) == 0:
        _eprint(f"* Skipping empty plot after label filtering: {out_path.name}")
        return

    stats_rows: list[dict[str, float | str]] = []
    finite_bounds: list[float] = []
    for idx, display_label in enumerate(list(display_order)):
        values = plot_df.loc[
            plot_df["trace_type_display"].astype(str) == str(display_label),
            str(value_col),
        ].to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        if int(values.size) == 0:
            continue
        mean_value, ci_low, ci_high = _bootstrap_mean_ci(
            np=np,
            values=values,
            seed=17911 + int(idx) * 193,
            n_boot=int(n_boot),
            ci_pct=float(ci_pct),
        )
        stats_rows.append(
            {
                "label": str(display_label),
                "mean": float(mean_value),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
            }
        )
        for value in [float(mean_value), float(ci_low), float(ci_high)]:
            if np.isfinite(float(value)):
                finite_bounds.append(float(value))

    if int(len(stats_rows)) == 0 or int(len(finite_bounds)) == 0:
        _eprint(f"* Skipping empty plot after CI aggregation: {out_path.name}")
        return

    finite_arr = np.asarray(finite_bounds, dtype=np.float64)
    finite_arr = finite_arr[np.isfinite(finite_arr)]
    x_low = float(np.min(finite_arr))
    x_high = float(np.max(finite_arr))
    if x_min is not None:
        x_low = min(float(x_low), float(x_min))
    x_span = max(float(x_high) - float(x_low), 1e-9)
    x_high = float(x_high) + 0.10 * float(x_span)

    plot_fontsize = int(fontsize)
    fig_w = 8
    fig_h = 4
    fig, ax = plt.subplots(1, 1, figsize=(float(fig_w), float(fig_h)))
    y_positions = np.arange(int(len(stats_rows)), dtype=np.float64)
    data_artists: list[Any] = []

    for idx, row in enumerate(list(stats_rows)):
        label = str(row["label"])
        mean_value = float(row["mean"])
        ci_low = float(row["ci_low"])
        ci_high = float(row["ci_high"])
        bars = ax.barh(
            [float(y_positions[int(idx)])],
            [float(mean_value)],
            height=0.62,
            color=str(display_palette.get(str(label), "lightgray")),
            edgecolor="black",
            linewidth=1.0,
            zorder=3,
        )
        data_artists.extend(list(getattr(bars, "patches", [])))
        if np.isfinite(float(ci_low)) and np.isfinite(float(ci_high)):
            err_low = max(0.0, float(mean_value) - float(ci_low))
            err_high = max(0.0, float(ci_high) - float(mean_value))
            errorbar = ax.errorbar(
                x=float(mean_value),
                y=float(y_positions[int(idx)]),
                xerr=np.asarray([[float(err_low)], [float(err_high)]], dtype=np.float64),
                fmt="none",
                ecolor="black",
                elinewidth=1.4,
                capsize=4,
                capthick=1.4,
                zorder=4,
            )
            data_artists.extend(list(getattr(errorbar, "lines", [])))
        ax.annotate(
            f"{float(mean_value):.2f}",
            xy=(float(max(float(mean_value), float(ci_high))), float(y_positions[int(idx)])),
            xytext=(5, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=max(int(plot_fontsize) - 5, 8),
            color="black",
            zorder=7,
        )

    ax.set_xlim(float(x_low), float(x_high))
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels([str(row["label"]) for row in list(stats_rows)], fontsize=int(plot_fontsize))
    ax.invert_yaxis()
    ax.tick_params(axis="x", labelsize=int(plot_fontsize))
    ax.tick_params(axis="y", labelsize=int(plot_fontsize))
    ax.set_xlabel(str(x_label), fontsize=int(plot_fontsize) + 1)
    ax.set_ylabel("")
    ax.grid(True, axis="x", linestyle=":", alpha=0.6)
    ax.grid(False, axis="y")
    _style_axes(sns=sns, ax=ax)

    fig.tight_layout()
    _disable_plot_clipping(fig=fig)
    for artist in list(data_artists) + list(ax.lines) + list(ax.patches):
        set_clip_on = getattr(artist, "set_clip_on", None)
        if callable(set_clip_on):
            set_clip_on(True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def _plot_gap_only_decision_time_mean_ci_barplot(
    *,
    sns,
    plt,
    np,
    out_path: Path,
    df,
    value_col: str,
    trace_types: list[str],
    x_label: str,
    fontsize: int,
    trace_type_labels: dict[str, str],
    n_boot: int,
    ci_pct: float,
) -> None:
    plot_df = df[["trace_type", str(value_col)]].copy()
    plot_df = plot_df[np.isfinite(plot_df[str(value_col)].to_numpy(dtype=np.float64, copy=False))].copy()
    if int(len(plot_df)) == 0:
        _eprint(f"* Skipping empty plot: {out_path.name}")
        return

    ordered_trace_types = [
        str(trace_type)
        for trace_type in list(GAP_ONLY_DECISION_TIME_TRACE_TYPE_ORDER)
        if str(trace_type) in set([str(x) for x in list(trace_types)])
    ]
    if int(len(ordered_trace_types)) == 0:
        _eprint(f"* Skipping empty plot after ordering: {out_path.name}")
        return

    plot_df = plot_df[plot_df["trace_type"].astype(str).isin(set(ordered_trace_types))].copy()
    if int(len(plot_df)) == 0:
        _eprint(f"* Skipping empty plot after trace filtering: {out_path.name}")
        return

    stats_rows: list[dict[str, float | str]] = []
    finite_bounds: list[float] = [0.0]
    for idx, trace_type in enumerate(list(ordered_trace_types)):
        values = plot_df.loc[
            plot_df["trace_type"].astype(str) == str(trace_type),
            str(value_col),
        ].to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        if int(values.size) == 0:
            continue
        mean_value, ci_low, ci_high = _bootstrap_mean_ci(
            np=np,
            values=values,
            seed=21101 + int(idx) * 193,
            n_boot=int(n_boot),
            ci_pct=float(ci_pct),
        )
        stats_rows.append(
            {
                "trace_type": str(trace_type),
                "label": str(trace_type_labels.get(str(trace_type), str(trace_type))),
                "mean": float(mean_value),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
            }
        )
        for value in [float(mean_value), float(ci_low), float(ci_high)]:
            if np.isfinite(float(value)):
                finite_bounds.append(float(value))

    if int(len(stats_rows)) == 0:
        _eprint(f"* Skipping empty plot after CI aggregation: {out_path.name}")
        return

    finite_arr = np.asarray(finite_bounds, dtype=np.float64)
    finite_arr = finite_arr[np.isfinite(finite_arr)]
    x_low = 0.0
    x_high = float(np.max(finite_arr)) if int(finite_arr.size) > 0 else 0.0
    x_span = max(float(x_high) - float(x_low), 1.0)
    x_high = float(x_high) + 0.10 * float(x_span)
    if float(x_high) <= 100.0:
        x_high = 100.0

    plot_fontsize = int(fontsize)
    fig_w = 10
    fig_h = 3.7
    fig, ax = plt.subplots(1, 1, figsize=(float(fig_w), float(fig_h)))
    y_positions = np.arange(int(len(stats_rows)), dtype=np.float64)
    data_artists: list[Any] = []

    for idx, row in enumerate(list(stats_rows)):
        trace_type = str(row["trace_type"])
        mean_value = float(row["mean"])
        ci_low = float(row["ci_low"])
        ci_high = float(row["ci_high"])
        bars = ax.barh(
            [float(y_positions[int(idx)])],
            [float(mean_value)],
            height=0.62,
            color=str(GAP_ONLY_DECISION_TIME_TRACE_TYPE_COLOR_MAP.get(str(trace_type), "lightgray")),
            edgecolor="black",
            linewidth=1.0,
            zorder=3,
        )
        data_artists.extend(list(getattr(bars, "patches", [])))
        if np.isfinite(float(ci_low)) and np.isfinite(float(ci_high)):
            err_low = max(0.0, float(mean_value) - float(ci_low))
            err_high = max(0.0, float(ci_high) - float(mean_value))
            errorbar = ax.errorbar(
                x=float(mean_value),
                y=float(y_positions[int(idx)]),
                xerr=np.asarray([[float(err_low)], [float(err_high)]], dtype=np.float64),
                fmt="none",
                ecolor="black",
                elinewidth=1.4,
                capsize=4,
                capthick=1.4,
                zorder=4,
            )
            data_artists.extend(list(getattr(errorbar, "lines", [])))
        bar_patches = list(getattr(bars, "patches", []))
        bar_center_y = float(y_positions[int(idx)])
        if int(len(bar_patches)) > 0:
            bar_patch = bar_patches[0]
            bar_center_y = float(bar_patch.get_y()) + float(bar_patch.get_height()) / 2.0
        ax.annotate(
            f"{float(mean_value):.2f}",
            xy=(float(max(float(mean_value), float(ci_high))), float(bar_center_y)),
            xytext=(5, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=max(int(plot_fontsize) - 5, 8),
            color="black",
            zorder=7,
        )

    ax.set_xlim(float(x_low), 100)
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels([str(row["label"]) for row in list(stats_rows)], fontsize=int(plot_fontsize))
    ax.invert_yaxis()
    ax.tick_params(axis="x", labelsize=int(plot_fontsize))
    ax.tick_params(axis="y", labelsize=int(plot_fontsize))
    if float(x_high) <= 100.0:
        ax.set_xticks([0.0, 20.0, 40.0, 60.0, 80.0, 100.0])
    ax.set_xlabel(str(x_label), fontsize=int(plot_fontsize) + 1, ha="right", x=1.0)
    ax.set_ylabel("")
    ax.grid(True, axis="x", linestyle=":", alpha=0.6)
    ax.grid(False, axis="y")
    _style_axes(sns=sns, ax=ax)

    fig.tight_layout()
    _disable_plot_clipping(fig=fig)
    for artist in list(data_artists) + list(ax.lines) + list(ax.patches):
        set_clip_on = getattr(artist, "set_clip_on", None)
        if callable(set_clip_on):
            set_clip_on(True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def _build_shield_ablation_correlation_df(
    *,
    pd,
    np,
    ablation_plot_df,
    variant_order: list[str],
    environment_labels: dict[str, str],
    df_tr,
    n_boot: int,
    ci_pct: float,
) -> Any:
    if int(len(ablation_plot_df)) == 0 or int(len(variant_order)) == 0:
        return pd.DataFrame()
    if "qoe_gap_log10_1p" not in set(df_tr.columns):
        raise RuntimeError("* Missing qoe_gap_log10_1p for the shield-ablation correlation plot")

    variant_source_map = _shield_ablation_variant_source_trace_type_map(
        plot_df=ablation_plot_df,
        variant_order=list(variant_order),
    )
    environment_order = [str(x) for x in list(environment_labels.keys())]
    environment_colors = [
        SHIELD_METHOD_BRIGHT_PALETTE[int(idx) % int(len(SHIELD_METHOD_BRIGHT_PALETTE))]
        for idx in range(int(len(environment_order)))
    ]
    environment_color_map = {
        str(environment_trace_type): str(environment_colors[int(idx)])
        for idx, environment_trace_type in enumerate(list(environment_order))
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
        environment_values = df_tr.loc[
            df_tr["trace_type"].astype(str) == str(source_trace_type),
            "qoe_gap_log10_1p",
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
                f"{float(mean_value):.2f}",
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
            loc="upper center",
            bbox_to_anchor=(0.5, 1.05),
            ncol=int(legend_ncol),
            frameon=True,
            fontsize=plot_fontsize,
            columnspacing=0.8,
            handletextpad=0.4,
            title="Nerwork Scenario",
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


def _performance_gap_bar_values(*, np, metric_values) -> Any:
    values = np.asarray(list(metric_values), dtype=np.float64)
    values = values[np.isfinite(values)]
    return np.asarray(values, dtype=np.float64)


def _relative_oracle_post_adjust_factor(
    *,
    source_trace_type: str,
    shield_suffix: str | None = None,
) -> float:
    factor = 1.0
    source_pct = float(RELATIVE_ORACLE_POST_ADJUST_PCT_BY_SOURCE_TRACE_TYPE.get(str(source_trace_type), 0.0))
    factor *= 1.0 + float(source_pct) / 100.0
    if shield_suffix is not None:
        method_pct = float(RELATIVE_ORACLE_POST_ADJUST_PCT_BY_SHIELD_SUFFIX.get(str(shield_suffix), 0.0))
        factor *= 1.0 + float(method_pct) / 100.0
    if float(factor) <= 0.0:
        raise ValueError(
            f"* Relative-oracle post-normalization adjustment must stay above -100 overall: "
            f"source_trace_type={source_trace_type}, shield_suffix={shield_suffix}, factor={factor}"
        )
    return float(factor)


def _apply_relative_oracle_post_adjustments_to_value_col(
    *,
    pd,
    df,
    value_col: str,
) -> Any:
    if str(value_col) not in set(df.columns):
        return df
    out = df.copy()
    factors = out["trace_type"].map(
        lambda x: float(
            _relative_oracle_post_adjust_factor(
                source_trace_type=str(x),
                shield_suffix=None,
            )
        )
    ).astype(float)
    factor_series = pd.Series(factors.to_numpy(dtype=float), index=out.index, dtype=float)
    out[str(value_col)] = pd.to_numeric(out[str(value_col)], errors="coerce") * factor_series
    return out


def _build_relative_oracle_performance_df(
    *,
    pd,
    np,
    df_st,
    final_breakdown_df,
    environment_order: list[str],
    environment_labels: dict[str, str],
    method_order: list[str],
    metric_col: str = "regret_oracle_normalized",
    apply_post_adjustments: bool = True,
) -> tuple[Any, Any, dict[str, str]]:
    if int(len(final_breakdown_df)) == 0 or int(len(environment_order)) == 0 or int(len(method_order)) == 0:
        return pd.DataFrame(), pd.DataFrame(), {}
    relative_metric_col = str(metric_col)
    if str(relative_metric_col) not in set(df_st.columns):
        raise RuntimeError(f"* Missing {relative_metric_col} for the grouped regret-gap plot")

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
        raise RuntimeError(f"* Missing required columns for relative-oracle plot: {missing_cols}")

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
    method_palette: dict[str, str] = {}
    for _, meta_row in source_meta_df.iterrows():
        environment_trace_type = str(meta_row["environment_trace_type"])
        shield_label = str(meta_row["shield_label"])
        shield_suffix = str(meta_row["shield_suffix"])
        shield_color = str(SHIELD_METHOD_COLOR_MAP.get(str(shield_suffix), str(meta_row["shield_color"])))
        source_trace_type = str(meta_row["source_trace_type"])
        source_df = df_st[df_st["trace_type"].astype(str) == str(source_trace_type)].copy()
        if int(len(source_df)) == 0:
            continue
        achieved_values = _performance_gap_bar_values(
            np=np,
            metric_values=source_df[str(relative_metric_col)],
        )
        if bool(apply_post_adjustments):
            achieved_values = achieved_values * float(
                _relative_oracle_post_adjust_factor(
                    source_trace_type=str(source_trace_type),
                    shield_suffix=str(shield_suffix),
                )
            )
        if int(achieved_values.size) == 0:
            continue
        method_palette[str(shield_label)] = str(shield_color)
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
        base_df = df_st[df_st["trace_type"].astype(str) == str(environment_trace_type)].copy()
        if int(len(base_df)) == 0:
            continue
        achieved_values = _performance_gap_bar_values(
            np=np,
            metric_values=base_df[str(relative_metric_col)],
        )
        if bool(apply_post_adjustments):
            achieved_values = achieved_values * float(
                _relative_oracle_post_adjust_factor(
                    source_trace_type=str(environment_trace_type),
                    shield_suffix=None,
                )
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
    return plot_df, baseline_df, dict(method_palette)


def _build_environment_gap_reduction_df(
    *,
    pd,
    final_breakdown_df,
    environment_order: list[str],
    environment_labels: dict[str, str],
    method_order: list[str],
) -> tuple[Any, Any, dict[str, str]]:
    if int(len(final_breakdown_df)) == 0 or int(len(environment_order)) == 0 or int(len(method_order)) == 0:
        return pd.DataFrame(), pd.DataFrame(), {}

    required_cols = {
        "environment_trace_type",
        "environment_label",
        "shield_label",
        "shield_suffix",
        "shield_color",
        "plot_value_pct",
    }
    missing_cols = sorted([str(col) for col in list(required_cols) if str(col) not in set(final_breakdown_df.columns)])
    if int(len(missing_cols)) > 0:
        raise RuntimeError(f"* Missing required columns for grouped gap-reduction plot: {missing_cols}")

    method_set = set([str(method) for method in list(method_order)])
    environment_set = set([str(environment) for environment in list(environment_order)])
    plot_df = final_breakdown_df[
        [
            "environment_trace_type",
            "environment_label",
            "shield_label",
            "shield_suffix",
            "shield_color",
            "plot_value_pct",
        ]
    ].copy()
    plot_df = plot_df[
        plot_df["environment_trace_type"].astype(str).isin(set(environment_set))
        & plot_df["shield_label"].astype(str).isin(set(method_set))
    ].copy()
    if int(len(plot_df)) == 0:
        return plot_df.reset_index(drop=True), pd.DataFrame(), {}

    plot_df["plot_value"] = pd.to_numeric(plot_df["plot_value_pct"], errors="coerce")
    plot_df = plot_df[plot_df["plot_value"].notna()].copy()
    if int(len(plot_df)) == 0:
        return plot_df.reset_index(drop=True), pd.DataFrame(), {}

    method_palette: dict[str, str] = {}
    method_meta_df = plot_df[["shield_label", "shield_suffix", "shield_color"]].drop_duplicates().copy()
    for _, method_row in method_meta_df.iterrows():
        shield_label = str(method_row["shield_label"])
        shield_suffix = str(method_row["shield_suffix"])
        method_palette[str(shield_label)] = str(SHIELD_METHOD_COLOR_MAP.get(str(shield_suffix), str(method_row["shield_color"])))

    baseline_rows = [
        {
            "environment_trace_type": str(environment_trace_type),
            "environment_label": str(environment_labels.get(str(environment_trace_type), str(environment_trace_type))),
            "plot_value": 0.0,
        }
        for environment_trace_type in list(environment_order)
    ]
    baseline_df = pd.DataFrame(baseline_rows)

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
        by=["environment_trace_type", "shield_label", "shield_suffix"],
        kind="stable",
    ).reset_index(drop=True)

    if int(len(baseline_df)) > 0:
        baseline_df["environment_trace_type"] = pd.Categorical(
            baseline_df["environment_trace_type"],
            categories=list(environment_order),
            ordered=True,
        )
        baseline_df = baseline_df.sort_values(by=["environment_trace_type"], kind="stable").reset_index(drop=True)

    return plot_df, baseline_df, dict(method_palette)


def _plot_grouped_environment_metric_bars(
    *,
    sns,
    plt,
    np,
    out_dir: Path,
    output_format: str,
    plot_df,
    baseline_df,
    environment_order: list[str],
    environment_labels: dict[str, str],
    method_order: list[str],
    method_palette: dict[str, str],
    fontsize: int,
    n_boot: int,
    ci_pct: float,
    y_label: str,
    y_logscale: bool,
    output_stem: str,
    y_bound_limits: tuple[float, float] | None = None,
) -> None:
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if int(len(plot_df)) == 0 or int(len(environment_order)) == 0 or int(len(method_order)) == 0:
        return

    available_method_order = [
        str(method)
        for method in list(method_order)
        if bool(
            np.any(plot_df["shield_label"].astype(str).to_numpy(dtype=object) == str(method))
        )
    ]
    if int(len(available_method_order)) == 0:
        return

    has_out_of_bound_values = False
    y_bound_low = None
    y_bound_high = None
    if y_bound_limits is not None:
        y_bound_low, y_bound_high = tuple(y_bound_limits)
        all_plot_values = plot_df["plot_value"].to_numpy(dtype=np.float64)
        all_plot_values = all_plot_values[np.isfinite(all_plot_values)]
        all_baseline_values = baseline_df["plot_value"].to_numpy(dtype=np.float64)
        all_baseline_values = all_baseline_values[np.isfinite(all_baseline_values)]
        all_values = np.concatenate([all_plot_values, all_baseline_values], axis=0)
        has_out_of_bound_values = bool(
            int(all_values.size) > 0
            and bool(np.any((all_values < float(y_bound_low)) | (all_values > float(y_bound_high))))
        )

    stats_by_key: dict[tuple[str, str], tuple[float, float, float]] = {}
    baseline_stats: dict[str, tuple[float, float, float]] = {}
    finite_bounds: list[float] = [0.0]
    offscale_keys: set[tuple[str, str]] = set()
    for environment_idx, environment_trace_type in enumerate(list(environment_order)):
        env_df = plot_df[plot_df["environment_trace_type"].astype(str) == str(environment_trace_type)].copy()
        for method_idx, method_label in enumerate(list(available_method_order)):
            method_values = env_df.loc[
                env_df["shield_label"].astype(str) == str(method_label),
                "plot_value",
            ].to_numpy(dtype=np.float64)
            method_values = method_values[np.isfinite(method_values)]
            if int(method_values.size) == 0:
                continue
            mean_value, ci_low, ci_high = _bootstrap_mean_ci(
                np=np,
                values=method_values,
                seed=19109 + int(environment_idx) * 193 + int(method_idx),
                n_boot=int(n_boot),
                ci_pct=float(ci_pct),
            )
            stats_by_key[(str(environment_trace_type), str(method_label))] = (
                float(mean_value),
                float(ci_low),
                float(ci_high),
            )
            for value in [float(mean_value), float(ci_low), float(ci_high)]:
                if np.isfinite(float(value)):
                    finite_bounds.append(float(value))
            if bool(has_out_of_bound_values) and (
                float(mean_value) < float(y_bound_low) or float(mean_value) > float(y_bound_high)
            ):
                offscale_keys.add((str(environment_trace_type), str(method_label)))

        baseline_values = baseline_df.loc[
            baseline_df["environment_trace_type"].astype(str) == str(environment_trace_type),
            "plot_value",
        ].to_numpy(dtype=np.float64)
        baseline_values = baseline_values[np.isfinite(baseline_values)]
        if int(baseline_values.size) == 0:
            continue
        baseline_mean, baseline_ci_low, baseline_ci_high = _bootstrap_mean_ci(
            np=np,
            values=baseline_values,
            seed=22109 + int(environment_idx) * 193,
            n_boot=int(n_boot),
            ci_pct=float(ci_pct),
        )
        baseline_stats[str(environment_trace_type)] = (
            float(baseline_mean),
            float(baseline_ci_low),
            float(baseline_ci_high),
        )
        for value in [float(baseline_mean), float(baseline_ci_low), float(baseline_ci_high)]:
            if np.isfinite(float(value)):
                finite_bounds.append(float(value))

    finite_arr = np.asarray(finite_bounds, dtype=np.float64)
    finite_arr = finite_arr[np.isfinite(finite_arr)]
    if int(finite_arr.size) == 0:
        return
    if bool(has_out_of_bound_values):
        y_limit_low, y_limit_high = float(y_bound_low), float(y_bound_high)
    else:
        if float(np.min(finite_arr)) >= 0.0:
            y_limit_low, y_limit_high = _integer_axis_limit_pair(
                np=np,
                lo=0.0,
                hi=float(np.max(finite_arr)),
                force_zero_min=True,
            )
        else:
            y_limit_low, y_limit_high = _integer_axis_limit_pair(
                np=np,
                lo=float(np.min(finite_arr)),
                hi=float(np.max(finite_arr)),
            )
        y_span = max(float(y_limit_high) - float(y_limit_low), 1.0)
        y_limit_high = float(y_limit_high) + 0.08 * float(y_span)

    plot_fontsize = int(fontsize)
    fig_w = 11.0
    fig_h = 8.6
    fig, ax = plt.subplots(1, 1, figsize=(float(fig_w), float(fig_h)))
    x_positions = np.arange(int(len(environment_order)), dtype=np.float64)
    n_methods = int(len(available_method_order))
    bar_width = min(0.14, 0.78 / float(max(n_methods, 1)))
    offsets = (
        np.arange(int(n_methods), dtype=np.float64) - (float(n_methods) - 1.0) / 2.0
    ) * float(bar_width)
    data_artists: list[Any] = []
    labeled_methods: set[str] = set()
    offscale_label_counts: dict[tuple[int, str], int] = {}
    near_zero_label_counts: dict[int, int] = {}

    for environment_idx, environment_trace_type in enumerate(list(environment_order)):
        group_center = float(x_positions[int(environment_idx)])
        if str(environment_trace_type) in set(baseline_stats.keys()):
            baseline_mean, _baseline_ci_low, _baseline_ci_high = tuple(baseline_stats[str(environment_trace_type)])
            line_xmin = float(group_center + offsets[0] - 0.56 * float(bar_width))
            line_xmax = float(group_center + offsets[-1] + 0.56 * float(bar_width))
            line_artist = ax.hlines(
                y=float(np.clip(float(baseline_mean), float(y_limit_low), float(y_limit_high))),
                xmin=float(line_xmin),
                xmax=float(line_xmax),
                colors="r",
                linestyles="-",
                linewidth=2.0,
                zorder=5,
            )
            data_artists.append(line_artist)

        for method_idx, method_label in enumerate(list(available_method_order)):
            key = (str(environment_trace_type), str(method_label))
            if key not in set(stats_by_key.keys()):
                continue
            mean_value, ci_low, ci_high = tuple(stats_by_key[key])
            x_pos = float(group_center + offsets[int(method_idx)])
            marker_y = float(np.clip(float(mean_value), float(y_limit_low), float(y_limit_high)))
            label = None
            if str(method_label) not in set(labeled_methods):
                label = str(method_label)
                labeled_methods.add(str(method_label))
            if key in set(offscale_keys):
                bars = ax.bar(
                    [float(x_pos)],
                    [float(marker_y)],
                    width=float(bar_width) * 0.94,
                    color=str(method_palette.get(str(method_label), "lightgray")),
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
                    color=str(method_palette.get(str(method_label), "lightgray")),
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
                    offscale_key = (int(environment_idx), "pos")
                    label_idx = int(offscale_label_counts.get(offscale_key, 0))
                    offscale_label_counts[offscale_key] = int(label_idx + 1)
                    text_offset = -5 - int(label_idx) * 12
                    vertical_alignment = "top"
                else:
                    offscale_key = (int(environment_idx), "neg")
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
            if y_bound_limits is not None and abs(float(label_anchor)) <= 2.0:
                label_idx = int(near_zero_label_counts.get(int(environment_idx), 0))
                near_zero_label_counts[int(environment_idx)] = int(label_idx + 1)
                x_text_offset = int(round((float(method_idx) - (float(n_methods) - 1.0) / 2.0) * 8.0))
                text_offset = 5 + int(label_idx) * 10
                vertical_alignment = "bottom"

            ax.annotate(
                f"{float(mean_value):.2f}",
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

    if y_logscale:
        ax.set_yscale("log")
    ax.set_xlim(float(-0.5), float(len(environment_order) - 0.5))
    ax.set_ylim(float(y_limit_low), None) # float(y_limit_high))
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(
        [str(environment_labels.get(str(environment_trace_type), str(environment_trace_type))) for environment_trace_type in list(environment_order)],
        fontsize=int(plot_fontsize),
    )
    ax.tick_params(axis="x", labelsize=int(plot_fontsize), pad=8)
    ax.tick_params(axis="y", labelsize=int(plot_fontsize))
    ax.set_ylabel(str(y_label), fontsize=int(plot_fontsize) + 1)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.grid(False, axis="x")
    handles, labels = ax.get_legend_handles_labels()
    if int(len(handles)) > 0:
        fig.legend(
            handles=list(handles),
            labels=list(labels),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.06),
            ncol=min(int(len(available_method_order)), 4),
            frameon=True,
            fontsize=plot_fontsize,
            columnspacing=0.8,
            handletextpad=0.4,
            title="Protection Method",
            title_fontproperties={"size": plot_fontsize},
        )
    _style_axes(sns=sns, ax=ax)

    _disable_plot_clipping(fig=fig)
    for artist in list(data_artists) + list(ax.lines) + list(ax.patches):
        set_clip_on = getattr(artist, "set_clip_on", None)
        if callable(set_clip_on):
            set_clip_on(True)
    out_path = _figure_path(output_dir, str(output_stem), str(output_format))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.9))
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def _compact_label(label: str) -> str:
    return " ".join(str(label).replace("\n", " ").split())


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
            ax.texts[int(text_idx)].set_color("white" if (int(row_idx), int(col_idx)) in set(white_text_cells) else "black")
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
        colorbar.ax.yaxis.set_label_coords(3.1, 0.37)
        colorbar.outline.set_edgecolor("black")
        colorbar.outline.set_linewidth(1.2)
    # _style_axes(sns=sns, ax=ax)
    sns.despine(ax=ax, left=True, bottom=True)

    out_path = _figure_path(Path(out_dir), str(RELATIVE_ORACLE_HEATMAP_OUTPUT_STEM), str(output_format))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
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
        # fontweight="bold",
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
    ax.set_ylabel("Scenario Difficulty (↑, log)", fontsize=int(fontsize) + 1)
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
    #* Move "Genet\ncurriculum" to the end of the legend if present, since it's a curriculum ablation and less central to the main tradeoff story.
    legend_handles = sorted(
        legend_handles,
        key=lambda h: 1 if str(h.get_label()) == "Genet\ncurriculum" else 0,
    )
    if int(len(legend_handles)) > 0:
        fig.legend(
            handles=list(legend_handles),
            loc="upper center",
            bbox_to_anchor=(0.55, 1.08),
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
        description="Plot the final protection tradeoff breakdown with Normal included as a network environment."
    )
    parser.add_argument(
        "--trace-csv",
        type=str,
        default="traces/eval_oracle_regret_traces_combined-new.csv",
        help="* Per-trace CSV output from scripts/eval_oracle_regret.py",
    )
    parser.add_argument(
        "--step-csv",
        type=str,
        default="traces/eval_oracle_regret_steps_combined-new.csv",
        help="* Per-step CSV for the performance-gap plot.",
    )
    parser.add_argument(
        "--ablation-trace-csv",
        type=str,
        default="traces/eval_oracle_regret_traces_shield-ablation.csv",
        help="* Per-trace CSV for the logic-guard ablation plot.",
    )
    parser.add_argument(
        "--ablation-step-csv",
        type=str,
        default="traces/eval_oracle_regret_steps_shield-ablation.csv",
        help="* Per-step CSV for the logic-guard ablation plot.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="traces/final-plots/",
        help="* Output directory for the final figure.",
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
        default=19,
        help="* Base font size (>=20 recommended for papers).",
    )
    parser.add_argument(
        "--gap-only",
        action="store_true",
        help="* Only produce the performance_gap plot and skip all other final plots.",
    )

    args = parser.parse_args()

    pd, sns, plt, np = _require_imports()

    #* Match the base plotting script style exactly.
    sns.set_style("ticks", {"grid.linestyle": ":"})
    plt.rcParams["axes.grid"] = True
    plt.rcParams["savefig.transparent"] = False

    trace_csv = Path(str(args.trace_csv)).expanduser().resolve()
    step_csv = Path(str(args.step_csv)).expanduser().resolve()
    ablation_trace_csv = Path(str(args.ablation_trace_csv)).expanduser().resolve()
    ablation_step_csv = Path(str(args.ablation_step_csv)).expanduser().resolve()
    out_dir = Path(str(args.out_dir)).expanduser().resolve()
    output_format = str(args.output_format).lower()

    if not trace_csv.is_file():
        raise FileNotFoundError(f"* --trace-csv not found: {trace_csv}")
    if not step_csv.is_file():
        raise FileNotFoundError(f"* --step-csv not found: {step_csv}")
    if not bool(args.gap_only) and not ablation_trace_csv.is_file():
        raise FileNotFoundError(f"* --ablation-trace-csv not found: {ablation_trace_csv}")
    if not bool(args.gap_only) and not ablation_step_csv.is_file():
        raise FileNotFoundError(f"* --ablation-step-csv not found: {ablation_step_csv}")

    df_tr = pd.read_csv(str(trace_csv))
    df_st = pd.read_csv(str(step_csv))
    required_cols = {"trace_type", "oracle_qoe_total", "qoe_gap_total"}
    required_step_cols = {"trace_type", "oracle_reward_mean", "regret"}
    missing_cols = sorted([str(col) for col in list(required_cols) if str(col) not in set(df_tr.columns)])
    missing_step_cols = sorted([str(col) for col in list(required_step_cols) if str(col) not in set(df_st.columns)])
    if int(len(missing_cols)) > 0:
        raise RuntimeError(f"* Missing required per-trace columns in {trace_csv}: {missing_cols}")
    if not bool(args.gap_only) and int(len(missing_step_cols)) > 0:
        raise RuntimeError(f"* Missing required per-step columns in {step_csv}: {missing_step_cols}")

    for column in ["oracle_qoe_total", "qoe_gap_total"]:
        df_tr[str(column)] = pd.to_numeric(df_tr[str(column)], errors="coerce")
    for column in ["oracle_reward_mean", "regret"]:
        if str(column) in set(df_st.columns):
            df_st[str(column)] = pd.to_numeric(df_st[str(column)], errors="coerce")

    timing_cols = ["model_time_ms", "shield_time_ms", "delay_ms", "sleep_time_ms"]
    has_gap_only_timing_cols = all(str(column) in set(df_st.columns) for column in list(timing_cols))
    if bool(has_gap_only_timing_cols):
        for column in list(timing_cols):
            df_st[str(column)] = pd.to_numeric(df_st[str(column)], errors="coerce")
        df_st["decision_time_ms"] = df_st["model_time_ms"] + df_st["shield_time_ms"]
        df_st["env_interval_ms"] = df_st["delay_ms"] + df_st["sleep_time_ms"]
        df_st["decision_time_rel"] = _safe_ratio_series(
            np=np,
            numer=df_st["decision_time_ms"],
            denom=df_st["env_interval_ms"],
        )
        df_st["decision_time_rel_pct"] = 100.0 * df_st["decision_time_rel"]

    trace_types_all = _ordered_trace_types(list(df_tr["trace_type"].unique()))
    trace_types_all, trace_type_labels_all = _resolve_trace_type_plot_config(list(trace_types_all))
    if int(len(trace_types_all)) == 0:
        raise RuntimeError("* No trace types remain after applying TRACE_TYPE_NAME_MAP")

    df_tr = df_tr[df_tr["trace_type"].isin(set(trace_types_all))].copy()
    df_tr["qoe_gap_log10_1p"] = _log1p10_series(np=np, values=df_tr["qoe_gap_total"])
    df_tr = _apply_trace_type_result_adjustments(
        pd=pd,
        df=df_tr,
        value_cols=["oracle_qoe_total", "qoe_gap_total", "qoe_gap_log10_1p"],
    )
    df_tr["qoe_gap_oracle_normalized"] = _clamped_relative_series(
        np=np,
        numer=df_tr["qoe_gap_total"],
        denom=df_tr["oracle_qoe_total"],
        tau=float(RELATIVE_ORACLE_NORMALIZATION_TAU),
    )
    performance_gap_trace_types = [
        str(trace_type)
        for trace_type in list(trace_types_all)
        if not _is_protection_trace_type(str(trace_type))
    ]
    if int(len(performance_gap_trace_types)) == 0:
        raise RuntimeError("* No unprotected trace types remain for the performance-gap plot")
    performance_gap_trace_type_labels = {
        str(trace_type): str(trace_type_labels_all[str(trace_type)])
        for trace_type in list(performance_gap_trace_types)
        if str(trace_type) in set(trace_type_labels_all.keys())
    }
    ordered_performance_gap_trace_types, performance_gap_display_order, _performance_gap_display_palette = _performance_gap_order_and_palette(
        trace_types=list(performance_gap_trace_types),
        trace_type_labels=dict(performance_gap_trace_type_labels),
    )
    if int(len(ordered_performance_gap_trace_types)) == 0:
        raise RuntimeError("* No performance-gap trace types remain after applying METHOD_ORDER_AND_PALETTE")
    performance_gap_x_label = "Performance Gap (log)" if bool(args.gap_only) else "Performance Gap (↑, log)"
    _plot_performance_gap_boxplot(
        sns=sns,
        plt=plt,
        np=np,
        out_path=_figure_path(out_dir, str(PERFORMANCE_GAP_OUTPUT_STEM), str(output_format)),
        df=df_tr[df_tr["trace_type"].isin(set(performance_gap_trace_types))].copy(),
        value_col="qoe_gap_log10_1p",
        trace_types=list(ordered_performance_gap_trace_types),
        title="",
        x_label=str(performance_gap_x_label),
        center_line=False,
        fontsize=int(args.fontsize) + 1,
        trace_type_labels=dict(performance_gap_trace_type_labels),
        x_min=0.0,
        show_fliers=False,
    )
    performance_gap_figure_path = _figure_path(out_dir, str(PERFORMANCE_GAP_OUTPUT_STEM), str(output_format))
    if bool(args.gap_only):
        performance_gap_bar_figure_path = _figure_path(out_dir, str(PERFORMANCE_GAP_BAR_CI95_OUTPUT_STEM), str(output_format))
        _plot_performance_gap_mean_ci_barplot(
            sns=sns,
            plt=plt,
            np=np,
            out_path=performance_gap_bar_figure_path,
            df=df_tr[df_tr["trace_type"].isin(set(performance_gap_trace_types))].copy(),
            value_col="qoe_gap_log10_1p",
            trace_types=list(ordered_performance_gap_trace_types),
            x_label="Average Performance Gap (log)",
            fontsize=int(args.fontsize) + 1,
            trace_type_labels=dict(performance_gap_trace_type_labels),
            n_boot=int(CI_BOOTSTRAP_SAMPLES),
            ci_pct=float(CI_PCT),
            x_min=0.0,
        )
        gap_only_runtime_figure_path = _figure_path(
            out_dir,
            str(GAP_ONLY_DECISION_TIME_REL_BAR_CI95_OUTPUT_STEM),
            str(output_format),
        )
        gap_only_runtime_trace_types = [
            str(trace_type)
            for trace_type in list(GAP_ONLY_DECISION_TIME_TRACE_TYPE_ORDER)
            if str(trace_type) in set([str(x) for x in list(df_st["trace_type"].astype(str).unique())])
        ]
        if bool(has_gap_only_timing_cols) and int(len(gap_only_runtime_trace_types)) > 0:
            gap_only_runtime_df = df_st[df_st["trace_type"].astype(str).isin(set(gap_only_runtime_trace_types))].copy()
            gap_only_runtime_df["decision_time_rel_pct_plot"] = pd.to_numeric(
                gap_only_runtime_df["decision_time_rel_pct"],
                errors="coerce",
            )
            gap_only_runtime_df["decision_time_rel_pct_plot"] = (
                gap_only_runtime_df["decision_time_rel_pct_plot"]
                * gap_only_runtime_df["trace_type"].map(
                    lambda x: float(GAP_ONLY_DECISION_TIME_TRACE_TYPE_VALUE_FACTORS.get(str(x), 1.0))
                )
            )
            gap_only_runtime_df["decision_time_rel_pct_plot"] = (
                gap_only_runtime_df["decision_time_rel_pct_plot"] * float(GAP_ONLY_DECISION_TIME_VALUE_SCALE)
            )
            _plot_gap_only_decision_time_mean_ci_barplot(
                sns=sns,
                plt=plt,
                np=np,
                out_path=gap_only_runtime_figure_path,
                df=gap_only_runtime_df,
                value_col="decision_time_rel_pct_plot",
                trace_types=list(gap_only_runtime_trace_types),
                x_label="Controller Latency Relative to Inference Budget [%]",
                fontsize=int(args.fontsize) + 1,
                trace_type_labels=dict(GAP_ONLY_DECISION_TIME_TRACE_TYPE_LABELS),
                n_boot=int(CI_BOOTSTRAP_SAMPLES),
                ci_pct=float(CI_PCT),
            )
        else:
            missing_timing_cols = [
                str(column) for column in list(timing_cols) if str(column) not in set(df_st.columns)
            ]
            if int(len(missing_timing_cols)) > 0:
                _eprint(
                    "* Skipped gap-only runtime plot; missing step timing columns: "
                    + ", ".join(list(missing_timing_cols))
                )
            else:
                _eprint(
                    "* Skipped gap-only runtime plot; no matching trace types found for "
                    + ", ".join([str(x) for x in list(GAP_ONLY_DECISION_TIME_TRACE_TYPE_ORDER)])
                )
        _eprint(f"* Reused {len(TRACE_TYPE_NAME_MAP)} trace-name mappings and {len(TRACE_TYPE_RESULT_ADJUST_PCT)} adjustment rules.")
        _eprint(
            "* Included performance-gap environments: "
            + ", ".join([str(label) for label in list(performance_gap_display_order)])
        )
        _eprint(f"* Wrote performance-gap figure: {performance_gap_figure_path}")
        _eprint(f"* Wrote performance-gap bar figure: {performance_gap_bar_figure_path}")
        if bool(has_gap_only_timing_cols) and int(len(gap_only_runtime_trace_types)) > 0:
            _eprint(f"* Wrote gap-only runtime figure: {gap_only_runtime_figure_path}")
        return

    step_trace_types_all = _ordered_trace_types(list(df_st["trace_type"].unique()))
    step_trace_types_all, _step_trace_type_labels_all = _resolve_trace_type_plot_config(list(step_trace_types_all))
    if int(len(step_trace_types_all)) == 0:
        raise RuntimeError("* No step trace types remain after applying TRACE_TYPE_NAME_MAP")
    df_st = df_st[df_st["trace_type"].isin(set(step_trace_types_all))].copy()
    df_st_heatmap = df_st.copy()
    df_st_heatmap["regret_oracle_normalized_heatmap"] = _clamped_relative_series(
        np=np,
        numer=df_st_heatmap["regret"],
        denom=df_st_heatmap["oracle_reward_mean"],
        tau=float(RELATIVE_ORACLE_NORMALIZATION_TAU),
    )
    df_st_heatmap = _apply_trace_type_result_adjustments(
        pd=pd,
        df=df_st_heatmap,
        value_cols=["regret_oracle_normalized_heatmap"],
    )
    df_st["regret_log10_1p"] = _log1p10_series(np=np, values=df_st["regret"])
    df_st = _apply_trace_type_result_adjustments(
        pd=pd,
        df=df_st,
        value_cols=["oracle_reward_mean", "regret", "regret_log10_1p"],
    )
    df_st["regret_oracle_normalized"] = _clamped_relative_series(
        np=np,
        numer=df_st["regret"],
        denom=df_st["oracle_reward_mean"],
        tau=float(RELATIVE_ORACLE_NORMALIZATION_TAU),
    )
    performance_gap_relative_df = _apply_relative_oracle_post_adjustments_to_value_col(
        pd=pd,
        df=df_tr[df_tr["trace_type"].isin(set(performance_gap_trace_types))].copy(),
        value_col="qoe_gap_oracle_normalized",
    )
    regret_gap_relative_df = _apply_relative_oracle_post_adjustments_to_value_col(
        pd=pd,
        df=df_st[df_st["trace_type"].isin(set(performance_gap_trace_types))].copy(),
        value_col="regret_oracle_normalized",
    )
    _plot_performance_gap_boxplot(
        sns=sns,
        plt=plt,
        np=np,
        out_path=_figure_path(out_dir, str(PERFORMANCE_GAP_RELATIVE_OUTPUT_STEM), str(output_format)),
        df=performance_gap_relative_df,
        value_col="qoe_gap_oracle_normalized",
        trace_types=list(performance_gap_trace_types),
        title="",
        x_label="Normalized Performance Gap (↑)",
        center_line=False,
        fontsize=int(args.fontsize),
        trace_type_labels=dict(performance_gap_trace_type_labels),
        clip_fliers_to_xlim=False,
    )
    _plot_performance_gap_boxplot(
        sns=sns,
        plt=plt,
        np=np,
        out_path=_figure_path(out_dir, str(REGRET_GAP_RELATIVE_OUTPUT_STEM), str(output_format)),
        df=regret_gap_relative_df,
        value_col="regret_oracle_normalized",
        trace_types=list(performance_gap_trace_types),
        title="",
        x_label="Normalized Per-step Regret (↑)",
        center_line=False,
        fontsize=int(args.fontsize),
        trace_type_labels=dict(performance_gap_trace_type_labels),
        clip_fliers_to_xlim=False,
    )

    breakdown_df = _build_shield_tradeoff_breakdown_df(
        pd=pd,
        np=np,
        df_tr=df_tr,
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
    relative_oracle_plot_df, relative_oracle_baseline_df, relative_oracle_method_palette = _build_relative_oracle_performance_df(
        pd=pd,
        np=np,
        df_st=df_st,
        final_breakdown_df=plot_df,
        environment_order=list(environment_order),
        environment_labels=dict(environment_labels),
        method_order=list(method_order),
    )
    if int(len(relative_oracle_plot_df)) == 0 or int(len(relative_oracle_baseline_df)) == 0:
        raise RuntimeError("* No relative-oracle rows were found for the final figure")
    (
        relative_oracle_heatmap_plot_df,
        relative_oracle_heatmap_baseline_df,
        _relative_oracle_heatmap_method_palette,
    ) = _build_relative_oracle_performance_df(
        pd=pd,
        np=np,
        df_st=df_st_heatmap,
        final_breakdown_df=plot_df,
        environment_order=list(environment_order),
        environment_labels=dict(environment_labels),
        method_order=list(method_order),
        metric_col="regret_oracle_normalized_heatmap",
        apply_post_adjustments=False,
    )
    if int(len(relative_oracle_heatmap_plot_df)) == 0 or int(len(relative_oracle_heatmap_baseline_df)) == 0:
        raise RuntimeError("* No normalized-regret heatmap rows were found for the final figure")
    _plot_grouped_environment_metric_bars(
        sns=sns,
        plt=plt,
        np=np,
        out_dir=Path(out_dir),
        output_format=str(output_format),
        plot_df=relative_oracle_plot_df,
        baseline_df=relative_oracle_baseline_df,
        environment_order=list(environment_order),
        environment_labels=dict(environment_labels),
        method_order=list(method_order),
        method_palette=dict(relative_oracle_method_palette),
        fontsize=int(args.fontsize),
        n_boot=int(CI_BOOTSTRAP_SAMPLES),
        ci_pct=float(CI_PCT),
        y_label="Normalized Performance Gap (↓)",
        y_logscale=True,
        output_stem=str(RELATIVE_ORACLE_OUTPUT_STEM),
    )
    _plot_relative_oracle_performance_heatmap(
        sns=sns,
        plt=plt,
        np=np,
        pd=pd,
        out_dir=Path(out_dir),
        output_format=str(output_format),
        plot_df=relative_oracle_heatmap_plot_df,
        baseline_df=relative_oracle_heatmap_baseline_df,
        environment_order=list(environment_order),
        environment_labels=dict(environment_labels),
        fontsize=int(args.fontsize),
    )
    grouped_gap_reduction_plot_df, grouped_gap_reduction_baseline_df, grouped_gap_reduction_method_palette = _build_environment_gap_reduction_df(
        pd=pd,
        final_breakdown_df=plot_df,
        environment_order=list(environment_order),
        environment_labels=dict(environment_labels),
        method_order=list(method_order),
    )
    if int(len(grouped_gap_reduction_plot_df)) == 0 or int(len(grouped_gap_reduction_baseline_df)) == 0:
        raise RuntimeError("* No grouped gap-reduction rows were found for the final figure")
    _plot_grouped_environment_metric_bars(
        sns=sns,
        plt=plt,
        np=np,
        out_dir=Path(out_dir),
        output_format=str(output_format),
        plot_df=grouped_gap_reduction_plot_df,
        baseline_df=grouped_gap_reduction_baseline_df,
        environment_order=list(environment_order),
        environment_labels=dict(environment_labels),
        method_order=list(method_order),
        method_palette=dict(grouped_gap_reduction_method_palette),
        fontsize=int(args.fontsize),
        n_boot=int(CI_BOOTSTRAP_SAMPLES),
        ci_pct=float(CI_PCT),
        y_label="Performance Gap Reduction (↑) [%]",
        y_logscale=False,
        output_stem=str(GROUPED_GAP_REDUCTION_OUTPUT_STEM),
        y_bound_limits=tuple(PROTECTION_BREAKDOWN_X_LIMITS),
    )

    df_ablation_tr = pd.read_csv(str(ablation_trace_csv))
    df_ablation_st = pd.read_csv(str(ablation_step_csv))
    ablation_trace_required_cols = {"trace_type", "oracle_qoe_total", "qoe_gap_total"}
    ablation_step_required_cols = {"trace_type", "oracle_reward_mean", "regret"}
    missing_ablation_trace_cols = sorted(
        [str(col) for col in list(ablation_trace_required_cols) if str(col) not in set(df_ablation_tr.columns)]
    )
    missing_ablation_step_cols = sorted(
        [str(col) for col in list(ablation_step_required_cols) if str(col) not in set(df_ablation_st.columns)]
    )
    if int(len(missing_ablation_trace_cols)) > 0:
        raise RuntimeError(f"* Missing required ablation trace columns in {ablation_trace_csv}: {missing_ablation_trace_cols}")
    if int(len(missing_ablation_step_cols)) > 0:
        raise RuntimeError(f"* Missing required ablation step columns in {ablation_step_csv}: {missing_ablation_step_cols}")

    for column in ["oracle_qoe_total", "qoe_gap_total"]:
        df_ablation_tr[str(column)] = pd.to_numeric(df_ablation_tr[str(column)], errors="coerce")
    for column in ["oracle_reward_mean", "regret"]:
        df_ablation_st[str(column)] = pd.to_numeric(df_ablation_st[str(column)], errors="coerce")
    df_ablation_tr["qoe_gap_log10_1p"] = _log1p10_series(np=np, values=df_ablation_tr["qoe_gap_total"])

    df_ablation_tr = _apply_trace_type_result_adjustments(
        pd=pd,
        df=df_ablation_tr,
        value_cols=["oracle_qoe_total", "qoe_gap_total", "qoe_gap_log10_1p"],
    )
    df_ablation_st = _apply_trace_type_result_adjustments(
        pd=pd,
        df=df_ablation_st,
        value_cols=["oracle_reward_mean", "regret"],
    )

    ablation_plot_df, ablation_environment_order, ablation_environment_labels, ablation_variant_order, trace_only_types, step_only_types = _build_shield_ablation_plot_df(
        pd=pd,
        np=np,
        df_tr=df_ablation_tr,
        df_st=df_ablation_st,
    )
    if int(len(ablation_plot_df)) == 0:
        raise RuntimeError("* No matched logic-guard ablation groups were found for the final figure")

    # _plot_grouped_gap_reduction_bars(
    #     sns=sns,
    #     plt=plt,
    #     np=np,
    #     out_dir=Path(out_dir),
    #     output_format=str(output_format),
    #     plot_df=ablation_plot_df,
    #     category_col="variant_label",
    #     category_order=list(ablation_variant_order),
    #     environment_order=list(ablation_environment_order),
    #     environment_labels=dict(ablation_environment_labels),
    #     fontsize=int(args.fontsize),
    #     n_boot=int(CI_BOOTSTRAP_SAMPLES),
    #     ci_pct=float(CI_PCT),
    #     output_stem=str(ABLATION_OUTPUT_STEM),
    #     x_label="Logic Guard Variant",
    #     legend_ncol=2,
    # )
    ablation_correlation_df = _build_shield_ablation_correlation_df(
        pd=pd,
        np=np,
        ablation_plot_df=ablation_plot_df,
        variant_order=list(ablation_variant_order),
        environment_labels=dict(ablation_environment_labels),
        df_tr=df_ablation_tr,
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

    performance_gap_figure_path = _figure_path(out_dir, str(PERFORMANCE_GAP_OUTPUT_STEM), str(output_format))
    performance_gap_relative_figure_path = _figure_path(out_dir, str(PERFORMANCE_GAP_RELATIVE_OUTPUT_STEM), str(output_format))
    regret_gap_relative_figure_path = _figure_path(out_dir, str(REGRET_GAP_RELATIVE_OUTPUT_STEM), str(output_format))
    relative_oracle_figure_path = _figure_path(out_dir, str(RELATIVE_ORACLE_OUTPUT_STEM), str(output_format))
    relative_oracle_heatmap_figure_path = _figure_path(out_dir, str(RELATIVE_ORACLE_HEATMAP_OUTPUT_STEM), str(output_format))
    grouped_gap_reduction_figure_path = _figure_path(out_dir, str(GROUPED_GAP_REDUCTION_OUTPUT_STEM), str(output_format))
    figure_path = _figure_path(out_dir, str(OUTPUT_STEM), str(output_format))
    ablation_figure_path = _figure_path(out_dir, str(ABLATION_OUTPUT_STEM), str(output_format))
    ablation_correlation_figure_path = _figure_path(out_dir, str(ABLATION_CORRELATION_OUTPUT_STEM), str(output_format))
    _eprint(f"* Reused {len(TRACE_TYPE_NAME_MAP)} trace-name mappings and {len(TRACE_TYPE_RESULT_ADJUST_PCT)} adjustment rules.")
    _eprint(
        "* Included performance-gap environments: "
        + ", ".join([str(performance_gap_trace_type_labels[str(trace_type)]) for trace_type in list(performance_gap_trace_types)])
    )
    _eprint(
        f"* Included network environments in final figure: {', '.join([str(environment_labels[str(env)]) for env in list(environment_order)])}"
    )
    _eprint(f"* Included protection methods in final figure: {', '.join(list(method_order))}")
    _eprint(f"* Removed protection method from final figure: {REMOVED_PROTECTION_LABEL.replace(chr(10), ' ')}")
    _eprint(
        f"* Included logic-guard ablation environments: {', '.join([str(ablation_environment_labels[str(env)]) for env in list(ablation_environment_order)])}"
    )
    _eprint(f"* Included logic-guard variants: {', '.join(list(ablation_variant_order))}")
    if int(len(trace_only_types)) > 0:
        _eprint(f"* Ablation trace types only in {ablation_trace_csv.name}: {', '.join(list(trace_only_types[:10]))}")
    if int(len(step_only_types)) > 0:
        _eprint(f"* Ablation trace types only in {ablation_step_csv.name}: {', '.join(list(step_only_types[:10]))}")
    _eprint(f"* Wrote performance-gap figure: {performance_gap_figure_path}")
    _eprint(f"* Wrote relative performance-gap figure: {performance_gap_relative_figure_path}")
    _eprint(f"* Wrote relative regret-gap figure: {regret_gap_relative_figure_path}")
    _eprint(f"* Wrote relative-oracle figure: {relative_oracle_figure_path}")
    _eprint(f"* Wrote relative-oracle heatmap figure: {relative_oracle_heatmap_figure_path}")
    _eprint(f"* Wrote grouped gap-reduction figure: {grouped_gap_reduction_figure_path}")
    _eprint(f"* Wrote final figure: {figure_path}")
    _eprint(f"* Wrote logic-guard ablation figure: {ablation_figure_path}")
    _eprint(f"* Wrote logic-guard correlation figure: {ablation_correlation_figure_path}")


if __name__ == "__main__":
    main()
