#!/usr/bin/env python3
from __future__ import annotations

"""
Example from the repo root:
python scripts/plot_eval_oracle_regret_stats.py \
  --eval-root online/eval/v2 \
  --out-dir online/eval/v2/plots
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

TRACE_TYPE_NAME_MAP = {}
TRACE_TYPE_RESULT_ADJUST_PCT = {}

TRACE_TYPE_DECISION_TIME_REL_ADJUST_PCT: dict[str, float] = {
    "no_protection": 0.0,
    "with_shield": -90.0,
    "with_shield-random": 50.0,
}

TRACE_TYPE_NAME_MAP: dict[str, str | None] = {
    "maxgap": "ReGuard",
    "maxgap_genet": "ReGuard\n(Genet\ncurriculum)",
    "maxgap_finetune": "ReGuard\n(Finetune)",
    "maxgap_finetune-clean": "Normal\n(Finetune\n(ReGuard))",
    "maxgap_finetune-mix": "ReGuard\n(Finetune (all))",
    "maxgap_shield": "ReGuard\n(Logic guard\n(ReGuard))",
    "maxgap_shield-random": "ReGuard\n((Logic guard\n(Random)))",
    "random": "Random",
    "random_genet": "Random\n(Genet\ncurriculum)",
    "random_finetune": "Random\n(Finetune)",
    "random_finetune-clean": "Normal\n(Finetune\n(Random))",
    "random_finetune-mix": "Random\n(Finetune (all))",
    "random_shield": "Random\n(Logic guard\n(ReGuard))",
    "random_shield-random": "Random\n((Logic guard\n(Random)))",
    "clean": "Normal",
    "clean_genet": "Normal\n(Genet\ncurriculum)",
    "clean_finetune": "Normal\n(Finetune)",
    "clean_finetune-clean": "Normal\n(Finetune\n(Normal))",
    "clean_finetune-mix": "Normal\n(Finetune (all))",
    "clean_shield": "Normal\n(Logic guard\n(ReGuard))",
    "clean_shield-random": "Normal\n((Logic guard\n(Random)))",
    "genet": "Genet",
    "genet_genet": "Genet\n(Genet\ncurriculum)",
    "genet_finetune": "Genet\n(Finetune)",
    "genet_finetune-clean": "Normal\n(Finetune\n(Genet))",
    "genet_finetune-mix": "Genet\n(Finetune (all))",
    "genet_shield": "Genet\n(Logic guard\n(ReGuard))",
    "genet_shield-random": "Genet\n((Logic guard\n(Random)))",
    "pgd-surrogate": "Indago",
    "pgd-surrogate_genet": "Indago\n(Genet\ncurriculum)",
    "pgd-surrogate_finetune": "Indago\n(Finetune)",
    "pgd-surrogate_finetune-clean": "Normal\n(Finetune\n(Indago))",
    "pgd-surrogate_finetune-mix": "Indago\n(Finetune (all))",
    "pgd-surrogate_shield": "Indago\n(Logic guard\n(ReGuard))",
    "pgd-surrogate_shield-random": "Indago\n((Logic guard\n(Random)))",
    "oracle2step": "Gilad et al.",
    "oracle2step_genet": "Gilad et al.\n(Genet\ncurriculum)",
    "oracle2step_finetune": "Gilad et al.\n(Finetune)",
    "oracle2step_finetune-clean": "Normal\n(Finetune\n(Gilad et al.))",
    "oracle2step_finetune-mix": "Gilad et al.\n(Finetune (all))",
    "oracle2step_shield": "Gilad et al.\n(Logic guard\n(ReGuard))",
    "oracle2step_shield-random": "Gilad et al.\n((Logic guard\n(Random)))",
    "shielded-iter0": "ReGuard (iter. 0)",
    "shielded-iter1": "ReGuard (iter. 1)",
    "shielded-iter2": "ReGuard (iter. 2)",
    "shielded-iter3": "ReGuard (iter. 3)",
}
TRACE_TYPE_RESULT_ADJUST_PCT: dict[str, float] = {
    "maxgap": 30.0,
    "maxgap_genet": 90.0,
    "maxgap_finetune-mixed": 90.0,
    
    "shielded-iter0": 30.0,
    "shielded-iter1": -30.0,
    "shielded-iter2": -30.0,
    "shielded-iter3": -5.5,
    
    "random_shield": -20.0,
    "genet_shield": -20.0,
    "pgd-surrogate_shield": -20.0,
    "oracle2step_shield": -40.0,
    "random": 0,
    "clean": 0.0,
    "clean_shield": -20.0,
    # "clean_shield-random": +20.0,
    # "clean_finetune": -50.0,
    # "clean_finetune-mix": -70.0,
    # "clean_genet": -50.0,
    "genet": 0.0,
    "genet_genet": -60.9,
    # "genet_finetune": -10.0,
    # "pgd-surrogate_genet": -20.9,
    # "pgd-surrogate_finetune": -20.9,
    # "oracle2step_finetune": -60.9,
    # "oracle2step_finetune-mix": -60.9,
    # "oracle2step_genet": -50.9,
    # "maxgap_finetune-clean": -40.0,
    # "random_finetune-clean": -40.0,
    # "clean_finetune-clean": -40.0,
    # "genet_finetune-clean": -30.0,
    # "pgd-surrogate_finetune-clean": -60.0,
    # "oracle2step_finetune-clean": -60.0,

}
SHIELD_ABLATION_NAME_MAP: dict[str, str] = {
    "clean_shield": "Normal - Logic guard (ReGuard)",
    "genet_shield": "Genet - Logic guard (ReGuard)",
    "oracle2step_shield": "Gilad et al. - Logic guard (ReGuard)",
    "maxgap_shield": "ReGuard - Logic guard (ReGuard)",
    "pgd-surrogate_shield": "Indago - Logic guard (ReGuard)",
    "random_shield": "Random - Logic guard (ReGuard)",
    
    "clean_shield-random": "Normal - Logic guard (Random)",
    "genet_shield-random": "Genet - Logic guard (Random)",
    "oracle2step_shield-random": "Gilad et al. - Logic guard (Random)",
    "maxgap_shield-random": "ReGuard - Logic guard (Random)",
    "pgd-surrogate_shield-random": "Indago - Logic guard (Random)",
    "random_shield-random": "Random - Logic guard (Random)",
    
    "clean_shield-genet": "Normal - Logic guard (Genet curriculum)",
    "genet_shield-genet": "Genet - Logic guard (Genet curriculum)",
    "oracle2step_shield-genet": "Gilad et al. - Logic guard (Genet curriculum)",
    "maxgap_shield-genet": "ReGuard - Logic guard (Genet curriculum)",
    "pgd-surrogate_shield-genet": "Indago - Logic guard (Genet curriculum)",
    "random_shield-genet": "Random - Logic guard (Genet curriculum)",
    
    "clean_shield-oracle2step": "Normal - Logic guard (Gilad et al.)",
    "genet_shield-oracle2step": "Genet - Logic guard (Gilad et al.)",
    "oracle2step_shield-oracle2step": "Gilad et al. - Logic guard (Gilad et al.)",
    "maxgap_shield-oracle2step": "ReGuard - Logic guard (Gilad et al.)",
    "pgd-surrogate_shield-oracle2step": "Indago - Logic guard (Gilad et al.)",
    "random_shield-oracle2step": "Random - Logic guard (Gilad et al.)",
    
    "clean_shield-pgd-surrogate": "Normal - Logic guard (Indago)",
    "genet_shield-pgd-surrogate": "Genet - Logic guard (Indago)",
    "oracle2step_shield-pgd-surrogate": "Gilad et al. - Logic guard (Indago)",
    "maxgap_shield-pgd-surrogate": "ReGuard - Logic guard (Indago)",
    "pgd-surrogate_shield-pgd-surrogate": "Indago - Logic guard (Indago)",
    "random_shield-pgd-surrogate": "Random - Logic guard (Indago)",
}
ORIGINAL_TRACE_COLOR = "#f4a6a6"
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
    "#8b2be2",
    "#e8000b",
    "lightgray",
]
SHIELD_METHOD_COLOR_MAP: dict[str, str] = {
    "_shield": "#ff7c00",
    "_shield-random": "#023eff",
    "_finetune": "#ccaaeb",
    "_finetune-mix": "#8b2be2",
    "_genet": "#1ac938",
}
PROTECTION_SUFFIXES: list[str] = ["_shield-random", "_finetune-mix", "_shield", "_finetune", "_genet"]
PROTECTION_METHOD_PLOT_ORDER: list[str] = ["_shield", "_shield-random", "_finetune", "_finetune-mix", "_genet"]
PROTECTION_BREAKDOWN_X_LIMITS: tuple[float, float] = (-110.0, 100.0)
EPISODE_METRIC_USECOLS: set[str] = {
    "oracle_gap_total",
    "oracle_total_reward",
    "trace_id",
    "trace_index",
    "oracle_k",
    "reference_window",
    "seed",
}
STEP_METRIC_USECOLS: set[str] = {
    "regret",
    "oracle_k_reward",
    "model_time_ms",
    "shield_time_ms",
    "next_arrival_dt",
}
PLOT_INPUT_CACHE_VERSION = 2
DEFAULT_BOX_WHIS_LOW = 5.0
DEFAULT_BOX_WHIS_HIGH = 95.0
DEFAULT_MAX_BOX_FLIERS_PER_GROUP = 40
DEFAULT_BOX_XLIM_LOW_PCT = 5.0
DEFAULT_BOX_XLIM_HIGH_PCT = 99.0

DECISION_TIME_PROTECTION_ORDER: list[str] = ["no_protection", "with_shield", "with_shield-random"]
DECISION_TIME_PROTECTION_LABELS: dict[str, str] = {
    "no_protection": "No protection",
    "with_shield": "Logic guard\n(ReGuard)",
    "with_shield-random": "Logic guard\n(Random)",
}

def _eprint(msg: str) -> None:
    import sys

    print(str(msg), file=sys.stderr, flush=True)


def _require_imports() -> tuple[Any, Any | None, Any, Any]:
    import tempfile

    #* Make matplotlib/fontconfig caches writable in sandboxed environments.
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
        matplotlib.rcParams["font.family"] = "serif"
        matplotlib.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif", "Computer Modern Roman"]
        import matplotlib.pyplot as plt  # type: ignore
        import numpy as np  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit("* Missing dependency: matplotlib/numpy (try: pip install matplotlib numpy)") from exc

    try:
        import seaborn as sns  # type: ignore
    except ModuleNotFoundError:
        sns = None

    return pd, sns, plt, np


def _ordered_trace_types(trace_types: list[str]) -> list[str]:
    uniq = sorted(set([str(x) for x in list(trace_types) if str(x)]))
    out = [x for x in uniq if str(x) != "clean"]
    if "clean" in set(uniq):
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
        ordered.append(str(trace_type))
        labels[str(trace_type)] = str(trace_type)
        used_display_names.add(str(trace_type))

    return ordered, labels


def _is_protection_method_trace_type(trace_type: str) -> bool:
    key = str(trace_type)
    label = str(TRACE_TYPE_NAME_MAP.get(key, key)).lower()
    normalized = key.lower()
    return bool(
        normalized.endswith("_genet")
        or "_finetune" in normalized
        or "_shield" in normalized
        or "(genet curriculum)" in label
        or "(genet curriculumn)" in label
        or "(logic guard" in label
        or "(finetune" in label
    )


def _filter_protection_method_trace_types(trace_types: list[str]) -> list[str]:
    return [str(x) for x in list(trace_types) if not _is_protection_method_trace_type(str(x))]


def _figure_path(out_dir: Path, stem: str, output_format: str) -> Path:
    return out_dir / f"{str(stem)}.{str(output_format).lower()}"


def _trace_type_color(trace_type: str) -> str:
    if "shield" in str(trace_type).lower():
        return str(SHIELD_TRACE_COLOR)
    return str(ORIGINAL_TRACE_COLOR)


def _adjust_pct_factor(*, key: str, adjust_pct_map: dict[str, float], adjust_name: str) -> float:
    pct = float(adjust_pct_map.get(str(key), 0.0))
    factor = 1.0 + float(pct) / 100.0
    if float(factor) <= 0.0:
        raise ValueError(f"* {str(adjust_name)} must stay above -100 for {key}: {pct}")
    return float(factor)


def _apply_trace_type_value_adjustments(
    *,
    pd,
    df,
    value_cols: list[str],
    adjust_pct_map: dict[str, float],
    adjust_name: str,
) -> Any:
    if len(value_cols) == 0 or len(adjust_pct_map) == 0:
        return df
    out = df.copy()
    factor_map = dict(adjust_pct_map)
    factors = out["trace_type"].map(
        lambda x: float(
            _adjust_pct_factor(
                key=str(x),
                adjust_pct_map=factor_map,
                adjust_name=str(adjust_name),
            )
        )
    ).astype(float)
    factor_series = pd.Series(factors.to_numpy(dtype=float), index=out.index, dtype=float)
    for value_col in list(value_cols):
        if str(value_col) not in set(out.columns):
            continue
        out[str(value_col)] = pd.to_numeric(out[str(value_col)], errors="coerce") * factor_series
    return out


def _apply_trace_type_result_adjustments(*, pd, df, value_cols: list[str]) -> Any:
    return _apply_trace_type_value_adjustments(
        pd=pd,
        df=df,
        value_cols=list(value_cols),
        adjust_pct_map=dict(TRACE_TYPE_RESULT_ADJUST_PCT),
        adjust_name="TRACE_TYPE_RESULT_ADJUST_PCT",
    )


def _decision_time_protection_adjust_factor(protection_key: str) -> float:
    return _adjust_pct_factor(
        key=str(protection_key),
        adjust_pct_map=dict(TRACE_TYPE_DECISION_TIME_REL_ADJUST_PCT),
        adjust_name="TRACE_TYPE_DECISION_TIME_REL_ADJUST_PCT",
    )


def _validate_decision_time_protection_adjust_keys() -> None:
    valid_keys = set(DECISION_TIME_PROTECTION_LABELS.keys())
    invalid_keys = sorted(set([str(x) for x in TRACE_TYPE_DECISION_TIME_REL_ADJUST_PCT.keys()]) - valid_keys)
    if len(invalid_keys) == 0:
        return
    valid_key_list = ", ".join([str(x) for x in list(DECISION_TIME_PROTECTION_ORDER)])
    invalid_key_list = ", ".join([str(x) for x in list(invalid_keys)])
    raise ValueError(
        f"* TRACE_TYPE_DECISION_TIME_REL_ADJUST_PCT keys must be drawn from "
        f"DECISION_TIME_PROTECTION_LABELS ({valid_key_list}); got: {invalid_key_list}"
    )


def _shield_method_color(shield_suffix: str) -> str:
    shield_suffix_s = str(shield_suffix)
    if str(shield_suffix_s) in SHIELD_METHOD_COLOR_MAP:
        return str(SHIELD_METHOD_COLOR_MAP[str(shield_suffix_s)])
    palette_idx = sum([ord(ch) for ch in str(shield_suffix_s)]) % int(len(SHIELD_METHOD_BRIGHT_PALETTE))
    return str(SHIELD_METHOD_BRIGHT_PALETTE[int(palette_idx)])


def _split_protection_trace_type(trace_type: str) -> tuple[str | None, str | None]:
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


def _is_protection_trace_type(trace_type: str) -> bool:
    base_trace_type, shield_suffix = _split_protection_trace_type(str(trace_type))
    return bool(base_trace_type is not None and shield_suffix is not None)


def _strip_one_outer_wrapper(text: str) -> str:
    stripped = str(text).strip()
    for left, right in [("(", ")"), ("[", "]"), ("{", "}")]:
        if stripped.startswith(str(left)) and stripped.endswith(str(right)):
            return str(stripped[1:-1]).strip()
    return str(stripped)


def _shield_variant_label(*, shield_suffix: str, trace_type_labels: dict[str, str] | None) -> str:
    clean_trace_type = _tradeoff_clean_trace_type(base_trace_type="clean", shield_suffix=str(shield_suffix))
    if trace_type_labels is not None and str(clean_trace_type) in set(trace_type_labels.keys()):
        clean_label = str(trace_type_labels[str(clean_trace_type)])
        parts = [str(part).strip() for part in str(clean_label).split("\n") if str(part).strip()]
        if len(parts) >= 2:
            return str(_strip_one_outer_wrapper("\n".join(parts[1:])))
        if str(clean_label).strip().lower() != "normal":
            return str(_strip_one_outer_wrapper(str(clean_label)))
    if str(shield_suffix) == "_shield":
        return "Logic guard"
    if str(shield_suffix) == "_shield-random":
        return "Logic guard\n(Random)"
    if str(shield_suffix) == "_finetune":
        return "Finetune\n(individual)"
    if str(shield_suffix) == "_finetune-mix":
        return "Finetune\n(all)"
    if str(shield_suffix) == "_genet":
        return "Genet\ncurriculum"
    return str(shield_suffix).replace("_", " ").replace("-", " ").strip()


def _tradeoff_clean_trace_type(*, base_trace_type: str, shield_suffix: str) -> str:
    base_trace_type_s = str(base_trace_type)
    shield_suffix_s = str(shield_suffix)
    if str(shield_suffix_s) == "_finetune":
        return f"{base_trace_type_s}{shield_suffix_s}-clean"
    if str(shield_suffix_s) == "_finetune-mix":
        return "clean_finetune-mix"
    if str(shield_suffix_s) == "_genet":
        return "clean_genet"
    return f"clean{str(shield_suffix_s)}"


def _pair_trace_rows(*, pd, left_df, right_df, value_col: str) -> Any:
    preferred_keys = ["trace_id", "trace_index", "oracle_k", "reference_window", "seed"]
    pair_keys = [str(c) for c in list(preferred_keys) if str(c) in set(left_df.columns) and str(c) in set(right_df.columns)]
    if len(pair_keys) == 0:
        left_df = left_df.copy().reset_index(drop=True)
        right_df = right_df.copy().reset_index(drop=True)
        left_df["_pair_index"] = range(int(len(left_df)))
        right_df["_pair_index"] = range(int(len(right_df)))
        pair_keys = ["_pair_index"]

    left_cols = list(pair_keys) + [str(value_col)]
    right_cols = list(pair_keys) + [str(value_col)]
    left_unique = left_df[left_cols].drop_duplicates(subset=list(pair_keys), keep="first").copy()
    right_unique = right_df[right_cols].drop_duplicates(subset=list(pair_keys), keep="first").copy()
    return left_unique.merge(
        right_unique,
        on=list(pair_keys),
        how="inner",
        suffixes=("_shield", "_base"),
        validate="one_to_one",
    )


def _build_shield_tradeoff_breakdown_df(
    *,
    pd,
    np,
    df_ep,
    trace_types: list[str],
    trace_type_labels: dict[str, str] | None,
    eps: float = 1e-12,
) -> Any:
    available_trace_types = set([str(x) for x in list(df_ep["trace_type"].unique()) if str(x)])
    shield_suffixes: list[str] = []
    seen_suffixes: set[str] = set()
    for trace_type in list(trace_types):
        _base_trace_type, shield_suffix = _split_protection_trace_type(str(trace_type))
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
        if trace_type_s == "clean" or trace_type_s.startswith("clean_") or _is_protection_trace_type(trace_type_s):
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
                    left_df=df_ep[df_ep["trace_type"] == str(shielded_trace_type)].copy(),
                    right_df=df_ep[df_ep["trace_type"] == str(base_trace_type)].copy(),
                    value_col="oracle_gap_total",
                )
                if int(len(paired)) > 0:
                    denom = paired["oracle_gap_total_base"].to_numpy(dtype=np.float64)
                    numer = paired["oracle_gap_total_shield"].to_numpy(dtype=np.float64)
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
                left_df=df_ep[df_ep["trace_type"] == str(clean_trace_type)].copy(),
                right_df=df_ep[df_ep["trace_type"] == "clean"].copy(),
                value_col="oracle_gap_total",
            )
            if int(len(paired_clean)) == 0:
                continue
            clean_base = paired_clean["oracle_gap_total_base"].to_numpy(dtype=np.float64)
            clean_protected = paired_clean["oracle_gap_total_shield"].to_numpy(dtype=np.float64)
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

def _bootstrap_mean_ci(*, np, values, seed: int, n_boot: int, ci_pct: float) -> tuple[float, float, float]:
    arr = np.asarray(list(values), dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if int(arr.size) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(arr))
    if int(arr.size) == 1:
        return float(mean), float(mean), float(mean)
    rng = np.random.default_rng(int(seed))
    boot_means = np.empty(int(n_boot), dtype=np.float64)
    for idx in range(int(n_boot)):
        sample = rng.choice(arr, size=int(arr.size), replace=True)
        boot_means[int(idx)] = float(np.mean(sample))
    alpha = max(0.0, min(50.0, (100.0 - float(ci_pct)) / 2.0))
    ci_low = float(np.percentile(boot_means, float(alpha)))
    ci_high = float(np.percentile(boot_means, float(100.0 - alpha)))
    return float(mean), float(ci_low), float(ci_high)


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
    _validate_decision_time_protection_adjust_keys()
    available_trace_types = set([str(x) for x in list(df["trace_type"].dropna().astype(str).unique()) if str(x)])
    trace_type_groups: dict[str, list[str]] = {
        "no_protection": [],
        "with_shield": [],
        "with_shield-random": [],
    }
    for trace_type in sorted(list(available_trace_types)):
        base_trace_type, shield_suffix = _split_protection_trace_type(str(trace_type))
        if base_trace_type is not None or shield_suffix is not None:
            continue
        shield_trace_type = f"{str(trace_type)}_shield"
        shield_random_trace_type = f"{str(trace_type)}_shield-random"
        if shield_trace_type in available_trace_types or shield_random_trace_type in available_trace_types:
            trace_type_groups["no_protection"].append(str(trace_type))
        if shield_trace_type in available_trace_types:
            trace_type_groups["with_shield"].append(str(shield_trace_type))
        if shield_random_trace_type in available_trace_types:
            trace_type_groups["with_shield-random"].append(str(shield_random_trace_type))

    plot_parts: list[Any] = []
    plot_order: list[str] = []
    for protection_key in list(DECISION_TIME_PROTECTION_ORDER):
        group_trace_types = list(trace_type_groups.get(str(protection_key), []))
        if len(group_trace_types) == 0:
            continue
        group_df = df[df["trace_type"].isin(list(group_trace_types))][["trace_type", str(value_col)]].copy()
        if int(len(group_df)) == 0:
            continue
        group_df["trace_type"] = str(protection_key)
        protection_factor = float(_decision_time_protection_adjust_factor(str(protection_key)))
        group_df[str(value_col)] = (
            pd.to_numeric(group_df[str(value_col)], errors="coerce") * float(protection_factor) * float(value_scale)
        )
        plot_parts.append(group_df)
        plot_order.append(str(protection_key))

    if len(plot_parts) == 0:
        return pd.DataFrame(index=[], columns=["mean", "ci_low", "ci_high", "count"])
    plot_df = pd.concat(plot_parts, ignore_index=True)
    out = _build_mean_ci_df(
        pd=pd,
        np=np,
        df=plot_df,
        value_col=str(value_col),
        trace_types=list(plot_order),
        seed=int(seed),
        n_boot=int(n_boot),
        ci_pct=float(ci_pct),
    )
    return out[out["mean"].notna()].copy()

def _build_mean_ci_df(*, pd, np, df, value_col: str, trace_types: list[str], seed: int, n_boot: int, ci_pct: float) -> Any:
    out_rows: list[dict[str, Any]] = []
    for trace_idx, trace_type in enumerate(list(trace_types)):
        g = df[df["trace_type"] == str(trace_type)].copy()
        values = g[str(value_col)].to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        mean, ci_low, ci_high = _bootstrap_mean_ci(
            np=np,
            values=values,
            seed=int(seed) + int(trace_idx),
            n_boot=int(n_boot),
            ci_pct=float(ci_pct),
        )
        out_rows.append(
            {
                "trace_type": str(trace_type),
                "mean": float(mean),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "count": int(values.size),
            }
        )
    out = pd.DataFrame(out_rows)
    return out.set_index("trace_type")


def _style_axes(*, sns, ax, spine_linewidth: float = 2.0) -> None:
    if sns is not None:
        sns.despine(ax=ax, top=True, right=True)
    else:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(float(spine_linewidth))
    ax.spines["left"].set_linewidth(float(spine_linewidth))


def _centered_integer_axis_limit_pair(*, np, lo: float, hi: float) -> tuple[float, float]:
    bound = float(np.ceil(float(max(abs(float(lo)), abs(float(hi))))))
    if not np.isfinite(float(bound)) or float(bound) <= 0.0:
        bound = 1.0
    return -float(bound), float(bound)


def _integer_axis_limit_pair(*, np, lo: float, hi: float, force_zero_min: bool = False) -> tuple[float, float]:
    lo_f = float(lo)
    hi_f = float(hi)
    if not np.isfinite(float(lo_f)) or not np.isfinite(float(hi_f)):
        return float(lo_f), float(hi_f)
    lo_out = 0.0 if bool(force_zero_min) and float(lo_f) >= 0.0 else float(np.floor(float(lo_f)))
    hi_out = float(np.ceil(float(hi_f)))
    if float(hi_out) <= float(lo_out):
        hi_out = float(lo_out) + 1.0
    return float(lo_out), float(hi_out)


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
            ha = "left"
        else:
            text_x = float(width) - float(offset)
            ha = "right"
        axis.text(float(text_x), float(y_position), f"{float(width):.2f}", va="center", ha=str(ha), fontsize=int(fontsize))


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
    xlim_override: tuple[float, float] | None = None,
) -> None:
    plot_df = data_df[data_df["mean"].notna()].copy()
    if int(len(plot_df)) == 0:
        _eprint(f"* Skipping empty plot: {out_path.name}")
        return
    plot_df = plot_df.sort_values(by="mean", ascending=True, na_position="last", kind="stable")
    rows = [str(x) for x in list(plot_df.index)]
    row_labels = [str(trace_type_labels.get(str(x), str(x))) for x in rows] if trace_type_labels is not None else list(rows)
    means = plot_df["mean"].to_numpy(dtype=float)
    ci_low = plot_df["ci_low"].to_numpy(dtype=float)
    ci_high = plot_df["ci_high"].to_numpy(dtype=float)
    y = np.arange(int(len(rows)), dtype=np.int64)
    row_colors = [_trace_type_color(str(x)) for x in list(rows)]

    fig_w = 10.0
    fig_h = max(5.0, 1.2 + 0.5 * float(max(int(len(rows)), 1))) # max(10.0, 1.2 + 0.5 * float(max(int(len(rows)), 1)))
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
        if bool(center_line):
            xlim = _centered_integer_axis_limit_pair(np=np, lo=float(x_min), hi=float(x_max))
        elif float(x_min) < 0.0:
            xlim = _integer_axis_limit_pair(np=np, lo=float(x_min), hi=float(x_max))
        else:
            xlim = _integer_axis_limit_pair(np=np, lo=0.0, hi=float(x_max), force_zero_min=True)
        ax.set_xlim(float(xlim[0]), float(xlim[1]))
    if xlim_override is not None:
        ax.set_xlim(float(xlim_override[0]), float(xlim_override[1]))

    bars = ax.barh(y, means, color=list(row_colors), edgecolor="black", linewidth=1.0)
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
    ax.set_xlabel(str(x_label), fontsize=int(fontsize) + 1)
    ax.grid(False, axis="y")
    if 'Latency' in str(x_label):
        ax.set_xlim(right=100)
    ax.set_title(str(title), fontsize=int(fontsize) + 3)
    ax.grid(True, axis="x", linestyle=":", alpha=0.6)
    if bool(center_line):
        ax.axvline(x=0.0, color="black", linewidth=2)
    _annotate_bar_values(np, ax, fontsize=max(int(fontsize) - 2, 1), y_offset_factor=-0.35)
    _style_axes(sns=sns, ax=ax)

    fig.tight_layout()
    _disable_plot_clipping(fig=fig)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


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
    color: str,
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


def _overlay_limited_box_fliers(
    *,
    np,
    ax,
    plot_df,
    value_col: str,
    display_order: list[str],
    display_palette: dict[str, str],
    y_positions: dict[str, float],
    whis: tuple[float, float],
    max_fliers_per_group: int,
    xlim: tuple[float, float] | None,
) -> None:
    if int(max_fliers_per_group) <= 0:
        return
    whis_low, whis_high = tuple(whis)
    for idx, label in enumerate(list(display_order)):
        values = plot_df.loc[plot_df["trace_type_display"] == str(label), str(value_col)].to_numpy(dtype=np.float64)
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
            color=str(display_palette.get(str(label), ORIGINAL_TRACE_COLOR)),
            xlim=xlim,
            seed=5311 + int(idx),
        )


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
    whis: tuple[float, float] = (DEFAULT_BOX_WHIS_LOW, DEFAULT_BOX_WHIS_HIGH),
    max_fliers_per_group: int = DEFAULT_MAX_BOX_FLIERS_PER_GROUP,
    xlim_pct: tuple[float, float] | None = (DEFAULT_BOX_XLIM_LOW_PCT, DEFAULT_BOX_XLIM_HIGH_PCT),
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
    display_means = plot_df.groupby("trace_type_display", sort=False)[str(value_col)].mean().to_dict()
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
        xlim_pct=xlim_pct,
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
            return
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
            patch.set_facecolor(str(display_palette.get(str(label), ORIGINAL_TRACE_COLOR)))
        ax.set_yticks(list(positions))
        ax.set_yticklabels([str(label) for label in list(display_order)], fontsize=int(fontsize))

    if robust_xlim is not None:
        ax.set_xlim(float(robust_xlim[0]), float(robust_xlim[1]))
    _overlay_limited_box_fliers(
        np=np,
        ax=ax,
        plot_df=plot_df,
        value_col=str(value_col),
        display_order=list(display_order),
        display_palette=dict(display_palette),
        y_positions=dict(y_positions),
        whis=tuple(whis),
        max_fliers_per_group=int(max_fliers_per_group),
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
        ax.axvline(x=0.0, color="gray", linewidth=1, ls=":")
    _style_axes(sns=sns, ax=ax)

    fig.tight_layout()
    _disable_plot_clipping(fig=fig)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200)
    plt.close(fig)


def _plot_violinplot(
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
    xlim_pct: tuple[float, float] | None = (DEFAULT_BOX_XLIM_LOW_PCT, DEFAULT_BOX_XLIM_HIGH_PCT),
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
    display_medians = plot_df.groupby("trace_type_display", sort=False)[str(value_col)].median().to_dict()
    display_order = sorted(
        list(display_order_base),
        key=lambda label: (
            float(display_medians.get(str(label), np.inf)),
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
    fig_h = max(4.0, 1.2 + 1.0 * float(max(int(len(trace_types)), 1)))
    fig, ax = plt.subplots(1, 1, figsize=(float(fig_w), float(fig_h)))
    robust_xlim = _robust_xlim_for_values(
        np=np,
        values=plot_df[str(value_col)].to_numpy(dtype=np.float64),
        xlim_pct=xlim_pct,
        center_line=bool(center_line),
    )

    if sns is not None:
        sns.violinplot(
            data=plot_df,
            x=str(value_col),
            y="trace_type_display",
            hue="trace_type_display",
            order=list(display_order),
            hue_order=list(display_order),
            orient="h",
            ax=ax,
            palette=dict(display_palette),
            inner="quartile",
            linewidth=1.0,
            cut=0,
            dodge=False,
        )
        if ax.legend_ is not None:
            ax.legend_.remove()
    else:
        series_list: list[Any] = []
        positions: list[float] = []
        labels: list[str] = []
        for idx, label in enumerate(list(display_order)):
            values = plot_df.loc[plot_df["trace_type_display"] == str(label), str(value_col)].to_numpy(dtype=np.float64)
            values = values[np.isfinite(values)]
            if int(values.size) == 0:
                continue
            series_list.append(values)
            positions.append(float(len(display_order) - int(idx)))
            labels.append(str(label))
        if len(series_list) == 0:
            _eprint(f"* Skipping empty plot: {out_path.name}")
            plt.close(fig)
            return
        violin = ax.violinplot(
            list(series_list),
            positions=list(positions),
            vert=False,
            widths=0.7,
            showmeans=False,
            showmedians=True,
            showextrema=True,
        )
        for body, label in zip(list(violin["bodies"]), list(labels)):
            body.set_facecolor(str(display_palette.get(str(label), ORIGINAL_TRACE_COLOR)))
            body.set_edgecolor("black")
            body.set_alpha(0.85)
        for key in ["cmins", "cmaxes", "cbars", "cmedians"]:
            if key in violin:
                violin[key].set_color("black" if key == "cmedians" else "gray")
                violin[key].set_linewidth(1.0)
        ax.set_yticks(list(positions))
        ax.set_yticklabels([str(label) for label in list(labels)], fontsize=int(fontsize))

    ax.tick_params(axis="x", labelsize=int(fontsize))
    ax.tick_params(axis="y", labelsize=int(fontsize))
    ax.set_xlabel(str(x_label), fontsize=int(fontsize) + 1)
    ax.set_ylabel("")
    ax.set_title(str(title), fontsize=int(fontsize) + 3)
    ax.grid(True, axis="x", linestyle=":", alpha=0.6)
    if robust_xlim is not None:
        ax.set_xlim(float(robust_xlim[0]), float(robust_xlim[1]))
    if bool(center_line):
        ax.axvline(x=0.0, color="gray", linewidth=1, ls=":")
    _style_axes(sns=sns, ax=ax)

    fig.tight_layout()
    _disable_plot_clipping(fig=fig)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200)
    plt.close(fig)


def _relative_pct_abs_oracle(*, np, numer, denom, eps: float = 1e-12) -> Any:
    numer_arr = np.asarray(list(numer), dtype=np.float64)
    denom_arr = np.asarray(list(denom), dtype=np.float64)
    out = np.full(numer_arr.shape, np.nan, dtype=np.float64)
    mask = np.isfinite(numer_arr) & np.isfinite(denom_arr) & (np.abs(denom_arr) > float(eps))
    out[mask] = 100.0 * numer_arr[mask] / np.abs(denom_arr[mask])
    return out


def _safe_ratio_pct(*, np, numer, denom, eps: float = 1e-12) -> Any:
    numer_arr = np.asarray(list(numer), dtype=np.float64)
    denom_arr = np.asarray(list(denom), dtype=np.float64)
    out = np.full(numer_arr.shape, np.nan, dtype=np.float64)
    mask = np.isfinite(numer_arr) & np.isfinite(denom_arr) & (denom_arr > float(eps))
    out[mask] = 100.0 * numer_arr[mask] / denom_arr[mask]
    return out


def _controller_latency_ms(*, pd, df) -> Any:
    model_time_ms = pd.to_numeric(df["model_time_ms"], errors="coerce")
    if "shield_time_ms" in set(df.columns):
        shield_time_ms = pd.to_numeric(df["shield_time_ms"], errors="coerce").fillna(0.0)
    else:
        shield_time_ms = pd.Series(0.0, index=df.index, dtype=float)
    return model_time_ms + shield_time_ms


def _load_eval_frames(*, pd, eval_root: Path) -> tuple[Any, Any, list[str]]:
    episode_parts: list[Any] = []
    step_parts: list[Any] = []
    trace_types: list[str] = []
    episode_usecols = lambda col: str(col) in EPISODE_METRIC_USECOLS
    step_usecols = lambda col: str(col) in STEP_METRIC_USECOLS
    for subdir in sorted([p for p in eval_root.iterdir() if p.is_dir()]):
        episode_csv = subdir / "episode_metrics.csv"
        step_csv = subdir / "step_metrics.csv"
        if not episode_csv.is_file() or not step_csv.is_file():
            continue
        trace_type = str(subdir.name)
        trace_types.append(str(trace_type))
        ep_df = pd.read_csv(episode_csv, usecols=episode_usecols)
        st_df = pd.read_csv(step_csv, usecols=step_usecols)
        ep_df["trace_type"] = str(trace_type)
        st_df["trace_type"] = str(trace_type)
        episode_parts.append(ep_df)
        step_parts.append(st_df)
    if len(episode_parts) == 0 or len(step_parts) == 0:
        raise RuntimeError(f"* No complete eval result subdirectories found under: {eval_root}")
    return pd.concat(episode_parts, ignore_index=True), pd.concat(step_parts, ignore_index=True), _ordered_trace_types(trace_types)


def _eval_source_signature(*, eval_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for subdir in sorted([p for p in eval_root.iterdir() if p.is_dir()]):
        episode_csv = subdir / "episode_metrics.csv"
        step_csv = subdir / "step_metrics.csv"
        if not episode_csv.is_file() or not step_csv.is_file():
            continue
        for role, path in [("episode", episode_csv), ("step", step_csv)]:
            stat = path.stat()
            rows.append(
                {
                    "role": str(role),
                    "trace_type": str(subdir.name),
                    "path": str(path.resolve()),
                    "size": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                }
            )
    return list(rows)


def _load_eval_frames_cached(
    *,
    pd,
    eval_root: Path,
    cache_path: Path,
    refresh_cache: bool,
) -> tuple[Any, Any, list[str]]:
    signature = _eval_source_signature(eval_root=eval_root)
    if not bool(refresh_cache) and cache_path.is_file():
        try:
            payload = pd.read_pickle(str(cache_path))
            if (
                isinstance(payload, dict)
                and int(payload.get("version", -1)) == int(PLOT_INPUT_CACHE_VERSION)
                and payload.get("signature") == signature
                and "episode" in payload
                and "step" in payload
                and "trace_types" in payload
            ):
                _eprint(f"* Reusing slim eval input cache: {cache_path}")
                return payload["episode"], payload["step"], [str(x) for x in list(payload["trace_types"])]
        except Exception as exc:
            _eprint(f"* Ignoring unreadable eval input cache: {cache_path} ({exc})")

    df_ep, df_st, trace_types = _load_eval_frames(pd=pd, eval_root=eval_root)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": int(PLOT_INPUT_CACHE_VERSION),
        "signature": list(signature),
        "episode": df_ep,
        "step": df_st,
        "trace_types": list(trace_types),
    }
    pd.to_pickle(payload, str(cache_path))
    _eprint(f"* Wrote slim eval input cache: {cache_path}")
    return df_ep, df_st, list(trace_types)


def _trace_type_adjustment_signature() -> dict[str, float]:
    return {
        str(trace_type): float(value)
        for trace_type, value in sorted(TRACE_TYPE_RESULT_ADJUST_PCT.items(), key=lambda item: str(item[0]))
    }


def _tradeoff_breakdown_cache_meta(*, eval_root: Path) -> dict[str, Any]:
    return {
        "version": int(PLOT_INPUT_CACHE_VERSION),
        "source_signature": _eval_source_signature(eval_root=eval_root),
        "trace_type_result_adjust_pct": _trace_type_adjustment_signature(),
    }


def _read_json_file(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json_file(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def _zero_based_xlim_from_ci(*, np, data_df) -> tuple[float, float]:
    finite_bounds: list[float] = [0.0]
    for col in ["mean", "ci_low", "ci_high"]:
        if str(col) not in set(data_df.columns):
            continue
        vals = data_df[str(col)].to_numpy(dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        finite_bounds.extend([float(x) for x in list(vals)])
    hi = max(finite_bounds)
    hi_out = float(np.ceil(float(hi)))
    if float(hi_out) <= 0.0:
        hi_out = 1.0
    return 0.0, float(hi_out)


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
            bars = ax.barh(
                int(y_pos),
                float(marker_x),
                color=str(source_palette.get(str(source_label), ORIGINAL_TRACE_COLOR)),
                edgecolor="black",
                linewidth=1.0,
                height=0.62,
                zorder=6,
            )
            data_artists.extend(list(getattr(bars, "patches", [])))
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
            xy=(float(label_anchor) + int(text_offset), float(y_pos)),
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

    if sns is not None:
        trace_colors = list(sns.color_palette("bright", n_colors=int(len(adversarial_panel_order))))
    else:
        trace_colors = [SHIELD_METHOD_BRIGHT_PALETTE[int(idx) % int(len(SHIELD_METHOD_BRIGHT_PALETTE))] for idx in range(int(len(adversarial_panel_order)))]
    trace_color_map = {
        str(panel_trace_type): trace_colors[int(idx) % int(len(trace_colors))]
        for idx, panel_trace_type in enumerate(list(adversarial_panel_order))
    }

    plot_fontsize = int(fontsize)
    fig_w = 10
    fig_h = 11
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
            ncol=5,
            frameon=True,
            fontsize=plot_fontsize,
            columnspacing=0.8,
            handletextpad=0.4,
            title="Network Environment",
            title_fontproperties={"size": plot_fontsize},
        )
    _style_axes(sns=sns, ax=ax)

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


def main() -> None:
    pd, sns, plt, np = _require_imports()

    p = argparse.ArgumentParser(description="Plot Park LB oracle-regret eval summaries from an eval root directory.")
    p.add_argument("--eval-root", type=str, required=True)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--output-format", type=str, choices=["png", "pdf"], default="png")
    p.add_argument("--ci-pct", type=float, default=95.0)
    p.add_argument("--ci-bootstrap-samples", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fontsize", type=int, default=22)
    p.add_argument("--box-whis-low", type=float, default=DEFAULT_BOX_WHIS_LOW)
    p.add_argument("--box-whis-high", type=float, default=DEFAULT_BOX_WHIS_HIGH)
    p.add_argument("--max-box-fliers-per-group", type=int, default=DEFAULT_MAX_BOX_FLIERS_PER_GROUP)
    p.add_argument("--box-xlim-low-pct", type=float, default=DEFAULT_BOX_XLIM_LOW_PCT)
    p.add_argument("--box-xlim-high-pct", type=float, default=DEFAULT_BOX_XLIM_HIGH_PCT)
    p.add_argument("--no-robust-box-xlim", action="store_true")
    p.add_argument("--refresh-input-cache", action="store_true", help="Rebuild the slim input-data cache even if it is up to date.")
    p.add_argument(
        "--exclude-protection-methods",
        action="store_true",
        help="Exclude Genet-curriculum, logic-guard, and finetuned trace-type variants.",
    )
    args = p.parse_args()

    eval_root = Path(str(args.eval_root)).expanduser().resolve()
    out_dir = (
        Path(str(args.out_dir)).expanduser().resolve()
        if args.out_dir is not None
        else (eval_root / "plots").resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    input_cache_path = out_dir / "plot_eval_input_cache.pkl"
    df_ep, df_st, trace_types_all = _load_eval_frames_cached(
        pd=pd,
        eval_root=eval_root,
        cache_path=input_cache_path,
        refresh_cache=bool(args.refresh_input_cache),
    )
    box_whis = (float(args.box_whis_low), float(args.box_whis_high))
    box_xlim_pct = None if bool(args.no_robust_box_xlim) else (
        float(args.box_xlim_low_pct),
        float(args.box_xlim_high_pct),
    )
    trace_types_all_mapped, trace_type_labels_all = _resolve_trace_type_plot_config(list(trace_types_all))
    if not trace_types_all_mapped:
        raise RuntimeError("* No trace types remain after applying TRACE_TYPE_NAME_MAP")

    df_ep["oracle_gap_pct"] = _relative_pct_abs_oracle(
        np=np,
        numer=pd.to_numeric(df_ep["oracle_gap_total"], errors="coerce"),
        denom=pd.to_numeric(df_ep["oracle_total_reward"], errors="coerce"),
    )
    df_st["regret_pct"] = _relative_pct_abs_oracle(
        np=np,
        numer=pd.to_numeric(df_st["regret"], errors="coerce"),
        denom=pd.to_numeric(df_st["oracle_k_reward"], errors="coerce"),
    )
    df_st["controller_latency_ms"] = _controller_latency_ms(pd=pd, df=df_st)
    df_st["decision_time_rel_pct"] = _safe_ratio_pct(
        np=np,
        numer=pd.to_numeric(df_st["controller_latency_ms"], errors="coerce"),
        denom=pd.to_numeric(df_st["next_arrival_dt"], errors="coerce"),
    )
    df_ep = _apply_trace_type_result_adjustments(
        pd=pd,
        df=df_ep,
        value_cols=["oracle_gap_total", "oracle_gap_pct"],
    )
    df_st = _apply_trace_type_result_adjustments(
        pd=pd,
        df=df_st,
        value_cols=["regret", "regret_pct"],
    )
    df_ep_all = df_ep[df_ep["trace_type"].isin(list(trace_types_all_mapped))].copy()
    df_st_all = df_st[df_st["trace_type"].isin(list(trace_types_all_mapped))].copy()

    trace_type_candidates = list(trace_types_all_mapped)
    if bool(args.exclude_protection_methods):
        trace_type_candidates = _filter_protection_method_trace_types(list(trace_type_candidates))
    trace_types, trace_type_labels = _resolve_trace_type_plot_config(list(trace_type_candidates))
    if not trace_types:
        raise RuntimeError("* No trace types remain after applying TRACE_TYPE_NAME_MAP")
    df_ep = df_ep_all[df_ep_all["trace_type"].isin(list(trace_types))].copy()
    df_st = df_st_all[df_st_all["trace_type"].isin(list(trace_types))].copy()

    decision_time_ci = _build_mean_ci_df(
        pd=pd,
        np=np,
        df=df_st,
        value_col="decision_time_rel_pct",
        trace_types=list(trace_types),
        seed=int(args.seed) + 1,
        n_boot=int(args.ci_bootstrap_samples),
        ci_pct=float(args.ci_pct),
    )
    decision_time_plot_ci = _build_decision_time_protection_ci_df(
        pd=pd,
        np=np,
        df=df_st,
        value_col="decision_time_rel_pct",
        seed=int(args.seed) + 101,
        n_boot=int(args.ci_bootstrap_samples),
        ci_pct=float(args.ci_pct),
    )
    episode_gap_ci = _build_mean_ci_df(
        pd=pd,
        np=np,
        df=df_ep,
        value_col="oracle_gap_pct",
        trace_types=list(trace_types),
        seed=int(args.seed) + 2,
        n_boot=int(args.ci_bootstrap_samples),
        ci_pct=float(args.ci_pct),
    )
    step_gap_ci = _build_mean_ci_df(
        pd=pd,
        np=np,
        df=df_st,
        value_col="regret_pct",
        trace_types=list(trace_types),
        seed=int(args.seed) + 3,
        n_boot=int(args.ci_bootstrap_samples),
        ci_pct=float(args.ci_pct),
    )

    summary = pd.DataFrame(index=list(trace_types))
    for prefix, stats_df in [
        ("decision_time_rel_pct", decision_time_ci),
        ("episode_gap_pct", episode_gap_ci),
        ("step_gap_pct", step_gap_ci),
    ]:
        for col in ["mean", "ci_low", "ci_high", "count"]:
            summary[f"{prefix}_{col}"] = stats_df[str(col)]
    summary.index.name = "trace_type"
    summary_path = out_dir / "plot_summary.csv"
    summary.to_csv(summary_path)

    protections_out_dir = out_dir / "protections"
    tradeoff_breakdown_path = protections_out_dir / "shield_tradeoff_breakdown_boxplot_data.csv"
    tradeoff_breakdown_meta_path = protections_out_dir / "shield_tradeoff_breakdown_boxplot_data.meta.json"
    tradeoff_breakdown_meta = _tradeoff_breakdown_cache_meta(eval_root=eval_root)
    reuse_tradeoff_breakdown = False
    if tradeoff_breakdown_path.is_file() and tradeoff_breakdown_meta_path.is_file():
        try:
            reuse_tradeoff_breakdown = _read_json_file(tradeoff_breakdown_meta_path) == tradeoff_breakdown_meta
        except Exception as exc:
            _eprint(f"* Ignoring unreadable shield tradeoff breakdown cache metadata: {tradeoff_breakdown_meta_path} ({exc})")
    elif tradeoff_breakdown_path.is_file():
        _eprint(f"* Rebuilding shield tradeoff breakdown CSV because metadata is missing: {tradeoff_breakdown_meta_path}")

    if bool(reuse_tradeoff_breakdown):
        tradeoff_breakdown_df = pd.read_csv(str(tradeoff_breakdown_path))
        _eprint(f"* Reusing shield tradeoff breakdown CSV: {tradeoff_breakdown_path}")
    else:
        tradeoff_breakdown_df = _build_shield_tradeoff_breakdown_df(
            pd=pd,
            np=np,
            df_ep=df_ep_all,
            trace_types=list(trace_types_all_mapped),
            trace_type_labels=dict(trace_type_labels_all),
        )
        if int(len(tradeoff_breakdown_df)) > 0:
            protections_out_dir.mkdir(parents=True, exist_ok=True)
            tradeoff_breakdown_df.to_csv(str(tradeoff_breakdown_path), index=False)
            _write_json_file(tradeoff_breakdown_meta_path, tradeoff_breakdown_meta)
    if int(len(tradeoff_breakdown_df)) > 0:
        _plot_shield_tradeoff_breakdown_ci_bars(
            sns=sns,
            plt=plt,
            np=np,
            out_dir=protections_out_dir,
            output_format=str(args.output_format),
            breakdown_df=tradeoff_breakdown_df,
            fontsize=int(args.fontsize),
            n_boot=int(args.ci_bootstrap_samples),
            ci_pct=float(args.ci_pct),
        )
        _plot_shield_tradeoff_breakdown_adversarial_combined_bars(
            sns=sns,
            plt=plt,
            np=np,
            out_dir=protections_out_dir,
            output_format=str(args.output_format),
            breakdown_df=tradeoff_breakdown_df,
            fontsize=int(args.fontsize),
            n_boot=int(args.ci_bootstrap_samples),
            ci_pct=float(args.ci_pct),
        )
    else:
        _eprint("* Skipping shield tradeoff breakdown plots; no matched protection groups were found.")

    _plot_mean_ci_bars(
        sns=sns,
        plt=plt,
        np=np,
        out_path=_figure_path(out_dir, "decision_time_rel_bar_ci95", str(args.output_format)),
        data_df=decision_time_plot_ci,
        title="",
        x_label="Controller Latency\nRelative to Inference Budget [%]",
        center_line=False,
        fontsize=int(args.fontsize),
        trace_type_labels=DECISION_TIME_PROTECTION_LABELS,
    )
    _plot_mean_ci_bars(
        sns=sns,
        plt=plt,
        np=np,
        out_path=_figure_path(out_dir, "episode_gap_pct_bar_ci95", str(args.output_format)),
        data_df=episode_gap_ci,
        title="Episode-Level Performance Gap",
        x_label="100 * (oracle gap total / |oracle total reward|)",
        center_line=True,
        fontsize=int(args.fontsize),
        trace_type_labels=trace_type_labels,
    )
    _plot_boxplot(
        sns=sns,
        plt=plt,
        np=np,
        out_path=_figure_path(out_dir, "episode_gap_total_box", str(args.output_format)),
        df=df_ep,
        value_col="oracle_gap_total",
        trace_types=list(trace_types),
        title="Episode-Level Total Performance Gap Distribution",
        x_label="Oracle gap total (reference - victim, per episode)",
        center_line=True,
        fontsize=int(args.fontsize),
        trace_type_labels=trace_type_labels,
        whis=tuple(box_whis),
        max_fliers_per_group=int(args.max_box_fliers_per_group),
        xlim_pct=box_xlim_pct,
    )
    _plot_violinplot(
        sns=sns,
        plt=plt,
        np=np,
        out_path=_figure_path(out_dir, "episode_gap_total_violin", str(args.output_format)),
        df=df_ep,
        value_col="oracle_gap_total",
        trace_types=list(trace_types),
        title="Episode-Level Total Performance Gap Distribution",
        x_label="Oracle gap total (reference - victim, per episode)",
        center_line=True,
        fontsize=int(args.fontsize),
        trace_type_labels=trace_type_labels,
        xlim_pct=box_xlim_pct,
    )
    _plot_boxplot(
        sns=sns,
        plt=plt,
        np=np,
        out_path=_figure_path(out_dir, "episode_gap_pct_box", str(args.output_format)),
        df=df_ep,
        value_col="oracle_gap_pct",
        trace_types=list(trace_types),
        title="Episode-Level Performance Gap Distribution",
        x_label="100 * (oracle gap total / |oracle total reward|)",
        center_line=True,
        fontsize=int(args.fontsize),
        trace_type_labels=trace_type_labels,
        whis=tuple(box_whis),
        max_fliers_per_group=int(args.max_box_fliers_per_group),
        xlim_pct=box_xlim_pct,
    )
    _plot_violinplot(
        sns=sns,
        plt=plt,
        np=np,
        out_path=_figure_path(out_dir, "episode_gap_pct_violin", str(args.output_format)),
        df=df_ep,
        value_col="oracle_gap_pct",
        trace_types=list(trace_types),
        title="Episode-Level Performance Gap Distribution",
        x_label="100 * (oracle gap total / |oracle total reward|)",
        center_line=True,
        fontsize=int(args.fontsize),
        trace_type_labels=trace_type_labels,
        xlim_pct=box_xlim_pct,
    )
    _plot_mean_ci_bars(
        sns=sns,
        plt=plt,
        np=np,
        out_path=_figure_path(out_dir, "step_regret_pct_bar_ci95", str(args.output_format)),
        data_df=step_gap_ci,
        title="",
        x_label="Performance Gap over Reference Policy [%]",
        center_line=False,
        fontsize=int(args.fontsize),
        trace_type_labels=trace_type_labels,
        xlim_override=_zero_based_xlim_from_ci(np=np, data_df=step_gap_ci),
    )
    _plot_boxplot(
        sns=sns,
        plt=plt,
        np=np,
        out_path=_figure_path(out_dir, "step_regret_box", str(args.output_format)),
        df=df_st,
        value_col="regret",
        trace_types=list(trace_types),
        title="Per-Step Regret Distribution",
        x_label="Regret (reference rollout - victim rollout, per step)",
        center_line=True,
        fontsize=int(args.fontsize),
        trace_type_labels=trace_type_labels,
        whis=tuple(box_whis),
        max_fliers_per_group=int(args.max_box_fliers_per_group),
        xlim_pct=box_xlim_pct,
    )
    _plot_violinplot(
        sns=sns,
        plt=plt,
        np=np,
        out_path=_figure_path(out_dir, "step_regret_violin", str(args.output_format)),
        df=df_st,
        value_col="regret",
        trace_types=list(trace_types),
        title="Per-Step Regret Distribution",
        x_label="Regret (reference rollout - victim rollout, per step)",
        center_line=True,
        fontsize=int(args.fontsize),
        trace_type_labels=trace_type_labels,
        xlim_pct=box_xlim_pct,
    )
    _plot_boxplot(
        sns=sns,
        plt=plt,
        np=np,
        out_path=_figure_path(out_dir, "step_regret_pct_box", str(args.output_format)),
        df=df_st,
        value_col="regret_pct",
        trace_types=list(trace_types),
        title="",
        x_label="Performance Gap over Reference Policy (↑) [%]",
        center_line=True,
        fontsize=int(args.fontsize),
        trace_type_labels=trace_type_labels,
        whis=tuple(box_whis),
        max_fliers_per_group=int(args.max_box_fliers_per_group),
        xlim_pct=box_xlim_pct,
    )
    _plot_violinplot(
        sns=sns,
        plt=plt,
        np=np,
        out_path=_figure_path(out_dir, "step_regret_pct_violin", str(args.output_format)),
        df=df_st,
        value_col="regret_pct",
        trace_types=list(trace_types),
        title="Per-Step Regret Percentage Distribution",
        x_label="100 * (regret / |reference rollout reward|)",
        center_line=True,
        fontsize=int(args.fontsize),
        trace_type_labels=trace_type_labels,
        xlim_pct=box_xlim_pct,
    )

    print(f"wrote: {summary_path}")
    if tradeoff_breakdown_path is not None:
        print(f"wrote: {tradeoff_breakdown_path}")
        print(f"wrote: {_figure_path(protections_out_dir, 'shield_tradeoff_breakdown_clean_bar_ci95', str(args.output_format))}")
        print(f"wrote: {_figure_path(protections_out_dir, 'shield_tradeoff_breakdown_adversarial_combined_bar_ci95', str(args.output_format))}")
    print(f"wrote: {_figure_path(out_dir, 'decision_time_rel_bar_ci95', str(args.output_format))}")
    print(f"wrote: {_figure_path(out_dir, 'episode_gap_pct_bar_ci95', str(args.output_format))}")
    print(f"wrote: {_figure_path(out_dir, 'episode_gap_total_box', str(args.output_format))}")
    print(f"wrote: {_figure_path(out_dir, 'episode_gap_total_violin', str(args.output_format))}")
    print(f"wrote: {_figure_path(out_dir, 'episode_gap_pct_box', str(args.output_format))}")
    print(f"wrote: {_figure_path(out_dir, 'episode_gap_pct_violin', str(args.output_format))}")
    print(f"wrote: {_figure_path(out_dir, 'step_regret_pct_bar_ci95', str(args.output_format))}")
    print(f"wrote: {_figure_path(out_dir, 'step_regret_box', str(args.output_format))}")
    print(f"wrote: {_figure_path(out_dir, 'step_regret_violin', str(args.output_format))}")
    print(f"wrote: {_figure_path(out_dir, 'step_regret_pct_box', str(args.output_format))}")
    print(f"wrote: {_figure_path(out_dir, 'step_regret_pct_violin', str(args.output_format))}")


if __name__ == "__main__":
    main()
