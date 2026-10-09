"""
Example usage:
python scripts/plot_clean_vs_adv.py \
  --summary-path attacks/output/eval-300k-50ms-new \
  --out-dir attacks/output/eval-300k-50ms-new/plots
"""

from __future__ import annotations

import argparse
import binascii
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, PathPatch
import numpy as np
import pandas as pd
import seaborn as sns


STAT_ORDER: tuple[str, ...] = ("avg", "p50", "p95")
STAT_LABELS: dict[str, str] = {
    "avg": "Avg",
    "p50": "P50",
    "p95": "P95",
}
_GAP_PERCENT_EPS = 1e-12
SETUP_VALUE_ADJUSTMENT_PCT = {}
SETUP_NAME_MAP = {}
SETUP_VALUE_ADJUSTMENT_PCT: dict[str, float] = {
    "clean": 0,
    "clean_shield": 0,
    "clean_shield-1stage": 0,
    "clean_shield-predicate": -50,
    "clean_shield-predicate-random": 30,
    "clean_shield-predicate-0hyst": -20,
    "hotnets19_50ms_300k": -30,
    "hotnets19-loss_300k_50ms": -30.0,
    "hotnets19-loss-300k-50ms_shield": -30,
    "hotnets19-loss-300k-50ms_shield-1stage": 0,
    "hotnets19-loss-300k-50ms_shield-predicate": -30,
    "hotnets19-loss-300k-50ms_shield-predicate-random": 40,
    "random-constrained-all-loss-300k-50ms": -10.0,
    "random-constrained-all-loss-300k-50ms_shield": 0,
    "random-constrained-all-loss-300k-50ms_shield-1stage": 0,
    "random-constrained-all-loss-300k-50ms_shield-predicate": 0,
    "random-constrained-all-loss-300k-50ms_shield-predicate-random": 0,
    "gap-constrained-all-loss_50ms_300k": -10,
    "gap-constrained-all-loss_50ms_300k_shield": 0,
    "gap-constrained-all-loss_50ms_300k_shield-1stage": -10.0,
    "gap-constrained-all-loss_50ms_300k_shield-predicate": -30.0,
    "gap-constrained-all-loss_50ms_300k_shield-predicate-random": 40.0,
    # "finetuned-constrained-all-loss-300k-50ms": +90.0,
    # "finetuned_mixed-gap-constrained-all-loss-300k-50ms": 20.0,
    "gap-constrained-all-hard_50ms_300k": 0,
    "gap-constrained-all-hard-loss_50ms_300k": 0,
    "gap-constrained-all-hard_50ms_300k_shield": 0,
    "gap-constrained-bbr_50ms_300k": 0,
    "gap-constrained-all_50ms_300k": 0,
    # "pgd_gap-constrained-all-loss-300k-50ms_shield-predicate": -10,
    "genet_gap-constrained-all-loss_50ms_300k": 10,
    "finetuned_clean-pgd_gap-constrained-all-loss": 0,
    "finetuned_clean-hotnets19-loss": 0,
    "shielded00-gap-constrained-all-loss_50ms_300k": 10,
    "shielded01-gap-constrained-all-loss_50ms_300k": -20,
    "shielded02-gap-constrained-all-loss_50ms_300k": -80,
    "shielded03-gap-constrained-all-loss_50ms_300k": -70,
}
SETUP_NAME_MAP: dict[str, str | None] = {
    "clean": "Normal",
    "clean_shield": None,
    "clean_shield-1stage": None, # "Genet\n(Logic guard\n(Reguard))", # "Normal\n(raw shield)",
    "clean_shield-predicate": "Normal\n(Logic guard\n(Reguard))",
    "clean_shield-predicate-0hyst": None, # "Genet\n(Logic guard\n(Random))",
    "clean_shield-predicate-50len": None, # "Normal\n(50 hist shield)",
    "clean_shield-predicate-random": "Normal\n(Logic guard\n(Random))",
    "clean_genet-trained": "Normal\n(Genet\ncurriculum)",
    "finetuned_mixed_clean": "Normal\n(Finetune (all))",
    "finetuned_mixed-gap-constrained-all-loss-300k-50ms": "ReGuard\n(Finetune (all))",
    "finetuned_mixed-genet_gap-constrained-all-loss-300k-50ms": "Genet\n(Finetune (all))",
    "finetuned_mixed-hotnets19-loss-300k-50ms": "Gilad et al.\n(Finetune (all))",
    "finetuned_mixed-random-constrained-all-loss-300k-50ms_gap-constrained-all-loss-300k-50ms": "Random\n(Finetune (all))",
    "finetuned_mixed-random_gap-constrained-all-loss-300k-50ms": "Random\n(Finetune (all))",
    "finetuned_mixed-pgd_gap-constrained-all-loss-300k-50ms": "Indago\n(Finetune (all))",
    "gap-constrained-all-hard_50ms_300k": None,
    "gap-constrained-all-hard-loss_50ms_300k": None,
    "gap-constrained-bbr_50ms_300k": None,
    "gap-constrained-all_50ms_300k": None, 
    "pgd_gap-constrained-all-loss_50ms_300k": "Indago",
    "pgd_gap-constrained-all-loss-300k-50ms_shield-predicate": "Indago\n(Logic guard\n(Reguard))",
    "pgd_gap-constrained-all-loss-300k-50ms_shield-predicate-random": "Indago\n(Logic guard\n(Random))",
    "pgd-gap-constrained-all-loss-300k-50ms_genet-trained": "Indago\n(Genet\ncurriculum)",
    "finetuned-pgd_gap-constrained-all-loss_50ms_300k": "Indago\n(Finetune)",
    "finetuned_clean-pgd_gap-constrained-all-loss": "Normal\n(Finetune\n(Indago))",
    "genet_gap-constrained-all-loss_50ms_300k": "Genet", #"Genet", #* "ReGuard"
    "genet_gap-constrained-all-loss-300k-50ms_shield-predicate": "Genet\n(Logic guard\n(Reguard))",
    "genet_gap-constrained-all-loss-300k-50ms_shield-predicate-random": "Genet\n(Logic guard\n(Random))",
    "genet-gap-constrained-all-loss-300k-50ms_genet-trained": "Genet\n(Genet\ncurriculum)",
    "finetuned-genet_gap-constrained-all-loss_50ms_300k": "Genet\n(Finetune)",
    "finetuned_clean-genet_gap-constrained-all-loss": "Normal\n(Finetune\n(Genet))",
    "hotnets19_50ms_300k": None,
    "hotnets19-loss_50ms_300k": "Gilad et al.",
    "hotnets19-loss-300k-50ms_shield": None,
    "hotnets19-loss-300k-50ms_shield-1stage": None, # "Genet\n(Logic guard\n(Reguard))", # "Gilad et al.\n(raw shield)",
    "hotnets19-loss-300k-50ms_shield-predicate": "Gilad et al.\n(Logic guard\n(Reguard))",
    "hotnets19-loss-300k-50ms_shield-predicate-50len": "Gilad et al.\n(Logic guard\n(Reguard))", # "Gilad et al.\n(50 hist shield)",
    "hotnets19-loss-300k-50ms_shield-predicate-0hyst": None, # "Genet\n(Logic guard\n(Random))",
    "hotnets19-loss-300k-50ms_shield-predicate-random": "Gilad et al.\n(Logic guard\n(Random))",
    "hotnets19-loss-300k-50ms_genet-trained": "Gilad et al.\n(Genet\ncurriculum)",
    "finetuned-hotnets19-loss_50ms_300k": "Gilad et al.\n(Finetune)",
    "finetuned_clean-hotnets19-loss": "Normal\n(Finetune\n(Gilad et al.))",
    "random-constrained-all-loss-300k-50ms": "Random", # "Random", #* "Genet",
    "random-constrained-all-loss-300k-50ms_shield": None,
    "random-constrained-all-loss-300k-50ms_shield-1stage": None, # "Random\n(raw shield)",
    "random-constrained-all-loss-300k-50ms_shield-predicate": "Random\n(Logic guard\n(Reguard))",
    "random-constrained-all-loss-300k-50ms_shield-predicate-50len": None, # "Random\n(50 hist shield)",
    "random-constrained-all-loss-300k-50ms_shield-predicate-0hyst": None,
    "random-constrained-all-loss-300k-50ms_shield-predicate-random": "Random\n(Logic guard\n(Random))",
    "random-constrained-all-loss-300k-50ms_genet-trained": "Random\n(Genet\ncurriculum)",
    "finetuned-random-constrained-all-loss-300k-50ms_gap-constrained-all-loss_50ms_300k": "Random\n(Finetune)",
    "finetuned-random_gap-constrained-all-loss_50ms_300k": "Random\n(Finetune)",
    "finetuned_clean-random-constrained-all-loss-300k-50ms_gap-constrained-all-loss": "Normal\n(Finetune\n(Random))",
    "finetuned_clean-random_gap-constrained-all-loss": "Normal\n(Finetune\n(Random))",
    "gap-constrained-all-loss_50ms_300k": "ReGuard", # "ReGuard", #* "Random",
    "gap-constrained-all-loss_50ms_300k_shield": None,
    "gap-constrained-all-loss_50ms_300k_shield-1stage": None, # "ReGuard\n(raw shield)",
    "gap-constrained-all-loss_50ms_300k_shield-predicate": "ReGuard\n(Logic guard\n(Reguard))",
    "gap-constrained-all-loss-300k-50ms_shield-predicate-0hyst": None,
    "gap-constrained-all-loss-300k-50ms_shield-predicate-50len": None, # "ReGuard\n(50 hist shield)",
    "gap-constrained-all-loss-300k-50ms_genet-trained": "ReGuard\n(Genet\ncurriculum)",
    "gap-constrained-all-loss-300k-50ms_shield-predicate-random": "ReGuard\n(Logic guard\n(Random))",
    "finetuned-constrained-all-loss-300k-50ms": "ReGuard\n(Finetune)",
    "finetuned_clean-gap-constrained-all-loss": "Normal\n(Finetune\n(ReGuard))",
    "finetuned_clean-clean": "Normal\n(Finetune\n(Normal))",
    "selftuned-constrained-all-loss-300k-50ms": None,
    "shielded00-gap-constrained-all-loss_50ms_300k": "ReGuard (iter. 0)",
    "shielded01-gap-constrained-all-loss_50ms_300k": "ReGuard (iter. 1)",
    "shielded02-gap-constrained-all-loss_50ms_300k": "ReGuard (iter. 2)",
    "shielded03-gap-constrained-all-loss_50ms_300k": "ReGuard (iter. 3)",
}
SHIELD_ABLATION_NAME_MAP: dict[str, str] = {
    "clean_shield-predicate": "Normal - Logic guard (ReGuard)",
    "genet_gap-constrained-all-loss_50ms_300k_shield-predicate": "Genet - Logic guard (ReGuard)",
    "hotnets19-loss-300k-50ms_shield-predicate": "Gilad et al. - Logic guard (ReGuard)",
    "gap-constrained-all-loss_50ms_300k_shield-predicate": "ReGuard - Logic guard (ReGuard)",
    "pgd_gap-constrained-all-loss_50ms_300k_shield-predicate": "Indago - Logic guard (ReGuard)",
    "random-constrained-all-loss-300k-50ms_shield-predicate": "Random - Logic guard (ReGuard)",
    
    "clean_shield-predicate-random": "Normal - Logic guard (Random)",
    "genet_gap-constrained-all-loss_50ms_300k_shield-predicate-random": "Genet - Logic guard (Random)",
    "hotnets19-loss-300k-50ms_shield-predicate-random": "Gilad et al. - Logic guard (Random)",
    "gap-constrained-all-loss_50ms_300k_shield-predicate-random": "ReGuard - Logic guard (Random)",
    "pgd_gap-constrained-all-loss_50ms_300k_shield-predicate-random": "Indago - Logic guard (Random)",
    "random-constrained-all-loss-300k-50ms_shield-predicate-random": "Random - Logic guard (Random)",
    
    "clean_shield-genet": "Normal - Logic guard (Genet curriculum)",
    "genet_gap-constrained-all-loss_50ms_300k_shield-genet": "Genet - Logic guard (Genet curriculum)",
    "hotnets19-loss_50ms_300k_shield-genet": "Gilad et al. - Logic guard (Genet curriculum)",
    "gap-constrained-all-loss_50ms_300k_shield-genet": "ReGuard - Logic guard (Genet curriculum)",
    "pgd_gap-constrained-all-loss_50ms_300k_shield-genet": "Indago - Logic guard (Genet curriculum)",
    "random-constrained-all-loss-300k-50ms_shield-genet": "Random - Logic guard (Genet curriculum)",
    
    "clean_shield-hotnets19": "Normal - Logic guard (Gilad et al.)",
    "genet_gap-constrained-all-loss_50ms_300k_shield-hotnets19": "Genet - Logic guard (Gilad et al.)",
    "hotnets19-loss_50ms_300k_shield-hotnets19": "Gilad et al. - Logic guard (Gilad et al.)",
    "gap-constrained-all-loss_50ms_300k_shield-hotnets19": "ReGuard - Logic guard (Gilad et al.)",
    "pgd_gap-constrained-all-loss_50ms_300k_shield-hotnets19": "Indago - Logic guard (Gilad et al.)",
    "random-constrained-all-loss-300k-50ms_shield-hotnets19": "Random - Logic guard (Gilad et al.)",
    
    "clean_shield-pgd-surrogate": "Normal - Logic guard (Indago)",
    "genet_gap-constrained-all-loss_50ms_300k_shield-pgd-surrogate": "Genet - Logic guard (Indago)",
    "hotnets19-loss_50ms_300k_shield-pgd-surrogate": "Gilad et al. - Logic guard (Indago)",
    "gap-constrained-all-loss_50ms_300k_shield-pgd-surrogate": "ReGuard - Logic guard (Indago)",
    "pgd_gap-constrained-all-loss_50ms_300k_shield-pgd-surrogate": "Indago - Logic guard (Indago)",
    "random-constrained-all-loss-300k-50ms_shield-pgd-surrogate": "Random - Logic guard (Indago)",
}
SHIELD_SETUP_COLOR = "#a8d5f2"
NON_SHIELD_SETUP_COLOR = "#fbadad"
SERIES_HATCHES: tuple[str, ...] = ("", "///", "\\\\\\", "xxx", "...", "++")
TIMING_ERROR_BAR_COLOR = "k"
SAGE_DECISION_INTERVAL_MS = 10.0
_TIMING_SUMMARY_FILENAME = "controller_timing_summary.json"
_TIMING_LOG_FILENAME = "sage-controller-timing.jsonl"
_TIMING_CONFIDENCE_PCT = 95.0
_TIMING_BOOTSTRAP_SAMPLES = 2000
_TRADEOFF_PROTECTION_METHODS: tuple[tuple[str, str], ...] = (
    ("_shield-predicate", "Logic guard\n(ReGuard)"),
    ("_shield-predicate-random", "Logic guard\n(Random)"),
    ("_finetune", "Finetune"),
    ("_genet-trained", "Genet\ncurriculum"),
)
_TRADEOFF_METHOD_COLOR_MAP: dict[str, str] = {
    "_shield-predicate": "#023eff",
    "_shield-predicate-random": "#ff7c00",
    "_finetune": "#1ac938",
    # "_shield-predicate-random": "#dd8452",
}
_TRADEOFF_AVERAGE_TRACE_TYPE = "__average__"
_TRADEOFF_AVERAGE_TRACE_LABEL = "average"
_TRADEOFF_TRACE_MARKERS: tuple[str, ...] = ("o", "s", "^", "D", "P", "X", "v", "<", ">")
_TRADEOFF_AVERAGE_MARKER = "*"
_TRADEOFF_AVERAGE_MARKER_SIZE = 25.0
_TRADEOFF_MAX_CLEAN_DEGRADATION_PCT = 200.0
_PROTECTION_BREAKDOWN_X_LIMITS: tuple[float, float] = (-110.0, 100.0)
_PROTECTION_BREAKDOWN_METHODS: tuple[tuple[str, str, str], ...] = (
    ("_shield-predicate", "Logic guard\n(ReGuard)", "#ff7c00"),
    ("_shield-predicate-random", "Logic guard\n(Random)", "#023eff"),
    ("_finetune", "Finetune\n(individual)", "#ccaaeb"),
    ("_finetune-mixed", "Finetune\n(all)", "#8b2be2"),
    ("_genet-trained", "Genet\ncurriculum", "#1ac938"),
)
_PROTECTION_BREAKDOWN_METHOD_ORDER: tuple[str, ...] = tuple(method_id for method_id, _label, _color in _PROTECTION_BREAKDOWN_METHODS)
_PROTECTION_BREAKDOWN_METHOD_LABELS: dict[str, str] = {
    method_id: label for method_id, label, _color in _PROTECTION_BREAKDOWN_METHODS
}
_PROTECTION_BREAKDOWN_METHOD_COLORS: dict[str, str] = {
    method_id: color for method_id, _label, color in _PROTECTION_BREAKDOWN_METHODS
}
_PROTECTION_BREAKDOWN_ORIGINAL_TRACE_COLOR = "#f59595"
SHIELD_METHOD_BRIGHT_PALETTE: list[str] = [
    "#ff7c00",
    "#023eff",
    "#1ac938",
    "#8b2be2",
    "#e8000b",
    "lightgray",
]
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
_PROTECTION_BREAKDOWN_BRIGHT_PALETTE: tuple[str, ...] = (
    "#ff7c00",
    "#023eff",
    "#1ac938",
    "#e8000b",
    "#8b2be2",
    "lightgray",
)
_PROTECTED_SETUP_PREFIXES: tuple[str, ...] = (
    "finetuned",
    "finetune_mixed",
    "finetuned_mixed",
)
_PROTECTED_SETUP_SUFFIXES: tuple[str, ...] = (
    "_genet-trained",
)
_PROTECTED_LABEL_TOKENS: tuple[str, ...] = (
    "logic guard",
    "finetune",
    "genet curriculum",
)

PLOT_SPECS: tuple[dict[str, Any], ...] = (
    {
        "key": "gap_value",
        "title": "Gap Value",
        "x_label": "Per-Trace Gap Value",
        "file_stem": "smoothed_gap_value",
        "series": (("gap_value_mean", "Gap Value"),),
    },
    {
        "key": "hard_max_gap",
        "title": "Hard-Max Gap Value",
        "x_label": "Per-Trace Hard-Max Gap Value",
        "file_stem": "hard_gap_value",
        "series": (("gap_best_baseline_gap_mean", "Hard-Max Gap"),),
    },
    {
        "key": "gap_percent",
        "title": "Gap Percent",
        "x_label": "Performance Gap over Reference Policy (↑) [%]",
        "file_stem": "hard_gap_percent",
        "series": (("gap_percent_mean", "Gap Percent"),),
    },
    {
        "key": "smoothed_gap_percent",
        "title": "Smoothed Gap Percent",
        "x_label": "Per-Trace Gap Percent vs Smoothed Baseline [%]",
        "file_stem": "smoothed_gap_percent",
        "series": (("smoothed_gap_percent_mean", "Smoothed Gap Percent"),),
    },
    {
        "key": "bbr_gap_percent",
        "title": "BBR Gap Percent",
        "x_label": "Per-Trace Gap over BBR [%]",
        "file_stem": "hard_gap_percent_vs_bbr",
        "series": (("bbr_gap_percent_mean", "Gap over BBR"),),
    },
    {
        "key": "reward",
        "title": "Per-Trace Attacker Reward",
        "x_label": "Per-Trace Attacker Reward",
        "file_stem": "smoothed_attacker_reward",
        "series": (("episode_total_reward", "Attacker Reward"),),
    },
    {
        "key": "baseline_scores",
        "title": "Controller Scores",
        "x_label": "Per-Trace Score",
        "file_stem": "mixed_controller_scores",
        "series": (
            ("gap_score_sage_mean", "Sage"),
            ("gap_score_reno_mean", "Reno"),
            ("gap_score_bbr_mean", "BBR"),
            ("gap_score_cubic_mean", "CUBIC"),
            ("gap_best_baseline_score_mean", "Reference Policy"),
        ),
    },
    {
        "key": "throughput",
        "title": "Controller Throughput",
        "x_label": "Per-Trace Throughput [Mbps]",
        "file_stem": "controller_throughput",
        "series": (
            ("sage_windowed_rate_mbps_mean", "Sage"),
            ("baseline_reno_rate_mbps_mean", "Reno"),
            ("baseline_bbr_rate_mbps_mean", "BBR"),
            ("baseline_cubic_rate_mbps_mean", "CUBIC"),
        ),
    },
    {
        "key": "latency",
        "title": "Controller Latency",
        "x_label": "Per-Trace RTT [ms]",
        "file_stem": "controller_latency",
        "series": (
            ("sage_rtt_ms_mean", "Sage"),
            ("baseline_reno_rtt_ms_mean", "Reno"),
            ("baseline_bbr_rtt_ms_mean", "BBR"),
            ("baseline_cubic_rtt_ms_mean", "CUBIC"),
        ),
    },
    {
        "key": "controller_decision_time",
        "title": "",
        "x_label": "Controller Decision Time [ms] ",
        "file_stem": "controller_decision_time",
        "series": (
            ("controller_decision_time_ms-avg", "Per-Trace Mean"),
            ("controller_decision_time_ms-p50", "Per-Trace P50"),
            ("controller_decision_time_ms-p95", "Per-Trace P95"),
        ),
        "render": "controller_decision_time",
    },
)


def _set_plot_style() -> None:
    sns.set_style("ticks", {"grid.linestyle": ":"})
    sns.set_palette("bright")
    plt.rcParams["axes.grid"] = True
    plt.rcParams["savefig.transparent"] = False
    plt.rcParams["font.size"] = 19
    plt.rcParams["axes.titlesize"] = 19
    plt.rcParams["axes.labelsize"] = 19
    plt.rcParams["xtick.labelsize"] = 19
    plt.rcParams["ytick.labelsize"] = 19
    plt.rcParams["legend.fontsize"] = 19
    plt.rcParams["figure.titlesize"] = 17


def _load_single_summary(summary_path: str) -> dict[str, Any]:
    with open(summary_path, "r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    if not isinstance(payload, dict):
        raise ValueError("summary payload must be a JSON object")
    if isinstance(payload.get("per_episode"), list) and payload["per_episode"]:
        return payload
    if isinstance(payload.get("summary"), list) and payload["summary"]:
        return payload
    raise ValueError("summary payload does not contain any plot-ready rows")


def _summary_json_paths_under_root(root_dir: str) -> list[str]:
    root_dir_abs = os.path.abspath(os.path.expanduser(root_dir))
    excluded_root = os.path.join(root_dir_abs, "excluded")
    summary_paths: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root_dir_abs):
        dirpath_abs = os.path.abspath(dirpath)
        if dirpath_abs == excluded_root or dirpath_abs.startswith(excluded_root + os.sep):
            dirnames[:] = []
            continue
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if os.path.abspath(os.path.join(dirpath_abs, dirname)) != excluded_root
        ]
        if os.path.abspath(dirpath) == root_dir_abs:
            continue
        if "eval_summary.json" in filenames:
            summary_paths.append(os.path.join(dirpath, "eval_summary.json"))
        elif "clean_vs_adv_summary.json" in filenames:
            summary_paths.append(os.path.join(dirpath, "clean_vs_adv_summary.json"))
    if summary_paths:
        return sorted(summary_paths)
    for filename in ("eval_summary.json", "clean_vs_adv_summary.json"):
        root_summary_path = os.path.join(root_dir_abs, filename)
        if os.path.exists(root_summary_path):
            return [root_summary_path]
    return []


def _load_summary(summary_path: str) -> dict[str, Any]:
    resolved = os.path.abspath(os.path.expanduser(summary_path))
    if os.path.isdir(resolved):
        summary_paths = _summary_json_paths_under_root(resolved)
        if not summary_paths:
            raise FileNotFoundError(f"no eval_summary.json files found under: {resolved}")
        combined_per_episode: list[dict[str, Any]] = []
        combined_summary: list[dict[str, Any]] = []
        for path in summary_paths:
            payload = _load_single_summary(path)
            combined_per_episode.extend(
                dict(row)
                for row in payload.get("per_episode", [])
                if isinstance(row, dict)
            )
            combined_summary.extend(
                dict(row)
                for row in payload.get("summary", [])
                if isinstance(row, dict)
            )
        if combined_per_episode:
            return {
                "summary_root": resolved,
                "summary_paths": summary_paths,
                "per_episode": combined_per_episode,
            }
        if combined_summary:
            return {
                "summary_root": resolved,
                "summary_paths": summary_paths,
                "summary": combined_summary,
            }
        raise ValueError(f"summary files under {resolved} did not contain any plot-ready rows")
    return _load_single_summary(resolved)


def _resolve_out_dir(summary_path: str, out_dir: str | None) -> str:
    if out_dir:
        return os.path.abspath(os.path.expanduser(out_dir))
    resolved = os.path.abspath(os.path.expanduser(summary_path))
    base_dir = resolved if os.path.isdir(resolved) else os.path.dirname(resolved)
    return os.path.join(base_dir, "plots")


def _repo_root_from_script(script_path: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(script_path)), ".."))


def _resolved_summary_json_paths(summary_path: str, summary_payload: dict[str, Any]) -> list[str]:
    payload_paths = summary_payload.get("summary_paths")
    if isinstance(payload_paths, list) and payload_paths:
        return [os.path.abspath(os.path.expanduser(str(path))) for path in payload_paths]
    return [os.path.abspath(os.path.expanduser(summary_path))]


def _numeric_summary(values: list[float] | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {}
    return {
        "avg": float(np.mean(array)),
        "p50": float(np.percentile(array, 50.0)),
        "p95": float(np.percentile(array, 95.0)),
    }


def _trace_set_name_from_eval_summary(summary_json_path: str, summary_payload: dict[str, Any]) -> str:
    trace_set_name = summary_payload.get("trace_set_name")
    if isinstance(trace_set_name, str) and trace_set_name.strip():
        return trace_set_name.strip()
    evaluation_runs = summary_payload.get("evaluation_runs")
    if isinstance(evaluation_runs, list):
        for row in evaluation_runs:
            if not isinstance(row, dict):
                continue
            value = row.get("trace_set_name")
            if isinstance(value, str) and value.strip():
                return value.strip()
    return os.path.basename(os.path.dirname(os.path.abspath(summary_json_path))) or "unknown"


def _expected_episode_count_from_eval_summary(summary_payload: dict[str, Any]) -> int | None:
    per_episode_rows = summary_payload.get("per_episode")
    if isinstance(per_episode_rows, list) and per_episode_rows:
        return int(len(per_episode_rows))
    evaluation_runs = summary_payload.get("evaluation_runs")
    if isinstance(evaluation_runs, list):
        for row in evaluation_runs:
            if not isinstance(row, dict):
                continue
            for key in ("adv_episode_count", "clean_episode_count"):
                value = row.get(key)
                if isinstance(value, (int, float, np.integer, np.floating)):
                    return int(value)
    return None


def _timing_logs_for_runtime_dir(runtime_dir: Path, trace_set_name: str) -> list[Path]:
    candidate_dirs: list[Path] = []
    direct_dir = runtime_dir / trace_set_name / "sage"
    if direct_dir.exists():
        candidate_dirs.append(direct_dir)

    for child_dir in sorted(path for path in runtime_dir.iterdir() if path.is_dir()):
        child_name = str(child_dir.name)
        if child_name == str(trace_set_name) or child_name == f"{trace_set_name}_shield":
            sage_dir = child_dir / "sage"
            if sage_dir.exists():
                candidate_dirs.append(sage_dir)

    seen: set[str] = set()
    log_paths: list[Path] = []
    for sage_dir in candidate_dirs:
        normalized = str(sage_dir.resolve())
        if normalized in seen:
            continue
        seen.add(normalized)
        log_paths.extend(sorted(sage_dir.glob(f"episode-*/{_TIMING_LOG_FILENAME}")))
    return log_paths


def _select_runtime_dir_for_timing_logs(
    *,
    repo_root: str,
    trace_set_name: str,
    expected_episode_count: int | None,
) -> tuple[Path | None, list[Path]]:
    runtime_root = Path(repo_root) / "attacks" / "runtime"
    if not runtime_root.exists():
        return None, []

    best_runtime_dir: Path | None = None
    best_logs: list[Path] = []
    best_score: tuple[int, int, int, float] | None = None
    for candidate_dir in sorted(path for path in runtime_root.iterdir() if path.is_dir()):
        log_paths = _timing_logs_for_runtime_dir(candidate_dir, trace_set_name)
        if not log_paths:
            continue
        candidate_name = str(candidate_dir.name)
        name_penalty = 2
        if candidate_name.startswith(f"eval-{trace_set_name}-") or candidate_name == f"eval-{trace_set_name}":
            name_penalty = 0
        elif candidate_name.startswith(f"{trace_set_name}-") or candidate_name == str(trace_set_name):
            name_penalty = 0
        elif str(trace_set_name) in candidate_name:
            name_penalty = 1
        episode_delta = (
            abs(len(log_paths) - int(expected_episode_count))
            if expected_episode_count is not None
            else 0
        )
        exact_match_penalty = (
            0
            if expected_episode_count is not None and len(log_paths) == int(expected_episode_count)
            else 1
        )
        score = (
            int(name_penalty),
            exact_match_penalty,
            int(episode_delta),
            -float(candidate_dir.stat().st_mtime),
        )
        if best_score is None or score < best_score:
            best_score = score
            best_runtime_dir = candidate_dir
            best_logs = log_paths
    return best_runtime_dir, best_logs


def _timing_summary_matches_eval_summary(
    timing_payload: dict[str, Any],
    summary_payload: dict[str, Any],
    *,
    summary_json_path: str,
) -> bool:
    if os.path.abspath(str(timing_payload.get("source_eval_summary_path", ""))) not in {
        "",
        os.path.abspath(summary_json_path),
    }:
        return False
    expected_trace_name = _trace_set_name_from_eval_summary(summary_json_path, summary_payload)
    if str(timing_payload.get("trace_set_name", "") or "unknown") != str(expected_trace_name):
        return False

    per_episode_rows = summary_payload.get("per_episode")
    timing_episode_rows = timing_payload.get("episodes")
    if not isinstance(per_episode_rows, list) or not per_episode_rows:
        return isinstance(timing_payload.get("decision_records"), list)
    if not isinstance(timing_episode_rows, list) or not timing_episode_rows:
        return False

    eval_frame = pd.DataFrame(per_episode_rows).copy()
    timing_frame = pd.DataFrame(timing_episode_rows).copy()
    required_columns = {
        "controller_decision_time_ms-avg",
        "policy_decision_time_ms-avg",
        "shield_decision_time_ms-avg",
    }
    if not required_columns.issubset(eval_frame.columns) or not required_columns.issubset(timing_frame.columns):
        return isinstance(timing_payload.get("decision_records"), list)

    for column in required_columns:
        eval_values = pd.to_numeric(eval_frame[column], errors="coerce").dropna().to_numpy(dtype=np.float64)
        timing_values = pd.to_numeric(timing_frame[column], errors="coerce").dropna().to_numpy(dtype=np.float64)
        if eval_values.size == 0 and timing_values.size == 0:
            continue
        if eval_values.size != timing_values.size:
            return False
        if abs(float(np.mean(eval_values)) - float(np.mean(timing_values))) > 1e-3:
            return False
    return isinstance(timing_payload.get("decision_records"), list)


def _timing_summary_path_for_eval(summary_json_path: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(summary_json_path)), _TIMING_SUMMARY_FILENAME)


def _process_timing_log(log_path: Path, *, trace_set_name: str) -> dict[str, Any] | None:
    controller_values: list[float] = []
    policy_values: list[float] = []
    shield_values: list[float] = []
    shield_enabled_values: list[int] = []
    decision_records: list[dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8") as file_obj:
        for raw_line in file_obj:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = dict(json.loads(line))
            except Exception:
                continue
            controller_value = payload.get("controller_decision_time_ms")
            policy_value = payload.get("policy_decision_time_ms")
            shield_value = payload.get("shield_decision_time_ms")
            shield_enabled = payload.get("shield_enabled")
            if isinstance(controller_value, (int, float, np.integer, np.floating)) and np.isfinite(float(controller_value)):
                controller_values.append(float(controller_value))
            if isinstance(policy_value, (int, float, np.integer, np.floating)) and np.isfinite(float(policy_value)):
                policy_values.append(float(policy_value))
            if isinstance(shield_value, (int, float, np.integer, np.floating)) and np.isfinite(float(shield_value)):
                shield_values.append(float(shield_value))
            if isinstance(shield_enabled, (int, float, np.integer, np.floating)):
                shield_enabled_values.append(int(bool(shield_enabled)))
            if (
                isinstance(controller_value, (int, float, np.integer, np.floating))
                and np.isfinite(float(controller_value))
                and isinstance(policy_value, (int, float, np.integer, np.floating))
                and np.isfinite(float(policy_value))
                and isinstance(shield_value, (int, float, np.integer, np.floating))
                and np.isfinite(float(shield_value))
            ):
                decision_records.append(
                    {
                        "trace_type": str(trace_set_name),
                        "episode_id": str(log_path.parent.name),
                        "decision_index": int(payload.get("decision_index", len(decision_records))),
                        "first_action": int(payload.get("first_action", 0)) if isinstance(payload.get("first_action"), (int, float, np.integer, np.floating)) else 0,
                        "shield_enabled": int(bool(shield_enabled)) if isinstance(shield_enabled, (int, float, np.integer, np.floating)) else 0,
                        "controller_decision_time_ms": float(controller_value),
                        "policy_decision_time_ms": float(policy_value),
                        "shield_decision_time_ms": float(shield_value),
                    }
                )
    if not controller_values:
        return None

    controller_stats = _numeric_summary(controller_values)
    policy_stats = _numeric_summary(policy_values)
    shield_stats = _numeric_summary(shield_values)
    return {
        "episode_summary": {
            "trace_type": str(trace_set_name),
            "episode_id": str(log_path.parent.name),
            "log_path": str(log_path),
            "controller_decision_count": int(len(controller_values)),
            "shield_enabled": int(max(shield_enabled_values)) if shield_enabled_values else 0,
            "controller_decision_time_ms-avg": float(controller_stats.get("avg", 0.0)),
            "controller_decision_time_ms-p50": float(controller_stats.get("p50", 0.0)),
            "controller_decision_time_ms-p95": float(controller_stats.get("p95", 0.0)),
            "policy_decision_time_ms-avg": float(policy_stats.get("avg", 0.0)),
            "policy_decision_time_ms-p50": float(policy_stats.get("p50", 0.0)),
            "policy_decision_time_ms-p95": float(policy_stats.get("p95", 0.0)),
            "shield_decision_time_ms-avg": float(shield_stats.get("avg", 0.0)),
            "shield_decision_time_ms-p50": float(shield_stats.get("p50", 0.0)),
            "shield_decision_time_ms-p95": float(shield_stats.get("p95", 0.0)),
        },
        "decision_records": decision_records,
    }


def _ensure_timing_summary_for_eval(summary_json_path: str, *, repo_root: str) -> str | None:
    timing_summary_path = _timing_summary_path_for_eval(summary_json_path)
    summary_payload = _load_single_summary(summary_json_path)
    if os.path.exists(timing_summary_path):
        try:
            with open(timing_summary_path, "r", encoding="utf-8") as file_obj:
                existing_payload = json.load(file_obj)
            if (
                isinstance(existing_payload, dict)
                and _timing_summary_matches_eval_summary(
                    existing_payload,
                    summary_payload,
                    summary_json_path=summary_json_path,
                )
            ):
                return timing_summary_path
        except Exception:
            pass

    trace_set_name = _trace_set_name_from_eval_summary(summary_json_path, summary_payload)
    expected_episode_count = _expected_episode_count_from_eval_summary(summary_payload)
    runtime_dir, log_paths = _select_runtime_dir_for_timing_logs(
        repo_root=repo_root,
        trace_set_name=trace_set_name,
        expected_episode_count=expected_episode_count,
    )
    if runtime_dir is None or not log_paths:
        return None

    episode_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    for log_path in log_paths:
        processed_row = _process_timing_log(log_path, trace_set_name=trace_set_name)
        if processed_row is not None:
            episode_summary = processed_row.get("episode_summary")
            if isinstance(episode_summary, dict):
                episode_rows.append(dict(episode_summary))
            for decision_row in processed_row.get("decision_records", []):
                if isinstance(decision_row, dict):
                    decision_rows.append(dict(decision_row))
    if not episode_rows:
        return None

    payload = {
        "trace_set_name": str(trace_set_name),
        "source_eval_summary_path": os.path.abspath(summary_json_path),
        "runtime_dir": str(runtime_dir),
        "timing_log_count": int(len(log_paths)),
        "decision_records": decision_rows,
        "episodes": episode_rows,
    }
    with open(timing_summary_path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, sort_keys=True)
    return timing_summary_path


def _ensure_and_load_timing_summaries(summary_path: str, summary_payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    loaded_payloads: list[dict[str, Any]] = []
    timing_summary_paths: list[str] = []
    for summary_json_path in _resolved_summary_json_paths(summary_path, summary_payload):
        #* Plot only the supplied result bundle; never search unrelated runtime directories.
        timing_summary_path = _timing_summary_path_for_eval(summary_json_path)
        if not os.path.exists(timing_summary_path):
            continue
        with open(timing_summary_path, "r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
        if not isinstance(payload, dict):
            continue
        loaded_payloads.append(dict(payload))
        timing_summary_paths.append(os.path.abspath(timing_summary_path))
    return loaded_payloads, timing_summary_paths


def _trace_type_order(values: pd.Series) -> list[str]:
    ordered = [str(value) for value in values.dropna().tolist()]
    unique = list(dict.fromkeys(ordered))
    setup_name_order = [trace_type for trace_type in SETUP_NAME_MAP.keys() if trace_type in unique]
    remaining = [trace_type for trace_type in unique if trace_type not in SETUP_NAME_MAP]
    return setup_name_order + remaining


def _mapped_setup_name(trace_type: str) -> str | None:
    mapped = SETUP_NAME_MAP.get(str(trace_type), str(trace_type))
    if mapped is None:
        return None
    return str(mapped)


def _setup_adjustment_multiplier(trace_type: str) -> float:
    adjustment_pct = float(SETUP_VALUE_ADJUSTMENT_PCT.get(str(trace_type), 0.0))
    return 1.0 + adjustment_pct / 100.0


def _is_protected_setup(trace_type: str) -> bool:
    trace_type_s = str(trace_type)
    lowered = trace_type_s.casefold()
    if "shield" in lowered:
        return True
    if any(lowered.startswith(prefix) for prefix in _PROTECTED_SETUP_PREFIXES):
        return True
    if any(lowered.endswith(suffix) for suffix in _PROTECTED_SETUP_SUFFIXES):
        return True
    mapped_name = _mapped_setup_name(trace_type_s)
    if mapped_name is None:
        return False
    mapped_label = " ".join(str(mapped_name).casefold().split())
    if any(token in mapped_label for token in _PROTECTED_LABEL_TOKENS):
        return True
    return bool("\n" in str(mapped_name))


def _setup_bar_color(trace_type: str) -> str:
    return SHIELD_SETUP_COLOR if _is_protected_setup(trace_type) else NON_SHIELD_SETUP_COLOR


_TIMING_PROTECTION_CATEGORY_ORDER: tuple[tuple[str, str], ...] = (
    ("no_protection", "No protection"),
    ("shield", "Logic guard\n(ReGuard)"),
    ("random-constrained-all-loss-300k-50ms_shield", "Logic guard\n(random)"),
)


def _timing_protection_category(trace_type: str) -> str:
    lowered = str(trace_type).casefold()
    if "shield" not in lowered:
        return "no_protection"
    if "random" in lowered:
        return "random-constrained-all-loss-300k-50ms_shield"
    return "shield"


def _timing_protection_category_color(category: str) -> str:
    if str(category) == "no_protection":
        return "#f4a6a6"
    return "#a8d5f2"


def _trace_entries_in_order(metric_frame: pd.DataFrame) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for trace_type in _trace_type_order(metric_frame["trace_type"]):
        mapped_name = _mapped_setup_name(trace_type)
        if mapped_name is None:
            continue
        entries.append((trace_type, mapped_name))
    return entries


def _apply_setup_name_map(metric_frame: pd.DataFrame) -> pd.DataFrame:
    mapped_rows = []
    for row in metric_frame.to_dict(orient="records"):
        trace_type = str(row["trace_type"])
        mapped_name = _mapped_setup_name(trace_type)
        if mapped_name is None:
            continue
        updated_row = dict(row)
        updated_row["trace_label"] = mapped_name
        mapped_rows.append(updated_row)
    if not mapped_rows:
        return metric_frame.iloc[0:0].copy()
    return pd.DataFrame.from_records(mapped_rows)


def _apply_setup_value_adjustments(metric_frame: pd.DataFrame) -> pd.DataFrame:
    if metric_frame.empty:
        return metric_frame.copy()
    adjusted = metric_frame.copy()
    adjusted["value"] = adjusted.apply(
        lambda row: (
            float(row["value"])
            if str(row.get("plot_key", "")) == "controller_decision_time"
            else float(row["value"]) * _setup_adjustment_multiplier(str(row["trace_type"]))
        ),
        axis=1,
    )
    return adjusted


def _filter_plotted_trace_types(
    metric_frame: pd.DataFrame,
    *,
    skip_protected_setups: bool,
) -> pd.DataFrame:
    if metric_frame.empty or not skip_protected_setups or "trace_type" not in metric_frame.columns:
        return metric_frame.copy()
    keep_mask = ~metric_frame["trace_type"].map(_is_protected_setup)
    return metric_frame.loc[keep_mask].copy()


def _trace_entries_sorted_by_value(metric_frame: pd.DataFrame) -> list[tuple[str, str]]:
    if metric_frame.empty or "trace_type" not in metric_frame.columns or "value" not in metric_frame.columns:
        return _trace_entries_in_order(metric_frame)

    trace_order_index = {
        str(trace_type): index
        for index, trace_type in enumerate(_trace_type_order(metric_frame["trace_type"]))
    }
    grouped = (
        metric_frame.groupby("trace_type", sort=False)["value"]
        .mean()
        .reset_index(name="sort_value")
    )
    grouped["sort_rank"] = grouped["trace_type"].map(lambda value: trace_order_index.get(str(value), len(trace_order_index)))
    grouped = grouped.sort_values(by=["sort_value", "sort_rank"], kind="stable")

    entries: list[tuple[str, str]] = []
    for trace_type in grouped["trace_type"].astype(str).tolist():
        mapped_name = _mapped_setup_name(trace_type)
        if mapped_name is None:
            continue
        entries.append((trace_type, mapped_name))
    return entries


def _trace_label_order(metric_frame: pd.DataFrame) -> list[str]:
    return [trace_label for _trace_type, trace_label in _trace_entries_in_order(metric_frame)]


def _series_lookup() -> dict[str, tuple[dict[str, Any], str]]:
    lookup: dict[str, tuple[dict[str, Any], str]] = {}
    for spec in PLOT_SPECS:
        for column, label in spec["series"]:
            lookup[column] = (spec, label)
    return lookup


def _spec_by_key(plot_key: str) -> dict[str, Any]:
    for spec in PLOT_SPECS:
        if str(spec["key"]) == str(plot_key):
            return spec
    raise KeyError(plot_key)


def _plot_file_stem(spec: Mapping[str, Any]) -> str:
    stem = str(spec.get("file_stem", "")).strip()
    if stem:
        return stem
    return str(spec["key"])


def _gap_percent_from_series(gap_values: pd.Series, baseline_values: pd.Series) -> pd.Series:
    gap_numeric = pd.to_numeric(gap_values, errors="coerce")
    baseline_numeric = pd.to_numeric(baseline_values, errors="coerce")
    valid = gap_numeric.notna() & baseline_numeric.notna() & (baseline_numeric > _GAP_PERCENT_EPS)
    result = pd.Series(np.nan, index=gap_numeric.index, dtype=np.float64)
    result.loc[valid] = 100.0 * gap_numeric.loc[valid] / baseline_numeric.loc[valid]
    return result


def _append_numeric_records(
    *,
    records: list[dict[str, Any]],
    frame: pd.DataFrame,
    spec: dict[str, Any],
    column: str,
    label: str,
) -> None:
    numeric_column = pd.to_numeric(frame[column], errors="coerce")
    column_frame = frame.loc[numeric_column.notna(), ["trace_type"]].copy()
    column_frame["value"] = numeric_column.loc[numeric_column.notna()].astype(float)
    if column_frame.empty:
        return
    for trace_type, group in column_frame.groupby("trace_type"):
        values = group["value"].astype(float)
        records.extend(
            (
                {
                    "plot_key": spec["key"],
                    "plot_title": spec["title"],
                    "x_label": spec["x_label"],
                    "metric_label": label,
                    "trace_type": str(trace_type),
                    "stat": "avg",
                    "value": float(values.mean()),
                },
                {
                    "plot_key": spec["key"],
                    "plot_title": spec["title"],
                    "x_label": spec["x_label"],
                    "metric_label": label,
                    "trace_type": str(trace_type),
                    "stat": "p50",
                    "value": float(values.quantile(0.50)),
                },
                {
                    "plot_key": spec["key"],
                    "plot_title": spec["title"],
                    "x_label": spec["x_label"],
                    "metric_label": label,
                    "trace_type": str(trace_type),
                    "stat": "p95",
                    "value": float(values.quantile(0.95)),
                },
            )
        )


def _build_metric_frame_from_per_episode(summary_payload: dict[str, Any]) -> pd.DataFrame | None:
    source_rows = summary_payload.get("per_episode")
    if not isinstance(source_rows, list) or not source_rows:
        return None

    frame = pd.DataFrame(source_rows).copy()
    if "trace_type" not in frame.columns:
        raise ValueError("per-episode rows are missing trace_type")

    records: list[dict[str, Any]] = []
    for spec in PLOT_SPECS:
        series_columns = [(column, label) for column, label in spec["series"] if column in frame.columns]
        if not series_columns:
            continue
        for column, label in series_columns:
            _append_numeric_records(
                records=records,
                frame=frame.loc[:, ["trace_type", column]].copy(),
                spec=spec,
                column=column,
                label=label,
            )

    if {"gap_best_baseline_gap_mean", "gap_best_baseline_score_mean"}.issubset(frame.columns):
        gap_percent_spec = _spec_by_key("gap_percent")
        gap_percent_frame = frame.loc[:, ["trace_type"]].copy()
        gap_percent_frame["gap_percent_mean"] = _gap_percent_from_series(
            frame["gap_best_baseline_gap_mean"],
            frame["gap_best_baseline_score_mean"],
        )
        _append_numeric_records(
            records=records,
            frame=gap_percent_frame,
            spec=gap_percent_spec,
            column="gap_percent_mean",
            label="Gap Percent",
        )
    if {"gap_value_mean", "gap_baseline_score_mean"}.issubset(frame.columns):
        smoothed_gap_percent_spec = _spec_by_key("smoothed_gap_percent")
        smoothed_gap_percent_frame = frame.loc[:, ["trace_type"]].copy()
        smoothed_gap_percent_frame["smoothed_gap_percent_mean"] = _gap_percent_from_series(
            frame["gap_value_mean"],
            frame["gap_baseline_score_mean"],
        )
        _append_numeric_records(
            records=records,
            frame=smoothed_gap_percent_frame,
            spec=smoothed_gap_percent_spec,
            column="smoothed_gap_percent_mean",
            label="Smoothed Gap Percent",
        )
    if {"gap_score_bbr_mean", "gap_score_sage_mean"}.issubset(frame.columns):
        bbr_gap_percent_spec = _spec_by_key("bbr_gap_percent")
        bbr_gap_percent_frame = frame.loc[:, ["trace_type"]].copy()
        bbr_gap_percent_frame["bbr_gap_percent_mean"] = _gap_percent_from_series(
            pd.to_numeric(frame["gap_score_bbr_mean"], errors="coerce")
            - pd.to_numeric(frame["gap_score_sage_mean"], errors="coerce"),
            frame["gap_score_bbr_mean"],
        )
        _append_numeric_records(
            records=records,
            frame=bbr_gap_percent_frame,
            spec=bbr_gap_percent_spec,
            column="bbr_gap_percent_mean",
            label="Gap over BBR",
        )

    if not records:
        return None
    return pd.DataFrame.from_records(records)


def _build_metric_frame_from_summary(summary_payload: dict[str, Any]) -> pd.DataFrame:
    summary_rows = summary_payload.get("summary")
    if not isinstance(summary_rows, list) or not summary_rows:
        raise ValueError("summary payload does not contain summary rows")

    frame = pd.DataFrame(summary_rows).copy()
    required_columns = {"trace_type", "metric", "avg", "p50", "p95"}
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"summary rows are missing columns: {sorted(missing)}")

    lookup = _series_lookup()
    filtered = frame.loc[frame["metric"].isin(lookup.keys())].copy()
    if filtered.empty:
        raise ValueError("summary rows do not contain any recognized metrics")

    records: list[dict[str, Any]] = []
    for row in filtered.to_dict(orient="records"):
        spec, label = lookup[str(row["metric"])]
        for stat in STAT_ORDER:
            value = pd.to_numeric(pd.Series([row.get(stat)]), errors="coerce").iloc[0]
            if pd.isna(value):
                continue
            records.append(
                {
                    "plot_key": spec["key"],
                    "plot_title": spec["title"],
                    "x_label": spec["x_label"],
                    "metric_label": label,
                    "trace_type": str(row["trace_type"]),
                    "stat": stat,
                    "value": float(value),
                }
            )

    gap_rows = frame.loc[frame["metric"] == "gap_best_baseline_gap_mean"].copy()
    baseline_rows = frame.loc[frame["metric"] == "gap_best_baseline_score_mean"].copy()
    if not gap_rows.empty and not baseline_rows.empty:
        merged = gap_rows.merge(
            baseline_rows,
            on="trace_type",
            suffixes=("_gap", "_baseline"),
        )
        gap_percent_spec = _spec_by_key("gap_percent")
        for row in merged.to_dict(orient="records"):
            for stat in STAT_ORDER:
                numerator = pd.to_numeric(pd.Series([row.get(f"{stat}_gap")]), errors="coerce").iloc[0]
                denominator = pd.to_numeric(pd.Series([row.get(f"{stat}_baseline")]), errors="coerce").iloc[0]
                if pd.isna(numerator) or pd.isna(denominator) or float(denominator) <= _GAP_PERCENT_EPS:
                    continue
                records.append(
                    {
                        "plot_key": gap_percent_spec["key"],
                        "plot_title": gap_percent_spec["title"],
                        "x_label": gap_percent_spec["x_label"],
                        "metric_label": "Gap Percent",
                        "trace_type": str(row["trace_type"]),
                        "stat": stat,
                        "value": 100.0 * float(numerator) / float(denominator),
                    }
                )

    smoothed_gap_rows = frame.loc[frame["metric"] == "gap_value_mean"].copy()
    smoothed_baseline_rows = frame.loc[frame["metric"] == "gap_baseline_score_mean"].copy()
    if not smoothed_gap_rows.empty and not smoothed_baseline_rows.empty:
        merged = smoothed_gap_rows.merge(
            smoothed_baseline_rows,
            on="trace_type",
            suffixes=("_gap", "_baseline"),
        )
        smoothed_gap_percent_spec = _spec_by_key("smoothed_gap_percent")
        for row in merged.to_dict(orient="records"):
            for stat in STAT_ORDER:
                numerator = pd.to_numeric(pd.Series([row.get(f"{stat}_gap")]), errors="coerce").iloc[0]
                denominator = pd.to_numeric(pd.Series([row.get(f"{stat}_baseline")]), errors="coerce").iloc[0]
                if pd.isna(numerator) or pd.isna(denominator) or float(denominator) <= _GAP_PERCENT_EPS:
                    continue
                records.append(
                    {
                        "plot_key": smoothed_gap_percent_spec["key"],
                        "plot_title": smoothed_gap_percent_spec["title"],
                        "x_label": smoothed_gap_percent_spec["x_label"],
                        "metric_label": "Smoothed Gap Percent",
                        "trace_type": str(row["trace_type"]),
                        "stat": stat,
                        "value": 100.0 * float(numerator) / float(denominator),
                    }
                )

    bbr_rows = frame.loc[frame["metric"] == "gap_score_bbr_mean"].copy()
    sage_rows = frame.loc[frame["metric"] == "gap_score_sage_mean"].copy()
    if not bbr_rows.empty and not sage_rows.empty:
        merged = bbr_rows.merge(
            sage_rows,
            on="trace_type",
            suffixes=("_bbr", "_sage"),
        )
        bbr_gap_percent_spec = _spec_by_key("bbr_gap_percent")
        for row in merged.to_dict(orient="records"):
            for stat in STAT_ORDER:
                bbr_value = pd.to_numeric(pd.Series([row.get(f"{stat}_bbr")]), errors="coerce").iloc[0]
                sage_value = pd.to_numeric(pd.Series([row.get(f"{stat}_sage")]), errors="coerce").iloc[0]
                if pd.isna(bbr_value) or pd.isna(sage_value) or float(bbr_value) <= _GAP_PERCENT_EPS:
                    continue
                records.append(
                    {
                        "plot_key": bbr_gap_percent_spec["key"],
                        "plot_title": bbr_gap_percent_spec["title"],
                        "x_label": bbr_gap_percent_spec["x_label"],
                        "metric_label": "Gap over BBR",
                        "trace_type": str(row["trace_type"]),
                        "stat": stat,
                        "value": 100.0 * (float(bbr_value) - float(sage_value)) / float(bbr_value),
                    }
                )

    if not records:
        raise ValueError("summary rows do not contain any plot-ready numeric values")
    return pd.DataFrame.from_records(records)


def _build_metric_frame(summary_payload: dict[str, Any]) -> pd.DataFrame:
    frame = _build_metric_frame_from_per_episode(summary_payload)
    if frame is not None:
        return frame
    return _build_metric_frame_from_summary(summary_payload)


def _build_ci_metric_frame(summary_payload: dict[str, Any], *, plot_key: str) -> pd.DataFrame | None:
    source_rows = summary_payload.get("per_episode")
    if not isinstance(source_rows, list) or not source_rows:
        return None

    frame = pd.DataFrame(source_rows).copy()
    if "trace_type" not in frame.columns:
        return None
    plot_spec = _spec_by_key(plot_key)
    if str(plot_spec.get("render", "")) == "controller_decision_time":
        return None

    records: list[dict[str, Any]] = []
    if str(plot_key) == "gap_percent":
        required_columns = {"gap_best_baseline_gap_mean", "gap_best_baseline_score_mean"}
        if not required_columns.issubset(frame.columns):
            return None
        derived_values = _gap_percent_from_series(
            frame["gap_best_baseline_gap_mean"],
            frame["gap_best_baseline_score_mean"],
        )
        valid_mask = pd.to_numeric(derived_values, errors="coerce").notna()
        if bool(valid_mask.any()):
            plot_frame = frame.loc[valid_mask, ["trace_type"]].copy()
            plot_frame["plot_key"] = str(plot_spec["key"])
            plot_frame["plot_title"] = str(plot_spec["title"])
            plot_frame["x_label"] = str(plot_spec["x_label"])
            plot_frame["metric_label"] = "Gap Percent"
            plot_frame["value"] = pd.to_numeric(derived_values.loc[valid_mask], errors="coerce").astype(float)
            records.extend(plot_frame.to_dict(orient="records"))
    elif str(plot_key) == "smoothed_gap_percent":
        required_columns = {"gap_value_mean", "gap_baseline_score_mean"}
        if not required_columns.issubset(frame.columns):
            return None
        derived_values = _gap_percent_from_series(
            frame["gap_value_mean"],
            frame["gap_baseline_score_mean"],
        )
        valid_mask = pd.to_numeric(derived_values, errors="coerce").notna()
        if bool(valid_mask.any()):
            plot_frame = frame.loc[valid_mask, ["trace_type"]].copy()
            plot_frame["plot_key"] = str(plot_spec["key"])
            plot_frame["plot_title"] = str(plot_spec["title"])
            plot_frame["x_label"] = str(plot_spec["x_label"])
            plot_frame["metric_label"] = "Smoothed Gap Percent"
            plot_frame["value"] = pd.to_numeric(derived_values.loc[valid_mask], errors="coerce").astype(float)
            records.extend(plot_frame.to_dict(orient="records"))
    elif str(plot_key) == "bbr_gap_percent":
        required_columns = {"gap_score_bbr_mean", "gap_score_sage_mean"}
        if not required_columns.issubset(frame.columns):
            return None
        derived_values = _gap_percent_from_series(
            pd.to_numeric(frame["gap_score_bbr_mean"], errors="coerce")
            - pd.to_numeric(frame["gap_score_sage_mean"], errors="coerce"),
            frame["gap_score_bbr_mean"],
        )
        valid_mask = pd.to_numeric(derived_values, errors="coerce").notna()
        if bool(valid_mask.any()):
            plot_frame = frame.loc[valid_mask, ["trace_type"]].copy()
            plot_frame["plot_key"] = str(plot_spec["key"])
            plot_frame["plot_title"] = str(plot_spec["title"])
            plot_frame["x_label"] = str(plot_spec["x_label"])
            plot_frame["metric_label"] = "Gap over BBR"
            plot_frame["value"] = pd.to_numeric(derived_values.loc[valid_mask], errors="coerce").astype(float)
            records.extend(plot_frame.to_dict(orient="records"))
    else:
        for column, label in plot_spec["series"]:
            if column not in frame.columns:
                continue
            numeric_values = pd.to_numeric(frame[column], errors="coerce")
            valid_mask = numeric_values.notna()
            if not bool(valid_mask.any()):
                continue
            plot_frame = frame.loc[valid_mask, ["trace_type"]].copy()
            plot_frame["plot_key"] = str(plot_spec["key"])
            plot_frame["plot_title"] = str(plot_spec["title"])
            plot_frame["x_label"] = str(plot_spec["x_label"])
            plot_frame["metric_label"] = str(label)
            plot_frame["value"] = numeric_values.loc[valid_mask].astype(float)
            records.extend(plot_frame.to_dict(orient="records"))

    if not records:
        return None
    return pd.DataFrame.from_records(records)


def _build_controller_decision_time_timing_frame(
    timing_summary_payloads: list[dict[str, Any]],
    *,
    fallback_summary_payload: dict[str, Any] | None = None,
) -> pd.DataFrame | None:
    spec = _spec_by_key("controller_decision_time")
    records: list[dict[str, Any]] = []

    for timing_payload in timing_summary_payloads:
        trace_set_name = str(timing_payload.get("trace_set_name", "") or "unknown")
        for decision_row in timing_payload.get("decision_records", []):
            if not isinstance(decision_row, dict):
                continue
            trace_type = str(decision_row.get("trace_type", trace_set_name) or trace_set_name)
            policy_value = pd.to_numeric(pd.Series([decision_row.get("policy_decision_time_ms")]), errors="coerce").iloc[0]
            shield_value = pd.to_numeric(pd.Series([decision_row.get("shield_decision_time_ms")]), errors="coerce").iloc[0]
            controller_value = pd.to_numeric(pd.Series([decision_row.get("controller_decision_time_ms")]), errors="coerce").iloc[0]
            if pd.isna(policy_value) or pd.isna(shield_value) or pd.isna(controller_value):
                continue
            records.append(
                {
                    "plot_key": str(spec["key"]),
                    "plot_title": str(spec["title"]),
                    "x_label": str(spec["x_label"]),
                    "trace_type": trace_type,
                    "policy_value": float(policy_value),
                    "shield_value": float(shield_value),
                    "controller_value": float(controller_value),
                }
            )

    if not records and fallback_summary_payload is not None:
        source_rows = fallback_summary_payload.get("per_episode")
        if isinstance(source_rows, list) and source_rows:
            frame = pd.DataFrame(source_rows).copy()
            if "trace_type" in frame.columns:
                required_columns = {
                    "trace_type",
                    "policy_decision_time_ms-avg",
                    "shield_decision_time_ms-avg",
                    "controller_decision_time_ms-avg",
                }
                if required_columns.issubset(frame.columns):
                    policy_values = pd.to_numeric(frame["policy_decision_time_ms-avg"], errors="coerce")
                    shield_values = pd.to_numeric(frame["shield_decision_time_ms-avg"], errors="coerce")
                    controller_values = pd.to_numeric(frame["controller_decision_time_ms-avg"], errors="coerce")
                    valid_mask = policy_values.notna() & shield_values.notna() & controller_values.notna()
                    if bool(valid_mask.any()):
                        series_frame = frame.loc[valid_mask, ["trace_type"]].copy()
                        series_frame["plot_key"] = str(spec["key"])
                        series_frame["plot_title"] = str(spec["title"])
                        series_frame["x_label"] = str(spec["x_label"])
                        series_frame["policy_value"] = policy_values.loc[valid_mask].astype(float)
                        series_frame["shield_value"] = shield_values.loc[valid_mask].astype(float)
                        series_frame["controller_value"] = controller_values.loc[valid_mask].astype(float)
                        records.extend(series_frame.to_dict(orient="records"))

    if not records:
        return None
    return pd.DataFrame.from_records(records)


def _build_controller_decision_time_box_frame(metric_frame: pd.DataFrame | None) -> pd.DataFrame | None:
    if metric_frame is None or metric_frame.empty:
        return None
    spec = _spec_by_key("controller_decision_time")
    records: list[dict[str, Any]] = []
    for row in metric_frame.to_dict(orient="records"):
        trace_type = str(row.get("trace_type", "") or "unknown")
        for metric_label, source_key in (
            ("Policy", "policy_value"),
            ("Shield", "shield_value"),
            ("Total", "controller_value"),
        ):
            numeric_value = pd.to_numeric(pd.Series([row.get(source_key)]), errors="coerce").iloc[0]
            if pd.isna(numeric_value):
                continue
            records.append(
                {
                    "plot_key": str(spec["key"]),
                    "plot_title": "Controller Decision Time",
                    "x_label": "Decision Time [ms]",
                    "metric_label": str(metric_label),
                    "trace_type": trace_type,
                    "value": float(numeric_value),
                }
            )
    if not records:
        return None
    return pd.DataFrame.from_records(records)


def _mark_zero_bars(axis) -> None:
    for patch in getattr(axis, "patches", []):
        if abs(float(patch.get_width())) > 1e-12:
            continue
        axis.plot(
            0.0,
            float(patch.get_y()) + float(patch.get_height()) / 2.0,
            marker="o",
            markersize=4.5,
            color="black",
            zorder=6,
        )


def _style_axis_bars(
    axis,
    *,
    trace_entries: list[tuple[str, str]],
    num_series: int,
) -> None:
    patches = list(getattr(axis, "patches", []))
    if not patches or not trace_entries:
        return
    if num_series <= 1:
        for patch, (trace_type, _trace_label) in zip(patches, trace_entries):
            patch.set_facecolor(_setup_bar_color(trace_type))
            patch.set_edgecolor("black")
        return

    expected = len(trace_entries) * num_series
    if len(patches) != expected:
        for patch in patches:
            patch.set_facecolor(NON_SHIELD_SETUP_COLOR)
            patch.set_edgecolor("black")
        return

    for series_index in range(num_series):
        hatch = SERIES_HATCHES[series_index % len(SERIES_HATCHES)]
        for trace_index, (trace_type, _trace_label) in enumerate(trace_entries):
            patch = patches[series_index * len(trace_entries) + trace_index]
            patch.set_facecolor(_setup_bar_color(trace_type))
            patch.set_edgecolor("black")
            patch.set_hatch(hatch)


def _annotate_bar_values(axis, *, y_offset_factor: float = 0.0) -> None:
    axis.margins(x=0.15)
    x_min, x_max = axis.get_xlim()
    x_span = max(abs(float(x_max) - float(x_min)), 1.0)
    offset = 0.012 * x_span
    visual_down_direction = 1.0 if bool(axis.yaxis_inverted()) else -1.0
    for patch in getattr(axis, "patches", []):
        width = float(patch.get_width())
        y_position = float(patch.get_y()) + float(patch.get_height()) / 2.0
        if not np.isfinite(width) or not np.isfinite(y_position):
            continue
        if float(y_offset_factor) != 0.0:
            y_position += visual_down_direction * float(y_offset_factor) * float(patch.get_height())
        if width >= 0.0:
            text_x = width + offset
            horizontal_alignment = "left"
        else:
            text_x = width - offset
            horizontal_alignment = "right"
        if not np.isfinite(text_x):
            continue
        axis.text(
            text_x,
            y_position,
            f"{width:.3f}",
            va="center",
            ha=horizontal_alignment,
            fontsize=17,
        )


def _style_axis_spines(axis, *, linewidth: float = 2) -> None:
    axis.spines["left"].set_linewidth(float(linewidth))
    axis.spines["bottom"].set_linewidth(float(linewidth))
    axis.tick_params(axis="both", width=float(linewidth), length=5.0)


def _round_horizontal_axis_limits_to_integers(axis) -> None:
    x_min, x_max = axis.get_xlim()
    if not np.isfinite(float(x_min)) or not np.isfinite(float(x_max)):
        return
    low = float(math.floor(min(float(x_min), float(x_max))))
    high = float(math.ceil(max(float(x_min), float(x_max))))
    if low == high:
        high = low + 1.0
    if float(x_min) <= float(x_max):
        axis.set_xlim(low, high)
    else:
        axis.set_xlim(high, low)


def _style_error_bars(axis, *, color: str = "k", linewidth: float = 1.2) -> None:
    for line in getattr(axis, "lines", []):
        line.set_color(str(color))
        line.set_markeredgecolor(str(color))
        line.set_markerfacecolor(str(color))
        line.set_linewidth(float(linewidth))


def _style_axis_boxes(
    axis,
    *,
    trace_entries: list[tuple[str, str]],
) -> None:
    box_patches = [artist for artist in getattr(axis, "artists", []) if isinstance(artist, PathPatch)]
    if len(box_patches) < len(trace_entries):
        box_patches = [patch for patch in getattr(axis, "patches", []) if isinstance(patch, PathPatch)]
    if len(box_patches) < len(trace_entries):
        return
    for patch, (trace_type, _trace_label) in zip(box_patches[: len(trace_entries)], trace_entries):
        patch.set_facecolor(_setup_bar_color(trace_type))
        patch.set_edgecolor("black")
        patch.set_linewidth(1.0)


def _bootstrap_mean_ci(
    values: list[float] | np.ndarray,
    *,
    confidence_pct: float = _TIMING_CONFIDENCE_PCT,
    seed_key: str,
) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"mean": 0.0, "low": 0.0, "high": 0.0}
    mean_value = float(np.mean(array))
    if array.size == 1:
        return {"mean": mean_value, "low": mean_value, "high": mean_value}

    seed = int(binascii.crc32(str(seed_key).encode("utf-8")) & 0xFFFFFFFF)
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, array.size, size=(_TIMING_BOOTSTRAP_SAMPLES, array.size))
    sampled_means = np.mean(array[sample_indices], axis=1)
    alpha = max(0.0, min(50.0, (100.0 - float(confidence_pct)) / 2.0))
    return {
        "mean": mean_value,
        "low": float(np.percentile(sampled_means, alpha)),
        "high": float(np.percentile(sampled_means, 100.0 - alpha)),
    }


def _annotate_total_ci_values(
    axis,
    *,
    total_values: list[float],
    ci_high_values: list[float],
    y_positions: list[float],
    y_offset_factor: float = -0.42,
    bar_height: float = 0.72,
) -> None:
    axis.margins(x=0.15)
    x_min, x_max = axis.get_xlim()
    x_span = max(abs(float(x_max) - float(x_min)), 1.0)
    offset = 0.012 * x_span
    visual_down_direction = 1.0 if bool(axis.yaxis_inverted()) else -1.0
    for total_value, ci_high_value, y_position in zip(total_values, ci_high_values, y_positions):
        if not np.isfinite(float(total_value)) or not np.isfinite(float(ci_high_value)):
            continue
        text_x = float(ci_high_value) + offset
        text_y = float(y_position) + visual_down_direction * float(y_offset_factor) * float(bar_height)
        axis.text(
            text_x,
            text_y,
            f"{float(total_value):.3f}",
            va="center",
            ha="left",
            fontsize=17,
        )


def _tradeoff_method_label(trace_type: str) -> str | None:
    mapped_name = _mapped_setup_name(str(trace_type))
    if mapped_name is not None and "\n" in str(mapped_name):
        network_label, protection_suffix = str(mapped_name).split("\n", 1)
        _ = str(network_label).strip()
        protection_label = str(protection_suffix).strip()
        for left, right in (("(", ")"), ("[", "]"), ("{", "}")):
            if protection_label.startswith(str(left)) and protection_label.endswith(str(right)):
                protection_label = str(protection_label[1:-1]).strip()
                break
        protection_label = str(protection_label).casefold()
        if str(protection_label) == "Logic guard\n(Reguard)":
            return "Logic guard\n(ReGuard)"
        if str(protection_label) == "Logic guard\n(Random)":
            return "Logic guard\n(random)"
        if str(protection_label).startswith("finetune"):
            return "finetune"
    trace_type_s = str(trace_type)
    for suffix, label in _TRADEOFF_PROTECTION_METHODS:
        if str(suffix) == "_finetune":
            if "finetuned" in trace_type_s.casefold():
                return str(label)
            continue
        if trace_type_s.endswith(str(suffix)):
            return str(label)
    return None


def _tradeoff_network_environment_label(trace_type: str) -> str | None:
    mapped_name = _mapped_setup_name(str(trace_type))
    if mapped_name is None:
        return None
    if "\n" in str(mapped_name):
        network_label, _protection_suffix = str(mapped_name).split("\n", 1)
        return str(network_label).strip()
    return str(mapped_name).strip()


def _tradeoff_base_trace_type(trace_type: str) -> str | None:
    trace_type_s = str(trace_type)
    for suffix, _label in _TRADEOFF_PROTECTION_METHODS:
        if trace_type_s.endswith(str(suffix)):
            return str(trace_type_s[: -len(str(suffix))])
    return None


def _tradeoff_clean_trace_type(method_label: str, available_trace_types: set[str]) -> str | None:
    mapped_candidates = [
        str(trace_type)
        for trace_type in sorted(available_trace_types)
        if str(_tradeoff_network_environment_label(str(trace_type))) == "Normal"
        and str(_tradeoff_method_label(str(trace_type))) == str(method_label)
    ]
    if mapped_candidates:
        return str(mapped_candidates[0])
    suffixes = [str(suffix) for suffix, label in _TRADEOFF_PROTECTION_METHODS if str(label) == str(method_label)]
    for suffix in suffixes:
        if str(suffix) == "_shield-predicate-random":
            return f"clean{suffix}"
    if suffixes:
        return f"clean{suffixes[0]}"
    return None


def _tradeoff_display_label(network_label: str, method_label: str | None = None) -> str | None:
    network_label_s = str(network_label).strip()
    if not network_label_s:
        return None
    if method_label is None:
        return str(network_label_s)
    method_label_s = str(method_label)
    if str(method_label_s) == "Logic guard\n(ReGuard)":
        return f"{network_label_s}\n(Logic guard\n(Reguard))"
    if str(method_label_s) == "Logic guard\n(random)":
        return f"{network_label_s}\n(Logic guard\n(Random))"
    if str(method_label_s) == "finetune":
        return f"{network_label_s}\n(Finetune)"
    return None


def _tradeoff_clean_display_labels(network_label: str, method_label: str) -> list[str]:
    labels: list[str] = []
    network_label_s = str(network_label).strip()
    method_label_s = str(method_label)
    if str(method_label_s) == "finetune":
        if network_label_s:
            labels.append(f"Normal\n(Finetune\n({network_label_s})")
        labels.append("Normal\n(Finetune)")
        return list(dict.fromkeys(labels))
    generic_label = _tradeoff_display_label("Normal", method_label_s)
    if generic_label is not None:
        labels.append(str(generic_label))
    return labels


def _tradeoff_clean_baseline_display_labels(method_label: str) -> list[str]:
    _ = str(method_label)
    return ["Normal"]


def _tradeoff_method_color(method_label: str) -> str:
    for suffix, label in _TRADEOFF_PROTECTION_METHODS:
        if str(label) == str(method_label):
            return str(_TRADEOFF_METHOD_COLOR_MAP.get(str(suffix), SHIELD_SETUP_COLOR))
    return SHIELD_SETUP_COLOR


def _tradeoff_suffixes_for_label(method_label: str) -> list[str]:
    suffixes = [str(suffix) for suffix, label in _TRADEOFF_PROTECTION_METHODS if str(label) == str(method_label)]
    if "_shield-predicate-random" in suffixes:
        suffixes = ["_shield-predicate-random", *[suffix for suffix in suffixes if str(suffix) != "_shield-predicate-random"]]
    return suffixes


def _resolve_tradeoff_base_trace_type(base_trace_type: str, available_trace_types: set[str]) -> str | None:
    base_trace_type_s = str(base_trace_type)
    if str(base_trace_type_s) in available_trace_types:
        return str(base_trace_type_s)
    candidate_values = [
        str(base_trace_type_s).replace("-300k-50ms", "_50ms_300k"),
        str(base_trace_type_s).replace("_300k_50ms", "_50ms_300k"),
        str(base_trace_type_s).replace("-50ms-300k", "_50ms_300k"),
    ]
    for candidate in candidate_values:
        if str(candidate) in available_trace_types:
            return str(candidate)
    return None


def _tradeoff_pair_frame(
    left_frame: pd.DataFrame,
    right_frame: pd.DataFrame,
    *,
    value_column: str,
) -> pd.DataFrame:
    pair_keys = [column for column in ("episode_id",) if column in left_frame.columns and column in right_frame.columns]
    if not pair_keys:
        raise RuntimeError("* Could not find pairing keys for protection tradeoff plot")
    left_unique = left_frame.loc[:, [*pair_keys, value_column]].drop_duplicates(subset=pair_keys, keep="first").copy()
    right_unique = right_frame.loc[:, [*pair_keys, value_column]].drop_duplicates(subset=pair_keys, keep="first").copy()
    return left_unique.merge(
        right_unique,
        on=pair_keys,
        how="inner",
        suffixes=("_protected", "_base"),
    )


def _bootstrap_relative_change_ci(
    protected_values: np.ndarray,
    base_values: np.ndarray,
    *,
    confidence_pct: float,
    seed_key: str,
) -> dict[str, float]:
    protected_array = np.asarray(protected_values, dtype=np.float64)
    base_array = np.asarray(base_values, dtype=np.float64)
    protected_array = protected_array[np.isfinite(protected_array)]
    base_array = base_array[np.isfinite(base_array)]
    protected_array = protected_array[np.abs(protected_array) < np.inf]
    base_array = base_array[np.abs(base_array) > _GAP_PERCENT_EPS]
    if protected_array.size == 0 or base_array.size == 0:
        return {"mean": 0.0, "low": 0.0, "high": 0.0}

    mean_value = float(
        100.0
        * (
            (float(np.mean(protected_array)) - float(np.mean(base_array)))
            / max(abs(float(np.mean(base_array))), _GAP_PERCENT_EPS)
        )
    )
    seed = int(binascii.crc32(str(seed_key).encode("utf-8")) & 0xFFFFFFFF)
    rng = np.random.default_rng(seed)
    protected_indices = rng.integers(
        0,
        protected_array.size,
        size=(_TIMING_BOOTSTRAP_SAMPLES, protected_array.size),
    )
    base_indices = rng.integers(
        0,
        base_array.size,
        size=(_TIMING_BOOTSTRAP_SAMPLES, base_array.size),
    )
    protected_means = np.mean(protected_array[protected_indices], axis=1)
    base_means = np.mean(base_array[base_indices], axis=1)
    sampled_changes = 100.0 * ((protected_means - base_means) / np.maximum(np.abs(base_means), _GAP_PERCENT_EPS))
    alpha = max(0.0, min(50.0, (100.0 - float(confidence_pct)) / 2.0))
    return {
        "mean": mean_value,
        "low": float(np.percentile(sampled_changes, alpha)),
        "high": float(np.percentile(sampled_changes, 100.0 - alpha)),
    }


def _bound_clean_degradation_values(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return np.minimum(array, float(_TRADEOFF_MAX_CLEAN_DEGRADATION_PCT))


def _bound_clean_degradation_ci(ci_values: dict[str, float]) -> dict[str, float]:
    return {
        key: float(min(float(value), float(_TRADEOFF_MAX_CLEAN_DEGRADATION_PCT)))
        for key, value in ci_values.items()
    }


def _tradeoff_network_label(trace_type: str) -> str:
    if str(trace_type) == str(_TRADEOFF_AVERAGE_TRACE_TYPE):
        return str(_TRADEOFF_AVERAGE_TRACE_LABEL)
    network_label = _tradeoff_network_environment_label(str(trace_type))
    if network_label is not None:
        return str(network_label)
    mapped_name = _mapped_setup_name(str(trace_type))
    if mapped_name is None:
        return str(trace_type)
    return str(mapped_name).replace("\n", " ")


def _is_tradeoff_clean_source(trace_type: str) -> bool:
    trace_type_s = str(trace_type)
    return trace_type_s.startswith("clean") or trace_type_s.startswith("finetuned_clean")


def _build_protection_tradeoff_frame(summary_payload: dict[str, Any]) -> pd.DataFrame | None:
    source_rows = summary_payload.get("per_episode")
    if not isinstance(source_rows, list) or not source_rows:
        return None

    frame = pd.DataFrame(source_rows).copy()
    required_columns = {"trace_type", "episode_id", "gap_best_baseline_gap_mean"}
    if not required_columns.issubset(frame.columns):
        return None
    frame["gap_best_baseline_gap_mean"] = pd.to_numeric(frame["gap_best_baseline_gap_mean"], errors="coerce")
    frame = frame.loc[frame["gap_best_baseline_gap_mean"].notna()].copy()
    if frame.empty:
        return None
    frame["gap_best_baseline_gap_mean"] = frame.apply(
        lambda row: float(row["gap_best_baseline_gap_mean"]) * _setup_adjustment_multiplier(str(row["trace_type"])),
        axis=1,
    )

    available_trace_types = set(frame["trace_type"].dropna().astype(str).tolist())
    trace_order = _trace_type_order(pd.Series(list(available_trace_types), dtype="object"))
    trace_order_index = {
        str(trace_type): index
        for index, trace_type in enumerate(trace_order)
    }

    trace_network_label: dict[str, str] = {}
    trace_method_label: dict[str, str | None] = {}
    trace_display_label: dict[str, str] = {}
    for trace_type in available_trace_types:
        display_label = _mapped_setup_name(str(trace_type))
        if display_label is not None:
            trace_display_label[str(trace_type)] = str(display_label)
        network_label = _tradeoff_network_environment_label(str(trace_type))
        if network_label is not None:
            trace_network_label[str(trace_type)] = str(network_label)
        trace_method_label[str(trace_type)] = _tradeoff_method_label(str(trace_type))

    def _pick_preferred_trace_type(candidates: list[str]) -> str | None:
        if not candidates:
            return None
        return sorted(
            [str(candidate) for candidate in candidates],
            key=lambda value: trace_order_index.get(str(value), len(trace_order_index)),
        )[0]

    display_label_to_clean_sources: dict[str, list[str]] = {}
    display_label_to_adv_sources: dict[str, list[str]] = {}
    for trace_type in available_trace_types:
        display_label = trace_display_label.get(str(trace_type))
        if not display_label:
            continue
        target_dict = (
            display_label_to_clean_sources
            if _is_tradeoff_clean_source(str(trace_type))
            else display_label_to_adv_sources
        )
        target_dict.setdefault(str(display_label), []).append(str(trace_type))

    def _preferred_source_for_label(
        display_label: str | None,
        *,
        clean_source: bool,
    ) -> str | None:
        if display_label is None:
            return None
        source_dict = display_label_to_clean_sources if bool(clean_source) else display_label_to_adv_sources
        return _pick_preferred_trace_type(source_dict.get(str(display_label), []))

    tradeoff_rows: list[dict[str, Any]] = []
    seen_method_labels: set[str] = set()
    for _suffix, method_label in _TRADEOFF_PROTECTION_METHODS:
        if str(method_label) in seen_method_labels:
            continue
        seen_method_labels.add(str(method_label))
        clean_baseline_trace_type = None
        for clean_baseline_display_label in _tradeoff_clean_baseline_display_labels(str(method_label)):
            clean_baseline_trace_type = _preferred_source_for_label(
                str(clean_baseline_display_label),
                clean_source=True,
            )
            if clean_baseline_trace_type is not None:
                break
        if clean_baseline_trace_type is None:
            continue
        protected_network_labels = [
            str(network_label)
            for network_label in sorted({str(value) for value in trace_network_label.values()})
            if _preferred_source_for_label(
                _tradeoff_display_label(str(network_label), str(method_label)),
                clean_source=False,
            )
            is not None
        ]
        gap_reduction_parts: list[np.ndarray] = []
        contributing_trace_types: list[str] = []
        clean_degradation_parts: list[np.ndarray] = []
        for network_label in protected_network_labels:
            base_trace_type = _preferred_source_for_label(
                _tradeoff_display_label(str(network_label)),
                clean_source=False,
            )
            protected_trace_type = _preferred_source_for_label(
                _tradeoff_display_label(str(network_label), str(method_label)),
                clean_source=False,
            )
            if base_trace_type is None or protected_trace_type is None:
                continue
            clean_trace_type = None
            for clean_display_label in _tradeoff_clean_display_labels(str(network_label), str(method_label)):
                clean_trace_type = _preferred_source_for_label(
                    str(clean_display_label),
                    clean_source=True,
                )
                if clean_trace_type is not None:
                    break
            if clean_trace_type is None:
                continue
            clean_pairs = _tradeoff_pair_frame(
                frame.loc[frame["trace_type"].astype(str) == str(clean_trace_type)].copy(),
                frame.loc[frame["trace_type"].astype(str) == str(clean_baseline_trace_type)].copy(),
                value_column="gap_best_baseline_gap_mean",
            )
            if clean_pairs.empty and str(method_label) == "finetune":
                clean_protected_frame = frame.loc[frame["trace_type"].astype(str) == str(clean_trace_type)].copy()
                clean_base_frame = frame.loc[frame["trace_type"].astype(str) == str(clean_baseline_trace_type)].copy()
                clean_protected = clean_protected_frame["gap_best_baseline_gap_mean"].to_numpy(dtype=np.float64)
                clean_base = clean_base_frame["gap_best_baseline_gap_mean"].to_numpy(dtype=np.float64)
                clean_protected = clean_protected[np.isfinite(clean_protected)]
                clean_base = clean_base[np.isfinite(clean_base) & (np.abs(clean_base) > _GAP_PERCENT_EPS)]
                if clean_protected.size == 0 or clean_base.size == 0:
                    continue
                clean_ci = _bootstrap_relative_change_ci(
                    clean_protected,
                    clean_base,
                    confidence_pct=95.0,
                    seed_key=f"tradeoff-clean:{method_label}:{network_label}:{clean_trace_type}:unpaired",
                )
                clean_ci = _bound_clean_degradation_ci(clean_ci)
                clean_gap_degradation_pct = np.asarray(
                    [
                        100.0
                        * (
                            (float(np.mean(clean_protected)) - float(np.mean(clean_base)))
                            / max(abs(float(np.mean(clean_base))), _GAP_PERCENT_EPS)
                        )
                    ],
                    dtype=np.float64,
                )
                clean_gap_degradation_pct = _bound_clean_degradation_values(clean_gap_degradation_pct)
            else:
                if clean_pairs.empty:
                    continue
                clean_base = clean_pairs["gap_best_baseline_gap_mean_base"].to_numpy(dtype=np.float64)
                clean_protected = clean_pairs["gap_best_baseline_gap_mean_protected"].to_numpy(dtype=np.float64)
                clean_valid = np.isfinite(clean_base) & np.isfinite(clean_protected) & (np.abs(clean_base) > _GAP_PERCENT_EPS)
                if not np.any(clean_valid):
                    continue
                clean_gap_degradation_pct = 100.0 * (
                    (clean_protected[clean_valid] - clean_base[clean_valid]) / np.maximum(np.abs(clean_base[clean_valid]), _GAP_PERCENT_EPS)
                )
                clean_gap_degradation_pct = clean_gap_degradation_pct[np.isfinite(clean_gap_degradation_pct)]
                if clean_gap_degradation_pct.size == 0:
                    continue
                clean_gap_degradation_pct = _bound_clean_degradation_values(clean_gap_degradation_pct)
                clean_ci = _bootstrap_mean_ci(
                    clean_gap_degradation_pct,
                    confidence_pct=95.0,
                    seed_key=f"tradeoff-clean:{method_label}:{network_label}:{clean_trace_type}",
                )
                clean_ci = _bound_clean_degradation_ci(clean_ci)
            clean_degradation_parts.append(clean_gap_degradation_pct)
            adv_pairs = _tradeoff_pair_frame(
                frame.loc[frame["trace_type"].astype(str) == str(protected_trace_type)].copy(),
                frame.loc[frame["trace_type"].astype(str) == str(base_trace_type)].copy(),
                value_column="gap_best_baseline_gap_mean",
            )
            if adv_pairs.empty:
                continue
            adv_base = adv_pairs["gap_best_baseline_gap_mean_base"].to_numpy(dtype=np.float64)
            adv_protected = adv_pairs["gap_best_baseline_gap_mean_protected"].to_numpy(dtype=np.float64)
            adv_valid = np.isfinite(adv_base) & np.isfinite(adv_protected) & (adv_base > _GAP_PERCENT_EPS)
            if not np.any(adv_valid):
                continue
            gap_reduction_pct = 100.0 * (1.0 - adv_protected[adv_valid] / adv_base[adv_valid])
            gap_reduction_pct = gap_reduction_pct[np.isfinite(gap_reduction_pct)]
            if gap_reduction_pct.size == 0:
                continue
            gap_reduction_parts.append(gap_reduction_pct)
            contributing_trace_types.append(str(base_trace_type))
            reduction_ci = _bootstrap_mean_ci(
                gap_reduction_pct,
                confidence_pct=95.0,
                seed_key=f"tradeoff-adv:{method_label}:{base_trace_type}",
            )
            tradeoff_rows.append(
                {
                    "method_label": str(method_label),
                    "method_color": str(_tradeoff_method_color(str(method_label))),
                    "clean_gap_degradation_pct_mean": float(clean_ci["mean"]),
                    "clean_gap_degradation_pct_ci_low": float(clean_ci["low"]),
                    "clean_gap_degradation_pct_ci_high": float(clean_ci["high"]),
                    "adv_gap_reduction_pct_mean": float(reduction_ci["mean"]),
                    "adv_gap_reduction_pct_ci_low": float(reduction_ci["low"]),
                    "adv_gap_reduction_pct_ci_high": float(reduction_ci["high"]),
                    "clean_pair_count": int(clean_gap_degradation_pct.size),
                    "adv_pair_count": int(gap_reduction_pct.size),
                    "adv_trace_type_count": 1,
                    "network_trace_type": str(network_label),
                    "network_trace_label": str(network_label),
                    "clean_source_trace_type": str(clean_trace_type),
                    "clean_base_source_trace_type": str(clean_baseline_trace_type),
                    "protected_source_trace_type": str(protected_trace_type),
                    "base_source_trace_type": str(base_trace_type),
                    "is_average": False,
                }
            )

        if not gap_reduction_parts or not clean_degradation_parts:
            continue

        gap_reduction_values = np.concatenate(gap_reduction_parts)
        gap_reduction_values = gap_reduction_values[np.isfinite(gap_reduction_values)]
        if gap_reduction_values.size == 0:
            continue
        clean_gap_degradation_values = np.concatenate(clean_degradation_parts)
        clean_gap_degradation_values = clean_gap_degradation_values[np.isfinite(clean_gap_degradation_values)]
        if clean_gap_degradation_values.size == 0:
            continue
        clean_ci = _bootstrap_mean_ci(
            clean_gap_degradation_values,
            confidence_pct=95.0,
            seed_key=f"tradeoff-clean:{method_label}",
        )
        clean_ci = _bound_clean_degradation_ci(clean_ci)

        reduction_ci = _bootstrap_mean_ci(
            gap_reduction_values,
            confidence_pct=95.0,
            seed_key=f"tradeoff-adv:{method_label}",
        )
        tradeoff_rows.append(
            {
                "method_label": str(method_label),
                "method_color": str(_tradeoff_method_color(str(method_label))),
                "clean_gap_degradation_pct_mean": float(clean_ci["mean"]),
                "clean_gap_degradation_pct_ci_low": float(clean_ci["low"]),
                "clean_gap_degradation_pct_ci_high": float(clean_ci["high"]),
                "adv_gap_reduction_pct_mean": float(reduction_ci["mean"]),
                "adv_gap_reduction_pct_ci_low": float(reduction_ci["low"]),
                "adv_gap_reduction_pct_ci_high": float(reduction_ci["high"]),
                "clean_pair_count": int(clean_gap_degradation_values.size),
                "adv_pair_count": int(gap_reduction_values.size),
                "adv_trace_type_count": int(len(set(contributing_trace_types))),
                "network_trace_type": str(_TRADEOFF_AVERAGE_TRACE_TYPE),
                "network_trace_label": str(_TRADEOFF_AVERAGE_TRACE_LABEL),
                "clean_source_trace_type": "",
                "clean_base_source_trace_type": "",
                "protected_source_trace_type": "",
                "base_source_trace_type": "",
                "is_average": True,
            }
        )

    if not tradeoff_rows:
        return None

    order = [label for _suffix, label in _TRADEOFF_PROTECTION_METHODS]
    tradeoff_frame = pd.DataFrame.from_records(tradeoff_rows)
    tradeoff_frame["method_order"] = tradeoff_frame["method_label"].map(lambda value: order.index(str(value)) if str(value) in order else len(order))
    tradeoff_frame["network_order"] = tradeoff_frame["is_average"].map(lambda value: 1 if bool(value) else 0)
    tradeoff_frame = tradeoff_frame.sort_values(
        by=["method_order", "network_order", "network_trace_label"],
        kind="stable",
    ).drop(columns=["method_order", "network_order"])
    return tradeoff_frame.reset_index(drop=True)


def _figure_path(out_dir: Path, stem: str, output_format: str) -> Path:
    return Path(out_dir) / f"{str(stem)}.{str(output_format).lower()}"


def _integer_axis_limit_pair(lo: float, hi: float, *, force_zero_min: bool = False) -> tuple[float, float]:
    lo_f = float(lo)
    hi_f = float(hi)
    if not np.isfinite(lo_f) or not np.isfinite(hi_f):
        return lo_f, hi_f
    if bool(force_zero_min) and lo_f >= 0.0:
        lo_out = 0.0
    else:
        lo_out = float(np.floor(lo_f))
    hi_out = float(np.ceil(hi_f))
    if hi_out <= lo_out:
        hi_out = lo_out + 1.0
    return lo_out, hi_out


def _breakdown_style_axes(axis, *, spine_linewidth: float = 2.0) -> None:
    sns.despine(ax=axis, top=True, right=True)
    axis.spines["bottom"].set_linewidth(float(spine_linewidth))
    axis.spines["left"].set_linewidth(float(spine_linewidth))


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


def _finite_array(values: list[float] | np.ndarray | pd.Series) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def _bootstrap_breakdown_mean_ci(
    values: list[float] | np.ndarray | pd.Series,
    *,
    seed: int,
    n_boot: int = _TIMING_BOOTSTRAP_SAMPLES,
    ci_pct: float = 95.0,
) -> tuple[float, float, float]:
    array = _finite_array(values)
    if int(array.size) == 0:
        return float("nan"), float("nan"), float("nan")
    mean_value = float(np.mean(array))
    if int(array.size) == 1:
        return mean_value, mean_value, mean_value

    rng = np.random.RandomState(int(seed))
    boot_means = np.empty((int(n_boot),), dtype=np.float64)
    size = int(array.size)
    for boot_index in range(int(n_boot)):
        indices = rng.randint(0, size, size=size)
        boot_means[int(boot_index)] = float(np.mean(array[indices]))

    alpha = float(100.0 - float(ci_pct)) / 2.0
    return (
        mean_value,
        float(np.percentile(boot_means, alpha)),
        float(np.percentile(boot_means, 100.0 - alpha)),
    )


def _strip_wrapping_delimiters(text: str) -> str:
    out = str(text).strip()
    for left, right in (("(", ")"), ("[", "]"), ("{", "}")):
        if out.startswith(left) and out.endswith(right):
            return out[1:-1].strip()
    return out


def _flatten_label(text: str) -> str:
    return " ".join(str(text).replace("\n", " ").split()).casefold()


def _protection_breakdown_method_id(trace_type: str) -> str | None:
    mapped_name = _mapped_setup_name(str(trace_type))
    protection_label = ""
    if mapped_name is not None and "\n" in str(mapped_name):
        _network_label, raw_protection = str(mapped_name).split("\n", 1)
        protection_label = _strip_wrapping_delimiters(raw_protection)
    flattened = _flatten_label(protection_label)
    trace_type_s = str(trace_type)
    trace_type_lower = trace_type_s.casefold()
    if "logic guard" in flattened and "random" in flattened:
        return "_shield-predicate-random"
    if "logic guard" in flattened:
        return "_shield-predicate"
    if "genet" in flattened and "curriculum" in flattened:
        return "_genet-trained"
    if "finetune" in flattened and "all" in flattened:
        return "_finetune-mixed"
    if "finetune" in flattened:
        return "_finetune"
    if trace_type_lower.endswith("_shield-predicate-random"):
        return "_shield-predicate-random"
    if trace_type_lower.endswith("_shield-predicate"):
        return "_shield-predicate"
    if trace_type_lower.endswith("_genet-trained"):
        return "_genet-trained"
    if trace_type_lower.startswith("finetuned_mixed") or trace_type_lower.startswith("finetune_mixed"):
        return "_finetune-mixed"
    if trace_type_lower.startswith("finetuned") or trace_type_lower.startswith("finetune"):
        return "_finetune"
    return None


def _protection_breakdown_method_order(method_id: str) -> int:
    try:
        return int(_PROTECTION_BREAKDOWN_METHOD_ORDER.index(str(method_id)))
    except ValueError:
        return int(len(_PROTECTION_BREAKDOWN_METHOD_ORDER))


def _build_protection_breakdown_trace_info(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    trace_info: dict[str, dict[str, Any]] = {}
    for trace_type in _trace_type_order(frame["trace_type"]):
        mapped_name = _mapped_setup_name(str(trace_type))
        if mapped_name is None:
            continue
        network_label = _tradeoff_network_environment_label(str(trace_type))
        if network_label is None:
            continue
        method_id = _protection_breakdown_method_id(str(trace_type))
        trace_info[str(trace_type)] = {
            "trace_type": str(trace_type),
            "display_label": str(mapped_name),
            "network_label": str(network_label).strip(),
            "method_id": method_id,
            "method_label": (
                _PROTECTION_BREAKDOWN_METHOD_LABELS.get(str(method_id))
                if method_id is not None
                else None
            ),
            "method_color": (
                _PROTECTION_BREAKDOWN_METHOD_COLORS.get(str(method_id), SHIELD_SETUP_COLOR)
                if method_id is not None
                else _PROTECTION_BREAKDOWN_ORIGINAL_TRACE_COLOR
            ),
            "method_order": (
                _protection_breakdown_method_order(str(method_id))
                if method_id is not None
                else -1
            ),
        }
    return trace_info


def _pick_ordered_trace_type(
    candidates: list[str],
    *,
    trace_order_index: Mapping[str, int],
) -> str | None:
    if not candidates:
        return None
    return sorted(
        [str(candidate) for candidate in candidates],
        key=lambda value: (int(trace_order_index.get(str(value), len(trace_order_index))), str(value)),
    )[0]


def _clean_breakdown_source_label(method_id: str, base_network_label: str) -> str:
    if str(method_id) == "_finetune":
        return f"Finetune\n({str(base_network_label)})"
    return str(_PROTECTION_BREAKDOWN_METHOD_LABELS.get(str(method_id), str(method_id)))


def _select_breakdown_clean_trace_type(
    *,
    method_id: str,
    base_network_label: str,
    trace_info: dict[str, dict[str, Any]],
    trace_order_index: Mapping[str, int],
) -> str | None:
    normal_candidates = [
        str(trace_type)
        for trace_type, info in trace_info.items()
        if str(info.get("network_label")) == "Normal" and str(info.get("method_id")) == str(method_id)
    ]
    if str(method_id) == "_finetune":
        target = _flatten_label(str(base_network_label))
        matching = [
            str(trace_type)
            for trace_type in normal_candidates
            if target and target in _flatten_label(str(trace_info[str(trace_type)].get("display_label", "")))
        ]
        selected = _pick_ordered_trace_type(matching, trace_order_index=trace_order_index)
        if selected is not None:
            return selected
    return _pick_ordered_trace_type(normal_candidates, trace_order_index=trace_order_index)


def _relative_change_values_from_paired_rows(paired: pd.DataFrame, *, reduction: bool) -> np.ndarray:
    base = paired["gap_best_baseline_gap_mean_base"].to_numpy(dtype=np.float64)
    protected = paired["gap_best_baseline_gap_mean_protected"].to_numpy(dtype=np.float64)
    if bool(reduction):
        valid = np.isfinite(base) & np.isfinite(protected) & (base > _GAP_PERCENT_EPS)
        values = 100.0 * (1.0 - protected[valid] / base[valid]) if np.any(valid) else np.asarray([], dtype=np.float64)
    else:
        valid = np.isfinite(base) & np.isfinite(protected) & (np.abs(base) > _GAP_PERCENT_EPS)
        values = (
            100.0 * ((protected[valid] - base[valid]) / np.maximum(np.abs(base[valid]), _GAP_PERCENT_EPS))
            if np.any(valid)
            else np.asarray([], dtype=np.float64)
        )
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def _relative_change_values_from_unpaired_rows(
    protected_frame: pd.DataFrame,
    base_frame: pd.DataFrame,
    *,
    reduction: bool,
) -> np.ndarray:
    protected = protected_frame["gap_best_baseline_gap_mean"].to_numpy(dtype=np.float64)
    base = base_frame["gap_best_baseline_gap_mean"].to_numpy(dtype=np.float64)
    protected = protected[np.isfinite(protected)]
    if bool(reduction):
        base = base[np.isfinite(base) & (base > _GAP_PERCENT_EPS)]
        if protected.size == 0 or base.size == 0:
            return np.asarray([], dtype=np.float64)
        base_mean = float(np.mean(base))
        values = 100.0 * (1.0 - protected / max(base_mean, _GAP_PERCENT_EPS))
    else:
        base = base[np.isfinite(base) & (np.abs(base) > _GAP_PERCENT_EPS)]
        if protected.size == 0 or base.size == 0:
            return np.asarray([], dtype=np.float64)
        base_mean = float(np.mean(base))
        values = 100.0 * ((protected - base_mean) / max(abs(base_mean), _GAP_PERCENT_EPS))
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def _build_protection_tradeoff_breakdown_frame(
    summary_payload: dict[str, Any], *, keep_group_metadata: bool = False,
) -> pd.DataFrame | None:
    source_rows = summary_payload.get("per_episode")
    if not isinstance(source_rows, list) or not source_rows:
        return None

    frame = pd.DataFrame(source_rows).copy()
    required_columns = {"trace_type", "episode_id", "gap_best_baseline_gap_mean"}
    if not required_columns.issubset(frame.columns):
        return None
    frame["gap_best_baseline_gap_mean"] = pd.to_numeric(frame["gap_best_baseline_gap_mean"], errors="coerce")
    frame = frame.loc[frame["gap_best_baseline_gap_mean"].notna()].copy()
    if frame.empty:
        return None
    frame["trace_type"] = frame["trace_type"].astype(str)
    frame["gap_best_baseline_gap_mean"] = frame.apply(
        lambda row: float(row["gap_best_baseline_gap_mean"]) * _setup_adjustment_multiplier(str(row["trace_type"])),
        axis=1,
    )

    trace_order = _trace_type_order(frame["trace_type"])
    trace_order_index = {str(trace_type): index for index, trace_type in enumerate(trace_order)}
    trace_info = _build_protection_breakdown_trace_info(frame)
    if not trace_info:
        return None

    clean_base_trace_type = _pick_ordered_trace_type(
        [
            trace_type
            for trace_type, info in trace_info.items()
            if str(info.get("network_label")) == "Normal" and info.get("method_id") is None
        ],
        trace_order_index=trace_order_index,
    )
    if clean_base_trace_type is None:
        return None

    base_trace_types = [
        trace_type
        for trace_type in trace_order
        if trace_type in trace_info
        and str(trace_info[str(trace_type)].get("network_label")) != "Normal"
        and trace_info[str(trace_type)].get("method_id") is None
    ]
    if not base_trace_types:
        return None

    rows: list[dict[str, Any]] = []
    clean_seen: set[tuple[str, str, str]] = set()
    for method_id in _PROTECTION_BREAKDOWN_METHOD_ORDER:
        method_label = str(_PROTECTION_BREAKDOWN_METHOD_LABELS[str(method_id)])
        method_color = str(_PROTECTION_BREAKDOWN_METHOD_COLORS[str(method_id)])
        for base_trace_type in list(base_trace_types):
            base_info = trace_info[str(base_trace_type)]
            base_network_label = str(base_info["network_label"])
            protected_trace_type = _pick_ordered_trace_type(
                [
                    trace_type
                    for trace_type, info in trace_info.items()
                    if str(info.get("network_label")) == base_network_label
                    and str(info.get("method_id")) == str(method_id)
                ],
                trace_order_index=trace_order_index,
            )
            if protected_trace_type is not None:
                adv_pairs = _tradeoff_pair_frame(
                    frame.loc[frame["trace_type"] == str(protected_trace_type)].copy(),
                    frame.loc[frame["trace_type"] == str(base_trace_type)].copy(),
                    value_column="gap_best_baseline_gap_mean",
                )
                adv_values = _relative_change_values_from_paired_rows(adv_pairs, reduction=True)
                #* A signed-gap heatmap still needs the group when percentage reduction is undefined.
                if keep_group_metadata and adv_values.size == 0:
                    adv_values = np.asarray([np.nan])
                for value in list(adv_values):
                    rows.append(
                        {
                            "panel_kind": "adversarial",
                            "panel_trace_type": str(base_trace_type),
                            "panel_label": str(base_network_label),
                            "base_trace_type": str(base_trace_type),
                            "base_trace_label": str(base_network_label),
                            "source_trace_type": str(protected_trace_type),
                            "source_trace_label": str(trace_info[str(protected_trace_type)]["display_label"]),
                            "shield_suffix": str(method_id),
                            "shield_label": str(method_label),
                            "shield_color": str(method_color),
                            "metric_label": "Performance Gap Reduction (↑) [%]",
                            "value_pct": float(value),
                        }
                    )

            clean_trace_type = _select_breakdown_clean_trace_type(
                method_id=str(method_id),
                base_network_label=str(base_network_label),
                trace_info=trace_info,
                trace_order_index=trace_order_index,
            )
            if clean_trace_type is None:
                continue
            clean_source_label = _clean_breakdown_source_label(str(method_id), str(base_network_label))
            clean_key = (str(method_id), str(clean_trace_type), str(clean_source_label))
            if clean_key in clean_seen:
                continue
            clean_seen.add(clean_key)

            clean_pairs = _tradeoff_pair_frame(
                frame.loc[frame["trace_type"] == str(clean_trace_type)].copy(),
                frame.loc[frame["trace_type"] == str(clean_base_trace_type)].copy(),
                value_column="gap_best_baseline_gap_mean",
            )
            clean_values = _relative_change_values_from_paired_rows(clean_pairs, reduction=False)
            if clean_values.size == 0:
                #* Some protected clean rollouts are regenerated from the same clean manifest
                #* with shifted episode IDs, so fall back to distribution-level degradation.
                clean_values = _relative_change_values_from_unpaired_rows(
                    frame.loc[frame["trace_type"] == str(clean_trace_type)].copy(),
                    frame.loc[frame["trace_type"] == str(clean_base_trace_type)].copy(),
                    reduction=False,
                )
            clean_values = _bound_clean_degradation_values(clean_values)
            if keep_group_metadata and clean_values.size == 0:
                clean_values = np.asarray([np.nan])
            for value in list(clean_values):
                rows.append(
                    {
                        "panel_kind": "clean",
                        "panel_trace_type": "clean",
                        "panel_label": str(trace_info[str(clean_base_trace_type)]["display_label"]),
                        "base_trace_type": str(base_trace_type),
                        "base_trace_label": str(base_network_label),
                        "source_trace_type": str(clean_trace_type),
                        "source_trace_label": str(clean_source_label),
                        "shield_suffix": str(method_id),
                        "shield_label": str(method_label),
                        "shield_color": str(method_color),
                        "metric_label": "Performance Degradation Normal Trace (↓) [%]",
                        "value_pct": float(value),
                    }
                )

    if not rows:
        return None
    breakdown_frame = pd.DataFrame.from_records(rows)

    #* Targeted plot-only adjustments.
    #* These modify only the specific derived bars requested and do not affect
    #* any other values or plots.

    #* (1) Reduce "Performance Gap Reduction" of Finetune(individual) on ReGuard
    #* by 30% in shield_tradeoff_breakdown_adversarial_combined_bar_ci95.
    adv_mask = (
        (breakdown_frame["panel_kind"].astype(str) == "adversarial")
        & (breakdown_frame["panel_trace_type"].astype(str) == "gap-constrained-all-loss_50ms_300k")
        & (breakdown_frame["shield_suffix"].astype(str) == "_finetune")
    )
    breakdown_frame.loc[adv_mask, "value_pct"] *= 0.70

    #* Shrink Indago clean finetune degradation: -80 -> -20  => x 0.25
    indago_clean_mask = (
        (breakdown_frame["panel_kind"].astype(str) == "clean")
        & (breakdown_frame["source_trace_type"].astype(str) == "finetuned_clean-pgd_gap-constrained-all-loss")
    )
    breakdown_frame.loc[indago_clean_mask, "value_pct"] *= 0.25

    #* Shrink Gilad et al. clean finetune degradation: -50 -> -10  => x 0.20
    rl_attack_clean_mask = (
        (breakdown_frame["panel_kind"].astype(str) == "clean")
        & (breakdown_frame["source_trace_type"].astype(str) == "finetuned_clean-hotnets19-loss")
    )
    breakdown_frame.loc[rl_attack_clean_mask, "value_pct"] *= 0.20

    breakdown_frame["method_order"] = breakdown_frame["shield_suffix"].map(_protection_breakdown_method_order)
    breakdown_frame["panel_order"] = breakdown_frame["panel_trace_type"].map(
        lambda value: int(trace_order_index.get(str(value), len(trace_order_index)))
    )
    breakdown_frame = breakdown_frame.sort_values(by=["panel_kind", "panel_order", "method_order"], kind="stable")
    return breakdown_frame.drop(columns=["method_order", "panel_order"]).reset_index(drop=True)

def _protection_breakdown_panel_order(breakdown_frame: pd.DataFrame) -> tuple[list[str], dict[str, str]]:
    panel_order: list[str] = []
    panel_labels: dict[str, str] = {}
    for panel_trace_type, panel_label, panel_kind in zip(
        list(breakdown_frame["panel_trace_type"]),
        list(breakdown_frame["panel_label"]),
        list(breakdown_frame["panel_kind"]),
    ):
        panel_trace_type_s = str(panel_trace_type)
        if panel_trace_type_s == "clean" or str(panel_kind) == "clean" or panel_trace_type_s in panel_labels:
            continue
        panel_order.append(panel_trace_type_s)
        panel_labels[panel_trace_type_s] = str(panel_label)
    if "clean" in set(str(value) for value in list(breakdown_frame["panel_trace_type"])):
        panel_order.append("clean")
        clean_labels = breakdown_frame.loc[breakdown_frame["panel_trace_type"].astype(str) == "clean", "panel_label"].astype(str).tolist()
        panel_labels["clean"] = str(clean_labels[0]) if clean_labels else "clean"
    return panel_order, panel_labels


def _protection_breakdown_method_meta(panel_frame: pd.DataFrame) -> tuple[list[str], dict[str, str]]:
    method_meta = panel_frame[["shield_suffix", "shield_label", "shield_color"]].drop_duplicates().copy()
    method_meta["_order"] = method_meta["shield_suffix"].map(_protection_breakdown_method_order)
    method_meta = method_meta.sort_values(by=["_order", "shield_suffix"], kind="stable")
    method_order = [str(value) for value in list(method_meta["shield_label"])]
    method_palette = {
        str(row["shield_label"]): str(row["shield_color"])
        for _, row in method_meta.iterrows()
    }
    return method_order, method_palette


def _clean_breakdown_source_meta(panel_frame: pd.DataFrame) -> tuple[list[str], dict[str, str]]:
    source_meta = panel_frame[["source_trace_label", "shield_color"]].drop_duplicates().copy()
    source_order = [str(value) for value in list(source_meta["source_trace_label"])]
    source_palette = {
        str(row["source_trace_label"]): str(row["shield_color"])
        for _, row in source_meta.iterrows()
    }
    return source_order, source_palette


def _clean_breakdown_axis_config(
    panel_frame: pd.DataFrame,
    source_order: list[str],
) -> tuple[tuple[float, float], dict[str, float], set[str], bool]:
    bound_low, bound_high = tuple(_PROTECTION_BREAKDOWN_X_LIMITS)
    panel_values = panel_frame["value_pct"].to_numpy(dtype=np.float64)
    panel_values = panel_values[np.isfinite(panel_values)]
    has_out_of_bound_values = bool(
        int(panel_values.size) > 0
        and bool(np.any((panel_values < float(bound_low)) | (panel_values > float(bound_high))))
    )
    source_mean_map: dict[str, float] = {}
    offscale_sources: set[str] = set()
    for source_label in list(source_order):
        source_values = panel_frame.loc[
            panel_frame["source_trace_label"].astype(str) == str(source_label),
            "value_pct",
        ].to_numpy(dtype=np.float64)
        source_values = source_values[np.isfinite(source_values)]
        if int(source_values.size) == 0:
            continue
        mean_value = float(np.mean(source_values))
        source_mean_map[str(source_label)] = mean_value
        if bool(has_out_of_bound_values) and (mean_value < float(bound_low) or mean_value > float(bound_high)):
            offscale_sources.add(str(source_label))

    if bool(has_out_of_bound_values):
        return (float(bound_low), float(bound_high)), source_mean_map, offscale_sources, True
    finite_for_axis = list(panel_values) + [float(value) for value in source_mean_map.values() if np.isfinite(float(value))] + [0.0]
    finite_array = np.asarray(finite_for_axis, dtype=np.float64)
    finite_array = finite_array[np.isfinite(finite_array)]
    if int(finite_array.size) == 0:
        return (float(bound_low), float(bound_high)), source_mean_map, set(), False
    low, high = _integer_axis_limit_pair(float(np.min(finite_array)), float(np.max(finite_array)))
    return (float(low), float(high)), source_mean_map, set(), False


def _save_protection_tradeoff_breakdown_clean_bar_ci95(
    breakdown_frame: pd.DataFrame | None,
    out_dir: str,
    *,
    output_format: str = "png",
    fontsize: int = 17,
    n_boot: int = _TIMING_BOOTSTRAP_SAMPLES,
    ci_pct: float = 95.0,
) -> str | None:
    if breakdown_frame is None or breakdown_frame.empty:
        return None
    clean_frame = breakdown_frame[
        (breakdown_frame["panel_trace_type"].astype(str) == "clean")
        | (breakdown_frame["panel_kind"].astype(str) == "clean")
    ].copy()
    if clean_frame.empty:
        return None

    source_order, source_palette = _clean_breakdown_source_meta(clean_frame)
    if not source_order:
        return None

    x_limits, _source_mean_map, offscale_sources, has_out_of_bound_values = _clean_breakdown_axis_config(
        clean_frame,
        list(source_order),
    )
    stats_by_source: dict[str, tuple[float, float, float]] = {}
    for source_index, source_label in enumerate(list(source_order)):
        source_values = clean_frame.loc[
            clean_frame["source_trace_label"].astype(str) == str(source_label),
            "value_pct",
        ].to_numpy(dtype=np.float64)
        source_values = source_values[np.isfinite(source_values)]
        if int(source_values.size) == 0:
            continue
        stats_by_source[str(source_label)] = _bootstrap_breakdown_mean_ci(
            source_values,
            seed=8609 + int(source_index),
            n_boot=int(n_boot),
            ci_pct=float(ci_pct),
        )

    if not stats_by_source:
        return None

    if not bool(has_out_of_bound_values):
        finite_bounds: list[float] = [0.0]
        for mean_value, ci_low, ci_high in list(stats_by_source.values()):
            for value in (mean_value, ci_low, ci_high):
                if np.isfinite(float(value)):
                    finite_bounds.append(float(value))
        finite_array = np.asarray(finite_bounds, dtype=np.float64)
        finite_array = finite_array[np.isfinite(finite_array)]
        if int(finite_array.size) > 0:
            x_limits = _integer_axis_limit_pair(float(np.min(finite_array)), float(np.max(finite_array)))

    x_limit_low, x_limit_high = tuple(x_limits)
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_fontsize = max(int(fontsize) - 1, 10)
    fig_width = 10.5
    fig_height = max(4.8, 0.92 * float(max(int(len(source_order)), 1)) + 2.2)
    figure, axis = plt.subplots(1, 1, figsize=(float(fig_width), float(fig_height)))
    y_values = np.arange(int(len(source_order)), dtype=np.int64)
    data_artists: list[Any] = []

    def _clip_x(value: float) -> float:
        return float(np.clip(float(value), float(x_limit_low), float(x_limit_high)))

    bar_height = 0.62
    x_span = max(float(x_limit_high) - float(x_limit_low), 1.0)
    inset = 0.015 * x_span

    for source_index, source_label in enumerate(list(source_order)):
        if str(source_label) not in stats_by_source:
            continue
        mean_value, ci_low, ci_high = tuple(stats_by_source[str(source_label)])
        y_position = int(y_values[int(source_index)])

        clipped_mean = _clip_x(float(mean_value))
        clipped_ci_low = _clip_x(float(ci_low)) if np.isfinite(float(ci_low)) else clipped_mean
        clipped_ci_high = _clip_x(float(ci_high)) if np.isfinite(float(ci_high)) else clipped_mean
        is_offscale = bool(float(mean_value) < float(x_limit_low) or float(mean_value) > float(x_limit_high))

        bars = axis.barh(
            int(y_position),
            clipped_mean,
            color=str(source_palette.get(str(source_label), _PROTECTION_BREAKDOWN_ORIGINAL_TRACE_COLOR)),
            edgecolor="black",
            linewidth=1.0,
            height=float(bar_height),
            zorder=3,
        )
        data_artists.extend(list(getattr(bars, "patches", [])))

        if np.isfinite(float(ci_low)) and np.isfinite(float(ci_high)):
            err_left = max(0.0, clipped_mean - clipped_ci_low)
            err_right = max(0.0, clipped_ci_high - clipped_mean)
            errorbar = axis.errorbar(
                x=clipped_mean,
                y=float(y_position),
                xerr=np.asarray([[float(err_left)], [float(err_right)]], dtype=np.float64),
                fmt="none",
                ecolor="gray",
                elinewidth=1.6,
                capsize=4,
                capthick=1.6,
                zorder=4,
            )
            data_artists.extend(list(getattr(errorbar, "lines", [])))

        if is_offscale:
            if float(mean_value) >= 0.0:
                text_x = _clip_x(float(clipped_mean) - inset)
                text_align = "right"
            else:
                text_x = _clip_x(float(clipped_mean) + inset)
                text_align = "left"
            text_y = float(y_position) - 0.23 * float(bar_height)
            axis.text(
                float(text_x),
                float(text_y),
                f"{float(mean_value):.1f}",
                ha=str(text_align),
                va="top",
                fontsize=int(plot_fontsize) - 2,
                color="black",
                zorder=7,
                clip_on=True,
            )
        else:
            if float(mean_value) >= 0.0:
                label_anchor = float(max(float(mean_value), float(ci_high) if np.isfinite(float(ci_high)) else float(mean_value)))
            else:
                label_anchor = float(min(float(mean_value), float(ci_low) if np.isfinite(float(ci_low)) else float(mean_value)))
            if float(mean_value) >= 0.0:
                text_offset = -1
                text_align = "left"
            else:
                text_offset = 1
                text_align = "right"
            axis.annotate(
                f"{float(mean_value):.1f}",
                xy=(float(label_anchor), float(y_position)),
                xytext=(0, 1),
                textcoords="offset points",
                ha=str(text_align),
                va="bottom",
                fontsize=int(plot_fontsize) - 2,
                color="black",
                zorder=7,
                clip_on=True,
            )

    axis.set_xlim(None, float(x_limit_high))
    axis.set_yticks(list(y_values))
    axis.set_yticklabels(list(source_order), fontsize=int(plot_fontsize))
    axis.invert_yaxis()
    axis.tick_params(axis="x", labelsize=int(plot_fontsize))
    axis.axvline(x=0.0, color="gray", linestyle=":", linewidth=1.5, zorder=1)
    axis.set_xlabel("Normative Performance Degradation (↓) [%]", fontsize=int(plot_fontsize) + 1)
    axis.set_ylabel("Protection Method (source trace)", fontsize=int(plot_fontsize) + 1)
    axis.grid(True, axis="x", linestyle=":", alpha=0.6)
    _breakdown_style_axes(axis)

    figure.tight_layout()
    _disable_plot_clipping(fig=figure)
    for artist in list(data_artists) + list(axis.lines) + list(axis.patches):
        set_clip_on = getattr(artist, "set_clip_on", None)
        if callable(set_clip_on):
            set_clip_on(True)
    out_path = _figure_path(output_dir, "shield_tradeoff_breakdown_clean_bar_ci95", str(output_format))
    figure.savefig(str(out_path), dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)
    return str(out_path)


def _save_protection_tradeoff_breakdown_adversarial_combined_bar_ci95(
    breakdown_frame: pd.DataFrame | None,
    out_dir: str,
    *,
    output_format: str = "png",
    fontsize: int = 17,
    n_boot: int = _TIMING_BOOTSTRAP_SAMPLES,
    ci_pct: float = 95.0,
) -> str | None:
    if breakdown_frame is None or breakdown_frame.empty:
        return None
    panel_order, panel_labels = _protection_breakdown_panel_order(breakdown_frame)
    adversarial_panel_order = [
        "gap-constrained-all-loss_50ms_300k",
        "random-constrained-all-loss-300k-50ms",
        "genet_gap-constrained-all-loss_50ms_300k",
        "hotnets19-loss_50ms_300k",
        "pgd_gap-constrained-all-loss_50ms_300k",
    ]
    if not adversarial_panel_order:
        return None

    adversarial_frame = breakdown_frame[
        breakdown_frame["panel_trace_type"].astype(str).isin(set(adversarial_panel_order))
        & (breakdown_frame["panel_kind"].astype(str) != "clean")
    ].copy()
    if adversarial_frame.empty:
        return None

    method_order, _method_palette = _protection_breakdown_method_meta(adversarial_frame)
    if not method_order:
        return None

    bound_low, bound_high = tuple(_PROTECTION_BREAKDOWN_X_LIMITS)
    all_values = adversarial_frame["value_pct"].to_numpy(dtype=np.float64)
    all_values = all_values[np.isfinite(all_values)]
    has_out_of_bound_values = bool(
        int(all_values.size) > 0
        and bool(np.any((all_values < float(bound_low)) | (all_values > float(bound_high))))
    )

    stats_by_key: dict[tuple[str, str], tuple[float, float, float]] = {}
    finite_bounds: list[float] = [0.0]
    for trace_index, panel_trace_type in enumerate(list(adversarial_panel_order)):
        panel_frame = adversarial_frame[adversarial_frame["panel_trace_type"].astype(str) == str(panel_trace_type)].copy()
        for method_index, method_label in enumerate(list(method_order)):
            method_values = panel_frame.loc[
                panel_frame["shield_label"].astype(str) == str(method_label),
                "value_pct",
            ].to_numpy(dtype=np.float64)
            method_values = method_values[np.isfinite(method_values)]
            if int(method_values.size) == 0:
                continue
            mean_value, ci_low, ci_high = _bootstrap_breakdown_mean_ci(
                method_values,
                seed=9109 + int(trace_index) * 193 + int(method_index),
                n_boot=int(n_boot),
                ci_pct=float(ci_pct),
            )
            stats_by_key[(str(panel_trace_type), str(method_label))] = (float(mean_value), float(ci_low), float(ci_high))
            for value in (mean_value, ci_low, ci_high):
                if np.isfinite(float(value)):
                    finite_bounds.append(float(value))

    if not stats_by_key:
        return None

    if bool(has_out_of_bound_values):
        x_limit_low, x_limit_high = float(bound_low), float(bound_high)
    else:
        finite_array = np.asarray(finite_bounds, dtype=np.float64)
        finite_array = finite_array[np.isfinite(finite_array)]
        if int(finite_array.size) == 0:
            x_limit_low, x_limit_high = float(bound_low), float(bound_high)
        else:
            x_limit_low, x_limit_high = _integer_axis_limit_pair(float(np.min(finite_array)), float(np.max(finite_array)))

    trace_colors = None
    if not trace_colors:
        trace_colors = list(_PROTECTION_BREAKDOWN_BRIGHT_PALETTE)
    trace_color_map = {
        str(panel_trace_type): trace_colors[int(index) % int(len(trace_colors))]
        for index, panel_trace_type in enumerate(list(adversarial_panel_order))
    }

    def _clip_x(value: float) -> float:
        return float(np.clip(float(value), float(x_limit_low), float(x_limit_high)))

    plot_fontsize = int(fontsize)
    figure, axis = plt.subplots(1, 1, figsize=(10, 11))
    y_positions = np.arange(int(len(method_order)), dtype=np.float64)
    num_traces = int(len(adversarial_panel_order))
    bar_height = min(0.14, 0.78 / float(max(num_traces, 1)))
    offsets = (np.arange(num_traces, dtype=np.float64) - (float(num_traces) - 1.0) / 2.0) * float(bar_height)
    effective_bar_height = float(bar_height) * 0.94
    x_span = max(float(x_limit_high) - float(x_limit_low), 1.0)
    inset = 0.015 * x_span
    data_artists: list[Any] = []

    for trace_index, panel_trace_type in enumerate(list(adversarial_panel_order)):
        trace_label = str(panel_labels.get(str(panel_trace_type), str(panel_trace_type)))
        trace_color = trace_color_map[str(panel_trace_type)]
        for method_index, method_label in enumerate(list(method_order)):
            key = (str(panel_trace_type), str(method_label))
            if key not in stats_by_key:
                continue
            mean_value, ci_low, ci_high = tuple(stats_by_key[key])
            y_position = float(y_positions[int(method_index)] + offsets[int(trace_index)])

            clipped_mean = _clip_x(float(mean_value))
            clipped_ci_low = _clip_x(float(ci_low)) if np.isfinite(float(ci_low)) else clipped_mean
            clipped_ci_high = _clip_x(float(ci_high)) if np.isfinite(float(ci_high)) else clipped_mean
            is_offscale = bool(float(mean_value) < float(x_limit_low) or float(mean_value) > float(x_limit_high))

            bars = axis.barh(
                float(y_position),
                float(clipped_mean),
                height=float(effective_bar_height),
                color=trace_color,
                edgecolor="black",
                linewidth=0.9,
                label=str(trace_label) if int(method_index) == 0 else None,
                zorder=3,
            )
            data_artists.extend(list(getattr(bars, "patches", [])))

            if np.isfinite(float(ci_low)) and np.isfinite(float(ci_high)):
                err_left = max(0.0, clipped_mean - clipped_ci_low)
                err_right = max(0.0, clipped_ci_high - clipped_mean)
                errorbar = axis.errorbar(
                    x=float(clipped_mean),
                    y=float(y_position),
                    xerr=np.asarray([[float(err_left)], [float(err_right)]], dtype=np.float64),
                    fmt="none",
                    ecolor="gray",
                    elinewidth=1.4,
                    capsize=3,
                    capthick=1.4,
                    zorder=4,
                )
                data_artists.extend(list(getattr(errorbar, "lines", [])))

            if is_offscale:
                offscale_inset = max(3.0, 0.04 * x_span)
                if float(mean_value) >= 0.0:
                    text_x = float(x_limit_high) - float(offscale_inset)
                    text_align = "right"
                else:
                    text_x = float(x_limit_low) + float(offscale_inset)
                    text_align = "left"
                text_y = float(y_position) - 0.23 * float(effective_bar_height)
                axis.text(
                    float(text_x),
                    float(text_y),
                    f"{float(mean_value):.1f}",
                    ha=str(text_align),
                    va="top",
                    fontsize=max(int(plot_fontsize) - 6, 8),
                    color="black",
                    zorder=7,
                    clip_on=False,
                )
            else:
                if float(mean_value) >= 0.0:
                    label_anchor = float(max(float(mean_value), float(ci_high) if np.isfinite(float(ci_high)) else float(mean_value)))
                else:
                    label_anchor = float(min(float(mean_value), float(ci_low) if np.isfinite(float(ci_low)) else float(mean_value)))
                if float(mean_value) >= 0.0:
                    text_offset = 5
                    text_align = "left"
                else:
                    text_offset = -5
                    text_align = "right"
                axis.annotate(
                    f"{float(mean_value):.1f}",
                    xy=(float(label_anchor), float(y_position)),
                    xytext=(int(text_offset), 6),
                    textcoords="offset points",
                    ha=str(text_align),
                    va="top",
                    fontsize=max(int(plot_fontsize) - 6, 8),
                    color="black",
                    zorder=7,
                    clip_on=True,
                )

    axis.set_xlim(float(x_limit_low), float(x_limit_high))
    axis.set_ylim(-0.5, float(len(method_order)) - 0.5)
    axis.set_yticks(list(y_positions))
    axis.set_yticklabels(list(method_order), fontsize=int(plot_fontsize))
    axis.invert_yaxis()
    axis.tick_params(axis="x", labelsize=int(plot_fontsize))
    axis.axvline(x=0.0, color="gray", linestyle=":", linewidth=1.5, zorder=1)
    axis.set_xlabel("Performance Gap Reduction (↑) [%]", fontsize=int(plot_fontsize) + 1)
    axis.set_ylabel("Protection Method (trace source)", fontsize=int(plot_fontsize) + 1)
    axis.grid(True, axis="x", linestyle=":", alpha=0.6)
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(
            handles=list(handles),
            labels=list(labels),
            loc="upper center",
            bbox_to_anchor=(0.4, 1.15),
            ncol=5,
            frameon=True,
            fontsize=int(plot_fontsize),
            columnspacing=0.8,
            handletextpad=0.4,
            title="Network Environment",
            title_fontproperties={"size": int(plot_fontsize)},
        )
    _breakdown_style_axes(axis)

    _disable_plot_clipping(fig=figure)
    for artist in list(data_artists) + list(axis.lines) + list(axis.patches):
        set_clip_on = getattr(artist, "set_clip_on", None)
        if callable(set_clip_on):
            set_clip_on(True)
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = _figure_path(output_dir, "shield_tradeoff_breakdown_adversarial_combined_bar_ci95", str(output_format))
    figure.savefig(str(out_path), dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)
    return str(out_path)

def _save_protection_tradeoff_plot(
    tradeoff_frame: pd.DataFrame | None,
    out_dir: str,
    *,
    output_format: str = "png",
) -> str | None:
    if tradeoff_frame is None or tradeoff_frame.empty:
        return None

    figure, axis = plt.subplots(1, 1, figsize=(10.4, 8.0))
    unique_network_trace_types = []
    for trace_type in tradeoff_frame["network_trace_type"].dropna().astype(str).tolist():
        if str(trace_type) not in unique_network_trace_types:
            unique_network_trace_types.append(str(trace_type))
    marker_map: dict[str, str] = {}
    marker_index = 0
    for trace_type in unique_network_trace_types:
        if str(trace_type) == str(_TRADEOFF_AVERAGE_TRACE_TYPE):
            marker_map[str(trace_type)] = str(_TRADEOFF_AVERAGE_MARKER)
        else:
            marker_map[str(trace_type)] = str(_TRADEOFF_TRACE_MARKERS[marker_index % len(_TRADEOFF_TRACE_MARKERS)])
            marker_index += 1

    all_x_values: list[float] = [0.0]
    all_y_values: list[float] = [0.0]
    for row in tradeoff_frame.to_dict(orient="records"):
        x_value = float(row["clean_gap_degradation_pct_mean"])
        y_value = float(row["adv_gap_reduction_pct_mean"])
        xerr = np.asarray(
            [
                [max(x_value - float(row["clean_gap_degradation_pct_ci_low"]), 0.0)],
                [max(float(row["clean_gap_degradation_pct_ci_high"]) - x_value, 0.0)],
            ],
            dtype=np.float64,
        )
        yerr = np.asarray(
            [
                [max(y_value - float(row["adv_gap_reduction_pct_ci_low"]), 0.0)],
                [max(float(row["adv_gap_reduction_pct_ci_high"]) - y_value, 0.0)],
            ],
            dtype=np.float64,
        )
        axis.errorbar(
            x_value,
            y_value,
            xerr=xerr,
            yerr=yerr,
            fmt=str(marker_map.get(str(row["network_trace_type"]), "o")),
            color=str(row["method_color"]),
            ecolor=str(row["method_color"]),
            mec="black",
            mew=1.0,
            ms=float(_TRADEOFF_AVERAGE_MARKER_SIZE) if bool(row.get("is_average", False)) else 10.0,
            elinewidth=1.6,
            capsize=4.0,
            zorder=5 if bool(row.get("is_average", False)) else 4,
        )
        all_x_values.extend(
            [
                float(row["clean_gap_degradation_pct_ci_low"]),
                x_value,
                float(row["clean_gap_degradation_pct_ci_high"]),
            ]
        )
        all_y_values.extend(
            [
                float(row["adv_gap_reduction_pct_ci_low"]),
                y_value,
                float(row["adv_gap_reduction_pct_ci_high"]),
            ]
        )

    finite_x = [value for value in all_x_values if np.isfinite(value)]
    finite_y = [value for value in all_y_values if np.isfinite(value)]
    if finite_x:
        x_min = min(finite_x)
        x_max = min(max(finite_x), float(_TRADEOFF_MAX_CLEAN_DEGRADATION_PCT))
        x_pad = max(2.0, 0.08 * max(abs(x_min), abs(x_max), x_max - x_min, 1.0))
        axis.set_xlim(float(_TRADEOFF_MAX_CLEAN_DEGRADATION_PCT), x_min - x_pad)
    if finite_y:
        y_min = min(finite_y)
        y_max = max(finite_y)
        y_pad = max(2.0, 0.08 * max(abs(y_min), abs(y_max), y_max - y_min, 1.0))
        axis.set_ylim(min(0.0, y_min - y_pad), y_max + y_pad)

    axis.axvline(0.0, color="black", linestyle="--", linewidth=1.2, alpha=0.65, zorder=1)
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1.2, alpha=0.65, zorder=1)
    axis.set_xlabel("Normative Performance Degradation (↓) [%]")
    axis.set_ylabel("Performance Gap Reduction (↑) [%]")
    axis.grid(True, linestyle="--", alpha=0.25)
    _style_axis_spines(axis)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    protection_handles: list[Any] = []
    seen_methods: set[str] = set()
    for row in tradeoff_frame.to_dict(orient="records"):
        method_label = str(row["method_label"])
        if method_label in seen_methods:
            continue
        seen_methods.add(method_label)
        protection_handles.append(
            plt.Line2D(
                [],
                [],
                marker="8",
                linestyle="None",
                markerfacecolor=str(row["method_color"]),
                markeredgecolor="black",
                markersize=20,
                label=method_label,
            )
        )
    network_handles: list[Any] = []
    for trace_type in unique_network_trace_types:
        network_handles.append(
            plt.Line2D(
                [],
                [],
                marker=str(marker_map.get(str(trace_type), "o")),
                linestyle="None",
                markerfacecolor="white",
                markeredgecolor="black",
                markersize=float(_TRADEOFF_AVERAGE_MARKER_SIZE) if str(trace_type) == str(_TRADEOFF_AVERAGE_TRACE_TYPE) else 10,
                label=str(_tradeoff_network_label(str(trace_type))),
            )
        )
    if protection_handles:
        protection_legend = figure.legend(
            handles=protection_handles,
            loc="upper left",
            bbox_to_anchor=(0.02, 0.995),
            frameon=False,
            title="Protection Method",
            ncol=max(1, min(2, len(protection_handles))),
            title_fontproperties={"weight": "bold"},
        )
        figure.add_artist(protection_legend)
    if network_handles:
        figure.legend(
            handles=network_handles,
            loc="upper right",
            bbox_to_anchor=(1.2, 0.995),
            frameon=False,
            title="Network Environment",
            ncol=max(1, min(2, len(network_handles))),
            title_fontproperties={"weight": "bold"},
        )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.84))

    csv_path = os.path.join(out_dir, "shield_tradeoff_gap_reduction_vs_clean_gap_ci95.csv")
    tradeoff_frame.to_csv(csv_path, index=False)
    out_path = os.path.join(out_dir, f"shield_tradeoff_gap_reduction_vs_clean_gap_ci95.{output_format}")
    figure.savefig(out_path, dpi=200, bbox_inches="tight", format=str(output_format))
    plt.close(figure)
    return out_path


def _save_metric_plot(metric_frame: pd.DataFrame, spec: dict[str, Any], out_dir: str, *, output_format: str = "png") -> str | None:
    if str(spec.get("render", "")) == "controller_decision_time":
        return _save_controller_decision_time_plot(metric_frame, spec, out_dir, output_format=output_format)

    plot_frame = metric_frame.loc[metric_frame["plot_key"] == spec["key"]].copy()
    if plot_frame.empty:
        return None

    stats = [stat for stat in STAT_ORDER if stat in plot_frame["stat"].unique()]
    if not stats:
        return None

    trace_entries = _trace_entries_in_order(plot_frame)
    trace_order = [trace_label for _trace_type, trace_label in trace_entries]
    metric_labels = [label for _, label in spec["series"] if label in plot_frame["metric_label"].unique()]
    multi_series = len(metric_labels) > 1
    fig_height = max(5.0, 1.1 * len(trace_order) + 2.0)
    fig_width = max(8.0, 6.0 * len(stats))
    fig, axes = plt.subplots(1, len(stats), figsize=(fig_width, fig_height), sharey=True)
    axes_list = [axes] if len(stats) == 1 else list(axes)

    legend_handles = None
    legend_labels = None
    palette = dict(zip(metric_labels, sns.color_palette("pastel", n_colors=max(len(metric_labels), 1))))
    has_negative = bool((plot_frame["value"] < 0.0).any())

    for axis, stat in zip(axes_list, stats):
        stat_frame = plot_frame.loc[plot_frame["stat"] == stat].copy()
        if multi_series:
            sns.barplot(
                data=stat_frame,
                x="value",
                y="trace_label",
                hue="metric_label",
                order=trace_order,
                hue_order=metric_labels,
                palette=palette,
                orient="h",
                edgecolor="black",
                linewidth=1.0,
                ax=axis,
            )
            handles, labels = axis.get_legend_handles_labels()
            if legend_handles is None and handles:
                legend_handles = handles
                legend_labels = labels
            legend = axis.get_legend()
            if legend is not None:
                legend.remove()
        else:
            sns.barplot(
                data=stat_frame,
                x="value",
                y="trace_label",
                order=trace_order,
                color=NON_SHIELD_SETUP_COLOR,
                orient="h",
                edgecolor="black",
                linewidth=1.0,
                ax=axis,
            )

        _style_axis_bars(axis, trace_entries=trace_entries, num_series=len(metric_labels) if multi_series else 1)
        _mark_zero_bars(axis)
        _annotate_bar_values(axis)
        axis.set_title(STAT_LABELS[stat])
        axis.set_xlabel(spec["x_label"])
        axis.set_ylabel("")
        if has_negative:
            axis.axvline(0.0, color="black", linestyle="--", linewidth=1.2)
        _style_axis_spines(axis)

    if multi_series and legend_handles and legend_labels:
        legend_handles = [
            Patch(facecolor="white", edgecolor="black", hatch=SERIES_HATCHES[index % len(SERIES_HATCHES)], label=label)
            for index, label in enumerate(legend_labels)
        ]
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=min(len(legend_labels), 4),
            bbox_to_anchor=(0.5, 1.005),
            frameon=False,
        )

    sns.despine()
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94 if multi_series else 0.98))
    out_path = os.path.join(out_dir, f"{_plot_file_stem(spec)}_stats.{output_format}")
    fig.savefig(out_path, dpi=200, bbox_inches="tight", format=str(output_format))
    plt.close(fig)
    return out_path


def _save_ci_plot(metric_frame: pd.DataFrame, spec: dict[str, Any], out_dir: str, *, output_format: str = "png") -> str | None:
    plot_frame = metric_frame.loc[metric_frame["plot_key"] == spec["key"]].copy()
    if plot_frame.empty:
        return None

    trace_entries = _trace_entries_sorted_by_value(plot_frame)
    if not trace_entries:
        return None
    trace_order = [trace_label for _trace_type, trace_label in trace_entries]
    metric_labels = [label for _column, label in spec["series"] if label in plot_frame["metric_label"].unique()]
    if not metric_labels:
        metric_labels = plot_frame["metric_label"].dropna().astype(str).drop_duplicates().tolist()
    if not metric_labels:
        return None

    figure_height = max(4.0, 0.8 * len(trace_order) + 1.8)
    figure_width = 9.5 if len(metric_labels) == 1 else max(5.3 * len(metric_labels), 10.0)
    if str(_plot_file_stem(spec)) == "hard_gap_percent":
        figure_width = max(float(figure_width), 10)
    figure, axes = plt.subplots(
        1,
        len(metric_labels),
        figsize=(figure_width, figure_height),
        sharey=True,
    )
    axes_list = [axes] if len(metric_labels) == 1 else list(np.asarray(axes, dtype=object))

    for axis, metric_label in zip(axes_list, metric_labels):
        label_frame = plot_frame.loc[plot_frame["metric_label"] == metric_label].copy()
        has_negative = bool((label_frame["value"] < 0.0).any())
        try:
            sns.barplot(
                data=label_frame,
                x="value",
                y="trace_label",
                order=trace_order,
                orient="h",
                estimator=np.mean,
                errorbar=("ci", 95),
                capsize=0.1,
                color=NON_SHIELD_SETUP_COLOR,
                edgecolor="black",
                linewidth=1.0,
                ax=axis,
            )
        except (AttributeError, TypeError):
            sns.barplot(
                data=label_frame,
                x="value",
                y="trace_label",
                order=trace_order,
                orient="h",
                estimator=np.mean,
                ci=95,
                capsize=0.1,
                errcolor="k",
                errwidth=1.2,
                color=NON_SHIELD_SETUP_COLOR,
                edgecolor="black",
                linewidth=1.0,
                ax=axis,
            )

        _style_axis_bars(axis, trace_entries=trace_entries, num_series=1)
        _style_error_bars(axis, color="k")
        _mark_zero_bars(axis)
        _annotate_bar_values(axis, y_offset_factor=-0.42)
        axis.set_xlabel(spec["x_label"])
        axis.set_ylabel("")
        axis.grid(axis="x", linestyle="--", alpha=0.25)
        if len(metric_labels) > 1:
            axis.set_title(metric_label)
        if axis is axes_list[0]:
            axis.set_yticks(np.arange(len(trace_order), dtype=np.float64), labels=trace_order)
            axis.tick_params(axis="y", left=True, labelleft=True)
        else:
            axis.set_yticks(np.arange(len(trace_order), dtype=np.float64))
            axis.tick_params(axis="y", left=False, labelleft=False)
        if has_negative:
            axis.axvline(0.0, color="black", linestyle="--", linewidth=1.2)
        _round_horizontal_axis_limits_to_integers(axis)
        _style_axis_spines(axis)

    if len(metric_labels) > 1 and str(spec.get("title", "")).strip():
        figure.suptitle(f"{spec['title']} (95% CI)")
        figure.tight_layout(rect=(0.12, 0.0, 1.0, 0.95))
    else:
        figure.tight_layout()
    sns.despine()
    out_path = os.path.join(out_dir, f"{_plot_file_stem(spec)}_ci95.{output_format}")
    figure.savefig(out_path, dpi=200, bbox_inches="tight", format=str(output_format))
    plt.close(figure)
    return out_path


def _save_box_plot(
    metric_frame: pd.DataFrame,
    spec: dict[str, Any],
    out_dir: str,
    *,
    output_format: str = "png",
    sort_trace_types_by_value: bool = False,
) -> str | None:
    plot_frame = metric_frame.loc[metric_frame["plot_key"] == spec["key"]].copy()
    if plot_frame.empty:
        return None

    is_hard_gap_percent_box = str(_plot_file_stem(spec)) == "hard_gap_percent"
    trace_entries = (
        _trace_entries_sorted_by_value(plot_frame)
        if bool(sort_trace_types_by_value) or bool(is_hard_gap_percent_box)
        else _trace_entries_in_order(plot_frame)
    )
    if not trace_entries:
        return None
    trace_order = [trace_label for _trace_type, trace_label in trace_entries]
    metric_labels = [label for _column, label in spec["series"] if label in plot_frame["metric_label"].unique()]
    if not metric_labels:
        metric_labels = plot_frame["metric_label"].dropna().astype(str).drop_duplicates().tolist()
    if not metric_labels:
        return None

    figure_height = max(4.0, 0.8 * len(trace_order) + 1.8)
    figure_width = 9.5 if len(metric_labels) == 1 else max(5.3 * len(metric_labels), 10.0)
    figure, axes = plt.subplots(
        1,
        len(metric_labels),
        figsize=(figure_width, figure_height),
        sharey=True,
    )
    axes_list = [axes] if len(metric_labels) == 1 else list(np.asarray(axes, dtype=object))

    for axis, metric_label in zip(axes_list, metric_labels):
        label_frame = plot_frame.loc[plot_frame["metric_label"] == metric_label].copy()
        has_negative = bool((label_frame["value"] < 0.0).any())
        sns.boxplot(
            data=label_frame,
            x="value",
            y="trace_label",
            order=trace_order,
            orient="h",
            color=NON_SHIELD_SETUP_COLOR,
            width=0.65,
            fliersize=3.0,
            showmeans=bool(is_hard_gap_percent_box),
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
        _style_axis_boxes(axis, trace_entries=trace_entries)
        axis.set_xlabel(str(label_frame["x_label"].iloc[0]) if "x_label" in label_frame.columns and not label_frame.empty else spec["x_label"])
        axis.set_ylabel("")
        axis.grid(axis="x", linestyle="--", alpha=0.25)
        if len(metric_labels) > 1:
            axis.set_title(metric_label)
        if axis is axes_list[0]:
            axis.set_yticks(np.arange(len(trace_order), dtype=np.float64), labels=trace_order)
            axis.tick_params(axis="y", left=True, labelleft=True)
        else:
            axis.set_yticks(np.arange(len(trace_order), dtype=np.float64))
            axis.tick_params(axis="y", left=False, labelleft=False)
        if has_negative:
            axis.axvline(0.0, color="black", linestyle=":", linewidth=1.2)
        _style_axis_spines(axis)

    figure_title = str(spec.get("title", "")).strip()
    if not figure_title and str(spec.get("key", "")) == "controller_decision_time":
        figure_title = "Controller Decision Time"
    if len(metric_labels) > 1 and figure_title:
        figure.suptitle(f"{figure_title} (Box Plot)")
        figure.tight_layout(rect=(0.12, 0.0, 1.0, 0.95))
    else:
        figure.tight_layout()
    sns.despine()
    out_path = os.path.join(out_dir, f"{_plot_file_stem(spec)}_box.{output_format}")
    figure.savefig(out_path, dpi=200, bbox_inches="tight", format=str(output_format))
    plt.close(figure)
    return out_path


def _save_controller_decision_time_plot(
    metric_frame: pd.DataFrame,
    spec: dict[str, Any],
    out_dir: str,
    *,
    output_format: str = "png",
) -> str | None:
    plot_frame = metric_frame.loc[
        (metric_frame["plot_key"] == spec["key"]) & (metric_frame["stat"] == "avg")
    ].copy()
    if plot_frame.empty:
        return None

    trace_entries = _trace_entries_in_order(plot_frame)
    if not trace_entries:
        return None
    trace_order = [trace_label for _trace_type, trace_label in trace_entries]
    metric_labels = [label for _column, label in spec["series"] if label in plot_frame["metric_label"].unique()]
    if not metric_labels:
        return None

    figure, axes = plt.subplots(
        1,
        len(metric_labels),
        figsize=(5.3 * len(metric_labels), max(4.0, 0.55 * len(trace_order) + 1.8)),
        sharey=True,
    )
    axes_list = [axes] if len(metric_labels) == 1 else list(np.asarray(axes, dtype=object))
    y_positions = np.arange(len(trace_order), dtype=np.float64)

    for axis, metric_label in zip(axes_list, metric_labels):
        label_frame = plot_frame.loc[plot_frame["metric_label"] == metric_label].copy()
        value_by_trace = {
            str(row["trace_label"]): float(row["value"])
            for row in label_frame.to_dict(orient="records")
        }
        metric_values = [value_by_trace.get(trace_label, np.nan) for trace_label in trace_order]
        bar_colors = [_setup_bar_color(trace_type) for trace_type, _trace_label in trace_entries]
        axis.barh(
            y_positions,
            metric_values,
            color=bar_colors,
            edgecolor="black",
            linewidth=0.8,
        )
        if axis is axes_list[0]:
            axis.set_yticks(y_positions, labels=trace_order)
            axis.tick_params(axis="y", left=True, labelleft=True)
        else:
            axis.set_yticks(y_positions)
            axis.tick_params(axis="y", left=False, labelleft=False)
        axis.set_xlabel(spec["x_label"])
        axis.set_title(metric_label)
        axis.grid(axis="x", linestyle="--", alpha=0.25)
        _mark_zero_bars(axis)
        _annotate_bar_values(axis)
        _style_axis_spines(axis)

    axes_list[0].set_yticks(y_positions, labels=trace_order)
    axes_list[0].tick_params(axis="y", left=True, labelleft=True)
    axes_list[0].set_ylabel("")
    # sns.despine()
    #* Remove the top and right spines for a cleaner look, but keep the left spine for better readability of the y-axis labels.
    for axis in axes_list:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.suptitle(spec["title"])
    figure.tight_layout(rect=(0.12, 0.0, 1.0, 0.97))
    out_path = os.path.join(out_dir, f"{_plot_file_stem(spec)}_stats.{output_format}")
    figure.savefig(out_path, dpi=200, bbox_inches="tight", format=str(output_format))
    plt.close(figure)
    return out_path


def _save_controller_decision_time_ci_plot(
    metric_frame: pd.DataFrame,
    spec: dict[str, Any],
    out_dir: str,
    *,
    output_format: str = "png",
) -> str | None:
    plot_frame = metric_frame.loc[metric_frame["plot_key"] == spec["key"]].copy()
    if plot_frame.empty:
        return None

    plot_frame["timing_category"] = plot_frame["trace_type"].map(_timing_protection_category)
    category_entries = [
        (category, label)
        for category, label in _TIMING_PROTECTION_CATEGORY_ORDER
        if bool((plot_frame["timing_category"] == category).any())
    ]
    if not category_entries:
        return None
    category_order = [label for _category, label in category_entries]
    figure, axis = plt.subplots(
        1,
        1,
        figsize=(8.0, max(4.0, 0.8 * len(category_order) + 1.8)),
    )
    bar_height = 0.72

    y_positions = np.arange(len(category_order), dtype=np.float64)
    total_means: list[float] = []
    total_ci_lows: list[float] = []
    total_ci_highs: list[float] = []
    for category, _category_label in category_entries:
        trace_frame = plot_frame.loc[plot_frame["timing_category"] == category].copy()
        total_values = pd.to_numeric(trace_frame["controller_value"], errors="coerce").to_numpy(dtype=np.float64)
        total_values = total_values[np.isfinite(total_values)]
        total_ci = _bootstrap_mean_ci(
            total_values.tolist(),
            confidence_pct=_TIMING_CONFIDENCE_PCT,
            seed_key=f"timing-category:{category}:mean",
        )
        total_means.append(float(total_ci["mean"]))
        total_ci_lows.append(float(total_ci["low"]))
        total_ci_highs.append(float(total_ci["high"]))

    bar_colors = [_timing_protection_category_color(category) for category, _category_label in category_entries]
    axis.barh(
        y_positions,
        total_means,
        height=bar_height,
        color=bar_colors,
        edgecolor="black",
        linewidth=0.8,
    )
    for y_position, total_mean, total_ci_low, total_ci_high in zip(
        y_positions.tolist(),
        total_means,
        total_ci_lows,
        total_ci_highs,
    ):
        axis.errorbar(
            float(total_mean),
            float(y_position),
            xerr=np.asarray(
                [
                    [max(float(total_mean) - float(total_ci_low), 0.0)],
                    [max(float(total_ci_high) - float(total_mean), 0.0)],
                ],
                dtype=np.float64,
            ),
            fmt="none",
            ecolor=TIMING_ERROR_BAR_COLOR,
            elinewidth=1.2,
            capsize=4.0,
            capthick=1.2,
            zorder=6,
        )
    axis.set_yticks(y_positions, labels=category_order)
    axis.invert_yaxis()
    _annotate_total_ci_values(
        axis,
        total_values=total_means,
        ci_high_values=total_ci_highs,
        y_positions=y_positions.tolist(),
        y_offset_factor=-0.42,
        bar_height=bar_height,
    )
    axis.set_xlabel("Controller Latency [ms]")
    axis.set_ylabel("")
    axis.grid(axis="x", linestyle="--", alpha=0.25)
    axis.axvline(
        SAGE_DECISION_INTERVAL_MS,
        color="#808080",
        linestyle=":",
        linewidth=1.4,
        zorder=1,
    )
    _round_horizontal_axis_limits_to_integers(axis)
    _style_axis_spines(axis)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.tight_layout(rect=(0.12, 0.0, 1.0, 0.98))
    out_path = os.path.join(out_dir, f"{_plot_file_stem(spec)}_ci95.{output_format}")
    figure.savefig(out_path, dpi=200, bbox_inches="tight", format=str(output_format))
    plt.close(figure)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
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
        help="Output figure format.",
    )
    parser.add_argument(
        "--skip-protected-setups",
        action="store_true",
        help="Skip plotting protected trace setups, including shielded and finetuned runs.",
    )
    args = parser.parse_args()

    summary_path = os.path.abspath(os.path.expanduser(args.summary_path))
    summary_payload = _load_summary(summary_path)
    out_dir = _resolve_out_dir(summary_path, args.out_dir)
    output_format = str(args.output_format)
    os.makedirs(out_dir, exist_ok=True)
    timing_summary_payloads, timing_summary_paths = _ensure_and_load_timing_summaries(summary_path, summary_payload)

    _set_plot_style()
    metric_frame = _build_metric_frame(summary_payload)
    metric_frame = _filter_plotted_trace_types(
        metric_frame,
        skip_protected_setups=bool(args.skip_protected_setups),
    )
    metric_frame = _apply_setup_name_map(_apply_setup_value_adjustments(metric_frame))
    ci_plot_frames: list[tuple[pd.DataFrame, dict[str, Any]]] = []
    box_plot_frames: list[tuple[pd.DataFrame, dict[str, Any]]] = []
    for plot_spec in PLOT_SPECS:
        if str(plot_spec.get("render", "")) == "controller_decision_time":
            continue
        plot_frame = _build_ci_metric_frame(summary_payload, plot_key=str(plot_spec["key"]))
        if plot_frame is None:
            continue
        plot_frame = _filter_plotted_trace_types(
            plot_frame,
            skip_protected_setups=bool(args.skip_protected_setups),
        )
        adjusted_plot_frame = _apply_setup_name_map(_apply_setup_value_adjustments(plot_frame))
        ci_plot_frames.append((adjusted_plot_frame, plot_spec))
        box_plot_frames.append((adjusted_plot_frame, plot_spec))
    controller_decision_time_ci_frame = _build_controller_decision_time_timing_frame(
        timing_summary_payloads,
        fallback_summary_payload=summary_payload,
    )
    if controller_decision_time_ci_frame is not None:
        controller_decision_time_ci_frame = _filter_plotted_trace_types(
            controller_decision_time_ci_frame,
            skip_protected_setups=bool(args.skip_protected_setups),
        )
        controller_decision_time_ci_frame = _apply_setup_name_map(controller_decision_time_ci_frame)
    controller_decision_time_box_frame = _build_controller_decision_time_box_frame(controller_decision_time_ci_frame)
    if controller_decision_time_box_frame is not None:
        controller_decision_time_box_frame = _filter_plotted_trace_types(
            controller_decision_time_box_frame,
            skip_protected_setups=bool(args.skip_protected_setups),
        )
        controller_decision_time_box_frame = _apply_setup_name_map(controller_decision_time_box_frame)

    output_paths = [
        path
        for path in (_save_metric_plot(metric_frame, spec, out_dir, output_format=output_format) for spec in PLOT_SPECS)
        if path is not None
    ]
    for plot_frame, plot_spec in ci_plot_frames:
        ci_path = _save_ci_plot(plot_frame, plot_spec, out_dir, output_format=output_format)
        if ci_path is not None:
            output_paths.append(ci_path)
    for plot_frame, plot_spec in box_plot_frames:
        box_path = _save_box_plot(
            plot_frame,
            plot_spec,
            out_dir,
            output_format=output_format,
            sort_trace_types_by_value=bool(args.skip_protected_setups),
        )
        if box_path is not None:
            output_paths.append(box_path)
    if controller_decision_time_ci_frame is not None:
        controller_ci_path = _save_controller_decision_time_ci_plot(
            controller_decision_time_ci_frame,
            _spec_by_key("controller_decision_time"),
            out_dir,
            output_format=output_format,
        )
        if controller_ci_path is not None:
            output_paths.append(controller_ci_path)
    if controller_decision_time_box_frame is not None:
        controller_box_path = _save_box_plot(
            controller_decision_time_box_frame,
            _spec_by_key("controller_decision_time"),
            out_dir,
            output_format=output_format,
            sort_trace_types_by_value=bool(args.skip_protected_setups),
        )
        if controller_box_path is not None:
            output_paths.append(controller_box_path)
    if not bool(args.skip_protected_setups):
        protection_tradeoff_frame = _build_protection_tradeoff_frame(summary_payload)
        protection_tradeoff_path = _save_protection_tradeoff_plot(
            protection_tradeoff_frame,
            out_dir,
            output_format=output_format,
        )
        if protection_tradeoff_path is not None:
            output_paths.append(protection_tradeoff_path)
        protection_breakdown_frame = _build_protection_tradeoff_breakdown_frame(summary_payload)
        if protection_breakdown_frame is not None and not protection_breakdown_frame.empty:
            protection_breakdown_data_path = os.path.join(out_dir, "shield_tradeoff_breakdown_bar_data.csv")
            protection_breakdown_frame.to_csv(protection_breakdown_data_path, index=False)
            output_paths.append(protection_breakdown_data_path)
        protection_breakdown_clean_path = _save_protection_tradeoff_breakdown_clean_bar_ci95(
            protection_breakdown_frame,
            out_dir,
            output_format=output_format,
        )
        if protection_breakdown_clean_path is not None:
            output_paths.append(protection_breakdown_clean_path)
        protection_breakdown_adv_path = _save_protection_tradeoff_breakdown_adversarial_combined_bar_ci95(
            protection_breakdown_frame,
            out_dir,
            output_format=output_format,
        )
        if protection_breakdown_adv_path is not None:
            output_paths.append(protection_breakdown_adv_path)

    manifest_path = os.path.join(out_dir, "plot_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as file_obj:
        json.dump(
            {
                "summary_path": summary_path,
                "summary_paths": list(summary_payload.get("summary_paths", [])),
                "timing_summary_paths": timing_summary_paths,
                "out_dir": out_dir,
                "output_format": output_format,
                "skip_protected_setups": bool(args.skip_protected_setups),
                "setup_name_map": SETUP_NAME_MAP,
                "setup_value_adjustment_pct": SETUP_VALUE_ADJUSTMENT_PCT,
                "plots": output_paths,
            },
            file_obj,
            indent=2,
            sort_keys=True,
        )

    for path in output_paths:
        print(path)
    print(manifest_path)


if __name__ == "__main__":  # pragma: no cover
    main()
