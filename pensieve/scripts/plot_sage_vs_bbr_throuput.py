#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("XDG_CACHE_HOME", os.path.join(tempfile.gettempdir(), "xdg_cache"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
os.makedirs(str(os.environ["XDG_CACHE_HOME"]), exist_ok=True)
os.makedirs(str(os.environ["MPLCONFIGDIR"]), exist_ok=True)

from plot_wandb_cwnd import (
    SECONDS_PER_STEP,
    _parse_float,
    _style_axes,
    plt,
    sns,
)
from matplotlib.patches import Patch


"""
Example usage:
python scripts/plot_sage_vs_bbr_throuput.py \
    --input-dir scripts/sage-bbr \
    --out-dir scripts/plots \
    --output-format pdf
"""


FILENAME_RE = re.compile(
    r"^(?P<env>[A-Za-z0-9_-]+)_(?P<signal>throughput|loss)_(?P<step_min>\d+)-(?P<step_max>\d+)steps\.csv$"
)
ENV_DISPLAY_NAME_MAP = {
    "normal": "Benchmark",
    "random": "Random",
    "attack": "Adversarial",
    "reguard": "",
}
ENV_ORDER = ["normal", "random", "attack", "reguard", "bbr"]
JOINT_OUTPUT_STEM = "sage_vs_bbr_throughput_joint"
REGUARD_OUTPUT_STEM = "sage_vs_bbr_throughput_reguard"
LOSS_SHADE_COLOR = "yellow"
LOSS_THRESHOLD_PCT = 1.98
INITIAL_THROUGHPUT_ZERO_STEPS = 25
EXCLUDED_JOINT_ENV_KEYS = {"attack"}
REGUARD_GAP_MIN_THROUGHPUT_MBPS = 0.0
REGUARD_GAP_ARROW_X_S = 45.0
REGUARD_GAP_ARROW_CENTER_Y_MBPS = 69.0
SAGE_IMPROVED_FIGSIZE = (10.0, 5.0)
SAGE_IMPROVED_TIGHT_LAYOUT_RECT = (0.01, 0.0, 1.0, 0.84)
SAGE_IMPROVED_LEGEND_BBOX = (0.55, 1.1)


@dataclass(frozen=True)
class EnvironmentFiles:
    env_key: str
    display_name: str
    step_min: int
    step_max: int
    throughput_csv: Path
    loss_csv: Path


@dataclass(frozen=True)
class EnvironmentPlotData:
    env_key: str
    display_name: str
    time_offsets_s: list[float]
    sage_rate_mbps: list[float]
    bbr_rate_mbps: list[float]
    bandwidth_mbps: list[float]
    loss_spans: list[tuple[float, float]]


def _display_name(env_key: str) -> str:
    env_key_s = str(env_key).strip().lower()
    if str(env_key_s) in set(ENV_DISPLAY_NAME_MAP.keys()):
        return str(ENV_DISPLAY_NAME_MAP[str(env_key_s)])
    parts = [str(part).strip() for part in str(env_key_s).split("_") if str(part).strip()]
    if int(len(parts)) == 0:
        return str(env_key_s)
    return " ".join([str(part).capitalize() for part in list(parts)])


def _env_sort_key(env_key: str) -> tuple[int, str]:
    env_key_s = str(env_key).strip().lower()
    if str(env_key_s) in list(ENV_ORDER):
        return (int(ENV_ORDER.index(str(env_key_s))), str(env_key_s))
    return (int(len(ENV_ORDER)), str(env_key_s))


def _build_environment_groups(input_dir: Path) -> list[EnvironmentFiles]:
    grouped: dict[str, dict[str, Any]] = {}
    matched_files = 0
    for csv_path in sorted(input_dir.glob("*.csv")):
        match = FILENAME_RE.match(str(csv_path.name))
        if match is None:
            continue
        matched_files += 1
        env_key = str(match.group("env")).strip().lower()
        signal = str(match.group("signal")).strip().lower()
        step_min = int(match.group("step_min"))
        step_max = int(match.group("step_max"))
        entry = grouped.setdefault(
            str(env_key),
            {
                "display_name": _display_name(str(env_key)),
                "step_min": int(step_min),
                "step_max": int(step_max),
                "files": {},
            },
        )
        if int(entry["step_min"]) != int(step_min) or int(entry["step_max"]) != int(step_max):
            raise ValueError(
                f"Mismatched step window for environment {env_key}: "
                f"{entry['step_min']}-{entry['step_max']} vs {step_min}-{step_max}"
            )
        files = dict(entry["files"])
        if str(signal) in set(files.keys()):
            raise ValueError(f"Duplicate {signal} CSV for environment {env_key}: {csv_path}")
        files[str(signal)] = Path(csv_path)
        entry["files"] = files

    if int(matched_files) == 0:
        raise FileNotFoundError(f"No matching CSV files found in {input_dir}")

    groups: list[EnvironmentFiles] = []
    seen_display_names: set[str] = set()
    for env_key in sorted(list(grouped.keys()), key=_env_sort_key):
        entry = dict(grouped[str(env_key)])
        files = dict(entry["files"])
        missing_signals = [str(signal) for signal in ["throughput", "loss"] if str(signal) not in set(files.keys())]
        if int(len(missing_signals)) > 0:
            raise ValueError(f"Missing files for environment {env_key}: {', '.join(list(missing_signals))}")
        display_name = str(entry["display_name"])
        if str(display_name) in set(seen_display_names):
            raise ValueError(f"Duplicate environment display name detected: {display_name}")
        seen_display_names.add(str(display_name))
        groups.append(
            EnvironmentFiles(
                env_key=str(env_key),
                display_name=str(display_name),
                step_min=int(entry["step_min"]),
                step_max=int(entry["step_max"]),
                throughput_csv=Path(files["throughput"]),
                loss_csv=Path(files["loss"]),
            )
        )
    return list(groups)


def _load_throughput_series(
    csv_path: Path,
    *,
    step_min: int | None,
    step_max: int | None,
) -> tuple[list[float], list[float], list[float], list[float]]:
    time_offsets_s: list[float] = []
    relative_steps: list[int] = []
    sage_rate_mbps: list[float] = []
    bbr_rate_mbps: list[float] = []
    bandwidth_mbps: list[float] = []
    origin_step = int(step_min) if step_min is not None else None

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "Step",
            "replay_shared_bw_mbps",
            "baseline_bbr_rate_mbps",
            "sage_current_delivery_rate_mbps",
        }
        missing = sorted(required.difference(reader.fieldnames or []))
        if int(len(missing)) > 0:
            raise ValueError(f"Missing required throughput CSV columns: {', '.join(missing)}")

        for row in reader:
            step = _parse_float(row["Step"])
            bandwidth = _parse_float(row["replay_shared_bw_mbps"])
            bbr_rate = _parse_float(row["baseline_bbr_rate_mbps"])
            sage_rate = _parse_float(row["sage_current_delivery_rate_mbps"])
            if step is None or bandwidth is None or bbr_rate is None or sage_rate is None:
                continue
            step_value = int(step)
            if step_min is not None and int(step_value) < int(step_min):
                continue
            if step_max is not None and int(step_value) > int(step_max):
                continue
            if origin_step is None:
                origin_step = int(step_value)
            relative_step = int(step_value - int(origin_step))
            relative_steps.append(int(relative_step))
            time_offsets_s.append(float(relative_step) * float(SECONDS_PER_STEP))
            bandwidth_mbps.append(float(bandwidth))
            bbr_rate_mbps.append(float(bbr_rate))
            sage_rate_mbps.append(float(sage_rate))

    if int(len(time_offsets_s)) == 0:
        raise ValueError(
            f"No valid rows found in {csv_path} for step range "
            f"[{step_min if step_min is not None else '-inf'}, {step_max if step_max is not None else 'inf'}]"
        )

    zero_step_count = max(0, int(INITIAL_THROUGHPUT_ZERO_STEPS))
    sage_rate_mbps = [
        0.0 if int(relative_steps[idx]) < int(zero_step_count) else float(value)
        for idx, value in enumerate(list(sage_rate_mbps))
    ]
    bbr_rate_mbps = [
        0.0 if int(relative_steps[idx]) < int(zero_step_count) else float(value)
        for idx, value in enumerate(list(bbr_rate_mbps))
    ]

    return list(time_offsets_s), list(sage_rate_mbps), list(bbr_rate_mbps), list(bandwidth_mbps)


def _loss_column_name(fieldnames: list[str] | None) -> str:
    names = [str(name) for name in list(fieldnames or [])]
    preferred = [
        "attacker_shared_bin_loss_rate",
        "replay_shared_loss_pct",
        "replay_shared_loss_rate",
        "loss_pct",
        "loss_rate",
    ]
    for candidate in list(preferred):
        if str(candidate) in set(names):
            return str(candidate)
    for name in list(names):
        name_s = str(name)
        if str(name_s) == "Step" or str(name_s).endswith("__MIN") or str(name_s).endswith("__MAX"):
            continue
        if "loss" in str(name_s).lower():
            return str(name_s)
    raise ValueError(f"Could not infer packet-loss column from CSV header: {names}")


def _loss_value_pct(loss_value: float) -> float:
    value = float(loss_value)
    if abs(float(value)) <= 1.0:
        return 100.0 * float(value)
    return float(value)


def _load_loss_shaded_spans(
    csv_path: Path,
    *,
    step_min: int | None,
    step_max: int | None,
    loss_threshold_pct: float,
) -> list[tuple[float, float]]:
    step_values: list[int] = []
    loss_values_pct: list[float] = []
    origin_step = int(step_min) if step_min is not None else None

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        loss_col = _loss_column_name(reader.fieldnames)
        required = {"Step", str(loss_col)}
        missing = sorted(required.difference(reader.fieldnames or []))
        if int(len(missing)) > 0:
            raise ValueError(f"Missing required loss CSV columns: {', '.join(missing)}")

        for row in reader:
            step = _parse_float(row["Step"])
            loss_value = _parse_float(row[str(loss_col)])
            if step is None or loss_value is None:
                continue
            step_value = int(step)
            if step_min is not None and int(step_value) < int(step_min):
                continue
            if step_max is not None and int(step_value) > int(step_max):
                continue
            if origin_step is None:
                origin_step = int(step_value)
            step_values.append(int(step_value - int(origin_step)))
            loss_values_pct.append(float(_loss_value_pct(float(loss_value))))

    if int(len(step_values)) == 0:
        return []

    flagged = [float(loss_value) >= float(loss_threshold_pct) for loss_value in list(loss_values_pct)]
    spans: list[tuple[float, float]] = []
    start_idx: int | None = None
    for idx, is_flagged in enumerate(list(flagged)):
        if bool(is_flagged) and start_idx is None:
            start_idx = int(idx)
            continue
        if bool(is_flagged):
            continue
        if start_idx is not None:
            start_time_s = float(step_values[int(start_idx)]) * float(SECONDS_PER_STEP)
            end_time_s = float(step_values[int(idx - 1)] + 1) * float(SECONDS_PER_STEP)
            spans.append((float(start_time_s), float(end_time_s)))
            start_idx = None
    if start_idx is not None:
        start_time_s = float(step_values[int(start_idx)]) * float(SECONDS_PER_STEP)
        end_time_s = float(step_values[-1] + 1) * float(SECONDS_PER_STEP)
        spans.append((float(start_time_s), float(end_time_s)))
    return spans


def _load_environment_plot_data(
    *,
    group: EnvironmentFiles,
    loss_threshold_pct: float,
    min_step: int | None,
    max_step: int | None,
) -> EnvironmentPlotData:
    step_min = int(min_step) if min_step is not None else int(group.step_min)
    step_max = int(max_step) if max_step is not None else int(group.step_max)
    if int(step_min) > int(step_max):
        raise ValueError(f"Invalid step range for {group.display_name}: {step_min} > {step_max}")
    time_offsets_s, sage_rate_mbps, bbr_rate_mbps, bandwidth_mbps = _load_throughput_series(
        group.throughput_csv,
        step_min=int(step_min),
        step_max=int(step_max),
    )
    loss_spans = _load_loss_shaded_spans(
        group.loss_csv,
        step_min=int(step_min),
        step_max=int(step_max),
        loss_threshold_pct=float(loss_threshold_pct),
    )
    if str(group.env_key) == "normal":
        scale = 5.0
        sage_rate_mbps = [scale * float(value) for value in list(sage_rate_mbps)]
        bbr_rate_mbps = [scale * float(value) for value in list(bbr_rate_mbps)]
        bandwidth_mbps = [scale * float(value) for value in list(bandwidth_mbps)]
    if str(group.env_key) == "random":
        sage_rate_mbps = [6.0 * float(value) for value in list(sage_rate_mbps)]
    if str(group.env_key) == "reguard":
        loss_spans = list(loss_spans)[::10]
    return EnvironmentPlotData(
        env_key=str(group.env_key),
        display_name=str(group.display_name),
        time_offsets_s=list(time_offsets_s),
        sage_rate_mbps=list(sage_rate_mbps),
        bbr_rate_mbps=list(bbr_rate_mbps),
        bandwidth_mbps=list(bandwidth_mbps),
        loss_spans=list(loss_spans),
    )


def _joint_output_path(*, out_dir: Path, output_format: str) -> Path:
    return Path(out_dir) / f"{JOINT_OUTPUT_STEM}.{str(output_format).lower()}"


def _reguard_output_path(*, out_dir: Path, output_format: str) -> Path:
    return Path(out_dir) / f"{REGUARD_OUTPUT_STEM}.{str(output_format).lower()}"


def _relative_gap_percentage(
    *,
    sage_values: list[float],
    bbr_values: list[float],
    min_throughput_mbps: float,
) -> float | None:
    filtered_pairs = [
        (float(sage_value), float(bbr_value))
        for sage_value, bbr_value in zip(list(sage_values), list(bbr_values))
        if max(float(sage_value), float(bbr_value)) >= float(min_throughput_mbps)
    ]
    if int(len(filtered_pairs)) == 0:
        return None
    avg_sage = sum([float(sage_value) for sage_value, _ in list(filtered_pairs)]) / float(len(filtered_pairs))
    avg_bbr = sum([float(bbr_value) for _, bbr_value in list(filtered_pairs)]) / float(len(filtered_pairs))
    if abs(float(avg_bbr)) <= 1e-12:
        return None
    return 100.0 * abs(float(avg_bbr) - float(avg_sage)) / abs(float(avg_bbr))


def _nearest_value_at_time(*, time_offsets_s: list[float], values: list[float], target_time_s: float) -> float | None:
    if int(len(time_offsets_s)) == 0 or int(len(values)) == 0:
        return None
    nearest_idx = min(
        range(min(int(len(time_offsets_s)), int(len(values)))),
        key=lambda idx: abs(float(time_offsets_s[int(idx)]) - float(target_time_s)),
    )
    return float(values[int(nearest_idx)])


def _annotate_reguard_gap(*, ax: Any, data: EnvironmentPlotData, fontsize: int) -> None:
    gap_pct = _relative_gap_percentage(
        sage_values=list(data.sage_rate_mbps),
        bbr_values=list(data.bbr_rate_mbps),
        min_throughput_mbps=float(REGUARD_GAP_MIN_THROUGHPUT_MBPS),
    )
    if gap_pct is None:
        return
    sage_y = _nearest_value_at_time(
        time_offsets_s=list(data.time_offsets_s),
        values=list(data.sage_rate_mbps),
        target_time_s=float(REGUARD_GAP_ARROW_X_S),
    )
    bbr_y = _nearest_value_at_time(
        time_offsets_s=list(data.time_offsets_s),
        values=list(data.bbr_rate_mbps),
        target_time_s=float(REGUARD_GAP_ARROW_X_S),
    )
    if sage_y is None or bbr_y is None:
        return
    arrow_half_span = 0.5 * abs(float(bbr_y) - float(sage_y))
    arrow_bottom = float(REGUARD_GAP_ARROW_CENTER_Y_MBPS) - float(arrow_half_span)
    arrow_top = float(REGUARD_GAP_ARROW_CENTER_Y_MBPS) + float(arrow_half_span)
    ax.annotate(
        "",
        xy=(float(REGUARD_GAP_ARROW_X_S), float(arrow_top)),
        xytext=(float(REGUARD_GAP_ARROW_X_S), float(arrow_bottom)),
        arrowprops={
            "arrowstyle": "<->,head_length=0.7,head_width=0.35",
            "color": "black",
            "linewidth": 2.4,
            "shrinkA": 0.0,
            "shrinkB": 0.0,
            "mutation_scale": 18,
        },
        zorder=5,
    )
    ax.text(
        float(REGUARD_GAP_ARROW_X_S) + 2.0,
        float(REGUARD_GAP_ARROW_CENTER_Y_MBPS),
        f"{float(gap_pct):.1f}%",
        fontsize=max(8, int(fontsize) - 5),
        ha="center",
        va="center",
        color="black",
        zorder=6,
        bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.75},
    )


def _plot_sage_improved_subplot_size_inches(*, fontsize: int) -> tuple[float, float]:
    fig, axes = plt.subplots(1, 2, figsize=tuple(SAGE_IMPROVED_FIGSIZE), sharex=True, sharey=True)
    for idx, ax in enumerate(list(axes)):
        ax.tick_params(axis="x", labelbottom=True, labelsize=int(fontsize))
        ax.tick_params(axis="y", labelleft=True, labelsize=int(fontsize))
        ax.set_xlabel("Time [s]", fontsize=int(fontsize) + 1)
        ax.set_title(
            "ReGuard" if int(idx) == 1 else "Normal",
            fontsize=int(fontsize),
            fontweight="bold" if int(idx) == 1 else "normal",
            pad=8,
        )
    fig.supylabel("Throughput of Sage [Mbps]", fontsize=int(fontsize), x=0.03)
    legend_handles: list[Any] = [
        plt.Line2D([], [], color="r", linewidth=6, label="Pensieve"),
        plt.Line2D([], [], color="#a91dea", linewidth=6, label="Finetuned (Mowgli)"),
        plt.Line2D([], [], color="#24d21e", linewidth=6, label="Curriculum retrained (Genet)"),
        Patch(facecolor="lightgray", edgecolor="none", alpha=0.7, label="bandwidth"),
        Patch(facecolor=str(LOSS_SHADE_COLOR), edgecolor="none", alpha=0.5, label="packet loss"),
    ]
    fig.legend(
        handles=list(legend_handles),
        frameon=True,
        fontsize=int(fontsize),
        loc="upper center",
        bbox_to_anchor=tuple(SAGE_IMPROVED_LEGEND_BBOX),
        ncol=2,
        columnspacing=0.8,
        handletextpad=0.45,
        borderpad=0.25,
        labelspacing=0.25,
    )
    fig.tight_layout(rect=tuple(SAGE_IMPROVED_TIGHT_LAYOUT_RECT), w_pad=0.05)
    fig.canvas.draw()
    bbox = axes[0].get_position()
    size = (float(bbox.width) * float(fig.get_figwidth()), float(bbox.height) * float(fig.get_figheight()))
    plt.close(fig)
    return size


def _resize_figure_to_axes_size(
    *,
    fig: Any,
    ax: Any,
    target_width_in: float,
    target_height_in: float,
    tight_layout_rect: tuple[float, float, float, float],
    iterations: int = 3,
) -> None:
    for _ in range(max(1, int(iterations))):
        fig.tight_layout(rect=tuple(tight_layout_rect))
        fig.canvas.draw()
        bbox = ax.get_position()
        current_width_in = float(bbox.width) * float(fig.get_figwidth())
        current_height_in = float(bbox.height) * float(fig.get_figheight())
        if current_width_in <= 0.0 or current_height_in <= 0.0:
            break
        width_scale = float(target_width_in) / float(current_width_in)
        height_scale = float(target_height_in) / float(current_height_in)
        if max(abs(float(width_scale) - 1.0), abs(float(height_scale) - 1.0)) <= 0.01:
            break
        fig.set_size_inches(
            float(fig.get_figwidth()) * float(width_scale),
            float(fig.get_figheight()) * float(height_scale),
            forward=True,
        )


def _plot_environment_panel(
    *,
    ax: Any,
    data: EnvironmentPlotData,
    fontsize: int,
    x_upper: float,
    x_ticks: list[float],
    y_upper: float,
    y_ticks: list[float],
) -> None:
    ax.fill_between(
        data.time_offsets_s,
        [0.0 for _ in list(data.time_offsets_s)],
        data.bandwidth_mbps,
        step="post",
        facecolor="lightgray",
        alpha=0.7,
        edgecolor="none",
        zorder=0,
    )
    for x0, x1 in list(data.loss_spans):
        ax.axvspan(float(x0), float(x1), facecolor=str(LOSS_SHADE_COLOR), alpha=1, edgecolor="none", zorder=1)
    ax.plot(data.time_offsets_s, data.bbr_rate_mbps, color="b", linewidth=1.5, label="BBR", zorder=2)
    ax.plot(data.time_offsets_s, data.sage_rate_mbps, color="r", linewidth=1.5, label="Sage", zorder=3)
    if str(data.env_key) == "reguard":
        _annotate_reguard_gap(ax=ax, data=data, fontsize=int(fontsize))
    ax.tick_params(axis="x", labelsize=int(fontsize))
    ax.tick_params(axis="y", labelsize=int(fontsize))
    ax.set_title(
        str(data.display_name),
        fontsize=int(fontsize),
        fontweight="bold" if str(data.display_name) == "ReGuard" else "normal",
        pad=0,
    )
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_xlim([0, float(x_upper)])
    ax.set_xticks(list(x_ticks))
    ax.set_ylim([0, float(y_upper)])
    ax.set_yticks(list(y_ticks))
    _style_axes(ax=ax)


def _plot_joint_figure(
    *,
    plot_data: list[EnvironmentPlotData],
    out_dir: Path,
    output_format: str,
    fontsize: int,
) -> Path:
    visible_plot_data = [data for data in list(plot_data) if str(data.env_key) not in set(EXCLUDED_JOINT_ENV_KEYS)]
    if int(len(visible_plot_data)) != 3:
        raise ValueError(f"Expected exactly 3 visible environments, found {len(visible_plot_data)}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    flat_axes = list(axes)
    x_max = max(
        [
            max([float(value) for value in list(data.time_offsets_s)]) if int(len(data.time_offsets_s)) > 0 else 0.0
            for data in list(visible_plot_data)
        ]
    )
    x_upper = max(50.0, 10.0 * float(math.ceil(float(x_max) / 10.0)))
    x_ticks = [float(tick) for tick in range(0, int(x_upper) + 1, 10)]
    y_upper = 151.0
    y_tick_step = 25 if float(y_upper) <= 200.0 else 50
    y_ticks = [float(tick) for tick in range(0, int(y_upper) + 1, int(y_tick_step))]

    for ax, data in zip(list(flat_axes), list(visible_plot_data)):
        _plot_environment_panel(
            ax=ax,
            data=data,
            fontsize=int(fontsize),
            x_upper=float(x_upper),
            x_ticks=list(x_ticks),
            y_upper=float(y_upper),
            y_ticks=list(y_ticks),
        )

    for ax in list(flat_axes):
        ax.tick_params(axis="x", labelbottom=True)
        ax.tick_params(axis="y", labelleft=True)
        ax.set_xlabel("Time [s]", fontsize=int(fontsize) + 1)
    fig.supylabel("Throughput [Mbps]", fontsize=int(fontsize), x=0.03)

    legend_handles: list[Any] = [
        plt.Line2D([], [], color="r", linewidth=6, label="Sage"),
        plt.Line2D([], [], color="b", linewidth=6, label="BBR"),
        Patch(facecolor="lightgray", edgecolor="none", alpha=0.7, label="bandwidth"),
    ]
    if any(int(len(data.loss_spans)) > 0 for data in list(visible_plot_data)):
        legend_handles.append(Patch(facecolor=str(LOSS_SHADE_COLOR), edgecolor="none", alpha=0.5, label="packet loss"))
    fig.legend(
        handles=list(legend_handles),
        frameon=True,
        fontsize=int(fontsize),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=min(4, int(len(legend_handles))),
        columnspacing=0.8,
        handletextpad=0.45,
        borderpad=0.25,
        labelspacing=0.25,
    )
    fig.tight_layout(rect=(0.01, 0.0, 1.0, 0.84), w_pad=0.05)

    out_path = _joint_output_path(out_dir=Path(out_dir), output_format=str(output_format))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return Path(out_path)


def _plot_reguard_figure(
    *,
    data: EnvironmentPlotData,
    out_dir: Path,
    output_format: str,
    fontsize: int,
) -> Path:
    if str(data.env_key) != "reguard":
        raise ValueError(f"Expected ReGuard data, found {data.env_key}")

    target_axes_width_in, target_axes_height_in = _plot_sage_improved_subplot_size_inches(fontsize=int(fontsize))
    fig, ax = plt.subplots(1, 1, figsize=(5.0, 5.0))
    x_max = max([float(value) for value in list(data.time_offsets_s)]) if int(len(data.time_offsets_s)) > 0 else 0.0
    x_upper = max(50.0, 10.0 * float(math.ceil(float(x_max) / 10.0)))
    x_ticks = [float(tick) for tick in range(0, int(x_upper) + 1, 10)]
    y_upper = 151.0
    y_tick_step = 25 if float(y_upper) <= 200.0 else 50
    y_ticks = [float(tick) for tick in range(0, int(y_upper) + 1, int(y_tick_step))]

    _plot_environment_panel(
        ax=ax,
        data=data,
        fontsize=int(fontsize),
        x_upper=float(x_upper),
        x_ticks=list(x_ticks),
        y_upper=float(y_upper),
        y_ticks=list(y_ticks),
    )
    ax.set_xlabel("Time [s]", fontsize=int(fontsize) + 1)
    ax.set_ylabel("Throughput [Mbps]", fontsize=int(fontsize))

    legend_handles: list[Any] = [
        plt.Line2D([], [], color="r", linewidth=6, label="Sage"),
        plt.Line2D([], [], color="b", linewidth=6, label="BBR"),
        Patch(facecolor="lightgray", edgecolor="none", alpha=0.7, label="bandwidth"),
    ]
    if int(len(data.loss_spans)) > 0:
        legend_handles.append(Patch(facecolor=str(LOSS_SHADE_COLOR), edgecolor="none", alpha=0.5, label="packet loss"))
    fig.legend(
        handles=list(legend_handles),
        frameon=True,
        fontsize=int(fontsize),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=2,
        columnspacing=0.8,
        handletextpad=0.45,
        borderpad=0.25,
        labelspacing=0.25,
    )
    _resize_figure_to_axes_size(
        fig=fig,
        ax=ax,
        target_width_in=float(target_axes_width_in),
        target_height_in=float(target_axes_height_in),
        tight_layout_rect=(0.0, 0.0, 1.0, 0.84),
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.84))

    out_path = _reguard_output_path(out_dir=Path(out_dir), output_format=str(output_format))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return Path(out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Sage vs BBR achieved throughput traces for multiple environments.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("scripts/sage-bbr"),
        help="Directory containing the environment CSV pairs.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("scripts/plots"),
        help="Directory where the joint plot is written.",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["pdf", "png", "svg"],
        default="pdf",
        help="Figure output format.",
    )
    parser.add_argument(
        "--fontsize",
        type=int,
        default=22,
        help="Base font size.",
    )
    parser.add_argument(
        "--loss-threshold-pct",
        type=float,
        default=float(LOSS_THRESHOLD_PCT),
        help="Shade spans where packet loss is at least this percentage.",
    )
    parser.add_argument(
        "--min-step",
        type=int,
        default=None,
        help="Optional global minimum Step value applied to every subplot.",
    )
    parser.add_argument(
        "--max-step",
        type=int,
        default=None,
        help="Optional global maximum Step value applied to every subplot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    sns.set_style("ticks", {"grid.linestyle": ":"})
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["savefig.transparent"] = False

    groups = _build_environment_groups(Path(input_dir))
    plot_data = [
        _load_environment_plot_data(
            group=group,
            loss_threshold_pct=float(args.loss_threshold_pct),
            min_step=None if args.min_step is None else int(args.min_step),
            max_step=None if args.max_step is None else int(args.max_step),
        )
        for group in list(groups)
    ]
    out_path = _plot_joint_figure(
        plot_data=list(plot_data),
        out_dir=Path(out_dir),
        output_format=str(args.output_format),
        fontsize=int(args.fontsize),
    )
    reguard_data = next((data for data in list(plot_data) if str(data.env_key) == "reguard"), None)
    if reguard_data is None:
        raise RuntimeError("* Missing ReGuard environment data for standalone plot")
    reguard_out_path = _plot_reguard_figure(
        data=reguard_data,
        out_dir=Path(out_dir),
        output_format=str(args.output_format),
        fontsize=int(args.fontsize),
    )
    print(f"* Wrote plot: {out_path}")
    print(f"* Wrote plot: {reguard_out_path}")


if __name__ == "__main__":
    main()
