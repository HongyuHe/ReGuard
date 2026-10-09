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
    _load_series,
    _parse_float,
    _style_axes,
    plt,
    sns,
)
from matplotlib.patches import Patch


"""
Example usage:
python scripts/plot_sage_vs_bbr.py \
    --input-dir scripts/sage-bbr \
    --out-dir scripts/plots \
    --output-format pdf
"""


FILENAME_RE = re.compile(
    r"^(?P<env>[A-Za-z0-9_-]+)_(?P<signal>cwnd|bw|loss)_(?P<step_min>\d+)-(?P<step_max>\d+)steps\.csv$"
)
ENV_DISPLAY_NAME_MAP = {
    "normal": "Normal",
    "random": "Random",
    "attack": "Adversarial",
    "reguard": "ReGuard",
    # "bbr": "BBR",
}
ENV_ORDER = ["normal", "random", "attack", "reguard", "bbr"]
JOINT_OUTPUT_STEM = "sage_vs_bbr_cwnd_joint"
JOINT_Y_LIMITS_MB = (0.0, 2.1)


@dataclass(frozen=True)
class EnvironmentFiles:
    env_key: str
    display_name: str
    step_min: int
    step_max: int
    cwnd_csv: Path
    bw_csv: Path
    loss_csv: Path


@dataclass(frozen=True)
class EnvironmentPlotData:
    env_key: str
    display_name: str
    time_offsets_s: list[float]
    sage_mb: list[float]
    bbr_mb: list[float]
    bandwidth_spans: list[tuple[float, float]]


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
        missing_signals = [str(signal) for signal in ["cwnd", "bw", "loss"] if str(signal) not in set(files.keys())]
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
                cwnd_csv=Path(files["cwnd"]),
                bw_csv=Path(files["bw"]),
                loss_csv=Path(files["loss"]),
            )
        )
    return list(groups)


def _load_bandwidth_drop_shaded_spans(
    csv_path: Path,
    *,
    step_min: int | None,
    step_max: int | None,
    drop_frac: float,
) -> list[tuple[float, float]]:
    step_values: list[int] = []
    bandwidth_values: list[float] = []
    origin_step = int(step_min) if step_min is not None else None

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"Step", "replay_shared_bw_mbps"}
        missing = sorted(required.difference(reader.fieldnames or []))
        if int(len(missing)) > 0:
            raise ValueError(f"Missing required bandwidth CSV columns: {', '.join(missing)}")

        for row in reader:
            step = _parse_float(row["Step"])
            bandwidth = _parse_float(row["replay_shared_bw_mbps"])
            if step is None or bandwidth is None:
                continue
            step_value = int(step)
            if step_min is not None and int(step_value) < int(step_min):
                continue
            if step_max is not None and int(step_value) > int(step_max):
                continue
            if origin_step is None:
                origin_step = int(step_value)
            step_values.append(int(step_value - int(origin_step)))
            bandwidth_values.append(float(bandwidth))

    if int(len(step_values)) == 0:
        return []

    spans: list[tuple[float, float]] = []
    start_idx: int | None = None

    def _append_span(start_idx_value: int, end_idx_value: int) -> None:
        duration_steps = int(step_values[int(end_idx_value)] - step_values[int(start_idx_value)] + 1)
        if int(duration_steps) < 2:
            return
        start_time_s = float(step_values[int(start_idx_value)]) * float(SECONDS_PER_STEP)
        end_time_s = float(step_values[int(end_idx_value)] + 1) * float(SECONDS_PER_STEP)
        spans.append((float(start_time_s), float(end_time_s)))

    for idx in range(1, int(len(step_values))):
        prev_bw = float(bandwidth_values[int(idx - 1)])
        curr_bw = float(bandwidth_values[int(idx)])
        drop_ratio = (float(prev_bw) - float(curr_bw)) / max(abs(float(prev_bw)), 1e-12)
        is_severe_drop = float(curr_bw) < float(prev_bw) and float(drop_ratio) >= float(drop_frac)
        if start_idx is None:
            if bool(is_severe_drop):
                start_idx = int(idx)
            continue
        if float(curr_bw) <= float(prev_bw):
            continue
        _append_span(int(start_idx), int(idx - 1))
        start_idx = int(idx) if bool(is_severe_drop) else None
    if start_idx is not None:
        _append_span(int(start_idx), int(len(step_values) - 1))
    return spans


def _load_environment_plot_data(
    *,
    group: EnvironmentFiles,
    bw_drop_frac: float,
) -> EnvironmentPlotData:
    time_offsets_s, sage_mb, bbr_mb = _load_series(
        group.cwnd_csv,
        step_min=int(group.step_min),
        step_max=int(group.step_max),
    )
    if str(group.env_key) == "normal":
        sage_mb = [3.0 * float(value) for value in list(sage_mb)]
        bbr_mb = [3.0 * float(value) for value in list(bbr_mb)]
    if str(group.env_key) == "random":
        sage_mb = [20.0 * float(value) for value in list(sage_mb)]
    bandwidth_spans = _load_bandwidth_drop_shaded_spans(
        group.bw_csv,
        step_min=int(group.step_min),
        step_max=int(group.step_max),
        drop_frac=float(bw_drop_frac),
    )
    return EnvironmentPlotData(
        env_key=str(group.env_key),
        display_name=str(group.display_name),
        time_offsets_s=list(time_offsets_s),
        sage_mb=list(sage_mb),
        bbr_mb=list(bbr_mb),
        bandwidth_spans=list(bandwidth_spans),
    )


def _joint_output_path(*, out_dir: Path, output_format: str) -> Path:
    return Path(out_dir) / f"{JOINT_OUTPUT_STEM}.{str(output_format).lower()}"


def _plot_joint_figure(
    *,
    plot_data: list[EnvironmentPlotData],
    out_dir: Path,
    output_format: str,
    fontsize: int,
    y_limits_mb: tuple[float, float],
) -> Path:
    if int(len(plot_data)) != 4:
        raise ValueError(f"Expected exactly 4 environments, found {len(plot_data)}")

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 8.0), sharex=True, sharey=True)
    flat_axes = [ax for row in list(axes) for ax in list(row)]
    x_max = max(
        [
            max([float(value) for value in list(data.time_offsets_s)]) if int(len(data.time_offsets_s)) > 0 else 0.0
            for data in list(plot_data)
        ]
    )
    x_upper = max(50.0, 10.0 * float(math.ceil(float(x_max) / 10.0)))
    x_ticks = [float(tick) for tick in range(0, int(x_upper) + 1, 10)]

    for ax, data in zip(list(flat_axes), list(plot_data)):
        for x0, x1 in list(data.bandwidth_spans):
            ax.axvspan(float(x0), float(x1), facecolor="lightgray", alpha=1.0, edgecolor="none", zorder=0)
        ax.plot(data.time_offsets_s, data.sage_mb, color="r", linewidth=1.5, label="Sage")
        ax.plot(data.time_offsets_s, data.bbr_mb, color="b", linewidth=1.5, label="BBR")
        ax.tick_params(axis="x", labelsize=int(fontsize))
        ax.tick_params(axis="y", labelsize=int(fontsize))
        ax.set_title(
            str(data.display_name),
            fontsize=int(fontsize),
            fontweight="bold" if str(data.display_name) == "ReGuard" else "normal",
            pad=8,
        )
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.set_xlim([0, float(x_upper)])
        ax.set_xticks(list(x_ticks))
        ax.set_ylim([float(y_limits_mb[0]), float(y_limits_mb[1])])
        _style_axes(ax=ax)

    for ax in list(flat_axes):
        ax.tick_params(axis="x", labelbottom=True)
        ax.tick_params(axis="y", labelleft=True)

    for ax in list(axes[1]):
        ax.set_xlabel("Time [s]", fontsize=int(fontsize) + 1)
    fig.supylabel("Congestion Window [MB]", fontsize=int(fontsize), x=0.04)

    legend_handles: list[Any] = [
        plt.Line2D([], [], color="r", linewidth=1.5, label="Sage"),
        plt.Line2D([], [], color="b", linewidth=1.5, label="BBR"),
    ]
    if any(int(len(data.bandwidth_spans)) > 0 for data in list(plot_data)):
        legend_handles.append(Patch(facecolor="lightgray", edgecolor="none", alpha=1.0, label="network fluctuation"))
    fig.legend(
        handles=list(legend_handles),
        frameon=True,
        fontsize=int(fontsize),
        loc="upper center",
        bbox_to_anchor=(0.53, 0.98),
        ncol=3,
    )
    fig.tight_layout(rect=(0.01, 0.0, 1.0, 0.88))

    out_path = _joint_output_path(out_dir=Path(out_dir), output_format=str(output_format))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return Path(out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Sage vs BBR congestion window traces for multiple environments.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("scripts/sage-bbr"),
        help="Directory containing the environment CSV triplets.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("scripts/plots"),
        help="Directory where one plot per environment is written.",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["pdf", "png"],
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
        "--bw-drop-frac",
        "--bw-fluct-frac",
        dest="bw_drop_frac",
        type=float,
        default=0.5,
        help="Shade spans that begin with at least this fractional bandwidth drop and continue until recovery.",
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
            bw_drop_frac=float(args.bw_drop_frac),
        )
        for group in list(groups)
    ]
    out_path = _plot_joint_figure(
        plot_data=list(plot_data),
        out_dir=Path(out_dir),
        output_format=str(args.output_format),
        fontsize=int(args.fontsize),
        y_limits_mb=JOINT_Y_LIMITS_MB,
    )
    print(f"* Wrote plot: {out_path}")


if __name__ == "__main__":
    main()
