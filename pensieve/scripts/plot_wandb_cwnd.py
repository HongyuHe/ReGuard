from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("XDG_CACHE_HOME", os.path.join(tempfile.gettempdir(), "xdg_cache"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
os.makedirs(str(os.environ["XDG_CACHE_HOME"]), exist_ok=True)
os.makedirs(str(os.environ["MPLCONFIGDIR"]), exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns


"""
Example usage:
python scripts/plot_wandb_cwnd.py   \
    --csv scripts/data/sage_bbr_trace.csv   \
    --bw-csv scripts/data/sage_bbr_bw.csv   \
    --step-min 4000 --step-max 5000   \
    --out /Users/hongyu/Projects/Formal/sdr/sage_bbr_trace_cwnd.pdf
"""

IPV4_TCP_MSS_BYTES = 1460.0
BYTES_PER_MB = 1_000_000.0
SECONDS_PER_STEP = 0.05


def _parse_float(value: str) -> float | None:
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _packets_to_mb(value: float) -> float:
    return float(value) * float(IPV4_TCP_MSS_BYTES) / float(BYTES_PER_MB)


def _default_output_path(csv_path: Path) -> Path:
    return csv_path.with_name(f"{csv_path.stem}_cwnd_plot.pdf")


def _default_bw_csv_path(csv_path: Path) -> Path | None:
    stem = str(csv_path.stem)
    candidates = []
    if "_trace" in stem:
        candidates.append(csv_path.with_name(f"{stem.replace('_trace', '_bw')}{csv_path.suffix}"))
    candidates.append(csv_path.with_name(f"{stem}_bw{csv_path.suffix}"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _style_axes(*, ax: Any, spine_linewidth: float = 2.0) -> None:
    sns.despine(ax=ax, top=True, right=True)
    ax.spines["bottom"].set_linewidth(float(spine_linewidth))
    ax.spines["left"].set_linewidth(float(spine_linewidth))


def _load_series(
    csv_path: Path,
    *,
    step_min: int | None,
    step_max: int | None,
) -> tuple[list[float], list[float], list[float]]:
    time_offsets_s: list[float] = []
    sage_mb: list[float] = []
    bbr_mb: list[float] = []
    origin_step = int(step_min) if step_min is not None else None

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"Step", "decisions/sage", "decisions/bbr"}
        missing = sorted(required.difference(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Missing required CSV columns: {', '.join(missing)}")

        for row in reader:
            step = _parse_float(row["Step"])
            sage = _parse_float(row["decisions/sage"])
            bbr = _parse_float(row["decisions/bbr"])
            if step is None or sage is None or bbr is None:
                continue
            step_value = int(step)
            if step_min is not None and int(step_value) < int(step_min):
                continue
            if step_max is not None and int(step_value) > int(step_max):
                continue
            if origin_step is None:
                origin_step = int(step_value)
            time_offsets_s.append(float(step_value - int(origin_step)) * float(SECONDS_PER_STEP))
            sage_mb.append(_packets_to_mb(float(sage)))
            bbr_mb.append(_packets_to_mb(float(bbr)))

    if not time_offsets_s:
        raise ValueError(
            f"No valid rows found in {csv_path} for step range "
            f"[{step_min if step_min is not None else '-inf'}, {step_max if step_max is not None else 'inf'}]"
        )

    return time_offsets_s, sage_mb, bbr_mb


def _load_bandwidth_shaded_spans(
    csv_path: Path,
    *,
    step_min: int | None,
    step_max: int | None,
    low_bw_frac: float,
    fluct_bw_frac: float,
) -> list[tuple[float, float]]:
    step_values: list[int] = []
    bandwidth_values: list[float] = []
    origin_step = int(step_min) if step_min is not None else None

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"Step", "replay_shared_bw_mbps"}
        missing = sorted(required.difference(reader.fieldnames or []))
        if missing:
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

    if len(step_values) == 0:
        return []

    spans: list[tuple[float, float]] = []
    start_idx: int | None = None

    def _append_span(start_idx_value: int, end_idx_value: int) -> None:
        start_step = float(step_values[int(start_idx_value)]) * float(SECONDS_PER_STEP)
        end_step = float(step_values[int(end_idx_value)] + 1) * float(SECONDS_PER_STEP)
        spans.append((float(start_step), float(end_step)))

    for idx in range(1, int(len(step_values))):
        prev_bw = float(bandwidth_values[int(idx - 1)])
        curr_bw = float(bandwidth_values[int(idx)])
        drop_ratio = (float(prev_bw) - float(curr_bw)) / max(abs(float(prev_bw)), 1e-12)
        is_sudden_drop = float(curr_bw) < float(prev_bw) and float(drop_ratio) >= float(fluct_bw_frac)
        if start_idx is None:
            if bool(is_sudden_drop):
                start_idx = int(idx)
            continue
        if float(curr_bw) <= float(prev_bw):
            continue
        _append_span(int(start_idx), int(idx - 1))
        start_idx = int(idx) if bool(is_sudden_drop) else None
    if start_idx is not None:
        _append_span(int(start_idx), int(len(step_values) - 1))

    return spans


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Sage and BBR congestion-window decisions from a W&B CSV export.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("scripts/data/sage_bbr_trace.csv"),
        help="Path to the W&B export CSV.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to <csv-stem>_cwnd_plot.png next to the CSV.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Sage vs BBR Congestion Window",
        help="Plot title.",
    )
    parser.add_argument(
        "--step-min",
        type=int,
        default=4000,
        help="Minimum Step value to include.",
    )
    parser.add_argument(
        "--step-max",
        type=int,
        default=5000,
        help="Maximum Step value to include.",
    )
    parser.add_argument(
        "--fontsize",
        type=int,
        default=20,
        help="Base font size.",
    )
    parser.add_argument(
        "--bw-csv",
        type=Path,
        default="./data/sage_bbr_bw.csv",
        help="Optional bandwidth CSV used to shade unstable / low-bandwidth spans. Defaults to an inferred sibling *_bw.csv when present.",
    )
    parser.add_argument(
        "--bw-low-frac",
        type=float,
        default=0.5,
        help="Deprecated; low-bandwidth-only shading is disabled because spans now start only on sudden drops.",
    )
    parser.add_argument(
        "--bw-fluct-frac",
        type=float,
        default=0.5,
        help="Shade spans that begin when bandwidth drops by at least this fractional threshold.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv).expanduser().resolve()
    out_path = _default_output_path(csv_path) if args.out is None else Path(args.out).expanduser().resolve()
    bw_csv_path = None
    if args.bw_csv is not None:
        bw_csv_path = Path(args.bw_csv).expanduser().resolve()
    else:
        bw_csv_path = _default_bw_csv_path(csv_path)

    sns.set_style("ticks", {"grid.linestyle": ":"})
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["savefig.transparent"] = False

    time_offsets_s, sage_mb, bbr_mb = _load_series(
        csv_path,
        step_min=args.step_min,
        step_max=args.step_max,
    )
    shaded_spans: list[tuple[float, float]] = []
    if bw_csv_path is not None and bw_csv_path.exists():
        shaded_spans = _load_bandwidth_shaded_spans(
            bw_csv_path,
            step_min=args.step_min,
            step_max=args.step_max,
            low_bw_frac=float(args.bw_low_frac),
            fluct_bw_frac=float(args.bw_fluct_frac),
        )

    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    for x0, x1 in list(shaded_spans):
        ax.axvspan(float(x0), float(x1), facecolor="lightgray", alpha=1, edgecolor="none", zorder=0)
    ax.plot(time_offsets_s, sage_mb, color="r", linewidth=1.5, label="Sage")
    ax.plot(time_offsets_s, bbr_mb, color="b", linewidth=1.5, label="BBR")
    ax.tick_params(axis="x", labelsize=int(args.fontsize))
    ax.tick_params(axis="y", labelsize=int(args.fontsize))
    ax.set_xlabel("Time [s]", fontsize=int(args.fontsize) + 1)
    ax.set_ylabel("Congestion Window [MB]", fontsize=int(args.fontsize))
    # if str(args.title).strip():
    #     ax.set_title(str(args.title), fontsize=int(args.fontsize) + 3)
    ax.grid(True, linestyle=":", alpha=0.5)
    x_max = max([float(x) for x in list(time_offsets_s)]) if len(time_offsets_s) > 0 else 0.0
    ax.set_xlim([0, float(x_max)])
    y_max = max([float(x) for x in list(sage_mb) + list(bbr_mb)]) if len(sage_mb) + len(bbr_mb) > 0 else 0.0
    ax.set_ylim([0, max(float(y_max) * 1.08, _packets_to_mb(1.0))])
    legend_handles = [
        plt.Line2D([], [], color="r", linewidth=1.5, label="Sage"),
        plt.Line2D([], [], color="b", linewidth=1.5, label="BBR"),
    ]
    if len(shaded_spans) > 0:
        legend_handles.append(Patch(facecolor="lightgray", edgecolor="none", alpha=1, label="bandwidth drop"))
    ax.legend(
        handles=legend_handles,
        frameon=True,
        fontsize=int(args.fontsize),
        loc="lower center",
        bbox_to_anchor=(0.45, 1.02),
        ncol=3,
    )
    _style_axes(ax=ax)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"* Wrote plot: {out_path}")


if __name__ == "__main__":
    main()
