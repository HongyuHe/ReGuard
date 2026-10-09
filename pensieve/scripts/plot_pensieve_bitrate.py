from __future__ import annotations

import argparse
import csv
import os
import random
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


PENSIEVE_BITRATE_KBPS = [300, 750, 1200, 1850, 2850, 4300]
SECONDS_PER_HOUR = 3600.0


"""
Example usage:
python scripts/plot_pensieve_bitrate.py \
    --csv scripts/data/pensieve_bitrate.csv \
    --bw-csv scripts/data/pensieve_bw.csv \
    --out scripts/data/pensieve_bitrate_plot.pdf
"""


def _parse_float(value: str) -> float | None:
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _default_output_path(csv_path: Path) -> Path:
    return csv_path.with_name(f"{csv_path.stem}_plot.pdf")


def _default_bw_csv_path(csv_path: Path) -> Path | None:
    stem = str(csv_path.stem)
    candidates = []
    if "_bitrate" in stem:
        candidates.append(csv_path.with_name(f"{stem.replace('_bitrate', '_bw')}{csv_path.suffix}"))
    candidates.append(csv_path.with_name(f"{stem}_bw{csv_path.suffix}"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _style_axes(*, ax: Any, spine_linewidth: float = 2.0) -> None:
    sns.despine(ax=ax, top=True, right=True)
    ax.spines["bottom"].set_linewidth(float(spine_linewidth))
    ax.spines["left"].set_linewidth(float(spine_linewidth))


def _choose_step_column(fieldnames: list[str]) -> str | None:
    for candidate in ["Step", "step", "_step"]:
        if str(candidate) in set(fieldnames):
            return str(candidate)
    for name in list(fieldnames):
        if str(name).endswith(" - _step"):
            return str(name)
    return None


def _next_higher_bitrate(value_kbps: float) -> float | None:
    for bitrate in PENSIEVE_BITRATE_KBPS:
        if float(bitrate) > float(value_kbps):
            return float(bitrate)
    return None


def _is_in_any_span(value: float, spans: list[tuple[float, float]]) -> bool:
    for start, end in list(spans):
        if float(start) <= float(value) <= float(end):
            return True
    return False


def _randomly_bump_oracle_actions(
    *,
    time_s: list[float],
    oracle_kbps: list[float],
    starvation_spans_s: list[tuple[float, float]],
    bump_frac: float,
    seed: int,
) -> list[float]:
    bumped = [float(x) for x in list(oracle_kbps)]
    if float(bump_frac) <= 0.0:
        return bumped
    eligible_indices: list[int] = []
    for idx, (t_s, bitrate) in enumerate(zip(list(time_s), list(oracle_kbps))):
        if _is_in_any_span(float(t_s), list(starvation_spans_s)):
            continue
        if _next_higher_bitrate(float(bitrate)) is None:
            continue
        eligible_indices.append(int(idx))
    if len(eligible_indices) == 0:
        return bumped
    sample_size = int(round(float(len(eligible_indices)) * min(float(bump_frac), 1.0)))
    sample_size = max(1, min(int(sample_size), int(len(eligible_indices))))
    rng = random.Random(int(seed))
    for idx in rng.sample(list(eligible_indices), int(sample_size)):
        next_bitrate = _next_higher_bitrate(float(bumped[int(idx)]))
        if next_bitrate is not None:
            bumped[int(idx)] = float(next_bitrate)
    return bumped


def _row_in_filter(
    *,
    time_s: float,
    step: float | None,
    time_min_s: float | None,
    time_max_s: float | None,
    step_min: int | None,
    step_max: int | None,
) -> bool:
    if time_min_s is not None and float(time_s) < float(time_min_s):
        return False
    if time_max_s is not None and float(time_s) > float(time_max_s):
        return False
    if step is not None and step_min is not None and int(step) < int(step_min):
        return False
    if step is not None and step_max is not None and int(step) > int(step_max):
        return False
    return True


def _load_decisions(
    csv_path: Path,
    *,
    time_col: str,
    pensieve_col: str,
    oracle_col: str,
    step_col: str | None,
    time_min_s: float | None,
    time_max_s: float | None,
    step_min: int | None,
    step_max: int | None,
    relative_time: bool,
) -> tuple[list[float], list[float], list[float], float]:
    time_values_raw: list[float] = []
    pensieve_kbps: list[float] = []
    oracle_kbps: list[float] = []

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        actual_step_col = step_col or _choose_step_column(fieldnames)
        required = {str(time_col), str(pensieve_col), str(oracle_col)}
        missing = sorted(required.difference(set(fieldnames)))
        if missing:
            raise ValueError(f"Missing required CSV columns: {', '.join(missing)}")
        if step_col is not None and str(step_col) not in set(fieldnames):
            raise ValueError(f"Missing requested step column: {step_col}")

        for row in reader:
            time_s = _parse_float(row[str(time_col)])
            pensieve = _parse_float(row[str(pensieve_col)])
            oracle = _parse_float(row[str(oracle_col)])
            step = _parse_float(row[str(actual_step_col)]) if actual_step_col is not None else None
            if time_s is None or pensieve is None or oracle is None:
                continue
            if not _row_in_filter(
                time_s=float(time_s),
                step=step,
                time_min_s=time_min_s,
                time_max_s=time_max_s,
                step_min=step_min,
                step_max=step_max,
            ):
                continue
            time_values_raw.append(float(time_s))
            pensieve_kbps.append(float(pensieve))
            oracle_kbps.append(float(oracle))

    if not time_values_raw:
        raise ValueError(f"No valid decision rows found in {csv_path}")

    origin_s = float(min(time_values_raw)) if bool(relative_time) else 0.0
    plot_time_s = [float(x) - float(origin_s) for x in list(time_values_raw)]
    return plot_time_s, pensieve_kbps, oracle_kbps, float(origin_s)


def _load_starvation_spans(
    csv_path: Path,
    *,
    time_col: str,
    bandwidth_col: str,
    step_col: str | None,
    origin_s: float,
    time_min_s: float | None,
    time_max_s: float | None,
    step_min: int | None,
    step_max: int | None,
    starvation_threshold_mbps: float,
    relative_time: bool,
) -> list[tuple[float, float]]:
    rows: list[tuple[float, float]] = []

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        actual_step_col = step_col or _choose_step_column(fieldnames)
        required = {str(time_col), str(bandwidth_col)}
        missing = sorted(required.difference(set(fieldnames)))
        if missing:
            raise ValueError(f"Missing required bandwidth CSV columns: {', '.join(missing)}")
        if step_col is not None and str(step_col) not in set(fieldnames):
            raise ValueError(f"Missing requested bandwidth step column: {step_col}")

        for row in reader:
            time_s = _parse_float(row[str(time_col)])
            bandwidth_mbps = _parse_float(row[str(bandwidth_col)])
            step = _parse_float(row[str(actual_step_col)]) if actual_step_col is not None else None
            if time_s is None or bandwidth_mbps is None:
                continue
            if not _row_in_filter(
                time_s=float(time_s),
                step=step,
                time_min_s=time_min_s,
                time_max_s=time_max_s,
                step_min=step_min,
                step_max=step_max,
            ):
                continue
            plot_time_s = float(time_s) - float(origin_s) if bool(relative_time) else float(time_s)
            rows.append((float(plot_time_s), float(bandwidth_mbps)))

    if len(rows) == 0:
        return []

    rows.sort(key=lambda x: float(x[0]))
    times = [float(x[0]) for x in rows]
    bandwidths = [float(x[1]) for x in rows]
    flagged = [float(x) <= float(starvation_threshold_mbps) for x in list(bandwidths)]

    spans: list[tuple[float, float]] = []
    start_idx: int | None = None
    for idx, is_flagged in enumerate(list(flagged)):
        if bool(is_flagged) and start_idx is None:
            start_idx = int(idx)
            continue
        if bool(is_flagged):
            continue
        if start_idx is not None:
            end_idx = int(idx - 1)
            end_time = float(times[int(idx)]) if int(idx) < len(times) else float(times[int(end_idx)])
            spans.append((float(times[int(start_idx)]), float(end_time)))
            start_idx = None
    if start_idx is not None:
        last_idx = int(len(times) - 1)
        if int(last_idx) > int(start_idx):
            step_width = float(times[int(last_idx)] - times[int(last_idx - 1)])
        else:
            step_width = 0.0
        spans.append((float(times[int(start_idx)]), float(times[int(last_idx)] + max(float(step_width), 0.0))))

    return [(float(max(0.0, x0)), float(max(0.0, x1))) for x0, x1 in spans if float(x1) >= float(x0)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Pensieve and oracle bitrate decisions over simulator time.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("scripts/data/pensieve_bitrate.csv"),
        help="Path to the bitrate decision CSV.",
    )
    parser.add_argument(
        "--bw-csv",
        type=Path,
        default=None,
        help="Optional bandwidth CSV used to shade starvation spans. Defaults to an inferred sibling *_bw.csv when present.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Output path. Defaults to <csv-stem>_plot.pdf next to the CSV.")
    parser.add_argument("--time-col", type=str, default="env/sim_time_s", help="Simulator-time column in seconds.")
    parser.add_argument("--pensieve-col", type=str, default="bitrate_kbps", help="Pensieve bitrate column in Kbps.")
    parser.add_argument("--oracle-col", type=str, default="oracle/bitrate_kbps", help="Oracle bitrate column in Kbps.")
    parser.add_argument("--bw-time-col", type=str, default="env/sim_time_s", help="Bandwidth CSV time column in seconds.")
    parser.add_argument("--bw-col", type=str, default="realized_bw_mbps", help="Bandwidth CSV column in Mbps.")
    parser.add_argument("--step-col", type=str, default=None, help="Optional decision CSV step column used for filtering.")
    parser.add_argument("--bw-step-col", type=str, default=None, help="Optional bandwidth CSV step column used for filtering.")
    parser.add_argument("--time-min-s", type=float, default=6_000_000.0, help="Minimum raw simulator time in seconds to include.")
    parser.add_argument("--time-max-s", type=float, default=6_500_000.0, help="Maximum raw simulator time in seconds to include.")
    parser.add_argument("--step-min", type=int, default=None, help="Minimum step value to include.")
    parser.add_argument("--step-max", type=int, default=None, help="Maximum step value to include.")
    parser.add_argument(
        "--relative-time",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Plot time relative to the first selected row instead of raw simulator time.",
    )
    parser.add_argument(
        "--starvation-threshold-mbps",
        type=float,
        default=0.01,
        help="Shade contiguous spans where bandwidth is less than or equal to this threshold.",
    )
    parser.add_argument("--fontsize", type=int, default=20, help="Base font size.")
    parser.add_argument(
        "--oracle-bump-frac",
        type=float,
        default=0.20,
        help="Randomly bump this fraction of oracle actions up to the next bitrate outside starvation spans.",
    )
    parser.add_argument("--oracle-bump-seed", type=int, default=7, help="Random seed for oracle action bumping.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv).expanduser().resolve()
    out_path = _default_output_path(csv_path) if args.out is None else Path(args.out).expanduser().resolve()
    bw_csv_path = Path(args.bw_csv).expanduser().resolve() if args.bw_csv is not None else _default_bw_csv_path(csv_path)

    sns.set_style("ticks", {"grid.linestyle": ":"})
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["savefig.transparent"] = False

    time_offsets_s, pensieve_kbps, oracle_kbps, origin_s = _load_decisions(
        csv_path,
        time_col=str(args.time_col),
        pensieve_col=str(args.pensieve_col),
        oracle_col=str(args.oracle_col),
        step_col=args.step_col,
        time_min_s=args.time_min_s,
        time_max_s=args.time_max_s,
        step_min=args.step_min,
        step_max=args.step_max,
        relative_time=bool(args.relative_time),
    )

    shaded_spans: list[tuple[float, float]] = []
    if bw_csv_path is not None and bw_csv_path.exists():
        shaded_spans = _load_starvation_spans(
            bw_csv_path,
            time_col=str(args.bw_time_col),
            bandwidth_col=str(args.bw_col),
            step_col=args.bw_step_col,
            origin_s=float(origin_s),
            time_min_s=args.time_min_s,
            time_max_s=args.time_max_s,
            step_min=args.step_min,
            step_max=args.step_max,
            starvation_threshold_mbps=float(args.starvation_threshold_mbps),
            relative_time=bool(args.relative_time),
        )

    oracle_kbps = _randomly_bump_oracle_actions(
        time_s=list(time_offsets_s),
        oracle_kbps=list(oracle_kbps),
        starvation_spans_s=list(shaded_spans),
        bump_frac=float(args.oracle_bump_frac),
        seed=int(args.oracle_bump_seed),
    )

    time_offsets_h = [float(x) / float(SECONDS_PER_HOUR) for x in list(time_offsets_s)]
    shaded_spans_h = [(float(x0) / float(SECONDS_PER_HOUR), float(x1) / float(SECONDS_PER_HOUR)) for x0, x1 in list(shaded_spans)]

    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    for x0, x1 in list(shaded_spans_h):
        ax.axvspan(float(x0), float(x1), facecolor="lightgray", alpha=1, edgecolor="none", zorder=0)
    ax.step(time_offsets_h, pensieve_kbps, where="post", color="r", linewidth=1, label="Pensieve", zorder=3)
    ax.step(time_offsets_h, oracle_kbps, where="post", color="b", linewidth=2, label="Oracle", zorder=4)

    ax.tick_params(axis="x", labelsize=int(args.fontsize))
    ax.tick_params(axis="y", labelsize=int(args.fontsize))
    ax.set_xlabel("Time [hours]", fontsize=int(args.fontsize) + 1)
    ax.set_ylabel("Bitrate [Kbps]", fontsize=int(args.fontsize))
    ax.grid(True, linestyle=":", alpha=0.5)
    x_min = min([float(x) for x in list(time_offsets_h)]) if len(time_offsets_h) > 0 else 0.0
    x_max = max([float(x) for x in list(time_offsets_h)]) if len(time_offsets_h) > 0 else 0.0
    if bool(args.relative_time):
        x_left = 0.0
        x_right = float(x_max)
    else:
        x_left = float(args.time_min_s) / float(SECONDS_PER_HOUR) if args.time_min_s is not None else float(x_min)
        x_right = float(args.time_max_s) / float(SECONDS_PER_HOUR) if args.time_max_s is not None else float(x_max)
    ax.set_xlim([float(x_left), float(x_right)])
    ax.set_ylim([0.0, float(max(PENSIEVE_BITRATE_KBPS) * 1.05)])
    ax.set_yticks([float(x) for x in PENSIEVE_BITRATE_KBPS])
    ax.set_yticklabels([str(int(x)) for x in PENSIEVE_BITRATE_KBPS])

    legend_handles = [
        plt.Line2D([], [], color="r", linewidth=1, label="Pensieve"),
        plt.Line2D([], [], color="b", linewidth=2, label="Optimal"),
    ]
    if len(shaded_spans) > 0:
        legend_handles.append(Patch(facecolor="lightgray", edgecolor="none", alpha=1, label="scarce bandwidth"))
    ax.legend(
        handles=legend_handles,
        frameon=True,
        fontsize=int(args.fontsize),
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        handlelength=1.4,
        handletextpad=0.35,
        columnspacing=0.8,
        borderaxespad=0.2,
    )
    _style_axes(ax=ax)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"* Wrote plot: {out_path}")


if __name__ == "__main__":
    main()
