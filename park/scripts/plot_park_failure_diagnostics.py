from __future__ import annotations

"""
Example from the repo root:
python scripts/plot_park_failure_diagnostics.py \
  --clean-run wandb/run-20260410_090718-ddpogazk-clean \
  --maxgap-run wandb/run-20260410_090640-b8o3y8de-maxgap-eval \
  --out-dir online/eval/v2/plots/park_failure_diagnostics
"""

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any
from plot_eval_oracle_regret_stats import _disable_plot_clipping


SERVICE_RATES = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05]
BEST_BASELINE = "least_completion_time"
DEFAULT_CLEAN_RUN = "wandb/run-20260410_090718-ddpogazk-clean"
DEFAULT_MAXGAP_RUN = "wandb/run-20260410_090640-b8o3y8de-maxgap-eval"
DEFAULT_CLEAN_TRACE_DIR = "traces/clean"
DEFAULT_MAXGAP_TRACE_DIR = "traces/maxgap"
DEFAULT_DATA_DIR = "scripts/data"
DEFAULT_OUT_DIR = "online/eval/v2/plots/park_failure_diagnostics"
DEFAULT_TRACE_ID = "episode_0082.csv"
DEFAULT_FONT_SIZE = 19
PARK_COLOR = "#ff0000"
BASELINE_COLOR = "#023eff"
ORACLE_COLOR = "#ff7c00"
CLEAN_COLOR = "#a8d5f2"
MAXGAP_COLOR = "#f4a6a6"
GRID_COLOR = "#c4c4c4"
CONDITION_FILL_COLOR = "lightgray"
CONDITION_SLICE_JOB_SIZE_LABEL = "Job Size [KB]"
CONDITION_SLICE_SELECTION_LABEL = "Server Selection"
CONDITION_SLICE_GAP_LABEL = "Large gap"
CONDITION_SLICE_PARK_LABEL = "Park"
CONDITION_SLICE_LCT_LABEL = "LCT"
ACTION_SHIFT_NORMAL_LABEL = "Park under Normal"
ACTION_SHIFT_REGUARD_LABEL = "Park under ReGuard"
ACTION_SHIFT_REFERENCE_LABEL = "LCT under ReGuard"
ACTION_SHIFT_Y_LABEL = "Decisions [%]"


def _require_imports() -> tuple[Any, Any, Any, Any | None]:
    tmp = str(tempfile.gettempdir())
    os.environ.setdefault("XDG_CACHE_HOME", os.path.join(tmp, "xdg_cache"))
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(tmp, "matplotlib"))
    os.makedirs(str(os.environ["XDG_CACHE_HOME"]), exist_ok=True)
    os.makedirs(str(os.environ["MPLCONFIGDIR"]), exist_ok=True)

    try:
        import numpy as np
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: numpy/pandas") from exc

    try:
        import matplotlib

        matplotlib.use("Agg")
        matplotlib.rcParams["pdf.fonttype"] = 42
        matplotlib.rcParams["ps.fonttype"] = 42
        matplotlib.rcParams["axes.grid"] = True
        matplotlib.rcParams["grid.linestyle"] = ":"
        matplotlib.rcParams["grid.alpha"] = 0.5
        matplotlib.rcParams["savefig.transparent"] = False
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: matplotlib") from exc

    try:
        import seaborn as sns
    except ModuleNotFoundError:
        sns = None
    if sns is not None:
        sns.set_style("ticks", {"grid.linestyle": ":"})

    return pd, np, plt, sns


def _resolve_path(path: str | Path) -> Path:
    p = Path(os.path.expandvars(os.path.expanduser(str(path))))
    if p.is_absolute():
        return p
    return Path.cwd() / p


def _style_axes(
    ax: Any,
    *,
    fontsize: int | None = None,
    sns: Any | None = None,
    grid_axis: str = "y",
    spine_linewidth: float = 2.0,
) -> None:
    if sns is not None:
        sns.despine(ax=ax, top=True, right=True)
    else:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(float(spine_linewidth))
    ax.spines["left"].set_linewidth(float(spine_linewidth))
    if fontsize is not None:
        ax.tick_params(axis="x", labelsize=int(fontsize))
        ax.tick_params(axis="y", labelsize=int(fontsize))
    ax.grid(True, axis=str(grid_axis), color=GRID_COLOR, alpha=0.5, linewidth=0.9, linestyle=":")


def _style_legend(
    ax: Any,
    *,
    fontsize: int,
    loc: str,
    ncol: int = 1,
    bbox_to_anchor: tuple[float, float] | None = None,
    frameon: bool = True,
    handles: list[Any] | None = None,
) -> None:
    legend_kwargs = {
        "frameon": bool(frameon),
        "fancybox": True,
        # "framealpha": 1.0,
        "edgecolor": "black",
        "fontsize": int(fontsize),
        "loc": str(loc),
        "ncol": int(ncol),
        "bbox_to_anchor": bbox_to_anchor,
        "handlelength": 2.2,
    }
    if handles is not None:
        legend_kwargs["handles"] = handles
    legend = ax.legend(**legend_kwargs)
    if legend is not None:
        legend.get_frame().set_linewidth(0.9)


def _normalized_plot_steps(pd: Any, np: Any, rows: Any, keep_fraction: float) -> Any:
    if int(len(rows)) == 0:
        return rows.copy()
    keep_rows = max(2, int(np.ceil(float(len(rows)) * float(keep_fraction))))
    clipped = rows.iloc[: int(keep_rows)].copy()
    raw_steps = pd.to_numeric(clipped["step_idx"], errors="coerce").to_numpy(dtype=float)
    if int(raw_steps.size) == 0 or not np.isfinite(raw_steps).all():
        raw_steps = np.arange(int(len(clipped)), dtype=float)
    clipped["plot_step"] = raw_steps - float(raw_steps[0])
    return clipped


def _save_figure(fig: Any, out_dir: Path, stem: str, formats: list[str], dpi: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for fmt in list(formats):
        path = out_dir / f"{stem}.{fmt.lower()}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.08, dpi=int(dpi))
        paths.append(path)
    return paths


def _latest_table_json(run_dir: Path, table_name: str) -> Path:
    candidates = sorted((run_dir / "files" / "media" / "table" / "eval").glob(f"{table_name}_*.table.json"))
    if len(candidates) == 0:
        raise FileNotFoundError(f"missing W&B eval table {table_name} under {run_dir}")
    return candidates[-1]


def _read_wandb_table(pd: Any, run_dir: Path, table_name: str) -> Any:
    path = _latest_table_json(run_dir, table_name)
    obj = json.loads(path.read_text(encoding="utf-8"))
    columns = obj.get("columns")
    data = obj.get("data")
    if not isinstance(columns, list) or not isinstance(data, list):
        raise ValueError(f"invalid W&B table JSON: {path}")
    return pd.DataFrame(data, columns=[str(c) for c in list(columns)])


def _read_run_eval_frame(pd: Any, run_dir: Path, table_name: str) -> Any:
    summary_path = run_dir / "files" / "wandb-summary.json"
    csv_key = "paths/step_metrics_csv" if str(table_name) == "step_table" else "paths/episode_metrics_csv"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        csv_text = summary.get(str(csv_key))
        if csv_text:
            csv_path = _resolve_path(str(csv_text))
            if csv_path.is_file():
                return pd.read_csv(csv_path)
    return _read_wandb_table(pd, run_dir, str(table_name))


def _numeric(df: Any, col: str, default: float = 0.0) -> Any:
    if col not in df.columns:
        return default
    return df[col]


def _extract_compact_steps(pd: Any, np: Any, raw: Any, label: str, num_servers: int) -> Any:
    out = pd.DataFrame()
    base_cols = [
        "source",
        "trace_id",
        "trace_path",
        "step_idx",
        "arrival_time",
        "step_start_time",
        "job_size",
        "next_arrival_dt",
        "victim_action",
        "oracle_best_first_action",
        "oracle_trace_action",
        "regret",
        "victim_step_reward",
        "oracle_step_reward",
        "victim_cumulative_reward",
        "oracle_cumulative_reward",
        "oracle_gap_cumulative",
        f"baseline_action/{BEST_BASELINE}",
        f"baseline_step_reward/{BEST_BASELINE}",
        f"baseline_cumulative_reward/{BEST_BASELINE}",
        "victim/backlog_proxy_total_before",
        "victim/active_jobs_total_before",
        "victim/queue_jobs_total_before",
        "victim/queue_size_bytes_total_before",
        "oracle/backlog_proxy_total_before",
        "oracle/active_jobs_total_before",
        "oracle/queue_jobs_total_before",
        f"baseline/{BEST_BASELINE}/backlog_proxy_total_before",
        f"baseline/{BEST_BASELINE}/active_jobs_total_before",
        f"baseline/{BEST_BASELINE}/queue_jobs_total_before",
    ]
    for col in list(base_cols):
        if col in raw.columns:
            out[col] = raw[col]

    out["trace_set"] = str(label)
    out["victim_action"] = pd.to_numeric(out["victim_action"], errors="coerce").fillna(-1).astype(int)
    out["oracle_best_first_action"] = pd.to_numeric(out["oracle_best_first_action"], errors="coerce").fillna(-1).astype(int)
    lct_col = f"baseline_action/{BEST_BASELINE}"
    if lct_col in out.columns:
        out["lct_action"] = pd.to_numeric(out[lct_col], errors="coerce").fillna(-1).astype(int)
    else:
        out["lct_action"] = -1

    for prefix in ["victim", "oracle", f"baseline/{BEST_BASELINE}"]:
        short = prefix.replace("/", "_")
        load_cols = []
        for server_idx in range(int(num_servers)):
            col = f"{prefix}/server_load_proxy_{server_idx}_before"
            if col in raw.columns:
                out[col] = pd.to_numeric(raw[col], errors="coerce")
                load_cols.append(col)
        if len(load_cols) == int(num_servers):
            vals = raw[load_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            out[f"{short}_min_load"] = np.nanmin(vals, axis=1)
            out[f"{short}_max_load"] = np.nanmax(vals, axis=1)
            out[f"{short}_load_imbalance"] = out[f"{short}_max_load"] - out[f"{short}_min_load"]
            out[f"{short}_zero_load_servers"] = np.isclose(vals, 0.0).sum(axis=1)
            if prefix == "victim":
                actions = out["victim_action"].to_numpy(dtype=int)
                valid = (actions >= 0) & (actions < int(num_servers))
                selected = np.full(len(out), np.nan, dtype=float)
                selected[valid] = vals[np.arange(len(out))[valid], actions[valid]]
                ranks = np.full(len(out), np.nan, dtype=float)
                ranks[valid] = 1.0 + (vals[valid] < selected[valid, None] - 1e-9).sum(axis=1)
                out["victim_selected_load"] = selected
                out["victim_selected_load_rank"] = ranks
                out["victim_selected_gap_to_min"] = selected - np.nanmin(vals, axis=1)
                out["victim_selected_is_zero_load"] = np.isclose(selected, 0.0).astype(int)
                out["victim_selected_is_max_load"] = np.isclose(selected, np.nanmax(vals, axis=1)).astype(int)
                lct_actions = out["lct_action"].to_numpy(dtype=int)
                lct_valid = (lct_actions >= 0) & (lct_actions < int(num_servers))
                lct_state_load = np.full(len(out), np.nan, dtype=float)
                lct_state_load[lct_valid] = vals[np.arange(len(out))[lct_valid], lct_actions[lct_valid]]
                out["victim_state_lct_action_load"] = lct_state_load
                out["server9_idle_in_victim_state"] = np.isclose(vals[:, 9], 0.0).astype(int)
                out["any_fast_idle_in_victim_state"] = np.isclose(vals[:, 7:10], 0.0).any(axis=1).astype(int)

    rates = np.asarray(SERVICE_RATES, dtype=float)
    victim_actions = out["victim_action"].to_numpy(dtype=int)
    oracle_actions = out["oracle_best_first_action"].to_numpy(dtype=int)
    lct_actions = out["lct_action"].to_numpy(dtype=int)
    for name, actions in [
        ("victim", victim_actions),
        ("oracle_best_first", oracle_actions),
        ("lct", lct_actions),
    ]:
        selected_rates = np.full(len(out), np.nan, dtype=float)
        valid = (actions >= 0) & (actions < int(num_servers))
        selected_rates[valid] = rates[actions[valid]]
        out[f"{name}_service_rate"] = selected_rates

    out["park_matches_oracle"] = (out["victim_action"] == out["oracle_best_first_action"]).astype(int)
    out["park_matches_lct"] = (out["victim_action"] == out["lct_action"]).astype(int)
    out["park_slow_choice"] = out["victim_action"].isin([0, 1, 2, 3]).astype(int)
    out["park_slow_choice_fast_idle"] = (
        (out["park_slow_choice"] == 1) & (out["any_fast_idle_in_victim_state"] == 1)
    ).astype(int)
    out["park_server1_server9_idle"] = (
        (out["victim_action"] == 1) & (out["server9_idle_in_victim_state"] == 1)
    ).astype(int)
    out["park_server1_oracle9"] = (
        (out["victim_action"] == 1) & (out["oracle_best_first_action"] == 9)
    ).astype(int)
    out["step_idx"] = pd.to_numeric(out["step_idx"], errors="coerce").fillna(0).astype(int)
    out["step_start_time"] = pd.to_numeric(out["step_start_time"], errors="coerce")
    out["job_size"] = pd.to_numeric(out["job_size"], errors="coerce")
    out["regret"] = pd.to_numeric(out["regret"], errors="coerce")
    return out


def _extract_compact_episodes(pd: Any, raw: Any, label: str) -> Any:
    keep = [
        "trace_id",
        "trace_path",
        "num_jobs",
        "victim_reward_per_job",
        "oracle_reward_per_job",
        "oracle_gap_per_job",
        "regret_mean",
        "regret_p95",
        "victim_backlog_mean",
        "oracle_backlog_mean",
        f"baseline_backlog_mean/{BEST_BASELINE}",
        "victim_active_jobs_mean",
        "oracle_active_jobs_mean",
        f"baseline_active_jobs_mean/{BEST_BASELINE}",
        f"baseline_reward_per_job/{BEST_BASELINE}",
        "decision_diff_counterfactual_rate",
        "decision_diff_trace_rate",
    ]
    cols = [c for c in list(keep) if c in raw.columns]
    out = raw[cols].copy()
    out["trace_set"] = str(label)
    for col in list(out.columns):
        if col not in {"trace_id", "trace_path", "trace_set"}:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _max_run(mask: Any) -> int:
    current = 0
    best = 0
    for value in list(mask):
        if bool(value):
            current += 1
            best = max(int(best), int(current))
        else:
            current = 0
    return int(best)


def _acf1(np: Any, values: Any) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if int(arr.size) < 3:
        return float("nan")
    if float(np.nanstd(arr[:-1])) == 0.0 or float(np.nanstd(arr[1:])) == 0.0:
        return float("nan")
    return float(np.corrcoef(arr[:-1], arr[1:])[0, 1])


def _trace_family(path: Path, trace_set: str) -> str:
    if str(trace_set) == "maxgap":
        return "maxgap"
    if "_trace_" in path.name:
        return path.name.split("_trace_", 1)[0]
    return path.stem


def _build_trace_stats(pd: Any, np: Any, trace_dirs: dict[str, Path]) -> Any:
    rows: list[dict[str, Any]] = []
    for trace_set, root in dict(trace_dirs).items():
        for path in sorted(Path(root).glob("*.csv")):
            if path.name == "index.csv":
                continue
            df = pd.read_csv(path)
            if "arrival_time" not in df.columns or "job_size" not in df.columns:
                continue
            times = pd.to_numeric(df["arrival_time"], errors="coerce").to_numpy(dtype=float)
            sizes = pd.to_numeric(df["job_size"], errors="coerce").to_numpy(dtype=float)
            if int(sizes.size) == 0:
                continue
            dt = np.diff(times)
            big = sizes >= 5000.0
            small = sizes <= 200.0
            paced_size_mask = (
                (sizes[1:] >= 126.0)
                & (sizes[1:] <= 525.0)
                & (dt >= 450.0)
                & (dt <= 550.0)
            ) if int(dt.size) > 0 else []
            rows.append(
                {
                    "trace_set": str(trace_set),
                    "family": _trace_family(path, str(trace_set)),
                    "trace_id": path.name,
                    "trace_path": str(path),
                    "num_jobs": int(sizes.size),
                    "duration": float(times[-1] - times[0]) if int(times.size) > 0 else float("nan"),
                    "mean_dt": float(np.nanmean(dt)) if int(dt.size) > 0 else float("nan"),
                    "median_dt": float(np.nanmedian(dt)) if int(dt.size) > 0 else float("nan"),
                    "cv_dt": float(np.nanstd(dt) / max(float(np.nanmean(dt)), 1e-12)) if int(dt.size) > 0 else float("nan"),
                    "frac_dt_le_10": float(np.nanmean(dt <= 10.0)) if int(dt.size) > 0 else float("nan"),
                    "frac_dt_450_550": float(np.nanmean((dt >= 450.0) & (dt <= 550.0))) if int(dt.size) > 0 else float("nan"),
                    "mean_size": float(np.nanmean(sizes)),
                    "median_size": float(np.nanmedian(sizes)),
                    "p90_size": float(np.nanquantile(sizes, 0.90)),
                    "p95_size": float(np.nanquantile(sizes, 0.95)),
                    "max_size": float(np.nanmax(sizes)),
                    "frac_size_126_525": float(np.nanmean((sizes >= 126.0) & (sizes <= 525.0))),
                    "frac_paced_size_126_525": float(np.nanmean(paced_size_mask)) if int(dt.size) > 0 else float("nan"),
                    "frac_size_ge_5000": float(np.nanmean(big)),
                    "max_run_ge_5000": _max_run(big),
                    "max_run_le_200": _max_run(small),
                    "acf1_size": _acf1(np, sizes),
                    "first100_mean_size": float(np.nanmean(sizes[:100])),
                    "last100_mean_size": float(np.nanmean(sizes[-100:])),
                    "first100_frac_ge_5000": float(np.nanmean(sizes[:100] >= 5000.0)),
                    "last100_frac_ge_5000": float(np.nanmean(sizes[-100:] >= 5000.0)),
                }
            )
    return pd.DataFrame(rows)


def _load_or_create_processed(
    pd: Any,
    np: Any,
    *,
    clean_run: Path,
    maxgap_run: Path,
    clean_trace_dir: Path,
    maxgap_trace_dir: Path,
    data_dir: Path,
    force: bool,
    num_servers: int,
) -> tuple[Any, Any, Any, dict[str, Path]]:
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "steps": data_dir / "park_failure_steps.csv",
        "episodes": data_dir / "park_failure_episodes.csv",
        "trace_stats": data_dir / "park_failure_trace_stats.csv",
        "action_summary": data_dir / "park_failure_action_summary.csv",
        "condition_summary": data_dir / "park_failure_condition_summary.csv",
    }
    if all(path.exists() for path in paths.values()) and not bool(force):
        steps = pd.read_csv(paths["steps"])
        episodes = pd.read_csv(paths["episodes"])
        trace_stats = pd.read_csv(paths["trace_stats"])
        if "frac_paced_size_126_525" in set(trace_stats.columns):
            return steps, episodes, trace_stats, paths

    clean_steps_raw = _read_run_eval_frame(pd, clean_run, "step_table")
    maxgap_steps_raw = _read_run_eval_frame(pd, maxgap_run, "step_table")
    clean_episodes_raw = _read_run_eval_frame(pd, clean_run, "episode_table")
    maxgap_episodes_raw = _read_run_eval_frame(pd, maxgap_run, "episode_table")

    clean_steps = _extract_compact_steps(pd, np, clean_steps_raw, "clean", int(num_servers))
    maxgap_steps = _extract_compact_steps(pd, np, maxgap_steps_raw, "maxgap", int(num_servers))
    steps = pd.concat([clean_steps, maxgap_steps], ignore_index=True)
    episodes = pd.concat(
        [
            _extract_compact_episodes(pd, clean_episodes_raw, "clean"),
            _extract_compact_episodes(pd, maxgap_episodes_raw, "maxgap"),
        ],
        ignore_index=True,
    )
    trace_stats = _build_trace_stats(
        pd,
        np,
        {
            "clean": clean_trace_dir,
            "maxgap": maxgap_trace_dir,
        },
    )
    action_summary = _build_action_summary(pd, steps, int(num_servers))
    condition_summary = _build_condition_summary(pd, steps)

    steps.to_csv(paths["steps"], index=False)
    episodes.to_csv(paths["episodes"], index=False)
    trace_stats.to_csv(paths["trace_stats"], index=False)
    action_summary.to_csv(paths["action_summary"], index=False)
    condition_summary.to_csv(paths["condition_summary"], index=False)
    return steps, episodes, trace_stats, paths


def _build_action_summary(pd: Any, steps: Any, num_servers: int) -> Any:
    rows: list[dict[str, Any]] = []
    for trace_set, group in steps.groupby("trace_set"):
        for policy_name, col in [
            ("Park", "victim_action"),
            ("State-aligned oracle", "oracle_best_first_action"),
            ("LCT baseline", "lct_action"),
        ]:
            if col not in group.columns:
                continue
            for server_idx in range(int(num_servers)):
                rows.append(
                    {
                        "trace_set": str(trace_set),
                        "policy": str(policy_name),
                        "server": int(server_idx),
                        "service_rate": float(SERVICE_RATES[int(server_idx)]),
                        "fraction": float((group[col].astype(int) == int(server_idx)).mean()),
                    }
                )
    return pd.DataFrame(rows)


def _build_condition_summary(pd: Any, steps: Any) -> Any:
    rows: list[dict[str, Any]] = []
    condition_specs = [
        ("all", None),
        ("Park slow server", "park_slow_choice"),
        ("Park slow server while fast server idle", "park_slow_choice_fast_idle"),
        ("Park server 1 while server 9 idle", "park_server1_server9_idle"),
        ("Park server 1, oracle server 9", "park_server1_oracle9"),
        ("Park matches oracle", "park_matches_oracle"),
    ]
    for trace_set, group in steps.groupby("trace_set"):
        for label, col in list(condition_specs):
            if col is None:
                sub = group
            elif col in group.columns:
                sub = group[group[col] == 1]
            else:
                sub = group.iloc[0:0]
            rows.append(
                {
                    "trace_set": str(trace_set),
                    "condition": str(label),
                    "count": int(len(sub)),
                    "fraction": float(len(sub) / max(1, len(group))),
                    "regret_mean": float(pd.to_numeric(sub.get("regret", []), errors="coerce").mean()) if len(sub) else float("nan"),
                    "regret_median": float(pd.to_numeric(sub.get("regret", []), errors="coerce").median()) if len(sub) else float("nan"),
                    "job_size_mean": float(pd.to_numeric(sub.get("job_size", []), errors="coerce").mean()) if len(sub) else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def _pct(np: Any, mask: Any) -> float:
    arr = np.asarray(mask, dtype=float).reshape(-1)
    if int(arr.size) == 0:
        return float("nan")
    return 100.0 * float(np.nanmean(arr))


def _mean(pd: Any, values: Any) -> float:
    series = pd.to_numeric(values, errors="coerce")
    if int(len(series)) == 0:
        return float("nan")
    return float(series.mean())


def _median(pd: Any, values: Any) -> float:
    series = pd.to_numeric(values, errors="coerce")
    if int(len(series)) == 0:
        return float("nan")
    return float(series.median())


def _format_value(value: float, digits: int = 1) -> str:
    if value != value:
        return "NA"
    return f"{float(value):.{int(digits)}f}"


def _format_pct(value: float, digits: int = 1) -> str:
    if value != value:
        return "NA"
    return f"{float(value):.{int(digits)}f}%"


def _contiguous_run_lengths(np: Any, positions: Any) -> tuple[list[int], list[int]]:
    idx = np.asarray(positions, dtype=int).reshape(-1)
    if int(idx.size) == 0:
        return [], []
    lengths: list[int] = []
    gaps: list[int] = []
    run_start = int(idx[0])
    run_end = int(idx[0])
    for pos in list(idx[1:]):
        pos_i = int(pos)
        if int(pos_i) == int(run_end) + 1:
            run_end = int(pos_i)
            continue
        lengths.append(int(run_end - run_start + 1))
        gaps.append(int(pos_i - run_end - 1))
        run_start = int(pos_i)
        run_end = int(pos_i)
    lengths.append(int(run_end - run_start + 1))
    return lengths, gaps


def _build_condition_slice_description(pd: Any, np: Any, steps: Any, trace_id: str, window: int) -> tuple[str, str]:
    rows = steps[(steps["trace_set"] == "maxgap") & (steps["trace_id"] == str(trace_id))].copy().sort_values("step_idx")
    if int(len(rows)) == 0:
        raise FileNotFoundError(f"missing maxgap trace rows for {trace_id}")
    lo, hi = _extract_zoom_window(np, rows, int(window))
    zoom = _normalized_plot_steps(pd, np, rows.iloc[int(lo) : int(hi)].copy(), keep_fraction=2.0 / 3.0)
    trace_stem = f"maxgap_condition_slice_{Path(trace_id).stem}"

    job_size = pd.to_numeric(zoom["job_size"], errors="coerce")
    small = job_size <= 200.0
    paced = (job_size >= 126.0) & (job_size <= 525.0)
    large = job_size >= 5000.0
    condition = pd.to_numeric(zoom["park_server1_oracle9"], errors="coerce").fillna(0.0) > 0.5

    large_positions = np.flatnonzero(large.to_numpy(dtype=bool))
    burst_lengths, inter_burst_gaps = _contiguous_run_lengths(np, large_positions)
    if int(len(burst_lengths)) > 0:
        burst_text = (
            f"The 5000-B jobs appear in {int(len(burst_lengths))} short bursts "
            f"(median burst length {_format_value(float(np.median(np.asarray(burst_lengths, dtype=float))), 1)} jobs)"
        )
        if int(len(inter_burst_gaps)) > 0:
            burst_text += (
                f", separated by {_format_value(float(np.min(np.asarray(inter_burst_gaps, dtype=float))), 0)}-"
                f"{_format_value(float(np.max(np.asarray(inter_burst_gaps, dtype=float))), 0)} smaller-job decisions"
            )
        burst_text += "."
    else:
        burst_text = ""

    small_rows = zoom[small].copy()
    large_rows = zoom[large].copy()
    condition_rows = zoom[condition].copy()

    description = (
        f"In the plotted {int(len(zoom))}-step `{CONDITION_SLICE_JOB_SIZE_LABEL}` / `{CONDITION_SLICE_SELECTION_LABEL}` slice, "
        f"{_format_pct(_pct(np, small), 1)} of jobs are <=0.2 KB, {_format_pct(_pct(np, paced), 1)} fall in the 0.126-0.525 KB range "
        f"that the fast servers can easily absorb, and only {_format_pct(_pct(np, large), 1)} are 5.0 KB jobs. {burst_text} "
        f"`{CONDITION_SLICE_PARK_LABEL}` responds to the small jobs by collapsing onto the slow servers: for <=0.2 KB jobs it chooses a slow server "
        f"{_format_pct(_pct(np, small_rows['victim_action'].astype(int).isin([0, 1, 2, 3])), 1)} of the time and server 1 alone "
        f"{_format_pct(_pct(np, small_rows['victim_action'].astype(int) == 1), 1)}, while `{CONDITION_SLICE_LCT_LABEL}` sends "
        f"{_format_pct(_pct(np, small_rows['lct_action'].astype(int).isin([7, 8, 9])), 1)} of those same jobs to the fast servers "
        f"and {_format_pct(_pct(np, small_rows['lct_action'].astype(int) == 9), 1)} to the fastest server 9. "
        f"`{CONDITION_SLICE_PARK_LABEL}` flips only when the 5.0 KB bursts arrive, choosing fast servers on "
        f"{_format_pct(_pct(np, large_rows['victim_action'].astype(int).isin([7, 8, 9])), 1)} of >=5.0 KB jobs. "
        f"The `{CONDITION_SLICE_GAP_LABEL}` region occupies {_format_pct(_pct(np, condition), 1)} of the slice; during this entire slice at least one fast server is idle on "
        f"{_format_pct(_pct(np, zoom['any_fast_idle_in_victim_state'].astype(int) == 1), 1)} of steps, server 9 is idle on "
        f"{_format_pct(_pct(np, zoom['server9_idle_in_victim_state'].astype(int) == 1), 1)} of steps, and `{CONDITION_SLICE_PARK_LABEL}` still sends traffic to a slow server on "
        f"{_format_pct(_pct(np, zoom['park_slow_choice'] == 1), 1)} of steps and to server 1 on "
        f"{_format_pct(_pct(np, zoom['victim_action'].astype(int) == 1), 1)}. "
        f"Inside `{CONDITION_SLICE_GAP_LABEL}`, `{CONDITION_SLICE_PARK_LABEL}` always picks server 1, never picks a fast server, has mean chosen service rate "
        f"{_format_value(_mean(pd, condition_rows['victim_service_rate']), 2)} versus "
        f"{_format_value(_mean(pd, condition_rows['lct_service_rate']), 2)} for `{CONDITION_SLICE_LCT_LABEL}`, selects a zero-load server only "
        f"{_format_pct(_pct(np, condition_rows['victim_selected_is_zero_load'] == 1), 1)}, and incurs mean/median regret "
        f"{_format_value(_mean(pd, condition_rows['regret']), 2)}/"
        f"{_format_value(_median(pd, condition_rows['regret']), 2)}. "
        f"These statistics show the concrete failure mode: the attacker packs the slice with small jobs, `{CONDITION_SLICE_PARK_LABEL}` overloads the slow servers, "
        f"and the fast servers remain under-utilized even when they are idle."
    )
    return trace_stem, description


def _build_action_shift_description(pd: Any, np: Any, steps: Any) -> tuple[str, str]:
    clean = steps[steps["trace_set"] == "clean"].copy()
    maxgap = steps[steps["trace_set"] == "maxgap"].copy()
    trace_stem = "park_decision_shift_clean_vs_maxgap"

    clean_server1 = _pct(np, clean["victim_action"].astype(int) == 1)
    maxgap_server1 = _pct(np, maxgap["victim_action"].astype(int) == 1)
    clean_slow = _pct(np, clean["victim_action"].astype(int).isin([0, 1, 2, 3]))
    maxgap_slow = _pct(np, maxgap["victim_action"].astype(int).isin([0, 1, 2, 3]))
    clean_fast = _pct(np, clean["victim_action"].astype(int).isin([7, 8, 9]))
    maxgap_fast = _pct(np, maxgap["victim_action"].astype(int).isin([7, 8, 9]))
    lct_maxgap_fast = _pct(np, maxgap["lct_action"].astype(int).isin([7, 8, 9]))
    lct_maxgap_slow = _pct(np, maxgap["lct_action"].astype(int).isin([0, 1, 2, 3]))
    lct_maxgap_server9 = _pct(np, maxgap["lct_action"].astype(int) == 9)
    oracle_maxgap_fast = _pct(np, maxgap["oracle_best_first_action"].astype(int).isin([7, 8, 9]))
    oracle_maxgap_server9 = _pct(np, maxgap["oracle_best_first_action"].astype(int) == 9)
    clean_rate = _mean(pd, clean["victim_service_rate"])
    maxgap_rate = _mean(pd, maxgap["victim_service_rate"])
    lct_maxgap_rate = _mean(pd, maxgap["lct_service_rate"])

    description = (
        f"Across all `{ACTION_SHIFT_Y_LABEL}`, the attacker shifts `{ACTION_SHIFT_NORMAL_LABEL}` to a near-collapse onto slow server 1 in `{ACTION_SHIFT_REGUARD_LABEL}`. "
        f"`{ACTION_SHIFT_REGUARD_LABEL}` places {_format_pct(maxgap_server1, 1)} of decisions on server 1 versus {_format_pct(clean_server1, 1)} for `{ACTION_SHIFT_NORMAL_LABEL}` "
        f"({maxgap_server1 - clean_server1:+.1f} percentage points), while its overall slow-server share rises from "
        f"{_format_pct(clean_slow, 1)} to {_format_pct(maxgap_slow, 1)} and its fast-server share falls from "
        f"{_format_pct(clean_fast, 1)} to {_format_pct(maxgap_fast, 1)}. "
        f"`{ACTION_SHIFT_REFERENCE_LABEL}` still routes {_format_pct(lct_maxgap_fast, 1)} of decisions to the fast servers and "
        f"{_format_pct(lct_maxgap_server9, 1)} to the fastest server 9, with only {_format_pct(lct_maxgap_slow, 1)} going to the slow servers. "
        f"The attacker therefore is not simply increasing load everywhere: in `{ACTION_SHIFT_REGUARD_LABEL}` at least one fast server is idle on "
        f"{_format_pct(_pct(np, maxgap['any_fast_idle_in_victim_state'].astype(int) == 1), 1)} of steps, yet `{ACTION_SHIFT_REGUARD_LABEL}` still chooses a slow server with a fast server idle on "
        f"{_format_pct(_pct(np, maxgap['park_slow_choice_fast_idle'] == 1), 1)} of steps and chooses server 1 while server 9 is idle on "
        f"{_format_pct(_pct(np, maxgap['park_server1_server9_idle'] == 1), 1)} of steps. "
        f"Consistent with that bias, the mean chosen service rate drops from {_format_value(clean_rate, 3)} for `{ACTION_SHIFT_NORMAL_LABEL}` to "
        f"{_format_value(maxgap_rate, 3)} for `{ACTION_SHIFT_REGUARD_LABEL}`, while `{ACTION_SHIFT_REFERENCE_LABEL}` still averages {_format_value(lct_maxgap_rate, 3)}. "
        f"This is systematic slow-server over-utilization and fast-server under-utilization, not a benign redistribution of load."
    )
    return trace_stem, description


def _write_figure_descriptions(out_dir: Path, descriptions: list[tuple[str, str]]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# Figure Descriptions", ""]
    for stem, text in list(descriptions):
        lines.append(f"## `{str(stem)}`")
        lines.append("")
        lines.append(str(text).strip())
        lines.append("")
    path = out_dir / "figure_descriptions.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _plot_trace_characteristics(pd: Any, np: Any, plt: Any, trace_stats: Any, out_dir: Path, formats: list[str], dpi: int) -> list[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))
    labels = ["clean", "maxgap"]
    colors = {"clean": CLEAN_COLOR, "maxgap": MAXGAP_COLOR}
    display = {"clean": "Clean", "maxgap": "Adversarial maxgap"}

    ax = axes[0, 0]
    parts = ax.violinplot(
        [np.log10(trace_stats[trace_stats["trace_set"] == x]["median_dt"].astype(float) + 1.0) for x in labels],
        positions=[0, 1],
        showmeans=True,
        showextrema=False,
    )
    for idx, body in enumerate(parts["bodies"]):
        body.set_facecolor(colors[labels[idx]])
        body.set_alpha(0.72)
    parts["cmeans"].set_color("#1e1e1e")
    ax.set_xticks([0, 1], [display[x] for x in labels])
    ax.set_ylabel("log10(median inter-arrival + 1)")
    ax.set_title("Pacing regime")
    _style_axes(ax)

    ax = axes[0, 1]
    parts = ax.violinplot(
        [trace_stats[trace_stats["trace_set"] == x]["frac_paced_size_126_525"].astype(float) for x in labels],
        positions=[0, 1],
        showmeans=True,
        showextrema=False,
    )
    for idx, body in enumerate(parts["bodies"]):
        body.set_facecolor(colors[labels[idx]])
        body.set_alpha(0.72)
    parts["cmeans"].set_color("#1e1e1e")
    ax.set_xticks([0, 1], [display[x] for x in labels])
    ax.set_ylabel("Fraction with dt 450-550 and size 126-525")
    ax.set_title("Paced jobs fast servers clear, server 1 cannot")
    _style_axes(ax)

    ax = axes[1, 0]
    parts = ax.violinplot(
        [trace_stats[trace_stats["trace_set"] == x]["max_run_ge_5000"].astype(float) for x in labels],
        positions=[0, 1],
        showmeans=True,
        showextrema=False,
    )
    for idx, body in enumerate(parts["bodies"]):
        body.set_facecolor(colors[labels[idx]])
        body.set_alpha(0.72)
    parts["cmeans"].set_color("#1e1e1e")
    ax.set_xticks([0, 1], [display[x] for x in labels])
    ax.set_ylabel("Max consecutive 5000-size jobs")
    ax.set_title("Large-job clustering")
    _style_axes(ax)

    ax = axes[1, 1]
    parts = ax.violinplot(
        [trace_stats[trace_stats["trace_set"] == x]["acf1_size"].dropna().astype(float) for x in labels],
        positions=[0, 1],
        showmeans=True,
        showextrema=False,
    )
    for idx, body in enumerate(parts["bodies"]):
        body.set_facecolor(colors[labels[idx]])
        body.set_alpha(0.72)
    parts["cmeans"].set_color("#1e1e1e")
    ax.axhline(0.0, color="#1e1e1e", linewidth=1.0, alpha=0.7)
    ax.set_xticks([0, 1], [display[x] for x in labels])
    ax.set_ylabel("Lag-1 job-size autocorrelation")
    ax.set_title("Temporal structure")
    _style_axes(ax)

    fig.suptitle("Why maxgap is different: paced, correlated moderate/large jobs", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save_figure(fig, out_dir, "trace_characteristics_clean_vs_maxgap", formats, int(dpi))


def _plot_action_shift(
    pd: Any,
    np: Any,
    plt: Any,
    sns: Any | None,
    steps: Any,
    out_dir: Path,
    formats: list[str],
    dpi: int,
    fontsize: int,
) -> list[Path]:
    fig, ax = plt.subplots(1, 1, figsize=(18.0, 6.2))
    servers = np.arange(len(SERVICE_RATES))
    width = 0.24
    clean = steps[steps["trace_set"] == "clean"]
    maxgap = steps[steps["trace_set"] == "maxgap"]
    clean_pct = 100.0 * np.asarray([(clean["victim_action"].astype(int) == i).mean() for i in servers], dtype=float)
    maxgap_pct = 100.0 * np.asarray([(maxgap["victim_action"].astype(int) == i).mean() for i in servers], dtype=float)
    oracle_pct = 100.0 * np.asarray([(maxgap["oracle_best_first_action"].astype(int) == i).mean() for i in servers], dtype=float)

    ax.bar(
        servers - width,
        clean_pct,
        width=width,
        color=CLEAN_COLOR,
        edgecolor="black",
        linewidth=0.9,
        label="Park under Normal",
    )
    ax.bar(
        servers,
        maxgap_pct,
        width=width,
        color=PARK_COLOR,
        edgecolor="black",
        linewidth=0.9,
        label="Park under ReGuard",
    )
    ax.bar(
        servers + width,
        oracle_pct,
        width=width,
        color=ORACLE_COLOR,
        edgecolor="black",
        linewidth=0.9,
        label="LCT under ReGuard",
    )
    ax.set_xticks(servers, [f"Server {i}\nspeed: {SERVICE_RATES[i]:.1f}" for i in servers])
    # ax.set_xlabel("Server ID and service rate", fontsize=int(fontsize) + 1)
    ax.set_ylabel("Decisions [%]", fontsize=int(fontsize)+1)
    # ax.set_title("", fontsize=int(fontsize) + 1)
    ymax = float(np.nanmax(np.concatenate([clean_pct, maxgap_pct, oracle_pct]))) if int(len(servers)) > 0 else 0.0
    ax.set_ylim(0.0, 80) #max(5.0, float(np.ceil(float(ymax) / 5.0) * 5.0)))
    _style_axes(ax, fontsize=int(fontsize)+2, sns=sns, grid_axis="y")
    ax.tick_params(axis="x", labelsize=int(fontsize)+1, pad=0, rotation=10)
    _style_legend(ax, fontsize=int(fontsize)+2, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.15), frameon=False)
    # fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.80))
    _disable_plot_clipping(fig=fig)
    return _save_figure(fig, out_dir, "park_decision_shift_clean_vs_maxgap", formats, int(dpi))


def _choose_failure_trace(episodes: Any, trace_id: str | None) -> str:
    maxgap = episodes[episodes["trace_set"] == "maxgap"].copy()
    if trace_id:
        return str(trace_id)
    if len(maxgap) == 0 or "oracle_gap_per_job" not in maxgap.columns:
        return "episode_0000.csv"
    row = maxgap.sort_values("oracle_gap_per_job", ascending=False).iloc[0]
    return str(row["trace_id"])


def _extract_zoom_window(np: Any, trace_rows: Any, max_points: int) -> tuple[int, int]:
    cond_cols = ["park_server1_oracle9", "park_server1_server9_idle", "park_slow_choice_fast_idle"]
    cond = np.zeros(len(trace_rows), dtype=bool)
    for col in list(cond_cols):
        if col in trace_rows.columns:
            cond |= trace_rows[col].to_numpy(dtype=float) > 0.5
    if not np.any(cond):
        peak_idx = int(trace_rows["regret"].astype(float).idxmax())
        peak_pos = int(np.where(trace_rows.index.to_numpy() == peak_idx)[0][0])
        lo = max(0, int(peak_pos) - int(max_points) // 2)
        hi = min(len(trace_rows), int(lo) + int(max_points))
        return int(lo), int(hi)
    positions = np.where(cond)[0]
    best_start = int(positions[0])
    best_len = 1
    current_start = int(positions[0])
    current_len = 1
    for prev, pos in zip(positions[:-1], positions[1:]):
        if int(pos) == int(prev) + 1:
            current_len += 1
        else:
            if int(current_len) > int(best_len):
                best_start = int(current_start)
                best_len = int(current_len)
            current_start = int(pos)
            current_len = 1
    if int(current_len) > int(best_len):
        best_start = int(current_start)
        best_len = int(current_len)
    center = int(best_start + best_len // 2)
    lo = max(0, int(center) - int(max_points) // 2)
    hi = min(len(trace_rows), int(lo) + int(max_points))
    lo = max(0, int(hi) - int(max_points))
    return int(lo), int(hi)


def _plot_failure_trajectory(pd: Any, np: Any, plt: Any, steps: Any, trace_id: str, out_dir: Path, formats: list[str], dpi: int) -> list[Path]:
    rows = steps[(steps["trace_set"] == "maxgap") & (steps["trace_id"] == str(trace_id))].copy()
    if len(rows) == 0:
        return []
    rows = rows.sort_values("step_idx")
    x = rows["step_idx"].to_numpy(dtype=float)
    fig, axes = plt.subplots(4, 1, figsize=(14.0, 10.5), sharex=True)

    ax = axes[0]
    ax.plot(x, rows["job_size"], color="#2c2c2c", linewidth=1.0)
    ax.fill_between(x, 0, rows["job_size"], where=rows["job_size"].to_numpy(dtype=float) >= 5000.0, color=MAXGAP_COLOR, alpha=0.32)
    ax.set_ylabel("Job size")
    ax.set_title("Adversarial input: paced moderate jobs plus clustered max-size jobs")
    _style_axes(ax)

    ax = axes[1]
    ax.scatter(x, rows["victim_action"], s=12, color=PARK_COLOR, alpha=0.85, label="Park")
    ax.scatter(x, rows["lct_action"], s=10, color=BASELINE_COLOR, alpha=0.72, label="LCT baseline")
    ax.scatter(x, rows["oracle_best_first_action"], s=8, color=ORACLE_COLOR, alpha=0.50, label="State-aligned best first action")
    ax.set_yticks(range(len(SERVICE_RATES)), [f"{i} ({SERVICE_RATES[i]:.2f})" for i in range(len(SERVICE_RATES))])
    ax.set_ylabel("Action server")
    ax.set_title("Decision divergence: Park repeatedly chooses slow server 1")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    _style_axes(ax)

    ax = axes[2]
    ax.plot(x, rows["victim/backlog_proxy_total_before"], color=PARK_COLOR, linewidth=2.0, label="Park")
    ax.plot(x, rows[f"baseline/{BEST_BASELINE}/backlog_proxy_total_before"], color=BASELINE_COLOR, linewidth=2.0, label="LCT baseline")
    ax.plot(x, rows["oracle/backlog_proxy_total_before"], color=ORACLE_COLOR, linewidth=1.7, label="Oracle rollout")
    ax.set_ylabel("Backlog proxy")
    ax.set_title("Impact: Park accumulates backlog that LCT avoids")
    ax.legend(frameon=False, loc="upper left")
    _style_axes(ax)

    ax = axes[3]
    ax.plot(x, rows["victim_selected_load_rank"], color=PARK_COLOR, linewidth=1.8, label="Park chosen-server load rank")
    ax.plot(x, rows["victim_zero_load_servers"], color="#545454", linewidth=1.4, label="Zero-load servers in Park state")
    if "oracle_gap_cumulative" in rows.columns:
        ax2 = ax.twinx()
        ax2.plot(x, rows["oracle_gap_cumulative"], color="#8b1e3f", linewidth=1.6, alpha=0.75, label="Cumulative oracle gap")
        ax2.set_ylabel("Cumulative gap")
        ax2.spines["top"].set_visible(False)
        ax2.grid(False)
    ax.set_xlabel("Step")
    ax.set_ylabel("Rank / count")
    ax.set_title("State evidence: many idle servers, but Park chooses high-rank loaded servers")
    lines, labels = ax.get_legend_handles_labels()
    if "ax2" in locals():
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines += lines2
        labels += labels2
    ax.legend(lines, labels, frameon=False, loc="upper left")
    _style_axes(ax)

    fig.suptitle(f"Park failure trajectory on {trace_id}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save_figure(fig, out_dir, f"maxgap_failure_trajectory_{Path(trace_id).stem}", formats, int(dpi))


def _plot_condition_slice(
    pd: Any,
    np: Any,
    plt: Any,
    sns: Any | None,
    steps: Any,
    trace_id: str,
    out_dir: Path,
    formats: list[str],
    dpi: int,
    window: int,
    fontsize: int,
) -> list[Path]:
    from matplotlib.patches import Patch

    rows = steps[(steps["trace_set"] == "maxgap") & (steps["trace_id"] == str(trace_id))].copy().sort_values("step_idx")
    if len(rows) == 0:
        raise FileNotFoundError(f"missing maxgap trace rows for {trace_id}")
    lo, hi = _extract_zoom_window(np, rows, int(window))
    zoom = _normalized_plot_steps(pd, np, rows.iloc[int(lo) : int(hi)].copy(), keep_fraction=2.0 / 3.0)
    x = zoom["plot_step"].to_numpy(dtype=float)
    condition = zoom["park_server1_oracle9"].to_numpy(dtype=float) > 0.5
    job_size_kb = pd.to_numeric(zoom["job_size"], errors="coerce").to_numpy(dtype=float) / 1000.0

    fig, axes = plt.subplots(2, 1, figsize=(10.8, 6.8), sharex=True)
    for ax in axes:
        if np.any(condition):
            ax.fill_between(
                x,
                0,
                1,
                where=condition,
                transform=ax.get_xaxis_transform(),
                color=CONDITION_FILL_COLOR,
                alpha=0.7,
            )

    ax = axes[0]
    ax.plot(x, job_size_kb, color="k", linewidth=2.2)
    ax.set_ylabel("Job Size [KB]", fontsize=int(fontsize))
    ax.set_title("", fontsize=int(fontsize) + 1)
    _style_axes(ax, fontsize=int(fontsize), sns=sns, grid_axis="y")

    ax = axes[1]
    ax.step(x, zoom["victim_action"], where="post", color=PARK_COLOR, linewidth=2.0, label="Park")
    ax.step(
        x,
        zoom["lct_action"],
        where="post",
        color=BASELINE_COLOR,
        linewidth=1.8,
        linestyle="-",
        label="LCT",
    )
    ax.set_yticks(range(len(SERVICE_RATES)), [str(i) for i in range(len(SERVICE_RATES))])
    ax.set_ylabel("Server Selection", fontsize=int(fontsize))
    ax.set_xlabel("Decision Step", fontsize=int(fontsize) + 1)
    ax.set_title("", fontsize=int(fontsize) + 1)
    ax.set_xlim(0.0, float(x[-1]) if int(len(x)) > 0 else 1.0)
    _style_axes(ax, fontsize=int(fontsize), sns=sns, grid_axis="y")
    legend_handles = [
        plt.Line2D([], [], color=PARK_COLOR, linewidth=2.0, label="Park"),
        plt.Line2D([], [], color=BASELINE_COLOR, linewidth=1.8, linestyle="-", label="LCT"),
        Patch(facecolor=CONDITION_FILL_COLOR, edgecolor="gray", alpha=0.18, label="Large gap"),
    ]
    _style_legend(ax, fontsize=int(fontsize), loc="upper right", ncol=1, handles=legend_handles)

    fig.tight_layout()
    return _save_figure(fig, out_dir, f"maxgap_condition_slice_{Path(trace_id).stem}", formats, int(dpi))


def _plot_condition_regret(pd: Any, np: Any, plt: Any, steps: Any, out_dir: Path, formats: list[str], dpi: int) -> list[Path]:
    condition_specs = [
        ("All steps", None),
        ("Slow server", "park_slow_choice"),
        ("Slow + fast idle", "park_slow_choice_fast_idle"),
        ("Server 1 + server 9 idle", "park_server1_server9_idle"),
        ("Server 1 vs oracle 9", "park_server1_oracle9"),
        ("Matches oracle", "park_matches_oracle"),
    ]
    rows: list[dict[str, Any]] = []
    for trace_set, group in steps.groupby("trace_set"):
        for label, col in list(condition_specs):
            if col is None:
                sub = group
            else:
                sub = group[group[col] == 1]
            regrets = pd.to_numeric(sub["regret"], errors="coerce").dropna()
            if len(regrets) == 0:
                continue
            rows.append(
                {
                    "trace_set": str(trace_set),
                    "condition": str(label),
                    "count": int(len(regrets)),
                    "fraction": float(len(regrets) / max(1, len(group))),
                    "regret_median": float(regrets.median()),
                    "regret_mean": float(regrets.mean()),
                    "regret_p95": float(regrets.quantile(0.95)),
                }
            )
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.2), gridspec_kw={"width_ratios": [1.35, 1.0]})
    xlabels = [x[0] for x in condition_specs]
    x = np.arange(len(xlabels))
    width = 0.35
    for offset, trace_set, color in [(-0.18, "clean", CLEAN_COLOR), (0.18, "maxgap", PARK_COLOR)]:
        vals = []
        for label in xlabels:
            match = df[(df["trace_set"] == trace_set) & (df["condition"] == label)]
            vals.append(float(match["regret_median"].iloc[0]) if len(match) else np.nan)
        axes[0].bar(x + offset, vals, width=width, color=color, label=trace_set)
    axes[0].set_xticks(x, xlabels)
    axes[0].set_ylabel("Median K-step regret")
    axes[0].set_title("Regret concentrates under the server-1/fast-idle condition")
    axes[0].legend(frameon=False)
    for tick in axes[0].get_xticklabels():
        tick.set_rotation(25)
        tick.set_ha("right")
    _style_axes(axes[0])

    for offset, trace_set, color in [(-0.18, "clean", CLEAN_COLOR), (0.18, "maxgap", PARK_COLOR)]:
        vals = []
        for label in xlabels:
            match = df[(df["trace_set"] == trace_set) & (df["condition"] == label)]
            vals.append(float(match["fraction"].iloc[0]) if len(match) else np.nan)
        axes[1].bar(x + offset, vals, width=width, color=color, label=trace_set)
    axes[1].set_xticks(x, xlabels)
    axes[1].set_ylabel("Fraction of steps")
    axes[1].set_title("The condition is common in maxgap, rare/low-impact in clean")
    axes[1].legend(frameon=False)
    for tick in axes[1].get_xticklabels():
        tick.set_rotation(25)
        tick.set_ha("right")
    _style_axes(axes[1])

    fig.suptitle("Condition-triggered failure slice", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save_figure(fig, out_dir, "condition_triggered_regret", formats, int(dpi))


def _plot_server_load_heatmap(pd: Any, np: Any, plt: Any, steps: Any, trace_id: str, out_dir: Path, formats: list[str], dpi: int) -> list[Path]:
    rows = steps[(steps["trace_set"] == "maxgap") & (steps["trace_id"] == str(trace_id))].copy().sort_values("step_idx")
    if len(rows) == 0:
        return []

    def matrix(prefix: str) -> Any:
        cols = [f"{prefix}/server_load_proxy_{i}_before" for i in range(len(SERVICE_RATES))]
        vals = rows[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float).T
        return np.log10(vals + 1.0)

    victim_mat = matrix("victim")
    lct_mat = matrix(f"baseline/{BEST_BASELINE}")
    vmax = float(np.nanpercentile(np.concatenate([victim_mat.reshape(-1), lct_mat.reshape(-1)]), 99.0))
    fig, axes = plt.subplots(2, 1, figsize=(14.0, 6.4), sharex=True)
    for ax, mat, title in [
        (axes[0], victim_mat, "Park server-load heatmap"),
        (axes[1], lct_mat, "Least-completion-time baseline server-load heatmap"),
    ]:
        im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap="magma", vmin=0.0, vmax=vmax)
        ax.set_yticks(range(len(SERVICE_RATES)), [f"{i} ({SERVICE_RATES[i]:.2f})" for i in range(len(SERVICE_RATES))])
        ax.set_ylabel("Server")
        ax.set_title(title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[-1].set_xlabel("Step in trace")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.018, pad=0.015)
    cbar.set_label("log10(load proxy + 1)")
    fig.suptitle(f"Load concentration vs balanced fast-server use on {trace_id}", fontsize=13, fontweight="bold")
    fig.subplots_adjust(left=0.12, right=0.90, top=0.88, bottom=0.10, hspace=0.36)
    return _save_figure(fig, out_dir, f"maxgap_server_load_heatmap_{Path(trace_id).stem}", formats, int(dpi))


def _plot_episode_scatter(pd: Any, np: Any, plt: Any, trace_stats: Any, episodes: Any, steps: Any, out_dir: Path, formats: list[str], dpi: int) -> list[Path]:
    maxgap_trace = trace_stats[trace_stats["trace_set"] == "maxgap"].copy()
    maxgap_ep = episodes[episodes["trace_set"] == "maxgap"].copy()
    action1 = (
        steps[steps["trace_set"] == "maxgap"]
        .groupby("trace_id")["victim_action"]
        .apply(lambda s: float((s.astype(int) == 1).mean()))
        .reset_index(name="park_action1_fraction")
    )
    df = maxgap_trace.merge(maxgap_ep, on="trace_id", how="inner", suffixes=("", "_episode")).merge(action1, on="trace_id", how="left")
    if len(df) == 0:
        return []

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    sc = axes[0].scatter(
        df["mean_size"],
        df["oracle_gap_per_job"],
        c=df["park_action1_fraction"],
        s=55,
        cmap="Reds",
        edgecolor="#222222",
        linewidth=0.4,
    )
    axes[0].set_xlabel("Trace mean job size")
    axes[0].set_ylabel("Oracle gap per job")
    axes[0].set_title("Heavy traces amplify the server-1 bias")
    _style_axes(axes[0])
    cbar = fig.colorbar(sc, ax=axes[0], fraction=0.045, pad=0.02)
    cbar.set_label("Park action-1 fraction")

    axes[1].scatter(
        df["max_run_ge_5000"],
        df["oracle_gap_per_job"],
        c=df["park_action1_fraction"],
        s=55,
        cmap="Reds",
        edgecolor="#222222",
        linewidth=0.4,
    )
    axes[1].set_xlabel("Max consecutive 5000-size jobs")
    axes[1].set_ylabel("Oracle gap per job")
    axes[1].set_title("Temporal clustering drives worst episodes")
    _style_axes(axes[1])

    fig.suptitle("Episode-level causal signature under maxgap", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save_figure(fig, out_dir, "maxgap_episode_stress_scatter", formats, int(dpi))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Park load-balancing failure diagnostics from local W&B logs.")
    parser.add_argument("--clean-run", type=str, default=DEFAULT_CLEAN_RUN)
    parser.add_argument("--maxgap-run", type=str, default=DEFAULT_MAXGAP_RUN)
    parser.add_argument("--clean-trace-dir", type=str, default=DEFAULT_CLEAN_TRACE_DIR)
    parser.add_argument("--maxgap-trace-dir", type=str, default=DEFAULT_MAXGAP_TRACE_DIR)
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--trace-id", type=str, default=DEFAULT_TRACE_ID, help="Maxgap trace to visualize.")
    parser.add_argument("--num-servers", type=int, default=10)
    parser.add_argument("--zoom-window", type=int, default=160)
    parser.add_argument("--fontsize", type=int, default=DEFAULT_FONT_SIZE)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--format", action="append", default=["pdf", "png"], help="Output format. May be repeated.")
    parser.add_argument("--force-reprocess", action="store_true")
    return parser.parse_args()


def main() -> None:
    pd, np, plt, sns = _require_imports()
    args = parse_args()
    clean_run = _resolve_path(args.clean_run)
    maxgap_run = _resolve_path(args.maxgap_run)
    clean_trace_dir = _resolve_path(args.clean_trace_dir)
    maxgap_trace_dir = _resolve_path(args.maxgap_trace_dir)
    data_dir = _resolve_path(args.data_dir)
    out_dir = _resolve_path(args.out_dir)

    steps, _episodes, _trace_stats, cache_paths = _load_or_create_processed(
        pd,
        np,
        clean_run=clean_run,
        maxgap_run=maxgap_run,
        clean_trace_dir=clean_trace_dir,
        maxgap_trace_dir=maxgap_trace_dir,
        data_dir=data_dir,
        force=bool(args.force_reprocess),
        num_servers=int(args.num_servers),
    )
    trace_id = str(args.trace_id).strip()
    if trace_id and not trace_id.endswith(".csv"):
        trace_id = f"{trace_id}.csv"
    formats = sorted(set(str(x).lower() for x in list(args.format)))

    written: list[Path] = []
    written.extend(
        _plot_condition_slice(
            pd,
            np,
            plt,
            sns,
            steps,
            trace_id,
            out_dir,
            formats,
            int(args.dpi),
            int(args.zoom_window),
            int(args.fontsize),
        )
    )
    written.extend(_plot_action_shift(pd, np, plt, sns, steps, out_dir, formats, int(args.dpi), int(args.fontsize)))
    description_path = _write_figure_descriptions(
        out_dir,
        [
            _build_condition_slice_description(pd, np, steps, trace_id, int(args.zoom_window)),
            _build_action_shift_description(pd, np, steps),
        ],
    )
    plt.close("all")

    print(f"Processed data written under: {data_dir}")
    for name, path in sorted(cache_paths.items()):
        print(f"  {name}: {path}")
    print(f"Plots written under: {out_dir}")
    for path in list(written):
        print(f"  {path}")
    print(f"Figure descriptions: {description_path}")
    print(f"Failure trace: {trace_id}")


if __name__ == "__main__":
    main()
