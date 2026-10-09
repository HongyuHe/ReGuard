#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import time
from typing import Any

"""
Example usage:
./venv/bin/python rules/mine/gen_window_stats_thresholds.py \
    --adv-csv rules/mine/data/pgd-feat.csv \
    --clean-csv rules/mine/data/clean.csv
"""


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _resolve_path(path: str) -> str:
    expanded = os.path.expandvars(os.path.expanduser(str(path)))
    if os.path.isabs(expanded):
        return expanded
    if os.path.exists(expanded):
        return os.path.abspath(expanded)

    root = _repo_root()
    under_root = os.path.join(root, expanded)
    if os.path.exists(under_root):
        return os.path.abspath(under_root)

    under_src = os.path.join(root, "Pensieve-PPO", "src", expanded)
    if os.path.exists(under_src):
        return os.path.abspath(under_src)

    return os.path.abspath(under_root)


def _default_out_adv_stats_csv(adv_csv: str) -> str:
    attack = os.path.splitext(os.path.basename(str(adv_csv)))[0]
    out_dir = os.path.dirname(str(adv_csv)) or "."
    return os.path.join(out_dir, f"{attack}_states_stats.csv")


def _default_out_thresholds_csv(clean_csv: str) -> str:
    out_dir = os.path.dirname(str(clean_csv)) or "."
    return os.path.join(out_dir, "clean_stats_thresholds.csv")


def _rename_legacy_columns(df: Any) -> Any:
    #* Keep backward compatibility with earlier dataset schemas.
    cols = set(str(c) for c in getattr(df, "columns", []))

    renames: dict[str, str] = {}
    for i in range(256):
        old = f"buffer_hist_s_{i}"
        new = f"buffer_hist_sec_{i}"
        if old in cols and new not in cols:
            renames[old] = new

    if renames:
        df = df.rename(columns=renames)

    cols = set(str(c) for c in getattr(df, "columns", []))
    if "video_chunk_remain" not in cols:
        if "video_chunk_remain_7" in cols:
            df["video_chunk_remain"] = df["video_chunk_remain_7"]

    return df


def _find_indexed_columns(df: Any, *, prefix: str) -> list[tuple[int, str]]:
    cols = []
    for c in list(getattr(df, "columns", [])):
        name = str(c)
        if not name.startswith(prefix + "_"):
            continue
        suffix = name[len(prefix) + 1 :]
        if not suffix.isdigit():
            continue
        cols.append((int(suffix), str(name)))
    return sorted(cols, key=lambda t: t[0])


def _require_columns(df: Any, required: list[str]) -> None:
    cols = set(str(c) for c in getattr(df, "columns", []))
    missing = [c for c in required if str(c) not in cols]
    if missing:
        raise RuntimeError(f"* Missing required columns: {missing}")


def _window_stats(
    df: Any,
    *,
    prefix: str,
    eps: float = 1e-12,
) -> Any:
    indexed = _find_indexed_columns(df, prefix=prefix)
    if not indexed:
        raise RuntimeError(f"* No indexed columns found for prefix: {prefix}")

    idxs = [i for i, _c in indexed]
    if idxs != list(range(int(max(idxs)) + 1)):
        raise RuntimeError(f"* Non-contiguous indices for {prefix}: {idxs}")

    cols = [c for _i, c in indexed]
    import numpy as np
    import pandas as pd

    x = df[cols].astype(float).to_numpy(dtype=np.float64, copy=False)
    l = int(x.shape[1])
    if l <= 0:
        raise RuntimeError(f"* Unexpected history length for {prefix}: {x.shape}")

    x_max = np.max(x, axis=1)
    x_min = np.min(x, axis=1)
    x_mean = np.mean(x, axis=1)
    x_std = np.std(x, axis=1)

    t = np.arange(l, dtype=np.float64)
    t_center = t - float(np.mean(t))
    denom = float(np.sum(t_center * t_center))
    slope = (x @ t_center) / float(denom) if float(denom) > 0.0 else np.full((x.shape[0],), np.nan)

    x_range = x_max - x_min
    x_cv = x_std / (np.abs(x_mean) + float(eps))
    if l >= 2:
        x_delta_last = x[:, -1] - x[:, -2]
    else:
        x_delta_last = np.full((x.shape[0],), np.nan)
    x_delta_first_last = x[:, -1] - x[:, 0]

    half = int(l // 2)
    if half >= 1:
        x_mean_first = np.mean(x[:, :half], axis=1)
        x_mean_second = np.mean(x[:, half:], axis=1)
    else:
        x_mean_first = np.full((x.shape[0],), np.nan)
        x_mean_second = np.full((x.shape[0],), np.nan)
    x_mean_second_minus_first = x_mean_second - x_mean_first

    return pd.DataFrame(
        {
            f"{prefix}_max": x_max,
            f"{prefix}_min": x_min,
            f"{prefix}_avg": x_mean,
            f"{prefix}_slope": slope,
            f"{prefix}_range": x_range,
            f"{prefix}_cv": x_cv,
            f"{prefix}_delta_last": x_delta_last,
            f"{prefix}_delta_first_last": x_delta_first_last,
            f"{prefix}_avg_firsthalf": x_mean_first,
            f"{prefix}_avg_secondhalf": x_mean_second,
            f"{prefix}_avg_secondhalf_minus_firsthalf": x_mean_second_minus_first,
        }
    )


def _next_chunk_stats(df: Any) -> Any:
    import numpy as np
    import pandas as pd

    indexed = _find_indexed_columns(df, prefix="next_chunk_size_bytes")
    if not indexed:
        #* Some CSVs may not contain chunk-size vectors; skip.
        return pd.DataFrame(index=df.index)

    idxs = [i for i, _c in indexed]
    if idxs != list(range(int(max(idxs)) + 1)):
        raise RuntimeError(f"* Non-contiguous indices for next_chunk_size_bytes: {idxs}")

    cols = [c for _i, c in indexed]
    x = df[cols].astype(float).to_numpy(dtype=np.float64, copy=False)
    x_max = np.max(x, axis=1)
    x_min = np.min(x, axis=1)
    x_mean = np.mean(x, axis=1)
    x_range = x_max - x_min

    out: dict[str, Any] = {
        "next_chunk_size_bytes_max": x_max,
        "next_chunk_size_bytes_min": x_min,
        "next_chunk_size_bytes_avg": x_mean,
        "next_chunk_size_bytes_range": x_range,
    }

    cols_set = set(str(c) for c in df.columns)
    if "policy_action_idx" in cols_set:
        idx = df["policy_action_idx"].astype(int).to_numpy(dtype=np.int64, copy=False)
        idx = np.clip(idx, 0, x.shape[1] - 1)
        out["next_chunk_size_bytes_policy"] = x[np.arange(x.shape[0]), idx]

    return pd.DataFrame(out)


def _percentiles(values: Any, qs: list[float]) -> dict[str, float]:
    import numpy as np

    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if int(arr.size) == 0:
        return {f"p{int(q)}": float("nan") for q in qs}

    out: dict[str, float] = {}
    for q in qs:
        out[f"p{int(q)}"] = float(np.percentile(arr, float(q)))
    return out


def _pooled_history_values(df: Any, *, prefix: str) -> Any:
    indexed = _find_indexed_columns(df, prefix=prefix)
    if not indexed:
        raise RuntimeError(f"* No indexed columns found for prefix: {prefix}")

    idxs = [i for i, _c in indexed]
    if idxs != list(range(int(max(idxs)) + 1)):
        raise RuntimeError(f"* Non-contiguous indices for {prefix}: {idxs}")

    cols = [c for _i, c in indexed]
    import numpy as np

    x = df[cols].astype(float).to_numpy(dtype=np.float64, copy=False)
    return np.asarray(x, dtype=np.float64).reshape(-1)


def _spike_counts(df: Any, *, prefix: str, tau: float, op: str) -> Any:
    indexed = _find_indexed_columns(df, prefix=prefix)
    if not indexed:
        raise RuntimeError(f"* No indexed columns found for prefix: {prefix}")

    idxs = [i for i, _c in indexed]
    if idxs != list(range(int(max(idxs)) + 1)):
        raise RuntimeError(f"* Non-contiguous indices for {prefix}: {idxs}")

    cols = [c for _i, c in indexed]
    import numpy as np
    import pandas as pd

    x = df[cols].astype(float).to_numpy(dtype=np.float64, copy=False)
    if str(op) == "gt":
        mask = x > float(tau)
    elif str(op) == "lt":
        mask = x < float(tau)
    else:
        raise ValueError(f"* Unknown op for spike count: {op}")
    return pd.Series(np.sum(mask, axis=1).astype(int), index=df.index)


def _max_run_length(df: Any, *, prefix: str, tau: float, op: str) -> Any:
    indexed = _find_indexed_columns(df, prefix=prefix)
    if not indexed:
        raise RuntimeError(f"* No indexed columns found for prefix: {prefix}")

    idxs = [i for i, _c in indexed]
    if idxs != list(range(int(max(idxs)) + 1)):
        raise RuntimeError(f"* Non-contiguous indices for {prefix}: {idxs}")

    cols = [c for _i, c in indexed]
    import numpy as np
    import pandas as pd

    x = df[cols].astype(float).to_numpy(dtype=np.float64, copy=False)
    if str(op) == "gt":
        mask = x > float(tau)
    elif str(op) == "lt":
        mask = x < float(tau)
    else:
        raise ValueError(f"* Unknown op for max-run length: {op}")

    n, l = mask.shape
    run = np.zeros((n,), dtype=np.int64)
    best = np.zeros((n,), dtype=np.int64)
    for j in range(int(l)):
        m = mask[:, j].astype(np.int64)
        run = (run + 1) * m
        best = np.maximum(best, run)
    return pd.Series(best.astype(int), index=df.index)


def _fixed_bitrate_thresholds_kbps() -> dict[str, float]:
    #* Use the fixed discrete bitrate levels as threshold candidates.
    levels = [300, 750, 1200, 1850, 2850, 4300]
    return {f"p{int(v)}": float(v) for v in levels}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute window-based statistics on logged ABR states and derive reference thresholds from clean traces."
    )
    parser.add_argument("--adv-csv", required=True, help="Adversarial logged states CSV (from gen_rule_mining_dataset.py).")
    parser.add_argument("--clean-csv", required=True, help="Clean logged states CSV (from gen_rule_mining_dataset.py).")
    parser.add_argument(
        "--out-adv-stats-csv",
        default=None,
        help="Output CSV path for adversarial state window statistics (default: '<attack>_states_stats.csv' next to --adv-csv).",
    )
    parser.add_argument(
        "--out-thresholds-csv",
        default=None,
        help="Output CSV path for clean-derived statistic thresholds (default: 'clean_stats_thresholds.csv' next to --clean-csv).",
    )
    parser.add_argument(
        "--percentiles",
        type=str,
        default="5,10,50,90,95",
        help="Comma-separated percentiles to export for clean-derived thresholds (e.g., '90,95').",
    )
    parser.add_argument(
        "--no-event-features",
        dest="event_features",
        action="store_false",
        help="Disable derived count/run event features (e.g., spike counts derived from pooled raw thresholds).",
    )
    parser.set_defaults(event_features=True)
    parser.add_argument(
        "--event-prefixes",
        type=str,
        default=None,
        help="Comma-separated prefixes to derive event features for (default: same as --stats prefixes).",
    )
    parser.add_argument(
        "--event-high-percentiles",
        type=str,
        default=None,
        help="Comma-separated q for spike events: count/run where x_i > p{q}(pooled raw). Default: all q>=90 from --percentiles.",
    )
    parser.add_argument(
        "--event-low-percentiles",
        type=str,
        default=None,
        help="Comma-separated q for dip events: count/run where x_i < p{q}(pooled raw). Default: none.",
    )
    args = parser.parse_args()

    adv_csv = _resolve_path(str(args.adv_csv))
    clean_csv = _resolve_path(str(args.clean_csv))
    out_adv_stats_csv = (
        _resolve_path(str(args.out_adv_stats_csv)) if args.out_adv_stats_csv else _default_out_adv_stats_csv(adv_csv)
    )
    out_thresholds_csv = (
        _resolve_path(str(args.out_thresholds_csv)) if args.out_thresholds_csv else _default_out_thresholds_csv(clean_csv)
    )

    if not os.path.isfile(adv_csv):
        raise FileNotFoundError(f"* --adv-csv not found: {adv_csv}")
    if not os.path.isfile(clean_csv):
        raise FileNotFoundError(f"* --clean-csv not found: {clean_csv}")

    try:
        import pandas as pd
    except Exception as e:  # pragma: no cover
        raise RuntimeError("* Missing dependencies (pandas). Use `venv/bin/python`.") from e

    adv_df = _rename_legacy_columns(pd.read_csv(adv_csv))
    clean_df = _rename_legacy_columns(pd.read_csv(clean_csv))

    _require_columns(adv_df, ["oracle_gap_pct"])
    _require_columns(clean_df, ["oracle_gap_pct"])

    prefixes = ["delay_hist_ms", "throughput_hist_mbps", "buffer_hist_sec", "bitrate_hist_kbps"]
    adv_stats_parts = []
    clean_stats_parts = []
    for p in prefixes:
        adv_stats_parts.append(_window_stats(adv_df, prefix=str(p)))
        clean_stats_parts.append(_window_stats(clean_df, prefix=str(p)))

    adv_stats_parts.append(_next_chunk_stats(adv_df))
    clean_stats_parts.append(_next_chunk_stats(clean_df))

    adv_stats = pd.concat(adv_stats_parts, axis=1)
    clean_stats = pd.concat(clean_stats_parts, axis=1)

    id_cols = [
        "trace_name",
        "trace_idx",
        "chunk_idx",
        "video_chunk_remain",
        "chunk_size_bytes",
        "last_action_kbps",
        "policy_action_kbps",
        "oracle_action_kbps",
        "oracle_best_reward_mean",
        "policy_reward_mean",
        "oracle_gap_mean",
        "oracle_gap_pct",
    ]
    adv_cols = set(str(c) for c in adv_df.columns)
    id_cols_present = [c for c in id_cols if c in adv_cols]

    adv_out = adv_df[id_cols_present].copy()
    adv_out = pd.concat([adv_out, adv_stats], axis=1)

    if "policy_action_kbps" in adv_out.columns and "oracle_action_kbps" in adv_out.columns:
        adv_out["policy_minus_oracle_kbps"] = adv_out["policy_action_kbps"].astype(float) - adv_out["oracle_action_kbps"].astype(float)
        adv_out["policy_vs_oracle"] = adv_out["policy_minus_oracle_kbps"].apply(
            lambda v: "aggressive" if float(v) > 0.0 else ("conservative" if float(v) < 0.0 else "equal")
        )

    #* Cross-signal features (policy/oracle action compared against history actions/bitrates).
    #* Commented out: these create "self-referential" predicates like policy_action_kbps > bitrate_hist_kbps_max,
    #* which can dominate and reduce interpretability of network-dynamics-based rules.
    #* If you want to re-enable them later, uncomment the block below (and the corresponding threshold block).
    #* if {"policy_action_kbps", "bitrate_hist_kbps_max"}.issubset(set(adv_out.columns)):
    #*     adv_out["policy_minus_bitrate_hist_kbps_max"] = adv_out["policy_action_kbps"].astype(float) - adv_out["bitrate_hist_kbps_max"].astype(float)
    #* if {"oracle_action_kbps", "bitrate_hist_kbps_max"}.issubset(set(adv_out.columns)):
    #*     adv_out["oracle_minus_bitrate_hist_kbps_max"] = adv_out["oracle_action_kbps"].astype(float) - adv_out["bitrate_hist_kbps_max"].astype(float)
    #* if {"policy_action_kbps", "last_action_kbps"}.issubset(set(adv_out.columns)):
    #*     adv_out["policy_minus_last_action_kbps"] = adv_out["policy_action_kbps"].astype(float) - adv_out["last_action_kbps"].astype(float)

    percentiles_raw = [s.strip() for s in str(args.percentiles).split(",") if s.strip()]
    qs: list[float] = []
    for s in percentiles_raw:
        try:
            qs.append(float(s))
        except Exception as e:
            raise ValueError(f"* Invalid percentile value: {s}") from e
    if not qs:
        raise ValueError("* --percentiles must contain at least one value")

    thresholds: dict[str, dict[str, float]] = {}
    #* Threshold policy:
    #* - Raw-scale stats (same physical units as raw history values): pool thresholds from raw values.
    #*   Example: delay_hist_ms_{max,min,avg,avg_firsthalf,avg_secondhalf} share thresholds
    #*   computed over pooled delay_hist_ms_0..delay_hist_ms_{L-1}.
    #* - Derived-scale stats (different scale or semantics): compute thresholds per feature.
    #*   Includes cv, range, delta_*, slope, and secondhalf_minus_firsthalf.
    raw_scale_suffixes = {"max", "min", "avg", "avg_firsthalf", "avg_secondhalf"}
    derived_scale_suffixes = {
        "cv",
        "range",
        "delta_last",
        "delta_first_last",
        "slope",
        "avg_secondhalf_minus_firsthalf",
    }

    pooled_values_by_prefix: dict[str, Any] = {}
    pooled_raw_percentiles: dict[str, dict[str, float]] = {}
    for p in prefixes:
        pooled_values_by_prefix[str(p)] = _pooled_history_values(clean_df, prefix=str(p))
        pooled_raw_percentiles[str(p)] = _percentiles(pooled_values_by_prefix[str(p)], qs)

    #* Raw-scale thresholds for each prefix:
    #* - delay/throughput/buffer: pooled raw percentiles
    #* - bitrate: fixed discrete action levels
    pooled_raw_thresholds: dict[str, dict[str, float]] = dict(pooled_raw_percentiles)
    pooled_raw_thresholds["bitrate_hist_kbps"] = _fixed_bitrate_thresholds_kbps()
    for p in prefixes:
        thresholds[f"{p}_value"] = dict(pooled_raw_thresholds[str(p)])

    #* Next-chunk size thresholds: pool over the raw next_chunk_size_bytes_0..A-1 vector.
    next_chunk_pooled_raw = None
    try:
        pooled_next_chunk = _pooled_history_values(clean_df, prefix="next_chunk_size_bytes")
        next_chunk_pooled_raw = _percentiles(pooled_next_chunk, qs)
        thresholds["next_chunk_size_bytes_value"] = dict(next_chunk_pooled_raw)
    except Exception:
        next_chunk_pooled_raw = None

    for col in list(clean_stats.columns):
        name = str(col)
        matched_prefix = None
        matched_suffix = None
        for p in prefixes:
            prefix = str(p)
            if not name.startswith(prefix + "_"):
                continue
            matched_prefix = prefix
            matched_suffix = name[len(prefix) + 1 :]
            break

        if matched_prefix is not None and matched_suffix is not None:
            if matched_suffix in raw_scale_suffixes:
                thresholds[name] = dict(pooled_raw_thresholds[str(matched_prefix)])
                continue
            if matched_suffix in derived_scale_suffixes:
                thresholds[name] = _percentiles(clean_stats[str(col)].to_numpy(), qs)
                continue
        if next_chunk_pooled_raw is not None and name.startswith("next_chunk_size_bytes_"):
            suffix = name[len("next_chunk_size_bytes_") :]
            if suffix in {"min", "max", "avg"}:
                thresholds[name] = dict(next_chunk_pooled_raw)
                continue

        thresholds[name] = _percentiles(clean_stats[str(col)].to_numpy(), qs)

    #* Discrete action-valued features: use the fixed bitrate levels as threshold candidates.
    #* This enables predicates like last_action_kbps > bitrate_300_kbps and (with --include-eq) last_action_kbps = bitrate_4300_kbps.
    discrete_action_feats = ["last_action_kbps", "policy_action_kbps", "oracle_action_kbps"]
    for feat in discrete_action_feats:
        if str(feat) in set(clean_df.columns):
            thresholds[str(feat)] = dict(_fixed_bitrate_thresholds_kbps())

    #* Cross-signal thresholds are commented out along with the cross-signal features above.
    #* cross_features: dict[str, Any] = {}
    #* if "policy_action_kbps" in set(clean_df.columns) and "bitrate_hist_kbps_max" in set(clean_stats.columns):
    #*     cross_features["policy_minus_bitrate_hist_kbps_max"] = (
    #*         clean_df["policy_action_kbps"].astype(float).to_numpy()
    #*         - clean_stats["bitrate_hist_kbps_max"].astype(float).to_numpy()
    #*     )
    #* if "oracle_action_kbps" in set(clean_df.columns) and "bitrate_hist_kbps_max" in set(clean_stats.columns):
    #*     cross_features["oracle_minus_bitrate_hist_kbps_max"] = (
    #*         clean_df["oracle_action_kbps"].astype(float).to_numpy()
    #*         - clean_stats["bitrate_hist_kbps_max"].astype(float).to_numpy()
    #*     )
    #* if {"policy_action_kbps", "last_action_kbps"}.issubset(set(clean_df.columns)):
    #*     cross_features["policy_minus_last_action_kbps"] = (
    #*         clean_df["policy_action_kbps"].astype(float).to_numpy()
    #*         - clean_df["last_action_kbps"].astype(float).to_numpy()
    #*     )
    #*
    #* for feat, values in cross_features.items():
    #*     qmap = _percentiles(values, qs)
    #*     qmap["p0"] = 0.0
    #*     thresholds[str(feat)] = qmap

    if bool(args.event_features):
        #* Event features are derived from pooled raw thresholds.
        event_prefixes = list(prefixes)
        if args.event_prefixes:
            event_prefixes = [s.strip() for s in str(args.event_prefixes).split(",") if s.strip()]

        def _parse_qs(s: str) -> list[float]:
            out: list[float] = []
            for part in [p.strip() for p in str(s).split(",") if p.strip()]:
                out.append(float(part))
            return out

        high_qs: list[float] = []
        low_qs: list[float] = []

        if args.event_high_percentiles:
            high_qs = _parse_qs(str(args.event_high_percentiles))
        else:
            high_qs_raw = [float(q) for q in qs if float(q) >= 90.0]
            high_qs = sorted(set(high_qs_raw)) if high_qs_raw else [max(float(q) for q in qs)]

        if args.event_low_percentiles:
            low_qs = _parse_qs(str(args.event_low_percentiles))

        for p in event_prefixes:
            if str(p) not in pooled_values_by_prefix:
                raise RuntimeError(f"* Missing pooled raw values for prefix={p}.")

            #* Rename event features so the percentile that defines the event threshold is embedded
            #* before the statistic name (e.g., throughput_hist_mbps_p95spike_count), avoiding
            #* confusing double-percentile suffixes like throughput_hist_mbps_spike_count_p95_p90.
            def _event_name(*, prefix: str, kind: str, q: float) -> str:
                qtag = f"p{int(q)}"
                if str(kind) == "spike_count":
                    return f"{prefix}_{qtag}spike_count"
                if str(kind) == "spike_run_maxlen":
                    return f"{prefix}_{qtag}spike_run_maxlen"
                if str(kind) == "dip_count":
                    return f"{prefix}_{qtag}dip_count"
                if str(kind) == "dip_run_maxlen":
                    return f"{prefix}_{qtag}dip_run_maxlen"
                raise ValueError(f"* Unknown event kind: {kind}")

            for q in high_qs:
                tau = float(pooled_raw_percentiles.get(str(p), {}).get(f"p{int(q)}", float("nan")))
                if not math.isfinite(float(tau)):
                    import numpy as np

                    tau = float(np.percentile(pooled_values_by_prefix[str(p)], float(q)))
                if not math.isfinite(float(tau)):
                    continue
                count_name = _event_name(prefix=str(p), kind="spike_count", q=float(q))
                run_name = _event_name(prefix=str(p), kind="spike_run_maxlen", q=float(q))
                adv_out[count_name] = _spike_counts(adv_df, prefix=str(p), tau=float(tau), op="gt")
                adv_out[run_name] = _max_run_length(adv_df, prefix=str(p), tau=float(tau), op="gt")
                clean_counts = _spike_counts(clean_df, prefix=str(p), tau=float(tau), op="gt")
                clean_runs = _max_run_length(clean_df, prefix=str(p), tau=float(tau), op="gt")
                thresholds[count_name] = _percentiles(clean_counts.to_numpy(), qs)
                thresholds[run_name] = _percentiles(clean_runs.to_numpy(), qs)

            for q in low_qs:
                tau = float(pooled_raw_percentiles.get(str(p), {}).get(f"p{int(q)}", float("nan")))
                if not math.isfinite(float(tau)):
                    import numpy as np

                    tau = float(np.percentile(pooled_values_by_prefix[str(p)], float(q)))
                if not math.isfinite(float(tau)):
                    continue
                count_name = _event_name(prefix=str(p), kind="dip_count", q=float(q))
                run_name = _event_name(prefix=str(p), kind="dip_run_maxlen", q=float(q))
                adv_out[count_name] = _spike_counts(adv_df, prefix=str(p), tau=float(tau), op="lt")
                adv_out[run_name] = _max_run_length(adv_df, prefix=str(p), tau=float(tau), op="lt")
                clean_counts = _spike_counts(clean_df, prefix=str(p), tau=float(tau), op="lt")
                clean_runs = _max_run_length(clean_df, prefix=str(p), tau=float(tau), op="lt")
                thresholds[count_name] = _percentiles(clean_counts.to_numpy(), qs)
                thresholds[run_name] = _percentiles(clean_runs.to_numpy(), qs)

    meta = {
        "created_at_unix_s": float(time.time()),
        "adv_csv": str(adv_csv),
        "clean_csv": str(clean_csv),
        "percentiles": [float(q) for q in qs],
        "stats_prefixes": list(prefixes),
        "event_features": bool(args.event_features),
    }

    os.makedirs(os.path.dirname(out_adv_stats_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_thresholds_csv) or ".", exist_ok=True)

    adv_out.to_csv(out_adv_stats_csv, index=False)

    rows: list[dict[str, Any]] = []
    for feat, qmap in thresholds.items():
        row: dict[str, Any] = {"feature": str(feat)}
        for k, v in qmap.items():
            row[str(k)] = float(v)
        rows.append(row)

    thresh_df = pd.DataFrame(rows)
    thresh_df["source_clean_csv"] = str(clean_csv)
    thresh_df["created_at_unix_s"] = float(meta["created_at_unix_s"])
    thresh_df["percentiles"] = ",".join(str(int(x)) if float(x).is_integer() else str(x) for x in meta["percentiles"])
    thresh_df.to_csv(out_thresholds_csv, index=False)

    print(f"* Wrote adversarial stats CSV: {out_adv_stats_csv} (rows={len(adv_out)})")
    print(f"* Wrote clean-derived thresholds CSV: {out_thresholds_csv} (features={len(thresholds)})")


if __name__ == "__main__":
    main()
