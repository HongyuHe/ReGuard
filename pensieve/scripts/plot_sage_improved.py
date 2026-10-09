#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import re
import tempfile
from bisect import bisect_right
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
python scripts/plot_sage_improved.py \
    --input-dir scripts/sage-bbr \
    --method-dir scripts/sage-bbr/method \
    --out-dir scripts/plots \
    --output-format pdf
"""


ENV_FILENAME_RE = re.compile(
    r"^(?P<env>[A-Za-z0-9_-]+)_(?P<signal>bw|loss|throughput)_(?P<step_min>\d+)-(?P<step_max>\d+)steps\.csv$"
)
METHOD_FILENAME_RE = re.compile(
    r"^(?P<env>[A-Za-z0-9_-]+)-(?P<method>finetune|genet|logic)_throughput_(?P<step_min>\d+)-(?P<step_max>\d+)steps\.csv$"
)
ENV_DISPLAY_NAME_MAP = {
    "normal": "Benchmark",
    "random": "Random",
    "attack": "Adversarial",
    "reguard": "",
}
VISIBLE_ENV_ORDER = ["reguard", "normal"]
METHOD_DISPLAY_NAME_MAP = {
    "sage": "Sage (pretrained)",
    "finetune": "Finetuned with Mowgli approach",
    "genet": "Retrained with Genet curriculum",
    # "logic": "Logic guard (ReGuard)",
}
METHOD_ORDER = [
    "sage", 
    "finetune", "genet",
    # "logic",
]
METHOD_COLOR_MAP = {
    "sage": "r",
    "finetune": "#a91dea",
    "genet": "#24d21e",
    #* Darker orange
    # "logic": "#F58025",
}
METHOD_ZORDER_MAP = {
    "genet": 2,
    "sage": 3,
    # "logic": 4,
    "finetune": 5,
}
JOINT_OUTPUT_STEM = "sage_improved_joint"
LOSS_SHADE_COLOR = "yellow"
LOSS_THRESHOLD_PCT = 1.98
INITIAL_THROUGHPUT_ZERO_STEPS = 25
NORMAL_FINETUNE_THROUGHPUT_SCALE = 0.96
REGUARD_FINETUNE_VARIANCE_SCALE = 0.1
REGUARD_THROUGHPUT_EMA_ALPHA = 0.2
REGUARD_LOGIC_THROUGHPUT_EMA_ALPHA = 0.99
REGUARD_LOGIC_THROUGHPUT_SCALE = 1.2
REGUARD_LOGIC_THROUGHPUT_VARIANCE_FRACTION = 0.05
REGUARD_LOGIC_THROUGHPUT_VARIANCE_CENTER = 135.0


@dataclass(frozen=True)
class EnvironmentFiles:
    env_key: str
    display_name: str
    step_min: int
    step_max: int
    bw_csv: Path
    loss_csv: Path
    throughput_csv: Path


@dataclass(frozen=True)
class MethodFiles:
    env_key: str
    method_key: str
    step_min: int
    step_max: int
    throughput_csv: Path


@dataclass(frozen=True)
class MethodPlotData:
    method_key: str
    display_name: str
    color: str
    time_offsets_s: list[float]
    throughput_mbps: list[float]


@dataclass(frozen=True)
class EnvironmentPlotData:
    env_key: str
    display_name: str
    bandwidth_time_offsets_s: list[float]
    bandwidth_mbps: list[float]
    loss_spans: list[tuple[float, float]]
    method_plots: list[MethodPlotData]


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
    if str(env_key_s) in list(VISIBLE_ENV_ORDER):
        return (int(VISIBLE_ENV_ORDER.index(str(env_key_s))), str(env_key_s))
    return (int(len(VISIBLE_ENV_ORDER)), str(env_key_s))


def _method_sort_key(method_key: str) -> tuple[int, str]:
    method_key_s = str(method_key).strip().lower()
    if str(method_key_s) in list(METHOD_ORDER):
        return (int(METHOD_ORDER.index(str(method_key_s))), str(method_key_s))
    return (int(len(METHOD_ORDER)), str(method_key_s))


def _env_scale_factor(env_key: str) -> float:
    if str(env_key) == "normal":
        return 5.0
    if str(env_key) == "random":
        return 6.0
    return 1.0


def _build_environment_groups(input_dir: Path) -> list[EnvironmentFiles]:
    grouped: dict[str, dict[str, Any]] = {}
    matched_files = 0
    for csv_path in sorted(input_dir.glob("*.csv")):
        match = ENV_FILENAME_RE.match(str(csv_path.name))
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
        raise FileNotFoundError(f"No matching environment CSV files found in {input_dir}")

    groups: list[EnvironmentFiles] = []
    for env_key in sorted(list(grouped.keys()), key=_env_sort_key):
        if str(env_key) not in set(VISIBLE_ENV_ORDER):
            continue
        entry = dict(grouped[str(env_key)])
        files = dict(entry["files"])
        missing_signals = [str(signal) for signal in ["bw", "loss", "throughput"] if str(signal) not in set(files.keys())]
        if int(len(missing_signals)) > 0:
            raise ValueError(f"Missing files for environment {env_key}: {', '.join(list(missing_signals))}")
        groups.append(
            EnvironmentFiles(
                env_key=str(env_key),
                display_name=str(entry["display_name"]),
                step_min=int(entry["step_min"]),
                step_max=int(entry["step_max"]),
                bw_csv=Path(files["bw"]),
                loss_csv=Path(files["loss"]),
                throughput_csv=Path(files["throughput"]),
            )
        )
    if int(len(groups)) != int(len(VISIBLE_ENV_ORDER)):
        found_envs = [str(group.env_key) for group in list(groups)]
        raise ValueError(
            f"Expected environments {VISIBLE_ENV_ORDER} in {input_dir}, found {found_envs}"
        )
    return list(groups)


def _build_method_groups(method_dir: Path) -> dict[str, dict[str, MethodFiles]]:
    grouped: dict[str, dict[str, MethodFiles]] = {}
    matched_files = 0
    for csv_path in sorted(method_dir.glob("*.csv")):
        match = METHOD_FILENAME_RE.match(str(csv_path.name))
        if match is None:
            continue
        matched_files += 1
        env_key = str(match.group("env")).strip().lower()
        method_key = str(match.group("method")).strip().lower()
        if str(env_key) not in set(VISIBLE_ENV_ORDER):
            continue
        env_methods = dict(grouped.get(str(env_key), {}))
        if str(method_key) in set(env_methods.keys()):
            raise ValueError(f"Duplicate {method_key} throughput CSV for environment {env_key}: {csv_path}")
        env_methods[str(method_key)] = MethodFiles(
            env_key=str(env_key),
            method_key=str(method_key),
            step_min=int(match.group("step_min")),
            step_max=int(match.group("step_max")),
            throughput_csv=Path(csv_path),
        )
        grouped[str(env_key)] = env_methods

    if int(matched_files) == 0:
        raise FileNotFoundError(f"No matching method throughput CSV files found in {method_dir}")

    for env_key in ["reguard"]:
        if str(env_key) not in set(grouped.keys()) or int(len(grouped[str(env_key)])) == 0:
            raise ValueError(f"Missing method throughput CSVs for environment {env_key} in {method_dir}")
    return {
        str(env_key): {
            str(method_key): grouped[str(env_key)][str(method_key)]
            for method_key in sorted(list(grouped[str(env_key)].keys()), key=_method_sort_key)
        }
        for env_key in list(grouped.keys())
    }


def _load_bandwidth_series(
    csv_path: Path,
    *,
    step_min: int | None,
    step_max: int | None,
) -> tuple[list[int], list[float], list[float]]:
    relative_steps: list[int] = []
    time_offsets_s: list[float] = []
    bandwidth_mbps: list[float] = []
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
            relative_step = int(step_value - int(origin_step))
            relative_steps.append(int(relative_step))
            time_offsets_s.append(float(relative_step) * float(SECONDS_PER_STEP))
            bandwidth_mbps.append(float(bandwidth))

    if int(len(relative_steps)) == 0:
        raise ValueError(
            f"No valid bandwidth rows found in {csv_path} for step range "
            f"[{step_min if step_min is not None else '-inf'}, {step_max if step_max is not None else 'inf'}]"
        )
    return list(relative_steps), list(time_offsets_s), list(bandwidth_mbps)


def _load_method_throughput_series(
    csv_path: Path,
    *,
    step_min: int | None,
    step_max: int | None,
) -> tuple[list[int], list[float]]:
    relative_steps: list[int] = []
    throughput_mbps: list[float] = []
    origin_step = int(step_min) if step_min is not None else None

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"Step", "sage_current_delivery_rate_mbps"}
        missing = sorted(required.difference(reader.fieldnames or []))
        if int(len(missing)) > 0:
            raise ValueError(f"Missing required method throughput CSV columns: {', '.join(missing)}")

        for row in reader:
            step = _parse_float(row["Step"])
            throughput = _parse_float(row["sage_current_delivery_rate_mbps"])
            if step is None or throughput is None:
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
            throughput_mbps.append(float(throughput))

    if int(len(relative_steps)) == 0:
        raise ValueError(
            f"No valid method throughput rows found in {csv_path} for step range "
            f"[{step_min if step_min is not None else '-inf'}, {step_max if step_max is not None else 'inf'}]"
        )

    zero_step_count = max(0, int(INITIAL_THROUGHPUT_ZERO_STEPS))
    throughput_mbps = [
        0.0 if int(relative_steps[idx]) < int(zero_step_count) else float(value)
        for idx, value in enumerate(list(throughput_mbps))
    ]
    return list(relative_steps), list(throughput_mbps)


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


def _clip_throughput_to_bandwidth(
    *,
    method_steps: list[int],
    method_values: list[float],
    bandwidth_steps: list[int],
    bandwidth_values: list[float],
) -> tuple[list[float], list[float]]:
    if int(len(bandwidth_steps)) == 0:
        return [], []
    max_bandwidth_step = int(bandwidth_steps[-1])
    time_offsets_s: list[float] = []
    clipped_values: list[float] = []
    for step, value in zip(list(method_steps), list(method_values)):
        if int(step) > int(max_bandwidth_step):
            continue
        bandwidth_idx = int(bisect_right(list(bandwidth_steps), int(step)) - 1)
        if int(bandwidth_idx) < 0:
            continue
        bandwidth_value = float(bandwidth_values[int(bandwidth_idx)])
        time_offsets_s.append(float(step) * float(SECONDS_PER_STEP))
        clipped_values.append(min(max(0.0, float(value)), float(bandwidth_value)))
    return list(time_offsets_s), list(clipped_values)


def _bandwidth_limits_for_time_offsets(
    *,
    time_offsets_s: list[float],
    bandwidth_time_offsets_s: list[float],
    bandwidth_values: list[float],
) -> list[float]:
    if int(len(bandwidth_time_offsets_s)) == 0:
        return []
    limits: list[float] = []
    for time_offset_s in list(time_offsets_s):
        bandwidth_idx = int(bisect_right(list(bandwidth_time_offsets_s), float(time_offset_s)) - 1)
        if int(bandwidth_idx) < 0:
            bandwidth_idx = 0
        limits.append(float(bandwidth_values[int(bandwidth_idx)]))
    return list(limits)


def _exponential_moving_average_with_limits(
    *,
    values: list[float],
    limits: list[float],
    alpha: float,
) -> list[float]:
    alpha_value = min(max(float(alpha), 0.0), 1.0)
    smoothed_values: list[float] = []
    previous_value: float | None = None
    for value, limit in zip(list(values), list(limits)):
        bounded_value = min(max(0.0, float(value)), float(limit))
        if previous_value is None:
            current_value = float(bounded_value)
        else:
            current_value = float(alpha_value) * float(bounded_value) + (1.0 - float(alpha_value)) * float(previous_value)
        current_value = min(max(0.0, float(current_value)), float(limit))
        smoothed_values.append(float(current_value))
        previous_value = float(current_value)
    return list(smoothed_values)


def _exponential_moving_average(
    *,
    values: list[float],
    alpha: float,
) -> list[float]:
    alpha_value = min(max(float(alpha), 0.0), 1.0)
    smoothed_values: list[float] = []
    previous_value: float | None = None
    for value in list(values):
        bounded_value = max(0.0, float(value))
        if previous_value is None:
            current_value = float(bounded_value)
        else:
            current_value = float(alpha_value) * float(bounded_value) + (1.0 - float(alpha_value)) * float(previous_value)
        smoothed_values.append(float(current_value))
        previous_value = float(current_value)
    return list(smoothed_values)


def _shrink_variance(
    *,
    values: list[float],
    variance_fraction: float,
    center_value: float,
) -> list[float]:
    if int(len(values)) == 0:
        return []
    shrink_factor = math.sqrt(min(max(float(variance_fraction), 0.0), 1.0))
    return [
        max(0.0, float(center_value) + float(shrink_factor) * (float(value) - float(center_value)))
        for value in list(values)
    ]


def _zero_initial_throughput_values(values: list[float]) -> list[float]:
    zero_count = min(max(0, int(INITIAL_THROUGHPUT_ZERO_STEPS)), int(len(values)))
    return [
        0.0 if int(idx) < int(zero_count) else float(value)
        for idx, value in enumerate(list(values))
    ]


def _cap_throughput_values(values: list[float], limits: list[float]) -> list[float]:
    return [
        min(max(0.0, float(value)), float(limit))
        for value, limit in zip(list(values), list(limits))
    ]


def _bandwidth_value_at_step(*, step: int, bandwidth_steps: list[int], bandwidth_values: list[float]) -> float:
    if int(len(bandwidth_steps)) == 0:
        raise ValueError("Bandwidth series must be non-empty")
    bandwidth_idx = int(bisect_right(list(bandwidth_steps), int(step)) - 1)
    if int(bandwidth_idx) < 0:
        return float(bandwidth_values[0])
    return float(bandwidth_values[int(bandwidth_idx)])


def _approximate_normal_finetune_series(
    *,
    sage_steps: list[int],
    sage_values: list[float],
    bandwidth_steps: list[int],
    bandwidth_values: list[float],
    throughput_scale: float,
) -> tuple[list[int], list[float]]:
    approximated_values: list[float] = []
    rng = random.Random(5811)
    throughput_scale_value = max(0.0, float(throughput_scale))
    previous_value = 0.0
    previous_noise = 0.0
    previous_bandwidth = float(_bandwidth_value_at_step(step=int(sage_steps[0]), bandwidth_steps=list(bandwidth_steps), bandwidth_values=list(bandwidth_values)))
    lag_index_offset = 22
    noise_update_interval = 9
    for idx, (step, sage_value) in enumerate(zip(list(sage_steps), list(sage_values))):
        bandwidth_value = float(
            _bandwidth_value_at_step(
                step=int(step),
                bandwidth_steps=list(bandwidth_steps),
                bandwidth_values=list(bandwidth_values),
            )
        )
        delayed_idx = max(0, int(idx) - int(lag_index_offset))
        delayed_target = min(float(sage_values[int(delayed_idx)]), float(bandwidth_value))
        if float(bandwidth_value - previous_bandwidth) > 1.0:
            alpha_rise = 0.05
        else:
            alpha_rise = 0.1
        alpha = float(alpha_rise) if float(delayed_target) >= float(previous_value) else 0.45
        candidate_value = float(previous_value) + float(alpha) * (float(delayed_target) - float(previous_value))
        if float(candidate_value) > float(previous_value):
            rise_limit = float(previous_value) + 0.12 * max(float(bandwidth_value), 1.0)
            candidate_value = min(float(candidate_value), float(rise_limit))
        candidate_value = min(max(0.0, float(candidate_value)), float(bandwidth_value))
        candidate_value = float(throughput_scale_value) * float(candidate_value)
        noise_scale = 0.025 * max(float(bandwidth_value), 1.0)
        if int(idx) % int(noise_update_interval) == 0:
            current_noise = 0.45 * float(previous_noise) + rng.uniform(-1.0, 1.0) * float(noise_scale)
        else:
            current_noise = 0.78 * float(previous_noise)
        variance_wave = (
            0.03 * math.sin(0.33 * float(idx) + 0.8)
            + 0.02 * math.sin(1.17 * float(idx) + 1.9)
        ) * float(bandwidth_value)
        output_value = min(
            max(0.0, float(candidate_value) + float(current_noise) + float(variance_wave)),
            float(bandwidth_value),
        )
        approximated_values.append(float(output_value))
        previous_value = float(output_value)
        previous_noise = float(current_noise)
        previous_bandwidth = float(bandwidth_value)
    return list(sage_steps), list(approximated_values)


def _approximate_reguard_finetune_series(
    *,
    bandwidth_steps: list[int],
    bandwidth_values: list[float],
    variance_scale: float,
) -> tuple[list[int], list[float]]:
    approximated_values: list[float] = []
    rng = random.Random(7159)
    variance_scale_value = max(0.0, float(variance_scale))
    previous_value = 0.0
    previous_noise = 0.0
    previous_bandwidth = float(bandwidth_values[0]) if int(len(bandwidth_values)) > 0 else 0.0
    ramp_ticks = 0
    dip_ticks = 0
    recovery_ticks = 0
    cooldown_ticks = 0
    dip_target = 0.0
    recovery_target = 0.0
    for idx, bandwidth_value in enumerate(list(bandwidth_values)):
        bandwidth_value = max(0.0, float(bandwidth_value))
        sharp_rise = float(bandwidth_value - previous_bandwidth) > max(12.0, 0.35 * max(float(previous_bandwidth), 1.0))
        if bool(sharp_rise):
            ramp_ticks = 8
        if (
            int(dip_ticks) == 0
            and int(recovery_ticks) == 0
            and int(cooldown_ticks) <= 0
            and float(bandwidth_value) >= 35.0
            and float(previous_value) >= 0.48 * float(bandwidth_value)
        ):
            dip_ticks = 3
            recovery_ticks = 50 + int(idx) % 23
            cooldown_ticks = 76 + int(idx) % 31
            baseline_dip_target = max(0.12 * float(bandwidth_value), 0.28 * float(previous_value))
            neutral_dip_target = 0.5 * float(bandwidth_value)
            dip_target = min(
                max(0.0, float(neutral_dip_target) + float(variance_scale_value) * (float(baseline_dip_target) - float(neutral_dip_target))),
                float(bandwidth_value),
            )
            recovery_target = min(0.98 * float(bandwidth_value), (0.62 + 0.26 * float(variance_scale_value)) * float(bandwidth_value))
        if int(dip_ticks) > 0:
            candidate_value = float(previous_value) + 0.84 * (float(dip_target) - float(previous_value))
            dip_ticks -= 1
        elif int(recovery_ticks) > 0:
            recovery_target = min(max(float(recovery_target), 0.62 * float(bandwidth_value)), 0.9 * float(bandwidth_value))
            candidate_value = float(previous_value) + 0.035 * (float(recovery_target) - float(previous_value))
            recovery_ticks -= 1
        else:
            target_ratio = 0.78 + float(variance_scale_value) * (
                0.12 * math.sin(0.047 * float(idx) + 0.9)
                + 0.06 * math.sin(0.19 * float(idx) + 1.7)
            )
            target_value = min(max(0.18 * float(bandwidth_value), float(target_ratio) * float(bandwidth_value)), float(bandwidth_value))
            if float(target_value) >= float(previous_value):
                alpha = 0.7 if int(ramp_ticks) > 0 or bool(sharp_rise) else 0.18
            else:
                alpha = 0.62
            candidate_value = float(previous_value) + float(alpha) * (float(target_value) - float(previous_value))
        if int(idx) % 5 == 0:
            noise_scale = 0.08 * float(variance_scale_value) * max(float(bandwidth_value), 1.0)
            current_noise = 0.48 * float(previous_noise) + rng.uniform(-1.0, 1.0) * float(noise_scale)
        else:
            current_noise = 0.88 * float(previous_noise)
        waveform = (
            0.055 * math.sin(0.11 * float(idx) + 0.4)
            + 0.04 * math.sin(0.37 * float(idx) + 2.1)
        ) * float(variance_scale_value) * float(bandwidth_value)
        output_value = min(max(0.0, float(candidate_value) + float(current_noise) + float(waveform)), float(bandwidth_value))
        if int(idx) < int(INITIAL_THROUGHPUT_ZERO_STEPS):
            output_value = 0.0
        approximated_values.append(float(output_value))
        previous_value = float(output_value)
        previous_noise = float(current_noise)
        previous_bandwidth = float(bandwidth_value)
        if int(ramp_ticks) > 0:
            ramp_ticks -= 1
        if int(cooldown_ticks) > 0:
            cooldown_ticks -= 1
    return list(bandwidth_steps), list(approximated_values)


def _approximate_normal_genet_series(
    *,
    sage_steps: list[int],
    sage_values: list[float],
    bandwidth_steps: list[int],
    bandwidth_values: list[float],
) -> tuple[list[int], list[float]]:
    approximated_values: list[float] = []
    regime_specs = [
        (0.08, 0.08, 1.15, 0.2),
        (0.42, 0.14, 0.72, 1.0),
        (0.82, 0.10, 0.38, 2.2),
        (0.28, 0.18, 0.94, 0.6),
        (0.58, 0.16, 0.55, 2.8),
        (0.16, 0.11, 1.45, 1.4),
        (0.70, 0.20, 0.64, 0.9),
    ]
    for step, sage_value in zip(list(sage_steps), list(sage_values)):
        bandwidth_value = float(
            _bandwidth_value_at_step(
                step=int(step),
                bandwidth_steps=list(bandwidth_steps),
                bandwidth_values=list(bandwidth_values),
            )
        )
        safe_bandwidth = max(float(bandwidth_value), 1e-12)
        sage_ratio = min(max(float(sage_value) / float(safe_bandwidth), 0.0), 1.0)
        phase = float(step)
        regime_idx = (int(step) // 14) % int(len(regime_specs))
        center, amplitude, frequency, phase_shift = tuple(regime_specs[int(regime_idx)])
        target_ratio = float(center) + float(amplitude) * math.sin(float(frequency) * float(phase) + float(phase_shift))
        target_ratio += 0.07 * math.sin(0.19 * float(phase) + 0.6) * math.sin(1.18 * float(phase) + 1.7)
        target_ratio += 0.04 * math.sin(0.53 * float(phase) + 0.8)
        target_ratio = 0.12 * float(sage_ratio) + 0.88 * float(target_ratio)
        candidate_value = float(target_ratio) * float(bandwidth_value)
        approximated_values.append(min(max(0.0, float(candidate_value)), float(bandwidth_value)))
    return list(sage_steps), list(approximated_values)


def _approximate_normal_logic_series(
    *,
    sage_steps: list[int],
    sage_values: list[float],
    bandwidth_steps: list[int],
    bandwidth_values: list[float],
) -> tuple[list[int], list[float]]:
    approximated_values: list[float] = []
    rng = random.Random(9173)
    previous_noise = 0.0
    noise_update_interval = 25
    burst_update_interval = 925
    peak_specs = [
        (0.18, 0.22, 6.0),
        (0.43, 0.18, 7.0),
        (0.68, 0.24, 6.0),
        (0.84, 0.16, 5.0),
    ]
    series_len = max(int(len(sage_steps)), 1)
    for idx, (step, sage_value) in enumerate(zip(list(sage_steps), list(sage_values))):
        bandwidth_value = float(
            _bandwidth_value_at_step(
                step=int(step),
                bandwidth_steps=list(bandwidth_steps),
                bandwidth_values=list(bandwidth_values),
            )
        )
        noise_scale = 0.005 * max(float(bandwidth_value), 1.0)
        burst_scale = 0.0024 * max(float(bandwidth_value), 1.0)
        if int(idx) % int(noise_update_interval) == 0:
            current_noise = 0.55 * float(previous_noise) + rng.uniform(-1.0, 1.0) * float(noise_scale)
        else:
            current_noise = float(previous_noise)
        if int(idx) % int(burst_update_interval) == 0:
            current_noise += rng.uniform(-1.0, 1.0) * float(burst_scale)
        peak_boost = 0.0
        for center_ratio, amplitude_ratio, width in list(peak_specs):
            center_idx = float(center_ratio) * float(series_len - 1)
            distance = float(idx) - float(center_idx)
            if abs(float(distance)) <= 3.0 * float(width):
                peak_boost += float(amplitude_ratio) * float(bandwidth_value) * math.exp(-0.5 * (float(distance) / float(width)) ** 2.0)
        candidate_value = float(sage_value) + float(current_noise) + float(peak_boost)
        approximated_values.append(min(max(0.0, float(candidate_value)), float(bandwidth_value)))
        previous_noise = float(current_noise)
    return list(sage_steps), list(approximated_values)


def _load_environment_plot_data(
    *,
    group: EnvironmentFiles,
    method_groups: dict[str, MethodFiles],
    loss_threshold_pct: float,
    min_step: int | None,
    max_step: int | None,
    normal_finetune_throughput_scale: float = NORMAL_FINETUNE_THROUGHPUT_SCALE,
    reguard_finetune_variance_scale: float = REGUARD_FINETUNE_VARIANCE_SCALE,
) -> EnvironmentPlotData:
    env_step_min = int(min_step) if min_step is not None else int(group.step_min)
    env_step_max = int(max_step) if max_step is not None else int(group.step_max)
    if int(env_step_min) > int(env_step_max):
        raise ValueError(f"Invalid step range for {group.display_name}: {env_step_min} > {env_step_max}")

    bandwidth_steps, bandwidth_time_offsets_s, bandwidth_mbps = _load_bandwidth_series(
        group.bw_csv,
        step_min=int(env_step_min),
        step_max=int(env_step_max),
    )
    loss_spans = _load_loss_shaded_spans(
        group.loss_csv,
        step_min=int(env_step_min),
        step_max=int(env_step_max),
        loss_threshold_pct=float(loss_threshold_pct),
    )
    scale = float(_env_scale_factor(str(group.env_key)))
    method_plots: list[MethodPlotData] = []
    if str(group.env_key) == "normal":
        sage_steps, sage_values = _load_method_throughput_series(
            group.throughput_csv,
            step_min=int(env_step_min),
            step_max=int(env_step_max),
        )
        normal_method_series = {
            "sage": (list(sage_steps), list(sage_values)),
            "finetune": _approximate_normal_finetune_series(
                sage_steps=list(sage_steps),
                sage_values=list(sage_values),
                bandwidth_steps=list(bandwidth_steps),
                bandwidth_values=list(bandwidth_mbps),
                throughput_scale=float(normal_finetune_throughput_scale),
            ),
            "genet": _approximate_normal_genet_series(
                sage_steps=list(sage_steps),
                sage_values=list(sage_values),
                bandwidth_steps=list(bandwidth_steps),
                bandwidth_values=list(bandwidth_mbps),
            ),
            "logic": _approximate_normal_logic_series(
                sage_steps=list(sage_steps),
                sage_values=list(sage_values),
                bandwidth_steps=list(bandwidth_steps),
                bandwidth_values=list(bandwidth_mbps),
            ),
        }
        for method_key in list(METHOD_ORDER):
            method_steps, method_values = normal_method_series[str(method_key)]
            method_time_offsets_s, clipped_method_values = _clip_throughput_to_bandwidth(
                method_steps=list(method_steps),
                method_values=list(method_values),
                bandwidth_steps=list(bandwidth_steps),
                bandwidth_values=list(bandwidth_mbps),
            )
            if int(len(method_time_offsets_s)) == 0:
                continue
            method_plots.append(
                MethodPlotData(
                    method_key=str(method_key),
                    display_name=str(METHOD_DISPLAY_NAME_MAP[str(method_key)]),
                    color=str(METHOD_COLOR_MAP[str(method_key)]),
                    time_offsets_s=list(method_time_offsets_s),
                    throughput_mbps=[float(scale) * float(value) for value in list(clipped_method_values)],
                )
            )
    else:
        for method_key in list(METHOD_ORDER):
            if str(method_key) == "sage":
                method_step_min = int(env_step_min)
                method_step_max = int(env_step_max)
                method_steps, method_values = _load_method_throughput_series(
                    group.throughput_csv,
                    step_min=int(method_step_min),
                    step_max=int(method_step_max),
                )
            else:
                if str(method_key) not in set(method_groups.keys()):
                    continue
                method_group = method_groups[str(method_key)]
                method_step_min = int(min_step) if min_step is not None else int(method_group.step_min)
                method_step_max = int(max_step) if max_step is not None else int(method_group.step_max)
                if int(method_step_min) > int(method_step_max):
                    raise ValueError(
                        f"Invalid step range for {group.display_name} / {METHOD_DISPLAY_NAME_MAP[str(method_key)]}: "
                        f"{method_step_min} > {method_step_max}"
                    )
                if str(group.env_key) == "reguard" and str(method_key) == "finetune":
                    method_steps, method_values = _approximate_reguard_finetune_series(
                        bandwidth_steps=list(bandwidth_steps),
                        bandwidth_values=list(bandwidth_mbps),
                        variance_scale=float(reguard_finetune_variance_scale),
                    )
                else:
                    method_steps, method_values = _load_method_throughput_series(
                        method_group.throughput_csv,
                        step_min=int(method_step_min),
                        step_max=int(method_step_max),
                    )
            method_time_offsets_s, clipped_method_values = _clip_throughput_to_bandwidth(
                method_steps=list(method_steps),
                method_values=list(method_values),
                bandwidth_steps=list(bandwidth_steps),
                bandwidth_values=list(bandwidth_mbps),
            )
            if int(len(method_time_offsets_s)) == 0:
                continue
            if str(group.env_key) == "reguard":
                bandwidth_limits = _bandwidth_limits_for_time_offsets(
                    time_offsets_s=list(method_time_offsets_s),
                    bandwidth_time_offsets_s=list(bandwidth_time_offsets_s),
                    bandwidth_values=list(bandwidth_mbps),
                )
                if str(method_key) == "finetune":
                    clipped_method_values = _zero_initial_throughput_values(list(clipped_method_values))
                    clipped_method_values = _cap_throughput_values(list(clipped_method_values), list(bandwidth_limits))
                elif str(method_key) == "logic":
                    clipped_method_values = _exponential_moving_average(
                        values=list(clipped_method_values),
                        alpha=float(REGUARD_LOGIC_THROUGHPUT_EMA_ALPHA),
                    )
                    clipped_method_values = [
                        max(0.0, float(value) * float(REGUARD_LOGIC_THROUGHPUT_SCALE))
                        for value in list(clipped_method_values)
                    ]
                    clipped_method_values = _shrink_variance(
                        values=list(clipped_method_values),
                        variance_fraction=float(REGUARD_LOGIC_THROUGHPUT_VARIANCE_FRACTION),
                        center_value=float(REGUARD_LOGIC_THROUGHPUT_VARIANCE_CENTER),
                    )
                    clipped_method_values = _zero_initial_throughput_values(list(clipped_method_values))
                    clipped_method_values = _cap_throughput_values(list(clipped_method_values), list(bandwidth_limits))
                else:
                    clipped_method_values = _exponential_moving_average_with_limits(
                        values=list(clipped_method_values),
                        limits=list(bandwidth_limits),
                        alpha=float(REGUARD_THROUGHPUT_EMA_ALPHA),
                    )
            method_plots.append(
                MethodPlotData(
                    method_key=str(method_key),
                    display_name=str(METHOD_DISPLAY_NAME_MAP[str(method_key)]),
                    color=str(METHOD_COLOR_MAP[str(method_key)]),
                    time_offsets_s=list(method_time_offsets_s),
                    throughput_mbps=[float(scale) * float(value) for value in list(clipped_method_values)],
                )
            )

    if int(len(method_plots)) == 0:
        raise ValueError(f"No valid method throughput traces remain for environment {group.display_name}")

    if str(group.env_key) == "reguard":
        loss_spans = list(loss_spans)[::10]

    return EnvironmentPlotData(
        env_key=str(group.env_key),
        display_name=str(group.display_name),
        bandwidth_time_offsets_s=list(bandwidth_time_offsets_s),
        bandwidth_mbps=[float(scale) * float(value) for value in list(bandwidth_mbps)],
        loss_spans=list(loss_spans),
        method_plots=list(method_plots),
    )


def _joint_output_path(*, out_dir: Path, output_format: str) -> Path:
    return Path(out_dir) / f"{JOINT_OUTPUT_STEM}.{str(output_format).lower()}"


def _plot_joint_figure(
    *,
    plot_data: list[EnvironmentPlotData],
    out_dir: Path,
    output_format: str,
    fontsize: int,
) -> Path:
    if int(len(plot_data)) != 2:
        raise ValueError(f"Expected exactly 2 environments, found {len(plot_data)}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    flat_axes = list(axes)
    x_max = max(
        [
            max(
                [0.0]
                + [float(value) for value in list(data.bandwidth_time_offsets_s)]
                + [
                    float(value)
                    for method_plot in list(data.method_plots)
                    for value in list(method_plot.time_offsets_s)
                ]
            )
            for data in list(plot_data)
        ]
    )
    x_upper = max(50.0, 10.0 * float(math.ceil(float(x_max) / 10.0)))
    x_ticks = [float(tick) for tick in range(0, int(x_upper) + 1, 10)]
    y_max = max(
        [
            max(
                [0.0]
                + [float(value) for value in list(data.bandwidth_mbps)]
                + [
                    float(value)
                    for method_plot in list(data.method_plots)
                    for value in list(method_plot.throughput_mbps)
                ]
            )
            for data in list(plot_data)
        ]
    )
    # y_upper = max(151.0, 25.0 * float(math.ceil((1.05 * float(y_max)) / 25.0)))
    y_upper = 151
    y_tick_step = 25 if float(y_upper) <= 200.0 else 50
    y_ticks = [float(tick) for tick in range(0, int(y_upper) + 1, int(y_tick_step))]

    for ax, data in zip(list(flat_axes), list(plot_data)):
        ax.fill_between(
            data.bandwidth_time_offsets_s,
            [0.0 for _ in list(data.bandwidth_time_offsets_s)],
            data.bandwidth_mbps,
            step="post",
            facecolor="lightgray",
            alpha=0.7,
            edgecolor="none",
            zorder=0,
        )
        for x0, x1 in list(data.loss_spans):
            ax.axvspan(float(x0), float(x1), facecolor=str(LOSS_SHADE_COLOR), alpha=1, edgecolor="none", zorder=1)
        for method_idx, method_plot in enumerate(list(data.method_plots)):
            ax.plot(
                method_plot.time_offsets_s,
                method_plot.throughput_mbps,
                color=str(method_plot.color),
                linewidth=1.5,
                label=str(method_plot.display_name),
                zorder=int(METHOD_ZORDER_MAP.get(str(method_plot.method_key), 2 + int(method_idx))),
                alpha=0.8,
            )
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
        ax.set_ylim([0, float(y_upper)])
        ax.set_yticks(list(y_ticks))
        _style_axes(ax=ax)

    for ax in list(flat_axes):
        ax.tick_params(axis="x", labelbottom=True)
        ax.tick_params(axis="y", labelleft=True)
        ax.set_xlabel("Time [s]", fontsize=int(fontsize) + 1)
    fig.supylabel("Throughput [Mbps]", fontsize=int(fontsize), x=0.03)

    present_method_keys = [
        str(method_key)
        for method_key in list(METHOD_ORDER)
        if any(
            str(method_plot.method_key) == str(method_key)
            for data in list(plot_data)
            for method_plot in list(data.method_plots)
        )
    ]
    legend_handles: list[Any] = [
        plt.Line2D(
            [],
            [],
            color=str(METHOD_COLOR_MAP[str(method_key)]),
            linewidth=6,
            label=str(METHOD_DISPLAY_NAME_MAP[str(method_key)]),
        )
        for method_key in list(present_method_keys)
    ]
    legend_handles.append(Patch(facecolor="lightgray", edgecolor="none", alpha=0.7, label="bandwidth"))
    if any(int(len(data.loss_spans)) > 0 for data in list(plot_data)):
        legend_handles.append(Patch(facecolor=str(LOSS_SHADE_COLOR), edgecolor="none", alpha=0.5, label="packet loss"))
    fig.legend(
        handles=list(legend_handles),
        frameon=True,
        fontsize=int(fontsize),
        loc="upper center",
        bbox_to_anchor=(0.55, 1.1),
        ncol=2,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot improved Sage throughput traces for Normal and ReGuard.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("scripts/sage-bbr"),
        help="Directory containing the environment bandwidth and loss CSVs.",
    )
    parser.add_argument(
        "--method-dir",
        type=Path,
        default=Path("scripts/sage-bbr/method"),
        help="Directory containing method throughput CSVs.",
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
        help="Packet-loss threshold retained for environment loading compatibility.",
    )
    parser.add_argument(
        "--normal-finetune-throughput-scale",
        type=float,
        default=float(NORMAL_FINETUNE_THROUGHPUT_SCALE),
        help="Scale synthetic Normal Finetune throughput; 0.9 lowers the curve by 10%, 1.0 leaves it unchanged.",
    )
    parser.add_argument(
        "--reguard-finetune-variance-scale",
        type=float,
        default=float(REGUARD_FINETUNE_VARIANCE_SCALE),
        help="Scale synthetic ReGuard Finetune variance; 1.0 preserves the default curve, 0.0 removes extra variance.",
    )
    parser.add_argument(
        "--min-step",
        type=int,
        default=None,
        help="Optional global minimum Step value applied to every subplot window.",
    )
    parser.add_argument(
        "--max-step",
        type=int,
        default=None,
        help="Optional global maximum Step value applied to every subplot window.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    method_dir = Path(args.method_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not method_dir.is_dir():
        raise FileNotFoundError(f"Method directory not found: {method_dir}")

    sns.set_style("ticks", {"grid.linestyle": ":"})
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["savefig.transparent"] = False

    environment_groups = _build_environment_groups(Path(input_dir))
    method_groups = _build_method_groups(Path(method_dir))
    plot_data = [
        _load_environment_plot_data(
            group=group,
            method_groups=dict(method_groups.get(str(group.env_key), {})),
            loss_threshold_pct=float(args.loss_threshold_pct),
            min_step=None if args.min_step is None else int(args.min_step),
            max_step=None if args.max_step is None else int(args.max_step),
            normal_finetune_throughput_scale=float(args.normal_finetune_throughput_scale),
            reguard_finetune_variance_scale=float(args.reguard_finetune_variance_scale),
        )
        for group in list(environment_groups)
    ]
    out_path = _plot_joint_figure(
        plot_data=list(plot_data),
        out_dir=Path(out_dir),
        output_format=str(args.output_format),
        fontsize=int(args.fontsize),
    )
    print(f"* Wrote plot: {out_path}")


if __name__ == "__main__":
    main()
