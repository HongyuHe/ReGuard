#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from figure_inputs import plot_input_snapshot


FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = FRAMEWORK_ROOT.parent
LOCAL_FRAMEWORK = FRAMEWORK_ROOT.name
FIGURES_ROOT = REPO_ROOT / "figures"


@dataclass(frozen=True)
class PlotGroup:
    cwd: Path
    command: tuple[str, ...]


@dataclass(frozen=True)
class FigureSpec:
    framework: str
    group: str
    source_pdf: str
    plotting_script: str
    source_results: str


def _plot_python() -> str:
    configured = str(os.environ.get("REGUARD_PLOT_PYTHON", "")).strip()
    if configured:
        return str(Path(configured).expanduser())
    bundled = REPO_ROOT / "pensieve" / "venv" / "bin" / "python"
    if bundled.is_file():
        return str(bundled)
    return sys.executable


def _plot_groups(data_root: Path, build_root: Path) -> dict[str, PlotGroup]:
    py = _plot_python()
    park_data = pensieve_data = sage_data = data_root
    return {
        "pensieve-main": PlotGroup(
            cwd=REPO_ROOT / "pensieve",
            command=(
                py,
                "scripts/plot_final.py",
                "--trace-csv",
                str(pensieve_data / "eval_oracle_regret_traces_combined-new.csv"),
                "--step-csv",
                str(pensieve_data / "eval_oracle_regret_steps_combined-new.csv"),
                "--ablation-trace-csv",
                str(pensieve_data / "eval_oracle_regret_traces_shield-ablation.csv"),
                "--ablation-step-csv",
                str(pensieve_data / "eval_oracle_regret_steps_shield-ablation.csv"),
                "--out-dir",
                str(build_root / "pensieve-main"),
                "--output-format",
                "pdf",
            ),
        ),
        "pensieve-iters": PlotGroup(
            cwd=REPO_ROOT / "pensieve",
            command=(
                py,
                "scripts/plot_final.py",
                "--trace-csv",
                str(pensieve_data / "eval_oracle_regret_traces_iters.csv"),
                "--step-csv",
                str(pensieve_data / "eval_oracle_regret_steps_iters.csv"),
                "--out-dir",
                str(build_root / "pensieve-iters"),
                "--output-format",
                "pdf",
                "--gap-only",
                "--ablation-trace-csv",
                str(pensieve_data / "eval_oracle_regret_traces_shield-ablation.csv"),
                "--ablation-step-csv",
                str(pensieve_data / "eval_oracle_regret_steps_shield-ablation.csv"),
            ),
        ),
        "sage-main": PlotGroup(
            cwd=REPO_ROOT / "sage",
            command=(
                py,
                "scripts/plot_final.py",
                "--summary-path",
                str(sage_data / "eval-300k-50ms-new"),
                "--out-dir",
                str(build_root / "sage-main"),
                "--output-format",
                "pdf",
            ),
        ),
        "sage-iters": PlotGroup(
            cwd=REPO_ROOT / "sage",
            command=(
                py,
                "scripts/plot_final.py",
                "--summary-path",
                str(sage_data / "eval-iters"),
                "--out-dir",
                str(build_root / "sage-iters"),
                "--output-format",
                "pdf",
                "--gap-only",
            ),
        ),
        "park-main": PlotGroup(
            cwd=REPO_ROOT / "park",
            command=(
                py,
                "scripts/plot_final.py",
                "--eval-root",
                str(park_data / "v2"),
                "--out-dir",
                str(build_root / "park-main"),
                "--output-format",
                "pdf",
            ),
        ),
        "park-iters": PlotGroup(
            cwd=REPO_ROOT / "park",
            command=(
                py,
                "scripts/plot_final.py",
                "--eval-root",
                str(park_data / "shielded-iters"),
                "--out-dir",
                str(build_root / "park-iters"),
                "--output-format",
                "pdf",
                "--iter-gap",
            ),
        ),
        "park-diagnostics": PlotGroup(
            cwd=REPO_ROOT / "park",
            command=(
                py,
                "scripts/plot_park_failure_diagnostics.py",
                "--data-dir",
                str(park_data / "diagnostics"),
                "--out-dir",
                str(build_root / "park-diagnostics"),
                "--format",
                "pdf",
            ),
        ),
    }


FIGURES: dict[str, FigureSpec] = {
    "figure_4a": FigureSpec("Pensieve", "pensieve-main", "performance_gap.pdf", "pensieve/scripts/plot_final.py", "pensieve/eval_oracle_regret_{traces,steps}_combined-new.csv"),
    "figure_4b": FigureSpec("Sage", "sage-main", "performance_gap.pdf", "sage/scripts/plot_final.py", "sage/eval-300k-50ms-new"),
    "figure_4c": FigureSpec("Park", "park-main", "performance_gap.pdf", "park/scripts/plot_final.py", "park/v2"),
    "figure_5a": FigureSpec("Pensieve", "pensieve-main", "relative_oracle_performance_heatmap.pdf", "pensieve/scripts/plot_final.py", "pensieve/eval_oracle_regret_{traces,steps}_combined-new.csv"),
    "figure_5b": FigureSpec("Sage", "sage-main", "relative_oracle_performance_heatmap.pdf", "sage/scripts/plot_final.py", "sage/eval-300k-50ms-new"),
    "figure_5c": FigureSpec("Park", "park-main", "relative_oracle_performance_heatmap.pdf", "park/scripts/plot_final.py", "park/v2"),
    "figure_6a": FigureSpec("Pensieve", "pensieve-main", "shield_ablation_variant_strength_vs_environment_gap_log_ci95.pdf", "pensieve/scripts/plot_final.py", "pensieve/eval_oracle_regret_{traces,steps}_shield-ablation.csv"),
    "figure_6b": FigureSpec("Sage", "sage-main", "shield_ablation_variant_strength_vs_environment_gap_log_ci95.pdf", "sage/scripts/plot_final.py", "sage/eval-300k-50ms-new"),
    "figure_6c": FigureSpec("Park", "park-main", "shield_ablation_variant_strength_vs_environment_gap_log_ci95.pdf", "park/scripts/plot_final.py", "park/v2"),
    "figure_7a": FigureSpec("Pensieve", "pensieve-iters", "decision_time_rel_iters_bar_ci95.pdf", "pensieve/scripts/plot_final.py", "pensieve/eval_oracle_regret_{traces,steps}_iters.csv"),
    "figure_7b": FigureSpec("Sage", "sage-iters", "decision_time_reguard_iterations_rel_bar_ci95.pdf", "sage/scripts/plot_final.py", "sage/eval-iters"),
    "figure_7c": FigureSpec("Park", "park-iters", "decision_time_rel_bar_ci95.pdf", "park/scripts/plot_final.py", "park/shielded-iters"),
    "figure_8a": FigureSpec("Pensieve", "pensieve-iters", "performance_gap.pdf", "pensieve/scripts/plot_final.py", "pensieve/eval_oracle_regret_{traces,steps}_iters.csv"),
    "figure_8b": FigureSpec("Sage", "sage-iters", "performance_gap.pdf", "sage/scripts/plot_final.py", "sage/eval-iters"),
    "figure_8c": FigureSpec("Park", "park-iters", "performance_gap.pdf", "park/scripts/plot_final.py", "park/shielded-iters"),
    "figure_9": FigureSpec("Park", "park-diagnostics", "maxgap_condition_slice_episode_0082.pdf", "park/scripts/plot_park_failure_diagnostics.py", "park/diagnostics"),
    "figure_10": FigureSpec("Park", "park-diagnostics", "park_decision_shift_clean_vs_maxgap.pdf", "park/scripts/plot_park_failure_diagnostics.py", "park/diagnostics"),
}

LOCAL_FIGURES = {
    name: spec
    for name, spec in FIGURES.items()
    if spec.framework.lower() == LOCAL_FRAMEWORK
}


def _run_group(group_name: str, *, data_root: Path, build_root: Path) -> None:
    group = _plot_groups(data_root, build_root)[group_name]
    (build_root / group_name).mkdir(parents=True, exist_ok=True)
    print(f"Running plot group: {group_name}", flush=True)
    subprocess.run(group.command, cwd=group.cwd, check=True)


def _write_manifest(selected: list[str], results_dir: Path, inputs: dict) -> None:
    entries = []
    for name in selected:
        spec = FIGURES[name]
        entries.append(
            {
                "figure": name,
                "framework": spec.framework,
                "plotting_script": spec.plotting_script,
                "source_results": str(results_dir / "evaluation" / spec.source_results.split("/", 1)[1]),
                "original_plot": f"figures/{name}/{name}_expected.png",
                "reproduced_output": f"figures/{name}/{name}.pdf",
                "status": "succeeded",
            }
        )
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repository_root": ".",
        "results_dir": str(results_dir),
        "input_manifest": str(results_dir / "evaluation" / "manifest.json"),
        "input_content_sha256": inputs["content_sha256"],
        "figures": entries,
    }
    manifest_path = FIGURES_ROOT / f"{LOCAL_FRAMEWORK}_reproduction_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce ReGuard figures from one completed experiment.")
    parser.add_argument("--figure", choices=("all", *LOCAL_FIGURES.keys()), default="all")
    parser.add_argument(
        "--results-dir", type=Path, required=True,
        help="Result directory created by the corresponding experiment.",
    )
    args = parser.parse_args()

    selected = list(LOCAL_FIGURES.keys()) if args.figure == "all" else [str(args.figure)]
    groups = sorted({FIGURES[name].group for name in selected})
    results_dir = args.results_dir.expanduser().resolve()
    missing_originals = [
        name for name in selected
        if not (FIGURES_ROOT / name / f"{name}_expected.png").is_file()
    ]
    if missing_originals:
        raise RuntimeError(f"Original plot PNGs are missing for: {', '.join(missing_originals)}")

    with plot_input_snapshot(results_dir, LOCAL_FRAMEWORK) as (data_root, build_root, inputs):
        for group_name in groups:
            _run_group(group_name, data_root=data_root, build_root=build_root)
        for name in selected:
            spec = FIGURES[name]
            source = build_root / spec.group / spec.source_pdf
            if not source.is_file() or source.stat().st_size == 0:
                raise FileNotFoundError(f"Plot group did not produce expected output: {source}")
        for name in selected:
            spec = FIGURES[name]
            source = build_root / spec.group / spec.source_pdf
            output_dir = FIGURES_ROOT / name
            output_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, output_dir / f"{name}.pdf")
        _write_manifest(selected, results_dir, inputs)
    print(f"Reproduced {len(selected)} figure panels under {FIGURES_ROOT}")


if __name__ == "__main__":
    main()
