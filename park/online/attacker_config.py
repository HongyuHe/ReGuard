from __future__ import annotations

import json
from pathlib import Path


def _candidate_config_paths(model_path: str) -> list[Path]:
    model = Path(model_path)
    candidates = [model.with_suffix(".config.json")]
    if model.stem.endswith("_best"):
        candidates.append(model.with_name(f"{model.stem[:-5]}.config.json"))
    if model.name == "best_model.zip":
        candidates.append(model.with_name("best_model.config.json"))
    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered.append(path)
    return ordered


def load_attacker_config(model_path: str) -> dict | None:
    for candidate in _candidate_config_paths(str(model_path)):
        if candidate.is_file():
            with candidate.open("r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                return obj
    return None


def infer_include_last_gap(model_path: str, default: bool = True) -> bool:
    config = load_attacker_config(str(model_path))
    if config is None:
        return bool(default)
    value = config.get("include_last_gap")
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        return bool(int(value))
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)
