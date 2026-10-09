import os
import sys
from typing import Any


def add_pensieve_src_to_path() -> str:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    src_dir = os.path.join(repo_root, "Pensieve-PPO", "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    return src_dir


def import_pensieve() -> dict[str, Any]:
    add_pensieve_src_to_path()

    import env as pensieve_env  # type: ignore
    import fixed_env as fixed_env  # type: ignore

    return {
        "env": pensieve_env,
        "fixed_env": fixed_env,
    }

