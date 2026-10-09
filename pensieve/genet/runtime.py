from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from online.pensieve_bridge import add_pensieve_src_to_path


def resolve_repo_path(path: str) -> str:
    expanded = os.path.expandvars(os.path.expanduser(str(path)))
    if os.path.isabs(expanded):
        return expanded

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    under_root = os.path.join(repo_root, expanded)
    if os.path.exists(under_root):
        return under_root

    under_src = os.path.join(repo_root, "Pensieve-PPO", "src", expanded)
    if os.path.exists(under_src):
        return under_src

    return under_root


@dataclass(frozen=True)
class PensieveRuntime:
    actor: Any
    util_mod: Any
    fixed_env_mod: Any


def load_runtime(model_path: str) -> PensieveRuntime:
    add_pensieve_src_to_path()

    import ppo2 as network  # type: ignore
    import util as util_mod  # type: ignore
    import fixed_env as fixed_env_mod  # type: ignore

    actor = network.Network(
        state_dim=[int(util_mod.S_INFO), int(util_mod.S_LEN)],
        action_dim=int(util_mod.A_DIM),
        learning_rate=float(util_mod.ACTOR_LR_RATE),
    )
    actor.load_model(resolve_repo_path(str(model_path)))
    return PensieveRuntime(actor=actor, util_mod=util_mod, fixed_env_mod=fixed_env_mod)

