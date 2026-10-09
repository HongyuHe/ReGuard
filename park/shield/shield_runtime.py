from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from shield.obs_features import feature_columns, features_from_obs


class _ShieldVar:
    def __init__(self, name: str) -> None:
        self.name = str(name)


class _ShieldCmp:
    def __init__(self, op: str, left: object, right: object) -> None:
        self.op = str(op)
        self.left = left
        self.right = right


class _ShieldImplies:
    def __init__(self, antecedent: object, consequent: object) -> None:
        self.antecedent = antecedent
        self.consequent = consequent


class _ShieldOr:
    def __init__(self, args: tuple[object, ...]) -> None:
        self.args = tuple(args)


_RULE_CACHE: dict[tuple[str, float, str], list[list[object]]] = {}


def _flip_op(op: str) -> str:
    if op == "<":
        return ">"
    if op == "<=":
        return ">="
    if op == ">":
        return "<"
    if op == ">=":
        return "<="
    return str(op)


def _flatten_and(expr: object) -> list[object]:
    if isinstance(expr, list):
        out: list[object] = []
        for item in expr:
            out.extend(_flatten_and(item))
        return out
    return [expr]


def _cmp_to_atom(expr: object) -> tuple[str, str, float] | None:
    if not isinstance(expr, _ShieldCmp):
        return None
    left = expr.left
    right = expr.right
    if isinstance(left, _ShieldVar) and isinstance(right, (int, float)):
        return str(left.name), str(expr.op), float(right)
    if isinstance(right, _ShieldVar) and isinstance(left, (int, float)):
        return str(right.name), _flip_op(str(expr.op)), float(left)
    return None


def _cmp(a: float, op: str, b: float) -> bool:
    if not (math.isfinite(float(a)) and math.isfinite(float(b))):
        return False
    if op == "<":
        return float(a) < float(b)
    if op == "<=":
        return float(a) <= float(b)
    if op == ">":
        return float(a) > float(b)
    if op == ">=":
        return float(a) >= float(b)
    if op == "==":
        return float(a) == float(b)
    if op == "!=":
        return float(a) != float(b)
    return False


def load_rules(rules_file: str | os.PathLike[str], *, output_var: str, positive_value: int = 1) -> list[list[object]]:
    path = os.path.abspath(str(rules_file))
    try:
        mtime = float(os.path.getmtime(path))
    except Exception:
        mtime = 0.0
    cache_key = (str(path), float(mtime), str(output_var))
    if cache_key in _RULE_CACHE:
        return _RULE_CACHE[cache_key]

    variable_cache: dict[str, _ShieldVar] = {}

    def Symbol(name: str, *args, **kwargs):
        del args, kwargs
        if str(name) not in variable_cache:
            variable_cache[str(name)] = _ShieldVar(str(name))
        return variable_cache[str(name)]

    def Integer(v):
        return int(v)

    def Float(v, precision=None):
        del precision
        return float(v)

    def And(*args):
        return list(args)

    def Or(*args):
        return _ShieldOr(tuple(args))

    def Implies(a, b):
        return _ShieldImplies(a, b)

    def Equality(a, b):
        return _ShieldCmp("==", a, b)

    def Unequality(a, b):
        return _ShieldCmp("!=", a, b)

    def StrictLessThan(a, b):
        return _ShieldCmp("<", a, b)

    def StrictGreaterThan(a, b):
        return _ShieldCmp(">", a, b)

    def LessThan(a, b):
        return _ShieldCmp("<=", a, b)

    def GreaterThan(a, b):
        return _ShieldCmp(">=", a, b)

    eval_env = {
        "__builtins__": {},
        "Symbol": Symbol,
        "Integer": Integer,
        "Float": Float,
        "And": And,
        "Or": Or,
        "Implies": Implies,
        "Equality": Equality,
        "Unequality": Unequality,
        "StrictLessThan": StrictLessThan,
        "StrictGreaterThan": StrictGreaterThan,
        "LessThan": LessThan,
        "GreaterThan": GreaterThan,
    }

    rules: list[list[object]] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            expr = eval(line, eval_env)
            if not isinstance(expr, _ShieldImplies):
                continue
            consequent = _cmp_to_atom(expr.consequent)
            if consequent is None:
                continue
            cons_var, cons_op, cons_val = consequent
            if str(cons_var) != str(output_var) or str(cons_op) != "==" or int(cons_val) != int(positive_value):
                continue
            atoms: list[object] = []
            ok = True
            for atom_expr in _flatten_and(expr.antecedent):
                if isinstance(atom_expr, _ShieldOr):
                    or_atoms: list[tuple[str, str, float]] = []
                    for sub_expr in atom_expr.args:
                        atom = _cmp_to_atom(sub_expr)
                        if atom is None:
                            ok = False
                            break
                        v, op, val = atom
                        or_atoms.append((str(v), str(op), float(val)))
                    if not ok or len(or_atoms) == 0:
                        ok = False
                        break
                    atoms.append(("or", or_atoms))
                    continue
                atom = _cmp_to_atom(atom_expr)
                if atom is None:
                    ok = False
                    break
                v, op, val = atom
                atoms.append((str(v), str(op), float(val)))
            if bool(ok):
                rules.append(list(atoms))

    _RULE_CACHE[cache_key] = rules
    return rules


def _atom_satisfied(atom: object, *, values: dict[str, float | int]) -> bool:
    if isinstance(atom, tuple) and len(atom) == 2 and str(atom[0]) == "or":
        for sub_atom in list(atom[1]):
            if _atom_satisfied(sub_atom, values=values):
                return True
        return False
    if not (isinstance(atom, tuple) and len(atom) == 3):
        return False
    var, op, val = atom
    if str(var) not in values:
        return False
    return _cmp(float(values[str(var)]), str(op), float(val))


def _rule_satisfied(rule_atoms: list[object], *, values: dict[str, float | int]) -> bool:
    for atom in list(rule_atoms):
        if not _atom_satisfied(atom, values=values):
            return False
    return True


def _any_rule_satisfied(rules: list[list[object]], *, values: dict[str, float | int]) -> bool:
    for rule_atoms in list(rules):
        if _rule_satisfied(list(rule_atoms), values=values):
            return True
    return False


def _argmax_action(scores: np.ndarray, *, mask: np.ndarray | None = None) -> int:
    vals = np.asarray(scores, dtype=np.float64).reshape(-1)
    if mask is None:
        return int(np.argmax(vals))
    allowed = np.asarray(mask, dtype=np.bool_).reshape(-1)
    if vals.shape[0] != allowed.shape[0]:
        raise ValueError("scores and mask must have the same length")
    masked = np.full_like(vals, -np.inf, dtype=np.float64)
    masked[allowed] = vals[allowed]
    if not np.any(np.isfinite(masked)):
        return int(np.argmax(vals))
    return int(np.argmax(masked))


def _sample_action(scores: np.ndarray, *, rng: np.random.Generator, mask: np.ndarray | None = None) -> int:
    vals = np.asarray(scores, dtype=np.float64).reshape(-1)
    noise = rng.gumbel(size=vals.shape)
    noisy = vals + noise
    if mask is None:
        return int(np.argmax(noisy))
    allowed = np.asarray(mask, dtype=np.bool_).reshape(-1)
    masked = np.full_like(noisy, -np.inf, dtype=np.float64)
    masked[allowed] = noisy[allowed]
    if not np.any(np.isfinite(masked)):
        return int(np.argmax(noisy))
    return int(np.argmax(masked))


class ObsOnlyShield:
    def __init__(
        self,
        *,
        num_servers: int,
        risk_rules: list[list[object]],
        allow_rules_by_action: dict[int, list[list[object]]],
        include_derived: bool = True,
        risk_trigger_steps: int = 1,
        intervention_cooldown_steps: int = 0,
        empty_fallback: str = "least_load_proxy",
    ) -> None:
        self.num_servers = int(num_servers)
        self.risk_rules = list(risk_rules)
        self.allow_rules_by_action = {int(k): list(v) for k, v in dict(allow_rules_by_action).items()}
        self.include_derived = bool(include_derived)
        self.risk_trigger_steps = int(max(1, int(risk_trigger_steps)))
        self.intervention_cooldown_steps = int(max(0, int(intervention_cooldown_steps)))
        self.empty_fallback = str(empty_fallback)
        if self.empty_fallback not in {"least_load_proxy", "keep_rl", "open_all"}:
            raise ValueError("empty_fallback must be one of: least_load_proxy, keep_rl, open_all")
        self.reset_runtime_state()

    def reset_runtime_state(self) -> None:
        self._risk_streak = 0
        self._cooldown_remaining = 0

    def snapshot_runtime_state(self) -> dict[str, int]:
        return {
            "risk_streak": int(self._risk_streak),
            "cooldown_remaining": int(self._cooldown_remaining),
        }

    def restore_runtime_state(self, snapshot: dict[str, int] | None) -> None:
        if snapshot is None:
            self.reset_runtime_state()
            return
        self._risk_streak = int(max(0, int(snapshot.get("risk_streak", 0))))
        self._cooldown_remaining = int(max(0, int(snapshot.get("cooldown_remaining", 0))))

    def feature_values(self, obs: np.ndarray) -> dict[str, float | int]:
        return features_from_obs(
            np.asarray(obs, dtype=np.float64),
            num_servers=int(self.num_servers),
            include_derived=bool(self.include_derived),
        )

    def risk_and_mask_from_obs(self, obs: np.ndarray) -> tuple[bool, np.ndarray, dict[str, float | int]]:
        values = self.feature_values(obs)
        risky = bool(_any_rule_satisfied(self.risk_rules, values=values))
        allowed = np.zeros((int(self.num_servers),), dtype=np.bool_)
        matched_allow = 0
        for action in range(int(self.num_servers)):
            rules = self.allow_rules_by_action.get(int(action), [])
            ok = bool(_any_rule_satisfied(list(rules), values=values))
            allowed[int(action)] = bool(ok)
            matched_allow += int(ok)
        stats: dict[str, float | int] = {
            "shield/risky_state": int(bool(risky)),
            "shield/allowed_count": int(np.sum(allowed)),
            "shield/matched_allow_count": int(matched_allow),
        }
        for action in range(int(self.num_servers)):
            stats[f"shield/allow_action_{action}"] = int(bool(allowed[int(action)]))
        return bool(risky), allowed, stats

    def select_action(
        self,
        *,
        obs: np.ndarray,
        scores: np.ndarray,
        base_action: int | None = None,
        action_selection: str = "argmax",
        rng: np.random.Generator | None = None,
    ) -> tuple[int, dict[str, float | int]]:
        start_ns = time.perf_counter_ns()
        score_arr = np.asarray(scores, dtype=np.float64).reshape(-1)
        if score_arr.size != int(self.num_servers):
            raise ValueError(f"expected {int(self.num_servers)} action scores, got {score_arr.size}")
        if base_action is None:
            if str(action_selection) == "sample":
                rng = np.random.default_rng() if rng is None else rng
                base_action = _sample_action(score_arr, rng=rng)
            else:
                base_action = _argmax_action(score_arr)
        base_action = int(base_action)

        risky, allowed, stats = self.risk_and_mask_from_obs(obs)
        if bool(risky):
            self._risk_streak = int(self._risk_streak) + 1
        else:
            self._risk_streak = 0

        cooldown_active = bool(int(self._cooldown_remaining) > 0)
        can_intervene = bool(
            risky and int(self._risk_streak) >= int(self.risk_trigger_steps) and not bool(cooldown_active)
        )
        empty_allowed = bool(int(np.sum(allowed)) == 0)
        rl_allowed = bool(0 <= int(base_action) < int(self.num_servers) and bool(allowed[int(base_action)]))
        fallback_used = False
        chosen = int(base_action)

        if bool(can_intervene):
            if bool(empty_allowed):
                fallback_used = True
                if self.empty_fallback == "least_load_proxy":
                    obs_arr = np.asarray(obs, dtype=np.float64).reshape(-1)
                    chosen = int(np.argmin(obs_arr[: int(self.num_servers)]))
                elif self.empty_fallback == "open_all":
                    chosen = _argmax_action(score_arr)
                else:
                    chosen = int(base_action)
            elif not bool(rl_allowed):
                if str(action_selection) == "sample":
                    rng = np.random.default_rng() if rng is None else rng
                    chosen = _sample_action(score_arr, rng=rng, mask=allowed)
                else:
                    chosen = _argmax_action(score_arr, mask=allowed)
                self._cooldown_remaining = int(self.intervention_cooldown_steps)
            else:
                chosen = int(base_action)
        elif bool(cooldown_active):
            self._cooldown_remaining = int(max(0, int(self._cooldown_remaining) - 1))

        intervention = int(bool(can_intervene and int(chosen) != int(base_action)))
        elapsed_ms = float(time.perf_counter_ns() - int(start_ns)) / 1_000_000.0
        stats.update(
            {
                "shield/base_action": int(base_action),
                "shield/selected_action": int(chosen),
                "shield/action_changed": int(int(chosen) != int(base_action)),
                "shield/intervene": int(intervention),
                "shield/rl_action_allowed": int(bool(rl_allowed)),
                "shield/empty_allowed": int(bool(empty_allowed)),
                "shield/fallback_used": int(bool(fallback_used)),
                "shield/risk_streak": int(self._risk_streak),
                "shield/cooldown_active": int(bool(cooldown_active)),
                "shield/cooldown_remaining": int(self._cooldown_remaining),
                "shield/risk_trigger_steps": int(self.risk_trigger_steps),
                "shield/intervention_cooldown_steps": int(self.intervention_cooldown_steps),
                "shield/time_ms": float(elapsed_ms),
            }
        )
        return int(chosen), stats


def load_shield_from_prefix(
    prefix: str | os.PathLike[str],
    *,
    num_servers: int | None = None,
    risk_trigger_steps: int = 1,
    intervention_cooldown_steps: int = 0,
    empty_fallback: str = "least_load_proxy",
) -> ObsOnlyShield:
    base = Path(prefix)
    manifest_path = Path(str(prefix) + "_manifest.json")
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    n = int(num_servers if num_servers is not None else manifest.get("num_servers", 10))
    include_derived = bool(manifest.get("include_derived", True))
    _ = feature_columns(int(n), include_derived=bool(include_derived))

    risk_path = Path(str(base) + "_risk.pl")
    if not risk_path.is_file():
        raise FileNotFoundError(f"shield risk rules not found: {risk_path}")
    risk_rules = load_rules(str(risk_path), output_var="Risk", positive_value=1)
    allow_rules_by_action: dict[int, list[list[object]]] = {}
    for action in range(int(n)):
        action_path = Path(str(base) + f"_allow_server{int(action)}.pl")
        if not action_path.is_file():
            raise FileNotFoundError(f"shield action rules not found: {action_path}")
        allow_rules_by_action[int(action)] = load_rules(
            str(action_path),
            output_var=f"AllowServer{int(action)}",
            positive_value=1,
        )
    return ObsOnlyShield(
        num_servers=int(n),
        risk_rules=list(risk_rules),
        allow_rules_by_action=allow_rules_by_action,
        include_derived=bool(include_derived),
        risk_trigger_steps=int(risk_trigger_steps),
        intervention_cooldown_steps=int(intervention_cooldown_steps),
        empty_fallback=str(empty_fallback),
    )
