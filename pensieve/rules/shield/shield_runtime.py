from __future__ import annotations

import math
import os
from typing import Any

import numpy as np

#* Pensieve constants (kept local to avoid importing `Pensieve-PPO/src/inference.py`).
VIDEO_BIT_RATE: list[int] = [300, 750, 1200, 1850, 2850, 4300]
A_DIM = 6
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0


class _ShieldVar:
    def __init__(self, name: str):
        self.name = str(name)


class _ShieldCmp:
    def __init__(self, op: str, left: object, right: object):
        self.op = str(op)
        self.left = left
        self.right = right


class _ShieldImplies:
    def __init__(self, antecedent: object, consequent: object):
        self.antecedent = antecedent
        self.consequent = consequent


class _ShieldOr:
    def __init__(self, args: tuple[object, ...]):
        self.args = tuple(args)


_SHIELD_RULES_CACHE: dict[tuple[str, float, str], list[tuple[list[object], int]]] = {}


def _shield_flip_op(op: str) -> str:
    if op == "<":
        return ">"
    if op == "<=":
        return ">="
    if op == ">":
        return "<"
    if op == ">=":
        return "<="
    return str(op)


def _shield_flatten_and(expr: object) -> list[object]:
    if isinstance(expr, list):
        out: list[object] = []
        for item in expr:
            out.extend(_shield_flatten_and(item))
        return out
    return [expr]


def _shield_cmp_to_atom(expr: object) -> tuple[str, str, float] | None:
    if not isinstance(expr, _ShieldCmp):
        return None
    left = expr.left
    right = expr.right
    if isinstance(left, _ShieldVar) and isinstance(right, (int, float)):
        return str(left.name), str(expr.op), float(right)
    if isinstance(right, _ShieldVar) and isinstance(left, (int, float)):
        return str(right.name), _shield_flip_op(str(expr.op)), float(left)
    return None


def _shield_parse_indexed(var_name: str, prefix: str) -> int | None:
    if not str(var_name).startswith(str(prefix)):
        return None
    suffix = str(var_name)[len(str(prefix)) :]
    return int(suffix) if suffix.isdigit() else None


def _shield_cmp(a: float, op: str, b: float) -> bool:
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


def load_rules(rules_file: str, output_var: str) -> list[tuple[list[object], int]]:
    path = os.path.abspath(str(rules_file))
    mtime = 0.0
    try:
        mtime = float(os.path.getmtime(path))
    except Exception:
        mtime = 0.0

    cache_key = (path, mtime, str(output_var))
    if cache_key in _SHIELD_RULES_CACHE:
        return _SHIELD_RULES_CACHE[cache_key]

    variable_cache: dict[str, _ShieldVar] = {}

    def Symbol(name: str, *args, **kwargs):
        if str(name) not in variable_cache:
            variable_cache[str(name)] = _ShieldVar(str(name))
        return variable_cache[str(name)]

    def Integer(v):
        return int(v)

    def Float(v, precision=None):
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

    rules: list[tuple[list[object], int]] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            expr = eval(line, eval_env)  # nosec - trusted local rules file
            if not isinstance(expr, _ShieldImplies):
                continue

            consequent_atom = _shield_cmp_to_atom(expr.consequent)
            if consequent_atom is None:
                continue
            cons_var, cons_op, cons_val = consequent_atom
            if str(cons_var) != str(output_var) or str(cons_op) != "==":
                continue

            try:
                out_kbps = int(cons_val)
            except Exception:
                continue

            atoms: list[object] = []
            ok = True
            for atom_expr in _shield_flatten_and(expr.antecedent):
                if isinstance(atom_expr, _ShieldOr):
                    or_atoms: list[tuple[str, str, float]] = []
                    for sub in atom_expr.args:
                        atom = _shield_cmp_to_atom(sub)
                        if atom is None:
                            ok = False
                            break
                        v, op, val = atom
                        if str(op) not in ("<", "<=", ">", ">=", "==", "!="):
                            ok = False
                            break
                        or_atoms.append((str(v), str(op), float(val)))
                    if not ok or len(or_atoms) == 0:
                        ok = False
                        break
                    atoms.append(("or", or_atoms))
                    continue

                atom = _shield_cmp_to_atom(atom_expr)
                if atom is None:
                    ok = False
                    break
                v, op, val = atom
                if str(op) not in ("<", "<=", ">", ">=", "==", "!="):
                    ok = False
                    break
                atoms.append((str(v), str(op), float(val)))
            if not ok:
                continue

            rules.append((atoms, int(out_kbps)))

    _SHIELD_RULES_CACHE[cache_key] = rules
    return rules


def _bitrate_kbps_to_action_idx(bitrate_kbps: float) -> int:
    bitrate_values = np.array(VIDEO_BIT_RATE, dtype=np.float64)
    if bitrate_values.size == 0:
        return 0
    target = float(bitrate_kbps)
    if not np.isfinite(target):
        return 0
    return int(np.argmin(np.abs(bitrate_values - float(target))))


def _policy_action_idx(
    *,
    p: np.ndarray,
    action_selection: str,
    rng: np.random.RandomState,
    lo: int = 0,
    hi: int | None = None,
    noise: np.ndarray | None = None,
) -> tuple[int, np.ndarray | None]:
    upper = int(A_DIM) - 1 if hi is None else int(hi)
    lower = int(max(0, min(int(A_DIM) - 1, int(lo))))
    upper = int(max(0, min(int(A_DIM) - 1, int(upper))))
    if lower > upper:
        lower, upper = 0, int(A_DIM) - 1

    probs = np.asarray(p, dtype=np.float64).reshape(-1)
    if str(action_selection) == "argmax":
        chosen = int(lower + int(np.argmax(np.array(probs[lower : upper + 1], copy=False))))
        return int(chosen), noise

    if str(action_selection) == "gumbel":
        gumbel_noise = noise
        if gumbel_noise is None:
            gumbel_noise = rng.gumbel(size=probs.shape)
        logits = np.log(probs + 1e-12) + np.asarray(gumbel_noise, dtype=np.float64)
        masked = np.full_like(logits, -np.inf, dtype=np.float64)
        masked[lower : upper + 1] = logits[lower : upper + 1]
        if not np.any(np.isfinite(masked[lower : upper + 1])):
            return int(lower), np.asarray(gumbel_noise, dtype=np.float64)
        return int(np.argmax(masked)), np.asarray(gumbel_noise, dtype=np.float64)

    raise ValueError(f"Unknown action_selection: {action_selection}")


def _shield_atom_satisfied(
    var: str,
    op: str,
    val: float,
    *,
    values: dict[str, float],
    delay_ms_hist: list[float],
    throughput_mbps_hist: list[float],
) -> bool:
    if str(var) in values:
        return _shield_cmp(float(values[str(var)]), str(op), float(val))

    idx_delay = _shield_parse_indexed(str(var), "DelayMs")
    if idx_delay is not None:
        if 0 <= int(idx_delay) < len(delay_ms_hist):
            return _shield_cmp(float(delay_ms_hist[int(idx_delay)]), str(op), float(val))
        return False

    idx_thr = _shield_parse_indexed(str(var), "ThroughputMbps")
    if idx_thr is not None:
        if 0 <= int(idx_thr) < len(throughput_mbps_hist):
            return _shield_cmp(float(throughput_mbps_hist[int(idx_thr)]), str(op), float(val))
        return False

    return False


def _shield_rule_satisfied(
    atoms: list[object],
    *,
    values: dict[str, float],
    delay_ms_hist: list[float],
    throughput_mbps_hist: list[float],
) -> bool:
    for atom in atoms:
        if isinstance(atom, tuple) and len(atom) == 2 and str(atom[0]) == "or":
            ok_any = False
            for v, op, val in list(atom[1]):
                if _shield_atom_satisfied(
                    str(v),
                    str(op),
                    float(val),
                    values=values,
                    delay_ms_hist=delay_ms_hist,
                    throughput_mbps_hist=throughput_mbps_hist,
                ):
                    ok_any = True
                    break
            if not ok_any:
                return False
            continue

        if not (isinstance(atom, tuple) and len(atom) == 3):
            return False
        v, op, val = atom
        if not _shield_atom_satisfied(
            str(v),
            str(op),
            float(val),
            values=values,
            delay_ms_hist=delay_ms_hist,
            throughput_mbps_hist=throughput_mbps_hist,
        ):
            return False
    return True


def policy_rule_inputs_from_state(
    state: np.ndarray,
    *,
    last_bitrate_idx: int,
    video_chunk_size_bytes: int,
) -> tuple[dict[str, float], list[float], list[float], list[int]]:
    history_len = int(state.shape[1])

    delay_ms_hist = [float(state[3, i]) * float(M_IN_K) * float(BUFFER_NORM_FACTOR) for i in range(history_len)]
    throughput_mbps_hist = [float(state[2, i]) * 8.0 for i in range(history_len)]

    chosen_chunk_bytes_hist: list[int] = []
    for i in range(history_len):
        chunk_bytes = float(state[2, i]) * float(delay_ms_hist[i]) * float(M_IN_K)
        if not np.isfinite(chunk_bytes):
            chunk_bytes = 0.0
        chosen_chunk_bytes_hist.append(int(max(1.0, round(chunk_bytes))))
    if history_len > 0:
        chosen_chunk_bytes_hist[-1] = int(video_chunk_size_bytes)

    values: dict[str, float] = {
        "BufferSeconds": float(state[1, -1]) * float(BUFFER_NORM_FACTOR),
        "LastBitrateKbps": float(VIDEO_BIT_RATE[int(last_bitrate_idx)]),
        "ChunksRemaining": float(state[5, -1]) * float(CHUNK_TIL_VIDEO_END_CAP),
    }
    for j in range(int(A_DIM)):
        values[f"NextChunkBytes{j}"] = float(state[4, j]) * float(M_IN_K) * float(M_IN_K)

    return values, delay_ms_hist, throughput_mbps_hist, chosen_chunk_bytes_hist


class ShieldBounds:
    def __init__(
        self,
        min_rules: list[tuple[list[object], int]],
        max_rules: list[tuple[list[object], int]],
        *,
        risk_trigger_steps: int = 1,
        intervention_cooldown_steps: int = 0,
    ):
        self._min_rules = list(min_rules)
        self._max_rules = list(max_rules)
        self._risk_trigger_steps = int(max(1, int(risk_trigger_steps)))
        self._intervention_cooldown_steps = int(max(0, int(intervention_cooldown_steps)))
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

    def bounds_from_state(
        self,
        *,
        state: np.ndarray,
        last_bitrate_idx: int,
        video_chunk_size_bytes: int,
    ) -> tuple[int, int, dict[str, float | int]]:
        values, delay_ms_hist, throughput_mbps_hist, _chosen_chunk_bytes_hist = policy_rule_inputs_from_state(
            state,
            last_bitrate_idx=int(last_bitrate_idx),
            video_chunk_size_bytes=int(video_chunk_size_bytes),
        )

        min_kbps = float(min(VIDEO_BIT_RATE)) if VIDEO_BIT_RATE else 0.0
        max_kbps = float(max(VIDEO_BIT_RATE)) if VIDEO_BIT_RATE else 0.0
        matched_min = 0
        matched_max = 0

        for atoms, out_kbps in self._min_rules:
            if _shield_rule_satisfied(
                atoms,
                values=values,
                delay_ms_hist=delay_ms_hist,
                throughput_mbps_hist=throughput_mbps_hist,
            ):
                matched_min += 1
                min_kbps = max(float(min_kbps), float(out_kbps))

        for atoms, out_kbps in self._max_rules:
            if _shield_rule_satisfied(
                atoms,
                values=values,
                delay_ms_hist=delay_ms_hist,
                throughput_mbps_hist=throughput_mbps_hist,
            ):
                matched_max += 1
                max_kbps = min(float(max_kbps), float(out_kbps))

        min_idx = _bitrate_kbps_to_action_idx(float(min_kbps))
        max_idx = _bitrate_kbps_to_action_idx(float(max_kbps))

        inconsistent = int(min_idx > max_idx)
        if inconsistent:
            min_idx = 0
            max_idx = int(A_DIM) - 1

        stats: dict[str, float | int] = {
            "shield/matched_min": int(matched_min),
            "shield/matched_max": int(matched_max),
            "shield/min_kbps": float(min_kbps),
            "shield/max_kbps": float(max_kbps),
            "shield/min_idx": int(min_idx),
            "shield/max_idx": int(max_idx),
            "shield/inconsistent": int(inconsistent),
        }
        return int(min_idx), int(max_idx), stats

    def select_action(
        self,
        *,
        p: np.ndarray,
        state: np.ndarray,
        last_bitrate_idx: int,
        video_chunk_size_bytes: int,
        action_selection: str,
        rng: np.random.RandomState,
    ) -> tuple[int, dict[str, float | int]]:
        mi, ma, stats = self.bounds_from_state(
            state=state,
            last_bitrate_idx=int(last_bitrate_idx),
            video_chunk_size_bytes=int(video_chunk_size_bytes),
        )
        lo = int(max(0, min(int(A_DIM) - 1, int(mi))))
        hi = int(max(0, min(int(A_DIM) - 1, int(ma))))
        if lo > hi:
            lo, hi = 0, int(A_DIM) - 1

        base_action_idx, sampled_noise = _policy_action_idx(
            p=p,
            action_selection=str(action_selection),
            rng=rng,
        )
        risky = bool(int(base_action_idx) < int(lo) or int(base_action_idx) > int(hi))
        if bool(risky):
            self._risk_streak = int(self._risk_streak) + 1
        else:
            self._risk_streak = 0

        cooldown_active = bool(int(self._cooldown_remaining) > 0)
        intervene = bool(risky and int(self._risk_streak) >= int(self._risk_trigger_steps) and not cooldown_active)
        if bool(intervene):
            chosen, _ = _policy_action_idx(
                p=p,
                action_selection=str(action_selection),
                rng=rng,
                lo=int(lo),
                hi=int(hi),
                noise=sampled_noise,
            )
            self._cooldown_remaining = int(self._intervention_cooldown_steps)
        else:
            chosen = int(base_action_idx)
            if cooldown_active:
                self._cooldown_remaining = int(max(0, int(self._cooldown_remaining) - 1))

        stats["shield/base_action_idx"] = int(base_action_idx)
        stats["shield/risky_state"] = int(bool(risky))
        stats["shield/risk_streak"] = int(self._risk_streak)
        stats["shield/intervene"] = int(bool(intervene))
        stats["shield/cooldown_active"] = int(bool(cooldown_active))
        stats["shield/cooldown_remaining"] = int(self._cooldown_remaining)
        stats["shield/risk_trigger_steps"] = int(self._risk_trigger_steps)
        stats["shield/intervention_cooldown_steps"] = int(self._intervention_cooldown_steps)
        stats["shield/selected_action_idx"] = int(chosen)
        return int(chosen), stats
