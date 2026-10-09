from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class Variable:
    name: str


@dataclass(frozen=True)
class Cmp:
    op: str
    left: Any
    right: Any


@dataclass(frozen=True)
class ImpliesExpr:
    antecedent: Any
    consequent: Any


@dataclass(frozen=True)
class OrExpr:
    args: tuple[Any, ...]


@dataclass(frozen=True)
class PolicyAtom:
    variable: str
    op: str
    value: float


@dataclass(frozen=True)
class PolicyOr:
    atoms: tuple[PolicyAtom, ...]


@dataclass(frozen=True)
class PolicyRule:
    bitrate_kbps: int
    atoms: tuple[PolicyAtom | PolicyOr, ...]
    text: str


@dataclass(frozen=True)
class DelayPerturbationCandidate:
    delay_ms_hist: tuple[float, ...]
    intervals: Mapping[int, tuple[float, float]]
    rule: PolicyRule
    l1_ms: float


def _flip_op(op: str) -> str:
    if op == "<":
        return ">"
    if op == "<=":
        return ">="
    if op == ">":
        return "<"
    if op == ">=":
        return "<="
    return op


def _flatten_and(expr: Any) -> list[Any]:
    if isinstance(expr, list):
        out: list[Any] = []
        for item in expr:
            out.extend(_flatten_and(item))
        return out
    return [expr]


def _cmp_to_atom(expr: Any) -> PolicyAtom | None:
    if not isinstance(expr, Cmp):
        return None

    left = expr.left
    right = expr.right

    if isinstance(left, Variable) and isinstance(right, (int, float)):
        return PolicyAtom(variable=str(left.name), op=str(expr.op), value=float(right))
    if isinstance(right, Variable) and isinstance(left, (int, float)):
        return PolicyAtom(variable=str(right.name), op=_flip_op(str(expr.op)), value=float(left))
    return None


def _parse_indexed(var_name: str, prefix: str) -> int | None:
    if not str(var_name).startswith(prefix):
        return None
    suffix = str(var_name)[len(prefix) :]
    return int(suffix) if suffix.isdigit() else None


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


def _tighten_interval(
    interval: tuple[float, float],
    *,
    op: str,
    value: float,
    eps: float,
) -> tuple[float, float] | None:
    lb, ub = float(interval[0]), float(interval[1])
    v = float(value)

    if not math.isfinite(v):
        return None

    if op == "<":
        ub = min(ub, v - float(eps))
    elif op == "<=":
        ub = min(ub, v)
    elif op == ">":
        lb = max(lb, v + float(eps))
    elif op == ">=":
        lb = max(lb, v)
    elif op == "==":
        lb = max(lb, v)
        ub = min(ub, v)
    else:
        return None

    if lb > ub:
        return None
    return float(lb), float(ub)


def _throughput_constraint_to_delay_interval(
    *,
    op: str,
    throughput_mbps: float,
    chosen_chunk_bytes: int,
    eps: float,
) -> tuple[str, float] | None:
    c = float(throughput_mbps)
    if not math.isfinite(c):
        return None

    if c <= 0.0:
        if op in (">", ">="):
            return None
        return None

    threshold_ms = float(chosen_chunk_bytes) * 8.0 / (1000.0 * c)
    if not math.isfinite(threshold_ms):
        return None

    if op == ">":
        return "<", float(threshold_ms)
    if op == ">=":
        return "<=", float(threshold_ms)
    if op == "<":
        return ">", float(threshold_ms)
    if op == "<=":
        return ">=", float(threshold_ms)
    if op == "==":
        return "==", float(threshold_ms)
    return None


_POLICY_RULES_CACHE: dict[tuple[str, float], dict[int, tuple[PolicyRule, ...]]] = {}


def load_policy_rules(rules_file: str) -> dict[int, tuple[PolicyRule, ...]]:
    path = os.path.abspath(str(rules_file))
    mtime = 0.0
    try:
        mtime = float(os.path.getmtime(path))
    except Exception:
        mtime = 0.0

    cache_key = (path, mtime)
    if cache_key in _POLICY_RULES_CACHE:
        return _POLICY_RULES_CACHE[cache_key]

    #* Custom eval env: parse SymPy-like expressions from rule extraction as structured objects.
    variable_cache: dict[str, Variable] = {}

    def Symbol(name: str, *args, **kwargs):
        if name not in variable_cache:
            variable_cache[name] = Variable(str(name))
        return variable_cache[name]

    def Integer(v):
        return int(v)

    def Float(v, precision=None):
        return float(v)

    def And(*args):
        return list(args)

    def Or(*args):
        return OrExpr(tuple(args))

    def Implies(a, b):
        return ImpliesExpr(a, b)

    def Equality(a, b):
        return Cmp("==", a, b)

    def Unequality(a, b):
        return Cmp("!=", a, b)

    def StrictLessThan(a, b):
        return Cmp("<", a, b)

    def StrictGreaterThan(a, b):
        return Cmp(">", a, b)

    def LessThan(a, b):
        return Cmp("<=", a, b)

    def GreaterThan(a, b):
        return Cmp(">=", a, b)

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

    grouped: dict[int, list[PolicyRule]] = {}
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            expr = eval(line, eval_env)  # nosec - trusted local rules file
            if not isinstance(expr, ImpliesExpr):
                continue

            consequent_atom = _cmp_to_atom(expr.consequent)
            if consequent_atom is None or consequent_atom.variable != "BitrateKbps" or consequent_atom.op != "==":
                continue
            try:
                bitrate_kbps = int(consequent_atom.value)
            except Exception:
                continue

            atoms: list[PolicyAtom | PolicyOr] = []
            ok = True
            for atom_expr in _flatten_and(expr.antecedent):
                if isinstance(atom_expr, OrExpr):
                    or_atoms: list[PolicyAtom] = []
                    for sub in atom_expr.args:
                        atom = _cmp_to_atom(sub)
                        if atom is None:
                            ok = False
                            break
                        if atom.op not in ("<", "<=", ">", ">=", "==", "!="):
                            ok = False
                            break
                        or_atoms.append(atom)
                    if not ok:
                        break
                    if len(or_atoms) == 0:
                        ok = False
                        break
                    atoms.append(PolicyOr(atoms=tuple(or_atoms)))
                    continue

                atom = _cmp_to_atom(atom_expr)
                if atom is None:
                    ok = False
                    break
                if atom.op not in ("<", "<=", ">", ">=", "==", "!="):
                    ok = False
                    break
                atoms.append(atom)
            if not ok:
                continue

            grouped.setdefault(int(bitrate_kbps), []).append(
                PolicyRule(bitrate_kbps=int(bitrate_kbps), atoms=tuple(atoms), text=str(line))
            )

    grouped_final: dict[int, tuple[PolicyRule, ...]] = {k: tuple(v) for k, v in grouped.items()}
    #* Keep cached parses for other files; drop stale entries for this path.
    for k in list(_POLICY_RULES_CACHE.keys()):
        if k[0] == path and k != cache_key:
            _POLICY_RULES_CACHE.pop(k, None)
    _POLICY_RULES_CACHE[cache_key] = grouped_final
    return grouped_final


def enumerate_delay_perturbation_candidates(
    *,
    rules_by_bitrate: Mapping[int, Sequence[PolicyRule]],
    target_bitrate_kbps: int,
    values: Mapping[str, float],
    delay_ms_hist: Sequence[float],
    throughput_mbps_hist: Sequence[float],
    chosen_chunk_bytes_hist: Sequence[int],
    allow_delay_idx: Sequence[int],
    eps_delay_ms: float,
    rng: random.Random | None = None,
    max_rules: int | None = None,
) -> list[DelayPerturbationCandidate]:
    allow = {int(i) for i in allow_delay_idx}
    if len(allow) == 0:
        return []

    cur_delay = [float(x) for x in delay_ms_hist]
    cur_thr = [float(x) for x in throughput_mbps_hist]
    rules = list(rules_by_bitrate.get(int(target_bitrate_kbps), []))
    if len(rules) == 0:
        return []

    if max_rules is not None and int(max_rules) > 0 and len(rules) > int(max_rules):
        if rng is None:
            rules = list(rules[: int(max_rules)])
        else:
            rules = rng.sample(rules, int(max_rules))

    out: list[DelayPerturbationCandidate] = []

    for rule in rules:
        intervals: dict[int, tuple[float, float]] = {i: (-float("inf"), float("inf")) for i in allow}
        feasible = True

        for atom in rule.atoms:
            if isinstance(atom, PolicyOr):
                #* H2O student rules can contain `Or(...)` clauses (typically over categorical features like
                #  `LastBitrateKbps`). We currently support `Or` only when it does not constrain perturbable
                #  delay/throughput history coordinates.
                satisfied = False
                for sub in atom.atoms:
                    var = str(sub.variable)
                    op = str(sub.op)
                    val = float(sub.value)

                    if _parse_indexed(var, "DelayMs") is not None or _parse_indexed(var, "ThroughputMbps") is not None:
                        continue
                    if var not in values:
                        continue
                    if _cmp(float(values[var]), op, val):
                        satisfied = True
                        break
                if not satisfied:
                    feasible = False
                    break
                continue

            var = str(atom.variable)
            op = str(atom.op)
            val = float(atom.value)

            idx_delay = _parse_indexed(var, "DelayMs")
            if idx_delay is not None:
                if int(idx_delay) in allow:
                    if op == "!=":
                        feasible = False
                        break
                    tightened = _tighten_interval(intervals[int(idx_delay)], op=op, value=val, eps=float(eps_delay_ms))
                    if tightened is None:
                        feasible = False
                        break
                    intervals[int(idx_delay)] = tightened
                else:
                    if int(idx_delay) < 0 or int(idx_delay) >= len(cur_delay):
                        feasible = False
                        break
                    if not _cmp(float(cur_delay[int(idx_delay)]), op, val):
                        feasible = False
                        break
                continue

            idx_thr = _parse_indexed(var, "ThroughputMbps")
            if idx_thr is not None:
                if int(idx_thr) < 0 or int(idx_thr) >= len(cur_thr):
                    feasible = False
                    break
                if int(idx_thr) in allow:
                    if op == "!=":
                        feasible = False
                        break
                    derived = _throughput_constraint_to_delay_interval(
                        op=op,
                        throughput_mbps=float(val),
                        chosen_chunk_bytes=int(chosen_chunk_bytes_hist[int(idx_thr)]),
                        eps=float(eps_delay_ms),
                    )
                    if derived is None:
                        feasible = False
                        break
                    delay_op, delay_val = derived
                    tightened = _tighten_interval(
                        intervals[int(idx_thr)], op=str(delay_op), value=float(delay_val), eps=float(eps_delay_ms)
                    )
                    if tightened is None:
                        feasible = False
                        break
                    intervals[int(idx_thr)] = tightened
                else:
                    if not _cmp(float(cur_thr[int(idx_thr)]), op, val):
                        feasible = False
                        break
                continue

            if var not in values:
                feasible = False
                break
            if not _cmp(float(values[var]), op, val):
                feasible = False
                break

        if not feasible:
            continue

        cand_delay = list(cur_delay)
        l1 = 0.0
        for idx in sorted(allow):
            interval = intervals.get(int(idx))
            if interval is None:
                feasible = False
                break
            lb, ub = float(interval[0]), float(interval[1])
            if lb > ub:
                feasible = False
                break
            cur = float(cur_delay[int(idx)])
            if cur < lb:
                new = float(lb)
            elif cur > ub:
                new = float(ub)
            else:
                new = float(cur)
            cand_delay[int(idx)] = float(new)
            l1 += abs(float(new) - float(cur))

        if not feasible or not math.isfinite(float(l1)):
            continue

        out.append(
            DelayPerturbationCandidate(
                delay_ms_hist=tuple(float(x) for x in cand_delay),
                intervals=dict(intervals),
                rule=PolicyRule(bitrate_kbps=int(rule.bitrate_kbps), atoms=rule.atoms, text=rule.text),
                l1_ms=float(l1),
            )
        )

    return out


def find_minimal_delay_perturbation(
    *,
    rules_by_bitrate: Mapping[int, Sequence[PolicyRule]],
    target_bitrate_kbps: int,
    values: Mapping[str, float],
    delay_ms_hist: Sequence[float],
    throughput_mbps_hist: Sequence[float],
    chosen_chunk_bytes_hist: Sequence[int],
    allow_delay_idx: Sequence[int],
    eps_delay_ms: float,
    rng: random.Random,
) -> tuple[list[float], PolicyRule, float] | None:
    candidates = enumerate_delay_perturbation_candidates(
        rules_by_bitrate=rules_by_bitrate,
        target_bitrate_kbps=int(target_bitrate_kbps),
        values=values,
        delay_ms_hist=delay_ms_hist,
        throughput_mbps_hist=throughput_mbps_hist,
        chosen_chunk_bytes_hist=chosen_chunk_bytes_hist,
        allow_delay_idx=allow_delay_idx,
        eps_delay_ms=float(eps_delay_ms),
        rng=rng,
        max_rules=None,
    )
    if len(candidates) == 0:
        return None

    best_l1 = min(float(c.l1_ms) for c in candidates)
    best = [
        c
        for c in candidates
        if abs(float(c.l1_ms) - float(best_l1)) <= 1e-9 * max(1.0, abs(float(best_l1)))
    ]
    chosen = rng.choice(best)
    return list(float(x) for x in chosen.delay_ms_hist), chosen.rule, float(chosen.l1_ms)
