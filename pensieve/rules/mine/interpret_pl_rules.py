#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import Any


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _resolve_path(repo_relative_or_abs: str) -> str:
    expanded = os.path.expandvars(os.path.expanduser(str(repo_relative_or_abs)))
    if os.path.isabs(expanded):
        return expanded
    return os.path.join(_repo_root(), expanded)


def _is_percentile_col(name: str) -> bool:
    if not str(name).startswith("p"):
        return False
    suffix = str(name)[1:]
    return bool(suffix.isdigit())


def _percentile_key(pcol: str) -> int:
    return int(str(pcol)[1:])


_CROSS_FEATURE_DECOMPOSITIONS: dict[str, tuple[str, str]] = {
    #* Derived feature -> (lhs_var, rhs_var) meaning: derived = lhs_var - rhs_var
    "policy_minus_bitrate_hist_kbps_max": ("policy_action_kbps", "bitrate_hist_kbps_max"),
    "oracle_minus_bitrate_hist_kbps_max": ("oracle_action_kbps", "bitrate_hist_kbps_max"),
    "policy_minus_last_action_kbps": ("policy_action_kbps", "last_action_kbps"),
}


@dataclass(frozen=True)
class _ThresholdIndex:
    by_feature: dict[str, dict[str, float]]
    percentile_cols: list[str]

    def match_percentile(
        self,
        *,
        feature: str,
        value: float,
        atol: float,
        rtol: float,
    ) -> str | None:
        if str(feature) not in self.by_feature:
            return None

        best_p: str | None = None
        best_diff = float("inf")
        candidates: list[tuple[int, str]] = []
        for pcol, thr in self.by_feature[str(feature)].items():
            if not math.isfinite(float(thr)):
                continue
            diff = abs(float(thr) - float(value))
            tol = max(float(atol), float(rtol) * max(1.0, abs(float(thr)), abs(float(value))))
            if diff <= tol:
                candidates.append((_percentile_key(str(pcol)), str(pcol)))
                if diff < best_diff:
                    best_diff = diff
                    best_p = str(pcol)

        if not candidates:
            return None

        #* Prefer the smallest percentile when multiple percentiles share the same numeric threshold.
        best = min(candidates, key=lambda t: int(t[0]))
        return str(best[1]) if best else best_p

    def semantic_constant_name(self, *, feature: str, pcol: str) -> str:
        if str(pcol) == "p0":
            return "0"

        #* Bitrate constants are discrete action levels, not percentiles.
        discrete_action_feats = {"last_action_kbps", "policy_action_kbps", "oracle_action_kbps"}
        if (str(feature).startswith("bitrate_hist_kbps") or str(feature) in discrete_action_feats) and _is_percentile_col(str(pcol)):
            v = _percentile_key(str(pcol))
            if v in {300, 750, 1200, 1850, 2850, 4300}:
                return f"bitrate_{v}_kbps"

        #* For readability, render percentile constants as bare "pNN" (without a feature prefix).
        #* Example: next_chunk_size_bytes_policy_p50 -> p50.
        if _is_percentile_col(str(pcol)):
            return str(pcol)

        return f"{feature}_{pcol}"


def _load_thresholds_csv(path: str) -> _ThresholdIndex:
    path = str(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"* Thresholds CSV not found: {path}")

    with open(path, "r") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError("* Thresholds CSV has no header row.")

        if "feature" not in list(reader.fieldnames):
            raise RuntimeError("* Thresholds CSV must include a 'feature' column.")

        percentile_cols = [str(c) for c in reader.fieldnames if _is_percentile_col(str(c))]
        if not percentile_cols:
            raise RuntimeError("* Thresholds CSV has no percentile columns (expected pNN columns like p90).")

        by_feature: dict[str, dict[str, float]] = {}
        for row in reader:
            feat = str(row["feature"])
            by_feature.setdefault(feat, {})
            for pcol in percentile_cols:
                raw = row.get(str(pcol), "")
                try:
                    v = float(raw)
                except Exception:
                    v = float("nan")
                by_feature[feat][str(pcol)] = float(v)

    percentile_cols_sorted = sorted(percentile_cols, key=_percentile_key)
    return _ThresholdIndex(by_feature=by_feature, percentile_cols=percentile_cols_sorted)


def _flatten_and(expr: Any) -> list[Any]:
    import sympy as sp

    if expr in (sp.true, True):
        return []
    if isinstance(expr, sp.And):
        return list(sp.And.make_args(expr))
    return [expr]


def _invert_rel_op(op: str) -> str:
    return {
        ">": "<",
        ">=": "<=",
        "<": ">",
        "<=": ">=",
        "==": "==",
        "!=": "!=",
    }[str(op)]


def _extract_relational_atom(expr: Any) -> tuple[str, str, float] | None:
    import sympy as sp
    from sympy.core.relational import Relational

    if not isinstance(expr, Relational):
        return None

    lhs = expr.lhs
    rhs = expr.rhs
    op = str(expr.rel_op)

    if isinstance(lhs, sp.Symbol) and isinstance(rhs, sp.Number):
        return str(lhs), str(op), float(rhs)
    if isinstance(rhs, sp.Symbol) and isinstance(lhs, sp.Number):
        return str(rhs), _invert_rel_op(str(op)), float(lhs)

    return None


@dataclass
class _Bounds:
    lower_val: float | None = None
    lower_strict: bool = False
    upper_val: float | None = None
    upper_strict: bool = False
    eq_val: float | None = None
    ne_vals: set[float] | None = None


def _simplify_conjunction(
    conjuncts: list[Any],
    *,
    tol: float,
) -> tuple[list[Any], bool]:
    import sympy as sp

    per_var: dict[str, _Bounds] = {}
    others: list[Any] = []

    for lit in conjuncts:
        atom = _extract_relational_atom(lit)
        if atom is None:
            others.append(lit)
            continue

        var, op, val = atom
        b = per_var.setdefault(str(var), _Bounds(ne_vals=set()))
        if b.ne_vals is None:
            b.ne_vals = set()

        if op in (">", ">="):
            strict = op == ">"
            if b.lower_val is None or float(val) > float(b.lower_val) + float(tol):
                b.lower_val = float(val)
                b.lower_strict = bool(strict)
            elif abs(float(val) - float(b.lower_val)) <= float(tol):
                b.lower_strict = bool(b.lower_strict or strict)
        elif op in ("<", "<="):
            strict = op == "<"
            if b.upper_val is None or float(val) < float(b.upper_val) - float(tol):
                b.upper_val = float(val)
                b.upper_strict = bool(strict)
            elif abs(float(val) - float(b.upper_val)) <= float(tol):
                b.upper_strict = bool(b.upper_strict or strict)
        elif op == "==":
            if b.eq_val is None:
                b.eq_val = float(val)
            elif abs(float(val) - float(b.eq_val)) > float(tol):
                return [], True
        elif op == "!=":
            b.ne_vals.add(float(val))
        else:
            others.append(lit)

    simplified: list[Any] = []
    for var in sorted(per_var.keys()):
        b = per_var[str(var)]
        v = sp.Symbol(str(var), real=True)

        if b.eq_val is not None:
            eqv = float(b.eq_val)
            if any(abs(eqv - float(nv)) <= float(tol) for nv in (b.ne_vals or set())):
                return [], True
            if b.lower_val is not None:
                if eqv < float(b.lower_val) - float(tol):
                    return [], True
                if abs(eqv - float(b.lower_val)) <= float(tol) and bool(b.lower_strict):
                    return [], True
            if b.upper_val is not None:
                if eqv > float(b.upper_val) + float(tol):
                    return [], True
                if abs(eqv - float(b.upper_val)) <= float(tol) and bool(b.upper_strict):
                    return [], True

            simplified.append(sp.Eq(v, sp.Float(eqv)))
            continue

        if b.lower_val is not None and b.upper_val is not None:
            lo = float(b.lower_val)
            hi = float(b.upper_val)
            if lo > hi + float(tol):
                return [], True
            if abs(lo - hi) <= float(tol):
                if bool(b.lower_strict) or bool(b.upper_strict):
                    return [], True
                simplified.append(sp.Eq(v, sp.Float(lo)))
                continue

        if b.lower_val is not None:
            lo = float(b.lower_val)
            simplified.append(sp.Gt(v, sp.Float(lo)) if bool(b.lower_strict) else sp.Ge(v, sp.Float(lo)))
        if b.upper_val is not None:
            hi = float(b.upper_val)
            simplified.append(sp.Lt(v, sp.Float(hi)) if bool(b.upper_strict) else sp.Le(v, sp.Float(hi)))
        if b.ne_vals:
            for nv in sorted(set(float(x) for x in b.ne_vals)):
                simplified.append(sp.Ne(v, sp.Float(float(nv))))

    simplified.extend(others)
    return simplified, False


def _simplify_rule(expr: Any, *, tol: float) -> Any | None:
    import sympy as sp

    if not isinstance(expr, sp.Implies):
        return expr

    antecedent, consequent = expr.args
    conjuncts = _flatten_and(antecedent)
    simplified_conjuncts, contradiction = _simplify_conjunction(conjuncts, tol=float(tol))
    if bool(contradiction):
        return None

    ant = sp.And(*simplified_conjuncts) if simplified_conjuncts else sp.true
    return sp.Implies(ant, consequent)


def _render_number(n: Any, *, feature: str | None, thr: _ThresholdIndex, atol: float, rtol: float) -> str:
    import sympy as sp

    if not isinstance(n, sp.Number):
        return str(n)

    if feature is not None:
        pcol = thr.match_percentile(feature=str(feature), value=float(n), atol=float(atol), rtol=float(rtol))
        if pcol is not None:
            return str(thr.semantic_constant_name(feature=str(feature), pcol=str(pcol)))

    if isinstance(n, sp.Integer):
        return str(int(n))
    try:
        v = float(n)
    except Exception:
        return str(n)
    if _is_int_like(v):
        return str(int(round(v)))
    return f"{v:.6g}"


def _is_int_like(x: float, *, tol: float = 1e-9) -> bool:
    if not math.isfinite(float(x)):
        return False
    return abs(float(x) - float(round(float(x)))) <= float(tol)


def _render_expr(expr: Any, *, thr: _ThresholdIndex, atol: float, rtol: float) -> str:
    import sympy as sp
    from sympy.core.relational import Relational

    if expr in (sp.true, True):
        return "⊤"
    if expr in (sp.false, False):
        return "⊥"

    if isinstance(expr, sp.Implies):
        a, b = expr.args
        return f"({ _render_expr(a, thr=thr, atol=atol, rtol=rtol) }) ⇒ ({ _render_expr(b, thr=thr, atol=atol, rtol=rtol) })"

    if isinstance(expr, sp.And):
        parts = [_render_expr(a, thr=thr, atol=atol, rtol=rtol) for a in list(sp.And.make_args(expr))]
        return " ∧ ".join(f"({p})" for p in parts) if parts else "⊤"

    if isinstance(expr, sp.Or):
        parts = [_render_expr(a, thr=thr, atol=atol, rtol=rtol) for a in list(sp.Or.make_args(expr))]
        return " ∨ ".join(f"({p})" for p in parts) if parts else "⊥"

    if isinstance(expr, sp.Not):
        return f"¬({_render_expr(expr.args[0], thr=thr, atol=atol, rtol=rtol)})"

    if isinstance(expr, Relational):
        atom = _extract_relational_atom(expr)
        if atom is not None:
            var, op, val = atom
            if str(var) in _CROSS_FEATURE_DECOMPOSITIONS:
                lhs_var, rhs_var = _CROSS_FEATURE_DECOMPOSITIONS[str(var)]
                if abs(float(val)) <= float(1e-12):
                    rhs_expr = str(rhs_var)
                else:
                    if float(val) > 0.0:
                        n = sp.Float(float(val))
                        rhs_expr = f"{rhs_var} + {_render_number(n, feature=None, thr=thr, atol=atol, rtol=rtol)}"
                    else:
                        n = sp.Float(abs(float(val)))
                        rhs_expr = f"{rhs_var} - {_render_number(n, feature=None, thr=thr, atol=atol, rtol=rtol)}"
                return _render_rel_op(str(lhs_var), str(op), str(rhs_expr))

        lhs = expr.lhs
        rhs = expr.rhs
        op = str(expr.rel_op)

        if isinstance(lhs, sp.Symbol) and isinstance(rhs, sp.Number):
            l = str(lhs)
            r = _render_number(rhs, feature=str(lhs), thr=thr, atol=atol, rtol=rtol)
            return _render_rel_op(l, op, r)
        if isinstance(rhs, sp.Symbol) and isinstance(lhs, sp.Number):
            op2 = _invert_rel_op(op)
            l = str(rhs)
            r = _render_number(lhs, feature=str(rhs), thr=thr, atol=atol, rtol=rtol)
            return _render_rel_op(l, op2, r)

        l = _render_expr(lhs, thr=thr, atol=atol, rtol=rtol)
        r = _render_expr(rhs, thr=thr, atol=atol, rtol=rtol)
        return _render_rel_op(l, op, r)

    if isinstance(expr, sp.Symbol):
        return str(expr)
    if isinstance(expr, sp.Number):
        return _render_number(expr, feature=None, thr=thr, atol=atol, rtol=rtol)

    return str(expr)


def _render_rel_op(lhs: str, op: str, rhs: str) -> str:
    sym = {
        ">": ">",
        ">=": "≥",
        "<": "<",
        "<=": "≤",
        "==": "=",
        "!=": "≠",
    }.get(str(op), str(op))
    return f"{lhs} {sym} {rhs}"


def _parse_pl_rules(path: str) -> list[Any]:
    import sympy as sp

    path = str(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"* Rules .pl not found: {path}")

    rules: list[Any] = []
    with open(path, "r") as f:
        for ln, raw in enumerate(f, start=1):
            line = str(raw).strip()
            if not line:
                continue
            try:
                rules.append(sp.sympify(line))
            except Exception as e:
                raise RuntimeError(f"* Failed to parse .pl line {ln}: {path}") from e
    return rules


def _iter_pl_files(rules_dir: str) -> list[str]:
    rules_dir = str(rules_dir)
    if not os.path.isdir(rules_dir):
        raise NotADirectoryError(f"* --rules-dir is not a directory: {rules_dir}")

    out: list[str] = []
    for root, _dirs, files in os.walk(rules_dir):
        for fn in files:
            if str(fn).endswith(".pl"):
                out.append(os.path.join(root, fn))
    return sorted(out)


def _interpret_one(
    *,
    rules_pl: str,
    thr: _ThresholdIndex,
    out_clj: str,
    simplify_tol: float,
    const_atol: float,
    const_rtol: float,
) -> tuple[int, int]:
    rules = _parse_pl_rules(rules_pl)

    interpreted: list[str] = []
    dropped = 0
    for r in rules:
        simplified = _simplify_rule(r, tol=float(simplify_tol))
        if simplified is None:
            dropped += 1
            continue
        interpreted.append(_render_expr(simplified, thr=thr, atol=float(const_atol), rtol=float(const_rtol)))

    os.makedirs(os.path.dirname(out_clj) or ".", exist_ok=True)
    with open(out_clj, "w") as f:
        for s in interpreted:
            f.write(str(s) + "\n")

    return len(interpreted), dropped


def main() -> None:
    parser = argparse.ArgumentParser(description="Interpret mined .pl rules into simplified, human-readable formulas.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--rules-pl", default=None, help="Path to .pl rules file (one sympy srepr expression per line).")
    group.add_argument("--rules-dir", default=None, help="Interpret all *.pl files under a directory (recursively).")
    parser.add_argument("--thresholds-csv", required=True, help="Clean-derived thresholds CSV (clean_stats_thresholds.csv).")
    parser.add_argument("--out-clj", default=None, help="Output .clj path (single-file mode only; default: <rules-pl>.clj).")
    parser.add_argument("--out-dir", default=None, help="Output directory (directory mode only; default: next to each .pl).")
    parser.add_argument(
        "--simplify-tol",
        type=float,
        default=1e-9,
        help="Tolerance for redundancy/contradiction detection when comparing numeric thresholds.",
    )
    parser.add_argument("--const-atol", type=float, default=1e-9)
    parser.add_argument("--const-rtol", type=float, default=1e-6)
    args = parser.parse_args()

    thresholds_csv = _resolve_path(str(args.thresholds_csv))

    thr = _load_thresholds_csv(thresholds_csv)

    if args.rules_pl:
        rules_pl = _resolve_path(str(args.rules_pl))
        out_clj = _resolve_path(str(args.out_clj)) if args.out_clj else str(rules_pl) + ".clj"
        n_rules, dropped = _interpret_one(
            rules_pl=str(rules_pl),
            thr=thr,
            out_clj=str(out_clj),
            simplify_tol=float(args.simplify_tol),
            const_atol=float(args.const_atol),
            const_rtol=float(args.const_rtol),
        )
        print(f"* Wrote interpreted rules: {out_clj} (rules={n_rules}, dropped_contradictions={dropped})")
        return

    rules_dir = _resolve_path(str(args.rules_dir))
    out_dir = _resolve_path(str(args.out_dir)) if args.out_dir else None
    pl_files = _iter_pl_files(rules_dir)
    if not pl_files:
        raise RuntimeError(f"* No .pl files found under: {rules_dir}")

    total_rules = 0
    total_dropped = 0
    for i, path in enumerate(pl_files, start=1):
        rel = os.path.relpath(str(path), str(rules_dir))
        out_clj = str(path) + ".clj"
        if out_dir is not None:
            out_clj = os.path.join(str(out_dir), str(rel)) + ".clj"

        n_rules, dropped = _interpret_one(
            rules_pl=str(path),
            thr=thr,
            out_clj=str(out_clj),
            simplify_tol=float(args.simplify_tol),
            const_atol=float(args.const_atol),
            const_rtol=float(args.const_rtol),
        )
        total_rules += int(n_rules)
        total_dropped += int(dropped)
        print(f"* [{i}/{len(pl_files)}] {rel} -> {out_clj} (rules={n_rules}, dropped={dropped})")

    print(f"* Done. files={len(pl_files)} rules={total_rules} dropped_contradictions={total_dropped}")


if __name__ == "__main__":
    main()
