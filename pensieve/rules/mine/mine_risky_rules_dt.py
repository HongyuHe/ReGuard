#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Iterable


"""
Example usage:
python rules/mine/mine_risky_rules_dt.py --adv-stats rules/mine/data/hotnets19_states_stats.csv \
    --thresholds rules/mine/data/clean_stats_thresholds.csv \
    --backend sklearn --gap-threshold-pct 80
"""

def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _resolve_path(repo_relative_or_abs: str) -> str:
    expanded = os.path.expandvars(os.path.expanduser(str(repo_relative_or_abs)))
    if os.path.isabs(expanded):
        return expanded
    return os.path.join(_repo_root(), expanded)


def _default_out_prefix(*, adv_stats_csv: str, backend: str, gap_pct: float, threshold_cols: list[str]) -> str:
    attack = os.path.splitext(os.path.basename(str(adv_stats_csv)))[0]
    cols_tag = '' # "-".join(str(c) for c in threshold_cols) if threshold_cols else "all"
    tag = f"{attack}_{backend}_gap{str(gap_pct).replace('.', 'p')}{cols_tag}"
    return os.path.join(_repo_root(), "rules", "mine", "out", tag)


@dataclass(frozen=True)
class _Predicate:
    name: str
    feature: str
    op: str
    threshold_col: str
    threshold_value: float


def _threshold_columns(df: Any) -> list[str]:
    cols = []
    for c in list(getattr(df, "columns", [])):
        name = str(c)
        if not name.startswith("p"):
            continue
        suffix = name[1:]
        if not suffix.isdigit():
            continue
        cols.append(name)
    return cols


def _parse_csv_list(s: str) -> list[str]:
    parts = [p.strip() for p in str(s).split(",")]
    return [p for p in parts if p]


def _is_int_like(x: float, *, tol: float = 1e-9) -> bool:
    if not math.isfinite(float(x)):
        return False
    return abs(float(x) - float(round(float(x)))) <= float(tol)


def _sym_number(v: float) -> Any:
    import sympy as sp

    if _is_int_like(float(v)):
        return sp.Integer(int(round(float(v))))
    return sp.Float(float(v))


def _load_threshold_predicates(
    thresholds_csv: str,
    *,
    threshold_cols: list[str] | None,
    include_eq: bool,
) -> tuple[list[_Predicate], dict[str, dict[str, float]]]:
    import pandas as pd

    df = pd.read_csv(str(thresholds_csv))
    if "feature" not in df.columns:
        raise RuntimeError("* Threshold CSV must include a 'feature' column.")

    all_thr_cols = _threshold_columns(df)
    if not all_thr_cols:
        raise RuntimeError("* No threshold columns found (expected pNN columns like p90, p95).")

    use_cols = list(all_thr_cols)
    if threshold_cols is not None:
        missing = [c for c in threshold_cols if c not in all_thr_cols]
        if missing:
            raise RuntimeError(f"* Unknown --threshold-cols entries (not in CSV): {missing}")
        use_cols = list(threshold_cols)

    thresholds_map: dict[str, dict[str, float]] = {}
    preds: list[_Predicate] = []

    for _, row in df.iterrows():
        feat = str(row["feature"])
        thresholds_map.setdefault(feat, {})
        for c in use_cols:
            raw = float(row.get(str(c), float("nan")))
            thresholds_map[feat][str(c)] = float(raw)
            if not math.isfinite(float(raw)):
                continue
            preds.append(
                _Predicate(
                    name=f"{feat}__gt__{c}",
                    feature=str(feat),
                    op="gt",
                    threshold_col=str(c),
                    threshold_value=float(raw),
                )
            )
            if bool(include_eq) and _is_int_like(float(raw)):
                preds.append(
                    _Predicate(
                        name=f"{feat}__eq__{c}",
                        feature=str(feat),
                        op="eq",
                        threshold_col=str(c),
                        threshold_value=float(raw),
                    )
                )

    preds = sorted({p.name: p for p in preds}.values(), key=lambda p: str(p.name))
    return preds, thresholds_map


def _build_predicate_matrix(df: Any, *, predicates: list[_Predicate], require_features: bool) -> tuple[Any, dict[str, _Predicate]]:
    import numpy as np
    import pandas as pd

    cols = set(str(c) for c in df.columns)
    by_feat: dict[str, list[_Predicate]] = {}
    for p in predicates:
        by_feat.setdefault(str(p.feature), []).append(p)

    missing_feats = sorted([f for f in by_feat.keys() if f not in cols])
    if missing_feats and bool(require_features):
        raise RuntimeError(f"* Missing predicate base feature(s) in adv stats CSV: {missing_feats}")

    out_data: dict[str, Any] = {}
    pred_map: dict[str, _Predicate] = {}

    for feat in sorted(by_feat.keys()):
        plist = sorted(by_feat[str(feat)], key=lambda p: str(p.name))
        if feat not in cols:
            continue
        values = df[str(feat)].astype(float).to_numpy(dtype=np.float64, copy=False)
        for p in plist:
            if str(p.op) == "gt":
                out_data[str(p.name)] = (values > float(p.threshold_value)).astype(np.int8)
            elif str(p.op) == "eq":
                out_data[str(p.name)] = (values == float(p.threshold_value)).astype(np.int8)
            else:
                raise ValueError(f"* Unknown predicate op: {p.op}")
            pred_map[str(p.name)] = p

    out = pd.DataFrame(out_data, index=df.index)
    if out.shape[1] == 0:
        raise RuntimeError("* No predicate features were created. Check thresholds/features alignment.")

    return out, pred_map


def _labels(df: Any, *, gap_threshold_pct: float) -> dict[str, Any]:
    import numpy as np

    required = ["oracle_gap_pct", "policy_action_kbps", "oracle_action_kbps"]
    cols = set(str(c) for c in df.columns)
    missing = [c for c in required if c not in cols]
    if missing:
        raise RuntimeError(f"* Missing required label columns in adv stats CSV: {missing}")

    gap = df["oracle_gap_pct"].astype(float).to_numpy(dtype=np.float64, copy=False)
    pol = df["policy_action_kbps"].astype(float).to_numpy(dtype=np.float64, copy=False)
    ora = df["oracle_action_kbps"].astype(float).to_numpy(dtype=np.float64, copy=False)

    y_gap = (gap > float(gap_threshold_pct)).astype(np.int32)
    y_aggr = (pol > ora).astype(np.int32)
    y_cons = (pol < ora).astype(np.int32)
    return {"high_gap": y_gap, "aggressive": y_aggr, "conservative": y_cons}


def _predicate_true_expr(pred: _Predicate) -> Any:
    import sympy as sp

    var = sp.Symbol(str(pred.feature), real=True)
    thr = _sym_number(float(pred.threshold_value))
    if str(pred.op) == "gt":
        return sp.Gt(var, thr)
    if str(pred.op) == "eq":
        return sp.Eq(var, thr)
    raise ValueError(f"* Unknown predicate op: {pred.op}")


def _predicate_false_expr(pred: _Predicate) -> Any:
    import sympy as sp

    var = sp.Symbol(str(pred.feature), real=True)
    thr = _sym_number(float(pred.threshold_value))
    if str(pred.op) == "gt":
        return sp.Le(var, thr)
    if str(pred.op) == "eq":
        return sp.Ne(var, thr)
    raise ValueError(f"* Unknown predicate op: {pred.op}")


def _extract_rules_from_sklearn_tree(
    model: Any,
    *,
    feature_cols: list[str],
    pred_map: dict[str, _Predicate],
    label_name: str,
    positive_class: int,
    leaf_purity: float,
) -> list[Any]:
    import numpy as np
    import sympy as sp

    tree = model.tree_
    feature_names = list(feature_cols)
    rules: list[Any] = []

    def recurse(node_id: int, path_conds: list[Any]) -> None:
        if int(tree.feature[node_id]) < 0:
            counts = np.asarray(tree.value[node_id][0], dtype=np.float64)
            total = float(np.sum(counts))
            if total <= 0.0:
                return
            probs = counts / total
            pred_idx = int(np.argmax(probs))
            cls = int(model.classes_[pred_idx])
            if int(cls) != int(positive_class):
                return
            if float(np.max(probs)) < float(leaf_purity):
                return
            consequent = sp.Eq(sp.Symbol(str(label_name), integer=True), sp.Integer(int(cls)))
            antecedent = sp.And(*path_conds) if path_conds else sp.true
            rules.append(sp.Implies(antecedent, consequent))
            return

        feat = int(tree.feature[node_id])
        thr = float(tree.threshold[node_id])
        feat_name = str(feature_names[feat])

        if feat_name in pred_map:
            pred = pred_map[feat_name]
            cond_left = _predicate_false_expr(pred)
            cond_right = _predicate_true_expr(pred)
        else:
            var = sp.Symbol(feat_name, real=True)
            cond_left = sp.Le(var, sp.Float(thr))
            cond_right = sp.Gt(var, sp.Float(thr))

        recurse(int(tree.children_left[node_id]), path_conds + [cond_left])
        recurse(int(tree.children_right[node_id]), path_conds + [cond_right])

    recurse(0, [])
    return sorted(set(rules), key=lambda e: str(e))


def _train_sklearn_tree(x: Any, y: Any, *, seed: int, max_depth: int | None, max_leaf_nodes: int | None, min_samples_leaf: int):
    from sklearn.tree import DecisionTreeClassifier  # type: ignore

    model = DecisionTreeClassifier(
        random_state=int(seed),
        max_depth=int(max_depth) if max_depth is not None else None,
        max_leaf_nodes=int(max_leaf_nodes) if max_leaf_nodes is not None else None,
        min_samples_leaf=int(min_samples_leaf),
    )
    model.fit(x, y)
    return model


def _train_h2o_tree(
    df_xy: Any,
    *,
    x_cols: list[str],
    y_col: str,
    seed: int,
    max_depth: int,
    min_rows: int,
    sample_rate: float,
    mtries: int | None,
) -> Any:
    import h2o  # type: ignore
    from h2o.estimators import H2ORandomForestEstimator  # type: ignore

    frame = h2o.H2OFrame(df_xy)
    frame[str(y_col)] = frame[str(y_col)].asfactor()

    params: dict[str, Any] = {
        "ntrees": 1,
        "max_depth": int(max_depth),
        "min_rows": int(min_rows),
        "sample_rate": float(sample_rate),
        "seed": int(seed),
    }
    if mtries is not None:
        params["mtries"] = int(mtries)

    model = H2ORandomForestEstimator(**params)
    model.train(x=list(x_cols), y=str(y_col), training_frame=frame)
    return model


def _extract_rules_from_h2o_tree(
    model: Any,
    *,
    pred_map: dict[str, _Predicate],
    label_name: str,
    positive_class: int,
    leaf_purity: float,
) -> list[Any]:
    import sympy as sp
    from h2o.tree import H2OTree  # type: ignore

    names = list(model._model_json["output"]["names"])
    domains = list(model._model_json["output"]["domains"])

    if str(label_name) not in names:
        raise RuntimeError(f"* H2O model does not expose label column name: {label_name}")

    label_domain = domains[names.index(str(label_name))]
    if label_domain is None:
        raise RuntimeError(f"* H2O model does not expose a categorical domain for {label_name}")

    pos_str = str(int(positive_class))
    if pos_str not in list(label_domain):
        raise RuntimeError(f"* Positive class {pos_str} not found in H2O label domain: {label_domain}")

    is_binomial = len(list(label_domain)) == 2
    binomial_domain0 = str(label_domain[0]) if is_binomial else None

    rules: list[Any] = []

    def recurse(node: Any, path_conds: list[Any]) -> None:
        if node.__class__.__name__ == "H2OLeafNode":
            purity = float(node.prediction)
            if is_binomial and binomial_domain0 is not None and pos_str != binomial_domain0:
                purity = 1.0 - purity
            if float(purity) < float(leaf_purity):
                return
            consequent = sp.Eq(sp.Symbol(str(label_name), integer=True), sp.Integer(int(positive_class)))
            antecedent = sp.And(*path_conds) if path_conds else sp.true
            rules.append(sp.Implies(antecedent, consequent))
            return

        varname = str(node.split_feature)
        if varname not in pred_map:
            raise RuntimeError(f"* Unexpected split feature not in predicate map: {varname}")

        pred = pred_map[varname]
        cond_left = _predicate_false_expr(pred)
        cond_right = _predicate_true_expr(pred)

        recurse(node.left_child, path_conds + [cond_left])
        recurse(node.right_child, path_conds + [cond_right])

    #* H2O exposes only tree_class=0 for binomial trees; leaf predictions correspond to the class-0 probability.
    tree_class = "0" if is_binomial else pos_str
    tree = H2OTree(model=model, tree_number=0, tree_class=tree_class)
    recurse(tree.root_node, [])
    return sorted(set(rules), key=lambda e: str(e))


def _save_rules(rules: list[Any], out_path: str) -> None:
    import sympy as sp

    rules = [r for r in rules if r not in (sp.true, sp.false, True, False)]
    os.makedirs(os.path.dirname(str(out_path)) or ".", exist_ok=True)
    with open(str(out_path), "w") as f:
        for r in rules:
            f.write(sp.srepr(r) + "\n")
    with open(str(out_path) + ".json", "w") as f:
        json.dump([str(r) for r in rules], f, indent=0, sort_keys=True)


def _print_label_stats(name: str, y: Any) -> None:
    import numpy as np

    arr = np.asarray(y, dtype=np.int32).reshape(-1)
    pos = int(np.sum(arr == 1))
    neg = int(np.sum(arr == 0))
    print(f"* label={name} pos={pos} neg={neg} pos_rate={pos / max(1, pos + neg):.4f}")


def _save_predicates(
    predicates: list[_Predicate],
    *,
    out_path: str,
) -> None:
    #* Persist all generated predicate definitions in sympy syntax (like *_rules.pl.json).
    payload = [str(_predicate_true_expr(p)) for p in list(predicates)]

    os.makedirs(os.path.dirname(str(out_path)) or ".", exist_ok=True)
    with open(str(out_path), "w") as f:
        json.dump(payload, f, indent=0, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mine template-aligned symbolic rules via boolean predicates (thresholds from clean stats) and decision trees."
    )
    parser.add_argument("--adv-stats", required=True, help="Adversarial states stats CSV (from gen_window_stats_thresholds.py).")
    parser.add_argument("--thresholds", required=True, help="Clean-derived thresholds CSV (clean_stats_thresholds.csv).")
    parser.add_argument("--backend", choices=["sklearn", "h2o"], default="sklearn")
    parser.add_argument("--out-prefix", default=None, help="Output prefix (writes <prefix>_{highgap,aggressive,conservative}.pl).")
    parser.add_argument("--gap-threshold-pct", type=float, default=80.0, help="High-gap label: oracle_gap_pct > threshold.")
    parser.add_argument(
        "--threshold-cols",
        type=str,
        default=None,
        help="Comma-separated subset of threshold columns to use (e.g., 'p90,p95'). Default: all pNN columns.",
    )
    parser.add_argument(
        "--include-eq",
        action="store_true",
        help="Also create equality predicates for integer-like thresholds (feature == threshold).",
    )
    parser.add_argument(
        "--require-threshold-features",
        action="store_true",
        help="Error if any threshold CSV feature is missing from the adv stats CSV (default: skip missing).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--leaf-purity", type=float, default=0.999999)
    parser.add_argument("--positive-class", type=int, default=1, help="Class label treated as positive for rule extraction.")

    parser.add_argument("--sklearn-max-depth", type=int, default=5)
    parser.add_argument("--sklearn-max-leaf-nodes", type=int, default=None)
    parser.add_argument("--sklearn-min-samples-leaf", type=int, default=1)

    parser.add_argument("--h2o-max-depth", type=int, default=30)
    parser.add_argument("--h2o-min-rows", type=int, default=1)
    parser.add_argument("--h2o-sample-rate", type=float, default=1.0)
    parser.add_argument("--h2o-mtries", type=int, default=None)

    args = parser.parse_args()

    adv_stats_csv = _resolve_path(str(args.adv_stats))
    thresholds_csv = _resolve_path(str(args.thresholds))

    if not os.path.isfile(adv_stats_csv):
        raise FileNotFoundError(f"* --adv-stats not found: {adv_stats_csv}")
    if not os.path.isfile(thresholds_csv):
        raise FileNotFoundError(f"* --thresholds not found: {thresholds_csv}")

    try:
        import pandas as pd
    except Exception as e:  # pragma: no cover
        raise RuntimeError("* Missing dependencies (pandas). Use `venv/bin/python`.") from e

    df_adv = pd.read_csv(str(adv_stats_csv))

    thr_cols = None
    if args.threshold_cols:
        thr_cols = _parse_csv_list(str(args.threshold_cols))

    predicates, thresholds_map = _load_threshold_predicates(
        thresholds_csv,
        threshold_cols=thr_cols,
        include_eq=bool(args.include_eq),
    )

    x_bin, pred_map = _build_predicate_matrix(
        df_adv,
        predicates=predicates,
        require_features=bool(args.require_threshold_features),
    )
    y_map = _labels(df_adv, gap_threshold_pct=float(args.gap_threshold_pct))

    import numpy as np

    risky_mask = np.asarray(y_map["high_gap"], dtype=np.int32).reshape(-1) == 1
    num_risky = int(np.sum(risky_mask))

    _print_label_stats("high_gap", y_map["high_gap"])
    if int(num_risky) > 0:
        _print_label_stats("aggressive (risky_only)", y_map["aggressive"][risky_mask])
        _print_label_stats("conservative (risky_only)", y_map["conservative"][risky_mask])
    else:
        _print_label_stats("aggressive (risky_only)", np.asarray([], dtype=np.int32))
        _print_label_stats("conservative (risky_only)", np.asarray([], dtype=np.int32))

    used_thr_cols = sorted({p.threshold_col for p in pred_map.values()})
    out_prefix = (
        _resolve_path(str(args.out_prefix))
        if args.out_prefix
        else _default_out_prefix(
            adv_stats_csv=str(adv_stats_csv),
            backend=str(args.backend),
            gap_pct=float(args.gap_threshold_pct),
            threshold_cols=list(used_thr_cols),
        )
    )

    os.makedirs(os.path.dirname(str(out_prefix)) or ".", exist_ok=True)

    predicates_out = os.path.join(
        os.path.dirname(str(out_prefix)) or ".",
        f"predicates_{os.path.basename(str(out_prefix))}.json",
    )
    _save_predicates(
        predicates,
        out_path=str(predicates_out),
    )

    meta_out = {
        "adv_stats": str(adv_stats_csv),
        "thresholds": str(thresholds_csv),
        "backend": str(args.backend),
        "gap_threshold_pct": float(args.gap_threshold_pct),
        "threshold_cols": list(used_thr_cols),
        "num_rows": int(len(df_adv)),
        "num_rows_risky": int(num_risky),
        "aggressive_conservative_train_subset": "oracle_gap_pct > gap_threshold_pct",
        "num_predicates": int(x_bin.shape[1]),
        "leaf_purity": float(args.leaf_purity),
        "positive_class": int(args.positive_class),
    }
    with open(str(out_prefix) + "_meta.json", "w") as f:
        json.dump(meta_out, f, indent=2, sort_keys=True)

    tasks = [
        ("highgap", "high_gap", False),
        ("aggressive", "aggressive", True),
        ("conservative", "conservative", True),
    ]

    #* Avoid tautological predictors for action-characterization labels.
    #* aggressive := policy_action_kbps > oracle_action_kbps
    #* conservative := policy_action_kbps < oracle_action_kbps
    #* So we exclude predicates derived from these action features when mining aggressive/conservative rules.
    action_label_leak_feats = {"policy_action_kbps", "oracle_action_kbps"}
    non_action_pred_cols: list[str] = []
    for c in list(x_bin.columns):
        p = pred_map.get(str(c))
        base_feat = str(getattr(p, "feature", ""))
        if base_feat not in action_label_leak_feats:
            non_action_pred_cols.append(str(c))

    if str(args.backend) == "sklearn":
        for out_tag, y_key, risky_only in tasks:
            x_base = x_bin if str(out_tag) == "highgap" else x_bin[non_action_pred_cols]
            x_task = x_base[risky_mask] if bool(risky_only) else x_base
            y_task = y_map[str(y_key)][risky_mask] if bool(risky_only) else y_map[str(y_key)]
            model = _train_sklearn_tree(
                x_task,
                y_task,
                seed=int(args.seed),
                max_depth=int(args.sklearn_max_depth) if args.sklearn_max_depth is not None else None,
                max_leaf_nodes=int(args.sklearn_max_leaf_nodes) if args.sklearn_max_leaf_nodes is not None else None,
                min_samples_leaf=int(args.sklearn_min_samples_leaf),
            )
            rules = _extract_rules_from_sklearn_tree(
                model,
                feature_cols=list(x_task.columns),
                pred_map=pred_map,
                label_name=str(out_tag),
                positive_class=int(args.positive_class),
                leaf_purity=float(args.leaf_purity),
            )
            out_path = str(out_prefix) + f"_{out_tag}.pl"
            _save_rules(rules, out_path)
            print(f"* wrote {len(rules)} rule(s): {out_path}")
    else:
        try:
            import h2o  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("* Missing h2o dependency. Use `venv/bin/python`.") from e

        h2o.init()
        try:
            for out_tag, y_key, risky_only in tasks:
                y_col = str(out_tag)
                x_base = x_bin if str(out_tag) == "highgap" else x_bin[non_action_pred_cols]
                x_task = x_base[risky_mask] if bool(risky_only) else x_base
                y_task = y_map[str(y_key)][risky_mask] if bool(risky_only) else y_map[str(y_key)]

                df_xy = x_task.copy()
                df_xy[y_col] = y_task

                model = _train_h2o_tree(
                    df_xy,
                    x_cols=list(x_task.columns),
                    y_col=str(y_col),
                    seed=int(args.seed),
                    max_depth=int(args.h2o_max_depth),
                    min_rows=int(args.h2o_min_rows),
                    sample_rate=float(args.h2o_sample_rate),
                    mtries=int(args.h2o_mtries) if args.h2o_mtries is not None else None,
                )
                rules = _extract_rules_from_h2o_tree(
                    model,
                    pred_map=pred_map,
                    label_name=str(y_col),
                    positive_class=int(args.positive_class),
                    leaf_purity=float(args.leaf_purity),
                )
                out_path = str(out_prefix) + f"_{out_tag}.pl"
                _save_rules(rules, out_path)
                print(f"* wrote {len(rules)} rule(s): {out_path}")
        finally:
            try:
                h2o.shutdown(prompt=False)
            except Exception:
                pass


if __name__ == "__main__":
    main()
