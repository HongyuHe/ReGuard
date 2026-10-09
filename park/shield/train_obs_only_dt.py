#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sympy as sp

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from shield.obs_features import feature_columns


def _resolve_path(path: str) -> Path:
    p = Path(os.path.expandvars(os.path.expanduser(str(path))))
    if p.is_absolute():
        return p
    return Path(__file__).resolve().parent.parent / p


def _infer_num_servers(df: pd.DataFrame) -> int:
    cols = [str(c) for c in df.columns if str(c).startswith("label_allow_server")]
    if len(cols) == 0:
        raise RuntimeError("missing label_allow_server* columns")
    return int(max(int(c.replace("label_allow_server", "")) for c in list(cols)) + 1)


def _build_features(df: pd.DataFrame, *, num_servers: int, include_derived: bool) -> pd.DataFrame:
    cols = feature_columns(int(num_servers), include_derived=bool(include_derived))
    missing = [str(c) for c in list(cols) if str(c) not in df.columns]
    if missing:
        raise RuntimeError(f"missing observable feature columns: {missing}")
    return df[list(cols)].copy()


def _sym_var(name: str, *, label_col: str) -> sp.Symbol:
    if str(name) == str(label_col) or str(name) == "Risk" or str(name).startswith("AllowServer"):
        return sp.Symbol(str(name), integer=True)
    if str(name).startswith("LoadProxyRank"):
        return sp.Symbol(str(name), integer=True)
    return sp.Symbol(str(name), real=True)


def _true_antecedent() -> sp.Expr:
    return sp.GreaterThan(sp.Symbol("IncomingSize", real=True), sp.Float(-1e18))


def _trivial_rule(*, label_col: str, positive_value: int = 1) -> sp.Expr:
    return sp.Implies(
        _true_antecedent(),
        sp.Eq(_sym_var(str(label_col), label_col=str(label_col)), sp.Integer(int(positive_value))),
    )


def _extract_sympy_rules_from_sklearn_tree(
    model,
    *,
    feature_cols: list[str],
    label_col: str,
    positive_value: int,
    leaf_purity: float,
) -> list[sp.Expr]:
    tree = model.tree_
    classes = [int(x) for x in list(model.classes_)]
    if int(positive_value) not in set(classes):
        return []
    positive_class_idx = int(classes.index(int(positive_value)))
    rules: list[sp.Expr] = []

    def recurse(node_id: int, path_conds: list[sp.Expr]) -> None:
        if int(tree.feature[node_id]) < 0:
            counts = np.asarray(tree.value[node_id][0], dtype=np.float64)
            total = float(np.sum(counts))
            if total <= 0.0:
                return
            probs = counts / total
            if float(probs[int(positive_class_idx)]) < float(leaf_purity):
                return
            antecedent = sp.And(*path_conds) if path_conds else _true_antecedent()
            consequent = sp.Eq(_sym_var(str(label_col), label_col=str(label_col)), sp.Integer(int(positive_value)))
            rules.append(sp.Implies(antecedent, consequent))
            return

        feat_idx = int(tree.feature[node_id])
        feat_name = str(feature_cols[int(feat_idx)])
        threshold = float(tree.threshold[node_id])
        var = _sym_var(feat_name, label_col=str(label_col))
        recurse(int(tree.children_left[node_id]), path_conds + [sp.Le(var, sp.Float(threshold))])
        recurse(int(tree.children_right[node_id]), path_conds + [sp.Gt(var, sp.Float(threshold))])

    recurse(0, [])
    return sorted(set(rules), key=lambda expr: str(expr))


def _extract_sympy_rules_from_h2o_tree(
    model,
    *,
    label_col: str,
    positive_value: int,
    leaf_purity: float,
) -> list[sp.Expr]:
    from h2o.tree import H2OTree

    names = [str(x) for x in list(model._model_json["output"]["names"])]
    domains = list(model._model_json["output"]["domains"])
    label_domain = domains[names.index(str(label_col))]
    if label_domain is None:
        raise RuntimeError(f"H2O model missing categorical domain for {label_col}")
    label_domain = [str(x) for x in list(label_domain)]
    positive_label = str(int(positive_value))
    if positive_label not in set(label_domain):
        return []

    def recurse(node, path_conds: list[sp.Expr], out: list[sp.Expr], *, keep_leaf_pred) -> None:
        if node.__class__.__name__ == "H2OLeafNode":
            pred = float(node.prediction)
            if not bool(keep_leaf_pred(pred)):
                return
            antecedent = sp.And(*path_conds) if path_conds else _true_antecedent()
            consequent = sp.Eq(_sym_var(str(label_col), label_col=str(label_col)), sp.Integer(int(positive_value)))
            out.append(sp.Implies(antecedent, consequent))
            return

        varname = str(node.split_feature)
        var = _sym_var(varname, label_col=str(label_col))
        if node.left_levels or node.right_levels:
            raise RuntimeError("categorical feature splits are not supported for Park LB shield rules")
        threshold = float(node.threshold)
        recurse(node.left_child, path_conds + [sp.Le(var, sp.Float(threshold))], out, keep_leaf_pred=keep_leaf_pred)
        recurse(node.right_child, path_conds + [sp.Gt(var, sp.Float(threshold))], out, keep_leaf_pred=keep_leaf_pred)

    rules: list[sp.Expr] = []
    if len(label_domain) == 1:
        if str(label_domain[0]) == str(positive_label):
            return [_trivial_rule(label_col=str(label_col), positive_value=int(positive_value))]
        return []

    if len(label_domain) == 2:
        tree = H2OTree(model=model, tree_number=0)
        built_cls = str(tree.tree_class) if tree.tree_class is not None else str(label_domain[0])
        if built_cls == str(positive_label):
            recurse(tree.root_node, [], rules, keep_leaf_pred=lambda p: float(p) >= float(leaf_purity))
        else:
            recurse(tree.root_node, [], rules, keep_leaf_pred=lambda p: float(p) <= float(1.0 - float(leaf_purity)))
        return sorted(set(rules), key=lambda expr: str(expr))

    tree = H2OTree(model=model, tree_number=0, tree_class=str(positive_label))
    recurse(tree.root_node, [], rules, keep_leaf_pred=lambda p: float(p) >= float(leaf_purity))
    return sorted(set(rules), key=lambda expr: str(expr))


def _save_rules(rules: list[sp.Expr], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned = [rule for rule in list(rules) if rule not in (sp.true, sp.false, True, False)]
    with out_path.open("w", encoding="utf-8") as f:
        for rule in list(cleaned):
            f.write(sp.srepr(rule) + "\n")
    with Path(str(out_path) + ".json").open("w", encoding="utf-8") as f:
        json.dump([str(rule) for rule in list(cleaned)], f, indent=2, sort_keys=True)


@dataclass(frozen=True)
class _SklearnParams:
    max_depth: int | None
    max_leaf_nodes: int | None
    min_samples_leaf: int


def _train_sklearn(x: pd.DataFrame, y: pd.Series, *, seed: int, params: _SklearnParams):
    from sklearn.tree import DecisionTreeClassifier

    model = DecisionTreeClassifier(
        random_state=int(seed),
        max_depth=int(params.max_depth) if params.max_depth is not None else None,
        max_leaf_nodes=int(params.max_leaf_nodes) if params.max_leaf_nodes is not None else None,
        min_samples_leaf=int(params.min_samples_leaf),
    )
    model.fit(x, y.astype(int))
    return model


def _train_h2o(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    label_col: str,
    seed: int,
    ntrees: int,
    max_depth: int,
    min_rows: int,
    sample_rate: float,
    mtries: int | None,
):
    import h2o
    from h2o.estimators import H2ORandomForestEstimator

    train_df = df[list(feature_cols) + [str(label_col)]].copy()
    train_df[str(label_col)] = train_df[str(label_col)].astype("category")
    frame = h2o.H2OFrame(train_df)
    frame[str(label_col)] = frame[str(label_col)].asfactor()
    params: dict[str, Any] = {
        "ntrees": int(ntrees),
        "max_depth": int(max_depth),
        "min_rows": int(min_rows),
        "sample_rate": float(sample_rate),
        "seed": int(seed),
    }
    if mtries is not None:
        params["mtries"] = int(mtries)
    model = H2ORandomForestEstimator(**params)
    model.train(x=list(feature_cols), y=str(label_col), training_frame=frame)
    return model


def _rules_for_label_sklearn(
    *,
    x: pd.DataFrame,
    y: pd.Series,
    label_col: str,
    positive_value: int,
    seed: int,
    leaf_purity: float,
    params: _SklearnParams,
) -> list[sp.Expr]:
    uniq = sorted(set(int(v) for v in y.astype(int).tolist()))
    if len(uniq) == 0:
        return []
    if len(uniq) == 1:
        if int(uniq[0]) == int(positive_value):
            return [_trivial_rule(label_col=str(label_col), positive_value=int(positive_value))]
        return []
    model = _train_sklearn(x, y, seed=int(seed), params=params)
    return _extract_sympy_rules_from_sklearn_tree(
        model,
        feature_cols=[str(c) for c in list(x.columns)],
        label_col=str(label_col),
        positive_value=int(positive_value),
        leaf_purity=float(leaf_purity),
    )


def _rules_for_label_h2o(
    *,
    x: pd.DataFrame,
    y: pd.Series,
    label_col: str,
    positive_value: int,
    seed: int,
    leaf_purity: float,
    args: argparse.Namespace,
) -> list[sp.Expr]:
    uniq = sorted(set(int(v) for v in y.astype(int).tolist()))
    if len(uniq) == 0:
        return []
    if len(uniq) == 1:
        if int(uniq[0]) == int(positive_value):
            return [_trivial_rule(label_col=str(label_col), positive_value=int(positive_value))]
        return []
    train_df = x.copy()
    train_df[str(label_col)] = y.astype(int)
    model = _train_h2o(
        train_df,
        feature_cols=[str(c) for c in list(x.columns)],
        label_col=str(label_col),
        seed=int(seed),
        ntrees=int(args.h2o_ntrees),
        max_depth=int(args.h2o_max_depth),
        min_rows=int(args.h2o_min_rows),
        sample_rate=float(args.h2o_sample_rate),
        mtries=int(args.h2o_mtries) if args.h2o_mtries is not None else None,
    )
    return _extract_sympy_rules_from_h2o_tree(
        model,
        label_col=str(label_col),
        positive_value=int(positive_value),
        leaf_purity=float(leaf_purity),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train observation-only Park LB shield decision trees.")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--backend", choices=["sklearn", "h2o"], default="sklearn")
    parser.add_argument("--out-prefix", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=1.0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--leaf-purity", type=float, default=0.999999)
    parser.add_argument("--raw-only", action="store_true")
    parser.add_argument("--train-allow-on-all-states", action="store_true")
    parser.add_argument("--sklearn-max-depth", type=int, default=None)
    parser.add_argument("--sklearn-max-leaf-nodes", type=int, default=None)
    parser.add_argument("--sklearn-min-samples-leaf", type=int, default=1)
    parser.add_argument("--h2o-ntrees", type=int, default=1)
    parser.add_argument("--h2o-max-depth", type=int, default=20)
    parser.add_argument("--h2o-min-rows", type=int, default=1)
    parser.add_argument("--h2o-sample-rate", type=float, default=1.0)
    parser.add_argument("--h2o-mtries", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = _resolve_path(str(args.data))
    out_prefix = _resolve_path(str(args.out_prefix))
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    if not data_path.is_file():
        raise FileNotFoundError(f"labeled data not found: {data_path}")
    if not (0.0 < float(args.train_frac) <= 1.0):
        raise ValueError("--train-frac must be in (0, 1]")

    df = pd.read_csv(data_path)
    if "label_risk" not in df.columns:
        raise RuntimeError("missing label_risk column")
    num_servers = _infer_num_servers(df)
    include_derived = bool(not args.raw_only)
    x_all = _build_features(df, num_servers=int(num_servers), include_derived=bool(include_derived))
    work_df = df.copy()
    work_x = x_all.copy()
    if int(args.max_samples) > 0 and len(work_df) > int(args.max_samples):
        rng = np.random.RandomState(int(args.seed))
        idx = rng.permutation(len(work_df))[: int(args.max_samples)]
        work_df = work_df.iloc[idx].reset_index(drop=True)
        work_x = work_x.iloc[idx].reset_index(drop=True)
    if float(args.train_frac) < 1.0 and len(work_df) > 1:
        rng = np.random.RandomState(int(args.seed))
        n_train = max(1, int(round(float(args.train_frac) * len(work_df))))
        idx = rng.permutation(len(work_df))[: int(n_train)]
        work_df = work_df.iloc[idx].reset_index(drop=True)
        work_x = work_x.iloc[idx].reset_index(drop=True)

    sklearn_params = _SklearnParams(
        max_depth=int(args.sklearn_max_depth) if args.sklearn_max_depth is not None else None,
        max_leaf_nodes=int(args.sklearn_max_leaf_nodes) if args.sklearn_max_leaf_nodes is not None else None,
        min_samples_leaf=int(args.sklearn_min_samples_leaf),
    )

    backend = str(args.backend)
    if backend == "h2o":
        import h2o

        try:
            from h2o.exceptions import H2ODependencyWarning

            warnings.filterwarnings("ignore", category=H2ODependencyWarning)
        except Exception:
            pass
        h2o.init()

    rule_files: dict[str, str] = {}
    try:
        if backend == "sklearn":
            risk_rules = _rules_for_label_sklearn(
                x=work_x,
                y=work_df["label_risk"].astype(int),
                label_col="Risk",
                positive_value=1,
                seed=int(args.seed),
                leaf_purity=float(args.leaf_purity),
                params=sklearn_params,
            )
        else:
            risk_rules = _rules_for_label_h2o(
                x=work_x,
                y=work_df["label_risk"].astype(int),
                label_col="Risk",
                positive_value=1,
                seed=int(args.seed),
                leaf_purity=float(args.leaf_purity),
                args=args,
            )
        risk_path = Path(str(out_prefix) + "_risk.pl")
        _save_rules(list(risk_rules), risk_path)
        rule_files["risk"] = str(risk_path)

        risky_mask = work_df["label_risk"].astype(int).to_numpy() == 1
        allow_df = work_df if bool(args.train_allow_on_all_states) else work_df.loc[risky_mask].reset_index(drop=True)
        allow_x = work_x if bool(args.train_allow_on_all_states) else work_x.loc[risky_mask].reset_index(drop=True)
        if len(allow_df) == 0:
            allow_df = work_df.copy()
            allow_x = work_x.copy()

        for action in range(int(num_servers)):
            label_col = f"AllowServer{int(action)}"
            data_col = f"label_allow_server{int(action)}"
            if data_col not in allow_df.columns:
                raise RuntimeError(f"missing {data_col}")
            if backend == "sklearn":
                rules = _rules_for_label_sklearn(
                    x=allow_x,
                    y=allow_df[data_col].astype(int),
                    label_col=str(label_col),
                    positive_value=1,
                    seed=int(args.seed) + int(action) + 1,
                    leaf_purity=float(args.leaf_purity),
                    params=sklearn_params,
                )
            else:
                rules = _rules_for_label_h2o(
                    x=allow_x,
                    y=allow_df[data_col].astype(int),
                    label_col=str(label_col),
                    positive_value=1,
                    seed=int(args.seed) + int(action) + 1,
                    leaf_purity=float(args.leaf_purity),
                    args=args,
                )
            out_path = Path(str(out_prefix) + f"_allow_server{int(action)}.pl")
            _save_rules(list(rules), out_path)
            rule_files[f"allow_server{int(action)}"] = str(out_path)
    finally:
        if backend == "h2o":
            try:
                import h2o

                h2o.shutdown(prompt=False)
            except Exception:
                pass

    manifest = {
        "backend": str(backend),
        "data": str(data_path),
        "out_prefix": str(out_prefix),
        "num_servers": int(num_servers),
        "include_derived": bool(include_derived),
        "feature_columns": feature_columns(int(num_servers), include_derived=bool(include_derived)),
        "train_rows": int(len(work_df)),
        "allow_train_rows": int(len(allow_df)),
        "risk_rate_train": float(np.mean(work_df["label_risk"].astype(int).to_numpy())) if len(work_df) else float("nan"),
        "leaf_purity": float(args.leaf_purity),
        "train_allow_on_all_states": bool(args.train_allow_on_all_states),
        "rule_files": dict(rule_files),
    }
    manifest_path = Path(str(out_prefix) + "_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote shield rules with prefix: {out_prefix}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
