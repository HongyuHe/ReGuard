#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import sympy as sp


VIDEO_BIT_RATE_KBPS = np.array([300, 750, 1200, 1850, 2850, 4300], dtype=int)
S_LEN = 8
A_DIM = 6


def _resolve_path(repo_relative_or_abs: str) -> str:
    expanded = os.path.expandvars(os.path.expanduser(str(repo_relative_or_abs)))
    if os.path.isabs(expanded):
        return expanded
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(repo_root, expanded)


def _feature_columns() -> list[str]:
    cols = ["LastBitrateKbps", "BufferSeconds"]
    cols += [f"ThroughputMbps{i}" for i in range(S_LEN)]
    cols += [f"DelayMs{i}" for i in range(S_LEN)]
    cols += [f"NextChunkBytes{i}" for i in range(A_DIM)]
    cols += ["ChunksRemaining"]
    return cols


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    cols = set(str(c) for c in df.columns)
    buffer_col = "buffer_hist_sec_7" if "buffer_hist_sec_7" in cols else ("buffer_hist_s_7" if "buffer_hist_s_7" in cols else None)
    remain_col = "video_chunk_remain" if "video_chunk_remain" in cols else ("video_chunk_remain_7" if "video_chunk_remain_7" in cols else None)

    required = ["last_action_kbps"]
    if buffer_col is not None:
        required.append(str(buffer_col))
    if remain_col is not None:
        required.append(str(remain_col))
    required += [f"throughput_hist_mbps_{i}" for i in range(S_LEN)]
    required += [f"delay_hist_ms_{i}" for i in range(S_LEN)]
    required += [f"next_chunk_size_bytes_{i}" for i in range(A_DIM)]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"* Missing required feature columns: {missing}")

    out = pd.DataFrame()
    out["LastBitrateKbps"] = df["last_action_kbps"].astype(int)
    if buffer_col is None:
        raise RuntimeError("* Missing buffer history column (expected buffer_hist_sec_7 or buffer_hist_s_7).")
    out["BufferSeconds"] = df[str(buffer_col)].astype(float)
    for i in range(S_LEN):
        out[f"ThroughputMbps{i}"] = df[f"throughput_hist_mbps_{i}"].astype(float)
        out[f"DelayMs{i}"] = df[f"delay_hist_ms_{i}"].astype(float)
    for i in range(A_DIM):
        out[f"NextChunkBytes{i}"] = df[f"next_chunk_size_bytes_{i}"].astype(int)
    if remain_col is None:
        raise RuntimeError("* Missing remaining-chunks column (expected video_chunk_remain or video_chunk_remain_7).")
    out["ChunksRemaining"] = df[str(remain_col)].astype(int)
    return out[_feature_columns()]


def _extract_sympy_rules_from_sklearn_tree(
    model,
    *,
    feature_cols: list[str],
    label_col: str,
    leaf_purity: float,
) -> list[sp.Expr]:
    tree = model.tree_
    feature_names = list(feature_cols)

    def sym_var(name: str) -> sp.Symbol:
        if name in {label_col, "BitrateKbps", "MinBitrateKbps", "MaxBitrateKbps"}:
            return sp.Symbol(name, integer=True)
        if name.startswith("NextChunkBytes") or name in {"ChunksRemaining", "LastBitrateKbps"}:
            return sp.Symbol(name, integer=True)
        return sp.Symbol(name, real=True)

    rules: list[sp.Expr] = []

    def recurse(node_id: int, path_conds: list[sp.Expr]) -> None:
        if int(tree.feature[node_id]) < 0:
            counts = np.asarray(tree.value[node_id][0], dtype=np.float64)
            total = float(np.sum(counts))
            if total <= 0.0:
                return
            probs = counts / total
            if float(np.max(probs)) < float(leaf_purity):
                return
            class_idx = int(np.argmax(probs))
            cls = int(model.classes_[class_idx])
            consequent = sp.Eq(sym_var(label_col), sp.Integer(cls))
            if path_conds:
                antecedent = sp.And(*path_conds)
            else:
                buf = sp.Symbol("BufferSeconds", real=True)
                antecedent = sp.GreaterThan(buf, sp.Float(-1e9))
            rules.append(sp.Implies(antecedent, consequent))
            return

        feat = int(tree.feature[node_id])
        thr = float(tree.threshold[node_id])
        feat_name = feature_names[feat]
        var = sym_var(feat_name)
        cond_left = sp.Le(var, sp.Float(thr))
        cond_right = sp.Gt(var, sp.Float(thr))

        recurse(int(tree.children_left[node_id]), path_conds + [cond_left])
        recurse(int(tree.children_right[node_id]), path_conds + [cond_right])

    recurse(0, [])
    return sorted(set(rules), key=lambda e: str(e))


def _extract_sympy_rules_from_h2o_tree(
    model,
    *,
    feature_cols: list[str],
    label_col: str,
    categorical_cols: list[str],
    leaf_purity: float,
) -> list[sp.Expr]:
    from h2o.tree import H2OTree  # type: ignore

    _ = list(feature_cols)
    _ = list(categorical_cols)

    names = list(model._model_json["output"]["names"])
    domains = list(model._model_json["output"]["domains"])
    domain_by_name = {str(n): d for n, d in zip(names, domains)}

    def sym_var(name: str) -> sp.Symbol:
        if name in {label_col, "BitrateKbps", "MinBitrateKbps", "MaxBitrateKbps"}:
            return sp.Symbol(name, integer=True)
        if name.startswith("NextChunkBytes") or name in {"ChunksRemaining", "LastBitrateKbps"}:
            return sp.Symbol(name, integer=True)
        return sp.Symbol(name, real=True)

    def as_int_if_possible(value):
        try:
            return int(value)
        except Exception:
            return value

    def categorical_levels_to_values(varname: str, level_indices: list) -> list:
        dom = domain_by_name.get(varname)
        out = []
        for raw in level_indices:
            try:
                idx = int(raw)
            except Exception:
                out.append(as_int_if_possible(raw))
                continue
            if dom is not None and 0 <= idx < len(dom):
                out.append(as_int_if_possible(dom[idx]))
            else:
                out.append(idx)
        return out

    def in_set_expr(var: sp.Symbol, values: list) -> sp.Expr:
        if len(values) == 0:
            return sp.true
        eqs = [sp.Eq(var, as_int_if_possible(v)) for v in values]
        return eqs[0] if len(eqs) == 1 else sp.Or(*eqs)

    def not_in_set_expr(var: sp.Symbol, values: list) -> sp.Expr:
        if len(values) == 0:
            return sp.true
        nes = [sp.Ne(var, as_int_if_possible(v)) for v in values]
        return nes[0] if len(nes) == 1 else sp.And(*nes)

    def recurse(
        node,
        path_conds: list[sp.Expr],
        out_rules: list[sp.Expr],
        cls_label: str,
        *,
        keep_leaf_pred,
    ):
        if node.__class__.__name__ == "H2OLeafNode":
            pred = float(node.prediction)
            if not bool(keep_leaf_pred(pred)):
                return
            consequent = sp.Eq(sym_var(label_col), sp.Integer(int(cls_label)))
            if path_conds:
                antecedent = sp.And(*path_conds)
            else:
                buf = sp.Symbol("BufferSeconds", real=True)
                antecedent = sp.GreaterThan(buf, sp.Float(-1e9))
            out_rules.append(sp.Implies(antecedent, consequent))
            return

        varname = str(node.split_feature)
        var = sym_var(varname)

        if node.left_levels or node.right_levels:
            left_levels = list(node.left_levels or [])
            right_levels = list(node.right_levels or [])
            if left_levels:
                left_values = categorical_levels_to_values(varname, left_levels)
                cond_left = in_set_expr(var, left_values)
                cond_right = not_in_set_expr(var, left_values)
            else:
                right_values = categorical_levels_to_values(varname, right_levels)
                cond_left = not_in_set_expr(var, right_values)
                cond_right = in_set_expr(var, right_values)
        else:
            thr = float(node.threshold)
            cond_left = sp.Le(var, sp.Float(thr))
            cond_right = sp.Gt(var, sp.Float(thr))

        recurse(node.left_child, path_conds + [cond_left], out_rules, cls_label, keep_leaf_pred=keep_leaf_pred)
        recurse(node.right_child, path_conds + [cond_right], out_rules, cls_label, keep_leaf_pred=keep_leaf_pred)

    label_domain = model._model_json["output"]["domains"][names.index(label_col)]
    if label_domain is None:
        raise RuntimeError(f"* H2O model does not expose a categorical domain for {label_col}")

    rules: list[sp.Expr] = []
    label_domain = [str(c) for c in list(label_domain)]
    if len(label_domain) == 1:
        consequent = sp.Eq(sym_var(label_col), sp.Integer(int(label_domain[0])))
        buf = sp.Symbol("BufferSeconds", real=True)
        antecedent = sp.GreaterThan(buf, sp.Float(-1e9))
        rules.append(sp.Implies(antecedent, consequent))
        return rules

    if len(label_domain) == 2:
        #* H2O binomial models build only one tree "class" per iteration. Requesting the other class errors.
        #* We extract the built-class rules directly and derive the other class via complement probability.
        tree = H2OTree(model=model, tree_number=0)
        built_cls = str(tree.tree_class) if tree.tree_class is not None else str(label_domain[0])
        if built_cls not in set(label_domain):
            built_cls = str(label_domain[0])
        other_cls = str(label_domain[1]) if built_cls == str(label_domain[0]) else str(label_domain[0])

        keep_built = lambda p: p >= float(leaf_purity)
        keep_other = lambda p: p <= float(1.0 - float(leaf_purity))
        recurse(tree.root_node, [], rules, built_cls, keep_leaf_pred=keep_built)
        recurse(tree.root_node, [], rules, other_cls, keep_leaf_pred=keep_other)
        return sorted(set(rules), key=lambda e: str(e))

    for cls in label_domain:
        tree = H2OTree(model=model, tree_number=0, tree_class=cls)
        recurse(tree.root_node, [], rules, str(cls), keep_leaf_pred=(lambda p: p >= float(leaf_purity)))

    return sorted(set(rules), key=lambda e: str(e))


def _save_rules(rules: list[sp.Expr], out_path: str) -> None:
    out_path = str(out_path)
    #* Filter trivial boolean rules (e.g., Implies(False, ...) -> True).
    rules = [r for r in rules if r not in (sp.true, sp.false, True, False)]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        for r in rules:
            f.write(sp.srepr(r) + "\n")
    with open(out_path + ".json", "w") as f:
        json.dump([str(r) for r in rules], f, indent=0, sort_keys=True)


@dataclass(frozen=True)
class _SklearnParams:
    max_depth: int | None
    max_leaf_nodes: int | None
    min_samples_leaf: int


def _train_sklearn(df: pd.DataFrame, *, x: pd.DataFrame, y_col: str, seed: int, params: _SklearnParams):
    from sklearn.tree import DecisionTreeClassifier  # type: ignore

    y = df[y_col].astype(int)
    model = DecisionTreeClassifier(
        random_state=int(seed),
        max_depth=int(params.max_depth) if params.max_depth is not None else None,
        max_leaf_nodes=int(params.max_leaf_nodes) if params.max_leaf_nodes is not None else None,
        min_samples_leaf=int(params.min_samples_leaf),
    )
    model.fit(x, y)
    return model


def _train_h2o(
    df: pd.DataFrame,
    *,
    label_col: str,
    categorical_cols: list[str],
    seed: int,
    ntrees: int,
    max_depth: int,
    min_rows: int,
    sample_rate: float,
    mtries: int | None,
):
    import h2o  # type: ignore
    from h2o.estimators import H2ORandomForestEstimator  # type: ignore

    df = df.copy()
    for col in categorical_cols + [label_col]:
        df[col] = df[col].astype("category")

    frame = h2o.H2OFrame(df)
    for col in categorical_cols + [label_col]:
        frame[col] = frame[col].asfactor()

    x_cols = [c for c in df.columns if c != label_col]
    y_col = label_col

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
    model.train(x=x_cols, y=y_col, training_frame=frame)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DTs for min/max bitrate bounds and extract rules.")
    parser.add_argument("--data", required=True, help="Labeled CSV from label_bounds.py.")
    parser.add_argument("--backend", choices=["sklearn", "h2o"], default="sklearn")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=1.0)
    parser.add_argument("--max-samples", type=int, default=0, help="Cap the number of training rows (0 disables).")
    parser.add_argument("--leaf-purity", type=float, default=0.999999)
    parser.add_argument("--out-prefix", required=True, help="Output prefix (writes <prefix>_min.pl and <prefix>_max.pl).")

    parser.add_argument("--sklearn-max-depth", type=int, default=None)
    parser.add_argument("--sklearn-max-leaf-nodes", type=int, default=None)
    parser.add_argument("--sklearn-min-samples-leaf", type=int, default=1)

    parser.add_argument("--h2o-ntrees", type=int, default=1)
    parser.add_argument("--h2o-max-depth", type=int, default=20)
    parser.add_argument("--h2o-min-rows", type=int, default=1)
    parser.add_argument("--h2o-sample-rate", type=float, default=1.0)
    parser.add_argument("--h2o-mtries", type=int, default=None)

    args = parser.parse_args()

    data_path = _resolve_path(str(args.data))
    out_prefix = _resolve_path(str(args.out_prefix))
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)

    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"* Labeled CSV not found: {data_path}")

    df = pd.read_csv(data_path)
    label_min_col = "label_min_action_kbps"
    label_max_col = "label_max_action_kbps"
    if label_min_col not in df.columns or label_max_col not in df.columns:
        raise RuntimeError(f"* Missing label columns: expected {label_min_col!r} and {label_max_col!r}")

    x = _build_features(df)
    y_min = df[label_min_col].astype(int)
    y_max = df[label_max_col].astype(int)

    features_df = x.copy()
    features_df["MinBitrateKbps"] = y_min
    features_df["MaxBitrateKbps"] = y_max

    if int(args.max_samples) > 0 and len(features_df) > int(args.max_samples):
        rng = np.random.RandomState(int(args.seed))
        idx = rng.permutation(len(features_df))[: int(args.max_samples)]
        features_df = features_df.iloc[idx].reset_index(drop=True)

    if float(args.train_frac) < 1.0 and len(features_df) > 1:
        n_train = max(int(float(args.train_frac) * len(features_df)), 1)
        rng = np.random.RandomState(int(args.seed))
        idx = rng.permutation(len(features_df))
        features_df = features_df.iloc[idx[:n_train]].reset_index(drop=True)

    feature_cols = _feature_columns()
    categorical_cols = ["LastBitrateKbps"]

    backend = str(args.backend)
    #* Handle degenerate single-class labels explicitly: H2O (and sometimes sklearn) can error when the response has <2 classes.
    #* We emit an always-true guard on `BufferSeconds` to keep rule format compatible with the runtime shield parser.
    uniq_min = sorted(set(int(x) for x in features_df["MinBitrateKbps"].astype(int).tolist()))
    uniq_max = sorted(set(int(x) for x in features_df["MaxBitrateKbps"].astype(int).tolist()))

    def _trivial_rule(label_col: str, value: int) -> sp.Expr:
        buf = sp.Symbol("BufferSeconds", real=True)
        label = sp.Symbol(str(label_col), integer=True)
        antecedent = sp.GreaterThan(buf, sp.Float(-1e9))
        consequent = sp.Eq(label, sp.Integer(int(value)))
        return sp.Implies(antecedent, consequent)

    rules_min: list[sp.Expr] | None = None
    rules_max: list[sp.Expr] | None = None
    if len(uniq_min) == 1:
        rules_min = [_trivial_rule("MinBitrateKbps", int(uniq_min[0]))]
    if len(uniq_max) == 1:
        rules_max = [_trivial_rule("MaxBitrateKbps", int(uniq_max[0]))]

    if backend == "sklearn":
        params = _SklearnParams(
            max_depth=int(args.sklearn_max_depth) if args.sklearn_max_depth is not None else None,
            max_leaf_nodes=int(args.sklearn_max_leaf_nodes) if args.sklearn_max_leaf_nodes is not None else None,
            min_samples_leaf=int(args.sklearn_min_samples_leaf),
        )
        if rules_min is None:
            model_min = _train_sklearn(
                features_df,
                x=features_df[feature_cols],
                y_col="MinBitrateKbps",
                seed=int(args.seed),
                params=params,
            )
            rules_min = _extract_sympy_rules_from_sklearn_tree(
                model_min,
                feature_cols=feature_cols,
                label_col="MinBitrateKbps",
                leaf_purity=float(args.leaf_purity),
            )
        if rules_max is None:
            model_max = _train_sklearn(
                features_df,
                x=features_df[feature_cols],
                y_col="MaxBitrateKbps",
                seed=int(args.seed),
                params=params,
            )
            rules_max = _extract_sympy_rules_from_sklearn_tree(
                model_max,
                feature_cols=feature_cols,
                label_col="MaxBitrateKbps",
                leaf_purity=float(args.leaf_purity),
            )

        out_min = out_prefix + "_min.pl"
        out_max = out_prefix + "_max.pl"
        _save_rules(list(rules_min or []), out_min)
        _save_rules(list(rules_max or []), out_max)

        print(f"* Wrote rules: {out_min} ({len(rules_min)}), {out_max} ({len(rules_max)})")
        return

    import h2o  # type: ignore

    #* Silence H2O dependency warnings about optional multi-threaded pandas conversion.
    try:
        from h2o.exceptions import H2ODependencyWarning  # type: ignore

        warnings.filterwarnings("ignore", category=H2ODependencyWarning)
    except Exception:
        pass

    #* If both labels are degenerate, skip spinning up H2O entirely.
    if rules_min is not None and rules_max is not None:
        out_min = out_prefix + "_min.pl"
        out_max = out_prefix + "_max.pl"
        _save_rules(list(rules_min), out_min)
        _save_rules(list(rules_max), out_max)
        print(f"* Wrote rules: {out_min} ({len(rules_min)}), {out_max} ({len(rules_max)})")
        return

    h2o.init()
    try:
        if rules_min is None:
            model_min = _train_h2o(
                features_df[feature_cols + ["MinBitrateKbps"]],
                label_col="MinBitrateKbps",
                categorical_cols=categorical_cols,
                seed=int(args.seed),
                ntrees=int(args.h2o_ntrees),
                max_depth=int(args.h2o_max_depth),
                min_rows=int(args.h2o_min_rows),
                sample_rate=float(args.h2o_sample_rate),
                mtries=int(args.h2o_mtries) if args.h2o_mtries is not None else None,
            )
            rules_min = _extract_sympy_rules_from_h2o_tree(
                model_min,
                feature_cols=feature_cols,
                label_col="MinBitrateKbps",
                categorical_cols=categorical_cols,
                leaf_purity=float(args.leaf_purity),
            )
        if rules_max is None:
            model_max = _train_h2o(
                features_df[feature_cols + ["MaxBitrateKbps"]],
                label_col="MaxBitrateKbps",
                categorical_cols=categorical_cols,
                seed=int(args.seed),
                ntrees=int(args.h2o_ntrees),
                max_depth=int(args.h2o_max_depth),
                min_rows=int(args.h2o_min_rows),
                sample_rate=float(args.h2o_sample_rate),
                mtries=int(args.h2o_mtries) if args.h2o_mtries is not None else None,
            )
            rules_max = _extract_sympy_rules_from_h2o_tree(
                model_max,
                feature_cols=feature_cols,
                label_col="MaxBitrateKbps",
                categorical_cols=categorical_cols,
                leaf_purity=float(args.leaf_purity),
            )

        out_min = out_prefix + "_min.pl"
        out_max = out_prefix + "_max.pl"
        _save_rules(list(rules_min or []), out_min)
        _save_rules(list(rules_max or []), out_max)

        print(f"* Wrote rules: {out_min} ({len(rules_min)}), {out_max} ({len(rules_max)})")
    finally:
        try:
            h2o.shutdown(prompt=False)
        except Exception:
            pass


if __name__ == "__main__":
    main()
