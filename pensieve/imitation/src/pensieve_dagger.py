import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
import sympy as sp


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _pensieve_src() -> str:
    return os.path.join(_repo_root(), "Pensieve-PPO", "src")


def _resolve_path_under_pensieve_src(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(_pensieve_src(), path)


def _resolve_repo_root(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(_repo_root(), path)


def _parse_tags(raw: str) -> list[str]:
    raw = str(raw or "")
    tags = []
    for t in raw.split(","):
        t = t.strip()
        if t:
            tags.append(t)
    return tags


#* ABR constants (mirrors `Pensieve-PPO/src/env.py`).
S_INFO = 6
S_LEN = 8
A_DIM = 6
VIDEO_BIT_RATE = np.array([300.0, 750.0, 1200.0, 1850.0, 2850.0, 4300.0])  # Kbps
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
REBUF_PENALTY = 4.3
SMOOTH_PENALTY = 1.0
DEFAULT_QUALITY = 1
RAND_RANGE = 1000


def _bitrate_idx_to_kbps(bitrate_idx: int) -> int:
    return int(VIDEO_BIT_RATE[int(bitrate_idx)])


def _bitrate_kbps_to_idx(bitrate_kbps: int) -> int:
    bitrate_kbps = int(bitrate_kbps)
    idx = int(np.where(VIDEO_BIT_RATE.astype(int) == bitrate_kbps)[0][0])
    return idx


def _get_reward(bit_rate: int, rebuf_s: float, last_bit_rate: int) -> float:
    return float(VIDEO_BIT_RATE[bit_rate]) / float(M_IN_K) - float(REBUF_PENALTY) * float(rebuf_s) - float(
        SMOOTH_PENALTY
    ) * float(abs(float(VIDEO_BIT_RATE[bit_rate]) - float(VIDEO_BIT_RATE[last_bit_rate]))) / float(M_IN_K)


def _throughput_mbps(video_chunk_size_bytes: float, delay_ms: float) -> float:
    delay_ms_f = float(delay_ms)
    if delay_ms_f <= 0.0:
        return 0.0
    return float(video_chunk_size_bytes) * 8.0 / (delay_ms_f * 1000.0)


def _state_to_features(state: np.ndarray, last_bitrate_idx: int) -> dict:
    #* The PPO state stores normalized quantities; convert to more interpretable units for rule extraction.
    #* Index 0 is the oldest history step; index 7 is the latest.
    buffer_seconds = float(state[1, -1]) * float(BUFFER_NORM_FACTOR)

    features: dict[str, float | int] = {}
    features["LastBitrateKbps"] = _bitrate_idx_to_kbps(last_bitrate_idx)
    features["BufferSeconds"] = float(buffer_seconds)
    for i in range(S_LEN):
        throughput_mbps = float(state[2, i]) * 8.0
        delay_ms = float(state[3, i]) * float(M_IN_K) * float(BUFFER_NORM_FACTOR)
        features[f"ThroughputMbps{i}"] = float(throughput_mbps)
        features[f"DelayMs{i}"] = float(delay_ms)
    for i in range(A_DIM):
        next_chunk_bytes = float(state[4, i]) * float(M_IN_K) * float(M_IN_K)
        features[f"NextChunkBytes{i}"] = int(round(next_chunk_bytes))
    chunks_remaining = int(round(float(state[5, -1]) * float(CHUNK_TIL_VIDEO_END_CAP)))
    features["ChunksRemaining"] = int(chunks_remaining)
    return features


def _feature_columns() -> list[str]:
    cols = ["LastBitrateKbps", "BufferSeconds"]
    cols += [f"ThroughputMbps{i}" for i in range(S_LEN)]
    cols += [f"DelayMs{i}" for i in range(S_LEN)]
    cols += [f"NextChunkBytes{i}" for i in range(A_DIM)]
    cols += ["ChunksRemaining"]
    return cols


@dataclass(frozen=True)
class RolloutStep:
    state: np.ndarray
    features: dict
    action_idx: int
    reward: float


class PPOTeacherPolicy:
    def __init__(self, model_path: str, *, action_selection: str = "argmax"):
        if action_selection not in {"argmax", "gumbel"}:
            raise ValueError(f"Unknown action_selection={action_selection!r}")

        sys.path.insert(0, _pensieve_src())
        import ppo2 as network  # type: ignore

        self._network = network.Network(state_dim=[S_INFO, S_LEN], action_dim=A_DIM, learning_rate=1e-4)
        self._network.load_model(model_path)
        self._action_selection = str(action_selection)

    def predict_proba(self, states: np.ndarray) -> np.ndarray:
        #* `network.Network.predict` returns a single-state distribution; batch manually.
        probs = []
        for s in states:
            probs.append(
                self._network.predict(np.reshape(s, (1, S_INFO, S_LEN))),
            )
        return np.asarray(probs, dtype=np.float32)

    def act(self, state: np.ndarray, *, rng: np.random.RandomState) -> int:
        pi = self._network.predict(np.reshape(state, (1, S_INFO, S_LEN)))
        if self._action_selection == "argmax":
            return int(np.argmax(pi))
        noise = rng.gumbel(size=len(pi))
        return int(np.argmax(np.log(pi) + noise))


class H2OStudentPolicy:
    def __init__(
        self,
        *,
        seed: int,
        ntrees: int = 1,
        max_depth: int = 20,
        min_rows: int = 1,
        sample_rate: float = 1.0,
        mtries: int | None = None,
    ):
        import h2o  # type: ignore
        from h2o.estimators import H2ORandomForestEstimator  # type: ignore

        self._h2o = h2o
        self._H2ORandomForestEstimator = H2ORandomForestEstimator
        self._params = {
            "ntrees": int(ntrees),
            "max_depth": int(max_depth),
            "min_rows": int(min_rows),
            "sample_rate": float(sample_rate),
            "seed": int(seed),
        }
        if mtries is not None:
            self._params["mtries"] = int(mtries)

        self._model = None

    @property
    def model(self):
        if self._model is None:
            raise RuntimeError("* Student model not trained yet")
        return self._model

    def train(self, df: pd.DataFrame, *, label_col: str, categorical_cols: list[str], train_frac: float) -> None:
        df = df.copy()
        for col in categorical_cols + [label_col]:
            df[col] = df[col].astype("category")

        frame = self._h2o.H2OFrame(df)
        for col in categorical_cols + [label_col]:
            frame[col] = frame[col].asfactor()

        x_cols = [c for c in df.columns if c != label_col]
        y_col = label_col

        model = self._H2ORandomForestEstimator(**self._params)
        model.train(x=x_cols, y=y_col, training_frame=frame)

        self._model = model

    def predict(self, features: dict) -> int:
        #* Predicts `BitrateKbps` and converts it to an action index.
        df = pd.DataFrame([features], columns=_feature_columns())
        df["LastBitrateKbps"] = df["LastBitrateKbps"].astype("category")
        frame = self._h2o.H2OFrame(df)
        frame["LastBitrateKbps"] = frame["LastBitrateKbps"].asfactor()

        pred = self.model.predict(frame)
        #* For classification, H2O outputs a 'predict' column with the predicted label.
        bitrate_kbps = int(pred["predict"].as_data_frame(use_multi_thread=True).iloc[0, 0])
        return _bitrate_kbps_to_idx(bitrate_kbps)


class SklearnStudentPolicy:
    def __init__(
        self,
        *,
        seed: int,
        max_depth: int | None = None,
        max_leaf_nodes: int | None = None,
        min_samples_leaf: int = 1,
    ):
        from sklearn.tree import DecisionTreeClassifier  # type: ignore

        self._DecisionTreeClassifier = DecisionTreeClassifier
        self._params = {
            "random_state": int(seed),
            "max_depth": int(max_depth) if max_depth is not None else None,
            "max_leaf_nodes": int(max_leaf_nodes) if max_leaf_nodes is not None else None,
            "min_samples_leaf": int(min_samples_leaf),
        }
        self._model = None

    @property
    def model(self):
        if self._model is None:
            raise RuntimeError("* Student model not trained yet")
        return self._model

    def train(self, df: pd.DataFrame, *, label_col: str, categorical_cols: list[str], train_frac: float) -> None:
        _ = list(categorical_cols)
        df = df.copy()
        x_cols = [c for c in df.columns if c != label_col]
        y_col = label_col

        if float(train_frac) < 1.0 and len(df) > 1:
            n_train = max(int(float(train_frac) * len(df)), 1)
            rng = np.random.RandomState(int(self._params["random_state"]))
            idx = rng.permutation(len(df))
            df = df.iloc[idx[:n_train]].reset_index(drop=True)

        x = df[x_cols]
        y = df[y_col].astype(int)

        model = self._DecisionTreeClassifier(**self._params)
        model.fit(x, y)
        self._model = model

    def predict(self, features: dict) -> int:
        df = pd.DataFrame([features], columns=_feature_columns())
        bitrate_kbps = int(self.model.predict(df)[0])
        return _bitrate_kbps_to_idx(bitrate_kbps)


class PensieveRolloutEnv:
    def __init__(self, trace_dir: str, *, seed: int):
        sys.path.insert(0, _pensieve_src())
        import load_trace  # type: ignore
        import fixed_env as env  # type: ignore

        trace_dir = str(trace_dir)
        if not os.path.isabs(trace_dir):
            trace_dir = _resolve_path_under_pensieve_src(trace_dir)
        if not trace_dir.endswith(os.sep):
            trace_dir += os.sep

        all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(trace_dir)
        self._file_names = [str(x) for x in all_file_names]
        self._net_env = env.Environment(all_cooked_time=all_cooked_time, all_cooked_bw=all_cooked_bw, random_seed=int(seed))

        self._state = np.zeros((S_INFO, S_LEN), dtype=np.float32)
        self._last_bitrate = int(DEFAULT_QUALITY)
        self._bitrate = int(DEFAULT_QUALITY)
        self._time_ms = 0.0

    def reset_episode(self) -> None:
        #* Resets the agent-side episode history; the underlying trace index advances inside `fixed_env`.
        self._state[:] = 0.0
        self._last_bitrate = int(DEFAULT_QUALITY)
        self._bitrate = int(DEFAULT_QUALITY)
        self._time_ms = 0.0

    def step(self, policy, *, rng: np.random.RandomState) -> tuple[bool, dict[str, float | int | str]]:
        #* Uses the current `self._bitrate` for this chunk, then chooses the next action from the updated state.
        cur_trace_idx = int(getattr(self._net_env, "trace_idx", 0))
        cur_trace_name = self._file_names[cur_trace_idx] if 0 <= cur_trace_idx < len(self._file_names) else str(cur_trace_idx)
        try:
            trace_bw_mbps = float(self._net_env.cooked_bw[self._net_env.mahimahi_ptr])
        except Exception:
            trace_bw_mbps = float("nan")

        cur_bitrate = int(self._bitrate)
        delay_ms, sleep_ms, buffer_s, rebuf_s, video_chunk_size, next_chunk_sizes, end_of_video, chunk_remain = (
            self._net_env.get_video_chunk(cur_bitrate)
        )
        self._time_ms += float(delay_ms) + float(sleep_ms)

        qoe_quality = float(VIDEO_BIT_RATE[cur_bitrate]) / float(M_IN_K)
        qoe_rebuf_s = float(rebuf_s)
        qoe_smoothness = float(abs(float(VIDEO_BIT_RATE[cur_bitrate]) - float(VIDEO_BIT_RATE[self._last_bitrate]))) / float(M_IN_K)
        qoe_reward = _get_reward(cur_bitrate, float(rebuf_s), int(self._last_bitrate))
        self._last_bitrate = int(cur_bitrate)

        #* Update state history (same normalization as `Pensieve-PPO/src/test.py`).
        state = np.array(self._state, copy=True)
        state = np.roll(state, -1, axis=1)
        state[0, -1] = float(VIDEO_BIT_RATE[cur_bitrate]) / float(np.max(VIDEO_BIT_RATE))
        state[1, -1] = float(buffer_s) / float(BUFFER_NORM_FACTOR)
        state[2, -1] = float(video_chunk_size) / float(delay_ms) / float(M_IN_K)
        state[3, -1] = float(delay_ms) / float(M_IN_K) / float(BUFFER_NORM_FACTOR)
        state[4, :A_DIM] = np.array(next_chunk_sizes, dtype=np.float32) / float(M_IN_K) / float(M_IN_K)
        state[5, -1] = float(min(chunk_remain, CHUNK_TIL_VIDEO_END_CAP)) / float(CHUNK_TIL_VIDEO_END_CAP)
        self._state = state

        if not bool(end_of_video):
            features = _state_to_features(self._state, last_bitrate_idx=self._last_bitrate)
            action = int(policy.act(self._state, rng=rng)) if hasattr(policy, "act") else int(policy.predict(features))
            self._bitrate = int(action)

        metrics: dict[str, float | int | str] = {
            "env/trace": str(cur_trace_name),
            "env/time_s": float(self._time_ms / 1000.0),
            "env/bitrate_kbps": float(VIDEO_BIT_RATE[cur_bitrate]),
            "env/buffer_s": float(buffer_s),
            "env/rebuf_s": float(rebuf_s),
            "env/video_chunk_size_bytes": float(video_chunk_size),
            "env/delay_ms": float(delay_ms),
            "env/trace_bw_mbps": float(trace_bw_mbps),
            "env/throughput_mbps": float(_throughput_mbps(float(video_chunk_size), float(delay_ms))),
            "qoe_reward": float(qoe_reward),
            "qoe/quality": float(qoe_quality),
            "qoe/rebuf_s": float(qoe_rebuf_s),
            "qoe/smoothness": float(qoe_smoothness),
        }
        return bool(end_of_video), metrics

    def rollout(self, policy, *, rng: np.random.RandomState) -> list[RolloutStep]:
        rollout: list[RolloutStep] = []
        self.reset_episode()

        while True:
            delay_ms, sleep_ms, buffer_s, rebuf_s, video_chunk_size, next_chunk_sizes, end_of_video, chunk_remain = (
                self._net_env.get_video_chunk(self._bitrate)
            )
            _ = float(sleep_ms)
            reward = _get_reward(self._bitrate, float(rebuf_s), self._last_bitrate)
            self._last_bitrate = int(self._bitrate)

            #* Update state history.
            state = np.array(self._state, copy=True)
            state = np.roll(state, -1, axis=1)
            state[0, -1] = float(VIDEO_BIT_RATE[self._bitrate]) / float(np.max(VIDEO_BIT_RATE))
            state[1, -1] = float(buffer_s) / float(BUFFER_NORM_FACTOR)
            state[2, -1] = float(video_chunk_size) / float(delay_ms) / float(M_IN_K)
            state[3, -1] = float(delay_ms) / float(M_IN_K) / float(BUFFER_NORM_FACTOR)
            state[4, :A_DIM] = np.array(next_chunk_sizes, dtype=np.float32) / float(M_IN_K) / float(M_IN_K)
            state[5, -1] = float(min(chunk_remain, CHUNK_TIL_VIDEO_END_CAP)) / float(CHUNK_TIL_VIDEO_END_CAP)
            self._state = state

            if end_of_video:
                break

            features = _state_to_features(self._state, last_bitrate_idx=self._last_bitrate)
            action = int(policy.act(self._state, rng=rng)) if hasattr(policy, "act") else int(policy.predict(features))
            rollout.append(RolloutStep(state=np.array(self._state, copy=True), features=features, action_idx=action, reward=reward))
            self._bitrate = int(action)

        return rollout


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
        if name in {"BitrateKbps"}:
            return sp.Symbol(name, integer=True)
        if name.startswith("NextChunkBytes") or name == "ChunksRemaining" or name == "LastBitrateKbps":
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
            antecedent = sp.And(*path_conds) if path_conds else sp.true
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

    names = list(model._model_json["output"]["names"])
    domains = list(model._model_json["output"]["domains"])
    domain_by_name = {str(n): d for n, d in zip(names, domains)}

    def sym_var(name: str) -> sp.Symbol:
        if name in {"BitrateKbps"}:
            return sp.Symbol(name, integer=True)
        if name.startswith("NextChunkBytes") or name == "ChunksRemaining" or name == "LastBitrateKbps":
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

    def recurse(node, path_conds: list[sp.Expr], out_rules: list[sp.Expr], cls_label: str):
        if node.__class__.__name__ == "H2OLeafNode":
            if float(node.prediction) < float(leaf_purity):
                return
            consequent = sp.Eq(sym_var(label_col), sp.Integer(int(cls_label)))
            antecedent = sp.And(*path_conds) if path_conds else sp.true
            out_rules.append(sp.Implies(antecedent, consequent))
            return

        varname = str(node.split_feature)
        var = sym_var(varname)

        #* Categorical split
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

        recurse(node.left_child, path_conds + [cond_left], out_rules, cls_label)
        recurse(node.right_child, path_conds + [cond_right], out_rules, cls_label)

    label_domain = model._model_json["output"]["domains"][names.index(label_col)]
    if label_domain is None:
        raise RuntimeError(f"* H2O model does not expose a categorical domain for {label_col}")

    rules: list[sp.Expr] = []
    for cls in label_domain:
        tree = H2OTree(model=model, tree_number=0, tree_class=cls)
        recurse(tree.root_node, [], rules, str(cls))

    #* Deduplicate (SymPy Expr is hashable).
    return sorted(set(rules), key=lambda e: str(e))


def _save_rules(rules: list[sp.Expr], out_path: str) -> None:
    out_path = str(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        for r in rules:
            f.write(sp.srepr(r) + "\n")
    with open(out_path + ".json", "w") as f:
        json.dump([str(r) for r in rules], f, indent=0, sort_keys=True)


def _build_agg_dataset(steps: list[RolloutStep], *, label_col: str) -> pd.DataFrame:
    rows = []
    for s in steps:
        row = dict(s.features)
        row[label_col] = int(_bitrate_idx_to_kbps(int(s.action_idx)))
        rows.append(row)
    return pd.DataFrame(rows, columns=_feature_columns() + [label_col])


def main() -> None:
    parser = argparse.ArgumentParser(description="DAgger imitation for Pensieve PPO + H2O rule extraction")
    parser.add_argument("--teacher-model", required=True, help="* Path to pretrained PPO model (.pth) for teacher")
    parser.add_argument("--trace-dir", default="./train/", help="* Trace dir under `Pensieve-PPO/src/` (e.g., ./train/)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dagger-iters", type=int, default=100)
    parser.add_argument("--rollouts-per-iter", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=2_000_000)
    parser.add_argument("--train-frac", type=float, default=1.0)
    parser.add_argument("--teacher-select", choices=["argmax", "gumbel"], default="argmax")
    parser.add_argument("--student-backend", choices=["h2o", "sklearn"], default="h2o")

    parser.add_argument("--student-ntrees", type=int, default=1)
    parser.add_argument("--student-max-depth", type=int, default=200)
    parser.add_argument("--student-min-rows", type=int, default=1)
    parser.add_argument("--student-sample-rate", type=float, default=1.0)
    parser.add_argument("--student-mtries", type=int, default=None)
    parser.add_argument("--sklearn-max-depth", type=int, default=None)
    parser.add_argument("--sklearn-max-leaf-nodes", type=int, default=None)
    parser.add_argument("--sklearn-min-samples-leaf", type=int, default=1)

    parser.add_argument("--leaf-purity", type=float, default=0.999999)
    parser.add_argument(
        "--out-rules",
        default="rules/pensieve_dagger_rules.pl",
        help="* Output rule file path (repo-root relative by default)",
    )
    parser.add_argument("--eval-rollouts", type=int, default=10, help="* # evaluation episodes after DAgger (0 disables)")
    parser.add_argument("--eval-trace-dir", default=None, help="* Trace dir for evaluation (defaults to --trace-dir)")
    parser.add_argument("--wandb-project", default=None, help="* Weights & Biases project name (enables logging)")
    parser.add_argument("--wandb-name", default=None, help="* Weights & Biases run name")
    parser.add_argument("--wandb-tags", default="", help="* Comma-separated W&B tags")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")

    args = parser.parse_args()

    use_h2o = str(args.student_backend) == "h2o"
    h2o = None
    if use_h2o:
        import h2o  # type: ignore

        #* Silence H2O dependency warnings about optional multi-threaded pandas conversion.
        try:
            from h2o.exceptions import H2ODependencyWarning  # type: ignore

            warnings.filterwarnings("ignore", category=H2ODependencyWarning)
        except Exception:
            pass

        h2o.init()

    wandb_run = None
    try:
        rng = np.random.RandomState(int(args.seed))
        teacher = PPOTeacherPolicy(
            model_path=str(_resolve_repo_root(args.teacher_model)),
            action_selection=str(args.teacher_select),
        )
        env = PensieveRolloutEnv(trace_dir=str(args.trace_dir), seed=int(args.seed))
        if use_h2o:
            student = H2OStudentPolicy(
                seed=int(args.seed),
                ntrees=int(args.student_ntrees),
                max_depth=int(args.student_max_depth),
                min_rows=int(args.student_min_rows),
                sample_rate=float(args.student_sample_rate),
                mtries=args.student_mtries,
            )
        else:
            student = SklearnStudentPolicy(
                seed=int(args.seed),
                max_depth=args.sklearn_max_depth,
                max_leaf_nodes=args.sklearn_max_leaf_nodes,
                min_samples_leaf=int(args.sklearn_min_samples_leaf),
            )

        label_col = "BitrateKbps"
        categorical_cols = ["LastBitrateKbps"]

        if args.wandb_project and str(args.wandb_mode) != "disabled":
            import wandb  # type: ignore

            wandb_run = wandb.init(
                project=str(args.wandb_project),
                name=str(args.wandb_name) if args.wandb_name else None,
                tags=_parse_tags(str(args.wandb_tags)),
                mode=str(args.wandb_mode),
                config=vars(args),
            )

        #* Initial teacher dataset.
        agg_steps: list[RolloutStep] = []
        for _ in range(int(args.rollouts_per_iter)):
            agg_steps.extend(env.rollout(teacher, rng=rng))

        for it in range(int(args.dagger_iters)):
            #* Train on a bounded, deterministic subset.
            df = _build_agg_dataset(agg_steps, label_col=label_col)
            if len(df) > int(args.max_samples):
                idx = rng.choice(len(df), size=int(args.max_samples), replace=False)
                df = df.iloc[idx].reset_index(drop=True)
            student.train(df, label_col=label_col, categorical_cols=categorical_cols, train_frac=float(args.train_frac))

            #* Roll out student, then relabel visited states with the teacher.
            student_steps: list[RolloutStep] = []
            for _ in range(int(args.rollouts_per_iter)):
                student_steps.extend(env.rollout(student, rng=rng))

            states = np.stack([s.state for s in student_steps], axis=0) if student_steps else np.zeros((0, S_INFO, S_LEN))
            teacher_actions = []
            for s in states:
                teacher_actions.append(int(teacher.act(s, rng=rng)))
            relabeled = []
            for step, teacher_action in zip(student_steps, teacher_actions):
                relabeled.append(
                    RolloutStep(
                        state=step.state,
                        features=step.features,
                        action_idx=int(teacher_action),
                        reward=step.reward,
                    )
                )
            agg_steps.extend(relabeled)

        out_rules = _resolve_repo_root(str(args.out_rules))
        if use_h2o:
            rules = _extract_sympy_rules_from_h2o_tree(
                student.model,
                feature_cols=_feature_columns(),
                label_col=label_col,
                categorical_cols=categorical_cols,
                leaf_purity=float(args.leaf_purity),
            )
        else:
            rules = _extract_sympy_rules_from_sklearn_tree(
                student.model,
                feature_cols=_feature_columns(),
                label_col=label_col,
                leaf_purity=float(args.leaf_purity),
            )
        _save_rules(rules, out_rules)

        if int(args.eval_rollouts) > 0:
            eval_trace_dir = str(args.eval_trace_dir) if args.eval_trace_dir else str(args.trace_dir)
            teacher_eval_env = PensieveRolloutEnv(trace_dir=eval_trace_dir, seed=int(args.seed))
            student_eval_env = PensieveRolloutEnv(trace_dir=eval_trace_dir, seed=int(args.seed))
            eval_rng = np.random.RandomState(int(args.seed) + 1)

            teacher_values: dict[str, list[float]] = {"qoe_reward": [], "qoe_quality": [], "qoe_rebuf_s": [], "qoe_smoothness": []}
            student_values: dict[str, list[float]] = {"qoe_reward": [], "qoe_quality": [], "qoe_rebuf_s": [], "qoe_smoothness": []}

            global_step = 0
            for ep in range(int(args.eval_rollouts)):
                teacher_eval_env.reset_episode()
                student_eval_env.reset_episode()
                chunk = 0
                while True:
                    t_done, t_metrics = teacher_eval_env.step(teacher, rng=eval_rng)
                    s_done, s_metrics = student_eval_env.step(student, rng=eval_rng)
                    if bool(t_done) != bool(s_done):
                        raise RuntimeError("* Teacher/student episode ended at different chunk counts")

                    teacher_values["qoe_reward"].append(float(t_metrics["qoe_reward"]))
                    teacher_values["qoe_quality"].append(float(t_metrics["qoe/quality"]))
                    teacher_values["qoe_rebuf_s"].append(float(t_metrics["qoe/rebuf_s"]))
                    teacher_values["qoe_smoothness"].append(float(t_metrics["qoe/smoothness"]))

                    student_values["qoe_reward"].append(float(s_metrics["qoe_reward"]))
                    student_values["qoe_quality"].append(float(s_metrics["qoe/quality"]))
                    student_values["qoe_rebuf_s"].append(float(s_metrics["qoe/rebuf_s"]))
                    student_values["qoe_smoothness"].append(float(s_metrics["qoe/smoothness"]))

                    if wandb_run is not None:
                        row: dict[str, float | int | str] = {"eval/episode": int(ep), "eval/chunk": int(chunk)}
                        row.update({f"teacher/{k}": v for k, v in t_metrics.items()})
                        row.update({f"student/{k}": v for k, v in s_metrics.items()})
                        wandb_run.log(row, step=int(global_step))

                    global_step += 1
                    chunk += 1
                    if bool(t_done):
                        break

            def mean(xs: list[float]) -> float:
                return float(np.mean(xs)) if xs else 0.0

            summary = {
                "teacher/qoe_reward-mean": mean(teacher_values["qoe_reward"]),
                "teacher/qoe/quality-mean": mean(teacher_values["qoe_quality"]),
                "teacher/qoe/rebuf_s-mean": mean(teacher_values["qoe_rebuf_s"]),
                "teacher/qoe/smoothness-mean": mean(teacher_values["qoe_smoothness"]),
                "student/qoe_reward-mean": mean(student_values["qoe_reward"]),
                "student/qoe/quality-mean": mean(student_values["qoe_quality"]),
                "student/qoe/rebuf_s-mean": mean(student_values["qoe_rebuf_s"]),
                "student/qoe/smoothness-mean": mean(student_values["qoe_smoothness"]),
            }

            if wandb_run is not None:
                for k, v in summary.items():
                    wandb_run.summary[k] = v
            else:
                print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        try:
            if wandb_run is not None:
                wandb_run.finish()
            if use_h2o and h2o is not None:
                h2o.shutdown(prompt=False)
        except Exception:
            pass


if __name__ == "__main__":
    main()
