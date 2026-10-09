from __future__ import annotations

from typing import Any

import numpy as np


def infer_num_servers_from_obs(obs: np.ndarray) -> int:
    arr = np.asarray(obs, dtype=np.float64).reshape(-1)
    if arr.size < 2:
        raise ValueError("Park LB observation must contain server loads plus incoming job size")
    return int(arr.size - 1)


def feature_columns(num_servers: int, *, include_derived: bool = True) -> list[str]:
    n = int(num_servers)
    cols = [f"LoadProxy{i}" for i in range(n)]
    cols.append("IncomingSize")
    if bool(include_derived):
        cols.extend(["TotalLoadProxy", "MeanLoadProxy", "MinLoadProxy", "MaxLoadProxy", "StdLoadProxy"])
        cols.extend([f"LoadProxyGapToMin{i}" for i in range(n)])
        cols.extend([f"LoadProxyRank{i}" for i in range(n)])
        cols.extend([f"LoadProxyPlusIncoming{i}" for i in range(n)])
    return cols


def features_from_obs(
    obs: np.ndarray,
    *,
    num_servers: int | None = None,
    include_derived: bool = True,
) -> dict[str, float | int]:
    arr = np.asarray(obs, dtype=np.float64).reshape(-1)
    n = infer_num_servers_from_obs(arr) if num_servers is None else int(num_servers)
    if arr.size != int(n) + 1:
        raise ValueError(f"expected observation length {int(n) + 1}, got {arr.size}")

    loads = np.asarray(arr[:n], dtype=np.float64)
    incoming = float(arr[n])
    out: dict[str, float | int] = {}
    for i in range(n):
        out[f"LoadProxy{i}"] = float(loads[i])
    out["IncomingSize"] = float(incoming)

    if bool(include_derived):
        min_load = float(np.min(loads)) if loads.size > 0 else 0.0
        order = np.argsort(loads, kind="stable")
        ranks = np.empty_like(order)
        ranks[order] = np.arange(n)
        out["TotalLoadProxy"] = float(np.sum(loads))
        out["MeanLoadProxy"] = float(np.mean(loads)) if loads.size > 0 else 0.0
        out["MinLoadProxy"] = float(min_load)
        out["MaxLoadProxy"] = float(np.max(loads)) if loads.size > 0 else 0.0
        out["StdLoadProxy"] = float(np.std(loads)) if loads.size > 0 else 0.0
        for i in range(n):
            out[f"LoadProxyGapToMin{i}"] = float(loads[i] - min_load)
            out[f"LoadProxyRank{i}"] = int(ranks[i])
            out[f"LoadProxyPlusIncoming{i}"] = float(loads[i] + incoming)
    return out


def feature_row_from_obs(
    obs: np.ndarray,
    *,
    num_servers: int | None = None,
    include_derived: bool = True,
) -> dict[str, Any]:
    n = infer_num_servers_from_obs(obs) if num_servers is None else int(num_servers)
    vals = features_from_obs(obs, num_servers=int(n), include_derived=bool(include_derived))
    cols = feature_columns(int(n), include_derived=bool(include_derived))
    return {str(col): vals[str(col)] for col in list(cols)}
