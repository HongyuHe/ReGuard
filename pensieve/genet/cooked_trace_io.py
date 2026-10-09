from __future__ import annotations

import os


def is_cooked_trace_file(path: str) -> bool:
    name = os.path.basename(str(path))
    if str(name).startswith("."):
        return False
    if str(name).endswith(".json") or str(name).endswith(".jsonl"):
        return False
    if not os.path.isfile(path):
        return False

    with open(str(path), "rb") as f:
        for line in f:
            parse = line.split()
            if not parse:
                continue
            if len(parse) < 2:
                return False
            try:
                float(parse[0])
                float(parse[1])
            except ValueError:
                return False
            return True
    return False


def list_cooked_trace_files(trace_dir: str) -> list[str]:
    out: list[str] = []
    for name in sorted(os.listdir(str(trace_dir))):
        path = os.path.join(str(trace_dir), str(name))
        if not is_cooked_trace_file(path):
            continue
        out.append(str(name))
    return out


def load_cooked_traces_dir(trace_dir: str, *, limit: int = 0) -> tuple[list[list[float]], list[list[float]], list[str]]:
    files = list_cooked_trace_files(str(trace_dir))
    if int(limit) > 0:
        files = list(files[: int(limit)])

    all_cooked_time: list[list[float]] = []
    all_cooked_bw: list[list[float]] = []
    all_names: list[str] = []
    for name in files:
        path = os.path.join(str(trace_dir), str(name))
        cooked_time: list[float] = []
        cooked_bw: list[float] = []
        with open(str(path), "rb") as f:
            for line in f:
                parse = line.split()
                if len(parse) < 2:
                    raise ValueError(f"* invalid cooked trace line in {path}")
                cooked_time.append(float(parse[0]))
                cooked_bw.append(float(parse[1]))
        all_cooked_time.append(cooked_time)
        all_cooked_bw.append(cooked_bw)
        all_names.append(str(name))
    return all_cooked_time, all_cooked_bw, all_names
