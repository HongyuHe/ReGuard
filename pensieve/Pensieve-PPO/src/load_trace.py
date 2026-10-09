import os


COOKED_TRACE_FOLDER = './train/'


def load_trace(cooked_trace_folder=COOKED_TRACE_FOLDER):
    #* Deterministic ordering so trace sampling is reproducible across runs.
    #* Two-pass trace directories include `manifest.json`; ignore non-trace files.
    cooked_trace_folder = os.path.expandvars(os.path.expanduser(str(cooked_trace_folder)))
    cooked_files = []
    for name in sorted(os.listdir(cooked_trace_folder)):
        if name.startswith("."):
            continue
        path = os.path.join(cooked_trace_folder, name)
        if not _is_cooked_trace_file(path):
            continue
        cooked_files.append(name)
    all_cooked_time = []
    all_cooked_bw = []
    all_file_names = []
    for cooked_file in cooked_files:
        file_path = os.path.join(cooked_trace_folder, cooked_file)
        cooked_time = []
        cooked_bw = []
        # print file_path
        with open(file_path, 'rb') as f:
            for line in f:
                parse = line.split()
                cooked_time.append(float(parse[0]))
                cooked_bw.append(float(parse[1]))
        all_cooked_time.append(cooked_time)
        all_cooked_bw.append(cooked_bw)
        all_file_names.append(cooked_file)

    return all_cooked_time, all_cooked_bw, all_file_names


def _is_cooked_trace_file(path):
    #* Accepts legacy cooked traces with arbitrary suffixes while excluding metadata sidecars.
    name = os.path.basename(str(path))
    if str(name).endswith(".json") or str(name).endswith(".jsonl"):
        return False
    if not os.path.isfile(path):
        return False

    with open(str(path), 'rb') as f:
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
