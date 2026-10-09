import sympy as sp
from rich import print as pprint
from z3 import *
import os
from functools import reduce


DEFAULT_TRACE_RULES_FILE = "rules/pensieve_trace_training_5steps.pl"

def _resolve_repo_root(path: str) -> str:
    if os.path.isabs(path):
        return path
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(repo_root, path)

# Environment constants (from fixed_env.py)
MILLISECONDS_IN_SECOND = 1000.0
B_IN_MB = 1000000.0
BITS_IN_BYTE = 8.0
VIDEO_CHUNCK_LEN = 4000.0  # millisec
BITRATE_LEVELS = 6
BUFFER_THRESH = 60.0 * MILLISECONDS_IN_SECOND
DRAIN_BUFFER_SLEEP_TIME = 500.0
PACKET_PAYLOAD_PORTION = 0.95
LINK_RTT = 80  # millisec

# Normalization factors
BUFFER_NORM_FACTOR = 10.0  # 10 seconds
M_IN_K = 1000.0

#* Relative tolerance for hard target matching (e.g., 0.10 => ±10% band).
TARGET_REL_TOL = 0.10

#* Hard cap on the maximum synthesized bandwidth (Mbps), regardless of trace-rule enforcement.
TRACE_BW_MAX_MBPS = 8.951518

#* Bandwidth smoothness bounds for adjacent samples:
#   bw_cur in [bw_prev * SMOOTHNESS_RATE_MIN, bw_prev * SMOOTHNESS_RATE_MAX]
SMOOTHNESS_RATE_MIN = 0.0
SMOOTHNESS_RATE_MAX = 5454.727273


def _realval(x: float) -> ArithRef:
    #* `RealVal` does not accept scientific notation; force fixed-point.
    from decimal import Decimal

    return RealVal(format(Decimal(str(float(x))), "f"))


_TRACE_RULES_CACHE: dict[
    tuple[str, int], tuple[list[tuple[str, BoolRef]], list[ArithRef], list[ArithRef]]
] = {}


def _get_canonical_trace_rules(
    rules_file: str, *, num_bw_samples: int
) -> tuple[list[tuple[str, BoolRef]], list[ArithRef], list[ArithRef]]:
    """
    Loads mined trace-dynamics rules and returns:
      (rules, canonical_bw_vars, canonical_duration_vars)
    where canonical vars are named:
      BandwidthMbps0..N-1 and DurationSec0..N-1.
    """
    key = (str(rules_file), int(num_bw_samples))
    if key in _TRACE_RULES_CACHE:
        return _TRACE_RULES_CACHE[key]

    n = int(num_bw_samples)
    if n <= 0:
        raise ValueError("* num_bw_samples must be positive")

    canonical_bw = [Real(f"BandwidthMbps{i}") for i in range(n)]
    canonical_dur = [Real(f"DurationSec{i}") for i in range(n)]

    varmap: dict[str, ArithRef] = {f"BandwidthMbps{i}": canonical_bw[i] for i in range(n)}
    varmap |= {f"DurationSec{i}": canonical_dur[i] for i in range(n)}

    def Symbol(name: str, *args, **kwargs):
        if name not in varmap:
            raise KeyError(f"Unknown trace rule variable: {name}")
        return varmap[name]

    def Integer(v):
        return IntVal(str(int(v)))

    def Float(v, *args, **kwargs):
        return _realval(float(v))

    def Mul(*args):
        return reduce(lambda a, b: a * b, args, _realval(1.0))

    def Add(*args):
        return reduce(lambda a, b: a + b, args, _realval(0.0))

    def Abs(x):
        return If(x >= 0, x, -x)

    def Or_(*args):
        return Or(*args)

    def And_(*args):
        return And(*args)

    def Equality(a, b):
        return a == b

    def Unequality(a, b):
        return a != b

    def StrictLessThan(a, b):
        return a < b

    def StrictGreaterThan(a, b):
        return a > b

    def LessThan(a, b):
        return a <= b

    def GreaterThan(a, b):
        return a >= b

    def Max_(*args):
        return reduce(lambda a, b: If(a >= b, a, b), args)

    def Min_(*args):
        return reduce(lambda a, b: If(a <= b, a, b), args)

    def Pow(a, b):
        return a**b

    eval_env = {
        "__builtins__": {},
        "Symbol": Symbol,
        "Integer": Integer,
        "Float": Float,
        "Mul": Mul,
        "Add": Add,
        "Abs": Abs,
        "Or": Or_,
        "And": And_,
        "Equality": Equality,
        "Unequality": Unequality,
        "StrictLessThan": StrictLessThan,
        "StrictGreaterThan": StrictGreaterThan,
        "LessThan": LessThan,
        "GreaterThan": GreaterThan,
        "Max": Max_,
        "Min": Min_,
        "Pow": Pow,
    }

    rules: list[tuple[str, BoolRef]] = []
    with open(str(rules_file), "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            expr = eval(line, eval_env)
            if not isinstance(expr, BoolRef):
                raise TypeError(f"* Trace rule must be BoolRef, got: {type(expr)}")
            #* Track rules using their string representation from the rules file.
            rule_repr = str(sp.sympify(line))
            rules.append((rule_repr, simplify(expr)))

    _TRACE_RULES_CACHE[key] = (rules, canonical_bw, canonical_dur)
    return rules, canonical_bw, canonical_dur


def _add_trace_rules_constraints(
    solver: Solver,
    *,
    rules_file: str,
    bw_vars: list[ArithRef],
    duration_vars: list[ArithRef],
    name_suffix: str = "",
) -> None:
    """
    Add mined trace-dynamics constraints to the solver, binding:
      BandwidthMbps{i} -> bw_vars[i]
      DurationSec{i}   -> duration_vars[i]
    """
    if len(bw_vars) != len(duration_vars):
        raise ValueError("* bw_vars and duration_vars must have the same length")

    rules, canonical_bw, canonical_dur = _get_canonical_trace_rules(
        _resolve_repo_root(str(rules_file)), num_bw_samples=len(bw_vars)
    )
    subs = [(canonical_bw[i], bw_vars[i]) for i in range(len(bw_vars))] + [
        (canonical_dur[i], duration_vars[i]) for i in range(len(duration_vars))
    ]
    suffix = str(name_suffix)
    if suffix:
        suffix = f" {suffix}"
    for rule_str, rule in rules:
        inst = substitute(rule, *subs)
        #* Track per-instantiation so unsat cores can identify which mined rule is responsible.
        solver.assert_and_track(inst, f"{rule_str}{suffix}")
    # print(f"✓ Added {len(rules)} trace rules constraints{suffix}")


def _add_basic_trace_physics_constraints(
    solver,
    *,
    bw_vars: list[ArithRef],
    duration_vars: list[ArithRef],
) -> None:
    #* When trace rules are disabled, we still enforce minimal "physics" constraints so the
    #* synthesized trace remains meaningful: bandwidth and durations must be non-negative, and
    #* bandwidth is globally capped to keep traces realistic.
    if len(bw_vars) != len(duration_vars):
        raise ValueError("* bw_vars and duration_vars must have the same length")
    bw_max = _realval(float(TRACE_BW_MAX_MBPS))
    for bw, dur in zip(bw_vars, duration_vars):
        solver.add(bw >= _realval(0.0))
        solver.add(bw <= bw_max)
        solver.add(dur >= _realval(0.0))


def _z3_max(a: ArithRef, b: ArithRef) -> ArithRef:
    return If(a >= b, a, b)


def _z3_min(a: ArithRef, b: ArithRef) -> ArithRef:
    return If(a <= b, a, b)


def _z3_max0(x: ArithRef) -> ArithRef:
    return If(x >= 0, x, _realval(0.0))


class DirectTraceWindowSynthesizer:
    """
    One-pass raw-trace SMT synthesis (replaces fixed-N per-chunk micro-traces).

    Key idea:
    - We synthesize ONE contiguous piecewise-constant "raw trace" of length `num_trace_segments`:
        (BandwidthMbps0..N-1, DurationSec0..N-1)
      and enforce the mined trace-dynamics rules on that trace once.
    - A chunk download then consumes a *prefix* of this trace (and may stop fractionally inside
      a segment), exactly like `fixed_env.Environment.get_video_chunk`.
    - For a multi-chunk window, we chain chunk downloads + optional sleep-time trace consumption
      on the SAME trace, so the resulting history is consistent.
    """

    def __init__(
        self,
        *,
        video_chunk_sizes: list[int],
        initial_buffer_ms: float,
        num_trace_segments: int,
        timeout_ms: int = 60000,
        duration_caps_s: list[float] | None = None,
        duration_cap_factor: float = 1.5,
        trace_rules_file: str = DEFAULT_TRACE_RULES_FILE,
        enforce_trace_rules: bool = True,
    ) -> None:
        self.video_chunk_sizes = [int(x) for x in list(video_chunk_sizes)]
        self.initial_buffer_ms = float(initial_buffer_ms)
        self.num_steps = int(len(self.video_chunk_sizes))
        self.num_trace_segments = int(num_trace_segments)
        self.duration_caps_s = None if duration_caps_s is None else [float(x) for x in list(duration_caps_s)]
        self.duration_cap_factor = float(duration_cap_factor)
        self.trace_rules_file = str(trace_rules_file)
        self.enforce_trace_rules = bool(enforce_trace_rules)

        if self.num_steps <= 0:
            raise ValueError("* video_chunk_sizes must be non-empty")
        if self.num_trace_segments <= 0:
            raise ValueError("* num_trace_segments must be positive")
        if self.duration_caps_s is not None and len(self.duration_caps_s) != self.num_steps:
            raise ValueError("* duration_caps_s length must match video_chunk_sizes length")

        # self.opt = Optimize()
        self.solver = Solver()
        self._timeout_ms = int(timeout_ms)
        self.solver.set(timeout=int(self._timeout_ms))

        #* Raw trace variables (Mbps, seconds). These are the ONLY "trace dynamics" variables.
        #* Terminology:
        #* - A "trace segment" is one piecewise-constant bandwidth interval in the raw Mahimahi trace.
        #* - A "video chunk" is the ABR-level unit downloaded each step (VIDEO_CHUNCK_LEN playback).
        #*   One chunk download may consume multiple trace segments and may stop fractionally inside
        #*   the last consumed trace segment.
        self.bw_vars = [Real(f"bw_{i}") for i in range(self.num_trace_segments)]
        self.duration_vars = [Real(f"duration_{i}") for i in range(self.num_trace_segments)]

        #* Always enforce basic physical constraints, regardless of mined-rule enforcement.
        _add_basic_trace_physics_constraints(self.solver, bw_vars=self.bw_vars, duration_vars=self.duration_vars)

        #^ Enforce mined trace rules on the contiguous raw trace window (optional).
        if bool(self.enforce_trace_rules):
            _add_trace_rules_constraints(
                self.solver,
                rules_file=self.trace_rules_file,
                bw_vars=self.bw_vars,
                duration_vars=self.duration_vars,
                name_suffix="[raw-trace]",
            )

        #& Segment boundaries in the synthesized raw trace (seconds).
        #* Formulation:
        #* Given durations duration_0..duration_{N-1}, define boundary times:
        #*   trace_seg_boundaries_s[0]   = 0
        #*   trace_seg_boundaries_s[i+1] = trace_seg_boundaries_s[i] + duration_i
        #* Then trace segment i occupies [trace_seg_boundaries_s[i], trace_seg_boundaries_s[i+1]].
        self.trace_seg_boundaries_s = [_realval(0.0)]
        for i in range(self.num_trace_segments):
            self.trace_seg_boundaries_s.append(self.trace_seg_boundaries_s[-1] + self.duration_vars[i])
        self.total_trace_time_s = self.trace_seg_boundaries_s[-1]

        #* Per-chunk download duration (RTT-free), sleep-time, and buffer recursion state.
        self.download_time_s = [Real(f"download_time_s_{t}") for t in range(self.num_steps)]
        self.sleep_steps = [Int(f"sleep_steps_{t}") for t in range(self.num_steps)]
        self.sleep_time_ms = [Real(f"sleep_time_ms_{t}") for t in range(self.num_steps)]
        self.buffer_before_ms = [Real(f"buffer_before_ms_{t}") for t in range(self.num_steps + 1)]

        self.solver.add(self.buffer_before_ms[0] == _realval(self.initial_buffer_ms))

        #& Build simulator constraints across the window on the SAME trace.
        self.start_time_s: list[ArithRef] = []
        self.end_time_s: list[ArithRef] = []
        self.delay_ms: list[ArithRef] = []
        self.buffer_after_ms: list[ArithRef] = []

        cur_start_s: ArithRef = _realval(0.0)

        #* Constant: bytes/sec per Mbps (payload bytes, i.e., after PACKET_PAYLOAD_PORTION).
        bytes_per_sec_per_mbps = _realval(float(B_IN_MB / BITS_IN_BYTE * PACKET_PAYLOAD_PORTION))

        for t in range(self.num_steps):
            chunk_bytes = int(self.video_chunk_sizes[t])
            chunk_bytes_real = _realval(float(chunk_bytes))

            dt_s = self.download_time_s[t]
            self.solver.add(dt_s >= _realval(0.0))

            start_s = cur_start_s
            end_s = start_s + dt_s

            self.start_time_s.append(start_s)
            self.end_time_s.append(end_s)

            #* Stay within the synthesized raw-trace horizon (no wrap-around).
            self.solver.assert_and_track(end_s <= self.total_trace_time_s, f"trace_horizon_{t}")

            #* Payload equation: integrate piecewise-constant bandwidth over [start_s, end_s].
            #* Formulation:
            #* Let [start_s, end_s] be the RTT-free download interval for this chunk.
            #* For each trace segment i with interval [a_i, b_i], the overlap duration is:
            #*   used_{t,i} = max(0, min(end_s, b_i) - max(start_s, a_i))
            #* Delivered payload bytes is then:
            #*   Σ_i bw_i * (B_IN_MB/BITS_IN_BYTE*PACKET_PAYLOAD_PORTION) * used_{t,i}
            #* and we enforce it equals the chunk size in bytes.
            delivered_bytes_terms: list[ArithRef] = []
            for i in range(self.num_trace_segments):
                trace_seg_start_s = self.trace_seg_boundaries_s[i]
                trace_seg_end_s = self.trace_seg_boundaries_s[i + 1]
                overlap_start_s = _z3_max(start_s, trace_seg_start_s)
                overlap_end_s = _z3_min(end_s, trace_seg_end_s)
                used_s = _z3_max0(overlap_end_s - overlap_start_s)
                delivered_bytes_terms.append(self.bw_vars[i] * bytes_per_sec_per_mbps * used_s)
            self.solver.assert_and_track(
                Sum(delivered_bytes_terms) == chunk_bytes_real,
                f"payload_{t}",
            )

            # #* Optional cap on RTT-free download duration (defensive, aligns with old solver).
            # if self.duration_caps_s is not None:
            #     cap_s = float(self.duration_cap_factor) * float(self.duration_caps_s[t])
            #     if np.isfinite(cap_s) and cap_s > 0.0:
            #         self.opt.assert_and_track(dt_s <= _realval(float(cap_s)), f"duration_cap_{t}")

            #* Delay includes RTT, but RTT does not advance trace time.
            delay_ms = dt_s * MILLISECONDS_IN_SECOND + LINK_RTT
            self.delay_ms.append(delay_ms)

            #* Sleep-time quantization (ceil) and buffer recursion (matches fixed_env.py).
            self.solver.add(self.sleep_steps[t] >= 0)
            self.solver.add(self.sleep_time_ms[t] == ToReal(self.sleep_steps[t]) * DRAIN_BUFFER_SLEEP_TIME)

            buffer_after_delay = If(
                self.buffer_before_ms[t] >= delay_ms,
                self.buffer_before_ms[t] - delay_ms,
                _realval(0.0),
            )
            buffer_after_add = buffer_after_delay + VIDEO_CHUNCK_LEN

            drain_ms = buffer_after_add - BUFFER_THRESH
            self.solver.add(
                Implies(
                    buffer_after_add > BUFFER_THRESH,
                    And(
                        self.sleep_time_ms[t] >= drain_ms,
                        self.sleep_time_ms[t] < drain_ms + DRAIN_BUFFER_SLEEP_TIME,
                    ),
                )
            )
            self.solver.add(Implies(buffer_after_add <= BUFFER_THRESH, self.sleep_time_ms[t] == _realval(0.0)))

            buffer_after = buffer_after_add - self.sleep_time_ms[t]
            self.buffer_after_ms.append(buffer_after)
            self.solver.add(self.buffer_before_ms[t + 1] == buffer_after)

            #* Advance trace time by (download_time + sleep_time). RTT does not advance.
            cur_start_s = end_s + self.sleep_time_ms[t] / MILLISECONDS_IN_SECOND

    def add_objectives(
        self,
        *,
        target_delay_norms: list[float],
        target_throughput_norms: list[float] | None = None,
        target_final_buffer_norm: float | None = None,
        enforce_throughput_targets: bool = False,
        enforce_final_buffer: bool = False,
        rel_tol: float = TARGET_REL_TOL,
    ) -> None:
        """
        Enforce hard target bands in normalized feature space.
        """
        tol = float(rel_tol)
        if tol < 0.0:
            tol = 0.0

        if len(target_delay_norms) != self.num_steps:
            raise ValueError("* target_delay_norms length must match video_chunk_sizes length")
        if target_throughput_norms is not None and len(target_throughput_norms) != self.num_steps:
            raise ValueError("* target_throughput_norms length must match video_chunk_sizes length")

        for t in range(self.num_steps):
            target_delay_ms = float(target_delay_norms[t]) * float(BUFFER_NORM_FACTOR) * float(M_IN_K)
            self.solver.assert_and_track(
                self.delay_ms[t] <= _realval((1.0 + tol) * target_delay_ms),
                f"delay_{t}_ub",
            )
            self.solver.assert_and_track(
                self.delay_ms[t] >= _realval((1.0 - tol) * target_delay_ms),
                f"delay_{t}_lb",
            )

            if target_throughput_norms is not None and bool(enforce_throughput_targets):
                target_rate = float(target_throughput_norms[t]) * float(M_IN_K)  # bytes/ms
                chunk_bytes = _realval(float(int(self.video_chunk_sizes[t])))
                self.solver.assert_and_track(
                    chunk_bytes <= _realval(1.0 + tol) * _realval(target_rate) * self.delay_ms[t],
                    f"throughput_{t}_ub",
                )
                self.solver.assert_and_track(
                    chunk_bytes >= _realval(1.0 - tol) * _realval(target_rate) * self.delay_ms[t],
                    f"throughput_{t}_lb",
                )

        if target_final_buffer_norm is not None and bool(enforce_final_buffer):
            target_buf_ms = float(target_final_buffer_norm) * float(BUFFER_NORM_FACTOR) * float(MILLISECONDS_IN_SECOND)
            self.solver.assert_and_track(
                self.buffer_after_ms[-1] <= _realval((1.0 + tol) * target_buf_ms),
                "buffer_final_ub",
            )
            self.solver.assert_and_track(
                self.buffer_after_ms[-1] >= _realval((1.0 - tol) * target_buf_ms),
                "buffer_final_lb",
            )

    def synthesize(self) -> tuple[list[float] | None, list[float] | None, bool]:
        result = self.solver.check()
        if result == unknown:
            print("\n","SMT solver returned UNKNOWN: ", self.solver.reason_unknown())
            return None, None, False
        if result == unsat:
            try:
                core = self.solver.unsat_core()
                print("\n", "=" * 20, f"{len(core)} UNSAT constraints", "=" * 20)
                pprint(core)
            except Exception:
                pass
            return None, None, False

        model = self.solver.model()

        def _to_float(val, default=0.0) -> float:
            if val is None:
                return float(default)
            if isinstance(val, RatNumRef):
                return float(val.as_fraction())
            try:
                s = val.as_decimal(30)
                if s.endswith("?"):
                    s = s[:-1]
                return float(s)
            except Exception:
                return float(default)

        bw = [_to_float(model.evaluate(v, model_completion=True), default=1.0) for v in self.bw_vars]
        tt = [_to_float(model.evaluate(v, model_completion=True), default=0.1) for v in self.duration_vars]
        return bw, tt, True

def synthesize_bandwidth_trace(
    *,
    video_chunk_size: int,
    initial_buffer_ms: float,
    target_buffer_norm: float,
    target_throughput_norm: float,
    target_delay_norm: float,
    num_bw_samples: int = 20,
    enforce_buffer_target: bool = True,
    enforce_throughput_target: bool = True,
    enforce_trace_rules: bool = True,
    rel_tol: float = TARGET_REL_TOL,
) -> tuple[list[float] | None, list[float] | None]:
    #* One chunk consumes a *prefix* of the synthesized raw trace (may end fractionally inside a segment),
    #  matching the Mahimahi download loop in `fixed_env.Environment.get_video_chunk`.
    synthesizer = DirectTraceWindowSynthesizer(
        video_chunk_sizes=[int(video_chunk_size)],
        initial_buffer_ms=float(initial_buffer_ms),
        num_trace_segments=int(num_bw_samples),
        timeout_ms=60000,
        duration_caps_s=None,
        duration_cap_factor=1.5,
        trace_rules_file=str(DEFAULT_TRACE_RULES_FILE),
        enforce_trace_rules=bool(enforce_trace_rules),
    )
    synthesizer.add_objectives(
        target_delay_norms=[float(target_delay_norm)],
        target_throughput_norms=([float(target_throughput_norm)] if bool(enforce_throughput_target) else None),
        target_final_buffer_norm=float(target_buffer_norm) if bool(enforce_buffer_target) else None,
        enforce_throughput_targets=bool(enforce_throughput_target),
        enforce_final_buffer=bool(enforce_buffer_target),
        rel_tol=float(rel_tol),
    )

    bw_trace, t_trace, success = synthesizer.synthesize()
    if not bool(success) or bw_trace is None or t_trace is None:
        return None, None

    return [float(x) for x in bw_trace], [float(x) for x in t_trace]
