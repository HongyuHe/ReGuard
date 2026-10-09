#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# ---------------------------
# Config (override via env)
# ---------------------------
VICTIM_CKPT="${VICTIM_CKPT:-$ROOT_DIR/runs/victim_clean_v1/best.pt}"

CLEAN_DIR="${CLEAN_DIR:-$ROOT_DIR/traces/workload_benign_stable}"
CLEAN_NUM_TRACES="${CLEAN_NUM_TRACES:-20}"
CLEAN_NUM_JOBS="${CLEAN_NUM_JOBS:-1000}"

NUM_SERVERS="${NUM_SERVERS:-10}"
EPISODE_JOBS="${EPISODE_JOBS:-1000}"
EPISODES="${EPISODES:-20}"
SEED="${SEED:-42}"
REFERENCE_WINDOW="${REFERENCE_WINDOW:-${ORACLE_WINDOW:-5}}"

DT_MIN="${DT_MIN:-1}"
DT_MAX="${DT_MAX:-500}"
SIZE_MIN="${SIZE_MIN:-10}"
SIZE_MAX="${SIZE_MAX:-5000}"
AUTO_BOUNDS_FROM_CLEAN="${AUTO_BOUNDS_FROM_CLEAN:-1}"

TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-300000}"
N_STEPS="${N_STEPS:-2048}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-3e-4}"

MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/online/models}"
ADV_TRACE_DIR="${ADV_TRACE_DIR:-$ROOT_DIR/online/adv-traces}"
EVAL_DIR="${EVAL_DIR:-$ROOT_DIR/online/eval}"
EVAL_RANDOM_DIR="${EVAL_RANDOM_DIR:-$ROOT_DIR/online/eval_random}"
EVAL_ATTACKER_VS_RANDOM_DIR="${EVAL_ATTACKER_VS_RANDOM_DIR:-$ROOT_DIR/online/eval_rl_attacker_vs_random}"

PYTHON_BIN="${PYTHON_BIN:-python}"

# Optional comma-separated service rates:
# SERVICE_RATES="0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95,1.05"
SERVICE_RATES="${SERVICE_RATES:-}"

if [[ ! -f "$VICTIM_CKPT" ]]; then
  echo "[error] victim checkpoint not found: $VICTIM_CKPT"
  exit 1
fi

mkdir -p "$MODEL_DIR" "$ADV_TRACE_DIR" "$EVAL_DIR" "$EVAL_RANDOM_DIR" "$EVAL_ATTACKER_VS_RANDOM_DIR"

# Build optional service-rate args once.
SR_ARGS=()
if [[ -n "$SERVICE_RATES" ]]; then
  SR_ARGS+=(--service-rates "$SERVICE_RATES")
fi

echo "[1/7] Ensure clean traces exist"
if [[ ! -d "$CLEAN_DIR" ]] || [[ -z "$(find "$CLEAN_DIR" -maxdepth 1 -type f \( -name '*.csv' -o -name '*.json' \) | head -n 1)" ]]; then
  echo "[info] generating clean traces at: $CLEAN_DIR"
  "$PYTHON_BIN" "$ROOT_DIR/scripts/generate_job_traces.py" \
    --out-dir "$CLEAN_DIR" \
    --num-traces "$CLEAN_NUM_TRACES" \
    --num-jobs "$CLEAN_NUM_JOBS" \
    --job-interval 55 \
    --job-size-pareto-shape 1.5 \
    --job-size-pareto-scale 100 \
    --seed "$SEED" \
    --time-mode absolute
else
  echo "[info] reusing existing clean traces in: $CLEAN_DIR"
fi

if [[ "$AUTO_BOUNDS_FROM_CLEAN" == "1" ]]; then
  echo "[info] auto-inferring attacker bounds from clean traces (q99-based)"
  BOUNDS="$("$PYTHON_BIN" - "$CLEAN_DIR" "$DT_MIN" "$DT_MAX" "$SIZE_MIN" "$SIZE_MAX" <<'PY'
import csv,glob,os,sys
clean_dir=sys.argv[1]
dt_min=float(sys.argv[2]); dt_max=float(sys.argv[3]); size_min=float(sys.argv[4]); size_max=float(sys.argv[5])
files=[p for p in sorted(glob.glob(os.path.join(clean_dir,'*.csv'))) if os.path.basename(p).lower()!='index.csv']
if not files:
    print(f"{dt_min} {dt_max} {size_min} {size_max}")
    raise SystemExit
sizes=[]; dts=[]
for p in files:
    last=None
    with open(p,newline='') as f:
        r=csv.DictReader(f)
        for row in r:
            s=row.get('job_size',row.get('size'))
            if s is not None:
                v=float(s)
                if v>0: sizes.append(v)
            at=row.get('arrival_time')
            if at is not None and at!='':
                t=float(at)
                if last is not None:
                    d=max(0.0,t-last); dts.append(d)
                last=t
if not sizes:
    print(f"{dt_min} {dt_max} {size_min} {size_max}")
    raise SystemExit
sizes.sort(); dts.sort() if dts else None
def q(arr,p):
    if not arr: return 0.0
    i=max(0,min(len(arr)-1,int(p*len(arr))-1))
    return float(arr[i])
size_q99=q(sizes,0.99)
dt_q99=q(dts,0.99) if dts else dt_max
# Expand modestly beyond q99 to preserve adversarial headroom while avoiding extreme outliers.
new_size_max=max(size_min, min(max(size_max, size_q99*2.0), 20000.0))
new_dt_max=max(dt_min, min(max(dt_max, dt_q99*1.25), 2000.0))
print(f"{dt_min} {new_dt_max:.0f} {size_min} {new_size_max:.0f}")
PY
)"
  DT_MIN="$(echo "$BOUNDS" | awk '{print $1}')"
  DT_MAX="$(echo "$BOUNDS" | awk '{print $2}')"
  SIZE_MIN="$(echo "$BOUNDS" | awk '{print $3}')"
  SIZE_MAX="$(echo "$BOUNDS" | awk '{print $4}')"
  echo "[info] attacker bounds -> dt:[${DT_MIN}, ${DT_MAX}] size:[${SIZE_MIN}, ${SIZE_MAX}]"
fi

echo "[2/7] Train RL attacker"
"$PYTHON_BIN" -m online.train_adv \
  --victim-checkpoint "$VICTIM_CKPT" \
  --num-servers "$NUM_SERVERS" \
  --episode-jobs "$EPISODE_JOBS" \
  --dt-min "$DT_MIN" \
  --dt-max "$DT_MAX" \
  --size-min "$SIZE_MIN" \
  --size-max "$SIZE_MAX" \
  --reference-window "$REFERENCE_WINDOW" \
  --total-timesteps "$TOTAL_TIMESTEPS" \
  --n-steps "$N_STEPS" \
  --batch-size "$BATCH_SIZE" \
  --learning-rate "$LR" \
  --seed "$SEED" \
  --out-dir "$MODEL_DIR" \
  "${SR_ARGS[@]}"

ADV_MODEL="$(ls -1t "$MODEL_DIR"/adv_ppo_lb_*.zip | head -n 1 || true)"
if [[ -z "$ADV_MODEL" ]]; then
  echo "[error] could not find trained attacker model in: $MODEL_DIR"
  exit 1
fi

echo "[3/7] Generate RL-attacker traces"
"$PYTHON_BIN" "$ROOT_DIR/scripts/generate_rl_attacker_traces.py" \
  --victim-checkpoint "$VICTIM_CKPT" \
  --attacker-model "$ADV_MODEL" \
  --episodes "$EPISODES" \
  --num-servers "$NUM_SERVERS" \
  --episode-jobs "$EPISODE_JOBS" \
  --dt-min "$DT_MIN" \
  --dt-max "$DT_MAX" \
  --size-min "$SIZE_MIN" \
  --size-max "$SIZE_MAX" \
  --reference-window "$REFERENCE_WINDOW" \
  --seed "$SEED" \
  --out-dir "$ADV_TRACE_DIR/RL-attacker" \
  "${SR_ARGS[@]}"

echo "[4/7] Generate random baseline traces"
"$PYTHON_BIN" "$ROOT_DIR/scripts/generate_uniform_random_traces.py" \
  --out-dir "$ADV_TRACE_DIR/random" \
  --num-traces "$EPISODES" \
  --num-servers "$NUM_SERVERS" \
  --num-jobs "$EPISODE_JOBS" \
  --dt-min-ms "$DT_MIN" \
  --dt-max-ms "$DT_MAX" \
  --job-size-min-bytes "$SIZE_MIN" \
  --job-size-max-bytes "$SIZE_MAX" \
  --seed "$SEED"

echo "[5/7] Evaluate oracle gap: clean vs RL-attacker"
"$PYTHON_BIN" -m online.eval_oracle_gap \
  --victim-checkpoint "$VICTIM_CKPT" \
  --clean-dir "$CLEAN_DIR" \
  --adv-dir "$ADV_TRACE_DIR/RL-attacker" \
  --num-servers "$NUM_SERVERS" \
  --reference-window "$REFERENCE_WINDOW" \
  --out-dir "$EVAL_DIR" \
  "${SR_ARGS[@]}"

echo "[6/7] Evaluate oracle gap: clean vs random"
"$PYTHON_BIN" -m online.eval_oracle_gap \
  --victim-checkpoint "$VICTIM_CKPT" \
  --clean-dir "$CLEAN_DIR" \
  --adv-dir "$ADV_TRACE_DIR/random" \
  --num-servers "$NUM_SERVERS" \
  --reference-window "$REFERENCE_WINDOW" \
  --out-dir "$EVAL_RANDOM_DIR" \
  "${SR_ARGS[@]}"

echo "[7/7] Evaluate oracle gap: RL-attacker vs random (primary fairness check)"
"$PYTHON_BIN" -m online.eval_oracle_gap \
  --victim-checkpoint "$VICTIM_CKPT" \
  --clean-dir "$ADV_TRACE_DIR/random" \
  --adv-dir "$ADV_TRACE_DIR/RL-attacker" \
  --num-servers "$NUM_SERVERS" \
  --reference-window "$REFERENCE_WINDOW" \
  --out-dir "$EVAL_ATTACKER_VS_RANDOM_DIR" \
  "${SR_ARGS[@]}"

echo ""
echo "Done."
echo "Attacker model: $ADV_MODEL"
echo "RL-attacker eval summary: $EVAL_DIR/oracle_gap_summary.json"
echo "Random eval summary: $EVAL_RANDOM_DIR/oracle_gap_summary.json"
echo "Primary (RL-attacker vs random) summary: $EVAL_ATTACKER_VS_RANDOM_DIR/oracle_gap_summary.json"
