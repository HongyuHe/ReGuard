#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-$ROOT/venv/bin/python}"
# PREFIX="${PREFIX:-hotnets19}"
# PREFIX="${PREFIX:-pgd-feat}"
# PREFIX="${PREFIX:-clean}"
PREFIX="${PREFIX:-shielded-iter4}"
# TRACE_DIR="${TRACE_DIR:-traces/online-adv-traces/pensieve}"
# TRACE_DIR="${TRACE_DIR:-traces/finetune/hotnets19}"
# TRACE_DIR="${TRACE_DIR:-traces/finetune/pgd-feat-0.1a-20i}"
# TRACE_DIR="${TRACE_DIR:-traces/online/pgd-surrogate}"
# TRACE_DIR="${TRACE_DIR:-Pensieve-PPO/src/train}"
# TRACE_DIR="${TRACE_DIR:-traces/finetune/pgd-feat-step-0.1a-20i}"
TRACE_DIR="${TRACE_DIR:-traces/shielded/iter4}"
MODEL="${MODEL:-Pensieve-PPO/src/ppo/nn_model_ep_426300.pth}"

K_STEPS="${K_STEPS:-5}"
TRACE_NUM="${TRACE_NUM:-0}"
MAX_LOGGED_STATES="${MAX_LOGGED_STATES:-0}"
ACTION_SELECTION="${ACTION_SELECTION:-argmax}"
SEED="${SEED:-42}"
PROGRESS_EVERY="${PROGRESS_EVERY:-400}"

OUT_DIR="${OUT_DIR:-rules/shield/data}"
DATA_CSV="${DATA_CSV:-$OUT_DIR/${PREFIX}.csv}"
LABELED_CSV="${LABELED_CSV:-$OUT_DIR/${PREFIX}_labeled.csv}"

GAP_PCT_OPEN_THRESHOLD="${GAP_PCT_OPEN_THRESHOLD:-75}"
REGRET_DELTA="${REGRET_DELTA:-10}"
REGRET_DELTA_MIN_ABS="${REGRET_DELTA_MIN_ABS:-0.3}"
REGRET_DELTA_MAX_ABS="${REGRET_DELTA_MAX_ABS:-0.5}"
ORACLE_BEST_MIN_ABS="${ORACLE_BEST_MIN_ABS:-0.1}"
NONCONTIGUOUS="${NONCONTIGUOUS:-best_action}"

BACKEND="${BACKEND:-h2o}" # h2o | sklearn
LEAF_PURITY="${LEAF_PURITY:-0.999999}"
TRAIN_FRAC="${TRAIN_FRAC:-1.0}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"

GAP_TAG="${GAP_TAG:-${GAP_PCT_OPEN_THRESHOLD//./p}}"
REGRET_TAG="${REGRET_TAG:-${REGRET_DELTA//./p}}"
REGRET_MIN_TAG="${REGRET_MIN_TAG:-${REGRET_DELTA_MIN_ABS//./p}}"
REGRET_MAX_TAG="${REGRET_MAX_TAG:-${REGRET_DELTA_MAX_ABS//./p}}"
OUT_PREFIX_DEFAULT="rules/shield/out/${PREFIX}_bounds_${BACKEND}_gap${GAP_TAG}_delta${REGRET_TAG}_dmin${REGRET_MIN_TAG}_dmax${REGRET_MAX_TAG}"
OUT_PREFIX="${OUT_PREFIX:-$OUT_PREFIX_DEFAULT}"

SKLEARN_MAX_DEPTH="${SKLEARN_MAX_DEPTH:-}"
SKLEARN_MAX_LEAF_NODES="${SKLEARN_MAX_LEAF_NODES:-}"
SKLEARN_MIN_SAMPLES_LEAF="${SKLEARN_MIN_SAMPLES_LEAF:-1}"

H2O_NTREES="${H2O_NTREES:-1}"
H2O_MAX_DEPTH="${H2O_MAX_DEPTH:-200}"
H2O_MIN_ROWS="${H2O_MIN_ROWS:-1}"
H2O_SAMPLE_RATE="${H2O_SAMPLE_RATE:-1.0}"
H2O_MTRIES="${H2O_MTRIES:-}"

mkdir -p "$OUT_DIR"
mkdir -p "$(dirname "$OUT_PREFIX")"

echo "Dataset: $DATA_CSV"
"$PY" rules/shield/gen_rule_mining_dataset.py \
  --trace-dir "$TRACE_DIR" \
  --model "$MODEL" \
  --out "$DATA_CSV" \
  --k-steps "$K_STEPS" \
  --trace-num "$TRACE_NUM" \
  --max-logged-states "$MAX_LOGGED_STATES" \
  --action-selection "$ACTION_SELECTION" \
  --seed "$SEED" \
  --progress-every "$PROGRESS_EVERY"

echo "Labeling: $LABELED_CSV"
"$PY" rules/shield/label_bounds.py \
  --in "$DATA_CSV" \
  --out "$LABELED_CSV" \
  --gap-pct-open-threshold "$GAP_PCT_OPEN_THRESHOLD" \
  --oracle-best-min-abs "$ORACLE_BEST_MIN_ABS" \
  --regret-delta "$REGRET_DELTA" \
  --regret-delta-min-abs "$REGRET_DELTA_MIN_ABS" \
  --regret-delta-max-abs "$REGRET_DELTA_MAX_ABS" \
  --noncontiguous "$NONCONTIGUOUS"

echo "Training + extracting rules: ${OUT_PREFIX}_{min,max}.pl"
TRAIN_CMD=( "$PY" rules/shield/train_bounds_dt.py
  --data "$LABELED_CSV"
  --backend "$BACKEND"
  --seed "$SEED"
  --train-frac "$TRAIN_FRAC"
  --max-samples "$MAX_SAMPLES"
  --leaf-purity "$LEAF_PURITY"
  --out-prefix "$OUT_PREFIX"
)

if [[ "$BACKEND" == "sklearn" ]]; then
  if [[ -n "$SKLEARN_MAX_DEPTH" ]]; then
    TRAIN_CMD+=( --sklearn-max-depth "$SKLEARN_MAX_DEPTH" )
  fi
  if [[ -n "$SKLEARN_MAX_LEAF_NODES" ]]; then
    TRAIN_CMD+=( --sklearn-max-leaf-nodes "$SKLEARN_MAX_LEAF_NODES" )
  fi
  TRAIN_CMD+=( --sklearn-min-samples-leaf "$SKLEARN_MIN_SAMPLES_LEAF" )
else
  TRAIN_CMD+=( --h2o-ntrees "$H2O_NTREES" )
  TRAIN_CMD+=( --h2o-max-depth "$H2O_MAX_DEPTH" )
  TRAIN_CMD+=( --h2o-min-rows "$H2O_MIN_ROWS" )
  TRAIN_CMD+=( --h2o-sample-rate "$H2O_SAMPLE_RATE" )
  if [[ -n "$H2O_MTRIES" ]]; then
    TRAIN_CMD+=( --h2o-mtries "$H2O_MTRIES" )
  fi
fi

"${TRAIN_CMD[@]}"
