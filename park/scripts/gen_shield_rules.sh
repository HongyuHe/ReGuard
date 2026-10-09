#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-$ROOT/.venv/bin/python}"
PREFIX="${PREFIX:-shielded-iter3}"
TRACE_DIR="${TRACE_DIR:-traces/shielded-iter3}"
MODEL="${MODEL:-models/parklb_v1_350k.pt}"

REFERENCE_WINDOW="${REFERENCE_WINDOW:-5}"
TRACE_LIMIT="${TRACE_LIMIT:-0}"
MAX_LOGGED_STATES="${MAX_LOGGED_STATES:-0}"
SEED="${SEED:-42}"
PROGRESS_EVERY="${PROGRESS_EVERY:-25}"

OUT_DIR="${OUT_DIR:-shield/data}"
RULE_DIR="${RULE_DIR:-shield/out}"
DATA_CSV="${DATA_CSV:-$OUT_DIR/${PREFIX}.csv}"
LABELED_CSV="${LABELED_CSV:-$OUT_DIR/${PREFIX}_labeled.csv}"
REGENERATE_DATASETS="${REGENERATE_DATASETS:-0}"

RISK_GAP_PCT_THRESHOLD="${RISK_GAP_PCT_THRESHOLD:-30}"
RISK_GAP_MIN_ABS="${RISK_GAP_MIN_ABS:-0}"
ACTION_REGRET_PCT="${ACTION_REGRET_PCT:-10}"
ACTION_REGRET_MIN_ABS="${ACTION_REGRET_MIN_ABS:-0}"
ACTION_REGRET_MAX_ABS="${ACTION_REGRET_MAX_ABS:-0}"
ALLOW_SOURCE="${ALLOW_SOURCE:-policies}"

BACKEND="${BACKEND:-h2o}"
LEAF_PURITY="${LEAF_PURITY:-0.999999}"
TRAIN_FRAC="${TRAIN_FRAC:-1.0}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
RAW_ONLY="${RAW_ONLY:-0}"
TRAIN_ALLOW_ON_ALL_STATES="${TRAIN_ALLOW_ON_ALL_STATES:-0}"

SKLEARN_MAX_DEPTH="${SKLEARN_MAX_DEPTH:-}"
SKLEARN_MAX_LEAF_NODES="${SKLEARN_MAX_LEAF_NODES:-}"
SKLEARN_MIN_SAMPLES_LEAF="${SKLEARN_MIN_SAMPLES_LEAF:-1}"

H2O_NTREES="${H2O_NTREES:-1}"
H2O_MAX_DEPTH="${H2O_MAX_DEPTH:-20}"
H2O_MIN_ROWS="${H2O_MIN_ROWS:-1}"
H2O_SAMPLE_RATE="${H2O_SAMPLE_RATE:-1.0}"
H2O_MTRIES="${H2O_MTRIES:-}"

GAP_TAG="${GAP_TAG:-${RISK_GAP_PCT_THRESHOLD//./p}}"
REGRET_TAG="${REGRET_TAG:-${ACTION_REGRET_PCT//./p}}"
OUT_PREFIX="${OUT_PREFIX:-$RULE_DIR/${PREFIX}_shield_${BACKEND}_gap${GAP_TAG}_regret${REGRET_TAG}}"

mkdir -p "$OUT_DIR" "$RULE_DIR"

GEN_CMD=( "$PY" shield/gen_rule_mining_dataset.py
  --victim-checkpoint "$MODEL"
  --trace-dir "$TRACE_DIR"
  --out "$DATA_CSV"
  --reference-window "$REFERENCE_WINDOW"
  --trace-limit "$TRACE_LIMIT"
  --max-logged-states "$MAX_LOGGED_STATES"
  --seed "$SEED"
  --progress-every "$PROGRESS_EVERY"
)
if [[ "$RAW_ONLY" == "1" ]]; then
  GEN_CMD+=( --raw-only )
fi
if [[ "$REGENERATE_DATASETS" == "1" ]]; then
  GEN_CMD+=( --overwrite )
fi

echo "Dataset: $DATA_CSV"
if [[ -f "$DATA_CSV" && "$REGENERATE_DATASETS" != "1" ]]; then
  echo "Dataset already exists; reusing it. Set REGENERATE_DATASETS=1 to rebuild."
else
  "${GEN_CMD[@]}"
fi

echo "Labeling: $LABELED_CSV"
if [[ -f "$LABELED_CSV" && "$REGENERATE_DATASETS" != "1" ]]; then
  echo "Labeled dataset already exists; reusing it. Set REGENERATE_DATASETS=1 to rebuild."
else
  "$PY" shield/label_risk_and_actions.py \
    --in "$DATA_CSV" \
    --out "$LABELED_CSV" \
    --risk-gap-pct-threshold "$RISK_GAP_PCT_THRESHOLD" \
    --risk-gap-min-abs "$RISK_GAP_MIN_ABS" \
    --action-regret-pct "$ACTION_REGRET_PCT" \
    --action-regret-min-abs "$ACTION_REGRET_MIN_ABS" \
    --action-regret-max-abs "$ACTION_REGRET_MAX_ABS" \
    --allow-source "$ALLOW_SOURCE"
fi

TRAIN_CMD=( "$PY" shield/train_obs_only_dt.py
  --data "$LABELED_CSV"
  --backend "$BACKEND"
  --seed "$SEED"
  --train-frac "$TRAIN_FRAC"
  --max-samples "$MAX_SAMPLES"
  --leaf-purity "$LEAF_PURITY"
  --out-prefix "$OUT_PREFIX"
)
if [[ "$RAW_ONLY" == "1" ]]; then
  TRAIN_CMD+=( --raw-only )
fi
if [[ "$TRAIN_ALLOW_ON_ALL_STATES" == "1" ]]; then
  TRAIN_CMD+=( --train-allow-on-all-states )
fi
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

echo "Training + extracting rules: ${OUT_PREFIX}_{risk,allow_server*.pl}"
"${TRAIN_CMD[@]}"
