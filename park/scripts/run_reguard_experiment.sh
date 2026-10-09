#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-$ROOT/.venv/bin/python}"
SEED="${SEED:-42}"
TRAIN_ITERS="${TRAIN_ITERS:-4000}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-4096}"
TRACE_LIMIT="${TRACE_LIMIT:-20}"
MAX_LOGGED_STATES="${MAX_LOGGED_STATES:-2048}"
TRACE_DIR="${TRACE_DIR:-$ROOT/traces/maxgap}"
RUN_NAME="${RUN_NAME:-park_lb_seed${SEED}_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="$ROOT/results/$RUN_NAME"
PAPER_TRACE_LIMIT="${PAPER_TRACE_LIMIT:-0}"

if [[ ! -x "$PY" ]]; then
  echo "Python environment not found: $PY" >&2
  exit 2
fi
if [[ -e "$RUN_DIR" ]]; then
  echo "Run directory already exists: $RUN_DIR" >&2
  exit 2
fi

"$PY" "$ROOT/scripts/prepare_model.py" --results-dir "$RUN_DIR" --check-only
"$PY" "$ROOT/scripts/evaluate_paper.py" --results-dir "$RUN_DIR" --check-inputs

TRAIN_DIR="$RUN_DIR/controller_training"
SHIELD_DIR="$RUN_DIR/shield"
EVAL_DIR="$RUN_DIR/evaluation"
RULE_PREFIX="$SHIELD_DIR/rules/park_shield_sklearn"
mkdir -p "$TRAIN_DIR" "$SHIELD_DIR/data" "$SHIELD_DIR/rules" "$EVAL_DIR"

echo "Park run directory: $RUN_DIR"
"$PY" train_ppo.py \
  --seed "$SEED" \
  --total-iters "$TRAIN_ITERS" \
  --rollout-steps "$ROLLOUT_STEPS" \
  --ppo-epochs 2 \
  --mini-batch 256 \
  --eval-every 1 \
  --eval-episodes 1 \
  --save-dir "$TRAIN_DIR" \
  --wandb-mode disabled

CONTROLLER_MODEL="${CONTROLLER_MODEL:-$TRAIN_DIR/last.pt}"
if [[ ! -f "$CONTROLLER_MODEL" ]]; then
  echo "Controller checkpoint was not produced: $CONTROLLER_MODEL" >&2
  exit 3
fi

PY="$PY" \
PREFIX="park_lb_seed${SEED}" \
TRACE_DIR="$TRACE_DIR" \
MODEL="$CONTROLLER_MODEL" \
TRACE_LIMIT="$TRACE_LIMIT" \
MAX_LOGGED_STATES="$MAX_LOGGED_STATES" \
SEED="$SEED" \
OUT_DIR="$SHIELD_DIR/data" \
RULE_DIR="$SHIELD_DIR/rules" \
OUT_PREFIX="$RULE_PREFIX" \
BACKEND=sklearn \
REGENERATE_DATASETS=1 \
bash scripts/gen_shield_rules.sh

"$PY" scripts/eval_trace_dir_oracle_regret.py \
  --victim-checkpoint "$CONTROLLER_MODEL" \
  --trace-dir "$TRACE_DIR" \
  --shield-rules-prefix "$RULE_PREFIX" \
  --eval-count "$TRACE_LIMIT" \
  --trace-limit "$TRACE_LIMIT" \
  --out-dir "$EVAL_DIR" \
  --no-wandb

test -s "${RULE_PREFIX}_risk.pl"
test -s "$EVAL_DIR/episode_metrics.csv"
"$PY" "$ROOT/scripts/evaluate_paper.py" --results-dir "$RUN_DIR" \
  --trace-limit "$PAPER_TRACE_LIMIT" --workers "${EVAL_WORKERS:-4}"
"$PY" "$ROOT/scripts/prepare_model.py" --results-dir "$RUN_DIR"
echo "Park experiment complete: $RUN_DIR"
