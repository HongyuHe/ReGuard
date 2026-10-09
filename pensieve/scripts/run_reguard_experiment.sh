#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-$ROOT/venv/bin/python}"
SEED="${SEED:-42}"
TRAIN_EPISODES="${TRAIN_EPISODES:-50000}"
TRACE_NUM="${TRACE_NUM:-40}"
MAX_LOGGED_STATES="${MAX_LOGGED_STATES:-2048}"
MAX_STEPS="${MAX_STEPS:-256}"
TRACE_DIR="${TRACE_DIR:-$ROOT/traces/maxregret-shield}"
BASE_MODEL="${BASE_MODEL:-$ROOT/Pensieve-PPO/src/ppo/nn_model_ep_426300.pth}"
RUN_NAME="${RUN_NAME:-pensieve_abr_seed${SEED}_$(date -u +%Y%m%dT%H%M%SZ)}"
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
RULE_PREFIX="$SHIELD_DIR/rules/pensieve_bounds_sklearn"
mkdir -p "$TRAIN_DIR" "$SHIELD_DIR/data" "$SHIELD_DIR/rules" "$EVAL_DIR"

echo "Pensieve run directory: $RUN_DIR"
WANDB_DIR="$TRAIN_DIR" "$PY" Pensieve-PPO/src/inference.py \
  --mode robustify \
  --model "$BASE_MODEL" \
  --adv-trace-dir "$TRACE_DIR" \
  --adv-trace-num "$TRACE_NUM" \
  --robustify-episodes "$TRAIN_EPISODES" \
  --robustify-batch-episodes 2 \
  --robustify-clean-ratio 0 \
  --robustify-out-dir "$TRAIN_DIR" \
  --seed "$SEED" \
  --wandb-mode disabled

CONTROLLER_MODEL="$(find "$TRAIN_DIR" -maxdepth 1 -type f -name '*.pt' -print -quit)"
if [[ -z "$CONTROLLER_MODEL" ]]; then
  echo "Robustified controller checkpoint was not produced under $TRAIN_DIR" >&2
  exit 3
fi

PY="$PY" \
PREFIX="pensieve_abr_seed${SEED}" \
TRACE_DIR="$TRACE_DIR" \
MODEL="$CONTROLLER_MODEL" \
TRACE_NUM="$TRACE_NUM" \
MAX_LOGGED_STATES="$MAX_LOGGED_STATES" \
SEED="$SEED" \
OUT_DIR="$SHIELD_DIR/data" \
OUT_PREFIX="$RULE_PREFIX" \
BACKEND=sklearn \
bash scripts/gen_shield_rules.sh

WANDB_DIR="$EVAL_DIR" "$PY" Pensieve-PPO/src/inference.py \
  --mode shield \
  --model "$CONTROLLER_MODEL" \
  --adv-trace-dir "$TRACE_DIR" \
  --adv-trace-num "$TRACE_NUM" \
  --shield-min-rules-file "${RULE_PREFIX}_min.pl" \
  --shield-max-rules-file "${RULE_PREFIX}_max.pl" \
  --max-steps "$MAX_STEPS" \
  --seed "$SEED" \
  --wandb-project reguard-pensieve-eval \
  --wandb-mode offline

test -s "${RULE_PREFIX}_min.pl"
test -s "${RULE_PREFIX}_max.pl"
find "$EVAL_DIR" -type f -print -quit | grep -q .
"$PY" "$ROOT/scripts/evaluate_paper.py" --results-dir "$RUN_DIR" \
  --trace-limit "$PAPER_TRACE_LIMIT" --workers "${EVAL_WORKERS:-4}"
"$PY" "$ROOT/scripts/prepare_model.py" --results-dir "$RUN_DIR"
echo "Pensieve experiment complete: $RUN_DIR"
