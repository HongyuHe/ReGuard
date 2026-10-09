#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-ppo/nn_model_ep_426300.pth}"
OBJECTIVE="${OBJECTIVE:-mpc_gap}"
ORACLE_K="${ORACLE_K:-5}"
ROUNDS="${ROUNDS:-20}"
SEED="${SEED:-42}"
BW_MIN="${BW_MIN:-1e-3}"
BW_MAX="${BW_MAX:-8.951518}"
DURATION_S="${DURATION_S:-330}"
BW_CHANGE_INTERVAL="${BW_CHANGE_INTERVAL:-1.0}"
OPTIMIZE_BW_CHANGE_INTERVAL="${OPTIMIZE_BW_CHANGE_INTERVAL:-0}"
BW_CHANGE_INTERVAL_MIN="${BW_CHANGE_INTERVAL_MIN:-1.0}"
BW_CHANGE_INTERVAL_MAX="${BW_CHANGE_INTERVAL_MAX:-100.0}"
TRACES_PER_EVAL="${TRACES_PER_EVAL:-10}"
BO_INIT_POINTS="${BO_INIT_POINTS:-10}"
BO_ITER="${BO_ITER:-5}"
CURRICULUM_ADD_PROB="${CURRICULUM_ADD_PROB:-0.3}"
UPDATES_PER_ROUND="${UPDATES_PER_ROUND:-5000}"
BATCH_EPISODES="${BATCH_EPISODES:-10}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
INIT_ENTROPY_WEIGHT="${INIT_ENTROPY_WEIGHT:-0.01}"
FREEZE_ENTROPY_WEIGHT="${FREEZE_ENTROPY_WEIGHT:-1}"
EXPORT_TRACES_PER_ROUND="${EXPORT_TRACES_PER_ROUND:-10}"
REAL_TRACE_DIR="${REAL_TRACE_DIR:-}"
REAL_TRACE_PROB="${REAL_TRACE_PROB:-0.0}"
REAL_TRACE_NUM="${REAL_TRACE_NUM:-0}"
VAL_TRACE_DIR="${VAL_TRACE_DIR:-}"
VAL_TRACE_NUM="${VAL_TRACE_NUM:-0}"
OUT_DIR="${OUT_DIR:-genet/runs/${OBJECTIVE}}"

if [[ -z "${PYTHON_BIN:-}" ]] && [[ -x "venv/bin/python" ]]; then
  PYTHON_BIN="venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

args=(
  -m genet.iterative_train
  --model "$MODEL"
  --objective "$OBJECTIVE"
  --oracle-k "$ORACLE_K"
  --rounds "$ROUNDS"
  --seed "$SEED"
  --bw-min "$BW_MIN"
  --bw-max "$BW_MAX"
  --duration-s "$DURATION_S"
  --bw-change-interval "$BW_CHANGE_INTERVAL"
  --traces-per-eval "$TRACES_PER_EVAL"
  --bo-init-points "$BO_INIT_POINTS"
  --bo-iter "$BO_ITER"
  --curriculum-add-prob "$CURRICULUM_ADD_PROB"
  --updates-per-round "$UPDATES_PER_ROUND"
  --batch-episodes "$BATCH_EPISODES"
  --learning-rate "$LEARNING_RATE"
  --init-entropy-weight "$INIT_ENTROPY_WEIGHT"
  --export-traces-per-round "$EXPORT_TRACES_PER_ROUND"
  --real-trace-prob "$REAL_TRACE_PROB"
  --real-trace-num "$REAL_TRACE_NUM"
  --val-trace-num "$VAL_TRACE_NUM"
  --out-dir "$OUT_DIR"
  --overwrite
)

if [[ "$FREEZE_ENTROPY_WEIGHT" == "1" ]]; then
  args+=( --freeze-entropy-weight )
else
  args+=( --no-freeze-entropy-weight )
fi

if [[ "$OPTIMIZE_BW_CHANGE_INTERVAL" == "1" ]]; then
  args+=(
    --optimize-bw-change-interval
    --bw-change-interval-min "$BW_CHANGE_INTERVAL_MIN"
    --bw-change-interval-max "$BW_CHANGE_INTERVAL_MAX"
  )
fi

if [[ -n "$REAL_TRACE_DIR" ]]; then
  args+=( --real-trace-dir "$REAL_TRACE_DIR" )
fi

if [[ -n "$VAL_TRACE_DIR" ]]; then
  args+=( --val-trace-dir "$VAL_TRACE_DIR" )
fi

"$PYTHON_BIN" "${args[@]}"
