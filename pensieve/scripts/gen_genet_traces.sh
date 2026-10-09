#!/usr/bin/env bash
set -euo pipefail

PENSIEVE_MODEL="${PENSIEVE_MODEL:-ppo/nn_model_ep_426300.pth}"
EPISODES="${EPISODES:-200}"
SEED="${SEED:-42424}"
BW_MIN="${BW_MIN:-1e-3}"
BW_MAX="${BW_MAX:-8.951518}"
OBJECTIVE="${OBJECTIVE:-mpc_gap}"
ORACLE_K="${ORACLE_K:-5}"
TRACES_PER_EVAL="${TRACES_PER_EVAL:-10}"
BO_INIT_POINTS="${BO_INIT_POINTS:-10}"
BO_ITER="${BO_ITER:-5}"
BW_CHANGE_INTERVAL="${BW_CHANGE_INTERVAL:-1.0}"
OPTIMIZE_BW_CHANGE_INTERVAL="${OPTIMIZE_BW_CHANGE_INTERVAL:-0}"
BW_CHANGE_INTERVAL_MIN="${BW_CHANGE_INTERVAL_MIN:-1.0}"
BW_CHANGE_INTERVAL_MAX="${BW_CHANGE_INTERVAL_MAX:-100.0}"
OUT_DIR="${OUT_DIR:-traces/genet/genet-mpc}"

if [[ -z "${PYTHON_BIN:-}" ]] && [[ -x "venv/bin/python" ]]; then
  PYTHON_BIN="venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

args=(
  -m genet.generate_traces
  --pensieve-model "$PENSIEVE_MODEL"
  --episodes "$EPISODES"
  --seed "$SEED"
  --bw-min "$BW_MIN"
  --bw-max "$BW_MAX"
  --bw-change-interval "$BW_CHANGE_INTERVAL"
  --traces-per-eval "$TRACES_PER_EVAL"
  --bo-init-points "$BO_INIT_POINTS"
  --bo-iter "$BO_ITER"
  --objective "$OBJECTIVE"
  --oracle-k "$ORACLE_K"
  --out-dir "$OUT_DIR"
  --overwrite
)

if [[ "$OPTIMIZE_BW_CHANGE_INTERVAL" == "1" ]]; then
  args+=(
    --optimize-bw-change-interval
    --bw-change-interval-min "$BW_CHANGE_INTERVAL_MIN"
    --bw-change-interval-max "$BW_CHANGE_INTERVAL_MAX"
  )
fi

"$PYTHON_BIN" "${args[@]}"
