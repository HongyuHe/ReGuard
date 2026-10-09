#!/usr/bin/env bash
set -euo pipefail

#* Evaluate ONE pretrained Pensieve model on:
#* 1) clean traces
#* 2) offline (feature-space) attack traces
#* 3) online (bandwidth-policy) attack traces

MODEL="${MODEL:-ppo/nn_model_ep_426300.pth}"
NUM_TRACES="${NUM_TRACES:-10}"
SEED="${SEED:-42}"
STEPS="${STEPS:--1}"

OFFLINE_DIR="${OFFLINE_DIR:-traces/dtgaphist-30t-0.9oracle/dtgap-sklearn-feat}"
CLEAN_DIR="${CLEAN_DIR:-Pensieve-PPO/src/test}"
ONLINE_DIR="${ONLINE_DIR:-online/adv-traces/pensieve}"

WANDB_MODE="${WANDB_MODE:-online}" # online | offline | disabled
WANDB_PROJECT="${WANDB_PROJECT:-attack-pensieve-eval}"
WANDB_GROUP="${WANDB_GROUP:-eval-trace-sets-$(date +%Y%m%d-%H%M%S)}"
WANDB_NAME="${WANDB_NAME:-}"
WANDB_TAGS="${WANDB_TAGS:-10traces-sk-last}"

if [[ -z "${PYTHON_BIN:-}" ]] && [[ -x "venv/bin/python" ]]; then
  PYTHON_BIN="venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

args=(
  --model "$MODEL"
  --trace-num "$NUM_TRACES"
  --seed "$SEED"
  --max-steps "$STEPS"
  --clean-trace-dir "$CLEAN_DIR"
  --offline-trace-dir "$OFFLINE_DIR"
  --online-trace-dir "$ONLINE_DIR"
  --wandb-mode "$WANDB_MODE"
  --wandb-project "$WANDB_PROJECT"
  --wandb-group "$WANDB_GROUP"
  --wandb-tags "$WANDB_TAGS"
)

if [[ -n "$WANDB_NAME" ]]; then
  args+=(--wandb-name "$WANDB_NAME")
fi

"$PYTHON_BIN" scripts/eval_trace_sets.py "${args[@]}"
