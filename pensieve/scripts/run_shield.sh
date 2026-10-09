#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-$ROOT/venv/bin/python}"

MODEL="${MODEL:-ppo/nn_model_ep_426300.pth}"
# ADV_TRACE_DIR="${ADV_TRACE_DIR:-./Pensieve-PPO/src/test}"
# ADV_TRACE_DIR="${ADV_TRACE_DIR:-./traces/robust-test/100traces-obj7test-0.01alpha/pgd-last-feat}"
ADV_TRACE_DIR="${ADV_TRACE_DIR:-./online/adv-traces/pensieve}"
ADV_TRACE_NUM="${ADV_TRACE_NUM:-100}"
MAX_STEPS="${MAX_STEPS:--1}"
ACTION_SELECTION="${ACTION_SELECTION:-argmax}"
SEED="${SEED:-42}"
SHIELD_RISK_TRIGGER_STEPS="${SHIELD_RISK_TRIGGER_STEPS:-1}"
SHIELD_COOLDOWN_STEPS="${SHIELD_COOLDOWN_STEPS:-0}"

SHIELD_MIN_RULES="${SHIELD_MIN_RULES:-rules/shield/out/pgd_obj7_bounds_h2o_gap80_delta0p3_min.pl}"
SHIELD_MAX_RULES="${SHIELD_MAX_RULES:-rules/shield/out/pgd_obj7_bounds_h2o_gap80_delta0p3_max.pl}"
# SHIELD_MIN_RULES="${SHIELD_MIN_RULES:-rules/shield/out/online_adv_bounds_h2o_gap80_delta0p3_min.pl}"
# SHIELD_MAX_RULES="${SHIELD_MAX_RULES:-rules/shield/out/online_adv_bounds_h2o_gap80_delta0p3_max.pl}"

WANDB_PROJECT="${WANDB_PROJECT:-robust-pensieve-online-vs-offline}"
WANDB_MODE="${WANDB_MODE:-online}"
WAMDN_NAME="${WANDB_NAME:-lejit-gap80-h2o-pgd}"

# WANDB_TAGS="${WANDB_TAGS:-eval-clean-5e}"
# WANDB_TAGS="${WANDB_TAGS:-eval-pgd-obj7-5e}"
WANDB_TAGS="${WANDB_TAGS:-eval-hotnets-5e}"

"$PY" Pensieve-PPO/src/inference.py --mode shield \
  -m "$MODEL" \
  --adv-trace-dir "$ADV_TRACE_DIR" \
  --adv-trace-num "$ADV_TRACE_NUM" \
  --shield-min-rules-file "$SHIELD_MIN_RULES" \
  --shield-max-rules-file "$SHIELD_MAX_RULES" \
  --shield-risk-trigger-steps "$SHIELD_RISK_TRIGGER_STEPS" \
  --shield-cooldown-steps "$SHIELD_COOLDOWN_STEPS" \
  --max-steps "$MAX_STEPS" \
  --select "$ACTION_SELECTION" \
  --seed "$SEED" \
  --project "$WANDB_PROJECT" \
  --wandb "$WANDB_MODE" \
  --name "$WAMDN_NAME" \
  --tags "$WANDB_TAGS"
