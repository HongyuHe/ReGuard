#!/usr/bin/env bash
set -euo pipefail

PY="${PY:-python}"
PENSIEVE_MODEL="${PENSIEVE_MODEL:-ppo/nn_model_ep_426300.pth}"
TOTAL_STEPS="${TOTAL_STEPS:-600000}"
LR="${LR:-3e-4}"
SEED="${SEED:-42}"
BW_MAX="${BW_MAX:-8.951518}"
BW_MIN="${BW_MIN:-1e-3}"
SMOOTH_PENALTY_SCALE="${SMOOTH_PENALTY_SCALE:-0}"
ORACLE_WINDOW="${ORACLE_WINDOW:-5}"
OBS_HISTORY="${OBS_HISTORY:-10}"
OUT_DIR="${OUT_DIR:-online/models}"
# REWARD_OBJECTIVE="${REWARD_OBJECTIVE:-oracle_floor_max_regret}"
# REWARD_OBJECTIVE="${REWARD_OBJECTIVE:-weighted_max_regret}"
REWARD_OBJECTIVE="${REWARD_OBJECTIVE:-max_regret}"
REWARD_TAU="${REWARD_TAU:-0.0}"
REWARD_BETA="${REWARD_BETA:-1.0}"
SHIELD_MIN_RULES_FILE="${SHIELD_MIN_RULES_FILE:-rules/shield/out/shielded-iter3_bounds_h2o_gap75_delta10_dmin0p3_dmax0p5_min.pl}"
SHIELD_MAX_RULES_FILE="${SHIELD_MAX_RULES_FILE:-rules/shield/out/shielded-iter3_bounds_h2o_gap75_delta10_dmin0p3_dmax0p5_max.pl}"
SHIELD_ACTION_SELECTION="${SHIELD_ACTION_SELECTION:-argmax}"
WANDB_PROJECT="${WANDB_PROJECT:-pensieve-online-adv}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_TAGS="${WANDB_TAGS:-}"

CMD=(
  "$PY" -m online.train_adv
  --pensieve-model "$PENSIEVE_MODEL"
  --total-steps "$TOTAL_STEPS"
  --lr "$LR"
  --seed "$SEED"
  --bw-min "$BW_MIN"
  --bw-max "$BW_MAX"
  --smooth-penalty-scale "$SMOOTH_PENALTY_SCALE"
  --reward-objective "$REWARD_OBJECTIVE"
  --reward-tau "$REWARD_TAU"
  --reward-beta "$REWARD_BETA"
  --obs-history "$OBS_HISTORY"
  --oracle-window "$ORACLE_WINDOW"
  --out-dir "$OUT_DIR"
)

SHIELD_TAG="unshielded"
WANDB_TAGS_EFFECTIVE="$WANDB_TAGS"
if [[ -n "$SHIELD_MIN_RULES_FILE" || -n "$SHIELD_MAX_RULES_FILE" ]]; then
  if [[ -z "$SHIELD_MIN_RULES_FILE" || -z "$SHIELD_MAX_RULES_FILE" ]]; then
    echo "* Set both SHIELD_MIN_RULES_FILE and SHIELD_MAX_RULES_FILE, or leave both empty." >&2
    exit 2
  fi
  CMD+=(
    --shield-min-rules-file "$SHIELD_MIN_RULES_FILE"
    --shield-max-rules-file "$SHIELD_MAX_RULES_FILE"
    --shield-action-selection "$SHIELD_ACTION_SELECTION"
  )
  SHIELD_TAG="shielded-${SHIELD_ACTION_SELECTION}"
  if [[ -z "$WANDB_TAGS_EFFECTIVE" ]]; then
    WANDB_TAGS_EFFECTIVE="shielded-victim-iter3"
  else
    WANDB_TAGS_EFFECTIVE="${WANDB_TAGS_EFFECTIVE},shielded-victim-iter3"
  fi
fi

CMD+=(
  --wandb
  --wandb-project "$WANDB_PROJECT"
  --wandb-mode "$WANDB_MODE"
  --wandb-name "train-online-${SHIELD_TAG}-obj-${REWARD_OBJECTIVE}-tau${REWARD_TAU}-beta${REWARD_BETA}-smooth${SMOOTH_PENALTY_SCALE}-oraclewin${ORACLE_WINDOW}-bwmin${BW_MIN}-bwmax${BW_MAX}"
)

if [[ -n "$WANDB_ENTITY" ]]; then
  CMD+=(--wandb-entity "$WANDB_ENTITY")
fi
if [[ -n "$WANDB_TAGS_EFFECTIVE" ]]; then
  CMD+=(--wandb-tags "$WANDB_TAGS_EFFECTIVE")
fi

exec "${CMD[@]}"
