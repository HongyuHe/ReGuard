#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

VICTIM_CKPT="${VICTIM_CKPT:-$ROOT_DIR/models/parklb_v1_350k.pt}"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/online/models/v2/win5-12m-nolastgap-shielded03}"

NUM_SERVERS="${NUM_SERVERS:-10}"
SERVICE_RATES="${SERVICE_RATES:-}"
EPISODE_JOBS="${EPISODE_JOBS:-1000}"
OBS_HIGH="${OBS_HIGH:-500000.0}"

DT_MIN="${DT_MIN:-1.0}"
DT_MAX="${DT_MAX:-500.0}"
SIZE_MIN="${SIZE_MIN:-10.0}"
SIZE_MAX="${SIZE_MAX:-5000.0}"
FIRST_JOB_SIZE="${FIRST_JOB_SIZE:-100}"
REFERENCE_WINDOW="${REFERENCE_WINDOW:-${ORACLE_WINDOW:-5}}"
DISABLE_LAST_GAP_FEATURE="${DISABLE_LAST_GAP_FEATURE:-1}"

SHIELD_RULES_PREFIX="${SHIELD_RULES_PREFIX:-${VICTIM_SHIELD_RULES_PREFIX:-shield/out/shielded-iter2_shield_h2o_gap30_regret10}}"
SHIELD_RISK_TRIGGER_STEPS="${SHIELD_RISK_TRIGGER_STEPS:-1}"
SHIELD_COOLDOWN_STEPS="${SHIELD_COOLDOWN_STEPS:-0}"
SHIELD_EMPTY_FALLBACK="${SHIELD_EMPTY_FALLBACK:-least_load_proxy}"

TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-12000000}"
N_STEPS="${N_STEPS:-8192}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EVAL_FREQ="${EVAL_FREQ:-$((N_STEPS * 4))}"
EVAL_EPISODES="${EVAL_EPISODES:-5}"
LR="${LR:-3e-4}"
SEED="${SEED:-42}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-park-lb-train-attacker-v2}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_NAME="${WANDB_NAME:-win5-12m-nolastgap-shielded03}"
WANDB_TAGS="${WANDB_TAGS:-win5}"
WANDB_GROUP="${WANDB_GROUP:-}"
WANDB_DIR="${WANDB_DIR:-}"
WANDB_LOG_FREQ="${WANDB_LOG_FREQ:-10}"

if [[ ! -f "$VICTIM_CKPT" ]]; then
  echo "[error] victim checkpoint not found: $VICTIM_CKPT"
  echo "[hint] set VICTIM_CKPT=/path/to/best.pt or train one with scripts/train_victim_vanilla.sh"
  exit 1
fi
if [[ -n "$SHIELD_RULES_PREFIX" ]]; then
  if [[ ! -f "${SHIELD_RULES_PREFIX}_risk.pl" ]]; then
    echo "[error] shield risk rules not found: ${SHIELD_RULES_PREFIX}_risk.pl"
    echo "[hint] set SHIELD_RULES_PREFIX to the prefix before _risk.pl and _allow_server*.pl"
    exit 1
  fi
  for server_id in $(seq 0 $((NUM_SERVERS - 1))); do
    if [[ ! -f "${SHIELD_RULES_PREFIX}_allow_server${server_id}.pl" ]]; then
      echo "[error] shield allow rules not found: ${SHIELD_RULES_PREFIX}_allow_server${server_id}.pl"
      exit 1
    fi
  done
fi

WANDB_ARGS=(
  --wandb-mode "$WANDB_MODE"
  --wandb-project "$WANDB_PROJECT"
  --wandb-name "$WANDB_NAME"
  --wandb-tags "$WANDB_TAGS"
  --wandb-log-freq "$WANDB_LOG_FREQ"
)
if [[ -n "$WANDB_ENTITY" ]]; then
  WANDB_ARGS+=(--wandb-entity "$WANDB_ENTITY")
fi
if [[ -n "$WANDB_GROUP" ]]; then
  WANDB_ARGS+=(--wandb-group "$WANDB_GROUP")
fi
if [[ -n "$WANDB_DIR" ]]; then
  WANDB_ARGS+=(--wandb-dir "$WANDB_DIR")
fi

SHIELD_ARGS=()
if [[ -n "$SHIELD_RULES_PREFIX" ]]; then
  SHIELD_ARGS+=(
    --shield-rules-prefix "$SHIELD_RULES_PREFIX"
    --shield-risk-trigger-steps "$SHIELD_RISK_TRIGGER_STEPS"
    --shield-cooldown-steps "$SHIELD_COOLDOWN_STEPS"
    --shield-empty-fallback "$SHIELD_EMPTY_FALLBACK"
  )
fi

mkdir -p "$MODEL_DIR"

CMD=(
  time "$PYTHON_BIN" -m online.train_adv
  --victim-checkpoint "$VICTIM_CKPT"
  --num-servers "$NUM_SERVERS"
  --episode-jobs "$EPISODE_JOBS"
  --obs-high "$OBS_HIGH"
  --dt-min "$DT_MIN"
  --dt-max "$DT_MAX"
  --size-min "$SIZE_MIN"
  --size-max "$SIZE_MAX"
  --first-job-size "$FIRST_JOB_SIZE"
  --reference-window "$REFERENCE_WINDOW"
  --total-timesteps "$TOTAL_TIMESTEPS"
  --n-steps "$N_STEPS"
  --batch-size "$BATCH_SIZE"
  --eval-freq "$EVAL_FREQ"
  --eval-episodes "$EVAL_EPISODES"
  --learning-rate "$LR"
  --seed "$SEED"
  --out-dir "$MODEL_DIR"
  "${SHIELD_ARGS[@]}"
  "${WANDB_ARGS[@]}"
)

if [[ -n "$SERVICE_RATES" ]]; then
  CMD+=(--service-rates "$SERVICE_RATES")
fi
if [[ "$DISABLE_LAST_GAP_FEATURE" == "1" ]]; then
  CMD+=(--disable-last-gap-feature)
fi

"${CMD[@]}"
