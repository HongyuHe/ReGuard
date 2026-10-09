#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${SEED:-42}"

TRACE_DIR="${TRACE_DIR:-$ROOT_DIR/traces/train}"
NUM_TRACES="${NUM_TRACES:-200}"
NUM_JOBS="${NUM_JOBS:-1000}"

SAVE_DIR="${SAVE_DIR:-$ROOT_DIR/pretrained}"
PENSIEVE_TRAIN_EPOCH="${PENSIEVE_TRAIN_EPOCH:-500000}"
PENSIEVE_TRAIN_SEQ_LEN="${PENSIEVE_TRAIN_SEQ_LEN:-1000}"
PENSIEVE_PPO_TRAINING_EPO="${PENSIEVE_PPO_TRAINING_EPO:-5}"
PENSIEVE_MODEL_SAVE_INTERVAL="${PENSIEVE_MODEL_SAVE_INTERVAL:-300}"
TOTAL_ITERS="${TOTAL_ITERS:-$PENSIEVE_TRAIN_EPOCH}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-$PENSIEVE_TRAIN_SEQ_LEN}"
PPO_EPOCHS="${PPO_EPOCHS:-$PENSIEVE_PPO_TRAINING_EPO}"
MINI_BATCH="${MINI_BATCH:-$ROLLOUT_STEPS}"
LR="${LR:-1e-4}"
HIDDEN="${HIDDEN:-128}"
GAMMA="${GAMMA:-0.99}"
GAE_LAMBDA="${GAE_LAMBDA:-1.0}"
VF_COEF="${VF_COEF:-0.5}"
ENT_COEF="${ENT_COEF:-2.302585092994046}"
REWARD_TIME_SCALE="${REWARD_TIME_SCALE:-1000}"
NORMALIZE_VALUE_LOSS="${NORMALIZE_VALUE_LOSS:-1}"
EVAL_EVERY="${EVAL_EVERY:-$PENSIEVE_MODEL_SAVE_INTERVAL}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-park-lb-train-victim}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_NAME="${WANDB_NAME:-v2}"
WANDB_TAGS="${WANDB_TAGS:-v2}"
WANDB_GROUP="${WANDB_GROUP:-}"
WANDB_DIR="${WANDB_DIR:-}"

mkdir -p "$TRACE_DIR" "$SAVE_DIR"

WANDB_ARGS=(
  --wandb-mode "$WANDB_MODE"
  --wandb-project "$WANDB_PROJECT"
  --wandb-name "$WANDB_NAME"
  --wandb-tags "$WANDB_TAGS"
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

# echo "[1/3] Generate (or refresh) benign training traces"
# "$PYTHON_BIN" scripts/generate_job_traces.py \
#   --out-dir "$TRACE_DIR" \
#   --num-traces "$NUM_TRACES" \
#   --num-jobs "$NUM_JOBS" \
#   --job-interval 55 \
#   --job-size-pareto-shape 1.5 \
#   --job-size-pareto-scale 100 \
#   --seed "$SEED" \
#   --time-mode absolute

echo "[2/3] Train victim PPO"
"$PYTHON_BIN" train_ppo.py \
  --seed "$SEED" \
  --num-servers 10 \
  --workload-trace-dir "$TRACE_DIR" \
  --trace-select-mode random \
  --total-iters "$TOTAL_ITERS" \
  --rollout-steps "$ROLLOUT_STEPS" \
  --ppo-epochs "$PPO_EPOCHS" \
  --mini-batch "$MINI_BATCH" \
  --gamma "$GAMMA" \
  --gae-lambda "$GAE_LAMBDA" \
  --vf-coef "$VF_COEF" \
  --ent-coef "$ENT_COEF" \
  --reward-time-scale "$REWARD_TIME_SCALE" \
  --normalize-value-loss "$NORMALIZE_VALUE_LOSS" \
  --lr "$LR" \
  --hidden "$HIDDEN" \
  --eval-every "$EVAL_EVERY" \
  --eval-episodes "$EVAL_EPISODES" \
  --save-dir "$SAVE_DIR" \
  "${WANDB_ARGS[@]}"

echo "[3/3] Evaluate best checkpoint"
"$PYTHON_BIN" eval_policy.py \
  --checkpoint "$SAVE_DIR/best.pt" \
  --episodes 20 \
  --seed "$SEED"

echo "Done."
echo "Best victim checkpoint: $SAVE_DIR/best.pt"
