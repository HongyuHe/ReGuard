#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-online/models/online_attacker_maxregret-shielded4_20260419.zip}"
# MODEL="${MODEL:-online/models/online_attacker_full_0min_20260225.zip}"
# MODEL="${MODEL:-online/models/online_attacker_weighted_0min_20260326.zip}"
# MODEL="${MODEL:-online/models/online_attacker_restricted_20260202.zip}"
# MODEL="${MODEL:-online/models/test-online/finetune-pgd-manual-0.1a-20i/adv_ppo_bw_20260216-034846.zip}"
EPISODES="${EPISODES:-200}"
SEED="${SEED:-333}"
DETERMINISTIC="${DETERMINISTIC:-0}"

#* Keep these aligned with scripts/train_online.sh unless you explicitly want a mismatch.
PENSIEVE_MODEL="${PENSIEVE_MODEL:-ppo/nn_model_ep_426300.pth}"
BW_MIN="${BW_MIN:-1e-3}"
BW_MAX="${BW_MAX:-8.951518}"
OBS_HISTORY="${OBS_HISTORY:-10}"
ORACLE_WINDOW="${ORACLE_WINDOW:-5}"
SMOOTH_PENALTY_SCALE="${SMOOTH_PENALTY_SCALE:-0}"
REWARD_OBJECTIVE="${REWARD_OBJECTIVE:-max_regret}"
REWARD_TAU="${REWARD_TAU:-0.0}"
REWARD_BETA="${REWARD_BETA:-1.0}"
OUT_DIR="${OUT_DIR:-traces/shielded/iter4}"

CMD=(
  python -m online.generate_traces
  --adv-model "$MODEL"
  --pensieve-model "$PENSIEVE_MODEL"
  --episodes "$EPISODES"
  --seed "$SEED"
  --bw-min "$BW_MIN"
  --bw-max "$BW_MAX"
  --obs-history "$OBS_HISTORY"
  --oracle-window "$ORACLE_WINDOW"
  --smooth-penalty-scale "$SMOOTH_PENALTY_SCALE"
  --reward-objective "$REWARD_OBJECTIVE"
  --reward-tau "$REWARD_TAU"
  --reward-beta "$REWARD_BETA"
  --no-random
  --flat-out-dir
  --out-dir "$OUT_DIR"
)

if [[ "$DETERMINISTIC" == "1" ]]; then
  CMD+=(--deterministic)
fi

exec "${CMD[@]}"
