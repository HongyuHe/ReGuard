#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/train_gen_online_modeldir.sh --model-dir DIR --episodes N [options]

Trains an online bandwidth adversary (SB3 PPO) against every Pensieve model under --model-dir,
then generates N adversarial traces for each trained adversary.

Outputs:
  - Adversary models: online/models/test-online/<model-suffix>/adv_ppo_bw_<timestamp>.zip
  - Traces: traces/test-online/<model-suffix>/episode_XXXX.trace (+ .segments.json)

Model suffix rule:
  - If the model stem contains "_robust_", suffix = substring after "_robust_"
  - Else suffix = full filename stem (no extension)

Options:
  --total-steps STEPS           (default: 600000)
  --lr LR                       (default: 3e-4)
  --seed SEED                   (default: 42)
  --bw-min MIN                  (default: 0.8)
  --bw-max MAX                  (default: 4.8)
  --obs-history H               (default: 10)
  --oracle-window W             (default: 4)
  --smooth-penalty-scale S      (default: 1.0)
  --reward-objective OBJ        (default: weighted_max_regret)
  --reward-tau TAU              (default: 0.0)
  --reward-beta BETA            (default: 1.0)
  --out-trace-root DIR          (default: traces/test-online)
  --adv-model-root DIR          (default: online/models/test-online)
  --jobs J                      Max concurrent trainings (default: all)
  --deterministic               Use mean action during trace generation (likely identical episodes)
  --wandb                       Enable wandb logging for training + generation
  --wandb-project NAME          (default: pensieve-online-adv)
  --wandb-entity ENTITY         (default: unset)
  --wandb-tags TAGS             Comma-separated (default: test-online)
EOF
}

if [[ -z "${PYTHON_BIN:-}" ]] && [[ -x "venv/bin/python" ]]; then
  PYTHON_BIN="venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

MODEL_DIR=""
EPISODES="200"
TOTAL_STEPS="600000"
LR="3e-4"
SEED="42"
BW_MIN="0.8" # "0"
BW_MAX="4.8" # "8.951518"
OBS_HISTORY="10"
ORACLE_WINDOW="4"
SMOOTH_PENALTY_SCALE="0"
REWARD_OBJECTIVE="weighted_max_regret"
REWARD_TAU="0.0"
REWARD_BETA="1.0"
OUT_TRACE_ROOT="traces/test-hotnets19"
ADV_MODEL_ROOT="online/models/test-fullrl"
JOBS="0"
DETERMINISTIC="0"
WANDB="0"
WANDB_PROJECT="pensieve-online-adv"
WANDB_ENTITY=""
WANDB_TAGS="test-fullrl"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-dir)
      MODEL_DIR="${2:-}"
      shift 2
      ;;
    --episodes)
      EPISODES="${2:-}"
      shift 2
      ;;
    --total-steps)
      TOTAL_STEPS="${2:-}"
      shift 2
      ;;
    --lr)
      LR="${2:-}"
      shift 2
      ;;
    --seed)
      SEED="${2:-}"
      shift 2
      ;;
    --bw-min)
      BW_MIN="${2:-}"
      shift 2
      ;;
    --bw-max)
      BW_MAX="${2:-}"
      shift 2
      ;;
    --obs-history)
      OBS_HISTORY="${2:-}"
      shift 2
      ;;
    --oracle-window)
      ORACLE_WINDOW="${2:-}"
      shift 2
      ;;
    --smooth-penalty-scale)
      SMOOTH_PENALTY_SCALE="${2:-}"
      shift 2
      ;;
    --reward-objective)
      REWARD_OBJECTIVE="${2:-}"
      shift 2
      ;;
    --reward-tau)
      REWARD_TAU="${2:-}"
      shift 2
      ;;
    --reward-beta)
      REWARD_BETA="${2:-}"
      shift 2
      ;;
    --out-trace-root)
      OUT_TRACE_ROOT="${2:-}"
      shift 2
      ;;
    --adv-model-root)
      ADV_MODEL_ROOT="${2:-}"
      shift 2
      ;;
    --jobs)
      JOBS="${2:-}"
      shift 2
      ;;
    --deterministic)
      DETERMINISTIC="1"
      shift 1
      ;;
    --wandb)
      WANDB="1"
      shift 1
      ;;
    --wandb-project)
      WANDB_PROJECT="${2:-}"
      shift 2
      ;;
    --wandb-entity)
      WANDB_ENTITY="${2:-}"
      shift 2
      ;;
    --wandb-tags)
      WANDB_TAGS="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "* Unknown arg: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$MODEL_DIR" ]]; then
  if [[ -d "Pensieve-PPO/src/models" ]]; then
    MODEL_DIR="Pensieve-PPO/src/models"
  elif [[ -d "models" ]]; then
    MODEL_DIR="models"
  else
    echo "* Missing required flag: --model-dir" >&2
    usage >&2
    exit 2
  fi
fi

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "* --model-dir does not exist: $MODEL_DIR" >&2
  exit 2
fi

mapfile -t MODELS < <(find "$MODEL_DIR" -maxdepth 1 -type f \( -name "*.pt" -o -name "*.pth" \) | LC_ALL=C sort)
if [[ "${#MODELS[@]}" -eq 0 ]]; then
  echo "* No models found under: $MODEL_DIR (expected .pt or .pth files)" >&2
  exit 2
fi

MODEL_PATHS=()
SUFFIXES=()
ADV_OUT_DIRS=()
TRACE_OUT_DIRS=()
TRAIN_LOGS=()
ADV_MODEL_PATH_FILES=()
PIDS=()

for model_path in "${MODELS[@]}"; do
  base="$(basename "$model_path")"
  stem="${base%.*}"
  suffix="$stem"
  if [[ "$stem" == *_robust_* ]]; then
    suffix="${stem#*_robust_}"
  fi

  adv_out_dir="${ADV_MODEL_ROOT}/${suffix}"
  trace_out_dir="${OUT_TRACE_ROOT}/${suffix}"
  train_log="${adv_out_dir}/train.log"
  adv_model_path_file="${adv_out_dir}/adv_model_path.txt"

  echo "* Victim model: $model_path"
  echo "* Model suffix: $suffix"
  echo "* Reward objective: $REWARD_OBJECTIVE"
  echo "* Reward tau: $REWARD_TAU"
  echo "* Reward beta: $REWARD_BETA"
  echo "* Adv out dir:  $adv_out_dir"
  echo "* Trace out dir: $trace_out_dir"

  mkdir -p "$adv_out_dir"
  mkdir -p "$trace_out_dir"

  MODEL_PATHS+=("$model_path")
  SUFFIXES+=("$suffix")
  ADV_OUT_DIRS+=("$adv_out_dir")
  TRACE_OUT_DIRS+=("$trace_out_dir")
  TRAIN_LOGS+=("$train_log")
  ADV_MODEL_PATH_FILES+=("$adv_model_path_file")
done

if [[ "$JOBS" == "0" ]]; then
  JOBS="${#MODEL_PATHS[@]}"
fi
if ! [[ "$JOBS" =~ ^[0-9]+$ ]] || [[ "$JOBS" == "0" ]]; then
  echo "* Invalid --jobs: $JOBS" >&2
  exit 2
fi

failed=0
active_indices=()
for i in "${!MODEL_PATHS[@]}"; do
  model_path="${MODEL_PATHS[$i]}"
  suffix="${SUFFIXES[$i]}"
  adv_out_dir="${ADV_OUT_DIRS[$i]}"
  train_log="${TRAIN_LOGS[$i]}"
  adv_model_path_file="${ADV_MODEL_PATH_FILES[$i]}"

  shopt -s nullglob
  adv_zips=("${adv_out_dir}"/*.zip)
  shopt -u nullglob
  if (( ${#adv_zips[@]} > 0 )); then
    adv_model_path="$(printf '%s\n' "${adv_zips[@]}" | LC_ALL=C sort | tail -n 1)"
    echo "* [train] skip suffix=$suffix (found trained adversary: $adv_model_path)"
    echo "$adv_model_path" >"$adv_model_path_file"
    continue
  fi

  echo "* [train] start suffix=$suffix (log: $train_log)"
  (
    set -euo pipefail
    wandb_train=()
    if [[ "$WANDB" == "1" ]]; then
      wandb_train+=(--wandb --wandb-project "$WANDB_PROJECT" --wandb-tags "$WANDB_TAGS")
      if [[ -n "$WANDB_ENTITY" ]]; then
        wandb_train+=(--wandb-entity "$WANDB_ENTITY")
      fi
      wandb_train+=(--wandb-name "online_train_${REWARD_OBJECTIVE}_tau${REWARD_TAU}_beta${REWARD_BETA}_${suffix}")
    fi

    "$PYTHON_BIN" -m online.train_adv \
      --pensieve-model "$model_path" \
      --total-steps "$TOTAL_STEPS" \
      --lr "$LR" \
      --seed "$SEED" \
      --bw-min "$BW_MIN" --bw-max "$BW_MAX" \
      --obs-history "$OBS_HISTORY" \
      --oracle-window "$ORACLE_WINDOW" \
      --smooth-penalty-scale "$SMOOTH_PENALTY_SCALE" \
      --reward-objective "$REWARD_OBJECTIVE" \
      --reward-tau "$REWARD_TAU" \
      --reward-beta "$REWARD_BETA" \
      --out-dir "$adv_out_dir" \
      "${wandb_train[@]}" \
      >"$train_log" 2>&1

    adv_model_path=""
    shopt -s nullglob
    adv_zips=("${adv_out_dir}"/*.zip)
    shopt -u nullglob

    if (( ${#adv_zips[@]} > 0 )); then
      adv_model_path="$(printf '%s\n' "${adv_zips[@]}" | LC_ALL=C sort | tail -n 1)"
    fi

    if [[ -z "$adv_model_path" ]]; then
      adv_model_path="$(tr -d '\r' <"$train_log" | awk '/\.zip$/ {last=$0} END {print last}')"
    fi

    if [[ -z "$adv_model_path" ]]; then
      echo "* Failed to parse trained adversary model path for suffix=$suffix" >&2
      exit 1
    fi
    if [[ ! -f "$adv_model_path" ]]; then
      echo "* Parsed adversary model path does not exist for suffix=$suffix: $adv_model_path" >&2
      exit 1
    fi
    echo "$adv_model_path" >"$adv_model_path_file"
  ) &

  pid=$!
  PIDS[$i]="$pid"
  active_indices+=("$i")

  if (( ${#active_indices[@]} >= JOBS )); then
    j="${active_indices[0]}"
    if ! wait "${PIDS[$j]}"; then
      echo "* [train] FAILED suffix=${SUFFIXES[$j]} (log: ${TRAIN_LOGS[$j]})" >&2
      failed=1
    else
      echo "* [train] done suffix=${SUFFIXES[$j]}"
    fi
    active_indices=("${active_indices[@]:1}")
  fi
done

for j in "${active_indices[@]}"; do
  if ! wait "${PIDS[$j]}"; then
    echo "* [train] FAILED suffix=${SUFFIXES[$j]} (log: ${TRAIN_LOGS[$j]})" >&2
    failed=1
  else
    echo "* [train] done suffix=${SUFFIXES[$j]}"
  fi
done

if [[ "$failed" != "0" ]]; then
  exit 1
fi

for i in "${!MODEL_PATHS[@]}"; do
  model_path="${MODEL_PATHS[$i]}"
  suffix="${SUFFIXES[$i]}"
  trace_out_dir="${TRACE_OUT_DIRS[$i]}"
  adv_model_path_file="${ADV_MODEL_PATH_FILES[$i]}"

  adv_model_path="$(cat "$adv_model_path_file" | head -n 1 | tr -d '\r')"
  if [[ -z "$adv_model_path" ]] || [[ "$adv_model_path" != *.zip ]]; then
    echo "* Missing/invalid adversary model path for suffix=$suffix (file: $adv_model_path_file)" >&2
    exit 1
  fi
  if [[ ! -f "$adv_model_path" ]]; then
    adv_out_dir="$(dirname "$adv_model_path_file")"
    shopt -s nullglob
    adv_zips=("${adv_out_dir}"/*.zip)
    shopt -u nullglob
    if (( ${#adv_zips[@]} > 0 )); then
      adv_model_path="$(printf '%s\n' "${adv_zips[@]}" | LC_ALL=C sort | tail -n 1)"
    fi
  fi
  if [[ ! -f "$adv_model_path" ]]; then
    echo "* Adversary model file not found for suffix=$suffix: $adv_model_path" >&2
    exit 1
  fi

  wandb_gen=()
  if [[ "$WANDB" == "1" ]]; then
    wandb_gen+=(--wandb --wandb-project "$WANDB_PROJECT" --wandb-tags "$WANDB_TAGS")
    if [[ -n "$WANDB_ENTITY" ]]; then
      wandb_gen+=(--wandb-entity "$WANDB_ENTITY")
    fi
    wandb_gen+=(--wandb-name "online_gen_${REWARD_OBJECTIVE}_tau${REWARD_TAU}_beta${REWARD_BETA}_${suffix}")
  fi

  extra_gen=()
  if [[ "$DETERMINISTIC" == "1" ]]; then
    extra_gen+=(--deterministic)
  fi

  "$PYTHON_BIN" -m online.generate_traces \
    --adv-model "$adv_model_path" \
    --pensieve-model "$model_path" \
    --episodes "$EPISODES" \
    --seed "$SEED" \
    --bw-min "$BW_MIN" --bw-max "$BW_MAX" \
    --obs-history "$OBS_HISTORY" \
    --oracle-window "$ORACLE_WINDOW" \
    --smooth-penalty-scale "$SMOOTH_PENALTY_SCALE" \
    --reward-objective "$REWARD_OBJECTIVE" \
    --reward-tau "$REWARD_TAU" \
    --reward-beta "$REWARD_BETA" \
    --out-dir "$trace_out_dir" \
    --no-random \
    --flat-out-dir \
    "${extra_gen[@]}" \
    "${wandb_gen[@]}"
done
