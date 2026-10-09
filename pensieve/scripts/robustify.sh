#!/usr/bin/env bash
set -euo pipefail

LR=1e-5
ADV_TRACE_NUM=200
ADV_TRACE_DIR=./traces/finetune-mixed/
CLEAN_TRACE_DIR=./train/
CLEAN_RATIO=0.7
BATCH_EPISODES=16
ROBUSTIFY_INIT_ENTROPY_WEIGHT="${ROBUSTIFY_INIT_ENTROPY_WEIGHT:-0.01}"
ROBUSTIFY_FREEZE_ENTROPY_WEIGHT="${ROBUSTIFY_FREEZE_ENTROPY_WEIGHT:-1}"
WANDB_MODE=online
MAX_PARALLEL=${MAX_PARALLEL:-30}
# TEST_TRACE_NUM=100
EPOCHS=189 #* About 1.5k updates with 16-episode PPO batches.
EPISODES=$((ADV_TRACE_NUM * EPOCHS))

TRACE_TYPE_DIRS=()
while IFS= read -r trace_type_dir; do
  TRACE_TYPE_DIRS+=("${trace_type_dir}")
done < <(find "${ADV_TRACE_DIR}" -mindepth 1 -maxdepth 1 -type d | sort)
if [ "${#TRACE_TYPE_DIRS[@]}" -eq 0 ]; then
  if find "${ADV_TRACE_DIR}" -maxdepth 1 -type f ! -name "*.json" | grep -q .; then
    TRACE_TYPE_DIRS=("${ADV_TRACE_DIR%/}")
  else
    echo "No subdirectories or trace files found under ${ADV_TRACE_DIR}" >&2
    exit 1
  fi
fi

run_robustify() {
  local trace_dir="$1"
  local trace_type
  local -a entropy_args
  trace_type="$(basename "${trace_dir}")"
  entropy_args=(--robustify-init-entropy-weight "${ROBUSTIFY_INIT_ENTROPY_WEIGHT}")
  if [[ "${ROBUSTIFY_FREEZE_ENTROPY_WEIGHT}" == "1" ]]; then
    entropy_args+=(--robustify-freeze-entropy-weight)
  else
    entropy_args+=(--no-robustify-freeze-entropy-weight)
  fi
  echo "[start] robustify ${trace_type}"
  python Pensieve-PPO/src/inference.py --mode robustify \
    -m ppo/nn_model_ep_426300.pth \
    --adv-trace-dir "${trace_dir}" \
    --adv-trace-num "${ADV_TRACE_NUM}" \
    --robustify-episodes "${EPISODES}" --robustify-lr "${LR}" \
    --robustify-clean-trace-dir "${CLEAN_TRACE_DIR}" --robustify-clean-ratio "${CLEAN_RATIO}" \
    --robustify-batch-episodes "${BATCH_EPISODES}" \
    "${entropy_args[@]}" \
    --wandb "${WANDB_MODE}" --project robustify-pensieve-online --tags "${EPOCHS}e-${BATCH_EPISODES}batch,${trace_type}"
  echo "[done] robustify ${trace_type}"
}

pids=()
running=0
failed=0
interrupted=0

handle_interrupt() {
  if [ "${interrupted}" -ne 0 ]; then
    return
  fi
  interrupted=1
  echo "Interrupt received; forwarding SIGINT to running robustify jobs..." >&2
  for pid in "${pids[@]}"; do
    kill -INT "${pid}" 2>/dev/null || true
  done
}

trap handle_interrupt INT TERM
for trace_dir in "${TRACE_TYPE_DIRS[@]}"; do
  if [ "${interrupted}" -ne 0 ]; then
    break
  fi
  run_robustify "${trace_dir}" &
  pids+=("$!")
  running=$((running + 1))

  if [ "${running}" -ge "${MAX_PARALLEL}" ]; then
    if ! wait "${pids[0]}"; then
      if [ "${interrupted}" -eq 0 ]; then
        failed=1
      fi
    fi
    pids=("${pids[@]:1}")
    running=$((running - 1))
  fi
done

for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    if [ "${interrupted}" -eq 0 ]; then
      failed=1
    fi
  fi
done

if [ "${interrupted}" -ne 0 ]; then
  echo "Robustify interrupted by user." >&2
  exit 130
fi

if [ "${failed}" -ne 0 ]; then
  echo "At least one robustify process failed." >&2
  exit 1
fi
#   --project robust-pensieve-online-vs-offline --tags eval-pgd-none-obj7-${EPOCHS}e,${ADV_TRACE_NUM}train


# python Pensieve-PPO/src/inference.py --mode robustify \
#   -m ppo/nn_model_ep_426300.pth -t traces/robust-test/alltraces-obj1-0.01alpha/pgd-last-feat -n ${TEST_TRACE_NUM} \
#   --adv-trace-dir ${ADV_TRACE_DIR} \
#   --adv-trace-num ${ADV_TRACE_NUM} \
#   --robustify-episodes ${EPISODES} --robustify-lr ${LR} \
#   --project robust-pensieve-online-vs-offline --tags eval-pgd-obj1-${EPOCHS}e,${ADV_TRACE_NUM}train

# python Pensieve-PPO/src/inference.py --mode robustify \
#   -m ppo/nn_model_ep_426300.pth -t traces/robust-test/alltraces-obj2-0.01alpha/pgd-last-feat -n ${TEST_TRACE_NUM} \
#   --adv-trace-dir ${ADV_TRACE_DIR} \
#   --adv-trace-num ${ADV_TRACE_NUM} \
#   --robustify-episodes ${EPISODES} --robustify-lr ${LR} \
#   --project robust-pensieve-online-vs-offline --tags eval-pgd-obj2-${EPOCHS}e,${ADV_TRACE_NUM}train

# python Pensieve-PPO/src/inference.py --mode robustify \
#   -m ppo/nn_model_ep_426300.pth -t traces/robust-test/100traces-obj7test-0.01alpha/pgd-last-feat -n ${TEST_TRACE_NUM} \
#   --adv-trace-dir ${ADV_TRACE_DIR} \
#   --adv-trace-num ${ADV_TRACE_NUM} \
#   --robustify-episodes ${EPISODES} --robustify-lr ${LR} \
#   --project robust-pensieve-online-vs-offline --tags eval-pgd-obj7-${EPOCHS}e,${ADV_TRACE_NUM}train

# python Pensieve-PPO/src/inference.py --mode robustify \
#   -m ppo/nn_model_ep_426300.pth -t traces/robust-test/100traces-obj7test-0.01alpha/pgd-last-none -n ${TEST_TRACE_NUM} \
#   --adv-trace-dir ${ADV_TRACE_DIR} \
#   --adv-trace-num ${ADV_TRACE_NUM} \
#   --robustify-episodes ${EPISODES} --robustify-lr ${LR} \
#   --project robust-pensieve-online-vs-offline --tags eval-pgd-none-obj7-${EPOCHS}e,${ADV_TRACE_NUM}train

# python Pensieve-PPO/src/inference.py --mode robustify \
#   -m ppo/nn_model_ep_426300.pth -t traces/robust-test/dtgaptest-100t-0.9oracle/dtgap-sklearn-feat -n ${TEST_TRACE_NUM} \
#   --adv-trace-dir ${ADV_TRACE_DIR} \
#   --adv-trace-num ${ADV_TRACE_NUM} \
#   --robustify-episodes ${EPISODES} --robustify-lr ${LR} \
#   --project robust-pensieve-online-vs-offline --tags eval-dt-sk-feat-${EPOCHS}e,${ADV_TRACE_NUM}train

# python Pensieve-PPO/src/inference.py --mode robustify \
#   -m ppo/nn_model_ep_426300.pth -t traces/robust-test/dtgaptest-100t-0.9oracle/dtgap-h2o-feat -n ${TEST_TRACE_NUM} \
#   --adv-trace-dir ${ADV_TRACE_DIR} \
#   --adv-trace-num ${ADV_TRACE_NUM} \
#   --robustify-episodes ${EPISODES} --robustify-lr ${LR} \
#   --project robust-pensieve-online-vs-offline --tags eval-dt-h2o-feat-${EPOCHS}e,${ADV_TRACE_NUM}train

# python Pensieve-PPO/src/inference.py --mode robustify \
#   -m ppo/nn_model_ep_426300.pth -t traces/robust-test/dtgaptest-100t-0.9oracle/dtgap-sklearn-none -n ${TEST_TRACE_NUM} \
#   --adv-trace-dir ${ADV_TRACE_DIR} \
#   --adv-trace-num ${ADV_TRACE_NUM} \
#   --robustify-episodes ${EPISODES} --robustify-lr ${LR} \
#   --project robust-pensieve-online-vs-offline --tags eval-dt-sk-none-${EPOCHS}e,${ADV_TRACE_NUM}train

# python Pensieve-PPO/src/inference.py --mode robustify \
#   -m ppo/nn_model_ep_426300.pth -t ./online/adv-traces/pensieve -n ${TEST_TRACE_NUM} \
#   --adv-trace-dir ${ADV_TRACE_DIR} \
#   --adv-trace-num ${ADV_TRACE_NUM} \
#   --robustify-episodes ${EPISODES} --robustify-lr ${LR} \
#   --project robust-pensieve-online-vs-offline --tags eval-hotnets-${EPOCHS}e,${ADV_TRACE_NUM}train

# python Pensieve-PPO/src/inference.py --mode robustify \
#   -m ppo/nn_model_ep_426300.pth -t ./test -n ${TEST_TRACE_NUM} \
#   --adv-trace-dir ${ADV_TRACE_DIR} \
#   --adv-trace-num ${ADV_TRACE_NUM} \
#   --robustify-episodes ${EPISODES} --robustify-lr ${LR} \
#   --project robust-pensieve-online-vs-offline --tags eval-clean-${EPOCHS}e,${ADV_TRACE_NUM}train
