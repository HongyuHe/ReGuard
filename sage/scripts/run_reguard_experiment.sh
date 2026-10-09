#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-$ROOT/.venv/bin/python}"
RUNTIME_PY="${RUNTIME_PY:-/home/${USER}/venvpy36/bin/python}"
SEED="${SEED:-42}"
TRACE_LIMIT="${TRACE_LIMIT:-24}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-$ROOT/attacks/adv_traces/gap-constrained-all-loss_50ms_300k/generated_manifest.json}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT/attacks/models/gap_adv_20260321_gap-constrained-all-loss_50ms_300k.config.json}"
RUN_NAME="${RUN_NAME:-sage_cc_seed${SEED}_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="$ROOT/results/$RUN_NAME"
PAPER_TRACE_LIMIT="${PAPER_TRACE_LIMIT:-0}"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "Sage must run on Linux; use the configured Linux server." >&2
  exit 2
fi
if [[ "$(uname -r)" != 4.19.112-0062* ]]; then
  echo "Sage requires kernel 4.19.112-0062; current kernel is $(uname -r)." >&2
  exit 2
fi
if [[ ! -x "$PY" || ! -x "$RUNTIME_PY" ]]; then
  echo "Run setup_post_reboot.sh and setup_python_envs.sh before this experiment." >&2
  exit 2
fi
if [[ ! -x /usr/local/bin/mm-adv-net ]]; then
  echo "Patched Mahimahi executable not found: /usr/local/bin/mm-adv-net" >&2
  exit 2
fi
if [[ "$(sysctl -n net.ipv4.ip_forward)" != "1" ]]; then
  echo "IPv4 forwarding is disabled; rerun setup_post_reboot.sh." >&2
  exit 2
fi
if [[ -e "$RUN_DIR" ]]; then
  echo "Run directory already exists: $RUN_DIR" >&2
  exit 2
fi

"$PY" "$ROOT/scripts/prepare_model.py" --results-dir "$RUN_DIR" --check-only
"$PY" "$ROOT/scripts/evaluate_paper.py" --results-dir "$RUN_DIR" --check-inputs

MANIFEST_DIR="$RUN_DIR/manifests"
DATASET_DIR="$RUN_DIR/shield/dataset"
RULE_DIR="$RUN_DIR/shield/rules"
EVAL_DIR="$RUN_DIR/evaluation"
RUNTIME_DIR="$RUN_DIR/runtime"
SUBSET_MANIFEST="$MANIFEST_DIR/generated_manifest.json"
mkdir -p "$MANIFEST_DIR" "$DATASET_DIR" "$RULE_DIR" "$EVAL_DIR" "$RUNTIME_DIR"

echo "Sage run directory: $RUN_DIR"
"$PY" scripts/subset_generated_manifest.py \
  --input "$SOURCE_MANIFEST" \
  --output "$SUBSET_MANIFEST" \
  --count "$TRACE_LIMIT"

"$RUNTIME_PY" sage_rl/rl_module/test_loading_sage_model.py

"$PY" scripts/gen_sage_shield_dataset.py \
  --generated-manifest "$SUBSET_MANIFEST" \
  --config-path "$CONFIG_PATH" \
  --out-dir "$DATASET_DIR" \
  --runtime-dir "$RUNTIME_DIR/dataset" \
  --skip-clean-rollout \
  --skip-thresholds

"$PY" scripts/train_sage_shield_dt.py \
  --dataset "$DATASET_DIR/sage_shield_dataset.csv" \
  --out-dir "$RULE_DIR" \
  --backend sklearn \
  --seed "$SEED"

"$PY" scripts/eval_sage_clean_vs_adv.py \
  --generated-manifest "$SUBSET_MANIFEST" \
  --config-path "$CONFIG_PATH" \
  --shield-rules-file "$RULE_DIR/sage_unified_shield_rules.json" \
  --out-dir "$EVAL_DIR" \
  --runtime-dir "$RUNTIME_DIR/evaluation" \
  --skip-clean-rollout \
  --seed "$SEED"

test -s "$DATASET_DIR/sage_shield_dataset.csv"
test -s "$RULE_DIR/sage_unified_shield_rules.json"
find "$EVAL_DIR" -type f -name eval_summary.csv -size +0c -print -quit | grep -q .
"$PY" "$ROOT/scripts/evaluate_paper.py" --results-dir "$RUN_DIR" \
  --trace-limit "$PAPER_TRACE_LIMIT" --workers "${EVAL_WORKERS:-1}"
"$PY" "$ROOT/scripts/prepare_model.py" --results-dir "$RUN_DIR"
echo "Sage experiment complete: $RUN_DIR"
