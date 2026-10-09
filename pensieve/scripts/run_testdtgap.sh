#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=libthrottle.sh
# shellcheck disable=SC1091
source "$SCRIPT_DIR/libthrottle.sh"

NUM_TRACES=142
SEED=42
MODEL_DIR=./models/
TRACE_DIR=./test/
STEPS=-1

WANDB_MODE="${WANDB_MODE:-online}" # online | offline | disabled
WANDB_PROJECT="${WANDB_PROJECT:-attack-pensieve-gap}"

POLICY_RULES_FILE_H2O="rules/pensieve_dagger_rules_h2o.pl"
POLICY_RULES_FILE_SKLEARN="rules/pensieve_dagger_rules_sklearn.pl"
FEATURE_RULES_FILE="rules/feature_space_rules_coupling.pl"

#* Symbolic DT candidate generation knobs.
SYM_MAX_RULES=0
SYM_SAMPLES_PER_RULE=0

#* dt-truegap oracle-gap knobs (can be overridden per run).
TRUEGAP_MAX_CANDIDATES_PER_ACTION=0
TRUEGAP_ORACLE_MIN_RATIO=0.9

# Convert negative NUM_TRACES to "all" for tagging
if (( NUM_TRACES < 0 )); then
  NUM_TRACES_TAG="all"
else
  NUM_TRACES_TAG="${NUM_TRACES}"
fi

TAGS="testdtgap-${NUM_TRACES_TAG}t-${TRUEGAP_ORACLE_MIN_RATIO}oracle-189e"

THROTTLE_LAUNCH_SLEEP_S="${THROTTLE_LAUNCH_SLEEP_S:-5}"
throttle_init

BASE_ARGS=(
    Pensieve-PPO/src/inference.py
    --mode end2end
    -t "$TRACE_DIR"
    -n "$NUM_TRACES"
    --steps "$STEPS"
    --seed "$SEED"
    --two-pass
    --wandb "$WANDB_MODE"
    --project "$WANDB_PROJECT"
    --tags "$TAGS"
    --attack
    --attack-method dt-truegap
    --no-coupling
    --no-history-throughput
    --sym-max-rules "$SYM_MAX_RULES"
    --sym-samples-per-rule "$SYM_SAMPLES_PER_RULE"
    --sym-truegap-max-candidates-per-action "$TRUEGAP_MAX_CANDIDATES_PER_ACTION"
    --sym-truegap-oracle-min-ratio "$TRUEGAP_ORACLE_MIN_RATIO"
    --model-dir "$MODEL_DIR"
)

run_case() {
    local name="$1"
    shift
    throttle_run python "${BASE_ARGS[@]}" --name "$name" "$@"
}

run_case_nohist() {
    local name="$1"
    shift
    run_case "$name" --no-history-delay "$@"
}

run_case_hist() {
    local name="$1"
    shift
    run_case "$name" --history-delay "$@"
}

#* Clean baseline (no attack).
throttle_run python Pensieve-PPO/src/inference.py --mode end2end \
    -t "$TRACE_DIR" -n "$NUM_TRACES" --steps "$STEPS" --seed "$SEED" \
    --two-pass --wandb "$WANDB_MODE" --project "$WANDB_PROJECT" --tags "$TAGS" \
    --no-attack --no-coupling --no-rules \
    --name clean

#* Compare:
#* 1) with/without feature-space rules (--rules)
#* 2) with/without trace rules (--no-trace-rules)
#* 3) H2O vs sklearn policy rule sets

# #* H2O rules.

# run_case_nohist dtgap-h2o-none \
#     --policy-rules-file "$POLICY_RULES_FILE_H2O" \
#     --no-rules \
#     --no-trace-rules

# run_case_nohist dtgap-h2o-feat \
#     --policy-rules-file "$POLICY_RULES_FILE_H2O" \
#     --rules --rules-file "$FEATURE_RULES_FILE" --no-rules-perturb-both \
#     --no-trace-rules

# run_case_nohist dtgap-h2o-trace \
#     --policy-rules-file "$POLICY_RULES_FILE_H2O" \
#     --no-rules
# run_case_nohist dtgap-h2o-both \
#     --policy-rules-file "$POLICY_RULES_FILE_H2O" \
#     --rules --rules-file "$FEATURE_RULES_FILE" --no-rules-perturb-both

# #* H2O rules + delay-history attack.

# run_case_hist dtgap-h2o-none-hist \
#     --policy-rules-file "$POLICY_RULES_FILE_H2O" \
#     --no-rules \
#     --no-trace-rules

# run_case_hist dtgap-h2o-feat-hist \
#     --policy-rules-file "$POLICY_RULES_FILE_H2O" \
#     --rules --rules-file "$FEATURE_RULES_FILE" --no-rules-perturb-both \
#     --no-trace-rules

# run_case_hist dtgap-h2o-trace-hist \
#     --policy-rules-file "$POLICY_RULES_FILE_H2O" \
#     --no-rules
# run_case_hist dtgap-h2o-both-hist \
#     --policy-rules-file "$POLICY_RULES_FILE_H2O" \
#     --rules --rules-file "$FEATURE_RULES_FILE" --no-rules-perturb-both

#* sklearn rules.

run_case_nohist dtgap-sk-none \
    --policy-rules-file "$POLICY_RULES_FILE_SKLEARN" \
    --no-rules \
    --no-trace-rules

run_case_nohist dtgap-sk-feat \
    --policy-rules-file "$POLICY_RULES_FILE_SKLEARN" \
    --rules --rules-file "$FEATURE_RULES_FILE" --no-rules-perturb-both \
    --no-trace-rules

# run_case_nohist dtgap-sklearn-trace \
#     --policy-rules-file "$POLICY_RULES_FILE_SKLEARN" \
#     --no-rules
# run_case_nohist dtgap-sklearn-both \
#     --policy-rules-file "$POLICY_RULES_FILE_SKLEARN" \
#     --rules --rules-file "$FEATURE_RULES_FILE" --no-rules-perturb-both

# #* sklearn rules + delay-history attack.

# run_case_hist dtgap-sklearn-none-hist \
#     --policy-rules-file "$POLICY_RULES_FILE_SKLEARN" \
#     --no-rules \
#     --no-trace-rules

# run_case_hist dtgap-sklearn-feat-hist \
#     --policy-rules-file "$POLICY_RULES_FILE_SKLEARN" \
#     --rules --rules-file "$FEATURE_RULES_FILE" --no-rules-perturb-both \
#     --no-trace-rules

# run_case_hist dtgap-sklearn-both-hist \
#     --policy-rules-file "$POLICY_RULES_FILE_SKLEARN" \
#     --rules --rules-file "$FEATURE_RULES_FILE" --no-rules-perturb-both
# run_case_hist dtgap-sklearn-trace-hist \
#     --policy-rules-file "$POLICY_RULES_FILE_SKLEARN" \
#     --no-rules

throttle_wait_all
