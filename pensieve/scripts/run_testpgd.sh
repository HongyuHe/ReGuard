#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=libthrottle.sh
# shellcheck disable=SC1091
source "$SCRIPT_DIR/libthrottle.sh"

NUM_TRACES=142
SEED=42
TRACE_DIR=./test/
MODEL_DIR=./models/
STEPS=-1
RULE_FILE="rules/feature_space_rules_coupling.pl"
PGD_OBJECTIVE_ID=7
PGD_ALPHA=0.1
PGD_ITERS=20
SUFFIX='189e'
TAGS="testpgd-${NUM_TRACES}t-obj${PGD_OBJECTIVE_ID}${SUFFIX}-${PGD_ALPHA}a-${PGD_ITERS}i"

THROTTLE_LAUNCH_SLEEP_S="${THROTTLE_LAUNCH_SLEEP_S:-5}"
throttle_init

# throttle_run python Pensieve-PPO/src/inference.py --mode end2end \
#     -t "$TRACE_DIR" -n "$NUM_TRACES" --steps "$STEPS" --seed "$SEED" \
#     --pgd-alpha "$PGD_ALPHA" --pgd-iters "$PGD_ITERS" \
#     --no-attack --no-coupling --no-rules \
#     --tags "$TAGS" --two-pass \
#     --model-dir "$MODEL_DIR" \
#     --name clean

throttle_run python Pensieve-PPO/src/inference.py --mode end2end \
    -t "$TRACE_DIR" -n "$NUM_TRACES" --steps "$STEPS" --seed "$SEED" \
    --pgd-alpha "$PGD_ALPHA" --pgd-iters "$PGD_ITERS" \
    --attack --no-coupling --no-rules \
    --objective-id "$PGD_OBJECTIVE_ID" \
    --no-history-delay --no-history-throughput \
    --tags "$TAGS" --two-pass \
    --no-trace-rules \
    --model-dir "$MODEL_DIR" \
    --name pgd-none

throttle_run python Pensieve-PPO/src/inference.py --mode end2end \
    -t "$TRACE_DIR" -n "$NUM_TRACES" --steps "$STEPS" --seed "$SEED" \
    --pgd-alpha "$PGD_ALPHA" --pgd-iters "$PGD_ITERS" \
    --no-coupling --no-history-delay --no-history-throughput \
    --attack --rules --rules-file "$RULE_FILE" --no-rules-perturb-both \
    --objective-id "$PGD_OBJECTIVE_ID" \
    --tags "$TAGS" --two-pass \
    --no-trace-rules \
    --model-dir "$MODEL_DIR" \
    --name pgd-feat

throttle_run python Pensieve-PPO/src/inference.py --mode end2end \
    -t "$TRACE_DIR" -n "$NUM_TRACES" --steps "$STEPS" --seed "$SEED" \
    --pgd-alpha "$PGD_ALPHA" --pgd-iters "$PGD_ITERS" \
    --no-coupling --no-history-delay --no-history-throughput \
    --attack --rules --rules-file "$RULE_FILE" --no-rules-perturb-both --rules-every-step \
    --objective-id "$PGD_OBJECTIVE_ID" \
    --tags "$TAGS" --two-pass \
    --no-trace-rules \
    --model-dir "$MODEL_DIR" \
    --name pgd-feat-step

throttle_run python Pensieve-PPO/src/inference.py --mode end2end \
    -t "$TRACE_DIR" -n "$NUM_TRACES" --steps "$STEPS" --seed "$SEED" \
    --pgd-alpha "$PGD_ALPHA" --pgd-iters "$PGD_ITERS" \
    --attack --coupling --no-rules \
    --objective-id "$PGD_OBJECTIVE_ID" \
    --no-history-delay --no-history-throughput \
    --tags "$TAGS" --two-pass \
    --no-trace-rules \
    --model-dir "$MODEL_DIR" \
    --name pgd-manual


# throttle_run python Pensieve-PPO/src/inference.py --mode end2end \
#     -t "$TRACE_DIR" -n "$NUM_TRACES" --steps "$STEPS" --seed "$SEED" \
#     --pgd-alpha "$PGD_ALPHA" --pgd-iters "$PGD_ITERS" \
#     --attack --no-coupling --no-rules \
#     --objective-id "$PGD_OBJECTIVE_ID" \
#     --no-history-delay --no-history-throughput \
#     --tags "$TAGS" --two-pass \
#     --name pgd-last-trace

# throttle_run python Pensieve-PPO/src/inference.py --mode end2end \
#     -t "$TRACE_DIR" -n "$NUM_TRACES" --steps "$STEPS" --seed "$SEED" \
#     --pgd-alpha "$PGD_ALPHA" --pgd-iters "$PGD_ITERS" \
#     --no-coupling --no-history-delay --no-history-throughput \
#     --attack --rules --rules-file "$RULE_FILE" --no-rules-perturb-both \
#     --objective-id "$PGD_OBJECTIVE_ID" \
#     --tags "$TAGS" --two-pass \
#     --name pgd-last-both

# throttle_run python Pensieve-PPO/src/inference.py --mode end2end \
#     -t "$TRACE_DIR" -n "$NUM_TRACES" --steps "$STEPS" --seed "$SEED" \
#     --pgd-alpha "$PGD_ALPHA" --pgd-iters "$PGD_ITERS" \
#     --no-coupling --no-history-delay --no-history-throughput \
#     --attack --rules --rules-file "$RULE_FILE" --no-rules-perturb-both --rules-every-step \
#     --objective-id "$PGD_OBJECTIVE_ID" \
#     --tags "$TAGS" --two-pass \
#     --name pgd-last-both-step

# throttle_run python Pensieve-PPO/src/inference.py --mode end2end \
#     -t "$TRACE_DIR" -n "$NUM_TRACES" --steps "$STEPS" --seed "$SEED" \
#     --pgd-alpha "$PGD_ALPHA" --pgd-iters "$PGD_ITERS" \
#     --attack --no-coupling --no-rules \
#     --objective-id "$PGD_OBJECTIVE_ID" \
#     --history-delay --history-throughput \
#     --tags "$TAGS" --two-pass \
#     --no-trace-rules \
#     --name pgd-hist-none

# throttle_run python Pensieve-PPO/src/inference.py --mode end2end \
#     -t "$TRACE_DIR" -n "$NUM_TRACES" --steps "$STEPS" --seed "$SEED" \
#     --pgd-alpha "$PGD_ALPHA" --pgd-iters "$PGD_ITERS" \
#     --attack --no-coupling --no-rules \
#     --objective-id "$PGD_OBJECTIVE_ID" \
#     --history-delay --history-throughput \
#     --tags "$TAGS" --two-pass \
#     --name pgd-hist-trace

# throttle_run python Pensieve-PPO/src/inference.py --mode end2end \
#     -t "$TRACE_DIR" -n "$NUM_TRACES" --steps "$STEPS" --seed "$SEED" \
#     --pgd-alpha "$PGD_ALPHA" --pgd-iters "$PGD_ITERS" \
#     --no-coupling --history-delay --no-history-throughput \
#     --attack --rules --rules-file "$RULE_FILE" --no-rules-perturb-both \
#     --objective-id "$PGD_OBJECTIVE_ID" \
#     --tags "$TAGS" --two-pass \
#     --no-trace-rules \
#     --name pgd-hist-feat

# throttle_run python Pensieve-PPO/src/inference.py --mode end2end \
#     -t "$TRACE_DIR" -n "$NUM_TRACES" --steps "$STEPS" --seed "$SEED" \
#     --pgd-alpha "$PGD_ALPHA" --pgd-iters "$PGD_ITERS" \
#     --no-coupling --history-delay --no-history-throughput \
#     --attack --rules --rules-file "$RULE_FILE" --no-rules-perturb-both \
#     --objective-id "$PGD_OBJECTIVE_ID" \
#     --tags "$TAGS" --two-pass \
#     --name pgd-hist-both

# throttle_run python Pensieve-PPO/src/inference.py --mode end2end \
#     -t "$TRACE_DIR" -n "$NUM_TRACES" --steps "$STEPS" --seed "$SEED" \
#     --pgd-alpha "$PGD_ALPHA" --pgd-iters "$PGD_ITERS" \
#     --no-coupling --history-delay --no-history-throughput \
#     --attack --rules --rules-file "$RULE_FILE" --no-rules-perturb-both \
#     --objective-id "$PGD_OBJECTIVE_ID" \
#     --tags "$TAGS" --two-pass \
#     --history-sparse \
#     --name pgd-hist-both-sparse

# throttle_run python Pensieve-PPO/src/inference.py --mode end2end \
#     -t "$TRACE_DIR" -n "$NUM_TRACES" --steps "$STEPS" --seed "$SEED" \
#     --pgd-alpha "$PGD_ALPHA" --pgd-iters "$PGD_ITERS" \
#     --no-coupling --history-delay --no-history-throughput \
#     --attack --rules --rules-file "$RULE_FILE" --no-rules-perturb-both --rules-every-step \
#     --objective-id "$PGD_OBJECTIVE_ID" \
#     --tags "$TAGS" --two-pass \
#     --name pgd-hist-both-step

# python Pensieve-PPO/src/inference.py --mode end2end \
#     -t "$TRACE_DIR" -n "$NUM_TRACES" --steps "$STEPS" --seed "$SEED" \
#     --attack --coupling --no-rules \
#     --history-delay --no-history-throughput \
#     --tags "$TAGS" --two-pass \
#     --history-sparse \
#     --name pgd-hist-manual




# python Pensieve-PPO/src/inference.py --mode end2end \
#     -t "$TRACE_DIR" -n "$NUM_TRACES" --steps "$STEPS" --seed "$SEED" \
#     --two-pass --attack --attack-method dt \
#     --no-coupling --history-delay --no-history-throughput \
#     --policy-rules-file rules/pensieve_dagger_rules_h2o.pl \
#     --no-rules \
#     --no-trace-rules \
#     --tags "$TAGS" \
#     --name dt-hist-h2o-none &
# sleep 2

# python Pensieve-PPO/src/inference.py --mode end2end \
#     -t "$TRACE_DIR" -n "$NUM_TRACES" --steps "$STEPS" --seed "$SEED" \
#     --two-pass --attack --attack-method dt \
#     --no-coupling --history-delay --no-history-throughput \
#     --policy-rules-file rules/pensieve_dagger_rules_h2o.pl \
#     --no-rules \
#     --tags "$TAGS" \
#     --name dt-hist-h2o-trace &
# sleep 2

# python Pensieve-PPO/src/inference.py --mode end2end \
#     -t "$TRACE_DIR" -n "$NUM_TRACES" --steps "$STEPS" --seed "$SEED" \
#     --two-pass --attack --attack-method dt \
#     --no-coupling --history-delay --no-history-throughput \
#     --policy-rules-file rules/pensieve_dagger_rules_h2o.pl \
#     --rules-file "$RULE_FILE" \
#     --no-trace-rules \
#     --tags "$TAGS" \
#     --name dt-hist-h2o-feat &
# sleep 2

# python Pensieve-PPO/src/inference.py --mode end2end \
#     -t "$TRACE_DIR" -n "$NUM_TRACES" --steps "$STEPS" --seed "$SEED" \
#     --two-pass --attack --attack-method dt \
#     --no-coupling --history-delay --no-history-throughput \
#     --policy-rules-file rules/pensieve_dagger_rules_h2o.pl \
#     --rules-file "$RULE_FILE" \
#     --tags "$TAGS" \
#     --name dt-hist-h2o-both &
# sleep 2

# python Pensieve-PPO/src/inference.py --mode end2end \
#     -t "$TRACE_DIR" -n "$NUM_TRACES" --steps "$STEPS" --seed "$SEED" \
#     --two-pass --attack --attack-method dt \
#     --no-coupling --history-delay --no-history-throughput \
#     --policy-rules-file rules/pensieve_dagger_rules_sklearn_1k.pl \
#     --no-rules \
#     --no-trace-rules \
#     --tags "$TAGS" \
#     --name dt-hist-sklearn-none &
# sleep 2

# python Pensieve-PPO/src/inference.py --mode end2end \
#     -t "$TRACE_DIR" -n "$NUM_TRACES" --steps "$STEPS" --seed "$SEED" \
#     --two-pass --attack --attack-method dt \
#     --no-coupling --history-delay --no-history-throughput \
#     --policy-rules-file rules/pensieve_dagger_rules_sklearn_1k.pl \
#     --no-rules \
#     --tags "$TAGS" \
#     --name dt-hist-sklearn-trace &
# sleep 2

# python Pensieve-PPO/src/inference.py --mode end2end \
#     -t "$TRACE_DIR" -n "$NUM_TRACES" --steps "$STEPS" --seed "$SEED" \
#     --two-pass --attack --attack-method dt \
#     --no-coupling --history-delay --no-history-throughput \
#     --policy-rules-file rules/pensieve_dagger_rules_sklearn_1k.pl \
#     --rules-file "$RULE_FILE" \
#     --tags "$TAGS" \
#     --no-trace-rules \
#     --name dt-hist-sklearn-feat &
# sleep 2

# python Pensieve-PPO/src/inference.py --mode end2end \
#     -t "$TRACE_DIR" -n "$NUM_TRACES" --steps "$STEPS" --seed "$SEED" \
#     --two-pass --attack --attack-method dt \
#     --no-coupling --history-delay --no-history-throughput \
#     --policy-rules-file rules/pensieve_dagger_rules_sklearn_1k.pl \
#     --rules-file "$RULE_FILE" \
#     --tags "$TAGS" \
#     --name dt-hist-sklearn-both &

# python Pensieve-PPO/src/inference.py --mode end2end \
#     -t "$TRACE_DIR" -n "$NUM_TRACES" --steps "$STEPS" --seed "$SEED" \
#     --no-coupling --history-delay --no-history-throughput \
#     --attack --rules --rules-file "$RULE_FILE"  --no-rules-perturb-both --rules-every-step \
#     --objective-id "$PGD_OBJECTIVE_ID" \
#     --tags "$TAGS" --two-pass \
#     --history-sparse \
#     --name pgd-hist-both-sparse-step &

throttle_wait_all
