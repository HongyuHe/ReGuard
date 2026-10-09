#!/usr/bin/env bash
set -euo pipefail

NUM_TRACES=-1 #* 127 in total
SEED=42
TRACE_DIR=./train-subset/
STEPS=-1
TAGS="5chunks"

python Pensieve-PPO/src/inference.py --mode end2end \
    -t "$TRACE_DIR" -n "$NUM_TRACES" --steps "$STEPS" --seed "$SEED" \
    --no-attack --no-coupling --no-rules \
    --tags "$TAGS" --two-pass \
    --name clean &

python Pensieve-PPO/src/inference.py --mode end2end \
    -t "$TRACE_DIR" -n "$NUM_TRACES" --steps "$STEPS" --seed "$SEED" \
    --attack --no-coupling --no-rules \
    --no-history-delay --no-history-throughput \
    --tags "$TAGS" --two-pass \
    --name pgd-last &

python Pensieve-PPO/src/inference.py --mode end2end \
    -t "$TRACE_DIR" -n "$NUM_TRACES" --steps "$STEPS" --seed "$SEED" \
    --attack --no-coupling --no-rules \
    --history-delay --history-throughput --two-pass \
    --tags "$TAGS" --two-pass \
    --name pgd-hist &

python Pensieve-PPO/src/inference.py --mode end2end \
    -t "$TRACE_DIR" -n "$NUM_TRACES" --steps "$STEPS" --seed "$SEED" \
    --attack --no-coupling --no-rules \
    --no-history-delay --no-history-throughput \
    --tags "$TAGS" --two-pass \
    --no-trace-rules \
    --name pgd-last-notrace &

python Pensieve-PPO/src/inference.py --mode end2end \
    -t "$TRACE_DIR" -n "$NUM_TRACES" --steps "$STEPS" --seed "$SEED" \
    --attack --no-coupling --no-rules \
    --history-delay --history-throughput --two-pass \
    --tags "$TAGS" --two-pass \
    --no-trace-rules \
    --name pgd-hist-notrace &

python Pensieve-PPO/src/inference.py --mode end2end \
    -t "$TRACE_DIR" -n "$NUM_TRACES" --steps "$STEPS" --seed "$SEED" \
    --no-coupling --no-history-delay --no-history-throughput \
    --attack --rules --rules-file rules/pensieve_feature_agent0_e80k_start0.pl --no-rules-perturb-both \
    --tags "$TAGS" --two-pass \
    --name pgd-last-rules &

python Pensieve-PPO/src/inference.py --mode end2end \
    -t "$TRACE_DIR" -n "$NUM_TRACES" --steps "$STEPS" --seed "$SEED" \
    --no-coupling --no-history-delay --no-history-throughput \
    --attack --rules --rules-file rules/pensieve_feature_agent0_e80k_start0.pl --no-rules-perturb-both \
    --tags "$TAGS" --two-pass \
    --no-trace-rules \
    --name pgd-last-rules-notrace &

python Pensieve-PPO/src/inference.py --mode end2end \
    -t "$TRACE_DIR" -n "$NUM_TRACES" --steps "$STEPS" --seed "$SEED" \
    --no-coupling --no-history-delay --no-history-throughput \
    --attack --rules --rules-file rules/pensieve_feature_agent0_e80k_start0.pl --no-rules-perturb-both \
    --tags "$TAGS" --two-pass \
    --history-sparse \
    --name pgd-last-rules-sparse &

python Pensieve-PPO/src/inference.py --mode end2end \
    -t "$TRACE_DIR" -n "$NUM_TRACES" --steps "$STEPS" --seed "$SEED" \
    --no-coupling --no-history-delay --no-history-throughput \
    --attack --rules --rules-file rules/pensieve_feature_agent0_e80k_start0.pl --no-rules-perturb-both --rules-every-step \
    --tags "$TAGS" --two-pass \
    --name pgd-last-rules-step &

python Pensieve-PPO/src/inference.py --mode end2end \
    -t "$TRACE_DIR" -n "$NUM_TRACES" --steps "$STEPS" --seed "$SEED" \
    --no-coupling --history-delay --no-history-throughput \
    --attack --rules --rules-file rules/pensieve_feature_agent0_e80k_start0.pl  --no-rules-perturb-both\
    --tags "$TAGS" --two-pass \
    --name pgd-hist-rules &

python Pensieve-PPO/src/inference.py --mode end2end \
    -t "$TRACE_DIR" -n "$NUM_TRACES" --steps "$STEPS" --seed "$SEED" \
    --no-coupling --history-delay --no-history-throughput \
    --attack --rules --rules-file rules/pensieve_feature_agent0_e80k_start0.pl  --no-rules-perturb-both\
    --tags "$TAGS" --two-pass \
    --no-trace-rules \
    --name pgd-hist-rules-notrace &

python Pensieve-PPO/src/inference.py --mode end2end \
    -t "$TRACE_DIR" -n "$NUM_TRACES" --steps "$STEPS" --seed "$SEED" \
    --no-coupling --history-delay --no-history-throughput \
    --attack --rules --rules-file rules/pensieve_feature_agent0_e80k_start0.pl  --no-rules-perturb-both\
    --tags "$TAGS" --two-pass \
    --history-sparse \
    --name pgd-hist-rules-sparse &

python Pensieve-PPO/src/inference.py --mode end2end \
    -t "$TRACE_DIR" -n "$NUM_TRACES" --steps "$STEPS" --seed "$SEED" \
    --no-coupling --history-delay --no-history-throughput \
    --attack --rules --rules-file rules/pensieve_feature_agent0_e80k_start0.pl --no-rules-perturb-both --rules-every-step \
    --tags "$TAGS" --two-pass \
    --name pgd-hist-rules-step &
