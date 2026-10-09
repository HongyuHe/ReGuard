#!/usr/bin/env bash
set -euo pipefail

BACKEND="${BACKEND:-sklearn}" # h2o | sklearn

python rules/mine/gen_window_stats_thresholds.py --adv-csv rules/mine/data/pgd-feat.csv --clean-csv rules/mine/data/clean.csv
python rules/mine/gen_window_stats_thresholds.py --adv-csv rules/mine/data/hotnets19.csv --clean-csv rules/mine/data/clean.csv

python rules/mine/mine_risky_rules_dt.py --adv-stats rules/mine/data/hotnets19_states_stats.csv \
    --thresholds rules/mine/data/clean_stats_thresholds.csv \
    --backend "$BACKEND" --gap-threshold-pct 80

python rules/mine/mine_risky_rules_dt.py --adv-stats rules/mine/data/pgd-feat_states_stats.csv \
    --thresholds rules/mine/data/clean_stats_thresholds.csv \
    --backend "$BACKEND" --gap-threshold-pct 80

python rules/mine/interpret_pl_rules.py --rules-dir rules/mine/out \
    --thresholds-csv rules/mine/data/clean_stats_thresholds.csv