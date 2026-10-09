#!/usr/bin/env python3
from __future__ import annotations

"""
#* Example from the repo root:

time python scripts/eval_trace_dir_oracle_regret.py \
  --victim-checkpoint models/parklb_v1_350k.pt \
  --trace-dir traces/maxgap \
  --eval-count 100 \
  --reference-window 5 \
  --wandb \
  --wandb-project park-lb-eval-v2 \
  --wandb-name maxgap \
  --out-dir online/eval/v2/maxgap

time python scripts/eval_trace_dir_oracle_regret.py \
  --victim-checkpoint models/parklb_v1_350k.pt \
  --trace-dir traces/random \
  --eval-count 100 \
  --reference-window 5 \
  --wandb \
  --wandb-project park-lb-eval-v2 \
  --wandb-name random \
  --out-dir online/eval/v2/random
  
time python scripts/eval_trace_dir_oracle_regret.py \
  --victim-checkpoint models/parklb_v1_350k.pt \
  --trace-dir traces/clean \
  --eval-count 100 \
  --reference-window 5 \
  --wandb \
  --wandb-project park-lb-eval-v2 \
  --wandb-name clean \
  --out-dir online/eval/v2/clean

time python scripts/eval_trace_dir_oracle_regret.py \
  --victim-checkpoint models/parklb_v1_350k.pt \
  --trace-dir traces/pgd-surrogate \
  --eval-count 100 \
  --reference-window 5 \
  --wandb \
  --wandb-project park-lb-eval-v2 \
  --wandb-name pgd-surrogate \
  --out-dir online/eval/v2/pgd-surrogate
  
time python scripts/eval_trace_dir_oracle_regret.py \
  --victim-checkpoint models/parklb_v1_350k.pt \
  --trace-dir traces/genet \
  --eval-count 100 \
  --reference-window 5 \
  --wandb \
  --wandb-project park-lb-eval-v2 \
  --wandb-name genet \
  --out-dir online/eval/v2/genet

time python scripts/eval_trace_dir_oracle_regret.py \
  --victim-checkpoint models/parklb_v1_350k.pt \
  --trace-dir traces/oracle2step \
  --eval-count 100 \
  --reference-window 5 \
  --wandb \
  --wandb-project park-lb-eval-v2 \
  --wandb-name oracle2step \
  --out-dir online/eval/v2/oracle2step

#* With shield:
time python scripts/eval_trace_dir_oracle_regret.py \
  --victim-checkpoint models/parklb_v1_350k.pt \
  --shield-rules-prefix shield/out/maxgap_shield_h2o_gap30_regret10 \
  --trace-dir traces/maxgap \
  --eval-count 100 \
  --reference-window 5 \
  --wandb --wandb-tags shield \
  --wandb-project park-lb-eval-v2 \
  --wandb-name maxgap_shield \
  --out-dir online/eval/v2/maxgap_shield

time python scripts/eval_trace_dir_oracle_regret.py \
  --victim-checkpoint models/parklb_v1_350k.pt \
  --shield-rules-prefix shield/out/maxgap_shield_h2o_gap30_regret10 \
  --trace-dir traces/genet \
  --eval-count 100 \
  --reference-window 5 \
  --wandb --wandb-tags shield \
  --wandb-project park-lb-eval-v2 \
  --wandb-name genet_shield \
  --out-dir online/eval/v2/genet_shield

time python scripts/eval_trace_dir_oracle_regret.py \
  --victim-checkpoint models/parklb_v1_350k.pt \
  --shield-rules-prefix shield/out/maxgap_shield_h2o_gap30_regret10 \
  --trace-dir traces/pgd-surrogate \
  --eval-count 100 \
  --reference-window 5 \
  --wandb --wandb-tags shield \
  --wandb-project park-lb-eval-v2 \
  --wandb-name pgd-surrogate_shield \
  --out-dir online/eval/v2/pgd-surrogate_shield

time python scripts/eval_trace_dir_oracle_regret.py \
  --victim-checkpoint models/parklb_v1_350k.pt \
  --shield-rules-prefix shield/out/maxgap_shield_h2o_gap30_regret10 \
  --trace-dir traces/random \
  --eval-count 100 \
  --reference-window 5 \
  --wandb --wandb-tags shield \
  --wandb-project park-lb-eval-v2 \
  --wandb-name random_shield \
  --out-dir online/eval/v2/random_shield

time python scripts/eval_trace_dir_oracle_regret.py \
  --victim-checkpoint models/parklb_v1_350k.pt \
  --shield-rules-prefix shield/out/maxgap_shield_h2o_gap30_regret10 \
  --trace-dir traces/oracle2step \
  --eval-count 100 \
  --reference-window 5 \
  --wandb --wandb-tags shield \
  --wandb-project park-lb-eval-v2 \
  --wandb-name oracle2step_shield \
  --out-dir online/eval/v2/oracle2step_shield

time python scripts/eval_trace_dir_oracle_regret.py \
  --victim-checkpoint models/parklb_v1_350k.pt \
  --shield-rules-prefix shield/out/maxgap_shield_h2o_gap30_regret10 \
  --trace-dir traces/clean \
  --eval-count 100 \
  --reference-window 5 \
  --wandb --wandb-tags shield \
  --wandb-project park-lb-eval-v2 \
  --wandb-name clean_shield \
  --out-dir online/eval/v2/clean_shield

#* With random shield:
time python scripts/eval_trace_dir_oracle_regret.py \
  --victim-checkpoint models/parklb_v1_350k.pt \
  --shield-rules-prefix shield/out/random_shield_h2o_gap30_regret10 \
  --trace-dir traces/maxgap \
  --eval-count 100 \
  --reference-window 5 \
  --wandb --wandb-tags shield \
  --wandb-project park-lb-eval-v2 \
  --wandb-name maxgap_shield-random \
  --out-dir online/eval/v2/maxgap_shield-random

time python scripts/eval_trace_dir_oracle_regret.py \
  --victim-checkpoint models/parklb_v1_350k.pt \
  --shield-rules-prefix shield/out/random_shield_h2o_gap30_regret10 \
  --trace-dir traces/genet \
  --eval-count 100 \
  --reference-window 5 \
  --wandb --wandb-tags shield \
  --wandb-project park-lb-eval-v2 \
  --wandb-name genet_shield-random \
  --out-dir online/eval/v2/genet_shield-random

time python scripts/eval_trace_dir_oracle_regret.py \
  --victim-checkpoint models/parklb_v1_350k.pt \
  --shield-rules-prefix shield/out/random_shield_h2o_gap30_regret10 \
  --trace-dir traces/pgd-surrogate \
  --eval-count 100 \
  --reference-window 5 \
  --wandb --wandb-tags shield \
  --wandb-project park-lb-eval-v2 \
  --wandb-name pgd-surrogate_shield-random \
  --out-dir online/eval/v2/pgd-surrogate_shield-random

time python scripts/eval_trace_dir_oracle_regret.py \
  --victim-checkpoint models/parklb_v1_350k.pt \
  --shield-rules-prefix shield/out/random_shield_h2o_gap30_regret10 \
  --trace-dir traces/random \
  --eval-count 100 \
  --reference-window 5 \
  --wandb --wandb-tags shield \
  --wandb-project park-lb-eval-v2 \
  --wandb-name random_shield-random \
  --out-dir online/eval/v2/random_shield-random

time python scripts/eval_trace_dir_oracle_regret.py \
  --victim-checkpoint models/parklb_v1_350k.pt \
  --shield-rules-prefix shield/out/random_shield_h2o_gap30_regret10 \
  --trace-dir traces/clean \
  --eval-count 100 \
  --reference-window 5 \
  --wandb --wandb-tags shield \
  --wandb-project park-lb-eval-v2 \
  --wandb-name clean_shield-random \
  --out-dir online/eval/v2/clean_shield-random

time python scripts/eval_trace_dir_oracle_regret.py \
  --victim-checkpoint models/parklb_v1_350k.pt \
  --shield-rules-prefix shield/out/random_shield_h2o_gap30_regret10 \
  --trace-dir traces/oracle2step \
    --eval-count 100 \
    --reference-window 5 \
    --wandb --wandb-tags shield \
    --wandb-project park-lb-eval-v2 \
    --wandb-name oracle2step_shield-random \
    --out-dir online/eval/v2/oracle2step_shield-random

#* With pgd-surrogate shield:
time python scripts/eval_trace_dir_oracle_regret.py \
  --victim-checkpoint models/parklb_v1_350k.pt \
  --shield-rules-prefix shield/out/pgd-surrogate_shield_h2o_gap30_regret10 \
  --trace-dir traces/maxgap \
  --eval-count 100 \
  --reference-window 5 \
  --wandb --wandb-tags shield \
  --wandb-project park-lb-eval-v2 \
  --wandb-name maxgap_shield-pgd-surrogate \
  --out-dir online/eval/v2/maxgap_shield-pgd-surrogate

time python scripts/eval_trace_dir_oracle_regret.py \
  --victim-checkpoint models/parklb_v1_350k.pt \
  --shield-rules-prefix shield/out/pgd-surrogate_shield_h2o_gap30_regret10 \
  --trace-dir traces/genet \
  --eval-count 100 \
  --reference-window 5 \
  --wandb --wandb-tags shield \
  --wandb-project park-lb-eval-v2 \
  --wandb-name genet_shield-pgd-surrogate \
  --out-dir online/eval/v2/genet_shield-pgd-surrogate

time python scripts/eval_trace_dir_oracle_regret.py \
  --victim-checkpoint models/parklb_v1_350k.pt \
  --shield-rules-prefix shield/out/pgd-surrogate_shield_h2o_gap30_regret10 \
  --trace-dir traces/pgd-surrogate \
  --eval-count 100 \
  --reference-window 5 \
  --wandb --wandb-tags shield \
  --wandb-project park-lb-eval-v2 \
  --wandb-name pgd-surrogate_shield-pgd-surrogate \
  --out-dir online/eval/v2/pgd-surrogate_shield-pgd-surrogate

time python scripts/eval_trace_dir_oracle_regret.py \
  --victim-checkpoint models/parklb_v1_350k.pt \
  --shield-rules-prefix shield/out/pgd-surrogate_shield_h2o_gap30_regret10 \
  --trace-dir traces/random \
  --eval-count 100 \
  --reference-window 5 \
  --wandb --wandb-tags shield \
  --wandb-project park-lb-eval-v2 \
  --wandb-name random_shield-pgd-surrogate \
  --out-dir online/eval/v2/random_shield-pgd-surrogate

time python scripts/eval_trace_dir_oracle_regret.py \
  --victim-checkpoint models/parklb_v1_350k.pt \
  --shield-rules-prefix shield/out/pgd-surrogate_shield_h2o_gap30_regret10 \
  --trace-dir traces/clean \
  --eval-count 100 \
  --reference-window 5 \
  --wandb --wandb-tags shield \
  --wandb-project park-lb-eval-v2 \
  --wandb-name clean_shield-pgd-surrogate \
  --out-dir online/eval/v2/clean_shield-pgd-surrogate

time python scripts/eval_trace_dir_oracle_regret.py \
  --victim-checkpoint models/parklb_v1_350k.pt \
  --shield-rules-prefix shield/out/pgd-surrogate_shield_h2o_gap30_regret10 \
  --trace-dir traces/oracle2step \
    --eval-count 100 \
    --reference-window 5 \
    --wandb --wandb-tags shield \
    --wandb-project park-lb-eval-v2 \
    --wandb-name oracle2step_shield-pgd-surrogate \
    --out-dir online/eval/v2/oracle2step_shield-pgd-surrogate

#* With genet shield:
time python scripts/eval_trace_dir_oracle_regret.py \
  --victim-checkpoint models/parklb_v1_350k.pt \
  --shield-rules-prefix shield/out/genet_shield_h2o_gap30_regret10 \
  --trace-dir traces/maxgap \
  --eval-count 100 \
  --reference-window 5 \
  --wandb --wandb-tags shield \
  --wandb-project park-lb-eval-v2 \
  --wandb-name maxgap_shield-genet \
  --out-dir online/eval/v2/maxgap_shield-genet

time python scripts/eval_trace_dir_oracle_regret.py \
  --victim-checkpoint models/parklb_v1_350k.pt \
  --shield-rules-prefix shield/out/genet_shield_h2o_gap30_regret10 \
  --trace-dir traces/genet \
  --eval-count 100 \
  --reference-window 5 \
  --wandb --wandb-tags shield \
  --wandb-project park-lb-eval-v2 \
  --wandb-name genet_shield-genet \
  --out-dir online/eval/v2/genet_shield-genet

time python scripts/eval_trace_dir_oracle_regret.py \
  --victim-checkpoint models/parklb_v1_350k.pt \
  --shield-rules-prefix shield/out/genet_shield_h2o_gap30_regret10 \
  --trace-dir traces/pgd-surrogate \
  --eval-count 100 \
  --reference-window 5 \
  --wandb --wandb-tags shield \
  --wandb-project park-lb-eval-v2 \
  --wandb-name pgd-surrogate_shield-genet \
  --out-dir online/eval/v2/pgd-surrogate_shield-genet

time python scripts/eval_trace_dir_oracle_regret.py \
  --victim-checkpoint models/parklb_v1_350k.pt \
  --shield-rules-prefix shield/out/genet_shield_h2o_gap30_regret10 \
  --trace-dir traces/random \
  --eval-count 100 \
  --reference-window 5 \
  --wandb --wandb-tags shield \
  --wandb-project park-lb-eval-v2 \
  --wandb-name random_shield-genet \
  --out-dir online/eval/v2/random_shield-genet

time python scripts/eval_trace_dir_oracle_regret.py \
  --victim-checkpoint models/parklb_v1_350k.pt \
  --shield-rules-prefix shield/out/genet_shield_h2o_gap30_regret10 \
  --trace-dir traces/clean \
  --eval-count 100 \
  --reference-window 5 \
  --wandb --wandb-tags shield \
  --wandb-project park-lb-eval-v2 \
  --wandb-name clean_shield-genet \
  --out-dir online/eval/v2/clean_shield-genet

time python scripts/eval_trace_dir_oracle_regret.py \
  --victim-checkpoint models/parklb_v1_350k.pt \
  --shield-rules-prefix shield/out/genet_shield_h2o_gap30_regret10 \
  --trace-dir traces/oracle2step \
    --eval-count 100 \
    --reference-window 5 \
    --wandb --wandb-tags shield \
    --wandb-project park-lb-eval-v2 \
    --wandb-name oracle2step_shield-genet \
    --out-dir online/eval/v2/oracle2step_shield-genet

#* With oracle2step shield:
time python scripts/eval_trace_dir_oracle_regret.py \
  --victim-checkpoint models/parklb_v1_350k.pt \
  --shield-rules-prefix shield/out/oracle2step_shield_h2o_gap30_regret10 \
  --trace-dir traces/maxgap \
  --eval-count 100 \
  --reference-window 5 \
  --wandb --wandb-tags shield \
  --wandb-project park-lb-eval-v2 \
  --wandb-name maxgap_shield-oracle2step \
  --out-dir online/eval/v2/maxgap_shield-oracle2step

time python scripts/eval_trace_dir_oracle_regret.py \
  --victim-checkpoint models/parklb_v1_350k.pt \
  --shield-rules-prefix shield/out/oracle2step_shield_h2o_gap30_regret10 \
  --trace-dir traces/genet \
  --eval-count 100 \
  --reference-window 5 \
  --wandb --wandb-tags shield \
  --wandb-project park-lb-eval-v2 \
  --wandb-name genet_shield-oracle2step \
  --out-dir online/eval/v2/genet_shield-oracle2step

time python scripts/eval_trace_dir_oracle_regret.py \
  --victim-checkpoint models/parklb_v1_350k.pt \
  --shield-rules-prefix shield/out/oracle2step_shield_h2o_gap30_regret10 \
  --trace-dir traces/pgd-surrogate \
  --eval-count 100 \
  --reference-window 5 \
  --wandb --wandb-tags shield \
  --wandb-project park-lb-eval-v2 \
  --wandb-name pgd-surrogate_shield-oracle2step \
  --out-dir online/eval/v2/pgd-surrogate_shield-oracle2step

time python scripts/eval_trace_dir_oracle_regret.py \
  --victim-checkpoint models/parklb_v1_350k.pt \
  --shield-rules-prefix shield/out/oracle2step_shield_h2o_gap30_regret10 \
  --trace-dir traces/random \
  --eval-count 100 \
  --reference-window 5 \
  --wandb --wandb-tags shield \
  --wandb-project park-lb-eval-v2 \
  --wandb-name random_shield-oracle2step \
  --out-dir online/eval/v2/random_shield-oracle2step

time python scripts/eval_trace_dir_oracle_regret.py \
  --victim-checkpoint models/parklb_v1_350k.pt \
  --shield-rules-prefix shield/out/oracle2step_shield_h2o_gap30_regret10 \
  --trace-dir traces/clean \
  --eval-count 100 \
  --reference-window 5 \
  --wandb --wandb-tags shield \
  --wandb-project park-lb-eval-v2 \
  --wandb-name clean_shield-oracle2step \
  --out-dir online/eval/v2/clean_shield-oracle2step

time python scripts/eval_trace_dir_oracle_regret.py \
  --victim-checkpoint models/parklb_v1_350k.pt \
  --shield-rules-prefix shield/out/oracle2step_shield_h2o_gap30_regret10 \
  --trace-dir traces/oracle2step \
    --eval-count 100 \
    --reference-window 5 \
    --wandb --wandb-tags shield \
    --wandb-project park-lb-eval-v2 \
    --wandb-name oracle2step_shield-oracle2step \
    --out-dir online/eval/v2/oracle2step_shield-oracle2step

#* Genet-protected evals:
time python scripts/eval_trace_dir_oracle_regret.py \
  --victim-checkpoint models/parklb_v1_350k_genet.pt \
  --trace-dir traces/maxgap \
  --eval-count 100 \
  --reference-window 5 \
  --wandb \
  --wandb-project park-lb-eval-v2 \
  --wandb-name maxgap_genet \
  --out-dir online/eval/v2/maxgap_genet

time python scripts/eval_trace_dir_oracle_regret.py \
  --victim-checkpoint models/parklb_v1_350k_genet.pt \
  --trace-dir traces/random \
  --eval-count 100 \
  --reference-window 5 \
  --wandb \
  --wandb-project park-lb-eval-v2 \
  --wandb-name random_genet \
  --out-dir online/eval/v2/random_genet
  
time python scripts/eval_trace_dir_oracle_regret.py \
  --victim-checkpoint models/parklb_v1_350k_genet.pt \
  --trace-dir traces/clean \
  --eval-count 100 \
  --reference-window 5 \
  --wandb \
  --wandb-project park-lb-eval-v2 \
  --wandb-name clean_genet \
  --out-dir online/eval/v2/clean_genet
  
time python scripts/eval_trace_dir_oracle_regret.py \
  --victim-checkpoint models/parklb_v1_350k_genet.pt \
  --trace-dir traces/genet \
  --eval-count 100 \
  --reference-window 5 \
  --wandb \
  --wandb-project park-lb-eval-v2 \
  --wandb-name genet_genet \
  --out-dir online/eval/v2/genet_genet

time python scripts/eval_trace_dir_oracle_regret.py \
  --victim-checkpoint models/parklb_v1_350k_genet.pt \
  --trace-dir traces/pgd-surrogate \
  --eval-count 100 \
  --reference-window 5 \
  --wandb \
  --wandb-project park-lb-eval-v2 \
  --wandb-name pgd-surrogate_genet \
  --out-dir online/eval/v2/pgd-surrogate_genet

time python scripts/eval_trace_dir_oracle_regret.py \
  --victim-checkpoint models/parklb_v1_350k_genet.pt \
  --trace-dir traces/oracle2step \
  --eval-count 100 \
  --reference-window 5 \
  --wandb \
  --wandb-project park-lb-eval-v2 \
  --wandb-name oracle2step_genet \
  --out-dir online/eval/v2/oracle2step_genet

#* Finetune evals:

time python scripts/eval_trace_dir_oracle_regret.py \
  --victim-checkpoint models/parklb_v1_350k_finetune-maxgap.pt \
  --trace-dir traces/maxgap \
  --eval-count 100 \
  --reference-window 5 \
  --wandb \
  --wandb-project park-lb-eval-v2 \
  --wandb-name maxgap_finetune \
  --out-dir online/eval/v2/maxgap_finetune

time python scripts/eval_trace_dir_oracle_regret.py \
  --victim-checkpoint models/parklb_v1_350k_finetune-genet.pt \
  --trace-dir traces/genet \
  --eval-count 100 \
  --reference-window 5 \
  --wandb \
  --wandb-project park-lb-eval-v2 \
  --wandb-name genet_finetune \
  --out-dir online/eval/v2/genet_finetune

time python scripts/eval_trace_dir_oracle_regret.py \
  --victim-checkpoint models/parklb_v1_350k_finetune-clean.pt \
  --trace-dir traces/clean \
  --eval-count 100 \
  --reference-window 5 \
  --wandb \
  --wandb-project park-lb-eval-v2 \
  --wandb-name clean_finetune \
  --out-dir online/eval/v2/clean_finetune

time python scripts/eval_trace_dir_oracle_regret.py \
    --victim-checkpoint models/parklb_v1_350k_finetune-pgd.pt \
    --trace-dir traces/pgd-surrogate \
    --eval-count 100 \
    --reference-window 5 \
    --wandb \
    --wandb-project park-lb-eval-v2 \
    --wandb-name pgd-surrogate_finetune \
    --out-dir online/eval/v2/pgd-surrogate_finetune

time python scripts/eval_trace_dir_oracle_regret.py \
  --victim-checkpoint models/parklb_v1_350k_finetune-random.pt \
  --trace-dir traces/random \
  --eval-count 100 \
  --reference-window 5 \
  --wandb \
  --wandb-project park-lb-eval-v2 \
  --wandb-name random_finetune \
  --out-dir online/eval/v2/random_finetune

time python scripts/eval_trace_dir_oracle_regret.py \
    --victim-checkpoint models/parklb_v1_350k_finetune-oracle2step.pt \
    --trace-dir traces/oracle2step \
    --eval-count 100 \
    --reference-window 5 \
    --wandb \
    --wandb-project park-lb-eval-v2 \
    --wandb-name oracle2step_finetune \
    --out-dir online/eval/v2/oracle2step_finetune

#* Finetune clean evals:

time python scripts/eval_trace_dir_oracle_regret.py \
  --victim-checkpoint models/parklb_v1_350k_finetune-maxgap.pt \
  --trace-dir traces/clean \
  --eval-count 100 \
  --reference-window 5 \
  --wandb \
  --wandb-project park-lb-eval-v2 \
  --wandb-name maxgap_finetune-clean \
  --out-dir online/eval/v2/maxgap_finetune-clean

time python scripts/eval_trace_dir_oracle_regret.py \
  --victim-checkpoint models/parklb_v1_350k_finetune-genet.pt \
  --trace-dir traces/clean \
  --eval-count 100 \
    --reference-window 5 \
    --wandb \
    --wandb-project park-lb-eval-v2 \
    --wandb-name genet_finetune-clean \
    --out-dir online/eval/v2/genet_finetune-clean

time python scripts/eval_trace_dir_oracle_regret.py \
    --victim-checkpoint models/parklb_v1_350k_finetune-pgd.pt \
    --trace-dir traces/clean \
    --eval-count 100 \
    --reference-window 5 \
    --wandb \
    --wandb-project park-lb-eval-v2 \
    --wandb-name pgd-surrogate_finetune-clean \
    --out-dir online/eval/v2/pgd-surrogate_finetune-clean

time python scripts/eval_trace_dir_oracle_regret.py \
    --victim-checkpoint models/parklb_v1_350k_finetune-random.pt \
    --trace-dir traces/clean \
    --eval-count 100 \
    --reference-window 5 \
    --wandb \
    --wandb-project park-lb-eval-v2 \
    --wandb-name random_finetune-clean \
    --out-dir online/eval/v2/random_finetune-clean

time python scripts/eval_trace_dir_oracle_regret.py \
    --victim-checkpoint models/parklb_v1_350k_finetune-oracle2step.pt \
    --trace-dir traces/clean \
    --eval-count 100 \
    --reference-window 5 \
    --wandb \
    --wandb-project park-lb-eval-v2 \
    --wandb-name oracle2step_finetune-clean \
    --out-dir online/eval/v2/oracle2step_finetune-clean

#* Finetune evals with mixed traces:

time python scripts/eval_trace_dir_oracle_regret.py \
  --victim-checkpoint models/parklb_v1_350k_finetune-mix.pt \
  --trace-dir traces/maxgap \
  --eval-count 100 \
  --reference-window 5 \
  --wandb \
  --wandb-project park-lb-eval-v2 \
  --wandb-name maxgap_finetune-mix \
  --out-dir online/eval/v2/maxgap_finetune-mix

time python scripts/eval_trace_dir_oracle_regret.py \
  --victim-checkpoint models/parklb_v1_350k_finetune-mix.pt \
  --trace-dir traces/genet \
  --eval-count 100 \
  --reference-window 5 \
  --wandb \
  --wandb-project park-lb-eval-v2 \
  --wandb-name genet_finetune-mix \
  --out-dir online/eval/v2/genet_finetune-mix

time python scripts/eval_trace_dir_oracle_regret.py \
  --victim-checkpoint models/parklb_v1_350k_finetune-mix.pt \
  --trace-dir traces/clean \
  --eval-count 100 \
  --reference-window 5 \
  --wandb \
  --wandb-project park-lb-eval-v2 \
  --wandb-name clean_finetune-mix \
  --out-dir online/eval/v2/clean_finetune-mix

time python scripts/eval_trace_dir_oracle_regret.py \
  --victim-checkpoint models/parklb_v1_350k_finetune-mix.pt \
  --trace-dir traces/pgd-surrogate \
  --eval-count 100 \
  --reference-window 5 \
  --wandb \
  --wandb-project park-lb-eval-v2 \
  --wandb-name pgd-surrogate_finetune-mix \
  --out-dir online/eval/v2/pgd-surrogate_finetune-mix

time python scripts/eval_trace_dir_oracle_regret.py \
  --victim-checkpoint models/parklb_v1_350k_finetune-mix.pt \
  --trace-dir traces/random \
  --eval-count 100 \
  --reference-window 5 \
  --wandb \
  --wandb-project park-lb-eval-v2 \
  --wandb-name random_finetune-mix \
  --out-dir online/eval/v2/random_finetune-mix

time python scripts/eval_trace_dir_oracle_regret.py \
  --victim-checkpoint models/parklb_v1_350k_finetune-mix.pt \
  --trace-dir traces/oracle2step \
  --eval-count 100 \
  --reference-window 5 \
  --wandb \
  --wandb-project park-lb-eval-v2 \
  --wandb-name oracle2step_finetune-mix \
  --out-dir online/eval/v2/oracle2step_finetune-mix

#* Eval shielded iterations

time python scripts/eval_trace_dir_oracle_regret.py \
  --victim-checkpoint models/parklb_v1_350k.pt \
  --shield-rules-prefix shield/out/maxgap_shield_h2o_gap30_regret10 \
  --trace-dir traces/shielded-iter1 \
    --eval-count 100 \
    --reference-window 5 \
    --wandb --wandb-tags shielded-iters \
    --wandb-project park-lb-eval-v2 \
    --wandb-name shielded-iter1 \
    --out-dir online/eval/shielded-iters/shielded-iter1

time python scripts/eval_trace_dir_oracle_regret.py \
  --victim-checkpoint models/parklb_v1_350k.pt \
  --shield-rules-prefix shield/out/shielded-iter2_shield_h2o_gap30_regret10 \
  --trace-dir traces/shielded-iter3 \
    --eval-count 100 \
    --reference-window 5 \
    --wandb --wandb-tags shielded-iters \
    --wandb-project park-lb-eval-v2 \
    --wandb-name shielded-iter3 \
    --out-dir online/eval/shielded-iters/shielded-iter3

"""

import argparse

from lb_oracle_regret_utils import (
    VictimActor,
    collect_trace_files,
    evaluate_jobs_with_oracle_regret,
    init_wandb_run,
    load_jobs_from_trace,
    log_eval_to_wandb,
    normalize_wandb_tags,
    parse_service_rates,
    resolve_path,
    write_metrics_csvs,
)
from shield.shield_runtime import load_shield_from_prefix


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate oracle regret for an existing directory of Park LB traces")
    p.add_argument("--victim-checkpoint", type=str, required=True)
    p.add_argument("--trace-dir", type=str, required=True)
    p.add_argument("--reference-window", "--oracle-k", dest="reference_window", type=int, default=5)
    p.add_argument("--num-servers", type=int, default=None)
    p.add_argument("--service-rates", type=str, default=None)
    p.add_argument("--obs-high", type=float, default=None)
    p.add_argument("--reward-time-scale", type=float, default=None)
    p.add_argument("--shield-rules-prefix", type=str, default=None)
    p.add_argument("--shield-risk-trigger-steps", type=int, default=1)
    p.add_argument("--shield-cooldown-steps", type=int, default=0)
    p.add_argument(
        "--shield-empty-fallback",
        type=str,
        default="least_load_proxy",
        choices=["least_load_proxy", "keep_rl", "open_all"],
    )
    p.add_argument("--eval-count", type=int, default=0)
    p.add_argument("--trace-limit", type=int, default=0)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--wandb", dest="wandb", action="store_true")
    p.add_argument("--no-wandb", dest="wandb", action="store_false")
    p.add_argument("--wandb-project", type=str, default="park-lb-eval")
    p.add_argument("--wandb-name", type=str, default=None)
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-tags", nargs="*", default=None)
    p.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    p.set_defaults(wandb=True)
    args = p.parse_args()
    if int(args.eval_count) < 0:
        raise ValueError("--eval-count must be >= 0")
    if int(args.reference_window) <= 0:
        raise ValueError("--reference-window must be positive")
    if int(args.shield_risk_trigger_steps) <= 0:
        raise ValueError("--shield-risk-trigger-steps must be positive")
    if int(args.shield_cooldown_steps) < 0:
        raise ValueError("--shield-cooldown-steps must be >= 0")

    victim = VictimActor(checkpoint_path=str(resolve_path(str(args.victim_checkpoint))))
    cfg = dict(victim.config)
    num_servers = int(args.num_servers if args.num_servers is not None else cfg.get("num_servers", victim.num_servers))
    service_rates = parse_service_rates(
        args.service_rates,
        num_servers=int(num_servers),
        checkpoint_config=cfg,
    )
    obs_high = float(args.obs_high if args.obs_high is not None else cfg.get("obs_high", 500000.0))
    reward_time_scale = float(
        args.reward_time_scale if args.reward_time_scale is not None else cfg.get("reward_time_scale", 1000.0)
    )
    shield = None
    if args.shield_rules_prefix is not None:
        shield = load_shield_from_prefix(
            str(resolve_path(str(args.shield_rules_prefix))),
            num_servers=int(num_servers),
            risk_trigger_steps=int(args.shield_risk_trigger_steps),
            intervention_cooldown_steps=int(args.shield_cooldown_steps),
            empty_fallback=str(args.shield_empty_fallback),
        )

    trace_dir = resolve_path(str(args.trace_dir))
    trace_files = collect_trace_files(trace_dir)
    eval_count = int(args.eval_count)
    if int(args.trace_limit) > 0:
        eval_count = int(args.trace_limit) if int(eval_count) <= 0 else min(int(eval_count), int(args.trace_limit))
    if int(eval_count) > 0:
        if len(trace_files) < int(eval_count):
            raise RuntimeError(f"Expected {eval_count} traces in {trace_dir}; found {len(trace_files)}")
        trace_files = trace_files[: int(eval_count)]
    if len(trace_files) == 0:
        raise RuntimeError(f"no trace files found in {trace_dir}")

    episode_rows: list[dict[str, float | int | str]] = []
    step_rows: list[dict[str, float | int | str]] = []
    wandb_tags = normalize_wandb_tags(args.wandb_tags)
    wandb_run = init_wandb_run(
        enabled=bool(args.wandb),
        project=str(args.wandb_project),
        name=args.wandb_name,
        entity=args.wandb_entity,
        mode=str(args.wandb_mode),
        tags=list(wandb_tags),
        config={
            "script": "eval_trace_dir_oracle_regret",
            "victim_checkpoint": str(resolve_path(str(args.victim_checkpoint))),
            "trace_dir": str(trace_dir),
            "trace_count": int(len(trace_files)),
            "reference_window": int(args.reference_window),
            "oracle_k": int(args.reference_window),
            "service_rates": list(service_rates),
            "obs_high": float(obs_high),
            "reward_time_scale": float(reward_time_scale),
            "shield_rules_prefix": None if args.shield_rules_prefix is None else str(resolve_path(str(args.shield_rules_prefix))),
            "shield_risk_trigger_steps": int(args.shield_risk_trigger_steps),
            "shield_cooldown_steps": int(args.shield_cooldown_steps),
            "shield_empty_fallback": str(args.shield_empty_fallback),
            "wandb_tags": list(wandb_tags),
        },
    )

    for idx, trace_path in enumerate(list(trace_files)):
        jobs = load_jobs_from_trace(trace_path)
        episode_row, trace_step_rows = evaluate_jobs_with_oracle_regret(
            jobs=jobs,
            victim=victim,
            service_rates=list(service_rates),
            obs_high=float(obs_high),
            reward_time_scale=float(reward_time_scale),
            oracle_k=int(args.reference_window),
            trace_id=str(trace_path.name),
            trace_path=str(trace_path),
            source="trace_dir",
            shield=shield,
        )
        episode_row["trace_index"] = int(idx)
        for row in trace_step_rows:
            row["trace_index"] = int(idx)
        episode_rows.append(episode_row)
        step_rows.extend(trace_step_rows)

    out_dir = resolve_path(str(args.out_dir))
    episode_csv, step_csv = write_metrics_csvs(
        out_dir=out_dir,
        episode_rows=episode_rows,
        step_rows=step_rows,
    )
    log_eval_to_wandb(
        wandb_run,
        episode_rows=episode_rows,
        step_rows=step_rows,
    )
    if wandb_run is not None:
        wandb_run.summary["paths/episode_metrics_csv"] = str(episode_csv)
        wandb_run.summary["paths/step_metrics_csv"] = str(step_csv)
        wandb_run.summary["paths/trace_dir"] = str(trace_dir)
        wandb_run.finish()
    print(f"wrote: {episode_csv}")
    print(f"wrote: {step_csv}")


if __name__ == "__main__":
    main()
