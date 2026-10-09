# Online RL Attacker for Park Load-Balance (Torch)

This directory mirrors the `sdr_new/sdr/online` experiment:

1. Train an RL attacker that generates challenging workload traces.
2. Generate attacker rollout traces (and optional random baseline traces).
3. Evaluate oracle gap for clean vs adversarial traces.

## What the attacker controls

At each attacker step, action `a_t = [a_dt, a_size]` in `[0, 1]^2` is mapped with log scaling to:

- next inter-arrival time `dt_t` in `[dt_min, dt_max]`
- next job size `size_t` in `[size_min, size_max]`

Reward is **oracle gap** (step-level):

- `reward_t = r_oracle_window_t - r_victim_window_t`

where the oracle is a rolling best-of-heuristics reference policy evaluated on the same generated arrival window. The default reference window is `5`.

## Files

- `attacker_env.py`: RL attacker environment (`OnlineTraceAttackerEnv`)
- `train_adv.py`: train attacker PPO (Stable-Baselines3)
- `../scripts/generate_rl_attacker_traces.py`: rollout a trained attacker policy to save traces
- `eval_oracle_gap.py`: compare oracle gap on clean vs adversarial traces
- `sim_core.py`: shared simulator + oracle/victim evaluation utilities

## Prerequisites

Run from repository root:

```bash
cd /nfs/park-load-balance-torch
```

Install dependencies (in your env):

```bash
pip install -r requirements.txt
pip install stable-baselines3
```

You also need a trained victim checkpoint from `train_ppo.py`, e.g.:

- `runs/ppo_lb_torch_from_trace_stable/checkpoint_0200.pt`

## 1) Train attacker

```bash
python -m online.train_adv \
  --victim-checkpoint /nfs/park-load-balance-torch/runs/ppo_lb_torch_from_trace_stable/checkpoint_0200.pt \
  --episode-jobs 1000 \
  --total-timesteps 300000 \
  --dt-min 1 --dt-max 500 \
  --size-min 10 --size-max 5000 \
  --reference-window 5 \
  --out-dir /nfs/park-load-balance-torch/online/models
```

Outputs:

- `online/models/adv_ppo_lb_YYYYmmdd-HHMMSS.zip`
- `online/models/adv_ppo_lb_YYYYmmdd-HHMMSS.config.json`

## 2) Generate attacker rollout traces

```bash
python scripts/generate_rl_attacker_traces.py \
  --victim-checkpoint /nfs/park-load-balance-torch/runs/ppo_lb_torch_from_trace_stable/checkpoint_0200.pt \
  --attacker-model /nfs/park-load-balance-torch/online/models/adv_ppo_lb_YYYYmmdd-HHMMSS.zip \
  --episodes 20 \
  --episode-jobs 1000 \
  --reference-window 5 \
  --out-dir /nfs/park-load-balance-torch/online/adv-traces/RL-attacker
```

Outputs:

- `online/adv-traces/RL-attacker/episode_0000.csv|json` (attacker-policy rollout)

Notes:

- Add `--deterministic` for deterministic attacker rollout.
- This script only rolls out a saved attacker policy. It does not evaluate trace quality or search for the strongest trace after generation.
- Generation disables oracle-gap computation for speed. If the attacker was trained with the `last_gap` feature enabled, that feature is present but fixed at `0.0` during generation.
- `--reference-window` is retained for config compatibility, but it does not change the generated jobs while oracle-gap computation is disabled.
- Use `scripts/generate_uniform_random_traces.py` separately if you want a random baseline trace set.

## 3) Evaluate oracle gap (clean vs adv)

```bash
python -m online.eval_oracle_gap \
  --victim-checkpoint /nfs/park-load-balance-torch/runs/ppo_lb_torch_from_trace_stable/checkpoint_0200.pt \
  --clean-dir /nfs/park-load-balance-torch/traces/workload_benign_stable \
  --adv-dir /nfs/park-load-balance-torch/online/adv-traces/aware \
  --out-dir /nfs/park-load-balance-torch/online/eval
```

Outputs:

- `online/eval/oracle_gap_clean.csv`
- `online/eval/oracle_gap_adv.csv`
- `online/eval/oracle_gap_summary.json`

`oracle_gap_summary.json` includes:

- `clean.oracle_gap_mean`
- `adv.oracle_gap_mean`
- `delta_oracle_gap_mean = adv - clean`
- per-job versions of the same metrics

## One-Command Pipeline

Use the wrapper script to run all stages:

```bash
cd /nfs/park-load-balance-torch
bash online/run_online.sh
```

It will:

1. Ensure clean traces exist (generate if needed).
2. Train attacker model.
3. Generate RL-attacker traces.
4. Generate random baseline traces.
5. Evaluate clean vs RL-attacker.
6. Evaluate clean vs random.
7. Evaluate RL-attacker vs random (primary fairness check).

You can override defaults with environment variables, for example:

```bash
VICTIM_CKPT=/nfs/park-load-balance-torch/runs/ppo_lb_torch_from_trace_stable/checkpoint_0200.pt \
TOTAL_TIMESTEPS=150000 \
EPISODES=10 \
bash /nfs/park-load-balance-torch/online/run_online.sh
```

`run_online.sh` now auto-infers attacker bounds from clean traces by default:
- `AUTO_BOUNDS_FROM_CLEAN=1` (default): sets `DT_MAX` and `SIZE_MAX` from clean-trace q99 statistics (with headroom).
- set `AUTO_BOUNDS_FROM_CLEAN=0` to use manual bounds directly.
- the wrapper generates random baseline traces separately with the same dt/size bounds passed to the attacker rollout script.

Primary summary to inspect after running:

- `online/eval_rl_attacker_vs_random/oracle_gap_summary.json`

## Optional: custom service rates

Default service rates (10 servers) follow Park:

`[0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05]`

To override, pass comma-separated values to each script:

```bash
--service-rates 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1
```

## Troubleshooting

- If `stable_baselines3` is missing, install it in the active environment.
- Run `online/` modules with `python -m online.<script>` and `scripts/` entrypoints with `python scripts/<name>.py` from repo root so imports resolve correctly.
