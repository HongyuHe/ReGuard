# Park Load-Balance

This directory is a PyTorch version of Park's `load_balance` task:

- Environment: `load_balance_env.py`
- PPO trainer: `train_ppo.py`
- Evaluation script: `eval_policy.py`

## Install

```bash
cd /nfs/park-load-balance-torch
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Train (PPO)

```bash
python train_ppo.py \
  --num-servers 10 \
  --num-stream-jobs 1000 \
  --total-iters 200 \
  --rollout-steps 4096 \
  --save-dir runs/ppo_lb_torch
```

Checkpoints will be saved under `runs/ppo_lb_torch/checkpoint_XXXX.pt`.

Train from pre-generated workload traces (instead of online random job sampling):

```bash
python train_ppo.py \
  --num-servers 10 \
  --workload-trace-dir ./traces/workload_benign \
  --trace-select-mode random \
  --save-dir runs/ppo_lb_torch_from_trace
```

## Evaluate

```bash
python eval_policy.py \
  --checkpoint runs/ppo_lb_torch/checkpoint_0200.pt \
  --episodes 20
```

## Benign Service-Rate Traces

`LoadBalanceTorchEnv` accepts `service_rates` as a per-server vector (fixed during an episode), not a time-varying per-step trace.

Original `park` provides default sample values in code (`park/param.py`):
`[0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05]`.

Generate reusable benign trace files:

```bash
python scripts/generate_service_rate_traces.py \
  --out-dir traces/service_rates_benign \
  --num-traces 100 \
  --num-servers 10 \
  --mode near_park \
  --noise-std 0.05
```

Each trace is saved as both `.json` and `.csv`, with an `index.csv` for easy sampling.

## Benign Workload Traces (Arrivals + Sizes)

Generate exogenous job-stream traces for training/evaluation:

```bash
python scripts/generate_job_traces.py \
  --out-dir traces/workload_benign \
  --num-traces 100 \
  --num-jobs 1000 \
  --job-interval 55 \
  --job-size-pareto-shape 1.5 \
  --job-size-pareto-scale 100 \
  --time-mode absolute
```

Output files include both:
- `trace_XXXXX.csv` (`arrival_time` + `job_size`, or `dt` + `job_size`)
- `trace_XXXXX.json` (jobs list with `arrival_time` and `job_size`)

## Notes

- The reward matches the original task design: negative cumulative active-job-time between decisions.
- Observation format is `[server_loads..., incoming_job_size]`.
- Action is the selected server index.
- This rewrite is standalone and does not require importing `park`.
