# Online Adversary (PPO)

This module trains an **online adversary** that controls the *network bandwidth* seen by a fixed, pretrained Pensieve ABR policy, with the goal of maximizing a configurable **regret-style QoE objective** while keeping bandwidth in a realistic range.

## Key Design Choices

- **Problem-space control**: adversary action is a scalar `a_t in [0, 1]`, mapped to bandwidth on a log scale as `bw_t = exp(log(bw_min) + a_t * (log(bw_max) - log(bw_min)))`, so `a_t = 0.5` yields the geometric-mean bandwidth.
- **Protocol does not observe `bw_t`**: Pensieve only sees *measured* throughput/delay after each chunk download.
- **Exact local oracle**: per step, compute `r_opt` as the best achievable linear QoE over the last `K=4` chunks by enumerating all `6^4 = 1296` bitrate sequences (vectorized in NumPy).
- **Reward**: `online.train_adv` supports `--reward-objective max_regret` for `reward_t = (r_opt_t - r_protocol_t) - λ * |bw_t - bw_{t-1}|`, `--reward-objective weighted_max_regret` for `reward_t = (r_opt_t - r_protocol_t) * r_opt_t - λ * |bw_t - bw_{t-1}|` (default), and `--reward-objective oracle_floor_max_regret` for `reward_t = (r_opt_t - r_protocol_t) * max(r_opt_t - τ, 0) - β * max(τ - r_opt_t, 0) - λ * |bw_t - bw_{t-1}|`, with `τ` set by `--reward-tau` and `β` set by `--reward-beta`.
- **Segments output**: generated traces are saved as `(bw, duration)` segments (plus a `.trace` file for replay).
- **Normalization matches SMT**: throughput and delay are normalized the same way as `Pensieve-PPO/src/smt.py` expects:
  - `throughput_norm = video_chunk_size_bytes / delay_ms / 1000`
  - `delay_norm = delay_ms / 1000 / 10`

## Install

Add these deps to your environment (in addition to `Pensieve-PPO/src/requirements.txt`):

```bash
pip install -r online/requirements.txt
```

## Train

```bash
python -m online.train_adv \
  --pensieve-model ppo/nn_model_ep_426300.pth \
  --total-steps 600000 \
  --lr 3e-4 \
  --seed 42 \
  --bw-min 0.8 --bw-max 4.8 \
  --reward-objective weighted_max_regret \
  --reward-tau 0.0 \
  --reward-beta 1.0 \
  --obs-history 10 \
  --oracle-window 4
```

Optional W&B:

```bash
python -m online.train_adv ... --wandb --wandb-project pensieve-online-adv
```

## Generate Traces (Adversary + Random Baseline)

```bash
python -m online.generate_traces \
  --adv-model online/models/adv_ppo_bw_YYYYmmdd-HHMMSS.zip \
  --episodes 200 \
  --seed 42 \
  --out-dir online/adv-traces \
  --deterministic
```

This produces:

- `online/adv-traces/pensieve/episode_0000.segments.json`
- `online/adv-traces/pensieve/episode_0000.trace`
- `online/adv-traces/random/episode_0000.segments.json`
- `online/adv-traces/random/episode_0000.trace`

The `.segments.json` file contains a list of segments:

```json
{"kind":"download","bw_mbps":1.2,"duration_s":0.53}
```

The `.trace` file is a cooked-trace style file (`time_s  bw_mbps`) derived from segment boundaries so it can be replayed by the existing simulator tooling.

## Metrics Logged to W&B

`online.train_adv` logs (from env `info`, every `--wandb-log-every` steps):

- `adv/action_raw`, `adv/action_clipped`, `adv/bw_mbps`, `adv/p_smoothing`
- `protocol/bitrate_idx`, `protocol/bitrate_mbps`, `protocol/buffer_s`, `protocol/rebuf_s`
- `protocol/delay_ms`, `protocol/throughput_norm`, `protocol/delay_norm`
- `oracle/r_opt`, `oracle/r_protocol`, `oracle/gap`, `oracle/gap_weighted`
- `oracle/r_opt_above_tau`, `oracle/r_opt_below_tau`, `oracle/gap_oracle_floor_weighted`, `oracle/oracle_floor_penalty`
- `env/reward_core`, `env/reward`
- `env/chunk_idx`

`online.generate_traces` logs:

- `eval/pensieve_qoe_protocol`, `eval/pensieve_rebuf_s_total`
- `eval/random_qoe_protocol`, `eval/random_rebuf_s_total`
