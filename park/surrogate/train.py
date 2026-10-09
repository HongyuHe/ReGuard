from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from surrogate.data import TransitionDataset
from surrogate.model import LearnedParkStepSurrogate


@dataclass(frozen=True)
class TrainResult:
    model: LearnedParkStepSurrogate
    reward_mean: float
    reward_std: float
    train_loss: float
    val_loss: float
    best_epoch: int


def _split_indices(n: int, *, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    perm = np.asarray(rng.permutation(int(n)), dtype=np.int64)
    val_count = int(max(1, round(float(val_fraction) * float(n))))
    val_count = int(min(max(1, val_count), max(1, n - 1)))
    val_idx = perm[:val_count]
    train_idx = perm[val_count:]
    if train_idx.size == 0:
        train_idx = val_idx[:1]
    return train_idx, val_idx


def _make_loader(dataset: TransitionDataset, indices: np.ndarray, *, batch_size: int, shuffle: bool) -> torch.utils.data.DataLoader:
    subset = torch.utils.data.Subset(dataset, [int(x) for x in np.asarray(indices).reshape(-1)])
    return torch.utils.data.DataLoader(
        subset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        drop_last=False,
    )


def _loss_batch(
    *,
    model: LearnedParkStepSurrogate,
    batch: dict[str, torch.Tensor],
    reward_mean: float,
    reward_std: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    next_summary_pred, reward_norm_pred = model(
        batch["summary"],
        batch["action_probs"],
        batch["control01"],
    )
    reward_target_norm = (batch["reward"] - float(reward_mean)) / float(max(reward_std, 1e-6))
    summary_loss = F.mse_loss(next_summary_pred, batch["next_summary"])
    reward_loss = F.mse_loss(reward_norm_pred, reward_target_norm)
    loss = summary_loss + reward_loss
    stats = {
        "summary_loss": float(summary_loss.detach().item()),
        "reward_loss": float(reward_loss.detach().item()),
        "loss": float(loss.detach().item()),
    }
    return loss, stats


def train_surrogate(
    *,
    arrays: dict[str, np.ndarray],
    num_servers: int,
    hidden: int,
    epochs: int,
    batch_size: int,
    lr: float,
    val_fraction: float,
    seed: int,
    device: str,
) -> TrainResult:
    dataset = TransitionDataset(arrays)
    reward_values = np.asarray(arrays["reward"], dtype=np.float32)
    reward_mean = float(np.mean(reward_values))
    reward_std = float(max(np.std(reward_values), 1e-6))
    train_idx, val_idx = _split_indices(len(dataset), val_fraction=float(val_fraction), seed=int(seed))
    train_loader = _make_loader(dataset, train_idx, batch_size=int(batch_size), shuffle=True)
    val_loader = _make_loader(dataset, val_idx, batch_size=int(batch_size), shuffle=False)
    model = LearnedParkStepSurrogate(
        num_servers=int(num_servers),
        hidden=int(hidden),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    best_state = None
    best_val = None
    best_epoch = 0
    last_train_loss = float("nan")
    for epoch_idx in range(int(epochs)):
        model.train(True)
        train_losses = []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            loss, _stats = _loss_batch(
                model=model,
                batch=batch,
                reward_mean=float(reward_mean),
                reward_std=float(reward_std),
            )
            loss.backward()
            opt.step()
            train_losses.append(float(loss.detach().item()))
        last_train_loss = float(np.mean(train_losses)) if len(train_losses) > 0 else float("nan")
        model.train(False)
        with torch.no_grad():
            val_losses = []
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                loss, _stats = _loss_batch(
                    model=model,
                    batch=batch,
                    reward_mean=float(reward_mean),
                    reward_std=float(reward_std),
                )
                val_losses.append(float(loss.detach().item()))
        val_loss = float(np.mean(val_losses)) if len(val_losses) > 0 else float("nan")
        if best_val is None or float(val_loss) < float(best_val):
            best_val = float(val_loss)
            best_epoch = int(epoch_idx)
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return TrainResult(
        model=model,
        reward_mean=float(reward_mean),
        reward_std=float(reward_std),
        train_loss=float(last_train_loss),
        val_loss=float(best_val if best_val is not None else float("nan")),
        best_epoch=int(best_epoch),
    )


def save_surrogate_checkpoint(
    *,
    path: str,
    train_result: TrainResult,
    meta: dict[str, Any],
    hidden: int,
) -> None:
    payload = {
        "model_state_dict": train_result.model.state_dict(),
        "surrogate_config": {
            "num_servers": int(meta["num_servers"]),
            "hidden": int(hidden),
            "dt_min": float(meta["dt_min"]),
            "dt_max": float(meta["dt_max"]),
            "size_min": float(meta["size_min"]),
            "size_max": float(meta["size_max"]),
            "reward_mean": float(train_result.reward_mean),
            "reward_std": float(train_result.reward_std),
            "obs_high": float(meta["obs_high"]),
            "reward_time_scale": float(meta["reward_time_scale"]),
            "service_rates": [float(x) for x in list(meta["service_rates"])],
            "oracle_k": int(meta["oracle_k"]),
        },
        "train_metrics": {
            "train_loss": float(train_result.train_loss),
            "val_loss": float(train_result.val_loss),
            "best_epoch": int(train_result.best_epoch),
        },
    }
    torch.save(payload, path)
