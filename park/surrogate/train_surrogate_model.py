#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from surrogate.common import resolve_path
from surrogate.data import load_transition_dataset
from surrogate.train import save_surrogate_checkpoint, train_surrogate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the Park PGD step surrogate.")
    p.add_argument("--data", type=str, required=True, help="Input dataset .npz path")
    p.add_argument("--out", type=str, required=True, help="Output checkpoint path")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    arrays, meta = load_transition_dataset(str(resolve_path(str(args.data))))
    result = train_surrogate(
        arrays=arrays,
        num_servers=int(meta["num_servers"]),
        hidden=int(args.hidden),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        val_fraction=float(args.val_fraction),
        seed=int(args.seed),
        device=str(args.device),
    )
    out_path = resolve_path(str(args.out))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_surrogate_checkpoint(
        path=str(out_path),
        train_result=result,
        meta=meta,
        hidden=int(args.hidden),
    )
    print(f"saved surrogate checkpoint: {out_path}")
    print(f"train_loss={result.train_loss:.6f} val_loss={result.val_loss:.6f} best_epoch={result.best_epoch}")


if __name__ == "__main__":
    main()
