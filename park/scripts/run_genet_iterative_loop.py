#!/usr/bin/env python3
"""
#* Example from the repo root:

time python scripts/run_genet_iterative_loop.py \
  --model models/parklb_v1_350k.pt \
  --rounds 20 \
  --reference-window 5 \
  --out-dir genet/runs/parklb_v1_iter20 \
  --overwrite
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from genet.iterative_train import main


if __name__ == "__main__":
    main()
