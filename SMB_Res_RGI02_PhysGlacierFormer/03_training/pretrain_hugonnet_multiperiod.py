"""Pretrain GlacierFormer with multi-period Hugonnet geodetic constraints."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "02_models"))

from config import (  # noqa: E402
    HUGONNET_MULTIPERIOD_SEQUENCES_NPZ,
    HUGONNET_PRETRAIN_PARAMS,
    HUGONNET_WEAK_RESULT_DIR,
)
from glacierformer_base import GlacierFormerBase  # noqa: E402
from training_utils import pretrain_on_multiperiod_constraints, set_seed  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weak-alpha", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    set_seed(42)

    os.makedirs(HUGONNET_WEAK_RESULT_DIR, exist_ok=True)
    out_path = os.path.join(
        HUGONNET_WEAK_RESULT_DIR,
        "glacierformer_hugonnet_multiperiod_pretrained.pt",
    )
    log_path = os.path.join(
        HUGONNET_WEAK_RESULT_DIR,
        "glacierformer_hugonnet_multiperiod_pretrain.log",
    )

    with open(log_path, "w", encoding="utf-8", buffering=1) as log:
        def log_print(message: str) -> None:
            print(message)
            log.write(message + "\n")

        log_print("=== GlacierFormer Hugonnet Multi-period Pretrain ===")
        log_print(f"Input: {HUGONNET_MULTIPERIOD_SEQUENCES_NPZ}")
        log_print(f"Weak alpha: {args.weak_alpha}")
        log_print(f"Epochs: {args.epochs}")

        data = np.load(HUGONNET_MULTIPERIOD_SEQUENCES_NPZ, allow_pickle=True)
        x_dyn = data["X_dyn"]
        x_sta = data["X_sta"]
        constraint_index = data["constraint_index"]
        target_by_constraint = data["target_by_constraint"]
        weight_by_constraint = data["weight_by_constraint"]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        log_print(f"Samples: {len(x_dyn):,}")
        log_print(f"Constraints: {len(target_by_constraint):,}")
        log_print(f"Device: {device}")

        state_dict = pretrain_on_multiperiod_constraints(
            GlacierFormerBase,
            HUGONNET_PRETRAIN_PARAMS,
            x_dyn,
            x_sta,
            constraint_index=constraint_index,
            target_by_constraint=target_by_constraint,
            weight_by_constraint=weight_by_constraint,
            weak_alpha=args.weak_alpha,
            epochs=args.epochs,
            device=device,
        )
        torch.save(state_dict, out_path)
        log_print(f"Saved pretrained weights -> {out_path}")


if __name__ == "__main__":
    main()
