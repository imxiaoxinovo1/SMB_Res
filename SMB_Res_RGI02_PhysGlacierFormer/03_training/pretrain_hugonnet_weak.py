"""Pretrain GlacierFormer on Hugonnet weak labels."""
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
    HUGONNET_PRETRAIN_PARAMS,
    HUGONNET_WEAK_RESULT_DIR,
    HUGONNET_WEAK_SEQUENCES_NPZ,
    RESULT_DIR,
)
from glacierformer_base import GlacierFormerBase  # noqa: E402
from training_utils import pretrain_on_weak_labels, set_seed  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weak-alpha", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=80)
    args = parser.parse_args()

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    set_seed(42)

    result_dir = HUGONNET_WEAK_RESULT_DIR
    os.makedirs(result_dir, exist_ok=True)
    out_path = os.path.join(result_dir, "glacierformer_hugonnet_pretrained.pt")
    log_path = os.path.join(result_dir, "glacierformer_hugonnet_pretrain.log")

    with open(log_path, "w", encoding="utf-8", buffering=1) as log:
        def log_print(message: str) -> None:
            print(message)
            log.write(message + "\n")

        log_print("=== GlacierFormer Hugonnet Pretrain ===")
        log_print(f"Weak sequence file: {HUGONNET_WEAK_SEQUENCES_NPZ}")
        log_print(f"Weak alpha: {args.weak_alpha}")
        log_print(f"Epochs: {args.epochs}")

        data = np.load(HUGONNET_WEAK_SEQUENCES_NPZ, allow_pickle=True)
        x_dyn = data["X_dyn"]
        x_sta = data["X_sta"]
        weak_target_by_glacier = data["weak_target_by_glacier"]
        weak_weight_by_glacier = data["weak_weight_by_glacier"]
        weak_glacier_index = data["weak_glacier_index"]

        params = HUGONNET_PRETRAIN_PARAMS
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log_print(f"Weak samples: {len(weak_glacier_index)}")
        log_print(f"Weak glaciers: {len(weak_target_by_glacier)}")
        log_print(f"Device: {device}")

        state_dict = pretrain_on_weak_labels(
            GlacierFormerBase,
            params,
            x_dyn,
            x_sta,
            weak_glacier_index=weak_glacier_index,
            weak_target_by_glacier=weak_target_by_glacier,
            weak_weight_by_glacier=weak_weight_by_glacier,
            weak_alpha=args.weak_alpha,
            epochs=args.epochs,
            device=device,
        )

        torch.save(state_dict, out_path)
        log_print(f"Saved pretrained weights -> {out_path}")


if __name__ == "__main__":
    main()
