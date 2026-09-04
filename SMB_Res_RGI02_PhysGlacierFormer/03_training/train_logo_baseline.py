"""Train baseline GlacierFormer with LOGO cross-validation."""
from __future__ import annotations

import os
import sys
import argparse

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "02_models"))

from config import (  # noqa: E402
    GLACIERFORMER_BASE_PARAMS,
    RESULT_DIR,
    SEQUENCES_NPZ,
    SEQUENCES_QC_NPZ,
    TRAIN_YEAR_MAX,
    TRAIN_YEAR_MIN,
)
from glacierformer_base import GlacierFormerBase  # noqa: E402
from training_utils import make_random_validation_mask, save_validation_outputs, set_seed, train_one_fold  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["raw", "qc"],
        default="raw",
        help="Use raw baseline sequences or quality-controlled sequences.",
    )
    parser.add_argument("--init-weights", default=None, help="Optional pretrained state_dict path.")
    parser.add_argument(
        "--init-scope",
        choices=["all", "dynamic"],
        default="all",
        help="Load all pretrained weights or only dynamic/month encoder weights.",
    )
    parser.add_argument("--tag-suffix", default="", help="Optional suffix appended to output tag.")
    args = parser.parse_args()

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    set_seed(42)

    tag = "glacierformer_base" if args.dataset == "raw" else "glacierformer_base_qc"
    tag = f"{tag}{args.tag_suffix}"
    sequence_path = SEQUENCES_NPZ if args.dataset == "raw" else SEQUENCES_QC_NPZ
    result_dir = os.path.join(RESULT_DIR, "baseline" if args.dataset == "raw" else "baseline_qc")
    os.makedirs(result_dir, exist_ok=True)
    log_path = os.path.join(result_dir, f"{tag}_logo_train.log")

    with open(log_path, "w", encoding="utf-8", buffering=1) as log:
        def log_print(message: str) -> None:
            print(message)
            log.write(message + "\n")

        log_print("=== Baseline GlacierFormer LOGO ===")
        log_print(f"Dataset: {args.dataset}")
        log_print(f"Input sequence file: {sequence_path}")

        data = np.load(sequence_path, allow_pickle=True)
        x_dyn = data["X_dyn"]
        x_sta = data["X_sta"]
        y = data["y"]
        years = data["years"]
        glacier_ids = data["glacier_ids"]

        train_mask = (
            (years >= TRAIN_YEAR_MIN)
            & (years <= TRAIN_YEAR_MAX)
            & (~np.isnan(y))
        )
        x_dyn = x_dyn[train_mask]
        x_sta = x_sta[train_mask]
        y = y[train_mask]
        years = years[train_mask]
        glacier_ids = glacier_ids[train_mask]

        params = GLACIERFORMER_BASE_PARAMS
        device = "cuda" if torch.cuda.is_available() else "cpu"
        init_state = None
        if args.init_weights:
            log_print(f"Init weights: {args.init_weights}")
            init_state = torch.load(args.init_weights, map_location="cpu", weights_only=True)
            if args.init_scope == "dynamic":
                init_state = {
                    key: value
                    for key, value in init_state.items()
                    if key.startswith(("dynamic_embedding", "month_embedding", "encoder"))
                }
                log_print("Init scope: dynamic")
            else:
                log_print("Init scope: all")
        log_print(f"Training samples: {len(y)}")
        log_print(f"Glaciers: {len(np.unique(glacier_ids))}")
        log_print(f"Device: {device}")

        all_obs: list[float] = []
        all_pred: list[float] = []
        all_gids: list = []
        all_years: list = []

        fold_glaciers = sorted(np.unique(glacier_ids).tolist())
        for fold_idx, gid in enumerate(fold_glaciers):
            test_mask = glacier_ids == gid
            train_fold_mask = glacier_ids != gid
            if test_mask.sum() == 0:
                continue

            fit_mask, val_mask = make_random_validation_mask(
                train_fold_mask,
                params["val_fraction"],
                seed=42 + fold_idx,
            )
            pred, stopped_epoch = train_one_fold(
                GlacierFormerBase,
                params,
                x_dyn,
                x_sta,
                y,
                fit_mask,
                val_mask,
                test_mask,
                device,
                init_state_dict=init_state,
            )

            all_pred.extend(pred.tolist())
            all_obs.extend(y[test_mask].tolist())
            all_gids.extend(glacier_ids[test_mask].tolist())
            all_years.extend(years[test_mask].tolist())

            log_print(
                f"Fold {fold_idx + 1:02d}/{len(fold_glaciers)} "
                f"gid={gid} n_test={int(test_mask.sum())} stopped@{stopped_epoch}"
            )

        metrics = save_validation_outputs(
            result_dir,
            tag,
            "LOGO",
            all_obs,
            all_pred,
            all_gids,
            all_years,
        )
        log_print(
            "LOGO "
            f"R2={metrics['r2']:.4f} "
            f"R={metrics['pearson_r']:.4f} "
            f"RMSE={metrics['rmse_mm']:.1f}mm "
            f"Bias={metrics['bias_mm']:.1f}mm"
        )


if __name__ == "__main__":
    main()
