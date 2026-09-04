"""Train hypsometry GlacierFormer with LOYO cross-validation."""
from __future__ import annotations

import os
import random
import sys

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "02_models"))

from config import GLACIERFORMER_HYPSOMETRY_PARAMS, RESULT_DIR, SEQUENCES_HYPSOMETRY_QC_NPZ, TRAIN_YEAR_MAX, TRAIN_YEAR_MIN  # noqa: E402
from glacierformer_hypsometry import GlacierFormerHypsometry  # noqa: E402
from hypsometry_training_utils import make_random_validation_mask, save_outputs, train_one_fold_hypsometry  # noqa: E402


def main() -> None:
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    tag = "glacierformer_hypsometry_qc"
    result_dir = os.path.join(RESULT_DIR, "hypsometry_qc")
    os.makedirs(result_dir, exist_ok=True)
    log_path = os.path.join(result_dir, f"{tag}_loyo_train.log")

    with open(log_path, "w", encoding="utf-8", buffering=1) as log:
        def log_print(message: str) -> None:
            print(message)
            log.write(message + "\n")

        log_print("=== Hypsometry GlacierFormer LOYO ===")
        data = np.load(SEQUENCES_HYPSOMETRY_QC_NPZ, allow_pickle=True)
        x_dyn, x_sta, x_hyp = data["X_dyn"], data["X_sta"], data["X_hyp"]
        y, years, gids = data["y"], data["years"], data["glacier_ids"]

        mask = (years >= TRAIN_YEAR_MIN) & (years <= TRAIN_YEAR_MAX) & (~np.isnan(y))
        x_dyn, x_sta, x_hyp = x_dyn[mask], x_sta[mask], x_hyp[mask]
        y, years, gids = y[mask], years[mask], gids[mask]

        params = GLACIERFORMER_HYPSOMETRY_PARAMS
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log_print(f"Training samples: {len(y)}")
        log_print(f"Years: {len(np.unique(years))}")
        log_print(f"X_hyp: {x_hyp.shape}")
        log_print(f"Device: {device}")

        all_obs, all_pred, all_gids, all_years = [], [], [], []
        fold_years = sorted(np.unique(years).tolist())
        for fold_idx, year in enumerate(fold_years):
            test_mask = years == year
            train_mask = years != year
            fit_mask, val_mask = make_random_validation_mask(
                train_mask,
                params["val_fraction"],
                seed=42 + fold_idx,
            )
            pred, stopped_epoch = train_one_fold_hypsometry(
                GlacierFormerHypsometry,
                params,
                x_dyn,
                x_sta,
                x_hyp,
                y,
                fit_mask,
                val_mask,
                test_mask,
                device,
            )
            all_pred.extend(pred.tolist())
            all_obs.extend(y[test_mask].tolist())
            all_gids.extend(gids[test_mask].tolist())
            all_years.extend(years[test_mask].tolist())
            log_print(
                f"Fold {fold_idx + 1:02d}/{len(fold_years)} "
                f"year={int(year)} n_test={int(test_mask.sum())} stopped@{stopped_epoch}"
            )

        metrics = save_outputs(result_dir, tag, "LOYO", all_obs, all_pred, all_gids, all_years)
        log_print(
            f"LOYO R2={metrics['r2']:.4f} R={metrics['pearson_r']:.4f} "
            f"RMSE={metrics['rmse_mm']:.1f}mm Bias={metrics['bias_mm']:.1f}mm"
        )


if __name__ == "__main__":
    main()
