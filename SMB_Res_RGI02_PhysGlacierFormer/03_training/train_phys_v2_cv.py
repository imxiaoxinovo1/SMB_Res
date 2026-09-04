"""Run leakage-safe LOGO or LOYO validation for PhysGlacierFormer v2."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "02_models"))

from config import (  # noqa: E402
    PHYS_GLACIERFORMER_V2_PARAMS,
    PHYS_V2_DYNAMIC_FEATURES,
    PHYS_V2_RESULT_DIR,
    PHYS_V2_SEQUENCES_NPZ,
)
from phys_glacierformer_v2 import PhysGlacierFormerV2  # noqa: E402
from phys_v2_training_utils import make_group_validation_masks, train_one_fold  # noqa: E402


FEATURE_SETS = {
    "all": PHYS_V2_DYNAMIC_FEATURES,
    "compact": [
        "t2m", "sd", "asn", "tp", "sf", "ssrd", "str", "slhf", "sshf",
        "t2m_anomaly", "tp_anomaly", "sf_anomaly", "ssrd_anomaly", "asn_anomaly",
    ],
    "minimal": ["t2m", "tp", "t2m_anomaly", "tp_anomaly"],
}

BASE_STATIC_FEATURES = [
    "slope_deg", "aspect_sin", "aspect_cos", "zmin_m", "zmax_m", "zmean_m", "zmed_m",
    "log1p_area_km2", "log1p_lmax_m", "cenlat", "cenlon", "clim_annual_t2m",
    "clim_winter_tp", "clim_summer_t2m", "clim_summer_ssrd", "clim_t2m_amplitude",
]


def regression_metrics(obs: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(obs) & np.isfinite(pred)
    obs, pred = obs[mask], pred[mask]
    r = float(pearsonr(obs, pred).statistic) if len(obs) >= 2 else np.nan
    return {
        "n": int(len(obs)),
        "r2": float(r2_score(obs, pred)) if len(obs) >= 2 else np.nan,
        "pearson_r": r,
        "rmse_mm": float(np.sqrt(mean_squared_error(obs, pred)) * 1000.0),
        "mae_mm": float(mean_absolute_error(obs, pred) * 1000.0),
        "bias_mm": float(np.mean(pred - obs) * 1000.0),
    }


def macro_rmse(frame: pd.DataFrame, group: str) -> float:
    values = frame.groupby(group).apply(
        lambda part: float(np.sqrt(np.mean((part["pred_annual"] - part["obs_annual"]) ** 2))),
        include_groups=False,
    )
    return float(values.mean() * 1000.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv", choices=["logo", "loyo", "loyo_buffered", "loso"], required=True)
    parser.add_argument("--feature-set", choices=sorted(FEATURE_SETS), default="compact")
    parser.add_argument("--static-set", choices=["base", "downscaled"], default="base")
    parser.add_argument("--seasonal-weight", type=float, default=PHYS_GLACIERFORMER_V2_PARAMS["seasonal_loss_weight"])
    parser.add_argument("--loss", choices=["mse", "huber"], default="huber")
    parser.add_argument("--uncertainty-weights", action="store_true")
    parser.add_argument("--balance-glaciers", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-folds", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    data = np.load(PHYS_V2_SEQUENCES_NPZ, allow_pickle=True)
    stored_features = [str(name) for name in data["dynamic_features"]]
    selected = FEATURE_SETS[args.feature_set]
    feature_idx = [stored_features.index(name) for name in selected]
    arrays = {key: data[key] for key in [
        "X_dyn", "X_sta", "X_hyp", "month_ids", "y_annual", "y_winter", "y_summer",
        "annual_unc", "winter_unc", "summer_unc", "glacier_ids", "rgi_ids", "o2regions", "years",
    ]}
    arrays["X_dyn"] = arrays["X_dyn"][:, :, feature_idx]
    if args.static_set == "base":
        stored_static = [str(name) for name in data["static_features"]]
        static_idx = [stored_static.index(name) for name in BASE_STATIC_FEATURES]
        arrays["X_sta"] = arrays["X_sta"][:, static_idx]

    params = dict(PHYS_GLACIERFORMER_V2_PARAMS)
    params["n_dynamic_features"] = len(selected)
    params["n_static_features"] = arrays["X_sta"].shape[1]
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if args.cv == "logo":
        outer_groups = arrays["rgi_ids"]
        inner_groups = arrays["rgi_ids"]
        cv_label = "LOGO"
    elif args.cv == "loso":
        outer_groups = arrays["o2regions"]
        inner_groups = arrays["rgi_ids"]
        cv_label = "LOSO"
    else:
        outer_groups = arrays["years"]
        inner_groups = arrays["years"]
        cv_label = "LOYO_BUFFERED" if args.cv == "loyo_buffered" else "LOYO"
    folds = np.unique(outer_groups)
    if args.max_folds is not None:
        folds = folds[: args.max_folds]

    tag_parts = ["phys_v2", args.feature_set, args.loss, f"sw{args.seasonal_weight:g}"]
    if args.static_set != "base":
        tag_parts.append(args.static_set)
    if args.uncertainty_weights:
        tag_parts.append("uw")
    if args.balance_glaciers:
        tag_parts.append("gb")
    tag = "_".join(tag_parts)
    result_dir = os.path.join(PHYS_V2_RESULT_DIR, tag)
    os.makedirs(result_dir, exist_ok=True)
    log_path = os.path.join(result_dir, f"{tag}_{args.cv}_train.log")

    predictions: list[pd.DataFrame] = []
    with open(log_path, "w", encoding="utf-8", buffering=1) as log:
        def report(message: str) -> None:
            print(message)
            log.write(message + "\n")

        report(f"=== {tag} {cv_label} ===")
        report(f"samples={len(arrays['y_annual'])} folds={len(folds)} features={len(selected)} device={device}")
        report(f"selected_features={selected}")
        for fold_index, held_out in enumerate(folds):
            test_mask = outer_groups == held_out
            if args.cv == "loyo_buffered":
                outer_train_mask = np.abs(arrays["years"] - int(held_out)) > 1
            else:
                outer_train_mask = ~test_mask
            fit_mask, val_mask = make_group_validation_masks(
                outer_train_mask, inner_groups, params["val_fraction"], args.seed + fold_index
            )
            pred, stopped_epoch, _ = train_one_fold(
                PhysGlacierFormerV2,
                params,
                arrays,
                fit_mask,
                val_mask,
                test_mask,
                device,
                args.seed + fold_index,
                args.seasonal_weight,
                args.loss,
                args.uncertainty_weights,
                args.balance_glaciers,
            )
            part = pd.DataFrame(
                {
                    "glacier_id": arrays["glacier_ids"][test_mask],
                    "rgi_id": arrays["rgi_ids"][test_mask],
                    "year": arrays["years"][test_mask],
                    "obs_annual": arrays["y_annual"][test_mask],
                    "pred_annual": pred["annual"],
                    "obs_winter": arrays["y_winter"][test_mask],
                    "pred_winter": pred["winter"],
                    "obs_summer": arrays["y_summer"][test_mask],
                    "pred_summer": pred["summer"],
                    "fold": str(held_out),
                }
            )
            predictions.append(part)
            fold_metrics = regression_metrics(part["obs_annual"].to_numpy(), part["pred_annual"].to_numpy())
            report(
                f"[{fold_index + 1:02d}/{len(folds)}] held_out={held_out} n={len(part)} "
                f"rmse={fold_metrics['rmse_mm']:.1f} stopped@{stopped_epoch}"
            )

    output = pd.concat(predictions, ignore_index=True)
    prediction_path = os.path.join(result_dir, f"{tag}_{args.cv}_predictions.csv")
    output.to_csv(prediction_path, index=False)

    rows = []
    for target in ["annual", "winter", "summer"]:
        metrics = regression_metrics(
            output[f"obs_{target}"].to_numpy(), output[f"pred_{target}"].to_numpy()
        )
        rows.append({"model": tag, "cv": cv_label, "target": target, **metrics})
    rows[0]["macro_glacier_rmse_mm"] = macro_rmse(output, "rgi_id")
    rows[0]["macro_year_rmse_mm"] = macro_rmse(output, "year")
    summary = pd.DataFrame(rows)
    summary_path = os.path.join(result_dir, f"{tag}_{args.cv}_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"Saved predictions -> {prediction_path}")
    print(f"Saved summary -> {summary_path}")


if __name__ == "__main__":
    main()
