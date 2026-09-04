"""Nested, leakage-safe amplitude calibration of fixed XGBoost OOF predictions."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from config import PHYS_V2_RESULT_DIR, PHYS_V2_SEQUENCES_NPZ  # noqa: E402
from train_tree_v2_cv import (  # noqa: E402
    FEATURE_SETS,
    calendar_flatten,
    hypsometry_quantiles,
    make_model,
    metrics,
)


BASE_TAG = "xgboost_v2_compact_monthly_all_hyp_p-regularized"
STATIC_FEATURES = [
    "slope_deg", "aspect_sin", "aspect_cos", "zmin_m", "zmax_m",
    "zmean_m", "zmed_m", "log1p_area_km2", "log1p_lmax_m", "cenlat", "cenlon",
    "clim_annual_t2m", "clim_winter_tp", "clim_summer_t2m",
    "clim_summer_ssrd", "clim_t2m_amplitude",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv", choices=["logo", "loyo"], required=True)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--slope-min", type=float, default=0.75)
    parser.add_argument("--slope-max", type=float, default=1.50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_features(data) -> np.ndarray:
    dynamic_names = [str(value) for value in data["dynamic_features"]]
    selected = FEATURE_SETS["compact"]
    dynamic_indices = [dynamic_names.index(name) for name in selected]
    dynamic = calendar_flatten(data["X_dyn"][:, :, dynamic_indices], data["month_ids"])
    static_names = [str(value) for value in data["static_features"]]
    static_indices = [static_names.index(name) for name in STATIC_FEATURES]
    hypsometry = hypsometry_quantiles(
        data["X_hyp"], data["hypsometry_band_centers_m"]
    )
    return np.column_stack([dynamic, data["X_sta"][:, static_indices], hypsometry]).astype(
        np.float32
    )


def fit_amplitude_calibrator(
    observed: np.ndarray,
    predicted: np.ndarray,
    slope_min: float,
    slope_max: float,
) -> tuple[float, float, float]:
    centered_prediction = predicted - np.mean(predicted)
    denominator = float(np.sum(centered_prediction**2))
    if denominator <= 1e-12:
        raw_slope = 1.0
    else:
        raw_slope = float(
            np.sum(centered_prediction * (observed - np.mean(observed))) / denominator
        )
    slope = float(np.clip(raw_slope, slope_min, slope_max))
    intercept = float(np.mean(observed) - slope * np.mean(predicted))
    return intercept, slope, raw_slope


def main() -> None:
    args = parse_args()
    if args.slope_min <= 0 or args.slope_max < args.slope_min:
        raise ValueError("Invalid slope bounds.")
    data = np.load(PHYS_V2_SEQUENCES_NPZ, allow_pickle=True)
    features = build_features(data)
    target = data["y_annual"].astype(np.float32)
    rgi_ids = data["rgi_ids"]
    years = data["years"].astype(int)
    outer_groups = rgi_ids if args.cv == "logo" else years

    base_path = os.path.join(
        PHYS_V2_RESULT_DIR,
        BASE_TAG,
        f"{BASE_TAG}_{args.cv}_predictions.csv",
    )
    base = pd.read_csv(base_path)
    metadata = pd.DataFrame(
        {
            "row_index": np.arange(len(target)),
            "glacier_id": data["glacier_ids"],
            "rgi_id": rgi_ids,
            "year": years,
        }
    )
    base = metadata.merge(
        base[["glacier_id", "rgi_id", "year", "pred_annual"]],
        on=["glacier_id", "rgi_id", "year"],
        how="left",
        validate="one_to_one",
    ).sort_values("row_index")
    base_prediction = base["pred_annual"].to_numpy(dtype=np.float32)
    if not np.isfinite(base_prediction).all():
        raise RuntimeError("Base OOF predictions do not align with the corrected dataset.")

    calibrated = np.full(len(target), np.nan, dtype=np.float32)
    parameter_rows = []
    for outer_index, held_out in enumerate(np.unique(outer_groups)):
        test = outer_groups == held_out
        train = ~test
        inner_groups = outer_groups[train]
        n_splits = min(args.inner_folds, len(np.unique(inner_groups)))
        inner_oof = np.full(int(train.sum()), np.nan, dtype=np.float32)
        outer_train_indices = np.flatnonzero(train)
        splitter = GroupKFold(n_splits=n_splits)
        for inner_index, (fit_local, val_local) in enumerate(
            splitter.split(features[train], target[train], groups=inner_groups)
        ):
            fit_indices = outer_train_indices[fit_local]
            val_indices = outer_train_indices[val_local]
            medians = np.nanmedian(features[fit_indices], axis=0)
            medians = np.where(np.isfinite(medians), medians, 0.0)
            x_fit = np.where(np.isnan(features[fit_indices]), medians, features[fit_indices])
            x_val = np.where(np.isnan(features[val_indices]), medians, features[val_indices])
            model = make_model(
                "xgboost",
                args.seed + outer_index * 100 + inner_index,
                xgb_profile="regularized",
            )
            model.fit(x_fit, target[fit_indices])
            inner_oof[val_local] = model.predict(x_val)
        if not np.isfinite(inner_oof).all():
            raise RuntimeError(f"Incomplete inner OOF predictions for held-out group {held_out}.")

        intercept, slope, raw_slope = fit_amplitude_calibrator(
            target[train], inner_oof, args.slope_min, args.slope_max
        )
        calibrated[test] = intercept + slope * base_prediction[test]
        parameter_rows.append(
            {
                "held_out": held_out,
                "n_train": int(train.sum()),
                "n_test": int(test.sum()),
                "intercept_mwe_yr": intercept,
                "slope": slope,
                "raw_slope": raw_slope,
                "slope_was_clipped": bool(slope != raw_slope),
            }
        )
        print(
            f"[{outer_index + 1:02d}/{len(np.unique(outer_groups))}] held_out={held_out} "
            f"slope={slope:.3f} intercept={intercept:+.3f}"
        )

    tag = f"{BASE_TAG}_nested-amplitude"
    result_dir = os.path.join(PHYS_V2_RESULT_DIR, tag)
    os.makedirs(result_dir, exist_ok=True)
    prediction_frame = pd.DataFrame(
        {
            "glacier_id": data["glacier_ids"],
            "rgi_id": rgi_ids,
            "year": years,
            "obs_annual": target,
            "pred_annual": calibrated,
            "pred_base": base_prediction,
        }
    )
    prediction_frame.to_csv(
        os.path.join(result_dir, f"{tag}_{args.cv}_predictions.csv"), index=False
    )
    result = {
        "model": tag,
        "cv": args.cv.upper(),
        **metrics(target, calibrated),
        "median_calibration_slope": float(np.median([row["slope"] for row in parameter_rows])),
        "n_clipped_slopes": int(sum(row["slope_was_clipped"] for row in parameter_rows)),
    }
    pd.DataFrame([result]).to_csv(
        os.path.join(result_dir, f"{tag}_{args.cv}_summary.csv"), index=False
    )
    pd.DataFrame(parameter_rows).to_csv(
        os.path.join(result_dir, f"{tag}_{args.cv}_calibration_parameters.csv"), index=False
    )
    print(pd.DataFrame([result]).to_string(index=False))


if __name__ == "__main__":
    main()
