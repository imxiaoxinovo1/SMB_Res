"""Nested, group-safe hyperparameter selection for the corrected XGBoost model."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from config import PHYS_V2_RESULT_DIR, PHYS_V2_SEQUENCES_NPZ  # noqa: E402
from train_tree_v2_cv import (  # noqa: E402
    FEATURE_SETS,
    calendar_flatten,
    hypsometry_quantiles,
    metrics,
)


CANDIDATES = [
    {
        "candidate": "reference",
        "n_estimators": 450,
        "max_depth": 3,
        "learning_rate": 0.03,
        "min_child_weight": 5,
        "subsample": 0.80,
        "colsample_bytree": 0.70,
        "reg_alpha": 0.2,
        "reg_lambda": 8.0,
    },
    {
        "candidate": "shallow",
        "n_estimators": 600,
        "max_depth": 2,
        "learning_rate": 0.03,
        "min_child_weight": 3,
        "subsample": 0.85,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.1,
        "reg_lambda": 10.0,
    },
    {
        "candidate": "regularized",
        "n_estimators": 700,
        "max_depth": 3,
        "learning_rate": 0.02,
        "min_child_weight": 8,
        "subsample": 0.80,
        "colsample_bytree": 0.70,
        "reg_alpha": 0.3,
        "reg_lambda": 12.0,
    },
    {
        "candidate": "deeper",
        "n_estimators": 350,
        "max_depth": 4,
        "learning_rate": 0.03,
        "min_child_weight": 8,
        "subsample": 0.80,
        "colsample_bytree": 0.70,
        "reg_alpha": 0.2,
        "reg_lambda": 10.0,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv", choices=["logo", "loyo"], required=True)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-folds", type=int, default=None)
    return parser.parse_args()


def make_model(candidate: dict, seed: int) -> XGBRegressor:
    parameters = {key: value for key, value in candidate.items() if key != "candidate"}
    return XGBRegressor(
        **parameters,
        objective="reg:squarederror",
        random_state=seed,
        n_jobs=-1,
    )


def fold_arrays(x: np.ndarray, train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    medians = np.nanmedian(x[train], axis=0)
    return np.where(np.isnan(x[train]), medians, x[train]), np.where(np.isnan(x[test]), medians, x[test])


def choose_candidate(
    x: np.ndarray,
    y: np.ndarray,
    outer_train: np.ndarray,
    inner_groups: np.ndarray,
    n_splits: int,
    seed: int,
) -> tuple[dict, list[dict]]:
    train_indices = np.where(outer_train)[0]
    groups = inner_groups[outer_train]
    splitter = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    rows = []
    for candidate_index, candidate in enumerate(CANDIDATES):
        squared_errors = []
        for inner_index, (fit_local, val_local) in enumerate(
            splitter.split(train_indices, y[outer_train], groups)
        ):
            fit = train_indices[fit_local]
            val = train_indices[val_local]
            fit_mask = np.zeros(len(y), dtype=bool)
            val_mask = np.zeros(len(y), dtype=bool)
            fit_mask[fit] = True
            val_mask[val] = True
            x_fit, x_val = fold_arrays(x, fit_mask, val_mask)
            model = make_model(candidate, seed + 100 * candidate_index + inner_index)
            model.fit(x_fit, y[fit])
            squared_errors.extend((model.predict(x_val) - y[val]) ** 2)
        rows.append(
            {
                "candidate": candidate["candidate"],
                "inner_rmse_mm": float(np.sqrt(np.mean(squared_errors)) * 1000.0),
            }
        )
    best_name = min(rows, key=lambda item: item["inner_rmse_mm"])["candidate"]
    best = next(candidate for candidate in CANDIDATES if candidate["candidate"] == best_name)
    return best, rows


def main() -> None:
    args = parse_args()
    data = np.load(PHYS_V2_SEQUENCES_NPZ, allow_pickle=True)
    dynamic_names = [str(name) for name in data["dynamic_features"]]
    dynamic_indices = [dynamic_names.index(name) for name in FEATURE_SETS["compact"]]
    dynamic = calendar_flatten(data["X_dyn"][:, :, dynamic_indices], data["month_ids"])

    static_names = [str(name) for name in data["static_features"]]
    base_static = [
        "slope_deg", "aspect_sin", "aspect_cos", "zmin_m", "zmax_m", "zmean_m", "zmed_m",
        "log1p_area_km2", "log1p_lmax_m", "cenlat", "cenlon", "clim_annual_t2m",
        "clim_winter_tp", "clim_summer_t2m", "clim_summer_ssrd", "clim_t2m_amplitude",
    ]
    static_indices = [static_names.index(name) for name in base_static]
    hypsometry = hypsometry_quantiles(
        data["X_hyp"], data["hypsometry_band_centers_m"]
    )
    x = np.column_stack([dynamic, data["X_sta"][:, static_indices], hypsometry]).astype(np.float32)
    y = data["y_annual"].astype(np.float32)
    rgi_ids, years = data["rgi_ids"], data["years"]
    outer_groups = rgi_ids if args.cv == "logo" else years
    inner_groups = rgi_ids if args.cv == "logo" else years
    folds = np.unique(outer_groups)
    if args.max_folds is not None:
        folds = folds[: args.max_folds]

    prediction = np.full(len(y), np.nan, dtype=np.float32)
    selection_rows = []
    for fold_index, held_out in enumerate(folds):
        test = outer_groups == held_out
        train = ~test
        best, candidate_rows = choose_candidate(
            x, y, train, inner_groups, args.inner_splits, args.seed + fold_index
        )
        for row in candidate_rows:
            selection_rows.append(
                {"outer_fold": str(held_out), "selected": row["candidate"] == best["candidate"], **row}
            )
        x_train, x_test = fold_arrays(x, train, test)
        model = make_model(best, args.seed + fold_index)
        model.fit(x_train, y[train])
        prediction[test] = model.predict(x_test)
        print(
            f"[{fold_index + 1:02d}/{len(folds)}] held_out={held_out} "
            f"selected={best['candidate']} n={int(test.sum())}"
        )

    selected = np.isfinite(prediction)
    tag = "xgboost_v2_compact_nested"
    result_dir = os.path.join(PHYS_V2_RESULT_DIR, tag)
    os.makedirs(result_dir, exist_ok=True)
    output = pd.DataFrame(
        {
            "glacier_id": data["glacier_ids"][selected],
            "rgi_id": rgi_ids[selected],
            "year": years[selected],
            "obs_annual": y[selected],
            "pred_annual": prediction[selected],
        }
    )
    summary = pd.DataFrame(
        [{"model": tag, "cv": args.cv.upper(), **metrics(y[selected], prediction[selected])}]
    )
    output.to_csv(os.path.join(result_dir, f"{tag}_{args.cv}_predictions.csv"), index=False)
    summary.to_csv(os.path.join(result_dir, f"{tag}_{args.cv}_summary.csv"), index=False)
    pd.DataFrame(selection_rows).to_csv(
        os.path.join(result_dir, f"{tag}_{args.cv}_hyperparameter_selection.csv"), index=False
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
