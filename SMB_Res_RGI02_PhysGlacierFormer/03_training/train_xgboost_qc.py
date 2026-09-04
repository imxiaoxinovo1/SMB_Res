"""Train XGBoost on the QC tabular dataset with LOGO or LOYO validation."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from config import RESULT_DIR, TRAIN_YEAR_MAX, TRAIN_YEAR_MIN, XGBOOST_PARAMS  # noqa: E402


TABULAR_QC_CSV = os.path.join(PROJECT_DIR, "data", "qc", "tabular_dataset_qc.csv")
RESULT_DIR_XGB_QC = os.path.join(RESULT_DIR, "xgboost_qc")


def evaluate(obs: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    r, _ = pearsonr(obs, pred)
    return {
        "r2": float(r2_score(obs, pred)),
        "pearson_r": float(r),
        "rmse_mm": float(np.sqrt(mean_squared_error(obs, pred)) * 1000.0),
        "bias_mm": float(np.mean(pred - obs) * 1000.0),
    }


def make_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    meta_cols = {"glacier_id", "year", "annual_balance_m"}
    feature_cols = [col for col in frame.columns if col not in meta_cols]
    x = frame[feature_cols].copy()
    y = frame["annual_balance_m"].copy()
    return x, y, feature_cols


def fit_predict_fold(
    x: pd.DataFrame,
    y: pd.Series,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    params: dict,
) -> np.ndarray:
    x_train = x.loc[train_mask].copy()
    x_test = x.loc[test_mask].copy()
    y_train = y.loc[train_mask].copy()

    medians = x_train.median(numeric_only=True)
    x_train = x_train.fillna(medians)
    x_test = x_test.fillna(medians)

    model = XGBRegressor(**params)
    model.fit(x_train, y_train)
    return model.predict(x_test)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv", choices=["logo", "loyo"], required=True)
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="Number of boosting trees. Use 500 for the full baseline if time allows.",
    )
    args = parser.parse_args()

    os.makedirs(RESULT_DIR_XGB_QC, exist_ok=True)
    tag = "xgboost_qc"
    cv_name = args.cv.upper()
    log_path = os.path.join(RESULT_DIR_XGB_QC, f"{tag}_{args.cv}_train.log")

    with open(log_path, "w", encoding="utf-8", buffering=1) as log:
        def log_print(message: str) -> None:
            print(message)
            log.write(message + "\n")

        log_print(f"=== XGBoost QC {cv_name} ===")
        log_print(f"n_estimators: {args.n_estimators}")
        data = pd.read_csv(TABULAR_QC_CSV)
        data = data[
            (data["year"] >= TRAIN_YEAR_MIN)
            & (data["year"] <= TRAIN_YEAR_MAX)
            & data["annual_balance_m"].notna()
        ].copy()
        data = data.reset_index(drop=True)

        x, y, feature_cols = make_features(data)
        log_print(f"Training/evaluation rows: {len(data)}")
        log_print(f"Feature columns: {len(feature_cols)}")
        log_print(f"Glaciers: {data['glacier_id'].nunique()}")
        log_print(f"Years: {data['year'].nunique()}")

        all_obs: list[float] = []
        all_pred: list[float] = []
        all_gids: list = []
        all_years: list = []

        fold_values = (
            sorted(data["glacier_id"].unique().tolist())
            if args.cv == "logo"
            else sorted(data["year"].unique().tolist())
        )
        fold_col = "glacier_id" if args.cv == "logo" else "year"

        for i, value in enumerate(fold_values):
            test_mask = (data[fold_col].values == value)
            train_mask = ~test_mask
            params = dict(XGBOOST_PARAMS)
            params["n_estimators"] = args.n_estimators
            pred = fit_predict_fold(x, y, train_mask, test_mask, params)

            obs = y.loc[test_mask].values.astype(float)
            all_obs.extend(obs.tolist())
            all_pred.extend(pred.tolist())
            all_gids.extend(data.loc[test_mask, "glacier_id"].tolist())
            all_years.extend(data.loc[test_mask, "year"].tolist())
            log_print(
                f"Fold {i + 1:02d}/{len(fold_values)} "
                f"{fold_col}={value} n_test={int(test_mask.sum())}"
            )

        metrics = evaluate(np.array(all_obs), np.array(all_pred))
        pd.DataFrame(
            {
                "glacier_id": all_gids,
                "year": all_years,
                "obs": all_obs,
                "pred": all_pred,
            }
        ).to_csv(os.path.join(RESULT_DIR_XGB_QC, f"{tag}_{args.cv}_predictions.csv"), index=False)

        pd.DataFrame([{"model": tag, "cv": cv_name, **metrics}]).to_csv(
            os.path.join(RESULT_DIR_XGB_QC, f"{tag}_{args.cv}_summary.csv"),
            index=False,
        )
        log_print(
            f"{cv_name} R2={metrics['r2']:.4f} R={metrics['pearson_r']:.4f} "
            f"RMSE={metrics['rmse_mm']:.1f}mm Bias={metrics['bias_mm']:.1f}mm"
        )


if __name__ == "__main__":
    main()
