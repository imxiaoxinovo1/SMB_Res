"""Post-hoc geodetic calibration using Hugonnet multi-period constraints."""
from __future__ import annotations

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from config import (  # noqa: E402
    HUGONNET_CALIBRATION_OFFSET_CLIP,
    HUGONNET_CALIBRATION_OFFSET_SHRINK,
    HUGONNET_CALIBRATION_QC_CSV,
    HUGONNET_CALIBRATION_REGIONAL_CSV,
    HUGONNET_MULTIPERIOD_LABELS_CSV,
    HYPSOMETRY_RECON_CALIBRATED_CSV,
    HYPSOMETRY_RECON_CONSERVATIVE_CSV,
    HYPSOMETRY_RECON_RAW_CSV,
    RECONSTRUCTION_DIR,
)


def safe_to_csv(frame: pd.DataFrame, path: str, **kwargs) -> str:
    try:
        frame.to_csv(path, **kwargs)
        return path
    except PermissionError:
        root, ext = os.path.splitext(path)
        fallback = f"{root}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
        frame.to_csv(fallback, **kwargs)
        print(f"WARNING: could not overwrite locked file. Saved fallback -> {fallback}")
        return fallback


def main() -> None:
    print("=== Reconstruction Step 02: Hugonnet Post-hoc Calibration ===")
    os.makedirs(RECONSTRUCTION_DIR, exist_ok=True)

    print("Loading raw reconstruction and Hugonnet constraints...")
    recon = pd.read_csv(HYPSOMETRY_RECON_RAW_CSV)
    labels = pd.read_csv(HUGONNET_MULTIPERIOD_LABELS_CSV)

    required_recon = {"rgi_id", "year", "predicted_smb_m"}
    missing_recon = required_recon - set(recon.columns)
    if missing_recon:
        raise RuntimeError(f"Missing reconstruction columns: {sorted(missing_recon)}")

    labels = labels[labels["qc_pass"]].copy() if "qc_pass" in labels.columns else labels.copy()
    labels = labels[
        labels["hugonnet_dmdtda_mwe_yr"].notna()
        & labels["weak_weight_norm"].notna()
    ].copy()

    recon_idx = recon.set_index(["rgi_id", "year"])["predicted_smb_m"]
    qc_rows = []
    offsets = []

    print("Computing period residuals...")
    for _, row in labels.iterrows():
        years = list(range(int(row["period_start_year"]), int(row["period_end_year"])))
        keys = [(row["rgi_id"], year) for year in years]
        if not all(key in recon_idx.index for key in keys):
            qc_rows.append(
                {
                    "issue": "missing_reconstruction_year",
                    "rgi_id": row["rgi_id"],
                    "period": row["period"],
                    "count": sum(key not in recon_idx.index for key in keys),
                }
            )
            continue

        model_mean = float(np.mean([recon_idx.loc[key] for key in keys]))
        residual = float(row["hugonnet_dmdtda_mwe_yr"] - model_mean)
        offsets.append(
            {
                "rgi_id": row["rgi_id"],
                "period": row["period"],
                "period_start_year": int(row["period_start_year"]),
                "period_end_year": int(row["period_end_year"]),
                "period_n_years": int(row["period_n_years"]),
                "model_period_mean_mwe_yr": model_mean,
                "hugonnet_dmdtda_mwe_yr": float(row["hugonnet_dmdtda_mwe_yr"]),
                "period_residual_mwe_yr": residual,
                "weak_weight_norm": float(row["weak_weight_norm"]),
                "hugonnet_err_dmdtda_mwe_yr": float(row["hugonnet_err_dmdtda_mwe_yr"]),
            }
        )

    offset_df = pd.DataFrame(offsets)
    if offset_df.empty:
        raise RuntimeError("No valid Hugonnet calibration offsets were computed.")

    print("Aggregating glacier-level offsets...")
    glacier_offsets = (
        offset_df.assign(weighted_residual=offset_df["period_residual_mwe_yr"] * offset_df["weak_weight_norm"])
        .groupby("rgi_id")
        .agg(
            calibration_offset_mwe_yr=("weighted_residual", "sum"),
            weight_sum=("weak_weight_norm", "sum"),
            n_constraints=("period", "count"),
            mean_abs_period_residual_before=("period_residual_mwe_yr", lambda x: float(np.mean(np.abs(x)))),
        )
        .reset_index()
    )
    glacier_offsets["calibration_offset_mwe_yr"] = (
        glacier_offsets["calibration_offset_mwe_yr"] / glacier_offsets["weight_sum"]
    )

    calibrated = recon.merge(
        glacier_offsets[["rgi_id", "calibration_offset_mwe_yr", "n_constraints"]],
        on="rgi_id",
        how="left",
    )
    calibrated["calibration_offset_mwe_yr"] = calibrated["calibration_offset_mwe_yr"].fillna(0.0)
    calibrated["n_constraints"] = calibrated["n_constraints"].fillna(0).astype(int)
    calibrated["predicted_smb_calibrated_m"] = (
        calibrated["predicted_smb_m"] + calibrated["calibration_offset_mwe_yr"]
    )
    calibrated["calibration_offset_clipped_mwe_yr"] = calibrated[
        "calibration_offset_mwe_yr"
    ].clip(
        -HUGONNET_CALIBRATION_OFFSET_CLIP,
        HUGONNET_CALIBRATION_OFFSET_CLIP,
    )
    calibrated["calibration_offset_conservative_mwe_yr"] = (
        HUGONNET_CALIBRATION_OFFSET_SHRINK
        * calibrated["calibration_offset_clipped_mwe_yr"]
    )
    calibrated["predicted_smb_conservative_m"] = (
        calibrated["predicted_smb_m"]
        + calibrated["calibration_offset_conservative_mwe_yr"]
    )

    # Evaluate constraint agreement before and after calibration.
    cal_idx = calibrated.set_index(["rgi_id", "year"])["predicted_smb_calibrated_m"]
    cons_idx = calibrated.set_index(["rgi_id", "year"])["predicted_smb_conservative_m"]
    after_rows = []
    for item in offsets:
        years = range(item["period_start_year"], item["period_end_year"])
        calibrated_values = [cal_idx.loc[(item["rgi_id"], year)] for year in years]
        conservative_values = [cons_idx.loc[(item["rgi_id"], year)] for year in years]
        calibrated_mean = float(np.mean(calibrated_values))
        conservative_mean = float(np.mean(conservative_values))
        after_rows.append(
            {
                **item,
                "calibrated_period_mean_mwe_yr": calibrated_mean,
                "calibrated_period_residual_mwe_yr": item["hugonnet_dmdtda_mwe_yr"] - calibrated_mean,
                "conservative_period_mean_mwe_yr": conservative_mean,
                "conservative_period_residual_mwe_yr": item["hugonnet_dmdtda_mwe_yr"] - conservative_mean,
            }
        )
    qc = pd.DataFrame(after_rows)

    offset_once = calibrated.drop_duplicates("rgi_id").copy()

    summary_rows = [
        {"metric": "constraints_used", "value": len(qc)},
        {"metric": "glaciers_calibrated", "value": glacier_offsets["rgi_id"].nunique()},
        {"metric": "mean_abs_residual_before", "value": float(qc["period_residual_mwe_yr"].abs().mean())},
        {"metric": "mean_abs_residual_full", "value": float(qc["calibrated_period_residual_mwe_yr"].abs().mean())},
        {"metric": "mean_abs_residual_conservative", "value": float(qc["conservative_period_residual_mwe_yr"].abs().mean())},
        {"metric": "rmse_residual_before", "value": float(np.sqrt(np.mean(qc["period_residual_mwe_yr"] ** 2)))},
        {"metric": "rmse_residual_full", "value": float(np.sqrt(np.mean(qc["calibrated_period_residual_mwe_yr"] ** 2)))},
        {"metric": "rmse_residual_conservative", "value": float(np.sqrt(np.mean(qc["conservative_period_residual_mwe_yr"] ** 2)))},
        {"metric": "full_mean_offset", "value": float(offset_once["calibration_offset_mwe_yr"].mean())},
        {"metric": "full_median_offset", "value": float(offset_once["calibration_offset_mwe_yr"].median())},
        {"metric": "full_std_offset", "value": float(offset_once["calibration_offset_mwe_yr"].std())},
        {"metric": "conservative_mean_offset", "value": float(offset_once["calibration_offset_conservative_mwe_yr"].mean())},
        {"metric": "conservative_median_offset", "value": float(offset_once["calibration_offset_conservative_mwe_yr"].median())},
        {"metric": "conservative_std_offset", "value": float(offset_once["calibration_offset_conservative_mwe_yr"].std())},
        {"metric": "clip_abs_mwe_yr", "value": HUGONNET_CALIBRATION_OFFSET_CLIP},
        {"metric": "shrink_factor", "value": HUGONNET_CALIBRATION_OFFSET_SHRINK},
    ]
    summary = pd.DataFrame(summary_rows)

    calibrated_path = safe_to_csv(calibrated, HYPSOMETRY_RECON_CALIBRATED_CSV, index=False)
    conservative_cols = [
        col
        for col in calibrated.columns
        if col != "predicted_smb_calibrated_m"
    ]
    conservative_path = safe_to_csv(
        calibrated[conservative_cols],
        HYPSOMETRY_RECON_CONSERVATIVE_CSV,
        index=False,
    )

    regional = []
    for year, group in calibrated.groupby("year"):
        weights = group["area_km2"]
        regional.append(
            {
                "year": int(year),
                "raw_area_weighted_smb_m": float(np.average(group["predicted_smb_m"], weights=weights)),
                "full_calibrated_area_weighted_smb_m": float(
                    np.average(group["predicted_smb_calibrated_m"], weights=weights)
                ),
                "conservative_area_weighted_smb_m": float(
                    np.average(group["predicted_smb_conservative_m"], weights=weights)
                ),
                "n_glaciers": int(len(group)),
                "total_area_km2": float(weights.sum()),
            }
        )
    regional_path = safe_to_csv(
        pd.DataFrame(regional),
        HUGONNET_CALIBRATION_REGIONAL_CSV,
        index=False,
    )

    try:
        qc_path = HUGONNET_CALIBRATION_QC_CSV
        f = open(qc_path, "w", encoding="utf-8", newline="")
    except PermissionError:
        root, ext = os.path.splitext(HUGONNET_CALIBRATION_QC_CSV)
        qc_path = f"{root}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
        print(f"WARNING: could not overwrite locked file. Saved fallback -> {qc_path}")
        f = open(qc_path, "w", encoding="utf-8", newline="")
    with f:
        summary.to_csv(f, index=False)
        f.write("\n# period_residuals\n")
        qc.to_csv(f, index=False)

    print(f"Saved calibrated reconstruction -> {calibrated_path}")
    print(f"Saved conservative reconstruction -> {conservative_path}")
    print(f"Saved regional time series -> {regional_path}")
    print(f"Saved calibration QC -> {qc_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
