"""Conservative Hugonnet post-hoc calibration without overwriting full outputs."""
from __future__ import annotations

import os
import sys
import argparse

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from config import (  # noqa: E402
    HUGONNET_CALIBRATION_CONSERVATIVE_QC_CSV,
    HUGONNET_CALIBRATION_CONSERVATIVE_REGIONAL_CSV,
    HUGONNET_CALIBRATION_OFFSET_CLIP,
    HUGONNET_CALIBRATION_OFFSET_SHRINK,
    HUGONNET_MULTIPERIOD_LABELS_CSV,
    HYPSOMETRY_RECON_CONSERVATIVE_CSV,
    HYPSOMETRY_RECON_RAW_CSV,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reconstruction", default=HYPSOMETRY_RECON_RAW_CSV)
    parser.add_argument("--output", default=HYPSOMETRY_RECON_CONSERVATIVE_CSV)
    parser.add_argument("--regional-output", default=HUGONNET_CALIBRATION_CONSERVATIVE_REGIONAL_CSV)
    parser.add_argument("--qc-output", default=HUGONNET_CALIBRATION_CONSERVATIVE_QC_CSV)
    parser.add_argument("--clip", type=float, default=HUGONNET_CALIBRATION_OFFSET_CLIP)
    parser.add_argument("--shrink", type=float, default=HUGONNET_CALIBRATION_OFFSET_SHRINK)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("=== Reconstruction Step 03: Conservative Hugonnet Calibration ===")
    for path in [args.output, args.regional_output, args.qc_output]:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    recon = pd.read_csv(args.reconstruction)
    labels = pd.read_csv(HUGONNET_MULTIPERIOD_LABELS_CSV)
    labels = labels[labels["qc_pass"]].copy() if "qc_pass" in labels.columns else labels.copy()
    labels = labels[
        labels["hugonnet_dmdtda_mwe_yr"].notna()
        & labels["weak_weight_norm"].notna()
    ].copy()

    recon_idx = recon.set_index(["rgi_id", "year"])["predicted_smb_m"]
    offsets = []
    for _, row in labels.iterrows():
        years = list(range(int(row["period_start_year"]), int(row["period_end_year"])))
        keys = [(row["rgi_id"], year) for year in years]
        if not all(key in recon_idx.index for key in keys):
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
        raise RuntimeError("No valid calibration constraints.")

    glacier_offsets = (
        offset_df.assign(weighted_residual=offset_df["period_residual_mwe_yr"] * offset_df["weak_weight_norm"])
        .groupby("rgi_id")
        .agg(
            full_offset=("weighted_residual", "sum"),
            weight_sum=("weak_weight_norm", "sum"),
            n_constraints=("period", "count"),
        )
        .reset_index()
    )
    glacier_offsets["full_offset"] = glacier_offsets["full_offset"] / glacier_offsets["weight_sum"]
    glacier_offsets["calibration_offset_conservative_mwe_yr"] = (
        args.shrink
        * glacier_offsets["full_offset"].clip(
            -args.clip,
            args.clip,
        )
    )

    out = recon.merge(
        glacier_offsets[["rgi_id", "full_offset", "calibration_offset_conservative_mwe_yr", "n_constraints"]],
        on="rgi_id",
        how="left",
    )
    out["full_offset"] = out["full_offset"].fillna(0.0)
    out["calibration_offset_conservative_mwe_yr"] = out[
        "calibration_offset_conservative_mwe_yr"
    ].fillna(0.0)
    out["n_constraints"] = out["n_constraints"].fillna(0).astype(int)
    out["predicted_smb_conservative_m"] = (
        out["predicted_smb_m"] + out["calibration_offset_conservative_mwe_yr"]
    )

    cons_idx = out.set_index(["rgi_id", "year"])["predicted_smb_conservative_m"]
    qc_rows = []
    for item in offsets:
        years = range(item["period_start_year"], item["period_end_year"])
        values = [cons_idx.loc[(item["rgi_id"], year)] for year in years]
        cons_mean = float(np.mean(values))
        qc_rows.append(
            {
                **item,
                "conservative_period_mean_mwe_yr": cons_mean,
                "conservative_period_residual_mwe_yr": item["hugonnet_dmdtda_mwe_yr"] - cons_mean,
            }
        )
    qc = pd.DataFrame(qc_rows)
    offset_once = out.drop_duplicates("rgi_id")
    summary = pd.DataFrame(
        [
            {"metric": "constraints_used", "value": len(qc)},
            {"metric": "glaciers_calibrated", "value": int((offset_once["n_constraints"] > 0).sum())},
            {"metric": "mean_abs_residual_before", "value": float(qc["period_residual_mwe_yr"].abs().mean())},
            {"metric": "mean_abs_residual_conservative", "value": float(qc["conservative_period_residual_mwe_yr"].abs().mean())},
            {"metric": "rmse_residual_before", "value": float(np.sqrt(np.mean(qc["period_residual_mwe_yr"] ** 2)))},
            {"metric": "rmse_residual_conservative", "value": float(np.sqrt(np.mean(qc["conservative_period_residual_mwe_yr"] ** 2)))},
            {"metric": "full_offset_mean", "value": float(offset_once["full_offset"].mean())},
            {"metric": "full_offset_std", "value": float(offset_once["full_offset"].std())},
            {"metric": "conservative_offset_mean", "value": float(offset_once["calibration_offset_conservative_mwe_yr"].mean())},
            {"metric": "conservative_offset_std", "value": float(offset_once["calibration_offset_conservative_mwe_yr"].std())},
            {"metric": "clip_abs_mwe_yr", "value": args.clip},
            {"metric": "shrink_factor", "value": args.shrink},
        ]
    )

    regional = []
    for year, group in out.groupby("year"):
        weights = group["area_km2"]
        regional.append(
            {
                "year": int(year),
                "raw_area_weighted_smb_m": float(np.average(group["predicted_smb_m"], weights=weights)),
                "conservative_area_weighted_smb_m": float(
                    np.average(group["predicted_smb_conservative_m"], weights=weights)
                ),
                "n_glaciers": int(len(group)),
                "total_area_km2": float(weights.sum()),
            }
        )

    with open(args.output, "w", encoding="utf-8", newline="") as f:
        out.iloc[:0].to_csv(f, index=False)
        for start in range(0, len(out), 50_000):
            out.iloc[start : start + 50_000].to_csv(f, index=False, header=False)
    pd.DataFrame(regional).to_csv(args.regional_output, index=False)
    with open(args.qc_output, "w", encoding="utf-8", newline="") as f:
        summary.to_csv(f, index=False)
        f.write("\n# period_residuals\n")
        qc.to_csv(f, index=False)

    print(f"Saved conservative reconstruction -> {args.output}")
    print(f"Saved regional time series -> {args.regional_output}")
    print(f"Saved QC -> {args.qc_output}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
