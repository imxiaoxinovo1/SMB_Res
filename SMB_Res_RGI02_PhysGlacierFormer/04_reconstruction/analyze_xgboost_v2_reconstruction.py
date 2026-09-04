"""Audit the final RGI02 reconstruction and compare regional totals consistently."""
from __future__ import annotations

import argparse
import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from scipy.stats import linregress

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from config import (  # noqa: E402
    MALLES_REGION_NC,
    PHYS_V2_MALLES_COMPARISON,
    PHYS_V2_RECONSTRUCTION_CALIBRATED_CSV,
    PHYS_V2_RECONSTRUCTION_OOD_SUMMARY,
    PHYS_V2_RECONSTRUCTION_QC_SUMMARY,
    PHYS_V2_RECONSTRUCTION_REGIONAL_CSV,
    PHYS_V2_ZEMP_COMPARISON,
    RGI02_SHP,
    ZEMP_RGI02_CSV,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reconstruction", default=PHYS_V2_RECONSTRUCTION_CALIBRATED_CSV)
    parser.add_argument("--year-min", type=int, default=1951)
    parser.add_argument("--year-max", type=int, default=2024)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--block-years", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def malles_region02() -> pd.DataFrame:
    with Dataset(MALLES_REGION_NC) as dataset:
        region_index = int(np.flatnonzero(np.asarray(dataset["Region"][:]) == 2)[0])
        values = np.asarray(dataset["Mass change"][:, :, region_index], dtype=float)
        areas = np.asarray(dataset["Area"][:, :, region_index], dtype=float)
        time_range = str(getattr(dataset["Time"], "range", "1901 - 2018"))
    start_year, end_year = [int(value.strip()) for value in time_range.split("-")]
    years = np.arange(start_year, end_year + 1)
    if values.shape[1] != len(years):
        raise RuntimeError("Malles time metadata does not match the time dimension.")
    valid = ~np.all(np.isnan(values), axis=0)
    values = values[:, valid]
    areas = areas[:, valid]
    years = years[valid]
    specific = np.divide(
        values * 1000.0,
        areas,
        out=np.full_like(values, np.nan),
        where=np.isfinite(areas) & (areas > 0),
    )
    return pd.DataFrame(
        {
            "year": years,
            "malles_mass_change_gt": np.nanmean(values, axis=0),
            "malles_p05_gt": np.nanpercentile(values, 5, axis=0),
            "malles_p95_gt": np.nanpercentile(values, 95, axis=0),
            "malles_area_km2": np.nanmean(areas, axis=0),
            "malles_specific_mwe_yr": np.nanmean(specific, axis=0),
            "malles_specific_p05_mwe_yr": np.nanpercentile(specific, 5, axis=0),
            "malles_specific_p95_mwe_yr": np.nanpercentile(specific, 95, axis=0),
        }
    )


def comparison_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(observed) & np.isfinite(predicted)
    observed, predicted = observed[valid], predicted[valid]
    residual = predicted - observed
    return {
        "n_years": int(len(observed)),
        "pearson_r": float(np.corrcoef(observed, predicted)[0, 1]),
        "rmse_gt": float(np.sqrt(np.mean(residual**2))),
        "mae_gt": float(np.mean(np.abs(residual))),
        "bias_gt": float(np.mean(residual)),
        "cumulative_model_gt": float(predicted.sum()),
        "cumulative_malles_gt": float(observed.sum()),
    }


def zemp_region02() -> pd.DataFrame:
    frame = pd.read_csv(ZEMP_RGI02_CSV, comment="#", skipinitialspace=True)
    frame.columns = frame.columns.str.strip()
    return frame[["Year", "INT_mwe", "sig_Total_mwe"]].rename(
        columns={
            "Year": "year",
            "INT_mwe": "zemp_specific_mass_change_mwe_yr",
            "sig_Total_mwe": "zemp_uncertainty_mwe_yr",
        }
    )


def block_bootstrap_intervals(
    observed: np.ndarray,
    predicted: np.ndarray,
    n_bootstrap: int,
    block_years: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    n = len(observed)
    if n < block_years * 2:
        return {}
    samples = {"pearson_r": [], "rmse_gt": [], "bias_gt": []}
    for _ in range(n_bootstrap):
        indices: list[int] = []
        while len(indices) < n:
            start = int(rng.integers(0, n - block_years + 1))
            indices.extend(range(start, start + block_years))
        selected = np.asarray(indices[:n])
        values = comparison_metrics(observed[selected], predicted[selected])
        for metric in samples:
            samples[metric].append(values[metric])
    intervals = {}
    for metric, values in samples.items():
        low, high = np.quantile(values, [0.025, 0.975])
        intervals[f"{metric}_ci_low"] = float(low)
        intervals[f"{metric}_ci_high"] = float(high)
    return intervals


def trend_per_decade(year: pd.Series, values: pd.Series, start_year: int = 1980) -> tuple[float, float]:
    mask = year >= start_year
    result = linregress(year[mask], values[mask])
    return float(result.slope * 10.0), float(result.pvalue)


def main() -> None:
    args = parse_args()
    usecols = [
        "rgi_id", "o2region", "year", "predicted_smb_m",
        "predicted_smb_conservative_m", "area_km2", "era5_source",
        "recommended_area_domain", "area_outside_training_range",
        "feature_outside_training_fraction", "n_constraints",
    ]
    reconstruction = pd.read_csv(args.reconstruction, usecols=usecols)
    inventory = gpd.read_file(RGI02_SHP, columns=["rgi_id", "area_km2"]).drop(columns="geometry")
    expected_years = np.arange(args.year_min, args.year_max + 1)
    expected_rows = len(inventory) * len(expected_years)

    duplicate_rows = int(reconstruction.duplicated(["rgi_id", "year"]).sum())
    year_counts = reconstruction.groupby("rgi_id")["year"].nunique()
    reconstructed_ids = set(reconstruction["rgi_id"].astype(str))
    inventory_ids = set(inventory["rgi_id"].astype(str))
    complete = (
        len(reconstruction) == expected_rows
        and duplicate_rows == 0
        and reconstructed_ids == inventory_ids
        and year_counts.eq(len(expected_years)).all()
        and reconstruction["year"].min() == args.year_min
        and reconstruction["year"].max() == args.year_max
    )

    regional_rows = []
    for year, group in reconstruction.groupby("year", sort=True):
        all_area = group["area_km2"].to_numpy()
        recommended = group["recommended_area_domain"].astype(bool).to_numpy()
        row = {"year": int(year), "n_glaciers_all": int(len(group))}
        for label, column in [
            ("raw", "predicted_smb_m"),
            ("calibrated", "predicted_smb_conservative_m"),
        ]:
            values = group[column].to_numpy()
            row[f"{label}_area_weighted_smb_all_m"] = float(np.average(values, weights=all_area))
            row[f"{label}_mass_change_all_gt"] = float(np.sum(values * all_area) * 0.001)
            row[f"{label}_area_weighted_smb_ge05_m"] = float(
                np.average(values[recommended], weights=all_area[recommended])
            )
            row[f"{label}_mass_change_ge05_gt"] = float(
                np.sum(values[recommended] * all_area[recommended]) * 0.001
            )
        row["n_glaciers_ge05"] = int(recommended.sum())
        row["inventory_area_all_km2"] = float(all_area.sum())
        row["inventory_area_ge05_km2"] = float(all_area[recommended].sum())
        regional_rows.append(row)
    regional = pd.DataFrame(regional_rows)

    qc_rows: list[dict[str, object]] = [
        {"metric": "complete_expected_glacier_year_grid", "value": bool(complete)},
        {"metric": "rows", "value": len(reconstruction)},
        {"metric": "expected_rows", "value": expected_rows},
        {"metric": "unique_glaciers", "value": reconstruction["rgi_id"].nunique()},
        {"metric": "inventory_glaciers", "value": len(inventory)},
        {"metric": "duplicate_glacier_year_rows", "value": duplicate_rows},
        {"metric": "missing_inventory_glaciers", "value": len(inventory_ids - reconstructed_ids)},
        {"metric": "unexpected_glaciers", "value": len(reconstructed_ids - inventory_ids)},
        {"metric": "year_min", "value": int(reconstruction["year"].min())},
        {"metric": "year_max", "value": int(reconstruction["year"].max())},
        {"metric": "nonfinite_raw_predictions", "value": int((~np.isfinite(reconstruction["predicted_smb_m"])).sum())},
        {"metric": "nonfinite_calibrated_predictions", "value": int((~np.isfinite(reconstruction["predicted_smb_conservative_m"])).sum())},
        {"metric": "alaska_era5_glaciers", "value": reconstruction.loc[reconstruction.era5_source == "alaska", "rgi_id"].nunique()},
        {"metric": "recommended_ge05_glaciers", "value": reconstruction.loc[reconstruction.recommended_area_domain, "rgi_id"].nunique()},
        {"metric": "calibrated_glaciers", "value": reconstruction.loc[reconstruction.n_constraints > 0, "rgi_id"].nunique()},
    ]
    for label in ["raw", "calibrated"]:
        column = f"{label}_area_weighted_smb_all_m"
        trend, pvalue = trend_per_decade(regional["year"], regional[column])
        qc_rows.extend(
            [
                {"metric": f"{label}_regional_mean_1951_2024_mwe_yr", "value": float(regional[column].mean())},
                {"metric": f"{label}_regional_trend_1980_2024_mwe_decade", "value": trend},
                {"metric": f"{label}_regional_trend_1980_2024_pvalue", "value": pvalue},
                {"metric": f"{label}_regional_2023_mwe_yr", "value": float(regional.loc[regional.year == 2023, column].iloc[0])},
                {"metric": f"{label}_regional_2024_mwe_yr", "value": float(regional.loc[regional.year == 2024, column].iloc[0])},
            ]
        )

    glacier_once = reconstruction.drop_duplicates("rgi_id").copy()
    glacier_once["feature_ood_class"] = pd.cut(
        glacier_once["feature_outside_training_fraction"],
        bins=[-np.inf, 0.0, 0.05, 0.10, np.inf],
        labels=["none", "low_0_5pct", "moderate_5_10pct", "high_gt10pct"],
    )
    ood = (
        glacier_once.groupby("feature_ood_class", observed=True)
        .agg(
            n_glaciers=("rgi_id", "size"),
            total_area_km2=("area_km2", "sum"),
            median_area_km2=("area_km2", "median"),
            median_feature_outside_fraction=("feature_outside_training_fraction", "median"),
            glaciers_outside_training_area=("area_outside_training_range", "sum"),
        )
        .reset_index()
    )

    malles = malles_region02()
    comparison = regional.merge(malles, on="year", how="inner")
    metric_rows = []
    rng = np.random.default_rng(args.seed)
    for label, mass_column, specific_column in [
        ("raw_all_rgi02", "raw_mass_change_all_gt", "raw_area_weighted_smb_all_m"),
        (
            "calibrated_all_rgi02",
            "calibrated_mass_change_all_gt",
            "calibrated_area_weighted_smb_all_m",
        ),
    ]:
        observed = comparison["malles_mass_change_gt"].to_numpy()
        predicted = comparison[mass_column].to_numpy()
        specific_observed = comparison["malles_specific_mwe_yr"].to_numpy()
        specific_predicted = comparison[specific_column].to_numpy()
        specific_metrics = comparison_metrics(specific_observed, specific_predicted)
        specific_intervals = block_bootstrap_intervals(
            specific_observed,
            specific_predicted,
            args.bootstrap,
            args.block_years,
            rng,
        )
        metric_rows.append(
            {
                "comparison": label,
                "start_year": int(comparison.year.min()),
                "end_year": int(comparison.year.max()),
                "bootstrap_method": f"moving blocks of {args.block_years} years",
                "geometry_note": "fixed RGI v7 area (this study) versus evolving Malles area",
                **comparison_metrics(observed, predicted),
                **block_bootstrap_intervals(
                    observed, predicted, args.bootstrap, args.block_years, rng
                ),
                "specific_pearson_r": specific_metrics["pearson_r"],
                "specific_rmse_mwe_yr": specific_metrics["rmse_gt"],
                "specific_mae_mwe_yr": specific_metrics["mae_gt"],
                "specific_bias_mwe_yr": specific_metrics["bias_gt"],
                "specific_pearson_r_ci_low": specific_intervals["pearson_r_ci_low"],
                "specific_pearson_r_ci_high": specific_intervals["pearson_r_ci_high"],
                "specific_rmse_mwe_yr_ci_low": specific_intervals["rmse_gt_ci_low"],
                "specific_rmse_mwe_yr_ci_high": specific_intervals["rmse_gt_ci_high"],
                "specific_bias_mwe_yr_ci_low": specific_intervals["bias_gt_ci_low"],
                "specific_bias_mwe_yr_ci_high": specific_intervals["bias_gt_ci_high"],
            }
        )
    comparison.attrs["metrics"] = pd.DataFrame(metric_rows)

    zemp_comparison = regional.merge(zemp_region02(), on="year", how="inner")
    zemp_metric_rows = []
    for label, column in [
        ("raw_all_rgi02", "raw_area_weighted_smb_all_m"),
        ("calibrated_all_rgi02", "calibrated_area_weighted_smb_all_m"),
    ]:
        observed = zemp_comparison["zemp_specific_mass_change_mwe_yr"].to_numpy()
        predicted = zemp_comparison[column].to_numpy()
        values = comparison_metrics(observed, predicted)
        intervals = block_bootstrap_intervals(
            observed, predicted, args.bootstrap, args.block_years, rng
        )
        zemp_metric_rows.append(
            {
                "comparison": label,
                "start_year": int(zemp_comparison.year.min()),
                "end_year": int(zemp_comparison.year.max()),
                "bootstrap_method": f"moving blocks of {args.block_years} years",
                "n_years": values["n_years"],
                "pearson_r": values["pearson_r"],
                "rmse_mwe_yr": values["rmse_gt"],
                "mae_mwe_yr": values["mae_gt"],
                "bias_mwe_yr": values["bias_gt"],
                "pearson_r_ci_low": intervals["pearson_r_ci_low"],
                "pearson_r_ci_high": intervals["pearson_r_ci_high"],
                "rmse_mwe_yr_ci_low": intervals["rmse_gt_ci_low"],
                "rmse_mwe_yr_ci_high": intervals["rmse_gt_ci_high"],
                "bias_mwe_yr_ci_low": intervals["bias_gt_ci_low"],
                "bias_mwe_yr_ci_high": intervals["bias_gt_ci_high"],
            }
        )

    os.makedirs(os.path.dirname(PHYS_V2_RECONSTRUCTION_QC_SUMMARY), exist_ok=True)
    pd.DataFrame(qc_rows).to_csv(PHYS_V2_RECONSTRUCTION_QC_SUMMARY, index=False)
    ood.to_csv(PHYS_V2_RECONSTRUCTION_OOD_SUMMARY, index=False)
    regional.to_csv(PHYS_V2_RECONSTRUCTION_REGIONAL_CSV, index=False)
    with open(PHYS_V2_MALLES_COMPARISON, "w", encoding="utf-8", newline="") as stream:
        pd.DataFrame(metric_rows).to_csv(stream, index=False)
        stream.write("\n# annual_series\n")
        comparison.to_csv(stream, index=False)
    with open(PHYS_V2_ZEMP_COMPARISON, "w", encoding="utf-8", newline="") as stream:
        pd.DataFrame(zemp_metric_rows).to_csv(stream, index=False)
        stream.write("\n# annual_series\n")
        zemp_comparison.to_csv(stream, index=False)

    print(pd.DataFrame(qc_rows).to_string(index=False))
    print("\nMalles & Marzeion comparison (full RGI02 only):")
    print(pd.DataFrame(metric_rows).to_string(index=False))
    print("\nZemp et al. comparison (shared WGMS information; not independent):")
    print(pd.DataFrame(zemp_metric_rows).to_string(index=False))
    print(f"Saved QC -> {PHYS_V2_RECONSTRUCTION_QC_SUMMARY}")
    print(f"Saved OOD summary -> {PHYS_V2_RECONSTRUCTION_OOD_SUMMARY}")
    print(f"Saved regional series -> {PHYS_V2_RECONSTRUCTION_REGIONAL_CSV}")
    print(f"Saved external comparison -> {PHYS_V2_MALLES_COMPARISON}")
    print(f"Saved Zemp consistency comparison -> {PHYS_V2_ZEMP_COMPARISON}")
    if not complete:
        raise RuntimeError("Reconstruction failed the expected glacier-year completeness checks.")


if __name__ == "__main__":
    main()
