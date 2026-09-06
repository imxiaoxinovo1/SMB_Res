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
    GLAMBIE_RGI02_CSV,
    MALLES_REGION_NC,
    PHYS_V2_AMPLITUDE_DIAGNOSTICS,
    PHYS_V2_AMPLITUDE_SERIES,
    PHYS_V2_GLAMBIE_ANNUAL_METRICS,
    PHYS_V2_GLAMBIE_ANNUAL_SERIES,
    PHYS_V2_GLAMBIE_COMPARISON,
    PHYS_V2_MALLES_COMPARISON,
    PHYS_V2_RECONSTRUCTION_CALIBRATED_CSV,
    PHYS_V2_RECONSTRUCTION_OOD_SUMMARY,
    PHYS_V2_RECONSTRUCTION_QC_SUMMARY,
    PHYS_V2_RECONSTRUCTION_REGIONAL_CSV,
    PHYS_V2_ZEMP_COMPARISON,
    PHYS_V2_RESULT_DIR,
    PHYS_V2_SEQUENCES_NPZ,
    RGI02_SHP,
    ZEMP_RGI02_CSV,
)


GLAMBIE_RGI02 = {
    "start_year": 2000,
    "end_year": 2023,
    "specific_mass_change_mwe_yr": -0.68,
    "specific_uncertainty_mwe_yr": 0.06,
    "mass_change_gt_yr": -9.0,
    "mass_change_uncertainty_gt_yr": 0.9,
    "area_2000_km2": 14_602.0,
    "source": "GlaMBIE Team (2025), Nature, doi:10.1038/s41586-024-08545-z, Table 1",
}


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
        "pearson_r": (
            float(np.corrcoef(observed, predicted)[0, 1])
            if len(observed) > 1 and np.std(observed) > 0 and np.std(predicted) > 0
            else np.nan
        ),
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
        low, high = np.nanquantile(values, [0.025, 0.975])
        intervals[f"{metric}_ci_low"] = float(low)
        intervals[f"{metric}_ci_high"] = float(high)
    return intervals


def trend_per_decade(year: pd.Series, values: pd.Series, start_year: int = 1980) -> tuple[float, float]:
    mask = year >= start_year
    result = linregress(year[mask], values[mask])
    return float(result.slope * 10.0), float(result.pvalue)


def glambie_period_comparison(regional: pd.DataFrame) -> pd.DataFrame:
    """Retain the published Table-1 mean as a separate time-support comparison."""
    reference = GLAMBIE_RGI02
    period = regional[regional["year"].between(reference["start_year"], reference["end_year"])]
    if len(period) != reference["end_year"] - reference["start_year"] + 1:
        raise RuntimeError("Reconstruction does not fully cover the GlaMBIE comparison period.")

    rows = []
    for label in ["raw", "calibrated"]:
        specific = float(period[f"{label}_area_weighted_smb_all_m"].mean())
        mass = float(period[f"{label}_mass_change_all_gt"].mean())
        rows.append(
            {
                "comparison": f"{label}_all_rgi02",
                "start_year": reference["start_year"],
                "end_year": reference["end_year"],
                "model_specific_mass_change_mwe_yr": specific,
                "glambie_specific_mass_change_mwe_yr": reference["specific_mass_change_mwe_yr"],
                "glambie_specific_uncertainty_mwe_yr": reference["specific_uncertainty_mwe_yr"],
                "specific_difference_mwe_yr": specific - reference["specific_mass_change_mwe_yr"],
                "model_mass_change_gt_yr": mass,
                "glambie_mass_change_gt_yr": reference["mass_change_gt_yr"],
                "glambie_mass_change_uncertainty_gt_yr": reference["mass_change_uncertainty_gt_yr"],
                "mass_difference_gt_yr": mass - reference["mass_change_gt_yr"],
                "model_fixed_area_km2": float(period["inventory_area_all_km2"].mean()),
                "glambie_area_2000_km2": reference["area_2000_km2"],
                "source": reference["source"],
                "interpretation": (
                    "period-mean external consistency only; GlaMBIE combines glaciological, "
                    "DEM-differencing, altimetry and other regional inputs and is not fully "
                    "independent of WGMS or Hugonnet; model uses fixed RGI v7 geometry"
                ),
            }
        )
    return pd.DataFrame(rows)


def glambie_region02(path: str = GLAMBIE_RGI02_CSV) -> pd.DataFrame:
    """Read annual Oct-Sep totals, labeled by the September end year."""
    frame = pd.read_csv(path).sort_values("start_dates").reset_index(drop=True)
    required = {
        "start_dates", "end_dates", "region", "glacier_area", "combined_gt",
        "combined_gt_errors", "combined_mwe", "combined_mwe_errors",
    }
    if not required.issubset(frame):
        raise ValueError(f"GlaMBIE columns missing: {sorted(required - set(frame))}")
    if frame.empty or not frame["region"].eq("western_canada_us").all():
        raise ValueError("Expected GlaMBIE western_canada_us regional data.")
    start = frame["start_dates"].to_numpy(float)
    end = frame["end_dates"].to_numpy(float)
    numeric = frame[sorted(required - {"region"})].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("Nonfinite GlaMBIE combined data or dates.")
    if not (
        np.allclose(end - start, 1.0, rtol=0, atol=1e-7)
        and np.allclose(start % 1, 0.75, rtol=0, atol=1e-7)
        and np.allclose(end % 1, 0.75, rtol=0, atol=1e-7)
    ):
        raise ValueError("Expected annual Oct-Sep intervals, not calendar-year data.")
    years = np.floor(end).astype(int)
    if not np.all(np.diff(years) == 1):
        raise ValueError("GlaMBIE intervals contain duplicate or missing years.")
    if (frame["glacier_area"] <= 0).any() or (
        frame[["combined_gt_errors", "combined_mwe_errors"]] < 0
    ).any().any():
        raise ValueError("Invalid GlaMBIE area or uncertainty.")
    frame.insert(0, "year", years)
    return frame.rename(columns={col: f"glambie_{col}" for col in frame if col != "year"})


def glambie_annual_comparison(
    regional: pd.DataFrame, n_bootstrap: int, block_years: int, seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference = glambie_region02()
    merged = reference.merge(regional, on="year", how="left", validate="one_to_one")
    model_columns = [
        f"{label}_{quantity}" for label in ["raw", "calibrated"]
        for quantity in ["area_weighted_smb_all_m", "mass_change_all_gt"]
    ]
    if not np.isfinite(merged[model_columns].to_numpy()).all():
        raise ValueError("Reconstruction does not cover all GlaMBIE hydrological years.")
    rows = []
    for source, start, end in [
        ("combined", 2000, 2023), ("combined", 2000, 2019), ("combined", 2020, 2023),
        ("altimetry", 2013, 2022),
    ]:
        subset = merged[merged.year.between(start, end)]
        if len(subset) != end - start + 1:
            raise ValueError(f"Incomplete GlaMBIE evaluation interval {start}-{end}.")
        if source == "altimetry" and not subset["glambie_altimetry_annual_variability"].eq(1).all():
            raise ValueError("Altimetry comparison requires its own annual variability.")
        for label in ["raw", "calibrated"]:
            for unit, reference_col, model_col in [
                ("m w.e. yr-1", f"glambie_{source}_mwe", f"{label}_area_weighted_smb_all_m"),
                ("Gt yr-1", f"glambie_{source}_gt", f"{label}_mass_change_all_gt"),
            ]:
                observed = subset[reference_col].to_numpy()
                predicted = subset[model_col].to_numpy()
                if not np.isfinite(observed).all():
                    raise ValueError(f"Incomplete GlaMBIE {source} values.")
                metrics = comparison_metrics(observed, predicted)
                # Same resampling indices for each raw/calibrated comparison.
                ci = block_bootstrap_intervals(
                    observed, predicted, n_bootstrap, block_years, np.random.default_rng(seed)
                )
                row = {
                    "model": label, "reference": source,
                    "start_year": start, "end_year": end, "unit": unit,
                    "n_years": len(subset), "pearson_r": metrics["pearson_r"],
                    "rmse": metrics["rmse_gt"], "mae": metrics["mae_gt"],
                    "bias": metrics["bias_gt"], "model_mean": float(predicted.mean()),
                    "glambie_mean": float(observed.mean()),
                    "std_ratio_model_to_reference": float(np.std(predicted) / np.std(observed)),
                    "model_on_reference_slope": float(linregress(observed, predicted).slope),
                    "bootstrap": f"{n_bootstrap} moving-block resamples; block={block_years} years",
                    "time_support": "October-September; September end-year label",
                    "source": "doi:10.5904/wgms-glambie-2024-07; Dataset 1.0.0",
                    "interpretation": "external consistency; shared WGMS/Hugonnet sources; fixed versus evolving area",
                }
                for key, value in ci.items():
                    row[key.replace("rmse_gt", "rmse").replace("bias_gt", "bias")] = value
                rows.append(row)
    return merged, pd.DataFrame(rows)


def cumulative_ensemble_summary(values: np.ndarray) -> dict[str, np.ndarray]:
    """Accumulate each complete forcing trajectory before computing quantiles."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("Expected nonempty forcing-by-year array.")
    complete = np.isfinite(values).all(axis=1)
    if not complete.any():
        raise ValueError("No complete ensemble trajectories over the cumulative interval.")
    cumulative = np.cumsum(values[complete], axis=1)
    return {
        "malles_cumulative_gt": cumulative.mean(axis=0),
        "malles_p05_cumulative_gt": np.percentile(cumulative, 5, axis=0),
        "malles_p95_cumulative_gt": np.percentile(cumulative, 95, axis=0),
    }


def amplitude_sampling_diagnostics(
    reconstruction: pd.DataFrame, regional: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate OOF temporal amplitude from modeled spatial aggregation effects."""
    data = np.load(PHYS_V2_SEQUENCES_NPZ, allow_pickle=True)
    samples = pd.DataFrame({
        "glacier_id": data["glacier_ids"], "rgi_id": data["rgi_ids"],
        "year": data["years"], "end_month": data["end_months"], "obs": data["y_annual"],
    })
    keys = ["glacier_id", "rgi_id", "year"]
    tag = "xgboost_v2_compact_monthly_all_hyp_p-regularized"
    for cv in ["logo", "loyo"]:
        predictions = pd.read_csv(os.path.join(PHYS_V2_RESULT_DIR, tag, f"{tag}_{cv}_predictions.csv"))
        samples = samples.merge(predictions[keys + ["obs_annual", "pred_annual"]],
                                on=keys, how="left", validate="one_to_one")
        # Compare in the target's original float32 precision after CSV round-trip.
        if not np.array_equal(samples["obs"].to_numpy(np.float32), samples["obs_annual"].to_numpy(np.float32)):
            raise ValueError("OOF target mismatch in amplitude diagnostics.")
        samples = samples.rename(columns={"pred_annual": cv}).drop(columns="obs_annual")
    samples = samples[samples.year.between(2000, 2023) & samples.end_month.eq(9)].copy()
    if samples.empty or samples.duplicated(["rgi_id", "year"]).any():
        raise ValueError("Amplitude diagnostics need unique September-end glacier-years.")
    selected = reconstruction[reconstruction.year.between(2000, 2023)]
    samples = samples.merge(selected[["rgi_id", "year", "area_km2", "predicted_smb_m"]],
                            on=["rgi_id", "year"], how="left", validate="one_to_one")
    samples = samples.rename(columns={"predicted_smb_m": "fitted_reconstruction"})
    value_cols = ["obs", "logo", "loyo", "fitted_reconstruction"]
    if not np.isfinite(samples[value_cols + ["area_km2"]].to_numpy()).all():
        raise ValueError("Missing values in matched amplitude diagnostics.")

    annual_rows = []
    fixed_ids = samples.rgi_id.unique()
    fixed_sites = selected[selected.rgi_id.isin(fixed_ids)]
    for year, group in samples.groupby("year", sort=True):
        fixed = fixed_sites[fixed_sites.year == year]
        row = {"year": int(year), "n_observed_glaciers": len(group),
               "sample_area_km2": float(group.area_km2.sum()), "n_fixed_sites": len(fixed_ids)}
        for weighting in ["equal", "area"]:
            weights = group.area_km2 if weighting == "area" else np.ones(len(group))
            for column in value_cols:
                row[f"sample_{column}_{weighting}_mwe"] = float(np.average(group[column], weights=weights))
            fixed_weights = fixed.area_km2 if weighting == "area" else np.ones(len(fixed))
            row[f"fixed_sites_raw_{weighting}_mwe"] = float(np.average(fixed.predicted_smb_m, weights=fixed_weights))
        annual_rows.append(row)
    annual = pd.DataFrame(annual_rows).merge(regional, on="year", validate="one_to_one")
    annual = annual.merge(glambie_region02(), on="year", validate="one_to_one")
    annual["sample_area_fraction"] = annual.sample_area_km2 / annual.inventory_area_all_km2
    rows = []

    def add_row(scope, reference_name, prediction_name, frame, reference, prediction, role):
        observed = frame[reference].to_numpy(float)
        predicted = frame[prediction].to_numpy(float)
        metric = comparison_metrics(observed, predicted)
        rows.append({
            "scope": scope, "reference": reference_name, "prediction": prediction_name,
            "n": len(frame), "pearson_r": metric["pearson_r"], "rmse_mwe_yr": metric["rmse_gt"],
            "bias_mwe_yr": metric["bias_gt"],
            "std_ratio": float(np.std(predicted) / np.std(observed)) if np.std(observed) > 0 else np.nan,
            "slope": float(linregress(observed, predicted).slope) if np.std(observed) > 0 else np.nan,
            "interpretation": role,
        })

    enough = samples.groupby("rgi_id").year.transform("nunique") >= 5
    anomalies = samples[enough].copy()
    # Descriptive centering after prediction; never fed back into model fitting.
    anomalies[value_cols] = anomalies[value_cols] - anomalies.groupby("rgi_id")[value_cols].transform("mean")
    for cv in ["logo", "loyo", "fitted_reconstruction"]:
        role = "OOF prediction" if cv != "fitted_reconstruction" else "in-sample fit diagnostic; not validation"
        add_row("matched_glacier_years", "WGMS", cv, samples, "obs", cv, role)
        add_row("within_glacier_centered_min5years", "WGMS anomalies", cv, anomalies, "obs", cv, role)
    for minimum in [1, 5]:
        subset = annual[annual.n_observed_glaciers >= minimum]
        for weighting in ["equal", "area"]:
            observed = f"sample_obs_{weighting}_mwe"
            for cv in ["logo", "loyo", "fitted_reconstruction"]:
                add_row(f"annual_{weighting}_min{minimum}sites", "same-site WGMS", cv, subset,
                        observed, f"sample_{cv}_{weighting}_mwe",
                        "changing observed sample; OOF" if cv != "fitted_reconstruction" else "in-sample diagnostic")
            add_row(f"annual_{weighting}_min{minimum}sites", "GlaMBIE combined", "sample WGMS", subset,
                    "glambie_combined_mwe", observed, "sample representativeness; shared information")
    for column, label in [
        ("fixed_sites_raw_equal_mwe", "fixed observed sites, equal weights"),
        ("fixed_sites_raw_area_mwe", "fixed observed sites, area weights"),
        ("raw_area_weighted_smb_all_m", "all RGI02, area weights"),
    ]:
        add_row("fixed_model_spatial_support", "GlaMBIE combined", label, annual,
                "glambie_combined_mwe", column, "modeled sampling diagnostic; not OOF validation")
    return annual, pd.DataFrame(rows)


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

    # Dynamic covariates vary by year: the first row is not a glacier-wide OOD summary.
    glacier_once = reconstruction.groupby("rgi_id", sort=False).agg(
        area_km2=("area_km2", "first"),
        area_outside_training_range=("area_outside_training_range", "first"),
        feature_outside_training_fraction=("feature_outside_training_fraction", "mean"),
        max_feature_outside_fraction=("feature_outside_training_fraction", "max"),
    ).reset_index()
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
            median_max_feature_outside_fraction=("max_feature_outside_fraction", "median"),
            glaciers_outside_training_area=("area_outside_training_range", "sum"),
        )
        .reset_index()
    )
    ood["classification_basis"] = "glacier mean feature-outside fraction across all reconstructed years"

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

    glambie_comparison = glambie_period_comparison(regional)
    glambie_annual, glambie_metrics = glambie_annual_comparison(
        regional, args.bootstrap, args.block_years, args.seed
    )
    amplitude_series, amplitude_metrics = amplitude_sampling_diagnostics(reconstruction, regional)

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
    glambie_comparison.to_csv(PHYS_V2_GLAMBIE_COMPARISON, index=False)
    glambie_annual.to_csv(PHYS_V2_GLAMBIE_ANNUAL_SERIES, index=False)
    glambie_metrics.to_csv(PHYS_V2_GLAMBIE_ANNUAL_METRICS, index=False)
    amplitude_series.to_csv(PHYS_V2_AMPLITUDE_SERIES, index=False)
    amplitude_metrics.to_csv(PHYS_V2_AMPLITUDE_DIAGNOSTICS, index=False)

    print(pd.DataFrame(qc_rows).to_string(index=False))
    print("\nMalles & Marzeion comparison (full RGI02 only):")
    print(pd.DataFrame(metric_rows).to_string(index=False))
    print("\nZemp et al. comparison (shared WGMS information; not independent):")
    print(pd.DataFrame(zemp_metric_rows).to_string(index=False))
    print("\nGlaMBIE period-mean consistency (shared source information; not independent):")
    print(glambie_comparison.to_string(index=False))
    print("\nGlaMBIE hydrological-year consistency (shared sources; not independent):")
    print(glambie_metrics[[
        "model", "reference", "start_year", "end_year", "unit", "pearson_r", "rmse", "bias",
        "std_ratio_model_to_reference",
    ]].to_string(index=False))
    print(f"Saved QC -> {PHYS_V2_RECONSTRUCTION_QC_SUMMARY}")
    print(f"Saved OOD summary -> {PHYS_V2_RECONSTRUCTION_OOD_SUMMARY}")
    print(f"Saved regional series -> {PHYS_V2_RECONSTRUCTION_REGIONAL_CSV}")
    print(f"Saved external comparison -> {PHYS_V2_MALLES_COMPARISON}")
    print(f"Saved Zemp consistency comparison -> {PHYS_V2_ZEMP_COMPARISON}")
    print(f"Saved GlaMBIE period comparison -> {PHYS_V2_GLAMBIE_COMPARISON}")
    print(f"Saved GlaMBIE annual series -> {PHYS_V2_GLAMBIE_ANNUAL_SERIES}")
    print(f"Saved GlaMBIE annual metrics -> {PHYS_V2_GLAMBIE_ANNUAL_METRICS}")
    print(f"Saved amplitude/sampling series -> {PHYS_V2_AMPLITUDE_SERIES}")
    print(f"Saved amplitude/sampling diagnostics -> {PHYS_V2_AMPLITUDE_DIAGNOSTICS}")
    if not complete:
        raise RuntimeError("Reconstruction failed the expected glacier-year completeness checks.")


if __name__ == "__main__":
    main()
