"""Build multi-period Hugonnet geodetic constraints for GlacierFormer.

Each Hugonnet row constrains the mean model prediction over its period, e.g.
mean(SMB_2010..SMB_2019) ~= dmdtda_2010_2020. Short one-year periods are
excluded by configuration because their uncertainties are too large for stable
weak supervision.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from config import (  # noqa: E402
    CAL_SUMMER_MONTHS,
    CAL_WINTER_MONTHS,
    ERA5_RGI02_FULL_CSV,
    HUGONNET_DATA_DIR,
    HUGONNET_MAX_AREA_RATIO_RGI7_TO_RGI60,
    HUGONNET_MAX_MAP_DIST_DEG,
    HUGONNET_MIN_AREA_RATIO_RGI7_TO_RGI60,
    HUGONNET_MIN_PERC_AREA_MEAS,
    HUGONNET_MIN_PERC_AREA_RES,
    HUGONNET_MIN_RGI60_OVERLAP_FRACTION,
    HUGONNET_MIN_RGI7_OVERLAP_FRACTION,
    HUGONNET_MIN_VALID_OBS_PY,
    HUGONNET_MULTIPERIOD_LABELS_CSV,
    HUGONNET_MULTIPERIOD_MAX_ERR_DMDTDA,
    HUGONNET_MULTIPERIOD_PERIODS,
    HUGONNET_MULTIPERIOD_SEQUENCE_SUMMARY_CSV,
    HUGONNET_MULTIPERIOD_SEQUENCES_NPZ,
    HUGONNET_RGI02_RATES,
    HYD_ABLAT_MONTHS,
    MONTHLY_CLIMATE_VARS,
    RGI02_ATTRIBUTES_CSV,
    SEASONAL_EXTRA_FEATURES,
    SEQUENCES_QC_NPZ,
    STATIC_FEATURES,
)
from step08_build_hugonnet_weak_labels import build_rgi7_to_rgi60_map  # noqa: E402


def parse_period(period: str) -> tuple[int, int]:
    start_s, end_s = period.split("_")
    start_year = datetime.fromisoformat(start_s).year
    end_year = datetime.fromisoformat(end_s).year
    return start_year, end_year


def safe_mean(frame: pd.DataFrame, col: str) -> float:
    return float(frame[col].mean()) if len(frame) > 0 and col in frame else np.nan


def safe_sum(frame: pd.DataFrame, col: str) -> float:
    return float(frame[col].sum()) if len(frame) > 0 and col in frame else np.nan


def seasonal_feats(glacier_era5: pd.DataFrame, year: int) -> np.ndarray:
    cur = glacier_era5[glacier_era5["year"] == year]
    prev = glacier_era5[glacier_era5["year"] == year - 1]
    cal_summer = cur[cur["month"].isin(CAL_SUMMER_MONTHS)]
    cal_winter = cur[cur["month"].isin(CAL_WINTER_MONTHS)]
    hyd_ablat = cur[cur["month"].isin(HYD_ABLAT_MONTHS)]
    hyd_accum = pd.concat(
        [
            prev[prev["month"].isin([10, 11, 12])],
            cur[cur["month"].isin([1, 2, 3, 4])],
        ],
        ignore_index=True,
    )
    values = np.array(
        [
            safe_mean(cal_summer, "t2m"),
            safe_sum(cal_winter, "tp"),
            safe_mean(hyd_ablat, "t2m"),
            safe_sum(hyd_accum, "sf"),
            safe_sum(hyd_ablat, "smlt"),
            safe_mean(cur, "t2m"),
            safe_sum(cur, "tp"),
            safe_sum(cal_summer, "ssrd"),
        ],
        dtype=np.float32,
    )
    assert len(values) == len(SEASONAL_EXTRA_FEATURES)
    return values


def build_multiperiod_labels() -> pd.DataFrame:
    attrs = pd.read_csv(RGI02_ATTRIBUTES_CSV)
    base_cols = [
        "rgi_id",
        "glims_id",
        "o2region",
        "cenlon",
        "cenlat",
        "area_km2",
        "term_type",
        "surge_type",
    ]
    attrs_base = attrs[base_cols].copy()
    id_map = build_rgi7_to_rgi60_map(attrs_base)
    attrs_base = attrs_base.merge(id_map, on="rgi_id", how="left", validate="one_to_one")

    overlap_available = (
        attrs_base["rgi7_area_fraction"].notna()
        & attrs_base["rgi6_area_fraction"].notna()
    )
    overlap_qc = (
        overlap_available
        & (attrs_base["rgi7_area_fraction"] >= HUGONNET_MIN_RGI7_OVERLAP_FRACTION)
        & (attrs_base["rgi6_area_fraction"] >= HUGONNET_MIN_RGI60_OVERLAP_FRACTION)
    )
    glims_qc = (
        ~overlap_available
        & attrs_base["map_dist_deg"].notna()
        & (attrs_base["map_dist_deg"] <= HUGONNET_MAX_MAP_DIST_DEG)
        & attrs_base["map_area_ratio_rgi7_to_rgi60"].between(
            HUGONNET_MIN_AREA_RATIO_RGI7_TO_RGI60,
            HUGONNET_MAX_AREA_RATIO_RGI7_TO_RGI60,
        )
    )
    attrs_base["map_qc_pass"] = attrs_base["rgi60_id"].notna() & (overlap_qc | glims_qc)

    attrs_full = attrs[
        [
            "rgi_id",
            "slope_deg",
            "aspect_deg",
            "zmin_m",
            "zmax_m",
            "zmean_m",
            "zmed_m",
            "area_km2",
            "lmax_m",
            "cenlat",
        ]
    ].copy()
    attrs_full["aspect_sin"] = np.sin(np.deg2rad(attrs_full["aspect_deg"]))
    attrs_full["aspect_cos"] = np.cos(np.deg2rad(attrs_full["aspect_deg"]))
    attrs_full = attrs_full[["rgi_id", *STATIC_FEATURES]]

    usecols = [
        "rgiid",
        "period",
        "area",
        "dhdt",
        "err_dhdt",
        "dmdtda",
        "err_dmdtda",
        "perc_area_meas",
        "perc_area_res",
        "valid_obs",
        "valid_obs_py",
        "reg",
    ]
    rates = pd.read_csv(HUGONNET_RGI02_RATES, usecols=usecols)
    rates = rates[rates["period"].isin(HUGONNET_MULTIPERIOD_PERIODS)].copy()
    for col in [col for col in usecols if col not in ("rgiid", "period")]:
        rates[col] = pd.to_numeric(rates[col], errors="coerce")
    rates = rates.rename(
        columns={
            "rgiid": "rgi60_id",
            "area": "hugonnet_area_m2",
            "dhdt": "hugonnet_dhdt_m_yr",
            "err_dhdt": "hugonnet_err_dhdt_m_yr",
            "dmdtda": "hugonnet_dmdtda_mwe_yr",
            "err_dmdtda": "hugonnet_err_dmdtda_mwe_yr",
        }
    )

    drop_static_overlap = [col for col in STATIC_FEATURES if col in attrs_base.columns]
    merged = attrs_base.drop(columns=drop_static_overlap).merge(
        rates,
        on="rgi60_id",
        how="inner",
    )
    merged = merged.merge(attrs_full, on="rgi_id", how="left", validate="many_to_one")
    merged["period_start_year"], merged["period_end_year"] = zip(
        *merged["period"].map(parse_period)
    )
    merged["period_n_years"] = merged["period_end_year"] - merged["period_start_year"]
    merged["qc_pass"] = (
        merged["map_qc_pass"]
        & merged["hugonnet_dmdtda_mwe_yr"].notna()
        & merged["hugonnet_err_dmdtda_mwe_yr"].notna()
        & (merged["hugonnet_err_dmdtda_mwe_yr"] <= HUGONNET_MULTIPERIOD_MAX_ERR_DMDTDA)
        & (merged["perc_area_meas"] >= HUGONNET_MIN_PERC_AREA_MEAS)
        & (merged["perc_area_res"] >= HUGONNET_MIN_PERC_AREA_RES)
        & (merged["valid_obs_py"] >= HUGONNET_MIN_VALID_OBS_PY)
    )
    labels = merged.loc[merged["qc_pass"]].copy()

    eps = 1e-6
    labels["weak_weight_inv_var"] = 1.0 / (
        labels["hugonnet_err_dmdtda_mwe_yr"].clip(lower=eps) ** 2
    )
    median_weight = float(labels["weak_weight_inv_var"].median())
    labels["weak_weight_norm"] = (labels["weak_weight_inv_var"] / median_weight).clip(0.05, 20.0)
    labels = labels.sort_values(["rgi_id", "period"]).reset_index(drop=True)
    labels["constraint_id"] = np.arange(len(labels), dtype=np.int32)
    labels.to_csv(HUGONNET_MULTIPERIOD_LABELS_CSV, index=False)
    return labels


def main() -> None:
    print("=== Step 10: Build Hugonnet Multi-period Sequence Dataset ===")
    os.makedirs(HUGONNET_DATA_DIR, exist_ok=True)

    print("Building multi-period Hugonnet labels...")
    labels = build_multiperiod_labels()
    if labels.empty:
        raise RuntimeError("No QC-pass Hugonnet multi-period labels.")

    wgms_terrain_qc = os.path.join(os.path.dirname(SEQUENCES_QC_NPZ), "training_glaciers_terrain_qc.csv")
    if os.path.exists(wgms_terrain_qc):
        wgms_rgi_ids = set(pd.read_csv(wgms_terrain_qc, usecols=["rgi_id"])["rgi_id"].dropna())
        n_before = len(labels)
        labels = labels[~labels["rgi_id"].isin(wgms_rgi_ids)].reset_index(drop=True)
        labels["constraint_id"] = np.arange(len(labels), dtype=np.int32)
        print(f"Excluded WGMS constraints: {n_before - len(labels):,}")

    norm = np.load(SEQUENCES_QC_NPZ, allow_pickle=True)
    dyn_mean = norm["dyn_mean"]
    dyn_std = norm["dyn_std"]
    sta_mean = norm["sta_mean"]
    sta_std = norm["sta_std"]
    sta_medians = norm["sta_medians"]

    min_year = int(labels["period_start_year"].min()) - 1
    max_year = int(labels["period_end_year"].max()) - 1
    weak_rgi_ids = set(labels["rgi_id"])
    usecols = ["rgi_id", "year", "month", *MONTHLY_CLIMATE_VARS]
    print(f"Loading ERA5 rows for {len(weak_rgi_ids):,} glaciers, years {min_year}-{max_year}...")
    chunks = []
    for chunk in pd.read_csv(ERA5_RGI02_FULL_CSV, usecols=usecols, chunksize=500_000):
        mask = (
            chunk["rgi_id"].isin(weak_rgi_ids)
            & (chunk["year"] >= min_year)
            & (chunk["year"] <= max_year)
        )
        if mask.any():
            chunks.append(chunk.loc[mask].copy())
    if not chunks:
        raise RuntimeError("No ERA5 rows match multi-period Hugonnet labels.")
    era5 = pd.concat(chunks, ignore_index=True)
    era5_by_rgi = {rgi_id: grp for rgi_id, grp in era5.groupby("rgi_id", sort=False)}

    x_dyn_list = []
    x_sta_list = []
    constraint_index = []
    rgi_id_list = []
    rgi60_id_list = []
    year_list = []
    skipped_no_era5 = 0
    skipped_incomplete = 0
    skipped_dyn_nan = 0
    filled_static_nan = 0
    kept_constraints = []
    new_constraint_id = {}

    for _, row in labels.iterrows():
        rgi_id = row["rgi_id"]
        glacier_era5 = era5_by_rgi.get(rgi_id)
        if glacier_era5 is None:
            skipped_no_era5 += 1
            continue

        years = range(int(row["period_start_year"]), int(row["period_end_year"]))
        local_indices = []
        sta_terrain = np.array([row[col] for col in STATIC_FEATURES], dtype=np.float32)
        for year in years:
            y_cur = glacier_era5[glacier_era5["year"] == year].sort_values("month")
            if len(y_cur) < 12:
                skipped_incomplete += 1
                continue
            dyn = y_cur[MONTHLY_CLIMATE_VARS].values.astype(np.float32)
            if np.isnan(dyn).any():
                skipped_dyn_nan += 1
                continue
            sta = np.concatenate([sta_terrain, seasonal_feats(glacier_era5, year)])
            if np.isnan(sta).any():
                filled_static_nan += int(np.isnan(sta).sum())
                sta = np.where(np.isnan(sta), sta_medians, sta)

            local_indices.append(len(x_dyn_list))
            x_dyn_list.append(dyn)
            x_sta_list.append(sta)
            rgi_id_list.append(rgi_id)
            rgi60_id_list.append(row["rgi60_id"])
            year_list.append(year)

        if len(local_indices) == int(row["period_n_years"]):
            cid = int(row["constraint_id"])
            new_constraint_id[cid] = len(kept_constraints)
            kept_constraints.append(row)
            constraint_index.extend([cid] * len(local_indices))
        else:
            for _ in local_indices:
                x_dyn_list.pop()
                x_sta_list.pop()
                rgi_id_list.pop()
                rgi60_id_list.pop()
                year_list.pop()

    if not x_dyn_list:
        raise RuntimeError("No valid multi-period sequence samples built.")

    constraint_index = np.array([new_constraint_id[int(cid)] for cid in constraint_index], dtype=np.int32)
    kept = pd.DataFrame(kept_constraints).reset_index(drop=True)
    x_dyn = np.stack(x_dyn_list).astype(np.float32)
    x_sta = np.stack(x_sta_list).astype(np.float32)
    x_dyn_norm = (x_dyn - dyn_mean) / dyn_std
    x_sta_norm = (x_sta - sta_mean) / sta_std
    if np.isnan(x_dyn_norm).any() or np.isnan(x_sta_norm).any():
        raise RuntimeError("NaNs remain after normalization.")

    np.savez_compressed(
        HUGONNET_MULTIPERIOD_SEQUENCES_NPZ,
        X_dyn=x_dyn_norm,
        X_sta=x_sta_norm,
        constraint_index=constraint_index,
        rgi_id=np.array(rgi_id_list),
        rgi60_id=np.array(rgi60_id_list),
        year=np.array(year_list, dtype=np.int32),
        target_by_constraint=kept["hugonnet_dmdtda_mwe_yr"].values.astype(np.float32),
        weight_by_constraint=kept["weak_weight_norm"].values.astype(np.float32),
        period_by_constraint=kept["period"].values,
        rgi_id_by_constraint=kept["rgi_id"].values,
        rgi60_id_by_constraint=kept["rgi60_id"].values,
        start_year_by_constraint=kept["period_start_year"].values.astype(np.int32),
        end_year_by_constraint=kept["period_end_year"].values.astype(np.int32),
    )

    summary = pd.DataFrame(
        [
            {"metric": "configured_periods", "value": len(HUGONNET_MULTIPERIOD_PERIODS)},
            {"metric": "qc_constraints_after_wgms_exclusion", "value": len(kept)},
            {"metric": "unique_glaciers", "value": kept["rgi_id"].nunique()},
            {"metric": "sequence_samples", "value": len(x_dyn_list)},
            {"metric": "skipped_no_era5_constraints", "value": skipped_no_era5},
            {"metric": "skipped_incomplete_years", "value": skipped_incomplete},
            {"metric": "skipped_dynamic_nan_years", "value": skipped_dyn_nan},
            {"metric": "filled_static_nan_values", "value": filled_static_nan},
            {"metric": "target_mean_mwe_yr", "value": float(kept["hugonnet_dmdtda_mwe_yr"].mean())},
            {"metric": "target_median_mwe_yr", "value": float(kept["hugonnet_dmdtda_mwe_yr"].median())},
            {"metric": "target_std_mwe_yr", "value": float(kept["hugonnet_dmdtda_mwe_yr"].std())},
        ]
    )
    summary.to_csv(HUGONNET_MULTIPERIOD_SEQUENCE_SUMMARY_CSV, index=False)

    print(f"QC constraints: {len(kept):,}")
    print(f"Unique glaciers: {kept['rgi_id'].nunique():,}")
    print(f"Sequence samples: {len(x_dyn_list):,}")
    print(f"Saved -> {HUGONNET_MULTIPERIOD_SEQUENCES_NPZ}")
    print(f"Summary -> {HUGONNET_MULTIPERIOD_SEQUENCE_SUMMARY_CSV}")


if __name__ == "__main__":
    main()
