"""Build weak-supervision sequences for Hugonnet geodetic constraints.

The Hugonnet target is a glacier-level 2000-2019 mean specific mass-change
rate. This script creates annual GlacierFormer inputs for the same period and
stores glacier grouping indices so training can constrain multi-year means
instead of treating the target as an annual SMB label.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (  # noqa: E402
    CAL_SUMMER_MONTHS,
    CAL_WINTER_MONTHS,
    ERA5_RGI02_FULL_CSV,
    HUGONNET_DATA_DIR,
    HUGONNET_WEAK_LABELS_CSV,
    HUGONNET_WEAK_SEQUENCE_SUMMARY_CSV,
    HUGONNET_WEAK_SEQUENCES_NPZ,
    HUGONNET_WEAK_YEAR_MAX,
    HUGONNET_WEAK_YEAR_MIN,
    HYD_ABLAT_MONTHS,
    HYD_ACCUM_MONTHS,
    MONTHLY_CLIMATE_VARS,
    RGI02_ATTRIBUTES_CSV,
    SEASONAL_EXTRA_FEATURES,
    SEQUENCES_QC_NPZ,
    STATIC_FEATURES,
)


def safe_mean(frame: pd.DataFrame, col: str) -> float:
    return float(frame[col].mean()) if len(frame) > 0 and col in frame else np.nan


def safe_sum(frame: pd.DataFrame, col: str) -> float:
    return float(frame[col].sum()) if len(frame) > 0 and col in frame else np.nan


def seasonal_feats(glacier_era5: pd.DataFrame, year: int) -> np.ndarray:
    """Build the static seasonal feature tail in the training feature order."""
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


def main() -> None:
    print("=== Step 09: Build Hugonnet Weak Sequence Dataset ===")
    os.makedirs(HUGONNET_DATA_DIR, exist_ok=True)

    print("Loading Hugonnet weak-label table...")
    weak = pd.read_csv(HUGONNET_WEAK_LABELS_CSV)
    if weak.empty:
        raise RuntimeError("Hugonnet weak-label table is empty. Run step08 first.")

    print("Loading RGI7 terrain attributes for weak-label glaciers...")
    attrs = pd.read_csv(
        RGI02_ATTRIBUTES_CSV,
        usecols=[
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
        ],
    )
    attrs["aspect_sin"] = np.sin(np.deg2rad(attrs["aspect_deg"]))
    attrs["aspect_cos"] = np.cos(np.deg2rad(attrs["aspect_deg"]))
    terrain_cols = ["rgi_id", *STATIC_FEATURES]
    weak = weak.drop(columns=[col for col in STATIC_FEATURES if col in weak.columns], errors="ignore")
    weak = weak.merge(attrs[terrain_cols], on="rgi_id", how="left", validate="one_to_one")
    missing_static = [col for col in STATIC_FEATURES if col not in weak.columns]
    if missing_static:
        raise RuntimeError(f"Missing weak-label static columns: {missing_static}")

    wgms_terrain_qc = os.path.join(os.path.dirname(SEQUENCES_QC_NPZ), "training_glaciers_terrain_qc.csv")
    wgms_rgi_ids: set[str] = set()
    if os.path.exists(wgms_terrain_qc):
        wgms_rgi_ids = set(pd.read_csv(wgms_terrain_qc, usecols=["rgi_id"])["rgi_id"].dropna())
        n_before = len(weak)
        weak = weak[~weak["rgi_id"].isin(wgms_rgi_ids)].copy()
        print(f"Excluded WGMS glaciers from weak labels: {n_before - len(weak):,}")

    print("Loading WGMS normalization statistics...")
    norm = np.load(SEQUENCES_QC_NPZ, allow_pickle=True)
    dyn_mean = norm["dyn_mean"]
    dyn_std = norm["dyn_std"]
    sta_mean = norm["sta_mean"]
    sta_std = norm["sta_std"]
    sta_medians = norm["sta_medians"]

    print("Loading full-region ERA5 monthly features...")
    usecols = ["rgi_id", "year", "month", *MONTHLY_CLIMATE_VARS]
    weak_rgi_ids = set(weak["rgi_id"])
    chunks = []
    for chunk in pd.read_csv(ERA5_RGI02_FULL_CSV, usecols=usecols, chunksize=500_000):
        mask = (
            chunk["rgi_id"].isin(weak_rgi_ids)
            & (chunk["year"] >= HUGONNET_WEAK_YEAR_MIN - 1)
            & (chunk["year"] <= HUGONNET_WEAK_YEAR_MAX)
        )
        if mask.any():
            chunks.append(chunk.loc[mask].copy())
    if not chunks:
        raise RuntimeError("No full-region ERA5 rows match the Hugonnet weak-label glaciers.")
    era5 = pd.concat(chunks, ignore_index=True)
    era5_by_rgi = {rgi_id: grp for rgi_id, grp in era5.groupby("rgi_id", sort=False)}

    x_dyn_list: list[np.ndarray] = []
    x_sta_list: list[np.ndarray] = []
    rgi_id_list: list[str] = []
    rgi60_id_list: list[str] = []
    year_list: list[int] = []
    target_list: list[float] = []
    weight_list: list[float] = []
    weak_glacier_index_list: list[int] = []

    skipped_no_era5 = 0
    skipped_incomplete_years = 0
    skipped_dynamic_nan = 0
    skipped_static_nan = 0
    kept_glaciers = 0

    weak = weak.sort_values("rgi_id").reset_index(drop=True)
    kept_weak_rows: list[pd.Series] = []
    weak_index_map: dict[int, int] = {}

    for original_weak_idx, row in weak.iterrows():
        rgi_id = row["rgi_id"]
        glacier_era5 = era5_by_rgi.get(rgi_id)
        if glacier_era5 is None:
            skipped_no_era5 += 1
            continue

        sta_terrain = np.array(
            [row[col] if col in row.index else np.nan for col in STATIC_FEATURES],
            dtype=np.float32,
        )
        glacier_sample_count = 0

        for year in range(HUGONNET_WEAK_YEAR_MIN, HUGONNET_WEAK_YEAR_MAX + 1):
            year_era5 = glacier_era5[glacier_era5["year"] == year].sort_values("month")
            if len(year_era5) < 12:
                skipped_incomplete_years += 1
                continue

            dynamic = year_era5[MONTHLY_CLIMATE_VARS].values.astype(np.float32)
            if np.isnan(dynamic).any():
                skipped_dynamic_nan += 1
                continue

            static = np.concatenate([sta_terrain, seasonal_feats(glacier_era5, year)])
            if np.isnan(static).any():
                skipped_static_nan += int(np.isnan(static).sum())
                static = np.where(np.isnan(static), sta_medians, static)

            x_dyn_list.append(dynamic)
            x_sta_list.append(static)
            rgi_id_list.append(rgi_id)
            rgi60_id_list.append(row["rgi60_id"])
            year_list.append(year)
            target_list.append(float(row["hugonnet_dmdtda_mwe_yr"]))
            weight_list.append(float(row["weak_weight_norm"]))
            if original_weak_idx not in weak_index_map:
                weak_index_map[original_weak_idx] = len(kept_weak_rows)
                kept_weak_rows.append(row)
            weak_glacier_index_list.append(weak_index_map[original_weak_idx])
            glacier_sample_count += 1

        if glacier_sample_count > 0:
            kept_glaciers += 1

    if not x_dyn_list:
        raise RuntimeError("No Hugonnet weak sequence samples were built.")

    x_dyn = np.stack(x_dyn_list).astype(np.float32)
    x_sta = np.stack(x_sta_list).astype(np.float32)
    static_nan = np.isnan(x_sta)
    if static_nan.any():
        x_sta = np.where(static_nan, sta_medians, x_sta)

    x_dyn_norm = (x_dyn - dyn_mean) / dyn_std
    x_sta_norm = (x_sta - sta_mean) / sta_std

    if np.isnan(x_dyn_norm).any():
        raise RuntimeError("X_dyn contains NaNs after normalization.")
    if np.isnan(x_sta_norm).any():
        raise RuntimeError("X_sta contains NaNs after normalization.")

    rgi_ids = np.array(rgi_id_list)
    years = np.array(year_list, dtype=np.int32)
    targets = np.array(target_list, dtype=np.float32)
    weights = np.array(weight_list, dtype=np.float32)
    weak_glacier_index = np.array(weak_glacier_index_list, dtype=np.int32)
    kept_weak = pd.DataFrame(kept_weak_rows).reset_index(drop=True)

    np.savez_compressed(
        HUGONNET_WEAK_SEQUENCES_NPZ,
        X_dyn=x_dyn_norm,
        X_sta=x_sta_norm,
        y_geodetic=targets,
        w=weights,
        rgi_id=rgi_ids,
        rgi60_id=np.array(rgi60_id_list),
        year=years,
        weak_glacier_index=weak_glacier_index,
        weak_target_by_glacier=kept_weak["hugonnet_dmdtda_mwe_yr"].values.astype(np.float32),
        weak_weight_by_glacier=kept_weak["weak_weight_norm"].values.astype(np.float32),
        weak_rgi_id_by_glacier=kept_weak["rgi_id"].values,
        weak_rgi60_id_by_glacier=kept_weak["rgi60_id"].values,
        year_min=np.array([HUGONNET_WEAK_YEAR_MIN], dtype=np.int32),
        year_max=np.array([HUGONNET_WEAK_YEAR_MAX], dtype=np.int32),
        source_label_csv=np.array([HUGONNET_WEAK_LABELS_CSV]),
    )

    unique_weak_glaciers = len(np.unique(rgi_ids))
    full_window = pd.Series(rgi_ids).value_counts().eq(
        HUGONNET_WEAK_YEAR_MAX - HUGONNET_WEAK_YEAR_MIN + 1
    ).sum()
    summary = pd.DataFrame(
        [
            {"metric": "weak_label_glaciers", "value": len(weak)},
            {"metric": "kept_glaciers_with_era5", "value": kept_glaciers},
            {"metric": "unique_weak_sequence_glaciers", "value": unique_weak_glaciers},
            {"metric": "full_20yr_window_glaciers", "value": int(full_window)},
            {"metric": "weak_sequence_samples", "value": len(targets)},
            {"metric": "year_min", "value": HUGONNET_WEAK_YEAR_MIN},
            {"metric": "year_max", "value": HUGONNET_WEAK_YEAR_MAX},
            {"metric": "skipped_no_era5_glaciers", "value": skipped_no_era5},
            {"metric": "skipped_incomplete_glacier_years", "value": skipped_incomplete_years},
            {"metric": "skipped_dynamic_nan_glacier_years", "value": skipped_dynamic_nan},
            {"metric": "filled_static_nan_values", "value": skipped_static_nan},
            {"metric": "target_mean_mwe_yr", "value": float(targets.mean())},
            {"metric": "target_median_mwe_yr", "value": float(np.median(targets))},
            {"metric": "target_std_mwe_yr", "value": float(targets.std())},
        ]
    )
    summary.to_csv(HUGONNET_WEAK_SEQUENCE_SUMMARY_CSV, index=False)

    print(f"Weak-label glaciers: {len(weak):,}")
    print(f"Kept glaciers with ERA5: {kept_glaciers:,}")
    print(f"Unique weak-sequence glaciers: {unique_weak_glaciers:,}")
    print(f"Weak sequence samples: {len(targets):,}")
    print(f"Full 20-year windows: {int(full_window):,}")
    print(f"Skipped no-ERA5 glaciers: {skipped_no_era5:,}")
    print(f"Skipped incomplete glacier-years: {skipped_incomplete_years:,}")
    print(f"Skipped dynamic-NaN glacier-years: {skipped_dynamic_nan:,}")
    print(f"Filled static NaN values: {skipped_static_nan:,}")
    print(
        "Target mean/median/std: "
        f"{targets.mean():.3f} / {np.median(targets):.3f} / {targets.std():.3f} m w.e. yr-1"
    )
    print(f"Saved weak sequences -> {HUGONNET_WEAK_SEQUENCES_NPZ}")
    print(f"Saved summary -> {HUGONNET_WEAK_SEQUENCE_SUMMARY_CSV}")


if __name__ == "__main__":
    main()
