"""Build normalized monthly-sequence tensors for GlacierFormer training."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (  # noqa: E402
    ERA5_MONTHLY_CSV,
    MASSBAL_RGI02_CSV,
    MONTHLY_CLIMATE_VARS,
    N_STATIC,
    SEASONAL_EXTRA_FEATURES,
    SEQUENCES_NPZ,
    STATIC_FEATURES,
    TABULAR_CSV,
    TERRAIN_CSV,
    TRAIN_YEAR_MAX,
)


def main() -> None:
    print("=== Step 05: Build Sequence Dataset ===")

    era5 = pd.read_csv(ERA5_MONTHLY_CSV)
    terrain = pd.read_csv(TERRAIN_CSV)
    mass_balance = pd.read_csv(MASSBAL_RGI02_CSV)
    tabular = pd.read_csv(TABULAR_CSV)

    missing_static = [col for col in STATIC_FEATURES if col not in terrain.columns]
    missing_seasonal = [col for col in SEASONAL_EXTRA_FEATURES if col not in tabular.columns]
    assert not missing_static, f"Missing static columns: {missing_static}"
    assert not missing_seasonal, f"Missing seasonal columns: {missing_seasonal}"

    terrain_idx = terrain.set_index("glacier_id")
    mb_idx = mass_balance.set_index(["glacier_id", "year"])
    tab_idx = tabular.set_index(["glacier_id", "year"])

    x_dyn_list = []
    x_sta_list = []
    y_list = []
    gid_list = []
    year_list = []

    for gid, glacier_era5 in era5.groupby("glacier_id"):
        if gid not in terrain_idx.index:
            continue
        static_terrain = terrain_idx.loc[gid, STATIC_FEATURES].values.astype(float)

        for year, year_era5 in glacier_era5.groupby("year"):
            year_era5 = year_era5.sort_values("month")
            if len(year_era5) < 12:
                continue

            dynamic = year_era5[MONTHLY_CLIMATE_VARS].values.astype(float)
            key = (gid, year)
            if key in tab_idx.index:
                seasonal = tab_idx.loc[key, SEASONAL_EXTRA_FEATURES].values.astype(float)
            else:
                seasonal = np.full(len(SEASONAL_EXTRA_FEATURES), np.nan)
            static = np.concatenate([static_terrain, seasonal])

            target = float(mb_idx.loc[key, "annual_balance"]) if key in mb_idx.index else np.nan

            x_dyn_list.append(dynamic)
            x_sta_list.append(static)
            y_list.append(target)
            gid_list.append(gid)
            year_list.append(year)

    x_dyn = np.array(x_dyn_list, dtype=np.float32)
    x_sta = np.array(x_sta_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    glacier_ids = np.array(gid_list)
    years = np.array(year_list)

    print(f"Total samples: {len(y):,}")
    print(f"Labeled samples: {int((~np.isnan(y)).sum()):,}")
    print(f"Raw X_dyn: {x_dyn.shape}, raw X_sta: {x_sta.shape}")

    train_mask = years <= TRAIN_YEAR_MAX
    static_medians = np.nanmedian(x_sta[train_mask], axis=0)
    static_nan_mask = np.isnan(x_sta)
    if static_nan_mask.any():
        x_sta = np.where(static_nan_mask, static_medians, x_sta)
        print(f"Filled static NaNs with training medians: {int(static_nan_mask.sum())}")

    dyn_mean = np.nanmean(x_dyn[train_mask], axis=(0, 1), keepdims=True)
    dyn_std = np.nanstd(x_dyn[train_mask], axis=(0, 1), keepdims=True) + 1e-8
    sta_mean = np.nanmean(x_sta[train_mask], axis=0, keepdims=True)
    sta_std = np.nanstd(x_sta[train_mask], axis=0, keepdims=True) + 1e-8

    x_dyn_norm = (x_dyn - dyn_mean) / dyn_std
    x_sta_norm = (x_sta - sta_mean) / sta_std

    assert x_dyn_norm.shape[1:] == (12, len(MONTHLY_CLIMATE_VARS))
    assert x_sta_norm.shape[1:] == (N_STATIC,)
    assert len(y) == len(x_dyn_norm) == len(x_sta_norm)

    labeled_count = int((~np.isnan(y)).sum())
    assert labeled_count > 100, f"Too few labeled samples: {labeled_count}"
    assert not np.isnan(x_dyn_norm).any(), "X_dyn contains NaNs after normalization."
    assert not np.isnan(x_sta_norm).any(), "X_sta contains NaNs after normalization."

    np.savez_compressed(
        SEQUENCES_NPZ,
        X_dyn=x_dyn_norm,
        X_sta=x_sta_norm,
        y=y,
        glacier_ids=glacier_ids,
        years=years,
        dyn_mean=dyn_mean,
        dyn_std=dyn_std,
        sta_mean=sta_mean,
        sta_std=sta_std,
        sta_medians=static_medians,
    )
    print(f"Saved sequence dataset: {SEQUENCES_NPZ}")


if __name__ == "__main__":
    main()
