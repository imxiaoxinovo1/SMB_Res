"""Extract monthly ERA5-Land climate variables for RGI02 training glaciers."""
from __future__ import annotations

import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (  # noqa: E402
    ERA5_MONTHLY_CSV,
    ERA5_RGI02_NC,
    MONTHLY_CLIMATE_VARS,
    RECON_YEAR_MAX,
    TERRAIN_CSV,
    TRAIN_YEAR_MIN,
)


TEMP_VARS = {"t2m", "skt", "d2m"}
ACCUM_VARS = {"tp", "sf", "smlt", "ssrd", "strd", "ssr", "str", "slhf", "sshf", "ro"}


def main() -> None:
    print("=== Step 03: ERA5 Monthly Extraction ===")

    glaciers = pd.read_csv(TERRAIN_CSV)
    lats = glaciers["latitude"].values
    lons = glaciers["longitude"].values
    glacier_ids = glaciers["glacier_id"].values
    print(f"Target glaciers: {len(glaciers)}")

    print(f"Opening ERA5-Land file: {ERA5_RGI02_NC}")
    ds = xr.open_dataset(ERA5_RGI02_NC)
    if "valid_time" in ds.dims:
        ds = ds.rename({"valid_time": "time"})
    if "expver" in ds.dims:
        print("Merging ERA5 expver versions.")
        ds = ds.sel(expver=1, drop=True).combine_first(ds.sel(expver=5, drop=True))

    ds = ds.sel(time=slice(f"{TRAIN_YEAR_MIN}-01", f"{RECON_YEAR_MAX}-12"))
    print(f"ERA5 time range: {str(ds.time.values[0])[:7]} to {str(ds.time.values[-1])[:7]}")

    lat_min = float(ds.latitude.min())
    lat_max = float(ds.latitude.max())
    lon_min = float(ds.longitude.min())
    lon_max = float(ds.longitude.max())
    lats_clamped = np.clip(lats, lat_min, lat_max)
    lons_clamped = np.clip(lons, lon_min, lon_max)

    n_clamped = int(np.sum((lats != lats_clamped) | (lons != lons_clamped)))
    if n_clamped:
        print(f"Clamped {n_clamped} glacier coordinates to ERA5 grid bounds.")

    lats_da = xr.DataArray(lats_clamped, dims="glacier")
    lons_da = xr.DataArray(lons_clamped, dims="glacier")

    times = pd.to_datetime(ds.time.values)
    records = [
        {"glacier_id": int(gid), "year": time.year, "month": time.month}
        for time in times
        for gid in glacier_ids
    ]
    print(f"Output rows: {len(records):,}")

    for i, var in enumerate(MONTHLY_CLIMATE_VARS, start=1):
        if var not in ds.data_vars:
            print(f"WARNING: {var} not found in ERA5 file; skipping.")
            continue

        interp = ds[var].interp(latitude=lats_da, longitude=lons_da, method="linear")
        values = interp.values

        # Linear interpolation can produce NaNs along coastal or domain edges.
        if np.isnan(values).any():
            nearest = ds[var].interp(latitude=lats_da, longitude=lons_da, method="nearest").values
            values = np.where(np.isnan(values), nearest, values)

        if var in TEMP_VARS:
            values = values - 273.15
        elif var in ACCUM_VARS:
            values = values * 1000.0

        n_glaciers = len(glacier_ids)
        for ti in range(len(times)):
            base = ti * n_glaciers
            for gi in range(n_glaciers):
                records[base + gi][var] = float(values[ti, gi])

        if i % 5 == 0 or i == len(MONTHLY_CLIMATE_VARS):
            print(f"Finished variable {i}/{len(MONTHLY_CLIMATE_VARS)}: {var}")

    out = pd.DataFrame(records)
    out.to_csv(ERA5_MONTHLY_CSV, index=False)
    print(f"Saved ERA5 monthly table: {ERA5_MONTHLY_CSV}")

    assert len(out) == len(times) * len(glacier_ids), "Unexpected ERA5 output row count."
    if "t2m" in out.columns:
        t2m_range = out["t2m"].agg(["min", "max"])
        assert t2m_range["min"] > -80 and t2m_range["max"] < 40, (
            f"t2m range looks wrong: {t2m_range}"
        )
        print(f"t2m range: {t2m_range['min']:.1f} to {t2m_range['max']:.1f} C")


if __name__ == "__main__":
    main()
