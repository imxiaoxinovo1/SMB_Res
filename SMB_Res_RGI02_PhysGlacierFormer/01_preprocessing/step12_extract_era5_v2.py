"""Extract physically consistent ERA5-Land monthly data for v2 training."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (  # noqa: E402
    ERA5_RGI02_NC,
    MONTHLY_CLIMATE_VARS,
    PHYS_V2_DATA_DIR,
    PHYS_V2_ERA5_CSV,
    PHYS_V2_TERRAIN_CSV,
    RECON_YEAR_MAX,
    TRAIN_YEAR_MIN,
)


TEMPERATURE_VARS = {"t2m", "skt", "d2m"}
HYDROLOGICAL_ACCUMULATIONS = {"tp", "sf", "smlt", "ro"}
ENERGY_ACCUMULATIONS = {"ssrd", "strd", "ssr", "str", "slhf", "sshf"}


def main() -> None:
    print("=== Step 12: Physically Consistent ERA5-Land Extraction ===")
    os.makedirs(PHYS_V2_DATA_DIR, exist_ok=True)
    glaciers = pd.read_csv(PHYS_V2_TERRAIN_CSV)
    glaciers = glaciers[glaciers["mapping_qc_pass"]].copy()

    ds = xr.open_dataset(ERA5_RGI02_NC)
    if "valid_time" in ds.dims:
        ds = ds.rename({"valid_time": "time"})
    if "expver" in ds.dims:
        ds = ds.sel(expver=1, drop=True).combine_first(ds.sel(expver=5, drop=True))
    ds = ds.sel(time=slice(f"{TRAIN_YEAR_MIN}-01", f"{RECON_YEAR_MAX}-12"))

    lat_min, lat_max = float(ds.latitude.min()), float(ds.latitude.max())
    lon_min, lon_max = float(ds.longitude.min()), float(ds.longitude.max())
    outside = (
        (glaciers["latitude"] < lat_min)
        | (glaciers["latitude"] > lat_max)
        | (glaciers["longitude"] < lon_min)
        | (glaciers["longitude"] > lon_max)
    )
    if outside.any():
        bad = glaciers.loc[outside, ["glacier_id", "name", "latitude", "longitude"]]
        qc_path = os.path.join(PHYS_V2_DATA_DIR, "era5_domain_exclusions_v2.csv")
        bad.to_csv(qc_path, index=False)
        print("WARNING: excluding training glaciers outside the downloaded ERA5 domain:")
        print(bad.to_string(index=False))
        print(f"Domain exclusions saved -> {qc_path}")
        glaciers = glaciers.loc[~outside].copy()

    times = pd.to_datetime(ds.time.values)
    lats = xr.DataArray(glaciers["latitude"].to_numpy(), dims="glacier")
    lons = xr.DataArray(glaciers["longitude"].to_numpy(), dims="glacier")
    glacier_ids = glaciers["glacier_id"].astype(int).to_numpy()

    records = [
        {"glacier_id": int(gid), "year": int(t.year), "month": int(t.month)}
        for t in times
        for gid in glacier_ids
    ]
    output = pd.DataFrame(records)
    days = times.days_in_month.to_numpy(dtype=np.float64)[:, None]

    extraction_variables = [*MONTHLY_CLIMATE_VARS, "sp"]
    for var in extraction_variables:
        if var not in ds:
            raise KeyError(f"ERA5 variable missing: {var}")
        values = ds[var].interp(latitude=lats, longitude=lons, method="linear").values
        if np.isnan(values).any():
            nearest = ds[var].interp(latitude=lats, longitude=lons, method="nearest").values
            values = np.where(np.isnan(values), nearest, values)

        if var in TEMPERATURE_VARS:
            values = values - 273.15
        elif var in HYDROLOGICAL_ACCUMULATIONS:
            # stream=moda is m w.e. per day; integrate to monthly mm w.e.
            values = values * 1000.0 * days
        elif var in ENERGY_ACCUMULATIONS:
            # stream=moda energy accumulations are J m-2 per day.
            values = values / 86400.0

        output[var] = values.reshape(-1)
        print(f"  extracted {var}")

    output.to_csv(PHYS_V2_ERA5_CSV, index=False)
    print(f"Rows: {len(output):,}; glaciers: {len(glacier_ids)}")
    print(f"Saved -> {PHYS_V2_ERA5_CSV}")


if __name__ == "__main__":
    main()
