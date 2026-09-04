"""Build one annual tabular feature row per glacier-year."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (  # noqa: E402
    CAL_SUMMER_MONTHS,
    CAL_WINTER_MONTHS,
    ERA5_MONTHLY_CSV,
    HYD_ABLAT_MONTHS,
    MASSBAL_RGI02_CSV,
    MONTHLY_CLIMATE_VARS,
    STATIC_FEATURES,
    TABULAR_CSV,
    TERRAIN_CSV,
)


TEMP_LIKE = {"t2m", "skt", "d2m", "sd", "asn"}


def aggregate_period(frame: pd.DataFrame, prefix: str) -> dict[str, float]:
    features = {}
    if frame.empty:
        return features
    for var in MONTHLY_CLIMATE_VARS:
        if var not in frame.columns:
            continue
        values = frame[var].values
        if var in TEMP_LIKE:
            features[f"{prefix}_{var}_mean"] = float(np.nanmean(values))
        else:
            features[f"{prefix}_{var}_sum"] = float(np.nansum(values))
    return features


def main() -> None:
    print("=== Step 04: Build Tabular Dataset ===")

    era5 = pd.read_csv(ERA5_MONTHLY_CSV)
    terrain = pd.read_csv(TERRAIN_CSV)
    mass_balance = pd.read_csv(MASSBAL_RGI02_CSV)

    print(f"ERA5 monthly rows: {len(era5):,}")
    terrain_cols = [col for col in STATIC_FEATURES if col in terrain.columns]
    terrain_idx = terrain.set_index("glacier_id")
    mb_idx = mass_balance.set_index(["glacier_id", "year"])

    rows = []
    for gid, glacier_era5 in era5.groupby("glacier_id"):
        if gid not in terrain_idx.index:
            continue
        static = terrain_idx.loc[gid, terrain_cols].to_dict()

        for year, year_era5 in glacier_era5.groupby("year"):
            if len(year_era5) < 12:
                continue
            row = {"glacier_id": gid, "year": year}
            row.update(aggregate_period(year_era5, "ann"))
            row.update(
                aggregate_period(
                    year_era5[year_era5["month"].isin(CAL_SUMMER_MONTHS)],
                    "cal_summer",
                )
            )
            row.update(
                aggregate_period(
                    year_era5[year_era5["month"].isin(CAL_WINTER_MONTHS)],
                    "cal_winter",
                )
            )
            row.update(
                aggregate_period(
                    year_era5[year_era5["month"].isin(HYD_ABLAT_MONTHS)],
                    "hyd_ablat",
                )
            )

            previous = glacier_era5[glacier_era5["year"] == year - 1]
            oct_dec = previous[previous["month"].isin([10, 11, 12])]
            jan_apr = year_era5[year_era5["month"].isin([1, 2, 3, 4])]
            hyd_accum = pd.concat([oct_dec, jan_apr], ignore_index=True)
            row.update(aggregate_period(hyd_accum, "hyd_accum"))

            row.update(static)
            key = (gid, year)
            row["annual_balance_m"] = (
                float(mb_idx.loc[key, "annual_balance"]) if key in mb_idx.index else np.nan
            )
            rows.append(row)

    out = pd.DataFrame(rows)
    print(f"Feature rows: {len(out):,}")
    print(f"Labeled rows: {out['annual_balance_m'].notna().sum():,}")

    labeled = out["annual_balance_m"].notna()
    if labeled.any():
        balance_range = out.loc[labeled, "annual_balance_m"].agg(["min", "max"])
        assert balance_range["min"] > -10 and balance_range["max"] < 10, (
            f"annual_balance_m range looks wrong: {balance_range}"
        )
        print(
            f"annual_balance_m range: {balance_range['min']:.2f} "
            f"to {balance_range['max']:.2f} m w.e."
        )

    out.to_csv(TABULAR_CSV, index=False)
    print(f"Saved tabular dataset: {TABULAR_CSV}")


if __name__ == "__main__":
    main()
