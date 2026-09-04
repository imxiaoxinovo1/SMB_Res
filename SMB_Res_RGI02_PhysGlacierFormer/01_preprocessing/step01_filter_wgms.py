"""Filter WGMS annual SMB records for RGI02 Western Canada and USA."""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (  # noqa: E402
    GLACIER_CSV,
    MASSBAL_CSV,
    MASSBAL_RGI02_CSV,
    RECON_YEAR_MAX,
    TRAINING_GLACIERS_CSV,
    TRAIN_YEAR_MIN,
)


def read_csv_fallback(path: str) -> pd.DataFrame:
    """Read WGMS CSV files robustly across local encoding variants."""
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1", low_memory=False)


def main() -> None:
    print("=== Step 01: WGMS RGI02 Filter ===")

    glaciers = read_csv_fallback(GLACIER_CSV)
    rgi02_glaciers = glaciers[glaciers["gtng_region"] == "02_western_canada_usa"].copy()
    print(f"WGMS glacier rows in RGI02: {len(rgi02_glaciers):,}")

    mass_balance = read_csv_fallback(MASSBAL_CSV)
    print(f"WGMS mass_balance rows: {len(mass_balance):,}")

    rgi02_ids = set(rgi02_glaciers["id"].values)
    mb_rgi02 = mass_balance[
        mass_balance["glacier_id"].isin(rgi02_ids)
        & mass_balance["annual_balance"].notna()
        & (mass_balance["year"] >= TRAIN_YEAR_MIN)
        & (mass_balance["year"] <= RECON_YEAR_MAX)
    ].copy()

    mb_rgi02["annual_balance"] = pd.to_numeric(mb_rgi02["annual_balance"], errors="coerce")
    mb_rgi02 = mb_rgi02.dropna(subset=["annual_balance"])

    print(f"RGI02 annual SMB records: {len(mb_rgi02):,}")
    print(f"Unique observed glaciers: {mb_rgi02['glacier_id'].nunique()}")

    balance_mean = mb_rgi02["annual_balance"].mean()
    assert abs(balance_mean) < 10, (
        "annual_balance unit looks wrong. Expected m w.e.; "
        f"got mean={balance_mean:.3f}"
    )
    print(f"annual_balance mean: {balance_mean:.3f} m w.e.")

    static = rgi02_glaciers[["id", "names", "latitude", "longitude", "gtng_region"]].copy()
    static = static.rename(columns={"id": "glacier_id", "names": "name"})

    stats = (
        mb_rgi02.groupby("glacier_id")
        .agg(
            n_years=("year", "count"),
            year_min=("year", "min"),
            year_max=("year", "max"),
            annual_balance_mean_m=("annual_balance", "mean"),
        )
        .reset_index()
    )

    out = pd.merge(stats, static, on="glacier_id", how="left")
    assert out[["latitude", "longitude"]].notna().all().all(), (
        "Some WGMS glaciers are missing coordinates."
    )
    out = out.sort_values("n_years", ascending=False)

    os.makedirs(os.path.dirname(TRAINING_GLACIERS_CSV), exist_ok=True)
    out.to_csv(TRAINING_GLACIERS_CSV, index=False)
    print(f"Saved training glacier summary: {TRAINING_GLACIERS_CSV}")

    keep_cols = ["glacier_id", "year", "annual_balance"]
    for col in ["winter_balance", "summer_balance"]:
        if col in mb_rgi02.columns:
            keep_cols.append(col)
    mb_rgi02[keep_cols].to_csv(MASSBAL_RGI02_CSV, index=False)
    print(f"Saved annual SMB records: {MASSBAL_RGI02_CSV}")

    print("\nTop observed glaciers:")
    print(
        out[
            [
                "glacier_id",
                "name",
                "latitude",
                "longitude",
                "n_years",
                "annual_balance_mean_m",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
