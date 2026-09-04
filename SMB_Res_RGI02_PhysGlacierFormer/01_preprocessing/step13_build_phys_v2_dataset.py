"""Build the raw, hydrological-year-aware PhysGlacierFormer v2 dataset."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (  # noqa: E402
    HUGONNET_WEAK_YEAR_MAX,
    MASSBAL_CSV,
    PHYS_V2_DATA_DIR,
    PHYS_V2_DATA_SUMMARY_CSV,
    PHYS_V2_DYNAMIC_FEATURES,
    PHYS_V2_ERA5_CSV,
    PHYS_V2_SEQUENCES_NPZ,
    PHYS_V2_STATIC_FEATURES,
    PHYS_V2_TERRAIN_CSV,
    RGI02_HYPSOMETRY_CSV,
    TRAIN_YEAR_MAX,
)


BASE_DYNAMIC_FEATURES = [
    "t2m", "d2m", "sd", "asn", "tp", "sf", "smlt",
    "ssrd", "strd", "str", "slhf", "sshf",
]
ANOMALY_FEATURES = ["t2m", "tp", "sf", "ssrd", "asn"]
DEFAULT_END_MONTH = 9


def read_csv_fallback(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1", low_memory=False)


def relative_humidity(t2m_c: np.ndarray, d2m_c: np.ndarray) -> np.ndarray:
    numerator = np.exp(17.625 * d2m_c / (243.04 + d2m_c))
    denominator = np.exp(17.625 * t2m_c / (243.04 + t2m_c))
    return np.clip(100.0 * numerator / denominator, 0.0, 100.0)


def observation_end_month(row: pd.Series) -> int:
    begin = pd.to_datetime(row.get("begin_date"), errors="coerce")
    end = pd.to_datetime(row.get("end_date"), errors="coerce")
    if pd.isna(begin) or pd.isna(end):
        return DEFAULT_END_MONTH
    duration = (end - begin).days
    if 300 <= duration <= 430 and int(end.year) == int(row["year"]):
        return int(end.month)
    return DEFAULT_END_MONTH


def build_hypsometry_lookup() -> tuple[dict[str, np.ndarray], np.ndarray]:
    hyp = pd.read_csv(RGI02_HYPSOMETRY_CSV)
    band_cols = [column for column in hyp.columns if column not in {"rgi_id", "area_km2"}]
    centers = np.asarray([float(column) for column in band_cols], dtype=np.float32)
    center_norm = (centers - centers.mean()) / (centers.std() + 1e-8)
    lookup: dict[str, np.ndarray] = {}
    for _, row in hyp.iterrows():
        fractions = row[band_cols].to_numpy(dtype=np.float32) / 1000.0
        total = float(fractions.sum())
        if total <= 0:
            continue
        fractions /= total
        mean_elevation = float(np.sum(fractions * centers))
        lookup[str(row["rgi_id"])] = np.column_stack(
            [fractions, center_norm, (centers - mean_elevation) / 1000.0]
        ).astype(np.float32)
    return lookup, centers


def main() -> None:
    print("=== Step 13: Build PhysGlacierFormer v2 Dataset ===")
    os.makedirs(PHYS_V2_DATA_DIR, exist_ok=True)

    terrain = pd.read_csv(PHYS_V2_TERRAIN_CSV)
    terrain = terrain[terrain["mapping_qc_pass"]].copy()
    terrain_idx = terrain.set_index("glacier_id")

    era5 = pd.read_csv(PHYS_V2_ERA5_CSV)
    era5["relative_humidity"] = relative_humidity(
        era5["t2m"].to_numpy(float), era5["d2m"].to_numpy(float)
    )
    mean_surface_pressure = era5.groupby("glacier_id")["sp"].mean()
    pressure_elevation = 44330.0 * (
        1.0 - np.power(mean_surface_pressure.clip(lower=1000.0) / 101325.0, 0.1903)
    )
    elevation_difference = {
        int(gid): float(terrain_idx.loc[int(gid), "zmean_m"] - elevation)
        for gid, elevation in pressure_elevation.items()
        if int(gid) in terrain_idx.index
    }
    era5["t2m_lapse"] = era5["t2m"] - 0.0065 * era5["glacier_id"].map(elevation_difference)
    era5_idx = era5.set_index(["glacier_id", "year", "month"]).sort_index()

    labels = read_csv_fallback(MASSBAL_CSV)
    labels = labels[
        labels["glacier_id"].isin(terrain_idx.index)
        & labels["annual_balance"].notna()
        & labels["year"].between(1951, TRAIN_YEAR_MAX)
    ].copy()
    labels["annual_balance"] = pd.to_numeric(labels["annual_balance"], errors="coerce")
    labels = labels.dropna(subset=["annual_balance"])

    reference = era5[era5["year"].between(1981, 2010)].copy()
    climatology = reference.groupby(["glacier_id", "month"])[
        BASE_DYNAMIC_FEATURES + ["relative_humidity", "t2m_lapse"]
    ].mean()

    climate_static: dict[int, dict[str, float]] = {}
    for gid, frame in climatology.reset_index().groupby("glacier_id"):
        winter = frame[frame["month"].isin([10, 11, 12, 1, 2, 3, 4])]
        summer = frame[frame["month"].isin([5, 6, 7, 8, 9])]
        climate_static[int(gid)] = {
            "clim_annual_t2m": float(frame["t2m"].mean()),
            "clim_winter_tp": float(winter["tp"].sum()),
            "clim_summer_t2m": float(summer["t2m"].mean()),
            "clim_summer_ssrd": float(summer["ssrd"].mean()),
            "clim_t2m_amplitude": float(frame["t2m"].max() - frame["t2m"].min()),
            "clim_annual_t2m_lapse": float(frame["t2m_lapse"].mean()),
            "clim_summer_t2m_lapse": float(summer["t2m_lapse"].mean()),
        }

    hyp_lookup, band_centers = build_hypsometry_lookup()
    x_dyn_list: list[np.ndarray] = []
    x_sta_list: list[np.ndarray] = []
    x_hyp_list: list[np.ndarray] = []
    month_ids_list: list[np.ndarray] = []
    metadata: list[dict] = []
    skipped = 0

    for _, label in labels.iterrows():
        gid, year = int(label["glacier_id"]), int(label["year"])
        terrain_row = terrain_idx.loc[gid]
        rgi_id = str(terrain_row["rgi_id"])
        if rgi_id not in hyp_lookup or gid not in climate_static:
            skipped += 1
            continue

        end_month = observation_end_month(label)
        periods = pd.period_range(end=f"{year}-{end_month:02d}", periods=12, freq="M")
        keys = [(gid, int(period.year), int(period.month)) for period in periods]
        try:
            climate = era5_idx.loc[keys].reset_index()
        except KeyError:
            skipped += 1
            continue
        if len(climate) != 12:
            skipped += 1
            continue

        dyn_columns: dict[str, np.ndarray] = {
            name: climate[name].to_numpy(dtype=np.float32)
            for name in BASE_DYNAMIC_FEATURES + ["relative_humidity", "t2m_lapse"]
        }
        for name in ANOMALY_FEATURES:
            normals = np.asarray(
                [climatology.loc[(gid, int(month)), name] for month in climate["month"]],
                dtype=np.float32,
            )
            dyn_columns[f"{name}_anomaly"] = dyn_columns[name] - normals
        dynamic = np.column_stack([dyn_columns[name] for name in PHYS_V2_DYNAMIC_FEATURES])
        if not np.isfinite(dynamic).all():
            skipped += 1
            continue

        static_values = {
            "slope_deg": float(terrain_row["slope_deg"]),
            "aspect_sin": float(terrain_row["aspect_sin"]),
            "aspect_cos": float(terrain_row["aspect_cos"]),
            "zmin_m": float(terrain_row["zmin_m"]),
            "zmax_m": float(terrain_row["zmax_m"]),
            "zmean_m": float(terrain_row["zmean_m"]),
            "zmed_m": float(terrain_row["zmed_m"]),
            "log1p_area_km2": float(np.log1p(terrain_row["area_km2"])),
            "log1p_lmax_m": float(np.log1p(terrain_row["lmax_m"])),
            "cenlat": float(terrain_row["cenlat"]),
            "cenlon": float(terrain_row["cenlon"]),
            "era5_pressure_elevation_m": float(pressure_elevation.loc[gid]),
            "elevation_difference_m": float(elevation_difference[gid]),
            **climate_static[gid],
        }
        static = np.asarray([static_values[name] for name in PHYS_V2_STATIC_FEATURES], dtype=np.float32)

        x_dyn_list.append(dynamic.astype(np.float32))
        x_sta_list.append(static)
        x_hyp_list.append(hyp_lookup[rgi_id])
        month_ids_list.append(climate["month"].to_numpy(dtype=np.int64))
        metadata.append(
            {
                "glacier_id": gid,
                "rgi_id": rgi_id,
                "o2region": str(terrain_row["o2region"]),
                "year": year,
                "end_month": end_month,
                "annual": float(label["annual_balance"]),
                "winter": pd.to_numeric(label.get("winter_balance"), errors="coerce"),
                "summer": pd.to_numeric(label.get("summer_balance"), errors="coerce"),
                "annual_unc": pd.to_numeric(label.get("annual_balance_unc"), errors="coerce"),
                "winter_unc": pd.to_numeric(label.get("winter_balance_unc"), errors="coerce"),
                "summer_unc": pd.to_numeric(label.get("summer_balance_unc"), errors="coerce"),
                "time_system": str(label.get("time_system", "")),
                "is_hugonnet_period": int(2000 <= year <= HUGONNET_WEAK_YEAR_MAX),
            }
        )

    meta = pd.DataFrame(metadata)
    if meta.empty:
        raise RuntimeError("No valid PhysGlacierFormer v2 samples were built.")

    np.savez_compressed(
        PHYS_V2_SEQUENCES_NPZ,
        X_dyn=np.asarray(x_dyn_list, dtype=np.float32),
        X_sta=np.asarray(x_sta_list, dtype=np.float32),
        X_hyp=np.asarray(x_hyp_list, dtype=np.float32),
        month_ids=np.asarray(month_ids_list, dtype=np.int64),
        y_annual=meta["annual"].to_numpy(dtype=np.float32),
        y_winter=meta["winter"].to_numpy(dtype=np.float32),
        y_summer=meta["summer"].to_numpy(dtype=np.float32),
        annual_unc=meta["annual_unc"].to_numpy(dtype=np.float32),
        winter_unc=meta["winter_unc"].to_numpy(dtype=np.float32),
        summer_unc=meta["summer_unc"].to_numpy(dtype=np.float32),
        glacier_ids=meta["glacier_id"].to_numpy(dtype=np.int64),
        rgi_ids=meta["rgi_id"].to_numpy(dtype=str),
        o2regions=meta["o2region"].to_numpy(dtype=str),
        years=meta["year"].to_numpy(dtype=np.int64),
        end_months=meta["end_month"].to_numpy(dtype=np.int64),
        time_system=meta["time_system"].to_numpy(dtype=str),
        dynamic_features=np.asarray(PHYS_V2_DYNAMIC_FEATURES),
        static_features=np.asarray(PHYS_V2_STATIC_FEATURES),
        hypsometry_features=np.asarray(
            ["area_fraction", "elevation_center_norm", "relative_elevation_km"]
        ),
        hypsometry_band_centers_m=band_centers,
    )

    summary = pd.DataFrame(
        [
            {"metric": "samples", "value": len(meta)},
            {"metric": "glaciers", "value": meta["glacier_id"].nunique()},
            {"metric": "years", "value": meta["year"].nunique()},
            {"metric": "winter_labels", "value": meta["winter"].notna().sum()},
            {"metric": "summer_labels", "value": meta["summer"].notna().sum()},
            {"metric": "annual_uncertainties", "value": meta["annual_unc"].notna().sum()},
            {"metric": "default_end_month_samples", "value": (meta["end_month"] == DEFAULT_END_MONTH).sum()},
            {"metric": "skipped", "value": skipped},
        ]
    )
    summary.to_csv(PHYS_V2_DATA_SUMMARY_CSV, index=False)
    print(summary.to_string(index=False))
    print(f"X_dyn: {(len(meta), 12, len(PHYS_V2_DYNAMIC_FEATURES))}")
    print(f"X_sta: {(len(meta), len(PHYS_V2_STATIC_FEATURES))}")
    print(f"Saved -> {PHYS_V2_SEQUENCES_NPZ}")


if __name__ == "__main__":
    main()
