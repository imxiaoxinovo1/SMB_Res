"""Stream corrected ERA5-Land data from two NetCDF domains and reconstruct all RGI02 glaciers."""
from __future__ import annotations

import argparse
import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from xgboost import XGBRegressor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "03_training"))

from config import (  # noqa: E402
    ERA5_ALASKA_NC,
    ERA5_RGI02_NC,
    PHYS_V2_FINAL_MODEL,
    PHYS_V2_FINAL_PREPROCESSOR,
    PHYS_V2_RECONSTRUCTION_CSV,
    RGI02_HYPSOMETRY_CSV,
    RGI02_SHP,
)
from train_xgboost_v2_final import BASE_STATIC_FEATURES  # noqa: E402


DYNAMIC_FEATURES = [
    "t2m", "sd", "asn", "tp", "sf", "ssrd", "str", "slhf", "sshf",
    "t2m_anomaly", "tp_anomaly", "sf_anomaly", "ssrd_anomaly", "asn_anomaly",
]
RAW_VARIABLES = ["t2m", "sd", "asn", "tp", "sf", "ssrd", "str", "slhf", "sshf"]
HYDROLOGY = {"tp", "sf"}
ENERGY = {"ssrd", "str", "slhf", "sshf"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year-min", type=int, default=1951)
    parser.add_argument("--year-max", type=int, default=2024)
    parser.add_argument("--batch-glaciers", type=int, default=128)
    parser.add_argument("--min-area-km2", type=float, default=0.0)
    parser.add_argument("--max-glaciers", type=int, default=None)
    parser.add_argument("--output", default=PHYS_V2_RECONSTRUCTION_CSV)
    return parser.parse_args()


def open_era5(path: str, year_max: int) -> xr.Dataset:
    dataset = xr.open_dataset(path)
    if "valid_time" in dataset.dims:
        dataset = dataset.rename({"valid_time": "time"})
    if "expver" in dataset.dims:
        dataset = dataset.sel(expver=1, drop=True).combine_first(
            dataset.sel(expver=5, drop=True)
        )
    return dataset.sel(time=slice("1950-01", f"{year_max}-12"))


def inside(dataset: xr.Dataset, glaciers: pd.DataFrame) -> np.ndarray:
    return (
        glaciers.cenlat.between(float(dataset.latitude.min()), float(dataset.latitude.max()))
        & glaciers.cenlon.between(float(dataset.longitude.min()), float(dataset.longitude.max()))
    ).to_numpy()


def extract_climate(dataset: xr.Dataset, glaciers: pd.DataFrame) -> dict[str, np.ndarray]:
    latitudes = xr.DataArray(glaciers.cenlat.to_numpy(), dims="glacier")
    longitudes = xr.DataArray(glaciers.cenlon.to_numpy(), dims="glacier")
    times = pd.to_datetime(dataset.time.values)
    days = times.days_in_month.to_numpy(dtype=np.float32)[None, :]
    output: dict[str, np.ndarray] = {}
    for name in RAW_VARIABLES:
        values = dataset[name].interp(
            latitude=latitudes, longitude=longitudes, method="linear"
        ).transpose("glacier", "time").values.astype(np.float32)
        if np.isnan(values).any():
            nearest = dataset[name].interp(
                latitude=latitudes, longitude=longitudes, method="nearest"
            ).transpose("glacier", "time").values.astype(np.float32)
            values = np.where(np.isnan(values), nearest, values)
        if name == "t2m":
            values -= 273.15
        elif name in HYDROLOGY:
            values *= 1000.0 * days
        elif name in ENERGY:
            values /= 86400.0
        output[name] = values
    output["years"] = times.year.to_numpy()
    output["months"] = times.month.to_numpy()
    return output


def hypsometry_quantile_lookup() -> dict[str, np.ndarray]:
    frame = pd.read_csv(RGI02_HYPSOMETRY_CSV)
    band_columns = [column for column in frame.columns if column not in {"rgi_id", "area_km2"}]
    centers = np.asarray([float(column) for column in band_columns], dtype=np.float32)
    lookup = {}
    for _, row in frame.iterrows():
        values = row[band_columns].to_numpy(dtype=np.float32)
        total = float(values.sum())
        if total <= 0:
            continue
        cumulative = np.cumsum(values / total)
        lookup[str(row["rgi_id"])] = np.asarray(
            [centers[np.argmax(cumulative >= quantile)] for quantile in [0.10, 0.25, 0.50, 0.75, 0.90]],
            dtype=np.float32,
        )
    return lookup


def build_glacier_features(
    glacier: pd.Series,
    climate: dict[str, np.ndarray],
    local_index: int,
    years_requested: range,
    hyp: np.ndarray,
) -> tuple[np.ndarray, list[int]]:
    years = climate["years"]
    months = climate["months"]
    reference = (years >= 1981) & (years <= 2010)
    normals = {
        name: np.asarray(
            [climate[name][local_index, reference & (months == month)].mean() for month in range(1, 13)],
            dtype=np.float32,
        )
        for name in ["t2m", "tp", "sf", "ssrd", "asn"]
    }
    winter_months = np.asarray([9, 10, 11, 0, 1, 2, 3])
    summer_months = np.asarray([4, 5, 6, 7, 8])
    static = np.asarray(
        [
            glacier.slope_deg,
            np.sin(np.deg2rad(glacier.aspect_deg)),
            np.cos(np.deg2rad(glacier.aspect_deg)),
            glacier.zmin_m,
            glacier.zmax_m,
            glacier.zmean_m,
            glacier.zmed_m,
            np.log1p(glacier.area_km2),
            np.log1p(glacier.lmax_m),
            glacier.cenlat,
            glacier.cenlon,
            normals["t2m"].mean(),
            normals["tp"][winter_months].sum(),
            normals["t2m"][summer_months].mean(),
            normals["ssrd"][summer_months].mean(),
            normals["t2m"].max() - normals["t2m"].min(),
        ],
        dtype=np.float32,
    )
    if len(static) != len(BASE_STATIC_FEATURES):
        raise RuntimeError("Static reconstruction feature mismatch.")

    rows, valid_years = [], []
    for year in years_requested:
        hydrological = ((years == year - 1) & (months >= 10)) | (
            (years == year) & (months <= 9)
        )
        if hydrological.sum() != 12:
            continue
        month_values = months[hydrological]
        columns = {name: climate[name][local_index, hydrological] for name in RAW_VARIABLES}
        for name in ["t2m", "tp", "sf", "ssrd", "asn"]:
            columns[f"{name}_anomaly"] = columns[name] - normals[name][month_values - 1]
        dynamic = np.column_stack([columns[name] for name in DYNAMIC_FEATURES])
        calendar = np.empty((12, len(DYNAMIC_FEATURES)), dtype=np.float32)
        calendar[month_values - 1] = dynamic
        rows.append(np.concatenate([calendar.T.reshape(-1), static, hyp]))
        valid_years.append(year)
    return np.asarray(rows, dtype=np.float32), valid_years


def main() -> None:
    args = parse_args()
    if args.year_min < 1951:
        raise ValueError("Hydrological-year reconstruction requires year_min >= 1951 (ERA5-Land begins in 1950).")
    glaciers = gpd.read_file(RGI02_SHP).drop(columns="geometry")
    glaciers = glaciers[glaciers.area_km2 >= args.min_area_km2].sort_values("rgi_id").reset_index(drop=True)
    if args.max_glaciers is not None:
        glaciers = glaciers.iloc[: args.max_glaciers].copy()

    model = XGBRegressor()
    model.load_model(PHYS_V2_FINAL_MODEL)
    preprocessing = np.load(PHYS_V2_FINAL_PREPROCESSOR, allow_pickle=True)
    medians = preprocessing["medians"]
    feature_min = preprocessing["feature_min"]
    feature_max = preprocessing["feature_max"]
    stored_feature_names = [str(value) for value in preprocessing["feature_names"]]
    area_min = float(preprocessing["training_area_min_km2"])
    area_max = float(preprocessing["training_area_max_km2"])
    hyp_lookup = hypsometry_quantile_lookup()
    expected_feature_names = [
        f"{name}_m{month:02d}" for name in DYNAMIC_FEATURES for month in range(1, 13)
    ] + BASE_STATIC_FEATURES + [
        "hyp_q10_m", "hyp_q25_m", "hyp_q50_m", "hyp_q75_m", "hyp_q90_m"
    ]
    if stored_feature_names != expected_feature_names:
        raise RuntimeError(
            "Final-model feature schema does not match reconstruction features. "
            "Retrain the final model or use its matching reconstruction code."
        )
    base = open_era5(ERA5_RGI02_NC, args.year_max)
    alaska = open_era5(ERA5_ALASKA_NC, args.year_max)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    if os.path.exists(args.output):
        os.remove(args.output)
    wrote_header = False
    missing_hypsometry = []
    total_rows = 0
    requested_years = range(args.year_min, args.year_max + 1)
    for start in range(0, len(glaciers), args.batch_glaciers):
        batch = glaciers.iloc[start : start + args.batch_glaciers].copy()
        base_mask = inside(base, batch)
        alaska_mask = ~base_mask & inside(alaska, batch)
        if not np.all(base_mask | alaska_mask):
            missing = batch.loc[~(base_mask | alaska_mask), "rgi_id"].tolist()
            raise RuntimeError(f"ERA5 domains do not cover target glaciers: {missing[:5]}")

        for source_name, source, mask in [("rgi02", base, base_mask), ("alaska", alaska, alaska_mask)]:
            subset = batch.loc[mask].reset_index(drop=True)
            if subset.empty:
                continue
            climate = extract_climate(source, subset)
            for local_index, glacier in subset.iterrows():
                hyp = hyp_lookup.get(str(glacier.rgi_id))
                if hyp is None:
                    missing_hypsometry.append(str(glacier.rgi_id))
                    continue
                features, valid_years = build_glacier_features(
                    glacier, climate, local_index, requested_years, hyp
                )
                if not len(valid_years):
                    continue
                features = np.where(np.isnan(features), medians, features)
                if not np.isfinite(features).all():
                    raise RuntimeError(f"Non-finite features for {glacier.rgi_id}")
                predictions = model.predict(features)
                outside_fraction = np.mean(
                    (features < feature_min) | (features > feature_max), axis=1
                )
                output = pd.DataFrame(
                    {
                        "rgi_id": glacier.rgi_id,
                        "o2region": glacier.o2region,
                        "year": valid_years,
                        "predicted_smb_m": predictions,
                        "area_km2": glacier.area_km2,
                        "cenlon": glacier.cenlon,
                        "cenlat": glacier.cenlat,
                        "era5_source": source_name,
                        "recommended_area_domain": glacier.area_km2 >= 0.5,
                        "area_outside_training_range": not (area_min <= glacier.area_km2 <= area_max),
                        "feature_outside_training_fraction": outside_fraction,
                    }
                )
                output.to_csv(args.output, mode="a", header=not wrote_header, index=False)
                wrote_header = True
                total_rows += len(output)
        print(f"Processed {min(start + args.batch_glaciers, len(glaciers)):,}/{len(glaciers):,}; rows={total_rows:,}")

    base.close()
    alaska.close()
    if missing_hypsometry:
        raise RuntimeError(f"Missing hypsometry for {len(missing_hypsometry)} glaciers.")
    print(f"Saved {total_rows:,} glacier-years -> {args.output}")


if __name__ == "__main__":
    main()
