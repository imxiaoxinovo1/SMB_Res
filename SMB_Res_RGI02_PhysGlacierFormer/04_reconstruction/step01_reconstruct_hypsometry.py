"""Reconstruct annual SMB for target RGI02 glaciers with final hypsometry model."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "02_models"))

from config import (  # noqa: E402
    CAL_SUMMER_MONTHS,
    CAL_WINTER_MONTHS,
    ERA5_RGI02_FULL_CSV,
    GLACIERFORMER_HYPSOMETRY_PARAMS,
    HYD_ABLAT_MONTHS,
    HYPSOMETRY_FINAL_WEIGHTS,
    HYPSOMETRY_RECON_QC_CSV,
    HYPSOMETRY_RECON_RAW_CSV,
    MONTHLY_CLIMATE_VARS,
    RECONSTRUCTION_DIR,
    RECON_YEAR_MAX,
    RECON_YEAR_MIN,
    RGI02_HYPSOMETRY_CSV,
    RGI02_TARGET_CSV,
    SEQUENCES_HYPSOMETRY_QC_NPZ,
    SEASONAL_EXTRA_FEATURES,
    STATIC_FEATURES,
)
from glacierformer_hypsometry import GlacierFormerHypsometry  # noqa: E402


INFER_BATCH_SIZE = 4096


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


def build_hypsometry_features(rgi_ids: list[str]) -> np.ndarray:
    hyp = pd.read_csv(RGI02_HYPSOMETRY_CSV)
    band_cols = [col for col in hyp.columns if col not in ["rgi_id", "area_km2"]]
    band_centers = np.array([float(col) for col in band_cols], dtype=np.float32)
    center_norm = (band_centers - float(band_centers.mean())) / float(band_centers.std() + 1e-8)
    hyp_by_rgi = hyp.set_index("rgi_id")

    x_hyp = np.zeros((len(rgi_ids), len(band_cols), 3), dtype=np.float32)
    missing = []
    for i, rgi_id in enumerate(rgi_ids):
        if rgi_id not in hyp_by_rgi.index:
            missing.append(rgi_id)
            continue
        area_fraction = hyp_by_rgi.loc[rgi_id, band_cols].values.astype(np.float32) / 1000.0
        total = float(area_fraction.sum())
        if total <= 0:
            missing.append(rgi_id)
            continue
        area_fraction = area_fraction / total
        mean_elevation = float(np.sum(area_fraction * band_centers))
        x_hyp[i, :, 0] = area_fraction
        x_hyp[i, :, 1] = center_norm
        x_hyp[i, :, 2] = (band_centers - mean_elevation) / 1000.0
    if missing:
        raise RuntimeError(f"Missing hypsometry for {len(missing)} target glaciers: {missing[:5]}")
    return x_hyp


def main() -> None:
    print("=== Reconstruction Step 01: Hypsometry GlacierFormer Raw Prediction ===")
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.makedirs(RECONSTRUCTION_DIR, exist_ok=True)

    print("Loading normalization statistics...")
    norm = np.load(SEQUENCES_HYPSOMETRY_QC_NPZ, allow_pickle=True)
    dyn_mean = norm["dyn_mean"]
    dyn_std = norm["dyn_std"]
    sta_mean = norm["sta_mean"]
    sta_std = norm["sta_std"]
    sta_medians = norm["sta_medians"]

    print("Loading model...")
    params = GLACIERFORMER_HYPSOMETRY_PARAMS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GlacierFormerHypsometry(
        n_dynamic_features=params["n_dynamic_features"],
        n_static_features=params["n_static_features"],
        n_hypsometry_features=params["n_hypsometry_features"],
        d_model=params["d_model"],
        n_heads=params["n_heads"],
        n_encoder_layers=params["n_encoder_layers"],
        ff_dim=params["ff_dim"],
        dropout=params["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(HYPSOMETRY_FINAL_WEIGHTS, map_location=device, weights_only=True))
    model.eval()
    print(f"Device: {device}")

    print("Loading target glaciers and ERA5...")
    terrain = pd.read_csv(RGI02_TARGET_CSV).set_index("rgi_id")
    rgi_ids = terrain.index.tolist()
    x_hyp_by_target = build_hypsometry_features(rgi_ids)
    hyp_lookup = {rgi_id: x_hyp_by_target[i] for i, rgi_id in enumerate(rgi_ids)}

    usecols = ["rgi_id", "year", "month", *MONTHLY_CLIMATE_VARS]
    era5 = pd.read_csv(ERA5_RGI02_FULL_CSV, usecols=usecols)
    era5_by_rgi = {rgi_id: grp for rgi_id, grp in era5.groupby("rgi_id", sort=False)}
    years_range = list(range(RECON_YEAR_MIN, RECON_YEAR_MAX + 1))

    sample_dyn = []
    sample_sta = []
    sample_hyp = []
    sample_meta = []
    skipped_no_era5 = 0
    skipped_dynamic_nan = []
    filled_static_nan = []

    print("Building samples...")
    for idx, rgi_id in enumerate(rgi_ids):
        if idx == 0 or (idx + 1) % 500 == 0:
            print(f"  {idx + 1}/{len(rgi_ids)} glaciers; samples={len(sample_meta):,}")
        glacier_era5 = era5_by_rgi.get(rgi_id)
        if glacier_era5 is None:
            skipped_no_era5 += 1
            continue
        terrain_row = terrain.loc[rgi_id]
        sta_terrain = np.array([terrain_row[col] for col in STATIC_FEATURES], dtype=np.float32)
        for year in years_range:
            year_era5 = glacier_era5[glacier_era5["year"] == year].sort_values("month")
            if len(year_era5) < 12:
                continue
            dyn = year_era5[MONTHLY_CLIMATE_VARS].values.astype(np.float32)
            if np.isnan(dyn).any():
                skipped_dynamic_nan.append({"rgi_id": rgi_id, "year": year, "count": int(np.isnan(dyn).sum())})
                continue
            sta = np.concatenate([sta_terrain, seasonal_feats(glacier_era5, year)])
            sample_dyn.append(dyn)
            sample_sta.append(sta)
            sample_hyp.append(hyp_lookup[rgi_id])
            sample_meta.append({"rgi_id": rgi_id, "year": year})

    if not sample_meta:
        raise RuntimeError("No reconstruction samples built.")

    x_dyn = np.stack(sample_dyn).astype(np.float32)
    x_sta = np.stack(sample_sta).astype(np.float32)
    x_hyp = np.stack(sample_hyp).astype(np.float32)
    static_nan = np.isnan(x_sta)
    if static_nan.any():
        for row_idx in np.where(static_nan.any(axis=1))[0]:
            filled_static_nan.append(
                {
                    "rgi_id": sample_meta[row_idx]["rgi_id"],
                    "year": sample_meta[row_idx]["year"],
                    "count": int(static_nan[row_idx].sum()),
                }
            )
        x_sta = np.where(static_nan, sta_medians, x_sta)

    x_dyn = (x_dyn - dyn_mean) / dyn_std
    x_sta = (x_sta - sta_mean) / sta_std
    if np.isnan(x_dyn).any() or np.isnan(x_sta).any() or np.isnan(x_hyp).any():
        raise RuntimeError("NaNs remain in reconstruction tensors.")

    print(f"Running inference on {len(sample_meta):,} samples...")
    preds = []
    with torch.no_grad():
        for start in range(0, len(sample_meta), INFER_BATCH_SIZE):
            end = min(start + INFER_BATCH_SIZE, len(sample_meta))
            pred = model(
                torch.tensor(x_dyn[start:end], dtype=torch.float32, device=device),
                torch.tensor(x_sta[start:end], dtype=torch.float32, device=device),
                torch.tensor(x_hyp[start:end], dtype=torch.float32, device=device),
            )
            preds.append(pred.detach().cpu().numpy())
    preds = np.concatenate(preds)

    out = pd.DataFrame(
        {
            "rgi_id": [m["rgi_id"] for m in sample_meta],
            "year": [m["year"] for m in sample_meta],
            "predicted_smb_m": preds.astype(float),
        }
    )
    terrain_out = terrain.reset_index()[["rgi_id", "area_km2", "cenlat", "cenlon"]]
    out = out.merge(terrain_out, on="rgi_id", how="left")
    out = out.sort_values(["rgi_id", "year"]).reset_index(drop=True)
    out.to_csv(HYPSOMETRY_RECON_RAW_CSV, index=False)

    qc_rows = []
    qc_rows.extend({"issue": "skipped_dynamic_nan", **row} for row in skipped_dynamic_nan)
    qc_rows.extend({"issue": "filled_static_nan", **row} for row in filled_static_nan)
    if skipped_no_era5:
        qc_rows.append({"issue": "skipped_no_era5_glaciers", "rgi_id": "", "year": "", "count": skipped_no_era5})
    pd.DataFrame(qc_rows, columns=["issue", "rgi_id", "year", "count"]).to_csv(HYPSOMETRY_RECON_QC_CSV, index=False)

    print(f"Saved raw reconstruction -> {HYPSOMETRY_RECON_RAW_CSV}")
    print(f"Saved QC -> {HYPSOMETRY_RECON_QC_CSV}")
    print(out["predicted_smb_m"].describe())


if __name__ == "__main__":
    main()
