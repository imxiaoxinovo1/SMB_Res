"""Add RGI v7 hypsometry tokens to the QC sequence dataset."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (  # noqa: E402
    HYPSOMETRY_DATA_DIR,
    RGI02_HYPSOMETRY_CSV,
    SEQUENCES_HYPSOMETRY_QC_NPZ,
    SEQUENCES_QC_NPZ,
)


def main() -> None:
    print("=== Step 07: Build Hypsometry Sequence Dataset ===")
    os.makedirs(HYPSOMETRY_DATA_DIR, exist_ok=True)

    seq = np.load(SEQUENCES_QC_NPZ, allow_pickle=True)
    glacier_ids = seq["glacier_ids"]
    years = seq["years"]
    y = seq["y"]

    terrain_qc = pd.read_csv(
        os.path.join(os.path.dirname(SEQUENCES_QC_NPZ), "training_glaciers_terrain_qc.csv")
    )
    id_to_rgi = terrain_qc.set_index("glacier_id")["rgi_id"].to_dict()

    hyp = pd.read_csv(RGI02_HYPSOMETRY_CSV)
    band_cols = [col for col in hyp.columns if col not in ["rgi_id", "area_km2"]]
    band_centers = np.array([float(col) for col in band_cols], dtype=np.float32)

    center_mean = float(band_centers.mean())
    center_std = float(band_centers.std() + 1e-8)
    center_norm = (band_centers - center_mean) / center_std

    hyp_by_rgi = hyp.set_index("rgi_id")
    x_hyp = np.zeros((len(glacier_ids), len(band_cols), 3), dtype=np.float32)
    missing = []

    for i, gid in enumerate(glacier_ids):
        rgi_id = id_to_rgi.get(int(gid))
        if rgi_id not in hyp_by_rgi.index:
            missing.append((int(gid), rgi_id))
            continue

        area_fraction = hyp_by_rgi.loc[rgi_id, band_cols].values.astype(np.float32) / 1000.0
        total = float(area_fraction.sum())
        if total <= 0:
            missing.append((int(gid), rgi_id))
            continue
        area_fraction = area_fraction / total

        mean_elevation = float(np.sum(area_fraction * band_centers))
        relative_elevation_km = (band_centers - mean_elevation) / 1000.0

        x_hyp[i, :, 0] = area_fraction
        x_hyp[i, :, 1] = center_norm
        x_hyp[i, :, 2] = relative_elevation_km

    if missing:
        raise RuntimeError(f"Missing hypsometry for {len(missing)} samples: {missing[:5]}")

    np.savez_compressed(
        SEQUENCES_HYPSOMETRY_QC_NPZ,
        X_dyn=seq["X_dyn"],
        X_sta=seq["X_sta"],
        X_hyp=x_hyp,
        y=y,
        glacier_ids=glacier_ids,
        years=years,
        dyn_mean=seq["dyn_mean"],
        dyn_std=seq["dyn_std"],
        sta_mean=seq["sta_mean"],
        sta_std=seq["sta_std"],
        sta_medians=seq["sta_medians"],
        hypsometry_band_centers_m=band_centers,
        hypsometry_features=np.array(
            ["area_fraction", "elevation_center_norm", "relative_elevation_km"]
        ),
    )

    print(f"Input QC samples: {len(y):,}")
    print(f"Labeled samples: {int((~np.isnan(y)).sum()):,}")
    print(f"X_hyp shape: {x_hyp.shape}")
    print(f"Area fraction sum range: {x_hyp[:, :, 0].sum(axis=1).min():.3f} to {x_hyp[:, :, 0].sum(axis=1).max():.3f}")
    print(f"Saved hypsometry sequence dataset: {SEQUENCES_HYPSOMETRY_QC_NPZ}")


if __name__ == "__main__":
    main()
