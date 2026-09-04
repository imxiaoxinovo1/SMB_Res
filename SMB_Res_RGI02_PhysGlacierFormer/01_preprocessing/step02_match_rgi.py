"""Match WGMS RGI02 glaciers to RGI v7 terrain attributes."""
from __future__ import annotations

import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import RGI02_SHP, STATIC_FEATURES, TERRAIN_CSV, TRAINING_GLACIERS_CSV  # noqa: E402


def main() -> None:
    print("=== Step 02: RGI Terrain Matching ===")

    wgms = pd.read_csv(TRAINING_GLACIERS_CSV)
    wgms_coords = np.column_stack([wgms["latitude"].values, wgms["longitude"].values])
    print(f"Training glaciers: {len(wgms)}")

    print(f"Reading RGI shapefile: {RGI02_SHP}")
    rgi = gpd.read_file(RGI02_SHP)[
        [
            "rgi_id",
            "cenlon",
            "cenlat",
            "area_km2",
            "slope_deg",
            "aspect_deg",
            "zmin_m",
            "zmax_m",
            "zmean_m",
            "zmed_m",
            "lmax_m",
        ]
    ].copy()
    print(f"RGI02 glacier outlines: {len(rgi):,}")

    rgi["aspect_sin"] = np.sin(np.deg2rad(rgi["aspect_deg"]))
    rgi["aspect_cos"] = np.cos(np.deg2rad(rgi["aspect_deg"]))

    rgi_coords = np.column_stack([rgi["cenlat"].values, rgi["cenlon"].values])
    tree = cKDTree(rgi_coords)
    dist, idx = tree.query(wgms_coords, k=1)

    distance_threshold_deg = 0.09
    far_mask = dist > distance_threshold_deg
    if far_mask.any():
        print(
            f"WARNING: {int(far_mask.sum())} matched glaciers exceed "
            f"{distance_threshold_deg} degrees."
        )
        for i in np.where(far_mask)[0]:
            print(
                f"  glacier_id={wgms.iloc[i]['glacier_id']} "
                f"distance={dist[i]:.4f} deg"
            )
    else:
        print(f"All match distances <= {distance_threshold_deg} degrees.")

    terrain_cols = [
        "slope_deg",
        "aspect_sin",
        "aspect_cos",
        "zmin_m",
        "zmax_m",
        "zmean_m",
        "zmed_m",
        "area_km2",
        "lmax_m",
        "cenlat",
    ]
    terrain = rgi.iloc[idx][terrain_cols + ["rgi_id"]].reset_index(drop=True)
    terrain["match_dist_deg"] = dist

    out = pd.concat([wgms.reset_index(drop=True), terrain], axis=1)
    missing = [col for col in STATIC_FEATURES if col not in out.columns]
    assert not missing, f"Missing static feature columns: {missing}"
    assert out[["cenlat"]].notna().all().all(), "Matched terrain contains NaN cenlat."

    out.to_csv(TERRAIN_CSV, index=False)
    print(f"Mean match distance: {dist.mean():.4f} deg")
    print(f"Max match distance: {dist.max():.4f} deg")
    print(f"Saved terrain features: {TERRAIN_CSV}")


if __name__ == "__main__":
    main()
