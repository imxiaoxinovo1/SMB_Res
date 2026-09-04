"""Build a robust WGMS-to-RGI v7 mapping for the PhysGlacierFormer v2 data."""
from __future__ import annotations

import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (  # noqa: E402
    GLACIER_CSV,
    PHYS_V2_DATA_DIR,
    PHYS_V2_TERRAIN_CSV,
    QC_MATCH_DIST_MAX_DEG,
    RGI02_RGI60_SHP,
    RGI02_SHP,
    TRAINING_GLACIERS_CSV,
)


RGI7_COLUMNS = [
    "rgi_id", "o2region", "cenlon", "cenlat", "area_km2", "slope_deg", "aspect_deg",
    "zmin_m", "zmax_m", "zmean_m", "zmed_m", "lmax_m", "geometry",
]


def read_csv_fallback(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1", low_memory=False)


def main() -> None:
    print("=== Step 11: Robust WGMS-to-RGI v7 Mapping ===")
    os.makedirs(PHYS_V2_DATA_DIR, exist_ok=True)

    observed = pd.read_csv(TRAINING_GLACIERS_CSV)
    wgms = read_csv_fallback(GLACIER_CSV)[["id", "rgi60_ids"]]
    observed = observed.merge(wgms, left_on="glacier_id", right_on="id", how="left")

    rgi7 = gpd.read_file(RGI02_SHP)[RGI7_COLUMNS].copy()
    rgi6 = gpd.read_file(RGI02_RGI60_SHP)[["RGIId", "geometry"]].copy()
    if rgi7.crs is None or rgi6.crs is None:
        raise RuntimeError("RGI shapefiles must define a CRS.")
    rgi6 = rgi6.to_crs(rgi7.crs)

    rgi7_eq = rgi7.to_crs("EPSG:6933")
    rgi6_eq = rgi6.to_crs("EPSG:6933").set_index("RGIId")
    rgi6_wgs = rgi6.set_index("RGIId")

    rows: list[dict] = []
    for _, glacier in observed.iterrows():
        rgi60_id = glacier.get("rgi60_ids")
        mapping_method = "nearest_outline"
        overlap_fraction = np.nan
        distance_km = np.nan

        if pd.notna(rgi60_id) and rgi60_id in rgi6_eq.index:
            old_geom = rgi6_eq.loc[rgi60_id, "geometry"]
            candidate_idx = list(rgi7_eq.sindex.query(old_geom, predicate="intersects"))
            if not candidate_idx:
                raise RuntimeError(f"No RGI v7 overlap found for {rgi60_id}.")
            overlap_areas = rgi7_eq.iloc[candidate_idx].geometry.intersection(old_geom).area
            best_position = int(np.argmax(overlap_areas.to_numpy()))
            best_idx = candidate_idx[best_position]
            matched = rgi7.iloc[best_idx]
            overlap_fraction = float(overlap_areas.iloc[best_position] / old_geom.area)
            mapping_method = "rgi6_polygon_overlap"
            old_centroid = rgi6_wgs.loc[rgi60_id, "geometry"].centroid
            distance_km = float(
                np.hypot(
                    (matched["cenlat"] - old_centroid.y) * 111.32,
                    (matched["cenlon"] - old_centroid.x)
                    * 111.32
                    * np.cos(np.deg2rad(old_centroid.y)),
                )
            )
        else:
            point = gpd.GeoSeries(
                [Point(float(glacier["longitude"]), float(glacier["latitude"]))],
                crs="EPSG:4326",
            ).to_crs("EPSG:6933").iloc[0]
            distances = rgi7_eq.geometry.distance(point)
            best_idx = int(distances.idxmin())
            matched = rgi7.loc[best_idx]
            distance_km = float(distances.loc[best_idx] / 1000.0)

        record = glacier.drop(labels=["id"], errors="ignore").to_dict()
        for column in RGI7_COLUMNS:
            if column != "geometry":
                record[column] = matched[column]
        record["aspect_sin"] = np.sin(np.deg2rad(float(matched["aspect_deg"])))
        record["aspect_cos"] = np.cos(np.deg2rad(float(matched["aspect_deg"])))
        record["mapping_method"] = mapping_method
        record["rgi6_overlap_fraction"] = overlap_fraction
        record["match_distance_km"] = distance_km
        record["mapping_qc_pass"] = bool(
            (mapping_method == "rgi6_polygon_overlap" and overlap_fraction >= 0.80)
            or (mapping_method == "nearest_outline" and distance_km <= QC_MATCH_DIST_MAX_DEG * 111.32)
        )
        rows.append(record)

    out = pd.DataFrame(rows).sort_values("n_years", ascending=False)
    duplicate_rgi = out.loc[out.duplicated("rgi_id", keep=False), ["glacier_id", "name", "rgi_id"]]
    if len(duplicate_rgi):
        print("WARNING: multiple WGMS series map to the same RGI v7 outline:")
        print(duplicate_rgi.to_string(index=False))

    out.to_csv(PHYS_V2_TERRAIN_CSV, index=False)
    print(f"Mapped glaciers: {len(out)}")
    print(f"RGI6 overlap mappings: {(out['mapping_method'] == 'rgi6_polygon_overlap').sum()}")
    print(f"Fallback outline mappings: {(out['mapping_method'] == 'nearest_outline').sum()}")
    print(f"QC-pass mappings: {out['mapping_qc_pass'].sum()}")
    print(f"Saved -> {PHYS_V2_TERRAIN_CSV}")


if __name__ == "__main__":
    main()
