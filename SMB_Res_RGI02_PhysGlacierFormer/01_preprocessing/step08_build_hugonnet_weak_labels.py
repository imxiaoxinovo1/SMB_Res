"""Build glacier-level Hugonnet weak labels for RGI02.

The output is a 2000-2020 geodetic specific-mass-change constraint
(`dmdtda`, m w.e. yr-1) mapped from RGI 6.0 IDs to local RGI 7.0 IDs.
It is intended for multi-year weak supervision, not as annual SMB labels.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (  # noqa: E402
    HUGONNET_DATA_DIR,
    HUGONNET_MAX_AREA_RATIO_RGI7_TO_RGI60,
    HUGONNET_MAX_MAP_DIST_DEG,
    HUGONNET_MATCH_QC_CSV,
    HUGONNET_MIN_PERC_AREA_MEAS,
    HUGONNET_MIN_PERC_AREA_RES,
    HUGONNET_MIN_AREA_RATIO_RGI7_TO_RGI60,
    HUGONNET_MIN_RGI60_OVERLAP_FRACTION,
    HUGONNET_MIN_RGI7_OVERLAP_FRACTION,
    HUGONNET_MIN_VALID_OBS_PY,
    HUGONNET_QC_SUMMARY_CSV,
    HUGONNET_RGI02_RATES,
    HUGONNET_WEAK_LABEL_PERIOD,
    HUGONNET_WEAK_LABELS_CSV,
    RGI02_ATTRIBUTES_CSV,
    RGI02_LINKS_CSV,
    RGI02_SHP,
    RGI02_RGI60_SHP,
)

EQUAL_AREA_CRS = "EPSG:6933"


def read_rgi60_attributes() -> pd.DataFrame:
    """Read the RGI 6.0 attributes needed to bridge to RGI 7.0."""
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise RuntimeError("geopandas is required to read the RGI 6.0 shapefile") from exc

    rgi60 = gpd.read_file(RGI02_RGI60_SHP)
    return rgi60[
        ["RGIId", "GLIMSId", "CenLon", "CenLat", "Area"]
    ].rename(
        columns={
            "RGIId": "rgi60_id",
            "GLIMSId": "rgi60_glims_id",
            "CenLon": "rgi60_cenlon",
            "CenLat": "rgi60_cenlat",
            "Area": "rgi60_area_km2",
        }
    )


def _read_geometries() -> tuple:
    """Read RGI 7.0 and RGI 6.0 outlines with columns needed for overlap."""
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise RuntimeError("geopandas is required for spatial RGI7-RGI6 matching") from exc

    rgi7 = gpd.read_file(RGI02_SHP)
    rgi7 = rgi7[
        ["rgi_id", "glims_id", "cenlon", "cenlat", "area_km2", "is_rgi6", "geometry"]
    ].copy()

    rgi60 = gpd.read_file(RGI02_RGI60_SHP)
    rgi60 = rgi60[
        ["RGIId", "GLIMSId", "CenLon", "CenLat", "Area", "geometry"]
    ].rename(
        columns={
            "RGIId": "rgi60_id",
            "GLIMSId": "rgi60_glims_id",
            "CenLon": "rgi60_cenlon",
            "CenLat": "rgi60_cenlat",
            "Area": "rgi60_area_km2",
        }
    )
    return rgi7, rgi60


def build_official_overlap_map() -> pd.DataFrame | None:
    """Use the official RGI7-RGI6 overlap table when it is informative."""
    links = pd.read_csv(RGI02_LINKS_CSV)
    if len(links) <= 100 or not {"rgi7_id", "rgi6_id"}.issubset(links.columns):
        return None

    out = links.rename(columns={"rgi7_id": "rgi_id", "rgi6_id": "rgi60_id"}).copy()
    out["mapping_method"] = "official_rgi6_links"
    out["map_area_ratio_rgi7_to_rgi60"] = np.nan
    out["map_dist_deg"] = np.nan
    return out[
        [
            "rgi_id",
            "rgi60_id",
            "mapping_method",
            "overlap_area_km2",
            "rgi7_area_fraction",
            "rgi6_area_fraction",
            "cluster_id",
            "n_rgi7",
            "n_rgi6",
            "map_dist_deg",
            "map_area_ratio_rgi7_to_rgi60",
        ]
    ]


def build_spatial_overlap_map() -> pd.DataFrame:
    """Create an RGI7-RGI6 crosswalk from polygon overlap.

    RGI 7.0 IDs are not numerically compatible with RGI 6.0 IDs. The official
    RGI product defines cross-version links by overlapping outlines, so this
    fallback reconstructs that logic locally when the packaged link table is
    incomplete.
    """
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise RuntimeError("geopandas is required for spatial RGI7-RGI6 matching") from exc

    rgi7, rgi60 = _read_geometries()
    rgi7_eq = rgi7.to_crs(EQUAL_AREA_CRS)
    rgi60_eq = rgi60.to_crs(EQUAL_AREA_CRS)

    left = rgi7_eq[
        ["rgi_id", "glims_id", "cenlon", "cenlat", "area_km2", "is_rgi6", "geometry"]
    ].copy()
    right = rgi60_eq[
        [
            "rgi60_id",
            "rgi60_glims_id",
            "rgi60_cenlon",
            "rgi60_cenlat",
            "rgi60_area_km2",
            "geometry",
        ]
    ].copy()

    pairs = gpd.sjoin(left, right, how="inner", predicate="intersects")
    if pairs.empty:
        raise RuntimeError("No RGI7-RGI6 polygon intersections found")

    right_geom = right.set_index("rgi60_id")["geometry"]
    pairs["rgi60_geometry"] = pairs["rgi60_id"].map(right_geom)
    pairs["overlap_area_km2"] = [
        geom.intersection(rgeom).area / 1_000_000.0
        for geom, rgeom in zip(pairs.geometry, pairs["rgi60_geometry"])
    ]
    pairs = pairs[pairs["overlap_area_km2"] > 0].copy()
    pairs["rgi7_area_fraction"] = pairs["overlap_area_km2"] / pairs["area_km2"]
    pairs["rgi6_area_fraction"] = pairs["overlap_area_km2"] / pairs["rgi60_area_km2"]
    pairs["map_area_ratio_rgi7_to_rgi60"] = pairs["area_km2"] / pairs["rgi60_area_km2"]
    pairs["map_dist_deg"] = np.sqrt(
        (pairs["cenlon"] - pairs["rgi60_cenlon"]) ** 2
        + (pairs["cenlat"] - pairs["rgi60_cenlat"]) ** 2
    )
    pairs["n_rgi6_candidates"] = pairs.groupby("rgi_id")["rgi60_id"].transform("nunique")
    pairs["n_rgi7_candidates"] = pairs.groupby("rgi60_id")["rgi_id"].transform("nunique")

    pairs = pairs.sort_values(
        ["rgi_id", "overlap_area_km2", "rgi7_area_fraction"],
        ascending=[True, False, False],
    )
    best = pairs.drop_duplicates("rgi_id", keep="first").copy()
    best["n_rgi7"] = best.groupby("rgi60_id")["rgi_id"].transform("nunique")
    best["n_rgi6"] = best["n_rgi6_candidates"]
    best["cluster_id"] = best["rgi60_id"].factorize()[0]
    best["mapping_method"] = "spatial_overlap"
    return best[
        [
            "rgi_id",
            "rgi60_id",
            "mapping_method",
            "rgi60_glims_id",
            "rgi60_cenlon",
            "rgi60_cenlat",
            "rgi60_area_km2",
            "overlap_area_km2",
            "rgi7_area_fraction",
            "rgi6_area_fraction",
            "cluster_id",
            "n_rgi7",
            "n_rgi6",
            "map_dist_deg",
            "map_area_ratio_rgi7_to_rgi60",
        ]
    ]


def build_glims_fallback_map(attrs: pd.DataFrame) -> pd.DataFrame:
    """Build a conservative GLIMS-ID fallback for non-overlap edge cases."""
    rgi60 = read_rgi60_attributes()
    bridge = attrs.merge(
        rgi60,
        left_on="glims_id",
        right_on="rgi60_glims_id",
        how="left",
        validate="many_to_many",
    )
    bridge = bridge[bridge["rgi60_id"].notna()].copy()
    if bridge.empty:
        return bridge

    bridge["map_dist_deg"] = np.sqrt(
        (bridge["cenlon"] - bridge["rgi60_cenlon"]) ** 2
        + (bridge["cenlat"] - bridge["rgi60_cenlat"]) ** 2
    )
    bridge = bridge.sort_values(["rgi_id", "map_dist_deg"]).drop_duplicates(
        "rgi_id",
        keep="first",
    )
    bridge["mapping_method"] = "glims_id_fallback"
    bridge["map_area_ratio_rgi7_to_rgi60"] = bridge["area_km2"] / bridge["rgi60_area_km2"]
    bridge["overlap_area_km2"] = np.nan
    bridge["rgi7_area_fraction"] = np.nan
    bridge["rgi6_area_fraction"] = np.nan
    bridge["cluster_id"] = np.nan
    bridge["n_rgi7"] = np.nan
    bridge["n_rgi6"] = np.nan
    return bridge[
        [
            "rgi_id",
            "rgi60_id",
            "mapping_method",
            "rgi60_glims_id",
            "rgi60_cenlon",
            "rgi60_cenlat",
            "rgi60_area_km2",
            "overlap_area_km2",
            "rgi7_area_fraction",
            "rgi6_area_fraction",
            "cluster_id",
            "n_rgi7",
            "n_rgi6",
            "map_dist_deg",
            "map_area_ratio_rgi7_to_rgi60",
        ]
    ]


def build_rgi7_to_rgi60_map(attrs: pd.DataFrame) -> pd.DataFrame:
    """Build a conservative RGI 7.0 to RGI 6.0 mapping.

    RGI 7.0 IDs are regenerated and must not be converted numerically to
    RGI 6.0 IDs. Use official overlap links if present; otherwise reconstruct
    overlap links from the two outline products. GLIMS IDs are only a fallback.
    """
    official = build_official_overlap_map()
    if official is not None:
        print(f"  Using official RGI6 link table: {len(official):,} rows")
        return official

    print("  Official RGI6 link table is incomplete; building polygon-overlap map...")
    overlap = build_spatial_overlap_map()
    fallback = build_glims_fallback_map(attrs)
    if fallback.empty:
        return overlap

    missing = attrs.loc[~attrs["rgi_id"].isin(overlap["rgi_id"]), ["rgi_id"]]
    fallback = missing.merge(fallback, on="rgi_id", how="inner")
    out = pd.concat([overlap, fallback], ignore_index=True, sort=False)
    return out.sort_values("rgi_id")


def main() -> None:
    print("=== Step 08: Build Hugonnet Weak Labels ===")
    os.makedirs(HUGONNET_DATA_DIR, exist_ok=True)

    print("Loading RGI02 attributes...")
    attrs = pd.read_csv(RGI02_ATTRIBUTES_CSV)
    required_attr_cols = [
        "rgi_id",
        "glims_id",
        "o2region",
        "cenlon",
        "cenlat",
        "area_km2",
        "zmin_m",
        "zmax_m",
        "zmed_m",
        "term_type",
        "surge_type",
    ]
    missing_attr = [col for col in required_attr_cols if col not in attrs.columns]
    if missing_attr:
        raise RuntimeError(f"Missing RGI attribute columns: {missing_attr}")

    attrs = attrs[required_attr_cols].copy()
    print("Building RGI7-to-RGI6 mapping...")
    id_map = build_rgi7_to_rgi60_map(attrs)
    attrs = attrs.merge(id_map, on="rgi_id", how="left", validate="one_to_one")

    print("Loading Hugonnet per-glacier rates...")
    usecols = [
        "rgiid",
        "period",
        "area",
        "dhdt",
        "err_dhdt",
        "dmdtda",
        "err_dmdtda",
        "perc_area_meas",
        "perc_area_res",
        "valid_obs",
        "valid_obs_py",
        "reg",
    ]
    rates = pd.read_csv(HUGONNET_RGI02_RATES, usecols=usecols)
    rates = rates[rates["period"].eq(HUGONNET_WEAK_LABEL_PERIOD)].copy()
    if rates.empty:
        raise RuntimeError(f"No Hugonnet rows found for {HUGONNET_WEAK_LABEL_PERIOD}")

    numeric_cols = [col for col in usecols if col not in ("rgiid", "period")]
    for col in numeric_cols:
        rates[col] = pd.to_numeric(rates[col], errors="coerce")

    rates = rates.rename(
        columns={
            "rgiid": "rgi60_id",
            "area": "hugonnet_area_m2",
            "dhdt": "hugonnet_dhdt_m_yr",
            "err_dhdt": "hugonnet_err_dhdt_m_yr",
            "dmdtda": "hugonnet_dmdtda_mwe_yr",
            "err_dmdtda": "hugonnet_err_dmdtda_mwe_yr",
        }
    )

    merged = attrs.merge(rates, on="rgi60_id", how="left", validate="many_to_one")
    merged["has_hugonnet"] = merged["hugonnet_dmdtda_mwe_yr"].notna()
    overlap_available = merged["rgi7_area_fraction"].notna() & merged["rgi6_area_fraction"].notna()
    overlap_qc = (
        overlap_available
        & (merged["rgi7_area_fraction"] >= HUGONNET_MIN_RGI7_OVERLAP_FRACTION)
        & (merged["rgi6_area_fraction"] >= HUGONNET_MIN_RGI60_OVERLAP_FRACTION)
    )
    glims_qc = (
        ~overlap_available
        & merged["map_dist_deg"].notna()
        & (merged["map_dist_deg"] <= HUGONNET_MAX_MAP_DIST_DEG)
        & merged["map_area_ratio_rgi7_to_rgi60"].between(
            HUGONNET_MIN_AREA_RATIO_RGI7_TO_RGI60,
            HUGONNET_MAX_AREA_RATIO_RGI7_TO_RGI60,
        )
    )
    merged["map_qc_pass"] = merged["rgi60_id"].notna() & (overlap_qc | glims_qc)
    merged["qc_pass"] = (
        merged["map_qc_pass"]
        & merged["has_hugonnet"]
        & merged["hugonnet_err_dmdtda_mwe_yr"].notna()
        & (merged["perc_area_meas"] >= HUGONNET_MIN_PERC_AREA_MEAS)
        & (merged["perc_area_res"] >= HUGONNET_MIN_PERC_AREA_RES)
        & (merged["valid_obs_py"] >= HUGONNET_MIN_VALID_OBS_PY)
    )

    eps = 1e-6
    merged["weak_weight_inv_var"] = np.where(
        merged["qc_pass"],
        1.0 / (merged["hugonnet_err_dmdtda_mwe_yr"].clip(lower=eps) ** 2),
        np.nan,
    )
    if merged["qc_pass"].any():
        median_weight = float(merged.loc[merged["qc_pass"], "weak_weight_inv_var"].median())
        merged["weak_weight_norm"] = merged["weak_weight_inv_var"] / median_weight
        merged["weak_weight_norm"] = merged["weak_weight_norm"].clip(0.05, 20.0)
    else:
        merged["weak_weight_norm"] = np.nan

    weak = merged.loc[merged["qc_pass"]].copy()
    weak = weak[
        [
            "rgi_id",
            "rgi60_id",
            "mapping_method",
            "overlap_area_km2",
            "rgi7_area_fraction",
            "rgi6_area_fraction",
            "n_rgi7",
            "n_rgi6",
            "map_dist_deg",
            "map_area_ratio_rgi7_to_rgi60",
            "o2region",
            "cenlon",
            "cenlat",
            "glims_id",
            "area_km2",
            "hugonnet_area_m2",
            "period",
            "hugonnet_dmdtda_mwe_yr",
            "hugonnet_err_dmdtda_mwe_yr",
            "hugonnet_dhdt_m_yr",
            "hugonnet_err_dhdt_m_yr",
            "perc_area_meas",
            "perc_area_res",
            "valid_obs",
            "valid_obs_py",
            "weak_weight_norm",
            "term_type",
            "surge_type",
        ]
    ].sort_values("rgi_id")

    summary = pd.DataFrame(
        [
            {"metric": "rgi7_glaciers", "value": len(merged)},
            {"metric": "hugonnet_period", "value": HUGONNET_WEAK_LABEL_PERIOD},
            {"metric": "mapped_rgi60", "value": int(merged["rgi60_id"].notna().sum())},
            {"metric": "mapping_qc_pass", "value": int(merged["map_qc_pass"].sum())},
            {
                "metric": "spatial_overlap_mapped",
                "value": int(merged["mapping_method"].eq("spatial_overlap").sum()),
            },
            {
                "metric": "glims_fallback_mapped",
                "value": int(merged["mapping_method"].eq("glims_id_fallback").sum()),
            },
            {"metric": "matched_hugonnet", "value": int(merged["has_hugonnet"].sum())},
            {"metric": "qc_pass", "value": int(merged["qc_pass"].sum())},
            {
                "metric": "min_rgi7_overlap_fraction",
                "value": HUGONNET_MIN_RGI7_OVERLAP_FRACTION,
            },
            {
                "metric": "min_rgi60_overlap_fraction",
                "value": HUGONNET_MIN_RGI60_OVERLAP_FRACTION,
            },
            {"metric": "min_perc_area_meas", "value": HUGONNET_MIN_PERC_AREA_MEAS},
            {"metric": "min_perc_area_res", "value": HUGONNET_MIN_PERC_AREA_RES},
            {"metric": "min_valid_obs_py", "value": HUGONNET_MIN_VALID_OBS_PY},
            {
                "metric": "weak_label_mean_mwe_yr",
                "value": float(weak["hugonnet_dmdtda_mwe_yr"].mean()),
            },
            {
                "metric": "weak_label_median_mwe_yr",
                "value": float(weak["hugonnet_dmdtda_mwe_yr"].median()),
            },
            {
                "metric": "weak_label_std_mwe_yr",
                "value": float(weak["hugonnet_dmdtda_mwe_yr"].std()),
            },
        ]
    )

    merged.to_csv(HUGONNET_MATCH_QC_CSV, index=False)
    weak.to_csv(HUGONNET_WEAK_LABELS_CSV, index=False)
    summary.to_csv(HUGONNET_QC_SUMMARY_CSV, index=False)

    print(f"RGI7 glaciers: {len(merged):,}")
    print(f"Mapped to RGI6: {int(merged['rgi60_id'].notna().sum()):,}")
    print(f"Mapping QC-pass: {int(merged['map_qc_pass'].sum()):,}")
    print(f"Matched Hugonnet rows: {int(merged['has_hugonnet'].sum()):,}")
    print(f"QC-pass weak labels: {len(weak):,}")
    print(
        "Weak label mean/median/std: "
        f"{weak['hugonnet_dmdtda_mwe_yr'].mean():.3f} / "
        f"{weak['hugonnet_dmdtda_mwe_yr'].median():.3f} / "
        f"{weak['hugonnet_dmdtda_mwe_yr'].std():.3f} m w.e. yr-1"
    )
    print(f"Saved weak labels -> {HUGONNET_WEAK_LABELS_CSV}")
    print(f"Saved match QC -> {HUGONNET_MATCH_QC_CSV}")
    print(f"Saved summary -> {HUGONNET_QC_SUMMARY_CSV}")


if __name__ == "__main__":
    main()
