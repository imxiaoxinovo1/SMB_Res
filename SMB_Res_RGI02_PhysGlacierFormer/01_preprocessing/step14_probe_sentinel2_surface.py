"""Small, polygon-masked Sentinel-2 feasibility pilot; not training features."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import requests
from matplotlib.ticker import FuncFormatter
from rasterio.enums import Resampling
from rasterio.features import geometry_mask, geometry_window
from rasterio.vrt import WarpedVRT
from requests.adapters import HTTPAdapter
from shapely.geometry import mapping
from urllib3.util.retry import Retry

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
from config import DATA_DIR, FIG_DIR, PHYS_V2_TERRAIN_CSV, RGI02_SHP  # noqa: E402

API = "https://earth-search.aws.element84.com/v1/search"
COLLECTION = "sentinel-2-c1-l2a"


def check_radiometry(asset, scale, offset, nodata):
    """Reject ambiguous scaling rather than guessing from scene brightness."""
    metadata = asset["raster:bands"][0]
    if nodata != metadata["nodata"] or not np.allclose(
        [scale, offset], [metadata["scale"], metadata["offset"]], rtol=0, atol=1e-10
    ):
        raise ValueError("COG and STAC radiometry disagree; scene is not safe to use.")


def scaled_reflectance(raw: np.ndarray, asset: dict) -> np.ndarray:
    metadata = asset["raster:bands"][0]
    scale, offset = float(metadata["scale"]), float(metadata["offset"])
    if not np.isfinite([scale, offset]).all() or scale <= 0:
        raise ValueError("Missing or invalid asset reflectance scaling.")
    values = np.asarray(raw, dtype=float)
    valid = np.isfinite(values) & (values != float(metadata["nodata"]))
    return np.where(valid, values * scale + offset, np.nan)


def ndsi_with_quality(green, swir, scl, inside):
    # SCL 11 includes both snow and ice; it is not a snow-only reference label.
    valid = inside & np.isin(scl, [4, 5, 11]) & np.isfinite(green) & np.isfinite(swir)
    valid &= (green >= 0) & (swir >= 0) & ((green + swir) > 1e-6)
    ndsi = np.full(green.shape, np.nan, dtype=np.float32)
    np.divide(green - swir, green + swir, out=ndsi, where=valid)
    return ndsi, valid


def search_scenes(session, bounds, start, end):
    params = {"collections": COLLECTION, "bbox": ",".join(map(str, bounds)),
              "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z", "limit": 100}
    response = session.get(API, params=params, timeout=(10, 30))
    response.raise_for_status()
    payload = response.json()
    # Earth Search may supply a trailing next link even on the final page.
    if payload.get("numberMatched", len(payload["features"])) > len(payload["features"]):
        raise ValueError("Pilot search exceeds one page; narrow the date interval.")
    return sorted(payload["features"], key=lambda item: (
        item["properties"].get("eo:cloud_cover", 100), item["id"]
    ))


def read_scene(item, glacier):
    if item.get("collection") != COLLECTION:
        raise ValueError("Only Collection 1 is supported; legacy COG scaling is ambiguous.")
    assets = item["assets"]
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                      CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif", GDAL_HTTP_TIMEOUT="30",
                      GDAL_HTTP_MAX_RETRY="2", GDAL_HTTP_RETRY_DELAY="1"):
        with rasterio.open(assets["scl"]["href"]) as source:
            projected = glacier.to_crs(source.crs)
            polygon = projected.geometry.iloc[0]
            window = geometry_window(source, [mapping(polygon)])
            transform = source.window_transform(window)
            scl = source.read(1, window=window)
            crs = source.crs
        inside = geometry_mask([mapping(polygon)], scl.shape, transform, invert=True)
        if not inside.any():
            raise ValueError("No pixel centers inside glacier polygon.")
        bands = []
        for name in ["green", "swir16"]:
            with rasterio.open(assets[name]["href"]) as source:
                check_radiometry(assets[name], source.scales[0], source.offsets[0], source.nodata)
                with WarpedVRT(source, crs=crs, transform=transform,
                               width=scl.shape[1], height=scl.shape[0],
                               resampling=Resampling.average, src_nodata=0,
                               nodata=np.nan, dtype="float32") as vrt:
                    raw = vrt.read(1)
            bands.append(scaled_reflectance(raw, assets[name]))
    green, swir = bands
    ndsi, valid = ndsi_with_quality(green, swir, scl, inside)
    pixel_area = abs(transform.a * transform.e - transform.b * transform.d)
    count = int(inside.sum())
    valid_count = int(valid.sum())
    footprint = float(count * pixel_area / polygon.area)
    valid_fraction = valid_count / count
    stats = {
        "polygon_pixels": count, "valid_pixels": valid_count,
        "reference_outline_coverage": footprint, "valid_fraction": valid_fraction,
        "ndsi_median": float(np.nanmedian(ndsi)) if valid_count else np.nan,
        "ndsi_p10": float(np.nanpercentile(ndsi, 10)) if valid_count else np.nan,
        "ndsi_p90": float(np.nanpercentile(ndsi, 90)) if valid_count else np.nan,
        "green_reflectance_median": float(np.median(green[valid])) if valid_count else np.nan,
        "snow_or_ice_scl_fraction_of_polygon": float(np.mean(scl[inside] == 11)),
        "cloud_cirrus_fraction_of_polygon": float(np.mean(np.isin(scl[inside], [8, 9, 10]))),
        "shadow_fraction_of_polygon": float(np.mean(np.isin(scl[inside], [2, 3]))),
        "water_fraction_of_polygon": float(np.mean(scl[inside] == 6)),
        "negative_reflectance_fraction_of_polygon": float(np.mean(
            (green[inside] < 0) | (swir[inside] < 0)
        )),
        "qc_pass": bool(footprint >= 0.95 and valid_fraction >= 0.70 and valid_count >= 20),
        "crs": str(crs), "pixel_size_m": float(abs(transform.a)),
    }
    return stats, (ndsi, inside, transform, projected)


def plot_pilot(selected, path):
    if not selected:
        # This path belongs to this script; do not leave a stale successful plot.
        if os.path.isfile(path):
            os.remove(path)
        return
    fig, axes = plt.subplots(1, len(selected), figsize=(4.0 * len(selected), 5.0), squeeze=False)
    fig.subplots_adjust(left=0.065, right=0.90, top=0.83, bottom=0.20, wspace=0.32)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#d9d9d9")
    for ax, (row, arrays) in zip(axes[0], selected.values()):
        ndsi, inside, transform, polygon = arrays
        height, width = ndsi.shape
        extent = [transform.c, transform.c + transform.a * width,
                  transform.f + transform.e * height, transform.f]
        display = np.ma.masked_invalid(ndsi)
        ax.set_facecolor("white")
        artist = ax.imshow(display, extent=extent, cmap=cmap, vmin=-1, vmax=1,
                           interpolation="nearest", alpha=inside.astype(float))
        polygon.boundary.plot(ax=ax, color="0.25", linewidth=0.55)
        ax.set_xlim(extent[:2])
        ax.set_ylim(extent[2:])
        ax.set_aspect("equal")
        ax.set_title(f"{row['name']}\n{row['date']} | usable pixels: {row['valid_fraction']:.0%}", fontsize=10)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:.1f}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:.1f}"))
        ax.set_xlabel("UTM easting (km)")
        ax.set_ylabel("UTM northing (km)")
    colorbar = fig.colorbar(artist, cax=fig.add_axes([0.925, 0.27, 0.014, 0.45]))
    colorbar.set_label("NDSI (green - SWIR) / (green + SWIR)")
    fig.text(0.5, 0.045,
             "Sentinel-2 Collection 1 L2A, 20 m grid; fixed RGI v7 outlines. Grey: masked retrievals.\n"
             "Snow/ice spectral diagnostic only: not broadband albedo, snowline, or glacier-wide snow fraction.",
             ha="center", fontsize=9, color="0.35")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glacier-ids", nargs="+", type=int, default=[57, 205, 10498])
    parser.add_argument("--start", default="2023-08-01")
    parser.add_argument("--end", default="2023-08-31")
    parser.add_argument("--scenes-per-glacier", type=int, default=2)
    args = parser.parse_args()
    if pd.Timestamp(args.end) < pd.Timestamp(args.start) or args.scenes_per_glacier < 1:
        raise ValueError("Invalid date interval or scene count.")
    terrain = pd.read_csv(PHYS_V2_TERRAIN_CSV)
    terrain = terrain[terrain.glacier_id.isin(args.glacier_ids)].copy()
    if set(terrain.glacier_id) != set(args.glacier_ids):
        raise ValueError("Requested glaciers are absent from corrected training terrain.")
    polygons = gpd.read_file(RGI02_SHP, columns=["rgi_id", "geometry"])
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=Retry(
        total=2, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"]
    )))
    rows, provenance, best = [], [], {}
    for _, glacier in terrain.iterrows():
        geometry = polygons[polygons.rgi_id == glacier.rgi_id].to_crs("EPSG:4326")
        if len(geometry) != 1:
            raise ValueError(f"Expected one RGI polygon for {glacier.rgi_id}")
        scenes = search_scenes(session, geometry.total_bounds, args.start, args.end)
        print(f"{glacier['name']}: {len(scenes)} candidate scenes", flush=True)
        if not scenes:
            rows.append({"glacier_id": int(glacier.glacier_id), "rgi_id": glacier.rgi_id,
                         "name": glacier['name'], "status": "no_scenes", "qc_pass": False})
        for item in scenes[:args.scenes_per_glacier]:
            row = {"glacier_id": int(glacier.glacier_id), "rgi_id": glacier.rgi_id,
                   "name": glacier['name'], "scene_id": item["id"],
                   "date": item["properties"]["datetime"][:10],
                   "tile_cloud_percent": item["properties"].get("eo:cloud_cover")}
            provenance.append({**row, "collection": item["collection"],
                               "properties": item["properties"],
                               "assets": {key: item["assets"][key] for key in ["green", "swir16", "scl"]}})
            try:
                stats, arrays = read_scene(item, geometry)
                row.update(stats, status="ok")
                print(f"  {item['id']}: usable={stats['valid_fraction']:.1%}, QC={stats['qc_pass']}", flush=True)
                if stats["qc_pass"] and (glacier.rgi_id not in best or stats["valid_fraction"] > best[glacier.rgi_id][0]["valid_fraction"]):
                    best[glacier.rgi_id] = (row, arrays)
            except (rasterio.errors.RasterioError, ValueError, KeyError) as exc:
                row.update(status="failed", error=str(exc), qc_pass=False)
                print(f"  Failed {item['id']}: {exc}", flush=True)
            rows.append(row)
    output_dir = os.path.join(DATA_DIR, "remote_sensing")
    os.makedirs(output_dir, exist_ok=True)
    tag = f"sentinel2_pilot_{args.start.replace('-', '')}_{args.end.replace('-', '')}"
    if sorted(args.glacier_ids) != [57, 205, 10498] or args.scenes_per_glacier != 2:
        tag += f"_g{'-'.join(map(str, sorted(args.glacier_ids)))}_n{args.scenes_per_glacier}"
    csv_path = os.path.join(output_dir, f"{tag}_qc.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    with open(os.path.join(output_dir, f"{tag}_sources.json"), "w", encoding="utf-8") as stream:
        json.dump({"retrieved_utc": datetime.now(timezone.utc).isoformat(), "api": API,
                   "collection": COLLECTION,
                   "arguments": vars(args), "scenes": provenance}, stream, indent=2)
    os.makedirs(FIG_DIR, exist_ok=True)
    plot_pilot(best, os.path.join(FIG_DIR, f"fig_{tag}.png"))
    print(f"QC saved -> {csv_path}")
    if not best:
        raise RuntimeError("No glacier had a QC-pass scene; do not use the pilot as model input.")


if __name__ == "__main__":
    main()
