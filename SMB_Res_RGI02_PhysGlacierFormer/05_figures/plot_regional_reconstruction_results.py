"""Plot temporal and spatial patterns of the final RGI02 SMB reconstruction."""
from __future__ import annotations

import os
import sys

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import linregress


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from config import (  # noqa: E402
    FIG_DIR,
    HYPSOMETRY_RECON_CONSERVATIVE_CSV,
    RGI02_SHP,
)


OUT_PNG = os.path.join(FIG_DIR, "fig_regional_reconstruction_results.png")
NATURAL_EARTH_DIR = r"H:\Code\SMB\Map\50m_physical"
NE_LAND_SHP = os.path.join(NATURAL_EARTH_DIR, "ne_50m_land.shp")
NE_COASTLINE_SHP = os.path.join(NATURAL_EARTH_DIR, "ne_50m_coastline.shp")

MAP_LON_LIMITS = (-151.0, -95.0)
MAP_LAT_LIMITS = (35.0, 61.0)
MAP_YEAR_MIN = 1980
MAP_YEAR_MAX = 2024

RGI_FILL = "#a8d5a2"
RGI_EDGE = "#3b7d44"
NEGATIVE_COLOR = "#c96b55"
POSITIVE_COLOR = "#4f91bd"
SMOOTH_COLOR = "#263f5b"

PERIODS = [
    (1950, 1979, "1950–1979"),
    (1980, 1999, "1980–1999"),
    (2000, 2024, "2000–2024"),
]


def load_reconstruction() -> pd.DataFrame:
    required = [
        "rgi_id",
        "year",
        "predicted_smb_conservative_m",
        "area_km2",
        "cenlat",
        "cenlon",
    ]
    frame = pd.read_csv(HYPSOMETRY_RECON_CONSERVATIVE_CSV, usecols=required)
    if frame.duplicated(["rgi_id", "year"]).any():
        raise RuntimeError("Reconstruction contains duplicate glacier-year rows.")
    if frame[required].isna().any().any():
        raise RuntimeError("Reconstruction contains missing values in required columns.")
    return frame


def area_weighted_mean(group: pd.DataFrame) -> float:
    return float(
        np.average(
            group["predicted_smb_conservative_m"],
            weights=group["area_km2"],
        )
    )


def build_regional_series(recon: pd.DataFrame) -> pd.DataFrame:
    regional = (
        recon.groupby("year", sort=True)
        .apply(area_weighted_mean, include_groups=False)
        .rename("area_weighted_smb_m")
        .reset_index()
    )
    regional["rolling_11yr_m"] = regional["area_weighted_smb_m"].rolling(
        11,
        center=True,
        min_periods=6,
    ).mean()
    return regional


def build_glacier_stats(recon: pd.DataFrame) -> pd.DataFrame:
    subset = recon[recon["year"].between(MAP_YEAR_MIN, MAP_YEAR_MAX)].copy()
    rows = []
    for rgi_id, group in subset.groupby("rgi_id", sort=False):
        group = group.sort_values("year")
        trend = linregress(
            group["year"].to_numpy(dtype=float),
            group["predicted_smb_conservative_m"].to_numpy(dtype=float),
        )
        first = group.iloc[0]
        rows.append(
            {
                "rgi_id": rgi_id,
                "mean_smb_m": float(group["predicted_smb_conservative_m"].mean()),
                "trend_m_decade": float(trend.slope * 10.0),
                "trend_p": float(trend.pvalue),
                "area_km2": float(first["area_km2"]),
                "cenlon": float(first["cenlon"]),
                "cenlat": float(first["cenlat"]),
                "n_years": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def plot_natural_earth_fallback(ax) -> bool:
    try:
        import geopandas as gpd
        from shapely.geometry import box
    except ImportError:
        return False
    if not os.path.exists(NE_LAND_SHP) or not os.path.exists(NE_COASTLINE_SHP):
        return False
    try:
        bbox = box(
            MAP_LON_LIMITS[0],
            MAP_LAT_LIMITS[0],
            MAP_LON_LIMITS[1],
            MAP_LAT_LIMITS[1],
        )
        land = gpd.read_file(NE_LAND_SHP, bbox=bbox).to_crs("EPSG:4326")
        coast = gpd.read_file(NE_COASTLINE_SHP, bbox=bbox).to_crs("EPSG:4326")
        land.plot(ax=ax, facecolor="#f3f0e8", edgecolor="none", zorder=-3)
        coast.plot(ax=ax, color="0.48", linewidth=0.35, zorder=-2)
        return True
    except Exception:
        return False


def add_map_context(ax) -> None:
    ax.set_xlim(*MAP_LON_LIMITS)
    ax.set_ylim(*MAP_LAT_LIMITS)
    try:
        import contextily as ctx

        ctx.add_basemap(
            ax,
            crs="EPSG:4326",
            source=ctx.providers.CartoDB.Positron,
            zoom=5,
            attribution=False,
            zorder=-3,
        )
    except Exception as exc:
        print(f"WARNING: CartoDB basemap unavailable ({exc}); using local fallback.")
        plot_natural_earth_fallback(ax)

    try:
        import geopandas as gpd

        glaciers = gpd.read_file(RGI02_SHP, columns=["geometry"])
        if glaciers.crs is not None:
            glaciers = glaciers.to_crs("EPSG:4326")
        glaciers["geometry"] = glaciers.geometry.simplify(0.003)
        glaciers.plot(
            ax=ax,
            facecolor=RGI_FILL,
            edgecolor=RGI_EDGE,
            linewidth=0.16,
            alpha=0.52,
            zorder=0,
            rasterized=True,
        )
    except Exception as exc:
        print(f"WARNING: RGI02 outlines unavailable ({exc}).")

    ax.set_xlim(*MAP_LON_LIMITS)
    ax.set_ylim(*MAP_LAT_LIMITS)
    mid_lat = 0.5 * (MAP_LAT_LIMITS[0] + MAP_LAT_LIMITS[1])
    ax.set_aspect(1.0 / np.cos(np.deg2rad(mid_lat)), adjustable="box")
    ax.grid(color="white", linewidth=0.4, alpha=0.7, zorder=1)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")


def plot_regional_series(ax, regional: pd.DataFrame) -> None:
    values = regional["area_weighted_smb_m"]
    colors = np.where(values >= 0, POSITIVE_COLOR, NEGATIVE_COLOR)
    ax.bar(regional["year"], values, color=colors, width=0.82, alpha=0.78, linewidth=0)
    ax.plot(
        regional["year"],
        regional["rolling_11yr_m"],
        color=SMOOTH_COLOR,
        linewidth=2.0,
        label="11-year moving mean",
    )
    ax.axhline(0, color="0.25", linewidth=0.8)

    recent = regional[regional["year"].between(MAP_YEAR_MIN, MAP_YEAR_MAX)]
    trend = linregress(recent["year"], recent["area_weighted_smb_m"])
    long_mean = float(values.mean())
    recent_mean = float(recent["area_weighted_smb_m"].mean())
    annotation = (
        f"1950–2024 mean: {long_mean:+.2f} m w.e. yr$^{{-1}}$\n"
        f"1980–2024 mean: {recent_mean:+.2f} m w.e. yr$^{{-1}}$\n"
        f"1980–2024 trend: {trend.slope * 10:+.2f} m w.e. decade$^{{-1}}$"
        f" (p={trend.pvalue:.3f})"
    )
    ax.text(
        0.015,
        0.035,
        annotation,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.75", alpha=0.90),
    )
    ax.set_title("(a) Area-weighted regional annual SMB")
    ax.set_ylabel("SMB (m w.e. yr$^{-1}$)")
    ax.set_xlim(regional["year"].min() - 1, regional["year"].max() + 1)
    ax.legend(frameon=False, fontsize=8, loc="upper right")


def plot_spatial_panel(
    ax,
    fig,
    stats: pd.DataFrame,
    value_col: str,
    title: str,
    colorbar_label: str,
    norm,
    significant_outline: bool = False,
) -> None:
    add_map_context(ax)
    sizes = np.clip(
        10.0 * np.log10(stats["area_km2"].clip(lower=0.0) + 1.0) + 3.0,
        3.0,
        34.0,
    )
    scatter = ax.scatter(
        stats["cenlon"],
        stats["cenlat"],
        c=stats[value_col],
        s=sizes,
        cmap="RdBu",
        norm=norm,
        alpha=0.80,
        edgecolors="0.25",
        linewidths=0.10,
        zorder=3,
        rasterized=True,
    )
    if significant_outline:
        sig = stats["trend_p"] < 0.05
        ax.scatter(
            stats.loc[sig, "cenlon"],
            stats.loc[sig, "cenlat"],
            s=sizes[sig] + 2.0,
            facecolors="none",
            edgecolors="black",
            linewidths=0.30,
            alpha=0.55,
            zorder=4,
            rasterized=True,
        )
        ax.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markerfacecolor="none",
                    markeredgecolor="black",
                    markeredgewidth=0.7,
                    label="Linear trend p < 0.05",
                )
            ],
            loc="lower left",
            fontsize=7,
            frameon=True,
            framealpha=0.86,
        )
    ax.set_title(title)
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.037, pad=0.018, extend="both")
    colorbar.set_label(colorbar_label, fontsize=8.5)
    colorbar.ax.tick_params(labelsize=7.5)


def plot_period_distributions(ax, recon: pd.DataFrame) -> None:
    distributions = []
    weighted_means = []
    for start, end, _ in PERIODS:
        subset = recon[recon["year"].between(start, end)]
        per_glacier = subset.groupby("rgi_id", sort=False).agg(
            period_mean=("predicted_smb_conservative_m", "mean"),
            area_km2=("area_km2", "first"),
        )
        distributions.append(per_glacier["period_mean"].to_numpy())
        weighted_means.append(
            float(np.average(per_glacier["period_mean"], weights=per_glacier["area_km2"]))
        )

    violin = ax.violinplot(
        distributions,
        positions=np.arange(1, len(PERIODS) + 1),
        widths=0.78,
        showmeans=False,
        showmedians=True,
        showextrema=False,
        quantiles=[[0.25, 0.75] for _ in PERIODS],
    )
    colors = ["#9ecae1", "#74a9cf", "#2b8cbe"]
    for body, color in zip(violin["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("#315a7d")
        body.set_alpha(0.72)
    violin["cmedians"].set_color("#303030")
    violin["cmedians"].set_linewidth(1.1)
    violin["cquantiles"].set_color("#565656")
    violin["cquantiles"].set_linewidth(0.8)

    ax.scatter(
        np.arange(1, len(PERIODS) + 1),
        weighted_means,
        marker="D",
        s=40,
        color="#b2182b",
        edgecolor="white",
        linewidth=0.6,
        zorder=5,
        label="Area-weighted mean",
    )
    ax.axhline(0, color="0.3", linewidth=0.8)
    ax.set_xticks(np.arange(1, len(PERIODS) + 1))
    ax.set_xticklabels([label for _, _, label in PERIODS])
    ax.set_ylabel("Per-glacier mean SMB (m w.e. yr$^{-1}$)")
    ax.set_title("(d) Shift in glacier-wide SMB distributions")
    ax.grid(axis="y", color="0.88", linewidth=0.5)
    ax.legend(frameon=False, fontsize=8, loc="upper right")


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
        }
    )

    recon = load_reconstruction()
    regional = build_regional_series(recon)
    stats = build_glacier_stats(recon)

    mean_limit = max(
        0.5,
        float(np.ceil(np.nanquantile(np.abs(stats["mean_smb_m"]), 0.99) * 10.0) / 10.0),
    )
    trend_limit = max(
        0.1,
        float(np.ceil(np.nanquantile(np.abs(stats["trend_m_decade"]), 0.99) * 20.0) / 20.0),
    )
    mean_norm = mcolors.TwoSlopeNorm(vmin=-mean_limit, vcenter=0.0, vmax=mean_limit)
    trend_norm = mcolors.TwoSlopeNorm(vmin=-trend_limit, vcenter=0.0, vmax=trend_limit)

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.0))
    fig.subplots_adjust(left=0.07, right=0.965, bottom=0.08, top=0.96, wspace=0.24, hspace=0.30)

    plot_regional_series(axes[0, 0], regional)
    plot_spatial_panel(
        axes[0, 1],
        fig,
        stats,
        "mean_smb_m",
        f"(b) Mean glacier-wide SMB ({MAP_YEAR_MIN}–{MAP_YEAR_MAX})",
        "Mean SMB (m w.e. yr$^{-1}$)",
        mean_norm,
    )
    plot_spatial_panel(
        axes[1, 0],
        fig,
        stats,
        "trend_m_decade",
        f"(c) Glacier-wide SMB trend ({MAP_YEAR_MIN}–{MAP_YEAR_MAX})",
        "Trend (m w.e. decade$^{-1}$)",
        trend_norm,
    )
    plot_period_distributions(axes[1, 1], recon)

    fig.text(
        0.5,
        0.018,
        "Conservative reconstruction; mass conversion and regional weighting use fixed RGI v7 glacier areas.",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="0.38",
    )
    fig.savefig(OUT_PNG, dpi=400, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)

    print(f"Glaciers: {stats['rgi_id'].nunique():,}")
    print(f"Years: {int(recon['year'].min())}-{int(recon['year'].max())}")
    print(f"Mean-map color limit: ±{mean_limit:.2f} m w.e. yr-1")
    print(f"Trend-map color limit: ±{trend_limit:.2f} m w.e. decade-1")
    print(f"Saved regional reconstruction figure -> {OUT_PNG}")


if __name__ == "__main__":
    main()
