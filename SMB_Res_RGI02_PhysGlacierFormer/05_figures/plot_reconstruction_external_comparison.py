"""Plot reconstruction diagnostics against Hugonnet and Malles & Marzeion."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from netCDF4 import Dataset

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "04_reconstruction"))

from analyze_xgboost_v2_reconstruction import cumulative_ensemble_summary  # noqa: E402

from config import (  # noqa: E402
    FIG_DIR,
    MALLES_REGION_NC,
    PHYS_V2_GLAMBIE_ANNUAL_SERIES,
    PHYS_V2_RECONSTRUCTION_CALIBRATED_CSV,
    RECONSTRUCTION_DIR,
    RGI02_SHP,
)

OUT_PNG = os.path.join(FIG_DIR, "fig_reconstruction_external_comparison.png")
OUT_CSV = os.path.join(FIG_DIR, "regional_mass_change_comparison.csv")
OUT_METRICS_CSV = os.path.join(FIG_DIR, "regional_mass_change_comparison_metrics.csv")
OUT_GLAMBIE_PNG = os.path.join(FIG_DIR, "fig_glambie_annual_comparison.png")
HUGONNET_TRANSFER_CSV = os.path.join(
    RECONSTRUCTION_DIR, "hugonnet_temporal_transfer_xgboost_v2.csv"
)
HUGONNET_TRANSFER_CROSSFIT_CSV = os.path.join(
    RECONSTRUCTION_DIR, "hugonnet_temporal_transfer_xgboost_v2_crossfit.csv"
)
NATURAL_EARTH_DIR = r"H:\Code\SMB\Map\50m_physical"
NE_LAND_SHP = os.path.join(NATURAL_EARTH_DIR, "ne_50m_land.shp")
NE_COASTLINE_SHP = os.path.join(NATURAL_EARTH_DIR, "ne_50m_coastline.shp")
COASTLINE_SHP = os.environ.get("SMB_COASTLINE_SHP", "")

COLOR_MALLES = "#355c7d"
COLOR_RAW = "#d17a22"
COLOR_CONSERVATIVE = "#2b8c6f"
COLOR_BAND = "#b8c7d9"
COLOR_GLAMBIE = "#6f6658"

# Full RGI02 extent used by the established spatial-distribution figure.
# Wider contextual extent lets the true lon/lat aspect fill the landscape panel
# without geometrically stretching RGI02.
MAP_LON_LIMITS = (-151.0, -95.0)
MAP_LAT_LIMITS = (35.0, 61.0)
RGI_FILL = "#a8d5a2"
RGI_EDGE = "#3b7d44"


def load_malles_region02(cumulative_start: int, cumulative_end: int) -> pd.DataFrame:
    """Load Malles & Marzeion RGI02 ensemble regional mass change."""
    with Dataset(MALLES_REGION_NC) as dataset:
        region_idx = np.where(dataset.variables["Region"][:] == 2)[0][0]
        mass_change = np.array(
            dataset.variables["Mass change"][:, :, region_idx], dtype=float
        )
        unc = np.array(
            dataset.variables["Mass change uncertainty"][:, :, region_idx], dtype=float
        )
        time_range = str(getattr(dataset.variables["Time"], "range", "1901 - 2018"))

    start_year, end_year = [int(value.strip()) for value in time_range.split("-")]
    years = np.arange(start_year, end_year + 1)
    if mass_change.shape[1] != len(years):
        raise RuntimeError("Malles time metadata does not match the time dimension.")
    valid = ~np.all(np.isnan(mass_change), axis=0)
    years = years[valid]
    mass_change = mass_change[:, valid]
    unc = unc[:, valid]
    ens_mean = np.nanmean(mass_change, axis=0)
    ens_p05 = np.nanpercentile(mass_change, 5, axis=0)
    ens_p95 = np.nanpercentile(mass_change, 95, axis=0)
    unc_mean = np.nanmean(unc, axis=0)
    frame = pd.DataFrame(
        {
            "year": years,
            "malles_mass_change_gt": ens_mean,
            "malles_p05_gt": ens_p05,
            "malles_p95_gt": ens_p95,
            "malles_unc_mean_gt": unc_mean,
            "malles_n_available_forcings": np.isfinite(mass_change).sum(axis=0),
        }
    )
    period = (years >= cumulative_start) & (years <= cumulative_end)
    if not period.any() or not np.all(np.diff(years[period]) == 1):
        raise ValueError("Malles cumulative years must be nonempty and contiguous.")
    for column, values in cumulative_ensemble_summary(mass_change[:, period]).items():
        frame[column] = np.nan
        frame.loc[period, column] = values
    complete = np.isfinite(mass_change[:, period]).all(axis=1)
    frame["malles_n_complete_forcings"] = int(complete.sum())
    frame["malles_complete_cohort_annual_gt"] = np.nan
    frame.loc[period, "malles_complete_cohort_annual_gt"] = mass_change[complete][:, period].mean(axis=0)
    return frame


def load_reconstruction_region() -> pd.DataFrame:
    recon = pd.read_csv(PHYS_V2_RECONSTRUCTION_CALIBRATED_CSV)
    raw = recon.groupby("year").apply(
        lambda g: float(np.sum(g["predicted_smb_m"] * g["area_km2"] * 0.001)),
        include_groups=False,
    )
    cons = recon.groupby("year").apply(
        lambda g: float(np.sum(g["predicted_smb_conservative_m"] * g["area_km2"] * 0.001)),
        include_groups=False,
    )
    out = pd.DataFrame(
        {
            "year": raw.index.astype(int),
            "raw_mass_change_gt": raw.values,
            "conservative_mass_change_gt": cons.values,
        }
    )
    return out


def load_hugonnet_temporal_transfer() -> pd.DataFrame:
    frame = pd.read_csv(HUGONNET_TRANSFER_CSV)
    required = {"shrink_factor", "rmse_mwe_yr", "rmse_ci_low", "rmse_ci_high"}
    if not required.issubset(frame.columns):
        raise RuntimeError(f"Hugonnet temporal-transfer file lacks {sorted(required - set(frame.columns))}")
    return frame.sort_values("shrink_factor")


def load_hugonnet_crossfit() -> pd.Series:
    frame = pd.read_csv(HUGONNET_TRANSFER_CROSSFIT_CSV)
    overall = frame[frame["scope"] == "overall_cross_fitted"]
    if len(overall) != 1:
        raise RuntimeError("Expected one overall glacier-cross-fitted Hugonnet row.")
    return overall.iloc[0]


def weighted_corr_rmse(obs: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    mask = np.isfinite(obs) & np.isfinite(pred)
    obs = obs[mask]
    pred = pred[mask]
    corr = float(np.corrcoef(obs, pred)[0, 1]) if len(obs) > 1 else np.nan
    rmse = float(np.sqrt(np.mean((pred - obs) ** 2)))
    bias = float(np.mean(pred - obs))
    return corr, rmse, bias


def build_metric_rows(comp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, col in [
        ("Raw reconstruction", "raw_mass_change_gt"),
        ("Conservative calibration", "conservative_mass_change_gt"),
    ]:
        obs = comp["malles_mass_change_gt"].values
        pred = comp[col].values
        r, rmse, bias = weighted_corr_rmse(obs, pred)
        rows.append(
            {
                "comparison": f"{label} vs Malles & Marzeion",
                "start_year": int(comp["year"].min()),
                "end_year": int(comp["year"].max()),
                "n_years": int(len(comp)),
                "pearson_r": r,
                "rmse_gt": rmse,
                "bias_gt": bias,
                "mae_gt": float(np.mean(np.abs(pred - obs))),
                "cumulative_model_gt": float(np.sum(pred)),
                "cumulative_malles_gt": float(np.sum(obs)),
                "malles_annual_member_count_min": int(comp["malles_n_available_forcings"].min()),
                "malles_annual_member_count_max": int(comp["malles_n_available_forcings"].max()),
                "malles_complete_forcing_count": int(comp["malles_n_complete_forcings"].iloc[0]),
                "cumulative_malles_complete_cohort_gt": float(comp["malles_complete_cohort_annual_gt"].sum()),
            }
        )
    return pd.DataFrame(rows)


def plot_rgi02_background(ax, lon_limits: tuple[float, float], lat_limits: tuple[float, float]) -> bool:
    """Add glacier polygons in lon/lat coordinates when geopandas is available."""
    try:
        import geopandas as gpd
        from shapely.geometry import box
    except ImportError:
        return False

    if not os.path.exists(RGI02_SHP):
        return False

    try:
        bbox = box(lon_limits[0], lat_limits[0], lon_limits[1], lat_limits[1])
        glaciers = gpd.read_file(RGI02_SHP, bbox=bbox)
        if glaciers.crs is not None:
            glaciers = glaciers.to_crs("EPSG:4326")
        glaciers.plot(
            ax=ax,
            facecolor="0.90",
            edgecolor="0.72",
            linewidth=0.12,
            alpha=0.55,
            zorder=0,
        )
    except Exception:
        return False
    return True


def plot_map_base(ax, lon_limits: tuple[float, float], lat_limits: tuple[float, float]) -> bool:
    """Draw a lightweight Natural Earth base map in lon/lat coordinates."""
    try:
        import geopandas as gpd
        from shapely.geometry import box
    except ImportError:
        return False

    coastline_path = COASTLINE_SHP if COASTLINE_SHP and os.path.exists(COASTLINE_SHP) else NE_COASTLINE_SHP
    if not os.path.exists(NE_LAND_SHP) or not os.path.exists(coastline_path):
        return False

    try:
        bbox = box(lon_limits[0], lat_limits[0], lon_limits[1], lat_limits[1])
        land = gpd.read_file(NE_LAND_SHP, bbox=bbox)
        coastline = gpd.read_file(coastline_path, bbox=bbox)
        if land.crs is not None:
            land = land.to_crs("EPSG:4326")
        if coastline.crs is not None:
            coastline = coastline.to_crs("EPSG:4326")
        land.plot(
            ax=ax,
            facecolor="#f3f0e8",
            edgecolor="none",
            alpha=0.86,
            zorder=-2,
        )
        coastline.plot(
            ax=ax,
            color="0.45",
            linewidth=0.38,
            alpha=0.88,
            zorder=-1,
        )
    except Exception:
        return False
    return True


def plot_spatial_distribution_base(ax) -> tuple[bool, bool]:
    """Draw a reproducible local Natural Earth/RGI context without web tiles."""
    has_base = False
    has_rgi = False
    ax.set_xlim(*MAP_LON_LIMITS)
    ax.set_ylim(*MAP_LAT_LIMITS)
    has_base = plot_map_base(ax, MAP_LON_LIMITS, MAP_LAT_LIMITS)

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
            linewidth=0.18,
            alpha=0.58,
            zorder=0,
            rasterized=True,
        )
        has_rgi = True
    except Exception as exc:
        print(f"WARNING: RGI02 outlines unavailable ({exc}).")

    ax.set_xlim(*MAP_LON_LIMITS)
    ax.set_ylim(*MAP_LAT_LIMITS)
    mid_lat = 0.5 * (MAP_LAT_LIMITS[0] + MAP_LAT_LIMITS[1])
    ax.set_aspect(1.0 / np.cos(np.deg2rad(mid_lat)), adjustable="box")
    return has_base, has_rgi


def padded_limits(
    values: pd.Series,
    pad_fraction: float,
    quantile_clip: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Expand map limits without changing subplot geometry."""
    if quantile_clip is None:
        vmin = float(values.min())
        vmax = float(values.max())
    else:
        vmin = float(values.quantile(quantile_clip[0]))
        vmax = float(values.quantile(quantile_clip[1]))
    span = vmax - vmin
    pad = span * pad_fraction
    return vmin - pad, vmax + pad


def plot_glambie_diagnostics(frame: pd.DataFrame) -> None:
    """Show annual variability and signed errors without treating shared data as a holdout."""
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.27, top=0.90, wspace=0.28)
    years = frame["year"].to_numpy()
    reference = frame["glambie_combined_mwe"].to_numpy()
    error = frame["glambie_combined_mwe_errors"].to_numpy()
    ax = axes[0]
    ax.fill_between(years, reference - error, reference + error, color="0.7", alpha=0.3)
    ax.plot(years, reference, "o-", ms=3, color="0.2", lw=1.3, label="GlaMBIE combined")
    altimetry = frame["glambie_altimetry_annual_variability"].eq(1)
    ax.plot(years[altimetry], frame.loc[altimetry, "glambie_altimetry_mwe"],
            "s:", ms=3, color=COLOR_MALLES, lw=1, label="GlaMBIE altimetry component")
    for label, color, name in [
        ("raw", COLOR_RAW, "Raw reconstruction"),
        ("calibrated", COLOR_CONSERVATIVE, "Conservative calibration"),
    ]:
        predicted = frame[f"{label}_area_weighted_smb_all_m"].to_numpy()
        ax.plot(years, predicted, color=color, lw=1.5, label=name)
        axes[1].plot(years, predicted - reference, "o-", color=color, ms=3, lw=1.3, label=name)
    ax.set_title("(a) Regional annual specific mass change")
    ax.set_ylabel("m w.e. yr$^{-1}$")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    axes[1].set_title("(b) Reconstruction minus GlaMBIE combined")
    axes[1].set_ylabel("Residual (m w.e. yr$^{-1}$)")
    axes[1].axvspan(2019.5, 2023.5, color="#f5e2c1", alpha=0.45, zorder=0)
    axes[1].text(0.97, 0.05, "2020-2023: 4-year diagnostic\nPositive residual = underestimated mass loss",
                 transform=axes[1].transAxes, ha="right", va="bottom", fontsize=7.5)
    axes[1].legend(frameon=False, fontsize=7.5, loc="upper left")
    for ax in axes:
        ax.axhline(0, color="0.45", lw=0.7)
        ax.set_xlim(1999.5, 2023.5)
        ax.set_xticks([2000, 2005, 2010, 2015, 2020, 2023])
        ax.set_xlabel("Hydrological end year (October-September)")
    fig.text(0.5, 0.055,
             "GlaMBIE shares glaciological/geodetic inputs; comparisons assess external consistency. "
             "Grey shading: reported GlaMBIE uncertainty.\n"
             "This study uses fixed RGI v7 geometry; GlaMBIE uses evolving area. "
             "Altimetry is shown only where it provides its own annual variability.",
             ha="center", va="bottom", fontsize=7.3, color="0.35")
    fig.savefig(OUT_GLAMBIE_PNG, dpi=300)
    plt.close(fig)


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

    recon = load_reconstruction_region()
    malles = load_malles_region02(int(recon.year.min()), int(recon.year.max()))
    merged = recon.merge(malles, on="year", how="left")
    merged["raw_cumulative_gt"] = merged["raw_mass_change_gt"].cumsum()
    merged["conservative_cumulative_gt"] = merged["conservative_mass_change_gt"].cumsum()

    overlap = merged["malles_mass_change_gt"].notna()
    first_overlap_year = int(merged.loc[overlap, "year"].min())
    merged["raw_cumulative_overlap_gt"] = np.nan
    merged["conservative_cumulative_overlap_gt"] = np.nan
    merged.loc[overlap, "raw_cumulative_overlap_gt"] = merged.loc[
        overlap, "raw_mass_change_gt"
    ].cumsum()
    merged.loc[overlap, "conservative_cumulative_overlap_gt"] = merged.loc[
        overlap, "conservative_mass_change_gt"
    ].cumsum()
    mm = malles[malles["year"].between(recon["year"].min(), recon["year"].max())].copy()
    mm["malles_available_mean_cumulative_gt"] = mm["malles_mass_change_gt"].cumsum()
    complete_count = int(mm["malles_n_complete_forcings"].iloc[0])

    comp = merged[merged["malles_mass_change_gt"].notna()].copy()
    raw_r, raw_rmse, raw_bias = weighted_corr_rmse(
        comp["malles_mass_change_gt"].values,
        comp["raw_mass_change_gt"].values,
    )
    con_r, con_rmse, con_bias = weighted_corr_rmse(
        comp["malles_mass_change_gt"].values,
        comp["conservative_mass_change_gt"].values,
    )
    metrics = build_metric_rows(comp)
    overlap_start = int(comp["year"].min())
    overlap_end = int(comp["year"].max())

    transfer = load_hugonnet_temporal_transfer()
    transfer_crossfit = load_hugonnet_crossfit()
    glambie = pd.read_csv(PHYS_V2_GLAMBIE_ANNUAL_SERIES)
    offset_once = pd.read_csv(PHYS_V2_RECONSTRUCTION_CALIBRATED_CSV).drop_duplicates("rgi_id")

    fig = plt.figure(figsize=(11.2, 7.4))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.0, 1.0, 0.035],
        height_ratios=[1.0, 1.0],
        left=0.075,
        right=0.965,
        bottom=0.13,
        top=0.965,
        wspace=0.24,
        hspace=0.30,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    cax_d = fig.add_subplot(gs[1, 2])

    ax = ax_a
    ax.fill_between(
        malles["year"],
        malles["malles_p05_gt"],
        malles["malles_p95_gt"],
        color=COLOR_BAND,
        alpha=0.45,
        label="Malles & Marzeion forcing ensemble (5th-95th pct.)",
    )
    ax.plot(malles["year"], malles["malles_mass_change_gt"], color=COLOR_MALLES, lw=1.4, label="Malles & Marzeion")
    ax.plot(recon["year"], recon["raw_mass_change_gt"], color=COLOR_RAW, lw=1.3, label="Raw reconstruction")
    ax.plot(recon["year"], recon["conservative_mass_change_gt"], color=COLOR_CONSERVATIVE, lw=1.5, label="Conservative calibration")
    glambie_years = glambie["year"]
    glambie_mean = glambie["glambie_combined_gt"]
    glambie_uncertainty = glambie["glambie_combined_gt_errors"]
    ax.fill_between(
        glambie_years,
        glambie_mean - glambie_uncertainty,
        glambie_mean + glambie_uncertainty,
        color=COLOR_GLAMBIE,
        alpha=0.12,
        zorder=1,
    )
    ax.plot(
        glambie_years,
        glambie_mean,
        color=COLOR_GLAMBIE,
        lw=1.2,
        ls=(0, (4, 2)),
        label="GlaMBIE annual estimate (reported uncertainty)",
        zorder=2,
    )
    ax.axhline(0, color="0.3", lw=0.7)
    ax.set_title("(a) Regional annual mass change")
    ax.set_ylabel("Gt yr$^{-1}$")
    ax.set_xlim(recon["year"].min() - 1, recon["year"].max() + 1)
    ax.text(
        0.985,
        0.965,
        f"Overlap with Malles ({overlap_start}-{overlap_end})\n"
        f"Raw: r={raw_r:.2f}, RMSE={raw_rmse:.2f} Gt, Bias={raw_bias:.2f} Gt\n"
        f"Calibrated: r={con_r:.2f}, RMSE={con_rmse:.2f} Gt, Bias={con_bias:.2f} Gt",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="0.75", alpha=0.88),
    )
    ax.legend(frameon=False, fontsize=8, loc="lower left")

    ax = ax_b
    ax.fill_between(
        mm["year"],
        mm["malles_p05_cumulative_gt"],
        mm["malles_p95_cumulative_gt"],
        color=COLOR_BAND,
        alpha=0.45,
    )
    ax.plot(mm["year"], mm["malles_cumulative_gt"], color=COLOR_MALLES, lw=1.4,
            label=f"Malles: {complete_count} complete forcing trajectories")
    ax.plot(mm["year"], mm["malles_available_mean_cumulative_gt"], color=COLOR_MALLES,
            lw=1.0, ls="--", label="Malles: accumulated available-member annual mean")
    ax.plot(
        merged.loc[overlap, "year"],
        merged.loc[overlap, "raw_cumulative_overlap_gt"],
        color=COLOR_RAW,
        lw=1.3,
        label="Raw reconstruction",
    )
    ax.plot(
        merged.loc[overlap, "year"],
        merged.loc[overlap, "conservative_cumulative_overlap_gt"],
        color=COLOR_CONSERVATIVE,
        lw=1.5,
        label="Conservative calibration",
    )
    ax.axhline(0, color="0.3", lw=0.7)
    ax.set_title(f"(b) Cumulative mass change since {first_overlap_year}")
    ax.set_ylabel("Gt")
    ax.legend(frameon=False, fontsize=8, loc="lower left")

    ax = ax_c
    x = transfer["shrink_factor"].to_numpy()
    rmse = transfer["rmse_mwe_yr"].to_numpy()
    ax.fill_between(
        x,
        transfer["rmse_ci_low"],
        transfer["rmse_ci_high"],
        color=COLOR_CONSERVATIVE,
        alpha=0.18,
        label="95% glacier bootstrap CI",
    )
    ax.plot(x, rmse, marker="o", color=COLOR_CONSERVATIVE, lw=1.7, label="2010-2020 RMSE")
    selected_shrink = float(transfer_crossfit["selected_shrink"])
    selected_rmse = float(transfer_crossfit["rmse_mwe_yr"])
    ax.scatter(
        [selected_shrink], [selected_rmse], s=48, marker="D", color="#a51c30",
        edgecolor="white", linewidth=0.6, zorder=4,
        label=f"5-fold glacier cross-fit (shrink={selected_shrink:.2f})",
    )
    ax.axhline(rmse[0], color="0.45", lw=0.9, ls="--", label="No-calibration RMSE")
    ax.set_title("(c) Hugonnet temporal-transfer sensitivity")
    ax.set_xlabel("Shrink applied to offsets estimated from 2000-2010")
    ax.set_ylabel("2010-2020 RMSE (m w.e. yr$^{-1}$)")
    ax.set_xticks(x)
    ax.legend(frameon=False, fontsize=8)

    ax = ax_d
    has_map_base, has_background = plot_spatial_distribution_base(ax)
    point_sizes = np.clip(
        10.0 * np.log10(offset_once["area_km2"].clip(lower=0.0) + 1.0) + 3.0,
        3.0,
        34.0,
    )
    offset_norm = mcolors.TwoSlopeNorm(vmin=-0.5, vcenter=0.0, vmax=0.5)
    sc = ax.scatter(
        offset_once["cenlon"],
        offset_once["cenlat"],
        c=offset_once["calibration_offset_conservative_mwe_yr"],
        s=point_sizes,
        cmap="RdBu_r",
        norm=offset_norm,
        alpha=0.78,
        linewidths=0.10,
        edgecolors="0.20",
        zorder=3,
        rasterized=True,
    )
    ax.set_title("(d) Spatial distribution of conservative offsets")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(*MAP_LON_LIMITS)
    ax.set_ylim(*MAP_LAT_LIMITS)
    ax.grid(color="white", lw=0.42, linestyle="-", alpha=0.72, zorder=1)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    cbar = fig.colorbar(sc, cax=cax_d)
    cbar.set_label("Offset (m w.e. yr$^{-1}$)")
    fig.text(
        0.5,
        0.025,
        "Fixed RGI v7 area in this study; evolving area in external products. GlaMBIE shares WGMS/Hugonnet inputs.\n"
        f"Malles shading: 5th-95th forcing percentiles; (a) available members, (b) {complete_count} complete trajectories. Neither is total uncertainty.",
        ha="center",
        va="bottom",
        fontsize=7.2,
        color="0.35",
    )
    try:
        fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
        merged.to_csv(OUT_CSV, index=False)
        metrics.to_csv(OUT_METRICS_CSV, index=False)
    except PermissionError as exc:
        raise PermissionError(
            "Could not write figure outputs. Close any open image/CSV preview "
            f"and rerun the script. Target directory: {FIG_DIR}"
        ) from exc
    print(f"Saved figure -> {OUT_PNG}")
    print(f"Saved comparison data -> {OUT_CSV}")
    print(f"Saved comparison metrics -> {OUT_METRICS_CSV}")
    plot_glambie_diagnostics(glambie)
    print(f"Saved GlaMBIE diagnostics -> {OUT_GLAMBIE_PNG}")
    if not has_map_base:
        print(
            "Natural Earth land/coastline was not drawn. Check the local map files "
            f"under {NATURAL_EARTH_DIR}, or set SMB_COASTLINE_SHP."
        )
    print(
        "Malles overlap annual comparison: "
        f"raw r={raw_r:.3f}, rmse={raw_rmse:.2f} Gt, bias={raw_bias:.2f} Gt; "
        f"conservative r={con_r:.3f}, rmse={con_rmse:.2f} Gt, bias={con_bias:.2f} Gt"
    )


if __name__ == "__main__":
    main()
