"""Compare glacier-wise reconstruction means with Hugonnet geodetic rates.

The comparison uses one non-overlapping 2000-2019 mean per glacier. The raw
reconstruction provides an external geodetic consistency check. Results after
Hugonnet calibration are constraint-consistency diagnostics, not independent
validation.
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from config import (  # noqa: E402
    FIG_DIR,
    HUGONNET_MULTIPERIOD_LABELS_CSV,
    HYPSOMETRY_RECON_CONSERVATIVE_CSV,
    HYPSOMETRY_RECON_RAW_CSV,
)
from plot_regional_reconstruction_results import add_map_context  # noqa: E402


COMPARISON_PERIOD = "2000-01-01_2020-01-01"
YEAR_MIN = 2000
YEAR_MAX = 2019

OUT_PNG = os.path.join(FIG_DIR, "fig_glacierwise_hugonnet_comparison.png")
OUT_CSV = os.path.join(FIG_DIR, "glacierwise_hugonnet_comparison_2000_2019.csv")
OUT_METRICS_CSV = os.path.join(
    FIG_DIR,
    "glacierwise_hugonnet_comparison_metrics.csv",
)

COLOR_RAW = "#d17a22"
COLOR_CALIBRATED = "#2b8c6f"
COLOR_REFERENCE = "#263f5b"


def load_period_means(
    path: str,
    value_column: str,
    output_column: str,
) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["rgi_id", "year", value_column])
    frame = frame[frame["year"].between(YEAR_MIN, YEAR_MAX)].copy()
    grouped = (
        frame.groupby("rgi_id", as_index=False)
        .agg(
            **{
                output_column: (value_column, "mean"),
                f"n_years_{output_column}": ("year", "nunique"),
            }
        )
    )
    return grouped


def build_comparison() -> pd.DataFrame:
    label_columns = [
        "rgi_id",
        "period",
        "hugonnet_dmdtda_mwe_yr",
        "hugonnet_err_dmdtda_mwe_yr",
        "area_km2",
        "cenlon",
        "cenlat",
        "perc_area_meas",
        "perc_area_res",
        "valid_obs_py",
        "mapping_method",
        "o2region",
        "zmean_m",
        "map_qc_pass",
        "qc_pass",
    ]
    labels = pd.read_csv(HUGONNET_MULTIPERIOD_LABELS_CSV, usecols=label_columns)
    labels = labels[
        (labels["period"] == COMPARISON_PERIOD)
        & labels["qc_pass"].fillna(False)
        & labels["map_qc_pass"].fillna(False)
    ].copy()
    if labels["rgi_id"].duplicated().any():
        duplicates = labels.loc[labels["rgi_id"].duplicated(), "rgi_id"].nunique()
        raise RuntimeError(
            f"Selected Hugonnet period contains {duplicates} duplicate glacier IDs."
        )

    raw = load_period_means(
        HYPSOMETRY_RECON_RAW_CSV,
        "predicted_smb_m",
        "raw_period_mean_mwe_yr",
    )
    calibrated = load_period_means(
        HYPSOMETRY_RECON_CONSERVATIVE_CSV,
        "predicted_smb_conservative_m",
        "calibrated_period_mean_mwe_yr",
    )
    comparison = labels.merge(raw, on="rgi_id", how="inner").merge(
        calibrated,
        on="rgi_id",
        how="inner",
    )

    expected_years = YEAR_MAX - YEAR_MIN + 1
    comparison = comparison[
        (comparison["n_years_raw_period_mean_mwe_yr"] == expected_years)
        & (
            comparison["n_years_calibrated_period_mean_mwe_yr"]
            == expected_years
        )
    ].copy()
    required = [
        "hugonnet_dmdtda_mwe_yr",
        "hugonnet_err_dmdtda_mwe_yr",
        "raw_period_mean_mwe_yr",
        "calibrated_period_mean_mwe_yr",
        "area_km2",
        "cenlon",
        "cenlat",
    ]
    comparison = comparison.dropna(subset=required).reset_index(drop=True)
    if comparison.empty:
        raise RuntimeError("No complete glacier-wise Hugonnet comparisons were found.")

    comparison["raw_residual_mwe_yr"] = (
        comparison["raw_period_mean_mwe_yr"]
        - comparison["hugonnet_dmdtda_mwe_yr"]
    )
    comparison["calibrated_residual_mwe_yr"] = (
        comparison["calibrated_period_mean_mwe_yr"]
        - comparison["hugonnet_dmdtda_mwe_yr"]
    )
    comparison["raw_within_hugonnet_1sigma"] = (
        comparison["raw_residual_mwe_yr"].abs()
        <= comparison["hugonnet_err_dmdtda_mwe_yr"]
    )
    comparison["calibrated_within_hugonnet_1sigma"] = (
        comparison["calibrated_residual_mwe_yr"].abs()
        <= comparison["hugonnet_err_dmdtda_mwe_yr"]
    )
    return comparison


def calculate_metrics(
    comparison: pd.DataFrame,
    prediction_column: str,
    role: str,
) -> dict[str, float | int | str]:
    observed = comparison["hugonnet_dmdtda_mwe_yr"].to_numpy(dtype=float)
    predicted = comparison[prediction_column].to_numpy(dtype=float)
    residual = predicted - observed
    denominator = np.sum((observed - observed.mean()) ** 2)
    r2 = 1.0 - np.sum(residual**2) / denominator
    within_sigma = np.mean(
        np.abs(residual)
        <= comparison["hugonnet_err_dmdtda_mwe_yr"].to_numpy(dtype=float)
    )
    return {
        "role": role,
        "period": f"{YEAR_MIN}-{YEAR_MAX}",
        "n_glaciers": len(comparison),
        "pearson_r": float(np.corrcoef(observed, predicted)[0, 1]),
        "r2": float(r2),
        "rmse_mwe_yr": float(np.sqrt(np.mean(residual**2))),
        "mae_mwe_yr": float(np.mean(np.abs(residual))),
        "bias_model_minus_hugonnet_mwe_yr": float(np.mean(residual)),
        "within_hugonnet_1sigma_fraction": float(within_sigma),
    }


def common_scatter_limits(comparison: pd.DataFrame) -> tuple[float, float]:
    values = np.concatenate(
        [
            comparison["hugonnet_dmdtda_mwe_yr"].to_numpy(),
            comparison["raw_period_mean_mwe_yr"].to_numpy(),
            comparison["calibrated_period_mean_mwe_yr"].to_numpy(),
        ]
    )
    # Keep the dense glacier population readable while retaining all rows in
    # exported data and metrics. Axis-excluded extremes are reported per panel.
    lower = np.floor(np.nanquantile(values, 0.001) * 4.0) / 4.0
    upper = np.ceil(np.nanquantile(values, 0.999) * 4.0) / 4.0
    padding = 0.04 * (upper - lower)
    return float(lower - padding), float(upper + padding)


def plot_density_scatter(
    ax,
    comparison: pd.DataFrame,
    prediction_column: str,
    metric: dict[str, float | int | str],
    title: str,
    limits: tuple[float, float],
    calibrated: bool,
) -> None:
    observed = comparison["hugonnet_dmdtda_mwe_yr"]
    predicted = comparison[prediction_column]
    outside_axes = int(
        (
            (observed < limits[0])
            | (observed > limits[1])
            | (predicted < limits[0])
            | (predicted > limits[1])
        ).sum()
    )
    ax.hexbin(
        observed,
        predicted,
        gridsize=48,
        mincnt=1,
        bins="log",
        cmap="Blues",
        linewidths=0.0,
        extent=(limits[0], limits[1], limits[0], limits[1]),
        rasterized=True,
    )
    ax.plot(limits, limits, color=COLOR_REFERENCE, linewidth=1.1, linestyle="--")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Hugonnet mass-change rate (m w.e. yr$^{-1}$)")
    ax.set_ylabel(r"Reconstructed period mean (m w.e. yr$^{-1}$)")
    ax.set_title(title, loc="left")
    metric_text = (
        f"n = {metric['n_glaciers']:,}\n"
        f"r = {metric['pearson_r']:.2f}\n"
        f"R² = {metric['r2']:.2f}\n"
        f"RMSE = {metric['rmse_mwe_yr']:.2f}\n"
        f"Bias = {metric['bias_model_minus_hugonnet_mwe_yr']:+.2f}"
    )
    ax.text(
        0.04,
        0.96,
        metric_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.4,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "0.72", "alpha": 0.92},
    )
    if calibrated:
        ax.text(
            0.96,
            0.04,
            "Constraint consistency\n(not independent validation)",
            transform=ax.transAxes,
            va="bottom",
            ha="right",
            fontsize=8.1,
            color="#8b2f24",
        )
    if outside_axes:
        ax.text(
            0.96,
            0.96,
            f"{outside_axes} extremes outside axes",
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize=7.8,
            color="0.35",
        )


def plot_residual_distributions(
    ax,
    comparison: pd.DataFrame,
    raw_metric: dict[str, float | int | str],
    calibrated_metric: dict[str, float | int | str],
) -> None:
    data = [
        comparison["raw_residual_mwe_yr"].to_numpy(),
        comparison["calibrated_residual_mwe_yr"].to_numpy(),
    ]
    violin = ax.violinplot(
        data,
        positions=[1, 2],
        widths=0.72,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        bw_method=0.22,
    )
    for body, color in zip(
        violin["bodies"],
        [COLOR_RAW, COLOR_CALIBRATED],
    ):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.48)

    box = ax.boxplot(
        data,
        positions=[1, 2],
        widths=0.20,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "#111111", "linewidth": 1.3},
        whiskerprops={"color": "#333333", "linewidth": 0.8},
        capprops={"color": "#333333", "linewidth": 0.8},
    )
    for patch, color in zip(box["boxes"], [COLOR_RAW, COLOR_CALIBRATED]):
        patch.set_facecolor(color)
        patch.set_alpha(0.82)
        patch.set_edgecolor("#333333")

    ax.axhline(0.0, color="0.25", linewidth=0.9)
    ax.set_xticks([1, 2], ["Raw", "Conservative\ncalibration"])
    ax.set_ylabel(r"Residual: reconstruction - Hugonnet (m w.e. yr$^{-1}$)")
    ax.set_title("(c) Glacier-wise residual distributions", loc="left")
    ax.grid(axis="y", color="0.88", linewidth=0.6)
    ax.text(
        0.04,
        0.96,
        "Within Hugonnet ±1σ\n"
        f"Raw: {100.0 * raw_metric['within_hugonnet_1sigma_fraction']:.1f}%\n"
        "Conservative: "
        f"{100.0 * calibrated_metric['within_hugonnet_1sigma_fraction']:.1f}%",
        transform=ax.transAxes,
        va="top",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "0.72", "alpha": 0.92},
    )


def plot_residual_map(ax, comparison: pd.DataFrame) -> None:
    add_map_context(ax)
    residual_limit = 1.0
    norm = mcolors.TwoSlopeNorm(
        vmin=-residual_limit,
        vcenter=0.0,
        vmax=residual_limit,
    )
    sizes = np.clip(
        8.0 + 10.0 * np.log10(comparison["area_km2"].to_numpy() + 1.0),
        8.0,
        46.0,
    )
    scatter = ax.scatter(
        comparison["cenlon"],
        comparison["cenlat"],
        c=comparison["raw_residual_mwe_yr"],
        s=sizes,
        cmap="RdBu_r",
        norm=norm,
        edgecolors="#252525",
        linewidths=0.18,
        alpha=0.88,
        zorder=3,
        rasterized=True,
    )
    colorbar = plt.colorbar(
        scatter,
        ax=ax,
        fraction=0.035,
        pad=0.02,
        extend="both",
    )
    colorbar.set_label(
        r"Raw residual: reconstruction - Hugonnet (m w.e. yr$^{-1}$)"
    )
    ax.set_title("(d) Spatial pattern of raw glacier-wise residuals", loc="left")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.text(
        0.015,
        0.025,
        f"n = {len(comparison):,} glaciers",
        transform=ax.transAxes,
        fontsize=8.2,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "0.72", "alpha": 0.9},
        zorder=5,
    )


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.2,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
        }
    )

    comparison = build_comparison()
    raw_metric = calculate_metrics(
        comparison,
        "raw_period_mean_mwe_yr",
        "External geodetic consistency of raw reconstruction",
    )
    calibrated_metric = calculate_metrics(
        comparison,
        "calibrated_period_mean_mwe_yr",
        "Post-calibration constraint consistency; not independent validation",
    )
    metrics = pd.DataFrame([raw_metric, calibrated_metric])

    comparison.to_csv(OUT_CSV, index=False)
    metrics.to_csv(OUT_METRICS_CSV, index=False)

    limits = common_scatter_limits(comparison)
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13.6, 10.0),
        gridspec_kw={"height_ratios": [1.0, 0.93]},
    )
    plot_density_scatter(
        axes[0, 0],
        comparison,
        "raw_period_mean_mwe_yr",
        raw_metric,
        "(a) Raw reconstruction vs Hugonnet (2000-2019)",
        limits,
        calibrated=False,
    )
    plot_density_scatter(
        axes[0, 1],
        comparison,
        "calibrated_period_mean_mwe_yr",
        calibrated_metric,
        "(b) Conservative calibration vs Hugonnet (2000-2019)",
        limits,
        calibrated=True,
    )
    plot_residual_distributions(
        axes[1, 0],
        comparison,
        raw_metric,
        calibrated_metric,
    )
    plot_residual_map(axes[1, 1], comparison)

    fig.text(
        0.5,
        0.008,
        "One complete-period value per glacier. Hugonnet is a geodetic mass-change "
        "product, not an annual in-situ SMB observation. Scatter axes show the "
        "central 99.8%; all rows are retained in metrics.",
        ha="center",
        va="bottom",
        fontsize=8.2,
        color="0.35",
    )
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 1.0), h_pad=2.1, w_pad=1.9)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved figure -> {OUT_PNG}")
    print(f"Saved glacier-wise data -> {OUT_CSV}")
    print(f"Saved metrics -> {OUT_METRICS_CSV}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
