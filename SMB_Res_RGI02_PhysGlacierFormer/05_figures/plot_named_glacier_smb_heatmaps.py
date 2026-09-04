"""Plot separate observed and reconstructed SMB heatmaps for named RGI02 glaciers."""
from __future__ import annotations

import copy
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from config import (  # noqa: E402
    FIG_DIR,
    HYPSOMETRY_RECON_CONSERVATIVE_CSV,
    MASSBAL_RGI02_CSV,
    QC_DATA_DIR,
    RGI02_ATTRIBUTES_CSV,
)


YEAR_MIN = 1980
YEAR_MAX = 2024
RECONSTRUCTION_MIN_AREA_KM2 = 0.5
MAPPING_CSV = os.path.join(QC_DATA_DIR, "training_glaciers_terrain_qc.csv")
OBSERVATION_CSV = os.path.join(QC_DATA_DIR, "massbal_rgi02_qc.csv")

# Nearest-centroid matching selected adjacent small RGI polygons for these glaciers.
# RGI v7 contains exact name matches at the corrected IDs below.
RGI_ID_CORRECTIONS = {
    "South Cascade Glacier": "RGI2000-v7.0-G-02-14143",
    "Sperry Glacier": "RGI2000-v7.0-G-02-18322",
    "Illecillewaet Glacier": "RGI2000-v7.0-G-02-11913",
}

OUT_OBS_PNG = os.path.join(FIG_DIR, "fig_named_glacier_smb_observed.png")
OUT_RECON_PNG = os.path.join(FIG_DIR, "fig_named_glacier_smb_reconstructed.png")
OUT_QC_CSV = os.path.join(FIG_DIR, "named_glacier_smb_heatmap_qc.csv")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mapping = pd.read_csv(MAPPING_CSV)
    obs_path = OBSERVATION_CSV if os.path.exists(OBSERVATION_CSV) else MASSBAL_RGI02_CSV
    observations = pd.read_csv(obs_path)
    reconstruction = pd.read_csv(HYPSOMETRY_RECON_CONSERVATIVE_CSV)
    attributes = pd.read_csv(
        RGI02_ATTRIBUTES_CSV,
        usecols=["rgi_id", "glac_name", "area_km2"],
    )

    required = {
        "mapping": (mapping, {"glacier_id", "name", "rgi_id"}),
        "observations": (observations, {"glacier_id", "year", "annual_balance"}),
        "reconstruction": (
            reconstruction,
            {"rgi_id", "year", "predicted_smb_conservative_m"},
        ),
    }
    for label, (frame, columns) in required.items():
        missing = columns.difference(frame.columns)
        if missing:
            raise ValueError(f"{label} data are missing columns: {sorted(missing)}")

    mapping["name"] = mapping["name"].fillna("").astype(str).str.strip()
    mapping = mapping[mapping["name"].ne("")].drop_duplicates("glacier_id").copy()
    mapping = mapping.rename(columns={"rgi_id": "original_rgi_id"})
    mapping["rgi_id"] = mapping["original_rgi_id"]
    mapping["mapping_method"] = "nearest RGI centroid"

    for glacier_name, corrected_id in RGI_ID_CORRECTIONS.items():
        mask = mapping["name"].eq(glacier_name)
        if mask.sum() != 1:
            raise RuntimeError(
                f"Expected one mapping row for {glacier_name!r}, found {int(mask.sum())}."
            )
        if corrected_id not in set(attributes["rgi_id"]):
            raise RuntimeError(f"Corrected RGI ID not found: {corrected_id}")
        mapping.loc[mask, "rgi_id"] = corrected_id
        mapping.loc[mask, "mapping_method"] = "exact RGI v7 glacier-name correction"

    return mapping, observations, reconstruction, attributes


def build_matrices() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mapping, observations, reconstruction, attributes = load_inputs()
    years = np.arange(YEAR_MIN, YEAR_MAX + 1)

    obs = observations[observations["year"].between(YEAR_MIN, YEAR_MAX)].merge(
        mapping[["glacier_id", "name", "original_rgi_id", "rgi_id", "mapping_method"]],
        on="glacier_id",
        how="inner",
        validate="many_to_one",
    )
    qc = (
        obs.groupby(
            ["glacier_id", "name", "original_rgi_id", "rgi_id", "mapping_method"],
            as_index=False,
        )
        .agg(
            n_observed=("annual_balance", "count"),
            first_observed_year=("year", "min"),
            last_observed_year=("year", "max"),
        )
        .merge(
            attributes.rename(
                columns={"glac_name": "rgi_glacier_name", "area_km2": "rgi_area_km2"}
            ),
            on="rgi_id",
            how="left",
            validate="many_to_one",
        )
    )

    recon = reconstruction[reconstruction["year"].between(YEAR_MIN, YEAR_MAX)].copy()
    reconstructed_ids = set(recon["rgi_id"].dropna().unique())
    qc["belongs_to_rgi02"] = qc["rgi_id"].str.contains("G-02-", regex=False)
    qc["meets_reconstruction_area_threshold"] = (
        qc["rgi_area_km2"] >= RECONSTRUCTION_MIN_AREA_KM2
    )
    qc["has_reconstruction"] = qc["rgi_id"].isin(reconstructed_ids)
    qc["status"] = np.select(
        [
            qc["has_reconstruction"],
            ~qc["meets_reconstruction_area_threshold"],
        ],
        [
            "reconstructed",
            "RGI02 glacier below 0.5 km2 reconstruction threshold",
        ],
        default="RGI02 glacier missing from reconstruction for another reason",
    )
    qc = qc.sort_values(
        ["n_observed", "first_observed_year", "name"],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    # Keep both figures directly comparable by plotting only reconstructed glaciers.
    order = qc.loc[qc["has_reconstruction"], "name"].tolist()
    obs_matrix = (
        obs.pivot_table(
            index="name",
            columns="year",
            values="annual_balance",
            aggfunc="mean",
        )
        .reindex(index=order, columns=years)
    )
    recon_named = recon.merge(
        qc[["rgi_id", "name"]],
        on="rgi_id",
        how="inner",
        validate="many_to_one",
    )
    recon_matrix = (
        recon_named.pivot_table(
            index="name",
            columns="year",
            values="predicted_smb_conservative_m",
            aggfunc="mean",
        )
        .reindex(index=order, columns=years)
    )

    covered_names = qc.loc[qc["has_reconstruction"], "name"]
    missing_covered = int(recon_matrix.loc[covered_names].isna().sum().sum())
    if missing_covered:
        raise RuntimeError(
            f"Reconstructed glaciers contain {missing_covered} missing glacier-years."
        )
    return obs_matrix, recon_matrix, qc


def robust_symmetric_limit(obs_matrix: pd.DataFrame, recon_matrix: pd.DataFrame) -> float:
    values = np.concatenate(
        [obs_matrix.to_numpy(dtype=float).ravel(), recon_matrix.to_numpy(dtype=float).ravel()]
    )
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise RuntimeError("No finite SMB values are available for plotting.")
    limit = float(np.nanquantile(np.abs(values), 0.98))
    return max(2.0, np.ceil(limit * 2.0) / 2.0)


def make_colormap(missing_color: str) -> LinearSegmentedColormap:
    cmap = LinearSegmentedColormap.from_list(
        "smb_red_neutral_blue",
        ["#b2182b", "#ef8a62", "#e7e0d3", "#67a9cf", "#2166ac"],
        N=256,
    )
    cmap = copy.copy(cmap)
    cmap.set_bad(missing_color)
    return cmap


def save_heatmap(
    matrix: pd.DataFrame,
    title: str,
    output_path: str,
    norm: TwoSlopeNorm,
    missing_color: str,
    missing_label: str | None,
) -> None:
    n_glaciers, n_years = matrix.shape
    fig_height = max(7.8, 0.32 * n_glaciers + 2.2)
    fig = plt.figure(figsize=(12.6, fig_height))
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.0, 0.025],
        left=0.19,
        right=0.92,
        bottom=0.11,
        top=0.94,
        wspace=0.05,
    )
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])

    cmap = make_colormap(missing_color)
    masked = np.ma.masked_invalid(matrix.to_numpy(dtype=float))
    image = ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto", interpolation="none")

    ax.set_yticks(np.arange(n_glaciers))
    ax.set_yticklabels(matrix.index, fontsize=8.6)
    ax.set_ylabel("Glacier", fontsize=10)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=9)

    major_positions = np.arange(0, n_years, 5)
    ax.set_xticks(major_positions)
    ax.set_xticklabels(matrix.columns[major_positions], fontsize=8.6)
    ax.set_xlabel("Year", fontsize=10)
    ax.set_xticks(np.arange(-0.5, n_years, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_glaciers, 1), minor=True)
    ax.grid(which="minor", color="#d7d7d7", linestyle=":", linewidth=0.42)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(which="major", length=3, width=0.7)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#333333")
        spine.set_linewidth(0.75)

    if missing_label is not None:
        legend_patch = Patch(
            facecolor=missing_color,
            edgecolor="#777777",
            linewidth=0.7,
            label=missing_label,
        )
        ax.legend(
            handles=[legend_patch],
            loc="upper right",
            frameon=True,
            framealpha=0.95,
            edgecolor="#cccccc",
            fontsize=8,
        )

    cbar = fig.colorbar(image, cax=cax, extend="both")
    cbar.set_label("Glacier-wide annual mass balance (m w.e.)", fontsize=10, labelpad=9)
    cbar.ax.tick_params(labelsize=8.5, width=0.7, length=3)
    cbar.outline.set_linewidth(0.7)
    fig.savefig(output_path, dpi=450, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    obs_matrix, recon_matrix, qc = build_matrices()
    color_limit = robust_symmetric_limit(obs_matrix, recon_matrix)
    norm = TwoSlopeNorm(vmin=-color_limit, vcenter=0.0, vmax=color_limit)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.linewidth": 0.75,
            "figure.dpi": 150,
            "savefig.facecolor": "white",
        }
    )

    save_heatmap(
        obs_matrix,
        "Observed annual mass balance",
        OUT_OBS_PNG,
        norm,
        missing_color="#ffffff",
        missing_label="No observation",
    )
    save_heatmap(
        recon_matrix,
        "GlacierFormer reconstructed annual mass balance",
        OUT_RECON_PNG,
        norm,
        missing_color="#ffffff",
        missing_label=None,
    )
    qc.to_csv(OUT_QC_CSV, index=False)

    print(f"Named RGI02 glaciers shown: {len(obs_matrix)}")
    print(f"Glaciers with reconstruction: {int(qc['has_reconstruction'].sum())}")
    print(f"Glaciers below area threshold: {int((~qc['meets_reconstruction_area_threshold']).sum())}")
    print(f"Observed glacier-years: {int(obs_matrix.notna().sum().sum())}")
    print(f"Reconstructed glacier-years: {int(recon_matrix.notna().sum().sum())}")
    print(f"Shared color range: {-color_limit:.1f} to {color_limit:.1f} m w.e.")
    print(f"Saved observed figure -> {OUT_OBS_PNG}")
    print(f"Saved reconstructed figure -> {OUT_RECON_PNG}")
    print(f"Saved QC -> {OUT_QC_CSV}")


if __name__ == "__main__":
    main()
