"""Plot grouped XGBoost SHAP contributions for the selected final model."""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "03_training"))

from config import (  # noqa: E402
    FIG_DIR,
    PHYS_V2_FINAL_MODEL,
    PHYS_V2_FINAL_PREPROCESSOR,
    PHYS_V2_RESULT_DIR,
    PHYS_V2_SEQUENCES_NPZ,
)
from train_tree_v2_cv import FEATURE_SETS, calendar_flatten, hypsometry_quantiles  # noqa: E402
from train_xgboost_v2_final import BASE_STATIC_FEATURES  # noqa: E402


OUT_PNG = Path(FIG_DIR) / "fig_xgboost_feature_contributions.png"
OUT_FEATURE_CSV = (
    Path(PHYS_V2_RESULT_DIR) / "publication_evaluation" / "xgboost_shap_feature_contributions.csv"
)
OUT_GROUP_CSV = (
    Path(PHYS_V2_RESULT_DIR) / "publication_evaluation" / "xgboost_shap_group_contributions.csv"
)

DYNAMIC_LABELS = {
    "t2m": "Air temperature",
    "sd": "Snow depth",
    "asn": "Snow albedo (ERA5-Land)",
    "tp": "Total precipitation",
    "sf": "Snowfall",
    "ssrd": "Downward shortwave radiation",
    "str": "Net longwave radiation",
    "slhf": "Latent heat flux",
    "sshf": "Sensible heat flux",
    "t2m_anomaly": "Temperature anomaly",
    "tp_anomaly": "Precipitation anomaly",
    "sf_anomaly": "Snowfall anomaly",
    "ssrd_anomaly": "Shortwave anomaly",
    "asn_anomaly": "Snow-albedo anomaly",
}


def final_feature_matrix() -> tuple[np.ndarray, list[str]]:
    data = np.load(PHYS_V2_SEQUENCES_NPZ, allow_pickle=True)
    preprocessor = np.load(PHYS_V2_FINAL_PREPROCESSOR, allow_pickle=True)
    feature_names = [str(value) for value in preprocessor["feature_names"]]
    dynamic_names = [str(value) for value in data["dynamic_features"]]
    selected = FEATURE_SETS["compact"]
    dynamic_indices = [dynamic_names.index(name) for name in selected]
    dynamic = calendar_flatten(data["X_dyn"][:, :, dynamic_indices], data["month_ids"])
    stored_static = [str(value) for value in data["static_features"]]
    static_indices = [stored_static.index(name) for name in BASE_STATIC_FEATURES]
    static = data["X_sta"][:, static_indices]
    hypsometry = hypsometry_quantiles(
        data["X_hyp"], data["hypsometry_band_centers_m"]
    )
    matrix = np.column_stack([dynamic, static, hypsometry]).astype(np.float32)
    matrix = np.where(np.isnan(matrix), preprocessor["medians"], matrix)
    if matrix.shape[1] != len(feature_names):
        raise RuntimeError("Feature matrix does not match the stored final-model schema.")
    return matrix, feature_names


def feature_group(name: str) -> str:
    dynamic_match = re.fullmatch(r"(.+)_m(\d{2})", name)
    if dynamic_match:
        variable = dynamic_match.group(1)
        return DYNAMIC_LABELS.get(variable, variable)
    if name.startswith("hyp_"):
        return "Elevation-band hypsometry"
    if name.startswith("clim_"):
        return "Glacier climate normals"
    if name in {"cenlat", "cenlon"}:
        return "Geographic location"
    return "Terrain and geometry"


def main() -> None:
    matrix, feature_names = final_feature_matrix()
    model = XGBRegressor()
    model.load_model(PHYS_V2_FINAL_MODEL)
    contributions = model.get_booster().predict(
        xgb.DMatrix(matrix), pred_contribs=True
    )
    if contributions.shape[1] != matrix.shape[1] + 1:
        raise RuntimeError("Unexpected XGBoost contribution matrix shape.")
    prediction = model.predict(matrix)
    if not np.allclose(contributions.sum(axis=1), prediction, rtol=1e-5, atol=1e-5):
        raise RuntimeError("SHAP contributions do not conserve the model prediction.")

    mean_abs = np.mean(np.abs(contributions[:, :-1]), axis=0)
    feature_table = pd.DataFrame(
        {
            "feature": feature_names,
            "group": [feature_group(name) for name in feature_names],
            "mean_abs_shap_mwe_yr": mean_abs,
        }
    ).sort_values("mean_abs_shap_mwe_yr", ascending=False)
    group_table = (
        feature_table.groupby("group", as_index=False)["mean_abs_shap_mwe_yr"]
        .sum()
        .sort_values("mean_abs_shap_mwe_yr", ascending=False)
    )
    total = float(group_table["mean_abs_shap_mwe_yr"].sum())
    group_table["fraction_of_total_abs_shap"] = (
        group_table["mean_abs_shap_mwe_yr"] / total
    )

    selected_dynamic = FEATURE_SETS["compact"]
    heatmap = np.empty((len(selected_dynamic), 12), dtype=float)
    for row, variable in enumerate(selected_dynamic):
        for month in range(1, 13):
            name = f"{variable}_m{month:02d}"
            heatmap[row, month - 1] = float(
                feature_table.loc[
                    feature_table["feature"] == name, "mean_abs_shap_mwe_yr"
                ].iloc[0]
            )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, (ax_group, ax_month) = plt.subplots(
        1, 2, figsize=(11.4, 5.3), gridspec_kw={"width_ratios": [0.86, 1.42]}
    )

    shown = group_table.head(12).sort_values("mean_abs_shap_mwe_yr")
    colors = ["#1f5a85" if "anomaly" in name.lower() else "#6f98ad" for name in shown["group"]]
    ax_group.barh(
        shown["group"], shown["mean_abs_shap_mwe_yr"], color=colors, edgecolor="white"
    )
    ax_group.set_title("(a) Aggregated feature-family contribution")
    ax_group.set_xlabel("Sum of mean |SHAP| (m w.e. yr$^{-1}$)")
    ax_group.grid(axis="x", color="0.9", lw=0.6)

    image = ax_month.imshow(heatmap, aspect="auto", cmap="YlGnBu")
    ax_month.set_title("(b) Monthly climate-feature contribution")
    ax_month.set_xlabel("Calendar month")
    ax_month.set_xticks(np.arange(12), np.arange(1, 13))
    ax_month.set_yticks(
        np.arange(len(selected_dynamic)),
        [DYNAMIC_LABELS.get(name, name) for name in selected_dynamic],
        fontsize=7.6,
    )
    colorbar = fig.colorbar(image, ax=ax_month, fraction=0.045, pad=0.025)
    colorbar.set_label("Mean |SHAP| (m w.e. yr$^{-1}$)")
    fig.text(
        0.5,
        0.012,
        "Descriptive interpretation of the final fitted model; SHAP magnitude does not imply causality.",
        ha="center",
        fontsize=7.2,
        color="0.35",
    )
    fig.subplots_adjust(left=0.22, right=0.97, bottom=0.16, top=0.91, wspace=0.38)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    OUT_FEATURE_CSV.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    feature_table.to_csv(OUT_FEATURE_CSV, index=False)
    group_table.to_csv(OUT_GROUP_CSV, index=False)
    print(group_table.to_string(index=False))
    print(f"Saved -> {OUT_PNG}")
    print(f"Saved -> {OUT_FEATURE_CSV}")
    print(f"Saved -> {OUT_GROUP_CSV}")


if __name__ == "__main__":
    main()
