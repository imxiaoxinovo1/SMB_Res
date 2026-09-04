"""Create the main publication figure for corrected Phys v2 validation."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from config import FIG_DIR, PHYS_V2_RESULT_DIR  # noqa: E402

MAIN_MODEL = "xgboost_v2_compact_monthly_all_hyp_p-regularized"
MAIN_DIR = Path(PHYS_V2_RESULT_DIR) / MAIN_MODEL
EVALUATION_CSV = (
    Path(PHYS_V2_RESULT_DIR) / "publication_evaluation" / "model_metrics_cluster_bootstrap.csv"
)
OUT_PNG = Path(FIG_DIR) / "fig_phys_v2_validation_performance.png"

COLORS = {
    "Mean baseline": "#a3a3a3",
    "Ridge": "#7f8c8d",
    "Random Forest": "#4f86a6",
    "PhysGlacierFormer": "#c86b3c",
    "XGBoost: raw climate": "#6a9f58",
    "XGBoost: climate + anomalies": "#1f5a85",
}


def predictions(cv: str) -> pd.DataFrame:
    path = MAIN_DIR / f"{MAIN_MODEL}_{cv.lower()}_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def draw_scatter(ax, frame: pd.DataFrame, title: str) -> None:
    observed = frame["obs_annual"].to_numpy()
    predicted = frame["pred_annual"].to_numpy()
    low = float(min(observed.min(), predicted.min()) - 0.15)
    high = float(max(observed.max(), predicted.max()) + 0.15)
    artist = ax.hexbin(
        observed, predicted, gridsize=35, mincnt=1, bins="log", cmap="Blues", linewidths=0
    )
    ax.plot([low, high], [low, high], color="0.15", lw=1.0, ls="--")
    ax.set(xlim=(low, high), ylim=(low, high), xlabel="Observed SMB (m w.e. yr$^{-1}$)",
           ylabel="Predicted SMB (m w.e. yr$^{-1}$)", title=title)
    ax.set_aspect("equal", adjustable="box")
    r2 = r2_score(observed, predicted)
    rmse = np.sqrt(mean_squared_error(observed, predicted))
    correlation = np.corrcoef(observed, predicted)[0, 1]
    bias = np.mean(predicted - observed)
    ax.text(
        0.04, 0.96,
        f"R$^2$ = {r2:.2f}\nr = {correlation:.2f}\nRMSE = {rmse:.2f} m\nBias = {bias:+.2f} m",
        transform=ax.transAxes, ha="left", va="top", fontsize=8.3,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
    )
    colorbar = plt.colorbar(artist, ax=ax, fraction=0.046, pad=0.035)
    colorbar.set_label("Sample count (log scale)")


def summary(model: str, cv: str) -> pd.Series:
    folder = Path(PHYS_V2_RESULT_DIR) / model
    path = folder / f"{model}_{cv.lower()}_summary.csv"
    frame = pd.read_csv(path)
    if "target" in frame.columns:
        frame = frame[frame["target"] == "annual"]
    return frame.iloc[0]


def draw_benchmarks(ax) -> None:
    models = [
        ("Mean baseline", "dummy_v2_compact_monthly_all_hyp"),
        ("Ridge", "ridge_v2_compact_monthly_all_hyp"),
        ("Random Forest", "randomforest_v2_compact_monthly_all_hyp"),
        ("PhysGlacierFormer", "phys_v2_compact_huber_sw2"),
        ("XGBoost: raw climate", "xgboost_v2_raw_compact_terrain_nohyp"),
        ("XGBoost: climate + anomalies", MAIN_MODEL),
    ]
    y = np.arange(len(models))
    logo = [float(summary(model, "logo")["r2"]) for _, model in models]
    loyo = [float(summary(model, "loyo")["r2"]) for _, model in models]
    colors = [COLORS[label] for label, _ in models]
    for position, logo_value, loyo_value, color in zip(y, logo, loyo, colors):
        ax.plot([logo_value, loyo_value], [position, position], color="0.72", lw=1.1, zorder=1)
        ax.scatter(logo_value, position, s=45, marker="o", color=color, edgecolor="white", lw=0.5, zorder=3)
        ax.scatter(loyo_value, position, s=45, marker="s", color=color, edgecolor="white", lw=0.5, zorder=3)
    ax.scatter([], [], s=40, marker="o", color="0.35", label="LOGO")
    ax.scatter([], [], s=40, marker="s", color="0.35", label="LOYO")
    ax.set_title("(c) Corrected-pipeline benchmarks")
    ax.set_xlabel("R$^2$")
    ax.set_yticks(y, [label for label, _ in models], fontsize=7.7)
    ax.set_xlim(-0.08, 0.70)
    ax.invert_yaxis()
    ax.axvline(0, color="0.45", lw=0.7)
    ax.legend(frameon=False, ncols=2, loc="upper right", fontsize=8)


def draw_robustness(ax) -> None:
    metrics = pd.read_csv(EVALUATION_CSV)
    wanted = [
        ("LOGO", "LOGO"),
        ("LOGO_BUFFERED", "LOGO\n50 km buffer"),
        ("LOYO", "LOYO"),
        ("LOYO_BUFFERED", "LOYO\n+/-1 yr buffer"),
        ("FORWARD", "Forward\n1980-2023"),
        ("LOSO", "LOSO\nsubregions"),
    ]
    rows = []
    for cv, label in wanted:
        match = metrics[(metrics["model"] == MAIN_MODEL) & (metrics["cv"] == cv)]
        if len(match) != 1:
            raise RuntimeError(f"Expected one complete metric row for {MAIN_MODEL} {cv}, found {len(match)}")
        rows.append((label, match.iloc[0]))
    x = np.arange(len(rows))
    values = np.asarray([float(row["r2"]) for _, row in rows])
    lower = values - np.asarray([float(row["r2_ci_low"]) for _, row in rows])
    upper = np.asarray([float(row["r2_ci_high"]) for _, row in rows]) - values
    colors = ["#1f5a85", "#4e86ad", "#2f8f83", "#67aaa0", "#c86b3c", "#9a6b9d"]
    ax.errorbar(x, values, yerr=np.vstack([lower, upper]), fmt="none", ecolor="0.3", capsize=3, lw=1)
    ax.scatter(x, values, s=48, c=colors, edgecolor="white", linewidth=0.6, zorder=3)
    for xpos, value in zip(x, values):
        ax.text(xpos, value + 0.022, f"{value:.2f}", ha="center", fontsize=7.5)
    ax.set_title("(d) Spatial and temporal stress tests")
    ax.set_ylabel("R$^2$ with cluster-bootstrap 95% CI")
    ax.set_xticks(x, [label for label, _ in rows], fontsize=7.6)
    ax.set_ylim(min(-0.05, float(np.min(values - lower)) - 0.05), 0.75)
    ax.axhline(0, color="0.45", lw=0.7)


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "0.9",
            "grid.linewidth": 0.6,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 9.0))
    draw_scatter(axes[0, 0], predictions("logo"), "(a) Leave-one-glacier-out (LOGO)")
    draw_scatter(axes[0, 1], predictions("loyo"), "(b) Leave-one-year-out (LOYO)")
    draw_benchmarks(axes[1, 0])
    draw_robustness(axes[1, 1])
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.09, top=0.97, wspace=0.25, hspace=0.28)
    fig.savefig(OUT_PNG, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved -> {OUT_PNG}")


if __name__ == "__main__":
    main()
