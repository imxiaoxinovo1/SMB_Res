"""Four-panel LOYO scatter plot for RF, XGBoost, LSTM, and GlacierFormer."""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, pearsonr
from sklearn.metrics import mean_squared_error


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
ROOT_DIR = os.path.dirname(PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)

from config import FIG_DIR  # noqa: E402


LEGACY_RESULTS = os.path.join(ROOT_DIR, "SMB_Res_ByClaudeV2", "results")
CURRENT_RESULTS = os.path.join(PROJECT_DIR, "results", "hypsometry_qc")

MODELS = [
    ("RF", os.path.join(LEGACY_RESULTS, "rf_loyo_predictions.csv")),
    ("XGBoost", os.path.join(LEGACY_RESULTS, "xgboost_loyo_predictions.csv")),
    ("LSTM", os.path.join(LEGACY_RESULTS, "lstm_loyo_predictions.csv")),
    (
        "GlacierFormer",
        os.path.join(
            CURRENT_RESULTS,
            "glacierformer_hypsometry_qc_loyo_predictions.csv",
        ),
    ),
]

OUT_PNG = os.path.join(FIG_DIR, "fig_loyo_4models.png")


def load_predictions(path: str) -> tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prediction file not found: {path}")
    frame = pd.read_csv(path)
    missing = {"obs", "pred"}.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    frame = frame[["obs", "pred"]].apply(pd.to_numeric, errors="coerce").dropna()
    return frame["obs"].to_numpy(), frame["pred"].to_numpy()


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.linewidth": 0.8,
            "savefig.facecolor": "white",
        }
    )

    loaded = [(label, *load_predictions(path)) for label, path in MODELS]
    global_lim = 1.1 * max(
        max(np.abs(obs).max(), np.abs(pred).max())
        for _, obs, pred in loaded
    )

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 9.2))
    axes = axes.ravel()
    fig.suptitle(
        "Evaluation of modelled annual glacier-wide SMB against observed SMB\n"
        "using LOYO cross-validation",
        fontsize=13,
        y=0.985,
    )

    for ax, (label, obs, pred) in zip(axes, loaded):
        density = gaussian_kde(np.vstack([obs, pred]))(np.vstack([obs, pred]))
        order = np.argsort(density)
        obs_plot, pred_plot, density_plot = obs[order], pred[order], density[order]

        ax.scatter(
            obs_plot,
            pred_plot,
            c=density_plot,
            cmap="viridis",
            s=15,
            alpha=0.75,
            linewidths=0,
            norm=Normalize(vmin=0, vmax=density.max()),
        )
        colorbar = fig.colorbar(
            ScalarMappable(norm=Normalize(0, density.max()), cmap="viridis"),
            ax=ax,
            fraction=0.046,
            pad=0.025,
        )
        colorbar.set_label("Density", fontsize=8)
        colorbar.ax.tick_params(labelsize=7)

        ax.plot([-global_lim, global_lim], [-global_lim, global_lim], "k-", lw=1.1)
        ax.set_xlim(-global_lim, global_lim)
        ax.set_ylim(-global_lim, global_lim)
        ax.set_aspect("equal", adjustable="box")

        pearson_r = pearsonr(obs, pred).statistic
        rmse = np.sqrt(mean_squared_error(obs, pred))
        mae = np.mean(np.abs(pred - obs))
        bias = np.mean(pred - obs)
        stats = (
            f"R = {pearson_r:.2f}\n"
            f"RMSE = {rmse:.2f}\n"
            f"MAE = {mae:.2f}\n"
            f"Bias = {bias:+.2f}\n"
            f"N = {len(obs):,}"
        )
        ax.text(
            0.04,
            0.96,
            stats,
            transform=ax.transAxes,
            va="top",
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.75", alpha=0.88),
        )
        ax.set_title(label, fontsize=12, pad=5)
        ax.set_xlabel("Observed SMB (m w.e. yr$^{-1}$)")
        ax.set_ylabel("Predicted SMB (m w.e. yr$^{-1}$)")
        ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.6)

    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.95))
    fig.savefig(OUT_PNG, dpi=450, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"Saved LOYO figure -> {OUT_PNG}")


if __name__ == "__main__":
    main()
