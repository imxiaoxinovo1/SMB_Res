"""Plot observed vs predicted scatter for best LOGO and LOYO ensembles."""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, pearsonr
from sklearn.metrics import mean_squared_error, r2_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from config import FIG_DIR, RESULT_DIR  # noqa: E402


PREDICTION_FILES = [
    (
        "LOGO",
        "LOGO validation",
        Path(RESULT_DIR) / "ensemble_qc" / "current" / "pred_xgb_hyp_logo.csv",
        "viridis",
    ),
    (
        "LOYO",
        "LOYO validation",
        Path(RESULT_DIR) / "ensemble_qc" / "current" / "pred_xgb_base_loyo.csv",
        "plasma",
    ),
]


def safe_savefig(fig, path: Path) -> Path:
    """Save PNG using unique filenames; fallback to tmp_ppt_work if needed."""
    candidates = [
        path,
        Path(r"H:\Code\SMB\tmp_ppt_work") / path.name,
    ]
    errors = []
    for candidate in candidates:
        try:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(candidate, dpi=320, bbox_inches="tight", facecolor="white")
            return candidate
        except OSError as exc:
            errors.append(f"{candidate}: {type(exc).__name__}: {exc}")
            continue
    raise PermissionError("Could not save figure. Attempts:\n" + "\n".join(errors))


def metrics(obs: np.ndarray, pred: np.ndarray) -> tuple[float, float, float, float]:
    r2 = r2_score(obs, pred)
    r, _ = pearsonr(obs, pred)
    rmse = np.sqrt(mean_squared_error(obs, pred)) * 1000.0
    bias = np.mean(pred - obs) * 1000.0
    return r2, r, rmse, bias


def draw_panel(ax, frame: pd.DataFrame, cv: str, title: str, cmap: str) -> None:
    obs = frame["obs"].values.astype(float)
    pred = frame["pred"].values.astype(float)
    r2, r, rmse, bias = metrics(obs, pred)

    xy = np.vstack([obs, pred])
    density = gaussian_kde(xy)(xy)
    order = density.argsort()
    obs_plot, pred_plot, density_plot = obs[order], pred[order], density[order]

    low = min(obs.min(), pred.min())
    high = max(obs.max(), pred.max())
    pad = (high - low) * 0.06 if high > low else 0.2
    lim_min = float(low - pad)
    lim_max = float(high + pad)

    scatter = ax.scatter(
        obs_plot,
        pred_plot,
        c=density_plot,
        s=20,
        cmap=cmap,
        alpha=0.9,
        edgecolors="none",
    )
    ax.plot([lim_min, lim_max], [lim_min, lim_max], color="black", linewidth=1.1)
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Observed SMB (m w.e. a$^{-1}$)")
    ax.set_ylabel("Predicted SMB (m w.e. a$^{-1}$)")
    ax.text(
        0.03,
        0.97,
        f"R = {r:.2f}\nR$^2$ = {r2:.2f}\nRMSE = {rmse / 1000.0:.2f}\nBias = {bias / 1000.0:+.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.7", alpha=0.86),
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4.5%", pad=0.06)
    cb = plt.colorbar(scatter, cax=cax)
    cb.set_label("Density", fontsize=9)
    cb.ax.tick_params(labelsize=8)


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 180,
        "savefig.dpi": 320,
        "savefig.bbox": "tight",
    })

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), dpi=180)
    fig.patch.set_facecolor("white")

    for ax, (cv, title, path, cmap) in zip(axes, PREDICTION_FILES):
        if not path.exists():
            raise FileNotFoundError(f"Missing prediction file: {path}")
        frame = pd.read_csv(path)
        draw_panel(ax, frame, cv, title, cmap)

    fig.suptitle("Ensemble model", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(FIG_DIR) / f"fig_ensemble_model_scatter_{timestamp}.png"
    saved_path = safe_savefig(fig, out_path)
    plt.close(fig)
    print(f"Saved: {saved_path}")


if __name__ == "__main__":
    main()
