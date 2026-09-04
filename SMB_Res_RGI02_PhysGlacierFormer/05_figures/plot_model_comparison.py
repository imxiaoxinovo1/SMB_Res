"""Plot LOGO/LOYO model comparison bars for R2 and RMSE."""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from config import FIG_DIR, RESULT_DIR  # noqa: E402


SUMMARY_FILES = [
    ("Base", "baseline", "glacierformer_base_logo_summary.csv"),
    ("Base", "baseline", "glacierformer_base_loyo_summary.csv"),
    ("QC Base", "baseline_qc", "glacierformer_base_qc_logo_summary.csv"),
    ("QC Base", "baseline_qc", "glacierformer_base_qc_loyo_summary.csv"),
    ("Hypsometry", "hypsometry_qc", "glacierformer_hypsometry_qc_logo_summary.csv"),
    ("Hypsometry", "hypsometry_qc", "glacierformer_hypsometry_qc_loyo_summary.csv"),
    ("XGBoost", "xgboost_qc", "xgboost_qc_logo_summary.csv"),
    ("XGBoost", "xgboost_qc", "xgboost_qc_loyo_summary.csv"),
    ("Ensemble", os.path.join("ensemble_qc", "current"), "summary_xgb_hyp_logo.csv"),
    ("Ensemble", os.path.join("ensemble_qc", "current"), "summary_xgb_base_loyo.csv"),
]

MODEL_ORDER = ["Base", "QC Base", "Hypsometry", "XGBoost", "Ensemble"]
MODEL_COLORS = {
    "Base": "#B8C0C8",
    "QC Base": "#4C78A8",
    "Hypsometry": "#59A14F",
    "XGBoost": "#F28E2B",
    "Ensemble": "#D62728",
}


def safe_savefig(fig, path: Path) -> Path:
    """Save PNG using a unique filename, falling back to tmp_ppt_work if needed."""
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


def read_results() -> pd.DataFrame:
    rows = []
    for label, folder, filename in SUMMARY_FILES:
        path = Path(RESULT_DIR) / folder / filename
        if not path.exists():
            print(f"WARNING: missing summary file: {path}")
            continue
        frame = pd.read_csv(path)
        frame.columns = [str(col).strip() for col in frame.columns]
        item = frame.iloc[0].to_dict()
        item["label"] = label
        item["cv"] = str(item["cv"]).upper()
        rows.append(item)
    if not rows:
        raise FileNotFoundError("No summary files were found.")
    out = pd.DataFrame(rows)
    for col in ["r2", "pearson_r", "rmse_mm", "bias_mm"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def grouped_values(data: pd.DataFrame, metric: str, cv: str) -> list[float]:
    values = []
    for model in MODEL_ORDER:
        match = data[(data["label"] == model) & (data["cv"] == cv)]
        values.append(float(match[metric].iloc[0]) if len(match) else np.nan)
    return values


def annotate_bars(ax, bars, fmt: str, y_offset: float) -> None:
    for bar in bars:
        height = bar.get_height()
        if np.isnan(height):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + y_offset,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=8,
            color="#303030",
        )


def main() -> None:
    data = read_results()
    os.makedirs(FIG_DIR, exist_ok=True)

    x = np.arange(len(MODEL_ORDER))
    width = 0.36
    colors = [MODEL_COLORS[m] for m in MODEL_ORDER]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=180)
    fig.patch.set_facecolor("white")

    logo_r2 = grouped_values(data, "r2", "LOGO")
    loyo_r2 = grouped_values(data, "r2", "LOYO")
    bars1 = axes[0].bar(x - width / 2, logo_r2, width, label="LOGO", color=colors, alpha=0.95)
    bars2 = axes[0].bar(x + width / 2, loyo_r2, width, label="LOYO", color=colors, alpha=0.55, hatch="//")
    axes[0].set_ylabel("R²")
    axes[0].set_ylim(0, max(np.nanmax(logo_r2), np.nanmax(loyo_r2)) + 0.12)
    axes[0].set_title("Predictive skill")
    annotate_bars(axes[0], bars1, "{:.2f}", 0.01)
    annotate_bars(axes[0], bars2, "{:.2f}", 0.01)

    logo_rmse = grouped_values(data, "rmse_mm", "LOGO")
    loyo_rmse = grouped_values(data, "rmse_mm", "LOYO")
    bars3 = axes[1].bar(x - width / 2, logo_rmse, width, label="LOGO", color=colors, alpha=0.95)
    bars4 = axes[1].bar(x + width / 2, loyo_rmse, width, label="LOYO", color=colors, alpha=0.55, hatch="//")
    axes[1].set_ylabel("RMSE (mm w.e.)")
    axes[1].set_ylim(700, max(np.nanmax(logo_rmse), np.nanmax(loyo_rmse)) + 70)
    axes[1].set_title("Prediction error")
    annotate_bars(axes[1], bars3, "{:.0f}", 8)
    annotate_bars(axes[1], bars4, "{:.0f}", 8)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_ORDER, rotation=18, ha="right")
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, loc="upper left")

    fig.suptitle("RGI02 SMB Reconstruction: Model Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(FIG_DIR) / f"fig_model_comparison_logo_loyo_{timestamp}.png"
    saved_path = safe_savefig(fig, out_path)
    plt.close(fig)
    print(f"Saved: {saved_path}")


if __name__ == "__main__":
    main()
