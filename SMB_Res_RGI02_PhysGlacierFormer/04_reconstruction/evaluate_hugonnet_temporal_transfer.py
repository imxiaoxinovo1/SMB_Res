"""Test whether an early Hugonnet offset transfers to an independent later period."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from config import (  # noqa: E402
    HUGONNET_MULTIPERIOD_LABELS_CSV,
    HYPSOMETRY_RECON_RAW_CSV,
    RECONSTRUCTION_DIR,
)


EARLY_PERIOD = "2000-01-01_2010-01-01"
LATE_PERIOD = "2010-01-01_2020-01-01"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reconstruction", default=HYPSOMETRY_RECON_RAW_CSV)
    parser.add_argument("--prediction-column", default="predicted_smb_m")
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default=os.path.join(RECONSTRUCTION_DIR, "hugonnet_temporal_transfer_qc.csv"),
    )
    return parser.parse_args()


def period_predictions(reconstruction: pd.DataFrame, labels: pd.DataFrame, column: str) -> pd.DataFrame:
    annual = reconstruction.set_index(["rgi_id", "year"])[column]
    rows = []
    for row in labels.itertuples(index=False):
        years = range(int(row.period_start_year), int(row.period_end_year))
        values = [annual.get((row.rgi_id, year), np.nan) for year in years]
        if not np.isfinite(values).all():
            continue
        model_mean = float(np.mean(values))
        rows.append(
            {
                "rgi_id": row.rgi_id,
                "period": row.period,
                "model_mean": model_mean,
                "geodetic_mean": float(row.hugonnet_dmdtda_mwe_yr),
                "residual": float(row.hugonnet_dmdtda_mwe_yr - model_mean),
            }
        )
    return pd.DataFrame(rows)


def score(residual: np.ndarray) -> dict[str, float]:
    return {
        "rmse_mwe_yr": float(np.sqrt(np.mean(residual**2))),
        "mae_mwe_yr": float(np.mean(np.abs(residual))),
        "model_minus_geodetic_bias_mwe_yr": float(-np.mean(residual)),
    }


def main() -> None:
    args = parse_args()
    reconstruction = pd.read_csv(args.reconstruction)
    labels = pd.read_csv(HUGONNET_MULTIPERIOD_LABELS_CSV)
    if "qc_pass" in labels:
        labels = labels[labels["qc_pass"]].copy()
    labels = labels[labels["period"].isin([EARLY_PERIOD, LATE_PERIOD])].copy()
    periods = period_predictions(reconstruction, labels, args.prediction_column)

    early = periods[periods.period == EARLY_PERIOD][["rgi_id", "residual"]].rename(
        columns={"residual": "early_residual"}
    )
    late = periods[periods.period == LATE_PERIOD].merge(
        early, on="rgi_id", how="inner", validate="one_to_one"
    )
    if late.empty:
        raise RuntimeError("No glaciers have both independent decadal constraints.")

    rng = np.random.default_rng(args.seed)
    rows = []
    for shrink in [0.0, 0.25, 0.5, 0.75, 1.0]:
        offset = shrink * late.early_residual.clip(-args.clip, args.clip)
        residual = late.geodetic_mean - (late.model_mean + offset)
        bootstrap_rmse = []
        for _ in range(args.bootstrap):
            indices = rng.integers(0, len(late), size=len(late))
            bootstrap_rmse.append(float(np.sqrt(np.mean(residual.to_numpy()[indices] ** 2))))
        low, high = np.quantile(bootstrap_rmse, [0.025, 0.975])
        rows.append(
            {
                "calibration_period": EARLY_PERIOD,
                "independent_evaluation_period": LATE_PERIOD,
                "n_glaciers": len(late),
                "clip_abs_mwe_yr": args.clip,
                "shrink_factor": shrink,
                **score(residual.to_numpy()),
                "rmse_ci_low": float(low),
                "rmse_ci_high": float(high),
            }
        )
    result = pd.DataFrame(rows)
    result["early_late_residual_correlation"] = float(
        np.corrcoef(late.early_residual, late.residual)[0, 1]
    )
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
