"""Calibrate cluster-excluded prediction intervals from out-of-fold residuals."""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from config import PHYS_V2_INTERVAL_CALIBRATION, PHYS_V2_RESULT_DIR  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def prediction_file(cv: str) -> str:
    pattern = os.path.join(
        PHYS_V2_RESULT_DIR,
        "xgboost_v2_compact_monthly_all_hyp_p-regularized",
        f"*_p-regularized_{cv}_predictions.csv",
    )
    matches = glob.glob(pattern)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one regularized {cv.upper()} prediction file, found {matches}")
    return matches[0]


def conformal_quantile(values: np.ndarray, coverage: float) -> float:
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan
    probability = min(1.0, np.ceil((len(values) + 1) * coverage) / len(values))
    return float(np.quantile(values, probability, method="higher"))


def cluster_bootstrap_coverage(
    frame: pd.DataFrame,
    group_column: str,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    groups = frame[group_column].unique()
    grouped = {group: part for group, part in frame.groupby(group_column)}
    estimates = []
    for _ in range(n_bootstrap):
        sample = rng.choice(groups, size=len(groups), replace=True)
        boot = pd.concat([grouped[group] for group in sample], ignore_index=True)
        estimates.append(float(boot["covered"].mean()))
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    rows = []
    for cv, group_column in [("logo", "rgi_id"), ("loyo", "year")]:
        frame = pd.read_csv(prediction_file(cv))
        frame["absolute_residual_m"] = (frame["pred_annual"] - frame["obs_annual"]).abs()
        for coverage in [0.80, 0.90, 0.95]:
            half_width = np.empty(len(frame), dtype=float)
            for group, indices in frame.groupby(group_column).groups.items():
                calibration = frame.loc[frame[group_column] != group, "absolute_residual_m"].to_numpy()
                half_width[np.asarray(list(indices), dtype=int)] = conformal_quantile(calibration, coverage)
            frame["half_width_m"] = half_width
            frame["covered"] = frame["absolute_residual_m"] <= frame["half_width_m"]
            low, high = cluster_bootstrap_coverage(
                frame, group_column, args.bootstrap, rng
            )
            global_width = conformal_quantile(frame["absolute_residual_m"].to_numpy(), coverage)
            rows.append(
                {
                    "cv": cv.upper(),
                    "target_coverage": coverage,
                    "n_samples": len(frame),
                    "n_clusters": frame[group_column].nunique(),
                    "cluster_column": group_column,
                    "cluster_excluded_empirical_coverage": float(frame["covered"].mean()),
                    "coverage_ci_low": low,
                    "coverage_ci_high": high,
                    "mean_cluster_excluded_half_width_m": float(frame["half_width_m"].mean()),
                    "final_global_half_width_m": global_width,
                    "method": "absolute OOF residual, leave-target-cluster excluded",
                }
            )
    result = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(PHYS_V2_INTERVAL_CALIBRATION), exist_ok=True)
    result.to_csv(PHYS_V2_INTERVAL_CALIBRATION, index=False)
    print(result.to_string(index=False))
    print(f"Saved -> {PHYS_V2_INTERVAL_CALIBRATION}")


if __name__ == "__main__":
    main()
