"""Build publication-oriented uncertainty and residual diagnostics for Phys v2 CV runs."""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from config import PHYS_V2_RESULT_DIR, PHYS_V2_SEQUENCES_NPZ  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def regression_metrics(obs: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(obs) & np.isfinite(pred)
    obs, pred = obs[valid], pred[valid]
    if len(obs) < 2:
        return {name: np.nan for name in ["r2", "pearson_r", "rmse_mm", "mae_mm", "bias_mm"]}
    residual = pred - obs
    obs_centered = obs - np.mean(obs)
    pred_centered = pred - np.mean(pred)
    sum_squared_observed = float(np.sum(obs_centered**2))
    correlation_denominator = float(
        np.sqrt(sum_squared_observed * np.sum(pred_centered**2))
    )
    return {
        "r2": float(1.0 - np.sum(residual**2) / sum_squared_observed),
        "pearson_r": float(
            np.sum(obs_centered * pred_centered) / correlation_denominator
        ) if correlation_denominator > 0 else np.nan,
        "rmse_mm": float(np.sqrt(np.mean(residual**2)) * 1000.0),
        "mae_mm": float(np.mean(np.abs(residual)) * 1000.0),
        "bias_mm": float(np.mean(residual) * 1000.0),
    }


def infer_run(path: str) -> tuple[str, str]:
    stem = Path(path).stem
    for cv in ["loyo_buffered", "logo_buffered", "forward", "logo", "loyo", "loso"]:
        suffix = f"_{cv}_predictions"
        if stem.endswith(suffix):
            return stem[: -len(suffix)], cv.upper()
    raise ValueError(f"Cannot infer CV scheme from {path}")


def add_metadata(frame: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    keys = ["glacier_id", "rgi_id", "year"]
    available = [key for key in keys if key in frame.columns and key in metadata.columns]
    if "o2region" not in frame.columns:
        frame = frame.merge(
            metadata[available + ["o2region"]].drop_duplicates(available),
            on=available,
            how="left",
            validate="many_to_one",
        )
    return frame


def bootstrap_cluster_column(cv: str) -> str:
    if cv in {"LOGO", "LOGO_BUFFERED"}:
        return "rgi_id"
    if cv == "LOSO":
        return "o2region"
    return "year"


def bootstrap_intervals(
    frame: pd.DataFrame,
    group_column: str,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    groups = frame[group_column].dropna().unique()
    if len(groups) < 2:
        return {}
    group_indices = {
        group: indices.to_numpy()
        for group, indices in frame.groupby(group_column, sort=False).groups.items()
    }
    observed = frame["obs_annual"].to_numpy()
    predicted = frame["pred_annual"].to_numpy()
    values: dict[str, list[float]] = {
        name: [] for name in ["r2", "pearson_r", "rmse_mm", "mae_mm", "bias_mm"]
    }
    for _ in range(n_bootstrap):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        indices = np.concatenate([group_indices[group] for group in sampled])
        metrics = regression_metrics(observed[indices], predicted[indices])
        for name, value in metrics.items():
            if np.isfinite(value):
                values[name].append(value)
    intervals: dict[str, float] = {}
    for name, samples in values.items():
        if samples:
            low, high = np.quantile(samples, [0.025, 0.975])
            intervals[f"{name}_ci_low"] = float(low)
            intervals[f"{name}_ci_high"] = float(high)
    return intervals


def residual_rows(model: str, cv: str, frame: pd.DataFrame) -> list[dict[str, float | str | int]]:
    obs = frame["obs_annual"].to_numpy()
    pred = frame["pred_annual"].to_numpy()
    lower, upper = np.quantile(obs, [0.10, 0.90])
    categories = {
        "all": np.ones(len(frame), dtype=bool),
        "most_negative_10pct": obs <= lower,
        "central_80pct": (obs > lower) & (obs < upper),
        "most_positive_10pct": obs >= upper,
    }
    rows: list[dict[str, float | str | int]] = []
    for category, mask in categories.items():
        values = regression_metrics(obs[mask], pred[mask])
        rows.append({"model": model, "cv": cv, "subset": category, "n": int(mask.sum()), **values})
    if len(frame) >= 3:
        fit = linregress(obs, pred)
        rows[0]["prediction_vs_observation_slope"] = float(fit.slope)
        rows[0]["prediction_vs_observation_intercept_m"] = float(fit.intercept)
    return rows


def paired_bootstrap_difference(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    group_column: str,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    keys = [column for column in ["glacier_id", "rgi_id", "year"] if column in candidate.columns]
    right = reference[keys + ["pred_annual"]].rename(columns={"pred_annual": "pred_reference"})
    paired = candidate.merge(right, on=keys, how="inner", validate="one_to_one")
    groups = paired[group_column].dropna().unique()
    group_indices = {
        group: indices.to_numpy()
        for group, indices in paired.groupby(group_column, sort=False).groups.items()
    }
    observed = paired["obs_annual"].to_numpy()
    candidate_prediction = paired["pred_annual"].to_numpy()
    reference_prediction = paired["pred_reference"].to_numpy()
    rmse_delta, r2_delta = [], []
    for _ in range(n_bootstrap):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        indices = np.concatenate([group_indices[group] for group in sampled])
        obs = observed[indices]
        candidate_metrics = regression_metrics(obs, candidate_prediction[indices])
        reference_metrics = regression_metrics(obs, reference_prediction[indices])
        rmse_delta.append(candidate_metrics["rmse_mm"] - reference_metrics["rmse_mm"])
        r2_delta.append(candidate_metrics["r2"] - reference_metrics["r2"])
    rmse_ci = np.quantile(rmse_delta, [0.025, 0.975])
    r2_ci = np.quantile(r2_delta, [0.025, 0.975])
    return {
        "n_common": len(paired),
        "rmse_delta_mm": float(np.mean(rmse_delta)),
        "rmse_delta_ci_low": float(rmse_ci[0]),
        "rmse_delta_ci_high": float(rmse_ci[1]),
        "r2_delta": float(np.mean(r2_delta)),
        "r2_delta_ci_low": float(r2_ci[0]),
        "r2_delta_ci_high": float(r2_ci[1]),
    }


def main() -> None:
    args = parse_args()
    data = np.load(PHYS_V2_SEQUENCES_NPZ, allow_pickle=True)
    metadata = pd.DataFrame(
        {
            "glacier_id": data["glacier_ids"],
            "rgi_id": data["rgi_ids"],
            "year": data["years"],
            "o2region": data["o2regions"],
        }
    )
    paths = sorted(glob.glob(os.path.join(PHYS_V2_RESULT_DIR, "**", "*_predictions.csv"), recursive=True))
    output_dir = os.path.join(PHYS_V2_RESULT_DIR, "publication_evaluation")
    os.makedirs(output_dir, exist_ok=True)

    metric_rows = []
    diagnostic_rows = []
    run_frames: dict[tuple[str, str], pd.DataFrame] = {}
    expected_n = len(data["y_annual"])
    rng = np.random.default_rng(args.seed)
    for path in paths:
        model, cv = infer_run(path)
        frame = pd.read_csv(path)
        if not {"obs_annual", "pred_annual"}.issubset(frame.columns):
            continue
        frame = add_metadata(frame, metadata)
        run_frames[(model, cv)] = frame
        group_column = bootstrap_cluster_column(cv)
        values = regression_metrics(frame.obs_annual.to_numpy(), frame.pred_annual.to_numpy())
        values.update(bootstrap_intervals(frame, group_column, args.bootstrap, rng))
        metric_rows.append(
            {
                "model": model,
                "cv": cv,
                "n": len(frame),
                "complete_oof": len(frame) == expected_n,
                "bootstrap_cluster": group_column,
                "n_clusters": frame[group_column].nunique(),
                **values,
            }
        )
        diagnostic_rows.extend(residual_rows(model, cv, frame))

    metrics = pd.DataFrame(metric_rows).sort_values(["cv", "rmse_mm"])
    diagnostics = pd.DataFrame(diagnostic_rows)
    paired_rows = []
    reference_candidates = [
        "xgboost_v2_compact_monthly_all_hyp_p-regularized",
        "xgboost_v2_compact_monthly_all_hyp",
        "xgboost_v2_compact",
    ]
    for cv in sorted({key[1] for key in run_frames}):
        reference_name = next(
            (name for name in reference_candidates if (name, cv) in run_frames), None
        )
        if reference_name is None:
            continue
        reference = run_frames[(reference_name, cv)]
        group_column = bootstrap_cluster_column(cv)
        for (model, model_cv), frame in run_frames.items():
            if model_cv != cv or model == reference_name or len(frame) != len(reference):
                continue
            differences = paired_bootstrap_difference(
                frame, reference, group_column, args.bootstrap, rng
            )
            if differences["n_common"] != len(reference):
                continue
            paired_rows.append(
                {"model": model, "reference": reference_name, "cv": cv, **differences}
            )
    paired = pd.DataFrame(paired_rows)
    metrics_path = os.path.join(output_dir, "model_metrics_cluster_bootstrap.csv")
    diagnostics_path = os.path.join(output_dir, "residual_extreme_diagnostics.csv")
    paired_path = os.path.join(output_dir, "paired_differences_vs_xgboost_compact.csv")
    metrics.to_csv(metrics_path, index=False)
    diagnostics.to_csv(diagnostics_path, index=False)
    paired.to_csv(paired_path, index=False)
    print(metrics.to_string(index=False))
    print(f"Saved -> {metrics_path}")
    print(f"Saved -> {diagnostics_path}")
    print(f"Saved -> {paired_path}")


if __name__ == "__main__":
    main()
