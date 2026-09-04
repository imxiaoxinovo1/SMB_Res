"""Build simple prediction-level ensembles for QC validation outputs."""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from config import RESULT_DIR  # noqa: E402


MODEL_PREDICTIONS = {
    "base": {
        "logo": os.path.join(RESULT_DIR, "baseline_qc", "glacierformer_base_qc_logo_predictions.csv"),
        "loyo": os.path.join(RESULT_DIR, "baseline_qc", "glacierformer_base_qc_loyo_predictions.csv"),
    },
    "hypsometry": {
        "logo": os.path.join(
            RESULT_DIR,
            "hypsometry_qc",
            "glacierformer_hypsometry_qc_logo_predictions.csv",
        ),
        "loyo": os.path.join(
            RESULT_DIR,
            "hypsometry_qc",
            "glacierformer_hypsometry_qc_loyo_predictions.csv",
        ),
    },
    "xgboost": {
        "logo": os.path.join(RESULT_DIR, "xgboost_qc", "xgboost_qc_logo_predictions.csv"),
        "loyo": os.path.join(RESULT_DIR, "xgboost_qc", "xgboost_qc_loyo_predictions.csv"),
    },
}

MODEL_SHORT_NAMES = {
    "base": "base",
    "hypsometry": "hyp",
    "xgboost": "xgb",
}


def evaluate(obs: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    r, _ = pearsonr(obs, pred)
    return {
        "r2": float(r2_score(obs, pred)),
        "pearson_r": float(r),
        "rmse_mm": float(np.sqrt(mean_squared_error(obs, pred)) * 1000.0),
        "bias_mm": float(np.mean(pred - obs) * 1000.0),
    }


def safe_to_csv(frame: pd.DataFrame, path: str) -> str:
    """Write CSV, falling back to a timestamped name if the target is locked."""
    csv_text = frame.to_csv(index=False)
    try:
        Path(path).write_text(csv_text, encoding="utf-8")
        return path
    except PermissionError:
        root, ext = os.path.splitext(path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback = f"{root}_{timestamp}{ext}"
        try:
            Path(fallback).write_text(csv_text, encoding="utf-8")
            print(f"WARNING: target file is locked, wrote fallback file: {fallback}")
            return fallback
        except PermissionError:
            print(
                "WARNING: could not write ensemble CSV output. "
                f"Original path: {path}; fallback path: {fallback}"
            )
            return ""


def load_pair(model_a: str, model_b: str, cv: str) -> pd.DataFrame:
    a = pd.read_csv(MODEL_PREDICTIONS[model_a][cv])
    b = pd.read_csv(MODEL_PREDICTIONS[model_b][cv])
    key_cols = ["glacier_id", "year"]

    duplicate_a = int(a.duplicated(key_cols).sum())
    duplicate_b = int(b.duplicated(key_cols).sum())
    if duplicate_a or duplicate_b:
        raise RuntimeError(
            f"Duplicate glacier-year rows found: {model_a}={duplicate_a}, {model_b}={duplicate_b}"
        )

    merged = a.merge(
        b,
        on=key_cols,
        suffixes=(f"_{model_a}", f"_{model_b}"),
        how="inner",
    )
    if len(merged) != len(a) or len(merged) != len(b):
        raise RuntimeError(
            f"Prediction rows do not align: {model_a}={len(a)}, {model_b}={len(b)}, merged={len(merged)}"
        )

    obs_a = merged[f"obs_{model_a}"].astype(float)
    obs_b = merged[f"obs_{model_b}"].astype(float)
    max_obs_diff = float((obs_a - obs_b).abs().max())
    if max_obs_diff > 1e-5:
        raise RuntimeError(
            f"Observed SMB values differ after glacier-year alignment: max_abs_diff={max_obs_diff}"
        )
    merged["obs"] = obs_a
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv", choices=["logo", "loyo"], required=True)
    parser.add_argument("--model-a", choices=sorted(MODEL_PREDICTIONS), default="xgboost")
    parser.add_argument("--model-b", choices=sorted(MODEL_PREDICTIONS), default="hypsometry")
    args = parser.parse_args()

    out_dir = os.path.join(RESULT_DIR, "ensemble_qc", "current")
    os.makedirs(out_dir, exist_ok=True)

    merged = load_pair(args.model_a, args.model_b, args.cv)
    pred_a = merged[f"pred_{args.model_a}"].values.astype(float)
    pred_b = merged[f"pred_{args.model_b}"].values.astype(float)
    obs = merged["obs"].values.astype(float)

    rows = []
    best = None
    for alpha in np.linspace(0.0, 1.0, 21):
        pred = alpha * pred_a + (1.0 - alpha) * pred_b
        metrics = evaluate(obs, pred)
        row = {
            "model": f"ensemble_{args.model_a}_{args.model_b}",
            "cv": args.cv.upper(),
            "alpha_model_a": float(alpha),
            "model_a": args.model_a,
            "model_b": args.model_b,
            **metrics,
        }
        rows.append(row)
        if best is None or row["rmse_mm"] < best["rmse_mm"]:
            best = row

    assert best is not None
    alpha = best["alpha_model_a"]
    merged["pred"] = alpha * pred_a + (1.0 - alpha) * pred_b
    name = f"{MODEL_SHORT_NAMES[args.model_a]}_{MODEL_SHORT_NAMES[args.model_b]}_{args.cv}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_to_csv(
        merged[["glacier_id", "year", "obs", "pred"]],
        os.path.join(out_dir, f"ens_{name}_{timestamp}.csv"),
    )
    safe_to_csv(
        pd.DataFrame(rows),
        os.path.join(out_dir, f"alpha_{name}_{timestamp}.csv"),
    )
    safe_to_csv(
        pd.DataFrame([best]),
        os.path.join(out_dir, f"summary_{name}_{timestamp}.csv"),
    )

    print(
        f"Best {args.cv.upper()} ensemble: alpha({args.model_a})={alpha:.2f}, "
        f"alpha({args.model_b})={1.0-alpha:.2f}, "
        f"R2={best['r2']:.4f}, R={best['pearson_r']:.4f}, "
        f"RMSE={best['rmse_mm']:.1f}mm, Bias={best['bias_mm']:.1f}mm"
    )
    print(f"Saved ensemble outputs to: {out_dir}")


if __name__ == "__main__":
    main()
