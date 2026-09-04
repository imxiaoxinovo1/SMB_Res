"""Collect validation summaries into comparison tables."""
from __future__ import annotations

import glob
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from config import RESULT_DIR  # noqa: E402


MODEL_ORDER = [
    "glacierformer_base",
    "glacierformer_base_qc",
    "glacierformer_hypsometry_qc",
    "xgboost_qc",
    "ensemble_xgboost_hypsometry",
    "ensemble_xgboost_base",
]


def discover_summary_files() -> list[str]:
    """Find both legacy '*_summary.csv' and ensemble 'summary_*.csv' files."""
    patterns = [
        os.path.join(RESULT_DIR, "**", "*_summary.csv"),
        os.path.join(RESULT_DIR, "**", "summary_*.csv"),
    ]
    paths: list[str] = []
    for pattern in patterns:
        paths.extend(glob.glob(pattern, recursive=True))
    return sorted(set(paths))


def read_summary(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = [str(col).strip() for col in frame.columns]
    frame = frame.dropna(how="all")
    frame["source_file"] = os.path.relpath(path, RESULT_DIR)
    return frame


def safe_to_csv(frame: pd.DataFrame, path: str) -> str:
    csv_text = frame.to_csv(index=False)
    try:
        Path(path).write_text(csv_text, encoding="utf-8")
        return path
    except PermissionError:
        root, ext = os.path.splitext(path)
        fallback = f"{root}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
        try:
            Path(fallback).write_text(csv_text, encoding="utf-8")
            print(f"WARNING: target file is locked, wrote fallback file: {fallback}")
            return fallback
        except PermissionError:
            analysis_dir = Path(PROJECT_DIR) / "docs" / "analysis"
            analysis_path = analysis_dir / Path(fallback).name
            try:
                analysis_dir.mkdir(parents=True, exist_ok=True)
                analysis_path.write_text(csv_text, encoding="utf-8")
                print(f"WARNING: results directory is locked, wrote analysis copy: {analysis_path}")
                return str(analysis_path)
            except PermissionError:
                print(
                    "WARNING: could not write comparison CSV. "
                    f"Original path: {path}; fallback path: {fallback}; analysis path: {analysis_path}"
                )
                return ""


def main() -> None:
    paths = discover_summary_files()
    rows = []
    for path in paths:
        rows.append(read_summary(path))

    if not rows:
        raise FileNotFoundError(f"No summary files found under {RESULT_DIR}")

    out = pd.concat(rows, ignore_index=True)
    for col in ["r2", "pearson_r", "rmse_mm", "bias_mm", "alpha_model_a"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out["cv"] = out["cv"].astype(str).str.upper()
    out["model"] = out["model"].astype(str)
    out["model_rank"] = out["model"].map(
        {model: rank for rank, model in enumerate(MODEL_ORDER)}
    ).fillna(999).astype(int)

    out = out.sort_values(["cv", "rmse_mm", "model_rank"], ascending=[True, True, True])
    out = out.drop(columns=["model_rank"])
    out_path = os.path.join(RESULT_DIR, "model_comparison_qc.csv")
    written_out_path = safe_to_csv(out, out_path)

    metric_cols = ["r2", "pearson_r", "rmse_mm", "bias_mm"]
    wide = out.pivot_table(index="model", columns="cv", values=metric_cols, aggfunc="first")
    wide.columns = [f"{metric}_{cv}" for metric, cv in wide.columns]
    wide = wide.reset_index()
    wide["model_rank"] = wide["model"].map(
        {model: rank for rank, model in enumerate(MODEL_ORDER)}
    ).fillna(999).astype(int)
    wide = wide.sort_values("model_rank").drop(columns=["model_rank"])
    wide_path = os.path.join(RESULT_DIR, "model_comparison_qc_wide.csv")
    written_wide_path = safe_to_csv(wide, wide_path)

    print(out.to_string(index=False))
    print(f"Saved: {written_out_path}")
    print(f"Saved: {written_wide_path}")


if __name__ == "__main__":
    main()
