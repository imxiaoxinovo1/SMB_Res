"""Train the selected corrected XGBoost model on all available WGMS labels."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from config import (  # noqa: E402
    PHYS_V2_FINAL_DIR,
    PHYS_V2_FINAL_IMPORTANCE,
    PHYS_V2_FINAL_MODEL,
    PHYS_V2_FINAL_PREPROCESSOR,
    PHYS_V2_SEQUENCES_NPZ,
)
from train_tree_v2_cv import FEATURE_SETS, calendar_flatten, hypsometry_quantiles  # noqa: E402
from train_xgboost_v2_nested_cv import CANDIDATES, make_model  # noqa: E402


BASE_STATIC_FEATURES = [
    "slope_deg", "aspect_sin", "aspect_cos", "zmin_m", "zmax_m", "zmean_m", "zmed_m",
    "log1p_area_km2", "log1p_lmax_m", "cenlat", "cenlon", "clim_annual_t2m",
    "clim_winter_tp", "clim_summer_t2m", "clim_summer_ssrd", "clim_t2m_amplitude",
]
HYP_FEATURES = ["hyp_q10_m", "hyp_q25_m", "hyp_q50_m", "hyp_q75_m", "hyp_q90_m"]


def main() -> None:
    data = np.load(PHYS_V2_SEQUENCES_NPZ, allow_pickle=True)
    dynamic_names = [str(name) for name in data["dynamic_features"]]
    selected_dynamic = FEATURE_SETS["compact"]
    dynamic_indices = [dynamic_names.index(name) for name in selected_dynamic]
    dynamic = calendar_flatten(data["X_dyn"][:, :, dynamic_indices], data["month_ids"])

    stored_static = [str(name) for name in data["static_features"]]
    static_indices = [stored_static.index(name) for name in BASE_STATIC_FEATURES]
    static = data["X_sta"][:, static_indices]
    hyp = hypsometry_quantiles(data["X_hyp"], data["hypsometry_band_centers_m"])
    x = np.column_stack([dynamic, static, hyp]).astype(np.float32)
    y = data["y_annual"].astype(np.float32)
    medians = np.nanmedian(x, axis=0)
    x = np.where(np.isnan(x), medians, x)

    candidate = next(item for item in CANDIDATES if item["candidate"] == "regularized")
    model = make_model(candidate, seed=42)
    model.fit(x, y)

    month_feature_names = [
        f"{name}_m{month:02d}" for name in selected_dynamic for month in range(1, 13)
    ]
    feature_names = month_feature_names + BASE_STATIC_FEATURES + HYP_FEATURES
    if len(feature_names) != x.shape[1]:
        raise RuntimeError("Feature-name count does not match the final training matrix.")

    os.makedirs(PHYS_V2_FINAL_DIR, exist_ok=True)
    model.save_model(PHYS_V2_FINAL_MODEL)
    np.savez_compressed(
        PHYS_V2_FINAL_PREPROCESSOR,
        medians=medians.astype(np.float32),
        feature_min=np.min(x, axis=0).astype(np.float32),
        feature_max=np.max(x, axis=0).astype(np.float32),
        feature_names=np.asarray(feature_names),
        dynamic_features=np.asarray(selected_dynamic),
        static_features=np.asarray(BASE_STATIC_FEATURES),
        hypsometry_features=np.asarray(HYP_FEATURES),
        training_area_min_km2=float(np.min(np.expm1(static[:, BASE_STATIC_FEATURES.index("log1p_area_km2")]))),
        training_area_max_km2=float(np.max(np.expm1(static[:, BASE_STATIC_FEATURES.index("log1p_area_km2")]))),
    )
    pd.DataFrame(
        {"feature": feature_names, "gain_importance": model.feature_importances_}
    ).sort_values("gain_importance", ascending=False).to_csv(PHYS_V2_FINAL_IMPORTANCE, index=False)
    metadata = {
        "model": "XGBoost",
        "candidate": candidate,
        "n_samples": int(len(y)),
        "n_unique_rgi_glaciers": int(len(np.unique(data["rgi_ids"]))),
        "target": "annual glacier-wide SMB (m w.e. yr-1)",
        "validation_note": "Use nested LOGO/LOYO outputs; training-set metrics are intentionally omitted.",
    }
    with open(os.path.join(PHYS_V2_FINAL_DIR, "training_metadata.json"), "w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    print(f"Samples: {len(y)}; features: {x.shape[1]}")
    print(f"Saved model -> {PHYS_V2_FINAL_MODEL}")
    print(f"Saved preprocessor -> {PHYS_V2_FINAL_PREPROCESSOR}")
    print(f"Saved importance -> {PHYS_V2_FINAL_IMPORTANCE}")


if __name__ == "__main__":
    main()
