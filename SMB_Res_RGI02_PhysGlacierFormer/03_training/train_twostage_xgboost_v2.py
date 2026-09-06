"""Leakage-safe two-stage spatial-mean plus annual-anomaly XGBoost experiment."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from config import PHYS_V2_RESULT_DIR, PHYS_V2_SEQUENCES_NPZ  # noqa: E402
from train_tree_v2_cv import (  # noqa: E402
    FEATURE_SETS,
    calendar_flatten,
    haversine_km,
    hypsometry_quantiles,
    make_model,
    metrics,
)


ANOMALY_FEATURES = [
    "t2m_anomaly",
    "tp_anomaly",
    "sf_anomaly",
    "ssrd_anomaly",
    "asn_anomaly",
]
STATIC_FEATURES = [
    "slope_deg",
    "aspect_sin",
    "aspect_cos",
    "zmin_m",
    "zmax_m",
    "zmean_m",
    "zmed_m",
    "log1p_area_km2",
    "log1p_lmax_m",
    "cenlat",
    "cenlon",
    "clim_annual_t2m",
    "clim_winter_tp",
    "clim_summer_t2m",
    "clim_summer_ssrd",
    "clim_t2m_amplitude",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cv",
        choices=["logo", "logo_buffered", "loyo", "loyo_buffered", "loso", "forward"],
        required=True,
    )
    parser.add_argument("--spatial-model", choices=["ridge", "randomforest"], default="ridge")
    parser.add_argument("--dynamic-set", choices=["anomalies", "compact"], default="anomalies")
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--forward-start-year", type=int, default=1980)
    parser.add_argument("--forward-min-train-samples", type=int, default=100)
    parser.add_argument("--spatial-buffer-km", type=float, default=50.0)
    return parser.parse_args()


def fold_masks(
    cv: str,
    held_out,
    groups: np.ndarray,
    years: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    spatial_buffer_km: float,
) -> tuple[np.ndarray, np.ndarray]:
    test = groups == held_out
    if cv == "forward":
        train = years < int(held_out)
    elif cv == "logo_buffered":
        distance = haversine_km(
            lat,
            lon,
            float(np.mean(lat[test])),
            float(np.mean(lon[test])),
        )
        train = (~test) & (distance > spatial_buffer_km)
    elif cv == "loyo_buffered":
        train = np.abs(years - int(held_out)) > 1
    else:
        train = ~test
    return train, test


def impute_from_train(
    values: np.ndarray,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    medians = np.nanmedian(values[train_mask], axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    return np.where(np.isnan(values), medians, values).astype(np.float32), medians


def glacier_level_training_data(
    spatial: np.ndarray,
    target: np.ndarray,
    rgi_ids: np.ndarray,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    for rgi_id in np.unique(rgi_ids[train_mask]):
        selected = train_mask & (rgi_ids == rgi_id)
        rows.append((rgi_id, spatial[np.flatnonzero(selected)[0]], float(np.mean(target[selected]))))
    return (
        np.asarray([row[0] for row in rows]),
        np.asarray([row[1] for row in rows], dtype=np.float32),
        np.asarray([row[2] for row in rows], dtype=np.float32),
    )


def cross_fitted_spatial_mean(
    spatial: np.ndarray,
    target: np.ndarray,
    rgi_ids: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    model_name: str,
    inner_folds: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return OOF train means and full-fit test means without glacier-ID leakage."""
    train_ids, glacier_x, glacier_y = glacier_level_training_data(
        spatial, target, rgi_ids, train_mask
    )
    n_splits = min(inner_folds, len(train_ids))
    if n_splits < 2:
        raise RuntimeError("Two-stage model requires at least two training glaciers.")

    oof_by_glacier: dict[object, float] = {}
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for inner_index, (fit_index, val_index) in enumerate(splitter.split(glacier_x)):
        inner_train = np.zeros(len(glacier_x), dtype=bool)
        inner_train[fit_index] = True
        inner_x, _ = impute_from_train(glacier_x, inner_train)
        model = make_model(model_name, seed + inner_index)
        model.fit(inner_x[fit_index], glacier_y[fit_index])
        for rgi_id, prediction in zip(train_ids[val_index], model.predict(inner_x[val_index])):
            oof_by_glacier[rgi_id] = float(prediction)

    train_prediction = np.full(len(target), np.nan, dtype=np.float32)
    for rgi_id, prediction in oof_by_glacier.items():
        train_prediction[train_mask & (rgi_ids == rgi_id)] = prediction
    if not np.isfinite(train_prediction[train_mask]).all():
        raise RuntimeError("Incomplete cross-fitted spatial predictions.")

    full_model = make_model(model_name, seed + 10_000)
    full_x, medians = impute_from_train(glacier_x, np.ones(len(glacier_x), dtype=bool))
    full_model.fit(full_x, glacier_y)
    test_x = np.where(np.isnan(spatial[test_mask]), medians, spatial[test_mask])
    test_prediction = full_model.predict(test_x).astype(np.float32)
    return train_prediction, test_prediction


def main() -> None:
    args = parse_args()
    data = np.load(PHYS_V2_SEQUENCES_NPZ, allow_pickle=True)
    dynamic_names = [str(value) for value in data["dynamic_features"]]
    selected_dynamic = (
        ANOMALY_FEATURES if args.dynamic_set == "anomalies" else FEATURE_SETS["compact"]
    )
    dynamic_indices = [dynamic_names.index(name) for name in selected_dynamic]
    dynamic = calendar_flatten(
        data["X_dyn"][:, :, dynamic_indices], data["month_ids"]
    )

    static_names = [str(value) for value in data["static_features"]]
    static_indices = [static_names.index(name) for name in STATIC_FEATURES]
    spatial = np.column_stack(
        [
            data["X_sta"][:, static_indices],
            hypsometry_quantiles(data["X_hyp"], data["hypsometry_band_centers_m"]),
        ]
    ).astype(np.float32)

    target = data["y_annual"].astype(np.float32)
    rgi_ids = data["rgi_ids"]
    years = data["years"].astype(int)
    regions = data["o2regions"]
    lat = data["X_sta"][:, static_names.index("cenlat")]
    lon = data["X_sta"][:, static_names.index("cenlon")]
    if args.cv in {"logo", "logo_buffered"}:
        groups = rgi_ids
    elif args.cv == "loso":
        groups = regions
    else:
        groups = years

    folds = np.unique(groups)
    if args.cv == "forward":
        folds = folds[folds >= args.forward_start_year]
    predictions = np.full(len(target), np.nan, dtype=np.float32)
    spatial_predictions = np.full(len(target), np.nan, dtype=np.float32)
    anomaly_predictions = np.full(len(target), np.nan, dtype=np.float32)

    for fold_index, held_out in enumerate(folds):
        train, test = fold_masks(
            args.cv,
            held_out,
            groups,
            years,
            lat,
            lon,
            args.spatial_buffer_km,
        )
        if args.cv == "forward" and int(train.sum()) < args.forward_min_train_samples:
            print(f"Skipping {held_out}: only {int(train.sum())} prior samples")
            continue
        imputed_dynamic, _ = impute_from_train(dynamic, train)
        train_spatial, test_spatial = cross_fitted_spatial_mean(
            spatial,
            target,
            rgi_ids,
            train,
            test,
            args.spatial_model,
            args.inner_folds,
            args.seed + fold_index * 100,
        )
        residual_target = target[train] - train_spatial[train]
        residual_model = make_model(
            "xgboost", args.seed + fold_index, xgb_profile="regularized"
        )
        residual_model.fit(imputed_dynamic[train], residual_target)
        test_anomaly = residual_model.predict(imputed_dynamic[test]).astype(np.float32)
        spatial_predictions[test] = test_spatial
        anomaly_predictions[test] = test_anomaly
        predictions[test] = test_spatial + test_anomaly
        print(
            f"[{fold_index + 1:02d}/{len(folds)}] held_out={held_out} "
            f"n_train={train.sum()} n_test={test.sum()}"
        )

    valid = np.isfinite(predictions)
    tag = f"twostage_{args.spatial_model}_{args.dynamic_set}_xgboost_v2"
    if args.cv == "forward" and args.forward_start_year != 1980:
        tag += f"_start{args.forward_start_year}"
    if args.cv == "logo_buffered" and args.spatial_buffer_km != 50.0:
        tag += f"_buffer{args.spatial_buffer_km:g}km"
    result_dir = os.path.join(PHYS_V2_RESULT_DIR, tag)
    os.makedirs(result_dir, exist_ok=True)
    frame = pd.DataFrame(
        {
            "glacier_id": data["glacier_ids"][valid],
            "rgi_id": rgi_ids[valid],
            "year": years[valid],
            "obs_annual": target[valid],
            "pred_annual": predictions[valid],
            "pred_spatial_mean": spatial_predictions[valid],
            "pred_annual_anomaly": anomaly_predictions[valid],
        }
    )
    frame.to_csv(os.path.join(result_dir, f"{tag}_{args.cv}_predictions.csv"), index=False)
    result = {
        "model": tag,
        "cv": args.cv.upper(),
        "evaluation_year_min": int(frame["year"].min()),
        "evaluation_year_max": int(frame["year"].max()),
        **metrics(frame["obs_annual"].to_numpy(), frame["pred_annual"].to_numpy()),
    }
    summary = pd.DataFrame([result])
    summary.to_csv(os.path.join(result_dir, f"{tag}_{args.cv}_summary.csv"), index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
