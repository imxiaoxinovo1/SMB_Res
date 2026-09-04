"""Classical-model baselines on the corrected PhysGlacierFormer v2 dataset."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from config import PHYS_V2_RESULT_DIR, PHYS_V2_SEQUENCES_NPZ  # noqa: E402


FEATURE_SETS = {
    "compact": [
        "t2m", "sd", "asn", "tp", "sf", "ssrd", "str", "slhf", "sshf",
        "t2m_anomaly", "tp_anomaly", "sf_anomaly", "ssrd_anomaly", "asn_anomaly",
    ],
    "minimal": ["t2m", "tp", "t2m_anomaly", "tp_anomaly"],
    "raw_compact": ["t2m", "sd", "asn", "tp", "sf", "ssrd", "str", "slhf", "sshf"],
    "compact_no_asn": [
        "t2m", "sd", "tp", "sf", "ssrd", "str", "slhf", "sshf",
        "t2m_anomaly", "tp_anomaly", "sf_anomaly", "ssrd_anomaly",
    ],
    "physical": [
        "t2m", "sd", "asn", "tp", "sf", "smlt", "ssrd", "str", "slhf", "sshf",
        "relative_humidity", "t2m_anomaly", "tp_anomaly", "sf_anomaly",
        "ssrd_anomaly", "asn_anomaly",
    ],
    "downscaled": [
        "t2m_lapse", "sd", "asn", "tp", "sf", "smlt", "ssrd", "str", "slhf", "sshf",
        "relative_humidity", "t2m_anomaly", "tp_anomaly", "sf_anomaly",
        "ssrd_anomaly", "asn_anomaly",
    ],
}


def metrics(obs: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(obs) & np.isfinite(pred)
    obs, pred = obs[valid], pred[valid]
    return {
        "n": len(obs),
        "r2": float(r2_score(obs, pred)),
        "pearson_r": float(pearsonr(obs, pred).statistic),
        "rmse_mm": float(np.sqrt(mean_squared_error(obs, pred)) * 1000.0),
        "mae_mm": float(mean_absolute_error(obs, pred) * 1000.0),
        "bias_mm": float(np.mean(pred - obs) * 1000.0),
    }


def hypsometry_quantiles(x_hyp: np.ndarray, centers: np.ndarray) -> np.ndarray:
    cumulative = np.cumsum(x_hyp[:, :, 0], axis=1)
    output = []
    for quantile in [0.10, 0.25, 0.50, 0.75, 0.90]:
        indices = np.argmax(cumulative >= quantile, axis=1)
        output.append(centers[indices])
    return np.column_stack(output).astype(np.float32)


def haversine_km(lat: np.ndarray, lon: np.ndarray, target_lat: float, target_lon: float) -> np.ndarray:
    lat_rad = np.deg2rad(lat)
    target_lat_rad = np.deg2rad(target_lat)
    delta_lat = lat_rad - target_lat_rad
    delta_lon = np.deg2rad(lon - target_lon)
    value = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat_rad) * np.cos(target_lat_rad) * np.sin(delta_lon / 2.0) ** 2
    )
    return 6371.0088 * 2.0 * np.arcsin(np.sqrt(np.clip(value, 0.0, 1.0)))


def calendar_flatten(x_dyn: np.ndarray, month_ids: np.ndarray) -> np.ndarray:
    n_samples, _, n_features = x_dyn.shape
    output = np.empty((n_samples, n_features * 12), dtype=np.float32)
    for sample in range(n_samples):
        calendar = np.empty((12, n_features), dtype=np.float32)
        calendar[month_ids[sample] - 1] = x_dyn[sample]
        output[sample] = calendar.T.reshape(-1)
    return output


def seasonal_aggregate(
    x_dyn: np.ndarray,
    month_ids: np.ndarray,
    feature_names: list[str],
) -> np.ndarray:
    output = np.empty((len(x_dyn), len(feature_names)), dtype=np.float32)
    for sample in range(len(x_dyn)):
        months = month_ids[sample]
        for feature_index, name in enumerate(feature_names):
            if name.startswith(("tp", "sf")):
                mask = (months >= 10) | (months <= 4)
                output[sample, feature_index] = float(np.sum(x_dyn[sample, mask, feature_index]))
            elif name.startswith("sd"):
                mask = (months >= 10) | (months <= 4)
                output[sample, feature_index] = float(np.max(x_dyn[sample, mask, feature_index]))
            else:
                mask = (months >= 5) & (months <= 9)
                output[sample, feature_index] = float(np.mean(x_dyn[sample, mask, feature_index]))
    return output


def physical_seasonal_indices(
    x_dyn: np.ndarray,
    month_ids: np.ndarray,
    feature_names: list[str],
) -> np.ndarray:
    """Compute compact accumulation/ablation proxies from correctly scaled monthly data."""
    index = {name: position for position, name in enumerate(feature_names)}
    temperature_name = "t2m_lapse" if "t2m_lapse" in index else "t2m"
    required = {temperature_name, "sd", "asn", "tp", "sf", "smlt", "ssrd", "str", "slhf", "sshf"}
    missing = sorted(required - set(index))
    if missing:
        raise ValueError(f"Physical indices require missing features: {missing}")

    output = np.empty((len(x_dyn), 12), dtype=np.float32)
    for sample, months in enumerate(month_ids):
        accumulation = (months >= 10) | (months <= 4)
        ablation = (months >= 5) & (months <= 9)
        spring = (months >= 3) & (months <= 5)
        values = x_dyn[sample]
        t2m = values[:, index[temperature_name]]
        tp = values[:, index["tp"]]
        sf = values[:, index["sf"]]
        asn = np.clip(values[:, index["asn"]], 0.0, 1.0)
        ssrd = values[:, index["ssrd"]]
        turbulent_longwave = (
            values[:, index["str"]]
            + values[:, index["slhf"]]
            + values[:, index["sshf"]]
        )
        winter_tp = float(np.sum(tp[accumulation]))
        winter_sf = float(np.sum(sf[accumulation]))
        output[sample] = [
            winter_tp,
            winter_sf,
            winter_sf / max(winter_tp, 1e-6),
            float(np.max(values[spring, index["sd"]])),
            float(np.sum(np.maximum(t2m[ablation], 0.0))),
            float(np.mean(t2m[ablation])),
            float(np.sum(values[ablation, index["smlt"]])),
            float(np.mean(asn[ablation])),
            float(np.mean(ssrd[ablation] * (1.0 - asn[ablation]))),
            float(np.mean(turbulent_longwave[ablation])),
            float(np.mean(values[ablation, index["t2m_anomaly"]])),
            float(np.sum(values[accumulation, index["sf_anomaly"]])),
        ]
    return output


def monthly_physical_monotonic_constraints(
    feature_names: list[str],
    n_static: int,
    n_hypsometry: int,
    n_extra_dynamic: int = 0,
) -> tuple[int, ...]:
    """Encode only robust seasonal signs; leave ambiguous predictors unconstrained."""
    constraints: list[int] = []
    for name in feature_names:
        for month in range(1, 13):
            value = 0
            if name in {"t2m", "t2m_anomaly", "ssrd", "ssrd_anomaly"} and 5 <= month <= 9:
                value = -1
            elif name in {"sf", "sf_anomaly"} and (month >= 10 or month <= 4):
                value = 1
            constraints.append(value)
    constraints.extend([0] * (n_extra_dynamic + n_static + n_hypsometry))
    return tuple(constraints)


def make_model(name: str, seed: int, xgb_profile: str = "reference"):
    if name == "dummy":
        return DummyRegressor(strategy="mean")
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    if name == "xgboost":
        profiles = {
            "reference": dict(n_estimators=450, max_depth=3, learning_rate=0.03, min_child_weight=5,
                              subsample=0.80, colsample_bytree=0.70, reg_alpha=0.2, reg_lambda=8.0),
            "regularized": dict(n_estimators=700, max_depth=3, learning_rate=0.02, min_child_weight=8,
                                subsample=0.80, colsample_bytree=0.70, reg_alpha=0.3, reg_lambda=12.0),
            "shallow": dict(n_estimators=600, max_depth=2, learning_rate=0.03, min_child_weight=3,
                            subsample=0.85, colsample_bytree=0.80, reg_alpha=0.1, reg_lambda=10.0),
            "deeper": dict(n_estimators=350, max_depth=4, learning_rate=0.03, min_child_weight=8,
                           subsample=0.80, colsample_bytree=0.70, reg_alpha=0.2, reg_lambda=10.0),
        }
        return XGBRegressor(
            **profiles[xgb_profile],
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=-1,
        )
    if name == "extratrees":
        return ExtraTreesRegressor(
            n_estimators=600,
            max_features=0.70,
            min_samples_leaf=2,
            max_depth=None,
            bootstrap=False,
            random_state=seed,
            n_jobs=-1,
        )
    if name == "randomforest":
        return RandomForestRegressor(
            n_estimators=600,
            max_features=0.70,
            min_samples_leaf=2,
            max_depth=None,
            bootstrap=True,
            random_state=seed,
            n_jobs=-1,
        )
    if name == "lightgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_estimators=450,
            learning_rate=0.03,
            num_leaves=15,
            max_depth=4,
            min_child_samples=25,
            subsample=0.80,
            colsample_bytree=0.70,
            reg_alpha=0.2,
            reg_lambda=4.0,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
    raise ValueError(name)


def fit_model(
    model,
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray | None,
):
    """Fit estimators while routing weights through sklearn pipelines correctly."""
    if sample_weight is None:
        return model.fit(x_train, y_train)
    if model_name == "ridge":
        return model.fit(x_train, y_train, ridge__sample_weight=sample_weight)
    return model.fit(x_train, y_train, sample_weight=sample_weight)


def build_sample_weights(
    glacier_groups: np.ndarray,
    uncertainty: np.ndarray,
    target: np.ndarray,
    train_mask: np.ndarray,
    mode: str,
) -> np.ndarray | None:
    """Build fold-local weights without using held-out observations."""
    if mode == "none":
        return None

    weights = np.ones(len(glacier_groups), dtype=np.float64)
    if mode in {"glacier", "combined", "glacier_extreme"}:
        groups, counts = np.unique(glacier_groups[train_mask], return_counts=True)
        reference = float(np.mean(counts))
        group_weights = {
            group: float(np.clip(np.sqrt(reference / count), 0.5, 2.0))
            for group, count in zip(groups, counts)
        }
        weights *= np.asarray(
            [group_weights.get(group, 1.0) for group in glacier_groups], dtype=np.float64
        )

    if mode in {"uncertainty", "combined"}:
        valid_train = train_mask & np.isfinite(uncertainty) & (uncertainty > 0)
        if valid_train.any():
            reference = float(np.median(uncertainty[valid_train]))
            valid = np.isfinite(uncertainty) & (uncertainty > 0)
            weights[valid] *= np.clip((reference / uncertainty[valid]) ** 2, 0.5, 2.0)

    if mode in {"extreme", "glacier_extreme"}:
        train_target = target[train_mask]
        median = float(np.median(train_target))
        q25, q75 = np.quantile(train_target, [0.25, 0.75])
        robust_scale = max(float((q75 - q25) / 1.349), 1e-6)
        standardized = np.abs(target[train_mask] - median) / robust_scale
        # Counter regression-to-the-mean without allowing a few records to dominate.
        weights[train_mask] *= np.clip(1.0 + 0.75 * standardized, 1.0, 3.0)

    weights /= float(np.mean(weights[train_mask]))
    return weights.astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cv",
        choices=["logo", "logo_buffered", "loyo", "loyo_buffered", "loso", "forward"],
        required=True,
    )
    parser.add_argument(
        "--model",
        choices=["dummy", "ridge", "randomforest", "xgboost", "lightgbm", "extratrees"],
        required=True,
    )
    parser.add_argument(
        "--xgb-profile",
        choices=["reference", "regularized", "shallow", "deeper"],
        default="reference",
    )
    parser.add_argument("--feature-set", choices=sorted(FEATURE_SETS), default="compact")
    parser.add_argument("--static-set", choices=["all", "terrain", "downscaled"], default="all")
    parser.add_argument("--no-hypsometry", action="store_true")
    parser.add_argument(
        "--representation", choices=["monthly", "monthly_phys", "seasonal"], default="monthly"
    )
    parser.add_argument(
        "--sample-weight",
        choices=["none", "glacier", "uncertainty", "combined", "extreme", "glacier_extreme"],
        default="none",
        help="Fold-local training weights; held-out labels are never used.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--forward-start-year", type=int, default=1980)
    parser.add_argument("--forward-min-train-samples", type=int, default=100)
    parser.add_argument("--spatial-buffer-km", type=float, default=50.0)
    parser.add_argument(
        "--monotonic-physics",
        action="store_true",
        help="Constrain ablation-season temperature/radiation and accumulation-season snowfall signs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = np.load(PHYS_V2_SEQUENCES_NPZ, allow_pickle=True)
    stored = [str(name) for name in data["dynamic_features"]]
    selected = FEATURE_SETS[args.feature_set]
    indices = [stored.index(name) for name in selected]

    selected_dynamic = data["X_dyn"][:, :, indices]
    n_extra_dynamic = 0
    if args.representation in {"monthly", "monthly_phys"}:
        dynamic = calendar_flatten(selected_dynamic, data["month_ids"])
        if args.representation == "monthly_phys":
            physical = physical_seasonal_indices(
                selected_dynamic, data["month_ids"], selected
            )
            dynamic = np.column_stack([dynamic, physical])
            n_extra_dynamic = physical.shape[1]
    else:
        dynamic = seasonal_aggregate(selected_dynamic, data["month_ids"], selected)
    stored_static = [str(name) for name in data["static_features"]]
    base_static_names = [
        "slope_deg", "aspect_sin", "aspect_cos", "zmin_m", "zmax_m",
        "zmean_m", "zmed_m", "log1p_area_km2", "log1p_lmax_m", "cenlat", "cenlon",
        "clim_annual_t2m", "clim_winter_tp", "clim_summer_t2m",
        "clim_summer_ssrd", "clim_t2m_amplitude",
    ]
    if args.static_set == "terrain":
        static_names = [
            "slope_deg", "aspect_sin", "aspect_cos", "zmin_m", "zmax_m",
            "zmean_m", "zmed_m", "log1p_area_km2", "log1p_lmax_m", "cenlat", "cenlon",
        ]
        static_idx = [stored_static.index(name) for name in static_names]
        static = data["X_sta"][:, static_idx]
    elif args.static_set == "all":
        static_idx = [stored_static.index(name) for name in base_static_names]
        static = data["X_sta"][:, static_idx]
    else:
        static = data["X_sta"]
    components = [dynamic, static]
    if not args.no_hypsometry:
        components.append(hypsometry_quantiles(data["X_hyp"], data["hypsometry_band_centers_m"]))
    x = np.column_stack(components).astype(np.float32)
    y = data["y_annual"].astype(np.float32)
    annual_uncertainty = data["annual_unc"].astype(np.float32)
    glacier_ids = data["glacier_ids"]
    rgi_ids = data["rgi_ids"]
    years = data["years"]
    if args.cv in {"logo", "logo_buffered"}:
        groups = rgi_ids
    elif args.cv == "loso":
        groups = data["o2regions"]
    else:
        groups = years

    predictions = np.full(len(y), np.nan, dtype=np.float32)
    seen_in_prior_training = np.zeros(len(y), dtype=bool)
    folds = np.unique(groups)
    if args.cv == "forward":
        folds = folds[folds >= args.forward_start_year]
    for fold_index, held_out in enumerate(folds):
        test = groups == held_out
        if args.cv == "forward":
            train = years < int(held_out)
            if int(train.sum()) < args.forward_min_train_samples:
                print(f"Skipping {held_out}: only {int(train.sum())} prior samples")
                continue
        elif args.cv == "logo_buffered":
            held_lat = float(np.mean(data["X_sta"][test, stored_static.index("cenlat")]))
            held_lon = float(np.mean(data["X_sta"][test, stored_static.index("cenlon")]))
            distance = haversine_km(
                data["X_sta"][:, stored_static.index("cenlat")],
                data["X_sta"][:, stored_static.index("cenlon")],
                held_lat,
                held_lon,
            )
            train = (~test) & (distance > args.spatial_buffer_km)
        elif args.cv == "loyo_buffered":
            train = np.abs(years - int(held_out)) > 1
        else:
            train = ~test
        medians = np.nanmedian(x[train], axis=0)
        x_train = np.where(np.isnan(x[train]), medians, x[train])
        x_test = np.where(np.isnan(x[test]), medians, x[test])
        model = make_model(args.model, args.seed + fold_index, args.xgb_profile)
        if args.monotonic_physics:
            if args.model != "xgboost" or args.representation not in {"monthly", "monthly_phys"}:
                raise ValueError("Physical monotonic constraints require monthly XGBoost.")
            constraints = monthly_physical_monotonic_constraints(
                selected,
                n_static=static.shape[1],
                n_hypsometry=0 if args.no_hypsometry else components[-1].shape[1],
                n_extra_dynamic=n_extra_dynamic,
            )
            if len(constraints) != x.shape[1]:
                raise RuntimeError("Monotonic constraint count does not match the feature matrix.")
            model.set_params(monotone_constraints=constraints)
        weights = build_sample_weights(
            rgi_ids, annual_uncertainty, y, train, args.sample_weight
        )
        fit_model(
            model,
            args.model,
            x_train,
            y[train],
            None if weights is None else weights[train],
        )
        predictions[test] = model.predict(x_test)
        if args.cv == "forward":
            seen_in_prior_training[test] = np.isin(rgi_ids[test], np.unique(rgi_ids[train]))
        print(f"[{fold_index + 1:02d}/{len(folds)}] held_out={held_out} n={test.sum()}")

    hyp_tag = "nohyp" if args.no_hypsometry else "hyp"
    weight_tag = "" if args.sample_weight == "none" else f"_w-{args.sample_weight}"
    seed_tag = "" if args.seed == 42 else f"_seed{args.seed}"
    profile_tag = "" if args.model != "xgboost" or args.xgb_profile == "reference" else f"_p-{args.xgb_profile}"
    monotonic_tag = "_mono-physics" if args.monotonic_physics else ""
    tag = (
        f"{args.model}_v2_{args.feature_set}_{args.representation}_"
        f"{args.static_set}_{hyp_tag}{weight_tag}{profile_tag}{monotonic_tag}{seed_tag}"
    )
    result_dir = os.path.join(PHYS_V2_RESULT_DIR, tag)
    os.makedirs(result_dir, exist_ok=True)
    valid_prediction = np.isfinite(predictions)
    frame = pd.DataFrame(
        {
            "glacier_id": glacier_ids[valid_prediction],
            "rgi_id": rgi_ids[valid_prediction],
            "year": years[valid_prediction],
            "obs_annual": y[valid_prediction],
            "pred_annual": predictions[valid_prediction],
            "seen_in_prior_training": seen_in_prior_training[valid_prediction],
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
    if args.cv == "forward":
        seen = frame["seen_in_prior_training"].to_numpy(dtype=bool)
        seen_metrics = metrics(
            frame.loc[seen, "obs_annual"].to_numpy(),
            frame.loc[seen, "pred_annual"].to_numpy(),
        )
        result.update({f"seen_glacier_{key}": value for key, value in seen_metrics.items()})
    result["macro_glacier_rmse_mm"] = float(
        frame.groupby("rgi_id").apply(
            lambda part: np.sqrt(np.mean((part.pred_annual - part.obs_annual) ** 2)),
            include_groups=False,
        ).mean() * 1000.0
    )
    result["macro_year_rmse_mm"] = float(
        frame.groupby("year").apply(
            lambda part: np.sqrt(np.mean((part.pred_annual - part.obs_annual) ** 2)),
            include_groups=False,
        ).mean() * 1000.0
    )
    summary = pd.DataFrame([result])
    summary.to_csv(os.path.join(result_dir, f"{tag}_{args.cv}_summary.csv"), index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
