"""Leakage-safe training utilities for PhysGlacierFormer v2."""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class FoldScaler:
    dyn_mean: np.ndarray
    dyn_std: np.ndarray
    sta_median: np.ndarray
    sta_mean: np.ndarray
    sta_std: np.ndarray


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_group_validation_masks(
    outer_train_mask: np.ndarray,
    groups: np.ndarray,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    eligible_groups = np.unique(groups[outer_train_mask])
    if len(eligible_groups) < 2:
        raise ValueError("At least two training groups are required for inner validation.")
    rng = np.random.default_rng(seed)
    shuffled = eligible_groups.copy()
    rng.shuffle(shuffled)
    n_val_groups = max(1, int(np.ceil(len(shuffled) * val_fraction)))
    val_groups = set(shuffled[:n_val_groups].tolist())
    val_mask = outer_train_mask & np.isin(groups, list(val_groups))
    fit_mask = outer_train_mask & ~val_mask
    if not fit_mask.any() or not val_mask.any():
        raise RuntimeError("Invalid grouped fit/validation split.")
    return fit_mask, val_mask


def fit_fold_scaler(x_dyn: np.ndarray, x_sta: np.ndarray, fit_mask: np.ndarray) -> FoldScaler:
    dyn_mean = np.nanmean(x_dyn[fit_mask], axis=(0, 1), keepdims=True)
    dyn_std = np.nanstd(x_dyn[fit_mask], axis=(0, 1), keepdims=True)
    dyn_std = np.where(dyn_std < 1e-6, 1.0, dyn_std)

    sta_median = np.nanmedian(x_sta[fit_mask], axis=0, keepdims=True)
    fit_static = np.where(np.isnan(x_sta[fit_mask]), sta_median, x_sta[fit_mask])
    sta_mean = np.mean(fit_static, axis=0, keepdims=True)
    sta_std = np.std(fit_static, axis=0, keepdims=True)
    sta_std = np.where(sta_std < 1e-6, 1.0, sta_std)
    return FoldScaler(dyn_mean, dyn_std, sta_median, sta_mean, sta_std)


def apply_fold_scaler(
    x_dyn: np.ndarray,
    x_sta: np.ndarray,
    scaler: FoldScaler,
) -> tuple[np.ndarray, np.ndarray]:
    dyn = (x_dyn - scaler.dyn_mean) / scaler.dyn_std
    sta = np.where(np.isnan(x_sta), scaler.sta_median, x_sta)
    sta = (sta - scaler.sta_mean) / scaler.sta_std
    if not np.isfinite(dyn).all() or not np.isfinite(sta).all():
        raise RuntimeError("Non-finite values remain after fold-specific preprocessing.")
    return dyn.astype(np.float32), sta.astype(np.float32)


def build_sample_weights(
    uncertainty: np.ndarray,
    glacier_groups: np.ndarray,
    fit_mask: np.ndarray,
    use_uncertainty: bool,
    balance_glaciers: bool,
) -> np.ndarray:
    weights = np.ones(len(uncertainty), dtype=np.float32)
    if use_uncertainty:
        reported = uncertainty[fit_mask & np.isfinite(uncertainty) & (uncertainty > 0)]
        if len(reported):
            reference = float(np.median(reported))
            valid = np.isfinite(uncertainty) & (uncertainty > 0)
            weights[valid] *= np.clip((reference / uncertainty[valid]) ** 2, 0.5, 2.0)
    if balance_glaciers:
        unique, counts = np.unique(glacier_groups[fit_mask], return_counts=True)
        reference = float(np.mean(counts))
        group_weight = {
            group: float(np.clip(np.sqrt(reference / count), 0.5, 2.0))
            for group, count in zip(unique, counts)
        }
        weights *= np.asarray([group_weight.get(group, 1.0) for group in glacier_groups])
    weights /= float(np.mean(weights[fit_mask]))
    return weights


def elementwise_loss(pred: torch.Tensor, target: torch.Tensor, loss_name: str, beta: float) -> torch.Tensor:
    if loss_name == "mse":
        return (pred - target) ** 2
    if loss_name == "huber":
        return F.smooth_l1_loss(pred, target, reduction="none", beta=beta)
    raise ValueError(f"Unsupported loss: {loss_name}")


def masked_weighted_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    loss_name: str,
    beta: float,
) -> torch.Tensor:
    mask = torch.isfinite(target)
    if not bool(mask.any()):
        return pred.sum() * 0.0
    values = elementwise_loss(pred[mask], target[mask], loss_name, beta)
    selected_weight = weight[mask]
    return torch.sum(values * selected_weight) / torch.sum(selected_weight).clamp_min(1e-8)


def train_one_fold(
    model_cls,
    params: dict,
    arrays: dict[str, np.ndarray],
    fit_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    device: str,
    seed: int,
    seasonal_loss_weight: float,
    loss_name: str,
    use_uncertainty_weights: bool,
    balance_glaciers: bool,
) -> tuple[dict[str, np.ndarray], int, FoldScaler]:
    seed_everything(seed)
    scaler = fit_fold_scaler(arrays["X_dyn"], arrays["X_sta"], fit_mask)
    x_dyn, x_sta = apply_fold_scaler(arrays["X_dyn"], arrays["X_sta"], scaler)

    annual_weight = build_sample_weights(
        arrays["annual_unc"], arrays["rgi_ids"], fit_mask,
        use_uncertainty_weights, balance_glaciers,
    )
    winter_weight = build_sample_weights(
        arrays["winter_unc"], arrays["rgi_ids"], fit_mask,
        use_uncertainty_weights, balance_glaciers,
    )
    summer_weight = build_sample_weights(
        arrays["summer_unc"], arrays["rgi_ids"], fit_mask,
        use_uncertainty_weights, balance_glaciers,
    )

    model = model_cls(**params).to(device)
    tensor = lambda value, dtype=torch.float32: torch.as_tensor(value, dtype=dtype, device=device)
    all_tensors = {
        "dyn": tensor(x_dyn),
        "sta": tensor(x_sta),
        "hyp": tensor(arrays["X_hyp"]),
        "months": tensor(arrays["month_ids"], torch.long),
        "annual": tensor(arrays["y_annual"]),
        "winter": tensor(arrays["y_winter"]),
        "summer": tensor(arrays["y_summer"]),
        "aw": tensor(annual_weight),
        "ww": tensor(winter_weight),
        "sw": tensor(summer_weight),
    }
    fit_idx = np.where(fit_mask)[0]
    loader = DataLoader(
        TensorDataset(*[all_tensors[key][fit_idx] for key in [
            "dyn", "sta", "hyp", "months", "annual", "winter", "summer", "aw", "ww", "sw"
        ]]),
        batch_size=params["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=params["lr"] * 0.02
    )

    best_loss = float("inf")
    best_state = None
    patience = 0
    stopped_epoch = params["epochs"]
    val_idx = np.where(val_mask)[0]

    for epoch in range(1, params["epochs"] + 1):
        model.train()
        for batch in loader:
            xd, xs, xh, months, ya, yw, ys, aw, ww, sw = batch
            optimizer.zero_grad(set_to_none=True)
            pa, pw, ps = model(xd, xs, xh, months)
            annual_loss = masked_weighted_loss(pa, ya, aw, loss_name, params["huber_beta"])
            winter_loss = masked_weighted_loss(pw, yw, ww, loss_name, params["huber_beta"])
            summer_loss = masked_weighted_loss(ps, ys, sw, loss_name, params["huber_beta"])
            loss = annual_loss + seasonal_loss_weight * 0.5 * (winter_loss + summer_loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_annual, _, _ = model(
                all_tensors["dyn"][val_idx], all_tensors["sta"][val_idx],
                all_tensors["hyp"][val_idx], all_tensors["months"][val_idx],
            )
            val_rmse = torch.sqrt(torch.mean((val_annual - all_tensors["annual"][val_idx]) ** 2)).item()
        scheduler.step(val_rmse)

        if epoch >= params["min_epochs"]:
            if val_rmse < best_loss - 1e-5:
                best_loss = val_rmse
                best_state = copy.deepcopy(model.state_dict())
                patience = 0
            else:
                patience += 1
                if patience >= params["early_stop_patience"]:
                    stopped_epoch = epoch
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_idx = np.where(test_mask)[0]
    model.eval()
    with torch.no_grad():
        annual, winter, summer = model(
            all_tensors["dyn"][test_idx], all_tensors["sta"][test_idx],
            all_tensors["hyp"][test_idx], all_tensors["months"][test_idx],
        )
    predictions = {
        "annual": annual.cpu().numpy(),
        "winter": winter.cpu().numpy(),
        "summer": summer.cpu().numpy(),
    }
    return predictions, stopped_epoch, scaler
