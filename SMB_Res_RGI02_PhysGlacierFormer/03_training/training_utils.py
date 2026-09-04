"""Shared helpers for GlacierFormer validation training."""
from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class FoldResult:
    predictions: np.ndarray
    observations: np.ndarray
    ids: np.ndarray
    years: np.ndarray
    stopped_epoch: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_regression(obs: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    r, _ = pearsonr(obs, pred)
    return {
        "r2": float(r2_score(obs, pred)),
        "pearson_r": float(r),
        "rmse_mm": float(np.sqrt(mean_squared_error(obs, pred)) * 1000.0),
        "bias_mm": float(np.mean(pred - obs) * 1000.0),
    }


def make_random_validation_mask(
    train_mask: np.ndarray,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    train_indices = np.where(train_mask)[0]
    if len(train_indices) < 2:
        raise ValueError("Not enough training samples to create validation split.")

    rng = np.random.default_rng(seed)
    val_size = max(1, int(len(train_indices) * val_fraction))
    val_indices = rng.choice(train_indices, size=val_size, replace=False)

    fit_mask = train_mask.copy()
    fit_mask[val_indices] = False

    val_mask = np.zeros_like(train_mask, dtype=bool)
    val_mask[val_indices] = True
    return fit_mask, val_mask


def train_one_fold(
    model_cls,
    params: dict,
    x_dyn: np.ndarray,
    x_sta: np.ndarray,
    y: np.ndarray,
    fit_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    device: str,
    init_state_dict: dict | None = None,
) -> tuple[np.ndarray, int]:
    model = model_cls(
        n_dynamic_features=params["n_dynamic_features"],
        n_static_features=params["n_static_features"],
        d_model=params["d_model"],
        n_heads=params["n_heads"],
        n_encoder_layers=params["n_encoder_layers"],
        ff_dim=params["ff_dim"],
        dropout=params["dropout"],
    ).to(device)
    if init_state_dict is not None:
        model.load_state_dict(init_state_dict, strict=False)

    x_dyn_fit = torch.tensor(x_dyn[fit_mask], dtype=torch.float32).to(device)
    x_sta_fit = torch.tensor(x_sta[fit_mask], dtype=torch.float32).to(device)
    y_fit = torch.tensor(y[fit_mask], dtype=torch.float32).to(device)

    x_dyn_val = torch.tensor(x_dyn[val_mask], dtype=torch.float32).to(device)
    x_sta_val = torch.tensor(x_sta[val_mask], dtype=torch.float32).to(device)
    y_val = torch.tensor(y[val_mask], dtype=torch.float32).to(device)

    x_dyn_test = torch.tensor(x_dyn[test_mask], dtype=torch.float32).to(device)
    x_sta_test = torch.tensor(x_sta[test_mask], dtype=torch.float32).to(device)

    loader = DataLoader(
        TensorDataset(x_dyn_fit, x_sta_fit, y_fit),
        batch_size=params["batch_size"],
        shuffle=True,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"],
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=params["epochs"],
        eta_min=params["lr"] * 0.01,
    )
    criterion = nn.MSELoss()

    best_loss = float("inf")
    best_state = None
    patience_count = 0
    stopped_epoch = params["epochs"]

    for epoch in range(params["epochs"]):
        model.train()
        for batch_dyn, batch_sta, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_dyn, batch_sta), batch_y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        if epoch < params["min_epochs"]:
            continue

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(x_dyn_val, x_sta_val), y_val).item()

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= params["early_stop_patience"]:
                stopped_epoch = epoch + 1
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(x_dyn_test, x_sta_test).detach().cpu().numpy()
    return pred, stopped_epoch


def pretrain_on_weak_labels(
    model_cls,
    params: dict,
    x_dyn: np.ndarray,
    x_sta: np.ndarray,
    weak_glacier_index: np.ndarray,
    weak_target_by_glacier: np.ndarray,
    weak_weight_by_glacier: np.ndarray | None,
    weak_alpha: float,
    epochs: int,
    device: str,
) -> dict:
    """Pretrain a model on glacier-level weak labels."""
    model = model_cls(
        n_dynamic_features=params["n_dynamic_features"],
        n_static_features=params["n_static_features"],
        d_model=params["d_model"],
        n_heads=params["n_heads"],
        n_encoder_layers=params["n_encoder_layers"],
        ff_dim=params["ff_dim"],
        dropout=params["dropout"],
    ).to(device)

    x_dyn_t = torch.tensor(x_dyn, dtype=torch.float32, device=device)
    x_sta_t = torch.tensor(x_sta, dtype=torch.float32, device=device)
    weak_target = torch.tensor(weak_target_by_glacier, dtype=torch.float32, device=device)
    weak_weight = torch.tensor(
        np.ones_like(weak_target_by_glacier) if weak_weight_by_glacier is None else weak_weight_by_glacier,
        dtype=torch.float32,
        device=device,
    )

    indices_by_glacier = {
        int(gid): np.where(weak_glacier_index == gid)[0]
        for gid in np.unique(weak_glacier_index)
    }
    glacier_ids = np.array(sorted(indices_by_glacier), dtype=np.int64)
    glacier_batch_size = max(1, params["batch_size"])

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"],
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=params["lr"] * 0.01,
    )

    rng = np.random.default_rng(42)
    for _epoch in range(max(1, epochs)):
        model.train()
        shuffled = rng.permutation(glacier_ids)
        for start in range(0, len(shuffled), glacier_batch_size):
            batch_gids = shuffled[start : start + glacier_batch_size]
            batch_indices = np.concatenate([indices_by_glacier[int(gid)] for gid in batch_gids])
            batch_indices_t = torch.tensor(batch_indices, dtype=torch.long, device=device)

            optimizer.zero_grad()
            pred = model(x_dyn_t[batch_indices_t], x_sta_t[batch_indices_t])
            loss_terms = []
            cursor = 0
            for gid in batch_gids:
                n_samples = len(indices_by_glacier[int(gid)])
                weak_pred = pred[cursor : cursor + n_samples].mean()
                gid_t = torch.tensor(int(gid), dtype=torch.long, device=device)
                weak_loss = (weak_pred - weak_target[gid_t]) ** 2 * weak_weight[gid_t]
                loss_terms.append(weak_loss)
                cursor += n_samples
            loss = torch.stack(loss_terms).mean() * weak_alpha
            loss.backward()
            optimizer.step()
        scheduler.step()

    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def pretrain_on_multiperiod_constraints(
    model_cls,
    params: dict,
    x_dyn: np.ndarray,
    x_sta: np.ndarray,
    constraint_index: np.ndarray,
    target_by_constraint: np.ndarray,
    weight_by_constraint: np.ndarray | None,
    weak_alpha: float,
    epochs: int,
    device: str,
) -> dict:
    """Pretrain using Hugonnet constraints over multiple time windows."""
    model = model_cls(
        n_dynamic_features=params["n_dynamic_features"],
        n_static_features=params["n_static_features"],
        d_model=params["d_model"],
        n_heads=params["n_heads"],
        n_encoder_layers=params["n_encoder_layers"],
        ff_dim=params["ff_dim"],
        dropout=params["dropout"],
    ).to(device)

    x_dyn_t = torch.tensor(x_dyn, dtype=torch.float32, device=device)
    x_sta_t = torch.tensor(x_sta, dtype=torch.float32, device=device)
    target_t = torch.tensor(target_by_constraint, dtype=torch.float32, device=device)
    weight_t = torch.tensor(
        np.ones_like(target_by_constraint) if weight_by_constraint is None else weight_by_constraint,
        dtype=torch.float32,
        device=device,
    )
    indices_by_constraint = {
        int(cid): np.where(constraint_index == cid)[0]
        for cid in np.unique(constraint_index)
    }
    constraint_ids = np.array(sorted(indices_by_constraint), dtype=np.int64)
    constraint_batch_size = max(1, params["batch_size"])

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"],
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=params["lr"] * 0.01,
    )
    rng = np.random.default_rng(42)

    for _epoch in range(max(1, epochs)):
        model.train()
        shuffled = rng.permutation(constraint_ids)
        for start in range(0, len(shuffled), constraint_batch_size):
            batch_cids = shuffled[start : start + constraint_batch_size]
            batch_indices = np.concatenate([indices_by_constraint[int(cid)] for cid in batch_cids])
            batch_indices_t = torch.tensor(batch_indices, dtype=torch.long, device=device)

            optimizer.zero_grad()
            pred = model(x_dyn_t[batch_indices_t], x_sta_t[batch_indices_t])
            losses = []
            cursor = 0
            for cid in batch_cids:
                n_samples = len(indices_by_constraint[int(cid)])
                pred_mean = pred[cursor : cursor + n_samples].mean()
                cid_t = torch.tensor(int(cid), dtype=torch.long, device=device)
                losses.append((pred_mean - target_t[cid_t]) ** 2 * weight_t[cid_t])
                cursor += n_samples
            loss = torch.stack(losses).mean() * weak_alpha
            loss.backward()
            optimizer.step()
        scheduler.step()

    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def save_validation_outputs(
    result_dir: str,
    tag: str,
    cv_name: str,
    observations: list[float],
    predictions: list[float],
    glacier_ids: list,
    years: list,
) -> dict[str, float]:
    os.makedirs(result_dir, exist_ok=True)

    metrics = evaluate_regression(np.array(observations), np.array(predictions))
    pred_path = os.path.join(result_dir, f"{tag}_{cv_name.lower()}_predictions.csv")
    summary_path = os.path.join(result_dir, f"{tag}_{cv_name.lower()}_summary.csv")

    pd.DataFrame(
        {
            "glacier_id": glacier_ids,
            "year": years,
            "obs": observations,
            "pred": predictions,
        }
    ).to_csv(pred_path, index=False)

    pd.DataFrame(
        [
            {
                "model": tag,
                "cv": cv_name,
                **metrics,
            }
        ]
    ).to_csv(summary_path, index=False)
    return metrics
