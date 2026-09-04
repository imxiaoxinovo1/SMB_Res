"""Training helpers for GlacierFormer models with hypsometry input."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset


def evaluate_regression(obs: np.ndarray, pred: np.ndarray) -> dict[str, float]:
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
    rng = np.random.default_rng(seed)
    val_size = max(1, int(len(train_indices) * val_fraction))
    val_indices = rng.choice(train_indices, size=val_size, replace=False)

    fit_mask = train_mask.copy()
    fit_mask[val_indices] = False
    val_mask = np.zeros_like(train_mask, dtype=bool)
    val_mask[val_indices] = True
    return fit_mask, val_mask


def train_one_fold_hypsometry(
    model_cls,
    params: dict,
    x_dyn: np.ndarray,
    x_sta: np.ndarray,
    x_hyp: np.ndarray,
    y: np.ndarray,
    fit_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    device: str,
) -> tuple[np.ndarray, int]:
    model = model_cls(
        n_dynamic_features=params["n_dynamic_features"],
        n_static_features=params["n_static_features"],
        n_hypsometry_features=params["n_hypsometry_features"],
        d_model=params["d_model"],
        n_heads=params["n_heads"],
        n_encoder_layers=params["n_encoder_layers"],
        ff_dim=params["ff_dim"],
        dropout=params["dropout"],
    ).to(device)

    tensors = {
        "xd_fit": torch.tensor(x_dyn[fit_mask], dtype=torch.float32).to(device),
        "xs_fit": torch.tensor(x_sta[fit_mask], dtype=torch.float32).to(device),
        "xh_fit": torch.tensor(x_hyp[fit_mask], dtype=torch.float32).to(device),
        "y_fit": torch.tensor(y[fit_mask], dtype=torch.float32).to(device),
        "xd_val": torch.tensor(x_dyn[val_mask], dtype=torch.float32).to(device),
        "xs_val": torch.tensor(x_sta[val_mask], dtype=torch.float32).to(device),
        "xh_val": torch.tensor(x_hyp[val_mask], dtype=torch.float32).to(device),
        "y_val": torch.tensor(y[val_mask], dtype=torch.float32).to(device),
        "xd_test": torch.tensor(x_dyn[test_mask], dtype=torch.float32).to(device),
        "xs_test": torch.tensor(x_sta[test_mask], dtype=torch.float32).to(device),
        "xh_test": torch.tensor(x_hyp[test_mask], dtype=torch.float32).to(device),
    }

    loader = DataLoader(
        TensorDataset(
            tensors["xd_fit"],
            tensors["xs_fit"],
            tensors["xh_fit"],
            tensors["y_fit"],
        ),
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
        for batch_dyn, batch_sta, batch_hyp, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_dyn, batch_sta, batch_hyp), batch_y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        if epoch < params["min_epochs"]:
            continue

        model.eval()
        with torch.no_grad():
            val_loss = criterion(
                model(tensors["xd_val"], tensors["xs_val"], tensors["xh_val"]),
                tensors["y_val"],
            ).item()

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
        pred = model(
            tensors["xd_test"],
            tensors["xs_test"],
            tensors["xh_test"],
        ).detach().cpu().numpy()
    return pred, stopped_epoch


def save_outputs(
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

    pd.DataFrame(
        {
            "glacier_id": glacier_ids,
            "year": years,
            "obs": observations,
            "pred": predictions,
        }
    ).to_csv(os.path.join(result_dir, f"{tag}_{cv_name.lower()}_predictions.csv"), index=False)

    pd.DataFrame([{"model": tag, "cv": cv_name, **metrics}]).to_csv(
        os.path.join(result_dir, f"{tag}_{cv_name.lower()}_summary.csv"),
        index=False,
    )
    return metrics
