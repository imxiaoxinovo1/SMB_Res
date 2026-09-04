"""Train one final hypsometry GlacierFormer model for reconstruction."""
from __future__ import annotations

import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "02_models"))

from config import (  # noqa: E402
    FINAL_MODEL_DIR,
    GLACIERFORMER_HYPSOMETRY_PARAMS,
    HYPSOMETRY_FINAL_WEIGHTS,
    SEQUENCES_HYPSOMETRY_QC_NPZ,
    TRAIN_YEAR_MAX,
    TRAIN_YEAR_MIN,
)
from glacierformer_hypsometry import GlacierFormerHypsometry  # noqa: E402


def main() -> None:
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
    log_path = os.path.join(FINAL_MODEL_DIR, "glacierformer_hypsometry_qc_final_train.log")

    with open(log_path, "w", encoding="utf-8", buffering=1) as log:
        def log_print(message: str) -> None:
            print(message)
            log.write(message + "\n")

        log_print("=== Train Final Hypsometry GlacierFormer ===")
        data = np.load(SEQUENCES_HYPSOMETRY_QC_NPZ, allow_pickle=True)
        x_dyn = data["X_dyn"]
        x_sta = data["X_sta"]
        x_hyp = data["X_hyp"]
        y = data["y"]
        years = data["years"]

        mask = (years >= TRAIN_YEAR_MIN) & (years <= TRAIN_YEAR_MAX) & (~np.isnan(y))
        x_dyn, x_sta, x_hyp, y = x_dyn[mask], x_sta[mask], x_hyp[mask], y[mask]
        log_print(f"Training samples: {len(y):,}")

        params = GLACIERFORMER_HYPSOMETRY_PARAMS
        rng = np.random.default_rng(42)
        all_idx = np.arange(len(y))
        val_size = max(1, int(len(y) * params["val_fraction"]))
        val_idx = rng.choice(all_idx, size=val_size, replace=False)
        fit_mask = np.ones(len(y), dtype=bool)
        fit_mask[val_idx] = False
        val_mask = ~fit_mask

        device = "cuda" if torch.cuda.is_available() else "cpu"
        log_print(f"Device: {device}")
        model = GlacierFormerHypsometry(
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
        patience = 0
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
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= params["early_stop_patience"]:
                    stopped_epoch = epoch + 1
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        torch.save(model.state_dict(), HYPSOMETRY_FINAL_WEIGHTS)
        log_print(f"Best validation loss: {best_loss:.6f}")
        log_print(f"Stopped epoch: {stopped_epoch}")
        log_print(f"Saved final weights -> {HYPSOMETRY_FINAL_WEIGHTS}")


if __name__ == "__main__":
    main()
