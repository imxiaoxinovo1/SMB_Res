# 02_models/glacioformer/train_loyo.py
"""
GlacioFormer LOYO (Leave-One-Year-Out) cross-validation.
Supports --variant full (all 15 vars) or selected (future use).
"""
import sys, os
# 将项目根目录（config.py 所在）和 02_models/（glacioformer 包所在）加入路径
_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_script_dir, '..', '..'))   # project root
sys.path.insert(0, os.path.join(_script_dir, '..'))          # 02_models/

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score, mean_squared_error
from config import (SEQUENCES_NPZ, RESULT_DIR, GLACIOFORMER_PARAMS,
                    TRAIN_YEAR_MIN, TRAIN_YEAR_MAX)
from glacioformer.model import GlacioFormer

parser = argparse.ArgumentParser()
parser.add_argument('--variant', default='full', choices=['full', 'selected'])
args = parser.parse_args()

os.makedirs(RESULT_DIR, exist_ok=True)
tag = f'glacioformer_{args.variant}'

# 日志文件（直接写入，避免 stdout 缓冲问题）
log_path = os.path.join(RESULT_DIR, f'{tag}_loyo_train.log')
log = open(log_path, 'w', encoding='utf-8', buffering=1)

def log_print(msg):
    print(msg)
    log.write(msg + '\n')
    log.flush()

log_print(f"=== GlacioFormer LOYO -- variant={args.variant} ===")

data = np.load(SEQUENCES_NPZ, allow_pickle=True)
X_dyn = data['X_dyn']    # (N, 12, 15) normalized
X_sta = data['X_sta']    # (N, 10) normalized
y     = data['y']
years = data['years']

train_mask = (years >= TRAIN_YEAR_MIN) & (years <= TRAIN_YEAR_MAX) & (~np.isnan(y))
X_dyn = X_dyn[train_mask]
X_sta = X_sta[train_mask]
y     = y[train_mask]
years_tr = years[train_mask]

log_print(f"Training samples: {len(y)}")

P = GLACIOFORMER_PARAMS
device = 'cuda' if torch.cuda.is_available() else 'cpu'
log_print(f"Device: {device}")

fold_years = sorted(set(years_tr.tolist()))
all_preds, all_obs = [], []

for yi, yr in enumerate(fold_years):
    tr = years_tr != yr
    te = years_tr == yr
    if te.sum() == 0:
        continue

    Xd_tr = torch.tensor(X_dyn[tr], dtype=torch.float32).to(device)
    Xs_tr = torch.tensor(X_sta[tr], dtype=torch.float32).to(device)
    yt_tr = torch.tensor(y[tr],     dtype=torch.float32).to(device)
    Xd_te = torch.tensor(X_dyn[te], dtype=torch.float32).to(device)
    Xs_te = torch.tensor(X_sta[te], dtype=torch.float32).to(device)

    model = GlacioFormer(
        n_dynamic_features=P['n_dynamic_features'],
        n_static_features=P['n_static_features'],
        d_model=P['d_model'], n_heads=P['n_heads'],
        n_encoder_layers=P['n_encoder_layers'],
        ff_dim=P['ff_dim'], dropout=P['dropout'],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=P['lr'],
                                  weight_decay=P['weight_decay'])
    criterion = nn.MSELoss()

    loader = DataLoader(TensorDataset(Xd_tr, Xs_tr, yt_tr),
                        batch_size=P['batch_size'], shuffle=True)
    best_loss, patience_cnt, best_state = float('inf'), 0, None

    for epoch in range(P['epochs']):
        model.train()
        for xd, xs, yt in loader:
            optimizer.zero_grad()
            criterion(model(xd, xs), yt).backward()
            optimizer.step()

        if epoch < P['min_epochs']:
            continue
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(Xd_tr, Xs_tr), yt_tr).item()
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= P['early_stop_patience']:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p = model(Xd_te, Xs_te).cpu().numpy()
    all_preds.extend(p.tolist())
    all_obs.extend(y[te].tolist())

    r2_so_far = r2_score(all_obs, all_preds)
    log_print(f"  Fold {yi+1}/{len(fold_years)} (year={int(yr)}) -- R2={r2_so_far:.3f}  stopped@ep={epoch+1}")

r2 = r2_score(all_obs, all_preds)
rmse = np.sqrt(mean_squared_error(all_obs, all_preds))
bias = np.mean(np.array(all_preds) - np.array(all_obs))
log_print(f"\nLOYO R2={r2:.4f}  RMSE={rmse*1000:.1f}mm  Bias={bias*1000:.1f}mm")

pd.DataFrame({'obs': all_obs, 'pred': all_preds}).to_csv(
    os.path.join(RESULT_DIR, f'{tag}_loyo_predictions.csv'), index=False)
pd.DataFrame([{'model': tag, 'cv': 'LOYO',
               'r2': r2, 'rmse_mm': rmse*1000, 'bias_mm': bias*1000}]).to_csv(
    os.path.join(RESULT_DIR, f'{tag}_loyo_summary.csv'), index=False)
log_print(f"Saved to {RESULT_DIR}")
log.close()
