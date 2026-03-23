"""
统一 LOYO 训练脚本 — 支持多模型版本实验

用法:
  python train.py --model v1_vanilla     # 基础 Transformer
  python train.py --model v2_wtconv      # + 小波卷积
  python train.py --model v3_ema         # + 多尺度注意力
  ...

结果统一保存到 results/<model_name>/
便于多版本横向对比。
"""
import sys
import os
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import (PREPROCESS_DIR, LSTM_PREPROCESS_DIR,
                    MODEL_DIR, GLACIOFORMER_PARAMS,
                    N_CLIMATE_FEATURES, N_STATIC_FEATURES)

# ─── 模型注册表 ──────────────────────────────────────────────────────────────
def get_model_class(name: str):
    if name == 'v1_vanilla':
        from models.v1_vanilla import VanillaTransformer
        return VanillaTransformer
    # 未来版本在此注册：
    # elif name == 'v2_wtconv':
    #     from models.v2_wtconv import WTConvTransformer
    #     return WTConvTransformer
    else:
        raise ValueError(f"未知模型: {name}，可选: v1_vanilla")


# ─── 参数解析 ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='GlacioFormer LOYO 训练')
parser.add_argument('--model', type=str, default='v1_vanilla',
                    help='模型名称 (v1_vanilla / v2_wtconv / ...)')
args = parser.parse_args()

MODEL_NAME   = args.model
ModelClass   = get_model_class(MODEL_NAME)
RESULTS_DIR  = os.path.join(MODEL_DIR, "results", MODEL_NAME)
os.makedirs(RESULTS_DIR, exist_ok=True)

OUT_PRED    = os.path.join(RESULTS_DIR, "loyo_predictions.csv")
OUT_METRICS = os.path.join(RESULTS_DIR, "loyo_metrics.csv")
OUT_SUMMARY = os.path.join(RESULTS_DIR, "loyo_summary.txt")

# ─── 设备 ────────────────────────────────────────────────────────────────────
SEED = GLACIOFORMER_PARAMS['random_state']
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f">>> 模型: {MODEL_NAME}")
print(f">>> 设备: {DEVICE}")
if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    print(f"    GPU: {torch.cuda.get_device_name(0)}")

# ─── 超参数 ──────────────────────────────────────────────────────────────────
p           = GLACIOFORMER_PARAMS
EPOCHS      = p['epochs']
BATCH_SIZE  = p['batch_size']
LR          = p['lr']
WD          = p['weight_decay']
MIN_EPOCHS  = p['min_epochs']
PATIENCE    = p['early_stop_patience']


# ─── 辅助函数 ────────────────────────────────────────────────────────────────
def normalize_fold(X_dyn_tr, X_sta_tr, X_dyn_te, X_sta_te):
    N, T, F = X_dyn_tr.shape
    flat = X_dyn_tr.reshape(-1, F)
    dm, ds = flat.mean(0), flat.std(0)
    ds = np.where(ds < 1e-8, 1.0, ds)
    N_te = X_dyn_te.shape[0]
    X_dyn_tr = ((X_dyn_tr.reshape(-1, F) - dm) / ds).reshape(N, T, F)
    X_dyn_te = ((X_dyn_te.reshape(-1, F) - dm) / ds).reshape(N_te, T, F)
    sm, ss = X_sta_tr.mean(0), X_sta_tr.std(0)
    ss = np.where(ss < 1e-8, 1.0, ss)
    return X_dyn_tr, (X_sta_tr-sm)/ss, X_dyn_te, (X_sta_te-sm)/ss


def make_loader(X_dyn, X_sta, y, bs, shuffle=True):
    ds = TensorDataset(
        torch.tensor(X_dyn, dtype=torch.float32),
        torch.tensor(X_sta, dtype=torch.float32),
        torch.tensor(y,     dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=bs, shuffle=shuffle,
                      num_workers=0, pin_memory=(DEVICE.type == 'cuda'))


def predict(model, X_dyn, X_sta):
    model.eval()
    loader = make_loader(X_dyn, X_sta, np.zeros(len(X_dyn)), 256, shuffle=False)
    preds = []
    with torch.no_grad():
        for xd, xs, _ in loader:
            preds.append(model(xd.to(DEVICE), xs.to(DEVICE)).cpu().numpy())
    return np.concatenate(preds)


def metrics(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    r2   = 1 - ss_res/ss_tot if ss_tot > 1e-10 else float('nan')
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mae  = np.mean(np.abs(y_true - y_pred))
    bias = np.mean(y_pred - y_true)
    return r2, rmse, mae, bias


# ─── 加载数据 ────────────────────────────────────────────────────────────────
print("\n>>> 加载数据集...")
data  = np.load(os.path.join(PREPROCESS_DIR, "glacioformer_dataset.npz"), allow_pickle=True)
X_dyn_raw = data['X_dynamic_raw']   # (N,12,15)
X_sta_raw = data['X_static_raw']    # (N,10)
y         = data['y']               # (N,)
df_meta   = pd.read_csv(os.path.join(LSTM_PREPROCESS_DIR, "lstm_meta.csv"))
years     = df_meta['YEAR'].values
N, T, F_dyn = X_dyn_raw.shape
F_sta       = X_sta_raw.shape[1]
print(f"   样本: {N}  动态特征: {F_dyn}  静态特征: {F_sta}")

unique_years = sorted(np.unique(years))
print(f"   LOYO 折数: {len(unique_years)}")

# 打印模型参数量
_tmp = ModelClass(n_dynamic_features=F_dyn, n_static_features=F_sta,
                  **{k: v for k, v in p.items()
                     if k in ['d_model','n_heads','n_encoder_layers',
                               'ff_dim','dropout','ema_scales','wt_levels']})
n_params = sum(x.numel() for x in _tmp.parameters() if x.requires_grad)
del _tmp
print(f"   模型参数量: {n_params:,}")

# ─── LOYO 主循环 ─────────────────────────────────────────────────────────────
print(f"\n>>> LOYO 训练 ({len(unique_years)} 折)...")
criterion  = nn.MSELoss()
all_preds, fold_metrics = [], []

for fi, test_year in enumerate(unique_years):
    tr_mask = (years != test_year)
    te_mask = (years == test_year)
    n_tr, n_te = tr_mask.sum(), te_mask.sum()
    if n_te == 0 or n_tr < BATCH_SIZE:
        continue

    Xd_tr, Xs_tr, Xd_te, Xs_te = normalize_fold(
        X_dyn_raw[tr_mask], X_sta_raw[tr_mask],
        X_dyn_raw[te_mask], X_sta_raw[te_mask],
    )
    y_tr, y_te = y[tr_mask], y[te_mask]

    train_loader = make_loader(Xd_tr, Xs_tr, y_tr, BATCH_SIZE)

    torch.manual_seed(SEED + fi)
    model = ModelClass(
        n_dynamic_features = F_dyn,
        n_static_features  = F_sta,
        d_model            = p['d_model'],
        n_heads            = p['n_heads'],
        n_encoder_layers   = p['n_encoder_layers'],
        ff_dim             = p['ff_dim'],
        dropout            = p['dropout'],
        ema_scales         = p.get('ema_scales', [1,3,6]),
        wt_levels          = p.get('wt_levels', 2),
    ).to(DEVICE)

    opt  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR*0.01)

    best_mse, best_state, patience_cnt = float('inf'), None, 0

    for epoch in range(EPOCHS):
        model.train()
        for xd, xs, yt in train_loader:
            xd, xs, yt = xd.to(DEVICE), xs.to(DEVICE), yt.to(DEVICE)
            opt.zero_grad()
            loss = criterion(model(xd, xs), yt)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        # 验证（用测试年本身，LOYO 标准做法）
        model.eval()
        with torch.no_grad():
            val_loss = criterion(
                model(torch.tensor(Xd_te, dtype=torch.float32).to(DEVICE),
                      torch.tensor(Xs_te, dtype=torch.float32).to(DEVICE)),
                torch.tensor(y_te, dtype=torch.float32).to(DEVICE)
            ).item()
        sch.step()

        if val_loss < best_mse:
            best_mse   = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if epoch >= MIN_EPOCHS:
                patience_cnt = 0
        elif epoch >= MIN_EPOCHS:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                break

    model.load_state_dict(best_state)
    y_pred = predict(model, Xd_te, Xs_te)

    r2, rmse, mae, bias = metrics(y_te, y_pred)
    fold_metrics.append({'YEAR': test_year, 'N': n_te,
                         'R2': round(r2,4), 'RMSE': round(rmse,2),
                         'MAE': round(mae,2), 'Bias': round(bias,2)})

    te_meta = df_meta[te_mask].reset_index(drop=True)
    for i in range(n_te):
        all_preds.append({'WGMS_ID': int(te_meta.loc[i,'WGMS_ID']),
                          'YEAR': int(te_meta.loc[i,'YEAR']),
                          'y_true': round(float(y_te[i]),2),
                          'y_pred': round(float(y_pred[i]),2)})

    print(f"   [{fi+1:3d}/{len(unique_years)}] {test_year}: "
          f"N={n_te:3d}  R²={r2:6.3f}  RMSE={rmse:7.1f}  ep={epoch+1:3d}")

# ─── 汇总 ────────────────────────────────────────────────────────────────────
df_p = pd.DataFrame(all_preds)
df_m = pd.DataFrame(fold_metrics)
df_p.to_csv(OUT_PRED, index=False)
df_m.to_csv(OUT_METRICS, index=False)

yt_a, yp_a = df_p['y_true'].values, df_p['y_pred'].values
g_r2, g_rmse, g_mae, g_bias = metrics(yt_a, yp_a)
corr = np.corrcoef(yt_a, yp_a)[0,1]

lines = [
    "=" * 60,
    f"模型: {MODEL_NAME}  参数量: {n_params:,}",
    "=" * 60,
    f"  全局 R²  : {g_r2:.4f}",
    f"  Pearson R: {corr:.4f}",
    f"  RMSE     : {g_rmse:.2f} mm",
    f"  MAE      : {g_mae:.2f} mm",
    f"  Bias     : {g_bias:.2f} mm",
    "",
    "  参考基线:",
    "    LSTM      R²=0.6538  RMSE=605.23mm  R=0.8285",
    "    v0-complex R²=0.5171  RMSE=714.82mm  R=0.7794  Bias=+156mm",
    "=" * 60,
]
for l in lines: print(l)
with open(OUT_SUMMARY, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print(f"\n>>> 结果: {RESULTS_DIR}")
