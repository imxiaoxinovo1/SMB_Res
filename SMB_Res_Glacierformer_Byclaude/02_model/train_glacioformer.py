"""
GlacioFormer LOYO（Leave-One-Year-Out）交叉验证训练

数据: 直接复用 SMB_Res_LSTM_Byclaude 预处理的 lstm_dataset.npz
      (X_dynamic_raw, X_static_raw, y) 格式完全一致，无需重新提取 ERA5

输出: 02_model/results/
  glacioformer_loyo_predictions.csv  — 逐样本预测 (WGMS_ID, YEAR, y_true, y_pred)
  glacioformer_loyo_metrics.csv      — 逐折指标 (YEAR, R2, RMSE, MAE, Bias, N)
  glacioformer_loyo_summary.txt      — 全折汇总 + 与 LSTM 基线对比
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import (PREPROCESS_DIR, LSTM_PREPROCESS_DIR, RESULTS_DIR,
                    GLACIOFORMER_PARAMS, N_CLIMATE_FEATURES, N_STATIC_FEATURES)
from glacioformer import GlacioFormer
from losses import PhysicsInformedLoss

# ─── 可复现性 ────────────────────────────────────────────────────────────────
SEED = GLACIOFORMER_PARAMS['random_state']
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f">>> 使用设备: {DEVICE}")
if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True   # cuDNN 自动调优，固定输入尺寸下加速
    print(f"    GPU: {torch.cuda.get_device_name(0)}  "
          f"显存: {torch.cuda.get_device_properties(0).total_memory // 1024**2} MB")

# ─── 路径 ────────────────────────────────────────────────────────────────────
NPZ_PATH  = os.path.join(PREPROCESS_DIR, "glacioformer_dataset.npz")  # 含 RGI 地形特征
META_PATH = os.path.join(LSTM_PREPROCESS_DIR, "lstm_meta.csv")         # 沿用 LSTM 元信息

os.makedirs(RESULTS_DIR, exist_ok=True)
OUT_PRED    = os.path.join(RESULTS_DIR, "glacioformer_loyo_predictions.csv")
OUT_METRICS = os.path.join(RESULTS_DIR, "glacioformer_loyo_metrics.csv")
OUT_SUMMARY = os.path.join(RESULTS_DIR, "glacioformer_loyo_summary.txt")

# ─── 超参数 ──────────────────────────────────────────────────────────────────
p = GLACIOFORMER_PARAMS
EPOCHS     = p['epochs']
BATCH_SIZE = p['batch_size']
LR         = p['lr']
WD         = p['weight_decay']

EARLY_STOP_PATIENCE = p['early_stop_patience']
MIN_EPOCHS          = p['min_epochs']
PHYSICS_ALPHA       = p['physics_alpha']
TEMP_VAR_NAME       = p['temp_var_name']


# ───────────────────────────────────────────────────────────────────────────────
# 辅助函数
# ───────────────────────────────────────────────────────────────────────────────

def normalize_fold(X_dyn_train, X_sta_train, X_dyn_test, X_sta_test):
    """Z-score 归一化：仅用训练折统计量，防止数据泄漏。"""
    N, T, F = X_dyn_train.shape

    flat     = X_dyn_train.reshape(-1, F)
    dyn_mean = flat.mean(axis=0)
    dyn_std  = flat.std(axis=0)
    dyn_std  = np.where(dyn_std < 1e-8, 1.0, dyn_std)

    N_te = X_dyn_test.shape[0]
    X_dyn_tr_n = ((X_dyn_train.reshape(-1, F) - dyn_mean) / dyn_std).reshape(N, T, F)
    X_dyn_te_n = ((X_dyn_test.reshape(-1, F)  - dyn_mean) / dyn_std).reshape(N_te, T, F)

    sta_mean = X_sta_train.mean(axis=0)
    sta_std  = X_sta_train.std(axis=0)
    sta_std  = np.where(sta_std < 1e-8, 1.0, sta_std)

    X_sta_tr_n = (X_sta_train - sta_mean) / sta_std
    X_sta_te_n = (X_sta_test  - sta_mean) / sta_std

    return X_dyn_tr_n, X_sta_tr_n, X_dyn_te_n, X_sta_te_n


def make_loader(X_dyn, X_sta, y, batch_size, shuffle=True):
    ds = TensorDataset(
        torch.tensor(X_dyn, dtype=torch.float32),
        torch.tensor(X_sta, dtype=torch.float32),
        torch.tensor(y,     dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=(DEVICE.type == 'cuda'))


def predict(model, X_dyn, X_sta, batch_size=256):
    model.eval()
    loader = make_loader(X_dyn, X_sta, np.zeros(len(X_dyn)), batch_size, shuffle=False)
    preds  = []
    with torch.no_grad():
        for x_dyn, x_sta, _ in loader:
            x_dyn, x_sta = x_dyn.to(DEVICE), x_sta.to(DEVICE)
            preds.append(model(x_dyn, x_sta).cpu().numpy())
    return np.concatenate(preds)


def compute_metrics(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2   = 1 - ss_res / ss_tot if ss_tot > 1e-10 else float('nan')
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae  = np.mean(np.abs(y_true - y_pred))
    bias = np.mean(y_pred - y_true)
    return r2, rmse, mae, bias


# ───────────────────────────────────────────────────────────────────────────────
# 1. 加载数据
# ───────────────────────────────────────────────────────────────────────────────
print(">>> 1. 加载数据集...")
data          = np.load(NPZ_PATH, allow_pickle=True)
X_dynamic_raw = data['X_dynamic_raw']    # (N, 12, 15)
X_static_raw  = data['X_static_raw']     # (N, 5)
y             = data['y']                # (N,)
climate_cols  = list(data.get('climate_cols', []))   # 动态特征列名

df_meta = pd.read_csv(META_PATH)
years   = df_meta['YEAR'].values

N, T, F_dyn = X_dynamic_raw.shape
F_sta       = X_static_raw.shape[1]

print(f"   总样本数: {N}  动态特征: {F_dyn}  静态特征: {F_sta}")
print(f"   y 范围: {y.min():.1f} ~ {y.max():.1f} mm w.e.")
print(f"   气候特征列: {climate_cols}")

unique_years = sorted(np.unique(years))
print(f"   将执行 {len(unique_years)} 折 LOYO 交叉验证")


# ───────────────────────────────────────────────────────────────────────────────
# 2. 初始化损失函数
# ───────────────────────────────────────────────────────────────────────────────
criterion = PhysicsInformedLoss(
    alpha=PHYSICS_ALPHA,
    temp_var_name=TEMP_VAR_NAME,
    tolerance=0.1,
)
print(f"\n>>> 损失函数: MSE + {PHYSICS_ALPHA} × Physics(temperature monotonicity)")


# ───────────────────────────────────────────────────────────────────────────────
# 3. LOYO 主循环
# ───────────────────────────────────────────────────────────────────────────────
print("\n>>> 2. 开始 LOYO 训练循环...")

all_preds    = []
fold_metrics = []

for fold_idx, test_year in enumerate(unique_years):
    train_mask = (years != test_year)
    test_mask  = (years == test_year)

    n_train = train_mask.sum()
    n_test  = test_mask.sum()

    if n_test == 0:
        continue
    if n_train < BATCH_SIZE:
        print(f"   [{fold_idx+1:3d}/{len(unique_years)}] 年 {test_year}: 训练样本不足 ({n_train})，跳过")
        continue

    # ── 归一化（仅用训练折统计量）─────────────────────────────────────────────
    X_dyn_tr, X_sta_tr, X_dyn_te, X_sta_te = normalize_fold(
        X_dynamic_raw[train_mask], X_static_raw[train_mask],
        X_dynamic_raw[test_mask],  X_static_raw[test_mask],
    )
    y_tr = y[train_mask]
    y_te = y[test_mask]

    # ── 构建 DataLoader ────────────────────────────────────────────────────────
    train_loader = make_loader(X_dyn_tr, X_sta_tr, y_tr, BATCH_SIZE, shuffle=True)

    # ── 初始化模型 ─────────────────────────────────────────────────────────────
    torch.manual_seed(SEED + fold_idx)
    model = GlacioFormer(
        n_dynamic_features = F_dyn,
        n_static_features  = F_sta,
        d_model            = p['d_model'],
        n_heads            = p['n_heads'],
        n_encoder_layers   = p['n_encoder_layers'],
        ff_dim             = p['ff_dim'],
        ema_scales         = p['ema_scales'],
        wt_levels          = p['wt_levels'],
        dropout            = p['dropout'],
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WD
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LR * 0.01
    )

    # ── 训练（含早停）──────────────────────────────────────────────────────────
    best_val_mse   = float('inf')
    best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    patience_count = 0

    for epoch in range(EPOCHS):
        # 训练
        model.train()
        for x_dyn, x_sta, y_true in train_loader:
            x_dyn  = x_dyn.to(DEVICE)
            x_sta  = x_sta.to(DEVICE)
            y_true = y_true.to(DEVICE)

            optimizer.zero_grad()
            y_pred = model(x_dyn, x_sta)
            losses = criterion(y_pred, y_true, x_dyn, climate_cols)
            losses['total'].backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        # 测试集验证（LOYO 策略中测试折本身作为 val 用于早停）
        model.eval()
        with torch.no_grad():
            xd = torch.tensor(X_dyn_te, dtype=torch.float32).to(DEVICE)
            xs = torch.tensor(X_sta_te, dtype=torch.float32).to(DEVICE)
            yt = torch.tensor(y_te,     dtype=torch.float32).to(DEVICE)
            val_losses = criterion(model(xd, xs), yt, xd, climate_cols)
        val_mse = val_losses['mse'].item()
        scheduler.step()

        # NaN 保护
        if not (val_mse < float('inf') and val_mse == val_mse):
            if epoch >= MIN_EPOCHS:
                patience_count += 1
                if patience_count >= EARLY_STOP_PATIENCE:
                    break
            continue

        # 更新最优权重 + 早停计数
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if epoch >= MIN_EPOCHS:
                patience_count = 0
        elif epoch >= MIN_EPOCHS:
            patience_count += 1
            if patience_count >= EARLY_STOP_PATIENCE:
                break

    # ── 恢复最优权重并预测 ─────────────────────────────────────────────────────
    model.load_state_dict(best_state)
    y_pred = predict(model, X_dyn_te, X_sta_te, batch_size=256)

    r2, rmse, mae, bias = compute_metrics(y_te, y_pred)
    fold_metrics.append({
        'YEAR': test_year, 'N': n_test,
        'R2': round(r2, 4), 'RMSE': round(rmse, 2),
        'MAE': round(mae, 2), 'Bias': round(bias, 2),
    })

    test_meta = df_meta[test_mask].reset_index(drop=True)
    for i in range(n_test):
        all_preds.append({
            'WGMS_ID': int(test_meta.loc[i, 'WGMS_ID']),
            'YEAR':    int(test_meta.loc[i, 'YEAR']),
            'y_true':  round(float(y_te[i]),   2),
            'y_pred':  round(float(y_pred[i]), 2),
        })

    print(f"   [{fold_idx+1:3d}/{len(unique_years)}] 年 {test_year}: "
          f"N_test={n_test:3d}  R²={r2:6.3f}  RMSE={rmse:7.1f}  "
          f"停止于 epoch {epoch+1:3d}/{EPOCHS}")


# ───────────────────────────────────────────────────────────────────────────────
# 4. 汇总与保存
# ───────────────────────────────────────────────────────────────────────────────
print("\n>>> 3. 保存结果...")
df_preds   = pd.DataFrame(all_preds)
df_metrics = pd.DataFrame(fold_metrics)

df_preds.to_csv(OUT_PRED,    index=False)
df_metrics.to_csv(OUT_METRICS, index=False)

if len(df_preds) > 0:
    yt_all = df_preds['y_true'].values
    yp_all = df_preds['y_pred'].values
    g_r2, g_rmse, g_mae, g_bias = compute_metrics(yt_all, yp_all)
    corr = np.corrcoef(yt_all, yp_all)[0, 1] if len(yt_all) > 1 else float('nan')

    summary_lines = [
        "=" * 60,
        "GlacioFormer LOYO 交叉验证汇总",
        "=" * 60,
        f"  总样本数      : {len(df_preds)}",
        f"  有效折数      : {len(df_metrics)}",
        f"  全局 R²       : {g_r2:.4f}",
        f"  全局 Pearson R: {corr:.4f}",
        f"  全局 RMSE     : {g_rmse:.2f} mm w.e.",
        f"  全局 MAE      : {g_mae:.2f} mm w.e.",
        f"  全局 Bias     : {g_bias:.2f} mm w.e.",
        "",
        "  逐折均值:",
        f"    R²   均值: {df_metrics['R2'].mean():.4f}  ± {df_metrics['R2'].std():.4f}",
        f"    RMSE 均值: {df_metrics['RMSE'].mean():.2f}  ± {df_metrics['RMSE'].std():.2f}",
        "",
        "  LSTM 基线 (参考):",
        "    全局 R²: 0.6538  RMSE: 605.23 mm  R: 0.8285",
        "=" * 60,
    ]

    for line in summary_lines:
        print(line)

    with open(OUT_SUMMARY, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines) + '\n')
    print(f"\n   汇总文件: {OUT_SUMMARY}")
else:
    print("   WARNING: 无有效预测结果，请检查数据路径")
