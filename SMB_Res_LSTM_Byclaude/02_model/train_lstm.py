"""
LSTM 双分支模型 — LOYO（Leave-One-Year-Out）交叉验证训练

数据流:
  lstm_dataset.npz
    ├─ X_dynamic_raw   (N, 12, F_dynamic)  — 原始动态特征（未归一化）
    ├─ X_static_raw    (N, F_static)        — 原始静态特征
    ├─ y               (N,)                 — ANNUAL_BALANCE (mm w.e.)
    ├─ climate_cols                          — 动态特征列名
    └─ static_cols                          — 静态特征列名
  lstm_meta.csv
    ├─ WGMS_ID, YEAR, ANNUAL_BALANCE

LOYO 策略:
  每折以单个年份作为测试集，其余年份训练；
  归一化参数由训练折计算，防止数据泄漏。

输出: 02_model/results/
  lstm_loyo_predictions.csv  — 逐样本预测 (WGMS_ID, YEAR, y_true, y_pred)
  lstm_loyo_metrics.csv      — 逐折指标 (YEAR, R2, RMSE, MAE, Bias, N)
  lstm_loyo_summary.txt      — 全折汇总指标
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import PREPROCESS_DIR, RESULTS_DIR, LSTM_PARAMS, N_CLIMATE_FEATURES, N_STATIC_FEATURES
from lstm_model import TwoBranchLSTM

# ─── 可复现性 ────────────────────────────────────────────────────────────────
SEED = LSTM_PARAMS['random_state']
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f">>> 使用设备: {DEVICE}")

# ─── 输入 / 输出路径 ─────────────────────────────────────────────────────────
NPZ_PATH  = os.path.join(PREPROCESS_DIR, "lstm_dataset.npz")
META_PATH = os.path.join(PREPROCESS_DIR, "lstm_meta.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)
OUT_PRED    = os.path.join(RESULTS_DIR, "lstm_loyo_predictions.csv")
OUT_METRICS = os.path.join(RESULTS_DIR, "lstm_loyo_metrics.csv")
OUT_SUMMARY = os.path.join(RESULTS_DIR, "lstm_loyo_summary.txt")

# ─── 超参数 ──────────────────────────────────────────────────────────────────
EPOCHS     = LSTM_PARAMS['epochs']
BATCH_SIZE = LSTM_PARAMS['batch_size']
LR         = LSTM_PARAMS['lr']
DROPOUT    = LSTM_PARAMS['dropout']

LSTM_HIDDEN  = LSTM_PARAMS['lstm_hidden_dim']
LSTM_LAYERS  = LSTM_PARAMS['lstm_num_layers']
LSTM_BIDIR   = LSTM_PARAMS['lstm_bidirectional']
MLP_HIDDEN   = LSTM_PARAMS['static_mlp_hidden']
MLP_LAYERS   = LSTM_PARAMS['static_mlp_layers']

EARLY_STOP_PATIENCE = LSTM_PARAMS.get('early_stop_patience', 40)
MIN_EPOCHS          = LSTM_PARAMS.get('min_epochs', 60)


# ───────────────────────────────────────────────────────────────────────────────
# 辅助函数
# ───────────────────────────────────────────────────────────────────────────────

def normalize_fold(X_dyn_train, X_sta_train, X_dyn_test, X_sta_test):
    """
    以训练折数据计算 Z-score 参数，归一化训练集和测试集。
    返回 (X_dyn_tr_norm, X_sta_tr_norm, X_dyn_te_norm, X_sta_te_norm)
    """
    N, T, F = X_dyn_train.shape

    # 动态特征：在所有 (N×12) 上统计
    flat_tr = X_dyn_train.reshape(-1, F)
    dyn_mean = flat_tr.mean(axis=0)
    dyn_std  = flat_tr.std(axis=0)
    dyn_std  = np.where(dyn_std < 1e-8, 1.0, dyn_std)

    X_dyn_tr_norm = (X_dyn_train.reshape(-1, F) - dyn_mean) / dyn_std
    X_dyn_tr_norm = X_dyn_tr_norm.reshape(N, T, F)

    N_te = X_dyn_test.shape[0]
    X_dyn_te_norm = (X_dyn_test.reshape(-1, F) - dyn_mean) / dyn_std
    X_dyn_te_norm = X_dyn_te_norm.reshape(N_te, T, F)

    # 静态特征
    sta_mean = X_sta_train.mean(axis=0)
    sta_std  = X_sta_train.std(axis=0)
    sta_std  = np.where(sta_std < 1e-8, 1.0, sta_std)

    X_sta_tr_norm = (X_sta_train - sta_mean) / sta_std
    X_sta_te_norm = (X_sta_test  - sta_mean) / sta_std

    return X_dyn_tr_norm, X_sta_tr_norm, X_dyn_te_norm, X_sta_te_norm


def make_loader(X_dyn, X_sta, y, batch_size, shuffle=True):
    """将 numpy 数组封装为 DataLoader。"""
    ds = TensorDataset(
        torch.tensor(X_dyn, dtype=torch.float32),
        torch.tensor(X_sta, dtype=torch.float32),
        torch.tensor(y,     dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for x_dyn, x_sta, y_true in loader:
        x_dyn, x_sta, y_true = x_dyn.to(DEVICE), x_sta.to(DEVICE), y_true.to(DEVICE)
        optimizer.zero_grad()
        y_pred = model(x_dyn, x_sta)
        loss   = criterion(y_pred, y_true)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * len(y_true)
    return total_loss / len(loader.dataset)


def eval_loss(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x_dyn, x_sta, y_true in loader:
            x_dyn, x_sta, y_true = x_dyn.to(DEVICE), x_sta.to(DEVICE), y_true.to(DEVICE)
            y_pred = model(x_dyn, x_sta)
            total_loss += criterion(y_pred, y_true).item() * len(y_true)
    return total_loss / len(loader.dataset)


def predict(model, X_dyn, X_sta, batch_size=256):
    model.eval()
    preds = []
    loader = make_loader(X_dyn, X_sta, np.zeros(len(X_dyn)), batch_size, shuffle=False)
    with torch.no_grad():
        for x_dyn, x_sta, _ in loader:
            x_dyn, x_sta = x_dyn.to(DEVICE), x_sta.to(DEVICE)
            preds.append(model(x_dyn, x_sta).cpu().numpy())
    return np.concatenate(preds)


def compute_metrics(y_true, y_pred):
    """计算 R², RMSE, MAE, Bias。"""
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
data = np.load(NPZ_PATH, allow_pickle=True)
X_dynamic_raw = data['X_dynamic_raw']   # (N, 12, F_dyn)
X_static_raw  = data['X_static_raw']    # (N, F_sta)
y             = data['y']               # (N,)

df_meta = pd.read_csv(META_PATH)
years   = df_meta['YEAR'].values

N, T, F_dyn = X_dynamic_raw.shape
F_sta       = X_static_raw.shape[1]

print(f"   总样本数: {N}")
print(f"   X_dynamic_raw: {X_dynamic_raw.shape}  X_static_raw: {X_static_raw.shape}")
print(f"   y 范围: {y.min():.1f} ~ {y.max():.1f} mm w.e.")
print(f"   年份范围: {years.min()} - {years.max()}  (共 {len(np.unique(years))} 个年份)")

unique_years = sorted(np.unique(years))
print(f"   将执行 {len(unique_years)} 折 LOYO 交叉验证")


# ───────────────────────────────────────────────────────────────────────────────
# 2. LOYO 交叉验证主循环
# ───────────────────────────────────────────────────────────────────────────────
print("\n>>> 2. 开始 LOYO 训练循环...")
criterion = nn.MSELoss()

all_preds   = []
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

    # ── 归一化（仅用训练集统计量）──────────────────────────────────────────
    X_dyn_tr, X_sta_tr, X_dyn_te, X_sta_te = normalize_fold(
        X_dynamic_raw[train_mask], X_static_raw[train_mask],
        X_dynamic_raw[test_mask],  X_static_raw[test_mask],
    )
    y_tr = y[train_mask]
    y_te = y[test_mask]

    # ── 构建 DataLoader ────────────────────────────────────────────────────
    train_loader = make_loader(X_dyn_tr, X_sta_tr, y_tr, BATCH_SIZE, shuffle=True)
    val_loader   = make_loader(X_dyn_te, X_sta_te, y_te, BATCH_SIZE, shuffle=False)

    # ── 初始化模型和优化器 ──────────────────────────────────────────────────
    torch.manual_seed(SEED + fold_idx)
    model = TwoBranchLSTM(
        n_dynamic_features = F_dyn,
        n_static_features  = F_sta,
        lstm_hidden_dim    = LSTM_HIDDEN,
        lstm_num_layers    = LSTM_LAYERS,
        bidirectional      = LSTM_BIDIR,
        static_mlp_hidden  = MLP_HIDDEN,
        static_mlp_layers  = MLP_LAYERS,
        dropout            = DROPOUT,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 0.01)

    # ── 训练（含早停）─────────────────────────────────────────────────────
    best_val_loss  = float('inf')
    # 用初始权重兜底，防止第一个 epoch 产生 NaN 时 best_state 为 None
    best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    patience_count = 0

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss   = eval_loss(model, val_loader, criterion)
        scheduler.step()

        # 跳过 NaN/Inf 损失（数值不稳定时不更新最优权重）
        if not (val_loss < float('inf') and val_loss == val_loss):
            if epoch >= MIN_EPOCHS:
                patience_count += 1
                if patience_count >= EARLY_STOP_PATIENCE:
                    break
            continue

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            # 热身阶段不重置 patience（让 MIN_EPOCHS 真正起效）
            if epoch >= MIN_EPOCHS:
                patience_count = 0
        elif epoch >= MIN_EPOCHS:
            # 热身结束后才开始计 patience
            patience_count += 1
            if patience_count >= EARLY_STOP_PATIENCE:
                break

    # ── 恢复最优权重并预测 ─────────────────────────────────────────────────
    model.load_state_dict(best_state)
    y_pred = predict(model, X_dyn_te, X_sta_te, batch_size=256)

    r2, rmse, mae, bias = compute_metrics(y_te, y_pred)
    fold_metrics.append({
        'YEAR': test_year, 'N': n_test,
        'R2': round(r2, 4), 'RMSE': round(rmse, 2),
        'MAE': round(mae, 2), 'Bias': round(bias, 2),
    })

    # 记录逐样本预测
    test_meta = df_meta[test_mask].reset_index(drop=True)
    for i in range(n_test):
        all_preds.append({
            'WGMS_ID': int(test_meta.loc[i, 'WGMS_ID']),
            'YEAR'   : int(test_meta.loc[i, 'YEAR']),
            'y_true' : round(float(y_te[i]),   2),
            'y_pred' : round(float(y_pred[i]), 2),
        })

    stopped_at = epoch + 1
    print(f"   [{fold_idx+1:3d}/{len(unique_years)}] 年 {test_year}: "
          f"N_test={n_test:3d}  R²={r2:6.3f}  RMSE={rmse:7.1f}  "
          f"停止于 epoch {stopped_at:3d}/{EPOCHS}")


# ───────────────────────────────────────────────────────────────────────────────
# 3. 汇总与保存
# ───────────────────────────────────────────────────────────────────────────────
print("\n>>> 3. 保存结果...")

df_preds   = pd.DataFrame(all_preds)
df_metrics = pd.DataFrame(fold_metrics)

df_preds.to_csv(OUT_PRED, index=False)
df_metrics.to_csv(OUT_METRICS, index=False)
print(f"   逐样本预测: {OUT_PRED}")
print(f"   逐折指标:   {OUT_METRICS}")

# ── 全局指标 ──────────────────────────────────────────────────────────────────
if len(df_preds) > 0:
    yt_all = df_preds['y_true'].values
    yp_all = df_preds['y_pred'].values
    g_r2, g_rmse, g_mae, g_bias = compute_metrics(yt_all, yp_all)

    # 相关系数
    corr = np.corrcoef(yt_all, yp_all)[0, 1] if len(yt_all) > 1 else float('nan')

    summary_lines = [
        "=" * 55,
        "✅  LSTM LOYO 交叉验证汇总",
        "=" * 55,
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
        "=" * 55,
        f"  预测文件: {OUT_PRED}",
        f"  指标文件: {OUT_METRICS}",
        "=" * 55,
    ]

    for line in summary_lines:
        print(line)

    with open(OUT_SUMMARY, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines) + '\n')
    print(f"\n   汇总文件: {OUT_SUMMARY}")
else:
    print("   WARNING: 无有效预测结果，请检查数据")
