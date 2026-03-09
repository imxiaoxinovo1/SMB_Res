"""
Step 02: 训练最终 LSTM 模型（使用全部训练数据）

在全部 745 个训练样本上训练最终 TwoBranchLSTM：
  - 归一化参数在全体数据上计算（不存在折叠泄漏问题）
  - 随机切分 10% 作验证集（仅用于早停，不影响最终预测）
  - 保存最优权重供重建脚本调用

输出: 03_reconstruction/data/
  lstm_final_model.pt      — 模型权重（best_state）
  final_scaler.npz         — 归一化参数 (dyn_mean/std, sta_mean/std)
  final_static_medians.npy — 静态特征原始中位数（重建时 NaN 填充）
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '02_model'))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import PREPROCESS_DIR, RECON_DIR, LSTM_PARAMS
from lstm_model import TwoBranchLSTM

# ─── 可复现性 ────────────────────────────────────────────────────────────────
SEED = LSTM_PARAMS['random_state']
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f">>> 使用设备: {DEVICE}")

DATA_DIR = os.path.join(RECON_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

NPZ_PATH    = os.path.join(PREPROCESS_DIR, "lstm_dataset.npz")
OUT_MODEL   = os.path.join(DATA_DIR, "lstm_final_model.pt")
OUT_SCALER  = os.path.join(DATA_DIR, "final_scaler.npz")
OUT_MEDIANS = os.path.join(DATA_DIR, "final_static_medians.npy")

# ─── 超参数 ──────────────────────────────────────────────────────────────────
EPOCHS     = LSTM_PARAMS['epochs']
BATCH_SIZE = LSTM_PARAMS['batch_size']
LR         = LSTM_PARAMS['lr']
DROPOUT    = LSTM_PARAMS['dropout']

LSTM_HIDDEN = LSTM_PARAMS['lstm_hidden_dim']
LSTM_LAYERS = LSTM_PARAMS['lstm_num_layers']
LSTM_BIDIR  = LSTM_PARAMS['lstm_bidirectional']
MLP_HIDDEN  = LSTM_PARAMS['static_mlp_hidden']
MLP_LAYERS  = LSTM_PARAMS['static_mlp_layers']

EARLY_STOP_PATIENCE = LSTM_PARAMS.get('early_stop_patience', 40)
MIN_EPOCHS          = LSTM_PARAMS.get('min_epochs', 60)
VAL_RATIO           = 0.10   # 10% 随机切分作早停验证集


# ===== 1. 加载训练数据 =====
print(">>> 1. 加载训练数据（原始未归一化）...")
data = np.load(NPZ_PATH, allow_pickle=True)
X_dynamic_raw = data['X_dynamic_raw']   # (N, 12, F_dyn)
X_static_raw  = data['X_static_raw']    # (N, F_sta)
y             = data['y']               # (N,)

N, T, F_dyn = X_dynamic_raw.shape
F_sta = X_static_raw.shape[1]
print(f"   总样本数: {N}  动态特征: {F_dyn}  静态特征: {F_sta}")
print(f"   y 范围: {y.min():.1f} ~ {y.max():.1f} mm w.e.")


# ===== 2. 全体归一化（不分折）=====
print("\n>>> 2. 计算全体归一化参数（Z-score）...")

flat     = X_dynamic_raw.reshape(-1, F_dyn)
dyn_mean = flat.mean(axis=0)
dyn_std  = flat.std(axis=0)
dyn_std  = np.where(dyn_std < 1e-8, 1.0, dyn_std)

sta_mean = X_static_raw.mean(axis=0)
sta_std  = X_static_raw.std(axis=0)
sta_std  = np.where(sta_std < 1e-8, 1.0, sta_std)

X_dyn_norm = (X_dynamic_raw.reshape(-1, F_dyn) - dyn_mean) / dyn_std
X_dyn_norm = X_dyn_norm.reshape(N, T, F_dyn)
X_sta_norm = (X_static_raw - sta_mean) / sta_std

# 保存静态特征原始中位数（重建时 NaN 用此值填充后再归一化）
sta_medians = np.nanmedian(X_static_raw, axis=0)
np.save(OUT_MEDIANS, sta_medians)
print(f"   静态特征中位数: {sta_medians}")

# 保存归一化参数
np.savez(OUT_SCALER,
         dyn_mean=dyn_mean, dyn_std=dyn_std,
         sta_mean=sta_mean, sta_std=sta_std)
print(f"   已保存归一化参数: {OUT_SCALER}")


# ===== 3. 划分训练 / 验证集（仅用于早停）=====
print(f"\n>>> 3. 划分数据集（验证比例 {VAL_RATIO:.0%}）...")
np.random.seed(SEED)
idx = np.arange(N)
np.random.shuffle(idx)
n_val   = max(1, int(N * VAL_RATIO))
n_train = N - n_val
idx_train = idx[:n_train]
idx_val   = idx[n_train:]


def make_loader(X_dyn, X_sta, y_arr, batch_size, shuffle):
    ds = TensorDataset(
        torch.tensor(X_dyn, dtype=torch.float32),
        torch.tensor(X_sta, dtype=torch.float32),
        torch.tensor(y_arr, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


train_loader = make_loader(X_dyn_norm[idx_train], X_sta_norm[idx_train],
                           y[idx_train], BATCH_SIZE, True)
val_loader   = make_loader(X_dyn_norm[idx_val],   X_sta_norm[idx_val],
                           y[idx_val],   BATCH_SIZE, False)
print(f"   训练: {n_train} 样本  验证: {n_val} 样本")


# ===== 4. 初始化模型 =====
print("\n>>> 4. 初始化模型...")
torch.manual_seed(SEED)
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

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   可训练参数: {total_params:,}")

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 0.01)
criterion = nn.MSELoss()


# ===== 5. 训练（含早停）=====
print(f"\n>>> 5. 开始训练（最多 {EPOCHS} epoch）...")
print(f"   热身阶段: 前 {MIN_EPOCHS} epoch 不计 patience")
print(f"   早停 patience: {EARLY_STOP_PATIENCE} epoch")

best_val_loss  = float('inf')
best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
patience_count = 0

for epoch in range(EPOCHS):
    # ── 训练 ──
    model.train()
    train_loss = 0.0
    for x_dyn, x_sta, y_true in train_loader:
        x_dyn, x_sta, y_true = x_dyn.to(DEVICE), x_sta.to(DEVICE), y_true.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x_dyn, x_sta), y_true)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        train_loss += loss.item() * len(y_true)
    train_loss /= len(train_loader.dataset)

    # ── 验证 ──
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x_dyn, x_sta, y_true in val_loader:
            x_dyn, x_sta, y_true = x_dyn.to(DEVICE), x_sta.to(DEVICE), y_true.to(DEVICE)
            val_loss += criterion(model(x_dyn, x_sta), y_true).item() * len(y_true)
    val_loss /= len(val_loader.dataset)
    scheduler.step()

    # ── NaN 检查 ──
    if not (val_loss < float('inf') and val_loss == val_loss):
        if epoch >= MIN_EPOCHS:
            patience_count += 1
            if patience_count >= EARLY_STOP_PATIENCE:
                print(f"   epoch {epoch+1}: NaN 触发早停")
                break
        continue

    # ── 更新最优权重 ──
    if val_loss < best_val_loss:
        best_val_loss  = val_loss
        best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if epoch >= MIN_EPOCHS:
            patience_count = 0
    elif epoch >= MIN_EPOCHS:
        patience_count += 1
        if patience_count >= EARLY_STOP_PATIENCE:
            print(f"   epoch {epoch+1}: 早停触发 (patience={EARLY_STOP_PATIENCE})")
            break

    if (epoch + 1) % 20 == 0:
        print(f"   epoch {epoch+1:3d}/{EPOCHS}  "
              f"train={train_loss:.1f}  val={val_loss:.1f}  "
              f"patience={patience_count:2d}  best={best_val_loss:.1f}")

print(f"\n   最终停止于 epoch {epoch+1}，最佳 val_loss={best_val_loss:.2f}")


# ===== 6. 保存最优模型 =====
print("\n>>> 6. 保存模型...")
model.load_state_dict(best_state)
torch.save(model.state_dict(), OUT_MODEL)
print(f"   已保存: {OUT_MODEL}")

print("\n" + "=" * 50)
print("OK  最终模型训练完成")
print("=" * 50)
print(f"   模型文件:   {OUT_MODEL}")
print(f"   归一化参数: {OUT_SCALER}")
print(f"   静态中位数: {OUT_MEDIANS}")
print(f"\n下一步: 运行 step03_reconstruct.py 执行重建预测")
