"""
将 RGI v7.0 地形特征合并进 LSTM 数据集，生成 GlacioFormer 专用 NPZ

输入:
  - SMB_Res_LSTM_Byclaude/01_preprocessing/data/lstm_dataset.npz  (原始数据)
  - SMB_Res_LSTM_Byclaude/01_preprocessing/data/lstm_meta.csv     (样本元信息)
  - 01_preprocessing/data/glacier_terrain_features.csv             (RGI v7.0 地形)

输出:
  - 01_preprocessing/data/glacioformer_dataset.npz
      X_dynamic_raw : (N, 12, 15) — 月度气候特征（不变）
      X_static_raw  : (N, 10)     — 静态特征（原 5 + 新增 5 个地形特征）
      y             : (N,)        — 年度 SMB (mm w.e.)
      climate_cols  : (15,)       — 动态特征名
      static_cols   : (10,)       — 静态特征名

新增地形特征 (后 5 列):
  slope_deg   — RGI v7.0 冰川平均坡度 (°)
  aspect_sin  — 坡向正弦（东向分量，避免 0/360° 不连续）
  aspect_cos  — 坡向余弦（北向分量）
  zmean_m     — DEM 均值高程 (m)，Copernicus DEM 30m
  zmed_m      — DEM 中位高程 (m)（常用作 ELA 代理变量）
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd

from config import LSTM_PREPROCESS_DIR, BASE_DIR

# ─── 路径 ────────────────────────────────────────────────────────────────────
IN_NPZ       = os.path.join(LSTM_PREPROCESS_DIR, "lstm_dataset.npz")
IN_META      = os.path.join(LSTM_PREPROCESS_DIR, "lstm_meta.csv")
TERRAIN_CSV  = os.path.join(BASE_DIR, "01_preprocessing", "data",
                             "glacier_terrain_features.csv")
OUT_DIR      = os.path.join(BASE_DIR, "01_preprocessing", "data")
OUT_NPZ      = os.path.join(OUT_DIR, "glacioformer_dataset.npz")

os.makedirs(OUT_DIR, exist_ok=True)

# ─── 1. 加载原始 LSTM 数据集 ──────────────────────────────────────────────────
print(">>> 1. 加载 LSTM 数据集...")
data          = np.load(IN_NPZ, allow_pickle=True)
X_dynamic_raw = data['X_dynamic_raw']   # (N, 12, 15)
X_static_raw  = data['X_static_raw']    # (N, 5)
y             = data['y']               # (N,)
climate_cols  = list(data['climate_cols'])
static_cols   = list(data['static_cols'])

df_meta = pd.read_csv(IN_META)          # WGMS_ID, YEAR per sample
N = len(y)
print(f"   样本数: {N}  动态特征: {X_dynamic_raw.shape[2]}  静态特征: {X_static_raw.shape[1]}")
print(f"   原始静态特征: {static_cols}")

# ─── 2. 加载地形特征表 ───────────────────────────────────────────────────────
print("\n>>> 2. 加载地形特征...")
df_terrain = pd.read_csv(TERRAIN_CSV)
# 以 WGMS_ID 为键构建查找表
terrain_cols = ['slope_deg', 'aspect_sin', 'aspect_cos', 'zmean_m', 'zmed_m']
terrain_map  = df_terrain.set_index('WGMS_ID')[terrain_cols]
print(f"   地形冰川数: {len(terrain_map)}")
print(f"   新增特征: {terrain_cols}")

# ─── 3. 按 WGMS_ID 逐样本拼接地形特征 ──────────────────────────────────────
print("\n>>> 3. 按样本拼接地形特征...")
terrain_array = np.zeros((N, len(terrain_cols)), dtype=np.float32)
missing_ids   = set()

for i, wgms_id in enumerate(df_meta['WGMS_ID'].values):
    if wgms_id in terrain_map.index:
        terrain_array[i] = terrain_map.loc[wgms_id, terrain_cols].values.astype(np.float32)
    else:
        missing_ids.add(wgms_id)

if missing_ids:
    print(f"   ⚠️  以下 WGMS_ID 无地形数据（将用列均值填充）: {missing_ids}")
    for col_idx, col in enumerate(terrain_cols):
        col_mean = terrain_array[:, col_idx]
        col_mean = col_mean[col_mean != 0].mean() if (col_mean != 0).any() else 0.0
        for i, wgms_id in enumerate(df_meta['WGMS_ID'].values):
            if wgms_id in missing_ids:
                terrain_array[i, col_idx] = col_mean

# ─── 4. 合并静态特征 (N, 5) + 地形 (N, 5) → (N, 10) ──────────────────────
X_static_new = np.concatenate([X_static_raw, terrain_array], axis=1)
new_static_cols = static_cols + terrain_cols

print(f"\n   合并后静态特征: {new_static_cols}")
print(f"   X_static_raw 形状: {X_static_raw.shape} → {X_static_new.shape}")

# ─── 5. 数值范围检查 ────────────────────────────────────────────────────────
print("\n>>> 4. 地形特征统计:")
df_check = pd.DataFrame(terrain_array, columns=terrain_cols)
print(df_check.describe().round(2))

# ─── 6. 保存新 NPZ ──────────────────────────────────────────────────────────
np.savez_compressed(
    OUT_NPZ,
    X_dynamic_raw = X_dynamic_raw,
    X_static_raw  = X_static_new,
    y             = y,
    climate_cols  = np.array(climate_cols),
    static_cols   = np.array(new_static_cols),
)

print(f"\n>>> 已保存: {OUT_NPZ}")
print(f"    X_dynamic_raw : {X_dynamic_raw.shape}  (不变)")
print(f"    X_static_raw  : {X_static_new.shape}   (5 → 10 特征)")
print(f"    y             : {y.shape}")
print(f"    静态特征列    : {new_static_cols}")
