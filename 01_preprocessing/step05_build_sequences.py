# 01_preprocessing/step05_build_sequences.py
"""
构建 GlacioFormer 月度序列数据集。
X_dyn: (N, 12, 15) — 每年12个月 × 15个气候变量（已归一化）
X_sta: (N, 10)     — 10个静态地形特征（已归一化）
y:     (N,)        — annual_balance_m (m w.e.)
归一化: 使用训练集 (year <= TRAIN_YEAR_MAX) 的 mean/std，保存到 npz 供重建使用
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
import numpy as np
from config import (ERA5_MONTHLY_CSV, TERRAIN_CSV, SEQUENCES_NPZ,
                    MASSBAL_RGI02_CSV, STATIC_FEATURES, MONTHLY_CLIMATE_VARS,
                    TRAIN_YEAR_MIN, TRAIN_YEAR_MAX, HOLDOUT_YEAR_MAX)

print("=== Step 05: Build Sequence Dataset ===")

df_era5    = pd.read_csv(ERA5_MONTHLY_CSV)
df_terrain = pd.read_csv(TERRAIN_CSV)
df_mb      = pd.read_csv(MASSBAL_RGI02_CSV)

terrain_cols = [c for c in STATIC_FEATURES if c in df_terrain.columns]

# ── Build one sample per (glacier_id, year) ──────────────────────────────────
X_dyn_list, X_sta_list, y_list, gid_list, year_list = [], [], [], [], []

df_mb_idx      = df_mb.set_index(['glacier_id', 'year'])
df_terrain_idx = df_terrain.set_index('glacier_id')

for gid, g_era5 in df_era5.groupby('glacier_id'):
    if gid not in df_terrain_idx.index:
        continue
    sta = df_terrain_idx.loc[gid, terrain_cols].values.astype(float)

    for year, y_era5 in g_era5.groupby('year'):
        y_era5_sorted = y_era5.sort_values('month')
        if len(y_era5_sorted) < 12:
            continue

        dyn = y_era5_sorted[MONTHLY_CLIMATE_VARS].values.astype(float)  # (12, 15)

        # Target (NaN if no observation — included for hold-out inference)
        key = (gid, year)
        target = df_mb_idx.loc[key, 'annual_balance'].item() \
                 if key in df_mb_idx.index else np.nan

        X_dyn_list.append(dyn)
        X_sta_list.append(sta)
        y_list.append(target)
        gid_list.append(gid)
        year_list.append(year)

X_dyn = np.array(X_dyn_list, dtype=np.float32)   # (N, 12, 15)
X_sta = np.array(X_sta_list, dtype=np.float32)   # (N, 10)
y     = np.array(y_list,    dtype=np.float32)    # (N,)
glacier_ids = np.array(gid_list)
years       = np.array(year_list)

print(f"Total samples: {len(y)}  (labeled: {(~np.isnan(y)).sum()})")
print(f"X_dyn raw: {X_dyn.shape}, X_sta raw: {X_sta.shape}")

# ── Compute normalization stats from training split only ─────────────────────
train_mask = years <= TRAIN_YEAR_MAX

dyn_mean = X_dyn[train_mask].mean(axis=(0, 1), keepdims=True)  # (1, 1, 15)
dyn_std  = X_dyn[train_mask].std(axis=(0, 1), keepdims=True) + 1e-8
sta_mean = X_sta[train_mask].mean(axis=0, keepdims=True)       # (1, 10)
sta_std  = X_sta[train_mask].std(axis=0, keepdims=True) + 1e-8

X_dyn_norm = (X_dyn - dyn_mean) / dyn_std
X_sta_norm = (X_sta - sta_mean) / sta_std

# Sanity checks
assert X_dyn_norm.shape[1:] == (12, 15), f"Expected (N,12,15), got {X_dyn_norm.shape}"
assert X_sta_norm.shape[1:] == (10,),    f"Expected (N,10), got {X_sta_norm.shape}"
assert len(y) == len(X_dyn_norm) == len(X_sta_norm)
labeled_count = int((~np.isnan(y)).sum())
assert labeled_count >= 1000, f"Expected ≥1000 labeled samples, got {labeled_count}"

np.savez_compressed(
    SEQUENCES_NPZ,
    X_dyn=X_dyn_norm, X_sta=X_sta_norm, y=y,
    glacier_ids=glacier_ids, years=years,
    dyn_mean=dyn_mean, dyn_std=dyn_std,
    sta_mean=sta_mean, sta_std=sta_std,
)
print(f"Saved → {SEQUENCES_NPZ}")
print(f"X_dyn: {X_dyn_norm.shape}, X_sta: {X_sta_norm.shape}, y: {y.shape}")
print(f"dyn_mean shape: {dyn_mean.shape}, sta_mean shape: {sta_mean.shape}")
print(f"Labeled samples: {labeled_count}")
