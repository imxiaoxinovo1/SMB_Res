"""
Step 03: 构建 LSTM 训练数据集

将月度气候序列与 WGMS 年度物质平衡观测值对齐，
生成双分支网络所需的三组张量：
  X_dynamic : (N, 12, F_dynamic)  — 时序分支输入（12 个月气象）
  X_static  : (N, F_static)       — 空间分支输入（5 个静态地形特征）
  y         : (N,)                — 目标（ANNUAL_BALANCE, mm w.e.）
  meta      : DataFrame           — 元数据（WGMS_ID, YEAR）

数据处理逻辑:
  1. 以 WGMS 有观测记录的 (冰川, 年) 对为主键
  2. 对每对，从月度 ERA5 中取 1-12 月完整序列（12 × F_dynamic）
  3. 缺失月份采用线性插值（≤3 个月）或跳过（>3 个月）
  4. 归一化：X_dynamic 按变量归一化，X_static 按特征归一化
  5. 输出 NPZ 文件（含缩放器参数供推理时使用）

输出: 01_preprocessing/data/
  lstm_dataset.npz         — 训练张量
  lstm_scaler_params.csv   — 归一化参数（均值、标准差）
  lstm_meta.csv            — 元数据（WGMS_ID, YEAR, ANNUAL_BALANCE）
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from config import (WGMS_MATCHED_CSV, PREPROCESS_DIR,
                    MONTHLY_CLIMATE_VARS, STATIC_FEATURES,
                    VAR_MAPPING, TRAIN_YEAR_MIN, TRAIN_YEAR_MAX, TARGET)

# ─── 输入 ────────────────────────────────────────────────────────────────────
monthly_csv     = os.path.join(PREPROCESS_DIR, "lstm_monthly_climate.csv")
glacier_list_csv= os.path.join(PREPROCESS_DIR, "lstm_glacier_list.csv")

# ─── 输出 ────────────────────────────────────────────────────────────────────
output_npz      = os.path.join(PREPROCESS_DIR, "lstm_dataset.npz")
output_meta_csv = os.path.join(PREPROCESS_DIR, "lstm_meta.csv")
output_scaler_csv=os.path.join(PREPROCESS_DIR, "lstm_scaler_params.csv")

MAX_MISSING_MONTHS = 3   # 缺失月份超过此值则丢弃该序列

# ===== 1. 读取月度气候数据 =====
print(">>> 1. 读取月度气候数据...")
df_monthly = pd.read_csv(monthly_csv)
print(f"   行数: {len(df_monthly)}  冰川数: {df_monthly['WGMS_ID'].nunique()}")

# 获取实际可用的气候列名（从 VAR_MAPPING 提取输出列名）
available_nc_vars = [v for v in MONTHLY_CLIMATE_VARS if v in VAR_MAPPING]
climate_cols = [VAR_MAPPING[v][0] for v in available_nc_vars]
climate_cols = [c for c in climate_cols if c in df_monthly.columns]
print(f"   动态特征列 ({len(climate_cols)}): {climate_cols}")

# ===== 2. 读取 WGMS 年度物质平衡（含静态特征）=====
print("\n>>> 2. 读取 WGMS 观测数据（含静态特征）...")
df_obs = pd.read_csv(WGMS_MATCHED_CSV)

# 只保留全冰川观测 (TAG=9999) 和有效年份
if 'TAG' in df_obs.columns:
    df_obs = df_obs[df_obs['TAG'] == 9999].copy()
df_obs = df_obs[(df_obs['YEAR'] >= TRAIN_YEAR_MIN) & (df_obs['YEAR'] <= TRAIN_YEAR_MAX)]
df_obs = df_obs.dropna(subset=[TARGET])
print(f"   有效观测记录: {len(df_obs)} 行，{df_obs['WGMS_ID'].nunique()} 个冰川")

# 将 WGMS 缺失标记 9999 替换为 NaN（LOWER_BOUND / UPPER_BOUND 海拔列）
for col in ['LOWER_BOUND', 'UPPER_BOUND']:
    if col in df_obs.columns:
        n_sentinel = (df_obs[col] == 9999).sum()
        if n_sentinel > 0:
            df_obs[col] = df_obs[col].replace(9999, np.nan)
            print(f"   {col}: 已将 {n_sentinel} 个 9999 替换为 NaN")

# 检查静态特征是否存在
missing_static = [c for c in STATIC_FEATURES if c not in df_obs.columns]
if missing_static:
    raise SystemExit(f"ERROR: 静态特征列缺失: {missing_static}\n请检查 WGMS_MATCHED_CSV 文件")

# ===== 3. 构建训练序列 =====
print("\n>>> 3. 构建 LSTM 训练序列...")
print(f"   策略: 以 WGMS 有观测的 (冰川, 年) 对为主键")
print(f"   每对提取 12 个月气象序列（缺失月份 ≤{MAX_MISSING_MONTHS} 时线性插值）")

X_dynamic_list = []
X_static_list  = []
y_list         = []
meta_list      = []

skipped_no_climate  = 0   # 月度数据中找不到该冰川
skipped_too_missing = 0   # 缺失月份过多
used = 0

# 对月度数据建立索引，加速查询
df_monthly_idx = df_monthly.set_index(['WGMS_ID', 'YEAR', 'MONTH'])

obs_glaciers = df_obs['WGMS_ID'].unique()
print(f"   待处理冰川: {len(obs_glaciers)} 个")

for wgms_id in sorted(obs_glaciers):
    df_g = df_obs[df_obs['WGMS_ID'] == wgms_id]

    # 检查该冰川是否有月度气候数据
    if wgms_id not in df_monthly['WGMS_ID'].values:
        skipped_no_climate += len(df_g)
        print(f"   SKIP 冰川 {wgms_id}: 月度气候数据不存在")
        continue

    # 取该冰川的静态特征（以所有观测记录的中位数为代表）
    static_vals = df_g[STATIC_FEATURES].median().values   # shape: (5,)

    for _, obs_row in df_g.iterrows():
        year = int(obs_row['YEAR'])
        smb  = float(obs_row[TARGET])

        # 提取该年 1-12 月的气候序列
        monthly_seq = []
        n_missing = 0

        for month in range(1, 13):
            try:
                row_vals = df_monthly_idx.loc[(wgms_id, year, month), climate_cols].values.astype(float)
                monthly_seq.append(row_vals)
            except KeyError:
                monthly_seq.append(None)
                n_missing += 1

        # 处理缺失月份
        if n_missing > MAX_MISSING_MONTHS:
            skipped_too_missing += 1
            continue

        if n_missing > 0:
            # 线性插值填补缺失月份（转换为 numpy，用 NaN 占位后插值）
            seq_arr = np.array([r if r is not None else np.full(len(climate_cols), np.nan)
                                for r in monthly_seq], dtype=float)   # (12, F)
            for f_idx in range(len(climate_cols)):
                col_vals = seq_arr[:, f_idx]
                nan_mask = np.isnan(col_vals)
                if nan_mask.any():
                    x_known = np.where(~nan_mask)[0]
                    y_known = col_vals[~nan_mask]
                    x_all   = np.arange(12)
                    if len(x_known) >= 2:
                        col_vals[nan_mask] = np.interp(x_all[nan_mask], x_known, y_known)
                    else:
                        col_vals[nan_mask] = np.nanmean(col_vals)
                    seq_arr[:, f_idx] = col_vals
        else:
            seq_arr = np.array(monthly_seq, dtype=float)   # (12, F)

        X_dynamic_list.append(seq_arr)
        X_static_list.append(static_vals)
        y_list.append(smb)
        meta_list.append({'WGMS_ID': wgms_id, 'YEAR': year, TARGET: smb})
        used += 1

print(f"\n   完成:")
print(f"   ✅ 有效序列: {used}")
print(f"   ⚠️  跳过（无月度数据）: {skipped_no_climate}")
print(f"   ⚠️  跳过（缺失月份>{MAX_MISSING_MONTHS}）: {skipped_too_missing}")

if used == 0:
    raise SystemExit("ERROR: 无有效序列！请检查 lstm_monthly_climate.csv 的冰川 ID 与 WGMS 数据是否匹配。")

# ===== 4. 转换为 NumPy 数组 =====
print("\n>>> 4. 转换为 NumPy 数组...")
X_dynamic = np.stack(X_dynamic_list, axis=0)   # (N, 12, F_dynamic)
X_static  = np.array(X_static_list,  dtype=float)  # (N, F_static)
y         = np.array(y_list,          dtype=float)  # (N,)

# 静态特征 NaN 插补（用各列中位数）
print("   静态特征 NaN 插补（列中位数）：")
for f_idx, col in enumerate(STATIC_FEATURES):
    nan_mask = np.isnan(X_static[:, f_idx])
    if nan_mask.any():
        col_median = np.nanmedian(X_static[:, f_idx])
        X_static[nan_mask, f_idx] = col_median
        print(f"   {col}: 填补 {nan_mask.sum()} 个 NaN → 中位数 {col_median:.2f}")
    else:
        print(f"   {col}: 无 NaN")

print(f"   X_dynamic shape: {X_dynamic.shape}  → (样本数, 月份=12, 动态特征={X_dynamic.shape[2]})")
print(f"   X_static  shape: {X_static.shape}   → (样本数, 静态特征={X_static.shape[1]})")
print(f"   y         shape: {y.shape}")

# ===== 5. 归一化 =====
print("\n>>> 5. 归一化（Z-score，按变量跨所有样本和月份计算）...")

# 5.1 动态特征：对每个变量，在所有 (N×12) 个观测上计算均值/标准差
#   reshape 到 (N*12, F_dynamic) → fit → reshape 回来
N, T, F = X_dynamic.shape
X_dyn_flat = X_dynamic.reshape(-1, F)   # (N*12, F)

dyn_mean = np.nanmean(X_dyn_flat, axis=0)
dyn_std  = np.nanstd( X_dyn_flat, axis=0)
dyn_std  = np.where(dyn_std < 1e-8, 1.0, dyn_std)   # 防止除以零

X_dynamic_norm = (X_dyn_flat - dyn_mean) / dyn_std
X_dynamic_norm = X_dynamic_norm.reshape(N, T, F)

# 5.2 静态特征：对每个静态变量计算均值/标准差
sta_mean = np.nanmean(X_static, axis=0)
sta_std  = np.nanstd( X_static, axis=0)
sta_std  = np.where(sta_std < 1e-8, 1.0, sta_std)

X_static_norm = (X_static - sta_mean) / sta_std

# ===== 6. 保存缩放器参数 =====
print("\n>>> 6. 保存归一化参数...")
scaler_rows = []
for i, col in enumerate(climate_cols):
    scaler_rows.append({'type': 'dynamic', 'feature': col,
                        'mean': dyn_mean[i], 'std': dyn_std[i]})
for i, col in enumerate(STATIC_FEATURES):
    scaler_rows.append({'type': 'static', 'feature': col,
                        'mean': sta_mean[i], 'std': sta_std[i]})

df_scaler = pd.DataFrame(scaler_rows)
df_scaler.to_csv(output_scaler_csv, index=False)
print(f"   已保存: {output_scaler_csv}")

# ===== 7. 保存元数据 =====
df_meta = pd.DataFrame(meta_list)
df_meta.to_csv(output_meta_csv, index=False)
print(f"   已保存: {output_meta_csv}")

# ===== 8. 保存 NPZ 数据集 =====
print("\n>>> 7. 保存 NPZ 数据集...")
np.savez_compressed(
    output_npz,
    X_dynamic      = X_dynamic_norm,          # 归一化后的动态特征
    X_static       = X_static_norm,           # 归一化后的静态特征
    X_dynamic_raw  = X_dynamic,               # 原始动态特征（供诊断用）
    X_static_raw   = X_static,                # 原始静态特征
    y              = y,                        # 目标值 (mm w.e.)
    climate_cols   = np.array(climate_cols),  # 动态特征列名
    static_cols    = np.array(STATIC_FEATURES),# 静态特征列名
    dyn_mean       = dyn_mean,
    dyn_std        = dyn_std,
    sta_mean       = sta_mean,
    sta_std        = sta_std,
)
print(f"   已保存: {output_npz}")

# ===== 9. 最终摘要 =====
print("\n" + "=" * 50)
print("✅ LSTM 数据集构建完成")
print("=" * 50)
print(f"  样本数 (N)          : {N}")
print(f"  时序长度            : {T} 个月（1-12 月）")
print(f"  动态特征 (F_dynamic): {F}  → {climate_cols}")
print(f"  静态特征 (F_static) : {len(STATIC_FEATURES)} → {STATIC_FEATURES}")
print(f"  目标变量            : {TARGET} (mm w.e.)")
print(f"  ANNUAL_BALANCE 范围 : {y.min():.1f} ~ {y.max():.1f} mm w.e.")
print(f"  ANNUAL_BALANCE 均值 : {y.mean():.1f} mm w.e.")
print(f"  冰川数              : {df_meta['WGMS_ID'].nunique()}")
print(f"  年份范围            : {df_meta['YEAR'].min()} - {df_meta['YEAR'].max()}")
print(f"\n  输出文件:")
print(f"    {output_npz}")
print(f"    {output_meta_csv}")
print(f"    {output_scaler_csv}")
print("=" * 50)
print("\n下一步: 运行 02_model/train_lstm.py 开始模型训练")
