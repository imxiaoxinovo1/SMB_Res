"""
Step 03: LSTM 重建 RGI02 所有冰川的年度物质平衡（RECON_YEAR_MIN ~ RECON_YEAR_MAX）

对 rgi02_monthly_climate.csv 中每个 (冰川, 年) 对，使用与训练完全一致的：
  - 特征列 (MONTHLY_CLIMATE_VARS → 15 个气候变量)
  - 缺失月份处理（线性插值，≤3 个月）
  - Z-score 归一化（用 final_scaler.npz 中的训练集参数）
  - 静态特征 NaN 填充（用训练集中位数 final_static_medians.npy）

调用最终 LSTM 模型，逐 (冰川, 年) 预测 ANNUAL_BALANCE（mm w.e.）。

输出: 03_reconstruction/results/RGI02_LSTM_reconstruction.csv
字段: WGMS_ID, NAME, POLITICAL_UNIT, YEAR, LATITUDE, LONGITUDE, AREA,
      LOWER_BOUND, UPPER_BOUND, PREDICTED_SMB_mm, PREDICTED_SMB_m
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '02_model'))

import numpy as np
import pandas as pd
import torch

from config import (RECON_DIR, RGI02_ERA5_CSV,
                    MONTHLY_CLIMATE_VARS, STATIC_FEATURES, VAR_MAPPING,
                    RECON_YEAR_MIN, RECON_YEAR_MAX, LSTM_PARAMS)
from lstm_model import TwoBranchLSTM

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f">>> 使用设备: {DEVICE}")

DATA_DIR    = os.path.join(RECON_DIR, "data")
RESULTS_DIR = os.path.join(RECON_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

MONTHLY_CSV  = os.path.join(DATA_DIR, "rgi02_monthly_climate.csv")
MODEL_PATH   = os.path.join(DATA_DIR, "lstm_final_model.pt")
SCALER_PATH  = os.path.join(DATA_DIR, "final_scaler.npz")
MEDIANS_PATH = os.path.join(DATA_DIR, "final_static_medians.npy")
OUTPUT_CSV   = os.path.join(RESULTS_DIR, "RGI02_LSTM_reconstruction.csv")

MAX_MISSING_MONTHS = 3   # 缺失月份超过此值则跳过该 (冰川, 年) 对


# ===== 1. 加载归一化参数与模型 =====
print(">>> 1. 加载归一化参数与模型...")

scaler      = np.load(SCALER_PATH)
dyn_mean    = scaler['dyn_mean']
dyn_std     = scaler['dyn_std']
sta_mean    = scaler['sta_mean']
sta_std     = scaler['sta_std']
sta_medians = np.load(MEDIANS_PATH)

F_dyn = len(dyn_mean)
F_sta = len(sta_mean)

model = TwoBranchLSTM(
    n_dynamic_features = F_dyn,
    n_static_features  = F_sta,
    lstm_hidden_dim    = LSTM_PARAMS['lstm_hidden_dim'],
    lstm_num_layers    = LSTM_PARAMS['lstm_num_layers'],
    bidirectional      = LSTM_PARAMS['lstm_bidirectional'],
    static_mlp_hidden  = LSTM_PARAMS['static_mlp_hidden'],
    static_mlp_layers  = LSTM_PARAMS['static_mlp_layers'],
    dropout            = LSTM_PARAMS['dropout'],
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print(f"   模型加载完成  F_dyn={F_dyn}  F_sta={F_sta}")


# ===== 2. 加载月度气候数据与静态特征 =====
print("\n>>> 2. 加载月度气候与静态特征数据...")

df_monthly = pd.read_csv(MONTHLY_CSV)
df_rgi02   = pd.read_csv(RGI02_ERA5_CSV)

# 获取气候列名（与训练时一致）
available_nc_vars = [v for v in MONTHLY_CLIMATE_VARS if v in VAR_MAPPING]
climate_cols = [VAR_MAPPING[v][0] for v in available_nc_vars]
climate_cols = [c for c in climate_cols if c in df_monthly.columns]
print(f"   动态特征列 ({len(climate_cols)}): {climate_cols}")

if len(climate_cols) != F_dyn:
    raise SystemExit(
        f"ERROR: 气候列数 ({len(climate_cols)}) 与模型输入维度 ({F_dyn}) 不一致！"
        f"\n请确认 rgi02_monthly_climate.csv 与训练数据使用相同变量。"
    )

# 静态特征：每冰川取中位数（RGI02 数据中无 9999 哨兵值）
df_static_vals = df_rgi02.groupby('WGMS_ID')[STATIC_FEATURES].median()
df_static_meta = df_rgi02.groupby('WGMS_ID')[['NAME', 'POLITICAL_UNIT']].first()
df_static = df_static_vals.join(df_static_meta).reset_index()
print(f"   冰川静态特征: {len(df_static)} 个冰川")

# 建立月度数据索引（加速查询）
df_monthly_idx = df_monthly.set_index(['WGMS_ID', 'YEAR', 'MONTH'])

glaciers = df_static['WGMS_ID'].values
years    = range(RECON_YEAR_MIN, RECON_YEAR_MAX + 1)
total_pairs = len(glaciers) * len(list(years))
print(f"   待处理 (冰川, 年) 对: {total_pairs}")


# ===== 3. 逐冰川逐年构建序列并预测 =====
print(f"\n>>> 3. 重建 {RECON_YEAR_MIN}-{RECON_YEAR_MAX}（共 {len(glaciers)} 个冰川）...")

records          = []
skipped_no_data  = 0
skipped_too_miss = 0

for g_i, wgms_id in enumerate(glaciers):
    if (g_i + 1) % 20 == 0 or g_i == 0:
        print(f"   进度: {g_i+1}/{len(glaciers)} 冰川  已记录: {len(records)}")

    # ── 静态特征 ──────────────────────────────────────────────────────────
    sta_row    = df_static[df_static['WGMS_ID'] == wgms_id].iloc[0]
    static_raw = np.array([sta_row[f] for f in STATIC_FEATURES], dtype=float)

    # NaN 填充（用训练集中位数，与 step03_build_lstm_dataset.py 一致）
    for i, val in enumerate(static_raw):
        if np.isnan(val):
            static_raw[i] = sta_medians[i]

    # 归一化静态特征
    static_norm = (static_raw - sta_mean) / sta_std

    name    = sta_row.get('NAME', '')
    polunit = sta_row.get('POLITICAL_UNIT', '')

    # ── 预先构建该冰川的静态张量（避免逐年重复创建）────────────────────────
    x_sta_tensor = torch.tensor(static_norm[None, :], dtype=torch.float32).to(DEVICE)

    for year in years:
        # ── 提取 12 个月气候序列 ──────────────────────────────────────────
        monthly_seq = []
        n_missing   = 0

        for month in range(1, 13):
            try:
                row_vals = (df_monthly_idx
                            .loc[(wgms_id, year, month), climate_cols]
                            .values.astype(float))
                monthly_seq.append(row_vals)
            except KeyError:
                monthly_seq.append(None)
                n_missing += 1

        # 跳过数据完全缺失的年份
        if n_missing == 12:
            skipped_no_data += 1
            continue

        # 跳过缺失月份过多的年份
        if n_missing > MAX_MISSING_MONTHS:
            skipped_too_miss += 1
            continue

        # ── 线性插值填补缺失月份 ──────────────────────────────────────────
        if n_missing > 0:
            seq_arr = np.array(
                [r if r is not None else np.full(len(climate_cols), np.nan)
                 for r in monthly_seq],
                dtype=float
            )   # (12, F_dyn)
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
            seq_arr = np.array(monthly_seq, dtype=float)   # (12, F_dyn)

        # ── 归一化动态特征 ────────────────────────────────────────────────
        dyn_norm = (seq_arr.reshape(-1, F_dyn) - dyn_mean) / dyn_std
        dyn_norm = dyn_norm.reshape(1, 12, F_dyn)

        # ── 推理 ──────────────────────────────────────────────────────────
        with torch.no_grad():
            x_dyn    = torch.tensor(dyn_norm, dtype=torch.float32).to(DEVICE)
            pred_mm  = float(model(x_dyn, x_sta_tensor).cpu().item())

        records.append({
            'WGMS_ID':          int(wgms_id),
            'NAME':             name,
            'POLITICAL_UNIT':   polunit,
            'YEAR':             int(year),
            'LATITUDE':         float(sta_row['LATITUDE']),
            'LONGITUDE':        float(sta_row['LONGITUDE']),
            'AREA':             float(sta_row['AREA']),
            'LOWER_BOUND':      float(static_raw[0]),
            'UPPER_BOUND':      float(static_raw[1]),
            'PREDICTED_SMB_mm': round(pred_mm, 2),
            'PREDICTED_SMB_m':  round(pred_mm / 1000, 4),
        })

print(f"\n   重建完成:")
print(f"   有效预测: {len(records)}")
print(f"   跳过（无月度数据）: {skipped_no_data}")
print(f"   跳过（缺失月份>{MAX_MISSING_MONTHS}）: {skipped_too_miss}")


# ===== 4. 物理合理性检查 =====
df_out = pd.DataFrame(records)
df_out = df_out.sort_values(['WGMS_ID', 'YEAR']).reset_index(drop=True)

mean_smb = df_out['PREDICTED_SMB_m'].mean()
min_smb  = df_out['PREDICTED_SMB_m'].min()
max_smb  = df_out['PREDICTED_SMB_m'].max()

print(f"\n>>> 4. 物理合理性检查:")
print(f"   预测均值: {mean_smb:.3f} m w.e.")
print(f"   预测范围: {min_smb:.3f} ~ {max_smb:.3f} m w.e.")

if mean_smb < -3 or mean_smb > 1:
    print("   WARNING: 预测均值超出合理范围 (-3 ~ 1 m w.e.)!")
else:
    print("   OK: 预测均值在合理范围内")

n_extreme = (df_out['PREDICTED_SMB_m'] < -5).sum()
if n_extreme > 0:
    print(f"   WARNING: 存在 {n_extreme} 个极端负值 (< -5 m w.e.)")


# ===== 5. 保存 =====
df_out.to_csv(OUTPUT_CSV, index=False)
print(f"\n>>> 已保存: {OUTPUT_CSV}")
print(f"   总行数: {len(df_out)}")
print(f"   冰川数: {df_out['WGMS_ID'].nunique()}")
print(f"   年份范围: {df_out['YEAR'].min()} - {df_out['YEAR'].max()}")
print(f"\n下一步: 运行 step04_hybrid_dataset.py 构建混合数据集")
