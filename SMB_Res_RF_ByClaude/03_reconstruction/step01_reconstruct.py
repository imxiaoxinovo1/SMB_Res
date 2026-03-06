"""
Step 01: 物质平衡重建
- 用全部训练数据训练最终 RF 模型
- 对所有 RGI02 冰川的所有年份进行预测
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from config import (TRAINING_DATA_CSV, PREPROCESS_DATA_DIR,
                    RECON_RESULTS_DIR, FEATURES, TARGET, RF_PARAMS)

os.makedirs(RECON_RESULTS_DIR, exist_ok=True)
output_csv = os.path.join(RECON_RESULTS_DIR, "RGI02_reconstruction.csv")

# ================= 1. 准备训练数据 =================
print(">>> 1. 加载训练数据...")
df_train = pd.read_csv(TRAINING_DATA_CSV)
df_train.columns = [c.replace('.1', '') for c in df_train.columns]
df_train = df_train.loc[:, ~df_train.columns.duplicated()]

# 修复 TAG=9999 海拔
df_bands = df_train[df_train['TAG'] != 9999]
if not df_bands.empty:
    glacier_bounds = df_bands.groupby('WGMS_ID').agg({
        'LOWER_BOUND': 'min', 'UPPER_BOUND': 'max'
    }).reset_index().rename(columns={
        'LOWER_BOUND': 'REAL_LOWER', 'UPPER_BOUND': 'REAL_UPPER'
    })
    df_train = pd.merge(df_train, glacier_bounds, on='WGMS_ID', how='left')
    mask = df_train['TAG'] == 9999
    df_train.loc[mask, 'LOWER_BOUND'] = df_train.loc[mask, 'REAL_LOWER'].fillna(df_train.loc[mask, 'LOWER_BOUND'])
    df_train.loc[mask, 'UPPER_BOUND'] = df_train.loc[mask, 'REAL_UPPER'].fillna(df_train.loc[mask, 'UPPER_BOUND'])
    df_train.drop(columns=['REAL_LOWER', 'REAL_UPPER'], inplace=True)

valid_features = [c for c in FEATURES if c in df_train.columns]
# sklearn RF (>=1.4) 支持特征 NaN，只需保证 TARGET 无 NaN
df_train_clean = df_train.dropna(subset=[TARGET])

print(f"   训练样本数: {len(df_train_clean)}")
print(f"   特征数: {len(valid_features)}")

# ================= 2. 训练最终模型 =================
print(">>> 2. 训练 RandomForest 模型...")
model = RandomForestRegressor(**RF_PARAMS)
model.fit(df_train_clean[valid_features], df_train_clean[TARGET])
print("   模型训练完成")

# ================= 3. 加载预测数据 =================
print(">>> 3. 加载重建输入数据...")
pred_csv = os.path.join(PREPROCESS_DATA_DIR, "rgi02_glaciers_era5.csv")
df_pred = pd.read_csv(pred_csv)
print(f"   原始行数: {len(df_pred)}, 冰川数: {df_pred['WGMS_ID'].nunique()}")

# 验证 AREA 单位
area_mean = df_pred['AREA'].mean()
if area_mean > 1000:
    print(f"   WARNING: AREA mean = {area_mean:.0f}, unit may be m2, converting...")
    df_pred['AREA'] = df_pred['AREA'] / 1_000_000
else:
    print(f"   OK: AREA mean = {area_mean:.2f} km2")

# 验证纬度
lat_max = df_pred['LATITUDE'].max()
if lat_max >= 60:
    print(f"   WARNING: 存在纬度 >= 60°N 的冰川，正在过滤...")
    df_pred = df_pred[df_pred['LATITUDE'] < 60].copy()
    print(f"   过滤后行数: {len(df_pred)}")

# 检查特征列对齐
pred_features = [c for c in valid_features if c in df_pred.columns]
feat_missing = set(valid_features) - set(pred_features)
if feat_missing:
    print(f"   ERROR: 预测数据缺失特征: {feat_missing}")
    sys.exit(1)

df_pred_clean = df_pred.dropna(subset=pred_features)
print(f"   有效预测行数: {len(df_pred_clean)}")

# ================= 4. 执行预测 =================
print(">>> 4. 执行预测...")
predictions_mm = model.predict(df_pred_clean[pred_features])
df_pred_clean = df_pred_clean.copy()
df_pred_clean['PREDICTED_SMB_mm'] = predictions_mm
df_pred_clean['PREDICTED_SMB_m'] = predictions_mm / 1000

# ================= 5. 物理合理性检查 =================
mean_smb = df_pred_clean['PREDICTED_SMB_m'].mean()
min_smb = df_pred_clean['PREDICTED_SMB_m'].min()
max_smb = df_pred_clean['PREDICTED_SMB_m'].max()

print(f"\n>>> 5. 物理合理性检查:")
print(f"   预测均值: {mean_smb:.3f} m w.e.")
print(f"   预测范围: {min_smb:.3f} ~ {max_smb:.3f} m w.e.")

if mean_smb < -3 or mean_smb > 1:
    print("   WARNING: 预测均值超出合理范围 (-3 ~ 1 m w.e.)!")
else:
    print("   OK: 预测均值在合理范围内")

if min_smb < -10:
    n_extreme = (df_pred_clean['PREDICTED_SMB_m'] < -5).sum()
    print(f"   WARNING: 存在 {n_extreme} 个极端负值 (< -5 m w.e.)")

# ================= 6. 保存 =================
output_cols = ['WGMS_ID', 'NAME', 'POLITICAL_UNIT', 'YEAR',
               'LATITUDE', 'LONGITUDE', 'AREA', 'LOWER_BOUND', 'UPPER_BOUND',
               'PREDICTED_SMB_mm', 'PREDICTED_SMB_m']
output_cols = [c for c in output_cols if c in df_pred_clean.columns]
df_output = df_pred_clean[output_cols].sort_values(['WGMS_ID', 'YEAR'])

df_output.to_csv(output_csv, index=False)
print(f"\n>>> 已保存: {output_csv}")
print(f"   总行数: {len(df_output)}")
print(f"   冰川数: {df_output['WGMS_ID'].nunique()}")
print(f"   年份范围: {df_output['YEAR'].min()} - {df_output['YEAR'].max()}")
