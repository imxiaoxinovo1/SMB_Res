"""
Step 01: 物质平衡重建
- 根据 compare_models.py 选出的最优模型进行重建
- 对所有 RGI02 冰川的所有年份进行预测
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from config import (TRAINING_DATA_CSV, RECONSTRUCTION_INPUT_CSV,
                    MODEL_RESULTS_DIR, RECON_RESULTS_DIR,
                    FEATURES, TARGET, XGB_PARAMS, LGB_PARAMS, RF_PARAMS)

os.makedirs(RECON_RESULTS_DIR, exist_ok=True)
output_csv = os.path.join(RECON_RESULTS_DIR, "RGI02_reconstruction.csv")

# ================= 0. 确定最优模型 =================
best_model_file = os.path.join(MODEL_RESULTS_DIR, "best_model.txt")
if os.path.exists(best_model_file):
    with open(best_model_file, 'r') as f:
        best_model_name = f.read().strip()
else:
    best_model_name = "XGBoost"
    print(f"   WARNING: best_model.txt 未找到，默认使用 {best_model_name}")

print(f">>> 使用最优模型: {best_model_name}")

if best_model_name == "XGBoost":
    from xgboost import XGBRegressor
    ModelClass = XGBRegressor
    model_params = XGB_PARAMS
elif best_model_name == "LightGBM":
    from lightgbm import LGBMRegressor
    ModelClass = LGBMRegressor
    model_params = LGB_PARAMS
else:
    from sklearn.ensemble import RandomForestRegressor
    ModelClass = RandomForestRegressor
    model_params = RF_PARAMS

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
df_train_clean = df_train.dropna(subset=[TARGET])

print(f"   训练样本数: {len(df_train_clean)}")
print(f"   特征数: {len(valid_features)}")

# ================= 2. 训练最终模型 =================
print(f">>> 2. 训练 {best_model_name} 模型...")
model = ModelClass(**model_params)
model.fit(df_train_clean[valid_features], df_train_clean[TARGET])
print("   模型训练完成")

# ================= 3. 加载预测数据 =================
print(">>> 3. 加载重建输入数据...")
df_pred = pd.read_csv(RECONSTRUCTION_INPUT_CSV)
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
    print(f"   WARNING: 存在纬度 >= 60N 的冰川，正在过滤...")
    df_pred = df_pred[df_pred['LATITUDE'] < 60].copy()
    print(f"   过滤后行数: {len(df_pred)}")

# 检查特征列对齐
pred_features = [c for c in valid_features if c in df_pred.columns]
feat_missing = set(valid_features) - set(pred_features)
if feat_missing:
    print(f"   ERROR: 预测数据缺失特征: {feat_missing}")
    sys.exit(1)

# XGBoost/LightGBM 原生支持 NaN，不需要 dropna 特征列
print(f"   有效预测行数: {len(df_pred)}")

# ================= 4. 执行预测 =================
print(">>> 4. 执行预测...")
predictions_mm = model.predict(df_pred[pred_features])
df_pred = df_pred.copy()
df_pred['PREDICTED_SMB_mm'] = predictions_mm
df_pred['PREDICTED_SMB_m'] = predictions_mm / 1000

# ================= 5. 物理合理性检查 =================
mean_smb = df_pred['PREDICTED_SMB_m'].mean()
min_smb = df_pred['PREDICTED_SMB_m'].min()
max_smb = df_pred['PREDICTED_SMB_m'].max()

print(f"\n>>> 5. 物理合理性检查:")
print(f"   预测均值: {mean_smb:.3f} m w.e.")
print(f"   预测范围: {min_smb:.3f} ~ {max_smb:.3f} m w.e.")

if mean_smb < -3 or mean_smb > 1:
    print("   WARNING: 预测均值超出合理范围 (-3 ~ 1 m w.e.)!")
else:
    print("   OK: 预测均值在合理范围内")

if min_smb < -10:
    n_extreme = (df_pred['PREDICTED_SMB_m'] < -5).sum()
    print(f"   WARNING: 存在 {n_extreme} 个极端负值 (< -5 m w.e.)")

# ================= 6. 保存 =================
output_cols = ['WGMS_ID', 'NAME', 'POLITICAL_UNIT', 'YEAR',
               'LATITUDE', 'LONGITUDE', 'AREA', 'LOWER_BOUND', 'UPPER_BOUND',
               'PREDICTED_SMB_mm', 'PREDICTED_SMB_m']
output_cols = [c for c in output_cols if c in df_pred.columns]
df_output = df_pred[output_cols].sort_values(['WGMS_ID', 'YEAR'])

df_output.to_csv(output_csv, index=False)
print(f"\n>>> 已保存: {output_csv}")
print(f"   总行数: {len(df_output)}")
print(f"   冰川数: {df_output['WGMS_ID'].nunique()}")
print(f"   年份范围: {df_output['YEAR'].min()} - {df_output['YEAR'].max()}")
print(f"   使用模型: {best_model_name}")
