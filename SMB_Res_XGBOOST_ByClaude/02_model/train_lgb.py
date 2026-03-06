"""
LightGBM 模型训练与交叉验证
- Leave-One-Year-Out (LOYO): 评估时间泛化能力
- Leave-One-Glacier-Out (LOGO): 评估空间泛化能力
- 特征重要性分析
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from config import (TRAINING_DATA_CSV, MODEL_RESULTS_DIR, FEATURES, TARGET, LGB_PARAMS,
                    TRAIN_YEAR_MIN, TRAIN_YEAR_MAX)

os.makedirs(MODEL_RESULTS_DIR, exist_ok=True)

# ================= 1. 加载与清理训练数据 =================
print(">>> 1. 加载训练数据...")
df = pd.read_csv(TRAINING_DATA_CSV)
df.columns = [c.replace('.1', '') for c in df.columns]
df = df.loc[:, ~df.columns.duplicated()]

print(f"   原始行数: {len(df)}, 冰川数: {df['WGMS_ID'].nunique()}")

# 修复 TAG=9999 的海拔值
print(">>> 2. 修复 TAG=9999 海拔...")
df_bands = df[df['TAG'] != 9999]
if not df_bands.empty:
    glacier_bounds = df_bands.groupby('WGMS_ID').agg({
        'LOWER_BOUND': 'min',
        'UPPER_BOUND': 'max'
    }).reset_index().rename(columns={
        'LOWER_BOUND': 'REAL_LOWER',
        'UPPER_BOUND': 'REAL_UPPER'
    })
    df = pd.merge(df, glacier_bounds, on='WGMS_ID', how='left')
    mask_9999 = df['TAG'] == 9999
    df.loc[mask_9999, 'LOWER_BOUND'] = df.loc[mask_9999, 'REAL_LOWER'].fillna(df.loc[mask_9999, 'LOWER_BOUND'])
    df.loc[mask_9999, 'UPPER_BOUND'] = df.loc[mask_9999, 'REAL_UPPER'].fillna(df.loc[mask_9999, 'UPPER_BOUND'])
    df.drop(columns=['REAL_LOWER', 'REAL_UPPER'], inplace=True)
    print(f"   已修复 {mask_9999.sum()} 条 TAG=9999 记录的海拔")

# 对齐特征
valid_features = [c for c in FEATURES if c in df.columns]
missing = set(FEATURES) - set(valid_features)
if missing:
    print(f"   WARNING: 缺失特征: {missing}")
print(f"   使用特征数: {len(valid_features)}")

# LightGBM 原生支持特征 NaN，只需保证 TARGET 无 NaN
df = df.dropna(subset=[TARGET])
print(f"   清洗后行数: {len(df)}")

# 年份过滤
df = df[(df['YEAR'] >= TRAIN_YEAR_MIN) & (df['YEAR'] <= TRAIN_YEAR_MAX)].copy()
print(f"   年份筛选 ({TRAIN_YEAR_MIN}-{TRAIN_YEAR_MAX}) 后行数: {len(df)}")


def compute_metrics(y_true, y_pred):
    """计算回归评估指标 (输入单位: m w.e.)"""
    if len(y_true) < 2:
        return {'r2': np.nan, 'r': np.nan, 'rmse': np.nan, 'mae': np.nan, 'bias': np.nan}
    return {
        'r2': r2_score(y_true, y_pred),
        'r': np.corrcoef(y_true, y_pred)[0, 1],
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'bias': np.mean(y_pred - y_true),
    }


# ================= 3. LOYO 验证 =================
print("\n" + "=" * 60)
print(">>> 3. Leave-One-Year-Out 交叉验证 (LOYO) - LightGBM")
print("=" * 60)

loyo_summary = []
loyo_preds = []

min_year, max_year = int(df['YEAR'].min()), int(df['YEAR'].max())
for test_year in range(min_year, max_year + 1):
    train = df[df['YEAR'] != test_year]
    test = df[(df['YEAR'] == test_year) & (df['TAG'] == 9999)]

    if len(test) == 0:
        continue

    model = LGBMRegressor(**LGB_PARAMS)
    model.fit(train[valid_features], train[TARGET])
    y_pred = model.predict(test[valid_features])

    y_test_m = test[TARGET].values / 1000
    y_pred_m = y_pred / 1000

    metrics = compute_metrics(y_test_m, y_pred_m)
    metrics['year'] = test_year
    metrics['n_samples'] = len(test)
    loyo_summary.append(metrics)

    for yt, yp in zip(y_test_m, y_pred_m):
        loyo_preds.append({'year': test_year, 'y_test': yt, 'y_pred': yp})

# 汇总 LOYO
df_loyo_summary = pd.DataFrame(loyo_summary)
df_loyo_preds = pd.DataFrame(loyo_preds)

y_all_true = df_loyo_preds['y_test'].values
y_all_pred = df_loyo_preds['y_pred'].values
loyo_overall = compute_metrics(y_all_true, y_all_pred)

print(f"\n   LOYO 整体结果 ({len(df_loyo_preds)} 个预测):")
print(f"   R2   = {loyo_overall['r2']:.4f}")
print(f"   R    = {loyo_overall['r']:.4f}")
print(f"   RMSE = {loyo_overall['rmse']:.4f} m w.e.")
print(f"   MAE  = {loyo_overall['mae']:.4f} m w.e.")
print(f"   Bias = {loyo_overall['bias']:.4f} m w.e.")

# ================= 4. LOGO 验证 =================
print("\n" + "=" * 60)
print(">>> 4. Leave-One-Glacier-Out 交叉验证 (LOGO) - LightGBM")
print("=" * 60)

logo_summary = []
logo_preds = []

unique_glaciers = df['WGMS_ID'].unique()
print(f"   总冰川数: {len(unique_glaciers)}")

for glacier_id in unique_glaciers:
    train = df[df['WGMS_ID'] != glacier_id]
    test = df[(df['WGMS_ID'] == glacier_id) & (df['TAG'] == 9999)]

    if len(test) == 0:
        continue

    model = LGBMRegressor(**LGB_PARAMS)
    model.fit(train[valid_features], train[TARGET])
    y_pred = model.predict(test[valid_features])

    y_test_m = test[TARGET].values / 1000
    y_pred_m = y_pred / 1000

    metrics = compute_metrics(y_test_m, y_pred_m)
    name_vals = df[df['WGMS_ID'] == glacier_id]['NAME'].dropna().unique()
    glacier_name = name_vals[0] if len(name_vals) > 0 else ''
    metrics['WGMS_ID'] = glacier_id
    metrics['NAME'] = glacier_name
    metrics['n_samples'] = len(test)
    logo_summary.append(metrics)

    for yt, yp in zip(y_test_m, y_pred_m):
        logo_preds.append({'WGMS_ID': glacier_id, 'y_test': yt, 'y_pred': yp})

# 汇总 LOGO
df_logo_summary = pd.DataFrame(logo_summary)
df_logo_preds = pd.DataFrame(logo_preds)

y_all_true_logo = df_logo_preds['y_test'].values
y_all_pred_logo = df_logo_preds['y_pred'].values
logo_overall = compute_metrics(y_all_true_logo, y_all_pred_logo)

print(f"\n   LOGO 整体结果 ({len(df_logo_preds)} 个预测):")
print(f"   R2   = {logo_overall['r2']:.4f}")
print(f"   R    = {logo_overall['r']:.4f}")
print(f"   RMSE = {logo_overall['rmse']:.4f} m w.e.")
print(f"   MAE  = {logo_overall['mae']:.4f} m w.e.")
print(f"   Bias = {logo_overall['bias']:.4f} m w.e.")

# 打印逐冰川 LOGO 结果
print("\n   逐冰川 LOGO 结果:")
print(f"   {'WGMS_ID':>8}  {'Name':<25}  {'R2':>7}  {'RMSE':>7}  {'n':>4}")
print("   " + "-" * 60)
for _, row in df_logo_summary.sort_values('r2', ascending=False).iterrows():
    name_display = str(row['NAME'])[:25] if pd.notnull(row['NAME']) else ''
    print(f"   {int(row['WGMS_ID']):>8}  {name_display:<25}  {row['r2']:>7.3f}  {row['rmse']:>7.3f}  {int(row['n_samples']):>4}")

# ================= 5. 特征重要性 =================
print("\n" + "=" * 60)
print(">>> 5. 全局特征重要性 - LightGBM")
print("=" * 60)

model_full = LGBMRegressor(**LGB_PARAMS)
model_full.fit(df[valid_features], df[TARGET])

importances = model_full.feature_importances_.astype(float)
importances = importances / importances.sum()  # 归一化
indices = np.argsort(importances)[::-1]

feat_imp_data = []
print(f"\n   {'排名':>4}  {'特征':<45}  {'重要性':>8}")
print("   " + "-" * 62)
for i in range(len(valid_features)):
    idx = indices[i]
    feat_name = valid_features[idx]
    imp_val = importances[idx]
    print(f"   {i+1:>4}  {feat_name:<45}  {imp_val:>8.4f}")
    feat_imp_data.append({'rank': i + 1, 'feature': feat_name, 'importance': imp_val})

# ================= 6. 保存结果 =================
print("\n>>> 6. 保存结果...")
df_loyo_summary.to_csv(os.path.join(MODEL_RESULTS_DIR, "lgb_loyo_summary.csv"), index=False)
df_loyo_preds.to_csv(os.path.join(MODEL_RESULTS_DIR, "lgb_loyo_predictions.csv"), index=False)
df_logo_summary.to_csv(os.path.join(MODEL_RESULTS_DIR, "lgb_logo_summary.csv"), index=False)
df_logo_preds.to_csv(os.path.join(MODEL_RESULTS_DIR, "lgb_logo_predictions.csv"), index=False)
pd.DataFrame(feat_imp_data).to_csv(os.path.join(MODEL_RESULTS_DIR, "lgb_feature_importance.csv"), index=False)

print(f"   已保存到: {MODEL_RESULTS_DIR}")
print("\n>>> LightGBM 训练完成!")
