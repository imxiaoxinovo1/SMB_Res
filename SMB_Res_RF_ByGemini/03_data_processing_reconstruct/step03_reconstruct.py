import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import STUDY_TEST_DIR, RECONSTRUCTION_DIR

# ================= 1. 配置路径 =================
# 路径由 config.py 统一管理
train_csv_path = os.path.join(STUDY_TEST_DIR, 'data_glacier_era5_matched.csv')
# 预测输入数据 (包含错误面积的大表)
predict_csv_path = os.path.join(RECONSTRUCTION_DIR, 'rgi02_glaciers_era5.csv')
# 结果输出路径
output_dir = RECONSTRUCTION_DIR
output_csv_path = os.path.join(output_dir, "RGI02_reconstruction_corrected.csv")

print(">>> 1. 准备训练...")
df_train = pd.read_csv(train_csv_path)

# 清洗列名
df_train.columns = [c.replace('.1', '') for c in df_train.columns]
df_train = df_train.loc[:, ~df_train.columns.duplicated()]

features = [
    "LOWER_BOUND", "UPPER_BOUND", "AREA", "LATITUDE", "LONGITUDE",
    "temperature_2m_year", "temperature_2m_summer",
    "total_precipitation_sum_year", "total_precipitation_sum_summer",
    "snowmelt_sum_year", "snowmelt_sum_summer",
    "snowfall_sum_year",
    "surface_net_solar_radiation_sum_summer",
    "surface_net_thermal_radiation_sum_summer",
    "surface_sensible_heat_flux_sum_summer",
    "surface_latent_heat_flux_sum_summer",
    "YEAR"
]
target = "ANNUAL_BALANCE"

# 训练模型
df_train_clean = df_train.dropna(subset=features + [target])
X_train = df_train_clean[features]
y_train = df_train_clean[target]

rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
print(f"   模型训练完毕 (训练集样本数: {len(X_train)})")

# ================= 2. 修正预测数据 =================
print("\n>>> 2. 读取并修正预测数据...")
df_pred = pd.read_csv(predict_csv_path)
print(f"   原始样本数: {len(df_pred)}")

# 🔧 修正1：面积单位换算 (m2 -> km2)
# 判断标准：如果面积均值极其巨大，说明是 m2
if df_pred['AREA'].mean() > 10000:
    print("   ⚠️ 检测到 AREA 单位为 m²，正在转换为 km² (除以 1,000,000)...")
    df_pred['AREA'] = df_pred['AREA'] / 1_000_000
else:
    print("   ✅ AREA 单位看起来正常 (km²)")

# 🔧 修正2：纬度筛选 (仅保留 RGI 02 区域)
# RGI 02 (Western Canada/US) 通常定义为 < 60N (育空/阿拉斯加边界以南)
# 60N 以上通常属于 RGI 03/04 (北极群岛) 或 RGI 01 (阿拉斯加)
print("   ⚠️ 正在剔除北纬 60° 以上的冰川 (排除北极群岛，锁定 RGI 02)...")
df_pred_rgi02 = df_pred[df_pred['LATITUDE'] < 60].copy()
print(f"   筛选后样本数: {len(df_pred_rgi02)} (剔除了 {len(df_pred) - len(df_pred_rgi02)} 行)")

# 剔除空值
df_pred_clean = df_pred_rgi02.dropna(subset=features)

# ================= 3. 执行预测 =================
print("\n>>> 3. 开始预测...")
X_pred = df_pred_clean[features]
pred_mm = rf.predict(X_pred)

# 结果处理
df_pred_clean['Predicted_SMB_m'] = pred_mm / 1000

# 保存
out_cols = ['WGMS_ID', 'NAME', 'POLITICAL_UNIT', 'YEAR', 'LATITUDE', 'LONGITUDE', 'AREA', 'Predicted_SMB_m']
df_final = df_pred_clean[[c for c in out_cols if c in df_pred_clean.columns]]

df_final.to_csv(output_csv_path, index=False)

print("-" * 40)
print(f"🏆 修正版重建完成！")
print(f"   输出文件: {output_csv_path}")
print(f"   预测均值: {df_final['Predicted_SMB_m'].mean():.3f} m w.e.")
print(f"   (合理的范围通常在 -1.5 到 +0.5 之间)")
print("-" * 40)
print(df_final.head())