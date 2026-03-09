import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import STUDY_TEST_DIR









#这个代码最终能够复现Figure8中的LOOCV结果和特征重要性结果






# ================= 1. 设置文件路径 =================
# 路径由 config.py 统一管理
csv_file = os.path.join(STUDY_TEST_DIR, 'data_glacier_era5_matched.csv')
output_summary = os.path.join(STUDY_TEST_DIR, 'result_summary.csv')
output_detail = os.path.join(STUDY_TEST_DIR, 'result_detail.csv')

print(f">>> 读取数据: {csv_file}")
df = pd.read_csv(csv_file)

# 简单的列名清洗 (去重)
df.columns = [c.replace('.1', '') for c in df.columns]
df = df.loc[:, ~df.columns.duplicated()]

# 去除缺失值
df = df.dropna()

# ================= 2. 定义特征列 =================
features_columns = [
    # --- 几何特征 ---
    "LOWER_BOUND", "UPPER_BOUND", "AREA", "LATITUDE", "LONGITUDE",
    
    # --- ERA5 气象特征 ---
    "temperature_2m_year", "temperature_2m_summer",
    "total_precipitation_sum_year", "total_precipitation_sum_summer",
    "snowmelt_sum_year", "snowmelt_sum_summer",
    "snowfall_sum_year",
    "surface_net_solar_radiation_sum_summer",
    "surface_net_thermal_radiation_sum_summer",
    "surface_sensible_heat_flux_sum_summer",
    "surface_latent_heat_flux_sum_summer",
    
    # 年份
    "YEAR" 
]

# 自动对齐 (防止列名报错)
features_columns = [c for c in features_columns if c in df.columns]
target_column = 'ANNUAL_BALANCE'

print(f"使用的特征 ({len(features_columns)}个): {features_columns}")

# ================= 3. LOOCV 循环 (验证模型精度) =================
results = {
    'year': [], 'test_samples': [], 'r_squared': [],
    'r': [], 'rmse': [], 'mae': [], 'bias': []
}
results1 = {'year': [], 'y_test': [], 'y_pred': []}

min_year = df['YEAR'].min()
max_year = df['YEAR'].max()
print(f"\n>>> 开始按年循环验证 ({min_year} - {max_year})...")
print("    (训练集: 含分带数据 | 测试集: 仅全冰川 TAG=9999)")

for test_year in range(min_year, max_year + 1):
    # 1. 划分训练集和测试集
    # 训练集: 不包含 test_year 的 *所有* 数据 (包含分带数据，这让模型学到了海拔规律！)
    train_data = df[df['YEAR'] != test_year]
    
    # 测试集: 仅包含 test_year 且 TAG=9999 (我们要预测的目标)
    test_data = df[(df['YEAR'] == test_year) & (df['TAG'] == 9999)]

    X_train = train_data[features_columns]
    y_train = train_data[target_column]
    X_test = test_data[features_columns]
    y_test = test_data[target_column]

    if len(X_test) > 1:
        # 2. 训练模型
        model = RandomForestRegressor(n_estimators=200, max_depth=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        # 3. 预测
        y_pred = model.predict(X_test)
        
        # 4. 单位换算 (mm -> m)
        # 假设你的数据是 mm (之前脚本清洗过)，为了看 m w.e. 的结果
        y_pred_m = y_pred / 1000
        y_test_m = y_test / 1000

        # 5. 计算指标
        r_squared = r2_score(y_test_m, y_pred_m)
        r = np.corrcoef(y_test_m, y_pred_m)[0, 1] if len(y_test_m) > 1 else 0
        rmse = np.sqrt(mean_squared_error(y_test_m, y_pred_m))
        mae = mean_absolute_error(y_test_m, y_pred_m)
        bias = np.mean(y_pred_m - y_test_m)

        # 存储
        results['year'].append(test_year)
        results['test_samples'].append(len(y_test))
        results['r_squared'].append(r_squared)
        results['r'].append(r)
        results['rmse'].append(rmse)
        results['mae'].append(mae)
        results['bias'].append(bias)

        results1['year'].extend([test_year] * len(y_test))
        results1['y_test'].extend(y_test_m)
        results1['y_pred'].extend(y_pred_m)

# ================= 4. 保存与评估 =================
pd.DataFrame(results).to_csv(output_summary, index=False)
pd.DataFrame(results1).to_csv(output_detail, index=False)

# 计算整体指标
y_true_all = np.array(results1['y_test'])
y_pred_all = np.array(results1['y_pred'])

if len(y_true_all) > 0:
    r2_final = r2_score(y_true_all, y_pred_all)
    rmse_final = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
    mae_final = mean_absolute_error(y_true_all, y_pred_all)
    #新增Pearson R 和 Bias
    r_final = np.corrcoef(y_true_all, y_pred_all)[0, 1]
    bias_final = np.mean(y_pred_all - y_true_all)
    
    print("\n" + "="*50)
    print("🏆 最终验证结果 (LOOCV)")
    print("="*50)
    print(f"R²   : {r2_final:.3f}")
    print(f"R    : {r_final:.3f}")       # 新增
    print(f"RMSE : {rmse_final:.3f} m w.e.")
    print(f"MAE  : {mae_final:.3f} m w.e.")
    print(f"Bias : {bias_final:.3f} m w.e.") # 新增
    print("="*50)

# ================= 5. 🔥 特征重要性分析 (Corrected) =================
print("\n>>> 计算全局特征重要性...")
print("    (使用全部数据训练，确保捕捉海拔效应)")

# ✅ 关键修正：使用整个 df (包含分带数据) 来计算重要性
X_final = df[features_columns]
y_final = df[target_column]

model_final = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model_final.fit(X_final, y_final)

importances = model_final.feature_importances_
indices = np.argsort(importances)[::-1]

print("\n📊 特征重要性排名 (Top 15):")
print("-" * 40)
for i in range(min(15, len(features_columns))):
    idx = indices[i]
    print(f"{i+1:2d}. {features_columns[idx]:<40} : {importances[idx]:.4f}")
print("-" * 40)

# 绘图
plt.figure(figsize=(10, 6))
top_n = min(15, len(features_columns))
sorted_feats = [features_columns[i] for i in indices[:top_n]]
sorted_imps = importances[indices[:top_n]]

# 颜色映射：把海拔标红
colors = ['red' if 'BOUND' in f else 'skyblue' for f in sorted_feats]

sns.barplot(x=sorted_imps, y=sorted_feats, hue=sorted_feats, palette=colors, legend=False)
plt.title("Feature Importance (Corrected: Trained on All Bands)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()