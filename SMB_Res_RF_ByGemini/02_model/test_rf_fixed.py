import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import gaussian_kde  # 新增：用于计算散点密度
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import STUDY_TEST_DIR

# ================= 1. 设置文件路径 =================
# ✅ 使用刚刚修复过海拔的文件 (Fixed)
csv_file = os.path.join(STUDY_TEST_DIR, 'data_glacier_era5_fixed.csv')
print(f">>> 读取数据: {csv_file}")
df = pd.read_csv(csv_file)

# 列名清洗
df.columns = [c.replace('.1', '') for c in df.columns]
df = df.loc[:, ~df.columns.duplicated()]

# 修改 锁定时间窗口 (1980-2024)
print(">>> 正在筛选年份 (1980 - 2024)...")
df = df[(df['YEAR'] >= 1980) & (df['YEAR'] <= 2024)].copy()

# ================= 2. 定义特征列  =================
features_columns = [
    # --- 1. 几何与位置 ---
    "LOWER_BOUND", "UPPER_BOUND", "AREA", "LATITUDE", "LONGITUDE",
    
    # --- 2. 气温与表面状态 ---
    "temperature_2m_year", "temperature_2m_summer",
    "skin_temperature_year", "skin_temperature_summer", 
    "dewpoint_temperature_2m_summer",
    
    # --- 3. 降水与积雪 ---
    "total_precipitation_sum_year", "total_precipitation_sum_summer",
    "snowfall_sum_year", "snowfall_sum_summer",
    "snow_depth_year", "snow_depth_summer", 
    "snow_density_summer", "snow_albedo_summer",

    # --- 4. 能量平衡 ---
    "surface_net_solar_radiation_sum_summer", 
    "surface_net_thermal_radiation_sum_summer",
    "surface_solar_radiation_downwards_sum_summer",
    "surface_thermal_radiation_downwards_sum_summer",

    # --- 5. 湍流与热通量 ---
    "surface_sensible_heat_flux_sum_summer",
    "surface_latent_heat_flux_sum_summer",
    
    # --- 6. 水分收支 ---
    "total_evaporation_sum_year", "total_evaporation_sum_summer",
    "snow_evaporation_sum_summer", 
    "runoff_sum_summer", 
    "snowmelt_sum_year", "snowmelt_sum_summer",

    # --- 7. 时间 ---
    "YEAR"
]

# 自动对齐
valid_features = [c for c in features_columns if c in df.columns]
target_column = 'ANNUAL_BALANCE'
print(f"使用的特征 ({len(valid_features)}个): {valid_features}")

# ================= 3. LOOCV 循环 =================
results1 = {'year': [], 'y_test': [], 'y_pred': []}

min_year = df['YEAR'].min()
max_year = df['YEAR'].max()
print(f"\n>>> 开始按年循环验证 ({min_year} - {max_year})...")

for test_year in range(min_year, max_year + 1):
    # 训练集: 不包含 test_year 的所有数据 (含分带)
    train_data = df[df['YEAR'] != test_year]
    # 测试集: 仅包含 test_year 且 TAG=9999
    test_data = df[(df['YEAR'] == test_year) & (df['TAG'] == 9999)]

    X_train = train_data[valid_features]
    y_train = train_data[target_column]
    X_test = test_data[valid_features]
    y_test = test_data[target_column]

    if len(X_test) > 0:
        model = RandomForestRegressor(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # 假设数据单位是 mm，转为 m
        y_pred_m = y_pred / 1000
        y_test_m = y_test / 1000

        results1['year'].extend([test_year] * len(y_test))
        results1['y_test'].extend(y_test_m)
        results1['y_pred'].extend(y_pred_m)

# ================= 4. 最终评估 =================

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


# ================= 6. 绘制论文同款核密度散点图 =================
print("\n>>> 正在绘制时间交叉验证 (LOOCV) 的密度散点图...")
# 计算点密度
xy = np.vstack([y_true_all, y_pred_all])
z = gaussian_kde(xy)(xy)

# 对点按密度排序，以便将最密集的点画在最上面
idx = z.argsort()
x_plot, y_plot, z_plot = y_true_all[idx], y_pred_all[idx], z[idx]

fig, ax = plt.subplots(figsize=(8, 7))

# 绘制密度散点图 (使用 plasma 配色，类似于文献中的紫->黄渐变)
scatter = ax.scatter(x_plot, y_plot, c=z_plot, s=40, cmap='plasma', alpha=0.7, edgecolors='none')

# 绘制 1:1 对角线 (黑色虚线)
min_val = min(np.min(x_plot), np.min(y_plot))
max_val = max(np.max(x_plot), np.max(y_plot))
ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)

# 设置标签和字体大小
ax.set_xlabel('Ground truth SMB (m w.e. $a^{-1}$)', fontsize=14)
ax.set_ylabel('Predicted SMB (m w.e. $a^{-1}$)', fontsize=14)
ax.set_title('LOOCV: Predicted vs Ground Truth', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=12)

# 添加 Colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Density', fontsize=12)

# 在图表左上角添加指标信息框
textstr = '\n'.join((
    f'R = {r_final:.2f}',
    f'RMSE = {rmse_final:.2f}',
    f'MAE = {mae_final:.2f}',
    f'Bias = {bias_final:.2f}'
))
# 调整文本框位置 (0.05, 0.95 代表左上角)
props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.show()