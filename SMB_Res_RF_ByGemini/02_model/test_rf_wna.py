import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import STUDY_TEST_DIR

# ================= 1. 读取数据 =================
# 路径由 config.py 统一管理
csv_file = os.path.join(STUDY_TEST_DIR, 'data_glacier_era5_matched.csv')
# csv_file = r"H:\Code\SMB\test\study_data_wna.csv" # 也可以换回师姐的数据验证

print(f">>> 读取数据: {csv_file}")
df = pd.read_csv(csv_file)

# 这里的去重是为了防止像之前那样出现重复列报错
df.columns = [c.replace('.1', '') for c in df.columns]
df = df.loc[:, ~df.columns.duplicated()]

# ================= 2. 定义特征 =================
features_columns = [
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

# 自动对齐列名（防止师姐数据和你的数据列名微小差异）
valid_features = [c for c in features_columns if c in df.columns]
target_column = 'ANNUAL_BALANCE'

print(f"使用的特征 ({len(valid_features)}个): {valid_features}")

# ================= 3. 计算特征重要性 (关键修正) =================
print("\n>>> 正在计算全局特征重要性...")
print("    注意：这次使用【全部数据】(含分带数据) 进行训练，这才是海拔起作用的原因！")

# ❌ 之前的错误：df_model = df[df['TAG'] == 9999]  <-- 我把带海拔的数据删了
# ✅ 正确的做法：直接用 df (包含 TAG=1800, 1900... 等真实海拔行)
X = df[valid_features]
y = df[target_column]

# 训练模型
model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X, y)

# 提取重要性
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# ================= 4. 打印与画图 =================
print("\n📊 特征重要性排名 (Top 15):")
print("-" * 40)
for i in range(min(15, len(valid_features))):
    idx = indices[i]
    print(f"{i+1:2d}. {valid_features[idx]:<40} : {importances[idx]:.4f}")
print("-" * 40)

# 绘图
plt.figure(figsize=(10, 8))
top_n = min(20, len(valid_features))
sorted_feats = [valid_features[i] for i in indices[:top_n]]
sorted_imps = importances[indices[:top_n]]

# 颜色映射：把海拔标红，看看是不是第一
colors = ['red' if 'BOUND' in f else 'skyblue' for f in sorted_feats]

sns.barplot(x=sorted_imps, y=sorted_feats, hue=sorted_feats, palette=colors, legend=False)
plt.title("Feature Importance (Trained on ALL Bands + Global)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()