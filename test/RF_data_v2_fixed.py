import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ================= 1. 读取与特征工程 =================
csv_path = r"H:\Code\SMB\test\result\integrated_glacier_data_v2_FIXED.csv"
df = pd.read_csv(csv_path)

# 清洗：删除没有物质平衡观测值的行
df_clean = df.dropna(subset=['ANNUAL_BALANCE'])

# --- 关键改进 1: 构建更完整的特征池 ---
# 1. 气候特征 (ERA5)
climate_features = [col for col in df_clean.columns if col.startswith('ERA5_')]

# 2. 地理与几何特征 (Geography & Geometry)
# 师姐的研究表明：海拔和面积至关重要！
# 我们检查一下CSV里是否有这些列 (WGMS通常有 area, begin_year等)
geo_candidates = ['LATITUDE', 'LONGITUDE','YEAR',  'area', 'ela']
geo_features = [col for col in geo_candidates if col in df_clean.columns]



all_features = geo_features + climate_features

print(f"原始特征池 ({len(all_features)}个): {all_features}")

# 再次清洗：特征列也不能有空值 (RF不支持空值)
# 用均值填充几何特征的空值 (防止因为缺少几个面积数据丢掉行)
for col in geo_features:
    if df_clean[col].isnull().sum() > 0:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mean())

X = df_clean[all_features]
y = df_clean['ANNUAL_BALANCE']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ================= 2. RFECV: 自动寻找最佳特征数量 =================
# 师姐用的是 RFE 手动测试，我们要用更高级的 RFECV 自动画出那条"准确率波动曲线"

print("\n>>> 正在进行 RFECV 特征筛选 (自动寻找最佳数量)...")
print("    这可能需要几分钟，请耐心等待...")

# 定义基模型
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

# 使用 K-Fold 交叉验证来评估每组特征的表现
min_features = 5 # 最少保留5个
step = 1         # 每次剔除1个
rfecv = RFECV(
    estimator=rf, 
    step=step, 
    cv=KFold(5), 
    scoring='neg_mean_squared_error', # 以 MSE 为评估标准
    min_features_to_select=min_features,
    n_jobs=-1
)

rfecv.fit(X_train, y_train)

print(f"\n✅ 最佳特征数量: {rfecv.n_features_}")

# 获取被选中的特征名
selected_features = np.array(all_features)[rfecv.support_]
print("✅ 最终保留的特征:")
for i, f in enumerate(selected_features, 1):
    print(f"   {i}. {f}")

# ================= 3. 结果可视化 (复刻师姐的图) =================

plt.figure(figsize=(10, 6))
# RFECV.cv_results_['mean_test_score'] 存的是负MSE，我们要取反并开根号看RMSE
n_features_range = range(min_features, min_features + len(rfecv.cv_results_['mean_test_score']))
rmse_scores = np.sqrt(-rfecv.cv_results_['mean_test_score'])

plt.plot(n_features_range, rmse_scores, marker='o', color='b')
plt.axvline(x=rfecv.n_features_, color='r', linestyle='--', label=f'Optimal ({rfecv.n_features_})')
plt.xlabel("Number of Features Selected")
plt.ylabel("Cross-Validation RMSE (m w.e.)")
plt.title("Recursive Feature Elimination with Cross-Validation (RFECV)")
plt.legend()
plt.grid(True)
plt.show()

# ================= 4. 最终模型评估 =================
print("\n>>> 使用最佳特征子集进行最终评估...")

# 重新训练模型 (只用选出来的特征)
rf_final = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf_final.fit(X_train[selected_features], y_train)
y_pred = rf_final.predict(X_test[selected_features])

# 计算指标
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print("-" * 30)
print(f"最终 R² : {r2:.3f}")
print(f"最终 RMSE: {rmse:.3f} m w.e.")
print(f"最终 MAE : {mae:.3f} m w.e.")
print("-" * 30)

# ================= 5. 特征重要性排名 =================
plt.figure(figsize=(12, 8))
importances = rf_final.feature_importances_
indices = np.argsort(importances)[::-1]
sorted_features = [selected_features[i] for i in indices]

sns.barplot(x=importances[indices], y=sorted_features, hue=sorted_features, palette="viridis", legend=False)
plt.title("Feature Importance (Optimal Subset)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()