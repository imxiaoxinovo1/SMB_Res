import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.metrics import r2_score, mean_squared_error

# ================= 1. 读取主数据 =================
print(">>> 1. 读取主数据 (Meteo + Balance)...")
main_csv_path = r"H:\Code\SMB\test\result\integrated_glacier_data_v2_FIXED.csv"
df_main = pd.read_csv(main_csv_path)

# ================= 2. 读取并处理 state.csv (提取海拔) =================
print(">>> 2. 处理 state.csv 提取海拔特征...")
# 请确保路径正确，指向你的 WGMS state.csv
state_csv_path = r"H:\Code\SMB\WGMS\FoG_DataBase\DOI-WGMS-FoG-2025-02b\data\state.csv"

# 读取时忽略低内存警告
df_state = pd.read_csv(state_csv_path, encoding='utf-8', low_memory=False)

# 统一 ID 列名
if 'glacier_id' in df_state.columns:
    df_state.rename(columns={'glacier_id': 'WGMS_ID'}, inplace=True)

# 筛选需要的列
geo_cols = ['WGMS_ID', 'highest_elevation', 'lowest_elevation', 'mean_elevation']
# 确保列存在
valid_geo_cols = [c for c in geo_cols if c in df_state.columns]
df_state = df_state[valid_geo_cols]

# 【关键步骤】: 聚合 (Aggregation)
# 因为一个冰川可能有多次测量记录，我们取平均值作为该冰川的"几何特征"
# 这样保证每个 WGMS_ID 只有一行数据，方便合并
df_geo_unique = df_state.groupby('WGMS_ID').mean().reset_index()

print(f"   提取到 {len(df_geo_unique)} 个冰川的海拔几何信息。")

# ================= 3. 数据合并 (Merge) =================
print(">>> 3. 合并数据...")
# 将海拔信息合并到主表
df_final = pd.merge(df_main, df_geo_unique, on='WGMS_ID', how='left')

# 清洗：删除没有物质平衡观测的行
df_clean = df_final.dropna(subset=['ANNUAL_BALANCE'])

# 填充几何数据的空值
# 如果某些冰川在 state.csv 里没记录，用整体均值填充，防止报错
for col in ['highest_elevation', 'lowest_elevation', 'mean_elevation']:
    if col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
            print(f"   已填充 {col} 的缺失值。")

# ================= 4. 构建特征池 =================
# 1. 气候特征
climate_features = [col for col in df_clean.columns if col.startswith('ERA5_')]

# 2. 几何特征 (核心!)
# 师姐最重要的特征: LOWER_BOUND (对应 lowest_elevation), UPPER_BOUND (对应 highest_elevation)
geometry_features = ['lowest_elevation', 'highest_elevation', 'mean_elevation', 'LATITUDE']
# 确保这些列都在
geometry_features = [c for c in geometry_features if c in df_clean.columns]

# 3. 时间特征 (作为对比)
time_feature = ['YEAR']

all_features = geometry_features + climate_features + time_feature
print(f"\n特征池 ({len(all_features)}个): {all_features}")

X = df_clean[all_features]
y = df_clean['ANNUAL_BALANCE']

# ================= 5. RFECV 特征筛选与建模 =================
print("\n>>> 4. 运行 RFECV (自动特征筛选)...")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

# 自动筛选
rfecv = RFECV(
    estimator=rf,
    step=1,
    cv=KFold(5),
    scoring='neg_mean_squared_error',
    min_features_to_select=10, # 至少保留10个
    n_jobs=-1
)

rfecv.fit(X_train, y_train)

# 获取最佳特征
selected_features = np.array(all_features)[rfecv.support_]
print(f"\n✅ 最佳特征数量: {rfecv.n_features_}")
print("✅ 最终入选特征 (按原始顺序):")
print(selected_features)

# ================= 6. 最终训练与验证 =================
print("\n>>> 5. 最终模型评估...")
rf_final = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf_final.fit(X_train[selected_features], y_train)
y_pred = rf_final.predict(X_test[selected_features])

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("-" * 30)
print(f"R² : {r2:.3f} (有了海拔加持，能超过0.6吗?)")
print(f"RMSE: {rmse:.3f} m w.e.")
print("-" * 30)

# ================= 7. 关键：特征重要性可视化 =================
plt.figure(figsize=(12, 8))
importances = rf_final.feature_importances_
indices = np.argsort(importances)[::-1]
sorted_features = [selected_features[i] for i in indices]

# 使用不同颜色区分不同类型的变量
colors = []
for feat in sorted_features:
    if 'elevation' in feat:
        colors.append('red')    # 几何特征 (重点关注!)
    elif 'YEAR' in feat:
        colors.append('black')  # 时间
    else:
        colors.append('skyblue') # 气象

sns.barplot(x=importances[indices], y=sorted_features, hue=sorted_features, palette="viridis", legend=False)

# 标注一下
plt.title("Feature Importance: Does Geometry Matter?")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()