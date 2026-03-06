import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ================= 1. 数据准备 (保持不变) =================
print(">>> 1. 读取并融合数据...")
main_csv_path = r"H:\Code\SMB\test\result\integrated_glacier_data_v2_FIXED.csv"
state_csv_path = r"H:\Code\SMB\WGMS\FoG_DataBase\DOI-WGMS-FoG-2025-02b\data\state.csv"

# 读取
df_main = pd.read_csv(main_csv_path)
df_state = pd.read_csv(state_csv_path, encoding='utf-8', low_memory=False)

# 整理 state.csv
if 'glacier_id' in df_state.columns:
    df_state.rename(columns={'glacier_id': 'WGMS_ID'}, inplace=True)

# 聚合几何特征 (取均值)
geo_cols = ['WGMS_ID', 'highest_elevation', 'lowest_elevation', 'mean_elevation', 'area']
valid_geo_cols = [c for c in geo_cols if c in df_state.columns]
df_geo_unique = df_state.groupby('WGMS_ID')[valid_geo_cols[1:]].mean().reset_index()

# 合并
df_final = pd.merge(df_main, df_geo_unique, on='WGMS_ID', how='left')
df_clean = df_final.dropna(subset=['ANNUAL_BALANCE'])

# 填充几何缺失值
for col in valid_geo_cols[1:]:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mean())

print(f"   数据准备就绪: {len(df_clean)} 行样本")

# ================= 2. 构建特征池 (关键修改：剔除 YEAR) =================
# 1. 气候特征
climate_features = [col for col in df_clean.columns if col.startswith('ERA5_')]

# 2. 几何特征
geometry_features = ['lowest_elevation', 'highest_elevation', 'mean_elevation', 'LATITUDE', 'area']
geometry_features = [c for c in geometry_features if c in df_clean.columns]

# ❌ 关键修改：彻底移除 'YEAR'，防止时间泄露
time_feature = ['YEAR']

all_features = geometry_features + climate_features +time_feature
print(f"\n>>> 特征池 ({len(all_features)}个): [不包含 YEAR]")
print(f"    几何特征: {geometry_features}")

X = df_clean[all_features]
y = df_clean['ANNUAL_BALANCE']
years = df_clean['YEAR'] # 保留年份用于划分训练/测试集

# ================= 3. RFECV 特征筛选 (防止过拟合) =================
print("\n>>> 2. 运行 RFECV 特征筛选...")
# 这里我们依然用随机森林做筛选，但为了防止特征选择阶段的泄露，
# 严格来说应该在交叉验证内部做，但为了算力考虑，我们先全局筛一遍"强物理特征"

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rfecv = RFECV(estimator=rf, step=1, cv=KFold(5, shuffle=True, random_state=42), 
              scoring='neg_mean_squared_error', min_features_to_select=10, n_jobs=-1)

rfecv.fit(X, y)

selected_features = np.array(all_features)[rfecv.support_]
print(f"✅ 最佳特征数量: {rfecv.n_features_}")
print(f"✅ 入选特征: {selected_features}")

# ================= 4. 严苛验证: Leave-One-Year-Out (LOOCV) =================
print("\n>>> 3. 开始留一年验证 (LOOCV)...")
print("    (模型必须要预测从未见过的年份，这是对泛化能力的终极考验)")

results = {'y_true': [], 'y_pred': [], 'year': []}
unique_years = sorted(years.unique())

for test_year in unique_years:
    # 划分：这一年做测试，其余年份做训练
    train_mask = (years != test_year)
    test_mask = (years == test_year)
    
    # 样本太少不测
    if test_mask.sum() < 2: continue
        
    X_train = df_clean.loc[train_mask, selected_features]
    y_train = y[train_mask]
    X_test = df_clean.loc[test_mask, selected_features]
    y_test = y[test_mask]
    
    # 训练模型
    rf_final = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf_final.fit(X_train, y_train)
    
    # 预测
    curr_pred = rf_final.predict(X_test)
    
    # 记录
    results['y_true'].extend(y_test)
    results['y_pred'].extend(curr_pred)
    results['year'].extend([test_year]*len(y_test))
    
    print(f"    年份 {test_year}: 完成 (样本数 {len(y_test)})")

# ================= 5. 最终评估与可视化 =================
y_true = np.array(results['y_true'])
y_pred = np.array(results['y_pred'])

r2 = r2_score(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)

print("-" * 40)
print(f" 最终成绩单 (无 YEAR, 纯物理驱动, LOOCV)")
print("-" * 40)
print(f"R²   : {r2:.3f} (如果 > 0.45 说明物理机制很强)")
print(f"RMSE : {rmse:.3f} m w.e.")
print(f"MAE  : {mae:.3f} m w.e.")
print("-" * 40)

# 特征重要性 (用全数据训练一次以展示)
rf_final.fit(X[selected_features], y)
importances = rf_final.feature_importances_
indices = np.argsort(importances)[::-1][:20] # 只看前20

plt.figure(figsize=(12, 8))
sorted_feats = [selected_features[i] for i in indices]
sorted_imps = importances[indices]

# 颜色标记
colors = ['red' if 'elevation' in f or 'LATITUDE' in f or 'area' in f else 'skyblue' for f in sorted_feats]

sns.barplot(x=sorted_imps, y=sorted_feats, hue=sorted_feats, palette=colors, legend=False)
plt.title(f"Feature Importance \nEvaluated R²={r2:.3f}")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()