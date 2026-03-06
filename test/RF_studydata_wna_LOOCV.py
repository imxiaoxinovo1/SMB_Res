import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ================= 1. 读取与筛选 =================
print(">>> 1. 读取数据...")
csv_path = r"H:\Code\SMB\test\study_data_wna.csv"
df = pd.read_csv(csv_path)

# ⚠️ 关键修正：对齐师姐的测试标准
print(f"   原始样本数: {len(df)}")
if 'TAG' in df.columns:
    df = df[df['TAG'] == 9999]
    print(f"   筛选 TAG=9999 后样本数: {len(df)} (这才是师姐用的数据集)")
else:
    print("   ❌ 警告：未找到 TAG 列，无法复刻筛选！")

target_col = 'ANNUAL_BALANCE'

# 定义特征 (排除列表)
exclude_cols = [
    target_col, 'WGMS_ID', 'NAME', 'POLITICAL_UNIT', 'gtng_region', 
    'country_x', 'country_y', 'outline_id', 'glims_id', 
    'rgi50_ids', 'rgi60_ids', 'rgi70_ids', 'wgi_id', 'parent_glacier_id',
    'references_x', 'remarks_x', 'references_y', 'remarks_y',
    'time_system', 'begin_date', 'end_date', 'midseason_date',
    'o1region', 'row', 'col', 'TAG',
    # 'YEAR' # 这里我们保留 YEAR，完全复刻她的条件
]

feature_candidates = [
    c for c in df.columns 
    if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])
]

# 清洗
X_raw = df[feature_candidates].dropna(axis=1, how='all')
X_raw = X_raw.loc[:, (X_raw != X_raw.iloc[0]).any()] 
feature_candidates = X_raw.columns.tolist()

valid_rows = df[target_col].notna()
X = X_raw[valid_rows].fillna(X_raw.mean())
y = df.loc[valid_rows, target_col]
years = df.loc[valid_rows, 'YEAR']

print(f"   最终建模特征池: {len(feature_candidates)} 个 (包含 YEAR)")

# ================= 2. RFECV 特征筛选 =================
print("\n>>> 2. 运行 RFECV (在 TAG=9999 数据集上筛选)...")

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rfecv = RFECV(
    estimator=rf, 
    step=1, 
    cv=KFold(5, shuffle=True, random_state=42), 
    scoring='neg_mean_squared_error', 
    min_features_to_select=10, 
    n_jobs=-1
)

rfecv.fit(X, y)

selected_features = np.array(feature_candidates)[rfecv.support_]
print(f"✅ 最佳特征数量: {rfecv.n_features_}")
print(f"✅ YEAR 是否入选: {'YEAR' in selected_features}")

# ================= 3. LOOCV 验证 =================
print("\n>>> 3. 开始留一年验证 (LOOCV)...")

results = {'y_true': [], 'y_pred': [], 'year': []}
unique_years = sorted(years.unique())

for test_year in unique_years:
    train_mask = (years != test_year)
    test_mask = (years == test_year)
    
    if test_mask.sum() < 1: continue # 允许哪怕只有1个样本也测
        
    X_train = X.loc[train_mask, selected_features]
    y_train = y[train_mask]
    X_test = X.loc[test_mask, selected_features]
    y_test = y[test_mask]
    
    rf_final = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf_final.fit(X_train, y_train)
    
    curr_pred = rf_final.predict(X_test)
    
    results['y_true'].extend(y_test)
    results['y_pred'].extend(curr_pred)
    results['year'].extend([test_year]*len(y_test))
    
    print(f"    年份 {test_year}: 完成 (样本数 {len(y_test)})")

# ================= 4. 评估 =================
y_true = np.array(results['y_true'])
y_pred = np.array(results['y_pred'])

# 单位换算 mm -> m
y_true_m = y_true / 1000
y_pred_m = y_pred / 1000

r2 = r2_score(y_true_m, y_pred_m)
rmse = np.sqrt(mean_squared_error(y_true_m, y_pred_m))
mae = mean_absolute_error(y_true_m, y_pred_m)

print("-" * 50)
print(f"🏆 修正后最终成绩单 (TAG=9999, LOOCV)")
print("-" * 50)
print(f"R²   : {r2:.3f} (预期接近 0.55)")
print(f"RMSE : {rmse:.3f} m w.e. (预期接近 0.66)")
print(f"MAE  : {mae:.3f} m w.e.")
print("-" * 50)