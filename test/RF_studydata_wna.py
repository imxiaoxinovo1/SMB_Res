import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ================= 1. 设置与读取 =================
csv_file = 'H:/Code/SMB/test/study_data_wna.csv'
df = pd.read_csv(csv_file)

# 你的 RFECV 筛选出的 Top 20 特征 (这是你的武器!)
features_columns = [
    'LOWER_BOUND', 'UPPER_BOUND', 'AREA', 'snowmelt_sum_year', 
    'evaporation_from_open_water_surfaces_excluding_oceans_sum_summer',
    'skin_reservoir_content_summer', 'snow_cover_summer', 
    'snowfall_sum_year', 'snow_density_year', 'LONGITUDE', 
    'evaporation_from_open_water_surfaces_excluding_oceans_sum_year',
    'snow_density', 'surface_runoff_sum_year', 'runoff_sum_year', 'PBLH',
    'surface_sensible_heat_flux_sum_summer', 'evaporation_from_the_top_of_canopy_sum_summer',
    'snowmelt_sum', 'sub_surface_runoff_sum_year', 'YEAR'
]
# 注意：有些列名可能需要微调（比如 .1 后缀），这里用标准名，如果有报错需检查列名

target_column = 'ANNUAL_BALANCE'

# 清洗数据
df = df.dropna(subset=features_columns + [target_column])

# ================= 2. 严酷测试：Leave-One-Year-Out Cross Validation =================
results1 = {
    'year': [],
    'y_test': [],
    'y_pred': [],
}

print(f">>> 开始按年交叉验证 (1980-2020)...")

for test_year in range(1980, 2021):
    # 分割：用除 test_year 以外的所有年份训练，预测 test_year
    train_data = df[df['YEAR'] != test_year]
    # 师姐代码里有这个 TAG==9999 的筛选，我们也加上以保持一致
    test_data = df[(df['YEAR'] == test_year) & (df['TAG'] == 9999)]

    if len(test_data) == 0:
        continue

    X_train = train_data[features_columns]
    y_train = train_data[target_column]
    X_test = test_data[features_columns]
    y_test = test_data[target_column]

    # 训练
    model = RandomForestRegressor(n_estimators=200, max_depth=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)
    
    # 单位换算 (mm -> m)
    y_pred = y_pred / 1000
    y_test = y_test / 1000

    # 存储结果
    results1['year'].extend([test_year] * len(y_test))
    results1['y_test'].extend(y_test)
    results1['y_pred'].extend(y_pred)
    
    print(f"   年份 {test_year}: 完成预测 (样本数 {len(y_test)})")

# ================= 3. 整体评估 =================
results_df1 = pd.DataFrame(results1)

r_squared = r2_score(results_df1['y_test'], results_df1['y_pred'])
r = np.corrcoef(results_df1['y_test'], results_df1['y_pred'])[0, 1]
rmse = np.sqrt(mean_squared_error(results_df1['y_test'], results_df1['y_pred']))
mae = mean_absolute_error(results_df1['y_test'], results_df1['y_pred'])
bias = np.mean(results_df1['y_pred'] - results_df1['y_test'])

print("\n" + "="*40)
print(" RF 整体评价指标 (Leave-One-Year-Out)")
print("="*40)
print(f" R²   : {r_squared:.3f} (师姐: 0.549)")
print(f" R    : {r:.3f} (师姐: 0.748)")
print(f" RMSE : {rmse:.3f} m w.e. (师姐: 0.669)")
print(f" MAE  : {mae:.3f} m w.e.")
print(f" Bias : {bias:.3f}")
print("="*40)