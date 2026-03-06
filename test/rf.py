import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


# 设置文件路径

csv_file = 'H:/Code/SMB/test/study_data_wna.csv'

# 读取数据
df = pd.read_csv(csv_file)

# 去除缺失值
df = df.dropna()

# 定义结果列表，用于存储每次循环的评价结果
results = {
    'year': [],
    'test_samples': [],
    'r_squared': [],
    'r': [],
    'rmse': [],
    'mae': [],
    'bias': []
}

results1 = {
    'year': [],
    'y_test': [],
    'y_pred': [],
}
features_columns = [    "LOWER_BOUND",
    "UPPER_BOUND",
    "snowmelt_sum_year",
    "AREA",
    "snow_cover_summer",
    "skin_reservoir_content_summer",
    "evaporation_from_open_water_surfaces_excluding_oceans_sum_summer",
    "runoff_sum_year",
    "LONGITUDE",
    "lake_mix_layer_temperature_summer",
    "snowfall_sum_year",
    "evaporation_from_the_top_of_canopy_sum_summer",
    "surface_runoff_sum_year",
    "evaporation_from_open_water_surfaces_excluding_oceans_sum_year",
    "sub_surface_runoff_sum_year",
    "PBLH",
    "RHOA",
    "PRECSNO",
    "GHTSKIN",
    "YEAR"            ]
target_column = 'ANNUAL_BALANCE'

# 循环遍历每个测试年份
for test_year in range(1980, 2021):
    # 分割训练集和测试集
    train_data = df[df['YEAR'] != test_year]
    test_data = df[(df['YEAR'] == test_year) & (df['TAG'] == 9999)]

    X_train = train_data[features_columns]
    y_train = train_data[target_column]
    X_test = test_data[features_columns]
    y_test = test_data[target_column]

    # 确保测试集中有多组数据
    if len(X_test) > 1:
        # 训练模型
        model = RandomForestRegressor(n_estimators=200, max_depth=100)
        model.fit(X_train, y_train)

        # 进行预测
        y_pred = model.predict(X_test)
        y_pred = y_pred / 1000
        y_test = y_test / 1000

        

        # 计算评价指标
        r_squared = r2_score(y_test, y_pred)
        r = np.corrcoef(y_test, y_pred)[0, 1]
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        bias = np.mean(y_pred - y_test)

        # 将评价指标存储到结果列表中
        results['year'].append(test_year)
        results['test_samples'].append(len(y_test))
        results['r_squared'].append(r_squared)
        results['r'].append(r)
        results['rmse'].append(rmse)
        results['mae'].append(mae)
        results['bias'].append(bias)

        # 将预测结果存储到结果列表中
        results1['year'].extend([test_year] * len(y_test))
        results1['y_test'].extend(y_test)
        results1['y_pred'].extend(y_pred)
# 将结果列表转换为DataFrame
results_df = pd.DataFrame(results)
results_df1 = pd.DataFrame(results1)


# 将结果写入CSV文件
results_df.to_csv('H:/Code/SMB/test/result/all_data_result_time_wna.csv', index=False)
results_df1.to_csv('H:/Code/SMB/test/result/pred_result_time_wna.csv', index=False)
# 计算评价指标
r_squared = r2_score(results_df1['y_test'], results_df1['y_pred'])
r = np.corrcoef(results_df1['y_test'], results_df1['y_pred'])[0, 1]
rmse = np.sqrt(mean_squared_error(results_df1['y_test'], results_df1['y_pred']))
mae = mean_absolute_error(results_df1['y_test'], results_df1['y_pred'])
bias = np.mean(results_df1['y_pred'] - results_df1['y_test'])

# 打印评价指标
print("\nRF整体评价指标:")
print("all_data_result_time R²:", r_squared)
print("all_data_result_time R:", r)
print("all_data_result_time RMSE:", rmse)
print("all_data_result_time MAE:", mae)
print("all_data_result_time Bias:", bias)











