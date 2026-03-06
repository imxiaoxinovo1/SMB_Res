import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy import stats

# 设置文件路径
csv_file = 'H:/Code/SMB/test/study_data_wna.csv'

# 确保结果文件夹存在
output_dir = 'H:/Code/SMB/test/result'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

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

# 特征列
features_columns = [
    "LOWER_BOUND", "UPPER_BOUND", "snowmelt_sum_year", "AREA", "snow_cover_summer",
    "skin_reservoir_content_summer", "evaporation_from_open_water_surfaces_excluding_oceans_sum_summer",
    "runoff_sum_year", "LONGITUDE", "lake_mix_layer_temperature_summer", "snowfall_sum_year",
    "evaporation_from_the_top_of_canopy_sum_summer", "surface_runoff_sum_year",
    "evaporation_from_open_water_surfaces_excluding_oceans_sum_year", "sub_surface_runoff_sum_year",
    "PBLH", "RHOA", "PRECSNO", "GHTSKIN", "YEAR"
]
target_column = 'ANNUAL_BALANCE'

# 初始化特征标准化器（线性回归对特征尺度敏感）
scaler = StandardScaler()

# 循环遍历每个测试年份
for test_year in range(1980, 2021):
    print(f"正在处理年份: {test_year}")
    
    # 分割训练集和测试集
    train_data = df[df['YEAR'] != test_year]
    test_data = df[(df['YEAR'] == test_year) & (df['TAG'] == 9999)]

    X_train = train_data[features_columns]
    y_train = train_data[target_column]
    X_test = test_data[features_columns]
    y_test = test_data[target_column]

    # 确保测试集中有多组数据
    if len(X_test) > 1:
        # 特征标准化
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 初始化并训练多元线性回归模型
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)

        # 进行预测
        y_pred = model.predict(X_test_scaled)
        
        # 单位转换
        y_pred = y_pred / 1000
        y_test = y_test / 1000

        # 计算评价指标
        r_squared = r2_score(y_test, y_pred)
        r = np.corrcoef(y_test, y_pred)[0, 1]
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        bias = np.mean(y_pred - y_test)

        # 存储评价指标
        results['year'].append(test_year)
        results['test_samples'].append(len(y_test))
        results['r_squared'].append(r_squared)
        results['r'].append(r)
        results['rmse'].append(rmse)
        results['mae'].append(mae)
        results['bias'].append(bias)

        # 存储预测结果
        results1['year'].extend([test_year] * len(y_test))
        results1['y_test'].extend(y_test)
        results1['y_pred'].extend(y_pred)
    else:
        print(f"  测试集样本不足，跳过年份 {test_year}")

# 转换结果为DataFrame
results_df = pd.DataFrame(results)
results_df1 = pd.DataFrame(results1)

# 保存结果到CSV
results_df.to_csv(os.path.join(output_dir, 'all_data_result_time_wna_mlr.csv'), index=False)
results_df1.to_csv(os.path.join(output_dir, 'pred_result_time_wna_mlr.csv'), index=False)

# 计算整体评价指标
if not results_df1.empty:
    r_squared = r2_score(results_df1['y_test'], results_df1['y_pred'])
    r = np.corrcoef(results_df1['y_test'], results_df1['y_pred'])[0, 1]
    rmse = np.sqrt(mean_squared_error(results_df1['y_test'], results_df1['y_pred']))
    mae = mean_absolute_error(results_df1['y_test'], results_df1['y_pred'])
    bias = np.mean(results_df1['y_pred'] - results_df1['y_test'])

    # 打印指标
    print("\n多元线性回归(MLR) 整体评价指标:")
    print(f"all_data_result_time R²: {r_squared:.4f}")
    print(f"all_data_result_time R: {r:.4f}")
    print(f"all_data_result_time RMSE: {rmse:.4f}")
    print(f"all_data_result_time MAE: {mae:.4f}")
    print(f"all_data_result_time Bias: {bias:.4f}")

 
else:
    print("没有有效的预测结果，可能是测试集数据不足导致")
