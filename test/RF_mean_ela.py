import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


# 设置文件路径
# <--- MODIFIED 1: 更改为包含 mean_elevation 的新文件
csv_file = r'H:\Code\SMB\test\result\study_data_wna_with_mean_ela.csv'

# 读取数据
df = pd.read_csv(csv_file)
print(f"成功加载: {csv_file}")

# <--- MODIFIED 2: 特征列表
# 1. 添加了 'mean_elevation'
features_columns = [
    "mean_elevation",  #ADDED
    "LOWER_BOUND", 
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
    "YEAR"
]
target_column = 'ANNUAL_BALANCE'

# 仅删除模型运行所必需的列中包含NaN的行
# 这可以防止因其他不相关列中的NaN而丢失数据
required_columns = features_columns + [target_column, 'YEAR', 'TAG']
original_rows = len(df)
df = df.dropna(subset=required_columns)
print(f"数据清理: 原始行数 {original_rows}, 清理后行数 {len(df)} (基于所需列)")


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

# 循环遍历每个测试年份
for test_year in range(1980, 2021):
    # 分割训练集和测试集
    train_data = df[df['YEAR'] != test_year]
    test_data = df[(df['YEAR'] == test_year) & (df['TAG'] == 9999)]

    # 检查是否有足够的训练数据
    if len(train_data) == 0:
        print(f"跳过年份 {test_year}: 没有训练数据。")
        continue

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
    elif len(X_test) <= 1:
        print(f"跳过年份 {test_year}: 测试样本不足 (样本数 = {len(X_test)})")

# 将结果列表转换为DataFrame
results_df = pd.DataFrame(results)
results_df1 = pd.DataFrame(results1)


# <--- MODIFIED 3: 更改输出文件
output_results_path = r'H:/Code/SMB/test/result/all_data_result_time_wna_with_mean_ela.csv'
output_pred_path = r'H:/Code/SMB/test/result/pred_result_time_wna_with_mean_ela.csv'

# 将结果写入CSV文件
results_df.to_csv(output_results_path, index=False)
results_df1.to_csv(output_pred_path, index=False)
print(f"结果已保存到: {output_results_path}")
print(f"预测值已保存到: {output_pred_path}")

# 检查是否有结果
if not results_df1.empty:
    # 计算评价指标
    r_squared = r2_score(results_df1['y_test'], results_df1['y_pred'])
    r = np.corrcoef(results_df1['y_test'], results_df1['y_pred'])[0, 1]
    rmse = np.sqrt(mean_squared_error(results_df1['y_test'], results_df1['y_pred']))
    mae = mean_absolute_error(results_df1['y_test'], results_df1['y_pred'])
    bias = np.mean(results_df1['y_pred'] - results_df1['y_test'])

    # 打印评价指标
    print("\n加入WGMS Mean Elevation后的RF整体评价指标:")
    print(f"all_data_result_time R²: {r_squared}")
    print(f"all_data_result_time R: {r}")
    print(f"all_data_result_time RMSE: {rmse}")
    print(f"all_data_result_time MAE: {mae}")
    print(f"all_data_result_time Bias: {bias}")
else:
    print("错误: 未生成任何预测结果，请检查数据。")