import pandas as pd
from sklearn.linear_model import Lasso  # 替换随机森林为Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler  # Lasso需要特征标准化
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

# 初始化特征标准化器（Lasso对特征尺度敏感，必须标准化）
scaler = StandardScaler()

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
        # 特征标准化（仅用训练集拟合scaler，避免数据泄露）
        X_train_scaled = scaler.fit_transform(X_train)  # 训练集：拟合+转换
        X_test_scaled = scaler.transform(X_test)        # 测试集：仅转换

        # 初始化并训练Lasso模型（alpha为正则化参数，需根据数据调整）
        # alpha建议通过交叉验证选择，这里先设为0.1（可后续优化）
        model = Lasso(alpha=0.1, max_iter=10000, random_state=42)
        model.fit(X_train_scaled, y_train)

        # 进行预测
        y_pred = model.predict(X_test_scaled)
        y_pred = y_pred / 1000  # 单位转换（与原代码一致）
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

# 转换结果为DataFrame
results_df = pd.DataFrame(results)
results_df1 = pd.DataFrame(results1)

# 保存结果到CSV（文件名区分Lasso和随机森林）
results_df.to_csv('H:/Code/SMB/test/result/all_data_result_time_wna_lasso.csv', index=False)
results_df1.to_csv('H:/Code/SMB/test/result/pred_result_time_wna_lasso.csv', index=False)

# 计算整体评价指标
r_squared = r2_score(results_df1['y_test'], results_df1['y_pred'])
r = np.corrcoef(results_df1['y_test'], results_df1['y_pred'])[0, 1]
rmse = np.sqrt(mean_squared_error(results_df1['y_test'], results_df1['y_pred']))
mae = mean_absolute_error(results_df1['y_test'], results_df1['y_pred'])
bias = np.mean(results_df1['y_pred'] - results_df1['y_test'])

# 打印指标
print("Lasso all_data_result_time R²:", r_squared)
print("Lasso all_data_result_time R:", r)
print("Lasso all_data_result_time RMSE:", rmse)
print("Lasso all_data_result_time MAE:", mae)
print("Lasso all_data_result_time Bias:", bias)

# （优化）特征系数可视化：按系数绝对值从大到小排列
plt.figure(figsize=(10, 8))  # 适当增大图幅，避免特征名重叠

# 提取系数并转换为Series，索引为特征名
coefficients = pd.Series(model.coef_, index=features_columns)

# 按系数绝对值从大到小排序（若想按原始值大小排序，去掉.abs()即可）
coefficients_sorted = coefficients.abs().sort_values(ascending=False)
# 保留排序后的原始系数值（而非绝对值）
coefficients_sorted = coefficients.reindex(coefficients_sorted.index)

# 绘制横向条形图
coefficients_sorted.plot(kind='barh', color=np.where(coefficients_sorted > 0, 'blue', 'red'))

# 添加标题和标签
plt.title('Lasso_Feature_importance', fontsize=12)
plt.xlabel('系数值（正值：正相关；负值：负相关）', fontsize=10)
plt.axvline(x=0, color='black', linestyle='--', alpha=0.3)  # 添加零点参考线
plt.tight_layout()  # 自动调整布局，避免文字截断

# 保存图片
plt.savefig('H:/Code/SMB/test/result/lasso_coefficients_sorted.png', dpi=300, bbox_inches='tight')
plt.show()