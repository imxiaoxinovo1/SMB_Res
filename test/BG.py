import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import norm
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

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
    'bias': [],
    'pred_std_mean': []  # 新增：预测标准差的均值（反映不确定性）
}

results1 = {
    'year': [],
    'y_test': [],
    'y_pred': [],
    'y_pred_std': []  # 新增：每个预测值的标准差
}

# 特征列（与原代码一致）
features_columns = [
    "LOWER_BOUND", "UPPER_BOUND", "snowmelt_sum_year", "AREA", "snow_cover_summer",
    "skin_reservoir_content_summer", "evaporation_from_open_water_surfaces_excluding_oceans_sum_summer",
    "runoff_sum_year", "LONGITUDE", "lake_mix_layer_temperature_summer", "snowfall_sum_year",
    "evaporation_from_the_top_of_canopy_sum_summer", "surface_runoff_sum_year",
    "evaporation_from_open_water_surfaces_excluding_oceans_sum_year", "sub_surface_runoff_sum_year",
    "PBLH", "RHOA", "PRECSNO", "GHTSKIN", "YEAR"
]
target_column = 'ANNUAL_BALANCE'

# 初始化特征标准化器（贝叶斯回归对特征尺度敏感）
scaler = StandardScaler()

# 贝叶斯线性回归类
class BayesianLinearRegression:
    def __init__(self, alpha=1.0, beta=1.0):
        """
        初始化贝叶斯线性回归
        alpha: 先验分布的精度参数（1/方差）
        beta: 似然函数的精度参数
        """
        self.alpha = alpha  # 权重先验的精度
        self.beta = beta    # 噪声的精度
        self.w_mean = None  # 权重后验均值
        self.w_cov = None   # 权重后验协方差

    def fit(self, X, y):
        """训练模型，计算权重的后验分布"""
        # 添加偏置项（截距）
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])  # 形状：(n_samples, n_features+1)
        n_features = X_bias.shape[1]

        # 先验协方差矩阵 (alphaI)
        prior_cov = np.eye(n_features) / self.alpha
        prior_inv = np.linalg.inv(prior_cov)

        # 计算后验协方差和均值
        X_T = X_bias.T
        self.w_cov = np.linalg.inv(prior_inv + self.beta * X_T @ X_bias)
        self.w_mean = self.beta * self.w_cov @ X_T @ y

    def predict(self, X, return_std=True):
        """预测并返回均值和标准差（不确定性）"""
        # 添加偏置项
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        
        # 预测均值
        y_mean = X_bias @ self.w_mean
        
        if return_std:
            # 预测标准差（包含模型不确定性和噪声）
            y_var = 1/self.beta + np.diag(X_bias @ self.w_cov @ X_bias.T)
            y_std = np.sqrt(y_var)
            return y_mean, y_std
        return y_mean

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

        # 初始化并训练贝叶斯线性回归模型
        model = BayesianLinearRegression(alpha=1.0, beta=0.1)  # 先验参数可调整
        model.fit(X_train_scaled, y_train)

        # 进行预测（获取均值和标准差）
        y_pred_mean, y_pred_std = model.predict(X_test_scaled, return_std=True)
        
        # 单位转换
        y_pred_mean = y_pred_mean / 1000
        y_pred_std = y_pred_std / 1000  # 标准差也同步转换单位
        y_test = y_test / 1000

        # 计算评价指标
        r_squared = r2_score(y_test, y_pred_mean)
        r = np.corrcoef(y_test, y_pred_mean)[0, 1]
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_mean))
        mae = mean_absolute_error(y_test, y_pred_mean)
        bias = np.mean(y_pred_mean - y_test)
        pred_std_mean = np.mean(y_pred_std)  # 计算平均预测不确定性

        # 存储评价指标
        results['year'].append(test_year)
        results['test_samples'].append(len(y_test))
        results['r_squared'].append(r_squared)
        results['r'].append(r)
        results['rmse'].append(rmse)
        results['mae'].append(mae)
        results['bias'].append(bias)
        results['pred_std_mean'].append(pred_std_mean)

        # 存储预测结果（包含不确定性）
        results1['year'].extend([test_year] * len(y_test))
        results1['y_test'].extend(y_test)
        results1['y_pred'].extend(y_pred_mean)
        results1['y_pred_std'].extend(y_pred_std)
    else:
        print(f"  测试集样本不足，跳过年份 {test_year}")

# 转换结果为DataFrame
results_df = pd.DataFrame(results)
results_df1 = pd.DataFrame(results1)

# 保存结果到CSV
results_df.to_csv(os.path.join(output_dir, 'all_data_result_time_wna_bayesian.csv'), index=False)
results_df1.to_csv(os.path.join(output_dir, 'pred_result_time_wna_bayesian.csv'), index=False)

# 计算整体评价指标
if not results_df1.empty:
    r_squared = r2_score(results_df1['y_test'], results_df1['y_pred'])
    r = np.corrcoef(results_df1['y_test'], results_df1['y_pred'])[0, 1]
    rmse = np.sqrt(mean_squared_error(results_df1['y_test'], results_df1['y_pred']))
    mae = mean_absolute_error(results_df1['y_test'], results_df1['y_pred'])
    bias = np.mean(results_df1['y_pred'] - results_df1['y_test'])
    overall_pred_std = np.mean(results_df1['y_pred_std'])

    # 打印指标（包含不确定性分析）
    print("\n贝叶斯回归 整体评价指标:")
    print(f"all_data_result_time R²: {r_squared:.4f}")
    print(f"all_data_result_time R: {r:.4f}")
    print(f"all_data_result_time RMSE: {rmse:.4f}")
    print(f"all_data_result_time MAE: {mae:.4f}")
    print(f"all_data_result_time Bias: {bias:.4f}")
    print(f"all_data_result_time 平均预测标准差（不确定性）: {overall_pred_std:.4f}")

    # 1. 预测值vs真实值（带不确定性区间）
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        results_df1['y_test'], 
        results_df1['y_pred'], 
        yerr=results_df1['y_pred_std'],  # 误差线：±1标准差
        fmt='o', 
        alpha=0.6, 
        ecolor='lightgray', 
        capsize=2,
        label='预测值（带不确定性）'
    )
    # 添加1:1参考线
    plt.plot(
        [min(results_df1['y_test']), max(results_df1['y_test'])],
        [min(results_df1['y_test']), max(results_df1['y_test'])],
        'r--', 
        label='1:1参考线'
    )
    plt.xlabel('真实年度平衡（m w.e.）')
    plt.ylabel('预测年度平衡（m w.e.）')
    plt.title(f'贝叶斯回归预测结果（R={r:.2f}，平均不确定性={overall_pred_std:.2f}）')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bayesian_pred_vs_obs.png'), dpi=300)
    plt.show()

    # 2. 特征系数后验分布可视化（展示不确定性）
    if 'model' in locals():  # 确保模型已训练
        plt.figure(figsize=(12, 8))
        # 提取系数均值和标准差（排除偏置项）
        coef_means = model.w_mean[1:]  # 第0项是偏置
        coef_stds = np.sqrt(np.diag(model.w_cov)[1:])
        # 按系数绝对值排序
        sorted_idx = np.argsort(np.abs(coef_means))[::-1]
        
        # 绘制系数及其95%置信区间
        plt.errorbar(
            coef_means[sorted_idx],
            np.arange(len(sorted_idx)),
            xerr=1.96*coef_stds[sorted_idx],  # 95%置信区间（±1.96σ）
            fmt='o',
            ecolor='lightgray',
            capsize=3,
            color='blue'
        )
        plt.yticks(np.arange(len(sorted_idx)), [features_columns[i] for i in sorted_idx])
        plt.axvline(x=0, color='red', linestyle='--')  # 零点参考线
        plt.xlabel('系数值（95%置信区间）')
        plt.title('贝叶斯回归特征系数后验分布（按绝对值排序）')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'bayesian_coef_distribution.png'), dpi=300)
        plt.show()
else:
    print("没有有效的预测结果，可能是测试集数据不足导致")
