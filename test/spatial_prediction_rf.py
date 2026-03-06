"""
冰川物质平衡空间预测脚本 - 使用随机森林模型
将训练好的模型应用到无观测数据的冰川区域

工作流程：
1. 使用所有有观测数据的冰川训练模型
2. 对没有观测数据的冰川区域（仅有气象数据）进行预测
3. 输出预测结果并评估不确定性
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import os

# ================= 配置区域 =================

# 输入文件：包含WGMS观测数据 + ERA5气象数据
observed_data_path = 'H:/Code/SMB/test/study_data_wna.csv'

# 输入文件：仅包含气象数据的目标冰川（待预测区域）
# 这个文件需要从 data_processing.py 生成，包含所有RGI冰川的气象数据
target_glaciers_path = 'H:/Code/SMB/test/result/target_glaciers_climate_data.csv'

# 输出目录
output_dir = 'H:/Code/SMB/test/result/spatial_prediction'
os.makedirs(output_dir, exist_ok=True)

# 特征列（与训练时保持一致）
features_columns = [
    "LOWER_BOUND", "UPPER_BOUND", "snowmelt_sum_year", "AREA",
    "snow_cover_summer", "skin_reservoir_content_summer",
    "evaporation_from_open_water_surfaces_excluding_oceans_sum_summer",
    "runoff_sum_year", "LONGITUDE", "lake_mix_layer_temperature_summer",
    "snowfall_sum_year", "evaporation_from_the_top_of_canopy_sum_summer",
    "surface_runoff_sum_year",
    "evaporation_from_open_water_surfaces_excluding_oceans_sum_year",
    "sub_surface_runoff_sum_year", "PBLH", "RHOA", "PRECSNO",
    "GHTSKIN", "YEAR"
]
target_column = 'ANNUAL_BALANCE'

# 模型参数（与验证时的最优参数保持一致）
N_ESTIMATORS = 200
MAX_DEPTH = 100
RANDOM_STATE = 42

# ================= 步骤1: 模型训练与验证 =================

print("=" * 60)
print("步骤1: 使用有观测数据的冰川训练随机森林模型")
print("=" * 60)

# 读取观测数据
df_observed = pd.read_csv(observed_data_path)
df_observed = df_observed.dropna()

print(f"\n观测数据统计:")
print(f"  - 总记录数: {len(df_observed)}")
print(f"  - 冰川数量: {df_observed['WGMS_ID'].nunique() if 'WGMS_ID' in df_observed.columns else 'N/A'}")
print(f"  - 时间范围: {df_observed['YEAR'].min():.0f} - {df_observed['YEAR'].max():.0f}")

# Leave-one-year-out 交叉验证（评估模型性能）
print("\n进行 Leave-One-Year-Out 交叉验证...")
cv_results = []

for test_year in range(int(df_observed['YEAR'].min()), int(df_observed['YEAR'].max()) + 1):
    train_data = df_observed[df_observed['YEAR'] != test_year]
    test_data = df_observed[(df_observed['YEAR'] == test_year) & (df_observed['TAG'] == 9999)]

    if len(test_data) < 2:
        continue

    X_train = train_data[features_columns]
    y_train = train_data[target_column]
    X_test = test_data[features_columns]
    y_test = test_data[target_column]

    # 训练模型
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)
    y_pred_m = y_pred / 1000  # mm -> m
    y_test_m = y_test / 1000

    # 评估指标
    r2 = r2_score(y_test_m, y_pred_m)
    r = np.corrcoef(y_test_m, y_pred_m)[0, 1]
    rmse = np.sqrt(mean_squared_error(y_test_m, y_pred_m))
    mae = mean_absolute_error(y_test_m, y_pred_m)
    bias = np.mean(y_pred_m - y_test_m)

    cv_results.append({
        'year': test_year,
        'n_samples': len(y_test),
        'r_squared': r2,
        'r': r,
        'rmse': rmse,
        'mae': mae,
        'bias': bias
    })

# 输出验证结果
df_cv = pd.DataFrame(cv_results)
print(f"\n交叉验证完成 ({len(df_cv)} 年)")
print(f"  - 平均 R²: {df_cv['r_squared'].mean():.4f}")
print(f"  - 平均 R: {df_cv['r'].mean():.4f}")
print(f"  - 平均 RMSE: {df_cv['rmse'].mean():.4f} m w.e.")
print(f"  - 平均 MAE: {df_cv['mae'].mean():.4f} m w.e.")
print(f"  - 平均 Bias: {df_cv['bias'].mean():.4f} m w.e.")

# ================= 步骤2: 训练最终模型 =================

print("\n" + "=" * 60)
print("步骤2: 使用全部观测数据训练最终预测模型")
print("=" * 60)

X_all = df_observed[features_columns]
y_all = df_observed[target_column]

final_model = RandomForestRegressor(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

print(f"\n训练最终模型...")
print(f"  - 训练样本数: {len(X_all)}")
print(f"  - 特征数量: {len(features_columns)}")

final_model.fit(X_all, y_all)

# 特征重要性分析
feature_importance = pd.DataFrame({
    'feature': features_columns,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n前10个最重要特征:")
for idx, row in feature_importance.head(10).iterrows():
    print(f"  {row['feature']:50s}: {row['importance']:.4f}")

# 保存特征重要性
feature_importance.to_csv(
    os.path.join(output_dir, 'feature_importance.csv'),
    index=False
)

# ================= 步骤3: 空间预测 =================

print("\n" + "=" * 60)
print("步骤3: 对无观测数据的冰川进行空间预测")
print("=" * 60)

# 检查目标文件是否存在
if not os.path.exists(target_glaciers_path):
    print(f"\n警告: 目标冰川数据文件不存在!")
    print(f"需要的文件: {target_glaciers_path}")
    print(f"\n请先运行以下步骤:")
    print(f"1. 使用 data_processing.py 处理 RGI 冰川数据")
    print(f"2. 提取所有冰川（不仅是WGMS观测冰川）的气象数据")
    print(f"3. 生成包含以下字段的CSV文件:")
    for col in features_columns:
        print(f"   - {col}")
    print(f"\n当前脚本将只保存已训练的模型，跳过预测步骤。")

    # 保存模型供后续使用
    import pickle
    model_path = os.path.join(output_dir, 'rf_model_final.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(final_model, f)
    print(f"\n模型已保存至: {model_path}")

else:
    # 读取目标冰川数据
    df_target = pd.read_csv(target_glaciers_path)
    print(f"\n目标冰川数据统计:")
    print(f"  - 总记录数: {len(df_target)}")
    print(f"  - 冰川数量: {df_target['GLACIER_ID'].nunique() if 'GLACIER_ID' in df_target.columns else 'N/A'}")
    print(f"  - 时间范围: {df_target['YEAR'].min():.0f} - {df_target['YEAR'].max():.0f}")

    # 检查必需特征是否存在
    missing_features = [f for f in features_columns if f not in df_target.columns]
    if missing_features:
        print(f"\n错误: 目标数据缺少以下特征:")
        for f in missing_features:
            print(f"  - {f}")
        print("\n请确保目标数据包含所有训练特征。")
    else:
        # 进行预测
        print(f"\n开始预测...")
        X_target = df_target[features_columns]

        # 预测物质平衡（mm w.e.）
        y_pred_target = final_model.predict(X_target)

        # 预测不确定性估计（使用树的标准差）
        # 获取每棵树的预测
        tree_predictions = np.array([tree.predict(X_target) for tree in final_model.estimators_])
        y_pred_std = np.std(tree_predictions, axis=0)  # 标准差

        # 添加预测结果到数据框
        df_target['PREDICTED_BALANCE_MM'] = y_pred_target
        df_target['PREDICTED_BALANCE_M'] = y_pred_target / 1000
        df_target['PREDICTION_STD_MM'] = y_pred_std
        df_target['PREDICTION_STD_M'] = y_pred_std / 1000

        # 保存预测结果
        output_path = os.path.join(output_dir, 'spatial_predictions.csv')
        df_target.to_csv(output_path, index=False)
        print(f"\n预测完成!")
        print(f"  - 预测结果已保存至: {output_path}")

        # 预测统计
        print(f"\n预测统计:")
        print(f"  - 平均物质平衡: {df_target['PREDICTED_BALANCE_M'].mean():.3f} m w.e.")
        print(f"  - 标准差: {df_target['PREDICTED_BALANCE_M'].std():.3f} m w.e.")
        print(f"  - 最小值: {df_target['PREDICTED_BALANCE_M'].min():.3f} m w.e.")
        print(f"  - 最大值: {df_target['PREDICTED_BALANCE_M'].max():.3f} m w.e.")
        print(f"  - 平均不确定性: {df_target['PREDICTION_STD_M'].mean():.3f} m w.e.")

        # 可视化预测分布
        plt.figure(figsize=(12, 5))

        # 子图1: 预测值分布
        plt.subplot(1, 2, 1)
        plt.hist(df_target['PREDICTED_BALANCE_M'], bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Predicted Mass Balance (m w.e.)', fontsize=11)
        plt.ylabel('Frequency', fontsize=11)
        plt.title('Distribution of Predicted Mass Balance', fontsize=12, fontweight='bold')
        plt.axvline(df_target['PREDICTED_BALANCE_M'].mean(),
                   color='red', linestyle='--', linewidth=2, label='Mean')
        plt.legend()
        plt.grid(alpha=0.3)

        # 子图2: 不确定性分布
        plt.subplot(1, 2, 2)
        plt.hist(df_target['PREDICTION_STD_M'], bins=50, edgecolor='black',
                alpha=0.7, color='orange')
        plt.xlabel('Prediction Uncertainty (m w.e.)', fontsize=11)
        plt.ylabel('Frequency', fontsize=11)
        plt.title('Distribution of Prediction Uncertainty', fontsize=12, fontweight='bold')
        plt.axvline(df_target['PREDICTION_STD_M'].mean(),
                   color='red', linestyle='--', linewidth=2, label='Mean')
        plt.legend()
        plt.grid(alpha=0.3)

        plt.tight_layout()
        fig_path = os.path.join(output_dir, 'prediction_distribution.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"  - 分布图已保存至: {fig_path}")
        plt.close()

print("\n" + "=" * 60)
print("处理完成!")
print("=" * 60)
