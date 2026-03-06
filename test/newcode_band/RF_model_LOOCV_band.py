"""
高程带数据 - 随机森林模型（留一年交叉验证）
使用 newcode_band 流程处理的数据集

本脚本实现：
1. RFE 特征选择（选择 R^2 最高时的特征）
2. 留一年交叉验证（LOOCV）
3. 随机森林回归
4. 特征重要性可视化

输入：
    - H:/Code/SMB/test/result_data_band/wgms_era5_band_merged_final.csv

输出：
    - 模型性能指标（R^2, RMSE, MAE）
    - 特征重要性图
    - 预测结果 CSV

特殊说明：
    - 高程带数据：每个冰川-年有多个高程带记录
    - 不包含 AAR 和 ELA（避免数据泄露）
    - 包含高程特征（ELEVATION_MIDPOINT, ELEVATION_NORMALIZED 等）

作者：Claude Code
日期：2025-12-29
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os
import warnings
warnings.filterwarnings('ignore')

# ================= 配置 =================

DATA_PATH = r"H:\Code\SMB\test\result_data_band\wgms_era5_band_merged_final.csv"
OUTPUT_DIR = r"H:\Code\SMB\test\result_data_band"
RESULTS_FILE = os.path.join(OUTPUT_DIR, "rf_loocv_predictions_band.csv")
METRICS_FILE = os.path.join(OUTPUT_DIR, "rf_loocv_metrics_band.txt")

# ================= 1. 数据加载和准备 =================

print("=" * 70)
print("高程带数据 - 随机森林模型（留一年交叉验证）")
print("=" * 70)

print("\n>>> 步骤 1: 加载数据...")
print(f"  路径: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print(f"  已加载: {len(df):,} 条高程带记录")
print(f"  冰川数: {df['WGMS_ID'].nunique()}")
print(f"  年份范围: {df['YEAR'].min()}-{df['YEAR'].max()}")
print(f"  冰川-年组合: {df.groupby(['WGMS_ID', 'YEAR']).ngroups:,}")

# 移除目标变量缺失的记录
df_clean = df.dropna(subset=['ANNUAL_BALANCE']).copy()
print(f"  移除 ANNUAL_BALANCE 缺失后: {len(df_clean):,} 条记录")

# ================= 2. 特征工程 =================

print("\n>>> 步骤 2: 构建特征池...")
print("  注意: 高程带数据不包含 AAR 和 ELA（避免数据泄露）")

# 1. 气候特征（年度）
climate_annual = [col for col in df_clean.columns if col.endswith('_year')]

# 2. 气候特征（夏季）
climate_summer = [col for col in df_clean.columns if col.endswith('_summer')]

# 3. 地理特征
geography_features = ['LATITUDE', 'LONGITUDE']

# 4. 高程带特征（这是高程带数据特有的）
elevation_features = [
    'ELEVATION_MIDPOINT',      # 高程带中点（绝对高程）
    'ELEVATION_NORMALIZED',    # 归一化高程位置（0-1）
    'ELEVATION_RANGE',         # 高程带宽度
    'ELEVATION_KM'             # 高程（公里）
]
# 只保留存在的高程特征
elevation_features = [f for f in elevation_features if f in df_clean.columns]

# 5. 冰川几何特征
geometry_features = []
potential_geom = ['BAND_AREA', 'GLACIER_MIN_ELEV', 'GLACIER_MAX_ELEV']
for feat in potential_geom:
    if feat in df_clean.columns:
        geometry_features.append(feat)

# 6. 时间特征
time_feature = ['YEAR']

# 合并所有特征
all_features = (geography_features + elevation_features + geometry_features +
                climate_annual + climate_summer + time_feature)

print(f"\n  特征类别:")
print(f"    地理位置         : {len(geography_features)} 个特征")
print(f"    高程带特征       : {len(elevation_features)} 个特征（高程带数据特有）")
print(f"    冰川几何         : {len(geometry_features)} 个特征")
print(f"    气候（年度）      : {len(climate_annual)} 个特征")
print(f"    气候（夏季）      : {len(climate_summer)} 个特征")
print(f"    时间             : {len(time_feature)} 个特征")
print(f"    总计             : {len(all_features)} 个特征")

# 填充特征缺失值（使用中位数）
print("\n  处理缺失值...")
missing_count_total = 0
for col in all_features:
    if col in df_clean.columns:
        missing_count = df_clean[col].isna().sum()
        if missing_count > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            missing_count_total += missing_count
            print(f"    已填充 {missing_count} 个缺失值: {col}")

if missing_count_total == 0:
    print(f"    无缺失值")

# 准备特征矩阵和目标变量
X = df_clean[all_features].copy()
y = df_clean['ANNUAL_BALANCE'].copy()
years = df_clean['YEAR'].copy()
glacier_ids = df_clean['WGMS_ID'].copy()

print(f"\n  最终数据集形状: X={X.shape}, y={y.shape}")

# ================= 3. RFE 特征选择（寻找最优特征数）=================

print("\n>>> 步骤 3: 运行 RFE 特征选择...")
print("  策略: 寻找使 R^2 最高的特征子集")
print("  这可能需要几分钟...")

# 基础随机森林模型
rf_base = RandomForestRegressor(
    n_estimators=100,
    max_depth=50,
    random_state=42,
    n_jobs=-1
)

# 尝试不同的特征数量，找到最优的
best_n_features = 10
best_score = -np.inf
scores_by_n = {}

# 从 10 到 min(50, 总特征数) 搜索最优特征数
max_features = min(50, len(all_features))
print(f"  搜索范围: 10 到 {max_features} 个特征")

for n_features in range(10, max_features + 1, 5):
    rfe = RFE(estimator=rf_base, n_features_to_select=n_features, step=5)
    rfe.fit(X, y)

    # 使用 5 折交叉验证评估
    X_selected = X.loc[:, rfe.support_]
    cv_scores = cross_val_score(rf_base, X_selected, y, cv=5,
                                scoring='r2', n_jobs=-1)
    mean_score = cv_scores.mean()
    scores_by_n[n_features] = mean_score

    print(f"    特征数 {n_features:2d}: R^2 = {mean_score:.4f}")

    if mean_score > best_score:
        best_score = mean_score
        best_n_features = n_features

print(f"\n  最优特征数量: {best_n_features}（R^2 = {best_score:.4f}）")

# 使用最优特征数重新训练 RFE
print(f"  使用 {best_n_features} 个特征重新训练...")
rfe_final = RFE(estimator=rf_base, n_features_to_select=best_n_features, step=1)
rfe_final.fit(X, y)

# 获取被选中的特征
selected_features = np.array(all_features)[rfe_final.support_]
n_selected = len(selected_features)

print(f"\n  被选中的 {n_selected} 个特征:")
for i, feat in enumerate(selected_features, 1):
    print(f"    {i:2d}. {feat}")

# ================= 4. 留一年交叉验证 =================

print("\n>>> 步骤 4: 运行留一年交叉验证...")
print("  为每个测试年份训练模型...")
print("  注意: 高程带数据中每个冰川-年有多个记录")

results = {
    'y_true': [],
    'y_pred': [],
    'year': [],
    'glacier_id': [],
    'elevation': []
}

yearly_metrics = []
unique_years = sorted(years.unique())

for test_year in unique_years:
    # 划分数据：当前年份用于测试，其他年份用于训练
    train_mask = (years != test_year)
    test_mask = (years == test_year)

    # 如果测试样本太少则跳过
    n_test = test_mask.sum()
    if n_test < 2:
        print(f"    年份 {test_year}: 跳过（仅 {n_test} 个高程带）")
        continue

    # 使用选中的特征准备训练/测试数据
    X_train = X.loc[train_mask, selected_features]
    y_train = y[train_mask]
    X_test = X.loc[test_mask, selected_features]
    y_test = y[test_mask]

    # 训练随机森林模型
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=100,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    # 预测测试年份
    y_pred = rf_model.predict(X_test)

    # 存储结果
    results['y_true'].extend(y_test.values)
    results['y_pred'].extend(y_pred)
    results['year'].extend([test_year] * n_test)
    results['glacier_id'].extend(glacier_ids[test_mask].values)

    # 存储高程信息（如果有）
    if 'ELEVATION_MIDPOINT' in df_clean.columns:
        results['elevation'].extend(df_clean.loc[test_mask, 'ELEVATION_MIDPOINT'].values)
    else:
        results['elevation'].extend([np.nan] * n_test)

    # 计算年度指标
    year_r2 = r2_score(y_test, y_pred)
    year_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    year_mae = mean_absolute_error(y_test, y_pred)

    yearly_metrics.append({
        'year': test_year,
        'n_samples': n_test,
        'r2': year_r2,
        'rmse': year_rmse,
        'mae': year_mae
    })

    print(f"    年份 {test_year}: n={n_test:3d}带, R^2={year_r2:6.3f}, RMSE={year_rmse:6.3f}, MAE={year_mae:6.3f}")

# ================= 5. 整体性能评估 =================

print("\n>>> 步骤 5: 整体性能评估...")

y_true = np.array(results['y_true'])
y_pred = np.array(results['y_pred'])

# 计算整体指标
overall_r2 = r2_score(y_true, y_pred)
overall_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
overall_mae = mean_absolute_error(y_true, y_pred)
overall_r = np.corrcoef(y_true, y_pred)[0, 1]
bias = np.mean(y_pred - y_true)

print("\n" + "=" * 70)
print("整体模型性能（高程带数据 - 留一年交叉验证）")
print("=" * 70)
print(f"  R^2（决定系数）            : {overall_r2:7.4f}")
print(f"  R（相关系数）              : {overall_r:7.4f}")
print(f"  RMSE（均方根误差）         : {overall_rmse:7.4f} m w.e.")
print(f"  MAE（平均绝对误差）        : {overall_mae:7.4f} m w.e.")
print(f"  Bias（平均预测误差）       : {bias:7.4f} m w.e.")
print(f"  测试样本总数（高程带）      : {len(y_true):7d}")
print("=" * 70)

# ================= 6. 特征重要性分析 =================

print("\n>>> 步骤 6: 特征重要性分析...")

# 在全数据集上训练模型以获取特征重要性
rf_final = RandomForestRegressor(n_estimators=200, max_depth=100, random_state=42, n_jobs=-1)
rf_final.fit(X[selected_features], y)

importances = rf_final.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': selected_features,
    'importance': importances
}).sort_values('importance', ascending=False)

print("\n  前 15 个最重要特征:")
for i, row in feature_importance_df.head(15).iterrows():
    # 标注高程带特征
    marker = " [高程带]" if row['feature'] in elevation_features else ""
    print(f"    {row['feature']:40s}: {row['importance']:.4f}{marker}")

# ================= 7. 保存结果 =================

print("\n>>> 步骤 7: 保存结果...")

# 保存预测值
pred_df = pd.DataFrame({
    'YEAR': results['year'],
    'WGMS_ID': results['glacier_id'],
    'ELEVATION_MIDPOINT': results['elevation'],
    'ANNUAL_BALANCE_TRUE': y_true,
    'ANNUAL_BALANCE_PRED': y_pred,
    'RESIDUAL': y_true - y_pred
})
pred_df.to_csv(RESULTS_FILE, index=False, encoding='utf-8-sig')
print(f"  预测结果已保存: {RESULTS_FILE}")

# 保存指标
with open(METRICS_FILE, 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("高程带数据 - 随机森林模型（留一年交叉验证）\n")
    f.write("=" * 70 + "\n\n")

    f.write("数据说明：\n")
    f.write(f"  数据类型：高程带分层数据\n")
    f.write(f"  每个冰川-年有多个高程带记录\n")
    f.write(f"  不包含 AAR 和 ELA（避免数据泄露）\n")
    f.write(f"  包含高程特征（ELEVATION_MIDPOINT, ELEVATION_NORMALIZED 等）\n\n")

    f.write("整体指标:\n")
    f.write(f"  R^2   : {overall_r2:.4f}\n")
    f.write(f"  R     : {overall_r:.4f}\n")
    f.write(f"  RMSE : {overall_rmse:.4f} m w.e.\n")
    f.write(f"  MAE  : {overall_mae:.4f} m w.e.\n")
    f.write(f"  Bias : {bias:.4f} m w.e.\n")
    f.write(f"  N    : {len(y_true)} 个高程带\n\n")

    f.write(f"RFE 特征选择结果:\n")
    f.write(f"  最优特征数量: {n_selected}\n")
    f.write(f"  交叉验证 R^2: {best_score:.4f}\n\n")

    f.write("被选中的特征:\n")
    for i, feat in enumerate(selected_features, 1):
        marker = " [高程带特征]" if feat in elevation_features else ""
        f.write(f"  {i:2d}. {feat}{marker}\n")

    f.write("\n" + "=" * 70 + "\n")
    f.write("特征重要性（前 20）:\n")
    f.write("=" * 70 + "\n")
    for i, row in feature_importance_df.head(20).iterrows():
        marker = " [高程带]" if row['feature'] in elevation_features else ""
        f.write(f"  {row['feature']:40s}: {row['importance']:.4f}{marker}\n")

    f.write("\n" + "=" * 70 + "\n")
    f.write("逐年性能:\n")
    f.write("=" * 70 + "\n")
    for metric in yearly_metrics:
        f.write(f"  年份 {metric['year']}: n={metric['n_samples']:3d}带, "
                f"R^2={metric['r2']:6.3f}, RMSE={metric['rmse']:6.3f}, MAE={metric['mae']:6.3f}\n")

print(f"  指标已保存: {METRICS_FILE}")

# ================= 8. 可视化 =================

print("\n>>> 步骤 8: 创建可视化图表...")

# 图 1: 特征重要性
plt.figure(figsize=(12, 8))

top_n = min(20, len(feature_importance_df))
top_features = feature_importance_df.head(top_n)

# 根据特征类型着色
colors = []
for feat in top_features['feature']:
    if feat in elevation_features:
        colors.append('purple')  # 高程带特征用紫色
    elif feat in geography_features or feat in geometry_features:
        colors.append('red')
    elif '_summer' in feat:
        colors.append('orange')
    elif feat == 'YEAR':
        colors.append('green')
    else:
        colors.append('skyblue')

sns.barplot(
    data=top_features,
    y='feature',
    x='importance',
    hue='feature',
    palette=colors,
    legend=False
)

plt.title(f'特征重要性（高程带数据 - 随机森林 LOOCV）\nR^2 = {overall_r2:.4f}, RMSE = {overall_rmse:.3f} m w.e.',
          fontsize=14, fontweight='bold')
plt.xlabel('重要性得分', fontsize=12)
plt.ylabel('特征', fontsize=12)
plt.tight_layout()

# 保存图表
fig1_path = os.path.join(OUTPUT_DIR, 'rf_loocv_feature_importance_band.png')
plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
print(f"  特征重要性图已保存: {fig1_path}")
plt.show()

# 图 2: 预测 vs 观测
plt.figure(figsize=(10, 10))

plt.scatter(y_true, y_pred, alpha=0.5, s=30, edgecolors='black', linewidth=0.3)

# 添加 1:1 线
min_val = min(y_true.min(), y_pred.min())
max_val = max(y_true.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1 线')

# 添加回归线
from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = linregress(y_true, y_pred)
x_line = np.array([min_val, max_val])
y_line = slope * x_line + intercept
plt.plot(x_line, y_line, 'b-', linewidth=2, label=f'回归线 (斜率={slope:.2f})')

plt.xlabel('观测年物质平衡（高程带）(m w.e.)', fontsize=12)
plt.ylabel('预测年物质平衡（高程带）(m w.e.)', fontsize=12)
plt.title(f'高程带数据 - 随机森林 LOOCV 性能\nR^2 = {overall_r2:.4f}, RMSE = {overall_rmse:.3f} m w.e., n = {len(y_true)}带',
          fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()

# 保存图表
fig2_path = os.path.join(OUTPUT_DIR, 'rf_loocv_scatter_band.png')
plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
print(f"  散点图已保存: {fig2_path}")
plt.show()

# 图 3: 残差图
plt.figure(figsize=(12, 6))

residuals = y_true - y_pred

plt.subplot(1, 2, 1)
plt.scatter(y_pred, residuals, alpha=0.5, s=30, edgecolors='black', linewidth=0.3)
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.xlabel('预测年物质平衡（高程带）(m w.e.)', fontsize=12)
plt.ylabel('残差（观测 - 预测）', fontsize=12)
plt.title('残差图', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
plt.xlabel('残差 (m w.e.)', fontsize=12)
plt.ylabel('频数', fontsize=12)
plt.title(f'残差分布\n均值 = {residuals.mean():.3f}, 标准差 = {residuals.std():.3f}',
          fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.tight_layout()
fig3_path = os.path.join(OUTPUT_DIR, 'rf_loocv_residuals_band.png')
plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
print(f"  残差图已保存: {fig3_path}")
plt.show()

# ================= 最终总结 =================

print("\n" + "=" * 70)
print("模型评估完成（高程带数据）")
print("=" * 70)
print("\n输出文件:")
print(f"  预测结果     : {RESULTS_FILE}")
print(f"  性能指标     : {METRICS_FILE}")
print(f"  特征重要性图 : {fig1_path}")
print(f"  散点图       : {fig2_path}")
print(f"  残差图       : {fig3_path}")
print("\n" + "=" * 70)
print("关键结果（高程带数据）:")
print(f"  R^2   = {overall_r2:.4f}")
print(f"  RMSE = {overall_rmse:.3f} m w.e.")
print(f"  MAE  = {overall_mae:.3f} m w.e.")
print(f"  Bias = {bias:.3f} m w.e.")
print(f"  测试高程带数 = {len(y_true)}")
print("=" * 70)
print("\n说明：")
print("  - 本分析基于高程带分层数据")
print("  - 不包含 AAR 和 ELA（避免数据泄露）")
print("  - 包含高程特征（ELEVATION_MIDPOINT, ELEVATION_NORMALIZED）")
print("  - 使用 RFE 选择最优特征子集（最大化 R^2）")
