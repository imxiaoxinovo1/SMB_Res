"""
随机森林模型 - 留一年交叉验证
使用 newcode 流程处理的数据集

本脚本实现：
1. RFECV 特征选择
2. 留一年交叉验证（LOOCV）
3. 随机森林回归
4. 特征重要性可视化

输入：
    - H:/Code/SMB/test/result_data/wgms_era5_merged_final.csv

输出：
    - 模型性能指标（R^2, RMSE, MAE）
    - 特征重要性图
    - 预测结果 CSV

作者：Claude Code
日期：2025-12-29

修改说明：
- 排除 AAR 和 ELA（这些是物质平衡的结果，会造成数据泄露）
- 仅保留物理驱动因素：气候、地理、冰川面积、时间
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os

# ================= 配置 =================

DATA_PATH = r"H:\Code\SMB\test\result_data\wgms_era5_merged_final.csv"
OUTPUT_DIR = r"H:\Code\SMB\test\result_data"
RESULTS_FILE = os.path.join(OUTPUT_DIR, "rf_loocv_predictions_no_leakage.csv")
METRICS_FILE = os.path.join(OUTPUT_DIR, "rf_loocv_metrics_no_leakage.txt")

# ================= 1. 数据加载和准备 =================

print("=" * 70)
print("随机森林模型 - 留一年交叉验证（无数据泄露版本）")
print("=" * 70)

print("\n>>> 步骤 1: 加载数据...")
print(f"  路径: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print(f"  已加载: {len(df):,} 条记录")
print(f"  冰川数: {df['WGMS_ID'].nunique()}")
print(f"  年份范围: {df['YEAR'].min()}-{df['YEAR'].max()}")

# 移除目标变量缺失的记录
df_clean = df.dropna(subset=['ANNUAL_BALANCE']).copy()
print(f"  移除 ANNUAL_BALANCE 缺失后: {len(df_clean):,} 条记录")

# ================= 2. 特征工程 =================

print("\n>>> 步骤 2: 构建特征池...")
print("  注意: 已排除 AAR 和 ELA（这些是物质平衡的结果，会导致数据泄露）")

# 1. 气候特征（年度）
climate_annual = [col for col in df_clean.columns if col.endswith('_year')]

# 2. 气候特征（夏季）
climate_summer = [col for col in df_clean.columns if col.endswith('_summer')]

# 3. 地理特征
geography_features = ['LATITUDE', 'LONGITUDE']

# 4. 冰川几何特征（仅保留面积和高程范围）
# 排除 AAR 和 ELA，因为它们是物质平衡的结果而非驱动因素
geometry_features = []
potential_geom = ['AREA', 'LOWER_BOUND', 'UPPER_BOUND']  # 移除了 'ELA', 'AAR'
for feat in potential_geom:
    if feat in df_clean.columns:
        geometry_features.append(feat)

# 5. 时间特征
time_feature = ['YEAR']

# 合并所有特征
all_features = geography_features + geometry_features + climate_annual + climate_summer + time_feature

print(f"\n  特征类别:")
print(f"    地理位置         : {len(geography_features)} 个特征")
print(f"    冰川几何（无泄露）: {len(geometry_features)} 个特征")
print(f"    气候（年度）      : {len(climate_annual)} 个特征")
print(f"    气候（夏季）      : {len(climate_summer)} 个特征")
print(f"    时间             : {len(time_feature)} 个特征")
print(f"    总计             : {len(all_features)} 个特征")

# 填充特征缺失值（使用中位数）
print("\n  处理缺失值...")
for col in all_features:
    if col in df_clean.columns:
        missing_count = df_clean[col].isna().sum()
        if missing_count > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            print(f"    已填充 {missing_count} 个缺失值: {col}")

# 准备特征矩阵和目标变量
X = df_clean[all_features].copy()
y = df_clean['ANNUAL_BALANCE'].copy()
years = df_clean['YEAR'].copy()
glacier_ids = df_clean['WGMS_ID'].copy()

print(f"\n  最终数据集形状: X={X.shape}, y={y.shape}")

# ================= 3. RFECV 特征选择 =================

print("\n>>> 步骤 3: 运行 RFECV 特征选择...")
print("  这可能需要几分钟...")

# 使用随机森林和 RFECV 选择最优特征
rf_selector = RandomForestRegressor(
    n_estimators=100,
    max_depth=50,
    random_state=42,
    n_jobs=-1
)

rfecv = RFECV(
    estimator=rf_selector,
    step=1,
    cv=KFold(5, shuffle=True, random_state=42),
    scoring='neg_mean_squared_error',
    min_features_to_select=10,
    n_jobs=-1
)

rfecv.fit(X, y)

# 获取被选中的特征
selected_features = np.array(all_features)[rfecv.support_]
n_selected = len(selected_features)

print(f"\n  最优特征数量: {n_selected}")
print(f"\n  被选中的特征:")
for i, feat in enumerate(selected_features, 1):
    print(f"    {i:2d}. {feat}")

# ================= 4. 留一年交叉验证 =================

print("\n>>> 步骤 4: 运行留一年交叉验证...")
print("  为每个测试年份训练模型...")

results = {
    'y_true': [],
    'y_pred': [],
    'year': [],
    'glacier_id': []
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
        print(f"    年份 {test_year}: 跳过（仅 {n_test} 个样本）")
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

    print(f"    年份 {test_year}: n={n_test:3d}, R^2={year_r2:6.3f}, RMSE={year_rmse:6.3f}, MAE={year_mae:6.3f}")

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
print("整体模型性能（留一年交叉验证，无数据泄露）")
print("=" * 70)
print(f"  R^2（决定系数）            : {overall_r2:7.4f}")
print(f"  R（相关系数）              : {overall_r:7.4f}")
print(f"  RMSE（均方根误差）         : {overall_rmse:7.4f} m w.e.")
print(f"  MAE（平均绝对误差）        : {overall_mae:7.4f} m w.e.")
print(f"  Bias（平均预测误差）       : {bias:7.4f} m w.e.")
print(f"  测试样本总数               : {len(y_true):7d}")
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
    print(f"    {row['feature']:40s}: {row['importance']:.4f}")

# ================= 7. 保存结果 =================

print("\n>>> 步骤 7: 保存结果...")

# 保存预测值
pred_df = pd.DataFrame({
    'YEAR': results['year'],
    'WGMS_ID': results['glacier_id'],
    'ANNUAL_BALANCE_TRUE': y_true,
    'ANNUAL_BALANCE_PRED': y_pred,
    'RESIDUAL': y_true - y_pred
})
pred_df.to_csv(RESULTS_FILE, index=False, encoding='utf-8-sig')
print(f"  预测结果已保存: {RESULTS_FILE}")

# 保存指标
with open(METRICS_FILE, 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("随机森林模型 - 留一年交叉验证结果（无数据泄露）\n")
    f.write("=" * 70 + "\n\n")

    f.write("重要说明：\n")
    f.write("  本次分析排除了 AAR 和 ELA 特征\n")
    f.write("  原因：这些特征是物质平衡的结果，会导致数据泄露\n")
    f.write("  仅使用物理驱动因素：气候、地理、冰川面积、时间\n\n")

    f.write("整体指标:\n")
    f.write(f"  R^2   : {overall_r2:.4f}\n")
    f.write(f"  R     : {overall_r:.4f}\n")
    f.write(f"  RMSE : {overall_rmse:.4f} m w.e.\n")
    f.write(f"  MAE  : {overall_mae:.4f} m w.e.\n")
    f.write(f"  Bias : {bias:.4f} m w.e.\n")
    f.write(f"  N    : {len(y_true)}\n\n")

    f.write(f"选中的特征数量: {n_selected}\n\n")

    f.write("被选中的特征:\n")
    for i, feat in enumerate(selected_features, 1):
        f.write(f"  {i:2d}. {feat}\n")

    f.write("\n" + "=" * 70 + "\n")
    f.write("特征重要性（前 20）:\n")
    f.write("=" * 70 + "\n")
    for i, row in feature_importance_df.head(20).iterrows():
        f.write(f"  {row['feature']:40s}: {row['importance']:.4f}\n")

    f.write("\n" + "=" * 70 + "\n")
    f.write("逐年性能:\n")
    f.write("=" * 70 + "\n")
    for metric in yearly_metrics:
        f.write(f"  年份 {metric['year']}: n={metric['n_samples']:3d}, "
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
    if feat in geography_features or feat in geometry_features:
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

plt.title(f'特征重要性（随机森林 LOOCV，无数据泄露）\nR^2 = {overall_r2:.4f}, RMSE = {overall_rmse:.3f} m w.e.',
          fontsize=14, fontweight='bold')
plt.xlabel('重要性得分', fontsize=12)
plt.ylabel('特征', fontsize=12)
plt.tight_layout()

# 保存图表
fig1_path = os.path.join(OUTPUT_DIR, 'rf_loocv_feature_importance_no_leakage.png')
plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
print(f"  特征重要性图已保存: {fig1_path}")
plt.show()

# 图 2: 预测 vs 观测
plt.figure(figsize=(10, 10))

plt.scatter(y_true, y_pred, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

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

plt.xlabel('观测年物质平衡 (m w.e.)', fontsize=12)
plt.ylabel('预测年物质平衡 (m w.e.)', fontsize=12)
plt.title(f'随机森林 LOOCV 性能（无数据泄露）\nR^2 = {overall_r2:.4f}, RMSE = {overall_rmse:.3f} m w.e., n = {len(y_true)}',
          fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()

# 保存图表
fig2_path = os.path.join(OUTPUT_DIR, 'rf_loocv_scatter_no_leakage.png')
plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
print(f"  散点图已保存: {fig2_path}")
plt.show()

# 图 3: 残差图
plt.figure(figsize=(12, 6))

residuals = y_true - y_pred

plt.subplot(1, 2, 1)
plt.scatter(y_pred, residuals, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.xlabel('预测年物质平衡 (m w.e.)', fontsize=12)
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
fig3_path = os.path.join(OUTPUT_DIR, 'rf_loocv_residuals_no_leakage.png')
plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
print(f"  残差图已保存: {fig3_path}")
plt.show()

# ================= 最终总结 =================

print("\n" + "=" * 70)
print("模型评估完成")
print("=" * 70)
print("\n输出文件:")
print(f"  预测结果     : {RESULTS_FILE}")
print(f"  性能指标     : {METRICS_FILE}")
print(f"  特征重要性图 : {fig1_path}")
print(f"  散点图       : {fig2_path}")
print(f"  残差图       : {fig3_path}")
print("\n" + "=" * 70)
print("关键结果（无数据泄露）:")
print(f"  R^2   = {overall_r2:.4f}")
print(f"  RMSE = {overall_rmse:.3f} m w.e.")
print(f"  MAE  = {overall_mae:.3f} m w.e.")
print(f"  Bias = {bias:.3f} m w.e.")
print("=" * 70)
print("\n说明：本次分析排除了 AAR 和 ELA，仅使用物理驱动因素")
