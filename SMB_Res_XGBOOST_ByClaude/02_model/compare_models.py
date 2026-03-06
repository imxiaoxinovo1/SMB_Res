"""
三模型对比: RandomForest vs XGBoost vs LightGBM
- 读取各模型的 LOYO 和 LOGO 预测结果
- 计算统一指标表
- 生成对比图
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from config import MODEL_RESULTS_DIR, RF_RESULTS_DIR

os.makedirs(MODEL_RESULTS_DIR, exist_ok=True)


def compute_metrics(y_true, y_pred):
    """计算回归评估指标"""
    if len(y_true) < 2:
        return {'R2': np.nan, 'R': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'Bias': np.nan}
    return {
        'R2': r2_score(y_true, y_pred),
        'R': np.corrcoef(y_true, y_pred)[0, 1],
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'Bias': np.mean(y_pred - y_true),
    }


# ================= 1. 加载各模型结果 =================
print(">>> 1. 加载各模型预测结果...")

models = {}

# RF (来自 RF 项目)
rf_loyo_path = os.path.join(RF_RESULTS_DIR, "loyo_predictions.csv")
rf_logo_path = os.path.join(RF_RESULTS_DIR, "logo_predictions.csv")
if os.path.exists(rf_loyo_path):
    models['RF'] = {
        'loyo': pd.read_csv(rf_loyo_path),
        'logo': pd.read_csv(rf_logo_path),
    }
    print(f"   RF:  LOYO={len(models['RF']['loyo'])}, LOGO={len(models['RF']['logo'])}")
else:
    print(f"   WARNING: RF 结果未找到: {rf_loyo_path}")

# XGBoost
xgb_loyo_path = os.path.join(MODEL_RESULTS_DIR, "xgb_loyo_predictions.csv")
xgb_logo_path = os.path.join(MODEL_RESULTS_DIR, "xgb_logo_predictions.csv")
if os.path.exists(xgb_loyo_path):
    models['XGBoost'] = {
        'loyo': pd.read_csv(xgb_loyo_path),
        'logo': pd.read_csv(xgb_logo_path),
    }
    print(f"   XGB: LOYO={len(models['XGBoost']['loyo'])}, LOGO={len(models['XGBoost']['logo'])}")
else:
    print(f"   WARNING: XGBoost 结果未找到: {xgb_loyo_path}")

# LightGBM
lgb_loyo_path = os.path.join(MODEL_RESULTS_DIR, "lgb_loyo_predictions.csv")
lgb_logo_path = os.path.join(MODEL_RESULTS_DIR, "lgb_logo_predictions.csv")
if os.path.exists(lgb_loyo_path):
    models['LightGBM'] = {
        'loyo': pd.read_csv(lgb_loyo_path),
        'logo': pd.read_csv(lgb_logo_path),
    }
    print(f"   LGB: LOYO={len(models['LightGBM']['loyo'])}, LOGO={len(models['LightGBM']['logo'])}")
else:
    print(f"   WARNING: LightGBM 结果未找到: {lgb_loyo_path}")

if len(models) == 0:
    print("ERROR: 未找到任何模型结果，请先运行训练脚本")
    sys.exit(1)

# ================= 2. 计算统一指标 =================
print("\n>>> 2. 计算统一指标...")

comparison_rows = []
for model_name, data in models.items():
    for cv_type in ['loyo', 'logo']:
        df_p = data[cv_type]
        yt = df_p['y_test'].values
        yp = df_p['y_pred'].values
        metrics = compute_metrics(yt, yp)
        metrics['Model'] = model_name
        metrics['CV_Type'] = cv_type.upper()
        metrics['N'] = len(yt)
        comparison_rows.append(metrics)

df_comparison = pd.DataFrame(comparison_rows)
col_order = ['Model', 'CV_Type', 'N', 'R2', 'R', 'RMSE', 'MAE', 'Bias']
df_comparison = df_comparison[col_order]

print("\n   模型对比表:")
print("   " + "-" * 80)
print(f"   {'Model':<10} {'CV':<6} {'N':>5} {'R2':>7} {'R':>7} {'RMSE':>7} {'MAE':>7} {'Bias':>7}")
print("   " + "-" * 80)
for _, row in df_comparison.iterrows():
    print(f"   {row['Model']:<10} {row['CV_Type']:<6} {int(row['N']):>5} "
          f"{row['R2']:>7.4f} {row['R']:>7.4f} {row['RMSE']:>7.4f} {row['MAE']:>7.4f} {row['Bias']:>7.4f}")

df_comparison.to_csv(os.path.join(MODEL_RESULTS_DIR, "model_comparison.csv"), index=False)
print(f"\n   已保存: model_comparison.csv")

# ================= 3. 对比图 =================
print("\n>>> 3. 生成对比图...")

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 150,
})

model_colors = {'RF': '#2ca02c', 'XGBoost': '#d62728', 'LightGBM': '#1f77b4'}

# --- 图1: 三模型 LOYO 散点图 ---
n_models = len(models)
fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6))
if n_models == 1:
    axes = [axes]

for ax, (model_name, data) in zip(axes, models.items()):
    df_p = data['loyo']
    yt = df_p['y_test'].values
    yp = df_p['y_pred'].values
    metrics = compute_metrics(yt, yp)

    color = model_colors.get(model_name, '#333333')
    ax.scatter(yt, yp, alpha=0.4, s=20, color=color, edgecolors='none')

    lim_min = min(yt.min(), yp.min()) - 0.5
    lim_max = max(yt.max(), yp.max()) + 0.5
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', linewidth=1.5)

    ax.set_xlabel('Observed SMB (m w.e.)')
    ax.set_ylabel('Predicted SMB (m w.e.)')
    ax.set_title(f'{model_name} - LOYO')
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.5)

    text = (f'R$^2$ = {metrics["R2"]:.3f}\nR = {metrics["R"]:.3f}\n'
            f'RMSE = {metrics["RMSE"]:.3f}\nMAE = {metrics["MAE"]:.3f}\n'
            f'Bias = {metrics["Bias"]:.3f}\nn = {len(yt)}')
    ax.text(0.05, 0.95, text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

fig.tight_layout()
fig.savefig(os.path.join(MODEL_RESULTS_DIR, "comparison_loyo_scatter.png"))
plt.close(fig)
print("   OK: comparison_loyo_scatter.png")

# --- 图2: 指标对比柱状图 ---
df_loyo = df_comparison[df_comparison['CV_Type'] == 'LOYO'].copy()
metrics_to_plot = ['R2', 'RMSE', 'MAE']

fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(5 * len(metrics_to_plot), 5))

for ax, metric in zip(axes, metrics_to_plot):
    model_names = df_loyo['Model'].values
    values = df_loyo[metric].values
    colors = [model_colors.get(m, '#333333') for m in model_names]

    bars = ax.bar(model_names, values, color=colors, edgecolor='white', linewidth=1.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel(metric)
    ax.set_title(f'LOYO - {metric}')
    ax.grid(axis='y', linestyle=':', alpha=0.5)

fig.suptitle('Model Comparison (LOYO)', fontsize=15, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(MODEL_RESULTS_DIR, "comparison_metrics_bar.png"), bbox_inches='tight')
plt.close(fig)
print("   OK: comparison_metrics_bar.png")

# --- 图3: 逐年 RMSE 对比折线图 ---
fig, ax = plt.subplots(figsize=(14, 6))

for model_name, data in models.items():
    df_p = data['loyo']
    # 逐年计算 RMSE
    yearly_rmse = []
    for year in sorted(df_p['year'].unique()):
        yr_data = df_p[df_p['year'] == year]
        if len(yr_data) >= 2:
            rmse = np.sqrt(mean_squared_error(yr_data['y_test'], yr_data['y_pred']))
            yearly_rmse.append({'year': year, 'rmse': rmse})
    df_yr = pd.DataFrame(yearly_rmse)

    color = model_colors.get(model_name, '#333333')
    ax.plot(df_yr['year'], df_yr['rmse'], marker='o', markersize=4,
            linewidth=1.5, color=color, label=model_name, alpha=0.8)

ax.set_xlabel('Year')
ax.set_ylabel('RMSE (m w.e.)')
ax.set_title('Year-by-Year RMSE Comparison (LOYO)')
ax.legend(loc='upper right', frameon=True)
ax.grid(True, linestyle=':', alpha=0.5)
fig.tight_layout()
fig.savefig(os.path.join(MODEL_RESULTS_DIR, "comparison_yearly_rmse.png"))
plt.close(fig)
print("   OK: comparison_yearly_rmse.png")

# ================= 4. 确定最优模型 =================
print("\n>>> 4. 最优模型选择...")
df_loyo_sorted = df_loyo.sort_values('R2', ascending=False)
best_model = df_loyo_sorted.iloc[0]['Model']
best_r2 = df_loyo_sorted.iloc[0]['R2']
print(f"   基于 LOYO R2，最优模型: {best_model} (R2={best_r2:.4f})")

# 保存最优模型信息
with open(os.path.join(MODEL_RESULTS_DIR, "best_model.txt"), 'w') as f:
    f.write(f"{best_model}\n")
print(f"   已保存最优模型名称到: best_model.txt")

print("\n>>> 模型对比完成!")
