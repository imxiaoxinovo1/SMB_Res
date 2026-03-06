"""
Step 03: 可视化分析
- 图1: 区域平均 SMB 趋势
- 图2: 典型冰川时间序列 (观测 vs 重建)
- 图3: 数据覆盖热图
- 图4: LOYO + LOGO 验证散点图
- 图5: 特征重要性
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from config import MODEL_RESULTS_DIR, RECON_RESULTS_DIR, RECON_FIGURES_DIR

os.makedirs(RECON_FIGURES_DIR, exist_ok=True)

# ================= 加载数据 =================
print(">>> 加载数据...")
hybrid_df = pd.read_csv(os.path.join(RECON_RESULTS_DIR, "RGI02_Hybrid_Dataset.csv"))
loyo_preds = pd.read_csv(os.path.join(MODEL_RESULTS_DIR, "loyo_predictions.csv"))
logo_preds = pd.read_csv(os.path.join(MODEL_RESULTS_DIR, "logo_predictions.csv"))
feat_imp = pd.read_csv(os.path.join(MODEL_RESULTS_DIR, "feature_importance.csv"))

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 150,
})

# ================= 图1: 区域平均 SMB 趋势 =================
print(">>> 绘制图1: 区域平均 SMB 趋势...")
yearly = hybrid_df.groupby('YEAR')['SMB_m'].agg(['mean', 'std']).reset_index()

fig, ax = plt.subplots(figsize=(12, 6))
ax.fill_between(yearly['YEAR'],
                yearly['mean'] - yearly['std'],
                yearly['mean'] + yearly['std'],
                color='#aec7e8', alpha=0.4, label='Regional Variability ($\\pm$1 std)')
ax.plot(yearly['YEAR'], yearly['mean'],
        color='#1f78b4', linewidth=2.5, label='Regional Mean SMB')

# 线性趋势
z = np.polyfit(yearly['YEAR'], yearly['mean'], 1)
p = np.poly1d(z)
trend_decade = z[0] * 10
ax.plot(yearly['YEAR'], p(yearly['YEAR']), 'r--', linewidth=2,
        label=f'Trend: {trend_decade:.3f} m w.e./decade')

ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)
ax.set_xlabel('Year')
ax.set_ylabel('Surface Mass Balance (m w.e.)')
ax.set_title('Reconstructed Mass Balance - Western North America (RGI 02)')
ax.legend(loc='lower left', frameon=True)
ax.grid(True, linestyle=':', alpha=0.5)
fig.tight_layout()
fig.savefig(os.path.join(RECON_FIGURES_DIR, "fig1_regional_smb_trend.png"))
plt.close(fig)
print("   OK")

# ================= 图2: 典型冰川时间序列 =================
print(">>> 绘制图2: 典型冰川时间序列...")

# 找一个观测数据量适中的冰川
obs_counts = hybrid_df[hybrid_df['DATA_SOURCE'] == 'observed'].groupby('WGMS_ID').size()
candidates = obs_counts[(obs_counts > 15) & (obs_counts < 60)]
if len(candidates) > 0:
    sample_id = candidates.index[0]
else:
    sample_id = obs_counts.idxmax()

g_data = hybrid_df[hybrid_df['WGMS_ID'] == sample_id].sort_values('YEAR')
glacier_name = g_data['NAME'].dropna().iloc[0] if not g_data['NAME'].dropna().empty else f'Glacier {sample_id}'

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(g_data['YEAR'], g_data['SMB_m'], color='gray', linewidth=1, alpha=0.5, zorder=1)

recon_pts = g_data[g_data['DATA_SOURCE'] != 'observed']
obs_pts = g_data[g_data['DATA_SOURCE'] == 'observed']

ax.scatter(recon_pts['YEAR'], recon_pts['SMB_m'],
           color='#d62728', marker='x', s=50, label='Reconstructed (Gap-filled)', zorder=2)
ax.scatter(obs_pts['YEAR'], obs_pts['SMB_m'],
           color='#1f77b4', marker='o', s=80, edgecolors='white', linewidth=1.5,
           label='Observed', zorder=3)

ax.axhline(0, color='black', linestyle='--', linewidth=1)
ax.set_xlabel('Year')
ax.set_ylabel('Mass Balance (m w.e.)')
ax.set_title(f'Mass Balance Reconstruction: {glacier_name}')
ax.legend(loc='lower left', frameon=True)
ax.grid(True, linestyle=':', alpha=0.5)
fig.tight_layout()
fig.savefig(os.path.join(RECON_FIGURES_DIR, "fig2_glacier_example.png"))
plt.close(fig)
print(f"   OK (示例冰川: {glacier_name})")

# ================= 图3: 数据覆盖热图 =================
print(">>> 绘制图3: 数据覆盖热图...")

# 仅使用有名字的冰川
df_named = hybrid_df.dropna(subset=['NAME']).copy()
if len(df_named) > 0:
    # 编码 DATA_SOURCE: observed=2, predicted_filled=1, predicted_only=0
    source_map = {'observed': 2, 'predicted_filled': 1, 'predicted_only': 0}
    df_named['source_code'] = df_named['DATA_SOURCE'].map(source_map)

    matrix = df_named.pivot_table(index='NAME', columns='YEAR',
                                  values='source_code', aggfunc='first')
    # 按观测量排序
    obs_score = (matrix == 2).sum(axis=1)
    matrix = matrix.loc[obs_score.sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(16, max(8, len(matrix) * 0.25)))
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['#f0f0f0', '#fdae6b', '#2b8cbe'])

    im = ax.imshow(matrix.fillna(-1).values, aspect='auto', cmap=cmap,
                   vmin=0, vmax=2, interpolation='nearest')

    # X 轴
    years = matrix.columns.values
    xtick_pos = np.arange(0, len(years), 5)
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(years[xtick_pos], fontsize=9)

    # Y 轴
    ax.set_yticks(range(len(matrix)))
    ax.set_yticklabels(matrix.index, fontsize=8)

    ax.set_xlabel('Year')
    ax.set_ylabel('Glacier')
    ax.set_title('Data Coverage: Observed vs Reconstructed')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2b8cbe', label='Observed'),
        Patch(facecolor='#fdae6b', label='Predicted (Fill)'),
        Patch(facecolor='#f0f0f0', edgecolor='gray', label='Predicted Only'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(RECON_FIGURES_DIR, "fig3_coverage_heatmap.png"))
    plt.close(fig)
    print("   OK")
else:
    print("   SKIP: 没有有名字的冰川")

# ================= 图4: 验证散点图 (LOYO + LOGO) =================
print(">>> 绘制图4: 验证散点图...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, df_p, title in [
    (axes[0], loyo_preds, 'Leave-One-Year-Out (LOYO)'),
    (axes[1], logo_preds, 'Leave-One-Glacier-Out (LOGO)')
]:
    yt = df_p['y_test'].values
    yp = df_p['y_pred'].values

    r2 = r2_score(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    mae = mean_absolute_error(yt, yp)
    bias = np.mean(yp - yt)
    r = np.corrcoef(yt, yp)[0, 1]

    ax.scatter(yt, yp, alpha=0.4, s=20, color='#1f78b4', edgecolors='none')

    # 1:1 线
    lim_min = min(yt.min(), yp.min()) - 0.5
    lim_max = max(yt.max(), yp.max()) + 0.5
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', linewidth=1.5, label='1:1 Line')

    ax.set_xlabel('Observed SMB (m w.e.)')
    ax.set_ylabel('Predicted SMB (m w.e.)')
    ax.set_title(title)
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.5)

    text = f'R$^2$ = {r2:.3f}\nR = {r:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}\nBias = {bias:.3f}\nn = {len(yt)}'
    ax.text(0.05, 0.95, text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

fig.tight_layout()
fig.savefig(os.path.join(RECON_FIGURES_DIR, "fig4_validation_scatter.png"))
plt.close(fig)
print("   OK")

# ================= 图5: 特征重要性 =================
print(">>> 绘制图5: 特征重要性...")

fig, ax = plt.subplots(figsize=(10, 7))
feat_sorted = feat_imp.sort_values('importance', ascending=True)

colors = ['#d62728' if 'BOUND' in f else '#4292c6' for f in feat_sorted['feature']]
ax.barh(range(len(feat_sorted)), feat_sorted['importance'], color=colors)
ax.set_yticks(range(len(feat_sorted)))
ax.set_yticklabels(feat_sorted['feature'], fontsize=10)
ax.set_xlabel('Importance Score')
ax.set_title('Random Forest Feature Importance')

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#d62728', label='Elevation Features'),
    Patch(facecolor='#4292c6', label='Other Features'),
]
ax.legend(handles=legend_elements, loc='lower right')
fig.tight_layout()
fig.savefig(os.path.join(RECON_FIGURES_DIR, "fig5_feature_importance.png"))
plt.close(fig)
print("   OK")

print(f"\n>>> 所有图表已保存到: {RECON_FIGURES_DIR}")
print(">>> 完成!")
