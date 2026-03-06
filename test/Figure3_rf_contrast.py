import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import os

# ---------------------- 1. 配置文件路径与输出设置 ----------------------
# (修改) 定义两个输入文件
rf_file_original = r'H:\Code\SMB\test\result\pred_result_time_wna.csv'
rf_file_improved = r'H:\Code\SMB\test\result\pred_result_time_wna_with_mean_ela.csv'

output_dir = r'H:\Code\SMB\test\result'
# (修改) 新的输出文件名
output_filename = 'RF_Comparison_SMB_prediction.png' 

# 颜色和线条样式配置
cmap = plt.cm.viridis
split_point = 0
first_segment_style = 'k-'
second_segment_style = 'k-' # 您原来两段都是 'k-'，保持一致


# ---------------------- 2. 评估指标计算函数 ----------------------
def calculate_metrics(y_test, y_pred):
    # (修改) 确保 y_test 和 y_pred 长度一致，以防万一
    mask = ~np.isnan(y_test) & ~np.isnan(y_pred)
    y_test, y_pred = y_test[mask], y_pred[mask]
    
    r = np.corrcoef(y_test, y_pred)[0, 1]
    rmse = np.sqrt(np.mean((y_test - y_pred) **2))
    mae = np.mean(np.abs(y_test - y_pred))
    bias = np.mean(y_pred - y_test)
    return round(r, 2), round(rmse, 2), round(mae, 2), round(bias, 2)


# ---------------------- 3. (新增) 绘图辅助函数 ----------------------
def plot_density_scatter(ax, fig, y_test, y_pred, title, metric_text, axis_range):
    """
    在指定的 ax 上绘制密度散点图、1:1 线和指标文本。
    """
    # 计算点密度
    xy = np.vstack([y_test, y_pred])
    z = gaussian_kde(xy)(xy)
    idx_sorted = z.argsort() # 按密度排序

    # 绘制密度散点图
    scatter = ax.scatter(
        y_test[idx_sorted], y_pred[idx_sorted],
        c=z[idx_sorted],
        cmap=cmap,
        alpha=0.8,
        s=25,  # (修改) 子图的点可以稍小一些
        edgecolor='none'
    )

    # 添加分段1:1参考线
    ax.plot(
        [axis_range[0], split_point],
        [axis_range[0], split_point],
        first_segment_style,
        alpha=0.7,
        linewidth=1.5
    )
    ax.plot(
        [split_point, axis_range[1]],
        [split_point, axis_range[1]],
        second_segment_style,
        alpha=0.7,
        linewidth=1.5
    )

    # 标注评估指标
    ax.text(
        0.05, 0.95, metric_text,
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', pad=6),
        fontsize=11
    )

    # 设置坐标轴
    ax.set_xlim(axis_range)
    ax.set_ylim(axis_range)
    ax.set_xlabel('Ground truth SMB (m w.e. a⁻¹)', fontsize=12)
    ax.set_ylabel('Predicted SMB (m w.e. a⁻¹)', fontsize=12)
    ax.set_aspect('equal')
    ax.tick_params(axis='both', labelsize=10)
    
    # (修改) 添加子图标题
    ax.set_title(title, fontsize=13, fontweight='bold')
    
    # 添加颜色条
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Density', rotation=270, labelpad=20, fontsize=11)


# ---------------------- 4. 读取所有数据并计算全局范围 ----------------------
print("正在加载数据...")
# (修改) 加载两个文件
df_orig = pd.read_csv(rf_file_original)
y_test_orig = df_orig['y_test'].dropna().values
y_pred_orig = df_orig['y_pred'].dropna().values

df_impr = pd.read_csv(rf_file_improved)
y_test_impr = df_impr['y_test'].dropna().values
y_pred_impr = df_impr['y_pred'].dropna().values

# (修改) 计算全局坐标轴范围，确保两图可比
print("计算全局坐标轴...")
data_min = min(y_test_orig.min(), y_pred_orig.min(), y_test_impr.min(), y_pred_impr.min())
data_max = max(y_test_orig.max(), y_pred_orig.max(), y_test_impr.max(), y_pred_impr.max())
axis_range = [data_min - 0.3, data_max + 0.3]


# ---------------------- 5. 绘制并排对比图 ----------------------
print("开始绘制对比图...")
# (修改) 创建 1x2 子图
fig, axes = plt.subplots(1, 2, figsize=(16, 7)) # 宽度加倍

# --- 准备图 (a): 原始 RF ---
r_orig, rmse_orig, mae_orig, bias_orig = calculate_metrics(y_test_orig, y_pred_orig)
metric_text_orig = (
    f'R = {r_orig}\n'
    f'RMSE = {rmse_orig} m w.e. a⁻¹\n'
    f'MAE = {mae_orig} m w.e. a⁻¹\n'
    f'Bias = {bias_orig} m w.e. a⁻¹'
)
plot_density_scatter(
    axes[0], fig, y_test_orig, y_pred_orig,
    '(a) Original RF', 
    metric_text_orig, 
    axis_range
)

# --- 准备图 (b): 改进的 RF + Mean Elev. ---
r_impr, rmse_impr, mae_impr, bias_impr = calculate_metrics(y_test_impr, y_pred_impr)
metric_text_impr = (
    f'R = {r_impr}\n'
    f'RMSE = {rmse_impr} m w.e. a⁻¹\n'
    f'MAE = {mae_impr} m w.e. a⁻¹\n'
    f'Bias = {bias_impr} m w.e. a⁻¹'
)
plot_density_scatter(
    axes[1], fig, y_test_impr, y_pred_impr,
    '(b) RF + Mean Elevation', 
    metric_text_impr, 
    axis_range
)

# (修改) 添加一个总标题
fig.suptitle('Model Comparison: Predicted vs Ground Truth SMB', fontsize=16, fontweight='bold')

# (修改) 调整布局以适应总标题
plt.tight_layout(rect=[0, 0.03, 1, 0.95])


# ---------------------- 6. 保存图片 ----------------------
os.makedirs(output_dir, exist_ok=True)
full_save_path = os.path.join(output_dir, output_filename)
plt.savefig(full_save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"并排对比图已保存至：{full_save_path}")