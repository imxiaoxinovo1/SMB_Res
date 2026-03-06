import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import os
import matplotlib.gridspec as gridspec  # 新增：用于更精确的布局控制

# ---------------------- 1. 配置文件路径与模型信息 ----------------------
model_files = {
    'MLR': 'test/result/pred_result_time_wna_mlr.csv',
    'Lasso': 'test/result/pred_result_time_wna_lasso.csv',
    'Bayesian': 'test/result/pred_result_time_wna_bayesian.csv',
    'Random Forest': 'test/result/pred_result_time_wna.csv'
}

# 定义输出路径和文件名
output_dir = 'H:/Code/SMB/test/result'
output_filename = 'Figure3.png'

# 颜色和线条样式配置
cmap = plt.cm.viridis
line_style_dict = {'MLR': 'k--', 'Lasso': 'k--', 'Bayesian': 'k--', 'Random Forest': 'k--'}


# ---------------------- 2. 评估指标计算函数 ----------------------
def calculate_metrics(y_test, y_pred):
    r = np.corrcoef(y_test, y_pred)[0, 1]
    rmse = np.sqrt(np.mean((y_test - y_pred) **2))
    mae = np.mean(np.abs(y_test - y_pred))
    bias = np.mean(y_pred - y_test)
    return round(r, 2), round(rmse, 2), round(mae, 2), round(bias, 2)


# ---------------------- 3. 读取数据并确定全局范围 ----------------------
all_data = {}
global_min = np.inf
global_max = -np.inf

for model_name, file_path in model_files.items():
    df = pd.read_csv(file_path)
    y_test = df['y_test'].dropna().values
    y_pred = df['y_pred'].dropna().values
    all_data[model_name] = {'y_test': y_test, 'y_pred': y_pred}
    
    current_min = min(y_test.min(), y_pred.min())
    current_max = max(y_test.max(), y_pred.max())
    global_min = min(global_min, current_min)
    global_max = max(global_max, current_max)

axis_range = [global_min - 0.3, global_max + 0.3]


# ---------------------- 4. 绘制2×2密度散点图（使用gridspec避免布局警告） ----------------------
fig = plt.figure(figsize=(12, 10))
# 使用gridspec替代默认布局，精确控制子图间距
gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.25, hspace=0.3)
axes = [
    fig.add_subplot(gs[0, 0]),  # 第一行第一列
    fig.add_subplot(gs[0, 1]),  # 第一行第二列
    fig.add_subplot(gs[1, 0]),  # 第二行第一列
    fig.add_subplot(gs[1, 1])   # 第二行第二列
]

density_scatter = None

for idx, (model_name, data) in enumerate(all_data.items()):
    ax = axes[idx]
    y_test = data['y_test']
    y_pred = data['y_pred']
    
    # 计算点密度
    xy = np.vstack([y_test, y_pred])
    z = gaussian_kde(xy)(xy)
    idx_sorted = z.argsort()
    y_test_sorted = y_test[idx_sorted]
    y_pred_sorted = y_pred[idx_sorted]
    z_sorted = z[idx_sorted]

    # 绘制密度散点图
    scatter = ax.scatter(
        y_test_sorted, y_pred_sorted,
        c=z_sorted,
        cmap=cmap,
        alpha=0.8,
        s=30,
        edgecolor='none'
    )
    if idx == 0:
        density_scatter = scatter

    # 添加1:1参考线
    ax.plot(
        axis_range, axis_range, 
        line_style_dict[model_name], 
        alpha=0.7,
        linewidth=1.5
    )

    # 标注评估指标
    r, rmse, mae, bias = calculate_metrics(y_test, y_pred)
    metric_text = (
        f'{model_name}\n'
        f'R = {r}\n'
        f'RMSE = {rmse}\n'
        f'MAE = {mae}\n'
        f'Bias = {bias}'
    )
    ax.text(
        0.05, 0.95, metric_text,
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=5),
        fontsize=10
    )

    # 设置坐标轴
    ax.set_xlim(axis_range)
    ax.set_ylim(axis_range)
    ax.set_xlabel('Ground truth SMB (m w.e. a⁻¹)', fontsize=11)
    ax.set_ylabel('Predicted SMB (m w.e. a⁻¹)', fontsize=11)
    ax.set_aspect('equal')


# ---------------------- 5. 添加全局颜色条与总标题 ----------------------
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = fig.colorbar(density_scatter, cax=cbar_ax)
cbar.set_label('Density', rotation=270, labelpad=20, fontsize=11)

plt.suptitle(
    'Evaluation of predicted SMB against the ground truth SMB data (both in m w.e. a⁻¹)',
    y=0.95,
    fontsize=14,
    fontweight='bold'
)

# 关键修改：使用subplots_adjust替代tight_layout，消除警告
plt.subplots_adjust(
    left=0.1,    # 左侧边距
    right=0.9,   # 右侧边距（预留颜色条空间）
    bottom=0.1,  # 底部边距
    top=0.9,     # 顶部边距（预留总标题空间）
    wspace=0.3,  # 子图间宽度间距
    hspace=0.3   # 子图间高度间距
)


# ---------------------- 6. 保存图片到指定路径 ----------------------
# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)
# 拼接完整路径
full_save_path = os.path.join(output_dir, output_filename)
# 保存图片
plt.savefig(full_save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"图片已成功保存至：{full_save_path}")
