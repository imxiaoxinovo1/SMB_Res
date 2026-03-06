import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import os

# ---------------------- 1. 配置文件路径与输出设置 ----------------------
# 仅保留随机森林的数据文件
rf_file = 'test/result/pred_result_time_wna.csv'  # 随机森林数据路径
output_dir = r'H:\Code\SMB\test\result'  # 输出路径
output_filename = 'RF_SMB_prediction.png'  # 单张图文件名

# 颜色和线条样式配置
cmap = plt.cm.viridis
# 分段参考线设置（根据论文需求调整分段点）
split_point = 0  # 例如：SMB=0处分段
first_segment_style = 'k-'  # 第一段（低值区）实线
second_segment_style = 'k-'  # 第二段（高值区）虚线


# ---------------------- 2. 评估指标计算函数 ----------------------
def calculate_metrics(y_test, y_pred):
    r = np.corrcoef(y_test, y_pred)[0, 1]
    rmse = np.sqrt(np.mean((y_test - y_pred) **2))
    mae = np.mean(np.abs(y_test - y_pred))
    bias = np.mean(y_pred - y_test)
    return round(r, 2), round(rmse, 2), round(mae, 2), round(bias, 2)


# ---------------------- 3. 读取随机森林数据 ----------------------
df = pd.read_csv(rf_file)
y_test = df['y_test'].dropna().values  # 真实SMB
y_pred = df['y_pred'].dropna().values  # 预测SMB

# 计算坐标轴范围（基于数据极值扩展，避免点贴边）
data_min = min(y_test.min(), y_pred.min())
data_max = max(y_test.max(), y_pred.max())
axis_range = [data_min - 0.3, data_max + 0.3]


# ---------------------- 4. 绘制单张密度散点图 ----------------------
plt.figure(figsize=(8, 7))  # 单图尺寸，比子图更大更清晰
ax = plt.gca()

# 计算点密度（颜色深浅核心）
xy = np.vstack([y_test, y_pred])
z = gaussian_kde(xy)(xy)
idx_sorted = z.argsort()  # 按密度排序，确保密集点在上方

# 绘制密度散点图
scatter = ax.scatter(
    y_test[idx_sorted], y_pred[idx_sorted],
    c=z[idx_sorted],
    cmap=cmap,
    alpha=0.8,
    s=40,  # 点大小可适当增大，单图更清晰
    edgecolor='none'
)

# 添加分段1:1参考线（虚实结合）
# 第一段：从最小值到分段点（实线）
ax.plot(
    [axis_range[0], split_point],
    [axis_range[0], split_point],
    first_segment_style,
    alpha=0.7,
    linewidth=1.5
)
# 第二段：从分段点到最大值（虚线）
ax.plot(
    [split_point, axis_range[1]],
    [split_point, axis_range[1]],
    second_segment_style,
    alpha=0.7,
    linewidth=1.5
)

# 标注评估指标（左上角，突出显示）
r, rmse, mae, bias = calculate_metrics(y_test, y_pred)
metric_text = (
    'Random Forest\n'
    f'R = {r}\n'
    f'RMSE = {rmse} m w.e. a⁻¹\n'
    f'MAE = {mae} m w.e. a⁻¹\n'
    f'Bias = {bias} m w.e. a⁻¹'
)
ax.text(
    0.05, 0.95, metric_text,
    transform=ax.transAxes,
    verticalalignment='top',
    bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', pad=6),  # 灰色边框更突出
    fontsize=11
)

# 设置坐标轴
ax.set_xlim(axis_range)
ax.set_ylim(axis_range)
ax.set_xlabel('Ground truth SMB (m w.e. a⁻¹)', fontsize=12)
ax.set_ylabel('Predicted SMB (m w.e. a⁻¹)', fontsize=12)
ax.set_aspect('equal')  # 确保1:1线为45度
ax.tick_params(axis='both', labelsize=10)  # 刻度字体大小


# ---------------------- 5. 添加颜色条与标题 ----------------------
cbar = plt.colorbar(scatter)
cbar.set_label('Density', rotation=270, labelpad=20, fontsize=11)

plt.title(
    'Random Forest: Predicted vs Ground Truth SMB',
    fontsize=13,
    fontweight='bold'
)

# 调整布局（避免标签被截断）
plt.tight_layout()


# ---------------------- 6. 保存图片 ----------------------
os.makedirs(output_dir, exist_ok=True)
full_save_path = os.path.join(output_dir, output_filename)
plt.savefig(full_save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"随机森林单张图已保存至：{full_save_path}")