import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import RECONSTRUCTION_DIR

# ================= 1. 读取数据 =================
# 路径由 config.py 统一管理
csv_path = os.path.join(RECONSTRUCTION_DIR, 'RGI02_reconstruction_corrected.csv')
df = pd.read_csv(csv_path)

# ================= 2. 数据计算 =================
# 按年份分组，计算全区域的平均值(mean)和标准差(std)
yearly_stats = df.groupby('YEAR')['Predicted_SMB_m'].agg(['mean', 'std']).reset_index()

# ================= 3. 开始绘图 =================
# 设置更美观的 Seaborn 风格
sns.set_theme(style="ticks", font_scale=1.1)
plt.figure(figsize=(12, 6))

# 1. 绘制阴影区域 (表示区域内的变异性 ±1倍标准差)
plt.fill_between(
    yearly_stats['YEAR'], 
    yearly_stats['mean'] - yearly_stats['std'], 
    yearly_stats['mean'] + yearly_stats['std'], 
    color="#9bd0ec", alpha=0.5, label='Regional Variability (±1 std)'
)

# 2. 绘制年均值主曲线
plt.plot(yearly_stats['YEAR'], yearly_stats['mean'], 
         color='#1f78b4', linewidth=2.5, marker='o', markersize=4, label='Regional Mean SMB')

# 3. 添加 0 平衡线 (参考线)
plt.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.8)

# 4. 添加线性趋势线 (红虚线)
# 计算斜率
z = np.polyfit(yearly_stats['YEAR'], yearly_stats['mean'], 1)
p = np.poly1d(z)
trend_slope = z[0] * 10  # 换算成每10年的变化率
plt.plot(yearly_stats['YEAR'], p(yearly_stats['YEAR']), "r--", linewidth=2, 
         label=f'Trend: {trend_slope:.3f} m w.e./decade')

# ================= 4. 图表美化 =================
plt.title('Reconstructed Mass Balance of Western North America (RGI 02)', fontsize=16, fontweight='bold', pad=15)
plt.ylabel('Surface Mass Balance (m w.e.)', fontsize=12)
plt.xlabel('Year', fontsize=12)

# 设置图例
plt.legend(loc='lower left', frameon=True, framealpha=0.9, fancybox=True)

# 设置坐标轴范围
plt.xlim(yearly_stats['YEAR'].min(), yearly_stats['YEAR'].max())
plt.grid(True, linestyle=':', alpha=0.6)


plt.tight_layout()
plt.show()