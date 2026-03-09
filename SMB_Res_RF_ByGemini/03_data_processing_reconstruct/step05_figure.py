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
csv_path = os.path.join(RECONSTRUCTION_DIR, 'RGI02_Hybrid_Dataset.csv')
print(f">>> 读取数据: {csv_path}")
df = pd.read_csv(csv_path)

# ================= 2. 方案 A: 单个典型冰川展示 =================
# 自动寻找一个“既有观测又有预测”的完美示例冰川
# 逻辑：找一个观测数据点在 10-40 个之间的冰川，这样既有蓝点又有红点，最好看
obs_counts = df[df['DATA_SOURCE'] == 'observed'].groupby('WGMS_ID').size()
candidates = obs_counts[(obs_counts > 10) & (obs_counts < 40)].index.tolist()

if candidates:
    sample_id = candidates[0] # 选第一个候选者
else:
    sample_id = obs_counts.idxmax() # 如果没有中间状态的，就选观测最多的

sample_data = df[df['WGMS_ID'] == sample_id].sort_values('YEAR')
glacier_name = sample_data['NAME'].iloc[0] if pd.notnull(sample_data['NAME'].iloc[0]) else f"Glacier {sample_id}"

print(f">>> 正在绘制示例冰川: {glacier_name} (ID: {sample_id})")

plt.figure(figsize=(12, 6), dpi=150)
sns.set_style("whitegrid")

# 1. 画一条灰色的连线表示整体趋势 (连接所有点)
plt.plot(sample_data['YEAR'], sample_data['SMB_m'], color='gray', linestyle='-', linewidth=1, alpha=0.5, zorder=1)

# 2. 画红色点 (Reconstructed / Filled)
# 筛选出填补的数据
recon_points = sample_data[sample_data['DATA_SOURCE'].isin(['predicted_filled', 'predicted_only'])]
plt.scatter(recon_points['YEAR'], recon_points['SMB_m'], 
            color='#d62728', label='Reconstructed (Gap-filled)', 
            s=50, marker='x', zorder=2)

# 3. 画蓝色点 (Observed) - 放在最上层
# 筛选出真实观测数据
obs_points = sample_data[sample_data['DATA_SOURCE'] == 'observed']
plt.scatter(obs_points['YEAR'], obs_points['SMB_m'], 
            color='#1f77b4', label='Observed Data', 
            s=80, marker='o', edgecolors='white', linewidth=1.5, zorder=3)

# 装饰
plt.title(f'Mass Balance Reconstruction: {glacier_name}', fontsize=16, fontweight='normal', pad=15)
plt.ylabel('Mass Balance (m w.e.)', fontsize=12)
plt.xlabel('Year', fontsize=12)
plt.axhline(0, color='black', linewidth=1, linestyle='--')
plt.legend(loc='lower left', frameon=True, fancybox=True, framealpha=0.9)
plt.grid(True, linestyle=':', alpha=0.6)


plt.show()

# ================= 3. 方案 B: 全区域平均趋势 =================
print(">>> 正在绘制全区域平均趋势...")

# 计算每年的平均值和标准差
yearly_stats = df.groupby('YEAR')['SMB_m'].agg(['mean', 'std']).reset_index()

plt.figure(figsize=(12, 6), dpi=150)

# 1. 绘制阴影 (区域变异性)
plt.fill_between(yearly_stats['YEAR'], 
                 yearly_stats['mean'] - yearly_stats['std'], 
                 yearly_stats['mean'] + yearly_stats['std'], 
                 color='#aec7e8', alpha=0.4, label='Regional Variability (±1 std)')

# 2. 绘制平均线
plt.plot(yearly_stats['YEAR'], yearly_stats['mean'], 
         color='#1f77b4', linewidth=2.5, label='Regional Mean SMB')

# 3. 添加趋势线
z = np.polyfit(yearly_stats['YEAR'], yearly_stats['mean'], 1)
p = np.poly1d(z)
trend = z[0] * 10
plt.plot(yearly_stats['YEAR'], p(yearly_stats['YEAR']), 
         "r--", linewidth=2, label=f'Trend: {trend:.2f} m w.e./decade')

# 装饰
plt.title('Reconstructed Regional Mass Balance (RGI 02)', fontsize=16, fontweight='bold', pad=15)
plt.ylabel('Surface Mass Balance (m w.e.)', fontsize=12)
plt.xlabel('Year', fontsize=12)
plt.axhline(0, color='black', linewidth=1, linestyle='--')
plt.legend(loc='lower left')
plt.grid(True, linestyle=':', alpha=0.6)


plt.show()

print("✅ 绘图完成！")