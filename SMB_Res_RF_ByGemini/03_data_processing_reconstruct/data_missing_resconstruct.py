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

# ================= 2. 数据筛选 =================
# 1. 筛选时间范围 (1980-2024)
df = df[(df['YEAR'] >= 1980) & (df['YEAR'] <= 2024)]

# 2. 筛选有名字的冰川
df_named = df.dropna(subset=['NAME']).copy()
glacier_names = df_named['NAME'].unique()
print(f"   共找到 {len(glacier_names)} 条有名字的冰川")

# ================= 3. 绘图 (单张大图) =================
plt.figure(figsize=(14, 8), dpi=150)
sns.set_style("whitegrid")

# 为了区分不同冰川，虽然我们主要用红蓝区分数据源，
# 但也可以给每条冰川的灰色底线加一点点微弱的透明度区别，或者保持一致。
# 这里我们重点突出红蓝数据源。

# 循环绘制每一条冰川
for name in glacier_names:
    # 提取该冰川的数据
    data = df_named[df_named['NAME'] == name].sort_values('YEAR')
    
    # 1. 绘制底线 (灰色，表示连续性)
    # alpha=0.3 让线条不抢眼，作为背景
    plt.plot(data['YEAR'], data['SMB_m'], color='gray', linestyle='-', linewidth=1, alpha=0.3, zorder=1)
    
    # 2. 绘制重建数据 (红色部分)
    # 筛选出填补的数据
    recon_data = data[data['DATA_SOURCE'] != 'observed']
    plt.scatter(recon_data['YEAR'], recon_data['SMB_m'], 
                color='#d62728', s=20, marker='x', alpha=0.7, zorder=2)
    
    # 3. 绘制观测数据 (蓝色部分)
    # 筛选出真实观测数据
    obs_data = data[data['DATA_SOURCE'] == 'observed']
    plt.scatter(obs_data['YEAR'], obs_data['SMB_m'], 
                color='#1f77b4', s=25, marker='o', alpha=0.9, zorder=3)
    
    # 可选：如果你希望观测数据之间也连成蓝色线（强调实测片段）
    # 但这可能会让图变得稍微杂乱，只画点通常更清晰。
    # plt.plot(obs_data['YEAR'], obs_data['SMB_m'], color='#1f77b4', linewidth=1.5, alpha=0.5, zorder=3)

# ================= 4. 装饰图表 =================
plt.title('Mass Balance Series of All Named Glaciers (1980-2024)', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Specific Mass Balance (m w.e.)', fontsize=14)

# 添加 0 平衡线
plt.axhline(0, color='black', linewidth=1.5, linestyle='--')

# 自定义图例 (因为我们画了多次，自动图例会重复，所以手动创建)
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Observed Data', 
           markerfacecolor='#1f77b4', markersize=8),
    Line2D([0], [0], marker='x', color='#d62728', label='Reconstructed (Gap-filled)', 
           markerfacecolor='#d62728', markersize=8, linestyle='None'), # 修正：去掉了多余的linewidth参数
    Line2D([0], [0], color='gray', lw=1, label='Continuity Line', alpha=0.5)
]
plt.legend(handles=legend_elements, loc='lower left', fontsize=12, frameon=True, fancybox=True)

plt.xlim(1980, 2024)
plt.grid(True, linestyle=':', alpha=0.6)

# 保存


plt.show()