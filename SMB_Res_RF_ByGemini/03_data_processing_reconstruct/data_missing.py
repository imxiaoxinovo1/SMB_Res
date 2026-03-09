import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import STUDY_TEST_DIR

# ================= 1. 读取数据 =================
csv_path = os.path.join(STUDY_TEST_DIR, 'data_glacier_era5_matched.csv')

print(f">>> 读取数据: {csv_path}")
df = pd.read_csv(csv_path)

# ================= 2. 数据预处理 =================
# 1. 筛选时间范围 (1980-2024)
df = df[(df['YEAR'] >= 1980) & (df['YEAR'] <= 2024)]

# 2. 筛选有名字的冰川 (剔除 NaN)
if 'NAME' not in df.columns:
    print("❌ 错误：数据中缺少 'NAME' 列，无法绘制有名字的冰川！")
    exit()

df_named = df.dropna(subset=['NAME']).copy()
print(f"   在该时间段内有名字的观测记录数: {len(df_named)}")

# 3. 构建矩阵 (Pivot Table)
# 行=冰川名, 列=年份, 值=1(有数据)
# 我们只关心是否有数据，不关心数值大小，所以只要不为空就设为1
matrix = df_named.pivot_table(index='NAME', columns='YEAR', values='ANNUAL_BALANCE', aggfunc='count')

# 将非空值设为 1，空值设为 0 (或者 NaN)
matrix = matrix.notnull().astype(int)

# ================= 3. 关键步骤：排序 =================
# 为了让图好看，我们按照“数据量多少”对冰川进行排序
# 数据最多的冰川排在最上面，形成一个漂亮的倒三角形或阶梯状
data_counts = matrix.sum(axis=1)
sorted_names = data_counts.sort_values(ascending=False).index
matrix_sorted = matrix.loc[sorted_names]

print(f"   共包含 {len(matrix_sorted)} 条有名字的冰川")

# ================= 4. 绘图 =================
plt.figure(figsize=(14, 10), dpi=150)

# 使用 Seaborn 画热力图
# cbar=False: 不需要色标条 (因为只有0和1)
# cmap: 自定义颜色 (0=浅灰/缺失, 1=深蓝/有数据)
ax = sns.heatmap(matrix_sorted, 
                 cmap=['#f0f0f0', '#2b8cbe'], 
                 cbar=False, 
                 linewidths=0.5, 
                 linecolor='white',
                 square=False)

# 设置标题和标签
plt.title('Observational Data Availability of RGI02 Named Glaciers (1980-2024)', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Glacier Name', fontsize=14)

# 调整 X 轴标签 
xticks = np.arange(0, len(matrix.columns), 5)
xticklabels = matrix.columns[xticks]
plt.xticks(xticks + 0.5, xticklabels, rotation=0, fontsize=12)
plt.yticks(fontsize=10)

# 添加边框
for _, spine in ax.spines.items():
    spine.set_visible(True)
    spine.set_color('black')
    spine.set_linewidth(1)

# 图例
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2b8cbe', edgecolor='black', label='Data Available'),
    Patch(facecolor='#f0f0f0', edgecolor='black', label='Missing Data')
]
plt.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(1, 0.995), ncol=2.0, fontsize=10)

plt.tight_layout()

plt.show()
