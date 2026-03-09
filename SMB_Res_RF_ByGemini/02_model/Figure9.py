import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import ZEMP_CSV, OLD_RF_RESULT_CSV

# ==========================================
# 1. 处理红色线条数据 (Zemp et al.)
# ==========================================
# 路径由 config.py 统一管理
filename_zemp = ZEMP_CSV

# 跳过前27行说明，读取数据
df_zemp = pd.read_csv(filename_zemp, header=0, skiprows=27)
df_zemp.columns = df_zemp.columns.str.strip()

# 筛选 1980-2016 年
df_zemp_filt = df_zemp[(df_zemp['Year'] >= 1980) & (df_zemp['Year'] <= 2016)].copy()

# 计算 SLE (mm)
df_zemp_filt['SLE_mm_yr'] = df_zemp_filt['INT_Gt'] / 361.8

# 建立年份到区域总面积(Area_AW_ref_km2)的映射
area_map = df_zemp.set_index('Year')['Area_AW_ref_km2'].to_dict()
last_known_area = df_zemp_filt['Area_AW_ref_km2'].iloc[-1]

# 计算红线的拟合函数
slope_z, intercept_z, r_z, p_z, std_err_z = stats.linregress(df_zemp_filt['Year'], df_zemp_filt['SLE_mm_yr'])
line_z = slope_z * df_zemp_filt['Year'] + intercept_z
eq_z_text = f"y = {slope_z:.4f}x + {intercept_z:.2f}"

# ==========================================
# 2. 处理蓝色线条数据 (Our study)
# ==========================================
# 【修改点 2】: 同样在路径前加 r，解决 \t 被识别为 Tab 的问题
filename_pred = OLD_RF_RESULT_CSV

df_pred = pd.read_csv(filename_pred)

# 计算每年的平均 SMB 预测值
df_blue = df_pred.groupby('year')['y_pred'].mean().reset_index()

# 定义一个获取面积的函数
def get_area(year):
    return area_map.get(year, last_known_area)

# 核心换算
df_blue['Area_km2'] = df_blue['year'].apply(get_area)
df_blue['Mass_Gt'] = df_blue['y_pred'] * df_blue['Area_km2'] / 1000.0
df_blue['SLE_mm_yr'] = df_blue['Mass_Gt'] / 361.8

# 筛选绘图年份
df_blue_plot = df_blue[(df_blue['year'] >= 1980) & (df_blue['year'] <= 2020)].copy()

# 计算蓝线的拟合函数
slope_b, intercept_b, r_b, p_b, std_err_b = stats.linregress(df_blue_plot['year'], df_blue_plot['SLE_mm_yr'])
line_b = slope_b * df_blue_plot['year'] + intercept_b
eq_b_text = f"y = {slope_b:.4f}x + {intercept_b:.2f}"

# ==========================================
# 3. 绘图
# ==========================================
plt.figure(figsize=(12, 6))

# --- 画红线 (Zemp) ---
plt.plot(df_zemp_filt['Year'], df_zemp_filt['SLE_mm_yr'], 
         marker='s', color='red', linewidth=2, label="Zemp et al.'s study")
plt.plot(df_zemp_filt['Year'], line_z, 
         color='red', linestyle='--', linewidth=2, alpha=0.8)

# --- 画蓝线 (Our study) ---
plt.plot(df_blue_plot['year'], df_blue_plot['SLE_mm_yr'], 
         marker='o', color='blue', linewidth=2, label="Our study")
plt.plot(df_blue_plot['year'], line_b, 
         color='blue', linestyle='--', linewidth=2, alpha=0.8)

# --- 添加拟合公式文本 ---
# 注意：这里可能需要根据你的数据范围微调坐标位置，我保留了原始代码的坐标
plt.text(1985, -0.040, eq_z_text, color='red', fontsize=12, fontweight='bold')
plt.text(1985, -0.005, eq_b_text, color='blue', fontsize=12, fontweight='bold')

# 设置标签和标题
plt.ylabel('Annual glacier mass change rates (mm SLE yr$^{-1}$)')
plt.xlabel('Year')
plt.title('Comparison of Glacier Mass Changes with Trend Lines')
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

print(f"Zemp (Red) Equation: {eq_z_text}")
print(f"Our Study (Blue) Equation: {eq_b_text}")