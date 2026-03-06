import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 读取整合好的文件
csv_path = r"H:\Code\SMB\test\result\integrated_glacier_data_v2.csv"
df = pd.read_csv(csv_path)

# ==========================================
# 2. 修正量级 (Fix Magnitudes)
# ==========================================
# ERA5 Monthly Mean 的累积变量通常是 "m/day" (平均日累积)
# 我们之前简单的 sum() 是把 12 个月的日均值加起来了
# 近似修正方法：乘以 365.25 / 12 (即平均每月天数 ~30.44) 
# 或者更简单：直接乘以 30.44，或者更粗略一点，乘以 30 左右，但为了严谨我们用 365.25/12

correction_factor = 365.25 / 12  # ≈ 30.4375

# 需要修正的列 (累积量)
cols_to_fix = [
    'ERA5_PRECIP_TOTAL', 'ERA5_SNOWFALL', 'ERA5_SNOWMELT', 
    'ERA5_EVAP_TOTAL', 'ERA5_POTENTIAL_EVAP', 'ERA5_SNOW_EVAP',
    'ERA5_RUNOFF', 'ERA5_SURF_RUNOFF', 'ERA5_SUB_RUNOFF'
]

print("正在修正累积变量的量级...")
for col in cols_to_fix:
    if col in df.columns:
        # 乘以前面的 sum 结果，变成真正的年总量
        df[col] = df[col] * correction_factor
        print(f" -> {col} 已修正")

# 保存修正后的文件
fixed_csv_path = r"H:\Code\SMB\test\result\integrated_glacier_data_v2_FIXED.csv"
df.to_csv(fixed_csv_path, index=False)
print(f"✅ 修正版文件已保存: {fixed_csv_path}")

# ==========================================
# 3. 可视化检查 (Sanity Check)
# ==========================================
print("\n正在生成相关性检查图...")

# 设置绘图风格
sns.set(style="whitegrid")
plt.figure(figsize=(14, 6))

# 图1: 气温 vs 物质平衡 (预期: 气温越高，平衡越负，应呈现负相关)
plt.subplot(1, 2, 1)
sns.scatterplot(data=df, x='ERA5_TEMP_2M', y='ANNUAL_BALANCE', alpha=0.6)
# 添加趋势线
sns.regplot(data=df, x='ERA5_TEMP_2M', y='ANNUAL_BALANCE', scatter=False, color='red')
plt.title('Temperature vs. Mass Balance\n(Expect Negative Correlation)')
plt.xlabel('Annual Mean Temp (K)')
plt.ylabel('Annual Mass Balance (m w.e.)')

# 图2: 降水 vs 物质平衡 (预期: 降水越多，平衡越正，应呈现正相关)
plt.subplot(1, 2, 2)
sns.scatterplot(data=df, x='ERA5_PRECIP_TOTAL', y='ANNUAL_BALANCE', alpha=0.6, color='green')
sns.regplot(data=df, x='ERA5_PRECIP_TOTAL', y='ANNUAL_BALANCE', scatter=False, color='blue')
plt.title('Precipitation vs. Mass Balance\n(Expect Positive Correlation)')
plt.xlabel('Annual Total Precip (m)')
plt.ylabel('Annual Mass Balance (m w.e.)')

plt.tight_layout()
plt.show()

# 打印一些统计数据看看是否合理
print("\n数据统计预览 (修正后):")
print(df[['ERA5_PRECIP_TOTAL', 'ERA5_TEMP_2M']].describe())