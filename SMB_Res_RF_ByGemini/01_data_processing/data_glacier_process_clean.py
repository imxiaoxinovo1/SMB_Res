import pandas as pd
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import MERGE_DIR

#步骤02：这个代码文件用于清理WGMS冰川数据，筛选出北美地区（CA和US）的冰川，并进行单位换算



# ================= 1. 读取刚刚合并好的数据 =================
# 路径由 config.py 统一管理
input_file = os.path.join(MERGE_DIR, 'data_glacier.csv')
output_file = os.path.join(MERGE_DIR, 'data_glacier_cleaned.csv')

print(">>> 1. 读取数据...")
df = pd.read_csv(input_file)

# ================= 2. 区域筛选 (CA & US) =================
print(">>> 2. 筛选区域 (CA, US)...")
# 检查一下 POLITICAL_UNIT 列里有哪些国家
print("筛选前国家分布:", df['POLITICAL_UNIT'].unique())

# 只保留 US 和 CA
target_countries = ['US', 'CA']
df_filtered = df[df['POLITICAL_UNIT'].isin(target_countries)].copy()

print(f"筛选后样本数: {len(df_filtered)} (原样本数: {len(df)})")

# ================= 3. 单位换算 =================
print(">>> 3. 执行单位换算...")

# 3.1 ANNUAL_BALANCE: m w.e. -> mm w.e. (乘以 1000)
# 注意：有些可能是空值，乘法会自动处理 NaN
df_filtered['ANNUAL_BALANCE'] = df_filtered['ANNUAL_BALANCE'] * 1000

# 3.2 AREA: m^2 -> km^2 (除以 1,000,000)
df_filtered['AREA'] = df_filtered['AREA'] / 1000000

# 打印换算后的前几行预览
print("\n换算后数据预览:")
print(df_filtered[['ANNUAL_BALANCE', 'AREA']].head())

# ================= 4. 保存最终清洗版数据 =================
print(f"\n>>> 4. 保存结果到: {output_file}")
df_filtered.to_csv(output_file, index=False)

print("-" * 30)
print("✅ 数据清洗与筛选完成！")
print("现在这份数据已经准备好，可以去和 ERA5 气象数据进行合并了。")
print("-" * 30)