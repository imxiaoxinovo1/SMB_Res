import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import WGMS_DATA_DIR, MERGE_DIR

#步骤01：这个代码文件用于将WGMS的全冰川质量平衡数据和分带数据进行合并，并关联冰川的地理信息（经纬度、名称等）

# ================= 1. 设置文件路径 =================
# 路径由 config.py 统一管理
mb_file = os.path.join(WGMS_DATA_DIR, 'mass_balance.csv')
band_file = os.path.join(WGMS_DATA_DIR, 'mass_balance_band.csv')
glacier_file = os.path.join(WGMS_DATA_DIR, 'glacier.csv')

print(">>> 1. 读取源文件...")
# 注意：glacier.csv 经常包含特殊字符，建议使用 latin1 编码
df_mb = pd.read_csv(mb_file)
df_band = pd.read_csv(band_file)
df_glacier = pd.read_csv(glacier_file, encoding='latin1')

# ================= 2. 标准化列名 =================
print(">>> 2. 标准化列名...")

# 2.1 处理 mass_balance.csv (全冰川数据)
mb_map = {
    'glacier_id': 'WGMS_ID',
    'year': 'YEAR',
    'annual_balance': 'ANNUAL_BALANCE',
    'area': 'AREA'
}
df_mb = df_mb.rename(columns=mb_map)

# 2.2 处理 mass_balance_band.csv (分带数据)
band_map = {
    'glacier_id': 'WGMS_ID',
    'year': 'YEAR',
    'annual_balance': 'ANNUAL_BALANCE',
    'area': 'AREA',
    'lower_elevation': 'LOWER_BOUND',
    'upper_elevation': 'UPPER_BOUND'
}
df_band = df_band.rename(columns=band_map)

# 2.3 处理 glacier.csv (元数据)
glacier_map = {
    'id': 'WGMS_ID',
    'names': 'NAME',
    'latitude': 'LATITUDE',
    'longitude': 'LONGITUDE',
    'country': 'POLITICAL_UNIT'
}
df_glacier = df_glacier.rename(columns=glacier_map)

# ================= 3. 构造关键特征列 =================
print(">>> 3. 构造几何特征列 (TAG, LOWER_BOUND, UPPER_BOUND)...")

# 3.1 对全冰川数据：所有几何边界强制设为 9999
df_mb['TAG'] = 9999
df_mb['LOWER_BOUND'] = 9999
df_mb['UPPER_BOUND'] = 9999

# 3.2 对分带数据：TAG 等于其下界 (LOWER_BOUND)
# 注意：如果重命名后列不存在，这里做个检查
if 'LOWER_BOUND' in df_band.columns:
    df_band['TAG'] = df_band['LOWER_BOUND']
else:
    raise ValueError("错误：mass_balance_band.csv 中找不到 lower_elevation 列")

# ================= 4. 纵向合并数据 =================
print(">>> 4. 合并全冰川数据与分带数据...")

# 选取需要的公共列
target_cols = [
    'WGMS_ID', 'YEAR', 'ANNUAL_BALANCE', 'AREA', 
    'LOWER_BOUND', 'UPPER_BOUND', 'TAG'
]

# pd.concat 自动对齐列名
df_combined = pd.concat([df_mb[target_cols], df_band[target_cols]], ignore_index=True)

# ================= 5. 横向关联元数据 =================
print(">>> 5. 关联冰川地理信息 (经纬度、名称)...")

# 只选取需要的元数据列
meta_cols = ['WGMS_ID', 'NAME', 'LATITUDE', 'LONGITUDE', 'POLITICAL_UNIT']

# 确保 ID 类型一致 (防止一个是int一个是str导致合并失败)
df_combined['WGMS_ID'] = pd.to_numeric(df_combined['WGMS_ID'], errors='coerce')
df_glacier['WGMS_ID'] = pd.to_numeric(df_glacier['WGMS_ID'], errors='coerce')

# Left Join: 保证所有观测数据都在，如果名录里找不到该冰川则元数据为空
df_final = pd.merge(df_combined, df_glacier[meta_cols], on='WGMS_ID', how='left')

# ================= 6. 排序与保存 =================
print(">>> 6. 排序并保存...")

# 按照 ID -> 年份 -> TAG (9999通常排在最后或最前，看具体数值) 排序
df_final = df_final.sort_values(by=['WGMS_ID', 'YEAR', 'TAG'])

# 保存
output_file = os.path.join(MERGE_DIR, 'data_glacier.csv')
df_final.to_csv(output_file, index=False)

print("-" * 30)
print(f"✅ 数据合并完成！")
print(f"文件已保存为: {output_file}")
print(f"总行数: {len(df_final)}")
print(f"包含列: {df_final.columns.tolist()}")
print("-" * 30)

# 打印前几行预览
print(df_final.head())