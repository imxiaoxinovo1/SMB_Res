import pandas as pd
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import WGMS_DATA_DIR, RECONSTRUCTION_DIR

# ================= 1. 配置路径 =================
# 路径由 config.py 统一管理
glacier_csv = os.path.join(WGMS_DATA_DIR, 'glacier.csv')
state_csv = os.path.join(WGMS_DATA_DIR, 'state.csv')
output_rgi02 = os.path.join(RECONSTRUCTION_DIR, 'rgi02_glaciers.csv')

# ================= 2. 提取 RGI02 冰川 =================
print(">>> 1. 读取冰川名录...")

# 加 encoding='latin1' 防止特殊字符报错
df_glacier = pd.read_csv(glacier_csv, encoding='latin1', low_memory=False)

# 🔧【关键修复】重命名原始列名 -> 标准列名
# 原始 glacier.csv 的列名通常是小写的
rename_dict = {
    'country': 'POLITICAL_UNIT',
    'id': 'WGMS_ID',
    'names': 'NAME',
    'latitude': 'LATITUDE',
    'longitude': 'LONGITUDE'
}
# 检查一下列名是否需要重命名
if 'country' in df_glacier.columns:
    df_glacier.rename(columns=rename_dict, inplace=True)
    print("   已将原始列名转换为标准大写格式。")

# 确保只保留 US 和 CA (即 RGI02 区域)
if 'POLITICAL_UNIT' not in df_glacier.columns:
    print(f"❌ 错误：找不到 'POLITICAL_UNIT' 或 'country' 列。现有列名: {df_glacier.columns.tolist()}")
    exit()

df_rgi02 = df_glacier[df_glacier['POLITICAL_UNIT'].isin(['US', 'CA'])].copy()

# 去重：确保每个冰川只保留一行静态信息
df_rgi02 = df_rgi02.drop_duplicates(subset=['WGMS_ID'])[['WGMS_ID', 'NAME', 'LATITUDE', 'LONGITUDE', 'POLITICAL_UNIT']]

print(f"   锁定 RGI02 区域冰川数量: {len(df_rgi02)}")

# ================= 3. 匹配几何信息 (LOWER/UPPER/AREA) =================
print(">>> 2. 匹配 state.csv 获取几何参数...")
df_state = pd.read_csv(state_csv, low_memory=False)

# state.csv 的 ID 列名通常是 'glacier_id'
if 'glacier_id' in df_state.columns:
    df_state.rename(columns={'glacier_id': 'WGMS_ID'}, inplace=True)

# 聚合计算每个冰川的几何均值 (面积、最低海拔、最高海拔)
# 注意：state.csv 里的列名通常也是小写的 (area, lowest_elevation)
geo_stats = df_state.groupby('WGMS_ID').agg({
    'area': 'mean',
    'lowest_elevation': 'mean',
    'highest_elevation': 'mean'
}).reset_index()

# 合并
df_target = pd.merge(df_rgi02, geo_stats, on='WGMS_ID', how='left')

# 重命名为模型特征名
df_target.rename(columns={
    'lowest_elevation': 'LOWER_BOUND',
    'highest_elevation': 'UPPER_BOUND',
    'area': 'AREA'
}, inplace=True)

# 剔除缺乏几何信息的冰川 (无法预测)
# 这一步很重要，因为没有海拔和面积模型没法跑
original_len = len(df_target)
df_target = df_target.dropna(subset=['LOWER_BOUND', 'UPPER_BOUND', 'AREA'])
print(f"   剔除了 {original_len - len(df_target)} 个缺失几何信息的冰川")

print(f"   最终有效目标冰川数: {len(df_target)}")
df_target.to_csv(output_rgi02, index=False)
print(f"✅ 目标冰川列表已保存: {output_rgi02}")