"""
Step 01: 准备 RGI02 区域冰川列表
- 从 WGMS FoG 数据库提取西加拿大/美国冰川
- 双重过滤: 国家 + 纬度 (30-60°N)
- 聚合几何参数: AREA (km²), LOWER_BOUND, UPPER_BOUND
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from config import (GLACIER_CSV, STATE_CSV, PREPROCESS_DATA_DIR,
                    TARGET_COUNTRIES, LATITUDE_MIN, LATITUDE_MAX)

os.makedirs(PREPROCESS_DATA_DIR, exist_ok=True)
output_path = os.path.join(PREPROCESS_DATA_DIR, "rgi02_glaciers.csv")

# ================= 1. 读取冰川名录 =================
print(">>> 1. 读取冰川名录...")
df_glacier = pd.read_csv(GLACIER_CSV, encoding='latin1', low_memory=False)

# 重命名列 (WGMS 原始列名为小写)
rename_map = {
    'country': 'POLITICAL_UNIT',
    'id': 'WGMS_ID',
    'names': 'NAME',
    'latitude': 'LATITUDE',
    'longitude': 'LONGITUDE',
}
for old, new in rename_map.items():
    if old in df_glacier.columns:
        df_glacier.rename(columns={old: new}, inplace=True)

print(f"   总冰川数: {df_glacier['WGMS_ID'].nunique()}")

# ================= 2. 双重过滤: 国家 + 纬度 =================
print(">>> 2. 筛选 RGI02 区域冰川...")
mask = (
    df_glacier['POLITICAL_UNIT'].isin(TARGET_COUNTRIES) &
    (df_glacier['LATITUDE'] > LATITUDE_MIN) &
    (df_glacier['LATITUDE'] < LATITUDE_MAX)
)
df_rgi02 = df_glacier[mask].copy()
df_rgi02 = df_rgi02.drop_duplicates(subset=['WGMS_ID'])[
    ['WGMS_ID', 'NAME', 'LATITUDE', 'LONGITUDE', 'POLITICAL_UNIT']
]
print(f"   国家筛选 ({TARGET_COUNTRIES}) + 纬度筛选 ({LATITUDE_MIN}-{LATITUDE_MAX}°N)")
print(f"   筛选后冰川数: {len(df_rgi02)}")

# ================= 3. 匹配几何信息 =================
print(">>> 3. 聚合 state.csv 获取几何参数...")
df_state = pd.read_csv(STATE_CSV, low_memory=False)

if 'glacier_id' in df_state.columns:
    df_state.rename(columns={'glacier_id': 'WGMS_ID'}, inplace=True)

geo_stats = df_state.groupby('WGMS_ID').agg({
    'area': 'mean',
    'lowest_elevation': 'min',
    'highest_elevation': 'max'
}).reset_index()

# 立即转换单位: m² → km²
geo_stats['area'] = geo_stats['area'] / 1_000_000

geo_stats.rename(columns={
    'area': 'AREA',
    'lowest_elevation': 'LOWER_BOUND',
    'highest_elevation': 'UPPER_BOUND'
}, inplace=True)

# ================= 4. 合并并清理 =================
df_target = pd.merge(df_rgi02, geo_stats, on='WGMS_ID', how='left')

n_before = len(df_target)
df_target = df_target.dropna(subset=['LOWER_BOUND', 'UPPER_BOUND', 'AREA'])
n_dropped = n_before - len(df_target)

print(f"   剔除缺失几何信息的冰川: {n_dropped} 个")
print(f"   最终有效冰川数: {len(df_target)}")

# ================= 5. 验证 =================
print("\n>>> 验证检查:")
print(f"   纬度范围: {df_target['LATITUDE'].min():.2f} - {df_target['LATITUDE'].max():.2f} °N")
print(f"   面积范围: {df_target['AREA'].min():.4f} - {df_target['AREA'].max():.2f} km2")
print(f"   面积均值: {df_target['AREA'].mean():.2f} km2")
print(f"   海拔范围: {df_target['LOWER_BOUND'].min():.0f} - {df_target['UPPER_BOUND'].max():.0f} m")

if df_target['LATITUDE'].max() >= 60:
    print("   WARNING: 仍有纬度 >= 60°N 的冰川!")
else:
    print("   OK: 所有冰川在 30-60°N 范围内")

if df_target['AREA'].mean() > 1000:
    print("   WARNING: AREA mean too large, unit may still be m2!")
else:
    print("   OK: AREA unit is km2")

# ================= 6. 保存 =================
df_target.to_csv(output_path, index=False)
print(f"\n>>> 已保存: {output_path}")
print(f"   共 {len(df_target)} 个冰川")
