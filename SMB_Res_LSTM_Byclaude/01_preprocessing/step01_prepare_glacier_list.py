"""
Step 01: 准备冰川列表
从 WGMS 观测数据中提取有年度物质平衡记录的冰川（US/CA），
输出含地理信息的冰川元数据 CSV。

输出: 01_preprocessing/data/lstm_glacier_list.csv
字段: WGMS_ID, NAME, LATITUDE, LONGITUDE, LOWER_BOUND, UPPER_BOUND, AREA
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from config import WGMS_DATA_DIR, WGMS_MATCHED_CSV, PREPROCESS_DIR, TRAIN_YEAR_MIN, TRAIN_YEAR_MAX

output_path = os.path.join(PREPROCESS_DIR, "lstm_glacier_list.csv")

# ===== 1. 读取已匹配的训练 CSV（含 ANNUAL_BALANCE）=====
print(">>> 1. 读取 WGMS 匹配数据...")
df = pd.read_csv(WGMS_MATCHED_CSV)
print(f"   原始行数: {len(df)}")

# 只保留全冰川观测记录（TAG=9999），筛选年份范围
if 'TAG' in df.columns:
    df = df[df['TAG'] == 9999].copy()
    print(f"   TAG=9999 筛选后: {len(df)} 行")

df = df[(df['YEAR'] >= TRAIN_YEAR_MIN) & (df['YEAR'] <= TRAIN_YEAR_MAX)].copy()
print(f"   年份范围 {TRAIN_YEAR_MIN}-{TRAIN_YEAR_MAX} 筛选后: {len(df)} 行")

# 移除 ANNUAL_BALANCE 缺失的行（无法作为监督学习标签）
df = df.dropna(subset=['ANNUAL_BALANCE'])
print(f"   移除 ANNUAL_BALANCE 缺失后: {len(df)} 行")

# ===== 2. 提取冰川级别的静态信息 =====
print("\n>>> 2. 提取冰川静态属性...")

# 静态列：取每个冰川的代表值（取中位数，避免异常值影响）
static_cols = ['WGMS_ID', 'LOWER_BOUND', 'UPPER_BOUND', 'AREA', 'LATITUDE', 'LONGITUDE']
missing_cols = [c for c in static_cols if c not in df.columns]
if missing_cols:
    print(f"   WARNING: 缺失列: {missing_cols}")
    print(f"   可用列: {df.columns.tolist()}")
    raise SystemExit(1)

# 按冰川 ID 聚合：取中位数（对面积、海拔更稳健）
df_static = df.groupby('WGMS_ID')[['LOWER_BOUND', 'UPPER_BOUND', 'AREA', 'LATITUDE', 'LONGITUDE']].median().reset_index()

# 尝试合并冰川名称
if 'NAME' in df.columns:
    df_name = df.groupby('WGMS_ID')['NAME'].first().reset_index()
    df_static = pd.merge(df_static, df_name, on='WGMS_ID', how='left')
else:
    # 从 WGMS glacier.csv 补充名称
    glacier_csv = os.path.join(WGMS_DATA_DIR, 'glacier.csv')
    if os.path.exists(glacier_csv):
        df_glacier = pd.read_csv(glacier_csv, encoding='latin1', low_memory=False)
        col_map = {'id': 'WGMS_ID', 'names': 'NAME'}
        df_glacier = df_glacier.rename(columns={k: v for k, v in col_map.items() if k in df_glacier.columns})
        if 'WGMS_ID' in df_glacier.columns and 'NAME' in df_glacier.columns:
            df_static = pd.merge(df_static, df_glacier[['WGMS_ID', 'NAME']], on='WGMS_ID', how='left')

# ===== 3. 统计每个冰川的有效观测年数 =====
print("\n>>> 3. 统计有效观测年数...")
df_obs_count = df.groupby('WGMS_ID').agg(
    n_years=('YEAR', 'count'),
    year_min=('YEAR', 'min'),
    year_max=('YEAR', 'max'),
).reset_index()

df_static = pd.merge(df_static, df_obs_count, on='WGMS_ID', how='left')
df_static = df_static.sort_values('n_years', ascending=False)

# ===== 4. 保存 =====
print(f"\n>>> 4. 保存冰川列表...")
col_order = ['WGMS_ID']
if 'NAME' in df_static.columns:
    col_order.append('NAME')
col_order += ['LATITUDE', 'LONGITUDE', 'LOWER_BOUND', 'UPPER_BOUND', 'AREA',
              'n_years', 'year_min', 'year_max']
col_order = [c for c in col_order if c in df_static.columns]

df_static[col_order].to_csv(output_path, index=False)

print("-" * 40)
print(f"✅ 冰川列表已保存: {output_path}")
print(f"   冰川总数: {len(df_static)}")
print(f"   有效观测年数范围: {df_static['n_years'].min()} ~ {df_static['n_years'].max()}")
print(f"   纬度范围: {df_static['LATITUDE'].min():.1f}° ~ {df_static['LATITUDE'].max():.1f}°N")
print(f"\n冰川列表预览（按观测年数降序）:")
print(df_static[col_order].head(10).to_string(index=False))
print("-" * 40)
