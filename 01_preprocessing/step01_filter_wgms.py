# 01_preprocessing/step01_filter_wgms.py
"""
从 WGMS FoG 2025-02b 筛选 RGI02 冰川的年度物质平衡记录。
筛选依据: gtng_region == '02_western_canada_usa'（GTN-G官方分区，不依赖国家字段）
注意: annual_balance 在 FoG 2025-02b 中已为 m w.e.，无需除以 1000
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
import numpy as np
from config import (GLACIER_CSV, MASSBAL_CSV, TRAINING_GLACIERS_CSV,
                    MASSBAL_RGI02_CSV, TRAIN_YEAR_MIN, HOLDOUT_YEAR_MAX)

print("=== Step 01: WGMS RGI02 Filter ===")

# 1. 读取 glacier.csv，按 gtng_region 筛选
df_gl = pd.read_csv(GLACIER_CSV, encoding='latin1', low_memory=False)
df_gl_rgi02 = df_gl[df_gl['gtng_region'] == '02_western_canada_usa'].copy()
print(f"glacier.csv RGI02 行数: {len(df_gl_rgi02):,}")

# 2. 读取 mass_balance.csv
df_mb = pd.read_csv(MASSBAL_CSV, encoding='latin1', low_memory=False)
print(f"mass_balance.csv 总行数: {len(df_mb):,}")

# 3. 筛选 RGI02 冰川、非空 annual_balance、时间范围
rgi02_ids = set(df_gl_rgi02['id'].values)
df_mb_rgi02 = df_mb[
    df_mb['glacier_id'].isin(rgi02_ids) &
    df_mb['annual_balance'].notna() &
    (df_mb['year'] >= TRAIN_YEAR_MIN) &
    (df_mb['year'] <= HOLDOUT_YEAR_MAX)
].copy()

print(f"RGI02 annual_balance 记录数: {len(df_mb_rgi02):,}")
print(f"唯一冰川数: {df_mb_rgi02['glacier_id'].nunique()}")

# 4. 单位验证：annual_balance 应为 m w.e.（均值约 -0.6，绝对值远小于 10）
ab_mean = df_mb_rgi02['annual_balance'].mean()
assert abs(ab_mean) < 10, f"annual_balance 单位可能有误: mean={ab_mean:.3f}（期望 m w.e.，绝对值 < 10）"
print(f"annual_balance 均值: {ab_mean:.3f} m w.e.  [单位验证通过 — 已是 m w.e., 无需转换]")

# 5. 合并冰川静态信息
df_static = df_gl_rgi02[['id', 'names', 'latitude', 'longitude', 'gtng_region']].copy()
df_static = df_static.rename(columns={'id': 'glacier_id', 'names': 'name'})

df_stats = df_mb_rgi02.groupby('glacier_id').agg(
    n_years=('year', 'count'),
    year_min=('year', 'min'),
    year_max=('year', 'max'),
    annual_balance_mean_m=('annual_balance', 'mean'),
).reset_index()

df_out = pd.merge(df_stats, df_static, on='glacier_id', how='left')
df_out = df_out.sort_values('n_years', ascending=False)

# 6. 保存
os.makedirs(os.path.dirname(TRAINING_GLACIERS_CSV), exist_ok=True)
df_out.to_csv(TRAINING_GLACIERS_CSV, index=False)
print(f"\n保存 {len(df_out)} 个冰川 → {TRAINING_GLACIERS_CSV}")

# 7. 保存观测记录表
keep_cols = ['glacier_id', 'year', 'annual_balance']
for c in ['winter_balance', 'summer_balance']:
    if c in df_mb_rgi02.columns:
        keep_cols.append(c)
df_mb_rgi02[keep_cols].to_csv(MASSBAL_RGI02_CSV, index=False)
print(f"保存观测记录 {len(df_mb_rgi02)} 行 → {MASSBAL_RGI02_CSV}")

print("\n前10个冰川（按观测年数排序）:")
print(df_out[['glacier_id', 'name', 'latitude', 'longitude', 'n_years', 'annual_balance_mean_m']].head(10).to_string(index=False))
