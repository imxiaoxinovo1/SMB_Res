# 01_preprocessing/step04_build_tabular.py
"""
构建 XGBoost 表格特征数据集。
每行 = 一个冰川 × 一年，双套季节聚合（cal_/hyd_ 前缀）
目标: annual_balance_m (m w.e.)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
import numpy as np
from config import (ERA5_MONTHLY_CSV, TERRAIN_CSV, TABULAR_CSV,
                    MASSBAL_RGI02_CSV, STATIC_FEATURES, MONTHLY_CLIMATE_VARS,
                    CAL_SUMMER_MONTHS, CAL_WINTER_MONTHS,
                    HYD_ACCUM_MONTHS, HYD_ABLAT_MONTHS)

print("=== Step 04: Build Tabular Dataset ===")

TEMP_LIKE = {'t2m', 'skt', 'd2m', 'sd', 'asn'}  # 均值聚合

def seasonal_agg(df_sub, prefix):
    """对给定月份子集做季节聚合，返回 {feature_name: value} 字典"""
    feats = {}
    if len(df_sub) == 0:
        return feats
    for v in MONTHLY_CLIMATE_VARS:
        if v not in df_sub.columns:
            continue
        vals = df_sub[v].values
        if v in TEMP_LIKE:
            feats[f'{prefix}_{v}_mean'] = float(np.nanmean(vals))
        else:
            feats[f'{prefix}_{v}_sum'] = float(np.nansum(vals))
    return feats

# 1. 读取数据
df_era5    = pd.read_csv(ERA5_MONTHLY_CSV)
df_terrain = pd.read_csv(TERRAIN_CSV)
df_mb      = pd.read_csv(MASSBAL_RGI02_CSV)

print(f"ERA5 月度行数: {len(df_era5):,}")
terrain_cols = [c for c in STATIC_FEATURES if c in df_terrain.columns]
df_terrain_idx = df_terrain.set_index('glacier_id')
df_mb_idx = df_mb.set_index(['glacier_id', 'year'])

# 2. 按冰川-年份构建特征
rows = []
for gid, g_era5 in df_era5.groupby('glacier_id'):
    # 静态地形特征
    if gid not in df_terrain_idx.index:
        continue
    sta = df_terrain_idx.loc[gid, terrain_cols].to_dict()

    for year, y_era5 in g_era5.groupby('year'):
        if len(y_era5) < 12:
            continue  # 跳过不完整年份

        row = {'glacier_id': gid, 'year': year}

        # 全年聚合
        row.update(seasonal_agg(y_era5, 'ann'))

        # 日历年季节
        row.update(seasonal_agg(
            y_era5[y_era5['month'].isin(CAL_SUMMER_MONTHS)], 'cal_summer'))
        row.update(seasonal_agg(
            y_era5[y_era5['month'].isin(CAL_WINTER_MONTHS)], 'cal_winter'))

        # 水文年消融期（当年 May-Sep）
        row.update(seasonal_agg(
            y_era5[y_era5['month'].isin(HYD_ABLAT_MONTHS)], 'hyd_ablat'))

        # 水文年积累期（上一年 Oct-Dec + 当年 Jan-Apr）
        prev = g_era5[g_era5['year'] == year - 1]
        oct_dec = prev[prev['month'].isin([10, 11, 12])]
        jan_apr = y_era5[y_era5['month'].isin([1, 2, 3, 4])]
        hyd_accum = pd.concat([oct_dec, jan_apr], ignore_index=True)
        row.update(seasonal_agg(hyd_accum, 'hyd_accum'))

        # 静态特征
        row.update(sta)

        # 目标变量
        key = (gid, year)
        if key in df_mb_idx.index:
            row['annual_balance_m'] = float(df_mb_idx.loc[key, 'annual_balance'])
        else:
            row['annual_balance_m'] = np.nan

        rows.append(row)

df_out = pd.DataFrame(rows)
print(f"特征行数: {len(df_out):,}")

# 标记有观测标签的行
labeled = df_out['annual_balance_m'].notna()
print(f"有观测标签: {labeled.sum()}")

# 3. 验证特征列数
non_meta = [c for c in df_out.columns
            if c not in ['glacier_id', 'year', 'annual_balance_m']]
print(f"特征列数: {len(non_meta)}")

# 抽样验证：标签行无极端值
if labeled.sum() > 0:
    ab_range = df_out.loc[labeled, 'annual_balance_m'].agg(['min', 'max'])
    assert ab_range['min'] > -10 and ab_range['max'] < 10, \
        f"annual_balance_m 超出合理范围: {ab_range}"
    print(f"annual_balance_m 范围: {ab_range['min']:.2f} ~ {ab_range['max']:.2f} m w.e. OK")

# 4. 保存
df_out.to_csv(TABULAR_CSV, index=False)
print(f"保存 → {TABULAR_CSV}")
