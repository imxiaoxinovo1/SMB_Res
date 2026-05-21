# 04_reconstruction/step01_prepare_rgi02.py
"""
从 RGI v7.0 提取全部 area_km2 >= 0.5 的 RGI02 冰川，计算地形特征。
输出: data/rgi02_target_glaciers.csv (~4,999 行)
列: rgi_id, cenlon, cenlat, area_km2, slope_deg, aspect_sin, aspect_cos,
    zmin_m, zmax_m, zmean_m, zmed_m, lmax_m
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import pandas as pd
import geopandas as gpd
from config import RGI_SHP, RGI02_TARGET_CSV, MIN_AREA_KM2

print("=== Reconstruction Step 01: Prepare RGI02 Target Glaciers ===")

gdf = gpd.read_file(RGI_SHP)[[
    'rgi_id', 'cenlon', 'cenlat', 'area_km2',
    'slope_deg', 'aspect_deg', 'zmin_m', 'zmax_m',
    'zmean_m', 'zmed_m', 'lmax_m',
]].copy()

gdf['aspect_sin'] = np.sin(np.deg2rad(gdf['aspect_deg']))
gdf['aspect_cos'] = np.cos(np.deg2rad(gdf['aspect_deg']))
gdf = gdf.drop(columns=['aspect_deg'])

df_target = gdf[gdf['area_km2'] >= MIN_AREA_KM2].reset_index(drop=True)
print(f"Total RGI02 glaciers: {len(gdf):,}")
print(f"area >= {MIN_AREA_KM2} km²: {len(df_target):,}")
print(f"Total area: {df_target['area_km2'].sum():.1f} km²")

os.makedirs(os.path.dirname(RGI02_TARGET_CSV), exist_ok=True)
df_target.to_csv(RGI02_TARGET_CSV, index=False)
print(f"Saved → {RGI02_TARGET_CSV}")
