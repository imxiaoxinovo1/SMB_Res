# 01_preprocessing/step02_match_rgi.py
"""
将 WGMS RGI02 训练冰川质心与 RGI v7.0 shapefile 做 KD-tree 最近邻匹配。
提取 10 个静态地形特征（aspect_deg 转换为 sin/cos 编码）。
输出: training_glaciers_terrain.csv
"""
import sys, os, io
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PYTHONIOENCODING'] = 'utf-8'
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
import numpy as np
import geopandas as gpd
from scipy.spatial import cKDTree
from config import (TRAINING_GLACIERS_CSV, RGI_SHP, TERRAIN_CSV, STATIC_FEATURES)

print("=== Step 02: RGI Terrain Matching ===")

# 1. 读取训练冰川
df_wgms = pd.read_csv(TRAINING_GLACIERS_CSV)
wgms_coords = np.column_stack([df_wgms['latitude'].values,
                                df_wgms['longitude'].values])
print(f"训练冰川数量: {len(df_wgms)}")

# 2. 读取 RGI shapefile（只取属性列，不需要几何体）
print(f"读取 RGI shapefile...")
gdf_rgi = gpd.read_file(RGI_SHP)[[
    'rgi_id', 'cenlon', 'cenlat', 'area_km2',
    'slope_deg', 'aspect_deg',
    'zmin_m', 'zmax_m', 'zmean_m', 'zmed_m', 'lmax_m'
]].copy()
print(f"RGI v7 冰川总数: {len(gdf_rgi):,}")

# 3. 计算 aspect_sin, aspect_cos
gdf_rgi['aspect_sin'] = np.sin(np.deg2rad(gdf_rgi['aspect_deg']))
gdf_rgi['aspect_cos'] = np.cos(np.deg2rad(gdf_rgi['aspect_deg']))

# 4. KD-tree 最近邻匹配
rgi_coords = np.column_stack([gdf_rgi['cenlat'].values,
                               gdf_rgi['cenlon'].values])
tree = cKDTree(rgi_coords)
dist, idx = tree.query(wgms_coords, k=1)

# 5. 距离阈值检查（约 10 km ≈ 0.09°）
DIST_THRESH = 0.09
n_far = (dist > DIST_THRESH).sum()
if n_far > 0:
    print(f"WARNING: {n_far} 个冰川超过距离阈值 {DIST_THRESH}°:")
    for i, d in enumerate(dist):
        if d > DIST_THRESH:
            print(f"  glacier_id={df_wgms.iloc[i]['glacier_id']}  dist={d:.4f}°")
    # 不中止：坐标可能是近似值，人工确认后继续
else:
    print(f"所有冰川匹配距离 <= {DIST_THRESH}  OK")

# 6. 提取匹配到的地形属性
terrain_cols = ['slope_deg', 'aspect_sin', 'aspect_cos',
                'zmin_m', 'zmax_m', 'zmean_m', 'zmed_m',
                'area_km2', 'lmax_m', 'cenlat']
df_terrain = gdf_rgi.iloc[idx][terrain_cols + ['rgi_id']].reset_index(drop=True)
df_terrain['match_dist_deg'] = dist

# 7. 合并并保存
df_out = pd.concat([df_wgms.reset_index(drop=True), df_terrain], axis=1)

# 验证 STATIC_FEATURES 全部存在
missing = [c for c in STATIC_FEATURES if c not in df_out.columns]
assert not missing, f"缺少静态特征列: {missing}"
print(f"10个静态特征全部存在 OK")

# 坐标非空验证
assert df_out[['cenlat']].notna().all().all(), "cenlat 存在 NaN"

df_out.to_csv(TERRAIN_CSV, index=False)
print(f"\n匹配距离统计: 均值={dist.mean():.4f}°  最大={dist.max():.4f}°")
print(f"保存 {len(df_out)} 行 → {TERRAIN_CSV}")
print(f"列: {df_out.columns.tolist()}")
