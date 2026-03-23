"""
从 RGI v7.0 提取冰川地形特征（坡度、坡向、高程统计）

输入:
  - SMB_Res_LSTM_Byclaude/01_preprocessing/data/lstm_glacier_list.csv  (31 个 WGMS 冰川)
  - H:/Code/SMB/RGI/RGI2000-v7.0-G-02_western_canada_usa/             (RGI v7.0 shapefiles)

输出:
  - 01_preprocessing/data/glacier_terrain_features.csv  — 31 冰川地形特征
    列: WGMS_ID, rgi_id, slope_deg, aspect_deg, aspect_sin, aspect_cos,
        zmin_m, zmax_m, zmed_m, zmean_m, dist_m (匹配距离)

说明:
  RGI v7.0 shapefile 已内置 slope_deg / aspect_deg / z* 字段（来自 Copernicus DEM 30m），
  无需额外下载 OGGM 或 DEM 数据，直接按最近质心距离匹配。
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from config import (LSTM_PREPROCESS_DIR, BASE_DIR)

# ─── 路径 ────────────────────────────────────────────────────────────────────
RGI_SHP   = r"H:\Code\SMB\RGI\RGI2000-v7.0-G-02_western_canada_usa\RGI2000-v7.0-G-02_western_canada_usa.shp"
WGMS_LIST = os.path.join(LSTM_PREPROCESS_DIR, "lstm_glacier_list.csv")
OUT_DIR   = os.path.join(BASE_DIR, "01_preprocessing", "data")
OUT_CSV   = os.path.join(OUT_DIR, "glacier_terrain_features.csv")

os.makedirs(OUT_DIR, exist_ok=True)

# ─── 1. 读取 WGMS 冰川列表 ───────────────────────────────────────────────────
print(">>> 1. 读取 WGMS 冰川列表...")
df_wgms = pd.read_csv(WGMS_LIST)
print(f"   WGMS 冰川数: {len(df_wgms)}")
print(f"   列: {df_wgms.columns.tolist()}")

# 构建 GeoDataFrame（WGMS 点，WGS84）
gdf_wgms = gpd.GeoDataFrame(
    df_wgms,
    geometry=[Point(lon, lat) for lon, lat in
              zip(df_wgms['LONGITUDE'], df_wgms['LATITUDE'])],
    crs="EPSG:4326"
)

# ─── 2. 读取 RGI v7.0 shapefile ──────────────────────────────────────────────
print("\n>>> 2. 读取 RGI v7.0 shapefile...")
rgi_cols = ['rgi_id', 'glac_name', 'cenlon', 'cenlat',
            'slope_deg', 'aspect_deg',
            'zmin_m', 'zmax_m', 'zmed_m', 'zmean_m',
            'area_km2', 'geometry']
gdf_rgi = gpd.read_file(RGI_SHP, columns=rgi_cols)
gdf_rgi = gdf_rgi.set_crs("EPSG:4326", allow_override=True)
print(f"   RGI v7.0 冰川总数: {len(gdf_rgi)}")

# 用质心点代替多边形（加速最近邻搜索）
# 先投影到等积坐标系再计算质心，避免地理坐标系下的精度警告
gdf_rgi_centroid = gdf_rgi.copy()
gdf_rgi_centroid['geometry'] = (
    gdf_rgi.geometry.to_crs("EPSG:32610").centroid.to_crs("EPSG:4326")
)

# ─── 3. 最近邻空间匹配 ───────────────────────────────────────────────────────
print("\n>>> 3. 空间最近邻匹配（WGMS → RGI v7.0）...")

# 投影到等积坐标系（UTM Zone 10N）以便计算真实距离（米）
crs_metric = "EPSG:32610"
gdf_wgms_proj    = gdf_wgms.to_crs(crs_metric)
gdf_rgi_proj     = gdf_rgi_centroid[['rgi_id', 'glac_name',
                                      'slope_deg', 'aspect_deg',
                                      'zmin_m', 'zmax_m', 'zmed_m', 'zmean_m',
                                      'area_km2', 'geometry']].to_crs(crs_metric)

matched = gpd.sjoin_nearest(
    gdf_wgms_proj,
    gdf_rgi_proj,
    how='left',
    distance_col='dist_m'
)

# ─── 4. 坡向分解 ─────────────────────────────────────────────────────────────
# aspect_deg: 0=North，顺时针。分解为 sin/cos 两个分量避免 0/360° 不连续
matched['aspect_sin'] = np.sin(np.radians(matched['aspect_deg']))  # 东向分量
matched['aspect_cos'] = np.cos(np.radians(matched['aspect_deg']))  # 北向分量

# ─── 5. 输出结果 ─────────────────────────────────────────────────────────────
out_cols = ['WGMS_ID', 'NAME', 'LATITUDE', 'LONGITUDE',
            'rgi_id', 'glac_name',
            'slope_deg', 'aspect_deg', 'aspect_sin', 'aspect_cos',
            'zmin_m', 'zmax_m', 'zmed_m', 'zmean_m',
            'area_km2', 'dist_m']

# 只保留存在的列
out_cols = [c for c in out_cols if c in matched.columns]
result = matched[out_cols].reset_index(drop=True)

print("\n>>> 匹配结果预览:")
print(result[['WGMS_ID', 'NAME', 'rgi_id', 'slope_deg', 'aspect_deg',
              'zmin_m', 'zmax_m', 'dist_m']].to_string(index=False))

# 检查匹配距离
print(f"\n   匹配距离统计 (米):")
print(f"   最小: {result['dist_m'].min():.0f}  最大: {result['dist_m'].max():.0f}  "
      f"均值: {result['dist_m'].mean():.0f}")

# 距离超过 5km 的可能匹配错误，提示检查
far = result[result['dist_m'] > 5000]
if len(far) > 0:
    print(f"\n   ⚠️  以下 {len(far)} 个冰川匹配距离 > 5km，请人工核查:")
    print(far[['WGMS_ID', 'NAME', 'rgi_id', 'glac_name', 'dist_m']].to_string(index=False))

result.to_csv(OUT_CSV, index=False)
print(f"\n>>> 地形特征已保存: {OUT_CSV}")
print(f"    行数: {len(result)}  列数: {len(result.columns)}")
print(f"    特征列: slope_deg, aspect_deg, aspect_sin, aspect_cos, "
      f"zmin_m, zmax_m, zmed_m, zmean_m")
