"""
Step 01: 提取 RGI02 所有冰川的月度 ERA5 气候数据（重建用）

从 rgi02_glaciers_era5.csv 获取冰川坐标，使用与训练流程完全相同的变量
（MONTHLY_CLIMATE_VARS）进行月度空间提取，覆盖重建年份 RECON_YEAR_MIN ~ RECON_YEAR_MAX。

输出: 03_reconstruction/data/rgi02_monthly_climate.csv
字段: WGMS_ID, YEAR, MONTH, <var1>, ..., <var15>
      每行 = 1 冰川 × 1年份 × 1月份 × 15 个气象值
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import xarray as xr
from config import (ERA5_NC_PATH, RECON_DIR, RGI02_ERA5_CSV,
                    VAR_MAPPING, MONTHLY_CLIMATE_VARS,
                    RECON_YEAR_MIN, RECON_YEAR_MAX)

DATA_DIR = os.path.join(RECON_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
output_path = os.path.join(DATA_DIR, "rgi02_monthly_climate.csv")

# ===== 1. 读取冰川列表 =====
print(">>> 1. 读取 RGI02 冰川列表...")
df_rgi02 = pd.read_csv(RGI02_ERA5_CSV)
# 每个冰川取第一行获取唯一坐标
df_glacier = (df_rgi02[['WGMS_ID', 'LATITUDE', 'LONGITUDE']]
              .drop_duplicates(subset='WGMS_ID')
              .reset_index(drop=True))
print(f"   冰川数量: {len(df_glacier)}")

lats = df_glacier['LATITUDE'].values
lons = df_glacier['LONGITUDE'].values

# ===== 2. 打开 ERA5-Land NetCDF =====
print("\n>>> 2. 读取 ERA5-Land NetCDF...")
print(f"   文件: {ERA5_NC_PATH}")
ds = xr.open_dataset(ERA5_NC_PATH, chunks='auto')

# 统一时间维度名
if 'valid_time' in ds.dims or 'valid_time' in ds.coords:
    ds = ds.rename({'valid_time': 'time'})
    print("   已将 valid_time 重命名为 time")

# 处理 expver（ERA5 数据版本维度）
if 'expver' in ds.dims:
    print("   检测到 expver 维度，合并版本...")
    ds = ds.sel(expver=1).combine_first(ds.sel(expver=5))

print(f"   时间范围: {str(ds.time.values[0])[:10]} ~ {str(ds.time.values[-1])[:10]}")
print(f"   数据变量: {list(ds.data_vars)}")

# ===== 3. 确认可用变量 =====
print(f"\n>>> 3. 确认可用变量（共 {len(MONTHLY_CLIMATE_VARS)} 个）...")
available = [v for v in MONTHLY_CLIMATE_VARS if v in ds.data_vars]
missing   = [v for v in MONTHLY_CLIMATE_VARS if v not in ds.data_vars]

if missing:
    print(f"   WARNING: NetCDF 中缺失: {missing}")
    print(f"   将仅提取可用变量: {available}")
else:
    print(f"   OK: 全部 {len(available)} 个变量均可用")

col_name_map = {nc_var: VAR_MAPPING[nc_var][0] for nc_var in available}
print(f"   变量映射（前5）: {dict(list(col_name_map.items())[:5])}")

# ===== 4. 处理经度格式 =====
lon_max = float(ds.longitude.max())
if lon_max > 180:
    print(f"\n   ERA5 使用 0-360° 经度（最大值 {lon_max:.1f}°），转换负值...")
    lons_era5 = np.where(lons < 0, lons + 360, lons)
else:
    lons_era5 = lons.copy()

# ===== 5. 向量化最近邻空间提取 =====
print("\n>>> 4. 向量化空间提取（最近邻插值）...")
xr_lats = xr.DataArray(lats, dims="glacier")
xr_lons = xr.DataArray(lons_era5, dims="glacier")

ds_points = ds[available].sel(
    latitude=xr_lats,
    longitude=xr_lons,
    method='nearest'
)
print(f"   提取后维度: {dict(ds_points.dims)}")

# ===== 6. 筛选重建年份范围 =====
print(f"\n>>> 5. 筛选年份 {RECON_YEAR_MIN}-{RECON_YEAR_MAX}（保留月度分辨率）...")
time_years = ds_points.time.dt.year
time_mask  = (time_years >= RECON_YEAR_MIN) & (time_years <= RECON_YEAR_MAX)
ds_monthly = ds_points.sel(time=time_mask)

n_months = len(ds_monthly.time)
n_years  = RECON_YEAR_MAX - RECON_YEAR_MIN + 1
print(f"   保留月度时步数: {n_months}（预期 {n_years * 12} = {n_years}年×12月）")

# ===== 7. 计算到内存 =====
print("\n>>> 6. 计算到内存（此步可能需要数分钟）...")
ds_computed = ds_monthly.compute()
print(f"   计算完成，time={len(ds_computed.time)}, glacier={len(df_glacier)}")

# ===== 8. 转换为 DataFrame =====
print("\n>>> 7. 转为 DataFrame...")
times       = pd.to_datetime(ds_computed.time.values)
time_years  = times.year
time_months = times.month

records = []
for g_idx, row in df_glacier.iterrows():
    wgms_id = row['WGMS_ID']
    for t_idx, (yr, mo) in enumerate(zip(time_years, time_months)):
        rec = {'WGMS_ID': wgms_id, 'YEAR': int(yr), 'MONTH': int(mo)}
        for nc_var in available:
            col = col_name_map[nc_var]
            rec[col] = float(ds_computed[nc_var].values[t_idx, g_idx])
        records.append(rec)

df_monthly = pd.DataFrame(records)
df_monthly = df_monthly.sort_values(['WGMS_ID', 'YEAR', 'MONTH']).reset_index(drop=True)

# ===== 9. 验证 =====
print(f"\n>>> 验证:")
print(f"   总行数: {len(df_monthly)}  (预期 {len(df_glacier) * n_months})")
print(f"   冰川数: {df_monthly['WGMS_ID'].nunique()}")
print(f"   年份范围: {df_monthly['YEAR'].min()} - {df_monthly['YEAR'].max()}")
print(f"   月份分布: {df_monthly['MONTH'].value_counts().sort_index().to_dict()}")

nan_total = df_monthly[list(col_name_map.values())].isna().sum().sum()
print(f"   NaN 总数: {nan_total}")

# ===== 10. 保存 =====
df_monthly.to_csv(output_path, index=False)
print(f"\nOK  月度气候数据已保存: {output_path}")
print(f"   列: WGMS_ID, YEAR, MONTH, {', '.join(list(col_name_map.values()))}")
print(f"   每行 = 1 冰川 x 1年 x 1月 x {len(available)} 气象变量")
print(f"\n下一步: 运行 step02_train_final_model.py 训练最终模型")
