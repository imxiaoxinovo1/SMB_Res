# 01_preprocessing/step03_extract_era5.py
"""
将 ERA5-Land 月度数据双线性插值提取到 63 个训练冰川质心。
ERA5 特征：valid_time 时间维，expver 版本维（需合并），单位需转换
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
import numpy as np
import xarray as xr
from config import (TERRAIN_CSV, ERA5_NC, ERA5_MONTHLY_CSV,
                    MONTHLY_CLIMATE_VARS, TRAIN_YEAR_MIN, RECON_YEAR_MAX)

print("=== Step 03: ERA5 Monthly Extraction ===")

# 1. 读取冰川坐标
df_gl = pd.read_csv(TERRAIN_CSV)
lats = df_gl['latitude'].values
lons = df_gl['longitude'].values
glacier_ids = df_gl['glacier_id'].values
print(f"目标冰川: {len(df_gl)}")

# 2. 打开 ERA5 NetCDF
print(f"打开 ERA5: {ERA5_NC}")
ds = xr.open_dataset(ERA5_NC, chunks={'valid_time': 120})

# 重命名时间维度
if 'valid_time' in ds.dims:
    ds = ds.rename({'valid_time': 'time'})

# 合并 expver 版本
if 'expver' in ds.dims:
    print("  合并 expver 版本...")
    ds = ds.sel(expver=1, drop=True).combine_first(ds.sel(expver=5, drop=True))

# 时间过滤
ds = ds.sel(time=slice(f'{TRAIN_YEAR_MIN}-01', f'{RECON_YEAR_MAX}-12'))
print(f"  时间范围: {str(ds.time.values[0])[:7]} → {str(ds.time.values[-1])[:7]}")
print(f"  可用变量: {list(ds.data_vars)}")

# 3. 单位转换定义
TEMP_VARS  = {'t2m', 'skt', 'd2m'}           # K → °C
ACCUM_VARS = {'tp', 'sf', 'smlt', 'ssrd', 'strd',
              'ssr', 'str', 'slhf', 'sshf', 'ro'}  # m → mm

# 4. 双线性插值到冰川坐标
# 将冰川坐标裁剪到 ERA5 网格范围内，避免边界 NaN（如质心刚好在最北端格点外）
era5_lat_min = float(ds.latitude.min())
era5_lat_max = float(ds.latitude.max())
era5_lon_min = float(ds.longitude.min())
era5_lon_max = float(ds.longitude.max())

lats_clamped = np.clip(lats, era5_lat_min, era5_lat_max)
lons_clamped = np.clip(lons, era5_lon_min, era5_lon_max)

n_clamped = int(np.sum((lats != lats_clamped) | (lons != lons_clamped)))
if n_clamped > 0:
    bad_mask = (lats != lats_clamped) | (lons != lons_clamped)
    print(f"  坐标 clip 修正了 {n_clamped} 个冰川（ERA5 边界之外）:")
    for i in np.where(bad_mask)[0]:
        print(f"    冰川 {glacier_ids[i]}: lat={lats[i]:.3f}→{lats_clamped[i]:.3f}, "
              f"lon={lons[i]:.3f}→{lons_clamped[i]:.3f}")

lats_da = xr.DataArray(lats_clamped, dims='glacier')
lons_da = xr.DataArray(lons_clamped, dims='glacier')

# 初始化记录列表
times_pd = pd.to_datetime(ds.time.values)
n_times = len(times_pd)
n_glaciers = len(glacier_ids)
records = []
for t in times_pd:
    for gi in range(n_glaciers):
        records.append({
            'glacier_id': int(glacier_ids[gi]),
            'year': t.year,
            'month': t.month,
        })

print(f"总记录数: {len(records):,}")

# 逐变量插值并填入 records
for vi, var in enumerate(MONTHLY_CLIMATE_VARS):
    if var not in ds.data_vars:
        print(f"  WARNING: {var} 不在 ERA5 中，跳过")
        continue

    da = ds[var]
    interp = da.interp(latitude=lats_da, longitude=lons_da,
                       method='linear').values  # (n_time, n_glacier)

    # 单位转换
    if var in TEMP_VARS:
        interp = interp - 273.15
    elif var in ACCUM_VARS:
        interp = interp * 1000.0

    # 填入 records
    for ti in range(n_times):
        for gi in range(n_glaciers):
            records[ti * n_glaciers + gi][var] = float(interp[ti, gi])

    if (vi + 1) % 5 == 0 or (vi + 1) == len(MONTHLY_CLIMATE_VARS):
        print(f"  已完成变量: {vi+1}/{len(MONTHLY_CLIMATE_VARS)} ({var})")

# 5. 转换为 DataFrame 并保存
df_out = pd.DataFrame(records)
df_out.to_csv(ERA5_MONTHLY_CSV, index=False)
print(f"\n保存 {len(df_out):,} 行 → {ERA5_MONTHLY_CSV}")
print(f"列: {df_out.columns.tolist()}")

# 快速验证
assert len(df_out) == n_times * n_glaciers, "行数不对"
assert df_out['t2m'].notna().mean() > 0.99, "t2m 有大量 NaN"
# 检查所有变量的 NaN 比例
for var in MONTHLY_CLIMATE_VARS:
    if var in df_out.columns:
        nan_frac = df_out[var].isna().mean()
        if nan_frac > 0.01:
            print(f"  WARNING: {var} NaN 比例 {nan_frac:.1%}")
t2m_range = df_out['t2m'].agg(['min', 'max'])
assert t2m_range['min'] > -80 and t2m_range['max'] < 40, f"t2m 范围异常: {t2m_range}"
print(f"t2m 范围验证: {t2m_range['min']:.1f}C ~ {t2m_range['max']:.1f}C OK")
print("=== Step 03 Done ===")
