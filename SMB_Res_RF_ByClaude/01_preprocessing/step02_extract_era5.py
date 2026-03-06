"""
Step 02: 提取 ERA5-Land 气候数据
- 对所有 RGI02 冰川进行最近邻空间提取
- 时间聚合: 全年 + 夏季 (5-9月)
- 列名严格匹配训练数据格式
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import xarray as xr
from config import (ERA5_NC_PATH, PREPROCESS_DATA_DIR,
                    VAR_MAPPING, SUMMER_MONTHS)

os.makedirs(PREPROCESS_DATA_DIR, exist_ok=True)

glacier_csv = os.path.join(PREPROCESS_DATA_DIR, "rgi02_glaciers.csv")
output_csv = os.path.join(PREPROCESS_DATA_DIR, "rgi02_glaciers_era5.csv")

# ================= 1. 读取冰川列表 =================
print(">>> 1. 读取冰川列表...")
df_glacier = pd.read_csv(glacier_csv)
print(f"   冰川数量: {len(df_glacier)}")

# ================= 2. 打开 ERA5 NetCDF =================
print(">>> 2. 读取 ERA5-Land NetCDF...")
ds = xr.open_dataset(ERA5_NC_PATH, chunks='auto')

# 处理时间维度名
if 'valid_time' in ds.dims or 'valid_time' in ds.coords:
    ds = ds.rename({'valid_time': 'time'})
    print("   已将 valid_time 重命名为 time")

# 处理 expver 维度 (ERA5 数据版本)
if 'expver' in ds.dims:
    print("   检测到 expver 维度，合并...")
    ds = ds.sel(expver=1).combine_first(ds.sel(expver=5))

print(f"   时间范围: {str(ds.time.values[0])[:10]} ~ {str(ds.time.values[-1])[:10]}")

# 筛选可用变量
available_vars = {}
for nc_var, (base_name, method) in VAR_MAPPING.items():
    if nc_var in ds.data_vars:
        available_vars[nc_var] = (base_name, method)
print(f"   可用变量: {len(available_vars)}/{len(VAR_MAPPING)}")

missing = set(VAR_MAPPING.keys()) - set(available_vars.keys())
if missing:
    print(f"   缺失变量: {missing}")

# ================= 3. 处理经度格式 =================
lats = df_glacier['LATITUDE'].values
lons = df_glacier['LONGITUDE'].values

lon_max = float(ds.longitude.max())
if lon_max > 180:
    print("   ERA5 使用 0-360° 经度，正在转换...")
    lons = np.where(lons < 0, lons + 360, lons)

# ================= 4. 向量化空间提取 =================
print("\n>>> 3. 向量化最近邻提取...")
xr_lats = xr.DataArray(lats, dims="glacier")
xr_lons = xr.DataArray(lons, dims="glacier")

var_list = list(available_vars.keys())
ds_points = ds[var_list].sel(
    latitude=xr_lats,
    longitude=xr_lons,
    method='nearest'
)
print(f"   提取维度: {dict(ds_points.dims)}")

# ================= 5. 时间聚合 =================
print("\n>>> 4. 计算年度和夏季聚合...")

sum_vars = [v for v, (_, m) in available_vars.items() if m == 'sum']
mean_vars = [v for v, (_, m) in available_vars.items() if m == 'mean']

# 5.1 全年聚合
print("   全年聚合...")
yearly_parts = []
if sum_vars:
    yearly_sum = ds_points[sum_vars].resample(time='1YE').sum()
    yearly_parts.append(yearly_sum)
if mean_vars:
    yearly_mean = ds_points[mean_vars].resample(time='1YE').mean()
    yearly_parts.append(yearly_mean)
ds_yearly = xr.merge(yearly_parts)

# 5.2 夏季聚合
print("   夏季聚合 (5-9月)...")
summer_mask = ds_points.time.dt.month.isin(SUMMER_MONTHS)
ds_summer_raw = ds_points.sel(time=summer_mask)

summer_parts = []
if sum_vars:
    summer_sum = ds_summer_raw[sum_vars].resample(time='1YE').sum()
    summer_parts.append(summer_sum)
if mean_vars:
    summer_mean = ds_summer_raw[mean_vars].resample(time='1YE').mean()
    summer_parts.append(summer_mean)
ds_summer = xr.merge(summer_parts)

# ================= 6. 计算并转为 DataFrame =================
print("\n>>> 5. 计算并转换...")
ds_yearly_computed = ds_yearly.compute()
ds_summer_computed = ds_summer.compute()

# 年度数据
records_yearly = []
years = pd.to_datetime(ds_yearly_computed.time.values).year

for g_idx in range(len(df_glacier)):
    for t_idx, year in enumerate(years):
        row = {'glacier_idx': g_idx, 'YEAR': int(year)}
        for nc_var, (base_name, method) in available_vars.items():
            suffix = "_sum_year" if method == 'sum' else "_year"
            col_name = f"{base_name}{suffix}"
            val = float(ds_yearly_computed[nc_var].values[t_idx, g_idx])
            row[col_name] = val
        records_yearly.append(row)

df_yearly = pd.DataFrame(records_yearly)

# 夏季数据
records_summer = []
years_s = pd.to_datetime(ds_summer_computed.time.values).year

for g_idx in range(len(df_glacier)):
    for t_idx, year in enumerate(years_s):
        row = {'glacier_idx': g_idx, 'YEAR': int(year)}
        for nc_var, (base_name, method) in available_vars.items():
            suffix = "_sum_summer" if method == 'sum' else "_summer"
            col_name = f"{base_name}{suffix}"
            val = float(ds_summer_computed[nc_var].values[t_idx, g_idx])
            row[col_name] = val
        records_summer.append(row)

df_summer = pd.DataFrame(records_summer)

# ================= 7. 合并气象数据 =================
print(">>> 6. 合并年度与夏季数据...")
df_era5 = pd.merge(df_yearly, df_summer, on=['glacier_idx', 'YEAR'], how='outer')

# ================= 8. 匹配回冰川静态属性 =================
print(">>> 7. 合并冰川属性...")
df_glacier_indexed = df_glacier.reset_index(drop=True)
df_glacier_indexed['glacier_idx'] = df_glacier_indexed.index

df_final = pd.merge(df_glacier_indexed, df_era5, on='glacier_idx', how='inner')
df_final.drop(columns=['glacier_idx'], inplace=True)
df_final.sort_values(by=['WGMS_ID', 'YEAR'], inplace=True)

# ================= 9. 验证 =================
print(f"\n>>> 验证:")
print(f"   总行数: {len(df_final)}")
print(f"   冰川数: {df_final['WGMS_ID'].nunique()}")
print(f"   年份范围: {df_final['YEAR'].min()} - {df_final['YEAR'].max()}")

# 检查关键特征列是否存在
required_features = [
    "temperature_2m_year", "temperature_2m_summer",
    "snowmelt_sum_year", "snowmelt_sum_summer",
    "snowfall_sum_year",
    "total_precipitation_sum_year", "total_precipitation_sum_summer",
    "surface_net_solar_radiation_sum_summer",
    "surface_net_thermal_radiation_sum_summer",
    "surface_sensible_heat_flux_sum_summer",
    "surface_latent_heat_flux_sum_summer",
]
missing_features = [f for f in required_features if f not in df_final.columns]
if missing_features:
    print(f"   WARNING: 缺失特征列: {missing_features}")
else:
    print(f"   OK: 所有模型所需的特征列均存在")

nan_count = df_final[required_features].isna().sum().sum()
print(f"   气候特征 NaN 总数: {nan_count}")

# ================= 10. 保存 =================
df_final.to_csv(output_csv, index=False)
print(f"\n>>> 已保存: {output_csv}")
