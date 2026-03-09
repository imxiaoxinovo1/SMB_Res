"""
Step 02: 提取月度 ERA5-Land 气候数据
核心改动：不进行年度聚合，保留完整的 12 个月时间序列。

相比 RF 管道的 step02_extract_era5.py，关键差异：
  RF 管道 : monthly → resample('1YE').sum/mean → 年度特征
  LSTM 管道: monthly → 直接保留月度值 → 时序输入

输出: 01_preprocessing/data/lstm_monthly_climate.csv
字段: WGMS_ID, YEAR, MONTH, <var1>, <var2>, ..., <var15>
      每行 = 一个冰川在某年某月的 15 个气象值
      总行数 ≈ n_glaciers × n_years × 12
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import xarray as xr
from config import (ERA5_NC_PATH, PREPROCESS_DIR, VAR_MAPPING,
                    MONTHLY_CLIMATE_VARS, TRAIN_YEAR_MIN, TRAIN_YEAR_MAX)

glacier_list_path = os.path.join(PREPROCESS_DIR, "lstm_glacier_list.csv")
output_path       = os.path.join(PREPROCESS_DIR, "lstm_monthly_climate.csv")

# ===== 1. 读取冰川列表 =====
print(">>> 1. 读取冰川列表...")
df_glacier = pd.read_csv(glacier_list_path)
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

# ===== 3. 筛选月度气候变量 =====
print(f"\n>>> 3. 确认可用变量（共 {len(MONTHLY_CLIMATE_VARS)} 个）...")
available = [v for v in MONTHLY_CLIMATE_VARS if v in ds.data_vars]
missing   = [v for v in MONTHLY_CLIMATE_VARS if v not in ds.data_vars]

if missing:
    print(f"   WARNING: NetCDF 中缺失以下变量: {missing}")
    print(f"   将仅提取可用变量: {available}")
else:
    print(f"   OK: 全部 {len(available)} 个月度变量均可用")

# 构建变量的输出列名映射（NetCDF key → 可读列名）
col_name_map = {nc_var: VAR_MAPPING[nc_var][0] for nc_var in available}
print(f"   变量映射: {col_name_map}")

# ===== 4. 处理经度格式 =====
lon_max = float(ds.longitude.max())
if lon_max > 180:
    print(f"\n   ERA5 使用 0-360° 经度（最大值 {lon_max:.1f}°），转换 WGS84 负值...")
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
# 预期: {time: N_months, glacier: N_glaciers}

# ===== 6. 筛选年份范围，保留月度分辨率 =====
print(f"\n>>> 5. 筛选年份 {TRAIN_YEAR_MIN}-{TRAIN_YEAR_MAX}（保留月度分辨率）...")

# 筛选目标年份的月份
time_years = ds_points.time.dt.year
time_mask  = (time_years >= TRAIN_YEAR_MIN) & (time_years <= TRAIN_YEAR_MAX)
ds_monthly = ds_points.sel(time=time_mask)

n_months = len(ds_monthly.time)
n_years  = TRAIN_YEAR_MAX - TRAIN_YEAR_MIN + 1
print(f"   保留月度时步数: {n_months}（预期 {n_years * 12} = {n_years}年×12月）")

# ===== 7. 计算到内存 =====
print("\n>>> 6. 计算到内存（此步可能需要数分钟）...")
ds_computed = ds_monthly.compute()
print(f"   计算完成，数据形状: time={len(ds_computed.time)}, glacier={len(df_glacier)}")

# ===== 8. 转换为 DataFrame =====
print("\n>>> 7. 转为 DataFrame（保留年份、月份信息）...")

times      = pd.to_datetime(ds_computed.time.values)
time_years = times.year
time_months= times.month

records = []
for g_idx, row in df_glacier.iterrows():
    wgms_id = row['WGMS_ID']
    for t_idx, (yr, mo) in enumerate(zip(time_years, time_months)):
        rec = {'WGMS_ID': wgms_id, 'YEAR': int(yr), 'MONTH': int(mo)}
        for nc_var in available:
            col = col_name_map[nc_var]
            val = float(ds_computed[nc_var].values[t_idx, g_idx])
            rec[col] = val
        records.append(rec)

df_monthly = pd.DataFrame(records)
df_monthly = df_monthly.sort_values(['WGMS_ID', 'YEAR', 'MONTH']).reset_index(drop=True)

# ===== 9. 验证 =====
print(f"\n>>> 验证:")
print(f"   总行数: {len(df_monthly)}")
print(f"   冰川数: {df_monthly['WGMS_ID'].nunique()}")
print(f"   年份范围: {df_monthly['YEAR'].min()} - {df_monthly['YEAR'].max()}")
print(f"   月份分布:\n{df_monthly['MONTH'].value_counts().sort_index().to_string()}")

nan_total = df_monthly[list(col_name_map.values())].isna().sum().sum()
print(f"   NaN 总数: {nan_total}")

# ===== 10. 保存 =====
df_monthly.to_csv(output_path, index=False)
print(f"\n✅ 月度气候数据已保存: {output_path}")
print(f"   列: WGMS_ID, YEAR, MONTH, {', '.join(list(col_name_map.values()))}")
print(f"   每行 = 1 冰川 × 1年份 × 1月份 × {len(available)} 气象变量")
