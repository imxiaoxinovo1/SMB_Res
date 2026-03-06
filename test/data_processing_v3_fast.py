import pandas as pd
import xarray as xr
import numpy as np
import os
import dask

# ================= 1. 配置区域 =================
nc_path = r"E:\ERA5-LAND\test_2\data_stream-moda.nc"
wgms_csv_path = r"H:\Code\SMB\test\result\WGMS_Region_02_Data.csv"
output_path = r"H:\Code\SMB\test\result\integrated_glacier_data_v3_fast.csv"

# ================= 2. 变量映射 =================
var_map = {
    't2m': 'ERA5_TEMP_2M', 'd2m': 'ERA5_DEW_2M', 'skt': 'ERA5_SKIN_TEMP',
    'tp': 'ERA5_PRECIP_TOTAL', 'sf': 'ERA5_SNOWFALL', 'sd': 'ERA5_SNOW_DEPTH',
    'rsn': 'ERA5_SNOW_DENSITY', 'asn': 'ERA5_SNOW_ALBEDO', 'smlt': 'ERA5_SNOWMELT',
    'es': 'ERA5_SNOW_EVAP', 'tsn': 'ERA5_SNOW_LAYER_TEMP', 'src': 'ERA5_SKIN_RESERVOIR',
    'ssrd': 'ERA5_SOLAR_DOWN', 'strd': 'ERA5_THERMAL_DOWN', 'ssr': 'ERA5_NET_SOLAR',
    'str': 'ERA5_NET_THERMAL', 'slhf': 'ERA5_LATENT_HEAT', 'sshf': 'ERA5_SENSIBLE_HEAT',
    'fal': 'ERA5_FORECAST_ALBEDO', 'e': 'ERA5_EVAP_TOTAL', 'pev': 'ERA5_POTENTIAL_EVAP',
    'ro': 'ERA5_RUNOFF', 'sro': 'ERA5_SURF_RUNOFF', 'ssro': 'ERA5_SUB_RUNOFF',
    'sp': 'ERA5_SURF_PRESS', 'u10': 'ERA5_WIND_U', 'v10': 'ERA5_WIND_V',
    'lai_hv': 'ERA5_LAI_HIGH', 'lai_lv': 'ERA5_LAI_LOW',
    'lblt': 'ERA5_LAKE_BOT_TEMP', 'licd': 'ERA5_LAKE_ICE_DEPTH', 'lict': 'ERA5_LAKE_ICE_TEMP',
    'lmld': 'ERA5_LAKE_MIX_DEPTH', 'lmlt': 'ERA5_LAKE_MIX_TEMP',
    'lshf': 'ERA5_LAKE_SHAPE', 'ltlt': 'ERA5_LAKE_TOTAL_TEMP'
}

# 累积变量列表 (需要乘以天数)
sum_vars = ['tp', 'sf', 'smlt', 'e', 'pev', 'es', 'ro', 'sro', 'ssro']
mean_vars = [k for k in var_map.keys() if k not in sum_vars]

# ================= 3. 极速处理核心 =================

def process_fast():
    print(">>> 1. 读取 WGMS 数据...")
    df_wgms = pd.read_csv(wgms_csv_path)
    unique_sites = df_wgms[['WGMS_ID', 'LATITUDE', 'LONGITUDE']].drop_duplicates().reset_index(drop=True)
    print(f"   待提取冰川数: {len(unique_sites)}")

    # 优化 Dask 调度器 (可选)
    dask.config.set(scheduler='threads') 

    print(">>> 2. 懒加载 NetCDF (Lazy Loading)...")
    # chunks='auto' 是关键，让 xarray 使用 dask
    ds = xr.open_dataset(nc_path, chunks='auto')

    # 坐标预处理 (0-360 转 -180-180 适配)
    lon_max = ds.longitude.max().item()
    target_lats = unique_sites['LATITUDE'].values
    target_lons = unique_sites['LONGITUDE'].values
    
    if lon_max > 180:
        target_lons = np.where(target_lons < 0, target_lons + 360, target_lons)
        print("   提示: 已修正查询经度格式 (0-360)")

    print(">>> 3. 向量化提取 (Vectorized Selection)...")
    
    # 构建 xarray 索引器
    xr_lats = xr.DataArray(target_lats, dims="glacier")
    xr_lons = xr.DataArray(target_lons, dims="glacier")
    
    # 一次性提取所有点 (Result shape: Time x Glacier)
    # 这一步很快，因为只是构建了任务图，没有真正读取数据
    ds_points = ds.sel(latitude=xr_lats, longitude=xr_lons, method='nearest')

    # 计算风速 (Lazy evaluation)
    if 'u10' in ds_points and 'v10' in ds_points:
        ds_points['wind_speed'] = np.sqrt(ds_points['u10']**2 + ds_points['v10']**2)
        var_map['wind_speed'] = 'ERA5_WIND_SPEED'
        mean_vars.append('wind_speed')

    # 获取时间维度名
    time_dim = 'valid_time' if 'valid_time' in ds.dims else 'time'

    # 【关键修正】计算天数权重
    # days_in_month 的形状是 (Time,)，会自动广播到 (Time, Glacier)
    days_in_month = ds_points[time_dim].dt.days_in_month

    print(">>> 4. 并行计算与重采样 (Resample & Compute)...")
    
    # 筛选实际存在的变量
    valid_sum_vars = [v for v in sum_vars if v in ds_points]
    valid_mean_vars = [v for v in mean_vars if v in ds_points]

    # --- 处理 Sum 变量 (带天数加权) ---
    # 1. 乘以天数: m/day -> m/month
    ds_weighted = ds_points[valid_sum_vars] * days_in_month
    # 2. 按年求和: m/month -> m/year
    # 使用 '1YE' (Year End) 兼容性更好
    ds_yearly_sum = ds_weighted.resample({time_dim: '1YE'}).sum()

    # --- 处理 Mean 变量 (直接平均) ---
    ds_yearly_mean = ds_points[valid_mean_vars].resample({time_dim: '1YE'}).mean()

    # --- 合并并触发计算 ---
    # merge 依然是 lazy 的
    ds_final_lazy = xr.merge([ds_yearly_sum, ds_yearly_mean])
    
    # compute() 会触发所有 Dask 任务，真正读取磁盘并计算
    # 这是最耗时的一步，但会比循环快得多
    print("   正在执行计算图 (这可能需要几十秒)...")
    ds_final = ds_final_lazy.compute()

    print(">>> 5. 格式转换与保存...")
    
    # 转为 DataFrame
    df_era5_raw = ds_final.to_dataframe().reset_index()
    
    # 提取年份
    df_era5_raw['YEAR'] = df_era5_raw[time_dim].dt.year
    
    # 映射回 WGMS_ID
    # df_era5_raw 的 'glacier' 列对应 unique_sites 的 index
    id_map = unique_sites['WGMS_ID'].to_dict()
    df_era5_raw['WGMS_ID'] = df_era5_raw['glacier'].map(id_map)
    
    # 重命名列
    df_era5_raw.rename(columns=var_map, inplace=True)
    
    # 合并
    cols_to_keep = ['WGMS_ID', 'YEAR'] + [c for c in var_map.values() if c in df_era5_raw.columns]
    df_meteo = df_era5_raw[cols_to_keep]
    
    df_final = pd.merge(df_wgms, df_meteo, on=['WGMS_ID', 'YEAR'], how='left')
    
    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ 成功！文件已保存: {output_path}")

if __name__ == "__main__":
    process_fast()