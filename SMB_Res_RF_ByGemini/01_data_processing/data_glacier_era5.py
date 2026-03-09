import pandas as pd
import xarray as xr
import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import MERGE_DIR, ERA5_NC_PATH

#步骤03:
#这个代码文件用于将冰川数据与ERA5-LAND气象数据进行空间和时间上的匹配与合并


# ================= 1. 配置路径与参数 =================
# 路径由 config.py 统一管理
glacier_csv_path = os.path.join(MERGE_DIR, 'data_glacier_cleaned.csv')
era5_nc_path = ERA5_NC_PATH

# 输出文件
output_csv_path = os.path.join(MERGE_DIR, 'data_glacier_era5.csv')

# 夏季定义 (北半球一般取5-9月)
SUMMER_MONTHS = [5, 6, 7, 8, 9]

print(">>> 1. 正在读取数据...")
# 读取冰川数据
df_glacier = pd.read_csv(glacier_csv_path)
print(f"   冰川样本数: {len(df_glacier)}")

# 读取 ERA5 NetCDF 数据
ds = xr.open_dataset(era5_nc_path)

# 🔧【核心修复】：将 'valid_time' 重命名为 'time'
print(f"   原始维度列表: {list(ds.dims)}")
if 'valid_time' in ds.dims or 'valid_time' in ds.coords:
    print("   ✅ 检测到时间维度名为 'valid_time'，正在强制重命名为 'time'...")
    ds = ds.rename({'valid_time': 'time'})

# 再次确认
if 'time' not in ds.dims:
    raise ValueError(f"依然无法找到 time 维度，当前维度: {list(ds.dims)}")

print("   ERA5 变量列表:", list(ds.data_vars))

#  2. 定义变量映射

var_mapping = {
    # 状态量 (均值) 
    't2m':  ('temperature_2m', 'mean'),           
    'd2m':  ('dewpoint_temperature_2m', 'mean'),  
    'skt':  ('skin_temperature', 'mean'),         
    'sp':   ('surface_pressure', 'mean'),         
    'fal':  ('forecast_albedo', 'mean'),          
    'asn':  ('snow_albedo', 'mean'),              
    'rsn':  ('snow_density', 'mean'),             
    'sd':   ('snow_depth', 'mean'),               
    'lai_hv': ('leaf_area_index_high_vegetation', 'mean'),
    'lai_lv': ('leaf_area_index_low_vegetation', 'mean'),
    
    #  通量 (累加) 
    'tp':   ('total_precipitation', 'sum'),       
    'sf':   ('snowfall', 'sum'),                  
    'smlt': ('snowmelt', 'sum'),                  
    'ssrd': ('surface_solar_radiation_downwards', 'sum'), 
    'strd': ('surface_thermal_radiation_downwards', 'sum'), 
    'ssr':  ('surface_net_solar_radiation', 'sum'),       
    'str':  ('surface_net_thermal_radiation', 'sum'),     
    'slhf': ('surface_latent_heat_flux', 'sum'),          
    'sshf': ('surface_sensible_heat_flux', 'sum'),        
    'e':    ('total_evaporation', 'sum'),         
    'pev':  ('potential_evaporation', 'sum'),     
    'ro':   ('runoff', 'sum'),                    
    'sro':  ('surface_runoff', 'sum'),            
    'ssro': ('sub_surface_runoff', 'sum'),        
    'es':   ('snow_evaporation', 'sum'),          
}

# 筛选存在的变量
available_vars = {}
for nc_var, (target_base, method) in var_mapping.items():
    if nc_var in ds.data_vars:
        available_vars[nc_var] = (target_base, method)

# ================= 3. 空间提取 (Nearest Neighbor) =================
print("\n>>> 2. 正在提取空间点数据 (这可能需要一些时间)...")

unique_coords = df_glacier[['LATITUDE', 'LONGITUDE']].drop_duplicates()
print(f"   需要提取的唯一坐标点数: {len(unique_coords)}")

def extract_point_timeseries(row, dataset):
    try:
        # 尝试标准名 latitude/longitude
        point_data = dataset.sel(
            latitude=row['LATITUDE'], 
            longitude=row['LONGITUDE'], 
            method='nearest'
        )
    except KeyError:
        # 尝试简称 lat/lon
        point_data = dataset.sel(
            lat=row['LATITUDE'], 
            lon=row['LONGITUDE'], 
            method='nearest'
        )
    
    # 转 DataFrame
    df_point = point_data.to_dataframe().reset_index()
    
    # 附加坐标信息
    df_point['LATITUDE'] = row['LATITUDE']
    df_point['LONGITUDE'] = row['LONGITUDE']
    return df_point

# 批量提取
extracted_dfs = []
total = len(unique_coords)
# 只加载需要的变量子集，节省内存
ds_subset = ds[list(available_vars.keys())]

for i, (_, row) in enumerate(unique_coords.iterrows()):
    if i % 100 == 0: print(f"   已处理 {i}/{total} 个点...")
    try:
        df_p = extract_point_timeseries(row, ds_subset)
        extracted_dfs.append(df_p)
    except Exception as e:
        print(f"   ❌ 提取点 ({row['LATITUDE']}, {row['LONGITUDE']}) 失败: {e}")

if not extracted_dfs:
    raise ValueError("没有提取到任何数据！")

df_era5_raw = pd.concat(extracted_dfs, ignore_index=True)
print("   空间提取完成，开始时间聚合...")

# ================= 4. 时间聚合 (Year & Summer) =================
print("\n>>> 3. 计算年均值与夏季均值...")

# 确保 time 列是 datetime 类型
df_era5_raw['time'] = pd.to_datetime(df_era5_raw['time'])
df_era5_raw['YEAR'] = df_era5_raw['time'].dt.year
df_era5_raw['MONTH'] = df_era5_raw['time'].dt.month

grouped = df_era5_raw.groupby(['LATITUDE', 'LONGITUDE', 'YEAR'])

# 1. 全年聚合
agg_rules = {v: method for v, (_, method) in available_vars.items()}
df_yearly = grouped.agg(agg_rules).reset_index()

# 重命名全年列
rename_dict_year = {}
for nc_var, (target_base, method) in available_vars.items():
    suffix = "_sum_year" if method == 'sum' else "_year"
    rename_dict_year[nc_var] = f"{target_base}{suffix}"
df_yearly = df_yearly.rename(columns=rename_dict_year)

# 2. 夏季聚合
df_summer_raw = df_era5_raw[df_era5_raw['MONTH'].isin(SUMMER_MONTHS)]
df_summer = df_summer_raw.groupby(['LATITUDE', 'LONGITUDE', 'YEAR']).agg(agg_rules).reset_index()

# 重命名夏季列
rename_dict_summer = {}
for nc_var, (target_base, method) in available_vars.items():
    suffix = "_sum_summer" if method == 'sum' else "_summer"
    rename_dict_summer[nc_var] = f"{target_base}{suffix}"
df_summer = df_summer.rename(columns=rename_dict_summer)

# 3. 合并年数据和夏数据
df_era5_final = pd.merge(df_yearly, df_summer, on=['LATITUDE', 'LONGITUDE', 'YEAR'], how='outer')

# ================= 5. 合并到冰川数据 =================
print("\n>>> 4. 合并到主数据集...")

df_final = pd.merge(
    df_glacier, 
    df_era5_final, 
    on=['LATITUDE', 'LONGITUDE', 'YEAR'], 
    how='left'
)

missing_era5 = df_final[f"{list(rename_dict_year.values())[0]}"].isnull().sum()
print(f"   合并后总行数: {len(df_final)}")
if missing_era5 > 0:
    print(f"⚠️ 警告: 有 {missing_era5} 行数据未能匹配到气象数据 (可能是这些年份ERA5数据缺失)")

# ================= 6. 保存 =================
print(f"\n>>> 5. 保存结果到: {output_csv_path}")
df_final.to_csv(output_csv_path, index=False)

print("-" * 30)
print("✅ 处理完成！")
print(f"   - 状态量示例: {list(rename_dict_year.values())[:2]}")
print(f"   - 通量示例: {list(rename_dict_summer.values())[:2]}")
print("-" * 30)