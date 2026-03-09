import pandas as pd
import xarray as xr
import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import RECONSTRUCTION_DIR, ERA5_NC_PATH

# ================= 1. 配置路径 =================
# 路径由 config.py 统一管理
glacier_csv_path = os.path.join(RECONSTRUCTION_DIR, 'rgi02_glaciers.csv')

# 输入2: ERA5 气象源文件 (NetCDF)
era5_nc_path = ERA5_NC_PATH

# 输出: 包含气象数据的最终重建输入表
output_csv_path = os.path.join(RECONSTRUCTION_DIR, 'rgi02_glaciers_era5.csv')

# 夏季定义
SUMMER_MONTHS = [5, 6, 7, 8, 9]

print(">>> 1. 正在读取冰川列表...")
df_glacier = pd.read_csv(glacier_csv_path)
print(f"   待重建冰川数量: {len(df_glacier)}")

# 确保有经纬度
if 'LATITUDE' not in df_glacier.columns or 'LONGITUDE' not in df_glacier.columns:
    print("❌ 错误: 冰川列表缺少经纬度列！")
    exit()

print(">>> 2. 正在读取 ERA5 NetCDF 文件...")
ds = xr.open_dataset(era5_nc_path)

# 🔧 自动修复时间维度名 (valid_time -> time)
if 'valid_time' in ds.dims or 'valid_time' in ds.coords:
    print("   检测到 'valid_time'，正在重命名为 'time'...")
    ds = ds.rename({'valid_time': 'time'})

# ================= 2. 定义变量映射 (需与训练时完全一致) =================
# 必须保证这里生成的列名，和你训练 Random Forest 时的特征名一模一样
var_mapping = {
    # --- 状态量 (均值) ---
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
    
    # --- 通量 (累加) ---
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
print("\n>>> 3. 开始提取气象数据 (这将花费较长时间)...")

unique_coords = df_glacier[['LATITUDE', 'LONGITUDE']].drop_duplicates()
print(f"   需要提取的唯一坐标点数: {len(unique_coords)}")

def extract_point_timeseries(row, dataset):
    # 提取该点所有时间步的数据
    try:
        point_data = dataset.sel(
            latitude=row['LATITUDE'], 
            longitude=row['LONGITUDE'], 
            method='nearest'
        )
    except KeyError:
        point_data = dataset.sel(
            lat=row['LATITUDE'], 
            lon=row['LONGITUDE'], 
            method='nearest'
        )
    
    df_point = point_data.to_dataframe().reset_index()
    # 附加坐标信息，以便后续合并
    df_point['LATITUDE'] = row['LATITUDE']
    df_point['LONGITUDE'] = row['LONGITUDE']
    return df_point

extracted_dfs = []
total = len(unique_coords)
ds_subset = ds[list(available_vars.keys())] # 只加载需要的变量

for i, (_, row) in enumerate(unique_coords.iterrows()):
    if i % 50 == 0: print(f"   已处理 {i}/{total} 个坐标点...")
    try:
        df_p = extract_point_timeseries(row, ds_subset)
        extracted_dfs.append(df_p)
    except Exception as e:
        print(f"   ❌ 提取失败 ({row['LATITUDE']}, {row['LONGITUDE']}): {e}")

if not extracted_dfs:
    print("❌ 错误：未提取到任何数据！")
    exit()

# 合并所有提取的时间序列
print("   正在合并时间序列...")
df_era5_raw = pd.concat(extracted_dfs, ignore_index=True)

# ================= 4. 时间聚合 (Year & Summer) =================
print("\n>>> 4. 计算年均值与夏季均值...")

# 确保时间列格式正确
df_era5_raw['time'] = pd.to_datetime(df_era5_raw['time'])
df_era5_raw['YEAR'] = df_era5_raw['time'].dt.year
df_era5_raw['MONTH'] = df_era5_raw['time'].dt.month

grouped = df_era5_raw.groupby(['LATITUDE', 'LONGITUDE', 'YEAR'])

# 4.1 全年聚合
agg_rules = {v: method for v, (_, method) in available_vars.items()}
df_yearly = grouped.agg(agg_rules).reset_index()

rename_dict_year = {}
for nc_var, (target_base, method) in available_vars.items():
    suffix = "_sum_year" if method == 'sum' else "_year"
    rename_dict_year[nc_var] = f"{target_base}{suffix}"
df_yearly = df_yearly.rename(columns=rename_dict_year)

# 4.2 夏季聚合
df_summer_raw = df_era5_raw[df_era5_raw['MONTH'].isin(SUMMER_MONTHS)]
df_summer = df_summer_raw.groupby(['LATITUDE', 'LONGITUDE', 'YEAR']).agg(agg_rules).reset_index()

rename_dict_summer = {}
for nc_var, (target_base, method) in available_vars.items():
    suffix = "_sum_summer" if method == 'sum' else "_summer"
    rename_dict_summer[nc_var] = f"{target_base}{suffix}"
df_summer = df_summer.rename(columns=rename_dict_summer)

# 4.3 合并气象数据
df_era5_final = pd.merge(df_yearly, df_summer, on=['LATITUDE', 'LONGITUDE', 'YEAR'], how='outer')

# ================= 5. 构建最终重建数据集 =================
print("\n>>> 5. 将气象数据匹配回冰川列表...")

# 这里的逻辑与训练时不同！
# 训练时：我们是用冰川表的年份去匹配气象 (Left Join)
# 重建时：我们要把冰川的静态属性 (面积、海拔) 广播到气象数据的每一年 (Right Join 或 Merge)

# df_glacier (静态: ID, Geo)  <-->  df_era5_final (动态: Lat, Lon, Year, Weather)
# 连接键: LATITUDE, LONGITUDE

df_final = pd.merge(
    df_glacier,           # 包含 WGMS_ID, LOWER_BOUND, UPPER_BOUND, AREA
    df_era5_final,        # 包含 1950-2024 所有年份的气象数据
    on=['LATITUDE', 'LONGITUDE'],
    how='inner'           # 只要能匹配上坐标的都保留
)

# 排序整理
df_final = df_final.sort_values(by=['WGMS_ID', 'YEAR'])

print(f"   重建数据集构建完成！")
print(f"   总行数: {len(df_final)} (包含所有冰川的所有年份)")
print(f"   年份范围: {df_final['YEAR'].min()} - {df_final['YEAR'].max()}")

# ================= 6. 保存 =================
print(f"\n>>> 6. 保存结果到: {output_csv_path}")
df_final.to_csv(output_csv_path, index=False)

print("-" * 30)
print("✅ Step 2 完成！")
print(f"   文件已保存: {output_csv_path}")
print("   现在你可以运行 Step 3 代码，用训练好的模型对这个文件进行预测了！")
print("-" * 30)