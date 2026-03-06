import xarray as xr
import pandas as pd

# ================= 配置区域 =================
# 你的 ERA5-Land 数据路径
nc_path = r"E:\ERA5-LAND\de89cd688f4eb5705ac501e3c4a30736.nc"
# ===========================================

try:
    ds = xr.open_dataset(nc_path)
    print(f"\n{'缩写 (Variable)':<10} | {'全名 (Long Name)':<45} | {'单位 (Units)':<15}")
    print("-" * 75)
    
    # 遍历所有变量，提取元数据
    found_vars = []
    long_name_count = 0
    for var_name in ds.data_vars:
        # 获取 long_name，如果没有则显示 None
        long_name = ds[var_name].attrs.get('long_name', 'Unknown')
        if long_name != 'Unknown':
            long_name_count += 1
        units = ds[var_name].attrs.get('units', 'Unknown')
        
        print(f"{var_name:<10} | {long_name:<45} | {units:<15}")
        found_vars.append(var_name)
        
    print("-" * 75)
    print(f"总共找到 {long_name_count} 个含有 'long_name' 的变量。")
    


except Exception as e:
    print(f"读取文件出错: {e}")



        