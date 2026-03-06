import pandas as pd
import xarray as xr
import numpy as np
import os

# ================= 1. 配置区域 =================

# 输入：ERA5-Land NetCDF 文件路径
nc_path = r"E:\ERA5-LAND\test_2\data_stream-moda.nc"

# 输入：WGMS 冰川数据 CSV 路径
wgms_csv_path = r"H:\Code\SMB\test\result\WGMS_Region_02_Data.csv"

# 输出：最终合并后的 CSV 保存路径
output_path = r"H:\Code\SMB\test\result\integrated_glacier_data_v2.csv"

# ================= 2. 变量映射 (完全对应你的 NetCDF) =================

# 字典格式: 'NetCDF变量名': '输出CSV列名'
var_map = {
    # --- 气温与湿度 ---
    't2m':  'ERA5_TEMP_2M',       # 2米气温 (K)
    'd2m':  'ERA5_DEW_2M',        # 2米露点 (K)
    'skt':  'ERA5_SKIN_TEMP',     # 地表温度 (K)
    
    # --- 降水与雪 ---
    'tp':   'ERA5_PRECIP_TOTAL',  # 总降水 (m) -> 需要求和
    'sf':   'ERA5_SNOWFALL',      # 降雪量 (m we) -> 需要求和
    'sd':   'ERA5_SNOW_DEPTH',    # 雪深 (m we) -> 水当量
    'rsn':  'ERA5_SNOW_DENSITY',  # 雪密度 (kg/m3)
    'asn':  'ERA5_SNOW_ALBEDO',   # 雪反照率 (0-1)
    'smlt': 'ERA5_SNOWMELT',      # 融雪量 (m we) -> 需要求和
    'es':   'ERA5_SNOW_EVAP',     # 雪面蒸发 (m we) -> 需要求和
    'tsn':  'ERA5_SNOW_LAYER_TEMP', # 雪层温度 (K)
    'src':  'ERA5_SKIN_RESERVOIR',  # 皮肤库含量 (m we)
    
    # --- 辐射与通量 (J/m2) ---
    'ssrd': 'ERA5_SOLAR_DOWN',    # 向下短波辐射
    'strd': 'ERA5_THERMAL_DOWN',  # 向下长波辐射
    'ssr':  'ERA5_NET_SOLAR',     # 净短波辐射
    'str':  'ERA5_NET_THERMAL',   # 净长波辐射
    'slhf': 'ERA5_LATENT_HEAT',   # 潜热通量
    'sshf': 'ERA5_SENSIBLE_HEAT', # 感热通量
    'fal':  'ERA5_FORECAST_ALBEDO', # 预报反照率
    
    # --- 水文与蒸发 ---
    'e':    'ERA5_EVAP_TOTAL',    # 总蒸发 (m we) -> 需要求和
    'pev':  'ERA5_POTENTIAL_EVAP',# 潜在蒸发 (m) -> 需要求和
    'ro':   'ERA5_RUNOFF',        # 总径流 (m) -> 需要求和
    'sro':  'ERA5_SURF_RUNOFF',   # 地表径流 (m) -> 需要求和
    'ssro': 'ERA5_SUB_RUNOFF',    # 地下径流 (m) -> 需要求和
    
    # --- 气压与风 ---
    'sp':   'ERA5_SURF_PRESS',    # 地表气压 (Pa)
    # u10 和 v10 单独处理用于计算风速，但也保留分量
    'u10':  'ERA5_WIND_U',
    'v10':  'ERA5_WIND_V',
    
    # --- 植被 ---
    'lai_hv': 'ERA5_LAI_HIGH',    # 高植被叶面积指数
    'lai_lv': 'ERA5_LAI_LOW',     # 低植被叶面积指数
    
    # --- 湖泊 (可选，视研究需要) ---
    'lblt': 'ERA5_LAKE_BOT_TEMP',
    'licd': 'ERA5_LAKE_ICE_DEPTH',
    'lict': 'ERA5_LAKE_ICE_TEMP',
    'lmld': 'ERA5_LAKE_MIX_DEPTH',
    'lmlt': 'ERA5_LAKE_MIX_TEMP',
    'lshf': 'ERA5_LAKE_SHAPE',
    'ltlt': 'ERA5_LAKE_TOTAL_TEMP'
}

# 聚合规则: 'sum' (累积量) 或 'mean' (状态量)
agg_rules = {
    # 这些变量本质是通量或累积量，需要乘以当月天数
    'tp': 'sum', 'sf': 'sum', 'smlt': 'sum', 
    'e': 'sum', 'pev': 'sum', 'es': 'sum', 
    'ro': 'sum', 'sro': 'sum', 'ssro': 'sum',
    
    # 辐射变量 (J/m2) 在 ERA5 月度数据中通常也是"日均累积"，所以求年总能量也需要 sum
    # 但如果在模型中你用的是"年均日辐射"，则保持 mean。
    # 这里为了物理意义的完整性(年总降水)，建议降水类用 sum，辐射类看你习惯 (通常保持 mean 即可代表辐射强度)
    
    # 其他默认 mean (气温、雪深、风速等)
}
# 其他未列出的默认走 'mean'


# ================= 3. 核心处理逻辑 =================

def process_data():
    print(">>> 1. 正在读取数据...")
    
    # 读取 CSV
    df_wgms = pd.read_csv(wgms_csv_path)
    print(f"   WGMS 数据加载: {len(df_wgms)} 条记录")

    # 读取 NetCDF
    try:
        ds = xr.open_dataset(nc_path)
        print("   ERA5-Land NetCDF 加载成功。")
    except Exception as e:
        print(f"❌ 读取 NetCDF 失败: {e}")
        return

    # [保险操作] 处理 expver (如果存在混合数据)
    if 'expver' in ds.coords:
        try:
            # 合并不同版本的实验数据 (通常取最新的)
            ds = ds.sel(expver=1).combine_first(ds.sel(expver=5))
            print("   ⚠️ 检测到 expver 维度，已自动合并。")
        except:
            pass # 如果只是坐标不是维度，跳过

    # 检查经度格式
    lon_max = ds.longitude.max().item()
    need_lon_shift = lon_max > 180
    if need_lon_shift:
        print("   提示: 检测到 NetCDF 经度为 0-360 格式，将自动转换查询坐标。")

    # 计算合成风速
    if 'u10' in ds.data_vars and 'v10' in ds.data_vars:
        ds['wind_speed'] = np.sqrt(ds['u10']**2 + ds['v10']**2)
        var_map['wind_speed'] = 'ERA5_WIND_SPEED' # 添加到提取列表

    # 准备提取
    unique_sites = df_wgms[['WGMS_ID', 'LATITUDE', 'LONGITUDE']].drop_duplicates()
    total_sites = len(unique_sites)
    print(f">>> 2. 开始处理 {total_sites} 个冰川点 (聚合方式: sum/mean)...")

    results = []
    
    # 获取时间维度名
    time_dim = 'valid_time' if 'valid_time' in ds.dims else 'time'

    for i, (_, row) in enumerate(unique_sites.iterrows()):
        gid = row['WGMS_ID']
        lat, lon = row['LATITUDE'], row['LONGITUDE']
        
        # 转换查询经度 (-120 -> 240)
        q_lon = lon + 360 if (need_lon_shift and lon < 0) else lon
        
        if i % 5 == 0: print(f"   处理进度: {i+1}/{total_sites} ...", end='\r')

        try:
            # A. 空间提取 (Nearest Neighbor)
            ds_point = ds.sel(latitude=lat, longitude=query_lon, method='nearest')
            
            # 【新增步骤 1】获取该时间序列每个月对应的天数 (自动处理闰年)
            # ds_point[time_dim] 是时间坐标，dt.days_in_month 可以直接拿到天数数组
            days_in_month = ds_point[time_dim].dt.days_in_month

            # C. 构建 DataFrame (存放结果)
            df_temp = pd.DataFrame()
            
            # 预先计算年份索引 (使用 resample mean 拿到年份)
            # 使用 '1YE' (Year End) 或 '1Y' 均可，新版 pandas/xarray 推荐 '1YE'
            resampled_time = ds_point.resample({time_dim: '1YE'}).mean()
            df_temp['YEAR'] = resampled_time[time_dim].dt.year
            
            # D. 提取并聚合变量
            for nc_var, csv_col in var_map.items():
                if nc_var not in ds.data_vars: continue

                # 获取聚合规则
                rule = agg_rules.get(nc_var, 'mean')
                
                # 提取该变量的时间序列
                da_var = ds_point[nc_var]

                if rule == 'sum':
                    # 【核心修改】: 先乘以当月天数，还原为“月总量”，再求和得到“年总量”
                    # 1. 计算月总量 (m/day * days = m/month)
                    monthly_total = da_var * days_in_month
                    
                    # 2. 按年求和 (Sum of monthly totals)
                    vals = monthly_total.resample({time_dim: '1YE'}).sum().values
                else:
                    # 状态量 (温度、雪深等) 直接求年平均
                    vals = da_var.resample({time_dim: '1YE'}).mean().values
                
                df_temp[csv_col] = vals
            
            # E. 标记 ID 并放入列表
            df_temp['WGMS_ID'] = gid
            results_list.append(df_temp)

        except Exception as e:
            print(f"\n 跳过冰川 {gid}: {e}")
            continue

    print("\n>>> 3. 正在合并并保存...")
    
    if results:
        # 合并所有冰川的气象数据
        df_meteo = pd.concat(results, ignore_index=True)
        
        # 与原始 WGMS 数据合并 (Left Join)
        df_final = pd.merge(df_wgms, df_meteo, on=['WGMS_ID', 'YEAR'], how='left')
        
        # 整理列顺序 (把 ERA5 列放到后面，保持整洁)
        base_cols = list(df_wgms.columns)
        era5_cols = [c for c in df_final.columns if c not in base_cols]
        df_final = df_final[base_cols + era5_cols]
        
        # 保存
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_final.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        print(f"✅ 处理完成！")
        print(f"   文件已保存至: {output_path}")
        print(f"   包含冰川数: {df_final['WGMS_ID'].nunique()}")
        print(f"   数据行数: {len(df_final)}")
    else:
        print("❌ 未提取到任何数据，请检查经纬度范围。")

if __name__ == "__main__":
    process_data()