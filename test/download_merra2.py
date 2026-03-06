import ee
import os

# -----------------------------------------------------------------
# GEE MERRA-2 下载脚本
# 目标: 导出师姐论文 (Table 2) 中提到的所有 MERRA-2 变量
# 区域: RGI "western_canada_usa" (o1region '02')
# 时间: 1980-2024 (从 1980-2020 更新)
# -----------------------------------------------------------------

def main():
    try:
        # --- 1. 初始化 GEE ---
        # 确保您的项目 ID 正确
        ee.Initialize(project='my-glacier-project-404', opt_url='https://earthengine-highvolume.googleapis.com')
        print("GEE 初始化成功。")
    except Exception as e:
        print(f"GEE 初始化失败: {e}")
        print("请检查您的项目 ID 'my-glacier-project-404' 是否正确。")
        return

    # --- 2. 定义研究区域 (AOI) ---
    # RGI v6.0 - 保持与师姐研究的一致性
    try:
        rgi_aoi = ee.FeatureCollection("GLIMS/RGI60") \
                    .filter(ee.Filter.eq('o1region', '02')) \
                    .geometry() # 将所有冰川融合成一个大的多边形
        print(f"已成功加载 RGI 区域: western_canada_usa (Region 02)")
    except Exception as e:
        print(f"加载 RGI 区域失败: {e}")
        return

    # --- 3. 定义时间和变量 ---
    start_date = '1980-01-01'
    end_date = '2024-12-31' # 更新到 2024 年底
    
    # <--- MODIFIED 1: 更改为稳定的“小时” (T-Hourly) Collection IDs
    # GEE 会在 M2T1... 中找到这些变量
    variable_map = {
        'NASA/GSFC/MERRA/lnd/2': [ # 陆地诊断 (Hourly)
            'PRECSNOLAND', 'TSH', 'TLML', 'QLML', 'ULML', 'VLML'
        ],
        'NASA/GSFC/MERRA/rad/2': [ # 辐射诊断 (Hourly)
            'NIRDF', 'NIRDR'
        ],
        'NASA/GSFC/MERRA/flx/2': [ # 地表通量诊断 (Hourly)
            'BSTAR', 'CDH', 'CDM', 'CDQ', 'CN', 'DISPH', 'EFLUX', 'EVAP', 
            'FRCAN', 'FRCCN', 'FRCLS', 'FRSEAICE', 'GHTSKIN', 'HFLUX', 
            'HLML', 'PBLH', 'PGENTOT', 'PRECANV', 'PRECCON', 'PRECLSC', 
            'PRECTOTCORR', 'PRECTOT', 'PREVTOT', 'QSH', 'QSTAR', 'RHOA', 
            'RISFC','SPEEDMAX' 'SPEED', 'TAUGWX', 'TAUGWY', 'TAUX', 'TAUY', 
            'TCZPBL', 'TSTAR', 'USTAR', 'Z0H', 'Z0M'
        ]
    }
    # <--- END MODIFICATION 1 ---

    # --- 4. 加载和合并数据 ---
    all_bands_collection = None
    
    for collection_id, var_list in variable_map.items():
        collection = ee.ImageCollection(collection_id) \
                       .filterDate(start_date, end_date) \
                       .select(var_list)
        
        if all_bands_collection is None:
            all_bands_collection = collection
        else:
            # <--- MODIFIED 2: 使用 merge() 来合并不同的 Collection
            all_bands_collection = all_bands_collection.merge(collection)
            
    print(f"已加载并合并 {len(variable_map)} 个 MERRA-2 小时数据集。")

    # --- 5. 提交导出任务 ---
    years = range(1980, 2025)
    months = range(1, 13)
    
    print(f"正在准备 {len(years) * len(months)} 个月均值导出任务...")
    
    tasks_submitted = 0
    for year in years:
        for month in months:
            # <--- MODIFIED 3: 计算月均值 ---
            month_start = f'{year}-{month:02d}-01'
            # 计算下个月的开始日期
            if month == 12:
                month_end = f'{year+1}-01-01'
            else:
                month_end = f'{year}-{month+1:02d}-01'
            
            # 关键：过滤该月所有影像，并计算均值 .mean()
            image = all_bands_collection.filterDate(month_start, month_end).mean()
            # <--- END MODIFICATION 3 ---
            
            # 确保影像存在
            try:
                # 检查影像是否真的包含波段
                if image.bandNames().size().getInfo() > 0:
                    file_name = f'MERRA2_MonthlyMean_{year}_{month:02d}'
                    
                    task = ee.batch.Export.image.toDrive(
                        image=image.clip(rgi_aoi), # 裁剪到您的研究区域
                        description=file_name,
                        folder='GEE_MERRA2_Download', # 在您 Google Drive 中的文件夹
                        fileNamePrefix=file_name,
                        scale=10000, # MERRA-2 的原始分辨率 (~50km)。10km 精度足够
                        fileFormat='GeoTIFF',
                        maxPixels=1e10
                    )
                    
                    task.start()
                    tasks_submitted += 1
                else:
                    print(f"警告: {year}-{month:02d} 没有找到波段, 跳过。")
            except Exception as e:
                # 捕获 GEE 计算错误 (例如：数据在末端月份尚不可用)
                print(f"警告: 无法处理 {year}-{month:02d}, 跳过。错误: {e}")
                continue


    print(f"\n--- 成功 ---")
    print(f"已成功提交 {tasks_submitted} 个导出任务。")
    print(f"请立即前往 GEE 代码编辑器的 'Tasks' 选项卡。")
    print(f"您会看到 {tasks_submitted} 个待处理的任务。")
    print(f"请手动点击每个任务旁边的 'Run' 按钮开始下载到您的 Google Drive。")

if __name__ == '__main__':
    main()