import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import STUDY_TEST_DIR

# ================= 配置 =================
input_file = os.path.join(STUDY_TEST_DIR, 'data_glacier_era5_matched.csv')
output_file = os.path.join(STUDY_TEST_DIR, 'data_glacier_era5_fixed.csv')

print(f">>> 读取数据: {input_file}")
df = pd.read_csv(input_file)

# ================= 修复: 修正 TAG=9999 的海拔 =================
print(">>> 正在修复海拔数据 (去除 9999)...")

# 1. 提取所有非 9999 的分带数据 (这些数据有真实海拔)
df_bands = df[df['TAG'] != 9999].copy()

if df_bands.empty:
    print("⚠️ 警告：没有找到分带数据！无法自动推断海拔。")
else:
    # 2. 计算每个冰川的真实海拔范围 (从分带数据中聚合)
    # 逻辑：一个冰川的最低海拔 = 它所有分带的最小下界
    #       一个冰川的最高海拔 = 它所有分带的最大上界
    glacier_stats = df_bands.groupby('WGMS_ID').agg({
        'LOWER_BOUND': 'min',
        'UPPER_BOUND': 'max'
    }).reset_index()
    
    # 重命名，方便合并
    glacier_stats.rename(columns={
        'LOWER_BOUND': 'REAL_LOWER', 
        'UPPER_BOUND': 'REAL_UPPER'
    }, inplace=True)
    
    # 3. 将真实海拔合并回原数据
    # 这样每一行(包括9999那行)都会知道自己这个冰川的真实上下界是多少
    df = pd.merge(df, glacier_stats, on='WGMS_ID', how='left')
    
    # 4. 仅对 TAG=9999 的行进行替换
    mask_9999 = df['TAG'] == 9999
    count_9999 = mask_9999.sum()
    
    # 替换 LOWER_BOUND
    # 优先用找到的 REAL_LOWER，如果没找到(fillna)，就还得保留原来的(虽然是9999，但也没办法)
    df.loc[mask_9999, 'LOWER_BOUND'] = df.loc[mask_9999, 'REAL_LOWER'].fillna(df.loc[mask_9999, 'LOWER_BOUND'])
    
    # 替换 UPPER_BOUND
    df.loc[mask_9999, 'UPPER_BOUND'] = df.loc[mask_9999, 'REAL_UPPER'].fillna(df.loc[mask_9999, 'UPPER_BOUND'])
    
    # 清理临时列
    df.drop(columns=['REAL_LOWER', 'REAL_UPPER'], inplace=True)
    
    print(f"   已修正 {count_9999} 条全冰川记录的海拔值。")
    
    # 验证一下
    sample_9999 = df[df['TAG'] == 9999].iloc[0]
    print(f"   [验证] ID {sample_9999['WGMS_ID']} (TAG=9999) 新海拔范围: {sample_9999['LOWER_BOUND']} - {sample_9999['UPPER_BOUND']}")

# ================= 保存 =================
df.to_csv(output_file, index=False)
print(f"✅ 修复完成！新文件已保存至: {output_file}")
print("   👉 请修改 test_rf.py 读取这个新文件，然后重新运行模型！")