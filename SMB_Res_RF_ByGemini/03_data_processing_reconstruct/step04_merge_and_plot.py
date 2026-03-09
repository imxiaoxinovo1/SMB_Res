"""
混合数据集生成脚本
目的：优先使用真实观测值，缺失部分用模型预测值填补

策略：
1. 对于31个有观测的冰川：
   - 有观测记录的年份 → 使用真实观测值（蓝色）
   - 缺测的年份 → 使用模型预测值（红色）
2. 对于剩下179个从未被观测过的冰川：
   - 全部使用模型预测值（红色）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import STUDY_TEST_DIR, RECONSTRUCTION_DIR

# ================= 1. 读取数据 =================
print(">>> 步骤 1: 读取数据...")

# 观测数据（31个冰川，包含真实ANNUAL_BALANCE）
obs_file = os.path.join(STUDY_TEST_DIR, 'data_glacier_era5_matched.csv')
obs_df = pd.read_csv(obs_file)

# 重建数据（210个冰川，1950-2025年，模型预测值）
recon_file = os.path.join(RECONSTRUCTION_DIR, 'RGI02_reconstruction_1980_2024.csv')
recon_df = pd.read_csv(recon_file)

print(f"  观测数据: {obs_df.shape[0]} 条记录, {obs_df['WGMS_ID'].nunique()} 个冰川")
print(f"  重建数据: {recon_df.shape[0]} 条记录, {recon_df['WGMS_ID'].nunique()} 个冰川")

# ================= 2. 数据预处理 =================
print("\n>>> 步骤 2: 数据预处理...")

# 2.1 提取观测数据中的关键列
# 注意：观测数据中可能有TAG列，我们只保留全冰川数据（TAG=9999）
if 'TAG' in obs_df.columns:
    obs_clean = obs_df[obs_df['TAG'] == 9999].copy()
    print(f"  过滤TAG=9999后，观测数据剩余: {obs_clean.shape[0]} 条")
else:
    obs_clean = obs_df.copy()

# 2.2 选择需要的列
obs_key_cols = ['WGMS_ID', 'YEAR', 'ANNUAL_BALANCE', 'NAME', 'LATITUDE', 'LONGITUDE',
                'AREA', 'LOWER_BOUND', 'UPPER_BOUND', 'POLITICAL_UNIT']
obs_key_cols = [col for col in obs_key_cols if col in obs_clean.columns]
obs_clean = obs_clean[obs_key_cols]

# 2.3 统一单位：观测数据可能是mm，重建数据有Predicted_SMB_mm和Predicted_SMB_m
# 我们统一使用mm单位
if 'Predicted_SMB_mm' in recon_df.columns:
    recon_df['SMB_predicted_mm'] = recon_df['Predicted_SMB_mm']
elif 'Predicted_SMB_m' in recon_df.columns:
    recon_df['SMB_predicted_mm'] = recon_df['Predicted_SMB_m'] * 1000
else:
    raise ValueError("重建数据中找不到预测的SMB列")

print(f"  观测数据列: {obs_clean.columns.tolist()}")
print(f"  观测数据年份范围: {obs_clean['YEAR'].min()} - {obs_clean['YEAR'].max()}")

# ================= 3. 生成混合数据集 =================
print("\n>>> 步骤 3: 生成混合数据集...")

# 3.1 识别有观测的冰川
observed_glaciers = set(obs_clean['WGMS_ID'].unique())
all_glaciers = set(recon_df['WGMS_ID'].unique())
unobserved_glaciers = all_glaciers - observed_glaciers

print(f"  有观测的冰川: {len(observed_glaciers)} 个")
print(f"  无观测的冰川: {len(unobserved_glaciers)} 个")
print(f"  总冰川数: {len(all_glaciers)} 个")

# 3.2 创建观测数据的索引（用于快速查找）
# 为每个(WGMS_ID, YEAR)组合创建一个字典，存储观测值
obs_dict = {}
for _, row in obs_clean.iterrows():
    key = (row['WGMS_ID'], row['YEAR'])
    obs_dict[key] = row['ANNUAL_BALANCE']

print(f"  观测数据索引创建完成: {len(obs_dict)} 条记录")

# 3.3 遍历重建数据，生成混合数据集
hybrid_data = []
stats = {
    'observed_used': 0,      # 使用真实观测值的记录数
    'predicted_filled': 0,   # 用预测值填补的记录数（有观测冰川的缺测年份）
    'predicted_only': 0      # 完全使用预测值的记录数（无观测冰川）
}

for _, row in recon_df.iterrows():
    wgms_id = row['WGMS_ID']
    year = row['YEAR']
    key = (wgms_id, year)

    # 判断是否有观测值
    if key in obs_dict:
        # 情况1: 有真实观测值 → 使用观测值（蓝色）
        smb_value = obs_dict[key]
        data_source = 'observed'
        stats['observed_used'] += 1
    elif wgms_id in observed_glaciers:
        # 情况2: 有观测的冰川，但这一年缺测 → 使用预测值（红色填补）
        smb_value = row['SMB_predicted_mm']
        data_source = 'predicted_filled'
        stats['predicted_filled'] += 1
    else:
        # 情况3: 从未被观测过的冰川 → 使用预测值（红色）
        smb_value = row['SMB_predicted_mm']
        data_source = 'predicted_only'
        stats['predicted_only'] += 1

    # 构建混合数据记录
    hybrid_record = {
        'WGMS_ID': wgms_id,
        'NAME': row['NAME'],
        'POLITICAL_UNIT': row['POLITICAL_UNIT'],
        'YEAR': year,
        'LATITUDE': row['LATITUDE'],
        'LONGITUDE': row['LONGITUDE'],
        'AREA': row['AREA'],
        'LOWER_BOUND': row['LOWER_BOUND'],
        'UPPER_BOUND': row['UPPER_BOUND'],
        'SMB_mm': smb_value,
        'SMB_m': smb_value / 1000,
        'DATA_SOURCE': data_source
    }
    hybrid_data.append(hybrid_record)

# 转换为DataFrame
hybrid_df = pd.DataFrame(hybrid_data)

print(f"\n  混合数据集生成完成:")
print(f"    - 使用真实观测值: {stats['observed_used']} 条")
print(f"    - 预测值填补（有观测冰川缺测年份）: {stats['predicted_filled']} 条")
print(f"    - 预测值（无观测冰川）: {stats['predicted_only']} 条")
print(f"    - 总计: {len(hybrid_df)} 条")

# ================= 4. 保存混合数据集 =================
print("\n>>> 步骤 4: 保存混合数据集...")

output_file = os.path.join(RECONSTRUCTION_DIR, 'RGI02_Hybrid_Dataset.csv')
hybrid_df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"  混合数据集已保存到: {output_file}")

# 保存统计信息
stats_file = os.path.join(RECONSTRUCTION_DIR, 'RGI02_Hybrid_Dataset_Stats.txt')
with open(stats_file, 'w', encoding='utf-8') as f:
    f.write("混合数据集统计信息\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"总记录数: {len(hybrid_df)}\n")
    f.write(f"冰川数量: {hybrid_df['WGMS_ID'].nunique()}\n")
    f.write(f"年份范围: {hybrid_df['YEAR'].min()} - {hybrid_df['YEAR'].max()}\n\n")
    f.write("数据来源分布:\n")
    f.write(f"  - 真实观测值: {stats['observed_used']} 条 ({stats['observed_used']/len(hybrid_df)*100:.1f}%)\n")
    f.write(f"  - 预测值填补（有观测冰川）: {stats['predicted_filled']} 条 ({stats['predicted_filled']/len(hybrid_df)*100:.1f}%)\n")
    f.write(f"  - 预测值（无观测冰川）: {stats['predicted_only']} 条 ({stats['predicted_only']/len(hybrid_df)*100:.1f}%)\n\n")
    f.write(f"有观测的冰川: {len(observed_glaciers)} 个\n")
    f.write(f"无观测的冰川: {len(unobserved_glaciers)} 个\n")

print(f"  统计信息已保存到: {stats_file}")
