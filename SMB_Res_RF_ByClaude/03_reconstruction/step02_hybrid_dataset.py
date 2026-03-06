"""
Step 02: 构建混合数据集
- 观测优先: 有实测值的用实测, 缺测的用模型预测填补
- 数据源标记: observed / predicted_filled / predicted_only
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from config import TRAINING_DATA_CSV, RECON_RESULTS_DIR

output_hybrid = os.path.join(RECON_RESULTS_DIR, "RGI02_Hybrid_Dataset.csv")
output_coverage = os.path.join(RECON_RESULTS_DIR, "coverage_stats.csv")
output_stats = os.path.join(RECON_RESULTS_DIR, "dataset_stats.txt")

# ================= 1. 读取数据 =================
print(">>> 1. 读取数据...")

# 观测数据
obs_df = pd.read_csv(TRAINING_DATA_CSV)
obs_df.columns = [c.replace('.1', '') for c in obs_df.columns]
obs_df = obs_df.loc[:, ~obs_df.columns.duplicated()]

# 仅保留全冰川观测 (TAG=9999)
if 'TAG' in obs_df.columns:
    obs_clean = obs_df[obs_df['TAG'] == 9999].copy()
else:
    obs_clean = obs_df.copy()

print(f"   观测数据: {len(obs_clean)} 条, {obs_clean['WGMS_ID'].nunique()} 个冰川")

# 重建数据
recon_csv = os.path.join(RECON_RESULTS_DIR, "RGI02_reconstruction.csv")
recon_df = pd.read_csv(recon_csv)
print(f"   重建数据: {len(recon_df)} 条, {recon_df['WGMS_ID'].nunique()} 个冰川")

# ================= 2. 构建观测索引 =================
print("\n>>> 2. 构建观测索引...")
obs_dict = {}
for _, row in obs_clean.iterrows():
    key = (row['WGMS_ID'], row['YEAR'])
    obs_dict[key] = row['ANNUAL_BALANCE']  # mm w.e.

observed_glacier_ids = set(obs_clean['WGMS_ID'].unique())
all_glacier_ids = set(recon_df['WGMS_ID'].unique())
unobserved_glacier_ids = all_glacier_ids - observed_glacier_ids

print(f"   有观测的冰川: {len(observed_glacier_ids)} 个")
print(f"   无观测的冰川: {len(unobserved_glacier_ids)} 个")

# ================= 3. 合并数据集 =================
print("\n>>> 3. 生成混合数据集...")

stats = {'observed': 0, 'predicted_filled': 0, 'predicted_only': 0}
hybrid_records = []

for _, row in recon_df.iterrows():
    wgms_id = row['WGMS_ID']
    year = row['YEAR']
    key = (wgms_id, year)

    if key in obs_dict:
        smb_mm = obs_dict[key]
        source = 'observed'
    elif wgms_id in observed_glacier_ids:
        smb_mm = row['PREDICTED_SMB_mm']
        source = 'predicted_filled'
    else:
        smb_mm = row['PREDICTED_SMB_mm']
        source = 'predicted_only'

    stats[source] += 1
    hybrid_records.append({
        'WGMS_ID': wgms_id,
        'NAME': row.get('NAME', ''),
        'POLITICAL_UNIT': row.get('POLITICAL_UNIT', ''),
        'YEAR': year,
        'LATITUDE': row['LATITUDE'],
        'LONGITUDE': row['LONGITUDE'],
        'AREA': row['AREA'],
        'LOWER_BOUND': row['LOWER_BOUND'],
        'UPPER_BOUND': row['UPPER_BOUND'],
        'SMB_mm': smb_mm,
        'SMB_m': smb_mm / 1000,
        'DATA_SOURCE': source,
    })

hybrid_df = pd.DataFrame(hybrid_records)
hybrid_df.sort_values(['WGMS_ID', 'YEAR'], inplace=True)

total = len(hybrid_df)
print(f"\n   混合数据集:")
print(f"   - 真实观测值:     {stats['observed']:>6} ({stats['observed']/total*100:.1f}%)")
print(f"   - 预测填补(有观测): {stats['predicted_filled']:>6} ({stats['predicted_filled']/total*100:.1f}%)")
print(f"   - 预测(无观测):    {stats['predicted_only']:>6} ({stats['predicted_only']/total*100:.1f}%)")
print(f"   - 总计:           {total:>6}")

# ================= 4. 保存混合数据集 =================
hybrid_df.to_csv(output_hybrid, index=False, encoding='utf-8-sig')
print(f"\n>>> 已保存: {output_hybrid}")

# ================= 5. 逐冰川覆盖率统计 =================
print(">>> 4. 计算覆盖率...")
coverage_data = []

for glacier_id in observed_glacier_ids:
    g_data = hybrid_df[hybrid_df['WGMS_ID'] == glacier_id]
    if len(g_data) == 0:
        continue
    n_total = len(g_data)
    n_obs = (g_data['DATA_SOURCE'] == 'observed').sum()
    n_filled = (g_data['DATA_SOURCE'] == 'predicted_filled').sum()
    name = g_data['NAME'].dropna().iloc[0] if not g_data['NAME'].dropna().empty else ''
    coverage_data.append({
        'WGMS_ID': glacier_id,
        'NAME': name,
        'Total_Years': n_total,
        'Observed_Years': n_obs,
        'Filled_Years': n_filled,
        'Coverage_Rate': f"{n_obs/n_total*100:.1f}%"
    })

coverage_df = pd.DataFrame(coverage_data).sort_values('Observed_Years', ascending=False)
coverage_df.to_csv(output_coverage, index=False, encoding='utf-8-sig')
print(f"   已保存: {output_coverage}")

# ================= 6. 保存统计信息 =================
with open(output_stats, 'w', encoding='utf-8') as f:
    f.write("RGI02 混合数据集统计信息\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"总记录数: {total}\n")
    f.write(f"冰川数量: {hybrid_df['WGMS_ID'].nunique()}\n")
    f.write(f"年份范围: {hybrid_df['YEAR'].min()} - {hybrid_df['YEAR'].max()}\n\n")
    f.write("数据来源分布:\n")
    f.write(f"  - 真实观测值: {stats['observed']} ({stats['observed']/total*100:.1f}%)\n")
    f.write(f"  - 预测填补(有观测冰川): {stats['predicted_filled']} ({stats['predicted_filled']/total*100:.1f}%)\n")
    f.write(f"  - 预测(无观测冰川): {stats['predicted_only']} ({stats['predicted_only']/total*100:.1f}%)\n\n")
    f.write(f"有观测的冰川: {len(observed_glacier_ids)} 个\n")
    f.write(f"无观测的冰川: {len(unobserved_glacier_ids)} 个\n\n")
    f.write(f"SMB 统计 (m w.e.):\n")
    f.write(f"  均值: {hybrid_df['SMB_m'].mean():.3f}\n")
    f.write(f"  标准差: {hybrid_df['SMB_m'].std():.3f}\n")
    f.write(f"  最小值: {hybrid_df['SMB_m'].min():.3f}\n")
    f.write(f"  最大值: {hybrid_df['SMB_m'].max():.3f}\n")

print(f"   已保存: {output_stats}")
print("\n>>> 完成!")
