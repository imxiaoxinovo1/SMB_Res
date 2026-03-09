"""
Step 04: 构建混合数据集（观测优先，预测填补）

数据源优先级:
  1. observed          — WGMS 实测年度物质平衡（TAG=9999，全冰川观测）
  2. predicted_filled  — 有观测冰川的缺测年份（LSTM 模型预测填补）
  3. predicted_only    — 无观测冰川的全部年份（仅 LSTM 模型预测）

输出: 03_reconstruction/results/
  RGI02_LSTM_Hybrid_Dataset.csv — 混合数据集（含 DATA_SOURCE 标记）
  lstm_coverage_stats.csv       — 逐冰川观测覆盖率统计
  lstm_dataset_stats.txt        — 全局统计摘要
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np

from config import (RECON_DIR, WGMS_MATCHED_CSV,
                    TRAIN_YEAR_MIN, TRAIN_YEAR_MAX, TARGET)

RESULTS_DIR = os.path.join(RECON_DIR, "results")
RECON_CSV   = os.path.join(RESULTS_DIR, "RGI02_LSTM_reconstruction.csv")
OUT_HYBRID  = os.path.join(RESULTS_DIR, "RGI02_LSTM_Hybrid_Dataset.csv")
OUT_COVERAGE= os.path.join(RESULTS_DIR, "lstm_coverage_stats.csv")
OUT_STATS   = os.path.join(RESULTS_DIR, "lstm_dataset_stats.txt")


# ===== 1. 读取数据 =====
print(">>> 1. 读取数据...")

# 观测数据（仅保留全冰川观测 TAG=9999）
obs_df = pd.read_csv(WGMS_MATCHED_CSV)
if 'TAG' in obs_df.columns:
    obs_df = obs_df[obs_df['TAG'] == 9999].copy()
obs_df = obs_df.dropna(subset=[TARGET])
obs_df = obs_df[(obs_df['YEAR'] >= TRAIN_YEAR_MIN) & (obs_df['YEAR'] <= TRAIN_YEAR_MAX)]

# 替换 WGMS 哨兵值 9999（与训练数据处理一致）
for col in ['LOWER_BOUND', 'UPPER_BOUND']:
    if col in obs_df.columns:
        n_sentinel = (obs_df[col] == 9999).sum()
        if n_sentinel > 0:
            obs_df[col] = obs_df[col].replace(9999, np.nan)
            print(f"   {col}: 已将 {n_sentinel} 个 9999 替换为 NaN")

print(f"   观测数据: {len(obs_df)} 条, {obs_df['WGMS_ID'].nunique()} 个冰川")
print(f"   观测年份范围: {obs_df['YEAR'].min()} - {obs_df['YEAR'].max()}")

# 重建数据
recon_df = pd.read_csv(RECON_CSV)
print(f"   重建数据: {len(recon_df)} 条, {recon_df['WGMS_ID'].nunique()} 个冰川")
print(f"   重建年份范围: {recon_df['YEAR'].min()} - {recon_df['YEAR'].max()}")


# ===== 2. 构建观测索引 =====
print("\n>>> 2. 构建观测索引...")

obs_dict = {
    (int(row['WGMS_ID']), int(row['YEAR'])): float(row[TARGET])
    for _, row in obs_df.iterrows()
}

observed_glacier_ids   = set(int(x) for x in obs_df['WGMS_ID'].unique())
all_glacier_ids        = set(int(x) for x in recon_df['WGMS_ID'].unique())
unobserved_glacier_ids = all_glacier_ids - observed_glacier_ids

print(f"   有观测冰川: {len(observed_glacier_ids)} 个")
print(f"   无观测冰川: {len(unobserved_glacier_ids)} 个")
print(f"   总冰川数:   {len(all_glacier_ids)} 个")


# ===== 3. 合并数据集 =====
print("\n>>> 3. 生成混合数据集...")

stats = {'observed': 0, 'predicted_filled': 0, 'predicted_only': 0}
hybrid_records = []

for _, row in recon_df.iterrows():
    wgms_id = int(row['WGMS_ID'])
    year    = int(row['YEAR'])
    key     = (wgms_id, year)

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
        'WGMS_ID':        wgms_id,
        'NAME':           row.get('NAME', ''),
        'POLITICAL_UNIT': row.get('POLITICAL_UNIT', ''),
        'YEAR':           year,
        'LATITUDE':       row['LATITUDE'],
        'LONGITUDE':      row['LONGITUDE'],
        'AREA':           row['AREA'],
        'LOWER_BOUND':    row['LOWER_BOUND'],
        'UPPER_BOUND':    row['UPPER_BOUND'],
        'SMB_mm':         round(float(smb_mm), 2),
        'SMB_m':          round(float(smb_mm) / 1000, 4),
        'DATA_SOURCE':    source,
    })

hybrid_df = pd.DataFrame(hybrid_records)
hybrid_df.sort_values(['WGMS_ID', 'YEAR'], inplace=True)

total = len(hybrid_df)
print(f"\n   混合数据集:")
print(f"   - 真实观测值:       {stats['observed']:>6} ({stats['observed']/total*100:.1f}%)")
print(f"   - 预测填补(有观测):  {stats['predicted_filled']:>6} ({stats['predicted_filled']/total*100:.1f}%)")
print(f"   - 预测(无观测):     {stats['predicted_only']:>6} ({stats['predicted_only']/total*100:.1f}%)")
print(f"   - 总计:             {total:>6}")


# ===== 4. 保存混合数据集 =====
hybrid_df.to_csv(OUT_HYBRID, index=False, encoding='utf-8-sig')
print(f"\n>>> 已保存: {OUT_HYBRID}")


# ===== 5. 逐冰川覆盖率统计 =====
print(">>> 4. 计算观测覆盖率...")

coverage_data = []
for glacier_id in sorted(observed_glacier_ids):
    g_data = hybrid_df[hybrid_df['WGMS_ID'] == glacier_id]
    if len(g_data) == 0:
        continue
    n_total  = len(g_data)
    n_obs    = (g_data['DATA_SOURCE'] == 'observed').sum()
    n_filled = (g_data['DATA_SOURCE'] == 'predicted_filled').sum()
    name     = (g_data['NAME'].dropna().iloc[0]
                if not g_data['NAME'].dropna().empty else '')
    coverage_data.append({
        'WGMS_ID':        glacier_id,
        'NAME':           name,
        'Total_Years':    n_total,
        'Observed_Years': n_obs,
        'Filled_Years':   n_filled,
        'Coverage_Rate':  f"{n_obs/n_total*100:.1f}%",
    })

coverage_df = (pd.DataFrame(coverage_data)
               .sort_values('Observed_Years', ascending=False))
coverage_df.to_csv(OUT_COVERAGE, index=False, encoding='utf-8-sig')
print(f"   已保存: {OUT_COVERAGE}")


# ===== 6. 全局统计摘要 =====
smb_obs    = hybrid_df.loc[hybrid_df['DATA_SOURCE'] == 'observed', 'SMB_m']
smb_pred   = hybrid_df.loc[hybrid_df['DATA_SOURCE'] != 'observed', 'SMB_m']

lines = [
    "RGI02 LSTM 混合数据集统计信息",
    "=" * 55,
    f"总记录数:   {total}",
    f"冰川数量:   {hybrid_df['WGMS_ID'].nunique()}",
    f"年份范围:   {hybrid_df['YEAR'].min()} - {hybrid_df['YEAR'].max()}",
    "",
    "数据来源分布:",
    f"  - 真实观测值:          {stats['observed']:>6} ({stats['observed']/total*100:.1f}%)",
    f"  - 预测填补(有观测冰川): {stats['predicted_filled']:>6} ({stats['predicted_filled']/total*100:.1f}%)",
    f"  - 预测(无观测冰川):    {stats['predicted_only']:>6} ({stats['predicted_only']/total*100:.1f}%)",
    "",
    f"有观测的冰川: {len(observed_glacier_ids)} 个",
    f"无观测的冰川: {len(unobserved_glacier_ids)} 个",
    "",
    "SMB 统计 (m w.e.) — 全体:",
    f"  均值:   {hybrid_df['SMB_m'].mean():.3f}",
    f"  标准差: {hybrid_df['SMB_m'].std():.3f}",
    f"  最小值: {hybrid_df['SMB_m'].min():.3f}",
    f"  最大值: {hybrid_df['SMB_m'].max():.3f}",
    "",
    "SMB 统计 (m w.e.) — 观测子集:",
    f"  均值:   {smb_obs.mean():.3f}",
    f"  标准差: {smb_obs.std():.3f}",
    "",
    "SMB 统计 (m w.e.) — 预测子集:",
    f"  均值:   {smb_pred.mean():.3f}",
    f"  标准差: {smb_pred.std():.3f}",
    "=" * 55,
]

for line in lines:
    print(line)

with open(OUT_STATS, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print(f"\n>>> 已保存: {OUT_STATS}")
print("\nOK  混合数据集构建完成！")
