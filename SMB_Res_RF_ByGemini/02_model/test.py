import pandas as pd
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import STUDY_REF_CSV, MERGE_DIR, STUDY_TEST_DIR

# ================= 1. 配置路径 =================
# 路径由 config.py 统一管理
ref_csv_path = STUDY_REF_CSV

# 你的待处理数据
target_csv_path = os.path.join(MERGE_DIR, 'data_glacier_era5_cleaned.csv')

# 输出目录和文件名
output_dir = STUDY_TEST_DIR
output_filename = "data_glacier_era5_matched.csv"
output_path = os.path.join(output_dir, output_filename)

# 确保输出目录存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"创建目录: {output_dir}")

# ================= 2. 读取数据 =================
print(">>> 1. 读取数据...")

# 读取师姐的数据 (只需要 WGMS_ID 列)
try:
    df_ref = pd.read_csv(ref_csv_path, usecols=['WGMS_ID'])
    # 获取唯一的 ID 列表
    valid_ids = df_ref['WGMS_ID'].unique()
    print(f"   基准表 (师姐) 中的唯一冰川数: {len(valid_ids)}")
except ValueError:
    # 如果读取出错(比如列名不叫WGMS_ID)，尝试读取全部再找列名
    df_ref = pd.read_csv(ref_csv_path)
    if 'WGMS_ID' in df_ref.columns:
        valid_ids = df_ref['WGMS_ID'].unique()
        print(f"   基准表 (师姐) 中的唯一冰川数: {len(valid_ids)}")
    else:
        print("❌ 错误: 师姐的数据表中找不到 'WGMS_ID' 列，请检查列名！")
        exit()

# 读取你的数据
df_target = pd.read_csv(target_csv_path)
print(f"   目标表 (你的) 原始行数: {len(df_target)}")
print(f"   目标表中的唯一冰川数: {df_target['WGMS_ID'].nunique()}")

# ================= 3. 执行筛选 (ID 匹配) =================
print("\n>>> 2. 执行 ID 匹配筛选...")

# 核心操作：isin()
# 保留那些 WGMS_ID 存在于 valid_ids 列表中的行
df_filtered = df_target[df_target['WGMS_ID'].isin(valid_ids)].copy()

filtered_count = len(df_filtered)
removed_count = len(df_target) - filtered_count

print(f"   筛选后行数: {filtered_count}")
print(f"   删除了行数: {removed_count}")
print(f"   筛选后唯一冰川数: {df_filtered['WGMS_ID'].nunique()}")

# ================= 4. 保存结果 =================
print(f"\n>>> 3. 保存结果到: {output_path}")
df_filtered.to_csv(output_path, index=False)

print("-" * 30)
print("✅ 完成！")
print(f"新文件仅包含师姐研究中涉及的 {len(valid_ids)} 个冰川的数据。")
print("-" * 30)