import pandas as pd
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import MERGE_DIR



#步骤04:
#这个代码文件用于清理合并了ERA5数据但包含大量空行的冰川数据文件




# ================= 1. 配置路径 =================
# 路径由 config.py 统一管理
input_file_path = os.path.join(MERGE_DIR, 'data_glacier_era5.csv')

# 输出文件 (清洗后的新文件)
output_file_path = os.path.join(MERGE_DIR, 'data_glacier_era5_cleaned.csv')

# ================= 2. 执行清理 =================
print(f">>> 1. 正在读取文件: {input_file_path} ...")
if not os.path.exists(input_file_path):
    print(f"❌ 错误: 找不到文件 {input_file_path}")
    exit()

df = pd.read_csv(input_file_path)
original_count = len(df)
print(f"   原始样本数: {original_count}")

# 检查空值情况
null_count = df['ANNUAL_BALANCE'].isnull().sum()
print(f"   发现缺少 ANNUAL_BALANCE 的行数: {null_count}")

# ================= 3. 剔除并保存 =================
if null_count > 0:
    print(">>> 2. 正在剔除无效行...")
    
    # 核心操作：删除指定列为空的行
    df_clean = df.dropna(subset=['ANNUAL_BALANCE'])
    
    # 保存为新文件
    print(f">>> 3. 保存结果到: {output_file_path}")
    df_clean.to_csv(output_file_path, index=False)
    
    final_count = len(df_clean)
    
    # ================= 4. 结果报告 =================
    print("-" * 40)
    print(f"✅ 清理完成！")
    print(f"   原始行数 : {original_count}")
    print(f"   剩余行数 : {final_count} (有效观测数据)")
    print(f"   剔除行数 : {original_count - final_count}")
    print(f"   新文件位置: {output_file_path}")
    print("-" * 40)
    
    # 验证是否接近师姐的数据量级
    if 1600 <= final_count <= 2000:
        print("🎉 完美！现在的样本量与师姐的数据量级 (1672) 非常接近。")
        print("   数据对齐成功，可以进行模型训练了！")
    else:
        print(f"⚠️ 提示: 当前样本数 ({final_count}) 与师姐的 (1672) 仍有一定差异，请检查是否还有其他筛选条件。")

else:
    print("✅ 文件中没有发现空值，无需清理。")
    # 如果没变，也顺便存一份新的以防万一
    df.to_csv(output_file_path, index=False)
    print(f"   已将原数据另存为: {output_file_path}")