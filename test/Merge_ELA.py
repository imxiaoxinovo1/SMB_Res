import pandas as pd
import os

# This script merges the study data with WGMS mass balance data to add elevation information.
# --- 文件定义 ---
# 解决方案：在路径字符串前加 'r' 来处理反斜杠
file_wna = r'H:\Code\SMB\test\study_data_wna.csv'
file_mb = r'H:\Code\SMB\WGMS\FoG_DataBase\DOI-WGMS-FoG-2025-02b\data\mass_balance.csv' 
file_output = r'H:\Code\SMB\test\result\study_data_wna_with_elevation.csv'

try:
    #1. 加载数据 
    print(f"正在加载研究数据: {file_wna}")
    df_wna = pd.read_csv(file_wna)
    print("加载成功。")

    print(f"正在加载WGMS物质平衡数据: {file_mb}")
    df_mb = pd.read_csv(file_mb)
    print("加载成功。")

    # 2. 准备合并
    # 检查关联键
    print(f"df_wna (您的数据) 列: {df_wna.columns.to_list()}")
    print(f"df_mb (WGMS数据) 列: {df_mb.columns.to_list()}")

    # 关联键
    left_keys = ['WGMS_ID', 'YEAR']
    # WGMS 文件的关联键
    right_keys = ['glacier_id', 'year']

    #执行合并
    print("正在合并两个数据集...")
    # 使用 'left' 合并，以保留您 study_data_wna 中的所有原始行
    df_merged = pd.merge(
        df_wna,
        df_mb,
        left_on=left_keys,
        right_on=right_keys,
        how='left'
    )
    print("合并完成。")

    # 4.提取所需的高程信息
    # 需要 'ela' (Equilibrium Line Altitude) 作为高程信息
    # 复制了 'ela' 列，使其在您的表中更清晰
    if 'ela' in df_merged.columns:
        df_merged['ELA'] = df_merged['ela'] 
        print("已成功将 'ela' 列复制到新的 'ELA' 列。")
    else:
        print("警告: 合并后的数据中未找到 'ela' 列。")

    #5. 导出结果
    print(f"正在将合并后的数据导出到: {file_output}")
    
    # 确保输出目录存在 
    os.makedirs(os.path.dirname(file_output), exist_ok=True)
    
    df_merged.to_csv(file_output, index=False)
    
    print("\n--- 处理成功 ---")
    print(f"文件 '{file_output}' 已成功保存在您的本地路径。")
    print("它包含了您 study_data_wna 的所有原始数据，以及匹配到的高程数据（例如 'ela'）。")

except FileNotFoundError as e:
    print(f"\n--- 错误：文件未找到 ---")
    print(f"错误: 找不到文件。 {e}")
    print("请仔细检查您的 'file_wna' 和 'file_mb' 路径是否正确。")
except KeyError as e:
    print(f"\n--- 错误：找不到列 ---")
    print(f"错误: 在数据中找不到关键的列: {e}。")
    print(f"请检查您的 '{file_wna}' 是否包含 {left_keys}")
    print(f"以及 '{file_mb}' 是否包含 {right_keys}")
except Exception as e:
    print(f"\n--- 发生意外错误 ---")
    print(f"错误: {e}")