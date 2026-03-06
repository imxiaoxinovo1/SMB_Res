import pandas as pd
import os

# -----------------------------------------------------------------
# 脚本目标: Merge_mean_ela_imputed.py
# 1. 加载研究数据 (study_data_wna.csv)
# 2. 加载 WGMS state.csv (获取 mean_elevation)
# 3. 聚合 state.csv 以获取每个冰川唯一的静态 mean_elevation
# 4. 将 mean_elevation 合并到研究数据中
# 5. (关键) 填补 (Impute) mean_elevation 中的缺失值，
#    使用 (LOWER_BOUND + UPPER_BOUND) / 2 作为代理值
# -----------------------------------------------------------------

# --- 文件定义 ---
base_dir_wgms = r'H:\Code\SMB\WGMS\FoG_DataBase\DOI-WGMS-FoG-2025-02b\data'
file_wna = r'H:\Code\SMB\test\study_data_wna.csv'
file_state = os.path.join(base_dir_wgms, 'state.csv')      # Mean Elevation
file_output = r'H:\Code\SMB\test\result\study_data_wna_with_mean_ela.csv' # 输出文件名保持不变

try:
    # --- 1. 加载所有数据 ---
    print(f"正在加载研究数据: {file_wna}")
    df_wna = pd.read_csv(file_wna)
    print("加载成功。")
    
    print(f"正在加载WGMS冰川状态数据 (Mean Elevation): {file_state}")
    df_state = pd.read_csv(file_state)
    print("加载成功。")

    # --- 2. 准备 Mean Elevation (静态特征) ---
    print("正在处理 'state.csv' 以提取唯一的平均高程...")
    # 筛选出包含 mean_elevation 的有效行
    df_state_valid = df_state.dropna(subset=['glacier_id', 'mean_elevation'])
    
    # 按 glacier_id 分组，计算每个冰川的平均高程（作为代表性的静态值）
    df_mean_elev = df_state_valid.groupby('glacier_id')['mean_elevation'].mean().reset_index()
    
    print(f"已为 {len(df_mean_elev)} 个冰川计算了静态平均高程。")

    # --- 3. 执行合并 (添加 Mean Elevation) ---
    print("正在合并数据集 (添加 mean_elevation)...")
    # 合并是基于 静态的 WGMS_ID/glacier_id，不使用年份
    df_final_merged = pd.merge(
        df_wna,
        df_mean_elev,
        left_on='WGMS_ID',
        right_on='glacier_id',
        how='left'
    )
    print("Mean Elevation 合并完成。")

    # --- 4. (方案 B) 填补缺失的 mean_elevation ---
    print("正在填补 'mean_elevation' 的缺失值...")
    # 检查有多少缺失值
    missing_count_before = df_final_merged['mean_elevation'].isnull().sum()
    print(f"填补前 'mean_elevation' 缺失行数: {missing_count_before}")

    if missing_count_before > 0:
        # 使用 (LOWER_BOUND + UPPER_BOUND) / 2 作为代理值来填补
        proxy_elevation = (df_final_merged['LOWER_BOUND'] + df_final_merged['UPPER_BOUND']) / 2
        df_final_merged['mean_elevation'] = df_final_merged['mean_elevation'].fillna(proxy_elevation)
        
        missing_count_after = df_final_merged['mean_elevation'].isnull().sum()
        print(f"填补后 'mean_elevation' 缺失行数: {missing_count_after}")
        
        if missing_count_after > 0:
            print("警告: 仍然存在缺失的 'mean_elevation'。")
            print("这可能是因为 'LOWER_BOUND' 或 'UPPER_BOUND' 也存在缺失值。")
    else:
        print("数据中没有缺失的 'mean_elevation' 值，无需填补。")
    # --- 填补完成 ---

    # --- 5. 导出最终结果 ---
    print(f"正在将最终合并后的数据导出到: {file_output}")
    
    # 确保输出目录存在 
    os.makedirs(os.path.dirname(file_output), exist_ok=True)
    
    df_final_merged.to_csv(file_output, index=False)
    
    print("\n--- 处理成功 ---")
    print(f"文件 '{file_output}' 已成功保存在您的本地路径。")
    print("它包含了您 study_data_wna 的所有原始数据，以及匹配到的 'mean_elevation' (静态) 数据。")
    print("(缺失的 'mean_elevation' 已被 (LOWER_BOUND + UPPER_BOUND) / 2 填补)")

except FileNotFoundError as e:
    print(f"\n--- 错误：文件未找到 ---")
    print(f"错误: 找不到文件。 {e}")
    print("请仔细检查您的文件路径是否正确。")
except KeyError as e:
    print(f"\n--- 错误：找不到列 ---")
    print(f"错误: 在数据中找不到关键的列: {e}。")
    print(f"请检查您的CSV文件是否包含所需的关联键（例如 WGMS_ID, glacier_id, LOWER_BOUND, UPPER_BOUND）。")
except Exception as e:
    print(f"\n--- 发生意外错误 ---")
    print(f"错误: {e}")