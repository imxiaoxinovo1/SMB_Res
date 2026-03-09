"""
SMB_Res_RF_ByGemini — 统一路径配置
修改 BASE_DIR 和外部数据路径后，所有脚本均可正常运行
"""
import os

# ─── 项目根目录 ───────────────────────────────────────────────────────────
BASE_DIR = r"H:\Code\SMB\SMB_Res_RF_ByGemini"

# ─── 外部数据（大型文件，不上传 GitHub） ──────────────────────────────────
WGMS_DATA_DIR = r"H:\Code\SMB\2025-02b\data"           # WGMS FoG 数据库
ERA5_NC_PATH  = r"H:\Code\SMB\test\data\ERA5-LAND\data_stream-moda.nc"
STUDY_REF_CSV = r"H:\Code\SMB\test\study_data_wna.csv"  # 师姐基准 CSV（test.py 用）

# ─── 项目内部数据目录（输出路径，.csv 被 .gitignore 排除） ────────────────
DATA_DIR          = os.path.join(BASE_DIR, "data")
MERGE_DIR         = os.path.join(DATA_DIR, "merge")
STUDY_TEST_DIR    = os.path.join(DATA_DIR, "study_test")
RECONSTRUCTION_DIR= os.path.join(DATA_DIR, "reconstruction")
RESULTS_DIR       = os.path.join(DATA_DIR, "results")

# ─── Figure9.py 专用（外部比较数据） ─────────────────────────────────────
ZEMP_CSV           = r"H:\Code\SMB\Zemp_results\Zemp_etal_DataTables2a-t_results_regions_global\Zemp_etal_results_region_2_WNA.csv"
OLD_RF_RESULT_CSV  = r"H:\Code\SMB\test\result\pred_result_time_wna.csv"

# ─── 目录自动创建 ─────────────────────────────────────────────────────────
for _d in [MERGE_DIR, STUDY_TEST_DIR, RECONSTRUCTION_DIR, RESULTS_DIR]:
    os.makedirs(_d, exist_ok=True)
