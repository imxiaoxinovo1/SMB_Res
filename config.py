"""Central configuration for SMB_Res_ByClaudeV2."""
import os

# ── 根目录 ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
RESULT_DIR = os.path.join(BASE_DIR, "results")
FIG_DIR    = os.path.join(BASE_DIR, "figures")

for d in [DATA_DIR, RESULT_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

# ── 外部数据路径 ───────────────────────────────────────────────────────────────
WGMS_DIR       = r"H:\Code\SMB\WGMS\FoG_DataBase\DOI-WGMS-FoG-2025-02b\data"
GLACIER_CSV    = os.path.join(WGMS_DIR, "glacier.csv")
MASSBAL_CSV    = os.path.join(WGMS_DIR, "mass_balance.csv")
RGI_SHP        = r"H:\Code\SMB\RGI\RGI2000-v7.0-G-02_western_canada_usa\RGI2000-v7.0-G-02_western_canada_usa.shp"
RGI_LINKS_CSV  = r"H:\Code\SMB\RGI\RGI2000-v7.0-G-02_western_canada_usa\RGI2000-v7.0-G-02_western_canada_usa-rgi6_links.csv"
ERA5_NC        = r"H:\Code\SMB\ERA5-LAND\data_stream-moda.nc"
HUGONNET_RATES = r"H:\Code\SMB\Hugonnet_results\time_series_02\dh_02_rgi60_pergla_rates.csv"

# ── 预处理输出路径 ─────────────────────────────────────────────────────────────
TRAINING_GLACIERS_CSV  = os.path.join(DATA_DIR, "training_glaciers.csv")
MASSBAL_RGI02_CSV      = os.path.join(DATA_DIR, "massbal_rgi02.csv")
TERRAIN_CSV            = os.path.join(DATA_DIR, "training_glaciers_terrain.csv")
ERA5_MONTHLY_CSV       = os.path.join(DATA_DIR, "era5_monthly_training.csv")
TABULAR_CSV            = os.path.join(DATA_DIR, "tabular_dataset.csv")
SEQUENCES_NPZ          = os.path.join(DATA_DIR, "sequences_training.npz")
SELECTED_VARS_JSON     = os.path.join(DATA_DIR, "selected_vars.json")

# ── 重建数据输出路径 ───────────────────────────────────────────────────────────
RGI02_TARGET_CSV       = os.path.join(DATA_DIR, "rgi02_target_glaciers.csv")
ERA5_RGI02_CSV         = os.path.join(DATA_DIR, "era5_monthly_rgi02.csv")

# ── 时间范围 ───────────────────────────────────────────────────────────────────
TRAIN_YEAR_MIN    = 1950
TRAIN_YEAR_MAX    = 2023     # 扩展至 2023，包含 WGMS 2025-02b 最新观测
HOLDOUT_YEAR_MIN  = None     # 已合并进训练集，LOYO 提供足够严格的时间验证
HOLDOUT_YEAR_MAX  = None
RECON_YEAR_MIN    = 1950
RECON_YEAR_MAX    = 2024

# ── ERA5 气候变量（15个）─────────────────────────────────────────────────────
MONTHLY_CLIMATE_VARS = [
    't2m', 'skt', 'd2m',
    'sd', 'asn',
    'tp', 'sf', 'smlt',
    'ssrd', 'strd', 'ssr', 'str',
    'slhf', 'sshf',
    'ro',
]
N_DYNAMIC = len(MONTHLY_CLIMATE_VARS)   # 15

# 静态特征（10个）：来自 RGI v7 地形 + 派生编码
# aspect_sin / aspect_cos 由 aspect_deg 在预处理时计算，非 shapefile 原始字段
STATIC_FEATURES = [
    'slope_deg', 'aspect_sin', 'aspect_cos',
    'zmin_m', 'zmax_m', 'zmean_m', 'zmed_m',
    'area_km2', 'lmax_m', 'cenlat',
]

# 季节聚合特征（8个）：从 tabular 数据集追加到静态分支，提供显式季节先验
# 使 GlacioFormer 无需从注意力中从头学习季节规律
SEASONAL_EXTRA_FEATURES = [
    'cal_summer_t2m_mean',   # 夏季气温（消融主驱动）
    'cal_winter_tp_sum',     # 冬季降水（积累主驱动）
    'hyd_ablat_t2m_mean',    # 水文消融季气温
    'hyd_accum_sf_sum',      # 积累季降雪量
    'hyd_ablat_smlt_sum',    # 消融季融雪量
    'ann_t2m_mean',          # 年均气温
    'ann_tp_sum',            # 年总降水
    'cal_summer_ssrd_sum',   # 夏季太阳辐射
]

N_STATIC = len(STATIC_FEATURES) + len(SEASONAL_EXTRA_FEATURES)   # 10 + 8 = 18

# ── 日历年季节月份 ─────────────────────────────────────────────────────────────
CAL_SUMMER_MONTHS  = [6, 7, 8]
CAL_WINTER_MONTHS  = [12, 1, 2]
HYD_ACCUM_MONTHS   = [10, 11, 12, 1, 2, 3, 4]
HYD_ABLAT_MONTHS   = [5, 6, 7, 8, 9]

# ── XGBoost 超参数 ─────────────────────────────────────────────────────────────
XGB_PARAMS = dict(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, n_jobs=-1,
)

# ── GlacioFormer 超参数 ────────────────────────────────────────────────────────
GLACIOFORMER_PARAMS = dict(
    n_dynamic_features=N_DYNAMIC,
    n_static_features=N_STATIC,
    d_model=32, n_heads=4, n_encoder_layers=1, ff_dim=64,   # 缩小适配1000样本
    dropout=0.35,                                            # 增强正则化
    batch_size=32, epochs=300, lr=1e-3,
    early_stop_patience=40, min_epochs=60,
    weight_decay=1e-4,
)

assert GLACIOFORMER_PARAMS['d_model'] % GLACIOFORMER_PARAMS['n_heads'] == 0, \
    "d_model must be divisible by n_heads"

# ── 重建冰川面积筛选阈值 ───────────────────────────────────────────────────────
MIN_AREA_KM2 = 0.5
