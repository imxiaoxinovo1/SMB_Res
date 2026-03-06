"""
统一配置文件 - RGI02 冰川物质平衡重建项目 (XGBoost / LightGBM)
所有路径、特征列表、超参数、变量映射集中管理
"""
import os

# ============= 数据源路径 (只读) =============
WGMS_DATA_DIR = r"H:\Code\SMB\2025-02b\data"
GLACIER_CSV = os.path.join(WGMS_DATA_DIR, "glacier.csv")
STATE_CSV = os.path.join(WGMS_DATA_DIR, "state.csv")
MASS_BALANCE_CSV = os.path.join(WGMS_DATA_DIR, "mass_balance.csv")
MASS_BALANCE_BAND_CSV = os.path.join(WGMS_DATA_DIR, "mass_balance_band.csv")

ERA5_NC_PATH = r"H:\Code\SMB\test\data\ERA5-LAND\data_stream-moda.nc"

# 已预处理的训练数据 (含分带和全冰川记录, ANNUAL_BALANCE 单位 mm w.e.)
TRAINING_DATA_CSV = r"H:\Code\SMB\test\data\study_test\data_glacier_era5_fixed.csv"

# 重建输入数据 (来自 RF 项目预处理)
RECONSTRUCTION_INPUT_CSV = r"H:\Code\SMB\SMB_Res_RF_ByClaude\01_preprocessing\data\rgi02_glaciers_era5.csv"

# 训练年份范围
TRAIN_YEAR_MIN = 1980
TRAIN_YEAR_MAX = 2024

# ============= 输出路径 =============
BASE_DIR = r"H:\Code\SMB\SMB_Res_XGBOOST_ByClaude"
MODEL_RESULTS_DIR = os.path.join(BASE_DIR, "02_model", "results")
RECON_RESULTS_DIR = os.path.join(BASE_DIR, "03_reconstruction", "results")
RECON_FIGURES_DIR = os.path.join(BASE_DIR, "03_reconstruction", "figures")

# RF 项目结果路径 (用于模型对比)
RF_RESULTS_DIR = r"H:\Code\SMB\SMB_Res_RF_ByClaude\02_model\results"

# ============= 区域过滤 =============
TARGET_COUNTRIES = ['US', 'CA']
LATITUDE_MIN = 30.0
LATITUDE_MAX = 60.0

# ============= ERA5 变量映射 =============
VAR_MAPPING = {
    't2m':    ('temperature_2m', 'mean'),
    'd2m':    ('dewpoint_temperature_2m', 'mean'),
    'skt':    ('skin_temperature', 'mean'),
    'sp':     ('surface_pressure', 'mean'),
    'fal':    ('forecast_albedo', 'mean'),
    'asn':    ('snow_albedo', 'mean'),
    'rsn':    ('snow_density', 'mean'),
    'sd':     ('snow_depth', 'mean'),
    'lai_hv': ('leaf_area_index_high_vegetation', 'mean'),
    'lai_lv': ('leaf_area_index_low_vegetation', 'mean'),
    'tp':     ('total_precipitation', 'sum'),
    'sf':     ('snowfall', 'sum'),
    'smlt':   ('snowmelt', 'sum'),
    'ssrd':   ('surface_solar_radiation_downwards', 'sum'),
    'strd':   ('surface_thermal_radiation_downwards', 'sum'),
    'ssr':    ('surface_net_solar_radiation', 'sum'),
    'str':    ('surface_net_thermal_radiation', 'sum'),
    'slhf':   ('surface_latent_heat_flux', 'sum'),
    'sshf':   ('surface_sensible_heat_flux', 'sum'),
    'e':      ('total_evaporation', 'sum'),
    'pev':    ('potential_evaporation', 'sum'),
    'ro':     ('runoff', 'sum'),
    'sro':    ('surface_runoff', 'sum'),
    'ssro':   ('sub_surface_runoff', 'sum'),
    'es':     ('snow_evaporation', 'sum'),
}

SUMMER_MONTHS = [5, 6, 7, 8, 9]

# ============= 模型特征 (31个, 与 RF 项目对齐) =============
FEATURES = [
    "LOWER_BOUND", "UPPER_BOUND", "AREA", "LATITUDE", "LONGITUDE",
    "temperature_2m_year", "temperature_2m_summer",
    "skin_temperature_year", "skin_temperature_summer",
    "dewpoint_temperature_2m_summer",
    "total_precipitation_sum_year", "total_precipitation_sum_summer",
    "snowfall_sum_year", "snowfall_sum_summer",
    "snow_depth_year", "snow_depth_summer",
    "snow_density_summer", "snow_albedo_summer",
    "surface_net_solar_radiation_sum_summer",
    "surface_net_thermal_radiation_sum_summer",
    "surface_solar_radiation_downwards_sum_summer",
    "surface_thermal_radiation_downwards_sum_summer",
    "surface_sensible_heat_flux_sum_summer",
    "surface_latent_heat_flux_sum_summer",
    "total_evaporation_sum_year", "total_evaporation_sum_summer",
    "snow_evaporation_sum_summer",
    "runoff_sum_summer",
    "snowmelt_sum_year", "snowmelt_sum_summer",
    "YEAR"
]
TARGET = "ANNUAL_BALANCE"  # mm w.e.

# ============= XGBoost 超参数 =============
XGB_PARAMS = {
    'n_estimators': 500,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
}

# ============= LightGBM 超参数 =============
LGB_PARAMS = {
    'n_estimators': 500,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'num_leaves': 31,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
}

# ============= RF 参考超参数 (用于 compare_models.py 对比) =============
RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': None,
    'random_state': 42,
    'n_jobs': -1,
}
