"""
统一配置文件 - RGI02 冰川物质平衡重建项目
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
# data_glacier_era5_fixed.csv: TAG=9999 海拔已修正，特征更完整 (31个)
TRAINING_DATA_CSV = r"H:\Code\SMB\test\data\study_test\data_glacier_era5_fixed.csv"

# 训练年份范围 (与 test_rf_fixed.py 保持一致)
TRAIN_YEAR_MIN = 1980
TRAIN_YEAR_MAX = 2024

# ============= 输出路径 =============
BASE_DIR = r"H:\Code\SMB\SMB_Res_RF_ByClaude"
PREPROCESS_DATA_DIR = os.path.join(BASE_DIR, "01_preprocessing", "data")
MODEL_RESULTS_DIR = os.path.join(BASE_DIR, "02_model", "results")
RECON_RESULTS_DIR = os.path.join(BASE_DIR, "03_reconstruction", "results")
RECON_FIGURES_DIR = os.path.join(BASE_DIR, "03_reconstruction", "figures")

# ============= 区域过滤 =============
TARGET_COUNTRIES = ['US', 'CA']
LATITUDE_MIN = 30.0   # 排除低纬度非冰川区
LATITUDE_MAX = 60.0   # 排除北极冰川 (RGI03/04)

# ============= ERA5 变量映射 =============
# 格式: 'NetCDF变量名': ('输出列基础名', '聚合方式')
VAR_MAPPING = {
    # 状态量 (均值聚合)
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
    # 通量 (累加聚合)
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

# ============= 模型特征 (31个, 与 test_rf_fixed.py 对齐) =============
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

# ============= RF 超参数 (全流程统一) =============
RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': None,
    'random_state': 42,
    'n_jobs': -1,
}
