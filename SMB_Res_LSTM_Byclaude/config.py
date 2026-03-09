"""
SMB_Res_LSTM_ByClaude — 统一路径与模型配置

双分支网络架构:
  时序分支 (LSTM) : 12 个月的气象序列 → 时序记忆向量
  空间分支 (MLP)  : 5 个静态地形特征  → 空间几何向量
  融合层          : Concat + Dense → 年度物质平衡 (SMB)

修改 BASE_DIR 和外部数据路径后，所有脚本均可正常运行。
"""
import os

# ─── 外部数据（大文件，本地保留，不上传 GitHub） ─────────────────────────────
WGMS_DATA_DIR    = r"H:\Code\SMB\2025-02b\data"
ERA5_NC_PATH     = r"H:\Code\SMB\test\data\ERA5-LAND\data_stream-moda.nc"

# 已有观测记录的训练 CSV（来自 RF 管道，含 ANNUAL_BALANCE）
# 字段: WGMS_ID, YEAR, LOWER_BOUND, UPPER_BOUND, AREA, LATITUDE, LONGITUDE,
#       ANNUAL_BALANCE, TAG, ... (年度/夏季 ERA5 特征)
WGMS_MATCHED_CSV = r"H:\Code\SMB\test\data\study_test\data_glacier_era5_fixed.csv"

# ─── 项目根目录 ───────────────────────────────────────────────────────────────
BASE_DIR       = r"H:\Code\SMB\SMB_Res_LSTM_ByClaude"
PREPROCESS_DIR = os.path.join(BASE_DIR, "01_preprocessing", "data")
MODEL_DIR      = os.path.join(BASE_DIR, "02_model")
RESULTS_DIR    = os.path.join(MODEL_DIR, "results")
RECON_DIR      = os.path.join(BASE_DIR, "03_reconstruction")

# ─── 区域过滤 ─────────────────────────────────────────────────────────────────
TARGET_COUNTRIES = ['US', 'CA']
TRAIN_YEAR_MIN   = 1980
TRAIN_YEAR_MAX   = 2024
TARGET           = 'ANNUAL_BALANCE'   # 单位: mm w.e.

# ─── ERA5 完整变量映射（25 个变量，与 RF 管道保持一致） ────────────────────────
# 格式: 'NetCDF变量名': ('输出列基础名', '聚合方式')
# 聚合方式仅供年度聚合使用；月度提取时直接保留原始月度值
VAR_MAPPING = {
    # 状态量（取月均值）
    't2m':    ('temperature_2m',                     'mean'),
    'd2m':    ('dewpoint_temperature_2m',             'mean'),
    'skt':    ('skin_temperature',                   'mean'),
    'sp':     ('surface_pressure',                   'mean'),
    'fal':    ('forecast_albedo',                    'mean'),
    'asn':    ('snow_albedo',                        'mean'),
    'rsn':    ('snow_density',                       'mean'),
    'sd':     ('snow_depth',                         'mean'),
    'lai_hv': ('leaf_area_index_high_vegetation',    'mean'),
    'lai_lv': ('leaf_area_index_low_vegetation',     'mean'),
    # 通量（月内累积值，moda 文件已为月均日通量，下游按需 ×天数）
    'tp':     ('total_precipitation',                'sum'),
    'sf':     ('snowfall',                           'sum'),
    'smlt':   ('snowmelt',                           'sum'),
    'ssrd':   ('surface_solar_radiation_downwards',  'sum'),
    'strd':   ('surface_thermal_radiation_downwards','sum'),
    'ssr':    ('surface_net_solar_radiation',         'sum'),
    'str':    ('surface_net_thermal_radiation',       'sum'),
    'slhf':   ('surface_latent_heat_flux',           'sum'),
    'sshf':   ('surface_sensible_heat_flux',         'sum'),
    'e':      ('total_evaporation',                  'sum'),
    'pev':    ('potential_evaporation',              'sum'),
    'ro':     ('runoff',                             'sum'),
    'sro':    ('surface_runoff',                     'sum'),
    'ssro':   ('sub_surface_runoff',                 'sum'),
    'es':     ('snow_evaporation',                   'sum'),
}

# ─── LSTM 时序分支：月度气象变量（15 个）─────────────────────────────────────
# 按物理过程分组，覆盖温度、降水、辐射、能量、积雪五个方面
# 去除生态指标 (lai_hv/lai_lv)、地表气压 (sp) 等对冰川影响较小的变量
MONTHLY_CLIMATE_VARS = [
    # 温度 (3)
    't2m',    # 2m 气温（月均）
    'skt',    # 地表皮肤温度（月均）
    'd2m',    # 2m 露点温度（月均，代表大气湿度）
    # 积雪状态 (2)
    'sd',     # 积雪深度（月均）
    'asn',    # 雪面反照率（月均）
    # 降水与融雪 (3)
    'tp',     # 总降水（月累积）
    'sf',     # 降雪量（月累积）
    'smlt',   # 融雪量（月累积）
    # 辐射 (4)
    'ssrd',   # 向下短波辐射（月累积，控制消融能量输入）
    'strd',   # 向下长波辐射（月累积）
    'ssr',    # 净短波辐射（月累积）
    'str',    # 净长波辐射（月累积）
    # 湍流热通量 (2)
    'slhf',   # 潜热通量（月累积，蒸发冷却）
    'sshf',   # 感热通量（月累积，大气加热）
    # 水文 (1)
    'ro',     # 径流（月累积，代表冰雪融水输出）
]
# 月度特征数（F_dynamic）
N_CLIMATE_FEATURES = len(MONTHLY_CLIMATE_VARS)   # 15

# ─── LSTM 空间分支：静态地形特征（5 个）─────────────────────────────────────
# 这些特征描述冰川的几何和位置属性，不随时间变化
STATIC_FEATURES = [
    'LOWER_BOUND',   # 冰川最低海拔 (m)
    'UPPER_BOUND',   # 冰川最高海拔 (m)
    'AREA',          # 冰川面积 (km²)
    'LATITUDE',      # 纬度 (°N)
    'LONGITUDE',     # 经度 (°W，负值)
]
N_STATIC_FEATURES = len(STATIC_FEATURES)   # 5

# ─── LSTM 模型超参数 ──────────────────────────────────────────────────────────
LSTM_PARAMS = {
    'n_months':            12,    # 时序长度（日历年 1-12 月）
    'lstm_hidden_dim':    128,    # LSTM 隐藏维度
    'lstm_num_layers':      2,    # LSTM 层数
    'lstm_bidirectional': False,  # 双向 LSTM（True 则 hidden×2）
    'static_mlp_hidden':   64,   # 静态 MLP 隐藏维度
    'static_mlp_layers':    2,   # 静态 MLP 深度
    'dropout':            0.30,  # Dropout 比率
    'batch_size':           32,
    'epochs':              200,   # 增大上限，早停负责实际截止
    'lr':               0.001,
    'random_state':        42,
    # 早停策略
    'min_epochs':          60,    # 前 60 epoch 不计 patience，先让模型热身
    'early_stop_patience': 40,    # 之后连续 40 epoch 无改善才停止
}

# ─── 重建配置 ─────────────────────────────────────────────────────────────────
RECON_YEAR_MIN   = 1950
RECON_YEAR_MAX   = 2024
# RF 管道已提取的 RGI02 冰川元数据（含年度 ERA5，用于获取冰川静态信息）
RGI02_ERA5_CSV   = r"H:\Code\SMB\SMB_Res_RF_ByClaude\01_preprocessing\data\rgi02_glaciers_era5.csv"

# ─── 自动创建输出目录 ─────────────────────────────────────────────────────────
for _d in [PREPROCESS_DIR, RESULTS_DIR, RECON_DIR,
           os.path.join(RECON_DIR, "results"),
           os.path.join(RECON_DIR, "figures")]:
    os.makedirs(_d, exist_ok=True)
