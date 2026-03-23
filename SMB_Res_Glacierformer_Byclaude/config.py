"""
SMB_Res_Glacierformer_ByClaude — 统一路径与模型配置

GlacioFormer 架构（Transformer + 频域 + 物理约束）:
  时序分支 : WTConv1D → FFT-Transformer × 2 → EMA → GlobalPool → (B, d_model)
  静态分支 : MLP → SimAM1D                                      → (B, d_model)
  融合层   : CrossModalFreqFusion → Prediction Head             → (B,) SMB

预处理数据直接复用 SMB_Res_LSTM_Byclaude 的输出，无需重新提取 ERA5。
修改 BASE_DIR 和外部数据路径后，所有脚本均可正常运行。
"""
import os

# ─── 外部数据（大文件，本地保留，不上传 GitHub） ─────────────────────────────
ERA5_NC_PATH     = r"H:\Code\SMB\test\data\ERA5-LAND\data_stream-moda.nc"
WGMS_MATCHED_CSV = r"H:\Code\SMB\test\data\study_test\data_glacier_era5_fixed.csv"

# ─── 项目根目录 ───────────────────────────────────────────────────────────────
BASE_DIR = r"H:\Code\SMB\SMB_Res_Glacierformer_Byclaude"

# ─── 预处理数据目录 ────────────────────────────────────────────────────────────
LSTM_PREPROCESS_DIR = r"H:\Code\SMB\SMB_Res_LSTM_Byclaude\01_preprocessing\data"
# GlacioFormer 专用数据集（含 RGI v7.0 地形特征，静态特征 5→10）
PREPROCESS_DIR = os.path.join(BASE_DIR, "01_preprocessing", "data")

# ─── 项目输出目录 ─────────────────────────────────────────────────────────────
MODEL_DIR   = os.path.join(BASE_DIR, "02_model")
RESULTS_DIR = os.path.join(MODEL_DIR, "results")
RECON_DIR   = os.path.join(BASE_DIR, "03_reconstruction")

# ─── 区域过滤（与 LSTM 保持一致）─────────────────────────────────────────────
TARGET_COUNTRIES = ['US', 'CA']
TRAIN_YEAR_MIN   = 1980
TRAIN_YEAR_MAX   = 2024
TARGET           = 'ANNUAL_BALANCE'   # 单位: mm w.e.

# ─── 特征定义（与 LSTM 保持一致）─────────────────────────────────────────────
VAR_MAPPING = {
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

MONTHLY_CLIMATE_VARS = [
    't2m', 'skt', 'd2m',       # 温度 (3)
    'sd', 'asn',               # 积雪状态 (2)
    'tp', 'sf', 'smlt',        # 降水与融雪 (3)
    'ssrd', 'strd', 'ssr', 'str',  # 辐射 (4)
    'slhf', 'sshf',            # 湍流热通量 (2)
    'ro',                      # 水文 (1)
]

STATIC_FEATURES = [
    # 原始 WGMS 地形特征 (5)
    'LOWER_BOUND',   # 冰川最低海拔 (m)
    'UPPER_BOUND',   # 冰川最高海拔 (m)
    'AREA',          # 冰川面积 (km²)
    'LATITUDE',      # 纬度 (°N)
    'LONGITUDE',     # 经度 (°W，负值)
    # RGI v7.0 地形特征 (5)
    'slope_deg',     # Copernicus DEM 30m 计算的冰川平均坡度 (°)
    'aspect_sin',    # 坡向正弦（东向分量）
    'aspect_cos',    # 坡向余弦（北向分量）
    'zmean_m',       # DEM 均值高程 (m)
    'zmed_m',        # DEM 中位高程 (m)，常用作 ELA 代理变量
]

N_CLIMATE_FEATURES = len(MONTHLY_CLIMATE_VARS)   # 15
N_STATIC_FEATURES  = len(STATIC_FEATURES)         # 10

# ─── GlacioFormer 超参数 ──────────────────────────────────────────────────────
GLACIOFORMER_PARAMS = {
    # 模型结构（恢复 d_model=64，只做保守调整）
    'd_model':          64,    # Transformer 隐藏维度
    'n_heads':           4,    # 多头注意力头数（d_model 需整除 n_heads）
    'n_encoder_layers':  2,    # FFT-Transformer 堆叠层数
    'ff_dim':          256,    # FFN 前馈维度（4×d_model）
    'ema_scales':  [1, 3, 6],  # EMA 多尺度：月、季、半年
    'wt_levels':         2,    # 小波分解层数
    'dropout':        0.15,    # 0.10→0.15，轻度增强正则
    # 训练
    'batch_size':       32,    # 恢复 32（690 训练样本，128 导致每 epoch 仅 5 次更新，严重欠训练）
    'epochs':          300,
    'lr':            3e-4,
    'weight_decay':  3e-4,
    'random_state':     42,
    # 早停策略
    'min_epochs':       60,    # 热身阶段，前 60 epoch 不计 patience
    'early_stop_patience': 50, # 热身后连续 50 epoch 无改善才停止
    # 物理约束损失（关闭，排查偏差来源）
    'physics_alpha':  0.00,    # 关闭物理约束（第一次运行 0.10 引入了 +84mm 偏差）
    'temp_var_name': 'temperature_2m',
}

# ─── 重建配置 ─────────────────────────────────────────────────────────────────
RECON_YEAR_MIN = 1950
RECON_YEAR_MAX = 2024
RGI02_ERA5_CSV = r"H:\Code\SMB\SMB_Res_RF_ByClaude\01_preprocessing\data\rgi02_glaciers_era5.csv"

# ─── 自动创建输出目录 ─────────────────────────────────────────────────────────
for _d in [RESULTS_DIR, RECON_DIR,
           os.path.join(RECON_DIR, "results"),
           os.path.join(RECON_DIR, "figures"),
           os.path.join(RECON_DIR, "data")]:
    os.makedirs(_d, exist_ok=True)
