"""Configuration for the RGI02 PhysGlacierFormer experiments."""
from __future__ import annotations

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULT_DIR = os.path.join(BASE_DIR, "results")
FIG_DIR = os.path.join(BASE_DIR, "figures")

for _path in (DATA_DIR, RESULT_DIR, FIG_DIR):
    os.makedirs(_path, exist_ok=True)

# External source data. These files are shared with the stable baseline project.
WGMS_DIR = r"H:\Code\SMB\WGMS\FoG_DataBase\DOI-WGMS-FoG-2025-02b\data"
GLACIER_CSV = os.path.join(WGMS_DIR, "glacier.csv")
MASSBAL_CSV = os.path.join(WGMS_DIR, "mass_balance.csv")

RGI02_SHP = (
    r"H:\Code\SMB\RGI\RGI2000-v7.0-G-02_western_canada_usa"
    r"\RGI2000-v7.0-G-02_western_canada_usa.shp"
)
RGI02_RGI60_SHP = (
    r"H:\Code\SMB\RGI\nsidc0770_02.rgi60.WesternCanadaUS"
    r"\02_rgi60_WesternCanadaUS.shp"
)
RGI02_LINKS_CSV = (
    r"H:\Code\SMB\RGI\RGI2000-v7.0-G-02_western_canada_usa"
    r"\RGI2000-v7.0-G-02_western_canada_usa-rgi6_links.csv"
)
RGI02_HYPSOMETRY_CSV = (
    r"H:\Code\SMB\RGI\RGI2000-v7.0-G-02_western_canada_usa"
    r"\RGI2000-v7.0-G-02_western_canada_usa-hypsometry.csv"
)
RGI02_ATTRIBUTES_CSV = (
    r"H:\Code\SMB\RGI\RGI2000-v7.0-G-02_western_canada_usa"
    r"\RGI2000-v7.0-G-02_western_canada_usa-attributes.csv"
)
ERA5_RGI02_NC = r"H:\Code\SMB\ERA5-LAND\data_stream-moda.nc"
ERA5_ALASKA_NC = r"H:\Code\SMB\ERA5-LAND\Rgi01\Alaska.nc"
HUGONNET_RGI02_RATES = (
    r"H:\Code\SMB\Hugonnet_results\time_series_02\dh_02_rgi60_pergla_rates.csv"
)
ERA5_RGI02_FULL_CSV = r"H:\Code\SMB\SMB_Res_ByClaudeV2\data\era5_monthly_rgi02.csv"
RGI02_TARGET_CSV = r"H:\Code\SMB\SMB_Res_ByClaudeV2\data\rgi02_target_glaciers.csv"

# Generated intermediate datasets for this clean experiment folder.
TRAINING_GLACIERS_CSV = os.path.join(DATA_DIR, "training_glaciers.csv")
MASSBAL_RGI02_CSV = os.path.join(DATA_DIR, "massbal_rgi02.csv")
TERRAIN_CSV = os.path.join(DATA_DIR, "training_glaciers_terrain.csv")
ERA5_MONTHLY_CSV = os.path.join(DATA_DIR, "era5_monthly_training.csv")
TABULAR_CSV = os.path.join(DATA_DIR, "tabular_dataset.csv")
SEQUENCES_NPZ = os.path.join(DATA_DIR, "sequences_training.npz")

# Optional future feature tables.
HYPSOMETRY_CSV = os.path.join(DATA_DIR, "hypsometry_features.csv")
REMOTE_SENSING_CSV = os.path.join(DATA_DIR, "remote_sensing_features.csv")

# Quality-controlled baseline dataset.
QC_DATA_DIR = os.path.join(DATA_DIR, "qc")
QC_MATCH_DIST_MAX_DEG = 0.09
SEQUENCES_QC_NPZ = os.path.join(QC_DATA_DIR, "sequences_training_qc.npz")

# Hypsometry-enhanced dataset.
HYPSOMETRY_DATA_DIR = os.path.join(DATA_DIR, "hypsometry")
SEQUENCES_HYPSOMETRY_QC_NPZ = os.path.join(
    HYPSOMETRY_DATA_DIR,
    "sequences_hypsometry_qc.npz",
)
N_HYPSOMETRY_FEATURES = 3

# Corrected, publication-oriented pipeline. These outputs are isolated from the
# historical baseline so that preprocessing changes can be audited explicitly.
PHYS_V2_DATA_DIR = os.path.join(DATA_DIR, "phys_v2")
PHYS_V2_TERRAIN_CSV = os.path.join(PHYS_V2_DATA_DIR, "training_glaciers_terrain_v2.csv")
PHYS_V2_ERA5_CSV = os.path.join(PHYS_V2_DATA_DIR, "era5_monthly_training_v2.csv")
PHYS_V2_SEQUENCES_NPZ = os.path.join(PHYS_V2_DATA_DIR, "sequences_phys_v2.npz")
PHYS_V2_DATA_SUMMARY_CSV = os.path.join(PHYS_V2_DATA_DIR, "sequences_phys_v2_summary.csv")
PHYS_V2_RESULT_DIR = os.path.join(RESULT_DIR, "phys_v2")
PHYS_V2_FINAL_DIR = os.path.join(PHYS_V2_RESULT_DIR, "final_xgboost")
PHYS_V2_FINAL_MODEL = os.path.join(PHYS_V2_FINAL_DIR, "xgboost_compact_regularized.json")
PHYS_V2_FINAL_PREPROCESSOR = os.path.join(PHYS_V2_FINAL_DIR, "xgboost_compact_preprocessor.npz")
PHYS_V2_FINAL_IMPORTANCE = os.path.join(PHYS_V2_FINAL_DIR, "xgboost_compact_feature_importance.csv")
PHYS_V2_RECONSTRUCTION_CSV = os.path.join(
    RESULT_DIR, "reconstruction", "RGI02_SMB_xgboost_v2_raw_all_glaciers.csv"
)
PHYS_V2_RECONSTRUCTION_CALIBRATED_CSV = os.path.join(
    RESULT_DIR, "reconstruction", "RGI02_SMB_xgboost_v2_hugonnet_conservative.csv"
)
PHYS_V2_RECONSTRUCTION_REGIONAL_CSV = os.path.join(
    RESULT_DIR, "reconstruction", "xgboost_v2_regional_timeseries.csv"
)
PHYS_V2_RECONSTRUCTION_CALIBRATION_QC = os.path.join(
    RESULT_DIR, "reconstruction", "xgboost_v2_hugonnet_calibration_qc.csv"
)
PHYS_V2_RECONSTRUCTION_QC_SUMMARY = os.path.join(
    RESULT_DIR, "reconstruction", "xgboost_v2_reconstruction_qc_summary.csv"
)
PHYS_V2_RECONSTRUCTION_OOD_SUMMARY = os.path.join(
    RESULT_DIR, "reconstruction", "xgboost_v2_reconstruction_ood_summary.csv"
)
PHYS_V2_MALLES_COMPARISON = os.path.join(
    RESULT_DIR, "reconstruction", "xgboost_v2_malles_comparison.csv"
)
PHYS_V2_ZEMP_COMPARISON = os.path.join(
    RESULT_DIR, "reconstruction", "xgboost_v2_zemp_comparison.csv"
)
PHYS_V2_INTERVAL_CALIBRATION = os.path.join(
    PHYS_V2_RESULT_DIR, "publication_evaluation", "prediction_interval_calibration.csv"
)
MALLES_REGION_NC = (
    r"H:\Code\SMB\Malles&Marzeion\suppl_reconstruction_data_region.nc"
)
ZEMP_RGI02_CSV = (
    r"H:\Code\SMB\Zemp_results\Zemp_etal_DataTables2a-t_results_regions_global"
    r"\Zemp_etal_results_region_2_WNA.csv"
)

# Hugonnet et al. (2021) weak-label dataset.
HUGONNET_DATA_DIR = os.path.join(DATA_DIR, "hugonnet")
HUGONNET_WEAK_LABELS_CSV = os.path.join(
    HUGONNET_DATA_DIR,
    "hugonnet_rgi02_weak_labels_2000_2020.csv",
)
HUGONNET_MATCH_QC_CSV = os.path.join(
    HUGONNET_DATA_DIR,
    "hugonnet_rgi02_match_qc_2000_2020.csv",
)
HUGONNET_QC_SUMMARY_CSV = os.path.join(
    HUGONNET_DATA_DIR,
    "hugonnet_rgi02_qc_summary.csv",
)
HUGONNET_WEAK_SEQUENCES_NPZ = os.path.join(
    HUGONNET_DATA_DIR,
    "sequences_hugonnet_weak_2000_2019.npz",
)
HUGONNET_WEAK_SEQUENCE_SUMMARY_CSV = os.path.join(
    HUGONNET_DATA_DIR,
    "sequences_hugonnet_weak_2000_2019_summary.csv",
)
HUGONNET_MULTIPERIOD_LABELS_CSV = os.path.join(
    HUGONNET_DATA_DIR,
    "hugonnet_rgi02_multiperiod_labels.csv",
)
HUGONNET_MULTIPERIOD_SEQUENCES_NPZ = os.path.join(
    HUGONNET_DATA_DIR,
    "sequences_hugonnet_multiperiod.npz",
)
HUGONNET_MULTIPERIOD_SEQUENCE_SUMMARY_CSV = os.path.join(
    HUGONNET_DATA_DIR,
    "sequences_hugonnet_multiperiod_summary.csv",
)
HUGONNET_WEAK_RESULT_DIR = os.path.join(RESULT_DIR, "hugonnet_weak")
FINAL_MODEL_DIR = os.path.join(RESULT_DIR, "final_models")
RECONSTRUCTION_DIR = os.path.join(RESULT_DIR, "reconstruction")
HYPSOMETRY_FINAL_WEIGHTS = os.path.join(
    FINAL_MODEL_DIR,
    "glacierformer_hypsometry_qc_final.pt",
)
HYPSOMETRY_RECON_RAW_CSV = os.path.join(
    RECONSTRUCTION_DIR,
    "RGI02_SMB_hypsometry_raw.csv",
)
HYPSOMETRY_RECON_CALIBRATED_CSV = os.path.join(
    RECONSTRUCTION_DIR,
    "RGI02_SMB_hypsometry_hugonnet_calibrated.csv",
)
HYPSOMETRY_RECON_QC_CSV = os.path.join(
    RECONSTRUCTION_DIR,
    "RGI02_SMB_hypsometry_reconstruction_qc.csv",
)
HUGONNET_CALIBRATION_QC_CSV = os.path.join(
    RECONSTRUCTION_DIR,
    "hugonnet_calibration_qc.csv",
)
HUGONNET_CALIBRATION_REGIONAL_CSV = os.path.join(
    RECONSTRUCTION_DIR,
    "hugonnet_calibration_regional_timeseries.csv",
)
HUGONNET_CALIBRATION_OFFSET_CLIP = 1.0
HUGONNET_CALIBRATION_OFFSET_SHRINK = 0.5
HUGONNET_CONSERVATIVE_DIR = RECONSTRUCTION_DIR
HYPSOMETRY_RECON_CONSERVATIVE_CSV = os.path.join(
    HUGONNET_CONSERVATIVE_DIR,
    "RGI02_SMB_hypsometry_hugonnet_conservative.csv",
)
HUGONNET_CALIBRATION_CONSERVATIVE_QC_CSV = os.path.join(
    HUGONNET_CONSERVATIVE_DIR,
    "hugonnet_calibration_qc.csv",
)
HUGONNET_CALIBRATION_CONSERVATIVE_REGIONAL_CSV = os.path.join(
    HUGONNET_CONSERVATIVE_DIR,
    "hugonnet_calibration_regional_timeseries.csv",
)
HUGONNET_WEAK_LABEL_PERIOD = "2000-01-01_2020-01-01"
HUGONNET_WEAK_YEAR_MIN = 2000
HUGONNET_WEAK_YEAR_MAX = 2019
HUGONNET_MULTIPERIOD_PERIODS = [
    "2000-01-01_2010-01-01",
    "2000-01-01_2020-01-01",
    "2004-01-01_2008-01-01",
    "2005-01-01_2010-01-01",
    "2008-01-01_2012-01-01",
    "2010-01-01_2015-01-01",
    "2010-01-01_2020-01-01",
    "2012-01-01_2016-01-01",
    "2015-01-01_2020-01-01",
    "2016-01-01_2020-01-01",
]
HUGONNET_MULTIPERIOD_YEAR_MIN = 2000
HUGONNET_MULTIPERIOD_YEAR_MAX = 2019
HUGONNET_MULTIPERIOD_MAX_ERR_DMDTDA = 1.5
HUGONNET_MIN_PERC_AREA_MEAS = 0.8
HUGONNET_MIN_PERC_AREA_RES = 0.8
HUGONNET_MIN_VALID_OBS_PY = 5.0
HUGONNET_MIN_RGI7_OVERLAP_FRACTION = 0.5
HUGONNET_MIN_RGI60_OVERLAP_FRACTION = 0.5
HUGONNET_MAX_MAP_DIST_DEG = 0.05
HUGONNET_MIN_AREA_RATIO_RGI7_TO_RGI60 = 0.5
HUGONNET_MAX_AREA_RATIO_RGI7_TO_RGI60 = 2.0

TRAIN_YEAR_MIN = 1950
TRAIN_YEAR_MAX = 2023
RECON_YEAR_MIN = 1950
RECON_YEAR_MAX = 2024

MONTHLY_CLIMATE_VARS = [
    "t2m", "skt", "d2m",
    "sd", "asn",
    "tp", "sf", "smlt",
    "ssrd", "strd", "ssr", "str",
    "slhf", "sshf",
    "ro",
]
N_DYNAMIC = len(MONTHLY_CLIMATE_VARS)

PHYS_V2_DYNAMIC_FEATURES = [
    "t2m", "d2m", "sd", "asn",
    "tp", "sf", "smlt",
    "ssrd", "strd", "str", "slhf", "sshf",
    "relative_humidity",
    "t2m_anomaly", "tp_anomaly", "sf_anomaly",
    "ssrd_anomaly", "asn_anomaly",
    "t2m_lapse",
]

PHYS_V2_STATIC_FEATURES = [
    "slope_deg", "aspect_sin", "aspect_cos",
    "zmin_m", "zmax_m", "zmean_m", "zmed_m",
    "log1p_area_km2", "log1p_lmax_m", "cenlat", "cenlon",
    "clim_annual_t2m", "clim_winter_tp", "clim_summer_t2m",
    "clim_summer_ssrd", "clim_t2m_amplitude",
    "era5_pressure_elevation_m", "elevation_difference_m",
    "clim_annual_t2m_lapse", "clim_summer_t2m_lapse",
]

STATIC_FEATURES = [
    "slope_deg", "aspect_sin", "aspect_cos",
    "zmin_m", "zmax_m", "zmean_m", "zmed_m",
    "area_km2", "lmax_m", "cenlat",
]

SEASONAL_EXTRA_FEATURES = [
    "cal_summer_t2m_mean",
    "cal_winter_tp_sum",
    "hyd_ablat_t2m_mean",
    "hyd_accum_sf_sum",
    "hyd_ablat_smlt_sum",
    "ann_t2m_mean",
    "ann_tp_sum",
    "cal_summer_ssrd_sum",
]

N_STATIC = len(STATIC_FEATURES) + len(SEASONAL_EXTRA_FEATURES)

CAL_SUMMER_MONTHS = [6, 7, 8]
CAL_WINTER_MONTHS = [12, 1, 2]
HYD_ACCUM_MONTHS = [10, 11, 12, 1, 2, 3, 4]
HYD_ABLAT_MONTHS = [5, 6, 7, 8, 9]

GLACIERFORMER_BASE_PARAMS = dict(
    n_dynamic_features=N_DYNAMIC,
    n_static_features=N_STATIC,
    d_model=64,
    n_heads=4,
    n_encoder_layers=2,
    ff_dim=256,
    dropout=0.15,
    batch_size=32,
    epochs=300,
    lr=1e-3,
    early_stop_patience=30,
    min_epochs=50,
    weight_decay=1e-4,
    val_fraction=0.15,
)
HUGONNET_PRETRAIN_PARAMS = {
    **GLACIERFORMER_BASE_PARAMS,
    "lr": 2e-4,
    "weight_decay": 1e-4,
    "batch_size": 16,
}

assert GLACIERFORMER_BASE_PARAMS["d_model"] % GLACIERFORMER_BASE_PARAMS["n_heads"] == 0

GLACIERFORMER_HYPSOMETRY_PARAMS = dict(
    **GLACIERFORMER_BASE_PARAMS,
    n_hypsometry_features=N_HYPSOMETRY_FEATURES,
)

PHYS_GLACIERFORMER_V2_PARAMS = dict(
    n_dynamic_features=len(PHYS_V2_DYNAMIC_FEATURES),
    n_static_features=len(PHYS_V2_STATIC_FEATURES),
    n_hypsometry_features=N_HYPSOMETRY_FEATURES,
    d_model=32,
    n_heads=4,
    n_encoder_layers=1,
    ff_dim=96,
    dropout=0.20,
    batch_size=32,
    epochs=220,
    lr=5e-4,
    early_stop_patience=25,
    min_epochs=35,
    weight_decay=5e-4,
    val_fraction=0.15,
    huber_beta=0.5,
    seasonal_loss_weight=0.30,
)

XGBOOST_PARAMS = dict(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
)

MIN_AREA_KM2 = 0.5
