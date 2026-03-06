# Glacier Surface Mass Balance Reconstruction (RGI Region 02)

## Overview

This project uses machine learning to predict annual glacier mass balance in Western North America (RGI Region 02) using ERA5-Land climate reanalysis data and WGMS glacier observations.

## Key Results

### Model Performance Comparison (Leave-One-Year-Out CV)

| Model | R² | RMSE | MAE | Improvement |
|-------|-----|------|-----|-------------|
| RandomForest (baseline) | 0.510 | 0.720 | 0.569 | - |
| **XGBoost** | **0.592** | **0.657** | **0.515** | **+16%** ✓ |
| LightGBM | 0.591 | 0.658 | 0.523 | +15% |

**XGBoost is the best model** with significant improvement over RF baseline.

### Reconstruction Output

- **159 glaciers** × **76 years** = 12,084 predictions
- Mean predicted SMB: **-1.070 m w.e.** (water equivalent)
- Data composition:
  - 6.6% observed measurements
  - 11.6% gap-filled predictions (for observed glaciers)
  - 81.8% predicted-only (for unobserved glaciers)

## Project Structure

```
SMB/
├── SMB_Res_RF_ByClaude/                    # RandomForest baseline
│   ├── 01_preprocessing/
│   ├── 02_model/
│   └── 03_reconstruction/
│
├── SMB_Res_XGBOOST_ByClaude/               # XGBoost + LightGBM (BEST)
│   ├── config.py                           # Unified configuration
│   ├── 02_model/
│   │   ├── train_xgb.py                    # XGBoost LOYO + LOGO
│   │   ├── train_lgb.py                    # LightGBM LOYO + LOGO
│   │   ├── compare_models.py               # RF vs XGB vs LGB comparison
│   │   └── results/
│   ├── 03_reconstruction/
│   │   ├── step01_reconstruct.py           # Final SMB reconstruction
│   │   ├── step02_hybrid_dataset.py        # Observed + gap-filled
│   │   ├── step03_visualize.py             # 6-panel figure suite
│   │   ├── results/
│   │   └── figures/
│
├── test/
│   ├── rf.py                               # Original RF training
│   ├── data_processing.py
│   └── data/
│
└── 2025-02b/                               # WGMS FoG database
    └── data/
```

## Quick Start

### 1. Install Dependencies

```bash
conda create -n smb python=3.10
conda activate smb
pip install pandas numpy scikit-learn matplotlib scipy xarray xgboost lightgbm
```

### 2. Run XGBoost Pipeline

```bash
cd SMB_Res_XGBOOST_ByClaude

# Train both models (parallel)
python 02_model/train_xgb.py
python 02_model/train_lgb.py

# Compare models and select best
python 02_model/compare_models.py

# Reconstruct full dataset
python 03_reconstruction/step01_reconstruct.py
python 03_reconstruction/step02_hybrid_dataset.py
python 03_reconstruction/step03_visualize.py
```

## Features (31 variables)

**Geometric & Location:**
- LOWER_BOUND, UPPER_BOUND, AREA, LATITUDE, LONGITUDE

**Temperature & Surface State:**
- temperature_2m_year/summer
- skin_temperature_year/summer
- dewpoint_temperature_2m_summer

**Precipitation & Snow:**
- total_precipitation_sum_year/summer
- snowfall_sum_year/summer
- snow_depth_year/summer
- snow_density_summer, snow_albedo_summer

**Energy Balance:**
- surface_net_solar/thermal_radiation_sum_summer
- surface_sensible/latent_heat_flux_sum_summer

**Water Balance:**
- total_evaporation_sum_year/summer
- snow_evaporation_sum_summer
- runoff_sum_summer
- snowmelt_sum_year/summer

**Time:**
- YEAR

## Key Findings

1. **XGBoost surpasses RandomForest** by 16% in R² score
2. **Snowmelt and runoff** are top climate predictors after elevation
3. **Spatial generalization (LOGO)** is challenging (R²=0.36), indicating glacier-specific variations
4. **Hybrid dataset** successfully extends coverage from 31 to 159 glaciers

## Data Sources

- **WGMS FoG Database**: Fluctuations of Glaciers (1980-2024)
- **ERA5-Land Reanalysis**: 70+ climate variables (temperature, precipitation, radiation, etc.)
- **RGI v7.0**: Glacier boundaries and metadata

## License

Research project - Contact for usage permissions

## Contact

Questions or collaboration: imxiaoxinvo1@github.com
