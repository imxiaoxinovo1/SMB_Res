# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a glacier surface mass balance (SMB) modeling project that uses machine learning to predict annual glacier mass balance using meteorological and glaciological features. The project works with WGMS (World Glacier Monitoring Service) glacier data and ERA5-Land/MERRA-2 climate reanalysis data for western Canada and USA (RGI Region 02).

## Core Dependencies

No `requirements.txt` exists, but these packages are required:
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `scikit-learn` - Machine learning models (RandomForest, Lasso, LinearRegression)
- `matplotlib` - Visualization
- `scipy` - Statistical functions
- `xarray` - NetCDF climate data processing

Install in a virtual environment:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install pandas numpy scikit-learn matplotlib scipy xarray
```

## Quick Start

**Recommended entry point**: [test/rf.py](test/rf.py)
- Uses relative paths within the repository
- Trains RandomForest model with leave-one-year-out cross-validation (1980-2020)
- Outputs to `test/result/` directory
- Run: `python test/rf.py`

**Root script**: [rf.py](rf.py)
- Contains hardcoded absolute paths (`H:\workspace4_ear5\...`) - modify with caution
- Use `test/rf.py` for experimentation instead

## Key Architecture & Workflow

### 1. Data Processing Pipeline

**Input data sources:**
- `test/study_data_wna.csv` - Pre-processed glacier + climate feature dataset
- `2025-02b/` - WGMS FoG (Fluctuations of Glaciers) database
- `RGI/` - Randolph Glacier Inventory v7.0 shapefiles and metadata
- ERA5-Land NetCDF files (external, referenced in [test/data_processing.py](test/data_processing.py))

**Data processing scripts:**
- [test/data_processing.py](test/data_processing.py) - Merges ERA5-Land climate variables with WGMS glacier data
  - Extracts ~70 climate variables (temperature, precipitation, snow, radiation, etc.)
  - Spatial extraction: nearest-neighbor matching to glacier coordinates
  - Temporal aggregation: monthly → annual (sum for fluxes, mean for state variables)
  - Handles longitude conversion (0-360 vs -180-180 formats)

- [test/data_read.py](test/data_read.py) - Optimized vectorized version using xarray
  - Processes all glacier points in one operation (much faster)
  - Uses Dask chunking for large NetCDF files

### 2. Model Training Pattern

All model scripts ([test/rf.py](test/rf.py), [test/MLR.py](test/MLR.py), [test/Lasso.py](test/Lasso.py), [test/BG.py](test/BG.py)) follow this structure:

```python
# 1. Feature definition (20 features)
features_columns = [
    "LOWER_BOUND", "UPPER_BOUND", "snowmelt_sum_year", "AREA",
    "snow_cover_summer", "skin_reservoir_content_summer",
    "evaporation_from_open_water_surfaces_excluding_oceans_sum_summer",
    "runoff_sum_year", "LONGITUDE", "lake_mix_layer_temperature_summer",
    "snowfall_sum_year", "evaporation_from_the_top_of_canopy_sum_summer",
    "surface_runoff_sum_year",
    "evaporation_from_open_water_surfaces_excluding_oceans_sum_year",
    "sub_surface_runoff_sum_year", "PBLH", "RHOA", "PRECSNO",
    "GHTSKIN", "YEAR"
]
target_column = 'ANNUAL_BALANCE'

# 2. Leave-one-year-out cross-validation
for test_year in range(1980, 2021):
    train_data = df[df['YEAR'] != test_year]
    test_data = df[(df['YEAR'] == test_year) & (df['TAG'] == 9999)]

    # 3. Train and predict (with unit conversion to meters)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test) / 1000  # mm w.e. → m w.e.
    y_test = y_test / 1000

    # 4. Compute metrics: R², R, RMSE, MAE, Bias
```

**Critical filtering**: `df['TAG'] == 9999` selects only specific test observations.

### 3. Output Structure

All results are written to `test/result/`:
- `*_all_data_result_time_*.csv` - Per-year evaluation metrics
- `*_pred_result_time_*.csv` - Individual predictions (year, y_test, y_pred)

Model-specific suffixes:
- `_wna.csv` - RandomForest
- `_wna_mlr.csv` - Multiple Linear Regression
- `_wna_lasso.csv` - Lasso Regression

### 4. Model-Specific Notes

**RandomForest** ([test/rf.py](test/rf.py)):
- Hyperparameters: `n_estimators=200, max_depth=100`
- No feature scaling required
- Primary model for this project

**Multiple Linear Regression** ([test/MLR.py](test/MLR.py)):
- Uses `StandardScaler` for feature normalization
- Baseline linear model

**Lasso** ([test/Lasso.py](test/Lasso.py)):
- L1 regularization: `alpha=0.1`
- Generates coefficient plot showing feature importance
- Requires feature scaling

## Data Schema Reference

The `2025-02b/datapackage.json` defines the WGMS FoG database structure:
- Primary key: `glacier.id`
- Foreign keys connect glacier, measurements, and metadata tables
- Contains field constraints, data types, and missing value conventions
- **Important**: Never modify CSV files in `2025-02b/data/` without verifying schema compliance

## Development Conventions

### File Path Strategy
- **Test scripts**: Use `H:/Code/SMB/test/` relative paths
- **Root scripts**: Contain legacy absolute paths - avoid editing
- When creating new scripts, follow the `test/` pattern for portability

### Standard Workflow
1. Read data: `df = pd.read_csv('H:/Code/SMB/test/study_data_wna.csv')`
2. Process with model (see pattern above)
3. Save to `test/result/` with descriptive filename
4. Print overall metrics after loop completes

### Code Style
- Chinese comments are acceptable (mixed with English)
- Variable names are English
- Keep scripts self-contained (no modular package structure)

## Visualization Scripts

- [test/Figure3.py](test/Figure3.py), [test/Figure3_RF.py](test/Figure3_RF.py) - Generate prediction vs observation scatter plots
- [test/Figure3_rf_contrast.py](test/Figure3_rf_contrast.py) - Model comparison visualizations
- Outputs to `SMB_figure/`

## Common Modifications

### To add a new model:
1. Copy `test/rf.py` as template
2. Change model initialization (line 71)
3. Update output filenames to avoid overwrites (lines 107-108)
4. Keep the same feature columns and cross-validation loop

### To change features:
- Modify `features_columns` list (must match columns in CSV)
- If adding ERA5 variables, update [test/data_processing.py](test/data_processing.py):
  - Add to `var_map` dictionary with output column name
  - Specify aggregation rule in `agg_rules` (sum vs mean)
  - Re-run processing to generate new `study_data_wna.csv`

### To modify cross-validation:
- Change year range in `for test_year in range(1980, 2021):` (line 58)
- Adjust test data filter condition if needed (line 61)

## External Data Locations

Climate data processing scripts reference these external paths:
- ERA5-Land NetCDF: `E:\ERA5-LAND\test_2\data_stream-moda.nc` ([data_processing.py:9](test/data_processing.py#L9))
- MERRA-2 data: Downloaded via [test/download_merra2.py](test/download_merra2.py) (requires authentication)

When working with these scripts, update paths to your local data location.

## Notes

- This is a research codebase focused on rapid experimentation
- Scripts are designed for sequential execution, not concurrent/production use
- No CI/CD or automated testing infrastructure
- Git repository is not initialized (consider `git init` if version control needed)
