# WGMS + ERA5-Land Data Processing Pipeline

Complete data preparation pipeline for glacier surface mass balance modeling using WGMS observations and ERA5-Land climate reanalysis.

## 📁 Directory Structure

```
H:/Code/SMB/test/
├── data/                      # Raw input data (not tracked in git)
│   ├── WGMS/                 # WGMS FoG database
│   │   └── FoG_DataBase/DOI-WGMS-FoG-2025-02b/data/
│   │       ├── glacier.csv
│   │       ├── mass_balance.csv
│   │       └── ...
│   ├── ERA5-LAND/            # ERA5-Land NetCDF files
│   │   └── data_stream-moda.nc
│   └── RGI/                  # RGI glacier inventory (for future use)
│
├── newcode/                   # Processing scripts (this directory)
│   ├── step1_extract_WGMS_data.py
│   ├── step2_extract_ERA5_data.py
│   ├── step3_merge_WGMS_ERA5.py
│   ├── run_all_steps.py
│   └── README.md (this file)
│
├── result_data/              # Processed output data
│   ├── wgms_region02_clean.csv
│   ├── era5_climate_data.csv
│   ├── wgms_era5_merged_final.csv
│   └── data_quality_report.txt
│
└── thesis/                   # Papers and documentation
```

## 🚀 Quick Start

### Option 1: Run Complete Pipeline

```bash
cd H:/Code/SMB/test/newcode
python run_all_steps.py
```

This will automatically run all 3 steps in sequence.

### Option 2: Run Steps Individually

```bash
# Step 1: Extract WGMS mass balance data
python step1_extract_WGMS_data.py

# Step 2: Extract ERA5-Land climate data
python step2_extract_ERA5_data.py

# Step 3: Merge both datasets
python step3_merge_WGMS_ERA5.py
```

## 📋 Pipeline Overview

### Step 1: Extract WGMS Data
**Script:** `step1_extract_WGMS_data.py`

**What it does:**
- Loads WGMS FoG database (glacier.csv + mass_balance.csv)
- Filters for Region 02 (Western Canada & USA)
- Merges glacier metadata with mass balance observations
- Cleans and standardizes column names
- Removes records with missing critical data

**Input:**
- `H:/Code/SMB/test/data/WGMS/FoG_DataBase/DOI-WGMS-FoG-2025-02b/data/glacier.csv`
- `H:/Code/SMB/test/data/WGMS/FoG_DataBase/DOI-WGMS-FoG-2025-02b/data/mass_balance.csv`

**Output:**
- `H:/Code/SMB/test/result_data/wgms_region02_clean.csv`

**Key columns:**
- WGMS_ID, NAME, YEAR, LATITUDE, LONGITUDE
- ANNUAL_BALANCE, WINTER_BALANCE, SUMMER_BALANCE
- ELA, AAR, AREA
- GLIMS_ID, RGI60_ID, RGI70_ID

---

### Step 2: Extract ERA5-Land Climate Data
**Script:** `step2_extract_ERA5_data.py`

**What it does:**
- Loads WGMS glacier locations from Step 1
- Extracts ERA5-Land variables for each glacier point (nearest neighbor)
- Resamples monthly data to annual and summer (Apr-Sep) aggregations
- Handles ~70 climate variables (temperature, precipitation, snow, radiation, etc.)
- Uses vectorized operations for fast processing

**Input:**
- `H:/Code/SMB/test/result_data/wgms_region02_clean.csv` (from Step 1)
- `H:/Code/SMB/test/data/ERA5-LAND/data_stream-moda.nc`

**Output:**
- `H:/Code/SMB/test/result_data/era5_climate_data.csv`

**Climate variables extracted:**
- **Temperature**: 2m air temp, dewpoint, skin temp
- **Precipitation**: total precip, snowfall, snow depth
- **Snow**: albedo, density, snowmelt, snow layer temp
- **Radiation**: solar down/up, thermal down/up, net radiation
- **Energy**: latent heat, sensible heat flux
- **Hydrology**: runoff, evaporation, subsurface flow
- **Atmosphere**: surface pressure, wind speed (u, v components)
- **Vegetation**: LAI (high/low vegetation)
- **Lake**: temperature, ice depth (if applicable)

**Temporal aggregations:**
- `*_year`: Annual total (for fluxes) or mean (for state variables)
- `*_summer`: Summer season (Apr-Sep) values

---

### Step 3: Merge Datasets
**Script:** `step3_merge_WGMS_ERA5.py`

**What it does:**
- Merges WGMS mass balance data with ERA5 climate data
- Matches records by WGMS_ID and YEAR
- Adds derived features (e.g., temperature in Celsius)
- Performs quality control checks
- Organizes columns in logical order
- Generates comprehensive data quality report

**Input:**
- `H:/Code/SMB/test/result_data/wgms_region02_clean.csv` (from Step 1)
- `H:/Code/SMB/test/result_data/era5_climate_data.csv` (from Step 2)

**Output:**
- `H:/Code/SMB/test/result_data/wgms_era5_merged_final.csv` (main dataset)
- `H:/Code/SMB/test/result_data/data_quality_report.txt` (QC report)

**Quality checks:**
- Missing value detection
- Value range validation (lat/lon, mass balance)
- Duplicate record detection
- Data completeness by variable category

---

## 📊 Expected Output

### Final Dataset Characteristics

**Typical size** (Region 02):
- ~1,000-1,500 records
- ~30-40 unique glaciers
- Time period: 1980-2020
- ~150-200 total variables

**Column groups:**
1. **Metadata** (7 cols): WGMS_ID, NAME, GLIMS_ID, RGI IDs, etc.
2. **Geography** (4 cols): LATITUDE, LONGITUDE, COUNTRY, REGION
3. **Time** (5 cols): YEAR, BEGIN_DATE, END_DATE, etc.
4. **Mass Balance** (9 cols): Annual/Winter/Summer balance + uncertainties
5. **Glacier Properties** (3 cols): AREA, ELA, AAR
6. **Climate Annual** (~70 cols): *_year variables
7. **Climate Summer** (~70 cols): *_summer variables
8. **References** (4 cols): INVESTIGATORS, AGENCIES, etc.

### File Locations

All output files are saved to:
```
H:/Code/SMB/test/result_data/
```

- **wgms_region02_clean.csv** - WGMS data only (~100-200 KB)
- **era5_climate_data.csv** - Climate data only (~500 KB - 1 MB)
- **wgms_era5_merged_final.csv** - Final merged dataset (~1-2 MB)
- **data_quality_report.txt** - QC summary (~5 KB)

---

## 🔧 Requirements

### Python Packages

```bash
pip install pandas numpy xarray netCDF4 scipy
```

**Package versions (tested):**
- pandas >= 1.5.0
- numpy >= 1.23.0
- xarray >= 2022.12.0
- netCDF4 >= 1.6.0

### Data Requirements

**WGMS FoG Database:**
- Download from: https://wgms.ch/data_databaseversions
- Version: 2025-02b or later
- Extract to: `H:/Code/SMB/test/data/WGMS/`

**ERA5-Land:**
- Source: Copernicus Climate Data Store
- Variables: All standard ERA5-Land variables
- Temporal resolution: Monthly
- Spatial resolution: 0.1° × 0.1°
- Coverage: Region 02 (Western Canada & USA)
- File location: `H:/Code/SMB/test/data/ERA5-LAND/data_stream-moda.nc`

---

## 🔍 Troubleshooting

### Common Issues

**1. File not found errors**
```
ERROR: WGMS file not found
```
**Solution:** Check that data paths in scripts match your actual file locations. Update paths at the top of each script if needed.

**2. Memory errors with ERA5 data**
```
MemoryError: Unable to allocate array
```
**Solution:** The scripts use `chunks='auto'` for lazy loading. If still problematic, process fewer glaciers at a time or increase available RAM.

**3. Missing climate variables**
```
WARNING: Variable 'xyz' not found in NetCDF
```
**Solution:** ERA5-Land variable names may vary. Check your NetCDF file with:
```python
import xarray as xr
ds = xr.open_dataset('path/to/era5.nc')
print(ds.data_vars)
```

**4. Encoding errors (Windows)**
```
UnicodeEncodeError: 'gbk' codec can't encode...
```
**Solution:** Scripts use UTF-8 encoding. Set console encoding:
```bash
chcp 65001  # Windows Command Prompt
```
Or run from PowerShell with UTF-8 support.

**5. Coordinate mismatch**
```
KeyError: 'longitude'
```
**Solution:** ERA5-Land longitude may be in 0-360° format. Scripts auto-detect and convert.

---

## 📈 Data Quality Metrics

After Step 3, check `data_quality_report.txt` for:

- **Missing values**: Should be <5% for critical variables
- **Value ranges**: Lat/Lon should be valid, mass balance typically -5 to +2 m w.e.
- **Duplicates**: Should be 0
- **Completeness**:
  - Mass balance variables: >90%
  - Temperature variables: >95%
  - Precipitation variables: >95%
  - Snow variables: >80% (less available at low elevations)
  - Radiation variables: >90%

---

## 🔄 Workflow Integration

### Next Steps After Data Preparation

1. **Exploratory Data Analysis**
   - Plot mass balance time series
   - Correlation matrix of climate variables
   - Spatial distribution of glaciers

2. **Feature Engineering**
   - Select most important climate variables
   - Create interaction terms if needed
   - Normalize/standardize features

3. **Machine Learning Model Training**
   - Random Forest (recommended baseline)
   - Lasso/Ridge regression
   - XGBoost or other ensemble methods

4. **Model Validation**
   - Leave-one-year-out cross-validation
   - Spatial cross-validation
   - Temporal cross-validation

### Example ML Workflow

```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut

# Load merged data
df = pd.read_csv('H:/Code/SMB/test/result_data/wgms_era5_merged_final.csv')

# Select features (example)
features = [
    'temp_2m_year', 'temp_2m_summer',
    'precip_total_year', 'snowfall_year',
    'snowmelt_year', 'LATITUDE', 'LONGITUDE', 'AREA'
]

X = df[features]
y = df['ANNUAL_BALANCE']

# Train model
model = RandomForestRegressor(n_estimators=200, max_depth=100)
model.fit(X, y)

# Predict (example)
predictions = model.predict(X)
```

---

## 📝 Notes

### Design Principles

1. **Modularity**: Each step is independent and can be run separately
2. **Reproducibility**: Fixed paths, clear outputs, version control ready
3. **Transparency**: Extensive logging and quality checks
4. **Efficiency**: Vectorized operations, lazy loading for large files
5. **Documentation**: Comprehensive comments and docstrings

### File Naming Conventions

- **Input data**: Original names from data sources
- **Intermediate outputs**: Descriptive names with step number
- **Final output**: `wgms_era5_merged_final.csv` (consistent name for downstream use)

### Future Enhancements

- [ ] Add RGI glacier geometry data (elevation bands, slope, aspect)
- [ ] Include MERRA-2 variables for comparison
- [ ] Add glacier-wide vs point measurements distinction
- [ ] Implement spatial interpolation for missing climate data
- [ ] Add automatic data download from sources

---

## 📖 References

### Data Sources

**WGMS (World Glacier Monitoring Service):**
> WGMS (2025): Fluctuations of Glaciers (FoG) Database. World Glacier Monitoring Service, Zurich, Switzerland.
> DOI: https://doi.org/10.5904/wgms-fog-2025-02b

**ERA5-Land:**
> Muñoz Sabater, J. (2019): ERA5-Land monthly averaged data from 1950 to present. Copernicus Climate Change Service (C3S) Climate Data Store (CDS).
> DOI: 10.24381/cds.68d2bb30

**RGI (Randolph Glacier Inventory):**
> RGI Consortium (2023): Randolph Glacier Inventory - A Dataset of Global Glacier Outlines, Version 7.0. NSIDC.
> DOI: https://doi.org/10.5067/F6JMOVY5NAVZ

---

## 👤 Contact

For questions or issues with this pipeline:
- Create an issue in the project repository
- Contact: [Your contact information]

---

## 📄 License

This processing pipeline is provided as-is for research purposes. Please cite the original data sources (WGMS, ERA5-Land) in any publications.

---

**Last Updated:** 2025-12-29
**Pipeline Version:** 1.0
**Compatible with:** WGMS FoG 2025-02b, ERA5-Land monthly data
