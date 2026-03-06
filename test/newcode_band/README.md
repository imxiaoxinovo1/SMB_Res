# WGMS Elevation Band Data Processing Pipeline

## Overview

This pipeline processes **WGMS elevation band mass balance data** merged with **ERA5-Land climate reanalysis data** for glaciers in western Canada and USA (RGI Region 02).

### What is Elevation Band Data?

Unlike glacier-wide annual mass balance, elevation band data provides **mass balance measurements at different elevation zones** on each glacier. This allows for:

- Analysis of mass balance gradients with elevation
- More precise climate-topography relationships
- Better understanding of accumulation and ablation zones
- Enhanced spatial prediction capabilities

## Data Sources

### Input Data

1. **WGMS FoG Database (2025-02b)**
   - `glacier.csv` - Glacier metadata (locations, IDs, regions)
   - `mass_balance_band.csv` - Elevation band mass balance observations
   - Location: `H:/Code/SMB/test/data/WGMS/FoG_DataBase/DOI-WGMS-FoG-2025-02b/data/`

2. **ERA5-Land Climate Reanalysis**
   - Monthly data from 1950-2025
   - ~37 climate variables (temperature, precipitation, snow, radiation, etc.)
   - Location: `H:/Code/SMB/test/data/ERA5-LAND/data_stream-moda.nc`

### Output Data

All outputs are saved to: `H:/Code/SMB/test/result_data_band/`

1. `wgms_region02_band_clean.csv` - Cleaned elevation band data
2. `era5_climate_band_data.csv` - Extracted climate data
3. `wgms_era5_band_merged_final.csv` - **Final merged dataset** (ready for ML)
4. `data_quality_report_band.txt` - Quality control report

## Pipeline Architecture

### Three-Step Processing Workflow

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Extract WGMS Elevation Band Data                  │
│  ─────────────────────────────────────────────────────────  │
│  Input:  glacier.csv + mass_balance_band.csv               │
│  Action: Filter Region 02, merge metadata, clean data     │
│  Output: wgms_region02_band_clean.csv                     │
│          (~X elevation band records, Y glaciers)          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 2: Extract ERA5-Land Climate Data                    │
│  ─────────────────────────────────────────────────────────  │
│  Input:  wgms_region02_band_clean.csv                     │
│          ERA5-Land NetCDF (5GB)                           │
│  Action: Vectorized extraction for glacier locations      │
│          Annual + summer aggregations with day-weighting  │
│  Output: era5_climate_band_data.csv                       │
│          (~74 climate variables per glacier-year)         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 3: Merge and Quality Control                         │
│  ─────────────────────────────────────────────────────────  │
│  Input:  wgms_region02_band_clean.csv                     │
│          era5_climate_band_data.csv                       │
│  Action: Merge on WGMS_ID + YEAR                          │
│          Add derived features (normalized elevation)       │
│          Perform QC checks, organize columns              │
│  Output: wgms_era5_band_merged_final.csv (ready for ML)   │
│          data_quality_report_band.txt                     │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

```bash
# Required Python packages
pip install pandas numpy xarray netCDF4 h5netcdf dask
```

### Running the Pipeline

**Option 1: Run all steps at once**
```bash
python run_all_steps.py
```

**Option 2: Run individual steps**
```bash
python step1_extract_WGMS_band_data.py
python step2_extract_ERA5_band_data.py
python step3_merge_WGMS_ERA5_band.py
```

**Option 3: Use batch file (Windows)**
```cmd
run_step1.bat
run_step2.bat
run_step3.bat
```

### Execution Time

- **Step 1**: ~5-10 seconds (reading CSVs, filtering)
- **Step 2**: ~30-60 seconds (NetCDF processing with Dask)
- **Step 3**: ~5-10 seconds (merging, QC)
- **Total**: ~1-2 minutes

## Data Schema

### Final Dataset Structure

Each row represents **one elevation band** on a glacier in a specific year.

#### Key Columns

**Identifiers & Metadata:**
- `WGMS_ID` - Glacier ID in WGMS database
- `SHORT_NAME` - Glacier name
- `YEAR` - Observation year
- `COUNTRY` - Country code (CA, US)
- `REGION` - RGI region code

**Geography:**
- `LATITUDE` - Glacier latitude (degrees)
- `LONGITUDE` - Glacier longitude (degrees)

**Elevation Band Information:**
- `LOWER_ELEVATION` - Band lower boundary (m a.s.l.)
- `UPPER_ELEVATION` - Band upper boundary (m a.s.l.)
- `ELEVATION_MIDPOINT` - Band center elevation (m a.s.l.)
- `ELEVATION_RANGE` - Band width (meters)
- `ELEVATION_NORMALIZED` - Position in glacier (0=lowest, 1=highest)
- `BAND_AREA` - Area of this elevation band (km²)

**Mass Balance:**
- `ANNUAL_BALANCE` - Annual mass balance (m w.e.)
- `WINTER_BALANCE` - Winter mass balance (m w.e., if available)
- `SUMMER_BALANCE` - Summer mass balance (m w.e., if available)
- `*_UNC` - Uncertainty estimates

**Climate Variables (Annual):**
- `temp_2m_year` - Annual mean 2m air temperature (K)
- `temp_2m_celsius_year` - Annual mean 2m air temperature (°C)
- `precip_total_year` - Annual total precipitation (m)
- `snowfall_year` - Annual total snowfall (m w.e.)
- `snowmelt_year` - Annual total snowmelt (m w.e.)
- `snow_depth_year` - Annual mean snow depth (m)
- `solar_radiation_down_year` - Annual mean downward solar radiation (J/m²)
- ... (37 variables total)

**Climate Variables (Summer, Apr-Sep):**
- Same variables as annual, with `_summer` suffix
- Example: `temp_2m_summer`, `snowmelt_summer`

**Total:** ~130+ variables

## Key Features

### 1. Day-Weighted Climate Aggregations

ERA5-Land variables are in units of "per day" (e.g., m/day for precipitation). We correctly aggregate them:

```python
# For cumulative variables (precip, snowfall, etc.)
monthly_total = daily_rate * days_in_month
annual_total = sum(monthly_total)

# For state variables (temperature, snow depth, etc.)
annual_mean = mean(monthly_values)
```

### 2. Elevation-Specific Features

Each elevation band gets:
- Absolute elevation metrics (lower, upper, midpoint)
- Normalized position within glacier (0-1 scale)
- Glacier-wide elevation range context

This enables analysis of:
- Mass balance gradients
- ELA (Equilibrium Line Altitude) estimation
- Vertical climate lapse rates

### 3. Seasonal Climate Data

Both **annual** and **summer season** (Apr-Sep) climate data are provided:
- Summer climate is critical for ablation zone processes
- Annual climate captures full mass balance year
- Enables seasonal contrast analysis

## Differences from Annual Mass Balance Pipeline

| Aspect | Annual Data (`newcode`) | Elevation Band Data (`newcode_band`) |
|--------|------------------------|-------------------------------------|
| **Spatial Resolution** | Glacier-wide average | Multiple bands per glacier |
| **Records per Glacier-Year** | 1 | 5-15 (typical) |
| **Mass Balance Detail** | Single value | Vertical gradient |
| **Elevation Info** | Glacier mean elevation | Band-specific elevations |
| **Use Cases** | Regional trends | Gradient analysis, ELA |
| **ML Target** | Glacier-wide balance | Band-level balance |

## Usage Examples

### Example 1: Load Final Dataset

```python
import pandas as pd

# Load merged data
df = pd.read_csv('H:/Code/SMB/test/result_data_band/wgms_era5_band_merged_final.csv')

print(f"Total elevation bands: {len(df)}")
print(f"Unique glaciers: {df['WGMS_ID'].nunique()}")
print(f"Year range: {df['YEAR'].min()}-{df['YEAR'].max()}")

# How many bands per glacier-year?
bands_per_gy = df.groupby(['WGMS_ID', 'YEAR']).size()
print(f"Average bands per glacier-year: {bands_per_gy.mean():.1f}")
```

### Example 2: Analyze Mass Balance Gradient

```python
# Select one glacier-year
glacier_id = 205  # South Cascade Glacier
year = 2010

data = df[(df['WGMS_ID'] == glacier_id) & (df['YEAR'] == year)]

# Plot mass balance vs elevation
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(data['ELEVATION_MIDPOINT'], data['ANNUAL_BALANCE'])
plt.xlabel('Elevation (m a.s.l.)')
plt.ylabel('Annual Mass Balance (m w.e.)')
plt.title(f'Mass Balance Gradient - Glacier {glacier_id}, {year}')
plt.grid(True)
plt.show()
```

### Example 3: Prepare Features for ML

```python
# Select features for Random Forest model
feature_cols = [
    # Elevation features
    'ELEVATION_MIDPOINT',
    'ELEVATION_NORMALIZED',
    'ELEVATION_RANGE',

    # Climate - annual
    'temp_2m_celsius_year',
    'precip_total_year',
    'snowfall_year',
    'snowmelt_year',
    'solar_radiation_down_year',

    # Climate - summer
    'temp_2m_celsius_summer',
    'snowmelt_summer',

    # Geographic
    'LATITUDE',
    'LONGITUDE'
]

target = 'ANNUAL_BALANCE'

X = df[feature_cols]
y = df[target]

# Train-test split by glacier (spatial cross-validation)
from sklearn.model_selection import GroupShuffleSplit

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=df['WGMS_ID']))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
```

## Quality Control

The pipeline performs automatic QC checks:

1. **Missing Value Check**
   - Critical fields must be complete
   - Reports missing percentages for all variables

2. **Range Validation**
   - Latitude: -90 to 90°
   - Longitude: -180 to 180°
   - Elevation bands: upper > lower
   - Mass balance: flags extreme values

3. **Duplicate Detection**
   - Checks for duplicate (WGMS_ID, YEAR, elevation band) combinations

4. **Completeness by Category**
   - Mass balance: Annual, winter, summer
   - Climate: Temperature, precipitation, snow, radiation, hydrology
   - Reports % complete for each category

Review the QC report: `data_quality_report_band.txt`

## Troubleshooting

### Issue: Step 1 fails with "File not found"

**Solution:** Check that WGMS database files exist at:
```
H:/Code/SMB/test/data/WGMS/FoG_DataBase/DOI-WGMS-FoG-2025-02b/data/
  - glacier.csv
  - mass_balance_band.csv
```

### Issue: Step 2 fails with "NetCDF library not found"

**Solution:** Install NetCDF dependencies:
```bash
pip install netCDF4 h5netcdf
```

### Issue: Step 2 is very slow

**Causes:**
- ERA5-Land file is large (5GB)
- First run requires loading entire NetCDF

**Solutions:**
- Be patient (~30-60 seconds is normal)
- Ensure SSD storage if possible
- Close other memory-intensive applications

### Issue: Merge rate < 100% in Step 3

**Causes:**
- Some WGMS years outside ERA5 time range (pre-1950)
- Some glacier locations outside ERA5 spatial domain

**Solution:** This is expected. Check match rate in console output.

## File Paths Summary

| Purpose | Path |
|---------|------|
| **Code** | `H:/Code/SMB/test/newcode_band/` |
| **Input Data** | `H:/Code/SMB/test/data/` |
| **Output Results** | `H:/Code/SMB/test/result_data_band/` |

## Next Steps

After running this pipeline:

1. **Review QC Report**
   ```bash
   notepad H:/Code/SMB/test/result_data_band/data_quality_report_band.txt
   ```

2. **Exploratory Data Analysis**
   - Visualize mass balance gradients
   - Analyze climate-elevation relationships
   - Check data distributions

3. **Machine Learning Modeling**
   - Use elevation band features for prediction
   - Implement spatial cross-validation (leave-glacier-out)
   - Consider hierarchical models (glacier-level + band-level effects)

4. **Compare with Annual Data**
   - Run the annual pipeline (`newcode/`)
   - Compare model performance: band-level vs glacier-wide

## References

- WGMS (2025). Fluctuations of Glaciers Database. DOI-WGMS-FoG-2025-02b
- Muñoz Sabater, J. (2019). ERA5-Land monthly averaged data. Copernicus Climate Change Service (C3S) Climate Data Store (CDS)
- RGI Consortium (2017). Randolph Glacier Inventory 6.0/7.0

## Support

For issues or questions:
- Check console output for error messages
- Review QC report for data quality issues
- Verify input file paths in configuration sections
- Ensure all dependencies are installed

---

**Created:** 2025-12-29
**Author:** Claude Code
**Version:** 1.0
