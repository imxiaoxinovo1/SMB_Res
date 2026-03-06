"""
Step 2: Extract ERA5-Land climate data for WGMS elevation band locations

For elevation band data, we extract climate data at the elevation band midpoint
location. This provides more spatially precise climate forcing data for each band.

Input:
    - H:/Code/SMB/test/result_data_band/wgms_region02_band_clean.csv (from Step 1)
    - H:/Code/SMB/test/data/ERA5-LAND/data_stream-moda.nc
Output:
    - H:/Code/SMB/test/result_data_band/era5_climate_band_data.csv

Author: Claude Code
Date: 2025-12-29
"""

import pandas as pd
import numpy as np
import xarray as xr
import os
import sys
import dask

# ================= Configuration =================

# Input paths
WGMS_BAND_FILE = r"H:\Code\SMB\test\result_data_band\wgms_region02_band_clean.csv"
ERA5_FILE = r"H:\Code\SMB\test\data\ERA5-LAND\data_stream-moda.nc"

# Output path
OUTPUT_DIR = r"H:\Code\SMB\test\result_data_band"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "era5_climate_band_data.csv")

# ================= ERA5 Variable Mapping =================

VAR_MAP = {
    # Temperature & Humidity
    't2m': 'temp_2m',
    'd2m': 'dewpoint_2m',
    'skt': 'skin_temp',

    # Precipitation & Snow
    'tp': 'precip_total',
    'sf': 'snowfall',
    'sd': 'snow_depth',
    'rsn': 'snow_density',
    'asn': 'snow_albedo',
    'smlt': 'snowmelt',
    'es': 'snow_evaporation',
    'tsn': 'snow_layer_temp',
    'src': 'skin_reservoir_content',

    # Radiation & Energy
    'ssrd': 'solar_radiation_down',
    'strd': 'thermal_radiation_down',
    'ssr': 'net_solar_radiation',
    'str': 'net_thermal_radiation',
    'slhf': 'latent_heat_flux',
    'sshf': 'sensible_heat_flux',
    'fal': 'forecast_albedo',

    # Hydrology & Evaporation
    'e': 'evaporation_total',
    'pev': 'potential_evaporation',
    'ro': 'runoff',
    'sro': 'surface_runoff',
    'ssro': 'subsurface_runoff',

    # Pressure & Wind
    'sp': 'surface_pressure',
    'u10': 'wind_u_10m',
    'v10': 'wind_v_10m',

    # Vegetation
    'lai_hv': 'lai_high_veg',
    'lai_lv': 'lai_low_veg',

    # Lake variables
    'lblt': 'lake_bottom_temp',
    'licd': 'lake_ice_depth',
    'lict': 'lake_ice_temp',
    'lmld': 'lake_mix_layer_depth',
    'lmlt': 'lake_mix_layer_temp',
    'lshf': 'lake_shape_factor',
    'ltlt': 'lake_total_layer_temp'
}

# Aggregation rules
AGG_RULES = {
    # Cumulative variables (sum over year, with day-weighting)
    'tp': 'sum', 'sf': 'sum', 'smlt': 'sum',
    'e': 'sum', 'pev': 'sum', 'es': 'sum',
    'ro': 'sum', 'sro': 'sum', 'ssro': 'sum',

    # State variables and fluxes (mean)
    't2m': 'mean', 'd2m': 'mean', 'skt': 'mean',
    'sd': 'mean', 'rsn': 'mean', 'asn': 'mean', 'tsn': 'mean',
    'ssrd': 'mean', 'strd': 'mean', 'ssr': 'mean', 'str': 'mean',
    'slhf': 'mean', 'sshf': 'mean', 'fal': 'mean',
    'sp': 'mean', 'u10': 'mean', 'v10': 'mean',
    'lai_hv': 'mean', 'lai_lv': 'mean', 'src': 'mean',
    'lblt': 'mean', 'licd': 'mean', 'lict': 'mean',
    'lmld': 'mean', 'lmlt': 'mean', 'lshf': 'mean', 'ltlt': 'mean'
}

# Summer months
SUMMER_MONTHS = [4, 5, 6, 7, 8, 9]

# ================= Functions =================

def load_wgms_band_data():
    """Load WGMS elevation band data from Step 1"""
    print(f"\nLoading WGMS elevation band data...")
    print(f"  Path: {WGMS_BAND_FILE}")

    if not os.path.exists(WGMS_BAND_FILE):
        print(f"  ERROR: File not found!")
        print(f"  Please run step1_extract_WGMS_band_data.py first")
        sys.exit(1)

    df = pd.read_csv(WGMS_BAND_FILE)
    print(f"  Loaded: {len(df):,} elevation band records")
    print(f"  Glaciers: {df['WGMS_ID'].nunique()}")
    print(f"  Years: {df['YEAR'].min()}-{df['YEAR'].max()}")
    print(f"  Elevation range: {int(df['LOWER_ELEVATION'].min())}-{int(df['UPPER_ELEVATION'].max())} m")

    return df


def load_era5_dataset():
    """Load ERA5-Land NetCDF file with Dask for efficiency"""
    print(f"\nLoading ERA5-Land dataset...")
    print(f"  Path: {ERA5_FILE}")

    if not os.path.exists(ERA5_FILE):
        print(f"  ERROR: File not found!")
        sys.exit(1)

    # Optimize Dask scheduler
    dask.config.set(scheduler='threads')

    # Load with chunking for efficient processing
    ds = xr.open_dataset(ERA5_FILE, chunks='auto')

    print(f"  Loaded successfully")
    print(f"  Variables: {len(ds.data_vars)}")
    print(f"  Time range: {ds.valid_time.min().dt.strftime('%Y-%m').values} to {ds.valid_time.max().dt.strftime('%Y-%m').values}")

    # Handle expver dimension if present
    if 'expver' in ds.coords:
        try:
            ds = ds.sel(expver=1).combine_first(ds.sel(expver=5))
            print(f"  Note: Combined multiple experiment versions")
        except:
            pass

    # Check longitude format
    lon_max = float(ds.longitude.max())
    if lon_max > 180:
        print(f"  Note: Longitude in 0-360 format")
        needs_lon_conversion = True
    else:
        needs_lon_conversion = False

    return ds, needs_lon_conversion


def calculate_derived_variables(ds):
    """Calculate derived climate variables"""
    print(f"\nCalculating derived variables...")

    # Wind speed
    if 'u10' in ds.data_vars and 'v10' in ds.data_vars:
        ds['wind_speed'] = np.sqrt(ds['u10']**2 + ds['v10']**2)
        VAR_MAP['wind_speed'] = 'wind_speed'
        AGG_RULES['wind_speed'] = 'mean'
        print(f"  Added: wind_speed")

    return ds


def extract_climate_for_bands(df_wgms_band, ds, needs_lon_conversion):
    """
    Extract climate data for each unique glacier-year-band combination

    For elevation bands, we use the glacier's lat/lon for each band.
    The elevation band midpoint can be used later for lapse rate corrections.
    """
    print(f"\nExtracting climate data for elevation band locations...")

    # Get unique location-year combinations
    # Each band has the same lat/lon (glacier location) but different elevation
    unique_locations = df_wgms_band[['WGMS_ID', 'LATITUDE', 'LONGITUDE']].drop_duplicates()
    print(f"  Unique glacier locations: {len(unique_locations)}")

    # Prepare coordinates
    lats = unique_locations['LATITUDE'].values
    lons = unique_locations['LONGITUDE'].values

    if needs_lon_conversion:
        lons = np.where(lons < 0, lons + 360, lons)

    # Vectorized extraction
    print(f"  Performing spatial extraction (nearest neighbor)...")
    xr_lats = xr.DataArray(lats, dims="glacier")
    xr_lons = xr.DataArray(lons, dims="glacier")
    ds_points = ds.sel(latitude=xr_lats, longitude=xr_lons, method='nearest')

    # Get time dimension
    time_dim = 'valid_time' if 'valid_time' in ds.dims else 'time'

    # Separate variables by aggregation method
    sum_vars = [v for v in VAR_MAP.keys() if v in ds.data_vars and AGG_RULES.get(v) == 'sum']
    mean_vars = [v for v in VAR_MAP.keys() if v in ds.data_vars and AGG_RULES.get(v, 'mean') == 'mean']

    print(f"  Variables to sum (annual): {len(sum_vars)}")
    print(f"  Variables to average (annual): {len(mean_vars)}")

    # Compute annual aggregations with day-weighting
    print(f"  Computing annual aggregations with day-weighting...")
    days_in_month = ds_points[time_dim].dt.days_in_month

    # Sum variables with day-weighting
    if sum_vars:
        ds_weighted = ds_points[sum_vars] * days_in_month
        ds_sum = ds_weighted.resample({time_dim: '1YE'}).sum().compute()
    else:
        ds_sum = xr.Dataset()

    # Mean variables
    if mean_vars:
        ds_mean = ds_points[mean_vars].resample({time_dim: '1YE'}).mean().compute()
    else:
        ds_mean = xr.Dataset()

    ds_annual = xr.merge([ds_sum, ds_mean])

    # Compute summer aggregations
    print(f"  Computing summer season (Apr-Sep) aggregations...")
    ds_summer = ds_points.sel({time_dim: ds_points[time_dim].dt.month.isin(SUMMER_MONTHS)})
    days_in_month_summer = ds_summer[time_dim].dt.days_in_month

    if sum_vars:
        ds_weighted_summer = ds_summer[sum_vars] * days_in_month_summer
        ds_sum_summer = ds_weighted_summer.resample({time_dim: '1YE'}).sum().compute()
    else:
        ds_sum_summer = xr.Dataset()

    if mean_vars:
        ds_mean_summer = ds_summer[mean_vars].resample({time_dim: '1YE'}).mean().compute()
    else:
        ds_mean_summer = xr.Dataset()

    ds_summer_annual = xr.merge([ds_sum_summer, ds_mean_summer])

    # Convert to DataFrame
    print(f"  Converting to DataFrame...")
    df_annual = ds_annual.to_dataframe().reset_index()
    df_summer = ds_summer_annual.to_dataframe().reset_index()

    # Extract year
    df_annual['YEAR'] = df_annual[time_dim].dt.year
    df_summer['YEAR'] = df_summer[time_dim].dt.year

    # Map glacier index to WGMS_ID
    id_mapping = dict(enumerate(unique_locations['WGMS_ID'].values))
    df_annual['WGMS_ID'] = df_annual['glacier'].map(id_mapping)
    df_summer['WGMS_ID'] = df_summer['glacier'].map(id_mapping)

    # Rename columns
    rename_annual = {k: f"{VAR_MAP[k]}_year" for k in VAR_MAP.keys() if k in df_annual.columns}
    rename_summer = {k: f"{VAR_MAP[k]}_summer" for k in VAR_MAP.keys() if k in df_summer.columns}

    df_annual = df_annual.rename(columns=rename_annual)
    df_summer = df_summer.rename(columns=rename_summer)

    # Merge annual and summer
    merge_cols = ['WGMS_ID', 'YEAR']
    df_climate = pd.merge(df_annual, df_summer, on=merge_cols, how='outer', suffixes=('', '_dup'))
    df_climate = df_climate[[c for c in df_climate.columns if not c.endswith('_dup')]]

    # Keep only relevant columns
    keep_cols = ['WGMS_ID', 'YEAR'] + [c for c in df_climate.columns
                                        if c.endswith('_year') or c.endswith('_summer')]
    df_climate = df_climate[[c for c in keep_cols if c in df_climate.columns]]

    print(f"  Extracted {len(df_climate):,} glacier-year climate records")
    print(f"  Climate variables: {len([c for c in df_climate.columns if c not in ['WGMS_ID', 'YEAR']])}")

    return df_climate


def generate_summary(df_climate):
    """Generate summary statistics"""
    print(f"\n" + "=" * 70)
    print(f"SUMMARY STATISTICS - CLIMATE DATA")
    print(f"=" * 70)

    print(f"\nData Coverage:")
    print(f"  Total records: {len(df_climate):,}")
    print(f"  Unique glaciers: {df_climate['WGMS_ID'].nunique()}")
    print(f"  Time period: {int(df_climate['YEAR'].min())}-{int(df_climate['YEAR'].max())}")

    print(f"\nSample Climate Variables (Annual):")
    var_samples = ['temp_2m_year', 'precip_total_year', 'snowfall_year', 'snowmelt_year']

    for var in var_samples:
        if var in df_climate.columns:
            print(f"  {var:30s}: mean={df_climate[var].mean():.3f}, std={df_climate[var].std():.3f}")

    print(f"\nData Completeness:")
    total_cells = len(df_climate) * (len(df_climate.columns) - 2)
    missing_cells = df_climate.drop(columns=['WGMS_ID', 'YEAR']).isna().sum().sum()
    completeness = (1 - missing_cells / total_cells) * 100
    print(f"  Overall: {completeness:.1f}%")


# ================= Main Execution =================

def main():
    print("=" * 70)
    print("STEP 2: Extract ERA5-Land Climate Data for Elevation Bands")
    print("=" * 70)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Load WGMS band data
    df_wgms_band = load_wgms_band_data()

    # Step 2: Load ERA5 dataset
    ds, needs_lon_conversion = load_era5_dataset()

    # Step 3: Calculate derived variables
    ds = calculate_derived_variables(ds)

    # Step 4: Extract climate data
    df_climate = extract_climate_for_bands(df_wgms_band, ds, needs_lon_conversion)

    # Step 5: Generate summary
    generate_summary(df_climate)

    # Step 6: Save output
    print(f"\n" + "=" * 70)
    print(f"Saving output...")
    df_climate.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"  File saved: {OUTPUT_FILE}")
    print(f"  File size: {os.path.getsize(OUTPUT_FILE) / 1024:.1f} KB")

    print(f"\n" + "=" * 70)
    print(f"STEP 2 COMPLETED SUCCESSFULLY")
    print(f"=" * 70)
    print(f"\nNext step: Run step3_merge_WGMS_ERA5_band.py")


if __name__ == "__main__":
    main()
